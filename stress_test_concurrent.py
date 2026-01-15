#!/usr/bin/env python3
"""
大规模压测：测试最大并发请求数
================================

测试目标：
1. 无压缩情况下，多少请求会OOM
2. 有压缩情况下，能支持多少请求
3. 验证压缩带来的内存节省和并发能力提升

"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys

import torch
import gc
import time
import argparse
from typing import List


def get_gpu_memory():
    """获取GPU内存使用情况"""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return allocated, reserved, total


def clear_gpu():
    """清理GPU"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def generate_long_prompts(num_prompts: int, target_tokens: int = 600) -> List[str]:
    """生成长prompts"""
    base = "USER: Please provide a comprehensive analysis of "
    topics = [
        "artificial intelligence and machine learning",
        "climate change and environmental sustainability",
        "quantum computing and its applications",
        "biotechnology and genetic engineering",
        "space exploration and colonization",
        "renewable energy technologies",
        "cybersecurity and data privacy",
        "autonomous vehicles and transportation",
        "virtual reality and augmented reality",
        "blockchain and cryptocurrency",
    ]

    # 扩展文本以达到目标token数
    expansion = " Include detailed history, current developments, future prospects, key challenges, major players, technological breakthroughs, societal impacts, and policy implications. "

    prompts = []
    for i in range(num_prompts):
        topic = topics[i % len(topics)]
        # 重复expansion以达到目标长度
        repeat_count = target_tokens // 30  # 大约每30字符20个token
        prompt = base + topic + ". " + expansion * repeat_count + " ASSISTANT:"
        prompts.append(prompt)

    return prompts


def test_max_concurrent_no_compression(model_path: str, prompt_tokens: int = 600):
    """测试无压缩情况下的最大并发数"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 测试1: 无压缩 - 最大并发请求数")
    print("=" * 70)

    clear_gpu()
    alloc, reserved, total = get_gpu_memory()
    print(f"GPU总内存: {total:.2f} GB")
    print(f"测试前已用: {alloc:.2f} GB")

    try:
        llm = LlavaLLM(
            model_path,
            enable_compression=False,
            enforce_eager=True,
            max_model_len=4096,
        )
    except Exception as e:
        print(f"模型加载失败: {e}")
        return 0, 0

    alloc_after_load, _, _ = get_gpu_memory()
    print(f"模型加载后: {alloc_after_load:.2f} GB")

    block_size = llm.scheduler.block_manager.block_size
    total_blocks = len(llm.scheduler.block_manager.blocks)
    initial_free = len(llm.scheduler.block_manager.free_block_ids)

    print(f"\n总blocks: {total_blocks}")
    print(f"Block size: {block_size}")
    print(f"初始空闲blocks: {initial_free}")

    # 生成测试prompts
    prompts = generate_long_prompts(100, prompt_tokens)  # 准备100个

    # 检查实际token长度
    sample_len = len(llm.tokenizer.encode(prompts[0]))
    blocks_per_prompt = (sample_len + block_size - 1) // block_size
    max_theoretical = initial_free // blocks_per_prompt

    print(f"\n每个prompt约 {sample_len} tokens")
    print(f"每个prompt需要 {blocks_per_prompt} blocks")
    print(f"理论最大并发: {max_theoretical}")

    # 逐步添加请求，直到OOM
    successful_requests = 0
    sampling_params = SamplingParams(max_tokens=64)

    print("\n开始添加请求...")

    try:
        for i in range(min(100, max_theoretical + 10)):
            free_blocks = len(llm.scheduler.block_manager.free_block_ids)

            if free_blocks < blocks_per_prompt:
                print(f"\n[{i}] 空闲blocks不足: {free_blocks} < {blocks_per_prompt}")
                break

            llm.add_request(prompts[i], sampling_params)
            successful_requests += 1

            if (i + 1) % 10 == 0:
                alloc, _, _ = get_gpu_memory()
                free = len(llm.scheduler.block_manager.free_block_ids)
                print(f"[{i+1}] 已添加 {i+1} 请求, GPU: {alloc:.2f}GB, 空闲blocks: {free}")

        # 尝试运行一步
        print("\n尝试运行prefill...")
        start = time.time()
        llm.step(apply_compression=False)
        elapsed = time.time() - start
        print(f"Prefill成功! 耗时: {elapsed:.2f}s")

        alloc_after, _, _ = get_gpu_memory()
        print(f"Prefill后GPU: {alloc_after:.2f} GB")

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ OOM! 在 {successful_requests} 个请求时")
        alloc, _, _ = get_gpu_memory()
        print(f"OOM时GPU: {alloc:.2f} GB")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    del llm
    clear_gpu()

    print(f"\n无压缩最大并发: {successful_requests}")
    return successful_requests, blocks_per_prompt


def test_max_concurrent_with_compression(
    model_path: str,
    compressor_path: str,
    prompt_tokens: int = 600,
    baseline_count: int = 0
):
    """测试有压缩情况下的最大并发数"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 测试2: 有压缩 - 最大并发请求数")
    print("=" * 70)

    clear_gpu()
    alloc, reserved, total = get_gpu_memory()
    print(f"测试前已用: {alloc:.2f} GB")

    try:
        llm = LlavaLLM(
            model_path,
            compressor_path=compressor_path,
            enable_compression=True,
            async_compression=False,
            compression_factor=5,
            enforce_eager=True,
            max_model_len=4096,
        )
    except Exception as e:
        print(f"模型加载失败: {e}")
        return 0

    alloc_after_load, _, _ = get_gpu_memory()
    print(f"模型加载后: {alloc_after_load:.2f} GB")

    block_size = llm.scheduler.block_manager.block_size
    total_blocks = len(llm.scheduler.block_manager.blocks)
    initial_free = len(llm.scheduler.block_manager.free_block_ids)

    print(f"\n总blocks: {total_blocks}")
    print(f"初始空闲blocks: {initial_free}")

    # 生成测试prompts
    prompts = generate_long_prompts(200, prompt_tokens)  # 准备更多

    sample_len = len(llm.tokenizer.encode(prompts[0]))
    blocks_per_prompt_original = (sample_len + block_size - 1) // block_size
    blocks_per_prompt_compressed = (sample_len // 5 + block_size - 1) // block_size  # 假设5x压缩

    print(f"\n每个prompt约 {sample_len} tokens")
    print(f"压缩前需要 {blocks_per_prompt_original} blocks")
    print(f"压缩后需要 {blocks_per_prompt_compressed} blocks")

    # 理论上压缩后能支持更多
    max_theoretical = initial_free // blocks_per_prompt_compressed
    print(f"理论最大并发(压缩后): {max_theoretical}")

    sampling_params = SamplingParams(max_tokens=64)
    successful_requests = 0
    total_blocks_released = 0

    print("\n开始逐个处理请求（每个请求完整运行后释放blocks）...")

    target = min(200, max_theoretical + 10, baseline_count * 5 if baseline_count > 0 else 100)

    try:
        for i in range(target):
            free_blocks = len(llm.scheduler.block_manager.free_block_ids)

            # 检查是否有足够的blocks
            if free_blocks < blocks_per_prompt_original:
                print(f"\n[{i}] 无法添加更多请求: 空闲blocks {free_blocks} < 需要 {blocks_per_prompt_original}")
                break

            # 添加并完成一个请求
            llm.add_request(prompts[i], sampling_params)

            # 运行直到完成
            while not llm.is_finished():
                llm.step(apply_compression=True)

            successful_requests += 1

            if (i + 1) % 20 == 0:
                alloc, _, _ = get_gpu_memory()
                free = len(llm.scheduler.block_manager.free_block_ids)
                print(f"[{i+1}] 已完成 {i+1} 请求, GPU: {alloc:.2f}GB, 空闲blocks: {free}")

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ OOM! 在 {successful_requests} 个请求时")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    del llm
    clear_gpu()

    print(f"\n有压缩完成请求数: {successful_requests}")
    return successful_requests


def test_concurrent_batch(
    model_path: str,
    compressor_path: str,
    prompt_tokens: int = 600
):
    """测试压缩情况下的真正并发批处理"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 测试3: 真正并发 - 多请求同时在running")
    print("=" * 70)

    clear_gpu()

    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        enable_compression=True,
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=4096,
    )

    block_size = llm.scheduler.block_manager.block_size
    initial_free = len(llm.scheduler.block_manager.free_block_ids)

    prompts = generate_long_prompts(100, prompt_tokens)
    sample_len = len(llm.tokenizer.encode(prompts[0]))
    blocks_per_prompt = (sample_len + block_size - 1) // block_size
    blocks_after_compress = (sample_len // 5 + block_size - 1) // block_size

    print(f"每个prompt: {sample_len} tokens")
    print(f"压缩前blocks: {blocks_per_prompt}, 压缩后: {blocks_after_compress}")
    print(f"初始空闲: {initial_free}")

    # 计算无压缩能支持多少
    max_no_compress = initial_free // blocks_per_prompt
    # 计算压缩后能支持多少
    max_with_compress = initial_free // blocks_after_compress

    print(f"\n无压缩理论最大并发: {max_no_compress}")
    print(f"压缩后理论最大并发: {max_with_compress}")
    print(f"提升倍数: {max_with_compress / max_no_compress:.1f}x")

    # 测试逐步添加
    sampling_params = SamplingParams(max_tokens=32)

    # 先添加到无压缩的极限
    print(f"\n添加 {max_no_compress} 个请求（无压缩极限）...")
    for i in range(max_no_compress):
        llm.add_request(prompts[i], sampling_params)

    print("运行prefill + 压缩...")
    step = 0
    while len(llm.scheduler.waiting) > 0 or step == 0:
        llm.step(apply_compression=True)
        step += 1
        free = len(llm.scheduler.block_manager.free_block_ids)
        running = len(llm.scheduler.running)
        waiting = len(llm.scheduler.waiting)
        print(f"  Step {step}: running={running}, waiting={waiting}, free_blocks={free}")

    # 压缩后释放了blocks，尝试添加更多
    free_after_compress = len(llm.scheduler.block_manager.free_block_ids)
    print(f"\n压缩后空闲blocks: {free_after_compress}")

    additional = free_after_compress // blocks_per_prompt
    if additional > 0:
        print(f"可以额外添加 {additional} 个请求!")
        for i in range(max_no_compress, max_no_compress + additional):
            if i < len(prompts):
                llm.add_request(prompts[i], sampling_params)

        print("运行额外请求...")
        while len(llm.scheduler.waiting) > 0:
            llm.step(apply_compression=True)
            free = len(llm.scheduler.block_manager.free_block_ids)
            running = len(llm.scheduler.running)
            print(f"  running={running}, free_blocks={free}")

    # 完成所有请求
    print("\n完成所有decode...")
    total_completed = 0
    while not llm.is_finished():
        outputs, _ = llm.step(apply_compression=True)
        total_completed += len(outputs)
        if total_completed % 10 == 0 and total_completed > 0:
            print(f"  已完成: {total_completed}")

    print(f"\n总共完成: {total_completed} 个请求")

    del llm
    clear_gpu()

    return max_no_compress, max_with_compress


def main():
    parser = argparse.ArgumentParser(description='大规模并发压测')
    parser.add_argument('--model', type=str,
                        default='/data/huggingface/llava-1.5-7b-hf',
                        help='模型路径')
    parser.add_argument('--compressor', type=str,
                        default=str(CKPT_DIR / "llava_mlp.pth"),
                        help='压缩器路径')
    parser.add_argument('--prompt_tokens', type=int, default=600,
                        help='每个prompt的目标token数')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'no_compress', 'compress', 'concurrent'],
                        help='测试类型')
    args = parser.parse_args()

    print("#" * 70)
    print(" nano-vllm + LLaVA 大规模并发压测")
    print("#" * 70)
    print(f"模型: {args.model}")
    print(f"压缩器: {args.compressor}")
    print(f"Prompt目标长度: {args.prompt_tokens} tokens")

    alloc, reserved, total = get_gpu_memory()
    print(f"GPU总内存: {total:.2f} GB")

    results = {}

    if args.test in ['all', 'no_compress']:
        max_no_compress, blocks_per = test_max_concurrent_no_compression(
            args.model, args.prompt_tokens
        )
        results['no_compress'] = max_no_compress

    if args.test in ['all', 'compress']:
        baseline = results.get('no_compress', 0)
        max_compress = test_max_concurrent_with_compression(
            args.model, args.compressor, args.prompt_tokens, baseline
        )
        results['compress'] = max_compress

    if args.test in ['all', 'concurrent']:
        test_concurrent_batch(args.model, args.compressor, args.prompt_tokens)

    # 总结
    print("\n" + "=" * 70)
    print(" 压测结果总结")
    print("=" * 70)

    if 'no_compress' in results:
        print(f"无压缩最大并发: {results['no_compress']}")
    if 'compress' in results:
        print(f"有压缩完成请求: {results['compress']}")

    if 'no_compress' in results and 'compress' in results:
        if results['no_compress'] > 0:
            ratio = results['compress'] / results['no_compress']
            print(f"\n压缩提升: {ratio:.1f}x")

    print("\n压测完成!")


if __name__ == '__main__':
    main()
