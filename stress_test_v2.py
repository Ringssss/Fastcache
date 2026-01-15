#!/usr/bin/env python3
"""
大规模压测：测试最大并发请求数（简化版）
========================================

逐个处理请求，测试：
1. 无压缩情况下能处理多少请求后OOM
2. 有压缩情况下能处理多少请求

"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys

import torch
import gc
import time
from typing import List


def get_gpu_memory():
    """获取GPU内存"""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return allocated, total


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def generate_prompts(num: int, target_tokens: int = 600) -> List[str]:
    """生成测试prompts"""
    base = "USER: Please explain "
    topics = ["AI", "ML", "DL", "NLP", "CV", "RL", "robotics", "quantum", "blockchain", "cloud"]
    expansion = " in great detail covering history, applications, challenges, and future. "

    prompts = []
    repeat = target_tokens // 25
    for i in range(num):
        topic = topics[i % len(topics)]
        prompt = base + topic + expansion * repeat + " ASSISTANT:"
        prompts.append(prompt)
    return prompts


def test_no_compression(model_path: str, prompt_tokens: int = 600):
    """测试无压缩"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 测试: 无压缩 - 并发处理能力")
    print("=" * 70)

    clear_gpu()

    llm = LlavaLLM(
        model_path,
        enable_compression=False,
        enforce_eager=True,
        max_model_len=4096,
    )

    block_size = llm.scheduler.block_manager.block_size
    total_blocks = len(llm.scheduler.block_manager.blocks)
    initial_free = len(llm.scheduler.block_manager.free_block_ids)

    alloc, total = get_gpu_memory()
    print(f"\nGPU: {alloc:.2f}/{total:.2f} GB")
    print(f"总blocks: {total_blocks}, 初始空闲: {initial_free}")

    prompts = generate_prompts(200, prompt_tokens)
    sample_len = len(llm.tokenizer.encode(prompts[0]))
    blocks_per = (sample_len + block_size - 1) // block_size

    print(f"每个prompt: {sample_len} tokens, 需要 {blocks_per} blocks")

    # 逐个完整运行请求
    completed = 0
    start_time = time.time()

    try:
        for i, prompt in enumerate(prompts):
            free = len(llm.scheduler.block_manager.free_block_ids)
            if free < blocks_per:
                print(f"\n空闲blocks不足: {free} < {blocks_per}")
                break

            llm.add_request(prompt, SamplingParams(max_tokens=32))

            # 完整运行这个请求
            while not llm.is_finished():
                llm.step(apply_compression=False)

            completed += 1

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                alloc, _ = get_gpu_memory()
                free = len(llm.scheduler.block_manager.free_block_ids)
                throughput = completed / elapsed
                print(f"[{completed}] GPU: {alloc:.2f}GB, 空闲blocks: {free}, 吞吐: {throughput:.1f} req/s")

    except torch.cuda.OutOfMemoryError:
        print(f"\n❌ OOM at request {completed}")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    elapsed = time.time() - start_time
    print(f"\n无压缩完成: {completed} 请求, 耗时: {elapsed:.1f}s")
    print(f"平均吞吐: {completed/elapsed:.2f} req/s")

    del llm
    clear_gpu()

    return completed, blocks_per


def test_with_compression(model_path: str, compressor_path: str, prompt_tokens: int = 600):
    """测试有压缩"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 测试: 有压缩 - 并发处理能力")
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
    total_blocks = len(llm.scheduler.block_manager.blocks)
    initial_free = len(llm.scheduler.block_manager.free_block_ids)

    alloc, total = get_gpu_memory()
    print(f"\nGPU: {alloc:.2f}/{total:.2f} GB")
    print(f"总blocks: {total_blocks}, 初始空闲: {initial_free}")

    prompts = generate_prompts(500, prompt_tokens)  # 准备更多
    sample_len = len(llm.tokenizer.encode(prompts[0]))
    blocks_per = (sample_len + block_size - 1) // block_size
    blocks_after_compress = (sample_len // 5 + block_size - 1) // block_size

    print(f"每个prompt: {sample_len} tokens")
    print(f"压缩前blocks: {blocks_per}, 压缩后: {blocks_after_compress}")

    completed = 0
    start_time = time.time()
    total_blocks_released = 0

    try:
        for i, prompt in enumerate(prompts):
            free = len(llm.scheduler.block_manager.free_block_ids)
            if free < blocks_per:
                print(f"\n空闲blocks不足: {free} < {blocks_per}")
                break

            llm.add_request(prompt, SamplingParams(max_tokens=32))

            # 完整运行这个请求（包括压缩）
            while not llm.is_finished():
                llm.step(apply_compression=True)

            completed += 1

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                alloc, _ = get_gpu_memory()
                free = len(llm.scheduler.block_manager.free_block_ids)
                throughput = completed / elapsed
                print(f"[{completed}] GPU: {alloc:.2f}GB, 空闲blocks: {free}, 吞吐: {throughput:.1f} req/s")

    except torch.cuda.OutOfMemoryError:
        print(f"\n❌ OOM at request {completed}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n有压缩完成: {completed} 请求, 耗时: {elapsed:.1f}s")
    print(f"平均吞吐: {completed/elapsed:.2f} req/s")

    del llm
    clear_gpu()

    return completed


def test_concurrent_capacity(model_path: str, compressor_path: str, prompt_tokens: int = 600):
    """测试真正的并发容量差异"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 测试: 并发容量对比（同时in-flight的请求数）")
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

    prompts = generate_prompts(500, prompt_tokens)  # 增加到500个
    sample_len = len(llm.tokenizer.encode(prompts[0]))
    blocks_per_original = (sample_len + block_size - 1) // block_size
    blocks_per_compressed = (sample_len // 5 + block_size - 1) // block_size

    # 计算理论并发能力
    max_concurrent_no_compress = initial_free // blocks_per_original
    max_concurrent_with_compress = initial_free // blocks_per_compressed

    print(f"\n每个prompt: {sample_len} tokens")
    print(f"压缩前blocks/请求: {blocks_per_original}")
    print(f"压缩后blocks/请求: {blocks_per_compressed}")
    print(f"\n初始空闲blocks: {initial_free}")
    print(f"无压缩理论最大并发: {max_concurrent_no_compress}")
    print(f"有压缩理论最大并发: {max_concurrent_with_compress}")
    print(f"提升倍数: {max_concurrent_with_compress / max_concurrent_no_compress:.1f}x")

    # 实际测试: 添加到无压缩极限，然后运行一轮prefill+压缩，看能否继续添加
    test_count = min(max_concurrent_no_compress, len(prompts))
    print(f"\n测试: 添加 {test_count} 个请求（无压缩极限）...")

    for i in range(test_count):
        llm.add_request(prompts[i], SamplingParams(max_tokens=32))

    free_after_add = len(llm.scheduler.block_manager.free_block_ids)
    waiting = len(llm.scheduler.waiting)
    print(f"添加后: waiting={waiting}, 空闲blocks={free_after_add}")

    # 运行prefill（会分配blocks）并压缩（会释放blocks）
    print("\n运行prefill + 压缩...")
    prefill_count = 0
    while len(llm.scheduler.waiting) > 0:
        llm.step(apply_compression=True)
        prefill_count += 1
        free = len(llm.scheduler.block_manager.free_block_ids)
        running = len(llm.scheduler.running)
        waiting = len(llm.scheduler.waiting)
        print(f"  Step {prefill_count}: running={running}, waiting={waiting}, free_blocks={free}")

    free_after_compress = len(llm.scheduler.block_manager.free_block_ids)
    print(f"\n压缩后空闲blocks: {free_after_compress}")

    # 看能否添加更多请求
    additional_possible = free_after_compress // blocks_per_original
    print(f"可额外添加请求数: {additional_possible}")

    if additional_possible > 0:
        print(f"\n✓ 压缩释放了足够的blocks，可以添加 {additional_possible} 个额外请求!")
        total_concurrent = max_concurrent_no_compress + additional_possible
        print(f"实际并发能力: {total_concurrent} (vs 无压缩 {max_concurrent_no_compress})")
        print(f"提升: {total_concurrent / max_concurrent_no_compress:.1f}x")

    del llm
    clear_gpu()

    return max_concurrent_no_compress, max_concurrent_with_compress


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data/huggingface/llava-1.5-7b-hf')
    parser.add_argument('--compressor', default=str(CKPT_DIR / "llava_mlp.pth"))
    parser.add_argument('--prompt_tokens', type=int, default=600)
    parser.add_argument('--test', choices=['all', 'no_compress', 'compress', 'capacity'], default='all')
    args = parser.parse_args()

    print("#" * 70)
    print(" nano-vllm + LLaVA 大规模并发压测")
    print("#" * 70)

    alloc, total = get_gpu_memory()
    print(f"GPU: {total:.2f} GB")

    results = {}

    if args.test in ['all', 'no_compress']:
        no_compress_count, blocks_per = test_no_compression(args.model, args.prompt_tokens)
        results['no_compress'] = no_compress_count

    if args.test in ['all', 'compress']:
        compress_count = test_with_compression(args.model, args.compressor, args.prompt_tokens)
        results['compress'] = compress_count

    if args.test in ['all', 'capacity']:
        test_concurrent_capacity(args.model, args.compressor, args.prompt_tokens)

    # 总结
    print("\n" + "=" * 70)
    print(" 总结")
    print("=" * 70)
    for k, v in results.items():
        print(f"{k}: {v} 请求")


if __name__ == '__main__':
    main()
