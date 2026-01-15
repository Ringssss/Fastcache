#!/usr/bin/env python3
"""
大规模压测：对比压缩前后的吞吐量和内存效率
==========================================

测试方式：逐个完整处理请求，对比：
1. 无压缩的吞吐量和blocks使用
2. 有压缩的吞吐量和blocks使用
3. blocks释放效果

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
    expansion = " in great detail covering history, applications, challenges, and future prospects for humanity. "

    prompts = []
    repeat = target_tokens // 25
    for i in range(num):
        topic = topics[i % len(topics)]
        prompt = base + topic + expansion * repeat + " ASSISTANT:"
        prompts.append(prompt)
    return prompts


def run_throughput_test(
    model_path: str,
    compressor_path: str,
    enable_compression: bool,
    num_requests: int = 50,
    prompt_tokens: int = 600
):
    """运行吞吐量测试"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    mode = "有压缩" if enable_compression else "无压缩"
    print(f"\n{'='*60}")
    print(f" 测试: {mode}")
    print(f"{'='*60}")

    clear_gpu()

    if enable_compression:
        llm = LlavaLLM(
            model_path,
            compressor_path=compressor_path,
            enable_compression=True,
            async_compression=False,
            compression_factor=5,
            enforce_eager=True,
            max_model_len=4096,
        )
    else:
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
    print(f"GPU: {alloc:.2f}/{total:.2f} GB")
    print(f"总blocks: {total_blocks}, 初始空闲: {initial_free}")

    prompts = generate_prompts(num_requests, prompt_tokens)
    sample_len = len(llm.tokenizer.encode(prompts[0]))
    blocks_per = (sample_len + block_size - 1) // block_size
    blocks_after_compress = (sample_len // 5 + block_size - 1) // block_size

    print(f"每个prompt: {sample_len} tokens, 原始需要 {blocks_per} blocks")
    if enable_compression:
        print(f"压缩后需要: {blocks_after_compress} blocks")

    # 记录blocks使用峰值
    peak_blocks_used = 0
    total_output_tokens = 0
    blocks_released_total = 0

    start_time = time.time()

    for i, prompt in enumerate(prompts):
        free_before = len(llm.scheduler.block_manager.free_block_ids)

        llm.add_request(prompt, SamplingParams(max_tokens=64))

        # 完整运行这个请求
        output_tokens = 0
        while not llm.is_finished():
            outputs, num_tokens = llm.step(apply_compression=enable_compression)
            if num_tokens < 0:  # decode
                output_tokens += (-num_tokens)

        total_output_tokens += output_tokens

        free_after = len(llm.scheduler.block_manager.free_block_ids)

        # 记录峰值blocks使用
        blocks_used = initial_free - free_before
        if blocks_used > peak_blocks_used:
            peak_blocks_used = blocks_used

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            throughput = total_output_tokens / elapsed
            print(f"[{i+1}/{num_requests}] 吞吐: {throughput:.1f} tok/s, 空闲blocks: {free_after}")

    elapsed = time.time() - start_time
    final_free = len(llm.scheduler.block_manager.free_block_ids)

    # 统计
    throughput = total_output_tokens / elapsed

    print(f"\n结果:")
    print(f"  完成请求: {num_requests}")
    print(f"  总输出tokens: {total_output_tokens}")
    print(f"  总耗时: {elapsed:.2f}s")
    print(f"  吞吐量: {throughput:.1f} tok/s")
    print(f"  峰值blocks使用: {peak_blocks_used}")
    print(f"  最终空闲blocks: {final_free}")

    del llm
    clear_gpu()

    return {
        'mode': mode,
        'requests': num_requests,
        'tokens': total_output_tokens,
        'time': elapsed,
        'throughput': throughput,
        'peak_blocks': peak_blocks_used,
        'blocks_per_req': blocks_per if not enable_compression else blocks_after_compress,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data/huggingface/llava-1.5-7b-hf')
    parser.add_argument('--compressor', default=str(CKPT_DIR / "llava_mlp.pth"))
    parser.add_argument('--num_requests', type=int, default=50)
    parser.add_argument('--prompt_tokens', type=int, default=800)
    args = parser.parse_args()

    print("#" * 70)
    print(" nano-vllm + LLaVA 压缩效果压测")
    print("#" * 70)
    print(f"请求数: {args.num_requests}")
    print(f"Prompt长度: ~{args.prompt_tokens} tokens")

    alloc, total = get_gpu_memory()
    print(f"GPU: {total:.2f} GB")

    # 测试无压缩
    result_no_compress = run_throughput_test(
        args.model, args.compressor,
        enable_compression=False,
        num_requests=args.num_requests,
        prompt_tokens=args.prompt_tokens
    )

    # 测试有压缩
    result_compress = run_throughput_test(
        args.model, args.compressor,
        enable_compression=True,
        num_requests=args.num_requests,
        prompt_tokens=args.prompt_tokens
    )

    # 对比
    print("\n" + "=" * 70)
    print(" 对比总结")
    print("=" * 70)

    print(f"\n{'指标':<20} {'无压缩':<15} {'有压缩':<15} {'提升':<15}")
    print("-" * 65)

    # 吞吐量
    t1, t2 = result_no_compress['throughput'], result_compress['throughput']
    improvement = (t2 - t1) / t1 * 100
    print(f"{'吞吐量(tok/s)':<20} {t1:<15.1f} {t2:<15.1f} {improvement:+.1f}%")

    # Blocks使用
    b1, b2 = result_no_compress['blocks_per_req'], result_compress['blocks_per_req']
    saving = (b1 - b2) / b1 * 100
    print(f"{'Blocks/请求':<20} {b1:<15} {b2:<15} {saving:.1f}% 节省")

    # 理论并发提升
    # 假设总blocks = 458
    total_blocks = 458
    max_concurrent_no = total_blocks // b1
    max_concurrent_yes = total_blocks // b2
    concurrent_improvement = (max_concurrent_yes - max_concurrent_no) / max_concurrent_no * 100
    print(f"{'理论最大并发':<20} {max_concurrent_no:<15} {max_concurrent_yes:<15} {concurrent_improvement:+.1f}%")

    print("\n✓ 压测完成!")


if __name__ == '__main__':
    main()
