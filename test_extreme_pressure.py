#!/usr/bin/env python3
"""
极端内存瓶颈测试：让压缩吞吐超过30%
=====================================

核心策略：
1. 长prompt（占用更多初始blocks）
2. 长输出（让请求长时间在decode阶段）
3. 大量请求（超过无压缩并发能力）

这样：
- 无压缩：只能同时decode N个请求，其他必须等待
- 有压缩：可以同时decode 4-5N个请求，大幅提升吞吐

"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys

import torch
import gc
import time
from typing import List


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def generate_prompts(num: int, target_tokens: int) -> List[str]:
    base = "USER: Please explain "
    topics = ["AI", "ML", "DL", "NLP", "CV", "RL", "robotics", "quantum", "blockchain", "cloud"]
    expansion = " in great detail covering history, applications, challenges, and future. "

    prompts = []
    repeat = target_tokens // 30
    for i in range(num):
        topic = topics[i % len(topics)]
        prompt = base + topic + expansion * repeat + " ASSISTANT:"
        prompts.append(prompt)
    return prompts


def test_extreme_memory_pressure(model_path: str, compressor_path: str):
    """极端内存瓶颈测试"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 极端内存瓶颈测试")
    print("=" * 70)

    # 配置：超长prompt + 超长输出 + 大量请求
    num_requests = 400
    prompt_tokens = 2000  # 非常长的prompt
    max_output = 128  # 长输出

    prompts = generate_prompts(num_requests, prompt_tokens)

    # =====================================================
    # 测试1：无压缩
    # =====================================================
    print("\n--- 无压缩 ---")
    clear_gpu()

    llm = LlavaLLM(
        model_path,
        enable_compression=False,
        enforce_eager=True,
        max_model_len=4096,
    )

    block_size = llm.scheduler.block_manager.block_size
    total_blocks = len(llm.scheduler.block_manager.blocks)
    sample_len = len(llm.tokenizer.encode(prompts[0]))
    blocks_per_prompt = (sample_len + block_size - 1) // block_size
    blocks_per_output = (max_output + block_size - 1) // block_size
    total_blocks_per_req = blocks_per_prompt + blocks_per_output
    max_concurrent = total_blocks // total_blocks_per_req

    print(f"配置:")
    print(f"  总blocks: {total_blocks}")
    print(f"  每个prompt: {sample_len} tokens = {blocks_per_prompt} blocks")
    print(f"  每个输出: {max_output} tokens = {blocks_per_output} blocks")
    print(f"  每个请求总计: {total_blocks_per_req} blocks")
    print(f"  理论最大并发(无压缩): {max_concurrent}")

    # 添加所有请求
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=max_output))

    # 运行
    total_output_tokens = 0
    start = time.time()
    completed = 0

    while not llm.is_finished():
        outputs, num_tokens = llm.step(apply_compression=False)
        if num_tokens < 0:
            total_output_tokens += (-num_tokens)
        completed += len(outputs)

        # 每5秒打印一次状态
        if int(time.time() - start) % 5 == 0 and int(time.time() - start) > 0:
            free = len(llm.scheduler.block_manager.free_block_ids)
            running = len(llm.scheduler.running)
            waiting = len(llm.scheduler.waiting)
            elapsed = time.time() - start
            throughput = total_output_tokens / elapsed if elapsed > 0 else 0
            print(f"  [{elapsed:.0f}s] running={running}, waiting={waiting}, "
                  f"completed={completed}, throughput={throughput:.0f} tok/s")

    no_compress_time = time.time() - start
    no_compress_throughput = total_output_tokens / no_compress_time

    print(f"\n无压缩结果:")
    print(f"  总输出tokens: {total_output_tokens}")
    print(f"  总时间: {no_compress_time:.2f}s")
    print(f"  输出吞吐: {no_compress_throughput:.1f} tok/s")

    del llm
    clear_gpu()

    # =====================================================
    # 测试2：有压缩
    # =====================================================
    print("\n--- 有压缩 ---")

    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        enable_compression=True,
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=4096,
    )

    blocks_per_compressed = (sample_len // 5 + block_size - 1) // block_size
    total_blocks_compressed = blocks_per_compressed + blocks_per_output
    max_concurrent_compress = total_blocks // total_blocks_compressed

    print(f"配置(压缩后):")
    print(f"  压缩后prompt blocks: {blocks_per_compressed}")
    print(f"  每个请求总计: {total_blocks_compressed} blocks")
    print(f"  理论最大并发(压缩后): {max_concurrent_compress}")
    print(f"  并发能力提升: {max_concurrent_compress / max_concurrent:.1f}x")

    # 添加所有请求
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=max_output))

    # 运行
    total_output_tokens = 0
    start = time.time()
    completed = 0

    while not llm.is_finished():
        outputs, num_tokens = llm.step(apply_compression=True)
        if num_tokens < 0:
            total_output_tokens += (-num_tokens)
        completed += len(outputs)

        # 每5秒打印一次状态
        elapsed = time.time() - start
        if int(elapsed) % 5 == 0 and int(elapsed) > 0:
            free = len(llm.scheduler.block_manager.free_block_ids)
            running = len(llm.scheduler.running)
            waiting = len(llm.scheduler.waiting)
            throughput = total_output_tokens / elapsed if elapsed > 0 else 0
            print(f"  [{elapsed:.0f}s] running={running}, waiting={waiting}, "
                  f"completed={completed}, throughput={throughput:.0f} tok/s")

    compress_time = time.time() - start
    compress_throughput = total_output_tokens / compress_time

    print(f"\n有压缩结果:")
    print(f"  总输出tokens: {total_output_tokens}")
    print(f"  总时间: {compress_time:.2f}s")
    print(f"  输出吞吐: {compress_throughput:.1f} tok/s")

    del llm
    clear_gpu()

    # =====================================================
    # 对比
    # =====================================================
    print("\n" + "=" * 70)
    print(" 最终对比")
    print("=" * 70)

    speedup = compress_throughput / no_compress_throughput
    improvement = (speedup - 1) * 100

    print(f"\n无压缩吞吐: {no_compress_throughput:.1f} tok/s")
    print(f"有压缩吞吐: {compress_throughput:.1f} tok/s")
    print(f"")
    print(f"🎯 吞吐提升: {improvement:.1f}%")
    print(f"")

    if improvement >= 30:
        print(f"✅ 成功！压缩带来 {improvement:.1f}% 吞吐提升 (超过30%目标)")
    elif improvement >= 0:
        print(f"⚠️ 吞吐提升 {improvement:.1f}% (未达到30%目标)")
    else:
        print(f"❌ 吞吐下降 {-improvement:.1f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data/huggingface/llava-1.5-7b-hf')
    parser.add_argument('--compressor', default=str(CKPT_DIR / "llava_mlp.pth"))
    args = parser.parse_args()

    print("#" * 70)
    print(" 极端内存瓶颈测试 - 目标: 压缩吞吐超过30%")
    print("#" * 70)

    test_extreme_memory_pressure(args.model, args.compressor)

    print("\n✓ 测试完成!")


if __name__ == '__main__':
    main()
