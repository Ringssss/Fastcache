#!/usr/bin/env python3
"""
内存瓶颈测试：验证压缩在高压力场景下的吞吐提升
==============================================

核心思路：
- 当内存不是瓶颈时，压缩只会带来开销
- 当内存是瓶颈时，压缩能显著提升吞吐

测试场景：
- 大量请求同时到达
- 长prompt（占用更多blocks）
- 无压缩情况下内存不足，请求被阻塞
- 有压缩情况下能处理更多并发

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


def generate_long_prompts(num: int, target_tokens: int = 1500) -> List[str]:
    """生成长prompts以创造内存压力"""
    base = "USER: Please provide a comprehensive analysis of "
    topics = ["AI", "ML", "DL", "NLP", "CV", "RL", "robotics", "quantum", "blockchain", "cloud"]
    expansion = " including detailed history, current developments, future prospects, key challenges, major players, technological breakthroughs, societal impacts, and policy implications. "

    prompts = []
    repeat = target_tokens // 30
    for i in range(num):
        topic = topics[i % len(topics)]
        prompt = base + topic + expansion * repeat + " ASSISTANT:"
        prompts.append(prompt)
    return prompts


def test_memory_pressure(model_path: str, compressor_path: str):
    """内存瓶颈测试"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 内存瓶颈场景测试")
    print("=" * 70)

    # 配置：长prompt + 大量请求
    num_requests = 300
    prompt_tokens = 1500  # 长prompt
    max_output = 64

    prompts = generate_long_prompts(num_requests, prompt_tokens)

    # =====================================================
    # 测试1：无压缩
    # =====================================================
    print("\n--- 无压缩 ---")
    clear_gpu()

    llm_no_compress = LlavaLLM(
        model_path,
        enable_compression=False,
        enforce_eager=True,
        max_model_len=4096,
    )

    block_size = llm_no_compress.scheduler.block_manager.block_size
    total_blocks = len(llm_no_compress.scheduler.block_manager.blocks)
    sample_len = len(llm_no_compress.tokenizer.encode(prompts[0]))
    blocks_per_prompt = (sample_len + block_size - 1) // block_size
    max_concurrent_no_compress = total_blocks // blocks_per_prompt

    print(f"总blocks: {total_blocks}")
    print(f"每个prompt: {sample_len} tokens = {blocks_per_prompt} blocks")
    print(f"理论最大并发(无压缩): {max_concurrent_no_compress}")

    # 添加所有请求
    for prompt in prompts:
        llm_no_compress.add_request(prompt, SamplingParams(max_tokens=max_output))

    # 运行
    total_output_tokens = 0
    start = time.time()
    step_count = 0
    completed = 0

    while not llm_no_compress.is_finished():
        outputs, num_tokens = llm_no_compress.step(apply_compression=False)
        if num_tokens < 0:
            total_output_tokens += (-num_tokens)
        step_count += 1
        completed += len(outputs)

        if step_count % 100 == 0:
            free = len(llm_no_compress.scheduler.block_manager.free_block_ids)
            running = len(llm_no_compress.scheduler.running)
            waiting = len(llm_no_compress.scheduler.waiting)
            elapsed = time.time() - start
            print(f"  Step {step_count}: running={running}, waiting={waiting}, "
                  f"free_blocks={free}, completed={completed}, elapsed={elapsed:.1f}s")

    no_compress_time = time.time() - start
    no_compress_throughput = total_output_tokens / no_compress_time

    print(f"\n无压缩结果:")
    print(f"  总输出tokens: {total_output_tokens}")
    print(f"  总时间: {no_compress_time:.2f}s")
    print(f"  输出吞吐: {no_compress_throughput:.1f} tok/s")

    del llm_no_compress
    clear_gpu()

    # =====================================================
    # 测试2：有压缩（即时压缩）
    # =====================================================
    print("\n--- 有压缩 (即时压缩) ---")

    llm_compress = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        enable_compression=True,
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=4096,
    )

    blocks_per_compressed = (sample_len // 5 + block_size - 1) // block_size
    max_concurrent_compress = total_blocks // blocks_per_compressed

    print(f"压缩后每个prompt: {blocks_per_compressed} blocks")
    print(f"理论最大并发(压缩后): {max_concurrent_compress}")
    print(f"并发能力提升: {max_concurrent_compress / max_concurrent_no_compress:.1f}x")

    # 添加所有请求
    for prompt in prompts:
        llm_compress.add_request(prompt, SamplingParams(max_tokens=max_output))

    # 运行
    total_output_tokens = 0
    start = time.time()
    step_count = 0
    completed = 0

    while not llm_compress.is_finished():
        outputs, num_tokens = llm_compress.step(apply_compression=True)
        if num_tokens < 0:
            total_output_tokens += (-num_tokens)
        step_count += 1
        completed += len(outputs)

        if step_count % 100 == 0:
            free = len(llm_compress.scheduler.block_manager.free_block_ids)
            running = len(llm_compress.scheduler.running)
            waiting = len(llm_compress.scheduler.waiting)
            elapsed = time.time() - start
            print(f"  Step {step_count}: running={running}, waiting={waiting}, "
                  f"free_blocks={free}, completed={completed}, elapsed={elapsed:.1f}s")

    compress_time = time.time() - start
    compress_throughput = total_output_tokens / compress_time

    print(f"\n有压缩结果:")
    print(f"  总输出tokens: {total_output_tokens}")
    print(f"  总时间: {compress_time:.2f}s")
    print(f"  输出吞吐: {compress_throughput:.1f} tok/s")

    del llm_compress
    clear_gpu()

    # =====================================================
    # 对比
    # =====================================================
    print("\n" + "=" * 70)
    print(" 对比结果")
    print("=" * 70)

    speedup = compress_throughput / no_compress_throughput
    improvement = (speedup - 1) * 100

    print(f"\n无压缩吞吐: {no_compress_throughput:.1f} tok/s")
    print(f"有压缩吞吐: {compress_throughput:.1f} tok/s")
    print(f"")
    print(f"吞吐提升: {improvement:.1f}%")
    print(f"")

    if improvement >= 30:
        print(f"✅ 压缩带来 {improvement:.1f}% 吞吐提升 (超过30%目标!)")
    else:
        print(f"⚠️ 吞吐提升 {improvement:.1f}% (未达到30%目标)")
        print(f"   原因可能: 内存压力不足或压缩开销过大")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data/huggingface/llava-1.5-7b-hf')
    parser.add_argument('--compressor', default=str(CKPT_DIR / "llava_mlp.pth"))
    args = parser.parse_args()

    print("#" * 70)
    print(" 内存瓶颈场景测试 - 验证压缩的吞吐提升")
    print("#" * 70)

    test_memory_pressure(args.model, args.compressor)

    print("\n✓ 测试完成!")


if __name__ == '__main__':
    main()
