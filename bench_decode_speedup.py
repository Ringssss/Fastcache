#!/usr/bin/env python3
"""
Decode阶段加速测试：验证压缩在大batch decode下的吞吐提升
===========================================================

核心思路：
- 压缩的真正价值是减少decode阶段的attention计算量
- 如果KV-cache从2000→400 tokens，attention计算量减少5x
- 需要长输出让decode阶段占主导，才能体现压缩优势

测试配置：
- 长prompt（1000-1500 tokens）
- 长输出（512-1024 tokens）
- 大batch size（128/256）

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
    """生成指定长度的prompts"""
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


def test_decode_speedup(
    model_path: str,
    compressor_path: str,
    batch_size: int = 128,
    prompt_tokens: int = 1000,
    max_output: int = 512
):
    """
    测试decode阶段的吞吐提升

    关键：长输出让decode占主导，体现压缩对attention计算的减少
    """
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(f" Decode加速测试: BS={batch_size}, Prompt={prompt_tokens}, Output={max_output}")
    print("=" * 70)

    prompts = generate_prompts(batch_size, prompt_tokens)

    # =====================================================
    # 测试1：无压缩
    # =====================================================
    print("\n--- 无压缩基线 ---")
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

    print(f"配置:")
    print(f"  Prompt长度: {sample_len} tokens")
    print(f"  输出长度: {max_output} tokens")
    print(f"  Batch size: {batch_size}")
    print(f"  总blocks: {total_blocks}")

    # 添加请求
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=max_output))

    # 分阶段计时
    prefill_time = 0
    prefill_tokens = 0
    decode_time = 0
    decode_tokens = 0

    start = time.time()
    step_count = 0

    while not llm.is_finished():
        step_start = time.time()
        outputs, num_tokens = llm.step(apply_compression=False)
        step_time = time.time() - step_start

        if num_tokens > 0:  # prefill
            prefill_time += step_time
            prefill_tokens += num_tokens
        else:  # decode
            decode_time += step_time
            decode_tokens += (-num_tokens)

        step_count += 1

        # 进度报告
        if step_count % 100 == 0:
            elapsed = time.time() - start
            print(f"  Step {step_count}: elapsed={elapsed:.1f}s, "
                  f"prefill={prefill_tokens}, decode={decode_tokens}")

    no_compress_total_time = time.time() - start
    no_compress_prefill_throughput = prefill_tokens / prefill_time if prefill_time > 0 else 0
    no_compress_decode_throughput = decode_tokens / decode_time if decode_time > 0 else 0
    no_compress_total_throughput = (prefill_tokens + decode_tokens) / no_compress_total_time

    print(f"\n无压缩结果:")
    print(f"  Prefill: {prefill_tokens} tokens, {prefill_time:.2f}s, {no_compress_prefill_throughput:.0f} tok/s")
    print(f"  Decode:  {decode_tokens} tokens, {decode_time:.2f}s, {no_compress_decode_throughput:.0f} tok/s")
    print(f"  Total:   {no_compress_total_throughput:.0f} tok/s")
    print(f"  Decode占比: {decode_time/no_compress_total_time*100:.1f}%")

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

    compressed_prompt_len = sample_len // 5
    print(f"压缩后配置:")
    print(f"  压缩后Prompt长度: ~{compressed_prompt_len} tokens")
    print(f"  理论attention计算减少: {sample_len / compressed_prompt_len:.1f}x")

    # 添加请求
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=max_output))

    # 分阶段计时
    prefill_time = 0
    prefill_tokens = 0
    compress_overhead = 0
    decode_time = 0
    decode_tokens = 0

    start = time.time()
    step_count = 0

    while not llm.is_finished():
        step_start = time.time()
        outputs, num_tokens = llm.step(apply_compression=True)
        step_time = time.time() - step_start

        if num_tokens > 0:  # prefill (包含压缩时间)
            prefill_time += step_time
            prefill_tokens += num_tokens
        else:  # decode
            decode_time += step_time
            decode_tokens += (-num_tokens)

        step_count += 1

        if step_count % 100 == 0:
            elapsed = time.time() - start
            print(f"  Step {step_count}: elapsed={elapsed:.1f}s, "
                  f"prefill={prefill_tokens}, decode={decode_tokens}")

    compress_total_time = time.time() - start
    compress_prefill_throughput = prefill_tokens / prefill_time if prefill_time > 0 else 0
    compress_decode_throughput = decode_tokens / decode_time if decode_time > 0 else 0
    compress_total_throughput = (prefill_tokens + decode_tokens) / compress_total_time

    print(f"\n有压缩结果:")
    print(f"  Prefill: {prefill_tokens} tokens, {prefill_time:.2f}s, {compress_prefill_throughput:.0f} tok/s")
    print(f"  Decode:  {decode_tokens} tokens, {decode_time:.2f}s, {compress_decode_throughput:.0f} tok/s")
    print(f"  Total:   {compress_total_throughput:.0f} tok/s")
    print(f"  Decode占比: {decode_time/compress_total_time*100:.1f}%")

    del llm
    clear_gpu()

    # =====================================================
    # 对比分析
    # =====================================================
    print("\n" + "=" * 70)
    print(" 对比分析")
    print("=" * 70)

    prefill_speedup = compress_prefill_throughput / no_compress_prefill_throughput if no_compress_prefill_throughput > 0 else 0
    decode_speedup = compress_decode_throughput / no_compress_decode_throughput if no_compress_decode_throughput > 0 else 0
    total_speedup = compress_total_throughput / no_compress_total_throughput if no_compress_total_throughput > 0 else 0

    print(f"\nPrefill吞吐变化: {(prefill_speedup-1)*100:+.1f}%")
    print(f"Decode吞吐变化:  {(decode_speedup-1)*100:+.1f}%")
    print(f"")
    print(f"无压缩总吞吐: {no_compress_total_throughput:.0f} tok/s")
    print(f"有压缩总吞吐: {compress_total_throughput:.0f} tok/s")
    print(f"")

    improvement = (total_speedup - 1) * 100
    print(f"🎯 总吞吐提升: {improvement:+.1f}%")

    if improvement >= 30:
        print(f"✅ 成功！压缩带来 {improvement:.1f}% 吞吐提升 (超过30%目标)")
    elif improvement >= 0:
        print(f"⚠️ 吞吐提升 {improvement:.1f}% (未达到30%目标)")
    else:
        print(f"❌ 吞吐下降 {-improvement:.1f}%")

    return {
        'batch_size': batch_size,
        'prompt_tokens': prompt_tokens,
        'max_output': max_output,
        'no_compress_throughput': no_compress_total_throughput,
        'compress_throughput': compress_total_throughput,
        'improvement': improvement,
        'decode_speedup': (decode_speedup - 1) * 100
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data/huggingface/llava-1.5-7b-hf')
    parser.add_argument('--compressor', default=str(CKPT_DIR / "llava_mlp.pth"))
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--prompt', type=int, default=1000, help='Prompt tokens')
    parser.add_argument('--output', type=int, default=512, help='Max output tokens')
    args = parser.parse_args()

    print("#" * 70)
    print(" Decode阶段加速测试 - 目标: 压缩吞吐超过30%")
    print("#" * 70)

    # 测试不同配置
    configs = [
        {'batch_size': args.bs, 'prompt_tokens': args.prompt, 'max_output': args.output},
    ]

    results = []
    for config in configs:
        result = test_decode_speedup(
            args.model,
            args.compressor,
            **config
        )
        results.append(result)

    # 总结
    print("\n" + "#" * 70)
    print(" 测试总结")
    print("#" * 70)

    for r in results:
        print(f"\nBS={r['batch_size']}, Prompt={r['prompt_tokens']}, Output={r['max_output']}:")
        print(f"  总吞吐提升: {r['improvement']:+.1f}%")
        print(f"  Decode加速: {r['decode_speedup']:+.1f}%")


if __name__ == '__main__':
    main()
