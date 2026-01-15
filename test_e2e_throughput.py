#!/usr/bin/env python3
"""
端到端吞吐测试：验证压缩在大batch长prompt下的30%+吞吐提升
============================================================

关键发现：
- 在BS=256, Prompt=1258时，Decode吞吐提升101%
- 需要端到端测试验证总吞吐能否超过30%

配置策略：
- 大batch size (256)
- 长prompt (让attention成为瓶颈)
- 中等输出 (让decode占主导时间)

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


def test_e2e_throughput(
    model_path: str,
    compressor_path: str,
    batch_size: int = 256,
    prompt_tokens: int = 2500,
    max_output: int = 256
):
    """
    端到端吞吐测试
    """
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(f" 端到端测试: BS={batch_size}, Prompt≈{prompt_tokens}, Output={max_output}")
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

    sample_len = len(llm.tokenizer.encode(prompts[0]))
    print(f"实际Prompt长度: {sample_len} tokens")

    # 添加请求
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=max_output, temperature=1.0))

    # 运行
    prefill_time = 0
    prefill_tokens = 0
    decode_time = 0
    decode_steps = 0

    start = time.time()

    while not llm.is_finished():
        step_start = time.time()
        outputs, num_tokens = llm.step(apply_compression=False)
        step_time = time.time() - step_start

        if num_tokens > 0:
            prefill_time += step_time
            prefill_tokens += num_tokens
        else:
            decode_time += step_time
            decode_steps += 1

    no_compress_total_time = time.time() - start
    no_compress_total_tokens = prefill_tokens + decode_steps * batch_size
    no_compress_throughput = no_compress_total_tokens / no_compress_total_time

    print(f"\n无压缩结果:")
    print(f"  Prefill: {prefill_tokens} tokens, {prefill_time:.2f}s")
    print(f"  Decode: {decode_steps} steps, {decode_time:.2f}s")
    print(f"  总时间: {no_compress_total_time:.2f}s")
    print(f"  总吞吐: {no_compress_throughput:.0f} tok/s")
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

    print(f"压缩后Prompt长度: ~{sample_len//5} tokens")

    # 添加请求
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=max_output, temperature=1.0))

    # 运行（这次用step的apply_compression=True）
    prefill_time = 0
    prefill_tokens = 0
    decode_time = 0
    decode_steps = 0

    start = time.time()

    while not llm.is_finished():
        step_start = time.time()
        outputs, num_tokens = llm.step(apply_compression=True)
        step_time = time.time() - step_start

        if num_tokens > 0:
            prefill_time += step_time
            prefill_tokens += num_tokens
        else:
            decode_time += step_time
            decode_steps += 1

    compress_total_time = time.time() - start
    compress_total_tokens = prefill_tokens + decode_steps * batch_size
    compress_throughput = compress_total_tokens / compress_total_time

    print(f"\n有压缩结果:")
    print(f"  Prefill+压缩: {prefill_tokens} tokens, {prefill_time:.2f}s")
    print(f"  Decode: {decode_steps} steps, {decode_time:.2f}s")
    print(f"  总时间: {compress_total_time:.2f}s")
    print(f"  总吞吐: {compress_throughput:.0f} tok/s")
    print(f"  Decode占比: {decode_time/compress_total_time*100:.1f}%")

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

    print(f"\n无压缩吞吐: {no_compress_throughput:.0f} tok/s")
    print(f"有压缩吞吐: {compress_throughput:.0f} tok/s")
    print(f"")
    print(f"🎯 总吞吐提升: {improvement:+.1f}%")

    if improvement >= 30:
        print(f"\n✅ 成功！压缩带来 {improvement:.1f}% 吞吐提升 (超过30%目标)")
    elif improvement >= 0:
        print(f"\n⚠️ 吞吐提升 {improvement:.1f}% (未达到30%目标)")
    else:
        print(f"\n❌ 吞吐下降 {-improvement:.1f}%")

    return {
        'batch_size': batch_size,
        'prompt_tokens': sample_len,
        'max_output': max_output,
        'no_compress_throughput': no_compress_throughput,
        'compress_throughput': compress_throughput,
        'improvement': improvement
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data/huggingface/llava-1.5-7b-hf')
    parser.add_argument('--compressor', default=str(CKPT_DIR / "llava_mlp.pth"))
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--prompt', type=int, default=2500)
    parser.add_argument('--output', type=int, default=256)
    args = parser.parse_args()

    print("#" * 70)
    print(" 端到端吞吐测试 - 目标: 超过30%吞吐提升")
    print("#" * 70)

    result = test_e2e_throughput(
        args.model,
        args.compressor,
        batch_size=args.bs,
        prompt_tokens=args.prompt,
        max_output=args.output
    )

    print("\n" + "#" * 70)
    print(f" 最终结果: {result['improvement']:+.1f}% 吞吐提升")
    print("#" * 70)


if __name__ == '__main__':
    main()
