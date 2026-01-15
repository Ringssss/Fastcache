#!/usr/bin/env python3
"""
压缩方法详细Profiling
=====================

分析各种压缩方法的:
1. 压缩本身的耗时
2. 压缩后decode的加速/减速
3. 不同序列长度下的效果
4. 是否存在"负效果"场景

这很重要：压缩不一定总是正收益！
"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys

import torch
import gc
import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class ProfileResult:
    """详细profiling结果"""
    model_name: str
    method: str
    compression_factor: int
    batch_size: int
    input_len: int
    output_len: int

    # 时间breakdown
    prefill_time_ms: float
    compression_time_ms: float  # 压缩本身耗时
    decode_time_ms: float
    total_time_ms: float

    # 吞吐
    prefill_throughput: float  # tokens/s
    decode_throughput: float   # tokens/s
    overall_throughput: float  # tokens/s

    # 压缩效果
    original_kv_len: int
    compressed_kv_len: int
    actual_compression_ratio: float

    # 内存
    memory_peak_gb: float

    # 是否有正收益
    speedup_vs_baseline: Optional[float] = None
    is_beneficial: Optional[bool] = None


def profile_no_compression(
    model_path: str,
    batch_size: int,
    input_len: int,
    output_len: int,
) -> ProfileResult:
    """Profile无压缩baseline"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    clear_gpu()

    llm = LlavaLLM(
        model_path,
        enable_compression=False,
        enforce_eager=True,
        max_model_len=4096,
    )

    # 生成prompts
    base_prompt = "Please explain artificial intelligence " * (input_len // 10)
    prompts = [base_prompt] * batch_size

    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=output_len))

    torch.cuda.reset_peak_memory_stats()

    # Prefill阶段计时
    prefill_start = time.time()
    prefill_tokens = 0

    # 第一个step是prefill
    outputs, num_tokens = llm.step(apply_compression=False)
    if num_tokens > 0:
        prefill_tokens = num_tokens
    else:
        prefill_tokens = -num_tokens

    prefill_time = time.time() - prefill_start

    # Decode阶段计时
    decode_start = time.time()
    decode_tokens = 0

    while not llm.is_finished():
        outputs, num_tokens = llm.step(apply_compression=False)
        if num_tokens > 0:
            decode_tokens += num_tokens
        else:
            decode_tokens += (-num_tokens)

    decode_time = time.time() - decode_start
    total_time = prefill_time + decode_time

    _, mem_peak = torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3

    # 获取原始KV长度
    original_kv_len = input_len  # 近似

    del llm
    clear_gpu()

    return ProfileResult(
        model_name=os.path.basename(model_path),
        method='none',
        compression_factor=1,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        prefill_time_ms=prefill_time * 1000,
        compression_time_ms=0,
        decode_time_ms=decode_time * 1000,
        total_time_ms=total_time * 1000,
        prefill_throughput=prefill_tokens / prefill_time if prefill_time > 0 else 0,
        decode_throughput=decode_tokens / decode_time if decode_time > 0 else 0,
        overall_throughput=(prefill_tokens + decode_tokens) / total_time,
        original_kv_len=original_kv_len,
        compressed_kv_len=original_kv_len,
        actual_compression_ratio=1.0,
        memory_peak_gb=mem_peak,
    )


def profile_with_compression(
    model_path: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    compression_backend: str = 'kvpress',
    kvpress_method: str = 'streaming_llm',
    compression_factor: int = 5,
    async_compress: bool = False,
    compressor_path: Optional[str] = None,
) -> ProfileResult:
    """Profile带压缩的方法"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    clear_gpu()

    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        enable_compression=True,
        compression_backend=compression_backend,
        kvpress_method=kvpress_method,
        compression_factor=compression_factor,
        async_compression=async_compress,
        enforce_eager=True,
        max_model_len=4096,
    )

    base_prompt = "Please explain artificial intelligence " * (input_len // 10)
    prompts = [base_prompt] * batch_size

    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=output_len))

    torch.cuda.reset_peak_memory_stats()

    # Prefill + 压缩计时
    prefill_start = time.time()
    prefill_tokens = 0

    outputs, num_tokens = llm.step(apply_compression=True)
    if num_tokens > 0:
        prefill_tokens = num_tokens
    else:
        prefill_tokens = -num_tokens

    prefill_compress_time = time.time() - prefill_start

    # Decode阶段计时
    decode_start = time.time()
    decode_tokens = 0

    while not llm.is_finished():
        outputs, num_tokens = llm.step(apply_compression=True)
        if num_tokens > 0:
            decode_tokens += num_tokens
        else:
            decode_tokens += (-num_tokens)

    decode_time = time.time() - decode_start
    total_time = prefill_compress_time + decode_time

    _, mem_peak = torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3

    # 计算压缩后的KV长度
    original_kv_len = input_len
    compressed_kv_len = max(1, int(original_kv_len / compression_factor))
    actual_ratio = original_kv_len / compressed_kv_len if compressed_kv_len > 0 else compression_factor

    method_name = kvpress_method if compression_backend == 'kvpress' else 'mlp'
    if async_compress:
        method_name += '_async'

    del llm
    clear_gpu()

    return ProfileResult(
        model_name=os.path.basename(model_path),
        method=method_name,
        compression_factor=compression_factor,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        prefill_time_ms=prefill_compress_time * 1000,  # 包含压缩
        compression_time_ms=0,  # 难以精确分离
        decode_time_ms=decode_time * 1000,
        total_time_ms=total_time * 1000,
        prefill_throughput=prefill_tokens / prefill_compress_time if prefill_compress_time > 0 else 0,
        decode_throughput=decode_tokens / decode_time if decode_time > 0 else 0,
        overall_throughput=(prefill_tokens + decode_tokens) / total_time,
        original_kv_len=original_kv_len,
        compressed_kv_len=compressed_kv_len,
        actual_compression_ratio=actual_ratio,
        memory_peak_gb=mem_peak,
    )


def run_comprehensive_profile(
    model_path: str,
    compressor_path: Optional[str] = None,
    batch_sizes: List[int] = [1, 8, 32, 64, 128],
    input_lens: List[int] = [256, 512, 1024, 2048],
    output_len: int = 64,
    compression_factor: int = 5,
) -> List[ProfileResult]:
    """运行全面的profiling"""

    results = []

    # 压缩方法列表
    methods = [
        ('none', None, False),  # baseline
        ('kvpress', 'streaming_llm', False),
        ('kvpress', 'streaming_llm', True),  # async
        ('kvpress', 'knorm', False),
        ('kvpress', 'random', False),
    ]

    # 如果有MLP压缩器，加入测试
    if compressor_path and os.path.exists(compressor_path):
        methods.append(('mlp', 'mlp', False))
        methods.append(('mlp', 'mlp', True))  # async

    for batch_size in batch_sizes:
        for input_len in input_lens:
            print(f"\n{'='*60}")
            print(f"Profiling: batch_size={batch_size}, input_len={input_len}")
            print(f"{'='*60}")

            baseline_result = None

            for backend, method, async_compress in methods:
                method_name = f"{backend}:{method}" + ("_async" if async_compress else "")
                print(f"\n  Testing {method_name}...", end=" ", flush=True)

                try:
                    if backend == 'none':
                        result = profile_no_compression(
                            model_path, batch_size, input_len, output_len
                        )
                        baseline_result = result
                    else:
                        result = profile_with_compression(
                            model_path, batch_size, input_len, output_len,
                            compression_backend=backend,
                            kvpress_method=method,
                            compression_factor=compression_factor,
                            async_compress=async_compress,
                            compressor_path=compressor_path if backend == 'mlp' else None,
                        )

                    # 计算相对baseline的加速
                    if baseline_result and backend != 'none':
                        speedup = result.overall_throughput / baseline_result.overall_throughput
                        result.speedup_vs_baseline = speedup
                        result.is_beneficial = speedup > 1.0

                    results.append(result)

                    tp = result.overall_throughput
                    speedup_str = f" ({result.speedup_vs_baseline:.2f}x)" if result.speedup_vs_baseline else ""
                    beneficial = "+" if result.is_beneficial else ("-" if result.is_beneficial == False else "")
                    print(f"{tp:.0f} tok/s{speedup_str} {beneficial}")

                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()

    return results


def analyze_results(results: List[ProfileResult]):
    """分析结果，找出负效果场景"""

    print("\n" + "="*80)
    print(" ANALYSIS: When is compression NOT beneficial?")
    print("="*80)

    negative_cases = [r for r in results if r.is_beneficial == False]
    positive_cases = [r for r in results if r.is_beneficial == True]

    print(f"\nTotal tests: {len(results)}")
    print(f"Positive cases (speedup > 1.0): {len(positive_cases)}")
    print(f"Negative cases (speedup < 1.0): {len(negative_cases)}")

    if negative_cases:
        print("\n NEGATIVE CASES (compression hurts performance):")
        print("-"*80)
        for r in negative_cases:
            print(f"  {r.method:20} bs={r.batch_size:3} in={r.input_len:4} "
                  f"→ {r.speedup_vs_baseline:.2f}x ({r.overall_throughput:.0f} tok/s)")

    # 按方法分组分析
    print("\n METHOD ANALYSIS:")
    print("-"*80)

    methods = set(r.method for r in results if r.method != 'none')
    for method in sorted(methods):
        method_results = [r for r in results if r.method == method]
        positive = sum(1 for r in method_results if r.is_beneficial)
        total = len(method_results)
        avg_speedup = sum(r.speedup_vs_baseline for r in method_results if r.speedup_vs_baseline) / total if total > 0 else 0

        print(f"  {method:25} Beneficial: {positive}/{total} ({100*positive/total:.0f}%)  Avg speedup: {avg_speedup:.2f}x")

    # 按batch size分析
    print("\n BATCH SIZE ANALYSIS:")
    print("-"*80)

    for bs in sorted(set(r.batch_size for r in results)):
        bs_results = [r for r in results if r.batch_size == bs and r.method != 'none']
        positive = sum(1 for r in bs_results if r.is_beneficial)
        total = len(bs_results)
        if total > 0:
            print(f"  bs={bs:3}  Beneficial: {positive}/{total} ({100*positive/total:.0f}%)")

    # 按input length分析
    print("\n INPUT LENGTH ANALYSIS:")
    print("-"*80)

    for in_len in sorted(set(r.input_len for r in results)):
        len_results = [r for r in results if r.input_len == in_len and r.method != 'none']
        positive = sum(1 for r in len_results if r.is_beneficial)
        total = len(len_results)
        if total > 0:
            print(f"  in_len={in_len:4}  Beneficial: {positive}/{total} ({100*positive/total:.0f}%)")


def save_results(results: List[ProfileResult], filename: str):
    """保存结果"""
    data = [asdict(r) for r in results]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filename}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compression Methods Profiling')
    parser.add_argument('--model', default='/data/huggingface/llava-1.5-7b-hf')
    parser.add_argument('--compressor', default=str(CKPT_DIR / "llava_mlp.pth"))
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[8, 32, 64, 128])
    parser.add_argument('--input-lens', type=int, nargs='+', default=[256, 512, 1024])
    parser.add_argument('--output-len', type=int, default=64)
    parser.add_argument('--compression-factor', type=int, default=5)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    print("#" * 80)
    print(" Compression Methods Profiling")
    print("#" * 80)
    print(f"\nModel: {args.model}")
    print(f"Compressor: {args.compressor}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Input lengths: {args.input_lens}")
    print(f"Output length: {args.output_len}")
    print(f"Compression factor: {args.compression_factor}")

    results = run_comprehensive_profile(
        model_path=args.model,
        compressor_path=args.compressor,
        batch_sizes=args.batch_sizes,
        input_lens=args.input_lens,
        output_len=args.output_len,
        compression_factor=args.compression_factor,
    )

    analyze_results(results)

    if args.output:
        save_results(results, args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(results, f"compression_profile_{timestamp}.json")


if __name__ == '__main__':
    main()
