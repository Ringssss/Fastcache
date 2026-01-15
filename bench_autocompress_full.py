#!/usr/bin/env python3
"""
AutoCompress 完整基准测试
=========================

测试三种模式的吞吐量对比:
1. 无压缩 (baseline)
2. 压缩 + 同步执行 (无流水线优化)
3. 压缩 + AutoCompress最优流水线

测试维度:
- Batch sizes: 1, 4, 8, 16, 32, 64, 128, 256
- 序列长度: 512, 1024, 2048
- 输出长度: 32, 64, 128

"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys

import torch
import gc
import time
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime


def clear_gpu():
    """清理GPU内存"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_gpu_memory():
    """获取GPU显存使用情况"""
    return torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3


@dataclass
class BenchmarkConfig:
    """测试配置"""
    model_path: str
    compressor_path: str
    batch_sizes: List[int]
    seq_lens: List[int]
    output_lens: List[int]
    warmup_iters: int = 2
    test_iters: int = 3


@dataclass
class BenchmarkResult:
    """测试结果"""
    batch_size: int
    seq_len: int
    output_len: int
    mode: str
    total_tokens: int
    elapsed_time: float
    throughput: float  # tokens/s
    prefill_time: float
    decode_time: float
    memory_used_gb: float
    memory_peak_gb: float


def generate_prompts(num: int, target_tokens: int = 600) -> List[str]:
    """生成测试prompts"""
    base = "USER: Please explain "
    topics = ["artificial intelligence", "machine learning", "deep learning",
              "natural language processing", "computer vision", "reinforcement learning",
              "robotics", "quantum computing", "blockchain", "cloud computing"]
    expansion = " in great detail covering its history, current applications, technical challenges, and future developments. "

    prompts = []
    repeat = max(1, target_tokens // 30)
    for i in range(num):
        topic = topics[i % len(topics)]
        prompt = base + topic + expansion * repeat + " ASSISTANT:"
        prompts.append(prompt)
    return prompts


def run_benchmark_no_compress(
    model_path: str,
    compressor_path: str,
    batch_size: int,
    seq_len: int,
    output_len: int,
) -> BenchmarkResult:
    """测试无压缩模式"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    clear_gpu()

    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        enable_compression=False,  # 关闭压缩
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=8192,
    )

    prompts = generate_prompts(batch_size, seq_len)

    # 添加请求
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=output_len))

    # 运行
    total_tokens = 0
    prefill_time = 0
    decode_time = 0

    start = time.time()

    while not llm.is_finished():
        t0 = time.time()
        outputs, num_tokens = llm.step(apply_compression=False)
        elapsed = time.time() - t0

        if num_tokens > 0:
            prefill_time += elapsed
            total_tokens += num_tokens
        else:
            decode_time += elapsed
            total_tokens += (-num_tokens)

    total_time = time.time() - start
    mem_used, mem_peak = get_gpu_memory()

    del llm
    clear_gpu()

    return BenchmarkResult(
        batch_size=batch_size,
        seq_len=seq_len,
        output_len=output_len,
        mode="no_compress",
        total_tokens=total_tokens,
        elapsed_time=total_time,
        throughput=total_tokens / total_time,
        prefill_time=prefill_time,
        decode_time=decode_time,
        memory_used_gb=mem_used,
        memory_peak_gb=mem_peak,
    )


def run_benchmark_compress_sync(
    model_path: str,
    compressor_path: str,
    batch_size: int,
    seq_len: int,
    output_len: int,
) -> BenchmarkResult:
    """测试压缩+同步模式（无流水线优化）"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    clear_gpu()

    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        enable_compression=True,
        async_compression=False,  # 同步压缩
        compression_factor=5,
        enforce_eager=True,
        max_model_len=8192,
    )

    prompts = generate_prompts(batch_size, seq_len)

    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=output_len))

    total_tokens = 0
    prefill_time = 0
    decode_time = 0

    start = time.time()

    while not llm.is_finished():
        t0 = time.time()
        outputs, num_tokens = llm.step(apply_compression=True)  # 启用压缩
        elapsed = time.time() - t0

        if num_tokens > 0:
            prefill_time += elapsed
            total_tokens += num_tokens
        else:
            decode_time += elapsed
            total_tokens += (-num_tokens)

    total_time = time.time() - start
    mem_used, mem_peak = get_gpu_memory()

    del llm
    clear_gpu()

    return BenchmarkResult(
        batch_size=batch_size,
        seq_len=seq_len,
        output_len=output_len,
        mode="compress_sync",
        total_tokens=total_tokens,
        elapsed_time=total_time,
        throughput=total_tokens / total_time,
        prefill_time=prefill_time,
        decode_time=decode_time,
        memory_used_gb=mem_used,
        memory_peak_gb=mem_peak,
    )


def run_benchmark_compress_async(
    model_path: str,
    compressor_path: str,
    batch_size: int,
    seq_len: int,
    output_len: int,
) -> BenchmarkResult:
    """测试压缩+异步模式（AutoCompress流水线）"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    clear_gpu()

    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        enable_compression=True,
        async_compression=True,  # 异步压缩 (AutoCompress)
        compression_factor=5,
        enforce_eager=True,
        max_model_len=8192,
    )

    prompts = generate_prompts(batch_size, seq_len)

    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=output_len))

    total_tokens = 0
    prefill_time = 0
    decode_time = 0

    start = time.time()

    while not llm.is_finished():
        t0 = time.time()
        outputs, num_tokens = llm.step(apply_compression=True)
        elapsed = time.time() - t0

        if num_tokens > 0:
            prefill_time += elapsed
            total_tokens += num_tokens
        else:
            decode_time += elapsed
            total_tokens += (-num_tokens)

    total_time = time.time() - start
    mem_used, mem_peak = get_gpu_memory()

    del llm
    clear_gpu()

    return BenchmarkResult(
        batch_size=batch_size,
        seq_len=seq_len,
        output_len=output_len,
        mode="compress_async",
        total_tokens=total_tokens,
        elapsed_time=total_time,
        throughput=total_tokens / total_time,
        prefill_time=prefill_time,
        decode_time=decode_time,
        memory_used_gb=mem_used,
        memory_peak_gb=mem_peak,
    )


def run_full_benchmark(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """运行完整基准测试"""
    results = []

    total_tests = len(config.batch_sizes) * len(config.seq_lens) * len(config.output_lens) * 3
    current_test = 0

    for batch_size in config.batch_sizes:
        for seq_len in config.seq_lens:
            for output_len in config.output_lens:
                print(f"\n{'='*70}")
                print(f"Testing: batch_size={batch_size}, seq_len={seq_len}, output_len={output_len}")
                print(f"{'='*70}")

                # 测试无压缩
                current_test += 1
                print(f"\n[{current_test}/{total_tests}] Mode: no_compress")
                try:
                    result = run_benchmark_no_compress(
                        config.model_path,
                        config.compressor_path,
                        batch_size,
                        seq_len,
                        output_len,
                    )
                    results.append(result)
                    print(f"  Throughput: {result.throughput:.1f} tok/s, Memory: {result.memory_peak_gb:.2f} GB")
                except Exception as e:
                    print(f"  Error: {e}")

                # 测试压缩+同步
                current_test += 1
                print(f"\n[{current_test}/{total_tests}] Mode: compress_sync")
                try:
                    result = run_benchmark_compress_sync(
                        config.model_path,
                        config.compressor_path,
                        batch_size,
                        seq_len,
                        output_len,
                    )
                    results.append(result)
                    print(f"  Throughput: {result.throughput:.1f} tok/s, Memory: {result.memory_peak_gb:.2f} GB")
                except Exception as e:
                    print(f"  Error: {e}")

                # 测试压缩+异步
                current_test += 1
                print(f"\n[{current_test}/{total_tests}] Mode: compress_async (AutoCompress)")
                try:
                    result = run_benchmark_compress_async(
                        config.model_path,
                        config.compressor_path,
                        batch_size,
                        seq_len,
                        output_len,
                    )
                    results.append(result)
                    print(f"  Throughput: {result.throughput:.1f} tok/s, Memory: {result.memory_peak_gb:.2f} GB")
                except Exception as e:
                    print(f"  Error: {e}")

    return results


def print_results_summary(results: List[BenchmarkResult]):
    """打印结果摘要"""
    print("\n" + "="*100)
    print(" BENCHMARK RESULTS SUMMARY")
    print("="*100)

    # 按batch_size分组
    by_batch = {}
    for r in results:
        key = (r.batch_size, r.seq_len, r.output_len)
        if key not in by_batch:
            by_batch[key] = {}
        by_batch[key][r.mode] = r

    print(f"\n{'Batch':>6} | {'SeqLen':>6} | {'Output':>6} | {'NoCompress':>12} | {'Sync':>12} | {'Async':>12} | {'Sync vs NC':>10} | {'Async vs NC':>11} | {'Async vs Sync':>13}")
    print("-"*120)

    for key in sorted(by_batch.keys()):
        batch, seq_len, output_len = key
        modes = by_batch[key]

        nc = modes.get('no_compress')
        sync = modes.get('compress_sync')
        async_ = modes.get('compress_async')

        nc_tp = f"{nc.throughput:.0f}" if nc else "N/A"
        sync_tp = f"{sync.throughput:.0f}" if sync else "N/A"
        async_tp = f"{async_.throughput:.0f}" if async_ else "N/A"

        sync_vs_nc = f"{sync.throughput/nc.throughput:.2f}x" if (nc and sync) else "N/A"
        async_vs_nc = f"{async_.throughput/nc.throughput:.2f}x" if (nc and async_) else "N/A"
        async_vs_sync = f"{async_.throughput/sync.throughput:.2f}x" if (sync and async_) else "N/A"

        print(f"{batch:>6} | {seq_len:>6} | {output_len:>6} | {nc_tp:>12} | {sync_tp:>12} | {async_tp:>12} | {sync_vs_nc:>10} | {async_vs_nc:>11} | {async_vs_sync:>13}")


def save_results(results: List[BenchmarkResult], filename: str):
    """保存结果到JSON"""
    data = []
    for r in results:
        data.append({
            'batch_size': r.batch_size,
            'seq_len': r.seq_len,
            'output_len': r.output_len,
            'mode': r.mode,
            'total_tokens': r.total_tokens,
            'elapsed_time': r.elapsed_time,
            'throughput': r.throughput,
            'prefill_time': r.prefill_time,
            'decode_time': r.decode_time,
            'memory_used_gb': r.memory_used_gb,
            'memory_peak_gb': r.memory_peak_gb,
        })

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {filename}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='AutoCompress Full Benchmark')
    parser.add_argument('--model', default='/data/huggingface/llava-1.5-7b-hf',
                        help='Model path')
    parser.add_argument('--compressor', default=str(CKPT_DIR / "llava_mlp.pth"),
                        help='Compressor checkpoint path')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                        default=[1, 4, 8, 16, 32, 64],
                        help='Batch sizes to test')
    parser.add_argument('--seq-lens', type=int, nargs='+',
                        default=[512, 1024],
                        help='Sequence lengths to test')
    parser.add_argument('--output-lens', type=int, nargs='+',
                        default=[64],
                        help='Output lengths to test')
    parser.add_argument('--output', default=None,
                        help='Output JSON file')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with fewer configs')
    args = parser.parse_args()

    if args.quick:
        args.batch_sizes = [1, 8, 32]
        args.seq_lens = [512]
        args.output_lens = [32]

    print("#" * 80)
    print(" AutoCompress Full Benchmark")
    print("#" * 80)
    print(f"\nModel: {args.model}")
    print(f"Compressor: {args.compressor}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Seq lengths: {args.seq_lens}")
    print(f"Output lengths: {args.output_lens}")

    config = BenchmarkConfig(
        model_path=args.model,
        compressor_path=args.compressor,
        batch_sizes=args.batch_sizes,
        seq_lens=args.seq_lens,
        output_lens=args.output_lens,
    )

    results = run_full_benchmark(config)

    print_results_summary(results)

    if args.output:
        save_results(results, args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(results, f"autocompress_bench_{timestamp}.json")


if __name__ == '__main__':
    main()
