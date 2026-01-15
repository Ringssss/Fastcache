#!/usr/bin/env python3
"""
Llama + kvpress 在 nano-vllm 中的基准测试
==========================================

测试kvpress压缩方法在Llama模型上通过nano-vllm框架的性能。

测试对比:
- 无压缩 (baseline)
- kvpress streaming_llm
- kvpress streaming_llm + async
- kvpress knorm

"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys

import torch
import gc
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def clear_gpu():
    """清理GPU内存"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_gpu_memory():
    """获取GPU显存使用情况"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3
    return 0, 0


@dataclass
class BenchResult:
    """基准测试结果"""
    model_name: str
    method: str
    compression_factor: int
    batch_size: int
    input_len: int
    output_len: int
    async_compress: bool
    total_time_ms: float
    throughput_tok_s: float
    memory_peak_gb: float
    error: Optional[str] = None


def generate_prompts(num: int, target_tokens: int = 600) -> List[str]:
    """生成测试prompts"""
    base = "Please explain "
    topics = ["artificial intelligence", "machine learning", "deep learning",
              "natural language processing", "computer vision", "reinforcement learning",
              "robotics", "quantum computing", "blockchain", "cloud computing"]
    expansion = " in great detail covering its history, current applications, technical challenges, and future developments. "

    prompts = []
    repeat = max(1, target_tokens // 30)
    for i in range(num):
        topic = topics[i % len(topics)]
        prompt = base + topic + expansion * repeat
        prompts.append(prompt)
    return prompts


def run_llama_no_compress(
    model_path: str,
    batch_size: int,
    seq_len: int,
    output_len: int,
) -> BenchResult:
    """测试无压缩模式"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    clear_gpu()

    llm = LlavaLLM(
        model_path,
        enable_compression=False,
        enforce_eager=True,
        max_model_len=4096,
    )

    prompts = generate_prompts(batch_size, seq_len)

    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=output_len))

    total_tokens = 0

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    while not llm.is_finished():
        outputs, num_tokens = llm.step(apply_compression=False)
        if num_tokens > 0:
            total_tokens += num_tokens
        else:
            total_tokens += (-num_tokens)

    total_time = time.time() - start
    _, mem_peak = get_gpu_memory()

    del llm
    clear_gpu()

    return BenchResult(
        model_name=os.path.basename(model_path),
        method='none',
        compression_factor=1,
        batch_size=batch_size,
        input_len=seq_len,
        output_len=output_len,
        async_compress=False,
        total_time_ms=total_time * 1000,
        throughput_tok_s=total_tokens / total_time,
        memory_peak_gb=mem_peak,
    )


def run_llama_kvpress(
    model_path: str,
    batch_size: int,
    seq_len: int,
    output_len: int,
    kvpress_method: str = 'streaming_llm',
    compression_factor: int = 5,
    async_compress: bool = False,
) -> BenchResult:
    """测试kvpress压缩模式"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    clear_gpu()

    llm = LlavaLLM(
        model_path,
        enable_compression=True,
        compression_backend='kvpress',
        kvpress_method=kvpress_method,
        compression_factor=compression_factor,
        async_compression=async_compress,
        enforce_eager=True,
        max_model_len=4096,
    )

    prompts = generate_prompts(batch_size, seq_len)

    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=output_len))

    total_tokens = 0

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    while not llm.is_finished():
        outputs, num_tokens = llm.step(apply_compression=True)
        if num_tokens > 0:
            total_tokens += num_tokens
        else:
            total_tokens += (-num_tokens)

    total_time = time.time() - start
    _, mem_peak = get_gpu_memory()

    del llm
    clear_gpu()

    method_name = kvpress_method + ('_async' if async_compress else '')

    return BenchResult(
        model_name=os.path.basename(model_path),
        method=method_name,
        compression_factor=compression_factor,
        batch_size=batch_size,
        input_len=seq_len,
        output_len=output_len,
        async_compress=async_compress,
        total_time_ms=total_time * 1000,
        throughput_tok_s=total_tokens / total_time,
        memory_peak_gb=mem_peak,
    )


def run_full_benchmark(
    model_path: str,
    batch_sizes: List[int],
    seq_len: int = 512,
    output_len: int = 64,
    compression_factor: int = 5,
) -> List[BenchResult]:
    """运行完整基准测试"""
    results = []

    for batch_size in batch_sizes:
        print(f"\n{'='*70}")
        print(f"Testing batch_size={batch_size}")
        print(f"{'='*70}")

        # 1. 无压缩 baseline
        print(f"\n[1] No compression (baseline)")
        try:
            result = run_llama_no_compress(
                model_path, batch_size, seq_len, output_len
            )
            results.append(result)
            print(f"    Throughput: {result.throughput_tok_s:.1f} tok/s, Memory: {result.memory_peak_gb:.2f} GB")
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()

        # 2. kvpress streaming_llm
        print(f"\n[2] kvpress (streaming_llm)")
        try:
            result = run_llama_kvpress(
                model_path, batch_size, seq_len, output_len,
                kvpress_method='streaming_llm',
                compression_factor=compression_factor,
                async_compress=False,
            )
            results.append(result)
            print(f"    Throughput: {result.throughput_tok_s:.1f} tok/s, Memory: {result.memory_peak_gb:.2f} GB")
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()

        # 3. kvpress streaming_llm + async
        print(f"\n[3] kvpress (streaming_llm) + async")
        try:
            result = run_llama_kvpress(
                model_path, batch_size, seq_len, output_len,
                kvpress_method='streaming_llm',
                compression_factor=compression_factor,
                async_compress=True,
            )
            results.append(result)
            print(f"    Throughput: {result.throughput_tok_s:.1f} tok/s, Memory: {result.memory_peak_gb:.2f} GB")
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()

        # 4. kvpress knorm
        print(f"\n[4] kvpress (knorm)")
        try:
            result = run_llama_kvpress(
                model_path, batch_size, seq_len, output_len,
                kvpress_method='knorm',
                compression_factor=compression_factor,
                async_compress=False,
            )
            results.append(result)
            print(f"    Throughput: {result.throughput_tok_s:.1f} tok/s, Memory: {result.memory_peak_gb:.2f} GB")
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()

    return results


def print_results_table(results: List[BenchResult]):
    """打印结果表格"""
    print("\n" + "="*100)
    print(" LLAMA + KVPRESS BENCHMARK RESULTS")
    print("="*100)

    # 按batch_size分组
    by_batch = {}
    for r in results:
        if r.error:
            continue
        if r.batch_size not in by_batch:
            by_batch[r.batch_size] = {}
        by_batch[r.batch_size][r.method] = r

    # 获取所有方法
    all_methods = sorted(set(r.method for r in results if not r.error))

    # 打印表头
    header = f"{'Batch':>6} |"
    for method in all_methods:
        header += f" {method:>20} |"
    print(header)
    print("-"*len(header))

    # 打印每行
    for batch in sorted(by_batch.keys()):
        methods = by_batch[batch]
        row = f"{batch:>6} |"

        baseline = methods.get('none')
        for method in all_methods:
            r = methods.get(method)
            if r:
                tp = r.throughput_tok_s
                if method != 'none' and baseline:
                    speedup = tp / baseline.throughput_tok_s
                    row += f" {tp:>12.0f} ({speedup:.2f}x) |"
                else:
                    row += f" {tp:>20.0f} |"
            else:
                row += f" {'N/A':>20} |"
        print(row)


def save_results(results: List[BenchResult], filename: str):
    """保存结果"""
    data = []
    for r in results:
        data.append({
            'model_name': r.model_name,
            'method': r.method,
            'compression_factor': r.compression_factor,
            'batch_size': r.batch_size,
            'input_len': r.input_len,
            'output_len': r.output_len,
            'async_compress': r.async_compress,
            'total_time_ms': r.total_time_ms,
            'throughput_tok_s': r.throughput_tok_s,
            'memory_peak_gb': r.memory_peak_gb,
            'error': r.error,
        })

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {filename}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Llama + kvpress in nano-vllm Benchmark')
    parser.add_argument('--model', default='/data/huggingface/Llama-3.1-8B-Instruct',
                        help='Model path')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                        default=[1, 8, 32, 64, 128],
                        help='Batch sizes to test')
    parser.add_argument('--seq-len', type=int, default=512,
                        help='Input sequence length')
    parser.add_argument('--output-len', type=int, default=64,
                        help='Output sequence length')
    parser.add_argument('--compression-factor', type=int, default=5,
                        help='Compression factor (5 means 5x compression)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    args = parser.parse_args()

    print("#" * 80)
    print(" Llama + kvpress in nano-vllm Benchmark")
    print("#" * 80)
    print(f"\nModel: {args.model}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Seq length: {args.seq_len}")
    print(f"Output length: {args.output_len}")
    print(f"Compression factor: {args.compression_factor}")

    results = run_full_benchmark(
        model_path=args.model,
        batch_sizes=args.batch_sizes,
        seq_len=args.seq_len,
        output_len=args.output_len,
        compression_factor=args.compression_factor,
    )

    print_results_table(results)

    if args.output:
        save_results(results, args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(results, f"llama_kvpress_nanovllm_{timestamp}.json")


if __name__ == '__main__':
    main()
