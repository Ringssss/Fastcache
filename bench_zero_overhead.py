#!/usr/bin/env python3
"""
Zero-Overhead Compression Benchmark
====================================

测试零开销压缩策略的效果：
1. Baseline: 无压缩
2. Sync Compress: 压缩后再decode（阻塞式）
3. Async Compress: 压缩与prefill重叠
4. Zero-Overhead: 压缩塞入decode间隙

关键指标：
- TPOT (Time Per Output Token): decode延迟
- 吞吐量: tokens/s
- 压缩开销: 额外增加的时间

"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys

import torch
import time
import argparse
from typing import List, Tuple, Dict
from dataclasses import dataclass
import numpy as np

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.llava_engine import LlavaLLM


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    total_time: float
    prefill_time: float
    decode_time: float
    compress_time: float
    num_output_tokens: int
    tpot_ms: float  # Time Per Output Token
    throughput: float  # tokens/s


def clear_gpu():
    """清理GPU缓存"""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def benchmark_no_compression(
    model_path: str,
    prompts: List[str],
    max_output: int = 128,
    warmup: int = 1
) -> BenchmarkResult:
    """
    基准测试：无压缩
    """
    print("\n" + "=" * 60)
    print("测试: 无压缩 (Baseline)")
    print("=" * 60)

    clear_gpu()

    llm = LlavaLLM(
        model_path,
        enable_compression=False,
        enforce_eager=True,
        max_model_len=4096,
    )

    sampling_params = [SamplingParams(max_tokens=max_output)] * len(prompts)

    # Warmup
    if warmup > 0:
        print(f"预热 {warmup} 次...")
        _ = llm.generate(prompts[:1], sampling_params[:1], use_tqdm=False)
        clear_gpu()

    # 测试
    print("开始测试...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    end_time = time.time()
    torch.cuda.synchronize()

    total_time = end_time - start_time
    num_tokens = sum(len(o['token_ids']) for o in outputs)

    result = BenchmarkResult(
        name="无压缩",
        total_time=total_time,
        prefill_time=0,  # 不分离测量
        decode_time=total_time,
        compress_time=0,
        num_output_tokens=num_tokens,
        tpot_ms=total_time / num_tokens * 1000 if num_tokens > 0 else 0,
        throughput=num_tokens / total_time if total_time > 0 else 0,
    )

    del llm
    clear_gpu()

    return result


def benchmark_sync_compression(
    model_path: str,
    prompts: List[str],
    max_output: int = 128,
    warmup: int = 1
) -> BenchmarkResult:
    """
    基准测试：同步压缩（阻塞式）
    """
    print("\n" + "=" * 60)
    print("测试: 同步压缩")
    print("=" * 60)

    clear_gpu()

    llm = LlavaLLM(
        model_path,
        enable_compression=True,
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=4096,
    )

    sampling_params = [SamplingParams(max_tokens=max_output)] * len(prompts)

    # Warmup
    if warmup > 0:
        print(f"预热 {warmup} 次...")
        _ = llm.generate(prompts[:1], sampling_params[:1], use_tqdm=False)
        clear_gpu()

    # 测试
    print("开始测试...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    end_time = time.time()
    torch.cuda.synchronize()

    total_time = end_time - start_time
    num_tokens = sum(len(o['token_ids']) for o in outputs)

    result = BenchmarkResult(
        name="同步压缩",
        total_time=total_time,
        prefill_time=0,
        decode_time=total_time,
        compress_time=0,  # 包含在total中
        num_output_tokens=num_tokens,
        tpot_ms=total_time / num_tokens * 1000 if num_tokens > 0 else 0,
        throughput=num_tokens / total_time if total_time > 0 else 0,
    )

    del llm
    clear_gpu()

    return result


def benchmark_async_compression(
    model_path: str,
    prompts: List[str],
    max_output: int = 128,
    warmup: int = 1
) -> BenchmarkResult:
    """
    基准测试：异步压缩（与prefill重叠）
    """
    print("\n" + "=" * 60)
    print("测试: 异步压缩（流水线）")
    print("=" * 60)

    clear_gpu()

    llm = LlavaLLM(
        model_path,
        enable_compression=True,
        async_compression=True,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=4096,
    )

    sampling_params = [SamplingParams(max_tokens=max_output)] * len(prompts)

    # Warmup
    if warmup > 0:
        print(f"预热 {warmup} 次...")
        _ = llm.generate(prompts[:1], sampling_params[:1], use_tqdm=False)
        clear_gpu()

    # 测试
    print("开始测试...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    end_time = time.time()
    torch.cuda.synchronize()

    total_time = end_time - start_time
    num_tokens = sum(len(o['token_ids']) for o in outputs)

    result = BenchmarkResult(
        name="异步压缩",
        total_time=total_time,
        prefill_time=0,
        decode_time=total_time,
        compress_time=0,
        num_output_tokens=num_tokens,
        tpot_ms=total_time / num_tokens * 1000 if num_tokens > 0 else 0,
        throughput=num_tokens / total_time if total_time > 0 else 0,
    )

    del llm
    clear_gpu()

    return result


def benchmark_zero_overhead_compression(
    model_path: str,
    prompts: List[str],
    max_output: int = 128,
    warmup: int = 1,
    layers_per_step: int = 4
) -> BenchmarkResult:
    """
    基准测试：零开销压缩（塞入decode间隙）

    核心实现：
    1. Prefill完成后，不立即压缩
    2. 每个decode step之后，在低优先级stream上压缩几层
    3. 利用decode的memory-bound特性，让压缩"免费"执行
    """
    print("\n" + "=" * 60)
    print(f"测试: 零开销压缩 (layers_per_step={layers_per_step})")
    print("=" * 60)

    clear_gpu()

    # 创建LLM（使用异步压缩作为基础）
    llm = LlavaLLM(
        model_path,
        enable_compression=True,
        async_compression=True,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=4096,
    )

    # 设置分层压缩参数
    if hasattr(llm.model_runner, 'batched_compressor'):
        # 配置零开销调度
        llm._zero_overhead_enabled = True
        llm._layers_per_step = layers_per_step
        print(f"✓ 零开销压缩已启用 (每步压缩 {layers_per_step} 层)")

    sampling_params = [SamplingParams(max_tokens=max_output)] * len(prompts)

    # Warmup
    if warmup > 0:
        print(f"预热 {warmup} 次...")
        _ = llm.generate(prompts[:1], sampling_params[:1], use_tqdm=False)
        clear_gpu()

    # 测试
    print("开始测试...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    end_time = time.time()
    torch.cuda.synchronize()

    total_time = end_time - start_time
    num_tokens = sum(len(o['token_ids']) for o in outputs)

    result = BenchmarkResult(
        name=f"零开销压缩(L={layers_per_step})",
        total_time=total_time,
        prefill_time=0,
        decode_time=total_time,
        compress_time=0,  # 理论上为0（被decode时间吸收）
        num_output_tokens=num_tokens,
        tpot_ms=total_time / num_tokens * 1000 if num_tokens > 0 else 0,
        throughput=num_tokens / total_time if total_time > 0 else 0,
    )

    del llm
    clear_gpu()

    return result


def print_results(results: List[BenchmarkResult]):
    """打印结果对比表"""
    print("\n")
    print("=" * 80)
    print(" 性能对比结果")
    print("=" * 80)

    # 表头
    print(f"{'配置':<25} {'总时间(s)':<12} {'吞吐量(tok/s)':<15} {'TPOT(ms)':<12} {'输出tokens':<12}")
    print("-" * 80)

    baseline = results[0] if results else None

    for r in results:
        speedup = ""
        if baseline and r.throughput > 0:
            ratio = r.throughput / baseline.throughput
            if ratio > 1:
                speedup = f" (+{(ratio-1)*100:.1f}%)"
            else:
                speedup = f" ({(ratio-1)*100:.1f}%)"

        print(f"{r.name:<25} {r.total_time:<12.3f} {r.throughput:<15.1f} {r.tpot_ms:<12.3f} {r.num_output_tokens:<12}{speedup}")

    print("=" * 80)

    # 关键洞察
    if len(results) >= 2:
        baseline = results[0]
        best_compress = max(results[1:], key=lambda x: x.throughput) if len(results) > 1 else None

        print("\n关键洞察:")
        if best_compress:
            overhead = (baseline.throughput - best_compress.throughput) / baseline.throughput * 100
            if overhead < 0:
                print(f"✓ 最佳压缩方案 '{best_compress.name}' 比无压缩快 {-overhead:.1f}%！")
            else:
                print(f"  压缩开销: {overhead:.1f}%")

            tpot_overhead = (best_compress.tpot_ms - baseline.tpot_ms) / baseline.tpot_ms * 100
            if tpot_overhead < 5:
                print(f"✓ TPOT几乎无增加 ({tpot_overhead:.1f}%)，压缩接近零开销！")
            else:
                print(f"  TPOT增加: {tpot_overhead:.1f}%")


def generate_prompts(num_prompts: int, input_len: int = 200) -> List[str]:
    """生成测试prompt"""
    base_prompt = "USER: Please provide a detailed analysis of the following topic: "
    topics = [
        "artificial intelligence and machine learning applications",
        "climate change and environmental sustainability",
        "global economic trends and market analysis",
        "healthcare innovations and medical research",
        "space exploration and astronomical discoveries",
        "renewable energy technologies and solutions",
        "cybersecurity threats and defense strategies",
        "educational technology and learning methods",
        "transportation infrastructure and urban planning",
        "agricultural practices and food security",
    ]

    prompts = []
    for i in range(num_prompts):
        topic = topics[i % len(topics)]
        # 添加一些填充文本以达到目标长度
        padding = " The analysis should cover historical context, current state, future projections, and potential challenges. " * (input_len // 50)
        prompt = f"{base_prompt}{topic}.{padding} ASSISTANT:"
        prompts.append(prompt)

    return prompts


def main():
    parser = argparse.ArgumentParser(description='Zero-Overhead Compression Benchmark')
    parser.add_argument('--model', type=str, default='/data/huggingface/llava-1.5-7b-hf',
                        help='Model path')
    parser.add_argument('--num_prompts', type=int, default=32,
                        help='Number of prompts')
    parser.add_argument('--max_output', type=int, default=256,
                        help='Maximum output tokens')
    parser.add_argument('--input_len', type=int, default=200,
                        help='Approximate input length')
    parser.add_argument('--warmup', type=int, default=1,
                        help='Number of warmup iterations')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline (no compression) test')
    args = parser.parse_args()

    print("#" * 80)
    print(" Zero-Overhead Compression Benchmark")
    print("#" * 80)
    print(f"模型: {args.model}")
    print(f"Prompts: {args.num_prompts}")
    print(f"最大输出: {args.max_output}")
    print(f"输入长度: ~{args.input_len}")

    # 生成测试prompts
    prompts = generate_prompts(args.num_prompts, args.input_len)
    print(f"生成了 {len(prompts)} 个prompts")

    results = []

    # 1. Baseline: 无压缩
    if not args.skip_baseline:
        try:
            r = benchmark_no_compression(args.model, prompts, args.max_output, args.warmup)
            results.append(r)
            print(f"✓ 无压缩完成: {r.throughput:.1f} tok/s")
        except Exception as e:
            print(f"✗ 无压缩测试失败: {e}")

    # 2. 同步压缩
    try:
        r = benchmark_sync_compression(args.model, prompts, args.max_output, args.warmup)
        results.append(r)
        print(f"✓ 同步压缩完成: {r.throughput:.1f} tok/s")
    except Exception as e:
        print(f"✗ 同步压缩测试失败: {e}")

    # 3. 异步压缩
    try:
        r = benchmark_async_compression(args.model, prompts, args.max_output, args.warmup)
        results.append(r)
        print(f"✓ 异步压缩完成: {r.throughput:.1f} tok/s")
    except Exception as e:
        print(f"✗ 异步压缩测试失败: {e}")

    # 4. 零开销压缩（不同粒度）
    for layers_per_step in [4, 8, 16]:
        try:
            r = benchmark_zero_overhead_compression(
                args.model, prompts, args.max_output, args.warmup, layers_per_step
            )
            results.append(r)
            print(f"✓ 零开销压缩(L={layers_per_step})完成: {r.throughput:.1f} tok/s")
        except Exception as e:
            print(f"✗ 零开销压缩(L={layers_per_step})测试失败: {e}")

    # 打印结果
    print_results(results)


if __name__ == '__main__':
    main()
