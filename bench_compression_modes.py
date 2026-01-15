#!/usr/bin/env python3
"""
压缩模式对比测试
================

对比三种模式的性能：
1. 无压缩 (Baseline)
2. 同步压缩
3. 异步压缩

测试关键指标：
- 吞吐量 (tokens/s)
- TPOT (Time Per Output Token)
- 压缩开销

"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys
import gc
import time
import argparse


import torch
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class TestResult:
    """测试结果"""
    mode: str
    total_time: float
    num_tokens: int
    throughput: float
    tpot_ms: float
    success: bool
    error: str = ""


def force_clear_gpu():
    """强制清理GPU内存"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"[GPU] 已分配: {allocated:.2f} GB")


def generate_prompts(num_prompts: int, complexity: str = "medium") -> List[str]:
    """生成测试prompts"""
    if complexity == "simple":
        base = "USER: What is {}? ASSISTANT:"
        topics = ["Python", "AI", "Machine Learning", "Deep Learning", "Neural Networks",
                  "Computer Vision", "NLP", "Reinforcement Learning", "Data Science", "Statistics"]
    else:
        base = "USER: Please explain {} in detail, covering its history, current applications, and future prospects. ASSISTANT:"
        topics = [
            "artificial intelligence",
            "machine learning algorithms",
            "deep neural networks",
            "computer vision applications",
            "natural language processing",
            "reinforcement learning",
            "generative AI models",
            "data mining techniques",
            "statistical analysis methods",
            "cloud computing infrastructure"
        ]

    prompts = []
    for i in range(num_prompts):
        topic = topics[i % len(topics)]
        prompts.append(base.format(topic))

    return prompts


def run_single_test(
    mode: str,  # 'none', 'sync', 'async'
    model_path: str,
    prompts: List[str],
    max_tokens: int,
    compression_factor: int = 5
) -> TestResult:
    """运行单个测试"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    mode_names = {
        'none': '无压缩',
        'sync': '同步压缩',
        'async': '异步压缩'
    }
    mode_name = mode_names.get(mode, mode)

    print(f"\n{'='*60}")
    print(f"测试: {mode_name}")
    print(f"{'='*60}")

    force_clear_gpu()

    try:
        # 配置参数
        enable_compression = (mode != 'none')
        async_compression = (mode == 'async')

        print(f"初始化LLM (compression={enable_compression}, async={async_compression})...")

        llm = LlavaLLM(
            model_path,
            enable_compression=enable_compression,
            async_compression=async_compression,
            compression_factor=compression_factor,
            enforce_eager=True,  # 禁用CUDA Graph以简化测试
            max_model_len=2048,
        )

        sampling_params = [SamplingParams(max_tokens=max_tokens)] * len(prompts)

        # 预热
        print("预热...")
        _ = llm.generate(prompts[:1], sampling_params[:1], use_tqdm=False)
        torch.cuda.synchronize()
        force_clear_gpu()

        # 正式测试
        print(f"开始测试 ({len(prompts)} prompts, max_tokens={max_tokens})...")
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        torch.cuda.synchronize()
        total_time = time.time() - start_time

        # 统计
        num_tokens = sum(len(o['token_ids']) for o in outputs)
        throughput = num_tokens / total_time if total_time > 0 else 0
        tpot = total_time / num_tokens * 1000 if num_tokens > 0 else 0

        print(f"\n结果:")
        print(f"  总时间: {total_time:.3f}s")
        print(f"  输出tokens: {num_tokens}")
        print(f"  吞吐量: {throughput:.1f} tok/s")
        print(f"  TPOT: {tpot:.2f} ms")

        result = TestResult(
            mode=mode,
            total_time=total_time,
            num_tokens=num_tokens,
            throughput=throughput,
            tpot_ms=tpot,
            success=True
        )

        del llm
        force_clear_gpu()

        return result

    except Exception as e:
        import traceback
        print(f"✗ {mode_name}测试失败: {e}")
        traceback.print_exc()

        force_clear_gpu()

        return TestResult(
            mode=mode,
            total_time=0,
            num_tokens=0,
            throughput=0,
            tpot_ms=0,
            success=False,
            error=str(e)
        )


def print_comparison(results: List[TestResult]):
    """打印对比结果"""
    print("\n")
    print("=" * 80)
    print(" 性能对比结果")
    print("=" * 80)

    successful = [r for r in results if r.success]
    if not successful:
        print("没有成功的测试")
        return

    mode_names = {
        'none': '无压缩 (Baseline)',
        'sync': '同步压缩',
        'async': '异步压缩'
    }

    print(f"{'模式':<22} {'时间(s)':<10} {'吞吐量(tok/s)':<16} {'TPOT(ms)':<12} {'对比基准':<15}")
    print("-" * 80)

    baseline = next((r for r in successful if r.mode == 'none'), None)

    for r in successful:
        name = mode_names.get(r.mode, r.mode)

        comparison = ""
        if baseline and r.mode != 'none':
            ratio = r.throughput / baseline.throughput
            if ratio >= 1:
                comparison = f"+{(ratio-1)*100:.1f}%"
            else:
                comparison = f"{(ratio-1)*100:.1f}%"

        print(f"{name:<22} {r.total_time:<10.3f} {r.throughput:<16.1f} {r.tpot_ms:<12.2f} {comparison:<15}")

    print("=" * 80)

    # 分析
    if baseline:
        print("\n📊 分析:")

        sync_result = next((r for r in successful if r.mode == 'sync'), None)
        async_result = next((r for r in successful if r.mode == 'async'), None)

        if sync_result:
            overhead = (baseline.throughput - sync_result.throughput) / baseline.throughput * 100
            print(f"  同步压缩开销: {overhead:.1f}%")

        if async_result:
            overhead = (baseline.throughput - async_result.throughput) / baseline.throughput * 100
            print(f"  异步压缩开销: {overhead:.1f}%")

        if sync_result and async_result:
            improvement = (async_result.throughput - sync_result.throughput) / sync_result.throughput * 100
            print(f"  异步 vs 同步: {improvement:+.1f}%")

        # 压缩收益分析
        print("\n💡 关键洞察:")
        if async_result and baseline:
            if async_result.throughput > baseline.throughput * 0.9:
                print(f"  ✓ 异步压缩开销很小（<10%），压缩接近零开销！")
            elif async_result.throughput > baseline.throughput * 0.8:
                print(f"  ○ 异步压缩开销适中（10-20%）")
            else:
                print(f"  ⚠ 异步压缩开销较大（>20%），需优化")


def main():
    parser = argparse.ArgumentParser(description='压缩模式对比测试')
    parser.add_argument('--model', type=str,
                        default='/data/huggingface/llava-1.5-7b-hf',
                        help='模型路径')
    parser.add_argument('--num_prompts', type=int, default=8,
                        help='测试prompts数量')
    parser.add_argument('--max_tokens', type=int, default=128,
                        help='最大输出tokens')
    parser.add_argument('--compression_factor', type=int, default=5,
                        help='压缩因子')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='跳过无压缩基准测试')
    parser.add_argument('--modes', type=str, default='all',
                        choices=['all', 'none', 'sync', 'async', 'compress'],
                        help='测试模式')
    args = parser.parse_args()

    print("#" * 80)
    print(" 压缩模式性能对比")
    print("#" * 80)
    print(f"模型: {args.model}")
    print(f"Prompts: {args.num_prompts}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"压缩因子: {args.compression_factor}")

    # 生成prompts
    prompts = generate_prompts(args.num_prompts, "medium")

    results = []

    # 确定要测试的模式
    if args.modes == 'all':
        modes_to_test = ['none', 'sync', 'async'] if not args.skip_baseline else ['sync', 'async']
    elif args.modes == 'compress':
        modes_to_test = ['sync', 'async']
    else:
        modes_to_test = [args.modes]

    # 运行测试
    for mode in modes_to_test:
        result = run_single_test(
            mode=mode,
            model_path=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            compression_factor=args.compression_factor
        )
        results.append(result)

    # 打印对比
    print_comparison(results)

    print("\n测试完成!")


if __name__ == '__main__':
    main()
