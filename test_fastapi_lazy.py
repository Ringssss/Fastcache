#!/usr/bin/env python3
"""
测试FastAPI服务的lazy压缩功能
==============================

这个脚本直接测试NanoVLLMServer类，不需要启动HTTP服务。

"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys

import torch
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def clear_gpu():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def test_single_mode(server, mode, batch_size, input_len, output_len):
    """测试单个模式"""
    # 切换模式
    server.switch_compression_mode(mode, 'streaming_llm')

    # 生成prompts
    base = "Please explain artificial intelligence in detail "
    prompts = [base * (input_len // 10) for _ in range(batch_size)]

    # 运行批量生成
    result = server.generate_batch(prompts, output_len)

    return {
        'throughput': result['avg_throughput'],
        'time_ms': result['total_time_ms'],
        'total_tokens': result['total_tokens'],
    }


def run_comprehensive_test():
    """运行全面测试"""
    from nanovllm_fastapi_server import NanoVLLMServer

    print("=" * 80)
    print(" FastAPI Lazy Compression Test")
    print("=" * 80)

    # 配置
    model_path = '/data/huggingface/llava-1.5-7b-hf'
    batch_sizes = [32, 64]  # 减少一些以加快测试
    input_lens = [256, 512]
    output_len = 64

    print(f"\nModel: {model_path}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Input lengths: {input_lens}")
    print(f"Output length: {output_len}")

    # 初始化服务器（以lazy模式启动）
    print("\nInitializing server...")
    server = NanoVLLMServer(
        model_path=model_path,
        compression_mode='lazy',
        compression_backend='kvpress',
        kvpress_method='streaming_llm',
        compression_factor=5,
        compression_threshold=0.3,
        max_model_len=8192,
    )

    results = []

    for batch_size in batch_sizes:
        for input_len in input_lens:
            print(f"\n{'='*60}")
            print(f"Testing: batch_size={batch_size}, input_len={input_len}")
            print(f"{'='*60}")

            workload_results = {
                'batch_size': batch_size,
                'input_len': input_len,
                'modes': {}
            }

            # 测试各种模式
            for mode in ['none', 'eager', 'lazy']:
                print(f"  [{mode}]...", end=" ", flush=True)

                result = test_single_mode(server, mode, batch_size, input_len, output_len)
                workload_results['modes'][mode] = result

                print(f"{result['throughput']:.0f} tok/s")

            # 计算speedup
            baseline_tp = workload_results['modes']['none']['throughput']
            for mode in ['eager', 'lazy']:
                tp = workload_results['modes'][mode]['throughput']
                workload_results['modes'][mode]['speedup'] = tp / baseline_tp if baseline_tp > 0 else 0
                workload_results['modes'][mode]['is_beneficial'] = tp > baseline_tp

            results.append(workload_results)

    # 打印结果表格
    print("\n" + "=" * 80)
    print(" RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Workload':<20} | {'baseline':>12} | {'eager':>18} | {'lazy':>18}")
    print("-" * 80)

    eager_wins = 0
    lazy_wins = 0

    for r in results:
        bs = r['batch_size']
        in_len = r['input_len']

        baseline_tp = r['modes']['none']['throughput']
        eager_tp = r['modes']['eager']['throughput']
        lazy_tp = r['modes']['lazy']['throughput']

        eager_speedup = r['modes']['eager']['speedup']
        lazy_speedup = r['modes']['lazy']['speedup']

        # 判断谁赢
        if lazy_tp > eager_tp:
            lazy_wins += 1
            winner = "lazy"
        else:
            eager_wins += 1
            winner = "eager"

        print(f"bs={bs:<3} in={in_len:<4}       | {baseline_tp:>12.0f} | "
              f"{eager_tp:>8.0f} ({eager_speedup:.2f}x) | "
              f"{lazy_tp:>8.0f} ({lazy_speedup:.2f}x) {'*' if winner == 'lazy' else ''}")

    # 统计
    print("\n" + "=" * 80)
    print(" STATISTICS")
    print("=" * 80)

    total = len(results)

    # 统计eager
    eager_beneficial = sum(1 for r in results if r['modes']['eager']['is_beneficial'])
    eager_avg_speedup = sum(r['modes']['eager']['speedup'] for r in results) / total

    # 统计lazy
    lazy_beneficial = sum(1 for r in results if r['modes']['lazy']['is_beneficial'])
    lazy_avg_speedup = sum(r['modes']['lazy']['speedup'] for r in results) / total

    print(f"\neager: 有效率 {eager_beneficial}/{total} ({100*eager_beneficial/total:.0f}%), 平均加速 {eager_avg_speedup:.2f}x")
    print(f"lazy:  有效率 {lazy_beneficial}/{total} ({100*lazy_beneficial/total:.0f}%), 平均加速 {lazy_avg_speedup:.2f}x")
    print(f"\nlazy vs eager: lazy胜出 {lazy_wins}/{total} ({100*lazy_wins/total:.0f}%)")

    # 推荐
    recommendation = 'lazy' if lazy_wins >= eager_wins else 'eager'
    print(f"\n推荐策略: {recommendation}")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"fastapi_lazy_test_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'eager': {
                    'beneficial_rate': eager_beneficial / total,
                    'avg_speedup': eager_avg_speedup,
                },
                'lazy': {
                    'beneficial_rate': lazy_beneficial / total,
                    'avg_speedup': lazy_avg_speedup,
                },
                'lazy_vs_eager_win_rate': lazy_wins / total,
                'recommendation': recommendation,
            }
        }, f, indent=2)

    print(f"\n结果已保存到: {output_file}")

    # 清理
    del server
    clear_gpu()


if __name__ == '__main__':
    run_comprehensive_test()
