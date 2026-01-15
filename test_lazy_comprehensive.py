#!/usr/bin/env python3
"""
Lazy压缩策略全面测试
====================

测试lazy压缩（阈值触发批量压缩）vs eager压缩（每次prefill后立即压缩）
在各种workload下的效果。

策略说明:
1. lazy压缩: 不在prefill后立即压缩，等空闲blocks低于阈值时才批量压缩
2. 批量压缩: 一次压缩多个序列，减少kernel launch开销
3. 这个策略的关键是: 减少压缩的频率，但不减少最终的压缩效果

支持kvpress方法的lazy压缩！

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
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def generate_prompts(num: int, target_tokens: int = 600) -> List[str]:
    base = "USER: Please explain "
    topics = ["AI", "ML", "DL", "NLP", "CV", "RL", "robotics", "quantum", "blockchain", "cloud"]
    expansion = " in great detail covering history, applications, challenges, and future. "

    prompts = []
    repeat = max(1, target_tokens // 25)
    for i in range(num):
        topic = topics[i % len(topics)]
        prompt = base + topic + expansion * repeat + " ASSISTANT:"
        prompts.append(prompt)
    return prompts


class LazyCompressionEngine:
    """
    懒压缩引擎 - 包装LlavaLLM，实现懒压缩策略

    支持kvpress和mlp两种压缩方法
    """

    def __init__(
        self,
        llm,
        compression_threshold: float = 0.3,
        compression_backend: str = 'kvpress',
    ):
        """
        Args:
            llm: LlavaLLM实例
            compression_threshold: 当空闲blocks低于总blocks的这个比例时触发压缩
            compression_backend: 'kvpress' 或 'mlp'
        """
        self.llm = llm
        self.compression_threshold = compression_threshold
        self.compression_backend = compression_backend

        # 获取block信息
        self.total_blocks = len(llm.scheduler.block_manager.blocks)
        self.threshold_blocks = int(self.total_blocks * compression_threshold)

        # 追踪未压缩的序列
        self.uncompressed_seqs = set()

        # 统计
        self.compression_count = 0
        self.total_compression_time = 0

        print(f"懒压缩引擎初始化:")
        print(f"  后端: {compression_backend}")
        print(f"  总blocks: {self.total_blocks}")
        print(f"  压缩阈值: {self.threshold_blocks} blocks ({compression_threshold*100:.0f}%)")

    def add_request(self, prompt, sampling_params):
        self.llm.add_request(prompt, sampling_params)

    def is_finished(self):
        return self.llm.is_finished()

    def step(self):
        """执行一步，带懒压缩策略"""
        # 检查是否需要压缩
        free_blocks = len(self.llm.scheduler.block_manager.free_block_ids)

        # 触发压缩条件: 空闲blocks低于阈值 且 有未压缩的序列
        if free_blocks < self.threshold_blocks and self.uncompressed_seqs:
            self._do_batch_compression()

        # 正常执行step（不压缩）
        outputs, num_tokens = self.llm.step(apply_compression=False)

        # 记录新的prefill序列
        if num_tokens > 0:  # prefill
            for seq in self.llm.scheduler.running:
                if seq.seq_id not in self.uncompressed_seqs:
                    self.uncompressed_seqs.add(seq.seq_id)

        # 清理已完成的序列
        for seq_id, _ in outputs:
            self.uncompressed_seqs.discard(seq_id)

        return outputs, num_tokens

    def _do_batch_compression(self):
        """执行批量压缩"""
        # 找到所有未压缩的running序列
        seqs_to_compress = [
            seq for seq in self.llm.scheduler.running
            if seq.seq_id in self.uncompressed_seqs
        ]

        if not seqs_to_compress:
            return

        start = time.time()

        # 批量压缩
        comp_time, comp_ratio = self.llm.model_runner.compress_kv_cache_batch(seqs_to_compress)

        # 释放blocks
        for seq in seqs_to_compress:
            self.llm._free_compressed_blocks([seq])
            self.uncompressed_seqs.discard(seq.seq_id)

        elapsed = time.time() - start
        self.compression_count += 1
        self.total_compression_time += elapsed

        free_after = len(self.llm.scheduler.block_manager.free_block_ids)
        # print(f"[懒压缩] 批量压缩 {len(seqs_to_compress)} 个序列, "
        #       f"耗时: {elapsed*1000:.1f}ms, 压缩比: {comp_ratio:.2f}x, "
        #       f"空闲blocks: {free_after}")


@dataclass
class TestResult:
    """测试结果"""
    model_name: str
    compression_mode: str  # 'none', 'eager', 'lazy', 'async'
    compression_backend: str  # 'kvpress', 'mlp', 'none'
    kvpress_method: str
    batch_size: int
    input_len: int
    output_len: int
    total_time_ms: float
    throughput_tok_s: float
    compression_count: int
    memory_peak_gb: float
    speedup_vs_none: Optional[float] = None
    speedup_vs_eager: Optional[float] = None


def run_test_no_compression(
    model_path: str,
    batch_size: int,
    input_len: int,
    output_len: int,
) -> TestResult:
    """测试无压缩baseline"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    clear_gpu()

    llm = LlavaLLM(
        model_path,
        enable_compression=False,
        enforce_eager=True,
        max_model_len=8192,
    )

    prompts = generate_prompts(batch_size, input_len)
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=output_len))

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    total_tokens = 0

    while not llm.is_finished():
        outputs, num_tokens = llm.step(apply_compression=False)
        total_tokens += abs(num_tokens)

    total_time = time.time() - start
    _, mem_peak = torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3

    del llm
    clear_gpu()

    return TestResult(
        model_name=os.path.basename(model_path),
        compression_mode='none',
        compression_backend='none',
        kvpress_method='none',
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        total_time_ms=total_time * 1000,
        throughput_tok_s=total_tokens / total_time,
        compression_count=0,
        memory_peak_gb=mem_peak,
    )


def run_test_eager_compression(
    model_path: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    compression_backend: str = 'kvpress',
    kvpress_method: str = 'streaming_llm',
    compression_factor: int = 5,
    compressor_path: Optional[str] = None,
) -> TestResult:
    """测试即时压缩（每次prefill后立即压缩）"""
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
        async_compression=False,
        enforce_eager=True,
        max_model_len=8192,
    )

    prompts = generate_prompts(batch_size, input_len)
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=output_len))

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    total_tokens = 0
    compression_count = 0

    while not llm.is_finished():
        outputs, num_tokens = llm.step(apply_compression=True)
        total_tokens += abs(num_tokens)
        if num_tokens > 0:  # prefill = compression
            compression_count += 1

    total_time = time.time() - start
    _, mem_peak = torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3

    del llm
    clear_gpu()

    return TestResult(
        model_name=os.path.basename(model_path),
        compression_mode='eager',
        compression_backend=compression_backend,
        kvpress_method=kvpress_method,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        total_time_ms=total_time * 1000,
        throughput_tok_s=total_tokens / total_time,
        compression_count=compression_count,
        memory_peak_gb=mem_peak,
    )


def run_test_async_compression(
    model_path: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    compression_backend: str = 'kvpress',
    kvpress_method: str = 'streaming_llm',
    compression_factor: int = 5,
    compressor_path: Optional[str] = None,
) -> TestResult:
    """测试异步压缩"""
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
        async_compression=True,
        enforce_eager=True,
        max_model_len=8192,
    )

    prompts = generate_prompts(batch_size, input_len)
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=output_len))

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    total_tokens = 0
    compression_count = 0

    while not llm.is_finished():
        outputs, num_tokens = llm.step(apply_compression=True)
        total_tokens += abs(num_tokens)
        if num_tokens > 0:
            compression_count += 1

    total_time = time.time() - start
    _, mem_peak = torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3

    del llm
    clear_gpu()

    return TestResult(
        model_name=os.path.basename(model_path),
        compression_mode='async',
        compression_backend=compression_backend,
        kvpress_method=kvpress_method,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        total_time_ms=total_time * 1000,
        throughput_tok_s=total_tokens / total_time,
        compression_count=compression_count,
        memory_peak_gb=mem_peak,
    )


def run_test_lazy_compression(
    model_path: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    compression_backend: str = 'kvpress',
    kvpress_method: str = 'streaming_llm',
    compression_factor: int = 5,
    compression_threshold: float = 0.3,
    compressor_path: Optional[str] = None,
) -> TestResult:
    """测试懒压缩（阈值触发批量压缩）"""
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
        async_compression=False,
        enforce_eager=True,
        max_model_len=8192,
    )

    lazy_engine = LazyCompressionEngine(
        llm,
        compression_threshold=compression_threshold,
        compression_backend=compression_backend,
    )

    prompts = generate_prompts(batch_size, input_len)
    for prompt in prompts:
        lazy_engine.add_request(prompt, SamplingParams(max_tokens=output_len))

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    total_tokens = 0

    while not lazy_engine.is_finished():
        outputs, num_tokens = lazy_engine.step()
        total_tokens += abs(num_tokens)

    total_time = time.time() - start
    _, mem_peak = torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3

    compression_count = lazy_engine.compression_count

    del llm, lazy_engine
    clear_gpu()

    return TestResult(
        model_name=os.path.basename(model_path),
        compression_mode='lazy',
        compression_backend=compression_backend,
        kvpress_method=kvpress_method,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        total_time_ms=total_time * 1000,
        throughput_tok_s=total_tokens / total_time,
        compression_count=compression_count,
        memory_peak_gb=mem_peak,
    )


def run_comprehensive_test(
    model_path: str,
    compressor_path: Optional[str] = None,
    batch_sizes: List[int] = [32, 64, 128],
    input_lens: List[int] = [256, 512, 1024],
    output_len: int = 64,
    kvpress_methods: List[str] = ['streaming_llm'],
    compression_factor: int = 5,
) -> List[TestResult]:
    """运行全面测试"""

    results = []

    for batch_size in batch_sizes:
        for input_len in input_lens:
            print(f"\n{'='*70}")
            print(f"Testing: batch_size={batch_size}, input_len={input_len}")
            print(f"{'='*70}")

            # 1. baseline
            print(f"  [1/5] No compression (baseline)...", end=" ", flush=True)
            try:
                baseline = run_test_no_compression(model_path, batch_size, input_len, output_len)
                results.append(baseline)
                print(f"{baseline.throughput_tok_s:.0f} tok/s")
            except Exception as e:
                print(f"Error: {e}")
                continue

            # 对每种kvpress方法测试
            for kvpress_method in kvpress_methods:
                # 2. eager compression
                print(f"  [2/5] Eager compression ({kvpress_method})...", end=" ", flush=True)
                try:
                    eager = run_test_eager_compression(
                        model_path, batch_size, input_len, output_len,
                        compression_backend='kvpress',
                        kvpress_method=kvpress_method,
                        compression_factor=compression_factor,
                    )
                    eager.speedup_vs_none = eager.throughput_tok_s / baseline.throughput_tok_s
                    results.append(eager)
                    print(f"{eager.throughput_tok_s:.0f} tok/s ({eager.speedup_vs_none:.2f}x)")
                except Exception as e:
                    print(f"Error: {e}")
                    continue

                # 3. async compression
                print(f"  [3/5] Async compression ({kvpress_method})...", end=" ", flush=True)
                try:
                    async_result = run_test_async_compression(
                        model_path, batch_size, input_len, output_len,
                        compression_backend='kvpress',
                        kvpress_method=kvpress_method,
                        compression_factor=compression_factor,
                    )
                    async_result.speedup_vs_none = async_result.throughput_tok_s / baseline.throughput_tok_s
                    async_result.speedup_vs_eager = async_result.throughput_tok_s / eager.throughput_tok_s
                    results.append(async_result)
                    print(f"{async_result.throughput_tok_s:.0f} tok/s ({async_result.speedup_vs_none:.2f}x vs baseline)")
                except Exception as e:
                    print(f"Error: {e}")
                    continue

                # 4. lazy compression
                print(f"  [4/5] Lazy compression ({kvpress_method})...", end=" ", flush=True)
                try:
                    lazy = run_test_lazy_compression(
                        model_path, batch_size, input_len, output_len,
                        compression_backend='kvpress',
                        kvpress_method=kvpress_method,
                        compression_factor=compression_factor,
                        compression_threshold=0.3,
                    )
                    lazy.speedup_vs_none = lazy.throughput_tok_s / baseline.throughput_tok_s
                    lazy.speedup_vs_eager = lazy.throughput_tok_s / eager.throughput_tok_s
                    results.append(lazy)
                    print(f"{lazy.throughput_tok_s:.0f} tok/s ({lazy.speedup_vs_none:.2f}x vs baseline, {lazy.speedup_vs_eager:.2f}x vs eager)")
                except Exception as e:
                    print(f"Error: {e}")

            # 5. MLP压缩 (如果有compressor)
            if compressor_path and os.path.exists(compressor_path):
                print(f"  [5/5] Lazy compression (MLP)...", end=" ", flush=True)
                try:
                    mlp_lazy = run_test_lazy_compression(
                        model_path, batch_size, input_len, output_len,
                        compression_backend='mlp',
                        kvpress_method='mlp',
                        compression_factor=compression_factor,
                        compression_threshold=0.3,
                        compressor_path=compressor_path,
                    )
                    mlp_lazy.speedup_vs_none = mlp_lazy.throughput_tok_s / baseline.throughput_tok_s
                    results.append(mlp_lazy)
                    print(f"{mlp_lazy.throughput_tok_s:.0f} tok/s ({mlp_lazy.speedup_vs_none:.2f}x)")
                except Exception as e:
                    print(f"Error: {e}")

    return results


def analyze_results(results: List[TestResult]):
    """分析结果"""
    print("\n" + "="*90)
    print(" COMPREHENSIVE LAZY COMPRESSION TEST RESULTS")
    print("="*90)

    # 按workload分组
    by_workload = {}
    for r in results:
        key = (r.batch_size, r.input_len)
        if key not in by_workload:
            by_workload[key] = {}
        by_workload[key][f"{r.compression_mode}_{r.kvpress_method}"] = r

    # 打印表格
    print(f"\n{'Workload':<20} | {'baseline':>12} | {'eager':>12} | {'async':>12} | {'lazy':>12} | {'lazy vs eager':>12}")
    print("-"*90)

    for (bs, in_len), modes in sorted(by_workload.items()):
        baseline = modes.get('none_none')
        eager = modes.get('eager_streaming_llm')
        async_r = modes.get('async_streaming_llm')
        lazy = modes.get('lazy_streaming_llm')

        if not baseline:
            continue

        baseline_tp = baseline.throughput_tok_s
        eager_tp = eager.throughput_tok_s if eager else 0
        async_tp = async_r.throughput_tok_s if async_r else 0
        lazy_tp = lazy.throughput_tok_s if lazy else 0

        eager_ratio = f"({eager_tp/baseline_tp:.2f}x)" if eager else ""
        async_ratio = f"({async_tp/baseline_tp:.2f}x)" if async_r else ""
        lazy_ratio = f"({lazy_tp/baseline_tp:.2f}x)" if lazy else ""
        lazy_vs_eager = f"{lazy_tp/eager_tp:.2f}x" if lazy and eager and eager_tp > 0 else "N/A"

        print(f"bs={bs:<3} in={in_len:<4}       | {baseline_tp:>12.0f} | {eager_tp:>6.0f} {eager_ratio:<5} | {async_tp:>6.0f} {async_ratio:<5} | {lazy_tp:>6.0f} {lazy_ratio:<5} | {lazy_vs_eager:>12}")

    # 统计分析
    print("\n" + "="*90)
    print(" SUMMARY")
    print("="*90)

    # 按模式统计
    for mode in ['eager', 'async', 'lazy']:
        mode_results = [r for r in results if r.compression_mode == mode and r.speedup_vs_none is not None]
        if mode_results:
            positive = sum(1 for r in mode_results if r.speedup_vs_none > 1.0)
            avg_speedup = sum(r.speedup_vs_none for r in mode_results) / len(mode_results)
            print(f"{mode:>10}: 有效率 {positive}/{len(mode_results)} ({100*positive/len(mode_results):.0f}%), 平均加速 {avg_speedup:.2f}x")

    # lazy vs eager对比
    lazy_vs_eager = [r for r in results if r.compression_mode == 'lazy' and r.speedup_vs_eager is not None]
    if lazy_vs_eager:
        positive = sum(1 for r in lazy_vs_eager if r.speedup_vs_eager > 1.0)
        avg = sum(r.speedup_vs_eager for r in lazy_vs_eager) / len(lazy_vs_eager)
        print(f"\nlazy vs eager: 更好率 {positive}/{len(lazy_vs_eager)} ({100*positive/len(lazy_vs_eager):.0f}%), 平均提升 {avg:.2f}x")


def save_results(results: List[TestResult], filename: str):
    """保存结果"""
    data = [asdict(r) for r in results]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filename}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive Lazy Compression Test')
    parser.add_argument('--model', default='/data/huggingface/llava-1.5-7b-hf')
    parser.add_argument('--compressor', default=str(CKPT_DIR / "llava_mlp.pth"))
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--input-lens', type=int, nargs='+', default=[256, 512, 1024])
    parser.add_argument('--output-len', type=int, default=64)
    parser.add_argument('--kvpress-methods', type=str, nargs='+', default=['streaming_llm'])
    parser.add_argument('--compression-factor', type=int, default=5)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    print("#" * 90)
    print(" Comprehensive Lazy Compression Test")
    print("#" * 90)
    print(f"\nModel: {args.model}")
    print(f"Compressor: {args.compressor}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Input lengths: {args.input_lens}")
    print(f"Output length: {args.output_len}")
    print(f"kvpress methods: {args.kvpress_methods}")
    print(f"Compression factor: {args.compression_factor}")

    results = run_comprehensive_test(
        model_path=args.model,
        compressor_path=args.compressor,
        batch_sizes=args.batch_sizes,
        input_lens=args.input_lens,
        output_len=args.output_len,
        kvpress_methods=args.kvpress_methods,
        compression_factor=args.compression_factor,
    )

    analyze_results(results)

    if args.output:
        save_results(results, args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(results, f"lazy_compression_test_{timestamp}.json")


if __name__ == '__main__':
    main()
