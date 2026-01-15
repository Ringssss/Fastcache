#!/usr/bin/env python3
"""
懒压缩策略测试
==============

策略说明:
1. 懒压缩: 不在prefill后立即压缩，而是等内存压力触发
2. 批量压缩: 一次压缩多个序列，减少kernel launch开销
3. 阈值触发: 当空闲blocks低于阈值时才触发压缩

"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys

import torch
import gc
import time
from typing import List
import numpy as np


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def generate_prompts(num: int, target_tokens: int = 600) -> List[str]:
    base = "USER: Please explain "
    topics = ["AI", "ML", "DL", "NLP", "CV", "RL", "robotics", "quantum", "blockchain", "cloud"]
    expansion = " in great detail covering history, applications, challenges, and future. "

    prompts = []
    repeat = target_tokens // 25
    for i in range(num):
        topic = topics[i % len(topics)]
        prompt = base + topic + expansion * repeat + " ASSISTANT:"
        prompts.append(prompt)
    return prompts


class LazyCompressionEngine:
    """
    懒压缩引擎 - 包装LlavaLLM，实现懒压缩策略
    """

    def __init__(self, llm, compression_threshold: float = 0.3):
        """
        Args:
            llm: LlavaLLM实例
            compression_threshold: 当空闲blocks低于总blocks的这个比例时触发压缩
        """
        self.llm = llm
        self.compression_threshold = compression_threshold

        # 获取block信息
        self.total_blocks = len(llm.scheduler.block_manager.blocks)
        self.threshold_blocks = int(self.total_blocks * compression_threshold)

        # 追踪未压缩的序列
        self.uncompressed_seqs = set()

        print(f"懒压缩引擎初始化:")
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

        # 批量压缩
        comp_time, comp_ratio = self.llm.model_runner.compress_kv_cache_batch(seqs_to_compress)

        # 释放blocks
        for seq in seqs_to_compress:
            self.llm._free_compressed_blocks([seq])
            self.uncompressed_seqs.discard(seq.seq_id)

        free_after = len(self.llm.scheduler.block_manager.free_block_ids)
        print(f"[懒压缩] 批量压缩 {len(seqs_to_compress)} 个序列, "
              f"耗时: {comp_time*1000:.1f}ms, 压缩比: {comp_ratio:.2f}x, "
              f"空闲blocks: {free_after}")


def test_lazy_vs_eager(model_path: str, compressor_path: str, num_requests: int = 30):
    """对比懒压缩 vs 即时压缩"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    prompts = generate_prompts(num_requests, 600)

    # 测试1: 即时压缩（当前策略）
    print("\n" + "=" * 70)
    print(" 测试1: 即时压缩 (每个prefill后立即压缩)")
    print("=" * 70)

    clear_gpu()

    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        enable_compression=True,
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=4096,
    )

    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=32))

    total_tokens = 0
    compression_calls = 0
    start = time.time()

    while not llm.is_finished():
        outputs, num_tokens = llm.step(apply_compression=True)
        total_tokens += abs(num_tokens)
        if num_tokens > 0:
            compression_calls += 1

    eager_time = time.time() - start
    eager_throughput = total_tokens / eager_time

    print(f"\n结果:")
    print(f"  总时间: {eager_time:.2f}s")
    print(f"  总tokens: {total_tokens}")
    print(f"  压缩调用次数: {compression_calls}")
    print(f"  吞吐: {eager_throughput:.1f} tok/s")

    del llm
    clear_gpu()

    # 测试2: 懒压缩
    print("\n" + "=" * 70)
    print(" 测试2: 懒压缩 (内存压力时才批量压缩)")
    print("=" * 70)

    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        enable_compression=True,
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=4096,
    )

    lazy_engine = LazyCompressionEngine(llm, compression_threshold=0.3)

    for prompt in prompts:
        lazy_engine.add_request(prompt, SamplingParams(max_tokens=32))

    total_tokens = 0
    start = time.time()

    while not lazy_engine.is_finished():
        outputs, num_tokens = lazy_engine.step()
        total_tokens += abs(num_tokens)

    lazy_time = time.time() - start
    lazy_throughput = total_tokens / lazy_time

    print(f"\n结果:")
    print(f"  总时间: {lazy_time:.2f}s")
    print(f"  总tokens: {total_tokens}")
    print(f"  吞吐: {lazy_throughput:.1f} tok/s")

    del llm, lazy_engine
    clear_gpu()

    # 测试3: 无压缩（baseline）
    print("\n" + "=" * 70)
    print(" 测试3: 无压缩 (baseline)")
    print("=" * 70)

    llm = LlavaLLM(
        model_path,
        enable_compression=False,
        enforce_eager=True,
        max_model_len=4096,
    )

    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=32))

    total_tokens = 0
    start = time.time()

    while not llm.is_finished():
        outputs, num_tokens = llm.step(apply_compression=False)
        total_tokens += abs(num_tokens)

    baseline_time = time.time() - start
    baseline_throughput = total_tokens / baseline_time

    print(f"\n结果:")
    print(f"  总时间: {baseline_time:.2f}s")
    print(f"  总tokens: {total_tokens}")
    print(f"  吞吐: {baseline_throughput:.1f} tok/s")

    del llm
    clear_gpu()

    # 对比
    print("\n" + "=" * 70)
    print(" 对比总结")
    print("=" * 70)
    print(f"\n{'策略':<20} {'吞吐(tok/s)':<15} {'vs baseline':<15}")
    print("-" * 50)
    print(f"{'无压缩(baseline)':<20} {baseline_throughput:<15.1f} {1.0:<15.2f}x")
    print(f"{'即时压缩':<20} {eager_throughput:<15.1f} {eager_throughput/baseline_throughput:<15.2f}x")
    print(f"{'懒压缩':<20} {lazy_throughput:<15.1f} {lazy_throughput/baseline_throughput:<15.2f}x")

    print(f"\n懒压缩 vs 即时压缩提升: {(lazy_throughput - eager_throughput) / eager_throughput * 100:.1f}%")


def test_high_concurrency(model_path: str, compressor_path: str):
    """测试高并发场景"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 高并发测试: 100个请求")
    print("=" * 70)

    prompts = generate_prompts(100, 600)

    # 懒压缩
    clear_gpu()

    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        enable_compression=True,
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=4096,
    )

    lazy_engine = LazyCompressionEngine(llm, compression_threshold=0.2)

    for prompt in prompts:
        lazy_engine.add_request(prompt, SamplingParams(max_tokens=32))

    total_tokens = 0
    start = time.time()
    step_count = 0

    while not lazy_engine.is_finished():
        outputs, num_tokens = lazy_engine.step()
        total_tokens += abs(num_tokens)
        step_count += 1

        if step_count % 50 == 0:
            free = len(llm.scheduler.block_manager.free_block_ids)
            running = len(llm.scheduler.running)
            waiting = len(llm.scheduler.waiting)
            elapsed = time.time() - start
            print(f"  Step {step_count}: running={running}, waiting={waiting}, "
                  f"free_blocks={free}, elapsed={elapsed:.1f}s")

    elapsed = time.time() - start
    throughput = total_tokens / elapsed

    print(f"\n高并发结果:")
    print(f"  处理请求: 100")
    print(f"  总tokens: {total_tokens}")
    print(f"  总时间: {elapsed:.2f}s")
    print(f"  吞吐: {throughput:.1f} tok/s")

    del llm, lazy_engine
    clear_gpu()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data/huggingface/llava-1.5-7b-hf')
    parser.add_argument('--compressor', default=str(CKPT_DIR / "llava_mlp.pth"))
    parser.add_argument('--num_requests', type=int, default=30)
    parser.add_argument('--test', choices=['compare', 'high_concurrency', 'all'], default='all')
    args = parser.parse_args()

    print("#" * 70)
    print(" 懒压缩策略测试")
    print("#" * 70)

    if args.test in ['all', 'compare']:
        test_lazy_vs_eager(args.model, args.compressor, args.num_requests)

    if args.test in ['all', 'high_concurrency']:
        test_high_concurrency(args.model, args.compressor)

    print("\n✓ 测试完成!")


if __name__ == '__main__':
    main()
