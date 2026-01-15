#!/usr/bin/env python3
"""
完整吞吐量对比测试
==================

对比:
1. 无压缩 baseline
2. 即时压缩 (当前)
3. 懒压缩 (优化)

在高并发场景下测试真实吞吐量

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
    """懒压缩引擎"""

    def __init__(self, llm, compression_threshold: float = 0.2):
        self.llm = llm
        self.compression_threshold = compression_threshold
        self.total_blocks = len(llm.scheduler.block_manager.blocks)
        self.threshold_blocks = int(self.total_blocks * compression_threshold)
        self.uncompressed_seqs = set()
        self.compression_count = 0
        self.total_compression_time = 0

    def add_request(self, prompt, sampling_params):
        self.llm.add_request(prompt, sampling_params)

    def is_finished(self):
        return self.llm.is_finished()

    def step(self):
        free_blocks = len(self.llm.scheduler.block_manager.free_block_ids)

        if free_blocks < self.threshold_blocks and self.uncompressed_seqs:
            self._do_batch_compression()

        outputs, num_tokens = self.llm.step(apply_compression=False)

        if num_tokens > 0:
            for seq in self.llm.scheduler.running:
                if seq.seq_id not in self.uncompressed_seqs:
                    self.uncompressed_seqs.add(seq.seq_id)

        for seq_id, _ in outputs:
            self.uncompressed_seqs.discard(seq_id)

        return outputs, num_tokens

    def _do_batch_compression(self):
        seqs_to_compress = [
            seq for seq in self.llm.scheduler.running
            if seq.seq_id in self.uncompressed_seqs
        ]

        if not seqs_to_compress:
            return

        start = time.time()
        comp_time, comp_ratio = self.llm.model_runner.compress_kv_cache_batch(seqs_to_compress)
        self.total_compression_time += time.time() - start
        self.compression_count += 1

        for seq in seqs_to_compress:
            self.llm._free_compressed_blocks([seq])
            self.uncompressed_seqs.discard(seq.seq_id)


def run_test(model_path, compressor_path, num_requests, prompt_tokens, mode):
    """运行测试"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    clear_gpu()

    enable_compression = mode != "no_compress"
    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path if enable_compression else None,
        enable_compression=enable_compression,
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=4096,
    )

    prompts = generate_prompts(num_requests, prompt_tokens)
    sample_len = len(llm.tokenizer.encode(prompts[0]))

    if mode == "lazy":
        engine = LazyCompressionEngine(llm, compression_threshold=0.2)
        for prompt in prompts:
            engine.add_request(prompt, SamplingParams(max_tokens=64))
    else:
        engine = llm
        for prompt in prompts:
            llm.add_request(prompt, SamplingParams(max_tokens=64))

    # 运行
    total_output_tokens = 0
    total_prefill_tokens = 0
    start = time.time()

    while not engine.is_finished():
        if mode == "lazy":
            outputs, num_tokens = engine.step()
        else:
            outputs, num_tokens = llm.step(apply_compression=(mode == "eager"))

        if num_tokens > 0:
            total_prefill_tokens += num_tokens
        else:
            total_output_tokens += (-num_tokens)

    elapsed = time.time() - start
    total_tokens = total_prefill_tokens + total_output_tokens

    result = {
        'mode': mode,
        'num_requests': num_requests,
        'prompt_tokens': sample_len,
        'prefill_tokens': total_prefill_tokens,
        'output_tokens': total_output_tokens,
        'total_tokens': total_tokens,
        'time': elapsed,
        'throughput': total_tokens / elapsed,
        'output_throughput': total_output_tokens / elapsed,
    }

    if mode == "lazy" and hasattr(engine, 'compression_count'):
        result['compression_calls'] = engine.compression_count
        result['compression_time'] = engine.total_compression_time

    del llm
    if mode == "lazy":
        del engine
    clear_gpu()

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data/huggingface/llava-1.5-7b-hf')
    parser.add_argument('--compressor', default=str(CKPT_DIR / "llava_mlp.pth"))
    parser.add_argument('--num_requests', type=int, default=100)
    parser.add_argument('--prompt_tokens', type=int, default=600)
    args = parser.parse_args()

    print("#" * 70)
    print(" 完整吞吐量对比测试")
    print("#" * 70)
    print(f"请求数: {args.num_requests}")
    print(f"Prompt长度: ~{args.prompt_tokens} tokens")

    results = []

    modes = ["no_compress", "eager", "lazy"]
    mode_names = {
        "no_compress": "无压缩 (baseline)",
        "eager": "即时压缩 (当前)",
        "lazy": "懒压缩 (优化)"
    }

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f" 测试: {mode_names[mode]}")
        print('=' * 60)

        result = run_test(
            args.model, args.compressor,
            args.num_requests, args.prompt_tokens, mode
        )
        results.append(result)

        print(f"\n结果:")
        print(f"  Prefill tokens: {result['prefill_tokens']}")
        print(f"  Output tokens: {result['output_tokens']}")
        print(f"  总时间: {result['time']:.2f}s")
        print(f"  总吞吐: {result['throughput']:.1f} tok/s")
        print(f"  输出吞吐: {result['output_throughput']:.1f} tok/s")
        if 'compression_calls' in result:
            print(f"  压缩调用: {result['compression_calls']} 次")
            print(f"  压缩耗时: {result['compression_time']:.2f}s")

    # 对比总结
    baseline = results[0]['throughput']
    output_baseline = results[0]['output_throughput']

    print("\n" + "=" * 70)
    print(" 对比总结")
    print("=" * 70)

    print(f"\n{'策略':<25} {'总吞吐':<15} {'输出吞吐':<15} {'vs baseline':<12}")
    print("-" * 67)

    for r in results:
        name = mode_names[r['mode']]
        total_tp = r['throughput']
        output_tp = r['output_throughput']
        ratio = total_tp / baseline
        print(f"{name:<25} {total_tp:<15.1f} {output_tp:<15.1f} {ratio:.2f}x")

    print("\n" + "=" * 70)
    print(" 关键发现")
    print("=" * 70)

    eager_result = results[1]
    lazy_result = results[2]

    print(f"\n1. 懒压缩 vs 即时压缩:")
    improvement = (lazy_result['throughput'] - eager_result['throughput']) / eager_result['throughput'] * 100
    print(f"   吞吐提升: {improvement:.1f}%")

    print(f"\n2. 懒压缩 vs 无压缩:")
    overhead = (results[0]['throughput'] - lazy_result['throughput']) / results[0]['throughput'] * 100
    print(f"   开销: {overhead:.1f}%")

    print(f"\n3. 即时压缩 vs 无压缩:")
    overhead = (results[0]['throughput'] - eager_result['throughput']) / results[0]['throughput'] * 100
    print(f"   开销: {overhead:.1f}%")

    print("\n✓ 测试完成!")


if __name__ == '__main__':
    main()
