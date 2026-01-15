#!/usr/bin/env python3
"""
详细吞吐量测试：分析压缩开销和并发prefill
==========================================

测试内容：
1. 当前压缩策略的开销分析
2. 并发prefill是否工作
3. 不同batch size的吞吐量
4. 懒压缩策略的潜在收益

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
    """生成测试prompts"""
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


def test_batch_prefill(model_path: str, compressor_path: str):
    """测试并发prefill是否真正工作"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 测试1: 并发Prefill验证")
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

    prompts = generate_prompts(10, 600)
    sample_len = len(llm.tokenizer.encode(prompts[0]))
    print(f"每个prompt: {sample_len} tokens")

    # 添加多个请求
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=16))

    print(f"\n添加了 {len(prompts)} 个请求")
    print(f"waiting队列: {len(llm.scheduler.waiting)}")

    # 执行一步，看调度了多少
    step = 0
    total_prefill_tokens = 0
    prefill_times = []

    while len(llm.scheduler.waiting) > 0:
        step += 1
        start = time.time()

        seqs, is_prefill = llm.scheduler.schedule()
        scheduled_count = len(seqs)
        scheduled_tokens = sum(len(seq) for seq in seqs)

        # 如果是prefill，测量时间
        if is_prefill:
            prefill_start = time.time()
            token_ids = llm.model_runner.run(seqs, is_prefill, apply_compression=False)
            prefill_time = time.time() - prefill_start
            prefill_times.append(prefill_time)
            total_prefill_tokens += scheduled_tokens

            print(f"  Step {step}: 调度了 {scheduled_count} 个序列, "
                  f"{scheduled_tokens} tokens, prefill时间: {prefill_time*1000:.1f}ms")

        llm.scheduler.postprocess(seqs, token_ids if is_prefill else [0]*len(seqs))

    print(f"\n并发Prefill分析:")
    print(f"  总step数: {step}")
    print(f"  总prefill tokens: {total_prefill_tokens}")
    if prefill_times:
        print(f"  平均prefill时间: {np.mean(prefill_times)*1000:.1f}ms")
        print(f"  理论吞吐: {total_prefill_tokens / sum(prefill_times):.0f} tokens/s")

    del llm
    clear_gpu()


def test_compression_overhead(model_path: str, compressor_path: str):
    """详细测试压缩开销"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 测试2: 压缩开销详细分析")
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

    prompts = generate_prompts(20, 800)
    sample_len = len(llm.tokenizer.encode(prompts[0]))
    print(f"每个prompt: {sample_len} tokens")

    # 测试无压缩
    print("\n--- 无压缩 ---")
    for prompt in prompts[:10]:
        llm.add_request(prompt, SamplingParams(max_tokens=32))

    no_compress_prefill = 0
    no_compress_decode = 0
    start = time.time()

    while not llm.is_finished():
        t0 = time.time()
        outputs, num_tokens = llm.step(apply_compression=False)
        elapsed = time.time() - t0
        if num_tokens > 0:
            no_compress_prefill += num_tokens
        else:
            no_compress_decode += (-num_tokens)

    no_compress_time = time.time() - start
    print(f"总时间: {no_compress_time:.2f}s")
    print(f"Prefill tokens: {no_compress_prefill}, Decode tokens: {no_compress_decode}")
    print(f"吞吐: {(no_compress_prefill + no_compress_decode) / no_compress_time:.1f} tok/s")

    # 清理
    llm.model_runner.clear_compressed_lens()

    # 测试有压缩
    print("\n--- 有压缩(每个序列立即压缩) ---")
    for prompt in prompts[10:20]:
        llm.add_request(prompt, SamplingParams(max_tokens=32))

    compress_prefill = 0
    compress_decode = 0
    compress_times = []
    start = time.time()

    while not llm.is_finished():
        t0 = time.time()
        outputs, num_tokens = llm.step(apply_compression=True)
        elapsed = time.time() - t0
        if num_tokens > 0:
            compress_prefill += num_tokens
            compress_times.append(elapsed)
        else:
            compress_decode += (-num_tokens)

    compress_time = time.time() - start
    print(f"总时间: {compress_time:.2f}s")
    print(f"Prefill tokens: {compress_prefill}, Decode tokens: {compress_decode}")
    print(f"吞吐: {(compress_prefill + compress_decode) / compress_time:.1f} tok/s")
    print(f"平均Prefill+压缩时间: {np.mean(compress_times)*1000:.1f}ms")

    # 分析
    print("\n--- 分析 ---")
    overhead = (compress_time - no_compress_time) / no_compress_time * 100
    print(f"压缩开销: {overhead:+.1f}%")
    print(f"无压缩吞吐: {(no_compress_prefill + no_compress_decode) / no_compress_time:.1f} tok/s")
    print(f"有压缩吞吐: {(compress_prefill + compress_decode) / compress_time:.1f} tok/s")

    del llm
    clear_gpu()


def test_batch_size_throughput(model_path: str, compressor_path: str):
    """测试不同batch size下的吞吐量"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 测试3: 不同Batch Size的吞吐量")
    print("=" * 70)

    batch_sizes = [1, 4, 8, 16, 32, 64]
    results = []

    for bs in batch_sizes:
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

        prompts = generate_prompts(bs, 600)

        # 添加所有请求
        for prompt in prompts:
            llm.add_request(prompt, SamplingParams(max_tokens=64))

        # 运行
        total_tokens = 0
        start = time.time()

        while not llm.is_finished():
            outputs, num_tokens = llm.step(apply_compression=True)
            total_tokens += abs(num_tokens)

        elapsed = time.time() - start
        throughput = total_tokens / elapsed

        results.append({
            'batch_size': bs,
            'tokens': total_tokens,
            'time': elapsed,
            'throughput': throughput
        })

        print(f"Batch={bs:2d}: {total_tokens} tokens, {elapsed:.2f}s, {throughput:.1f} tok/s")

        del llm

    # 分析
    print("\n--- 分析 ---")
    baseline = results[0]['throughput']
    for r in results:
        speedup = r['throughput'] / baseline
        print(f"Batch={r['batch_size']:2d}: {r['throughput']:.1f} tok/s, {speedup:.2f}x vs bs=1")

    clear_gpu()


def test_concurrent_vs_sequential(model_path: str, compressor_path: str):
    """对比真正并发 vs 串行处理的差异"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 测试4: 并发处理 vs 串行处理")
    print("=" * 70)

    num_requests = 20

    # 测试1: 并发处理 - 先全部添加，然后一起运行
    print("\n--- 并发处理 (全部添加后运行) ---")
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

    prompts = generate_prompts(num_requests, 600)

    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=32))

    total_tokens = 0
    start = time.time()
    prefill_count = 0
    decode_count = 0

    while not llm.is_finished():
        outputs, num_tokens = llm.step(apply_compression=True)
        if num_tokens > 0:
            prefill_count += 1
            total_tokens += num_tokens
        else:
            decode_count += 1
            total_tokens += (-num_tokens)

    concurrent_time = time.time() - start
    concurrent_throughput = total_tokens / concurrent_time

    print(f"总时间: {concurrent_time:.2f}s")
    print(f"Prefill steps: {prefill_count}, Decode steps: {decode_count}")
    print(f"总tokens: {total_tokens}")
    print(f"吞吐: {concurrent_throughput:.1f} tok/s")

    del llm

    # 测试2: 串行处理 - 每个请求完全处理完再处理下一个
    print("\n--- 串行处理 (逐个完成) ---")
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

    total_tokens = 0
    start = time.time()

    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=32))

        # 完全处理这个请求
        while not llm.is_finished():
            outputs, num_tokens = llm.step(apply_compression=True)
            total_tokens += abs(num_tokens)

    sequential_time = time.time() - start
    sequential_throughput = total_tokens / sequential_time

    print(f"总时间: {sequential_time:.2f}s")
    print(f"总tokens: {total_tokens}")
    print(f"吞吐: {sequential_throughput:.1f} tok/s")

    # 对比
    print("\n--- 对比 ---")
    speedup = concurrent_throughput / sequential_throughput
    print(f"并发吞吐: {concurrent_throughput:.1f} tok/s")
    print(f"串行吞吐: {sequential_throughput:.1f} tok/s")
    print(f"并发加速比: {speedup:.2f}x")

    del llm
    clear_gpu()


def analyze_compression_pattern(model_path: str, compressor_path: str):
    """分析当前压缩模式的问题"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 测试5: 压缩模式分析 - 找出瓶颈")
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

    prompts = generate_prompts(10, 800)
    sample_len = len(llm.tokenizer.encode(prompts[0]))

    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=32))

    print(f"\n每个prompt: {sample_len} tokens")
    print(f"总请求数: {len(prompts)}")

    # 详细跟踪每一步
    step_times = []
    step_types = []
    step_counts = []
    compression_calls = 0

    print("\n执行过程:")
    while not llm.is_finished():
        t0 = time.time()

        # 先看调度情况
        seqs, is_prefill = llm.scheduler.schedule()
        schedule_count = len(seqs)

        # 复原（因为schedule会修改状态）
        # 实际用step来执行
        for seq in seqs:
            if is_prefill:
                llm.scheduler.waiting.append(seq)
                llm.scheduler.running.remove(seq)

        outputs, num_tokens = llm.step(apply_compression=True)
        elapsed = time.time() - t0

        step_type = "prefill" if num_tokens > 0 else "decode"
        step_times.append(elapsed)
        step_types.append(step_type)
        step_counts.append(abs(num_tokens) if step_type == "prefill" else schedule_count)

        if step_type == "prefill":
            compression_calls += 1

    # 统计
    prefill_times = [t for t, tp in zip(step_times, step_types) if tp == "prefill"]
    decode_times = [t for t, tp in zip(step_times, step_types) if tp == "decode"]

    print(f"\n统计分析:")
    print(f"  总steps: {len(step_times)}")
    print(f"  Prefill steps: {len(prefill_times)}")
    print(f"  Decode steps: {len(decode_times)}")
    print(f"  压缩调用次数: {compression_calls}")
    print(f"  ")
    print(f"  Prefill平均时间: {np.mean(prefill_times)*1000:.1f}ms")
    print(f"  Decode平均时间: {np.mean(decode_times)*1000:.1f}ms")
    print(f"  ")
    print(f"  Prefill总时间: {sum(prefill_times):.2f}s ({sum(prefill_times)/sum(step_times)*100:.1f}%)")
    print(f"  Decode总时间: {sum(decode_times):.2f}s ({sum(decode_times)/sum(step_times)*100:.1f}%)")

    print(f"\n问题分析:")
    print(f"  每次prefill都调用压缩 -> 压缩被调用了 {compression_calls} 次")
    print(f"  如果batch prefill {len(prompts)} 个序列，压缩只需调用 1 次")
    print(f"  潜在节省: {(compression_calls - 1)} 次压缩调用开销")

    del llm
    clear_gpu()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data/huggingface/llava-1.5-7b-hf')
    parser.add_argument('--compressor', default=str(CKPT_DIR / "llava_mlp.pth"))
    parser.add_argument('--test', choices=['all', 'batch', 'overhead', 'batchsize', 'concurrent', 'pattern'],
                        default='all')
    args = parser.parse_args()

    print("#" * 70)
    print(" nano-vllm 详细吞吐量分析")
    print("#" * 70)

    if args.test in ['all', 'batch']:
        test_batch_prefill(args.model, args.compressor)

    if args.test in ['all', 'overhead']:
        test_compression_overhead(args.model, args.compressor)

    if args.test in ['all', 'batchsize']:
        test_batch_size_throughput(args.model, args.compressor)

    if args.test in ['all', 'concurrent']:
        test_concurrent_vs_sequential(args.model, args.compressor)

    if args.test in ['all', 'pattern']:
        analyze_compression_pattern(args.model, args.compressor)

    print("\n" + "=" * 70)
    print(" 测试完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
