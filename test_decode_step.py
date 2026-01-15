#!/usr/bin/env python3
"""
精确测量单次decode step的时间
============================

问题诊断：为什么压缩后decode反而更慢？

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


def generate_prompts(num: int, target_tokens: int) -> List[str]:
    base = "USER: Please explain "
    topics = ["AI", "ML", "DL", "NLP", "CV", "RL", "robotics", "quantum", "blockchain", "cloud"]
    expansion = " in great detail covering history, applications, challenges, and future. "

    prompts = []
    repeat = target_tokens // 30
    for i in range(num):
        topic = topics[i % len(topics)]
        prompt = base + topic + expansion * repeat + " ASSISTANT:"
        prompts.append(prompt)
    return prompts


def measure_decode_step_time(
    model_path: str,
    compressor_path: str,
    batch_size: int = 128,
    prompt_tokens: int = 1000,
    num_decode_steps: int = 100
):
    """
    精确测量decode step时间

    流程：
    1. Prefill所有序列
    2. 如果启用压缩，执行压缩
    3. 精确测量每个decode step的时间
    """
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(f" Decode Step精确测量: BS={batch_size}, Prompt={prompt_tokens}")
    print("=" * 70)

    prompts = generate_prompts(batch_size, prompt_tokens)

    # =====================================================
    # 测试1：无压缩
    # =====================================================
    print("\n--- 无压缩 ---")
    clear_gpu()

    llm = LlavaLLM(
        model_path,
        enable_compression=False,
        enforce_eager=True,
        max_model_len=4096,
    )

    sample_len = len(llm.tokenizer.encode(prompts[0]))
    print(f"Prompt长度: {sample_len} tokens")

    # 添加请求（使用大的max_tokens确保不会提前停止）
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=num_decode_steps + 10, temperature=1.0))

    # Prefill阶段
    prefill_steps = 0
    while True:
        seqs, is_prefill = llm.scheduler.schedule()
        if not is_prefill:
            break
        llm.model_runner.run(seqs, is_prefill, apply_compression=False)
        llm.scheduler.postprocess(seqs, [1] * len(seqs))  # dummy tokens
        prefill_steps += 1

    print(f"Prefill完成 ({prefill_steps} steps)")

    # 获取当前context_lens
    running_seqs = list(llm.scheduler.running)
    if running_seqs:
        context_len = running_seqs[0].kv_cache_len if hasattr(running_seqs[0], 'kv_cache_len') else len(running_seqs[0])
        print(f"初始context_len: {context_len}")

    # 测量decode步骤
    decode_times = []
    torch.cuda.synchronize()

    for step in range(num_decode_steps):
        seqs, is_prefill = llm.scheduler.schedule()
        if is_prefill or len(seqs) == 0:
            break

        torch.cuda.synchronize()
        start = time.perf_counter()

        token_ids = llm.model_runner.run(seqs, is_prefill=False, apply_compression=False)
        llm.scheduler.postprocess(seqs, token_ids)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        decode_times.append(elapsed)

    avg_decode_time = sum(decode_times) / len(decode_times) if decode_times else 0
    no_compress_throughput = batch_size / avg_decode_time if avg_decode_time > 0 else 0

    print(f"Decode steps: {len(decode_times)}")
    print(f"平均decode时间: {avg_decode_time*1000:.2f} ms")
    print(f"Decode吞吐: {no_compress_throughput:.0f} tok/s")

    # 最终context_len
    running_seqs = list(llm.scheduler.running)
    if running_seqs:
        final_context_len = running_seqs[0].kv_cache_len if hasattr(running_seqs[0], 'kv_cache_len') else len(running_seqs[0])
        print(f"最终context_len: {final_context_len}")

    del llm
    clear_gpu()

    # =====================================================
    # 测试2：有压缩
    # =====================================================
    print("\n--- 有压缩 ---")

    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        enable_compression=True,
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=4096,
    )

    # 添加请求
    for prompt in prompts:
        llm.add_request(prompt, SamplingParams(max_tokens=num_decode_steps + 10, temperature=1.0))

    # Prefill + 压缩阶段
    prefill_steps = 0
    while True:
        seqs, is_prefill = llm.scheduler.schedule()
        if not is_prefill:
            break
        # Prefill时应用压缩
        token_ids = llm.model_runner.run(seqs, is_prefill, apply_compression=True)

        # 压缩后释放blocks
        for seq in seqs:
            compressed_len = llm.model_runner._compressed_lens.get(seq.seq_id)
            if compressed_len is not None:
                seq.kv_cache_len = compressed_len
                # 释放多余blocks
                blocks_to_keep = llm.model_runner.get_compressed_block_count(seq.seq_id)
                if blocks_to_keep > 0 and len(seq.block_table) > blocks_to_keep:
                    for block_id in seq.block_table[blocks_to_keep:]:
                        if block_id in llm.scheduler.block_manager.used_block_ids:
                            block = llm.scheduler.block_manager.blocks[block_id]
                            block.ref_count -= 1
                            if block.ref_count == 0:
                                llm.scheduler.block_manager._deallocate_block(block_id)
                    seq.block_table = seq.block_table[:blocks_to_keep]

        llm.scheduler.postprocess(seqs, [1] * len(seqs))  # dummy tokens
        prefill_steps += 1

    print(f"Prefill+压缩完成 ({prefill_steps} steps)")

    # 获取压缩后的context_lens
    running_seqs = list(llm.scheduler.running)
    if running_seqs:
        context_len = running_seqs[0].kv_cache_len if hasattr(running_seqs[0], 'kv_cache_len') else len(running_seqs[0])
        print(f"压缩后初始context_len: {context_len}")

    # 测量decode步骤
    decode_times = []
    torch.cuda.synchronize()

    for step in range(num_decode_steps):
        seqs, is_prefill = llm.scheduler.schedule()
        if is_prefill or len(seqs) == 0:
            break

        torch.cuda.synchronize()
        start = time.perf_counter()

        token_ids = llm.model_runner.run(seqs, is_prefill=False, apply_compression=False)
        llm.scheduler.postprocess(seqs, token_ids)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        decode_times.append(elapsed)

    avg_decode_time_compress = sum(decode_times) / len(decode_times) if decode_times else 0
    compress_throughput = batch_size / avg_decode_time_compress if avg_decode_time_compress > 0 else 0

    print(f"Decode steps: {len(decode_times)}")
    print(f"平均decode时间: {avg_decode_time_compress*1000:.2f} ms")
    print(f"Decode吞吐: {compress_throughput:.0f} tok/s")

    # 最终context_len
    running_seqs = list(llm.scheduler.running)
    if running_seqs:
        final_context_len = running_seqs[0].kv_cache_len if hasattr(running_seqs[0], 'kv_cache_len') else len(running_seqs[0])
        print(f"最终context_len: {final_context_len}")

    del llm
    clear_gpu()

    # =====================================================
    # 对比
    # =====================================================
    print("\n" + "=" * 70)
    print(" Decode Step对比")
    print("=" * 70)

    speedup = compress_throughput / no_compress_throughput if no_compress_throughput > 0 else 0
    improvement = (speedup - 1) * 100

    print(f"\n无压缩Decode吞吐: {no_compress_throughput:.0f} tok/s")
    print(f"有压缩Decode吞吐: {compress_throughput:.0f} tok/s")
    print(f"")
    print(f"🎯 Decode吞吐提升: {improvement:+.1f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data/huggingface/llava-1.5-7b-hf')
    parser.add_argument('--compressor', default=str(CKPT_DIR / "llava_mlp.pth"))
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--prompt', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=50)
    args = parser.parse_args()

    print("#" * 70)
    print(" Decode Step精确测量")
    print("#" * 70)

    measure_decode_step_time(
        args.model,
        args.compressor,
        batch_size=args.bs,
        prompt_tokens=args.prompt,
        num_decode_steps=args.steps
    )


if __name__ == '__main__':
    main()
