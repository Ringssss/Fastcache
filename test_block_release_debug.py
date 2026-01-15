#!/usr/bin/env python3
"""
调试KV-cache压缩后的内存释放
"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys

import torch
import gc

def test_block_release_debug():
    """详细调试block释放"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("=" * 70)
    print(" 调试KV-cache压缩后的Block释放")
    print("=" * 70)

    model_path = "/data/huggingface/llava-1.5-7b-hf"
    compressor_path = str(CKPT_DIR / "llava_mlp.pth")

    gc.collect()
    torch.cuda.empty_cache()

    # 使用同步压缩
    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path,  # 添加压缩器路径
        enable_compression=True,
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=2048,
    )

    block_size = llm.scheduler.block_manager.block_size
    print(f"\nBlock size: {block_size}")
    print(f"压缩因子: {llm.compression_factor}")
    print(f"压缩器状态: {llm.model_runner.compressor is not None}")
    print(f"批量压缩器状态: {getattr(llm.model_runner, 'use_batched_compressor', False)}")

    initial_free = len(llm.scheduler.block_manager.free_block_ids)
    print(f"初始空闲blocks: {initial_free}")

    # 单个长prompt - 需要足够长以使用多个blocks (>256 tokens)
    # 构造一个很长的prompt，目标是使用5-6个blocks
    base = "USER: Please provide a very detailed and comprehensive explanation of the history of artificial intelligence. "
    # 重复以达到足够长度 - 目标是1200+ tokens
    prompt = base + "Include information about all major milestones, key researchers, breakthrough algorithms, and significant applications. Discuss the evolution from symbolic AI to connectionist approaches, and explain how deep learning revolutionized the field. " * 30 + "ASSISTANT:"

    sampling_params = SamplingParams(max_tokens=64)

    prompt_len = len(llm.tokenizer.encode(prompt))
    print(f"\nPrompt长度: {prompt_len} tokens")
    expected_blocks = (prompt_len + block_size - 1) // block_size
    expected_blocks_after = (prompt_len // 5 + block_size - 1) // block_size
    print(f"预期使用blocks: {expected_blocks}")
    print(f"压缩后预期blocks: {expected_blocks_after} (假设5x压缩)")
    print(f"预期释放blocks: {expected_blocks - expected_blocks_after}")

    # 添加请求
    llm.add_request(prompt, sampling_params)

    print("\n开始推理...")
    step_count = 0
    while not llm.is_finished():
        outputs, num_tokens = llm.step(apply_compression=True)
        step_count += 1

        # 检查压缩状态
        if step_count == 1:  # prefill后
            print(f"\n[Step {step_count}] Prefill完成")
            print(f"  _compressed_lens: {llm.model_runner._compressed_lens}")

            # 检查每个running序列
            for seq in llm.scheduler.running:
                seq_id = seq.seq_id
                block_count = len(seq.block_table)
                compressed_block_count = llm.model_runner.get_compressed_block_count(seq_id)
                compressed_len = llm.model_runner._compressed_lens.get(seq_id, -1)

                print(f"  序列{seq_id}:")
                print(f"    当前block数: {block_count}")
                print(f"    压缩后block数: {compressed_block_count}")
                print(f"    压缩后长度: {compressed_len}")
                print(f"    block_table: {seq.block_table[:5]}...")

        if step_count <= 3:
            free_blocks = len(llm.scheduler.block_manager.free_block_ids)
            print(f"[Step {step_count}] 空闲blocks: {free_blocks}")

        if outputs:
            for seq_id, tokens in outputs:
                print(f"\n完成序列{seq_id}: {len(tokens)} tokens")

    final_free = len(llm.scheduler.block_manager.free_block_ids)
    print(f"\n最终空闲blocks: {final_free}")
    print(f"变化: {final_free - initial_free}")


if __name__ == "__main__":
    test_block_release_debug()
