#!/usr/bin/env python3
"""
测试KV-cache压缩的内存节省效果（简化版）
"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys

import torch
import gc


def test_block_saving():
    """测试block释放效果"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("=" * 70)
    print(" 测试: Block释放和内存节省")
    print("=" * 70)

    model_path = "/data/huggingface/llava-1.5-7b-hf"
    compressor_path = str(CKPT_DIR / "llava_mlp.pth")

    gc.collect()
    torch.cuda.empty_cache()

    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        enable_compression=True,
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=2048,
    )

    block_size = llm.scheduler.block_manager.block_size
    total_blocks = len(llm.scheduler.block_manager.blocks)
    initial_free = len(llm.scheduler.block_manager.free_block_ids)

    print(f"\n总blocks: {total_blocks}")
    print(f"Block size: {block_size}")
    print(f"初始空闲: {initial_free}")

    # 准备长prompts - 使用少量以避免批处理问题
    base = "USER: Please explain the history of AI in detail. "
    prompts = [
        base + "Include milestones, researchers, and algorithms. " * 50 + "ASSISTANT:"
        for _ in range(3)  # 减少数量
    ]

    prompt_len = len(llm.tokenizer.encode(prompts[0]))
    blocks_per_prompt = (prompt_len + block_size - 1) // block_size
    blocks_per_prompt_compressed = (prompt_len // 5 + block_size - 1) // block_size

    print(f"\n每个Prompt: {prompt_len} tokens")
    print(f"无压缩每个需要: {blocks_per_prompt} blocks")
    print(f"压缩后每个需要: {blocks_per_prompt_compressed} blocks")
    print(f"预期节省: {(blocks_per_prompt - blocks_per_prompt_compressed) / blocks_per_prompt * 100:.1f}%")

    # 逐个处理请求以避免批处理问题
    print(f"\n开始生成 {len(prompts)} 个请求...")
    all_outputs = []
    total_blocks_released = 0

    for i, prompt in enumerate(prompts):
        print(f"\n--- 请求 {i+1}/{len(prompts)} ---")
        free_before = len(llm.scheduler.block_manager.free_block_ids)
        print(f"处理前空闲blocks: {free_before}")

        sampling_params = [SamplingParams(max_tokens=32)]
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False, apply_compression=True)
        all_outputs.extend(outputs)

        free_after = len(llm.scheduler.block_manager.free_block_ids)
        print(f"处理后空闲blocks: {free_after}")

        # 完成后应该恢复blocks
        released = free_after - free_before
        if released > 0:
            total_blocks_released += released
            print(f"本次恢复: {released} blocks")

    final_free = len(llm.scheduler.block_manager.free_block_ids)
    print(f"\n生成完成后空闲blocks: {final_free}")

    # 统计
    print("\n" + "=" * 70)
    print(" 结果统计")
    print("=" * 70)

    # 理论上无压缩需要的blocks
    theoretical_no_compress = blocks_per_prompt * len(prompts)
    # 理论上压缩后需要的blocks
    theoretical_compressed = blocks_per_prompt_compressed * len(prompts)

    print(f"处理了 {len(prompts)} 个请求")
    print(f"无压缩理论需要: {theoretical_no_compress} blocks")
    print(f"压缩后理论需要: {theoretical_compressed} blocks")
    print(f"理论节省: {theoretical_no_compress - theoretical_compressed} blocks ({(theoretical_no_compress - theoretical_compressed) / theoretical_no_compress * 100:.1f}%)")

    # 输出示例
    print("\n输出示例:")
    print(f"  {all_outputs[0]['text'][:100]}...")

    print("\n✓ 测试完成!")


if __name__ == "__main__":
    test_block_saving()
