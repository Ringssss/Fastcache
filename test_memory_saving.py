#!/usr/bin/env python3
"""
测试KV-cache压缩的内存节省效果
==============================

验证：
1. 压缩后blocks被正确释放
2. 释放的blocks能被后续请求重用
3. 内存节省使得可以处理更多并发请求

"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys

import torch
import gc


def get_gpu_memory():
    """获取GPU内存使用情况"""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return allocated, reserved


def test_block_reuse():
    """测试释放的blocks能被重用"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("=" * 70)
    print(" 测试: 释放的Blocks能被重用")
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
    print(f"\n总blocks: {total_blocks}, Block size: {block_size}")

    # 生成长prompt（会使用多个blocks）
    base = "USER: Please explain the history of AI in great detail. "
    long_prompt = base + "Include milestones, researchers, and algorithms. " * 50 + "ASSISTANT:"
    short_prompt = "USER: What is 2+2? ASSISTANT:"

    prompt_len = len(llm.tokenizer.encode(long_prompt))
    expected_blocks = (prompt_len + block_size - 1) // block_size
    expected_after_compress = (prompt_len // 5 + block_size - 1) // block_size
    print(f"长Prompt: {prompt_len} tokens, 需要 {expected_blocks} blocks")
    print(f"压缩后预期: {expected_after_compress} blocks")

    # 场景1: 不使用压缩，看能处理多少并发请求
    print("\n--- 场景1: 无压缩 ---")
    llm_no_compress = LlavaLLM(
        model_path,
        enable_compression=False,
        enforce_eager=True,
        max_model_len=2048,
    )

    initial_free = len(llm_no_compress.scheduler.block_manager.free_block_ids)
    print(f"初始空闲blocks: {initial_free}")

    # 添加多个长请求直到blocks用完
    num_added = 0
    while len(llm_no_compress.scheduler.block_manager.free_block_ids) >= expected_blocks:
        llm_no_compress.add_request(long_prompt, SamplingParams(max_tokens=32))
        num_added += 1
        if num_added >= 20:  # 安全限制
            break

    free_after_no_compress = len(llm_no_compress.scheduler.block_manager.free_block_ids)
    print(f"无压缩模式添加了 {num_added} 个请求")
    print(f"剩余空闲blocks: {free_after_no_compress}")

    del llm_no_compress
    gc.collect()
    torch.cuda.empty_cache()

    # 场景2: 使用压缩，看能处理多少并发请求
    print("\n--- 场景2: 有压缩 ---")
    llm_compress = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        enable_compression=True,
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=2048,
    )

    initial_free_compress = len(llm_compress.scheduler.block_manager.free_block_ids)
    print(f"初始空闲blocks: {initial_free_compress}")

    # 添加一个长请求并运行prefill（触发压缩和block释放）
    llm_compress.add_request(long_prompt, SamplingParams(max_tokens=32))

    # 运行一步（prefill + 压缩）
    llm_compress.step(apply_compression=True)

    free_after_compress = len(llm_compress.scheduler.block_manager.free_block_ids)
    print(f"第一个请求压缩后空闲blocks: {free_after_compress}")

    # 现在可以添加更多请求，因为释放了blocks
    additional_added = 0
    while len(llm_compress.scheduler.block_manager.free_block_ids) >= expected_after_compress:
        llm_compress.add_request(long_prompt, SamplingParams(max_tokens=32))
        additional_added += 1
        # 运行一步（prefill + 压缩）
        llm_compress.step(apply_compression=True)
        if additional_added >= 20:  # 安全限制
            break

    total_added_compress = 1 + additional_added
    final_free_compress = len(llm_compress.scheduler.block_manager.free_block_ids)
    print(f"压缩模式添加了 {total_added_compress} 个请求")
    print(f"剩余空闲blocks: {final_free_compress}")

    # 结果对比
    print("\n" + "=" * 70)
    print(" 结果对比")
    print("=" * 70)
    print(f"无压缩: 可处理 {num_added} 个并发长请求")
    print(f"有压缩: 可处理 {total_added_compress} 个并发长请求")

    if total_added_compress > num_added:
        improvement = (total_added_compress - num_added) / num_added * 100
        print(f"\n✓ 压缩使并发能力提升 {improvement:.1f}%！")
    else:
        print("\n⚠ 压缩未能显著提升并发能力")

    print("\n测试完成!")


def test_memory_usage_comparison():
    """对比有无压缩的实际内存使用"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 测试: 实际GPU内存使用对比")
    print("=" * 70)

    model_path = "/data/huggingface/llava-1.5-7b-hf"
    compressor_path = str(CKPT_DIR / "llava_mlp.pth")

    # 准备prompts
    base = "USER: Please explain machine learning in detail. "
    prompts = [
        base + "Cover supervised, unsupervised, and reinforcement learning. " * 40 + "ASSISTANT:"
        for _ in range(4)
    ]

    # 无压缩
    print("\n--- 无压缩 ---")
    gc.collect()
    torch.cuda.empty_cache()

    llm_no = LlavaLLM(
        model_path,
        enable_compression=False,
        enforce_eager=True,
        max_model_len=2048,
    )

    mem_before_no, _ = get_gpu_memory()
    outputs_no = llm_no.generate(prompts, [SamplingParams(max_tokens=64)] * len(prompts), use_tqdm=False)
    mem_after_no, _ = get_gpu_memory()
    blocks_used_no = len(llm_no.scheduler.block_manager.blocks) - len(llm_no.scheduler.block_manager.free_block_ids)

    print(f"生成前内存: {mem_before_no:.2f} GB")
    print(f"生成后内存: {mem_after_no:.2f} GB")
    print(f"使用blocks: {blocks_used_no}")

    del llm_no
    gc.collect()
    torch.cuda.empty_cache()

    # 有压缩
    print("\n--- 有压缩 ---")

    llm_yes = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        enable_compression=True,
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=2048,
    )

    mem_before_yes, _ = get_gpu_memory()
    outputs_yes = llm_yes.generate(prompts, [SamplingParams(max_tokens=64)] * len(prompts), use_tqdm=False, apply_compression=True)
    mem_after_yes, _ = get_gpu_memory()
    blocks_used_yes = len(llm_yes.scheduler.block_manager.blocks) - len(llm_yes.scheduler.block_manager.free_block_ids)

    print(f"生成前内存: {mem_before_yes:.2f} GB")
    print(f"生成后内存: {mem_after_yes:.2f} GB")
    print(f"使用blocks: {blocks_used_yes}")

    # 对比
    print("\n" + "=" * 70)
    print(" 内存对比")
    print("=" * 70)
    print(f"无压缩blocks使用: {blocks_used_no}")
    print(f"有压缩blocks使用: {blocks_used_yes}")

    if blocks_used_yes < blocks_used_no:
        saving = (blocks_used_no - blocks_used_yes) / blocks_used_no * 100
        print(f"\n✓ Block使用减少 {saving:.1f}%！")
    else:
        print("\n⚠ Block使用未减少")


if __name__ == "__main__":
    test_block_reuse()
    test_memory_usage_comparison()
