#!/usr/bin/env python3
"""
测试KV-cache压缩后的内存释放
=============================

验证压缩后多余的blocks是否被正确释放

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


def test_memory_release():
    """测试压缩后的内存释放"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("=" * 70)
    print(" 测试KV-cache压缩后的内存释放")
    print("=" * 70)

    model_path = "/data/huggingface/llava-1.5-7b-hf"

    # 清理GPU
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    alloc_before, reserved_before = get_gpu_memory()
    print(f"\n初始GPU内存: 已分配={alloc_before:.2f}GB, 已保留={reserved_before:.2f}GB")

    # 创建LLM实例
    print("\n加载模型...")
    llm = LlavaLLM(
        model_path,
        enable_compression=True,
        async_compression=False,  # 使用同步压缩，方便观察
        compression_factor=5,
        enforce_eager=True,
        max_model_len=2048,
    )

    alloc_after_load, reserved_after_load = get_gpu_memory()
    print(f"模型加载后: 已分配={alloc_after_load:.2f}GB, 已保留={reserved_after_load:.2f}GB")

    # 获取初始空闲blocks数量
    initial_free_blocks = len(llm.scheduler.block_manager.free_block_ids)
    print(f"初始空闲blocks: {initial_free_blocks}")

    # 生成长文本以使用更多blocks
    prompts = [
        "USER: Please provide a very detailed and comprehensive explanation of the history of artificial intelligence, covering all major milestones from 1950s to present. Include information about key researchers, breakthrough algorithms, and significant applications. ASSISTANT:",
    ]

    sampling_params = [SamplingParams(max_tokens=256)]

    print("\n" + "=" * 70)
    print(" 测试1: 无压缩")
    print("=" * 70)

    # 先测试无压缩
    llm.enable_compression = False
    outputs_no_compress = llm.generate(prompts, sampling_params, use_tqdm=False, apply_compression=False)

    alloc_no_compress, reserved_no_compress = get_gpu_memory()
    free_blocks_no_compress = len(llm.scheduler.block_manager.free_block_ids)
    print(f"无压缩后: 已分配={alloc_no_compress:.2f}GB, 空闲blocks={free_blocks_no_compress}")

    # 重置
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(" 测试2: 有压缩（观察block释放）")
    print("=" * 70)

    # 创建新实例测试压缩
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    llm2 = LlavaLLM(
        model_path,
        enable_compression=True,
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=2048,
    )

    initial_free_blocks_2 = len(llm2.scheduler.block_manager.free_block_ids)
    print(f"初始空闲blocks: {initial_free_blocks_2}")

    # 测试压缩
    print("\n开始生成（启用压缩）...")
    outputs_compress = llm2.generate(prompts, sampling_params, use_tqdm=False, apply_compression=True)

    alloc_compress, reserved_compress = get_gpu_memory()
    free_blocks_compress = len(llm2.scheduler.block_manager.free_block_ids)
    print(f"\n压缩后: 已分配={alloc_compress:.2f}GB, 空闲blocks={free_blocks_compress}")

    # 对比
    print("\n" + "=" * 70)
    print(" 结果对比")
    print("=" * 70)
    print(f"无压缩 - 空闲blocks: {free_blocks_no_compress}")
    print(f"有压缩 - 空闲blocks: {free_blocks_compress}")
    print(f"差异: {free_blocks_compress - free_blocks_no_compress} blocks")

    if free_blocks_compress > free_blocks_no_compress:
        print("\n✓ 压缩后成功释放了更多blocks！")
    else:
        print("\n⚠ 压缩后blocks数量没有增加，可能需要检查释放逻辑")

    # 输出对比
    print("\n输出长度对比:")
    print(f"无压缩: {len(outputs_no_compress[0]['token_ids'])} tokens")
    print(f"有压缩: {len(outputs_compress[0]['token_ids'])} tokens")

    print("\n测试完成!")


def test_multiple_sequences():
    """测试多个序列的内存释放"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    print("\n" + "=" * 70)
    print(" 测试多序列内存释放")
    print("=" * 70)

    model_path = "/data/huggingface/llava-1.5-7b-hf"

    gc.collect()
    torch.cuda.empty_cache()

    llm = LlavaLLM(
        model_path,
        enable_compression=True,
        async_compression=False,
        compression_factor=5,
        enforce_eager=True,
        max_model_len=2048,
    )

    initial_free = len(llm.scheduler.block_manager.free_block_ids)
    print(f"初始空闲blocks: {initial_free}")

    # 多个长prompt
    prompts = [
        "USER: Explain the complete history of machine learning in detail. ASSISTANT:",
        "USER: Describe all major programming languages and their evolution. ASSISTANT:",
        "USER: What are all the breakthroughs in computer vision research? ASSISTANT:",
        "USER: Explain natural language processing from basics to transformers. ASSISTANT:",
    ]

    sampling_params = [SamplingParams(max_tokens=128)] * len(prompts)

    print(f"\n生成{len(prompts)}个序列...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True, apply_compression=True)

    final_free = len(llm.scheduler.block_manager.free_block_ids)
    print(f"\n最终空闲blocks: {final_free}")
    print(f"恢复的blocks: {final_free - initial_free + len(prompts) * 10}（估算）")  # 大致估算

    print("\n各序列输出长度:")
    for i, out in enumerate(outputs):
        print(f"  序列{i}: {len(out['token_ids'])} tokens")


if __name__ == "__main__":
    test_memory_release()
    print("\n" + "=" * 70)
    test_multiple_sequences()
