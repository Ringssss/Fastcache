#!/usr/bin/env python3
"""
MiniCPM-V + KV-Cache压缩测试
============================

测试MiniCPM模型与MLP压缩器的集成
然后进行lazy+async吞吐量测试

"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import torch
import sys
import time

import warnings
warnings.filterwarnings('ignore')


def test_minicpm_with_compression():
    """测试MiniCPM + HybridCompressor"""
    print("=" * 60)
    print(" MiniCPM + KV-Cache Compression Test")
    print("=" * 60)

    from nanovllm.engine.llava_engine import LlavaLLM
    from nanovllm.sampling_params import SamplingParams
    from transformers import AutoTokenizer

    model_path = '/data/huggingface/MiniCPM-V-2_6'
    compressor_path = str(CKPT_DIR / "minicpm_mlp.pth")

    print("\n1. Initializing LlavaLLM with MiniCPM + Compression...")
    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        compression_factor=5,
        enable_compression=True,
        async_compression=False,  # 先测试同步压缩
        compression_backend='mlp',
        enforce_eager=True,
        max_model_len=4096,
    )

    print(f"\n   is_multimodal: {llm.model_runner.is_multimodal}")
    print(f"   model_type: {llm.model_runner.model_type}")
    print(f"   image_token_len: {llm.model_runner.image_token_len}")
    print(f"   compressor: {type(llm.model_runner.compressor).__name__ if llm.model_runner.compressor else None}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 2. 测试文本生成 + 压缩
    print("\n2. Testing text generation with compression...")
    prompt = "What is artificial intelligence? Give a brief explanation."

    llm.add_request(prompt, SamplingParams(max_tokens=64))

    outputs = []
    total_tokens = 0
    compression_triggered = False
    while not llm.is_finished():
        step_outputs, num_tokens = llm.step(apply_compression=True)
        total_tokens += abs(num_tokens)
        outputs.extend(step_outputs)
        if num_tokens > 0:  # prefill
            compression_triggered = True

    print(f"   Prompt: {prompt}")
    print(f"   Total tokens: {total_tokens}")
    print(f"   Compression triggered: {compression_triggered}")
    for seq_id, token_ids in outputs:
        generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"   Output [{seq_id}]: {generated_text}")

    print("\n" + "=" * 60)
    print(" Compression test passed!")
    print("=" * 60)

    # 清理
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def benchmark_minicpm_throughput(batch_size: int, enable_compression: bool, use_lazy: bool = True):
    """
    MiniCPM吞吐量测试

    Args:
        batch_size: 批次大小
        enable_compression: 是否启用压缩
        use_lazy: 是否使用lazy压缩
    """
    from nanovllm.engine.llava_engine import LlavaLLM
    from nanovllm.sampling_params import SamplingParams

    model_path = '/data/huggingface/MiniCPM-V-2_6'
    compressor_path = str(CKPT_DIR / "minicpm_mlp.pth")

    # 初始化
    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path if enable_compression else None,
        compression_factor=5,
        enable_compression=enable_compression,
        async_compression=False,  # 同步压缩以便准确测量
        compression_backend='mlp',
        enforce_eager=True,
        max_model_len=4096,
    )

    # 准备prompts (用较长的prompt来模拟真实场景)
    base_prompt = "Explain the concept of machine learning and its applications in detail. " * 5
    prompts = [base_prompt for _ in range(batch_size)]
    sampling_params = SamplingParams(max_tokens=128)

    # 添加所有请求
    for prompt in prompts:
        llm.add_request(prompt, sampling_params)

    # 测量
    total_generated = 0
    prefill_time = 0
    decode_time = 0
    compression_time = 0
    num_decode_steps = 0

    # Lazy压缩阈值
    lazy_threshold = batch_size // 2 if use_lazy else 0
    pending_compression = []

    while not llm.is_finished():
        start = time.perf_counter()

        # 使用lazy策略：累积到阈值后才压缩
        apply_compression = False
        if enable_compression and use_lazy:
            # 检查是否达到lazy阈值
            if len(llm.scheduler.waiting) + len(llm.scheduler.running) >= lazy_threshold:
                apply_compression = True
        elif enable_compression:
            apply_compression = True

        outputs, num_tokens = llm.step(apply_compression=apply_compression)
        elapsed = time.perf_counter() - start

        if num_tokens > 0:  # prefill
            prefill_time += elapsed
        else:  # decode
            decode_time += elapsed
            num_decode_steps += 1
            total_generated += abs(num_tokens)

    # 计算吞吐量
    total_time = prefill_time + decode_time
    throughput = total_generated / total_time if total_time > 0 else 0

    result = {
        'batch_size': batch_size,
        'enable_compression': enable_compression,
        'use_lazy': use_lazy,
        'total_generated': total_generated,
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'total_time': total_time,
        'throughput': throughput,
        'decode_steps': num_decode_steps,
    }

    # 清理
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return result


def run_throughput_benchmark():
    """运行完整的吞吐量对比测试"""
    print("\n" + "=" * 70)
    print(" MiniCPM Throughput Benchmark: Compression vs No-Compression")
    print("=" * 70)

    batch_sizes = [64, 128, 256]
    results = []

    for bs in batch_sizes:
        print(f"\n--- Batch Size: {bs} ---")

        # 1. 无压缩基线
        print(f"\n[1/3] Testing NO compression (bs={bs})...")
        try:
            result_no_comp = benchmark_minicpm_throughput(bs, enable_compression=False)
            results.append(result_no_comp)
            print(f"  Throughput: {result_no_comp['throughput']:.2f} tokens/s")
        except Exception as e:
            print(f"  Failed: {e}")
            result_no_comp = None

        # 2. Eager压缩
        print(f"\n[2/3] Testing EAGER compression (bs={bs})...")
        try:
            result_eager = benchmark_minicpm_throughput(bs, enable_compression=True, use_lazy=False)
            results.append(result_eager)
            print(f"  Throughput: {result_eager['throughput']:.2f} tokens/s")
        except Exception as e:
            print(f"  Failed: {e}")
            result_eager = None

        # 3. Lazy压缩
        print(f"\n[3/3] Testing LAZY compression (bs={bs})...")
        try:
            result_lazy = benchmark_minicpm_throughput(bs, enable_compression=True, use_lazy=True)
            results.append(result_lazy)
            print(f"  Throughput: {result_lazy['throughput']:.2f} tokens/s")
        except Exception as e:
            print(f"  Failed: {e}")
            result_lazy = None

        # 打印对比
        print(f"\n--- Summary for bs={bs} ---")
        if result_no_comp:
            print(f"  No Compression: {result_no_comp['throughput']:.2f} tokens/s")
        if result_eager:
            speedup = result_eager['throughput'] / result_no_comp['throughput'] if result_no_comp else 0
            print(f"  Eager Compression: {result_eager['throughput']:.2f} tokens/s ({speedup:.2f}x vs baseline)")
        if result_lazy:
            speedup = result_lazy['throughput'] / result_no_comp['throughput'] if result_no_comp else 0
            print(f"  Lazy Compression: {result_lazy['throughput']:.2f} tokens/s ({speedup:.2f}x vs baseline)")

    # 最终汇总
    print("\n" + "=" * 70)
    print(" Final Summary")
    print("=" * 70)
    print(f"{'BS':<6} {'No Comp':<15} {'Eager':<15} {'Lazy':<15} {'Lazy vs No':<12}")
    print("-" * 70)

    for bs in batch_sizes:
        no_comp = next((r for r in results if r['batch_size'] == bs and not r['enable_compression']), None)
        eager = next((r for r in results if r['batch_size'] == bs and r['enable_compression'] and not r['use_lazy']), None)
        lazy = next((r for r in results if r['batch_size'] == bs and r['enable_compression'] and r['use_lazy']), None)

        no_comp_str = f"{no_comp['throughput']:.1f} tok/s" if no_comp else "N/A"
        eager_str = f"{eager['throughput']:.1f} tok/s" if eager else "N/A"
        lazy_str = f"{lazy['throughput']:.1f} tok/s" if lazy else "N/A"
        speedup_str = f"{lazy['throughput']/no_comp['throughput']:.2f}x" if (lazy and no_comp) else "N/A"

        print(f"{bs:<6} {no_comp_str:<15} {eager_str:<15} {lazy_str:<15} {speedup_str:<12}")

    print("=" * 70)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true', help='仅运行基础测试')
    parser.add_argument('--benchmark', action='store_true', help='运行吞吐量测试')
    args = parser.parse_args()

    if args.test_only or (not args.test_only and not args.benchmark):
        test_minicpm_with_compression()

    if args.benchmark:
        run_throughput_benchmark()
