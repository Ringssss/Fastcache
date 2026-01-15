"""
流水线异步压缩 vs 同步压缩 性能对比测试
"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys
import time
import gc
from random import randint, seed

import torch


from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.llava_engine import LlavaLLM


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def generate_random_prompts(num_seqs, max_input_len, seed_val=42):
    seed(seed_val)
    prompts = []
    for _ in range(num_seqs):
        seq_len = randint(100, max_input_len)
        tokens = [randint(0, 31999) for _ in range(seq_len)]
        prompts.append(tokens)
    return prompts


def benchmark_compression_mode(
    model_path: str,
    compressor_path: str,
    num_seqs: int,
    max_input_len: int,
    max_output_len: int,
    async_compression: bool,
    mode_name: str
):
    """测试特定压缩模式"""
    print(f"\n{'='*60}")
    print(f"测试 {mode_name}")
    print(f"{'='*60}")

    clear_gpu()

    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path,
        compression_factor=5,
        enable_compression=True,
        async_compression=async_compression,
        enforce_eager=False,
        max_model_len=4096,
    )

    prompts = generate_random_prompts(num_seqs, max_input_len)

    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(50, max_output_len)
        )
        for _ in range(num_seqs)
    ]

    total_input_tokens = sum(len(p) for p in prompts)
    total_output_tokens = sum(sp.max_tokens for sp in sampling_params)

    print(f"  序列数: {num_seqs}")
    print(f"  总输入tokens: {total_input_tokens}")
    print(f"  总输出tokens: {total_output_tokens}")

    # 预热
    llm.generate(["Warmup: "], SamplingParams(max_tokens=10), use_tqdm=False)
    clear_gpu()

    # 测试
    start = time.time()
    outputs = llm.generate(
        prompts, sampling_params,
        use_tqdm=True, apply_compression=True
    )
    elapsed = time.time() - start

    actual_output_tokens = sum(len(o['token_ids']) for o in outputs)
    throughput = actual_output_tokens / elapsed

    print(f"\n结果:")
    print(f"  总时间: {elapsed:.3f}s")
    print(f"  输出tokens: {actual_output_tokens}")
    print(f"  吞吐量: {throughput:.2f} tok/s")

    del llm
    clear_gpu()

    return {
        'time': elapsed,
        'throughput': throughput,
        'output_tokens': actual_output_tokens
    }


def main():
    model_path = "/data/huggingface/llava-1.5-7b-hf"
    compressor_path = str(CKPT_DIR / "llava_mlp.pth")

    # 测试配置
    test_configs = [
        (64, 256, 128),   # bs=64
        (128, 256, 128),  # bs=128
    ]

    print("#" * 70)
    print("流水线异步压缩 vs 同步压缩 性能对比")
    print("#" * 70)

    results = {}

    for num_seqs, max_input, max_output in test_configs:
        print(f"\n\n{'#'*70}")
        print(f"配置: bs={num_seqs}, max_input={max_input}, max_output={max_output}")
        print(f"{'#'*70}")

        # 同步压缩
        sync_result = benchmark_compression_mode(
            model_path, compressor_path,
            num_seqs, max_input, max_output,
            async_compression=False,
            mode_name="同步压缩 (批量GEMM)"
        )

        time.sleep(2)

        # 异步压缩
        async_result = benchmark_compression_mode(
            model_path, compressor_path,
            num_seqs, max_input, max_output,
            async_compression=True,
            mode_name="流水线异步压缩 (批量GEMM)"
        )

        results[f"bs{num_seqs}"] = {
            'sync': sync_result,
            'async': async_result
        }

        speedup = sync_result['time'] / async_result['time']
        print(f"\n加速比: {speedup:.2f}x")

    # 汇总
    print("\n\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)
    print(f"{'配置':<15} {'同步(s)':<12} {'异步(s)':<12} {'加速比':<10}")
    print("-" * 70)

    for config, data in results.items():
        sync_time = data['sync']['time']
        async_time = data['async']['time']
        speedup = sync_time / async_time
        print(f"{config:<15} {sync_time:<12.3f} {async_time:<12.3f} {speedup:<10.2f}x")


if __name__ == "__main__":
    main()
