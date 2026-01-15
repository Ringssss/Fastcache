"""
测试异步压缩功能

比较同步压缩和异步压缩的性能差异
"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys
import gc
import torch
import time
from PIL import Image

# 添加项目路径


def clear_gpu():
    """清理GPU内存"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")


def test_sync_compression():
    """测试同步压缩"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    model_path = "/data/huggingface/llava-1.5-7b-hf"
    compressor_path = str(CKPT_DIR / "llava_mlp.pth")

    print("=" * 60)
    print("测试同步压缩")
    print("=" * 60)

    prompt = "USER: Explain what machine learning is in one sentence. ASSISTANT:"

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=30,
    )

    try:
        print("\n初始化LLaVA Engine (同步压缩)...")
        clear_gpu()

        llm = LlavaLLM(
            model_path,
            compressor_path=compressor_path,
            compression_factor=5,
            enable_compression=True,
            async_compression=False,  # 同步
            enforce_eager=False,
            max_model_len=1024,
        )

        # 预热
        _ = llm.generate([prompt], sampling_params, use_tqdm=False)
        torch.cuda.synchronize()

        # 正式测试
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False, apply_compression=True)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        print(f"\n同步压缩结果:")
        print(f"  输出: {outputs[0]['text']}")
        print(f"  总时间: {elapsed:.3f}s")
        print(f"  Tokens: {len(outputs[0]['token_ids'])}")
        print(f"  吞吐量: {len(outputs[0]['token_ids'])/elapsed:.1f} tok/s")

        del llm
        clear_gpu()
        return elapsed

    except Exception as e:
        print(f"同步压缩测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_async_compression():
    """测试异步压缩"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    model_path = "/data/huggingface/llava-1.5-7b-hf"
    compressor_path = str(CKPT_DIR / "llava_mlp.pth")

    print("=" * 60)
    print("测试异步压缩")
    print("=" * 60)

    prompt = "USER: Explain what machine learning is in one sentence. ASSISTANT:"

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=30,
    )

    try:
        print("\n初始化LLaVA Engine (异步压缩)...")
        clear_gpu()

        llm = LlavaLLM(
            model_path,
            compressor_path=compressor_path,
            compression_factor=5,
            enable_compression=True,
            async_compression=True,  # 异步
            enforce_eager=False,
            max_model_len=1024,
        )

        # 预热
        _ = llm.generate([prompt], sampling_params, use_tqdm=False)
        torch.cuda.synchronize()

        # 正式测试
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False, apply_compression=True)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        print(f"\n异步压缩结果:")
        print(f"  输出: {outputs[0]['text']}")
        print(f"  总时间: {elapsed:.3f}s")
        print(f"  Tokens: {len(outputs[0]['token_ids'])}")
        print(f"  吞吐量: {len(outputs[0]['token_ids'])/elapsed:.1f} tok/s")

        del llm
        clear_gpu()
        return elapsed

    except Exception as e:
        print(f"异步压缩测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_batch_comparison():
    """测试批量请求下的压缩性能"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    model_path = "/data/huggingface/llava-1.5-7b-hf"
    compressor_path = str(CKPT_DIR / "llava_mlp.pth")

    print("=" * 60)
    print("批量请求压缩性能测试")
    print("=" * 60)

    prompts = [
        "USER: What is Python? ASSISTANT:",
        "USER: Explain deep learning briefly. ASSISTANT:",
        "USER: What is a neural network? ASSISTANT:",
        "USER: Define artificial intelligence. ASSISTANT:",
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=30,
    )

    results = {}

    for mode_name, async_mode in [("同步", False), ("异步", True)]:
        print(f"\n--- 测试{mode_name}压缩 (batch_size={len(prompts)}) ---")
        try:
            clear_gpu()

            llm = LlavaLLM(
                model_path,
                compressor_path=compressor_path,
                compression_factor=5,
                enable_compression=True,
                async_compression=async_mode,
                enforce_eager=False,
                max_model_len=1024,
            )

            # 预热
            _ = llm.generate(prompts[:1], sampling_params, use_tqdm=False)
            torch.cuda.synchronize()

            # 正式测试
            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False, apply_compression=True)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time

            total_tokens = sum(len(o['token_ids']) for o in outputs)

            results[mode_name] = {
                'time': elapsed,
                'tokens': total_tokens,
                'throughput': total_tokens / elapsed
            }

            print(f"  总时间: {elapsed:.3f}s")
            print(f"  总Tokens: {total_tokens}")
            print(f"  吞吐量: {total_tokens/elapsed:.1f} tok/s")

            del llm
            clear_gpu()

        except Exception as e:
            print(f"{mode_name}压缩测试失败: {e}")
            import traceback
            traceback.print_exc()

    # 对比
    if len(results) == 2:
        print("\n--- 性能对比 ---")
        sync_time = results['同步']['time']
        async_time = results['异步']['time']
        speedup = sync_time / async_time if async_time > 0 else 0
        print(f"同步压缩时间: {sync_time:.3f}s")
        print(f"异步压缩时间: {async_time:.3f}s")
        print(f"加速比: {speedup:.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sync", "async", "batch", "all"], default="all")
    args = parser.parse_args()

    if args.mode == "sync":
        test_sync_compression()
    elif args.mode == "async":
        test_async_compression()
    elif args.mode == "batch":
        test_batch_comparison()
    else:
        sync_time = test_sync_compression()
        print("\n" + "=" * 60 + "\n")
        async_time = test_async_compression()

        if sync_time and async_time:
            print("\n" + "=" * 60)
            print("总结")
            print("=" * 60)
            print(f"同步压缩时间: {sync_time:.3f}s")
            print(f"异步压缩时间: {async_time:.3f}s")
            speedup = sync_time / async_time if async_time > 0 else 0
            print(f"加速比: {speedup:.2f}x")

        print("\n" + "=" * 60 + "\n")
        test_batch_comparison()
