"""
LLaVA + KV-Cache压缩 吞吐量基准测试
====================================

测试高并发(bs=256)random input下的纯吞吐性能

用法:
    # 纯文本测试 (无图像)
    python bench_llava.py --mode text --bs 256

    # 带图像测试 (multimodal)
    python bench_llava.py --mode image --bs 64

    # 压缩对比测试
    python bench_llava.py --mode compare --bs 128

    # 指定输入输出长度
    python bench_llava.py --mode text --bs 256 --max_input 512 --max_output 512
"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys
import time
import argparse
import gc
from random import randint, seed, choice

import torch

# 添加项目路径

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.llava_engine import LlavaLLM


def clear_gpu():
    """清理GPU内存"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_gpu_memory():
    """获取GPU内存使用"""
    allocated = torch.cuda.memory_allocated() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    return allocated, max_allocated


def generate_random_prompts(
    num_seqs: int,
    max_input_len: int,
    vocab_size: int = 32000,
    with_image_token: bool = False,
    image_token_id: int = 32000,
    seed_val: int = 42
):
    """
    生成随机输入prompts

    Args:
        num_seqs: 序列数量
        max_input_len: 最大输入长度
        vocab_size: 词表大小
        with_image_token: 是否包含图像token
        image_token_id: 图像token ID
        seed_val: 随机种子

    Returns:
        List of token ID lists
    """
    seed(seed_val)
    prompts = []

    for _ in range(num_seqs):
        seq_len = randint(100, max_input_len)

        if with_image_token:
            # 在开头附近插入图像token
            tokens = [randint(0, vocab_size - 1) for _ in range(seq_len)]
            # 在位置10-30之间插入图像token
            insert_pos = randint(10, min(30, seq_len - 1))
            tokens[insert_pos] = image_token_id
        else:
            tokens = [randint(0, vocab_size - 1) for _ in range(seq_len)]

        prompts.append(tokens)

    return prompts


def generate_fake_images(num_images: int, image_size: int = 336):
    """
    生成假图像用于测试

    Args:
        num_images: 图像数量
        image_size: 图像尺寸

    Returns:
        List of PIL Images
    """
    from PIL import Image
    import numpy as np

    images = []
    for _ in range(num_images):
        # 生成随机RGB图像
        arr = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        img = Image.fromarray(arr, 'RGB')
        images.append(img)

    return images


def benchmark_text_only(
    model_path: str,
    num_seqs: int = 256,
    max_input_len: int = 1024,
    max_output_len: int = 1024,
    enable_compression: bool = False,
    compressor_path: str = None,
    compression_factor: int = 5,
    enforce_eager: bool = False,
    max_model_len: int = 4096,
):
    """
    纯文本吞吐量测试

    Args:
        model_path: 模型路径
        num_seqs: 批次大小
        max_input_len: 最大输入长度
        max_output_len: 最大输出长度
        enable_compression: 是否启用压缩
        compressor_path: 压缩器路径
        compression_factor: 压缩因子
        enforce_eager: 是否使用eager模式
        max_model_len: 最大模型长度

    Returns:
        测试结果字典
    """
    print(f"\n{'='*60}")
    print(f"纯文本吞吐量测试")
    print(f"{'='*60}")
    print(f"  批次大小: {num_seqs}")
    print(f"  最大输入长度: {max_input_len}")
    print(f"  最大输出长度: {max_output_len}")
    print(f"  压缩: {'启用' if enable_compression else '禁用'}")

    clear_gpu()

    # 初始化模型
    print("\n初始化模型...")
    t_init = time.time()
    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path if enable_compression else None,
        compression_factor=compression_factor,
        enable_compression=enable_compression,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
    )
    init_time = time.time() - t_init
    print(f"模型初始化: {init_time:.2f}s")

    # 生成随机prompts
    print("\n生成随机输入...")
    prompt_token_ids = generate_random_prompts(
        num_seqs=num_seqs,
        max_input_len=max_input_len,
        with_image_token=False
    )

    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(100, max_output_len)
        )
        for _ in range(num_seqs)
    ]

    total_input_tokens = sum(len(p) for p in prompt_token_ids)
    total_output_tokens = sum(sp.max_tokens for sp in sampling_params)

    print(f"  总输入tokens: {total_input_tokens}")
    print(f"  总输出tokens: {total_output_tokens}")

    # 预热
    print("\n预热...")
    llm.generate(
        ["Warmup: "],
        SamplingParams(max_tokens=10),
        use_tqdm=False
    )
    clear_gpu()

    # 正式测试
    print("\n开始吞吐测试...")
    mem_before = get_gpu_memory()

    t_start = time.time()
    outputs = llm.generate(
        prompt_token_ids,
        sampling_params,
        use_tqdm=True,
        apply_compression=enable_compression
    )
    t_elapsed = time.time() - t_start

    mem_after = get_gpu_memory()

    # 计算统计
    actual_output_tokens = sum(len(o['token_ids']) for o in outputs)
    throughput = actual_output_tokens / t_elapsed
    input_throughput = total_input_tokens / t_elapsed

    results = {
        'num_seqs': num_seqs,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': actual_output_tokens,
        'time': t_elapsed,
        'throughput': throughput,
        'input_throughput': input_throughput,
        'total_throughput': (total_input_tokens + actual_output_tokens) / t_elapsed,
        'memory_peak': mem_after[1],
        'compression': enable_compression,
    }

    print(f"\n{'='*60}")
    print(f"测试结果")
    print(f"{'='*60}")
    print(f"  总时间: {t_elapsed:.2f}s")
    print(f"  实际输出tokens: {actual_output_tokens}")
    print(f"  输出吞吐量: {throughput:.2f} tok/s")
    print(f"  总吞吐量: {results['total_throughput']:.2f} tok/s")
    print(f"  GPU峰值内存: {mem_after[1]:.2f} GB")

    # 清理
    del llm
    clear_gpu()

    return results


def benchmark_with_images(
    model_path: str,
    num_seqs: int = 64,
    max_input_len: int = 256,
    max_output_len: int = 512,
    enable_compression: bool = False,
    compressor_path: str = None,
    compression_factor: int = 5,
    enforce_eager: bool = True,
    max_model_len: int = 4096,
):
    """
    带图像的吞吐量测试

    Args:
        与benchmark_text_only相同

    Returns:
        测试结果字典
    """
    print(f"\n{'='*60}")
    print(f"多模态(图像)吞吐量测试")
    print(f"{'='*60}")
    print(f"  批次大小: {num_seqs}")
    print(f"  最大输入长度: {max_input_len}")
    print(f"  最大输出长度: {max_output_len}")
    print(f"  压缩: {'启用' if enable_compression else '禁用'}")

    clear_gpu()

    # 初始化模型
    print("\n初始化模型...")
    t_init = time.time()
    llm = LlavaLLM(
        model_path,
        compressor_path=compressor_path if enable_compression else None,
        compression_factor=compression_factor,
        enable_compression=enable_compression,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
    )
    init_time = time.time() - t_init
    print(f"模型初始化: {init_time:.2f}s")

    # 生成随机prompts (带图像token)
    print("\n生成随机输入...")
    prompt_token_ids = generate_random_prompts(
        num_seqs=num_seqs,
        max_input_len=max_input_len,
        with_image_token=True,
        image_token_id=32000  # LLaVA默认
    )

    # 生成假图像
    print("生成测试图像...")
    images = generate_fake_images(num_seqs, image_size=336)

    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(100, max_output_len)
        )
        for _ in range(num_seqs)
    ]

    total_input_tokens = sum(len(p) for p in prompt_token_ids)
    # 图像token会被扩展为576个tokens
    image_tokens_per_image = (336 // 14) ** 2  # 576
    total_image_tokens = num_seqs * image_tokens_per_image
    total_output_tokens = sum(sp.max_tokens for sp in sampling_params)

    print(f"  总输入tokens: {total_input_tokens}")
    print(f"  图像tokens: {total_image_tokens} ({num_seqs}张图像 x {image_tokens_per_image})")
    print(f"  总输出tokens: {total_output_tokens}")

    # 预热
    print("\n预热...")
    llm.generate(
        ["Warmup: "],
        SamplingParams(max_tokens=10),
        use_tqdm=False
    )
    clear_gpu()

    # 正式测试
    print("\n开始吞吐测试...")
    mem_before = get_gpu_memory()

    t_start = time.time()
    outputs = llm.generate(
        prompt_token_ids,
        sampling_params,
        images=images,
        use_tqdm=True,
        apply_compression=enable_compression
    )
    t_elapsed = time.time() - t_start

    mem_after = get_gpu_memory()

    # 计算统计
    actual_output_tokens = sum(len(o['token_ids']) for o in outputs)
    throughput = actual_output_tokens / t_elapsed
    effective_input = total_input_tokens + total_image_tokens
    total_throughput = (effective_input + actual_output_tokens) / t_elapsed

    results = {
        'num_seqs': num_seqs,
        'total_input_tokens': total_input_tokens,
        'total_image_tokens': total_image_tokens,
        'total_output_tokens': actual_output_tokens,
        'time': t_elapsed,
        'throughput': throughput,
        'total_throughput': total_throughput,
        'memory_peak': mem_after[1],
        'compression': enable_compression,
    }

    print(f"\n{'='*60}")
    print(f"测试结果")
    print(f"{'='*60}")
    print(f"  总时间: {t_elapsed:.2f}s")
    print(f"  实际输出tokens: {actual_output_tokens}")
    print(f"  输出吞吐量: {throughput:.2f} tok/s")
    print(f"  总吞吐量: {total_throughput:.2f} tok/s")
    print(f"  GPU峰值内存: {mem_after[1]:.2f} GB")

    # 清理
    del llm
    clear_gpu()

    return results


def benchmark_compression_comparison(
    model_path: str,
    compressor_path: str,
    num_seqs: int = 128,
    max_input_len: int = 512,
    max_output_len: int = 512,
    compression_factor: int = 5,
    max_model_len: int = 4096,
):
    """
    压缩 vs 无压缩 对比测试
    """
    print(f"\n{'='*70}")
    print(f"压缩对比吞吐量测试")
    print(f"{'='*70}")

    results = {}

    # 测试无压缩
    print("\n" + "="*30 + " 无压缩 " + "="*30)
    results['no_compression'] = benchmark_text_only(
        model_path=model_path,
        num_seqs=num_seqs,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        enable_compression=False,
        enforce_eager=False,  # 使用CUDA Graph
        max_model_len=max_model_len,
    )

    clear_gpu()
    time.sleep(2)  # 等待GPU完全释放

    # 测试带压缩
    if compressor_path and os.path.exists(compressor_path):
        print("\n" + "="*30 + " 带压缩 " + "="*30)
        results['with_compression'] = benchmark_text_only(
            model_path=model_path,
            num_seqs=num_seqs,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            enable_compression=True,
            compressor_path=compressor_path,
            compression_factor=compression_factor,
            enforce_eager=False,  # 使用CUDA Graph加速decode
            max_model_len=max_model_len,
        )
    else:
        print(f"\n跳过压缩测试 - 压缩器不存在: {compressor_path}")

    # 打印对比结果
    print(f"\n{'='*70}")
    print(f"对比结果汇总")
    print(f"{'='*70}")
    print(f"{'配置':<20} {'吞吐量(tok/s)':<20} {'时间(s)':<15} {'峰值内存(GB)':<15}")
    print("-" * 70)

    for name, r in results.items():
        label = "无压缩" if name == 'no_compression' else "有压缩"
        print(f"{label:<20} {r['throughput']:<20.2f} {r['time']:<15.2f} {r['memory_peak']:<15.2f}")

    if 'no_compression' in results and 'with_compression' in results:
        speedup = results['with_compression']['throughput'] / results['no_compression']['throughput']
        print("-" * 70)
        print(f"吞吐量提升: {speedup:.2f}x")

    return results


def main():
    parser = argparse.ArgumentParser(description="LLaVA + KV-Cache压缩 吞吐量基准测试")

    parser.add_argument("--mode", type=str, default="text",
                        choices=["text", "image", "compare"],
                        help="测试模式: text(纯文本), image(多模态), compare(压缩对比)")
    parser.add_argument("--bs", type=int, default=256,
                        help="批次大小 (默认: 256)")
    parser.add_argument("--max_input", type=int, default=1024,
                        help="最大输入长度 (默认: 1024)")
    parser.add_argument("--max_output", type=int, default=1024,
                        help="最大输出长度 (默认: 1024)")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="最大模型长度 (默认: 4096)")
    parser.add_argument("--compression", action="store_true",
                        help="启用压缩")
    parser.add_argument("--compression_factor", type=int, default=5,
                        help="压缩因子 (默认: 5)")
    parser.add_argument("--eager", action="store_true",
                        help="使用eager模式 (禁用CUDA Graph)")
    parser.add_argument("--model", type=str,
                        default="/data/huggingface/llava-1.5-7b-hf",
                        help="模型路径")
    parser.add_argument("--compressor", type=str,
                        default=str(CKPT_DIR / "llava_mlp.pth"),
                        help="压缩器路径")

    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"LLaVA + KV-Cache压缩 吞吐量基准测试")
    print(f"{'#'*70}")
    print(f"模型: {args.model}")
    print(f"模式: {args.mode}")
    print(f"批次大小: {args.bs}")
    print(f"最大输入/输出: {args.max_input}/{args.max_output}")

    if args.mode == "text":
        benchmark_text_only(
            model_path=args.model,
            num_seqs=args.bs,
            max_input_len=args.max_input,
            max_output_len=args.max_output,
            enable_compression=args.compression,
            compressor_path=args.compressor,
            compression_factor=args.compression_factor,
            enforce_eager=args.eager,
            max_model_len=args.max_model_len,
        )

    elif args.mode == "image":
        # 图像模式建议使用较小的bs
        if args.bs > 64:
            print(f"\n警告: 图像模式建议bs<=64, 当前bs={args.bs}")
        benchmark_with_images(
            model_path=args.model,
            num_seqs=args.bs,
            max_input_len=args.max_input,
            max_output_len=args.max_output,
            enable_compression=args.compression,
            compressor_path=args.compressor,
            compression_factor=args.compression_factor,
            enforce_eager=True,  # 图像模式强制eager
            max_model_len=args.max_model_len,
        )

    elif args.mode == "compare":
        benchmark_compression_comparison(
            model_path=args.model,
            compressor_path=args.compressor,
            num_seqs=args.bs,
            max_input_len=args.max_input,
            max_output_len=args.max_output,
            compression_factor=args.compression_factor,
            max_model_len=args.max_model_len,
        )


if __name__ == "__main__":
    main()
