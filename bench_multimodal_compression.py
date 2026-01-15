"""
多模态KV-Cache压缩吞吐量测试
============================

使用真实图片 + 随机文本tokens测试压缩性能

数据集:
- GQA: /data/huggingface/LLaVA-Instruct-150K/llava_v1_5_mix665k.json
- 图片: datasets/

用法:
    # 基础测试
    python bench_multimodal_compression.py --bs 32 --max_output 256

    # 压缩对比测试
    python bench_multimodal_compression.py --bs 32 --max_output 512 --compare

    # 长输出测试 (预期压缩会胜出)
    python bench_multimodal_compression.py --bs 16 --max_output 1024 --compare
"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys
import time
import argparse
import gc
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image
from tqdm import tqdm


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


def load_gqa_images(
    json_path: str,
    imgsets_path: str,
    num_samples: int,
    seed: int = 42
) -> List[Tuple[str, Image.Image]]:
    """
    加载GQA数据集的图片

    Args:
        json_path: JSON文件路径
        imgsets_path: 图片集路径
        num_samples: 需要的样本数量
        seed: 随机种子

    Returns:
        List of (image_path, PIL.Image) tuples
    """
    random.seed(seed)

    # 加载JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 过滤出GQA数据集的图片
    target_datasets = ['gqa']
    data = [d for d in data if 'image' in d.keys() and d['image'].split('/')[0] in target_datasets]

    # 随机采样
    if len(data) > num_samples:
        data = random.sample(data, num_samples)

    # 加载图片
    images = []
    for item in tqdm(data, desc="Loading images"):
        image_path = os.path.join(imgsets_path, item['image'])
        try:
            img = Image.open(image_path).convert("RGB")
            images.append((image_path, img))
        except Exception as e:
            print(f"Warning: Failed to load {image_path}: {e}")
            continue

        if len(images) >= num_samples:
            break

    print(f"Loaded {len(images)} images from GQA dataset")
    return images


def generate_multimodal_prompts(
    images: List[Tuple[str, Image.Image]],
    max_text_tokens: int = 100,
    seed: int = 42
) -> List[Tuple[str, Image.Image, List[int]]]:
    """
    生成多模态prompts (真实图片 + 随机文本tokens)

    Args:
        images: List of (path, PIL.Image) tuples
        max_text_tokens: 最大文本token数
        seed: 随机种子

    Returns:
        List of (prompt_text, image, random_tokens) tuples
    """
    random.seed(seed)

    prompts = []
    for path, img in images:
        # 生成随机文本token IDs (模拟用户输入)
        # 在实际场景中，这些会是真实的问题
        num_tokens = random.randint(20, max_text_tokens)
        random_tokens = [random.randint(100, 31999) for _ in range(num_tokens)]

        # 构建prompt (LLaVA格式)
        prompt_text = f"USER: <image>\nDescribe what you see in this image. ASSISTANT:"

        prompts.append((prompt_text, img, random_tokens))

    return prompts


def benchmark_multimodal(
    model_path: str,
    images: List[Tuple[str, Image.Image]],
    num_seqs: int,
    max_output_len: int,
    enable_compression: bool,
    compressor_path: Optional[str] = None,
    compression_factor: int = 5,
    max_model_len: int = 4096,
) -> dict:
    """
    多模态吞吐量测试

    Args:
        model_path: 模型路径
        images: 图片列表
        num_seqs: 序列数量
        max_output_len: 最大输出长度
        enable_compression: 是否启用压缩
        compressor_path: 压缩器路径
        compression_factor: 压缩因子
        max_model_len: 最大模型长度

    Returns:
        测试结果字典
    """
    mode_name = "有压缩" if enable_compression else "无压缩"
    print(f"\n{'='*60}")
    print(f"多模态吞吐量测试 - {mode_name}")
    print(f"{'='*60}")
    print(f"  序列数: {num_seqs}")
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
        enforce_eager=True,  # 多模态用eager模式
        max_model_len=max_model_len,
    )
    init_time = time.time() - t_init
    print(f"模型初始化: {init_time:.2f}s")

    # 准备prompts
    print("\n准备多模态输入...")
    selected_images = images[:num_seqs]
    if len(selected_images) < num_seqs:
        # 如果图片不够，循环使用
        while len(selected_images) < num_seqs:
            selected_images.extend(images[:min(num_seqs - len(selected_images), len(images))])

    prompts = []
    pil_images = []
    for path, img in selected_images[:num_seqs]:
        prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
        prompts.append(prompt)
        pil_images.append(img)

    # 采样参数
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=random.randint(max_output_len // 2, max_output_len)
        )
        for _ in range(num_seqs)
    ]

    total_output_tokens_expected = sum(sp.max_tokens for sp in sampling_params)
    image_tokens_per_image = (336 // 14) ** 2  # 576
    total_image_tokens = num_seqs * image_tokens_per_image

    print(f"  图片tokens: {total_image_tokens} ({num_seqs}张 × {image_tokens_per_image})")
    print(f"  预期输出tokens: {total_output_tokens_expected}")

    # 跳过预热 - 避免block manager状态问题
    # 直接进行正式测试
    print("\n跳过预热（避免block状态问题）...")

    # 正式测试
    print("\n开始吞吐测试...")
    mem_before = get_gpu_memory()

    t_start = time.time()
    try:
        outputs = llm.generate(
            prompts,
            sampling_params,
            images=pil_images,
            use_tqdm=True,
            apply_compression=enable_compression
        )
        success = True
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        outputs = []
        success = False

    t_elapsed = time.time() - t_start
    mem_after = get_gpu_memory()

    # 计算统计
    if success and outputs:
        actual_output_tokens = sum(len(o['token_ids']) for o in outputs)
        throughput = actual_output_tokens / t_elapsed
        effective_input = total_image_tokens + num_seqs * 20  # 估计文本tokens
        total_throughput = (effective_input + actual_output_tokens) / t_elapsed
    else:
        actual_output_tokens = 0
        throughput = 0
        total_throughput = 0

    results = {
        'mode': mode_name,
        'num_seqs': num_seqs,
        'total_image_tokens': total_image_tokens,
        'total_output_tokens': actual_output_tokens,
        'time': t_elapsed,
        'throughput': throughput,
        'total_throughput': total_throughput,
        'memory_peak': mem_after[1],
        'compression': enable_compression,
        'success': success,
    }

    print(f"\n{'='*60}")
    print(f"测试结果 - {mode_name}")
    print(f"{'='*60}")
    print(f"  总时间: {t_elapsed:.2f}s")
    print(f"  图片tokens: {total_image_tokens}")
    print(f"  输出tokens: {actual_output_tokens}")
    print(f"  输出吞吐量: {throughput:.2f} tok/s")
    print(f"  总吞吐量: {total_throughput:.2f} tok/s")
    print(f"  GPU峰值内存: {mem_after[1]:.2f} GB")

    # 清理
    del llm
    clear_gpu()

    return results


def run_comparison_test(
    model_path: str,
    compressor_path: str,
    images: List[Tuple[str, Image.Image]],
    num_seqs: int,
    max_output_len: int,
    compression_factor: int = 5,
    max_model_len: int = 4096,
):
    """运行压缩对比测试"""
    print(f"\n{'#'*70}")
    print(f"多模态压缩对比测试")
    print(f"{'#'*70}")
    print(f"配置: bs={num_seqs}, max_output={max_output_len}")

    results = {}

    # 无压缩
    results['no_compression'] = benchmark_multimodal(
        model_path=model_path,
        images=images,
        num_seqs=num_seqs,
        max_output_len=max_output_len,
        enable_compression=False,
        max_model_len=max_model_len,
    )

    time.sleep(2)
    clear_gpu()

    # 有压缩
    results['with_compression'] = benchmark_multimodal(
        model_path=model_path,
        images=images,
        num_seqs=num_seqs,
        max_output_len=max_output_len,
        enable_compression=True,
        compressor_path=compressor_path,
        compression_factor=compression_factor,
        max_model_len=max_model_len,
    )

    # 打印对比
    print(f"\n{'='*70}")
    print(f"对比结果汇总")
    print(f"{'='*70}")
    print(f"{'配置':<20} {'吞吐量(tok/s)':<20} {'时间(s)':<15} {'峰值内存(GB)':<15}")
    print("-" * 70)

    for name, r in results.items():
        label = "无压缩" if name == 'no_compression' else "有压缩"
        print(f"{label:<20} {r['throughput']:<20.2f} {r['time']:<15.2f} {r['memory_peak']:<15.2f}")

    if results['no_compression']['success'] and results['with_compression']['success']:
        speedup = results['with_compression']['throughput'] / results['no_compression']['throughput']
        print("-" * 70)
        if speedup >= 1.0:
            print(f"压缩加速比: {speedup:.2f}x  ✓ 压缩更快!")
        else:
            print(f"压缩加速比: {speedup:.2f}x  (压缩略慢)")

    return results


def main():
    parser = argparse.ArgumentParser(description="多模态KV-Cache压缩吞吐量测试")

    parser.add_argument("--bs", type=int, default=32,
                        help="批次大小 (默认: 32)")
    parser.add_argument("--max_output", type=int, default=256,
                        help="最大输出长度 (默认: 256)")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="最大模型长度 (默认: 4096)")
    parser.add_argument("--compression_factor", type=int, default=5,
                        help="压缩因子 (默认: 5)")
    parser.add_argument("--compare", action="store_true",
                        help="运行压缩对比测试")
    parser.add_argument("--compression", action="store_true",
                        help="仅测试压缩模式")
    parser.add_argument("--model", type=str,
                        default="/data/huggingface/llava-1.5-7b-hf",
                        help="模型路径")
    parser.add_argument("--compressor", type=str,
                        default=str(CKPT_DIR / "llava_mlp.pth"),
                        help="压缩器路径")
    parser.add_argument("--json_path", type=str,
                        default="/data/huggingface/LLaVA-Instruct-150K/llava_v1_5_mix665k.json",
                        help="数据集JSON路径")
    parser.add_argument("--img_path", type=str,
                        default=str(DATASETS_DIR) + "/",
                        help="数据集图片路径")
    parser.add_argument("--num_images", type=int, default=100,
                        help="加载的图片数量 (默认: 100)")

    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"多模态KV-Cache压缩吞吐量测试")
    print(f"{'#'*70}")
    print(f"模型: {args.model}")
    print(f"批次大小: {args.bs}")
    print(f"最大输出: {args.max_output}")

    # 加载图片
    print(f"\n加载GQA数据集图片...")
    images = load_gqa_images(
        json_path=args.json_path,
        imgsets_path=args.img_path,
        num_samples=max(args.num_images, args.bs)
    )

    if len(images) == 0:
        print("错误: 未能加载任何图片!")
        return

    if args.compare:
        # 对比测试
        run_comparison_test(
            model_path=args.model,
            compressor_path=args.compressor,
            images=images,
            num_seqs=args.bs,
            max_output_len=args.max_output,
            compression_factor=args.compression_factor,
            max_model_len=args.max_model_len,
        )
    else:
        # 单独测试
        benchmark_multimodal(
            model_path=args.model,
            images=images,
            num_seqs=args.bs,
            max_output_len=args.max_output,
            enable_compression=args.compression,
            compressor_path=args.compressor if args.compression else None,
            compression_factor=args.compression_factor,
            max_model_len=args.max_model_len,
        )


if __name__ == "__main__":
    main()
