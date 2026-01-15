"""
LLaVA + KV-Cache压缩 完整测试脚本 (基于nano-vllm)
=================================================

测试nano-vllm中LLaVA模型和KV-cache压缩的完整集成

功能：
1. 验证系统能正常运行
2. 验证压缩功能正常
3. 对比压缩vs无压缩的输出质量
4. 对比吞吐量

Date: 2024
"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys
import time
import json
import torch
import random
from datetime import datetime
from PIL import Image
from typing import List, Tuple, Dict, Any
import statistics

# 添加项目路径

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.llava_engine import LlavaLLM


def compute_text_similarity(text1: str, text2: str) -> float:
    """计算文本相似度（Jaccard相似度）"""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union) if union else 0.0


def load_test_samples(
    json_path: str = '/data/huggingface/LLaVA-Instruct-150K/llava_v1_5_mix665k.json',
    img_base_path: str = str(DATASETS_DIR) + "/",
    num_samples: int = 5,
    seed: int = 42
) -> List[Tuple[Image.Image, str]]:
    """加载测试样本"""
    random.seed(seed)

    print(f"加载测试数据...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 过滤有图像的数据
    target_datasets = ['gqa', 'coco', 'vg']
    data = [d for d in data if 'image' in d and d['image'].split('/')[0] in target_datasets]

    random.shuffle(data)
    selected = data[:num_samples * 3]  # 多选一些以防加载失败

    samples = []
    for d in selected:
        try:
            img_path = os.path.join(img_base_path, d['image'])
            if not os.path.exists(img_path):
                continue

            image = Image.open(img_path).convert('RGB')

            if d['conversations'] and d['conversations'][0]['from'] == 'human':
                prompt = d['conversations'][0]['value'].replace('<image>\n', '').replace('<image>', '')
                samples.append((image, prompt))

        except Exception as e:
            print(f"跳过样本: {e}")
            continue

        if len(samples) >= num_samples:
            break

    print(f"成功加载 {len(samples)} 个测试样本")
    return samples


def build_llava_prompt(question: str) -> str:
    """构建LLaVA提示格式"""
    # LLaVA 1.5使用的提示格式
    return f"USER: <image>\n{question} ASSISTANT:"


def run_benchmark(
    model_path: str,
    compressor_path: str,
    test_samples: List[Tuple[Image.Image, str]],
    max_new_tokens: int = 64,
    compression_factor: int = 5,
) -> Dict[str, Any]:
    """
    运行基准测试

    Args:
        model_path: LLaVA模型路径
        compressor_path: 压缩器权重路径
        test_samples: 测试样本列表
        max_new_tokens: 最大生成token数
        compression_factor: 压缩因子

    Returns:
        测试结果字典
    """
    results = {
        "original": {
            "times": [],
            "outputs": [],
            "throughputs": [],
        },
        "compressed": {
            "times": [],
            "outputs": [],
            "throughputs": [],
            "compression_ratios": [],
        },
        "comparison": {
            "similarities": [],
        }
    }

    # ===================== 测试无压缩版本 =====================
    print("\n" + "=" * 60)
    print("测试无压缩版本")
    print("=" * 60)

    try:
        llm_original = LlavaLLM(
            model_path,
            enable_compression=False,
            enforce_eager=True,  # 调试时使用eager模式
            max_model_len=2048,
        )

        sampling_params = SamplingParams(
            temperature=0.0,  # 使用greedy以便对比
            max_tokens=max_new_tokens,
        )

        for i, (image, question) in enumerate(test_samples):
            print(f"\n[原始 {i+1}/{len(test_samples)}] 处理中...")
            prompt = build_llava_prompt(question)

            start_time = time.time()
            outputs = llm_original.generate(
                [prompt],
                sampling_params,
                images=[image],
                use_tqdm=False,
            )
            elapsed = time.time() - start_time

            output_text = outputs[0]['text'] if outputs else ""
            num_tokens = len(outputs[0]['token_ids']) if outputs else 0
            throughput = num_tokens / elapsed if elapsed > 0 else 0

            results["original"]["times"].append(elapsed)
            results["original"]["outputs"].append(output_text)
            results["original"]["throughputs"].append(throughput)

            print(f"  时间: {elapsed:.3f}s, 吞吐量: {throughput:.1f} tok/s")
            print(f"  输出: {output_text[:100]}...")

        del llm_original
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"无压缩版本测试失败: {e}")
        import traceback
        traceback.print_exc()

    # ===================== 测试压缩版本 =====================
    print("\n" + "=" * 60)
    print("测试压缩版本")
    print("=" * 60)

    try:
        llm_compressed = LlavaLLM(
            model_path,
            compressor_path=compressor_path,
            compression_factor=compression_factor,
            enable_compression=True,
            enforce_eager=True,
            max_model_len=2048,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

        for i, (image, question) in enumerate(test_samples):
            print(f"\n[压缩 {i+1}/{len(test_samples)}] 处理中...")
            prompt = build_llava_prompt(question)

            start_time = time.time()
            outputs = llm_compressed.generate(
                [prompt],
                sampling_params,
                images=[image],
                use_tqdm=False,
                apply_compression=True,
            )
            elapsed = time.time() - start_time

            output_text = outputs[0]['text'] if outputs else ""
            num_tokens = len(outputs[0]['token_ids']) if outputs else 0
            throughput = num_tokens / elapsed if elapsed > 0 else 0

            results["compressed"]["times"].append(elapsed)
            results["compressed"]["outputs"].append(output_text)
            results["compressed"]["throughputs"].append(throughput)

            print(f"  时间: {elapsed:.3f}s, 吞吐量: {throughput:.1f} tok/s")
            print(f"  输出: {output_text[:100]}...")

        del llm_compressed
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"压缩版本测试失败: {e}")
        import traceback
        traceback.print_exc()

    # ===================== 计算对比指标 =====================
    print("\n" + "=" * 60)
    print("计算对比指标")
    print("=" * 60)

    for i in range(len(test_samples)):
        if i < len(results["original"]["outputs"]) and i < len(results["compressed"]["outputs"]):
            orig_text = results["original"]["outputs"][i]
            comp_text = results["compressed"]["outputs"][i]
            similarity = compute_text_similarity(orig_text, comp_text)
            results["comparison"]["similarities"].append(similarity)
            print(f"样本 {i+1}: 相似度 = {similarity:.2%}")

    return results


def print_summary(results: Dict[str, Any]):
    """打印汇总结果"""
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    orig = results["original"]
    comp = results["compressed"]
    comparison = results["comparison"]

    if orig["times"]:
        print("\n--- 无压缩 ---")
        print(f"  平均时间: {statistics.mean(orig['times']):.3f}s")
        print(f"  平均吞吐量: {statistics.mean(orig['throughputs']):.1f} tok/s")

    if comp["times"]:
        print("\n--- 压缩后 ---")
        print(f"  平均时间: {statistics.mean(comp['times']):.3f}s")
        print(f"  平均吞吐量: {statistics.mean(comp['throughputs']):.1f} tok/s")

    if comparison["similarities"]:
        print("\n--- 对比 ---")
        print(f"  平均输出相似度: {statistics.mean(comparison['similarities']):.2%}")
        print(f"  相似度范围: [{min(comparison['similarities']):.2%}, {max(comparison['similarities']):.2%}]")

    if orig["throughputs"] and comp["throughputs"]:
        orig_avg = statistics.mean(orig["throughputs"])
        comp_avg = statistics.mean(comp["throughputs"])
        if orig_avg > 0:
            speedup = comp_avg / orig_avg
            print(f"\n  吞吐量提升: {speedup:.2f}x")


def main():
    """主函数"""
    print("=" * 70)
    print("LLaVA + KV-Cache压缩 nano-vllm集成测试")
    print("=" * 70)

    # 配置
    model_path = "/data/huggingface/llava-1.5-7b-hf"
    compressor_path = str(CKPT_DIR / "llava_mlp.pth")
    num_samples = 3  # 测试样本数
    max_new_tokens = 64

    # 检查路径
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return

    if not os.path.exists(compressor_path):
        print(f"警告: 压缩器路径不存在: {compressor_path}")
        print("将只测试无压缩版本")
        compressor_path = None

    # 加载测试样本
    test_samples = load_test_samples(num_samples=num_samples)

    if not test_samples:
        print("错误: 没有可用的测试样本")
        return

    # 运行测试
    results = run_benchmark(
        model_path=model_path,
        compressor_path=compressor_path,
        test_samples=test_samples,
        max_new_tokens=max_new_tokens,
    )

    # 打印汇总
    print_summary(results)

    # 保存结果
    result_dir = str(RESULTS_DIR / "results_benchmark")
    os.makedirs(result_dir, exist_ok=True)

    result_path = os.path.join(
        result_dir,
        f"nanovllm_llava_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    # 转换为可序列化格式
    save_results = {
        "config": {
            "model_path": model_path,
            "compressor_path": compressor_path,
            "num_samples": num_samples,
            "max_new_tokens": max_new_tokens,
        },
        "original": {
            "avg_time": statistics.mean(results["original"]["times"]) if results["original"]["times"] else 0,
            "avg_throughput": statistics.mean(results["original"]["throughputs"]) if results["original"]["throughputs"] else 0,
        },
        "compressed": {
            "avg_time": statistics.mean(results["compressed"]["times"]) if results["compressed"]["times"] else 0,
            "avg_throughput": statistics.mean(results["compressed"]["throughputs"]) if results["compressed"]["throughputs"] else 0,
        },
        "comparison": {
            "avg_similarity": statistics.mean(results["comparison"]["similarities"]) if results["comparison"]["similarities"] else 0,
        }
    }

    with open(result_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
