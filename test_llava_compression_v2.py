"""
测试LLaVA + KV-Cache压缩
"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys
import gc
import torch
from PIL import Image
import time

# 添加项目路径

def clear_gpu():
    """清理GPU内存"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

def test_llava_with_compression():
    """测试LLaVA + 压缩"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    model_path = "/data/huggingface/llava-1.5-7b-hf"
    compressor_path = str(CKPT_DIR / "llava_mlp.pth")

    print("=" * 60)
    print("测试LLaVA + KV-Cache压缩")
    print("=" * 60)

    # 加载测试图像
    img_path = str(DATASETS_DIR / "gqa" / "demo1.jpg")
    if not os.path.exists(img_path):
        print(f"找不到测试图像: {img_path}")
        return False

    print(f"测试图像: {img_path}")
    image = Image.open(img_path).convert('RGB')
    prompt = "USER: <image>\nDescribe this image briefly. ASSISTANT:"

    # 检查压缩器权重
    if not os.path.exists(compressor_path):
        print(f"压缩器权重不存在: {compressor_path}")
        compressor_path = None

    try:
        print("\n初始化LLaVA Engine (带压缩)...")
        clear_gpu()

        llm = LlavaLLM(
            model_path,
            compressor_path=compressor_path,
            compression_factor=5,
            enable_compression=True,
            enforce_eager=True,
            max_model_len=2048,
        )

        print("\n开始生成 (apply_compression=True)...")
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=50,
        )

        start_time = time.time()
        outputs = llm.generate(
            [prompt],
            sampling_params,
            images=[image],
            use_tqdm=True,
            apply_compression=True,
        )
        elapsed = time.time() - start_time

        print(f"\n生成时间: {elapsed:.2f}s")
        print("\n生成结果:")
        for i, output in enumerate(outputs):
            print(f"  [{i}] {output['text']}")

        # 清理
        del llm
        clear_gpu()

        print("\n压缩测试成功!")
        return True

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        clear_gpu()


def compare_with_without_compression():
    """对比有压缩和无压缩的结果"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    model_path = "/data/huggingface/llava-1.5-7b-hf"
    compressor_path = str(CKPT_DIR / "llava_mlp.pth")

    print("=" * 60)
    print("对比测试: 压缩 vs 无压缩")
    print("=" * 60)

    # 使用纯文本测试（更稳定）
    prompt = "USER: Explain what machine learning is in one sentence. ASSISTANT:"

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=30,
    )

    results = {}

    # 测试无压缩
    print("\n--- 测试无压缩 ---")
    try:
        clear_gpu()
        llm = LlavaLLM(
            model_path,
            enable_compression=False,
            enforce_eager=True,
            max_model_len=1024,
        )

        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        elapsed = time.time() - start_time

        results['no_compression'] = {
            'text': outputs[0]['text'],
            'time': elapsed,
            'tokens': len(outputs[0]['token_ids'])
        }
        print(f"输出: {results['no_compression']['text']}")
        print(f"时间: {elapsed:.2f}s, 吞吐量: {results['no_compression']['tokens']/elapsed:.1f} tok/s")

        del llm
        clear_gpu()

    except Exception as e:
        print(f"无压缩测试失败: {e}")
        return

    # 测试带压缩
    if os.path.exists(compressor_path):
        print("\n--- 测试带压缩 ---")
        try:
            clear_gpu()
            llm = LlavaLLM(
                model_path,
                compressor_path=compressor_path,
                compression_factor=5,
                enable_compression=True,
                enforce_eager=True,
                max_model_len=1024,
            )

            start_time = time.time()
            outputs = llm.generate(
                [prompt],
                sampling_params,
                use_tqdm=False,
                apply_compression=True
            )
            elapsed = time.time() - start_time

            results['with_compression'] = {
                'text': outputs[0]['text'],
                'time': elapsed,
                'tokens': len(outputs[0]['token_ids'])
            }
            print(f"输出: {results['with_compression']['text']}")
            print(f"时间: {elapsed:.2f}s, 吞吐量: {results['with_compression']['tokens']/elapsed:.1f} tok/s")

            del llm
            clear_gpu()

        except Exception as e:
            print(f"压缩测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n跳过压缩测试 - 压缩器权重不存在: {compressor_path}")

    # 对比结果
    print("\n--- 对比结果 ---")
    if 'no_compression' in results:
        print(f"无压缩: {results['no_compression']['text']}")
    if 'with_compression' in results:
        print(f"有压缩: {results['with_compression']['text']}")

        # 计算相似度
        words1 = set(results['no_compression']['text'].lower().split())
        words2 = set(results['with_compression']['text'].lower().split())
        if words1 | words2:
            similarity = len(words1 & words2) / len(words1 | words2)
            print(f"相似度: {similarity:.2%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compression", "compare"], default="compare")
    args = parser.parse_args()

    if args.mode == "compression":
        test_llava_with_compression()
    else:
        compare_with_without_compression()
