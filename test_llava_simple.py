"""
简化的LLaVA测试脚本
"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys
import gc
import torch
from PIL import Image

# 添加项目路径

def clear_gpu():
    """清理GPU内存"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

def test_llava_basic():
    """基本的LLaVA测试"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    model_path = "/data/huggingface/llava-1.5-7b-hf"

    print("=" * 60)
    print("测试LLaVA基本功能")
    print("=" * 60)

    # 加载测试图像
    img_path = str(DATASETS_DIR / "gqa" / "demo1.jpg")
    if not os.path.exists(img_path):
        # 尝试其他图像
        img_path = str(DATASETS_DIR / "gqa" / "demo1.jpg")

    if not os.path.exists(img_path):
        print(f"找不到测试图像，使用纯文本测试")
        image = None
        prompt = "USER: Hello, who are you? ASSISTANT:"
    else:
        print(f"使用测试图像: {img_path}")
        image = Image.open(img_path).convert('RGB')
        prompt = "USER: <image>\nWhat do you see in this image? ASSISTANT:"

    try:
        print("\n初始化LLaVA Engine...")
        clear_gpu()

        llm = LlavaLLM(
            model_path,
            enable_compression=False,
            enforce_eager=True,
            max_model_len=2048,
        )

        print("\n开始生成...")
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=50,
        )

        if image is not None:
            outputs = llm.generate(
                [prompt],
                sampling_params,
                images=[image],
                use_tqdm=True,
            )
        else:
            outputs = llm.generate(
                [prompt],
                sampling_params,
                use_tqdm=True,
            )

        print("\n生成结果:")
        for i, output in enumerate(outputs):
            print(f"  [{i}] {output['text']}")

        # 清理
        del llm
        clear_gpu()

        print("\n测试成功!")
        return True

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        clear_gpu()


def test_text_only():
    """测试纯文本模式（用于调试）"""
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    model_path = "/data/huggingface/llava-1.5-7b-hf"

    print("=" * 60)
    print("测试纯文本模式")
    print("=" * 60)

    try:
        print("\n初始化LLaVA Engine...")
        clear_gpu()

        llm = LlavaLLM(
            model_path,
            enable_compression=False,
            enforce_eager=True,
            max_model_len=1024,
        )

        prompt = "USER: Tell me a short joke. ASSISTANT:"

        print(f"\n输入提示: {prompt}")
        print("\n开始生成...")

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=30,
        )

        outputs = llm.generate(
            [prompt],
            sampling_params,
            use_tqdm=True,
        )

        print("\n生成结果:")
        for i, output in enumerate(outputs):
            print(f"  [{i}] {output['text']}")

        # 清理
        del llm
        clear_gpu()

        print("\n测试成功!")
        return True

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        clear_gpu()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "image", "both"], default="image")
    args = parser.parse_args()

    if args.mode == "text":
        test_text_only()
    elif args.mode == "image":
        test_llava_basic()
    else:
        test_text_only()
        test_llava_basic()
