#!/usr/bin/env python3
"""
MiniCPM模型基础测试
==================

测试MiniCPM模型的加载和基本推理功能

"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import torch
import sys

from transformers import AutoConfig, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


def test_minicpm_model_loading():
    """测试MiniCPM模型加载（不含推理）"""
    print("=" * 60)
    print(" MiniCPM Model Loading Test")
    print("=" * 60)

    model_path = '/data/huggingface/MiniCPM-V-2_6'

    # 1. 加载配置
    print("\n1. Loading config...")
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"   model_type: {hf_config.model_type}")
    print(f"   hidden_size: {hf_config.hidden_size}")
    print(f"   num_layers: {hf_config.num_hidden_layers}")
    print(f"   query_num: {getattr(hf_config, 'query_num', 'N/A')}")

    # 2. 创建模型
    print("\n2. Creating MiniCPM model...")
    from nanovllm.models.minicpm import MiniCPMForConditionalGeneration

    torch.set_default_dtype(hf_config.torch_dtype)
    torch.set_default_device('cuda')

    model = MiniCPMForConditionalGeneration(hf_config)
    print(f"   Model created: {type(model).__name__}")
    print(f"   image_token_len: {model.image_token_len}")

    # 3. 加载LLM权重
    print("\n3. Loading LLM weights...")
    from nanovllm.utils.loader import load_model
    load_model(model.llm, model_path, prefix='llm.')

    # 验证权重
    param = model.llm.model.layers[0].self_attn.qkv_proj.weight
    print(f"   qkv_proj shape: {param.shape}")
    print(f"   weight mean: {param.mean().item():.6f}")

    # 4. 加载视觉模块
    print("\n4. Loading vision modules...")
    from transformers import AutoModel
    hf_model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=hf_config.torch_dtype,
        trust_remote_code=True,
        device_map='cuda'
    )
    model.load_vision_modules(hf_model)
    print(f"   vpm type: {type(model.vpm).__name__}")
    print(f"   resampler type: {type(model.resampler).__name__}")

    # 清理HF模型
    del hf_model.llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(" Model loading test passed!")
    print("=" * 60)

    # 清理
    del model
    gc.collect()
    torch.cuda.empty_cache()


def test_minicpm_engine():
    """测试MiniCPM通过LlavaLLM引擎"""
    print("\n" + "=" * 60)
    print(" MiniCPM Engine Test")
    print("=" * 60)

    from nanovllm.engine.llava_engine import LlavaLLM
    from nanovllm.sampling_params import SamplingParams
    from transformers import AutoTokenizer

    model_path = '/data/huggingface/MiniCPM-V-2_6'

    print("\n1. Initializing LlavaLLM with MiniCPM...")
    llm = LlavaLLM(
        model_path,
        enable_compression=False,
        enforce_eager=True,
        max_model_len=4096,
    )

    print(f"   is_multimodal: {llm.model_runner.is_multimodal}")
    print(f"   model_type: {llm.model_runner.model_type}")
    print(f"   image_token_len: {llm.model_runner.image_token_len}")

    # 加载tokenizer用于解码
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 2. 测试文本生成
    print("\n2. Testing text generation...")
    prompt = "What is artificial intelligence?"

    llm.add_request(prompt, SamplingParams(max_tokens=64))

    outputs = []
    total_tokens = 0
    while not llm.is_finished():
        step_outputs, num_tokens = llm.step(apply_compression=False)
        total_tokens += abs(num_tokens)
        outputs.extend(step_outputs)

    print(f"   Prompt: {prompt}")
    print(f"   Total tokens: {total_tokens}")
    for seq_id, token_ids in outputs:
        # 解码生成的token IDs
        generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"   Output [{seq_id}]: {generated_text}")

    print("\n" + "=" * 60)
    print(" Engine test passed!")
    print("=" * 60)

    # 清理
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_minicpm_model_loading()
    print("\n\n")
    test_minicpm_engine()
