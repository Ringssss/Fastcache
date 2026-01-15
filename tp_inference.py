#!/usr/bin/env python3
"""
Tensor Parallel Inference with nano-vllm
=========================================

使用多GPU进行张量并行推理。

运行方式:
    # 单卡测试
    python tp_inference.py --model /data/huggingface/Llama-2-7b-chat-hf --tp-size 1

    # 双卡TP
    torchrun --nproc_per_node=2 tp_inference.py --model /data/huggingface/Llama-3.1-70B-Instruct --tp-size 2

    # 三卡TP
    torchrun --nproc_per_node=3 tp_inference.py --model /data/huggingface/Llama-3.1-70B-Instruct --tp-size 3

"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys

import torch
import time
import argparse
import warnings
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description='TP Inference with nano-vllm')
    parser.add_argument('--model', default='/data/huggingface/Llama-2-7b-chat-hf',
                        help='Model path')
    parser.add_argument('--tp-size', type=int, default=1,
                        help='Tensor parallel size (number of GPUs)')
    parser.add_argument('--prompt', default='Please explain artificial intelligence in detail.',
                        help='Prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=64,
                        help='Max tokens to generate')
    parser.add_argument('--max-model-len', type=int, default=4096,
                        help='Max model length')
    parser.add_argument('--enable-compression', action='store_true',
                        help='Enable KV-cache compression')
    parser.add_argument('--compression-method', default='streaming_llm',
                        help='Compression method')
    args = parser.parse_args()

    # 初始化TP
    from nanovllm.utils.tp import init_tensor_parallel, get_tp_rank, get_tp_size

    # 从环境变量获取rank
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # 初始化TP
    init_tensor_parallel(tp_size=args.tp_size, rank=local_rank)

    tp_rank = get_tp_rank()
    tp_size = get_tp_size()

    if tp_rank == 0:
        print(f"Tensor Parallel Inference")
        print(f"  Model: {args.model}")
        print(f"  TP Size: {tp_size}")
        print(f"  Compression: {args.enable_compression}")

    # 导入引擎
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.engine.llava_engine import LlavaLLM

    # 初始化引擎
    llm = LlavaLLM(
        args.model,
        enable_compression=args.enable_compression,
        compression_backend='kvpress',
        kvpress_method=args.compression_method,
        compression_factor=5,
        async_compression=False,
        enforce_eager=True,
        max_model_len=args.max_model_len,
    )

    if tp_rank == 0:
        print(f"\nEngine initialized!")
        print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # 添加请求
    llm.add_request(args.prompt, SamplingParams(max_tokens=args.max_tokens))

    # 运行生成
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    total_tokens = 0
    while not llm.is_finished():
        outputs, num_tokens = llm.step(apply_compression=args.enable_compression)
        if num_tokens > 0:
            total_tokens += num_tokens
        else:
            total_tokens += (-num_tokens)

    total_time = time.time() - start_time
    throughput = total_tokens / total_time

    _, mem_peak = torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3

    if tp_rank == 0:
        print(f"\nResults:")
        print(f"  Tokens generated: {total_tokens}")
        print(f"  Total time: {total_time*1000:.1f} ms")
        print(f"  Throughput: {throughput:.1f} tok/s")
        print(f"  Peak memory: {mem_peak:.2f} GB")

    # 清理
    from nanovllm.utils.tp import cleanup_tensor_parallel
    cleanup_tensor_parallel()


if __name__ == '__main__':
    main()
