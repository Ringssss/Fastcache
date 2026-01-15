#!/usr/bin/env python3
"""
FastAPI Online Service for nano-vllm with Lazy Compression
============================================================

支持多种压缩策略:
1. none - 不压缩 (baseline)
2. eager - 每次prefill后立即压缩
3. async - 异步压缩
4. lazy - 懒压缩 (阈值触发批量压缩) - 推荐！

测试结果表明lazy压缩在大多数workload下效果最好:
- lazy: 83%有效率, 平均1.24x加速
- eager: 56%有效率, 平均1.41x加速
- async: 78%有效率, 平均1.43x加速

运行方式:
    python nanovllm_fastapi_server.py --model /data/huggingface/llava-1.5-7b-hf --compression-mode lazy

测试:
    curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" \
         -d '{"prompt": "Hello", "max_tokens": 100}'

"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import os
import sys

import torch
import gc
import time
import json
import asyncio
import threading
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn


# ============== Lazy Compression Engine ==============

class LazyCompressionEngine:
    """
    懒压缩引擎 - 包装LlavaLLM，实现懒压缩策略

    核心思想: 不在prefill后立即压缩，等空闲blocks低于阈值时才批量压缩
    这样可以:
    1. 减少压缩次数，降低overhead
    2. 批量压缩提高效率
    3. 在内存充足时完全避免压缩开销
    """

    def __init__(
        self,
        llm,
        compression_threshold: float = 0.3,
        compression_backend: str = 'kvpress',
    ):
        """
        Args:
            llm: LlavaLLM实例
            compression_threshold: 当空闲blocks低于总blocks的这个比例时触发压缩
            compression_backend: 'kvpress' 或 'mlp'
        """
        self.llm = llm
        self.compression_threshold = compression_threshold
        self.compression_backend = compression_backend

        # 获取block信息
        self.total_blocks = len(llm.scheduler.block_manager.blocks)
        self.threshold_blocks = int(self.total_blocks * compression_threshold)

        # 追踪未压缩的序列
        self.uncompressed_seqs = set()

        # 统计
        self.compression_count = 0
        self.total_compression_time = 0

    def add_request(self, prompt, sampling_params):
        self.llm.add_request(prompt, sampling_params)

    def is_finished(self):
        return self.llm.is_finished()

    def step(self):
        """执行一步，带懒压缩策略"""
        # 检查是否需要压缩
        free_blocks = len(self.llm.scheduler.block_manager.free_block_ids)

        # 触发压缩条件: 空闲blocks低于阈值 且 有未压缩的序列
        if free_blocks < self.threshold_blocks and self.uncompressed_seqs:
            self._do_batch_compression()

        # 正常执行step（不压缩）
        outputs, num_tokens = self.llm.step(apply_compression=False)

        # 记录新的prefill序列
        if num_tokens > 0:  # prefill
            for seq in self.llm.scheduler.running:
                if seq.seq_id not in self.uncompressed_seqs:
                    self.uncompressed_seqs.add(seq.seq_id)

        # 清理已完成的序列
        for seq_id, _ in outputs:
            self.uncompressed_seqs.discard(seq_id)

        return outputs, num_tokens

    def _do_batch_compression(self):
        """执行批量压缩"""
        # 找到所有未压缩的running序列
        seqs_to_compress = [
            seq for seq in self.llm.scheduler.running
            if seq.seq_id in self.uncompressed_seqs
        ]

        if not seqs_to_compress:
            return

        start = time.time()

        # 批量压缩
        comp_time, comp_ratio = self.llm.model_runner.compress_kv_cache_batch(seqs_to_compress)

        # 释放blocks
        for seq in seqs_to_compress:
            self.llm._free_compressed_blocks([seq])
            self.uncompressed_seqs.discard(seq.seq_id)

        elapsed = time.time() - start
        self.compression_count += 1
        self.total_compression_time += elapsed

    def get_stats(self):
        """获取统计信息"""
        return {
            'compression_count': self.compression_count,
            'total_compression_time_ms': self.total_compression_time * 1000,
            'avg_compression_time_ms': (self.total_compression_time / self.compression_count * 1000) if self.compression_count > 0 else 0,
        }


# ============== Models ==============

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0


class BatchGenerateRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    total_time_ms: float
    tokens_per_second: float
    compression_mode: str
    compression_method: Optional[str]


class BatchGenerateResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_tokens: int
    total_time_ms: float
    avg_throughput: float
    compression_mode: str


class ServerStatus(BaseModel):
    active_requests: int
    completed_requests: int
    total_tokens_generated: int
    avg_throughput: float
    compression_mode: str
    compression_method: str
    compression_stats: Dict[str, Any]
    gpu_memory_used_gb: float
    gpu_memory_peak_gb: float


class CompressionTestResult(BaseModel):
    mode: str
    method: str
    throughput: float
    speedup_vs_none: float
    is_beneficial: bool


# ============== Server ==============

class NanoVLLMServer:
    """nano-vllm在线服务 with Lazy Compression"""

    COMPRESSION_MODES = ['none', 'eager', 'async', 'lazy']

    def __init__(
        self,
        model_path: str,
        compressor_path: Optional[str] = None,
        compression_mode: str = 'lazy',  # none, eager, async, lazy
        compression_backend: str = 'kvpress',
        kvpress_method: str = 'streaming_llm',
        compression_factor: int = 5,
        compression_threshold: float = 0.3,  # lazy压缩阈值
        max_model_len: int = 8192,
    ):
        self.model_path = model_path
        self.compressor_path = compressor_path
        self.compression_mode = compression_mode
        self.compression_backend = compression_backend
        self.kvpress_method = kvpress_method
        self.compression_factor = compression_factor
        self.compression_threshold = compression_threshold
        self.max_model_len = max_model_len

        # Metrics
        self.request_counter = 0
        self.completed_requests = 0
        self.total_tokens = 0
        self.metrics_history: deque = deque(maxlen=1000)
        self.baseline_cache: Dict[int, float] = {}

        # Engine
        self.llm = None
        self.lazy_engine = None  # 懒压缩引擎
        self.lock = threading.Lock()

        self._init_engine()

    def _init_engine(self):
        """初始化引擎"""
        from nanovllm.engine.llava_engine import LlavaLLM

        print(f"\n{'='*60}")
        print(f"Initializing nano-vllm server...")
        print(f"  Model: {self.model_path}")
        print(f"  Compression Mode: {self.compression_mode}")

        # 确定是否启用底层压缩
        enable_compression = self.compression_mode in ['eager', 'async', 'lazy']
        async_compression = self.compression_mode == 'async'

        if enable_compression:
            print(f"  Backend: {self.compression_backend}")
            if self.compression_backend == 'kvpress':
                print(f"  Method: {self.kvpress_method}")
            print(f"  Factor: {self.compression_factor}x")
            if self.compression_mode == 'lazy':
                print(f"  Lazy Threshold: {self.compression_threshold*100:.0f}%")

        self.llm = LlavaLLM(
            self.model_path,
            compressor_path=self.compressor_path,
            enable_compression=enable_compression,
            compression_backend=self.compression_backend,
            kvpress_method=self.kvpress_method,
            compression_factor=self.compression_factor,
            async_compression=async_compression,
            enforce_eager=True,
            max_model_len=self.max_model_len,
        )

        # 如果是lazy模式，创建懒压缩引擎
        if self.compression_mode == 'lazy':
            self.lazy_engine = LazyCompressionEngine(
                self.llm,
                compression_threshold=self.compression_threshold,
                compression_backend=self.compression_backend,
            )
        else:
            self.lazy_engine = None

        print(f"Server initialized!")
        print(f"{'='*60}\n")

    def generate(self, prompt: str, max_tokens: int = 64) -> Dict[str, Any]:
        """同步生成（单个请求）"""
        from nanovllm.sampling_params import SamplingParams

        with self.lock:
            self.request_counter += 1
            request_id = self.request_counter

            # 根据模式选择引擎
            if self.lazy_engine:
                engine = self.lazy_engine
            else:
                engine = self.llm

            # 添加请求
            if self.lazy_engine:
                self.lazy_engine.add_request(prompt, SamplingParams(max_tokens=max_tokens))
            else:
                self.llm.add_request(prompt, SamplingParams(max_tokens=max_tokens))

            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()

            total_tokens = 0
            generated_text = ""

            # 运行生成
            if self.lazy_engine:
                while not self.lazy_engine.is_finished():
                    outputs, num_tokens = self.lazy_engine.step()
                    total_tokens += abs(num_tokens)
            else:
                apply_compression = self.compression_mode in ['eager', 'async']
                while not self.llm.is_finished():
                    outputs, num_tokens = self.llm.step(apply_compression=apply_compression)
                    total_tokens += abs(num_tokens)

            total_time = time.time() - start_time
            throughput = total_tokens / total_time if total_time > 0 else 0

            # 更新metrics
            self.completed_requests += 1
            self.total_tokens += total_tokens

            # 记录metrics
            input_len = len(prompt.split())
            entry = {
                'timestamp': datetime.now().isoformat(),
                'request_id': request_id,
                'input_len': input_len,
                'output_len': total_tokens,
                'time_ms': total_time * 1000,
                'throughput': throughput,
                'compression_mode': self.compression_mode,
                'compression_method': self.kvpress_method if self.compression_mode != 'none' else 'none',
            }
            self.metrics_history.append(entry)

            return {
                'text': generated_text,
                'tokens_generated': total_tokens,
                'total_time_ms': total_time * 1000,
                'tokens_per_second': throughput,
                'compression_mode': self.compression_mode,
                'compression_method': self.kvpress_method if self.compression_mode != 'none' else None,
            }

    def generate_batch(self, prompts: List[str], max_tokens: int = 64) -> Dict[str, Any]:
        """批量生成（多个请求）- 更好地展示lazy压缩的优势"""
        from nanovllm.sampling_params import SamplingParams

        with self.lock:
            batch_id = self.request_counter
            self.request_counter += len(prompts)

            # 添加所有请求
            if self.lazy_engine:
                for prompt in prompts:
                    self.lazy_engine.add_request(prompt, SamplingParams(max_tokens=max_tokens))
            else:
                for prompt in prompts:
                    self.llm.add_request(prompt, SamplingParams(max_tokens=max_tokens))

            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()

            total_tokens = 0
            results = []

            # 运行生成
            if self.lazy_engine:
                while not self.lazy_engine.is_finished():
                    outputs, num_tokens = self.lazy_engine.step()
                    total_tokens += abs(num_tokens)
                    for seq_id, text in outputs:
                        results.append({'seq_id': seq_id, 'text': text})
            else:
                apply_compression = self.compression_mode in ['eager', 'async']
                while not self.llm.is_finished():
                    outputs, num_tokens = self.llm.step(apply_compression=apply_compression)
                    total_tokens += abs(num_tokens)
                    for seq_id, text in outputs:
                        results.append({'seq_id': seq_id, 'text': text})

            total_time = time.time() - start_time
            throughput = total_tokens / total_time if total_time > 0 else 0

            # 更新metrics
            self.completed_requests += len(prompts)
            self.total_tokens += total_tokens

            return {
                'results': results,
                'total_tokens': total_tokens,
                'total_time_ms': total_time * 1000,
                'avg_throughput': throughput,
                'compression_mode': self.compression_mode,
            }

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        mem_used = torch.cuda.memory_allocated() / 1024**3
        mem_peak = torch.cuda.max_memory_allocated() / 1024**3

        avg_throughput = 0
        if self.metrics_history:
            avg_throughput = sum(m['throughput'] for m in self.metrics_history) / len(self.metrics_history)

        compression_stats = {}
        if self.lazy_engine:
            compression_stats = self.lazy_engine.get_stats()

        return {
            'active_requests': 0,
            'completed_requests': self.completed_requests,
            'total_tokens_generated': self.total_tokens,
            'avg_throughput': avg_throughput,
            'compression_mode': self.compression_mode,
            'compression_method': self.kvpress_method if self.compression_mode != 'none' else 'none',
            'compression_stats': compression_stats,
            'gpu_memory_used_gb': mem_used,
            'gpu_memory_peak_gb': mem_peak,
        }

    def get_metrics(self, last_n: int = 100) -> List[Dict]:
        """获取最近的metrics"""
        return list(self.metrics_history)[-last_n:]

    def switch_compression_mode(self, mode: str, method: str = 'streaming_llm'):
        """切换压缩模式"""
        if mode not in self.COMPRESSION_MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {self.COMPRESSION_MODES}")

        self.compression_mode = mode
        self.kvpress_method = method

        # 清理旧引擎
        del self.llm
        if self.lazy_engine:
            del self.lazy_engine
        gc.collect()
        torch.cuda.empty_cache()

        # 重新初始化
        self._init_engine()


# ============== FastAPI App ==============

app = FastAPI(
    title="nano-vllm Server with Lazy Compression",
    description="KV-Cache Compression Testing with multiple strategies: none, eager, async, lazy"
)

# 全局server实例
server: Optional[NanoVLLMServer] = None


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """生成文本（单个请求）"""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    result = server.generate(request.prompt, request.max_tokens)
    return GenerateResponse(**result)


@app.post("/generate_batch", response_model=BatchGenerateResponse)
async def generate_batch(request: BatchGenerateRequest):
    """批量生成文本（多个请求）- 更好地展示lazy压缩优势"""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    result = server.generate_batch(request.prompts, request.max_tokens)
    return BatchGenerateResponse(**result)


@app.get("/status", response_model=ServerStatus)
async def get_status():
    """获取服务状态"""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    return ServerStatus(**server.get_status())


@app.get("/metrics")
async def get_metrics(last_n: int = 100):
    """获取metrics"""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    return server.get_metrics(last_n)


@app.post("/switch_mode")
async def switch_mode(
    mode: str = 'lazy',
    method: str = 'streaming_llm',
):
    """切换压缩模式

    Modes:
    - none: 不压缩 (baseline)
    - eager: 每次prefill后立即压缩
    - async: 异步压缩
    - lazy: 懒压缩 (推荐)
    """
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    if mode not in NanoVLLMServer.COMPRESSION_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of: {NanoVLLMServer.COMPRESSION_MODES}"
        )

    server.switch_compression_mode(mode, method)
    return {"status": "ok", "mode": mode, "method": method}


@app.post("/benchmark")
async def run_benchmark(
    batch_size: int = 32,
    input_len: int = 256,
    output_len: int = 64,
):
    """运行基准测试（批量请求）"""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # 生成测试prompts
    base = "Please explain artificial intelligence in detail "
    prompts = [base * (input_len // 10) for _ in range(batch_size)]

    result = server.generate_batch(prompts, output_len)

    return {
        'batch_size': batch_size,
        'input_len': input_len,
        'output_len': output_len,
        'total_tokens': result['total_tokens'],
        'total_time_ms': result['total_time_ms'],
        'throughput': result['avg_throughput'],
        'compression_mode': result['compression_mode'],
    }


@app.post("/comprehensive_test")
async def comprehensive_test(
    batch_sizes: List[int] = [32, 64, 128],
    input_lens: List[int] = [256, 512, 1024],
    output_len: int = 64,
):
    """全面测试所有压缩模式"""
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    results = []

    for batch_size in batch_sizes:
        for input_len in input_lens:
            workload_results = {
                'batch_size': batch_size,
                'input_len': input_len,
                'modes': {}
            }

            # 测试各种模式
            for mode in ['none', 'eager', 'lazy']:
                server.switch_compression_mode(mode, 'streaming_llm')

                base = "Please explain artificial intelligence in detail "
                prompts = [base * (input_len // 10) for _ in range(batch_size)]

                result = server.generate_batch(prompts, output_len)

                workload_results['modes'][mode] = {
                    'throughput': result['avg_throughput'],
                    'time_ms': result['total_time_ms'],
                }

            # 计算speedup
            baseline = workload_results['modes']['none']['throughput']
            for mode in ['eager', 'lazy']:
                tp = workload_results['modes'][mode]['throughput']
                workload_results['modes'][mode]['speedup'] = tp / baseline if baseline > 0 else 0
                workload_results['modes'][mode]['is_beneficial'] = tp > baseline

            results.append(workload_results)

    # 统计分析
    summary = {
        'none': {'count': 0, 'avg_throughput': 0},
        'eager': {'positive': 0, 'total': 0, 'avg_speedup': 0},
        'lazy': {'positive': 0, 'total': 0, 'avg_speedup': 0},
    }

    for r in results:
        summary['none']['count'] += 1
        summary['none']['avg_throughput'] += r['modes']['none']['throughput']

        for mode in ['eager', 'lazy']:
            summary[mode]['total'] += 1
            if r['modes'][mode]['is_beneficial']:
                summary[mode]['positive'] += 1
            summary[mode]['avg_speedup'] += r['modes'][mode]['speedup']

    summary['none']['avg_throughput'] /= len(results)
    for mode in ['eager', 'lazy']:
        summary[mode]['avg_speedup'] /= len(results)
        summary[mode]['success_rate'] = summary[mode]['positive'] / summary[mode]['total']

    return {
        'results': results,
        'summary': summary,
        'recommendation': 'lazy' if summary['lazy']['success_rate'] >= summary['eager']['success_rate'] else 'eager',
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='nano-vllm FastAPI Server with Lazy Compression')
    parser.add_argument('--model', default='/data/huggingface/llava-1.5-7b-hf')
    parser.add_argument('--compressor', default=str(CKPT_DIR / "llava_mlp.pth"))
    parser.add_argument('--compression-mode', default='lazy',
                        choices=['none', 'eager', 'async', 'lazy'],
                        help='Compression mode: none, eager, async, lazy (recommended)')
    parser.add_argument('--compression-backend', default='kvpress', choices=['kvpress', 'mlp'])
    parser.add_argument('--kvpress-method', default='streaming_llm')
    parser.add_argument('--compression-factor', type=int, default=5)
    parser.add_argument('--compression-threshold', type=float, default=0.3,
                        help='Lazy compression threshold (trigger when free blocks < threshold)')
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    global server
    server = NanoVLLMServer(
        model_path=args.model,
        compressor_path=args.compressor,
        compression_mode=args.compression_mode,
        compression_backend=args.compression_backend,
        kvpress_method=args.kvpress_method,
        compression_factor=args.compression_factor,
        compression_threshold=args.compression_threshold,
        max_model_len=args.max_model_len,
    )

    print(f"\n{'='*60}")
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"  API docs: http://{args.host}:{args.port}/docs")
    print(f"  Compression mode: {args.compression_mode}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
