"""
Zero-Overhead Compute-Memory Overlap Compression
================================================

核心思想：
1. Decode阶段是Memory-bound（带宽占满，算力空闲）
2. MLP压缩是Compute-bound（纯GEMM运算）
3. 将压缩任务"塞"进Decode的计算间隙，实现零开销

实现策略：
1. CUDA Stream优先级控制：Decode用高优先级，压缩用低优先级
2. 分层压缩：将32层的压缩拆分成小块，逐步塞入decode间隙
3. 负载感知调度：监测decode负载，动态发射压缩任务

"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time


class CompressionState(Enum):
    """压缩状态"""
    PENDING = 0      # 等待压缩
    IN_PROGRESS = 1  # 压缩中
    COMPLETED = 2    # 压缩完成


@dataclass
class CompressionTask:
    """压缩任务"""
    seq_id: int
    layer_start: int
    layer_end: int
    kv_cache_slice: List[Tuple[torch.Tensor, torch.Tensor]]  # 待压缩的KV-cache切片
    it_len: List[int]
    priority: int = 0  # 优先级，越小越优先


class StreamPriorityManager:
    """
    CUDA Stream优先级管理器

    CUDA支持stream优先级（-1最高，0最低）
    我们用高优先级stream做decode，低优先级stream做压缩
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda')

        # 获取设备支持的优先级范围
        # CUDA: priority越小优先级越高
        self.priority_range = torch.cuda.get_device_properties(self.device).major

        # 创建不同优先级的streams
        # 高优先级: decode（必须快速响应）
        self.decode_stream = torch.cuda.Stream(
            device=self.device,
            priority=-1  # 最高优先级
        )

        # 低优先级: 压缩（可以被抢占）
        self.compress_stream = torch.cuda.Stream(
            device=self.device,
            priority=0   # 最低优先级
        )

        # 同步事件
        self.compress_events: Dict[int, torch.cuda.Event] = {}

        # 统计信息
        self.stats = {
            'compress_overlapped_time': 0.0,
            'compress_standalone_time': 0.0,
            'decode_time': 0.0,
            'overlap_ratio': 0.0,
        }

    def get_decode_stream(self) -> torch.cuda.Stream:
        """获取decode stream（高优先级）"""
        return self.decode_stream

    def get_compress_stream(self) -> torch.cuda.Stream:
        """获取compress stream（低优先级）"""
        return self.compress_stream

    def record_compress_event(self, seq_id: int) -> torch.cuda.Event:
        """记录压缩完成事件"""
        event = torch.cuda.Event()
        event.record(self.compress_stream)
        self.compress_events[seq_id] = event
        return event

    def wait_compress_complete(self, seq_id: int, stream: torch.cuda.Stream = None):
        """等待特定序列的压缩完成"""
        if seq_id in self.compress_events:
            event = self.compress_events.pop(seq_id)
            if stream is not None:
                stream.wait_event(event)
            else:
                event.synchronize()


class LayerwiseCompressor(nn.Module):
    """
    分层压缩器

    将压缩任务拆分成多个小块，每块处理几层
    这样可以更细粒度地塞入decode间隙
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        compressed_dim: int,
        num_kv_heads: int,
        head_dim: int,
        layers_per_chunk: int = 4,  # 每个chunk处理的层数
        dtype: torch.dtype = torch.float16,
        device: torch.device = None
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.compressed_dim = compressed_dim
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layers_per_chunk = layers_per_chunk
        self.dtype = dtype
        self.device = device or torch.device('cuda')

        # 计算chunk数量
        self.num_chunks = (num_layers + layers_per_chunk - 1) // layers_per_chunk

        # 每层的压缩权重
        # 分开存储以支持分层压缩
        self.compress_weights_k = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dim, compressed_dim, dtype=dtype, device=device))
            for _ in range(num_layers)
        ])
        self.compress_weights_v = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dim, compressed_dim, dtype=dtype, device=device))
            for _ in range(num_layers)
        ])

        # 初始化权重
        for w in self.compress_weights_k:
            nn.init.xavier_uniform_(w)
        for w in self.compress_weights_v:
            nn.init.xavier_uniform_(w)

    def get_chunk_layers(self, chunk_idx: int) -> Tuple[int, int]:
        """获取chunk对应的层范围"""
        start = chunk_idx * self.layers_per_chunk
        end = min(start + self.layers_per_chunk, self.num_layers)
        return start, end

    def compress_chunk(
        self,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        chunk_idx: int,
        stream: torch.cuda.Stream = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        压缩单个chunk的KV-cache

        Args:
            kv_cache: 完整的KV-cache
            chunk_idx: chunk索引
            stream: 使用的CUDA stream

        Returns:
            压缩后的KV-cache（只包含该chunk的层）
        """
        start_layer, end_layer = self.get_chunk_layers(chunk_idx)

        compressed_chunk = []

        with torch.cuda.stream(stream) if stream else torch.cuda.stream(torch.cuda.current_stream()):
            for layer_idx in range(start_layer, end_layer):
                if layer_idx >= len(kv_cache):
                    break

                k, v = kv_cache[layer_idx]
                # k, v shape: [batch, heads, seq, dim]

                # 压缩K
                # [batch, heads, seq, dim] -> [batch, heads, seq, compressed_dim]
                batch, heads, seq_len, dim = k.shape
                k_flat = k.permute(0, 2, 1, 3).reshape(batch * seq_len, heads * dim)
                k_compressed = torch.mm(k_flat, self.compress_weights_k[layer_idx])
                k_compressed = k_compressed.view(batch, seq_len, heads, -1).permute(0, 2, 1, 3)

                # 压缩V
                v_flat = v.permute(0, 2, 1, 3).reshape(batch * seq_len, heads * dim)
                v_compressed = torch.mm(v_flat, self.compress_weights_v[layer_idx])
                v_compressed = v_compressed.view(batch, seq_len, heads, -1).permute(0, 2, 1, 3)

                compressed_chunk.append((k_compressed, v_compressed))

        return compressed_chunk

    def compress_all(
        self,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        stream: torch.cuda.Stream = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """压缩所有层"""
        compressed = []
        for chunk_idx in range(self.num_chunks):
            chunk_result = self.compress_chunk(kv_cache, chunk_idx, stream)
            compressed.extend(chunk_result)
        return compressed


class ZeroOverheadScheduler:
    """
    零开销调度器

    核心策略：
    1. 监测decode负载（通过batch size和KV-cache长度估算）
    2. 当decode是memory-bound时，发射压缩任务
    3. 分层压缩，每个decode step后塞入一个压缩chunk
    """

    def __init__(
        self,
        compressor: 'BatchedGEMMCompressor',
        stream_manager: StreamPriorityManager,
        layers_per_step: int = 8,  # 每个decode step压缩的层数
    ):
        self.compressor = compressor
        self.stream_manager = stream_manager
        self.layers_per_step = layers_per_step

        # 压缩队列
        self.pending_compressions: Dict[int, dict] = {}  # seq_id -> compression state

        # 性能统计
        self.stats = {
            'total_compressions': 0,
            'overlapped_compressions': 0,
            'overlap_time_saved': 0.0,
        }

    def estimate_decode_load(
        self,
        batch_size: int,
        avg_context_len: int,
        num_kv_heads: int = 8,
        head_dim: int = 128
    ) -> float:
        """
        估算decode负载

        返回值：0-1之间，越高表示越memory-bound

        Memory bandwidth estimation:
        - 读取KV-cache: batch_size * avg_context_len * num_kv_heads * head_dim * 2 * 2 bytes
        - A100带宽: ~2TB/s

        Compute estimation:
        - QK^T: batch_size * num_heads * avg_context_len FLOPs
        - A100算力: ~300 TFLOPS (FP16)
        """
        # 每次decode的内存访问量 (bytes)
        kv_read_bytes = batch_size * avg_context_len * num_kv_heads * head_dim * 2 * 2  # K+V, FP16

        # 每次decode的计算量 (FLOPs)
        # 简化估算：主要是attention的QK^T和softmax*V
        compute_flops = batch_size * 32 * avg_context_len * head_dim * 2  # 32 heads

        # 估算intensity (FLOPs/byte)
        intensity = compute_flops / max(kv_read_bytes, 1)

        # A100的roofline拐点大约是 150 FLOPs/byte
        # intensity < 150 意味着memory-bound
        memory_bound_ratio = min(1.0, 150.0 / max(intensity, 1.0))

        return memory_bound_ratio

    def should_compress_now(
        self,
        batch_size: int,
        avg_context_len: int
    ) -> bool:
        """判断是否应该现在发射压缩任务"""
        load = self.estimate_decode_load(batch_size, avg_context_len)
        # 当memory-bound比例 > 0.7时，说明有足够的计算间隙
        return load > 0.7

    def schedule_compression(
        self,
        seq_id: int,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        it_len: List[int]
    ):
        """
        调度压缩任务

        不立即执行，而是分片存储，等待decode间隙
        """
        self.pending_compressions[seq_id] = {
            'kv_cache': kv_cache,
            'it_len': it_len,
            'current_layer': 0,
            'compressed_result': [None] * len(kv_cache),
            'state': CompressionState.PENDING,
        }

    def execute_compression_chunk(
        self,
        seq_id: int,
        num_layers: int = None
    ) -> bool:
        """
        执行一个压缩chunk

        在decode间隙调用，使用低优先级stream

        Returns:
            是否完成所有层的压缩
        """
        if seq_id not in self.pending_compressions:
            return True

        task = self.pending_compressions[seq_id]
        if task['state'] == CompressionState.COMPLETED:
            return True

        task['state'] = CompressionState.IN_PROGRESS

        num_layers = num_layers or self.layers_per_step
        start_layer = task['current_layer']
        end_layer = min(start_layer + num_layers, len(task['kv_cache']))

        # 在低优先级stream上执行压缩
        compress_stream = self.stream_manager.get_compress_stream()

        with torch.cuda.stream(compress_stream):
            for layer_idx in range(start_layer, end_layer):
                k, v = task['kv_cache'][layer_idx]

                # 使用batched compressor的单层压缩
                with torch.no_grad():
                    compressed_k, compressed_v = self._compress_single_layer(
                        k, v, layer_idx, task['it_len']
                    )
                task['compressed_result'][layer_idx] = (compressed_k, compressed_v)

        task['current_layer'] = end_layer

        if end_layer >= len(task['kv_cache']):
            task['state'] = CompressionState.COMPLETED
            self.stream_manager.record_compress_event(seq_id)
            self.stats['total_compressions'] += 1
            self.stats['overlapped_compressions'] += 1
            return True

        return False

    def _compress_single_layer(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        it_len: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """压缩单层（使用现有compressor的权重）"""
        # 这里调用batched_compressor的单层压缩逻辑
        # 简化实现：直接用完整压缩器
        return self.compressor.compress_layer(k, v, layer_idx, it_len)

    def get_compressed_result(self, seq_id: int) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """获取压缩结果"""
        if seq_id not in self.pending_compressions:
            return None

        task = self.pending_compressions[seq_id]
        if task['state'] != CompressionState.COMPLETED:
            return None

        result = task['compressed_result']
        del self.pending_compressions[seq_id]
        return result

    def is_compression_complete(self, seq_id: int) -> bool:
        """检查压缩是否完成"""
        if seq_id not in self.pending_compressions:
            return True
        return self.pending_compressions[seq_id]['state'] == CompressionState.COMPLETED


class ZeroOverheadCompressEngine:
    """
    零开销压缩引擎

    集成到LlavaEngine中，实现真正的零开销压缩
    """

    def __init__(
        self,
        compressor: nn.Module,
        num_layers: int = 32,
        layers_per_decode_step: int = 4,  # 每个decode step压缩几层
        device: torch.device = None
    ):
        self.device = device or torch.device('cuda')
        self.compressor = compressor
        self.num_layers = num_layers
        self.layers_per_decode_step = layers_per_decode_step

        # Stream管理
        self.stream_manager = StreamPriorityManager(self.device)

        # 压缩状态追踪
        self.compression_states: Dict[int, dict] = {}

        # 性能统计
        self.perf_stats = {
            'decode_only_time': 0.0,
            'decode_with_compress_time': 0.0,
            'compress_standalone_time': 0.0,
            'overlap_efficiency': 0.0,
        }

    def start_compression(
        self,
        seq_id: int,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        it_len: List[int]
    ):
        """
        启动压缩任务（不阻塞）

        压缩会分散到后续的decode步骤中执行
        """
        self.compression_states[seq_id] = {
            'kv_cache': kv_cache,
            'it_len': it_len,
            'next_layer': 0,
            'compressed': [None] * len(kv_cache),
            'start_time': time.time(),
            'complete': False,
        }

    def step_compression(self, seq_id: int) -> bool:
        """
        执行一步压缩（在decode之后调用）

        利用decode后的计算间隙执行压缩

        Returns:
            是否完成所有压缩
        """
        if seq_id not in self.compression_states:
            return True

        state = self.compression_states[seq_id]
        if state['complete']:
            return True

        # 在低优先级stream上执行
        compress_stream = self.stream_manager.get_compress_stream()

        start_layer = state['next_layer']
        end_layer = min(start_layer + self.layers_per_decode_step, self.num_layers)

        with torch.cuda.stream(compress_stream):
            with torch.no_grad():
                for layer_idx in range(start_layer, end_layer):
                    if layer_idx < len(state['kv_cache']):
                        k, v = state['kv_cache'][layer_idx]
                        # 调用压缩器
                        compressed = self._compress_layer(k, v, layer_idx, state['it_len'])
                        state['compressed'][layer_idx] = compressed

        state['next_layer'] = end_layer

        if end_layer >= self.num_layers:
            state['complete'] = True
            self.stream_manager.record_compress_event(seq_id)
            elapsed = time.time() - state['start_time']
            self.perf_stats['decode_with_compress_time'] += elapsed
            return True

        return False

    def _compress_layer(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        it_len: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """压缩单层"""
        # 使用compressor的单层压缩
        # 这需要compressor支持分层压缩接口
        if hasattr(self.compressor, 'compress_layer'):
            return self.compressor.compress_layer(k, v, layer_idx, it_len)
        else:
            # Fallback: 完整压缩然后取出该层
            full_compressed = self.compressor([(k, v)], it_len=it_len)
            return full_compressed[0]

    def wait_compression(self, seq_id: int):
        """等待压缩完成"""
        if seq_id in self.compression_states:
            self.stream_manager.wait_compress_complete(seq_id)

    def get_compressed_cache(
        self,
        seq_id: int
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """获取压缩后的KV-cache"""
        if seq_id not in self.compression_states:
            return None

        state = self.compression_states[seq_id]
        if not state['complete']:
            # 等待完成
            self.wait_compression(seq_id)

        result = state['compressed']
        del self.compression_states[seq_id]
        return result

    def get_stats(self) -> dict:
        """获取性能统计"""
        return self.perf_stats.copy()


def create_zero_overhead_compressor(
    original_compressor: nn.Module,
    num_layers: int = 32,
    layers_per_step: int = 4,
    device: torch.device = None
) -> ZeroOverheadCompressEngine:
    """
    创建零开销压缩引擎

    Args:
        original_compressor: 原始压缩器（BatchedGEMMCompressor）
        num_layers: 模型层数
        layers_per_step: 每个decode step压缩的层数
        device: 设备

    Returns:
        ZeroOverheadCompressEngine实例
    """
    return ZeroOverheadCompressEngine(
        compressor=original_compressor,
        num_layers=num_layers,
        layers_per_decode_step=layers_per_step,
        device=device
    )
