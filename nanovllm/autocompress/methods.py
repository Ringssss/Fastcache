"""
压缩方法适配器

提供统一的接口来集成各种KV-Cache压缩方法：
1. MLP-based (BatchedGEMMCompressor)
2. kvpress库的所有方法 (SnapKV, StreamingLLM, H2O, etc.)
3. 量化方法 (INT8, INT4, FP8)

核心设计：
- 每种方法都有对应的Profile，描述其计算特性
- 每种方法都有对应的Adapter，提供统一的调用接口
- AutoCompressScheduler根据Profile选择最优的pipeline策略
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List, Tuple, TYPE_CHECKING
from enum import Enum, auto
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .profiles import (
    CompressMethodProfile,
    ComputeType,
    register_profile,
    get_profile,
    BUILTIN_PROFILES,
)
from .context import WorkloadContext, HardwareProfile
from .execution import ExecutionTiming, ExecutionPlan

if TYPE_CHECKING:
    from ..kernels.batched_compress import BatchedGEMMCompressor


# ============================================================================
# 统一的压缩方法适配器接口
# ============================================================================

class CompressionMethod(ABC):
    """
    压缩方法的统一抽象基类

    所有压缩方法都需要实现这个接口。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """方法名称"""
        pass

    @property
    @abstractmethod
    def profile(self) -> CompressMethodProfile:
        """获取计算特性Profile"""
        pass

    @abstractmethod
    def compress(self,
                 keys: torch.Tensor,
                 values: torch.Tensor,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行压缩

        Args:
            keys: Key张量 [batch, heads, seq_len, head_dim]
            values: Value张量 [batch, heads, seq_len, head_dim]
            **kwargs: 额外参数（如attention_weights, hidden_states等）

        Returns:
            (compressed_keys, compressed_values)
        """
        pass

    def get_optimal_timing(self,
                           workload: WorkloadContext,
                           hw_profile: HardwareProfile) -> ExecutionTiming:
        """
        获取最优执行时机

        默认实现，子类可以覆盖以提供更精确的决策
        """
        profile = self.profile

        # 需要attention结果的方法
        if profile.requires_attention:
            compress_latency = profile.estimate_latency(workload.seq_len, workload.batch_size)
            if compress_latency < 1.0:
                return ExecutionTiming.INLINE_PREFILL
            return ExecutionTiming.AFTER_PREFILL

        # GEMM类型方法
        if profile.compute_type == ComputeType.GEMM:
            decode_memory_bound = hw_profile.is_decode_memory_bound(
                workload.batch_size, workload.seq_len,
                workload.num_kv_heads, workload.head_dim
            )
            if decode_memory_bound and profile.decode_compatibility > 0.5:
                return ExecutionTiming.ASYNC_DECODE
            return ExecutionTiming.AFTER_PREFILL

        # Memory类型方法
        if profile.compute_type == ComputeType.MEMORY:
            compress_latency = profile.estimate_latency(workload.seq_len, workload.batch_size)
            if compress_latency < 0.5:
                return ExecutionTiming.INLINE_PREFILL
            return ExecutionTiming.AFTER_PREFILL

        # Mixed类型
        return ExecutionTiming.AFTER_PREFILL


# ============================================================================
# MLP-based 压缩方法适配器
# ============================================================================

class MLPCompressionMethod(CompressionMethod):
    """
    MLP-based KV-Cache压缩

    使用学习的MLP网络将KV-Cache映射到低维空间。
    计算特性：GEMM密集型，可与Memory-bound的Decode并行。

    最优Pipeline策略：
    - 大batch + 长序列：ASYNC_DECODE（与Decode并行）
    - 小batch：AFTER_PREFILL（同步执行）
    """

    def __init__(self,
                 compressor: Optional[nn.Module] = None,
                 compression_ratio: float = 5.0,
                 num_layers: int = 32):
        self._compressor = compressor
        self._compression_ratio = compression_ratio
        self._num_layers = num_layers

        # 创建Profile
        self._profile = self._create_profile()

        # 注册到全局
        if 'mlp_compress' not in BUILTIN_PROFILES:
            register_profile(self._profile)

    def _create_profile(self) -> CompressMethodProfile:
        """创建MLP压缩的Profile"""

        def mlp_latency(seq_len: int, batch_size: int) -> float:
            # MLP压缩是GEMM密集型
            # 32层 × 3个GEMM × (seq_len × hidden_dim)
            flops_per_token = self._num_layers * 3 * 640 * 128 * 2
            base_throughput = 150e12  # A100 FP16 ~150 TFLOPS
            efficiency = min(1.0, 0.3 + 0.7 * (1 - math.exp(-batch_size / 8)))
            total_flops = seq_len * batch_size * flops_per_token
            return (total_flops / (base_throughput * efficiency)) * 1000

        return CompressMethodProfile(
            name='mlp_compress',
            compute_type=ComputeType.GEMM,
            decode_compatibility=0.95,  # 几乎完美互补
            chunkable=True,
            min_chunk_size=4,
            max_chunk_size=8,
            requires_attention=False,
            modifies_seq_len=True,
            compression_ratio=self._compression_ratio,
            _custom_latency_fn=mlp_latency,
        )

    @property
    def name(self) -> str:
        return 'mlp_compress'

    @property
    def profile(self) -> CompressMethodProfile:
        return self._profile

    def compress(self,
                 keys: torch.Tensor,
                 values: torch.Tensor,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._compressor is None:
            raise RuntimeError("MLP compressor not initialized")

        # 调用BatchedGEMMCompressor
        # 需要适配输入格式
        kv_cache = torch.stack([keys, values], dim=0)
        compressed = self._compressor(kv_cache, **kwargs)

        if isinstance(compressed, tuple):
            return compressed[0], compressed[1]
        else:
            # 假设输出格式与输入相同
            return compressed[0], compressed[1]

    def get_optimal_timing(self,
                           workload: WorkloadContext,
                           hw_profile: HardwareProfile) -> ExecutionTiming:
        """MLP特定的最优时机决策"""

        # 计算Decode的Memory-bound程度
        decode_memory_bound = hw_profile.is_decode_memory_bound(
            workload.batch_size, workload.seq_len,
            workload.num_kv_heads, workload.head_dim
        )

        # 小batch：即使Decode理论上是memory-bound，
        # 实际上小GEMM效率也低，不如同步执行
        if workload.batch_size < 4:
            return ExecutionTiming.AFTER_PREFILL

        # 大batch + Memory-bound Decode：异步并行
        if decode_memory_bound and workload.batch_size >= 4:
            return ExecutionTiming.ASYNC_DECODE

        return ExecutionTiming.AFTER_PREFILL


# ============================================================================
# kvpress库方法适配器
# ============================================================================

class KVPressMethod(CompressionMethod):
    """
    kvpress库方法的通用适配器

    支持的方法：
    - SnapKVPress: 基于窗口attention的稀疏化
    - StreamingLLMPress: Sink + Recent窗口
    - ObservedAttentionPress: 利用已计算的attention
    - ExpectedAttentionPress: 预测未来attention
    - KnormPress: 基于Key范数
    - TOVAPress: 最后token的attention
    - ThinKPress: 维度压缩
    - SimLayerKVPress: 分层自适应
    - H2O: Heavy-Hitter Oracle
    """

    # 方法到Profile的映射
    METHOD_PROFILES = {
        'snapkv': {
            'compute_type': ComputeType.MIXED,
            'decode_compatibility': 0.3,
            'requires_attention': False,  # 可以自己计算
            'modifies_seq_len': True,
            'optimal_timing': ExecutionTiming.INLINE_PREFILL,
        },
        'streaming_llm': {
            'compute_type': ComputeType.MEMORY,
            'decode_compatibility': 0.1,
            'requires_attention': False,
            'modifies_seq_len': True,
            'optimal_timing': ExecutionTiming.INLINE_PREFILL,
        },
        'observed_attention': {
            'compute_type': ComputeType.MEMORY,
            'decode_compatibility': 0.2,
            'requires_attention': True,
            'modifies_seq_len': True,
            'optimal_timing': ExecutionTiming.INLINE_PREFILL,
        },
        'expected_attention': {
            'compute_type': ComputeType.MIXED,
            'decode_compatibility': 0.3,
            'requires_attention': False,
            'modifies_seq_len': True,
            'optimal_timing': ExecutionTiming.AFTER_PREFILL,
        },
        'knorm': {
            'compute_type': ComputeType.MEMORY,
            'decode_compatibility': 0.1,
            'requires_attention': False,
            'modifies_seq_len': True,
            'optimal_timing': ExecutionTiming.INLINE_PREFILL,
        },
        'tova': {
            'compute_type': ComputeType.MIXED,
            'decode_compatibility': 0.25,
            'requires_attention': True,
            'modifies_seq_len': True,
            'optimal_timing': ExecutionTiming.INLINE_PREFILL,
        },
        'think': {
            'compute_type': ComputeType.MIXED,
            'decode_compatibility': 0.3,
            'requires_attention': False,
            'modifies_seq_len': False,  # 只压缩维度
            'modifies_precision': False,
            'optimal_timing': ExecutionTiming.AFTER_PREFILL,
        },
        'simlayer': {
            'compute_type': ComputeType.MIXED,
            'decode_compatibility': 0.35,
            'requires_attention': False,
            'modifies_seq_len': True,
            'optimal_timing': ExecutionTiming.AFTER_PREFILL,
        },
        'h2o': {
            'compute_type': ComputeType.MIXED,
            'decode_compatibility': 0.4,
            'requires_attention': True,
            'modifies_seq_len': True,
            'optimal_timing': ExecutionTiming.INLINE_PREFILL,
        },
        'random': {
            'compute_type': ComputeType.MEMORY,
            'decode_compatibility': 0.1,
            'requires_attention': False,
            'modifies_seq_len': True,
            'optimal_timing': ExecutionTiming.INLINE_PREFILL,
        },
    }

    def __init__(self,
                 method_name: str,
                 compression_ratio: float = 0.5,
                 **method_kwargs):
        """
        Args:
            method_name: kvpress方法名称 (snapkv, streaming_llm, etc.)
            compression_ratio: 压缩比 (0-1, 表示保留的比例)
            **method_kwargs: 传递给kvpress方法的额外参数
        """
        self._method_name = method_name.lower()
        self._compression_ratio = compression_ratio
        self._method_kwargs = method_kwargs

        # 延迟初始化kvpress对象
        self._press = None

        # 创建Profile
        self._profile = self._create_profile()

    def _create_profile(self) -> CompressMethodProfile:
        """根据方法名创建Profile"""

        config = self.METHOD_PROFILES.get(self._method_name, {
            'compute_type': ComputeType.MIXED,
            'decode_compatibility': 0.3,
            'requires_attention': False,
            'modifies_seq_len': True,
            'optimal_timing': ExecutionTiming.AFTER_PREFILL,
        })

        # 创建延迟估算函数
        def latency_fn(seq_len: int, batch_size: int) -> float:
            if self._method_name == 'snapkv':
                # SnapKV: 窗口attention + TopK + Gather
                window_size = self._method_kwargs.get('window_size', 64)
                attention_latency = (window_size * seq_len * batch_size) / 1e9 * 0.1
                topk_latency = seq_len * math.log2(max(seq_len, 2)) / 1e9 * 0.5
                return (attention_latency + topk_latency) * 1000

            elif self._method_name == 'streaming_llm':
                # StreamingLLM: 几乎零开销
                return 0.1

            elif self._method_name == 'observed_attention':
                # 只需要聚合已有的attention权重
                return seq_len * batch_size / 1e7

            elif self._method_name == 'expected_attention':
                # 需要计算协方差矩阵，较重
                head_dim = 128
                return seq_len * batch_size * head_dim / 1e8

            elif self._method_name == 'knorm':
                # 计算范数，轻量
                return seq_len * batch_size / 1e8

            elif self._method_name == 'tova':
                # 最后token的attention
                return seq_len * batch_size / 1e7

            elif self._method_name == 'think':
                # 维度压缩
                return seq_len * batch_size / 1e7

            elif self._method_name == 'simlayer':
                # 分层检测 + 可能的压缩
                return seq_len * batch_size / 1e7 * 1.5

            elif self._method_name == 'h2o':
                # 累积统计 + TopK
                return seq_len * batch_size / 1e6 * 0.3

            else:
                # 默认估算
                return seq_len * batch_size / 1e7

        return CompressMethodProfile(
            name=f'kvpress_{self._method_name}',
            compute_type=config['compute_type'],
            decode_compatibility=config['decode_compatibility'],
            chunkable=False,  # kvpress方法通常不支持分块
            requires_attention=config['requires_attention'],
            modifies_seq_len=config.get('modifies_seq_len', True),
            modifies_precision=config.get('modifies_precision', False),
            compression_ratio=1.0 / (1.0 - self._compression_ratio),  # 转换比例
            _custom_latency_fn=latency_fn,
        )

    def _init_press(self):
        """延迟初始化kvpress对象"""
        if self._press is not None:
            return

        try:
            import kvpress

            press_map = {
                'snapkv': kvpress.SnapKVPress,
                'streaming_llm': kvpress.StreamingLLMPress,
                'observed_attention': kvpress.ObservedAttentionPress,
                'expected_attention': kvpress.ExpectedAttentionPress,
                'knorm': kvpress.KnormPress,
                'tova': kvpress.TOVAPress,
                'think': kvpress.ThinKPress,
                'simlayer': kvpress.SimLayerKVPress,
                'random': kvpress.RandomPress,
            }

            press_class = press_map.get(self._method_name)
            if press_class is None:
                raise ValueError(f"Unknown kvpress method: {self._method_name}")

            # 创建press实例
            if self._method_name == 'think':
                self._press = press_class(
                    key_channel_compression_ratio=self._compression_ratio,
                    **self._method_kwargs
                )
            else:
                self._press = press_class(
                    compression_ratio=self._compression_ratio,
                    **self._method_kwargs
                )
        except ImportError:
            raise ImportError("kvpress library is required for this method")

    @property
    def name(self) -> str:
        return f'kvpress_{self._method_name}'

    @property
    def profile(self) -> CompressMethodProfile:
        return self._profile

    def compress(self,
                 keys: torch.Tensor,
                 values: torch.Tensor,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行kvpress压缩

        注意：kvpress设计用于forward hook模式，
        这里提供独立调用的适配。
        """
        self._init_press()

        # kvpress需要module参数，这里创建一个mock
        module = kwargs.get('module', None)
        hidden_states = kwargs.get('hidden_states', None)
        attentions = kwargs.get('attentions', None)

        if module is None:
            # 创建一个简单的mock module
            class MockModule:
                def __init__(self, head_dim, num_heads, num_key_value_heads):
                    self.head_dim = head_dim
                    self.num_heads = num_heads
                    self.num_key_value_heads = num_key_value_heads
                    self.num_key_value_groups = num_heads // num_key_value_heads

            _, num_kv_heads, seq_len, head_dim = keys.shape
            num_heads = kwargs.get('num_heads', num_kv_heads * 4)
            module = MockModule(head_dim, num_heads, num_kv_heads)

        # 调用kvpress的compress方法
        compressed_keys, compressed_values = self._press.compress(
            module=module,
            hidden_states=hidden_states,
            keys=keys,
            values=values,
            attentions=attentions,
            kwargs=kwargs,
        )

        return compressed_keys, compressed_values

    def get_optimal_timing(self,
                           workload: WorkloadContext,
                           hw_profile: HardwareProfile) -> ExecutionTiming:
        """获取最优执行时机"""
        config = self.METHOD_PROFILES.get(self._method_name, {})
        return config.get('optimal_timing', ExecutionTiming.AFTER_PREFILL)


# ============================================================================
# 量化方法适配器
# ============================================================================

class QuantizationMethod(CompressionMethod):
    """
    KV-Cache量化压缩

    支持的量化方法：
    - INT8: 8位整数量化
    - INT4: 4位整数量化 (AWQ/GPTQ风格)
    - FP8: 8位浮点量化 (E4M3/E5M2)
    - NF4: 4位Normal Float量化 (QLoRA风格)

    最优Pipeline策略：
    - 量化操作本身是Memory-bound，与Decode有竞争
    - 推荐在Prefill期间逐层量化，或Prefill后同步执行
    - Decode时需要融合反量化进Attention kernel
    """

    class QuantType(Enum):
        INT8 = auto()
        INT4 = auto()
        FP8_E4M3 = auto()
        FP8_E5M2 = auto()
        NF4 = auto()

    def __init__(self,
                 quant_type: str = 'int8',
                 group_size: int = 128,
                 symmetric: bool = True):
        """
        Args:
            quant_type: 量化类型 (int8, int4, fp8_e4m3, fp8_e5m2, nf4)
            group_size: 量化分组大小（用于per-group量化）
            symmetric: 是否对称量化
        """
        self._quant_type = self._parse_quant_type(quant_type)
        self._group_size = group_size
        self._symmetric = symmetric

        self._profile = self._create_profile()

    def _parse_quant_type(self, quant_type: str) -> 'QuantizationMethod.QuantType':
        mapping = {
            'int8': self.QuantType.INT8,
            'int4': self.QuantType.INT4,
            'fp8': self.QuantType.FP8_E4M3,
            'fp8_e4m3': self.QuantType.FP8_E4M3,
            'fp8_e5m2': self.QuantType.FP8_E5M2,
            'nf4': self.QuantType.NF4,
        }
        return mapping.get(quant_type.lower(), self.QuantType.INT8)

    def _create_profile(self) -> CompressMethodProfile:
        """创建量化方法的Profile"""

        # 不同量化类型的压缩比
        compression_ratios = {
            self.QuantType.INT8: 2.0,      # FP16 -> INT8
            self.QuantType.INT4: 4.0,      # FP16 -> INT4
            self.QuantType.FP8_E4M3: 2.0,  # FP16 -> FP8
            self.QuantType.FP8_E5M2: 2.0,
            self.QuantType.NF4: 4.0,       # FP16 -> NF4
        }

        def quant_latency(seq_len: int, batch_size: int) -> float:
            # 量化是memory-bound的打包操作
            elements = seq_len * batch_size * 32 * 128  # heads × head_dim

            if self._quant_type in [self.QuantType.INT4, self.QuantType.NF4]:
                # 4位量化需要更多处理
                ops_per_element = 3  # scale计算 + 量化 + 打包
            else:
                ops_per_element = 2  # scale计算 + 量化

            throughput = 500e9  # ~500GB/s内存带宽
            return (elements * 2 * ops_per_element / throughput) * 1000

        return CompressMethodProfile(
            name=f'quant_{self._quant_type.name.lower()}',
            compute_type=ComputeType.MEMORY,
            decode_compatibility=0.2,  # 与Decode竞争内存带宽
            chunkable=True,
            min_chunk_size=1,
            max_chunk_size=32,
            requires_attention=False,
            modifies_seq_len=False,
            modifies_precision=True,
            compression_ratio=compression_ratios[self._quant_type],
            _custom_latency_fn=quant_latency,
        )

    @property
    def name(self) -> str:
        return f'quant_{self._quant_type.name.lower()}'

    @property
    def profile(self) -> CompressMethodProfile:
        return self._profile

    def compress(self,
                 keys: torch.Tensor,
                 values: torch.Tensor,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行量化压缩

        Returns:
            量化后的(keys, values)
            注意：返回的是量化后的张量和scale因子的打包
        """
        quantized_keys, key_scales = self._quantize(keys)
        quantized_values, value_scales = self._quantize(values)

        # 将scale信息附加到张量
        # 实际实现中可能需要更复杂的打包方式
        return (quantized_keys, key_scales), (quantized_values, value_scales)

    def _quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行量化"""

        if self._quant_type == self.QuantType.INT8:
            return self._quantize_int8(tensor)
        elif self._quant_type == self.QuantType.INT4:
            return self._quantize_int4(tensor)
        elif self._quant_type in [self.QuantType.FP8_E4M3, self.QuantType.FP8_E5M2]:
            return self._quantize_fp8(tensor)
        elif self._quant_type == self.QuantType.NF4:
            return self._quantize_nf4(tensor)
        else:
            raise ValueError(f"Unknown quantization type: {self._quant_type}")

    def _quantize_int8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """INT8量化"""
        # Per-channel量化
        if self._symmetric:
            scale = tensor.abs().amax(dim=-1, keepdim=True) / 127.0
            scale = scale.clamp(min=1e-8)
            quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        else:
            x_min = tensor.amin(dim=-1, keepdim=True)
            x_max = tensor.amax(dim=-1, keepdim=True)
            scale = (x_max - x_min) / 255.0
            scale = scale.clamp(min=1e-8)
            zero_point = (-x_min / scale).round().clamp(0, 255)
            quantized = ((tensor - x_min) / scale).round().clamp(0, 255).to(torch.uint8)
            scale = torch.cat([scale, zero_point], dim=-1)

        return quantized, scale

    def _quantize_int4(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """INT4量化 (AWQ风格)"""
        # Per-group量化
        original_shape = tensor.shape
        tensor = tensor.view(-1, self._group_size)

        scale = tensor.abs().amax(dim=-1, keepdim=True) / 7.0
        scale = scale.clamp(min=1e-8)

        quantized = (tensor / scale).round().clamp(-8, 7)

        # 打包两个INT4到一个INT8
        quantized = quantized.view(-1, 2)
        packed = ((quantized[:, 0] + 8) | ((quantized[:, 1] + 8) << 4)).to(torch.uint8)

        return packed.view(*original_shape[:-1], -1), scale.view(*original_shape[:-1], -1)

    def _quantize_fp8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """FP8量化"""
        # 使用PyTorch的float8支持（如果可用）
        try:
            if self._quant_type == self.QuantType.FP8_E4M3:
                dtype = torch.float8_e4m3fn
            else:
                dtype = torch.float8_e5m2

            scale = tensor.abs().amax() / 448.0  # FP8 E4M3的最大值
            scale = scale.clamp(min=1e-8)
            quantized = (tensor / scale).to(dtype)
            return quantized, scale.expand_as(tensor[..., :1])
        except AttributeError:
            # PyTorch版本不支持float8，回退到模拟
            scale = tensor.abs().amax(dim=-1, keepdim=True) / 448.0
            scale = scale.clamp(min=1e-8)
            quantized = (tensor / scale).clamp(-448, 448).to(torch.float16)
            return quantized, scale

    def _quantize_nf4(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """NF4量化 (QLoRA风格)"""
        # NF4使用预定义的分位数
        NF4_QUANT_TABLE = torch.tensor([
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
        ], device=tensor.device, dtype=tensor.dtype)

        # Per-block量化
        original_shape = tensor.shape
        tensor = tensor.view(-1, self._group_size)

        # 计算absmax scale
        scale = tensor.abs().amax(dim=-1, keepdim=True)
        scale = scale.clamp(min=1e-8)
        normalized = tensor / scale

        # 找到最近的NF4值
        diff = (normalized.unsqueeze(-1) - NF4_QUANT_TABLE).abs()
        quantized = diff.argmin(dim=-1).to(torch.uint8)

        # 打包
        quantized = quantized.view(-1, 2)
        packed = (quantized[:, 0] | (quantized[:, 1] << 4)).to(torch.uint8)

        return packed.view(*original_shape[:-1], -1), scale.view(*original_shape[:-1], -1)

    def dequantize(self,
                   quantized: torch.Tensor,
                   scale: torch.Tensor,
                   dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """反量化"""
        if self._quant_type == self.QuantType.INT8:
            return quantized.to(dtype) * scale

        elif self._quant_type == self.QuantType.INT4:
            # 解包INT4
            low = (quantized & 0x0F).to(dtype) - 8
            high = ((quantized >> 4) & 0x0F).to(dtype) - 8
            unpacked = torch.stack([low, high], dim=-1).view(*quantized.shape[:-1], -1)
            return unpacked * scale

        elif self._quant_type in [self.QuantType.FP8_E4M3, self.QuantType.FP8_E5M2]:
            return quantized.to(dtype) * scale

        elif self._quant_type == self.QuantType.NF4:
            NF4_QUANT_TABLE = torch.tensor([
                -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
            ], device=quantized.device, dtype=dtype)

            low = quantized & 0x0F
            high = (quantized >> 4) & 0x0F

            unpacked = torch.stack([
                NF4_QUANT_TABLE[low],
                NF4_QUANT_TABLE[high]
            ], dim=-1).view(*quantized.shape[:-1], -1)

            return unpacked * scale

        else:
            raise ValueError(f"Unknown quantization type: {self._quant_type}")

    def get_optimal_timing(self,
                           workload: WorkloadContext,
                           hw_profile: HardwareProfile) -> ExecutionTiming:
        """量化方法的最优执行时机"""

        # 量化操作本身是memory-bound
        # 但我们需要考虑反量化的开销

        # 短序列：Prefill后同步量化更好
        if workload.seq_len < 1024:
            return ExecutionTiming.AFTER_PREFILL

        # 长序列：考虑在Prefill期间逐层量化
        # 这样可以减少Prefill后的等待时间
        compress_latency = self.profile.estimate_latency(
            workload.seq_len, workload.batch_size
        )

        if compress_latency < 2.0:  # < 2ms可以inline
            return ExecutionTiming.INLINE_PREFILL

        return ExecutionTiming.AFTER_PREFILL


# ============================================================================
# 方法注册表
# ============================================================================

class MethodRegistry:
    """
    压缩方法注册表

    管理所有可用的压缩方法，提供统一的创建接口。
    """

    _methods: Dict[str, type] = {}
    _instances: Dict[str, CompressionMethod] = {}

    @classmethod
    def register(cls, name: str, method_class: type):
        """注册方法类"""
        cls._methods[name] = method_class

    @classmethod
    def create(cls, name: str, **kwargs) -> CompressionMethod:
        """创建方法实例"""

        # 检查缓存
        cache_key = f"{name}_{hash(frozenset(kwargs.items()))}"
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # MLP方法
        if name in ['mlp', 'mlp_compress', 'batched_gemm']:
            instance = MLPCompressionMethod(**kwargs)

        # kvpress方法
        elif name.startswith('kvpress_') or name in KVPressMethod.METHOD_PROFILES:
            method_name = name.replace('kvpress_', '')
            instance = KVPressMethod(method_name, **kwargs)

        # 量化方法
        elif name.startswith('quant_') or name in ['int8', 'int4', 'fp8', 'nf4']:
            quant_type = name.replace('quant_', '')
            instance = QuantizationMethod(quant_type, **kwargs)

        # 从注册表查找
        elif name in cls._methods:
            instance = cls._methods[name](**kwargs)

        else:
            raise ValueError(f"Unknown compression method: {name}")

        cls._instances[cache_key] = instance
        return instance

    @classmethod
    def list_methods(cls) -> List[str]:
        """列出所有可用方法"""
        methods = list(cls._methods.keys())
        methods.extend(['mlp', 'mlp_compress'])
        methods.extend([f'kvpress_{m}' for m in KVPressMethod.METHOD_PROFILES.keys()])
        methods.extend(['quant_int8', 'quant_int4', 'quant_fp8', 'quant_nf4'])
        return sorted(set(methods))


# 便捷函数
def get_method(name: str, **kwargs) -> CompressionMethod:
    """获取压缩方法实例"""
    return MethodRegistry.create(name, **kwargs)


def list_methods() -> List[str]:
    """列出所有可用方法"""
    return MethodRegistry.list_methods()
