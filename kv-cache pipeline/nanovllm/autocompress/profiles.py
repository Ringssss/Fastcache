"""
压缩方法的计算特性画像 (Compute Profile)

每种压缩方法都有固有的计算特性，这是调度决策的核心依据。
通过Profile，调度器可以判断：
1. 该方法与Decode的兼容性（是否可以并行）
2. 最优的执行时机
3. 是否支持分块执行
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Optional, List, Any
import math


class ComputeType(Enum):
    """计算类型"""
    GEMM = auto()      # MLP/SVD等，TensorCore密集型
    MEMORY = auto()    # StreamingLLM/Gather等，带宽密集型
    MIXED = auto()     # SnapKV/H2O等，先计算后访存


@dataclass
class CompressMethodProfile:
    """
    压缩方法的计算特性画像

    这是调度决策的核心依据。每种压缩方法在注册时需要提供这个Profile。

    Attributes:
        name: 压缩方法名称
        compute_type: 计算类型 (GEMM/MEMORY/MIXED)
        decode_compatibility: 与Decode的兼容性 (0-1)
            - 1.0: 完美互补（可Zero-Overhead并行）
            - 0.5: 部分竞争
            - 0.0: 完全竞争（必须串行）
        latency_params: 延迟模型参数
        chunkable: 是否支持分层/分块执行
        min_chunk_size: 最小分块粒度
        requires_attention: 是否需要attention结果
        modifies_seq_len: 是否改变序列长度
        modifies_precision: 是否改变精度
        compression_ratio: 压缩比（输出长度/输入长度）
    """

    name: str
    compute_type: ComputeType
    decode_compatibility: float  # 0-1

    # 延迟模型参数: latency = a * seq_len + b * batch_size + c
    latency_params: Dict[str, float] = field(default_factory=lambda: {
        'seq_coef': 0.0,
        'batch_coef': 0.0,
        'base_latency': 0.0,
    })

    # 分块特性
    chunkable: bool = False
    min_chunk_size: int = 1
    max_chunk_size: int = 32

    # 执行约束
    requires_attention: bool = False
    modifies_seq_len: bool = False
    modifies_precision: bool = False

    # 压缩特性
    compression_ratio: float = 1.0  # 1.0表示无压缩

    # 可选的自定义延迟计算函数
    _custom_latency_fn: Optional[Callable[[int, int], float]] = None

    def estimate_latency(self, seq_len: int, batch_size: int) -> float:
        """
        估算压缩延迟 (ms)

        Args:
            seq_len: 序列长度
            batch_size: 批大小

        Returns:
            预估延迟 (毫秒)
        """
        if self._custom_latency_fn is not None:
            return self._custom_latency_fn(seq_len, batch_size)

        params = self.latency_params
        return (
            params['seq_coef'] * seq_len +
            params['batch_coef'] * batch_size +
            params['base_latency']
        )

    def get_output_seq_len(self, input_seq_len: int) -> int:
        """计算压缩后的序列长度"""
        if not self.modifies_seq_len:
            return input_seq_len
        return int(input_seq_len / self.compression_ratio)

    def can_overlap_with_decode(self, decode_is_memory_bound: bool) -> bool:
        """
        判断是否可以与Decode并行

        关键洞察：
        - GEMM压缩 + Memory-bound Decode → 可并行（互补）
        - Memory压缩 + Memory-bound Decode → 不可并行（竞争）
        """
        if self.compute_type == ComputeType.GEMM:
            return decode_is_memory_bound and self.decode_compatibility > 0.5
        elif self.compute_type == ComputeType.MEMORY:
            return False  # 都是内存瓶颈，竞争
        else:  # MIXED
            return decode_is_memory_bound and self.decode_compatibility > 0.3


# ============ 内置的压缩方法Profile ============

def _mlp_latency(seq_len: int, batch_size: int) -> float:
    """
    MLP压缩延迟模型

    MLP压缩是GEMM密集型：
    - 32层 × 3个GEMM × (seq_len × hidden_dim)
    - 主要受TensorCore吞吐限制
    """
    # 假设: 每个token的压缩开销约0.05ms (在A100上)
    # 实际值需要通过calibration获得
    flops_per_token = 32 * 3 * 640 * 128 * 2  # 32层，3个GEMM，640->128
    base_throughput = 150e12  # A100 FP16 TensorCore吞吐 (150 TFLOPS)

    # batch效率因子（小batch效率低）
    efficiency = min(1.0, 0.3 + 0.7 * (1 - math.exp(-batch_size / 8)))

    total_flops = seq_len * batch_size * flops_per_token
    return (total_flops / (base_throughput * efficiency)) * 1000  # 转换为ms


def _snapkv_latency(seq_len: int, batch_size: int) -> float:
    """
    SnapKV延迟模型

    SnapKV需要：
    1. 计算最后window_size个token的attention
    2. TopK选择
    3. Gather操作
    """
    window_size = 64
    # Attention计算: O(window_size × seq_len)
    attention_latency = (window_size * seq_len * batch_size) / 1e9 * 0.1
    # TopK: O(seq_len log seq_len)
    topk_latency = seq_len * math.log2(max(seq_len, 2)) / 1e9 * 0.5
    # Gather: O(kept_tokens × head_dim)
    gather_latency = (seq_len * 0.5 * 128) / 1e9 * 0.2

    return (attention_latency + topk_latency + gather_latency) * 1000


def _streaming_llm_latency(seq_len: int, batch_size: int) -> float:
    """
    StreamingLLM延迟模型

    几乎零开销：只是索引操作
    """
    return 0.1  # 几乎可以忽略


def _h2o_latency(seq_len: int, batch_size: int) -> float:
    """
    H2O (Heavy-Hitter Oracle) 延迟模型

    需要累积attention权重的统计
    """
    # 累积统计 + TopK
    return seq_len * batch_size / 1e6 * 0.3


def _int4_quant_latency(seq_len: int, batch_size: int) -> float:
    """
    INT4量化延迟模型

    量化是memory-bound的打包操作
    """
    # 每个元素需要pack成INT4
    elements = seq_len * batch_size * 32 * 128  # heads × head_dim
    pack_throughput = 500e9  # ~500GB/s内存带宽
    return (elements * 2 / pack_throughput) * 1000  # FP16->INT4


# 内置Profile注册表
BUILTIN_PROFILES: Dict[str, CompressMethodProfile] = {
    'mlp': CompressMethodProfile(
        name='mlp',
        compute_type=ComputeType.GEMM,
        decode_compatibility=0.95,  # 几乎完美互补
        latency_params={
            'seq_coef': 0.05,
            'batch_coef': 0.01,
            'base_latency': 1.0,
        },
        chunkable=True,
        min_chunk_size=4,
        max_chunk_size=8,
        requires_attention=False,
        modifies_seq_len=True,
        compression_ratio=5.0,
        _custom_latency_fn=_mlp_latency,
    ),

    'snapkv': CompressMethodProfile(
        name='snapkv',
        compute_type=ComputeType.MIXED,
        decode_compatibility=0.3,
        latency_params={
            'seq_coef': 0.01,
            'batch_coef': 0.005,
            'base_latency': 0.5,
        },
        chunkable=False,
        requires_attention=True,
        modifies_seq_len=True,
        compression_ratio=2.0,
        _custom_latency_fn=_snapkv_latency,
    ),

    'streaming_llm': CompressMethodProfile(
        name='streaming_llm',
        compute_type=ComputeType.MEMORY,
        decode_compatibility=0.1,
        latency_params={
            'seq_coef': 0.0,
            'batch_coef': 0.0,
            'base_latency': 0.1,
        },
        chunkable=False,
        requires_attention=False,
        modifies_seq_len=True,
        compression_ratio=2.0,  # 取决于配置
        _custom_latency_fn=_streaming_llm_latency,
    ),

    'h2o': CompressMethodProfile(
        name='h2o',
        compute_type=ComputeType.MIXED,
        decode_compatibility=0.4,
        latency_params={
            'seq_coef': 0.008,
            'batch_coef': 0.003,
            'base_latency': 0.3,
        },
        chunkable=False,
        requires_attention=True,
        modifies_seq_len=True,
        compression_ratio=2.0,
        _custom_latency_fn=_h2o_latency,
    ),

    'int4_quant': CompressMethodProfile(
        name='int4_quant',
        compute_type=ComputeType.MEMORY,
        decode_compatibility=0.2,
        latency_params={
            'seq_coef': 0.02,
            'batch_coef': 0.005,
            'base_latency': 0.2,
        },
        chunkable=True,
        min_chunk_size=1,
        max_chunk_size=32,
        requires_attention=False,
        modifies_seq_len=False,
        modifies_precision=True,
        compression_ratio=4.0,  # FP16->INT4
        _custom_latency_fn=_int4_quant_latency,
    ),

    'int8_quant': CompressMethodProfile(
        name='int8_quant',
        compute_type=ComputeType.MEMORY,
        decode_compatibility=0.25,
        latency_params={
            'seq_coef': 0.015,
            'batch_coef': 0.004,
            'base_latency': 0.15,
        },
        chunkable=True,
        min_chunk_size=1,
        max_chunk_size=32,
        requires_attention=False,
        modifies_seq_len=False,
        modifies_precision=True,
        compression_ratio=2.0,  # FP16->INT8
    ),
}


# 用户自定义Profile注册
_custom_profiles: Dict[str, CompressMethodProfile] = {}


def register_profile(profile: CompressMethodProfile) -> None:
    """
    注册自定义压缩方法Profile

    Example:
        ```python
        my_profile = CompressMethodProfile(
            name='my_compress',
            compute_type=ComputeType.GEMM,
            decode_compatibility=0.8,
            ...
        )
        register_profile(my_profile)
        ```
    """
    _custom_profiles[profile.name] = profile


def get_profile(name: str) -> CompressMethodProfile:
    """获取压缩方法的Profile"""
    if name in _custom_profiles:
        return _custom_profiles[name]
    if name in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name]
    raise ValueError(f"Unknown compression method: {name}. "
                     f"Available: {list(BUILTIN_PROFILES.keys()) + list(_custom_profiles.keys())}")


def list_profiles() -> List[str]:
    """列出所有可用的压缩方法"""
    return list(BUILTIN_PROFILES.keys()) + list(_custom_profiles.keys())
