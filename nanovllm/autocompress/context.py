"""
Workload和Hardware上下文定义

包含：
- WorkloadContext: 当前推理任务的特征
- HardwareProfile: 硬件特性
- 自动硬件检测
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum, auto
import math


class GPUArch(Enum):
    """GPU架构"""
    AMPERE = auto()     # A100, RTX 30xx
    HOPPER = auto()     # H100
    ADA = auto()        # RTX 40xx
    VOLTA = auto()      # V100
    TURING = auto()     # RTX 20xx
    UNKNOWN = auto()


@dataclass
class HardwareProfile:
    """
    硬件特性画像

    用于估算计算强度和内存带宽，指导调度决策。
    """

    # 基础信息
    gpu_name: str = "Unknown"
    gpu_arch: GPUArch = GPUArch.UNKNOWN
    compute_capability: tuple = (0, 0)

    # 计算能力
    fp16_tflops: float = 0.0          # FP16 TensorCore吞吐 (TFLOPS)
    fp32_tflops: float = 0.0          # FP32吞吐 (TFLOPS)
    int8_tops: float = 0.0            # INT8吞吐 (TOPS)

    # 内存
    memory_bandwidth_gbps: float = 0.0  # 内存带宽 (GB/s)
    memory_size_gb: float = 0.0         # 显存大小 (GB)
    l2_cache_mb: float = 0.0            # L2缓存大小 (MB)

    # SM信息
    sm_count: int = 0
    max_threads_per_sm: int = 0

    # 计算得到的roofline点
    _roofline_intensity: Optional[float] = None

    @property
    def roofline_intensity(self) -> float:
        """
        Roofline模型的拐点 (FLOP/Byte)

        当 Arithmetic Intensity < roofline_intensity 时，任务是Memory-bound
        当 Arithmetic Intensity > roofline_intensity 时，任务是Compute-bound
        """
        if self._roofline_intensity is not None:
            return self._roofline_intensity

        if self.fp16_tflops > 0 and self.memory_bandwidth_gbps > 0:
            # roofline = peak_flops / memory_bandwidth
            self._roofline_intensity = (self.fp16_tflops * 1e12) / (self.memory_bandwidth_gbps * 1e9)
            return self._roofline_intensity

        # 默认值（A100的典型值）
        return 150.0

    def is_decode_memory_bound(self, batch_size: int, seq_len: int,
                                num_heads: int = 32, head_dim: int = 128) -> bool:
        """
        判断Decode是否是Memory-bound

        Decode的Arithmetic Intensity = FLOPs / Bytes
        - FLOPs: batch × seq_len × num_heads × head_dim × 2 (Q×K)
        - Bytes: batch × seq_len × num_heads × head_dim × 2 × 2 (K+V, FP16)
        """
        flops = batch_size * seq_len * num_heads * head_dim * 2
        bytes_read = batch_size * seq_len * num_heads * head_dim * 2 * 2

        intensity = flops / max(bytes_read, 1)

        # 如果intensity < roofline的50%，认为是明显的memory-bound
        return intensity < self.roofline_intensity * 0.5

    def estimate_decode_latency_ms(self, batch_size: int, seq_len: int,
                                    num_heads: int = 32, head_dim: int = 128) -> float:
        """
        估算单次Decode延迟 (ms)

        基于Roofline模型
        """
        flops = batch_size * seq_len * num_heads * head_dim * 2
        bytes_read = batch_size * seq_len * num_heads * head_dim * 2 * 2

        intensity = flops / max(bytes_read, 1)

        if intensity < self.roofline_intensity:
            # Memory-bound: latency = bytes / bandwidth
            latency_s = bytes_read / (self.memory_bandwidth_gbps * 1e9)
        else:
            # Compute-bound: latency = flops / throughput
            latency_s = flops / (self.fp16_tflops * 1e12)

        # 加上kernel launch开销
        kernel_overhead_ms = 0.01

        return latency_s * 1000 + kernel_overhead_ms


@dataclass
class WorkloadContext:
    """
    当前推理任务的Workload特征

    调度器基于这些特征做出决策。
    """

    # 基础参数
    batch_size: int = 1
    seq_len: int = 512
    expected_output_len: int = 100

    # 模型参数
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    hidden_size: int = 4096

    # 内存状态
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0

    # 历史统计（用于更准确的预测）
    avg_output_len: Optional[float] = None
    avg_decode_latency_ms: Optional[float] = None

    @property
    def memory_pressure(self) -> float:
        """内存压力 (0-1)"""
        if self.memory_total_gb <= 0:
            return 0.0
        return self.memory_used_gb / self.memory_total_gb

    @property
    def kv_cache_size_mb(self) -> float:
        """当前KV-Cache大小 (MB)"""
        # K + V, FP16 (2 bytes)
        size_bytes = (
            self.batch_size *
            self.seq_len *
            self.num_layers *
            self.num_kv_heads *
            self.head_dim *
            2 *  # K + V
            2    # FP16
        )
        return size_bytes / (1024 * 1024)

    def estimate_compressed_kv_size_mb(self, compression_ratio: float) -> float:
        """估算压缩后的KV-Cache大小"""
        return self.kv_cache_size_mb / compression_ratio

    def get_total_decode_time_ms(self, decode_latency_per_token_ms: float) -> float:
        """估算总Decode时间"""
        output_len = self.expected_output_len
        if self.avg_output_len is not None:
            output_len = int(self.avg_output_len)
        return decode_latency_per_token_ms * output_len


# ============ 硬件自动检测 ============

# 预定义的GPU Profile
_GPU_PROFILES: Dict[str, HardwareProfile] = {
    # NVIDIA A100
    'A100': HardwareProfile(
        gpu_name='NVIDIA A100',
        gpu_arch=GPUArch.AMPERE,
        compute_capability=(8, 0),
        fp16_tflops=312.0,
        fp32_tflops=156.0,
        int8_tops=624.0,
        memory_bandwidth_gbps=2039.0,
        memory_size_gb=80.0,
        l2_cache_mb=40.0,
        sm_count=108,
        max_threads_per_sm=2048,
    ),

    # NVIDIA A100 40GB
    'A100-40GB': HardwareProfile(
        gpu_name='NVIDIA A100 40GB',
        gpu_arch=GPUArch.AMPERE,
        compute_capability=(8, 0),
        fp16_tflops=312.0,
        fp32_tflops=156.0,
        int8_tops=624.0,
        memory_bandwidth_gbps=1555.0,
        memory_size_gb=40.0,
        l2_cache_mb=40.0,
        sm_count=108,
        max_threads_per_sm=2048,
    ),

    # NVIDIA H100 SXM
    'H100': HardwareProfile(
        gpu_name='NVIDIA H100',
        gpu_arch=GPUArch.HOPPER,
        compute_capability=(9, 0),
        fp16_tflops=989.0,
        fp32_tflops=494.0,
        int8_tops=1979.0,
        memory_bandwidth_gbps=3350.0,
        memory_size_gb=80.0,
        l2_cache_mb=50.0,
        sm_count=132,
        max_threads_per_sm=2048,
    ),

    # NVIDIA RTX 4090
    'RTX4090': HardwareProfile(
        gpu_name='NVIDIA RTX 4090',
        gpu_arch=GPUArch.ADA,
        compute_capability=(8, 9),
        fp16_tflops=165.0,
        fp32_tflops=82.6,
        int8_tops=330.0,
        memory_bandwidth_gbps=1008.0,
        memory_size_gb=24.0,
        l2_cache_mb=72.0,
        sm_count=128,
        max_threads_per_sm=1536,
    ),

    # NVIDIA RTX 4080
    'RTX4080': HardwareProfile(
        gpu_name='NVIDIA RTX 4080',
        gpu_arch=GPUArch.ADA,
        compute_capability=(8, 9),
        fp16_tflops=97.5,
        fp32_tflops=48.7,
        int8_tops=195.0,
        memory_bandwidth_gbps=716.8,
        memory_size_gb=16.0,
        l2_cache_mb=64.0,
        sm_count=76,
        max_threads_per_sm=1536,
    ),

    # NVIDIA RTX 3090
    'RTX3090': HardwareProfile(
        gpu_name='NVIDIA RTX 3090',
        gpu_arch=GPUArch.AMPERE,
        compute_capability=(8, 6),
        fp16_tflops=71.0,
        fp32_tflops=35.6,
        int8_tops=142.0,
        memory_bandwidth_gbps=936.2,
        memory_size_gb=24.0,
        l2_cache_mb=6.0,
        sm_count=82,
        max_threads_per_sm=1536,
    ),

    # NVIDIA V100
    'V100': HardwareProfile(
        gpu_name='NVIDIA V100',
        gpu_arch=GPUArch.VOLTA,
        compute_capability=(7, 0),
        fp16_tflops=125.0,
        fp32_tflops=15.7,
        int8_tops=62.0,
        memory_bandwidth_gbps=900.0,
        memory_size_gb=32.0,
        l2_cache_mb=6.0,
        sm_count=80,
        max_threads_per_sm=2048,
    ),
}


def detect_hardware() -> HardwareProfile:
    """
    自动检测当前GPU硬件

    Returns:
        HardwareProfile: 检测到的硬件Profile
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return HardwareProfile(gpu_name="CPU", gpu_arch=GPUArch.UNKNOWN)

        # 获取GPU信息
        device_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)

        # 尝试匹配预定义Profile
        for key, profile in _GPU_PROFILES.items():
            if key.upper() in device_name.upper().replace(' ', ''):
                # 更新实际显存大小
                props = torch.cuda.get_device_properties(0)
                profile.memory_size_gb = props.total_memory / (1024**3)
                return profile

        # 无法匹配，创建一个基于检测值的Profile
        props = torch.cuda.get_device_properties(0)

        # 根据compute capability估算性能
        if compute_cap >= (9, 0):
            arch = GPUArch.HOPPER
            fp16_factor = 8.0
        elif compute_cap >= (8, 9):
            arch = GPUArch.ADA
            fp16_factor = 4.0
        elif compute_cap >= (8, 0):
            arch = GPUArch.AMPERE
            fp16_factor = 4.0
        elif compute_cap >= (7, 5):
            arch = GPUArch.TURING
            fp16_factor = 2.0
        elif compute_cap >= (7, 0):
            arch = GPUArch.VOLTA
            fp16_factor = 2.0
        else:
            arch = GPUArch.UNKNOWN
            fp16_factor = 1.0

        # 估算吞吐量（基于SM数量和时钟频率）
        sm_count = props.multi_processor_count
        clock_ghz = props.clock_rate / 1e6  # kHz -> GHz

        # 每个SM每周期的FP32 ops（估算）
        fp32_ops_per_sm_per_cycle = 64 * 2  # 两个warp scheduler
        fp32_tflops = sm_count * clock_ghz * fp32_ops_per_sm_per_cycle / 1000

        return HardwareProfile(
            gpu_name=device_name,
            gpu_arch=arch,
            compute_capability=compute_cap,
            fp16_tflops=fp32_tflops * fp16_factor,
            fp32_tflops=fp32_tflops,
            int8_tops=fp32_tflops * fp16_factor * 2,
            memory_bandwidth_gbps=props.total_memory / (1024**3) * 100,  # 粗略估算
            memory_size_gb=props.total_memory / (1024**3),
            l2_cache_mb=props.l2_cache_size / (1024**2) if hasattr(props, 'l2_cache_size') else 6.0,
            sm_count=sm_count,
            max_threads_per_sm=props.max_threads_per_multi_processor,
        )

    except ImportError:
        return HardwareProfile(gpu_name="Unknown (torch not available)")
    except Exception as e:
        return HardwareProfile(gpu_name=f"Unknown (error: {e})")


def get_hardware_profile(name: Optional[str] = None) -> HardwareProfile:
    """
    获取硬件Profile

    Args:
        name: 指定的GPU名称（如 'A100'），如果为None则自动检测

    Returns:
        HardwareProfile
    """
    if name is not None:
        if name in _GPU_PROFILES:
            return _GPU_PROFILES[name]
        raise ValueError(f"Unknown GPU: {name}. Available: {list(_GPU_PROFILES.keys())}")

    return detect_hardware()
