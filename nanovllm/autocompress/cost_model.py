"""
Cost Model: 性能预测模型

核心功能：
1. 预测Decode延迟（有/无压缩）
2. 预测压缩延迟
3. 计算预期加速比
4. 支持在线校准

关键保证：
- 只有当预期加速比 > 1.0 时才推荐压缩
- 留10%安全边际
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import json
from pathlib import Path

from .profiles import CompressMethodProfile, get_profile, ComputeType
from .context import HardwareProfile, WorkloadContext


@dataclass
class CalibrationData:
    """
    校准数据

    存储实际测量的延迟数据，用于校准Cost Model
    """

    # Decode延迟测量: (batch_size, seq_len) -> latency_ms
    decode_measurements: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # 压缩延迟测量: (method, seq_len, batch_size) -> latency_ms
    compress_measurements: Dict[Tuple[str, int, int], float] = field(default_factory=dict)

    # 校准时间戳
    timestamp: str = ""

    # 硬件信息
    hardware_name: str = ""

    def add_decode_measurement(self, batch_size: int, seq_len: int, latency_ms: float):
        """添加Decode延迟测量"""
        self.decode_measurements[(batch_size, seq_len)] = latency_ms

    def add_compress_measurement(self, method: str, seq_len: int,
                                  batch_size: int, latency_ms: float):
        """添加压缩延迟测量"""
        self.compress_measurements[(method, seq_len, batch_size)] = latency_ms

    def save(self, path: str):
        """保存校准数据"""
        data = {
            'decode_measurements': {
                f"{k[0]},{k[1]}": v for k, v in self.decode_measurements.items()
            },
            'compress_measurements': {
                f"{k[0]},{k[1]},{k[2]}": v for k, v in self.compress_measurements.items()
            },
            'timestamp': self.timestamp,
            'hardware_name': self.hardware_name,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> 'CalibrationData':
        """加载校准数据"""
        data = json.loads(Path(path).read_text())
        instance = cls()
        instance.timestamp = data.get('timestamp', '')
        instance.hardware_name = data.get('hardware_name', '')

        for k, v in data.get('decode_measurements', {}).items():
            parts = k.split(',')
            instance.decode_measurements[(int(parts[0]), int(parts[1]))] = v

        for k, v in data.get('compress_measurements', {}).items():
            parts = k.split(',')
            instance.compress_measurements[(parts[0], int(parts[1]), int(parts[2]))] = v

        return instance


class CostModel:
    """
    性能预测模型

    核心功能：
    1. 基于硬件Profile预测延迟
    2. 支持校准数据微调
    3. 计算加速比，指导调度决策
    """

    def __init__(self, hw_profile: HardwareProfile,
                 calibration: Optional[CalibrationData] = None):
        """
        Args:
            hw_profile: 硬件Profile
            calibration: 可选的校准数据
        """
        self.hw_profile = hw_profile
        self.calibration = calibration

        # 校准系数（用于调整理论预测）
        self._decode_scale = 1.0
        self._compress_scale: Dict[str, float] = {}

        if calibration is not None:
            self._fit_calibration()

    def _fit_calibration(self):
        """根据校准数据调整模型"""
        if self.calibration is None:
            return

        # 拟合Decode scale
        if self.calibration.decode_measurements:
            ratios = []
            for (batch_size, seq_len), actual_latency in self.calibration.decode_measurements.items():
                predicted = self._raw_decode_latency(batch_size, seq_len)
                if predicted > 0:
                    ratios.append(actual_latency / predicted)
            if ratios:
                self._decode_scale = sum(ratios) / len(ratios)

        # 拟合各压缩方法的scale
        method_ratios: Dict[str, List[float]] = {}
        for (method, seq_len, batch_size), actual_latency in self.calibration.compress_measurements.items():
            try:
                profile = get_profile(method)
                predicted = profile.estimate_latency(seq_len, batch_size)
                if predicted > 0:
                    if method not in method_ratios:
                        method_ratios[method] = []
                    method_ratios[method].append(actual_latency / predicted)
            except ValueError:
                continue

        for method, ratios in method_ratios.items():
            if ratios:
                self._compress_scale[method] = sum(ratios) / len(ratios)

    def _raw_decode_latency(self, batch_size: int, seq_len: int,
                            num_heads: int = 32, head_dim: int = 128) -> float:
        """
        原始Decode延迟预测（未校准）

        基于Roofline模型
        """
        # 计算intensity
        flops = batch_size * seq_len * num_heads * head_dim * 2
        bytes_read = batch_size * seq_len * num_heads * head_dim * 2 * 2

        intensity = flops / max(bytes_read, 1)
        roofline = self.hw_profile.roofline_intensity

        if intensity < roofline:
            # Memory-bound
            latency_s = bytes_read / (self.hw_profile.memory_bandwidth_gbps * 1e9)
        else:
            # Compute-bound
            latency_s = flops / (self.hw_profile.fp16_tflops * 1e12)

        # Kernel launch overhead
        overhead_ms = 0.02 * (1 + math.log2(max(batch_size, 1)) * 0.1)

        return latency_s * 1000 + overhead_ms

    def estimate_decode_latency(self, workload: WorkloadContext,
                                 compressed: bool = False,
                                 compression_ratio: float = 1.0) -> float:
        """
        估算单次Decode延迟 (ms)

        Args:
            workload: Workload上下文
            compressed: 是否使用压缩后的KV-cache
            compression_ratio: 压缩比

        Returns:
            预估延迟 (毫秒)
        """
        seq_len = workload.seq_len
        if compressed:
            seq_len = int(seq_len / compression_ratio)

        raw_latency = self._raw_decode_latency(
            workload.batch_size,
            seq_len,
            workload.num_kv_heads,
            workload.head_dim
        )

        return raw_latency * self._decode_scale

    def estimate_compress_latency(self, method: str,
                                   workload: WorkloadContext) -> float:
        """
        估算压缩延迟 (ms)

        Args:
            method: 压缩方法名称
            workload: Workload上下文

        Returns:
            预估延迟 (毫秒)
        """
        profile = get_profile(method)
        raw_latency = profile.estimate_latency(workload.seq_len, workload.batch_size)

        scale = self._compress_scale.get(method, 1.0)
        return raw_latency * scale

    def estimate_total_time(self, workload: WorkloadContext,
                            method: Optional[str] = None,
                            async_mode: bool = False) -> Tuple[float, Dict[str, Any]]:
        """
        估算总推理时间

        Args:
            workload: Workload上下文
            method: 压缩方法（None表示不压缩）
            async_mode: 是否异步执行压缩

        Returns:
            (总时间ms, 详细breakdown)
        """
        output_len = workload.expected_output_len

        if method is None:
            # 不压缩
            decode_latency = self.estimate_decode_latency(workload, compressed=False)
            total_time = decode_latency * output_len

            return total_time, {
                'decode_latency_per_token': decode_latency,
                'total_decode_time': total_time,
                'compress_latency': 0,
                'compressed': False,
            }

        # 使用压缩
        profile = get_profile(method)
        compress_latency = self.estimate_compress_latency(method, workload)
        decode_latency_compressed = self.estimate_decode_latency(
            workload, compressed=True, compression_ratio=profile.compression_ratio
        )

        if async_mode:
            # 异步模式：压缩与前几个Decode重叠
            # 前几个Decode使用原始KV，后续使用压缩KV
            decode_latency_original = self.estimate_decode_latency(workload, compressed=False)

            # 计算压缩完成前能执行多少个Decode
            overlap_tokens = min(
                output_len,
                int(compress_latency / decode_latency_original) + 1
            )

            # 总时间 = max(压缩时间, 前几个Decode时间) + 剩余Decode时间
            parallel_phase = max(compress_latency, decode_latency_original * overlap_tokens)
            remaining_tokens = output_len - overlap_tokens
            sequential_phase = decode_latency_compressed * remaining_tokens

            total_time = parallel_phase + sequential_phase

            return total_time, {
                'decode_latency_original': decode_latency_original,
                'decode_latency_compressed': decode_latency_compressed,
                'compress_latency': compress_latency,
                'overlap_tokens': overlap_tokens,
                'remaining_tokens': remaining_tokens,
                'parallel_phase': parallel_phase,
                'sequential_phase': sequential_phase,
                'compressed': True,
                'async_mode': True,
            }

        else:
            # 同步模式：先压缩，再Decode
            total_decode_time = decode_latency_compressed * output_len
            total_time = compress_latency + total_decode_time

            return total_time, {
                'decode_latency_compressed': decode_latency_compressed,
                'compress_latency': compress_latency,
                'total_decode_time': total_decode_time,
                'compressed': True,
                'async_mode': False,
            }

    def compute_speedup(self, workload: WorkloadContext,
                        method: str,
                        async_mode: bool = False) -> Tuple[float, Dict[str, Any]]:
        """
        计算预期加速比

        Args:
            workload: Workload上下文
            method: 压缩方法
            async_mode: 是否异步模式

        Returns:
            (加速比, 详细分析)
        """
        time_no_compress, breakdown_no_compress = self.estimate_total_time(
            workload, method=None
        )
        time_with_compress, breakdown_with_compress = self.estimate_total_time(
            workload, method=method, async_mode=async_mode
        )

        speedup = time_no_compress / max(time_with_compress, 1e-6)

        return speedup, {
            'time_no_compress_ms': time_no_compress,
            'time_with_compress_ms': time_with_compress,
            'speedup': speedup,
            'breakdown_no_compress': breakdown_no_compress,
            'breakdown_with_compress': breakdown_with_compress,
            'recommendation': 'compress' if speedup > 1.1 else 'no_compress',
        }

    def find_best_config(self, workload: WorkloadContext,
                         methods: List[str]) -> Tuple[Optional[str], bool, float, Dict]:
        """
        找到最优配置

        Args:
            workload: Workload上下文
            methods: 候选压缩方法列表

        Returns:
            (最优方法, 是否异步, 加速比, 详细分析)
        """
        best_method = None
        best_async = False
        best_speedup = 1.0  # 必须比不压缩快
        best_analysis = {}

        for method in methods:
            profile = get_profile(method)

            # 尝试同步模式
            speedup_sync, analysis_sync = self.compute_speedup(
                workload, method, async_mode=False
            )

            if speedup_sync > best_speedup:
                best_speedup = speedup_sync
                best_method = method
                best_async = False
                best_analysis = analysis_sync

            # 尝试异步模式（如果支持）
            if profile.chunkable or profile.decode_compatibility > 0.5:
                decode_is_memory_bound = self.hw_profile.is_decode_memory_bound(
                    workload.batch_size,
                    workload.seq_len,
                    workload.num_kv_heads,
                    workload.head_dim
                )

                if profile.can_overlap_with_decode(decode_is_memory_bound):
                    speedup_async, analysis_async = self.compute_speedup(
                        workload, method, async_mode=True
                    )

                    if speedup_async > best_speedup:
                        best_speedup = speedup_async
                        best_method = method
                        best_async = True
                        best_analysis = analysis_async

        return best_method, best_async, best_speedup, best_analysis

    def get_break_even_output_len(self, workload: WorkloadContext,
                                   method: str) -> int:
        """
        计算收支平衡的输出长度

        即：生成多少token后，压缩开始比不压缩快

        Returns:
            收支平衡的token数
        """
        profile = get_profile(method)
        compress_latency = self.estimate_compress_latency(method, workload)

        decode_original = self.estimate_decode_latency(workload, compressed=False)
        decode_compressed = self.estimate_decode_latency(
            workload, compressed=True, compression_ratio=profile.compression_ratio
        )

        # 求解: compress_latency + decode_compressed * N = decode_original * N
        # N = compress_latency / (decode_original - decode_compressed)

        latency_diff = decode_original - decode_compressed
        if latency_diff <= 0:
            return float('inf')  # 压缩后反而更慢，永远不会break even

        break_even = compress_latency / latency_diff
        return int(math.ceil(break_even))


# ============ 便捷函数 ============

def should_compress(workload: WorkloadContext,
                    method: str,
                    hw_profile: Optional[HardwareProfile] = None,
                    min_speedup: float = 1.1) -> Tuple[bool, float, str]:
    """
    快速判断是否应该压缩

    Args:
        workload: Workload上下文
        method: 压缩方法
        hw_profile: 硬件Profile（None则自动检测）
        min_speedup: 最小要求加速比

    Returns:
        (是否压缩, 预期加速比, 原因)
    """
    from .context import detect_hardware

    if hw_profile is None:
        hw_profile = detect_hardware()

    model = CostModel(hw_profile)

    # 检查输出长度是否足够
    break_even = model.get_break_even_output_len(workload, method)
    if workload.expected_output_len < break_even:
        return False, 0.0, f"Output length {workload.expected_output_len} < break-even point {break_even}"

    # 计算加速比
    speedup, analysis = model.compute_speedup(workload, method, async_mode=True)

    if speedup < min_speedup:
        return False, speedup, f"Speedup {speedup:.2f}x < minimum {min_speedup}x"

    return True, speedup, f"Expected speedup: {speedup:.2f}x"
