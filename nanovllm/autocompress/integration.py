"""
AutoCompress与nano-vllm的集成层

提供：
1. 与现有BatchedGEMMCompressor的集成
2. 与AsyncCompressionManager的集成
3. 简化的API接口
"""

from typing import Optional, Dict, Any, List, Callable, TYPE_CHECKING
from dataclasses import dataclass
import torch
import torch.nn as nn

from .profiles import CompressMethodProfile, get_profile, register_profile, ComputeType
from .context import WorkloadContext, HardwareProfile, detect_hardware
from .cost_model import CostModel
from .scheduler import AutoCompressScheduler, SchedulerConfig, auto_schedule
from .execution import (
    ExecutionPlan, ExecutionTiming, ParallelismMode,
    PipelineExecutor, StepType, get_stream_manager
)

if TYPE_CHECKING:
    from ..kernels.batched_compress import BatchedGEMMCompressor


@dataclass
class CompressionConfig:
    """用户级压缩配置"""
    method: str = 'mlp'                    # 压缩方法
    compression_ratio: float = 5.0         # 目标压缩比
    auto_schedule: bool = True             # 是否自动调度
    force_async: bool = False              # 强制异步模式
    force_sync: bool = False               # 强制同步模式
    min_seq_len: int = 256                 # 最小序列长度
    min_speedup: float = 1.1               # 最小加速比要求


class AutoCompressManager:
    """
    自动压缩管理器

    这是AutoCompress系统的主入口。
    负责：
    1. 接收用户配置
    2. 调用调度器生成执行计划
    3. 执行压缩流水线
    4. 与现有系统集成

    使用方式：
    ```python
    manager = AutoCompressManager(config=CompressionConfig(method='mlp'))

    # 在推理循环中
    plan = manager.get_plan(batch_size=4, seq_len=2048, expected_output=100)
    if plan.estimated_speedup > 1.0:
        manager.execute(plan, kv_cache, compressor)
    ```
    """

    def __init__(self,
                 config: Optional[CompressionConfig] = None,
                 hw_profile: Optional[HardwareProfile] = None):
        """
        Args:
            config: 压缩配置
            hw_profile: 硬件Profile（None则自动检测）
        """
        self.config = config or CompressionConfig()
        self.hw_profile = hw_profile or detect_hardware()

        # 初始化调度器
        scheduler_config = SchedulerConfig(
            min_speedup=self.config.min_speedup,
            min_seq_len=self.config.min_seq_len,
        )
        self.scheduler = AutoCompressScheduler(self.hw_profile, scheduler_config)

        # 初始化执行器
        self.executor = PipelineExecutor()

        # Cost Model
        self.cost_model = CostModel(self.hw_profile)

        # 统计
        self._stats = {
            'total_compressions': 0,
            'total_speedup': 0.0,
            'skipped_compressions': 0,
        }

    def get_plan(self,
                 batch_size: int,
                 seq_len: int,
                 expected_output_len: int = 100,
                 num_layers: int = 32,
                 num_kv_heads: int = 8,
                 head_dim: int = 128) -> ExecutionPlan:
        """
        获取执行计划

        Args:
            batch_size: 批大小
            seq_len: 序列长度
            expected_output_len: 预期输出长度
            num_layers: 模型层数
            num_kv_heads: KV头数
            head_dim: 头维度

        Returns:
            ExecutionPlan
        """
        workload = WorkloadContext(
            batch_size=batch_size,
            seq_len=seq_len,
            expected_output_len=expected_output_len,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        # 强制模式
        force_timing = None
        if self.config.force_async:
            force_timing = ExecutionTiming.ASYNC_DECODE
        elif self.config.force_sync:
            force_timing = ExecutionTiming.AFTER_PREFILL

        return self.scheduler.schedule(
            self.config.method,
            workload,
            force_timing=force_timing
        )

    def should_compress(self,
                        batch_size: int,
                        seq_len: int,
                        expected_output_len: int = 100) -> bool:
        """
        快速判断是否应该压缩

        Returns:
            bool: 是否应该压缩
        """
        plan = self.get_plan(batch_size, seq_len, expected_output_len)
        return plan.estimated_speedup >= self.config.min_speedup

    def execute_sync(self,
                     kv_cache: torch.Tensor,
                     compressor: nn.Module,
                     **kwargs) -> torch.Tensor:
        """
        同步执行压缩

        Args:
            kv_cache: KV-cache张量
            compressor: 压缩器模块

        Returns:
            压缩后的KV-cache
        """
        with torch.cuda.stream(torch.cuda.current_stream()):
            compressed = compressor(kv_cache, **kwargs)
        torch.cuda.synchronize()
        return compressed

    def execute_async(self,
                      kv_cache: torch.Tensor,
                      compressor: nn.Module,
                      chunk_size: int = 4,
                      callback: Optional[Callable] = None,
                      **kwargs) -> 'AsyncCompressionHandle':
        """
        异步执行压缩

        Args:
            kv_cache: KV-cache张量
            compressor: 压缩器模块
            chunk_size: 分块大小
            callback: 完成回调

        Returns:
            AsyncCompressionHandle
        """
        stream_manager = get_stream_manager()
        compress_stream = stream_manager.get_compress_stream()

        handle = AsyncCompressionHandle(
            kv_cache=kv_cache,
            compressor=compressor,
            stream=compress_stream,
            chunk_size=chunk_size,
            callback=callback,
            kwargs=kwargs,
        )

        # 启动异步压缩
        handle.start()

        return handle

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        if stats['total_compressions'] > 0:
            stats['avg_speedup'] = stats['total_speedup'] / stats['total_compressions']
        else:
            stats['avg_speedup'] = 0.0
        return stats


class AsyncCompressionHandle:
    """
    异步压缩句柄

    用于管理异步压缩的生命周期
    """

    def __init__(self,
                 kv_cache: torch.Tensor,
                 compressor: nn.Module,
                 stream: Any,
                 chunk_size: int = 4,
                 callback: Optional[Callable] = None,
                 kwargs: Optional[Dict] = None):
        self.kv_cache = kv_cache
        self.compressor = compressor
        self.stream = stream
        self.chunk_size = chunk_size
        self.callback = callback
        self.kwargs = kwargs or {}

        # 状态
        self._started = False
        self._completed = False
        self._result = None
        self._event = None
        self._current_chunk = 0
        self._num_chunks = 32 // chunk_size  # 假设32层

        # 中间结果
        self._chunk_results: List[torch.Tensor] = []

    def start(self):
        """启动异步压缩"""
        if self._started:
            return

        self._started = True

        with torch.cuda.stream(self.stream):
            self._event = torch.cuda.Event()

            # 执行压缩（分块）
            # 这里简化实现，实际应该逐chunk执行
            self._result = self.compressor(self.kv_cache, **self.kwargs)

            self._event.record(self.stream)

        self._completed = True

        if self.callback:
            self.callback(self._result)

    def is_ready(self) -> bool:
        """检查是否完成"""
        if not self._started:
            return False
        if self._event is None:
            return self._completed
        return self._event.query()

    def wait(self) -> torch.Tensor:
        """等待完成并返回结果"""
        if self._event is not None:
            self._event.synchronize()
        return self._result

    def get_result(self) -> Optional[torch.Tensor]:
        """获取结果（不等待）"""
        if self.is_ready():
            return self._result
        return None


class AdaptiveCompressionWrapper:
    """
    自适应压缩包装器

    包装现有的压缩器，添加自动调度能力。

    使用方式：
    ```python
    # 包装现有压缩器
    compressor = BatchedGEMMCompressor(...)
    adaptive = AdaptiveCompressionWrapper(compressor, method='mlp')

    # 使用自适应压缩
    result = adaptive.compress(kv_cache, batch_size=4, seq_len=2048)
    ```
    """

    def __init__(self,
                 compressor: nn.Module,
                 method: str = 'mlp',
                 auto_config: Optional[CompressionConfig] = None):
        """
        Args:
            compressor: 现有的压缩器
            method: 压缩方法名称
            auto_config: 自动压缩配置
        """
        self.compressor = compressor
        self.method = method
        self.config = auto_config or CompressionConfig(method=method)

        # 自动压缩管理器
        self.manager = AutoCompressManager(self.config)

        # 缓存
        self._last_plan: Optional[ExecutionPlan] = None

    def compress(self,
                 kv_cache: torch.Tensor,
                 batch_size: Optional[int] = None,
                 seq_len: Optional[int] = None,
                 expected_output_len: int = 100,
                 **kwargs) -> torch.Tensor:
        """
        自适应压缩

        Args:
            kv_cache: KV-cache张量
            batch_size: 批大小（可从张量推断）
            seq_len: 序列长度（可从张量推断）
            expected_output_len: 预期输出长度
            **kwargs: 传递给压缩器的额外参数

        Returns:
            压缩后的KV-cache（或原始cache如果不压缩）
        """
        # 从张量推断维度
        if batch_size is None or seq_len is None:
            # 假设kv_cache形状为 [num_layers, 2, batch, heads, seq_len, head_dim]
            # 或其他常见格式
            shape = kv_cache.shape
            if len(shape) >= 4:
                # 尝试推断
                if batch_size is None:
                    batch_size = shape[-4] if len(shape) >= 4 else 1
                if seq_len is None:
                    seq_len = shape[-2] if len(shape) >= 2 else 512

        # 获取执行计划
        plan = self.manager.get_plan(
            batch_size=batch_size,
            seq_len=seq_len,
            expected_output_len=expected_output_len,
        )
        self._last_plan = plan

        # 检查是否应该压缩
        if plan.estimated_speedup < self.config.min_speedup:
            # 不压缩
            return kv_cache

        # 根据计划执行
        if plan.parallelism.mode == ParallelismMode.SYNC:
            return self.manager.execute_sync(kv_cache, self.compressor, **kwargs)

        elif plan.parallelism.mode == ParallelismMode.ASYNC_STREAM:
            handle = self.manager.execute_async(
                kv_cache, self.compressor,
                chunk_size=plan.chunking.chunk_size,
                **kwargs
            )
            # 对于简化版本，直接等待结果
            return handle.wait()

        else:
            # Fused模式等，目前回退到同步
            return self.manager.execute_sync(kv_cache, self.compressor, **kwargs)

    def get_last_plan(self) -> Optional[ExecutionPlan]:
        """获取最后一次执行计划"""
        return self._last_plan


# ============ 便捷函数 ============

def create_auto_compress_manager(
    method: str = 'mlp',
    compression_ratio: float = 5.0,
    auto_schedule: bool = True,
    **kwargs
) -> AutoCompressManager:
    """
    创建自动压缩管理器

    Example:
        ```python
        manager = create_auto_compress_manager(method='mlp', compression_ratio=5.0)
        plan = manager.get_plan(batch_size=4, seq_len=2048)
        ```
    """
    config = CompressionConfig(
        method=method,
        compression_ratio=compression_ratio,
        auto_schedule=auto_schedule,
        **kwargs
    )
    return AutoCompressManager(config)


def wrap_compressor(compressor: nn.Module,
                    method: str = 'mlp') -> AdaptiveCompressionWrapper:
    """
    包装压缩器为自适应版本

    Example:
        ```python
        compressor = BatchedGEMMCompressor(...)
        adaptive = wrap_compressor(compressor, method='mlp')
        result = adaptive.compress(kv_cache)
        ```
    """
    return AdaptiveCompressionWrapper(compressor, method)


def analyze_workload(batch_size: int,
                     seq_len: int,
                     expected_output_len: int = 100,
                     methods: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    分析workload并推荐最优压缩配置

    Args:
        batch_size: 批大小
        seq_len: 序列长度
        expected_output_len: 预期输出长度
        methods: 候选方法列表（None则使用所有内置方法）

    Returns:
        分析结果，包含推荐配置
    """
    if methods is None:
        methods = ['mlp', 'snapkv', 'streaming_llm', 'h2o', 'int4_quant']

    workload = WorkloadContext(
        batch_size=batch_size,
        seq_len=seq_len,
        expected_output_len=expected_output_len,
    )

    scheduler = AutoCompressScheduler()

    results = {
        'workload': {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'expected_output_len': expected_output_len,
        },
        'hardware': {
            'gpu_name': scheduler.hw_profile.gpu_name,
            'roofline': scheduler.hw_profile.roofline_intensity,
        },
        'methods': {},
        'recommendation': None,
    }

    best_method = None
    best_speedup = 1.0

    for method in methods:
        try:
            analysis = scheduler.analyze(method, workload)
            plan = scheduler.schedule(method, workload)

            results['methods'][method] = {
                'speedup_sync': analysis['performance']['speedup_sync'],
                'speedup_async': analysis['performance']['speedup_async'],
                'recommended_timing': plan.timing.name,
                'estimated_speedup': plan.estimated_speedup,
                'compress_latency_ms': plan.estimated_compress_latency_ms,
            }

            if plan.estimated_speedup > best_speedup:
                best_speedup = plan.estimated_speedup
                best_method = method

        except Exception as e:
            results['methods'][method] = {'error': str(e)}

    if best_method:
        results['recommendation'] = {
            'method': best_method,
            'speedup': best_speedup,
        }

    return results
