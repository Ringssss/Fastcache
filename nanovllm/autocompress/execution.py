"""
执行计划和执行器

定义：
- ExecutionTiming: 执行时机枚举
- ParallelismConfig: 并行配置
- ChunkingConfig: 分块配置
- ExecutionPlan: 完整执行计划
- ExecutionStep: 单个执行步骤
- PipelineExecutor: 流水线执行器
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Callable, Tuple
import threading
import queue


class ExecutionTiming(Enum):
    """执行时机"""
    INLINE_PREFILL = auto()   # 融合进Prefill
    AFTER_PREFILL = auto()    # Prefill后同步执行
    ASYNC_DECODE = auto()     # 与Decode异步并行


class ParallelismMode(Enum):
    """并行模式"""
    SYNC = auto()             # 同步执行
    ASYNC_STREAM = auto()     # 异步Stream执行
    FUSED = auto()            # 融合进其他kernel


@dataclass
class ParallelismConfig:
    """
    并行配置
    """
    mode: ParallelismMode = ParallelismMode.SYNC

    # ASYNC_STREAM模式的配置
    stream_priority: int = 0          # CUDA stream优先级 (-1=low, 0=normal)
    num_streams: int = 1              # 使用的stream数量

    # FUSED模式的配置
    fusion_point: str = 'after_layer'  # 融合点: 'after_attention', 'after_layer'

    # 同步点配置
    sync_after_chunks: int = 4        # 每N个chunk同步一次
    use_events: bool = True           # 是否使用CUDA events进行同步


@dataclass
class ChunkingConfig:
    """
    分块配置
    """
    enabled: bool = False
    chunk_size: int = 4               # 每块的层数
    num_chunks: int = 8               # 总块数
    interleave_with: str = 'decode'   # 交错方式: 'decode', 'none'

    def get_layers(self, chunk_idx: int) -> Tuple[int, int]:
        """获取指定chunk的层范围"""
        start = chunk_idx * self.chunk_size
        end = min(start + self.chunk_size, 32)  # 假设32层
        return start, end

    @classmethod
    def create(cls, num_layers: int = 32,
               chunk_size: int = 4,
               interleave: bool = True) -> 'ChunkingConfig':
        """创建分块配置"""
        num_chunks = (num_layers + chunk_size - 1) // chunk_size
        return cls(
            enabled=True,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            interleave_with='decode' if interleave else 'none'
        )


class StepType(Enum):
    """执行步骤类型"""
    WAIT_PREFILL = auto()
    START_DECODE = auto()
    SYNC_COMPRESS = auto()
    ASYNC_COMPRESS_CHUNK = auto()
    ASYNC_COMPRESS_ALL = auto()
    FUSED_COMPRESS = auto()
    SYNC_EVENT = auto()
    CHECK_PROGRESS = auto()
    SWITCH_KV = auto()              # 切换到压缩后的KV


@dataclass
class ExecutionStep:
    """
    单个执行步骤
    """
    type: StepType
    config: Dict[str, Any] = field(default_factory=dict)

    # 可选的回调函数
    callback: Optional[Callable] = None

    def __repr__(self):
        return f"ExecutionStep({self.type.name}, {self.config})"


@dataclass
class ExecutionPlan:
    """
    完整的执行计划

    这是调度器输出的"指令"，执行器照此执行。
    """

    # 基础信息
    method: str = ""
    timing: ExecutionTiming = ExecutionTiming.AFTER_PREFILL

    # 配置
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    # 执行步骤
    steps: List[ExecutionStep] = field(default_factory=list)

    # 性能预期
    estimated_speedup: float = 1.0
    estimated_compress_latency_ms: float = 0.0
    estimated_total_time_ms: float = 0.0

    # 警告信息
    warnings: List[str] = field(default_factory=list)

    # 调试信息
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: ExecutionStep):
        """添加执行步骤"""
        self.steps.append(step)

    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)

    def is_valid(self) -> bool:
        """检查计划是否有效"""
        return len(self.steps) > 0 and self.estimated_speedup >= 1.0

    def to_schedule(self) -> List[ExecutionStep]:
        """转换为执行调度"""
        return self.steps.copy()

    def summary(self) -> str:
        """生成计划摘要"""
        lines = [
            f"ExecutionPlan for '{self.method}'",
            f"  Timing: {self.timing.name}",
            f"  Parallelism: {self.parallelism.mode.name}",
            f"  Chunking: {'enabled' if self.chunking.enabled else 'disabled'}",
            f"  Estimated Speedup: {self.estimated_speedup:.2f}x",
            f"  Estimated Latency: {self.estimated_compress_latency_ms:.2f}ms",
            f"  Steps: {len(self.steps)}",
        ]
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return '\n'.join(lines)

    @classmethod
    def create_sync_plan(cls, method: str,
                         compress_latency_ms: float,
                         speedup: float) -> 'ExecutionPlan':
        """创建同步执行计划"""
        plan = cls(
            method=method,
            timing=ExecutionTiming.AFTER_PREFILL,
            parallelism=ParallelismConfig(mode=ParallelismMode.SYNC),
            chunking=ChunkingConfig(enabled=False),
            estimated_speedup=speedup,
            estimated_compress_latency_ms=compress_latency_ms,
        )

        plan.add_step(ExecutionStep(StepType.WAIT_PREFILL))
        plan.add_step(ExecutionStep(
            StepType.SYNC_COMPRESS,
            config={'method': method}
        ))
        plan.add_step(ExecutionStep(StepType.SWITCH_KV))
        plan.add_step(ExecutionStep(StepType.START_DECODE))

        return plan

    @classmethod
    def create_async_plan(cls, method: str,
                          compress_latency_ms: float,
                          speedup: float,
                          chunk_size: int = 4,
                          num_layers: int = 32) -> 'ExecutionPlan':
        """创建异步执行计划"""
        chunking = ChunkingConfig.create(
            num_layers=num_layers,
            chunk_size=chunk_size,
            interleave=True
        )

        plan = cls(
            method=method,
            timing=ExecutionTiming.ASYNC_DECODE,
            parallelism=ParallelismConfig(
                mode=ParallelismMode.ASYNC_STREAM,
                stream_priority=-1,
                use_events=True,
            ),
            chunking=chunking,
            estimated_speedup=speedup,
            estimated_compress_latency_ms=compress_latency_ms,
        )

        # 生成执行步骤
        plan.add_step(ExecutionStep(StepType.WAIT_PREFILL))
        plan.add_step(ExecutionStep(StepType.START_DECODE))

        for chunk_idx in range(chunking.num_chunks):
            layers = chunking.get_layers(chunk_idx)
            plan.add_step(ExecutionStep(
                StepType.ASYNC_COMPRESS_CHUNK,
                config={
                    'chunk_idx': chunk_idx,
                    'layers': layers,
                    'stream_priority': -1,
                }
            ))

            # 每几个chunk检查一次进度
            if (chunk_idx + 1) % plan.parallelism.sync_after_chunks == 0:
                plan.add_step(ExecutionStep(
                    StepType.CHECK_PROGRESS,
                    config={'check_decode_latency': True}
                ))

        # 最终同步和切换
        plan.add_step(ExecutionStep(
            StepType.SYNC_EVENT,
            config={'wait_all_chunks': True}
        ))
        plan.add_step(ExecutionStep(StepType.SWITCH_KV))

        return plan

    @classmethod
    def create_inline_plan(cls, method: str,
                           compress_latency_ms: float,
                           speedup: float) -> 'ExecutionPlan':
        """创建inline（融合）执行计划"""
        plan = cls(
            method=method,
            timing=ExecutionTiming.INLINE_PREFILL,
            parallelism=ParallelismConfig(
                mode=ParallelismMode.FUSED,
                fusion_point='after_attention',
            ),
            chunking=ChunkingConfig(enabled=False),
            estimated_speedup=speedup,
            estimated_compress_latency_ms=compress_latency_ms,
        )

        plan.add_step(ExecutionStep(
            StepType.FUSED_COMPRESS,
            config={
                'method': method,
                'fusion_point': 'after_attention',
            }
        ))
        # inline模式下，压缩后直接使用压缩KV
        plan.add_step(ExecutionStep(StepType.START_DECODE))

        return plan

    @classmethod
    def create_no_compress_plan(cls) -> 'ExecutionPlan':
        """创建不压缩的执行计划"""
        plan = cls(
            method='none',
            timing=ExecutionTiming.AFTER_PREFILL,
            parallelism=ParallelismConfig(mode=ParallelismMode.SYNC),
            chunking=ChunkingConfig(enabled=False),
            estimated_speedup=1.0,
            estimated_compress_latency_ms=0.0,
        )

        plan.add_step(ExecutionStep(StepType.WAIT_PREFILL))
        plan.add_step(ExecutionStep(StepType.START_DECODE))

        return plan


class PipelineExecutor:
    """
    流水线执行器

    负责按照ExecutionPlan执行压缩流水线。
    与具体的压缩实现解耦，通过回调函数执行实际操作。
    """

    def __init__(self):
        # 状态
        self._current_plan: Optional[ExecutionPlan] = None
        self._step_index: int = 0
        self._is_running: bool = False

        # CUDA相关（延迟初始化）
        self._compress_stream = None
        self._events: Dict[int, Any] = {}  # chunk_idx -> event

        # 回调函数
        self._callbacks: Dict[StepType, Callable] = {}

        # 监控
        self._step_latencies: List[float] = []
        self._warnings: List[str] = []

    def register_callback(self, step_type: StepType, callback: Callable):
        """
        注册步骤回调

        Args:
            step_type: 步骤类型
            callback: 回调函数，签名为 callback(config: Dict) -> None
        """
        self._callbacks[step_type] = callback

    def set_plan(self, plan: ExecutionPlan):
        """设置执行计划"""
        self._current_plan = plan
        self._step_index = 0
        self._step_latencies = []
        self._warnings = []

    def execute_step(self, step: ExecutionStep) -> bool:
        """
        执行单个步骤

        Returns:
            是否成功
        """
        import time

        start_time = time.perf_counter()

        try:
            # 使用自定义回调
            if step.callback is not None:
                step.callback(step.config)

            # 使用注册的回调
            elif step.type in self._callbacks:
                self._callbacks[step.type](step.config)

            # 默认处理
            else:
                self._default_step_handler(step)

            elapsed = (time.perf_counter() - start_time) * 1000
            self._step_latencies.append(elapsed)
            return True

        except Exception as e:
            self._warnings.append(f"Step {step.type.name} failed: {e}")
            return False

    def _default_step_handler(self, step: ExecutionStep):
        """默认步骤处理器"""
        if step.type == StepType.WAIT_PREFILL:
            # 等待Prefill完成（通常由外部控制）
            pass

        elif step.type == StepType.START_DECODE:
            # 开始Decode循环（通常由外部控制）
            pass

        elif step.type == StepType.SYNC_EVENT:
            # 同步CUDA events
            self._sync_all_events()

        elif step.type == StepType.CHECK_PROGRESS:
            # 检查进度（可以添加延迟监控逻辑）
            pass

        elif step.type == StepType.SWITCH_KV:
            # 切换KV-cache（需要外部实现）
            pass

    def _sync_all_events(self):
        """同步所有CUDA events"""
        try:
            import torch
            for event in self._events.values():
                event.synchronize()
        except ImportError:
            pass

    def execute_all(self) -> bool:
        """
        执行所有步骤

        Returns:
            是否全部成功
        """
        if self._current_plan is None:
            return False

        self._is_running = True
        success = True

        for step in self._current_plan.steps:
            if not self.execute_step(step):
                success = False
                # 继续执行，但记录失败

        self._is_running = False
        return success

    def get_statistics(self) -> Dict[str, Any]:
        """获取执行统计"""
        return {
            'total_steps': len(self._step_latencies),
            'total_time_ms': sum(self._step_latencies),
            'avg_step_time_ms': sum(self._step_latencies) / max(len(self._step_latencies), 1),
            'warnings': self._warnings.copy(),
        }


class StreamManager:
    """
    CUDA Stream管理器

    管理主流和压缩流，处理同步。
    """

    def __init__(self):
        self._main_stream = None
        self._compress_stream = None
        self._initialized = False

    def initialize(self):
        """延迟初始化CUDA streams"""
        if self._initialized:
            return

        try:
            import torch
            if torch.cuda.is_available():
                # 主流使用默认stream
                self._main_stream = torch.cuda.current_stream()

                # 压缩流使用低优先级
                # 注意：优先级支持取决于GPU型号
                self._compress_stream = torch.cuda.Stream(priority=-1)

                self._initialized = True
        except ImportError:
            pass

    def get_main_stream(self):
        """获取主流"""
        self.initialize()
        return self._main_stream

    def get_compress_stream(self):
        """获取压缩流"""
        self.initialize()
        return self._compress_stream

    def synchronize_compress_to_main(self):
        """同步压缩流到主流"""
        if self._compress_stream is not None:
            try:
                import torch
                event = torch.cuda.Event()
                event.record(self._compress_stream)
                self._main_stream.wait_event(event)
            except ImportError:
                pass


# 全局Stream管理器实例
_stream_manager: Optional[StreamManager] = None


def get_stream_manager() -> StreamManager:
    """获取全局Stream管理器"""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager
