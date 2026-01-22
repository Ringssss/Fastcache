"""
AutoCompressScheduler: 核心调度器

核心功能：
1. 根据压缩方法的Profile和Workload特征，决定最优执行时机
2. 生成完整的执行计划
3. 保证性能：只有当压缩能带来加速时才启用

调度决策流程：
1. 分析压缩方法的计算特性 (Profile)
2. 分析当前Workload特征
3. 判断Decode是Memory-bound还是Compute-bound
4. 决定执行时机 (Inline/After Prefill/Async with Decode)
5. 决定并行策略和分块策略
6. 生成ExecutionPlan
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum, auto
import math

from .profiles import CompressMethodProfile, get_profile, ComputeType
from .context import WorkloadContext, HardwareProfile, detect_hardware
from .cost_model import CostModel
from .execution import (
    ExecutionPlan, ExecutionStep, ExecutionTiming,
    ParallelismConfig, ParallelismMode,
    ChunkingConfig, StepType
)


@dataclass
class SchedulerConfig:
    """调度器配置"""

    # 性能阈值
    min_speedup: float = 1.1          # 最小要求加速比（留10%余量）
    min_seq_len: int = 256            # 最小序列长度（短序列不压缩）
    min_output_len: int = 10          # 最小输出长度

    # 分块配置
    default_chunk_size: int = 4       # 默认分块大小
    max_chunks: int = 16              # 最大分块数

    # 异步配置
    async_threshold: float = 0.5      # Decode memory-bound比例阈值
    overlap_ratio: float = 0.7        # 期望的重叠比例

    # 安全配置
    enable_fallback: bool = True      # 启用降级机制
    latency_check_interval: int = 4   # 延迟检查间隔（chunk数）


class AutoCompressScheduler:
    """
    自动压缩调度器

    核心职责：为任意压缩方法找到最优执行方案

    使用方式：
    ```python
    scheduler = AutoCompressScheduler()
    plan = scheduler.schedule('mlp', workload)
    print(plan.summary())
    ```
    """

    def __init__(self,
                 hw_profile: Optional[HardwareProfile] = None,
                 config: Optional[SchedulerConfig] = None):
        """
        Args:
            hw_profile: 硬件Profile，None则自动检测
            config: 调度器配置
        """
        self.hw_profile = hw_profile or detect_hardware()
        self.config = config or SchedulerConfig()
        self.cost_model = CostModel(self.hw_profile)

        # 缓存
        self._plan_cache: Dict[str, ExecutionPlan] = {}

    def schedule(self, method: str,
                 workload: WorkloadContext,
                 force_timing: Optional[ExecutionTiming] = None) -> ExecutionPlan:
        """
        核心调度函数

        Args:
            method: 用户指定的压缩方法
            workload: 当前Workload上下文
            force_timing: 强制指定执行时机（可选）

        Returns:
            ExecutionPlan: 完整的执行计划
        """
        # Step 0: 基础检查
        if not self._should_compress(workload):
            return ExecutionPlan.create_no_compress_plan()

        # Step 1: 获取压缩方法Profile
        profile = get_profile(method)

        # Step 2: 决定执行时机
        if force_timing is not None:
            timing = force_timing
        else:
            timing = self._decide_timing(profile, workload)

        # Step 3: 决定并行策略
        parallelism = self._decide_parallelism(profile, workload, timing)

        # Step 4: 决定分块策略
        chunking = self._decide_chunking(profile, workload, timing, parallelism)

        # Step 5: 计算性能预期
        speedup, analysis = self._compute_speedup(profile, workload, timing)

        # Step 6: 检查性能保证
        if speedup < self.config.min_speedup:
            # 压缩无法带来加速，返回不压缩
            plan = ExecutionPlan.create_no_compress_plan()
            plan.add_warning(
                f"Compression would not improve performance. "
                f"Expected speedup: {speedup:.2f}x < {self.config.min_speedup}x"
            )
            plan.debug_info = analysis
            return plan

        # Step 7: 生成执行计划
        plan = self._generate_plan(
            method, profile, workload, timing, parallelism, chunking, speedup, analysis
        )

        return plan

    def _should_compress(self, workload: WorkloadContext) -> bool:
        """判断是否应该尝试压缩"""
        # 序列太短
        if workload.seq_len < self.config.min_seq_len:
            return False

        # 输出太短（压缩开销无法摊销）
        if workload.expected_output_len < self.config.min_output_len:
            return False

        return True

    def _decide_timing(self, profile: CompressMethodProfile,
                       workload: WorkloadContext) -> ExecutionTiming:
        """
        决定执行时机

        决策逻辑：
        1. 需要attention结果的 → INLINE_PREFILL 或 AFTER_PREFILL
        2. GEMM类型 + Decode Memory-bound → ASYNC_DECODE
        3. Memory类型 → AFTER_PREFILL（避免与Decode竞争）
        4. 低开销方法 → INLINE_PREFILL
        """

        # 规则1: 需要attention结果的方法
        if profile.requires_attention:
            compress_latency = profile.estimate_latency(workload.seq_len, workload.batch_size)
            if compress_latency < 1.0:  # < 1ms，可以inline
                return ExecutionTiming.INLINE_PREFILL
            else:
                return ExecutionTiming.AFTER_PREFILL

        # 规则2: 判断Decode的Memory-bound程度
        decode_is_memory_bound = self.hw_profile.is_decode_memory_bound(
            workload.batch_size,
            workload.seq_len,
            workload.num_kv_heads,
            workload.head_dim
        )

        # 规则3: 根据计算类型和互补性决定
        if profile.compute_type == ComputeType.GEMM:
            # GEMM类型：检查是否可以与Decode并行
            if decode_is_memory_bound and profile.decode_compatibility > 0.5:
                return ExecutionTiming.ASYNC_DECODE
            else:
                return ExecutionTiming.AFTER_PREFILL

        elif profile.compute_type == ComputeType.MEMORY:
            # Memory类型：避免与Decode竞争
            compress_latency = profile.estimate_latency(workload.seq_len, workload.batch_size)
            if compress_latency < 0.5:  # 极低开销，可以inline
                return ExecutionTiming.INLINE_PREFILL
            else:
                return ExecutionTiming.AFTER_PREFILL

        else:  # MIXED
            if decode_is_memory_bound and profile.decode_compatibility > 0.3:
                return ExecutionTiming.ASYNC_DECODE
            else:
                return ExecutionTiming.AFTER_PREFILL

    def _decide_parallelism(self, profile: CompressMethodProfile,
                            workload: WorkloadContext,
                            timing: ExecutionTiming) -> ParallelismConfig:
        """决定并行策略"""

        if timing == ExecutionTiming.ASYNC_DECODE:
            return ParallelismConfig(
                mode=ParallelismMode.ASYNC_STREAM,
                stream_priority=-1,  # 低优先级
                num_streams=1,
                use_events=True,
                sync_after_chunks=self.config.latency_check_interval,
            )

        elif timing == ExecutionTiming.INLINE_PREFILL:
            fusion_point = 'after_attention' if profile.requires_attention else 'after_layer'
            return ParallelismConfig(
                mode=ParallelismMode.FUSED,
                fusion_point=fusion_point,
            )

        else:  # AFTER_PREFILL
            return ParallelismConfig(mode=ParallelismMode.SYNC)

    def _decide_chunking(self, profile: CompressMethodProfile,
                         workload: WorkloadContext,
                         timing: ExecutionTiming,
                         parallelism: ParallelismConfig) -> ChunkingConfig:
        """决定分块策略"""

        # 不支持分块的方法
        if not profile.chunkable:
            return ChunkingConfig(enabled=False)

        # 同步模式不需要分块
        if parallelism.mode == ParallelismMode.SYNC:
            return ChunkingConfig(enabled=False)

        # Inline模式不需要分块
        if parallelism.mode == ParallelismMode.FUSED:
            return ChunkingConfig(enabled=False)

        # 异步模式：计算最优分块大小
        if parallelism.mode == ParallelismMode.ASYNC_STREAM:
            return self._compute_optimal_chunking(profile, workload)

        return ChunkingConfig(enabled=False)

    def _compute_optimal_chunking(self, profile: CompressMethodProfile,
                                   workload: WorkloadContext) -> ChunkingConfig:
        """
        计算最优分块大小

        目标：每个chunk的执行时间 ≈ 一次Decode的时间
        这样可以最大化重叠
        """
        # 估算Decode延迟
        decode_latency = self.cost_model.estimate_decode_latency(workload, compressed=False)

        # 估算总压缩延迟
        total_compress_latency = profile.estimate_latency(workload.seq_len, workload.batch_size)

        # 计算理想chunk数
        if decode_latency > 0:
            ideal_chunks = max(1, int(total_compress_latency / decode_latency))
        else:
            ideal_chunks = workload.num_layers // self.config.default_chunk_size

        # 限制chunk数
        ideal_chunks = min(ideal_chunks, self.config.max_chunks)
        ideal_chunks = max(ideal_chunks, 1)

        # 计算chunk大小
        chunk_size = max(
            profile.min_chunk_size,
            workload.num_layers // ideal_chunks
        )
        chunk_size = min(chunk_size, profile.max_chunk_size)

        # 实际chunk数
        num_chunks = (workload.num_layers + chunk_size - 1) // chunk_size

        return ChunkingConfig(
            enabled=True,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            interleave_with='decode'
        )

    def _compute_speedup(self, profile: CompressMethodProfile,
                         workload: WorkloadContext,
                         timing: ExecutionTiming) -> Tuple[float, Dict[str, Any]]:
        """计算预期加速比"""

        async_mode = (timing == ExecutionTiming.ASYNC_DECODE)

        speedup, analysis = self.cost_model.compute_speedup(
            workload, profile.name, async_mode=async_mode
        )

        # 添加额外分析
        analysis['timing'] = timing.name
        analysis['compute_type'] = profile.compute_type.name
        analysis['decode_compatibility'] = profile.decode_compatibility

        return speedup, analysis

    def _generate_plan(self, method: str,
                       profile: CompressMethodProfile,
                       workload: WorkloadContext,
                       timing: ExecutionTiming,
                       parallelism: ParallelismConfig,
                       chunking: ChunkingConfig,
                       speedup: float,
                       analysis: Dict[str, Any]) -> ExecutionPlan:
        """生成执行计划"""

        compress_latency = profile.estimate_latency(workload.seq_len, workload.batch_size)

        if timing == ExecutionTiming.INLINE_PREFILL:
            plan = ExecutionPlan.create_inline_plan(method, compress_latency, speedup)

        elif timing == ExecutionTiming.AFTER_PREFILL:
            plan = ExecutionPlan.create_sync_plan(method, compress_latency, speedup)

        elif timing == ExecutionTiming.ASYNC_DECODE:
            plan = ExecutionPlan.create_async_plan(
                method, compress_latency, speedup,
                chunk_size=chunking.chunk_size,
                num_layers=workload.num_layers
            )

        else:
            plan = ExecutionPlan.create_no_compress_plan()

        # 更新配置
        plan.parallelism = parallelism
        plan.chunking = chunking
        plan.debug_info = analysis

        # 添加降级检查
        if self.config.enable_fallback:
            self._add_fallback_checks(plan, workload)

        return plan

    def _add_fallback_checks(self, plan: ExecutionPlan,
                              workload: WorkloadContext):
        """添加降级检查步骤"""

        if plan.timing != ExecutionTiming.ASYNC_DECODE:
            return

        # 在异步执行中添加延迟检查
        # 如果检测到Decode延迟异常增加，可以提前切换到压缩KV
        for i, step in enumerate(plan.steps):
            if step.type == StepType.CHECK_PROGRESS:
                step.config['fallback_enabled'] = True
                step.config['latency_threshold_ms'] = (
                    self.cost_model.estimate_decode_latency(workload, compressed=False) * 1.2
                )

    def schedule_multi(self, methods: List[str],
                       workload: WorkloadContext) -> ExecutionPlan:
        """
        为多个候选方法选择最优方案

        Args:
            methods: 候选压缩方法列表
            workload: Workload上下文

        Returns:
            最优的ExecutionPlan
        """
        best_plan = ExecutionPlan.create_no_compress_plan()
        best_speedup = 1.0

        for method in methods:
            try:
                plan = self.schedule(method, workload)
                if plan.estimated_speedup > best_speedup:
                    best_speedup = plan.estimated_speedup
                    best_plan = plan
            except ValueError:
                continue

        return best_plan

    def analyze(self, method: str,
                workload: WorkloadContext) -> Dict[str, Any]:
        """
        分析压缩方案（不生成执行计划）

        用于调试和理解调度决策
        """
        profile = get_profile(method)

        # 基础分析
        decode_is_memory_bound = self.hw_profile.is_decode_memory_bound(
            workload.batch_size,
            workload.seq_len,
            workload.num_kv_heads,
            workload.head_dim
        )

        compress_latency = profile.estimate_latency(workload.seq_len, workload.batch_size)
        decode_latency_original = self.cost_model.estimate_decode_latency(workload, compressed=False)
        decode_latency_compressed = self.cost_model.estimate_decode_latency(
            workload, compressed=True, compression_ratio=profile.compression_ratio
        )

        # 计算break-even点
        break_even = self.cost_model.get_break_even_output_len(workload, method)

        # 各模式的加速比
        speedup_sync, _ = self.cost_model.compute_speedup(workload, method, async_mode=False)
        speedup_async, _ = self.cost_model.compute_speedup(workload, method, async_mode=True)

        return {
            'method': method,
            'profile': {
                'compute_type': profile.compute_type.name,
                'decode_compatibility': profile.decode_compatibility,
                'compression_ratio': profile.compression_ratio,
                'chunkable': profile.chunkable,
                'requires_attention': profile.requires_attention,
            },
            'workload': {
                'batch_size': workload.batch_size,
                'seq_len': workload.seq_len,
                'expected_output_len': workload.expected_output_len,
                'kv_cache_size_mb': workload.kv_cache_size_mb,
            },
            'hardware': {
                'gpu_name': self.hw_profile.gpu_name,
                'roofline_intensity': self.hw_profile.roofline_intensity,
                'decode_is_memory_bound': decode_is_memory_bound,
            },
            'latency': {
                'compress_latency_ms': compress_latency,
                'decode_original_ms': decode_latency_original,
                'decode_compressed_ms': decode_latency_compressed,
            },
            'performance': {
                'break_even_tokens': break_even,
                'speedup_sync': speedup_sync,
                'speedup_async': speedup_async,
                'recommended_mode': 'async' if speedup_async > speedup_sync else 'sync',
            },
        }


# ============ 便捷函数 ============

def auto_schedule(method: str,
                  batch_size: int = 1,
                  seq_len: int = 512,
                  expected_output_len: int = 100,
                  num_layers: int = 32,
                  num_kv_heads: int = 8,
                  head_dim: int = 128) -> ExecutionPlan:
    """
    快速调度函数

    Example:
        ```python
        plan = auto_schedule('mlp', batch_size=4, seq_len=2048, expected_output_len=200)
        print(plan.summary())
        ```
    """
    workload = WorkloadContext(
        batch_size=batch_size,
        seq_len=seq_len,
        expected_output_len=expected_output_len,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    scheduler = AutoCompressScheduler()
    return scheduler.schedule(method, workload)


def analyze_compression(method: str,
                        batch_size: int = 1,
                        seq_len: int = 512,
                        expected_output_len: int = 100) -> Dict[str, Any]:
    """
    快速分析函数

    Example:
        ```python
        analysis = analyze_compression('mlp', batch_size=4, seq_len=2048)
        print(analysis)
        ```
    """
    workload = WorkloadContext(
        batch_size=batch_size,
        seq_len=seq_len,
        expected_output_len=expected_output_len,
    )

    scheduler = AutoCompressScheduler()
    return scheduler.analyze(method, workload)
