"""
统一的Pipeline编排器

核心功能：
1. 根据压缩方法的特性，自动选择最优的Pipeline策略
2. 支持组合多种压缩方法
3. 保证每种方法都以最优的形式执行

Pipeline策略分类：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
方法类型         │ 计算特性      │ 最优Pipeline策略
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MLP压缩         │ GEMM (计算密集) │ ASYNC_DECODE (与Decode并行)
SnapKV/H2O     │ MIXED         │ INLINE_PREFILL (利用attention结果)
StreamingLLM   │ MEMORY (极低)  │ INLINE_PREFILL (几乎零开销)
量化方法        │ MEMORY        │ INLINE_PREFILL (逐层量化)
ThinK          │ MIXED         │ AFTER_PREFILL (需要额外计算)
组合策略        │ 混合          │ 按依赖顺序分阶段执行
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum, auto
import math

import torch
import torch.nn as nn

from .profiles import CompressMethodProfile, ComputeType, get_profile
from .context import WorkloadContext, HardwareProfile, detect_hardware
from .cost_model import CostModel
from .execution import (
    ExecutionPlan, ExecutionStep, ExecutionTiming,
    ParallelismConfig, ParallelismMode, ChunkingConfig, StepType,
    get_stream_manager,
)
from .methods import (
    CompressionMethod, get_method, list_methods,
    MLPCompressionMethod, KVPressMethod, QuantizationMethod,
)


class PipelineStrategy(Enum):
    """Pipeline策略类型"""
    INLINE_PREFILL = auto()      # 融合进Prefill，几乎零开销
    SYNC_AFTER_PREFILL = auto()  # Prefill后同步执行
    ASYNC_WITH_DECODE = auto()   # 与Decode异步并行
    PROGRESSIVE = auto()         # 渐进式（逐层/分块）
    COMPOSED = auto()            # 组合策略


@dataclass
class PipelineConfig:
    """Pipeline配置"""
    strategy: PipelineStrategy
    method: CompressionMethod
    timing: ExecutionTiming

    # 分块配置（如果需要）
    chunk_size: int = 4
    num_chunks: int = 8

    # 异步配置
    stream_priority: int = -1
    use_events: bool = True

    # 性能预测
    estimated_speedup: float = 1.0
    estimated_latency_ms: float = 0.0

    # 依赖关系（用于组合策略）
    depends_on: Optional[str] = None


@dataclass
class ComposedPipelineConfig:
    """组合Pipeline配置"""
    methods: List[PipelineConfig]
    total_compression_ratio: float = 1.0
    estimated_speedup: float = 1.0


class PipelineOrchestrator:
    """
    Pipeline编排器

    负责：
    1. 分析每种压缩方法的最优执行策略
    2. 生成最优的Pipeline配置
    3. 执行Pipeline

    使用示例：
    ```python
    orchestrator = PipelineOrchestrator()

    # 单方法
    config = orchestrator.create_pipeline('mlp', workload)
    result = orchestrator.execute(config, kv_cache)

    # 组合方法
    config = orchestrator.create_composed_pipeline(
        ['snapkv', 'mlp'],
        workload
    )
    result = orchestrator.execute_composed(config, kv_cache)
    ```
    """

    def __init__(self,
                 hw_profile: Optional[HardwareProfile] = None,
                 min_speedup: float = 1.1):
        self.hw_profile = hw_profile or detect_hardware()
        self.cost_model = CostModel(self.hw_profile)
        self.min_speedup = min_speedup

        # 策略选择规则
        self._strategy_rules = self._build_strategy_rules()

    def _build_strategy_rules(self) -> Dict[str, Callable]:
        """构建策略选择规则"""
        return {
            'mlp': self._mlp_strategy,
            'snapkv': self._snapkv_strategy,
            'streaming_llm': self._streaming_strategy,
            'observed_attention': self._attention_based_strategy,
            'expected_attention': self._expected_attention_strategy,
            'h2o': self._h2o_strategy,
            'knorm': self._knorm_strategy,
            'tova': self._tova_strategy,
            'think': self._think_strategy,
            'simlayer': self._simlayer_strategy,
            'quant_int8': self._quant_strategy,
            'quant_int4': self._quant_strategy,
            'quant_fp8': self._quant_strategy,
            'quant_nf4': self._quant_strategy,
        }

    def create_pipeline(self,
                        method_name: str,
                        workload: WorkloadContext,
                        **method_kwargs) -> PipelineConfig:
        """
        为单个方法创建最优Pipeline配置

        Args:
            method_name: 方法名称
            workload: 工作负载
            **method_kwargs: 方法参数

        Returns:
            PipelineConfig
        """
        # 获取方法实例
        method = get_method(method_name, **method_kwargs)

        # 选择策略
        strategy, timing = self._select_strategy(method, workload)

        # 计算性能预期
        speedup = self._estimate_speedup(method, workload, timing)
        latency = method.profile.estimate_latency(workload.seq_len, workload.batch_size)

        # 配置分块（如果需要）
        chunk_size, num_chunks = self._compute_chunking(method, workload, strategy)

        return PipelineConfig(
            strategy=strategy,
            method=method,
            timing=timing,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            stream_priority=-1 if strategy == PipelineStrategy.ASYNC_WITH_DECODE else 0,
            use_events=strategy == PipelineStrategy.ASYNC_WITH_DECODE,
            estimated_speedup=speedup,
            estimated_latency_ms=latency,
        )

    def _select_strategy(self,
                         method: CompressionMethod,
                         workload: WorkloadContext) -> Tuple[PipelineStrategy, ExecutionTiming]:
        """
        选择最优策略

        这是核心决策逻辑，确保每种方法都以最优的形式执行。
        """
        profile = method.profile

        # 获取方法特定的策略规则
        method_name = method.name.replace('kvpress_', '').replace('quant_', 'quant_')
        if method_name in self._strategy_rules:
            return self._strategy_rules[method_name](method, workload)

        # 默认策略选择
        return self._default_strategy(method, workload)

    def _default_strategy(self,
                          method: CompressionMethod,
                          workload: WorkloadContext) -> Tuple[PipelineStrategy, ExecutionTiming]:
        """默认策略选择逻辑"""
        profile = method.profile
        latency = profile.estimate_latency(workload.seq_len, workload.batch_size)

        # 判断Decode是否Memory-bound
        decode_memory_bound = self.hw_profile.is_decode_memory_bound(
            workload.batch_size, workload.seq_len,
            workload.num_kv_heads, workload.head_dim
        )

        # GEMM类型 + Memory-bound Decode → 异步并行
        if profile.compute_type == ComputeType.GEMM:
            if decode_memory_bound and profile.decode_compatibility > 0.5 and workload.batch_size >= 4:
                return PipelineStrategy.ASYNC_WITH_DECODE, ExecutionTiming.ASYNC_DECODE
            return PipelineStrategy.SYNC_AFTER_PREFILL, ExecutionTiming.AFTER_PREFILL

        # Memory类型 → 尽量inline
        if profile.compute_type == ComputeType.MEMORY:
            if latency < 0.5:
                return PipelineStrategy.INLINE_PREFILL, ExecutionTiming.INLINE_PREFILL
            return PipelineStrategy.SYNC_AFTER_PREFILL, ExecutionTiming.AFTER_PREFILL

        # Mixed类型 → 取决于attention需求
        if profile.requires_attention:
            return PipelineStrategy.INLINE_PREFILL, ExecutionTiming.INLINE_PREFILL

        return PipelineStrategy.SYNC_AFTER_PREFILL, ExecutionTiming.AFTER_PREFILL

    # ========== 方法特定的策略规则 ==========

    def _mlp_strategy(self,
                      method: CompressionMethod,
                      workload: WorkloadContext) -> Tuple[PipelineStrategy, ExecutionTiming]:
        """
        MLP压缩的最优策略

        MLP是GEMM密集型，与Memory-bound的Decode互补。
        但小batch时，压缩本身效率也低。
        """
        decode_memory_bound = self.hw_profile.is_decode_memory_bound(
            workload.batch_size, workload.seq_len,
            workload.num_kv_heads, workload.head_dim
        )

        # 大batch + Memory-bound Decode → 异步并行
        if workload.batch_size >= 4 and decode_memory_bound:
            return PipelineStrategy.ASYNC_WITH_DECODE, ExecutionTiming.ASYNC_DECODE

        # 大batch但Decode是compute-bound → 还是异步，但优先级调整
        if workload.batch_size >= 8:
            return PipelineStrategy.ASYNC_WITH_DECODE, ExecutionTiming.ASYNC_DECODE

        # 小batch → 同步执行
        return PipelineStrategy.SYNC_AFTER_PREFILL, ExecutionTiming.AFTER_PREFILL

    def _snapkv_strategy(self,
                         method: CompressionMethod,
                         workload: WorkloadContext) -> Tuple[PipelineStrategy, ExecutionTiming]:
        """
        SnapKV的最优策略

        SnapKV需要计算窗口attention，但可以复用Prefill的结果。
        """
        # 如果可以复用attention结果 → inline
        # 否则 → after prefill
        return PipelineStrategy.INLINE_PREFILL, ExecutionTiming.INLINE_PREFILL

    def _streaming_strategy(self,
                            method: CompressionMethod,
                            workload: WorkloadContext) -> Tuple[PipelineStrategy, ExecutionTiming]:
        """
        StreamingLLM的最优策略

        几乎零开销，只是索引选择。
        """
        return PipelineStrategy.INLINE_PREFILL, ExecutionTiming.INLINE_PREFILL

    def _attention_based_strategy(self,
                                  method: CompressionMethod,
                                  workload: WorkloadContext) -> Tuple[PipelineStrategy, ExecutionTiming]:
        """
        基于Attention的方法的策略（ObservedAttention, H2O）

        需要attention权重，必须inline。
        """
        return PipelineStrategy.INLINE_PREFILL, ExecutionTiming.INLINE_PREFILL

    def _expected_attention_strategy(self,
                                     method: CompressionMethod,
                                     workload: WorkloadContext) -> Tuple[PipelineStrategy, ExecutionTiming]:
        """
        ExpectedAttention的策略

        需要计算协方差矩阵，计算量较大，after prefill更好。
        """
        return PipelineStrategy.SYNC_AFTER_PREFILL, ExecutionTiming.AFTER_PREFILL

    def _h2o_strategy(self,
                      method: CompressionMethod,
                      workload: WorkloadContext) -> Tuple[PipelineStrategy, ExecutionTiming]:
        """H2O策略"""
        return PipelineStrategy.INLINE_PREFILL, ExecutionTiming.INLINE_PREFILL

    def _knorm_strategy(self,
                        method: CompressionMethod,
                        workload: WorkloadContext) -> Tuple[PipelineStrategy, ExecutionTiming]:
        """Knorm策略 - 轻量级，inline"""
        return PipelineStrategy.INLINE_PREFILL, ExecutionTiming.INLINE_PREFILL

    def _tova_strategy(self,
                       method: CompressionMethod,
                       workload: WorkloadContext) -> Tuple[PipelineStrategy, ExecutionTiming]:
        """TOVA策略"""
        return PipelineStrategy.INLINE_PREFILL, ExecutionTiming.INLINE_PREFILL

    def _think_strategy(self,
                        method: CompressionMethod,
                        workload: WorkloadContext) -> Tuple[PipelineStrategy, ExecutionTiming]:
        """
        ThinK策略

        维度压缩，需要额外计算，after prefill。
        """
        return PipelineStrategy.SYNC_AFTER_PREFILL, ExecutionTiming.AFTER_PREFILL

    def _simlayer_strategy(self,
                           method: CompressionMethod,
                           workload: WorkloadContext) -> Tuple[PipelineStrategy, ExecutionTiming]:
        """SimLayerKV策略"""
        return PipelineStrategy.SYNC_AFTER_PREFILL, ExecutionTiming.AFTER_PREFILL

    def _quant_strategy(self,
                        method: CompressionMethod,
                        workload: WorkloadContext) -> Tuple[PipelineStrategy, ExecutionTiming]:
        """
        量化方法的策略

        量化操作是memory-bound，推荐在Prefill期间逐层执行。
        """
        latency = method.profile.estimate_latency(workload.seq_len, workload.batch_size)

        # 短序列或低延迟 → inline
        if latency < 2.0 or workload.seq_len < 1024:
            return PipelineStrategy.PROGRESSIVE, ExecutionTiming.INLINE_PREFILL

        # 长序列 → after prefill
        return PipelineStrategy.SYNC_AFTER_PREFILL, ExecutionTiming.AFTER_PREFILL

    def _compute_chunking(self,
                          method: CompressionMethod,
                          workload: WorkloadContext,
                          strategy: PipelineStrategy) -> Tuple[int, int]:
        """计算分块配置"""
        profile = method.profile

        if not profile.chunkable:
            return 1, 1

        if strategy != PipelineStrategy.ASYNC_WITH_DECODE:
            return workload.num_layers, 1

        # 异步模式：计算最优分块
        decode_latency = self.cost_model.estimate_decode_latency(workload)
        compress_latency = profile.estimate_latency(workload.seq_len, workload.batch_size)

        if decode_latency > 0:
            ideal_chunks = max(1, int(compress_latency / decode_latency))
        else:
            ideal_chunks = 8

        ideal_chunks = min(ideal_chunks, 16)
        chunk_size = max(profile.min_chunk_size, workload.num_layers // ideal_chunks)
        num_chunks = (workload.num_layers + chunk_size - 1) // chunk_size

        return chunk_size, num_chunks

    def _estimate_speedup(self,
                          method: CompressionMethod,
                          workload: WorkloadContext,
                          timing: ExecutionTiming) -> float:
        """估算加速比"""
        profile = method.profile
        async_mode = (timing == ExecutionTiming.ASYNC_DECODE)

        # 估算各项延迟
        compress_latency = profile.estimate_latency(workload.seq_len, workload.batch_size)
        decode_latency_original = self.cost_model.estimate_decode_latency(workload, compressed=False)
        decode_latency_compressed = self.cost_model.estimate_decode_latency(
            workload, compressed=True, compression_ratio=profile.compression_ratio
        )

        output_len = workload.expected_output_len

        # 不压缩的总时间
        time_no_compress = decode_latency_original * output_len

        if async_mode:
            # 异步模式：压缩与前几个Decode重叠
            overlap_tokens = min(
                output_len,
                int(compress_latency / max(decode_latency_original, 0.001)) + 1
            )
            parallel_phase = max(compress_latency, decode_latency_original * overlap_tokens)
            remaining_tokens = output_len - overlap_tokens
            sequential_phase = decode_latency_compressed * remaining_tokens
            time_with_compress = parallel_phase + sequential_phase
        else:
            # 同步模式：压缩后再Decode
            time_with_compress = compress_latency + decode_latency_compressed * output_len

        speedup = time_no_compress / max(time_with_compress, 0.001)
        return speedup

    # ========== 组合Pipeline ==========

    def create_composed_pipeline(self,
                                 method_names: List[str],
                                 workload: WorkloadContext,
                                 method_kwargs: Optional[Dict[str, Dict]] = None) -> ComposedPipelineConfig:
        """
        创建组合Pipeline

        支持多种压缩方法的组合，如：
        - SnapKV + MLP: 先稀疏化再MLP压缩
        - StreamingLLM + 量化: 先截断再量化
        - ThinK + SnapKV: 维度压缩 + 序列压缩

        方法会按照最优顺序执行。
        """
        method_kwargs = method_kwargs or {}

        # 创建各方法的Pipeline配置
        configs = []
        for name in method_names:
            kwargs = method_kwargs.get(name, {})
            config = self.create_pipeline(name, workload, **kwargs)
            configs.append(config)

        # 按执行顺序排序
        configs = self._sort_by_execution_order(configs)

        # 计算总压缩比
        total_ratio = 1.0
        for config in configs:
            total_ratio *= config.method.profile.compression_ratio

        # 估算总加速比（考虑组合效应）
        total_speedup = self._estimate_composed_speedup(configs, workload)

        return ComposedPipelineConfig(
            methods=configs,
            total_compression_ratio=total_ratio,
            estimated_speedup=total_speedup,
        )

    def _sort_by_execution_order(self,
                                 configs: List[PipelineConfig]) -> List[PipelineConfig]:
        """
        按执行顺序排序

        排序规则：
        1. INLINE_PREFILL先执行
        2. 稀疏化方法先于GEMM压缩（减少后续计算量）
        3. 量化方法最后执行
        """
        def sort_key(config):
            # 按timing排序
            timing_order = {
                ExecutionTiming.INLINE_PREFILL: 0,
                ExecutionTiming.AFTER_PREFILL: 1,
                ExecutionTiming.ASYNC_DECODE: 2,
            }

            # 按方法类型排序
            type_order = {
                ComputeType.MEMORY: 0,   # 稀疏化类先执行
                ComputeType.MIXED: 1,
                ComputeType.GEMM: 2,     # GEMM类后执行
            }

            return (
                timing_order.get(config.timing, 1),
                type_order.get(config.method.profile.compute_type, 1),
            )

        return sorted(configs, key=sort_key)

    def _estimate_composed_speedup(self,
                                   configs: List[PipelineConfig],
                                   workload: WorkloadContext) -> float:
        """估算组合策略的加速比"""
        # 简化估算：假设各方法的收益可以累积（实际上可能有折扣）
        total_speedup = 1.0

        # 更新workload中的seq_len以反映压缩效果
        current_seq_len = workload.seq_len

        for config in configs:
            # 估算当前方法的加速
            temp_workload = WorkloadContext(
                batch_size=workload.batch_size,
                seq_len=current_seq_len,
                expected_output_len=workload.expected_output_len,
                num_layers=workload.num_layers,
                num_kv_heads=workload.num_kv_heads,
                head_dim=workload.head_dim,
            )

            speedup = self._estimate_speedup(config.method, temp_workload, config.timing)

            # 组合效应：后续方法基于压缩后的数据
            total_speedup *= speedup ** 0.8  # 添加折扣因子

            # 更新seq_len
            if config.method.profile.modifies_seq_len:
                current_seq_len = int(current_seq_len / config.method.profile.compression_ratio)

        return total_speedup

    # ========== 执行 ==========

    def execute(self,
                config: PipelineConfig,
                keys: torch.Tensor,
                values: torch.Tensor,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行单方法Pipeline

        Args:
            config: Pipeline配置
            keys: Key张量
            values: Value张量
            **kwargs: 额外参数

        Returns:
            (compressed_keys, compressed_values)
        """
        method = config.method

        if config.strategy == PipelineStrategy.ASYNC_WITH_DECODE:
            return self._execute_async(config, keys, values, **kwargs)
        else:
            return method.compress(keys, values, **kwargs)

    def _execute_async(self,
                       config: PipelineConfig,
                       keys: torch.Tensor,
                       values: torch.Tensor,
                       **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """异步执行"""
        stream_manager = get_stream_manager()
        compress_stream = stream_manager.get_compress_stream()

        with torch.cuda.stream(compress_stream):
            result = config.method.compress(keys, values, **kwargs)

        # 创建事件用于同步
        event = torch.cuda.Event()
        event.record(compress_stream)

        return result, event  # 返回结果和事件

    def execute_composed(self,
                         config: ComposedPipelineConfig,
                         keys: torch.Tensor,
                         values: torch.Tensor,
                         **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行组合Pipeline

        按照配置的顺序依次执行各方法。
        """
        current_keys = keys
        current_values = values

        for method_config in config.methods:
            current_keys, current_values = self.execute(
                method_config, current_keys, current_values, **kwargs
            )

            # 处理异步返回
            if isinstance(current_keys, tuple) and len(current_keys) == 2:
                result, event = current_keys
                event.synchronize()
                current_keys = result

        return current_keys, current_values


# ============================================================================
# 便捷函数
# ============================================================================

def create_optimal_pipeline(method: str,
                            batch_size: int,
                            seq_len: int,
                            expected_output_len: int = 100,
                            **method_kwargs) -> PipelineConfig:
    """
    快速创建最优Pipeline

    Example:
        ```python
        config = create_optimal_pipeline('mlp', batch_size=4, seq_len=2048)
        print(f"Strategy: {config.strategy.name}")
        print(f"Expected speedup: {config.estimated_speedup:.2f}x")
        ```
    """
    workload = WorkloadContext(
        batch_size=batch_size,
        seq_len=seq_len,
        expected_output_len=expected_output_len,
    )

    orchestrator = PipelineOrchestrator()
    return orchestrator.create_pipeline(method, workload, **method_kwargs)


def analyze_all_methods(batch_size: int,
                        seq_len: int,
                        expected_output_len: int = 100) -> Dict[str, Dict[str, Any]]:
    """
    分析所有方法的最优策略

    Returns:
        方法名到分析结果的映射
    """
    workload = WorkloadContext(
        batch_size=batch_size,
        seq_len=seq_len,
        expected_output_len=expected_output_len,
    )

    orchestrator = PipelineOrchestrator()
    results = {}

    methods_to_analyze = [
        'mlp',
        'kvpress_snapkv',
        'kvpress_streaming_llm',
        'kvpress_h2o',
        'kvpress_knorm',
        'kvpress_tova',
        'kvpress_think',
        'kvpress_simlayer',
        'quant_int8',
        'quant_int4',
        'quant_fp8',
    ]

    for method_name in methods_to_analyze:
        try:
            config = orchestrator.create_pipeline(method_name, workload)
            results[method_name] = {
                'strategy': config.strategy.name,
                'timing': config.timing.name,
                'estimated_speedup': config.estimated_speedup,
                'estimated_latency_ms': config.estimated_latency_ms,
                'chunk_size': config.chunk_size,
                'num_chunks': config.num_chunks,
                'compute_type': config.method.profile.compute_type.name,
                'compression_ratio': config.method.profile.compression_ratio,
            }
        except Exception as e:
            results[method_name] = {'error': str(e)}

    return results
