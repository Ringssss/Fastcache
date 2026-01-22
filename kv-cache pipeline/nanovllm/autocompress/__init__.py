"""
AutoCompress: 自动化KV-Cache压缩流水线调度系统

核心设计理念：
- 用户负责选择压缩方法
- 系统自动找到最优的执行位置和流水线编排
- 性能保证：压缩后不比不压缩慢

支持的压缩方法：
- MLP-based: BatchedGEMM压缩（GEMM密集型，与Decode互补）
- kvpress库: SnapKV, StreamingLLM, H2O, TOVA, ThinK, SimLayerKV等
- 量化方法: INT8, INT4, FP8, NF4

Pipeline策略：
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

快速开始：
```python
from nanovllm.autocompress import (
    PipelineOrchestrator,
    analyze_all_methods,
    create_optimal_pipeline,
)

# 方式1: 分析所有方法的最优策略
analysis = analyze_all_methods(batch_size=4, seq_len=2048, expected_output_len=100)
for method, info in analysis.items():
    print(f"{method}: {info['strategy']} -> speedup={info['estimated_speedup']:.2f}x")

# 方式2: 创建最优Pipeline
config = create_optimal_pipeline('mlp', batch_size=4, seq_len=2048)
print(f"Strategy: {config.strategy.name}")
print(f"Timing: {config.timing.name}")

# 方式3: 使用Pipeline编排器
orchestrator = PipelineOrchestrator()
composed = orchestrator.create_composed_pipeline(['snapkv', 'mlp'], workload)
print(f"Total compression: {composed.total_compression_ratio:.1f}x")
```
"""

# 基础抽象
from .profiles import (
    CompressMethodProfile,
    ComputeType,
    BUILTIN_PROFILES,
    register_profile,
    get_profile,
    list_profiles,
)

# 上下文
from .context import (
    WorkloadContext,
    HardwareProfile,
    GPUArch,
    detect_hardware,
    get_hardware_profile,
)

# 性能模型
from .cost_model import (
    CostModel,
    CalibrationData,
    should_compress,
)

# 执行计划
from .execution import (
    ExecutionPlan,
    ExecutionStep,
    ExecutionTiming,
    ParallelismMode,
    ParallelismConfig,
    ChunkingConfig,
    StepType,
    PipelineExecutor,
    StreamManager,
    get_stream_manager,
)

# 调度器
from .scheduler import (
    AutoCompressScheduler,
    SchedulerConfig,
    auto_schedule,
    analyze_compression,
)

# Kernel生成
from .kernels import (
    TileLangKernelGenerator,
    KernelConfig,
    get_kernel_generator,
    get_mlp_kernel,
    get_sparse_gather_kernel,
    get_quantize_kernels,
)

# 集成
from .integration import (
    CompressionConfig,
    AutoCompressManager,
    AsyncCompressionHandle,
    AdaptiveCompressionWrapper,
    create_auto_compress_manager,
    wrap_compressor,
    analyze_workload,
)

# 方法适配器
from .methods import (
    CompressionMethod,
    MLPCompressionMethod,
    KVPressMethod,
    QuantizationMethod,
    MethodRegistry,
    get_method,
    list_methods,
)

# Pipeline编排器
from .pipeline import (
    PipelineStrategy,
    PipelineConfig,
    ComposedPipelineConfig,
    PipelineOrchestrator,
    create_optimal_pipeline,
    analyze_all_methods,
)

__all__ = [
    # Profiles
    'CompressMethodProfile',
    'ComputeType',
    'BUILTIN_PROFILES',
    'register_profile',
    'get_profile',
    'list_profiles',

    # Context
    'WorkloadContext',
    'HardwareProfile',
    'GPUArch',
    'detect_hardware',
    'get_hardware_profile',

    # Cost Model
    'CostModel',
    'CalibrationData',
    'should_compress',

    # Execution
    'ExecutionPlan',
    'ExecutionStep',
    'ExecutionTiming',
    'ParallelismMode',
    'ParallelismConfig',
    'ChunkingConfig',
    'StepType',
    'PipelineExecutor',
    'StreamManager',
    'get_stream_manager',

    # Scheduler
    'AutoCompressScheduler',
    'SchedulerConfig',
    'auto_schedule',
    'analyze_compression',

    # Kernels
    'TileLangKernelGenerator',
    'KernelConfig',
    'get_kernel_generator',
    'get_mlp_kernel',
    'get_sparse_gather_kernel',
    'get_quantize_kernels',

    # Integration
    'CompressionConfig',
    'AutoCompressManager',
    'AsyncCompressionHandle',
    'AdaptiveCompressionWrapper',
    'create_auto_compress_manager',
    'wrap_compressor',
    'analyze_workload',

    # Methods
    'CompressionMethod',
    'MLPCompressionMethod',
    'KVPressMethod',
    'QuantizationMethod',
    'MethodRegistry',
    'get_method',
    'list_methods',

    # Pipeline
    'PipelineStrategy',
    'PipelineConfig',
    'ComposedPipelineConfig',
    'PipelineOrchestrator',
    'create_optimal_pipeline',
    'analyze_all_methods',
]

__version__ = '0.1.0'
