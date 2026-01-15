#!/usr/bin/env python3
"""
AutoCompress 完整演示脚本

演示所有KV-Cache压缩方法的自动Pipeline编排：
- MLP-based压缩 (BatchedGEMM)
- kvpress库方法 (SnapKV, StreamingLLM, H2O, TOVA, ThinK, SimLayerKV等)
- 量化方法 (INT8, INT4, FP8, NF4)

核心设计理念：
- 用户负责选择压缩方法
- 系统自动找到最优的执行位置和流水线编排
- 性能保证：压缩后不比不压缩慢
"""

from fastcache_paths import ensure_sys_paths, CKPT_DIR, DATASETS_DIR, RESULTS_DIR

ensure_sys_paths()

import sys

from nanovllm.autocompress import (
    # Pipeline编排器
    PipelineOrchestrator,
    PipelineStrategy,
    PipelineConfig,
    analyze_all_methods,
    create_optimal_pipeline,

    # 方法
    CompressionMethod,
    MLPCompressionMethod,
    KVPressMethod,
    QuantizationMethod,
    get_method,
    list_methods,

    # 分析工具
    analyze_workload,
    analyze_compression,
    auto_schedule,

    # 核心组件
    AutoCompressScheduler,
    AutoCompressManager,
    CompressionConfig,
    WorkloadContext,

    # Profile
    list_profiles,
    get_profile,

    # 硬件检测
    detect_hardware,
)


def print_header(title: str, char: str = "="):
    """打印标题"""
    print()
    print(char * 80)
    print(f" {title}")
    print(char * 80)


def demo_hardware():
    """演示硬件检测"""
    print_header("硬件检测")

    hw = detect_hardware()
    print(f"\n检测到的GPU: {hw.gpu_name}")
    print(f"架构: {hw.gpu_arch.name}")
    print(f"Compute Capability: {hw.compute_capability}")
    print(f"FP16吞吐: {hw.fp16_tflops:.1f} TFLOPS")
    print(f"内存带宽: {hw.memory_bandwidth_gbps:.1f} GB/s")
    print(f"显存大小: {hw.memory_size_gb:.1f} GB")
    print(f"Roofline点: {hw.roofline_intensity:.1f} FLOP/Byte")


def demo_available_methods():
    """演示所有可用的压缩方法"""
    print_header("可用的压缩方法")

    methods = list_methods()
    print(f"\n共 {len(methods)} 种压缩方法可用:\n")

    # 按类型分组
    mlp_methods = [m for m in methods if 'mlp' in m]
    kvpress_methods = [m for m in methods if 'kvpress' in m]
    quant_methods = [m for m in methods if 'quant' in m]

    print("MLP压缩方法 (GEMM密集型):")
    for m in mlp_methods:
        print(f"  - {m}")

    print("\nkvpress稀疏化方法:")
    for m in kvpress_methods:
        print(f"  - {m}")

    print("\n量化方法:")
    for m in quant_methods:
        print(f"  - {m}")


def demo_pipeline_strategies():
    """演示Pipeline策略分析 - 核心功能"""
    print_header("Pipeline策略分析 (核心功能)", "━")

    print("""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                      Pipeline策略说明                                        ┃
┣━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ 策略                 ┃ 含义                                                  ┃
┣━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ INLINE_PREFILL       ┃ 融合进Prefill阶段，几乎零开销                          ┃
┃ SYNC_AFTER_PREFILL   ┃ Prefill后同步执行，有一定延迟                          ┃
┃ ASYNC_WITH_DECODE    ┃ 与Decode异步并行，利用计算/内存互补                     ┃
┃ PROGRESSIVE          ┃ 渐进式执行（分块/分层）                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
""")

    results = analyze_all_methods(
        batch_size=4,
        seq_len=2048,
        expected_output_len=100
    )

    # 按策略分组显示
    strategies = {}
    for method, info in results.items():
        if 'error' not in info:
            strategy = info['strategy']
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append((method, info))

    print("\n分析结果 (batch=4, seq_len=2048, output=100):\n")

    for strategy, methods in sorted(strategies.items()):
        print(f"\n【{strategy}】")
        print("-" * 70)
        print(f"{'方法':<25} | {'计算类型':<10} | {'压缩比':>8} | {'预期加速':>10}")
        print("-" * 70)
        for method, info in methods:
            print(f"{method:<25} | {info['compute_type']:<10} | {info['compression_ratio']:>6.1f}x | {info['estimated_speedup']:>8.2f}x")


def demo_workload_analysis():
    """演示不同Workload下的策略选择"""
    print_header("不同Workload下的策略选择")

    scenarios = [
        ('小batch短序列', 1, 512, 50),
        ('小batch长序列', 1, 4096, 200),
        ('大batch中序列', 8, 2048, 100),
        ('大batch长序列', 8, 4096, 200),
    ]

    key_methods = ['mlp', 'kvpress_snapkv', 'kvpress_h2o', 'kvpress_think', 'quant_int4']

    orchestrator = PipelineOrchestrator()

    for scenario_name, batch, seq, output in scenarios:
        print(f"\n\n{'─' * 80}")
        print(f"场景: {scenario_name} (batch={batch}, seq_len={seq}, output={output})")
        print(f"{'─' * 80}")
        print(f"{'方法':<25} | {'策略':<22} | {'预期加速':>10}")
        print("-" * 65)

        workload = WorkloadContext(
            batch_size=batch,
            seq_len=seq,
            expected_output_len=output,
        )

        for method in key_methods:
            try:
                config = orchestrator.create_pipeline(method, workload)
                speedup_str = f"{config.estimated_speedup:.2f}x"
                if config.estimated_speedup < 1.0:
                    speedup_str += " (不推荐)"
                print(f"{method:<25} | {config.strategy.name:<22} | {speedup_str:>10}")
            except Exception as e:
                print(f"{method:<25} | ERROR: {str(e)[:40]}")


def demo_strategy_reasoning():
    """演示策略选择的推理逻辑"""
    print_header("策略选择推理逻辑")

    print("""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                       策略选择决策树                                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        压缩方法                                          │
  └────────────────────────────┬────────────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │  GEMM密集型  │   │  Memory型    │   │  Mixed型     │
    │  (MLP压缩)   │   │ (流式/量化)  │   │ (SnapKV等)   │
    └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
           │                  │                  │
           ▼                  ▼                  ▼
    ┌──────────────┐   ┌──────────────┐   需要Attention?
    │ batch >= 4?  │   │ 延迟 < 0.5ms?│         │
    └──────┬───────┘   └──────┬───────┘    ┌────┴────┐
           │                  │            ▼         ▼
      ┌────┴────┐        ┌────┴────┐     是        否
      ▼         ▼        ▼         ▼     │          │
     是        否    INLINE    AFTER     ▼          ▼
     │          │   PREFILL   PREFILL  INLINE    AFTER
     ▼          ▼                      PREFILL   PREFILL
  ASYNC    SYNC AFTER
  DECODE   PREFILL

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                       方法特性对照表                                          ┃
┣━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ 方法                 ┃ 计算类型 ┃ Decode兼容性 ┃ 最优策略                     ┃
┣━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ MLP压缩              ┃ GEMM     ┃ 0.95 (高)    ┃ ASYNC_WITH_DECODE           ┃
┃ SnapKV               ┃ MIXED    ┃ 0.30 (低)    ┃ INLINE_PREFILL              ┃
┃ StreamingLLM         ┃ MEMORY   ┃ 0.10 (极低)  ┃ INLINE_PREFILL              ┃
┃ H2O                  ┃ MIXED    ┃ 0.40 (中)    ┃ INLINE_PREFILL              ┃
┃ ThinK                ┃ MIXED    ┃ 0.30 (低)    ┃ SYNC_AFTER_PREFILL          ┃
┃ INT8量化             ┃ MEMORY   ┃ 0.20 (低)    ┃ PROGRESSIVE                 ┃
┃ INT4量化             ┃ MEMORY   ┃ 0.20 (低)    ┃ PROGRESSIVE                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
""")


def demo_composed_pipeline():
    """演示组合Pipeline"""
    print_header("组合Pipeline演示")

    orchestrator = PipelineOrchestrator()
    workload = WorkloadContext(
        batch_size=4,
        seq_len=2048,
        expected_output_len=100,
    )

    # 组合方案
    compositions = [
        (['kvpress_snapkv', 'mlp'], 'SnapKV稀疏化 + MLP压缩'),
        (['kvpress_streaming_llm', 'quant_int4'], 'StreamingLLM + INT4量化'),
        (['kvpress_h2o', 'kvpress_think'], 'H2O + ThinK维度压缩'),
    ]

    for methods, description in compositions:
        print(f"\n【{description}】")
        print(f"方法组合: {' → '.join(methods)}")

        try:
            composed = orchestrator.create_composed_pipeline(methods, workload)
            print(f"总压缩比: {composed.total_compression_ratio:.1f}x")
            print(f"预期加速: {composed.estimated_speedup:.2f}x")
            print("执行顺序:")
            for i, config in enumerate(composed.methods, 1):
                print(f"  {i}. {config.method.name:<20} | {config.strategy.name:<22} | {config.timing.name}")
        except Exception as e:
            print(f"  Error: {e}")


def demo_auto_compress_manager():
    """演示AutoCompressManager的使用"""
    print_header("AutoCompressManager使用演示")

    # 创建管理器
    config = CompressionConfig(
        method='mlp',
        compression_ratio=5.0,
        min_speedup=1.1,
    )
    manager = AutoCompressManager(config)

    print("\n配置:")
    print(f"  方法: {config.method}")
    print(f"  压缩比: {config.compression_ratio}x")
    print(f"  最小加速要求: {config.min_speedup}x")

    # 测试不同场景
    scenarios = [
        (1, 256, 10),
        (1, 512, 50),
        (4, 2048, 100),
        (8, 4096, 200),
    ]

    print(f"\n{'batch':>5} | {'seq':>5} | {'output':>6} | {'决策':^8} | {'预期加速':>10}")
    print("-" * 55)

    for batch, seq, output in scenarios:
        should = manager.should_compress(batch, seq, output)
        plan = manager.get_plan(batch, seq, output)

        decision = "压缩" if should else "不压缩"
        speedup = f"{plan.estimated_speedup:.2f}x"

        print(f"{batch:>5} | {seq:>5} | {output:>6} | {decision:^8} | {speedup:>10}")


def demo_method_profiles():
    """演示方法Profile详情"""
    print_header("方法Profile详情")

    methods_to_show = ['mlp', 'kvpress_snapkv', 'kvpress_streaming_llm', 'kvpress_h2o', 'quant_int8']

    for name in methods_to_show:
        try:
            method = get_method(name)
            profile = method.profile

            print(f"\n【{name}】")
            print(f"  计算类型: {profile.compute_type.name}")
            print(f"  Decode兼容性: {profile.decode_compatibility:.2f}")
            print(f"  可分块: {profile.chunkable}")
            print(f"  需要Attention: {profile.requires_attention}")
            print(f"  改变序列长度: {profile.modifies_seq_len}")
            print(f"  压缩比: {profile.compression_ratio:.1f}x")

            # 估算延迟
            latency_512 = profile.estimate_latency(512, 1)
            latency_2048 = profile.estimate_latency(2048, 4)
            print(f"  估算延迟 (seq=512, batch=1): {latency_512:.2f}ms")
            print(f"  估算延迟 (seq=2048, batch=4): {latency_2048:.2f}ms")

        except Exception as e:
            print(f"\n【{name}】 Error: {e}")


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  AutoCompress - 自动化KV-Cache压缩流水线调度系统  ".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    print("""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 核心设计理念:                                                                  ┃
┃   • 用户负责选择压缩方法                                                        ┃
┃   • 系统自动找到最优的执行位置和流水线编排                                         ┃
┃   • 性能保证：压缩后不比不压缩慢                                                 ┃
┃                                                                               ┃
┃ 支持的压缩方法:                                                                ┃
┃   • MLP-based: BatchedGEMM压缩（GEMM密集型，与Decode互补）                       ┃
┃   • kvpress库: SnapKV, StreamingLLM, H2O, TOVA, ThinK, SimLayerKV等           ┃
┃   • 量化方法: INT8, INT4, FP8, NF4                                            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
""")

    # 运行所有演示
    demo_hardware()
    demo_available_methods()
    demo_pipeline_strategies()
    demo_strategy_reasoning()
    demo_workload_analysis()
    demo_composed_pipeline()
    demo_method_profiles()
    demo_auto_compress_manager()

    print_header("演示完成!", "█")
    print("\n使用示例:")
    print("""
    from nanovllm.autocompress import (
        PipelineOrchestrator,
        create_optimal_pipeline,
        analyze_all_methods,
    )

    # 方式1: 快速创建最优Pipeline
    config = create_optimal_pipeline('mlp', batch_size=4, seq_len=2048)
    print(f"Strategy: {config.strategy.name}")
    print(f"Expected speedup: {config.estimated_speedup:.2f}x")

    # 方式2: 分析所有方法
    analysis = analyze_all_methods(batch_size=4, seq_len=2048)
    for method, info in analysis.items():
        print(f"{method}: {info['strategy']} -> speedup={info['estimated_speedup']:.2f}x")

    # 方式3: 创建组合Pipeline
    orchestrator = PipelineOrchestrator()
    composed = orchestrator.create_composed_pipeline(['snapkv', 'mlp'], workload)
    print(f"Total compression: {composed.total_compression_ratio:.1f}x")
""")


if __name__ == '__main__':
    main()
