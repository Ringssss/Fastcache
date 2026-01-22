# KV-cache Pipeline

This folder contains the method-aware KV-cache compression pipeline and runtime control loop used for high-throughput LLM serving.

## Key Components

- Method profiles and cost model:
  - `nanovllm/autocompress/profiles.py`
  - `nanovllm/autocompress/cost_model.py`
  - `nanovllm/autocompress/context.py`
- Pipeline orchestrator (prefill-bound vs decode-bound strategies):
  - `nanovllm/autocompress/pipeline.py`
- Online controller (pipeline binding, overlap depth, greenctx/stream knobs, dual-batch):
  - `nanovllm/engine/auto_compress_controller.py`
- Runtime integration points:
  - `nanovllm/engine/llava_engine.py`
  - `nanovllm/engine/llava_model_runner.py`
  - `nanovllm/engine/llava_scheduler.py`
- GreenContext stream construction:
  - `nanovllm/utils/streams.py`
- Benchmarks:
  - `bench/bench_kvcache_matrix.py`

## Notes

- GreenContext support requires an `sgl_kernel` build that exposes `create_greenctx_stream_by_value`. If unavailable, the system falls back to regular CUDA streams.
- This code is extracted from the nano-vllm integration. It is intended as a reference implementation of the pipeline control logic and runtime wiring.
