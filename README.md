# Fastcache

Fastcache is a lightweight KV-cache compression framework and method-aware runtime for multimodal LLM serving. This repository includes the KV-cache pipeline implementation (method-aware pipeline binding, GreenContext SM partitioning, and dual-batch scheduling) extracted from the nano-vllm integration.

## KV-cache Pipeline

The core runtime design lives in `kv-cache pipeline/`:
- `kv-cache pipeline/nanovllm/autocompress/`: method profiles, cost model, and pipeline orchestrator.
- `kv-cache pipeline/nanovllm/engine/`: controller + runtime integration (pipeline binding, overlap depth control, greenctx toggles, dual-batch decisions).
- `kv-cache pipeline/nanovllm/utils/streams.py`: GreenContext-aware stream construction.
- `kv-cache pipeline/bench/`: benchmark utilities.

## Reference

If you use this repository, please consider citing:

Zhu, J., Wu, H., Wang, H., Li, Y., Hou, B., Li, R., & Zhai, J. (2025). Fastcache: Optimizing multimodal llm serving through lightweight kv-cache compression framework. arXiv preprint arXiv:2503.08461.
