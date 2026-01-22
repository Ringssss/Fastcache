#!/usr/bin/env python3
"""
Comprehensive KV-cache compression matrix for nano-vllm.

Runs baseline + compression variants across:
  - Workload types (light vs heavy)
  - Models (Qwen3-8B, Qwen3-32B, LLaVA MLP)
  - Config toggles (greenctx, auto_tune, auto_dual)

Results are stored in a single JSON file with full metadata.
"""

import argparse
import csv
import gc
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.llava_engine import LlavaLLM


def clear_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def iter_dataset_files(dataset_dir: str) -> List[str]:
    paths: List[str] = []
    for name in os.listdir(dataset_dir):
        if name.endswith((".jsonl", ".json", ".csv")):
            paths.append(os.path.join(dataset_dir, name))
    return sorted(paths)


def extract_text(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("prompt", "text", "question", "query", "instruction", "input"):
            if key in obj and isinstance(obj[key], str):
                return obj[key]
        if "conversations" in obj and isinstance(obj["conversations"], list):
            for item in obj["conversations"]:
                if not isinstance(item, dict):
                    continue
                if item.get("from") in ("human", "user"):
                    return item.get("value")
                if item.get("role") == "user":
                    return item.get("content")
        if "messages" in obj and isinstance(obj["messages"], list):
            for item in obj["messages"]:
                if not isinstance(item, dict):
                    continue
                if item.get("role") == "user":
                    return item.get("content")
    return None


def iter_prompts_from_path(path: str) -> Iterable[str]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = extract_text(obj)
                if text:
                    yield text
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for item in obj:
                text = extract_text(item)
                if text:
                    yield text
        elif isinstance(obj, dict):
            for key in ("data", "examples", "instances"):
                if key in obj and isinstance(obj[key], list):
                    for item in obj[key]:
                        text = extract_text(item)
                        if text:
                            yield text
    elif path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = extract_text(row)
                if text:
                    yield text


def build_prompt_pool(
    tokenizer,
    dataset_dir: str,
    num_prompts: int,
    min_tokens: int,
    max_tokens: int,
) -> List[str]:
    prompts: List[str] = []
    for path in iter_dataset_files(dataset_dir):
        for text in iter_prompts_from_path(path):
            try:
                token_len = len(tokenizer.encode(text))
            except Exception:
                continue
            if token_len < min_tokens or token_len > max_tokens:
                continue
            prompts.append(text)
            if len(prompts) >= num_prompts:
                return prompts
    if prompts:
        while len(prompts) < num_prompts:
            prompts.append(prompts[len(prompts) % len(prompts)])
    return prompts


def get_fill_token_id(tokenizer) -> int:
    candidates = [" Hello", " hello", " the", " test", " a"]
    for text in candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            return ids[0]
    if tokenizer.unk_token_id is not None:
        return tokenizer.unk_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return 0


def load_trace_entries(
    path: str,
    *,
    min_context: int,
    max_context: int,
    min_output: int,
    max_output: int,
    topk: int,
) -> List[Tuple[int, int]]:
    entries: List[Tuple[int, int]] = []
    heap: List[Tuple[int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ctx = int(row.get("ContextTokens", 0))
                out = int(row.get("GeneratedTokens", 0))
            except ValueError:
                continue
            if ctx < min_context or out < min_output:
                continue
            if max_context > 0 and ctx > max_context:
                continue
            if max_output > 0 and out > max_output:
                continue
            if topk > 0:
                import heapq
                if len(heap) < topk:
                    heapq.heappush(heap, (ctx, out))
                else:
                    if ctx > heap[0][0]:
                        heapq.heapreplace(heap, (ctx, out))
            else:
                entries.append((ctx, out))
    if topk > 0:
        entries = sorted(heap, key=lambda x: x[0], reverse=True)
    return entries


def build_trace_prompts(
    tokenizer,
    trace_entries: List[Tuple[int, int]],
    num_prompts: int,
    max_model_len: int,
) -> Tuple[List[List[int]], List[int]]:
    prompts: List[List[int]] = []
    outputs: List[int] = []
    fill_id = get_fill_token_id(tokenizer)
    idx = 0
    while len(prompts) < num_prompts and trace_entries:
        ctx_len, out_len = trace_entries[idx % len(trace_entries)]
        max_ctx = max(1, max_model_len - max(out_len, 1))
        ctx_len = min(ctx_len, max_ctx)
        if ctx_len <= 0:
            idx += 1
            continue
        prompts.append([fill_id] * ctx_len)
        outputs.append(out_len)
        idx += 1
    return prompts, outputs


def build_synthetic_prompts(
    tokenizer,
    num_prompts: int,
    max_model_len: int,
    context_len: int,
    output_len: int,
    dialogue: bool = True,
    token_text: Optional[str] = None,
) -> Tuple[List[List[int]], List[int]]:
    if output_len <= 0:
        output_len = max(1, min(512, max_model_len // 4))
    max_ctx = max(1, max_model_len - output_len)
    context_len = max(1, min(context_len or max_ctx, max_ctx))

    pattern_ids: List[int] = []
    if token_text:
        pattern_ids = tokenizer.encode(token_text, add_special_tokens=False)
    if dialogue and not pattern_ids:
        pattern = (
            "User: Summarize the previous steps and propose next actions.\n"
            "Assistant: Thought: I should plan and reason carefully.\n"
            "Action: Analyze memory/computation tradeoffs.\n"
            "Observation: Compression reduces KV reads but adds compute.\n"
            "Assistant: Next, propose a scheduling plan.\n"
        )
        pattern_ids = tokenizer.encode(pattern, add_special_tokens=False)

    if not pattern_ids:
        fill_id = get_fill_token_id(tokenizer)
        pattern_ids = [fill_id]

    prompts: List[List[int]] = []
    outputs: List[int] = []
    for i in range(num_prompts):
        tokens: List[int] = []
        while len(tokens) < context_len:
            tokens.extend(pattern_ids)
        prompts.append(tokens[:context_len])
        outputs.append(output_len)
    return prompts, outputs

def run_generation(
    model_path: str,
    prompts: List[Any],
    max_tokens: int | List[int],
    engine_kwargs: Dict[str, Any],
    apply_compression: bool,
) -> Dict[str, Any]:
    clear_gpu()
    llm = LlavaLLM(model_path, **engine_kwargs)
    if isinstance(max_tokens, list):
        max_tokens_list = max_tokens
    else:
        max_tokens_list = [max_tokens] * len(prompts)
    for prompt, out_len in zip(prompts, max_tokens_list):
        llm.add_request(prompt, SamplingParams(max_tokens=out_len, ignore_eos=True))

    total_tokens = 0
    start = time.time()
    while not llm.is_finished():
        _, num_tokens = llm.step(apply_compression=apply_compression)
        if num_tokens > 0:
            total_tokens += num_tokens
        else:
            total_tokens += (-num_tokens)
    elapsed = time.time() - start
    throughput = total_tokens / max(elapsed, 1e-6)

    result = {
        "total_tokens": total_tokens,
        "elapsed_s": elapsed,
        "throughput_tok_s": throughput,
    }
    if getattr(llm, "_auto_controller", None) is not None:
        last_decision = getattr(llm._auto_controller, "_last_decision", None)
        if last_decision is not None:
            result["auto_last_decision"] = asdict(last_decision)
    green_info = getattr(getattr(llm, "model_runner", None), "_greenctx_info", None)
    if green_info is not None:
        result["greenctx_available"] = green_info.available
        result["greenctx_reason"] = green_info.reason
    del llm
    clear_gpu()
    return result


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    mode: str  # "trace", "dataset", or "synthetic"
    num_batches: int
    max_output_tokens: int
    min_prompt_tokens: int = 0
    max_prompt_tokens: int = 0
    trace_path: str = ""
    trace_min_context: int = 0
    trace_max_context: int = 0
    trace_min_output: int = 0
    trace_max_output: int = 0
    synthetic_context: int = 0
    synthetic_output: int = 0
    synthetic_dialogue: bool = True
    synthetic_token_text: str = ""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str
    backend: str  # "kvpress" or "mlp"
    method: str
    compression_factor: int
    max_model_len: int
    enforce_eager: bool
    compressor_path: str = ""
    batch_sizes_heavy: Tuple[int, ...] = ()
    batch_sizes_light: Tuple[int, ...] = ()


@dataclass(frozen=True)
class ConfigVariant:
    label: str
    auto_tune: bool
    greenctx: bool
    auto_dual: bool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="kvcache_matrix_results.json")
    parser.add_argument("--models", default="qwen3-8b,qwen3-32b,llava-mlp")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset-dir", default="/home/zhujianian/fastdllm/datasets")
    parser.add_argument("--trace-conv", default="/data/huggingface/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv")
    parser.add_argument("--qwen3-8b", default="/data/huggingface/Qwen3-8B")
    parser.add_argument("--qwen3-32b", default="/data/huggingface/Qwen3-32B")
    parser.add_argument("--llava", default="/data/huggingface/llava-1.5-7b-hf")
    parser.add_argument("--compressor-path", default="/home/zhujianian/cvpr/ckpt_store/best_finetune_mlp_1030_mm_9.pth")
    parser.add_argument("--synthetic-context", type=int, default=3072)
    parser.add_argument("--synthetic-output", type=int, default=512)
    parser.add_argument("--synthetic-dialogue", action="store_true")
    parser.add_argument("--synthetic-token", default="")
    parser.add_argument("--synthetic-num-batches", type=int, default=2)
    parser.add_argument("--synthetic-batch-sizes", default="")
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--workloads", default="")
    parser.add_argument("--variants", default="")
    parser.add_argument("--greenctx-compress-ratio", type=float, default=None)
    parser.add_argument("--greenctx-main-ratio", type=float, default=None)
    parser.add_argument("--greenctx-main-stream", action="store_true")
    args = parser.parse_args()

    model_specs = {
        "qwen3-8b": ModelSpec(
            name="qwen3-8b",
            path=args.qwen3_8b,
            backend="kvpress",
            method="h2o",
            compression_factor=5,
            max_model_len=4096,
            enforce_eager=False,
            batch_sizes_heavy=(16, 64),
            batch_sizes_light=(16, 64),
        ),
        "qwen3-32b": ModelSpec(
            name="qwen3-32b",
            path=args.qwen3_32b,
            backend="kvpress",
            method="h2o",
            compression_factor=5,
            max_model_len=4096,
            enforce_eager=False,
            batch_sizes_heavy=(8, 16),
            batch_sizes_light=(8, 16),
        ),
        "llava-mlp": ModelSpec(
            name="llava-mlp",
            path=args.llava,
            backend="mlp",
            method="mlp",
            compression_factor=5,
            max_model_len=4096,
            enforce_eager=True,
            compressor_path=args.compressor_path,
            batch_sizes_heavy=(1, 4),
            batch_sizes_light=(1, 4),
        ),
    }

    workloads = [
        WorkloadSpec(
            name="heavy-trace",
            mode="trace",
            num_batches=4,
            max_output_tokens=512,
            trace_path=args.trace_conv,
            trace_min_context=1024,
            trace_min_output=256,
            trace_max_output=512,
        ),
        WorkloadSpec(
            name="light-dataset",
            mode="dataset",
            num_batches=1,
            max_output_tokens=64,
            min_prompt_tokens=32,
            max_prompt_tokens=128,
        ),
        WorkloadSpec(
            name="synthetic-long",
            mode="synthetic",
            num_batches=max(1, args.synthetic_num_batches),
            max_output_tokens=args.synthetic_output,
            synthetic_context=args.synthetic_context,
            synthetic_output=args.synthetic_output,
            synthetic_dialogue=args.synthetic_dialogue,
            synthetic_token_text=args.synthetic_token,
        ),
    ]

    workload_filter = {name.strip() for name in args.workloads.split(",") if name.strip()}
    if workload_filter:
        workloads = [w for w in workloads if w.name in workload_filter]

    synthetic_batch_sizes: Optional[Tuple[int, ...]] = None
    if args.synthetic_batch_sizes:
        parsed = [int(x) for x in args.synthetic_batch_sizes.split(",") if x.strip().isdigit()]
        if parsed:
            synthetic_batch_sizes = tuple(parsed)

    variants = [
        ConfigVariant("compression-only", auto_tune=False, greenctx=False, auto_dual=False),
        ConfigVariant("greenctx-only", auto_tune=False, greenctx=True, auto_dual=False),
        ConfigVariant("auto-tune+greenctx", auto_tune=True, greenctx=True, auto_dual=False),
        ConfigVariant("triad", auto_tune=True, greenctx=True, auto_dual=True),
    ]
    variant_filter = {name.strip() for name in args.variants.split(",") if name.strip()}
    if variant_filter:
        variants = [v for v in variants if v.label in variant_filter]

    greenctx_compress_ratio = args.greenctx_compress_ratio
    greenctx_main_ratio = args.greenctx_main_ratio
    if greenctx_compress_ratio is None:
        greenctx_compress_ratio = 0.25
    if greenctx_main_ratio is None:
        greenctx_main_ratio = max(0.05, 1.0 - greenctx_compress_ratio)

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    results: List[Dict[str, Any]] = []
    baseline_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    done_keys: set[Tuple[str, str, int, str]] = set()

    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
            for row in results:
                key = (row.get("model"), row.get("workload"), row.get("batch_size"), row.get("variant"))
                done_keys.add(key)
                if row.get("variant") == "baseline" and row.get("backend") == "none":
                    cache_key = (row.get("model"), row.get("workload"), row.get("batch_size"))
                    if "throughput_tok_s" in row:
                        baseline_cache[cache_key] = row
            print(f"Resume enabled: loaded {len(results)} rows from {args.output}")
        except Exception as exc:
            print(f"Resume failed: {exc}")
            results = []
            done_keys = set()
    prompt_cache: Dict[Tuple[str, str], Tuple[List[Any], Optional[List[int]]]] = {}

    for model_key in selected_models:
        spec = model_specs.get(model_key)
        if spec is None:
            print(f"Skip unknown model key: {model_key}")
            continue
        if not os.path.exists(spec.path):
            print(f"Skip missing model: {spec.path}")
            continue

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(spec.path, use_fast=True, trust_remote_code=True)

        for workload in workloads:
            workload_id = workload.name
            if workload.mode == "trace" and not os.path.exists(workload.trace_path):
                print(f"Trace missing: {workload.trace_path}")
                continue

            if workload.name == "heavy-trace":
                batch_sizes = spec.batch_sizes_heavy
            elif workload.name == "synthetic-long" and synthetic_batch_sizes:
                batch_sizes = synthetic_batch_sizes
            else:
                batch_sizes = spec.batch_sizes_light
            max_batch = max(batch_sizes) if batch_sizes else 1
            num_prompts = max_batch * max(1, workload.num_batches)
            cache_key = (spec.name, workload_id)

            if cache_key not in prompt_cache:
                if workload.mode == "trace":
                    trace_entries = load_trace_entries(
                        workload.trace_path,
                        min_context=workload.trace_min_context,
                        max_context=workload.trace_max_context,
                        min_output=workload.trace_min_output,
                        max_output=workload.trace_max_output,
                        topk=num_prompts,
                    )
                    prompts, outputs = build_trace_prompts(
                        tokenizer,
                        trace_entries,
                        num_prompts=num_prompts,
                        max_model_len=spec.max_model_len,
                    )
                    prompt_cache[cache_key] = (prompts, outputs)
                elif workload.mode == "synthetic":
                    prompts, outputs = build_synthetic_prompts(
                        tokenizer,
                        num_prompts=num_prompts,
                        max_model_len=spec.max_model_len,
                        context_len=workload.synthetic_context,
                        output_len=workload.synthetic_output or workload.max_output_tokens,
                        dialogue=workload.synthetic_dialogue,
                        token_text=workload.synthetic_token_text or None,
                    )
                    prompt_cache[cache_key] = (prompts, outputs)
                else:
                    prompts = build_prompt_pool(
                        tokenizer,
                        args.dataset_dir,
                        num_prompts=num_prompts,
                        min_tokens=workload.min_prompt_tokens,
                        max_tokens=workload.max_prompt_tokens,
                    )
                    prompt_cache[cache_key] = (prompts, None)

            prompt_pool, output_pool = prompt_cache[cache_key]
            if not prompt_pool:
                print(f"No prompts for {spec.name} workload={workload_id}")
                continue

            prompt_tokens = None
            if prompt_pool and isinstance(prompt_pool[0], list):
                prompt_tokens = len(prompt_pool[0])

            for batch_size in batch_sizes:
                prompts = prompt_pool[: batch_size * max(1, workload.num_batches)]
                outputs = output_pool[: len(prompts)] if output_pool else None
                baseline_key = (spec.name, workload_id, batch_size)
                baseline_done_key = (spec.name, workload_id, batch_size, "baseline")
                if baseline_done_key not in done_keys and baseline_key not in baseline_cache:
                    try:
                        base = run_generation(
                            spec.path,
                            prompts,
                            outputs if outputs else workload.max_output_tokens,
                            {
                                "enable_compression": False,
                                "enforce_eager": spec.enforce_eager,
                                "max_model_len": spec.max_model_len,
                                "gpu_memory_utilization": args.gpu_mem_util,
                            },
                            apply_compression=False,
                        )
                        baseline_cache[baseline_key] = base
                        results.append({
                            "model": spec.name,
                            "workload": workload_id,
                            "batch_size": batch_size,
                            "num_batches": workload.num_batches,
                            "backend": "none",
                            "method": "none",
                            "variant": "baseline",
                            "prompt_tokens": prompt_tokens,
                            **base,
                        })
                        done_keys.add(baseline_done_key)
                        print(f"[{spec.name}] {workload_id} bs={batch_size} baseline {base['throughput_tok_s']:.1f} tok/s")
                    except Exception as exc:
                        baseline_cache[baseline_key] = {"error": str(exc)}
                        results.append({
                            "model": spec.name,
                            "workload": workload_id,
                            "batch_size": batch_size,
                            "num_batches": workload.num_batches,
                            "backend": "none",
                            "method": "none",
                            "variant": "baseline",
                            "prompt_tokens": prompt_tokens,
                            "error": str(exc),
                        })
                        done_keys.add(baseline_done_key)
                        print(f"[{spec.name}] {workload_id} bs={batch_size} baseline error: {exc}")

                for variant in variants:
                    variant_done_key = (spec.name, workload_id, batch_size, variant.label)
                    if variant_done_key in done_keys:
                        continue
                    try:
                        out = run_generation(
                            spec.path,
                            prompts,
                            outputs if outputs else workload.max_output_tokens,
                            {
                                "enable_compression": True,
                                "compression_backend": spec.backend,
                                "kvpress_method": spec.method,
                                "compressor_path": spec.compressor_path or None,
                                "compression_factor": spec.compression_factor,
                                "async_compression": True,
                                "compression_pipeline": "auto",
                                "compression_streams": 2,
                                "greenctx_enabled": variant.greenctx,
                                "greenctx_compress_ratio": greenctx_compress_ratio,
                                "greenctx_main_ratio": greenctx_main_ratio,
                                "greenctx_main_stream": args.greenctx_main_stream,
                                "decode_layers_per_step": 4,
                                "auto_tune": variant.auto_tune,
                                "auto_tune_interval": 4,
                                "auto_dual_batch": variant.auto_dual,
                                "auto_dual_interval": 4,
                                "enforce_eager": spec.enforce_eager,
                                "max_model_len": spec.max_model_len,
                                "gpu_memory_utilization": args.gpu_mem_util,
                            },
                            apply_compression=True,
                        )
                        base = baseline_cache.get(baseline_key, {})
                        speedup = None
                        if "throughput_tok_s" in base:
                            speedup = out["throughput_tok_s"] / base["throughput_tok_s"]
                        results.append({
                            "model": spec.name,
                            "workload": workload_id,
                            "batch_size": batch_size,
                            "num_batches": workload.num_batches,
                            "backend": spec.backend,
                            "method": spec.method,
                            "variant": variant.label,
                            "auto_tune": variant.auto_tune,
                            "greenctx": variant.greenctx,
                            "auto_dual": variant.auto_dual,
                            "greenctx_compress_ratio": greenctx_compress_ratio,
                            "greenctx_main_ratio": greenctx_main_ratio,
                            "greenctx_main_stream": args.greenctx_main_stream,
                            "speedup_vs_baseline": speedup,
                            "prompt_tokens": prompt_tokens,
                            **out,
                        })
                        done_keys.add(variant_done_key)
                        print(f"[{spec.name}] {workload_id} bs={batch_size} {variant.label} {out['throughput_tok_s']:.1f} tok/s")
                    except Exception as exc:
                        results.append({
                            "model": spec.name,
                            "workload": workload_id,
                            "batch_size": batch_size,
                            "num_batches": workload.num_batches,
                            "backend": spec.backend,
                            "method": spec.method,
                            "variant": variant.label,
                            "auto_tune": variant.auto_tune,
                            "greenctx": variant.greenctx,
                            "auto_dual": variant.auto_dual,
                            "greenctx_compress_ratio": greenctx_compress_ratio,
                            "greenctx_main_ratio": greenctx_main_ratio,
                            "greenctx_main_stream": args.greenctx_main_stream,
                            "prompt_tokens": prompt_tokens,
                            "error": str(exc),
                        })
                        done_keys.add(variant_done_key)
                        print(f"[{spec.name}] {workload_id} bs={batch_size} {variant.label} error: {exc}")

                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Checkpoint saved to {args.output}")

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
