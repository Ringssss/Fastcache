# FastCache

FastCache provides ready-to-run scripts and utilities to test KV-cache compression for MiniCPM-V-2.6 and LLaVA models under concurrent workloads. Two entry scripts are provided:

- scripts/minicpm_testcon2.6.py
- scripts/llava_testcon2.6.py

These share a common utils package in utils_ccm/.

## 1) Environment

- Python 3.10+
- NVIDIA GPU with CUDA (tested with A100)
- Recommended: create a fresh virtualenv

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2) Prepare Models

You need the base model weights locally (HF or internal mirror). Example paths below; adjust for your machine.

- MiniCPM-V-2.6 (vision-language): put at e.g. /data/huggingface/MiniCPM-V-2_6
- LLaVA 1.5 7B HF: put at e.g. /data/huggingface/llava-1.5-7b-hf

The scripts use `trust_remote_code=True` for these repos.

## 3) Compressor MLP Weights (CCM)

Two MLP checkpoints (MiniCPM + LLaVA) are stored in this repo via Git LFS so you can run out of the box:

- ckpt/minicpm_mlp.pth   (MiniCPM)
- ckpt/llava_mlp.pth     (LLaVA)

Make sure you have Git LFS:

```bash
git lfs install
# If the files are not present after clone, fetch them explicitly:
git lfs pull
```

If you want to host weights elsewhere (e.g., GitHub Releases/Hugging Face) just update `--ckpt_path` or configs/experiments.yaml.

## 4) Datasets

The examples expect either GQA-style or MileBench-style VQA data. Provide two paths:

- --datasets_js_path: annotation json or folder (e.g. /data/huggingface/LLaVA-Instruct-150K/llava_v1_5_mix665k.json)
- --datasets_img_path: image root folder (e.g. ./datasets)

You can swap to your own data; see the `load_image_data*` helpers in each script.

## 5) Quick Start

MiniCPM (single run, CCM):

```bash
python scripts/minicpm_testcon2.6.py \
  --model_path /data/huggingface/MiniCPM-V-2_6 \
  --ckpt_path ckpt/minicpm_mlp.pth \
  --datasets_js_path /data/huggingface/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
  --datasets_img_path ./datasets \
  --use_compression --comp_mode ccm \
  --prefill_batch_size 15 --compress_batch_size 45 --decoding_batch_size 90 \
  --req_per_sec 10.0 --num_samples 90 --torch_dtype float16
```

LLaVA (single run, CCM):

```bash
python scripts/llava_testcon2.6.py \
  --model_path /data/huggingface/llava-1.5-7b-hf \
  --ckpt_path ckpt/llava_mlp.pth \
  --datasets_js_path /data/huggingface/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
  --datasets_img_path ./datasets \
  --use_compression --comp_mode ccm \
  --prefill_batch_size 15 --compress_batch_size 45 --decoding_batch_size 90 \
  --req_per_sec 10.0 --num_samples 90 --torch_dtype float16
```

Batch (YAML) mode (MiniCPM example):

```bash
python scripts/minicpm_testcon2.6.py --config configs/experiments.yaml
```

## 6) What Gets Compressed

- For CCM (`--comp_mode ccm`), the scripts build either `KVCacheHybridCompressor` (MiniCPM) or `KVCacheLinearDecoupleCompressor` (LLaVA) from utils_ccm/module_ccm_v11.py and load your MLP `.pth`.
- Other modes like Knorm / SnapKV / ExpectedAttention are available via kvpress; ensure the package is installed.

## 7) GPU/Cache Notes

- utils_ccm/utils_kvcachePool.py provides a simple KV cache pool for merging batched requests.
- utils_ccm/utils_schedule_v11.py implements a basic dynamic scheduler and GPU monitoring (pynvml required).
- NVTX ranges are used for profiling (nvtx). If you don’t have it, remove those imports.

## 8) Paths To Change

- configs/experiments.yaml: set your real `model_path`, `ckpt_path`, dataset paths.
- Or pass them via CLI as shown above.

## 9) Troubleshooting

- Transformers versions: OffloadedCache/LLaVA classes require recent transformers (>=4.43). If missing, upgrade.
- CUDA OOM: reduce batch sizes and/or use `--torch_dtype float16`.
- kvpress import errors: `pip install kvpress`.

## 10) Repository Layout

- scripts/: entry points
- utils_ccm/: shared modules (compressors, scheduler, kv-pool, helpers)
- configs/: example YAML for batch experiments
- ckpt/: place your two MLP weights here locally (or download from Release)
