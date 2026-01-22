# KV-cache Pipeline 端到端使用文档（中文）

本指南覆盖从环境搭建到 SM 切分与端到端测试的完整流程，适用于 `Fastcache/kv-cache pipeline` 中的 method-aware KV-cache 压缩流水线。

## 1. 目录位置

进入仓库后，核心代码在：

```
Fastcache/
└── kv-cache pipeline/
    ├── nanovllm/                 # 运行时与调度器
    ├── bench/bench_kvcache_matrix.py
    └── README.md
```

建议先进入目录：

```
cd "Fastcache/kv-cache pipeline"
```

## 2. 环境搭建

### 2.1 基础环境

- OS: Linux (推荐 Ubuntu 20.04/22.04)
- GPU: NVIDIA A100/H100/4090 等支持 CUDA 的 GPU
- CUDA: 12.x（与 PyTorch 对齐）
- Python: 3.10/3.11

### 2.2 创建虚拟环境

```
conda create -n fastcache python=3.10 -y
conda activate fastcache
```

### 2.3 安装依赖（最小集合）

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers numpy tqdm
```

如果你要跑 kvpress / MLP 压缩器，请根据你的环境补充对应依赖（例如 kvpress、flashinfer、opencv 等）。

### 2.4 sgl_kernel 与 Green Context

GreenContext 需要 `sgl_kernel` 暴露 `create_greenctx_stream_by_value`：

```
pip install sgl-kernel
```

如果出现版本不匹配导致 import 失败或 greenctx unavailable：

1) 优先选择与你的 PyTorch/CUDA 匹配的 sgl-kernel 版本  
2) 若 ABI 仍不匹配，可用 shim 方式临时修复（见 2.5）

### 2.5（可选）sgl_kernel shim 方式

当出现 ABI mismatch 时，可编译一个 shim 动态库并 `LD_PRELOAD`：

```
g++ -shared -fPIC -O2 sgl_kernel_shim.cc -o sgl_kernel_shim.so \
  -I$(python - <<'PY'
import torch, sys
from torch.utils.cpp_extension import include_paths
print(" ".join("-I"+p for p in include_paths()))
PY
) -L$(python - <<'PY'
import torch, sys
from torch.utils.cpp_extension import library_paths
print(" ".join("-L"+p for p in library_paths()))
PY
) -lc10 -ltorch -ltorch_cpu -Wl,-rpath,$(python - <<'PY'
import torch, sys
from torch.utils.cpp_extension import library_paths
print(":".join(library_paths()))
PY
)
```

然后运行时：

```
LD_PRELOAD=/path/to/sgl_kernel_shim.so python ...
```

## 3. SM 切分（GreenContext）

GreenContext 由以下参数控制：

- `greenctx_enabled`: 是否启用
- `greenctx_compress_ratio`: 分配给压缩流的 SM 比例
- `greenctx_main_ratio`: 分配给主流的 SM 比例
- `greenctx_main_stream`: 是否让主流也绑定 greenctx

### 3.1 可用性自检

```
python - <<'PY'
import sgl_kernel
print(hasattr(sgl_kernel, "create_greenctx_stream_by_value"))
PY
```

如果输出 False，则会 fallback 到普通 CUDA stream。

## 4. 端到端测试脚本

脚本入口：

```
python bench/bench_kvcache_matrix.py --help
```

### 4.1 基础测试（baseline / compression-only / triad）

示例（Qwen3-8B，synthetic Hello）：

```
CUDA_VISIBLE_DEVICES=0 \
python bench/bench_kvcache_matrix.py \
  --models qwen3-8b \
  --workloads synthetic-long \
  --variants compression-only,triad \
  --synthetic-context 4096 \
  --synthetic-output 128 \
  --synthetic-token "Hello" \
  --synthetic-num-batches 1 \
  --synthetic-batch-sizes "32,64" \
  --qwen3-8b /data/huggingface/Qwen3-8B
```

### 4.2 SM 比例 sweep（greenctx-only）

```
CUDA_VISIBLE_DEVICES=0 \
LD_PRELOAD=/path/to/sgl_kernel_shim.so \
python bench/bench_kvcache_matrix.py \
  --models qwen3-8b \
  --workloads synthetic-long \
  --variants greenctx-only \
  --synthetic-context 4096 \
  --synthetic-output 128 \
  --synthetic-token "Hello" \
  --synthetic-num-batches 1 \
  --synthetic-batch-sizes "64" \
  --greenctx-compress-ratio 0.2 \
  --greenctx-main-ratio 0.8
```

### 4.3 Triad SM ratio sweep（auto + greenctx）

```
CUDA_VISIBLE_DEVICES=0 \
LD_PRELOAD=/path/to/sgl_kernel_shim.so \
python bench/bench_kvcache_matrix.py \
  --models qwen3-8b \
  --workloads synthetic-long \
  --variants triad \
  --synthetic-context 4096 \
  --synthetic-output 128 \
  --synthetic-token "Hello" \
  --synthetic-num-batches 1 \
  --synthetic-batch-sizes "32" \
  --greenctx-compress-ratio 0.2 \
  --greenctx-main-ratio 0.8
```

### 4.4 LLaVA MLP 压缩测试

```
CUDA_VISIBLE_DEVICES=0 \
python bench/bench_kvcache_matrix.py \
  --models llava-mlp \
  --workloads synthetic-long \
  --variants compression-only,triad \
  --synthetic-context 4096 \
  --synthetic-output 128 \
  --synthetic-token "Hello" \
  --synthetic-num-batches 1 \
  --synthetic-batch-sizes "2,4" \
  --llava /data/huggingface/llava-1.5-7b-hf \
  --compressor-path /path/to/mlp_compressor.pth \
  --gpu-mem-util 0.7
```

## 5. 输出与结果解析

所有测试会生成 JSON 文件，核心字段：

- `throughput_tok_s`: 吞吐（tokens/s）
- `prompt_tokens`: 实际 prompt 长度
- `variant`: baseline / compression-only / greenctx-only / triad
- `greenctx_available`: 是否真正启用 greenctx

简单解析脚本：

```
python - <<'PY'
import json
with open("hello_ctx4096_qwen8_bs32_64_128_gc.json") as f:
    data = json.load(f)
for row in data:
    print(row["variant"], row["batch_size"], row["throughput_tok_s"])
PY
```

## 6. 常见问题

1) **greenctx unavailable**  
   - 说明 sgl_kernel 不匹配或缺少 `create_greenctx_stream_by_value`  
   - 解决：换兼容版本或使用 shim

2) **CUDA OOM**  
   - 减小 batch size / 降低 `--gpu-mem-util`  
   - 可加环境变量：  
     `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

3) **ctx=4096 实际变成 3968**  
   - `max_model_len=4096` 且 `output=128`，所以 prompt 会被裁剪到 3968

## 7. 推荐实验矩阵（示例）

- Qwen3-8B：ctx=1024/2048/4096，bs=32/64/128
- Qwen3-32B：ctx=1024/2048/4096，bs=16/32/64
- LLaVA-MLP：ctx=1024/2048/4096，bs=2/4（视显存调整）

## 8. 参考文献

Zhu, J., Wu, H., Wang, H., Li, Y., Hou, B., Li, R., & Zhai, J. (2025). Fastcache: Optimizing multimodal llm serving through lightweight kv-cache compression framework. arXiv preprint arXiv:2503.08461.
