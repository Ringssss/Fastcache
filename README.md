
## 简介

FastCache 是一个高效的多模态大语言模型 (MLLM) 服务框架，专注于通过轻量级 KV-Cache 压缩优化推理性能。该框架有效解决并发服务场景下的处理开销和排队延迟问题。

基于论文 [FastCache: Optimizing Multimodal LLM Serving through Lightweight KV-Cache Compression Framework](http://arxiv.org/abs/2503.08461)。

## 核心特点

- **高效 KV-Cache 压缩**：支持多种压缩方法如 Knorm、StreamingLLM、SnapKV、TOVA 等
- **动态批处理策略**：优化 prefill、压缩和 decode 阶段的请求调度
- **内存池管理**：通过 KV-Cache 池机制消除内存碎片，保持高 GPU 利用率
- **并发请求处理**：高效管理同时到来的多个请求，显著提升系统吞吐量
- **全面性能监控**：详细的性能指标收集和分析工具

## 系统架构

```
├── utils_compress.py   # KV-Cache 压缩实现
├── utils_kvcachePool.py # KV-Cache 内存池管理
├── utils_requests.py   # 请求管理和性能指标
├── utils_profiler.py   # 性能分析工具
└── testcon.py          # 并发测试框架
```

## 性能优势

在 GQA 和 MileBench 数据集的评测中，FastCache 相比现有方案取得了:

- 首字时间 (TTFT) 减少高达 19.3 倍
- 吞吐量提升高达 12.1 倍
- 高并发场景 (高达 40 req/s) 下稳定性能
- 平均内存消耗减少 20%

## 快速开始

### 安装依赖

```bash
pip install torch transformers pillow tqdm
```

### 基本用法示例

```python
from utils_compress import press_select
from utils_kvcachePool import KVCachePool
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# 2. 创建 KV-Cache 池
kv_cache_pool = KVCachePool(device="cuda")

# 3. 选择压缩方法
compression_method = "Knorm"  # 可选: "StreamingLLM", "SnapKV", "TOVA" 等
compression_ratio = 0.8
compressor = press_select(compression_method, compression_ratio)

# 4. 推理示例
input_text = "请为这张图片生成描述："
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# 5. 执行推理并使用压缩
with compressor(model):
    outputs = model.generate(**inputs, max_new_tokens=128)
    
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 并发测试

```bash
python testcon.py --model_name "llava-v1.5-7b" \
                 --dataset "milebench" \
                 --num_samples 100 \
                 --max_workers 4 \
                 --compression_method "Knorm" \
                 --compression_ratio 0.8
```

## 引用

如果您使用了本项目，请引用我们的论文：

```bibtex
@article{zhu2025fastcache,
  title={FastCache: Optimizing Multimodal LLM Serving through Lightweight KV-Cache Compression Framework},
  author={Zhu, Jianian and Wu, Hang and Wang, Haojie and Li, Yinghui and Hou, Biao and Li, Ruixuan and Zhai, Jidong},
  journal={arXiv preprint arXiv:2503.08461},
  year={2025}
}
```

## 许可证

MIT
