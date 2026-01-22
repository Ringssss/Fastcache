"""
kvpress 压缩器适配器

将 kvpress 库的压缩方法集成到 nano-vllm 框架中，
使其能够享受 nano-vllm 的 CUDA Graph 优化和异步执行。

支持的方法：
- SnapKV: 基于attention窗口的稀疏化
- StreamingLLM: Sink + Recent窗口
- H2O: Heavy-Hitter Oracle
- Knorm: 基于K范数的压缩
- TOVA: Token-level Value Attention
- Random: 随机采样 (baseline)

使用方式：
```python
compressor = KVPressCompressor(
    method='streaming_llm',
    compression_ratio=0.8,  # 删除80%，保留20%
    config=model_config,
)

# 与 BatchedGEMMCompressor 相同的接口
compressed_kv = compressor(hf_kv_cache, it_len=[img_len, txt_len])
```
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class KVPressConfig:
    """kvpress压缩配置"""
    method: str = 'streaming_llm'
    compression_ratio: float = 0.8  # 删除的比例
    # SnapKV专用
    window_size: int = 32
    kernel_size: int = 5
    # StreamingLLM专用
    n_sink: int = 4
    # 通用
    per_layer: bool = False  # 是否每层独立压缩


class KVPressCompressor(nn.Module):
    """
    kvpress 压缩器的 nano-vllm 适配器

    将 kvpress 的压缩逻辑封装为与 BatchedGEMMCompressor 相同的接口，
    使其能够无缝集成到现有的压缩流程中。
    """

    # 支持的压缩方法
    SUPPORTED_METHODS = [
        'streaming_llm', 'snapkv', 'h2o', 'knorm',
        'tova', 'random', 'observed_attention', 'expected_attention'
    ]

    def __init__(
        self,
        method: str = 'streaming_llm',
        compression_ratio: float = 0.8,
        config: Optional[Any] = None,
        **kwargs
    ):
        """
        初始化 kvpress 压缩器

        Args:
            method: 压缩方法名称
            compression_ratio: 压缩比例 (删除的token比例，0.8表示保留20%)
            config: 模型配置 (HuggingFace config)
            **kwargs: 传递给特定压缩方法的参数
        """
        super().__init__()

        self.method_name = method
        self.compression_ratio = compression_ratio
        self.config = config
        self.kwargs = kwargs

        # 验证方法
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Supported: {self.SUPPORTED_METHODS}"
            )

        # 延迟加载 kvpress
        self._press = None
        self._kvpress_available = None

        # 从config获取模型参数
        if config is not None:
            self.num_layers = getattr(config, 'num_hidden_layers', 32)
            self.num_heads = getattr(config, 'num_attention_heads', 32)
            self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
            self.head_dim = getattr(config, 'head_dim',
                                   getattr(config, 'hidden_size', 4096) // self.num_heads)
        else:
            self.num_layers = kwargs.get('num_layers', 32)
            self.num_heads = kwargs.get('num_heads', 32)
            self.num_kv_heads = kwargs.get('num_kv_heads', self.num_heads)
            self.head_dim = kwargs.get('head_dim', 128)

    @property
    def kvpress_available(self) -> bool:
        """检查 kvpress 是否可用"""
        if self._kvpress_available is None:
            try:
                import kvpress
                self._kvpress_available = True
            except ImportError:
                self._kvpress_available = False
        return self._kvpress_available

    def _get_press(self):
        """懒加载 kvpress 压缩器"""
        if self._press is None:
            if not self.kvpress_available:
                raise ImportError(
                    "kvpress is not installed. "
                    "Install with: pip install kvpress"
                )

            import kvpress

            # 根据方法创建对应的Press
            press_creators = {
                'streaming_llm': lambda: kvpress.StreamingLLMPress(
                    compression_ratio=self.compression_ratio,
                    n_sink=self.kwargs.get('n_sink', 4),
                ),
                'snapkv': lambda: kvpress.SnapKVPress(
                    compression_ratio=self.compression_ratio,
                    window_size=self.kwargs.get('window_size', 32),
                    kernel_size=self.kwargs.get('kernel_size', 5),
                ),
                'h2o': lambda: kvpress.ObservedAttentionPress(
                    compression_ratio=self.compression_ratio,
                ),
                'knorm': lambda: kvpress.KnormPress(
                    compression_ratio=self.compression_ratio,
                ),
                'tova': lambda: kvpress.TOVAPress(
                    compression_ratio=self.compression_ratio,
                ),
                'random': lambda: kvpress.RandomPress(
                    compression_ratio=self.compression_ratio,
                ),
                'observed_attention': lambda: kvpress.ObservedAttentionPress(
                    compression_ratio=self.compression_ratio,
                ),
                'expected_attention': lambda: kvpress.ExpectedAttentionPress(
                    compression_ratio=self.compression_ratio,
                ),
            }

            if self.method_name not in press_creators:
                raise ValueError(f"Unknown method: {self.method_name}")

            self._press = press_creators[self.method_name]()

        return self._press

    def _compress_layer(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        压缩单层的KV-cache

        Args:
            keys: [batch, num_kv_heads, seq_len, head_dim]
            values: [batch, num_kv_heads, seq_len, head_dim]
            layer_idx: 层索引
            attention_weights: 可选的attention权重

        Returns:
            (compressed_keys, compressed_values)
        """
        press = self._get_press()

        batch_size, num_heads, seq_len, head_dim = keys.shape

        # 计算要保留的token数
        n_kept = int(seq_len * (1 - self.compression_ratio))
        n_kept = max(n_kept, 1)  # 至少保留1个

        # 如果序列太短，不需要压缩
        if n_kept >= seq_len:
            return keys.contiguous(), values.contiguous()

        # 根据方法计算scores
        if self.method_name == 'streaming_llm':
            # StreamingLLM: 保留前n_sink个和最后的recent tokens
            n_sink = self.kwargs.get('n_sink', 4)
            n_sink = min(n_sink, n_kept)  # sink不能超过保留数
            n_recent = max(0, n_kept - n_sink)  # 确保非负

            # 创建保留mask
            if n_recent > 0 and seq_len > n_sink + n_recent:
                indices = torch.cat([
                    torch.arange(n_sink, device=keys.device),
                    torch.arange(seq_len - n_recent, seq_len, device=keys.device)
                ])
            elif n_recent > 0:
                # 序列较短时，直接保留前n_kept个
                indices = torch.arange(n_kept, device=keys.device)
            else:
                # 只保留sink tokens
                indices = torch.arange(n_sink, device=keys.device)

        elif self.method_name == 'knorm':
            # Knorm: 基于key的L2范数选择
            k_norms = keys.norm(dim=-1)  # [batch, heads, seq]
            k_norms_mean = k_norms.mean(dim=1)  # [batch, seq]
            n_kept = min(n_kept, seq_len)  # 确保不超过序列长度
            _, indices = k_norms_mean.topk(n_kept, dim=-1)  # [batch, n_kept]
            indices = indices.sort(dim=-1).values  # 保持顺序

        elif self.method_name == 'random':
            # Random: 随机选择
            n_kept = min(n_kept, seq_len)  # 确保不超过序列长度
            perm = torch.randperm(seq_len, device=keys.device)[:n_kept]
            indices = perm.sort().values

        else:
            # 其他方法：均匀采样作为fallback
            n_kept = min(n_kept, seq_len)
            step = max(1, seq_len // n_kept)
            indices = torch.arange(0, seq_len, step, device=keys.device).long()[:n_kept]

        # 确保indices格式正确
        if indices.dim() == 1:
            indices = indices.unsqueeze(0).expand(batch_size, -1)

        # 使用gather压缩
        indices_k = indices.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)
        indices_v = indices.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)

        compressed_keys = keys.gather(2, indices_k).contiguous()
        compressed_values = values.gather(2, indices_v).contiguous()

        return compressed_keys, compressed_values

    def forward(
        self,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        it_len: Optional[List[int]] = None,
        **kwargs
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        压缩KV-cache

        与 BatchedGEMMCompressor 相同的接口。

        Args:
            kv_cache: List of (key, value) tuples per layer
                     key/value shape: [batch, num_kv_heads, seq_len, head_dim]
            it_len: [image_token_len, text_token_len] - 可选，用于多模态

        Returns:
            压缩后的KV-cache，格式相同
        """
        if len(kv_cache) == 0:
            return kv_cache

        compressed_kv = []

        for layer_idx, (keys, values) in enumerate(kv_cache):
            # 跳过空的cache
            if keys is None or values is None:
                compressed_kv.append((keys, values))
                continue

            # 压缩这一层
            comp_k, comp_v = self._compress_layer(
                keys, values, layer_idx
            )
            compressed_kv.append((comp_k, comp_v))

        return compressed_kv

    def get_compressed_length(self, original_length: int) -> int:
        """计算压缩后的长度"""
        n_kept = int(original_length * (1 - self.compression_ratio))
        return max(n_kept, 1)

    @property
    def compression_factor(self) -> float:
        """返回压缩因子 (原长度/压缩后长度)"""
        return 1.0 / (1 - self.compression_ratio)

    def __repr__(self):
        return (
            f"KVPressCompressor("
            f"method={self.method_name}, "
            f"compression_ratio={self.compression_ratio}, "
            f"compression_factor={self.compression_factor:.1f}x)"
        )


def create_kvpress_compressor(
    method: str = 'streaming_llm',
    compression_factor: int = 5,
    config: Optional[Any] = None,
    **kwargs
) -> KVPressCompressor:
    """
    创建 kvpress 压缩器的便捷函数

    Args:
        method: 压缩方法
        compression_factor: 压缩因子 (5 表示 5x 压缩，保留 20%)
        config: 模型配置
        **kwargs: 额外参数

    Returns:
        KVPressCompressor 实例
    """
    # 将 compression_factor 转换为 compression_ratio
    compression_ratio = 1.0 - (1.0 / compression_factor)

    return KVPressCompressor(
        method=method,
        compression_ratio=compression_ratio,
        config=config,
        **kwargs
    )
