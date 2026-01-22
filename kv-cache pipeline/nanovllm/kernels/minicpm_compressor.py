"""
MiniCPM KV-Cache Compressor for nano-vllm
==========================================

这是 KVCacheHybridCompressor 的完整版本，用于MiniCPM模型。
原始 module_ccm_v11.py 中的版本把 compress_ik/iv 注释掉了，
但训练的checkpoint包含这些权重，所以需要这个完整版本。

Author: Claude Code
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class MiniCPMKVCompressor(nn.Module):
    """
    MiniCPM的KV-Cache压缩器

    完整版本，包含compress_tk/tv/ik/iv和attention模块，
    与训练时的版本一致。
    """

    def __init__(self, src_config, compression_factor=5, min_seq_len=2):
        super(MiniCPMKVCompressor, self).__init__()

        # MiniCPM配置：直接从顶层读取
        self.n_layer = src_config.num_hidden_layers
        self.nhead = src_config.num_key_value_heads
        self.d_model = src_config.hidden_size
        # 关键：head_dim = hidden_size / num_attention_heads, 不是 / num_key_value_heads
        self.head_dim = src_config.hidden_size // src_config.num_attention_heads

        self.compression_factor = compression_factor
        self.min_seq_len = min_seq_len

        print(f"MiniCPMKVCompressor初始化:")
        print(f"  n_layer={self.n_layer}, nhead={self.nhead}, head_dim={self.head_dim}")
        print(f"  compression_factor={compression_factor}, min_seq_len={min_seq_len}")

        # Text compression modules
        self.compress_tk = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim * compression_factor, self.head_dim),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(self.head_dim, self.head_dim),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(self.head_dim, self.head_dim)
            ) for _ in range(self.n_layer)
        ])

        self.compress_tv = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim * compression_factor, self.head_dim),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(self.head_dim, self.head_dim),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(self.head_dim, self.head_dim)
            ) for _ in range(self.n_layer)
        ])

        # Image compression modules (完整版本需要这些)
        self.compress_ik = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim * compression_factor, self.head_dim),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(self.head_dim, self.head_dim),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(self.head_dim, self.head_dim)
            ) for _ in range(self.n_layer)
        ])

        self.compress_iv = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim * compression_factor, self.head_dim),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(self.head_dim, self.head_dim),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(self.head_dim, self.head_dim)
            ) for _ in range(self.n_layer)
        ])

        # Attention computation
        self.attention = nn.ModuleList([
            nn.Linear(self.head_dim, 1) for _ in range(self.n_layer)
        ])

    def compress_layer(self, layer_cache, layer_idx, compressor_k, compressor_v):
        """压缩单层的KV cache"""
        k, v = layer_cache
        batch_size, nhead, seq_len, head_dim = k.shape

        # 计算压缩后的序列长度
        compressed_seq_len = seq_len // self.compression_factor

        # 如果压缩后长度过短，返回原始序列
        if compressed_seq_len < self.min_seq_len:
            return k, v

        # 计算需要压缩的序列长度
        compress_len = compressed_seq_len * self.compression_factor

        # Reshape for compression
        k_to_compress = k[:, :, :compress_len, :].reshape(
            batch_size, nhead, compressed_seq_len, self.head_dim * self.compression_factor
        )
        v_to_compress = v[:, :, :compress_len, :].reshape(
            batch_size, nhead, compressed_seq_len, self.head_dim * self.compression_factor
        )

        # 压缩
        compressed_k = compressor_k[layer_idx](k_to_compress)
        compressed_v = compressor_v[layer_idx](v_to_compress)

        # 处理剩余的部分
        if seq_len > compress_len:
            remaining_k = k[:, :, compress_len:, :]
            remaining_v = v[:, :, compress_len:, :]
            compressed_k = torch.cat([compressed_k, remaining_k], dim=2)
            compressed_v = torch.cat([compressed_v, remaining_v], dim=2)

        return compressed_k, compressed_v

    def forward(
        self,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        it_len: List[int] = [0, 1],
        forward_mode: str = 'val'
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        压缩KV cache

        Args:
            kv_cache: List of (key, value) tuples, each [batch, heads, seq, dim]
            it_len: [image_len, text_len] - 图像和文本token的长度
            forward_mode: 'train' or 'val'

        Returns:
            压缩后的KV cache
        """
        compressed_kv_cache = []

        for layer_idx in range(self.n_layer):
            layer_k, layer_v = kv_cache[layer_idx]

            if it_len[0] == 0:
                # 纯文本：只使用text压缩器
                tk = layer_k
                tv = layer_v
                compressed_tk, compressed_tv = self.compress_layer(
                    (tk, tv), layer_idx, self.compress_tk, self.compress_tv
                )
                compressed_k = compressed_tk
                compressed_v = compressed_tv
            else:
                # 有图像：分别压缩图像和文本部分
                ik = layer_k[:, :, :it_len[0], :]
                iv = layer_v[:, :, :it_len[0], :]
                tk = layer_k[:, :, it_len[0]:, :]
                tv = layer_v[:, :, it_len[0]:, :]

                # 压缩图像部分
                compressed_ik, compressed_iv = self.compress_layer(
                    (ik, iv), layer_idx, self.compress_ik, self.compress_iv
                )
                # 压缩文本部分
                compressed_tk, compressed_tv = self.compress_layer(
                    (tk, tv), layer_idx, self.compress_tk, self.compress_tv
                )

                # 合并
                compressed_k = torch.cat([compressed_ik, compressed_tk], dim=2)
                compressed_v = torch.cat([compressed_iv, compressed_tv], dim=2)

            # 存储结果
            if forward_mode == 'train':
                compressed_kv_cache.append((compressed_k, compressed_v))
            else:
                compressed_kv_cache.append((
                    compressed_k.detach().contiguous(),
                    compressed_v.detach().contiguous()
                ))

        return compressed_kv_cache


def load_minicpm_compressor(compressor_path: str, hf_config, compression_factor: int = 5):
    """
    加载MiniCPM压缩器

    Args:
        compressor_path: checkpoint路径
        hf_config: MiniCPM的HuggingFace配置
        compression_factor: 压缩因子

    Returns:
        加载好权重的压缩器
    """
    import os

    compressor = MiniCPMKVCompressor(
        src_config=hf_config,
        compression_factor=compression_factor,
        min_seq_len=2
    ).cuda()

    if os.path.exists(compressor_path):
        checkpoint = torch.load(compressor_path, map_location='cuda')
        if 'model_state_dict' in checkpoint:
            if 'compressor' in checkpoint['model_state_dict']:
                compressor.load_state_dict(checkpoint['model_state_dict']['compressor'])
            else:
                compressor.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            compressor.load_state_dict(checkpoint, strict=False)
        print(f"✓ MiniCPM压缩器加载成功: {compressor_path}")
    else:
        print(f"⚠ 压缩器权重未找到: {compressor_path}")

    compressor.eval()
    return compressor
