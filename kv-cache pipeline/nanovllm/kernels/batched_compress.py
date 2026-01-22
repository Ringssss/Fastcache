"""
Batched GEMM KV-Cache Compressor
================================

核心优化思路：
1. 将32层的压缩合并成批量GEMM，提高GPU利用率
2. 使用Triton实现融合的 Linear + ReLU kernel
3. 支持流水线异步压缩

性能对比:
- 原始实现: 32层 × 4个MLP × 3个Linear = 384次小GEMM
- 批量实现: 4个批量GEMM × 3层 = 12次大GEMM

Author: Claude Code
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, List, Tuple, Dict
from enum import Enum
import threading


# ============================================================================
# Triton Kernels for Batched GEMM with Fused Activation
# ============================================================================

@triton.jit
def batched_linear_relu_kernel(
    # Input: [batch_gemm, M, K]
    X_ptr,
    # Weight: [batch_gemm, K, N]
    W_ptr,
    # Bias: [batch_gemm, N]
    B_ptr,
    # Output: [batch_gemm, M, N]
    Y_ptr,
    # Dimensions
    batch_gemm,  # number of parallel GEMMs (e.g., 32 layers)
    M,  # tokens per layer
    N,  # output_dim
    K,  # input_dim
    # Apply ReLU
    apply_relu: tl.constexpr,
    # Strides for X
    stride_xb, stride_xm, stride_xk,
    # Strides for W
    stride_wb, stride_wk, stride_wn,
    # Strides for B
    stride_bb, stride_bn,
    # Strides for Y
    stride_yb, stride_ym, stride_yn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batched GEMM with optional fused ReLU

    Computes: Y[b] = ReLU(X[b] @ W[b] + B[b]) for b in [0, batch_gemm)

    This kernel processes multiple independent GEMMs in parallel,
    which is perfect for compressing 32 layers simultaneously.
    """
    # Program IDs
    pid_batch = tl.program_id(0)  # Which layer/batch
    pid_m = tl.program_id(1)      # Which M block
    pid_n = tl.program_id(2)      # Which N block

    # Bounds check for batch
    if pid_batch >= batch_gemm:
        return

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Compute base pointers for this batch
    X_batch_ptr = X_ptr + pid_batch * stride_xb
    W_batch_ptr = W_ptr + pid_batch * stride_wb

    # Tiled matrix multiplication
    for k in range(0, K, BLOCK_K):
        # Load X block: [BLOCK_M, BLOCK_K]
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = k + tl.arange(0, BLOCK_K)

        x_ptrs = X_batch_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

        # Load W block: [BLOCK_K, BLOCK_N]
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        w_ptrs = W_batch_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(x, w)

    # Add bias
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_ptrs = B_ptr + pid_batch * stride_bb + offs_n * stride_bn
    b_mask = offs_n < N
    b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    # Apply ReLU if needed
    if apply_relu:
        acc = tl.maximum(acc, 0.0)

    # Store result
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    Y_batch_ptr = Y_ptr + pid_batch * stride_yb
    y_ptrs = Y_batch_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=y_mask)


def batched_linear_relu(
    x: torch.Tensor,      # [batch_gemm, M, K]
    weight: torch.Tensor, # [batch_gemm, K, N]
    bias: torch.Tensor,   # [batch_gemm, N]
    apply_relu: bool = True
) -> torch.Tensor:
    """
    Batched Linear + ReLU using Triton

    Args:
        x: Input tensor [batch_gemm, M, K]
        weight: Weight tensor [batch_gemm, K, N]
        bias: Bias tensor [batch_gemm, N]
        apply_relu: Whether to apply ReLU

    Returns:
        Output tensor [batch_gemm, M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()

    batch_gemm, M, K = x.shape
    _, K_, N = weight.shape
    assert K == K_

    # Allocate output
    y = torch.empty((batch_gemm, M, N), device=x.device, dtype=x.dtype)

    # Block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    # Grid
    grid = (batch_gemm, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
    batched_linear_relu_kernel[grid](
        x, weight, bias, y,
        batch_gemm, M, N, K,
        apply_relu,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1), weight.stride(2),
        bias.stride(0), 1,  # bias stride
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )

    return y


# ============================================================================
# Batched GEMM Compressor
# ============================================================================

class BatchedGEMMCompressor(nn.Module):
    """
    使用批量GEMM优化的KV-Cache压缩器

    优化点：
    1. 将32层的权重堆叠成批量张量 [32, in_dim, out_dim]
    2. 将32层的输入堆叠成 [32, batch*heads*seq, dim]
    3. 用一次批量GEMM完成所有层的计算

    性能提升预期：
    - 减少kernel launch开销 (384次 -> 12次)
    - 更好的GPU利用率 (大矩阵 vs 小矩阵)
    - 更少的Python循环开销
    """

    def __init__(
        self,
        num_layers: int = 32,
        num_heads: int = 8,
        head_dim: int = 128,
        compression_factor: int = 5,
        min_seq_len: int = 1,
        use_triton: bool = True
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.compression_factor = compression_factor
        self.min_seq_len = min_seq_len
        self.use_triton = use_triton

        input_dim = head_dim * compression_factor  # 640
        hidden_dim = head_dim  # 128

        # 批量权重: [num_layers, in_dim, out_dim]
        # Text K compressor (3 layers MLP)
        self.w1_tk = nn.Parameter(torch.empty(num_layers, input_dim, hidden_dim))
        self.b1_tk = nn.Parameter(torch.empty(num_layers, hidden_dim))
        self.w2_tk = nn.Parameter(torch.empty(num_layers, hidden_dim, hidden_dim))
        self.b2_tk = nn.Parameter(torch.empty(num_layers, hidden_dim))
        self.w3_tk = nn.Parameter(torch.empty(num_layers, hidden_dim, hidden_dim))
        self.b3_tk = nn.Parameter(torch.empty(num_layers, hidden_dim))

        # Text V compressor
        self.w1_tv = nn.Parameter(torch.empty(num_layers, input_dim, hidden_dim))
        self.b1_tv = nn.Parameter(torch.empty(num_layers, hidden_dim))
        self.w2_tv = nn.Parameter(torch.empty(num_layers, hidden_dim, hidden_dim))
        self.b2_tv = nn.Parameter(torch.empty(num_layers, hidden_dim))
        self.w3_tv = nn.Parameter(torch.empty(num_layers, hidden_dim, hidden_dim))
        self.b3_tv = nn.Parameter(torch.empty(num_layers, hidden_dim))

        # Image K compressor
        self.w1_ik = nn.Parameter(torch.empty(num_layers, input_dim, hidden_dim))
        self.b1_ik = nn.Parameter(torch.empty(num_layers, hidden_dim))
        self.w2_ik = nn.Parameter(torch.empty(num_layers, hidden_dim, hidden_dim))
        self.b2_ik = nn.Parameter(torch.empty(num_layers, hidden_dim))
        self.w3_ik = nn.Parameter(torch.empty(num_layers, hidden_dim, hidden_dim))
        self.b3_ik = nn.Parameter(torch.empty(num_layers, hidden_dim))

        # Image V compressor
        self.w1_iv = nn.Parameter(torch.empty(num_layers, input_dim, hidden_dim))
        self.b1_iv = nn.Parameter(torch.empty(num_layers, hidden_dim))
        self.w2_iv = nn.Parameter(torch.empty(num_layers, hidden_dim, hidden_dim))
        self.b2_iv = nn.Parameter(torch.empty(num_layers, hidden_dim))
        self.w3_iv = nn.Parameter(torch.empty(num_layers, hidden_dim, hidden_dim))
        self.b3_iv = nn.Parameter(torch.empty(num_layers, hidden_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for name, param in self.named_parameters():
            if 'w' in name:
                nn.init.xavier_uniform_(param)
            elif 'b' in name:
                nn.init.zeros_(param)

    def load_from_original_compressor(self, original_compressor):
        """
        从原始的KVCacheLinearDecoupleCompressor加载权重

        原始结构:
        - compress_tk[layer_idx] = Sequential(Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear)
        - compress_tv[layer_idx] = ...
        - compress_ik[layer_idx] = ...
        - compress_iv[layer_idx] = ...
        """
        device = next(original_compressor.parameters()).device
        dtype = next(original_compressor.parameters()).dtype

        with torch.no_grad():
            for layer_idx in range(self.num_layers):
                # Text K
                mlp = original_compressor.compress_tk[layer_idx]
                self.w1_tk[layer_idx] = mlp[0].weight.T.contiguous()
                self.b1_tk[layer_idx] = mlp[0].bias
                self.w2_tk[layer_idx] = mlp[3].weight.T.contiguous()
                self.b2_tk[layer_idx] = mlp[3].bias
                self.w3_tk[layer_idx] = mlp[6].weight.T.contiguous()
                self.b3_tk[layer_idx] = mlp[6].bias

                # Text V
                mlp = original_compressor.compress_tv[layer_idx]
                self.w1_tv[layer_idx] = mlp[0].weight.T.contiguous()
                self.b1_tv[layer_idx] = mlp[0].bias
                self.w2_tv[layer_idx] = mlp[3].weight.T.contiguous()
                self.b2_tv[layer_idx] = mlp[3].bias
                self.w3_tv[layer_idx] = mlp[6].weight.T.contiguous()
                self.b3_tv[layer_idx] = mlp[6].bias

                # Image K
                mlp = original_compressor.compress_ik[layer_idx]
                self.w1_ik[layer_idx] = mlp[0].weight.T.contiguous()
                self.b1_ik[layer_idx] = mlp[0].bias
                self.w2_ik[layer_idx] = mlp[3].weight.T.contiguous()
                self.b2_ik[layer_idx] = mlp[3].bias
                self.w3_ik[layer_idx] = mlp[6].weight.T.contiguous()
                self.b3_ik[layer_idx] = mlp[6].bias

                # Image V
                mlp = original_compressor.compress_iv[layer_idx]
                self.w1_iv[layer_idx] = mlp[0].weight.T.contiguous()
                self.b1_iv[layer_idx] = mlp[0].bias
                self.w2_iv[layer_idx] = mlp[3].weight.T.contiguous()
                self.b2_iv[layer_idx] = mlp[3].bias
                self.w3_iv[layer_idx] = mlp[6].weight.T.contiguous()
                self.b3_iv[layer_idx] = mlp[6].bias

        # Move to same device and dtype
        self.to(device=device, dtype=dtype)
        print(f"✓ 批量压缩器权重加载完成 (device={device}, dtype={dtype})")

    def _batched_mlp_forward(
        self,
        x: torch.Tensor,  # [num_layers, M, input_dim]
        w1: torch.Tensor, b1: torch.Tensor,
        w2: torch.Tensor, b2: torch.Tensor,
        w3: torch.Tensor, b3: torch.Tensor,
    ) -> torch.Tensor:
        """
        批量执行3层MLP

        Args:
            x: [num_layers, M, input_dim]
            w1, b1: Layer 1 weights [num_layers, input_dim, hidden_dim]
            w2, b2: Layer 2 weights [num_layers, hidden_dim, hidden_dim]
            w3, b3: Layer 3 weights [num_layers, hidden_dim, hidden_dim]

        Returns:
            [num_layers, M, hidden_dim]
        """
        if self.use_triton and x.is_cuda:
            # Triton path - 融合Linear + ReLU
            x = x.contiguous()

            # Layer 1: Linear + ReLU
            x = batched_linear_relu(x, w1, b1, apply_relu=True)

            # Layer 2: Linear + ReLU
            x = batched_linear_relu(x, w2, b2, apply_relu=True)

            # Layer 3: Linear only
            x = batched_linear_relu(x, w3, b3, apply_relu=False)
        else:
            # PyTorch fallback - 使用torch.bmm
            # Layer 1
            x = torch.bmm(x, w1) + b1.unsqueeze(1)
            x = F.relu(x)

            # Layer 2
            x = torch.bmm(x, w2) + b2.unsqueeze(1)
            x = F.relu(x)

            # Layer 3
            x = torch.bmm(x, w3) + b3.unsqueeze(1)

        return x

    def _compress_batched(
        self,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        w1_k, b1_k, w2_k, b2_k, w3_k, b3_k,
        w1_v, b1_v, w2_v, b2_v, w3_v, b3_v,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        批量压缩KV-cache

        Args:
            kv_cache: List of (K, V) tuples, each [batch, heads, seq, dim]
            w*_k, b*_k: K压缩器权重
            w*_v, b*_v: V压缩器权重

        Returns:
            Compressed KV-cache
        """
        if len(kv_cache) == 0:
            return kv_cache

        # 获取形状信息
        batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape
        num_layers = len(kv_cache)

        # 计算压缩参数
        compressed_seq_len = seq_len // self.compression_factor
        if compressed_seq_len < self.min_seq_len:
            return kv_cache

        compress_len = compressed_seq_len * self.compression_factor

        # Step 1: 堆叠所有层的KV-cache
        # [num_layers, batch, heads, seq, dim]
        all_k = torch.stack([kv[0] for kv in kv_cache], dim=0)
        all_v = torch.stack([kv[1] for kv in kv_cache], dim=0)

        # Step 2: 截取需要压缩的部分并reshape
        # [num_layers, batch, heads, compress_len, dim]
        # -> [num_layers, batch*heads*compressed_seq, dim*factor]
        k_to_compress = all_k[:, :, :, :compress_len, :].reshape(
            num_layers, batch_size * num_heads * compressed_seq_len,
            head_dim * self.compression_factor
        )
        v_to_compress = all_v[:, :, :, :compress_len, :].reshape(
            num_layers, batch_size * num_heads * compressed_seq_len,
            head_dim * self.compression_factor
        )

        # Step 3: 批量MLP压缩
        k_compressed = self._batched_mlp_forward(
            k_to_compress, w1_k, b1_k, w2_k, b2_k, w3_k, b3_k
        )
        v_compressed = self._batched_mlp_forward(
            v_to_compress, w1_v, b1_v, w2_v, b2_v, w3_v, b3_v
        )

        # Step 4: Reshape回原始格式
        # [num_layers, batch*heads*compressed_seq, dim]
        # -> [num_layers, batch, heads, compressed_seq, dim]
        k_compressed = k_compressed.reshape(
            num_layers, batch_size, num_heads, compressed_seq_len, head_dim
        )
        v_compressed = v_compressed.reshape(
            num_layers, batch_size, num_heads, compressed_seq_len, head_dim
        )

        # Step 5: 处理剩余部分并重组
        result = []
        for layer_idx in range(num_layers):
            k_out = k_compressed[layer_idx]
            v_out = v_compressed[layer_idx]

            # 拼接剩余部分
            if seq_len > compress_len:
                k_out = torch.cat([k_out, all_k[layer_idx, :, :, compress_len:, :]], dim=2)
                v_out = torch.cat([v_out, all_v[layer_idx, :, :, compress_len:, :]], dim=2)

            result.append((k_out.contiguous(), v_out.contiguous()))

        return result

    def forward(
        self,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        it_len: List[int] = [0, 1]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        压缩KV-cache

        Args:
            kv_cache: List of (K, V) tuples for each layer
                     Each tensor shape: [batch, heads, seq, dim]
            it_len: [image_len, text_len]

        Returns:
            Compressed KV-cache in same format
        """
        if len(kv_cache) == 0:
            return kv_cache

        batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape
        image_len = it_len[0]

        if image_len > 0:
            # 分离图像和文本部分
            image_kv = [(k[:, :, :image_len, :], v[:, :, :image_len, :]) for k, v in kv_cache]
            text_kv = [(k[:, :, image_len:, :], v[:, :, image_len:, :]) for k, v in kv_cache]

            # 分别压缩
            compressed_image = self._compress_batched(
                image_kv,
                self.w1_ik, self.b1_ik, self.w2_ik, self.b2_ik, self.w3_ik, self.b3_ik,
                self.w1_iv, self.b1_iv, self.w2_iv, self.b2_iv, self.w3_iv, self.b3_iv,
            )
            compressed_text = self._compress_batched(
                text_kv,
                self.w1_tk, self.b1_tk, self.w2_tk, self.b2_tk, self.w3_tk, self.b3_tk,
                self.w1_tv, self.b1_tv, self.w2_tv, self.b2_tv, self.w3_tv, self.b3_tv,
            )

            # 合并
            result = []
            for (ik, iv), (tk, tv) in zip(compressed_image, compressed_text):
                k = torch.cat([ik, tk], dim=2)
                v = torch.cat([iv, tv], dim=2)
                result.append((k.contiguous(), v.contiguous()))

            return result
        else:
            # 只压缩文本
            return self._compress_batched(
                kv_cache,
                self.w1_tk, self.b1_tk, self.w2_tk, self.b2_tk, self.w3_tk, self.b3_tk,
                self.w1_tv, self.b1_tv, self.w2_tv, self.b2_tv, self.w3_tv, self.b3_tv,
            )

    # =========================================================================
    # 分层压缩 - 用于零开销调度
    # =========================================================================

    def compress_layers(
        self,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        layer_start: int,
        layer_end: int,
        it_len: List[int] = [0, 1],
        stream: torch.cuda.Stream = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        压缩指定范围的层

        用于将压缩任务分片，塞入decode的计算间隙

        Args:
            kv_cache: 完整的KV-cache (所有层)
            layer_start: 起始层（包含）
            layer_end: 结束层（不包含）
            it_len: [image_len, text_len]
            stream: 使用的CUDA stream（None=当前stream）

        Returns:
            压缩后的KV-cache（只包含指定层）
        """
        if layer_end <= layer_start or len(kv_cache) == 0:
            return []

        # 提取指定层
        layer_kv = kv_cache[layer_start:layer_end]

        if len(layer_kv) == 0:
            return []

        batch_size, num_heads, seq_len, head_dim = layer_kv[0][0].shape
        image_len = it_len[0]

        # 在指定stream上执行
        ctx = torch.cuda.stream(stream) if stream else torch.cuda.stream(torch.cuda.current_stream())

        with ctx:
            if image_len > 0:
                # 分离图像和文本
                image_kv = [(k[:, :, :image_len, :], v[:, :, :image_len, :]) for k, v in layer_kv]
                text_kv = [(k[:, :, image_len:, :], v[:, :, image_len:, :]) for k, v in layer_kv]

                # 分别压缩（使用对应层的权重切片）
                compressed_image = self._compress_layer_slice(
                    image_kv, layer_start, layer_end,
                    self.w1_ik, self.b1_ik, self.w2_ik, self.b2_ik, self.w3_ik, self.b3_ik,
                    self.w1_iv, self.b1_iv, self.w2_iv, self.b2_iv, self.w3_iv, self.b3_iv,
                )
                compressed_text = self._compress_layer_slice(
                    text_kv, layer_start, layer_end,
                    self.w1_tk, self.b1_tk, self.w2_tk, self.b2_tk, self.w3_tk, self.b3_tk,
                    self.w1_tv, self.b1_tv, self.w2_tv, self.b2_tv, self.w3_tv, self.b3_tv,
                )

                # 合并
                result = []
                for (ik, iv), (tk, tv) in zip(compressed_image, compressed_text):
                    k = torch.cat([ik, tk], dim=2)
                    v = torch.cat([iv, tv], dim=2)
                    result.append((k.contiguous(), v.contiguous()))
                return result
            else:
                # 只压缩文本
                return self._compress_layer_slice(
                    layer_kv, layer_start, layer_end,
                    self.w1_tk, self.b1_tk, self.w2_tk, self.b2_tk, self.w3_tk, self.b3_tk,
                    self.w1_tv, self.b1_tv, self.w2_tv, self.b2_tv, self.w3_tv, self.b3_tv,
                )

    def _compress_layer_slice(
        self,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        layer_start: int,
        layer_end: int,
        w1_k, b1_k, w2_k, b2_k, w3_k, b3_k,
        w1_v, b1_v, w2_v, b2_v, w3_v, b3_v,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        压缩指定层切片（使用对应的权重切片）
        """
        if len(kv_cache) == 0:
            return kv_cache

        batch_size, num_heads, seq_len, head_dim = kv_cache[0][0].shape
        num_layers = len(kv_cache)

        # 计算压缩参数
        compressed_seq_len = seq_len // self.compression_factor
        if compressed_seq_len < self.min_seq_len:
            return kv_cache

        compress_len = compressed_seq_len * self.compression_factor

        # 堆叠KV
        all_k = torch.stack([kv[0] for kv in kv_cache], dim=0)
        all_v = torch.stack([kv[1] for kv in kv_cache], dim=0)

        # Reshape
        k_to_compress = all_k[:, :, :, :compress_len, :].reshape(
            num_layers, batch_size * num_heads * compressed_seq_len,
            head_dim * self.compression_factor
        )
        v_to_compress = all_v[:, :, :, :compress_len, :].reshape(
            num_layers, batch_size * num_heads * compressed_seq_len,
            head_dim * self.compression_factor
        )

        # 获取权重切片
        w1_k_slice = w1_k[layer_start:layer_end]
        b1_k_slice = b1_k[layer_start:layer_end]
        w2_k_slice = w2_k[layer_start:layer_end]
        b2_k_slice = b2_k[layer_start:layer_end]
        w3_k_slice = w3_k[layer_start:layer_end]
        b3_k_slice = b3_k[layer_start:layer_end]

        w1_v_slice = w1_v[layer_start:layer_end]
        b1_v_slice = b1_v[layer_start:layer_end]
        w2_v_slice = w2_v[layer_start:layer_end]
        b2_v_slice = b2_v[layer_start:layer_end]
        w3_v_slice = w3_v[layer_start:layer_end]
        b3_v_slice = b3_v[layer_start:layer_end]

        # 批量MLP压缩
        k_compressed = self._batched_mlp_forward(
            k_to_compress, w1_k_slice, b1_k_slice, w2_k_slice, b2_k_slice, w3_k_slice, b3_k_slice
        )
        v_compressed = self._batched_mlp_forward(
            v_to_compress, w1_v_slice, b1_v_slice, w2_v_slice, b2_v_slice, w3_v_slice, b3_v_slice
        )

        # Reshape回去
        k_compressed = k_compressed.reshape(
            num_layers, batch_size, num_heads, compressed_seq_len, head_dim
        )
        v_compressed = v_compressed.reshape(
            num_layers, batch_size, num_heads, compressed_seq_len, head_dim
        )

        # 重组结果
        result = []
        for idx in range(num_layers):
            k_out = k_compressed[idx]
            v_out = v_compressed[idx]

            if seq_len > compress_len:
                k_out = torch.cat([k_out, all_k[idx, :, :, compress_len:, :]], dim=2)
                v_out = torch.cat([v_out, all_v[idx, :, :, compress_len:, :]], dim=2)

            result.append((k_out.contiguous(), v_out.contiguous()))

        return result


# ============================================================================
# Pipeline Async Compression Manager
# ============================================================================

class CompressionState(Enum):
    """压缩状态"""
    NONE = 0
    PENDING = 1
    IN_PROGRESS = 2
    COMPLETED = 3


class PipelineAsyncCompressor:
    """
    流水线异步压缩器

    核心思想：让当前batch的压缩与下一个batch的prefill重叠

    时间线示例:
    Request1: [Prefill1][--------Compress1--------]
    Request2:          [Prefill2][--------Compress2--------]
    Request3:                   [Prefill3][--------Compress3--------]

    主Stream:  Prefill1 -> Prefill2 -> Prefill3 -> ...
    压缩Stream:         Compress1 -> Compress2 -> Compress3 -> ...

    关键点：
    1. Prefill完成后立即启动压缩（在压缩stream）
    2. 下一个Prefill可以立即开始（在主stream）
    3. Decode需要等待对应的压缩完成
    """

    def __init__(
        self,
        compressor: BatchedGEMMCompressor,
        num_streams: int = 2  # 压缩流数量
    ):
        self.compressor = compressor
        self.num_streams = num_streams

        # 创建CUDA streams
        self.compress_streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_stream_idx = 0

        # 状态跟踪
        self.seq_states: Dict[int, CompressionState] = {}
        self.seq_events: Dict[int, torch.cuda.Event] = {}
        self.seq_compressed_lens: Dict[int, int] = {}

        # 原始KV-cache备份（用于压缩期间的decode）
        self.pending_compressions: Dict[int, Tuple] = {}  # seq_id -> (kv_cache, it_len)

        # 线程锁
        self.lock = threading.Lock()

    def _get_next_stream(self) -> torch.cuda.Stream:
        """轮询获取下一个可用的压缩stream"""
        stream = self.compress_streams[self.current_stream_idx]
        self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams
        return stream

    def submit_compression(
        self,
        seq_id: int,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_table: List[int],
        original_seq_len: int,
        it_len: List[int],
        write_callback  # 回调函数，用于写回压缩结果
    ) -> torch.cuda.Event:
        """
        提交异步压缩任务

        Args:
            seq_id: 序列ID
            kv_cache: HF格式的KV-cache
            block_table: 块表（用于写回）
            original_seq_len: 原始序列长度
            it_len: [image_len, text_len]
            write_callback: 写回压缩结果的回调函数

        Returns:
            完成事件，可用于同步等待
        """
        with self.lock:
            self.seq_states[seq_id] = CompressionState.PENDING

        # 获取压缩stream
        compress_stream = self._get_next_stream()

        # 记录主stream的事件
        main_event = torch.cuda.Event()
        main_event.record()

        # 压缩stream等待主stream完成
        compress_stream.wait_event(main_event)

        # 在压缩stream中执行压缩
        with torch.cuda.stream(compress_stream):
            with self.lock:
                self.seq_states[seq_id] = CompressionState.IN_PROGRESS

            try:
                # 执行压缩
                with torch.no_grad():
                    compressed_kv = self.compressor(kv_cache, it_len=it_len)

                # 获取压缩后长度
                compressed_seq_len = compressed_kv[0][0].shape[2]

                # 写回压缩结果
                write_callback(compressed_kv, block_table)

                with self.lock:
                    self.seq_compressed_lens[seq_id] = compressed_seq_len
                    self.seq_states[seq_id] = CompressionState.COMPLETED

            except Exception as e:
                print(f"异步压缩失败 seq_id={seq_id}: {e}")
                with self.lock:
                    self.seq_states[seq_id] = CompressionState.NONE

        # 记录完成事件
        done_event = torch.cuda.Event()
        done_event.record(compress_stream)

        with self.lock:
            self.seq_events[seq_id] = done_event

        return done_event

    def is_compression_ready(self, seq_id: int) -> bool:
        """检查压缩是否完成"""
        with self.lock:
            state = self.seq_states.get(seq_id, CompressionState.NONE)
            if state != CompressionState.COMPLETED:
                return False

            event = self.seq_events.get(seq_id)
            if event is not None:
                return event.query()
            return True

    def wait_compression(self, seq_id: int):
        """等待指定序列的压缩完成"""
        event = None
        with self.lock:
            event = self.seq_events.get(seq_id)

        if event is not None:
            event.synchronize()

    def get_compressed_len(self, seq_id: int) -> Optional[int]:
        """获取压缩后的长度"""
        with self.lock:
            return self.seq_compressed_lens.get(seq_id)

    def cleanup(self, seq_id: int):
        """清理序列状态"""
        with self.lock:
            self.seq_states.pop(seq_id, None)
            self.seq_events.pop(seq_id, None)
            self.seq_compressed_lens.pop(seq_id, None)
            self.pending_compressions.pop(seq_id, None)


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_batched_vs_original():
    """比较批量GEMM和原始实现的性能"""
    import time
    import sys
    sys.path.append('/home/zhujianian/cvpr')
    sys.path.append('/home/zhujianian/cvpr/utils_ccm')

    device = torch.device('cuda')
    dtype = torch.float16

    # 模拟参数
    num_layers = 32
    batch_size = 1
    num_heads = 8
    seq_len = 600
    head_dim = 128
    compression_factor = 5

    print("=" * 60)
    print("批量GEMM vs 原始实现 性能对比")
    print("=" * 60)
    print(f"Layers: {num_layers}, Batch: {batch_size}, Heads: {num_heads}")
    print(f"Seq: {seq_len}, HeadDim: {head_dim}, Factor: {compression_factor}")

    # 创建测试数据
    kv_cache = [
        (
            torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype),
            torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        )
        for _ in range(num_layers)
    ]

    it_len = [0, seq_len]  # 纯文本

    # 测试批量GEMM压缩器
    print("\n--- 批量GEMM压缩器 ---")
    batched_compressor = BatchedGEMMCompressor(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        compression_factor=compression_factor,
        use_triton=True
    ).to(device=device, dtype=dtype)

    # 预热
    for _ in range(5):
        with torch.no_grad():
            _ = batched_compressor(kv_cache, it_len)
    torch.cuda.synchronize()

    # 计时
    num_iters = 50
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = batched_compressor(kv_cache, it_len)
    torch.cuda.synchronize()
    batched_time = (time.time() - start) / num_iters * 1000
    print(f"Triton批量GEMM: {batched_time:.3f} ms")

    # PyTorch fallback
    batched_compressor.use_triton = False
    for _ in range(5):
        with torch.no_grad():
            _ = batched_compressor(kv_cache, it_len)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = batched_compressor(kv_cache, it_len)
    torch.cuda.synchronize()
    pytorch_batched_time = (time.time() - start) / num_iters * 1000
    print(f"PyTorch批量bmm: {pytorch_batched_time:.3f} ms")

    # 测试原始压缩器
    print("\n--- 原始压缩器 ---")
    try:
        from utils_ccm.module_ccm_v11 import KVCacheLinearDecoupleCompressor
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained("/data/huggingface/llava-1.5-7b-hf")
        original_compressor = KVCacheLinearDecoupleCompressor(
            src_config=config,
            compression_factor=compression_factor,
            min_seq_len=1
        ).to(device=device, dtype=dtype).eval()

        # 预热
        for _ in range(5):
            with torch.no_grad():
                _ = original_compressor(kv_cache, it_len)
        torch.cuda.synchronize()

        # 计时
        start = time.time()
        for _ in range(num_iters):
            with torch.no_grad():
                _ = original_compressor(kv_cache, it_len)
        torch.cuda.synchronize()
        original_time = (time.time() - start) / num_iters * 1000
        print(f"原始实现: {original_time:.3f} ms")

        print("\n--- 加速比 ---")
        print(f"Triton批量 vs 原始: {original_time / batched_time:.2f}x")
        print(f"PyTorch批量 vs 原始: {original_time / pytorch_batched_time:.2f}x")

    except Exception as e:
        print(f"原始压缩器测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    benchmark_batched_vs_original()
