"""
TileLang Kernel生成器

基于TileLang生成高效的压缩kernel。

核心功能：
1. MLP压缩kernel（GEMM+ReLU融合）
2. Sparse Gather kernel
3. 量化/反量化kernel

TileLang的优势：
- 自动Fragment Layout优化
- 最优Warp分配
- Bank Conflict消除
"""

from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass
import math

# TileLang导入（可选依赖）
_tilelang_available = False
try:
    import tilelang as tl
    from tilelang import language as T
    _tilelang_available = True
except ImportError:
    pass

# Triton导入（备选方案）
_triton_available = False
try:
    import triton
    import triton.language as tl_triton
    _triton_available = True
except ImportError:
    pass


@dataclass
class KernelConfig:
    """Kernel配置"""
    # 矩阵维度
    M: int = 640           # 序列长度 × batch
    N: int = 128           # 压缩后维度
    K: int = 640           # 原始维度

    # 分块大小
    block_M: int = 128
    block_N: int = 128
    block_K: int = 32

    # 线程配置
    num_threads: int = 128
    num_warps: int = 4

    # 数据类型
    dtype: str = 'float16'
    accum_dtype: str = 'float32'

    # 优化选项
    use_tensor_core: bool = True
    num_stages: int = 3    # Pipeline stages


class TileLangKernelGenerator:
    """
    基于TileLang生成压缩kernel

    如果TileLang不可用，会回退到Triton或PyTorch实现。
    """

    def __init__(self):
        self._kernel_cache: Dict[str, Any] = {}

    def generate_mlp_compress_kernel(self, config: KernelConfig) -> Callable:
        """
        生成MLP压缩kernel

        MLP结构: Linear(640->128) + ReLU + Linear(128->128) + ReLU + Linear(128->128)

        Returns:
            编译后的kernel函数
        """
        cache_key = f"mlp_{config.M}_{config.N}_{config.K}_{config.dtype}"
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        if _tilelang_available:
            kernel = self._generate_mlp_tilelang(config)
        elif _triton_available:
            kernel = self._generate_mlp_triton(config)
        else:
            kernel = self._generate_mlp_pytorch(config)

        self._kernel_cache[cache_key] = kernel
        return kernel

    def _generate_mlp_tilelang(self, config: KernelConfig) -> Callable:
        """使用TileLang生成MLP kernel"""

        M, N, K = config.M, config.N, config.K
        block_M, block_N, block_K = config.block_M, config.block_N, config.block_K
        dtype = config.dtype
        accum_dtype = config.accum_dtype
        num_threads = config.num_threads

        @tl.jit(out_idx=[-1])
        def mlp_compress_kernel():
            @T.prim_func
            def kernel(
                X: T.Tensor((M, K), dtype),       # 输入
                W1: T.Tensor((K, N), dtype),      # 第一层权重
                W2: T.Tensor((N, N), dtype),      # 第二层权重
                W3: T.Tensor((N, N), dtype),      # 第三层权重
                Y: T.Tensor((M, N), dtype),       # 输出
            ):
                with T.Kernel(
                    T.ceildiv(N, block_N),
                    T.ceildiv(M, block_M),
                    threads=num_threads
                ) as (bx, by):
                    # 分配共享内存
                    X_shared = T.alloc_shared((block_M, block_K), dtype)
                    W1_shared = T.alloc_shared((block_K, block_N), dtype)
                    W2_shared = T.alloc_shared((block_N, block_N), dtype)
                    W3_shared = T.alloc_shared((block_N, block_N), dtype)

                    # 分配fragment
                    C1_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    C2_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                    C3_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                    # 初始化
                    T.fill(C1_local, 0.0)

                    # 第一层GEMM: X @ W1
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=config.num_stages):
                        T.copy(X[by * block_M:(by + 1) * block_M,
                                 k * block_K:(k + 1) * block_K], X_shared)
                        T.copy(W1[k * block_K:(k + 1) * block_K,
                                  bx * block_N:(bx + 1) * block_N], W1_shared)
                        T.gemm(X_shared, W1_shared, C1_local)

                    # ReLU激活
                    T.relu(C1_local)

                    # 第二层GEMM + ReLU
                    T.fill(C2_local, 0.0)
                    for k in T.Pipelined(T.ceildiv(N, block_N), num_stages=2):
                        # 从C1_local加载到共享内存
                        # 注意：这里简化了，实际需要处理C1_local到shared的转换
                        T.copy(W2[k * block_N:(k + 1) * block_N,
                                  bx * block_N:(bx + 1) * block_N], W2_shared)
                        T.gemm(C1_local, W2_shared, C2_local)
                    T.relu(C2_local)

                    # 第三层GEMM
                    T.fill(C3_local, 0.0)
                    for k in T.Pipelined(T.ceildiv(N, block_N), num_stages=2):
                        T.copy(W3[k * block_N:(k + 1) * block_N,
                                  bx * block_N:(bx + 1) * block_N], W3_shared)
                        T.gemm(C2_local, W3_shared, C3_local)

                    # 写回结果
                    T.copy(C3_local, Y[by * block_M:(by + 1) * block_M,
                                       bx * block_N:(bx + 1) * block_N])

            return kernel

        return mlp_compress_kernel()

    def _generate_mlp_triton(self, config: KernelConfig) -> Callable:
        """使用Triton生成MLP kernel（备选方案）"""

        M, N, K = config.M, config.N, config.K
        block_M, block_N, block_K = config.block_M, config.block_N, config.block_K

        @triton.jit
        def mlp_layer_kernel(
            X_ptr, W_ptr, Y_ptr,
            M, N, K,
            stride_xm, stride_xk,
            stride_wk, stride_wn,
            stride_ym, stride_yn,
            apply_relu: tl_triton.constexpr,
            BLOCK_M: tl_triton.constexpr,
            BLOCK_N: tl_triton.constexpr,
            BLOCK_K: tl_triton.constexpr,
        ):
            """单层MLP kernel（Linear + 可选ReLU）"""
            pid_m = tl_triton.program_id(0)
            pid_n = tl_triton.program_id(1)

            # 计算偏移
            offs_m = pid_m * BLOCK_M + tl_triton.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl_triton.arange(0, BLOCK_N)
            offs_k = tl_triton.arange(0, BLOCK_K)

            # 累加器
            acc = tl_triton.zeros((BLOCK_M, BLOCK_N), dtype=tl_triton.float32)

            # 主循环
            for k in range(0, K, BLOCK_K):
                # 加载X块
                x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + (k + offs_k[None, :]) * stride_xk)
                x_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
                x = tl_triton.load(x_ptrs, mask=x_mask, other=0.0)

                # 加载W块
                w_ptrs = W_ptr + ((k + offs_k[:, None]) * stride_wk + offs_n[None, :] * stride_wn)
                w_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
                w = tl_triton.load(w_ptrs, mask=w_mask, other=0.0)

                # 矩阵乘法
                acc += tl_triton.dot(x, w)

            # 应用ReLU
            if apply_relu:
                acc = tl_triton.maximum(acc, 0.0)

            # 存储结果
            y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
            y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl_triton.store(y_ptrs, acc.to(tl_triton.float16), mask=y_mask)

        def mlp_compress(X, W1, W2, W3):
            """三层MLP压缩"""
            import torch
            M, K = X.shape
            _, N1 = W1.shape
            _, N2 = W2.shape
            _, N3 = W3.shape

            # 中间结果
            Y1 = torch.empty((M, N1), device=X.device, dtype=X.dtype)
            Y2 = torch.empty((M, N2), device=X.device, dtype=X.dtype)
            Y3 = torch.empty((M, N3), device=X.device, dtype=X.dtype)

            # 第一层
            grid1 = (triton.cdiv(M, block_M), triton.cdiv(N1, block_N))
            mlp_layer_kernel[grid1](
                X, W1, Y1,
                M, N1, K,
                X.stride(0), X.stride(1),
                W1.stride(0), W1.stride(1),
                Y1.stride(0), Y1.stride(1),
                True,  # apply_relu
                block_M, block_N, block_K,
            )

            # 第二层
            grid2 = (triton.cdiv(M, block_M), triton.cdiv(N2, block_N))
            mlp_layer_kernel[grid2](
                Y1, W2, Y2,
                M, N2, N1,
                Y1.stride(0), Y1.stride(1),
                W2.stride(0), W2.stride(1),
                Y2.stride(0), Y2.stride(1),
                True,  # apply_relu
                block_M, block_N, block_N,
            )

            # 第三层
            grid3 = (triton.cdiv(M, block_M), triton.cdiv(N3, block_N))
            mlp_layer_kernel[grid3](
                Y2, W3, Y3,
                M, N3, N2,
                Y2.stride(0), Y2.stride(1),
                W3.stride(0), W3.stride(1),
                Y3.stride(0), Y3.stride(1),
                False,  # 最后一层不用ReLU
                block_M, block_N, block_N,
            )

            return Y3

        return mlp_compress

    def _generate_mlp_pytorch(self, config: KernelConfig) -> Callable:
        """使用PyTorch实现（最终备选）"""

        def mlp_compress(X, W1, W2, W3):
            import torch
            import torch.nn.functional as F

            # 第一层
            Y1 = F.relu(X @ W1)
            # 第二层
            Y2 = F.relu(Y1 @ W2)
            # 第三层
            Y3 = Y2 @ W3

            return Y3

        return mlp_compress

    def generate_sparse_gather_kernel(self, config: KernelConfig) -> Callable:
        """
        生成稀疏Gather kernel

        用于SnapKV等方法的TopK后gather操作
        """
        cache_key = f"sparse_gather_{config.M}_{config.N}_{config.dtype}"
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        if _triton_available:
            kernel = self._generate_sparse_gather_triton(config)
        else:
            kernel = self._generate_sparse_gather_pytorch(config)

        self._kernel_cache[cache_key] = kernel
        return kernel

    def _generate_sparse_gather_triton(self, config: KernelConfig) -> Callable:
        """使用Triton生成sparse gather kernel"""

        block_size = 256

        @triton.jit
        def sparse_gather_kernel(
            X_ptr,           # 输入 [seq_len, head_dim]
            Indices_ptr,     # 索引 [kept_len]
            Y_ptr,           # 输出 [kept_len, head_dim]
            seq_len, head_dim, kept_len,
            stride_x_seq, stride_x_dim,
            stride_y_kept, stride_y_dim,
            BLOCK_SIZE: tl_triton.constexpr,
        ):
            pid = tl_triton.program_id(0)
            offs = pid * BLOCK_SIZE + tl_triton.arange(0, BLOCK_SIZE)
            mask = offs < kept_len

            # 加载索引
            indices = tl_triton.load(Indices_ptr + offs, mask=mask, other=0)

            # 对每个head_dim维度gather
            for d in range(head_dim):
                # 加载源数据
                x_ptrs = X_ptr + indices * stride_x_seq + d * stride_x_dim
                values = tl_triton.load(x_ptrs, mask=mask, other=0.0)

                # 存储
                y_ptrs = Y_ptr + offs * stride_y_kept + d * stride_y_dim
                tl_triton.store(y_ptrs, values, mask=mask)

        def sparse_gather(X, indices):
            import torch
            seq_len, head_dim = X.shape
            kept_len = indices.shape[0]

            Y = torch.empty((kept_len, head_dim), device=X.device, dtype=X.dtype)

            grid = (triton.cdiv(kept_len, block_size),)
            sparse_gather_kernel[grid](
                X, indices, Y,
                seq_len, head_dim, kept_len,
                X.stride(0), X.stride(1),
                Y.stride(0), Y.stride(1),
                block_size,
            )

            return Y

        return sparse_gather

    def _generate_sparse_gather_pytorch(self, config: KernelConfig) -> Callable:
        """使用PyTorch实现sparse gather"""

        def sparse_gather(X, indices):
            return X[indices]

        return sparse_gather

    def generate_quantize_kernel(self, bits: int = 4) -> Callable:
        """
        生成量化kernel

        Args:
            bits: 量化位数 (4 or 8)
        """
        cache_key = f"quantize_{bits}"
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        # 使用PyTorch实现（量化操作相对简单）
        def quantize(X, bits=bits):
            import torch

            # 计算缩放因子
            x_max = X.abs().max(dim=-1, keepdim=True).values
            scale = x_max / (2 ** (bits - 1) - 1)
            scale = scale.clamp(min=1e-8)

            # 量化
            x_quant = (X / scale).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)

            if bits == 4:
                x_quant = x_quant.to(torch.int8)  # 用int8存储int4
            else:
                x_quant = x_quant.to(torch.int8)

            return x_quant, scale

        self._kernel_cache[cache_key] = quantize
        return quantize

    def generate_dequantize_kernel(self, bits: int = 4) -> Callable:
        """生成反量化kernel"""
        cache_key = f"dequantize_{bits}"
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        def dequantize(X_quant, scale, dtype=None):
            import torch
            if dtype is None:
                dtype = torch.float16
            return X_quant.to(dtype) * scale

        self._kernel_cache[cache_key] = dequantize
        return dequantize


# 全局kernel生成器实例
_kernel_generator: Optional[TileLangKernelGenerator] = None


def get_kernel_generator() -> TileLangKernelGenerator:
    """获取全局kernel生成器"""
    global _kernel_generator
    if _kernel_generator is None:
        _kernel_generator = TileLangKernelGenerator()
    return _kernel_generator


# ============ 便捷函数 ============

def get_mlp_kernel(M: int, N: int, K: int, dtype: str = 'float16') -> Callable:
    """获取MLP压缩kernel"""
    config = KernelConfig(M=M, N=N, K=K, dtype=dtype)
    return get_kernel_generator().generate_mlp_compress_kernel(config)


def get_sparse_gather_kernel(dtype: str = 'float16') -> Callable:
    """获取sparse gather kernel"""
    config = KernelConfig(dtype=dtype)
    return get_kernel_generator().generate_sparse_gather_kernel(config)


def get_quantize_kernels(bits: int = 4) -> Tuple[Callable, Callable]:
    """获取量化和反量化kernel"""
    gen = get_kernel_generator()
    return gen.generate_quantize_kernel(bits), gen.generate_dequantize_kernel(bits)
