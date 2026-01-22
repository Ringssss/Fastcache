"""
Tensor Parallelism Support for nano-vllm
========================================

支持多GPU张量并行推理。

使用方法:
    # 初始化TP
    from nanovllm.utils.tp import init_tensor_parallel, get_tp_rank, get_tp_size

    init_tensor_parallel(tp_size=2)

    # 在模型中使用
    tp_size = get_tp_size()
    tp_rank = get_tp_rank()

Author: Claude Code
"""

import os
import torch
import torch.distributed as dist
from typing import Optional

# 全局TP状态
_TP_INITIALIZED = False
_TP_SIZE = 1
_TP_RANK = 0
_TP_GROUP = None
_TP_DEVICE = None


def init_tensor_parallel(
    tp_size: int = 1,
    rank: Optional[int] = None,
    backend: str = 'nccl',
):
    """
    初始化张量并行

    Args:
        tp_size: 张量并行大小 (GPU数量)
        rank: 当前进程的rank，如果为None则从环境变量获取
        backend: 通信后端 ('nccl' 或 'gloo')
    """
    global _TP_INITIALIZED, _TP_SIZE, _TP_RANK, _TP_GROUP, _TP_DEVICE

    if _TP_INITIALIZED:
        print(f"Warning: TP already initialized with size {_TP_SIZE}")
        return

    if tp_size == 1:
        _TP_INITIALIZED = True
        _TP_SIZE = 1
        _TP_RANK = 0
        _TP_DEVICE = torch.device('cuda:0')
        print(f"TP disabled (single GPU mode)")
        return

    # 获取rank
    if rank is None:
        rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))

    # 初始化进程组
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    world_size = dist.get_world_size()

    if world_size != tp_size:
        raise ValueError(f"World size ({world_size}) != tp_size ({tp_size})")

    # 创建TP组
    ranks = list(range(tp_size))
    _TP_GROUP = dist.new_group(ranks)

    _TP_SIZE = tp_size
    _TP_RANK = rank
    _TP_DEVICE = torch.device(f'cuda:{rank}')
    _TP_INITIALIZED = True

    # 设置当前设备
    torch.cuda.set_device(_TP_DEVICE)

    print(f"TP initialized: rank={_TP_RANK}, size={_TP_SIZE}, device={_TP_DEVICE}")


def get_tp_size() -> int:
    """获取TP大小"""
    return _TP_SIZE


def get_tp_rank() -> int:
    """获取当前TP rank"""
    return _TP_RANK


def get_tp_group():
    """获取TP通信组"""
    return _TP_GROUP


def get_tp_device() -> torch.device:
    """获取当前设备"""
    return _TP_DEVICE if _TP_DEVICE else torch.device('cuda:0')


def is_tp_initialized() -> bool:
    """检查TP是否已初始化"""
    return _TP_INITIALIZED


def tp_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """
    TP组内all-reduce

    Args:
        tensor: 要reduce的张量

    Returns:
        reduce后的张量
    """
    if _TP_SIZE == 1:
        return tensor

    dist.all_reduce(tensor, group=_TP_GROUP)
    return tensor


def tp_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    TP组内all-gather

    Args:
        tensor: 要gather的张量
        dim: gather的维度

    Returns:
        gather后的张量
    """
    if _TP_SIZE == 1:
        return tensor

    # 创建输出列表
    gather_list = [torch.empty_like(tensor) for _ in range(_TP_SIZE)]
    dist.all_gather(gather_list, tensor, group=_TP_GROUP)

    # 沿指定维度拼接
    return torch.cat(gather_list, dim=dim)


def tp_broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    从src rank广播张量

    Args:
        tensor: 要广播的张量
        src: 源rank

    Returns:
        广播后的张量
    """
    if _TP_SIZE == 1:
        return tensor

    dist.broadcast(tensor, src=src, group=_TP_GROUP)
    return tensor


def tp_barrier():
    """TP组内barrier同步"""
    if _TP_SIZE > 1:
        dist.barrier(group=_TP_GROUP)


def cleanup_tensor_parallel():
    """清理TP资源"""
    global _TP_INITIALIZED, _TP_SIZE, _TP_RANK, _TP_GROUP, _TP_DEVICE

    if dist.is_initialized():
        dist.destroy_process_group()

    _TP_INITIALIZED = False
    _TP_SIZE = 1
    _TP_RANK = 0
    _TP_GROUP = None
    _TP_DEVICE = None


# 便捷函数
def get_tensor_model_parallel_world_size() -> int:
    """vLLM兼容接口"""
    return get_tp_size()


def get_tensor_model_parallel_rank() -> int:
    """vLLM兼容接口"""
    return get_tp_rank()
