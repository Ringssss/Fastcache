# kvcache_pool.py

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import threading
from contextlib import contextmanager
import time


"""
v1.1 kvcachepool
保留了预分配，将在v1.2版本去掉

v1.2
添加了充分的内存移交程序。


"""


@dataclass
class KVCacheItem:
    """简化的KVCache数据结构，只存储模型生成的KV Cache"""
    layers: List[Tuple[torch.Tensor, torch.Tensor]]  # 存储模型生成的past_key_value
    layers_size: int  # 层数
    batch_size: int  # batch大小
    sequence_length: int  # 序列长度
    last_access_time: float  # 最后访问时间


class KVCachePool:
    """简化版的KV Cache存储池，专注于存储模型生成的KV Cache"""

    def __init__(
            self,
            device: str = "cuda"
    ):
        """初始化KV Cache存储池"""
        self.device = device

        # 存储KV Cache的字典
        self._cache_store: Dict[str, KVCacheItem] = {}

        # 线程锁
        self._lock = threading.Lock()

    @contextmanager
    def _pool_lock(self):
        """线程锁的上下文管理器"""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    def store_kvcache(
            self,
            request_ids: Union[str, List[str]],
            past_key_value: List[Tuple[torch.Tensor, torch.Tensor]],
            sequence_length: Optional[int] = None
    ) -> bool:
        """存储模型生成的past_key_value

        Args:
            request_ids: 单个或多个请求ID
            past_key_value: 模型生成的past_key_value列表
            sequence_length: 可选的序列长度参数

        Returns:
            bool: 存储是否成功
        """
        if isinstance(request_ids, str):
            request_ids = [request_ids]

        batch_size = past_key_value[0][0].shape[0]
        seq_length = sequence_length or past_key_value[0][0].shape[2]

        # 检查batch_size和请求数的关系
        if batch_size < len(request_ids):
            raise ValueError("Batch size smaller than number of request IDs")

        with self._lock:
            try:
                # 对每个请求ID分别处理
                for idx, request_id in enumerate(request_ids):
                    # 如果是batch>1的情况，需要分离对应的KV Cache
                    if batch_size > 1:
                        layers = []
                        for k, v in past_key_value:
                            # 提取对应idx的key和value
                            k_single = k[idx:idx + 1]
                            v_single = v[idx:idx + 1]
                            layers.append((k_single, v_single))
                    else:
                        # batch=1的情况，直接使用
                        layers = past_key_value

                    # 创建新的cache item
                    cache_item = KVCacheItem(
                        layers=layers,
                        layers_size=len(past_key_value),
                        batch_size=1,  # 每个cache item只存储一个样本
                        sequence_length=seq_length,
                        last_access_time=time.time()
                    )

                    # 存储到缓存字典中
                    self._cache_store[request_id] = cache_item

                return True

            except Exception as e:
                print(f"存储KV Cache时发生错误: {str(e)}")
                return False

    def get_kvcache(
            self,
            request_ids: Union[str, List[str]]
    ) -> Optional[Union[List[Tuple[torch.Tensor, torch.Tensor]], List[List[Tuple[torch.Tensor, torch.Tensor]]]]]:
        """获取存储的KV Cache"""
        with self._lock:
            if isinstance(request_ids, str):
                return self._cache_store[request_ids].layers if request_ids in self._cache_store else None

            if not all(rid in self._cache_store for rid in request_ids):
                return None

            return [self._cache_store[rid].layers for rid in request_ids]

    def release_kvcache(self, request_ids: Union[str, List[str]]) -> bool:
        """释放指定请求的KV Cache

        Args:
            request_ids: 单个或多个请求ID

        Returns:
            bool: 释放是否成功
        """
        if isinstance(request_ids, str):
            request_ids = [request_ids]

        with self._lock:
            try:
                # 收集要释放的cache_items
                items_to_release = []
                for rid in request_ids:
                    if rid in self._cache_store:
                        items_to_release.append(self._cache_store[rid])
                        del self._cache_store[rid]

                # 释放收集到的cache_items的内存
                for cache_item in items_to_release:
                    for k, v in cache_item.layers:
                        # 处理梯度
                        if k.requires_grad:
                            k.detach_()
                        if v.requires_grad:
                            v.detach_()
                        # 清零内存
                        k.zero_()
                        v.zero_()
                        # 删除张量引用
                        del k
                        del v
                    # 删除layers引用
                    del cache_item.layers

                # 删除临时列表
                del items_to_release

                # # 触发垃圾回收
                # gc.collect()
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()

                return True

            except Exception as e:
                print(f"释放KV Cache时发生错误: {str(e)}")
                return False

    def clear_kvcache(self):
        """清除所有KV Cache并释放内存"""
        with self._lock:
            try:
                # 获取所有cache_items的副本，避免在迭代时修改字典
                cache_items = list(self._cache_store.values())

                # 清空存储字典
                self._cache_store.clear()

                # 释放所有KV Cache的内存
                for cache_item in cache_items:
                    for k, v in cache_item.layers:
                        # 处理需要梯度的情况
                        if k.requires_grad:
                            k.detach_()
                        if v.requires_grad:
                            v.detach_()
                        # 清零内存
                        k.zero_()
                        v.zero_()
                        # 删除引用
                        del k
                        del v
                    # 删除layers引用
                    del cache_item.layers

                # 删除cache_items列表
                del cache_items

                # # 强制触发垃圾回收
                # gc.collect()
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()

            except Exception as e:
                print(f"清除KV Cache时发生错误: {str(e)}")

    # def update_kvcache(
    #         self,
    #         request_id: str,
    #         new_past_key_value: List[Tuple[torch.Tensor, torch.Tensor]]
    # ) -> bool:
    #     """更新指定请求的KV Cache"""
    #     with self._lock:
    #         if request_id not in self._cache_store:
    #             return False
    #
    #         try:
    #             cache_item = self._cache_store[request_id]
    #
    #             # 释放旧的KV Cache内存
    #             for old_k, old_v in cache_item.layers:
    #                 # 确保张量在GPU上且需要梯度时也能正确释放
    #                 if old_k.requires_grad:
    #                     old_k.detach_()
    #                 if old_v.requires_grad:
    #                     old_v.detach_()
    #                 old_k.zero_()
    #                 old_v.zero_()
    #                 del old_k
    #                 del old_v
    #             cache_item.layers.clear()
    #
    #             cache_item.layers = new_past_key_value
    #             cache_item.last_access_time = time.time()
    #             cache_item.sequence_length = new_past_key_value[0][0].shape[2]
    #             return True
    #         except Exception as e:
    #             print(f"更新KV Cache时发生错误: {str(e)}")
    #             return False

    def merge_and_pop(
            self,
            request_ids: List[str]
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """将多个请求的KVCache合并并从内存池中移除，转移内存所有权给调用方

        Args:
            request_ids: 请求ID列表，按照需要合并的顺序排列

        Returns:
            如果成功，返回合并后的past_key_value列表；如果失败返回None
            返回的past_key_value中的张量按request_ids的顺序在batch维度上合并
            调用方接管返回的张量的内存所有权
        """
        with self._lock:
            # 检查所有请求ID是否都存在
            if not all(rid in self._cache_store for rid in request_ids):
                return None

            # try:
                # 获取第一个缓存的基本信息
            first_cache = self._cache_store[request_ids[0]]
            num_layers = first_cache.layers_size

            # 准备收集要释放的items
            items_to_release = []
            items_to_release_rid = []

            # 准备合并后的past_key_value列表
            merged_past_kv = []

            # 对每一层进行合并
            for layer_idx in range(num_layers):
                # 收集这一层所有请求的key和value
                keys = []
                values = []
                for rid in request_ids:
                    cache_item = self._cache_store[rid]
                    k, v = cache_item.layers[layer_idx]
                    # 创建连续内存的副本
                    keys.append(k.contiguous())
                    values.append(v.contiguous())
                    if rid not in items_to_release_rid:
                        items_to_release.append(cache_item)
                        items_to_release_rid.append(rid)

                # 在batch维度上合并
                merged_k = torch.cat(keys, dim=0)
                merged_v = torch.cat(values, dim=0)
                merged_past_kv.append((merged_k, merged_v))

            # 从存储中移除这些KVCache
            for rid in request_ids:
                self._cache_store[rid].layers.clear()
                del self._cache_store[rid]

            # 释放原始cache items的内存
            for cache_item in items_to_release:
                for k, v in cache_item.layers:
                    if k.requires_grad:
                        k.detach_()
                    if v.requires_grad:
                        v.detach_()
                    k.zero_()
                    v.zero_()
                    del k
                    del v
                cache_item.layers.clear()
                del cache_item.layers

            # 删除临时列表和引用
            del items_to_release
            del keys
            del values

            # # 触发垃圾回收
            # gc.collect()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

            return merged_past_kv

            # except Exception as e:
            #     print(f"合并KVCache时发生错误: {str(e)}")
            #     return None

    def pop_kvcache(
            self,
            request_ids: List[str]
    ) -> Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """从内存池中移除多个KVCache并转移内存所有权给调用方

        Args:
            request_ids: 请求ID列表，按照需要的顺序排列

        Returns:
            如果成功，返回KVCache列表的列表；如果失败返回None
            返回的列表中每个元素对应一个请求的past_key_value
            调用方接管返回的张量的内存所有权
        """
        with self._lock:
            # 检查所有请求ID是否都存在
            if not all(rid in self._cache_store for rid in request_ids):
                return None

            try:
                # 准备返回结果
                kvcache_list = []

                # 按请求ID顺序获取并移除KVCache
                for rid in request_ids:
                    cache_item = self._cache_store[rid]
                    # 直接使用原始layers，避免创建新的副本
                    kvcache_list.append(cache_item.layers)
                    # 清空对象但保留layers引用
                    cache_item.layers = []
                    # 从存储中移除
                    del self._cache_store[rid]
                    # 删除cache_item对象
                    del cache_item

                return kvcache_list

            except Exception as e:
                print(f"移除KVCache时发生错误: {str(e)}")
                return None

    def get_stats(self) -> Dict:
        """获取存储池状态统计"""
        with self._lock:
            total_memory = sum(
                sum(k.nelement() * k.element_size() + v.nelement() * v.element_size()
                    for k, v in cache.layers)
                for cache in self._cache_store.values()
            )

            return {
                "active_caches": len(self._cache_store),
                "total_memory": total_memory
            }



#
# # 使用示例
# def main():
#     # 初始化简化版KV Cache存储池
#     cache_pool = SimplifiedKVCachePool(device="cuda")
#
#     # 模拟存储KV Cache
#     mock_past_kv = [
#         (torch.randn(1, 32, 128, 128), torch.randn(1, 32, 128, 128))
#         for _ in range(32)  # 假设32层
#     ]
#
#     # 存储KV Cache
#     cache_pool.store_kvcache("req_1", mock_past_kv)
#     print(cache_pool.get_stats())
#
#     # 获取存储的KV Cache
#     stored_kv = cache_pool.get_kvcache("req_1")
#
#     # 释放KV Cache
#     cache_pool.release_kvcache("req_1")
#
#     # 打印统计信息
#     print(cache_pool.get_stats())
#
#
# if __name__ == "__main__":
#     main()