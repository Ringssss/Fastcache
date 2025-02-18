import os
import torch
import json
import time
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime
from PIL import Image
from dataclasses import dataclass
import statistics
import logging
from typing import List, Dict, Optional, Any
from threading import Thread, Lock, Event
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import nvtx
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from utils_ccm.module_ccm import KVCacheLinearDecoupleCompressor
from utils_ccm.utils_compress import *
from utils_ccm.utils_schedule import *

"""
v1.0
最基本的全流程静态并发系统。测量时长无bug。

v1.1
实现pd分离管理，测试性能。
于2024.12.26日23点完成。

v1.2
加入测量指标监视器。
于2024.12.29日21点完成。

v1.2.1
comp_mode添加进设置中

v1.2.2
kvc attention mask debug

v1.3
完成c环节独立batching

v1.3.1 *
指标测量改动。ccm超参最优。

v1.4
结束cuda.syn的保护，由此带来大批涨速。


"""


# @dataclass
# class TestResult:
#     method: str  # 'orig' 或 'comp'
#     latency: List[float]
#     success: bool
#     valid_tokens: int = 0
#     memory_usage: float = 0.0
#     request_size: int = 0
#     throughput: float = 0.0
#     compression_ratio: Optional[float] = None
#     compression_time: Optional[float] = None
@dataclass
class TestResult:
    method: str
    latency: List[float]
    success: bool
    memory_usage: float
    request_id: List[str]
    request_size: int
    throughput: float
    valid_tokens: int
    compression_ratio: float = None
    compression_time: float = None
    timestamps: Dict[str, float] = None  # 添加时间戳信息


class PerformanceMonitor:
    """性能监控类"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.results = {
            'orig': {
                'latencies': [],
                'throughputs': [],
                'memory_usage': [],
                'request_id': [],
                'request_size': [],
                'success_count': 0,
                'valid_tokens': 0,
                'total_count': 0
            },
            'comp': {
                'latencies': [],
                'throughputs': [],
                'memory_usage': [],
                'request_id': [],
                'request_size': [],
                'compression_ratios': [],
                'compression_times': [],
                'success_count': 0,
                'valid_tokens': 0,
                'total_count': 0
            }
        }
        self.start_time = time.time()

    def add_result(self, result: TestResult):
        print('result.request_id', result.request_id)
        method = result.method
        self.results[method]['latencies'].append(result.latency)
        self.results[method]['throughputs'].append(result.throughput)
        self.results[method]['memory_usage'].append(result.memory_usage)
        self.results[method]['request_id'].append(result.request_id)
        self.results[method]['request_size'].append(result.request_size)
        self.results[method]['valid_tokens'] += int(result.valid_tokens)
        self.results[method]['success_count'] += int(result.success)
        self.results[method]['total_count'] += 1
        # print(result)

        if method == 'comp' and result.success:
            if result.compression_ratio:
                self.results[method]['compression_ratios'].append(result.compression_ratio)
            if result.compression_time:
                self.results[method]['compression_times'].append(result.compression_time)

    def proc_latencies(self):
        ttft_list = []
        tpot_list = []

        for method, data in self.results.items():
            if not data['latencies']:
                continue
            for i in range(len(data['latencies'])):
                ttft_list += [data['latencies'][i][1]] * data['request_size'][i]
                tpot_list += [data['latencies'][i][2]] * data['request_size'][i]
        return ttft_list, tpot_list

    def calculate_metrics(self, durations_dict, system_completion_times):
        metrics = {}
        duration = system_completion_times

        for method, data in self.results.items():
            if not data['latencies']:
                continue
            ttft_list, tpot_list = durations_dict['TTFT'], durations_dict['TPOT']

            sorted_ttft = sorted(ttft_list)
            sorted_tpot = sorted(tpot_list)
            total = len(sorted_ttft)
            p50_idx = int(total * 0.5)
            p90_idx = int(total * 0.90)
            p99_idx = min(int(total * 0.99), total - 1)
            metrics[method] = {
                "total_requests": data['total_count'],
                "success_rate": (data['success_count'] / data['total_count'] * 100) if data['total_count'] > 0 else 0,
                "requests_per_second": data['total_count'] / duration if duration > 0 else 0,
                "avg_ttft": statistics.mean(sorted_ttft),
                "median_ttft": sorted_ttft[p50_idx],
                "p90_ttft": sorted_ttft[p90_idx],
                "p99_ttft": sorted_ttft[p99_idx],
                "min_ttft": min(sorted_ttft),
                "max_ttft": max(sorted_ttft),
                "avg_tpot": statistics.mean(sorted_tpot),
                "median_tpot": sorted_tpot[p50_idx],
                "p90_tpot": sorted_tpot[p90_idx],
                "p99_tpot": sorted_tpot[p99_idx],
                "min_tpot": min(sorted_tpot),
                "max_tpot": max(sorted_tpot),
                "avg_memory_usage": statistics.mean(data['memory_usage']),
                "avg_throughput": data['valid_tokens'] / duration,
                # "avg_throughput": statistics.mean(data['throughputs']),
                "throughput_stddev": statistics.stdev(data['throughputs']) if len(data['throughputs']) > 1 else 0
            }

            if method == 'comp' and data.get('compression_ratios'):
                metrics[method].update({
                    "avg_compression_ratio": statistics.mean(data['compression_ratios']),
                    "avg_compression_time": statistics.mean(data['compression_times']) if data[
                        'compression_times'] else 0
                })

        return metrics

    # def calculate_metrics(self):
    #     metrics = {}
    #     duration = time.time() - self.start_time
    #
    #     for method, data in self.results.items():
    #         if not data['latencies']:
    #             continue
    #         ttft_list, tpot_list = self.proc_latencies()
    #
    #         sorted_ttft = sorted(ttft_list)
    #         sorted_tpot = sorted(tpot_list)
    #         total = len(sorted_ttft)
    #         p50_idx = int(total * 0.5)
    #         p90_idx = int(total * 0.90)
    #         p99_idx = min(int(total * 0.99), total - 1)
    #         metrics[method] = {
    #             "total_requests": data['total_count'],
    #             "success_rate": (data['success_count'] / data['total_count'] * 100) if data['total_count'] > 0 else 0,
    #             "requests_per_second": data['total_count'] / duration if duration > 0 else 0,
    #             "avg_ttft": statistics.mean(sorted_ttft),
    #             "median_ttft": sorted_ttft[p50_idx],
    #             "p90_ttft": sorted_ttft[p90_idx],
    #             "p99_ttft": sorted_ttft[p99_idx],
    #             "min_ttft": min(sorted_ttft),
    #             "max_ttft": max(sorted_ttft),
    #             "avg_tpot": statistics.mean(sorted_tpot),
    #             "median_tpot": sorted_tpot[p50_idx],
    #             "p90_tpot": sorted_tpot[p90_idx],
    #             "p99_tpot": sorted_tpot[p99_idx],
    #             "min_tpot": min(sorted_tpot),
    #             "max_tpot": max(sorted_tpot),
    #             "avg_memory_usage": statistics.mean(data['memory_usage']),
    #             "avg_throughput": data['valid_tokens'] / duration,
    #             # "avg_throughput": statistics.mean(data['throughputs']),
    #             "throughput_stddev": statistics.stdev(data['throughputs']) if len(data['throughputs']) > 1 else 0
    #         }
    #
    #         if method == 'comp' and data.get('compression_ratios'):
    #             metrics[method].update({
    #                 "avg_compression_ratio": statistics.mean(data['compression_ratios']),
    #                 "avg_compression_time": statistics.mean(data['compression_times']) if data[
    #                     'compression_times'] else 0
    #             })
    #
    #     return metrics

    # def calculate_metrics(self):
    #     metrics = {}
    #     duration = time.time() - self.start_time
    #
    #     ttft_list, tpot_list = self.proc_latencies()
    #
    #     for method, data in self.results.items():
    #         if not data['latencies']:
    #             continue
    #
    #         sorted_latencies = sorted(data['latencies'])
    #         total = len(sorted_latencies)
    #         p50_idx = int(total * 0.5)
    #         p95_idx = int(total * 0.95)
    #         p99_idx = min(int(total * 0.99), total - 1)
    #         metrics[method] = {
    #             "total_requests": data['total_count'],
    #             "success_rate": (data['success_count'] / data['total_count'] * 100) if data['total_count'] > 0 else 0,
    #             "requests_per_second": data['total_count'] / duration if duration > 0 else 0,
    #             "avg_latency": statistics.mean(data['latencies']),
    #             "median_latency": sorted_latencies[p50_idx],
    #             "p95_latency": sorted_latencies[p95_idx],
    #             "p99_latency": sorted_latencies[p99_idx],
    #             "min_latency": min(sorted_latencies),
    #             "max_latency": max(sorted_latencies),
    #             "avg_memory_usage": statistics.mean(data['memory_usage']),
    #             "avg_throughput": data['valid_tokens']/duration,
    #             # "avg_throughput": statistics.mean(data['throughputs']),
    #             "throughput_stddev": statistics.stdev(data['throughputs']) if len(data['throughputs']) > 1 else 0
    #         }
    #
    #         if method == 'comp' and data.get('compression_ratios'):
    #             metrics[method].update({
    #                 "avg_compression_ratio": statistics.mean(data['compression_ratios']),
    #                 "avg_compression_time": statistics.mean(data['compression_times']) if data['compression_times'] else 0
    #             })
    #
    #     return metrics


class ServiceMiddleware:
    def __init__(self, max_workers, max_serve_batch_size, min_batch_threshold, std_batch_threshold_decoding,
                 max_wait_time):
        self.max_workers = max_workers
        self.max_serve_batch_size = max_serve_batch_size
        self.min_batch_threshold = min_batch_threshold
        self.max_wait_time = max_wait_time

        # self.waiting_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.idle_workers = max_workers
        self.idle_lock = Lock()
        self.processed_prefill_batches = 0
        self.processed_decode_batches = 0
        self.system_total_time = 0
        self.batch_times = []
        self.completion_event = Event()
        self.should_stop = False
        self.start_time = None

        # 新增: 初始化动态调度器
        self.scheduler = DynamicScheduler(
            max_batch_size=max_serve_batch_size,
            min_batch_size=min_batch_threshold,
            std_batch_threshold_decoding=std_batch_threshold_decoding,
            max_wait_time=max_wait_time
        )

        # 使用Queue替代原有的waiting_queue
        self.waiting_queue = Queue()


class CustomImageTextDataset(Dataset):
    """自定义数据集类"""

    def __init__(self, data, processor, max_length=128):
        self.data = data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dialog_data = self.data[idx]
        questions = dialog_data[0]
        image_text, image = questions[0]

        vision_outputs = self.processor.image_processor(
            images=image,
            return_tensors="pt"
        )

        text_outputs = self.processor.tokenizer(
            image_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "pixel_values": vision_outputs.pixel_values.squeeze(0),
            "input_ids": text_outputs.input_ids.squeeze(0),
            "attention_mask": text_outputs.attention_mask.squeeze(0)
        }


class ImprovedConcurrentTester:
    """并发测试类"""

    def __init__(self, model, processor, compressor, max_new_tokens, num_samples, compress_batch,
                 compression_ratio_list, use_compression, comp_mode,
                 device):
        self.model = model
        self.processor = processor
        self.compressor = compressor
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples
        self.compress_batch = compress_batch
        self.compression_ratio_list = compression_ratio_list
        self.use_compression = use_compression
        self.comp_mode = comp_mode
        self.cuda_stream = torch.cuda.Stream()
        # self.timeCheck = TimeTik()
        self.monitor = PerformanceMonitor()
        self.timestamp_manager = TimestampManager()
        self.GPU_monitor = GPUMonitor(monitor_sys=False)

        self._init_press_list()

    def _init_press_list(self):
        self.press_type = [
            'ccm',
            'Knorm',
            'StreamingLLM',
            'RandomPress',
            'SnapKV',
            'ExpectedAttention',
            'Quantized',
        ]

    def _get_kv_cache_memory(self, past_key_values):
        """计算KV-cache内存占用"""
        total_memory = 0
        for layer in past_key_values:
            k, v = layer
            total_memory += k.numel() * k.element_size()
            total_memory += v.numel() * v.element_size()
        return total_memory / (1024 ** 2)  # 转换为MB

    def _split_past_kv_multi_batch(self, past_key_values, batch_indices=None):
        """
        从past_key_values中提取多个batch的KV缓存
        Args:
            past_key_values: 原始KV缓存元组
            batch_indices: list of int, 要提取的batch索引列表
        Returns:
            list of tuple, 每个batch对应的KV缓存列表
        """
        batch_size = past_key_values[0][0].shape[0]
        if batch_size == 1:
            return [past_key_values]
        if batch_indices is None:
            # 获取batch大小并生成完整索引列表
            batch_indices = list(range(batch_size))
        # batch_idx_tensor = torch.tensor(batch_indices, device=past_key_values[0][0].device)

        # num_layers = len(past_key_values)
        # batch_size = len(batch_indices)

        # 创建batch_size个空的past_key_values结构
        result = []
        for batch_idx in batch_indices:
            # 对每个batch，创建其完整的层结构
            batch_kv = list(
                (
                    layer_key.index_select(0, torch.tensor([batch_idx], device=layer_key.device)),
                    layer_value.index_select(0, torch.tensor([batch_idx], device=layer_value.device))
                )
                for layer_key, layer_value in past_key_values
            )
            result.append(batch_kv)
        return result

    def _integrated_compress_ccm(self, past_key_values_list, inputs, request_ids_list):
        past_key_values_list_final = []
        comp_past_key_values_list = []
        # print("past_key_values_list", len(past_key_values_list), len(past_key_values_list[0]))

        start_split_time = time.time()
        for past_key_values in past_key_values_list:
            past_key_values_list_final.extend(self._split_past_kv_multi_batch(past_key_values))

        print(len(past_key_values_list_final))
        print("start_split_time", time.time() - start_split_time)
        compress_batch_size = self.compress_batch
        start_p = 0
        comp_duration_time = 0
        final_compression_ratio_list = []
        # print("past_key_values_list_final", len(past_key_values_list_final))

        while len(past_key_values_list_final) != 0:
            target_past_key_values = past_key_values_list_final[:compress_batch_size]
            past_key_values_list_final = past_key_values_list_final[len(target_past_key_values):]
            end_p = start_p + len(target_past_key_values)
            orig_past_key_values = self.merge_kv_cache(target_past_key_values)

            orig_memory = self._get_kv_cache_memory(orig_past_key_values)
            kv_len = orig_past_key_values[0][0].shape[2]
            # print(inputs["input_ids"][start_p:end_p,:], start_p, end_p)
            comp_len = self.get_comp_len(inputs["input_ids"][start_p:end_p, :],
                                         inputs["attention_mask"][start_p:end_p, :], kv_len)
            # print("comp_len", comp_len)
            comp_start = time.time()
            self.timestamp_manager.record_timestamp(request_ids_list[start_p:end_p], 'compress_start', comp_start)

            comp_past_key_values = self.run_compressor(self.compressor,
                                                       orig_past_key_values, comp_len)
            # print("comp_past_key_values", comp_past_key_values[0][0].shape)
            # torch.cuda.synchronize()
            _ = comp_past_key_values[0][0].shape # torch.cuda.synchronize()代替

            compress_finish = time.time()
            self.timestamp_manager.record_timestamp(request_ids_list[start_p:end_p], 'compress_finish', compress_finish)
            self.timestamp_manager.record_timestamp(request_ids_list[start_p:end_p], 'decoding_queueing',
                                                    compress_finish)
            # print(self.compressor.)
            # print(comp_past_key_values)
            comp_past_key_values_list.append(comp_past_key_values)

            memory_usage = self._get_kv_cache_memory(comp_past_key_values)
            start_p = end_p
            comp_duration_time += compress_finish - comp_start
            final_compression_ratio_list.append(orig_memory / memory_usage if memory_usage > 0 else 0.0)
        final_compression_ratio = sum(final_compression_ratio_list) / len(final_compression_ratio_list)

        return comp_past_key_values_list, comp_duration_time, final_compression_ratio

    def find_valid_lens(self, tensor, token_id=2):
        """计算有效序列长度"""
        matches = tensor == token_id
        positions = torch.argmax(matches.int(), dim=1, keepdim=True)
        no_token_mask = ~matches.any(dim=1, keepdim=True)
        positions = positions.masked_fill(no_token_mask, tensor.shape[1] - 1)
        positions_list = positions.flatten().tolist()
        return positions.sum().item(), positions_list

    def get_comp_len(self, input_ids, attention_mask, kv_len):

        """计算压缩长度"""
        real_position = attention_mask.long().cumsum(-1) - 1
        # 预先访问一次tensor
        # image_insert_pos = torch.where(input_ids == 32000)[1]
        image_insert_pos = (input_ids == 32000).nonzero()[:, 1]
        padding_over_pos = (real_position == 0).nonzero()[:, 1]
        # padding_over_pos = torch.where(real_position == 0)[1]
        batch_size = input_ids.shape[0]
        comp_len = [[0, 0, 0] for _ in range(batch_size)]
        for i_idx, i_input in enumerate(input_ids):
            comp_len[i_idx][0] = int(padding_over_pos[i_idx])
            comp_len[i_idx][2] = i_input[image_insert_pos[i_idx] + 1:].shape[-1]
            comp_len[i_idx][1] = kv_len - comp_len[i_idx][2] - comp_len[i_idx][0]
        return comp_len

    def run_compressor(self, compressor, past_key_values, comp_len):
        """执行KV-Cache压缩"""
        it_len = comp_len[0][1:]  # 取出图像和文本长度
        compressed_past_key_values = compressor(past_key_values, it_len)
        return compressed_past_key_values

    def _exec_compress(self, comp_mode, inputs, request_ids_list):
        comp_past_key_values = None
        comp_duration_time = 0
        final_compression_ratio = 0
        prefill_time = 0

        if comp_mode == 'ccm':
            prefill_start = time.time()
            self.timestamp_manager.record_timestamp(request_ids_list, 'prefill_start', prefill_start)
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                use_cache=True,
                return_dict=True
            )
            # 获取原始KV cache并计算内存占用
            # orig_past_key_values = outputs.past_key_values
            # torch.cuda.synchronize()
            prefill_finish = time.time()
            prefill_time = prefill_finish - prefill_start
            # compress_start = time.time()
            self.timestamp_manager.record_timestamp(request_ids_list, 'prefill_finish', prefill_finish)
            orig_past_key_values = outputs.past_key_values
            # self.timestamp_manager.record_timestamp(request_ids_list, 'compress_queueing', prefill_finish)
            # self.timestamp_manager.record_timestamp(request_ids_list, 'compress_start', compress_start)


            # ccm方法则直接传送
            # orig_memory = self._get_kv_cache_memory(orig_past_key_values)
            # # past_key_values_split = self._split_past_kv_multi_batch(orig_past_key_values)
            # kv_len = orig_past_key_values[0][0].shape[2]
            # comp_len = self.get_comp_len(inputs["input_ids"], inputs["attention_mask"], kv_len)
            # comp_start_time = time.time()
            # comp_past_key_values = self.run_compressor(self.compressor,
            #                             orig_past_key_values, inputs["attention_mask"], comp_len)
            # compress_finish = time.time()
            # self.timestamp_manager.record_timestamp(request_ids_list, 'compress_finish', compress_finish)
            # # print(self.compressor.)
            # memory_usage = self._get_kv_cache_memory(comp_past_key_values)
            # comp_duration_time = compress_finish - comp_start_time
            # final_compression_ratio = orig_memory / memory_usage if memory_usage > 0 else 0.0
            return orig_past_key_values, 0, 1, prefill_time
        elif comp_mode in self.press_type:  # other comp

            # prefill_start = time.time()
            press = press_select(comp_mode, 1 - 1 / self.compression_ratio_list[1])
            print(press)
            underlying_model = get_underlying_model(self.model)
            prefill_start = time.time()
            self.timestamp_manager.record_timestamp(request_ids_list, 'prefill_start', prefill_start)
            comp_past_key_values = exec_compression_methods(self.model,
                                                            underlying_model, inputs, press,
                                                            comp_mode)

            _ = comp_past_key_values[0][0].shape
            # torch.cuda.synchronize()
            # todo press.totalTime的计时需不需要cuda syn
            compress_finish = time.time()
            comp_duration_time = press.totalTime
            prefill_finish = compress_finish - comp_duration_time
            self.timestamp_manager.record_timestamp(request_ids_list, 'prefill_finish', prefill_finish)
            self.timestamp_manager.record_timestamp(request_ids_list, 'compress_queueing', prefill_finish)
            self.timestamp_manager.record_timestamp(request_ids_list, 'compress_start', prefill_finish)
            self.timestamp_manager.record_timestamp(request_ids_list, 'compress_finish', compress_finish)
            final_compression_ratio = 1 / (1 - 1 / self.compression_ratio_list[1])
            return comp_past_key_values, comp_duration_time, final_compression_ratio, prefill_time
            # prefill_fini_ = time.time()
            # prefill_time = prefill_fini_ - prefill_start
            # 0.63 0.047 存在很大的系统开销
            # print("compress_time", prefill_time, comp_duration_time)

        else:
            print(f"Warn in _exec_compress: {comp_mode} matching failed.")
            return comp_past_key_values, comp_duration_time, final_compression_ratio, prefill_time
        # self.timeCheck.show_and_reset()

        # return comp_past_key_values, comp_duration_time, final_compression_ratio, prefill_time

    # def _process_batch(self, batch, use_compression=False, comp_mode='ccm'):
    #     # server
    #     """处理单个批次"""
    #     # try:
    #     with torch.cuda.stream(torch.cuda.Stream()):
    #         # 将输入移到设备上
    #         inputs = {
    #             'input_ids': batch['input_ids'].to(self.device),
    #             'attention_mask': batch['attention_mask'].to(self.device),
    #             'pixel_values': batch['pixel_values'].to(self.device)
    #         }
    #
    #         with torch.no_grad(), autocast(dtype=torch.float):
    #             # Forward pass 阶段
    #             input_ids_forward = inputs["input_ids"][:, :-1]
    #             request_size = inputs["input_ids"].shape[0]
    #             # outputs = self.model(
    #             #     input_ids=input_ids_forward,
    #             #     attention_mask=inputs["attention_mask"][:, :-1],
    #             #     pixel_values=inputs["pixel_values"],
    #             #     use_cache=True,
    #             #     return_dict=True
    #             # )
    #             #
    #             # # 获取原始KV cache并计算内存占用
    #             # orig_past_key_values = outputs.past_key_values
    #             # orig_memory = self._get_kv_cache_memory(orig_past_key_values)
    #             input_forward = {
    #                 "input_ids": inputs["input_ids"][:, :-1],
    #                 "attention_mask": inputs["attention_mask"][:, :-1],
    #                 "pixel_values": inputs["pixel_values"]
    #             }
    #             # 压缩阶段（如果启用）
    #             if use_compression:
    #                 past_key_values, compression_time, compression_ratio, \
    #                     prefill_time = self._exec_compress(comp_mode, input_forward)
    #                 memory_usage = self._get_kv_cache_memory(past_key_values)
    #                 # compression_ratio = orig_memory / memory_usage if memory_usage > 0 else 0.0
    #             else:
    #                 prefill_start = time.time()
    #                 outputs = self.model(
    #                     input_ids=input_ids_forward,
    #                     attention_mask=inputs["attention_mask"][:, :-1],
    #                     pixel_values=inputs["pixel_values"],
    #                     use_cache=True,
    #                     return_dict=True
    #                 )
    #                 torch.cuda.synchronize()
    #                 prefill_finish_ = time.time()
    #                 prefill_time = prefill_finish_ - prefill_start
    #                 # 获取原始KV cache并计算内存占用
    #                 orig_past_key_values = outputs.past_key_values
    #                 orig_memory = self._get_kv_cache_memory(orig_past_key_values)
    #                 past_key_values = orig_past_key_values
    #                 memory_usage = orig_memory
    #                 compression_ratio = None
    #                 compression_time = 0
    #             # prefill_time = time.time() - prefill_start
    #
    #             # 生成阶段
    #             generation_start = time.time()
    #
    #             # 准备生成输入
    #             batch_size = inputs['input_ids'].shape[0]
    #             kv_len = past_key_values[0][0].shape[2]
    #             kv_attention_mask = torch.cat([
    #                 inputs['attention_mask'],
    #                 torch.ones(
    #                     (batch_size, kv_len - inputs['attention_mask'].shape[1]),
    #                     dtype=inputs['attention_mask'].dtype,
    #                     device=self.device
    #                 )
    #             ], dim=-1)
    #
    #             # 准备generation输入
    #             input_ids = torch.cat([kv_attention_mask, inputs['input_ids'][:, -1].unsqueeze(-1)], dim=1)
    #             attention_mask = torch.cat([kv_attention_mask, inputs['attention_mask'][:, -1].unsqueeze(-1)], dim=1)
    #
    #             # 生成文本
    #             gen_outputs = self.model.generate(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 past_key_values=past_key_values,
    #                 max_new_tokens=512 - inputs['input_ids'].shape[1],
    #                 do_sample=True,
    #                 temperature=0.7,
    #                 use_cache=True,
    #                 return_dict_in_generate=True,
    #                 output_scores=True,
    #             )
    #
    #             # 处理生成的序列并计算性能指标
    #             sequences = gen_outputs.sequences
    #             start_idx = int(kv_attention_mask.shape[1])
    #             generated_sequences = sequences[:, start_idx:]
    #
    #             decoding_time = time.time() - generation_start
    #
    #             #todo waiting time
    #
    #             # todo valid_lens
    #             valid_lens = self.find_valid_lens(generated_sequences, self.processor.tokenizer.eos_token_id)
    #             # valid_lens = generated_sequences.numel()
    #             throughput = valid_lens / decoding_time if decoding_time > 0 else 0
    #
    #             tokens_time = decoding_time / valid_lens
    #             latency = [0, prefill_time, tokens_time]
    #
    #             # 清理GPU内存
    #             torch.cuda.empty_cache()
    #
    #             return TestResult(
    #                 method='comp' if use_compression else 'orig',
    #                 latency=latency,  # 只使用生成时间作为延迟
    #                 success=True,
    #                 memory_usage=memory_usage,
    #                 request_size=request_size,
    #                 throughput=throughput,
    #                 valid_tokens=valid_lens,
    #                 compression_ratio=compression_ratio,
    #                 compression_time=compression_time,
    #
    #             )
    #
    #     # except Exception as e:
    #     #     print(f"Error in batch processing: {str(e)}")
    #     #     return TestResult(
    #     #         method='comp' if use_compression else 'orig',
    #     #         latency=0,
    #     #         success=False,
    #     #         memory_usage=0,
    #     #         request_size=0,
    #     #         compression_time=0,
    #     #         throughput=0,
    #     #         valid_tokens=0,
    #     #         compression_ratio=None,
    #     #         compression_time=None
    #     #     )

    def _batching_tensors(self, dict_list):
        """

        Args:
            dict_list (List[Dict[str, torch.Tensor]]): 字典列表，每个字典包含形状为[1, ...]的张量

        Returns:
            Dict[str, torch.Tensor]: 合并后的字典，包含batched张量
        """
        if not dict_list:
            return {}

        # 获取所有可能的keys
        keys = set()
        for d in dict_list:
            keys.update(d.keys())

        # 初始化结果字典
        batched_dict = {}

        # 对每个key，收集所有张量并进行batching
        for key in keys:
            # 收集所有包含该key的张量
            # check_none
            if dict_list[0][key] == None:
                batched_dict[key] = None
                continue
            tensors = []
            for d in dict_list:
                if key in d:
                    tensor = d[key]
                    # # 确保张量的第一个维度是1
                    # if tensor.shape[0] != 1:
                    #     raise ValueError(f"张量 {key} 的第一个维度不是1，实际形状为 {tensor.shape}")

                    tensors.append(tensor)

            if tensors:
                # 使用torch.cat沿着第一个维度连接张量
                batched_dict[key] = torch.cat(tensors, dim=0)

            # print(key, batched_dict[key].shape[0])

        return batched_dict



    # here
    def run_concurrent_test(self, dataloader, max_workers=4, num_samples=10, max_serve_batch_size=8,
                            min_batch_threshold=4, std_batch_threshold_decoding=2,
                            max_wait_time=5., worker_check_time=0.01, req_per_sec=1., use_compression=False,
                            comp_mode='ccm'):
        """
        middleware: 维护请求队列和服务状态
        collector_thread: 收集dataloader数据的独立线程
        dispatcher_thread: 调度任务到服务端的独立线程
        """
        fstr = f"max_workers: {max_workers}, " \
               f"num_batches: {num_samples}, " \
               f"max_serve_batch_size: {max_serve_batch_size}, " \
               f"min_batch_threshold: {min_batch_threshold}, " \
               f"max_wait_time: {max_wait_time}, " \
               f"use_compression: {use_compression}, " \
               f""
        print(fstr)
        middleware = ServiceMiddleware(max_workers, max_serve_batch_size, min_batch_threshold,
                                       std_batch_threshold_decoding, max_wait_time)

        # 启动收集线程和调度线程
        collector = Thread(target=self._collect_requests, args=(dataloader, middleware, num_samples, req_per_sec))
        dispatcher = Thread(target=self._dispatch_requests,
                            args=(middleware, num_samples, use_compression, comp_mode, worker_check_time))

        self.monitor.start_time = time.time()
        self.GPU_monitor.start_monitoring()
        collector.start()
        dispatcher.start()

        # 等待所有任务完成
        collector.join()
        dispatcher.join()
        self.GPU_monitor.stop_monitoring()
        # # 启动收集线程和调度线程
        # collector = Thread(target=self._collect_requests, args=(dataloader, middleware, num_samples, req_per_sec))
        # dispatcher = Thread(target=self._dispatch_requests,
        #                     args=(middleware, num_samples, use_compression, worker_check_time))
        #
        # self.monitor.start_time = time.time()
        # collector.start()
        # # dispatcher.start()
        #
        # # 等待所有任务完成
        # collector.join()
        # self._dispatch_requests(middleware, num_samples, use_compression, worker_check_time)
        # # self._dispatch_requests(middleware, num_samples)

        analyze_times_result, durations = self.timestamp_manager.analyze_timestamps()
        system_completion_time = middleware.system_total_time
        system_total_generated_lens = analyze_times_result['total_generated_lens']
        metrics = self.monitor.calculate_metrics(durations, system_completion_time)
        return metrics, analyze_times_result, system_completion_time, system_total_generated_lens

    def _collect_requests(self, dataloader, middleware, num_batches, req_per_sec=1.):
        """修改后的收集请求函数，将请求添加到调度器"""
        print("num_batches", num_batches)
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                print("i", i)
                break

            time.sleep(1 / req_per_sec)
            print(f'send {i}')

            timestamps = time.time()

            # 创建请求对象并添加到调度器
            request = PriorityRequest(batch=batch,
                                      request_type='prefill',
                                      arrival_time=timestamps)
            # print(request.request_id)
            self.timestamp_manager.create_request(request.request_id)
            # timestamps = time.time()
            self.timestamp_manager.record_timestamp(request.request_id, 'prefill_queueing_start', timestamps)
            middleware.scheduler.add_request(request, 'prefill')

    def _dispatch_requests(self, middleware, num_batches, use_compression, comp_mode, worker_check_time=0.01):
        """使用动态调度的调度函数"""
        middleware.start_time = time.time()
        # last_process_time = time.time()

        while not middleware.should_stop:
            current_time = time.time()

            selected_batch = None
            mode = None
            if middleware.idle_workers > 0:
                # 使用调度器选择要处理的批次
                selected_batch, mode = middleware.scheduler.select_batch()
                # print("middleware.idle_workers", middleware.idle_workers)

            if selected_batch and middleware.idle_workers > 0:
                batch_start_time = time.time()

                with middleware.idle_lock:
                    middleware.idle_workers -= 1

                if mode == 'prefill':
                    # 处理prefill请求
                    batches_tensor = self._batching_tensors([req.batch for req in selected_batch])
                    # request_ids_list = ([[].extend(req.request_id) for req in selected_batch])
                    request_ids_list = []
                    for req in selected_batch:
                        request_ids_list.extend(req.request_id)
                    # print('request_ids_list_process_prefill_wrapper', request_ids_list)
                    future = middleware.executor.submit(
                        self._process_prefill_wrapper,
                        batches_tensor,  # 提取batch数据
                        request_ids_list,
                        middleware,
                        batch_start_time,
                        use_compression,
                        comp_mode
                    )
                    future.add_done_callback(self._on_prefill_complete)
                else:
                    # 处理decoding请求

                    results_batched = self._merge_prefill_results(selected_batch)

                    # todo 时间测量层面之后再梳理
                    future = middleware.executor.submit(
                        self._process_decoding_wrapper,
                        results_batched,  # 提取batch数据
                        middleware,
                        batch_start_time,
                        use_compression,
                        comp_mode
                    )
                    future.add_done_callback(self._on_decoding_complete)

                last_process_time = current_time


                # 检查是否达到目标批次数
            if middleware.processed_decode_batches >= num_batches:
                middleware.should_stop = True
                logging.info("Reached target batch count, stopping service...")
                break

            time.sleep(worker_check_time)

        # 等待所有任务完成
        middleware.executor.shutdown(wait=True)
        total_time = time.time() - middleware.start_time
        # print(self.timestamp_manager.timestamps_map)

        # 输出服务统计信息
        # print(f"Service completed in {total_time:.2f} seconds")
        middleware.system_total_time = total_time
        logging.info(f"Service completed in {total_time:.2f} seconds")
        logging.info(f"Processed {len(middleware.batch_times)} batches")
        if middleware.batch_times:
            avg_batch_time = sum(middleware.batch_times) / len(middleware.batch_times)
            logging.info(f"Average batch processing time: {avg_batch_time:.2f} seconds")

    def _on_prefill_complete(self, future):
        """Prefill完成的回调"""
        # try:
        #     prefill_result = future.result()
        #     # 创建decoding请求并添加到调度器
        #
        #     # with middleware.idle_lock:
        #     #     middleware.idle_workers += 1
        #
        # except Exception as e:
        #     logging.error(f"Prefill processing failed: {e}")
        # finally:
        #     torch.cuda.empty_cache()
        # prefill_result = future.result()
        # torch.cuda.empty_cache()
        with torch.cuda.stream(self.cuda_stream):
            stop_time = time.time()
            torch.cuda.current_stream().synchronize()
            torch.cuda.empty_cache()
            print('stop_time', time.time()-stop_time)

    def _on_decoding_complete(self, future):
        """Decoding完成的回调"""
        # try:
        #     result = future.result()
        #     self.monitor.add_result(result)
        #     # with middleware.idle_lock:
        #     #     middleware.idle_workers += 1
        #     #     middleware.processed_batches += result.request_size
        #     #     middleware.completion_event.set()
        #
        # except Exception as e:
        #     logging.error(f"Decoding processing failed: {e}")
        # finally:
        #     torch.cuda.empty_cache()
        result = future.result()
        self.monitor.add_result(result)
        with torch.cuda.stream(self.cuda_stream):
            torch.cuda.current_stream().synchronize()
            torch.cuda.empty_cache()
        # print('1finisl')

    def _process_prefill_wrapper(self, batches, request_ids_list, middleware, batch_start_time, use_compression,
                                 comp_mode):
        """包装批处理函数，添加时间统计"""
        result = None
        # try:
        #     result = self._process_batch(batches, use_compression, comp_mode=comp_mode)  # 需要修改原始_process_batch以支持批处理
        #     process_time = time.time() - batch_start_time
        #     middleware.batch_times.append(process_time)
        # finally:
        #     with middleware.idle_lock:
        #         middleware.idle_workers += 1
        #         # print("batches", batches['input_ids'].shape)
        #         middleware.processed_batches += batches['input_ids'].shape[0]
        #         middleware.completion_event.set()
        #     return result
        print('\nstart prefill')
        result, batch_size, seq_lens, processing_time, memory_usage = self._process_prefill(batches, request_ids_list,
                                                                                            use_compression,
                                                                                            comp_mode)  # 需要修改原始_process_batch以支持批处理
        # 更新性能指标
        middleware.scheduler.update_performance({
            'stage': 'prefill',
            'batch_size': batch_size,
            'sequence_length': seq_lens,
            'processing_time': result['prefill_time'],
            'memory_usage': result['memory_usage']
        })
        process_time = time.time() - batch_start_time
        middleware.batch_times.append(process_time)
        request = PriorityRequest(batch=result,
                                  request_type='decoding',
                                  arrival_time=time.time(),
                                  request_id=result['request_id'])

        with middleware.idle_lock:
            middleware.idle_workers += 1
            # print("batches", batches['input_ids'].shape)
            middleware.processed_prefill_batches += batches['input_ids'].shape[0]
            # todo 优化空间 add_request能否外移
            # if use_compression == False or comp_mode!='ccm':
            decoding_queueing = time.time()
            if self.use_compression and self.comp_mode=='ccm':
                self.timestamp_manager.record_timestamp(request_ids_list, 'compress_queueing', decoding_queueing)
            self.timestamp_manager.record_timestamp(request_ids_list, 'decoding_queueing', decoding_queueing)
            middleware.scheduler.add_request(request, 'decoding')
            print(f"wait_prefill_count:{middleware.scheduler.prefill_count}, "
                  f"wait_decoding_count:{middleware.scheduler.decoding_count},"
                  f"finish_decoding_count:{middleware.processed_decode_batches},")
            middleware.completion_event.set()
        return result

    def _process_prefill(self, batch, request_ids_list, use_compression, comp_mode):
        """处理prefill阶段，返回KV Cache等中间结果"""
        with nvtx.annotate("prefill_processing", color="blue"):
            with torch.cuda.stream(self.cuda_stream):
                # 准备输入
                # to device
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'pixel_values': batch['pixel_values'].to(self.device)
                }
                # inputs = batch
                batch_size, seq_lens = inputs['input_ids'].shape

                with torch.no_grad(), autocast(dtype=torch.float):
                    input_forward = {
                        "input_ids": inputs["input_ids"][:, :-1],
                        "attention_mask": inputs["attention_mask"][:, :-1],
                        "pixel_values": inputs["pixel_values"]
                    }

                    if use_compression:
                        # 使用压缩模式
                        past_key_values, compression_time, compression_ratio, prefill_time = \
                            self._exec_compress(comp_mode, input_forward, request_ids_list)
                        memory_usage = self._get_kv_cache_memory(past_key_values)
                        # if comp_mode =="ccm":
                        #     prefill_finish = time.time()
                        #     self.timestamp_manager.record_timestamp(request_ids_list, 'prefill_finish', prefill_finish)
                    else:
                        prefill_start = time.time()
                        # timestamps = time.time()
                        self.timestamp_manager.record_timestamp(request_ids_list, 'prefill_start', prefill_start)
                        # 不使用压缩
                        outputs = self.model(
                            input_ids=input_forward["input_ids"],
                            attention_mask=input_forward["attention_mask"],
                            pixel_values=input_forward["pixel_values"],
                            use_cache=True,
                            return_dict=True
                        )
                        # torch.cuda.synchronize()
                        past_key_values = outputs.past_key_values
                        memory_usage = self._get_kv_cache_memory(past_key_values)
                        prefill_over = time.time()
                        prefill_time = prefill_over - prefill_start
                        self.timestamp_manager.record_timestamp(request_ids_list, 'prefill_finish', prefill_over)
                        self.timestamp_manager.record_timestamp(request_ids_list, 'compress_queueing', prefill_over)
                        self.timestamp_manager.record_timestamp(request_ids_list, 'compress_start', prefill_over)
                        self.timestamp_manager.record_timestamp(request_ids_list, 'compress_finish', prefill_over)
                        # past_key_values_split = self._split_past_kv_multi_batch(past_key_values)
                        # print(len(past_key_values_split))
                        compression_ratio = 1.
                        compression_time = 0

                    # # 准备生成输入
                    # batch_size = inputs['input_ids'].shape[0]
                    # kv_len = past_key_values[0][0].shape[2]
                    # kv_attention_mask = torch.cat([
                    #     inputs['attention_mask'],
                    #     torch.ones(
                    #         (batch_size, kv_len - inputs['attention_mask'].shape[1]),
                    #         dtype=inputs['attention_mask'].dtype,
                    #         device=self.device
                    #     )
                    # ], dim=-1)
                    #
                    # # 准备generation输入
                    # input_ids = torch.cat([kv_attention_mask, inputs['input_ids'][:, -1].unsqueeze(-1)], dim=1)
                    # attention_mask = torch.cat([kv_attention_mask, inputs['attention_mask'][:, -1].unsqueeze(-1)], dim=1)
                    # inputs = {
                    # 'input_ids': input_ids.to(self.device),
                    # 'attention_mask': attention_mask.to(self.device),
                    # 'pixel_values': None
                    # }
                    inputs['pixel_values'] = None

                    # 准备返回结果
                    prefill_result = {
                        'past_key_values': past_key_values,
                        'memory_usage': memory_usage,
                        'comp_mode': comp_mode,
                        'compression_ratio': compression_ratio,
                        'compression_time': compression_time,
                        'prefill_time': prefill_time,
                        'inputs': inputs,  # 保存原始输入供decoding阶段使用
                        'request_size': batch_size,
                        'request_id': request_ids_list
                    }
                    return prefill_result, batch_size, seq_lens, prefill_time, memory_usage

    def _process_decoding_wrapper(self, prefill_result, middleware, batch_start_time, use_compression, comp_mode):
        """包装批处理函数，添加时间统计"""
        # try:
        #     result = self._process_batch(batches, use_compression, comp_mode=comp_mode)  # 需要修改原始_process_batch以支持批处理
        #     process_time = time.time() - batch_start_time
        #     middleware.batch_times.append(process_time)
        # finally:
        #     with middleware.idle_lock:
        #         middleware.idle_workers += 1
        #         # print("batches", batches['input_ids'].shape)
        #         middleware.processed_batches += batches['input_ids'].shape[0]
        #         middleware.completion_event.set()
        #     return result
        print("decoding start")
        print("batching kv shape", prefill_result["past_key_values"][0][0].shape)
        decode_start = time.time()
        self.timestamp_manager.record_timestamp(prefill_result["request_id"], 'decode_start', decode_start)
        result, batch_size, valid_lens, generated_lens, valid_lens_list, decoding_time, throughput = \
                        self._process_decoding(prefill_result, use_compression, comp_mode)
        decoding_finish = time.time()
        self.timestamp_manager.record_timestamp(prefill_result["request_id"], 'decoding_finish', decoding_finish)
        for i_req, item_req in enumerate(prefill_result["request_id"]):
            self.timestamp_manager.record_timestamp([prefill_result["request_id"][i_req]], 'generated_lens', valid_lens_list[i_req])
        # 需要修改原始_process_batch以支持批处理
        # 更新性能指标
        middleware.scheduler.update_performance({
            'stage': 'decoding',
            'batch_size': batch_size,
            'tokens_generated': valid_lens,
            'processing_time': decoding_time,
            'throughput': throughput,

        })
        process_time = time.time() - batch_start_time
        middleware.batch_times.append(process_time)
        with middleware.idle_lock:
            middleware.idle_workers += 1
            # print("batches", batches['input_ids'].shape)
            middleware.processed_decode_batches += batch_size
            # print('middleware.processed_decode_batches', middleware.processed_decode_batches)
            print(f"完成进度: {middleware.processed_decode_batches}/{self.num_samples}")
            middleware.completion_event.set()
        return result

    def _process_decoding(self, prefill_result, use_compression, comp_mode):
        """处理decoding阶段，生成文本并返回结果"""
        with torch.cuda.stream(self.cuda_stream):
            # 从prefill结果中提取必要信息
            past_key_values = prefill_result['past_key_values']
            inputs = prefill_result['inputs']

            # 准备生成输入
            batch_size = inputs['input_ids'].shape[0]
            kv_len = past_key_values[0][0].shape[2]

            # # 构建attention mask
            # kv_attention_mask = torch.cat([
            #     inputs['attention_mask'],
            #     torch.ones(
            #         (batch_size, kv_len - inputs['attention_mask'].shape[1]),
            #         dtype=inputs['attention_mask'].dtype,
            #         device=self.device
            #     )
            # ], dim=-1)
            if use_compression:
                if comp_mode == "ccm":
                    padding_lens, valid_lens, comp_padding_lens = self.get_comp_padding_valid_lengths(
                        inputs['attention_mask'],
                        compress_ratio=self.compression_ratio_list[0])
                else:
                    padding_lens, valid_lens, comp_padding_lens = self.get_comp_padding_valid_lengths(
                        inputs['attention_mask'],
                        compress_ratio=self.compression_ratio_list[1])
            else:
                padding_lens, valid_lens, comp_padding_lens = self.get_comp_padding_valid_lengths(
                    inputs['attention_mask'],
                    compress_ratio=1.0)

            kv_attention_mask = torch.ones(
                (batch_size, kv_len),
                dtype=inputs['attention_mask'].dtype,
                device=inputs['attention_mask'].device
            )
            kv_attention_mask = self.create_attention_mask_from_padding(comp_padding_lens, kv_attention_mask)

            # 准备generation输入
            input_ids = torch.cat([
                kv_attention_mask,
                inputs['input_ids'][:, -1].unsqueeze(-1)
            ], dim=1)
            attention_mask = torch.cat([
                kv_attention_mask,
                inputs['attention_mask'][:, -1].unsqueeze(-1)
            ], dim=1)

            # 开始生成
            generation_start = time.time()

            gen_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

            # 处理生成结果
            sequences = gen_outputs.sequences
            start_idx = int(kv_attention_mask.shape[1])
            generated_sequences = sequences[:, start_idx:]
            generated_lens = int(generated_sequences.shape[1])
            # print("generated_sequences.shape", generated_sequences.shape)

            decoding_time = time.time() - generation_start
            valid_lens, valid_lens_list = self.find_valid_lens(generated_sequences,
                            self.processor.tokenizer.eos_token_id)
            # set valid id
            print("valid_lens", valid_lens, valid_lens_list)
            throughput = valid_lens / decoding_time if decoding_time > 0 else 0

            # 构建返回结果
            result = TestResult(
                method='comp' if prefill_result['compression_ratio'] else 'orig',
                latency=[0, prefill_result['prefill_time'], decoding_time / valid_lens],
                success=True,
                request_id=prefill_result["request_id"],
                memory_usage=prefill_result['memory_usage'],
                request_size=prefill_result['request_size'],
                throughput=throughput,
                valid_tokens=valid_lens,
                compression_ratio=prefill_result['compression_ratio'],
                compression_time=prefill_result['compression_time']
            )

            # 清理显存
            torch.cuda.empty_cache()

            return result, batch_size, valid_lens, generated_lens, valid_lens_list, decoding_time, throughput

    def merge_kv_cache(self, kv_cache_list):
        """
        合并多个transformer格式的KV cache列表
        Args:
            kv_cache_list: 列表的列表，每个元素是一个层的(key, value)元组的列表
                          格式为: [[(k1,v1), (k2,v2),...], [(k1,v1), (k2,v2),...], ...]
                          其中k,v是tensor，shape通常为[batch_size, num_heads, seq_len, head_dim]
        Returns:
            merged_kv_cache: 合并后的KV cache列表
        """
        if not kv_cache_list:
            return []

        # 获取第一个cache的结构信息
        num_layers = len(kv_cache_list[0])

        # 初始化结果列表
        merged_cache = []

        # 对每一层进行合并
        for layer_idx in range(num_layers):
            # 收集所有batch的当前层的key和value
            keys = []
            values = []

            # 从每个batch中获取当前层的kv pair
            for batch_cache in kv_cache_list:
                key, value = batch_cache[layer_idx]
                keys.append(key)
                values.append(value)

            # 沿batch维度拼接
            merged_key = torch.cat(keys, dim=0).contiguous()  # dim=0 是batch维度
            merged_value = torch.cat(values, dim=0).contiguous()



            # 将合并后的key-value对添加到结果中
            merged_cache.append((merged_key, merged_value))

        # print(merged_cache[0][0].shape)

        return merged_cache

    def _merge_prefill_results(self, requests_list):
        if not requests_list:
            return {}

        merged_result = {
            'past_key_values': [],  # 按顺序连接
            'memory_usage': 0,  # 取最大值
            'compression_ratio': 0,  # 加权平均
            'compression_time': 0,  # 累加
            'prefill_time': 0,  # 累加
            'inputs': [],  # 按顺序连接
            'request_id': [],
            'request_size': 0  # 累加
        }
        past_key_values_list = []
        input_list = []
        request_ids_list = []

        total_batch_size = 0  # 用于计算加权平均

        for request in requests_list:
            request_ids_list.extend(request.request_id)
            result = request.batch
            # past_key_values 连接
            past_key_values_list.append(result['past_key_values'])

            # memory_usage 取最大值
            merged_result['memory_usage'] += result['memory_usage']

            # compression_ratio 加权平均
            batch_size = result['request_size']
            total_batch_size += batch_size

            merged_result['compression_ratio'] += (
                    result['compression_ratio'] * batch_size
            )

            # 时间相关指标累加
            merged_result['compression_time'] += result['compression_time']
            # todo 不管指标测量
            merged_result['prefill_time'] += result['prefill_time']

            # inputs 连接
            input_list.append(result['inputs'])

            # request_size 累加
            merged_result['request_size'] += result['request_size']
            print("batching decoding", merged_result['request_size'])

        # 完成 compression_ratio 的加权平均计算
        if total_batch_size > 0:
            merged_result['compression_ratio'] /= total_batch_size
            merged_result['inputs'] = self._batching_tensors(input_list)
            if self.use_compression and self.comp_mode == 'ccm':
                past_key_values_list, comp_duration_time, final_compression_ratio = \
                    self._integrated_compress_ccm(past_key_values_list, merged_result['inputs'], request_ids_list)
                merged_result['compression_time'] = comp_duration_time
                merged_result['compression_ratio'] = final_compression_ratio
            merged_result['past_key_values'] = self.merge_kv_cache(past_key_values_list)
            merged_result['request_id'] = request_ids_list
            print("merged_result_decoding", merged_result['request_id'])

        return merged_result

    def get_comp_padding_valid_lengths(self, attention_mask: torch.Tensor, compress_ratio=1.0):
        """
        计算注意力掩码中每个 batch 的有效位数量，支持左填充和右填充

        Args:
            attention_mask: 形状为 [batch_size, seq_lens] 的注意力掩码张量
                           1 表示有效位置，0 表示被掩码的位置
            left_padded: 是否为左侧填充（默认为 True）
                        - True: 0在左边，有效token在右边
                        - False: 有效token在左边，0在右边

        Returns:
            torch.Tensor: 形状为 [batch_size] 的张量，表示每个 batch 中有效位的数量
        """
        assert len(attention_mask.shape) == 2, \
            f"AttentionMask should be 2D tensor, got shape {attention_mask.shape}"
        valid_lens = attention_mask.sum(dim=1).long()
        padding_lens = attention_mask.shape[1] - valid_lens
        comp_padding_lens = (padding_lens / compress_ratio).int()
        return padding_lens, valid_lens, comp_padding_lens

    def create_attention_mask_from_padding(self, padding_lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        根据 padding 长度创建 attention mask

        Args:
            padding_lengths: 形状为 [batch_size] 的张量，表示每个序列的 padding 长度
            seq_length: 序列总长度，如果为 None，则使用最大 padding 长度加上最长有效长度

        Returns:
            torch.Tensor: 形状为 [batch_size, seq_length] 的 attention mask

        Example:
            tensor([[0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 1, 1, 1, 1]])
        """
        batch_size = padding_lengths.size(0)

        # # 如果没有指定序列长度，则计算所需的最小长度
        # if seq_length is None:
        #     max_valid_length = max(torch.arange(batch_size).size(0) - padding_lengths)
        #     seq_length = max_valid_length + padding_lengths.max()
        #
        # # 创建全 1 tensor
        # mask = torch.ones((batch_size, seq_length), device=padding_lengths.device)

        # 为每个序列的左侧填充位置设置 0
        for i in range(batch_size):
            mask[i, :padding_lengths[i]] = 0

        return mask

    def _calculate_performance_metrics(self, metrics_window):
        """计算详细的性能指标"""
        if not metrics_window:
            return {}

        # 分别统计prefill和decoding阶段的指标
        prefill_metrics = [m for m in metrics_window if m['stage'] == 'prefill']
        decoding_metrics = [m for m in metrics_window if m['stage'] == 'decoding']

        metrics = {}

        # Prefill阶段指标
        if prefill_metrics:
            metrics['prefill'] = {
                'avg_time': np.mean([m['processing_time'] for m in prefill_metrics]),
                'avg_batch_size': np.mean([m['batch_size'] for m in prefill_metrics]),
                'avg_sequence_length': np.mean([m['sequence_length'] for m in prefill_metrics]),
                'avg_memory': np.mean([m['memory_usage'] for m in prefill_metrics])
            }

        # Decoding阶段指标
        if decoding_metrics:
            metrics['decoding'] = {
                'avg_time': np.mean([m['processing_time'] for m in decoding_metrics]),
                'avg_throughput': np.mean([m['throughput'] for m in decoding_metrics]),
                'avg_tokens': np.mean([m['tokens_generated'] for m in decoding_metrics])
            }

        # 整体系统指标
        metrics['system'] = {
            'prefill_ratio': len(prefill_metrics) / len(metrics_window),
            'avg_batch_size': np.mean([m['batch_size'] for m in metrics_window]),
            'total_tokens': sum(m.get('tokens_generated', 0) for m in decoding_metrics)
        }

        return metrics

    # def _process_batch_wrapper(self, batches, middleware, batch_start_time, use_compression, comp_mode):
    #     """包装批处理函数，添加时间统计"""
    #     result = None
    #     # try:
    #     #     result = self._process_batch(batches, use_compression, comp_mode=comp_mode)  # 需要修改原始_process_batch以支持批处理
    #     #     process_time = time.time() - batch_start_time
    #     #     middleware.batch_times.append(process_time)
    #     # finally:
    #     #     with middleware.idle_lock:
    #     #         middleware.idle_workers += 1
    #     #         # print("batches", batches['input_ids'].shape)
    #     #         middleware.processed_batches += batches['input_ids'].shape[0]
    #     #         middleware.completion_event.set()
    #     #     return result
    #     result = self._process_batch(batches, use_compression, comp_mode=comp_mode)  # 需要修改原始_process_batch以支持批处理
    #     process_time = time.time() - batch_start_time
    #     middleware.batch_times.append(process_time)
    #     with middleware.idle_lock:
    #         middleware.idle_workers += 1
    #         # print("batches", batches['input_ids'].shape)
    #         middleware.processed_batches += batches['input_ids'].shape[0]
    #         middleware.completion_event.set()
    #     return result


def load_checkpoint_only_model(model, filename, device=None):
    """加载检查点"""
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        if device is None:
            device = next(model.parameters()).device
        checkpoint = torch.load(filename, map_location=device)

        print("Available keys in checkpoint:", checkpoint.keys())
        print("Available keys in model_state_dict:", checkpoint['model_state_dict'].keys())

        try:
            model.load_state_dict(checkpoint['model_state_dict']['compressor'])
            print(f"Successfully loaded checkpoint '{filename}'")
            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', float('inf'))
        except Exception as e:
            print(f"Error in strict loading: {str(e)}")
            print("Attempting non-strict load...")
            try:
                if 'compressor' in checkpoint['model_state_dict']:
                    model.load_state_dict(checkpoint['model_state_dict']['compressor'], strict=False)
                else:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("Successfully loaded checkpoint with non-strict mode")
                epoch = checkpoint.get('epoch', 0)
                loss = checkpoint.get('loss', float('inf'))
            except Exception as e2:
                print(f"Error in non-strict loading: {str(e2)}")
                print("Using model without loading checkpoint")
                epoch = 0
                loss = float('inf')

        return epoch, loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, float('inf')


# def parse_string_with_image(text, special_token="<image>"):
#     """解析包含图像标记的文本"""
#     is_exist_flag = 0
#     if special_token in text:
#         remaining_text = text.replace(special_token, '')
#         remaining_text = remaining_text.strip()
#         is_exist_flag = 1
#     else:
#         remaining_text = text
#     return remaining_text, is_exist_flag

# def load_image_data(json_path: str, imgsets_path: str, num_samples: int, mode=1) -> List[str]:
#     """加载图像数据"""
#     target_datasets = ['gqa']
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     data = [d for d in data if 'image' in d.keys() and d['image'].split('/')[0] in target_datasets]
#     random.shuffle(data)
#
#     if num_samples < 0:
#         target_data = data
#     else:
#         target_data = data[:num_samples]
#
#     target_texts_image = []
#     for d in target_data:
#         if d['conversations'] and d['conversations'][0]["from"] == "human":
#             questions_list = []
#             answers_list = []
#             for d_sentence in d['conversations']:
#                 if d_sentence['from'] == 'human':
#                     questions_list.append((
#                         f"USER: <image> \nWhat's the content of the image? I am sick. "
#                         f"Please describe every detail in the image with specific and detailed language. ASSISTANT:",
#                         Image.open(f"{imgsets_path}{d['image']}").convert("RGB")
#                     ))
#                     answers_list.append(1)
#                 else:
#                     answers_list.append(d_sentence['value'])
#             target_texts_image.append([questions_list, answers_list])
#
#     return target_texts_image
def load_image_data(json_path: str, imgsets_path: str, num_samples: int, mode=1) -> List[str]:
    """加载图像数据"""
    target_datasets = ['gqa']
    with open(json_path, 'r') as f:
        data = json.load(f)
    data = [d for d in data if 'image' in d.keys() and d['image'].split('/')[0] in target_datasets]
    random.shuffle(data)

    if num_samples < 0:
        target_data = data
    else:
        target_data = data[:num_samples]

    target_texts_image = []
    for d in target_data:
        if d['conversations'] and d['conversations'][0]["from"] == "human":
            questions_list = []
            answers_list = []
            for d_sentence in d['conversations']:
                if d_sentence['from'] == 'human':
                    questions_list.append((
                        f"USER: <image> \nWhat's the content of the image? I am sick. "
                        f"Please describe every detail in the image with specific and detailed language. ASSISTANT:",
                        Image.open(f"{imgsets_path}{d['image']}").convert("RGB")
                    ))
                    answers_list.append(1)
                else:
                    answers_list.append(d_sentence['value'])
            target_texts_image.append([questions_list, answers_list])

    return target_texts_image


def print_metrics(metrics: Dict[str, Dict[str, float]]):
    """打印性能指标"""
    print("\n=== Performance Test Results ===")
    for method in ['orig', 'comp']:
        print(f"\n--- {method.upper()} Method ---")
        if not metrics.get(method):
            print("No data available.")
            continue

        print(f"Total Requests: {metrics[method]['total_requests']:.0f}")
        print(f"Success Rate: {metrics[method]['success_rate']:.2f}%")
        print(f"Requests per Second: {metrics[method]['requests_per_second']:.2f}")

        print("\n=== ttft Metrics (seconds) ===")
        print(f"Average: {metrics[method]['avg_ttft']:.3f}")
        print(f"Median (P50): {metrics[method]['median_ttft']:.3f}")
        print(f"P90: {metrics[method]['p90_ttft']:.3f}")
        print(f"P99: {metrics[method]['p99_ttft']:.3f}")
        print(f"Min: {metrics[method]['min_ttft']:.3f}")
        print(f"Max: {metrics[method]['max_ttft']:.3f}")

        print("\n=== tpot Metrics (seconds) ===")
        print(f"Average: {metrics[method]['avg_tpot']:.3f}")
        print(f"Median (P50): {metrics[method]['median_tpot']:.3f}")
        print(f"P90: {metrics[method]['p90_tpot']:.3f}")
        print(f"P99: {metrics[method]['p99_tpot']:.3f}")
        print(f"Min: {metrics[method]['min_tpot']:.3f}")
        print(f"Max: {metrics[method]['max_tpot']:.3f}")

        print("\n=== Resource Usage ===")
        print(f"Average Memory Usage: {metrics[method]['avg_memory_usage']:.2f} MB")
        print(f"Average Throughput: {metrics[method]['avg_throughput']:.2f} tokens/s")
        print(f"Throughput StdDev: {metrics[method]['throughput_stddev']:.2f} tokens/s")

        if method == 'comp':
            print(f"Average Compression Ratio: {metrics[method].get('avg_compression_ratio', 0):.2f}")
            print(f"Average Compression Time: {metrics[method].get('avg_compression_time', 0):.3f} seconds")

    print("\n=== Summary ===")
    if 'orig' in metrics and 'comp' in metrics:
        latency_reduction = metrics['orig']['avg_latency'] - metrics['comp']['avg_latency']
        memory_reduction = metrics['orig']['avg_memory_usage'] - metrics['comp']['avg_memory_usage']
        throughput_increase = metrics['comp']['avg_throughput'] - metrics['orig']['avg_throughput']

        # 安全计算吞吐量改进百分比
        if metrics['orig']['avg_throughput'] > 0:
            throughput_improvement = (metrics['comp']['avg_throughput'] / metrics['orig']['avg_throughput'] * 100) - 100
        else:
            throughput_improvement = 0.0 if metrics['comp']['avg_throughput'] == 0 else float('inf')

        print(f"Latency Reduction: {latency_reduction:.3f} seconds")
        print(f"Memory Reduction: {memory_reduction:.2f} MB")
        print(f"Throughput Increase: {throughput_increase:.2f} tokens/s")
        print(f"Throughput Improvement: {throughput_improvement:.2f}%")


def main():
    # 配置参数
    model_path = "/home/zhujianian/workspace/Uneed/huggingface_download/llava-1.5-7b-hf"
    ckpt_path = "/home/zhujianian/cvpr/ckpt_store/best_finetune_mlp_1030_mm_8.pth"
    json_path = '/home/zhujianian/workspace/Uneed/huggingface_download/LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
    imgsets_path = '/home/zhujianian/cvpr/datasets/'
    output_path = f"logs/compression_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # 基础参数配置
    seed = 48
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 测试参数配置
    num_samples =30  # 测试样本数
    batch_size = 1  # 批处理大小
    max_input_tokens = 128  # 最大输入长度
    max_new_tokens = 512
    # compression_factor = 5  # 压缩率
    compression_ratio_list = [5., 2.]  # 压缩率
    max_workers = 1  # 并发数

    # use_compression = False
    use_compression = True
    comp_mode = 'ccm'
    # comp_mode = 'Knorm'
    # comp_mode = 'StreamingLLM'
    # comp_mode = 'RandomPress'
    # comp_mode = 'SnapKV'
    # comp_mode = 'ExpectedAttention'
    # comp_mode = 'Quantized'

    # num_batches = 16    # 测试批次数

    # req_per_sec = 20.
    # max_serve_batch_size = 10 # 测试批次数
    # min_batch_threshold = 10
    # max_wait_time = 5
    req_per_sec = 10.
    std_batch_threshold_prefill = 10
    compress_batch = 1
    max_serve_batch_size = std_batch_threshold_prefill  # 测试批次数
    min_batch_threshold = std_batch_threshold_prefill
    std_batch_threshold_decoding = int(20 / std_batch_threshold_prefill)
    max_wait_time = 5
    worker_check_time = 0.5

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型和处理器
    print("Loading model and processor...")
    model = LlavaForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = LlavaProcessor.from_pretrained(model_path)

    # 设置处理器参数
    vision_config = model.config.vision_config
    processor.image_processor.size = {"height": vision_config.image_size, "width": vision_config.image_size}
    processor.image_processor.patch_size = vision_config.patch_size

    # 初始化压缩器
    print("Initializing compressor...")
    compressor = KVCacheLinearDecoupleCompressor(
        src_config=model.config,
        compression_factor=int(compression_ratio_list[0]),
        min_seq_len=2,
    ).to(device)

    # 加载压缩器权重
    print("Loading compressor checkpoint...")
    epoch, best_val_loss = load_checkpoint_only_model(compressor, ckpt_path, device)

    # 加载数据
    print("Loading test data...")
    data = load_image_data(json_path, imgsets_path, num_samples)
    test_data = CustomImageTextDataset(data, processor, max_length=max_input_tokens)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # 设置评估模式
    model.eval()
    compressor.eval()

    # 创建测试器
    print("Creating tester...")
    args_ImprovedConcurrentTester = [
        model, processor, compressor, max_new_tokens, num_samples, compress_batch,
        compression_ratio_list, use_compression, comp_mode, device
    ]
    tester = ImprovedConcurrentTester(*args_ImprovedConcurrentTester)

    # 运行并发测试
    print("\nStarting concurrent testing...")
    with autocast(dtype=torch.float):
        metrics, analyze_times_result, \
        system_completion_time, system_total_generated_lens\
            = tester.run_concurrent_test(
            test_dataloader,
            max_workers=max_workers,
            num_samples=num_samples,
            max_serve_batch_size=max_serve_batch_size,
            min_batch_threshold=min_batch_threshold,
            std_batch_threshold_decoding=std_batch_threshold_decoding,
            max_wait_time=max_wait_time,
            worker_check_time=worker_check_time,
            req_per_sec=req_per_sec,
            use_compression=use_compression,
            comp_mode=comp_mode
        )

    # 打印结果
    print(comp_mode)
    print_metrics(metrics)
    print(analyze_times_result)
    print(f"Service completed in {system_completion_time:.2f} seconds.")
    print(f"Service generated {system_total_generated_lens} tokens.")
    print(f"Thoughtput: {system_total_generated_lens/(system_completion_time+1e-5):.2f} tokens/s.")

    # 保存结果
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()