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
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from utils_ccm.module_ccm import KVCacheLinearDecoupleCompressor
from utils_ccm.utils_compress import *
from utils_ccm.utils_schedule import *

"""
v1.0
最基本的全流程静态并发系统。测量时长无bug。

v1.1 *
实现pd双队列管理，测试性能。
于2024.12.26日23点完成。



"""



@dataclass
class TestResult:
    method: str  # 'orig' 或 'comp'
    latency: List[float]
    success: bool
    valid_tokens: int = 0
    memory_usage: float = 0.0
    request_size: int = 0
    throughput: float = 0.0
    compression_ratio: Optional[float] = None
    compression_time: Optional[float] = None


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
                'request_size': [],
                'success_count': 0,
                'valid_tokens': 0,
                'total_count': 0
            },
            'comp': {
                'latencies': [],
                'throughputs': [],
                'memory_usage': [],
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
        method = result.method
        self.results[method]['latencies'].append(result.latency)
        self.results[method]['throughputs'].append(result.throughput)
        self.results[method]['memory_usage'].append(result.memory_usage)
        self.results[method]['request_size'].append(result.request_size)
        self.results[method]['valid_tokens'] += int(result.valid_tokens)
        self.results[method]['success_count'] += int(result.success)
        self.results[method]['total_count'] += 1
        print(result)
        
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
                ttft_list += [data['latencies'][i][1]]*data['request_size'][i]
                tpot_list += [data['latencies'][i][2]]*data['request_size'][i]
        return ttft_list, tpot_list

    def calculate_metrics(self):
        metrics = {}
        duration = time.time() - self.start_time

        for method, data in self.results.items():
            if not data['latencies']:
                continue
            ttft_list, tpot_list = self.proc_latencies()

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
    def __init__(self, max_workers, max_serve_batch_size, min_batch_threshold, max_wait_time):
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
        self.batch_times = []
        self.completion_event = Event()
        self.should_stop = False
        self.start_time = None

        # 新增: 初始化动态调度器
        self.scheduler = DynamicScheduler(
            max_batch_size=max_serve_batch_size,
            min_batch_size=min_batch_threshold,
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
    def __init__(self, model, processor, compressor, device):
        self.model = model
        self.processor = processor
        self.compressor = compressor
        self.device = device
        self.timeCheck = TimeTik()
        self.monitor = PerformanceMonitor()
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

    def find_valid_lens(self, tensor, token_id=2):
        """计算有效序列长度"""
        matches = tensor == token_id
        positions = torch.argmax(matches.int(), dim=1, keepdim=True)
        no_token_mask = ~matches.any(dim=1, keepdim=True)
        positions = positions.masked_fill(no_token_mask, tensor.shape[1]-1)
        return positions.sum().item()

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
            comp_len[i_idx][2] = i_input[image_insert_pos[i_idx]+1:].shape[-1]
            comp_len[i_idx][1] = kv_len - comp_len[i_idx][2] - comp_len[i_idx][0]
        return comp_len

    def run_compressor(self, compressor, past_key_values, attention_mask, comp_len):
        """执行KV-Cache压缩"""
        it_len = comp_len[0][1:]  # 取出图像和文本长度
        compressed_past_key_values = compressor(past_key_values, it_len)
        return compressed_past_key_values

    def _exec_compress(self, comp_mode, inputs):
        comp_past_key_values = None
        comp_duration_time = 0
        final_compression_ratio = 0
        prefill_time = 0

        if comp_mode == 'ccm':
            prefill_start = time.time()
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                use_cache=True,
                return_dict=True
            )
            torch.cuda.synchronize()
            prefill_mid_ = time.time()
            prefill_time =  prefill_mid_- prefill_start

            # 获取原始KV cache并计算内存占用
            orig_past_key_values = outputs.past_key_values
            orig_memory = self._get_kv_cache_memory(orig_past_key_values)
            kv_len = orig_past_key_values[0][0].shape[2]
            comp_len = self.get_comp_len(inputs["input_ids"], inputs["attention_mask"], kv_len)

            comp_start_time = time.time()
            comp_past_key_values = self.run_compressor(self.compressor,
                                        orig_past_key_values, inputs["attention_mask"], comp_len)
            # print(self.compressor.)
            memory_usage = self._get_kv_cache_memory(comp_past_key_values)
            comp_duration_time = time.time() - comp_start_time
            final_compression_ratio = orig_memory / memory_usage if memory_usage > 0 else 0.0
        elif comp_mode in self.press_type: # other comp
            compression_ratio = 0.5
            # prefill_start = time.time()
            press = press_select(comp_mode, compression_ratio)
            print(press)
            underlying_model = get_underlying_model(self.model)
            prefill_start = time.time()
            comp_past_key_values = exec_compression_methods(self.model,
                                          underlying_model, inputs, press,
                                          comp_mode)
            comp_duration_time = press.totalTime
            final_compression_ratio = 1/(1-compression_ratio)
            prefill_fini_ = time.time()
            prefill_time = prefill_fini_ - prefill_start
            # 0.63 0.047 存在很大的系统开销
            print("compress_time", prefill_time, comp_duration_time)

        else:
            print(f"Warn in _exec_compress: {comp_mode} matching failed.")
        self.timeCheck.show_and_reset()

        return comp_past_key_values, comp_duration_time, final_compression_ratio, prefill_time

    def _process_batch(self, batch, use_compression=False, comp_mode='ccm'):
        # server
        """处理单个批次"""
        # try:
        with torch.cuda.stream(torch.cuda.Stream()):
            # 将输入移到设备上
            inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'pixel_values': batch['pixel_values'].to(self.device)
            }

            with torch.no_grad(), autocast(dtype=torch.float):
                # Forward pass 阶段
                input_ids_forward = inputs["input_ids"][:, :-1]
                request_size = inputs["input_ids"].shape[0]
                # outputs = self.model(
                #     input_ids=input_ids_forward,
                #     attention_mask=inputs["attention_mask"][:, :-1],
                #     pixel_values=inputs["pixel_values"],
                #     use_cache=True,
                #     return_dict=True
                # )
                #
                # # 获取原始KV cache并计算内存占用
                # orig_past_key_values = outputs.past_key_values
                # orig_memory = self._get_kv_cache_memory(orig_past_key_values)
                input_forward = {
                    "input_ids": inputs["input_ids"][:, :-1],
                    "attention_mask": inputs["attention_mask"][:, :-1],
                    "pixel_values": inputs["pixel_values"]
                }
                # 压缩阶段（如果启用）
                if use_compression:
                    past_key_values, compression_time, compression_ratio, prefill_time = self._exec_compress(comp_mode, input_forward)
                    memory_usage = self._get_kv_cache_memory(past_key_values)
                    # compression_ratio = orig_memory / memory_usage if memory_usage > 0 else 0.0
                else:
                    prefill_start = time.time()
                    outputs = self.model(
                        input_ids=input_ids_forward,
                        attention_mask=inputs["attention_mask"][:, :-1],
                        pixel_values=inputs["pixel_values"],
                        use_cache=True,
                        return_dict=True
                    )
                    torch.cuda.synchronize()
                    prefill_finish_ = time.time()
                    prefill_time = prefill_finish_ - prefill_start
                    # 获取原始KV cache并计算内存占用
                    orig_past_key_values = outputs.past_key_values
                    orig_memory = self._get_kv_cache_memory(orig_past_key_values)
                    past_key_values = orig_past_key_values
                    memory_usage = orig_memory
                    compression_ratio = None
                    compression_time = 0
                # prefill_time = time.time() - prefill_start

                # 生成阶段
                generation_start = time.time()

                # 准备生成输入
                batch_size = inputs['input_ids'].shape[0]
                kv_len = past_key_values[0][0].shape[2]
                kv_attention_mask = torch.cat([
                    inputs['attention_mask'],
                    torch.ones(
                        (batch_size, kv_len - inputs['attention_mask'].shape[1]),
                        dtype=inputs['attention_mask'].dtype,
                        device=self.device
                    )
                ], dim=-1)

                # 准备generation输入
                input_ids = torch.cat([kv_attention_mask, inputs['input_ids'][:, -1].unsqueeze(-1)], dim=1)
                attention_mask = torch.cat([kv_attention_mask, inputs['attention_mask'][:, -1].unsqueeze(-1)], dim=1)

                # 生成文本
                gen_outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    max_new_tokens=512 - inputs['input_ids'].shape[1],
                    do_sample=True,
                    temperature=0.7,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                # 处理生成的序列并计算性能指标
                sequences = gen_outputs.sequences
                start_idx = int(kv_attention_mask.shape[1])
                generated_sequences = sequences[:, start_idx:]

                decoding_time = time.time() - generation_start

                #todo waiting time

                # todo valid_lens
                valid_lens = self.find_valid_lens(generated_sequences, self.processor.tokenizer.eos_token_id)
                # valid_lens = generated_sequences.numel()
                throughput = valid_lens / decoding_time if decoding_time > 0 else 0

                tokens_time = decoding_time / valid_lens
                latency = [0, prefill_time, tokens_time]

                # 清理GPU内存
                torch.cuda.empty_cache()

                return TestResult(
                    method='comp' if use_compression else 'orig',
                    latency=latency,  # 只使用生成时间作为延迟
                    success=True,
                    memory_usage=memory_usage,
                    request_size=request_size,
                    throughput=throughput,
                    valid_tokens=valid_lens,
                    compression_ratio=compression_ratio,
                    compression_time=compression_time,

                )

        # except Exception as e:
        #     print(f"Error in batch processing: {str(e)}")
        #     return TestResult(
        #         method='comp' if use_compression else 'orig',
        #         latency=0,
        #         success=False,
        #         memory_usage=0,
        #         request_size=0,
        #         compression_time=0,
        #         throughput=0,
        #         valid_tokens=0,
        #         compression_ratio=None,
        #         compression_time=None
        #     )

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
            merged_key = torch.cat(keys, dim=0)  # dim=0 是batch维度
            merged_value = torch.cat(values, dim=0)

            # 将合并后的key-value对添加到结果中
            merged_cache.append((merged_key, merged_value))

        print(merged_cache[0][0].shape)

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
            'request_size': 0  # 累加
        }
        past_key_values_list = []
        input_list = []

        total_batch_size = 0  # 用于计算加权平均

        for request in requests_list:
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
            print(merged_result['request_size'])

        # 完成 compression_ratio 的加权平均计算
        if total_batch_size > 0:
            merged_result['compression_ratio'] /= total_batch_size
            merged_result['inputs'] = self._batching_tensors(input_list)
            merged_result['past_key_values'] = self.merge_kv_cache(past_key_values_list)

        return merged_result

    def run_concurrent_test(self, dataloader, max_workers=4, num_samples=10, max_serve_batch_size=8,
                            min_batch_threshold=4, max_wait_time=5., worker_check_time=0.01, req_per_sec=1., use_compression=False):
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
        middleware = ServiceMiddleware(max_workers, max_serve_batch_size, min_batch_threshold, max_wait_time)

        # 启动收集线程和调度线程
        collector = Thread(target=self._collect_requests, args=(dataloader, middleware, num_samples, req_per_sec))
        dispatcher = Thread(target=self._dispatch_requests, args=(middleware, num_samples, use_compression, worker_check_time))

        self.monitor.start_time = time.time()
        collector.start()
        dispatcher.start()

        # 等待所有任务完成
        collector.join()
        dispatcher.join()
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
        return self.monitor.calculate_metrics()

    def _collect_requests(self, dataloader, middleware, num_batches, req_per_sec=1.):
        """修改后的收集请求函数，将请求添加到调度器"""
        print("num_batches", num_batches)
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                print("i", i)
                break

            time.sleep(1 / req_per_sec)
            print(f'send {i}')

            # 创建请求对象并添加到调度器
            request = PriorityRequest(batch=batch,
                                      request_type='prefill',
                                      arrival_time=time.time())
            middleware.scheduler.add_request(request, 'prefill')

    def _dispatch_requests(self, middleware, num_batches, use_compression, worker_check_time=0.01):
        """使用动态调度的调度函数"""
        middleware.start_time = time.time()
        last_process_time = time.time()

        while not middleware.should_stop:
            current_time = time.time()

            selected_batch = None
            mode = None
            if middleware.idle_workers > 0:
                # 使用调度器选择要处理的批次
                selected_batch, mode = middleware.scheduler.select_batch()
                print()
                print("middleware.idle_workers", middleware.idle_workers)

            if selected_batch and middleware.idle_workers > 0:
                batch_start_time = time.time()

                with middleware.idle_lock:
                    middleware.idle_workers -= 1

                if mode == 'prefill':
                    # 处理prefill请求
                    comp_mode = 'ccm'
                    batches_tensor = self._batching_tensors([req.batch for req in selected_batch])
                    future = middleware.executor.submit(
                        self._process_prefill_wrapper,
                        batches_tensor,  # 提取batch数据
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
                        batch_start_time
                    )
                    future.add_done_callback(self._on_decoding_complete)

                last_process_time = current_time
                print(middleware.processed_decode_batches, num_batches)

                # 检查是否达到目标批次数
            if middleware.processed_decode_batches >= num_batches:
                middleware.should_stop = True
                logging.info("Reached target batch count, stopping service...")
                break

            time.sleep(worker_check_time)

        # 等待所有任务完成
        middleware.executor.shutdown(wait=True)
        total_time = time.time() - middleware.start_time

        # 输出服务统计信息
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
        torch.cuda.empty_cache()

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
        torch.cuda.empty_cache()
        print('1finisl')

    def _process_prefill_wrapper(self, batches, middleware, batch_start_time, use_compression, comp_mode):
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
        result, batch_size, seq_lens, processing_time, memory_usage = self._process_prefill(batches,
                                                                        use_compression, comp_mode)  # 需要修改原始_process_batch以支持批处理
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
                                  arrival_time=time.time())

        with middleware.idle_lock:
            middleware.idle_workers += 1
            # print("batches", batches['input_ids'].shape)
            middleware.processed_prefill_batches += batches['input_ids'].shape[0]
            # todo 优化空间 add_request能否外移
            middleware.scheduler.add_request(request, 'decoding')
            print(middleware.scheduler.prefill_count, middleware.scheduler.decoding_count)
            middleware.completion_event.set()
        return result


    def _process_prefill(self, batch, use_compression, comp_mode):
        """处理prefill阶段，返回KV Cache等中间结果"""
        with torch.cuda.stream(torch.cuda.Stream()):
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

                prefill_start = time.time()
                if use_compression:
                    # 使用压缩模式
                    past_key_values, compression_time, compression_ratio, prefill_time = \
                        self._exec_compress(comp_mode, input_forward)
                    memory_usage = self._get_kv_cache_memory(past_key_values)
                else:
                    # 不使用压缩
                    outputs = self.model(
                        input_ids=input_forward["input_ids"],
                        attention_mask=input_forward["attention_mask"],
                        pixel_values=input_forward["pixel_values"],
                        use_cache=True,
                        return_dict=True
                    )
                    prefill_time = time.time() - prefill_start
                    past_key_values = outputs.past_key_values
                    memory_usage = self._get_kv_cache_memory(past_key_values)
                    compression_ratio = None
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
                    'compression_ratio': compression_ratio,
                    'compression_time': compression_time,
                    'prefill_time': prefill_time,
                    'inputs': inputs,  # 保存原始输入供decoding阶段使用
                    'request_size': batch_size
                }
                return prefill_result, batch_size, seq_lens, prefill_time, memory_usage


    def _process_decoding_wrapper(self, prefill_result, middleware, batch_start_time):
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
        print(1111)
        result, batch_size, valid_lens, decoding_time, throughput = self._process_decoding(prefill_result)  # 需要修改原始_process_batch以支持批处理
        # 更新性能指标
        middleware.scheduler.update_performance({
            'stage': 'decoding',
            'batch_size': batch_size,
            'tokens_generated': valid_lens,
            'processing_time': decoding_time,
            'throughput': throughput
        })
        process_time = time.time() - batch_start_time
        middleware.batch_times.append(process_time)
        with middleware.idle_lock:
            middleware.idle_workers += 1
            # print("batches", batches['input_ids'].shape)
            middleware.processed_decode_batches += batch_size
            print('middleware.processed_decode_batches',middleware.processed_decode_batches)
            middleware.completion_event.set()
        return result

    def _process_decoding(self, prefill_result):
        """处理decoding阶段，生成文本并返回结果"""
        with torch.cuda.stream(torch.cuda.Stream()):
            # 从prefill结果中提取必要信息
            # print(prefill_result)
            past_key_values = prefill_result['past_key_values']
            inputs = prefill_result['inputs']
            # print(inputs)

            # 准备生成输入
            batch_size = inputs['input_ids'].shape[0]
            kv_len = past_key_values[0][0].shape[2]

            # 构建attention mask
            kv_attention_mask = torch.cat([
                inputs['attention_mask'],
                torch.ones(
                    (batch_size, kv_len - inputs['attention_mask'].shape[1]),
                    dtype=inputs['attention_mask'].dtype,
                    device=self.device
                )
            ], dim=-1)

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
                max_new_tokens=512 - inputs['input_ids'].shape[1],
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

            decoding_time = time.time() - generation_start
            valid_lens = self.find_valid_lens(generated_sequences,
                                              self.processor.tokenizer.eos_token_id)
            throughput = valid_lens / decoding_time if decoding_time > 0 else 0

            # 构建返回结果
            result = TestResult(
                method='comp' if prefill_result['compression_ratio'] else 'orig',
                latency=[0, prefill_result['prefill_time'], decoding_time / valid_lens],
                success=True,
                memory_usage=prefill_result['memory_usage'],
                request_size=prefill_result['request_size'],
                throughput=throughput,
                valid_tokens=valid_lens,
                compression_ratio=prefill_result['compression_ratio'],
                compression_time=prefill_result['compression_time']
            )

            # 清理显存
            torch.cuda.empty_cache()

            return result, batch_size, valid_lens, decoding_time, throughput

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
    num_samples = 50     # 测试样本数
    batch_size = 1       # 批处理大小
    max_input_tokens = 128  # 最大输入长度
    compression_factor = 5  # 压缩率
    max_workers = 1    # 并发数
    # num_batches = 16    # 测试批次数

    # req_per_sec = 20.
    # max_serve_batch_size = 10 # 测试批次数
    # min_batch_threshold = 10
    # max_wait_time = 5
    req_per_sec = 10.
    max_serve_batch_size = 20  # 测试批次数
    min_batch_threshold = 1
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
        compression_factor=compression_factor,
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
    tester = ImprovedConcurrentTester(model, processor, compressor, device)
    
    # 运行并发测试
    print("\nStarting concurrent testing...")
    with autocast(dtype=torch.float):
        metrics = tester.run_concurrent_test(
            test_dataloader,
            max_workers=max_workers,
            num_samples=num_samples,
            max_serve_batch_size=max_serve_batch_size,
            min_batch_threshold=min_batch_threshold,
            max_wait_time=max_wait_time,
            worker_check_time=worker_check_time,
            req_per_sec=req_per_sec,
            use_compression=True
        )

    # 打印结果
    print_metrics(metrics)

    # 保存结果
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()