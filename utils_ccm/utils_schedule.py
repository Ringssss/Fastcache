from queue import PriorityQueue, Empty
from dataclasses import dataclass, field
from typing import Any
import time
import torch
from itertools import accumulate

import time
import threading
from typing import Dict, List, Any
from dataclasses import dataclass
import uuid
import pynvml
from collections import defaultdict


class TimestampManager:
    """全局时间戳管理器"""

    def __init__(self):
        self.timestamps_map = {}  # request_id -> timestamps dict
        self.duratime_map = {}
        self._lock = threading.Lock()

    def create_request(self, request_id_list):
        """创建新请求并初始化时间戳"""
        # request_id = str(uuid.uuid4())
        with self._lock:
            for request_id in request_id_list:
                self.timestamps_map[request_id] = {
                    'prefill_queueing_start': None,
                    'prefill_start': None,
                    'prefill_finish': None,
                    'compress_queueing': None,
                    'compress_start': None,
                    'compress_finish': None,
                    'decoding_queueing': None,
                    'decode_start': None,
                    'decoding_finish': None,
                    'generated_lens': None,
                }

    def record_timestamp(self, request_id_list: List[str], stage: str, timestamps: float):
        """记录某个请求在特定阶段的时间戳"""
        with self._lock:
            for request_id in request_id_list:
                if request_id in self.timestamps_map:
                    self.timestamps_map[request_id][stage] = timestamps

    def get_timestamps(self, request_id: str) -> Dict:
        """获取某个请求的所有时间戳"""
        with self._lock:
            return self.timestamps_map.get(request_id, {}).copy()

    def remove_request(self, request_id: str):
        """请求完成后清理时间戳数据"""
        with self._lock:
            self.timestamps_map.pop(request_id, None)

    # analyze
    def calculate_stage_durations(self):
        """计算每个请求的各阶段耗时"""
        durations = {
            'prefill_queue_time': [],  # 预填充排队时间
            'prefill_time': [],  # 预填充处理时间
            'compress_queue_time': [],  # 压缩排队时间
            'compress_time': [],  # 压缩处理时间
            'decode_queue_time': [],  # 解码排队时间
            'decode_time': [],  # 解码处理时间
            'transmit_time': [],  # 解码处理时间
            'total_time': [],  # 总处理时间
            'TTFT': [],  # 总处理时间
            'TPOT': [],  # 总处理时间
            'latency': [],  # 总处理时间
            'generated_lens': [],
            'norm_latency': [],
        }

        for request_id, timestamps in self.timestamps_map.items():
            durations_for_req = {
                'prefill_queue_time': None,  # 预填充排队时间
                'prefill_time': None,  # 预填充处理时间
                'compress_queue_time': None,  # 压缩排队时间
                'compress_time': None,  # 压缩处理时间
                'decode_queue_time': None,  # 解码排队时间
                'decode_time': None,  # 解码处理时间
                'transmit_time': None,  # 解码处理时间
                'total_time': None,  # 总处理时间
                'TTFT': None,  # 总处理时间
                'TPOT': None,  # 总处理时间
                'latency': None,  # 总处理时间
                'generated_lens': None,  # 总处理时间
                'norm_latency': None  # 总处理时间
            }
            # 只处理完整的时间戳记录
            if not self._is_valid_timestamps(timestamps):
                continue

            transmit_time = 0

            # 计算各阶段时间
            prefill_queue = timestamps['prefill_start'] - timestamps['prefill_queueing_start']
            prefill_process = timestamps['prefill_finish'] - timestamps['prefill_start']
            transmit_time += timestamps['compress_queueing'] - timestamps['prefill_finish']
            compress_queue = timestamps['compress_start'] - timestamps['compress_queueing']
            compress_process = timestamps['compress_finish'] - timestamps['compress_start']
            transmit_time += timestamps['decoding_queueing'] - timestamps['compress_finish']
            decode_queue = timestamps['decode_start'] - timestamps['decoding_queueing']
            decode_process = timestamps['decoding_finish'] - timestamps['decode_start']
            total = timestamps['decoding_finish'] - timestamps['prefill_queueing_start']
            TTFT = timestamps['prefill_finish'] - timestamps['prefill_queueing_start']
            TPOT = decode_process/timestamps['generated_lens']
            # TPOT = (timestamps['decoding_finish'] - timestamps['prefill_finish'])/timestamps['generated_lens']
            latency = timestamps['decoding_finish'] - timestamps['prefill_queueing_start']
            generated_lens = timestamps['generated_lens']
            norm_latency = latency/generated_lens

            # 添加到对应列表
            durations['prefill_queue_time'].append(prefill_queue)
            durations['prefill_time'].append(prefill_process)
            durations['compress_queue_time'].append(compress_queue)
            durations['compress_time'].append(compress_process)
            durations['decode_queue_time'].append(decode_queue)
            durations['decode_time'].append(decode_process)
            durations['transmit_time'].append(transmit_time)
            durations['total_time'].append(total)
            durations['TTFT'].append(TTFT)
            durations['TPOT'].append(TPOT)
            durations['latency'].append(latency)
            durations['generated_lens'].append(generated_lens)
            durations['norm_latency'].append(norm_latency)

            durations_for_req['prefill_queue_time'] = prefill_queue
            durations_for_req['prefill_time']=prefill_process
            durations_for_req['compress_queue_time']=compress_queue
            durations_for_req['compress_time']=compress_process
            durations_for_req['decode_queue_time']=decode_queue
            durations_for_req['decode_time']=decode_process
            durations_for_req['transmit_time']=transmit_time
            durations_for_req['total_time']=total
            durations_for_req['TTFT']=TTFT
            durations_for_req['TPOT']=TPOT
            durations_for_req['latency']=latency
            durations_for_req['generated_lens']=generated_lens
            durations_for_req['norm_latency']=norm_latency

            self.duratime_map[request_id] = durations_for_req

        return durations

    def calculate_average_times(self, durations):
        """计算平均时间"""
        averages = {}
        for stage, times in durations.items():
            if times:  # 确保有数据
                avg_time = sum(times) / len(times)
                averages[stage] = round(avg_time, 3)  # 保留3位小数
            else:
                averages[stage] = 0
        return averages

    def calculate_generated_lens(self, durations):
        """计算平均时间"""
        generated_lens = sum(durations['generated_lens'])
        return generated_lens

    def _is_valid_timestamps(self, timestamps):
        """检查时间戳记录是否完整有效"""
        return all(timestamp is not None for timestamp in timestamps.values())

    def analyze_timestamps(self):
        """分析所有数据并返回结果"""
        durations = self.calculate_stage_durations()
        print("durations", durations)
        averages = self.calculate_average_times(durations)
        total_generated_lens = self.calculate_generated_lens(durations)

        return {
            # 'detailed_durations': durations,  # 保存所有请求的详细时间
            'average_times': averages,  # 保存平均时间
            # 'durations_for_req': self.duratime_map,
            'total_generated_lens': total_generated_lens,
        }, durations


class GPUMonitor:
    def __init__(self, interval=0.1):  # 采样间隔默认0.1秒
        """初始化GPU监控器"""
        self.interval = interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.state_name = ''
        self.state_round = [0,0,0]
        self.state_dict = {'prefill':0, 'compress':1, 'decoding':2}

        self.gpu_stats = defaultdict(list)
        # 初始化NVIDIA Management Library
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 假设使用第一个GPU

    def _collect_gpu_stats(self):
        """收集GPU统计信息"""
        while self.is_monitoring:
            try:
                # 获取GPU利用率
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                gpu_util = utilization.gpu

                # 获取显存使用情况
                memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                mem_used = memory.used / 1024 / 1024  # 转换为MB

                allocated = torch.cuda.memory_allocated() / 1024 / 1024

                # 记录当前时间戳和统计信息
                timestamp = time.time()
                round_state = self.get_state_round_info()

                self.gpu_stats['timestamp'].append(timestamp)
                self.gpu_stats['gpu_util'].append(gpu_util)
                self.gpu_stats['memory_used'].append(mem_used)
                self.gpu_stats['cuda_allocated'].append(allocated)
                self.gpu_stats['state_round'].append(round_state)

                time.sleep(self.interval)
            except Exception as e:
                print(f"Error collecting GPU stats: {e}")
                break

    def check_gpu_stats(self):
        # if self.is_monitoring:
        try:
            # 获取GPU利用率
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = utilization.gpu

            # 获取显存使用情况
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            mem_used = memory.used / 1024 / 1024  # 转换为MB

            allocated = torch.cuda.memory_allocated() / 1024 / 1024

            # 记录当前时间戳和统计信息
            timestamp = time.time()
            round_state = self.get_state_round_info()
            gpu_stats_dict = {
                'timestamp': timestamp,
                'gpu_util': gpu_util,
                'memory_used': mem_used,
                'cuda_allocated': allocated,
                'state_round': round_state,
            }
            return gpu_stats_dict
        except Exception as e:
            print(f"Error collecting GPU stats: {e}")
        # return None

    def start_monitoring(self):
        """启动GPU监控"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.gpu_stats.clear()  # 清除之前的数据
            self.monitor_thread = threading.Thread(target=self._collect_gpu_stats)
            self.monitor_thread.start()

    def stop_monitoring(self):
        """停止GPU监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def accum_state_round(self, state_name):
        self.state_round[self.state_dict[state_name]] += 1
        self.state_name = state_name

    def get_state_round_info(self):
        if self.state_name in self.state_dict.keys():
            return f'{self.state_name} {self.state_round[self.state_dict[self.state_name]]}'
        else:
            return f'ready -1'

    def get_stats(self):
        """获取统计结果"""
        return dict(self.gpu_stats)

    def save_stats(self, filename):
        """保存统计结果到文件"""
        import pandas as pd
        print('pd_len', len(self.gpu_stats['state_round']), len(self.gpu_stats['gpu_util']))
        df = pd.DataFrame(self.gpu_stats)
        df.to_csv(filename, index=False)

    def __del__(self):
        """清理资源"""
        pynvml.nvmlShutdown()

# class GPUMonitor:
#     def __init__(self, interval=0.03, monitor_sys=False):
#         pynvml.nvmlInit()
#         self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#         self._stop_flag = False
#         self._lock = threading.Lock()
#         self.stream_info = {}
#         self.monitor_sys = monitor_sys
#         self.interval = interval
#         self.gpu_stats = defaultdict(list)
#
#     def _collect_gpu_stats(self):
#         """收集GPU统计信息并存储"""
#         try:
#             # 获取基础信息
#             utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
#             memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
#
#             # 记录时间戳和数据
#             timestamp = time.time()
#             self.gpu_stats['timestamp'].append(timestamp)
#             self.gpu_stats['gpu_util'].append(utilization.gpu)
#             self.gpu_stats['memory_used'].append(memory.used / 1024 ** 2)
#             self.gpu_stats['memory_total'].append(memory.total / 1024 ** 2)
#             self.gpu_stats['memory_utilization'].append((memory.used / memory.total) * 100)
#
#             with self._lock:
#                 active_streams = len(self.stream_info)
#                 self.gpu_stats['active_streams'].append(active_streams)
#
#                 # 记录stream信息
#                 if active_streams > 0:
#                     stream_details = []
#                     for stream_id, info in self.stream_info.items():
#                         stream_details.append({
#                             'id': stream_id,
#                             'operation': info['operation'],
#                             'status': info['status']
#                         })
#                     self.gpu_stats['stream_details'].append(stream_details)
#
#             # 实时输出（如果需要）
#             if self.monitor_sys:
#                 self._print_current_stats(utilization, memory, active_streams)
#
#         except pynvml.NVMLError as e:
#             print(f"NVML Error: {e}")
#
#     def _monitor_loop(self):
#         while not self._stop_flag:
#             self._collect_gpu_stats()
#             time.sleep(self.interval)
#
#     def start_monitoring(self):
#         """启动GPU监控"""
#         if not self.is_monitoring:
#             self.is_monitoring = True
#             self.gpu_stats.clear()  # 清除之前的数据
#             self.monitor_thread = threading.Thread(target=self._collect_gpu_stats)
#             self.monitor_thread.start()
#
#     def stop_monitoring(self):
#         """停止GPU监控"""
#         self.is_monitoring = False
#         if self.monitor_thread:
#             self.monitor_thread.join()
#
#     def _print_current_stats(self, utilization, memory, active_streams):
#         print(f"\n{'=' * 50}")
#         print(f"GPU Utilization: {utilization.gpu}%")
#         print(f"Memory Used: {memory.used / 1024 ** 2:.2f} MB")
#         print(f"Memory Total: {memory.total / 1024 ** 2:.2f} MB")
#         print(f"Memory Utilization: {(memory.used / memory.total) * 100:.2f}%")
#         print(f"Active Streams: {active_streams}")
#
#         if active_streams > 0:
#             print("\nStream Details:")
#             with self._lock:
#                 for stream_id, info in self.stream_info.items():
#                     print(f"Stream {stream_id}:")
#                     print(f"  Operation: {info['operation']}")
#                     print(f"  Status: {info['status']}")
#
#     def save_stats(self, filename):
#         """保存统计结果到文件"""
#         import pandas as pd
#         df = pd.DataFrame(self.gpu_stats)
#         df.to_csv(filename, index=False)
#
#     def get_stats(self):
#         """获取统计结果"""
#         return dict(self.gpu_stats)
#
#     def __del__(self):
#         pynvml.nvmlShutdown()

# @dataclass
# class BatchedRequest:
#     """增强的批处理请求类"""
#     batch: Any  # 实际的批处理数据
#     request_type: str  # prefill 或 decoding
#     request_ids: List[str]  # 该批次包含的所有请求ID
#     arrival_time: float


# # 修改现有的PriorityRequest类
# class PriorityRequest:
#     def __init__(self, batch, request_type: str, arrival_time: float):
#         self.batch = batch
#         self.request_type = request_type
#         self.arrival_time = arrival_time
#         self.request_id = str(uuid.uuid4())  # 为每个请求添加唯一ID


# 增强的TestResult类




@dataclass
class PriorityRequest:
    batch: Any = field(default=None)
    """带优先级的请求，用于PriorityQueue"""
    request_type: str = field(default='prefill')
    request_id : List[str] = field(default_factory=lambda: [str(uuid.uuid4())], compare=False)
    # 防止同优先级时比较request造成错误
    # arrival_time: float = field(default=0.0)
    arrival_time: float = field(default_factory=time.time)

@dataclass(order=True)
class PrioritizedRequest:
    request: Any = field(compare=False)
    """带优先级的请求，用于PriorityQueue"""
    priority: float = field(default=0.0)
    # 防止同优先级时比较request造成错误
    # insert_time: float = field(default=0.0)
    insert_time: float = field(default_factory=time.time)



class DynamicScheduler:
    """使用Queue的动态调度器"""

    def __init__(self, max_batch_size=32, min_batch_size=1,
                 std_batch_threshold_decoding=2,
                 priority_levels=5, max_wait_time=2.0):
        # 使用PriorityQueue替代列表
        self.prefill_queue = PriorityQueue()
        self.decoding_queue = PriorityQueue()

        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.std_batch_threshold_decoding = std_batch_threshold_decoding
        self.priority_levels = priority_levels
        self.max_wait_time = max_wait_time

        # 性能监控
        self.perf_window = []
        self.window_size = 50
        self.current_mode = 'prefill'

        # PID控制器参数
        self.kp = 0.5
        self.ki = 0.2
        self.kd = 0.1
        self.prev_error = 0
        self.integral = 0

        # 队列统计信息
        self.prefill_count = 0
        self.decoding_count = 0
        self.decoding_lens_list = []

    def add_request(self, request, request_type):
        if request_type == 'prefill':
            """添加新请求到对应队列"""
            priority = self._calculate_initial_priority(request.batch['input_ids'].shape[1])
            prioritized_req = PrioritizedRequest(
                priority=priority,
                request=request
            )
            self.prefill_queue.put(prioritized_req)
            self.prefill_count += 1
        else:
            """添加新请求到对应队列"""
            priority = self._calculate_initial_priority(request.batch['inputs']['input_ids'].shape[1])
            prioritized_req = PrioritizedRequest(
                priority=priority,
                request=request
            )
            self.decoding_queue.put(prioritized_req)
            self.decoding_count += 1
            self.decoding_lens_list.append(request.batch['request_size'])

    def _calculate_initial_priority(self, sequence_length):
        """计算请求的初始优先级"""

        # sequence_length = request.batch['input_ids'].shape[1]
        # print(request.batch['input_ids'].shape)
        # 序列越短优先级越高
        length_factor = 1.0 - (sequence_length / 2048)  # 假设最大长度2048
        return length_factor * 100

    def update_priorities(self):
        """更新队列中请求的优先级"""
        current_time = time.time()
        # 更新prefill队列
        temp_prefill = []
        while not self.prefill_queue.empty():
            try:
                item = self.prefill_queue.get_nowait()
                wait_time = current_time - item.insert_time

                # 计算新优先级
                base_score = min(wait_time / self.max_wait_time, 1.0) * 100
                length_factor = 1.0 - (item.request.batch['input_ids'].shape[1] / 2048)
                # new_priority = base_score * (1 + length_factor)
                new_priority = min(max(base_score * (1 + length_factor), 0), 100)

                temp_prefill.append(PrioritizedRequest(
                    priority=new_priority,
                    insert_time=item.insert_time,
                    request=item.request
                ))
            except Empty:
                print('Error update_priorities')
                break

        # 将更新后的请求重新放入队列
        for item in temp_prefill:
            self.prefill_queue.put(item)

        # 更新decoding队列
        temp_decoding = []
        while not self.decoding_queue.empty():
            try:
                item = self.decoding_queue.get_nowait()
                wait_time = current_time - item.insert_time
                # new_priority = wait_time / self.max_wait_time * 100
                new_priority = min(max(wait_time / self.max_wait_time * 100, 0), 100)

                temp_decoding.append(PrioritizedRequest(
                    priority=new_priority,
                    insert_time=item.insert_time,
                    request=item.request
                ))
            except Empty:
                print('Error update_priorities')
                break

        # 将更新后的请求重新放入队列
        for item in temp_decoding:
            self.decoding_queue.put(item)

    def binary_search_last_le(self, lst, x):
        left, right = 0, len(lst) - 1
        result = -1  # 初始化为-1，表示没找到

        while left <= right:
            mid = left + (right - left) // 2  # 防止溢出
            if lst[mid] <= x:
                result = mid  # 记录当前位置
                left = mid + 1  # 继续向右找可能存在的更大位置
            else:
                right = mid - 1
        return result

    def adaptive_batch_size(self, queue_type, current_performance):
        if queue_type == 'prefill':
            """动态调整批处理大小"""
            if len(self.perf_window) == 0:
                return self.max_batch_size

            avg_perf = sum(self.perf_window) / len(self.perf_window)
            error = current_performance - avg_perf

            # PID控制
            self.integral += error
            derivative = error - self.prev_error

            adjustment = (self.kp * error +
                          self.ki * self.integral +
                          self.kd * derivative)

            # 更新历史
            self.prev_error = error

            # 获取当前队列大小
            current_size = self.prefill_count
            # todo dynamic adjustment
            new_size = current_size
            # new_size = current_size + int(adjustment)

            # 限制在合理范围内
            return max(min(new_size, self.max_batch_size), self.min_batch_size)
        else:
            # """动态调整批处理大小"""
            # if len(self.perf_window) == 0:
            #     return self.max_batch_size
            #
            # avg_perf = sum(self.perf_window) / len(self.perf_window)
            # error = current_performance - avg_perf

            # # PID控制
            # self.integral += error
            # derivative = error - self.prev_error
            #
            # adjustment = (self.kp * error +
            #               self.ki * self.integral +
            #               self.kd * derivative)
            #
            # # 更新历史
            # self.prev_error = error

            # 获取当前队列大小
            current_size = sum(self.decoding_lens_list[:self.std_batch_threshold_decoding])
            print("current_size", current_size)
            cumsum_list = list(accumulate(self.decoding_lens_list))
            print("cumsum_list", cumsum_list)
            i_count = self.binary_search_last_le(cumsum_list, current_size)+1
            print("i_count", i_count)

            # todo dynamic adjustment
            new_size = i_count
            # new_size = current_size + int(adjustment)

            # 限制在合理范围内
            # return max(min(new_size, self.max_batch_size), self.min_batch_size)
            return new_size

    def select_batch(self, last_prefill_time, max_wait_time):
        """选择下一批处理的请求"""
        self.update_priorities()

        # 检查队列压力
        prefill_pressure = self.prefill_count
        decoding_pressure = self.decoding_count

        # 动态决策处理模式
        # if prefill_pressure > decoding_pressure * 1.2:
        if prefill_pressure > 0:
            self.current_mode = 'prefill'
            target_queue = self.prefill_queue
            count = self.prefill_count
            cur_time = time.time()
            if self.prefill_count < self.max_batch_size and cur_time-last_prefill_time<max_wait_time:
                return None, self.current_mode
        else:
            self.current_mode = 'decoding'
            target_queue = self.decoding_queue
            count = self.decoding_count

        # # traditional
        # if decoding_pressure > 0:
        #     self.current_mode = 'decoding'
        #     target_queue = self.decoding_queue
        #     count = self.decoding_count
        # else:
        #     self.current_mode = 'prefill'
        #     target_queue = self.prefill_queue
        #     count = self.prefill_count

        if count == 0:
            return None, None

        # 确定批大小
        current_perf = self.get_current_performance()
        batch_size = self.adaptive_batch_size(self.current_mode, current_perf)
        batch_size = min(batch_size, count)

        # 选择最高优先级的请求
        selected_requests = []
        print("batch_size_decoding", batch_size)
        for i in range(batch_size):
            try:
                item = target_queue.get_nowait()
                selected_requests.append(item.request)
                if self.current_mode == 'prefill':
                    self.prefill_count -= 1
                else:
                    self.decoding_count -= 1
                    print("self.decoding_count", self.decoding_count)
                    self.decoding_lens_list.pop(0)
                    print("self.decoding_lens_list", self.decoding_lens_list)
            except Empty:
                break

        return selected_requests, self.current_mode
        # if self.current_mode == 'prefill':
        #     for _ in range(batch_size):
        #         try:
        #             item = target_queue.get_nowait()
        #             selected_requests.append(item.request)
        #             self.prefill_count -= 1
        #         except Empty:
        #             break
        #
        # else:
        #     for _ in range(batch_size):
        #         try:
        #             item = target_queue.get_nowait()
        #             selected_requests.append(item.request)
        #             self.decoding_count -=
        #         except Empty:
        #             break
        #     self.decoding_lens_list.pop(0)

        # return selected_requests, self.current_mode

    def get_current_performance(self):
        """获取当前性能指标"""
        if not self.perf_window:
            return 0

        recent_perf = self.perf_window[-min(10, len(self.perf_window)):]
        print(recent_perf)
        return sum(recent_perf) / len(recent_perf)

    def update_performance(self, metrics):
        """更新性能统计"""
        # todo get processing_time
        perf_metrics = metrics['batch_size']
        self.perf_window.append(perf_metrics)
        if len(self.perf_window) > self.window_size:
            self.perf_window.pop(0)

    def get_queue_status(self):
        """获取队列状态信息"""
        return {
            'prefill_count': self.prefill_count,
            'decoding_count': self.decoding_count,
            'current_mode': self.current_mode
        }






















