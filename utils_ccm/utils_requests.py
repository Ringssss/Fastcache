import time
from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum


class RequestStatus(Enum):
    """请求状态枚举类"""
    CREATED = "created"  # 请求创建
    QUEUED = "queued"  # 在队列中等待
    PROCESSING = "processing"  # 正在处理
    COMPLETED = "completed"  # 处理完成
    FAILED = "failed"  # 处理失败


@dataclass
class RequestMetrics:
    """请求相关的时间指标"""
    create_time: float = 0.0  # 创建时间
    queue_time: float = 0.0  # 进入队列时间
    start_time: float = 0.0  # 开始处理时间
    prefill_start: float = 0.0  # prefill开始时间
    prefill_end: float = 0.0  # prefill结束时间
    decode_start: float = 0.0  # decode开始时间
    decode_end: float = 0.0  # decode结束时间
    complete_time: float = 0.0  # 完成时间

    @property
    def wait_time(self) -> float:
        """等待时间：从进入队列到开始处理的时间"""
        return self.start_time - self.queue_time if self.start_time > 0 else 0.0

    @property
    def prefill_time(self) -> float:
        """prefill阶段耗时"""
        return self.prefill_end - self.prefill_start if self.prefill_end > 0 else 0.0

    @property
    def decode_time(self) -> float:
        """decode阶段耗时"""
        return self.decode_end - self.decode_start if self.decode_end > 0 else 0.0

    @property
    def total_time(self) -> float:
        """总处理时间：从创建到完成的时间"""
        return self.complete_time - self.create_time if self.complete_time > 0 else 0.0


class Request:
    """请求管理类"""

    def __init__(self, request_id: int, input_data: Any):
        self.request_id = request_id  # 请求唯一标识
        self.input_data = input_data  # 输入数据
        self.status = RequestStatus.CREATED  # 请求状态
        self.metrics = RequestMetrics()  # 时间指标
        self.result = None  # 处理结果
        self.error = None  # 错误信息
        self.batch_id: Optional[int] = None  # 所属批次ID

        # 初始化创建时间
        self.metrics.create_time = time.time()

    def enter_queue(self) -> None:
        """请求进入队列"""
        self.status = RequestStatus.QUEUED
        self.metrics.queue_time = time.time()

    def start_processing(self) -> None:
        """开始处理请求"""
        self.status = RequestStatus.PROCESSING
        self.metrics.start_time = time.time()

    def start_prefill(self) -> None:
        """开始prefill阶段"""
        self.metrics.prefill_start = time.time()

    def end_prefill(self) -> None:
        """结束prefill阶段"""
        self.metrics.prefill_end = time.time()

    def start_decode(self) -> None:
        """开始decode阶段"""
        self.metrics.decode_start = time.time()

    def end_decode(self) -> None:
        """结束decode阶段"""
        self.metrics.decode_end = time.time()

    def complete(self, result: Any) -> None:
        """请求处理完成"""
        self.status = RequestStatus.COMPLETED
        self.result = result
        self.metrics.complete_time = time.time()

    def fail(self, error: Exception) -> None:
        """请求处理失败"""
        self.status = RequestStatus.FAILED
        self.error = error
        self.metrics.complete_time = time.time()

    def to_dict(self) -> dict:
        """将请求信息转换为字典格式"""
        return {
            "request_id": self.request_id,
            "batch_id": self.batch_id,
            "status": self.status.value,
            "metrics": {
                "create_time": self.metrics.create_time,
                "queue_time": self.metrics.queue_time,
                "start_time": self.metrics.start_time,
                "wait_time": self.metrics.wait_time,
                "prefill_time": self.metrics.prefill_time,
                "decode_time": self.metrics.decode_time,
                "total_time": self.metrics.total_time,
            },
            "error": str(self.error) if self.error else None
        }

    def __str__(self) -> str:
        """字符串表示"""
        return (f"Request(id={self.request_id}, "
                f"status={self.status.value}, "
                f"wait_time={self.metrics.wait_time:.3f}s, "
                f"prefill_time={self.metrics.prefill_time:.3f}s, "
                f"decode_time={self.metrics.decode_time:.3f}s, "
                f"total_time={self.metrics.total_time:.3f}s)")