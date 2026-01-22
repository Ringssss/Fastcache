"""
LLaVA Scheduler - 支持多模态的调度器
====================================

主要修改：
1. 多模态prefill每次只处理一个序列（当前LLaVA限制）
2. 纯文本decode可以批量处理

Author: Claude Code
Date: 2024
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class LlavaScheduler:
    """
    LLaVA多模态调度器

    关键修改：对于多模态序列，prefill时每次只处理一个
    这是因为当前LLaVA实现不支持批量图片处理
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_finished = 0
        self.num_tokens = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _is_multimodal_seq(self, seq: Sequence) -> bool:
        """检查序列是否是多模态的"""
        return hasattr(seq, 'has_image') and seq.has_image

    def schedule(self, prefer_decode: bool = False) -> tuple[list[Sequence], bool]:
        """
        调度序列

        Returns:
            (scheduled_seqs, is_prefill)

        对于多模态序列，prefill时每次只返回一个序列
        """
        def schedule_prefill() -> list[Sequence]:
            scheduled: list[Sequence] = []
            num_seqs = 0
            num_batched_tokens = 0

            while self.waiting and num_seqs < self.max_num_seqs:
                seq = self.waiting[0]

                # 检查是否是多模态序列
                is_mm = self._is_multimodal_seq(seq)

                # 多模态序列：每次只处理一个
                if is_mm and scheduled:
                    # 已经有序列了，且当前是多模态，停止
                    break

                if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                    break

                num_seqs += 1
                self.block_manager.allocate(seq)
                num_batched_tokens += len(seq) - seq.num_cached_tokens
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled.append(seq)

                # 多模态序列：处理完一个就停止
                if is_mm:
                    break

            return scheduled

        def schedule_decode() -> list[Sequence]:
            scheduled: list[Sequence] = []
            num_seqs = 0
            # decode - 可以批量处理
            while self.running and num_seqs < self.max_num_seqs:
                seq = self.running.popleft()
                while not self.block_manager.can_append(seq):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    num_seqs += 1
                    self.block_manager.may_append(seq)
                    scheduled.append(seq)

            running = deque(scheduled)
            running.extend(self.running)
            self.running = running
            return scheduled

        if prefer_decode:
            scheduled_seqs = schedule_decode()
            if scheduled_seqs:
                return scheduled_seqs, False
            scheduled_seqs = schedule_prefill()
            if scheduled_seqs:
                return scheduled_seqs, True
        else:
            scheduled_seqs = schedule_prefill()
            if scheduled_seqs:
                return scheduled_seqs, True
            scheduled_seqs = schedule_decode()
            if scheduled_seqs:
                return scheduled_seqs, False

        assert scheduled_seqs
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        self.num_tokens += len(token_ids)
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                self.num_finished += 1
