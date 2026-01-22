from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self._num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        # 压缩相关状态
        self._kv_cache_len = None  # None表示未压缩，使用len(token_ids)

    def __len__(self):
        return len(self.token_ids)

    def __lt__(self, other):
        return self.seq_id < other.seq_id

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def num_completion_tokens(self):
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def num_cached_tokens(self):
        return self._num_cached_tokens

    @num_cached_tokens.setter
    def num_cached_tokens(self, num_cached_tokens):
        assert num_cached_tokens % self.block_size == 0
        self._num_cached_tokens = num_cached_tokens

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (len(self.token_ids) + self.block_size - 1) // self.block_size

    @property
    def last_token(self):
        return self.token_ids[-1]

    def block(self, i):
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def last_block(self):
        n = self.num_blocks
        return self.token_ids[(n-1)*self.block_size:]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        # 如果序列已压缩，同步增加kv_cache_len
        if self._kv_cache_len is not None:
            self._kv_cache_len += 1

    @property
    def kv_cache_len(self):
        """获取实际的KV-cache长度（压缩后可能小于token数）"""
        if self._kv_cache_len is not None:
            return self._kv_cache_len
        return len(self.token_ids)

    @kv_cache_len.setter
    def kv_cache_len(self, value):
        """设置压缩后的KV-cache长度"""
        self._kv_cache_len = value

    @property
    def is_compressed(self):
        """是否已压缩"""
        return self._kv_cache_len is not None

    @property
    def kv_num_blocks(self):
        """基于实际KV-cache长度计算需要的blocks数"""
        return (self.kv_cache_len + self.block_size - 1) // self.block_size
