
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from torch.nn import functional as F
# from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, LlavaForConditionalGeneration
# # from typing import List, Dict, Tuple
# from transformers.generation.utils import *
# import math
from dataclasses import dataclass
import time
from utils_ccm.module_ccm import *
from transformers import QuantizedCache
import torch.nn as nn


from kvpress import KnormPress, RandomPress, SnapKVPress, StreamingLLMPress, \
    ExpectedAttentionPress, TOVAPress
from kvpress.presses.scorer_press import ScorerPress


class ImprovedKVQuantizer:
    """改进的KV-cache量化器，包含完整的量化和反量化流程"""

    def __init__(self, nbits=4):
        self.nbits = nbits
        self.cache = {}  # 用于存储量化参数

    def __call__(self, model):
        return self.QuantizeContext(self)

    def quantize_cache(self, past_key_values):
        """量化KV-cache，包含完整的量化和校准流程"""
        quantized = []
        for layer_idx, layer_cache in enumerate(past_key_values):
            k, v = layer_cache
            # 对key和value分别进行量化
            k_scaled, k_params = self._quantize_tensor_with_params(k, f"k_{layer_idx}")
            v_scaled, v_params = self._quantize_tensor_with_params(v, f"v_{layer_idx}")
            # 存储量化参数
            self.cache[f"k_{layer_idx}"] = k_params
            self.cache[f"v_{layer_idx}"] = v_params
            quantized.append((k_scaled, v_scaled))
        return tuple(quantized)

    def _quantize_tensor_with_params(self, x, key):
        """增强的张量量化，包含参数计算和存储"""
        # 计算每个通道的缩放因子和零点
        max_val = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
        scale = max_val / (2 ** (self.nbits - 1) - 1)

        # 量化
        x_quant = torch.round(x / scale)
        x_quant = torch.clamp(x_quant, -2 ** (self.nbits - 1), 2 ** (self.nbits - 1) - 1)

        # 反量化
        x_dequant = x_quant * scale

        # 存储量化参数
        params = {
            'scale': scale,
            'max_val': max_val,
        }

        return x_dequant, params

    def dequantize_cache(self, quantized_cache):
        """反量化KV-cache"""
        dequantized = []
        for layer_idx, (k_quant, v_quant) in enumerate(quantized_cache):
            k_params = self.cache[f"k_{layer_idx}"]
            v_params = self.cache[f"v_{layer_idx}"]

            # 反量化key和value
            k_dequant = k_quant * k_params['scale']
            v_dequant = v_quant * v_params['scale']

            dequantized.append((k_dequant, v_dequant))
        return tuple(dequantized)

    class QuantizeContext:
        def __init__(self, quantizer):
            self.quantizer = quantizer

        def __enter__(self):
            return self.quantizer

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

class BaseCompressionMethod:
    """压缩方法的基类"""

    def __init__(self, compression_ratio=0.5):
        self.compression_ratio = compression_ratio

    def __call__(self, model):
        underlying_model = get_underlying_model(model)
        return self.CompressionContext(underlying_model, self.compression_ratio)

    class CompressionContext:
        def __init__(self, model, compression_ratio):
            self.model = model
            self.compression_ratio = compression_ratio

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

class SimpleKVQuantizer:
    """简单的KV-cache量化器"""

    def __init__(self, nbits=4):
        self.nbits = nbits

    def reset_timeCheck(self):
        self.totalTime = 0

    def __call__(self, model):
        start_time = time.time()
        final_time = time.time()
        result = self.QuantizeContext(self)
        self.totalTime += final_time - start_time
        return result

    def quantize_cache(self, past_key_values):
        """量化KV-cache"""
        quantized = []
        for layer_cache in past_key_values:
            k, v = layer_cache
            # 简单的对称量化
            k_scaled = self._quantize_tensor(k)
            v_scaled = self._quantize_tensor(v)
            quantized.append((k_scaled, v_scaled))
        return tuple(quantized)

    def _quantize_tensor(self, x):
        """基础的对称量化"""
        max_val = torch.max(torch.abs(x))
        scale = max_val / (2 ** (self.nbits - 1) - 1)
        # 量化
        x_quant = torch.round(x / scale)
        # 裁剪到量化范围
        x_quant = torch.clamp(x_quant, -2 ** (self.nbits - 1), 2 ** (self.nbits - 1) - 1)
        # 反量化
        x_dequant = x_quant * scale
        return x_dequant

    class QuantizeContext:
        def __init__(self, quantizer):
            self.quantizer = quantizer

        def __enter__(self):
            return self.quantizer

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

class QuantizedCacheWrapper:
    def __init__(self, nbits=4):
        self.nbits = nbits

    def __call__(self, model):
        return self.QuantizedContext(model, self.nbits)

    class QuantizedContext:
        def __init__(self, model, nbits):
            self.model = model
            self.nbits = nbits
            self.original_impl = None

        def __enter__(self):
            # 保存原始实现
            self.original_impl = getattr(self.model.config, "cache_implementation", None)
            # 设置量化实现
            self.model.config.cache_implementation = "quantized"
            self.model.config.cache_config = {
                "backend": "quanto",
                "nbits": self.nbits
            }
            return self.model

        def __exit__(self, exc_type, exc_val, exc_tb):
            # 恢复原始实现
            if self.original_impl is not None:
                self.model.config.cache_implementation = self.original_impl
            else:
                delattr(self.model.config, "cache_implementation")
            if hasattr(self.model.config, "cache_config"):
                delattr(self.model.config, "cache_config")


class ScorerPressWithTime(ScorerPress):

    def reset_timeCheck(self):
        self.totalTime = 0
        # if not hasattr(self, 'totalTime'):
        #     self.totalTime = 0
        # else:
        #     self.totalTime = 0

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Default forward hook called after the forward pass of an attention layer.
        The hook calls the compress method to compress the KV cache while ensuring:
            - compression is only applied only during the pre-filling phase
            - KV cache quantization is handled correctly

        Parameters
        ----------
        module :
            Transformer attention layer.
        input :
            Input to the hook. This is the input to the forward pass of the layer.
        kwargs :
            Keyword arguments, as given to the forward pass of the layer.
        output :
            Output of the hook. This is the original output of the forward pass of the layer.

        Returns
        -------
            Modified output of the forward pass of the layer.

        """
        # See e.g. LlamaDecoderLayer.forward for the output structure
        if len(output) == 3:
            _, attentions, cache = output
        else:
            attentions, cache = None, output[-1]

        hidden_states = kwargs["hidden_states"]
        q_len = hidden_states.shape[1]

        # Don't compress after pre-filling
        if cache.seen_tokens > q_len:
            return output

        if isinstance(cache, QuantizedCache):
            keys = cache._dequantize(cache._quantized_key_cache[module.layer_idx])
            values = cache._dequantize(cache._quantized_value_cache[module.layer_idx])
        else:
            keys = cache.key_cache[module.layer_idx]
            values = cache.value_cache[module.layer_idx]
        # torch.cuda.synchronize()
        _ = keys.shape[2]
        start_time = time.time()
        keys, values = self.compress(module, hidden_states, keys, values, attentions, kwargs)

        if isinstance(cache, QuantizedCache):
            cache._quantized_key_cache[module.layer_idx] = cache._quantize(keys, axis=cache.axis_key)
            cache._quantized_value_cache[module.layer_idx] = cache._quantize(values, axis=cache.axis_value)
            cache.key_cache[module.layer_idx] = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache.value_cache[module.layer_idx] = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache._seen_tokens = keys.shape[2]
        else:
            cache.key_cache[module.layer_idx] = keys
            cache.value_cache[module.layer_idx] = values
        # torch.cuda.synchronize()
        _ = cache.key_cache[module.layer_idx].shape[2]
        finish_time = time.time()
        self.totalTime += finish_time - start_time

        return output

@dataclass
class StreamingLLMPressWithTime(ScorerPressWithTime, StreamingLLMPress):
    pass

@dataclass
class RandomPressWithTime(ScorerPressWithTime, RandomPress):
    pass

@dataclass
class SnapKVPressWithTime(ScorerPressWithTime, SnapKVPress):
    pass

@dataclass
class ExpectedAttentionPressWithTime(ScorerPressWithTime, ExpectedAttentionPress):
    pass

@dataclass
class KnormPressWithTime(ScorerPressWithTime, KnormPress):
    pass

@dataclass
class TOVAPressWithTime(ScorerPressWithTime, TOVAPress):
    pass


def press_select(compression_method='Knorm', compression_ratio=0.8):
    # 选定压缩方法处理
    if compression_method == "StreamingLLM":
        press = StreamingLLMPressWithTime(compression_ratio=compression_ratio)
        press.reset_timeCheck()
    elif compression_method == "RandomPress":
        press = RandomPressWithTime(compression_ratio=compression_ratio)
        press.reset_timeCheck()
    elif compression_method == "SnapKV":
        press = SnapKVPressWithTime(compression_ratio=compression_ratio)
        press.reset_timeCheck()
    elif compression_method == "ExpectedAttention":
        press = ExpectedAttentionPressWithTime(compression_ratio=compression_ratio)
        press.reset_timeCheck()
    elif compression_method == "TOVA":
        press = TOVAPressWithTime(compression_ratio=compression_ratio)
        press.reset_timeCheck()
    elif compression_method == "Quantized":
        press = SimpleKVQuantizer(nbits=4)
    else:
        press = KnormPressWithTime(compression_ratio=compression_ratio)
        press.reset_timeCheck()
    return press

def get_underlying_model(model):
    possible_model_attrs = ['language_model', 'llava_model', 'llama_model', 'decoder']
    underlying_model = None
    for attr in possible_model_attrs:
        if hasattr(model, attr):
            underlying_model = getattr(model, attr)
            break
    if underlying_model is None:
        raise ValueError("Could not find underlying transformer model")

    return underlying_model

def exec_compression_methods(model, underlying_model, inputs, press, compression_method):
    if compression_method == "Quantized":
        outputs_compressed = model(**inputs)
        orig_past_key_values = outputs_compressed.past_key_values
        compressed_past_key_values = press.quantize_cache(orig_past_key_values)
    else:
        with press(underlying_model):
            outputs_compressed = model(**inputs)
            compressed_past_key_values = outputs_compressed.past_key_values
    return compressed_past_key_values


class TimeTik(object):
    def __init__(self):
        self.timeTable = {}
        self.timeTimesTable = {}
        self.startTimeTable = {}
        self.countTable = {}
        self.pos_reset()
        self.pos_tik_check = 0


    def timeTik(self, token_name=''):
        if token_name not in self.timeTable.keys():
            self.timeTable[token_name] = 0
            self.timeTimesTable[token_name] = 0
            # self.startTimeTable[token_name] = 0
        self.startTimeTable[token_name] = time.time()

    def timeTok(self, token_name=''):
        cumul_time = time.time()-self.startTimeTable[token_name]
        self.timeTable[token_name] += cumul_time
        self.timeTimesTable[token_name] += 1

    def timeTikPos(self):
        if self.pos_tik_check:
            self.generated_id_time.append(time.time())

    def timeTokPos(self, next_id):
        if self.pos_tik_check:
            self.generated_pos_latency.append(time.time() - self.generated_id_time[self.src_id])
            self.src_id = next_id

    def get_token_time(self):
        return self.generated_pos_latency

    def reset(self):
        for token_name in self.timeTable.keys():
            self.timeTable[token_name] = 0
            self.timeTimesTable[token_name] = 0

    def show(self):
        for token_name in self.timeTable.keys():
            print(f'{token_name}: {self.timeTable[token_name]:.4f}. {self.timeTimesTable[token_name]}')

    def show_and_reset(self):
        for token_name in self.timeTable.keys():
            print(f'{token_name}: {self.timeTable[token_name]:.4f}. {self.timeTimesTable[token_name]}')
            self.timeTable[token_name] = 0
            self.timeTimesTable[token_name] = 0

    def pos_reset(self):
        self.generated_pos_latency = []
        self.generated_id_time = []
        self.src_id = 0

    def show_posTik(self):
        print('generated_pos_latency', self.generated_pos_latency)
        # print('generated_id_time', self.generated_id_time)

    def show_posTik_and_reset(self):
        print('generated_pos_latency', self.generated_pos_latency)
        # print('generated_id_time', self.generated_id_time)
        self.pos_reset()

