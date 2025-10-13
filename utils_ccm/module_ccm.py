
import sys

# 创建虚拟的flash_attn模块
class MockFlashAttn:
    def __getattr__(self, name):
        def dummy(*args, **kwargs):
            raise NotImplementedError("Flash Attention is disabled")
        return dummy

# 屏蔽flash_attn相关模块
sys.modules['flash_attn'] = MockFlashAttn()
sys.modules['flash_attn.flash_attn_interface'] = MockFlashAttn()
sys.modules['flash_attn.bert_padding'] = MockFlashAttn()
sys.modules['flash_attn_2_cuda'] = MockFlashAttn()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, LlavaForConditionalGeneration
from typing import List, Dict, Tuple, Optional, Union
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, \
        GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.generation.streamers import BaseStreamer
# from transformers.generation.utils import *

import math


class LLM:
    def __init__(self, model_path: str, trust_remote_code: bool = True, processor_path = None, torch_dtype=torch.float32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        # self.model = LlavaForConditionalGenerationDcache.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_path, trust_remote_code=trust_remote_code,
                                                                   torch_dtype=torch_dtype)
        if processor_path is None:
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            self.processor = AutoProcessor.from_pretrained(processor_path)
        # self.processor.pad_token = self.processor.unk_token
        # self.processor.padding_side = "left"
        # 确保model知道pad token id
        # self.model.config.pad_token_id = self.processor.pad_token_id

        self.model.eval()
        self.model.to(self.device)

class LLaMAKVCacheSampler:
    def __init__(self, recent_size=128, device='cuda'):
        self.recent_size = recent_size
        self.device = device

    def hybrid_sampling(self, past_key_values, num_samples):
        num_layers = len(past_key_values)
        _, _, seq_len, _ = past_key_values[0][0].shape

        # 1. 保留最近的tokens
        num_recent = min(self.recent_size, seq_len)
        recent_indices = torch.arange(seq_len - num_recent, seq_len).to(self.device)

        # 2. 对历史tokens进行重要性采样
        remaining_samples = num_samples - num_recent
        if remaining_samples > 0 and seq_len > num_recent:
            historical_indices = self._sample_historical(
                past_key_values,
                remaining_samples,
                exclude_indices=recent_indices
            )
            # 合并并排序索引
            indices = torch.cat([historical_indices, recent_indices])
            indices = torch.sort(indices)[0]
        else:
            indices = recent_indices

        # 3. 应用采样到所有层
        new_past_key_values = []
        for layer_idx in range(num_layers):
            key_cache, value_cache = past_key_values[layer_idx]
            sampled_key = key_cache.index_select(2, indices)
            sampled_value = value_cache.index_select(2, indices)
            new_past_key_values.append((sampled_key, sampled_value))

        return tuple(new_past_key_values), indices

    def _sample_historical(self, past_key_values, num_samples, exclude_indices):
        # 获取最后一层的注意力分数
        last_layer = past_key_values[-1]
        key_cache = last_layer[0]

        # 计算注意力分数
        query = key_cache[:, :, -1:, :]
        scores = torch.matmul(query, key_cache.transpose(-2, -1)) / \
                 math.sqrt(key_cache.size(-1))

        # 屏蔽已选择的位置
        mask = torch.ones(scores.shape[-1], dtype=torch.bool, device=self.device)
        mask[exclude_indices] = False
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), -float('inf'))

        # 计算重要性分数并采样
        importance = torch.softmax(scores, dim=-1).mean(dim=(0, 1, 2))
        _, indices = torch.topk(importance, k=num_samples)

        return indices


class CustomDataset(Dataset):
    def __init__(self, conversations: List[str], tokenizer, max_length: int = 512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.tokenizer_draft = 0
        self.max_length = max_length

    def new_tokenizer(self, tokenizer):
        self.tokenizer_draft = tokenizer

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conversation = self.conversations[idx]
        # encoding = self.tokenizer(
        #     conversation,
        #     truncation=True,
        #     padding='max_length',
        #     max_length=self.max_length,
        #     return_tensors='pt'
        # )
        encoding = self.tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        if self.tokenizer_draft != 0:
            encoding_draft = self.tokenizer_draft(
                conversation,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'draft_input_ids': encoding_draft['input_ids'].squeeze(0),
                'draft_attention_mask': encoding_draft['attention_mask'].squeeze(0)
            }
        else:

            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            }

        # return {
        #     'input_ids': encoding['input_ids'],
        #     'attention_mask': encoding['attention_mask']
        # }

class CustomImageTextDataset(Dataset):
    def __init__(self, conversations: List[str], processor, max_length: int = 512, launch_padding_flag=True):
        self.conversations = conversations
        self.processor = processor
        self.max_length = max_length
        self.launch_padding_flag = launch_padding_flag

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conversation = self.conversations[idx]
        # encoding = self.tokenizer(
        #     conversation,
        #     truncation=True,
        #     padding='max_length',
        #     max_length=self.max_length,
        #     return_tensors='pt'
        # )
        encoding_list = []
        for i_con in range(len(conversation)):
            if conversation[1][i_con] == 1: # load image
                if self.launch_padding_flag:
                    encoding = self.processor(
                        images=conversation[0][i_con][1],
                        text=conversation[0][i_con][0],
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                else:
                    encoding = self.processor(
                        images=conversation[0][i_con][1],
                        text=conversation[0][i_con][0],
                        # padding='max_length',
                        truncation=True,
                        # max_length=self.max_length,
                        return_tensors='pt'
                    )
                encoding_result = {
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'pixel_values': encoding['pixel_values'].squeeze(0),
                    'type': torch.Tensor([1])
                }
            else:
                # encoding = self.processor(
                #     text=conversation[0][i_con],
                #     truncation=True,
                #     max_length=self.max_length,
                #     return_tensors='pt'
                # )
                # encoding_answer = self.processor(
                #     text=conversation[1][i_con],
                #     truncation=True,
                #     max_length=self.max_length,
                #     return_tensors='pt'
                # )
                # encoding_result = {
                #     'input_ids': encoding['input_ids'].squeeze(0),
                #     'attention_mask': encoding['attention_mask'].squeeze(0),
                #     'answer': encoding_answer['input_ids'].squeeze(0),
                #     'type': torch.Tensor([0])
                # }
                return encoding_list #todo skip multi_round
            encoding_list.append(encoding_result)

        return encoding_list

        # return {
        #     'input_ids': encoding['input_ids'],
        #     'attention_mask': encoding['attention_mask']
        # }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MAEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, kv_dim: int):
        super().__init__()
        self.projection = nn.Linear(kv_dim, input_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8),
            num_layers=num_layers
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

    def forward(self, kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        combined_kv = []
        for k, v in kv_cache:
            combined = torch.cat([k, v], dim=-1)
            combined = combined.permute(0, 2, 1, 3).reshape(k.size(0), k.size(2), -1)
            combined_kv.append(combined)

        x = torch.stack(combined_kv, dim=1)
        batch_size, num_layers, seq_length, feature_dim = x.shape
        x = x.reshape(batch_size, num_layers * seq_length, feature_dim)

        x = self.projection(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x.permute(1, 0, 2)  # [seq_len, batch_size, input_dim]
        x = self.transformer(x)
        x = torch.split(x, [1, x.shape[0 ] -1], dim=0)[1]
        return x.permute(1, 0, 2)  # [batch_size, seq_len, input_dim]




class CompressedKVCache:
    def __init__(self, num_layers, num_heads, head_dim, device):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.device = device
        self.reset()

    def reset(self):
        self.key_cache = [torch.zeros(0, self.num_heads, 0, self.head_dim, device=self.device) for _ in
                          range(self.num_layers)]
        self.value_cache = [torch.zeros(0, self.num_heads, 0, self.head_dim, device=self.device) for _ in
                            range(self.num_layers)]

    def update(self, key_states, value_states, layer_idx):
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get(self):
        return list(zip(self.key_cache, self.value_cache))


class CompressedDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, compression_ratio, num_layers, num_heads, num_llm_layers):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.num_layers = num_layers
        self.num_llm_layers = num_llm_layers
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        # self.transformer_compressor = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_compressor = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        # self.transformer_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.output_projection = nn.Linear(hidden_dim, output_dim * 2)  # 双倍输出维度，为key和value

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, feature_dim = x.shape
        # num_layers = 32
        # seq_length_ilayer = seq_len // num_layers
        # x = x.reshape(batch_size, self.num_layers, -1, feature_dim)
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        x = self.transformer_compressor(x)
        seq_length_ilayer = seq_len // self.num_llm_layers

        # x = x.reshape(batch_size, self.num_layers, -1, feature_dim)
        compressed_len = max(1, seq_length_ilayer // self.compression_ratio)
        x_layer_list = torch.split(x, seq_length_ilayer, dim=0)
        kv_cache_compressed = []
        for x in x_layer_list:
            # xcompress_layer_list = []
            # x = F.adaptive_avg_pool1d(x.permute(2, 1, 0, 3), compressed_len).permute(2, 1, 0, 3)
            x = F.adaptive_avg_pool1d(x.permute(1, 2, 0), compressed_len).permute(2, 0, 1)
            x = self.output_projection(x)  # [compressed_len, batch_size, output_dim * 2]
            x = x.permute(1, 0, 2)
            x = x.reshape(batch_size, compressed_len, -1, feature_dim).permute(0, 2, 1, 3)
            # xcompress_layer_list.append()
            kv_cache_compressed.append((torch.split(x, x.shape[3]//2, dim=3)))

        # Reshape to separate key and value
        # x = torch.stack(xcompress_layer_list).permute(2,0,1,3)
        # x = x.view(compressed_len, batch_size, 2, self.num_heads, self.head_dim)
        # k, v = x[:, :, 0], x[:, :, 1]
        # return k.permute(1, 2, 0, 3), v.permute(1, 2, 0, 3)  # [batch_size, num_layers, compressed_len, out]
        return kv_cache_compressed  # [batch_size, num_layers, compressed_len, out]

    def compress_with_len(self, x, compressed_len):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, feature_dim = x.shape
        # num_layers = 32
        # seq_length_ilayer = seq_len // num_layers
        # x = x.reshape(batch_size, self.num_layers, -1, feature_dim)
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        x = self.transformer_compressor(x)
        seq_length_ilayer = seq_len // self.num_llm_layers

        # x = x.reshape(batch_size, self.num_layers, -1, feature_dim)
        # compressed_len = max(1, seq_length_ilayer // self.compression_ratio)
        x_layer_list = torch.split(x, seq_length_ilayer, dim=0)
        kv_cache_compressed = []
        for x in x_layer_list:
            # xcompress_layer_list = []
            # x = F.adaptive_avg_pool1d(x.permute(2, 1, 0, 3), compressed_len).permute(2, 1, 0, 3)
            x = F.adaptive_avg_pool1d(x.permute(1, 2, 0), compressed_len).permute(2, 0, 1)
            x = self.output_projection(x)  # [compressed_len, batch_size, output_dim * 2]
            x = x.permute(1, 0, 2)
            x = x.reshape(batch_size, compressed_len, -1, feature_dim).permute(0, 2, 1, 3)
            # xcompress_layer_list.append()
            kv_cache_compressed.append((torch.split(x, x.shape[3]//2, dim=3)))

        # Reshape to separate key and value
        # x = torch.stack(xcompress_layer_list).permute(2,0,1,3)
        # x = x.view(compressed_len, batch_size, 2, self.num_heads, self.head_dim)
        # k, v = x[:, :, 0], x[:, :, 1]
        # return k.permute(1, 2, 0, 3), v.permute(1, 2, 0, 3)  # [batch_size, num_layers, compressed_len, out]
        return kv_cache_compressed  # [batch_size, num_layers, compressed_len, out]

class CompressedDecoder_OnlyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, compression_ratio, num_layers, num_heads):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.num_layers = num_layers
        self.num_heads = num_heads
        # self.head_dim = output_dim // num_heads
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        # self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.output_projection = nn.Linear(hidden_dim, output_dim * 2)  # 双倍输出维度，为key和value

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, feature_dim = x.shape
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        # x = self.transformer_layer(x)
        seq_length_ilayer = seq_len // self.num_layers

        # x = x.reshape(batch_size, self.num_layers, -1, feature_dim)
        compressed_len = max(1, seq_length_ilayer // self.compression_ratio)
        x_layer_list = torch.split(x, seq_length_ilayer, dim=0)
        kv_cache_compressed = []
        for x in x_layer_list:
            # xcompress_layer_list = []
            # x = F.adaptive_avg_pool1d(x.permute(2, 1, 0, 3), compressed_len).permute(2, 1, 0, 3)
            x = F.adaptive_avg_pool1d(x.permute(1, 2, 0), compressed_len).permute(2, 0, 1)
            x = self.output_projection(x)  # [compressed_len, batch_size, output_dim * 2]
            x = x.permute(1, 0, 2)
            x = x.reshape(batch_size, compressed_len, -1, feature_dim).permute(0, 2, 1, 3)
            # xcompress_layer_list.append()
            kv_cache_compressed.append((torch.split(x, x.shape[3]//2, dim=3)))

        # Reshape to separate key and value
        # x = torch.stack(xcompress_layer_list).permute(2,0,1,3)
        # x = x.view(compressed_len, batch_size, 2, self.num_heads, self.head_dim)
        # k, v = x[:, :, 0], x[:, :, 1]
        # return k.permute(1, 2, 0, 3), v.permute(1, 2, 0, 3)  # [batch_size, num_layers, compressed_len, out]
        return kv_cache_compressed  # [batch_size, num_layers, compressed_len, out]



# class KVCacheLinearCompressor(nn.Module):
#     def __init__(self, src_config, compression_factor=2, min_seq_len=8):
#         # n_layer=32,d_model = , nhead=32
#         super(KVCacheLinearCompressor, self).__init__()

#         self.n_layer = src_config.text_config.num_hidden_layers
#         self.nhead = src_config.text_config.num_key_value_heads
#         self.d_model = src_config.text_config.hidden_size

#         self.head_dim = self.d_model // self.nhead
#         self.compression_factor = compression_factor
#         self.min_seq_len = min_seq_len  # 新增：最小序列长度阈值

#         # 为每一层创建压缩层
#         # self.compress_k = nn.ModuleList(
#         #     [nn.Linear(self.head_dim * compression_factor, self.head_dim) for _ in range(self.n_layer)])
#         # self.compress_v = nn.ModuleList(
#         #     [nn.Linear(self.head_dim * compression_factor, self.head_dim) for _ in range(self.n_layer)])

#         # 为每一层创建压缩层
#         self.compress_k = nn.ModuleList(
#             [nn.Sequential(nn.Linear(self.head_dim * compression_factor, self.head_dim),
#                            nn.ReLU(),
#                            nn.Dropout(p=0.4),
#                            nn.Linear(self.head_dim, self.head_dim),
#                            nn.ReLU(),
#                            nn.Dropout(p=0.4),
#                            nn.Linear(self.head_dim, self.head_dim)
#                            ) for _ in range(self.n_layer)])
#         self.compress_v = nn.ModuleList(
#             [nn.Sequential(nn.Linear(self.head_dim * compression_factor, self.head_dim),
#                            nn.ReLU(),
#                            nn.Dropout(p=0.4),
#                            nn.Linear(self.head_dim, self.head_dim),
#                            nn.ReLU(),
#                            nn.Dropout(p=0.4),
#                            nn.Linear(self.head_dim, self.head_dim)
#                            ) for _ in range(self.n_layer)])

#         # 为每一层创建注意力计算
#         self.attention = nn.ModuleList([nn.Linear(self.head_dim, 1) for _ in range(self.n_layer)])

#     def compress_layer(self, layer_cache, layer_idx):
#         k, v = layer_cache
#         batch_size, nhead, seq_len, head_dim = k.shape

#         # 计算压缩后的序列长度
#         compressed_seq_len = seq_len // self.compression_factor

#         # 检查压缩后的序列长度是否小于最小阈值
#         if compressed_seq_len < self.min_seq_len:
#             return k, v  # 如果压缩后长度过短，则返回原始序列
#         # 计算需要压缩的序列长度
#         compress_len = compressed_seq_len * self.compression_factor
#         # 重塑key和value以进行压缩
#         k_to_compress = k[:, :, :compress_len, :].reshape(batch_size, nhead, compressed_seq_len,
#                                                           self.head_dim * self.compression_factor)
#         v_to_compress = v[:, :, :compress_len, :].reshape(batch_size, nhead, compressed_seq_len,
#                                                           self.head_dim * self.compression_factor)
#         # 压缩key和value
#         compressed_k = self.compress_k[layer_idx](k_to_compress)
#         compressed_v = self.compress_v[layer_idx](v_to_compress)

#         # 处理剩余的部分（如果有的话）
#         if seq_len > compress_len:
#             remaining_k = k[:, :, compress_len:, :]
#             remaining_v = v[:, :, compress_len:, :]
#             compressed_k = torch.cat([compressed_k, remaining_k], dim=2)
#             compressed_v = torch.cat([compressed_v, remaining_v], dim=2)

#         return compressed_k, compressed_v

#     def forward(self, kv_cache):
#         # kv_cache: list of n_layer tuples, each tuple contains (k, v)
#         compressed_kv_cache = []
#         for layer_idx, layer_cache in enumerate(kv_cache):
#             compressed_k, compressed_v = self.compress_layer(layer_cache, layer_idx)
#             compressed_kv_cache.append((compressed_k, compressed_v))
#         return compressed_kv_cache
# try:
#     import triton
#     import triton.language as tl
#     TRITON_AVAILABLE = True
# except ImportError:
#     print("Triton not found. FusedLinearReLUTriton will fallback to standard PyTorch layers.")
#     TRITON_AVAILABLE = False

# # --- 1. 定义 Fused Linear + ReLU 的 Triton 内核 (简化示例) ---
# # !!! 警告：这个内核假设 M, N, K 是固定的，无法处理动态形状 !!!
# if TRITON_AVAILABLE:
#     @triton.autotune(
#         configs=[
#             triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
#             triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
#             triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
#             triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
#             triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 5, 'num_warps': 2}),
#         ],
#         key=['M', 'N', 'K'], # 内核会根据 M, N, K 特化
#     )
#     @triton.jit
#     def fused_linear_relu_kernel(
#         a_ptr, b_ptr, c_ptr, bias_ptr,
#         M, N, K,
#         stride_am, stride_ak,
#         stride_bk, stride_bn,
#         stride_cm, stride_cn,
#         stride_biasn,
#         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
#         HAS_BIAS: tl.constexpr,
#         GROUP_M: tl.constexpr = 8
#     ):
#         """
#         Kernel for computing the fused operation: C = ReLU(A @ B + bias)
#         A: (M, K) matrix
#         B: (K, N) matrix (corresponding to weight.T of nn.Linear)
#         C: (M, N) matrix (output)
#         bias: (N) vector (optional)
#         """
#         pid = tl.program_id(axis=0)
#         num_pid_m = tl.cdiv(M, BLOCK_M)
#         num_pid_n = tl.cdiv(N, BLOCK_N)
#         num_pid_in_group = GROUP_M * num_pid_n
#         group_id = pid // num_pid_in_group
#         first_pid_m = group_id * GROUP_M
#         group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
#         pid_m = first_pid_m + (pid % group_size_m)
#         pid_n = (pid % num_pid_in_group) // group_size_m

#         offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
#         offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
#         offs_k = tl.arange(0, BLOCK_K)
#         a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#         b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

#         accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
#         for k in range(0, tl.cdiv(K, BLOCK_K)):
#             a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
#             b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
#             accumulator += tl.dot(a, b)
#             a_ptrs += BLOCK_K * stride_ak
#             b_ptrs += BLOCK_K * stride_bk

#         if HAS_BIAS:
#             offs_biasn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
#             bias_ptrs = bias_ptr + offs_biasn * stride_biasn
#             # Bias shape [N], load into [1, BLOCK_N] fragment safely
#             bias = tl.load(bias_ptrs, mask=offs_bn[None,:] < N, other=0.0).to(tl.float32)
#             accumulator = accumulator + bias # Correct broadcasting

#         # Apply ReLU
#         c = tl.maximum(accumulator, 0) # ReLU = max(0, x)
#         c = c.to(c_ptr.dtype.element_ty)

#         offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
#         offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
#         c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
#         c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#         tl.store(c_ptrs, c, mask=c_mask)
# else:
#     # 提供一个占位符函数，如果Triton不可用
#     def fused_linear_relu_kernel(*args, **kwargs):
#         raise ImportError("Triton is not available, cannot run fused_linear_relu_kernel.")

# from .fused_linear_relu_module import FusedLinearReLUTriton
class KVCacheLinearDecoupleCompressor(nn.Module):
    def __init__(self, src_config, compression_factor=2, min_seq_len=1):
        # n_layer=32,d_model = , nhead=32
        super(KVCacheLinearDecoupleCompressor, self).__init__()

        self.n_layer = src_config.text_config.num_hidden_layers
        self.nhead = src_config.text_config.num_key_value_heads
        self.d_model = src_config.text_config.hidden_size

        self.head_dim = self.d_model // self.nhead
        self.compression_factor = compression_factor
        self.min_seq_len = min_seq_len  # 新增：最小序列长度阈值

        # 为每一层创建压缩层
        # self.compress_k = nn.ModuleList(
        #     [nn.Linear(self.head_dim * compression_factor, self.head_dim) for _ in range(self.n_layer)])
        # self.compress_v = nn.ModuleList(
        #     [nn.Linear(self.head_dim * compression_factor, self.head_dim) for _ in range(self.n_layer)])

        # 为每一层创建压缩层
        self.compress_tk = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.head_dim * compression_factor, self.head_dim),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(self.head_dim, self.head_dim),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(self.head_dim, self.head_dim)
                           ) for _ in range(self.n_layer)])
        self.compress_tv = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.head_dim * compression_factor, self.head_dim),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(self.head_dim, self.head_dim),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(self.head_dim, self.head_dim)
                           ) for _ in range(self.n_layer)])

        self.compress_ik = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.head_dim * compression_factor, self.head_dim),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(self.head_dim, self.head_dim),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(self.head_dim, self.head_dim)
                           ) for _ in range(self.n_layer)])
        self.compress_iv = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.head_dim * compression_factor, self.head_dim),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(self.head_dim, self.head_dim),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(self.head_dim, self.head_dim)
                           ) for _ in range(self.n_layer)])

        # 为每一层创建注意力计算
        self.attention = nn.ModuleList([nn.Linear(self.head_dim, 1) for _ in range(self.n_layer)])

    def compress_layer(self, layer_cache, layer_idx, compressor_k, compressor_v):
        k, v = layer_cache
        batch_size, nhead, seq_len, head_dim = k.shape

        # 计算压缩后的序列长度
        compressed_seq_len = seq_len // self.compression_factor

        # 检查压缩后的序列长度是否小于最小阈值
        if compressed_seq_len < self.min_seq_len:
            return k, v  # 如果压缩后长度过短，则返回原始序列
        # 计算需要压缩的序列长度
        compress_len = compressed_seq_len * self.compression_factor
        # 重塑key和value以进行压缩
        k_to_compress = k[:, :, :compress_len, :].reshape(batch_size, nhead, compressed_seq_len,
                                                          self.head_dim * self.compression_factor)
        v_to_compress = v[:, :, :compress_len, :].reshape(batch_size, nhead, compressed_seq_len,
                                                          self.head_dim * self.compression_factor)
        # 压缩key和value
        compressed_k = compressor_k[layer_idx](k_to_compress)
        compressed_v = compressor_v[layer_idx](v_to_compress)

        # 处理剩余的部分（如果有的话）
        if seq_len > compress_len:
            remaining_k = k[:, :, compress_len:, :]
            remaining_v = v[:, :, compress_len:, :]
            compressed_k = torch.cat([compressed_k, remaining_k], dim=2)
            compressed_v = torch.cat([compressed_v, remaining_v], dim=2)

        return compressed_k, compressed_v


    def forward(self, kv_cache, it_len=[1, 1], strategy_com="retain"):
        # kv_cache: list of n_layer tuples, each tuple contains (k, v)
        compressed_kv_cache = []
        # image_kv_cache = tuple(tuple([I_kv_cache[0][:, :, :it_len[0], :], I_kv_cache[1][:, :, :it_len[0], :]])
        #                        for I_kv_cache in kv_cache)
        # text_kv_cache = tuple(tuple([I_k_cache[:, :, it_len[0]:, :], I_v_cache[:, :, it_len[0]:, :]])
        #                       for I_k_cache, I_v_cache in kv_cache)
        for layer_idx, layer_cache in enumerate(kv_cache):
            # 直接切片，避免创建中间tuple
            ik = layer_cache[0][:, :, :it_len[0], :]
            iv = layer_cache[1][:, :, :it_len[0], :]
            tk = layer_cache[0][:, :, it_len[0]:, :]
            tv = layer_cache[1][:, :, it_len[0]:, :]

            # 压缩并立即使用
            compressed_ik, compressed_iv = self.compress_layer((ik, iv), layer_idx,
                                                               self.compress_ik, self.compress_iv)
            compressed_tk, compressed_tv = self.compress_layer((tk, tv), layer_idx,
                                                               self.compress_tk, self.compress_tv)

            # 合并并立即存储
            compressed_k = torch.concat([compressed_ik, compressed_tk], dim=2)
            compressed_v = torch.concat([compressed_iv, compressed_tv], dim=2)
            compressed_kv_cache.append((compressed_k, compressed_v))

            # # 手动删除中间变量
            # del ik, iv, tk, tv
            # del compressed_ik, compressed_iv, compressed_tk, compressed_tv

        # return compressed_kv_cache
        # return [(k.clone(), v.clone()) for k, v in compressed_kv_cache]
        # return [(k.detach().clone(memory_format=torch.contiguous_format), v.detach().clone(memory_format=torch.contiguous_format)) for k, v in compressed_kv_cache]
        return [(k.detach().contiguous(), v.detach().contiguous()) for k, v in compressed_kv_cache]
class KVCacheLinearDecoupleCompressor_Triton(nn.Module):
    def __init__(self, src_config, compression_factor=5, min_seq_len=1):
        super(KVCacheLinearDecoupleCompressor_Triton, self).__init__()

        # 假设 src_config 有 text_config 属性
        # 需要根据你的实际 LlamaConfig 结构调整
        if not hasattr(src_config, 'text_config'):
             # 如果 src_config 就是 LlamaConfig
             text_config = src_config
        else:
             text_config = src_config.text_config

        self.n_layer = text_config.num_hidden_layers
        # 确保 num_key_value_heads 存在，否则使用 num_attention_heads
        self.nhead = getattr(text_config, 'num_key_value_heads', text_config.num_attention_heads)
        if self.nhead == 0: # Handle potential config issues
             self.nhead = text_config.num_attention_heads
             print(f"Warning: num_key_value_heads is 0, falling back to num_attention_heads ({self.nhead})")

        self.d_model = text_config.hidden_size

        # 处理 d_model 不能被 nhead 整除的情况
        if self.nhead == 0:
             raise ValueError("Number of heads cannot be zero.")
        if self.d_model % self.nhead != 0:
            # Option 1: Raise error (safer)
            # raise ValueError(f"hidden_size {self.d_model} must be divisible by num_key_value_heads {self.nhead}")
            # Option 2: Adjust head_dim (might break downstream if not handled)
            print(f"Warning: hidden_size {self.d_model} is not divisible by num_key_value_heads {self.nhead}. Adjusting head_dim.")
            self.head_dim = self.d_model // self.nhead # Use integer division, might lose some dim info if not divisible
        else:
            self.head_dim = self.d_model // self.nhead

        self.compression_factor = compression_factor if compression_factor > 0 else 1 # Ensure > 0
        self.min_seq_len = min_seq_len

        # 定义特征维度
        # Ensure head_dim is calculated before use
        if self.head_dim <= 0:
             raise ValueError("Calculated head_dim is not positive.")
        input_feat1 = self.head_dim * self.compression_factor
        hidden_feat = self.head_dim
        output_feat = self.head_dim
        # 假设所有线性层都有 bias，如果不同，需要单独处理
        # TODO: Check src_config for bias flags if available, otherwise assume True
        bias_flag = getattr(text_config, 'attention_bias', True) # Example: check attention bias flag


        def create_compress_block():
            # --- 替换第一个 Linear + ReLU ---
            fused_layer1 = FusedLinearReLUTriton(input_feat1, hidden_feat, bias=bias_flag)
            # ---

            # 标准的后续层
            dropout1 = nn.Dropout(p=0.4)
            linear2 = nn.Linear(hidden_feat, hidden_feat, bias=bias_flag)
            relu2 = nn.ReLU()
            dropout2 = nn.Dropout(p=0.4)
            linear3 = nn.Linear(hidden_feat, output_feat, bias=bias_flag)

            # 构建序列
            layers = [
                fused_layer1, # 已融合 Linear + ReLU
                dropout1,
                linear2,
                relu2,
                dropout2,
                linear3
            ]
            # --- (可选) 如果想融合第二个 Linear+ReLU ---
            # fused_layer2 = FusedLinearReLUTriton(hidden_feat, hidden_feat, bias=bias_flag)
            # layers = [fused_layer1, dropout1, fused_layer2, dropout2, linear3]
            # ---

            return nn.Sequential(*layers)

        # 创建 ModuleLists
        self.compress_tk = nn.ModuleList([create_compress_block() for _ in range(self.n_layer)])
        self.compress_tv = nn.ModuleList([create_compress_block() for _ in range(self.n_layer)])
        self.compress_ik = nn.ModuleList([create_compress_block() for _ in range(self.n_layer)])
        self.compress_iv = nn.ModuleList([create_compress_block() for _ in range(self.n_layer)])

        # 这个 attention 层在 forward 中没有用到，如果确实不需要可以移除
        # self.attention = nn.ModuleList([nn.Linear(self.head_dim, 1, bias=bias_flag) for _ in range(self.n_layer)])

    def compress_layer(self, layer_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
                       layer_idx: int,
                       compressor_list_k: nn.ModuleList,
                       compressor_list_v: nn.ModuleList) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compresses a single layer's KV cache."""
        if layer_cache is None or layer_cache[0] is None or layer_cache[1] is None:
             return None, None
        k, v = layer_cache
        if k.numel() == 0 or v.numel() == 0: # Handle empty tensors
             return k, v

        # Validate shapes before proceeding
        if k.dim() != 4 or v.dim() != 4:
             print(f"Warning: Invalid shape for layer {layer_idx}. K: {k.shape}, V: {v.shape}. Skipping compression.")
             return k, v
        batch_size, nhead, seq_len, head_dim = k.shape

        if head_dim != self.head_dim:
             print(f"Warning: Head dimension mismatch in layer {layer_idx}. Expected {self.head_dim}, got {head_dim}. Skipping compression.")
             return k, v

        # Ensure compression factor is valid
        if self.compression_factor <= 1:
             return k, v

        # 计算压缩后的序列长度
        compressed_seq_len = seq_len // self.compression_factor

        # 检查压缩后的序列长度是否小于最小阈值
        if compressed_seq_len < self.min_seq_len:
            return k, v  # 如果压缩后长度过短，则返回原始序列

        # 计算需要压缩的序列长度
        compress_len = compressed_seq_len * self.compression_factor
        input_feature_dim = self.head_dim * self.compression_factor

        # 重塑key和value以进行压缩
        # Ensure slicing is valid
        if compress_len > seq_len:
             print(f"Warning: Calculated compress_len {compress_len} > seq_len {seq_len}. Adjusting.")
             compress_len = seq_len
             compressed_seq_len = seq_len // self.compression_factor # Recalculate based on adjustment
             if compressed_seq_len == 0: return k,v # Cannot compress if result is 0 len

        k_part_to_compress = k[:, :, :compress_len, :]
        v_part_to_compress = v[:, :, :compress_len, :]

        # Reshape safely
        try:
             k_to_compress = k_part_to_compress.reshape(batch_size, nhead, compressed_seq_len, input_feature_dim)
             v_to_compress = v_part_to_compress.reshape(batch_size, nhead, compressed_seq_len, input_feature_dim)
        except RuntimeError as e:
             print(f"Error during reshape in layer {layer_idx}: {e}. Shapes: k_part={k_part_to_compress.shape}, v_part={v_part_to_compress.shape}. Skipping compression.")
             return k, v # Return original if reshape fails

        # 压缩key和value (调用包含 Triton 层的 Sequential)
        try:
            compressed_k_part = compressor_list_k[layer_idx](k_to_compress)
            compressed_v_part = compressor_list_v[layer_idx](v_to_compress)
        except Exception as e:
            print(f"Error during compression call in layer {layer_idx}: {e}. Skipping compression.")
            # Potentially log input shapes for debugging
            # print(f"Input shapes to compressor: K={k_to_compress.shape}, V={v_to_compress.shape}")
            return k, v # Return original on error

        # 处理剩余的部分（如果有的话）
        final_k, final_v = compressed_k_part, compressed_v_part
        if seq_len > compress_len:
            remaining_k = k[:, :, compress_len:, :]
            remaining_v = v[:, :, compress_len:, :]
            # Ensure dimensions match for concatenation
            if compressed_k_part.shape[-1] == remaining_k.shape[-1]:
                 final_k = torch.cat([compressed_k_part, remaining_k], dim=2)
            else:
                 print(f"Warning: Dimension mismatch after compression in layer {layer_idx}. Cannot concatenate remaining part. Compressed K dim: {compressed_k_part.shape[-1]}, Remaining K dim: {remaining_k.shape[-1]}")
                 # Decide handling: return only compressed, return original, or raise error
                 # Returning only compressed part for now:
                 final_k = compressed_k_part


            if compressed_v_part.shape[-1] == remaining_v.shape[-1]:
                 final_v = torch.cat([compressed_v_part, remaining_v], dim=2)
            else:
                 print(f"Warning: Dimension mismatch after compression in layer {layer_idx}. Cannot concatenate remaining part. Compressed V dim: {compressed_v_part.shape[-1]}, Remaining V dim: {remaining_v.shape[-1]}")
                 final_v = compressed_v_part

        return final_k, final_v


    def forward(self, kv_cache: List[Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]],
                it_len: List[int] = [1, 1],
                strategy_com: str = "retain") -> List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Compresses the KV cache for each layer.

        Args:
            kv_cache: A list where each element is a tuple (key_tensor, value_tensor) for a layer,
                      or None, or a tuple containing None(s).
            it_len: A list containing [image_length, text_length_so_far].
            strategy_com: Compression strategy (currently unused in this logic).

        Returns:
            A list of compressed (key_tensor, value_tensor) tuples for each layer.
        """
        compressed_kv_cache = []

        if kv_cache is None or not isinstance(kv_cache, list):
             print("Warning: Invalid kv_cache input to KVCacheLinearDecoupleCompressor.forward. Returning empty list.")
             return []

        # Basic validation for it_len
        if not isinstance(it_len, list) or len(it_len) != 2 or not all(isinstance(x, int) and x >= 0 for x in it_len):
             print(f"Warning: Invalid it_len provided: {it_len}. Using default [1, 1].")
             it_len = [1, 1]

        # Get device from the first parameter found
        module_device = next(self.parameters()).device

        for layer_idx, layer_cache in enumerate(kv_cache):
            if layer_cache is None or not isinstance(layer_cache, tuple) or len(layer_cache) != 2:
                # Handle layers with invalid cache format
                compressed_kv_cache.append((None, None))
                continue

            k_full, v_full = layer_cache

            # Handle layers with None or empty tensors
            if k_full is None or v_full is None or k_full.numel() == 0 or v_full.numel() == 0:
                 compressed_kv_cache.append((k_full, v_full))
                 continue

            # Ensure tensors are on the correct device
            try:
                 k_full = k_full.to(module_device)
                 v_full = v_full.to(module_device)
            except Exception as e:
                 print(f"Error moving layer {layer_idx} cache to device {module_device}: {e}. Skipping compression for this layer.")
                 compressed_kv_cache.append(layer_cache) # Return original uncompressed cache for this layer
                 continue


            # --- Splitting based on it_len ---
            current_seq_len = k_full.shape[2]
            image_split_point = min(it_len[0], current_seq_len)
            text_start_point = image_split_point

            ik = k_full[:, :, :image_split_point, :] if image_split_point > 0 else None
            iv = v_full[:, :, :image_split_point, :] if image_split_point > 0 else None
            tk = k_full[:, :, text_start_point:, :] if text_start_point < current_seq_len else None
            tv = v_full[:, :, text_start_point:, :] if text_start_point < current_seq_len else None

            # --- Compression ---
            # Compress image part
            compressed_ik, compressed_iv = self.compress_layer((ik, iv), layer_idx, self.compress_ik, self.compress_iv)

            # Compress text part
            compressed_tk, compressed_tv = self.compress_layer((tk, tv), layer_idx, self.compress_tk, self.compress_tv)

            # --- Merging ---
            # Merge K
            if compressed_ik is not None and compressed_tk is not None:
                # Ensure dimensions match before cat (compress_layer should handle this, but double-check)
                 if compressed_ik.shape[-1] == compressed_tk.shape[-1]:
                      compressed_k = torch.cat([compressed_ik, compressed_tk], dim=2)
                 else:
                      print(f"Warning: K dimension mismatch during merge in layer {layer_idx}. IK: {compressed_ik.shape}, TK: {compressed_tk.shape}. Using only the larger part if possible or None.")
                      # Handle mismatch - e.g., return None or prioritize one part
                      compressed_k = None # Example: set to None on mismatch
            elif compressed_ik is not None:
                compressed_k = compressed_ik
            elif compressed_tk is not None:
                compressed_k = compressed_tk
            else:
                compressed_k = None

            # Merge V
            if compressed_iv is not None and compressed_tv is not None:
                 if compressed_iv.shape[-1] == compressed_tv.shape[-1]:
                     compressed_v = torch.cat([compressed_iv, compressed_tv], dim=2)
                 else:
                      print(f"Warning: V dimension mismatch during merge in layer {layer_idx}. IV: {compressed_iv.shape}, TV: {compressed_tv.shape}. Using only the larger part if possible or None.")
                      compressed_v = None # Example: set to None on mismatch
            elif compressed_iv is not None:
                compressed_v = compressed_iv
            elif compressed_tv is not None:
                compressed_v = compressed_tv
            else:
                compressed_v = None

            # Ensure contiguity and detach (if for inference)
            final_k = compressed_k.contiguous().detach() if compressed_k is not None else None
            final_v = compressed_v.contiguous().detach() if compressed_v is not None else None

            compressed_kv_cache.append((final_k, final_v))

        return compressed_kv_cache


    # def forward(self, kv_cache, it_len=[1 ,1], strategy_com="retain"):
    #     # kv_cache: list of n_layer tuples, each tuple contains (k, v)
    #     compressed_kv_cache = []
    #     image_kv_cache = tuple(tuple([I_kv_cache[0][: ,: ,:it_len[0] ,:] ,I_kv_cache[1][: ,: ,:it_len[0] ,:]])
    #                            for I_kv_cache in kv_cache)
    #     text_kv_cache = tuple(tuple([I_k_cache[:, :, it_len[0]:, :], I_v_cache[:, :, it_len[0]:, :]])
    #                           for I_k_cache, I_v_cache in kv_cache)
    #     if strategy_com == "retain":
    #         for layer_idx, layer_cache in enumerate(zip(image_kv_cache ,text_kv_cache)):
    #             compressed_ik, compressed_iv = self.compress_layer(layer_cache[0], layer_idx,self.compress_ik,self.compress_iv)
    #             # compressed_ik, compressed_iv = layer_cache[0]
    #             compressed_tk, compressed_tv = self.compress_layer(layer_cache[1], layer_idx,self.compress_tk,self.compress_tv)
    #             # compressed_tk, compressed_tv = layer_cache[1]
    #             compressed_k = torch.concat([compressed_ik ,compressed_tk] ,dim=2)
    #             compressed_v = torch.concat([compressed_iv ,compressed_tv] ,dim=2)
    #             compressed_kv_cache.append((compressed_k, compressed_v))
    #     else:
    #         for layer_idx, layer_cache in enumerate(zip(image_kv_cache ,text_kv_cache)):
    #             compressed_ik, compressed_iv = self.compress_layer(layer_cache[0], layer_idx ,self.compress_ik
    #                                                                ,self.compress_iv)
    #             compressed_tk, compressed_tv = self.compress_layer(layer_cache[1], layer_idx ,self.compress_tk
    #                                                                ,self.compress_tv)
    #             compressed_k = torch.concat([compressed_ik ,compressed_tk] ,dim=2)
    #             compressed_v = torch.concat([compressed_iv ,compressed_tv] ,dim=2)
    #             compressed_kv_cache.append((compressed_k, compressed_v))
    #
    #
    #     return compressed_kv_cache

    # def compress_layer(self, layer_cache, layer_idx, compressor_k, compressor_v):
    #     k, v = layer_cache
    #     batch_size, nhead, seq_len, head_dim = k.shape
    #     compressed_seq_len = seq_len // self.compression_factor
    #
    #     # 如果压缩后长度过短，直接返回原始序列
    #     if compressed_seq_len < self.min_seq_len:
    #         # return k.clone(), v.clone()
    #         return k, v
    #
    #     compress_len = compressed_seq_len * self.compression_factor
    #
    #     # 直接在原tensor上进行reshape，避免创建新的内存
    #     k_to_compress = k[:, :, :compress_len, :].view(batch_size, nhead, compressed_seq_len,
    #                                                    self.head_dim * self.compression_factor)
    #     v_to_compress = v[:, :, :compress_len, :].view(batch_size, nhead, compressed_seq_len,
    #                                                    self.head_dim * self.compression_factor)
    #
    #     # 压缩并立即存储结果
    #     compressed_k = compressor_k[layer_idx](k_to_compress)
    #     compressed_v = compressor_v[layer_idx](v_to_compress)
    #
    #     # 释放中间变量
    #     del k_to_compress, v_to_compress
    #
    #     # 只在需要处理剩余部分时才创建新的tensor
    #     if seq_len > compress_len:
    #         # 使用原始tensor的view而不是创建新的copy
    #         remaining_k = k[:, :, compress_len:, :]
    #         remaining_v = v[:, :, compress_len:, :]
    #
    #         # 连接并立即释放中间结果
    #         final_k = torch.cat([compressed_k, remaining_k], dim=2).clone()
    #         final_v = torch.cat([compressed_v, remaining_v], dim=2).clone()
    #
    #         del compressed_k, compressed_v
    #         del remaining_k, remaining_v
    #
    #         return final_k, final_v
    #
    #     return compressed_k.clone(), compressed_v.clone()



class CrossModelKVCompressor(nn.Module):
    def __init__(self, src_config, tgt_config, compression_factor=2, device=None):
        super(CrossModelKVCompressor, self).__init__()
        self.src_layers = src_config.num_hidden_layers
        self.tgt_layers = tgt_config.num_hidden_layers
        self.src_heads = src_config.num_key_value_heads
        self.tgt_heads = tgt_config.num_key_value_heads
        self.src_dim = src_config.hidden_size // self.src_heads
        self.tgt_dim = 64
        self.src_dim = src_config.hidden_size // self.src_heads
        self.tgt_dim = 64
        # self.tgt_dim = tgt_config.hidden_size // self.tgt_heads
        self.compression_factor = compression_factor
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 可学习的层映射网络
        self.layer_map_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.src_layers)
        ).to(self.device)

        # 创建压缩和维度调整层
        self.compressors = nn.ModuleList([
            self._create_compressor(i).to(self.device) for i in range(self.tgt_layers)
        ])

        self.to(self.device)

    def _create_layer_map(self, tgt_layer_idx):
        # 将目标层索引转换为张量并归一化
        tgt_layer_tensor = torch.tensor([[tgt_layer_idx / self.tgt_layers]], dtype=torch.float32, device=self.device)
        # 使用神经网络生成源层的权重
        src_layer_weights = self.layer_map_net(tgt_layer_tensor).squeeze()
        # 使用 softmax 将权重转换为概率分布
        src_layer_probs = F.softmax(src_layer_weights, dim=0)
        return src_layer_probs

    def _create_compressor(self, layer_idx):
        return nn.Sequential(
            nn.Linear(self.src_dim * self.compression_factor, self.src_dim),
            nn.ReLU(),
            nn.Linear(self.src_dim, self.tgt_dim)
        )

    def forward(self, src_kv_cache):
        adapted_kv_cache = []
        for tgt_layer in range(self.tgt_layers):
            src_layer_probs = self._create_layer_map(tgt_layer)

            # 使用加权和来组合源层的 KV 缓存
            k_combined = torch.zeros_like(src_kv_cache[0][0], device=self.device)
            v_combined = torch.zeros_like(src_kv_cache[0][1], device=self.device)
            for src_layer, prob in enumerate(src_layer_probs):
                k, v = src_kv_cache[src_layer]
                k, v = k.to(self.device), v.to(self.device)
                k_combined += prob * k
                v_combined += prob * v

            # 压缩序列长度
            seq_len = k_combined.size(2)
            compressed_len = seq_len // self.compression_factor
            remainder = seq_len % self.compression_factor

            # 压缩主要部分
            k_main = k_combined[:, :, :compressed_len * self.compression_factor, :]
            v_main = v_combined[:, :, :compressed_len * self.compression_factor, :]

            k_main = k_main.view(k_main.size(0), self.src_heads, compressed_len, -1)
            v_main = v_main.view(v_main.size(0), self.src_heads, compressed_len, -1)

            # 应用压缩器
            k_main = self.compressors[tgt_layer](k_main)
            v_main = self.compressors[tgt_layer](v_main)

            # 处理余数部分
            if remainder > 0:
                k_remainder = k_combined[:, :, -remainder:, :]
                v_remainder = v_combined[:, :, -remainder:, :]

                # 对余数部分应用简单的线性压缩
                k_remainder = k_remainder.view(k_remainder.size(0), self.src_heads, 1, -1)
                v_remainder = v_remainder.view(v_remainder.size(0), self.src_heads, 1, -1)

                k_remainder = self.compressors[tgt_layer](k_remainder)
                v_remainder = self.compressors[tgt_layer](v_remainder)

                # 合并主要部分和余数部分
                k = torch.cat([k_main, k_remainder], dim=2)
                v = torch.cat([v_main, v_remainder], dim=2)
            else:
                k, v = k_main, v_main

            # 调整头数
            k = k.view(k.size(0), self.tgt_heads, -1, self.tgt_dim)
            v = v.view(v.size(0), self.tgt_heads, -1, self.tgt_dim)

            adapted_kv_cache.append((k, v))

        return adapted_kv_cache

    def to(self, device):
        # 重写 to 方法以更新 self.device
        self.device = device
        return super().to(device)

class CrossModelRealKVCompressor(nn.Module):
    def __init__(self, src_config, tgt_config, compression_factor=2, device=None):
        super(CrossModelRealKVCompressor, self).__init__()
        self.src_layers = src_config.num_hidden_layers
        self.tgt_layers = tgt_config.num_hidden_layers
        self.src_heads = src_config.num_key_value_heads
        self.tgt_heads = tgt_config.num_key_value_heads
        self.heads_ratio = self.src_heads // self.tgt_heads
        self.src_dim = src_config.hidden_size // self.src_heads
        self.tgt_dim = 64
        # self.tgt_dim = tgt_config.hidden_size // self.tgt_heads
        self.compression_factor = compression_factor
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 可学习的层映射网络
        self.layer_map_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.src_layers)
        ).to(self.device)

        # 创建压缩和维度调整层
        self.compressors = nn.ModuleList([
            self._create_compressor(i).to(self.device) for i in range(self.tgt_layers)
        ])

        self.to(self.device)

    def _create_layer_map(self, tgt_layer_idx):
        # 将目标层索引转换为张量并归一化
        tgt_layer_tensor = torch.tensor([[tgt_layer_idx / self.tgt_layers]], dtype=torch.float32, device=self.device)
        # 使用神经网络生成源层的权重
        src_layer_weights = self.layer_map_net(tgt_layer_tensor).squeeze()
        # 使用 softmax 将权重转换为概率分布
        src_layer_probs = F.softmax(src_layer_weights, dim=0)
        return src_layer_probs

    def _create_compressor(self, layer_idx):
        return nn.Sequential(
            nn.Linear(self.src_dim * self.compression_factor * self.heads_ratio, self.src_dim),
            nn.ReLU(),
            nn.Linear(self.src_dim, self.tgt_dim)
        )

    def forward(self, src_kv_cache):
        adapted_kv_cache = []
        for tgt_layer in range(self.tgt_layers):
            src_layer_probs = self._create_layer_map(tgt_layer)

            # 使用加权和来组合源层的 KV 缓存
            k_combined = torch.zeros_like(src_kv_cache[0][0], device=self.device)
            v_combined = torch.zeros_like(src_kv_cache[0][1], device=self.device)
            for src_layer, prob in enumerate(src_layer_probs):
                k, v = src_kv_cache[src_layer]
                k, v = k.to(self.device), v.to(self.device)
                k_combined += prob * k
                v_combined += prob * v

            # 压缩序列长度
            seq_len = k_combined.size(2)
            compressed_len = seq_len // self.compression_factor
            remainder = seq_len % self.compression_factor

            # 压缩主要部分
            k_main = k_combined[:, :, :compressed_len * self.compression_factor, :]
            v_main = v_combined[:, :, :compressed_len * self.compression_factor, :]

            # k_main = k_main.view(k_main.size(0), self.tgt_heads, compressed_len, -1)
            # v_main = v_main.view(v_main.size(0), self.tgt_heads, compressed_len, -1)

            k_main = k_main.reshape(k_main.size(0), self.tgt_heads, compressed_len, -1)
            v_main = v_main.reshape(v_main.size(0), self.tgt_heads, compressed_len, -1)

            # 应用压缩器
            k_main = self.compressors[tgt_layer](k_main)
            v_main = self.compressors[tgt_layer](v_main)

            # 处理余数部分
            if remainder > 0:
                k_remainder = k_combined[:, :, -remainder:, :]
                v_remainder = v_combined[:, :, -remainder:, :]

                # 对余数部分应用简单的线性压缩
                k_remainder = k_remainder.view(k_remainder.size(0), self.tgt_heads, 1, -1)
                v_remainder = v_remainder.view(v_remainder.size(0), self.tgt_heads, 1, -1)

                k_remainder = self.compressors[tgt_layer](k_remainder)
                v_remainder = self.compressors[tgt_layer](v_remainder)

                # 合并主要部分和余数部分
                k = torch.cat([k_main, k_remainder], dim=2)
                v = torch.cat([v_main, v_remainder], dim=2)
            else:
                k, v = k_main, v_main

            # 调整头数
            k = k.view(k.size(0), self.tgt_heads, -1, self.tgt_dim)
            v = v.view(v.size(0), self.tgt_heads, -1, self.tgt_dim)

            adapted_kv_cache.append((k, v))

        return adapted_kv_cache

    def to(self, device):
        # 重写 to 方法以更新 self.device
        self.device = device
        return super().to(device)

class KVCacheCompressionLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, epsilon=1e-8, label_strategy="logit"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = 1 - alpha - beta
        self.epsilon = epsilon  # 防止log(0)
        self.label_strategy = label_strategy
        self.mse_loss = nn.MSELoss()
        self.cos_loss = nn.CosineSimilarity(dim=-1)
        self.high_weight_count = 30
        self.low_weight = 1
        self.high_weight = 3
        self.p_threshold = 0.9
        # self.num_prompt_end_tokens = 5
        # self.num_random_tokens = 500

    def forward(self, mid_probs, output_probs, pos_len=[0 ,0 ,0]):
        # 确保输入是概率分布
        # assert torch.allclose(mid_probs.sum(dim=-1), torch.ones_like(mid_probs.sum(dim=-1)))
        # assert torch.allclose(output_probs.sum(dim=-1), torch.ones_like(output_probs.sum(dim=-1)))

        if self.label_strategy == "logit":

            # 1. 交叉熵损失
            ce_loss = -torch.sum(mid_probs * torch.log(output_probs + self.epsilon), dim=-1).mean()

            # 2. JS散度
            m = 0.5 * (mid_probs + output_probs)
            js_div = 0.5 * (
                    torch.sum(mid_probs * torch.log(mid_probs / m + self.epsilon), dim=-1) +
                    torch.sum(output_probs * torch.log(output_probs / m + self.epsilon), dim=-1)
            ).mean()

            # 3. L1距离
            l1_loss = torch.abs(mid_probs - output_probs).sum(dim=-1).mean()

            # 组合损失
            total_loss = self.alpha * ce_loss + self.beta * js_div + (1 - self.alpha - self.beta) * l1_loss
            # total_loss = ce_loss

            return total_loss
        elif self.label_strategy == "label":

            mid_labels = logits_to_onehot(mid_probs)

            # 1. 交叉熵损失
            ce_loss = -torch.sum(mid_labels * torch.log(output_probs + self.epsilon), dim=-1).mean()

            # 2. JS散度
            m = 0.5 * (mid_probs + output_probs)
            js_div = 0.5 * (
                    torch.sum(mid_probs * torch.log(mid_probs / m + self.epsilon), dim=-1) +
                    torch.sum(output_probs * torch.log(output_probs / m + self.epsilon), dim=-1)
            ).mean()

            # 3. L1距离
            l1_loss = torch.abs(mid_probs - output_probs).sum(dim=-1).mean()

            # 组合损失
            total_loss = self.alpha * ce_loss + self.beta * js_div + (1 - self.alpha - self.beta) * l1_loss
            # total_loss = ce_loss

            return total_loss

        elif self.label_strategy == "label_twice":

            mid_labels = logits_to_onehot(mid_probs)
            # mid_labels = logits_to_onehot(mid_probs[:,:pos_len[0],:])

            # 1. 交叉熵损失
            ce_loss = -torch.sum(mid_labels * torch.log(output_probs + self.epsilon), dim=-1).mean()
            # ce_loss = -torch.sum(mid_labels * torch.log(output_probs[:,:pos_len[0],:] + self.epsilon), dim=-1).mean()

            # 2. JS散度
            m = 0.5 * (mid_probs + output_probs)
            js_div = 0.5 * (
                    torch.sum(mid_probs * torch.log(mid_probs / m + self.epsilon), dim=-1) +
                    torch.sum(output_probs * torch.log(output_probs / m + self.epsilon), dim=-1)
            ).mean()

            # 3. L1距离
            l1_loss = torch.abs(mid_probs - output_probs).sum(dim=-1).mean()

            # 组合损失
            total_loss = self.alpha * ce_loss + self.beta * js_div + (1 - self.alpha - self.beta) * l1_loss
            # total_loss = ce_loss

            return total_loss

        elif self.label_strategy == "label_confident":

            # 1. 交叉熵损失

            max_probs, teacher_pred = torch.max(mid_probs, dim=-1)

            mid_labels = logits_to_onehot(mid_probs)

            # 1. 交叉熵损失
            ce_loss = -torch.sum(mid_labels * torch.log(output_probs + self.epsilon), dim=-1).mean()

            # 2. JS散度
            m = 0.5 * (mid_probs + output_probs)
            js_div = 0.5 * (
                    torch.sum(mid_probs * torch.log(mid_probs / m + self.epsilon), dim=-1) +
                    torch.sum(output_probs * torch.log(output_probs / m + self.epsilon), dim=-1)
            ).mean()

            # 3. L1距离
            l1_loss = torch.abs(mid_probs - output_probs).sum(dim=-1).mean()

            mean_confidence = max_probs.mean().item()

            # 组合损失
            total_loss = ( 1 -self.gamma ) * \
                        (mean_confidence * ce_loss + ( 1 -mean_confidence) * js_div) + self.gamma * l1_loss
            return total_loss

    def indices_to_mask(self, indices, hidden_dim, mask=None):
        """
        创建一个3D mask并根据indices在指定位置的特定hidden维度上填充True

        参数:
            indices: torch.LongTensor, 形状为 [batch_size, seq_len] 的位置索引张量
                    其中的值表示要在hidden_dim维度上设置为True的位置
            hidden_dim: int, 隐藏层维度大小

        返回:
            torch.BoolTensor, 形状为 [batch_size, seq_len, hidden_dim] 的mask张量,
            默认全False，在indices[b,s]=h时，mask[b,s,h]=True
        """
        batch_size, seq_len = indices.size()

        # 创建全0的3D mask
        if mask is None:
            mask = torch.zeros(batch_size, seq_len, hidden_dim, dtype=torch.bool, device=indices.device)

        # 创建用于scatter_的索引张量
        batch_idx = torch.arange(batch_size, device=indices.device).unsqueeze(1).expand(-1, seq_len)
        seq_idx = torch.arange(seq_len, device=indices.device).unsqueeze(0).expand(batch_size, -1)

        # 使用scatter_将True填充到指定位置
        mask[batch_idx, seq_idx, indices] = True

        return mask

    def get_top_p_mask(self, probs):
        """获取累积概率超过p_threshold的token位置"""
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # 找到累积概率首次超过阈值的位置
        mask = cumulative_probs <= self.p_threshold
        # # 确保至少选择一个token
        # mask[:, :, 0] = True
        threshold_positions = torch.min(mask.sum(dim=-1) + 1, torch.tensor(probs.size(-1)))-1
        # original_indices = sorted_indices[:, :, threshold_positions]
        mask = self.indices_to_mask(threshold_positions, mask.shape[-1], mask)
        # thresh_mask = torch.zeros_like(probs, dtype=torch.bool)
        # batch_indices = torch.arange(probs.shape[0])
        # thresh_mask[batch_indices, threshold_positions] = True

        # 创建原始词表大小的mask
        final_mask = torch.zeros_like(probs, dtype=torch.bool)
        final_mask.scatter_(-1, sorted_indices, mask)

        return final_mask

    def forward_attn(self, original_twice_outputs_logits, compress_output_logits, pos_len,
                     orig_hidden_states, compress_hidden_states, attention_mask=None):
        # mid_labels = logits_to_onehot(original_twice_outputs_logits[:, :(pos_len[0]+pos_len[1]), :])

        top_p_mask = self.get_top_p_mask(original_twice_outputs_logits[:, :(pos_len[0]+pos_len[1]), :])
        # 计算在top-p范围内的交叉熵损失
        masked_base_probs = original_twice_outputs_logits[:, :(pos_len[0]+pos_len[1]), :] * top_p_mask.float()
        mid_labels = masked_base_probs / (masked_base_probs.sum(dim=-1, keepdim=True) + self.epsilon)

        # mid_labels = logits_to_onehot(original_twice_outputs_logits[:,:pos_len[0],:])

        # 1. 交叉熵损失
        ce_loss = -torch.sum(mid_labels * torch.log(compress_output_logits[:, :(pos_len[0]+pos_len[1]), :] + self.epsilon), dim=-1).mean()
        # ce_loss = -torch.sum(mid_labels * torch.log(compress_output_logits[:,:pos_len[0],:] + self.epsilon), dim=-1).mean()

        # 2. JS散度
        m = 0.5 * (original_twice_outputs_logits + compress_output_logits)
        js_div = 0.5 * (
                torch.sum(original_twice_outputs_logits * torch.log(original_twice_outputs_logits / (m + self.epsilon) + self.epsilon), dim=-1) +
                torch.sum(compress_output_logits * torch.log(compress_output_logits / (m + self.epsilon) + self.epsilon), dim=-1)
        ).mean()

        # 3. L1距离
        l1_loss = torch.abs(original_twice_outputs_logits - compress_output_logits).sum(dim=-1).mean()

        layer_loss = 0

        for layer_idx,(orig_states,comp_states) in enumerate(zip(orig_hidden_states, compress_hidden_states)):
            if layer_idx == 0:
                continue
            # MSE损失
            mseLoss = self.mse_loss(comp_states, orig_states)
            # mseLoss=0
            orig_states = torch.softmax(orig_states, dim=-1)
            comp_states = torch.softmax(comp_states, dim=-1)


            # 余弦相似度损失 (1 - 相似度，使得相似度趋近1)
            cosLoss = 1 - self.cos_loss(
                comp_states.view(-1, comp_states.size(-1)),
                orig_states.view(-1, orig_states.size(-1))
            ).mean()
            # klLoss = F.kl_div(
            #     F.log_softmax(comp_states, dim=-1),
            #     F.softmax(orig_states, dim=-1),
            #     reduction='batchmean'
            # )
            # m = 0.5 * (orig_states + comp_states)
            # js_state_loss = 0.5 * (
            #         torch.sum(orig_states * torch.log(orig_states / (m + self.epsilon) + self.epsilon), dim=-1) +
            #         torch.sum(comp_states * torch.log(comp_states / (m + self.epsilon) + self.epsilon), dim=-1)
            # ).mean()
            js_state_loss= 0

            # layer_loss += (mseLoss+cosLoss+klLoss)
            layer_loss += (mseLoss+cosLoss+js_state_loss)
        layer_loss /= 2

        # 组合损失
        total_loss = 0.5 *(self.alpha * ce_loss + self.beta * js_div +
                    (1 - self.alpha - self.beta) * l1_loss) + 0.5 * layer_loss
        return total_loss

    # def forward_kv(self, probs_original, probs_compressed, orig_kv_cache, compressed_kv_cache, hidden_states,
    #                generated_len, q_proj, attention_mask=None):
    #     assert torch.allclose(probs_original.sum(dim=-1), torch.ones_like(probs_original.sum(dim=-1)))
    #     assert torch.allclose(probs_compressed.sum(dim=-1), torch.ones_like(probs_compressed.sum(dim=-1)))
    #     if self.label_strategy == "label":
    #         mid_labels = logits_to_onehot(probs_original)
    #
    #         # 1. 交叉熵损失
    #         ce_loss = -torch.sum(mid_labels * torch.log(probs_compressed + self.epsilon), dim=-1).mean()
    #
    #         # 2. JS散度
    #         m = 0.5 * (probs_original + probs_compressed)
    #         js_div = 0.5 * (
    #                 torch.sum(probs_original * torch.log(probs_original / m + self.epsilon), dim=-1) +
    #                 torch.sum(probs_compressed * torch.log(probs_compressed / m + self.epsilon), dim=-1)
    #         ).mean()
    #
    #         # 3. L1距离
    #         l1_loss = torch.abs(probs_original - probs_compressed).sum(dim=-1).mean()
    #
    #         # 获取选中的queries
    #         # hidden_states_cvt = self.cvt_hidden_states(hidden_states)
    #         query_states = q_proj(hidden_states)
    #         # # 获取选中的queries
    #         # selected_queries, selected_positions = self.get_selected_queries(query_states, generated_len)
    #
    #         # 分别解包原始和压缩后的KV Cache
    #         orig_key, orig_value = orig_kv_cache
    #         compressed_key, compressed_value = compressed_kv_cache
    #
    #         # 如果有attention mask，需要选择对应的行
    #         if attention_mask is not None:
    #             attention_mask = attention_mask[:, selected_positions, :]
    #
    #         # 计算原始和压缩后的attention patterns
    #         orig_attention = self.compute_attention_patterns(
    #             selected_queries, orig_key, orig_value, attention_mask
    #         )
    #         compressed_attention = self.compute_attention_patterns(
    #             selected_queries, compressed_key, compressed_value, attention_mask
    #         )
    #
    #         # structure_loss = self.compute_structure_loss(kv_original, kv_compressed)
    #
    #         # # 3. 注意力一致性损失
    #         # attn_loss = self.compute_attention_consistency(
    #         #     kv_original, kv_compressed)
    #         #
    #         # # 4. 动态权重
    #         # weights = self.get_dynamic_weights(self.current_epoch)
    #
    #         return weights['ce'] * ce_loss + \
    #                weights['js'] * js_div + \
    #                weights['l1'] * l1_loss + \
    #                weights['structure'] * structure_loss + \
    #                weights['attn'] * attn_loss
    #
    #         # # 组合损失
    #         # total_loss = self.alpha * ce_loss + self.beta * js_div + (1 - self.alpha - self.beta) * l1_loss
    #         # # total_loss = ce_loss
    #         #
    #         # return total_loss

    def cvt_hidden_states(self, hidden_states):
        """
        Convert hidden states from format:
        tuple:seq_len(tuple:layer_num(batch_size, 1, hidden_dim))
        to:
        tuple:layer_num(batch_size, seq_len, hidden_dim)

        Args:
            hidden_states: Original hidden states in sequence-first format

        Returns:
            Tuple of tensors in layer-first format
        """
        # Get dimensions
        seq_len = len(hidden_states)
        num_layers = len(hidden_states[0])
        batch_size = hidden_states[0][0].shape[0]
        hidden_dim = hidden_states[0][0].shape[-1]

        # Initialize list to store results
        converted_states = []

        # For each layer
        for layer_idx in range(num_layers):
            # Gather all sequences for this layer
            layer_tensors = []
            for seq_idx in range(seq_len):
                # Get the tensor for this sequence position and layer
                # Remove the singleton dimension (1) in the middle
                tensor = hidden_states[seq_idx][layer_idx]
                layer_tensors.append(tensor)

            # Stack all sequences for this layer
            layer_combined = torch.concat(layer_tensors, dim=1)  # dim=1 for sequence dimension
            converted_states.append(layer_combined)

        return tuple(converted_states)

    # def compute_structure_loss(self, kv_orig, kv_comp):
    #     # 计算KV Cache的结构相似性
    #     orig_gram = torch.matmul(kv_orig, kv_orig.transpose(-2, -1))
    #     comp_gram = torch.matmul(kv_comp, kv_comp.transpose(-2, -1))
    #     return F.mse_loss(orig_gram, comp_gram)
    #
    # def compute_attention_consistency(self, kv_orig, kv_comp):
    #     # 确保压缩前后注意力模式相似
    #     attn_orig = self.compute_attention(kv_orig)
    #     attn_comp = self.compute_attention(kv_comp)
    #     return F.kl_div(
    #         F.log_softmax(attn_comp, dim=-1),
    #         F.softmax(attn_orig, dim=-1)
    #     )

    def compute_attention_patterns(self, query, key, value, attention_mask=None):
        """
        计算attention patterns, 考虑多头注意力机制
        Args:
            query: [batch_size, num_queries, num_heads, head_dim]
            key: [batch_size, seq_len, num_heads, head_dim]
            value: [batch_size, seq_len, num_heads, head_dim]
            attention_mask: [batch_size, num_queries, seq_len]
        """
        # 调整维度以进行批量注意力计算
        # [batch_size, num_heads, num_queries, head_dim]
        query = query.transpose(1, 2)
        # [batch_size, num_heads, seq_len, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # 计算attention scores
        # [batch_size, num_heads, num_queries, seq_len]
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(query.size(-1))

        if attention_mask is not None:
            # 扩展attention mask以适应多头
            attention_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, num_queries, seq_len]
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        # 应用softmax获得attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        return attention_weights

    def loss_weight(self, mid_probs, output_probs):
        # 确保输入是概率分布
        assert torch.allclose(mid_probs.sum(dim=-1), torch.ones_like(mid_probs.sum(dim=-1)))
        assert torch.allclose(output_probs.sum(dim=-1), torch.ones_like(output_probs.sum(dim=-1)))

        if self.label_strategy == "logit":

            # 1. 交叉熵损失
            ce_loss = -torch.sum(mid_probs * torch.log(output_probs + self.epsilon), dim=-1).mean()

            # 2. JS散度
            m = 0.5 * (mid_probs + output_probs)
            js_div = 0.5 * (
                    torch.sum(mid_probs * torch.log(mid_probs / m + self.epsilon), dim=-1) +
                    torch.sum(output_probs * torch.log(output_probs / m + self.epsilon), dim=-1)
            ).mean()

            # 3. L1距离
            l1_loss = torch.abs(mid_probs - output_probs).sum(dim=-1).mean()

            # 组合损失
            total_loss = self.alpha * ce_loss + self.beta * js_div + (1 - self.alpha - self.beta) * l1_loss
            # total_loss = ce_loss

            return total_loss
        elif self.label_strategy == "label":

            mid_labels = logits_to_onehot(mid_probs)
            seq_len = output_probs.shape[1]
            weights = torch.full((output_probs.shape), self.low_weight, device=output_probs.device)
            weights[:, :min(self.high_weight_count, seq_len)] = self.high_weight

            # 1. 交叉熵损失
            ce_loss = -torch.sum(mid_labels * torch.log(output_probs + self.epsilon) * weights, dim=-1).mean()

            # 2. JS散度
            m = 0.5 * (mid_probs + output_probs)
            js_div = 0.5 * (
                    torch.sum(mid_probs * torch.log(mid_probs / m + self.epsilon), dim=-1) +
                    torch.sum(output_probs * torch.log(output_probs / m + self.epsilon), dim=-1)
            ).mean()

            # 3. L1距离
            l1_loss = torch.abs(mid_probs - output_probs).sum(dim=-1).mean()

            # 组合损失
            total_loss = self.alpha * ce_loss + self.beta * js_div + (1 - self.alpha - self.beta) * l1_loss
            # total_loss = ce_loss

            return total_loss

    def loss_gen(self, mid_probs, output_probs, gen_output_probs):
        # 确保输入是概率分布
        assert torch.allclose(mid_probs.sum(dim=-1), torch.ones_like(mid_probs.sum(dim=-1)))
        assert torch.allclose(output_probs.sum(dim=-1), torch.ones_like(output_probs.sum(dim=-1)))
        assert torch.allclose(gen_output_probs.sum(dim=-1), torch.ones_like(gen_output_probs.sum(dim=-1)))

        if self.label_strategy == "logit":

            # 1. 交叉熵损失
            ce_loss = -torch.sum(mid_probs * torch.log(output_probs + self.epsilon), dim=-1).mean()

            # 2. JS散度
            m = 0.5 * (mid_probs + output_probs)
            js_div = 0.5 * (
                    torch.sum(mid_probs * torch.log(mid_probs / m + self.epsilon), dim=-1) +
                    torch.sum(output_probs * torch.log(output_probs / m + self.epsilon), dim=-1)
            ).mean()

            # 3. L1距离
            l1_loss = torch.abs(mid_probs - output_probs).sum(dim=-1).mean()

            # 组合损失
            total_loss = self.alpha * ce_loss + self.beta * js_div + (1 - self.alpha - self.beta) * l1_loss
            # total_loss = ce_loss

            return total_loss
        elif self.label_strategy == "label":

            mid_gen_label = mid_labels = logits_to_onehot(mid_probs)

            if gen_output_probs.shape[1] < mid_gen_label.shape[1]:
                mid_gen_label = mid_gen_label[:, :gen_output_probs.shape[1]]
            # 计算 logits 是否相等的布尔张量
            is_equal = (torch.argmax(mid_gen_label, dim=-1) == torch.argmax(gen_output_probs, dim=-1))

            # 找到第一个不匹配的位置
            mismatch_indices = torch.argmin(is_equal.long(), dim=1)

            # 检查是否存在完全匹配的序列，将其索引设为 -1
            perfect_match = is_equal.all(dim=1)
            mismatch_indices[perfect_match] = -1
            ff_num = max(mismatch_indices)
            if ff_num != -1:
                mid_gen_label = mid_gen_label[:, :(ff_num + 1), :]
                gen_output_probs = gen_output_probs[:, :(ff_num + 1), :]

            # 1. 交叉熵损失
            ce_loss1 = -torch.sum(mid_labels * torch.log(output_probs + self.epsilon), dim=-1).mean()
            ce_loss2 = -torch.sum(mid_gen_label * torch.log(gen_output_probs + self.epsilon), dim=-1).mean()

            # 2. JS散度
            m = 0.5 * (mid_probs + output_probs + self.epsilon)
            js_div1 = 0.5 * (
                    torch.sum(mid_probs * torch.log(mid_probs / m + self.epsilon), dim=-1) +
                    torch.sum(output_probs * torch.log(output_probs / m + self.epsilon), dim=-1)
            ).mean()

            # m = 0.5 * (mid_gen_label + gen_output_probs + self.epsilon)
            # js_div2 = 0.5 * (
            #         torch.sum(mid_gen_label * torch.log(mid_gen_label / m + self.epsilon), dim=-1) +
            #         torch.sum(gen_output_probs * torch.log(gen_output_probs / m + self.epsilon), dim=-1)
            # ).mean()

            # 3. L1距离
            l1_loss1 = torch.abs(mid_probs - output_probs).sum(dim=-1).mean()
            l1_loss2 = torch.abs(mid_gen_label - gen_output_probs).sum(dim=-1).mean()

            # 组合损失
            weight1 = 0.3
            weight2 = 1 - weight1
            # total_loss = self.alpha * (weight1 * ce_loss1 + weight2 * ce_loss2) + \
            #                     self.beta * (weight1 * js_div1 + weight2 * js_div2) + \
            #              (1 - self.alpha - self.beta) * (weight1 * l1_loss1 + weight2 * l1_loss2)
            total_loss = self.alpha * (weight1 * ce_loss1 + weight2 * ce_loss2) + \
                         self.beta * (weight1 * js_div1 + weight2) + \
                         (1 - self.alpha - self.beta) * (weight1 * l1_loss1 + weight2 * l1_loss2)
            # total_loss = ce_loss

            return total_loss

    def calc_metrics(self, mid_probs, output_probs):
        accuracy = (torch.argmax(mid_probs, dim=-1) == torch.argmax(output_probs, dim=-1)).float().mean()
        return accuracy, torch.argmax(mid_probs, dim=-1), torch.argmax(output_probs, dim=-1)
        # return accuracy

    def calc_metrics2(self, mid_probs, output_probs, gen_output_probs):
        is_equal1 = (torch.argmax(mid_probs, dim=-1) == torch.argmax(output_probs, dim=-1))
        mid_gen_label = mid_probs
        if gen_output_probs.shape[1] < mid_probs.shape[1]:
            mid_gen_label = mid_probs[:, :gen_output_probs.shape[1]]
        is_equal2 = (torch.argmax(mid_gen_label, dim=-1) == torch.argmax(gen_output_probs, dim=-1))
        accuracy1 = is_equal1.float().mean()
        accuracy2 = is_equal2.float().mean()
        ff_tensor1 = count_true_segments_with_zeros(is_equal1.reshape(-1))
        ff_tensor2 = count_true_segments_with_zeros(is_equal2.reshape(-1))
        avg_ff1 = float(sum(ff_tensor1)) / (len(ff_tensor1) + 1e-6)
        avg_ff2 = float(sum(ff_tensor2)) / (len(ff_tensor2) + 1e-6)

        # # 找到第一个不匹配的位置
        # mismatch_indices1 = torch.argmin(is_equal1.long(), dim=1)
        # # 检查是否存在完全匹配的序列，将其索引设为 -1
        # perfect_match1 = is_equal1.all(dim=1)
        # mismatch_indices1[perfect_match1] = output_probs.shape[1]
        # firstFalse1 = min(mismatch_indices1)
        #
        # # 找到第一个不匹配的位置
        # mismatch_indices2 = torch.argmin(is_equal2.long(), dim=1)
        # # 检查是否存在完全匹配的序列，将其索引设为 -1
        # perfect_match2 = is_equal2.all(dim=1)
        # mismatch_indices2[perfect_match2] = output_probs.shape[1]
        # firstFalse2 = min(mismatch_indices2)

        return accuracy1, accuracy2, avg_ff1, avg_ff2, torch.argmax(mid_probs, dim=-1), torch.argmax(output_probs,
                                                                                                     dim=-1), torch.argmax(
            gen_output_probs, dim=-1)
        # return accuracy

    def calc_metrics3(self, mid_probs, output_probs):
        is_equal = (torch.argmax(mid_probs, dim=-1) == torch.argmax(output_probs, dim=-1))
        accuracy1 = is_equal.float().mean()
        # # 找到第一个不匹配的位置
        # mismatch_indices = torch.argmin(is_equal.long(), dim=1)
        #
        # # 检查是否存在完全匹配的序列，将其索引设为 -1
        # perfect_match = is_equal.all(dim=1)
        # mismatch_indices[perfect_match] = output_probs.shape[1]
        # firstFalse = min(mismatch_indices)
        ff_tensor1 = count_true_segments_with_zeros(is_equal.reshape(-1))
        avg_ff1 = float(sum(ff_tensor1)) / len(ff_tensor1)
        mid_labels = logits_to_onehot(mid_probs)
        ce_loss = -torch.sum(mid_labels * torch.log(output_probs + self.epsilon), dim=-1).mean()
        pp = torch.exp(ce_loss)

        return accuracy1, avg_ff1, pp, torch.argmax(mid_probs, dim=-1), torch.argmax(output_probs, dim=-1)
        # return accuracy

    def calc_metrics4(self, mid_probs, output_probs, dmodel_probs):
        is_equal = (torch.argmax(mid_probs, dim=-1) == torch.argmax(output_probs, dim=-1))
        is_equal2 = (torch.argmax(mid_probs, dim=-1) == torch.argmax(dmodel_probs, dim=-1))
        is_equal3 = (is_equal | is_equal2)
        accuracy1 = is_equal.float().mean()
        accuracy2 = is_equal2.float().mean()
        accuracy3 = is_equal3.float().mean()
        ff_tensor1 = count_true_segments_with_zeros(is_equal.reshape(-1))
        ff_tensor2 = count_true_segments_with_zeros(is_equal2.reshape(-1))
        ff_tensor3 = count_true_segments_with_zeros(is_equal3.reshape(-1))
        # # 找到第一个不匹配的位置
        # mismatch_indices = torch.argmin(is_equal.long(), dim=1)
        #
        # # 检查是否存在完全匹配的序列，将其索引设为 -1
        # perfect_match = is_equal.all(dim=1)
        # mismatch_indices[perfect_match] = output_probs.shape[1]
        # firstFalse = min(mismatch_indices)
        # # mid_labels = logits_to_onehot(mid_probs)
        # ce_loss = -torch.sum(mid_labels * torch.log(output_probs + self.epsilon), dim=-1).mean()
        # pp = torch.exp(ce_loss)

        return accuracy1.item(), accuracy2.item(), accuracy3.item(), \
               float(sum(ff_tensor1)) / (len(ff_tensor1) + 1e-5), float(sum(ff_tensor2)) / (len(ff_tensor2) + 1e-5), \
               float(sum(ff_tensor3)) / (len(ff_tensor3) + 1e-5)

class LlavaForConditionalGenerationDcache(LlavaForConditionalGeneration):

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            print(model_inputs['position_ids'])

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            # todo
            model_kwargs.update({"position_ids": model_kwargs['position_ids']+1} if model_kwargs['position_ids'] is not None else None)

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1, :].float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

def logits_to_onehot(logits, dim=-1):
    # Method 1: Using argmax and one_hot
    indices = torch.argmax(logits, dim=dim)
    return F.one_hot(indices, num_classes=logits.size(dim))

def count_true_segments_with_zeros(bool_tensor: torch.Tensor) -> tuple[int, torch.Tensor]:
    bool_int_list = bool_tensor.int().tolist()
    bool_list = "".join([str(i) for i in bool_int_list])
    num_list = [len(list(i)) for i in bool_list.split('0')]
    if sum(num_list) == len(bool_int_list):
        # num_list = [len(bool_int_list)]
        return num_list
    num_list = [k+1 for i,k in enumerate(num_list) if i != len(num_list)-1] + [num_list[len(num_list)-1]]

    return num_list
