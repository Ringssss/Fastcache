import torch
from transformers import AutoModel, AutoConfig, PreTrainedModel, PretrainedConfig, CLIPVisionConfig, PretrainedConfig
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, List, Tuple
import numpy as np
import json

class LLMConfigWrapper:
    """LLM配置包装器，包装原始配置并添加计算强度分析所需的额外属性"""

    def __init__(self, config: PretrainedConfig):
        self.original_config = config

        # 特殊处理 LLaVA 配置
        if hasattr(config, 'text_config'):
            # LLaVA 的语言模型配置存储在 text_config 中
            text_config = config.text_config
            self.hidden_size = text_config.hidden_size
            self.num_attention_heads = text_config.num_attention_heads
            self.num_hidden_layers = text_config.num_hidden_layers
            self.vocab_size = text_config.vocab_size
            # 设置中间层大小
            self.intermediate_size = getattr(text_config, "intermediate_size",
                                             4 * text_config.hidden_size)
            # 设置模型类型
            self.model_type = text_config.model_type
        else:
            # 原有的标准配置处理
            self.hidden_size = config.hidden_size
            self.num_attention_heads = config.num_attention_heads
            self.num_hidden_layers = config.num_hidden_layers
            self.vocab_size = config.vocab_size
            # 设置中间层大小
            self.intermediate_size = getattr(config, "intermediate_size",
                                             4 * config.hidden_size)
            # 设置模型类型
            self.model_type = config.model_type

        # 设置dtype_size
        self.dtype_size = self._determine_dtype_size(config)

        # 复制其他可能存在的属性
        self.rotary_dim = getattr(config, "rotary_dim", None)
        self.use_gqa = getattr(config, "use_gqa", False)
        self.num_key_value_heads = getattr(config, "num_key_value_heads",
                                           self.num_attention_heads)
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.layer_norm_eps = getattr(config, "layer_norm_eps", 1e-12)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", None)

    def _determine_dtype_size(self, config: PretrainedConfig) -> int:
        """确定数据类型的大小（以字节为单位）"""
        if hasattr(config, 'torch_dtype'):
            if config.torch_dtype in ['float32', torch.float32]:
                return 4
            elif config.torch_dtype in ['float16', torch.float16,
                                        'bfloat16', torch.bfloat16]:
                return 2
            elif config.torch_dtype in ['int8', torch.int8]:
                return 1
        # 默认使用 float16
        return 2

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "LLMConfigWrapper":
        """从预训练模型创建配置包装器"""
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name_or_path)
        return cls(config)

    def __getattr__(self, name: str):
        """当属性不存在时，尝试从原始配置获取"""
        return getattr(self.original_config, name)

@dataclass
class LLMConfig:
    """通用LLM配置"""
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    vocab_size: int
    intermediate_size: int
    model_type: str
    # 移除默认值，在from_pretrained中设置
    dtype_size: int
    rotary_dim: Optional[int] = None
    use_gqa: bool = False
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-12
    use_kv_cache: bool = True
    max_position_embeddings: Optional[int] = None

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "LLMConfig":
        """从预训练模型加载配置"""
        config = AutoConfig.from_pretrained(model_name_or_path)

        # 根据模型dtype设置dtype_size
        dtype_size = 2  # 默认FP16
        if hasattr(config, 'torch_dtype'):
            if config.torch_dtype in ['float32', torch.float32]:
                dtype_size = 4
            elif config.torch_dtype in ['float16', torch.float16, 'bfloat16', torch.bfloat16]:
                dtype_size = 2
            elif config.torch_dtype in ['int8', torch.int8]:
                dtype_size = 1

        return cls(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            vocab_size=config.vocab_size,
            intermediate_size=getattr(config, "intermediate_size", 4 * config.hidden_size),
            model_type=config.model_type,
            rotary_dim=getattr(config, "rotary_dim", None),
            use_gqa=getattr(config, "use_gqa", False),
            num_key_value_heads=getattr(config, "num_key_value_heads", None),
            rope_theta=getattr(config, "rope_theta", 10000.0),
            layer_norm_eps=getattr(config, "layer_norm_eps", 1e-12),
            max_position_embeddings=getattr(config, "max_position_embeddings", None),
            dtype_size=dtype_size  # 添加dtype_size
        )


@dataclass
class ComputeStats:
    """计算统计"""
    flops: int = 0
    memory_read: int = 0
    memory_write: int = 0
    batch_size: int = 1  # 添加batch_size字段
    seq_length: int = 0  # 添加序列长度字段，方便统计

    def __add__(self, other: "ComputeStats") -> "ComputeStats":
        return ComputeStats(
            flops=self.flops + other.flops,
            memory_read=self.memory_read + other.memory_read,
            memory_write=self.memory_write + other.memory_write,
            batch_size=max(self.batch_size, other.batch_size),  # 取较大的batch_size
            seq_length=max(self.seq_length, other.seq_length)  # 取较大的seq_length
        )

    @property
    def total_memory(self) -> int:
        return self.memory_read + self.memory_write

    @property
    def compute_intensity(self) -> float:
        return self.flops / self.total_memory if self.total_memory > 0 else 0

    @property
    def per_sample_flops(self) -> float:
        """每个样本的平均FLOPS"""
        return self.flops / self.batch_size if self.batch_size > 0 else self.flops

    @property
    def per_sample_memory(self) -> float:
        """每个样本的平均内存访问量"""
        return self.total_memory / self.batch_size if self.batch_size > 0 else self.total_memory


class LLMIntensityAnalyzer:
    def __init__(self, config: Union[str, PretrainedConfig, LLMConfigWrapper]):
        """
        初始化分析器
        Args:
            config: 模型配置、模型路径或配置包装器
        """
        if isinstance(config, str):
            self.config = LLMConfigWrapper.from_pretrained(config)
        elif isinstance(config, LLMConfigWrapper):
            self.config = config
        else:
            self.config = LLMConfigWrapper(config)

        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
    def _rotary_embedding_computation(self, seq_len: int) -> ComputeStats:
        """计算RoPE的计算开销"""
        stats = ComputeStats()
        if not self.config.rotary_dim:
            return stats

        rotary_dim = self.config.rotary_dim
        # 计算旋转矩阵和应用旋转
        stats.flops += seq_len * rotary_dim * 4  # sin, cos, 和两次旋转操作
        stats.memory_read += seq_len * rotary_dim * self.config.dtype_size
        stats.memory_write += seq_len * rotary_dim * self.config.dtype_size

        return stats

    def _attention_computation(
            self,
            batch_size: int,
            seq_len: int,
            past_len: int = 0,
            use_cache: bool = True
    ) -> ComputeStats:
        """计算注意力层的开销"""
        stats = ComputeStats()
        total_len = seq_len + (past_len if use_cache else 0)

        # 确定key/value头的数量
        num_kv_heads = (self.config.num_key_value_heads
                        if self.config.use_gqa
                        else self.config.num_attention_heads)

        # QKV投影
        qkv_hidden = self.config.hidden_size * (1 + 2 * num_kv_heads / self.config.num_attention_heads)
        stats.flops += batch_size * seq_len * self.config.hidden_size * qkv_hidden
        stats.memory_read += (batch_size * seq_len * self.config.hidden_size +
                              self.config.hidden_size * qkv_hidden) * self.config.dtype_size
        stats.memory_write += batch_size * seq_len * qkv_hidden * self.config.dtype_size

        # RoPE计算
        if self.config.rotary_dim:
            rope_stats = self._rotary_embedding_computation(seq_len)
            stats += rope_stats

        # 注意力分数计算
        stats.flops += (batch_size *
                        self.config.num_attention_heads *
                        seq_len *
                        total_len *
                        self.head_dim)

        # 读取KV cache
        if use_cache and past_len > 0:
            stats.memory_read += (batch_size *
                                  past_len *
                                  2 *
                                  num_kv_heads *
                                  self.head_dim *
                                  self.config.dtype_size)

        # Softmax
        stats.flops += batch_size * seq_len * total_len * self.config.num_attention_heads * 3
        stats.memory_read += batch_size * seq_len * total_len * self.config.num_attention_heads * self.config.dtype_size
        stats.memory_write += batch_size * seq_len * total_len * self.config.num_attention_heads * self.config.dtype_size

        # 注意力输出计算
        stats.flops += batch_size * seq_len * self.head_dim * total_len * self.config.num_attention_heads
        stats.memory_read += batch_size * (seq_len * total_len * self.config.num_attention_heads +
                                           total_len * self.config.hidden_size) * self.config.dtype_size
        stats.memory_write += batch_size * seq_len * self.config.hidden_size * self.config.dtype_size

        # 输出投影
        stats.flops += batch_size * seq_len * self.config.hidden_size * self.config.hidden_size
        stats.memory_read += (batch_size * seq_len * self.config.hidden_size +
                              self.config.hidden_size * self.config.hidden_size) * self.config.dtype_size
        stats.memory_write += batch_size * seq_len * self.config.hidden_size * self.config.dtype_size

        return stats

    def _ffn_computation(self, batch_size: int, seq_len: int) -> ComputeStats:
        """计算前馈网络的开销"""
        stats = ComputeStats()

        # 第一个线性层
        stats.flops += batch_size * seq_len * self.config.hidden_size * self.config.intermediate_size
        stats.memory_read += (batch_size * seq_len * self.config.hidden_size +
                              self.config.hidden_size * self.config.intermediate_size) * self.config.dtype_size
        stats.memory_write += batch_size * seq_len * self.config.intermediate_size * self.config.dtype_size

        # 激活函数 (GELU/SiLU等)
        stats.flops += batch_size * seq_len * self.config.intermediate_size * 4
        stats.memory_read += batch_size * seq_len * self.config.intermediate_size * self.config.dtype_size
        stats.memory_write += batch_size * seq_len * self.config.intermediate_size * self.config.dtype_size

        # 第二个线性层
        stats.flops += batch_size * seq_len * self.config.intermediate_size * self.config.hidden_size
        stats.memory_read += (batch_size * seq_len * self.config.intermediate_size +
                              self.config.intermediate_size * self.config.hidden_size) * self.config.dtype_size
        stats.memory_write += batch_size * seq_len * self.config.hidden_size * self.config.dtype_size

        return stats

    def _layer_norm_computation(self, batch_size: int, seq_len: int, hidden_size: int) -> ComputeStats:
        """计算层标准化的开销"""
        stats = ComputeStats()

        # 均值和方差计算
        stats.flops += batch_size * seq_len * hidden_size * 4
        stats.memory_read += batch_size * seq_len * hidden_size * self.config.dtype_size
        stats.memory_write += batch_size * seq_len * hidden_size * self.config.dtype_size

        # 标准化和缩放
        stats.flops += batch_size * seq_len * hidden_size * 2
        stats.memory_read += batch_size * seq_len * hidden_size * self.config.dtype_size
        stats.memory_write += batch_size * seq_len * hidden_size * self.config.dtype_size

        return stats

    def analyze_prefill(
            self,
            input_ids: torch.Tensor,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None
    ) -> Dict:
        """分析prefill阶段的计算强度"""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size, seq_len = input_ids.shape
        total_stats = ComputeStats()

        # Embedding层
        total_stats.memory_read += self.config.vocab_size * self.config.hidden_size * self.config.dtype_size
        total_stats.memory_write += batch_size * seq_len * self.config.hidden_size * self.config.dtype_size

        # 位置编码
        if position_ids is not None and self.config.max_position_embeddings:
            total_stats.memory_read += (self.config.max_position_embeddings *
                                        self.config.hidden_size *
                                        self.config.dtype_size)

        # Transformer层
        for _ in range(self.config.num_hidden_layers):
            # 第一个LayerNorm
            total_stats += self._layer_norm_computation(batch_size, seq_len, self.config.hidden_size)

            # 注意力层
            attn_stats = self._attention_computation(batch_size, seq_len, use_cache=False)
            total_stats += attn_stats

            # 第二个LayerNorm
            total_stats += self._layer_norm_computation(batch_size, seq_len, self.config.hidden_size)

            # FFN
            ffn_stats = self._ffn_computation(batch_size, seq_len)
            total_stats += ffn_stats

            # 残差连接
            total_stats.flops += batch_size * seq_len * self.config.hidden_size * 2
            total_stats.memory_read += batch_size * seq_len * self.config.hidden_size * 2 * self.config.dtype_size
            total_stats.memory_write += batch_size * seq_len * self.config.hidden_size * self.config.dtype_size

        # 最终LayerNorm
        total_stats += self._layer_norm_computation(batch_size, seq_len, self.config.hidden_size)

        # 输出层
        total_stats.flops += batch_size * seq_len * self.config.hidden_size * self.config.vocab_size
        total_stats.memory_read += (batch_size * seq_len * self.config.hidden_size +
                                    self.config.hidden_size * self.config.vocab_size) * self.config.dtype_size
        total_stats.memory_write += batch_size * seq_len * self.config.vocab_size * self.config.dtype_size

        return {
            "phase": "prefill",
            "batch_size": batch_size,
            "seq_length": seq_len,
            "flops": total_stats.flops,
            "memory_read": total_stats.memory_read,
            "memory_write": total_stats.memory_write,
            "total_memory": total_stats.total_memory,
            "compute_intensity": total_stats.compute_intensity,
            "theoretical_time": total_stats.flops / (312e12)  # A100 FP16 peak TFLOPS
        }

    def analyze_decode(
            self,
            input_ids: torch.Tensor,
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None
    ) -> Dict:
        """分析decode阶段的计算强度"""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size, seq_len = input_ids.shape
        assert seq_len == 1, "Decode阶段每次只处理一个token"

        past_len = past_key_values[0][0].size(2)
        total_stats = ComputeStats()

        # Embedding层
        total_stats.memory_read += self.config.vocab_size * self.config.hidden_size * self.config.dtype_size
        total_stats.memory_write += batch_size * self.config.hidden_size * self.config.dtype_size

        # 位置编码
        if position_ids is not None and self.config.max_position_embeddings:
            total_stats.memory_read += self.config.max_position_embeddings * self.config.hidden_size * self.config.dtype_size

        # Transformer层
        for layer_idx in range(self.config.num_hidden_layers):
            # 第一个LayerNorm
            total_stats += self._layer_norm_computation(batch_size, seq_len, self.config.hidden_size)

            # 注意力层（使用KV cache）
            attn_stats = self._attention_computation(
                batch_size,
                seq_len,
                past_len=past_len,
                use_cache=True
            )
            total_stats += attn_stats

            # 第二个LayerNorm
            total_stats += self._layer_norm_computation(batch_size, seq_len, self.config.hidden_size)

            # FFN
            ffn_stats = self._ffn_computation(batch_size, seq_len)
            total_stats += ffn_stats

            # 残差连接
            total_stats.flops += batch_size * seq_len * self.config.hidden_size * 2
            total_stats.memory_read += batch_size * seq_len * self.config.hidden_size * 2 * self.config.dtype_size
            total_stats.memory_write += batch_size * seq_len * self.config.hidden_size * self.config.dtype_size

        # 最终LayerNorm
        total_stats += self._layer_norm_computation(batch_size, seq_len, self.config.hidden_size)

        # 输出层
        total_stats.flops += batch_size * seq_len * self.config.hidden_size * self.config.vocab_size
        total_stats.memory_read += (batch_size * seq_len * self.config.hidden_size +
                                    self.config.hidden_size * self.config.vocab_size) * self.config.dtype_size
        total_stats.memory_write += batch_size * seq_len * self.config.vocab_size * self.config.dtype_size

        return {
            "phase": "decode",
            "batch_size": batch_size,
            "context_length": past_len,
            "flops": total_stats.flops,
            "memory_read": total_stats.memory_read,
            "memory_write": total_stats.memory_write,
            "total_memory": total_stats.total_memory,
            "compute_intensity": total_stats.compute_intensity,
            "theoretical_time": total_stats.flops / (312e12)  # A100 FP16 peak TFLOPS
        }

def format_number(number: float, base: int = 1024) -> str:
    """格式化数字为人类可读形式"""
    units = ['', 'K', 'M', 'G', 'T', 'P']
    unit_index = 0
    while number >= base and unit_index < len(units) - 1:
        number /= base
        unit_index += 1
    return f"{number:.2f}{units[unit_index]}"


# def example_usage():
#     """使用示例"""
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     import torch
#
#     # 加载模型和分词器
#     model_name = '/home/zhujianian/workspace/Uneed/huggingface_download/Llama-2-7b-chat-hf'
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     analyzer = LLMIntensityAnalyzer(model.config)
#
#     # 示例输入
#     input_text = ["这是一个测试输入，用于分析计算强度。"]*100
#     inputs = tokenizer(input_text, return_tensors="pt")
#     print(inputs)
#
#     # Prefill阶段分析
#     prefill_stats = analyzer.analyze_prefill(
#         input_ids=inputs.input_ids,
#         attention_mask=inputs.attention_mask
#     )
#
#     print("\nPrefill Phase Analysis:")
#     print(f"Sequence Length: {prefill_stats['seq_length']}")
#     print(f"FLOPS: {format_number(prefill_stats['flops'])}")
#     print(f"Memory Access: {format_number(prefill_stats['total_memory'])} bytes")
#     print(f"Compute Intensity: {prefill_stats['compute_intensity']:.2f} FLOPS/byte")
#     print(f"Theoretical Time: {prefill_stats['theoretical_time'] * 1000:.2f} ms")
#
#     # 生成一个token进行decode阶段分析
#     with torch.no_grad():
#         outputs = model(
#             input_ids=inputs.input_ids,
#             attention_mask=inputs.attention_mask,
#             use_cache=True
#         )
#
#         next_token = outputs.logits[:, -1:].argmax(dim=-1)
#         past_key_values = outputs.past_key_values
#
#     # Decode阶段分析
#     decode_stats = analyzer.analyze_decode(
#         input_ids=next_token,
#         past_key_values=past_key_values,
#         attention_mask=torch.cat([inputs.attention_mask, torch.ones_like(next_token)], dim=1)
#     )
#
#     print("\nDecode Phase Analysis:")
#     print(f"Context Length: {decode_stats['context_length']}")
#     print(f"FLOPS: {format_number(decode_stats['flops'])}")
#     print(f"Memory Access: {format_number(decode_stats['total_memory'])} bytes")
#     print(f"Compute Intensity: {decode_stats['compute_intensity']:.2f} FLOPS/byte")
#     print(f"Theoretical Time: {decode_stats['theoretical_time'] * 1000:.2f} ms")



@dataclass
class ComputeStats:
    """计算统计"""
    flops: int = 0
    memory_read: int = 0
    memory_write: int = 0

    def __add__(self, other: "ComputeStats") -> "ComputeStats":
        return ComputeStats(
            flops=self.flops + other.flops,
            memory_read=self.memory_read + other.memory_read,
            memory_write=self.memory_write + other.memory_write
        )

    @property
    def total_memory(self) -> int:
        return self.memory_read + self.memory_write

    @property
    def compute_intensity(self) -> float:
        return self.flops / self.total_memory if self.total_memory > 0 else 0


class VisionEncoderAnalyzer:
    """视觉编码器计算强度分析器"""

    def __init__(self, image_size: int = 224, patch_size: int = 14,
                 hidden_size: int = 1024, num_attention_heads: int = 16,
                 num_hidden_layers: int = 24, dtype_size: int = 2):
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.dtype_size = dtype_size
        self.num_patches = (image_size // patch_size) ** 2

    def analyze_vision_encoding(self, batch_size: int = 1) -> ComputeStats:
        stats = ComputeStats()

        # 1. 图像补丁编码
        patch_dim = 3 * self.patch_size * self.patch_size  # RGB * patch dimensions
        stats.flops += batch_size * self.num_patches * patch_dim * self.hidden_size
        stats.memory_read += batch_size * self.image_size * self.image_size * 3 * self.dtype_size
        stats.memory_write += batch_size * self.num_patches * self.hidden_size * self.dtype_size

        # 2. 位置编码
        stats.memory_read += self.num_patches * self.hidden_size * self.dtype_size
        stats.memory_write += batch_size * self.num_patches * self.hidden_size * self.dtype_size

        # 3. Transformer层处理
        for _ in range(self.num_hidden_layers):
            # 自注意力
            stats.flops += (batch_size * self.num_patches * self.hidden_size * self.hidden_size * 3 +
                            batch_size * self.num_attention_heads * self.num_patches * self.num_patches *
                            (self.hidden_size // self.num_attention_heads))

            # FFN
            stats.flops += batch_size * self.num_patches * self.hidden_size * self.hidden_size * 4

            # 读写内存
            stats.memory_read += batch_size * self.num_patches * self.hidden_size * 3 * self.dtype_size
            stats.memory_write += batch_size * self.num_patches * self.hidden_size * self.dtype_size

        return stats


class LLaVAConfigWrapper(LLMConfigWrapper):
    """LLaVA配置包装器，扩展自LLMConfigWrapper"""

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        # 提取视觉编码器配置
        self.vision_config = self._extract_vision_config(config)

        # 提取多模态映射配置
        self.mm_config = self._extract_mm_config(config)

    def _extract_vision_config(self, config: PretrainedConfig) -> Dict[str, Any]:
        """从LLaVA配置中提取视觉编码器配置"""
        vision_config = {}


        if hasattr(config, 'vision_config'):
            clip_config = config.vision_config
            if isinstance(clip_config, dict):
                clip_config = CLIPVisionConfig(**clip_config)

            vision_config.update({
                'image_size': getattr(clip_config, 'image_size', 224),
                'patch_size': getattr(clip_config, 'patch_size', 14),
                'hidden_size': getattr(clip_config, 'hidden_size', 1024),
                'num_attention_heads': getattr(clip_config, 'num_attention_heads', 16),
                'num_hidden_layers': getattr(clip_config, 'num_hidden_layers', 24),
                'dtype_size': self._determine_dtype_size(clip_config)
            })
        elif hasattr(config, 'mm_vision_tower'):
            vision_config.update({
                'image_size': getattr(config, 'image_size', 224),
                'patch_size': getattr(config, 'patch_size', 14),
                'hidden_size': getattr(config, 'vision_hidden_size', 1024),
                'num_attention_heads': getattr(config, 'num_attention_heads', 16),
                'num_hidden_layers': getattr(config, 'num_hidden_layers', 24),
                'dtype_size': self.dtype_size
            })

        return vision_config

    def _extract_mm_config(self, config: PretrainedConfig) -> Dict[str, Any]:
        """提取多模态映射配置"""
        mm_config = {}

        if hasattr(config, 'mm_projector_type'):
            mm_config['projector_type'] = config.mm_projector_type

        if hasattr(config, 'mm_hidden_size'):
            mm_config['hidden_size'] = config.mm_hidden_size

        if hasattr(config, 'mm_vision_select_layer'):
            mm_config['vision_select_layer'] = config.mm_vision_select_layer

        return mm_config


class LLaVAIntensityAnalyzer(LLMIntensityAnalyzer):
    """LLaVA模型计算强度分析器"""

    def __init__(self, config: Union[str, PretrainedConfig, LLaVAConfigWrapper]):
        if isinstance(config, str):
            config = LLaVAConfigWrapper.from_pretrained(config)
        elif not isinstance(config, LLaVAConfigWrapper):
            config = LLaVAConfigWrapper(config)

        super().__init__(config)
        self.vision_encoder = VisionEncoderAnalyzer(**self.config.vision_config)

    def _mm_projection_computation(self, batch_size: int, num_patches: int) -> ComputeStats:
        """计算多模态投影的开销"""
        stats = ComputeStats()

        if not hasattr(self.config, 'mm_config'):
            return stats

        mm_config = self.config.mm_config
        hidden_size = mm_config.get('hidden_size', self.config.hidden_size)

        # 计算投影层开销
        stats.flops += batch_size * num_patches * hidden_size * self.config.hidden_size
        stats.memory_read += (batch_size * num_patches * hidden_size +
                              hidden_size * self.config.hidden_size) * self.config.dtype_size
        stats.memory_write += batch_size * num_patches * self.config.hidden_size * self.config.dtype_size

        return stats

    def analyze_multimodal_prefill(
            self,
            input_ids: torch.Tensor,
            pixel_values: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Dict:
        """分析多模态预填充阶段的计算强度，包括图像编码和特征集成"""
        stats = ComputeStats()

        # 处理视觉输入
        if pixel_values is not None:
            batch_size = len(pixel_values)

            # 1. 视觉编码
            vision_stats = self.vision_encoder.analyze_vision_encoding(batch_size)
            stats += vision_stats

            # 2. 多模态投影 - 将视觉特征投影到语言空间
            projection_stats = self._mm_projection_computation(
                batch_size=batch_size,
                num_patches=self.vision_encoder.num_patches
            )
            stats += projection_stats

            # 3. 将视觉特征整合到KV cache
            # 这部分计算包含在标准的prefill计算中，因为视觉特征会被当作序列的一部分处理

        # 4. 语言模型处理
        lm_stats = super().analyze_prefill(input_ids, **kwargs)

        # 合并统计结果
        for key in ['flops', 'memory_read', 'memory_write']:
            lm_stats[key] += getattr(stats, key)

        # 添加多模态特定信息
        lm_stats.update({
            'has_vision_input': pixel_values is not None,
            'num_image_tokens': self.vision_encoder.num_patches if pixel_values is not None else 0
        })

        return lm_stats

    def analyze_decode(
            self,
            input_ids: torch.Tensor,
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
            **kwargs
    ) -> Dict:
        """分析LLaVA解码阶段的计算强度（图像特征已在KV cache中）"""
        # 直接使用父类的decode分析，因为图像特征已经在KV cache中
        decode_stats = super().analyze_decode(
            input_ids=input_ids,
            past_key_values=past_key_values,
            **kwargs
        )

        # 添加多模态信息
        decode_stats['has_vision_input'] = True
        decode_stats['vision_in_kv_cache'] = True

        return decode_stats

    def _analyze_mlp_computation(
            self,
            batch_size: int,
            seq_len: int,
            input_dim: int,
            output_dim: int
    ) -> ComputeStats:
        """分析MLP层的计算开销"""
        stats = ComputeStats()
        # stats = ComputeStats(batch_size=batch_size, seq_length=seq_len)

        # Linear层计算
        stats.flops += batch_size * seq_len * input_dim * output_dim
        stats.memory_read += (batch_size * seq_len * input_dim +
                              input_dim * output_dim) * self.config.dtype_size
        stats.memory_write += batch_size * seq_len * output_dim * self.config.dtype_size

        # ReLU激活
        stats.flops += batch_size * seq_len * output_dim
        stats.memory_read += batch_size * seq_len * output_dim * self.config.dtype_size
        stats.memory_write += batch_size * seq_len * output_dim * self.config.dtype_size

        return stats

    def analyze_kv_cache_compression(
            self,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            image_len: int,
            text_len: int,
            compression_factor: int = 2
    ) -> Dict:
        """分析KV Cache压缩的计算开销

        Args:
            kv_cache: KV缓存列表
            image_len: 图像token长度
            text_len: 文本token长度
            compression_factor: 压缩因子
        """
        total_stats = ComputeStats()

        # 获取维度信息
        batch_size = kv_cache[0][0].size(0)
        num_heads = kv_cache[0][0].size(1)
        head_dim = kv_cache[0][0].size(-1)

        for layer_idx in range(len(kv_cache)):
            # 1. 图像部分压缩
            if image_len >= compression_factor:
                compressed_image_len = image_len // compression_factor
                # 图像K压缩
                image_k_stats = self._analyze_mlp_computation(
                    batch_size=batch_size * num_heads,
                    seq_len=compressed_image_len,
                    input_dim=head_dim * compression_factor,
                    output_dim=head_dim
                )
                total_stats += image_k_stats

                # 图像V压缩
                image_v_stats = self._analyze_mlp_computation(
                    batch_size=batch_size * num_heads,
                    seq_len=compressed_image_len,
                    input_dim=head_dim * compression_factor,
                    output_dim=head_dim
                )
                total_stats += image_v_stats

            # 2. 文本部分压缩
            if text_len >= compression_factor:
                compressed_text_len = text_len // compression_factor
                # 文本K压缩
                text_k_stats = self._analyze_mlp_computation(
                    batch_size=batch_size * num_heads,
                    seq_len=compressed_text_len,
                    input_dim=head_dim * compression_factor,
                    output_dim=head_dim
                )
                total_stats += text_k_stats

                # 文本V压缩
                text_v_stats = self._analyze_mlp_computation(
                    batch_size=batch_size * num_heads,
                    seq_len=compressed_text_len,
                    input_dim=head_dim * compression_factor,
                    output_dim=head_dim
                )
                total_stats += text_v_stats

            # 3. Concat操作的内存开销
            total_new_len = (
                    (image_len // compression_factor if image_len >= compression_factor else image_len) +
                    (text_len // compression_factor if text_len >= compression_factor else text_len)
            )
            concat_memory = (batch_size * num_heads * total_new_len *
                             head_dim * self.config.dtype_size)
            total_stats.memory_write += concat_memory * 2  # K和V各一次

        # return {
        #     "phase": "kv_cache_compression",
        #     "batch_size": batch_size,
        #     "image_length": image_len,
        #     "text_length": text_len,
        #     "compression_factor": compression_factor,
        #     "compressed_length": total_new_len,
        #     "flops": total_stats.flops,
        #     "memory_read": total_stats.memory_read,
        #     "memory_write": total_stats.memory_write,
        #     "total_memory": total_stats.total_memory,
        #     "compute_intensity": total_stats.compute_intensity,
        #     "theoretical_time": total_stats.flops / (312e12),  # A100 FP16 peak TFLOPS
        #     "per_sample_flops": total_stats.per_sample_flops,
        #     "per_sample_memory": total_stats.per_sample_memory
        # }
        return {
            "phase": "decode",
            "batch_size": batch_size,
            "context_length": image_len+text_len,
            "flops": total_stats.flops,
            "memory_read": total_stats.memory_read,
            "memory_write": total_stats.memory_write,
            "total_memory": total_stats.total_memory,
            "compute_intensity": total_stats.compute_intensity,
            "theoretical_time": total_stats.flops / (312e12)  # A100 FP16 peak TFLOPS
        }

# def format_number(number: float, base: int = 1000) -> str:
#     """格式化数字为人类可读形式"""
#     units = ['', 'K', 'M', 'G', 'T', 'P']
#     unit_index = 0
#     while number >= base and unit_index < len(units) - 1:
#         number /= base
#         unit_index += 1
#     return f"{number:.2f}{units[unit_index]}"

# def example_llava_usage():
#     """LLaVA分析器使用示例"""
#     from transformers import AutoProcessor, LlavaForConditionalGeneration
#     import torch
#     from PIL import Image
#
#     # 加载模型和处理器
#     model_name = "/home/zhujianian/workspace/Uneed/huggingface_download/llava-1.5-7b-hf"
#     model = LlavaForConditionalGeneration.from_pretrained(model_name)
#     processor = AutoProcessor.from_pretrained(model_name)
#     analyzer = LLaVAIntensityAnalyzer(model.config)
#
#     # 准备输入
#     text = ["<image>What's in this image?"]*20
#     image = Image.open("/home/zhujianian/cvpr/datasets/gqa/images/1013.jpg")
#     image_list = [image] * 20
#     # 1014
#     # image = Image.open("/home/zhujianian/cvpr/images/asuka.jpg")
#
#
#     inputs = processor(text, image_list, return_tensors="pt")
#
#     # 分析多模态预填充阶段（包括图像处理和特征集成）
#     prefill_stats = analyzer.analyze_multimodal_prefill(
#         input_ids=inputs["input_ids"],
#         pixel_values=inputs["pixel_values"]
#     )
#
#     print("\nMultimodal Prefill Analysis (including image processing):")
#     print(f"Sequence Length: {prefill_stats['seq_length']}")
#     print(f"batch_size: {prefill_stats['batch_size']}")
#     print(f"Number of Image Tokens: {prefill_stats['num_image_tokens']}")
#     print(f"FLOPS: {format_number(prefill_stats['flops'])}")
#     print(f"Memory Access: {format_number(prefill_stats['total_memory'])} bytes")
#     print(f"Compute Intensity: {prefill_stats['compute_intensity']:.2f} FLOPS/byte")
#     print(f"Theoretical Time: {prefill_stats['theoretical_time'] * 1000:.2f} ms")
#
#     # 模拟生成第一个token
#     with torch.no_grad():
#         outputs = model(**inputs, use_cache=True)
#         next_token = outputs.logits[:, -1:].argmax(dim=-1)
#         past_key_values = outputs.past_key_values
#
#     # 分析解码阶段（图像特征已在KV cache中）
#     decode_stats = analyzer.analyze_decode(
#         input_ids=next_token,
#         past_key_values=past_key_values
#     )
#
#     print("\nDecode Analysis (with image features in KV cache):")
#     print(f"Context Length: {decode_stats['context_length']}")
#     print(f"FLOPS: {format_number(decode_stats['flops'])}")
#     print(f"Memory Access: {format_number(decode_stats['total_memory'])} bytes")
#     print(f"Compute Intensity: {decode_stats['compute_intensity']:.2f} FLOPS/byte")
#     print(f"Theoretical Time: {decode_stats['theoretical_time'] * 1000:.2f} ms")
#
# if __name__ == "__main__":
#     example_llava_usage()


