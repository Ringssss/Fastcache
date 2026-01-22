"""
LLaVA Model Implementation for nano-vllm
=========================================

支持LLaVA 1.5 7B模型，包含：
- Vision Tower (CLIP ViT-L/14-336)
- Multimodal Projector (2层MLP)
- Language Model (Llama-2-7B架构)

Author: Claude Code
Date: 2024
"""

import torch
from torch import nn
from typing import Optional, Tuple, List
import math

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


# ==================== Vision Tower ====================

class CLIPVisionEmbeddings(nn.Module):
    """CLIP视觉嵌入层"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # [B, embed_dim, H/P, W/P]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPAttention(nn.Module):
    """CLIP的多头注意力"""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class CLIPMLP(nn.Module):
    """CLIP的MLP层"""

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    """CLIP编码器层"""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    """CLIP视觉编码器"""

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class CLIPVisionTransformer(nn.Module):
    """CLIP视觉Transformer"""

    def __init__(self, config):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        hidden_states = self.encoder(hidden_states)
        # 返回所有patch的特征（不包含CLS token）
        hidden_states = hidden_states[:, 1:, :]
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states


class CLIPVisionModel(nn.Module):
    """CLIP视觉模型"""

    def __init__(self, config):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)


# ==================== Multimodal Projector ====================

class LlavaMultiModalProjector(nn.Module):
    """LLaVA多模态投影器 - 2层MLP"""

    def __init__(self, vision_hidden_size: int, text_hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden_size, text_hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# ==================== Language Model (Llama架构) ====================

class LlamaAttention(nn.Module):
    """Llama注意力层 - 适配nano-vllm"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = 1
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class LlamaMLP(nn.Module):
    """Llama MLP层"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class LlamaDecoderLayer(nn.Module):
    """Llama解码器层"""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
            max_position=getattr(config, 'max_position_embeddings', 4096),
            rms_norm_eps=config.rms_norm_eps,
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, 'hidden_act', 'silu'),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(nn.Module):
    """Llama语言模型"""

    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# ==================== LlamaForCausalLM (匹配HuggingFace结构) ====================

class LlamaForCausalLM(nn.Module):
    """
    Llama语言模型 (匹配HuggingFace权重结构)

    权重路径: language_model.model.layers... 和 language_model.lm_head...
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


# ==================== LLaVA完整模型 ====================

class LlavaConfig:
    """LLaVA配置类"""

    def __init__(self, hf_config):
        # 从HuggingFace配置中提取
        self.text_config = hf_config.text_config
        self.vision_config = hf_config.vision_config

        # 文本模型配置
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.num_attention_heads = self.text_config.num_attention_heads
        self.num_key_value_heads = getattr(self.text_config, 'num_key_value_heads', self.num_attention_heads)
        self.intermediate_size = self.text_config.intermediate_size
        self.rms_norm_eps = getattr(self.text_config, 'rms_norm_eps', 1e-6)
        self.max_position_embeddings = getattr(self.text_config, 'max_position_embeddings', 4096)
        self.rope_theta = getattr(self.text_config, 'rope_theta', 10000)
        self.head_dim = self.hidden_size // self.num_attention_heads

        # 视觉模型配置
        self.vision_hidden_size = self.vision_config.hidden_size
        self.image_size = self.vision_config.image_size
        self.patch_size = self.vision_config.patch_size
        self.num_channels = getattr(self.vision_config, 'num_channels', 3)

        # 图像token数量
        self.image_token_len = (self.image_size // self.patch_size) ** 2


class LlavaForConditionalGeneration(nn.Module):
    """
    LLaVA for nano-vllm

    权重结构匹配HuggingFace llava-1.5-7b-hf:
    - vision_tower.vision_model...
    - multi_modal_projector.linear_1/2...
    - language_model.model.layers...
    - language_model.lm_head...
    """

    # 注意：packed_modules_mapping需要加上language_model前缀
    packed_modules_mapping = {
        "language_model.model.layers.{}.self_attn.q_proj": ("language_model.model.layers.{}.self_attn.qkv_proj", "q"),
        "language_model.model.layers.{}.self_attn.k_proj": ("language_model.model.layers.{}.self_attn.qkv_proj", "k"),
        "language_model.model.layers.{}.self_attn.v_proj": ("language_model.model.layers.{}.self_attn.qkv_proj", "v"),
        "language_model.model.layers.{}.mlp.gate_proj": ("language_model.model.layers.{}.mlp.gate_up_proj", 0),
        "language_model.model.layers.{}.mlp.up_proj": ("language_model.model.layers.{}.mlp.gate_up_proj", 1),
        # 简化版本 - loader会处理
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        self.config = LlavaConfig(config) if not isinstance(config, LlavaConfig) else config

        # Vision Tower
        self.vision_tower = CLIPVisionModel(self.config.vision_config)

        # Multimodal Projector
        self.multi_modal_projector = LlavaMultiModalProjector(
            self.config.vision_hidden_size,
            self.config.hidden_size
        )

        # Language Model (使用LlamaForCausalLM以匹配权重结构)
        self.language_model = LlamaForCausalLM(self.config.text_config)

        # 图像token ID
        self.image_token_index = 32000  # LLaVA默认

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """提取图像特征并投影到语言模型空间"""
        image_features = self.vision_tower(pixel_values)
        image_features = self.multi_modal_projector(image_features)
        return image_features

    def merge_input_ids_with_image_features(
        self,
        input_ids: torch.Tensor,
        image_features: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将图像特征插入到input_ids对应位置

        nano-vllm使用扁平化的1D tensor格式，不是batched格式

        Args:
            input_ids: 输入token IDs, 1D tensor [seq_len]
            image_features: 图像特征 [batch, num_patches, hidden] 或 [num_patches, hidden]
            positions: 位置编码, 1D tensor [seq_len]

        Returns:
            (merged_embeds, merged_positions)
        """
        # nano-vllm使用1D tensor
        # 先获取text embeddings
        text_embeds = self.language_model.model.embed_tokens(input_ids)  # [seq_len, hidden]

        # 确保image_features是2D [num_patches, hidden]
        if image_features.dim() == 3:
            # [batch, num_patches, hidden] -> [num_patches, hidden]
            image_features = image_features.squeeze(0)

        # 找到图像token位置
        img_positions = (input_ids == self.image_token_index).nonzero(as_tuple=True)[0]

        if len(img_positions) == 0:
            # 没有图像token，直接返回
            return text_embeds, positions

        # 取第一个图像token的位置
        img_idx = img_positions[0].item()
        num_image_tokens = image_features.shape[0]

        # 分割embeddings
        before_embeds = text_embeds[:img_idx]  # [img_idx, hidden]
        after_embeds = text_embeds[img_idx + 1:]  # [seq_len - img_idx - 1, hidden]

        # 合并embeddings
        merged_embeds = torch.cat([before_embeds, image_features, after_embeds], dim=0)

        # 调整positions
        before_pos = positions[:img_idx]
        # 图像tokens的positions从img_idx开始连续编号
        image_pos = torch.arange(
            img_idx,
            img_idx + num_image_tokens,
            dtype=positions.dtype,
            device=positions.device
        )
        # after部分的positions需要加上图像token数量-1的偏移
        after_pos = positions[img_idx + 1:] + (num_image_tokens - 1)

        merged_positions = torch.cat([before_pos, image_pos, after_pos])

        return merged_embeds, merged_positions

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: 输入token IDs
            positions: 位置编码
            pixel_values: 图像像素值 [batch, C, H, W]

        Returns:
            hidden_states
        """
        if pixel_values is not None:
            # 多模态推理
            image_features = self.get_image_features(pixel_values)
            inputs_embeds, positions = self.merge_input_ids_with_image_features(
                input_ids, image_features, positions
            )
            hidden_states = self.language_model(
                input_ids=None,
                positions=positions,
                inputs_embeds=inputs_embeds
            )
        else:
            # 纯文本推理（decode阶段）
            hidden_states = self.language_model(input_ids, positions)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """计算logits"""
        logits = self.language_model.lm_head(hidden_states)
        return logits


# ==================== 权重加载映射 ====================

def get_llava_weight_mapping():
    """获取LLaVA权重映射"""
    return {
        # Vision Tower
        "vision_tower.vision_model.embeddings.class_embedding": "vision_tower.vision_model.embeddings.class_embedding",
        "vision_tower.vision_model.embeddings.patch_embedding.weight": "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "vision_tower.vision_model.embeddings.position_embedding.weight": "vision_tower.vision_model.embeddings.position_embedding.weight",
        # ... 更多映射

        # Multimodal Projector
        "multi_modal_projector.linear_1.weight": "multi_modal_projector.linear_1.weight",
        "multi_modal_projector.linear_1.bias": "multi_modal_projector.linear_1.bias",
        "multi_modal_projector.linear_2.weight": "multi_modal_projector.linear_2.weight",
        "multi_modal_projector.linear_2.bias": "multi_modal_projector.linear_2.bias",

        # Language Model - 使用packed_modules_mapping处理
    }
