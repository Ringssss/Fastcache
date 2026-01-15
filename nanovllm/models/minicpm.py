"""
MiniCPM-V Model Adapter for nano-vllm
=====================================

适配MiniCPM-V-2_6多模态模型到nano-vllm推理引擎。

MiniCPM-V架构:
- SigLIP Vision Encoder (vpm): 处理图像
- Resampler: 将视觉特征压缩到固定数量的query tokens
- Qwen2 LLM (llm): 语言模型

由于MiniCPM使用trust_remote_code，我们采用HuggingFace包装策略，
使用nano-vllm的KV-cache管理和压缩功能，同时复用原始模型的前向传播。

Date: 2024
"""

import torch
from torch import nn
from typing import Optional, Tuple, List, Dict, Any
import math

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


# ==================== Qwen2架构 (MiniCPM的LLM部分) ====================

class Qwen2Attention(nn.Module):
    """Qwen2注意力层 - 适配nano-vllm"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 32768,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 1000000.0,
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
            bias=True,  # Qwen2使用bias
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


class Qwen2MLP(nn.Module):
    """Qwen2 MLP层"""

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


class Qwen2DecoderLayer(nn.Module):
    """Qwen2解码器层"""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
            max_position=getattr(config, 'max_position_embeddings', 32768),
            rms_norm_eps=config.rms_norm_eps,
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen2MLP(
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


class Qwen2Model(nn.Module):
    """Qwen2语言模型"""

    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)
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


class Qwen2ForCausalLM(nn.Module):
    """
    Qwen2语言模型 (匹配MiniCPM的llm权重结构)

    权重路径: llm.model.layers... 和 llm.lm_head...
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
        self.model = Qwen2Model(config)
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


# ==================== MiniCPM配置类 ====================

class MiniCPMConfig:
    """MiniCPM配置类"""

    def __init__(self, hf_config):
        # MiniCPM-V直接使用顶级配置作为LLM配置
        self.vocab_size = hf_config.vocab_size
        self.hidden_size = hf_config.hidden_size
        self.num_hidden_layers = hf_config.num_hidden_layers
        self.num_attention_heads = hf_config.num_attention_heads
        self.num_key_value_heads = getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads)
        self.intermediate_size = hf_config.intermediate_size
        self.rms_norm_eps = getattr(hf_config, 'rms_norm_eps', 1e-6)
        self.max_position_embeddings = getattr(hf_config, 'max_position_embeddings', 32768)
        self.rope_theta = getattr(hf_config, 'rope_theta', 1000000.0)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.hidden_act = getattr(hf_config, 'hidden_act', 'silu')

        # 视觉相关配置
        if hasattr(hf_config, 'vision_config'):
            self.vision_config = hf_config.vision_config
            self.vision_hidden_size = self.vision_config.hidden_size
            self.image_size = getattr(self.vision_config, 'image_size', 980)
            self.patch_size = getattr(self.vision_config, 'patch_size', 14)
        else:
            self.vision_config = None
            self.vision_hidden_size = 1152  # default
            self.image_size = 980
            self.patch_size = 14

        # MiniCPM特有配置
        self.query_num = getattr(hf_config, 'query_num', 64)  # Resampler的query数量
        self.slice_mode = getattr(hf_config, 'slice_mode', True)

        # 图像token数量 = query_num (Resampler输出)
        self.image_token_len = self.query_num


# ==================== MiniCPM完整模型 (nano-vllm适配) ====================

class MiniCPMForConditionalGeneration(nn.Module):
    """
    MiniCPM-V for nano-vllm

    权重结构匹配MiniCPM-V-2_6:
    - vpm... (SigLIP视觉编码器)
    - resampler... (视觉特征重采样器)
    - llm.model.layers... (Qwen2 LLM)
    - llm.lm_head... (语言模型头)

    注意：视觉部分(vpm + resampler)需要从HuggingFace原始模型加载，
    LLM部分使用nano-vllm的高效实现。
    """

    packed_modules_mapping = {
        # LLM层的映射
        "llm.model.layers.{}.self_attn.q_proj": ("llm.model.layers.{}.self_attn.qkv_proj", "q"),
        "llm.model.layers.{}.self_attn.k_proj": ("llm.model.layers.{}.self_attn.qkv_proj", "k"),
        "llm.model.layers.{}.self_attn.v_proj": ("llm.model.layers.{}.self_attn.qkv_proj", "v"),
        "llm.model.layers.{}.mlp.gate_proj": ("llm.model.layers.{}.mlp.gate_up_proj", 0),
        "llm.model.layers.{}.mlp.up_proj": ("llm.model.layers.{}.mlp.gate_up_proj", 1),
        # 简化版本
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, hf_model=None):
        """
        初始化MiniCPM模型

        Args:
            config: MiniCPMConfig或HuggingFace配置
            hf_model: 可选的预加载HuggingFace模型（用于视觉部分）
        """
        super().__init__()
        self.config = MiniCPMConfig(config) if not isinstance(config, MiniCPMConfig) else config

        # 视觉部分占位（需要从HF模型复制）
        self.vpm = None  # SigLIP视觉编码器
        self.resampler = None  # 视觉特征重采样器

        # LLM部分使用nano-vllm的高效实现
        self.llm = Qwen2ForCausalLM(self.config)

        # 图像token数量 = Resampler的query数量
        self.image_token_len = self.config.query_num

        # 记录是否已加载视觉部分
        self._vision_loaded = False

    def load_vision_modules(self, hf_model):
        """
        从HuggingFace模型加载视觉模块

        Args:
            hf_model: 已加载的HuggingFace MiniCPM模型
        """
        # 直接引用HF模型的视觉模块
        self.vpm = hf_model.vpm
        self.resampler = hf_model.resampler
        self._vision_loaded = True
        print("✓ MiniCPM视觉模块已加载")

    def get_vllm_embedding(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, List]:
        """
        获取嵌入向量（包括视觉特征）

        这是MiniCPM特有的方法，用于处理多模态输入

        Args:
            data: 包含input_ids, pixel_values, image_bound等的字典

        Returns:
            (inputs_embeds, vision_hidden_states)
        """
        if not self._vision_loaded:
            raise RuntimeError("视觉模块未加载，请先调用load_vision_modules()")

        dtype = self.llm.model.embed_tokens.weight.dtype
        device = self.llm.model.embed_tokens.weight.device
        tgt_sizes = data['tgt_sizes']
        pixel_values_list = data['pixel_values']
        vision_hidden_states = []
        all_pixel_values = []
        img_cnt = []

        for pixel_values in pixel_values_list:
            img_cnt.append(len(pixel_values))
            all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

        # 处理图像
        if all_pixel_values:
            tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
            tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

            max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

            all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                all_pixel_values, batch_first=True, padding_value=0.0)
            B, L, _ = all_pixel_values.shape
            all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

            patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
            for i in range(B):
                patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

            all_pixel_values = all_pixel_values.type(dtype)
            vision_embedding = self.vpm(
                all_pixel_values,
                patch_attention_mask=patch_attn_mask,
                tgt_sizes=tgt_sizes
            ).last_hidden_state
            vision_embedding = self.resampler(vision_embedding, tgt_sizes)

            start = 0
            for pixel_values in pixel_values_list:
                img_cnt = len(pixel_values)
                if img_cnt > 0:
                    vision_hidden_states.append(vision_embedding[start: start + img_cnt])
                    start += img_cnt
                else:
                    vision_hidden_states.append([])
        else:
            for _ in range(len(pixel_values_list)):
                vision_hidden_states.append([])

        # 获取文本嵌入
        vllm_embedding = self.llm.model.embed_tokens(data['input_ids'])
        new_vllm_embedding = vllm_embedding.clone()

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i
            for i in vision_hidden_states
        ]

        # 将视觉特征插入到对应位置
        bs = len(data['input_ids'])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data['image_bound'][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)

                    new_vllm_embedding[i] = cur_vllm_emb.scatter(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1])
                    )

        return new_vllm_embedding, vision_hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: 输入token IDs
            positions: 位置编码
            inputs_embeds: 可选的预计算嵌入
            pixel_values: 可选的图像像素值（未使用，因为MiniCPM需要特殊处理）

        Returns:
            hidden_states
        """
        if inputs_embeds is not None:
            # 使用预计算的嵌入（包含视觉特征）
            hidden_states = self.llm(
                input_ids=None,
                positions=positions,
                inputs_embeds=inputs_embeds
            )
        else:
            # 纯文本推理
            hidden_states = self.llm(input_ids, positions)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """计算logits"""
        return self.llm.lm_head(hidden_states)


# ==================== 权重加载辅助函数 ====================

def get_minicpm_weight_mapping():
    """获取MiniCPM权重映射"""
    return {
        # LLM Embeddings
        "llm.model.embed_tokens.weight": "llm.model.embed_tokens.weight",
        "llm.lm_head.weight": "llm.lm_head.weight",

        # LLM Layers - 通过packed_modules_mapping处理QKV和MLP

        # 视觉部分需要单独处理，因为使用HF原始模块
    }
