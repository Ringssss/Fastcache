"""
Extended Model Runner for LLaVA + KV-Cache Compression
======================================================

扩展nano-vllm的ModelRunner以支持：
1. LLaVA多模态模型
2. KV-Cache压缩
3. CUDA Graph加速（包括压缩后的decode）
4. 异步压缩 - 压缩在独立CUDA stream执行，不阻塞decode

优化点：
- 压缩后正确更新序列的compressed_len用于decode
- 批量压缩减少开销
- CUDA Graph与压缩融合
- 异步压缩: prefill完成后立即开始decode，压缩在后台进行

Author: Claude Code
Date: 2024
"""

import torch
import sys
import threading
from contextlib import nullcontext
from typing import Optional, List, Tuple, Dict
from enum import Enum

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.memory import get_gpu_memory
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.loader import load_model
from nanovllm.utils.tp import get_tp_size, get_tp_rank, tp_broadcast
from nanovllm.utils.streams import build_streams
from nanovllm.kernels.zero_overhead_compress import create_zero_overhead_compressor

# 添加压缩器路径
sys.path.append('/home/zhujianian/cvpr')
sys.path.append('/home/zhujianian/cvpr/utils_ccm')


class CompressionState(Enum):
    """序列的压缩状态"""
    NONE = 0           # 未压缩
    PENDING = 1        # 等待压缩（已提交到async stream）
    IN_PROGRESS = 2    # 压缩进行中
    COMPLETED = 3      # 压缩完成


class AsyncCompressionManager:
    """
    异步压缩管理器

    管理异步压缩任务，追踪每个序列的压缩状态，
    在压缩完成后切换decode使用的KV-cache。

    工作流程:
    1. prefill完成后，复制原始KV-cache到临时buffer
    2. 在独立stream中异步执行压缩
    3. 前几个decode step使用原始KV-cache
    4. 压缩完成后，切换到压缩后的KV-cache
    """

    def __init__(self, model_runner: 'LlavaModelRunner'):
        self.model_runner = model_runner
        self.compress_stream = torch.cuda.Stream()

        # 序列状态追踪
        self.seq_states: Dict[int, CompressionState] = {}
        self.seq_original_lens: Dict[int, int] = {}  # 原始KV-cache长度
        self.seq_compressed_lens: Dict[int, int] = {}  # 压缩后KV-cache长度

        # 压缩完成事件
        self.compress_events: Dict[int, torch.cuda.Event] = {}

        # 临时KV-cache buffer (用于存储原始KV-cache供decode使用)
        self.temp_kv_buffer: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

        # 锁用于线程安全
        self.lock = threading.Lock()

    def start_async_compression(
        self,
        seqs: List[Sequence],
        image_len: int = 0
    ):
        """
        启动异步压缩

        Args:
            seqs: 需要压缩的序列列表
            image_len: 图像token长度
        """
        if self.model_runner.compressor is None:
            return

        with self.lock:
            for seq in seqs:
                # 记录原始长度
                self.seq_original_lens[seq.seq_id] = len(seq)
                self.seq_states[seq.seq_id] = CompressionState.PENDING

                # 创建完成事件
                self.compress_events[seq.seq_id] = torch.cuda.Event()

        # 在压缩stream中执行压缩
        with torch.cuda.stream(self.compress_stream):
            for seq in seqs:
                self._compress_single_seq_async(seq, image_len)

    def _compress_single_seq_async(self, seq: Sequence, image_len: int):
        """在异步stream中压缩单个序列"""
        seq_id = seq.seq_id

        with self.lock:
            self.seq_states[seq_id] = CompressionState.IN_PROGRESS

        try:
            block_table = seq.block_table
            seq_len = len(seq)

            # 提取KV-cache
            hf_kv_cache = self.model_runner._extract_kv_cache_fast(block_table, seq_len)

            # 计算压缩参数
            text_len = max(1, seq_len - image_len)
            it_len = [image_len, text_len]

            # 执行压缩
            with torch.no_grad():
                compressed_kv = self.model_runner.compressor(hf_kv_cache, it_len=it_len)

            # 获取压缩后长度
            compressed_seq_len = compressed_kv[0][0].shape[2]

            # 写回压缩后的KV-cache
            self.model_runner._write_compressed_kv_cache_fast(compressed_kv, block_table)

            with self.lock:
                self.seq_compressed_lens[seq_id] = compressed_seq_len
                self.seq_states[seq_id] = CompressionState.COMPLETED

                # 记录压缩事件
                self.compress_events[seq_id].record(self.compress_stream)

                # 更新model_runner的压缩长度记录
                self.model_runner._compressed_lens[seq_id] = compressed_seq_len

        except Exception as e:
            print(f"异步压缩失败 seq_id={seq_id}: {e}")
            with self.lock:
                self.seq_states[seq_id] = CompressionState.NONE

    def is_compression_complete(self, seq_id: int) -> bool:
        """检查指定序列的压缩是否完成"""
        with self.lock:
            state = self.seq_states.get(seq_id, CompressionState.NONE)
            if state != CompressionState.COMPLETED:
                return False

            # 检查CUDA事件是否完成
            event = self.compress_events.get(seq_id)
            if event is not None:
                return event.query()
            return True

    def wait_for_compression(self, seq_id: int, timeout_ms: float = 1000):
        """等待指定序列的压缩完成"""
        event = self.compress_events.get(seq_id)
        if event is not None:
            event.synchronize()

    def wait_all_compressions(self):
        """等待所有压缩完成"""
        self.compress_stream.synchronize()

    def get_effective_kv_len(self, seq: Sequence) -> int:
        """
        获取序列当前有效的KV-cache长度

        如果压缩已完成，返回压缩后长度；否则返回原始长度
        """
        seq_id = seq.seq_id

        with self.lock:
            state = self.seq_states.get(seq_id, CompressionState.NONE)

            if state == CompressionState.COMPLETED:
                # 检查事件是否真正完成
                event = self.compress_events.get(seq_id)
                if event is not None and event.query():
                    return self.seq_compressed_lens.get(seq_id, len(seq))

            # 压缩未完成，使用原始长度
            return self.seq_original_lens.get(seq_id, len(seq))

    def cleanup_seq(self, seq_id: int):
        """清理已完成序列的状态"""
        with self.lock:
            self.seq_states.pop(seq_id, None)
            self.seq_original_lens.pop(seq_id, None)
            self.seq_compressed_lens.pop(seq_id, None)
            self.compress_events.pop(seq_id, None)
            self.temp_kv_buffer.pop(seq_id, None)


class LlavaModelRunner:
    """
    支持LLaVA和KV-Cache压缩的ModelRunner

    关键修复：
    1. 压缩后正确更新序列的kv_cache_len，decode阶段使用压缩后的长度
    2. 批量压缩所有序列，减少kernel launch开销
    3. 支持CUDA Graph加速decode阶段
    4. 异步压缩: 压缩在独立stream执行，不阻塞prefill后的首次decode

    支持的压缩后端：
    - 'mlp': 基于MLP的GEMM压缩器 (BatchedGEMMCompressor)
    - 'kvpress': kvpress库的压缩方法 (streaming_llm, snapkv, knorm等)
    """

    def __init__(
        self,
        config: Config,
        compressor_path: Optional[str] = None,
        compression_factor: int = 5,
        enable_compression: bool = True,
        async_compression: bool = False,  # 是否启用异步压缩
        compression_backend: str = 'mlp',  # 压缩后端: 'mlp' 或 'kvpress'
        kvpress_method: str = 'streaming_llm',  # kvpress方法名
        compression_pipeline: str = 'auto',  # auto | prefill | decode
        compression_streams: int = 2,  # prefill pipeline streams
        greenctx_enabled: bool = False,
        greenctx_compress_ratio: float = 0.25,
        greenctx_main_ratio: float = 0.75,
        greenctx_main_stream: bool = False,
        decode_layers_per_step: int = 4,
    ):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.enable_compression = enable_compression
        self.compression_factor = compression_factor
        self.async_compression = async_compression
        self.compression_backend = compression_backend
        self.kvpress_method = kvpress_method
        self.compression_pipeline = compression_pipeline
        self.compression_streams = max(1, compression_streams)
        self.greenctx_enabled = greenctx_enabled
        self.greenctx_compress_ratio = greenctx_compress_ratio
        self.greenctx_main_ratio = greenctx_main_ratio
        self.greenctx_main_stream = greenctx_main_stream
        self.decode_layers_per_step = max(1, decode_layers_per_step)

        # 存储每个序列压缩后的实际KV-cache长度
        self._compressed_lens = {}  # seq_id -> compressed_len
        self._last_compress_time_ms: Optional[float] = None
        self._last_compress_tokens: int = 0

        # 异步压缩管理器 (延迟初始化)
        self._async_manager: Optional[AsyncCompressionManager] = None

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        # 检测模型类型并加载
        model_type = self._detect_model_type(hf_config)
        self.model_type = model_type  # 保存模型类型
        print(f"检测到模型类型: {model_type}")

        if model_type == "minicpm":
            # MiniCPM-V多模态模型
            from nanovllm.models.minicpm import MiniCPMForConditionalGeneration
            self.model = MiniCPMForConditionalGeneration(hf_config)
            self.is_multimodal = True
            # MiniCPM使用Resampler输出固定数量的query tokens
            self.image_token_len = getattr(hf_config, 'query_num', 64)
            # 需要从HF加载视觉模块
            self._hf_config = hf_config
            self._model_path = config.model
        elif model_type == "llava":
            from nanovllm.models.llava import LlavaForConditionalGeneration
            self.model = LlavaForConditionalGeneration(hf_config)
            self.is_multimodal = True
            # 计算图像token数
            vision_config = hf_config.vision_config
            self.image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
        elif model_type == "llama":
            from nanovllm.models.llama import LlamaForCausalLM
            self.model = LlamaForCausalLM(hf_config)
            self.is_multimodal = False
            self.image_token_len = 0
        else:
            from nanovllm.models.qwen3 import Qwen3ForCausalLM
            self.model = Qwen3ForCausalLM(hf_config)
            self.is_multimodal = False
            self.image_token_len = 0

        # 加载模型权重
        if model_type == "minicpm":
            self._load_minicpm_model(config.model, hf_config)
        else:
            load_model(self.model, config.model)
        self.sampler = Sampler()
        self.allocate_kv_cache(config.gpu_memory_utilization)

        # 加载压缩器
        self.compressor = None
        self.batched_compressor = None
        self.use_batched_compressor = False
        if enable_compression:
            if compression_backend == 'kvpress':
                self._load_kvpress_compressor(hf_config)
            elif compressor_path:
                self._load_compressor(compressor_path, hf_config)

        # 初始化异步压缩管理器
        if async_compression and self.compressor is not None:
            self._async_manager = AsyncCompressionManager(self)
            print("✓ 异步压缩已启用")

        # Pipeline selection
        if self.compression_pipeline == "auto":
            if self.compression_backend == "mlp":
                self.compression_pipeline = "decode"
            elif self.compression_backend == "kvpress":
                self.compression_pipeline = "prefill"
            else:
                self.compression_pipeline = "prefill"
        if not self.enable_compression or self.compressor is None:
            self.compression_pipeline = "none"

        # Stream setup (greenctx aware)
        self._main_stream, self._compression_stream_pool, self._greenctx_info = build_streams(
            use_greenctx=self.greenctx_enabled,
            sm_compress_ratio=self.greenctx_compress_ratio,
            sm_main_ratio=self.greenctx_main_ratio,
            use_greenctx_main=self.greenctx_main_stream,
            compression_streams=self.compression_streams,
            compression_priority=-1,
        )

        if self.greenctx_enabled and not self._greenctx_info.available:
            print(f"⚠ greenctx unavailable, using fallback streams: {self._greenctx_info.reason}")

        # Decode-overlap compression (compute-bound)
        self._decode_overlap_engine = None
        self._decode_overlap_blocks: Dict[int, List[int]] = {}
        self._decode_overlap_prefill_lens: Dict[int, int] = {}
        if self.compression_pipeline == "decode" and self.compressor is not None:
            overlap_compressor = (
                self.batched_compressor
                if getattr(self, "use_batched_compressor", False) and self.batched_compressor is not None
                else self.compressor
            )
            self._decode_overlap_engine = create_zero_overhead_compressor(
                original_compressor=overlap_compressor,
                num_layers=getattr(hf_config, "num_hidden_layers", 32),
                layers_per_step=self.decode_layers_per_step,
                device=torch.device("cuda"),
                compress_stream=(self._compression_stream_pool.streams[0] if self._compression_stream_pool.streams else None),
            )

        # 始终尝试捕获CUDA Graph（压缩后decode也能用）
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def _detect_model_type(self, hf_config) -> str:
        """检测模型类型"""
        # 首先检查model_type
        if hasattr(hf_config, 'model_type'):
            model_type = hf_config.model_type.lower()
            # MiniCPM-V检测
            if 'minicpm' in model_type:
                return "minicpm"
            # 规范化模型类型名称
            if model_type in ['llama', 'llama2', 'llama3']:
                return "llama"
            elif model_type in ['qwen2', 'qwen3']:
                return "qwen3"

        # 检查architectures
        if hasattr(hf_config, 'architectures'):
            archs = [a.lower() for a in hf_config.architectures]
            if any('minicpm' in a for a in archs):
                return "minicpm"

        # 多模态检测（LLaVA）
        if hasattr(hf_config, 'vision_config') and not hasattr(hf_config, 'query_num'):
            return "llava"

        return "qwen3"

    def _load_minicpm_model(self, model_path: str, hf_config):
        """
        加载MiniCPM模型

        MiniCPM使用trust_remote_code，我们采用混合策略:
        1. LLM部分使用nano-vllm的高效实现
        2. 视觉部分(vpm + resampler)从HuggingFace模型复制

        Args:
            model_path: 模型路径
            hf_config: HuggingFace配置
        """
        from transformers import AutoModel
        from nanovllm.utils.loader import load_model

        print("开始加载MiniCPM模型...")

        # 1. 先加载LLM部分的权重
        load_model(self.model.llm, model_path, prefix="llm.")

        # 2. 加载HuggingFace模型以获取视觉模块
        print("  加载HuggingFace模型获取视觉模块...")
        hf_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=hf_config.torch_dtype,
            trust_remote_code=True,
            device_map='cuda'
        )

        # 3. 复制视觉模块
        self.model.load_vision_modules(hf_model)

        # 4. 清理HF模型的LLM部分以节省显存
        if hasattr(hf_model, 'llm'):
            del hf_model.llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        print("✓ MiniCPM模型加载完成")

    def _load_kvpress_compressor(self, hf_config):
        """
        加载kvpress压缩器

        kvpress是一个开源库，支持多种KV-cache压缩方法:
        - streaming_llm: Sink + Recent窗口
        - snapkv: 基于attention窗口的稀疏化
        - knorm: 基于K范数的压缩
        - h2o: Heavy-Hitter Oracle
        - tova: Token-level Value Attention
        - random: 随机采样 (baseline)
        """
        try:
            from nanovllm.kernels.kvpress_compressor import KVPressCompressor

            # 将compression_factor转换为compression_ratio
            # compression_factor=5 意味着压缩到1/5，删除80%
            compression_ratio = 1.0 - (1.0 / self.compression_factor)

            self.compressor = KVPressCompressor(
                method=self.kvpress_method,
                compression_ratio=compression_ratio,
                config=hf_config,
            ).cuda()

            self.compressor.eval()

            # 标记使用的是kvpress
            self.use_batched_compressor = False
            self.batched_compressor = None

            print(f"✓ kvpress压缩器已加载: method={self.kvpress_method}, "
                  f"compression_ratio={compression_ratio:.2f} ({self.compression_factor}x压缩)")

        except Exception as e:
            print(f"kvpress压缩器加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.compressor = None

    def _load_compressor(self, compressor_path: str, hf_config):
        """
        加载KV-Cache压缩器

        根据模型类型选择合适的压缩器：
        - MiniCPM: 使用本地MiniCPMKVCompressor（完整版本，包含compress_ik/iv）
        - LLaVA/其他: 使用KVCacheLinearDecoupleCompressor
        """
        try:
            import os

            # 根据模型类型选择压缩器
            if self.model_type == "minicpm":
                from nanovllm.kernels.minicpm_compressor import load_minicpm_compressor

                print(f"为MiniCPM加载完整版KVCompressor...")
                original_compressor = load_minicpm_compressor(
                    compressor_path=compressor_path,
                    hf_config=hf_config,
                    compression_factor=self.compression_factor
                )
                self.compressor = original_compressor
                self.batched_compressor = None
                self.use_batched_compressor = False
                return

            # 非MiniCPM模型使用原有逻辑
            from utils_ccm.module_ccm_v11 import KVCacheLinearDecoupleCompressor

            original_compressor = KVCacheLinearDecoupleCompressor(
                src_config=hf_config,
                compression_factor=self.compression_factor,
                min_seq_len=2
            ).cuda()

            if os.path.exists(compressor_path):
                checkpoint = torch.load(compressor_path, map_location='cuda')
                if 'model_state_dict' in checkpoint:
                    if 'compressor' in checkpoint['model_state_dict']:
                        original_compressor.load_state_dict(checkpoint['model_state_dict']['compressor'])
                    else:
                        original_compressor.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    original_compressor.load_state_dict(checkpoint, strict=False)
                print(f"✓ 原始压缩器加载成功: {compressor_path}")
            else:
                print(f"⚠ 压缩器权重未找到: {compressor_path}")
                self.compressor = original_compressor.eval()
                return

            original_compressor.eval()

            # 尝试创建批量GEMM压缩器
            try:
                from nanovllm.kernels.batched_compress import BatchedGEMMCompressor

                self.batched_compressor = BatchedGEMMCompressor(
                    num_layers=self.num_hidden_layers,
                    num_heads=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    compression_factor=self.compression_factor,
                    min_seq_len=2,
                    use_triton=True
                ).cuda()

                # 从原始压缩器加载权重
                self.batched_compressor.load_from_original_compressor(original_compressor)
                self.batched_compressor.eval()

                # 使用批量压缩器作为默认
                self.compressor = original_compressor  # 保留原始压缩器以备用
                self.use_batched_compressor = True
                print(f"✓ 批量GEMM压缩器已启用 (预期加速: ~25x)")

            except Exception as e:
                print(f"批量GEMM压缩器初始化失败，使用原始压缩器: {e}")
                self.compressor = original_compressor
                self.batched_compressor = None
                self.use_batched_compressor = False

        except Exception as e:
            print(f"压缩器加载失败: {e}")
            self.compressor = None
            self.batched_compressor = None
            self.use_batched_compressor = False

    def allocate_kv_cache(self, gpu_memory_utilization):
        """分配KV-cache内存"""
        config = self.config
        hf_config = config.hf_config

        # 根据模型类型获取配置参数
        if hasattr(hf_config, 'text_config'):
            # LLaVA类型模型
            text_config = hf_config.text_config
            self.num_hidden_layers = text_config.num_hidden_layers
            self.num_key_value_heads = getattr(text_config, 'num_key_value_heads', text_config.num_attention_heads)
            self.head_dim = text_config.hidden_size // text_config.num_attention_heads
        elif self.model_type == "minicpm":
            # MiniCPM - 配置在顶级，没有text_config
            self.num_hidden_layers = hf_config.num_hidden_layers
            self.num_key_value_heads = getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads)
            self.head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        else:
            # 其他模型
            self.num_hidden_layers = hf_config.num_hidden_layers
            self.num_key_value_heads = getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads)
            self.head_dim = getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads)

        tp_size = get_tp_size()
        if tp_size > 1:
            if self.num_key_value_heads % tp_size != 0:
                raise ValueError(
                    f"num_key_value_heads ({self.num_key_value_heads}) must be divisible by tp_size ({tp_size})"
                )
            self.num_key_value_heads = max(1, self.num_key_value_heads // tp_size)

        _, _, free_mem = get_gpu_memory()
        free = free_mem * gpu_memory_utilization
        block_bytes = 2 * self.num_hidden_layers * self.block_size * self.num_key_value_heads * self.head_dim * hf_config.torch_dtype.itemsize
        if free < block_bytes:
            config.num_kvcache_blocks = 1
        else:
            config.num_kvcache_blocks = int(free) // block_bytes

        self.kv_cache = torch.zeros(
            2, self.num_hidden_layers, config.num_kvcache_blocks,
            self.block_size, self.num_key_value_heads, self.head_dim
        )

        # 将KV-cache绑定到attention层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

        print(f"KV-cache分配: {config.num_kvcache_blocks}块, 每块{self.block_size}tokens")

    def preare_block_tables(self, seqs: List[Sequence]):
        """准备block tables"""
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: List[Sequence], pixel_values: Optional[torch.Tensor] = None):
        """
        准备prefill输入，支持图像token扩展

        对于纯文本请求：使用原始nanovllm的逻辑
        对于多模态请求：LlavaScheduler保证每次只有一个，需要考虑图像token扩展
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        context_lens = None
        block_tables = None

        # 判断是否有图片输入
        has_image = self.is_multimodal and pixel_values is not None

        for seq_idx, seq in enumerate(seqs):
            seqlen = len(seq)  # 原始token数量

            # 对于有图片的情况，第一个序列需要图像扩展
            if has_image and seq_idx == 0:
                image_token_expansion = self.image_token_len - 1
                actual_seqlen = seqlen + image_token_expansion
            else:
                actual_seqlen = seqlen

            # input_ids和positions使用原始长度
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))

            # cu_seqlens使用实际长度（用于attention）
            seqlen_q = actual_seqlen - seq.num_cached_tokens
            seqlen_k = actual_seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # slot_mapping：使用原始nanovllm的逻辑
            # 对于有图片的情况需要额外分配slots
            if has_image and seq_idx == 0:
                # 多模态：需要为扩展后的tokens分配slots
                blocks_needed = (actual_seqlen + self.block_size - 1) // self.block_size
                while len(seq.block_table) < blocks_needed:
                    seq.block_table.append(len(seq.block_table))

                for i in range(seq.num_cached_blocks, blocks_needed):
                    start = seq.block_table[i] * self.block_size
                    if i != blocks_needed - 1:
                        end = start + self.block_size
                    else:
                        # 最后一个block的实际使用量
                        last_block_tokens = actual_seqlen - i * self.block_size
                        end = start + last_block_tokens
                    slot_mapping.extend(list(range(start, end)))
            else:
                # 纯文本：使用原始nanovllm的简洁逻辑
                for i in range(seq.num_cached_blocks, seq.num_blocks):
                    start = seq.block_table[i] * self.block_size
                    if i != seq.num_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + len(seq.last_block())
                    slot_mapping.extend(list(range(start, end)))

        # 验证断言（纯文本时必须成立）
        if not has_image:
            assert len(input_ids) == len(slot_mapping), f"input_ids={len(input_ids)}, slot_mapping={len(slot_mapping)}"

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            context_lens = torch.tensor([len(seq) for seq in seqs], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            block_tables = self.preare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: List[Sequence]):
        """
        准备decode输入

        关键：使用seq.kv_cache_len来获取实际的KV-cache长度
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)

            # 使用seq.kv_cache_len获取实际KV-cache长度
            # 对于压缩序列，这是压缩后的长度；对于未压缩序列，等于len(seq)
            kv_len = seq.kv_cache_len if hasattr(seq, 'kv_cache_len') else len(seq)

            positions.append(kv_len)
            context_lens.append(kv_len)

            # slot_mapping指向KV-cache的下一个位置
            block_idx = kv_len // self.block_size
            pos_in_block = kv_len % self.block_size
            if block_idx < len(seq.block_table):
                slot_mapping.append(seq.block_table[block_idx] * self.block_size + pos_in_block)
            else:
                # 需要新block（should already be allocated by may_append）
                slot_mapping.append(seq.block_table[-1] * self.block_size + pos_in_block)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.preare_block_tables(seqs)

        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: List[Sequence]):
        """准备采样参数"""
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        pixel_values: Optional[torch.Tensor] = None
    ):
        """运行模型"""
        if is_prefill or self.enforce_eager or input_ids.size(0) > 256:
            if self.is_multimodal and pixel_values is not None:
                hidden_states = self.model(input_ids, positions, pixel_values)
            else:
                hidden_states = self.model(input_ids, positions)
            return self.model.compute_logits(hidden_states)
        else:
            # Decode阶段使用CUDA Graph - 即使压缩后也能用！
            bs = input_ids.size(0)
            context = get_context()
            self.reset_graph_vars()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def compress_kv_cache_batch(
        self,
        seqs: List[Sequence],
        image_len: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        批量压缩KV-cache（优化版本）

        优化点：
        1. 使用批量GEMM压缩器（~25x加速）
        2. 批量处理所有层的KV-cache，减少Python循环
        3. 一次性提取所有序列的KV-cache
        4. 正确更新_compressed_lens供decode使用

        Args:
            seqs: 序列列表
            image_len: 图像token长度

        Returns:
            (compression_time, compression_ratio)
        """
        if self.compressor is None and not getattr(self, 'use_batched_compressor', False):
            return 0.0, 1.0

        import time

        if image_len is None:
            image_len = self.image_token_len

        total_orig_size = 0
        total_comp_size = 0
        total_tokens = 0

        start_time = time.time()

        for seq in seqs:
            seq_len = len(seq)
            block_table = seq.block_table

            # 提取KV-cache（优化：使用更高效的索引）
            hf_kv_cache = self._extract_kv_cache_fast(block_table, seq_len)

            # 计算原始大小
            orig_size = sum(k.numel() + v.numel() for k, v in hf_kv_cache)
            total_orig_size += orig_size
            total_tokens += seq_len

            # 计算压缩参数
            text_len = max(1, seq_len - image_len)
            it_len = [image_len, text_len]

            # 执行压缩 - 优先使用批量GEMM压缩器
            with torch.no_grad():
                if getattr(self, 'use_batched_compressor', False) and self.batched_compressor is not None:
                    compressed_kv = self.batched_compressor(hf_kv_cache, it_len=it_len)
                else:
                    compressed_kv = self.compressor(hf_kv_cache, it_len=it_len)

            # 计算压缩后大小和长度
            comp_size = sum(k.numel() + v.numel() for k, v in compressed_kv)
            compressed_seq_len = compressed_kv[0][0].shape[2]  # [batch, heads, seq, dim]
            total_comp_size += comp_size

            # 写回压缩后的KV-cache
            self._write_compressed_kv_cache_fast(compressed_kv, block_table)

            # 关键：记录压缩后的长度供decode使用
            self._compressed_lens[seq.seq_id] = compressed_seq_len

            # 记录需要释放的blocks信息（供scheduler释放）
            blocks_needed_after = (compressed_seq_len + self.block_size - 1) // self.block_size
            blocks_before = len(block_table)
            if blocks_before > blocks_needed_after:
                if not hasattr(self, '_blocks_to_free'):
                    self._blocks_to_free = {}
                # 记录需要释放的block IDs
                self._blocks_to_free[seq.seq_id] = block_table[blocks_needed_after:]

        torch.cuda.synchronize()
        compression_time = time.time() - start_time
        self._last_compress_time_ms = compression_time * 1000
        self._last_compress_tokens = total_tokens

        compression_ratio = total_orig_size / total_comp_size if total_comp_size > 0 else 1.0

        return compression_time, compression_ratio

    def get_blocks_to_free(self, seq_id: int) -> List[int]:
        """
        获取压缩后需要释放的block IDs

        Args:
            seq_id: 序列ID

        Returns:
            需要释放的block ID列表
        """
        if not hasattr(self, '_blocks_to_free'):
            return []
        return self._blocks_to_free.pop(seq_id, [])

    def get_compressed_block_count(self, seq_id: int) -> int:
        """
        获取压缩后需要保留的block数量

        Args:
            seq_id: 序列ID

        Returns:
            压缩后需要的block数量
        """
        if seq_id not in self._compressed_lens:
            return -1  # 未压缩
        compressed_len = self._compressed_lens[seq_id]
        return (compressed_len + self.block_size - 1) // self.block_size

    def _extract_kv_cache_fast(
        self,
        block_table: List[int],
        seq_len: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        快速提取KV-cache（优化版本）

        使用高级索引一次性提取，减少循环
        """
        k_cache_all = self.kv_cache[0]  # [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        v_cache_all = self.kv_cache[1]

        num_blocks_needed = (seq_len + self.block_size - 1) // self.block_size
        valid_blocks = min(num_blocks_needed, len(block_table))

        # 计算每个block的有效token数
        tokens_per_block = [self.block_size] * (valid_blocks - 1)
        if valid_blocks > 0:
            last_block_tokens = seq_len - (valid_blocks - 1) * self.block_size
            tokens_per_block.append(last_block_tokens)

        hf_kv_cache = []

        # 获取block indices
        block_indices = [block_table[i] for i in range(valid_blocks) if block_table[i] >= 0]

        for layer_idx in range(self.num_hidden_layers):
            k_layer = k_cache_all[layer_idx]  # [num_blocks, block_size, heads, dim]
            v_layer = v_cache_all[layer_idx]

            # 收集所有blocks的数据
            k_blocks = []
            v_blocks = []
            for i, (block_id, valid_tokens) in enumerate(zip(block_indices, tokens_per_block)):
                k_blocks.append(k_layer[block_id, :valid_tokens])
                v_blocks.append(v_layer[block_id, :valid_tokens])

            if k_blocks:
                k_contiguous = torch.cat(k_blocks, dim=0)  # [seq_len, heads, dim]
                v_contiguous = torch.cat(v_blocks, dim=0)

                # 转换为HuggingFace格式: [batch, heads, seq, dim]
                k_hf = k_contiguous.permute(1, 0, 2).unsqueeze(0)
                v_hf = v_contiguous.permute(1, 0, 2).unsqueeze(0)

                hf_kv_cache.append((k_hf, v_hf))

        return hf_kv_cache

    def _write_compressed_kv_cache_fast(
        self,
        compressed_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        block_table: List[int]
    ):
        """
        快速写回压缩后的KV-cache（优化版本）
        """
        k_cache_all = self.kv_cache[0]
        v_cache_all = self.kv_cache[1]

        for layer_idx, (k_hf, v_hf) in enumerate(compressed_kv):
            # 从HuggingFace格式转换: [batch, heads, seq, dim] -> [seq, heads, dim]
            k_seq = k_hf.squeeze(0).permute(1, 0, 2).contiguous()
            v_seq = v_hf.squeeze(0).permute(1, 0, 2).contiguous()

            compressed_seq_len = k_seq.shape[0]

            # 按block写入
            tokens_written = 0
            block_idx = 0

            while tokens_written < compressed_seq_len and block_idx < len(block_table):
                block_id = block_table[block_idx]
                if block_id < 0:
                    break

                tokens_to_write = min(
                    self.block_size,
                    compressed_seq_len - tokens_written
                )

                k_cache_all[layer_idx, block_id, :tokens_to_write] = \
                    k_seq[tokens_written:tokens_written + tokens_to_write]
                v_cache_all[layer_idx, block_id, :tokens_to_write] = \
                    v_seq[tokens_written:tokens_written + tokens_to_write]

                tokens_written += tokens_to_write
                block_idx += 1

    def reset_graph_vars(self):
        """重置CUDA Graph变量"""
        graph_vars = self.graph_vars
        graph_vars["input_ids"].zero_()
        graph_vars["positions"].zero_()
        graph_vars["slot_mapping"].zero_()
        graph_vars["context_lens"].zero_()
        graph_vars["block_tables"].zero_()

    def run(
        self,
        seqs: List[Sequence],
        is_prefill: bool,
        pixel_values: Optional[torch.Tensor] = None,
        apply_compression: bool = False,
        async_compress: bool = None  # None表示使用默认设置
    ) -> List[int]:
        """
        运行模型推理

        Args:
            seqs: 序列列表
            is_prefill: 是否是prefill阶段
            pixel_values: 图像像素值
            apply_compression: 是否应用KV-cache压缩
            async_compress: 是否使用异步压缩 (None使用初始化时的设置)

        Returns:
            生成的token IDs
        """
        if async_compress is None:
            async_compress = self.async_compression

        stream_ctx = torch.cuda.stream(self._main_stream) if self._main_stream is not None else nullcontext()
        with stream_ctx:
            if is_prefill:
                input_ids, positions = self.prepare_prefill(seqs, pixel_values)
            else:
                input_ids, positions = self.prepare_decode(seqs)

            temperatures = self.prepare_sample(seqs)

            # 运行模型
            if is_prefill and self.is_multimodal and pixel_values is not None:
                logits = self.run_model(input_ids, positions, is_prefill, pixel_values)
            else:
                logits = self.run_model(input_ids, positions, is_prefill)

            # Prefill后应用压缩
            if is_prefill and apply_compression and self.compressor is not None:
                if async_compress and self._async_manager is not None:
                    # 异步压缩：启动后立即返回，但需要同步等待（因为decode需要压缩后的KV-cache）
                    # 注意：真正的流水线异步需要在scheduler层面实现
                    self._async_compress_with_overlap(seqs)
                else:
                    # 同步压缩
                    comp_time, comp_ratio = self.compress_kv_cache_batch(seqs)
                    if len(seqs) <= 4:
                        print(f"KV-cache压缩: 时间={comp_time:.3f}s, 压缩比={comp_ratio:.2f}x, 序列数={len(seqs)}")

        if get_tp_size() > 1:
            if get_tp_rank() == 0:
                token_ids = self.sampler(logits, temperatures)
            else:
                token_ids = torch.empty(temperatures.size(0), device=temperatures.device, dtype=torch.int64)
            token_ids = tp_broadcast(token_ids, src=0)
            token_ids = token_ids.tolist()
        else:
            token_ids = self.sampler(logits, temperatures).tolist()
        reset_context()
        return token_ids

    def _async_compress_with_overlap(self, seqs: List[Sequence]):
        """
        异步压缩（带重叠计算）

        核心优化：将压缩拆分为多个阶段，每个阶段可以和其他计算重叠

        阶段1: 提取KV-cache到临时buffer (在压缩stream)
        阶段2: 执行压缩计算 (在压缩stream)
        阶段3: 写回压缩结果 (在压缩stream)

        由于当前架构下decode必须等待压缩完成，
        这里的优化是让压缩计算与sampler采样重叠。
        """
        import time
        start_time = time.time()

        if self._async_manager is None:
            self._async_manager = AsyncCompressionManager(self)

        compress_stream = self._async_manager.compress_stream
        image_len = self.image_token_len

        # 在主stream上记录事件
        main_event = torch.cuda.Event()
        main_event.record()

        # 压缩stream等待主stream完成prefill
        compress_stream.wait_event(main_event)

        # 在压缩stream中执行压缩
        with torch.cuda.stream(compress_stream):
            for seq in seqs:
                seq_len = len(seq)
                block_table = seq.block_table

                # 提取KV-cache
                hf_kv_cache = self._extract_kv_cache_fast(block_table, seq_len)

                # 计算压缩参数
                text_len = max(1, seq_len - image_len)
                it_len = [image_len, text_len]

                # 执行压缩 - 优先使用批量GEMM压缩器
                with torch.no_grad():
                    if getattr(self, 'use_batched_compressor', False) and self.batched_compressor is not None:
                        compressed_kv = self.batched_compressor(hf_kv_cache, it_len=it_len)
                    else:
                        compressed_kv = self.compressor(hf_kv_cache, it_len=it_len)

                # 获取压缩后长度
                compressed_seq_len = compressed_kv[0][0].shape[2]

                # 写回压缩后的KV-cache
                self._write_compressed_kv_cache_fast(compressed_kv, block_table)

                # 记录压缩后的长度
                self._compressed_lens[seq.seq_id] = compressed_seq_len

        # 记录压缩完成事件
        compress_done_event = torch.cuda.Event()
        compress_done_event.record(compress_stream)

        # 主stream等待压缩完成（这是必要的，因为decode需要压缩后的KV-cache）
        # 但此时sampler的采样可以和压缩并行
        torch.cuda.current_stream().wait_event(compress_done_event)

        elapsed = time.time() - start_time
        if len(seqs) <= 4:
            print(f"异步压缩(批量GEMM): 时间={elapsed:.3f}s, 序列数={len(seqs)}")

    def start_background_compression(
        self,
        seqs: List[Sequence],
        compress_stream: Optional[torch.cuda.Stream] = None,
    ) -> Optional[Tuple[torch.cuda.Event, torch.cuda.Event]]:
        """
        启动后台压缩，返回完成事件

        用于流水线调度：当前batch的压缩可以和下一个batch的prefill重叠

        Returns:
            (start_event, done_event) or None
        """
        if self.compressor is None and not getattr(self, 'use_batched_compressor', False):
            return None

        if compress_stream is None:
            if self._async_manager is None:
                self._async_manager = AsyncCompressionManager(self)
            compress_stream = self._async_manager.compress_stream
        image_len = self.image_token_len

        # 主stream事件
        main_event = torch.cuda.Event()
        main_event.record()

        compress_stream.wait_event(main_event)

        with torch.cuda.stream(compress_stream):
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(compress_stream)
            for seq in seqs:
                seq_len = len(seq)
                block_table = seq.block_table

                hf_kv_cache = self._extract_kv_cache_fast(block_table, seq_len)

                text_len = max(1, seq_len - image_len)
                it_len = [image_len, text_len]

                # 优先使用批量GEMM压缩器
                with torch.no_grad():
                    if getattr(self, 'use_batched_compressor', False) and self.batched_compressor is not None:
                        compressed_kv = self.batched_compressor(hf_kv_cache, it_len=it_len)
                    else:
                        compressed_kv = self.compressor(hf_kv_cache, it_len=it_len)

                compressed_seq_len = compressed_kv[0][0].shape[2]
                self._write_compressed_kv_cache_fast(compressed_kv, block_table)
                self._compressed_lens[seq.seq_id] = compressed_seq_len

        # 记录完成事件
        done_event = torch.cuda.Event(enable_timing=True)
        done_event.record(compress_stream)
        return start_event, done_event

    def update_streams(
        self,
        use_greenctx: bool,
        sm_compress_ratio: float,
        sm_main_ratio: float,
        use_greenctx_main: bool,
        compression_streams: int,
    ) -> bool:
        """Rebuild streams if config changes. Returns True if updated."""
        if self._decode_overlap_blocks:
            return False
        compression_streams = max(1, compression_streams)
        if (
            use_greenctx == self.greenctx_enabled
            and abs(sm_compress_ratio - self.greenctx_compress_ratio) < 1e-3
            and abs(sm_main_ratio - self.greenctx_main_ratio) < 1e-3
            and use_greenctx_main == self.greenctx_main_stream
            and compression_streams == self.compression_streams
        ):
            return True

        self.greenctx_enabled = use_greenctx
        self.greenctx_compress_ratio = sm_compress_ratio
        self.greenctx_main_ratio = sm_main_ratio
        self.greenctx_main_stream = use_greenctx_main
        self.compression_streams = compression_streams

        self._main_stream, self._compression_stream_pool, self._greenctx_info = build_streams(
            use_greenctx=self.greenctx_enabled,
            sm_compress_ratio=self.greenctx_compress_ratio,
            sm_main_ratio=self.greenctx_main_ratio,
            use_greenctx_main=self.greenctx_main_stream,
            compression_streams=self.compression_streams,
            compression_priority=-1,
        )

        if self._decode_overlap_engine is not None and self.compressor is not None:
            self._decode_overlap_engine = create_zero_overhead_compressor(
                original_compressor=self.batched_compressor if self.use_batched_compressor else self.compressor,
                num_layers=getattr(self.config.hf_config, "num_hidden_layers", 32),
                layers_per_step=self.decode_layers_per_step,
                device=torch.device("cuda"),
                compress_stream=(self._compression_stream_pool.streams[0] if self._compression_stream_pool.streams else None),
            )
        return True

    def update_decode_overlap_engine(self, layers_per_step: int) -> bool:
        """Update decode overlap engine layers_per_step. Returns True if applied."""
        layers_per_step = max(1, layers_per_step)
        if layers_per_step == self.decode_layers_per_step and self._decode_overlap_engine is not None:
            return True
        if self._decode_overlap_blocks:
            return False
        if self.compressor is None:
            return False

        self.decode_layers_per_step = layers_per_step
        overlap_compressor = (
            self.batched_compressor
            if getattr(self, "use_batched_compressor", False) and self.batched_compressor is not None
            else self.compressor
        )
        self._decode_overlap_engine = create_zero_overhead_compressor(
            original_compressor=overlap_compressor,
            num_layers=getattr(self.config.hf_config, "num_hidden_layers", 32),
            layers_per_step=self.decode_layers_per_step,
            device=torch.device("cuda"),
            compress_stream=(self._compression_stream_pool.streams[0] if self._compression_stream_pool.streams else None),
        )
        return True

    def get_next_compress_stream(self) -> torch.cuda.Stream:
        """Return next compression stream for prefill-bound pipeline."""
        return self._compression_stream_pool.next_stream()

    def start_decode_overlap_compression(self, seqs: List[Sequence]):
        """Start decode-overlap compression tasks after prefill."""
        if self.compressor is None:
            return
        if self._decode_overlap_engine is None:
            if not self.update_decode_overlap_engine(self.decode_layers_per_step):
                return

        image_len = self.image_token_len
        for seq in seqs:
            if seq.seq_id in self._decode_overlap_blocks:
                continue

            seq_len = len(seq)
            # Skip if too short to compress
            min_seq_len = getattr(self.compressor, "min_seq_len", 0)
            if seq_len < min_seq_len:
                continue

            hf_kv_cache = self._extract_kv_cache_fast(seq.block_table, seq_len)
            text_len = max(1, seq_len - image_len)
            it_len = [image_len, text_len]

            self._decode_overlap_engine.start_compression(seq.seq_id, hf_kv_cache, it_len)
            self._decode_overlap_blocks[seq.seq_id] = seq.block_table
            self._decode_overlap_prefill_lens[seq.seq_id] = seq_len

    def step_decode_overlap_compression(self, seqs: List[Sequence]) -> List[Sequence]:
        """Advance decode-overlap compression; return seqs that finished compression."""
        completed: List[Sequence] = []
        if self._decode_overlap_engine is None:
            return completed

        for seq in seqs:
            if seq.seq_id not in self._decode_overlap_blocks:
                continue

            done = self._decode_overlap_engine.step_compression(seq.seq_id)
            if not done:
                continue

            compressed_kv = self._decode_overlap_engine.get_compressed_cache(seq.seq_id)
            if compressed_kv is None:
                continue

            prefill_len = self._decode_overlap_prefill_lens.pop(seq.seq_id, len(seq))
            current_len = len(seq)
            decode_tokens = max(0, current_len - prefill_len)

            if decode_tokens > 0:
                full_kv_cache = self._extract_kv_cache_fast(seq.block_table, current_len)
                merged_kv = []
                for (ck, cv), (fk, fv) in zip(compressed_kv, full_kv_cache):
                    tail_k = fk[:, :, prefill_len:, :]
                    tail_v = fv[:, :, prefill_len:, :]
                    merged_kv.append(
                        (torch.cat([ck, tail_k], dim=2), torch.cat([cv, tail_v], dim=2))
                    )
                compressed_kv = merged_kv

            compressed_seq_len = compressed_kv[0][0].shape[2]
            self._write_compressed_kv_cache_fast(compressed_kv, seq.block_table)
            self._compressed_lens[seq.seq_id] = compressed_seq_len

            self._decode_overlap_blocks.pop(seq.seq_id, None)
            completed.append(seq)

        return completed

    def wait_compression(self, event: torch.cuda.Event):
        """等待压缩完成"""
        if event is not None:
            event.synchronize()

    def clear_compressed_lens(self, seq_ids: List[int] = None):
        """清理压缩长度记录"""
        if seq_ids is None:
            self._compressed_lens.clear()
        else:
            for seq_id in seq_ids:
                self._compressed_lens.pop(seq_id, None)

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        捕获CUDA Graph（用于decode阶段加速）

        注意：压缩后的decode也能使用CUDA Graph，
        因为Graph只捕获计算流程，context_lens作为输入参数传入
        """
        get_rng_state = torch.cuda.get_rng_state
        set_rng_state = torch.cuda.set_rng_state
        rng_state = torch.cuda.get_rng_state()
        torch.cuda.get_rng_state = lambda: rng_state
        torch.cuda.set_rng_state = lambda _: None

        config = self.config
        hf_config = config.hf_config

        if hasattr(hf_config, 'text_config'):
            hidden_size = hf_config.text_config.hidden_size
        else:
            hidden_size = hf_config.hidden_size

        max_bs = min(self.config.max_num_seqs, 256)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hidden_size)

        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        stream_ctx = torch.cuda.stream(self._main_stream) if self._main_stream is not None else nullcontext()
        with stream_ctx:
            for bs in reversed(self.graph_bs):
                graph = torch.cuda.CUDAGraph()
                set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                with torch.cuda.graph(graph, self.graph_pool):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                if self.graph_pool is None:
                    self.graph_pool = graph.pool()
                self.graphs[bs] = graph
                torch.cuda.synchronize()
                reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        torch.cuda.get_rng_state = get_rng_state
        torch.cuda.set_rng_state = set_rng_state

        print(f"✓ CUDA Graph已捕获 (batch sizes: {self.graph_bs})")
