"""
LLaVA LLM Engine for nano-vllm
==============================

支持LLaVA多模态模型和KV-Cache压缩的推理引擎

Date: 2024
"""

import torch
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from PIL import Image
from typing import Optional, List, Union, Tuple

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.llava_scheduler import LlavaScheduler
from nanovllm.engine.llava_model_runner import LlavaModelRunner


class LlavaSequence(Sequence):
    """扩展Sequence以支持多模态数据"""

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams,
        pixel_values: Optional[torch.Tensor] = None,
        image_token_len: int = 0
    ):
        super().__init__(token_ids, sampling_params)
        self.pixel_values = pixel_values
        self.image_token_len = image_token_len
        self.has_image = pixel_values is not None


class LlavaLLMEngine:
    """
    支持LLaVA和KV-Cache压缩的推理引擎

    支持两种压缩后端:
    - mlp: 基于MLP的GEMM压缩器 (需要compressor_path)
    - kvpress: kvpress库的压缩方法 (streaming_llm, snapkv, knorm等)
    """

    def __init__(
        self,
        model: str,
        compressor_path: Optional[str] = None,
        compression_factor: int = 5,
        enable_compression: bool = False,
        async_compression: bool = False,  # 异步压缩
        compression_backend: str = 'mlp',  # 压缩后端: 'mlp' 或 'kvpress'
        kvpress_method: str = 'streaming_llm',  # kvpress方法名
        **kwargs
    ):
        """
        初始化LLaVA推理引擎

        Args:
            model: 模型路径
            compressor_path: 压缩器权重路径 (仅mlp后端需要)
            compression_factor: 压缩因子 (5表示5x压缩)
            enable_compression: 是否启用压缩
            async_compression: 是否启用异步压缩
            compression_backend: 压缩后端 ('mlp' 或 'kvpress')
            kvpress_method: kvpress压缩方法名
            **kwargs: 其他配置参数
        """
        config = Config(model)
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

        Sequence.block_size = config.kvcache_block_size
        config.hf_config = AutoConfig.from_pretrained(config.model, trust_remote_code=True)

        # 处理LLaVA配置
        if hasattr(config.hf_config, 'text_config'):
            max_pos = getattr(config.hf_config.text_config, 'max_position_embeddings', 4096)
        else:
            max_pos = getattr(config.hf_config, 'max_position_embeddings', 4096)
        config.max_model_len = min(config.max_model_len, max_pos)

        # 加载tokenizer和processor
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        try:
            self.processor = AutoProcessor.from_pretrained(config.model, trust_remote_code=True)
        except Exception:
            self.processor = None
            print("Warning: Could not load processor, using tokenizer only")

        config.eos = self.tokenizer.eos_token_id

        # 使用LlavaModelRunner
        self.model_runner = LlavaModelRunner(
            config,
            compressor_path=compressor_path,
            compression_factor=compression_factor,
            enable_compression=enable_compression,
            async_compression=async_compression,
            compression_backend=compression_backend,
            kvpress_method=kvpress_method,
        )
        # 使用LlavaScheduler（支持多模态的调度器）
        self.scheduler = LlavaScheduler(config)

        self.enable_compression = enable_compression
        self.async_compression = async_compression
        self.compression_factor = compression_factor
        self.compression_backend = compression_backend
        self.kvpress_method = kvpress_method

        # 获取图像token相关信息
        if self.model_runner.is_multimodal:
            self.image_token_len = self.model_runner.image_token_len
            self.image_token_id = 32000  # LLaVA默认
        else:
            self.image_token_len = 0
            self.image_token_id = None

        print(f"LLaVA Engine初始化完成")
        print(f"  - 多模态: {self.model_runner.is_multimodal}")
        print(f"  - 压缩启用: {enable_compression}")
        print(f"  - 压缩后端: {compression_backend}")
        if compression_backend == 'kvpress':
            print(f"  - kvpress方法: {kvpress_method}")
        print(f"  - 异步压缩: {async_compression}")
        print(f"  - 压缩因子: {compression_factor}")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """预处理图像"""
        if self.processor is None:
            raise ValueError("Processor not available for image processing")

        # 使用image_processor组件处理图像（避免LlavaProcessor需要text参数的问题）
        if hasattr(self.processor, 'image_processor'):
            inputs = self.processor.image_processor(images=image, return_tensors="pt")
        else:
            # 回退到完整processor，但提供dummy text
            inputs = self.processor(text="", images=image, return_tensors="pt")

        # 转换为模型使用的dtype（通常是float16）
        pixel_values = inputs.pixel_values.cuda()
        if hasattr(self.model_runner.model, 'vision_tower'):
            # 获取vision tower的dtype
            vision_dtype = next(self.model_runner.model.vision_tower.parameters()).dtype
            pixel_values = pixel_values.to(vision_dtype)

        return pixel_values

    def add_request(
        self,
        prompt: Union[str, list[int]],
        sampling_params: SamplingParams,
        image: Optional[Image.Image] = None
    ):
        """
        添加推理请求

        Args:
            prompt: 文本提示或token IDs
            sampling_params: 采样参数
            image: 可选的输入图像
        """
        pixel_values = None
        image_token_len = 0

        if image is not None:
            pixel_values = self.preprocess_image(image)
            image_token_len = self.image_token_len

        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        seq = LlavaSequence(
            prompt,
            sampling_params,
            pixel_values=pixel_values,
            image_token_len=image_token_len
        )
        self.scheduler.add(seq)

    def step(self, apply_compression: bool = False) -> Tuple[list, int]:
        """
        执行一步推理

        Args:
            apply_compression: 是否在prefill后应用压缩

        Returns:
            (outputs, num_tokens)
        """
        seqs, is_prefill = self.scheduler.schedule()

        # 收集图像数据
        # 注意：当前架构每次prefill只支持处理一张图片
        # 由于LlavaScheduler保证多模态序列每次只调度一个，这里是安全的
        pixel_values = None
        multimodal_seq = None
        if is_prefill:
            for seq in seqs:
                if isinstance(seq, LlavaSequence) and seq.has_image:
                    pixel_values = seq.pixel_values
                    multimodal_seq = seq
                    break

        # 流水线异步压缩：在decode前等待之前提交的压缩完成
        if not is_prefill and self.async_compression and self.enable_compression:
            self._wait_pending_compressions(seqs)
            # 异步压缩完成后，释放多余的blocks
            self._free_compressed_blocks(seqs)

        # 运行模型
        token_ids = self.model_runner.run(
            seqs,
            is_prefill,
            pixel_values=pixel_values,
            apply_compression=(apply_compression and is_prefill and self.enable_compression and not self.async_compression)
        )

        # 同步压缩后，立即释放多余的blocks
        if is_prefill and apply_compression and self.enable_compression and not self.async_compression:
            self._free_compressed_blocks(seqs)

        # 流水线异步压缩：prefill后立即启动异步压缩（不等待）
        if is_prefill and apply_compression and self.enable_compression and self.async_compression:
            self._start_pipeline_compression(seqs)

        self.scheduler.postprocess(seqs, token_ids)

        outputs = [
            (seq.seq_id, seq[seq.num_prompt_tokens:])
            for seq in seqs
            if seq.status == SequenceStatus.FINISHED
        ]

        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def _start_pipeline_compression(self, seqs):
        """
        启动流水线异步压缩

        不等待压缩完成，让它在后台执行，这样可以与下一个batch的prefill重叠
        """
        if not hasattr(self, '_pending_compression_events'):
            self._pending_compression_events = {}

        # 启动异步压缩并保存事件
        event = self.model_runner.start_background_compression(seqs)
        if event is not None:
            for seq in seqs:
                self._pending_compression_events[seq.seq_id] = event

    def _wait_pending_compressions(self, seqs):
        """
        等待序列的压缩完成

        在decode开始前调用，确保KV-cache已经压缩完成
        """
        if not hasattr(self, '_pending_compression_events'):
            return

        for seq in seqs:
            event = self._pending_compression_events.get(seq.seq_id)
            if event is not None:
                event.synchronize()
                del self._pending_compression_events[seq.seq_id]

    def _free_compressed_blocks(self, seqs):
        """
        释放压缩后多余的blocks

        压缩后KV-cache长度减少，多余的blocks可以释放回block_manager
        这是实现真正内存节省的关键！

        重要：
        1. 设置seq.kv_cache_len为压缩后的长度
        2. 释放多余的blocks
        3. 重置最后一个block的hash状态
        """
        block_manager = self.scheduler.block_manager

        for seq in seqs:
            # 获取压缩后的KV-cache长度
            compressed_len = self.model_runner._compressed_lens.get(seq.seq_id)
            if compressed_len is None:
                continue

            # 关键：设置seq的kv_cache_len，用于后续may_append计算
            seq.kv_cache_len = compressed_len

            # 获取压缩后需要保留的block数量
            blocks_to_keep = self.model_runner.get_compressed_block_count(seq.seq_id)

            if blocks_to_keep > 0:
                # 计算当前block数量
                current_blocks = len(seq.block_table)

                if current_blocks > blocks_to_keep:
                    # 释放多余的blocks
                    blocks_to_free = seq.block_table[blocks_to_keep:]
                    freed_count = 0

                    for block_id in blocks_to_free:
                        if block_id in block_manager.used_block_ids:
                            block = block_manager.blocks[block_id]
                            block.ref_count -= 1
                            if block.ref_count == 0:
                                block_manager._deallocate_block(block_id)
                                freed_count += 1

                    # 更新sequence的block_table
                    seq.block_table = seq.block_table[:blocks_to_keep]

                    # 重置最后一个block的hash状态为-1
                    if seq.block_table:
                        last_block_id = seq.block_table[-1]
                        last_block = block_manager.blocks[last_block_id]
                        last_block.hash = -1
                        last_block.token_ids = []

                    if freed_count > 0:
                        print(f"[Memory] 序列{seq.seq_id}: 释放了{freed_count}个blocks "
                              f"({current_blocks}->{blocks_to_keep}), "
                              f"空闲blocks: {len(block_manager.free_block_ids)}")

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: List[str],
        sampling_params: Union[SamplingParams, List[SamplingParams]],
        images: Optional[List[Image.Image]] = None,
        use_tqdm: bool = True,
        apply_compression: bool = False,
    ) -> List[dict]:
        """
        批量生成

        Args:
            prompts: 提示列表
            sampling_params: 采样参数
            images: 图像列表（可选）
            use_tqdm: 是否显示进度条
            apply_compression: 是否应用压缩

        Returns:
            生成结果列表
        """
        if use_tqdm:
            pbar = tqdm(
                total=len(prompts),
                desc="Generating",
                dynamic_ncols=True,
            )

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        if images is None:
            images = [None] * len(prompts)

        for prompt, sp, image in zip(prompts, sampling_params, images):
            self.add_request(prompt, sp, image)

        outputs = {}
        prefill_throughput = decode_throughput = 0.

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step(apply_compression=apply_compression)

            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [
            {"text": self.tokenizer.decode(token_ids, skip_special_tokens=True), "token_ids": token_ids}
            for token_ids in outputs
        ]

        if use_tqdm:
            pbar.close()

        return outputs


class LlavaLLM(LlavaLLMEngine):
    """便捷接口类"""
    pass
