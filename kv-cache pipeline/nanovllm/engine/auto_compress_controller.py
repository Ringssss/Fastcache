from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from nanovllm.autocompress.context import WorkloadContext, detect_hardware
from nanovllm.autocompress.methods import get_method
from nanovllm.autocompress.pipeline import PipelineOrchestrator, PipelineStrategy
from nanovllm.autocompress.profiles import ComputeType


class EWMAMeter:
    def __init__(self, alpha: float = 0.2):
        self._alpha = alpha
        self._value: Optional[float] = None
        self.samples = 0

    @property
    def value(self) -> Optional[float]:
        return self._value

    def update(self, value: Optional[float]) -> None:
        if value is None:
            return
        if self._value is None:
            self._value = value
        else:
            self._value = self._alpha * value + (1 - self._alpha) * self._value
        self.samples += 1


@dataclass
class AutoCompressionDecision:
    pipeline: str  # prefill | decode | none
    async_compress: bool
    compression_streams: int
    decode_layers_per_step: int
    greenctx_enabled: bool
    sm_compress_ratio: float
    sm_main_ratio: float


class AutoCompressionController:
    def __init__(
        self,
        *,
        compression_backend: str,
        kvpress_method: Optional[str],
        compression_factor: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        greenctx_enabled: bool,
        decision_interval: int = 4,
        min_seq_len: int = 256,
        min_output_len: int = 64,
        min_speedup: float = 1.05,
        min_sm_ratio: float = 0.10,
        max_sm_ratio: float = 0.60,
        greenctx_min_batch: int = 8,
        greenctx_min_output_len: int = 128,
        force_compress: bool = False,
        dual_batch_enabled: bool = True,
        dual_batch_interval: int = 4,
        dual_batch_min_decode_share: float = 0.20,
        dual_batch_max_decode_share: float = 0.80,
    ):
        self._compression_backend = compression_backend
        self._kvpress_method = kvpress_method
        self._compression_factor = compression_factor
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._greenctx_enabled = greenctx_enabled
        self._decision_interval = max(1, decision_interval)
        self._min_seq_len = min_seq_len
        self._min_output_len = min_output_len
        self._min_speedup = min_speedup
        self._min_sm_ratio = min_sm_ratio
        self._max_sm_ratio = max_sm_ratio
        self._greenctx_min_batch = greenctx_min_batch
        self._greenctx_min_output_len = greenctx_min_output_len
        self._force_compress = force_compress
        self._dual_batch_enabled = dual_batch_enabled
        self._dual_batch_interval = max(1, dual_batch_interval)
        self._dual_batch_min_decode_share = dual_batch_min_decode_share
        self._dual_batch_max_decode_share = dual_batch_max_decode_share

        self._hw_profile = detect_hardware()
        self._orchestrator = PipelineOrchestrator(self._hw_profile, min_speedup=min_speedup)

        self._method_name, self._method_kwargs = self._build_method()
        self._method = get_method(self._method_name, **self._method_kwargs)
        self._compute_type = self._method.profile.compute_type

        if self._compute_type == ComputeType.GEMM:
            self._base_sm_ratio = 0.40
        elif self._compute_type == ComputeType.MEMORY:
            self._base_sm_ratio = 0.20
        else:
            self._base_sm_ratio = 0.25

        self._sm_ratio = self._base_sm_ratio
        self._compression_streams = 2
        self._decode_layers_per_step = 4
        self._pipeline = "prefill"
        self._last_decision: Optional[AutoCompressionDecision] = None
        self._decision_steps = 0
        self._dual_step = 0
        self._dual_decode_budget = 0
        self._dual_decode_share = 0.5
        self._dual_last_choice = "prefill"

        self._prefill_ms_per_token = EWMAMeter()
        self._prefill_step_ms = EWMAMeter()
        self._decode_step_ms = EWMAMeter()
        self._compress_ms = EWMAMeter()
        self._compress_ms_per_layer = EWMAMeter()

    def decide_batch_type(
        self,
        *,
        waiting: int,
        running: int,
        pending_compressions: int,
    ) -> str:
        if waiting <= 0:
            return "decode"
        if running <= 0:
            return "prefill"
        if not self._dual_batch_enabled:
            return "prefill"

        if self._pipeline == "none":
            return "prefill"
        if self._pipeline == "prefill":
            return "prefill"

        self._dual_step += 1
        total = max(1, waiting + running)
        run_ratio = running / total
        wait_ratio = waiting / total

        if wait_ratio >= 0.65:
            return "prefill"
        if run_ratio >= 0.85 and wait_ratio <= 0.15:
            return "decode"

        if self._prefill_step_ms.value and self._decode_step_ms.value:
            prefill_ms = self._prefill_step_ms.value
            decode_ms = self._decode_step_ms.value
            if prefill_ms > decode_ms * 1.80 and self._pipeline != "decode":
                return "prefill"
            if decode_ms > prefill_ms * 1.80 and wait_ratio <= 0.40:
                return "decode"

        if (self._dual_step % self._dual_batch_interval) == 1 or self._dual_decode_budget <= 0:
            target_decode = run_ratio

            if self._decode_step_ms.value and self._prefill_step_ms.value:
                ratio = self._decode_step_ms.value / max(self._prefill_step_ms.value, 1e-6)
                if ratio > 1.25:
                    target_decode += 0.15
                elif ratio < 0.80:
                    target_decode -= 0.15

            if self._pipeline == "prefill":
                target_decode -= 0.20
            elif self._pipeline == "decode":
                target_decode += 0.15

            min_share = self._dual_batch_min_decode_share
            max_share = self._dual_batch_max_decode_share
            if self._pipeline == "prefill":
                min_share = min(min_share, 0.10)
                max_share = min(max_share, 0.65)
            elif self._pipeline == "decode":
                min_share = max(min_share, 0.25)
                max_share = max(max_share, 0.85)

            target_decode = max(min_share, min(max_share, target_decode))
            self._dual_decode_share = target_decode
            budget = int(round(target_decode * self._dual_batch_interval))
            if target_decode < 0.15:
                budget = 0
            self._dual_decode_budget = max(0, min(self._dual_batch_interval - 1, budget))

        if self._dual_decode_budget > 0:
            self._dual_decode_budget -= 1
            self._dual_last_choice = "decode"
            return "decode"
        self._dual_last_choice = "prefill"
        return "prefill"

    def _build_method(self) -> tuple[str, dict]:
        if self._compression_backend == "kvpress":
            method = self._kvpress_method or "streaming_llm"
            drop_ratio = 1.0 - (1.0 / max(self._compression_factor, 1))
            return f"kvpress_{method}", {"compression_ratio": drop_ratio}
        if self._compression_backend == "mlp":
            return "mlp", {
                "compression_ratio": float(self._compression_factor),
                "num_layers": self._num_layers,
            }
        return "mlp", {
            "compression_ratio": float(self._compression_factor),
            "num_layers": self._num_layers,
        }

    def update_prefill(self, elapsed_ms: float, tokens: int) -> None:
        if tokens > 0:
            self._prefill_ms_per_token.update(elapsed_ms / tokens)
        self._prefill_step_ms.update(elapsed_ms)

    def update_decode(self, elapsed_ms: float, batch_size: int) -> None:
        self._decode_step_ms.update(elapsed_ms)

    def update_compress(self, elapsed_ms: float) -> None:
        self._compress_ms.update(elapsed_ms)
        if self._num_layers > 0:
            self._compress_ms_per_layer.update(elapsed_ms / self._num_layers)

    def _maybe_adjust_sm_ratio(self, pending_compressions: int) -> float:
        ratio = self._sm_ratio
        if self._compute_type == ComputeType.GEMM:
            if self._compress_ms.value and self._decode_step_ms.value:
                if self._compress_ms.value > self._decode_step_ms.value * 1.10:
                    ratio += 0.05
                elif self._compress_ms.value < self._decode_step_ms.value * 0.60:
                    ratio -= 0.05
        else:
            if self._compress_ms.value and self._prefill_step_ms.value:
                if self._compress_ms.value > self._prefill_step_ms.value * 0.80:
                    ratio += 0.05
                elif self._compress_ms.value < self._prefill_step_ms.value * 0.40:
                    ratio -= 0.05

        if pending_compressions > self._compression_streams:
            ratio += 0.05

        ratio = max(self._min_sm_ratio, min(self._max_sm_ratio, ratio))
        return ratio

    def _maybe_adjust_streams(self, pending_compressions: int) -> int:
        streams = self._compression_streams
        if self._compress_ms.value and self._prefill_step_ms.value:
            ratio = self._compress_ms.value / max(self._prefill_step_ms.value, 1e-6)
            if ratio > 0.80:
                streams += 1
            elif ratio < 0.40:
                streams -= 1

        if pending_compressions > streams:
            streams += 1
        if pending_compressions == 0 and streams > 1:
            streams -= 1

        return max(1, min(4, streams))

    def _maybe_adjust_layers_per_step(self) -> int:
        layers = self._decode_layers_per_step
        if self._compress_ms_per_layer.value and self._decode_step_ms.value:
            target = self._decode_step_ms.value / max(self._compress_ms_per_layer.value, 1e-6)
            candidate = int(round(target))
            candidate = max(1, min(8, candidate))
            if abs(candidate - layers) >= 2:
                layers = layers + 1 if candidate > layers else layers - 1
            elif candidate != layers:
                layers = candidate
        return layers

    def decide(
        self,
        *,
        batch_size: int,
        seq_len: int,
        expected_output_len: int,
        pending_compressions: int,
    ) -> AutoCompressionDecision:
        self._decision_steps += 1
        if (
            self._last_decision is not None
            and (self._decision_steps % self._decision_interval) != 0
        ):
            return self._last_decision

        workload = WorkloadContext(
            batch_size=batch_size,
            seq_len=seq_len,
            expected_output_len=expected_output_len,
            num_layers=self._num_layers,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
        )

        if (
            seq_len < self._min_seq_len
            or expected_output_len < self._min_output_len
        ) and not self._force_compress:
            decision = AutoCompressionDecision(
                pipeline="none",
                async_compress=False,
                compression_streams=self._compression_streams,
                decode_layers_per_step=self._decode_layers_per_step,
                greenctx_enabled=False,
                sm_compress_ratio=0.0,
                sm_main_ratio=1.0,
            )
            self._pipeline = "none"
            self._last_decision = decision
            return decision

        plan = self._orchestrator.create_pipeline(
            self._method_name,
            workload,
            **self._method_kwargs,
        )

        if plan.estimated_speedup < self._min_speedup and not self._force_compress:
            pipeline = "none"
        else:
            pipeline = (
                "decode"
                if plan.strategy == PipelineStrategy.ASYNC_WITH_DECODE
                else "prefill"
            )

        sm_ratio = self._maybe_adjust_sm_ratio(pending_compressions)
        sm_main = max(0.05, 1.0 - sm_ratio)

        greenctx_enabled = self._greenctx_enabled
        if pipeline == "none":
            greenctx_enabled = False
        elif self._compute_type == ComputeType.GEMM:
            if batch_size < self._greenctx_min_batch or expected_output_len < self._greenctx_min_output_len:
                greenctx_enabled = False
        if greenctx_enabled and self._compress_ms.samples >= 3:
            if pipeline == "prefill" and self._prefill_step_ms.value:
                ratio = self._compress_ms.value / max(self._prefill_step_ms.value, 1e-6)
                if ratio < 0.35:
                    greenctx_enabled = False
            elif pipeline == "decode" and self._decode_step_ms.value:
                ratio = self._compress_ms.value / max(self._decode_step_ms.value, 1e-6)
                if ratio < 0.35:
                    greenctx_enabled = False

        compression_streams = self._compression_streams
        decode_layers = self._decode_layers_per_step

        if pipeline == "prefill":
            compression_streams = self._maybe_adjust_streams(pending_compressions)
        elif pipeline == "decode":
            decode_layers = self._maybe_adjust_layers_per_step()
            if plan.chunk_size > 0:
                decode_layers = max(1, min(8, plan.chunk_size))

        async_compress = False
        if pipeline == "prefill":
            if pending_compressions > 0:
                async_compress = True
            elif self._compress_ms.value and self._prefill_step_ms.value:
                async_compress = self._compress_ms.value > self._prefill_step_ms.value * 0.40
            elif batch_size >= 8:
                async_compress = True

        decision = AutoCompressionDecision(
            pipeline=pipeline,
            async_compress=async_compress,
            compression_streams=compression_streams,
            decode_layers_per_step=decode_layers,
            greenctx_enabled=greenctx_enabled,
            sm_compress_ratio=sm_ratio,
            sm_main_ratio=sm_main,
        )

        self._sm_ratio = sm_ratio
        self._compression_streams = compression_streams
        self._decode_layers_per_step = decode_layers
        self._pipeline = pipeline
        self._last_decision = decision
        return decision
