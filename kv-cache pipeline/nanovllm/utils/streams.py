"""CUDA stream helpers (greenctx-aware)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class GreenCtxInfo:
    available: bool
    reason: Optional[str] = None


class StreamPool:
    """Round-robin pool for compression streams."""

    def __init__(self, streams: List[torch.cuda.Stream]):
        self._streams = streams
        self._idx = 0

    def next_stream(self) -> torch.cuda.Stream:
        if not self._streams:
            return torch.cuda.current_stream()
        stream = self._streams[self._idx]
        self._idx = (self._idx + 1) % len(self._streams)
        return stream

    @property
    def streams(self) -> List[torch.cuda.Stream]:
        return self._streams


def _get_sm_count(device_id: int) -> int:
    props = torch.cuda.get_device_properties(device_id)
    return props.multi_processor_count


def _try_import_sgl_kernel() -> Tuple[Optional[object], GreenCtxInfo]:
    try:
        import sgl_kernel  # type: ignore

        return sgl_kernel, GreenCtxInfo(available=True)
    except Exception as exc:  # pragma: no cover - import error path
        return None, GreenCtxInfo(available=False, reason=str(exc))


def try_create_greenctx_streams(
    sm_compress_ratio: float,
    sm_main_ratio: float,
    device_id: Optional[int] = None,
) -> Tuple[Optional[torch.cuda.Stream], Optional[torch.cuda.Stream], GreenCtxInfo]:
    """Create greenctx streams for compress/main. Returns (compress, main, info)."""
    if device_id is None:
        device_id = torch.cuda.current_device()

    sgl_kernel, info = _try_import_sgl_kernel()
    if not info.available or sgl_kernel is None:
        return None, None, info

    if not hasattr(sgl_kernel, "create_greenctx_stream_by_value"):
        return None, None, GreenCtxInfo(available=False, reason="missing create_greenctx_stream_by_value")

    try:
        sm_total = (
            sgl_kernel.get_sm_available(device_id)
            if hasattr(sgl_kernel, "get_sm_available")
            else _get_sm_count(device_id)
        )
        sm_compress = max(1, int(sm_total * sm_compress_ratio))
        sm_main = max(1, int(sm_total * sm_main_ratio))
        if sm_compress + sm_main > sm_total:
            sm_main = max(1, sm_total - sm_compress)
        if sm_compress + sm_main > sm_total:
            sm_compress = max(1, sm_total - sm_main)

        compress_stream, main_stream = sgl_kernel.create_greenctx_stream_by_value(
            sm_compress,
            sm_main,
            device_id,
        )
        return compress_stream, main_stream, GreenCtxInfo(available=True)
    except Exception as exc:  # pragma: no cover - runtime error path
        return None, None, GreenCtxInfo(available=False, reason=str(exc))


def build_streams(
    use_greenctx: bool,
    sm_compress_ratio: float,
    sm_main_ratio: float,
    use_greenctx_main: bool,
    compression_streams: int,
    compression_priority: int = -1,
    device_id: Optional[int] = None,
) -> Tuple[torch.cuda.Stream, StreamPool, GreenCtxInfo]:
    """Build main stream + compression stream pool (greenctx aware)."""
    if device_id is None:
        device_id = torch.cuda.current_device()

    main_stream = torch.cuda.current_stream()
    compress_streams: List[torch.cuda.Stream] = []
    info = GreenCtxInfo(available=False, reason="disabled")

    if use_greenctx:
        compress_stream, green_main, info = try_create_greenctx_streams(
            sm_compress_ratio=sm_compress_ratio,
            sm_main_ratio=sm_main_ratio,
            device_id=device_id,
        )
        if compress_stream is not None:
            compress_streams.append(compress_stream)
        if use_greenctx_main and green_main is not None:
            main_stream = green_main

    if not compress_streams:
        compress_streams = [
            torch.cuda.Stream(priority=compression_priority) for _ in range(compression_streams)
        ]
        info = GreenCtxInfo(available=False, reason="fallback")

    return main_stream, StreamPool(compress_streams), info

