"""PyTorch SConv-style spatial convolution baseline for real comparisons.

This module provides a concrete, executable spatial-halo convolution transform
for benchmarking against EdgeSlice on the same real images.  It is intentionally
implemented as normal PyTorch modules so results come from actual forward passes,
not analytical estimates.

The transform preserves numerical equivalence for zero-padded ``nn.Conv2d`` by
computing output-height tiles with the required input halo.  Unsupported padding
modes fall back to the original convolution rather than returning incorrect data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Proposal 5.3 Baseline 3: a local PyTorch spatial-halo stand-in for SConv, not
# the published SConvTransform. Comparisons against it are reference-vs-reference.
IMPLEMENTATION_KIND = "pytorch_reference"
IS_REFERENCE_IMPLEMENTATION = True


@dataclass
class SConvConfig:
    """Configuration for the PyTorch SConv-style spatial-halo baseline."""

    l1_cache_bytes: int = 512 * 1024
    tile_output_rows: Optional[int] = None
    min_out_channels: int = 64
    enabled: bool = True
    verbose: bool = False


def _pair(value) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    if isinstance(value, list):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def compute_spatial_halo_tile_rows(
    *,
    batch: int,
    in_channels: int,
    in_w: int,
    out_channels: int,
    out_w: int,
    out_h: int,
    kernel_h: int,
    dilation_h: int,
    dtype_size: int,
    l1_cache_bytes: int,
) -> int:
    """Row-budget formula for spatial-halo output-row tiling.

    Pulled out of :meth:`SpatialHaloSConv2d._compute_tile_rows` so the
    analytical DAG-based Baseline 3 planner in
    ``benchmarks/run_evaluation.py`` (proposal §5.3) computes the same tile
    size the real executable module would choose, from one definition
    instead of two that could silently drift apart.
    """
    halo_rows = dilation_h * (kernel_h - 1) + 1
    bytes_per_output_row = batch * out_channels * out_w * dtype_size
    bytes_per_input_row = batch * in_channels * in_w * dtype_size
    bytes_per_row = bytes_per_output_row + bytes_per_input_row
    if bytes_per_row <= 0:
        return out_h

    rows = int((l1_cache_bytes - halo_rows * bytes_per_input_row) / bytes_per_row)
    rows = max(1, rows)
    return max(1, min(rows, out_h))


class SpatialHaloSConv2d(nn.Module):
    """Conv2d wrapper that tiles output rows and includes input halo rows."""

    def __init__(self, original_conv: nn.Conv2d, config: SConvConfig):
        super().__init__()
        self.original_conv = original_conv
        self.config = config
        self.tile_output_rows: Optional[int] = None
        self.needs_tiling = False
        self.fallback_reason: Optional[str] = None

    @staticmethod
    def _conv_out_dim(
        input_size: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ) -> int:
        return math.floor(
            (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
            + 1
        )

    def _supports_exact_tiling(self) -> bool:
        if isinstance(self.original_conv.padding, str):
            self.fallback_reason = "string padding is not supported by local SConv"
            return False
        if self.original_conv.padding_mode != "zeros":
            self.fallback_reason = (
                f"padding_mode={self.original_conv.padding_mode!r} is not supported"
            )
            return False
        return True

    def _compute_tile_rows(self, x: torch.Tensor, out_h: int, out_w: int) -> int:
        if self.config.tile_output_rows is not None:
            return max(1, min(int(self.config.tile_output_rows), out_h))

        kernel_h, _ = _pair(self.original_conv.kernel_size)
        dilation_h, _ = _pair(self.original_conv.dilation)
        return compute_spatial_halo_tile_rows(
            batch=int(x.shape[0]),
            in_channels=int(x.shape[1]),
            in_w=int(x.shape[3]),
            out_channels=int(self.original_conv.out_channels),
            out_w=out_w,
            out_h=out_h,
            kernel_h=kernel_h,
            dilation_h=dilation_h,
            dtype_size=x.element_size(),
            l1_cache_bytes=self.config.l1_cache_bytes,
        )

    def _tile_conv_height(self, x: torch.Tensor, out_h: int) -> torch.Tensor:
        kernel_h, kernel_w = _pair(self.original_conv.kernel_size)
        stride_h, stride_w = _pair(self.original_conv.stride)
        pad_h, pad_w = _pair(self.original_conv.padding)
        dilation_h, dilation_w = _pair(self.original_conv.dilation)
        input_h = int(x.shape[2])

        output_tiles: List[torch.Tensor] = []
        assert self.tile_output_rows is not None
        for out_start in range(0, out_h, self.tile_output_rows):
            out_end = min(out_start + self.tile_output_rows, out_h)
            first_needed = out_start * stride_h - pad_h
            last_needed_exclusive = (
                (out_end - 1) * stride_h - pad_h + dilation_h * (kernel_h - 1) + 1
            )
            input_start = max(0, first_needed)
            input_end = min(input_h, last_needed_exclusive)
            top_padding = max(0, -first_needed)
            bottom_padding = max(0, last_needed_exclusive - input_h)

            x_tile = x[:, :, input_start:input_end, :]
            if top_padding or bottom_padding or pad_w:
                x_tile = F.pad(x_tile, (pad_w, pad_w, top_padding, bottom_padding))
            y_tile = F.conv2d(
                x_tile,
                self.original_conv.weight,
                self.original_conv.bias,
                stride=self.original_conv.stride,
                padding=0,
                dilation=(dilation_h, dilation_w),
                groups=self.original_conv.groups,
            )

            expected_rows = out_end - out_start
            if int(y_tile.shape[2]) != expected_rows:
                y_tile = y_tile[:, :, :expected_rows, :]
            if bottom_padding and int(y_tile.shape[2]) < expected_rows:
                raise RuntimeError(
                    "SConv tile produced too few output rows; check Conv2d geometry"
                )
            output_tiles.append(y_tile)

        return torch.cat(output_tiles, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.enabled:
            return self.original_conv(x)
        if not self._supports_exact_tiling():
            if self.config.verbose:
                print(f"SConv fallback: {self.fallback_reason}")
            return self.original_conv(x)

        kernel_h, kernel_w = _pair(self.original_conv.kernel_size)
        stride_h, stride_w = _pair(self.original_conv.stride)
        pad_h, pad_w = _pair(self.original_conv.padding)
        dilation_h, dilation_w = _pair(self.original_conv.dilation)
        out_h = self._conv_out_dim(int(x.shape[2]), kernel_h, stride_h, pad_h, dilation_h)
        out_w = self._conv_out_dim(int(x.shape[3]), kernel_w, stride_w, pad_w, dilation_w)
        if out_h <= 0 or out_w <= 0:
            return self.original_conv(x)

        if self.tile_output_rows is None:
            self.tile_output_rows = self._compute_tile_rows(x, out_h, out_w)
            self.needs_tiling = self.tile_output_rows < out_h
            if self.config.verbose:
                print(
                    f"SpatialHaloSConv2d: out_h={out_h} out_w={out_w} "
                    f"tile_output_rows={self.tile_output_rows}"
                )

        if not self.needs_tiling:
            return self.original_conv(x)
        return self._tile_conv_height(x, out_h)


class SConvTransform:
    """Replace eligible Conv2d layers with exact spatial-halo SConv wrappers."""

    def __init__(self, config: Optional[SConvConfig] = None):
        self.config = config or SConvConfig()
        self.transformed_layers: Dict[str, SpatialHaloSConv2d] = {}

    def transform_model(self, model: nn.Module) -> nn.Module:
        model = model.eval()
        modules = list(model.named_modules())
        for name, module in modules:
            if not name:
                continue
            if isinstance(module, nn.Conv2d) and module.out_channels >= self.config.min_out_channels:
                wrapped = SpatialHaloSConv2d(module, self.config)
                self._replace_module(model, name, wrapped)
                self.transformed_layers[name] = wrapped
                if self.config.verbose:
                    print(f"SConv: wrapped Conv2d at {name}")
        return model

    @staticmethod
    def _replace_module(model: nn.Module, target_name: str, new_module: nn.Module) -> None:
        parts = target_name.split(".")
        # Any: nn.Sequential/ModuleList indexing is opaque to nn.Module typing.
        parent: Any = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        last_part = parts[-1]
        if last_part.isdigit():
            parent[int(last_part)] = new_module
        else:
            setattr(parent, last_part, new_module)
