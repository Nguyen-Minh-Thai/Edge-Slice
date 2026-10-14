"""
================================================================================
PYTORCH DEMO / BYPASS — NOT THE CORE EDGESLICE COMPILER PIPELINE
================================================================================

This module is a **quick numerical simulation** for environments where the full
MLIR toolchain is unavailable. It does **not** implement the research proposal's
main compilation path and must not be cited as evidence that EdgeSlice works
end-to-end.

**Core pipeline (proposal):**
  PyTorch model
    → DAGParser (torch-mlir)
    → LivenessAnalyzer + PeakMemoryEstimator
    → DimensionSelector + TileSizeCalculator
    → SlicedRewrite (MLIR Transform Dialect: extract_slice → scf.forall → parallel_insert_slice)
    → EdgeSliceMLIRPipeline (transform-interpreter + lowering)
    → JIT execution

**This bypass module instead:**
  - Wraps ``nn.Conv2d`` / ``nn.Linear`` at runtime with tile-and-concat loops
  - Uses heuristics mirrored from ``edgeslide.tiling.dimension_selector`` and
    ``edgeslide.tiling.tile_size`` for approximate parity with the frontend
  - Produces no MLIR artifacts and does not run the transform interpreter

Use only for rapid sanity checks. For publication-quality results, use
``EdgeSlicePassManager`` + ``EdgeSliceMLIRPipeline`` and the tests in
``tests/test_cpu_pipeline.py`` / ``tests/test_sliced_mlir_numerical_equivalence.py``.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from edgeslide.tiling.dimension_selector import dimension_index_to_name, select_dimension_index
from edgeslide.tiling.tile_size import TileBudgetError, TileSizeCalculator

# Stamped onto every benchmark record produced through this module so a result
# can never be read as compiler output. See EDGESLICE_TRANSFORM_ARCHITECTURE.md.
IMPLEMENTATION_KIND = "pytorch_reference"
IS_REFERENCE_IMPLEMENTATION = True


@dataclass
class SlicingConfig:
    """Configuration for the PyTorch demo slicer."""

    l1_cache_bytes: int = 512 * 1024
    tile_size: int = 64
    enabled: bool = True
    verbose: bool = False


class TensorSlicer:
    """Slice large tensors into tiles that fit the configured L1 budget."""

    def __init__(self, config: Optional[SlicingConfig] = None):
        self.config = config or SlicingConfig()
        self.slicing_plan: Dict[str, Any] = {}
        self._tile_calc = TileSizeCalculator(self.config.l1_cache_bytes)

    def should_slice(self, tensor: torch.Tensor) -> bool:
        tensor_bytes = tensor.numel() * tensor.element_size()
        return self.should_slice_bytes(tensor_bytes)

    def should_slice_bytes(self, tensor_bytes: int) -> bool:
        return tensor_bytes > self.config.l1_cache_bytes

    def should_slice_op(
        self,
        tensor: torch.Tensor,
        output_shape: Optional[tuple] = None,
    ) -> bool:
        """Gate on the op, not just its input.

        The MLIR frontend selects candidates on the *output* tensor
        (proposal §4.5, ``output_size(op) > L1_threshold``). Gating the demo on
        the input alone silently skips channel-expanding convs whose input fits
        L1 but whose output does not — exactly the ops the frontend targets, so
        any baseline-vs-EdgeSlide comparison would be measuring different plans.
        """
        if self.should_slice(tensor):
            return True
        if output_shape is None:
            return False
        elements = 1
        for dim in output_shape:
            elements *= int(dim)
        return self.should_slice_bytes(elements * tensor.element_size())

    def select_slicing_dimension(self, tensor: torch.Tensor, op_type: str = "conv2d") -> int:
        """Mirror the MLIR heuristic, with Linear input-feature semantics."""
        if op_type == "matmul" and tensor.ndim == 2 and tensor.shape[0] <= 1:
            return 1
        return select_dimension_index(tuple(tensor.shape), op_type)

    def select_slicing_dimension_for_shape(self, shape: tuple, op_type: str) -> int:
        """Heuristic on an explicit shape — the op's output, per proposal §4.4."""
        return select_dimension_index(tuple(shape), op_type)

    def calculate_tile_size(
        self, tensor: torch.Tensor, dim: int, op_type: str
    ) -> int:
        """Use ``TileSizeCalculator`` — matches the MLIR frontend tile heuristic."""
        dim_name = dimension_index_to_name(dim, op_type, tensor.ndim)
        try:
            return self._tile_calc.compute(
                tuple(tensor.shape),
                dim_name,
                dtype_size=tensor.element_size(),
                op_type=op_type,
            )
        except TileBudgetError:
            return int(tensor.shape[dim])

    def calculate_op_tile_size(
        self,
        output_shape: tuple,
        dim: int,
        op_type: str,
        *,
        dtype_size: int = 4,
        **cost_kwargs,
    ) -> int:
        """Tile size on the op's OUTPUT shape with working-set metadata."""
        tile_size, _ = self.plan_op_tile(
            output_shape, dim, op_type, dtype_size=dtype_size, **cost_kwargs
        )
        return tile_size

    def plan_op_tile(
        self,
        output_shape: tuple,
        dim: int,
        op_type: str,
        *,
        dtype_size: int = 4,
        **cost_kwargs,
    ) -> tuple:
        """``(tile_size, rejection_reason)`` on the op's OUTPUT shape.

        This mirrors ``EdgeSlicePassManager._tile_cost_kwargs``: the tile is
        expressed in output units and the budget accounts for the input halo and
        weight footprint, matching the MLIR frontend exactly.

        When even a unit tile overflows the budget the frontend rejects the plan
        (proposal §4.8.4) rather than claim cache residency. The demo does the
        same — it returns the full extent, which reads as "not sliced" — and
        hands back the reason so callers can tell a deliberate rejection apart
        from an op that simply fit L1.
        """
        dim_name = dimension_index_to_name(dim, op_type, len(output_shape))
        try:
            tile_size = self._tile_calc.compute(
                tuple(output_shape),
                dim_name,
                dtype_size=dtype_size,
                op_type=op_type,
                **cost_kwargs,
            )
        except TileBudgetError as exc:
            return int(output_shape[dim]), str(exc)
        return tile_size, None

    def slice_tensor(self, tensor: torch.Tensor, dim: int, tile_size: int) -> List[torch.Tensor]:
        tiles = []
        dim_size = tensor.shape[dim]
        for start_idx in range(0, dim_size, tile_size):
            end_idx = min(start_idx + tile_size, dim_size)
            slices = [slice(None)] * tensor.ndim
            slices[dim] = slice(start_idx, end_idx)
            tiles.append(tensor[tuple(slices)].clone())
        return tiles

    def concat_tiles(self, tiles: List[torch.Tensor], dim: int) -> torch.Tensor:
        if not tiles:
            raise ValueError("Cannot concatenate empty list of tiles")
        return torch.cat(tiles, dim=dim)


class SlicedConv2d(nn.Module):
    """Conv2d wrapper that tile-and-concats along the selected dimension."""

    def __init__(self, original_conv: nn.Conv2d, slicer: TensorSlicer):
        super().__init__()
        self.original_conv = original_conv
        self.slicer = slicer
        self.op_type = "conv_2d"

        self.slice_dim: Optional[int] = None
        self.tile_size: Optional[int] = None
        self.needs_slicing = False
        # Set when no positive tile fits the budget (proposal §4.8.4), so that
        # "not sliced" is distinguishable from "fits L1 already".
        self.plan_rejected_reason: Optional[str] = None

    @staticmethod
    def _is_nchw_spatial_dim(dim: int, rank: int) -> bool:
        return rank >= 4 and dim in (2, 3)

    @staticmethod
    def _pair(value) -> tuple[int, int]:
        if isinstance(value, tuple):
            return int(value[0]), int(value[1])
        if isinstance(value, list):
            return int(value[0]), int(value[1])
        return int(value), int(value)

    @staticmethod
    def _conv_out_dim(
        input_size: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ) -> int:
        return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def _supports_spatial_halo(self) -> bool:
        return not isinstance(self.original_conv.padding, str) and (
            self.original_conv.padding_mode == "zeros"
        )

    def _conv_output_shape(self, x: torch.Tensor) -> Optional[tuple]:
        """NCHW output shape for the wrapped conv, or None for string padding."""
        if isinstance(self.original_conv.padding, str):
            return None
        kernel_h, kernel_w = self._pair(self.original_conv.kernel_size)
        stride_h, stride_w = self._pair(self.original_conv.stride)
        pad_h, pad_w = self._pair(self.original_conv.padding)
        dilation_h, dilation_w = self._pair(self.original_conv.dilation)
        out_h = self._conv_out_dim(int(x.shape[2]), kernel_h, stride_h, pad_h, dilation_h)
        out_w = self._conv_out_dim(int(x.shape[3]), kernel_w, stride_w, pad_w, dilation_w)
        if out_h <= 0 or out_w <= 0:
            return None
        return (int(x.shape[0]), int(self.original_conv.out_channels), out_h, out_w)

    def _forward_spatial_halo(self, x: torch.Tensor, slice_dim: int) -> torch.Tensor:
        """Exact H/W tiling with convolution halo for zero-padded NCHW Conv2d."""
        if not self._supports_spatial_halo():
            return self.original_conv(x)

        kernel_h, kernel_w = self._pair(self.original_conv.kernel_size)
        stride_h, stride_w = self._pair(self.original_conv.stride)
        pad_h, pad_w = self._pair(self.original_conv.padding)
        dilation_h, dilation_w = self._pair(self.original_conv.dilation)
        input_h, input_w = int(x.shape[2]), int(x.shape[3])
        out_h = self._conv_out_dim(input_h, kernel_h, stride_h, pad_h, dilation_h)
        out_w = self._conv_out_dim(input_w, kernel_w, stride_w, pad_w, dilation_w)
        if out_h <= 0 or out_w <= 0:
            return self.original_conv(x)

        tile = max(1, int(self.tile_size or 1))
        output_tiles = []
        if slice_dim == 2:
            for out_start in range(0, out_h, tile):
                out_end = min(out_start + tile, out_h)
                first_needed = out_start * stride_h - pad_h
                last_needed = (
                    (out_end - 1) * stride_h
                    - pad_h
                    + dilation_h * (kernel_h - 1)
                    + 1
                )
                input_start = max(0, first_needed)
                input_end = min(input_h, last_needed)
                top_padding = max(0, -first_needed)
                bottom_padding = max(0, last_needed - input_h)
                x_tile = x[:, :, input_start:input_end, :]
                if top_padding or bottom_padding or pad_w:
                    x_tile = F.pad(x_tile, (pad_w, pad_w, top_padding, bottom_padding))
                y_tile = F.conv2d(
                    x_tile,
                    self.original_conv.weight,
                    self.original_conv.bias,
                    stride=self.original_conv.stride,
                    padding=0,
                    dilation=self.original_conv.dilation,
                    groups=self.original_conv.groups,
                )
                output_tiles.append(y_tile[:, :, : out_end - out_start, :out_w])
            return torch.cat(output_tiles, dim=2)

        if slice_dim == 3:
            for out_start in range(0, out_w, tile):
                out_end = min(out_start + tile, out_w)
                first_needed = out_start * stride_w - pad_w
                last_needed = (
                    (out_end - 1) * stride_w
                    - pad_w
                    + dilation_w * (kernel_w - 1)
                    + 1
                )
                input_start = max(0, first_needed)
                input_end = min(input_w, last_needed)
                left_padding = max(0, -first_needed)
                right_padding = max(0, last_needed - input_w)
                x_tile = x[:, :, :, input_start:input_end]
                if left_padding or right_padding or pad_h:
                    x_tile = F.pad(x_tile, (left_padding, right_padding, pad_h, pad_h))
                y_tile = F.conv2d(
                    x_tile,
                    self.original_conv.weight,
                    self.original_conv.bias,
                    stride=self.original_conv.stride,
                    padding=0,
                    dilation=self.original_conv.dilation,
                    groups=self.original_conv.groups,
                )
                output_tiles.append(y_tile[:, :, :out_h, : out_end - out_start])
            return torch.cat(output_tiles, dim=3)

        return self.original_conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The slicing plan assumes batched NCHW input. For unbatched (C, H, W)
        # input the selector's indices shift onto the contracted channel dim,
        # so fall back to the unsliced convolution.
        if not self.slicer.config.enabled or x.ndim != 4:
            return self.original_conv(x)

        output_shape = self._conv_output_shape(x)
        if not self.slicer.should_slice_op(x, output_shape):
            return self.original_conv(x)

        if self.slice_dim is None:
            if output_shape is not None:
                # Mirror the MLIR frontend end to end: the dimension heuristic
                # runs on the OUTPUT shape (proposal §4.4) and the tile is
                # expressed in output units — the same units
                # ``_forward_spatial_halo`` iterates — with the conv's halo and
                # weight footprint charged to the cache budget.
                self.slice_dim = self.slicer.select_slicing_dimension_for_shape(
                    output_shape, self.op_type
                )
                self.tile_size, self.plan_rejected_reason = self.slicer.plan_op_tile(
                    output_shape,
                    self.slice_dim,
                    self.op_type,
                    dtype_size=x.element_size(),
                    input_shape=tuple(x.shape),
                    weight_shape=tuple(self.original_conv.weight.shape),
                    kernel_shape=self._pair(self.original_conv.kernel_size),
                    stride=self._pair(self.original_conv.stride),
                    dilation=self._pair(self.original_conv.dilation),
                )
                self.needs_slicing = self.tile_size < output_shape[self.slice_dim]
            else:
                self.slice_dim = self.slicer.select_slicing_dimension(x, self.op_type)
                self.tile_size = self.slicer.calculate_tile_size(
                    x, self.slice_dim, self.op_type
                )
                self.needs_slicing = self.tile_size < x.shape[self.slice_dim]

            if self.slicer.config.verbose:
                print(
                    f"SlicedConv2d: dim={self.slice_dim} "
                    f"tile_size={self.tile_size} "
                    f"sliced={self.needs_slicing}"
                    + (
                        f" rejected={self.plan_rejected_reason}"
                        if self.plan_rejected_reason
                        else ""
                    )
                )

        if not self.needs_slicing:
            return self.original_conv(x)

        if self._is_nchw_spatial_dim(self.slice_dim, x.ndim):
            return self._forward_spatial_halo(x, self.slice_dim)

        assert self.tile_size is not None  # set when needs_slicing was decided
        input_tiles = self.slicer.slice_tensor(x, self.slice_dim, self.tile_size)
        output_tiles = [self.original_conv(tile) for tile in input_tiles]
        return self.slicer.concat_tiles(output_tiles, self.slice_dim)


class SlicedLinear(nn.Module):
    """Linear wrapper with dimension-aware, numerically correct tiling."""

    def __init__(self, original_linear: nn.Linear, slicer: TensorSlicer):
        super().__init__()
        self.original_linear = original_linear
        self.slicer = slicer
        self.op_type = "matmul"

        self.slice_dim: Optional[int] = None
        self.tile_size: Optional[int] = None
        self.needs_slicing = False

    def _forward_input_features(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce partial products when slicing Linear's contracted dimension."""
        if x.shape[-1] != self.original_linear.in_features:
            return self.original_linear(x)

        tile_size = max(1, int(self.tile_size or x.shape[-1]))
        output: Optional[torch.Tensor] = None

        for start in range(0, x.shape[-1], tile_size):
            end = min(start + tile_size, x.shape[-1])
            partial = F.linear(
                x[..., start:end],
                self.original_linear.weight[:, start:end],
                bias=None,
            )
            output = partial if output is None else output + partial

        if output is None:
            return self.original_linear(x)
        if self.original_linear.bias is not None:
            output = output + self.original_linear.bias
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_shape = tuple(x.shape[:-1]) + (self.original_linear.out_features,)
        if not self.slicer.config.enabled or not self.slicer.should_slice_op(
            x, output_shape
        ):
            return self.original_linear(x)

        if self.slice_dim is None:
            self.slice_dim = self.slicer.select_slicing_dimension(x, self.op_type)
            self.tile_size = self.slicer.calculate_tile_size(x, self.slice_dim, self.op_type)
            self.needs_slicing = self.tile_size < x.shape[self.slice_dim]

        if not self.needs_slicing:
            return self.original_linear(x)

        if self.slice_dim == x.ndim - 1:
            return self._forward_input_features(x)

        assert self.tile_size is not None  # set when needs_slicing was decided
        input_tiles = self.slicer.slice_tensor(x, self.slice_dim, self.tile_size)
        output_tiles = [self.original_linear(tile) for tile in input_tiles]
        return self.slicer.concat_tiles(output_tiles, self.slice_dim)


class EdgeSlideTransform:
    """
    Apply the PyTorch demo slicer to eligible layers.

    This is a convenience wrapper for experiments — not the MLIR compiler entry point.
    """

    def __init__(self, config: Optional[SlicingConfig] = None):
        self.config = config or SlicingConfig()
        self.slicer = TensorSlicer(self.config)
        self.transformed_layers: Dict[str, nn.Module] = {}

    def transform_model(self, model: nn.Module) -> nn.Module:
        model = model.eval()

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels >= 64:
                sliced_conv = SlicedConv2d(module, self.slicer)
                self._replace_module(model, name, sliced_conv)
                self.transformed_layers[name] = sliced_conv
                if self.config.verbose:
                    print(f"EdgeSlide demo: wrapped Conv2d at {name}")

            elif isinstance(module, nn.Linear) and module.out_features >= 512:
                sliced_linear = SlicedLinear(module, self.slicer)
                self._replace_module(model, name, sliced_linear)
                self.transformed_layers[name] = sliced_linear
                if self.config.verbose:
                    print(f"EdgeSlide demo: wrapped Linear at {name}")

        return model

    def _replace_module(self, model: nn.Module, target_name: str, new_module: nn.Module):
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


class MemoryProfiler:
    """Profile per-layer output memory for demo diagnostics."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List[Any] = []
        self.memory_usage: Dict[str, float] = {}

    def profile(self, input_data: torch.Tensor) -> Dict[str, float]:
        self.memory_usage = {}

        def hook_fn(name):
            def hook(module, input, output):
                if output is not None:
                    self.memory_usage[name] = (
                        output.numel() * output.element_size() / (1024 ** 2)
                    )
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.hooks.append(module.register_forward_hook(hook_fn(name)))

        with torch.no_grad():
            _ = self.model(input_data)

        for hook in self.hooks:
            hook.remove()
        return self.memory_usage

    def get_memory_summary(self) -> str:
        lines = [
            "=== Memory Profile Summary ===",
            "Layer                                          Output (MB)",
        ]
        for name, size_mb in sorted(self.memory_usage.items(), key=lambda x: -x[1]):
            lines.append(f"{name:50} {size_mb:10.2f}")
        return "\n".join(lines)
