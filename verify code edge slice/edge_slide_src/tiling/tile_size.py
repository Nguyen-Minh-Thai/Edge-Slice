"""Tile Size Calculator — compute hardware-aware tile sizes for sliced ops."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from edgeslide.tiling.dimension_selector import dimension_name_to_index, is_depthwise_op


class TileBudgetError(ValueError):
    """Raised when even a unit tile exceeds the configured cache budget."""


def _pair_tuple(value: Any) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            return int(value[0]), int(value[1])
        if len(value) == 1:
            return int(value[0]), int(value[0])
    return int(value), int(value)


def node_cost_kwargs(node: Any) -> Dict[str, Any]:
    """Describe a DAG node to the tile cost model.

    Shared by the planner (which sizes each anchor's tile) and the fusion
    planner (which sizes the whole fused loop body) so both measure an op the
    same way; ``rewrite`` cannot import ``pass_manager`` for it.
    """
    lowered = node.op_type.lower()
    kwargs: Dict[str, Any] = {
        "dtype_size": node.output_tensor.element_size_bytes,
        "op_type": node.op_type,
    }
    if "conv" not in lowered:
        # Non-conv ops — matmul and every linalg.generic, i.e. exactly what
        # fusion chains are built from — used to be described to the cost model
        # by dtype and op type alone, which left it costing "input + output,
        # both shaped like the output". That silently dropped every other
        # operand the op reads: a matmul's B matrix, a broadcast bias, the
        # second addend of an elementwise add. Those tiles are live at the same
        # program point, so budget admission built on the old model
        # under-counted the working set (proposal §4.8.4 Theorem 3 is a
        # guarantee about the *whole* tile working set, not the output alone).
        # Hand over every operand and let estimate_tile_footprint decide which
        # ones the tiled dimension actually shrinks.
        kwargs["operand_shapes"] = tuple(
            tuple(tensor.shape) for tensor in node.input_tensors
        )
        return kwargs

    if node.input_tensors:
        kwargs["input_shape"] = node.input_tensors[0].shape
    if len(node.input_tensors) >= 2:
        kwargs["weight_shape"] = node.input_tensors[1].shape

    strides = node.params.get("strides") or node.params.get("stride")
    dilations = node.params.get("dilations") or node.params.get("dilation")
    if strides:
        kwargs["stride"] = _pair_tuple(strides)
    if dilations:
        kwargs["dilation"] = _pair_tuple(dilations)
    # >= 3, not >= 4: a dense conv's weight is [F, C, Kh, Kw] (rank 4), but a
    # depthwise conv's is [C, Kh, Kw] (rank 3, no separate F) — requiring
    # rank >= 4 silently dropped kernel_shape for every depthwise conv, which
    # made _effective_kernel default to a 1x1 kernel and underestimate the
    # halo a spatial tile actually needs.
    if len(node.input_tensors) >= 2 and len(node.input_tensors[1].shape) >= 3:
        weight_shape = node.input_tensors[1].shape
        if "nhwc" in lowered and len(weight_shape) >= 3:
            kwargs["kernel_shape"] = (int(weight_shape[1]), int(weight_shape[2]))
        else:
            kwargs["kernel_shape"] = (int(weight_shape[-2]), int(weight_shape[-1]))
    return kwargs

@dataclass(frozen=True)
class HardwareConfig:
    """Target cache/vector properties used by the tile cost model.

    The default ``l1_cache_bytes`` of 512 KB is a *budget* knob inherited from
    the original calculator, not an L1 size any shipping edge CPU has. No edge
    part has an L1D above ~64 KB: the Raspberry Pi 4's Cortex-A72 has 32 KB
    (which ``benchmarks/hardware_profiler.py`` already records in its
    cachegrind model), a Cortex-A55 has 32-64 KB, an A76 64 KB. Numbers in the
    256 KB - 1 MB range are L2 sizes. See ``EDGE_CACHE_LEVELS`` for the real
    two-level configurations and ``for_device`` to build one.

    When ``l2_cache_bytes`` is set, the calculator runs a two-level search:
    prefer a tile that fits the L1 budget, and if none exists accept one that
    fits the L2 budget, recording which level bound the decision. That is the
    real strategy on a two-level machine — tile for the outer level and stream
    through the inner one — and it is what lets an op whose unit tile already
    overflows L1 still be tiled instead of rejected.
    """

    l1_cache_bytes: int = 512 * 1024
    l2_cache_bytes: Optional[int] = None
    vector_width_bytes: int = 16
    cache_line_bytes: int = 64
    l1_budget_fraction: float = 0.80
    l2_budget_fraction: float = 0.90
    # Whether an L1-infeasible tile may be accepted as L2-resident. Off by
    # default so a config that merely *records* an L2 size does not silently
    # loosen the admission rule every existing caller relies on.
    allow_l2_resident: bool = False
    # Name used in plans and diagnostics for the innermost explicitly managed
    # memory. CPU targets keep the historical "L1" label; accelerator targets
    # created with ``for_sram`` use "SRAM" so cache residency is never confused
    # with actual scratchpad placement.
    primary_memory_level: str = "L1"

    @property
    def l1_budget_bytes(self) -> int:
        return max(1, int(self.l1_cache_bytes * self.l1_budget_fraction))

    @property
    def l2_budget_bytes(self) -> Optional[int]:
        if self.l2_cache_bytes is None:
            return None
        return max(1, int(self.l2_cache_bytes * self.l2_budget_fraction))

    def budget_levels(self) -> List[Tuple[str, int]]:
        """Admissible budgets, innermost first, as (level name, bytes).

        A single-level config yields just ``[("L1", ...)]``. With
        ``allow_l2_resident`` the L2 budget follows — and only if it is
        genuinely larger, because an "L2" smaller than L1 would otherwise widen
        nothing while claiming to.
        """
        levels = [(self.primary_memory_level, self.l1_budget_bytes)]
        l2 = self.l2_budget_bytes
        if self.allow_l2_resident and l2 is not None and l2 > self.l1_budget_bytes:
            levels.append(("L2", l2))
        return levels

    @classmethod
    def for_sram(
        cls,
        capacity_bytes: int,
        *,
        double_buffer: bool = True,
        vector_width_bytes: int = 16,
        alignment_bytes: int = 64,
    ) -> "HardwareConfig":
        """Build a single-level, explicitly managed SRAM target.

        With double buffering, one tile may consume at most half of SRAM so
        the next tile can be transferred while the current tile computes.
        This is deliberately conservative (it reserves a second copy of the
        entire working set) but, unlike calling an L2-sized value "L1", it is
        an enforceable capacity contract for a scratchpad backend.
        """
        if capacity_bytes <= 0:
            raise ValueError("SRAM capacity must be positive")
        return cls(
            l1_cache_bytes=capacity_bytes,
            l2_cache_bytes=None,
            vector_width_bytes=vector_width_bytes,
            cache_line_bytes=alignment_bytes,
            l1_budget_fraction=0.50 if double_buffer else 0.90,
            allow_l2_resident=False,
            primary_memory_level="SRAM",
        )

    @classmethod
    def for_device(cls, device: str, **overrides: Any) -> "HardwareConfig":
        """A real two-level configuration for a named edge target."""
        if device not in EDGE_CACHE_LEVELS:
            raise ValueError(
                f"unknown device {device!r}; known: {sorted(EDGE_CACHE_LEVELS)}"
            )
        spec = dict(EDGE_CACHE_LEVELS[device])
        spec.pop("description", None)
        spec.update(overrides)
        return cls(**spec)


# Measured cache geometries for the evaluation targets (proposal §5.1). L1D and
# L2 are the real per-core figures from the vendor TRMs; the "generic_256k"
# entry is the legacy single-level budget the evaluation used before this table
# existed, kept so old results stay reproducible and comparable.
EDGE_CACHE_LEVELS: Dict[str, Dict[str, Any]] = {
    "pi4": {
        "l1_cache_bytes": 32 * 1024,
        "l2_cache_bytes": 1024 * 1024,
        "allow_l2_resident": True,
        "description": "Raspberry Pi 4, Cortex-A72: 32 KB L1D, 1 MB shared L2",
    },
    "cortex_a55": {
        "l1_cache_bytes": 32 * 1024,
        "l2_cache_bytes": 256 * 1024,
        "allow_l2_resident": True,
        "description": "Cortex-A55: 32 KB L1D, 256 KB L2",
    },
    "cortex_a76": {
        "l1_cache_bytes": 64 * 1024,
        "l2_cache_bytes": 512 * 1024,
        "allow_l2_resident": True,
        "description": "Cortex-A76: 64 KB L1D, 512 KB L2",
    },
    "generic_256k": {
        "l1_cache_bytes": 256 * 1024,
        "l2_cache_bytes": None,
        "allow_l2_resident": False,
        "description": "Legacy single-level 256 KB budget (not a real L1D)",
    },
}

@dataclass(frozen=True)
class TileFootprint:
    """Estimated working set for one candidate tile."""

    input_bytes: int
    output_bytes: int
    weight_bytes: int
    halo_bytes: int
    total_bytes: int
    # Operands past the first: a matmul's B matrix, a broadcast bias, the
    # second addend of an elementwise add. Appended rather than inserted so
    # existing positional construction keeps working, and defaulted to 0 so the
    # convolution path — which routes its second operand through
    # ``weight_bytes`` — still satisfies input + output + weight == total.
    operand_bytes: int = 0

@dataclass(frozen=True)
class TileDecision:
    """An accepted tile together with the budget evidence behind it.

    ``compute`` returns a bare int for backward compatibility, which leaves the
    caller unable to say how much of the budget the tile it was handed actually
    claims. ``utilization`` (footprint / binding budget, in (0, 1]) closes that
    gap: it is what makes a bad round-down visible. The power-of-two round-down
    this calculator used to apply routinely shipped tiles at ~0.5 utilization —
    half the budget the search had just proven affordable, discarded silently.
    """

    tile_size: int
    footprint_bytes: int
    budget_bytes: int
    utilization: float
    # Which cache level's budget the accepted tile fits: "L1", or "L2" when the
    # op had no L1-resident tile and the two-level search fell through. The
    # distribution of this across a model is a result in itself — it says how
    # much of the network can be made L1-resident at all.
    budget_level: str = "L1"

# Which axis of ``(stride, dilation, kernel_shape)`` a spatial dimension name
# indexes. Shared by the conv and non-conv halo rules so they cannot drift.
_SPATIAL_HALO_AXIS = {"spatial_h": 0, "spatial_w": 1}

class TileSizeCalculator:
    """Compute tile sizes that fit the configured cache budget.

    The default API is intentionally backward compatible with the original
    calculator.  Callers can pass convolution metadata to account for halo
    duplication and weight footprint, and can override cache/vector properties
    via ``HardwareConfig``.
    """

    def __init__(
        self,
        l1_cache_bytes: int = 512 * 1024,
        max_tile_size: Optional[int] = None,
        *,
        hardware_config: Optional[HardwareConfig] = None,
        l2_cache_bytes: Optional[int] = None,
        vector_width_bytes: int = 16,
        cache_line_bytes: int = 64,
    ):
        if hardware_config is None:
            hardware_config = HardwareConfig(
                l1_cache_bytes=l1_cache_bytes,
                l2_cache_bytes=l2_cache_bytes,
                vector_width_bytes=vector_width_bytes,
                cache_line_bytes=cache_line_bytes,
            )
        elif l1_cache_bytes != 512 * 1024 and hardware_config.l1_cache_bytes != l1_cache_bytes:
            hardware_config = replace(hardware_config, l1_cache_bytes=l1_cache_bytes)

        self.hardware_config = hardware_config
        self.l1_cache_bytes = hardware_config.l1_cache_bytes
        self.max_tile_size = max_tile_size

    def compute(
        self,
        shape: Tuple[int, ...],
        slice_dim: str,
        *,
        input_shape: Optional[Tuple[int, ...]] = None,
        weight_shape: Optional[Tuple[int, ...]] = None,
        kernel_shape: Optional[Tuple[int, int]] = None,
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        dtype_size: int = 4,
        op_type: str = "",
        operand_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> int:
        """Compute a cache-aware tile size for ``shape`` along ``slice_dim``.

        Thin wrapper over :meth:`compute_detailed`; use that one when the
        accepted tile's footprint and budget utilization matter too.
        """
        return self.compute_detailed(
            shape,
            slice_dim,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            dilation=dilation,
            dtype_size=dtype_size,
            op_type=op_type,
            operand_shapes=operand_shapes,
        ).tile_size

    def compute_detailed(
        self,
        shape: Tuple[int, ...],
        slice_dim: str,
        *,
        input_shape: Optional[Tuple[int, ...]] = None,
        weight_shape: Optional[Tuple[int, ...]] = None,
        kernel_shape: Optional[Tuple[int, int]] = None,
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        dtype_size: int = 4,
        op_type: str = "",
        operand_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> TileDecision:
        """:meth:`compute`, plus the budget evidence for the tile it accepted.

        Every return path is costed and budget-checked, which is the whole of
        proposal §4.8.4 Theorem 3: the tile handed back fits the configured
        budget, or ``TileBudgetError`` is raised and the caller picks another
        dimension. There is deliberately no third outcome.
        """
        cost: Dict[str, Any] = {
            "input_shape": input_shape,
            "weight_shape": weight_shape,
            "kernel_shape": kernel_shape,
            "stride": stride,
            "dilation": dilation,
            "dtype_size": dtype_size,
            "op_type": op_type,
            "operand_shapes": operand_shapes,
        }
        dim_idx = self._dim_name_to_index(slice_dim, shape, op_type)
        if dim_idx >= len(shape):
            # Unreachable for every layout ``dimension_indices`` knows (all the
            # indices it hands out are < rank, and unknown names raise), but
            # kept as a defensive fallback — and costed like any other
            # candidate, so it cannot become a second way to leak an
            # over-budget tile past Theorem 3.
            return self._accept(
                shape,
                slice_dim,
                shape[0] if shape else 1,
                cost,
                label="whole-operation tile",
                note=(
                    f"; {slice_dim!r} is not a dimension of rank-{len(shape)} "
                    f"shape {tuple(shape)}, so there is nothing to slice"
                ),
            )

        full_size = shape[dim_idx]
        if full_size <= 1:
            # A size-<=1 (or degenerate size-0) dimension leaves exactly one
            # candidate tile, so there is nothing to search — but "only one
            # candidate" is not "always affordable", and this path used to
            # return 1 with no budget check at all. That broke Theorem 3
            # outright: compute((1, 64, 224, 224), "batch") on a 256 KB L1
            # returned tile 1 whose footprint is 25.7 MB against a 205 KB
            # budget — 123x over — and reported success, so the planner emitted
            # a tiling transform for an op that can never be cache-resident on
            # that axis. Cost the unit tile exactly as the non-degenerate paths
            # do and refuse it when it does not fit; the dimension cascade in
            # ``EdgeSlicePassManager._build_rewrite_plan`` then moves on to the
            # next dimension instead of trusting a tile that cannot deliver.
            return self._accept(
                shape,
                slice_dim,
                1,
                cost,
                label="unit tile",
                note=(
                    f"; dimension {slice_dim!r} has extent {full_size} (<= 1), "
                    "so no smaller tile exists"
                ),
            )

        if input_shape is not None or weight_shape is not None or kernel_shape is not None:
            best = self._compute_with_working_set(shape, slice_dim, full_size, cost)
        else:
            best = self._compute_operand_aware_tile(shape, slice_dim, full_size, cost)

        return self._accept(shape, slice_dim, best, cost, label=f"tile {best}")

    def compute_pair(
        self,
        shape: Tuple[int, ...],
        primary_dim: str,
        secondary_dim: str,
        *,
        input_shape: Optional[Tuple[int, ...]] = None,
        weight_shape: Optional[Tuple[int, ...]] = None,
        kernel_shape: Optional[Tuple[int, int]] = None,
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        dtype_size: int = 4,
        op_type: str = "",
        operand_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> Tuple[int, int]:
        """Cache-fitting 2D tile: (primary tile, secondary tile).

        For ops where no single dimension has a unit tile that fits the budget
        (e.g. a 256x256x256 feature map against a 256 KB L1: one row and one
        channel are each 256 KB), slicing two dimensions at once can still
        produce cache-resident tiles. The search maximizes the primary tile
        with the secondary at unit size, then grows the secondary with the
        primary fixed — both via the same monotone footprint model
        ``compute`` uses. Raises ``TileBudgetError`` when even the unit x unit
        tile exceeds the budget.
        """
        p_idx = self._dim_name_to_index(primary_dim, shape, op_type)
        s_idx = self._dim_name_to_index(secondary_dim, shape, op_type)
        if p_idx == s_idx:
            raise ValueError(
                f"primary and secondary slice dimensions must differ "
                f"({primary_dim!r} and {secondary_dim!r} both map to index {p_idx})"
            )
        p_full = shape[p_idx] if p_idx < len(shape) else 1
        s_full = shape[s_idx] if s_idx < len(shape) else 1

        def footprint(p_tile: int, s_tile: int) -> int:
            return self.estimate_tile_footprint(
                shape,
                primary_dim,
                p_tile,
                input_shape=input_shape,
                weight_shape=weight_shape,
                kernel_shape=kernel_shape,
                stride=stride,
                dilation=dilation,
                dtype_size=dtype_size,
                op_type=op_type,
                operand_shapes=operand_shapes,
                secondary=(secondary_dim, s_tile),
            ).total_bytes

        unit_bytes = footprint(1, 1)
        if not self._fits_cache_budget(unit_bytes):
            raise TileBudgetError(
                f"unit {primary_dim}x{secondary_dim} tile requires {unit_bytes} "
                "bytes, exceeding cache budget"
            )

        def search(upper: int, fits: Callable[[int], int]) -> int:
            low, high, best = 1, max(1, upper), 1
            while low <= high:
                candidate = (low + high) // 2
                if self._fits_cache_budget(fits(candidate)):
                    best = candidate
                    low = candidate + 1
                else:
                    high = candidate - 1
            return best

        p_upper = min(p_full, self.max_tile_size or p_full)
        best_p = search(p_upper, lambda t: footprint(t, 1))
        # _round_down_to_granule rounds down, so the budget check still holds.
        best_p = self._round_down_to_granule(best_p, dtype_size)
        best_p = self._align_tile_to_vector(best_p, primary_dim, dtype_size)
        best_p = max(1, min(best_p, p_upper))

        s_upper = min(s_full, self.max_tile_size or s_full)
        best_s = search(s_upper, lambda t: footprint(best_p, t))
        best_s = self._round_down_to_granule(best_s, dtype_size)
        best_s = self._align_tile_to_vector(best_s, secondary_dim, dtype_size)
        best_s = max(1, min(best_s, s_upper))

        return best_p, best_s

    def estimate_tile_footprint(
        self,
        shape: Tuple[int, ...],
        slice_dim: str,
        tile_size: int,
        *,
        input_shape: Optional[Tuple[int, ...]] = None,
        weight_shape: Optional[Tuple[int, ...]] = None,
        kernel_shape: Optional[Tuple[int, int]] = None,
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        dtype_size: int = 4,
        op_type: str = "",
        secondary: Optional[Tuple[str, int]] = None,
        operand_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> TileFootprint:
        """Estimate the bytes a candidate tile keeps live.

        ``secondary`` adds a second (dimension name, tile size) shrink for 2D
        tiles: the output is shrunk along both dimensions, the input tile
        applies the spatial-halo/batch rules per member, and the weight tile
        applies the channel rule for whichever member is the channel dim.

        Two operand models live here, selected by the metadata the caller has
        (``node_cost_kwargs`` produces exactly one of them per op):

        *Convolution* — ``input_shape`` / ``weight_shape`` / ``kernel_shape``.
        Output tile + halo-expanded input tile + the weight slice the tiled
        output channels need. Unchanged.

        *Everything else* — ``operand_shapes``, one entry per tensor the op
        reads. Output tile + one tile per operand. An operand is shrunk along a
        tiled dimension only where it demonstrably indexes it: matmul-family
        operands by position (``C[m,n] = sum_k A[m,k] B[k,n]`` means the LHS is
        indexed by M and the RHS by N), any other operand only when it is
        index- and extent-aligned with the output on that axis. Everything the
        model cannot prove — a reduction axis, a broadcast, a rank mismatch —
        is counted at the operand's FULL extent. That is the conservative
        direction on purpose: proposal §4.8.4 Theorem 3 is a guarantee, so the
        model must over- rather than under-estimate, and a tile that reads a
        whole operand really does keep that operand live. ``kernel_shape``, if
        supplied, applies the same overlap rule the conv path uses to spatial
        members, so a windowed generic pays for its halo too.

        With no operand metadata at all the historical assumption stands — a
        single operand shaped like the output — so callers that pass no shapes
        (and nodes that record no input tensors) keep their previous numbers
        rather than silently dropping the input term.
        """
        members: List[Tuple[str, int]] = [(slice_dim, tile_size)]
        if secondary is not None:
            members.append(secondary)

        output_shape = list(shape)
        for dim_name, dim_tile in members:
            dim_idx = self._dim_name_to_index(dim_name, shape, op_type)
            if 0 <= dim_idx < len(output_shape):
                output_shape[dim_idx] = min(max(1, dim_tile), output_shape[dim_idx])
        output_bytes = self._aligned_bytes(math.prod(output_shape), dtype_size)

        input_bytes = output_bytes
        halo_bytes = 0
        operand_bytes = 0
        if input_shape is not None:
            input_tile = tuple(input_shape)
            total_halo = 0
            # Spatial axes are independent, so the halo rule applies per member
            # on the already-shrunk input tile; a dense conv's channel member
            # leaves the input untouched (every output channel reduces over
            # all input channels), but a depthwise conv's channel member
            # shrinks the input 1:1 with no halo (see _estimate_input_tile_shape).
            for dim_name, dim_tile in members:
                input_tile, halo_elements = self._estimate_input_tile_shape(
                    input_tile,
                    tuple(shape),
                    dim_name,
                    dim_tile,
                    kernel_shape=kernel_shape,
                    stride=stride,
                    dilation=dilation,
                    op_type=op_type,
                )
                total_halo += halo_elements
            input_bytes = self._aligned_bytes(math.prod(input_tile), dtype_size)
            halo_bytes = self._aligned_bytes(total_halo, dtype_size) if total_halo else 0
        elif operand_shapes:
            # Non-conv path. Every operand is live simultaneously inside the
            # tiled loop body, so they are summed, not maxed. Operand 0 keeps
            # reporting as ``input_bytes`` (that is what it is, and what every
            # existing reader of TileFootprint expects); the rest — the term
            # that was missing entirely — lands in ``operand_bytes``.
            operand_tiles = [
                self._estimate_operand_tile_shape(
                    tuple(operand),
                    tuple(shape),
                    tuple(output_shape),
                    members,
                    index,
                    kernel_shape=kernel_shape,
                    stride=stride,
                    dilation=dilation,
                    op_type=op_type,
                )
                for index, operand in enumerate(operand_shapes)
            ]
            operand_sizes = [
                self._aligned_bytes(math.prod(tile), dtype_size)
                for tile, _halo in operand_tiles
            ]
            total_halo = sum(halo for _tile, halo in operand_tiles)
            input_bytes = operand_sizes[0]
            operand_bytes = sum(operand_sizes[1:])
            halo_bytes = self._aligned_bytes(total_halo, dtype_size) if total_halo else 0

        weight_bytes = 0
        if weight_shape is not None:
            weight_tile = tuple(weight_shape)
            for dim_name, dim_tile in members:
                weight_tile = self._estimate_weight_tile_shape(
                    weight_tile, tuple(shape), dim_name, dim_tile, op_type
                )
            weight_bytes = self._aligned_bytes(math.prod(weight_tile), dtype_size)

        total = input_bytes + output_bytes + weight_bytes + operand_bytes
        return TileFootprint(
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            weight_bytes=weight_bytes,
            halo_bytes=halo_bytes,
            total_bytes=total,
            operand_bytes=operand_bytes,
        )

    def estimate_group_footprint_bytes(
        self,
        members: Sequence[Tuple[Tuple[int, ...], Dict[str, Any]]],
        slice_dim: str,
        tile_size: int,
        *,
        secondary: Optional[Tuple[str, int]] = None,
    ) -> int:
        """Combined working set of ops that share one tiled loop body.

        ``members`` are ``(output_shape, cost_kwargs)`` pairs: the tiled anchor
        first, then every producer fused into the same ``scf.forall``. Their
        tiles are live simultaneously, so the budget must be checked against the
        sum rather than the anchor alone.
        """
        return sum(
            self.estimate_tile_footprint(
                shape, slice_dim, tile_size, secondary=secondary, **kwargs
            ).total_bytes
            for shape, kwargs in members
        )

    def compute_for_group(
        self,
        members: Sequence[Tuple[Tuple[int, ...], Dict[str, Any]]],
        slice_dim: str,
        *,
        max_tile_size: Optional[int] = None,
        secondary: Optional[Tuple[str, int]] = None,
    ) -> int:
        """Largest tile whose *combined* group footprint fits the cache budget.

        Same binary search as :meth:`compute`'s working-set path
        (``_compute_with_working_set``, proposal §4.8.4 Theorem 3), but the
        monotone cost function is the sum over the group. Note ``compute``
        only takes that path when called with conv shape info (see
        ``node_cost_kwargs``); without it, ``compute`` falls back to the
        closed-form ``_compute_legacy_activation_tile`` estimate instead of
        searching. Raises ``TileBudgetError`` when even a unit tile does not
        fit, so the caller can reject the plan instead of claiming cache
        residency it cannot deliver.
        """
        if not members:
            raise ValueError("compute_for_group requires at least one member")

        anchor_shape, anchor_kwargs = members[0]
        dim_idx = self._dim_name_to_index(
            slice_dim, anchor_shape, anchor_kwargs.get("op_type", "")
        )
        full_size = anchor_shape[dim_idx] if dim_idx < len(anchor_shape) else 1
        upper = min(full_size, max_tile_size or full_size, self.max_tile_size or full_size)
        upper = max(1, upper)

        unit_group_bytes = self.estimate_group_footprint_bytes(
            members, slice_dim, 1, secondary=secondary
        )
        if not self._fits_cache_budget(unit_group_bytes):
            raise TileBudgetError(
                "unit tile for the fused group requires "
                f"{unit_group_bytes} bytes, "
                "exceeding cache budget"
            )

        low, high, best = 1, upper, 1
        while low <= high:
            candidate = (low + high) // 2
            if self._fits_cache_budget(
                self.estimate_group_footprint_bytes(
                    members, slice_dim, candidate, secondary=secondary
                )
            ):
                best = candidate
                low = candidate + 1
            else:
                high = candidate - 1

        dtype_size = int(anchor_kwargs.get("dtype_size", 4))
        best = self._round_down_to_granule(best, dtype_size)
        best = self._align_tile_to_vector(best, slice_dim, dtype_size)
        return max(1, min(best, upper))

    def get_num_tiles(
        self,
        shape: Tuple[int, ...],
        slice_dim: str,
        op_type: str = "",
        *,
        input_shape: Optional[Tuple[int, ...]] = None,
        weight_shape: Optional[Tuple[int, ...]] = None,
        kernel_shape: Optional[Tuple[int, int]] = None,
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        dtype_size: int = 4,
        operand_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> int:
        """Number of tiles needed to cover ``slice_dim``'s full extent.

        Takes the same cost-model metadata as :meth:`compute` and forwards all
        of it. It used to call ``compute(shape, slice_dim, op_type=op_type)``
        bare: a planner that sized its tile *with* conv/operand metadata (as
        ``EdgeSlicePassManager`` does, via ``node_cost_kwargs``) then got a
        tile count describing a completely different, larger tile — and
        silently, because both calls succeed. Pass whatever
        ``node_cost_kwargs`` produced for the op so a report cannot diverge
        from the plan it claims to describe.
        """
        decision = self.compute_detailed(
            shape,
            slice_dim,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            dilation=dilation,
            dtype_size=dtype_size,
            op_type=op_type,
            operand_shapes=operand_shapes,
        )
        return math.ceil(self._full_extent(shape, slice_dim, op_type) / decision.tile_size)

    def summary(
        self,
        shape: Tuple[int, ...],
        slice_dim: str,
        op_type: str = "",
        *,
        input_shape: Optional[Tuple[int, ...]] = None,
        weight_shape: Optional[Tuple[int, ...]] = None,
        kernel_shape: Optional[Tuple[int, int]] = None,
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        dtype_size: int = 4,
        operand_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> str:
        """Human-readable description of the tile this calculator would pick.

        Threads the cost-model metadata through for the same reason
        :meth:`get_num_tiles` does, and derives the tile count from the single
        :meth:`compute_detailed` call below instead of re-computing it — two
        independent ``compute`` calls were two more chances for the report to
        describe a tile nobody planned. The footprint/utilization lines make
        any remaining divergence self-evident rather than silent.
        """
        decision = self.compute_detailed(
            shape,
            slice_dim,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            dilation=dilation,
            dtype_size=dtype_size,
            op_type=op_type,
            operand_shapes=operand_shapes,
        )
        num_tiles = math.ceil(
            self._full_extent(shape, slice_dim, op_type) / decision.tile_size
        )
        dim_idx = self._dim_name_to_index(slice_dim, shape, op_type)
        return (f"  Tile Size Configuration:\n"
                f"    Shape: {shape}\n"
                f"    Slice dimension: {slice_dim} (index {dim_idx})\n"
                f"    Full size: {shape[dim_idx] if dim_idx < len(shape) else 0}\n"
                f"    Tile size: {decision.tile_size}\n"
                f"    Number of tiles: {num_tiles}\n"
                f"    Tile working set: {decision.footprint_bytes} bytes\n"
                f"    Budget utilization: {decision.utilization:.1%} of "
                f"{decision.budget_bytes} bytes\n"
                f"    L1 cache budget: {self.l1_cache_bytes / 1024:.0f} KB\n"
                f"    L2 cache budget: {self.hardware_config.l2_cache_bytes or 'not set'}\n"
                f"    Vector width: {self.hardware_config.vector_width_bytes} bytes\n"
                f"    Cache line: {self.hardware_config.cache_line_bytes} bytes")

    def _compute_operand_aware_tile(
        self,
        shape: Tuple[int, ...],
        slice_dim: str,
        full_size: int,
        cost: Dict[str, Any],
    ) -> int:
        """Largest budget-fitting tile for a non-convolution op.

        The same monotone binary search ``_compute_with_working_set`` runs
        (proposal §4.8.4 Theorem 3), but over the operand-aware footprint
        instead of the closed form this replaced,
        ``budget / (bytes_per_output_unit * 2)``. That form hard-coded "one
        operand, shaped exactly like the output": it could not see a matmul's B
        matrix, a broadcast bias, or a second addend, and it had no halo term
        at all. Since it is the path every ``matmul`` and every
        ``linalg.generic`` takes — the ops fusion chains are made of — its
        under-estimate propagated straight into fusion admission.
        """
        max_candidate = min(full_size, self.max_tile_size or full_size)
        dtype_size = int(cost.get("dtype_size", 4))
        unit_bytes = self.estimate_tile_footprint(shape, slice_dim, 1, **cost).total_bytes
        if not self._fits_cache_budget(unit_bytes):
            raise TileBudgetError(
                f"unit tile requires {unit_bytes} bytes, exceeding cache budget"
            )

        low, high = 1, max_candidate
        best = 1
        while low <= high:
            candidate = (low + high) // 2
            footprint = self.estimate_tile_footprint(shape, slice_dim, candidate, **cost)
            if self._fits_cache_budget(footprint.total_bytes):
                best = candidate
                low = candidate + 1
            else:
                high = candidate - 1

        best = self._round_down_to_granule(best, dtype_size)
        best = self._align_tile_to_vector(best, slice_dim, dtype_size)
        return max(1, min(best, max_candidate))

    def _compute_with_working_set(
        self,
        shape: Tuple[int, ...],
        slice_dim: str,
        full_size: int,
        cost: Dict[str, Any],
    ) -> int:
        max_candidate = min(full_size, self.max_tile_size or full_size)
        dtype_size = int(cost.get("dtype_size", 4))
        unit_footprint = self.estimate_tile_footprint(shape, slice_dim, 1, **cost)
        if not self._fits_cache_budget(unit_footprint.total_bytes):
            raise TileBudgetError(
                f"unit tile requires {unit_footprint.total_bytes} bytes, exceeding cache budget"
            )
        low, high = 1, max_candidate
        best = 1
        while low <= high:
            candidate = (low + high) // 2
            footprint = self.estimate_tile_footprint(shape, slice_dim, candidate, **cost)
            if self._fits_cache_budget(footprint.total_bytes):
                best = candidate
                low = candidate + 1
            else:
                high = candidate - 1

        best = self._round_down_to_granule(best, dtype_size)
        best = self._align_tile_to_vector(best, slice_dim, dtype_size)
        return max(1, min(best, max_candidate))

    def _accept(
        self,
        shape: Tuple[int, ...],
        slice_dim: str,
        tile_size: int,
        cost: Dict[str, Any],
        *,
        label: str,
        note: str = "",
    ) -> TileDecision:
        """Cost ``tile_size``, enforce the budget, and package the evidence.

        Single choke point for proposal §4.8.4 Theorem 3: nothing leaves
        :meth:`compute_detailed` without passing through here, so a future path
        that forgets to check the budget cannot exist.
        """
        footprint = self.estimate_tile_footprint(shape, slice_dim, tile_size, **cost)
        level, budget = self._binding_level(footprint.total_bytes)
        if not self._fits_cache_budget(footprint.total_bytes):
            raise TileBudgetError(
                f"{label} requires {footprint.total_bytes} bytes, exceeding cache "
                f"budget of {budget} bytes{note}"
            )
        return TileDecision(
            tile_size=tile_size,
            footprint_bytes=footprint.total_bytes,
            budget_bytes=budget,
            utilization=footprint.total_bytes / budget,
            budget_level=level,
        )

    def _full_extent(self, shape: Tuple[int, ...], slice_dim: str, op_type: str) -> int:
        dim_idx = self._dim_name_to_index(slice_dim, shape, op_type)
        return shape[dim_idx] if dim_idx < len(shape) else 1

    def _estimate_input_tile_shape(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        slice_dim: str,
        tile_size: int,
        *,
        kernel_shape: Optional[Tuple[int, int]],
        stride: Tuple[int, int],
        dilation: Tuple[int, int],
        op_type: str,
    ) -> Tuple[Tuple[int, ...], int]:
        input_tile = list(input_shape)
        halo_elements = 0
        output_extent = max(1, tile_size)

        if len(input_shape) >= 4 and len(output_shape) >= 4 and slice_dim == "spatial_h":
            h_index = dimension_name_to_index("spatial_h", op_type, len(input_shape))
            kernel_h = self._effective_kernel(kernel_shape, dilation, axis=0)
            needed_h = (output_extent - 1) * stride[0] + kernel_h
            non_halo_h = max(1, output_extent * stride[0])
            input_tile[h_index] = min(input_shape[h_index], max(1, needed_h))
            halo_h = max(0, input_tile[h_index] - min(input_shape[h_index], non_halo_h))
            if halo_h:
                halo_shape = list(input_tile)
                halo_shape[h_index] = halo_h
                halo_elements = math.prod(halo_shape)
        elif len(input_shape) >= 4 and len(output_shape) >= 4 and slice_dim == "spatial_w":
            w_index = dimension_name_to_index("spatial_w", op_type, len(input_shape))
            kernel_w = self._effective_kernel(kernel_shape, dilation, axis=1)
            needed_w = (output_extent - 1) * stride[1] + kernel_w
            non_halo_w = max(1, output_extent * stride[1])
            input_tile[w_index] = min(input_shape[w_index], max(1, needed_w))
            halo_w = max(0, input_tile[w_index] - min(input_shape[w_index], non_halo_w))
            if halo_w:
                halo_shape = list(input_tile)
                halo_shape[w_index] = halo_w
                halo_elements = math.prod(halo_shape)
        elif (
            len(input_shape) >= 2
            and len(output_shape) >= 2
            and slice_dim == "channel"
            and is_depthwise_op(op_type)
        ):
            # Depthwise only: output channel c reads exactly input channel c,
            # so the input tile needs only the matching channel slice — no
            # halo, unlike spatial tiling which must overlap-read by
            # kernel-1 pixels.
            c_index = dimension_name_to_index("channel", op_type, len(input_shape))
            input_tile[c_index] = min(input_shape[c_index], output_extent)
        elif len(input_shape) >= 1 and slice_dim == "batch":
            input_tile[0] = min(input_shape[0], output_extent)

        return tuple(input_tile), halo_elements

    def _estimate_operand_tile_shape(
        self,
        operand_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        tiled_output_shape: Tuple[int, ...],
        members: Sequence[Tuple[str, int]],
        operand_index: int,
        *,
        kernel_shape: Optional[Tuple[int, int]],
        stride: Tuple[int, int],
        dilation: Tuple[int, int],
        op_type: str,
    ) -> Tuple[Tuple[int, ...], int]:
        """One non-conv operand's tile shape, plus the halo elements it re-reads.

        Shrinking an operand along a tiled output dimension is only sound when
        that dimension actually indexes it; where it does not — a reduction
        axis, a broadcast — the tile reads the operand whole, so the model
        keeps its full extent (see :meth:`estimate_tile_footprint`'s docstring
        for why over-estimating is the required direction here).
        """
        operand_tile = list(operand_shape)
        halo_elements = 0
        if not operand_shape:
            return tuple(operand_tile), 0

        if (
            len(operand_shape) != len(output_shape)
            and math.prod(operand_shape) == math.prod(output_shape)
        ):
            # Rank-changing but element-preserving: a reshape / collapse_shape
            # / expand_shape sitting in the chain. No axis of the operand lines
            # up with a tiled output axis, but the element counts do, so a tile
            # covering a fraction f of the output reads exactly f of the
            # operand. Falling back to "reads the whole operand" here is not
            # conservatism, it is wrong by the tile count — and it would keep
            # every reshape out of every fused loop
            # (see test_a_producer_of_a_different_output_rank_is_not_fused).
            return (math.prod(tiled_output_shape),), 0

        for dim_name, dim_tile in members:
            axis = self._operand_axis_for(
                dim_name, operand_shape, output_shape, operand_index, op_type
            )
            if axis is None:
                continue
            extent = max(1, min(dim_tile, operand_shape[axis]))
            halo_axis = _SPATIAL_HALO_AXIS.get(dim_name)
            if halo_axis is not None and kernel_shape is not None:
                # Same overlap rule as the conv path (proposal §4.8.4:
                # h_in = (h_out - 1)*s_h + d_h*(k_h - 1) + 1). A windowed
                # linalg.generic — a pooling body, a stencil — re-reads
                # kernel-1 rows per tile, and the old non-conv model had no
                # halo term at all, so it under-counted exactly the bytes
                # overlap slicing exists to pay for.
                span = self._effective_kernel(kernel_shape, dilation, axis=halo_axis)
                needed = (extent - 1) * stride[halo_axis] + span
                non_halo = min(operand_shape[axis], max(1, extent * stride[halo_axis]))
                extent = min(operand_shape[axis], max(1, needed))
                operand_tile[axis] = extent
                overlap = max(0, extent - non_halo)
                if overlap:
                    halo_shape = list(operand_tile)
                    halo_shape[axis] = overlap
                    halo_elements += math.prod(halo_shape)
                continue
            operand_tile[axis] = extent

        return tuple(operand_tile), halo_elements

    @staticmethod
    def _operand_axis_for(
        dim_name: str,
        operand_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        operand_index: int,
        op_type: str,
    ) -> Optional[int]:
        """Axis of ``operand_shape`` that the tiled output dim ``dim_name`` indexes.

        ``None`` means "cannot prove this operand is indexed by that
        dimension", and the caller then keeps the operand's full extent.
        """
        lowered = op_type.lower()
        if "matmul" in lowered and operand_index < 2:
            # C[.., m, n] = sum_k A[.., m, k] * B[.., k, n]. The LHS is indexed
            # by M (and the batch), the RHS by N (and the batch); K is a
            # reduction axis, so an M-tile still reads *all* of B and an N-tile
            # all of A. Resolving this positionally rather than by comparing
            # extents matters precisely for square matmuls — the common case —
            # where K == M == N would let extent-matching shrink the wrong
            # operand and halve the estimated working set.
            wanted = ("batch", "M") if operand_index == 0 else ("batch", "N")
            if dim_name not in wanted:
                return None
            try:
                axis = dimension_name_to_index(dim_name, op_type, len(operand_shape))
            except ValueError:
                return None
            return axis if 0 <= axis < len(operand_shape) else None

        # linalg.generic and friends carry no indexing maps here, so equal rank
        # plus an equal extent on the axis is the only available evidence that
        # the operand is walked in lockstep with the output. A differing extent
        # is a broadcast or a reduction; either way the tile touches the whole
        # extent of that axis.
        if len(operand_shape) != len(output_shape):
            return None
        try:
            axis = dimension_name_to_index(dim_name, op_type, len(output_shape))
        except ValueError:
            return None
        if not (0 <= axis < len(operand_shape)):
            return None
        if operand_shape[axis] != output_shape[axis]:
            return None
        return axis

    @staticmethod
    def _estimate_weight_tile_shape(
        weight_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        slice_dim: str,
        tile_size: int,
        op_type: str,
    ) -> Tuple[int, ...]:
        weight_tile = list(weight_shape)
        if slice_dim == "channel" and len(weight_shape) >= 1 and len(output_shape) >= 2:
            channel_index = dimension_name_to_index("channel", op_type, len(output_shape))
            output_channels = output_shape[channel_index]
            if weight_shape[0] == output_channels:
                weight_tile[0] = min(weight_shape[0], tile_size)
            elif len(weight_shape) >= 4 and weight_shape[-1] == output_channels:
                weight_tile[-1] = min(weight_shape[-1], tile_size)
        return tuple(weight_tile)

    @staticmethod
    def _effective_kernel(
        kernel_shape: Optional[Tuple[int, int]],
        dilation: Tuple[int, int],
        *,
        axis: int,
    ) -> int:
        if kernel_shape is None:
            kernel = 1
        else:
            kernel = kernel_shape[axis]
        return dilation[axis] * (kernel - 1) + 1

    @property
    def budget_bytes(self) -> int:
        """The outermost admissible budget, in bytes (proposal §4.8.4).

        This is the budget a tile must fit to be accepted at all: the L1 budget
        on a single-level config, the L2 budget when the two-level search is
        enabled. Utilization is reported against the level that actually bound
        the tile (``TileDecision.budget_level``), not against this, so an
        L1-resident tile on a two-level machine is not made to look like 3% of
        the budget when it is 94% of L1.
        """
        return self.hardware_config.budget_levels()[-1][1]

    @property
    def l1_budget_bytes(self) -> int:
        """The innermost budget — what "cache-resident" means by default."""
        return self.hardware_config.l1_budget_bytes

    def budget_utilization(self, footprint_bytes: int) -> float:
        """Fraction of the binding budget a footprint claims.

        Exposed separately from :class:`TileDecision` so callers holding a
        footprint they computed themselves — the 2D and fused-group paths in
        ``EdgeSlicePassManager`` — can report utilization per accepted tile
        without re-running a search.
        """
        return footprint_bytes / self.budget_bytes

    def _fits_cache_budget(self, bytes_required: int) -> bool:
        """Whether a footprint is admissible at *any* configured cache level.

        Previously this required a tile to fit L1 *and* L2, which inverted the
        point of having two levels: L2 is larger, so the conjunction meant
        configuring an L2 could only ever make the rule stricter, and never
        admitted anything L1 alone rejected. A tile that overflows L1 but fits
        L2 is legitimately tileable — it streams through L1 one line at a time
        — and refusing it is what forced whole ops to be dropped.
        """
        return any(
            bytes_required <= budget
            for _, budget in self.hardware_config.budget_levels()
        )

    def binding_level_name(self, bytes_required: int) -> str:
        """Name of the innermost cache level that holds *bytes_required*.

        For callers that computed a footprint themselves — the 2D cascade and
        the uniform baseline, neither of which runs a budget search — and need
        to report the level without a :class:`TileDecision`.
        """
        return self._binding_level(bytes_required)[0]

    def _binding_level(self, bytes_required: int) -> Tuple[str, int]:
        """The innermost level that holds *bytes_required*, else the outermost.

        Falling back to the outermost on overflow keeps the utilization ratio
        reported by a rejection meaningful (>1.0 against the budget that was
        actually available) instead of dividing by the wrong level.
        """
        levels = self.hardware_config.budget_levels()
        for name, budget in levels:
            if bytes_required <= budget:
                return name, budget
        return levels[-1]

    def _aligned_bytes(self, elements: int, dtype_size: int) -> int:
        raw_bytes = max(0, elements) * dtype_size
        return self._align_up(raw_bytes, self.hardware_config.cache_line_bytes)

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        if alignment <= 1:
            return value
        return ((value + alignment - 1) // alignment) * alignment

    def _align_tile_to_vector(self, tile_size: int, slice_dim: str, dtype_size: int) -> int:
        if slice_dim != "channel":
            return tile_size
        lanes = max(1, self.hardware_config.vector_width_bytes // max(1, dtype_size))
        if tile_size < lanes:
            return tile_size
        return max(lanes, (tile_size // lanes) * lanes)

    @staticmethod
    def _dim_name_to_index(name: str, shape: Tuple[int, ...], op_type: str = "") -> int:
        return dimension_name_to_index(name, op_type, len(shape))

    def _round_down_to_granule(self, tile_size: int, dtype_size: int) -> int:
        """Round a proven-feasible tile down to a whole cache line of elements.

        Replaces a power-of-two round-down. Proposal §4.8.4 asks for
        "power-of-two friendly" tiles, but halving is not a rounding: a best
        feasible tile of 63 came back as 32, discarding 49% of a budget the
        binary search had just proven affordable, and the worst case is always
        a hair under 50%. A cache-line granule keeps the two properties the
        rounding existed for — tiles that begin and end on line boundaries, and
        a whole number of vectors, since a 64-byte line is an integral number
        of 16-byte vectors — at a cost of at most ``granule - 1`` elements.

        Only ever rounds down: the search's feasibility guarantee is monotone
        downward, so rounding up could hand back an over-budget tile. And never
        below the granule: a best that is already sub-line cannot be made
        line-aligned, so clamping it to 1 would throw away everything for
        nothing.
        """
        if tile_size <= 0:
            return 1
        granule = self._tile_granule(dtype_size)
        if tile_size < granule:
            return tile_size
        return max(1, (tile_size // granule) * granule)

    def _tile_granule(self, dtype_size: int) -> int:
        """Elements per cache line — the tile-size quantum on this target."""
        return max(1, self.hardware_config.cache_line_bytes // max(1, dtype_size))
