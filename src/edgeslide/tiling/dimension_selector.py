"""Dimension selection and layout-aware dimension mapping."""

from __future__ import annotations

from typing import List, Optional, Tuple

from edgeslide.dag.graph import OperationDAG

# NCHW conv/pool dimension indices
_NCHW_BATCH = 0
_NCHW_CHANNEL = 1
_NCHW_SPATIAL_H = 2
_NCHW_SPATIAL_W = 3


class DimensionSelector:
    """Select the best dimension to slice for a given operation."""

    PRIORITY = {
        "conv_2d": [0, 2, 3, 1],
        "conv_2d_nhwc": [0, 1, 2, 3],
        # Channel before spatial: a depthwise output channel reads only the
        # matching input channel, so a channel tile needs no halo and shrinks
        # the input tile too (see rank_dimension_indices / tile_size.py's
        # _estimate_input_tile_shape). A dense conv's channel tile leaves the
        # input untouched, which is why conv_2d ranks spatial first instead.
        "depthwise_conv": [0, 1, 2, 3],
        "matmul": [0, 1],
        "batch_matmul": [0, 1, 2],
        "elementwise_add": [0, 1, 2, 3],
        "elementwise_mul": [0, 1, 2, 3],
        "relu": [0, 1, 2, 3],
        "pooling": [0, 2, 3, 1],
    }

    MIN_TILE_SIZE = 1

    def __init__(self, dag: OperationDAG):
        self.dag = dag

    def select(self, op_id: int) -> str:
        """Select the best slicing dimension for the given op."""
        node = self.dag.nodes.get(op_id)
        if node is None:
            return "batch"
        idx = select_dimension_index(node.output_tensor.shape, node.op_type)
        return dimension_index_to_name(idx, node.op_type, len(node.output_tensor.shape))

    def rank(self, op_id: int) -> list:
        """Ordered slice-dimension names for the op, most preferred first."""
        node = self.dag.nodes.get(op_id)
        if node is None:
            return ["batch"]
        shape = node.output_tensor.shape
        return [
            dimension_index_to_name(index, node.op_type, len(shape))
            for index in rank_dimension_indices(shape, node.op_type)
        ]

    def rank_pairs(self, op_id: int) -> List[Tuple[str, str]]:
        """Ordered (primary, secondary) dimension-name pairs for 2D tiling.

        Used when no single dimension has a cache-fitting unit tile (e.g. a
        256x256x256 map against a 256 KB L1: one row and one channel are each
        a whole cache). Pairs follow the 1D preference order.
        """
        node = self.dag.nodes.get(op_id)
        if node is None:
            return []
        shape = node.output_tensor.shape
        rank = len(shape)
        return [
            (
                dimension_index_to_name(primary, node.op_type, rank),
                dimension_index_to_name(secondary, node.op_type, rank),
            )
            for primary, secondary in rank_dimension_index_pairs(shape, node.op_type)
        ]

    def select_all(self) -> dict:
        """Select slicing dimensions for all ops in the DAG."""
        return {op_id: self.select(op_id) for op_id in self.dag.nodes}


def _normalize_op_type(op_type: str) -> str:
    lowered = op_type.lower()
    # Must run before the conv_2d_nchw check below: torch-mlir's depthwise
    # linalg op is named "linalg.depthwise_conv_2d_nchw_chw", which contains
    # "conv_2d_nchw" as a substring and would otherwise be misclassified as a
    # dense conv_2d, silently making the "depthwise_conv" PRIORITY entry dead
    # code (verified: it was never reached before this check existed).
    if "depthwise" in lowered:
        return "depthwise_conv"
    if "conv_2d_nchw" in lowered or "conv_2d_nhwc" in lowered:
        return "conv_2d"
    if "conv" in lowered:
        return "conv_2d"
    if "pool" in lowered:
        return "pooling"
    if "batch_matmul" in lowered:
        return "batch_matmul"
    if "matmul" in lowered:
        return "matmul"
    return op_type


def is_nhwc_op(op_type: str) -> bool:
    return "nhwc" in op_type.lower()


def is_depthwise_op(op_type: str) -> bool:
    """True for a depthwise/grouped conv (e.g. linalg.depthwise_conv_2d_nchw_chw).

    Unlike a dense conv, each output channel of a depthwise conv reads only
    the matching input channel — no cross-channel reduction. That changes
    which slice dimension is cheapest (see rank_dimension_indices) and how
    much a channel-dim tile shrinks the input (see tile_size.py's
    _estimate_input_tile_shape).
    """
    return "depthwise" in op_type.lower()


def dimension_indices(op_type: str, rank: int) -> dict[str, int]:
    """Return semantic tensor-dimension indices for a supported layout."""
    lowered = op_type.lower()
    if "batch_matmul" in lowered and rank >= 3:
        return {"batch": 0, "M": rank - 2, "N": rank - 1}
    if "matmul" in lowered and rank >= 2:
        return {"M": rank - 2, "N": rank - 1}
    if rank >= 4 and is_nhwc_op(op_type):
        return {"batch": 0, "spatial_h": 1, "spatial_w": 2, "channel": 3}
    if rank >= 4:
        return {"batch": 0, "channel": 1, "spatial_h": 2, "spatial_w": 3}
    return {f"dim_{index}": index for index in range(rank)}


def dimension_name_to_index(name: str, op_type: str, rank: int) -> int:
    """Map a semantic dimension name to its tensor/parallel-loop index."""
    indices = dimension_indices(op_type, rank)
    if name in indices:
        return indices[name]
    if name.startswith("dim_"):
        try:
            index = int(name.split("_", 1)[1])
            if 0 <= index < rank:
                return index
        except ValueError:
            pass
    raise ValueError(f"Unsupported slice dimension {name!r} for {op_type} rank {rank}")


def rank_dimension_indices(shape: Tuple[int, ...], op_type: str) -> List[int]:
    """All slicing dimensions in preference order, most preferred first.

    Same priority as :func:`select_dimension_index` (proposal §4.4) but keeps
    the runners-up so a caller can fall back when the preferred dimension has
    no tile size that fits the cache budget. Committing to one dimension and
    giving up is what makes batch>1 workloads plan zero tiles: a unit batch
    tile is one whole sample, which for large models never fits L1.
    """
    rank = len(shape)
    if rank == 0:
        return [0]

    indices = dimension_indices(op_type, rank)
    ordered: List[int] = []

    def push(index: Optional[int]) -> None:
        if index is not None and 0 <= index < rank and index not in ordered:
            ordered.append(index)

    batch_idx = indices.get("batch")
    if batch_idx is not None and shape[batch_idx] > 1:
        push(batch_idx)

    spatial = [indices[name] for name in ("spatial_h", "spatial_w") if name in indices]
    channel_idx = indices.get("channel")

    if is_depthwise_op(op_type):
        # Depthwise: output channel c reads only input channel c, so a
        # channel tile needs no halo and shrinks the input tile too. A dense
        # conv's channel tile leaves the input untouched (every output
        # channel reduces over all input channels), which is why the dense
        # path below ranks spatial ahead of channel instead. Measured on
        # hardware_profiler's depthwise_conv model: spatial-first tiling
        # made D1 misses 42% worse and wall clock 79% slower than untiled.
        push(channel_idx)
        for index in sorted(spatial, key=lambda i: shape[i], reverse=True):
            push(index)
    else:
        for index in sorted(spatial, key=lambda i: shape[i], reverse=True):
            push(index)

    matrix = [indices[name] for name in ("M", "N") if name in indices]
    for index in sorted(matrix, key=lambda i: shape[i], reverse=True):
        push(index)

    push(channel_idx)

    normalized = _normalize_op_type(op_type)
    for dim_idx in DimensionSelector.PRIORITY.get(normalized, list(range(rank))):
        # A size-1 dim cannot be sliced any further; picking it would
        # silently disable slicing even when a larger dim is available.
        if dim_idx < rank and shape[dim_idx] > DimensionSelector.MIN_TILE_SIZE:
            push(dim_idx)

    if not ordered:
        ordered.append(0)
    return ordered


def rank_dimension_index_pairs(
    shape: Tuple[int, ...], op_type: str
) -> List[Tuple[int, int]]:
    """Ordered (primary, secondary) index pairs for 2D tiling.

    Built from the top of :func:`rank_dimension_indices`: both members must be
    sliceable (extent > 1), the primary is the higher-ranked dimension, and
    pairs are emitted in rank order. Only the top three ranked dimensions are
    combined — a fourth choice means three dimensions already failed, and 3D
    tiling is out of scope.

    Pairs that leave the innermost (contiguous, fastest-varying) dimension
    whole come first. Tiling that axis slices each row into fragments, so a
    tile touches one partial cache line per row: measured on simple_cnn, an
    ``h x w`` tile with ``w = 1`` raised D1 misses ~20% over the untiled
    binary while the equivalent-footprint ``h x channel`` tile kept rows
    contiguous. The byte-footprint cost model cannot see this — it counts
    bytes, not locality — so the ordering has to encode it.
    """
    ranked = rank_dimension_indices(shape, op_type)
    usable = [index for index in ranked if index < len(shape) and shape[index] > 1][:3]
    pairs = [
        (primary, secondary)
        for position, primary in enumerate(usable)
        for secondary in usable[position + 1:]
    ]
    contiguous_axis = len(shape) - 1
    return sorted(pairs, key=lambda pair: contiguous_axis in pair)


def select_dimension_index(shape: Tuple[int, ...], op_type: str) -> int:
    """
    Shared dimension heuristic for MLIR and PyTorch demo paths.

    Priority: batch (when N > 1) > spatial argmax(H, W) (when N == 1) > channel.
    Proposal §4.4: batch=1 defers to argmax(H, W) instead of static priority.
    """
    return rank_dimension_indices(shape, op_type)[0]


def dimension_index_to_name(idx: int, op_type: str, rank: int) -> str:
    """Map a dimension index to a human-readable slice dimension name."""
    for name, index in dimension_indices(op_type, rank).items():
        if index == idx:
            return name
    return f"dim_{idx}"


def select_dimension_name(shape: Tuple[int, ...], op_type: str) -> str:
    """Convenience wrapper used by the PyTorch demo slicer."""
    idx = select_dimension_index(shape, op_type)
    return dimension_index_to_name(idx, op_type, len(shape))
