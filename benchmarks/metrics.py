"""Shared metrics for EdgeSlice evaluation (proposal §5.2)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from edgeslide.dag.graph import OperationDAG
from edgeslide.liveness.peak_memory import PeakMemoryEstimator
from edgeslide.tiling.dimension_selector import dimension_name_to_index


def dim_name_to_index(dim: str, rank: int, op_type: str = "") -> int:
    """Semantic dimension name → tensor index.

    Delegates to ``edgeslide.tiling.dimension_selector`` so the estimator
    shrinks the same dimension the transform actually tiles (the two mappings
    had already diverged once for rank-2 matmul). Raises ``ValueError`` for a
    (dim, op_type, rank) combination the transform itself would reject —
    every plan builder in this repo stamps ``op_type``, so an unmappable name
    means the plan is malformed and must fail loudly rather than silently
    shrink the wrong dimension.
    """
    return dimension_name_to_index(dim, op_type, rank)


def tiled_shape(
    shape: Tuple[int, ...],
    slice_dim: str,
    tile_size: int,
    op_type: str = "",
    secondary: Optional[Tuple[str, int]] = None,
) -> Tuple[int, ...]:
    """Shrink one dimension — or two, for a 2D tile — to the tile sizes."""
    tiled = list(shape)
    members = [(slice_dim, tile_size)]
    if secondary is not None:
        members.append(secondary)
    for dim_name, dim_tile in members:
        idx = dim_name_to_index(dim_name, len(shape), op_type)
        if idx < len(tiled):
            tiled[idx] = min(dim_tile, tiled[idx])
    return tuple(tiled)


def plan_secondary(plan: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    """The plan entry's secondary (dim, tile) pair, or None for a 1D tile."""
    dim2 = plan.get("slice_dimension_2")
    if not dim2:
        return None
    return (str(dim2), int(plan["tile_size_2"]))


def anchor_output_shrinks(
    dag: OperationDAG,
    anchor_op_id: int,
    loop_member_ids: FrozenSet[int],
) -> bool:
    """Whether tiling an anchor actually shrinks the anchor's own buffer.

    Usually it does not, and this is the single largest correction to the
    frontend estimate. ``transform.structured.tile_using_forall`` writes each
    tile of the anchor's result into a full-size destination through
    ``tensor.parallel_insert_slice``; the destination is materialised at full
    extent before the loop and is still full extent after it. So the anchor's
    output buffer is exactly as large as it was untiled, and crediting it with
    the tile footprint — as this estimator originally did for every plan entry
    — reports a reduction the rewrite does not produce.

    The one case where the anchor's buffer *does* disappear is when nothing
    outside the loop ever reads it: every consumer is itself a member of the
    same ``scf.forall``, so the value is produced and consumed tile-by-tile and
    never needs a full-size home. That is what this checks.

    An anchor with no consumers at all is a function result: it is returned at
    full size, so it does not shrink.
    """
    node = dag.nodes.get(anchor_op_id)
    if node is None:
        return False
    consumers = set(node.consumers)
    if not consumers:
        return False
    return consumers <= (loop_member_ids - {anchor_op_id})


def estimate_post_slice_peak_bytes(
    dag: OperationDAG,
    rewrite_plan: List[Dict[str, Any]],
    l1_cache_bytes: int,
    *,
    estimator: Optional[PeakMemoryEstimator] = None,
    credit_anchor_outputs: bool = False,
) -> Tuple[int, int]:
    """
    Estimate peak live memory after tiling using per-op tile footprints.

    Only buffers that genuinely stop being materialised at full size are
    credited:

    * **Fused producers** (``fused_producer_op_ids``). A producer fused into
      the anchor's ``scf.forall`` is computed tile-by-tile and consumed
      immediately, so its full-size buffer never materialises at any program
      point. Without this credit the estimator reports 0% for exactly the plans
      where fusion does the most work (torch-mlir's empty→fill→conv→generic
      chains, whose per-op SSA results otherwise count as several live copies
      of one activation).
    * **Anchors whose every consumer is inside the same loop** — see
      ``anchor_output_shrinks``. Rare, but real.

    A tiled anchor is otherwise charged its **full** output, because
    ``tensor.parallel_insert_slice`` writes into a full-size destination that
    the loop does not shrink. Set ``credit_anchor_outputs=True`` to restore the
    old behaviour of crediting every anchor; it is retained only so the
    magnitude of the correction can be reported, and must not be used for a
    headline number.

    Returns (peak_op_id, peak_bytes). Pass a shared ``estimator`` to reuse its
    memoized liveness sweep across several strategy evaluations of the same
    DAG.
    """
    if estimator is None:
        estimator = PeakMemoryEstimator(dag, l1_cache_bytes)
    elif estimator.dag is not dag:
        # The liveness sweep comes from the estimator but output sizes are
        # read from *dag*; mixing two graphs would blend silently.
        raise ValueError("estimator was built for a different DAG")
    if not rewrite_plan:
        return estimator.get_peak()

    def _element_bytes(op_id: int) -> int:
        """Element size of the op's output, defaulting to f32.

        The tile footprint is compared against ``output_tensor.size_bytes``,
        which honours the tensor's dtype — hardcoding 4 here overstates the
        tile for f16/bf16/i8 DAGs and can make ``tile_bytes < out_bytes``
        false, reporting no reduction for a tiling that did shrink the buffer.
        """
        node = dag.nodes.get(op_id)
        return node.output_tensor.element_size_bytes if node is not None else 4

    tile_footprint_by_op: Dict[int, int] = {}

    # Anchors first, so an explicit anchor entry always wins over a
    # fused-membership credit for the same op.
    for plan in rewrite_plan:
        op_id = plan["op_id"]
        loop_members = frozenset([op_id, *(plan.get("fused_producer_op_ids") or [])])
        if not (
            credit_anchor_outputs or anchor_output_shrinks(dag, op_id, loop_members)
        ):
            # The destination stays full-size; charge the untiled buffer.
            continue
        tiled = tiled_shape(
            tuple(plan["original_shape"]),
            plan["slice_dimension"],
            int(plan["tile_size"]),
            op_type=str(plan.get("op_type", "")),
            secondary=plan_secondary(plan),
        )
        tile_footprint_by_op[op_id] = math.prod(tiled) * _element_bytes(op_id)

    for plan in rewrite_plan:
        for fused_id in plan.get("fused_producer_op_ids") or []:
            if fused_id in tile_footprint_by_op:
                continue
            node = dag.nodes.get(fused_id)
            if node is None:
                continue
            try:
                fused_tiled = tiled_shape(
                    tuple(node.output_tensor.shape),
                    plan["slice_dimension"],
                    int(plan["tile_size"]),
                    op_type=node.op_type,
                    secondary=plan_secondary(plan),
                )
            except ValueError:
                # The anchor's slice dimension has no meaning for this
                # producer's layout; keep its full buffer (conservative).
                continue
            tile_footprint_by_op[fused_id] = math.prod(fused_tiled) * _element_bytes(
                fused_id
            )

    # Substitute the tile footprints at every program point the credited
    # buffers are live at, not just where each is defined — a fused producer's
    # full-size buffer never exists anywhere, and the peak is usually at the
    # anchor rather than at the producer.
    adjusted = estimator.estimate_with_output_overrides(tile_footprint_by_op)
    if not adjusted:
        return (-1, 0)
    peak_op = max(adjusted, key=lambda op_id: adjusted[op_id])
    return peak_op, adjusted[peak_op]


def peak_reduction_pct(
    baseline_bytes: int, sliced_bytes: int
) -> Optional[float]:
    """Percentage reduction, or None when there is no baseline to reduce.

    A zero-or-negative baseline means the baseline measurement did not happen
    — a peak of 0 bytes is not a small peak, it is the absence of one. This
    returned 0.0 there, which in a result file is indistinguishable from a
    genuine "measured, no reduction".
    """
    if baseline_bytes <= 0:
        return None
    return round(100.0 * (1.0 - sliced_bytes / baseline_bytes), 2)

def bootstrap_ci(
    samples: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
    statistic: str = "median",
) -> Optional[Tuple[float, float]]:
    """Return a bootstrap confidence interval, or None when there isn't one.

    Two cases used to return an interval anyway, and both read as precision
    that was not measured:

    * **no samples** returned ``(0.0, 0.0)`` — an interval of zero width
      around zero, i.e. "the median is exactly 0, with certainty";
    * **one sample** returned ``(x, x)`` — a zero-width interval from a single
      observation. Resampling one value can only ever yield that value, so the
      bootstrap has no variance to estimate. A zero-width CI beside a real one
      in a results table is a claim of perfect precision.

    Both now return None, so a caller must either report "not available" or
    collect more trials.
    """
    values = np.asarray(samples, dtype=float)
    if values.size < 2:
        return None

    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, values.size), replace=True)
    if statistic == "mean":
        stats = np.mean(draws, axis=1)
    elif statistic == "median":
        stats = np.median(draws, axis=1)
    else:
        raise ValueError(f"Unsupported bootstrap statistic {statistic!r}")

    alpha = 1.0 - confidence
    lower, upper = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lower), float(upper))

def wilcoxon_signed_rank(
    baseline_samples: Sequence[float],
    candidate_samples: Sequence[float],
) -> Dict[str, Any]:
    """Run a paired Wilcoxon signed-rank test on timing samples."""
    paired_count = min(len(baseline_samples), len(candidate_samples))
    if paired_count < 2:
        return {
            "test": "wilcoxon_signed_rank",
            "paired_count": paired_count,
            "statistic": None,
            "p_value": None,
            "reason": "requires at least two paired samples",
        }

    baseline = np.asarray(baseline_samples[:paired_count], dtype=float)
    candidate = np.asarray(candidate_samples[:paired_count], dtype=float)
    if np.allclose(baseline, candidate, rtol=0.0, atol=0.0):
        return {
            "test": "wilcoxon_signed_rank",
            "paired_count": paired_count,
            "statistic": 0.0,
            "p_value": 1.0,
            "reason": "all paired differences are zero",
        }

    from scipy.stats import rankdata, wilcoxon

    result = wilcoxon(baseline, candidate, zero_method="wilcox", alternative="two-sided")
    differences = baseline - candidate
    nonzero = differences[differences != 0]
    ranks = rankdata(np.abs(nonzero))
    positive = float(np.sum(ranks[nonzero > 0]))
    negative = float(np.sum(ranks[nonzero < 0]))
    total_rank = positive + negative
    rank_biserial = (positive - negative) / total_rank if total_rank else 0.0
    return {
        "test": "wilcoxon_signed_rank",
        "paired_count": paired_count,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "rank_biserial_effect_size": float(rank_biserial),
    }

def summarize_timing_samples(samples: Sequence[float]) -> Dict[str, Any]:
    """Summarize timing samples with median and bootstrap CI.

    ``mean_ms``/``median_ms`` are None with no samples — a timing of 0 ms is
    not a fast measurement, it is the absence of one — and ``median_ci95_ms``
    is None whenever fewer than two samples make an interval meaningless.
    """
    values = [float(value) for value in samples]
    if not values:
        return {
            "mean_ms": None,
            "median_ms": None,
            "median_ci95_ms": None,
            "sample_count": 0,
        }
    interval = bootstrap_ci(values)
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "median_ci95_ms": list(interval) if interval is not None else None,
        "sample_count": len(values),
    }

def compare_paired_timings(
    baseline_samples: Sequence[float],
    candidate_samples: Sequence[float],
) -> Dict[str, Any]:
    """Compare paired timing samples without assuming normality."""
    baseline_summary = summarize_timing_samples(baseline_samples)
    candidate_summary = summarize_timing_samples(candidate_samples)
    baseline_median = baseline_summary["median_ms"]
    candidate_median = candidate_summary["median_ms"]
    # None, not 0.0: "no speedup measured" and "measured, no speedup" are
    # different findings, and only one of them belongs in a results table.
    speedup: Optional[float] = None
    if baseline_median and candidate_median is not None and baseline_median > 0:
        speedup = (baseline_median - candidate_median) / baseline_median * 100.0
    paired_count = min(len(baseline_samples), len(candidate_samples))
    paired_diff = [
        float(baseline_samples[index]) - float(candidate_samples[index])
        for index in range(paired_count)
    ]
    return {
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "median_paired_difference_ms": (
            float(np.median(paired_diff)) if paired_diff else None
        ),
        "median_speedup_pct": speedup,
        "wilcoxon": wilcoxon_signed_rank(baseline_samples, candidate_samples),
    }

def numerical_equivalence(
    actual: Any,
    expected: Any,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """Return a NumPy allclose summary for benchmark outputs.

    Tolerance is 1e-4 for the reason given in
    ``tests.mlir_helpers.assert_sliced_mlir_numerical_equivalence``: the
    pipeline's measured worst-case error across five workloads is 1.53e-05, so
    the previous 1e-3 left 66x of slack in which a tenfold numerical
    regression would have passed unnoticed.

    ``headroom`` is returned alongside the verdict so a caller can see how
    close a passing result came, not just that it passed.
    """
    actual_arr = np.asarray(actual, dtype=np.float32)
    expected_arr = np.asarray(expected, dtype=np.float32)
    same_shape = actual_arr.shape == expected_arr.shape
    if same_shape:
        diff = np.abs(actual_arr - expected_arr)
        max_abs_diff = float(np.max(diff)) if diff.size else 0.0
        mean_abs_diff = float(np.mean(diff)) if diff.size else 0.0
        allclose = bool(np.allclose(actual_arr, expected_arr, rtol=rtol, atol=atol))
    else:
        max_abs_diff = None
        mean_abs_diff = None
        allclose = False
    return {
        "allclose": allclose,
        "rtol": rtol,
        "atol": atol,
        "headroom": (
            None
            if max_abs_diff is None
            else (float("inf") if max_abs_diff == 0 else atol / max_abs_diff)
        ),
        "same_shape": same_shape,
        "actual_shape": list(actual_arr.shape),
        "expected_shape": list(expected_arr.shape),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }

def parse_perf_stat_csv(stderr_text: str) -> Dict[str, Any]:
    """Parse ``perf stat -x ,`` stderr into a hardware-counter dictionary."""
    counters: Dict[str, Any] = {}
    for raw_line in stderr_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        value_text, _unit, event = parts[:3]
        if not event:
            continue
        normalized = value_text.replace(",", "")
        if normalized in {"<not supported>", "<not counted>", ""}:
            counters[event] = None
            continue
        try:
            counters[event] = int(normalized)
        except ValueError:
            try:
                counters[event] = float(normalized)
            except ValueError:
                counters[event] = value_text
    return counters


@dataclass
class CompileTiming:
    """DAG analysis compile-time breakdown (proposal §5.2)."""

    parse_ms: float = 0.0
    liveness_ms: float = 0.0
    planning_ms: float = 0.0
    transform_ir_ms: float = 0.0
    total_ms: float = 0.0
    vertices: int = 0
    edges: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parse_ms": round(self.parse_ms, 3),
            "liveness_ms": round(self.liveness_ms, 3),
            "planning_ms": round(self.planning_ms, 3),
            "transform_ir_ms": round(self.transform_ir_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "dag_vertices": self.vertices,
            "dag_edges": self.edges,
            "analysis_complexity": f"O(V+E) V={self.vertices} E={self.edges}",
        }


@dataclass
class StrategyResult:
    """Peak-memory outcome for one slicing strategy."""

    name: str
    peak_memory_bytes: int
    peak_op_id: int
    rewrite_ops: int
    compile_timing: CompileTiming = field(default_factory=CompileTiming)
    rewrite_plan_summary: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def peak_memory_mb(self) -> float:
        return round(self.peak_memory_bytes / (1024 * 1024), 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_memory_mb": self.peak_memory_mb,
            "peak_op_id": self.peak_op_id,
            "rewrite_ops": self.rewrite_ops,
            "compile_timing": self.compile_timing.to_dict(),
            "rewrite_plan_summary": self.rewrite_plan_summary,
        }


def summarize_plan(plan: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    return [
        {
            "op_id": p["op_id"],
            "op_type": p["op_type"],
            "slice_dimension": p["slice_dimension"],
            "tile_size": p["tile_size"],
            "original_shape": list(p["original_shape"]),
        }
        for p in plan[:limit]
    ]
