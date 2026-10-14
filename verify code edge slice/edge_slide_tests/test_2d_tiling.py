"""Tests for 2D (dimension-pair) tiling.

A single sliced dimension is not always enough to reach a cache-resident
tile: for a (1, 256, 256, 256) f32 map against a 256 KB L1, one spatial row
*and* one channel are each 256 KB, so every 1D unit tile overflows and the
planner drops the op — the case behind resnet50's 0% peak reduction at
1024x1024. Slicing two dimensions at once fixes that, and these tests pin the
behaviour end to end: cost model, dimension pairing, planner cascade,
transform IR, and the benchmark estimator's credit.
"""

from __future__ import annotations

import pytest

from edgeslide.dag.graph import OperationDAG, TensorInfo
from edgeslide.tiling.dimension_selector import (
    DimensionSelector,
    rank_dimension_index_pairs,
)
from edgeslide.tiling.tile_size import TileBudgetError, TileSizeCalculator
from edgeslide.transform.pass_manager import EdgeSlicePassConfig, EdgeSlicePassManager
from edgeslide.transform.rewrite import SlicedRewrite

pytestmark = pytest.mark.cpu

L1 = 256 * 1024
# Every 1D unit tile of this shape is exactly one 256 KB cache: 256*256*4 for a
# spatial row, and the same for one channel.
HARD_SHAPE = (1, 256, 256, 256)


@pytest.fixture
def two_d_dag():
    """DAG whose first op needs a 2D tile to fit L1 at all.

    The sink is deliberately tiny so the peak program point is op 0's own —
    where the estimator can credit its tile. (Tiling a producer does not lower
    the peak at a *consumer's* point unless the producer is fused into the
    consumer's loop; that is the ablation-D behaviour, not a 2D-tiling
    question.)
    """
    dag = OperationDAG()
    inp = TensorInfo((1, 3, 256, 256), "f32")
    big = TensorInfo(HARD_SHAPE, "f32")
    out = TensorInfo((1, 8, 8, 8), "f32")

    n0 = dag.add_node("linalg.generic", inputs=[inp], output=big,
                      params={"edgeslice_op_id": "op_0", "elementwise_identity": True})
    n1 = dag.add_node("linalg.generic", inputs=[big], output=out,
                      params={"edgeslice_op_id": "op_1", "elementwise_identity": True})
    dag.add_edge(n0, n1)
    return dag


def test_every_single_dimension_overflows_but_a_pair_fits():
    calc = TileSizeCalculator(L1)
    for dim in ("spatial_h", "spatial_w", "channel"):
        with pytest.raises(TileBudgetError):
            calc.compute(HARD_SHAPE, dim, op_type="linalg.generic")

    primary, secondary = calc.compute_pair(
        HARD_SHAPE, "spatial_h", "channel", op_type="linalg.generic"
    )
    assert primary >= 1 and secondary >= 1
    footprint = calc.estimate_tile_footprint(
        HARD_SHAPE,
        "spatial_h",
        primary,
        op_type="linalg.generic",
        secondary=("channel", secondary),
    )
    assert footprint.total_bytes <= calc.hardware_config.l1_budget_bytes


def test_compute_pair_shrinks_both_dimensions_in_the_footprint():
    calc = TileSizeCalculator(L1)
    one_d = calc.estimate_tile_footprint(
        HARD_SHAPE, "spatial_h", 4, op_type="linalg.generic"
    ).total_bytes
    two_d = calc.estimate_tile_footprint(
        HARD_SHAPE,
        "spatial_h",
        4,
        op_type="linalg.generic",
        secondary=("channel", 8),
    ).total_bytes
    # 8 of 256 channels alongside the same 4 rows.
    assert two_d < one_d


def test_compute_pair_returns_the_largest_budget_fitting_pair():
    """Maximality, not just feasibility.

    Every other assertion in this file is satisfied by a unit tile, so a
    ``compute_pair`` that returned ``(1, 1)`` passed the whole suite — while
    emitting one-element loop nests: maximal loop overhead, no vectorization,
    and a slower binary than no tiling at all, with the byte model still
    reporting a peak reduction.
    """
    calc = TileSizeCalculator(L1)
    budget = calc.hardware_config.l1_budget_bytes

    def footprint(primary: int, secondary: int) -> int:
        return calc.estimate_tile_footprint(
            HARD_SHAPE,
            "spatial_h",
            primary,
            op_type="linalg.generic",
            secondary=("channel", secondary),
        ).total_bytes

    primary, secondary = calc.compute_pair(
        HARD_SHAPE, "spatial_h", "channel", op_type="linalg.generic"
    )
    assert footprint(primary, secondary) <= budget
    # The search grows the primary first, so it must have grown at all.
    assert primary > 1
    # Doubling either axis (the search's own granularity) must overflow.
    assert footprint(primary * 2, secondary) > budget
    assert footprint(primary, secondary * 2) > budget


def test_compute_pair_rejects_a_repeated_dimension():
    calc = TileSizeCalculator(L1)
    with pytest.raises(ValueError, match="must differ"):
        calc.compute_pair(
            HARD_SHAPE, "spatial_h", "spatial_h", op_type="linalg.generic"
        )


def test_compute_pair_still_raises_when_even_a_unit_pair_overflows():
    # 4 bytes of budget: a 1x1 tile of any pair is already too big.
    calc = TileSizeCalculator(8)
    with pytest.raises(TileBudgetError):
        calc.compute_pair(HARD_SHAPE, "spatial_h", "channel", op_type="linalg.generic")


def test_rank_pairs_are_distinct_sliceable_dimensions_in_rank_order():
    pairs = rank_dimension_index_pairs(HARD_SHAPE, "linalg.generic")
    assert pairs, "expected at least one usable dimension pair"
    for primary, secondary in pairs:
        assert primary != secondary
        assert HARD_SHAPE[primary] > 1 and HARD_SHAPE[secondary] > 1

    # Batch is size 1 here, so it can never appear in a pair.
    assert all(0 not in pair for pair in pairs)


def test_pairs_that_keep_rows_contiguous_come_first():
    """Regression: the innermost axis must be the last one tiled.

    Ordering pairs purely by 1D rank puts (spatial_h, spatial_w) first, and
    the byte-only cost model happily accepts w=1 — a tile that touches one
    partial cache line per row. Measured on simple_cnn that raised D1 misses
    ~20% above the untiled binary; keeping rows whole turned the same
    workload into a 33% traffic reduction.
    """
    nchw_pairs = rank_dimension_index_pairs(HARD_SHAPE, "linalg.generic")
    contiguous = len(HARD_SHAPE) - 1
    assert contiguous not in nchw_pairs[0], nchw_pairs
    # Pairs touching the contiguous axis are kept, just demoted.
    assert any(contiguous in pair for pair in nchw_pairs)
    ranks = [contiguous in pair for pair in nchw_pairs]
    assert ranks == sorted(ranks), f"contiguous-axis pairs must sort last: {nchw_pairs}"

    # NHWC puts channel innermost, so the rule follows the layout.
    nhwc_pairs = rank_dimension_index_pairs(HARD_SHAPE, "linalg.conv_2d_nhwc_fhwc")
    assert contiguous not in nhwc_pairs[0], nhwc_pairs


def test_planner_prefers_a_locality_friendly_pair(two_d_dag):
    plan = next(
        p
        for p in EdgeSlicePassManager(
            two_d_dag, config=EdgeSlicePassConfig(l1_cache_bytes=L1)
        ).run()["rewrite_plan"]
        if p["op_id"] == 0
    )
    assert "spatial_w" not in (plan["slice_dimension"], plan["slice_dimension_2"]), (
        "the innermost NCHW axis must stay whole while another pair fits"
    )


def test_selector_rank_pairs_returns_names(two_d_dag):
    pairs = DimensionSelector(two_d_dag).rank_pairs(0)
    assert ("spatial_h", "spatial_w") in pairs or ("spatial_h", "channel") in pairs
    for primary, secondary in pairs:
        assert isinstance(primary, str) and isinstance(secondary, str)
        assert primary != secondary


def test_planner_falls_back_to_a_2d_tile(two_d_dag):
    pm = EdgeSlicePassManager(
        two_d_dag, config=EdgeSlicePassConfig(l1_cache_bytes=L1)
    )
    results = pm.run()
    plans = {p["op_id"]: p for p in results["rewrite_plan"]}

    assert 0 in plans, "the op that needs a 2D tile must be planned, not rejected"
    plan = plans[0]
    assert plan["slice_dimension_2"], "expected a secondary slice dimension"
    assert plan["tile_size_2"] >= 1
    assert plan["slice_dimension"] != plan["slice_dimension_2"]
    assert plan["estimated_tile_working_set_bytes"] <= int(L1 * 0.8)


def test_disabling_the_2d_cascade_rejects_the_op(two_d_dag):
    pm = EdgeSlicePassManager(
        two_d_dag,
        config=EdgeSlicePassConfig(l1_cache_bytes=L1, enable_2d_cascade=False),
    )
    results = pm.run()

    assert all(p["op_id"] != 0 for p in results["rewrite_plan"])
    reasons = {r["op_id"]: r["reason"] for r in results["planning_rejections"]}
    assert "no slice dimension fits the cache budget" in reasons[0]


def test_transform_ir_emits_two_nonzero_tile_sizes(two_d_dag):
    pm = EdgeSlicePassManager(
        two_d_dag, config=EdgeSlicePassConfig(l1_cache_bytes=L1)
    )
    results = pm.run()
    plan = next(p for p in results["rewrite_plan"] if p["op_id"] == 0)

    rewriter = SlicedRewrite(two_d_dag, [plan], l1_cache_bytes=L1)
    sizes = rewriter._build_tile_sizes(
        tuple(plan["original_shape"]),
        plan["slice_dimension"],
        plan["tile_size"],
        plan["op_type"],
        (plan["slice_dimension_2"], plan["tile_size_2"]),
    )
    assert sum(1 for s in sizes if s) == 2, sizes

    ir = rewriter.generate_transform_ir()
    tile_line = next(line for line in ir.splitlines() if "tile_sizes [" in line)
    emitted = [
        int(value)
        for value in tile_line.split("[", 1)[1].split("]", 1)[0].split(",")
    ]
    assert emitted == sizes


@pytest.mark.parametrize("style", ["structured", "legacy_for", "aggressive"])
def test_every_emission_style_carries_the_secondary_tile(two_d_dag, style):
    pm = EdgeSlicePassManager(
        two_d_dag, config=EdgeSlicePassConfig(l1_cache_bytes=L1)
    )
    plan = next(p for p in pm.run()["rewrite_plan"] if p["op_id"] == 0)
    ir = SlicedRewrite(
        two_d_dag, [plan], l1_cache_bytes=L1
    ).generate_transform_ir(style=style)

    tile_line = next(line for line in ir.splitlines() if "tile_sizes [" in line)
    emitted = [
        int(value)
        for value in tile_line.split("[", 1)[1].split("]", 1)[0].split(",")
    ]
    assert sum(1 for value in emitted if value) == 2, (style, emitted)


def test_tiled_shape_shrinks_both_dimensions():
    pytest.importorskip("numpy", reason="numpy is required by benchmarks.metrics")
    from benchmarks.metrics import tiled_shape

    assert tiled_shape(
        HARD_SHAPE, "spatial_h", 4, op_type="linalg.generic",
        secondary=("channel", 8),
    ) == (1, 8, 4, 256)


def test_metrics_credits_the_secondary_dimension():
    """A 2D plan must model a lower peak than the same plan read as 1D.

    Uses a fused chain (two same-shape ops, tiny sink) because that is where
    tiling actually moves the peak: the anchor's loop consumes the producer's
    tile, so both buffers shrink at the same program point.
    """
    pytest.importorskip("numpy", reason="numpy is required by benchmarks.metrics")
    from benchmarks.metrics import estimate_post_slice_peak_bytes

    dag = OperationDAG()
    inp = TensorInfo((1, 3, 256, 256), "f32")
    first = TensorInfo(HARD_SHAPE, "f32")
    second = TensorInfo(HARD_SHAPE, "f32")
    sink = TensorInfo((1, 8, 8, 8), "f32")
    ids = [
        dag.add_node("linalg.generic", inputs=[src], output=dst,
                     params={"edgeslice_op_id": f"op_{index}",
                             "elementwise_identity": True})
        for index, (src, dst) in enumerate(
            ((inp, first), (first, second), (second, sink))
        )
    ]
    dag.add_edge(ids[0], ids[1])
    dag.add_edge(ids[1], ids[2])

    plan = EdgeSlicePassManager(
        dag, config=EdgeSlicePassConfig(l1_cache_bytes=L1)
    ).run()["rewrite_plan"]
    anchor = next(p for p in plan if p.get("fused_producer_op_ids"))
    assert anchor["slice_dimension_2"], "expected the anchor to use a 2D tile"

    _, peak_2d = estimate_post_slice_peak_bytes(dag, plan, L1)
    one_d_plan = [
        {k: v for k, v in p.items()
         if k not in ("slice_dimension_2", "tile_size_2")}
        for p in plan
    ]
    _, peak_1d = estimate_post_slice_peak_bytes(dag, one_d_plan, L1)
    assert peak_2d < peak_1d, "the secondary tile must lower the modeled peak"
