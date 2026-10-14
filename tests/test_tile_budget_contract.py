"""Regressions for the tile cost model's budget contract (proposal §4.8.4).

Theorem 3 is the whole reason ``TileSizeCalculator`` exists: *every* tile it
hands back fits the configured analytical cache budget, or the call raises
``TileBudgetError`` so the planner can pick another dimension. Four verified
ways that guarantee — and the numbers built on it — used to be wrong:

* a degenerate (extent <= 1) dimension returned tile 1 with no budget check;
* the non-convolution footprint counted only ``input + output``, so every
  ``matmul`` and every ``linalg.generic`` under-reported its working set;
* the chosen tile was rounded DOWN to a power of two, discarding up to half of
  a budget the search had just proven affordable;
* ``get_num_tiles`` / ``summary`` re-computed the tile *without* the cost
  metadata the planner used, so reports described a tile nobody planned.

Each test below fails against the pre-fix calculator.
"""

from __future__ import annotations

import math

import pytest

from edgeslide.dag.graph import OperationDAG, TensorInfo
from edgeslide.tiling.tile_size import (
    TileBudgetError,
    TileSizeCalculator,
    node_cost_kwargs,
)

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------
# Item 2 — Theorem 3 must hold on the degenerate (extent <= 1) path too.
# ---------------------------------------------------------------------------

def test_unit_extent_dimension_is_budget_checked_not_waved_through():
    """The exact reproduction from the paper-readiness review.

    ``compute`` used to short-circuit any dimension of extent <= 1 to tile 1
    *before* costing anything, on the reasoning that there is no smaller tile
    to search for. True, but irrelevant: "no alternative" is not "affordable".
    A (1, 64, 224, 224) f32 activation sliced on ``batch`` has exactly one
    candidate tile whose working set is 25.7 MB against a 205 KB budget — 123x
    over — and the old code returned it while reporting success, so the planner
    emitted a tiling transform for an op that can never be cache-resident on
    that axis.
    """
    calc = TileSizeCalculator(l1_cache_bytes=256 * 1024)
    shape = (1, 64, 224, 224)

    footprint = calc.estimate_tile_footprint(shape, "batch", 1).total_bytes
    assert footprint == 25_690_112
    assert footprint > 100 * calc.budget_bytes, "the reproduction lost its teeth"

    with pytest.raises(TileBudgetError) as excinfo:
        calc.compute(shape, "batch")

    # The caller has to be able to see *why* there is no retry on this axis.
    message = str(excinfo.value)
    assert "extent 1 (<= 1)" in message, message
    assert "no smaller tile exists" in message, message


def test_a_unit_extent_dimension_that_does_fit_is_still_accepted():
    """The check is a budget check, not a ban on degenerate dimensions.

    Guards against "fix the raise by always raising": a batch of 1 whose whole
    sample is cache-resident is a perfectly good (if pointless) tile, and the
    2D cascade relies on the distinction.
    """
    calc = TileSizeCalculator(l1_cache_bytes=1024 * 1024)
    decision = calc.compute_detailed((1, 8, 16, 16), "batch")

    assert decision.tile_size == 1
    assert decision.footprint_bytes <= decision.budget_bytes


@pytest.mark.parametrize("slice_dim", ["batch", "channel", "spatial_h", "spatial_w"])
def test_every_accepted_tile_fits_the_budget_on_every_axis(slice_dim):
    """Theorem 3 as a property, over a shape with a degenerate axis in it."""
    calc = TileSizeCalculator(l1_cache_bytes=512 * 1024)
    shape = (1, 128, 96, 96)
    try:
        decision = calc.compute_detailed(shape, slice_dim)
    except TileBudgetError:
        return  # A refusal is the other legal outcome.
    assert decision.footprint_bytes <= decision.budget_bytes
    assert 0.0 < decision.utilization <= 1.0


# ---------------------------------------------------------------------------
# Item 3 — the non-conv footprint has to see every operand, and the halo.
# ---------------------------------------------------------------------------

# C = A @ B with a non-square K, so an operand mistakenly matched by extent
# would be visible: A is 2 MB, B is 2 MB, C is 4 MB.
MM_OUT = (1024, 1024)
MM_LHS = (1024, 512)
MM_RHS = (512, 1024)


def test_matmul_footprint_counts_the_whole_second_operand():
    """An M-tile of ``C = A @ B`` reads *all* of B, because K is a reduction.

    The old model counted ``input + output`` with both shaped like the output,
    so B — the single largest term in a matmul's working set — was simply
    absent. It reported 256 KB for a tile whose real working set is 2.2 MB.
    """
    calc = TileSizeCalculator(l1_cache_bytes=8 * 1024 * 1024)
    footprint = calc.estimate_tile_footprint(
        MM_OUT, "M", 32, op_type="linalg.matmul", operand_shapes=(MM_LHS, MM_RHS)
    )

    rhs_bytes = math.prod(MM_RHS) * 4
    assert footprint.operand_bytes == rhs_bytes, "all of B is live for any M-tile"
    # A's M axis *is* tiled, so it shrinks; only K stays whole.
    assert footprint.input_bytes == 32 * 512 * 4
    assert footprint.output_bytes == 32 * 1024 * 4
    assert footprint.total_bytes == (
        footprint.input_bytes + footprint.output_bytes + footprint.operand_bytes
    )
    # What the pre-fix model claimed for the same tile.
    assert footprint.total_bytes > 8 * (32 * 1024 * 4)


def test_matmul_tile_that_cannot_hold_the_second_operand_is_refused():
    """B alone busts this budget, so no M-tile is cache-resident at all.

    Pre-fix this returned tile 32 and reported a 262 KB working set against a
    410 KB budget — a plan the hardware cannot honour, admitted as compliant.
    """
    calc = TileSizeCalculator(l1_cache_bytes=512 * 1024)
    assert math.prod(MM_RHS) * 4 > calc.budget_bytes

    with pytest.raises(TileBudgetError):
        calc.compute(MM_OUT, "M", op_type="linalg.matmul",
                     operand_shapes=(MM_LHS, MM_RHS))


def test_matmul_operands_are_resolved_positionally_not_by_extent():
    """A square matmul is where extent-matching would silently halve the model.

    With K == M == N every operand axis "matches" the tiled output axis, so a
    rule that shrinks whatever lines up numerically would shrink B along its
    reduction axis. B is indexed by N, not M: an M-tile reads it whole.
    """
    calc = TileSizeCalculator(l1_cache_bytes=16 * 1024 * 1024)
    square = (512, 512)
    footprint = calc.estimate_tile_footprint(
        square, "M", 8, op_type="linalg.matmul", operand_shapes=(square, square)
    )
    assert footprint.operand_bytes == math.prod(square) * 4

    # Tiling N is the mirror image: now A is read whole and B shrinks.
    mirrored = calc.estimate_tile_footprint(
        square, "N", 8, op_type="linalg.matmul", operand_shapes=(square, square)
    )
    assert mirrored.input_bytes == math.prod(square) * 4
    assert mirrored.operand_bytes == 8 * 512 * 4


def test_elementwise_add_counts_both_addends():
    """Two same-shaped operands plus the output — three tiles, not two."""
    calc = TileSizeCalculator(l1_cache_bytes=512 * 1024)
    shape = (1, 32, 64, 64)
    two = calc.estimate_tile_footprint(
        shape, "spatial_h", 4, op_type="linalg.generic", operand_shapes=(shape, shape)
    )
    one = calc.estimate_tile_footprint(shape, "spatial_h", 4, op_type="linalg.generic")

    assert two.total_bytes == 3 * one.output_bytes
    assert two.total_bytes > one.total_bytes


def test_a_broadcast_operand_is_charged_at_full_extent():
    """A mismatched extent means a broadcast or a reduction on that axis.

    Either way the tile touches the operand's whole extent there, and
    over-estimating is the only direction Theorem 3 tolerates.
    """
    calc = TileSizeCalculator(l1_cache_bytes=4 * 1024 * 1024)
    output = (1, 256, 64, 64)
    bias = (1, 1, 64, 64)  # broadcast along channel
    footprint = calc.estimate_tile_footprint(
        output, "channel", 8, op_type="linalg.generic", operand_shapes=(output, bias)
    )
    assert footprint.operand_bytes == math.prod(bias) * 4


def test_a_windowed_generic_pays_for_its_halo():
    """The non-conv path had no halo term at all; pooling bodies need one."""
    calc = TileSizeCalculator(l1_cache_bytes=4 * 1024 * 1024)
    shape = (1, 32, 64, 64)
    windowed = calc.estimate_tile_footprint(
        shape, "spatial_h", 4, op_type="linalg.generic",
        operand_shapes=(shape,), kernel_shape=(3, 3),
    )
    pointwise = calc.estimate_tile_footprint(
        shape, "spatial_h", 4, op_type="linalg.generic", operand_shapes=(shape,)
    )

    assert windowed.halo_bytes > 0
    assert pointwise.halo_bytes == 0
    # 4 output rows through a 3-tall window read 6 input rows.
    assert windowed.input_bytes == pointwise.input_bytes * 6 // 4


def test_no_operand_metadata_keeps_the_historical_input_plus_output_model():
    """Callers that pass nothing must not silently lose the input term."""
    calc = TileSizeCalculator(l1_cache_bytes=512 * 1024)
    shape = (1, 32, 64, 64)
    for operands in (None, ()):
        footprint = calc.estimate_tile_footprint(
            shape, "spatial_h", 4, op_type="linalg.generic", operand_shapes=operands
        )
        assert footprint.input_bytes == footprint.output_bytes
        assert footprint.operand_bytes == 0


def test_node_cost_kwargs_describes_non_conv_operands():
    """The planner only benefits if the DAG description carries the operands."""
    dag = OperationDAG()
    matmul = dag.add_node(
        "linalg.matmul",
        inputs=[TensorInfo(MM_LHS, "f32"), TensorInfo(MM_RHS, "f32")],
        output=TensorInfo(MM_OUT, "f32"),
    )
    conv = dag.add_node(
        "linalg.conv_2d_nchw_fchw",
        inputs=[TensorInfo((1, 64, 58, 58), "f32"), TensorInfo((64, 64, 3, 3), "f32")],
        output=TensorInfo((1, 64, 56, 56), "f32"),
        params={"strides": [1, 1]},
    )

    assert node_cost_kwargs(dag.nodes[matmul])["operand_shapes"] == (MM_LHS, MM_RHS)
    # The conv path is unchanged: it describes its operands as input/weight.
    conv_kwargs = node_cost_kwargs(dag.nodes[conv])
    assert "operand_shapes" not in conv_kwargs
    assert conv_kwargs["input_shape"] == (1, 64, 58, 58)
    assert conv_kwargs["weight_shape"] == (64, 64, 3, 3)


def test_a_rank_changing_reshape_operand_is_scaled_not_charged_whole():
    """Element-preserving rank changes are exact, not unknowable.

    A collapse/expand feeding the tiled op has no axis lining up with the
    tiled output axis, but the element counts match, so a tile covering 1/n of
    the output reads 1/n of the operand. Charging the whole buffer here is not
    conservatism — it is wrong by the tile count, and it kept every reshape out
    of every fused loop.
    """
    calc = TileSizeCalculator(l1_cache_bytes=4 * 1024 * 1024)
    # `linalg.generic` has no named layout, so its dimensions are addressed
    # positionally (dimension_selector.dimension_indices falls through to
    # dim_<n>); "M"/"N" exist only for matmul-shaped ops.
    footprint = calc.estimate_tile_footprint(
        (1024, 1024), "dim_0", 64, op_type="linalg.generic",
        operand_shapes=((1, 64, 128, 128),),
    )
    # 64/1024 of the output rows reads 64/1024 of the operand's 1,048,576
    # elements = 65,536 elements = 256 KB, not the whole 4 MB buffer.
    assert footprint.input_bytes == 64 * 1024 * 4


# ---------------------------------------------------------------------------
# Item 4 — round down to a cache-line granule, and report utilization.
# ---------------------------------------------------------------------------

# 832 f32 elements per sliced row (3328 bytes, an exact multiple of a 64-byte
# line) against a 419430-byte budget: the largest feasible spatial_h tile is
# 63, i.e. the worst case for a power-of-two round-down.
GRANULE_SHAPE = (1, 1, 1000, 832)


def test_the_chosen_tile_does_not_throw_away_half_the_budget():
    calc = TileSizeCalculator(l1_cache_bytes=512 * 1024)
    granule = 64 // 4  # cache line / element size

    feasible_best = max(
        tile
        for tile in range(1, GRANULE_SHAPE[2] + 1)
        if calc.estimate_tile_footprint(GRANULE_SHAPE, "spatial_h", tile).total_bytes
        <= calc.budget_bytes
    )
    assert feasible_best == 63, "the fixture drifted; pick a new shape"

    decision = calc.compute_detailed(GRANULE_SHAPE, "spatial_h")

    assert decision.footprint_bytes <= decision.budget_bytes
    assert decision.tile_size % granule == 0, "tiles must start/end on a line"
    # A power-of-two round-down gave 32 here: 51% utilization for a tile the
    # search had proven affordable at 63. Rounding may cost at most one granule.
    assert decision.tile_size > feasible_best - granule
    assert decision.tile_size == 48
    assert decision.utilization > 0.75


def test_a_tile_smaller_than_one_granule_is_not_clamped_to_one():
    """Rounding a sub-line best down to 1 would throw away everything.

    (1, 64, 256, 256) on ``spatial_h`` is feasible only up to 3 rows against a
    512 KB budget; 3 is below the 16-element granule, so it is kept as-is.
    """
    calc = TileSizeCalculator(l1_cache_bytes=512 * 1024)
    decision = calc.compute_detailed((1, 64, 256, 256), "spatial_h")
    assert decision.tile_size == 3
    assert decision.utilization > 0.9


def test_compute_delegates_to_compute_detailed():
    calc = TileSizeCalculator(l1_cache_bytes=512 * 1024)
    assert calc.compute(GRANULE_SHAPE, "spatial_h") == (
        calc.compute_detailed(GRANULE_SHAPE, "spatial_h").tile_size
    )


def test_budget_utilization_is_reported_against_the_level_that_bound_the_tile():
    """A single-level config reports against L1, and the ratio is <= 1."""
    calc = TileSizeCalculator(l1_cache_bytes=512 * 1024)
    assert calc.budget_bytes == calc.hardware_config.l1_budget_bytes
    decision = calc.compute_detailed((1, 8, 64, 64), "spatial_h")
    assert decision.budget_level == "L1"
    assert decision.budget_bytes == calc.hardware_config.l1_budget_bytes
    assert decision.utilization == pytest.approx(
        calc.budget_utilization(decision.footprint_bytes)
    )
    assert decision.utilization <= 1.0


def test_an_l2_smaller_than_l1_widens_nothing():
    """A config whose "L2" is smaller than L1 must not become the binding rule.

    L2 exists to *widen* the feasible set — a tile too big for L1 can still be
    L2-resident and stream through L1. Requiring a tile to fit both, as this
    calculator used to, inverted that: configuring an L2 could only ever make
    admission stricter and never admitted anything L1 alone rejected.
    """
    calc = TileSizeCalculator(
        l1_cache_bytes=512 * 1024, l2_cache_bytes=128 * 1024
    )
    assert calc.budget_bytes == calc.hardware_config.l1_budget_bytes


def test_a_two_level_config_admits_an_l2_resident_tile_l1_alone_rejects():
    """The 2-level cascade, on the geometry that motivates it.

    A 64x224x224 f32 activation has no L1-resident tile at 32 KB — one row is
    64*224*4 = 57 KB before the input tile is counted. On a Pi 4 that op is not
    droppable: its tile is L2-resident and streams through L1.
    """
    from edgeslide.tiling.tile_size import HardwareConfig

    shape = (1, 64, 224, 224)
    l1_only = TileSizeCalculator(
        hardware_config=HardwareConfig(l1_cache_bytes=32 * 1024)
    )
    with pytest.raises(TileBudgetError):
        l1_only.compute(shape, "spatial_h")

    two_level = TileSizeCalculator(hardware_config=HardwareConfig.for_device("pi4"))
    decision = two_level.compute_detailed(shape, "spatial_h")
    assert decision.budget_level == "L2"
    assert decision.footprint_bytes > two_level.l1_budget_bytes
    assert decision.footprint_bytes <= two_level.budget_bytes
    assert decision.utilization <= 1.0


def test_the_device_table_records_real_cache_geometry():
    """No shipping edge CPU has an L1D above 64 KB; the table must reflect that."""
    from edgeslide.tiling.tile_size import EDGE_CACHE_LEVELS, HardwareConfig

    for device, spec in EDGE_CACHE_LEVELS.items():
        if device == "generic_256k":
            continue  # the legacy budget knob, deliberately not an L1D
        assert spec["l1_cache_bytes"] <= 64 * 1024, f"{device}: implausible L1D"
        assert spec["l2_cache_bytes"] > spec["l1_cache_bytes"]
        config = HardwareConfig.for_device(device)
        assert [name for name, _ in config.budget_levels()] == ["L1", "L2"]


# ---------------------------------------------------------------------------
# Item 5 — a report must describe the tile the planner actually chose.
# ---------------------------------------------------------------------------

# A 3x3 conv whose halo + weights dominate the budget: costed properly one
# output row fits; costed as bare activations, 4 rows appear to.
REPORT_OUT = (1, 64, 56, 56)
REPORT_COST = {
    "input_shape": (1, 64, 58, 58),
    "weight_shape": (64, 64, 3, 3),
    "kernel_shape": (3, 3),
    "stride": (1, 1),
}
REPORT_OP = "linalg.conv_2d_nchw_fchw"


def test_get_num_tiles_uses_the_cost_model_the_planner_used():
    calc = TileSizeCalculator(l1_cache_bytes=256 * 1024)

    planned = calc.compute(REPORT_OUT, "spatial_h", op_type=REPORT_OP, **REPORT_COST)
    assert planned == 1, "fixture drifted; the halo/weight terms must bind"

    reported = calc.get_num_tiles(REPORT_OUT, "spatial_h", REPORT_OP, **REPORT_COST)
    assert reported == math.ceil(56 / planned) == 56

    # Without the metadata the same call describes a 7-row tile in 8 tiles:
    # no weight term and no halo, so seven times more rows look affordable than
    # the planner will actually accept. That 7x gap is the silent misreport this
    # fix removes. Pinned as an assertion so the two paths stay genuinely
    # different rather than becoming accidentally equal.
    assert calc.get_num_tiles(REPORT_OUT, "spatial_h", REPORT_OP) == 8


def test_summary_reports_the_planned_tile_and_its_budget_utilization():
    calc = TileSizeCalculator(l1_cache_bytes=256 * 1024)
    decision = calc.compute_detailed(
        REPORT_OUT, "spatial_h", op_type=REPORT_OP, **REPORT_COST
    )
    text = calc.summary(REPORT_OUT, "spatial_h", REPORT_OP, **REPORT_COST)

    assert f"Tile size: {decision.tile_size}" in text
    assert "Number of tiles: 56" in text
    assert f"Tile working set: {decision.footprint_bytes} bytes" in text
    assert "Budget utilization:" in text
    assert f"{decision.budget_bytes} bytes" in text


def test_reporting_apis_accept_exactly_what_node_cost_kwargs_produces():
    """The planner's own vocabulary has to be forwardable verbatim.

    ``EdgeSlicePassManager`` sizes every tile with ``node_cost_kwargs(node)``;
    if the reporting APIs cannot take the same mapping, threading it through is
    something every caller has to hand-translate — which is how the two
    diverged in the first place.
    """
    dag = OperationDAG()
    conv = dag.add_node(
        REPORT_OP,
        inputs=[
            TensorInfo(REPORT_COST["input_shape"], "f32"),
            TensorInfo(REPORT_COST["weight_shape"], "f32"),
        ],
        output=TensorInfo(REPORT_OUT, "f32"),
        params={"strides": [1, 1]},
    )
    generic = dag.add_node(
        "linalg.generic",
        inputs=[TensorInfo((1, 32, 64, 64), "f32"), TensorInfo((1, 32, 64, 64), "f32")],
        output=TensorInfo((1, 32, 64, 64), "f32"),
        params={"elementwise_identity": True},
    )
    calc = TileSizeCalculator(l1_cache_bytes=256 * 1024)

    for op_id in (conv, generic):
        node = dag.nodes[op_id]
        cost = node_cost_kwargs(node)
        shape = node.output_tensor.shape
        planned = calc.compute(shape, "spatial_h", **cost)
        assert calc.get_num_tiles(shape, "spatial_h", **cost) == math.ceil(
            shape[2] / planned
        )
        assert f"Tile size: {planned}" in calc.summary(shape, "spatial_h", **cost)
