"""Tests for the benchmark estimator's tile accounting.

Every assertion here was chosen by mutation testing: each one fails when the
behaviour it names is reverted. Before they existed, deleting the fused-producer
credit or the dtype-aware element size left all 280 tests green — while the
published peak-reduction numbers moved.
"""

from __future__ import annotations

import pytest

from edgeslide.dag.graph import OperationDAG, TensorInfo
from edgeslide.transform.pass_manager import EdgeSlicePassConfig, EdgeSlicePassManager

pytest.importorskip("numpy", reason="numpy is required by benchmarks.metrics")

from benchmarks.metrics import (  # noqa: E402
    estimate_post_slice_peak_bytes,
    peak_reduction_pct,
)

pytestmark = pytest.mark.cpu

L1 = 256 * 1024


def _fused_chain_dag(dtype: str = "f32") -> OperationDAG:
    """Two same-shape ops feeding a tiny sink — the anchor fuses the producer."""
    dag = OperationDAG()
    shape = (1, 256, 256, 256)
    inp = TensorInfo((1, 3, 256, 256), dtype)
    first = TensorInfo(shape, dtype)
    second = TensorInfo(shape, dtype)
    sink = TensorInfo((1, 8, 8, 8), dtype)
    for index, (src, dst) in enumerate(
        ((inp, first), (first, second), (second, sink))
    ):
        dag.add_node(
            "linalg.generic",
            inputs=[src],
            output=dst,
            params={"edgeslice_op_id": f"op_{index}", "elementwise_identity": True},
        )
    dag.add_edge(0, 1)
    dag.add_edge(1, 2)
    return dag


def _plan_for(dag: OperationDAG):
    return EdgeSlicePassManager(
        dag, config=EdgeSlicePassConfig(l1_cache_bytes=L1)
    ).run()["rewrite_plan"]


def _without_fusion_credit(plan):
    return [
        {k: v for k, v in entry.items() if k != "fused_producer_op_ids"}
        for entry in plan
    ]


def test_fused_producers_are_credited_at_the_anchors_program_point():
    """The credit is what turns a fused plan's reported reduction from 0%.

    A producer fused into the anchor's loop is computed tile-by-tile and
    consumed immediately, so its full-size buffer never materializes.
    """
    dag = _fused_chain_dag()
    plan = _plan_for(dag)
    assert any(p.get("fused_producer_op_ids") for p in plan), "fixture must fuse"

    _, with_credit = estimate_post_slice_peak_bytes(dag, plan, L1)
    _, without_credit = estimate_post_slice_peak_bytes(
        dag, _without_fusion_credit(plan), L1
    )
    assert with_credit < without_credit

    # The symptom the credit exists to remove: a fused plan reporting no gain.
    _, baseline = estimate_post_slice_peak_bytes(dag, [], L1)
    assert peak_reduction_pct(baseline, with_credit) > 0.0


def test_an_anchor_entry_wins_over_a_fused_membership_credit():
    """An op listed as both must keep its own tile, not the anchor's."""
    dag = _fused_chain_dag()
    plan = _plan_for(dag)
    anchor = next(p for p in plan if p.get("fused_producer_op_ids"))
    fused_id = anchor["fused_producer_op_ids"][0]

    # Give the fused producer its own (much smaller) anchor entry as well.
    own_entry = dict(anchor)
    own_entry["op_id"] = fused_id
    own_entry["original_shape"] = dag.nodes[fused_id].output_tensor.shape
    own_entry["tile_size"] = 1
    own_entry.pop("fused_producer_op_ids", None)

    _, both = estimate_post_slice_peak_bytes(dag, [own_entry, *plan], L1)
    _, credit_only = estimate_post_slice_peak_bytes(dag, plan, L1)
    assert both <= credit_only


@pytest.mark.parametrize("dtype", ["f16", "i8"])
def test_tile_accounting_honours_the_tensors_dtype(dtype):
    """For one fixed plan, the reduction must not depend on the element size.

    The tile footprint used to hardcode 4 bytes/element while the baseline it
    is compared against honours the dtype, so an i8 model — the primary edge
    deployment case — had its reduction understated fourfold, and a tight tile
    reported zero reduction outright.

    The *plan* is held fixed on purpose: the planner legitimately chooses
    larger tiles for smaller dtypes (more elements fit the same budget), so
    re-planning per dtype would measure that instead of the accounting.
    """
    f32_dag = _fused_chain_dag("f32")
    plan = _plan_for(f32_dag)

    def reduction_for(dag: OperationDAG) -> float:
        _, baseline = estimate_post_slice_peak_bytes(dag, [], L1)
        _, sliced = estimate_post_slice_peak_bytes(dag, plan, L1)
        return peak_reduction_pct(baseline, sliced)

    f32_reduction = reduction_for(f32_dag)
    assert f32_reduction > 0.0
    assert reduction_for(_fused_chain_dag(dtype)) == pytest.approx(f32_reduction)
