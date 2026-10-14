"""Tests for budget-bounded producer fusion (proposal H1).

Greedy fusion pulls every fusable producer into the anchor's ``scf.forall``.
That is wrong when the producer's tile is not cache-resident there: a conv
fused into a *channel*-tiled loop needs its whole input for every tile, since
the reduction runs over all input channels — measured on resnet50@1024, one
such group's working set was 527x the L1 budget. Admission is therefore capped
by the budget, and rejected producers become anchors of their own.
"""

from __future__ import annotations

import pytest

from edgeslide.dag.graph import OperationDAG, TensorInfo
from edgeslide.transform.pass_manager import EdgeSlicePassConfig, EdgeSlicePassManager
from edgeslide.transform.rewrite import SlicedRewrite

pytestmark = pytest.mark.cpu

L1 = 256 * 1024
BUDGET = int(L1 * 0.8)


@pytest.fixture
def conv_into_channel_loop_dag():
    """Chain whose conv producer cannot be cache-resident in the anchor's loop.

    The conv's output has more channels than rows, so the anchor tiles on
    ``channel``; a channel tile of a conv still reads every input channel, so
    the producer's tile is the entire input.
    """
    dag = OperationDAG()
    image = TensorInfo((1, 256, 256, 256), "f32")
    weights = TensorInfo((512, 256, 1, 1), "f32")
    conv_out = TensorInfo((1, 512, 128, 128), "f32")
    relu_out = TensorInfo((1, 512, 128, 128), "f32")
    sink = TensorInfo((1, 8, 8, 8), "f32")

    conv = dag.add_node(
        "linalg.conv_2d_nchw_fchw",
        inputs=[image, weights],
        output=conv_out,
        params={"edgeslice_op_id": "op_conv", "strides": [2, 2]},
    )
    relu = dag.add_node(
        "linalg.generic",
        inputs=[conv_out],
        output=relu_out,
        params={"edgeslice_op_id": "op_relu", "elementwise_identity": True},
    )
    tail = dag.add_node(
        "linalg.generic",
        inputs=[relu_out],
        output=sink,
        params={"edgeslice_op_id": "op_tail", "elementwise_identity": True},
    )
    dag.add_edge(conv, relu)
    dag.add_edge(relu, tail)
    return dag


def _run(dag, **config):
    return EdgeSlicePassManager(
        dag, config=EdgeSlicePassConfig(l1_cache_bytes=L1, **config)
    ).run()


def test_no_fused_group_exceeds_the_budget(conv_into_channel_loop_dag):
    results = _run(conv_into_channel_loop_dag)

    assert results["fusion_budget_warnings"] == []
    for plan in results["rewrite_plan"]:
        assert plan["estimated_fused_working_set_bytes"] <= BUDGET, plan


def test_the_oversized_producer_is_not_fused_and_is_accounted_for(
    conv_into_channel_loop_dag,
):
    """It must stay out of the loop — and never vanish without a reason.

    Fusion candidates are not the same set as plan entries: a producer the
    planner already rejected can still be pulled into a consumer's loop. When
    the budget keeps it out, it has to surface as an explicit rejection rather
    than silently disappearing from both the plan and the chains.
    """
    results = _run(conv_into_channel_loop_dag)
    plans = {p["op_id"]: p for p in results["rewrite_plan"]}
    conv_id = next(
        op_id
        for op_id, node in conv_into_channel_loop_dag.nodes.items()
        if "conv" in node.op_type
    )

    fused_anywhere = {
        producer
        for plan in results["rewrite_plan"]
        for producer in (plan.get("fused_producer_op_ids") or [])
    }
    assert conv_id not in fused_anywhere, "conv must not be fused into a channel loop"

    rejected = {r["op_id"]: r["reason"] for r in results["planning_rejections"]}
    assert conv_id in plans or conv_id in rejected
    if conv_id in rejected:
        assert "cache budget" in rejected[conv_id]


def test_disabling_the_budget_restores_greedy_fusion(conv_into_channel_loop_dag):
    """The bound is an ablation switch, and without it the warning returns."""
    bounded = _run(conv_into_channel_loop_dag)
    greedy = _run(conv_into_channel_loop_dag, enable_fusion_budget=False)

    greedy_fused = sum(
        len(p.get("fused_producer_op_ids") or []) for p in greedy["rewrite_plan"]
    )
    bounded_fused = sum(
        len(p.get("fused_producer_op_ids") or []) for p in bounded["rewrite_plan"]
    )
    assert greedy_fused > bounded_fused
    assert greedy["fusion_budget_warnings"], (
        "unbounded fusion should still report the H1 violation it creates"
    )


def test_every_absorbed_op_still_appears_in_exactly_one_chain(
    conv_into_channel_loop_dag,
):
    """The bound must not strand an op: absorbed ops are neither tiled nor fused."""
    results = _run(conv_into_channel_loop_dag)

    anchors = {p["op_id"] for p in results["rewrite_plan"]}
    fused: list[int] = [
        producer
        for plan in results["rewrite_plan"]
        for producer in (plan.get("fused_producer_op_ids") or [])
    ]
    assert len(fused) == len(set(fused)), "a producer was fused into two loops"
    assert set(results["absorbed_into_fusion"]) == set(fused)
    assert not (anchors & set(fused)), "an op is both an anchor and fused"


def test_cheap_elementwise_producers_are_still_fused():
    """The bound must not disable fusion for groups that do fit."""
    dag = OperationDAG()
    shape = (1, 64, 128, 128)
    first = TensorInfo(shape, "f32")
    second = TensorInfo(shape, "f32")
    sink = TensorInfo((1, 8, 8, 8), "f32")
    ids = [
        dag.add_node(
            "linalg.generic",
            inputs=[src],
            output=dst,
            params={"edgeslice_op_id": f"op_{index}", "elementwise_identity": True},
        )
        for index, (src, dst) in enumerate(
            ((TensorInfo((1, 3, 128, 128), "f32"), first), (first, second), (second, sink))
        )
    ]
    dag.add_edge(ids[0], ids[1])
    dag.add_edge(ids[1], ids[2])

    results = _run(dag)
    fused = [
        producer
        for plan in results["rewrite_plan"]
        for producer in (plan.get("fused_producer_op_ids") or [])
    ]
    assert fused, "elementwise chains are cache-resident and must still fuse"
    assert results["fusion_budget_warnings"] == []


@pytest.mark.cpu
def test_an_elementwise_producer_fuses_into_a_matmul_anchor():
    """The tiled axis is positional, so a producer need not share the naming.

    A rank-2 ``linalg.generic`` knows "dim_0"/"dim_1", never "M"/"N". Matching
    the anchor's dimension by name dropped every such producer, silently
    disabling fusion for the whole elementwise→matmul family (transformer and
    MLP workloads) with nothing recorded anywhere.
    """
    dag = OperationDAG()
    shape = (1024, 1024)
    source = TensorInfo(shape, "f32")
    intermediate = TensorInfo(shape, "f32")
    result = TensorInfo(shape, "f32")
    producer = dag.add_node(
        "linalg.generic",
        inputs=[source],
        output=intermediate,
        params={"edgeslice_op_id": "op_producer", "elementwise_identity": True},
    )
    anchor = dag.add_node(
        "linalg.matmul",
        inputs=[intermediate],
        output=result,
        params={"edgeslice_op_id": "op_anchor"},
    )
    dag.add_edge(producer, anchor)

    plan = [{
        "op_id": anchor,
        "op_type": "linalg.matmul",
        "slice_dimension": "M",
        "tile_size": 64,
        "original_shape": shape,
        "params": dag.nodes[anchor].params,
    }]
    rewriter = SlicedRewrite(dag, plan, l1_cache_bytes=L1)

    assert rewriter.fusion_chains[anchor] == [producer]


@pytest.mark.cpu
def test_a_producer_of_a_different_output_rank_is_not_fused():
    """A rank change means the anchor's axis names a different dimension.

    The budget would then be computed against the wrong axis, so the group
    could report compliance while the emitted loop does something else.
    """
    dag = OperationDAG()
    feature = TensorInfo((1, 64, 128, 128), "f32")
    flat = TensorInfo((1024, 1024), "f32")
    result = TensorInfo((1024, 1024), "f32")
    rank4 = dag.add_node(
        "linalg.generic",
        inputs=[TensorInfo((1, 3, 128, 128), "f32")],
        output=feature,
        params={"edgeslice_op_id": "op_rank4", "elementwise_identity": True},
    )
    reshape = dag.add_node(
        "linalg.generic",
        inputs=[feature],
        output=flat,
        params={"edgeslice_op_id": "op_reshape", "elementwise_identity": True},
    )
    anchor = dag.add_node(
        "linalg.matmul",
        inputs=[flat],
        output=result,
        params={"edgeslice_op_id": "op_anchor"},
    )
    dag.add_edge(rank4, reshape)
    dag.add_edge(reshape, anchor)

    plan = [{
        "op_id": anchor,
        "op_type": "linalg.matmul",
        "slice_dimension": "M",
        "tile_size": 64,
        "original_shape": (1024, 1024),
        "params": dag.nodes[anchor].params,
    }]
    chain = SlicedRewrite(dag, plan, l1_cache_bytes=L1).fusion_chains[anchor]

    assert reshape in chain, "the same-rank producer should still fuse"
    assert rank4 not in chain, "a rank-4 producer must not join a rank-2 loop"
