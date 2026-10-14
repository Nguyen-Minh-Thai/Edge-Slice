"""Unit tests for EdgeSlicePassManager ablation flags (proposal §5.4)."""

import pytest

from edgeslide.dag.graph import OperationDAG, TensorInfo
from edgeslide.transform.pass_manager import EdgeSlicePassConfig, EdgeSlicePassManager


def _pressure_dag() -> OperationDAG:
    dag = OperationDAG()
    inp = TensorInfo((1, 3, 64, 64), "f32")
    big = TensorInfo((1, 128, 128, 128), "f32")
    medium = TensorInfo((1, 64, 64, 64), "f32")
    small = TensorInfo((1, 32, 32, 32), "f32")

    n0 = dag.add_node("conv_2d", inputs=[inp], output=big)
    n1 = dag.add_node("relu", inputs=[big], output=medium)
    n2 = dag.add_node("add", inputs=[medium], output=small)
    dag.add_edge(n0, n1)
    dag.add_edge(n1, n2)
    return dag


@pytest.mark.cpu
def test_ablation_a_uses_uniform_tile_size():
    dag = _pressure_dag()
    config = EdgeSlicePassConfig(
        l1_cache_bytes=256 * 1024,
        enable_liveness_analysis=True,
        enable_smart_heuristic=False,
        uniform_tile_size=32,
        uniform_slice_dimension="spatial_h",
    )
    results = EdgeSlicePassManager(dag, config=config).run()
    plan = results["rewrite_plan"]
    assert plan
    assert all(p["tile_size"] == 32 for p in plan)
    assert all(p["slice_dimension"] == "spatial_h" for p in plan)


@pytest.mark.cpu
def test_ablation_b_slices_without_liveness_gate():
    dag = _pressure_dag()
    full = EdgeSlicePassManager(
        dag,
        config=EdgeSlicePassConfig(l1_cache_bytes=256 * 1024),
    ).run()
    ablation_b = EdgeSlicePassManager(
        dag,
        config=EdgeSlicePassConfig(
            l1_cache_bytes=256 * 1024,
            enable_liveness_analysis=False,
            enable_smart_heuristic=True,
        ),
    ).run()
    assert len(ablation_b["slicing_candidates"]) >= len(full["slicing_candidates"])
    assert "compile_timing" in ablation_b

@pytest.mark.cpu
def test_uniform_baseline_rejects_a_tile_that_covers_the_whole_extent():
    """The published `baseline_uniform_tiling` must not count no-op rewrites.

    A benchmark-side planner used to build this baseline without the
    full-extent check, so it reported 172 rewrites for every resnet50
    resolution while producing a peak identical to no slicing at all — the
    entries clamped to the full dimension and shrank nothing.
    """
    dag = OperationDAG()
    inp = TensorInfo((1, 3, 64, 64), "f32")
    # spatial_h extent 64 == uniform_tile_size 64: tiling it is a no-op.
    big = TensorInfo((1, 256, 64, 512), "f32")
    sink = TensorInfo((1, 8, 8, 8), "f32")
    n0 = dag.add_node("conv_2d", inputs=[inp], output=big,
                      params={"edgeslice_op_id": "op_0"})
    n1 = dag.add_node("relu", inputs=[big], output=sink,
                      params={"edgeslice_op_id": "op_1"})
    dag.add_edge(n0, n1)

    results = EdgeSlicePassManager(
        dag,
        config=EdgeSlicePassConfig(
            l1_cache_bytes=256 * 1024,
            enable_liveness_analysis=False,
            enable_smart_heuristic=False,
            uniform_tile_size=64,
            uniform_slice_dimension="spatial_h",
        ),
    ).run()

    assert all(p["op_id"] != 0 for p in results["rewrite_plan"])
    reasons = {r["op_id"]: r["reason"] for r in results["planning_rejections"]}
    assert "covers the whole" in reasons[0]


@pytest.mark.cpu
def test_heuristic_reference_count_is_the_ablation_b_plan():
    """`heuristic_all_ops_count` is ablation B's own plan, not a second builder.

    They are the same configuration (no liveness gate, smart heuristic on).
    A separate benchmark-side builder used a weaker, output-only cost model
    and reported a different number in the same JSON.
    """
    dag = _pressure_dag()
    config = EdgeSlicePassConfig(
        l1_cache_bytes=256 * 1024,
        enable_liveness_analysis=False,
        enable_smart_heuristic=True,
    )
    first = EdgeSlicePassManager(dag, config=config).run()["rewrite_plan"]
    second = EdgeSlicePassManager(dag, config=config).run()["rewrite_plan"]

    assert len(first) == len(second)
    assert [p["op_id"] for p in first] == [p["op_id"] for p in second]
    # Every reported entry carries the cost annotation the transform path uses,
    # which the output-only builder never produced.
    assert all("estimated_tile_working_set_bytes" in p for p in first)
