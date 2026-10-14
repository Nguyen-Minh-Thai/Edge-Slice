"""Tests for the SRAM-residency planning objective."""

import pytest

from edgeslide.dag.graph import OperationDAG, TensorInfo
from edgeslide.transform.pass_manager import EdgeSlicePassConfig, EdgeSlicePassManager


def _dag_with_large_sink() -> tuple[OperationDAG, int]:
    dag = OperationDAG()
    inp = TensorInfo((1, 32, 64, 64), "f32")
    out = TensorInfo((1, 64, 64, 64), "f32")
    sink = dag.add_node(
        "conv_2d",
        inputs=[inp],
        output=out,
        params={"edgeslice_op_id": "sink"},
    )
    return dag, sink


@pytest.mark.cpu
def test_sram_objective_includes_a_compute_heavy_sink():
    dag, sink = _dag_with_large_sink()
    config = EdgeSlicePassConfig.for_sram(64 * 1024, double_buffer=False)

    results = EdgeSlicePassManager(dag, config=config).run()

    candidates = {entry["op_id"]: entry for entry in results["slicing_candidates"]}
    assert sink in candidates
    assert candidates[sink]["full_working_set_bytes"] > candidates[sink][
        "fast_memory_budget_bytes"
    ]
    assert any(entry["op_id"] == sink for entry in results["rewrite_plan"])
    assert all(
        entry.get("estimated_fused_working_set_bytes", entry["estimated_tile_working_set_bytes"])
        <= entry["fast_memory_budget_bytes"]
        for entry in results["rewrite_plan"]
    )
    plan = next(entry for entry in results["rewrite_plan"] if entry["op_id"] == sink)
    assert plan["estimated_tile_count"] >= 2
    assert plan["estimated_offchip_traffic_bytes_upper_bound"] > 0


@pytest.mark.cpu
def test_double_buffering_reserves_half_of_sram():
    single = EdgeSlicePassConfig.for_sram(128 * 1024, double_buffer=False)
    double = EdgeSlicePassConfig.for_sram(128 * 1024, double_buffer=True)

    assert single.tile_calculator().hardware_config.budget_levels() == [
        ("SRAM", int(128 * 1024 * 0.90))
    ]
    assert double.tile_calculator().hardware_config.budget_levels() == [
        ("SRAM", 64 * 1024)
    ]


@pytest.mark.cpu
def test_peak_memory_objective_remains_available_as_a_legacy_ablation():
    dag, sink = _dag_with_large_sink()
    config = EdgeSlicePassConfig(
        l1_cache_bytes=64 * 1024,
        planning_objective="peak_memory",
        enable_liveness_analysis=True,
    )

    results = EdgeSlicePassManager(dag, config=config).run()

    assert sink not in {entry["op_id"] for entry in results["slicing_candidates"]}
