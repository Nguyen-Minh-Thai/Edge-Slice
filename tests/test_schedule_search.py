from __future__ import annotations

import json

import pytest

from edgeslide.dag.graph import OperationDAG, TensorInfo
from edgeslide.tiling.schedule_search import CapacityScheduleSearch, TargetAutotuner
from edgeslide.tiling.tile_size import HardwareConfig, TileSizeCalculator
from edgeslide.transform.pass_manager import EdgeSlicePassConfig, EdgeSlicePassManager

pytestmark = pytest.mark.cpu


def test_search_enumerates_only_budget_legal_schedules():
    calculator = TileSizeCalculator(
        hardware_config=HardwareConfig.for_sram(64 * 1024, double_buffer=False)
    )
    search = CapacityScheduleSearch(calculator, max_candidates=512)
    result = search.search(
        (1, 32, 64, 64),
        "linalg.generic",
        cost_kwargs={
            "dtype_size": 4,
            "op_type": "linalg.generic",
            "operand_shapes": ((1, 32, 64, 64),),
        },
    )

    assert result.candidates
    assert result.winner.tile_count > 1
    assert all(c.footprint_bytes <= c.budget_bytes for c in result.candidates)
    assert result.winner == min(
        (c for c in result.candidates if c.tile_count > 1),
        key=lambda c: c.analytical_rank,
    )


def test_target_measurement_selects_fastest_shortlisted_schedule():
    calculator = TileSizeCalculator(l1_cache_bytes=256 * 1024)
    search = CapacityScheduleSearch(calculator, shortlist_size=6)

    def measure(candidate):
        # A deterministic stand-in for a target runner: prefer the smallest
        # primary tile even when the analytical traffic rank prefers another.
        return float(candidate.tile_sizes[0])

    result = search.search(
        (1, 16, 32, 32),
        "linalg.generic",
        cost_kwargs={
            "dtype_size": 4,
            "op_type": "linalg.generic",
            "operand_shapes": ((1, 16, 32, 32),),
        },
        measure=measure,
    )

    assert result.measured
    assert result.winner.measured_latency_ms == min(
        candidate.measured_latency_ms for candidate in result.shortlist
    )


def test_autotuner_repeats_medians_and_reuses_persistent_cache(tmp_path):
    calculator = TileSizeCalculator(l1_cache_bytes=64 * 1024)
    candidate = CapacityScheduleSearch(calculator).search(
        (1, 8, 32, 32),
        "linalg.generic",
        cost_kwargs={
            "dtype_size": 4,
            "op_type": "linalg.generic",
            "operand_shapes": ((1, 8, 32, 32),),
        },
    ).winner
    calls = []
    samples = iter([99.0, 5.0, 3.0, 4.0])

    def measure(_candidate):
        value = next(samples)
        calls.append(value)
        return value

    cache = tmp_path / "autotune.json"
    tuner = TargetAutotuner(
        measure, warmups=1, trials=3, cache_path=cache, target_fingerprint="test-cpu"
    )
    assert tuner(candidate) == 4.0
    assert calls == [99.0, 5.0, 3.0, 4.0]
    assert tuner(candidate) == 4.0
    assert len(calls) == 4
    assert json.loads(cache.read_text())


def test_pass_manager_publishes_search_evidence():
    dag = OperationDAG()
    shape = (1, 32, 64, 64)
    op_id = dag.add_node(
        "linalg.generic",
        inputs=[TensorInfo(shape, "f32")],
        output=TensorInfo(shape, "f32"),
        params={"elementwise_identity": True, "edgeslice_op_id": "op_0"},
    )
    config = EdgeSlicePassConfig.for_sram(
        64 * 1024,
        double_buffer=False,
        planning_strategy="search",
        enable_fusion=False,
    )
    plan = EdgeSlicePassManager(dag, config=config).run()["rewrite_plan"]

    entry = next(item for item in plan if item["op_id"] == op_id)
    assert entry["planning_strategy"] == "search"
    assert entry["schedule_candidates_evaluated"] > 1
    assert entry["schedule_dataflow"] == "output_stationary"
    assert entry["estimated_tile_working_set_bytes"] <= entry["fast_memory_budget_bytes"]
