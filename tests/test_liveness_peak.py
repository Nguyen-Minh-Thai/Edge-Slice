"""
CPU Tests for Liveness Analysis and Peak Memory Estimation (Stage 2).

Validates:
- Live range computation per tensor
- Peak memory tracking across DAG
- Identification of slicing candidates (tensors exceeding L1)

Pure Python — no torch_mlir or GPU required.
"""
from dataclasses import dataclass

import pytest

from edgeslide.liveness.analyzer import BufferizedMemoryAnalyzer, LivenessAnalyzer
from edgeslide.liveness.peak_memory import BufferizedPeakMemoryEstimator, PeakMemoryEstimator


@dataclass(frozen=True)
class _PressureDAGExpectations:
    l1_cache_bytes: int
    peak_op_id: int
    peak_memory_bytes: int
    candidate_op_ids: tuple
    candidate_op_types: tuple
    excluded_op_id: int
    excluded_op_type: str


PRESSURE_DAG_EXPECTATIONS = _PressureDAGExpectations(
    l1_cache_bytes=256 * 1024,
    peak_op_id=1,
    peak_memory_bytes=9 * 1024 * 1024,
    candidate_op_ids=(0, 1),
    candidate_op_types=("conv_big", "relu"),
    excluded_op_id=2,
    excluded_op_type="add",
)


@pytest.mark.cpu
def test_liveness_live_ranges(synthetic_dag):
    analyzer = LivenessAnalyzer(synthetic_dag)
    live_at = analyzer.analyze()

    assert len(live_at) == 3
    assert any(len(v) > 0 for v in live_at.values())


@pytest.mark.cpu
def test_peak_memory_static_dag_precise(pressure_test_dag):
    """
    Static DAG with analytically known peak op and slicing candidates.

    conv_big (8 MB) -> relu (1 MB) -> add (128 KB sink)
    Peak live memory occurs at relu when both big and medium tensors are live.
    """
    exp = PRESSURE_DAG_EXPECTATIONS
    est = PeakMemoryEstimator(pressure_test_dag, l1_cache_bytes=exp.l1_cache_bytes)

    peak_op, peak_bytes = est.get_peak()
    assert peak_op == exp.peak_op_id
    assert peak_bytes == exp.peak_memory_bytes

    memory_at = est.estimate()
    # At each program point the op's input and its output are both live: the
    # input cannot be freed until the op that reads it has produced its result.
    input_bytes = 1 * 3 * 64 * 64 * 4
    big_bytes = 1 * 128 * 128 * 128 * 4
    medium_bytes = 1 * 64 * 64 * 64 * 4
    small_bytes = 1 * 32 * 32 * 32 * 4
    assert memory_at[0] == big_bytes + input_bytes
    assert memory_at[1] == big_bytes + medium_bytes == exp.peak_memory_bytes
    assert memory_at[2] == medium_bytes + small_bytes

    exceeds = est.exceeds_l1()
    assert exceeds == [0, 1, 2]

    candidates = est.slicing_candidates()
    candidate_ids = [c["op_id"] for c in candidates]
    candidate_types = [c["op_type"] for c in candidates]

    assert candidate_ids == list(exp.candidate_op_ids)
    assert candidate_types == list(exp.candidate_op_types)

    excluded = [c for c in candidates if c["op_id"] == exp.excluded_op_id]
    assert excluded == []
    assert not pressure_test_dag.nodes[exp.excluded_op_id].consumers
    assert pressure_test_dag.nodes[exp.excluded_op_id].op_type == exp.excluded_op_type


@pytest.mark.cpu
def test_slicing_candidates_exclude_sink_without_consumers(pressure_test_dag):
    """Sink ops may exceed L1 but must not appear in slicing_candidates."""
    exp = PRESSURE_DAG_EXPECTATIONS
    est = PeakMemoryEstimator(pressure_test_dag, l1_cache_bytes=exp.l1_cache_bytes)

    assert exp.excluded_op_id in est.exceeds_l1()
    candidate_ids = {c["op_id"] for c in est.slicing_candidates()}
    assert exp.excluded_op_id not in candidate_ids


@pytest.mark.cpu
def test_estimate_cache_invalidated_by_inplace_shape_mutation(pressure_test_dag):
    """A count-preserving in-place shape change must not serve stale results.

    The memo fingerprint once covered only node/edge counts, so a pass that
    shrank a tensor in place (same graph structure) would silently reuse the
    pre-mutation sweep for every later strategy sharing the estimator.
    """
    exp = PRESSURE_DAG_EXPECTATIONS
    est = PeakMemoryEstimator(pressure_test_dag, l1_cache_bytes=exp.l1_cache_bytes)
    _, peak_before = est.get_peak()
    assert peak_before == exp.peak_memory_bytes

    big_tensor = pressure_test_dag.nodes[0].output_tensor
    big_tensor.shape = (1, 128, 16, 16)  # shrink conv_big's output in place

    _, peak_after = est.get_peak()
    assert peak_after < peak_before
    medium_bytes = 1 * 64 * 64 * 64 * 4
    shrunk_big_bytes = 1 * 128 * 16 * 16 * 4
    assert peak_after != exp.peak_memory_bytes
    assert est.estimate()[1] == shrunk_big_bytes + medium_bytes


@pytest.mark.cpu
def test_summary_does_not_crash(synthetic_dag):
    est = PeakMemoryEstimator(synthetic_dag, l1_cache_bytes=128 * 1024)
    summary = est.summary()
    assert "Peak Memory Estimation" in summary
    assert "KB" in summary

@pytest.mark.cpu
def test_bufferized_memory_analyzer_tracks_alloc_dealloc_lifetimes():
    mlir = """
    func.func @main() {
      %0 = memref.alloc() : memref<4x4xf32>
      %1 = memref.alloc() : memref<8x4xf32>
      memref.dealloc %0 : memref<4x4xf32>
      %2 = memref.alloc() : memref<2x4xf32>
      memref.dealloc %1 : memref<8x4xf32>
      memref.dealloc %2 : memref<2x4xf32>
      return
    }
    """

    report = BufferizedMemoryAnalyzer(mlir).analyze()
    peak_index, peak_bytes = report.get_peak()

    assert peak_index >= 0
    assert peak_bytes == (4 * 4 * 4) + (8 * 4 * 4)
    assert len(report.allocations) == 3

@pytest.mark.cpu
def test_bufferized_memory_analyzer_requires_bounds_for_dynamic_memrefs():
    mlir = "%0 = memref.alloc(%n) : memref<?x4xf32>"

    with pytest.raises(ValueError, match="dynamic memref dimension"):
        BufferizedMemoryAnalyzer(mlir).analyze()

    report = BufferizedMemoryAnalyzer(mlir, shape_bounds={"%0": {0: 8}}).analyze()
    assert report.get_peak()[1] == 8 * 4 * 4

@pytest.mark.cpu
def test_bufferized_peak_memory_estimator_wraps_memref_lifetimes():
    mlir = """
    func.func @main() {
      %0 = memref.alloc() : memref<4x4xf32>
      %1 = memref.alloc() : memref<2x4xf32>
      memref.dealloc %0 : memref<4x4xf32>
      memref.dealloc %1 : memref<2x4xf32>
      return
    }
    """

    estimator = BufferizedPeakMemoryEstimator(bufferized_mlir=mlir)
    assert estimator.get_peak()[1] == (4 * 4 * 4) + (2 * 4 * 4)
    assert "Bufferized MLIR Memory" in estimator.summary()


# --- Destination-passing aliasing (proposal §4.8.2 Corollary 1) ----------------
#
# Frontend IR is destination-passing: torch-mlir emits one activation as a chain
# of SSA values (`tensor.empty` -> `linalg.fill` -> `tensor.insert_slice`) that
# bufferization maps onto a single allocation. The estimator must sum over
# buffers, not values. These tests pin both directions of that rule — the
# coalescing AND the cases where coalescing would understate the peak.

_DPS_SHAPE = (1, 64, 512, 512)
_DPS_BYTES = 1 * 64 * 512 * 512 * 4


def _tensor(shape=_DPS_SHAPE):
    from edgeslide.dag.graph import TensorInfo

    return TensorInfo(shape, "f32")


@pytest.mark.cpu
def test_destination_passing_chain_counts_one_buffer():
    """empty -> fill -> insert_slice is one allocation, not three."""
    from edgeslide.dag.graph import OperationDAG

    dag = OperationDAG()
    empty_t, fill_t, insert_t = _tensor(), _tensor(), _tensor()
    src = _tensor((1, 64, 256, 256))

    empty = dag.add_node("tensor.empty", inputs=[], output=empty_t)
    fill = dag.add_node("linalg.fill", inputs=[empty_t], output=fill_t)
    insert = dag.add_node(
        "tensor.insert_slice", inputs=[src, fill_t], output=insert_t
    )
    dag.add_edge(empty, fill)
    dag.add_edge(fill, insert)

    peak_op, peak_bytes = PeakMemoryEstimator(dag, 256 * 1024).get_peak()

    # The chain contributes exactly one full-size buffer. `src` (16 MB) is live
    # at the insert, so allow it, but the chain itself must not be doubled.
    assert peak_bytes < 2 * _DPS_BYTES, (
        f"chain counted more than once: {peak_bytes} bytes for a "
        f"{_DPS_BYTES}-byte activation"
    )
    assert peak_bytes == _DPS_BYTES + src.size_bytes


@pytest.mark.cpu
def test_shared_destination_is_not_coalesced():
    """A CSE'd `tensor.empty` feeding several ops does not merge their results.

    torch-mlir emits one `tensor.empty` for N identical destinations. Only one
    consumer can bufferize in place; the rest need copies, so their results are
    genuinely distinct buffers. Coalescing them would understate the peak —
    the failure direction that a green suite would otherwise hide.
    """
    from edgeslide.dag.graph import OperationDAG

    dag = OperationDAG()
    shared = _tensor()
    results = [_tensor() for _ in range(3)]

    empty = dag.add_node("tensor.empty", inputs=[], output=shared)
    consumers = [
        dag.add_node(
            "linalg.generic",
            inputs=[shared],
            output=out,
            params={"elementwise_identity": True},
        )
        for out in results
    ]
    for consumer in consumers:
        dag.add_edge(empty, consumer)
    # Keep all three results live simultaneously.
    sink = dag.add_node("linalg.generic", inputs=results, output=_tensor((1, 8, 8, 8)))
    for consumer in consumers:
        dag.add_edge(consumer, sink)

    _, peak_bytes = PeakMemoryEstimator(dag, 256 * 1024).get_peak()

    assert peak_bytes >= 3 * _DPS_BYTES, (
        "three separately-materialized results were merged into one buffer; "
        f"got {peak_bytes}, expected at least {3 * _DPS_BYTES}"
    )


@pytest.mark.cpu
def test_matching_ins_operand_is_not_treated_as_destination():
    """A unary elementwise reads an `ins` of the result's exact type.

    Type-matching alone cannot distinguish that from an `outs`; only
    provenance (the operand originating at a `tensor.empty`) can. Aliasing
    here would merge two real buffers.
    """
    from edgeslide.dag.graph import OperationDAG

    dag = OperationDAG()
    first, second = _tensor(), _tensor()

    producer = dag.add_node(
        "linalg.generic",
        inputs=[_tensor((1, 3, 256, 256))],
        output=first,
        params={"elementwise_identity": True},
    )
    consumer = dag.add_node(
        "linalg.generic",
        inputs=[first],
        output=second,
        params={"elementwise_identity": True},
    )
    sink = dag.add_node("linalg.generic", inputs=[second], output=_tensor((1, 8, 8, 8)))
    # `producer` deliberately has exactly ONE consumer, so the single-consumer
    # guard cannot be what rejects this merge — provenance has to. Both buffers
    # are live at `consumer`: `first` until its last use there, `second` on
    # production.
    dag.add_edge(producer, consumer)
    dag.add_edge(consumer, sink)
    assert len(dag.nodes[producer].consumers) == 1

    _, peak_bytes = PeakMemoryEstimator(dag, 256 * 1024).get_peak()

    assert peak_bytes >= 2 * _DPS_BYTES, (
        "an `ins` operand was mistaken for a destination and its buffer merged "
        f"with the result; got {peak_bytes}, expected at least {2 * _DPS_BYTES}"
    )
