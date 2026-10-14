"""
Liveness analysis must run once per pass-manager run, not once per consumer.

``get_peak``, ``exceeds_l1``, ``slicing_candidates`` and ``summary`` all need
the per-program-point byte totals. Without the cache in
``PeakMemoryEstimator.estimate`` each one re-runs the O(|V|.K) liveness scan,
so a single ``EdgeSlicePassManager.run()`` repeats it roughly nine times.
"""
from __future__ import annotations

import pytest

from edgeslide.dag.graph import OperationDAG, TensorInfo
from edgeslide.liveness.peak_memory import PeakMemoryEstimator
from edgeslide.transform.pass_manager import EdgeSlicePassManager

pytestmark = pytest.mark.cpu


def _chain_dag(length: int = 6) -> OperationDAG:
    dag = OperationDAG()
    previous_tensor = TensorInfo((1, 3, 128, 128), "f32")
    previous_id = None
    for index in range(length):
        output = TensorInfo((1, 32 + index, 128, 128), "f32")
        op_id = dag.add_node(
            "linalg.conv_2d_nchw_fchw",
            inputs=[previous_tensor],
            output=output,
            params={"edgeslice_op_id": f"op_{index}"},
        )
        if previous_id is not None:
            dag.add_edge(previous_id, op_id)
        previous_id, previous_tensor = op_id, output
    return dag


def _counting_estimator(dag: OperationDAG, l1_cache_bytes: int):
    """Estimator whose analyzer counts how many times analyze() is called."""
    estimator = PeakMemoryEstimator(dag, l1_cache_bytes)
    analyzer = estimator.analyzer
    original = analyzer.analyze
    calls = {"count": 0}

    def counting_analyze(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    analyzer.analyze = counting_analyze
    return estimator, calls


def test_repeated_queries_reuse_one_liveness_pass():
    estimator, calls = _counting_estimator(_chain_dag(), 64 * 1024)

    estimator.estimate()
    estimator.get_peak()
    estimator.exceeds_l1()
    estimator.slicing_candidates()
    estimator.summary()
    estimator.estimate()

    assert calls["count"] == 1, (
        f"liveness analysis ran {calls['count']} times; the estimate() cache "
        "is not being used"
    )


def test_pass_manager_run_analyses_liveness_once(monkeypatch):
    """The whole frontend run must not re-derive liveness per consumer."""
    calls = {"count": 0}
    original_analyze = None

    import edgeslide.liveness.peak_memory as peak_memory

    real_cls = peak_memory.PeakMemoryEstimator

    class CountingEstimator(real_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            nonlocal original_analyze
            original_analyze = self.analyzer.analyze

            def counting_analyze(*a, **kw):
                calls["count"] += 1
                return original_analyze(*a, **kw)

            self.analyzer.analyze = counting_analyze

    monkeypatch.setattr(
        "edgeslide.transform.pass_manager.PeakMemoryEstimator", CountingEstimator
    )

    EdgeSlicePassManager(_chain_dag(), l1_cache_bytes=64 * 1024).run()

    assert calls["count"] == 1, (
        f"EdgeSlicePassManager.run() triggered {calls['count']} liveness passes"
    )


def test_cache_is_invalidated_when_the_dag_changes():
    """A stale cache would report the old graph's memory profile."""
    dag = _chain_dag(3)
    estimator = PeakMemoryEstimator(dag, 64 * 1024)
    before = dict(estimator.estimate())

    tail = TensorInfo((1, 256, 128, 128), "f32")
    new_id = dag.add_node(
        "linalg.conv_2d_nchw_fchw",
        inputs=[dag.nodes[max(dag.nodes)].output_tensor],
        output=tail,
        params={"edgeslice_op_id": "op_tail"},
    )
    dag.add_edge(max(op for op in dag.nodes if op != new_id), new_id)

    after = estimator.estimate()
    assert after != before, "estimate() returned a stale cache after the DAG grew"
    assert new_id in after


# ---------------------------------------------------------------------------
# LivenessAnalyzer's own cache (proposal §4.8.5).
# ---------------------------------------------------------------------------

def test_liveness_cache_is_invalidated_by_a_count_preserving_mutation():
    """A shape change that keeps node and edge counts must not serve a stale sweep.

    The fingerprint used to be (node count, edge count), so a pass that swapped
    a node's output tensor for a differently shaped one got the previous
    live-set map back. PeakMemoryEstimator already guarded against this for its
    own cache; LivenessAnalyzer sits behind the same call chain and did not.
    """
    from edgeslide.dag.graph import OperationDAG, TensorInfo
    from edgeslide.liveness.analyzer import LivenessAnalyzer

    dag = OperationDAG()
    first = TensorInfo((1, 64, 128, 128))
    second = TensorInfo((1, 64, 128, 128))
    dag.add_node("linalg.generic", inputs=[first], output=second)
    dag.add_node("linalg.generic", inputs=[second], output=TensorInfo((1, 8, 8, 8)))
    dag.add_edge(0, 1)

    analyzer = LivenessAnalyzer(dag)
    before = analyzer._dag_fingerprint()
    analyzer.analyze()

    # In-place shape change: same node count, same edge count.
    dag.nodes[0].output_tensor = TensorInfo((1, 64, 8, 8))
    dag.nodes[0].output_tensors = [dag.nodes[0].output_tensor]

    assert analyzer._dag_fingerprint() != before


def test_liveness_cache_is_reused_when_the_dag_is_untouched():
    """The cache must still work, or the sweep runs on every accessor."""
    from edgeslide.dag.graph import OperationDAG, TensorInfo
    from edgeslide.liveness.analyzer import LivenessAnalyzer

    dag = OperationDAG()
    mid = TensorInfo((1, 64, 128, 128))
    dag.add_node("linalg.generic", inputs=[TensorInfo((1, 3, 128, 128))], output=mid)
    dag.add_node("linalg.generic", inputs=[mid], output=TensorInfo((1, 8, 8, 8)))
    dag.add_edge(0, 1)

    analyzer = LivenessAnalyzer(dag)
    assert analyzer._dag_fingerprint() == analyzer._dag_fingerprint()
    first = analyzer.analyze()
    assert analyzer.analyze() == first
