"""The three-number memory contract and the anchor-credit correction.

Mutation-style: each assertion fails if the behaviour it names is reverted.
The anchor-credit tests in particular guard the correction that took the
reported ResNet-50 reduction from 7.6% to 0.0% — reverting it silently
restores a headline number the rewrite does not produce.
"""

from __future__ import annotations

import pytest

from edgeslide.dag.graph import OperationDAG, TensorInfo
from edgeslide.liveness.peak_memory import PeakMemoryEstimator

pytest.importorskip("numpy", reason="numpy is required by benchmarks.metrics")

from benchmarks.metrics import (  # noqa: E402
    anchor_output_shrinks,
    estimate_post_slice_peak_bytes,
)
from benchmarks.physical import (  # noqa: E402
    CellMeasurement,
    measure_peak_rss,
    parse_cachegrind_summary,
    predictor_error,
    summarize_predictor_accuracy,
)

pytestmark = pytest.mark.cpu

L1 = 256 * 1024


BIG = (1, 256, 256, 256)
BIG_BYTES = 256 * 256 * 256 * 4


def _chain_dag() -> OperationDAG:
    """producer -> anchor -> sink, all rank-4, so tiling has somewhere to go.

    The *same* TensorInfo object is passed as a producer's output and its
    consumer's input, which is what DAGParser does. Building a fresh TensorInfo
    per edge instead makes liveness see two distinct buffers of the same shape
    and doubles the reported peak — a fixture artifact, not a property of the
    estimator.
    """
    dag = OperationDAG()
    source = TensorInfo((1, 3, 256, 256))
    first = TensorInfo(BIG)
    second = TensorInfo(BIG)
    sink = TensorInfo((1, 8, 8, 8))
    params = {"elementwise_identity": True}
    dag.add_node("linalg.generic", inputs=[source], output=first, params=params)
    dag.add_node("linalg.generic", inputs=[first], output=second, params=params)
    dag.add_node("linalg.generic", inputs=[second], output=sink, params=params)
    dag.add_edge(0, 1)
    dag.add_edge(1, 2)
    return dag


# ---------------------------------------------------------------------------
# anchor_output_shrinks
# ---------------------------------------------------------------------------

def test_an_anchor_with_a_consumer_outside_the_loop_does_not_shrink():
    """The default case, and the one the old estimator got wrong.

    tile_using_forall writes each tile into a full-size destination via
    tensor.parallel_insert_slice. The destination is full extent before the
    loop and full extent after it, so the anchor's own buffer is exactly as
    large as it was untiled.
    """
    dag = _chain_dag()
    assert not anchor_output_shrinks(dag, 1, frozenset({1, 0}))


def test_an_anchor_whose_every_consumer_is_in_the_loop_does_shrink():
    dag = _chain_dag()
    # Op 1's only consumer is op 2; put op 2 in the same loop.
    assert anchor_output_shrinks(dag, 1, frozenset({1, 0, 2}))


def test_an_anchor_with_no_consumers_is_a_function_result_and_does_not_shrink():
    dag = _chain_dag()
    assert not anchor_output_shrinks(dag, 2, frozenset({2}))


# ---------------------------------------------------------------------------
# The estimator itself
# ---------------------------------------------------------------------------

def _plan(op_id: int, shape, tile: int, fused=()):
    return [{
        "op_id": op_id,
        "op_type": "linalg.generic",
        "slice_dimension": "channel",
        "tile_size": tile,
        "original_shape": list(shape),
        "fused_producer_op_ids": list(fused),
    }]


def test_tiling_an_anchor_alone_reports_no_reduction():
    """Because it produces none. This is the correction, stated directly."""
    dag = _chain_dag()
    est = PeakMemoryEstimator(dag, L1)
    _, baseline = est.get_peak()
    _, sliced = estimate_post_slice_peak_bytes(
        dag, _plan(1, BIG, 8), L1, estimator=est
    )
    assert sliced == baseline


def test_the_old_behaviour_is_still_reachable_for_reporting_the_delta():
    dag = _chain_dag()
    est = PeakMemoryEstimator(dag, L1)
    _, baseline = est.get_peak()
    _, inflated = estimate_post_slice_peak_bytes(
        dag, _plan(1, BIG, 8), L1,
        estimator=est, credit_anchor_outputs=True,
    )
    assert inflated < baseline, "the opt-in flag must restore the old credit"


def test_a_fused_producer_is_credited_everywhere_it_is_live():
    """Not only at its own program point — which is the second half of the fix.

    The peak is at the anchor, where the producer's buffer is still live. A
    credit applied only at the producer's own definition point leaves it
    counted at full size exactly where it matters.
    """
    dag = _chain_dag()
    est = PeakMemoryEstimator(dag, L1)
    _, baseline = est.get_peak()
    _, sliced = estimate_post_slice_peak_bytes(
        dag, _plan(1, BIG, 8, fused=(0,)), L1, estimator=est
    )
    assert sliced < baseline
    # Op 0's 64 MB buffer is what disappears; the anchor's stays.
    assert sliced == baseline - BIG_BYTES + 8 * 256 * 256 * 4


def test_an_override_may_only_shrink_a_buffer():
    """A "credit" that enlarges a buffer is not a credit and must be ignored."""
    dag = _chain_dag()
    est = PeakMemoryEstimator(dag, L1)
    baseline = est.estimate()
    enlarged = est.estimate_with_output_overrides({0: 1 << 40})
    assert enlarged == baseline


# ---------------------------------------------------------------------------
# Predictor accuracy
# ---------------------------------------------------------------------------

def test_predictor_error_is_signed_and_none_without_a_measurement():
    assert predictor_error(150, 100) == 50.0
    assert predictor_error(50, 100) == -50.0
    assert predictor_error(100, 0) is None, "no measurement is not zero error"


def test_predictor_summary_reports_spread_not_just_a_mean():
    summary = summarize_predictor_accuracy([(100, 90), (200, 150), (50, 60), (300, 280)])
    assert summary["sample_count"] == 4
    assert summary["overestimates"] == 3
    assert summary["underestimates"] == 1
    assert summary["max_abs_rel_error_pct"] >= summary["median_abs_rel_error_pct"]
    assert 0.0 < summary["pearson_r"] <= 1.0


def test_predictor_summary_refuses_a_correlation_it_cannot_compute():
    summary = summarize_predictor_accuracy([(100, 50), (100, 50), (100, 50)])
    assert summary["pearson_r"] is None
    assert "constant" in summary["correlation_note"]


def test_predictor_summary_needs_more_than_one_usable_cell():
    assert "reason" in summarize_predictor_accuracy([(100, 0)])


# ---------------------------------------------------------------------------
# RSS and cachegrind plumbing
# ---------------------------------------------------------------------------

def test_measured_rss_is_positive_for_a_real_process():
    result = measure_peak_rss(["/bin/sh", "-c", "exit 0"])
    assert result.peak_rss_kb > 0
    assert result.exit_code == 0
    assert result.method != "unavailable"


def test_measured_rss_grows_with_the_allocation():
    small = measure_peak_rss(["/bin/sh", "-c", "exit 0"])
    big = measure_peak_rss(
        ["/bin/sh", "-c", "a=$(head -c 20000000 /dev/zero | base64); echo ${#a} >/dev/null"]
    )
    assert big.peak_rss_kb > small.peak_rss_kb


def test_cachegrind_summary_parses_the_miss_counters():
    text = """
==1== I   refs:      1,000,000
==1== I1  misses:        1,234
==1== D1  misses:       56,789
==1== LLd misses:        4,321
==1== LL  misses:        9,999
"""
    counters = parse_cachegrind_summary(text)
    assert counters["D1"] == 56789
    assert counters["LL"] == 9999
    assert counters["I1"] == 1234


def test_no_overrides_reproduces_the_baseline_sweep_exactly():
    """The override path must not be a second, subtly different memory model.

    It reimplements the per-point summation so it can substitute sizes, and the
    aliasing/`tensor.empty`/destination-passing rules it shares with
    ``estimate()`` are exactly what kept the baseline and post-rewrite arms
    comparable. An empty override set is the identity check on that.
    """
    estimator = PeakMemoryEstimator(_chain_dag(), L1)
    assert estimator.estimate_with_output_overrides({}) == estimator.estimate()


def test_an_override_for_an_unknown_op_is_ignored_not_an_error():
    """A plan may name an op a later pass removed; that must not crash a run."""
    estimator = PeakMemoryEstimator(_chain_dag(), L1)
    assert estimator.estimate_with_output_overrides({9999: 1}) == estimator.estimate()


def test_rss_dilution_factor_reports_how_much_wider_rss_scope_is() -> None:
    """RSS covers weights; the bufferized peak does not. Quantify the gap.

    2 MB of activations beside 46 MB of weight globals: an X% reduction in the
    peak can move RSS by at most X/24.
    """
    cell = CellMeasurement(
        arm="untiled",
        bufferized_peak_bytes=2_000_000,
        attribution_totals={"constant_global_bytes": 46_000_000},
    )
    assert cell.rss_dilution_factor() == 24.0
    assert cell.to_dict()["weight_global_bytes"] == 46_000_000
    assert cell.to_dict()["rss_dilution_factor"] == 24.0


def test_rss_dilution_factor_is_none_when_it_cannot_be_computed() -> None:
    """No peak or no weight count means unknown — not 1.0.

    Returning 1.0 would assert that the two scopes agree, which is the claim
    the field exists to stop anyone making by accident.
    """
    no_attribution = CellMeasurement(arm="untiled", bufferized_peak_bytes=2_000_000)
    assert no_attribution.rss_dilution_factor() is None

    no_peak = CellMeasurement(
        arm="untiled", attribution_totals={"constant_global_bytes": 46_000_000}
    )
    assert no_peak.rss_dilution_factor() is None
