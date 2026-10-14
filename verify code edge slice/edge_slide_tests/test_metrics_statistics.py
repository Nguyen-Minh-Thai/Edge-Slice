"""CPU tests for benchmark statistics and structural metrics."""

import pytest

pytest.importorskip("numpy", reason="numpy is required by benchmarks.metrics")

from benchmarks.metrics import (
    compare_paired_timings,
    numerical_equivalence,
    parse_perf_stat_csv,
    summarize_timing_samples,
)

pytestmark = pytest.mark.cpu


def test_summarize_timing_samples_reports_median_ci():
    summary = summarize_timing_samples([10.0, 12.0, 14.0, 16.0])

    assert summary["median_ms"] == 13.0
    assert summary["sample_count"] == 4
    assert len(summary["median_ci95_ms"]) == 2


def test_compare_paired_timings_uses_wilcoxon():
    # The Wilcoxon path is the only place scipy is required; skip explicitly
    # rather than failing on an environment that lacks the optional dep.
    pytest.importorskip("scipy", reason="scipy is required for the Wilcoxon test")
    comparison = compare_paired_timings(
        [10.0, 11.0, 12.0, 13.0],
        [8.0, 9.0, 10.0, 11.0],
    )

    assert comparison["median_speedup_pct"] > 0
    assert comparison["wilcoxon"]["test"] == "wilcoxon_signed_rank"
    assert comparison["wilcoxon"]["paired_count"] == 4


def test_numerical_equivalence_uses_numpy_allclose():
    result = numerical_equivalence([1.0, 2.0], [1.0, 2.00001], rtol=1e-3, atol=1e-3)

    assert result["allclose"] is True
    assert result["same_shape"] is True
    assert result["max_abs_diff"] is not None


def test_parse_perf_stat_csv_extracts_counters():
    stderr = """
    1000,,cache-references,100.00,percent
    25,,cache-misses,100.00,percent
    <not supported>,,L1-dcache-loads,100.00,percent
    """

    counters = parse_perf_stat_csv(stderr)

    assert counters["cache-references"] == 1000
    assert counters["cache-misses"] == 25
    assert counters["L1-dcache-loads"] is None
