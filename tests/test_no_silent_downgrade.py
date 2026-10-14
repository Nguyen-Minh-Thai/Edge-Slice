"""A requested compilation mode is delivered or refused — never downgraded.

Every entry here corresponds to a downgrade that shipped, produced a working
binary, and reported ordinary-looking numbers while measuring something other
than what was asked for. That combination is what makes a silent fallback worse
than a crash: the run succeeds, so nobody looks.

  - vectorization was requested, the vectorized IR did not parse, and the
    scalar pipeline compiled instead — for every model, for as long as the
    feature had existed;
  - real buffer deallocation was unavailable, the legacy pass was accepted and
    inserted nothing, and the payload leaked every buffer while still
    reporting a peak;
  - torchvision weights were unavailable, a hand-written topology loaded a
    checkpoint into itself, and failing that, random initialisation ran behind
    an environment variable. Shape-derived metrics survive random weights, so
    the numbers stayed plausible.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest

pytest.importorskip("torch", reason="PyTorch is required")

from edgeslide.mlir_transform.errors import (  # noqa: E402
    DeallocationUnavailableError,
    MLIRVectorizationError,
    SilentDowngradeError,
)
from edgeslide.mlir_transform.lowering import (  # noqa: E402
    SlicedPayloadLoweringConfig,
)

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------
# The defaults are what matter: a downgrade must require asking for it.
# ---------------------------------------------------------------------------

def test_scalar_fallback_is_off_by_default():
    assert SlicedPayloadLoweringConfig().allow_scalar_fallback is False


def test_leaking_deallocation_is_off_by_default():
    assert SlicedPayloadLoweringConfig().allow_leaking_deallocation is False


def test_both_downgrades_share_a_base_class():
    """So a caller can catch "I did not get what I asked for" as one thing."""
    assert issubclass(MLIRVectorizationError, SilentDowngradeError)
    assert issubclass(DeallocationUnavailableError, SilentDowngradeError)


# ---------------------------------------------------------------------------
# Vectorization: asking and getting scalar must raise.
# ---------------------------------------------------------------------------

def test_a_failed_vectorized_lowering_raises_rather_than_compiling_scalar(
    tmp_path, monkeypatch
):
    from edgeslide.mlir_transform import lowering as low

    config = SlicedPayloadLoweringConfig(
        output_dir=str(tmp_path),
        entry_func="forward",
        enable_stride1_vectorization=True,
    )
    engine = low.SlicedPayloadLowering(config)

    monkeypatch.setattr(engine, "_run_stage1", lambda *a, **k: "module { }")
    monkeypatch.setattr(low, "outline_stride1_convs", lambda text: (text, 3))
    monkeypatch.setattr(low, "flatten_nested_module", lambda text: text)

    def _explode(*args, **kwargs):
        raise low.MLIRTransformError(["mlir-opt"], 1, "", "error: expected foo")

    monkeypatch.setattr(engine, "_run_stage2", _explode)

    with pytest.raises(MLIRVectorizationError) as caught:
        engine._lower_with_vectorization_attempt(
            "mlir-opt", tmp_path / "in.mlir", tmp_path / "out.mlir"
        )
    message = str(caught.value)
    # A parse failure is a defect in what this repo generated, and must be
    # named as one rather than blamed on the payload.
    assert "did not PARSE" in message
    assert "outline_stride1_convs" in message
    assert "allow_scalar_fallback" in message


def test_the_opt_in_still_permits_the_downgrade(tmp_path, monkeypatch):
    """The escape hatch has to work, or someone will delete the check instead."""
    from edgeslide.mlir_transform import lowering as low

    config = SlicedPayloadLoweringConfig(
        output_dir=str(tmp_path),
        entry_func="forward",
        enable_stride1_vectorization=True,
        allow_scalar_fallback=True,
    )
    engine = low.SlicedPayloadLowering(config)

    monkeypatch.setattr(engine, "_run_stage1", lambda *a, **k: "module { }")
    monkeypatch.setattr(low, "outline_stride1_convs", lambda text: (text, 3))
    monkeypatch.setattr(low, "flatten_nested_module", lambda text: text)

    calls = []

    def _stage2(tool, text, out, passes, label):
        calls.append(label)
        if "vectorized" in label:
            raise low.MLIRTransformError(["mlir-opt"], 1, "", "error: expected foo")

    monkeypatch.setattr(engine, "_run_stage2", _stage2)
    engine._lower_with_vectorization_attempt(
        "mlir-opt", tmp_path / "in.mlir", tmp_path / "out.mlir"
    )
    assert any("vectorized" in label for label in calls)
    assert any("vectorized" not in label for label in calls), "no scalar retry"


# ---------------------------------------------------------------------------
# Workloads: real pretrained weights or an exception.
# ---------------------------------------------------------------------------

def test_no_loader_can_return_random_weights():
    """The removed tiers, stated as a property of the module's surface.

    A reimplemented topology or a random-init path reachable from `load_model`
    is what made a synthetic run indistinguishable from a real one.
    """
    from benchmarks import workloads

    assert not hasattr(workloads, "_ResNet50Fallback")
    assert not hasattr(workloads, "MobileNetV2Workload")
    assert not hasattr(workloads, "EfficientNetB0Workload")
    assert not hasattr(workloads, "UNetWorkload")
    assert not hasattr(workloads, "_load_pretrained_topology")

    source = (
        __import__("pathlib").Path(workloads.__file__).read_text()
    )
    assert "EDGESLIDE_ALLOW_RANDOM_WEIGHTS" not in source, (
        "the random-weight opt-in is back; a benchmark run can silently "
        "become synthetic again"
    )


def test_the_unet_role_is_filled_by_published_weights_not_a_topology():
    """The hand-written U-Net had no matching checkpoint, so every number
    measured on it came from random weights. The role is now filled by a
    published pretrained U-Net, so `unet` is back — but as real weights.

    Asserted through the loader rather than the name, because the name is
    exactly what stayed the same across that change.
    """
    from benchmarks import workloads

    assert "unet" in workloads.SUPPORTED_MODELS
    source = pathlib.Path(workloads.__file__).read_text()
    assert "class UNetWorkload" not in source

    loader_source = inspect.getsource(workloads.load_unet)
    # The loader must reach published weights, not define a topology.
    assert "pretrained=True" in loader_source
    assert "nn.Conv2d" not in loader_source, (
        "the loader is building an architecture again"
    )


def test_third_party_code_execution_is_recorded_in_every_result():
    """U-Net loads via torch.hub, which executes hubconf.py from GitHub.

    That is a deliberate choice, and the reliability requirement on it is not
    that it be avoided but that it be *visible in the numbers* — a reader of a
    result file must not have to read the loader to find out that a model came
    through a third-party repository.
    """
    from benchmarks.workloads import load_model, weights_provenance

    unet = weights_provenance(load_model("unet"), digest=False)
    assert unet["hub_reference"] == "mateuszbuda/brain-segmentation-pytorch#unet"
    assert unet["executes_third_party_code"] is True
    assert unet["checkpoint_url"].startswith("https://")

    # A torchvision model must not be flagged, or the flag means nothing.
    resnet = weights_provenance(load_model("resnet18"), digest=False)
    assert resnet["hub_reference"] is None
    assert resnet["executes_third_party_code"] is False


# ---------------------------------------------------------------------------
# Reported numbers: measurement is the default, estimate is opt-in.
# ---------------------------------------------------------------------------

def test_the_evaluation_runner_measures_by_default():
    import argparse
    import inspect

    from benchmarks import run_evaluation

    source = inspect.getsource(run_evaluation.main)
    assert "--no-physical" in source, "opting out must be explicit"
    assert '"--physical"' not in source, (
        "measurement is the default; a --physical opt-*in* means an "
        "analysis-only run is what someone gets by forgetting a flag"
    )

    # And the default actually resolves to True.
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-physical", dest="physical", action="store_false")
    parser.set_defaults(physical=True)
    assert parser.parse_args([]).physical is True
    assert parser.parse_args(["--no-physical"]).physical is False


def test_no_reference_implementation_runs_by_default():
    from benchmarks.run_sixcam_sconv_comparison import DEFAULT_METHODS

    assert "edgeslice" not in DEFAULT_METHODS
    assert "sconv" not in DEFAULT_METHODS


# ---------------------------------------------------------------------------
# An unmeasured cell must not contribute its estimate to the matrix.
# ---------------------------------------------------------------------------

def _cell(**physical):
    return {"arms": [], **physical}

def _measured_arm(name="untiled", **overrides):
    """A fully measured arm: all three numbers present, no errors."""
    arm = {
        "arm": name,
        "errors": [],
        "bufferized_peak_bytes": 1 << 20,
        "measured_rss": {"peak_rss_kb": 4096},
    }
    arm.update(overrides)
    return arm


def _complete_cell(**overrides):
    """All three arms measured. Tests override one arm to isolate one defect."""
    from benchmarks.alloc_attribution import ARMS

    arms = [_measured_arm(name) for name in ARMS]
    if overrides:
        arms[0].update(overrides)
    return {"arms": arms}




def test_a_cell_whose_measurement_failed_is_refused():
    from benchmarks.run_evaluation import (
        MeasurementUnavailable,
        _require_measured_cell,
    )

    with pytest.raises(MeasurementUnavailable, match="physical measurement failed"):
        _require_measured_cell(
            "resnet50", (1, 3, 64, 64), _cell(error="clang not found")
        )


def test_a_cell_with_no_arms_is_refused():
    from benchmarks.run_evaluation import (
        MeasurementUnavailable,
        _require_measured_cell,
    )

    with pytest.raises(MeasurementUnavailable, match="no arms were measured"):
        _require_measured_cell("resnet50", (1, 3, 64, 64), _cell())


def test_a_cell_with_a_broken_arm_is_refused():
    """One arm that would not compile makes the comparison meaningless.

    The reported reduction is untiled-vs-tiled; if either side is missing there
    is nothing to compare, and reporting the analytical estimate in its place
    would blend a measured column with an unmeasured one.
    """
    from benchmarks.run_evaluation import (
        MeasurementUnavailable,
        _require_measured_cell,
    )

    physical = _complete_cell(errors=["compile: no"], bufferized_peak_bytes=0)
    with pytest.raises(MeasurementUnavailable, match="unmeasured"):
        _require_measured_cell("resnet50", (1, 3, 64, 64), physical)


def test_a_fully_measured_cell_is_accepted():
    from benchmarks.run_evaluation import _require_measured_cell

    _require_measured_cell("resnet50", (1, 3, 64, 64), _complete_cell())


def test_the_sweep_records_an_unmeasured_cell_as_an_error_not_a_result():
    """It must land in `errors`, and say the estimate was thrown away."""
    import inspect

    from benchmarks import run_evaluation

    source = inspect.getsource(run_evaluation.run_evaluation)
    assert "MeasurementUnavailable" in source
    assert "analytical_estimate_discarded" in source
    assert "results.append" in source
    # The unmeasured branch must not reach results.append.
    unmeasured = source[source.index("except MeasurementUnavailable"):]
    unmeasured = unmeasured[: unmeasured.index("except Exception")]
    assert "results.append" not in unmeasured


def test_a_cell_with_no_measured_rss_is_refused():
    """`measurement_scope` promises three numbers; a zero is the absence of one.

    RSS comes from an OS high-water mark, which is never legitimately zero for
    a process that ran — so a zero means the measurement did not happen, and
    the cell would otherwise be reported as fully measured.
    """
    from benchmarks.run_evaluation import (
        MeasurementUnavailable,
        _require_measured_cell,
    )

    physical = _complete_cell(
        measured_rss={"peak_rss_kb": 0, "error": "gnu time unavailable"}
    )
    with pytest.raises(MeasurementUnavailable, match="no measured RSS"):
        _require_measured_cell("resnet50", (1, 3, 64, 64), physical)


def test_requested_cachegrind_counters_must_actually_arrive():
    """`--cachegrind` asked for cache traffic; a missing counter is not a pass.

    Cachegrind fails quietly in exactly the ways that matter — valgrind absent,
    the simulation timing out, or `--cache-sim=yes` missing so the summary
    carries no D1 line at all. Treating that as fine would let a matrix
    advertise cache-traffic numbers it does not have.
    """
    from benchmarks.run_evaluation import (
        MeasurementUnavailable,
        _require_measured_cell,
    )

    # Without --cachegrind, a cell with no counters at all is complete.
    _require_measured_cell("resnet50", (1, 3, 64, 64), _complete_cell())

    # Each of the three ways cachegrind comes back empty.
    for block in (None, {"error": "valgrind not found"}, {"exit_code": 0}):
        with pytest.raises(MeasurementUnavailable, match="counters missing"):
            _require_measured_cell(
                "resnet50",
                (1, 3, 64, 64),
                _complete_cell(cachegrind=block),
                require_cachegrind=True,
            )

    # And the case that must pass: real counters on every arm.
    from benchmarks.alloc_attribution import ARMS

    counters = {"D1": 1234, "LL": 56, "exit_code": 0}
    _require_measured_cell(
        "resnet50",
        (1, 3, 64, 64),
        {"arms": [_measured_arm(a, cachegrind=counters) for a in ARMS]},
        require_cachegrind=True,
    )


def test_a_cell_missing_an_arm_is_refused():
    """Every physical number is a difference; a missing arm has no difference.

    `build_arms` used to log and skip an arm whose transform failed, which left
    the attribution table rendering normally and the cell counting as measured
    while the comparison it reported had one side.
    """
    from benchmarks.alloc_attribution import ARMS
    from benchmarks.run_evaluation import (
        MeasurementUnavailable,
        _require_measured_cell,
    )

    def _arm(name):
        return {
            "arm": name,
            "errors": [],
            "bufferized_peak_bytes": 1 << 20,
            "measured_rss": {"peak_rss_kb": 4096},
        }

    # Only the untiled arm: nothing to compare it against.
    with pytest.raises(MeasurementUnavailable, match="never built"):
        _require_measured_cell(
            "resnet50", (1, 3, 64, 64), {"arms": [_arm("untiled")]}
        )

    # All three: accepted.
    _require_measured_cell(
        "resnet50", (1, 3, 64, 64), {"arms": [_arm(a) for a in ARMS]}
    )


def test_build_arms_raises_rather_than_dropping_an_arm():
    import inspect

    from benchmarks import alloc_attribution

    source = inspect.getsource(alloc_attribution.build_arms)
    assert "ArmUnavailable" in source
    # The two former skip paths must no longer `continue`.
    assert "continue" not in source, "an arm can still be silently dropped"


# ---------------------------------------------------------------------------
# hardware_profiler: a skipped measurement and a wrong binary.
# ---------------------------------------------------------------------------

def test_a_skipped_cachegrind_run_reports_none_not_zero():
    """0.0% reads as "measured, no effect"; None reads as "not measured".

    With --skip-cachegrind every reduction used to be 0.0, which landed in the
    JSON indistinguishable from a genuine zero result, and the summary averaged
    those zeros into a headline "mean traffic reduction".
    """
    import inspect

    from benchmarks import hardware_profiler

    source = inspect.getsource(hardware_profiler.profile_model)
    assert "reduction = None" in source
    assert "reduction = 0.0" not in source

    summary = inspect.getsource(hardware_profiler.run_hardware_validation)
    assert "measured_reductions" in summary
    # The aggregate must not fabricate a zero when nothing was measured.
    assert "else 0.0" not in summary.split("mean_traffic_reduction_pct")[1][:400]


def test_the_report_distinguishes_skipped_from_zero():
    """A null reduction alone cannot say which of the two it is."""
    import inspect

    from benchmarks import hardware_profiler

    assert "cachegrind_skipped" in inspect.getsource(
        hardware_profiler._profile_to_dict
    )
    fields = hardware_profiler.ModelHardwareReport.__dataclass_fields__
    assert "cachegrind_skipped" in fields


def test_a_numerically_wrong_binary_is_refused_not_warned():
    """Its cache and timing numbers describe a different computation.

    This used to print a warning saying the numbers "must not be cited" and
    then return them in the JSON beside every valid cell, with nothing in the
    data marking the difference.
    """
    import inspect

    from benchmarks import hardware_profiler

    assert issubclass(hardware_profiler.NumericallyInvalidBinary, RuntimeError)
    source = inspect.getsource(hardware_profiler.profile_model)
    assert "raise NumericallyInvalidBinary" in source
    assert "do not cite its numbers" not in source, (
        "the warning-and-continue path is back"
    )

    # And the sweep records it without letting it into any aggregate.
    sweep = inspect.getsource(hardware_profiler.run_hardware_validation)
    assert "except NumericallyInvalidBinary" in sweep
    assert "numerically_invalid" in sweep
    invalid_branch = sweep[sweep.index("except NumericallyInvalidBinary"):]
    invalid_branch = invalid_branch[: invalid_branch.index("reports.append")]
    assert "continue" in invalid_branch


def test_bufferization_refuses_to_run_without_deallocation():
    """The primary metric's input must not be a module that never frees.

    `run_one_shot_bufferize(insert_deallocs=True)` used to retry without the
    deallocation pipeline and return success, handing
    `BufferizedMemoryAnalyzer` a module whose peak is the sum of every
    allocation — which its own docstring described, and then did anyway.
    """
    import inspect

    from edgeslide.liveness import analyzer

    source = inspect.getsource(analyzer.run_one_shot_bufferize)
    assert "DeallocationUnavailableError" in source
    assert "attempts" not in source, "the retry loop is back"


# ---------------------------------------------------------------------------
# "Cannot tell" must not become "fits" or "no effect".
# ---------------------------------------------------------------------------

def test_an_uncostable_plan_entry_is_refused_not_zeroed():
    """A zero-byte working set satisfies every cache budget.

    Proposal H1 is checked by comparing a plan's working set against the
    budget, so defaulting an uncostable entry to 0 turned "we cannot tell
    whether this fits" into "this fits" — and reported a 0-byte tile.
    """
    from edgeslide.dag.graph import OperationDAG, TensorInfo
    from edgeslide.transform.pass_manager import (
        EdgeSlicePassConfig,
        UncostablePlanEntry,
        plan_with_fusion,
    )

    dag = OperationDAG()
    dag.add_node(
        "linalg.conv_2d_nchw_fchw",
        inputs=[TensorInfo((1, 8, 34, 34)), TensorInfo((16, 8, 3, 3))],
        output=TensorInfo((1, 16, 32, 32)),
    )
    config = EdgeSlicePassConfig(l1_cache_bytes=256 * 1024)

    # An op the DAG does not contain.
    with pytest.raises(UncostablePlanEntry, match="not in the DAG"):
        plan_with_fusion(
            dag,
            [
                {
                    "op_id": 999,
                    "op_type": "linalg.conv_2d_nchw_fchw",
                    "slice_dimension": "spatial_h",
                    "tile_size": 4,
                    "original_shape": [1, 16, 32, 32],
                }
            ],
            config,
        )

    # A dimension that does not map onto this op's layout.
    with pytest.raises(UncostablePlanEntry, match="does not map onto"):
        plan_with_fusion(
            dag,
            [
                {
                    "op_id": 0,
                    "op_type": "linalg.conv_2d_nchw_fchw",
                    "slice_dimension": "M",  # matmul-only name
                    "tile_size": 4,
                    "original_shape": [1, 16, 32, 32],
                }
            ],
            config,
        )


def test_no_baseline_means_no_reduction_percentage():
    """A peak of 0 bytes is the absence of a measurement, not a small peak."""
    from benchmarks.metrics import peak_reduction_pct

    assert peak_reduction_pct(0, 1 << 20) is None
    assert peak_reduction_pct(-1, 1 << 20) is None
    # And a real pair still computes.
    assert peak_reduction_pct(1000, 750) == 25.0
    # Including a genuine zero reduction, which must stay distinguishable.
    assert peak_reduction_pct(1000, 1000) == 0.0


def test_the_summary_says_how_many_cells_it_averaged():
    """A mean over a subset must not read as a mean over the matrix."""
    from benchmarks.run_evaluation import _build_summary

    def _cell(model, reduction):
        return {
            "model": model,
            "weights_provenance": {"is_pretrained": True},
            "metrics": {
                "peak_memory_reduction_pct": reduction,
                "spatial_halo_reduction_pct": 0.0,
                "compile_time_overhead_ms": 10.0,
            },
        }

    summary = _build_summary([_cell("a", 10.0), _cell("a", None), _cell("b", 20.0)])
    assert summary["configurations"] == 3
    assert summary["reduction_measured_in"] == 2, "the None cell must be excluded"
    assert summary["avg_peak_memory_reduction_pct"] == 15.0
    assert summary["by_model"]["a"]["configurations"] == 2
    assert summary["by_model"]["a"]["reduction_measured_in"] == 1

    # Nothing measured at all reports None, not 0.0.
    empty = _build_summary([_cell("a", None)])
    assert empty["reduction_measured_in"] == 0
    assert empty["avg_peak_memory_reduction_pct"] is None
    assert empty["max_peak_memory_reduction_pct"] is None


# ---------------------------------------------------------------------------
# Statistics must not invent precision they did not measure.
# ---------------------------------------------------------------------------

def test_a_bootstrap_interval_needs_at_least_two_samples():
    """A zero-width CI is a claim of perfect precision.

    No samples used to return (0.0, 0.0) — "the median is exactly 0, with
    certainty". One sample returned (x, x), because resampling a single value
    can only ever yield that value, so the bootstrap has no variance to
    estimate. Beside a real interval in a results table, either reads as the
    most precise measurement in the run.
    """
    from benchmarks.metrics import bootstrap_ci

    assert bootstrap_ci([]) is None
    assert bootstrap_ci([12.5]) is None

    interval = bootstrap_ci([10.0, 11.0, 12.0, 13.0, 14.0])
    assert interval is not None
    low, high = interval
    assert low < high, "a real interval has width"


def test_an_empty_timing_summary_reports_none_not_zero_milliseconds():
    from benchmarks.metrics import summarize_timing_samples

    empty = summarize_timing_samples([])
    assert empty["sample_count"] == 0
    assert empty["mean_ms"] is None
    assert empty["median_ms"] is None
    assert empty["median_ci95_ms"] is None

    one = summarize_timing_samples([7.0])
    assert one["median_ms"] == 7.0, "a single sample is still a median"
    assert one["median_ci95_ms"] is None, "but it is not an interval"


def test_a_speedup_without_timings_is_none_not_zero():
    """"No speedup measured" and "measured, no speedup" are different findings."""
    from benchmarks.metrics import compare_paired_timings

    nothing = compare_paired_timings([], [])
    assert nothing["median_speedup_pct"] is None
    assert nothing["median_paired_difference_ms"] is None

    real = compare_paired_timings([10.0, 10.0, 10.0], [5.0, 5.0, 5.0])
    assert real["median_speedup_pct"] == pytest.approx(50.0)


def test_an_unevaluable_hypothesis_is_none_not_false():
    """H4 with no samples is unevaluated; False would read as "H4 violated"."""
    import inspect

    from benchmarks import run_evaluation

    source = inspect.getsource(run_evaluation._collect_compile_time_statistics)
    assert "baseline_median is not None" in source
    assert "candidate_median is not None" in source


def test_the_report_formatters_survive_a_missing_statistic():
    """plot.py coerced with float(), which raises on None."""
    import importlib.util
    import pathlib as _pathlib

    spec = importlib.util.spec_from_file_location(
        "_edgeslide_plot",
        _pathlib.Path(__file__).resolve().parents[1] / "scripts" / "plot.py",
    )
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)

    assert plot._fmt_ms(None) == "n/a"
    assert plot._fmt_pct(None) == "n/a"
    assert plot._fmt_ms(12.34) == "12.3 ms"
    assert plot._fmt_pct(5.0).startswith("+")
    assert plot._maybe_float(None) is None
    assert plot._maybe_float("3.5") == 3.5


def test_a_profiler_comparison_needs_equal_vectorization():
    """The confound found in Phase 4, checked where it also applies.

    `enable_vectorization` is passed to both binaries, but
    `outline_stride1_convs` matches per-op and a *tiled* payload can present
    fewer matchable convolutions than an untiled one — on ResNet-18 at 64x64
    the untiled arm vectorized 133 convolutions and the tiled arms 93. A
    "traffic reduction" across that gap measures tiling plus 40 fewer
    vectorized convs, and nothing in the output said so.
    """
    from benchmarks import hardware_profiler

    assert issubclass(hardware_profiler.UnequalVectorization, RuntimeError)
    source = inspect.getsource(hardware_profiler.profile_model)
    assert "raise UnequalVectorization" in source
    assert "count_simd_types" in source

    # And the sweep records it without letting it into any aggregate.
    sweep = inspect.getsource(hardware_profiler.run_hardware_validation)
    assert "except UnequalVectorization" in sweep
    branch = sweep[sweep.index("except UnequalVectorization"):]
    branch = branch[: branch.index("reports.append")]
    assert "continue" in branch

    # What was achieved is recorded, not only what was requested.
    assert "simd_types_emitted" in hardware_profiler.ModelHardwareReport.__dataclass_fields__


def test_simd_types_are_counted_in_the_llvm_ir(tmp_path):
    """The MLIR `vector.*` ops are gone by the end of lowering.

    Counting them in the stage-2 output always reads zero whether or not
    vectorization happened, which is how a broken vectorizer went unnoticed
    for as long as it did.
    """
    from benchmarks.hardware_profiler import count_simd_types

    scalar = tmp_path / "scalar.ll"
    scalar.write_text("define void @f() {\n  ret void\n}\n")
    assert count_simd_types(scalar) == 0

    vector = tmp_path / "vector.ll"
    vector.write_text(
        "define void @f(ptr %p) {\n"
        "  %a = load <8 x float>, ptr %p\n"
        "  %b = fmul <8 x float> %a, %a\n"
        "  store <8 x float> %b, ptr %p\n"
        "  ret void\n}\n"
    )
    assert count_simd_types(vector) == 3


def test_the_phase4_script_labels_unreadable_rows():
    """Two of four resolutions produced comparisons that are not about tiling.

    Reported rather than raised: the numbers are still worth seeing, and the
    reader needs to know which rows they cannot compare.
    """
    import pathlib as _pathlib

    path = (
        _pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "phase4_vectorized_transfer.py"
    )
    source = path.read_text()
    namespace: dict = {"List": list, "Dict": dict, "Any": object}
    exec(
        source[
            source.index("def _validity_warnings") : source.index("def main() -> int:")
        ],
        namespace,
    )
    warn = namespace["_validity_warnings"]

    def _row(arm, vectorized, ops, rss_kb):
        return {
            "arm": arm,
            "vectorization_requested": vectorized,
            "vectorized_ops": ops,
            "rss_kb": rss_kb,
        }

    # Unequal vectorization (the 64x64 case).
    unequal = warn([
        _row("untiled", True, 133, 50_000),
        _row("tiled_fused", True, 93, 50_000),
    ])
    assert any("did NOT vectorize equally" in w for w in unequal)

    # A 3x footprint difference (the 96x96 case).
    blown = warn([
        _row("untiled", True, 133, 92_000),
        _row("tiled_fused", True, 133, 276_000),
    ])
    assert any("footprints differ" in w for w in blown)

    # And a clean pair says so, rather than staying silent.
    clean = warn([
        _row("untiled", True, 133, 97_000),
        _row("tiled_fused", True, 133, 98_000),
    ])
    assert any("attributable to" in w for w in clean)


def test_an_unsizable_arm_keeps_its_rss_instead_of_losing_the_cell():
    """A refusal to size an allocation is not a failed measurement.

    The binary built and ran, so its RSS is a real observation; only the static
    peak is missing. Discarding the whole cell throws away a number that *was*
    measured because a different one could not be computed — which is how five
    of ten Phase 1 cells went missing.
    """
    from benchmarks.run_evaluation import _require_measured_cell

    cell = _complete_cell()
    fused = next(a for a in cell["arms"] if a["arm"] == "tiled_fused")
    fused["errors"] = [
        "bufferize: allocation %alloc_123 has type memref<1x128x?x8xf32>, whose "
        "dynamic extent(s) [2] were resolved from an affine upper bound that the "
        "iteration domain is not known to reach."
    ]

    _require_measured_cell("resnet50", (1, 3, 64, 64), cell)

    assert fused["bufferized_peak_bytes"] is None
    assert fused["bufferized_peak_mb"] is None
    assert "not known to reach" in fused["bufferized_peak_unavailable_reason"]
    assert fused["measured_rss"]["peak_rss_kb"], "the RSS measurement was discarded"
    assert "tiled_fused" in cell["bufferized_peak_unavailable"]


def test_an_arm_that_failed_to_build_still_loses_the_cell():
    """Only a sizing refusal is excused; a real failure is still a failure."""
    from benchmarks.run_evaluation import MeasurementUnavailable, _require_measured_cell

    cell = _complete_cell()
    fused = next(a for a in cell["arms"] if a["arm"] == "tiled_fused")
    fused["errors"] = ["bufferize: mlir-opt exited 1: could not parse payload"]

    with pytest.raises(MeasurementUnavailable, match="could not parse"):
        _require_measured_cell("resnet50", (1, 3, 64, 64), cell)


def test_a_reduction_is_not_reported_against_an_unsized_arm():
    """Reducing against a zero would read as a 100% saving."""
    from benchmarks.physical import CellMeasurement
    from benchmarks.run_evaluation import _bufferized_peak_reduction

    untiled = CellMeasurement(arm="untiled", bufferized_peak_bytes=8_000_000)
    unsized = CellMeasurement(arm="tiled_fused", bufferized_peak_bytes=0)
    assert _bufferized_peak_reduction(untiled, unsized) is None, (
        "an arm with no sized peak produced a reduction figure"
    )

    sized = CellMeasurement(arm="tiled_fused", bufferized_peak_bytes=4_000_000)
    assert _bufferized_peak_reduction(untiled, sized) == pytest.approx(50.0)
