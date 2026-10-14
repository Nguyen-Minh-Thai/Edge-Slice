#!/usr/bin/env python3
"""
EdgeSlice evaluation runner — proposal §5.1–5.4.

Runs ResNet-50 and YOLO-style workloads at 512×512, 1024×1024, and 1600×900
with baselines, full EdgeSlice, and ablation studies. Writes JSON metrics
including real peak_memory_reduction_pct and compile-time overhead.

Usage:
    python benchmarks/run_evaluation.py
    python benchmarks/run_evaluation.py --model resnet50 --l1-kb 256
    python benchmarks/run_evaluation.py --all-models --output benchmarks/results/evaluation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from edgeslide.liveness.peak_memory import PeakMemoryEstimator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.metrics import (  # noqa: E402
    CompileTiming,
    StrategyResult,
    compare_paired_timings,
    estimate_post_slice_peak_bytes,
    peak_reduction_pct,
    summarize_plan,
)
from benchmarks.workloads import (  # noqa: E402
    STANDARD_RESOLUTIONS,
    SUPPORTED_MODELS,
    load_model,
    make_example_input,
    resolution_to_input_shape,
    weights_provenance,
)


class MeasurementUnavailable(RuntimeError):
    """A cell was asked for a measurement and could not produce one.

    Distinct from a crash so the sweep records it and continues, and distinct
    from success so its analytical estimate never reaches the summary.
    """


def _require_torch_mlir() -> None:
    try:
        import torch_mlir  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "torch_mlir is required for DAG-based evaluation. "
            "Install torch-mlir in the project venv."
        ) from exc


def _parse_resolution(text: str) -> Tuple[int, int]:
    if "x" not in text.lower():
        raise argparse.ArgumentTypeError(f"Expected HxW, got {text!r}")
    parts = text.lower().split("x")
    return int(parts[0]), int(parts[1])


def _evaluate_strategy(
    name: str,
    dag,
    rewrite_plan: List[dict],
    l1_cache_bytes: int,
    *,
    compile_timing: Optional[CompileTiming] = None,
    estimator: Optional[PeakMemoryEstimator] = None,
) -> StrategyResult:
    from edgeslide.liveness.peak_memory import PeakMemoryEstimator

    # One estimator (memoized liveness sweep) is shared across all strategies
    # of a configuration; only build a fresh one when none was provided.
    # Note this dedupes only the evaluation-side sweeps: EdgeSlicePassManager
    # deliberately builds its own estimator inside run() so each strategy's
    # published liveness_ms measures the real cost of a cold sweep — injecting
    # a warm one there would understate compile-time overhead.
    if estimator is None:
        estimator = PeakMemoryEstimator(dag, l1_cache_bytes)

    baseline_op, baseline_peak = estimator.get_peak()
    if rewrite_plan:
        peak_op, peak_bytes = estimate_post_slice_peak_bytes(
            dag, rewrite_plan, l1_cache_bytes, estimator=estimator
        )
    else:
        peak_op, peak_bytes = baseline_op, baseline_peak

    return StrategyResult(
        name=name,
        peak_memory_bytes=peak_bytes,
        peak_op_id=peak_op,
        rewrite_ops=len(rewrite_plan),
        compile_timing=compile_timing or CompileTiming(),
        rewrite_plan_summary=summarize_plan(rewrite_plan),
    )


def _spatial_halo_baseline_rewrite_plan(
    dag,
    l1_cache_bytes: int,
    *,
    min_out_channels: int = 64,
) -> List[dict]:
    """Proposal §5.3 Baseline 3: per-op spatial-halo tiling, no DAG analysis.

    **Not SConv.** This is a spatial-halo tiling *strategy* evaluated under the
    same analytical peak-memory model as every other arm here, so the arms
    differ only in their tiling decisions. It borrows two parameters from
    ``edgeslide.sconv_pytorch`` — a gate on output channels and a row-budget
    formula — because they are a reasonable per-op halo policy, not because
    the result is attributable to Ferrari et al. The published transform is a
    separate, real arm: ``benchmarks/run_sixcam_sconv_comparison.py
    --methods sconv_upstream`` runs `transform.structured.sconv` itself.

    Borrows the gate (``out_channels >= min_out_channels``) and the row-budget
    formula
    (``sconv_pytorch.compute_spatial_halo_tile_rows``), but applied to DAG node
    shapes instead of live tensors so the result is a rewrite plan comparable,
    byte-for-byte, with the other three strategies via
    ``estimate_post_slice_peak_bytes`` — the same static analytical peak-memory
    model, not a separate real-execution measurement. Unlike EdgeSlice, there is
    no liveness sweep here: every sufficiently-wide conv is tiled independently,
    which is exactly the "no DAG analysis" half of the baseline's proposal
    description.
    """
    from edgeslide.sconv_pytorch import compute_spatial_halo_tile_rows
    from edgeslide.tiling.dimension_selector import dimension_indices
    from edgeslide.tiling.tile_size import node_cost_kwargs

    plan: List[dict] = []
    for op_id, node in dag.nodes.items():
        if "conv" not in node.op_type.lower():
            continue

        out_shape = node.output_tensor.shape
        rank = len(out_shape)
        indices = dimension_indices(node.op_type, rank)
        if "channel" not in indices or "spatial_h" not in indices:
            continue
        out_channels = int(out_shape[indices["channel"]])
        if out_channels < min_out_channels:
            continue

        kwargs = node_cost_kwargs(node)
        input_shape = kwargs.get("input_shape")
        kernel_shape = kwargs.get("kernel_shape")
        if input_shape is None or kernel_shape is None:
            continue
        in_indices = dimension_indices(node.op_type, len(input_shape))
        if "channel" not in in_indices or "spatial_w" not in in_indices:
            continue

        dilation_h, _ = kwargs.get("dilation", (1, 1))
        out_h = int(out_shape[indices["spatial_h"]])
        tile_output_rows = compute_spatial_halo_tile_rows(
            batch=int(input_shape[0]),
            in_channels=int(input_shape[in_indices["channel"]]),
            in_w=int(input_shape[in_indices["spatial_w"]]),
            out_channels=out_channels,
            out_w=int(out_shape[indices["spatial_w"]]),
            out_h=out_h,
            kernel_h=int(kernel_shape[0]),
            dilation_h=int(dilation_h),
            dtype_size=node.output_tensor.element_size_bytes,
            l1_cache_bytes=l1_cache_bytes,
        )
        if tile_output_rows >= out_h:
            continue  # SConv itself would skip tiling here (needs_tiling=False)

        plan.append(
            {
                "op_id": op_id,
                "op_type": node.op_type,
                "slice_dimension": "spatial_h",
                "tile_size": tile_output_rows,
                "original_shape": list(out_shape),
            }
        )
    return plan


def _run_pass_manager_strategy(
    dag,
    l1_cache_bytes: int,
    *,
    enable_liveness_analysis: bool,
    enable_smart_heuristic: bool,
    uniform_tile_size: int = 64,
    uniform_slice_dimension: str = "spatial_h",
    name: str,
    estimator: Optional[PeakMemoryEstimator] = None,
) -> StrategyResult:
    from edgeslide.transform.pass_manager import EdgeSlicePassConfig, EdgeSlicePassManager

    config = EdgeSlicePassConfig(
        l1_cache_bytes=l1_cache_bytes,
        enable_liveness_analysis=enable_liveness_analysis,
        enable_smart_heuristic=enable_smart_heuristic,
        uniform_tile_size=uniform_tile_size,
        uniform_slice_dimension=uniform_slice_dimension,
    )
    pm = EdgeSlicePassManager(dag, config=config)
    results = pm.run()
    timing = CompileTiming(
        liveness_ms=results["compile_timing"]["liveness_ms"],
        planning_ms=results["compile_timing"]["planning_ms"],
        transform_ir_ms=results["compile_timing"]["transform_ir_ms"],
        total_ms=results["compile_timing"]["total_ms"],
        vertices=results["compile_timing"]["dag_vertices"],
        edges=results["compile_timing"]["dag_edges"],
    )
    return _evaluate_strategy(
        name,
        dag,
        results.get("rewrite_plan", []),
        l1_cache_bytes,
        compile_timing=timing,
        estimator=estimator,
    )


def _collect_compile_time_statistics(
    model_name: str,
    height: int,
    width: int,
    l1_cache_bytes: int,
    *,
    uniform_tile_size: int,
    batch_size: int,
    use_real_images: bool,
    image_dir: Optional[str],
    allow_dummy_fallback: bool,
    trials: int,
) -> Dict[str, Any]:
    """Repeated-trial compile-time samples for proposal §5.5 / H4.

    Peak-memory bytes are deterministic given the same DAG, so re-deriving
    them adds no information — but wall-clock compile time is genuinely
    noisy, and that noise is exactly what a Wilcoxon/bootstrap comparison and
    H4 ("compile-time overhead < 2x baseline compile time", proposal §6.1)
    are meant to characterize. Each trial reloads the model and re-parses the
    DAG from scratch so the samples are independent runs, not repeated reads
    of one cached result.
    """
    from edgeslide.dag.parser import DAGParser

    baseline_ms: List[float] = []
    edgeslide_ms: List[float] = []
    for _ in range(trials):
        trial_model = load_model(model_name)
        trial_input, _ = make_example_input(
            model_name,
            height,
            width,
            batch_size=batch_size,
            use_real_images=use_real_images,
            image_dir=image_dir,
            allow_dummy_fallback=allow_dummy_fallback,
            return_meta=True,
        )
        t0 = time.perf_counter()
        trial_dag = DAGParser(trial_model, trial_input).parse()
        trial_parse_ms = (time.perf_counter() - t0) * 1000
        baseline_ms.append(trial_parse_ms)

        trial_result = _run_pass_manager_strategy(
            trial_dag,
            l1_cache_bytes,
            enable_liveness_analysis=True,
            enable_smart_heuristic=True,
            uniform_tile_size=uniform_tile_size,
            name="edgeslide_full_trial",
        )
        edgeslide_ms.append(trial_parse_ms + trial_result.compile_timing.total_ms)

    comparison = compare_paired_timings(baseline_ms, edgeslide_ms)
    comparison["trials"] = trials
    comparison["description"] = (
        "baseline = DAG parse only (Baseline 1's compile cost); "
        "candidate = DAG parse + EdgeSlice liveness+planning+transform-IR "
        "(proposal §5.5 statistical analysis, testing H4 from §6.1)"
    )
    baseline_median = comparison["baseline"]["median_ms"]
    candidate_median = comparison["candidate"]["median_ms"]
    comparison["h4_overhead_threshold_multiplier"] = 2.0
    # None when either side has no samples: an unevaluable hypothesis is not a
    # failed one, and `False` here would read as "H4 violated".
    comparison["h4_satisfied"] = (
        candidate_median <= 2.0 * baseline_median
        if baseline_median is not None
        and candidate_median is not None
        and baseline_median > 0
        else None
    )
    return comparison


def evaluate_configuration(
    model_name: str,
    height: int,
    width: int,
    l1_cache_kb: int,
    *,
    uniform_tile_size: int = 64,
    batch_size: int = 1,
    use_real_images: bool = False,
    image_dir: Optional[str] = None,
    allow_dummy_fallback: bool = False,
    trials: int = 1,
    halo_min_out_channels: int = 64,
    physical: bool = False,
    physical_dir: Optional[str] = None,
    cachegrind: bool = False,
    cachegrind_device: str = "pi4",
) -> Dict[str, Any]:
    """Evaluate all baselines + EdgeSlice + ablations for one model/resolution.

    With ``physical=True`` the cell additionally compiles the payload for real
    and records the bufferized peak and the measured RSS beside the analytical
    estimate, so no number in the headline path is analysis-only.
    """
    _require_torch_mlir()
    from edgeslide.dag.parser import DAGParser
    from edgeslide.liveness.peak_memory import PeakMemoryEstimator

    l1_cache_bytes = l1_cache_kb * 1024
    input_shape = resolution_to_input_shape(height, width, batch_size=batch_size)

    model = load_model(model_name)
    example_input, input_meta = make_example_input(
        model_name,
        height,
        width,
        batch_size=batch_size,
        use_real_images=use_real_images,
        image_dir=image_dir,
        allow_dummy_fallback=allow_dummy_fallback,
        return_meta=True,
    )

    t_parse = time.perf_counter()
    dag = DAGParser(model, example_input).parse()
    parse_ms = (time.perf_counter() - t_parse) * 1000

    baseline_est = PeakMemoryEstimator(dag, l1_cache_bytes)
    baseline_op, baseline_peak = baseline_est.get_peak()
    candidates = baseline_est.slicing_candidates()

    # Baseline 1: no slicing
    baseline_no_slicing = StrategyResult(
        name="baseline_no_slicing",
        peak_memory_bytes=baseline_peak,
        peak_op_id=baseline_op,
        rewrite_ops=0,
        compile_timing=CompileTiming(
            parse_ms=parse_ms,
            vertices=len(dag.nodes),
            edges=len(dag.edges),
        ),
    )

    # Baseline 2: uniform tiling (fixed tile/dim, no liveness gate).
    # Planned by the pass manager like every other strategy: a separate
    # benchmark-side planner used a weaker cost model, so it accepted ops the
    # transform rejects and its plans got no fusion credit — the baseline was
    # being measured under a different model than what it is compared against.
    baseline_uniform = _run_pass_manager_strategy(
        dag,
        l1_cache_bytes,
        enable_liveness_analysis=False,
        enable_smart_heuristic=False,
        uniform_tile_size=uniform_tile_size,
        name="baseline_uniform_tiling",
        estimator=baseline_est,
    )
    baseline_uniform.compile_timing.parse_ms = parse_ms

    # EdgeSlice full (Ablation C)
    edgeslide = _run_pass_manager_strategy(
        dag,
        l1_cache_bytes,
        enable_liveness_analysis=True,
        enable_smart_heuristic=True,
        name="edgeslide_full",
        estimator=baseline_est,
    )
    edgeslide.compile_timing.parse_ms = parse_ms

    # Ablation A: DAG + liveness, uniform heuristic
    ablation_a = _run_pass_manager_strategy(
        dag,
        l1_cache_bytes,
        enable_liveness_analysis=True,
        enable_smart_heuristic=False,
        uniform_tile_size=uniform_tile_size,
        name="ablation_a_dag_only",
        estimator=baseline_est,
    )
    ablation_a.compile_timing.parse_ms = parse_ms

    # Ablation B: heuristic only, no liveness gate
    ablation_b = _run_pass_manager_strategy(
        dag,
        l1_cache_bytes,
        enable_liveness_analysis=False,
        enable_smart_heuristic=True,
        name="ablation_b_heuristic_only",
        estimator=baseline_est,
    )
    ablation_b.compile_timing.parse_ms = parse_ms

    # Baseline 3: per-op spatial-halo tiling, no DAG analysis.
    #
    # Run through the *same* SlicedRewrite chain construction and fused
    # working-set re-fit as every other arm. Without that, this plan carried no
    # ``fused_producer_op_ids`` and was therefore structurally incapable of
    # receiving fusion credit, while EdgeSlice's plan received it — so the two
    # arms were being compared under different memory models and the gap
    # between them included that asymmetry. What differs now is only the tiling
    # decision, which is the thing under comparison.
    from edgeslide.transform.pass_manager import (
        EdgeSlicePassConfig as _FusionConfig,
    )
    from edgeslide.transform.pass_manager import (
        plan_with_fusion,
    )

    halo_plan = _spatial_halo_baseline_rewrite_plan(
        dag, l1_cache_bytes, min_out_channels=halo_min_out_channels
    )
    halo_plan, halo_fusion_warnings = plan_with_fusion(
        dag, halo_plan, _FusionConfig(l1_cache_bytes=l1_cache_bytes)
    )
    baseline_halo = _evaluate_strategy(
        "baseline_halo_spatial_halo",
        dag,
        halo_plan,
        l1_cache_bytes,
        estimator=baseline_est,
    )
    baseline_halo.compile_timing.parse_ms = parse_ms

    edgeslide_reduction = peak_reduction_pct(
        baseline_no_slicing.peak_memory_bytes,
        edgeslide.peak_memory_bytes,
    )
    uniform_reduction = peak_reduction_pct(
        baseline_no_slicing.peak_memory_bytes,
        baseline_uniform.peak_memory_bytes,
    )
    halo_reduction = peak_reduction_pct(
        baseline_no_slicing.peak_memory_bytes,
        baseline_halo.peak_memory_bytes,
    )

    compile_time_statistics = None
    if trials > 1:
        compile_time_statistics = _collect_compile_time_statistics(
            model_name,
            height,
            width,
            l1_cache_bytes,
            uniform_tile_size=uniform_tile_size,
            batch_size=batch_size,
            use_real_images=use_real_images,
            image_dir=image_dir,
            allow_dummy_fallback=allow_dummy_fallback,
            trials=trials,
        )

    physical_block: Optional[Dict[str, Any]] = None
    if physical:
        physical_block = _measure_physical(
            model,
            model_name,
            input_shape,
            l1_cache_kb=l1_cache_kb,
            baseline_bytes=baseline_no_slicing.peak_memory_bytes,
            edgeslide_bytes=edgeslide.peak_memory_bytes,
            physical_dir=physical_dir,
            cachegrind=cachegrind,
            cachegrind_device=cachegrind_device,
        )
        # A cell that was asked to measure and could not is not a cell with an
        # analytical result — it is a failure. Returning the estimate here
        # would put a number in the matrix that the run had explicitly
        # declined to trust, indistinguishable from a measured one except by
        # reading `physical.error`, which nobody does.
        _require_measured_cell(
            model_name,
            input_shape,
            physical_block,
            require_cachegrind=cachegrind,
        )

    return {
        "model": model_name,
        "weights_source": getattr(
            model, "_edgeslide_weights_source", "unlabeled"
        ),
        # A digest of the weights that actually ran, so a result can be traced
        # back to them. "pretrained" is a claim; a hash is checkable.
        "weights_provenance": weights_provenance(model),
        "physical": physical_block,
        "input_shape": list(input_shape),
        "resolution_hw": [height, width],
        "l1_cache_kb": l1_cache_kb,
        "batch_size": batch_size,
        "input_meta": input_meta,
        "dag_nodes": len(dag.nodes),
        "dag_edges": len(dag.edges),
        "liveness_candidates": len(candidates),
        "baseline_no_slicing": baseline_no_slicing.to_dict(),
        "baseline_uniform_tiling": baseline_uniform.to_dict(),
        "baseline_halo_spatial_halo": baseline_halo.to_dict(),
        "baseline_halo_fusion_budget_warnings": halo_fusion_warnings,
        "edgeslide_full": edgeslide.to_dict(),
        "ablation_a_dag_only": ablation_a.to_dict(),
        "ablation_b_heuristic_only": ablation_b.to_dict(),
        "compile_time_statistics": compile_time_statistics,
        "metrics": {
            "peak_memory_reduction_pct": edgeslide_reduction,
            "uniform_tiling_reduction_pct": uniform_reduction,
            "spatial_halo_reduction_pct": halo_reduction,
            "spatial_halo_ops_tiled": baseline_halo.rewrite_ops,
            "ablation_a_reduction_pct": peak_reduction_pct(
                baseline_no_slicing.peak_memory_bytes,
                ablation_a.peak_memory_bytes,
            ),
            "ablation_b_reduction_pct": peak_reduction_pct(
                baseline_no_slicing.peak_memory_bytes,
                ablation_b.peak_memory_bytes,
            ),
            "baseline_peak_memory_mb": baseline_no_slicing.peak_memory_mb,
            "edgeslide_peak_memory_mb": edgeslide.peak_memory_mb,
            "compile_time_overhead_ms": round(
                edgeslide.compile_timing.total_ms, 3
            ),
            "dag_parse_ms": round(parse_ms, 3),
            "analysis_time_ms": round(
                edgeslide.compile_timing.liveness_ms
                + edgeslide.compile_timing.planning_ms,
                3,
            ),
            "transform_ir_generation_ms": round(
                edgeslide.compile_timing.transform_ir_ms, 3
            ),
            "dag_vertices": len(dag.nodes),
            "dag_edges": len(dag.edges),
            # Ablation B *is* the heuristic-on-all-ops configuration (no
            # liveness gate), so its own plan is the reference count. A
            # separate benchmark-side builder used to report this and drifted:
            # it published 172 for every resnet resolution while ablation B
            # planned 128/106/113.
            "heuristic_all_ops_count": ablation_b.rewrite_ops,
        },
    }


def _measure_physical(
    model,
    model_name: str,
    input_shape: Tuple[int, ...],
    *,
    l1_cache_kb: int,
    baseline_bytes: int,
    edgeslide_bytes: int,
    physical_dir: Optional[str],
    cachegrind: bool,
    cachegrind_device: str = "pi4",
) -> Dict[str, Any]:
    """Compile the payload for real and measure what it actually costs.

    Three numbers per arm — frontend estimate, bufferized peak, measured RSS —
    because they disagree for understandable reasons and the disagreement is
    itself a result (see benchmarks/physical.py). Failures are recorded rather
    than raised: one arm that will not compile must not delete the whole cell,
    and a silently missing arm would make the matrix look more complete than
    it is.
    """
    import tempfile

    from benchmarks.physical import (
        environment_provenance,
        measure_arms,
        predictor_error,
    )

    # The frontend's own numbers, keyed by the arm they describe: the untiled
    # payload is the no-slicing baseline, the fused arm is what EdgeSlice emits.
    estimates = {
        "untiled": baseline_bytes,
        "tiled": baseline_bytes,
        "tiled_fused": edgeslide_bytes,
    }

    context: Any
    if physical_dir:
        work_dir = Path(physical_dir) / f"{model_name}_{input_shape[2]}x{input_shape[3]}"
        work_dir.mkdir(parents=True, exist_ok=True)
        context = None
    else:
        context = tempfile.TemporaryDirectory(prefix="edgeslide_physical_")
        work_dir = Path(context.name)

    try:
        cells = measure_arms(
            model,
            input_shape,
            l1_cache_kb=l1_cache_kb,
            work_dir=work_dir,
            model_name=model_name,
            frontend_estimates=estimates,
            run_cachegrind=cachegrind,
            cachegrind_device=None if cachegrind_device == "host" else cachegrind_device,
        )
    except Exception as exc:  # torch-mlir, mlir-opt, or clang unavailable
        return {"error": str(exc)[:600], "arms": []}
    finally:
        if context is not None:
            context.cleanup()

    arms = []
    for cell in cells:
        entry = cell.to_dict()
        entry["frontend_vs_bufferized_error_pct"] = predictor_error(
            cell.frontend_estimate_bytes, cell.bufferized_peak_bytes
        )
        arms.append(entry)

    by_arm = {cell.arm: cell for cell in cells}
    untiled = by_arm.get("untiled")
    fused = by_arm.get("tiled_fused")
    summary: Dict[str, Any] = {"arms": arms, "environment": environment_provenance()}
    reduction = _bufferized_peak_reduction(untiled, fused)
    if reduction is not None:
        summary["bufferized_peak_reduction_pct"] = reduction
    if (
        untiled is not None
        and fused is not None
        and untiled.measured_rss.peak_rss_kb
    ):
        summary["measured_rss_reduction_pct"] = peak_reduction_pct(
            untiled.measured_rss.peak_rss_kb, fused.measured_rss.peak_rss_kb
        )
    return summary



def _bufferized_peak_reduction(untiled, fused) -> Optional[float]:
    """Peak reduction between two arms, or None when either has no sized peak.

    Both arms, not just the baseline. An arm whose allocation could not be
    sized carries 0, and reducing against 0 reports a 100% saving for a cell
    where nothing was measured.
    """
    if untiled is None or fused is None:
        return None
    if not untiled.bufferized_peak_bytes or not fused.bufferized_peak_bytes:
        return None
    return peak_reduction_pct(
        untiled.bufferized_peak_bytes, fused.bufferized_peak_bytes
    )


def _IS_SIZE_REFUSAL(error: str) -> bool:
    """Is this arm error a refusal to size an allocation, not a failure to run?

    `InferredSizeRefused` declines to put a number on a dynamically-shaped
    allocation whose extent the iteration domain is not known to reach. The
    payload still compiled and still ran, so RSS for that arm is a real
    measurement; only the static peak is missing. Matched on the text because
    it arrives as a string through the arm's `errors` list.
    """
    return "not known to reach" in error or "dynamic memref dimension" in error


def _require_measured_cell(
    model_name: str,
    input_shape: Tuple[int, ...],
    physical: Optional[Dict[str, Any]],
    *,
    require_cachegrind: bool = False,
) -> None:
    """Raise unless every arm of this cell produced a real measurement.

    ``run_evaluation`` catches this per configuration and records it in the
    report's ``errors`` list, so one uncompilable cell neither aborts the sweep
    nor quietly contributes an estimate to the summary.
    """
    label = f"{model_name}@{input_shape[2]}x{input_shape[3]}"
    if not physical or physical.get("error"):
        reason = (physical or {}).get("error", "no physical block produced")
        raise MeasurementUnavailable(
            f"{label}: physical measurement failed ({str(reason)[:300]}). "
            "Re-run with --no-physical to get analytical estimates, "
            "acknowledging they are not measurements."
        )

    arms = physical.get("arms") or []
    if not arms:
        raise MeasurementUnavailable(f"{label}: no arms were measured")


    # Each of the three numbers `measurement_scope` promises has to be present.
    # A zero is not a small measurement, it is the absence of one: RSS comes
    # from an OS high-water mark that is never legitimately zero for a process
    # that ran, and a bufferized peak of zero means the analyzer found no
    # allocations, which no compiled payload has.
    #
    # One exception, and it is a distinction rather than a loophole. A refusal
    # to *size* a dynamically-shaped allocation is not a failed measurement:
    # the binary built, ran, and its RSS is a real observation. Only the static
    # column is unavailable. Discarding the cell for that throws away a number
    # that was measured because a different number could not be computed. Such
    # an arm keeps its RSS and reports `bufferized_peak_bytes: null` with a
    # reason, and the cell's bufferized-peak reduction becomes None rather than
    # a figure derived from a partial set of arms.
    broken: Dict[str, str] = {}
    unsizable: Dict[str, str] = {}
    for arm in arms:
        errors = arm.get("errors") or []
        refusals = [e for e in errors if _IS_SIZE_REFUSAL(e)]
        other = [e for e in errors if not _IS_SIZE_REFUSAL(e)]
        if other:
            broken[arm["arm"]] = "; ".join(other)
            continue
        if refusals:
            unsizable[arm["arm"]] = "; ".join(refusals)
            arm["bufferized_peak_bytes"] = None
            arm["bufferized_peak_mb"] = None
            arm["bufferized_peak_unavailable_reason"] = unsizable[arm["arm"]][:300]
        elif not arm.get("bufferized_peak_bytes"):
            broken[arm["arm"]] = "no bufferized peak"
        if not (arm.get("measured_rss") or {}).get("peak_rss_kb"):
            rss_error = (arm.get("measured_rss") or {}).get("error") or "unknown"
            broken[arm["arm"]] = f"no measured RSS ({rss_error})"
    if broken:
        detail = "; ".join(f"{arm}: {why}" for arm, why in broken.items())
        raise MeasurementUnavailable(f"{label}: arm(s) unmeasured — {detail[:600]}")
    if unsizable:
        physical["bufferized_peak_unavailable"] = {
            arm: why[:300] for arm, why in unsizable.items()
        }
    # Every reported physical number is a *difference* between arms, so a
    # missing arm is not a partial result — the reduction it feeds simply
    # disappears from the output while the cell still counts as measured.
    from benchmarks.alloc_attribution import ARMS

    present = {arm["arm"] for arm in arms}
    absent = [arm for arm in ARMS if arm not in present]
    if absent:
        raise MeasurementUnavailable(
            f"{label}: arm(s) {absent} were never built, so the "
            "untiled-vs-tiled comparison cannot be made"
        )

    if not require_cachegrind:
        return
    # --cachegrind was asked for, so a cell without counters has not delivered
    # what the run requested. Counting a missing counter as "fine" would let a
    # matrix advertise cache-traffic numbers it does not have — and cachegrind
    # fails quietly in exactly the ways that matter: valgrind absent, the
    # simulation timing out, or `--cache-sim=yes` producing no D1 line.
    missing = {}
    for arm in arms:
        counters = arm.get("cachegrind")
        if not counters:
            missing[arm["arm"]] = "no cachegrind block"
        elif counters.get("error"):
            missing[arm["arm"]] = str(counters["error"])[:160]
        elif counters.get("D1") is None:
            missing[arm["arm"]] = "no D1 counter"
    if missing:
        detail = "; ".join(f"{arm}: {why}" for arm, why in missing.items())
        raise MeasurementUnavailable(
            f"{label}: --cachegrind requested but counters missing — "
            f"{detail[:600]}"
        )


def run_evaluation(
    models: List[str],
    resolutions: List[Tuple[int, int]],
    l1_cache_kb: int,
    *,
    uniform_tile_size: int = 64,
    batch_size: int = 1,
    use_real_images: bool = False,
    image_dir: Optional[str] = None,
    allow_dummy_fallback: bool = False,
    trials: int = 1,
    halo_min_out_channels: int = 64,
    physical: bool = False,
    physical_dir: Optional[str] = None,
    cachegrind: bool = False,
    cachegrind_device: str = "pi4",
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for model_name in models:
        for height, width in resolutions:
            label = f"{model_name}@{height}x{width}"
            print(
                f"\n=== Evaluating {label} (L1={l1_cache_kb} KB, "
                f"batch={batch_size}, real_images={use_real_images}) ==="
            )
            try:
                entry = evaluate_configuration(
                    model_name,
                    height,
                    width,
                    l1_cache_kb,
                    uniform_tile_size=uniform_tile_size,
                    batch_size=batch_size,
                    use_real_images=use_real_images,
                    image_dir=image_dir,
                    allow_dummy_fallback=allow_dummy_fallback,
                    trials=trials,
                    halo_min_out_channels=halo_min_out_channels,
                    physical=physical,
                    physical_dir=physical_dir,
                    cachegrind=cachegrind,
                    cachegrind_device=cachegrind_device,
                )
                results.append(entry)
                m = entry["metrics"]
                src = entry.get("input_meta", {}).get("source", "?")
                estimate_label = (
                    "est. reduction" if physical else "ANALYTICAL reduction"
                )
                print(
                    f"  Input source: {src} | "
                    f"Baseline peak: {m['baseline_peak_memory_mb']} MB | "
                    f"EdgeSlice peak: {m['edgeslide_peak_memory_mb']} MB | "
                    f"{estimate_label}: {_pct(m['peak_memory_reduction_pct'])} | "
                    f"halo-baseline: {_pct(m['spatial_halo_reduction_pct'])} | "
                    f"Compile: {m['compile_time_overhead_ms']} ms"
                )
                phys = entry.get("physical") or {}
                if phys and not phys.get("error"):
                    for arm in phys.get("arms", []):
                        # The dilution factor rides along with the RSS because
                        # the two are not the same scope: bufferized peak is
                        # heap allocations, RSS is every resident page,
                        # weights included. `x13.5` reads "RSS's scope is
                        # 13.5x the peak's, so a 1% peak change moves RSS by
                        # at most 0.07%".
                        peak_mb = arm.get("bufferized_peak_mb")
                        peak_txt = (
                            f"{peak_mb:>8.1f} MB" if peak_mb is not None
                            else "  unsized"
                        )
                        dilution = arm.get("rss_dilution_factor")
                        scope = f" (x{dilution:.1f} scope)" if dilution else ""
                        print(
                            f"    [{arm['arm']:<11}] frontend "
                            f"{arm['frontend_estimate_mb']:>8.1f} MB | bufferized "
                            f"{peak_txt} | RSS "
                            f"{arm['measured_rss']['peak_rss_mb']:>8.1f} MB{scope}"
                            + (f" | {'; '.join(arm['errors'])}" if arm["errors"] else "")
                        )
                elif phys.get("error"):
                    print(f"    physical measurement unavailable: {phys['error'][:200]}")
                stats = entry.get("compile_time_statistics")
                if stats:
                    h4 = stats.get("h4_satisfied")
                    print(
                        f"  Compile-time stats ({stats['trials']} trials): "
                        f"Wilcoxon p={stats['wilcoxon']['p_value']} | "
                        f"H4 (<2x baseline) satisfied: {h4}"
                    )
            except MeasurementUnavailable as exc:
                # Recorded as unmeasured rather than as a result. The
                # analytical estimate for this cell exists but is deliberately
                # discarded: including it would blend measured and unmeasured
                # numbers in one summary column.
                print(f"  UNMEASURED: {exc}")
                errors.append(
                    {
                        "configuration": label,
                        "error": str(exc),
                        "kind": "measurement_unavailable",
                        "analytical_estimate_discarded": True,
                    }
                )
            except Exception as exc:
                print(f"  FAILED: {exc}")
                errors.append({"configuration": label, "error": str(exc)})

    return {
        "metadata": {
            "benchmark_type": "edgeslide_evaluation",
            "description": "Proposal §5.1–5.4 frontend evaluation",
            "measurement_scope": (
                "static_liveness_estimate+bufferized_peak+measured_rss"
                if physical
                else "static_liveness_estimate"
            ),
            "peak_memory_is_runtime_rss": bool(physical),
            "runtime_speedup_claim_supported": False,
            # Baseline 3 is a per-op spatial-halo tiling *strategy* scored
            # under this runner's analytical model, not an execution of any
            # published implementation. It is deliberately no longer called
            # "sconv": the published transform is a separate real arm
            # (run_sixcam_sconv_comparison.py --methods sconv_upstream), and
            # naming this one after it invited attributing its numbers to
            # Ferrari et al. See docs/SCONV_COMPARISON.md.
            "halo_baseline_is_analytical_strategy": True,
            "halo_baseline_is_not_sconv": True,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "models": models,
            "resolutions": [list(r) for r in resolutions],
            "l1_cache_kb": l1_cache_kb,
            "uniform_tile_size": uniform_tile_size,
            "batch_size": batch_size,
            "use_real_images": use_real_images,
            "image_dir": image_dir,
            "trials": trials,
            "halo_min_out_channels": halo_min_out_channels,
            "cachegrind_device": cachegrind_device if cachegrind else None,
            "device": "cpu",
        },
        "results": results,
        "errors": errors,
        "summary": _build_summary(results),
    }


def _pct(value: Optional[float]) -> str:
    """Render a percentage, or "n/a" when there was nothing to measure."""
    return "n/a" if value is None else f"{value}%"


def _build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"configurations": 0}

    # `peak_reduction_pct` returns None when a cell had no baseline to reduce
    # against. Averaging over that used to raise; treating it as 0.0 would be
    # worse — it would pull a headline mean toward "no effect" using cells that
    # measured nothing. Skip them and say how many were skipped.
    def _measured(key: str, rows: List[Dict[str, Any]]) -> List[float]:
        return [
            float(r["metrics"][key])
            for r in rows
            if r["metrics"].get(key) is not None
        ]

    def _mean(values: List[float], digits: int = 2) -> Optional[float]:
        return round(sum(values) / len(values), digits) if values else None

    def _max(values: List[float], digits: int = 2) -> Optional[float]:
        return round(max(values), digits) if values else None

    reductions = _measured("peak_memory_reduction_pct", results)
    halo_reductions = _measured("spatial_halo_reduction_pct", results)
    compile_times = _measured("compile_time_overhead_ms", results)
    by_model: Dict[str, Dict[str, Any]] = {}
    for model in sorted({str(r.get("model", "unknown")) for r in results}):
        model_results = [r for r in results if r.get("model") == model]
        model_reductions = _measured("peak_memory_reduction_pct", model_results)
        by_model[model] = {
            "configurations": len(model_results),
            "reduction_measured_in": len(model_reductions),
            "avg_peak_memory_reduction_pct": _mean(model_reductions),
            "weights_sources": sorted(
                {str(r.get("weights_source", "unknown")) for r in model_results}
            ),
        }

    # How well the pre-bufferization estimate predicts the physical peak. This
    # is what justifies keeping a frontend estimator at all, and it is only
    # computable when the matrix was run with --physical.
    predictor: Optional[Dict[str, Any]] = None
    # Arms whose peak could not be sized contribute nothing here rather than a
    # zero, which would read as a 100% over-prediction.
    pairs = [
        (arm.get("frontend_estimate_bytes") or 0, arm["bufferized_peak_bytes"])
        for r in results
        for arm in ((r.get("physical") or {}).get("arms") or [])
        if arm.get("bufferized_peak_bytes")
    ]
    if pairs:
        from benchmarks.physical import summarize_predictor_accuracy

        predictor = summarize_predictor_accuracy(pairs)

    synthetic = sorted({
        str(r.get("model"))
        for r in results
        if not (r.get("weights_provenance") or {}).get("is_pretrained", True)
    })

    return {
        "configurations": len(results),
        "frontend_estimate_vs_bufferized_peak": predictor,
        # Named explicitly rather than left to be inferred from per-cell
        # metadata: a reader must not have to dig to find out which rows are
        # not pretrained.
        "models_with_synthetic_weights": synthetic,
        # How many of `configurations` actually contributed a reduction, so a
        # mean over a subset cannot be read as a mean over the matrix.
        "reduction_measured_in": len(reductions),
        "avg_peak_memory_reduction_pct": _mean(reductions),
        "max_peak_memory_reduction_pct": _max(reductions),
        "avg_spatial_halo_reduction_pct": _mean(halo_reductions),
        "avg_compile_time_overhead_ms": _mean(compile_times, 3),
        "max_compile_time_overhead_ms": _max(compile_times, 3),
        "by_model": by_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EdgeSlice evaluation runner")
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="resnet50",
        help="Model workload (default: resnet50)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all standard workloads",
    )
    parser.add_argument(
        "--resolution",
        type=_parse_resolution,
        action="append",
        default=[],
        help="Input resolution HxW (repeatable). Default: all standard resolutions.",
    )
    parser.add_argument("--l1-kb", type=int, default=256)
    parser.add_argument("--uniform-tile", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--real-images",
        action="store_true",
        help="Load real RGB images instead of torch.randn dummies "
        "(default dir: data/real_images; prepare via scripts/prepare_real_images.py)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory of .jpg/.png images (overrides data/real_images)",
    )
    parser.add_argument(
        "--allow-dummy-fallback",
        action="store_true",
        help="If --real-images is set but no files found, fall back to randn",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Repeated compile-time trials per configuration for proposal §5.5 "
        "statistics (Wilcoxon signed-rank + 95%% bootstrap CI on baseline vs "
        "EdgeSlice compile time, testing H4). 1 = stats disabled (default).",
    )
    parser.add_argument(
        "--halo-min-out-channels",
        type=int,
        default=64,
        help="Baseline 3 (SConv-style spatial-halo) gate: only tile conv ops "
        "with at least this many output channels, matching SConvTransform's "
        "default (src/edgeslide/sconv_pytorch.py SConvConfig.min_out_channels).",
    )
    # Measurement is the default. Analysis-only was the default, and the
    # numbers it produced were reported as results: the frontend estimator
    # over-predicts the physical peak in 24 of 24 measured arms (median 359%),
    # so an unlabelled analytical reduction is not a weaker measurement, it is
    # a different quantity. Opting out is possible and has to be said out loud.
    parser.add_argument(
        "--no-physical",
        dest="physical",
        action="store_false",
        help="Skip compilation and report the analytical estimate only. The "
        "estimate over-predicts the physical peak by a median of 359%% across "
        "the measured matrix, so nothing from such a run is a measurement; "
        "the output records measurement_scope accordingly.",
    )
    parser.set_defaults(physical=True)
    parser.add_argument(
        "--physical-dir",
        type=str,
        default=None,
        help="Keep the compiled IR and binaries from --physical in this "
        "directory instead of a temp dir (needed to inspect a cell afterwards).",
    )
    parser.add_argument(
        "--cachegrind-device",
        default="pi4",
        choices=("pi4", "cortex_a55", "cortex_a76", "host"),
        help="Cache geometry cachegrind simulates. Defaults to the Pi 4 "
        "(Cortex-A72), the evaluation target; 'host' uses valgrind's own "
        "detection, which models the development machine and is not "
        "comparable across hosts.",
    )
    parser.add_argument(
        "--cachegrind",
        action="store_true",
        help="Also collect D1/LL miss counts per arm under --physical. "
        "Deterministic but slow (valgrind runs the binary interpreted).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "benchmarks" / "results" / "evaluation_results.json"),
    )
    args = parser.parse_args()

    models = list(SUPPORTED_MODELS) if args.all_models else [args.model]
    resolutions = args.resolution or list(STANDARD_RESOLUTIONS)

    try:
        report = run_evaluation(
            models,
            resolutions,
            args.l1_kb,
            uniform_tile_size=args.uniform_tile,
            batch_size=args.batch_size,
            use_real_images=args.real_images,
            image_dir=args.image_dir,
            allow_dummy_fallback=args.allow_dummy_fallback,
            trials=args.trials,
            halo_min_out_channels=args.halo_min_out_channels,
            physical=args.physical,
            physical_dir=args.physical_dir,
            cachegrind=args.cachegrind,
            cachegrind_device=args.cachegrind_device,
        )
    except RuntimeError as exc:
        print(f"Evaluation aborted: {exc}")
        sys.exit(2)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")

    if not args.physical:
        print(
            "\n*** --no-physical: every peak-memory number above is an "
            "analytical estimate.\n"
            "    Nothing was compiled or executed. The estimator over-predicts "
            "the physical\n    peak by a median of 359% across the measured "
            "matrix; it orders configurations\n    correctly and does not "
            "quantify them. Re-run without --no-physical to measure."
        )

    if report["errors"]:
        print(f"Completed with {len(report['errors'])} error(s).")
        sys.exit(1)

    summary = report["summary"]
    print(
        f"\nSummary: {summary['configurations']} configs | "
        f"avg reduction {_pct(summary.get('avg_peak_memory_reduction_pct'))} "
        f"over {summary.get('reduction_measured_in', 0)}/"
        f"{summary.get('configurations', 0)} configs | "
        f"avg compile {summary.get('avg_compile_time_overhead_ms', 0)} ms"
    )


if __name__ == "__main__":
    main()
