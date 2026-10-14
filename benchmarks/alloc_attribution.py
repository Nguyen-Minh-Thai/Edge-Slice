"""Allocation attribution for the bufferized payload (proposal §5.2, Phase 0).

Answers one question: when a tiled payload's physical peak is *worse* than the
untiled one, which allocations account for the difference, and could a
bufferization pass have removed them?

Every allocation in the bufferized module is attributed to the region that
contains it, so the table separates

  * buffers at function scope — paid once,
  * buffers inside an ``scf.forall``/``scf.for`` body that are freed inside it
    — one tile live at a time, which is the intended shape,
  * buffers inside a loop body whose free is parked outside it — one buffer
    *per iteration*, which is the regression,

and reports for each whether ``--buffer-loop-hoisting`` could lift it out.
``memref.copy`` is counted alongside: a non-zero count means bufferization
failed to turn a ``tensor.extract_slice`` into a ``memref.subview`` and
inserted alloc+copy instead, which both adds a buffer and adds traffic the
cost model never priced.

The same three arms (untiled / tiled / tiled+fused) are run through several
deallocation and hoisting pass configurations, because the placement of frees
is what decides the peak — not the number of allocations.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from edgeslide.liveness.analyzer import (
    BufferizedMemoryAnalyzer,
    InferredSizeRefused,
)
from edgeslide.mlir_transform.errors import require_tool, run_tool
from edgeslide.mlir_transform.lowering import (
    decompose_tensor_concat,
    payload_needs_concat_decomposition,
    strip_transform_dialect,
    unwrap_redundant_module,
)

logger = logging.getLogger(__name__)

# Bufferization variants, as the pass list that follows one-shot bufferization.
# Keys are the arm labels used in result files and in the paper's Figure 2.
#
# `baseline` is what src/edgeslide/mlir_transform/lowering.py shipped: the
# deallocation pipeline plus --normalize-memrefs. The rest isolate one
# hypothesis each about where the tiled arm's extra memory comes from.
BUFFERIZE_VARIANTS: Dict[str, List[str]] = {
    # Shipped stage-1 configuration, minus the tiling/lowering passes that do
    # not affect allocation placement.
    "baseline": [
        "--buffer-deallocation-pipeline",
        "--normalize-memrefs",
    ],
    # H1: --normalize-memrefs rewrites layout maps and can defeat the
    # subview-instead-of-copy decision, turning slices into alloc+copy.
    "no_normalize": [
        "--buffer-deallocation-pipeline",
    ],
    # H2: the legacy pass sinks each free to its last use instead of parking it
    # at function exit. Decisive on flat IR; the question is whether scf.forall
    # regions block the sinking.
    "legacy_dealloc": [
        "--buffer-deallocation",
        "--normalize-memrefs",
    ],
    # H3: the ownership-based pass is the upstream replacement for both of the
    # above and places frees per-block rather than per-function.
    "ownership": [
        "--ownership-based-buffer-deallocation",
        "--canonicalize",
        "--buffer-deallocation-simplification",
        "--bufferization-lower-deallocations",
    ],
    # H4 (the fix under test): hoist loop-invariant allocations out of the loop
    # so one tile buffer is reused across iterations instead of N being
    # allocated, then promote what is left and small enough onto the stack.
    # Hoisting must run *before* deallocation: the dealloc passes place frees
    # against the allocation sites they find, so hoisting afterwards would move
    # an alloc away from its own free.
    "hoisted": [
        "--buffer-hoisting",
        "--buffer-loop-hoisting",
        "--promote-buffers-to-stack",
        "--buffer-deallocation-pipeline",
        "--normalize-memrefs",
    ],
    # H4 minus --normalize-memrefs, so the two effects can be separated.
    "hoisted_no_normalize": [
        "--buffer-hoisting",
        "--buffer-loop-hoisting",
        "--promote-buffers-to-stack",
        "--buffer-deallocation-pipeline",
    ],
    # Route A: the ownership pass plus the liveness refinement in mlir/, which
    # discharges each statically-owned buffer from the block-exit merged
    # dealloc at its last use. Expressed as a pipeline string rather than a
    # flag list because a pass loaded from a plugin is registered after
    # mlir-opt has already built its per-pass flag table, so `--pass-pipeline`
    # is the only way to name it. See PIPELINE_VARIANTS below.
    #
    # H4 on top of the ownership-based deallocation rather than the pipeline.
    "hoisted_ownership": [
        "--buffer-hoisting",
        "--buffer-loop-hoisting",
        "--promote-buffers-to-stack",
        "--ownership-based-buffer-deallocation",
        "--canonicalize",
        "--buffer-deallocation-simplification",
        "--bufferization-lower-deallocations",
    ],
}

# Variants whose passes cannot be named as individual mlir-opt flags. A pass
# loaded with --load-pass-plugin is registered while the command line is being
# parsed, which is after mlir-opt built the flag table its pass options live
# in, so the flag never exists; --pass-pipeline is parsed later and does see
# it. The string also has to nest explicitly -- the flag form auto-nests, the
# textual form does not -- which is why func.func appears here by hand.
PIPELINE_VARIANTS: Dict[str, str] = {
    "ownership_refined": (
        "builtin.module("
        "one-shot-bufferize{bufferize-function-boundaries=true},"
        "ownership-based-buffer-deallocation,"
        "canonicalize,"
        "edgeslide-liveness-refined-dealloc,"
        "buffer-deallocation-simplification,"
        "bufferization-lower-deallocations"
        ")"
    ),
}

# The ownership variant this one refines, so a report can put them side by side
# without the reader matching names.
PIPELINE_VARIANT_BASELINE = {"ownership_refined": "ownership"}

DEALLOC_PLUGIN_ENV = "EDGESLIDE_DEALLOC_PLUGIN"
_DEFAULT_PLUGIN = Path(__file__).resolve().parent.parent / "mlir" / "libEdgeSlideDealloc.so"


def dealloc_plugin_path() -> Path:
    """Absolute path to the liveness-refinement plugin, or an error.

    Raised rather than skipped: a variant that names the plugin and silently
    runs without it would report the unrefined ownership numbers under the
    refined label, which is exactly the class of silent downgrade this study
    exists to rule out. Build it with `bash mlir/build.sh`.
    """
    path = Path(os.environ.get(DEALLOC_PLUGIN_ENV, _DEFAULT_PLUGIN))
    if not path.is_file():
        raise ArmUnavailable(
            f"deallocation plugin not found at {path}. Build it with "
            f"`bash mlir/build.sh`, or set ${DEALLOC_PLUGIN_ENV}."
        )
    return path


# Variants that keep --normalize-memrefs, recorded so the runner can report the
# flag alongside the numbers rather than the reader inferring it from the name.
_NORMALIZES = {"baseline", "legacy_dealloc", "hoisted"}

ARMS = ("untiled", "tiled", "tiled_fused")


class ArmUnavailable(RuntimeError):
    """One arm of the comparison could not be produced.

    Raised rather than logged. Every number this module reports is a
    *difference* between arms, so a comparison missing an arm is not a partial
    result — and the attribution table renders regardless, which is what made
    a dropped arm invisible.
    """


@dataclass
class ArmResult:
    """One (arm, bufferization variant) cell of the attribution study."""

    arm: str
    variant: str
    ok: bool
    peak_dealloc_bytes: int = 0
    peak_last_use_bytes: int = 0
    totals: Dict[str, Any] = field(default_factory=dict)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    bufferized_path: Optional[str] = None
    error: str = ""

    def to_dict(self, *, max_rows: int = 25) -> Dict[str, Any]:
        return {
            "arm": self.arm,
            "variant": self.variant,
            "ok": self.ok,
            "normalize_memrefs": self.variant in _NORMALIZES,
            "peak_dealloc_bytes": self.peak_dealloc_bytes,
            "peak_dealloc_mb": round(self.peak_dealloc_bytes / (1024 * 1024), 3),
            "peak_last_use_bytes": self.peak_last_use_bytes,
            "peak_last_use_mb": round(self.peak_last_use_bytes / (1024 * 1024), 3),
            "totals": self.totals,
            "top_allocations": self.rows[:max_rows],
            "bufferized_path": self.bufferized_path,
            "error": self.error,
        }


def bufferize(
    input_path: Path,
    output_path: Path,
    variant: str,
    *,
    mlir_opt: Optional[str] = None,
) -> Path:
    """Bufferize *input_path* under one named pass variant."""
    if variant not in BUFFERIZE_VARIANTS and variant not in PIPELINE_VARIANTS:
        raise KeyError(f"unknown bufferization variant {variant!r}")
    tool = mlir_opt or require_tool("mlir-opt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # `tensor.concat` has no bufferization in MLIR 19, so a payload carrying one
    # (U-Net's skip connections) fails here rather than at lowering time. Same
    # decomposition the lowering path applies; a no-op for payloads without it.
    source = Path(input_path)
    text = source.read_text()
    if payload_needs_concat_decomposition(text):
        # decompose_tensor_concat strips and unwraps on its way out.
        prepared = decompose_tensor_concat(
            strip_transform_dialect(text),
            work_dir=output_path.parent,
            mlir_opt=tool,
        )
    else:
        # Unwrap for every other payload too. The transform interpreter leaves
        # a tiled payload one module deeper than it found it, and a func-level
        # pass silently skips a function nested that deep: it reports success
        # and inserts nothing. Four of the seven variants here are built from
        # func-level passes -- `legacy_dealloc`, `ownership`, and the three
        # hoisting passes in the `hoisted*` variants -- so without this their
        # columns measure a pass that never ran, which is how the tiled arms
        # came to look like tiling was the problem. See
        # docs/PHASE0_MEMORY_DIAGNOSIS.md.
        prepared = unwrap_redundant_module(strip_transform_dialect(text))
    if prepared != text:
        source = output_path.with_suffix(".prepared.mlir")
        source.write_text(prepared)

    if variant in PIPELINE_VARIANTS:
        # The pipeline string carries one-shot-bufferize itself, because
        # --pass-pipeline cannot be combined with individual pass flags.
        argv = [
            tool,
            str(source),
            f"--load-pass-plugin={dealloc_plugin_path()}",
            f"--pass-pipeline={PIPELINE_VARIANTS[variant]}",
            "-o",
            str(output_path),
        ]
    else:
        argv = [
            tool,
            str(source),
            "--one-shot-bufferize=bufferize-function-boundaries",
            *BUFFERIZE_VARIANTS[variant],
            "-o",
            str(output_path),
        ]
    run_tool(argv, stage=f"bufferize[{variant}]")
    return output_path


def attribute(
    arm: str,
    variant: str,
    payload_path: Path,
    work_dir: Path,
    *,
    mlir_opt: Optional[str] = None,
) -> ArmResult:
    """Bufferize one arm under one variant and attribute its allocations."""
    out = work_dir / f"{arm}.{variant}.bufferized.mlir"
    try:
        bufferize(payload_path, out, variant, mlir_opt=mlir_opt)
    except Exception as exc:  # a pass may not exist on this mlir-opt build
        return ArmResult(arm=arm, variant=variant, ok=False, error=str(exc)[:600])

    # Strict: this arm's peak gets reported, so an allocation whose extent
    # would have to be inferred must fail here rather than arrive as a number
    # indistinguishable from a measured one. Every payload measured so far is
    # statically shaped — 0 of the allocations in benchmarks/results/*.json are
    # flagged size_is_upper_bound — so this costs nothing today and catches the
    # first payload for which it would have started mattering.
    analyzer = BufferizedMemoryAnalyzer.from_bufferized_file(
        out, allow_inferred_sizes=False
    )
    try:
        report = analyzer.analyze()
    except (InferredSizeRefused, ValueError) as exc:
        # Two refusal paths, both meaning "this cell's size would be a guess":
        # InferredSizeRefused when an affine upper bound was derivable and
        # strict mode declined it, ValueError when no bound was derivable at
        # all. Recorded as an unavailable cell rather than raised, so one
        # dynamically-sized allocation does not discard the other twenty cells
        # of the sweep -- efficientnet_b0 at 224 has exactly one, in the fused
        # arm, out of 249.
        return ArmResult(arm=arm, variant=variant, ok=False, error=str(exc)[:600])
    return ArmResult(
        arm=arm,
        variant=variant,
        ok=True,
        peak_dealloc_bytes=report.get_peak()[1],
        peak_last_use_bytes=report.get_peak_last_use()[1],
        totals=report.attribution_totals(),
        rows=report.allocation_attribution(),
        bufferized_path=str(out),
    )


def build_arms(
    model: Any,
    input_shape: Tuple[int, ...],
    *,
    l1_cache_kb: int,
    work_dir: Path,
    model_name: str = "model",
) -> Dict[str, Path]:
    """Emit the three payload arms as tensor-level MLIR files.

    ``untiled`` is the payload straight out of torch-mlir; ``tiled`` runs the
    transform script with producer fusion disabled (pass-manager ablation D);
    ``tiled_fused`` is the shipped configuration. Returns arm → path, omitting
    any arm the compiler could not produce.
    """
    from edgeslide.mlir_transform.pipeline import (
        EdgeSlideMLIRConfig,
        EdgeSlideMLIRPipeline,
        EdgeSlideTransformPass,
        RunContext,
    )
    from edgeslide.transform.pass_manager import (
        EdgeSlicePassConfig,
        EdgeSlicePassManager,
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    config = EdgeSlideMLIRConfig(l1_cache_kb=l1_cache_kb, output_dir=str(work_dir))
    pipeline = EdgeSlideMLIRPipeline(config)

    parser, dag = pipeline.parse_model(model, input_shape)
    run = RunContext.create(str(work_dir), model_name, "arms")
    transform_pass = EdgeSlideTransformPass(config, run)

    mlir_path = parser.save_mlir(str(run.path(f"{model_name}.mlir")))
    payload_path = Path(transform_pass.write_payload_mlir(mlir_path))

    arms: Dict[str, Path] = {"untiled": payload_path}

    for arm, enable_fusion in (("tiled", False), ("tiled_fused", True)):
        # `enable_fusion` is only reachable through the config object; the
        # positional constructor has no keyword for it.
        pass_manager = EdgeSlicePassManager(
            dag,
            config=EdgeSlicePassConfig(
                l1_cache_bytes=l1_cache_kb * 1024,
                enable_fusion=enable_fusion,
            ),
        )
        results = pass_manager.run(transform_style="structured")
        transform_ir = results["transform_ir"]
        if "__transform_main" not in transform_ir or "No slicing candidates" in transform_ir:
            # "Nothing to tile at this budget" is a real outcome, not a
            # failure, but it must be raised rather than logged: a comparison
            # of untiled against tiled cannot be made from two arms when one of
            # them does not exist, and dropping the arm quietly leaves a table
            # that looks complete.
            raise ArmUnavailable(
                f"arm {arm!r}: the planner found no slicing candidates at "
                f"L1={l1_cache_kb} KB, so there is no tiled payload to compare "
                "against. Raise the resolution or lower the budget until the "
                "planner has something to tile."
            )
        transform_path = transform_pass.write_transform_mlir(
            transform_ir, name=f"transform_{arm}.mlir"
        )
        try:
            sliced = transform_pass.apply_transform(
                payload_path, transform_path, output_name=f"sliced_{arm}.mlir"
            )
        except Exception as exc:
            # Previously logged and skipped. A missing arm makes every
            # downstream comparison silently one-sided while the attribution
            # table still renders, so this has to stop the cell.
            raise ArmUnavailable(
                f"arm {arm!r}: the transform interpreter failed, so this arm "
                f"has no payload: {str(exc)[:400]}"
            ) from exc
        arms[arm] = Path(sliced)

    missing = set(ARMS) - set(arms)
    if missing:
        raise ArmUnavailable(f"arms not built: {sorted(missing)}")
    return arms


def run_study(
    model: Any,
    input_shape: Tuple[int, ...],
    *,
    l1_cache_kb: int,
    work_dir: Path,
    model_name: str = "model",
    variants: Sequence[str] = tuple(BUFFERIZE_VARIANTS) + tuple(PIPELINE_VARIANTS),
) -> Dict[str, Any]:
    """Full arm x variant sweep. Returns a JSON-serialisable result dict."""
    arms = build_arms(
        model,
        input_shape,
        l1_cache_kb=l1_cache_kb,
        work_dir=work_dir,
        model_name=model_name,
    )
    cells: List[Dict[str, Any]] = []
    for arm, path in arms.items():
        for variant in variants:
            result = attribute(arm, variant, path, work_dir)
            cells.append(result.to_dict())
            logger.info(
                "%s/%s: dealloc-peak %.1f MB, last-use peak %.1f MB",
                arm,
                variant,
                result.peak_dealloc_bytes / (1024 * 1024),
                result.peak_last_use_bytes / (1024 * 1024),
            )
    return {
        "model": model_name,
        # Which weights produced this. A bare model name does not distinguish
        # a published checkpoint from a randomly-initialised stand-in, and this
        # study accepts both (`--model tiny_conv` is a machinery check). Same
        # stamp the evaluation runs carry, so an artifact is self-describing
        # about whether it is evidence.
        "weights_provenance": _weights_provenance(model),
        "input_shape": list(input_shape),
        "l1_cache_kb": l1_cache_kb,
        "arms": {arm: str(path) for arm, path in arms.items()},
        "cells": cells,
    }


def _weights_provenance(model: Any) -> Dict[str, Any]:
    """Weight provenance for *model*, or why it could not be determined.

    Never silently omitted: an artifact with no provenance field and one whose
    provenance could not be read must not look the same.
    """
    try:
        from benchmarks.workloads import weights_provenance

        return weights_provenance(model)
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        return {
            "weights_source": "undeterminable",
            "is_pretrained": False,
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }


def format_table(study: Dict[str, Any]) -> str:
    """Render the study as the alloc-attribution table (paper Figure 2)."""
    header = (
        f"{'arm':<12} {'variant':<22} {'peak(dealloc)':>14} {'peak(last-use)':>15} "
        f"{'allocs':>7} {'in-loop':>8} {'hoist':>6} {'accum':>6} "
        f"{'copies':>7} {'copy-dst':>9} {'peak=sum':>9}"
    )
    lines = [
        f"=== Allocation attribution: {study['model']} "
        f"{tuple(study['input_shape'])} L1={study['l1_cache_kb']} KB ===",
        header,
        "-" * len(header),
    ]
    for cell in study["cells"]:
        if not cell["ok"]:
            lines.append(
                f"{cell['arm']:<12} {cell['variant']:<22} "
                f"{'FAILED: ' + cell['error'].splitlines()[0][:60]:>14}"
            )
            continue
        totals = cell["totals"]
        lines.append(
            f"{cell['arm']:<12} {cell['variant']:<22} "
            f"{cell['peak_dealloc_mb']:>13.1f}M {cell['peak_last_use_mb']:>14.1f}M "
            f"{totals['allocation_count']:>7} {totals['in_loop_count']:>8} "
            f"{totals['hoistable_count']:>6} {totals['accumulating_count']:>6} "
            f"{totals['copy_count']:>7} {totals['copy_destination_count']:>9} "
            f"{str(totals['dealloc_peak_equals_total']):>9}"
        )
    lines.append("")
    lines.append(
        "peak=sum True means every free is parked past the last allocation, so "
        "the dealloc peak is the sum of every allocation and carries no "
        "lifetime information. Read peak(last-use) as the achievable peak."
    )
    return "\n".join(lines)


def write_study(study: Dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(study, indent=2))
    return output
