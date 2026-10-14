"""
Lower sliced_payload.mlir (linalg/scf/tensor) to LLVM dialect / LLVM IR / JIT.

Execution paths:
  1. llvmlite MCJIT on mlir-translate LLVM IR (preferred; stable with system mlir-opt)
  2. CLI fallback: mlir-opt → mlir-translate → optional mlir-cpu-runner
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import torch

from edgeslide.mlir_transform.errors import (
    DeallocationUnavailableError,
    LLVMCompilationError,
    MLIRTransformError,
    MLIRVectorizationError,
    ToolNotFoundError,
    require_input_file,
    require_tool,
    run_tool,
)
from edgeslide.mlir_transform.vectorize_conv import outline_stride1_convs

logger = logging.getLogger(__name__)

RUNNER_UTILS_C = Path(__file__).with_name("runner_utils.c")

# mlir-opt passes for sliced payload (tensor → LLVM dialect).
# Deallocation is a correctness requirement, not an optimisation: without it
# every buffer the payload allocates leaks, so a repeatedly invoked entry point
# grows without bound.
#
# Placement then decides the *reported* peak, and on this pipeline it decides
# almost all of it. Measured on ResNet-50 (1x3x224x224, L1 256 KB) with
# scripts/phase0_alloc_attribution.py — dealloc-placement peak, and beside it
# the peak the same IR would reach if every buffer were freed at its last use:
#
#                        --buffer-deallocation  --buffer-deallocation-pipeline   last-use
#   untiled payload                    16.8 MB                        97.1 MB    16.8 MB
#   tiled (scf.forall)                 16.8 MB                       131.2 MB    16.8 MB
#   tiled + fused                      16.8 MB                       127.0 MB    16.8 MB
#
# CORRECTED. This table previously read 131.2 MB and 129.0 MB in the legacy
# column for the two tiled arms, and the text below concluded that scf.forall
# "blocks the sinking" and that the gap "is not closable by pass selection".
# All three statements were artifacts of the *measurement*, not properties of
# tiling:
#
#   `--buffer-deallocation` is a func-level pass, and the transform-interpreter
#   path leaves the payload wrapped in `module { module { func } } `. A
#   func-level pass silently skips a function nested that deep — it reported
#   success, changed nothing, and inserted 0 deallocations, so the peak
#   degenerated to the sum of every allocation. `--buffer-deallocation-pipeline`
#   is anchored on builtin.module and recurses, which is why only the legacy
#   column was affected.
#
#   Run `unwrap_redundant_module(strip_transform_dialect(...))` first and the
#   legacy pass reaches the 16.8 MB last-use optimum on *every* arm, tiled
#   included. Reproduced minimally on a single Conv2d(3->32, 3x3) at 64x64:
#   3 allocs, wrapped -> 0 deallocs; unwrapped -> 2 deallocs.
#
# Three things follow, and they are the Phase 0 diagnosis (see
# docs/PHASE0_MEMORY_DIAGNOSIS.md):
#
# 1. The pipeline parks 138 of 139 frees past the last allocation, so its
#    "peak" is exactly the sum of every allocation (verified: peak == total to
#    within a KB) and carries no lifetime information at all.
# 2. The achievable peak is 16.8 MB in every arm. Tiling does not regress
#    physical memory here — it does not change it. The 131.2 MB is the
#    allocation *count* rising (106 -> 139), not lifetimes lengthening.
# 3. The added allocations are copy destinations: 78 of the tiled arm's 139
#    allocations exist only to receive a `memref.copy`, against 41 of 106
#    untiled. `tensor.extract_slice` is becoming alloc+copy rather than
#    `memref.subview`.
#
# The legacy pass sinks each free to its last use, matching the achievable
# bound exactly on all three arms once the payload is unwrapped (above).
# Verified numerically equivalent end to end through the JIT on the untiled
# path (max abs error 9.3e-09); the tiled path's equivalence under this pass
# is the outstanding check before switching DEALLOCATION_PASS, since a
# deallocation pass that frees too early is wrong rather than efficient.
#
# `--buffer-hoisting`, `--buffer-loop-hoisting`, `--promote-buffers-to-stack`
# and `--ownership-based-buffer-deallocation` were all measured and none moves
# either number: the tiled arm has no allocations inside a loop body to hoist,
# and the 22 the fused arm does have are already freed in-region. Dropping
# `--normalize-memrefs` changes nothing either. The gap between 97-131 MB
# achieved and 16.8 MB achievable *is* closable by pass selection — see the
# correction above; what it needs first is unwrapping the payload.
#
# CORRECTED. This block used to say that since LLVM ~17
# `--buffer-deallocation` "is accepted but inserts nothing", citing tiny_conv
# measured at 3 allocs / 0 deallocs, and concluded the pipeline pass was the
# only working option. The measurement was real; the diagnosis was not. That
# payload was nested inside an extra module, and a func-level pass skips a
# function nested that deep -- silently, exit 0, no diagnostic. Unwrapped, the
# same tiny_conv gives 3 allocs / 2 deallocs (the third buffer is the result,
# which the caller owns and frees), and ResNet-50 gives 106 / 105.
#
# So the pass was never broken by a rename or an LLVM version. Reading it as
# broken is what put the 5.8x-worse pipeline pass in the shipped path.
#
# `--buffer-deallocation` is the older of the two deallocation algorithms in
# MLIR 19.1.1; `--ownership-based-buffer-deallocation` is the one
# `--buffer-deallocation-pipeline` is built from. Checkable difference: the
# legacy pass is declared `Pass<"buffer-deallocation", "func::FuncOp">` and the
# ownership pass is declared with no anchor. Neither carries a deprecation
# marker, so nothing here supports a claim that the legacy one is going away --
# but if an LLVM upgrade does remove it, DEALLOCATION_PASSES_FALLBACK below is
# what the lowering degrades to, and it does so loudly.
DEALLOCATION_PASS = "--buffer-deallocation-pipeline"
DEALLOCATION_PASS_LEGACY = "--buffer-deallocation"

# What the lowering actually runs. `--buffer-deallocation` places each free at
# its last use and reaches the achievable peak exactly; the default pipeline
# parks every free at function exit. The two cleanups after it are not
# optional:
#
#   Tiling bufferizes `tensor.parallel_insert_slice` into a copy of a subview
#   onto *the identical subview of the same buffer* -- one per scf.forall, 124
#   of them in ResNet-50. Lowered, each becomes `memcpy(p, p, 4)`, which is
#   undefined behaviour: valgrind reports it 10M times (its cap) on the tiled
#   arm. `--buffer-deallocation-pipeline` ends in `cse, canonicalize` and so
#   removes them by accident, which is the only reason the default path is
#   clean. `--cse` merges the two identical `memref.subview` ops into one
#   value and `--canonicalize` then folds the self-copy away.
#
# Measured end to end on ResNet-50 (1x3x224x224, L1 256 KB), two genuinely
# different binaries per arm, checksum identical (0.097950) in all four:
#
#                 IR peak   peak RSS   peak heap (massif)   valgrind
#   untiled
#     default     97.1 MiB   196.6 MiB        97.7 MiB      0 errors
#     this         16.8 MiB   111.5 MiB        17.4 MiB      0 errors
#   tiled
#     default    131.2 MiB   230.8 MiB       131.8 MiB      0 errors
#     this         16.8 MiB   111.5 MiB        17.4 MiB      0 errors
#
# 1.76x less resident memory untiled, 2.07x tiled; 5.6x and 7.6x less peak
# heap. The residual 94 MiB of RSS is the weight globals, which live in the
# binary's read-only data and are not heap (see PHASE1_METRIC_CORRECTION.md).
DEALLOCATION_PASSES = [DEALLOCATION_PASS_LEGACY, "--cse", "--canonicalize"]

# Where the lowering degrades to if `--buffer-deallocation` is ever missing.
# Correct -- it deallocates, so nothing leaks -- but it parks every free at
# function exit, which costs the 4.1x-7.7x peak-heap difference measured
# above. That is a different failure from leaking and needs a different flag:
# `allow_leaking_deallocation` covers a payload that never frees,
# `allow_degraded_deallocation` covers one that frees too late. Neither is on
# by default, because both silently change the number this project reports.
DEALLOCATION_PASSES_FALLBACK = [DEALLOCATION_PASS]

# The forward path: the ownership model plus the liveness refinement built from
# mlir/LivenessRefinedDealloc.cpp.
#
# `--buffer-deallocation` above is the legacy pass and reaches the achievable
# peak, but it is declared `Pass<"buffer-deallocation", "func::FuncOp">` --
# anchored at func.func, which is the property that made it silently skip a
# function nested one module deeper and report success. The ownership pass is
# declared with no anchor. (Neither is marked deprecated in MLIR 19.1.1; no
# claim is made here about what a later release does.) The ownership pass
# emits one merged `bufferization.dealloc` per block at the block's exit; a
# bufferized forward pass is a single basic block, so every buffer then lives
# until the function returns and the peak becomes the sum of every allocation --
# 4.1x-7.7x worse, measured. The refinement discharges from that merged dealloc
# every buffer whose ownership is statically held and whose liveness stays
# inside the block, placing each free at its last use, and leaves everything
# else where the ownership pass put it.
#
# Measured on ResNet-18 at 224x224: the stage-1 IR this produces is *byte
# identical* to what DEALLOCATION_PASSES produces, on both the untiled and the
# tiled arm. Across five models the analyzer peak matches the legacy pass in
# every arm and reaches the achievable last-use bound exactly. See
# docs/ROUTE_A_RESULTS.md.
REFINED_DEALLOCATION_PIPELINE = (
    "ownership-based-buffer-deallocation,"
    "canonicalize,"
    "edgeslide-liveness-refined-dealloc,"
    "buffer-deallocation-simplification,"
    "bufferization-lower-deallocations,"
    "cse,"
    "canonicalize"
)

# The refinement is loaded from a plugin, and a plugin pass is registered while
# mlir-opt is parsing its command line -- after it has already built the flag
# table that per-pass `--<name>` options live in. So no `--edgeslide-...` flag
# ever exists and the pass can only be named inside `--pass-pipeline`, which is
# parsed later. That in turn cannot be combined with individual pass flags, so
# the whole of stage 1 has to be rendered as one pipeline string.
#
# `--pass-pipeline` also does not auto-nest the way the flag form does: a
# func-level pass named at module level is an error rather than an implicit
# nesting. Each flag therefore needs its anchor stated, and a flag missing from
# this table is an error rather than a guess -- guessing the anchor wrong is
# how a pass silently ends up running on the wrong operation.
_PIPELINE_SPELLING: Dict[str, str] = {
    "--one-shot-bufferize=bufferize-function-boundaries":
        "one-shot-bufferize{bufferize-function-boundaries=true}",
    "--normalize-memrefs": "normalize-memrefs",
    "--scf-forall-to-for": "func.func(scf-forall-to-for)",
    "--scf-forall-to-parallel": "func.func(scf-forall-to-parallel)",
}

DEALLOC_PLUGIN_ENV = "EDGESLIDE_DEALLOC_PLUGIN"
_DEFAULT_DEALLOC_PLUGIN = (
    Path(__file__).resolve().parents[3] / "mlir" / "libEdgeSlideDealloc.so"
)


def refined_deallocation_plugin() -> Path:
    """Path to the built refinement plugin, or an error naming how to build it.

    Raised rather than skipped. A run that asked for the refinement and got the
    unrefined ownership placement instead would report a peak several times
    higher under the refined label, which is the same class of silent downgrade
    as a func-level pass quietly skipping a nested function.
    """
    path = Path(os.environ.get(DEALLOC_PLUGIN_ENV, _DEFAULT_DEALLOC_PLUGIN))
    if not path.is_file():
        raise DeallocationUnavailableError(
            f"the liveness-refined deallocation plugin is not built: {path} "
            f"does not exist. Build it with `bash mlir/build.sh`, or point "
            f"${DEALLOC_PLUGIN_ENV} at it. Running without it would place "
            "every free at the block exit and raise peak heap several fold, "
            "so it is refused rather than substituted."
        )
    return path


def refined_stage1_pipeline(passes: Sequence[str]) -> str:
    """Render *passes* as a --pass-pipeline string, refinement in place of the
    legacy deallocation group.

    Raises KeyError for a pass with no recorded anchor, so adding one to
    SLICED_PAYLOAD_OPT_PASSES_STAGE1 without recording how it nests fails here
    rather than producing a pipeline that runs it somewhere unintended.
    """
    rendered: List[str] = []
    index = 0
    passes = list(passes)
    width = len(DEALLOCATION_PASSES)
    while index < len(passes):
        if passes[index : index + width] == DEALLOCATION_PASSES:
            rendered.append(REFINED_DEALLOCATION_PIPELINE)
            index += width
            continue
        flag = passes[index]
        if flag not in _PIPELINE_SPELLING:
            raise KeyError(
                f"no --pass-pipeline spelling recorded for {flag!r}; add it to "
                "_PIPELINE_SPELLING in lowering.py together with the operation "
                "it must be nested under"
            )
        rendered.append(_PIPELINE_SPELLING[flag])
        index += 1
    if REFINED_DEALLOCATION_PIPELINE not in rendered:
        raise ValueError(
            "the deallocation group was not found in the stage-1 pass list, so "
            "the refinement would have nothing to replace"
        )
    return "builtin.module(" + ",".join(rendered) + ")"


def _with_deallocation_fallback(cmd: List[str]) -> Optional[List[str]]:
    """*cmd* with DEALLOCATION_PASSES swapped for the fallback, or None.

    None when the command does not contain the primary passes, so a caller
    cannot mistake "nothing to swap" for "swapped". The previous version
    substituted a single pass name that the command had stopped containing, so
    it always produced an unchanged command and its retry could never fire --
    while its error message still named the pass it had failed to substitute.
    """
    width = len(DEALLOCATION_PASSES)
    for index in range(len(cmd) - width + 1):
        if cmd[index : index + width] == DEALLOCATION_PASSES:
            return cmd[:index] + DEALLOCATION_PASSES_FALLBACK + cmd[index + width :]
    return None

# Split so the pipeline can be interrupted between stages: stage 1 produces
# the bufferized, tiling-lowered IR that `outline_stride1_convs` (see
# vectorize_conv.py) operates on; stage 2 finishes the lowering to LLVM
# dialect. `SLICED_PAYLOAD_OPT_PASSES` (stage1 + stage2) is the historical,
# always-scalar single-shot pipeline and remains the default.
# How the tiled `scf.forall` is taken to executable code. This is a scope
# decision, not a detail, and it was previously made implicitly.
#
# `transform.structured.tile_using_forall` emits `scf.forall`, which is a
# *parallel* loop: proposal §4.4's rationale for preferring the batch dimension
# ("maximizes parallelism across independent samples") is a statement about it.
# Three lowerings were measured on the ResNet-50 tiled+fused payload:
#
#   serial      --scf-forall-to-for                          113 forall -> scf.for
#   implicit    (nothing)                                    113 forall -> lowered
#                                                            anyway by
#                                                            --convert-scf-to-cf
#   parallel    --scf-forall-to-parallel --convert-scf-to-openmp
#                                                            678 omp.* ops
#
# Two things follow. First, `--scf-forall-to-for` is **not required**: the
# stage-2 pipeline lowers the payload with 0 remaining `scf.forall` whether or
# not it runs, so its presence was not load-bearing. Second, both of the
# default paths produce a sequential CFG, so §4.4's parallelism rationale is
# unsupported *on the default lowering* — the parallelism is expressed in the
# IR and then discarded. The OpenMP path preserves it and works, but it pulls
# in a runtime dependency and has not been evaluated for numerical equivalence
# across the workload matrix, so it stays opt-in.
#
# The pass is kept explicit rather than removed because "serialize here" is the
# honest description of what the default does, and a reader of the pipeline
# should see that decision rather than infer it from `--convert-scf-to-cf`.
FORALL_LOWERING_SERIAL: List[str] = ["--scf-forall-to-for"]
FORALL_LOWERING_PARALLEL: List[str] = ["--scf-forall-to-parallel"]

SLICED_PAYLOAD_OPT_PASSES_STAGE1: List[str] = [
    '--one-shot-bufferize=bufferize-function-boundaries',
    *DEALLOCATION_PASSES,
    "--normalize-memrefs",
    *FORALL_LOWERING_SERIAL,
]

SLICED_PAYLOAD_OPT_PASSES_STAGE2: List[str] = [
    "--convert-linalg-to-loops",
    # Vector lowering. No-ops for a payload that contains no vector ops, which
    # is every payload not run through `outline_stride1_convs`: MLIR 19's
    # `transform.structured.vectorize` refuses 2D convolutions (see
    # docs/EDGESLICE_TRANSFORM_ARCHITECTURE.md), so nothing upstream produces
    # them by default. Without these stages a payload that *did* vectorize
    # would carry vector ops into mlir-translate and fail there, far from the
    # cause.
    "--convert-vector-to-scf",
    "--lower-affine",
    "--convert-scf-to-cf",
    "--expand-strided-metadata",
    "--finalize-memref-to-llvm",
    "--lower-affine",
    "--convert-arith-to-llvm",
    "--convert-math-to-llvm",
    "--convert-index-to-llvm",
    "--convert-vector-to-llvm",
    "--convert-func-to-llvm",
    "--convert-cf-to-llvm",
    "--reconcile-unrealized-casts",
]

SLICED_PAYLOAD_OPT_PASSES: List[str] = (
    SLICED_PAYLOAD_OPT_PASSES_STAGE1 + SLICED_PAYLOAD_OPT_PASSES_STAGE2
)


def _stage2_vectorized_passes(vector_width: int) -> List[str]:
    """Stage 2 with the legacy affine vectorizer targeting stride-1 conv helpers.

    Inserted right after `--convert-linalg-to-loops`: by that point every
    `linalg.*` op `outline_stride1_convs` didn't touch has already become
    `scf.for` (opaque to `--affine-super-vectorize`, which only looks at
    `affine.for`), and the only `affine.for` loops left in the module are the
    hand-authored ones inside the outlined helper functions. So this can
    never reach an op we didn't deliberately choose to vectorize.

    `test-fastest-varying=0` is not a generic "innermost loop" selector — it's
    an index into an internal fastest-varying-pattern search, empirically
    verified (see vectorize_conv.py's module docstring for the JIT-execution
    check that caught this) to consistently select the output-width loop for
    the exact fixed 7-level loop nest `outline_stride1_convs` always
    generates. A different value on this same generator selected the
    output-*height* loop instead — extent 4 against a vector width of 8, an
    unsound access that MLIR's own lowering refused to finish rather than
    silently miscompiling. Do not change this value without re-verifying
    against a real payload via JIT execution, not just an opcount check.
    """
    passes = list(SLICED_PAYLOAD_OPT_PASSES_STAGE2)
    insert_at = passes.index("--convert-linalg-to-loops") + 1
    passes.insert(
        insert_at,
        f"--affine-super-vectorize=virtual-vector-size={vector_width} test-fastest-varying=0",
    )
    return passes

SLICED_PAYLOAD_PASS_PIPELINE = (
    "builtin.module("
    "one-shot-bufferize{bufferize-function-boundaries},"
    # Same rename as DEALLOCATION_PASS above — the old name is a silent no-op.
    "buffer-deallocation-pipeline,"
    "normalize-memrefs,"
    "scf-forall-to-for,"
    "func.func(convert-linalg-to-loops),"
    # Mirrors SLICED_PAYLOAD_OPT_PASSES; see the note there.
    "convert-vector-to-scf,"
    "lower-affine,"
    "func.func(convert-scf-to-cf),"
    "expand-strided-metadata,"
    "finalize-memref-to-llvm,"
    "lower-affine,"
    "convert-arith-to-llvm,"
    "convert-math-to-llvm,"
    "convert-index-to-llvm,"
    "convert-vector-to-llvm,"
    "convert-func-to-llvm,"
    "convert-cf-to-llvm,"
    "reconcile-unrealized-casts"
    ")"
)


def find_mlir_tool(name: str) -> Optional[str]:
    """Resolve an MLIR binary, or return None.

    Probe form: callers use the None to decide whether to skip work. Stages that
    require the tool call :func:`require_mlir_tool` instead, so a missing
    toolchain surfaces as ToolNotFoundError rather than a degraded result.
    """
    try:
        return require_tool(name)
    except ToolNotFoundError:
        return None


def require_mlir_tool(name: str, explicit: Optional[str] = None) -> str:
    """Resolve an MLIR binary or raise :class:`ToolNotFoundError`."""
    return require_tool(name, explicit)


def strip_transform_dialect(mlir_text: str) -> str:
    """Remove transform.named_sequence modules from post-interpreter IR."""
    lines = mlir_text.splitlines()
    kept: List[str] = []
    skip_depth = 0

    for line in lines:
        stripped = line.strip()
        if "transform.with_named_sequence" in stripped:
            skip_depth = 1
            continue
        if skip_depth > 0:
            skip_depth += line.count("{") - line.count("}")
            if skip_depth <= 0:
                skip_depth = 0
            continue
        kept.append(line)

    result = "\n".join(kept).strip()
    if not result:
        raise ValueError("No payload IR remaining after stripping transform dialect")
    return result + "\n"


def _find_function_body_brace(mlir_text: str, open_paren_index: int) -> int:
    """Index of the ``{`` opening the function body, scanning from the arg list.

    A regex cannot do this. Both an argument attribute dictionary
    (``%arg0: tensor<...> {torch.name = "x"}``) and the function's own
    ``attributes {...}`` dictionary contain braces, so taking the first ``{``
    would splice the new attribute into the argument list or duplicate the
    ``attributes`` keyword — either way producing IR that no longer parses.

    ``<``/``>`` are deliberately not tracked: they are balanced inside type
    syntax and never wrap a brace, while the ``>`` of ``->`` would otherwise
    unbalance the count.
    """
    depth = 0
    index = open_paren_index
    while index < len(mlir_text):
        char = mlir_text[index]
        if char == "{" and depth == 0:
            # Either the `attributes {...}` dict or the body. Only the former is
            # preceded by the keyword.
            if mlir_text[:index].rstrip().endswith("attributes"):
                depth += 1
            else:
                return index
        elif char in "({[":
            depth += 1
        elif char in ")}]":
            depth = max(0, depth - 1)
        index += 1
    return -1


def ensure_emit_c_interface(
    mlir_text: str,
    entry_func: str = "conv2d",
) -> str:
    """Add llvm.emit_c_interface so ExecutionEngine can invoke via _mlir_ciface_*."""
    header = re.search(rf"func\.func @{re.escape(entry_func)}\(", mlir_text)
    if header is None:
        raise ValueError(f"Could not locate func.func @{entry_func} in MLIR")

    body_brace = _find_function_body_brace(mlir_text, header.end() - 1)
    if body_brace < 0:
        raise ValueError(f"Could not locate the body of func.func @{entry_func}")

    signature = mlir_text[header.start() : body_brace]
    attrs_match = re.search(r"\s+attributes\s+\{([^{}]*)\}\s*$", signature)
    if attrs_match is not None:
        attrs = attrs_match.group(1).strip()
        if "llvm.emit_c_interface" in attrs:
            return mlir_text
        merged = f"llvm.emit_c_interface, {attrs}" if attrs else "llvm.emit_c_interface"
        head = signature[: attrs_match.start()]
    else:
        merged = "llvm.emit_c_interface"
        head = signature.rstrip()

    return (
        mlir_text[: header.start()]
        + f"{head} attributes {{{merged}}} "
        + mlir_text[body_brace:]
    )


def flatten_nested_module(mlir_text: str) -> str:
    """Hoist a single nested module body to the top-level module for JIT."""
    lines = mlir_text.splitlines()
    if not any("torch.debug_module_name" in line for line in lines):
        return mlir_text

    depth = 0
    start = None
    end = None
    for idx, line in enumerate(lines):
        if start is None and line.strip().startswith("module") and "{" in line:
            depth = line.count("{") - line.count("}")
            if depth > 0:
                start = idx
            continue
        if start is not None:
            depth += line.count("{") - line.count("}")
            if depth == 0:
                end = idx
                break

    if start is None or end is None:
        return mlir_text

    outer_prefix = lines[:start]
    body = lines[start + 1 : end]

    if (
        len(body) >= 2
        and body[0].strip().startswith("module")
        and "{" in body[0]
        and body[-1].strip() == "}"
    ):
        body = body[1:-1]

    flattened = outer_prefix + ["module {"] + body + ["}"] + lines[end + 1 :]
    return "\n".join(flattened) + "\n"


# LLVM 21 removed `mul` constant expressions, but mlir-translate (LLVM 19) still
# emits the `sizeof(T) * N` idiom for llvm.memcpy sizes. llvmlite bundles a newer
# LLVM than the system mlir-translate, so its parser rejects that IR with
# "mul constexprs are no longer supported". Only payloads large enough to emit a
# real llvm.memcpy hit this — small ones use the memrefCopy helper instead.
_LLVM_TYPE_SIZE_BYTES = {
    "i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8, "i128": 16,
    "half": 2, "bfloat": 2, "float": 4, "double": 8,
}

_SIZEOF_MUL_CONSTEXPR_RE = re.compile(
    r"mul\s*\(\s*i64\s+ptrtoint\s*\(\s*ptr\s+getelementptr\s*\(\s*"
    r"(?P<ty>[\w.]+)\s*,\s*ptr\s+null\s*,\s*i32\s+(?P<count>\d+)\s*\)\s*to\s+i64\s*\)\s*,"
    r"\s*i64\s+(?P<factor>\d+)\s*\)"
)


def fold_llvm_mul_constexprs(llvm_ir: str) -> str:
    """Fold `mul (sizeof(T), N)` constexprs into literals for newer LLVM parsers.

    Leaves unrecognized element types untouched so the parser still reports them
    rather than this function silently miscomputing a byte count.
    """

    def _fold(match: "re.Match[str]") -> str:
        elem_bytes = _LLVM_TYPE_SIZE_BYTES.get(match.group("ty"))
        if elem_bytes is None:
            return match.group(0)
        return str(elem_bytes * int(match.group("count")) * int(match.group("factor")))

    return _SIZEOF_MUL_CONSTEXPR_RE.sub(_fold, llvm_ir)


# `tensor.concat` has no bufferization implementation in MLIR 19: one-shot
# bufferize fails outright with "op was not bufferized", which takes the whole
# payload down. U-Net's skip connections are concats, so before this the U-Net
# workload could be parsed and planned but never compiled or executed — every
# U-Net number in the evaluation was a static estimate with no physical
# counterpart, and nothing said so.
#
# MLIR 19 does ship a decomposition, but only as a transform-dialect pattern
# (`transform.apply_patterns.tensor.decompose_concat`), not as a pass, so it has
# to be applied through the interpreter. It rewrites
#
#   tensor.concat dim(1) %a, %b
#
# into a `tensor.empty` plus one `tensor.insert_slice` per operand, which
# bufferizes normally. Verified end to end: U-Net at 64x64 and 96x96 now reach
# numerical equivalence through the JIT.
_DECOMPOSE_CONCAT_TRANSFORM = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.tensor.decompose_concat
    } : !transform.any_op
    transform.yield
  }
}
"""


def payload_needs_concat_decomposition(mlir_text: str) -> bool:
    """Whether *mlir_text* contains a ``tensor.concat`` that would fail to bufferize."""
    return "tensor.concat" in mlir_text


def decompose_tensor_concat(
    mlir_text: str,
    *,
    work_dir: Union[str, Path],
    mlir_opt: Optional[str] = None,
) -> str:
    """Rewrite ``tensor.concat`` into ``tensor.insert_slice`` form.

    Returns *mlir_text* unchanged when it holds no concat, so this is safe and
    free to call on every payload. Needs a scratch directory because the
    decomposition runs through ``mlir-opt --transform-interpreter``, which is
    file-to-file.

    The interpreter leaves the transform module it just ran nested in the
    output, so ``strip_transform_dialect`` is applied to the result — otherwise
    the next stage would try to lower a transform script.
    """
    if not payload_needs_concat_decomposition(mlir_text):
        return mlir_text

    tool = mlir_opt or require_tool("mlir-opt")
    directory = Path(work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    combined = directory / "payload_with_concat_decomposition.mlir"
    combined.write_text(mlir_text.rstrip("\n") + "\n" + _DECOMPOSE_CONCAT_TRANSFORM)
    decomposed = directory / "payload_concat_decomposed.mlir"

    _run_subprocess(
        [tool, str(combined), "--transform-interpreter", "-o", str(decomposed)],
        "mlir-opt decompose tensor.concat",
        MLIRTransformError,
    )
    return unwrap_redundant_module(strip_transform_dialect(decomposed.read_text()))


def unwrap_redundant_module(mlir_text: str) -> str:
    """Drop `module { module { ... } }` wrappers until only one level is left.

    Applied repeatedly, because levels accumulate one per interpreter run: a
    U-Net payload goes through the tiling interpreter *and* the concat
    decomposition, arriving three modules deep, and a single unwrap leaves it
    two deep -- still enough for a func-level pass to skip the function
    silently. That is why U-Net's tiled arms kept reporting
    `peak == sum of every allocation` after the single-level unwrap fixed the
    other four models.
    """
    previous = None
    text = mlir_text
    while previous != text:
        previous = text
        text = _unwrap_one_redundant_module(text)
    return text


def _unwrap_one_redundant_module(mlir_text: str) -> str:
    """Drop a single `module { module { ... } }` wrapper down to one level.

    A payload that is already a `module {...}` gains a level when a second
    top-level op is concatenated onto it: mlir-opt wraps the pair in an implicit
    outer module, and removing the transform module afterwards leaves the
    payload nested one deeper than it started.

    That nesting is not cosmetic. ``mlir-translate --mlir-to-llvmir`` exports
    the *outer* module, which by then contains nothing but the inner one, so it
    emits an LLVM module with **zero function definitions** — and the failure
    surfaces much later as "Unknown function symbol _mlir_ciface_forward" from
    the JIT, with nothing in between to suggest the module was empty.
    ``flatten_nested_module`` does not catch this case because it keys on a
    ``torch.debug_module_name`` attribute the interpreter's output no longer
    carries.

    Only a lone, immediately-nested module is removed. A module that holds two
    modules is real structure, and unwrapping it would splice their contents
    together.
    """
    lines = mlir_text.splitlines()

    def _module_extent(open_index: int) -> Optional[int]:
        """Index of the line closing the module opened at *open_index*."""
        depth = 0
        for index in range(open_index, len(lines)):
            depth += lines[index].count("{") - lines[index].count("}")
            if depth == 0:
                return index
        return None

    def _opens_module(line: str) -> bool:
        return line.strip().startswith("module") and line.rstrip().endswith("{")

    outer = next((i for i, line in enumerate(lines) if _opens_module(line)), None)
    if outer is None or outer + 1 >= len(lines) or not _opens_module(lines[outer + 1]):
        return mlir_text

    outer_close = _module_extent(outer)
    inner_close = _module_extent(outer + 1)
    if outer_close is None or inner_close is None:
        return mlir_text
    # The inner module must be the outer module's *only* content, or there is
    # more than one nested module and the nesting is meaningful.
    if inner_close != outer_close - 1:
        return mlir_text

    body = lines[outer + 2 : inner_close]
    dedented = [line[2:] if line.startswith("  ") else line for line in body]
    return "\n".join([*lines[:outer], lines[outer], *dedented, lines[outer_close], *lines[outer_close + 1 :]]) + "\n"


def prepare_sliced_payload_for_lowering(
    mlir_text: str,
    entry_func: str = "conv2d",
    emit_c_interface: bool = True,
) -> str:
    """Strip transform dialect, unwrap, and optionally tag the entry for JIT."""
    payload = strip_transform_dialect(mlir_text)
    # The transform interpreter leaves its payload nested one module deeper
    # than it found it, and a func-level pass silently skips a function nested
    # that deep -- it reports success and changes nothing. That is how
    # `--buffer-deallocation` came to insert 0 deallocations on every tiled
    # payload while exiting 0, and how `--buffer-hoisting`,
    # `--buffer-loop-hoisting` and `--promote-buffers-to-stack` came to be
    # measured as having no effect when they had never run.
    payload = unwrap_redundant_module(payload)
    if emit_c_interface:
        payload = ensure_emit_c_interface(payload, entry_func=entry_func)
    return payload


def _run_subprocess(
    cmd: Sequence[str], step: str, error_cls: type = LLVMCompilationError
) -> subprocess.CompletedProcess:
    """Run one lowering step, raising a typed error carrying its diagnostics."""
    return run_tool(cmd, stage=step, error_cls=error_cls)


def build_runner_shared_lib(output_dir: Union[str, Path]) -> str:
    """
    Build libedgeslide_runner_utils.so providing memrefCopy/malloc for JIT.

    Falls back to compiling runner_utils.c when system libmlir_runner_utils is absent.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = output_dir / "libedgeslide_runner_utils.so"

    for env_key in ("MLIR_RUNNER_UTILS", "MLIR_C_RUNNER_UTILS"):
        candidate = os.environ.get(env_key)
        if candidate and Path(candidate).is_file():
            return candidate

    for candidate in (
        "/usr/lib/llvm-19/lib/libmlir_runner_utils.so",
        "/usr/lib/llvm-19/lib/libmlir_c_runner_utils.so",
        "/usr/lib/libmlir_runner_utils.so",
    ):
        if Path(candidate).is_file():
            return candidate

    if not RUNNER_UTILS_C.is_file():
        raise FileNotFoundError(f"Missing runner utils source: {RUNNER_UTILS_C}")

    cmd = [
        "cc",
        "-shared",
        "-fPIC",
        "-O2",
        str(RUNNER_UTILS_C),
        "-o",
        str(lib_path),
    ]
    _run_subprocess(cmd, "compile runner utils")
    return str(lib_path)


@dataclass
class LoweringArtifacts:
    """Paths produced while lowering sliced payload MLIR."""
    prepared_mlir: str
    llvm_dialect_mlir: str
    llvm_ir: str
    shared_lib: Optional[str] = None


@dataclass
class SlicedPayloadLoweringConfig:
    output_dir: str = "build/mlir"
    entry_func: str = "conv2d"
    emit_c_interface: bool = True
    opt_level: int = 2
    mlir_opt: Optional[str] = None
    mlir_translate: Optional[str] = None
    shared_libs: List[str] = field(default_factory=list)
    # Opt-in: outline stride-1 linalg.conv_2d_nchw_fchw tiles into local-alloc
    # helper functions and vectorize them with the legacy affine vectorizer
    # (see vectorize_conv.py). Off by default — even though the fallback below
    # means a failure can't corrupt output, this is new enough (verified on
    # one real payload shape so far, not the full op/shape zoo) to keep opt-in
    # rather than silently changing every caller's default compile behavior.
    enable_stride1_vectorization: bool = False
    vector_width: int = 8
    # Keep the tiled `scf.forall`'s parallelism by lowering through
    # `scf.parallel` + OpenMP instead of serializing it to `scf.for`. Opt-in:
    # it adds a libomp dependency at link time and has not been checked for
    # numerical equivalence across the whole workload matrix. See the
    # FORALL_LOWERING_* note above the stage-1 pass list.
    parallelize_forall: bool = False
    # Whether a failed vectorized lowering may quietly produce scalar code.
    #
    # False by default, and that default is the point. The old behaviour was to
    # catch any failure, fall back to the scalar pipeline, and log a warning —
    # which is how a doubled `>` in every generated memref type went unnoticed:
    # `enable_stride1_vectorization=True` reported success while emitting
    # unparseable IR and compiling scalar code on every model. A caller that
    # asked for vectorization and got scalar has been given a different
    # artifact than the one it measured, so it must be told by an exception,
    # not a log line it will not read.
    allow_scalar_fallback: bool = False
    # Whether a payload that cannot get real deallocation may be compiled
    # anyway. False by default: see the retry site in `lower`.
    allow_leaking_deallocation: bool = False
    # Whether the lowering may fall back to `--buffer-deallocation-pipeline`
    # when the primary deallocation passes fail. That fallback is correct --
    # it frees -- but it frees at function exit rather than last use, which
    # cost 4.1x-7.7x peak heap across the measured models. Distinct from
    # `allow_leaking_deallocation`, which covers a payload that never frees at
    # all; this one covers a payload that frees too late.
    allow_degraded_deallocation: bool = False
    # Use the ownership model plus the liveness refinement instead of the
    # legacy `--buffer-deallocation`. Off by default only because it needs
    # mlir/libEdgeSlideDealloc.so built; the two produce byte-identical stage-1
    # IR on every arm measured so far (docs/ROUTE_A_RESULTS.md), and this one
    # stays inside the ownership model rather than depending on the func-level
    # legacy pass.
    use_refined_deallocation: bool = False

    def stage1_passes(self) -> List[str]:
        """Stage-1 pass list for this configuration."""
        if not self.parallelize_forall:
            return list(SLICED_PAYLOAD_OPT_PASSES_STAGE1)
        return [
            arg if arg not in FORALL_LOWERING_SERIAL else FORALL_LOWERING_PARALLEL[0]
            for arg in SLICED_PAYLOAD_OPT_PASSES_STAGE1
        ]

    def all_passes(self) -> List[str]:
        """Full single-shot pipeline for this configuration."""
        return self.stage1_passes() + list(SLICED_PAYLOAD_OPT_PASSES_STAGE2)


def _require_deallocations(
    llvm_text: str, path: Path, config: "SlicedPayloadLoweringConfig"
) -> None:
    """A payload that allocates must also free. Checked, not assumed.

    The old guard inferred this from *which pass name* was in the command
    line. That cannot see the failure it exists to prevent: a deallocation
    pass can run, exit 0, emit no diagnostic, and insert nothing.
    `--buffer-deallocation` does exactly that on a function nested inside an
    extra module -- which every transform-interpreter output was, so every
    tiled payload leaked every buffer while the guard reported success.
    Checking the emitted IR catches that case and any future one like it.
    """
    allocates = "@malloc" in llvm_text
    frees = "@free" in llvm_text
    if not allocates or frees:
        return
    message = (
        f"{path} allocates (@malloc) but never frees (@free): the "
        "deallocation pass ran without inserting anything, so this payload "
        "leaks every buffer it allocates and no peak-memory number taken "
        "from it measures anything. A func-level deallocation pass silently "
        "skips a function nested inside an extra module -- check that "
        "prepare_sliced_payload_for_lowering unwrapped the payload."
    )
    if not config.allow_leaking_deallocation:
        raise DeallocationUnavailableError(message)
    logger.warning("%s (allow_leaking_deallocation=True)", message)


class SlicedPayloadLowering:
    """Lower sliced_payload.mlir through LLVM dialect to LLVM IR."""

    def __init__(self, config: Optional[SlicedPayloadLoweringConfig] = None):
        self.config = config or SlicedPayloadLoweringConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def lower(self, sliced_payload_path: str) -> LoweringArtifacts:
        # Both binaries are resolved before any work starts: discovering the
        # translator is missing only after mlir-opt has written the LLVM
        # dialect file leaves a half-built output directory behind.
        mlir_opt = require_mlir_tool("mlir-opt", self.config.mlir_opt)
        mlir_translate = require_mlir_tool(
            "mlir-translate", self.config.mlir_translate
        )

        payload_path = require_input_file(sliced_payload_path, "sliced payload MLIR")
        mlir_text = payload_path.read_text()

        # Concat decomposition sits between the two halves of prepare, and the
        # order is load-bearing in both directions:
        #
        #   strip first — a sliced payload still carries the transform module
        #     the tiling interpreter ran, so appending a second
        #     `@__transform_main` makes the symbol ambiguous and the
        #     decomposition silently applies to nothing;
        #   emit_c_interface last — the interpreter's output nests the module it
        #     ran on, which loses an attribute applied beforehand (the JIT then
        #     reports "Unknown function symbol _mlir_ciface_forward").
        decomposed = decompose_tensor_concat(
            strip_transform_dialect(mlir_text),
            work_dir=self.output_dir,
            mlir_opt=mlir_opt,
        )
        prepared = prepare_sliced_payload_for_lowering(
            decomposed,
            entry_func=self.config.entry_func,
            emit_c_interface=self.config.emit_c_interface,
        )
        prepared_path = self.output_dir / "sliced_payload_prepared.mlir"
        prepared_path.write_text(prepared)

        llvm_dialect_path = self.output_dir / "sliced_payload_llvm.mlir"

        if self.config.enable_stride1_vectorization:
            self._lower_with_vectorization_attempt(mlir_opt, prepared_path, llvm_dialect_path)
        elif self.config.use_refined_deallocation:
            # Two invocations rather than one. The refinement is a plugin pass
            # and can only be named inside --pass-pipeline, which cannot be
            # combined with individual pass flags, so stage 1 goes through the
            # pipeline string and stage 2 keeps the flag form. No fallback: the
            # caller asked for this placement, and the alternative placements
            # differ by several times the peak.
            stage1_path = self.output_dir / "sliced_payload_stage1.mlir"
            _run_subprocess(
                [
                    mlir_opt,
                    str(prepared_path),
                    f"--load-pass-plugin={refined_deallocation_plugin()}",
                    f"--pass-pipeline={refined_stage1_pipeline(self.config.stage1_passes())}",
                    "-o",
                    str(stage1_path),
                ],
                "mlir-opt lowering (refined deallocation, stage 1)",
                MLIRTransformError,
            )
            _run_subprocess(
                [
                    mlir_opt,
                    str(stage1_path),
                    *SLICED_PAYLOAD_OPT_PASSES_STAGE2,
                    "-o",
                    str(llvm_dialect_path),
                ],
                "mlir-opt lowering (stage 2)",
                MLIRTransformError,
            )
        else:
            opt_cmd = [
                mlir_opt,
                str(prepared_path),
                *self.config.all_passes(),
                "-o",
                str(llvm_dialect_path),
            ]
            try:
                _run_subprocess(opt_cmd, "mlir-opt lowering", MLIRTransformError)
            except MLIRTransformError as exc:
                # The primary passes failed -- most likely an mlir-opt that has
                # dropped the legacy `--buffer-deallocation`. The fallback
                # still deallocates, so nothing leaks, but it parks every free
                # at function exit and the peak rises by the 4.1x-7.7x
                # measured above. A result produced that way is not comparable
                # with one produced normally, so it is refused unless asked
                # for.
                fallback_cmd = _with_deallocation_fallback(opt_cmd)
                if fallback_cmd is None or not self.config.allow_degraded_deallocation:
                    raise DeallocationUnavailableError(
                        f"{' '.join(DEALLOCATION_PASSES)} failed on this "
                        f"mlir-opt. Falling back to {DEALLOCATION_PASS} would "
                        "still deallocate, but it parks every free at function "
                        "exit, which raised peak heap by 4.1x-7.7x across the "
                        "measured models -- so its peak memory is not "
                        "comparable with any other result here. Install an "
                        "mlir-opt with --buffer-deallocation (LLVM 19 has it), "
                        "or set allow_degraded_deallocation=True to accept the "
                        "worse placement explicitly.\n"
                        f"{exc}"
                    ) from exc
                logger.warning(
                    "%s failed; falling back to %s. Buffers ARE freed, but at "
                    "function exit rather than last use, so this artifact's "
                    "peak memory is several times higher than normal and is "
                    "not comparable with other results "
                    "(allow_degraded_deallocation=True).",
                    " ".join(DEALLOCATION_PASSES),
                    DEALLOCATION_PASS,
                )
                _run_subprocess(fallback_cmd, "mlir-opt lowering", MLIRTransformError)

        llvm_text = llvm_dialect_path.read_text()
        _require_deallocations(llvm_text, llvm_dialect_path, self.config)
        flattened = flatten_nested_module(llvm_text)
        if flattened != llvm_text:
            llvm_dialect_path.write_text(flattened)

        llvm_ir_path = self.output_dir / "sliced_payload.ll"
        translate_cmd = [
            mlir_translate,
            "--mlir-to-llvmir",
            str(llvm_dialect_path),
            "-o",
            str(llvm_ir_path),
        ]
        _run_subprocess(translate_cmd, "mlir-translate")

        shared_lib = None
        if self.config.emit_c_interface:
            try:
                shared_lib = build_runner_shared_lib(self.output_dir)
            except (FileNotFoundError, RuntimeError) as exc:
                # Optional: only the JIT path links it, and lower() still
                # produced valid LLVM IR without it.
                logger.warning("runner utils unavailable (%s)", exc)

        logger.info("LLVM dialect: %s", llvm_dialect_path)
        logger.info("LLVM IR: %s", llvm_ir_path)

        return LoweringArtifacts(
            prepared_mlir=str(prepared_path),
            llvm_dialect_mlir=str(llvm_dialect_path),
            llvm_ir=str(llvm_ir_path),
            shared_lib=shared_lib,
        )

    def _run_stage1(self, mlir_opt: str, prepared_path: Path) -> str:
        """Run stage-1 passes (bufferize through scf-forall-to-for), return the text."""
        stage1_path = self.output_dir / "sliced_payload_stage1.mlir"
        cmd = [
            mlir_opt,
            str(prepared_path),
            *self.config.stage1_passes(),
            "-o",
            str(stage1_path),
        ]
        try:
            _run_subprocess(cmd, "mlir-opt lowering (stage 1)", MLIRTransformError)
        except MLIRTransformError:
            # Same degradation as in `lower`, same reason for not taking it
            # quietly: the fallback frees at function exit instead of last use.
            fallback_cmd = _with_deallocation_fallback(cmd)
            if fallback_cmd is None or not self.config.allow_degraded_deallocation:
                raise
            logger.warning(
                "%s failed on this mlir-opt; falling back to %s for stage 1. "
                "Buffers ARE freed, but at function exit, so this artifact's "
                "peak memory is several times higher than normal "
                "(allow_degraded_deallocation=True).",
                " ".join(DEALLOCATION_PASSES),
                DEALLOCATION_PASS,
            )
            _run_subprocess(fallback_cmd, "mlir-opt lowering (stage 1)", MLIRTransformError)
        return stage1_path.read_text()

    def _run_stage2(
        self, mlir_opt: str, stage_input_text: str, llvm_dialect_path: Path, passes: List[str], label: str
    ) -> None:
        stage_input_path = self.output_dir / "sliced_payload_stage2_input.mlir"
        stage_input_path.write_text(stage_input_text)
        cmd = [mlir_opt, str(stage_input_path), *passes, "-o", str(llvm_dialect_path)]
        _run_subprocess(cmd, label, MLIRTransformError)

    def _lower_with_vectorization_attempt(
        self, mlir_opt: str, prepared_path: Path, llvm_dialect_path: Path
    ) -> None:
        """Try the stride-1-vectorized pipeline; fall back to scalar on any failure.

        A failure here means the vectorized IR didn't fully lower to LLVM
        dialect for some op shape not yet exercised (see the module docstring
        in vectorize_conv.py for the one already found and worked around) —
        not a correctness risk, since a half-lowered module fails the
        `mlir-opt` subprocess outright rather than producing wrong output.
        The fallback re-runs the untouched, always-scalar stage 2 on the
        unmodified stage-1 text.
        """
        stage1_text = self._run_stage1(mlir_opt, prepared_path)

        outlined_text, outlined_count = outline_stride1_convs(stage1_text)
        if outlined_count > 0:
            outlined_text = flatten_nested_module(outlined_text)
            try:
                self._run_stage2(
                    mlir_opt,
                    outlined_text,
                    llvm_dialect_path,
                    _stage2_vectorized_passes(self.config.vector_width),
                    "mlir-opt lowering (stage 2, vectorized)",
                )
                logger.info(
                    "Vectorized %d stride-1 conv op(s) with the affine "
                    "vectorizer (width %d).",
                    outlined_count,
                    self.config.vector_width,
                )
                return
            except MLIRTransformError as exc:
                detail = str(exc)
                # A parse failure is a bug in what this repo generated; a
                # lowering failure is a limitation of the payload or of MLIR.
                # They need different words, because conflating them is what
                # let the generator stay broken.
                parse_failure = "expected" in detail or "invalid" in detail
                if parse_failure:
                    reason = (
                        f"the vectorized IR for {outlined_count} stride-1 "
                        "conv op(s) did not PARSE. That is a defect in "
                        "outline_stride1_convs, not a payload limitation: it "
                        "emitted malformed MLIR"
                    )
                else:
                    reason = (
                        f"the vectorized IR for {outlined_count} stride-1 "
                        "conv op(s) parsed but did not lower to LLVM dialect"
                    )
                if not self.config.allow_scalar_fallback:
                    raise MLIRVectorizationError(
                        f"enable_stride1_vectorization was requested but "
                        f"{reason}. Refusing to compile scalar code under a "
                        "vectorized label — a caller that measured this "
                        "artifact would attribute scalar behaviour to the "
                        "vectorized path. Set allow_scalar_fallback=True to "
                        "accept the downgrade explicitly.\n"
                        f"mlir-opt said: {detail[-800:]}"
                    ) from exc
                logger.warning(
                    "Falling back to the scalar pipeline: %s "
                    "(allow_scalar_fallback=True).",
                    reason,
                )

        self._run_stage2(
            mlir_opt,
            stage1_text,
            llvm_dialect_path,
            SLICED_PAYLOAD_OPT_PASSES_STAGE2,
            "mlir-opt lowering (stage 2)",
        )

    def lower_via_bindings(self, sliced_payload_path: str):
        """Lower using torch_mlir PassManager (Way 1 pipeline driver)."""
        from torch_mlir.ir import Context, Module
        from torch_mlir.passmanager import PassManager

        with open(sliced_payload_path) as f:
            mlir_text = f.read()

        prepared = prepare_sliced_payload_for_lowering(
            mlir_text,
            entry_func=self.config.entry_func,
            emit_c_interface=self.config.emit_c_interface,
        )

        with Context() as ctx:
            module = Module.parse(prepared, ctx)
            pm = PassManager.parse(SLICED_PAYLOAD_PASS_PIPELINE)
            pm.run(module.operation)
            return module


_LLVMLITE_INITIALIZED = False


def _init_llvmlite():
    """Initialise llvmlite's native target, tolerating either API generation.

    llvmlite < 0.45 requires the explicit ``initialize_*`` calls; 0.45 removed
    them (LLVM is initialised on import) and raises ``RuntimeError`` from the
    deprecated stubs. Both generations are in use across the supported
    environments, and 0.45 is the floor for reading MLIR 19's opaque-pointer
    output (``declare void @free(ptr)`` is a parse error on the LLVM 14 that
    llvmlite 0.43 bundles), so probe rather than pin behaviour.
    """
    global _LLVMLITE_INITIALIZED
    if _LLVMLITE_INITIALIZED:
        return
    import llvmlite.binding as llvm

    for name in ("initialize_native_target", "initialize_native_asmprinter"):
        init = getattr(llvm, name, None)
        if init is None:
            continue
        try:
            init()
        except RuntimeError as exc:  # llvmlite >= 0.45 deprecation stub
            if "deprecated" not in str(exc):
                raise
    _LLVMLITE_INITIALIZED = True


class _ExecutionEngineShim:
    """Backward-compatible shim for scripts that call jit.ee.raw_lookup()."""

    def __init__(self, jit: "SlicedPayloadJitEngine"):
        self._jit = jit

    def raw_lookup(self, name: str) -> int:
        return self._jit._lookup_symbol(name)


class SlicedPayloadJitEngine:
    """
    JIT execute lowered sliced payload via llvmlite MCJIT on translated LLVM IR.

    Uses mlir-opt/mlir-translate (LLVM 19) for lowering, then llvmlite to run the
    generated IR with memrefCopy/malloc from runner utils. torch_mlir ExecutionEngine
    is avoided because it can segfault on IR produced by a newer system mlir-opt.
    """

    def __init__(
        self,
        llvm_ir_path: str,
        opt_level: int = 2,
        shared_libs: Optional[Sequence[str]] = None,
        output_dir: Union[str, Path] = "build/mlir",
    ):
        import llvmlite.binding as llvm

        _init_llvmlite()

        libs = list(shared_libs or [])
        if not libs:
            libs = [build_runner_shared_lib(output_dir)]

        for lib in libs:
            llvm.load_library_permanently(lib)

        mod = llvm.parse_assembly(
            fold_llvm_mul_constexprs(Path(llvm_ir_path).read_text())
        )
        triple = llvm.get_process_triple()
        target = llvm.Target.from_triple(triple)
        target_machine = target.create_target_machine(opt=opt_level)

        self._mcjit = llvm.create_mcjit_compiler(mod, target_machine)
        self._mcjit.__enter__()
        self._mcjit.finalize_object()

        self._result_holder = None
        self._func_cache: Dict[tuple, Any] = {}
        self.ee = _ExecutionEngineShim(self)

    def _lookup_symbol(self, symbol: str) -> int:
        ptr = self._mcjit.get_function_address(symbol)
        return int(ptr or 0)

    @staticmethod
    def _ciface_func_type(num_args: int):
        """ctypes signature for _mlir_ciface_*: void(result, arg0, arg1, ...)."""
        return ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            *[ctypes.c_void_p] * num_args,
        )

    def _get_ciface_func(self, symbol: str, num_args: int):
        cache_key = (symbol, num_args)
        if cache_key not in self._func_cache:
            ptr = self._lookup_symbol(symbol)
            if not ptr:
                raise RuntimeError(f"Unknown function symbol {symbol}")
            self._func_cache[cache_key] = self._ciface_func_type(num_args)(ptr)
        return self._func_cache[cache_key]

    @classmethod
    def from_sliced_payload(
        cls,
        sliced_payload_path: str,
        config: Optional[SlicedPayloadLoweringConfig] = None,
        llvm_dialect_path: Optional[str] = None,
        llvm_ir_path: Optional[str] = None,
    ) -> "SlicedPayloadJitEngine":
        """
        Build JIT engine from sliced payload.

        Lowering uses mlir-opt CLI (torch_mlir may not register scf-forall-to-for),
        then MCJIT-compiles the translated LLVM IR via llvmlite.
        """
        cfg = config or SlicedPayloadLoweringConfig()
        lowering = SlicedPayloadLowering(cfg)

        if llvm_ir_path and Path(llvm_ir_path).is_file():
            ir_path = llvm_ir_path
        else:
            if llvm_dialect_path and Path(llvm_dialect_path).is_file():
                artifacts = LoweringArtifacts(
                    prepared_mlir="",
                    llvm_dialect_mlir=llvm_dialect_path,
                    llvm_ir=str(Path(cfg.output_dir) / "sliced_payload.ll"),
                )
                if not Path(artifacts.llvm_ir).is_file():
                    mlir_translate = require_mlir_tool(
                        "mlir-translate", cfg.mlir_translate
                    )
                    _run_subprocess(
                        [
                            mlir_translate,
                            "--mlir-to-llvmir",
                            llvm_dialect_path,
                            "-o",
                            artifacts.llvm_ir,
                        ],
                        "mlir-translate",
                    )
            else:
                artifacts = lowering.lower(sliced_payload_path)
            ir_path = artifacts.llvm_ir

        shared_libs = list(cfg.shared_libs)
        if not shared_libs:
            try:
                shared_libs = [build_runner_shared_lib(cfg.output_dir)]
            except (FileNotFoundError, RuntimeError):
                shared_libs = []

        return cls(
            ir_path,
            opt_level=cfg.opt_level,
            shared_libs=shared_libs,
            output_dir=cfg.output_dir,
        )

    def __del__(self):
        mcjit = getattr(self, "_mcjit", None)
        if mcjit is not None:
            try:
                mcjit.__exit__(None, None, None)
            except Exception:
                pass

    @staticmethod
    def _torch_dtype_to_ctype(dtype):
        import torch

        mapping = {
            torch.float32: ctypes.c_float,
            torch.float64: ctypes.c_double,
            torch.int32: ctypes.c_int32,
            torch.int64: ctypes.c_int64,
        }
        if dtype not in mapping:
            raise TypeError(f"Unsupported torch dtype for memref: {dtype}")
        return mapping[dtype]

    def _tensor_to_memref(self, tensor) -> ctypes.Structure:
        """Describe *tensor* to the C interface as a ranked memref.

        The descriptor holds a raw pointer, so whatever owns that memory must
        outlive it. ``.contiguous()`` / ``ascontiguousarray`` return a *copy*
        for a non-dense input, and that copy would otherwise be owned only by
        a local here — the descriptor then points into freed memory and the
        kernel reads garbage, producing wrong results with no crash. Each
        returned descriptor therefore carries a reference to its buffer.
        """
        from torch_mlir.runtime import get_ranked_memref_descriptor, make_nd_memref_descriptor

        def _owning(descriptor: ctypes.Structure, owner) -> ctypes.Structure:
            descriptor._edgeslide_buffer_owner = owner
            return descriptor

        if hasattr(tensor, "detach"):
            tensor = tensor.detach().contiguous().cpu()
            try:
                array = tensor.numpy()
                # `array` shares the (now dense) tensor's storage; keeping the
                # tensor alive keeps the buffer alive.
                return _owning(get_ranked_memref_descriptor(array), (tensor, array))
            except RuntimeError:
                ctype = self._torch_dtype_to_ctype(tensor.dtype)
                desc_type = make_nd_memref_descriptor(tensor.ndim, ctype)
                desc = desc_type()
                data_ptr = tensor.data_ptr()
                desc.allocated = data_ptr
                desc.aligned = ctypes.cast(data_ptr, ctypes.POINTER(ctype))
                desc.offset = 0
                for i, size in enumerate(tensor.shape):
                    desc.shape[i] = int(size)
                for i, stride in enumerate(tensor.stride()):
                    desc.strides[i] = int(stride)
                return _owning(desc, tensor)

        array = np.asarray(tensor)
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        return _owning(get_ranked_memref_descriptor(array), array)

    @staticmethod
    def _memref_to_torch(result_holder, dtype=None):
        import torch

        dtype = dtype or torch.float32
        ctype = SlicedPayloadJitEngine._torch_dtype_to_ctype(dtype)
        rank = len(result_holder.shape)
        numel = 1
        shape = []
        for i in range(rank):
            dim = int(result_holder.shape[i])
            shape.append(dim)
            numel *= dim

        out = torch.empty(shape, dtype=dtype)
        src_ptr = ctypes.cast(result_holder.aligned, ctypes.POINTER(ctype))
        dst_ptr = ctypes.cast(out.data_ptr(), ctypes.POINTER(ctype))

        if rank == 0:
            dst_ptr[0] = src_ptr[int(result_holder.offset)]
            return out

        if tuple(result_holder.strides[:rank]) == tuple(out.stride()):
            ctypes.memmove(dst_ptr, src_ptr, numel * ctypes.sizeof(ctype))
            return out

        # Fallback for non-contiguous result layouts.
        coords = [0] * rank
        while True:
            linear = int(result_holder.offset)
            for dim in range(rank):
                linear += coords[dim] * int(result_holder.strides[dim])
            flat_idx = 0
            multiplier = 1
            for dim in range(rank - 1, -1, -1):
                flat_idx += coords[dim] * multiplier
                multiplier *= shape[dim]
            dst_ptr[flat_idx] = src_ptr[linear]
            dim = rank - 1
            while dim >= 0 and coords[dim] + 1 >= shape[dim]:
                coords[dim] = 0
                dim -= 1
            if dim < 0:
                break
            coords[dim] += 1
        return out

    def invoke(
        self,
        entry_func: str,
        *tensors,
        result_rank: Optional[int] = None,
        result_dtype: str = "f32",
    ):
        """Invoke JIT entry function with PyTorch/NumPy tensors as memref arguments.

        ``result_rank``/``result_dtype`` describe the entry function's result
        memref (see ``parse_entry_result_spec``). Without an explicit rank the
        result descriptor falls back to the first input's rank — correct for
        conv-shaped nets only; a mismatched rank corrupts ``result_holder``.
        """
        from torch_mlir.runtime import make_nd_memref_descriptor, ranked_memref_to_numpy

        entry_func = entry_func.split("@")[-1]
        if len(tensors) < 1:
            raise ValueError("invoke() expects at least one input tensor")

        arg_descs = [ctypes.pointer(self._tensor_to_memref(tensor)) for tensor in tensors]
        if result_rank is None:
            input_tensor = tensors[0]
            result_rank = (
                len(input_tensor.shape)
                if hasattr(input_tensor, "shape")
                else input_tensor.ndim
            )
        result_ctype = _MEMREF_CTYPE_BY_MLIR_DTYPE.get(result_dtype)
        if result_ctype is None:
            raise TypeError(f"Unsupported MLIR result dtype: {result_dtype}")
        result_type = make_nd_memref_descriptor(int(result_rank), result_ctype)
        result_holder = result_type()
        self._result_holder = result_holder

        symbol = f"_mlir_ciface_{entry_func}"
        func = self._get_ciface_func(symbol, len(tensors))
        func(
            ctypes.cast(ctypes.pointer(result_holder), ctypes.c_void_p),
            *[ctypes.cast(desc, ctypes.c_void_p) for desc in arg_descs],
        )
        try:
            return ranked_memref_to_numpy(ctypes.pointer(result_holder))
        except Exception as exc:
            import torch

            logger.debug(
                "ranked_memref_to_numpy failed (%s); falling back to manual "
                "strided memref copy",
                exc,
            )
            torch_dtype = torch_dtype_for_mlir_dtype(result_dtype, torch.float32)
            return self._memref_to_torch(result_holder, dtype=torch_dtype)


_MEMREF_CTYPE_BY_MLIR_DTYPE = {
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
}


def torch_dtype_for_mlir_dtype(mlir_dtype: str, default=None):
    """Look up a torch dtype lazily so importing this module never needs torch."""
    import torch

    mapping = {
        "f32": torch.float32,
        "f64": torch.float64,
        "i32": torch.int32,
        "i64": torch.int64,
    }
    return mapping.get(mlir_dtype, default if default is not None else torch.float32)


def parse_entry_result_spec(
    mlir_text: str, entry_func: str
) -> Optional[Tuple[int, str]]:
    """Extract (rank, dtype) of the entry function's ranked tensor/memref result.

    Returns None when the signature cannot be parsed (e.g. multiple results,
    unranked types) — callers should then fall back to input-rank heuristics.
    """
    match = re.search(
        rf"func\.func @{re.escape(entry_func)}\((?:[^()]|\([^()]*\))*\)\s*"
        rf"->\s*(?:tensor|memref)<([^>]*)>",
        mlir_text,
    )
    if not match:
        return None
    # "1x8x32x32xf32" (possibly "..., layout" for memrefs) → dims + dtype.
    parts = match.group(1).split(",", 1)[0].strip().split("x")
    dtype = parts[-1].strip()
    if not dtype or any(ch in dtype for ch in "<>?*"):
        return None
    return len(parts) - 1, dtype


def detect_entry_func(mlir_text: str, preferred: Optional[str] = None) -> str:
    """Detect the payload entry function name from MLIR text."""
    if preferred:
        if re.search(rf"func\.func @{re.escape(preferred)}\b", mlir_text):
            return preferred

    matches = re.findall(r"func\.func @(\w+)", mlir_text)
    for name in matches:
        if name != "__transform_main" and not name.startswith("_"):
            return name
    if matches:
        return matches[0]
    raise ValueError("No entry function found in MLIR module")


class SlicedPayloadCliRunner:
    """CLI fallback execution via mlir-cpu-runner when bindings are unavailable."""

    def __init__(self, config: Optional[SlicedPayloadLoweringConfig] = None):
        self.config = config or SlicedPayloadLoweringConfig()
        self.lowering = SlicedPayloadLowering(self.config)

    def run_cpu_runner(
        self,
        llvm_dialect_path: str,
        entry_func: Optional[str] = None,
        runner_libs: Optional[Sequence[str]] = None,
    ) -> subprocess.CompletedProcess:
        runner = require_mlir_tool("mlir-cpu-runner")

        entry_func = entry_func or self.config.entry_func
        libs = list(runner_libs or [])
        if not libs:
            libs = [build_runner_shared_lib(self.config.output_dir)]

        cmd = [
            runner,
            llvm_dialect_path,
            f"-e={entry_func}",
            "-entry-point-result=void",
            f"-shared-libs={','.join(libs)}",
        ]
        return _run_subprocess(cmd, "mlir-cpu-runner")

    def lower_and_translate(self, sliced_payload_path: str) -> LoweringArtifacts:
        return self.lowering.lower(sliced_payload_path)


def mlir_output_to_torch(output) -> "torch.Tensor":
    """Convert JIT memref/numpy output to a torch.Tensor."""
    import torch

    if isinstance(output, torch.Tensor):
        return output.float()
    return torch.from_numpy(np.asarray(output)).float()


def execute_sliced_mlir_jit(
    sliced_payload_path: str,
    input_tensor,
    entry_func: Optional[str] = None,
    output_dir: Union[str, Path] = "build/mlir",
) -> "torch.Tensor":
    """
    Lower sliced_payload.mlir and execute via ExecutionEngine.

    Returns:
        torch.Tensor result from the JIT-compiled entry function.
    """
    output_dir = Path(output_dir)
    sliced_text = Path(sliced_payload_path).read_text()
    entry_func = entry_func or detect_entry_func(sliced_text, preferred="forward")

    config = SlicedPayloadLoweringConfig(
        output_dir=str(output_dir),
        entry_func=entry_func,
    )
    artifacts = SlicedPayloadLowering(config).lower(sliced_payload_path)
    shared_libs = [artifacts.shared_lib] if artifacts.shared_lib else []

    jit = SlicedPayloadJitEngine.from_sliced_payload(
        sliced_payload_path,
        config=SlicedPayloadLoweringConfig(
            output_dir=str(output_dir),
            entry_func=entry_func,
            shared_libs=shared_libs,
        ),
        llvm_ir_path=artifacts.llvm_ir,
    )
    result_spec = parse_entry_result_spec(sliced_text, entry_func)
    if result_spec is not None:
        result = jit.invoke(
            entry_func,
            input_tensor,
            result_rank=result_spec[0],
            result_dtype=result_spec[1],
        )
    else:
        result = jit.invoke(entry_func, input_tensor)
    return mlir_output_to_torch(result)


def lower_sliced_payload(
    sliced_payload_path: str,
    output_dir: str = "build/mlir",
    entry_func: str = "conv2d",
) -> Dict[str, str]:
    """
    Convenience API: sliced_payload.mlir → LLVM dialect + LLVM IR.

    Returns dict with keys: prepared_mlir, llvm_dialect_mlir, llvm_ir, shared_lib.
    """
    config = SlicedPayloadLoweringConfig(
        output_dir=output_dir,
        entry_func=entry_func,
    )
    artifacts = SlicedPayloadLowering(config).lower(sliced_payload_path)
    results = {
        "prepared_mlir": artifacts.prepared_mlir,
        "llvm_dialect_mlir": artifacts.llvm_dialect_mlir,
        "llvm_ir": artifacts.llvm_ir,
    }
    if artifacts.shared_lib:
        results["shared_lib"] = artifacts.shared_lib
    return results