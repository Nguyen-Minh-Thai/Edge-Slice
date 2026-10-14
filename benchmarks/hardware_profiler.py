#!/usr/bin/env python3
"""
Hardware validation via Valgrind Cachegrind on native MLIR binaries.

Workflow:
  1. Load a small PyTorch model (TinyConv / SimpleCNN).
  2. Compile payload.mlir (original) and sliced_payload.mlir via EdgeSlide pipeline.
  3. Sanity-check sliced IR (scf.forall + tiled conv when expected).
  4. Lower each to LLVM IR and link a C harness (mlir_bench_main.c).
  5. Run cachegrind / wall-clock on both binaries.
  6. Write benchmarks/results/hardware_validation.json (or pi4 path).

Requires: torch, torch_mlir, mlir-opt, mlir-translate, cc, valgrind.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edgeslide.mlir_transform.lowering import (  # noqa: E402
    RUNNER_UTILS_C,
    SlicedPayloadLowering,
    SlicedPayloadLoweringConfig,
    _init_llvmlite,
    detect_entry_func,
    find_mlir_tool,
    fold_llvm_mul_constexprs,
)
from edgeslide.mlir_transform.pipeline import (  # noqa: E402
    EdgeSlideMLIRConfig,
    EdgeSlideMLIRPipeline,
)

BENCH_MAIN_C = Path(__file__).with_name("mlir_bench_main.c")
RESULTS_PATH = ROOT / "benchmarks" / "results" / "hardware_validation.json"
PI4_RESULTS_PATH = ROOT / "benchmarks" / "results" / "pi4_hardware_validation.json"

TENSOR_SHAPE_RE = re.compile(
    r"func\.func @(\w+)\([^)]*tensor<([^>]+)>\s*\)\s*->\s*tensor<([^>]+)>"
)
CACHEGRIND_D_REFS_RE = re.compile(r"D\s+refs:\s+([\d,]+)")
CACHEGRIND_D1_MISSES_RE = re.compile(r"D1\s+misses:\s+([\d,]+)")
CACHEGRIND_LL_MISSES_RE = re.compile(r"LL\s+misses:\s+([\d,]+)")
# Matches the bench binary's stdout line, including glibc's non-finite
# rendering of %.6f ("nan"/"-nan"/"inf"): Python's float() accepts all of
# those tokens. The iters group lets the harness verify the binary honored
# the requested per-launch iteration count.
BENCH_OUTPUT_RE = re.compile(
    r"mlir_bench_checksum=([-+]?(?:nan|inf|[0-9][0-9.eE+-]*|\.[0-9][0-9.eE+-]*))"
    r"\s+iters=(\d+)"
)

# Cachegrind models: size, associativity, line size (bytes)
# Raspberry Pi 4 Cortex-A72: L1D 32KB, L1I ~48KB, line 64B
DEVICE_CACHEGRIND: Dict[str, Dict[str, Any]] = {
    "generic": {
        "i1": "32768,8,64",
        "d1": "262144,4,64",  # legacy 256KB edge SoC model
        "default_l1_cache_kb": 256,
        "description": "Generic / legacy 256KB D1 model",
    },
    "pi4": {
        "i1": "49152,3,64",
        "d1": "32768,2,64",
        "default_l1_cache_kb": 32,
        "description": "Raspberry Pi 4 (Cortex-A72) L1D≈32KB",
    },
}


def small_model_provenance() -> Dict[str, Any]:
    """What a `hardware_validation.json` is measured on, stated in the file.

    SMALL_MODELS are hand-written kernels with random weights, existing so a
    pipeline change can be validated in seconds. Their docstrings have always
    said no number from them is benchmark evidence — but a docstring does not
    travel with the JSON, and that file otherwise looks exactly like a
    benchmark result. The evaluated workloads are the published-weight models
    in workloads.py, whose runs carry `weights_provenance` instead.
    """
    return {
        "kind": "synthetic_machinery_kernels",
        "pretrained_weights": False,
        "is_benchmark_evidence": False,
        "models": sorted(SMALL_MODELS),
        "note": (
            "Hand-written conv kernels with random initialisation. Use these "
            "to check that the compiler pipeline still works, never to "
            "characterise a workload."
        ),
    }


class TinyConv(nn.Module):
    """Single Conv+ReLU — fast sanity model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class SimpleCNN(nn.Module):
    """Two-layer CNN with out_channels >= 64 to trigger EdgeSlice tiling."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthwiseConv(nn.Module):
    """Pointwise expand (groups=1) + depthwise conv (groups=channels).

    Every other SMALL_MODELS entry uses a plain (groups=1) conv. This is the
    smallest model that exercises a grouped conv through the DAGParser/
    torch-mlir lowering pipeline — the op MobileNetV2 and EfficientNet-B0 use
    throughout, and which this pipeline had never compiled before.

    Synthetic on purpose, and only ever a machinery check: SMALL_MODELS exists
    so a pipeline change can be validated in seconds. No result reported from
    these is a benchmark number — the evaluated workloads are the real
    pretrained models in benchmarks/workloads.py, and their weight digests are
    recorded per run.
    """

    def __init__(self) -> None:
        super().__init__()
        self.expand = nn.Conv2d(3, 64, kernel_size=1)
        self.depthwise = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.depthwise(self.relu(self.expand(x))))


# Edge-oriented workload matrix (Option 2 / Pi 4 plan).
SMALL_MODELS: Dict[str, Tuple[type, Tuple[int, ...]]] = {
    "tiny_conv": (TinyConv, (1, 3, 256, 256)),
    "tiny_conv_512": (TinyConv, (1, 3, 512, 512)),
    "simple_cnn": (SimpleCNN, (1, 3, 256, 256)),
    "simple_cnn_512": (SimpleCNN, (1, 3, 512, 512)),
    "depthwise_conv": (DepthwiseConv, (1, 3, 256, 256)),
    # Same op, small spatial extent: at 256x256 a single channel's full H*W
    # plane (~266KB padded) already exceeds the default 256KB L1 budget on
    # its own, so channel-dim tiling can never fit regardless of priority
    # order — only a shape this small (mirrors a deep MobileNetV2/
    # EfficientNet stage with a wide channel count) is small enough
    # spatially for both channel and spatial tiling to fit L1, which is what
    # actually lets dimension_selector's depthwise channel-first priority
    # change the outcome. 64x64, not smaller: the untiled output
    # (1*64*64*64*4 = 1MiB) must exceed the default 256KB L1 budget, or
    # pass_manager's candidate filter skips tiling this op entirely as
    # "already fits L1" before dimension selection ever runs (verified: 32x32
    # hits exactly this — its 256KiB output equals the budget itself).
    "depthwise_conv_64": (DepthwiseConv, (1, 3, 64, 64)),
}


@dataclass
class CachegrindStats:
    d_refs: int
    d1_misses: int
    ll_misses: Optional[int] = None
    cachegrind_out_file: Optional[str] = None
    stderr_excerpt: str = ""


@dataclass
class WallClockStats:
    """Per-iteration wall-clock stats.

    Each trial launches the binary once with ``iters_per_trial`` kernel
    invocations and divides the elapsed time by that count, so process
    startup is amortized instead of dominating single-kernel timings.
    """

    trials: int
    mean_s: float
    std_s: float
    min_s: float
    max_s: float
    checksum: Optional[float] = None
    max_rss_kb: Optional[int] = None
    iters_per_trial: int = 1


@dataclass
class BinaryProfile:
    label: str
    binary_path: str
    llvm_ir_path: str
    stats: CachegrindStats
    wall_clock: Optional[WallClockStats] = None


@dataclass
class IRSanityReport:
    has_scf_forall: bool
    forall_count: int
    has_conv: bool
    conv_inside_forall: bool
    tile_using_forall_in_transform: bool
    notes: List[str] = field(default_factory=list)

    @property
    def ok_for_measure(self) -> bool:
        """Require tiling structure; prefer conv inside forall when conv exists."""
        if not self.has_scf_forall:
            return False
        if self.has_conv and not self.conv_inside_forall:
            return False
        return True


class UnequalVectorization(RuntimeError):
    """The two binaries were not vectorized to the same degree.

    A difference between them then measures the vectorizer as well as the
    tiling, and the two cannot be separated after the fact. Raised rather than
    reported, because the number it would produce answers a question nobody
    asked.
    """


class NumericallyInvalidBinary(RuntimeError):
    """The sliced binary does not compute the same result as the original.

    Its cache-traffic and wall-clock numbers describe a different computation,
    so they are not a faster version of the same work and must not appear
    beside numbers that are.
    """


@dataclass
class ModelHardwareReport:
    model_name: str
    input_shape: List[int]
    l1_cache_kb: int
    entry_func: str
    original: BinaryProfile
    sliced: BinaryProfile
    # None when cachegrind did not run (--skip-cachegrind). Previously 0.0,
    # which is a real-looking number for a measurement that never happened —
    # and it landed in the JSON indistinguishable from a genuine 0% result.
    traffic_reduction_pct: Optional[float]
    cachegrind_skipped: bool = False
    # SIMD types in the emitted LLVM IR, equal across both arms by
    # construction (see UnequalVectorization). 0 means a scalar comparison.
    simd_types_emitted: int = 0
    ir_sanity: Optional[IRSanityReport] = None
    payload_mlir: Optional[str] = None
    sliced_payload_mlir: Optional[str] = None
    transform_mlir: Optional[str] = None


def _parse_tensor_shape(type_str: str) -> Tuple[int, ...]:
    """Parse '1x3x32x32xf32' → (1, 3, 32, 32)."""
    parts = type_str.split("x")
    dims: List[int] = []
    for part in parts:
        if part and part[0].isdigit():
            try:
                dims.append(int(part))
            except ValueError:
                break
        else:
            break
    if not dims:
        raise ValueError(f"Could not parse tensor shape from '{type_str}'")
    return tuple(dims)


def parse_entry_and_shapes(
    mlir_text: str, preferred: str = "forward"
) -> Tuple[str, Tuple[int, ...], Tuple[int, ...]]:
    entry_func = detect_entry_func(mlir_text, preferred=preferred)
    match = TENSOR_SHAPE_RE.search(mlir_text)
    if not match:
        raise ValueError(f"Could not parse entry tensor shapes for @{entry_func}")
    if match.group(1) != entry_func and preferred not in mlir_text:
        pattern = re.compile(
            rf"func\.func @{re.escape(entry_func)}\([^)]*tensor<([^>]+)>\s*\)\s*->\s*tensor<([^>]+)>"
        )
        m2 = pattern.search(mlir_text)
        if not m2:
            raise ValueError(f"No shape signature for @{entry_func}")
        in_shape = _parse_tensor_shape(m2.group(1))
        out_shape = _parse_tensor_shape(m2.group(2))
        return entry_func, in_shape, out_shape

    in_shape = _parse_tensor_shape(match.group(2))
    out_shape = _parse_tensor_shape(match.group(3))
    return entry_func, in_shape, out_shape


def sanitize_sliced_payload_for_lowering(mlir_text: str) -> str:
    """Drop leftover transform modules from interpreter output if present."""
    from edgeslide.mlir_transform.lowering import strip_transform_dialect

    return strip_transform_dialect(mlir_text)


def analyze_sliced_ir(
    sliced_mlir_text: str,
    transform_mlir_text: Optional[str] = None,
) -> IRSanityReport:
    """Check that EdgeSlice actually produced a tiled payload (PR-C gate)."""
    text = sanitize_sliced_payload_for_lowering(sliced_mlir_text)
    forall_count = len(re.findall(r"\bscf\.forall\b", text))
    has_conv = bool(
        re.search(r"linalg\.conv_2d|linalg\.depthwise_conv", text, re.IGNORECASE)
    )
    notes: List[str] = []

    # Heuristic: if a conv appears after an scf.forall open and before its close,
    # treat as "conv inside forall". Also accept extract_slice feeding conv inside loop.
    conv_inside = False
    if has_conv and forall_count > 0:
        # Split on scf.forall regions (brace-depth naive on lines).
        depth = 0
        in_forall = False
        for line in text.splitlines():
            if "scf.forall" in line and "in_parallel" not in line:
                in_forall = True
                depth = line.count("{") - line.count("}")
                if "linalg.conv" in line:
                    conv_inside = True
                continue
            if in_forall:
                depth += line.count("{") - line.count("}")
                if re.search(r"linalg\.conv_2d|linalg\.depthwise_conv", line):
                    conv_inside = True
                if depth <= 0:
                    in_forall = False
        if not conv_inside:
            notes.append(
                "conv ops present but none found inside scf.forall — "
                "likely transform.match used wrong op name (see rewrite._op_type_to_mlir)"
            )
    elif not has_conv:
        notes.append("no linalg.conv in sliced payload (elementwise-only tile may be OK)")

    tile_in_transform = False
    if transform_mlir_text:
        tile_in_transform = "tile_using_forall" in transform_mlir_text or (
            "tile_using_for" in transform_mlir_text
        )

    if forall_count == 0:
        notes.append("no scf.forall in sliced payload — tiling did not apply")

    return IRSanityReport(
        has_scf_forall=forall_count > 0,
        forall_count=forall_count,
        has_conv=has_conv,
        conv_inside_forall=conv_inside if has_conv else True,
        tile_using_forall_in_transform=tile_in_transform,
        notes=notes,
    )


def assert_ir_sane_for_measure(
    report: IRSanityReport,
    *,
    strict: bool = True,
    model_name: str = "",
) -> None:
    prefix = f"[{model_name}] " if model_name else ""
    for note in report.notes:
        print(f"[ir-sanity] {prefix}{note}")
    if not strict:
        return
    if not report.ok_for_measure:
        raise RuntimeError(
            f"{prefix}IR sanity gate failed: forall={report.has_scf_forall} "
            f"count={report.forall_count} has_conv={report.has_conv} "
            f"conv_inside={report.conv_inside_forall}. Notes: {report.notes}"
        )
    print(
        f"[ir-sanity] {prefix}OK forall={report.forall_count} "
        f"conv_inside_forall={report.conv_inside_forall}"
    )


def count_simd_types(llvm_ir_path: Path) -> int:
    """How many vector types the emitted LLVM IR contains.

    Read from the LLVM IR because it is the only place the evidence survives:
    MLIR's own ``vector.*`` ops are consumed by ``--convert-vector-to-llvm``
    before lowering ends, so counting them there always reads zero whether or
    not vectorization happened.
    """
    return len(re.findall(r"<\d+ x (?:float|half|double)>", llvm_ir_path.read_text()))


def llvm_ir_to_object(llvm_ir_path: Path, object_path: Path, opt_level: int = 2) -> None:
    """Compile LLVM IR text to a native object file via llvmlite (no clang required).

    Both the constexpr folding and the initialisation probe are shared with the
    JIT path rather than reimplemented: mlir-translate 19 emits
    ``mul (i64 ptrtoint (...), i64 N)`` for llvm.memcpy sizes, which LLVM 22 —
    what llvmlite 0.49 bundles — rejects outright ("mul constexprs are no
    longer supported"). Only payloads big enough to emit a real llvm.memcpy hit
    it, which is why it showed up first on a full model rather than in the unit
    tests, and it showed up here and not in the JIT because the JIT already
    folded and this path did not.
    """
    import llvmlite.binding as llvm

    _init_llvmlite()

    mod = llvm.parse_assembly(
        fold_llvm_mul_constexprs(llvm_ir_path.read_text())
    )
    triple = llvm.get_process_triple()
    target = llvm.Target.from_triple(triple)
    target_machine = target.create_target_machine(opt=opt_level, reloc="pic")
    mod.triple_ = triple
    object_path.write_bytes(target_machine.emit_object(mod))


def compile_mlir_artifacts(
    mlir_path: Path,
    work_dir: Path,
    label: str,
    *,
    enable_vectorization: bool = False,
    vector_width: int = 8,
    use_refined_deallocation: bool = False,
) -> Tuple[str, Tuple[int, ...], Tuple[int, ...], str, str]:
    """Lower MLIR → LLVM IR → native binary with mlir_bench_main.c.

    ``enable_vectorization`` threads ``SlicedPayloadLoweringConfig.
    enable_stride1_vectorization`` (see lowering.py) through to both the
    "original" and "sliced" compiles, so a wall-clock comparison stays
    apples-to-apples: whichever stride-1 convs each payload has get the same
    outline-and-vectorize treatment either way. It is opt-in and off by
    default, mirroring the config's own default (verified on one real
    payload shape so far, not the full op/shape zoo) — see
    ``--enable-vectorization`` on the CLI.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    mlir_text = mlir_path.read_text()
    # Sliced interpreter output may still embed transform modules.
    cleaned = sanitize_sliced_payload_for_lowering(mlir_text)
    if cleaned.strip() != mlir_text.strip():
        cleaned_path = work_dir / f"{label}_payload_clean.mlir"
        cleaned_path.write_text(cleaned + "\n")
        mlir_path = cleaned_path
        mlir_text = cleaned

    entry_func, in_shape, out_shape = parse_entry_and_shapes(mlir_text)

    lowering = SlicedPayloadLowering(
        SlicedPayloadLoweringConfig(
            output_dir=str(work_dir),
            entry_func=entry_func,
            opt_level=2,
            enable_stride1_vectorization=enable_vectorization,
            vector_width=vector_width,
            use_refined_deallocation=use_refined_deallocation,
        )
    )
    artifacts = lowering.lower(str(mlir_path))
    runner_obj = work_dir / "runner_utils.o"
    _run(
        ["cc", "-c", "-O2", "-fPIC", str(RUNNER_UTILS_C), "-o", str(runner_obj)],
        "compile runner_utils.o",
    )

    if in_shape[0] != out_shape[0]:
        raise ValueError(f"Batch mismatch: input {in_shape} vs output {out_shape}")
    # Rank is passed to the harness rather than fixed at 4. Every classifier in
    # the evaluation matrix ends in a rank-2 logits tensor, so a rank-4-only
    # harness silently excluded ResNet-50, MobileNetV2 and EfficientNet from
    # every RSS and cachegrind measurement.
    if not in_shape or not out_shape:
        raise ValueError(f"Empty tensor shape: input {in_shape} → output {out_shape}")

    binary_path = work_dir / f"{label}_{entry_func}"
    object_path = work_dir / f"{label}_{entry_func}.o"
    llvm_ir_to_object(Path(artifacts.llvm_ir), object_path, opt_level=2)

    def _sizes_macro(shape: Tuple[int, ...]) -> str:
        return "{" + ",".join(str(int(dim)) for dim in shape) + "}"

    # Statically linked when asked. The measurement target (a Pi 4, glibc 2.36
    # on Raspberry Pi OS bookworm) is older than this build host (2.39), so a
    # dynamically linked binary built here refuses to start there. Both are
    # aarch64 and the payload is baseline ARMv8-A -- no -mcpu=native anywhere --
    # so a static binary is portable between them. Off by default because it
    # costs ~600 KB and makes nothing better on the build host itself.
    static_link = os.environ.get("EDGESLICE_STATIC_LINK") == "1"
    compile_cmd = [
        "cc",
        "-O2",
        *(["-static"] if static_link else []),
        "-DNUM_ITERS=1",
        f"-DIN_RANK={len(in_shape)}",
        f"-DIN_SIZES={_sizes_macro(in_shape)}",
        f"-DOUT_RANK={len(out_shape)}",
        f"-DOUT_SIZES={_sizes_macro(out_shape)}",
        f"-DMLIR_ENTRY_NAME={entry_func}",
        str(BENCH_MAIN_C),
        str(object_path),
        str(runner_obj),
        "-o",
        str(binary_path),
        "-lm",
    ]
    _run(compile_cmd, f"link {label} binary")

    if not binary_path.is_file():
        raise RuntimeError(f"Binary not produced: {binary_path}")

    return str(binary_path), in_shape, out_shape, entry_func, artifacts.llvm_ir


def _timeout_from_env(var: str, default_s: int) -> int:
    """Seconds for a subprocess timeout, overridable via *var*.

    One policy for every timeout in this file: bounded so a wedged run fails
    loudly instead of hanging the job, and raisable via env var for very slow
    targets (e.g. Pi 4) without editing source.
    """
    raw = os.environ.get(var)
    if raw is None:
        return default_s
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(
            f"{var} must be an integer number of seconds, got {raw!r}"
        ) from None
    if value <= 0:
        raise RuntimeError(f"{var} must be a positive number of seconds, got {value}")
    return value


def _run(cmd: List[str], step: str) -> subprocess.CompletedProcess:
    print(f"[hardware_profiler] {step}: {' '.join(cmd)}")
    timeout_s = _timeout_from_env("EDGESLIDE_STEP_TIMEOUT_S", 600)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{step} timed out after {timeout_s}s; "
            "set EDGESLIDE_STEP_TIMEOUT_S to raise the limit"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"{step} failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    return result


def _parse_cachegrind_int(pattern: re.Pattern, text: str, label: str) -> int:
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not find {label} in cachegrind output")
    return int(match.group(1).replace(",", ""))


def find_valgrind() -> Optional[str]:
    override = os.environ.get("VALGRIND")
    if override and Path(override).is_file():
        return override
    found = shutil.which("valgrind")
    if found:
        return found
    for candidate in (
        "/tmp/valgrind_extracted/usr/bin/valgrind",
        "/usr/bin/valgrind",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


def _valgrind_env(valgrind_bin: str) -> Dict[str, str]:
    """Ensure VALGRIND_LIB is set when using an extracted .deb tree."""
    env = os.environ.copy()
    if env.get("VALGRIND_LIB"):
        return env
    valgrind_path = Path(valgrind_bin).resolve()
    for libexec in (
        valgrind_path.parent.parent / "libexec" / "valgrind",
        Path("/tmp/valgrind_extracted/usr/libexec/valgrind"),
    ):
        if (libexec / "cachegrind-arm64-linux").is_file() or any(
            libexec.glob("cachegrind-*")
        ):
            env["VALGRIND_LIB"] = str(libexec)
            break
    return env


def run_cachegrind(
    binary_path: str,
    work_dir: Path,
    *,
    device: str = "generic",
) -> CachegrindStats:
    valgrind = find_valgrind()
    if not valgrind:
        raise FileNotFoundError(
            "valgrind not found. Install with: sudo apt install valgrind "
            "(or set VALGRIND=/path/to/valgrind)"
        )

    cfg = DEVICE_CACHEGRIND.get(device, DEVICE_CACHEGRIND["generic"])
    cg_file = work_dir / f"{Path(binary_path).name}.cachegrind"
    cmd = [
        valgrind,
        "--tool=cachegrind",
        "--cache-sim=yes",
        f"--I1={cfg['i1']}",
        f"--D1={cfg['d1']}",
        f"--cachegrind-out-file={cg_file}",
        binary_path,
    ]

    # Cachegrind slows execution 20-100x, hence its own (larger) default.
    timeout_s = _timeout_from_env("EDGESLIDE_CACHEGRIND_TIMEOUT_S", 7200)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=_valgrind_env(valgrind),
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"cachegrind timed out after {timeout_s}s on {binary_path}; "
            "set EDGESLIDE_CACHEGRIND_TIMEOUT_S to raise the limit"
        ) from exc
    combined = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"cachegrind failed on {binary_path} (exit {result.returncode}):\n"
            f"{combined}"
        )

    d_refs = _parse_cachegrind_int(CACHEGRIND_D_REFS_RE, combined, "D refs")
    d1_misses = _parse_cachegrind_int(CACHEGRIND_D1_MISSES_RE, combined, "D1 misses")
    ll_misses: Optional[int] = None
    ll_match = CACHEGRIND_LL_MISSES_RE.search(combined)
    if ll_match:
        ll_misses = int(ll_match.group(1).replace(",", ""))

    excerpt = "\n".join(
        line for line in combined.splitlines() if "refs:" in line or "misses:" in line
    )
    return CachegrindStats(
        d_refs=d_refs,
        d1_misses=d1_misses,
        ll_misses=ll_misses,
        cachegrind_out_file=str(cg_file) if cg_file.is_file() else None,
        stderr_excerpt=excerpt,
    )


def run_wall_clock(
    binary_path: str,
    *,
    trials: int = 5,
    warmup: int = 1,
    pin_cpu: Optional[int] = 0,
    iters_per_trial: int = 8,
) -> WallClockStats:
    """Native wall-clock timing (edge latency, not cachegrind).

    The binary accepts an iteration count as argv[1]; running several kernel
    invocations per launch amortizes process startup, and the recorded times
    are per-iteration.
    """
    times: List[float] = []
    checksum: Optional[float] = None
    max_rss = 0
    if iters_per_trial < 1:
        raise ValueError(f"iters_per_trial must be >= 1, got {iters_per_trial}")
    trial_timeout_s = _timeout_from_env("EDGESLIDE_WALLCLOCK_TIMEOUT_S", 1800)

    def _cmd(iters: int) -> List[str]:
        cmd = [binary_path, str(iters)]
        if pin_cpu is not None and shutil.which("taskset"):
            return ["taskset", "-c", str(pin_cpu), *cmd]
        return cmd

    for i in range(warmup + trials):
        # RSS of this process is not the binary's; use /usr/bin/time when available.
        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                _cmd(iters_per_trial),
                capture_output=True,
                text=True,
                timeout=trial_timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"wall-clock trial timed out after {trial_timeout_s}s "
                f"({binary_path}); set EDGESLIDE_WALLCLOCK_TIMEOUT_S to raise the limit"
            ) from exc
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            raise RuntimeError(
                f"wall-clock run failed ({binary_path}): {result.stderr}"
            )
        m = BENCH_OUTPUT_RE.search(result.stdout)
        if not m:
            raise RuntimeError(
                f"could not parse benchmark output of {binary_path}; "
                f"stdout was: {result.stdout!r}"
            )
        checksum = float(m.group(1))
        reported_iters = int(m.group(2))
        if reported_iters != iters_per_trial:
            # Dividing elapsed by the *requested* count below is only valid if
            # the binary honored argv[1]; a stale mlir_bench_main.c ignores it.
            raise RuntimeError(
                f"{binary_path} reported iters={reported_iters} but "
                f"{iters_per_trial} were requested; rebuild the benchmark "
                "binary from the current benchmarks/mlir_bench_main.c"
            )
        if i >= warmup:
            times.append(elapsed / iters_per_trial)

    # Best-effort max RSS via GNU time. One iteration suffices: peak RSS is
    # reached during the first kernel invocation (one input + one live result).
    if shutil.which("time") or Path("/usr/bin/time").is_file():
        time_bin = "/usr/bin/time" if Path("/usr/bin/time").is_file() else "time"
        try:
            tr = subprocess.run(
                [time_bin, "-f", "%M", *(_cmd(1))],
                capture_output=True,
                text=True,
                timeout=trial_timeout_s,
            )
            # GNU time writes to stderr
            for line in reversed((tr.stderr or "").strip().splitlines()):
                line = line.strip()
                if line.isdigit():
                    max_rss = int(line)
                    break
        except (OSError, subprocess.TimeoutExpired):
            pass

    if not times:
        raise RuntimeError("no wall-clock trials recorded")

    return WallClockStats(
        trials=len(times),
        mean_s=statistics.mean(times),
        std_s=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_s=min(times),
        max_s=max(times),
        checksum=checksum,
        max_rss_kb=max_rss or None,
        iters_per_trial=iters_per_trial,
    )


# Slicing reorders the floating-point summation, so bit-equality is too
# strict; matches metrics.numerical_equivalence's default relative tolerance.
CHECKSUM_REL_TOL = 1e-3


def checksum_rel_diff(original: Optional[float], sliced: Optional[float]) -> Optional[float]:
    """Relative difference of the two binaries' checksums, or None if unknown."""
    if original is None or sliced is None:
        return None
    denom = max(abs(original), abs(sliced), 1e-12)
    return abs(original - sliced) / denom


def traffic_reduction_pct(d1_original: int, d1_sliced: int) -> float:
    if d1_original <= 0:
        return 0.0
    return (d1_original - d1_sliced) / d1_original * 100.0


def compile_pipeline_mlir(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    work_dir: Path,
    l1_cache_kb: int,
) -> Dict[str, str]:
    config = EdgeSlideMLIRConfig(l1_cache_kb=l1_cache_kb, output_dir=str(work_dir))
    pipeline = EdgeSlideMLIRPipeline(config)
    return pipeline.compile(
        model,
        input_shape,
        model_name="hw_profile",
        entry_func="forward",
    )


def _metadata_output(cmd: List[str], *, cwd: Optional[str] = None) -> Optional[str]:
    """Stdout of a best-effort metadata command, or None if it did not work.

    Bounded like every other subprocess here: ``git rev-parse`` runs against a
    checkout that may sit on a network-backed mount, and a hang there would
    strand the job after the expensive compile and cachegrind phases but
    before any results are written.
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=_timeout_from_env("EDGESLIDE_METADATA_TIMEOUT_S", 30),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def collect_host_metadata(device: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "device": device,
        "device_description": DEVICE_CACHEGRIND.get(device, {}).get("description", ""),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cachegrind_model": {
            "I1": DEVICE_CACHEGRIND.get(device, DEVICE_CACHEGRIND["generic"])["i1"],
            "D1": DEVICE_CACHEGRIND.get(device, DEVICE_CACHEGRIND["generic"])["d1"],
        },
    }
    sha = _metadata_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT))
    if sha:
        meta["git_sha"] = sha

    lscpu = _metadata_output(["lscpu"])
    if lscpu:
        meta["lscpu_excerpt"] = [
            line.strip()
            for line in lscpu.splitlines()
            if any(
                key in line
                for key in (
                    "Model name",
                    "Architecture",
                    "CPU(s)",
                    "L1d",
                    "L1i",
                    "L2",
                    "L3",
                    "Vendor",
                )
            )
        ]

    uname = _metadata_output(["uname", "-a"])
    if uname:
        meta["uname"] = uname

    return meta


def profile_model(
    model_name: str,
    model: nn.Module,
    input_shape: Tuple[int, ...],
    l1_cache_kb: int,
    work_root: Path,
    *,
    device: str = "generic",
    strict_ir: bool = True,
    skip_cachegrind: bool = False,
    wall_clock_trials: int = 5,
    enable_vectorization: bool = False,
    vector_width: int = 8,
) -> ModelHardwareReport:
    model_dir = work_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    artifacts = compile_pipeline_mlir(model, input_shape, model_dir, l1_cache_kb)
    payload_path = Path(artifacts["payload_mlir"])
    sliced_path = Path(artifacts["sliced_payload_mlir"])
    transform_path = Path(artifacts.get("transform_mlir", model_dir / "transform.mlir"))

    sliced_text = sliced_path.read_text()
    transform_text = transform_path.read_text() if transform_path.is_file() else None
    ir_sanity = analyze_sliced_ir(sliced_text, transform_text)
    assert_ir_sane_for_measure(ir_sanity, strict=strict_ir, model_name=model_name)

    orig_bin, _, _, entry_func, orig_ll = compile_mlir_artifacts(
        payload_path,
        model_dir / "original",
        "original",
        enable_vectorization=enable_vectorization,
        vector_width=vector_width,
    )
    sliced_bin, _, _, _, sliced_ll = compile_mlir_artifacts(
        sliced_path,
        model_dir / "sliced",
        "sliced",
        enable_vectorization=enable_vectorization,
        vector_width=vector_width,
    )

    reduction: Optional[float]
    if skip_cachegrind:
        orig_stats = CachegrindStats(d_refs=0, d1_misses=0, stderr_excerpt="skipped")
        sliced_stats = CachegrindStats(d_refs=0, d1_misses=0, stderr_excerpt="skipped")
        reduction = None
    else:
        orig_stats = run_cachegrind(orig_bin, model_dir / "original", device=device)
        sliced_stats = run_cachegrind(sliced_bin, model_dir / "sliced", device=device)
        reduction = traffic_reduction_pct(orig_stats.d1_misses, sliced_stats.d1_misses)

    # The traffic and timing comparison below is about tiling only if the two
    # binaries differ in nothing else. `enable_vectorization` is passed to
    # both, but `outline_stride1_convs` matches per-op, and a *tiled* payload
    # can present fewer matchable convolutions than an untiled one — measured
    # on ResNet-18 at 64x64, the untiled arm vectorized 133 convolutions and
    # the tiled arms 93. A "reduction" across that gap measures tiling plus 40
    # fewer vectorized convs. Checked rather than assumed.
    orig_simd = count_simd_types(Path(orig_ll))
    sliced_simd = count_simd_types(Path(sliced_ll))
    if orig_simd != sliced_simd:
        raise UnequalVectorization(
            f"{model_name}: the original binary emitted {orig_simd} SIMD types "
            f"and the sliced binary {sliced_simd}. The two differ in how much "
            "was vectorized, so a traffic or timing difference between them is "
            "not attributable to tiling. Re-run with --no-vectorization for a "
            "comparison that isolates the tiling decision."
        )

    orig_wc = run_wall_clock(orig_bin, trials=wall_clock_trials)
    sliced_wc = run_wall_clock(sliced_bin, trials=wall_clock_trials)

    # The checksums are the only hardware-level numerical-correctness signal;
    # divergence means the sliced binary computes a different result and its
    # speedup/traffic numbers must not be cited as validated evidence.
    rel_diff = checksum_rel_diff(orig_wc.checksum, sliced_wc.checksum)
    if rel_diff is not None and not rel_diff <= CHECKSUM_REL_TOL:
        # This used to print a warning and return the report anyway, with the
        # comment above saying the numbers "must not be cited" — while putting
        # them in the JSON next to every valid cell, indistinguishable from
        # them. A binary that computes the wrong answer has no meaningful cache
        # or timing profile, so this is a failure: the caller records it and
        # continues to the next model.
        raise NumericallyInvalidBinary(
            f"{model_name}: original and sliced checksums diverge "
            f"(relative difference {rel_diff:.3e} > tolerance "
            f"{CHECKSUM_REL_TOL:.3e}: {orig_wc.checksum} vs "
            f"{sliced_wc.checksum}). The sliced binary computes a different "
            "result, so its traffic and timing numbers describe the wrong "
            "computation and are not reported."
        )

    return ModelHardwareReport(
        model_name=model_name,
        input_shape=list(input_shape),
        l1_cache_kb=l1_cache_kb,
        entry_func=entry_func,
        original=BinaryProfile("original", orig_bin, orig_ll, orig_stats, orig_wc),
        sliced=BinaryProfile("sliced", sliced_bin, sliced_ll, sliced_stats, sliced_wc),
        traffic_reduction_pct=reduction,
        cachegrind_skipped=skip_cachegrind,
        simd_types_emitted=orig_simd,
        ir_sanity=ir_sanity,
        payload_mlir=str(payload_path),
        sliced_payload_mlir=str(sliced_path),
        transform_mlir=str(transform_path) if transform_path.is_file() else None,
    )


def _profile_to_dict(report: ModelHardwareReport) -> Dict[str, Any]:
    def binary_dict(b: BinaryProfile) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "label": b.label,
            "binary_path": b.binary_path,
            "llvm_ir_path": b.llvm_ir_path,
            "d_refs": b.stats.d_refs,
            "d1_misses": b.stats.d1_misses,
            "ll_misses": b.stats.ll_misses,
            "cachegrind_out_file": b.stats.cachegrind_out_file,
            "stderr_excerpt": b.stats.stderr_excerpt,
        }
        if b.wall_clock is not None:
            out["wall_clock"] = asdict(b.wall_clock)
        return out

    payload: Dict[str, Any] = {
        "model_name": report.model_name,
        "input_shape": report.input_shape,
        "l1_cache_kb": report.l1_cache_kb,
        "entry_func": report.entry_func,
        "original": binary_dict(report.original),
        "sliced": binary_dict(report.sliced),
        "traffic_reduction_pct": report.traffic_reduction_pct,
        # Distinguishes "measured, and the reduction was zero" from "not
        # measured", which a null reduction alone cannot.
        "cachegrind_skipped": report.cachegrind_skipped,
        "simd_types_emitted": report.simd_types_emitted,
        "d_refs_original": report.original.stats.d_refs,
        "d_refs_sliced": report.sliced.stats.d_refs,
        "d1_misses_original": report.original.stats.d1_misses,
        "d1_misses_sliced": report.sliced.stats.d1_misses,
        "payload_mlir": report.payload_mlir,
        "sliced_payload_mlir": report.sliced_payload_mlir,
        "transform_mlir": report.transform_mlir,
    }
    if report.ir_sanity is not None:
        payload["ir_sanity"] = asdict(report.ir_sanity)
    if report.original.wall_clock and report.sliced.wall_clock:
        o = report.original.wall_clock.mean_s
        s = report.sliced.wall_clock.mean_s
        payload["wall_clock_speedup_pct"] = (
            (o - s) / o * 100.0 if o > 0 else 0.0
        )
        rel_diff = checksum_rel_diff(
            report.original.wall_clock.checksum,
            report.sliced.wall_clock.checksum,
        )
        if rel_diff is not None:
            payload["checksum_rel_diff"] = rel_diff
            # A NaN diff compares False against the tolerance → mismatch.
            payload["checksums_match"] = bool(rel_diff <= CHECKSUM_REL_TOL)
    return payload


def check_prerequisites(*, need_valgrind: bool = True) -> List[str]:
    missing: List[str] = []
    for tool in ("cc", "mlir-opt"):
        if tool == "mlir-opt":
            if find_mlir_tool("mlir-opt") is None:
                missing.append("mlir-opt")
        elif shutil.which(tool) is None:
            missing.append(tool)
    if need_valgrind and find_valgrind() is None:
        missing.append("valgrind")
    if find_mlir_tool("mlir-translate") is None:
        missing.append("mlir-translate")
    try:
        import torch_mlir  # noqa: F401
    except ImportError:
        missing.append("torch_mlir")
    try:
        import llvmlite  # noqa: F401
    except ImportError:
        missing.append("llvmlite")
    return missing


def _compile_only_smoke(
    models: Optional[List[str]] = None,
    l1_cache_kb: int = 32,
    work_dir: Optional[Path] = None,
    strict_ir: bool = True,
    enable_vectorization: bool = False,
    vector_width: int = 8,
) -> None:
    """Build payload/sliced binaries and execute once (no valgrind)."""
    selected = models or ["simple_cnn"]
    if work_dir is not None:
        work_root = work_dir
        work_root.mkdir(parents=True, exist_ok=True)
        _smoke_loop(selected, l1_cache_kb, work_root, strict_ir, enable_vectorization, vector_width)
        return

    with tempfile.TemporaryDirectory(prefix="edgeslide_hw_smoke_") as tmp:
        _smoke_loop(selected, l1_cache_kb, Path(tmp), strict_ir, enable_vectorization, vector_width)


def _smoke_loop(
    selected: List[str],
    l1_cache_kb: int,
    work_root: Path,
    strict_ir: bool,
    enable_vectorization: bool = False,
    vector_width: int = 8,
) -> None:
    for name in selected:
        if name not in SMALL_MODELS:
            raise ValueError(f"Unknown model '{name}'. Available: {list(SMALL_MODELS)}")
        model_cls, input_shape = SMALL_MODELS[name]
        model_dir = work_root / name
        model_dir.mkdir(parents=True, exist_ok=True)
        artifacts = compile_pipeline_mlir(
            model_cls().eval().cpu(), input_shape, model_dir, l1_cache_kb
        )
        sliced_path = Path(artifacts["sliced_payload_mlir"])
        transform_path = Path(
            artifacts.get("transform_mlir", model_dir / "transform.mlir")
        )
        transform_text = (
            transform_path.read_text() if transform_path.is_file() else None
        )
        ir_sanity = analyze_sliced_ir(sliced_path.read_text(), transform_text)
        assert_ir_sane_for_measure(ir_sanity, strict=strict_ir, model_name=name)

        for label, mlir_key in (
            ("original", "payload_mlir"),
            ("sliced", "sliced_payload_mlir"),
        ):
            bin_path, *_ = compile_mlir_artifacts(
                Path(artifacts[mlir_key]),
                model_dir / label,
                label,
                enable_vectorization=enable_vectorization,
                vector_width=vector_width,
            )
            smoke_timeout_s = _timeout_from_env("EDGESLIDE_WALLCLOCK_TIMEOUT_S", 1800)
            try:
                result = subprocess.run(
                    [bin_path], capture_output=True, text=True, timeout=smoke_timeout_s
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Smoke run timed out after {smoke_timeout_s}s for {name}/{label}; "
                    "set EDGESLIDE_WALLCLOCK_TIMEOUT_S to raise the limit"
                ) from exc
            if result.returncode != 0:
                raise RuntimeError(
                    f"Smoke run failed for {name}/{label}: {result.stderr}"
                )
            print(f"[smoke] {name}/{label}: {result.stdout.strip()}")
            print(f"[smoke] binary={bin_path}")


def run_hardware_validation(
    models: Optional[List[str]] = None,
    l1_cache_kb: int = 32,
    output_path: Path = RESULTS_PATH,
    *,
    device: str = "generic",
    work_dir: Optional[Path] = None,
    strict_ir: bool = True,
    skip_cachegrind: bool = False,
    wall_clock_trials: int = 5,
    enable_vectorization: bool = False,
    vector_width: int = 8,
) -> Dict[str, Any]:
    selected = models or ["simple_cnn"]
    missing = check_prerequisites(need_valgrind=not skip_cachegrind)
    if missing:
        raise RuntimeError(f"Missing prerequisites: {', '.join(missing)}")

    reports: List[ModelHardwareReport] = []

    invalid: List[Dict[str, str]] = []

    def _run_all(work_root: Path) -> None:
        for name in selected:
            if name not in SMALL_MODELS:
                raise ValueError(
                    f"Unknown model '{name}'. Available: {list(SMALL_MODELS)}"
                )
            model_cls, input_shape = SMALL_MODELS[name]
            print(f"\n=== Profiling {name} input_shape={input_shape} device={device} ===")
            model = model_cls().eval().cpu()
            try:
                report = profile_model(
                    name,
                    model,
                    input_shape,
                    l1_cache_kb,
                    work_root,
                    device=device,
                    strict_ir=strict_ir,
                    skip_cachegrind=skip_cachegrind,
                    wall_clock_trials=wall_clock_trials,
                    enable_vectorization=enable_vectorization,
                    vector_width=vector_width,
                )
            except UnequalVectorization as exc:
                # Recorded and skipped, like a numerically invalid binary: the
                # cell produced no comparable measurement.
                print(f"  UNEQUAL VECTORIZATION: {exc}")
                invalid.append({"model": name, "error": str(exc)})
                continue
            except NumericallyInvalidBinary as exc:
                # Recorded and skipped rather than aborting the sweep, and
                # kept out of `reports` so it contributes to no aggregate. A
                # wrong binary's numbers are not a slower or faster version of
                # the right answer.
                print(f"  NUMERICALLY INVALID: {exc}")
                invalid.append({"model": name, "error": str(exc)})
                continue
            reports.append(report)
            if report.traffic_reduction_pct is None:
                print("  D1 misses: not measured (--skip-cachegrind)")
            else:
                print(
                    f"  D1 misses: original={report.original.stats.d1_misses:,} "
                    f"sliced={report.sliced.stats.d1_misses:,} "
                    f"traffic_reduction={report.traffic_reduction_pct:.2f}%"
                )
            if report.original.wall_clock and report.sliced.wall_clock:
                print(
                    f"  Wall clock mean: original={report.original.wall_clock.mean_s*1000:.2f}ms "
                    f"sliced={report.sliced.wall_clock.mean_s*1000:.2f}ms"
                )

    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        _run_all(work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="edgeslide_hw_") as tmp:
            _run_all(Path(tmp))

    wall_clock_speedups_pct = [
        (r.original.wall_clock.mean_s - r.sliced.wall_clock.mean_s)
        / r.original.wall_clock.mean_s
        * 100.0
        for r in reports
        if r.original.wall_clock
        and r.sliced.wall_clock
        and r.original.wall_clock.mean_s > 0
    ]

    measured_reductions = [
        r.traffic_reduction_pct
        for r in reports
        if r.traffic_reduction_pct is not None
    ]

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "l1_cache_kb": l1_cache_kb,
        "host": collect_host_metadata(device),
        "models": [_profile_to_dict(r) for r in reports],
        "workload_provenance": small_model_provenance(),
        # Models whose sliced binary computed a different result. Kept out of
        # `models` and out of every aggregate below, and listed here so their
        # absence is visible rather than inferred from a shorter list.
        "numerically_invalid": invalid,
        # Stamped so a result file is self-describing about whether the
        # stride-1 outline-and-vectorize path (vectorize_conv.py) was in the
        # pipeline for this run — without this, "sliced was Nx slower" and
        # "sliced was Nx faster" results are indistinguishable after the
        # fact. See lowering.py SlicedPayloadLoweringConfig.
        "enable_stride1_vectorization": enable_vectorization,
        # What was *requested*. `simd_types_emitted` per model records what
        # was achieved — the lowering falls back to scalar on failure, so the
        # two are different facts.
        "vector_width": vector_width if enable_vectorization else None,
        "summary": {
            # None, not 0.0, when nothing was measured: with
            # --skip-cachegrind every `traffic_reduction_pct` is None, and a
            # mean of zero measurements reported as "0%" reads as a real
            # result showing no effect.
            "mean_traffic_reduction_pct": (
                sum(measured_reductions) / len(measured_reductions)
                if measured_reductions
                else None
            ),
            "traffic_measured_models": len(measured_reductions),
            "mean_wall_clock_speedup_pct": (
                sum(wall_clock_speedups_pct) / len(wall_clock_speedups_pct)
                if wall_clock_speedups_pct
                else None
            ),
            "model_count": len(reports),
            "device": device,
            "enable_stride1_vectorization": enable_vectorization,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n✅ Hardware validation written to {output_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EdgeSlide Cachegrind / wall-clock hardware profiler"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["simple_cnn"],
        choices=list(SMALL_MODELS.keys()),
        help="Models to profile (default: simple_cnn @ 256)",
    )
    parser.add_argument(
        "--device",
        choices=list(DEVICE_CACHEGRIND.keys()),
        default="generic",
        help="Cachegrind + L1 preset (pi4 → D1=32KB, default L1 budget 32KB)",
    )
    parser.add_argument(
        "--l1-cache-kb",
        type=int,
        default=None,
        help="L1 cache budget for EdgeSlide pipeline "
        "(default: device preset, pi4=32, generic=256)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default depends on --device)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Persist IR/binaries here (default: temp dir)",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Build and smoke-run binaries only (skip valgrind)",
    )
    parser.add_argument(
        "--skip-cachegrind",
        action="store_true",
        help="Skip valgrind; still record wall-clock (faster on Pi)",
    )
    parser.add_argument(
        "--wall-clock-trials",
        type=int,
        default=5,
        help="Timed native runs per binary after 1 warmup (default: 5)",
    )
    parser.add_argument(
        "--no-strict-ir",
        action="store_true",
        help="Do not fail when sliced IR lacks scf.forall / tiled conv",
    )
    parser.add_argument(
        "--enable-vectorization",
        action="store_true",
        help="Outline stride-1 linalg.conv_2d_nchw_fchw tiles and vectorize "
        "them with the legacy affine vectorizer before lowering to LLVM "
        "(SlicedPayloadLoweringConfig.enable_stride1_vectorization; see "
        "src/edgeslide/mlir_transform/vectorize_conv.py). Off by default. "
        "Applies to both the 'original' and 'sliced' binaries so the "
        "wall-clock comparison stays fair. Stride>1 convs are always left "
        "scalar regardless of this flag — verified unsafe for the legacy "
        "vectorizer, not just unsupported.",
    )
    parser.add_argument(
        "--vector-width",
        type=int,
        default=8,
        help="Virtual vector width for --affine-super-vectorize when "
        "--enable-vectorization is set (default: 8)",
    )
    args = parser.parse_args()

    device = args.device
    l1_cache_kb = args.l1_cache_kb
    if l1_cache_kb is None:
        l1_cache_kb = int(DEVICE_CACHEGRIND[device]["default_l1_cache_kb"])

    output = args.output
    if output is None:
        output = PI4_RESULTS_PATH if device == "pi4" else RESULTS_PATH

    work_dir = args.work_dir
    if work_dir is None and device == "pi4":
        work_dir = ROOT / "build" / "pi4_profile"

    strict_ir = not args.no_strict_ir

    try:
        if args.compile_only:
            _compile_only_smoke(
                models=args.models,
                l1_cache_kb=l1_cache_kb,
                work_dir=work_dir,
                strict_ir=strict_ir,
                enable_vectorization=args.enable_vectorization,
                vector_width=args.vector_width,
            )
        else:
            run_hardware_validation(
                models=args.models,
                l1_cache_kb=l1_cache_kb,
                output_path=output,
                device=device,
                work_dir=work_dir,
                strict_ir=strict_ir,
                skip_cachegrind=args.skip_cachegrind,
                wall_clock_trials=args.wall_clock_trials,
                enable_vectorization=args.enable_vectorization,
                vector_width=args.vector_width,
            )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        # Expected operator-facing failures: the tool wrappers already embed
        # the failing command and its stdout/stderr in the message, so the
        # traceback adds nothing. Anything else is a defect in this file and
        # propagates with its traceback — printing bare `ERROR: 'payload_mlir'`
        # for a KeyError left no file, no line, and no frame to work from.
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
