#!/usr/bin/env python3
"""Real synchronized six-camera EdgeSlice vs SConv comparison runner.

This runner is strict by design:

* input batches must come from real synchronized six-camera nuImages metadata;
* missing cameras fail the run instead of repeating or fabricating images;
* each model/method benchmark runs in a fresh subprocess;
* reported latency and RSS values are measured from actual forwards.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.metrics import (  # noqa: E402
    compare_paired_timings,
    numerical_equivalence,
    parse_perf_stat_csv,
)
from benchmarks.nuimages_sixcam import (  # noqa: E402
    DEFAULT_NUIMAGES_ROOT,
    build_manifest,
    flatten_sixcam_tensor,
    load_sixcam_tensor_from_manifest,
    select_sixcam_batches,
    summarize_sixcam_availability,
    write_manifest,
)

RESULTS_PATH = ROOT / "benchmarks" / "results" / "sixcam_sconv_comparison.json"
WORK_DIR = ROOT / "benchmarks" / "results" / "sixcam_sconv_work"
SUPPORTED_MODELS = ("resnet50", "mobilenet_v2")
SUPPORTED_METHODS = (
    "baseline",
    "edgeslice",
    "sconv",
    "edgeslice_mlir",
    "sconv_upstream",
)

# Methods run when --methods is not given: only arms that run real
# implementations.
#
#   baseline         plain PyTorch — the reference every other arm is measured
#                    against, and genuinely what it says it is
#   edgeslice_mlir   the MLIR-compiled payload through the JIT — the
#                    contribution
#   sconv_upstream   the published SConvTransform, built against MLIR 19
#
# `edgeslice` and `sconv` are deliberately **excluded**. Both are PyTorch
# reference implementations that declare `IS_REFERENCE_IMPLEMENTATION = True`:
# `edgeslice` is the demo bypass its own module header calls "a quick numerical
# simulation ... must not be cited as evidence that EdgeSlice works
# end-to-end", and `sconv` is a local spatial-halo stand-in, not Ferrari et
# al.'s transform. With both in the defaults, a plain `--methods`-less run
# compared one reference implementation against another and reported it as a
# result. They remain available by name for reference-vs-reference checks.
#
# `sconv_upstream` skips itself (into the report's `skipped` list, with a
# reason) when `sconv-opt` has not been built — see
# third_party/SConvTransform/build_llvm19.sh. That is visible in the output,
# unlike a stand-in silently standing in for it.
DEFAULT_METHODS = ("baseline", "edgeslice_mlir", "sconv_upstream")

# Where build_llvm19.sh puts the upstream binary, and where an earlier
# hand-build may have left one.
_SCONV_OPT_CANDIDATES = (
    ROOT / "third_party" / "SConvTransform" / "build-llvm19" / "bin" / "sconv-opt",
    ROOT / "third_party" / "SConvTransform" / "build" / "bin" / "sconv-opt",
)


class UnsupportedConfiguration(RuntimeError):
    """This model/method pair is out of scope, as opposed to having failed.

    Kept distinct from a run failure so one unsupported cell (e.g. the JIT arm
    needs the model's output rank to match its input rank, which rules out
    classifier heads) neither aborts the sweep nor is silently omitted: it is
    recorded with its reason in the report's ``skipped`` list.
    """
PERF_EVENTS = (
    "cache-references",
    "cache-misses",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "LLC-loads",
    "LLC-load-misses",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _parse_resolution(text: str) -> Tuple[int, int]:
    if "x" not in text.lower():
        raise argparse.ArgumentTypeError(f"Expected HxW, got {text!r}")
    left, right = text.lower().split("x", 1)
    return int(left), int(right)


def _current_rss_kb() -> int:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1])
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class PeakRSSSampler:
    """Poll current process RSS while a forward loop is running."""

    def __init__(self, interval_s: float = 0.002):
        self.interval_s = interval_s
        self.peak_kb = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.peak_kb = _current_rss_kb()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_kb = max(self.peak_kb, _current_rss_kb())
        return self.peak_kb

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_kb = max(self.peak_kb, _current_rss_kb())
            time.sleep(self.interval_s)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _list_single_image_paths(
    root: Path,
    *,
    image_dir: Optional[Path] = None,
    include_sweeps: bool = False,
) -> List[Path]:
    """List real image files for one-image-at-a-time benchmarking."""

    if image_dir is not None:
        search_roots = [image_dir]
    else:
        search_roots = [root / "samples"]
        if include_sweeps:
            search_roots.append(root / "sweeps")

    paths: List[Path] = []
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        paths.extend(
            sorted(
                path
                for path in search_root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        )
    return paths


def _build_single_image_manifest(
    paths: Sequence[Path],
    *,
    nuimages_root: Path,
    height: int,
    width: int,
    normalize: str,
) -> Dict[str, Any]:
    return {
        "source": "single_real_images_sequential",
        "nuimages_root": str(nuimages_root),
        "image_count": len(paths),
        "height": height,
        "width": width,
        "normalize": normalize,
        "paths": [str(path) for path in paths],
    }


def _load_single_image_tensors(manifest_path: Path):
    import torch

    from benchmarks.image_inputs import load_image_tensor

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source") != "single_real_images_sequential":
        raise ValueError(f"Not a single-image manifest: {manifest_path}")
    tensors = [
        load_image_tensor(
            path,
            int(manifest["height"]),
            int(manifest["width"]),
            normalize=manifest.get("normalize", "imagenet"),
        ).unsqueeze(0)
        for path in manifest["paths"]
    ]
    if not tensors:
        raise FileNotFoundError("single-image manifest has no paths")
    return [tensor.to(dtype=torch.float32, device="cpu") for tensor in tensors], manifest


def _set_torch_threads(torch_module, num_threads: int) -> None:
    torch_module.set_num_threads(num_threads)
    try:
        torch_module.set_num_interop_threads(max(1, min(2, num_threads)))
    except RuntimeError:
        pass


def _load_model(model_name: str):
    import torch

    from benchmarks.workloads import load_model

    torch.manual_seed(0)
    model = load_model(model_name).eval().cpu()
    return model


def _apply_method(model, method: str, l1_cache_kb: int) -> Tuple[Any, Dict[str, Any]]:
    if method == "baseline":
        return model, {
            "implementation": "plain_pytorch",
            "num_transformed_layers": 0,
            "transformed_layers": [],
        }

    if method == "edgeslice":
        from edgeslide.edgeslide_pytorch import EdgeSlideTransform, SlicingConfig

        transform = EdgeSlideTransform(
            SlicingConfig(l1_cache_bytes=l1_cache_kb * 1024, enabled=True, verbose=False)
        )
        transformed = transform.transform_model(model)
        return transformed, {
            "implementation": "edgeslice_pytorch_demo_batch_tiling",
            "num_transformed_layers": len(transform.transformed_layers),
            "transformed_layers": list(transform.transformed_layers.keys()),
        }

    if method == "sconv":
        from edgeslide.sconv_pytorch import SConvConfig, SConvTransform

        transform = SConvTransform(
            SConvConfig(l1_cache_bytes=l1_cache_kb * 1024, enabled=True, verbose=False)
        )
        transformed = transform.transform_model(model)
        return transformed, {
            "implementation": "local_exact_spatial_halo_sconv",
            "num_transformed_layers": len(transform.transformed_layers),
            "transformed_layers": list(transform.transformed_layers.keys()),
        }

    raise ValueError(f"Unknown method {method!r}")


def find_sconv_opt() -> Optional[Path]:
    """Locate the upstream `sconv-opt`, or None when it has not been built."""
    override = os.environ.get("SCONV_OPT")
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    for candidate in _SCONV_OPT_CANDIDATES:
        if candidate.is_file():
            return candidate
    found = shutil.which("sconv-opt")
    return Path(found) if found else None


def sconv_upstream_provenance(sconv_opt: Path) -> Dict[str, Any]:
    """Toolchain identity for the upstream arm.

    Recorded so a result can never be confused with the local stand-in's, which
    is the whole reason this arm exists. `implementation` is the field a reader
    uses to tell them apart: `local_exact_spatial_halo_sconv` versus
    `sconv_upstream_sconvtransform`.
    """
    def _git(*args: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", "-C", str(ROOT), *args],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.stdout.strip() or None if proc.returncode == 0 else None

    version = None
    try:
        proc = subprocess.run(
            [str(sconv_opt), "--version"], capture_output=True, text=True, timeout=60
        )
        version = (proc.stdout or proc.stderr).strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired, IndexError):
        pass

    return {
        "implementation": "sconv_upstream_sconvtransform",
        "sconv_opt_path": str(sconv_opt),
        "upstream_mlir_version": version,
        # The vendored tree is checked in, so its provenance is this repo's
        # commit for that subtree rather than an upstream remote's.
        "vendored_commit": _git("rev-parse", "HEAD"),
        "vendored_subtree_commit": _git(
            "log", "-1", "--format=%H", "--", "third_party/SConvTransform"
        ),
        # The upstream repo declares sgemm_blas_kernel and does not ship it, so
        # any throughput number here uses this repo's substitute microkernel.
        # Stated in the result rather than left to a doc nobody reads next to
        # the number.
        "upstream_kernel_exact": False,
        "microkernel_note": (
            "SConvTransform declares the external symbol sgemm_blas_kernel but "
            "does not ship an implementation. Throughput measured here uses "
            "this repo's substitute AArch64 NEON microkernel, not the authors' "
            "OpenBLAS kernel, and must not be attributed to them."
        ),
    }


def _run_sconv_upstream_worker(
    args: argparse.Namespace,
    model,
    input_tensors: Sequence[Any],
    input_meta: Mapping[str, Any],
    input_shape: Sequence[int],
    input_mode: str,
) -> Dict[str, Any]:
    """Run the published SConvTransform on the same payload EdgeSlice tiles.

    MLIR-level, not PyTorch-level: `transform.structured.sconv` is a
    transform-dialect op, so the honest place to apply it is the same Linalg
    payload the EdgeSlice arm receives. That makes the two arms differ in the
    transform applied and in nothing else.

    Everything the upstream op cannot handle raises ``UnsupportedConfiguration``
    rather than a bare error, so one out-of-scope cell lands in the report's
    `skipped` list with a reason instead of aborting the sweep.
    """
    sconv_opt = find_sconv_opt()
    if sconv_opt is None:
        raise UnsupportedConfiguration(
            "upstream sconv-opt not found; build it with "
            "third_party/SConvTransform/build_llvm19.sh, or set $SCONV_OPT"
        )
    if not input_tensors:
        raise UnsupportedConfiguration("sconv_upstream requires at least one input")

    import torch

    from edgeslide.mlir_transform.errors import ExternalToolError, run_tool
    from edgeslide.mlir_transform.pipeline import (
        EdgeSlideMLIRConfig,
        EdgeSlideMLIRPipeline,
    )

    first_input = input_tensors[0]
    output_dir = WORK_DIR / f"sconv_upstream_{os.getpid()}_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 only: the Linalg payload, before any EdgeSlice tiling.
    pipeline = EdgeSlideMLIRPipeline(
        EdgeSlideMLIRConfig(l1_cache_kb=args.l1_cache_kb, output_dir=str(output_dir))
    )
    with torch.no_grad():
        parser, _ = pipeline.parse_model(
            model, tuple(first_input.shape)
        )
    payload = Path(parser.save_mlir(str(output_dir / f"{args.model}.mlir")))

    # How many ops the transform is supposed to consume. Needed to tell
    # "transformed them all" from "there were none to transform": the latter
    # also leaves zero behind, and reporting it as a successful SConv arm would
    # be a silent no-op dressed as a result.
    convs_before = payload.read_text().count("linalg.conv_2d_nchw_fchw")
    if convs_before == 0:
        raise UnsupportedConfiguration(
            f"{args.model} payload contains no linalg.conv_2d_nchw_fchw ops; "
            "structured.sconv has nothing to apply to"
        )

    transform_script = output_dir / "sconv_transform.mlir"
    transform_script.write_text(_SCONV_UPSTREAM_TRANSFORM)

    compile_start = time.perf_counter()
    sliced = output_dir / "sconv_upstream_payload.mlir"
    try:
        # sconv-opt is a transform-dialect driver, not an mlir-opt clone: the
        # script is passed with -transform=<file> and there is no
        # --transform-interpreter flag.
        result = run_tool(
            [
                str(sconv_opt),
                f"-transform={transform_script}",
                str(payload),
            ],
            stage="sconv-opt structured.sconv",
        )
    except ExternalToolError as exc:
        raise UnsupportedConfiguration(
            f"upstream structured.sconv did not apply to {args.model}: "
            f"{str(exc)[:400]}"
        ) from exc
    sliced.write_text(result.stdout)
    compile_ms = (time.perf_counter() - compile_start) * 1000

    remaining = result.stdout.count("linalg.conv_2d_nchw_fchw")
    if remaining:
        # A partial application is not a result either: the arm would be a
        # mixture of SConv and stock lowering with no way to tell the split.
        raise UnsupportedConfiguration(
            f"upstream structured.sconv left {remaining} of {convs_before} "
            "linalg.conv_2d_nchw_fchw op(s) untransformed; the arm did not "
            "fully apply"
        )

    meta = sconv_upstream_provenance(sconv_opt)
    meta.update(
        {
            "num_transformed_layers": convs_before,
            "transformed_layers": [],
            "payload_mlir": str(payload),
            "transformed_mlir": str(sliced),
            "compile_ms": round(compile_ms, 3),
            "loop_nest_depth": result.stdout.count("scf.for"),
            "gemm_generics": result.stdout.count("linalg.generic"),
        }
    )
    return {
        "model": args.model,
        "method": "sconv_upstream",
        "input_shape": list(input_shape),
        "input_mode": input_mode,
        "input_meta": dict(input_meta),
        "method_meta": meta,
        # Runtime numbers come from benchmarks/run_sconv_upstream.py, which
        # owns the microkernel shim and the isolated-trial protocol. Reporting
        # a timing here too would invite comparing two different harnesses.
        "timing": None,
        "runtime_measured": False,
        "runtime_note": (
            "compile-and-transform only; for throughput use "
            "benchmarks/run_sconv_upstream.py, which supplies the microkernel"
        ),
    }


# Matches the checked-in third_party/SConvTransform/test/sconv.mlir.
_SCONV_UPSTREAM_TRANSFORM = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op) {
    %conv = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg0
      : (!transform.any_op) -> !transform.op<"linalg.conv_2d_nchw_fchw">

    %res, %loops:6 = transform.structured.sconv %conv
      : (!transform.op<"linalg.conv_2d_nchw_fchw">)
      -> (!transform.op<"linalg.generic">, !transform.any_op, !transform.any_op,
          !transform.any_op, !transform.any_op, !transform.any_op,
          !transform.any_op)

    transform.yield
  }
}
"""



def _toolchain_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    """Tool versions and the upstream-SConv build state for the report header.

    Kept separate from ``method_meta`` because it describes the *run*, not one
    arm: a reader comparing two report files needs to know whether they were
    produced by the same compiler before comparing any number in them.
    """
    metadata: Dict[str, Any] = {}
    try:
        from benchmarks.physical import environment_provenance

        metadata.update(environment_provenance())
    except Exception as exc:  # pragma: no cover - provenance must never abort a run
        metadata["environment_error"] = str(exc)[:200]

    sconv_opt = find_sconv_opt()
    if sconv_opt is None:
        metadata["sconv_upstream"] = {
            "built": False,
            "reason": "sconv-opt not found; build with "
            "third_party/SConvTransform/build_llvm19.sh",
        }
    else:
        metadata["sconv_upstream"] = {"built": True, **sconv_upstream_provenance(sconv_opt)}
    metadata["methods_requested"] = list(getattr(args, "methods", []) or [])
    return metadata


def _tensor_checksum(value) -> float:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return float(value.detach().float().sum().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.astype("float32").sum())
    if isinstance(value, Mapping):
        return sum(_tensor_checksum(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_checksum(v) for v in value)
    return 0.0


def _shape_payload(value):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, np.ndarray):
        return list(value.shape)
    if isinstance(value, Mapping):
        return {key: _shape_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape_payload(item) for item in value]
    return None


def _collect_runtime_slicing_meta(model, method: str) -> Dict[str, Any]:
    layers: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        if method == "edgeslice" and hasattr(module, "needs_slicing"):
            layers.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "needs_slicing": bool(getattr(module, "needs_slicing", False)),
                    "slice_dim": getattr(module, "slice_dim", None),
                    "tile_size": getattr(module, "tile_size", None),
                }
            )
        elif method == "sconv" and hasattr(module, "needs_tiling"):
            layers.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "needs_tiling": bool(getattr(module, "needs_tiling", False)),
                    "tile_output_rows": getattr(module, "tile_output_rows", None),
                    "fallback_reason": getattr(module, "fallback_reason", None),
                }
            )

    active_count = sum(
        1
        for layer in layers
        if layer.get("needs_slicing") or layer.get("needs_tiling")
    )
    return {
        "runtime_sliced_layers": active_count,
        "runtime_layer_details": layers,
    }


def _run_mlir_worker(
    args: argparse.Namespace,
    model,
    input_tensors: Sequence[Any],
    input_meta: Mapping[str, Any],
    input_shape: Sequence[int],
    input_mode: str,
) -> Dict[str, Any]:
    """Compile EdgeSlice MLIR once and time the compiled/JIT payload."""
    import contextlib
    import io

    import torch

    if not input_tensors:
        raise RuntimeError("edgeslice_mlir requires at least one input tensor")

    first_input = input_tensors[0]
    with torch.no_grad():
        reference_first = model(first_input)
    if not isinstance(reference_first, torch.Tensor):
        raise UnsupportedConfiguration(
            "edgeslice_mlir currently supports single-tensor model outputs"
        )
    if reference_first.ndim != first_input.ndim:
        raise UnsupportedConfiguration(
            "edgeslice_mlir currently requires output rank to match input rank "
            f"for the JIT result descriptor (input rank={first_input.ndim}, "
            f"output rank={reference_first.ndim})"
        )

    from edgeslide.mlir_transform.lowering import (
        SlicedPayloadJitEngine,
        SlicedPayloadLoweringConfig,
        detect_entry_func,
    )
    from edgeslide.mlir_transform.pipeline import EdgeSlideMLIRConfig, EdgeSlideMLIRPipeline

    output_dir = WORK_DIR / f"edgeslice_mlir_{os.getpid()}_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    compile_start = time.perf_counter()
    compile_log = io.StringIO()
    with contextlib.redirect_stdout(compile_log):
        pipeline = EdgeSlideMLIRPipeline(
            EdgeSlideMLIRConfig(
                l1_cache_kb=args.l1_cache_kb,
                output_dir=str(output_dir),
            )
        )
        artifacts = pipeline.compile(
            model,
            tuple(first_input.shape),
            model_name=f"{args.model}_edgeslice_mlir",
            entry_func="forward",
        )
        sliced_payload = Path(artifacts["sliced_payload_mlir"])
        entry_func = detect_entry_func(sliced_payload.read_text(), preferred="forward")
        jit = SlicedPayloadJitEngine.from_sliced_payload(
            str(sliced_payload),
            config=SlicedPayloadLoweringConfig(
                output_dir=str(output_dir),
                entry_func=entry_func,
            ),
        )
    compile_ms = (time.perf_counter() - compile_start) * 1000.0

    with torch.no_grad():
        for _ in range(args.warmup):
            for input_tensor in input_tensors:
                _ = jit.invoke(entry_func, input_tensor)

    gc.collect()
    rss_before_forward = _current_rss_kb()
    sampler = PeakRSSSampler(interval_s=args.rss_sample_interval_ms / 1000.0)
    sampler.start()
    times_ms: List[float] = []
    output = None
    last_input_tensor = None
    for _ in range(args.runs):
        for input_tensor in input_tensors:
            start = time.perf_counter()
            output = jit.invoke(entry_func, input_tensor)
            times_ms.append((time.perf_counter() - start) * 1000.0)
            last_input_tensor = input_tensor
    peak_forward_rss = sampler.stop()

    if output is None or last_input_tensor is None:
        raise RuntimeError("No timed MLIR output was produced")

    with torch.no_grad():
        reference_output = model(last_input_tensor)
    equivalence = numerical_equivalence(output, reference_output.detach().cpu().numpy())

    param_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "model": args.model,
        "method": args.method,
        "method_meta": {
            "implementation": "edgeslice_mlir_jit_compiled_payload",
            "num_transformed_layers": None,
            "transformed_layers": [],
            "compile_ms": compile_ms,
            "compile_log_tail": compile_log.getvalue().splitlines()[-20:],
            "entry_func": entry_func,
            "artifacts": {
                key: str(value) for key, value in artifacts.items() if isinstance(value, str)
            },
        },
        "input_mode": input_mode,
        "input_shape": list(input_shape),
        "output_shape": _shape_payload(output),
        "parameter_count": int(param_count),
        "warmup_runs": args.warmup,
        "timed_runs": args.runs,
        "timed_forwards": len(times_ms),
        "images_per_run": len(input_tensors),
        "num_threads": args.num_threads,
        "manifest_path": str(args.manifest),
        "manifest_sha256": _sha256(Path(args.manifest)),
        "input_meta": input_meta,
        "wall_clock_ms": {
            "mean": statistics.mean(times_ms),
            "median": statistics.median(times_ms),
            "std": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
            "min": min(times_ms),
            "max": max(times_ms),
            "samples": times_ms,
        },
        "rss_kb": {
            "before_forward": rss_before_forward,
            "peak_during_forward_loop": peak_forward_rss,
            "ru_maxrss_process": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "sample_interval_ms": args.rss_sample_interval_ms,
        },
        "output_checksum": _tensor_checksum(output),
        "numerical_equivalence": equivalence,
        "pid": os.getpid(),
        "measurement_type": "compiled_mlir_jit_wall_clock_and_rss",
    }


def _run_worker(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    _set_torch_threads(torch, args.num_threads)
    torch.manual_seed(0)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    manifest_path = Path(args.manifest)
    manifest_source = json.loads(manifest_path.read_text()).get("source")
    if manifest_source == "single_real_images_sequential":
        input_tensors, input_meta = _load_single_image_tensors(manifest_path)
        input_shape = list(input_tensors[0].shape)
        input_mode = "single_image_sequential"
    else:
        sixcam, input_meta = load_sixcam_tensor_from_manifest(manifest_path)
        input_tensors = [flatten_sixcam_tensor(sixcam).contiguous()]
        input_shape = list(input_tensors[0].shape)
        input_mode = "sixcam_flat_batch"

    model = _load_model(args.model)
    if args.method == "sconv_upstream":
        return _run_sconv_upstream_worker(
            args,
            model,
            input_tensors,
            input_meta,
            input_shape,
            input_mode,
        )
    if args.method == "edgeslice_mlir":
        return _run_mlir_worker(
            args,
            model,
            input_tensors,
            input_meta,
            input_shape,
            input_mode,
        )

    reference_model = copy.deepcopy(model).eval().cpu() if args.method != "baseline" else None
    model, method_meta = _apply_method(model, args.method, args.l1_cache_kb)
    model.eval().cpu()

    param_count = sum(parameter.numel() for parameter in model.parameters())
    rss_before_forward = _current_rss_kb()

    with torch.no_grad():
        for _ in range(args.warmup):
            for input_tensor in input_tensors:
                _ = model(input_tensor)

    gc.collect()
    sampler = PeakRSSSampler(interval_s=args.rss_sample_interval_ms / 1000.0)
    sampler.start()
    times_ms: List[float] = []
    output = None
    last_input_tensor = None
    with torch.no_grad():
        for _ in range(args.runs):
            for input_tensor in input_tensors:
                start = time.perf_counter()
                output = model(input_tensor)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                times_ms.append(elapsed_ms)
                last_input_tensor = input_tensor
    peak_forward_rss = sampler.stop()

    if output is None or last_input_tensor is None:
        raise RuntimeError("No timed output was produced")

    method_meta.update(_collect_runtime_slicing_meta(model, args.method))
    equivalence = None
    if reference_model is not None:
        with torch.no_grad():
            reference_output = reference_model(last_input_tensor)
        equivalence = numerical_equivalence(
            output.detach().cpu().numpy(),
            reference_output.detach().cpu().numpy(),
        )

    return {
        "model": args.model,
        "method": args.method,
        "method_meta": method_meta,
        "input_mode": input_mode,
        "input_shape": input_shape,
        "output_shape": _shape_payload(output),
        "parameter_count": int(param_count),
        "warmup_runs": args.warmup,
        "timed_runs": args.runs,
        "timed_forwards": len(times_ms),
        "images_per_run": len(input_tensors),
        "num_threads": args.num_threads,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "input_meta": input_meta,
        "wall_clock_ms": {
            "mean": statistics.mean(times_ms),
            "median": statistics.median(times_ms),
            "std": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
            "min": min(times_ms),
            "max": max(times_ms),
            "samples": times_ms,
        },
        "rss_kb": {
            "before_forward": rss_before_forward,
            "peak_during_forward_loop": peak_forward_rss,
            "ru_maxrss_process": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "sample_interval_ms": args.rss_sample_interval_ms,
        },
        "output_checksum": _tensor_checksum(output),
        "numerical_equivalence": equivalence,
        "pid": os.getpid(),
        "measurement_type": "real_subprocess_wall_clock_and_rss",
    }


def _worker_cmd(
    args: argparse.Namespace,
    *,
    manifest: Path,
    model: str,
    method: str,
) -> List[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--manifest",
        str(manifest),
        "--model",
        model,
        "--method",
        method,
        "--l1-cache-kb",
        str(args.l1_cache_kb),
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
        "--num-threads",
        str(args.num_threads),
        "--rss-sample-interval-ms",
        str(args.rss_sample_interval_ms),
    ]


def _run_child(
    cmd: List[str],
    *,
    pin_cpu: Optional[int],
    perf_stat: bool = False,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    pythonpath = [str(ROOT), str(SRC)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    full_cmd = cmd
    if pin_cpu is not None:
        taskset = shutil.which("taskset")
        if taskset is None:
            raise RuntimeError("--pin-cpu requested but taskset is not installed")
        full_cmd = [taskset, "-c", str(pin_cpu), *cmd]

    if perf_stat:
        perf = shutil.which("perf")
        if perf is None:
            raise RuntimeError("--perf-stat requested but perf is not installed")
        full_cmd = [
            perf,
            "stat",
            "-x",
            ",",
            "-e",
            ",".join(PERF_EVENTS),
            *full_cmd,
        ]

    return subprocess.run(full_cmd, capture_output=True, text=True, env=env)


def _summarize_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_model: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for result in results:
        by_model.setdefault(str(result["model"]), {})[str(result["method"])] = result

    summary: Dict[str, Any] = {}
    for model, methods in by_model.items():
        model_summary: Dict[str, Any] = {}
        baseline = methods.get("baseline")
        for method, result in methods.items():
            samples = result["wall_clock_ms"].get("samples", [])
            entry = {
                "mean_ms": float(result["wall_clock_ms"]["mean"]),
                "median_ms": float(result["wall_clock_ms"]["median"]),
            }
            if baseline and method != "baseline":
                comparison = compare_paired_timings(
                    baseline["wall_clock_ms"].get("samples", []),
                    samples,
                )
                entry["paired_vs_baseline"] = comparison
                entry["median_speedup_vs_baseline_pct"] = comparison["median_speedup_pct"]
            model_summary[method] = entry
        if "edgeslice" in methods and "sconv" in methods:
            model_summary["edgeslice_vs_sconv"] = compare_paired_timings(
                methods["sconv"]["wall_clock_ms"].get("samples", []),
                methods["edgeslice"]["wall_clock_ms"].get("samples", []),
            )
        summary[model] = model_summary
    return summary


def _run_parent(args: argparse.Namespace) -> Dict[str, Any]:
    height, width = args.resolution
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.input_mode == "single-image":
        image_count = args.image_count or args.batches
        paths = _list_single_image_paths(
            args.nuimages_root,
            image_dir=args.image_dir,
            include_sweeps=args.include_sweeps,
        )
        if len(paths) < image_count:
            raise ValueError(
                f"Need {image_count} real images for --input-mode single-image, "
                f"found {len(paths)} under {args.image_dir or args.nuimages_root / 'samples'}"
            )
        manifest = _build_single_image_manifest(
            paths[:image_count],
            nuimages_root=args.nuimages_root,
            height=height,
            width=width,
            normalize=args.normalize,
        )
        manifest_path = write_manifest(manifest, work_dir / "single_image_manifest.json")
    else:
        selected = select_sixcam_batches(
            args.nuimages_root,
            version=args.version,
            batch_count=args.batches,
        )
        manifest = build_manifest(
            selected,
            nuimages_root=args.nuimages_root,
            version=args.version,
            height=height,
            width=width,
            normalize=args.normalize,
        )
        manifest_path = write_manifest(manifest, work_dir / "sixcam_manifest.json")

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for model in args.models:
        for method in args.methods:
            label = f"{model}/{method}"
            mode_label = (
                "one real image per forward"
                if args.input_mode == "single-image"
                else "real six-camera batches"
            )
            print(f"\n=== Running {label} on {mode_label} ===")
            cmd = _worker_cmd(args, manifest=manifest_path, model=model, method=method)
            child = _run_child(cmd, pin_cpu=args.pin_cpu, perf_stat=args.perf_stat)
            if child.returncode != 0:
                error = {
                    "configuration": label,
                    "returncode": child.returncode,
                    "stdout": child.stdout,
                    "stderr": child.stderr,
                }
                errors.append(error)
                print(f"  FAILED: {child.stderr.strip() or child.stdout.strip()}")
                if not args.continue_on_error:
                    break
                continue
            try:
                result = json.loads(child.stdout)
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "configuration": label,
                        "returncode": child.returncode,
                        "stdout": child.stdout,
                        "stderr": child.stderr,
                        "error": f"worker did not emit JSON: {exc}",
                    }
                )
                if not args.continue_on_error:
                    break
                continue

            if result.get("skipped"):
                # Out of scope, not broken: keep going even without
                # --continue-on-error, and record why the cell is empty so the
                # report never just omits a configuration.
                skipped.append(
                    {
                        "configuration": label,
                        "model": model,
                        "method": method,
                        "reason": result.get("reason", ""),
                    }
                )
                print(f"  SKIPPED: {result.get('reason', '')}")
                continue

            if args.perf_stat:
                result["hardware_counters"] = {
                    "tool": "linux_perf_stat",
                    "events": list(PERF_EVENTS),
                    "counters": parse_perf_stat_csv(child.stderr),
                }

            results.append(result)
            wall = result["wall_clock_ms"]
            rss = result["rss_kb"]
            print(
                f"  mean={wall['mean']:.3f} ms median={wall['median']:.3f} ms "
                f"peak_rss={rss['peak_during_forward_loop']} KB "
                f"layers={result['method_meta']['num_transformed_layers']}"
            )
        if errors and not args.continue_on_error:
            break

    report = {
        "metadata": {
            "benchmark_type": (
                "real_nuimages_single_image_edgeslice_sconv"
                if args.input_mode == "single-image"
                else "real_nuimages_sixcam_edgeslice_sconv"
            ),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "no_dummy_inputs": True,
            "no_repeated_images": True,
            "no_analytical_estimates": True,
            "statistical_tests": "median + 95% bootstrap CI + paired Wilcoxon signed-rank",
            "numerical_equivalence": "numpy.allclose for transformed outputs where a PyTorch reference is available",
            # Toolchain identity, so a result can be traced to what produced
            # it. Recorded unconditionally: the absence of an upstream build is
            # itself information when a report has no sconv_upstream arm.
            "toolchain": _toolchain_metadata(args),
            "nuimages_root": str(args.nuimages_root),
            "version": args.version,
            "input_mode": args.input_mode,
            "batch_count": args.batches if args.input_mode == "sixcam" else None,
            "requested_images": args.image_count or args.batches,
            "camera_count": 6 if args.input_mode == "sixcam" else None,
            "image_count": manifest.get("image_count"),
            "flattened_image_count": (
                manifest.get("flattened_image_count") or manifest.get("image_count")
            ),
            "resolution_hw": [height, width],
            "normalize": args.normalize,
            "models": args.models,
            "methods": args.methods,
            "l1_cache_kb": args.l1_cache_kb,
            "runs": args.runs,
            "warmup": args.warmup,
            "num_threads": args.num_threads,
            "pin_cpu": args.pin_cpu,
            "perf_stat": args.perf_stat,
            "perf_events": list(PERF_EVENTS) if args.perf_stat else [],
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
        },
        "input_manifest": manifest,
        "results": results,
        "errors": errors,
        "skipped": skipped,
        "summary": _summarize_results(results),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {output}")
    return report


def _inspect_only(args: argparse.Namespace) -> Dict[str, Any]:
    if args.input_mode == "single-image":
        image_count = args.image_count or args.batches
        paths = _list_single_image_paths(
            args.nuimages_root,
            image_dir=args.image_dir,
            include_sweeps=args.include_sweeps,
        )
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_mode": "single-image",
            "requested_images": image_count,
            "available_images": len(paths),
            "can_run_requested_benchmark": len(paths) >= image_count,
            "image_root": str(args.image_dir or args.nuimages_root / "samples"),
            "first_paths": [str(path) for path in paths[: min(10, len(paths))]],
        }
        print(json.dumps(payload, indent=2))
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2) + "\n")
        return payload

    summary = summarize_sixcam_availability(args.nuimages_root, version=args.version)
    complete = summary["complete_sixcam_samples"]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "requested_batches": args.batches,
        "can_run_requested_benchmark": complete >= args.batches,
        "availability": summary,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark EdgeSlice vs SConv on 10 real synchronized six-camera batches"
    )
    parser.add_argument("--nuimages-root", type=Path, default=DEFAULT_NUIMAGES_ROOT)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument(
        "--input-mode",
        choices=("sixcam", "single-image"),
        default="sixcam",
        help="Use synchronized six-camera batches, or run one real image per forward",
    )
    parser.add_argument(
        "--image-count",
        type=int,
        default=None,
        help="Number of images for --input-mode single-image (default: --batches)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Image directory for --input-mode single-image (recursive)",
    )
    parser.add_argument(
        "--include-sweeps",
        action="store_true",
        help="Include nuImages sweeps as well as samples for single-image mode",
    )
    parser.add_argument("--resolution", type=_parse_resolution, default=(224, 224))
    parser.add_argument("--normalize", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, default=["resnet50", "mobilenet_v2"])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--l1-cache-kb", type=int, default=256)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--pin-cpu", type=int, default=None)
    parser.add_argument(
        "--perf-stat",
        action="store_true",
        help="Wrap each worker in linux perf stat and collect L1/LLC cache counters",
    )
    parser.add_argument("--rss-sample-interval-ms", type=float, default=2.0)
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="resnet50", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=SUPPORTED_METHODS, default="baseline", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.num_threads < 1:
        parser.error("--num-threads must be >= 1")

    if args.worker:
        if args.manifest is None:
            parser.error("--worker requires --manifest")
        try:
            print(json.dumps(_run_worker(args)))
        except UnsupportedConfiguration as exc:
            # Reported through the normal stdout-is-JSON channel and exit 0:
            # the parent records it as skipped rather than as a failed run.
            print(json.dumps({"skipped": True, "reason": str(exc)}))
        except Exception as exc:
            print(f"worker failed: {exc}", file=sys.stderr)
            raise
        return

    if args.inspect_only:
        _inspect_only(args)
        return

    try:
        report = _run_parent(args)
    except ValueError as exc:
        print(f"Benchmark aborted: {exc}", file=sys.stderr)
        sys.exit(2)

    if report["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
