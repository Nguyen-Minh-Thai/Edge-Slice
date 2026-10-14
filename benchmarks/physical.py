"""Physical peak-memory measurement for one evaluation cell (proposal §5.2).

Every cell of the evaluation matrix reports **three** numbers for the same
payload, so a reader can see where the model stops matching the machine:

1. **frontend estimate** — ``estimate_post_slice_peak_bytes`` on the tensor DAG,
   before bufferization exists. This is a *predictor*, not a measurement.
The three are not one quantity measured three ways. In particular (2) counts
heap allocations while (3) counts every resident page, weights included, and
the weights are 73-96% of the total for the shipped models. Each cell reports
``rss_dilution_factor`` so the difference in scope is a number rather than an
omission; see docs/PHASE1_METRIC_CORRECTION.md.

2. **bufferized peak** — ``BufferizedMemoryAnalyzer.get_peak_last_use()`` on the
   bufferized MLIR. Real ``memref.alloc`` sizes with real lifetimes; last-use
   rather than dealloc placement, because the deallocation pipeline parks frees
   at function exit and that makes the dealloc-based peak degenerate to
   "sum of every allocation".
3. **measured RSS** — high-water mark of the compiled native binary's address
   space, from the OS.

The three disagree for understandable reasons and the gaps are themselves
results: (1) vs (2) is how wrong a pre-bufferization static model is, and
(2) vs (3) is allocator overhead plus weights plus the runtime's own
footprint. Reporting only one of them is what let an analytical model credit a
reduction the rewrite never produced.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_TIME_BIN = "/usr/bin/time"


@dataclass
class RssMeasurement:
    """Peak resident set size of a child process, in kilobytes."""

    peak_rss_kb: int = 0
    method: str = "unavailable"
    exit_code: Optional[int] = None
    stdout: str = ""
    error: str = ""

    @property
    def peak_rss_bytes(self) -> int:
        return self.peak_rss_kb * 1024

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_rss_kb": self.peak_rss_kb,
            "peak_rss_mb": round(self.peak_rss_kb / 1024, 3),
            "method": self.method,
            "exit_code": self.exit_code,
            "error": self.error,
        }


def _rss_via_gnu_time(cmd: Sequence[str], timeout_s: int) -> Optional[RssMeasurement]:
    """``/usr/bin/time -f %M`` — per-invocation, so no cross-run contamination."""
    if not Path(_TIME_BIN).is_file():
        return None
    try:
        proc = subprocess.run(
            [_TIME_BIN, "-f", "%M", *cmd],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return RssMeasurement(method="gnu_time", error=str(exc)[:300])

    # GNU time writes the format string to stderr, after the child's own stderr.
    for line in reversed((proc.stderr or "").strip().splitlines()):
        candidate = line.strip()
        if candidate.isdigit():
            return RssMeasurement(
                peak_rss_kb=int(candidate),
                method="gnu_time_maxrss",
                exit_code=proc.returncode,
                stdout=proc.stdout,
            )
    return RssMeasurement(
        method="gnu_time",
        exit_code=proc.returncode,
        stdout=proc.stdout,
        error="could not parse %M from stderr",
    )


# Run in a *fresh* interpreter: getrusage(RUSAGE_CHILDREN).ru_maxrss is the
# maximum over every child this process has ever reaped, so reading it from a
# long-lived runner returns whichever earlier child was largest. Isolating it in
# a one-shot process makes the number belong to this command alone.
_RUSAGE_HELPER = r"""
import json, os, resource, subprocess, sys
cmd = json.loads(sys.argv[1])
proc = subprocess.run(cmd, capture_output=True, text=True)
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
json.dump(
    {"maxrss": usage.ru_maxrss, "rc": proc.returncode, "stdout": proc.stdout[-4000:]},
    sys.stdout,
)
"""


def _rss_via_getrusage(cmd: Sequence[str], timeout_s: int) -> RssMeasurement:
    """``getrusage(RUSAGE_CHILDREN).ru_maxrss`` from an isolated wrapper."""
    import json

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUSAGE_HELPER, json.dumps(list(cmd))],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        payload = json.loads(proc.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return RssMeasurement(method="getrusage_children", error=str(exc)[:300])

    # ru_maxrss is kilobytes on Linux, bytes on macOS.
    scale = 1 if sys.platform != "darwin" else 1024
    return RssMeasurement(
        peak_rss_kb=int(payload.get("maxrss", 0)) // scale,
        method="getrusage_children_ru_maxrss",
        exit_code=payload.get("rc"),
        stdout=payload.get("stdout", ""),
    )


def measure_peak_rss(
    cmd: Sequence[str],
    *,
    timeout_s: int = 1800,
    pin_cpu: Optional[int] = None,
) -> RssMeasurement:
    """Peak RSS of running *cmd* once, preferring GNU time, falling back cleanly.

    Both methods report the OS high-water mark of the child's resident set, so
    they include the allocator's slack and any weights the binary embeds. That
    is the point: it is the number an edge deployment actually has to fit.
    """
    if pin_cpu is not None and shutil.which("taskset"):
        cmd = ["taskset", "-c", str(pin_cpu), *cmd]

    measurement = _rss_via_gnu_time(cmd, timeout_s)
    if measurement is not None and measurement.peak_rss_kb > 0:
        return measurement
    fallback = _rss_via_getrusage(cmd, timeout_s)
    if fallback.peak_rss_kb > 0:
        return fallback
    # Return whichever carries the more informative error.
    return measurement or fallback


@dataclass
class CellMeasurement:
    """The three memory numbers, plus cache counters, for one payload arm."""

    arm: str
    frontend_estimate_bytes: int = 0
    bufferized_peak_bytes: int = 0
    bufferized_peak_dealloc_bytes: int = 0
    measured_rss: RssMeasurement = field(default_factory=RssMeasurement)
    bufferize_variant: str = "baseline"
    attribution_totals: Dict[str, Any] = field(default_factory=dict)
    cachegrind: Optional[Dict[str, Any]] = None
    binary_path: str = ""
    errors: List[str] = field(default_factory=list)

    def rss_dilution_factor(self) -> Optional[float]:
        """How much larger RSS's scope is than the peak's, or None if unknown.

        ``(weights + peak) / peak``. An X% reduction in the activation peak can
        move measured RSS by at most X divided by this. None when the
        bufferization failed, so there is nothing to divide — never 1.0, which
        would claim the two scopes agree.
        """
        peak = self.bufferized_peak_bytes
        weights = self.attribution_totals.get("constant_global_bytes")
        if not peak or weights is None:
            return None
        return round((peak + weights) / peak, 2)

    def to_dict(self) -> Dict[str, Any]:
        mib = 1024 * 1024
        return {
            "arm": self.arm,
            "bufferize_variant": self.bufferize_variant,
            "frontend_estimate_bytes": self.frontend_estimate_bytes,
            "frontend_estimate_mb": round(self.frontend_estimate_bytes / mib, 3),
            "bufferized_peak_bytes": self.bufferized_peak_bytes,
            "bufferized_peak_mb": round(self.bufferized_peak_bytes / mib, 3),
            "bufferized_peak_dealloc_bytes": self.bufferized_peak_dealloc_bytes,
            "bufferized_peak_dealloc_mb": round(
                self.bufferized_peak_dealloc_bytes / mib, 3
            ),
            "measured_rss": self.measured_rss.to_dict(),
            # Beside the RSS, because this is where the two scopes get
            # conflated. The bufferized peak counts heap allocations;
            # measured RSS counts every resident page, weights included, and
            # the weights are 73-96% of the total. So a reduction of X% in the
            # peak can move RSS by at most X/rss_dilution_factor. Emitted next
            # to the number it qualifies rather than buried in the totals.
            "weight_global_bytes": self.attribution_totals.get(
                "constant_global_bytes"
            ),
            "rss_dilution_factor": self.rss_dilution_factor(),
            "attribution_totals": self.attribution_totals,
            "cachegrind": self.cachegrind,
            "binary_path": self.binary_path,
            "errors": self.errors,
        }


def predictor_error(
    frontend_bytes: int, bufferized_bytes: int
) -> Optional[float]:
    """Signed relative error of the frontend estimate against the real peak.

    Positive means the static model over-estimated. ``None`` when there is no
    measured peak to compare against, which must not be silently read as 0.
    """
    if bufferized_bytes <= 0:
        return None
    return round(100.0 * (frontend_bytes - bufferized_bytes) / bufferized_bytes, 2)


def summarize_predictor_accuracy(
    pairs: Sequence[Any],
) -> Dict[str, Any]:
    """How well the frontend estimate predicts the bufferized peak.

    Takes (frontend_bytes, bufferized_bytes) pairs across the whole matrix and
    reports the statistics a "our static model predicts physical peak within
    X%" claim needs: mean and median absolute relative error, the worst case,
    and Pearson/Spearman correlation. This is a contribution in its own right
    and it is what justifies keeping a pre-bufferization frontend at all.
    """
    import numpy as np

    usable = [
        (float(f), float(b))
        for f, b in pairs
        if b and b > 0 and f is not None
    ]
    if len(usable) < 2:
        return {
            "sample_count": len(usable),
            "reason": "need at least two cells with a measured peak",
        }

    frontend = np.array([f for f, _ in usable])
    physical = np.array([b for _, b in usable])
    relative = (frontend - physical) / physical
    absolute = np.abs(relative)

    result: Dict[str, Any] = {
        "sample_count": len(usable),
        "mean_abs_rel_error_pct": round(float(np.mean(absolute)) * 100, 2),
        "median_abs_rel_error_pct": round(float(np.median(absolute)) * 100, 2),
        "p90_abs_rel_error_pct": round(float(np.quantile(absolute, 0.9)) * 100, 2),
        "max_abs_rel_error_pct": round(float(np.max(absolute)) * 100, 2),
        "mean_signed_rel_error_pct": round(float(np.mean(relative)) * 100, 2),
        "overestimates": int(np.sum(relative > 0)),
        "underestimates": int(np.sum(relative < 0)),
    }
    # Correlation is undefined when either series is constant; report why
    # rather than emitting a NaN that reads as a number.
    if np.std(frontend) == 0 or np.std(physical) == 0:
        result["pearson_r"] = None
        result["correlation_note"] = "one series is constant across the matrix"
        return result
    result["pearson_r"] = round(float(np.corrcoef(frontend, physical)[0, 1]), 4)
    try:
        from scipy.stats import spearmanr

        rho, p_value = spearmanr(frontend, physical)
        result["spearman_rho"] = round(float(rho), 4)
        result["spearman_p"] = float(p_value)
    except Exception:  # scipy optional at this layer
        pass
    return result


def hardware_counters_available() -> bool:
    """Whether cachegrind can run here (it is the deterministic cache metric)."""
    return shutil.which("valgrind") is not None


# Cachegrind prefixes every summary line with "==<pid>== ", so this cannot be
# anchored at the start of the line. The key is bounded on the left by that
# prefix or by line start, and `misses` is matched non-greedily so
# "D1  miss rate:" lines never bind.
_MISS_RE = re.compile(
    r"(?:^|==\s*)(?P<key>D1|LLd|LLi|LL|I1|D)\s+misses:\s+(?P<value>[\d,]+)", re.M
)


def parse_cachegrind_summary(text: str) -> Dict[str, int]:
    """Pull the miss counters out of cachegrind's own summary block."""
    counters: Dict[str, int] = {}
    for match in _MISS_RE.finditer(text):
        counters[match.group("key")] = int(match.group("value").replace(",", ""))
    return counters


def environment_provenance() -> Dict[str, Any]:
    """Tool versions and host facts a result file must carry to be reproducible."""

    def version_of(tool: str, *args: str) -> str:
        path = shutil.which(tool)
        if path is None:
            return "not found"
        try:
            proc = subprocess.run(
                [path, *(args or ("--version",))],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"error: {exc}"
        return (proc.stdout or proc.stderr).strip().splitlines()[0] if (
            proc.stdout or proc.stderr
        ) else "unknown"

    provenance: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "machine": os.uname().machine,
        "mlir_opt": version_of("mlir-opt"),
        "mlir_translate": version_of("mlir-translate"),
        "clang": version_of("clang"),
        "valgrind": version_of("valgrind"),
        "cpu_count": os.cpu_count(),
    }
    for module in ("torch", "torchvision", "numpy", "llvmlite"):
        try:
            provenance[module] = __import__(module).__version__
        except Exception:
            provenance[module] = "not installed"
    try:
        import torch_mlir  # noqa: F401

        provenance["torch_mlir"] = getattr(torch_mlir, "__version__", "installed")
    except Exception:
        provenance["torch_mlir"] = "not installed"
    # The reproducibility risk the risk table did not list: the torch-mlir
    # wheels and the mlir-opt CLI come from different LLVM generations.
    provenance["llvm_generation_mismatch_note"] = (
        "torch-mlir 20240127.1096 is built against its own pinned LLVM; the "
        "mlir-opt/mlir-translate CLI is a separate 19.x install. IR produced by "
        "the former is consumed by the latter, so an LLVM upgrade on either "
        "side can break the pipeline independently of this repo's code."
    )
    return provenance


def measure_arms(
    model: Any,
    input_shape: Sequence[int],
    *,
    l1_cache_kb: int,
    work_dir: Path,
    model_name: str = "model",
    frontend_estimates: Optional[Dict[str, int]] = None,
    bufferize_variant: str = "baseline",
    build_binaries: bool = True,
    run_cachegrind: bool = False,
    cachegrind_device: Optional[str] = "pi4",
) -> List[CellMeasurement]:
    """Compile each arm for real and measure all three memory numbers.

    ``frontend_estimates`` maps arm name → the analytical estimate the frontend
    produced for it, so the predictor-vs-physical gap can be reported per arm.
    Arms whose compilation fails are still returned, carrying the error: a cell
    that could not be built is information, and silently dropping it would make
    the matrix look more complete than it is.
    """
    from benchmarks.alloc_attribution import attribute, build_arms

    work_dir.mkdir(parents=True, exist_ok=True)
    estimates = frontend_estimates or {}
    arms = build_arms(
        model,
        tuple(input_shape),
        l1_cache_kb=l1_cache_kb,
        work_dir=work_dir,
        model_name=model_name,
    )

    measurements: List[CellMeasurement] = []
    for arm, payload_path in arms.items():
        cell = CellMeasurement(
            arm=arm,
            frontend_estimate_bytes=int(estimates.get(arm, 0)),
            bufferize_variant=bufferize_variant,
        )

        attribution = attribute(arm, bufferize_variant, payload_path, work_dir)
        if attribution.ok:
            # Last-use, not dealloc placement: the deallocation pipeline parks
            # frees at function exit, which makes the dealloc peak degenerate
            # to "sum of every allocation" and hides all reuse.
            cell.bufferized_peak_bytes = attribution.peak_last_use_bytes
            cell.bufferized_peak_dealloc_bytes = attribution.peak_dealloc_bytes
            cell.attribution_totals = attribution.totals
        else:
            cell.errors.append(f"bufferize: {attribution.error[:300]}")

        if build_binaries:
            try:
                from benchmarks.hardware_profiler import compile_mlir_artifacts

                binary, _, _, _, _ = compile_mlir_artifacts(
                    payload_path, work_dir / f"bin_{arm}", arm
                )
                cell.binary_path = binary
            except Exception as exc:
                cell.errors.append(f"compile: {str(exc)[:300]}")
            else:
                cell.measured_rss = measure_peak_rss([binary, "1"], pin_cpu=0)
                if run_cachegrind and hardware_counters_available():
                    cell.cachegrind = _cachegrind(
                        binary, work_dir / f"cg_{arm}", device=cachegrind_device
                    )

        measurements.append(cell)
    return measurements


# Cache geometries for `--I1/--D1/--LL` as "size,assoc,line-size". Simulating
# the *target's* cache rather than the development host's is the whole point:
# a miss count against a 6 MB desktop LL says nothing about a Pi 4.
CACHEGRIND_DEVICES: Dict[str, Dict[str, str]] = {
    # Raspberry Pi 4, Cortex-A72: 48 KB 3-way L1I, 32 KB 2-way L1D, 1 MB L2.
    "pi4": {"I1": "49152,3,64", "D1": "32768,2,64", "LL": "1048576,16,64"},
    # Cortex-A55: 32 KB L1I/L1D, 256 KB L2.
    "cortex_a55": {"I1": "32768,4,64", "D1": "32768,4,64", "LL": "262144,8,64"},
    # Cortex-A76: 64 KB L1I/L1D, 512 KB L2.
    "cortex_a76": {"I1": "65536,4,64", "D1": "65536,4,64", "LL": "524288,8,64"},
}


def _cachegrind(
    binary: str, out_dir: Path, *, device: Optional[str] = "pi4"
) -> Dict[str, Any]:
    """D1/LL miss counts for one binary, simulated against a target cache.

    This is the primary runtime metric (proposal §5.3), not a supporting one:
    it is deterministic across runs, whereas wall clock on a shared development
    host is not — a reported speedup on this repo's own dev machine has swung
    from +0.48% to -8.75% (27% relative standard deviation) on unchanged
    binaries. Wall clock belongs on pinned target hardware; cache traffic can
    be measured anywhere and reproduced exactly.

    ``--cache-sim=yes`` is required: valgrind 3.22's cachegrind defaults to
    instruction counting only, so without it the summary carries no D1 or LL
    line at all and the parse comes back empty.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "cachegrind.out"
    command = [
        "valgrind",
        "--tool=cachegrind",
        "--cache-sim=yes",
        f"--cachegrind-out-file={out_file}",
    ]
    geometry = CACHEGRIND_DEVICES.get(device or "", {})
    for level, spec in geometry.items():
        command.append(f"--{level}={spec}")
    command += [binary, "1"]

    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=7200
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)[:300]}
    counters: Dict[str, Any] = dict(parse_cachegrind_summary(proc.stderr or ""))
    counters["exit_code"] = proc.returncode
    counters["simulated_device"] = device
    counters["simulated_geometry"] = geometry
    if proc.returncode == 0 and "D1" not in counters:
        # Fail loudly rather than reporting an empty dict as "no misses".
        counters["error"] = (
            "cachegrind produced no D1 counter; summary was: "
            + " | ".join((proc.stderr or "").strip().splitlines()[-6:])
        )
    return counters
