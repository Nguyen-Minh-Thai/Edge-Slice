#!/usr/bin/env python3
"""Run a reproducible baseline vs upstream SConvTransform native benchmark.

The upstream pass emits an external ``sgemm_blas_kernel`` symbol.  This runner
links the checked-in ABI-compatible runtime shim, lowers both variants with
the same LLVM/MLIR toolchain, and fails if output checksums differ.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "SConvTransform"
TEMPLATE = UPSTREAM / "scripts" / "convbench" / "payloads" / "correctness_payload.mlir"
RUNTIME = UPSTREAM / "runtime" / "benchmark_runtime.c"

LOWERING_FLAGS = [
    "--one-shot-bufferize=bufferize-function-boundaries",
    "--convert-linalg-to-affine-loops",
    "--canonicalize", "--cse", "--expand-strided-metadata", "--lower-affine",
    "--convert-scf-to-cf", "--normalize-memrefs", "--memref-expand",
    "--finalize-memref-to-llvm", "--convert-func-to-llvm", "--convert-arith-to-llvm",
    "--convert-cf-to-llvm", "--reconcile-unrealized-casts", "--llvm-legalize-for-export",
]
CHECKSUM_RE = re.compile(r"checksum=([-+0-9.eE]+)")
SUMSQ_RE = re.compile(r"sumsq=([-+0-9.eE]+)")
FLOPS_RE = re.compile(r"flops_per_s=([-+0-9.eE]+)")
RSS_RE = re.compile(r"__RSS__(\d+)")


def run(cmd: list[str], *, env: dict[str, str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=timeout)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-5000:]}"
        )
    return result


def make_payload(work: Path, *, ci: int, do: int, hi: int, wi: int, runs: int) -> Path:
    values = {
        "NI": 1, "CI": ci, "HI": hi, "WI": wi, "NO": 1, "DO": do,
        "HO": hi, "WO": wi, "HK": 3, "WK": 3,
        "HPTOP": 1, "HPBOTTOM": 1, "WPLEFT": 1, "WRIGHT": 1,
        "HS": 1, "WS": 1, "HD": 1, "WD": 1, "GROUP": 1, "BIAS": 0,
        "RUNS": runs,
    }
    # The template uses WPRIGHT (the CSV column is WPRIGHT too).
    values["WPRIGHT"] = values.pop("WRIGHT")
    values["PHI"] = hi + 2
    values["PWI"] = wi + 2
    values["FLOPS"] = 2 * hi * wi * do * ci * 3 * 3
    text = TEMPLATE.read_text()
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    payload = work / "payload.mlir"
    payload.write_text(text)
    return payload


def lower(mlir_opt: Path, source: Path, output: Path, *, env: dict[str, str]) -> None:
    run([str(mlir_opt), str(source), *LOWERING_FLAGS, "-o", str(output)], env=env)
    run([str(mlir_opt), str(output), "--verify-each", "-o", os.devnull], env=env)


def run_native(runner: Path, module: Path, runtime: Path, *, env: dict[str, str]) -> dict[str, float]:
    # The runner utility libraries are supplied separately; keep this path
    # construction explicit so a missing LLVM build fails clearly.
    llvm_lib = Path(env["EDGESLIDE_LLVM_LIB"])
    libs = ",".join([
        str(runtime),
        str(llvm_lib / "libmlir_c_runner_utils.so"),
        str(llvm_lib / "libmlir_runner_utils.so"),
    ])
    started = time.perf_counter()
    result = run([
        "/usr/bin/time", "-f", "__RSS__%M", str(runner), str(module),
        "-e", "main", "-entry-point-result=void", f"-shared-libs={libs}",
    ], env=env)
    elapsed = time.perf_counter() - started
    text = result.stderr
    checksum = CHECKSUM_RE.search(text)
    sumsq = SUMSQ_RE.search(text)
    flops = FLOPS_RE.search(text)
    rss = RSS_RE.search(text)
    if checksum is None:
        raise RuntimeError(f"runner emitted no checksum:\n{text[-3000:]}")
    if sumsq is None:
        raise RuntimeError(f"runner emitted no sumsq:\n{text[-3000:]}")
    return {
        "checksum": float(checksum.group(1)),
        "sumsq": float(sumsq.group(1)),
        "flops_per_s": float(flops.group(1)) if flops else float("nan"),
        "peak_rss_kb": float(rss.group(1)) if rss else float("nan"),
        "process_wall_s": elapsed,
    }


def _default_llvm_prefix() -> Path:
    """Prefix holding bin/mlir-opt and lib/, discovered rather than hardcoded.

    A distro llvm-19 install (/usr/lib/llvm-19) has the same bin//lib/ shape as
    a source build, so resolving whichever mlir-opt is on PATH covers both and
    removes the "required file not found: /tmp/llvm-project/build/bin/..."
    failure that only made sense on the original author's machine.
    """
    found = shutil.which("mlir-opt")
    if found is not None:
        # .../bin/mlir-opt -> ...; resolve() so a symlink into an LLVM prefix
        # (as a venv wrapper or /usr/bin/mlir-opt-19 would be) lands correctly.
        candidate = Path(found).resolve().parent.parent
        if (candidate / "lib").is_dir():
            return candidate
    # No usable prefix. Returning a hard-coded /tmp path here produced
    # "required file not found: /tmp/llvm-project/build/bin/mlir-opt", which
    # reads as a missing file rather than as "no LLVM install was found" and
    # sent readers looking for a directory that had never existed on their
    # machine. Fail with the actual problem instead.
    raise SystemExit(
        "no usable LLVM prefix found: mlir-opt is not on PATH, or the prefix "
        "it resolves to has no lib/ directory. Pass --llvm-build "
        "<prefix containing bin/ and lib/>, e.g. /usr/lib/llvm-19."
    )


def _default_sconv_opt() -> Path:
    """First existing upstream sconv-opt among the build layouts we produce."""
    for relative in ("build-llvm19/bin/sconv-opt", "build/bin/sconv-opt"):
        candidate = UPSTREAM / relative
        if candidate.exists():
            return candidate
    found = shutil.which("sconv-opt")
    if found is not None:
        return Path(found)
    return UPSTREAM / "build-llvm19/bin/sconv-opt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci", type=int, default=8)
    parser.add_argument("--do", type=int, default=16)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--backend", choices=("neon", "cblas"), default="neon",
        help="sgemm_blas_kernel backend (default: AArch64 NEON 8x16 microkernel)",
    )
    parser.add_argument(
        "--llvm-build",
        type=Path,
        default=_default_llvm_prefix(),
        help="Prefix containing bin/mlir-opt and lib/. Defaults to the prefix "
        "of whichever mlir-opt is on PATH (which is also how a distro "
        "llvm-19 install is laid out), falling back to a source build at "
        "/tmp/llvm-project/build.",
    )
    parser.add_argument(
        "--sconv-opt",
        type=Path,
        default=_default_sconv_opt(),
        help="Upstream sconv-opt binary; build one with "
        "third_party/SConvTransform/build_llvm19.sh.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/sconv_upstream.json")
    args = parser.parse_args()
    if args.runs < 1 or args.trials < 1:
        raise SystemExit("--runs and --trials must be >= 1")

    llvm_bin = args.llvm_build / "bin"
    llvm_lib = args.llvm_build / "lib"
    runner = llvm_bin / "mlir-cpu-runner"
    mlir_opt = llvm_bin / "mlir-opt"
    for path in (runner, mlir_opt, args.sconv_opt, RUNTIME):
        if not path.exists():
            raise SystemExit(f"required file not found: {path}")

    env = os.environ.copy()
    env["EDGESLIDE_LLVM_LIB"] = str(llvm_lib)
    env["LD_LIBRARY_PATH"] = f"{llvm_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    with tempfile.TemporaryDirectory(prefix="edgeslide_sconv_") as temp:
        work = Path(temp)
        runtime_so = work / "libbenchmark_runtime.so"
        runtime_cmd = ["cc", "-shared", "-fPIC", "-O3"]
        if args.backend == "neon":
            runtime_cmd.append("-DEDGESLIDE_USE_NEON")
        else:
            runtime_cmd.append("-DEDGESLIDE_USE_CBLAS")
        runtime_cmd.extend([str(RUNTIME), "-o", str(runtime_so)])
        if args.backend == "cblas":
            runtime_cmd.append("-lblas")
        run(runtime_cmd, env=env)
        payload = make_payload(work, ci=args.ci, do=args.do, hi=args.height, wi=args.width, runs=args.runs)
        baseline = work / "baseline.llvm.mlir"
        sconv_lowered = work / "sconv.llvm.mlir"
        lower(mlir_opt, payload, baseline, env=env)
        transformed = work / "payload.sconv.mlir"
        transformed_result = run([
            str(args.sconv_opt),
            "-transform=" + str(UPSTREAM / "test/lowering/sconv-blas.mlir"),
            str(payload),
        ], env=env)
        transformed.write_text(transformed_result.stdout)
        lower(mlir_opt, transformed, sconv_lowered, env=env)

        rows: dict[str, list[dict[str, float]]] = {"baseline": [], "sconv_upstream": []}
        for _ in range(args.trials):
            rows["baseline"].append(run_native(runner, baseline, runtime_so, env=env))
            rows["sconv_upstream"].append(run_native(runner, sconv_lowered, runtime_so, env=env))

    baseline_checksums = rows["baseline"]
    sconv_checksums = rows["sconv_upstream"]
    reference = baseline_checksums[0]["checksum"]
    reference_sumsq = baseline_checksums[0]["sumsq"]
    if not math.isfinite(reference) or not math.isfinite(reference_sumsq):
        raise SystemExit("baseline produced a non-finite numerical summary")
    max_abs_error = max(abs(row["checksum"] - reference) for row in sconv_checksums)
    max_sumsq_error = max(abs(row["sumsq"] - reference_sumsq) for row in sconv_checksums)
    if max_abs_error > max(1e-3, abs(reference) * 1e-5) or max_sumsq_error > max(1e-2, abs(reference_sumsq) * 1e-5):
        raise SystemExit(
            f"numerical equivalence failed: checksum_error={max_abs_error} "
            f"sumsq_error={max_sumsq_error}"
        )

    summary = {}
    for method, samples in rows.items():
        summary[method] = {
            "checksum": samples[0]["checksum"],
            "sumsq": samples[0]["sumsq"],
            "median_flops_per_s": statistics.median(s["flops_per_s"] for s in samples),
            "median_peak_rss_kb": statistics.median(s["peak_rss_kb"] for s in samples),
            "median_process_wall_s": statistics.median(s["process_wall_s"] for s in samples),
            "trials": len(samples),
        }
    base_rss = summary["baseline"]["median_peak_rss_kb"]
    sconv_rss = summary["sconv_upstream"]["median_peak_rss_kb"]
    report = {
        "measurement_type": "native_sconv_upstream_runtime",
        "compiler_evidence": True,
        "performance_evidence_scope": "SConvTransform rewrite plus selected local sgemm runtime",
        "runtime_rss_scope": "whole mlir-cpu-runner process (JIT and runtime included)",
        "llvm_build": str(args.llvm_build),
        "sconv_opt": str(args.sconv_opt),
        "sconv_transform": "upstream:test/lowering/sconv-blas.mlir",
        "sconv_runtime": (
            "local AArch64 NEON 8x16 microkernel"
            if args.backend == "neon"
            else "local ABI shim calling system CBLAS (-lblas)"
        ),
        "upstream_kernel_exact": False,
        "kernel_provenance": "SConvTransform emits sgemm_blas_kernel but does not ship its implementation; this run supplies a checked-in NEON/CBLAS-compatible ABI shim.",
        "case": {"ci": args.ci, "do": args.do, "height": args.height, "width": args.width, "runs_per_process": args.runs},
        "numerical_equivalence": {
            "passed": True,
            "max_abs_checksum_error": max_abs_error,
            "max_abs_sumsq_error": max_sumsq_error,
        },
        "results": summary,
        "median_peak_rss_reduction_pct": round(100 * (1 - sconv_rss / base_rss), 2) if base_rss else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
