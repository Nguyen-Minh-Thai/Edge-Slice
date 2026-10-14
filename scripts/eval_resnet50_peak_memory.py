#!/usr/bin/env python3
"""ResNet-50 peak memory, measured — frontend estimate, bufferized peak, RSS.

This script used to report the analytical frontend estimate alone, as a
headline "peak reduction" percentage. That number is not a measurement of peak
memory: across the evaluation matrix the estimator over-predicts the bufferized
peak in 24 of 24 arms, with a median absolute error of 359% (see
docs/PHASE1_METRIC_CORRECTION.md). It orders configurations correctly and does
not quantify them, so quoting it as a reduction was reporting the wrong
quantity.

It now compiles the payload and reports all three numbers per arm. It is a thin
front end over ``benchmarks.run_evaluation``, which owns the measurement path;
this exists only to keep the single-model entry point that scripts and notes
already reference.

Usage:
    python scripts/eval_resnet50_peak_memory.py
    python scripts/eval_resnet50_peak_memory.py --input-size 1024 --l1-kb 256
    python scripts/eval_resnet50_peak_memory.py --output results/resnet50.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--l1-kb", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--cachegrind",
        action="store_true",
        help="Also simulate D1/LL misses against the target cache geometry. "
        "Slow: valgrind interprets every instruction, and a scalar ResNet-50 "
        "payload takes hours.",
    )
    parser.add_argument(
        "--physical-dir",
        default="",
        help="Keep the compiled IR and binaries here instead of a temp dir.",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    from benchmarks.run_evaluation import evaluate_configuration

    try:
        entry = evaluate_configuration(
            "resnet50",
            args.input_size,
            args.input_size,
            args.l1_kb,
            batch_size=args.batch_size,
            physical=True,
            physical_dir=args.physical_dir or None,
            cachegrind=args.cachegrind,
        )
    except RuntimeError as exc:
        print(f"Evaluation aborted: {exc}")
        return 2

    metrics = entry["metrics"]
    provenance = entry["weights_provenance"]
    physical = entry.get("physical") or {}

    print("\n=== ResNet-50 peak memory ===")
    print(f"Input shape:     {tuple(entry['input_shape'])}")
    print(f"Cache budget:    {entry['l1_cache_kb']} KB")
    print(f"DAG:             {entry['dag_nodes']} nodes, {entry['dag_edges']} edges")
    print(
        f"Weights:         {provenance['weights_source']} "
        f"sha256={provenance['state_dict_sha256'][:16]} "
        f"({provenance['parameter_count']:,} params)"
    )

    if physical.get("error"):
        print(f"\nMeasurement unavailable: {physical['error'][:300]}")
        return 1
    if not physical.get("arms"):
        print("\nNo arms were measured.")
        return 1

    print(
        f"\n{'arm':<13}{'frontend est.':>15}{'bufferized':>13}"
        f"{'measured RSS':>15}{'est. error':>12}"
    )
    print("-" * 68)
    for arm in physical["arms"]:
        error = arm["frontend_vs_bufferized_error_pct"]
        error_text = f"{error:+.0f}%" if error is not None else "n/a"
        print(
            f"{arm['arm']:<13}{arm['frontend_estimate_mb']:>13.1f} MB"
            f"{arm['bufferized_peak_mb']:>11.1f} MB"
            f"{arm['measured_rss']['peak_rss_mb']:>13.1f} MB"
            f"{error_text:>12}"
        )
        for failure in arm["errors"]:
            print(f"{'':<13}  ! {failure[:100]}")

    if "bufferized_peak_reduction_pct" in physical:
        print(
            f"\nBufferized peak reduction (tiled+fused vs untiled): "
            f"{physical['bufferized_peak_reduction_pct']}%"
        )
    if "measured_rss_reduction_pct" in physical:
        print(
            f"Measured RSS reduction:                            "
            f"{physical['measured_rss_reduction_pct']}%"
        )
    print(
        f"\nFrontend estimate alone would have reported "
        f"{metrics['peak_memory_reduction_pct']}% — kept in the JSON as a "
        "predictor, not a result."
    )

    if args.output:
        import json

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(entry, indent=2))
        print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
