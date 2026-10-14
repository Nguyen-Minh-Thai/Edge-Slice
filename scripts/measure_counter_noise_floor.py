#!/usr/bin/env python3
"""Measure the reproducibility floor of the cache counters.

Every cache-traffic result in this project is a *difference* between two
binaries, so the smallest difference worth reporting is set by how much the
counters move when nothing meaningful changes. That floor was asserted
("deterministic and reproducible") before it was measured. This measures it.

Two sources are separated, because they behave differently:

* **Same binary, repeated runs.** Cachegrind simulates a cache from the
  instruction stream, so this should be exactly zero. Verified: 0 variation
  over three runs.
* **Rebuilds in different directories.** This is not zero. The work-directory
  path reaches the process — argv, environment, the paths embedded by the
  driver — and shifts stack and heap layout enough to move a handful of
  accesses across cache-line boundaries. Measured at 1-15 D1 misses out of
  316,020, i.e. **<= 0.005%**.

The second number is the one that matters for a result file: two runs of this
repo on the same machine in different scratch directories will not produce
byte-identical counters, and a reported effect has to clear that floor.

    python scripts/measure_counter_noise_floor.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _profile_once(work_dir: Path, model: str, l1_kb: int, device: str) -> Dict[str, Any]:
    """One full build-and-measure, in its own directory."""
    output = work_dir.with_suffix(".json")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "hardware_profiler.py"),
            "--models",
            model,
            "--l1-cache-kb",
            str(l1_kb),
            "--device",
            device,
            "--work-dir",
            str(work_dir),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text())["models"][0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiny_conv")
    parser.add_argument("--l1-kb", type=int, default=32)
    parser.add_argument("--device", default="pi4")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Rebuilds to compare. Each gets a distinct directory *and* a "
        "distinct path length, because path length is one of the things that "
        "moves the counters.",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if shutil.which("valgrind") is None:
        raise SystemExit("valgrind not found; cannot measure counter noise")

    runs: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="edgeslide_noise_") as temp:
        base = Path(temp)
        for index in range(args.repeats):
            # Deliberately varying path length: it is a real source of
            # variation, so a floor measured at one fixed length understates it.
            work = base / ("d" * (index + 1) + f"_{index}")
            model = _profile_once(work, args.model, args.l1_kb, args.device)
            runs.append(
                {
                    "path_length": len(str(work)),
                    "d1_original": model["d1_misses_original"],
                    "d1_sliced": model["d1_misses_sliced"],
                    "traffic_reduction_pct": model["traffic_reduction_pct"],
                }
            )
            print(
                f"  rebuild {index + 1} (path len {len(str(work))}): "
                f"orig D1={model['d1_misses_original']:,} "
                f"sliced D1={model['d1_misses_sliced']:,} "
                f"reduction={model['traffic_reduction_pct']:.4f}%"
            )

    def _spread(key: str) -> Dict[str, Any]:
        values = [r[key] for r in runs]
        low, high = min(values), max(values)
        return {
            "min": low,
            "max": high,
            "absolute": high - low,
            "relative_pct": (100.0 * (high - low) / low) if low else None,
        }

    summary = {
        "model": args.model,
        "device": args.device,
        "repeats": args.repeats,
        "d1_original": _spread("d1_original"),
        "d1_sliced": _spread("d1_sliced"),
        "traffic_reduction_pct": _spread("traffic_reduction_pct"),
        "runs": runs,
    }

    floor = max(
        summary["d1_original"]["relative_pct"] or 0.0,
        summary["d1_sliced"]["relative_pct"] or 0.0,
    )
    summary["noise_floor_pct"] = floor
    print(
        f"\nnoise floor across {args.repeats} rebuilds: {floor:.4f}% "
        f"(D1 spread {summary['d1_original']['absolute']} original / "
        f"{summary['d1_sliced']['absolute']} sliced)"
    )
    print(
        "An effect below roughly 10x this is not resolvable by this "
        "measurement and must not be reported as a finding."
    )

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
