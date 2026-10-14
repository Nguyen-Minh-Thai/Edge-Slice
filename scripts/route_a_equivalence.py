#!/usr/bin/env python3
"""Route A: does the refined ownership pipeline match the legacy pass, exactly?

The legacy `--buffer-deallocation` reaches the achievable peak but predates the
ownership model and upstream has superseded it. The ownership pass that
replaces it parks every free at the block exit, which on a single-block forward
pass means the peak is the sum of every allocation. `mlir/` adds a refinement
that discharges each statically-owned buffer from that merged dealloc at its
last use.

The claim this script is here to check is the strong form: the refinement does
not merely land near the legacy pass, it produces *the same program*. Three
levels of evidence, escalating only when the previous one is not conclusive:

  1. stage-1 IR SHA-256. Identical IR is identical behaviour, full stop, and
     needs no run to confirm.
  2. binary SHA-256, for the arms where the IR matched -- a cross-check that
     nothing downstream reintroduces a difference.
  3. execution checksum and peak RSS, for any arm where the IR differed. A
     difference there is a real result and has to be measured, not argued.

    python scripts/route_a_equivalence.py --models resnet18,mobilenet_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", default="resnet18,mobilenet_v2,efficientnet_b0,unet,resnet50")
    parser.add_argument("--arms", default="untiled,tiled,tiled_fused")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--l1-kb", type=int, default=32)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--out", default="benchmarks/results/route_a/route_a_equivalence.json")
    parser.add_argument("--work", default="build/route_a_equivalence")
    args = parser.parse_args()

    import torch

    from benchmarks.alloc_attribution import build_arms
    from benchmarks.hardware_profiler import compile_mlir_artifacts, run_wall_clock
    from benchmarks.workloads import load_model, weights_provenance
    from edgeslide.mlir_transform.errors import require_tool, run_tool
    from edgeslide.mlir_transform.lowering import SLICED_PAYLOAD_OPT_PASSES_STAGE1

    mlir_opt = require_tool("mlir-opt")
    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    results = {
        "resolution": args.resolution,
        "l1_cache_kb": args.l1_kb,
        "stage1_passes_legacy": list(SLICED_PAYLOAD_OPT_PASSES_STAGE1),
        "arms": [],
        "skipped": [],
    }

    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {name} ===", flush=True)
        try:
            with torch.no_grad():
                model = load_model(name).eval()
                provenance = weights_provenance(model)
                arms = build_arms(
                    model,
                    (1, 3, args.resolution, args.resolution),
                    l1_cache_kb=args.l1_kb,
                    work_dir=work / name,
                    model_name=name,
                )
        except Exception as exc:
            print(f"  ARMS FAILED {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            results["skipped"].append({"model": name, "stage": "build_arms",
                                       "error": str(exc)[:400]})
            continue

        for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
            if arm not in arms:
                continue
            record = {"model": name, "arm": arm, "weights_provenance": provenance}
            try:
                # The refined compile also writes the prepared payload and its
                # own stage-1 output, which is what makes the IR comparison
                # possible without reimplementing the preparation steps.
                refined_dir = work / f"{name}_{arm}_refined"
                refined_bin, in_shape, out_shape, _, _ = compile_mlir_artifacts(
                    arms[arm], refined_dir, "refined", use_refined_deallocation=True
                )
                prepared = refined_dir / "sliced_payload_prepared.mlir"
                refined_stage1 = refined_dir / "sliced_payload_stage1.mlir"

                # The legacy stage 1, run on the identical prepared payload so
                # the only difference between the two is the deallocation.
                legacy_stage1 = refined_dir / "legacy_stage1.mlir"
                run_tool(
                    [mlir_opt, str(prepared), *SLICED_PAYLOAD_OPT_PASSES_STAGE1,
                     "-o", str(legacy_stage1)],
                    stage=f"legacy stage1[{name}/{arm}]",
                )

                legacy_dir = work / f"{name}_{arm}_legacy"
                legacy_bin, _, _, _, _ = compile_mlir_artifacts(
                    arms[arm], legacy_dir, "legacy"
                )
            except Exception as exc:
                print(f"  {arm:12s} BUILD FAILED {type(exc).__name__}: {str(exc)[:160]}",
                      flush=True)
                results["skipped"].append({"model": name, "arm": arm,
                                           "error": str(exc)[:400]})
                continue

            record["input_shape"] = list(in_shape)
            record["output_shape"] = list(out_shape)
            record["stage1_sha_refined"] = sha256(refined_stage1)
            record["stage1_sha_legacy"] = sha256(legacy_stage1)
            record["stage1_identical"] = (
                record["stage1_sha_refined"] == record["stage1_sha_legacy"]
            )
            record["binary_sha_refined"] = sha256(Path(refined_bin))
            record["binary_sha_legacy"] = sha256(Path(legacy_bin))
            record["binary_identical"] = (
                record["binary_sha_refined"] == record["binary_sha_legacy"]
            )

            # Level 3. Run both whenever the IR differed -- and also when the
            # binaries differ despite identical IR, which would mean something
            # downstream is not deterministic and is worth knowing.
            if not record["stage1_identical"] or not record["binary_identical"]:
                for label, binary in (("refined", refined_bin), ("legacy", legacy_bin)):
                    stats = run_wall_clock(binary, trials=args.trials, warmup=1,
                                           iters_per_trial=1)
                    record[f"checksum_{label}"] = stats.checksum
                    record[f"peak_rss_kb_{label}"] = stats.max_rss_kb
                    record[f"mean_ms_{label}"] = stats.mean_s * 1000.0
                    record[f"min_ms_{label}"] = stats.min_s * 1000.0
                record["checksum_identical"] = (
                    record["checksum_refined"] == record["checksum_legacy"]
                )

            # Two full lowerings of a 224x224 payload are several gigabytes of
            # intermediate IR and object files per arm, and there are fifteen
            # arms. Everything this script reports has already been read out of
            # them by this point, so they go now rather than at the end.
            for directory in (refined_dir, legacy_dir):
                shutil.rmtree(directory, ignore_errors=True)

            verdict = (
                "IR identical" if record["stage1_identical"]
                else ("same checksum" if record.get("checksum_identical")
                      else "DIFFERS")
            )
            print(f"  {arm:12s} {verdict:16s} "
                  f"stage1 {'==' if record['stage1_identical'] else '!='} "
                  f"binary {'==' if record['binary_identical'] else '!='}",
                  flush=True)
            results["arms"].append(record)

        shutil.rmtree(work / name, ignore_errors=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    total = len(results["arms"])
    same_ir = sum(1 for r in results["arms"] if r["stage1_identical"])
    differing = [r for r in results["arms"]
                 if not r["stage1_identical"] and not r.get("checksum_identical")]
    print(f"\n{same_ir}/{total} arms produce byte-identical stage-1 IR; "
          f"{len(differing)} arms differ in output; {len(results['skipped'])} skipped")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
