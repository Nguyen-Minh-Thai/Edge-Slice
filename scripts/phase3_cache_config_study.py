#!/usr/bin/env python3
"""Phase 3a: tile-admission distribution over *real* edge cache geometries.

The evaluation used to sweep "L1 = 256 KB / 512 KB / 1 MB". Those are not L1
sizes — no shipping edge CPU has an L1D above ~64 KB, and this repo's own
cachegrind model already records the Raspberry Pi 4's Cortex-A72 at 32 KB.
Sweeping L2-sized budgets as if they were L1 made almost every op look
1D-tileable, which hid the behaviour that matters: at a real L1 most large
activations have **no feasible 1D tile at all**, and the 2D cascade is what
keeps them tileable rather than dropped.

For each (model, device) this reports how each tiled op was admitted:

    1D          a single dimension had a budget-fitting tile
    2D          every single dimension overflowed at unit size; a dimension
                *pair* fitted (the cascade earning its place)
    L2-resident the tile fits L2 but not L1 — it streams through L1
    rejected    no dimension and no pair fitted; the op is not tiled

    python scripts/phase3_cache_config_study.py --models resnet50 mobilenet_v2
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import benchmarks  # noqa: E402,F401  (applies the aarch64 torch workaround)
import torch  # noqa: E402
from benchmarks.workloads import (  # noqa: E402
    load_model,
    resolution_to_input_shape,
    weights_provenance,
)

from edgeslide.dag.parser import DAGParser  # noqa: E402
from edgeslide.mlir_transform.pipeline import TRACING_SEED  # noqa: E402
from edgeslide.tiling.tile_size import EDGE_CACHE_LEVELS, HardwareConfig  # noqa: E402
from edgeslide.transform.pass_manager import (  # noqa: E402
    EdgeSlicePassConfig,
    EdgeSlicePassManager,
)


def study_one(dag, device: str) -> Dict[str, Any]:
    """Admission distribution for one DAG on one device's cache geometry."""
    hardware = HardwareConfig.for_device(device)
    config = EdgeSlicePassConfig(
        l1_cache_bytes=hardware.l1_cache_bytes, hardware_config=hardware
    )
    manager = EdgeSlicePassManager(dag, config=config)
    results = manager.run()
    plan = results["rewrite_plan"]
    rejections = results.get("planning_rejections", [])
    candidates = results.get("slicing_candidates", []) or []
    # `rewrite_plan` holds anchors only. Producers the fusion planner absorbs
    # into an anchor's scf.forall leave the plan and reappear as that anchor's
    # `fused_producer_op_ids`, so anchor count alone is not "ops tiled" and
    # cannot be compared across cache sizes: a larger budget admits longer
    # fusion chains and therefore *fewer* anchors covering *more* ops.
    absorbed = results.get("absorbed_into_fusion", []) or []
    fused_members = {
        member for entry in plan for member in (entry.get("fused_producer_op_ids") or [])
    }

    # "Rejected" conflates three outcomes that mean opposite things, and
    # reporting them as one number reads as a larger cache performing worse:
    # a Pi 4's 1 MB L2 leaves 89 ops needing no tiling at all, which the raw
    # count showed as 89 more "rejections" than a 256 KB L2.
    reasons = collections.Counter()
    for entry in rejections:
        reason = str(entry.get("reason", ""))
        if "nothing to slice" in reason:
            reasons["fits_whole"] += 1
        elif "no slice dimension fits" in reason:
            reasons["no_feasible_tile"] += 1
        else:
            reasons["not_tileable"] += 1

    by_dimensionality = collections.Counter(
        int(entry.get("tile_dimensionality", 1)) for entry in plan
    )
    by_level = collections.Counter(
        str(entry.get("tile_budget_level", "L1")) for entry in plan
    )
    utilizations = [
        float(entry.get("tile_budget_utilization", 0.0))
        for entry in plan
        if entry.get("tile_budget_utilization") is not None
    ]
    return {
        "device": device,
        "device_description": EDGE_CACHE_LEVELS[device].get("description", ""),
        "l1_cache_bytes": hardware.l1_cache_bytes,
        "l2_cache_bytes": hardware.l2_cache_bytes,
        "l1_budget_bytes": hardware.l1_budget_bytes,
        "anchors": len(plan),
        "fused_into_anchors": len(fused_members),
        "ops_covered": len(plan) + len(fused_members),
        "absorbed_into_fusion": len(absorbed),
        "rejected_ops": len(rejections),
        # Already budget-resident: no tiling needed. A success, not a failure.
        "fits_whole": reasons["fits_whole"],
        # The genuine rejection: no dimension and no pair had a fitting tile.
        "no_feasible_tile": reasons["no_feasible_tile"],
        # Structural, budget-independent (multi-output, unsupported layout).
        "not_tileable": reasons["not_tileable"],
        "candidates": len(candidates),
        "tiled_1d": by_dimensionality.get(1, 0),
        "tiled_2d": by_dimensionality.get(2, 0),
        "l1_resident": by_level.get("L1", 0),
        "l2_resident": by_level.get("L2", 0),
        "mean_budget_utilization": (
            round(sum(utilizations) / len(utilizations), 4) if utilizations else None
        ),
        "min_budget_utilization": round(min(utilizations), 4) if utilizations else None,
        "fusion_budget_warnings": len(results.get("fusion_budget_warnings", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["resnet50"])
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument(
        "--devices", nargs="+", default=list(EDGE_CACHE_LEVELS), choices=list(EDGE_CACHE_LEVELS)
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for name in args.models:
        model = load_model(name)
        shape = resolution_to_input_shape(args.resolution, args.resolution)
        with torch.no_grad():
            # Same seed as the compile pipeline, so a DAG built here and one
            # built there are the same DAG.
            generator = torch.Generator().manual_seed(TRACING_SEED)
            dag = DAGParser(model, torch.randn(*shape, generator=generator)).parse()
        # Which weights produced these rows. Not decoration: the U-Net rows in
        # docs/PHASE3_CACHE_AND_BASELINES.md went stale when `load_model("unet")`
        # switched from a vendored architecture to the published torch.hub one
        # — a change of 2,977 parameters, so a different network — and nothing
        # in the artifact recorded which of the two it had measured. Stamped
        # per row so a row can be checked against the model it names.
        provenance = weights_provenance(model)
        for device in args.devices:
            row = study_one(dag, device)
            row["model"] = name
            row["resolution"] = args.resolution
            row["weights_provenance"] = provenance
            rows.append(row)

    header = (
        f"{'model':<16}{'device':<14}{'L1':>7}{'L2':>8}{'cand':>6}{'anch':>6}"
        f"{'fused':>6}{'cover':>6}{'1D':>5}{'2D':>5}{'L1r':>5}{'L2r':>5}"
        f"{'whole':>7}{'nofit':>7}{'until':>7}{'util':>7}"
    )
    print(f"\n=== Tile admission by real cache geometry (res {args.resolution}) ===")
    print(header)
    print("-" * len(header))
    for r in rows:
        l2 = f"{r['l2_cache_bytes'] // 1024}K" if r["l2_cache_bytes"] else "-"
        util = f"{r['mean_budget_utilization']:.1%}" if r["mean_budget_utilization"] else "-"
        print(
            f"{r['model']:<16}{r['device']:<14}{r['l1_cache_bytes'] // 1024:>6}K{l2:>8}"
            f"{r['candidates']:>6}{r['anchors']:>6}{r['fused_into_anchors']:>6}"
            f"{r['ops_covered']:>6}{r['tiled_1d']:>5}{r['tiled_2d']:>5}"
            f"{r['l1_resident']:>5}{r['l2_resident']:>5}"
            f"{r['fits_whole']:>7}{r['no_feasible_tile']:>7}"
            f"{r['not_tileable']:>7}{util:>7}"
        )
    print(
        "\ncand=liveness candidates | anch=scf.forall anchors | fused=producers "
        "absorbed into an anchor's loop | cover=anch+fused, the ops tiling "
        "reaches | 1D/2D=how the anchor's tile was found | L1r/L2r=which level "
        "holds it | whole=already budget-resident, no tiling needed | "
        "nofit=no dimension or pair fitted (the real rejection) | "
        "until=structurally untileable, budget-independent | util=mean share "
        "of the binding budget claimed."
    )

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows": rows}, indent=2))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
