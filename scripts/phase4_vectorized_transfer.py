#!/usr/bin/env python3
"""Phase 4: do the tiling decisions still pay off once codegen vectorizes?

The default lowering emits a scalar payload, because MLIR 19's
`transform.structured.vectorize` refuses 2D convolution (see
docs/EDGESLICE_TRANSFORM_ARCHITECTURE.md for the six attempts). That is a
codegen limitation, not a tiling one — but it invites a fair objection: *would
the tiling decision still matter against a real vectorized backend, or is the
measured cache benefit an artifact of slow scalar code?*

This answers it on one real pretrained model, which is all that is needed. Each
arm is compiled twice — once scalar, once with `enable_stride1_vectorization`,
which outlines stride-1 `linalg.conv_2d_nchw_fchw` into helpers the legacy
affine vectorizer *can* handle — and the tiled-vs-untiled comparison is made
**within** each codegen mode. If the sign and rough magnitude of the
cache-traffic difference survive vectorization, the decision transfers.

The vehicle is ResNet-18 with published ImageNet weights, not a synthetic conv
stack: a hand-built stack would be a model written to suit the measurement.
ResNet-18 has the dense stride-1 3x3 convolutions the outliner matches (the
depthwise models match none) and is small enough that cachegrind, which
interprets every instruction, completes.

Cachegrind is the metric, simulated against the target's cache geometry, for
the same reason as everywhere else: it is deterministic, and wall clock on a
development host is not.

    python scripts/phase4_vectorized_transfer.py --model mobilenet_v2 --resolution 64
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import benchmarks  # noqa: E402,F401  (applies the aarch64 torch workaround)
import torch  # noqa: E402
from benchmarks.alloc_attribution import build_arms  # noqa: E402
from benchmarks.physical import (  # noqa: E402
    _cachegrind,
    hardware_counters_available,
    measure_peak_rss,
)
from benchmarks.workloads import (  # noqa: E402
    SUPPORTED_MODELS,
    load_model,
    resolution_to_input_shape,
)


def compile_arm(
    payload: Path, work_dir: Path, label: str, *, vectorize: bool
) -> Dict[str, Any]:
    """Compile one arm in one codegen mode; report what actually came out.

    ``vectorized_ops`` is read back from the emitted IR rather than assumed:
    the lowering falls back to the scalar pipeline whenever the vectorized one
    fails to lower, so "requested vectorization" and "got vectorization" are
    different facts and conflating them would let a scalar run masquerade as a
    vectorized one.
    """
    from benchmarks.hardware_profiler import compile_mlir_artifacts

    out: Dict[str, Any] = {"arm": label, "vectorization_requested": vectorize}
    try:
        binary, _, _, _, llvm_ir = compile_mlir_artifacts(
            payload,
            work_dir / f"{label}_{'vec' if vectorize else 'scalar'}",
            label,
            enable_vectorization=vectorize,
        )
    except Exception as exc:
        out["error"] = str(exc)[:400]
        return out

    ir_text = Path(llvm_ir).read_text()
    # `<N x T>` is LLVM's vector type syntax, and the LLVM IR is the only place
    # the evidence survives: MLIR's own `vector.*` ops are consumed by
    # --convert-vector-to-llvm before stage 2 ends, so counting them there
    # always reads zero whether or not vectorization happened.
    out["vectorized_ops"] = len(re.findall(r"<\d+ x (?:float|half|double)>", ir_text))
    out["binary"] = binary
    out["rss_kb"] = measure_peak_rss([binary, "1"], pin_cpu=0).peak_rss_kb
    return out


def _validity_warnings(rows: List[Dict[str, Any]]) -> List[str]:
    """Report confounds that make a within-mode comparison unreadable.

    The comparison is only about tiling if the arms differ in *nothing else*.
    Two confounds were found the hard way, both after a single-resolution run
    had already been written up:

    * **Unequal vectorization.** `outline_stride1_convs` matches per-op, and a
      tiled payload can present fewer matchable convolutions than an untiled
      one — at 64x64 the untiled arm vectorized 133 ops and the tiled arms 93.
      A "within-mode" difference there measures tiling *plus* 40 fewer
      vectorized convolutions.
    * **A memory blow-up in one arm.** At 96x96 the vectorized tiled arms used
      276 MB against the untiled arm's 91 MB, and their last-level misses rose
      accordingly. A cache comparison across a 3x difference in footprint is
      not a cache comparison.

    Printed rather than raised: the numbers are still worth seeing, and the
    reader needs to know which rows they cannot compare.
    """
    warnings: List[str] = []
    by_mode: Dict[bool, List[Dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(bool(row["vectorization_requested"]), []).append(row)

    for vectorized, mode_rows in sorted(by_mode.items()):
        label = "vector" if vectorized else "scalar"

        counts = {r["arm"]: r.get("vectorized_ops", 0) for r in mode_rows}
        if len(set(counts.values())) > 1:
            warnings.append(
                f"\n!! {label}: arms did NOT vectorize equally {counts}. "
                "The within-mode difference for this row measures tiling plus "
                "a different amount of vectorization, and is not attributable "
                "to tiling."
            )

        rss = {r["arm"]: r.get("rss_kb", 0) for r in mode_rows if r.get("rss_kb")}
        if len(rss) > 1:
            low, high = min(rss.values()), max(rss.values())
            if low and high / low > 1.5:
                worst = max(rss, key=lambda arm: rss[arm])
                warnings.append(
                    f"\n!! {label}: arm footprints differ by "
                    f"{high / low:.1f}x ({worst} at {high / 1024:.0f} MB vs "
                    f"{low / 1024:.0f} MB). A cache comparison across that "
                    "difference measures the footprint, not the tiling."
                )

    if not warnings:
        warnings.append(
            "\nValidity: arms vectorized equally and their footprints are "
            "within 1.5x, so the within-mode differences are attributable to "
            "tiling."
        )
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=list(SUPPORTED_MODELS),
        help="Real pretrained workload. ResNet-18 by default: it has the same "
        "dense stride-1 3x3 convolutions as ResNet-50 (the only form the "
        "outliner matches, which rules out the depthwise models) at a third "
        "the parameters, and cachegrind interprets every instruction — a "
        "scalar ResNet-50 payload cannot be simulated in reasonable time.",
    )
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--l1-kb", type=int, default=32)
    parser.add_argument("--vector-width", type=int, default=8)
    parser.add_argument("--device", default="pi4")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--no-cachegrind",
        action="store_true",
        help="Skip cache simulation (compile and RSS only).",
    )
    args = parser.parse_args()

    model = load_model(args.model).eval()
    shape = resolution_to_input_shape(args.resolution, args.resolution)

    from benchmarks.workloads import weights_provenance

    provenance = weights_provenance(model)
    print(
        f"model={args.model} weights={provenance['weights_source']} "
        f"sha256={provenance['state_dict_sha256'][:16]} "
        f"params={provenance['parameter_count']:,}"
    )
    if not provenance["is_pretrained"]:
        raise SystemExit(
            "refusing to report a codegen comparison from non-pretrained "
            f"weights ({provenance['weights_source']})"
        )

    rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="edgeslide_phase4_") as temp:
        work = Path(temp)
        with torch.no_grad():
            arms = build_arms(
                model,
                shape,
                l1_cache_kb=args.l1_kb,
                work_dir=work,
                model_name=args.model,
            )
        for arm, payload in arms.items():
            for vectorize in (False, True):
                row = compile_arm(payload, work, arm, vectorize=vectorize)
                if (
                    not args.no_cachegrind
                    and "binary" in row
                    and hardware_counters_available()
                ):
                    tag = f"cg_{arm}_{'vec' if vectorize else 'scalar'}"
                    row["cachegrind"] = _cachegrind(
                        row["binary"], work / tag, device=args.device
                    )
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "binary"}),
                      flush=True)

    header = (
        f"{'arm':<12}{'codegen':<9}{'vec ops':>9}{'D1 misses':>14}"
        f"{'LL misses':>12}{'RSS MB':>9}"
    )
    print(f"\n=== Does the tiling decision survive vectorization? "
          f"{args.model} @ {args.resolution}^2, {args.device} cache ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        cg = row.get("cachegrind") or {}
        mode = "vector" if row["vectorization_requested"] else "scalar"
        # A vectorized *request* that produced no vector ops fell back.
        if row["vectorization_requested"] and not row.get("vectorized_ops"):
            mode = "vec(fell back)"
        print(
            f"{row['arm']:<12}{mode:<9}{row.get('vectorized_ops', 0):>9}"
            f"{cg.get('D1', 0):>14,}{cg.get('LL', 0):>12,}"
            f"{row.get('rss_kb', 0) / 1024:>9.1f}"
            + (f"  {row['error'][:60]}" if row.get("error") else "")
        )

    print(
        "\nRead this by comparing tiled_fused against untiled *within* a codegen\n"
        "mode, never across modes: the two modes emit different code, so a\n"
        "cross-mode difference measures the vectorizer, not the tiling."
    )

    for line in _validity_warnings(rows):
        print(line)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "model": args.model,
            "resolution": args.resolution,
            "device": args.device,
            "weights_provenance": provenance,
            "rows": rows,
        }, indent=2))
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
