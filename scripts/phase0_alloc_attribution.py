#!/usr/bin/env python3
"""Phase 0 gate: attribute the bufferized peak of each arm to its allocations.

Produces the alloc-attribution table (paper Figure 2) for untiled / tiled /
tiled+fused payloads across the deallocation and hoisting pass variants. The
gate question this answers: is the tiled arm's peak dominated by per-iteration
allocations inside the `scf.forall` body that a hoisting pass could remove?

    python scripts/phase0_alloc_attribution.py --model resnet50 --resolution 224
    python scripts/phase0_alloc_attribution.py --model tiny_conv --l1-kb 32
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from benchmarks.alloc_attribution import (  # noqa: E402
    BUFFERIZE_VARIANTS,
    PIPELINE_VARIANTS,
    format_table,
    run_study,
    write_study,
)

# Flag-list variants and pipeline-string variants are selectable alike; the
# distinction is an mlir-opt command-line detail, not a property of the study.
ALL_VARIANTS = list(BUFFERIZE_VARIANTS) + list(PIPELINE_VARIANTS)


class _TinyConv(nn.Module):
    """Smallest payload that still produces a tiled scf.forall with fusion."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


def load_model(name: str):
    if name == "tiny_conv":
        model = _TinyConv().eval()
        # Labelled, so the study artifact reads "random_init" rather than
        # "unlabeled". Both come out is_pretrained=False, but "unlabeled" is
        # the answer for a model nobody checked, and this one is known.
        model._edgeslide_weights_source = "random_init"
        model._edgeslide_model_kind = "machinery_check_kernel"
        return model, "tiny_conv"
    from benchmarks import workloads

    # workloads.load_model covers every supported name, including `unet`,
    # which has no `load_unet` of its own — the per-model getattr lookup this
    # replaced silently reported it as "unknown or unavailable".
    if name not in workloads.SUPPORTED_MODELS:
        raise SystemExit(
            f"unknown model {name!r}; known: tiny_conv, "
            + ", ".join(workloads.SUPPORTED_MODELS)
        )
    return workloads.load_model(name).eval(), name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiny_conv")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--l1-kb", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=ALL_VARIANTS,
        choices=ALL_VARIANTS,
    )
    parser.add_argument("--output", default="", help="JSON output path")
    parser.add_argument("--keep-artifacts", default="", help="keep IR in this dir")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    model, model_name = load_model(args.model)
    shape = (args.batch_size, 3, args.resolution, args.resolution)

    if args.keep_artifacts:
        work_dir = Path(args.keep_artifacts)
        work_dir.mkdir(parents=True, exist_ok=True)
        ctx = None
    else:
        ctx = tempfile.TemporaryDirectory(prefix="edgeslide_phase0_")
        work_dir = Path(ctx.name)

    try:
        with torch.no_grad():
            study = run_study(
                model,
                shape,
                l1_cache_kb=args.l1_kb,
                work_dir=work_dir,
                model_name=model_name,
                variants=args.variants,
            )
    finally:
        if ctx is not None and not args.output:
            pass  # table is printed below; artifacts drop with the temp dir

    print(format_table(study))

    if args.output:
        path = write_study(study, Path(args.output))
        print(f"\nwrote {path}")
    if ctx is not None:
        ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
