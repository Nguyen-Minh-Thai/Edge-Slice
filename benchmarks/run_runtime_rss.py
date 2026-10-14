#!/usr/bin/env python3
"""Measure runtime peak RSS in isolated child processes.

This is an end-to-end PyTorch reference measurement. It is intentionally
separate from the MLIR static liveness evaluator and labels that limitation.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _worker(model_name: str, height: int, width: int, method: str, image_dir: str | None, out) -> None:
    import torch

    from benchmarks.workloads import load_model, make_example_input
    from edgeslide.edgeslide_pytorch import EdgeSlideTransform, SlicingConfig

    torch.set_num_threads(1)
    model = load_model(model_name).eval().cpu()
    if method == "edgeslide_reference":
        model = EdgeSlideTransform(SlicingConfig(l1_cache_bytes=256 * 1024)).transform_model(model).eval()
    x, meta = make_example_input(
        model_name, height, width, use_real_images=image_dir is not None,
        image_dir=image_dir, return_meta=True,
    )
    with torch.inference_mode():
        model(x)  # warmup
        model(x)
    out.send({
        "method": method,
        "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "input_meta": meta,
        "weights_source": getattr(model, "_edgeslide_weights_source", "unknown"),
    })
    out.close()


def measure(model: str, h: int, w: int, method: str, image_dir: str | None) -> dict:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(False)
    proc = ctx.Process(target=_worker, args=(model, h, w, method, image_dir, child))
    proc.start()
    child.close()
    result = parent.recv() if parent.poll(300) else {"error": "worker timeout"}
    proc.join(30)
    if proc.exitcode != 0:
        result["error"] = f"worker exited with code {proc.exitcode}"
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Isolated runtime RSS benchmark")
    p.add_argument("--model", choices=("resnet50", "mobilenet_v2", "efficientnet_b0"), default="resnet50")
    p.add_argument("--resolution", default="512x512")
    p.add_argument("--image-dir", default=None)
    p.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/runtime_rss.json")
    args = p.parse_args()
    h, w = (int(x) for x in args.resolution.lower().split("x", 1))
    rows = [measure(args.model, h, w, method, args.image_dir)
            for method in ("baseline", "edgeslide_reference")]
    valid = [r for r in rows if r.get("peak_rss_kb")]
    if len(valid) == 2:
        base, sliced = (r["peak_rss_kb"] for r in valid)
        # `base` is an OS high-water mark, which is never legitimately zero for
        # a process that ran — so a zero means the measurement did not happen,
        # and reporting a 0.0% reduction for it would put a real-looking number
        # where there is none.
        reduction = round(100 * (1 - sliced / base), 2)
    else:
        reduction = None
    report = {
        "measurement_type": "isolated_process_runtime_peak_rss",
        "compiler_evidence": False,
        "model": args.model, "resolution": [h, w], "results": rows,
        "peak_rss_reduction_pct": reduction,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
