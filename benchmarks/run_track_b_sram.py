#!/usr/bin/env python3
"""Compile and verify Track-B explicit-SRAM artifacts.

This runner is intentionally a placement benchmark, not a CPU timing proxy.
It records schedule-search evidence, actual non-default-memory-space
allocations, copy operations, and the verified SRAM high-water mark.  Native
latency remains unavailable until a target backend maps the memory space and
copies to the device SRAM/DMA implementation.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edgeslide.mlir_transform.pipeline import (  # noqa: E402
    EdgeSlideMLIRConfig,
    EdgeSlideMLIRPipeline,
)


class TinyConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(value))


class ConvStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


CASES = {
    "tiny_conv": (TinyConv, (1, 3, 32, 32)),
    "conv_stack": (ConvStack, (1, 3, 32, 32)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--sram-kb", type=int, default=32)
    parser.add_argument("--memory-space", type=int, default=1)
    parser.add_argument("--no-double-buffer", action="store_true")
    parser.add_argument(
        "--work-dir", type=Path, default=ROOT / "build" / "track_b_sram"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "track_b_sram.json",
    )
    args = parser.parse_args()
    if args.sram_kb <= 0 or args.memory_space == 0:
        raise SystemExit("--sram-kb must be positive and --memory-space non-zero")

    torch.manual_seed(0)
    rows = []
    for name in args.models:
        model_type, shape = CASES[name]
        pipeline = EdgeSlideMLIRPipeline(
            EdgeSlideMLIRConfig(
                output_dir=str(args.work_dir),
                sram_capacity_kb=args.sram_kb,
                sram_double_buffer=not args.no_double_buffer,
                sram_memory_space=args.memory_space,
            )
        )
        started = time.perf_counter()
        result = pipeline.compile(
            model_type().eval(), shape, model_name=name, entry_func="forward"
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        plans = result.get("planning", {}).get("rewrite_plan", [])
        rows.append(
            {
                "model": name,
                "input_shape": shape,
                "compile_wall_ms": round(elapsed_ms, 3),
                "search": {
                    "strategy": "search",
                    "planned_ops": len(plans),
                    "candidates_evaluated": sum(
                        int(plan.get("schedule_candidates_evaluated", 0))
                        for plan in plans
                    ),
                    "measured_on_target": any(
                        bool(plan.get("schedule_measured")) for plan in plans
                    ),
                    "plans": plans,
                },
                "placement": result["sram_placement"],
                "artifacts": {
                    "sram_payload_mlir": result["sram_payload_mlir"],
                    "sram_promotion_mlir": result["sram_promotion_mlir"],
                },
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "track": "B-explicit-sram-placement",
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "configuration": {
            "sram_capacity_bytes": args.sram_kb * 1024,
            "memory_space": args.memory_space,
            "double_buffer_capacity_reserved": not args.no_double_buffer,
        },
        "execution": {
            "status": "unavailable_no_target_backend",
            "latency_is_reported": False,
            "reason": (
                "The generic CPU backend cannot map non-zero MLIR memory spaces "
                "or lower scratchpad copies to the target DMA engine."
            ),
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
