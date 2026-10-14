#!/usr/bin/env python3
"""
EdgeSlice CPU test runner.

Required tests (DAG, liveness, tiling, PyTorch demo bypass) always run.
Optional tests (torch_mlir parser, MLIR JIT) report [SKIPPED] when dependencies
are missing — they never count as [PASSED].

Usage:
    python tests/run_cpu_tests.py
    python tests/run_cpu_tests.py --verbose
"""
import argparse
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List

import numpy as np

# Ensure import paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import warnings

import torch

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Force CPU
torch.set_default_device("cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PRESSURE_DAG_L1_BYTES = 256 * 1024
PRESSURE_DAG_PEAK_OP_ID = 1
PRESSURE_DAG_PEAK_BYTES = 9 * 1024 * 1024
PRESSURE_DAG_CANDIDATE_IDS = (0, 1)
PRESSURE_DAG_CANDIDATE_TYPES = ("conv_big", "relu")
PRESSURE_DAG_EXCLUDED_OP_ID = 2


class TestStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SkipTest(Exception):
    """Raise from optional tests when required tools are unavailable."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class TestResult:
    name: str
    status: TestStatus
    duration_ms: float
    skip_reason: str = ""
    error: str = ""
    tb: str = ""

    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED

    @property
    def skipped(self) -> bool:
        return self.status == TestStatus.SKIPPED


def _seed_deterministic():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_pressure_dag():
    from edgeslide.dag.graph import OperationDAG, TensorInfo

    dag = OperationDAG()
    inp = TensorInfo((1, 3, 64, 64), "f32")
    big = TensorInfo((1, 128, 128, 128), "f32")
    medium = TensorInfo((1, 64, 64, 64), "f32")
    small = TensorInfo((1, 32, 32, 32), "f32")

    n0 = dag.add_node("conv_big", inputs=[inp], output=big)
    n1 = dag.add_node("relu", inputs=[big], output=medium)
    n2 = dag.add_node("add", inputs=[medium], output=small)
    dag.add_edge(n0, n1)
    dag.add_edge(n1, n2)
    return dag


def run_test(name: str, fn: Callable) -> TestResult:
    start = time.perf_counter()
    try:
        fn()
        dur = (time.perf_counter() - start) * 1000
        return TestResult(name, TestStatus.PASSED, dur)
    except SkipTest as exc:
        dur = (time.perf_counter() - start) * 1000
        return TestResult(
            name,
            TestStatus.SKIPPED,
            dur,
            skip_reason=str(exc.reason),
        )
    except Exception as e:
        dur = (time.perf_counter() - start) * 1000
        return TestResult(
            name,
            TestStatus.FAILED,
            dur,
            error=f"{type(e).__name__}: {e}",
            tb=traceback.format_exc(),
        )


def test_dag_basic():
    from edgeslide.dag.graph import OperationDAG, TensorInfo

    dag = OperationDAG()
    t0 = TensorInfo((1, 32, 64, 64))
    t1 = TensorInfo((1, 64, 32, 32))
    n0 = dag.add_node("conv", [], t0)
    n1 = dag.add_node("conv", [t0], t1)
    dag.add_edge(n0, n1)
    order = dag.topological_order()
    assert len(order) == 2 and order[0] < order[1]


def test_liveness_peak():
    from edgeslide.liveness.peak_memory import PeakMemoryEstimator

    dag = _build_pressure_dag()
    est = PeakMemoryEstimator(dag, l1_cache_bytes=PRESSURE_DAG_L1_BYTES)

    peak_op, peak_bytes = est.get_peak()
    assert peak_op == PRESSURE_DAG_PEAK_OP_ID
    assert peak_bytes == PRESSURE_DAG_PEAK_BYTES

    candidate_ids = [c["op_id"] for c in est.slicing_candidates()]
    candidate_types = [c["op_type"] for c in est.slicing_candidates()]
    assert candidate_ids == list(PRESSURE_DAG_CANDIDATE_IDS)
    assert candidate_types == list(PRESSURE_DAG_CANDIDATE_TYPES)
    assert PRESSURE_DAG_EXCLUDED_OP_ID not in candidate_ids
    assert not dag.nodes[PRESSURE_DAG_EXCLUDED_OP_ID].consumers


def test_dimension_and_tile():
    from edgeslide.dag.graph import OperationDAG, TensorInfo
    from edgeslide.tiling.dimension_selector import DimensionSelector
    from edgeslide.tiling.tile_size import TileSizeCalculator

    dag = OperationDAG()
    out = TensorInfo((1, 64, 256, 256))
    nid = dag.add_node("conv_2d", [], out)
    sel = DimensionSelector(dag)
    d = sel.select(nid)
    assert d in ("batch", "spatial_h", "spatial_w", "channel")
    calc = TileSizeCalculator(128 * 1024)
    ts = calc.compute((1, 64, 256, 256), "spatial_h")
    assert 1 <= ts <= 256


def test_standalone_slicing_correctness():
    import copy

    import torch.nn as nn

    from edgeslide.edgeslide_pytorch import EdgeSlideTransform, SlicingConfig

    _seed_deterministic()

    baseline = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
    ).eval().cpu()

    # batch > 1 → slice along batch (safe for padded Conv2d); spatial H/W
    # tiling without halo is rejected for padding > 0 in the demo slicer.
    x = torch.randn(4, 3, 256, 256, device="cpu")

    cfg = SlicingConfig(l1_cache_bytes=256 * 1024, enabled=True)
    model = copy.deepcopy(baseline)

    with torch.no_grad():
        y_ref = baseline(x)

    sliced_model = EdgeSlideTransform(cfg).transform_model(model)

    wrapped_layers = [
        m for m in sliced_model.modules() if m.__class__.__name__ == "SlicedConv2d"
    ]
    assert len(wrapped_layers) >= 1, (
        "Demo slicer did not wrap any Conv2d — check out_channels >= 64 threshold"
    )

    with torch.no_grad():
        y_sliced = sliced_model(x)

    assert any(layer.needs_slicing for layer in wrapped_layers), (
        "Wrapped Conv2d layers did not trigger tiling — increase input size or "
        "lower l1_cache_bytes"
    )
    assert y_ref.shape == y_sliced.shape
    assert torch.allclose(y_ref, y_sliced, rtol=5e-3, atol=5e-3)


def test_sliced_mlir_numerical_equivalence_if_available():
    """
    Full e2e: TinyConv PyTorch vs sliced_payload.mlir JIT (tests/test_sliced_mlir_numerical_equivalence.py).

    Optional — needs torch_mlir, mlir-opt, mlir-translate, and llvmlite.
    """
    try:
        import torch.nn as nn
        import torch_mlir  # noqa: F401

        from tests.mlir_helpers import (
            assert_sliced_mlir_numerical_equivalence,
            mlir_jit_missing_tools,
        )
    except ImportError as exc:
        raise SkipTest(f"import failed: {exc}") from exc

    missing = mlir_jit_missing_tools()
    if missing:
        raise SkipTest(f"MLIR JIT stack not available: {', '.join(missing)}")

    _seed_deterministic()

    class TinyConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    model = TinyConv().eval().cpu()
    x = torch.randn(1, 3, 32, 32, device="cpu")
    assert_sliced_mlir_numerical_equivalence(
        model,
        x,
        l1_cache_kb=32,
        entry_func="forward",
        rtol=1e-3,
        atol=1e-3,
        timeout_sec=180,
    )


def test_mlir_numerical_equivalence_if_available():
    """MLIR pipeline end-to-end JIT equivalence (optional — needs torch_mlir + mlir tools)."""
    try:
        import torch.nn as nn
        import torch_mlir  # noqa: F401

        from edgeslide.mlir_transform.lowering import find_mlir_tool
        from tests.mlir_helpers import compile_sliced_payload, run_jit_in_subprocess
    except ImportError as exc:
        raise SkipTest(f"import failed: {exc}") from exc

    missing = [
        name
        for name in ("mlir-opt", "mlir-translate")
        if find_mlir_tool(name) is None
    ]
    if missing:
        raise SkipTest(f"MLIR JIT tools not found: {', '.join(missing)}")

    import tempfile
    from pathlib import Path

    _seed_deterministic()

    class TinyConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    model = TinyConv().eval().cpu()
    x = torch.randn(1, 3, 32, 32, device="cpu")

    with torch.no_grad():
        y_ref = model(x)

    with tempfile.TemporaryDirectory(prefix="edgeslide_mlir_cpu_") as tmp:
        output_dir = Path(tmp)
        artifacts = compile_sliced_payload(
            model, x, output_dir, l1_cache_kb=32, entry_func="forward"
        )
        sliced_path = Path(artifacts["sliced_payload_mlir"])
        assert sliced_path.is_file()

        y_mlir = run_jit_in_subprocess(
            sliced_path, output_dir, x, entry_func="forward", timeout_sec=180
        )

    assert y_mlir.shape == y_ref.shape
    assert torch.allclose(y_mlir, y_ref, rtol=5e-3, atol=5e-3)


def test_parser_and_rewrite_if_available():
    """EdgeSlicePassManager frontend test (optional — needs torch_mlir)."""
    try:
        import torch.nn as nn
        import torch_mlir  # noqa: F401

        from edgeslide.dag.parser import DAGParser
        from edgeslide.transform.pass_manager import EdgeSlicePassManager
    except ImportError as exc:
        raise SkipTest(f"torch_mlir not available: {exc}") from exc

    _seed_deterministic()

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, x):
            return self.c1(x)

    model = Tiny().eval().cpu()
    x = torch.randn(1, 3, 32, 32, device="cpu")
    dag = DAGParser(model, x).parse()
    assert len(dag.nodes) > 0

    results = EdgeSlicePassManager(dag, l1_cache_bytes=16 * 1024).run()
    assert "transform_ir" in results
    transform_ir = results["transform_ir"]
    assert "transform" in transform_ir or "No slicing" in transform_ir


def _status_label(result: TestResult) -> str:
    if result.status == TestStatus.SKIPPED:
        return "[SKIPPED]"
    if result.status == TestStatus.PASSED:
        return "[PASSED]"
    return "[FAILED]"


def main():
    parser = argparse.ArgumentParser(description="EdgeSlide CPU test runner")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full tracebacks on test failures",
    )
    args = parser.parse_args()

    tests: List[tuple] = [
        ("DAG basic", test_dag_basic),
        ("Liveness + PeakMemory", test_liveness_peak),
        ("DimensionSelector + TileSize", test_dimension_and_tile),
        ("Standalone slicing numerical correctness (PyTorch demo bypass)", test_standalone_slicing_correctness),
        ("Parser + PassManager (optional)", test_parser_and_rewrite_if_available),
        (
            "Sliced MLIR e2e numerical equivalence (optional)",
            test_sliced_mlir_numerical_equivalence_if_available,
        ),
        ("MLIR JIT numerical equivalence (optional)", test_mlir_numerical_equivalence_if_available),
    ]

    results: List[TestResult] = []
    print("=" * 60)
    print("EdgeSlide CPU Test Suite")
    print("Required tests always run; optional tests skip when tools are missing.")
    print("All tests forced to device='cpu'. No GPU used.")
    print("=" * 60)

    for name, fn in tests:
        print(f"▶ {name} ... ", end="", flush=True)
        res = run_test(name, fn)
        print(f"{_status_label(res)}  ({res.duration_ms:.1f} ms)")
        if res.status == TestStatus.SKIPPED:
            print(f"   Reason: {res.skip_reason}")
        elif res.status == TestStatus.FAILED:
            print(f"   Error: {res.error}")
            if args.verbose and res.tb:
                print(res.tb)
        results.append(res)

    print("-" * 60)
    passed = sum(1 for r in results if r.status == TestStatus.PASSED)
    skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
    failed = sum(1 for r in results if r.status == TestStatus.FAILED)
    total = len(results)
    print(
        f"Result: {passed} passed, {skipped} skipped, {failed} failed "
        f"(total {total})"
    )

    if failed > 0:
        print("\nSome tests failed. See errors above.")
        sys.exit(1)

    if skipped > 0 and passed > 0:
        print("ok (required tests passed; some optional tests skipped)")
    elif skipped > 0:
        print("ok (all runnable tests passed; remaining tests skipped)")
    else:
        print("ok")


if __name__ == "__main__":
    main()