"""Shared helpers for MLIR integration tests."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

# Helper module imported by test modules: skipping here skips every
# importer, with a reason, instead of aborting collection.
torch = pytest.importorskip("torch", reason="PyTorch is required for MLIR integration helpers")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch_mlir  # noqa: F401

    HAS_TORCH_MLIR = True
except ImportError:
    HAS_TORCH_MLIR = False


def require_torch_mlir():
    if not HAS_TORCH_MLIR:
        pytest.skip("torch_mlir not available")


def require_mlir_execution_tools():
    """Skip unless mlir-opt and mlir-cpu-runner are available."""
    from edgeslide.mlir_transform.lowering import find_mlir_tool

    missing = [
        name
        for name in ("mlir-opt", "mlir-cpu-runner")
        if find_mlir_tool(name) is None
    ]
    if missing:
        pytest.skip(f"MLIR execution tools not found: {', '.join(missing)}")


def require_mlir_jit_tools():
    """Skip unless mlir-opt and mlir-translate are available for JIT lowering."""
    missing = mlir_jit_missing_tools()
    if missing:
        pytest.skip(f"MLIR JIT tools not found: {', '.join(missing)}")


def mlir_jit_missing_tools() -> list[str]:
    """Return missing tool names for MLIR JIT (mlir-opt, mlir-translate, llvmlite)."""
    from edgeslide.mlir_transform.lowering import find_mlir_tool

    missing = [
        name
        for name in ("mlir-opt", "mlir-translate")
        if find_mlir_tool(name) is None
    ]
    try:
        import llvmlite  # noqa: F401
    except ImportError:
        missing.append("llvmlite")
    return missing


def assert_sliced_mlir_numerical_equivalence(
    model,
    input_tensor: torch.Tensor,
    *,
    l1_cache_kb: int = 32,
    entry_func: str = "forward",
    rtol: float = 1e-4,
    atol: float = 1e-4,
    timeout_sec: int = 180,
) -> None:
    """
    Shared e2e check: PyTorch forward vs sliced_payload.mlir JIT output.

    Raises AssertionError on numerical mismatch; callers should catch
    ImportError and missing-tool cases for skip semantics.

    **Tolerance is 1e-4, chosen from measurement rather than habit.** It was
    1e-3, which the pipeline clears by 66x to 2,000,000x — so a change that
    degraded numerics tenfold would still have passed. Measured max absolute
    error across all five workloads at 64x64 and 96x96:

        resnet50         1.5e-05 / 6.2e-06      mobilenet_v2  1.4e-05 / 1.2e-05
        resnet18         7.4e-06 / 5.2e-06      efficientnet  4.9e-06 / 5.1e-06
        unet             4.4e-10 / 1.5e-09

    The worst cell is 1.53e-05, so 1e-4 keeps at least 7x headroom on every
    one of the ten — enough for platform variation, and ten times tighter than
    before.

    The achieved margin is printed on success as well as failure. A tolerance
    only catches a regression that crosses it; a recorded margin makes a
    regression that stays inside it visible too, which is the case that would
    otherwise accumulate silently.
    """
    with torch.no_grad():
        pytorch_output = model(input_tensor)

    with tempfile.TemporaryDirectory(prefix="edgeslide_equiv_") as tmp:
        output_dir = Path(tmp)
        artifacts = compile_sliced_payload(
            model,
            input_tensor,
            output_dir,
            l1_cache_kb=l1_cache_kb,
            entry_func=entry_func,
        )
        sliced_path = Path(artifacts["sliced_payload_mlir"])
        if not sliced_path.is_file():
            raise AssertionError(f"Missing sliced_payload.mlir at {sliced_path}")

        mlir_output = run_jit_in_subprocess(
            sliced_path,
            output_dir,
            input_tensor,
            entry_func=entry_func,
            timeout_sec=timeout_sec,
        )

    if mlir_output.shape != pytorch_output.shape:
        raise AssertionError(
            f"Shape mismatch: mlir={tuple(mlir_output.shape)} "
            f"pytorch={tuple(pytorch_output.shape)}"
        )

    difference = (mlir_output - pytorch_output).abs()
    max_diff = difference.max().item()
    mean_diff = difference.mean().item()
    if not torch.allclose(mlir_output, pytorch_output, rtol=rtol, atol=atol):
        raise AssertionError(
            f"MLIR output diverges from PyTorch "
            f"(max_diff={max_diff:.3e}, mean_diff={mean_diff:.3e}, "
            f"rtol={rtol}, atol={atol})"
        )
    # Recorded on success too: a tolerance only catches a regression that
    # crosses it, and one that stays inside would otherwise accumulate unseen.
    headroom = atol / max_diff if max_diff > 0 else float("inf")
    logger.info(
        "numerical equivalence OK: max_diff=%.3e mean_diff=%.3e "
        "(%.0fx headroom under atol=%g)",
        max_diff,
        mean_diff,
        headroom,
        atol,
    )


def compile_sliced_payload(
    model,
    input_tensor: torch.Tensor,
    output_dir: Path,
    *,
    l1_cache_kb: int = 32,
    entry_func: str = "forward",
) -> dict:
    from edgeslide.mlir_transform.pipeline import EdgeSlideMLIRConfig, EdgeSlideMLIRPipeline

    config = EdgeSlideMLIRConfig(
        l1_cache_kb=l1_cache_kb,
        output_dir=str(output_dir),
    )
    pipeline = EdgeSlideMLIRPipeline(config)
    return pipeline.compile(
        model,
        tuple(input_tensor.shape),
        model_name="tiny_conv_relu",
        entry_func=entry_func,
    )


def run_jit_in_subprocess(
    sliced_payload_path: Path,
    output_dir: Path,
    input_tensor: torch.Tensor,
    entry_func: str = "forward",
    timeout_sec: int = 180,
) -> torch.Tensor:
    """Execute sliced MLIR via JIT in an isolated subprocess."""
    logger = logging.getLogger(__name__)

    with tempfile.TemporaryDirectory(prefix="edgeslide_jit_") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.pt"
        result_path = tmp_path / "mlir_output.pt"
        log_path = tmp_path / "jit.log"

        torch.save(input_tensor.cpu(), input_path)

        worker = f"""
import logging
import sys
import traceback

import torch

sys.path.insert(0, {str(SRC)!r})

logging.basicConfig(
    filename={str(log_path)!r},
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)

def main():
    from edgeslide.mlir_transform.lowering import execute_sliced_mlir_jit

    x = torch.load({str(input_path)!r}, weights_only=True)
    y = execute_sliced_mlir_jit(
        {str(sliced_payload_path)!r},
        x,
        entry_func={entry_func!r},
        output_dir={str(output_dir)!r},
    )
    torch.save(y.cpu(), {str(result_path)!r})

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("MLIR JIT execution failed")
        traceback.print_exc()
        raise
"""
        worker_path = tmp_path / "jit_worker.py"
        worker_path.write_text(worker)

        proc = subprocess.run(
            [sys.executable, str(worker_path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(ROOT),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )

        log_text = log_path.read_text() if log_path.is_file() else ""
        if proc.returncode != 0:
            logger.error("MLIR JIT subprocess failed (exit %s)", proc.returncode)
            logger.error("stdout:\n%s", proc.stdout)
            logger.error("stderr:\n%s", proc.stderr)
            logger.error("jit.log:\n%s", log_text)
            pytest.fail(
                "MLIR JIT execution failed — see captured logs above "
                f"(exit code {proc.returncode})"
            )

        if not result_path.is_file():
            logger.error("jit.log:\n%s", log_text)
            pytest.fail("MLIR JIT subprocess exited 0 but mlir_output.pt was not written")

        return torch.load(result_path, weights_only=True)