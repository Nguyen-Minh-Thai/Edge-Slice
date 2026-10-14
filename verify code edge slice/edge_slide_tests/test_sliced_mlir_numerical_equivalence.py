"""
Integration test: sliced_payload.mlir numerical equivalence vs PyTorch.

Flow (strict):
  [TinyConv PyTorch Model]
       ├─► Forward Pass ──► [pytorch_output]
       ▼ (torch-mlir + EdgeSlicePassManager)
  [sliced_payload.mlir]
       ▼ (llvmlite MCJIT via execute_sliced_mlir_jit)
  Feed Input Tensor ──► Invoke JIT ──► [mlir_output]
                              ▼
               assert torch.allclose(mlir_output, pytorch_output)

Requires: torch, torch_mlir, llvmlite, mlir-opt, mlir-translate, cc (for runner utils).
JIT runs in a subprocess so MLIR/Python failures are logged without killing pytest.
"""

from __future__ import annotations

import logging
import subprocess

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for numerical equivalence tests")

from tests.mlir_helpers import (
    assert_sliced_mlir_numerical_equivalence,
    require_mlir_jit_tools,
    require_torch_mlir,
)

logger = logging.getLogger(__name__)

INPUT_SHAPE = (1, 3, 32, 32)
L1_CACHE_KB = 32
# Measured, not habitual: the pipeline's worst max-absolute error across all
# five workloads at 64x64 and 96x96 is 1.53e-05 (resnet50 @64), so 1e-4 keeps
# at least 7x headroom on every cell. The previous 1e-3 left 66x to 2,000,000x
# of slack, in which a tenfold numerical regression would have passed. See
# tests/mlir_helpers.assert_sliced_mlir_numerical_equivalence for the table.
RTOL = 1e-4
ATOL = 1e-4
JIT_TIMEOUT_SEC = 180


@pytest.mark.integration
@pytest.mark.cpu
@pytest.mark.slow
def test_sliced_mlir_numerical_equivalence(tiny_conv_relu_model, deterministic_env):
    """
    End-to-end: PyTorch TinyConv vs sliced_payload.mlir JIT output.

    Verifies the EdgeSlide MLIR pipeline produces executable IR that matches
    the original PyTorch forward pass within rtol/atol tolerance.
    """
    require_torch_mlir()
    require_mlir_jit_tools()

    logging.basicConfig(level=logging.INFO)

    input_tensor = torch.randn(*INPUT_SHAPE, device="cpu")

    try:
        assert_sliced_mlir_numerical_equivalence(
            tiny_conv_relu_model,
            input_tensor,
            l1_cache_kb=L1_CACHE_KB,
            entry_func="forward",
            rtol=RTOL,
            atol=ATOL,
            timeout_sec=JIT_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"MLIR JIT execution timed out after {JIT_TIMEOUT_SEC}s")
    except Exception as exc:
        logger.exception("EdgeSlice MLIR e2e equivalence failed")
        pytest.fail(str(exc))