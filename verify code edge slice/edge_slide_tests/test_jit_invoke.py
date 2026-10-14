"""Unit tests for SlicedPayloadJitEngine.invoke() C-interface calling convention."""

from __future__ import annotations

import ctypes
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

np = pytest.importorskip("numpy", reason="numpy is required for JIT invoke tests")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edgeslide.mlir_transform.lowering import SlicedPayloadJitEngine


def test_ciface_func_type_arity():
    single_arg = SlicedPayloadJitEngine._ciface_func_type(1)
    assert single_arg._argtypes_ == (ctypes.c_void_p, ctypes.c_void_p)

    two_arg = SlicedPayloadJitEngine._ciface_func_type(2)
    assert two_arg._argtypes_ == (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


def test_invoke_passes_all_tensor_arguments():
    """Multi-arg _mlir_ciface_* must receive result + every input memref pointer."""
    engine = object.__new__(SlicedPayloadJitEngine)
    engine._result_holder = None
    engine._func_cache = {}

    captured_args = []

    def fake_ciface(*args):
        captured_args.extend(args)

    holder_type = type(
        "MemrefHolder",
        (ctypes.Structure,),
        {"_fields_": [("aligned", ctypes.c_void_p)]},
    )

    engine._get_ciface_func = MagicMock(return_value=fake_ciface)
    engine._tensor_to_memref = MagicMock(side_effect=lambda t: holder_type())

    input_a = np.ones((2, 2), dtype=np.float32)
    input_b = np.ones((3, 3), dtype=np.float32)

    runtime = types.ModuleType("torch_mlir.runtime")
    runtime.make_nd_memref_descriptor = lambda rank, ctype: holder_type
    runtime.ranked_memref_to_numpy = lambda holder: np.zeros((2, 2), dtype=np.float32)
    torch_mlir = types.ModuleType("torch_mlir")
    torch_mlir.runtime = runtime

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "torch_mlir", torch_mlir)
        mp.setitem(sys.modules, "torch_mlir.runtime", runtime)
        engine.invoke("conv2d", input_a, input_b)

    engine._get_ciface_func.assert_called_once_with("_mlir_ciface_conv2d", 2)
    assert len(captured_args) == 3
    assert captured_args[0] is not None
    assert captured_args[1] is not None
    assert captured_args[2] is not None
