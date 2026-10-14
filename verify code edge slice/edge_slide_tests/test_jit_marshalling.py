"""Tests for the JIT engine's memref marshalling.

``SlicedPayloadJitEngine`` is how compiled payloads are actually called, and a
bug here does not crash — it silently returns wrong numbers, which is the worst
failure mode this project has. The existing unit test mocks
``_tensor_to_memref`` and the ciface lookup, i.e. both methods that do the work,
so nothing pinned the marshalling itself.

These tests substitute only the *callee*: a real ``ctypes`` function pointer
built from a Python callback. Argument marshalling, pointer ordering, the
result descriptor, and the conversion back all run for real.
"""

from __future__ import annotations

import ctypes

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for JIT marshalling")
np = pytest.importorskip("numpy")
pytest.importorskip("torch_mlir.runtime", reason="torch-mlir runtime is required")

from torch_mlir.runtime import (  # noqa: E402
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
)

from edgeslide.mlir_transform.lowering import SlicedPayloadJitEngine  # noqa: E402

pytestmark = pytest.mark.cpu


def _bare_engine() -> SlicedPayloadJitEngine:
    """An engine with no JIT behind it — only the marshalling paths."""
    engine = object.__new__(SlicedPayloadJitEngine)
    engine._result_holder = None
    engine._func_cache = {}
    return engine


def _descriptor_at(pointer, rank: int, ctype=ctypes.c_float):
    desc_type = make_nd_memref_descriptor(rank, ctype)
    return ctypes.cast(pointer, ctypes.POINTER(desc_type)).contents


def _install_callee(engine, entry: str, num_args: int, body):
    functype = SlicedPayloadJitEngine._ciface_func_type(num_args)
    # Keep the ctypes callback alive for the engine's lifetime: a garbage
    # collected trampoline would leave a dangling function pointer.
    callback = functype(body)
    engine._func_cache[(f"_mlir_ciface_{entry}", num_args)] = callback
    return callback


def test_tensor_to_memref_describes_the_tensor_it_was_given():
    engine = _bare_engine()
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    desc = engine._tensor_to_memref(tensor)

    assert [int(desc.shape[i]) for i in range(3)] == [2, 3, 4]
    assert [int(desc.strides[i]) for i in range(3)] == [12, 4, 1]
    # The descriptor must point at data equal to the tensor's, element for
    # element — a right-shaped descriptor over the wrong buffer is the exact
    # bug that returns plausible-looking wrong numbers.
    data = ctypes.cast(desc.aligned, ctypes.POINTER(ctypes.c_float))
    assert [data[i] for i in range(24)] == list(range(24))


def test_a_non_contiguous_tensor_is_made_contiguous_before_marshalling():
    """A transposed view has non-monotonic strides; memref callees assume dense."""
    engine = _bare_engine()
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    transposed = source.t()
    assert not transposed.is_contiguous()

    desc = engine._tensor_to_memref(transposed)

    assert [int(desc.shape[i]) for i in range(2)] == [4, 3]
    assert [int(desc.strides[i]) for i in range(2)] == [3, 1]
    data = ctypes.cast(desc.aligned, ctypes.POINTER(ctypes.c_float))
    assert [data[i] for i in range(12)] == [
        float(v) for v in transposed.contiguous().flatten()
    ]


def test_a_non_contiguous_numpy_array_is_made_contiguous():
    engine = _bare_engine()
    array = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    assert not array.flags["C_CONTIGUOUS"]

    desc = engine._tensor_to_memref(array)

    assert [int(desc.shape[i]) for i in range(2)] == [3, 2]
    data = ctypes.cast(desc.aligned, ctypes.POINTER(ctypes.c_float))
    assert [data[i] for i in range(6)] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def test_invoke_passes_inputs_in_order_with_their_own_shapes():
    """Ordering is what a mocked marshaller cannot check.

    The two inputs have different ranks and shapes on purpose: a swapped or
    duplicated argument changes what the callee reads, and nothing downstream
    would notice.
    """
    engine = _bare_engine()
    first = torch.ones(2, 3, dtype=torch.float32)
    second = torch.ones(4, 5, 6, dtype=torch.float32) * 2
    seen: list[tuple] = []
    keepalive: list = []

    def callee(result_ptr, arg0, arg1):
        seen.append(
            (
                [int(_descriptor_at(arg0, 2).shape[i]) for i in range(2)],
                [int(_descriptor_at(arg1, 3).shape[i]) for i in range(3)],
            )
        )
        produced = np.zeros((2, 3), dtype=np.float32)
        keepalive.append(produced)
        desc = get_ranked_memref_descriptor(produced)
        keepalive.append(desc)
        ctypes.memmove(result_ptr, ctypes.byref(desc), ctypes.sizeof(desc))

    _install_callee(engine, "forward", 2, callee)
    engine.invoke("forward", first, second, result_rank=2)

    assert seen == [([2, 3], [4, 5, 6])]


def test_invoke_returns_the_buffer_the_callee_wrote():
    engine = _bare_engine()
    expected = np.arange(6, dtype=np.float32).reshape(2, 3) * 1.5
    keepalive: list = []

    def callee(result_ptr, _arg0):
        produced = np.ascontiguousarray(expected)
        keepalive.append(produced)
        desc = get_ranked_memref_descriptor(produced)
        keepalive.append(desc)
        ctypes.memmove(result_ptr, ctypes.byref(desc), ctypes.sizeof(desc))

    _install_callee(engine, "forward", 1, callee)
    result = engine.invoke(
        "forward", torch.zeros(2, 3, dtype=torch.float32), result_rank=2
    )

    np.testing.assert_allclose(np.asarray(result), expected)


def test_invoke_rejects_an_unsupported_result_dtype():
    engine = _bare_engine()
    with pytest.raises(TypeError, match="Unsupported MLIR result dtype"):
        engine.invoke(
            "forward", torch.zeros(2, 2), result_rank=2, result_dtype="bf16"
        )


def test_invoke_requires_at_least_one_input():
    engine = _bare_engine()
    with pytest.raises(ValueError, match="at least one input tensor"):
        engine.invoke("forward")


def test_invoke_strips_a_leading_symbol_sigil_from_the_entry_name():
    """Callers pass MLIR symbol references like ``@forward``."""
    engine = _bare_engine()
    keepalive: list = []

    def callee(result_ptr, _arg0):
        produced = np.zeros((1,), dtype=np.float32)
        keepalive.append(produced)
        desc = get_ranked_memref_descriptor(produced)
        keepalive.append(desc)
        ctypes.memmove(result_ptr, ctypes.byref(desc), ctypes.sizeof(desc))

    _install_callee(engine, "forward", 1, callee)
    engine.invoke("@forward", torch.zeros(1, dtype=torch.float32), result_rank=1)


def test_memref_to_torch_copies_a_strided_result():
    """The fallback path when the result layout is not the destination's.

    Exercised whenever ``ranked_memref_to_numpy`` refuses a layout, which is
    precisely when the numbers would otherwise be read in the wrong order.
    """
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    transposed = source.t().contiguous()
    desc = get_ranked_memref_descriptor(transposed.numpy())

    out = SlicedPayloadJitEngine._memref_to_torch(desc, dtype=torch.float32)

    assert list(out.shape) == [4, 3]
    torch.testing.assert_close(out, transposed)
