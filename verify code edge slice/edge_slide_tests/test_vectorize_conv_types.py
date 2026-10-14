"""The outlined helper's memref types must parse, or the feature does nothing.

`enable_stride1_vectorization` outlines stride-1 convolutions into helper
functions the legacy affine vectorizer can handle. It matched ops correctly
(46 of 53 on ResNet-50) and still produced zero vector instructions on every
model, because `_memref_shape_and_dtype`'s character class was `[^,]+` — greedy
over the closing bracket. Every type it built came out with a doubled bracket
(`memref<1x8x34x34xf32>>`) and every scalar op with a broken dtype
(`arith.mulf %a, %b : f32>`); mlir-opt rejected the module, and the pipeline's
fallback caught that and logged it as though the payload were merely
unsupported.

The lesson these tests encode: an opcount check on the *final* LLVM dialect
cannot detect this, because by then `--convert-vector-to-llvm` has consumed
every `vector.*` op. Assert on the emitted types, and on SIMD types in the
LLVM IR.
"""

from __future__ import annotations

import re

import pytest

from edgeslide.mlir_transform.vectorize_conv import (
    _identity_memref,
    _memref_shape_and_dtype,
    outline_stride1_convs,
)

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize(
    "memref_type,dims,dtype",
    [
        ("memref<1x8x34x34xf32>", [1, 8, 34, 34], "f32"),
        ("memref<16x8x3x3xf32>", [16, 8, 3, 3], "f32"),
        ("memref<1x16x32x32xf16>", [1, 16, 32, 32], "f16"),
        # A layout map must be ignored, not parsed into the dtype.
        (
            "memref<1x16x32x32xf32, strided<[16384, 1024, 32, 1], offset: 0>>",
            [1, 16, 32, 32],
            "f32",
        ),
    ],
)
def test_the_dtype_does_not_swallow_the_closing_bracket(memref_type, dims, dtype):
    assert _memref_shape_and_dtype(memref_type) == (dims, dtype)


def test_a_dynamic_extent_is_rejected_rather_than_guessed():
    assert _memref_shape_and_dtype("memref<?x8xf32>") is None


def test_a_rebuilt_type_round_trips():
    """The exact identity that was broken: parse then rebuild must be a no-op."""
    original = "memref<1x8x34x34xf32>"
    dims, dtype = _memref_shape_and_dtype(original)
    assert _identity_memref(dims, dtype) == original


CONV_IR = """module {
  func.func @forward(%in: memref<1x8x34x34xf32>, %filt: memref<16x8x3x3xf32>,
                     %out: memref<1x16x32x32xf32>) {
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, \
strides = dense<1> : tensor<2xi64>} ins(%in, %filt : memref<1x8x34x34xf32>, \
memref<16x8x3x3xf32>) outs(%out : memref<1x16x32x32xf32>)
    return
  }
}
"""


def test_the_outlined_helper_has_no_doubled_brackets():
    outlined, count = outline_stride1_convs(CONV_IR)
    assert count == 1
    helper = outlined[outlined.index("__edgeslide_vec_conv") :]
    assert "f32>>" not in helper, "doubled bracket in a generated memref type"
    assert ": f32>" not in helper, "broken dtype in a generated scalar op"


def test_the_outlined_helper_parses(tmp_path):
    """The assertion that would have caught the bug on day one."""
    from tests.mlir_helpers import require_mlir_jit_tools

    require_mlir_jit_tools()
    from edgeslide.mlir_transform.errors import require_tool, run_tool

    outlined, count = outline_stride1_convs(CONV_IR)
    assert count == 1
    source = tmp_path / "outlined.mlir"
    source.write_text(outlined)
    # mlir-opt with no passes is a parse-and-print round trip.
    run_tool(
        [require_tool("mlir-opt"), str(source), "-o", str(tmp_path / "out.mlir")],
        stage="parse outlined helper",
    )


@pytest.mark.integration
@pytest.mark.slow
def test_the_vectorized_pipeline_emits_simd_types(tmp_path):
    """End to end: SIMD types appear in the LLVM IR, and only when asked for.

    Counted as `<N x float>` in the *LLVM IR*, which is the only place the
    evidence survives: the MLIR `vector.*` ops are gone by the end of stage 2.
    """
    from tests.mlir_helpers import require_mlir_jit_tools, require_torch_mlir

    require_torch_mlir()
    require_mlir_jit_tools()
    import torch

    from edgeslide.mlir_transform.lowering import (
        SlicedPayloadLowering,
        SlicedPayloadLoweringConfig,
    )
    from edgeslide.mlir_transform.pipeline import (
        EdgeSlideMLIRConfig,
        EdgeSlideMLIRPipeline,
    )

    model = torch.nn.Conv2d(8, 16, 3, padding=1, bias=False).eval()
    pipeline = EdgeSlideMLIRPipeline(
        EdgeSlideMLIRConfig(l1_cache_kb=32, output_dir=str(tmp_path))
    )
    with torch.no_grad():
        parser, _ = pipeline.parse_model(model, (1, 8, 32, 32))
    payload = parser.save_mlir(str(tmp_path / "payload.mlir"))

    counts = {}
    for vectorize in (False, True):
        lowering = SlicedPayloadLowering(
            SlicedPayloadLoweringConfig(
                output_dir=str(tmp_path / f"low_{vectorize}"),
                entry_func="forward",
                enable_stride1_vectorization=vectorize,
                vector_width=8,
            )
        )
        ir = (tmp_path / f"low_{vectorize}" / "sliced_payload.ll")
        lowering.lower(payload)
        counts[vectorize] = len(re.findall(r"<\d+ x float>", ir.read_text()))

    assert counts[False] == 0, "the scalar pipeline must emit no SIMD types"
    assert counts[True] > 0, (
        "the vectorized pipeline emitted no SIMD types; it has fallen back to "
        "scalar (check the log for a PARSE error from the outlined helper)"
    )
