"""Tests for the stride-1 conv outlining transform (no MLIR toolchain needed).

The critical property under test is the stride guard: outlining a stride>1
conv into this vectorization path was verified (via JIT execution against a
NumPy reference, see edgeslide_vectorize_conv module docstring) to silently
corrupt results, so `outline_stride1_convs` must never match one.
"""

import pytest

from edgeslide.mlir_transform.vectorize_conv import (
    _split_top_level,
    outline_stride1_convs,
)

pytestmark = pytest.mark.cpu

# Real bufferized, tiled payload text (post one-shot-bufferize +
# buffer-deallocation-pipeline + normalize-memrefs + scf-forall-to-for) for
# the tiny_conv workload — the exact shape this transform was verified
# end-to-end against.
STRIDE1_PAYLOAD = """\
module {
  func.func @forward(%arg0: memref<1x3x256x256xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x8x256x256xf32> attributes {llvm.emit_c_interface} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x3x258x258xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x8x256x256xf32>
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    scf.for %arg1 = %c0 to %c64 step %c1 {
      %2 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg1)
      %subview_2 = memref.subview %alloc[0, 0, %2, 0] [1, 3, 6, 258] [1, 1, 1, 1] : memref<1x3x258x258xf32> to memref<1x3x6x258xf32, strided<[199692, 66564, 258, 1], offset: ?>>
      %subview_3 = memref.subview %alloc_1[0, 0, %2, 0] [1, 8, 4, 256] [1, 1, 1, 1] : memref<1x8x256x256xf32> to memref<1x8x4x256xf32, strided<[524288, 65536, 256, 1], offset: ?>>
      linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, edgeslice.memory_tier = "sram", edgeslice.op_id = "op_8", edgeslice.working_set_bytes = 117824 : i64, strides = dense<1> : vector<2xi64>} ins(%subview_2, %0 : memref<1x3x6x258xf32, strided<[199692, 66564, 258, 1], offset: ?>>, memref<8x3x3x3xf32>) outs(%subview_3 : memref<1x8x4x256xf32, strided<[524288, 65536, 256, 1], offset: ?>>)
    }
    return %alloc_1 : memref<1x8x256x256xf32>
  }
}
"""

STRIDE2_PAYLOAD = STRIDE1_PAYLOAD.replace(
    'strides = dense<1> : vector<2xi64>', 'strides = dense<2> : vector<2xi64>'
)


def test_split_top_level_respects_nested_brackets():
    text = "%a, %b : memref<1x2xf32, strided<[1, 2], offset: ?>>, memref<8xf32>"
    names_part, _, types_part = text.partition(":")

    assert _split_top_level(names_part, sep=",") == ["%a", "%b"]
    assert _split_top_level(types_part, sep=",") == [
        "memref<1x2xf32, strided<[1, 2], offset: ?>>",
        "memref<8xf32>",
    ]


def test_outlines_stride1_conv():
    new_text, count = outline_stride1_convs(STRIDE1_PAYLOAD)

    assert count == 1
    assert "func.func private @__edgeslide_vec_conv_op_8_0" in new_text
    assert "func.call @__edgeslide_vec_conv_op_8_0" in new_text
    # The original inline conv op must be gone, replaced by the call.
    assert "linalg.conv_2d_nchw_fchw" not in new_text
    # Helper does its own local-alloc + copy-in (input, filter, output-init)
    # + copy-out (result) around the op: 4 memref.copy total.
    assert new_text.count("memref.copy") == 4
    assert "affine.for" in new_text


def test_never_outlines_stride2_conv():
    """Correctness-critical: stride>1 must never reach this vectorization path.

    Verified separately (JIT execution vs. a NumPy reference) that the
    legacy affine vectorizer silently produces wrong results for a strided
    access pattern — this guard is what prevents that from ever being
    reachable, not just an optimization-scope choice.
    """
    new_text, count = outline_stride1_convs(STRIDE2_PAYLOAD)

    assert count == 0
    assert new_text == STRIDE2_PAYLOAD
    assert "func.call @__edgeslide_vec_conv" not in new_text


def test_no_match_returns_input_unchanged():
    text = "module { func.func @f() { return } }"
    new_text, count = outline_stride1_convs(text)

    assert count == 0
    assert new_text == text


TWO_STRIDE1_CONVS_PAYLOAD = """\
module {
  func.func @forward(%arg0: memref<1x3x256x256xf32, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x8x256x256xf32> attributes {llvm.emit_c_interface} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x3x258x258xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x8x256x256xf32>
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, edgeslice.op_id = "op_8", strides = dense<1> : vector<2xi64>} ins(%subview_a, %0 : memref<1x3x6x258xf32, strided<[199692, 66564, 258, 1], offset: ?>>, memref<8x3x3x3xf32>) outs(%subview_b : memref<1x8x4x256xf32, strided<[524288, 65536, 256, 1], offset: ?>>)
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, edgeslice.op_id = "op_9", strides = dense<1> : vector<2xi64>} ins(%subview_c, %1 : memref<1x8x4x256xf32, strided<[524288, 65536, 256, 1], offset: ?>>, memref<8x8x3x3xf32>) outs(%subview_d : memref<1x8x4x256xf32, strided<[524288, 65536, 256, 1], offset: ?>>)
    return %alloc_1 : memref<1x8x256x256xf32>
  }
}
"""


def test_multiple_stride1_convs_get_distinct_helpers():
    new_text, count = outline_stride1_convs(TWO_STRIDE1_CONVS_PAYLOAD)

    assert count == 2
    assert "@__edgeslide_vec_conv_op_8_0" in new_text
    assert "@__edgeslide_vec_conv_op_9_1" in new_text
    assert "linalg.conv_2d_nchw_fchw" not in new_text
