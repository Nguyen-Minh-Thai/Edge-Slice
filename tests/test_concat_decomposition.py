"""`tensor.concat` must be decomposed before bufferization, or U-Net cannot compile.

MLIR 19 has no bufferization for `tensor.concat`; one-shot bufferize fails with
"op was not bufferized" and takes the whole payload down. U-Net's skip
connections are concats, so before this the workload could be parsed and
planned but never compiled or executed — every U-Net number was a static
estimate with no physical counterpart, and nothing in the repo said so.

The decomposition exists in MLIR 19 only as a transform-dialect pattern, so it
runs through the interpreter, and the interpreter's output nesting is what makes
the surrounding plumbing subtle. Each assertion here corresponds to a failure
mode that was actually hit while wiring it up.
"""

from __future__ import annotations

import pytest

from edgeslide.mlir_transform.lowering import (
    payload_needs_concat_decomposition,
    unwrap_redundant_module,
)

pytestmark = pytest.mark.cpu


def test_a_payload_without_concat_is_detected_as_not_needing_work():
    assert not payload_needs_concat_decomposition("func.func @f() { return }")


def test_a_payload_with_concat_is_detected():
    assert payload_needs_concat_decomposition(
        "%c = tensor.concat dim(1) %a, %b : (tensor<1xf32>, tensor<1xf32>)"
        " -> tensor<2xf32>"
    )


# ---------------------------------------------------------------------------
# unwrap_redundant_module
# ---------------------------------------------------------------------------

DOUBLE = """module {
  module {
    func.func @forward() {
      return
    }
  }
}
"""


def test_a_doubly_nested_module_is_flattened_by_one_level():
    """The failure this prevents is silent and reported far from its cause.

    mlir-translate exports the *outer* module, which by then contains only the
    inner one, so it emits an LLVM module with zero function definitions. The
    JIT then reports "Unknown function symbol _mlir_ciface_forward" with
    nothing in between to suggest the module was empty.
    """
    result = unwrap_redundant_module(DOUBLE)
    assert result.count("module {") == 1
    assert "func.func @forward" in result
    # Braces must balance, or mlir-opt reports "expected operation name in
    # quotes" on the stray one.
    assert result.count("{") == result.count("}")


def test_the_flattened_body_is_dedented():
    result = unwrap_redundant_module(DOUBLE)
    assert "  func.func @forward() {" in result


def test_a_singly_nested_module_is_left_alone():
    single = "module {\n  func.func @f() {\n    return\n  }\n}\n"
    assert unwrap_redundant_module(single) == single


def test_a_module_holding_two_real_modules_is_left_alone():
    """Only an *immediately* nested lone module is redundant structure."""
    two = (
        "module {\n"
        "  module {\n    func.func @a() {\n      return\n    }\n  }\n"
        "  module {\n    func.func @b() {\n      return\n    }\n  }\n"
        "}\n"
    )
    assert unwrap_redundant_module(two) == two


def test_content_before_the_nested_module_is_preserved():
    """Affine-map aliases sit above the module and must survive."""
    text = "#map = affine_map<(d0) -> (d0)>\n" + DOUBLE
    result = unwrap_redundant_module(text)
    assert result.startswith("#map = affine_map<(d0) -> (d0)>")
    assert "func.func @forward" in result
    assert result.count("{") == result.count("}")


def test_unbalanced_input_is_returned_unchanged_rather_than_corrupted():
    broken = "module {\n  module {\n    func.func @f() {\n"
    assert unwrap_redundant_module(broken) == broken


# ---------------------------------------------------------------------------
# The real decomposition, against mlir-opt
# ---------------------------------------------------------------------------

def test_decomposition_removes_the_concat_and_bufferizes(tmp_path):
    from tests.mlir_helpers import require_mlir_jit_tools

    require_mlir_jit_tools()
    from edgeslide.mlir_transform.errors import require_tool, run_tool
    from edgeslide.mlir_transform.lowering import decompose_tensor_concat

    payload = """module {
  func.func @f(%a: tensor<1x2x3x3xf32>, %b: tensor<1x2x3x3xf32>)
      -> tensor<1x4x3x3xf32> {
    %c = tensor.concat dim(1) %a, %b
      : (tensor<1x2x3x3xf32>, tensor<1x2x3x3xf32>) -> tensor<1x4x3x3xf32>
    return %c : tensor<1x4x3x3xf32>
  }
}
"""
    decomposed = decompose_tensor_concat(payload, work_dir=tmp_path)
    assert "tensor.concat" not in decomposed
    assert "tensor.insert_slice" in decomposed
    assert "func.func @f" in decomposed

    # The point of the decomposition: the result bufferizes.
    source = tmp_path / "decomposed.mlir"
    source.write_text(decomposed)
    run_tool(
        [
            require_tool("mlir-opt"),
            str(source),
            "--one-shot-bufferize=bufferize-function-boundaries",
            "-o",
            str(tmp_path / "bufferized.mlir"),
        ],
        stage="bufferize decomposed concat",
    )


def test_a_payload_without_concat_is_returned_unchanged(tmp_path):
    from edgeslide.mlir_transform.lowering import decompose_tensor_concat

    payload = "module {\n  func.func @f() {\n    return\n  }\n}\n"
    # No mlir-opt invocation happens on this path, so it works toolchain-free.
    assert decompose_tensor_concat(payload, work_dir=tmp_path) == payload
