"""
Generated Transform IR must be valid MLIR, not merely plausible text.

``SlicedRewrite`` builds transform scripts by string concatenation, so a
malformed handle list or a bad tile-size vector is invisible until the
interpreter runs — by which point the failure surfaces as an opaque parse
diagnostic far from the code that emitted it. These tests pin the emitted text
(golden) and hand every generated script to the real ``mlir-opt`` (syntax and
dialect validity).

No torch required: the DAG is built by hand.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from edgeslide.dag.graph import OperationDAG, TensorInfo
from edgeslide.mlir_transform.errors import InvalidTransformStyleError
from edgeslide.mlir_transform.pipeline import find_mlir_opt
from edgeslide.transform.pass_manager import EdgeSlicePassManager
from edgeslide.transform.rewrite import TRANSFORM_STYLES, SlicedRewrite

GOLDEN_DIR = Path(__file__).parent / "golden"

# Regenerate goldens with:
#   EDGESLIDE_UPDATE_GOLDEN=1 pytest tests/test_transform_ir_validation.py
UPDATE_ENV = "EDGESLIDE_UPDATE_GOLDEN"


def _two_conv_dag() -> OperationDAG:
    """conv -> conv, both far larger than the 64 KB budget used below.

    Fixed shapes on purpose: the goldens encode the exact tile sizes the
    planner derives from them.
    """
    dag = OperationDAG()
    mid = TensorInfo(shape=(1, 32, 128, 128), dtype="f32")
    out = TensorInfo(shape=(1, 64, 64, 64), dtype="f32")
    first = dag.add_node(
        "linalg.conv_2d_nchw_fchw",
        inputs=[TensorInfo((1, 3, 128, 128), "f32")],
        output=mid,
        params={"edgeslice_op_id": "op_0", "kernel_size": (3, 3)},
    )
    second = dag.add_node(
        "linalg.conv_2d_nchw_fchw",
        inputs=[mid],
        output=out,
        params={"edgeslice_op_id": "op_1", "kernel_size": (3, 3)},
    )
    dag.add_edge(first, second)
    return dag


def _generate(style: str) -> str:
    manager = EdgeSlicePassManager(_two_conv_dag(), l1_cache_bytes=64 * 1024)
    return manager.run(transform_style=style)["transform_ir"]


def _assert_matches_golden(name: str, actual: str) -> None:
    import os

    golden = GOLDEN_DIR / name
    if os.environ.get(UPDATE_ENV):
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(actual)
        pytest.skip(f"regenerated golden {name}")
    if not golden.is_file():
        pytest.fail(
            f"Missing golden {golden}. Regenerate with {UPDATE_ENV}=1 pytest"
        )
    assert actual == golden.read_text(), (
        f"Generated {name} drifted from the golden file. If the change is "
        f"intended, regenerate with {UPDATE_ENV}=1 pytest and review the diff."
    )


def _mlir_opt_or_skip() -> str:
    mlir_opt = find_mlir_opt()
    if mlir_opt is None:
        pytest.skip("mlir-opt not found; set $MLIR_OPT or install LLVM/MLIR")
    return mlir_opt


# --------------------------------------------------------------------------
# Golden tests (Phase 3.5)
# --------------------------------------------------------------------------


@pytest.mark.cpu
def test_structured_transform_ir_matches_golden():
    _assert_matches_golden("transform_structured.mlir", _generate("structured"))


@pytest.mark.cpu
def test_aggressive_transform_ir_matches_golden():
    _assert_matches_golden("transform_aggressive.mlir", _generate("aggressive"))


@pytest.mark.cpu
def test_legacy_for_transform_ir_matches_golden():
    _assert_matches_golden("transform_legacy_for.mlir", _generate("legacy_for"))


# --------------------------------------------------------------------------
# Real-toolchain validation (Phase 3.6)
# --------------------------------------------------------------------------


@pytest.mark.mlir
@pytest.mark.parametrize("style", sorted(TRANSFORM_STYLES))
def test_generated_ir_parses_with_mlir_opt(style, tmp_path):
    """Every emitted style must be syntactically and dialect-valid MLIR."""
    mlir_opt = _mlir_opt_or_skip()
    ir_path = tmp_path / f"transform_{style}.mlir"
    ir_path.write_text(_generate(style))

    result = subprocess.run(
        [mlir_opt, str(ir_path), "--verify-roundtrip"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{style} transform IR failed to parse:\n{result.stderr}"
    )


@pytest.mark.mlir
def test_aggressive_style_declares_one_handle_per_result(tmp_path):
    """tile_using_forall yields two handles; both must be bound.

    A result-count mismatch is a parse error, so mlir-opt accepting the script
    is the assertion. Pinned separately from the parametrized test because the
    aggressive path builds its handle list independently of the structured one.
    """
    mlir_opt = _mlir_opt_or_skip()
    ir = _generate("aggressive")
    assert "%tiled_" in ir and "%loop_" in ir

    ir_path = tmp_path / "aggressive.mlir"
    ir_path.write_text(ir)
    result = subprocess.run(
        [mlir_opt, str(ir_path), "--verify-roundtrip"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.mlir
def test_zero_tile_size_means_do_not_tile_that_dimension(tmp_path):
    """`0` is MLIR's own encoding for an untiled dimension — pin it.

    ``_build_tile_sizes`` emits 0 for every dimension except the sliced one.
    This drives a real payload through the interpreter to prove that produces a
    loop over the named dimension only, rather than a degenerate or empty nest.
    """
    mlir_opt = _mlir_opt_or_skip()
    merged = tmp_path / "payload_and_transform.mlir"
    # Tiling dim 2 (H = 32) by 8 must yield exactly 4 iterations.
    merged.write_text(
        """
module {
  func.func @forward(%in: tensor<1x3x34x34xf32>, %w: tensor<8x3x3x3xf32>,
                     %out: tensor<1x8x32x32xf32>) -> tensor<1x8x32x32xf32> {
    %0 = linalg.conv_2d_nchw_fchw {edgeslice.op_id = "op_0"}
      ins(%in, %w : tensor<1x3x34x34xf32>, tensor<8x3x3x3xf32>)
      outs(%out : tensor<1x8x32x32xf32>) -> tensor<1x8x32x32xf32>
    return %0 : tensor<1x8x32x32xf32>
  }
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %t = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]}
      attributes {edgeslice.op_id = "op_0"} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall %t
      tile_sizes [0, 0, 8, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
"""
    )
    result = subprocess.run(
        [mlir_opt, str(merged), "--transform-interpreter"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # 32 / 8 == 4 tiles along H, and only along H.
    assert "scf.forall (%arg3) in (4)" in result.stdout, result.stdout
    assert "tensor.extract_slice" in result.stdout
    assert "tensor.parallel_insert_slice" in result.stdout


# --------------------------------------------------------------------------
# Style validation (Phase 3.1)
# --------------------------------------------------------------------------


@pytest.mark.cpu
@pytest.mark.parametrize("style", sorted(TRANSFORM_STYLES))
def test_known_styles_are_accepted(style):
    assert _generate(style).strip()


@pytest.mark.cpu
@pytest.mark.parametrize("style", ["structred", "STRUCTURED", "", None, "forall"])
def test_unknown_style_is_rejected_by_pass_manager(style):
    manager = EdgeSlicePassManager(_two_conv_dag(), l1_cache_bytes=64 * 1024)
    with pytest.raises(InvalidTransformStyleError):
        manager.run(transform_style=style)


@pytest.mark.cpu
def test_unknown_style_is_rejected_even_with_no_candidates():
    """The empty-plan shortcut must not swallow a typo'd style."""
    rewriter = SlicedRewrite(OperationDAG(), [])
    assert not rewriter.rewrite_plan
    with pytest.raises(InvalidTransformStyleError):
        rewriter.generate_transform_ir(style="aggresive")


@pytest.mark.cpu
def test_invalid_style_error_lists_the_valid_styles():
    with pytest.raises(InvalidTransformStyleError) as excinfo:
        SlicedRewrite(OperationDAG(), []).generate_transform_ir(style="nope")
    message = str(excinfo.value)
    for style in TRANSFORM_STYLES:
        assert style in message


def _two_d_conv_dag() -> OperationDAG:
    """Conv whose tile needs two dimensions at a 64 KB budget.

    A single spatial row of this output is 64x64x4 = 16 KB, and the conv's
    input window plus weights push every 1D candidate over the budget, so the
    planner falls through to the dimension-pair cascade.
    """
    dag = OperationDAG()
    out = TensorInfo((1, 64, 64, 64), "f32")
    sink = TensorInfo((1, 8, 8, 8), "f32")
    conv = dag.add_node(
        "linalg.conv_2d_nchw_fchw",
        inputs=[TensorInfo((1, 32, 66, 66), "f32"), TensorInfo((64, 32, 3, 3), "f32")],
        output=out,
        params={"edgeslice_op_id": "op_0", "kernel_size": (3, 3)},
    )
    tail = dag.add_node(
        "linalg.generic",
        inputs=[out],
        output=sink,
        params={"edgeslice_op_id": "op_1", "elementwise_identity": True},
    )
    dag.add_edge(conv, tail)
    return dag


@pytest.mark.mlir
def test_a_two_dimensional_tile_lowers_to_a_two_induction_variable_loop(tmp_path):
    """Two nonzero tile sizes must produce one nest over both dimensions.

    The 1D case is pinned above; this is its 2D counterpart, and it is the only
    check that the pair cascade emits something MLIR actually accepts. Trip
    counts are deliberately unequal so the induction variables cannot be
    confused for one another: channel 8/2 = 4, spatial_h 32/4 = 8.
    """
    mlir_opt = _mlir_opt_or_skip()
    merged = tmp_path / "payload_and_transform_2d.mlir"
    merged.write_text(
        """
module {
  func.func @forward(%in: tensor<1x3x34x34xf32>, %w: tensor<8x3x3x3xf32>,
                     %out: tensor<1x8x32x32xf32>) -> tensor<1x8x32x32xf32> {
    %0 = linalg.conv_2d_nchw_fchw {edgeslice.op_id = "op_0"}
      ins(%in, %w : tensor<1x3x34x34xf32>, tensor<8x3x3x3xf32>)
      outs(%out : tensor<1x8x32x32xf32>) -> tensor<1x8x32x32xf32>
    return %0 : tensor<1x8x32x32xf32>
  }
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %t = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]}
      attributes {edgeslice.op_id = "op_0"} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall %t
      tile_sizes [0, 2, 4, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
"""
    )
    result = subprocess.run(
        [mlir_opt, str(merged), "--transform-interpreter"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # Two induction variables, and the order follows tensor dimension order.
    assert "scf.forall (%arg3, %arg4) in (4, 8)" in result.stdout, result.stdout
    # Both tiled dimensions appear in the output slice: 2 channels x 4 rows.
    assert "[1, 2, 4, 32]" in result.stdout, result.stdout
    # Channel tiling shrinks the weight tensor too — the cost model counts on it.
    assert "tensor.extract_slice %arg1[%" in result.stdout
    assert "[2, 3, 3, 3]" in result.stdout, result.stdout


@pytest.mark.mlir
def test_the_planners_own_two_dimensional_script_is_accepted_by_mlir(tmp_path):
    """End-to-end: what the pair cascade emits must parse and verify.

    The hand-written case above pins the semantics; this one pins that the
    planner actually produces that shape of script for a DAG that needs it.
    """
    mlir_opt = _mlir_opt_or_skip()
    results = EdgeSlicePassManager(
        _two_d_conv_dag(), l1_cache_bytes=64 * 1024
    ).run()
    plan = next(
        (p for p in results["rewrite_plan"] if p.get("slice_dimension_2")), None
    )
    assert plan is not None, "fixture must exercise the 2D cascade"

    ir = results["transform_ir"]
    tile_line = next(line for line in ir.splitlines() if "tile_sizes [" in line)
    emitted = [
        int(value)
        for value in tile_line.split("[", 1)[1].split("]", 1)[0].split(",")
    ]
    assert sum(1 for value in emitted if value) == 2, emitted

    ir_path = tmp_path / "planner_2d.mlir"
    ir_path.write_text(ir)
    result = subprocess.run(
        [mlir_opt, str(ir_path), "--verify-roundtrip"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
