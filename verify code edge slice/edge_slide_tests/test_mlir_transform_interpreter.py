"""Integration coverage for the executable EdgeSlice transform script."""

from __future__ import annotations

import re

import pytest

from edgeslide.dag.graph import OperationDAG, TensorInfo
from edgeslide.mlir_transform.pipeline import find_mlir_opt, run_transform_interpreter
from edgeslide.transform.rewrite import SlicedRewrite

PAYLOAD_MLIR = """
func.func @matmul(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>)
    -> tensor<8x8xf32> {
  %empty = tensor.empty() : tensor<8x8xf32>
  %result = linalg.matmul {edgeslice.op_id = "op_0"}
      ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
      outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}
"""


def _matmul_transform_ir(tag: str = "op_0") -> str:
    dag = OperationDAG()
    output = TensorInfo((8, 8), "f32")
    op_id = dag.add_node(
        "linalg.matmul",
        inputs=[TensorInfo((8, 8), "f32"), TensorInfo((8, 8), "f32")],
        output=output,
    )
    plan = [{
        "op_id": op_id,
        "op_type": "linalg.matmul",
        # "M", not "batch": a plain rank-2 matmul has no batch dimension, and
        # the shared mapping in dimension_selector rejects the wrong name
        # rather than silently resolving it to index 0.
        "slice_dimension": "M",
        "tile_size": 4,
        "original_shape": output.shape,
        "edgeslice_op_id": tag,
    }]
    return SlicedRewrite(dag, plan).generate_transform_ir()


def _plan_with_dimension(slice_dimension: str, op_type: str, shape: tuple):
    dag = OperationDAG()
    output = TensorInfo(shape, "f32")
    op_id = dag.add_node(op_type, inputs=[TensorInfo(shape, "f32")], output=output)
    return dag, [{
        "op_id": op_id,
        "op_type": op_type,
        "slice_dimension": slice_dimension,
        "tile_size": 4,
        "original_shape": shape,
        "edgeslice_op_id": "op_0",
    }]


@pytest.mark.cpu
def test_batch_is_rejected_for_a_rank_2_matmul():
    """A matmul has no batch axis; naming one must fail, not resolve to M.

    Silently mapping "batch" to index 0 would tile the M dimension while the
    plan, the cost model, and the emitted comment all claim a batch tiling.
    """
    dag, plan = _plan_with_dimension("batch", "linalg.matmul", (8, 8))
    with pytest.raises(ValueError, match="Unsupported slice dimension 'batch'"):
        SlicedRewrite(dag, plan).generate_transform_ir()


@pytest.mark.cpu
def test_batch_is_accepted_for_a_batch_matmul():
    """The same name is valid once the op actually has a batch dimension."""
    dag, plan = _plan_with_dimension("batch", "linalg.batch_matmul", (4, 8, 8))
    assert "tile_using_forall" in SlicedRewrite(dag, plan).generate_transform_ir()


@pytest.mark.integration
@pytest.mark.cpu
def test_transform_interpreter_materializes_real_payload_slices(tmp_path):
    mlir_opt = find_mlir_opt()
    if not mlir_opt:
        pytest.skip("mlir-opt with transform-interpreter is not installed")

    payload_path = tmp_path / "payload.mlir"
    transform_path = tmp_path / "transform.mlir"
    output_path = tmp_path / "sliced.mlir"
    payload_path.write_text(PAYLOAD_MLIR)
    transform_path.write_text(_matmul_transform_ir())

    run_transform_interpreter(
        str(payload_path),
        str(transform_path),
        str(output_path),
        mlir_opt=mlir_opt,
    )

    transformed = output_path.read_text()
    assert re.search(r"(?m)^\s*%[^=]+\s*=\s*scf\.forall\b", transformed)
    assert re.search(r"(?m)^\s*%[^=]+\s*=\s*tensor\.extract_slice\b", transformed)
    assert re.search(r"(?m)^\s*tensor\.parallel_insert_slice\b", transformed)


@pytest.mark.integration
@pytest.mark.cpu
def test_transform_interpreter_rejects_an_unmatched_plan(tmp_path):
    mlir_opt = find_mlir_opt()
    if not mlir_opt:
        pytest.skip("mlir-opt with transform-interpreter is not installed")

    payload_path = tmp_path / "payload.mlir"
    transform_path = tmp_path / "transform.mlir"
    output_path = tmp_path / "sliced.mlir"
    payload_path.write_text(PAYLOAD_MLIR)
    transform_path.write_text(_matmul_transform_ir(tag="missing_op"))

    with pytest.raises(RuntimeError, match="without materializing"):
        run_transform_interpreter(
            str(payload_path),
            str(transform_path),
            str(output_path),
            mlir_opt=mlir_opt,
        )
