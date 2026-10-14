from __future__ import annotations

import re

import pytest

from edgeslide.dag.graph import OperationDAG, TensorInfo
from edgeslide.mlir_transform.pipeline import find_mlir_opt
from edgeslide.mlir_transform.sram import (
    generate_sram_promotion_transform,
    materialize_sram_placement,
)

MEMREF_PAYLOAD = r"""
func.func @tile(%input: memref<8x8xf32>, %output: memref<8x8xf32>) {
  %in_tile = memref.subview %input[0, 0] [4, 8] [1, 1]
    : memref<8x8xf32> to memref<4x8xf32, strided<[8, 1]>>
  %out_tile = memref.subview %output[0, 0] [4, 8] [1, 1]
    : memref<8x8xf32> to memref<4x8xf32, strided<[8, 1]>>
  linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"], edgeslice.op_id = "op_0"}
      ins(%in_tile : memref<4x8xf32, strided<[8, 1]>>)
      outs(%out_tile : memref<4x8xf32, strided<[8, 1]>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}
"""


def _promotion_ir() -> str:
    dag = OperationDAG()
    shape = (4, 8)
    op_id = dag.add_node(
        "linalg.generic",
        inputs=[TensorInfo(shape, "f32")],
        output=TensorInfo(shape, "f32"),
        params={"elementwise_identity": True, "edgeslice_op_id": "op_0"},
    )
    return generate_sram_promotion_transform(
        [{
            "op_id": op_id,
            "op_type": "linalg.generic",
            "edgeslice_op_id": "op_0",
        }],
        dag,
        memory_space=1,
        alignment=64,
    )


@pytest.mark.cpu
def test_promotion_transform_names_real_memory_space_and_all_operands():
    transform = _promotion_ir()
    assert "operands_to_promote = [0, 1]" in transform
    assert "memory_space = 1 : i64" in transform
    assert "transform.structured.promote" in transform


@pytest.mark.integration
@pytest.mark.mlir
def test_materialization_allocates_and_copies_in_sram_space(tmp_path):
    mlir_opt = find_mlir_opt()
    if not mlir_opt:
        pytest.skip("mlir-opt is unavailable")
    payload = tmp_path / "payload.mlir"
    transform = tmp_path / "promote.mlir"
    output = tmp_path / "sram.mlir"
    payload.write_text(MEMREF_PAYLOAD)
    transform.write_text(_promotion_ir())

    report = materialize_sram_placement(
        payload,
        transform,
        output,
        capacity_bytes=4096,
        memory_space=1,
        double_buffer=False,
        mlir_opt=mlir_opt,
    )

    text = output.read_text()
    assert report.promoted_allocation_count >= 2
    assert report.copy_count >= 2
    assert re.search(r"memref<4x8xf32,\s*1>", text)
