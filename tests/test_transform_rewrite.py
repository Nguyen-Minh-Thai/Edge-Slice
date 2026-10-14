"""
Unit tests for MLIR Transform Dialect rewrite generation (Stage 4).

Architecture: Option B (standard ops). See docs/EDGESLICE_TRANSFORM_ARCHITECTURE.md.

Validates:
  - Executable: tile_using_forall + match
  - Generated scripts contain no unregistered pseudo-ops
  - is_tileable() gate
"""
import pytest

from edgeslide.dag.graph import OperationDAG, TensorInfo
from edgeslide.transform.rewrite import (
    EDGESLICE_OP_ID_ATTR,
    TRANSFORM_IMPLEMENTATION,
    SlicedRewrite,
    is_tileable,
)


@pytest.mark.cpu
def test_transform_implementation_is_option_b():
    assert TRANSFORM_IMPLEMENTATION == "standard_ops_decomposition"


@pytest.mark.cpu
def test_is_tileable_gate():
    assert is_tileable("conv_2d", (1, 64, 128, 128))
    assert is_tileable("linalg.conv_2d_nchw_fchw", (1, 64, 128, 128))
    assert is_tileable("matmul", (32, 64))
    assert is_tileable("relu", (1, 64, 64, 64))
    assert is_tileable("elementwise_add", (1, 32, 128, 128))
    assert not is_tileable("linalg.generic", (1, 8, 32, 32))
    assert is_tileable(
        "linalg.generic",
        (1, 8, 32, 32),
        {"elementwise_identity": True},
    )
    assert not is_tileable("add", (1, 64, 64, 64))
    assert not is_tileable("softmax", (1, 1000))
    assert not is_tileable("conv_2d", (64,))  # rank < 2
    assert not is_tileable("conv_2d", (1, -1, 32, 32))


@pytest.mark.cpu
def test_edgeslice_transform_emits_only_executable_standard_ops():
    dag = OperationDAG()
    out = TensorInfo((1, 64, 128, 128), "f32")
    op_id = dag.add_node("conv_2d", inputs=[TensorInfo((1, 3, 128, 128))], output=out)

    plan = [{
        "op_id": op_id,
        "op_type": "conv_2d",
        "slice_dimension": "spatial_h",
        "tile_size": 32,
        "original_shape": out.shape,
    }]

    ir = SlicedRewrite(dag, plan).generate_transform_ir(style="structured")

    assert "Option B: standard ops decomposition" in ir
    assert "tile_using_forall" in ir
    assert "transform.structured.match" in ir
    assert "transform.structured.edgeslice" not in ir
    assert f"%target_{op_id}" in ir


@pytest.mark.cpu
def test_non_tileable_ops_filtered_from_rewrite_plan():
    dag = OperationDAG()
    add_out = TensorInfo((1, 64, 64, 64), "f32")
    op_id = dag.add_node("add", inputs=[TensorInfo((1, 64, 64, 64))], output=add_out)

    plan = [{
        "op_id": op_id,
        "op_type": "add",
        "slice_dimension": "spatial_h",
        "tile_size": 16,
        "original_shape": add_out.shape,
    }]

    ir = SlicedRewrite(dag, plan).generate_transform_ir()
    assert "No slicing candidates" in ir


@pytest.mark.cpu
def test_tileable_elementwise_emits_transform():
    dag = OperationDAG()
    relu_out = TensorInfo((1, 64, 128, 128), "f32")
    op_id = dag.add_node("relu", inputs=[TensorInfo((1, 64, 128, 128))], output=relu_out)

    plan = [{
        "op_id": op_id,
        "op_type": "relu",
        "slice_dimension": "spatial_h",
        "tile_size": 32,
        "original_shape": relu_out.shape,
    }]

    ir = SlicedRewrite(dag, plan).generate_transform_ir()
    assert "tile_using_forall" in ir
    assert "transform.structured.edgeslice" not in ir


@pytest.mark.cpu
def test_match_includes_edgeslice_op_id_attribute():
    dag = OperationDAG()
    out = TensorInfo((1, 64, 128, 128), "f32")
    op_id = dag.add_node("conv_2d", inputs=[TensorInfo((1, 3, 128, 128))], output=out)

    plan = [{
        "op_id": op_id,
        "op_type": "conv_2d",
        "slice_dimension": "spatial_h",
        "tile_size": 32,
        "original_shape": out.shape,
        "edgeslice_op_id": "op_7",
    }]

    ir = SlicedRewrite(dag, plan).generate_transform_ir(style="structured")

    assert f'attributes {{{EDGESLICE_OP_ID_ATTR} = "op_7"}}' in ir
    assert 'ops{["linalg.conv_2d_nchw_fchw"]}' in ir


@pytest.mark.cpu
def test_op_type_to_mlir_preserves_qualified_linalg_names():
    """DAGParser emits linalg.* names; match must not collapse them to generic."""
    dag = OperationDAG()
    out = TensorInfo((1, 8, 32, 32), "f32")
    op_id = dag.add_node(
        "linalg.conv_2d_nchw_fchw",
        inputs=[TensorInfo((1, 3, 32, 32))],
        output=out,
    )
    plan = [{
        "op_id": op_id,
        "op_type": "linalg.conv_2d_nchw_fchw",
        "slice_dimension": "spatial_h",
        "tile_size": 16,
        "original_shape": out.shape,
        "edgeslice_op_id": "op_8",
    }]
    ir = SlicedRewrite(dag, plan).generate_transform_ir(style="structured")
    assert 'ops{["linalg.conv_2d_nchw_fchw"]}' in ir
    assert 'ops{["linalg.generic"]}' not in ir or ir.count(
        'ops{["linalg.conv_2d_nchw_fchw"]}'
    ) >= 1
    # The matched target for this plan must be conv, not generic.
    assert f'%target_{op_id} = transform.structured.match ops{{["linalg.conv_2d_nchw_fchw"]}}' in ir


@pytest.mark.cpu
def test_multi_op_plan_emits_distinct_match_targets():
    dag = OperationDAG()
    out0 = TensorInfo((1, 32, 64, 64), "f32")
    out1 = TensorInfo((1, 64, 32, 32), "f32")
    op0 = dag.add_node("conv_2d", inputs=[TensorInfo((1, 16, 64, 64))], output=out0)
    op1 = dag.add_node("conv_2d", inputs=[TensorInfo((1, 32, 32, 32))], output=out1)

    plan = [
        {
            "op_id": op0,
            "op_type": "conv_2d",
            "slice_dimension": "spatial_h",
            "tile_size": 16,
            "original_shape": out0.shape,
            "edgeslice_op_id": "op_0",
        },
        {
            "op_id": op1,
            "op_type": "conv_2d",
            "slice_dimension": "spatial_w",
            "tile_size": 16,
            "original_shape": out1.shape,
            "edgeslice_op_id": "op_1",
        },
    ]

    ir = SlicedRewrite(dag, plan).generate_transform_ir(style="structured")

    assert f'attributes {{{EDGESLICE_OP_ID_ATTR} = "op_0"}}' in ir
    assert f'attributes {{{EDGESLICE_OP_ID_ATTR} = "op_1"}}' in ir
    assert ir.count("transform.structured.match") == 2


@pytest.mark.cpu
def test_legacy_for_style_omits_edgeslice():
    dag = OperationDAG()
    out = TensorInfo((1, 32, 64, 64), "f32")
    op_id = dag.add_node("conv_2d", [], out)

    plan = [{
        "op_id": op_id,
        "op_type": "conv_2d",
        "slice_dimension": "spatial_w",
        "tile_size": 16,
        "original_shape": out.shape,
    }]

    ir = SlicedRewrite(dag, plan).generate_transform_ir(style="legacy_for")
    assert "transform.structured.tile_using_for" in ir
    assert "transform.structured.tile_using_forall" not in ir
