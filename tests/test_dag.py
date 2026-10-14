"""
CPU Tests for DAG construction and graph utilities (Stage 1).

These tests validate the OperationDAG data structures used by EdgeSlice
to represent PyTorch models after torch-mlir lowering to Linalg.

Can run without torch_mlir (pure Python logic).
"""
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for DAG parser tests")

from edgeslide.dag.graph import OperationDAG, TensorInfo


@pytest.mark.cpu
def test_tensor_info_size_bytes():
    t = TensorInfo(shape=(1, 64, 32, 32), dtype="f32")
    assert t.size_bytes == 1 * 64 * 32 * 32 * 4

    t16 = TensorInfo(shape=(2, 16, 8), dtype="f16")
    assert t16.size_bytes == 2 * 16 * 8 * 2

@pytest.mark.cpu
def test_tensor_info_dynamic_shape_requires_bounds():
    t = TensorInfo(shape=(1, -1, 32, 32), dtype="f32", name="activation")

    with pytest.raises(ValueError, match="dynamic dimension"):
        _ = t.size_bytes

@pytest.mark.cpu
def test_tensor_info_dynamic_shape_uses_bounds():
    t = TensorInfo(
        shape=(1, -1, 32, 32),
        dtype="f16",
        name="activation",
        shape_bounds={1: 64},
    )

    assert t.resolved_shape == (1, 64, 32, 32)
    assert t.size_bytes == 1 * 64 * 32 * 32 * 2


@pytest.mark.cpu
def test_add_node_and_edges():
    dag = OperationDAG()
    out0 = TensorInfo((1, 16, 32, 32))
    out1 = TensorInfo((1, 32, 16, 16))

    n0 = dag.add_node("linalg.conv_2d", inputs=[TensorInfo((1,3,32,32))], output=out0)
    n1 = dag.add_node("linalg.conv_2d", inputs=[out0], output=out1)

    dag.add_edge(n0, n1)

    assert len(dag.nodes) == 2
    assert len(dag.edges) == 1
    assert dag.nodes[n0].consumers == [n1]


@pytest.mark.cpu
def test_topological_order_basic(synthetic_dag):
    order = synthetic_dag.topological_order()
    assert len(order) == 3
    # producer must appear before consumer
    assert order.index(0) < order.index(1) < order.index(2)


@pytest.mark.cpu
def test_topological_order_detects_no_cycle():
    dag = OperationDAG()
    t = TensorInfo((4, 4))
    a = dag.add_node("linalg.matmul", inputs=[], output=t)
    b = dag.add_node("linalg.generic", inputs=[t], output=TensorInfo((4,)))
    dag.add_edge(a, b)
    order = dag.topological_order()
    assert order == [a, b]


@pytest.mark.cpu
def test_synthetic_dag_has_expected_shapes(synthetic_dag):
    assert synthetic_dag.nodes[0].output_tensor.shape == (1, 32, 128, 128)
    assert synthetic_dag.nodes[1].output_tensor.shape == (1, 64, 64, 64)
