"""
CPU Integration Tests: Full EdgeSlice pipeline on CPU (Stages 1-4).

Requires: torch + torch_mlir
Skips gracefully when torch_mlir is unavailable or lowering fails.

Tests:
- Parse small model → DAG on CPU
- Liveness + peak memory analysis
- Dimension selection + rewrite plan generation
- Transform IR emission (textual)
- MLIR end-to-end numerical equivalence (Phase 2 & 3)

These validate that the pipeline described in the proposal
produces reasonable artifacts on CPU-only machines.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for pipeline tests")
nn = torch.nn

from tests.mlir_helpers import (
    compile_sliced_payload,
    require_mlir_execution_tools,
    require_mlir_jit_tools,
    require_torch_mlir,
    run_jit_in_subprocess,
)

try:
    import torch_mlir  # noqa: F401

    HAS_TORCH_MLIR = True
except ImportError:
    HAS_TORCH_MLIR = False


pytestmark = pytest.mark.integration

MLIR_E2E_RTOL = 5e-3
MLIR_E2E_ATOL = 5e-3
MLIR_E2E_INPUT_SHAPE = (1, 3, 32, 32)
MLIR_E2E_L1_CACHE_KB = 32
JIT_TIMEOUT_SEC = 180

logger = logging.getLogger(__name__)


@pytest.mark.cpu
def test_parser_stamps_edgeslice_op_id_on_dag_and_mlir(tiny_conv_model, small_input):
    """Each Linalg op gets edgeslice_op_id in DAG params and payload MLIR."""
    require_torch_mlir()
    from edgeslide.dag.parser import DAGParser

    parser = DAGParser(tiny_conv_model, small_input)
    dag = parser.parse()

    linalg_nodes = [
        (op_id, node)
        for op_id, node in dag.nodes.items()
        if "linalg" in node.op_type or "conv" in node.op_type
    ]
    assert len(linalg_nodes) >= 2

    tags = {node.params["edgeslice_op_id"] for _, node in linalg_nodes}
    assert len(tags) == len(linalg_nodes)

    with tempfile.TemporaryDirectory(prefix="edgeslice_payload_") as tmp:
        mlir_path = Path(tmp) / "payload.mlir"
        parser.save_mlir(str(mlir_path))
        mlir_text = mlir_path.read_text()

    for op_id, node in linalg_nodes:
        tag = node.params["edgeslice_op_id"]
        assert tag == f"op_{op_id}"
        assert f'edgeslice.op_id = "{tag}"' in mlir_text


@pytest.mark.cpu
def test_pass_manager_rewrite_plan_includes_edgeslice_op_id(tiny_conv_model, small_input):
    require_torch_mlir()
    from edgeslide.dag.parser import DAGParser
    from edgeslide.transform.pass_manager import EdgeSlicePassManager

    dag = DAGParser(tiny_conv_model, small_input).parse()
    results = EdgeSlicePassManager(dag, l1_cache_bytes=64 * 1024).run()

    for entry in results.get("rewrite_plan", []):
        assert "edgeslice_op_id" in entry
        assert entry["edgeslice_op_id"] == f"op_{entry['op_id']}"


@pytest.mark.cpu
def test_parse_tiny_model_to_dag_on_cpu(tiny_conv_model, small_input):
    require_torch_mlir()
    from edgeslide.dag.parser import DAGParser

    parser = DAGParser(tiny_conv_model, small_input.to("cpu"))
    dag = parser.parse()

    assert len(dag.nodes) > 0
    op_types = {n.op_type for n in dag.nodes.values()}
    assert any("conv" in ot or "linalg" in ot for ot in op_types)


@pytest.mark.cpu
def test_full_analysis_pipeline_produces_candidates(tiny_conv_model, small_input):
    require_torch_mlir()
    from edgeslide.dag.parser import DAGParser
    from edgeslide.liveness.peak_memory import PeakMemoryEstimator
    from edgeslide.tiling.dimension_selector import DimensionSelector

    parser = DAGParser(tiny_conv_model, small_input)
    dag = parser.parse()

    est = PeakMemoryEstimator(dag, l1_cache_bytes=128 * 1024)
    cands = est.slicing_candidates()
    dims = DimensionSelector(dag).select_all()

    assert isinstance(cands, list)
    assert isinstance(dims, dict)


@pytest.mark.cpu
def test_edgeslice_pass_manager_emits_transform_ir(tiny_conv_model, small_input):
    """Stages 1-3 via EdgeSlicePassManager — main frontend API entry point."""
    require_torch_mlir()
    from edgeslide.dag.parser import DAGParser
    from edgeslide.transform.pass_manager import EdgeSlicePassManager

    dag = DAGParser(tiny_conv_model, small_input).parse()
    results = EdgeSlicePassManager(dag, l1_cache_bytes=64 * 1024).run()

    assert "transform_ir" in results
    transform_ir = results["transform_ir"]
    assert isinstance(transform_ir, str) and len(transform_ir) > 0
    assert "transform.named_sequence" in transform_ir or "No slicing" in transform_ir
    if results.get("rewrite_plan"):
        assert "transform.structured.tile_using_forall" in transform_ir
        assert "transform.structured.edgeslice" not in transform_ir
    assert "peak_op" in results
    assert "slicing_candidates" in results
    assert isinstance(results["slicing_candidates"], list)


@pytest.mark.cpu
def test_transform_rewrite_emits_valid_text(tiny_conv_model, small_input):
    require_torch_mlir()
    from edgeslide.dag.parser import DAGParser
    from edgeslide.liveness.peak_memory import PeakMemoryEstimator
    from edgeslide.tiling.dimension_selector import DimensionSelector
    from edgeslide.transform.rewrite import SlicedRewrite, is_tileable

    parser = DAGParser(tiny_conv_model, small_input)
    dag = parser.parse()

    est = PeakMemoryEstimator(dag, l1_cache_bytes=64 * 1024)
    cands = est.slicing_candidates()
    sel = DimensionSelector(dag)

    plan = []
    for c in cands:
        op_id = c["op_id"]
        if not is_tileable(c["op_type"], dag.nodes[op_id].output_tensor.shape):
            continue
        plan.append({
            "op_id": op_id,
            "op_type": c["op_type"],
            "slice_dimension": sel.select(op_id),
            "tile_size": 32,
            "original_shape": dag.nodes[op_id].output_tensor.shape,
        })
        if len(plan) >= 2:
            break

    ir_text = SlicedRewrite(dag, plan).generate_transform_ir(style="structured")
    assert "transform.named_sequence" in ir_text or "No slicing" in ir_text
    if plan:
        assert "tile_using_forall" in ir_text
        assert "transform.structured.edgeslice" not in ir_text


@pytest.mark.cpu
def test_pipeline_respects_cpu_device_explicitly(tiny_conv_model, deterministic_env):
    require_torch_mlir()
    from edgeslide.dag.parser import DAGParser

    x = torch.randn(1, 3, 48, 48, device="cpu")
    parser = DAGParser(tiny_conv_model.to("cpu"), x)
    dag = parser.parse()

    for node in dag.nodes.values():
        assert all(isinstance(d, int) for d in node.output_tensor.shape)


@pytest.mark.integration
@pytest.mark.cpu
@pytest.mark.slow
def test_mlir_pipeline_execution_if_available(tiny_conv_relu_model, deterministic_env):
    """
    End-to-end Phase 2 & 3: compile sliced_payload.mlir and verify numerical
    equivalence against the PyTorch reference forward pass.
    """
    require_torch_mlir()
    require_mlir_execution_tools()
    require_mlir_jit_tools()

    logging.basicConfig(level=logging.INFO)

    input_tensor = torch.randn(*MLIR_E2E_INPUT_SHAPE, device="cpu")
    with torch.no_grad():
        y_ref = tiny_conv_relu_model(input_tensor)

    logger.info(
        "PyTorch reference shape=%s mean=%.6f",
        tuple(y_ref.shape),
        y_ref.mean().item(),
    )

    with tempfile.TemporaryDirectory(prefix="edgeslide_mlir_e2e_") as tmp:
        output_dir = Path(tmp)
        try:
            artifacts = compile_sliced_payload(
                tiny_conv_relu_model,
                input_tensor,
                output_dir,
                l1_cache_kb=MLIR_E2E_L1_CACHE_KB,
                entry_func="forward",
            )
        except Exception as exc:
            logger.exception("EdgeSlideMLIRPipeline.compile() failed")
            pytest.fail(f"MLIR compile pipeline failed: {exc}")

        sliced_path = Path(artifacts["sliced_payload_mlir"])
        assert sliced_path.is_file(), f"Missing sliced_payload.mlir at {sliced_path}"
        assert "func.func @forward" in sliced_path.read_text()

        try:
            y_mlir = run_jit_in_subprocess(
                sliced_path,
                output_dir,
                input_tensor,
                entry_func="forward",
                timeout_sec=JIT_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"MLIR JIT execution timed out after {JIT_TIMEOUT_SEC}s")

        logger.info(
            "MLIR output shape=%s mean=%.6f",
            tuple(y_mlir.shape),
            y_mlir.mean().item(),
        )

        assert y_mlir.shape == y_ref.shape, (
            f"Shape mismatch: mlir={tuple(y_mlir.shape)} ref={tuple(y_ref.shape)}"
        )

        if not torch.allclose(y_mlir, y_ref, rtol=MLIR_E2E_RTOL, atol=MLIR_E2E_ATOL):
            max_diff = (y_mlir - y_ref).abs().max().item()
            mean_diff = (y_mlir - y_ref).abs().mean().item()
            pytest.fail(
                f"MLIR output diverges from PyTorch "
                f"(max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
                f"rtol={MLIR_E2E_RTOL}, atol={MLIR_E2E_ATOL})"
            )
