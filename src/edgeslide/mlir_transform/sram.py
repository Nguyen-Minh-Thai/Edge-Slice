"""Explicit scratchpad placement for budget-approved tiled Linalg ops.

The tensor tiling transform only creates slices.  This module performs the
separate, semantically visible placement step after bufferization: promoted
operands are allocated in a non-default MLIR memory space and populated with
copies.  A target backend is responsible for mapping that address space to its
physical SRAM and for lowering copies to DMA; the generic CPU backend must not
pretend that address space 1 is ordinary malloc memory.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from edgeslide.mlir_transform.errors import (
    MLIRTransformError,
    TransformVerificationError,
    atomic_output,
    atomic_write_text,
    prepare_output_path,
    require_input_file,
    require_tool,
    run_tool,
)


@dataclass(frozen=True)
class SRAMPlacementReport:
    output_path: str
    bufferized_path: str
    promoted_allocation_count: int
    copy_count: int
    memory_space: int
    capacity_bytes: int
    double_buffer_reserved: bool
    double_buffer_materialized: bool
    sram_peak_bytes: int
    capacity_verified: bool
    inferred_size_count: int


def _mlir_op_name(op_type: str) -> str:
    lowered = op_type.lower()
    if op_type.startswith("linalg."):
        return op_type
    if "depthwise" in lowered:
        return "linalg.depthwise_conv_2d_nchw_chw"
    if "conv" in lowered:
        return "linalg.conv_2d_nchw_fchw"
    if "batch_matmul" in lowered:
        return "linalg.batch_matmul"
    if "matmul" in lowered:
        return "linalg.matmul"
    return "linalg.generic"


def generate_sram_promotion_transform(
    rewrite_plan: Iterable[Mapping[str, Any]],
    dag: Any,
    *,
    memory_space: int = 1,
    alignment: int = 64,
) -> str:
    """Generate a post-bufferization promotion transform.

    Each plan entry is matched by its stable ``edgeslice.op_id``. Every input
    and destination operand is promoted, so the resulting linalg op reads and
    writes scratchpad-backed buffers rather than merely carrying an annotation.
    """
    if memory_space == 0:
        raise ValueError("SRAM memory space must be non-zero")
    if alignment <= 0:
        raise ValueError("SRAM alignment must be positive")

    lines = [
        "// EdgeSlice explicit SRAM promotion (post-bufferization)",
        "module attributes {transform.with_named_sequence} {",
        "  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {",
    ]
    count = 0
    for entry in rewrite_plan:
        op_id = int(entry["op_id"])
        node = dag.nodes.get(op_id)
        if node is None:
            continue
        tag = entry.get("edgeslice_op_id", node.params.get("edgeslice_op_id", f"op_{op_id}"))
        op_name = _mlir_op_name(str(entry["op_type"]))
        operands = list(range(len(node.input_tensors) + 1))
        operand_text = ", ".join(str(index) for index in operands)
        lines.extend([
            f"    %sram_target_{op_id} = transform.structured.match",
            f'      ops{{["{op_name}"]}} attributes {{edgeslice.op_id = "{tag}"}} in %arg0',
            "      : (!transform.any_op) -> !transform.any_op",
            f"    %sram_op_{op_id} = transform.structured.promote %sram_target_{op_id}",
            "      {operands_to_promote = ["
            f"{operand_text}], use_full_tiles_by_default, "
            f"memory_space = {memory_space} : i64, alignment = {alignment}}}",
            "      : (!transform.any_op) -> !transform.any_op",
        ])
        count += 1
    if count == 0:
        return "// No SRAM promotion candidates.\n"
    lines.extend(["    transform.yield", "  }", "}"])
    return "\n".join(lines) + "\n"


def materialize_sram_placement(
    payload_path: str | Path,
    promotion_transform_path: str | Path,
    output_path: str | Path,
    *,
    capacity_bytes: int,
    memory_space: int = 1,
    double_buffer: bool = True,
    mlir_opt: Optional[str] = None,
) -> SRAMPlacementReport:
    """Bufferize, promote into SRAM memory space, and verify the artifact."""
    from edgeslide.mlir_transform.lowering import prepare_sliced_payload_for_lowering
    from edgeslide.mlir_transform.pipeline import run_transform_interpreter
    if capacity_bytes <= 0:
        raise ValueError("SRAM capacity must be positive")
    payload = require_input_file(payload_path, "sliced payload MLIR")
    transform = require_input_file(promotion_transform_path, "SRAM promotion transform")
    output = prepare_output_path(output_path, "SRAM-promoted payload MLIR")
    mlir_opt_bin = require_tool("mlir-opt", mlir_opt)
    tag = uuid.uuid4().hex[:8]
    prepared = output.parent / f"{output.stem}_prepared_{tag}.mlir"
    bufferized = output.parent / f"{output.stem}_bufferized_{tag}.mlir"

    # The first transform-interpreter run leaves its transform schedule next
    # to a nested payload. A second interpreter would execute both schedules;
    # strip the consumed one and unwrap the payload before bufferization.
    atomic_write_text(
        prepared,
        prepare_sliced_payload_for_lowering(
            payload.read_text(), emit_c_interface=False
        ),
        description="prepared SRAM payload",
    )

    with atomic_output(bufferized, description="bufferized SRAM payload") as temporary:
        run_tool(
            [
                mlir_opt_bin,
                str(prepared),
                "--one-shot-bufferize=bufferize-function-boundaries",
                "--canonicalize",
                "-o",
                str(temporary),
            ],
            stage="bufferize before SRAM promotion",
            error_cls=MLIRTransformError,
        )

    run_transform_interpreter(
        bufferized,
        transform,
        output,
        mlir_opt=mlir_opt_bin,
        verify_transformed=False,
    )
    text = output.read_text()
    # MLIR prints integer memory spaces as the final memref type parameter.
    promoted = len(re.findall(rf"memref\.alloc[^\n]*memref<[^\n>]*,\s*{memory_space}>", text))
    if promoted == 0:
        raise TransformVerificationError(
            f"SRAM promotion produced no memref.alloc in memory space {memory_space}"
        )
    copies = len(re.findall(r"(?m)^\s*(?:memref|linalg)\.copy\b", text))
    if copies == 0:
        raise TransformVerificationError(
            "SRAM promotion produced allocations but no copy-in/copy-out operations"
        )
    from edgeslide.liveness.analyzer import BufferizedMemoryAnalyzer

    memory_report = BufferizedMemoryAnalyzer(text).analyze()
    space_pattern = re.compile(rf",\s*{memory_space}\s*>")
    sram_allocations = [
        allocation
        for allocation in memory_report.allocations
        if space_pattern.search(allocation.memref_type)
    ]
    if not sram_allocations:
        raise TransformVerificationError(
            f"memory analyzer found no allocations in SRAM space {memory_space}"
        )
    events: Dict[int, int] = {}
    for allocation in sram_allocations:
        events[allocation.alloc_index] = events.get(allocation.alloc_index, 0) + allocation.size_bytes
        if allocation.dealloc_index is not None:
            events[allocation.dealloc_index] = (
                events.get(allocation.dealloc_index, 0) - allocation.size_bytes
            )
    live = 0
    peak = 0
    for _, delta in sorted(events.items()):
        live += delta
        peak = max(peak, live)
    usable_capacity = capacity_bytes // 2 if double_buffer else int(capacity_bytes * 0.90)
    if peak > usable_capacity:
        raise TransformVerificationError(
            f"materialized SRAM high-water mark {peak} bytes exceeds usable "
            f"capacity {usable_capacity} bytes"
        )
    return SRAMPlacementReport(
        output_path=str(output),
        bufferized_path=str(bufferized),
        promoted_allocation_count=promoted,
        copy_count=copies,
        memory_space=memory_space,
        capacity_bytes=capacity_bytes,
        double_buffer_reserved=double_buffer,
        # Capacity is conservatively split into two slots, but emitting a
        # pipelined DMA ping-pong loop is target-specific and is not invented
        # by this generic MLIR placement pass.
        double_buffer_materialized=False,
        sram_peak_bytes=peak,
        capacity_verified=True,
        inferred_size_count=sum(
            1 for allocation in sram_allocations if allocation.size_is_upper_bound
        ),
    )
