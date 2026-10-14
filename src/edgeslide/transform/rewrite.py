"""
extract_slice → scf.forall (tiled compute) → tensor.parallel_insert_slice

Implementation mapping:
  - ``transform.structured.match``        — locate payload op
  - ``transform.structured.tile_using_forall`` — steps 1+2 (slice + scf.forall)
  - The interpreter materializes ``tensor.parallel_insert_slice`` reassembly
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from edgeslide.dag.graph import OperationDAG
from edgeslide.mlir_transform.errors import InvalidTransformStyleError
from edgeslide.tiling.dimension_selector import (
    dimension_index_to_name,
    dimension_name_to_index,
)
from edgeslide.tiling.tile_size import HardwareConfig, TileSizeCalculator, node_cost_kwargs

# Emission styles accepted by :meth:`SlicedRewrite.generate_transform_ir`.
# Kept as a frozenset so an unknown style is rejected rather than silently
# treated as "structured": an ablation run labelled "aggressive" would
# otherwise report structured numbers.
TRANSFORM_STYLES: frozenset[str] = frozenset(
    {"structured", "legacy_for", "aggressive"}
)

# One entry of the slicing plan: the keys "op_id", "op_type", "slice_dimension",
# "tile_size", and "original_shape" are required; planners add optional cost
# annotations ("estimated_tile_working_set_bytes", "edgeslice_op_id", ...).
# A 2D tile adds "slice_dimension_2"/"tile_size_2" — both present or neither.
# Kept as an alias rather than a dataclass because the dict is passed through
# analysis, IR emission, and the JSON benchmark reports unchanged.
RewritePlan = Dict[str, Any]
RewritePlanList = List[RewritePlan]

# Ops eligible for EdgeSlice tiling per proposal §4.3–4.5:
#   {conv2d, matmul, elementwise} with rank >= 2 and static shapes.
TILEABLE_OP_TYPES: Set[str] = {
    # Conv / matmul family
    "conv_2d",
    "conv_2d_nhwc",
    "depthwise_conv",
    "matmul",
    "batch_matmul",
    "pooling",
    "linalg.conv_2d",
    "linalg.conv_2d_nchw_fchw",
    "linalg.conv_2d_nhwc_fhwc",
    "linalg.batch_matmul",
    "linalg.matmul",
    # Elementwise (proposal §4.4 is_tileable)
    "relu",
    "elementwise_add",
    "elementwise_mul",
    "linalg.generic",
    "linalg.elemwise_unary",
    "linalg.elemwise_binary",
}

# Intentionally excluded from tiling (not in proposal §4.4):
#   - Data movement: reshape, transpose, slice, concat (meta-ops)
#   - Global reductions: softmax, batch_norm, layer_norm
#   - Control / glue: constant, return, func, add (DAG sink labels)
#   - Rank-1 or scalar tensors (rank < 2)
EXCLUDED_OP_SUBSTRINGS: tuple[str, ...] = (
    "reshape",
    "transpose",
    "softmax",
    "batch_norm",
    "layer_norm",
    "constant",
    "return",
    "func.",
)


def is_tileable(
    op_type: str,
    shape: tuple | None = None,
    params: Dict[str, Any] | None = None,
) -> bool:
    """
    Return True when the op supports EdgeSlice tiling (proposal §4.4).

    Requires a supported op family and output rank >= 2 when *shape* is given.
    """
    if not op_type:
        return False

    lowered = op_type.lower()

    # The current transform path only supports concrete static iteration spaces.
    if shape is not None:
        if len(shape) < 2 or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            return False

    for excluded in EXCLUDED_OP_SUBSTRINGS:
        if excluded in lowered:
            return False

    # Bare "add" is a DAG glue label, not proposal elementwise tiling.
    if lowered == "add":
        return False

    # A generic op is safe only after the parser proves elementwise identity
    # indexing. Arbitrary linalg.generic may contain reductions/permutations.
    if lowered == "linalg.generic":
        return bool((params or {}).get("elementwise_identity"))

    if op_type in TILEABLE_OP_TYPES:
        return True

    if "conv" in lowered or "matmul" in lowered or "pool" in lowered:
        return True

    if "elementwise" in lowered or lowered == "relu":
        return True

    return False


# Option B: standard-op decomposition. Option A (custom C++ edgeslice op) deferred.
TRANSFORM_IMPLEMENTATION = "standard_ops_decomposition"

EDGESLICE_OP_ID_ATTR = "edgeslice.op_id"

# Producers that are always safe to pull into a tiled loop even though they are
# not tiling candidates themselves. Fusing the destination initializer is what
# actually removes the full-size intermediate buffer: fusing the compute alone
# leaves the original allocation live and raises peak memory instead.
ALWAYS_FUSABLE_OP_TYPES: Set[str] = {"linalg.fill"}

EDGESLICE_MEMORY_TIER_ATTR = "edgeslice.memory_tier"
EDGESLICE_WORKING_SET_ATTR = "edgeslice.working_set_bytes"
EDGESLICE_OUTPUT_BYTES_ATTR = "edgeslice.output_bytes"
# Names the anchor whose scf.forall absorbed this op. Its footprint is counted
# once, in that anchor's chain total — never add it again per producer.
EDGESLICE_FUSED_INTO_ATTR = "edgeslice.fused_into"

# Tier names recorded on payload ops. These are compiler *intent*, not physical
# placement: on a CPU target L1 is a hardware-managed cache, so nothing here
# moves data. A memref memory space would express placement, but allocating in a
# non-default space cannot lower on CPU (malloc/free are address-space 0), so an
# op attribute is used instead and lowering is unaffected.
MEMORY_TIER_SRAM = "sram"
MEMORY_TIER_DRAM = "dram"


def validate_transform_style(style: str) -> str:
    """Return *style* if it names a supported emission style, else raise."""
    if style not in TRANSFORM_STYLES:
        raise InvalidTransformStyleError(style, sorted(TRANSFORM_STYLES))
    return style


class SlicedRewrite:
    """
    Generate MLIR Transform Dialect IR for EdgeSlice tensor slicing.

    Emits interpreter-safe IR using only registered upstream transform ops.
    """

    def __init__(
        self,
        dag: OperationDAG,
        rewrite_plan: RewritePlanList,
        *,
        enable_fusion: bool = True,
        l1_cache_bytes: int = 512 * 1024,
        hardware_config: Optional[HardwareConfig] = None,
        annotate_memory_tier: bool = True,
        enforce_fusion_budget: bool = True,
    ):
        self.dag = dag
        self.enable_fusion = enable_fusion
        self.enforce_fusion_budget = enforce_fusion_budget
        self.hardware_config = hardware_config or HardwareConfig(
            l1_cache_bytes=l1_cache_bytes
        )
        self.tile_calculator = TileSizeCalculator(hardware_config=self.hardware_config)
        self.fast_memory_level, self.fast_memory_budget_bytes = (
            self.hardware_config.budget_levels()[0]
        )
        # Backward-compatible attribute for external readers. Internally the
        # usable budget, not physical SRAM capacity, governs every admission.
        self.l1_cache_bytes = self.fast_memory_budget_bytes
        self.annotate_memory_tier = annotate_memory_tier
        self._fused_op_ids: Set[int] = set()
        # Only emit transforms for tileable ops (proposal is_tileable gate).
        self.rewrite_plan = [
            plan
            for plan in rewrite_plan
            if is_tileable(
                plan.get("op_type", ""),
                tuple(plan.get("original_shape", ())),
                plan.get("params"),
            )
        ]
        self._producers: Dict[int, List[int]] = defaultdict(list)
        for producer_id, consumer_id in dag.edges:
            if producer_id in dag.nodes and consumer_id in dag.nodes:
                self._producers[consumer_id].append(producer_id)
        self._tileable_plans = list(self.rewrite_plan)
        self._plan_by_op_id: Dict[int, RewritePlan] = {
            plan["op_id"]: plan for plan in self._tileable_plans
        }
        self.absorbed_op_ids: Set[int] = set()
        self._anchor_ids: Set[int] = set()
        # anchor op_id -> producers fused into its scf.forall, nearest first.
        self.fusion_chains: Dict[int, List[int]] = {}
        self._plan_fusion()

    def _plan_fusion(self) -> None:
        """Split the tileable plans into tiled anchors and fused producers.

        Absorption and chain construction apply overlapping but not identical
        conditions, so this iterates until they agree. The invariant that has to
        hold when it stops: every absorbed op appears in exactly one anchor's
        chain. An absorbed op missing from every chain would be neither tiled
        nor fused — silently absent from the emitted script.

        Termination: each round can only move ops out of ``absorbed_op_ids``
        (back to being anchors), and the set is finite.
        """
        self.absorbed_op_ids = self._absorb_chained_anchors()

        for _ in range(len(self._tileable_plans) + 1):
            self.rewrite_plan = [
                plan
                for plan in self._tileable_plans
                if plan["op_id"] not in self.absorbed_op_ids
            ]
            self._anchor_ids = {plan["op_id"] for plan in self.rewrite_plan}
            self.fusion_chains = (
                {op_id: self._fusion_chain(op_id) for op_id in self._anchor_ids}
                if self.enable_fusion
                else {}
            )
            fused = {
                producer_id
                for chain in self.fusion_chains.values()
                for producer_id in chain
            }
            orphans = self.absorbed_op_ids - fused
            if not orphans:
                return
            # Restore orphans as anchors of their own and re-derive the chains:
            # an orphan promoted to anchor can pull its own producers in.
            self.absorbed_op_ids -= orphans

        raise RuntimeError(  # pragma: no cover - guarded by the loop bound
            "fusion planning did not converge; absorbed ops without a fusion "
            f"chain: {sorted(self.absorbed_op_ids)}"
        )

    def _absorb_chained_anchors(self) -> Set[int]:
        """Drop anchors that feed another anchor so they can be fused instead.

        Two candidates in a producer/consumer chain would otherwise be tiled into
        two separate loops, and the intermediate between them stays full size.
        Tiling only the last op of the chain and fusing the rest upstream is what
        collapses that intermediate down to a tile.

        An anchor is absorbed only when it would actually be fusable. Absorbing
        an op that ``_emit_fusion`` later declines to emit would drop it from the
        script entirely: neither tiled nor fused, and silently so.
        """
        if not self.enable_fusion:
            return set()
        planned = {plan["op_id"] for plan in self._tileable_plans}
        absorbed = set()
        for op_id in planned:
            node = self.dag.nodes.get(op_id)
            if node is None or len(node.consumers) != 1:
                continue
            if node.consumers[0] not in planned:
                continue
            if not self._is_fusable_node(op_id):
                continue
            absorbed.add(op_id)
        return absorbed

    def generate_transform_ir(self, style: str = "structured") -> str:
        """
        Generate Transform Dialect IR.

        Args:
            style: "structured" (proposal edgeslice + parallel_insert_slice, default),
                   "legacy_for" (tile_using_for), or
                   "aggressive" (verbose decomposed comments).

        Raises:
            InvalidTransformStyleError: *style* is not a known emission style.
        """
        # Validated before the empty-plan shortcut below, so a typo'd style is
        # caught even for a DAG with no slicing candidates.
        validate_transform_style(style)

        if not self.rewrite_plan:
            return "// No slicing candidates found — no transforms generated."

        lines = [
            "// EdgeSlice Transform Script (Option B: standard ops decomposition)",
            "// Architecture: docs/EDGESLICE_TRANSFORM_ARCHITECTURE.md",
            "// Pattern: extract_slice → scf.forall → tensor.parallel_insert_slice (§4.5)",
            "// Executable: match + tile_using_forall",
            "",
            "module attributes {transform.with_named_sequence} {",
        ]

        lines.extend([
            "  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {",
            "",
        ])

        for plan in self.rewrite_plan:
            if style == "aggressive":
                lines.extend(self._emit_aggressive_slicing(plan))
            elif style == "legacy_for":
                lines.extend(self._emit_legacy_tile_for(plan))
            else:
                lines.extend(self._emit_edgeslice_transform(plan))

        if self.annotate_memory_tier and style == "structured":
            lines.extend(self._emit_dram_annotations())

        lines.extend([
            "    transform.yield",
            "  }",
            "}",
        ])
        return "\n".join(lines)

    def _emit_edgeslice_transform(self, plan: RewritePlan) -> List[str]:
        """
        Proposal §4.5 rewrite via Option B (standard ops).

        Executable: ``tile_using_forall`` (extract_slice + scf.forall + parallel_insert).
        ``mlir-opt --transform-interpreter`` materializes all three payload steps.
        """
        op_id = plan["op_id"]
        op_type = plan["op_type"]
        dim = plan["slice_dimension"]
        tile_size = plan["tile_size"]
        shape = plan["original_shape"]

        mlir_op = self._op_type_to_mlir(op_type)
        edgeslice_tag = plan.get("edgeslice_op_id", f"op_{op_id}")
        dim_idx = self._dim_name_to_index(dim, len(shape), op_type)
        effective_tile = min(tile_size, shape[dim_idx]) if dim_idx < len(shape) else tile_size
        secondary = self._plan_secondary(plan)
        tile_sizes = self._build_tile_sizes(
            shape, dim, effective_tile, op_type, secondary
        )
        tile_sizes_str = ", ".join(str(s) for s in tile_sizes)
        slice_desc = f"slice dim '{dim}' (index {dim_idx}), tile={effective_tile}"
        if secondary is not None:
            sec_idx = self._dim_name_to_index(secondary[0], len(shape), op_type)
            slice_desc += (
                f" x dim '{secondary[0]}' (index {sec_idx}), tile={secondary[1]}"
            )
        lines = [
            f"    // Op {op_id}: {op_type}  shape={shape}",
            f"    // EdgeSlice: {slice_desc}",
            f'    %target_{op_id} = transform.structured.match ops{{["{mlir_op}"]}}'
            f' attributes {{{EDGESLICE_OP_ID_ATTR} = "{edgeslice_tag}"}} in %arg0',
            "      : (!transform.any_op) -> !transform.any_op",
            "",
            "    // Step 1+2: extract_slice + scf.forall (tile_using_forall)",
            f"    %tiled_{op_id}, %loop_{op_id} = transform.structured.tile_using_forall",
            f"      %target_{op_id} tile_sizes [{tile_sizes_str}]",
            "      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)",
            "",
            "    // Step 3: scf.forall.in_parallel reassembles tiles via tensor.parallel_insert_slice",
            f"    // Payload writes each tile into the destination tensor along dim {dim_idx}",
            "",
        ]
        if self.annotate_memory_tier:
            lines.extend(self._emit_tier_annotation(plan))
        if self.enable_fusion:
            lines.extend(self._emit_fusion(op_id))
        return lines

    def _is_fusable_node(self, op_id: int) -> bool:
        """Whether this op can be pulled into a consumer's loop at all.

        Multi-consumer producers are rejected: fusing them into one loop forces
        the value to be recomputed for every other consumer. A missing
        ``edgeslice_op_id`` means ``transform.structured.match`` has no way to
        address the op unambiguously, so it cannot be fused either.
        """
        node = self.dag.nodes.get(op_id)
        if node is None or not node.has_single_output:
            return False
        if len(node.consumers) != 1:
            return False
        if "edgeslice_op_id" not in node.params:
            return False
        if node.op_type in ALWAYS_FUSABLE_OP_TYPES:
            return True
        return is_tileable(node.op_type, tuple(node.output_tensor.shape), node.params)

    def _is_fusable_producer(self, op_id: int) -> bool:
        """``_is_fusable_node`` plus the anchor exclusion.

        Anchors are rejected because ``tile_using_forall`` already consumed their
        transform handle and rewrote the payload op.
        """
        if op_id in self._anchor_ids:
            return False
        return self._is_fusable_node(op_id)

    def _fusion_chain(self, anchor_op_id: int) -> List[int]:
        """Producers to fuse into the anchor's scf.forall, nearest first.

        Breadth-first over the reversed DAG, so each producer is emitted only
        after the op that consumes it is already inside the loop.

        Producers of a different output rank are skipped. The slicing plan names
        one dimension ("spatial_h", "batch", ...) and resolves it per rank, so a
        rank change means the anchor's tiled dimension does not name the same
        axis in the producer — the tile-budget model would then be describing a
        different slice than the one the loop actually takes.

        A producer is admitted only while the fused body still fits the cache
        budget (proposal H1). Fusing past that point is not a smaller working
        set that merely overflows: a conv pulled into a channel-tiled loop
        needs its *whole* input for every tile, because the reduction runs over
        all input channels, so the loop re-streams that input once per tile.
        A rejected producer stays outside the loop and becomes an anchor of its
        own (see ``_plan_fusion``), which is the honest outcome — it was never
        going to be cache-resident in there.
        """
        anchor = self.dag.nodes.get(anchor_op_id)
        anchor_rank = len(anchor.output_tensor.shape) if anchor is not None else None
        admit = self._fusion_budget_admitter(anchor_op_id)

        chain: List[int] = []
        seen = {anchor_op_id}
        frontier = [anchor_op_id]
        while frontier:
            current = frontier.pop(0)
            for producer_id in self._producers.get(current, []):
                if producer_id in seen:
                    continue
                seen.add(producer_id)
                if not self._is_fusable_producer(producer_id):
                    continue
                producer = self.dag.nodes[producer_id]
                if anchor_rank is not None and (
                    len(producer.output_tensor.shape) != anchor_rank
                ):
                    continue
                if not admit(producer_id):
                    # Do not walk past a rejected producer: its own producers
                    # would then be fused into a loop that no longer contains
                    # the op consuming them.
                    continue
                chain.append(producer_id)
                frontier.append(producer_id)
        return chain

    def _fusion_budget_admitter(self, anchor_op_id: int) -> Callable[[int], bool]:
        """Predicate admitting producers while the fused body can still fit.

        Measured at a *unit* primary tile — the same check
        ``compute_for_group`` makes before re-fitting the anchor's tile to the
        fused body. Testing the anchor's planned tile instead would reject
        every producer, because that tile is sized to fill the budget on its
        own; what matters here is whether any tile of the resulting group is
        cache-resident. Returns a stateful closure: each accepted producer adds
        its tile to the running total.
        """
        if not self.enforce_fusion_budget:
            return lambda _producer_id: True

        plan = self._plan_by_op_id.get(anchor_op_id)
        anchor = self.dag.nodes.get(anchor_op_id)
        if plan is None or anchor is None:
            return lambda _producer_id: True

        calc = self.tile_calculator
        budget = calc.hardware_config.l1_budget_bytes
        slice_dim = plan["slice_dimension"]
        secondary = self._plan_secondary(plan)
        anchor_rank = len(anchor.output_tensor.shape)

        def dim_in(node, dim_name: str) -> str:
            """The anchor's tiled axis, named in *node*'s own layout.

            The loop tiles an index of the anchor's iteration space; producers
            are fused into that same loop, so the axis is positional. Matching
            by *name* instead drops every producer whose op type does not use
            the anchor's vocabulary — a rank-2 ``linalg.generic`` feeding a
            matmul anchor knows "dim_0", not "M", so the whole
            elementwise→matmul family lost fusion silently. Ranks are equal
            here: ``_fusion_chain`` filters mismatches before admission.
            """
            index = dimension_name_to_index(dim_name, anchor.op_type, anchor_rank)
            return dimension_index_to_name(
                index, node.op_type, len(node.output_tensor.shape)
            )

        def member_bytes(node) -> int:
            node_secondary = (
                (dim_in(node, secondary[0]), secondary[1])
                if secondary is not None
                else None
            )
            return calc.estimate_tile_footprint(
                tuple(node.output_tensor.shape),
                dim_in(node, slice_dim),
                1,
                secondary=node_secondary,
                **node_cost_kwargs(node),
            ).total_bytes

        try:
            running = member_bytes(anchor)
        except ValueError:
            # The anchor's dimension does not resolve for its own layout;
            # leave admission to the (unbudgeted) structural rules.
            return lambda _producer_id: True

        def admit(producer_id: int) -> bool:
            nonlocal running
            node = self.dag.nodes.get(producer_id)
            if node is None:
                return False
            try:
                cost = member_bytes(node)
            except ValueError:
                # The anchor's slice dimension has no meaning for this
                # producer's layout, so its tile is not the one the loop takes.
                return False
            if running + cost > budget:
                return False
            running += cost
            return True

        return admit

    def _emit_fusion(self, anchor_op_id: int) -> List[str]:
        """Emit fuse_into_containing_op for the anchor's producer chain."""
        chain = self.fusion_chains.get(anchor_op_id) or self._fusion_chain(anchor_op_id)
        if not chain:
            return []

        lines = [
            f"    // Step 4: fuse {len(chain)} single-consumer producer(s) into "
            f"%loop_{anchor_op_id}",
            "    // Without this the full-size intermediate stays allocated and "
            "peak memory does not drop.",
        ]
        loop_handle = f"%loop_{anchor_op_id}"
        for index, producer_id in enumerate(chain, start=1):
            node = self.dag.nodes[producer_id]
            producer_op = self._op_type_to_mlir(node.op_type)
            producer_tag = node.params.get("edgeslice_op_id", f"op_{producer_id}")
            next_loop = f"%loop_{anchor_op_id}_{index}"
            lines.extend([
                f"    %prod_{anchor_op_id}_{producer_id} = transform.structured.match"
                f' ops{{["{producer_op}"]}}'
                f' attributes {{{EDGESLICE_OP_ID_ATTR} = "{producer_tag}"}} in %arg0',
                "      : (!transform.any_op) -> !transform.any_op",
                f"    %fused_{anchor_op_id}_{producer_id}, {next_loop} ="
                f" transform.structured.fuse_into_containing_op",
                f"      %prod_{anchor_op_id}_{producer_id} into {loop_handle}",
                "      : (!transform.any_op, !transform.any_op)"
                " -> (!transform.any_op, !transform.any_op)",
            ])
            self._fused_op_ids.add(producer_id)
            if self.annotate_memory_tier:
                handle = f"%fused_{anchor_op_id}_{producer_id}"
                lines.extend(self._emit_annotation(
                    handle,
                    f"ftier_{anchor_op_id}_{producer_id}",
                    EDGESLICE_MEMORY_TIER_ATTR,
                    MEMORY_TIER_SRAM,
                ))
                anchor_node = self.dag.nodes.get(anchor_op_id)
                anchor_tag = (
                    anchor_node.params.get("edgeslice_op_id", f"op_{anchor_op_id}")
                    if anchor_node is not None
                    else f"op_{anchor_op_id}"
                )
                lines.extend(self._emit_annotation(
                    handle,
                    f"finto_{anchor_op_id}_{producer_id}",
                    EDGESLICE_FUSED_INTO_ATTR,
                    anchor_tag,
                ))
            loop_handle = next_loop
        lines.append("")
        return lines

    @staticmethod
    def _emit_annotation(
        handle: str, suffix: str, attr_name: str, value: Any
    ) -> List[str]:
        """Attach one attribute to a payload op via transform.annotate."""
        if isinstance(value, str):
            literal = f'"{value}" -> !transform.any_param'
        else:
            literal = f"{int(value)} : i64 -> !transform.any_param"
        param = f"%p_{suffix}"
        return [
            f"    {param} = transform.param.constant {literal}",
            f'    transform.annotate {handle} "{attr_name}" = {param}'
            f" : !transform.any_op, !transform.any_param",
        ]

    def _emit_tier_annotation(self, plan: RewritePlan) -> List[str]:
        """Record that the tiled op fits the configured fast-memory budget."""
        op_id = plan["op_id"]
        # With fusion the loop holds the anchor tile *and* every fused
        # producer's tile, so the chain total is the honest number. It is filled
        # in by EdgeSlicePassManager._annotate_fusion before IR generation.
        working_set = plan.get("estimated_fused_working_set_bytes")
        if working_set is None:
            working_set = plan.get("estimated_tile_working_set_bytes")
        lines = [
            f"    // Memory tier: tile working set {working_set} B vs "
            f"{self.fast_memory_level} budget {self.fast_memory_budget_bytes} B",
        ]
        lines += self._emit_annotation(
            f"%tiled_{op_id}", f"tier_{op_id}",
            EDGESLICE_MEMORY_TIER_ATTR, MEMORY_TIER_SRAM,
        )
        if working_set is not None:
            lines += self._emit_annotation(
                f"%tiled_{op_id}", f"ws_{op_id}",
                EDGESLICE_WORKING_SET_ATTR, working_set,
            )
        lines.append("")
        return lines

    def _output_bytes(self, op_id: int) -> Optional[int]:
        node = self.dag.nodes.get(op_id)
        if node is None or not node.has_single_output:
            return None
        try:
            return node.output_tensor.size_bytes
        except (ValueError, TypeError):
            # Dynamic or unsupported dtype: no trustworthy byte count.
            return None

    def _emit_dram_annotations(self) -> List[str]:
        """Tag untiled ops whose output cannot fit fast memory as DRAM-resident."""
        handled = self._anchor_ids | self._fused_op_ids
        lines: List[str] = []
        for op_id in sorted(self.dag.nodes):
            if op_id in handled:
                continue
            node = self.dag.nodes[op_id]
            if "edgeslice_op_id" not in node.params:
                continue
            size = self._output_bytes(op_id)
            if size is None or size <= self.fast_memory_budget_bytes:
                continue
            mlir_op = self._op_type_to_mlir(node.op_type)
            tag = node.params["edgeslice_op_id"]
            handle = f"%dram_{op_id}"
            lines.extend([
                f"    // Op {op_id}: {node.op_type} output {size} B exceeds "
                f"{self.fast_memory_level} budget "
                f"and is not tiled",
                f"    {handle} = transform.structured.match"
                f' ops{{["{mlir_op}"]}}'
                f' attributes {{{EDGESLICE_OP_ID_ATTR} = "{tag}"}} in %arg0',
                "      : (!transform.any_op) -> !transform.any_op",
            ])
            lines += self._emit_annotation(
                handle, f"dtier_{op_id}",
                EDGESLICE_MEMORY_TIER_ATTR, MEMORY_TIER_DRAM,
            )
            lines += self._emit_annotation(
                handle, f"dsz_{op_id}", EDGESLICE_OUTPUT_BYTES_ATTR, size
            )
            lines.append("")
        if lines:
            lines.insert(0, "    // === Memory tier: untiled ops that stay in DRAM ===")
        return lines

    def _emit_legacy_tile_for(self, plan: RewritePlan) -> List[str]:
        """Legacy path: tile_using_for."""
        op_id = plan["op_id"]
        op_type = plan["op_type"]
        dim = plan["slice_dimension"]
        tile_size = plan["tile_size"]
        shape = plan["original_shape"]

        mlir_op = self._op_type_to_mlir(op_type)
        edgeslice_tag = plan.get("edgeslice_op_id", f"op_{op_id}")
        tile_sizes_str = ", ".join(
            str(s)
            for s in self._build_tile_sizes(
                shape, dim, tile_size, op_type, self._plan_secondary(plan)
            )
        )

        return [
            f"    // Op {op_id}: {op_type}  shape={shape} (legacy tile_using_for)",
            f'    %target_{op_id} = transform.structured.match ops{{["{mlir_op}"]}}'
            f' attributes {{{EDGESLICE_OP_ID_ATTR} = "{edgeslice_tag}"}} in %arg0',
            "      : (!transform.any_op) -> !transform.any_op",
            f"    %tiled_{op_id}, %loop_{op_id} = transform.structured.tile_using_for",
            f"      %target_{op_id} tile_sizes [{tile_sizes_str}]",
            "      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)",
            "",
        ]

    def _emit_aggressive_slicing(self, plan: RewritePlan) -> List[str]:
        """Verbose decomposed edgeslice pattern with parallel-insert references."""
        op_id = plan["op_id"]
        op_type = plan["op_type"]
        dim = plan["slice_dimension"]
        tile_size = plan["tile_size"]
        shape = plan["original_shape"]

        dim_idx = self._dim_name_to_index(dim, len(shape), op_type)
        mlir_op = self._op_type_to_mlir(op_type)
        edgeslice_tag = plan.get("edgeslice_op_id", f"op_{op_id}")
        secondary = self._plan_secondary(plan)
        tile_sizes_str = ", ".join(
            str(s)
            for s in self._build_tile_sizes(shape, dim, tile_size, op_type, secondary)
        )
        dim_desc = f"{dim} (index {dim_idx}), tile_size = {tile_size}"
        if secondary is not None:
            dim_desc += (
                f" x {secondary[0]} "
                f"(index {self._dim_name_to_index(secondary[0], len(shape), op_type)}), "
                f"tile_size = {secondary[1]}"
            )

        return [
            f"    // Op {op_id}: {op_type}  shape={shape} (aggressive / fully decomposed)",
            f"    // Slicing dimension: {dim_desc}",
            "",
            f'    %target_{op_id} = transform.structured.match ops{{["{mlir_op}"]}}'
            f' attributes {{{EDGESLICE_OP_ID_ATTR} = "{edgeslice_tag}"}} in %arg0',
            "      : (!transform.any_op) -> !transform.any_op",
            "",
            f"    %tiled_{op_id}, %loop_{op_id} = transform.structured.tile_using_forall",
            f"      %target_{op_id} tile_sizes [{tile_sizes_str}]",
            "      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)",
            f"    // scf.forall.in_parallel inserts tile results along dimension {dim_idx}",
            "    // using tensor.parallel_insert_slice",
            "",
        ]

    def _op_type_to_mlir(self, op_type: str) -> str:
        """Map DAG / frontend op_type to an MLIR op name for transform.match.

        Critical: already-qualified names like ``linalg.conv_2d_nchw_fchw`` must
        pass through unchanged. Falling back to ``linalg.generic`` silently
        prevents the transform interpreter from matching real conv ops
        (observed on TinyConv hw_profile artifacts).
        """
        if not op_type:
            return "linalg.generic"

        lowered = op_type.strip()
        # Already a dialect-qualified MLIR op name from DAGParser.
        if "." in lowered and not lowered.startswith("torch."):
            # e.g. linalg.conv_2d_nchw_fchw, linalg.generic, linalg.matmul
            return lowered

        mapping = {
            "conv_2d": "linalg.conv_2d_nchw_fchw",
            "conv_2d_nhwc": "linalg.conv_2d_nhwc_fhwc",
            "depthwise_conv": "linalg.depthwise_conv_2d_nchw_chw",
            "matmul": "linalg.matmul",
            "batch_matmul": "linalg.batch_matmul",
            "elementwise_add": "linalg.add",
            "elementwise_mul": "linalg.mul",
            "pooling": "linalg.pooling_nchw_max",
            "relu": "linalg.generic",
        }
        if lowered in mapping:
            return mapping[lowered]

        # Heuristic fallback for free-form labels (never map conv → generic).
        low = lowered.lower()
        if "conv_2d_nhwc" in low or ("nhwc" in low and "conv" in low):
            return "linalg.conv_2d_nhwc_fhwc"
        if "conv" in low:
            return "linalg.conv_2d_nchw_fchw"
        if "batch_matmul" in low:
            return "linalg.batch_matmul"
        if "matmul" in low:
            return "linalg.matmul"
        if "pool" in low:
            return "linalg.pooling_nchw_max"
        return "linalg.generic"

    def _build_tile_sizes(
        self,
        shape: tuple,
        dim: str,
        tile_size: int,
        op_type: str,
        secondary: Optional[Tuple[str, int]] = None,
    ) -> List[int]:
        """Per-loop tile-size vector for ``tile_using_forall``.

        A 2D plan simply sets two nonzero entries — the transform op tiles
        every dimension with a nonzero size in one ``scf.forall``.
        """
        sizes = [0] * len(shape)
        members = [(dim, tile_size)]
        if secondary is not None:
            members.append(secondary)
        for dim_name, dim_tile in members:
            idx = self._dim_name_to_index(dim_name, len(shape), op_type)
            if 0 <= idx < len(shape):
                sizes[idx] = min(dim_tile, shape[idx])
        return sizes

    @staticmethod
    def _plan_secondary(plan: RewritePlan) -> Optional[Tuple[str, int]]:
        """The plan's secondary (dim, tile) pair, or None for 1D plans."""
        dim2 = plan.get("slice_dimension_2")
        if not dim2:
            return None
        return (str(dim2), int(plan["tile_size_2"]))

    def _dim_name_to_index(self, dim: str, rank: int, op_type: str) -> int:
        return dimension_name_to_index(dim, op_type, rank)

    @staticmethod
    def _shape_to_mlir_type(shape: tuple, elem_type: str = "f32") -> str:
        dims = "x".join(str(d) for d in shape)
        return f"tensor<{dims}x{elem_type}>"
