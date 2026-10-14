"""EdgeSlicePassManager — Proposal-aligned frontend compiler pipeline.

Orchestrates:
  1. Liveness + Peak Memory analysis
  2. Dimension selection + tile size computation
  3. Generation of MLIR Transform Dialect IR (structured tiling by default)

Transform emission uses Option B (standard ops decomposition); see
docs/EDGESLICE_TRANSFORM_ARCHITECTURE.md for the custom-op decision record.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from edgeslide.dag.graph import OperationDAG
from edgeslide.liveness.peak_memory import PeakMemoryEstimator
from edgeslide.mlir_transform.errors import EdgeSlideError
from edgeslide.mlir_transform.sram import generate_sram_promotion_transform
from edgeslide.tiling.dimension_selector import (
    DimensionSelector,
    dimension_index_to_name,
    dimension_name_to_index,
)
from edgeslide.tiling.tile_size import (
    HardwareConfig,
    TileBudgetError,
    TileDecision,
    TileSizeCalculator,
    node_cost_kwargs,
)
from edgeslide.tiling.schedule_search import (
    CapacityScheduleSearch,
    ScheduleCandidate,
    SearchResult,
)
from edgeslide.transform.rewrite import (
    RewritePlan,
    RewritePlanList,
    SlicedRewrite,
    is_tileable,
    validate_transform_style,
)

# Everything EdgeSlicePassManager.run() publishes, keyed by stage name
# ("liveness", "rewrite_plan", "transform_ir", "compile_timing", ...).
PassManagerResults = Dict[str, Any]


@dataclass
class EdgeSlicePassConfig:
    """Frontend pass configuration with ablation-study toggles (proposal §5.4)."""

    l1_cache_bytes: int = 512 * 1024
    max_tile_size: Optional[int] = None
    # Full EdgeSlice (C): both True.  Ablation A: liveness on, heuristic off.
    # Ablation B: liveness off, heuristic on.
    enable_liveness_analysis: bool = True
    enable_smart_heuristic: bool = True
    uniform_tile_size: int = 64
    uniform_slice_dimension: str = "spatial_h"
    # Ablation D: emit tiling without producer fusion. Tiling alone leaves the
    # full-size intermediate allocated, so peak memory is unchanged.
    enable_fusion: bool = True
    # Stamp edgeslice.memory_tier on payload ops so the tier decision is
    # inspectable in the IR. Annotation only: it does not place data.
    annotate_memory_tier: bool = True
    # Ablation E: when the preferred slice dimension has no tile that fits the
    # budget, try the next-preferred instead of dropping the op. Without this,
    # batch>1 workloads plan zero tiles (a unit batch tile is a whole sample).
    enable_dimension_cascade: bool = True
    # When every single dimension is rejected (unit tile alone overflows the
    # budget — e.g. one row of a 256x256x256 map is a whole 256 KB L1), try
    # dimension pairs so the op can still get a cache-resident 2D tile.
    enable_2d_cascade: bool = True
    # Ablation F: admit producers into an anchor's scf.forall only while the
    # fused body can still be made cache-resident (proposal H1). Turning this
    # off restores unbounded greedy fusion, which fuses more per loop but
    # emits groups whose working set exceeds the budget — reported in
    # ``fusion_budget_warnings``.
    enable_fusion_budget: bool = True
    # Full cache geometry for the tile cost model. When set it takes precedence
    # over ``l1_cache_bytes``, which is a single-level budget knob and not an
    # L1 size any shipping edge CPU has (see tile_size.EDGE_CACHE_LEVELS).
    hardware_config: Optional[HardwareConfig] = None
    # The production objective is local fast-memory residency. Peak-memory
    # pressure remains available as a legacy research ablation, but it is not a
    # sound candidate gate for SRAM performance: a costly sink or an op away
    # from the global liveness peak may still spend most of its time in DRAM.
    planning_objective: str = "sram_residency"
    # ``search`` enumerates legal 1D/2D tile shapes and ranks them by an
    # auditable traffic/locality objective. ``cascade`` keeps the historical
    # first-feasible-dimension heuristic as an ablation.
    planning_strategy: str = "search"
    schedule_search_max_candidates: int = 4096
    schedule_shortlist_size: int = 8
    schedule_dataflows: Tuple[str, ...] = ("output_stationary",)
    # Optional real-target measurement of a materializable schedule. The
    # callback is invoked only for the analytical shortlist; TargetAutotuner
    # supplies repeated trials and persistent caching when desired.
    target_measure: Optional[Callable[[ScheduleCandidate], float]] = None
    materialize_sram: bool = False
    sram_memory_space: int = 1
    sram_alignment_bytes: int = 64

    def tile_calculator(self) -> TileSizeCalculator:
        """One construction point, so every path costs tiles identically."""
        if self.hardware_config is not None:
            return TileSizeCalculator(
                hardware_config=self.hardware_config,
                max_tile_size=self.max_tile_size,
            )
        return TileSizeCalculator(self.l1_cache_bytes, self.max_tile_size)

    @classmethod
    def for_sram(
        cls,
        capacity_bytes: int,
        *,
        double_buffer: bool = True,
        **overrides: Any,
    ) -> "EdgeSlicePassConfig":
        """Configuration for an explicitly managed on-chip SRAM target."""
        materialize = bool(overrides.pop("materialize_sram", True))
        memory_space = int(overrides.pop("sram_memory_space", 1))
        alignment = int(overrides.pop("sram_alignment_bytes", 64))
        return cls(
            l1_cache_bytes=capacity_bytes,
            hardware_config=HardwareConfig.for_sram(
                capacity_bytes, double_buffer=double_buffer
            ),
            planning_objective="sram_residency",
            materialize_sram=materialize,
            sram_memory_space=memory_space,
            sram_alignment_bytes=alignment,
            **overrides,
        )


class UncostablePlanEntry(EdgeSlideError, ValueError):
    """A rewrite-plan entry whose tile footprint cannot be computed.

    Raised rather than defaulted to zero. Proposal H1 is checked by comparing
    a plan's working set against the cache budget, and a zero-byte working set
    satisfies every budget — so defaulting turned "we cannot tell whether this
    fits" into "this fits", silently.
    """


def plan_with_fusion(
    dag: OperationDAG,
    rewrite_plan: List["RewritePlan"],
    config: EdgeSlicePassConfig,
) -> Tuple[List["RewritePlan"], List[Dict[str, Any]]]:
    """Put an externally-built plan through EdgeSlice's own fusion accounting.

    Baseline strategies build their rewrite plans with a different planner —
    per-op spatial-halo tiling, say — and used to be evaluated straight from
    that plan. Because the plan carried no ``fused_producer_op_ids``, the
    peak-memory estimator could not give the baseline any fusion credit,
    while EdgeSlice's own plan got it. The two arms were therefore measured
    under different memory models, and the difference between them included
    that asymmetry rather than only the difference in tiling decisions.

    Running every arm through this function removes that: each plan is passed
    through the same ``SlicedRewrite`` chain construction and the same fused
    working-set re-fit, so whatever fusion a plan's own tiling decisions make
    available is credited to it. What remains between arms is the tiling
    decision, which is the thing under comparison.

    Returns (final plan, fusion budget warnings).
    """
    manager = EdgeSlicePassManager(dag, config=config)
    prepared = [
        _with_tile_working_set(dag, dict(entry), config) for entry in rewrite_plan
    ]
    rewriter = SlicedRewrite(
        dag,
        prepared,
        enable_fusion=config.enable_fusion,
        l1_cache_bytes=config.l1_cache_bytes,
        hardware_config=config.hardware_config,
        annotate_memory_tier=config.annotate_memory_tier,
        enforce_fusion_budget=config.enable_fusion_budget,
    )
    manager._annotate_fusion(rewriter)
    return rewriter.rewrite_plan, manager.results.get("fusion_budget_warnings", [])


def _with_tile_working_set(
    dag: OperationDAG,
    entry: Dict[str, Any],
    config: EdgeSlicePassConfig,
) -> Dict[str, Any]:
    """Backfill the cost fields a plan built outside the pass manager lacks.

    An external planner records what it decided — dimension, tile size, shape —
    but not what that tile costs, and the fused re-fit needs the cost. Deriving
    it here with the same ``TileSizeCalculator`` the pass manager uses is what
    keeps the arms on one cost model; letting each planner supply its own
    number is how they drifted apart in the first place.
    """
    if "estimated_tile_working_set_bytes" in entry:
        return entry
    op_id = entry.get("op_id")
    node = dag.nodes.get(op_id) if isinstance(op_id, int) else None
    if node is None:
        # A plan naming an op the DAG does not contain is malformed. Filling in
        # a zero made it report a 0-byte working set, which trivially satisfies
        # proposal H1 ("every accepted tile plan fits the analytical cache
        # budget") — so an uncostable entry passed the very check it should
        # have failed, and did so without a warning.
        raise UncostablePlanEntry(
            f"plan entry names op_id {op_id!r}, which is not in the DAG, so "
            "its tile cannot be costed. A plan whose working set is unknown "
            "cannot be checked against the cache budget."
        )

    calc = config.tile_calculator()
    secondary = None
    if entry.get("slice_dimension_2"):
        secondary = (entry["slice_dimension_2"], int(entry["tile_size_2"]))
    try:
        footprint = calc.estimate_tile_footprint(
            tuple(entry["original_shape"]),
            entry["slice_dimension"],
            int(entry["tile_size"]),
            secondary=secondary,
            **node_cost_kwargs(node),
        )
    except (ValueError, KeyError) as exc:
        # Zero is not conservative here, it is the opposite: a zero-byte
        # working set fits every budget, so an entry whose dimension does not
        # map onto this op's layout would be admitted rather than rejected.
        raise UncostablePlanEntry(
            f"plan entry for op {op_id} slices {entry.get('slice_dimension')!r}, "
            f"which does not map onto {node.op_type} of rank "
            f"{len(entry.get('original_shape') or ())}: {exc}"
        ) from exc
    entry["estimated_tile_working_set_bytes"] = footprint.total_bytes
    entry.setdefault("params", node.params)
    entry.setdefault(
        "edgeslice_op_id", node.params.get("edgeslice_op_id", f"op_{op_id}")
    )
    return entry


class EdgeSlicePassManager:
    """Main class for the EdgeSlide frontend compiler analysis + IR generation."""

    def __init__(
        self,
        dag: OperationDAG,
        l1_cache_bytes: int = 512 * 1024,
        max_tile_size: Optional[int] = None,
        *,
        config: Optional[EdgeSlicePassConfig] = None,
        enable_liveness_analysis: Optional[bool] = None,
        enable_smart_heuristic: Optional[bool] = None,
        uniform_tile_size: int = 64,
        uniform_slice_dimension: str = "spatial_h",
    ):
        if config is not None:
            self.config = config
        else:
            self.config = EdgeSlicePassConfig(
                l1_cache_bytes=l1_cache_bytes,
                max_tile_size=max_tile_size,
                enable_liveness_analysis=(
                    True if enable_liveness_analysis is None else enable_liveness_analysis
                ),
                enable_smart_heuristic=(
                    True if enable_smart_heuristic is None else enable_smart_heuristic
                ),
                uniform_tile_size=uniform_tile_size,
                uniform_slice_dimension=uniform_slice_dimension,
            )
        self.dag = dag
        self.results: PassManagerResults = {}
        self._planning_rejections: List[RewritePlan] = []

    @property
    def l1_cache_bytes(self) -> int:
        return self.config.l1_cache_bytes

    def run(self, transform_style: str = "structured") -> PassManagerResults:
        """
        Run the proposal pipeline with optional ablation flags.

        Args:
            transform_style: "structured" (default), "legacy_for", or "aggressive"

        Raises:
            InvalidTransformStyleError: *transform_style* is not a known style.
        """
        # Rejected up front rather than at emission time: the liveness and
        # planning work below is wasted if the style was a typo, and the
        # results dict would still be stamped with the unusable style.
        validate_transform_style(transform_style)
        if self.config.planning_objective not in {"sram_residency", "peak_memory"}:
            raise ValueError(
                "planning_objective must be 'sram_residency' or 'peak_memory', "
                f"got {self.config.planning_objective!r}"
            )
        if self.config.planning_strategy not in {"search", "cascade"}:
            raise ValueError(
                "planning_strategy must be 'search' or 'cascade', got "
                f"{self.config.planning_strategy!r}"
            )

        t0 = time.perf_counter()

        # 1. Liveness + peak memory (current materialized live sets: O(V*K))
        t_liveness = time.perf_counter()
        estimator = PeakMemoryEstimator(self.dag, self.config.l1_cache_bytes)
        memory_at = estimator.estimate()
        peak_op, peak_mem = estimator.get_peak()
        liveness_ms = (time.perf_counter() - t_liveness) * 1000

        if self.config.planning_objective == "sram_residency":
            candidates = self._sram_residency_candidates(memory_at)
        elif self.config.enable_liveness_analysis:
            candidates = estimator.slicing_candidates()
        else:
            # Ablation B: no DAG liveness gate — all large tileable ops.
            candidates = self._all_large_tileable_ops(memory_at)

        self.results["liveness"] = estimator.analyzer.summary()
        self.results["peak_memory"] = estimator.summary()
        self.results["peak_op"] = peak_op
        self.results["peak_memory_bytes"] = peak_mem
        self.results["slicing_candidates"] = candidates
        self.results["ablation"] = {
            "planning_objective": self.config.planning_objective,
            "enable_liveness_analysis": self.config.enable_liveness_analysis,
            "enable_smart_heuristic": self.config.enable_smart_heuristic,
            "enable_fusion": self.config.enable_fusion,
            "planning_strategy": self.config.planning_strategy,
        }

        # 2. Rewrite planning
        t_plan = time.perf_counter()
        rewrite_plan = self._build_rewrite_plan(candidates)
        planning_ms = (time.perf_counter() - t_plan) * 1000

        # 3. Transform IR generation
        t_ir = time.perf_counter()
        rewriter = SlicedRewrite(
            self.dag,
            rewrite_plan,
            enable_fusion=self.config.enable_fusion,
            l1_cache_bytes=self.config.l1_cache_bytes,
            hardware_config=self.config.hardware_config,
            annotate_memory_tier=self.config.annotate_memory_tier,
            enforce_fusion_budget=self.config.enable_fusion_budget,
        )
        # ``SlicedRewrite`` drops chained anchors so they can be fused instead.
        # Report the plan it actually emits, not the pre-absorption one — the
        # difference is exactly the ops that get no scf.forall of their own.
        self._annotate_fusion(rewriter)
        self.results["rewrite_plan"] = rewriter.rewrite_plan
        self.results["planned_before_fusion"] = rewrite_plan
        self.results["absorbed_into_fusion"] = sorted(rewriter.absorbed_op_ids)
        self.results["planning_rejections"] = list(self._planning_rejections)
        self.results["transform_ir"] = rewriter.generate_transform_ir(style=transform_style)
        if self.config.materialize_sram:
            self.results["sram_promotion_transform_ir"] = (
                generate_sram_promotion_transform(
                    rewriter.rewrite_plan,
                    self.dag,
                    memory_space=self.config.sram_memory_space,
                    alignment=self.config.sram_alignment_bytes,
                )
            )
        self.results["transform_style"] = transform_style
        transform_ir_ms = (time.perf_counter() - t_ir) * 1000

        total_ms = (time.perf_counter() - t0) * 1000
        self.results["compile_timing"] = {
            "liveness_ms": round(liveness_ms, 3),
            "planning_ms": round(planning_ms, 3),
            "transform_ir_ms": round(transform_ir_ms, 3),
            "total_ms": round(total_ms, 3),
            "dag_vertices": len(self.dag.nodes),
            "dag_edges": len(self.dag.edges),
        }

        return self.results

    def _sram_residency_candidates(self, memory_at: Dict[int, int]) -> List[dict]:
        """Ops whose complete untiled compute working set exceeds fast SRAM.

        Candidate selection is deliberately independent of the global liveness
        peak. The capacity question is local: can the operands, weights and
        output needed by this operation reside together in the target SRAM?
        Sinks are included because they still execute and transfer data.
        """
        calc = self.config.tile_calculator()
        _, fast_memory_budget = calc.hardware_config.budget_levels()[0]
        candidates: List[dict] = []
        for op_id in self.dag.topological_order():
            node = self.dag.nodes[op_id]
            shape = tuple(node.output_tensor.shape) if node.has_single_output else ()
            if not node.has_single_output or not is_tileable(node.op_type, shape, node.params):
                continue
            if not shape:
                continue
            dim = dimension_index_to_name(0, node.op_type, len(shape))
            try:
                footprint = calc.estimate_tile_footprint(
                    shape,
                    dim,
                    int(shape[0]),
                    **self._tile_cost_kwargs(node),
                )
            except (TypeError, ValueError):
                continue
            if footprint.total_bytes <= fast_memory_budget:
                continue
            candidates.append({
                "op_id": op_id,
                "op_type": node.op_type,
                "output_size_bytes": node.output_tensor.size_bytes,
                "full_working_set_bytes": footprint.total_bytes,
                "fast_memory_budget_bytes": fast_memory_budget,
                "live_memory_bytes": memory_at.get(op_id, 0),
                "consumers": node.consumers,
                "candidate_reason": "untiled working set exceeds fast-memory budget",
            })
        return candidates

    def _all_large_tileable_ops(self, memory_at: Dict[int, int]) -> List[dict]:
        """Tileable ops whose output alone exceeds L1 (no liveness candidate filter)."""
        candidates = []
        for op_id in self.dag.topological_order():
            node = self.dag.nodes[op_id]
            if not node.has_single_output:
                continue
            if not is_tileable(node.op_type, node.output_tensor.shape, node.params):
                continue
            if node.output_tensor.size_bytes <= self.config.l1_cache_bytes:
                continue
            candidates.append({
                "op_id": op_id,
                "op_type": node.op_type,
                "output_size_bytes": node.output_tensor.size_bytes,
                "live_memory_bytes": memory_at.get(op_id, 0),
                "consumers": node.consumers,
            })
        return candidates

    def _build_rewrite_plan(self, candidates: List[dict]) -> RewritePlanList:
        dim_selector = DimensionSelector(self.dag)
        tile_calc = self.config.tile_calculator()

        rewrite_plan = []
        self._planning_rejections = []
        for candidate in candidates:
            op_id = candidate["op_id"]
            node = self.dag.nodes[op_id]
            if not node.has_single_output:
                self._planning_rejections.append({
                    "op_id": op_id,
                    "reason": "multi-result operations are analysis-only",
                })
                continue
            if not is_tileable(node.op_type, node.output_tensor.shape, node.params):
                self._planning_rejections.append({
                    "op_id": op_id,
                    "reason": "operation does not satisfy static safe-tiling preconditions",
                })
                continue

            cost_kwargs = self._tile_cost_kwargs(node)
            dim2: Optional[str] = None
            tile_size2: Optional[int] = None
            # Set only on the smart-heuristic path; the uniform baseline fixes
            # its tile by configuration and never runs a budget search, so its
            # binding level has to be derived from the footprint instead.
            decision: Optional[TileDecision] = None
            search_result: Optional[SearchResult] = None
            if self.config.enable_smart_heuristic:
                if self.config.planning_strategy == "search":
                    search = CapacityScheduleSearch(
                        tile_calc,
                        dataflows=self.config.schedule_dataflows,
                        max_candidates=self.config.schedule_search_max_candidates,
                        shortlist_size=self.config.schedule_shortlist_size,
                    )
                    try:
                        search_result = search.search(
                            tuple(node.output_tensor.shape),
                            node.op_type,
                            cost_kwargs=cost_kwargs,
                            allow_2d=self.config.enable_2d_cascade,
                            measure=self.config.target_measure,
                        )
                    except ValueError as exc:
                        self._planning_rejections.append({
                            "op_id": op_id,
                            "reason": str(exc),
                        })
                        continue
                    winner = search_result.winner
                    dim = winner.dimensions[0]
                    tile_size = winner.tile_sizes[0]
                    dim2 = winner.dimensions[1] if len(winner.dimensions) == 2 else None
                    tile_size2 = winner.tile_sizes[1] if len(winner.tile_sizes) == 2 else None
                    ranked_names = dim_selector.rank(op_id)
                    dim_rank = ranked_names.index(dim) if dim in ranked_names else len(ranked_names)
                elif self.config.enable_dimension_cascade:
                    dim_choices = dim_selector.rank(op_id)
                else:
                    dim_choices = [dim_selector.select(op_id)]
                if self.config.planning_strategy == "cascade":
                    dim = None
                    tile_size = None
                    last_error: Optional[TileBudgetError] = None
                    for rank_index, candidate_dim in enumerate(dim_choices):
                        try:
                            decision = tile_calc.compute_detailed(
                                node.output_tensor.shape,
                                candidate_dim,
                                **cost_kwargs,
                            )
                        except TileBudgetError as exc:
                            last_error = exc
                            continue
                        tile_size = decision.tile_size
                        dim = candidate_dim
                        dim_rank = rank_index
                        break
                    pair_choices: List[Tuple[str, str]] = []
                    if dim is None and self.config.enable_2d_cascade:
                        # Every single dimension overflowed at unit size; a 2D
                        # tile can still be cache-resident.
                        pair_choices = dim_selector.rank_pairs(op_id)
                        for pair_index, (primary, secondary) in enumerate(pair_choices):
                            try:
                                tile_size, tile_size2 = tile_calc.compute_pair(
                                    node.output_tensor.shape,
                                    primary,
                                    secondary,
                                    **cost_kwargs,
                                )
                            except TileBudgetError as exc:
                                last_error = exc
                                continue
                            dim = primary
                            dim2 = secondary
                            dim_rank = len(dim_choices) + pair_index
                            break
                    if dim is None:
                        tried = list(dim_choices) + [
                            f"{p}+{s}" for p, s in pair_choices
                        ]
                        self._planning_rejections.append({
                            "op_id": op_id,
                            "reason": (
                                f"no slice dimension fits the cache budget "
                                f"(tried {', '.join(tried)}); last: {last_error}"
                            ),
                        })
                        continue
            else:
                # Ablation A / uniform baseline: fixed dim + tile.
                dim = self.config.uniform_slice_dimension
                tile_size = self.config.uniform_tile_size
                dim_rank = 0
                try:
                    dimension_name_to_index(dim, node.op_type, len(node.output_tensor.shape))
                except ValueError as exc:
                    self._planning_rejections.append({
                        "op_id": op_id,
                        "reason": str(exc),
                    })
                    continue

            assert tile_size is not None  # both branches above set it or continue
            shape = tuple(node.output_tensor.shape)
            dim_idx = dimension_name_to_index(dim, node.op_type, len(shape))
            extent = shape[dim_idx]
            if min(tile_size, extent) >= extent:
                # tile == full extent gives a single-iteration scf.forall: no
                # working-set reduction, and canonicalization folds the loop
                # away so the pipeline's materialization check would then fail.
                self._planning_rejections.append({
                    "op_id": op_id,
                    "reason": (
                        f"tile {tile_size} covers the whole '{dim}' extent "
                        f"{extent}; nothing to slice"
                    ),
                })
                continue
            if dim2 is not None and tile_size2 is not None:
                dim2_idx = dimension_name_to_index(dim2, node.op_type, len(shape))
                if min(tile_size2, shape[dim2_idx]) >= shape[dim2_idx]:
                    # Secondary grew to the full extent: the pair degenerated
                    # to a 1D tile along the primary.
                    dim2 = None
                    tile_size2 = None

            secondary_pair = (
                (dim2, tile_size2)
                if dim2 is not None and tile_size2 is not None
                else None
            )
            tile_footprint = tile_calc.estimate_tile_footprint(
                node.output_tensor.shape,
                dim,
                tile_size,
                secondary=secondary_pair,
                **cost_kwargs,
            )
            tile_count = math.ceil(extent / tile_size)
            if secondary_pair is not None:
                dim2_idx = dimension_name_to_index(
                    secondary_pair[0], node.op_type, len(shape)
                )
                tile_count *= math.ceil(shape[dim2_idx] / secondary_pair[1])
            # Conservative traffic proxy: assume each tile reloads all of its
            # read-only operands and writes its output once. A later target
            # tuner may prove weights persistent across tiles and lower this
            # bound; the static planner must not silently assume such reuse.
            per_tile_read_bytes = (
                tile_footprint.input_bytes
                + tile_footprint.weight_bytes
                + tile_footprint.operand_bytes
            )
            estimated_offchip_bytes = (
                tile_count * per_tile_read_bytes + node.output_tensor.size_bytes
            )

            # Which cache level holds this tile, and how much of that level it
            # claims. Recorded per op because the *distribution* over a model
            # is the result: how much of the network can be made L1-resident at
            # all, and how much only fits with the 2D cascade (proposal §5.1).
            if decision is not None and decision.tile_size == tile_size:
                tile_budget_level = decision.budget_level
            else:
                # 2D cascade, or a uniform tile that was never budget-searched.
                tile_budget_level = tile_calc.binding_level_name(
                    tile_footprint.total_bytes
                )

            plan_entry: RewritePlan = {
                "op_id": op_id,
                "op_type": node.op_type,
                "slice_dimension": dim,
                "tile_budget_level": tile_budget_level,
                "tile_budget_utilization": round(
                    tile_calc.budget_utilization(tile_footprint.total_bytes), 4
                ),
                "tile_dimensionality": 2 if secondary_pair is not None else 1,
                # 0 = the heuristic's first choice; >0 means the preferred
                # dimension had no tile that fit and the cascade fell back.
                "slice_dimension_rank": dim_rank,
                "tile_size": tile_size,
                "original_shape": node.output_tensor.shape,
                "estimated_tile_working_set_bytes": tile_footprint.total_bytes,
                "estimated_tile_count": tile_count,
                "estimated_offchip_traffic_bytes_upper_bound": estimated_offchip_bytes,
                "full_working_set_bytes": candidate.get("full_working_set_bytes"),
                "fast_memory_budget_bytes": candidate.get("fast_memory_budget_bytes"),
                "params": node.params,
                "edgeslice_op_id": node.params.get(
                    "edgeslice_op_id", f"op_{op_id}"
                ),
            }
            if search_result is not None:
                winner = search_result.winner
                plan_entry.update({
                    "planning_strategy": "search",
                    "schedule_candidates_evaluated": len(search_result.candidates),
                    "schedule_shortlist_size": len(search_result.shortlist),
                    "schedule_dataflow": winner.dataflow,
                    "schedule_analytical_rank": list(winner.analytical_rank),
                    "schedule_measured": search_result.measured,
                    "schedule_measured_latency_ms": winner.measured_latency_ms,
                })
            if secondary_pair is not None:
                plan_entry["slice_dimension_2"] = secondary_pair[0]
                plan_entry["tile_size_2"] = secondary_pair[1]
            rewrite_plan.append(plan_entry)
        return rewrite_plan

    def _annotate_fusion(self, rewriter: SlicedRewrite) -> None:
        """Re-fit each anchor's tile to the *fused* loop body, then record it.

        ``estimated_tile_working_set_bytes`` covers the anchor alone. Producers
        fused into the same ``scf.forall`` keep their own tiles live at the same
        time, so proposal H1 ("every accepted tile plan fits the configured
        analytical cache budget") has to be evaluated on the sum. The tile is
        shrunk until the sum fits; when even a unit tile cannot fit, the anchor
        keeps its own feasible tile and the shortfall is reported instead of
        being passed off as a budget-compliant plan.
        """
        tile_calc = self.config.tile_calculator()
        budget = tile_calc.budget_bytes
        over_budget: List[Dict[str, Any]] = []

        for plan in rewriter.rewrite_plan:
            chain = rewriter.fusion_chains.get(plan["op_id"], [])
            plan["fused_producer_op_ids"] = list(chain)
            if not chain:
                plan["estimated_fused_working_set_bytes"] = plan[
                    "estimated_tile_working_set_bytes"
                ]
                continue

            anchor = self.dag.nodes[plan["op_id"]]
            slice_dim = plan["slice_dimension"]
            # 2D anchors keep their secondary tile fixed during the re-fit;
            # only the primary tile shrinks to fit the fused group.
            secondary: Optional[Tuple[str, int]] = None
            if "slice_dimension_2" in plan:
                secondary = (plan["slice_dimension_2"], plan["tile_size_2"])
            members = [
                (tuple(anchor.output_tensor.shape), self._tile_cost_kwargs(anchor))
            ] + [
                (
                    tuple(self.dag.nodes[pid].output_tensor.shape),
                    self._tile_cost_kwargs(self.dag.nodes[pid]),
                )
                for pid in chain
            ]

            if not self.config.enable_smart_heuristic:
                # Ablation A fixes the tile by configuration. Re-fitting it here
                # would quietly turn the uniform baseline into a budget-driven
                # one, so only the measurement below is recorded.
                plan["estimated_fused_working_set_bytes"] = (
                    tile_calc.estimate_group_footprint_bytes(
                        members, slice_dim, plan["tile_size"], secondary=secondary
                    )
                )
                if plan["estimated_fused_working_set_bytes"] > budget:
                    over_budget.append({
                        "op_id": plan["op_id"],
                        "estimated_fused_working_set_bytes": plan[
                            "estimated_fused_working_set_bytes"
                        ],
                        "l1_budget_bytes": budget,
                        "fused_producer_op_ids": list(chain),
                        "reason": "uniform tile (ablation A) is not budget-fitted",
                    })
                continue

            try:
                fused_tile = tile_calc.compute_for_group(
                    members,
                    slice_dim,
                    max_tile_size=plan["tile_size"],
                    secondary=secondary,
                )
            except TileBudgetError as exc:
                # A unit tile of the fused body already overflows: keep the
                # anchor's own feasible tile and surface the gap.
                plan["fused_budget_reason"] = str(exc)
            else:
                if fused_tile < plan["tile_size"]:
                    plan["tile_size_before_fusion"] = plan["tile_size"]
                    plan["tile_size"] = fused_tile
                    plan["estimated_tile_working_set_bytes"] = (
                        tile_calc.estimate_tile_footprint(
                            anchor.output_tensor.shape,
                            slice_dim,
                            fused_tile,
                            secondary=secondary,
                            **self._tile_cost_kwargs(anchor),
                        ).total_bytes
                    )

            total = tile_calc.estimate_group_footprint_bytes(
                members, slice_dim, plan["tile_size"], secondary=secondary
            )
            plan["estimated_fused_working_set_bytes"] = total
            if total > budget:
                over_budget.append({
                    "op_id": plan["op_id"],
                    "estimated_fused_working_set_bytes": total,
                    "l1_budget_bytes": budget,
                    "fused_producer_op_ids": list(chain),
                })

        self.results["fusion_budget_warnings"] = over_budget

    @staticmethod
    def _tile_cost_kwargs(node) -> Dict[str, Any]:
        """Provide op-specific memory metadata to the tile cost model."""
        return node_cost_kwargs(node)

    def get_report(self) -> str:
        lines = [
            "=" * 70,
            "EdgeSlice Frontend Compiler Report (Proposal-aligned)",
            "=" * 70,
            "",
            self.results.get("liveness", ""),
            "",
            self.results.get("peak_memory", ""),
            "",
            f"Transform Style: {self.results.get('transform_style', 'structured')}",
            "",
            "Rewrite Plan:",
        ]
        for plan in self.results.get("rewrite_plan", []):
            lines.append(
                f"  Op {plan['op_id']} ({plan['op_type']}): "
                f"dim={plan['slice_dimension']}, tile={plan['tile_size']}"
            )

        lines.append("\nGenerated Transform Dialect IR:\n")
        lines.append(self.results.get("transform_ir", ""))
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)
