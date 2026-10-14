"""Capacity-constrained schedule enumeration and target selection.

Unlike the historical dimension cascade, this module makes the planner's
search space explicit.  Every returned schedule has been costed with the same
``TileSizeCalculator`` used by the pass manager and therefore satisfies the
configured fast-memory budget.  The analytical score is deliberately a tuple
of auditable quantities rather than a single fitted constant; an optional
measurement callback can select the fastest member of the analytical
shortlist on the real target.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from edgeslide.tiling.dimension_selector import (
    dimension_name_to_index,
    rank_dimension_index_pairs,
    rank_dimension_indices,
)
from edgeslide.tiling.tile_size import TileSizeCalculator

DATAFLOWS: Tuple[str, ...] = (
    "output_stationary",
    "input_stationary",
    "weight_stationary",
)


@dataclass(frozen=True)
class ScheduleCandidate:
    workload_key: str
    dimensions: Tuple[str, ...]
    tile_sizes: Tuple[int, ...]
    dataflow: str
    footprint_bytes: int
    budget_bytes: int
    budget_level: str
    tile_count: int
    estimated_offchip_bytes: int
    bank_conflict_penalty: int
    vector_utilization: float
    budget_utilization: float
    analytical_rank: Tuple[float, ...]
    measured_latency_ms: Optional[float] = None

    def with_latency(self, latency_ms: float) -> "ScheduleCandidate":
        values = asdict(self)
        values["measured_latency_ms"] = float(latency_ms)
        # JSON/asdict turns tuples into tuple-compatible values at construction.
        values["dimensions"] = tuple(values["dimensions"])
        values["tile_sizes"] = tuple(values["tile_sizes"])
        values["analytical_rank"] = tuple(values["analytical_rank"])
        return ScheduleCandidate(**values)


@dataclass(frozen=True)
class SearchResult:
    winner: ScheduleCandidate
    candidates: Tuple[ScheduleCandidate, ...]
    shortlist: Tuple[ScheduleCandidate, ...]
    measured: bool


MeasureSchedule = Callable[[ScheduleCandidate], float]


def _candidate_sizes(extent: int, maximum: int, granule: int) -> List[int]:
    """Finite coverage of boundary, divisor and power-of-two tile sizes."""
    maximum = max(1, min(int(extent), int(maximum)))
    values = {1, maximum}
    value = 1
    while value < maximum:
        values.add(value)
        value *= 2
    # Divisors avoid tail tiles; cache-line granules cover non-power-of-two
    # capacity boundaries without enumerating every integer.
    for divisor in range(1, int(math.sqrt(extent)) + 1):
        if extent % divisor == 0:
            if divisor <= maximum:
                values.add(divisor)
            paired = extent // divisor
            if paired <= maximum:
                values.add(paired)
    if granule > 1:
        values.update(range(granule, maximum + 1, granule))
    return sorted(values)


class CapacityScheduleSearch:
    """Enumerate legal 1D/2D schedules and rank them by traffic/locality."""

    def __init__(
        self,
        calculator: TileSizeCalculator,
        *,
        # Only output-stationary is materialized by the current Transform
        # dialect emitter. Other dataflows are accepted explicitly for target
        # backends that implement them; they are never searched under a label
        # the code generator cannot honour.
        dataflows: Sequence[str] = ("output_stationary",),
        max_dimensions: int = 3,
        max_candidates: int = 4096,
        shortlist_size: int = 8,
    ):
        unknown = set(dataflows) - set(DATAFLOWS)
        if unknown:
            raise ValueError(f"unknown dataflow(s): {sorted(unknown)}")
        if max_candidates <= 0 or shortlist_size <= 0:
            raise ValueError("search limits must be positive")
        self.calculator = calculator
        self.dataflows = tuple(dataflows)
        self.max_dimensions = max(1, int(max_dimensions))
        self.max_candidates = int(max_candidates)
        self.shortlist_size = int(shortlist_size)

    def enumerate(
        self,
        shape: Tuple[int, ...],
        op_type: str,
        *,
        cost_kwargs: Dict[str, Any],
        allow_2d: bool = True,
    ) -> List[ScheduleCandidate]:
        ranked_indices = rank_dimension_indices(shape, op_type)[: self.max_dimensions]
        dimensions = [
            self._dimension_name(index, op_type, len(shape)) for index in ranked_indices
        ]
        vector_elems = max(
            1,
            self.calculator.hardware_config.vector_width_bytes
            // max(1, int(cost_kwargs.get("dtype_size", 4))),
        )
        schedules: List[ScheduleCandidate] = []

        for dimension in dimensions:
            index = dimension_name_to_index(dimension, op_type, len(shape))
            extent = int(shape[index])
            maximum = min(extent, self.calculator.max_tile_size or extent)
            for tile in _candidate_sizes(extent, maximum, vector_elems):
                self._append_if_legal(
                    schedules, shape, op_type, (dimension,), (tile,), cost_kwargs
                )
                if len(schedules) >= self.max_candidates:
                    return sorted(schedules, key=lambda item: item.analytical_rank)

        if allow_2d:
            for first_index, second_index in rank_dimension_index_pairs(shape, op_type):
                if first_index not in ranked_indices or second_index not in ranked_indices:
                    continue
                first = self._dimension_name(first_index, op_type, len(shape))
                second = self._dimension_name(second_index, op_type, len(shape))
                first_extent, second_extent = int(shape[first_index]), int(shape[second_index])
                first_max = min(first_extent, self.calculator.max_tile_size or first_extent)
                second_max = min(second_extent, self.calculator.max_tile_size or second_extent)
                first_sizes = _candidate_sizes(first_extent, first_max, vector_elems)
                second_sizes = _candidate_sizes(second_extent, second_max, vector_elems)
                for first_tile in first_sizes:
                    for second_tile in second_sizes:
                        self._append_if_legal(
                            schedules,
                            shape,
                            op_type,
                            (first, second),
                            (first_tile, second_tile),
                            cost_kwargs,
                        )
                        if len(schedules) >= self.max_candidates:
                            return sorted(schedules, key=lambda item: item.analytical_rank)

        return sorted(schedules, key=lambda item: item.analytical_rank)

    def search(
        self,
        shape: Tuple[int, ...],
        op_type: str,
        *,
        cost_kwargs: Dict[str, Any],
        allow_2d: bool = True,
        measure: Optional[MeasureSchedule] = None,
    ) -> SearchResult:
        candidates = self.enumerate(
            shape, op_type, cost_kwargs=cost_kwargs, allow_2d=allow_2d
        )
        # A full-extent tile is legal but is not a transformation. Keep it in
        # the audit trail, while preferring schedules that actually partition.
        effective = [candidate for candidate in candidates if candidate.tile_count > 1]
        if not effective:
            raise ValueError(
                "no slice dimension fits the cache budget with a partitioning schedule"
            )
        shortlist = effective[: self.shortlist_size]
        if measure is None:
            return SearchResult(effective[0], tuple(candidates), tuple(shortlist), False)

        measured: List[ScheduleCandidate] = []
        for candidate in shortlist:
            latency = float(measure(candidate))
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"target measurement returned invalid latency {latency!r}")
            measured.append(candidate.with_latency(latency))
        measured.sort(
            key=lambda candidate: (
                candidate.measured_latency_ms,
                candidate.analytical_rank,
            )
        )
        return SearchResult(measured[0], tuple(candidates), tuple(measured), True)

    @staticmethod
    def _dimension_name(index: int, op_type: str, rank: int) -> str:
        from edgeslide.tiling.dimension_selector import dimension_index_to_name

        return dimension_index_to_name(index, op_type, rank)

    def _append_if_legal(
        self,
        output: List[ScheduleCandidate],
        shape: Tuple[int, ...],
        op_type: str,
        dimensions: Tuple[str, ...],
        tile_sizes: Tuple[int, ...],
        cost_kwargs: Dict[str, Any],
    ) -> None:
        secondary = (
            (dimensions[1], tile_sizes[1]) if len(dimensions) == 2 else None
        )
        footprint = self.calculator.estimate_tile_footprint(
            shape,
            dimensions[0],
            tile_sizes[0],
            secondary=secondary,
            **cost_kwargs,
        )
        legal_levels = [
            (name, budget)
            for name, budget in self.calculator.hardware_config.budget_levels()
            if footprint.total_bytes <= budget
        ]
        if not legal_levels:
            return
        level, budget = legal_levels[0]
        tile_count = 1
        for dimension, tile in zip(dimensions, tile_sizes):
            index = dimension_name_to_index(dimension, op_type, len(shape))
            tile_count *= math.ceil(int(shape[index]) / int(tile))

        read_bytes = footprint.input_bytes + footprint.weight_bytes + footprint.operand_bytes
        traffic = tile_count * read_bytes + math.prod(shape) * int(
            cost_kwargs.get("dtype_size", 4)
        )
        dtype_size = max(1, int(cost_kwargs.get("dtype_size", 4)))
        vector_elems = max(
            1, self.calculator.hardware_config.vector_width_bytes // dtype_size
        )
        innermost = dimension_name_to_index(dimensions[-1], op_type, len(shape))
        vector_utilization = min(1.0, tile_sizes[-1] / vector_elems)
        if tile_sizes[-1] % vector_elems:
            vector_utilization *= tile_sizes[-1] / (
                math.ceil(tile_sizes[-1] / vector_elems) * vector_elems
            )
        # Fragmenting the contiguous axis creates partial cache-line accesses
        # for each outer row. This is an auditable proxy, not a hardware claim.
        bank_penalty = tile_count if innermost == len(shape) - 1 else 0
        utilization = footprint.total_bytes / budget
        workload_payload = {
            "shape": shape,
            "op_type": op_type,
            "dtype_size": cost_kwargs.get("dtype_size"),
            "operand_shapes": cost_kwargs.get("operand_shapes"),
            "kernel_shape": cost_kwargs.get("kernel_shape"),
            "strides": cost_kwargs.get("strides"),
            "dilations": cost_kwargs.get("dilations"),
            "groups": cost_kwargs.get("groups"),
        }
        workload_key = hashlib.sha256(
            json.dumps(workload_payload, sort_keys=True, default=str).encode()
        ).hexdigest()

        for dataflow in self.dataflows:
            # No persistence is credited without a target-specific proof.  The
            # dataflow remains explicit so target measurement can distinguish
            # implementations and the result never invents traffic savings.
            rank = (
                float(bank_penalty),
                float(traffic),
                -float(vector_utilization),
                float(tile_count),
                -float(utilization),
                float(DATAFLOWS.index(dataflow)),
            )
            output.append(
                ScheduleCandidate(
                    workload_key=workload_key,
                    dimensions=dimensions,
                    tile_sizes=tile_sizes,
                    dataflow=dataflow,
                    footprint_bytes=footprint.total_bytes,
                    budget_bytes=budget,
                    budget_level=level,
                    tile_count=tile_count,
                    estimated_offchip_bytes=traffic,
                    bank_conflict_penalty=bank_penalty,
                    vector_utilization=round(vector_utilization, 6),
                    budget_utilization=round(utilization, 6),
                    analytical_rank=rank,
                )
            )


class TargetAutotuner:
    """Repeat, robustly summarize and cache real target measurements."""

    def __init__(
        self,
        measure_once: MeasureSchedule,
        *,
        warmups: int = 1,
        trials: int = 5,
        cache_path: Optional[Path] = None,
        target_fingerprint: str = "unknown-target",
    ):
        if warmups < 0 or trials <= 0:
            raise ValueError("warmups must be >= 0 and trials must be > 0")
        self.measure_once = measure_once
        self.warmups = int(warmups)
        self.trials = int(trials)
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.target_fingerprint = target_fingerprint
        self._cache = self._load_cache()

    def __call__(self, candidate: ScheduleCandidate) -> float:
        key = self._key(candidate)
        cached = self._cache.get(key)
        if cached is not None:
            return float(cached["median_ms"])
        for _ in range(self.warmups):
            self.measure_once(candidate)
        samples = [float(self.measure_once(candidate)) for _ in range(self.trials)]
        if any(not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError(f"invalid target samples: {samples}")
        median = statistics.median(samples)
        self._cache[key] = {
            "median_ms": median,
            "samples_ms": samples,
            "timestamp_ns": time.time_ns(),
        }
        self._save_cache()
        return median

    def _key(self, candidate: ScheduleCandidate) -> str:
        payload = {
            "target": self.target_fingerprint,
            "workload": candidate.workload_key,
            "dimensions": candidate.dimensions,
            "tile_sizes": candidate.tile_sizes,
            "dataflow": candidate.dataflow,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_path is None or not self.cache_path.is_file():
            return {}
        data = json.loads(self.cache_path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"autotune cache must be an object: {self.cache_path}")
        return data

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._cache, indent=2, sort_keys=True))
        temporary.replace(self.cache_path)
