"""Operation DAG — data structures for tensor dependency graphs."""

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple, Union


@dataclass
class TensorInfo:
    """Metadata for an intermediate tensor."""
    shape: Tuple[int, ...]
    dtype: str = "f32"
    name: Optional[str] = None
    shape_bounds: Dict[Union[int, str], int] = field(default_factory=dict)

    DYNAMIC_DIM: ClassVar[int] = -1
    DTYPE_SIZES: ClassVar[Dict[str, int]] = {
        "f16": 2,
        "bf16": 2,
        "f32": 4,
        "f64": 8,
        "i8": 1,
        "i16": 2,
        "i32": 4,
        "i64": 8,
        "index": 8,
    }

    @property
    def resolved_shape(self) -> Tuple[int, ...]:
        """Return a concrete shape, resolving dynamic dims via ``shape_bounds``.

        ``DAGParser`` represents dynamic dimensions as ``-1``.  Multiplying
        those dimensions directly would silently produce negative memory sizes,
        so memory estimation either needs static shapes or conservative bounds
        for each dynamic dimension.
        """
        resolved = []
        for index, dim in enumerate(self.shape):
            if dim == self.DYNAMIC_DIM:
                resolved.append(self._resolve_dynamic_dim(index))
            elif dim < 0:
                raise ValueError(
                    f"Tensor {self._display_name()} has invalid negative dimension "
                    f"{dim} at index {index}; only -1 is supported as dynamic"
                )
            else:
                resolved.append(dim)
        return tuple(resolved)

    @property
    def element_size_bytes(self) -> int:
        if self.dtype not in self.DTYPE_SIZES:
            raise ValueError(f"Unsupported dtype {self.dtype!r} for tensor {self._display_name()}")
        return self.DTYPE_SIZES[self.dtype]

    @property
    def size_bytes(self) -> int:
        import math
        return math.prod(self.resolved_shape) * self.element_size_bytes

    def _resolve_dynamic_dim(self, index: int) -> int:
        for key in (index, str(index), f"dim_{index}"):
            if key in self.shape_bounds:
                bound = int(self.shape_bounds[key])
                if bound <= 0:
                    raise ValueError(
                        f"Tensor {self._display_name()} has non-positive bound "
                        f"{bound} for dynamic dimension {index}"
                    )
                return bound
        raise ValueError(
            f"Tensor {self._display_name()} has dynamic dimension at index {index}; "
            "provide shape_bounds to estimate memory safely"
        )

    def _display_name(self) -> str:
        return self.name or "<unnamed>"


@dataclass
class OpNode:
    """A node in the operation DAG."""
    op_id: int
    op_type: str
    input_tensors: List[TensorInfo]
    output_tensor: TensorInfo
    output_tensors: List[TensorInfo] = field(default_factory=list)
    params: Dict = field(default_factory=dict)
    consumers: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.output_tensors:
            self.output_tensors = [self.output_tensor]

    @property
    def has_single_output(self) -> bool:
        return len(self.output_tensors) == 1


class OperationDAG:
    """Directed acyclic graph of operations with tensor dependencies."""

    def __init__(self):
        self.nodes: Dict[int, OpNode] = {}
        self.edges: List[Tuple[int, int]] = []
        self._next_id = 0

    def add_node(
        self,
        op_type: str,
        inputs: List[TensorInfo],
        output: TensorInfo,
        params: Optional[Dict] = None,
        *,
        outputs: Optional[List[TensorInfo]] = None,
    ) -> int:
        op_id = self._next_id
        self._next_id += 1
        node = OpNode(
            op_id=op_id,
            op_type=op_type,
            input_tensors=inputs,
            output_tensor=output,
            output_tensors=list(outputs) if outputs is not None else [output],
            params=params or {},
        )
        self.nodes[op_id] = node
        return op_id

    def add_edge(self, producer_id: int, consumer_id: int):
        self.edges.append((producer_id, consumer_id))
        if producer_id in self.nodes:
            self.nodes[producer_id].consumers.append(consumer_id)

    def topological_order(self) -> List[int]:
        """Compute a topological ordering of nodes using Kahn's algorithm."""
        # Build adjacency list and in-degree count
        in_degree = {node_id: 0 for node_id in self.nodes}
        adjacency = defaultdict(list)

        for producer, consumer in self.edges:
            # An edge touching an unknown node constrains nothing and must not
            # be counted: an in-degree nobody can decrement is later reported as
            # a cycle, and an unknown consumer raises KeyError outright.
            if producer not in self.nodes or consumer not in self.nodes:
                continue
            adjacency[producer].append(consumer)
            in_degree[consumer] += 1

        # Kahn's algorithm: start with nodes that have no incoming edges.
        # A min-heap keyed by node id keeps the order faithful to program
        # order (ids are assigned in MLIR walk order), so liveness ranges
        # measured on this order match real buffer lifetimes instead of
        # hoisting every root (weights/inits) to the front in one wave.
        heap = [node_id for node_id, degree in in_degree.items() if degree == 0]
        heapq.heapify(heap)
        result = []

        while heap:
            node = heapq.heappop(heap)
            result.append(node)

            # Reduce in-degree for neighbors
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    heapq.heappush(heap, neighbor)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains cycle")

        return result
