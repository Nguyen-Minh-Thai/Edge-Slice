from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from edgeslide.dag.graph import OperationDAG, TensorInfo
from edgeslide.liveness.analyzer import BufferizedMemoryAnalyzer, LivenessAnalyzer

# Tensor-dialect ops that write into a destination operand instead of into a
# fresh buffer. Every linalg op is destination-passing too (it carries `outs`),
# which `_is_destination_passing` handles by dialect prefix.
_DPS_TENSOR_OP_TYPES = frozenset({"tensor.insert_slice", "tensor.parallel_insert_slice"})


def _is_destination_passing(op_type: str) -> bool:
    """True when the op's result is written into one of its operands.

    The parser records tensor-typed operands in MLIR order, so the destination
    is the *last* one: linalg ops are ``ins(...) outs(...)``, and
    ``tensor.insert_slice %src into %dest`` puts `dest` after `src` (its
    trailing offset/size/stride operands are index-typed, so
    ``_get_or_create_tensor_info`` never records them).
    """
    lowered = op_type.lower()
    return lowered.startswith("linalg.") or lowered in _DPS_TENSOR_OP_TYPES


class BufferizedPeakMemoryEstimator:
    """Peak memory estimator for bufferized MLIR allocation lifetimes.

    Prefer this estimator when a tensor-level payload can be processed by
    ``mlir-opt --one-shot-bufferize``. It tracks concrete ``memref.alloc`` and
    ``memref.dealloc`` lifetimes, reflecting bufferization and memory reuse
    decisions that frontend DAG liveness cannot model.
    """

    def __init__(
        self,
        bufferized_mlir: Optional[str] = None,
        *,
        bufferized_path: Optional[Union[str, Path]] = None,
        tensor_mlir_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        mlir_opt: Optional[str] = None,
        extra_bufferize_args: Optional[List[str]] = None,
        shape_bounds: Optional[Mapping[Union[int, str], Any]] = None,
    ):
        provided = [
            bufferized_mlir is not None,
            bufferized_path is not None,
            tensor_mlir_path is not None,
        ]
        if sum(provided) != 1:
            raise ValueError(
                "Provide exactly one of bufferized_mlir, bufferized_path, or tensor_mlir_path"
            )

        if bufferized_mlir is not None:
            self.analyzer = BufferizedMemoryAnalyzer(
                bufferized_mlir,
                shape_bounds=shape_bounds,
            )
        elif bufferized_path is not None:
            self.analyzer = BufferizedMemoryAnalyzer.from_bufferized_file(
                bufferized_path,
                shape_bounds=shape_bounds,
            )
        else:
            assert tensor_mlir_path is not None
            self.analyzer = BufferizedMemoryAnalyzer.from_tensor_mlir_file(
                tensor_mlir_path,
                output_path=output_path,
                mlir_opt=mlir_opt,
                extra_bufferize_args=extra_bufferize_args,
                shape_bounds=shape_bounds,
            )

    def estimate(self) -> Dict[int, int]:
        return self.analyzer.estimate()

    def get_peak(self) -> tuple:
        return self.analyzer.get_peak()

    def summary(self) -> str:
        return self.analyzer.summary()


class PeakMemoryEstimator:
    """Estimate frontend tensor-liveness memory at each DAG program point.

    This is a pre-bufferization estimator. Use ``BufferizedPeakMemoryEstimator``
    when a payload can be analyzed after ``--one-shot-bufferize``.
    """

    def __init__(self, dag: OperationDAG, l1_cache_bytes: int = 512 * 1024):
        self.dag = dag
        self.l1_cache_bytes = l1_cache_bytes
        self.analyzer = LivenessAnalyzer(dag)
        self._memory_at: Optional[Dict[int, int]] = None
        self._memory_fingerprint: Optional[tuple] = None

    def _fingerprint(self) -> tuple:
        """Cache key covering everything the sweep reads from the DAG.

        Node/edge counts alone would serve stale results to an in-place
        mutation that preserves counts (e.g. a pass shrinking tensor shapes),
        so per-op tensor sizes are included. Building this is O(|V|) — noise
        next to the O(|V|·K) sweep it guards.
        """
        return (
            len(self.dag.nodes),
            len(self.dag.edges),
            tuple(
                (
                    op_id,
                    tuple(t.size_bytes for t in node.output_tensors),
                    tuple(t.size_bytes for t in node.input_tensors),
                )
                for op_id, node in self.dag.nodes.items()
            ),
        )

    def estimate(self) -> Dict[int, int]:
        """Bytes live at each program point, cached per DAG contents.

        ``get_peak``/``exceeds_l1``/``slicing_candidates``/``summary`` all go
        through here, so without the cache one pass-manager run repeats the
        O(|V|·K) liveness scan and the byte accounting roughly nine times.
        """
        fingerprint = self._fingerprint()
        if self._memory_at is not None and self._memory_fingerprint == fingerprint:
            return self._memory_at

        live_at = self.analyzer.analyze()
        tensor_idx_to_info = self._build_tensor_map()
        representative, no_allocation = self._alias_state(tensor_idx_to_info)
        memory_at: Dict[int, int] = {}
        for op_id, live_tensors in live_at.items():
            # Sum over distinct *buffers*, not distinct SSA values: several
            # values in one destination-passing chain share an allocation.
            total = 0
            counted: Set[int] = set()
            for t in live_tensors:
                if t not in tensor_idx_to_info or t in no_allocation:
                    continue
                buffer_id = representative.get(t, t)
                if buffer_id in counted:
                    continue
                counted.add(buffer_id)
                total += tensor_idx_to_info[t].size_bytes
            memory_at[op_id] = total
        self._memory_at = memory_at
        self._memory_fingerprint = fingerprint
        return memory_at

    def estimate_with_output_overrides(
        self, overrides: Mapping[int, int]
    ) -> Dict[int, int]:
        """``estimate()`` with some ops' output buffers resized everywhere.

        ``overrides`` maps op_id → the number of bytes that op's output really
        occupies after a rewrite. The substitution applies at **every** program
        point where the buffer is live, not only at the point that defines it —
        which is the whole difference between modelling fusion correctly and
        not.

        A producer fused into an anchor's ``scf.forall`` is computed
        tile-by-tile and consumed inside the loop, so its full-size buffer
        never exists at any point in the program. Crediting it only at its own
        program point leaves it counted at full size everywhere it is still
        live — including at the anchor, which is usually exactly where the peak
        is. ``benchmarks.metrics`` did that, and the resulting shortfall was
        masked by also crediting the anchor's own output, which does *not*
        shrink (its destination is full-size). Removing the wrong credit
        exposes the missing one, so both are fixed together.

        Buffer aliasing, ``tensor.empty`` handling and destination-passing
        merging are shared with ``estimate()`` rather than duplicated, so the
        baseline and the post-rewrite estimate are always computed under one
        memory model. Not cached: overrides vary per candidate plan, and the
        expensive part (the liveness sweep) is already memoized inside
        ``LivenessAnalyzer``.
        """
        live_at = self.analyzer.analyze()
        tensor_idx_to_info = self._build_tensor_map()
        representative, no_allocation = self._alias_state(tensor_idx_to_info)

        size_of: Dict[int, int] = {
            idx: info.size_bytes for idx, info in tensor_idx_to_info.items()
        }
        index_of_id: Dict[int, int] = {
            id(info): idx for idx, info in tensor_idx_to_info.items()
        }
        for op_id, tile_bytes in overrides.items():
            node = self.dag.nodes.get(op_id)
            if node is None:
                continue
            for output_tensor in node.output_tensors:
                idx = index_of_id.get(id(output_tensor))
                if idx is None:
                    continue
                # An override may only shrink: a rewrite that would enlarge a
                # buffer is not a credit and must not be recorded as one.
                size_of[idx] = min(size_of[idx], int(tile_bytes))

        memory_at: Dict[int, int] = {}
        for op_id, live_tensors in live_at.items():
            total = 0
            counted: Set[int] = set()
            for t in live_tensors:
                if t not in tensor_idx_to_info or t in no_allocation:
                    continue
                buffer_id = representative.get(t, t)
                if buffer_id in counted:
                    continue
                counted.add(buffer_id)
                total += size_of[t]
            memory_at[op_id] = total
        return memory_at

    def get_peak(self) -> tuple:
        memory_at = self.estimate()
        if not memory_at:
            return (-1, 0)
        peak_op = max(memory_at, key=lambda op: memory_at[op])
        return (peak_op, memory_at[peak_op])

    def exceeds_l1(self) -> List[int]:
        memory_at = self.estimate()
        return [op_id for op_id, mem in memory_at.items() if mem > self.l1_cache_bytes]

    def slicing_candidates(self) -> List[dict]:
        memory_at = self.estimate()
        candidates = []
        for op_id in self.exceeds_l1():
            node = self.dag.nodes.get(op_id)
            if node is None or not node.consumers:
                continue
            if not node.has_single_output:
                continue
            if node.output_tensor.size_bytes <= self.l1_cache_bytes:
                continue
            candidates.append({
                "op_id": op_id,
                "op_type": node.op_type,
                "output_size_bytes": node.output_tensor.size_bytes,
                "live_memory_bytes": memory_at.get(op_id, 0),
                "consumers": node.consumers,
            })
        return candidates

    def summary(self) -> str:
        memory_at = self.estimate()
        peak_op, peak_mem = self.get_peak()
        exceeds = self.exceeds_l1()
        candidates = self.slicing_candidates()
        lines = [
            "=== Peak Memory Estimation ===",
            f"  L1 Cache threshold: {self.l1_cache_bytes / 1024:.0f} KB",
            f"  Global peak: {peak_mem / 1024:.1f} KB at Op {peak_op}",
            f"  Ops exceeding L1: {len(exceeds)}",
            f"  Slicing candidates: {len(candidates)}",
            "",
            "  Per-op memory:",
        ]
        for op_id in sorted(memory_at.keys()):
            flag = " EXCEEDS L1" if op_id in exceeds else ""
            lines.append(f"    Op {op_id}: {memory_at[op_id] / 1024:.1f} KB{flag}")
        return "\n".join(lines)

    def _alias_state(
        self, tensor_idx_to_info: Dict[int, TensorInfo]
    ) -> Tuple[Dict[int, int], Set[int]]:
        """Group tensor indices that share one buffer, and flag the byte-free ones.

        Frontend liveness tracks SSA *values*; peak memory is about *buffers*.
        Two corrections bridge the gap, both of which torch-mlir output makes
        load-bearing rather than theoretical:

        1. ``tensor.empty`` is an uninitialized destination placeholder, not
           data. Its bytes belong to whichever destination-passing op
           materializes it, so it contributes none of its own. This also stops
           a CSE'd empty — torch-mlir emits a single one for N identical
           destinations — from holding a full-size buffer live across the whole
           span between its first and last consumer. (Observed on U-Net at
           512x512: one 32 MB empty shared by nine ``linalg.generic``
           destinations, live across 143 ops.)

        2. A destination-passing op's result aliases its destination operand —
           but only when this op is that destination's *sole* consumer. With
           more than one consumer, bufferization must copy for all but one, so
           the results really are separate buffers and merging them would
           understate the peak.

        Without these, ``empty -> fill -> insert_slice`` counts one padded
        activation three times. ``benchmarks.metrics`` used to see this only
        from the sliced side, where fused-producer credit hid it; the baseline
        kept the inflated number, so the two arms were not measured under the
        same memory model.
        """
        index_of: Dict[int, int] = {
            id(info): idx for idx, info in tensor_idx_to_info.items()
        }
        parent: Dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        producer_of: Dict[int, int] = {}
        no_allocation: Set[int] = set()
        # Tensors that are destination buffers rather than freshly computed
        # data: `tensor.empty` results, plus every DPS result that writes into
        # one. Traversal below is in topological order, so a chain member is
        # always added before the op that writes into it is examined.
        destination_buffers: Set[int] = set()
        for op_id in self.dag.topological_order():
            node = self.dag.nodes[op_id]
            for out in node.output_tensors:
                idx = index_of.get(id(out))
                if idx is None:
                    continue
                producer_of.setdefault(idx, op_id)
                if node.op_type.lower() == "tensor.empty":
                    no_allocation.add(idx)
                    destination_buffers.add(idx)

        for op_id in self.dag.topological_order():
            node = self.dag.nodes[op_id]
            if not _is_destination_passing(node.op_type) or not node.input_tensors:
                continue
            dest = node.input_tensors[-1]
            out = node.output_tensor
            # A real destination has the result's exact type; anything else is
            # an `ins` operand that happens to sit last, and aliasing it would
            # be wrong.
            if dest.shape != out.shape or dest.dtype != out.dtype:
                continue
            dest_idx = index_of.get(id(dest))
            out_idx = index_of.get(id(out))
            if dest_idx is None or out_idx is None:
                continue
            # Matching types are necessary but nowhere near sufficient: a
            # unary elementwise `linalg.generic` reads an `ins` of exactly the
            # result's type, and treating that as a destination would merge two
            # genuinely distinct buffers and understate the peak. Require the
            # operand to be a destination by provenance — every DPS destination
            # is created by `tensor.empty` and stays one along the chain that
            # writes into it — so an operand whose origin we cannot establish
            # is left alone.
            if dest_idx not in destination_buffers:
                continue
            dest_producer = producer_of.get(dest_idx)
            if dest_producer is None:
                continue  # a block argument; never written in place
            if len(self.dag.nodes[dest_producer].consumers) != 1:
                continue  # shared destination: bufferization must copy
            union(dest_idx, out_idx)
            destination_buffers.add(out_idx)

        representative = {idx: find(idx) for idx in parent}
        return representative, no_allocation

    def _build_tensor_map(self) -> Dict[int, TensorInfo]:
        tensor_map: Dict[int, TensorInfo] = {}
        next_idx = 0
        seen_ids: Dict[int, int] = {}
        for op_id in self.dag.topological_order():
            node = self.dag.nodes[op_id]
            for output_tensor in node.output_tensors:
                out_id = id(output_tensor)
                if out_id not in seen_ids:
                    seen_ids[out_id] = next_idx
                    tensor_map[next_idx] = output_tensor
                    next_idx += 1
            for inp in node.input_tensors:
                inp_id = id(inp)
                if inp_id not in seen_ids:
                    seen_ids[inp_id] = next_idx
                    tensor_map[next_idx] = inp
                    next_idx += 1
        return tensor_map
