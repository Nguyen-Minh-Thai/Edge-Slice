import math
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union, cast

from edgeslide.dag.graph import OperationDAG
from edgeslide.mlir_transform.errors import (
    DeallocationUnavailableError,
    ToolNotFoundError,
    require_tool,
)

ShapeBounds = Mapping[Union[int, str], Any]

_ALLOC_RE = re.compile(
    r"(?P<name>%[\w.$-]+)\s*=\s*(?:memref\.alloc\b|\"memref\.alloc\")"
)
_ALLOCA_RE = re.compile(
    r"(?P<name>%[\w.$-]+)\s*=\s*(?:memref\.alloca\b|\"memref\.alloca\")"
)
_DEALLOC_RE = re.compile(
    r"(?:memref\.dealloc\s+(?P<plain>%[\w.$-]+)|\"memref\.dealloc\"\((?P<quoted>%[\w.$-]+)\))"
)


class InferredSizeRefused(RuntimeError):
    """A strict analyzer met an allocation whose size it would have to estimate.

    Raised instead of returning a number that looks like a measurement and is
    not. See ``BufferizedMemoryAnalyzer.strict_sizes``.
    """


@dataclass
class MemRefAllocation:
    """Lifetime metadata for one bufferized memref allocation."""

    name: str
    memref_type: str
    size_bytes: int
    alloc_index: int
    dealloc_index: Optional[int] = None
    allocation_kind: str = "alloc"
    # Highest operation index that still reads the buffer or any alias of it.
    # A buffer is dead after this point even if its dealloc comes much later.
    last_use_index: Optional[int] = None
    # True when a ``?`` dimension was resolved from an inferred upper bound
    # rather than a literal extent, so size_bytes is an over-estimate.
    size_is_upper_bound: bool = False
    # Region attribution (proposal §5.2, alloc-attribution study). The stack of
    # ops enclosing this allocation, outermost first — e.g.
    # ``("func.func", "scf.forall")`` for a buffer materialised inside a tiled
    # loop body. Empty for an allocation at module scope.
    enclosing_ops: Tuple[str, ...] = ()

    @property
    def region_depth(self) -> int:
        """Nesting depth, counting the enclosing ``func.func`` as depth 1."""
        return len(self.enclosing_ops)

    @property
    def enclosing_loops(self) -> Tuple[str, ...]:
        """Just the loop-carrying ops in the enclosing stack."""
        return tuple(op for op in self.enclosing_ops if op in _LOOP_OPS)

    @property
    def inside_loop(self) -> bool:
        """True when the allocation is executed once per loop iteration."""
        return bool(self.enclosing_loops)

    @property
    def has_dynamic_extent(self) -> bool:
        """True when the memref type carries a ``?`` dimension."""
        return "?" in self.memref_type.split(",", 1)[0]

    # True when a ``memref.copy`` writes into this buffer. Set by the analyzer
    # after copies are collected, because it is a property of the module rather
    # than of the allocation line.
    is_copy_destination: bool = False

    @property
    def hoistable(self) -> bool:
        """True when this in-loop allocation could be lifted out of the loop.

        The condition ``--buffer-loop-hoisting`` applies: the allocation is
        inside a loop and its size does not depend on the loop induction
        variable. A fully static memref type is trivially loop-invariant; a
        dynamic one is reported as not-hoistable, which is conservative —
        the extent may still be invariant, but proving that needs the operand's
        definition, not just its type.
        """
        return self.inside_loop and not self.has_dynamic_extent

    @property
    def accumulates_across_iterations(self) -> bool:
        """True when each loop iteration adds a buffer that is never freed inside.

        This is the shape of the ResNet-50 tiled-arm regression: a producer
        fused into an ``scf.forall`` body gets a fresh ``memref.alloc`` per
        iteration, and if the matching ``memref.dealloc`` is parked outside the
        loop, live memory grows with the trip count instead of staying at one
        tile. ``size_bytes`` still counts the buffer once, so the reported peak
        is a *lower* bound whenever this is True.
        """
        if not self.inside_loop:
            return False
        if self.dealloc_index is None:
            # Nothing frees it at all, so every iteration adds one.
            return True
        if self.loop_exit_index is None:
            # Inside a loop whose region end could not be located: refuse to
            # claim accumulation rather than guess. `is None` rather than a
            # truthiness test, so a legitimate index 0 is not read as "unknown".
            return False
        return self.dealloc_index > self.loop_exit_index

    # Operation index at which the innermost enclosing loop's region closes.
    # None when the allocation is not inside a loop.
    loop_exit_index: Optional[int] = None


@dataclass(frozen=True)
class MemRefCopy:
    """One ``memref.copy``, recorded with the region that contains it.

    A non-zero count is the signature of bufferization failing to turn a
    ``tensor.extract_slice`` into a ``memref.subview``: instead of a view it
    allocates a fresh buffer and copies into it, which both adds a buffer to
    the live set and adds traffic the cost model never priced. Counted per arm
    so the tiled and untiled pipelines can be compared on it directly.
    """

    src: str
    dst: str
    op_index: int
    enclosing_ops: Tuple[str, ...] = ()

    @property
    def inside_loop(self) -> bool:
        return any(op in _LOOP_OPS for op in self.enclosing_ops)


@dataclass(frozen=True)
class BufferizedMemoryReport:
    """Peak memory report derived from bufferized MLIR allocation lifetimes."""

    memory_at: Dict[int, int]
    allocations: List[MemRefAllocation]
    # Same sweep but each buffer dies at its last use instead of at its
    # dealloc. ``--buffer-deallocation-pipeline`` parks deallocs at the end of
    # the function, which makes the dealloc-based peak degenerate to "sum of
    # every allocation" and hides any reuse a scheduler could exploit.
    memory_at_last_use: Dict[int, int] = field(default_factory=dict)
    # Every ``memref.copy`` in the bufferized module. See MemRefCopy.
    copies: List[MemRefCopy] = field(default_factory=list)
    # Bytes held in constant ``memref.global``s (the weights), and how many
    # there are. Reported alongside the peak, never added to it: see
    # _CONSTANT_GLOBAL_RE for why the distinction has to stay visible.
    constant_global_bytes: int = 0
    constant_global_count: int = 0

    def allocation_attribution(self) -> List[Dict[str, Any]]:
        """One row per allocation: site → bytes → inside-loop? → hoistable?

        This is the alloc-attribution table the memory-regression diagnosis is
        built on (proposal §5.2). Sorted by size, largest first, because the
        question it answers is "which buffers make up the peak".
        """
        rows = [
            {
                "name": alloc.name,
                "alloc_index": alloc.alloc_index,
                "bytes": alloc.size_bytes,
                "memref_type": alloc.memref_type,
                "kind": alloc.allocation_kind,
                "region_depth": alloc.region_depth,
                "enclosing_ops": list(alloc.enclosing_ops),
                "enclosing_loops": list(alloc.enclosing_loops),
                "inside_loop": alloc.inside_loop,
                "hoistable": alloc.hoistable,
                "accumulates_across_iterations": alloc.accumulates_across_iterations,
                "is_copy_destination": alloc.is_copy_destination,
                "dealloc_index": alloc.dealloc_index,
                "last_use_index": alloc.last_use_index,
                "size_is_upper_bound": alloc.size_is_upper_bound,
            }
            for alloc in self.allocations
        ]
        # Rows are heterogeneous by construction (ints, bools, lists), so the
        # sort key is narrowed explicitly rather than left to inference.
        rows.sort(key=lambda row: int(cast(int, row["bytes"])), reverse=True)
        return rows

    def attribution_totals(self) -> Dict[str, Any]:
        """Aggregate the attribution table into the numbers a figure reports."""
        in_loop = [a for a in self.allocations if a.inside_loop]
        hoistable = [a for a in in_loop if a.hoistable]
        accumulating = [a for a in in_loop if a.accumulates_across_iterations]
        copy_dsts = [a for a in self.allocations if a.is_copy_destination]
        peak_bytes = self.get_peak()[1]
        total_bytes = sum(a.size_bytes for a in self.allocations)
        return {
            "allocation_count": len(self.allocations),
            "total_bytes": total_bytes,
            # A buffer that exists only to receive a copy is bufferization
            # having declined to make a `tensor.extract_slice` into a
            # `memref.subview`. Counting them separates "tiling changed the
            # live set" from "tiling added copies", which are different
            # findings with different fixes.
            "copy_destination_count": len(copy_dsts),
            "copy_destination_bytes": sum(a.size_bytes for a in copy_dsts),
            # When these are equal, every free has been parked past the last
            # allocation and the dealloc-based peak carries no lifetime
            # information at all: it is the sum of every allocation. Reporting
            # it as a peak in that state is what made a tiled arm look like a
            # memory regression when its achievable peak was unchanged.
            "dealloc_peak_equals_total": abs(peak_bytes - total_bytes) < 1024,
            "in_loop_count": len(in_loop),
            "in_loop_bytes": sum(a.size_bytes for a in in_loop),
            "hoistable_count": len(hoistable),
            "hoistable_bytes": sum(a.size_bytes for a in hoistable),
            "accumulating_count": len(accumulating),
            "accumulating_bytes": sum(a.size_bytes for a in accumulating),
            "max_region_depth": max(
                (a.region_depth for a in self.allocations), default=0
            ),
            "copy_count": len(self.copies),
            "copy_in_loop_count": sum(1 for c in self.copies if c.inside_loop),
            "peak_dealloc_bytes": self.get_peak()[1],
            "peak_last_use_bytes": self.get_peak_last_use()[1],
            # Weights, held as constant globals in the binary's read-only data.
            # Deliberately NOT part of any peak above: those are heap
            # allocations. Present so that a comparison against measured RSS,
            # which does include this, can say by how much the two scopes
            # differ instead of quietly differing.
            "constant_global_bytes": self.constant_global_bytes,
            "constant_global_count": self.constant_global_count,
            "peak_plus_constants_bytes": (
                self.get_peak_last_use()[1] + self.constant_global_bytes
            ),
        }

    def get_peak(self) -> Tuple[int, int]:
        if not self.memory_at:
            return (-1, 0)
        peak_index = max(self.memory_at, key=lambda idx: self.memory_at[idx])
        return peak_index, self.memory_at[peak_index]

    def get_peak_last_use(self) -> Tuple[int, int]:
        """Peak achievable if every buffer were freed right after its last use."""
        if not self.memory_at_last_use:
            return (-1, 0)
        peak_index = max(self.memory_at_last_use, key=lambda idx: self.memory_at_last_use[idx])
        return peak_index, self.memory_at_last_use[peak_index]


def find_mlir_opt() -> Optional[str]:
    """Locate ``mlir-opt``, or return None when it is not installed.

    Delegates to the package-wide resolver so this path sees the same tool as
    the compile pipeline. Searching only ``$MLIR_OPT`` and ``PATH`` (as this
    did) made the bufferized estimator unavailable on hosts where mlir-opt is
    installed under an LLVM prefix but not linked onto PATH — while the
    pipeline compiled there fine.
    """
    try:
        return require_tool("mlir-opt")
    except ToolNotFoundError:
        return None


def run_one_shot_bufferize(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    mlir_opt: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    insert_deallocs: bool = True,
) -> str:
    """Run MLIR one-shot bufferization before allocation-lifetime analysis.

    Since LLVM ~17, ``--one-shot-bufferize`` no longer inserts
    ``memref.dealloc`` itself. Without ``--buffer-deallocation-pipeline`` every
    allocation lives forever, so the lifetime analysis degenerates to
    "peak = sum of all allocations" — a number that carries no lifetime
    information at all.

    ``insert_deallocs=True`` therefore means it: if the pipeline is unavailable
    this **raises** rather than bufferizing without it. The previous behaviour
    was to retry without deallocation and return successfully, which handed
    ``BufferizedMemoryAnalyzer`` — the primary peak-memory metric in this repo
    — a module whose peak was meaningless, with nothing in the result to say
    so. Pass ``insert_deallocs=False`` to ask for that deliberately; the
    analyzer's ``dealloc_peak_equals_total`` flag then reports the consequence.
    """
    mlir_opt_bin = mlir_opt or require_tool("mlir-opt")

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        mlir_opt_bin,
        str(input_path),
        "--one-shot-bufferize",
        *(extra_args or []),
    ]
    cmd = list(base_cmd)
    if insert_deallocs:
        cmd.append("--buffer-deallocation-pipeline")
    cmd += ["-o", str(output_path)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return str(output_path)

    stderr = result.stderr or ""
    if insert_deallocs and "buffer-deallocation-pipeline" in stderr:
        raise DeallocationUnavailableError(
            "this mlir-opt does not support --buffer-deallocation-pipeline. "
            "Bufferizing without it produces a module in which nothing is "
            "ever freed, so its peak is the sum of every allocation and "
            "measures nothing. Install mlir-opt >= 17, or pass "
            "insert_deallocs=False to request that deliberately.\n"
            f"stderr: {stderr}"
        )
    raise RuntimeError(
        "mlir-opt --one-shot-bufferize failed:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {stderr}"
    )



# Ops whose result is a view of an existing buffer rather than a fresh one. A
# buffer stays live for as long as any of these aliases is still read.
_ALIAS_OPS = (
    "memref.subview", "memref.cast", "memref.reinterpret_cast",
    "memref.expand_shape", "memref.collapse_shape", "memref.view",
    "memref.transpose",
)
_SSA_RE = re.compile(r"%[\w.$-]+")
_RESULT_RE = re.compile(r"^\s*(?P<name>%[\w.$-]+)\s*=")
_ALLOC_OPERANDS_RE = re.compile(r"memref\.alloca?\((?P<operands>[^)]*)\)")
_AFFINE_MAP_DECL_RE = re.compile(
    r"^(?P<alias>#\w+)\s*=\s*affine_map<\((?P<dims>[^)]*)\)(?:\[[^\]]*\])?"
    r"\s*->\s*\((?P<results>.*)\)>\s*$"
)
_AFFINE_OP_RE = re.compile(
    r"affine\.(?P<kind>min|max|apply)\s+(?:(?P<alias>#\w+)"
    r"|affine_map<\((?P<inline_dims>[^)]*)\)(?:\[[^\]]*\])?"
    r"\s*->\s*\((?P<inline>[^)]*)\)>)"
)
_INDEX_CONST_RE = re.compile(r"arith\.constant\s+(?P<value>\d+)\s*:\s*index")
_COPY_RE = re.compile(r"memref\.copy\s+(?P<src>%[\w.$-]+)\s*,\s*(?P<dst>%[\w.$-]+)")
# Constant `memref.global`s hold the model's weights. They are not allocations
# — they live in the binary's read-only data, not the heap — so they are
# correctly absent from the allocation peak. But the measured RSS of the same
# binary *does* include them, so the two numbers have different scopes and are
# not comparable until the difference is stated. Hence: count them, report
# them, never fold them into the peak.
_CONSTANT_GLOBAL_RE = re.compile(
    r'memref\.global\s+"[^"]*"\s+constant\s+(?P<name>@[\w.$-]+)\s*:\s*'
    r"(?P<type>memref<[^>]*>)"
)
# The op that owns a region, taken from the first `dialect.op` token on the
# line that opens it. `%3 = scf.for ...` and `scf.forall ... {` both resolve.
_REGION_OP_RE = re.compile(r"(?:^|\s)(?P<op>[a-z_][\w]*\.[a-zA-Z_][\w.]*)")
# Region-carrying ops whose body executes more than once. An allocation nested
# in one of these is paid per iteration unless it is hoisted, which is the
# whole subject of the alloc-attribution study (proposal §5.2).
_LOOP_OPS = frozenset({
    "scf.for", "scf.forall", "scf.parallel", "scf.while",
    "affine.for", "affine.parallel",
})


def _scan_braces(line: str) -> Tuple[int, int]:
    """Split a line's braces into (closes of outer regions, new regions opened).

    Counting ``{`` and ``}`` separately — as this module used to — miscounts
    every line carrying an attribute dictionary: ``memref.alloc() {alignment =
    64 : i64}`` reads as one close followed by one open, which pops the
    enclosing region and pushes the alloc line in its place. Depth is
    preserved, so the bug is invisible in a depth counter, but region identity
    is not, and identity is what scope resolution keys on. Braces that open and
    close within the same line are paired off here and never reach the region
    stack. Quoted strings are skipped so a brace inside a symbol name does not
    count.
    """
    same_line: int = 0
    closes = 0
    opens = 0
    in_string = False
    index = 0
    length = len(line)
    while index < length:
        char = line[index]
        if in_string:
            if char == "\\":
                index += 2
                continue
            if char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            same_line += 1
            opens += 1
        elif char == "}":
            if same_line:
                same_line -= 1
                opens -= 1
            else:
                closes += 1
        index += 1
    return closes, opens


def _region_op_name(line: str) -> str:
    """The dialect op that opens a region on *line* ("" when undeterminable)."""
    body = line.split("=", 1)[1] if _RESULT_RE.match(line) else line
    match = _REGION_OP_RE.search(body)
    return match.group("op") if match else ""


def _region_stacks(operation_lines) -> Dict[int, Tuple[str, ...]]:
    """Map every operation index to the stack of op names enclosing it.

    Index 0 of the tuple is the outermost region (normally ``func.func``); the
    last entry is the immediately enclosing one. A line that opens a region is
    reported *outside* that region — it is the op, not its body.
    """
    stacks: Dict[int, Tuple[str, ...]] = {}
    stack: List[str] = []
    for op_index, line in operation_lines:
        closes, opens = _scan_braces(line)
        for _ in range(closes):
            if stack:
                stack.pop()
        stacks[op_index] = tuple(stack)
        for _ in range(opens):
            stack.append(_region_op_name(line))
    return stacks


def _loop_exit_indices(operation_lines) -> Dict[int, int]:
    """For each operation index, where its innermost enclosing loop region ends.

    Needed to tell "freed inside the loop body" (one tile buffer live at a
    time) from "freed after the loop" (one buffer per iteration accumulating),
    which is the distinction the alloc-attribution study turns on.
    """
    exits: Dict[int, int] = {}
    # (op_name, [indices seen inside this region])
    open_stack: List[Tuple[str, List[int]]] = []
    for op_index, line in operation_lines:
        closes, opens = _scan_braces(line)
        for _ in range(closes):
            if not open_stack:
                continue
            op_name, members = open_stack.pop()
            if op_name in _LOOP_OPS:
                for member in members:
                    exits.setdefault(member, op_index)
            # An inner region's members are also members of the outer one.
            if open_stack:
                open_stack[-1][1].extend(members)
        if open_stack:
            open_stack[-1][1].append(op_index)
        for _ in range(opens):
            open_stack.append((_region_op_name(line), []))
    return exits
_AFFINE_TERM_SCAN = re.compile(
    r"(?P<sign>[+-])?(?P<body>[ds]\d+\*-?\d+|-?\d+\*[ds]\d+|[ds]\d+|\d+)"
)


def _affine_linear_upper_bound(
    expr: str, operand_uppers: List[Optional[int]], num_dims: int
) -> Optional[int]:
    """Upper bound of a linear affine expression, or None if not derivable.

    Index values in bufferized IR are loop bounds and extents, so every symbol
    is assumed non-negative; a term with a negative coefficient is therefore
    maximised at zero. Anything non-linear (``ceildiv``/``floordiv``/``mod``)
    or not fully consumed by the term scanner yields None rather than a guess.
    """
    if any(token in expr for token in ("ceildiv", "floordiv", "mod", "(")):
        return None
    compact = re.sub(r"\s+", "", expr)
    if not compact:
        return None

    total = 0
    position_in_expr = 0
    for match in _AFFINE_TERM_SCAN.finditer(compact):
        if match.start() != position_in_expr:
            return None  # a gap means the scanner did not understand the term
        position_in_expr = match.end()
        sign = -1 if match.group("sign") == "-" else 1
        body = match.group("body")

        if "*" in body:
            left, right = body.split("*", 1)
            if left[:1] in ("d", "s"):
                symbol, coeff = left, int(right)
            else:
                symbol, coeff = right, int(left)
        elif body[:1] in ("d", "s"):
            symbol, coeff = body, 1
        else:
            total += sign * int(body)
            continue

        factor = sign * coeff
        if factor <= 0:
            continue  # maximised at the symbol's lower bound of zero
        # Operands are laid out dims-first, then symbols, so sN indexes past
        # every dN rather than restarting at zero.
        index = int(symbol[1:]) + (num_dims if symbol[0] == "s" else 0)
        if index >= len(operand_uppers):
            return None
        upper = operand_uppers[index]
        if upper is None:
            return None
        total += factor * upper

    if position_in_expr != len(compact):
        return None
    return total


_FORALL_DOMAIN_RE = re.compile(
    r"scf\.forall\s*\((?P<ivs>[^)]*)\)\s*in\s*\((?P<extents>[^)]*)\)"
)


def _forall_induction_domains(operation_lines) -> Dict[str, List[Tuple[int, int]]]:
    """Iteration-space extent of each ``scf.forall`` induction variable.

    ``scf.forall (%i, %j) in (256, 2)`` gives ``%i in [0, 256)`` and
    ``%j in [0, 2)``. Keyed by name with the defining op index, like the other
    bound tables here, because the interpreter reuses names across sibling
    regions.

    Only the all-literal form is read. A dynamic or lower-bounded forall
    returns nothing for its variables, which leaves any bound derived from
    them an upper bound rather than a proven extent -- the safe direction.
    """
    domains: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for op_index, line in operation_lines:
        match = _FORALL_DOMAIN_RE.search(line)
        if match is None:
            continue
        ivs = [v.strip() for v in match.group("ivs").split(",") if v.strip()]
        extents = [e.strip() for e in match.group("extents").split(",") if e.strip()]
        if len(ivs) != len(extents):
            continue
        for iv, extent in zip(ivs, extents):
            if iv.startswith("%") and extent.isdigit() and int(extent) > 0:
                domains[iv].append((op_index, int(extent)))
    return domains


def _affine_eval(expr: str, value: int) -> Optional[int]:
    """Evaluate a single-dimension linear affine expression at ``d0 = value``.

    None for anything the term scanner does not fully consume, or that
    mentions a symbol or a second dimension -- the caller then falls back to
    the sound upper bound instead of a proven one.
    """
    if any(token in expr for token in ("ceildiv", "floordiv", "mod", "(", "s0", "d1")):
        return None
    compact = re.sub(r"\s+", "", expr)
    if not compact:
        return None
    total = 0
    position = 0
    for match in _AFFINE_TERM_SCAN.finditer(compact):
        if match.start() != position:
            return None
        position = match.end()
        sign = -1 if match.group("sign") == "-" else 1
        body = match.group("body")
        if "*" in body:
            left, right = body.split("*", 1)
            coeff = int(right) if left[:1] == "d" else int(left)
            total += sign * coeff * value
        elif body[:1] == "d":
            total += sign * value
        elif body[:1] == "s":
            return None
        else:
            total += sign * int(body)
    if position != len(compact):
        return None
    return total


def _attained_min_maximum(results: List[str], extent: int) -> Optional[int]:
    """max over d0 in [0, extent) of min(results), or None if not derivable.

    A min of affine functions is concave, so over a finite integer range its
    maximum sits at an endpoint or next to a pairwise crossing. Evaluating
    that candidate set is exact, not a bound -- which is the whole point: a
    peak built on an extent that is *attained* is a measurement, while one
    built on an extent that is merely *not exceeded* is an estimate.
    """
    if not results or extent <= 0:
        return None
    candidates = {0, extent - 1}
    parsed = []
    for expr in results:
        at_zero = _affine_eval(expr, 0)
        at_one = _affine_eval(expr, 1)
        if at_zero is None or at_one is None:
            return None
        parsed.append((at_one - at_zero, at_zero))  # (slope, intercept)
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            (m1, c1), (m2, c2) = parsed[i], parsed[j]
            if m1 == m2:
                continue
            crossing = (c2 - c1) / (m1 - m2)
            for near in (math.floor(crossing), math.ceil(crossing)):
                if 0 <= near < extent:
                    candidates.add(int(near))
    best = None
    for d0 in candidates:
        value = min(slope * d0 + intercept for slope, intercept in parsed)
        if best is None or value > best:
            best = value
    return best


def _infer_index_upper_bounds(
    operation_lines,
) -> Dict[str, List[Tuple[int, int, bool]]]:
    """Upper bounds for SSA index values, as (op_index, bound) per name.

    Bounds are kept per definition site because bufferized IR reuses the same
    SSA name in sibling regions; a use must resolve against the nearest
    preceding definition, not the last one in the file.
    """
    maps: Dict[str, Tuple[int, List[str]]] = {}
    for _, line in operation_lines:
        decl = _AFFINE_MAP_DECL_RE.match(line)
        if decl is not None:
            dims = [d for d in decl.group("dims").split(",") if d.strip()]
            maps[decl.group("alias")] = (
                len(dims),
                _split_affine_results(decl.group("results")),
            )

    bounds: Dict[str, List[Tuple[int, int, bool]]] = defaultdict(list)
    domains = _forall_induction_domains(operation_lines)

    def latest(name: str, before: int) -> Optional[int]:
        candidates = [b for idx, b, _ in bounds.get(name, []) if idx < before]
        return candidates[-1] if candidates else None

    def domain_of(line: str, before: int) -> Optional[int]:
        """Extent of the single forall index this affine op reads, if unique."""
        seen = {
            name
            for name in _SSA_RE.findall(line)
            if any(idx < before for idx, _ in domains.get(name, []))
        }
        if len(seen) != 1:
            return None
        name = seen.pop()
        extents = [e for idx, e in domains[name] if idx < before]
        return extents[-1] if extents else None

    for op_index, line in operation_lines:
        result = _RESULT_RE.match(line)
        if result is None:
            continue
        name = result.group("name")

        const = _INDEX_CONST_RE.search(line)
        if const is not None:
            bounds[name].append((op_index, int(const.group("value")), True))
            continue

        affine_op = _AFFINE_OP_RE.search(line)
        if affine_op is None:
            continue
        if affine_op.group("alias"):
            entry = maps.get(affine_op.group("alias"))
        else:
            inline_dims = [
                d for d in (affine_op.group("inline_dims") or "").split(",") if d.strip()
            ]
            entry = (inline_dims and len(inline_dims) or 0,
                     _split_affine_results(affine_op.group("inline") or ""))
        if not entry or not entry[1]:
            continue
        num_dims, results = entry

        operand_uppers = [
            latest(operand, op_index)
            for operand in _SSA_RE.findall(line[affine_op.end():])
        ]
        values = [
            _affine_linear_upper_bound(expr, operand_uppers, num_dims)
            for expr in results
        ]
        candidates = [value for value in values if value is not None]
        kind = affine_op.group("kind")
        if kind == "min" and candidates:
            # min() can never exceed the tightest bound among its results.
            bound = min(candidates)
            # ...but for a peak we want the extent the loop actually reaches,
            # not one it merely never exceeds. Over a known iteration domain
            # the maximum of a min-of-affines is computable exactly, and when
            # it agrees with the bound the size is a fact rather than an
            # estimate. `affine.min(-5*d0 + 8, 5)` over d0 in [0,2) takes the
            # values 5 and 3, so 5 is attained and the allocation sized by it
            # is measured, not approximated.
            extent = domain_of(line, op_index)
            attained = (
                _attained_min_maximum(results, extent) if extent is not None else None
            )
            bounds[name].append((op_index, bound, attained == bound))
        elif kind == "max" and candidates and len(candidates) == len(values):
            # max() needs every branch bounded; one unbounded branch is enough
            # to make the result unbounded.
            bounds[name].append((op_index, max(candidates), False))
        elif kind == "apply" and len(results) == 1 and candidates:
            extent = domain_of(line, op_index)
            exact = (
                extent is not None
                and _attained_min_maximum(results, extent) == candidates[0]
            )
            bounds[name].append((op_index, candidates[0], exact))
    return bounds


def _split_affine_results(results: str) -> List[str]:
    """Split the comma-separated result list of an affine map."""
    parts, depth, current = [], 0, ""
    for char in results:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return parts


def _region_scoped_definitions(operation_lines):
    """Index every SSA definition together with the point its region closes.

    MLIR recycles short names such as ``%67`` in sibling regions, so a use can
    only be matched to the definition that is still in scope at that point.
    Matching by name alone attributes unrelated regions' uses to the same
    buffer and inflates its live range to the whole function.
    """
    region_close: Dict[int, int] = {}
    open_regions: List[int] = []
    definitions: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    pending: List[Tuple[str, int, Optional[int]]] = []

    total = len(operation_lines)
    for op_index, line in operation_lines:
        closes, opens = _scan_braces(line)
        for _ in range(closes):
            if open_regions:
                region_close[open_regions.pop()] = op_index
        result = _RESULT_RE.match(line)
        if result is not None:
            enclosing = open_regions[-1] if open_regions else None
            pending.append((result.group("name"), op_index, enclosing))
        for _ in range(opens):
            open_regions.append(op_index)

    for name, op_index, enclosing in pending:
        scope_end = total if enclosing is None else region_close.get(enclosing, total)
        definitions[name].append((op_index, scope_end))
    for entries in definitions.values():
        entries.sort()
    return definitions


def _resolve_definition(
    definitions, name: str, use_index: int
) -> Optional[Tuple[int, int]]:
    """Definition instance of *name* visible at *use_index*, nearest first."""
    best = None
    for def_index, scope_end in definitions.get(name, []):
        if def_index < use_index <= scope_end:
            best = (def_index, scope_end)
    return best


def _alias_roots(operation_lines, definitions, roots):
    """Map every SSA definition instance that views a root buffer to that root.

    Keyed by ``(name, def_index)`` rather than by name, so an alias created in
    one loop body never claims a same-named value in a sibling body.
    """
    owner: Dict[Tuple[str, int], str] = {}
    for name, def_index in roots:
        owner[(name, def_index)] = name

    for op_index, line in operation_lines:
        if not any(alias_op in line for alias_op in _ALIAS_OPS):
            continue
        result = _RESULT_RE.match(line)
        if result is None:
            continue
        for operand in _SSA_RE.findall(line[result.end():]):
            binding = _resolve_definition(definitions, operand, op_index)
            if binding is None:
                continue
            root = owner.get((operand, binding[0]))
            if root is not None:
                owner.setdefault((result.group("name"), op_index), root)
                break
    return owner


class BufferizedMemoryAnalyzer:
    """Estimate runtime memory from bufferized MLIR alloc/dealloc lifetimes.

    This complements ``LivenessAnalyzer``.  ``LivenessAnalyzer`` works on the
    frontend tensor DAG and is useful before MLIR lowering exists; this analyzer
    should be used when an MLIR payload can be bufferized.  It tracks concrete
    ``memref.alloc``/``memref.dealloc`` lifetimes and therefore reflects memory
    reuse introduced by bufferization instead of assuming every tensor remains a
    distinct live allocation.
    """

    def __init__(
        self,
        bufferized_mlir: str,
        *,
        shape_bounds: Optional[ShapeBounds] = None,
        allow_inferred_sizes: bool = True,
    ):
        """
        ``allow_inferred_sizes`` controls what happens to an allocation whose
        memref type carries a ``?`` extent. When True (the default, for
        backward compatibility) the extent is derived from an affine upper
        bound found in the surrounding IR and the allocation is flagged
        ``size_is_upper_bound``. When False, such an allocation raises.

        The flag exists because an inferred extent makes the resulting peak an
        *estimate* wearing the same type as a measurement. Every reported
        number in this repo comes from statically-shaped payloads where no
        inference is needed, so a run that silently starts inferring has
        changed what its output means. ``strict_sizes()`` is the constructor
        for a reporting path: it refuses rather than approximates.
        """
        self.bufferized_mlir = bufferized_mlir
        self.shape_bounds = shape_bounds or {}
        self.allow_inferred_sizes = allow_inferred_sizes
        self.allocations: List[MemRefAllocation] = []

    @classmethod
    def strict_sizes(
        cls,
        bufferized_mlir: str,
        *,
        shape_bounds: Optional[ShapeBounds] = None,
    ) -> "BufferizedMemoryAnalyzer":
        """Analyzer that raises on any allocation it would have to estimate."""
        return cls(
            bufferized_mlir,
            shape_bounds=shape_bounds,
            allow_inferred_sizes=False,
        )

    @classmethod
    def from_bufferized_file(
        cls,
        path: Union[str, Path],
        *,
        shape_bounds: Optional[ShapeBounds] = None,
        allow_inferred_sizes: bool = True,
    ) -> "BufferizedMemoryAnalyzer":
        """Analyzer over a bufferized file.

        ``allow_inferred_sizes=False`` for any path whose output is reported:
        see ``strict_sizes``. It defaults True so that exploratory and
        diagnostic use is not blocked by an allocation nobody is going to
        publish a number about.
        """
        return cls(
            Path(path).read_text(),
            shape_bounds=shape_bounds,
            allow_inferred_sizes=allow_inferred_sizes,
        )

    @classmethod
    def from_tensor_mlir_file(
        cls,
        input_path: Union[str, Path],
        *,
        output_path: Optional[Union[str, Path]] = None,
        mlir_opt: Optional[str] = None,
        extra_bufferize_args: Optional[List[str]] = None,
        shape_bounds: Optional[ShapeBounds] = None,
    ) -> "BufferizedMemoryAnalyzer":
        input_path = Path(input_path)
        output_path = Path(output_path or input_path.with_suffix(".bufferized.mlir"))
        bufferized_path = run_one_shot_bufferize(
            input_path,
            output_path,
            mlir_opt=mlir_opt,
            extra_args=extra_bufferize_args,
        )
        return cls.from_bufferized_file(bufferized_path, shape_bounds=shape_bounds)

    def analyze(self) -> BufferizedMemoryReport:
        lines = self._operation_lines()
        self.allocations = self._collect_allocations()
        self._assign_last_use(self.allocations, lines)
        copies = self._collect_copies(lines)
        copy_destinations = {copy.dst for copy in copies}
        for allocation in self.allocations:
            allocation.is_copy_destination = allocation.name in copy_destinations
        global_bytes, global_count = self._collect_constant_globals(lines)
        return BufferizedMemoryReport(
            memory_at=self._memory_timeline(self.allocations),
            allocations=self.allocations,
            memory_at_last_use=self._memory_timeline(
                self.allocations, use_last_use=True
            ),
            copies=copies,
            constant_global_bytes=global_bytes,
            constant_global_count=global_count,
        )

    @staticmethod
    def _collect_constant_globals(operation_lines) -> Tuple[int, int]:
        """Total bytes and count of constant ``memref.global``s (the weights)."""
        total = 0
        count = 0
        for _, line in operation_lines:
            match = _CONSTANT_GLOBAL_RE.search(line)
            if match is None:
                continue
            # A constant global is always statically shaped, so no bounds are
            # needed; a dynamic one would raise rather than be estimated.
            total += _memref_size_bytes(
                match.group("type"),
                alloc_name=match.group("name"),
                shape_bounds={},
            )
            count += 1
        return total, count

    @staticmethod
    def _collect_copies(operation_lines) -> List[MemRefCopy]:
        """Every ``memref.copy``, tagged with the region that contains it."""
        stacks = _region_stacks(operation_lines)
        copies: List[MemRefCopy] = []
        for op_index, line in operation_lines:
            match = _COPY_RE.search(line)
            if match is None:
                continue
            copies.append(
                MemRefCopy(
                    src=match.group("src"),
                    dst=match.group("dst"),
                    op_index=op_index,
                    enclosing_ops=stacks.get(op_index, ()),
                )
            )
        return copies

    def estimate(self) -> Dict[int, int]:
        """Memory in bytes after each allocation/deallocation program point."""
        return self.analyze().memory_at

    def get_peak(self) -> Tuple[int, int]:
        return self.analyze().get_peak()

    def get_peak_last_use(self) -> Tuple[int, int]:
        """Peak if every buffer were freed right after its last use."""
        return self.analyze().get_peak_last_use()

    def _collect_allocations(self) -> List[MemRefAllocation]:
        allocations: List[MemRefAllocation] = []
        by_name: Dict[str, List[MemRefAllocation]] = defaultdict(list)
        operation_lines = self._operation_lines()
        index_bounds = _infer_index_upper_bounds(operation_lines)
        stacks = _region_stacks(operation_lines)
        loop_exits = _loop_exit_indices(operation_lines)

        for op_index, line in operation_lines:
            alloc_match = _ALLOC_RE.search(line)
            alloca_match = _ALLOCA_RE.search(line)
            if alloc_match or alloca_match:
                match = alloc_match or alloca_match
                assert match is not None
                name = match.group("name")
                memref_type = _extract_memref_type(line)
                if memref_type is None:
                    raise ValueError(
                        f"Cannot determine memref type for allocation: {line}"
                    )
                inferred, approximate = self._infer_alloc_bounds(
                    line, memref_type, index_bounds, op_index
                )
                # An explicit caller-supplied bound always wins over inference.
                explicit = dict(self.shape_bounds.get(name) or {})
                # Refuse only when the size would *actually* rest on inference:
                # the same condition that sets `size_is_upper_bound` below. A
                # caller who supplied the extent has made it a fact again, so
                # strict mode has nothing to object to.
                unproven = {d for d in approximate if d not in explicit}
                if unproven and not self.allow_inferred_sizes:
                    raise InferredSizeRefused(
                        f"allocation {name} has type {memref_type}, whose "
                        f"dynamic extent(s) {sorted(unproven)} were resolved "
                        "from an affine upper bound that the iteration domain "
                        "is not known to reach. That makes the resulting peak "
                        "an estimate, not a measurement. Supply shape_bounds "
                        "for this allocation, or use the default constructor "
                        "to accept the approximation explicitly."
                    )
                merged = {**inferred, **explicit}
                bounds = dict(self.shape_bounds)
                if merged:
                    bounds[name] = merged
                size_bytes = _memref_size_bytes(
                    memref_type,
                    alloc_name=name,
                    shape_bounds=bounds,
                )
                enclosing = stacks.get(op_index, ())
                allocation = MemRefAllocation(
                    name=name,
                    memref_type=memref_type,
                    size_bytes=size_bytes,
                    alloc_index=op_index,
                    allocation_kind="alloc" if alloc_match else "alloca",
                    size_is_upper_bound=bool(unproven),
                    enclosing_ops=enclosing,
                    loop_exit_index=loop_exits.get(op_index),
                )
                allocations.append(allocation)
                by_name[name].append(allocation)

            dealloc_match = _DEALLOC_RE.search(line)
            if dealloc_match:
                name = dealloc_match.group("plain") or dealloc_match.group("quoted")
                open_alloc = self._latest_open_allocation(by_name.get(name, []))
                if open_alloc is not None:
                    open_alloc.dealloc_index = op_index

        return allocations

    @staticmethod
    def _assign_last_use(
        allocations: List[MemRefAllocation], operation_lines
    ) -> None:
        """Record the last operation that reads each buffer or an alias of it.

        Resolution is region-scoped: a use counts only when it binds to the
        definition instance that is actually in scope there.
        """
        if not allocations:
            return
        definitions = _region_scoped_definitions(operation_lines)
        roots = [(a.name, a.alloc_index) for a in allocations]
        owner = _alias_roots(operation_lines, definitions, roots)

        # A root buffer can be allocated more than once under the same name in
        # sibling regions, so track last use per allocation instance.
        last_use: Dict[Tuple[str, int], int] = {key: key[1] for key in roots}
        for op_index, line in operation_lines:
            if _DEALLOC_RE.search(line):
                continue
            for operand in set(_SSA_RE.findall(line)):
                binding = _resolve_definition(definitions, operand, op_index)
                if binding is None:
                    continue
                instance = (operand, binding[0])
                root_name = owner.get(instance)
                if root_name is None:
                    continue
                for key in last_use:
                    if key[0] == root_name and key[1] <= binding[0]:
                        last_use[key] = max(last_use[key], op_index)
                        break

        for allocation in allocations:
            end = last_use.get((allocation.name, allocation.alloc_index),
                               allocation.alloc_index)
            # Never claim a buffer outlives its dealloc; that would invent
            # pressure the IR does not have.
            if allocation.dealloc_index is not None:
                end = min(end, allocation.dealloc_index)
            allocation.last_use_index = max(end, allocation.alloc_index)

    @staticmethod
    def _infer_alloc_bounds(
        line: str,
        memref_type: str,
        index_bounds: Dict[str, List[Tuple[int, int, bool]]],
        op_index: int,
    ) -> Tuple[Dict[int, int], Set[int]]:
        """Each ``?`` dimension's extent, plus which of them are only bounds.

        Dynamic extents are passed as leading operands of ``memref.alloc`` in
        the order the ``?`` dimensions appear in the type.

        The second return value is the subset whose extent is an upper bound
        the loop is not known to reach. An extent proven attained over the
        iteration domain is left out of it: that size is the one the program
        actually allocates, so a peak built on it is measured. See
        ``_attained_min_maximum``.
        """
        content = memref_type[len("memref<") : -1]
        dims = content.split(",", 1)[0].split("x")[:-1]
        dynamic_positions = [i for i, dim in enumerate(dims) if dim == "?"]
        if not dynamic_positions:
            return {}, set()
        operand_match = _ALLOC_OPERANDS_RE.search(line)
        if operand_match is None:
            return {}, set()
        operands = _SSA_RE.findall(operand_match.group("operands"))
        bounds: Dict[int, int] = {}
        approximate: Set[int] = set()
        for position, operand in zip(dynamic_positions, operands):
            candidates = [
                (value, exact)
                for idx, value, exact in index_bounds.get(operand, [])
                if idx < op_index
            ]
            if not candidates:
                continue
            value, exact = candidates[-1]
            bounds[position] = value
            if not exact:
                approximate.add(position)
        return bounds, approximate

    @staticmethod
    def _latest_open_allocation(
        allocations: List[MemRefAllocation],
    ) -> Optional[MemRefAllocation]:
        for allocation in reversed(allocations):
            if allocation.dealloc_index is None:
                return allocation
        return None

    def _operation_lines(self) -> List[Tuple[int, str]]:
        operations: List[Tuple[int, str]] = []
        for line in self.bufferized_mlir.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            operations.append((len(operations), stripped))
        return operations

    @staticmethod
    def _memory_timeline(
        allocations: List[MemRefAllocation], *, use_last_use: bool = False
    ) -> Dict[int, int]:
        events: Dict[int, int] = defaultdict(int)
        last_point = 0
        for allocation in allocations:
            events[allocation.alloc_index] += allocation.size_bytes
            last_point = max(last_point, allocation.alloc_index)
            end = allocation.dealloc_index
            if use_last_use and allocation.last_use_index is not None:
                end = allocation.last_use_index
            if end is not None:
                events[end] -= allocation.size_bytes
                last_point = max(last_point, end)

        current = 0
        memory_at: Dict[int, int] = {}
        for point in range(last_point + 1):
            current += events.get(point, 0)
            memory_at[point] = current
        return memory_at

    def summary(self) -> str:
        report = self.analyze()
        peak_index, peak_bytes = report.get_peak()
        live_index, live_bytes = report.get_peak_last_use()
        total_bytes = sum(alloc.size_bytes for alloc in report.allocations)
        estimated = sum(1 for alloc in report.allocations if alloc.size_is_upper_bound)
        lines = [
            "=== Bufferized MLIR Memory Estimation ===",
            f"  Allocations: {len(report.allocations)}",
            f"  Peak (dealloc placement): {peak_bytes / 1024:.1f} KB "
            f"at program point {peak_index}",
            f"  Peak (last use):          {live_bytes / 1024:.1f} KB "
            f"at program point {live_index}",
        ]
        if peak_bytes and peak_bytes == total_bytes:
            lines.append(
                "  Note: the dealloc-based peak equals the sum of every allocation, "
                "so deallocation is parked at function exit and that number is an "
                "upper bound. Use the last-use peak to judge reuse."
            )
        if estimated:
            lines.append(
                f"  Note: {estimated} allocation(s) have a dynamic extent resolved "
                "from an inferred upper bound; their size is an over-estimate."
            )
        lines.extend(["", "  Allocation lifetimes:"])
        for alloc in report.allocations:
            end = alloc.dealloc_index if alloc.dealloc_index is not None else "function-exit"
            last_use = alloc.last_use_index if alloc.last_use_index is not None else "?"
            marker = " ~" if alloc.size_is_upper_bound else ""
            lines.append(
                f"    {alloc.name}: {alloc.size_bytes / 1024:.1f} KB{marker} "
                f"[{alloc.alloc_index}, last-use {last_use}, dealloc {end}] "
                f"{alloc.memref_type}"
            )
        return "\n".join(lines)


def _extract_memref_type(line: str) -> Optional[str]:
    start = line.find("memref<")
    if start < 0:
        return None

    depth = 0
    for offset, char in enumerate(line[start:], start=start):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                return line[start : offset + 1]
    return None


def _memref_size_bytes(
    memref_type: str,
    *,
    alloc_name: str,
    shape_bounds: ShapeBounds,
) -> int:
    content = memref_type[len("memref<") : -1]
    shape_and_dtype = content.split(",", 1)[0]
    parts = shape_and_dtype.split("x")
    if not parts:
        raise ValueError(f"Invalid memref type: {memref_type}")

    dtype = parts[-1]
    dims = parts[:-1]
    dtype_sizes = {
        "f32": 4,
        "f16": 2,
        "bf16": 2,
        "f64": 8,
        "i1": 1,
        "i8": 1,
        "i16": 2,
        "i32": 4,
        "i64": 8,
        "index": 8,
    }
    if dtype not in dtype_sizes:
        # Defaulting to 4 would silently mis-size every buffer of the unknown
        # type, and the resulting peak would still look like a measurement.
        raise ValueError(
            f"Unknown element type {dtype!r} in {memref_type} (allocation "
            f"{alloc_name}): refusing to guess its width. Add it to "
            "dtype_sizes if it is real."
        )
    elem_size = dtype_sizes[dtype]

    resolved_dims = []
    for index, dim in enumerate(dims):
        if dim == "?":
            resolved_dims.append(_resolve_memref_dynamic_dim(alloc_name, index, shape_bounds))
        else:
            try:
                value = int(dim)
            except ValueError as exc:
                raise ValueError(f"Unsupported memref dimension {dim!r} in {memref_type}") from exc
            if value < 0:
                raise ValueError(f"Invalid negative memref dimension {value} in {memref_type}")
            resolved_dims.append(value)

    return math.prod(resolved_dims or [1]) * elem_size


def _resolve_memref_dynamic_dim(
    alloc_name: str,
    index: int,
    shape_bounds: ShapeBounds,
) -> int:
    for alloc_key in (alloc_name, alloc_name.lstrip("%")):
        allocation_bounds = shape_bounds.get(alloc_key)
        if isinstance(allocation_bounds, Mapping):
            value = _lookup_dim_bound(allocation_bounds, index)
            if value is not None:
                return value

    value = _lookup_dim_bound(shape_bounds, index)
    if value is not None:
        return value


    raise ValueError(
        f"Allocation {alloc_name} has dynamic memref dimension {index}; "
        "provide shape_bounds to estimate bufferized memory safely"
    )


def _lookup_dim_bound(bounds: Mapping[Union[int, str], Any], index: int) -> Optional[int]:
    for key in (index, str(index), f"dim_{index}"):
        if key in bounds:
            value = int(bounds[key])
            if value <= 0:
                raise ValueError(f"Dynamic dimension {index} has non-positive bound {value}")
            return value
    return None


class LivenessAnalyzer:
    """Analyze tensor liveness across the operation DAG.

    A tensor is "live" between the op that produces it
    and the last op that consumes it.
    """

    def __init__(self, dag: OperationDAG):
        self.dag = dag
        self._live_ranges: Dict[int, Tuple[int, int]] = {}
        self._live_at: Dict[int, Set[int]] = defaultdict(set)
        self._analyzed_fingerprint: Optional[Tuple[Any, ...]] = None

    def _dag_fingerprint(self) -> Tuple[Any, ...]:
        """Cache key covering everything the sweep reads from the DAG.

        Node and edge counts alone are not enough. The sweep keys tensors by
        ``id()`` and walks ``output_tensors``/``input_tensors``/``consumers``,
        so an in-place mutation that preserves the counts — a pass that swaps a
        node's output tensor for a differently shaped one, or rewires an edge
        without changing how many there are — would be served a stale live-set
        map. ``PeakMemoryEstimator`` already had this fixed for its own cache
        (``PeakMemoryEstimator._fingerprint``); this one did not, and the two
        caches sit behind the same call chain.

        Tensor identity is included alongside shape because the sweep's tensor
        indices are identity-derived: two same-shaped tensors are distinct
        buffers to it, so replacing one with an equal-shaped other really does
        invalidate the result. Building this is O(|V|), which is noise next to
        the O(|V|·K) sweep it guards (proposal §4.8.5).
        """
        return (
            len(self.dag.nodes),
            len(self.dag.edges),
            tuple(sorted(self.dag.edges)),
            tuple(
                (
                    op_id,
                    tuple((id(t), t.shape) for t in node.output_tensors),
                    tuple((id(t), t.shape) for t in node.input_tensors),
                    tuple(node.consumers),
                )
                for op_id, node in sorted(self.dag.nodes.items())
            ),
        )

    def analyze(self) -> Dict[int, Set[int]]:
        """Run liveness analysis.

        The result is cached: the pass manager reaches the estimator through
        ``estimate``/``get_peak``/``exceeds_l1``/``slicing_candidates``/
        ``summary``, and this scan is O(|V|·K) (proposal §4.8.5). Recomputing it
        for each of them would also make the reported ``liveness_ms`` cover a
        fraction of the work actually done. The cache is dropped when the DAG's
        contents change — see ``_dag_fingerprint`` for what that covers.

        Returns:
            Dict mapping op_id to the set of live tensor indices.
        """
        fingerprint = self._dag_fingerprint()
        if self._analyzed_fingerprint == fingerprint:
            return dict(self._live_at)

        self._live_ranges = {}
        self._live_at = defaultdict(set)
        order = self.dag.topological_order()
        op_id_to_index = {op: i for i, op in enumerate(order)}

        # Step 1: assign unique index to every tensor, track producers & consumers
        tensor_producer: Dict[int, int] = {}
        tensor_consumers: Dict[int, List[int]] = defaultdict(list)
        tensor_idx_map: Dict[int, int] = {}
        next_idx = 0

        for op_id in order:
            node = self.dag.nodes[op_id]

            for output_tensor in node.output_tensors:
                out_id = id(output_tensor)
                if out_id not in tensor_idx_map:
                    tensor_idx_map[out_id] = next_idx
                    next_idx += 1
                tensor_producer[tensor_idx_map[out_id]] = op_id

            for inp in node.input_tensors:
                inp_id = id(inp)
                if inp_id not in tensor_idx_map:
                    tensor_idx_map[inp_id] = next_idx
                    next_idx += 1
                tensor_consumers[tensor_idx_map[inp_id]].append(op_id)

        # Step 2: compute live range [producer, last_consumer] for each tensor
        for t_idx, producer in tensor_producer.items():
            consumers = tensor_consumers.get(t_idx, [])
            if consumers:
                last = max(consumers, key=lambda x: op_id_to_index.get(x, 0))
            else:
                last = producer
            self._live_ranges[t_idx] = (producer, last)

        # Producer-less tensors are the program's external inputs (function
        # block arguments). They are live from program entry until their last
        # consumer — omitting them undercounts peak memory at early ops.
        if order:
            first_op = order[0]
            for t_idx, consumers in tensor_consumers.items():
                if t_idx in tensor_producer or not consumers:
                    continue
                last = max(consumers, key=lambda x: op_id_to_index.get(x, 0))
                self._live_ranges[t_idx] = (first_op, last)

        # Step 3: compute live set at each program point
        for op_id in order:
            point = op_id_to_index[op_id]
            live = set()
            for t_idx, (start, end) in self._live_ranges.items():
                start_idx = op_id_to_index[start]
                end_idx = op_id_to_index[end]
                if start_idx <= point <= end_idx:
                    live.add(t_idx)
            self._live_at[op_id] = live

        self._analyzed_fingerprint = fingerprint
        return dict(self._live_at)

    def get_live_range(self, tensor_idx: int) -> Tuple[int, int]:
        """Return (start_op, end_op) for a tensor."""
        return self._live_ranges.get(tensor_idx, (-1, -1))

    def get_live_tensors(self, op_id: int) -> Set[int]:
        """Return set of live tensor indices at a given op."""
        return self._live_at.get(op_id, set())

    def summary(self) -> str:
        """Print liveness summary."""
        lines = ["=== Liveness Analysis Summary ==="]
        for t_idx, (start, end) in sorted(self._live_ranges.items()):
            lines.append(f"  Tensor #{t_idx}: live from Op {start} to Op {end}")
        lines.append(f"  Total tensors: {len(self._live_ranges)}")
        return "\n".join(lines)


MEMORY_TIER_RE = re.compile(
    r'edgeslice\.memory_tier\s*=\s*"(?P<tier>\w+)"'
)
_TIER_LINE_RE = re.compile(
    r'^\s*(?:%[\w.$-]+\s*=\s*)?"?(?P<op>[\w.]+)"?.*?'
    r'edgeslice\.memory_tier\s*=\s*"(?P<tier>\w+)"'
)
_TIER_SIZE_RE = re.compile(
    r'edgeslice\.(?P<size_kind>working_set_bytes|output_bytes)\s*=\s*(?P<size>\d+)'
)
_FUSED_INTO_RE = re.compile(r'edgeslice\.fused_into\s*=\s*"(?P<anchor>[\w.]+)"')


def summarize_memory_tiers(mlir_text: str) -> Dict[str, Any]:
    """Group payload ops by the memory tier EdgeSlice assigned them.

    Reads ``edgeslice.memory_tier`` annotations left on the payload by the
    transform script. These describe the compiler's intent, not physical
    placement: on a CPU target L1 is a hardware-managed cache.
    """
    by_tier: Dict[str, List[Dict[str, Any]]] = {}
    for line in mlir_text.splitlines():
        match = _TIER_LINE_RE.search(line)
        if match is None:
            continue
        size_match = _TIER_SIZE_RE.search(line)
        fused_match = _FUSED_INTO_RE.search(line)
        by_tier.setdefault(match.group("tier"), []).append({
            "op": match.group("op"),
            "size_kind": size_match.group("size_kind") if size_match else None,
            "bytes": int(size_match.group("size")) if size_match else None,
            "fused_into": fused_match.group("anchor") if fused_match else None,
        })

    totals = {
        tier: sum(entry["bytes"] or 0 for entry in entries)
        for tier, entries in by_tier.items()
    }
    return {
        "by_tier": by_tier,
        "counts": {tier: len(entries) for tier, entries in by_tier.items()},
        "total_bytes": totals,
    }


def format_memory_tier_report(mlir_text: str) -> str:
    """Human-readable rendering of :func:`summarize_memory_tiers`."""
    summary = summarize_memory_tiers(mlir_text)
    if not summary["by_tier"]:
        return "=== Memory Tiers ===\n  (no edgeslice.memory_tier annotations found)"

    lines = [
        "=== Memory Tiers (compiler intent, not physical placement) ===",
    ]
    for tier in sorted(summary["by_tier"]):
        entries = summary["by_tier"][tier]
        total = summary["total_bytes"][tier]
        lines.append(f"  {tier.upper()}: {len(entries)} op(s), {total / 1024:.1f} KB tracked")
        for entry in entries:
            if entry["bytes"] is not None:
                size = f"{entry['bytes'] / 1024:.1f} KB ({entry['size_kind']})"
            elif entry["fused_into"]:
                size = f"counted in {entry['fused_into']} chain total"
            else:
                size = "size not annotated"
            lines.append(f"    - {entry['op']}: {size}")
    return "\n".join(lines)
