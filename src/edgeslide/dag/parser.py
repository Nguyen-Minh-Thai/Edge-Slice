"""DAG Parser — Parse PyTorch model to Operation DAG via torch-mlir using native MLIR Python bindings."""

import logging
import re
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

if TYPE_CHECKING:
    import torch

try:
    # Imported for the availability probe only; both must be present for
    # any parsing path to work.
    import torch as _torch  # noqa: F401
    import torch_mlir
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from edgeslide.dag.graph import OperationDAG, TensorInfo

logger = logging.getLogger(__name__)

# Seed for the example input used to trace a model. Fixed, and a named constant
# rather than a literal, so a result file can cite it and a caller can see there
# is one.
#
# Measured, not assumed: for every shipped workload the parsed DAG is identical
# under two different seeds, so shape-derived results really are properties of
# the model and not of whichever tensor reached the tracer. U-Net included, even
# though its decoder emits a TracerWarning about an input-dependent size. So the
# seed is not fixing a live bug — it is what keeps that property checkable, and
# what stops a future model from silently trading it away, since tracing *can*
# constant-fold an input-dependent value into the graph.
# ``tests/test_tracing_reproducibility.py`` pins both halves.
TRACING_SEED = 0

# torch-mlir's Python bindings import a dialect module lazily the first time
# they meet an op from that dialect, and that import is not thread-safe: one
# thread can observe another's half-initialised entry and get
# ``KeyError: 'torch_mlir.dialects.linalg'`` from the middle of an IR walk.
# The failure is racy, so it shows up as an occasional unexplained parse error
# rather than a reproducible one.
#
# The lock covers compilation and traversal only. Everything expensive in the
# pipeline — the mlir-opt and mlir-translate subprocesses — runs outside it, so
# concurrent compiles still overlap where it matters.
_TORCH_MLIR_LOCK = threading.Lock()


class MLIRContextError(Exception):
    """Exception raised when MLIR context operations fail."""
    pass


class MLIRTraversalError(Exception):
    """Exception raised when MLIR IR traversal fails."""
    pass


class DAGParser:
    """Parse a PyTorch model and construct an operation DAG using native MLIR Python bindings.

    Pipeline:
        1. Load .pt/.pth model
        2. Convert to Torch dialect IR via torch-mlir
        3. Lower to Linalg dialect (LINALG_ON_TENSORS)
        4. Traverse IR programmatically to build DAG (nodes = ops, edges = tensor deps)
    """

    def __init__(self, model: "torch.nn.Module", example_input: "torch.Tensor"):
        self.model = model
        self.example_input = example_input
        self.tensor_map: Dict[Any, TensorInfo] = {}
        self.op_id_map: Dict[Any, int] = {}
        self.ssa_value_to_producer_id: Dict[Any, int] = {}
        self.op_operand_ssa_values: Dict[int, List[Any]] = {}
        self.mlir_module = None
        self.mlir_op_by_dag_id: Dict[int, Any] = {}

    def parse(self) -> OperationDAG:
        if not HAS_TORCH:
            raise ImportError(
                "torch and torch-mlir are required for model parsing. "
                "`pip install torch-mlir` does not produce a working "
                "environment: the wheels are CPython 3.11 snapshot builds "
                "published as GitHub release assets, ABI-locked to a specific "
                "torch dev snapshot and to NumPy 1.x. Follow "
                "docs/DEPENDENCIES.md \u00a7 'Building it from scratch'."
            )

        try:
            # Availability probe: the helpful ImportError below is the point.
            from torch_mlir.ir import Context, Module  # noqa: F401
        except ImportError:
            raise ImportError(
                "torch-mlir is installed but its IR bindings "
                "(torch_mlir.ir) are not importable, so the DAG cannot be "
                "built. This is what a version mismatch looks like: "
                "`torch-mlir-nightly` in particular is a different, "
                "incompatible distribution. The verified pin is "
                "torch-mlir==20240127.1096 on CPython 3.11 — see "
                "docs/DEPENDENCIES.md."
            )

        # Serialised: see _TORCH_MLIR_LOCK.
        with _TORCH_MLIR_LOCK:
            # Step 1: Compile to Linalg on Tensors (proper lowering)
            mlir_module = self._compile_to_linalg(self.model, self.example_input)
            self.mlir_module = mlir_module
            self.mlir_op_by_dag_id = {}

            dag = self._build_dag_native(mlir_module)
        return dag

    @staticmethod
    def _compile_to_linalg(model: "torch.nn.Module", example_input: "torch.Tensor"):
        """Lower to linalg-on-tensors across torch-mlir API generations.

        Older wheels (pinned in .venv-mlir) expose ``torch_mlir.compile`` +
        ``torch_mlir.OutputType``; both were removed from recent releases in
        favor of the FX importer (``torch_mlir.fx.export_and_import``).
        """
        if hasattr(torch_mlir, "compile") and hasattr(torch_mlir, "OutputType"):
            # torch_mlir.compile(..., use_tracing=True) would otherwise call
            # torch.jit.trace_module with its defaults: check_trace=True (runs
            # the forward pass a *second* time just to diff it against the
            # first) and gradients enabled if the model's params do (builds an
            # unused autograd graph). Neither is needed — this trace only
            # supplies shapes/structure, never backprop, and EdgeSlice's own
            # equivalence tests verify correctness downstream. Pre-tracing
            # ourselves and handing torch_mlir.compile the resulting
            # ScriptModule skips its internal re-trace entirely (it takes the
            # "already scripted" fast path). Measured on resnet50 @ 1600x900:
            # 8.7s -> 0.18s end to end, ~50x, with identical output.
            import torch

            with torch.no_grad():
                scripted = torch.jit.trace(model, example_input, check_trace=False)
            return torch_mlir.compile(
                scripted,
                example_input,
                output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
                use_tracing=True,
            )

        try:
            from torch_mlir import fx as torch_mlir_fx
        except ImportError as exc:
            raise ImportError(
                "This torch-mlir wheel exposes neither torch_mlir.compile() "
                "nor torch_mlir.fx.export_and_import(); install either the "
                "pinned legacy wheel or a release with the FX importer."
            ) from exc

        return torch_mlir_fx.export_and_import(
            model,
            example_input,
            output_type="linalg-on-tensors",
        )

    @staticmethod
    def edgeslice_op_tag(op_id: int) -> str:
        """Stable payload attribute value for transform.structured.match."""
        return f"op_{op_id}"

    def save_mlir(self, path: str) -> str:
        """Write annotated Linalg payload IR (includes edgeslice.op_id attributes)."""
        if self.mlir_module is None:
            raise RuntimeError("Call parse() before save_mlir()")
        with open(path, "w") as f:
            f.write(str(self.mlir_module))
        return path

    def _build_dag_native(self, mlir_module) -> OperationDAG:
        """Traverse Linalg IR programmatically and construct the DAG.

        This implementation:
        - Walks the MLIR IR using native Python bindings
        - Extracts operations, tensor shapes, and types programmatically
        - Builds data dependencies based on SSA value use-def chains
        """
        dag = OperationDAG()
        self.tensor_map = {}
        self.ssa_value_to_producer_id = {}
        self.op_operand_ssa_values = {}

        try:
            # Step 1: Walk the module to collect all operations and SSA values
            self._walk_module(mlir_module, dag)

            # Step 2: Connect edges via SSA producer lookup (O(ops × arity))
            self._connect_edges_using_producer_map(dag)

        except Exception as e:
            raise MLIRTraversalError(
                f"Failed to traverse MLIR IR: {e!r}"
            ) from e

        return dag

    def _walk_module(self, mlir_module, dag: OperationDAG):
        """Walk the MLIR module and extract operations programmatically."""

        # Walk through all regions in the module
        for region in mlir_module.operation.regions:
            for block in region.blocks:
                self._walk_block(block, dag)

    def _walk_block(self, block, dag: OperationDAG):
        """Walk a block and process all operations."""

        for op in block.operations:
            self._process_operation(op, dag)

    def _process_operation(self, op, dag: OperationDAG):
        """Process a single MLIR operation and add it to the DAG."""

        # Convert operation name to string (it's a StringAttr)
        op_name = str(op.name)

        # Skip module-level operations
        if op_name in ['module', 'func.func', 'func.return']:
            # Walk into nested regions
            for region in op.regions:
                for block in region.blocks:
                    self._walk_block(block, dag)
            return

        # Extract operation type
        op_type = self._get_op_type(op_name)

        # Process different operation types
        if op_type.startswith('linalg.'):
            self._process_linalg_op(op, op_type, dag)
        elif op_type.startswith('arith.') or op_type.startswith('tensor.'):
            self._process_arith_tensor_op(op, op_type, dag)
        elif op_name == 'torch.global_slot':
            pass  # Skip global slots
        else:
            # Walk into nested regions for other ops
            for region in op.regions:
                for block in region.blocks:
                    self._walk_block(block, dag)

    def _get_op_type(self, op_name: str) -> str:
        """Extract the operation type from the full operation name."""
        # op_name might be like "linalg.conv_2d_nchw_fchw" or "arith.addf"
        return op_name

    def _process_linalg_op(self, op, op_type: str, dag: OperationDAG):
        """Process a linalg dialect operation."""

        # Extract operands (inputs)
        input_tensors = []
        for operand in op.operands:
            tensor_info = self._get_or_create_tensor_info(operand)
            if tensor_info:
                input_tensors.append(tensor_info)

        # Extract results (outputs)
        output_tensors = []
        for result in op.results:
            result_type = result.type
            tensor_info = self._create_tensor_info_from_type(result_type, result)
            if tensor_info:
                output_tensors.append(tensor_info)
                self.tensor_map[result] = tensor_info

        if not output_tensors:
            # Create default output tensor info
            output_tensors = [TensorInfo(shape=(1,), dtype="f32", name=str(op.result))]
        output_tensor = output_tensors[0]

        # Extract operation attributes (strides, padding, dilation, etc.)
        params = self._extract_op_attributes(op)
        if op_type == "linalg.generic":
            params["elementwise_identity"] = self._is_elementwise_identity_generic(op)

        # Add operation to DAG
        op_id = dag.add_node(
            op_type,
            input_tensors,
            output_tensor,
            params,
            outputs=output_tensors,
        )
        tag = self.edgeslice_op_tag(op_id)
        dag.nodes[op_id].params["edgeslice_op_id"] = tag
        self._annotate_mlir_op(op, tag)
        self.mlir_op_by_dag_id[op_id] = op
        self.op_operand_ssa_values[op_id] = list(op.operands)

        for result in op.results:
            self.ssa_value_to_producer_id[result] = op_id

        for operand in op.operands:
            if operand not in self.op_id_map:
                self.op_id_map[operand] = op_id

    def _process_arith_tensor_op(self, op, op_type: str, dag: OperationDAG):
        """Process arithmetic or tensor operations."""

        # Extract operands (inputs)
        input_tensors = []
        for operand in op.operands:
            tensor_info = self._get_or_create_tensor_info(operand)
            if tensor_info:
                input_tensors.append(tensor_info)

        # Extract results (outputs)
        output_tensors = []
        for result in op.results:
            result_type = result.type
            tensor_info = self._create_tensor_info_from_type(result_type, result)
            if tensor_info:
                output_tensors.append(tensor_info)
                self.tensor_map[result] = tensor_info

        if not output_tensors:
            # Create default output tensor info
            output_tensors = [TensorInfo(shape=(1,), dtype="f32", name=str(op.result))]
        output_tensor = output_tensors[0]

        op_id = dag.add_node(
            op_type,
            input_tensors,
            output_tensor,
            {},
            outputs=output_tensors,
        )
        self.op_operand_ssa_values[op_id] = list(op.operands)

        for result in op.results:
            self.ssa_value_to_producer_id[result] = op_id

        for operand in op.operands:
            if operand not in self.op_id_map:
                self.op_id_map[operand] = op_id

    def _get_or_create_tensor_info(self, ssa_value) -> Optional[TensorInfo]:
        """Get or create TensorInfo for an SSA value."""

        if ssa_value in self.tensor_map:
            return self.tensor_map[ssa_value]

        # Create from the SSA value's type
        result_type = ssa_value.type
        tensor_info = self._create_tensor_info_from_type(result_type, ssa_value)
        if tensor_info:
            self.tensor_map[ssa_value] = tensor_info
        return tensor_info

    def _create_tensor_info_from_type(self, mlir_type, ssa_value) -> Optional[TensorInfo]:
        """Create TensorInfo from an MLIR type."""

        try:
            # Get the string representation of the type
            type_str = str(mlir_type)

            # Check if it's a tensor type
            if type_str.startswith('tensor<'):
                # Parse shape and dtype
                # Format: tensor<shapexdtype> or tensor<dtype>
                content = type_str[7:-1]  # Remove 'tensor<' and '>'

                parts = content.split('x')

                shape: List[int] = []
                if len(parts) == 1:
                    # Scalar or 1D tensor
                    dtype = parts[0]
                else:
                    # Multi-dimensional tensor
                    for part in parts[:-1]:
                        if part == '?':
                            shape.append(-1)  # Dynamic dimension
                        else:
                            try:
                                shape.append(int(part))
                            except ValueError:
                                shape.append(-1)
                    dtype = parts[-1]

                return TensorInfo(
                    shape=tuple(shape),
                    dtype=dtype,
                    name=str(ssa_value)
                )
        except Exception as exc:
            # Best-effort probe over heterogeneous MLIR value objects; a value
            # we cannot read is not fatal, but leave a trace for diagnosis.
            logger.debug("Could not extract TensorInfo from %r: %s", ssa_value, exc)

        return None

    def _annotate_mlir_op(self, op, tag: str) -> None:
        """Stamp a unique edgeslice.op_id on the payload Linalg op for transform matching."""
        try:
            from torch_mlir.ir import StringAttr

            op.attributes["edgeslice.op_id"] = StringAttr.get(tag, context=op.context)
        except Exception as exc:
            raise MLIRTraversalError(
                f"Cannot annotate payload op {op.name} with edgeslice.op_id={tag!r}"
            ) from exc

    @staticmethod
    def _is_elementwise_identity_generic(op) -> bool:
        """Recognize a linalg.generic that is safe to tile by output dimension.

        Requires all-parallel iterators, a single result, and an identity output
        indexing map — that is what makes "tile output dim *k*" meaningful.
        Input maps may broadcast: each of their results must be a domain dim or
        a constant, nothing compound. Structured tiling derives the input slice
        from the map itself, so a broadcast is safe; an expression like
        ``d0 + d1`` is not, and is rejected.

        Demanding the *input* maps also be the identity — as an earlier version
        did — silently excluded every elementwise op at batch 1: torch-mlir
        folds the unit batch dimension to a constant, so a ReLU's input map is
        ``(d0, d1, d2, d3) -> (0, d1, d2, d3)``. Since fusion chains run through
        those ReLUs, both elementwise tiling and fusion were dead in exactly the
        single-stream configuration the proposal targets (§1.3, §5.1).
        """
        text = str(op)

        scoped = re.search(
            r"indexing_maps\s*=\s*\[(.*?)\]\s*,\s*iterator_types",
            text,
            flags=re.DOTALL,
        )
        maps_text = scoped.group(1) if scoped else text

        iterator_match = re.search(r"iterator_types\s*=\s*\[([^\]]*)\]", text)
        if iterator_match is None:
            return False
        iterator_text = iterator_match.group(1)
        # MLIR has printed these as quoted strings and, in newer builds, as
        # `#linalg.iterator_type<parallel>` enum attributes.
        iterators = re.findall(r'"([^"]+)"', iterator_text) or re.findall(
            r"#linalg\.iterator_type<([^>]+)>", iterator_text
        )
        if not iterators or any(iterator != "parallel" for iterator in iterators):
            return False

        try:
            if len(op.results) != 1:
                return False
        except (AttributeError, TypeError):  # pragma: no cover - op stubs in unit tests
            pass

        maps = re.findall(
            r"affine_map<\(([^)]*)\)\s*->\s*\(([^)]*)\)>",
            maps_text,
        )
        if len(maps) < 2:
            return False

        def parts(text_):
            return [part.strip() for part in text_.split(",") if part.strip()]

        domain = parts(maps[0][0])
        if len(domain) != len(iterators):
            return False

        # The output operand's map is the last one; it must be the identity.
        if parts(maps[-1][1]) != domain:
            return False

        for map_domain, map_result in maps[:-1]:
            if parts(map_domain) != domain:
                return False
            for expression in parts(map_result):
                if expression in domain:
                    continue
                if re.fullmatch(r"-?\d+", expression):
                    continue
                return False
        return True

    def _extract_op_attributes(self, op) -> Dict[str, Any]:
        """Extract attributes from a linalg operation.

        This method extracts common attributes like:
        - strides: vector<int>
        - dilations: vector<int>
        - padding: (if present)
        - kernel_size: (if present)
        """

        params = {}

        try:
            # Access operation attributes using the correct API
            # MLIR attributes are accessed via the 'attributes' property
            if hasattr(op, 'attributes'):
                attrs = op.attributes

                # Access attributes by index
                # NamedAttribute has 'name' and 'attr' properties
                for i in range(len(attrs)):
                    try:
                        named_attr = attrs[i]
                        attr_name = str(named_attr.name)
                        attr_value = named_attr.attr  # Use 'attr' not 'value'

                        # Store common linalg attributes
                        if attr_name in ['strides', 'dilations', 'padding', 'kernel_size']:
                            parsed_value = self._parse_attribute_value(attr_value)
                            if parsed_value is not None:
                                params[attr_name] = parsed_value
                    except Exception as exc:
                        logger.debug(
                            "Skipping unreadable attribute %d on %s: %s",
                            i,
                            getattr(op, "name", "<op>"),
                            exc,
                        )
                        continue

        except Exception as exc:
            # If attribute extraction fails, continue with empty params.
            logger.debug(
                "Attribute extraction failed on %s: %s",
                getattr(op, "name", "<op>"),
                exc,
            )

        return params

    def _parse_attribute_value(self, attr_value) -> Any:
        """Parse an MLIR attribute value to Python types.

        Handles:
        - Dense vectors: dense<1, 2> : vector<2xi64>
        - Integer arrays
        - String attributes
        """

        if attr_value is None:
            return None

        try:
            # Convert to string for parsing
            value_str = str(attr_value)

            # Handle dense attrs: splat `dense<2>` and non-splat
            # `dense<[2, 1]> : tensor<2xi64>` (MLIR prints arrays in brackets)
            if 'dense<' in value_str:
                # Extract values between dense< and the closing >
                start = value_str.find('<') + 1
                # Find the first > after dense< (not the one from vector<...>)
                end = value_str.find('>', start)
                values_str = value_str[start:end].strip()
                # Strip the bracket syntax of non-splat dense arrays
                values_str = values_str.strip('[]')

                # Parse comma-separated values
                values = [v.strip() for v in values_str.split(',')]
                try:
                    return [int(v) for v in values if v]
                except ValueError:
                    return values

            # Handle simple integer values
            try:
                return int(value_str)
            except ValueError:
                pass

            return value_str
        except Exception as exc:
            logger.debug("Could not parse MLIR attribute value: %s", exc)
            return None

    def _connect_edges_using_producer_map(self, dag: OperationDAG):
        """Connect edges using SSA use-def chains built during the walk pass."""

        seen_edges: Set[Tuple[int, int]] = set()

        for consumer_id, operands in self.op_operand_ssa_values.items():
            for ssa_value in operands:
                producer_id = self.ssa_value_to_producer_id.get(ssa_value)
                if producer_id is None or producer_id == consumer_id:
                    continue
                edge = (producer_id, consumer_id)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                dag.add_edge(producer_id, consumer_id)


def parse_pytorch_model(model: "torch.nn.Module", example_input: "torch.Tensor") -> OperationDAG:
    """Convenience function to parse a PyTorch model into an OperationDAG.

    Args:
        model: PyTorch model to parse
        example_input: Example input tensor for tracing

    Returns:
        OperationDAG representing the model
    """
    parser = DAGParser(model, example_input)
    return parser.parse()


def load_model_from_file(
    model_path: str,
    device: str = "cpu",
    model_factory: Optional[Callable[[], "torch.nn.Module"]] = None,
) -> "torch.nn.Module":
    """Load a ``.pt``/``.pth`` file as an eval-mode ``nn.Module`` on ``device``.

    ``.pt`` is not one format, it is three, and only two of them carry enough
    information to be traced:

    1. **A pickled ``nn.Module``** (``torch.save(model, path)``). Self-describing,
       so no ``model_factory`` is needed. Unpickling it executes the module's
       defining code, hence ``weights_only=False`` — see the security note below.
    2. **A TorchScript archive** (``torch.jit.save``). Handled by the same branch
       for free: ``torch.load`` sniffs the zip header and delegates to
       ``torch.jit.load``, whose ``RecursiveScriptModule`` is an ``nn.Module``.
    3. **A state dict** (``torch.save(model.state_dict(), path)``) — the most
       common thing people have on disk, and the one that is *not* a model. It
       is a plain mapping of parameter tensors with no graph, no forward, and no
       class. Nothing can recover the architecture from it, so the caller has to
       supply it as ``model_factory``.

    Security: without ``model_factory`` this must unpickle arbitrary Python
    (``weights_only=False``) because that is the only way case 1 can work at all
    — the pickle names a user class. Only point it at checkpoint files you
    trust. Passing ``model_factory`` takes the safe path (``weights_only=True``),
    which refuses to execute anything and only materialises tensors.

    Args:
        model_path: Path to a ``.pt`` / ``.pth`` file.
        device: Device string passed to ``torch.load`` as ``map_location`` and
            used to place the returned module.
        model_factory: Zero-argument callable returning a freshly constructed,
            randomly initialised module with the architecture that produced the
            checkpoint. Required for state-dict files, ignored for case 1/2.

    Returns:
        The module, moved to ``device`` and switched to eval mode (dropout and
        batch-norm must be in inference mode or the traced graph — and therefore
        the DAG and every downstream memory estimate — describes training).

    Raises:
        ValueError: The file holds a state dict but no ``model_factory`` was
            given, or ``model_factory`` produced a module the state dict does
            not fit.
        TypeError: The file holds something that is neither a module nor a
            state dict, or ``model_factory`` did not return an ``nn.Module``.
    """
    import torch

    if model_factory is not None:
        # weights_only=True is the whole point of taking a factory: the
        # architecture comes from caller-supplied code, so the file only has to
        # yield tensors and never gets to execute anything.
        state = torch.load(model_path, map_location=device, weights_only=True)
        state_dict = _extract_state_dict(state)
        if state_dict is None:
            raise TypeError(
                f"{model_path!r} does not contain a state dict (loaded a "
                f"{type(state).__name__}), but model_factory was supplied. Drop "
                "model_factory to load a pickled nn.Module, or point at the "
                "checkpoint that actually holds the weights."
            )

        model = model_factory()
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "model_factory must return a torch.nn.Module, got "
                f"{type(model).__name__}."
            )
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            # Surfaced as ValueError with the original text attached: torch's
            # message already names the mismatched keys/shapes, which is the
            # only thing that tells the caller their factory is the wrong
            # architecture rather than their file being corrupt.
            raise ValueError(
                f"State dict in {model_path!r} does not fit the module built by "
                f"model_factory: {exc}"
            ) from exc
    else:
        model = torch.load(model_path, map_location=device, weights_only=False)
        if not isinstance(model, torch.nn.Module):
            # The old code called .eval() straight on whatever came back, so a
            # state dict died with "'dict' object has no attribute 'eval'" —
            # true, useless, and three frames away from the real problem.
            if _extract_state_dict(model) is not None:
                raise ValueError(
                    f"{model_path!r} contains a state dict "
                    f"({type(model).__name__} of tensors), not a model. A state "
                    "dict has weights but no architecture, so it cannot be "
                    "traced on its own. Either re-save the whole module with "
                    "torch.save(model, path) / torch.jit.save(...), or call "
                    "parse_from_file(..., model_factory=YourModel) so the "
                    "architecture is available to load the weights into."
                )
            raise TypeError(
                f"{model_path!r} loaded as {type(model).__name__}, which is "
                "neither an nn.Module nor a state dict."
            )

    # .to() is in-place for modules but returns self; assignment keeps mypy and
    # any future non-module return honest.
    model = model.to(device)
    model.eval()
    return model


def _extract_state_dict(obj: Any) -> Optional[Mapping[str, Any]]:
    """Return ``obj`` as a state dict, unwrapping the usual checkpoint envelopes.

    Training scripts rarely save a bare ``state_dict()``; they save
    ``{"epoch": ..., "state_dict": ...}`` or the ``"model_state_dict"`` spelling
    from the PyTorch saving tutorial. Recognising those here means the caller
    does not have to unwrap by hand just to find out which convention their
    checkpoint used. Returns ``None`` when ``obj`` is not a state dict at all,
    which is how the branches above tell "wrong shape of file" apart from
    "missing architecture".
    """
    import torch

    if isinstance(obj, Mapping):
        for key in ("state_dict", "model_state_dict"):
            inner = obj.get(key)
            if isinstance(inner, Mapping):
                return inner
        # A bare state dict: every value is a tensor. Checked rather than
        # assumed so a config dict that happens to be a Mapping is reported as
        # "not a state dict" instead of failing later inside load_state_dict.
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    return None


def parse_from_file(
    model_path: str,
    example_input_shape: Tuple[int, ...],
    device: str = "cpu",
    model_factory: Optional[Callable[[], "torch.nn.Module"]] = None,
) -> OperationDAG:
    """Parse a PyTorch model file into an OperationDAG (proposal §1.3 input path).

    Thin wrapper: :func:`load_model_from_file` resolves the file to a module,
    then the module is traced exactly like :func:`parse_pytorch_model` would.
    The example input is random because torch-mlir only reads its shape and
    dtype — the values never reach the DAG.

    Args:
        model_path: Path to a ``.pt`` / ``.pth`` file. See
            :func:`load_model_from_file` for which flavours are accepted.
        example_input_shape: Shape of the example input, e.g. ``(1, 3, 224, 224)``.
        device: Device to load the model and build the example input on.
        model_factory: Architecture callable, required only when the file holds
            a state dict rather than a whole module.

    Returns:
        OperationDAG representing the model.
    """
    import torch

    model = load_model_from_file(model_path, device=device, model_factory=model_factory)
    # Seeded: tracing can bake an input-dependent value into the graph, so an
    # unseeded example input would make the DAG non-reproducible.
    generator = torch.Generator().manual_seed(TRACING_SEED)
    example_input = torch.randn(*example_input_shape, generator=generator).to(device)

    return parse_pytorch_model(model, example_input)
