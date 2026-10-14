"""
EdgeSlide MLIR Compiler Pipeline

End-to-end pipeline for compiling PyTorch models with EdgeSlide transforms:
1. PyTorch model → torch-mlir → MLIR (Linalg dialect)
2. Analyze MLIR to identify slicing candidates
3. Apply EdgeSlide Transform Dialect pass
4. Lower to LLVM IR
5. Compile to executable binary
"""

from __future__ import annotations

import logging
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from edgeslide.mlir_transform.errors import (
    LLVMCompilationError,
    MLIRTransformError,
    StageResult,
    ToolNotFoundError,
    TransformVerificationError,
    atomic_output,
    atomic_write_text,
    prepare_output_path,
    require_input_file,
    require_tool,
    run_tool,
)

logger = logging.getLogger(__name__)

# Artifact paths and stage records produced by EdgeSlideMLIRPipeline.compile().
# Values are mostly paths, plus "stages" (list[StageResult]), "transform_applied"
# (bool), and optionally "jit_engine".
PipelineResults = Dict[str, Any]

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("torch is required for PyTorch-to-MLIR compilation")


def find_mlir_opt() -> Optional[str]:
    """Locate mlir-opt, or return None.

    Probe form, kept for callers that decide whether to skip work. Stages that
    genuinely need the binary use :func:`require_mlir_opt` so a missing
    toolchain raises instead of degrading silently.
    """
    try:
        return require_tool("mlir-opt")
    except ToolNotFoundError:
        return None


def require_mlir_opt(explicit: Optional[str] = None) -> str:
    """Locate mlir-opt or raise :class:`ToolNotFoundError`."""
    return require_tool("mlir-opt", explicit)


def _count_payload_ops(mlir_text: str, op_name: str) -> int:
    """Count payload ops without matching transform op names or comments."""
    pattern = rf"(?m)^\s*(?:%[^=]+\s*=\s*)?{re.escape(op_name)}(?![\w.])"
    return len(re.findall(pattern, mlir_text))


def validate_transformed_payload(
    payload_mlir: str,
    transform_mlir: str,
    transformed_mlir: str,
) -> None:
    """Fail if requested forall tiling did not materialize in the payload."""
    requested = _count_payload_ops(
        transform_mlir,
        "transform.structured.tile_using_forall",
    )
    if requested == 0:
        return

    required_payload_ops = (
        "scf.forall",
        "tensor.extract_slice",
        "tensor.parallel_insert_slice",
    )
    missing = []
    for op_name in required_payload_ops:
        before = _count_payload_ops(payload_mlir, op_name)
        after = _count_payload_ops(transformed_mlir, op_name)
        if after - before < requested:
            missing.append(f"{op_name} (added {after - before}, expected >= {requested})")

    if missing:
        raise TransformVerificationError(
            "Transform interpreter completed without materializing every requested "
            "EdgeSlice tile. Missing payload rewrites: " + ", ".join(missing)
        )


def merge_payload_and_transform(payload_mlir: str, transform_mlir: str) -> str:
    """
    Merge payload and transform dialect scripts into one module for mlir-opt.

    mlir-opt accepts a single input file; the transform-interpreter pass expects
    both the payload program and the transform.named_sequence in the same module.
    """
    payload = payload_mlir.strip()
    transform = transform_mlir.strip()
    if not payload:
        raise ValueError("payload MLIR is empty")
    if not transform:
        raise ValueError("transform MLIR is empty")
    return f"{payload}\n\n{transform}\n"


def run_transform_interpreter(
    payload_path: Union[str, Path],
    transform_path: Union[str, Path],
    output_path: Union[str, Path],
    mlir_opt: Optional[str] = None,
    keep_merged: bool = False,
    verify_transformed: bool = True,
) -> str:
    """
    Run mlir-opt --transform-interpreter to produce sliced payload MLIR.

    Args:
        payload_path: Path to payload.mlir (Linalg program from torch-mlir)
        transform_path: Path to transform.mlir (from EdgeSlicePassManager)
        output_path: Path for sliced_payload.mlir output
        mlir_opt: Optional path to mlir-opt binary
        keep_merged: If True, also write the merged input next to output_path
        verify_transformed: Require requested tiling to appear in the payload

    Returns:
        Path to sliced_payload.mlir

    Raises:
        ToolNotFoundError: mlir-opt binary not found
        FileNotFoundError: payload or transform MLIR missing
        MLIRTransformError: mlir-opt exited non-zero
        TransformVerificationError: requested tiling did not materialize
    """
    # Inputs are validated before the toolchain is resolved: a missing or
    # empty payload is the caller's bug either way, and reporting "mlir-opt
    # not found" for it sends them off installing LLVM to fix a wrong path.
    # (mlir-opt itself reports a missing input as a parse diagnostic, which
    # reads as broken IR rather than a missing file — hence checking here.)
    payload_path = require_input_file(payload_path, "payload MLIR")
    transform_path = require_input_file(transform_path, "transform MLIR")
    output_path = prepare_output_path(output_path, "sliced payload MLIR")

    payload_mlir = payload_path.read_text()
    transform_mlir = transform_path.read_text()

    merged_mlir = merge_payload_and_transform(payload_mlir, transform_mlir)

    mlir_opt_bin = require_mlir_opt(mlir_opt)

    # Unique suffix: two interpreter runs writing the same stem in one
    # directory would otherwise share these intermediates.
    scratch_tag = uuid.uuid4().hex[:8]
    merged_path = output_path.parent / f"{output_path.stem}_merged_{scratch_tag}.mlir"
    atomic_write_text(merged_path, merged_mlir, description="merged MLIR")
    if not keep_merged:
        merged_path_to_cleanup = merged_path
    else:
        merged_path_to_cleanup = None

    # Two mlir-opt invocations on purpose. The materialization check has to see
    # the interpreter's own output: --canonicalize/--cse legitimately fold
    # extract_slice chains and single-iteration scf.forall loops, so running the
    # check after cleanup would report a rewrite as "missing" that did happen.
    interpreted_path = (
        output_path.parent / f"{output_path.stem}_interpreted_{scratch_tag}.mlir"
    )

    try:
        # mlir-opt writes into a temp path that is promoted only on exit 0, so
        # a failed interpreter run leaves no partial IR for the next stage to
        # pick up and treat as a successful transform.
        with atomic_output(
            interpreted_path, description="interpreted MLIR"
        ) as tmp_interpreted:
            run_tool(
                [
                    mlir_opt_bin,
                    str(merged_path),
                    "--transform-interpreter",
                    "-o",
                    str(tmp_interpreted),
                ],
                stage="mlir-opt --transform-interpreter",
                error_cls=MLIRTransformError,
            )
    finally:
        # The merged input is cleaned up even on failure, but only after the
        # exception has captured the command that referenced it.
        if merged_path_to_cleanup is not None and merged_path_to_cleanup.exists():
            merged_path_to_cleanup.unlink()

    try:
        if verify_transformed:
            validate_transformed_payload(
                payload_mlir, transform_mlir, interpreted_path.read_text()
            )
    except BaseException:
        # A payload that failed verification is not a usable artifact; leaving
        # it behind invites a later stage to consume it.
        if interpreted_path.exists():
            interpreted_path.unlink()
        raise

    # Fusion leaves the original producer behind as dead code; without this
    # cleanup its full-size buffer survives bufferization and peak memory goes
    # up rather than down.
    try:
        with atomic_output(output_path, description="sliced payload MLIR") as tmp_out:
            run_tool(
                [
                    mlir_opt_bin,
                    str(interpreted_path),
                    "--canonicalize",
                    "--cse",
                    "-o",
                    str(tmp_out),
                ],
                stage="mlir-opt --canonicalize --cse",
                error_cls=MLIRTransformError,
            )
    finally:
        if not keep_merged and interpreted_path.exists():
            interpreted_path.unlink()

    logger.info("Sliced payload MLIR written to %s", output_path)
    return str(output_path)


def new_run_id() -> str:
    """Short, collision-resistant id for one pipeline invocation."""
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class RunContext:
    """Filesystem namespace for the artifacts of a single ``compile()`` call.

    Every stage writes through this instead of naming files directly. Fixed
    names like ``payload.mlir`` and ``sliced_payload.mlir`` under a shared
    output directory mean two pipelines compiling different models at the same
    time silently overwrite each other's IR — and because each stage reads the
    file the previous one *named* rather than the one it wrote, the second run
    can lower the first run's payload and report success.
    """

    base_dir: Path
    model_name: str
    run_id: str

    @classmethod
    def create(
        cls,
        base_dir: Union[str, Path],
        model_name: str,
        run_id: Optional[str] = None,
    ) -> "RunContext":
        context = cls(Path(base_dir), model_name, run_id or new_run_id())
        context.directory.mkdir(parents=True, exist_ok=True)
        return context

    @property
    def directory(self) -> Path:
        """Per-run subdirectory: ``<base>/<model_name>_<run_id>``."""
        return self.base_dir / f"{self.model_name}_{self.run_id}"

    def path(self, name: str) -> Path:
        """Absolute path for an artifact inside this run's directory."""
        return self.directory / name


# Re-exported so callers that only touch the MLIR pipeline can cite the seed
# without reaching into the DAG layer.
from edgeslide.dag.parser import TRACING_SEED  # noqa: E402


@dataclass
class EdgeSlideMLIRConfig:
    """Configuration for MLIR compilation."""
    l1_cache_kb: int = 512
    # Set for an explicitly managed accelerator scratchpad. When present, the
    # planner uses a single SRAM level instead of treating the byte value as a
    # CPU cache and reserves a ping-pong slot when double buffering is enabled.
    sram_capacity_kb: Optional[int] = None
    sram_double_buffer: bool = True
    sram_memory_space: int = 1
    sram_alignment_bytes: int = 64
    planning_strategy: str = "search"
    schedule_search_max_candidates: int = 4096
    schedule_shortlist_size: int = 8
    # A target hook measures one already-materializable schedule. When set,
    # the pipeline wraps it in TargetAutotuner (warmups, median trials and a
    # persistent cache) and uses measured latency to select the shortlist
    # winner. It is intentionally absent by default: host timing is not a
    # substitute for an SRAM accelerator target.
    target_measure_once: Optional[Callable[[Any], float]] = None
    autotune_warmups: int = 1
    autotune_trials: int = 5
    autotune_cache_path: Optional[str] = None
    target_fingerprint: str = "unknown-target"
    tile_size: int = 64
    output_dir: str = "build/mlir"
    opt_level: str = "O2"  # Optimization level
    target_triple: str = "native"  # Native machine code


class PyTorchToMLIRCompiler:
    """
    Compile PyTorch models to MLIR using torch-mlir.
    """

    def __init__(self, config: EdgeSlideMLIRConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compile_to_mlir(self, model: nn.Module,
                        input_shape: tuple,
                        model_name: str = "model") -> str:
        """
        Compile PyTorch model to MLIR Linalg dialect.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            model_name: Name for the model

        Returns:
            Path to generated MLIR file
        """
        _require_torch()
        try:
            import torch_mlir
        except ImportError:
            raise ImportError(
                "torch-mlir is required. Install with:\n"
                "pip install torch-mlir -f https://llvm.github.io/torch-mlir/package-index/"
            )

        # Create example input for tracing, seeded (see TRACING_SEED).
        generator = torch.Generator().manual_seed(TRACING_SEED)
        example_input = torch.randn(*input_shape, generator=generator)

        logger.info("Compiling %s to MLIR...", model_name)

        # Use torch-mlir to compile
        mlir_module = torch_mlir.compile(
            model,
            example_args=(example_input,),
            output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
            use_tracing=True
        )

        # Save MLIR to file
        mlir_path = self.output_dir / f"{model_name}.mlir"
        with open(mlir_path, 'w') as f:
            f.write(str(mlir_module))

        logger.info("MLIR saved to: %s", mlir_path)
        return str(mlir_path)

    def extract_conv_info(self, model: nn.Module) -> List[Dict[str, Any]]:
        """
        Extract Conv2d layer information from PyTorch model.

        Returns:
            List of Conv2d descriptors
        """
        _require_torch()
        conv_info = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Get input shape by doing a forward pass with dummy data
                # For simplicity, we'll extract the configuration
                conv_info.append({
                    "name": name,
                    "type": "Conv2d",
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": module.kernel_size,
                    "stride": module.stride,
                    "padding": module.padding,
                    "dilation": module.dilation,
                    "groups": module.groups,
                    "bias": module.bias is not None
                })

        return conv_info


class EdgeSlideTransformPass:
    """
    Apply EdgeSlide transforms to MLIR module.

    This is the core of the compiler - it applies the overlap slicing
    transform to Conv2d operations that exceed L1 cache via the MLIR
    Transform Interpreter (mlir-opt --transform-interpreter).
    """

    def __init__(
        self,
        config: EdgeSlideMLIRConfig,
        run: Optional[RunContext] = None,
    ):
        self.config = config
        self.run = run
        # With a RunContext every artifact lands in that run's own directory;
        # without one this falls back to the shared output_dir, which is fine
        # for a single ad-hoc invocation but collides under concurrency.
        self.output_dir = run.directory if run is not None else Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def apply_transform(
        self,
        payload_mlir_path: Union[str, Path],
        transform_mlir_path: Union[str, Path],
        output_name: str = "sliced_payload.mlir",
    ) -> str:
        """
        Apply EdgeSlide transform script to payload MLIR via transform-interpreter.

        Args:
            payload_mlir_path: Path to payload.mlir (Linalg program from torch-mlir)
            transform_mlir_path: Path to transform.mlir (from EdgeSlicePassManager)
            output_name: Output filename (default: sliced_payload.mlir)

        Returns:
            Path to sliced_payload.mlir
        """
        payload_mlir_path = require_input_file(payload_mlir_path, "payload MLIR")
        transform_mlir_path = require_input_file(
            transform_mlir_path, "transform MLIR"
        )
        output_path = prepare_output_path(
            self.output_dir / output_name, "sliced payload MLIR"
        )

        logger.info(
            "Applying EdgeSlide transform interpreter (payload=%s, transform=%s)",
            payload_mlir_path,
            transform_mlir_path,
        )

        return run_transform_interpreter(
            str(payload_mlir_path),
            str(transform_mlir_path),
            str(output_path),
        )

    def write_transform_mlir(self, transform_ir: str, name: str = "transform.mlir") -> str:
        """Persist transform dialect IR produced by EdgeSlicePassManager."""
        transform_path = atomic_write_text(
            self.output_dir / name, transform_ir, description="transform MLIR"
        )
        logger.info("Transform MLIR saved to: %s", transform_path)
        return str(transform_path)

    def write_payload_mlir(
        self, payload_mlir_path: Union[str, Path], name: str = "payload.mlir"
    ) -> str:
        """Copy or alias torch-mlir output as payload.mlir for the interpreter stage."""
        payload_mlir_path = require_input_file(payload_mlir_path, "source MLIR")
        # Copied through a temp file: a concurrent reader must never see a
        # half-copied payload and try to parse it.
        with atomic_output(self.output_dir / name, description="payload MLIR") as tmp:
            shutil.copy2(payload_mlir_path, tmp)
        payload_path = self.output_dir / name
        logger.info("Payload MLIR saved to: %s", payload_path)
        return str(payload_path)


class MLIRToLLVMCompiler:
    """
    Lower sliced_payload.mlir to LLVM dialect / LLVM IR and optionally JIT.
    """

    def __init__(
        self,
        config: EdgeSlideMLIRConfig,
        run: Optional[RunContext] = None,
    ):
        self.config = config
        self.run = run
        self.output_dir = run.directory if run is not None else Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compile_to_llvm(self, sliced_payload_path: str,
                        entry_func: str = "conv2d") -> str:
        """
        Lower sliced_payload.mlir to LLVM IR via mlir-opt + mlir-translate.

        Args:
            sliced_payload_path: Path to sliced_payload.mlir
            entry_func: Entry function name for emit_c_interface tagging

        Returns:
            Path to the textual LLVM IR (.ll)

        Raises:
            ToolNotFoundError: mlir-opt or mlir-translate missing
            MLIRTransformError / LLVMCompilationError: a lowering step failed

        Failures propagate rather than returning None: a caller that reads a
        None as "nothing to do" reports a successful compile for a stage that
        never produced an artifact.
        """
        from edgeslide.mlir_transform.lowering import (
            SlicedPayloadLowering,
            SlicedPayloadLoweringConfig,
        )

        require_input_file(sliced_payload_path, "sliced payload MLIR")
        logger.info(
            "Lowering %s to LLVM dialect then LLVM IR", sliced_payload_path
        )

        lowering = SlicedPayloadLowering(
            SlicedPayloadLoweringConfig(
                output_dir=str(self.output_dir),
                entry_func=entry_func,
                opt_level=int(self.config.opt_level.replace("O", "")),
            )
        )
        artifacts = lowering.lower(sliced_payload_path)
        return artifacts.llvm_ir

    def create_jit_engine(self, sliced_payload_path: str,
                          entry_func: str = "conv2d"):
        """
        Build a torch_mlir ExecutionEngine for sliced payload (Way 1).

        Returns:
            SlicedPayloadJitEngine ready to invoke with PyTorch tensors
        """
        from edgeslide.mlir_transform.lowering import (
            SlicedPayloadJitEngine,
            SlicedPayloadLoweringConfig,
        )

        config = SlicedPayloadLoweringConfig(
            output_dir=str(self.output_dir),
            entry_func=entry_func,
            opt_level=int(self.config.opt_level.replace("O", "")),
        )
        return SlicedPayloadJitEngine.from_sliced_payload(
            sliced_payload_path,
            config=config,
        )

    def compile_to_binary(self, llvm_path: str,
                          output_name: str = "model_executable",
                          shared_lib: Optional[str] = None) -> str:
        """
        Compile LLVM IR to a native executable with clang.

        Args:
            llvm_path: Path to the textual LLVM IR (.ll) file
            output_name: Name for the output executable (no extension)
            shared_lib: Optional runner utils shared library to link

        Returns:
            Path to the compiled executable

        Raises:
            ToolNotFoundError: clang missing
            LLVMCompilationError: clang exited non-zero
        """
        require_input_file(llvm_path, "LLVM IR")
        # The artifact is a native executable, so the path carries no
        # extension — .ll here would name a file that is not LLVM IR.
        output_path = prepare_output_path(
            self.output_dir / output_name, "executable"
        )
        clang = require_tool("clang")

        logger.info("Compiling %s to executable %s", llvm_path, output_path)

        cmd = [
            clang,
            str(llvm_path),
            f"-{self.config.opt_level}",
            "-o", str(output_path),
            "-lm",
        ]
        if shared_lib:
            cmd.extend([shared_lib, "-Wl,-rpath," + str(Path(shared_lib).parent)])

        run_tool(cmd, stage="clang", error_cls=LLVMCompilationError)
        logger.info("Executable saved to: %s", output_path)
        return str(output_path)


class EdgeSlideMLIRPipeline:
    """
    Complete end-to-end pipeline for EdgeSlide MLIR compilation.

    Pipeline stages:
    1. PyTorch → MLIR (Linalg)
    2. Apply EdgeSlide Transform Dialect
    3. MLIR → LLVM IR
    4. LLVM IR → Executable binary
    """

    def __init__(self, config: Optional[EdgeSlideMLIRConfig] = None):
        self.config = config or EdgeSlideMLIRConfig()
        self._last_pass_results: Optional[Dict[str, Any]] = None

        self.pytorch_compiler = PyTorchToMLIRCompiler(self.config)
        self.transform_pass = EdgeSlideTransformPass(self.config)
        self.llvm_compiler = MLIRToLLVMCompiler(self.config)

    def parse_model(
        self, model: nn.Module, input_shape: tuple
    ) -> Tuple[Any, Any]:
        """Parse the model once, returning ``(parser, dag)``.

        The parser holds the annotated MLIR module (``save_mlir`` writes the
        payload); the DAG carries a per-op ``edgeslice_op_id`` linking analysis
        decisions back to payload ops.

        Public because callers outside the compile pipeline legitimately need
        stage 1 on its own: the alloc-attribution study builds untiled and
        tiled arms from one payload, and the upstream-SConv arm applies a
        different transform to the same payload. Both were reaching into the
        private name, which made this method's signature load-bearing without
        saying so.
        """
        _require_torch()
        from edgeslide.dag.parser import DAGParser

        # Seeded (see TRACING_SEED) so the traced graph is reproducible.
        generator = torch.Generator().manual_seed(TRACING_SEED)
        example_input = torch.randn(*input_shape, generator=generator)
        parser = DAGParser(model, example_input)
        dag = parser.parse()
        return parser, dag

    def _parse_model_with_annotations(
        self, model: nn.Module, input_shape: tuple
    ) -> Tuple[Any, Any]:
        """Deprecated alias for :meth:`parse_model`; kept for older callers."""
        return self.parse_model(model, input_shape)

    def _generate_transform_ir(
        self,
        model: nn.Module,
        input_shape: tuple,
        dag=None,
    ) -> str:
        """Generate transform.mlir via EdgeSlicePassManager."""
        from edgeslide.tiling.schedule_search import TargetAutotuner
        from edgeslide.transform.pass_manager import EdgeSlicePassConfig, EdgeSlicePassManager

        if dag is None:
            _, dag = self.parse_model(model, input_shape)
        target_measure = None
        if self.config.target_measure_once is not None:
            target_measure = TargetAutotuner(
                self.config.target_measure_once,
                warmups=self.config.autotune_warmups,
                trials=self.config.autotune_trials,
                cache_path=(
                    Path(self.config.autotune_cache_path)
                    if self.config.autotune_cache_path is not None
                    else None
                ),
                target_fingerprint=self.config.target_fingerprint,
            )
        search_options = {
            "planning_strategy": self.config.planning_strategy,
            "schedule_search_max_candidates": (
                self.config.schedule_search_max_candidates
            ),
            "schedule_shortlist_size": self.config.schedule_shortlist_size,
            "target_measure": target_measure,
        }
        if self.config.sram_capacity_kb is not None:
            pass_config = EdgeSlicePassConfig.for_sram(
                self.config.sram_capacity_kb * 1024,
                double_buffer=self.config.sram_double_buffer,
                sram_memory_space=self.config.sram_memory_space,
                sram_alignment_bytes=self.config.sram_alignment_bytes,
                **search_options,
            )
        else:
            pass_config = EdgeSlicePassConfig(
                l1_cache_bytes=self.config.l1_cache_kb * 1024,
                planning_objective="sram_residency",
                **search_options,
            )
        pass_manager = EdgeSlicePassManager(dag, config=pass_config)
        results = pass_manager.run(transform_style="structured")
        self._last_pass_results = results
        return results["transform_ir"]

    def compile(self, model: nn.Module,
                input_shape: tuple,
                model_name: str = "edgeslide_model",
                compile_to_binary: bool = False,
                compile_to_jit: bool = False,
                transform_mlir_path: Optional[str] = None,
                entry_func: str = "conv2d",
                run_id: Optional[str] = None) -> PipelineResults:
        """
        Run the complete compilation pipeline.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            model_name: Model name
            compile_to_binary: Whether to compile to binary
            run_id: Explicit id for this run's artifact directory. Defaults to
                a fresh random id, which is what keeps concurrent compilations
                sharing one output_dir from overwriting each other.

        Returns:
            Dictionary with paths to generated artifacts
        """
        # Each invocation gets its own directory, so two models compiling into
        # the same output_dir never share an artifact name.
        run = RunContext.create(self.config.output_dir, model_name, run_id)
        transform_pass = EdgeSlideTransformPass(self.config, run)
        llvm_compiler = MLIRToLLVMCompiler(self.config, run)

        logger.info("EdgeSlide MLIR compiler pipeline")
        logger.info("  model: %s", model_name)
        logger.info("  input shape: %s", input_shape)
        logger.info("  L1 cache: %d KB", self.config.l1_cache_kb)
        logger.info("  run directory: %s", run.directory)

        results: Dict[str, Any] = {}
        stages: List[StageResult] = []
        results["stages"] = stages
        results["run_id"] = run.run_id
        results["run_dir"] = str(run.directory)

        # Stage 1: PyTorch → annotated MLIR payload (edgeslice.op_id per Linalg op)
        logger.info("Stage 1: PyTorch to MLIR (Linalg, edgeslice.op_id annotated)")
        parser, dag = self.parse_model(model, input_shape)
        mlir_path = parser.save_mlir(str(run.path(f"{model_name}.mlir")))
        logger.info("MLIR saved to: %s", mlir_path)
        results["mlir"] = mlir_path
        stages.append(StageResult("torch_to_mlir", "ok", output_path=mlir_path))

        # Stage 2: Generate transform.mlir + run transform-interpreter
        logger.info("Stage 2: Transform dialect + transform interpreter")
        payload_path = transform_pass.write_payload_mlir(mlir_path)

        if transform_mlir_path:
            transform_path = str(transform_mlir_path)
        else:
            transform_ir = self._generate_transform_ir(model, input_shape, dag=dag)
            transform_path = transform_pass.write_transform_mlir(transform_ir)
            # Publish the planner evidence with the artifact paths.  This is
            # deliberately JSON-shaped so benchmark runners do not need to
            # reach into the pipeline's private ``_last_pass_results`` state.
            if self._last_pass_results is not None:
                results["planning"] = {
                    "ablation": self._last_pass_results.get("ablation", {}),
                    "rewrite_plan": self._last_pass_results.get("rewrite_plan", []),
                    "planning_rejections": self._last_pass_results.get(
                        "planning_rejections", []
                    ),
                    "compile_timing": self._last_pass_results.get(
                        "compile_timing", {}
                    ),
                }

        results["payload_mlir"] = payload_path
        results["transform_mlir"] = transform_path

        transform_ir = Path(transform_path).read_text()
        sliced_output = run.path("sliced_payload.mlir")
        # "Nothing to tile" and "tiling failed" are different outcomes. The copy
        # below is only ever the former; a transform script that exists but does
        # not apply raises out of apply_transform rather than falling back here.
        has_transform = (
            "No slicing candidates" not in transform_ir
            and "__transform_main" in transform_ir
        )
        if not has_transform:
            logger.info(
                "No slicing candidates — payload.mlir is used unchanged as "
                "sliced_payload.mlir"
            )
            with atomic_output(
                sliced_output, description="sliced payload MLIR"
            ) as tmp_sliced:
                shutil.copy2(payload_path, tmp_sliced)
            sliced_path = str(sliced_output)
            stages.append(StageResult(
                "transform_interpreter",
                "skipped",
                output_path=sliced_path,
                diagnostics="transform script contains no slicing candidates",
            ))
        else:
            sliced_path = transform_pass.apply_transform(
                payload_path,
                transform_path,
                output_name="sliced_payload.mlir",
            )
            stages.append(StageResult(
                "transform_interpreter", "ok", output_path=sliced_path
            ))
        results["sliced_payload_mlir"] = sliced_path
        results["transform_applied"] = has_transform

        # Explicit scratchpad placement is a distinct post-bufferization stage.
        # The generic CPU lowering cannot execute non-zero memory spaces, so
        # this produces a target-backend artifact and refuses a native CPU JIT
        # rather than silently treating the SRAM as ordinary heap memory.
        if self.config.sram_capacity_kb is not None and has_transform:
            if transform_mlir_path is not None or self._last_pass_results is None:
                raise MLIRTransformError(
                    "explicit SRAM placement requires the internally generated "
                    "rewrite plan; an external transform has no auditable SRAM plan"
                )
            from edgeslide.mlir_transform.sram import materialize_sram_placement

            promotion_ir = self._last_pass_results.get("sram_promotion_transform_ir")
            if not promotion_ir or "No SRAM promotion candidates" in promotion_ir:
                raise TransformVerificationError(
                    "SRAM mode transformed the payload but produced no promotion plan"
                )
            promotion_path = run.path("sram_promotion.mlir")
            atomic_write_text(
                promotion_path,
                promotion_ir,
                description="SRAM promotion transform",
            )
            promoted_path = run.path("sram_payload.mlir")
            placement = materialize_sram_placement(
                sliced_path,
                promotion_path,
                promoted_path,
                capacity_bytes=self.config.sram_capacity_kb * 1024,
                memory_space=self.config.sram_memory_space,
                double_buffer=self.config.sram_double_buffer,
            )
            results["sram_promotion_mlir"] = str(promotion_path)
            results["sram_payload_mlir"] = placement.output_path
            results["sram_placement"] = {
                "promoted_allocation_count": placement.promoted_allocation_count,
                "copy_count": placement.copy_count,
                "memory_space": placement.memory_space,
                "capacity_bytes": placement.capacity_bytes,
                "double_buffer_reserved": placement.double_buffer_reserved,
                "double_buffer_materialized": (
                    placement.double_buffer_materialized
                ),
                "sram_peak_bytes": placement.sram_peak_bytes,
                "capacity_verified": placement.capacity_verified,
                "inferred_size_count": placement.inferred_size_count,
            }
            stages.append(StageResult(
                "sram_promotion", "ok", output_path=placement.output_path
            ))
            if compile_to_binary or compile_to_jit:
                raise LLVMCompilationError(
                    "the generic CPU backend cannot lower explicit SRAM memory "
                    f"space {self.config.sram_memory_space}; provide a target "
                    "backend/DMA lowering and execute the sram_payload.mlir artifact"
                )

        # Stage 3–4: Lowering → LLVM IR / JIT / binary (optional)
        if compile_to_binary or compile_to_jit:
            logger.info("Stage 3: Sliced payload to LLVM dialect and LLVM IR")
            llvm_path = llvm_compiler.compile_to_llvm(
                sliced_path, entry_func=entry_func
            )
            results["llvm_ir"] = llvm_path
            results["llvm_dialect_mlir"] = str(run.path("sliced_payload_llvm.mlir"))
            stages.append(StageResult("mlir_to_llvm_ir", "ok", output_path=llvm_path))

            if compile_to_jit:
                logger.info("Stage 4a: JIT via torch_mlir ExecutionEngine")
                jit_engine = llvm_compiler.create_jit_engine(
                    sliced_path, entry_func=entry_func
                )
                results["jit_engine"] = jit_engine
                stages.append(StageResult("jit_engine", "ok"))
                logger.info("JIT engine ready (invoke with PyTorch memref tensors)")

            if compile_to_binary:
                logger.info("Stage 4b: LLVM IR to machine code (clang)")
                shared_lib = str(run.path("libedgeslide_runner_utils.so"))
                binary_path = llvm_compiler.compile_to_binary(
                    llvm_path,
                    model_name,
                    shared_lib=shared_lib if Path(shared_lib).is_file() else None,
                )
                results["binary"] = binary_path
                stages.append(StageResult(
                    "llvm_ir_to_binary", "ok", output_path=binary_path
                ))

        logger.info("Pipeline complete")
        for stage in stages:
            logger.info("  %s: %s %s", stage.name, stage.status, stage.output_path or "")

        return results


# Example usage
if __name__ == "__main__":
    # Simple test model
    class SimpleConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x

    # Create model
    model = SimpleConvModel()
    model.eval()

    # Run pipeline
    config = EdgeSlideMLIRConfig(
        l1_cache_kb=512,
        output_dir="build/mlir",
        opt_level="O2"
    )

    # CLI layer: this is the only place that turns log records into console
    # output. The library modules never print.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    pipeline = EdgeSlideMLIRPipeline(config)
    results = pipeline.compile(
        model,
        input_shape=(1, 3, 224, 224),
        model_name="simple_conv",
        compile_to_binary=False  # Skip binary compilation for now
    )
