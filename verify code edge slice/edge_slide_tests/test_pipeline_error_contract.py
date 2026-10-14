"""
The pipeline must fail loudly instead of handing back untransformed input.

Every case here is a way the old code could report success for a compile that
did not happen: a transform script that was never read, a payload that does not
parse, a missing tool reported as a warning, a stage returning ``None``. The
tests assert the typed exception and, where a real toolchain is present, that
the transformed output actually differs from the input.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from edgeslide.mlir_transform.errors import (
    ExternalToolError,
    LLVMCompilationError,
    MLIRTransformError,
    StageResult,
    ToolNotFoundError,
    prepare_output_path,
    require_input_file,
    require_tool,
    run_tool,
)
from edgeslide.mlir_transform.pipeline import (
    EdgeSlideMLIRConfig,
    EdgeSlideTransformPass,
    MLIRToLLVMCompiler,
    find_mlir_opt,
    run_transform_interpreter,
)

# A conv large enough that tiling it visibly changes the IR.
PAYLOAD = """module {
  func.func @forward(%in: tensor<1x3x34x34xf32>, %w: tensor<8x3x3x3xf32>,
                     %out: tensor<1x8x32x32xf32>) -> tensor<1x8x32x32xf32> {
    %0 = linalg.conv_2d_nchw_fchw {edgeslice.op_id = "op_0"}
      ins(%in, %w : tensor<1x3x34x34xf32>, tensor<8x3x3x3xf32>)
      outs(%out : tensor<1x8x32x32xf32>) -> tensor<1x8x32x32xf32>
    return %0 : tensor<1x8x32x32xf32>
  }
}
"""

TRANSFORM = """module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %t = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]}
      attributes {edgeslice.op_id = "op_0"} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall %t
      tile_sizes [0, 0, 8, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
"""

# A transform script that parses but matches nothing in the payload above.
TRANSFORM_MATCHES_NOTHING = TRANSFORM.replace("op_0", "op_does_not_exist")

INVALID_PAYLOAD = """module {
  func.func @forward(%arg0: tensor<1x3x32x32xf32>) -> tensor<1x8x32x32xf32> {
    %0 = linalg.this_op_does_not_exist %arg0 garbage syntax here
  }
}
"""


def _mlir_opt_or_skip() -> str:
    mlir_opt = find_mlir_opt()
    if mlir_opt is None:
        pytest.skip("mlir-opt not found; set $MLIR_OPT or install LLVM/MLIR")
    return mlir_opt


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


@pytest.mark.cpu
def test_missing_payload_is_reported_as_a_missing_file(tmp_path):
    """Not as an mlir-opt parse diagnostic, which reads as 'your IR is broken'."""
    transform = _write(tmp_path, "transform.mlir", TRANSFORM)
    with pytest.raises(FileNotFoundError, match="payload MLIR does not exist"):
        run_transform_interpreter(
            str(tmp_path / "absent.mlir"), str(transform), str(tmp_path / "out.mlir")
        )


@pytest.mark.cpu
def test_missing_transform_script_is_reported_as_a_missing_file(tmp_path):
    payload = _write(tmp_path, "payload.mlir", PAYLOAD)
    with pytest.raises(FileNotFoundError, match="transform MLIR does not exist"):
        run_transform_interpreter(
            str(payload), str(tmp_path / "absent.mlir"), str(tmp_path / "out.mlir")
        )


@pytest.mark.cpu
def test_empty_payload_is_rejected(tmp_path):
    payload = _write(tmp_path, "payload.mlir", "")
    transform = _write(tmp_path, "transform.mlir", TRANSFORM)
    with pytest.raises(ValueError, match="payload MLIR is empty"):
        run_transform_interpreter(
            str(payload), str(transform), str(tmp_path / "out.mlir")
        )


@pytest.mark.cpu
def test_require_input_file_rejects_a_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a regular file"):
        require_input_file(tmp_path, "payload MLIR")


@pytest.mark.cpu
def test_prepare_output_path_creates_parents_and_rejects_directories(tmp_path):
    target = prepare_output_path(tmp_path / "nested" / "out.mlir", "output")
    assert target.parent.is_dir()
    with pytest.raises(IsADirectoryError):
        prepare_output_path(tmp_path, "output")


# --------------------------------------------------------------------------
# Tool resolution
# --------------------------------------------------------------------------


@pytest.mark.cpu
def test_missing_tool_raises_tool_not_found_naming_the_env_var():
    with pytest.raises(ToolNotFoundError) as excinfo:
        require_tool("edgeslide-tool-that-does-not-exist")
    message = str(excinfo.value)
    assert "EDGESLIDE_TOOL_THAT_DOES_NOT_EXIST" in message
    assert "Searched" in message


@pytest.mark.cpu
def test_tool_not_found_is_catchable_as_file_not_found():
    """Backward compatibility for existing except FileNotFoundError handlers."""
    with pytest.raises(FileNotFoundError):
        require_tool("edgeslide-tool-that-does-not-exist")


@pytest.mark.cpu
def test_explicit_tool_path_that_does_not_exist_is_rejected(tmp_path):
    with pytest.raises(ToolNotFoundError):
        require_tool("mlir-opt", str(tmp_path / "nope"))


# --------------------------------------------------------------------------
# External tool failure carries full diagnostics
# --------------------------------------------------------------------------


@pytest.mark.cpu
def test_failed_tool_error_records_command_returncode_and_streams():
    with pytest.raises(ExternalToolError) as excinfo:
        run_tool(
            ["sh", "-c", "echo out-text; echo err-text >&2; exit 3"],
            stage="unit-test stage",
        )
    error = excinfo.value
    assert error.returncode == 3
    assert error.command[0] == "sh"
    assert "out-text" in error.stdout
    assert "err-text" in error.stderr
    # The message alone must be enough to debug without re-running the tool.
    rendered = str(error)
    assert "exit 3" in rendered and "err-text" in rendered


@pytest.mark.cpu
def test_tool_error_subclasses_are_distinguishable():
    assert issubclass(MLIRTransformError, ExternalToolError)
    assert issubclass(LLVMCompilationError, ExternalToolError)
    assert not issubclass(MLIRTransformError, LLVMCompilationError)


# --------------------------------------------------------------------------
# Real interpreter behaviour
# --------------------------------------------------------------------------


@pytest.mark.mlir
def test_invalid_payload_fails_instead_of_falling_back(tmp_path):
    """Unparseable MLIR must raise — never be copied through as 'sliced'."""
    _mlir_opt_or_skip()
    payload = _write(tmp_path, "payload.mlir", INVALID_PAYLOAD)
    transform = _write(tmp_path, "transform.mlir", TRANSFORM)
    output = tmp_path / "sliced.mlir"

    with pytest.raises(MLIRTransformError) as excinfo:
        run_transform_interpreter(str(payload), str(transform), str(output))

    assert excinfo.value.returncode != 0
    assert excinfo.value.stderr.strip()
    assert not output.exists(), "a failed transform must not leave an output artifact"


@pytest.mark.mlir
def test_transform_script_is_consumed_and_changes_the_payload(tmp_path):
    """The transform must actually be applied, not merely accepted."""
    _mlir_opt_or_skip()
    payload = _write(tmp_path, "payload.mlir", PAYLOAD)
    transform = _write(tmp_path, "transform.mlir", TRANSFORM)
    output = tmp_path / "sliced.mlir"

    result_path = run_transform_interpreter(
        str(payload), str(transform), str(output)
    )
    sliced = Path(result_path).read_text()

    assert sliced != PAYLOAD, "output is identical to input — transform was a no-op"
    assert "scf.forall" in sliced
    assert "tensor.extract_slice" in sliced
    assert "tensor.parallel_insert_slice" in sliced

    # The interpreter rewrites the payload but leaves the transform module in
    # the output; lowering drops it later via strip_transform_dialect.
    from edgeslide.mlir_transform.lowering import strip_transform_dialect

    payload_only = strip_transform_dialect(sliced)
    assert "transform.named_sequence" not in payload_only
    assert "scf.forall" in payload_only


@pytest.mark.mlir
def test_transform_that_matches_nothing_is_reported_not_ignored(tmp_path):
    """mlir-opt exits 0 for a script that matches nothing; the check must not."""
    _mlir_opt_or_skip()
    payload = _write(tmp_path, "payload.mlir", PAYLOAD)
    transform = _write(tmp_path, "transform.mlir", TRANSFORM_MATCHES_NOTHING)

    with pytest.raises(RuntimeError, match="materializing"):
        run_transform_interpreter(
            str(payload), str(transform), str(tmp_path / "sliced.mlir")
        )


@pytest.mark.mlir
def test_apply_transform_validates_paths_before_running(tmp_path):
    _mlir_opt_or_skip()
    transform_pass = EdgeSlideTransformPass(
        EdgeSlideMLIRConfig(output_dir=str(tmp_path / "build"))
    )
    transform = _write(tmp_path, "transform.mlir", TRANSFORM)
    with pytest.raises(FileNotFoundError):
        transform_pass.apply_transform(str(tmp_path / "absent.mlir"), str(transform))


@pytest.mark.mlir
def test_apply_transform_produces_a_tiled_payload(tmp_path):
    _mlir_opt_or_skip()
    transform_pass = EdgeSlideTransformPass(
        EdgeSlideMLIRConfig(output_dir=str(tmp_path / "build"))
    )
    payload = _write(tmp_path, "payload.mlir", PAYLOAD)
    transform = _write(tmp_path, "transform.mlir", TRANSFORM)

    sliced = Path(transform_pass.apply_transform(str(payload), str(transform)))
    assert sliced.is_file()
    assert "scf.forall" in sliced.read_text()


# --------------------------------------------------------------------------
# Lowering never reports failure as success
# --------------------------------------------------------------------------


@pytest.mark.cpu
def test_compile_to_llvm_raises_on_missing_input(tmp_path):
    """It used to catch the error, print a warning, and return None."""
    # compile_to_llvm imports edgeslide.mlir_transform.lowering, which needs
    # numpy at module level; the file's other tests stay numpy-free.
    pytest.importorskip("numpy", reason="numpy is required by the lowering module")
    compiler = MLIRToLLVMCompiler(EdgeSlideMLIRConfig(output_dir=str(tmp_path)))
    with pytest.raises(FileNotFoundError):
        compiler.compile_to_llvm(str(tmp_path / "absent.mlir"))


@pytest.mark.cpu
def test_compile_to_binary_raises_on_missing_input(tmp_path):
    compiler = MLIRToLLVMCompiler(EdgeSlideMLIRConfig(output_dir=str(tmp_path)))
    with pytest.raises(FileNotFoundError):
        compiler.compile_to_binary(str(tmp_path / "absent.ll"))


@pytest.mark.cpu
def test_stage_result_never_presents_a_failure_as_usable():
    failed = StageResult("lowering", "failed", diagnostics="clang exited 1")
    assert not failed.ok
    assert failed.output_path is None
    assert StageResult("lowering", "ok", output_path="/tmp/x.ll").ok
    assert failed.as_dict()["status"] == "failed"


# --------------------------------------------------------------------------
# End-to-end native compilation smoke test
# --------------------------------------------------------------------------


@pytest.mark.hardware
def test_minimal_module_compiles_to_a_runnable_executable(tmp_path):
    """mlir-opt -> mlir-translate -> clang, then actually run the binary."""
    from edgeslide.mlir_transform.lowering import find_mlir_tool

    missing = [
        name
        for name in ("mlir-opt", "mlir-translate", "clang")
        if find_mlir_tool(name) is None
    ]
    if missing:
        pytest.skip(f"native toolchain incomplete: {', '.join(missing)} not found")

    mlir_opt = find_mlir_tool("mlir-opt")
    mlir_translate = find_mlir_tool("mlir-translate")

    source = _write(
        tmp_path,
        "minimal.mlir",
        """module {
  func.func @main() -> i32 {
    %0 = arith.constant 0 : i32
    return %0 : i32
  }
}
""",
    )

    llvm_dialect = tmp_path / "minimal_llvm.mlir"
    subprocess.run(
        [
            mlir_opt,
            str(source),
            "--convert-func-to-llvm",
            "--convert-arith-to-llvm",
            "--reconcile-unrealized-casts",
            "-o",
            str(llvm_dialect),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # Extension matches the artifact: .ll is textual LLVM IR.
    llvm_ir = tmp_path / "minimal.ll"
    subprocess.run(
        [mlir_translate, "--mlir-to-llvmir", str(llvm_dialect), "-o", str(llvm_ir)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert llvm_ir.is_file() and llvm_ir.read_text().strip()

    compiler = MLIRToLLVMCompiler(EdgeSlideMLIRConfig(output_dir=str(tmp_path)))
    # The executable path carries no extension.
    binary = Path(compiler.compile_to_binary(str(llvm_ir), "minimal_exe"))
    assert binary.is_file(), "clang reported success but produced no executable"

    run = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
