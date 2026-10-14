"""
Two pipeline runs sharing one output directory must not touch each other's files.

Before per-run directories every invocation wrote ``payload.mlir``,
``transform.mlir`` and ``sliced_payload.mlir`` into the configured output
directory under those exact names. Two models compiling concurrently therefore
overwrote each other, and because each stage reads the file the previous stage
*named* rather than the one it wrote, a run could lower another run's payload
and still report success.
"""
from __future__ import annotations

import stat
import threading
from pathlib import Path

import pytest

from edgeslide.mlir_transform.errors import atomic_output, atomic_write_text
from edgeslide.mlir_transform.pipeline import (
    EdgeSlideMLIRConfig,
    EdgeSlideTransformPass,
    RunContext,
    find_mlir_opt,
    new_run_id,
    run_transform_interpreter,
)

pytestmark = pytest.mark.cpu

PAYLOAD_A = """module {
  func.func @forward(%in: tensor<1x3x34x34xf32>, %w: tensor<8x3x3x3xf32>,
                     %out: tensor<1x8x32x32xf32>) -> tensor<1x8x32x32xf32> {
    %0 = linalg.conv_2d_nchw_fchw {edgeslice.op_id = "op_a"}
      ins(%in, %w : tensor<1x3x34x34xf32>, tensor<8x3x3x3xf32>)
      outs(%out : tensor<1x8x32x32xf32>) -> tensor<1x8x32x32xf32>
    return %0 : tensor<1x8x32x32xf32>
  }
}
"""

# Deliberately different shapes so the two runs' IR cannot be confused.
PAYLOAD_B = PAYLOAD_A.replace("op_a", "op_b").replace("8x3x3x3", "16x3x3x3").replace(
    "1x8x32x32", "1x16x32x32"
)


def _transform_for(tag: str) -> str:
    return f"""module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {{
    %t = transform.structured.match ops{{["linalg.conv_2d_nchw_fchw"]}}
      attributes {{edgeslice.op_id = "{tag}"}} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall %t
      tile_sizes [0, 0, 8, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }}
}}
"""


def _mlir_opt_or_skip() -> str:
    mlir_opt = find_mlir_opt()
    if mlir_opt is None:
        pytest.skip("mlir-opt not found; set $MLIR_OPT or install LLVM/MLIR")
    return mlir_opt


# ---------------------------------------------------------------------------
# RunContext naming
# ---------------------------------------------------------------------------


def test_two_runs_of_the_same_model_get_distinct_directories(tmp_path):
    first = RunContext.create(tmp_path, "resnet50")
    second = RunContext.create(tmp_path, "resnet50")

    assert first.directory != second.directory
    assert first.path("payload.mlir") != second.path("payload.mlir")
    assert first.directory.is_dir() and second.directory.is_dir()


def test_run_directory_carries_model_name_and_run_id(tmp_path):
    run = RunContext.create(tmp_path, "yolov5", run_id="abc123")

    assert run.directory.name == "yolov5_abc123"
    assert run.directory.parent == tmp_path


def test_different_models_do_not_share_a_directory(tmp_path):
    a = RunContext.create(tmp_path, "resnet50")
    b = RunContext.create(tmp_path, "yolov5")

    assert a.directory != b.directory


def test_run_ids_are_unique_across_many_calls():
    ids = {new_run_id() for _ in range(2000)}
    assert len(ids) == 2000


def test_explicit_run_id_is_reproducible(tmp_path):
    """Callers that want a stable path can pin one."""
    a = RunContext.create(tmp_path, "m", run_id="fixed")
    b = RunContext.create(tmp_path, "m", run_id="fixed")

    assert a.directory == b.directory


# ---------------------------------------------------------------------------
# Stage helpers honour the run context
# ---------------------------------------------------------------------------


def test_transform_pass_writes_into_its_run_directory(tmp_path):
    config = EdgeSlideMLIRConfig(output_dir=str(tmp_path))
    run_a = RunContext.create(tmp_path, "model", run_id="aaa")
    run_b = RunContext.create(tmp_path, "model", run_id="bbb")

    path_a = EdgeSlideTransformPass(config, run_a).write_transform_mlir("// A\n")
    path_b = EdgeSlideTransformPass(config, run_b).write_transform_mlir("// B\n")

    assert path_a != path_b
    # The decisive check: writing B must not have changed A.
    assert Path(path_a).read_text() == "// A\n"
    assert Path(path_b).read_text() == "// B\n"


def test_same_filename_in_two_runs_holds_different_content(tmp_path):
    config = EdgeSlideMLIRConfig(output_dir=str(tmp_path))
    runs = [RunContext.create(tmp_path, "model") for _ in range(5)]

    written = [
        EdgeSlideTransformPass(config, run).write_transform_mlir(f"// run {index}\n")
        for index, run in enumerate(runs)
    ]

    assert len({Path(p).name for p in written}) == 1, "same basename expected"
    assert len(set(written)) == 5, "paths must still be distinct"
    for index, path in enumerate(written):
        assert Path(path).read_text() == f"// run {index}\n"


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


def test_atomic_output_leaves_no_file_when_the_stage_fails(tmp_path):
    target = tmp_path / "out.mlir"

    with pytest.raises(RuntimeError):
        with atomic_output(target) as tmp:
            tmp.write_text("half-written IR")
            raise RuntimeError("stage failed")

    assert not target.exists(), "a failed stage must not leave an output artifact"
    assert not list(tmp_path.glob("*.tmp")), "temp file was not cleaned up"


def test_atomic_output_does_not_clobber_an_existing_file_on_failure(tmp_path):
    target = tmp_path / "out.mlir"
    target.write_text("previous good IR")

    with pytest.raises(RuntimeError):
        with atomic_output(target) as tmp:
            tmp.write_text("garbage")
            raise RuntimeError("boom")

    assert target.read_text() == "previous good IR"


def test_atomic_output_reports_a_tool_that_wrote_nothing(tmp_path):
    """Exit 0 with no output file is a failure, not an empty success."""
    with pytest.raises(FileNotFoundError, match="was not produced"):
        with atomic_output(tmp_path / "out.mlir"):
            pass


def test_atomic_write_text_replaces_content_wholesale(tmp_path):
    target = tmp_path / "a.txt"
    atomic_write_text(target, "one")
    atomic_write_text(target, "two")

    assert target.read_text() == "two"
    assert not list(tmp_path.glob(".*tmp"))


# ---------------------------------------------------------------------------
# Real interpreter, concurrent
# ---------------------------------------------------------------------------


@pytest.mark.mlir
def test_two_sequential_interpreter_runs_keep_independent_artifacts(tmp_path):
    _mlir_opt_or_skip()
    config = EdgeSlideMLIRConfig(output_dir=str(tmp_path))

    outputs = {}
    for tag, payload in (("op_a", PAYLOAD_A), ("op_b", PAYLOAD_B)):
        run = RunContext.create(tmp_path, "model")
        transform_pass = EdgeSlideTransformPass(config, run)
        payload_path = run.path("payload.mlir")
        payload_path.write_text(payload)
        transform_path = transform_pass.write_transform_mlir(_transform_for(tag))
        outputs[tag] = Path(
            transform_pass.apply_transform(str(payload_path), transform_path)
        )

    assert outputs["op_a"] != outputs["op_b"]
    text_a = outputs["op_a"].read_text()
    text_b = outputs["op_b"].read_text()
    # Each output must carry its own op id and its own shapes.
    assert "op_a" in text_a and "op_b" not in text_a
    assert "op_b" in text_b and "op_a" not in text_b
    assert "1x8x32x32" in text_a
    assert "1x16x32x32" in text_b


@pytest.mark.mlir
def test_concurrent_interpreter_runs_do_not_cross_contaminate(tmp_path):
    """The regression this phase exists for, exercised with real threads."""
    _mlir_opt_or_skip()
    config = EdgeSlideMLIRConfig(output_dir=str(tmp_path))

    results = {}
    errors = []
    barrier = threading.Barrier(2)

    def compile_one(tag: str, payload: str) -> None:
        try:
            run = RunContext.create(tmp_path, "shared_model")
            transform_pass = EdgeSlideTransformPass(config, run)
            payload_path = run.path("payload.mlir")
            payload_path.write_text(payload)
            transform_path = transform_pass.write_transform_mlir(_transform_for(tag))
            # Both threads reach the interpreter at the same moment.
            barrier.wait(timeout=60)
            results[tag] = Path(
                transform_pass.apply_transform(str(payload_path), transform_path)
            )
        except BaseException as exc:  # surfaced in the assertions below
            errors.append((tag, exc))
            try:
                barrier.abort()
            except Exception:
                pass

    threads = [
        threading.Thread(target=compile_one, args=("op_a", PAYLOAD_A)),
        threading.Thread(target=compile_one, args=("op_b", PAYLOAD_B)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert not errors, f"concurrent compile failed: {errors}"
    assert set(results) == {"op_a", "op_b"}
    assert results["op_a"] != results["op_b"]

    text_a = results["op_a"].read_text()
    text_b = results["op_b"].read_text()
    assert "op_a" in text_a and "op_b" not in text_a
    assert "op_b" in text_b and "op_a" not in text_b
    assert "1x8x32x32" in text_a and "1x16x32x32" in text_b
    assert "scf.forall" in text_a and "scf.forall" in text_b


@pytest.mark.mlir
def test_failed_run_leaves_no_sliced_payload_beside_a_good_one(tmp_path):
    """A failure must not look like a success in the directory listing."""
    _mlir_opt_or_skip()
    config = EdgeSlideMLIRConfig(output_dir=str(tmp_path))

    good_run = RunContext.create(tmp_path, "model", run_id="good")
    good_pass = EdgeSlideTransformPass(config, good_run)
    good_payload = good_run.path("payload.mlir")
    good_payload.write_text(PAYLOAD_A)
    good_transform = good_pass.write_transform_mlir(_transform_for("op_a"))
    good_sliced = Path(good_pass.apply_transform(str(good_payload), good_transform))

    bad_run = RunContext.create(tmp_path, "model", run_id="bad")
    bad_pass = EdgeSlideTransformPass(config, bad_run)
    bad_payload = bad_run.path("payload.mlir")
    bad_payload.write_text("module { this is not valid MLIR")
    bad_transform = bad_pass.write_transform_mlir(_transform_for("op_a"))

    with pytest.raises(RuntimeError):
        bad_pass.apply_transform(str(bad_payload), bad_transform)

    assert good_sliced.is_file(), "the healthy run's artifact was collateral damage"
    assert "scf.forall" in good_sliced.read_text()
    assert not bad_run.path("sliced_payload.mlir").exists()
    # No scratch files survive either run.
    assert not list(bad_run.directory.glob("*_merged_*.mlir"))
    assert not list(bad_run.directory.glob("*_interpreted_*.mlir"))


@pytest.mark.mlir
def test_interpreter_cleans_up_its_intermediates(tmp_path):
    _mlir_opt_or_skip()
    run = RunContext.create(tmp_path, "model")
    payload = run.path("payload.mlir")
    payload.write_text(PAYLOAD_A)
    transform = run.path("transform.mlir")
    transform.write_text(_transform_for("op_a"))

    run_transform_interpreter(
        str(payload), str(transform), str(run.path("sliced_payload.mlir"))
    )

    leftovers = sorted(
        p.name for p in run.directory.iterdir() if "_merged_" in p.name or "_interpreted_" in p.name
    )
    assert leftovers == [], f"scratch files left behind: {leftovers}"


# ---------------------------------------------------------------------------
# Stage-level atomicity, driven by a stub mlir-opt
# ---------------------------------------------------------------------------


def _stub_mlir_opt(tmp_path: Path, *, fail_on: str) -> str:
    """A fake mlir-opt that writes output but fails on the named pass.

    The real binary rarely fails at the canonicalize/cse step, so standing in
    for it is the only way to exercise that stage's failure handling.
    """
    template = """#!/bin/sh
# Emulate `mlir-opt IN <passes> -o OUT`.
out=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "-o" ]; then out="$arg"; fi
  prev="$arg"
done
for arg in "$@"; do
  if [ "$arg" = "__FAIL_ON__" ]; then
    echo "stub failure on __FAIL_ON__" >&2
    # Write a partial file first: without atomic promotion this truncated
    # output would survive at the destination.
    if [ -n "$out" ]; then printf 'PARTIAL GARBAGE' > "$out"; fi
    exit 1
  fi
done
if [ -n "$out" ]; then printf 'module { }\n' > "$out"; fi
exit 0
"""
    script = tmp_path / "stub_mlir_opt"
    script.write_text(template.replace("__FAIL_ON__", fail_on))
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


def test_cleanup_stage_failure_leaves_no_output_file(tmp_path):
    """mlir-opt --canonicalize failing must not leave truncated IR behind."""
    run = RunContext.create(tmp_path, "model")
    payload = run.path("payload.mlir")
    payload.write_text(PAYLOAD_A)
    transform = run.path("transform.mlir")
    transform.write_text(_transform_for("op_a"))
    output = run.path("sliced_payload.mlir")

    stub = _stub_mlir_opt(tmp_path, fail_on="--canonicalize")

    with pytest.raises(RuntimeError):
        run_transform_interpreter(
            str(payload),
            str(transform),
            str(output),
            mlir_opt=stub,
            verify_transformed=False,
        )

    assert not output.exists(), "partial output survived a failed cleanup stage"
    assert not list(run.directory.glob("*.tmp"))
    assert not list(run.directory.glob("*_interpreted_*.mlir"))


def test_cleanup_stage_failure_does_not_clobber_a_previous_good_output(tmp_path):
    run = RunContext.create(tmp_path, "model")
    payload = run.path("payload.mlir")
    payload.write_text(PAYLOAD_A)
    transform = run.path("transform.mlir")
    transform.write_text(_transform_for("op_a"))

    output = run.path("sliced_payload.mlir")
    output.write_text("previously produced good IR")

    stub = _stub_mlir_opt(tmp_path, fail_on="--canonicalize")

    with pytest.raises(RuntimeError):
        run_transform_interpreter(
            str(payload),
            str(transform),
            str(output),
            mlir_opt=stub,
            verify_transformed=False,
        )

    assert output.read_text() == "previously produced good IR"


def test_interpreter_stage_failure_leaves_no_output_file(tmp_path):
    run = RunContext.create(tmp_path, "model")
    payload = run.path("payload.mlir")
    payload.write_text(PAYLOAD_A)
    transform = run.path("transform.mlir")
    transform.write_text(_transform_for("op_a"))
    output = run.path("sliced_payload.mlir")

    stub = _stub_mlir_opt(tmp_path, fail_on="--transform-interpreter")

    with pytest.raises(RuntimeError):
        run_transform_interpreter(
            str(payload),
            str(transform),
            str(output),
            mlir_opt=stub,
            verify_transformed=False,
        )

    assert not output.exists()
    assert not list(run.directory.glob("*_merged_*.mlir"))


# ---------------------------------------------------------------------------
# End-to-end concurrent compile (requires torch + torch_mlir)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.mlir
def test_two_models_compile_concurrently_into_one_output_dir(tmp_path):
    """Phase 2's headline requirement, end to end.

    Two different models compile at the same moment into the *same* configured
    output directory. Each must come back with its own artifacts, and neither
    IR may contain the other's tensor shapes.

    This also covers a second hazard: torch-mlir's lazy dialect import is not
    thread-safe, so before DAGParser serialised it this raced and one thread
    died with ``KeyError: 'torch_mlir.dialects.linalg'`` mid-traversal.
    """
    pytest.importorskip("torch", reason="PyTorch is required")
    pytest.importorskip("torch_mlir", reason="torch-mlir is required")
    _mlir_opt_or_skip()

    import torch.nn as nn

    from edgeslide.mlir_transform.pipeline import EdgeSlideMLIRPipeline

    class ConvA(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    class ConvB(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 5, padding=2)

        def forward(self, x):
            return self.conv(x)

    results = {}
    errors = []
    barrier = threading.Barrier(2)

    def compile_model(name, model):
        try:
            config = EdgeSlideMLIRConfig(l1_cache_kb=16, output_dir=str(tmp_path))
            pipeline = EdgeSlideMLIRPipeline(config)
            barrier.wait(timeout=120)
            results[name] = pipeline.compile(
                model, (1, 3, 64, 64), model_name=name, entry_func="forward"
            )
        except BaseException as exc:
            errors.append((name, repr(exc)))
            try:
                barrier.abort()
            except Exception:
                pass

    threads = [
        threading.Thread(target=compile_model, args=("model_a", ConvA().eval())),
        threading.Thread(target=compile_model, args=("model_b", ConvB().eval())),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=600)

    assert not errors, f"concurrent compile failed: {errors}"
    assert set(results) == {"model_a", "model_b"}

    a, b = results["model_a"], results["model_b"]
    assert a["run_dir"] != b["run_dir"]
    assert a["payload_mlir"] != b["payload_mlir"]
    assert a["sliced_payload_mlir"] != b["sliced_payload_mlir"]

    # The decisive check: neither run's original payload carries the other's
    # channel count.  Do not inspect the tiled payload here: a legal tile size
    # (for example, an 8-channel tile of a 16-channel result) can naturally
    # equal the other model's full channel extent without any artifact leak.
    text_a = Path(a["payload_mlir"]).read_text()
    text_b = Path(b["payload_mlir"]).read_text()
    assert "x8x" in text_a and "x16x" not in text_a
    assert "x16x" in text_b and "x8x" not in text_b

    # The shared directory holds exactly the two run directories.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        Path(a["run_dir"]).name,
        Path(b["run_dir"]).name,
    ]
