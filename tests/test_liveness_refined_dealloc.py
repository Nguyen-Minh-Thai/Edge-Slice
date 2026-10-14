"""Tests for the liveness-refined ownership deallocation pass in ``mlir/``.

The pass takes the single merged ``bufferization.dealloc`` that
``--ownership-based-buffer-deallocation`` parks at a block's exit and
discharges from it every buffer whose ownership is statically held and whose
liveness stays inside the block, placing that free at the buffer's last use.

Two properties matter and both are checked here rather than argued:

* it must free earlier -- otherwise it does nothing, and the 4x-8x peak it
  exists to recover stays lost;
* it must never free earlier than it may -- every case the analysis cannot
  prove has to come back as a *named* refusal and stay in the block-exit
  dealloc, because a wrong answer here is a use-after-free.

The refusal vocabulary is asserted against a fixed list, so a new refusal
reason cannot slip into a result file as an unrecognised string.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT / "src"), str(ROOT)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

pytest.importorskip("torch", reason="benchmarks.alloc_attribution imports the lowering stack")

from benchmarks.alloc_attribution import (  # noqa: E402
    ArmUnavailable,
    DEALLOC_PLUGIN_ENV,
    PIPELINE_VARIANTS,
    dealloc_plugin_path,
)
from edgeslide.liveness.analyzer import BufferizedMemoryAnalyzer  # noqa: E402
from edgeslide.mlir_transform.lowering import find_mlir_tool  # noqa: E402

# Every decision string the pass can emit. Asserted against the pass output so
# that adding a reason without adding it here fails a test instead of arriving
# in a result file as an unexplained label.
KNOWN_DECISIONS = {
    "sunk",
    "condition-not-statically-true",
    "operand-is-not-an-allocation",
    "allocation-in-another-block",
    "retained-alias-exists",
    "alias-used-outside-block",
    "unmodelled-aliasing-user",
    "call-may-capture-buffer",
    "last-use-is-terminator",
    "already-deallocated",
    "dealloc-precedes-last-use",
    "undeclared-memory-effects",
}

OWNERSHIP_ONLY = "ownership-based-buffer-deallocation,canonicalize"
REFINED = OWNERSHIP_ONLY + ",edgeslide-liveness-refined-dealloc{report=%s}"

# The deallocation tail the sweep actually measures. It matters that the peak
# comparison uses this and not OWNERSHIP_ONLY: the merged
# `bufferization.dealloc` names every buffer it owns as an operand, so while it
# is still in the IR it *is* a use of each of them, sitting at the block exit.
# Measured there, the last-use bound comes back as the sum of every allocation
# -- an artefact of the measurement point, not of the program.
# `bufferization-lower-deallocations` replaces it with per-buffer
# `memref.dealloc`, which the analyzer reads as a release rather than a use.
LOWERING_TAIL = ",buffer-deallocation-simplification,bufferization-lower-deallocations"
OWNERSHIP_FULL = OWNERSHIP_ONLY + LOWERING_TAIL
REFINED_FULL = REFINED + LOWERING_TAIL

# Three buffers in a chain: the first two die inside the block, the third is
# returned. The minimal shape of every straight-line forward pass.
CHAIN = """
func.func @chain(%arg0: memref<1024xf32>) -> memref<1024xf32> {
  %a = memref.alloc() : memref<1024xf32>
  memref.copy %arg0, %a : memref<1024xf32> to memref<1024xf32>
  %b = memref.alloc() : memref<1024xf32>
  memref.copy %a, %b : memref<1024xf32> to memref<1024xf32>
  %c = memref.alloc() : memref<1024xf32>
  memref.copy %b, %c : memref<1024xf32> to memref<1024xf32>
  return %c : memref<1024xf32>
}
"""


@pytest.fixture(scope="session")
def mlir_opt() -> str:
    tool = find_mlir_tool("mlir-opt")
    if tool is None:
        pytest.skip("mlir-opt not found")
    return tool


@pytest.fixture(scope="session")
def plugin() -> Path:
    """The built plugin, building it first if the toolchain is present.

    Skipping outright when the .so is merely absent would leave the pass
    untested on every machine that could in fact run it, which is most of them:
    the build needs only clang++ and the MLIR headers already required to run
    any of the lowering tests.
    """
    path = ROOT / "mlir" / "libEdgeSlideDealloc.so"
    if path.is_file():
        return path
    script = ROOT / "mlir" / "build.sh"
    try:
        subprocess.run(
            ["bash", str(script)], check=True, capture_output=True, timeout=900
        )
    except Exception as exc:  # no compiler, no headers, or a build error
        pytest.skip(f"deallocation plugin is not built and cannot be built here: {exc}")
    return path


def run_pipeline(mlir_opt: str, plugin: Path, source: str, pipeline: str,
                 tmp_path: Path) -> str:
    """Run one textual pass pipeline over *source*, returning the printed IR."""
    src = tmp_path / "in.mlir"
    src.write_text(source)
    result = subprocess.run(
        [
            mlir_opt,
            str(src),
            f"--load-pass-plugin={plugin}",
            f"--pass-pipeline=builtin.module({pipeline})",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def refine(mlir_opt: str, plugin: Path, source: str, tmp_path: Path):
    """Ownership deallocation followed by the refinement. Returns (IR, report)."""
    report = tmp_path / "report.json"
    text = run_pipeline(mlir_opt, plugin, source, REFINED % report, tmp_path)
    return text, json.loads(report.read_text())


def dealloc_order(text: str) -> list[str]:
    """Operation mnemonics in order, for asserting where a free landed."""
    return [
        line.strip().split(" ")[0].lstrip("%").split("=")[0].strip() or line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def test_refinement_frees_each_dead_buffer_at_its_last_use(mlir_opt, plugin, tmp_path):
    text, report = refine(mlir_opt, plugin, CHAIN, tmp_path)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Each free must sit immediately after the copy that last reads the buffer,
    # not at the block exit where the ownership pass left it.
    copies = [i for i, ln in enumerate(lines) if ln.startswith("memref.copy")]
    frees = [i for i, ln in enumerate(lines) if ln.startswith("memref.dealloc")]
    assert len(copies) == 3
    assert len(frees) == 2, text
    assert frees[0] == copies[1] + 1, text
    assert frees[1] == copies[2] + 1, text

    decisions = [pair["decision"] for pair in report["pairs"]]
    assert decisions.count("sunk") == 2
    assert "retained-alias-exists" in decisions


def test_ownership_alone_parks_every_free_at_the_block_exit(mlir_opt, plugin, tmp_path):
    """The control. Without the refinement no free moves, which is the point."""
    text = run_pipeline(mlir_opt, plugin, CHAIN, OWNERSHIP_ONLY, tmp_path)
    assert "memref.dealloc" not in text
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    merged = [i for i, ln in enumerate(lines) if ln.startswith("%") and "bufferization.dealloc" in ln]
    assert len(merged) == 1, text
    # One merged dealloc, and it is the last thing before the terminator.
    assert lines[merged[0] + 1].startswith("return"), text


def test_peak_reaches_the_achievable_bound_only_after_refinement(
    mlir_opt, plugin, tmp_path
):
    """The number the pass exists to move, measured by the project's analyzer."""
    unrefined = tmp_path / "own.mlir"
    unrefined.write_text(run_pipeline(mlir_opt, plugin, CHAIN, OWNERSHIP_FULL, tmp_path))
    refined = tmp_path / "refined.mlir"
    report = tmp_path / "peak_report.json"
    refined.write_text(
        run_pipeline(mlir_opt, plugin, CHAIN, REFINED_FULL % report, tmp_path)
    )

    def peaks(path: Path):
        report = BufferizedMemoryAnalyzer.from_bufferized_file(
            path, allow_inferred_sizes=False
        ).analyze()
        return report.get_peak()[1], report.get_peak_last_use()[1]

    own_peak, own_bound = peaks(unrefined)
    ref_peak, ref_bound = peaks(refined)

    # The bound is a property of the program, not of where the frees are, so it
    # must be unchanged: the refinement moves frees, it does not remove buffers.
    assert own_bound == ref_bound
    # Three 4 KiB buffers allocated, two of them live at once at the peak.
    assert ref_bound == 2 * 4096
    assert own_peak == 3 * 4096
    assert ref_peak == ref_bound


def test_buffer_passed_to_a_call_is_refused(mlir_opt, plugin, tmp_path):
    """A callee may store the pointer, so the free stays at the block exit."""
    source = """
func.func private @sink(memref<1024xf32>)
func.func @escape() {
  %a = memref.alloc() : memref<1024xf32>
  func.call @sink(%a) : (memref<1024xf32>) -> ()
  return
}
"""
    text, report = refine(mlir_opt, plugin, source, tmp_path)
    assert "memref.dealloc" not in text
    assert [p["decision"] for p in report["pairs"]] == ["call-may-capture-buffer"]


def test_buffer_selected_into_an_escaping_value_is_refused(mlir_opt, plugin, tmp_path):
    """`scf.if` yielding a memref canonicalizes to `arith.select`.

    That is an operation producing a memref result whose aliasing the pass does
    not model, so both candidates must come back refused rather than freed.
    """
    source = """
func.func @cond(%c: i1) -> memref<1024xf32> {
  %a = memref.alloc() : memref<1024xf32>
  %b = memref.alloc() : memref<1024xf32>
  %r = scf.if %c -> memref<1024xf32> {
    scf.yield %a : memref<1024xf32>
  } else {
    scf.yield %b : memref<1024xf32>
  }
  return %r : memref<1024xf32>
}
"""
    text, report = refine(mlir_opt, plugin, source, tmp_path)
    assert "memref.dealloc" not in text
    decisions = [p["decision"] for p in report["pairs"]]
    assert decisions.count("unmodelled-aliasing-user") == 2, decisions


def test_a_use_through_a_subview_keeps_the_buffer_alive(mlir_opt, plugin, tmp_path):
    """`memref.subview` touches no memory, but what reads the subview does.

    The free has to land after that read. Anchoring it on the subview itself
    would be a use-after-free; treating the subview as a use would pin every
    free back to the block exit, because the ownership pass emits an
    `extract_strided_metadata` for each buffer right before the merged dealloc.
    """
    source = """
func.func @via_subview() {
  %cst = arith.constant 0.000000e+00 : f32
  %a = memref.alloc() : memref<64xf32>
  linalg.fill ins(%cst : f32) outs(%a : memref<64xf32>)
  %sv = memref.subview %a[0] [32] [1] : memref<64xf32> to memref<32xf32, strided<[1]>>
  %b = memref.alloc() : memref<32xf32>
  memref.copy %sv, %b : memref<32xf32, strided<[1]>> to memref<32xf32>
  return
}
"""
    text, report = refine(mlir_opt, plugin, source, tmp_path)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    copy = next(i for i, ln in enumerate(lines) if ln.startswith("memref.copy"))
    subview = next(i for i, ln in enumerate(lines) if "memref.subview" in ln)
    frees = [i for i, ln in enumerate(lines) if ln.startswith("memref.dealloc")]
    assert len(frees) == 2, text
    assert all(index > copy > subview for index in frees), text
    assert [p["decision"] for p in report["pairs"]] == ["sunk", "sunk"]


@pytest.mark.parametrize("source", [CHAIN], ids=["chain"])
def test_every_allocation_is_released_exactly_once(mlir_opt, plugin, source, tmp_path):
    """No double free and no leak: the invariant a misplaced free would break."""
    text, _ = refine(mlir_opt, plugin, source, tmp_path)
    allocated = set(re.findall(r"(%\w+) = memref\.alloc\(\)", text))
    assert allocated

    released: dict[str, int] = {name: 0 for name in allocated}
    for name in re.findall(r"memref\.dealloc (%\w+)", text):
        released[name] = released.get(name, 0) + 1
    for group in re.findall(r"bufferization\.dealloc \(([^:]+):", text):
        for name in re.findall(r"%\w+", group):
            released[name] = released.get(name, 0) + 1
    retained: set[str] = set()
    for group in re.findall(r"retain \(([^:]+):", text):
        retained.update(re.findall(r"%\w+", group))

    for name in allocated:
        assert released[name] <= 1, f"{name} released {released[name]} times\n{text}"
        assert released[name] == 1 or name in retained, f"{name} leaked\n{text}"


def test_the_merged_dealloc_inflates_the_bound_until_it_is_lowered(
    mlir_opt, plugin, tmp_path
):
    """Why the peak comparison runs the deallocation lowering first.

    The merged `bufferization.dealloc` takes every buffer it owns as an
    operand, so for as long as it is in the IR it reads as a use of each of
    them at the block exit, and the measured last-use bound is the sum of every
    allocation. `bufferization-lower-deallocations` turns it into per-buffer
    `memref.dealloc`, which the analyzer reads as a release. Pinned here so the
    measurement point stays a deliberate choice rather than an accident -- the
    same 3x that the refinement is credited with recovering would otherwise
    appear for free, from nothing but where the pipeline was cut.
    """
    early = tmp_path / "early.mlir"
    early.write_text(run_pipeline(mlir_opt, plugin, CHAIN, OWNERSHIP_ONLY, tmp_path))
    late = tmp_path / "late.mlir"
    late.write_text(run_pipeline(mlir_opt, plugin, CHAIN, OWNERSHIP_FULL, tmp_path))

    def bound(path: Path) -> int:
        return BufferizedMemoryAnalyzer.from_bufferized_file(
            path, allow_inferred_sizes=False
        ).analyze().get_peak_last_use()[1]

    assert "bufferization.dealloc" in early.read_text()
    assert "bufferization.dealloc" not in late.read_text()
    assert "memref.dealloc" in late.read_text()
    assert bound(early) == 3 * 4096
    assert bound(late) == 2 * 4096


def test_report_uses_only_documented_decisions(mlir_opt, plugin, tmp_path):
    for source in (CHAIN,):
        _, report = refine(mlir_opt, plugin, source, tmp_path)
        for pair in report["pairs"]:
            assert pair["decision"] in KNOWN_DECISIONS, pair
            assert pair["function"]
            assert isinstance(pair["bytes"], int)


def test_a_missing_plugin_raises_instead_of_running_unrefined(monkeypatch, tmp_path):
    """The refined variant must not degrade into the ownership numbers.

    Without this, a machine where the plugin was never built would report the
    unrefined peak under the refined label -- the same class of silent
    downgrade that made a func-level pass look like it had run when it had
    silently skipped the payload.
    """
    monkeypatch.setenv(DEALLOC_PLUGIN_ENV, str(tmp_path / "absent.so"))
    with pytest.raises(ArmUnavailable, match="plugin not found"):
        dealloc_plugin_path()


def test_the_refined_variant_names_the_pass_it_depends_on():
    """The variant is a pipeline string, so a typo is a runtime error far away."""
    pipeline = PIPELINE_VARIANTS["ownership_refined"]
    assert "edgeslide-liveness-refined-dealloc" in pipeline
    assert "ownership-based-buffer-deallocation" in pipeline
    # It must carry its own bufferization: --pass-pipeline cannot be combined
    # with individual pass flags, so bufferize() passes no --one-shot-bufferize.
    assert "one-shot-bufferize" in pipeline
