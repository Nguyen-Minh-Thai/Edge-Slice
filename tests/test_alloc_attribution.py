"""Region attribution and copy counting on bufferized MLIR (proposal §5.2).

These back the Phase 0 alloc-attribution table. Each assertion names a
behaviour the diagnosis depends on: if an allocation inside a tiled loop body
is not distinguished from one at function scope, or a buffer freed after the
loop is not distinguished from one freed inside it, the table cannot tell a
per-iteration allocation from a reused tile — which is the only question it
exists to answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edgeslide.liveness.analyzer import (
    BufferizedMemoryAnalyzer,
    _region_stacks,
    _scan_braces,
)

pytestmark = pytest.mark.cpu


def _report(mlir: str):
    return BufferizedMemoryAnalyzer(mlir).analyze()


def _by_name(report):
    return {alloc.name: alloc for alloc in report.allocations}


# ---------------------------------------------------------------------------
# Brace scanning: an attribute dictionary is not a region.
# ---------------------------------------------------------------------------

def test_an_attribute_dictionary_does_not_open_a_region():
    """`memref.alloc() {alignment = 64 : i64}` closes and opens nothing.

    Counting `{` and `}` separately reads this line as one close followed by
    one open: depth is preserved, so the bug hides in any depth counter, but
    the enclosing region is popped and the alloc line pushed in its place —
    and region *identity* is what scope resolution keys on.
    """
    assert _scan_braces("%0 = memref.alloc() {alignment = 64 : i64} : memref<4xf32>") == (0, 0)


def test_a_region_opener_and_an_else_are_counted_correctly():
    assert _scan_braces("scf.forall (%i) in (8) {") == (0, 1)
    assert _scan_braces("} else {") == (1, 1)
    assert _scan_braces("}") == (1, 0)


def test_braces_inside_a_quoted_string_are_ignored():
    assert _scan_braces('%0 = foo.bar {sym = "a{b}c"} : i32') == (0, 0)


def test_region_stacks_name_the_enclosing_ops():
    lines = list(
        enumerate(
            line.strip()
            for line in [
                "func.func @f() {",
                "  scf.forall (%i) in (4) {",
                "    %t = memref.alloc() : memref<8xf32>",
                "  }",
                "  return",
                "}",
            ]
        )
    )
    stacks = _region_stacks(lines)
    assert stacks[2] == ("func.func", "scf.forall")
    # The op that opens a region is reported outside it.
    assert stacks[1] == ("func.func",)
    assert stacks[0] == ()


# ---------------------------------------------------------------------------
# Attribution: where a buffer lives, and whether a pass could move it.
# ---------------------------------------------------------------------------

NESTED = """
func.func @f(%arg0: memref<4x4xf32>) {
  %big = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xf32>
  scf.forall (%i) in (8) {
    %tile = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
    memref.copy %arg0, %arg0 : memref<4x4xf32> to memref<4x4xf32>
    memref.dealloc %tile : memref<64x64xf32>
  }
  memref.dealloc %big : memref<1024x1024xf32>
  return
}
"""


def test_a_function_scope_buffer_is_not_reported_as_in_loop():
    alloc = _by_name(_report(NESTED))["%big"]
    assert alloc.region_depth == 1
    assert alloc.enclosing_ops == ("func.func",)
    assert not alloc.inside_loop
    assert not alloc.hoistable


def test_a_loop_body_buffer_is_attributed_to_its_loop_and_is_hoistable():
    alloc = _by_name(_report(NESTED))["%tile"]
    assert alloc.region_depth == 2
    assert alloc.enclosing_loops == ("scf.forall",)
    assert alloc.inside_loop
    # Static extents, so its size cannot depend on the induction variable.
    assert alloc.hoistable


def test_a_buffer_freed_inside_the_loop_does_not_accumulate():
    """One tile live at a time is the intended shape, not the regression."""
    assert not _by_name(_report(NESTED))["%tile"].accumulates_across_iterations


ACCUMULATING = """
func.func @f() {
  scf.forall (%i) in (8) {
    %t = memref.alloc() : memref<64x64xf32>
    linalg.fill ins(%i : index) outs(%t : memref<64x64xf32>)
  }
  return
}
"""


def test_a_loop_buffer_with_no_dealloc_inside_accumulates():
    """This is the shape of the tiled arm's regression.

    Each iteration adds a buffer nothing frees before the next one starts, so
    live memory grows with the trip count while ``size_bytes`` still counts it
    once — which makes the reported peak a *lower* bound for exactly the arm
    the tiling was supposed to improve.
    """
    alloc = _by_name(_report(ACCUMULATING))["%t"]
    assert alloc.inside_loop
    assert alloc.accumulates_across_iterations


def test_a_dynamic_extent_in_a_loop_is_not_claimed_hoistable():
    mlir = """
    func.func @f(%n: index) {
      scf.for %i = %n to %n step %n {
        %t = memref.alloc(%n) : memref<?xf32>
      }
      return
    }
    """
    # A bound is required before the analyzer will size a dynamic allocation
    # at all; supplying one isolates the hoistability question from that.
    report = BufferizedMemoryAnalyzer(mlir, shape_bounds={"%t": {0: 128}}).analyze()
    alloc = _by_name(report)["%t"]
    assert alloc.inside_loop
    assert alloc.has_dynamic_extent
    # Conservative: the extent may be invariant, but the type cannot prove it.
    assert not alloc.hoistable


# ---------------------------------------------------------------------------
# memref.copy: the signature of a slice that did not become a subview.
# ---------------------------------------------------------------------------

def test_copies_are_counted_and_located():
    report = _report(NESTED)
    assert len(report.copies) == 1
    assert report.copies[0].inside_loop
    assert report.attribution_totals()["copy_count"] == 1
    assert report.attribution_totals()["copy_in_loop_count"] == 1


def test_attribution_totals_separate_in_loop_bytes_from_the_rest():
    totals = _report(NESTED).attribution_totals()
    assert totals["allocation_count"] == 2
    assert totals["in_loop_count"] == 1
    assert totals["in_loop_bytes"] == 64 * 64 * 4
    assert totals["hoistable_bytes"] == 64 * 64 * 4
    assert totals["max_region_depth"] == 2


def test_the_attribution_table_is_ordered_by_size():
    rows = _report(NESTED).allocation_attribution()
    assert [row["name"] for row in rows] == ["%big", "%tile"]
    assert rows[0]["bytes"] > rows[1]["bytes"]


# ---------------------------------------------------------------------------
# Strict sizing: an inferred extent is an estimate, not a measurement.
# ---------------------------------------------------------------------------

# `scf.for`, deliberately: `_forall_induction_domains` reads only
# `scf.forall`, so this loop's index has no known domain and its
# `affine.min` bound cannot be proven attained. That keeps a realistic
# unprovable case in the suite — `tests/test_affine_attainment.py` covers the
# provable side. Teaching the analyzer `scf.for` domains would make this
# fixture provable and would need a different one here.
DYNAMIC = """
func.func @f(%n: index) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c128 step %c1 {
    %d = affine.min affine_map<(d0) -> (d0 * -1 + 128, 8)>(%i)
    %t = memref.alloc(%d) : memref<?xf32>
    linalg.fill ins(%i : index) outs(%t : memref<?xf32>)
  }
  return
}
"""


def test_the_default_analyzer_infers_and_flags_the_estimate():
    report = BufferizedMemoryAnalyzer(DYNAMIC).analyze()
    estimated = [a for a in report.allocations if a.size_is_upper_bound]
    assert estimated, "expected the dynamic extent to be inferred"
    assert estimated[0].size_bytes > 0


def test_a_strict_analyzer_refuses_to_estimate():
    """A peak built on an inferred extent is an estimate wearing a measurement's type.

    Every reported number in this repo comes from statically-shaped payloads,
    so a run that silently begins inferring has changed what its output means
    without changing anything visible about it.
    """
    from edgeslide.liveness.analyzer import InferredSizeRefused

    with pytest.raises(InferredSizeRefused, match="not known to reach"):
        BufferizedMemoryAnalyzer.strict_sizes(DYNAMIC).analyze()


def test_a_strict_analyzer_accepts_a_caller_supplied_bound():
    """Supplying the extent makes it a fact again, so strict mode allows it."""
    report = BufferizedMemoryAnalyzer.strict_sizes(
        DYNAMIC, shape_bounds={"%t": {0: 8}}
    ).analyze()
    assert report.allocations
    assert not any(a.size_is_upper_bound for a in report.allocations)


def test_a_strict_analyzer_is_unaffected_by_static_payloads():
    """The statically-shaped case — every payload this repo reports on."""
    strict = BufferizedMemoryAnalyzer.strict_sizes(NESTED).analyze()
    default = BufferizedMemoryAnalyzer(NESTED).analyze()
    assert strict.get_peak() == default.get_peak()
    assert strict.attribution_totals() == default.attribution_totals()


def test_attribute_refuses_an_inferred_size_instead_of_reporting_it(tmp_path):
    """The reporting path is strict, not just the analyzer it could have used.

    ``strict_sizes`` existed for a while while ``attribute()`` — the function
    that actually produces the numbers in benchmarks/results/*.json — used the
    permissive constructor. A guard only the tests call guards nothing.

    Bufferization is stubbed out here: the point under test is which analyzer
    ``attribute`` builds, not mlir-opt.
    """
    import benchmarks.alloc_attribution as module

    payload = tmp_path / "payload.mlir"
    payload.write_text(DYNAMIC)

    def _passthrough(src, out, variant, *, mlir_opt=None):
        Path(out).write_text(Path(src).read_text())
        return out

    original = module.bufferize
    module.bufferize = _passthrough
    try:
        result = module.attribute("untiled", "baseline", payload, tmp_path)
    finally:
        module.bufferize = original

    assert not result.ok, (
        "an allocation whose extent had to be inferred was reported as a "
        "measured peak"
    )
    assert "not known to reach" in result.error
    assert result.peak_last_use_bytes == 0


def test_study_records_which_weights_produced_it():
    """A model name does not say whether the weights were published.

    ``run_study`` accepts both a pretrained model and ``--model tiny_conv``, a
    randomly-initialised machinery kernel. Before this the two produced
    artifacts that differed only in a name string, so a reader could not tell
    evidence from a smoke test.
    """
    import torch.nn as nn
    from benchmarks.alloc_attribution import _weights_provenance

    unlabelled = nn.Conv2d(3, 4, 3)
    assert _weights_provenance(unlabelled)["is_pretrained"] is False

    labelled = nn.Conv2d(3, 4, 3)
    labelled._edgeslide_weights_source = "torchvision_pretrained"
    provenance = _weights_provenance(labelled)
    assert provenance["is_pretrained"] is True
    assert provenance["weights_source"] == "torchvision_pretrained"


def test_weights_provenance_records_a_failure_instead_of_omitting_the_field():
    """An artifact with no provenance and one that could not read it must differ."""
    from benchmarks.alloc_attribution import _weights_provenance

    class Hostile:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    provenance = _weights_provenance(Hostile())
    assert provenance["is_pretrained"] is False
    assert provenance["weights_source"] == "undeterminable"
    assert "boom" in provenance["error"]


NESTED_TILED_PAYLOAD = """\
module {
  module {
    module attributes {torch.debug_module_name = "M"} {
      func.func @forward(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
        return %arg0 : tensor<1x4xf32>
      }
    }
  }
  transform.with_named_sequence {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.yield
    }
  }
}
"""


def test_bufferize_hands_mlir_opt_an_unwrapped_payload(tmp_path, monkeypatch):
    """The sweep must unwrap before bufferizing, and nothing tested that.

    Four of the seven bufferization variants are built from func-level passes
    -- `legacy_dealloc`, `ownership`, and the three hoisting passes in the
    `hoisted*` variants. A func-level pass silently skips a function nested
    inside an extra module: it reports success and inserts nothing. Without the
    unwrap those four columns measure a pass that never ran, which is exactly
    how the tiled arms came to look like tiling was the problem.

    A mutation audit found this was the one guard in the reliability set with
    no test behind it: deleting the unwrap left the whole suite green.

    mlir-opt is stubbed -- what is under test is the file `bufferize` hands it.
    """
    import benchmarks.alloc_attribution as module

    payload = tmp_path / "payload.mlir"
    payload.write_text(NESTED_TILED_PAYLOAD)
    seen: dict[str, str] = {}

    def _capture(cmd, *, stage=""):
        # cmd is [tool, source, *passes, "-o", output]
        seen["source"] = Path(cmd[1]).read_text()
        Path(cmd[-1]).write_text(seen["source"])
        return cmd[-1]

    monkeypatch.setattr(module, "run_tool", _capture)
    module.bufferize(payload, tmp_path / "out.mlir", "legacy_dealloc")

    prepared = seen["source"]
    assert prepared.count("module") == 1, (
        "mlir-opt was handed a payload still nested in extra modules; every "
        f"func-level pass in this variant will silently skip it:\n{prepared}"
    )
    assert "transform." not in prepared, (
        "the transform dialect reached bufferization"
    )
    assert "func.func @forward" in prepared, "unwrapping lost the payload function"
