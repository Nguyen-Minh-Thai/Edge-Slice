"""An extent the loop reaches is a measurement; one it merely never exceeds is not.

`affine.min` feeds most dynamically-sized allocations in tiled IR. The sound
upper bound of a min-of-affines is the min of each branch's maximum, and those
maxima can sit at different indices — so the bound can exceed anything the loop
actually allocates. Sizing a peak with it is then an estimate.

Over a known iteration domain the true maximum is computable exactly, because a
min of affine functions is concave: its maximum over a finite integer range is
at an endpoint or beside a pairwise crossing. When that value equals the bound,
the bound is attained and the size is a fact.
"""

import pytest

from edgeslide.liveness.analyzer import (
    BufferizedMemoryAnalyzer,
    InferredSizeRefused,
    _attained_min_maximum,
    _forall_induction_domains,
)


def _lines(mlir: str):
    return [(i, line) for i, line in enumerate(mlir.strip().splitlines())]


# --------------------------------------------------------------------------
# the arithmetic
# --------------------------------------------------------------------------


def test_ragged_tile_bound_is_attained() -> None:
    """min(-5*d0 + 8, 5) over d0 in [0,2) takes 5 then 3, so 5 is reached.

    This is the shape MLIR emits for a 2D tile with a ragged remainder, and it
    is the one that blocked five cells of the Phase 1 matrix.
    """
    assert _attained_min_maximum(["d0 * -5 + 8", "5"], 2) == 5


def test_single_iteration_domain() -> None:
    """One trip: the value at d0 = 0 is both the max and attained."""
    assert _attained_min_maximum(["d0 * -5 + 8", "5"], 1) == 5


def test_bound_can_exceed_everything_the_loop_reaches() -> None:
    """min(d0, 10 - d0) peaks at 5, while each branch alone reaches 10.

    The sound bound is 10 and nothing ever allocates that much. This is the
    case the attainment check exists to keep out of a reported peak.
    """
    assert _attained_min_maximum(["d0", "-d0 + 10"], 11) == 5


def test_unparsable_expression_declines_rather_than_guessing() -> None:
    """floordiv is not linear, so no claim is made either way."""
    assert _attained_min_maximum(["d0 floordiv 4", "5"], 8) is None
    assert _attained_min_maximum(["d0 * 2 + s0", "5"], 8) is None


def test_empty_domain_declines() -> None:
    assert _attained_min_maximum(["5"], 0) is None


# --------------------------------------------------------------------------
# reading the domain out of the IR
# --------------------------------------------------------------------------


def test_forall_domains_are_read_per_induction_variable() -> None:
    domains = _forall_induction_domains(
        _lines("scf.forall (%arg1, %arg2) in (256, 2) {")
    )
    assert [e for _, e in domains["%arg1"]] == [256]
    assert [e for _, e in domains["%arg2"]] == [2]


def test_a_non_literal_forall_yields_no_domain() -> None:
    """No domain means no proof, which leaves the size an upper bound."""
    domains = _forall_induction_domains(_lines("scf.forall (%i) in (%n) {"))
    assert "%i" not in domains


# --------------------------------------------------------------------------
# end to end through the analyzer
# --------------------------------------------------------------------------

ATTAINED = """
#map = affine_map<(d0) -> (d0 * -5 + 8, 5)>
module {
  func.func @f() {
    scf.forall (%arg1, %arg2) in (256, 2) {
      %0 = affine.min #map(%arg2)
      %alloc = memref.alloc(%0) {alignment = 64 : i64} : memref<1x128x?x8xf32>
      memref.dealloc %alloc : memref<1x128x?x8xf32>
    }
    return
  }
}
"""

NOT_ATTAINED = """
#map = affine_map<(d0) -> (d0, -d0 + 10)>
module {
  func.func @f() {
    scf.forall (%arg1) in (11) {
      %0 = affine.min #map(%arg1)
      %alloc = memref.alloc(%0) {alignment = 64 : i64} : memref<1x128x?x8xf32>
      memref.dealloc %alloc : memref<1x128x?x8xf32>
    }
    return
  }
}
"""


def test_an_attained_extent_is_accepted_by_strict_analysis() -> None:
    report = BufferizedMemoryAnalyzer.strict_sizes(ATTAINED).analyze()
    allocation = report.allocations[0]
    assert not allocation.size_is_upper_bound
    # 1 x 128 x 5 x 8 x f32
    assert allocation.size_bytes == 128 * 5 * 8 * 4


def test_an_unattained_bound_is_still_refused() -> None:
    """The proof must not become a rubber stamp for every dynamic extent."""
    with pytest.raises(InferredSizeRefused, match="not known to reach"):
        BufferizedMemoryAnalyzer.strict_sizes(NOT_ATTAINED).analyze()

    permissive = BufferizedMemoryAnalyzer(NOT_ATTAINED).analyze()
    assert permissive.allocations[0].size_is_upper_bound
