"""Known-answer cases for the bufferized memory analyzer.

Every peak-memory number in the evaluation comes out of this analyzer, and the
analyzer reads MLIR as text. So its parsing has to be pinned against cases
whose answer is computable by hand — otherwise a text-level regression shows up
as a plausible-looking number rather than as a failure.

Each case below states the expected bytes in the docstring and derives nothing
from the code under test.
"""

import pytest

from edgeslide.liveness.analyzer import BufferizedMemoryAnalyzer

KB = 1024


def _analyze(mlir: str):
    return BufferizedMemoryAnalyzer(mlir).analyze()


def test_sequential_buffers_do_not_stack() -> None:
    """Two 4 KB buffers, the first dead before the second is born: peak 4 KB.

    1024xf32 = 4096 bytes. Under last-use lifetimes %a ends at the copy, so the
    two never overlap and the peak is one buffer, not their sum.
    """
    mlir = """
    module {
      func.func @f() {
        %a = memref.alloc() : memref<1024xf32>
        %b = memref.alloc() : memref<1024xf32>
        memref.copy %a, %b : memref<1024xf32> to memref<1024xf32>
        memref.dealloc %a : memref<1024xf32>
        memref.dealloc %b : memref<1024xf32>
        return
      }
    }
    """
    report = _analyze(mlir)
    assert report.get_peak()[1] == 8 * KB, "dealloc-based peak: both alive"
    assert report.get_peak_last_use()[1] == 8 * KB, (
        "%a's last use is the copy, which is after %b is allocated, so they do "
        "overlap; a peak of 4 KB here would mean the copy was missed"
    )


def test_overlapping_buffers_sum() -> None:
    """4 KB + 8 KB alive at the same time: peak 12 KB."""
    mlir = """
    module {
      func.func @f() {
        %a = memref.alloc() : memref<1024xf32>
        %b = memref.alloc() : memref<2048xf32>
        memref.copy %a, %b : memref<1024xf32> to memref<2048xf32>
        memref.dealloc %b : memref<2048xf32>
        memref.dealloc %a : memref<1024xf32>
        return
      }
    }
    """
    assert _analyze(mlir).get_peak()[1] == 12 * KB


def test_subview_keeps_its_root_alive() -> None:
    """A subview read after the root's other uses must extend the root's life.

    %big is 16 KB and %small is a *view* of it, not a second buffer. If the
    analyzer counted the view as an allocation the peak would be 20 KB; if it
    ignored the view's read it would end %big's life early. Correct answer:
    16 KB + the 4 KB sink, and %big still live at the sink's allocation.
    """
    mlir = """
    module {
      func.func @f() {
        %big = memref.alloc() : memref<4096xf32>
        %small = memref.subview %big[0] [1024] [1] :
          memref<4096xf32> to memref<1024xf32>
        %sink = memref.alloc() : memref<1024xf32>
        memref.copy %small, %sink : memref<1024xf32> to memref<1024xf32>
        memref.dealloc %sink : memref<1024xf32>
        memref.dealloc %big : memref<4096xf32>
        return
      }
    }
    """
    report = _analyze(mlir)
    assert len(report.allocations) == 2, (
        "memref.subview is a view; counting it as an allocation would inflate "
        f"every tiled arm's peak (got {[a.name for a in report.allocations]})"
    )
    assert report.get_peak_last_use()[1] == 20 * KB


def test_attribute_dict_braces_do_not_shift_region_depth() -> None:
    """An attribute dict on an op line must not read as a region opening.

    `{alignment = 64 : i64}` contains braces. If they were counted as a region,
    every allocation after it would be reported one level deeper — and deep
    enough, inside a loop that does not exist, which is what the hoistable and
    in-loop attribution are built on.

    Stated differentially: adding the attribute must change nothing. That pins
    the brace handling without hard-coding how the analyzer counts the module
    and func regions themselves.
    """
    template = """
    module {{
      func.func @f() {{
        %a = memref.alloc(){attributes} : memref<1024xf32>
        memref.dealloc %a : memref<1024xf32>
        return
      }}
    }}
    """
    plain = _analyze(template.format(attributes=""))
    with_attrs = _analyze(template.format(attributes=" {alignment = 64 : i64}"))

    assert len(plain.allocations) == len(with_attrs.allocations) == 1
    assert not with_attrs.allocations[0].inside_loop, (
        "attribute braces read as a loop region"
    )
    assert with_attrs.allocations[0].region_depth == plain.allocations[0].region_depth, (
        f"attribute dict shifted region depth: "
        f"{plain.allocations[0].region_depth} -> "
        f"{with_attrs.allocations[0].region_depth}"
    )
    assert with_attrs.get_peak()[1] == plain.get_peak()[1] == 4 * KB


def test_allocation_in_loop_counts_once_when_freed_in_loop() -> None:
    """Allocated and freed inside the body: one live copy, not one per trip."""
    mlir = """
    module {
      func.func @f(%lb: index, %ub: index, %step: index) {
        scf.for %i = %lb to %ub step %step {
          %t = memref.alloc() : memref<1024xf32>
          memref.dealloc %t : memref<1024xf32>
        }
        return
      }
    }
    """
    report = _analyze(mlir)
    assert len(report.allocations) == 1
    assert report.allocations[0].inside_loop
    assert report.get_peak()[1] == 4 * KB, (
        "a loop-local buffer that is freed in the same body occupies one "
        "buffer's worth of memory regardless of trip count"
    )


def test_constant_globals_are_counted_but_never_in_the_peak() -> None:
    """Weights are read-only data, so they sit beside the peak, not inside it.

    One 4 KB global and one 4 KB heap buffer: peak 4 KB, globals 4 KB,
    peak_plus_constants 8 KB. Measured RSS of the same binary would include
    both, which is why the split has to be reportable.
    """
    mlir = """
    module {
      memref.global "private" constant @w : memref<1024xf32>
      func.func @f() {
        %a = memref.alloc() : memref<1024xf32>
        %w = memref.get_global @w : memref<1024xf32>
        memref.copy %w, %a : memref<1024xf32> to memref<1024xf32>
        memref.dealloc %a : memref<1024xf32>
        return
      }
    }
    """
    totals = _analyze(mlir).attribution_totals()
    assert totals["allocation_count"] == 1, "get_global is not an allocation"
    assert totals["peak_last_use_bytes"] == 4 * KB
    assert totals["constant_global_count"] == 1
    assert totals["constant_global_bytes"] == 4 * KB
    assert totals["peak_plus_constants_bytes"] == 8 * KB


def test_unknown_element_type_refuses_rather_than_assuming_four_bytes() -> None:
    """An unrecognised dtype must fail, not silently size at 4 bytes/element."""
    mlir = """
    module {
      func.func @f() {
        %a = memref.alloc() : memref<1024xf99>
        memref.dealloc %a : memref<1024xf99>
        return
      }
    }
    """
    with pytest.raises(ValueError, match="Unknown element type"):
        _analyze(mlir)
