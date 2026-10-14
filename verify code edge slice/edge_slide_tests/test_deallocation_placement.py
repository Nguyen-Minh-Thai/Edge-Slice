"""The lowering must actually deallocate, and must not be silently skipped.

Two defects sat behind these checks for the whole evaluation:

1. The transform interpreter leaves its payload nested one module deeper than
   it found it. `--buffer-deallocation` is a func-level pass, and a func-level
   pass silently skips a function nested that deep -- it reports success,
   emits no diagnostic, exits 0, and inserts nothing. Every tiled payload
   therefore leaked every buffer, and the measured "peak" degenerated to the
   sum of every allocation.

2. Tiling bufferizes `tensor.parallel_insert_slice` into a copy of a subview
   onto the identical subview of the same buffer. Lowered, that is
   `memcpy(p, p, 4)` -- undefined behaviour. `--cse` merges the two identical
   subview ops and `--canonicalize` folds the copy away.
"""

from __future__ import annotations

import pytest

from edgeslide.mlir_transform.lowering import (
    DEALLOCATION_PASSES,
    SLICED_PAYLOAD_OPT_PASSES_STAGE1,
    prepare_sliced_payload_for_lowering,
)

pytestmark = pytest.mark.cpu

NESTED = """\
module {
  module attributes {torch.debug_module_name = "M"} {
    func.func @forward(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
      return %arg0 : tensor<1x4xf32>
    }
  }
}
"""


def test_prepare_unwraps_the_interpreters_extra_module() -> None:
    """Otherwise every func-level pass in the pipeline is a silent no-op."""
    prepared = prepare_sliced_payload_for_lowering(
        NESTED, entry_func="forward", emit_c_interface=False
    )
    assert prepared.count("module") == 1, (
        "payload still nested; a func-level pass would skip the function and "
        f"report success:\n{prepared}"
    )
    assert "func.func @forward" in prepared


def test_deallocation_is_followed_by_the_cleanups_that_remove_self_copies() -> None:
    """`--cse` then `--canonicalize`, in that order, after the dealloc pass.

    Order is load-bearing and was measured: canonicalize-then-cse leaves the
    self-copies in place (124 of them on ResNet-50), cse-then-canonicalize
    removes them.
    """
    assert DEALLOCATION_PASSES[0] == "--buffer-deallocation"
    assert DEALLOCATION_PASSES[1:] == ["--cse", "--canonicalize"]

    stage1 = list(SLICED_PAYLOAD_OPT_PASSES_STAGE1)
    for name in DEALLOCATION_PASSES:
        assert name in stage1, f"{name} missing from the shipped pass list"
    assert stage1.index("--buffer-deallocation") < stage1.index("--cse")
    assert stage1.index("--cse") < stage1.index("--canonicalize")


def test_a_payload_that_allocates_without_freeing_is_refused() -> None:
    """The guard checks the emitted IR, not which pass name was requested.

    A pass name in the command line does not prove the pass did anything --
    that is exactly how defect (1) above went unnoticed.
    """
    from pathlib import Path

    from edgeslide.mlir_transform.errors import DeallocationUnavailableError
    from edgeslide.mlir_transform.lowering import (
        SlicedPayloadLoweringConfig,
        _require_deallocations,
    )

    leaking = 'llvm.func @f() { %0 = llvm.call @malloc(%c) : (i64) -> !llvm.ptr }'
    strict = SlicedPayloadLoweringConfig(allow_leaking_deallocation=False)
    with pytest.raises(DeallocationUnavailableError, match="never frees"):
        _require_deallocations(leaking, Path("payload.mlir"), strict)

    # Frees present: accepted.
    freeing = leaking + "\nllvm.call @free(%0) : (!llvm.ptr) -> ()"
    _require_deallocations(freeing, Path("payload.mlir"), strict)

    # Allocates nothing: nothing to free, so not an error.
    _require_deallocations("llvm.func @f() { llvm.return }", Path("p.mlir"), strict)

    # Explicitly opted in: warns rather than raises.
    permissive = SlicedPayloadLoweringConfig(allow_leaking_deallocation=True)
    _require_deallocations(leaking, Path("payload.mlir"), permissive)


def test_unwrap_removes_every_redundant_level_not_just_one() -> None:
    """Levels accumulate one per interpreter run, so unwrapping must iterate.

    A U-Net payload passes through the tiling interpreter and then the
    `tensor.concat` decomposition, arriving three modules deep. A single
    unwrap leaves it two deep -- still enough for a func-level pass to skip
    the function silently, which is why U-Net's tiled arms kept reporting
    `peak == sum of every allocation` after the one-level unwrap had fixed the
    other four models.
    """
    from edgeslide.mlir_transform.lowering import unwrap_redundant_module

    triple = (
        "module {\n"
        "  module {\n"
        '    module attributes {torch.debug_module_name = "M"} {\n'
        "      func.func @f() { return }\n"
        "    }\n"
        "  }\n"
        "}\n"
    )
    unwrapped = unwrap_redundant_module(triple)
    assert unwrapped.count("module") == 1, unwrapped
    assert "func.func @f" in unwrapped


def test_unwrap_leaves_two_sibling_modules_alone() -> None:
    """Two modules under one parent are real structure, not a wrapper.

    Splicing them together would merge two symbol tables. The iteration must
    not turn this into an excuse to keep going.
    """
    from edgeslide.mlir_transform.lowering import unwrap_redundant_module

    siblings = (
        "module {\n"
        "  module { func.func @a() { return } }\n"
        "  module { func.func @b() { return } }\n"
        "}\n"
    )
    assert unwrap_redundant_module(siblings).count("module") == 3


def test_fallback_swap_replaces_the_whole_primary_group() -> None:
    """The fallback swaps all three primary passes, not one name.

    The previous version substituted a single pass name that the command had
    stopped containing, so the retry could never fire while its message still
    named the pass it had failed to substitute.
    """
    from edgeslide.mlir_transform.lowering import (
        DEALLOCATION_PASSES,
        DEALLOCATION_PASSES_FALLBACK,
        SLICED_PAYLOAD_OPT_PASSES_STAGE1,
        _with_deallocation_fallback,
    )

    cmd = ["mlir-opt", "in.mlir", *SLICED_PAYLOAD_OPT_PASSES_STAGE1, "-o", "out.mlir"]
    swapped = _with_deallocation_fallback(cmd)

    assert swapped is not None
    for primary in DEALLOCATION_PASSES:
        assert primary not in swapped, f"{primary} survived the swap"
    for fallback in DEALLOCATION_PASSES_FALLBACK:
        assert fallback in swapped
    assert swapped[:2] == cmd[:2] and swapped[-2:] == cmd[-2:], "swap disturbed the ends"


def test_fallback_swap_reports_no_match_rather_than_an_unchanged_command() -> None:
    """None, not the input, so "nothing to swap" cannot read as "swapped".

    That conflation is exactly what made the old retry a silent no-op.
    """
    from edgeslide.mlir_transform.lowering import _with_deallocation_fallback

    assert _with_deallocation_fallback(["mlir-opt", "in.mlir", "-o", "out.mlir"]) is None


def test_degraded_deallocation_is_a_separate_opt_in_from_leaking() -> None:
    """Freeing too late and never freeing are different failures.

    The fallback pipeline deallocates, so `allow_leaking_deallocation` is the
    wrong gate for it -- it would refuse a correct artifact for the wrong
    reason, or permit one under a promise it does not need.
    """
    from edgeslide.mlir_transform.lowering import SlicedPayloadLoweringConfig

    config = SlicedPayloadLoweringConfig()
    assert config.allow_leaking_deallocation is False
    assert config.allow_degraded_deallocation is False

    degraded = SlicedPayloadLoweringConfig(allow_degraded_deallocation=True)
    assert degraded.allow_degraded_deallocation is True
    assert degraded.allow_leaking_deallocation is False, (
        "opting into worse placement must not also opt into leaking"
    )
