"""The upstream-SConv arm must be impossible to confuse with the stand-in.

`edgeslide.sconv_pytorch` is a *local* spatial-halo approximation that declares
`IS_REFERENCE_IMPLEMENTATION = True`; the arm added here runs the published
`transform.structured.sconv`. Numbers from the two are not interchangeable, and
attributing the stand-in's to Ferrari et al. would be a misattribution. These
tests pin the three properties that prevent it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="PyTorch is required")
pytest.importorskip("numpy")

from benchmarks.run_sixcam_sconv_comparison import (  # noqa: E402
    DEFAULT_METHODS,
    SUPPORTED_METHODS,
    UnsupportedConfiguration,
    _run_sconv_upstream_worker,
    find_sconv_opt,
    sconv_upstream_provenance,
)

pytestmark = pytest.mark.cpu


def test_the_upstream_arm_is_registered_and_is_a_default():
    """It self-skips cleanly when unbuilt, which beats a stand-in substituting.

    A missing build lands in the report's `skipped` list with a reason — that
    absence is visible. A local stand-in quietly occupying the SConv slot is
    not.
    """
    assert "sconv_upstream" in SUPPORTED_METHODS
    assert "sconv_upstream" in DEFAULT_METHODS


def test_no_reference_implementation_is_a_default_arm():
    """The property that matters: a default run compares only real things.

    `edgeslice` and `sconv` both declare `IS_REFERENCE_IMPLEMENTATION = True`
    — the first is the PyTorch demo bypass whose own module header says it
    "must not be cited as evidence that EdgeSlice works end-to-end", the second
    a local spatial-halo stand-in that is not Ferrari et al.'s transform. With
    both in the defaults, a plain run compared one reference implementation
    against another and reported it as a result.
    """
    from edgeslide import edgeslide_pytorch, sconv_pytorch

    assert edgeslide_pytorch.IS_REFERENCE_IMPLEMENTATION is True
    assert sconv_pytorch.IS_REFERENCE_IMPLEMENTATION is True
    assert "edgeslice" not in DEFAULT_METHODS
    assert "sconv" not in DEFAULT_METHODS
    # Still reachable by name, for reference-vs-reference checks.
    assert "edgeslice" in SUPPORTED_METHODS
    assert "sconv" in SUPPORTED_METHODS


def test_a_missing_build_is_unsupported_not_a_crash(monkeypatch, tmp_path):
    """One unbuilt arm lands in the report's `skipped` list, not a stack trace."""
    import benchmarks.run_sixcam_sconv_comparison as runner

    monkeypatch.setattr(runner, "find_sconv_opt", lambda: None)
    with pytest.raises(UnsupportedConfiguration, match="sconv-opt not found"):
        _run_sconv_upstream_worker(
            _args(), object(), [object()], {}, [1, 3, 8, 8], "smoke"
        )


def test_no_inputs_is_unsupported_not_a_crash(monkeypatch):
    import benchmarks.run_sixcam_sconv_comparison as runner

    monkeypatch.setattr(runner, "find_sconv_opt", lambda: __file__)
    with pytest.raises(UnsupportedConfiguration, match="at least one input"):
        _run_sconv_upstream_worker(_args(), object(), [], {}, [1, 3, 8, 8], "smoke")


def _args():
    import types

    return types.SimpleNamespace(model="tiny_conv", l1_cache_kb=32)


# ---------------------------------------------------------------------------
# Against a real build
# ---------------------------------------------------------------------------

def _require_sconv_opt():
    path = find_sconv_opt()
    if path is None:
        pytest.skip(
            "upstream sconv-opt not built "
            "(third_party/SConvTransform/build_llvm19.sh)"
        )
    return path


def test_provenance_cannot_be_mistaken_for_the_local_stand_in():
    path = _require_sconv_opt()
    provenance = sconv_upstream_provenance(path)
    assert provenance["implementation"] == "sconv_upstream_sconvtransform"
    # The stand-in's label, which must never appear on this arm.
    assert provenance["implementation"] != "local_exact_spatial_halo_sconv"
    # The microkernel caveat travels with the number, not just in a doc.
    assert provenance["upstream_kernel_exact"] is False
    assert "OpenBLAS" in provenance["microkernel_note"]


@pytest.mark.integration
@pytest.mark.slow
def test_the_arm_actually_transforms_a_conv():
    """End to end: the published transform applies to a torch-mlir payload."""
    _require_sconv_opt()
    pytest.importorskip("torch_mlir", reason="torch-mlir is required")
    import torch

    class _Conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Conv2d(8, 16, 3, bias=False)

        def forward(self, x):
            return self.c(x)

    result = _run_sconv_upstream_worker(
        _args(),
        _Conv().eval(),
        [torch.randn(1, 8, 32, 32)],
        {},
        [1, 8, 32, 32],
        "smoke",
    )
    meta = result["method_meta"]
    assert meta["implementation"] == "sconv_upstream_sconvtransform"
    # A transform that applied produces a loop nest around GEMM generics.
    assert meta["loop_nest_depth"] > 0
    assert meta["gemm_generics"] > 0
    # Compile-only: throughput belongs to benchmarks/run_sconv_upstream.py,
    # which owns the microkernel shim. Two harnesses, one number each.
    assert result["runtime_measured"] is False
    assert result["timing"] is None


@pytest.mark.integration
@pytest.mark.slow
def test_a_payload_with_no_matching_conv_is_reported_unsupported():
    """A transform that applies to nothing must not pass as a success.

    `structured.sconv` exits 0 on a payload it matches nothing in, and leaves
    zero convs behind — which is indistinguishable, on the output alone, from
    having transformed them all. The arm therefore counts the convs in the
    *input* too. Checking only the output was the first version of this guard,
    and this test is what caught it: a pure-ReLU payload was reported as a
    successful SConv arm.
    """
    _require_sconv_opt()
    pytest.importorskip("torch_mlir", reason="torch-mlir is required")
    import torch

    # Pure elementwise: no linalg.conv_2d_nchw_fchw for sconv to match.
    model = torch.nn.Sequential(torch.nn.ReLU()).eval()
    with pytest.raises(UnsupportedConfiguration, match="no linalg.conv"):
        _run_sconv_upstream_worker(
            _args(), model, [torch.randn(1, 8, 8, 8)], {}, [1, 8, 8, 8], "smoke"
        )
