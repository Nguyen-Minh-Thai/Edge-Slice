"""Contract tests: numerical equivalence of the sliced Conv2d wrappers.

The wrappers advertise two guarantees, and these tests pin both across the
full Conv2d configuration space (stride, dilation, groups, padding, dtype,
asymmetric parameters):

1. **Equivalence when slicing is active** — tiled output equals the unsliced
   ``nn.Conv2d`` within floating-point tolerance, for spatial-H, spatial-W and
   batch slicing.
2. **Fallback correctness** — configurations the halo math does not support
   (string padding, non-``zeros`` padding modes) must delegate to the original
   convolution and return its exact result, never silently-wrong numbers.

These are demo-bypass wrappers (see ``edgeslide.edgeslide_pytorch``); passing
here does not validate the MLIR path. The MLIR equivalents live in
``tests/test_cpu_pipeline.py`` and the JIT equivalence tests.
"""

import pytest

torch = pytest.importorskip(
    "torch", reason="PyTorch is required for conv contract tests"
)
nn = torch.nn

from edgeslide.edgeslide_pytorch import (
    SlicedConv2d,
    SlicingConfig,
    TensorSlicer,
)
from edgeslide.sconv_pytorch import SConvConfig, SpatialHaloSConv2d

# Per-dtype tolerances: conv reassociation differs between the tiled and the
# monolithic kernels, so bitwise equality is not the contract — closeness is.
TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-5},
    torch.float64: {"rtol": 1e-9, "atol": 1e-11},
}

# (id, in_ch, out_ch, kernel, stride, padding, dilation, groups, bias)
CONV_CASES = [
    ("k3s1p1", 6, 8, 3, 1, 1, 1, 1, True),
    ("k3s2p1", 6, 8, 3, 2, 1, 1, 1, True),
    ("k5s1p2", 4, 8, 5, 1, 2, 1, 1, True),
    ("k3s1p0", 6, 8, 3, 1, 0, 1, 1, True),
    ("k1s1p0_pointwise", 6, 8, 1, 1, 0, 1, 1, True),
    ("k2s2p0_even_kernel", 6, 8, 2, 2, 0, 1, 1, True),
    ("k3s1p1_no_bias", 6, 8, 3, 1, 1, 1, 1, False),
    ("k3s1p2_dilated", 6, 8, 3, 1, 2, 2, 1, True),
    ("k5s2p3_dilated", 4, 8, 5, 2, 3, 2, 1, True),
    ("k3s1p1_grouped", 6, 9, 3, 1, 1, 1, 3, True),
    ("k3s1p1_depthwise", 6, 12, 3, 1, 1, 1, 6, True),
    ("asym_k35_s21_p12_d12", 6, 8, (3, 5), (2, 1), (1, 2), (1, 2), 1, True),
]

CASE_IDS = [case[0] for case in CONV_CASES]


def make_conv(case, dtype=torch.float32, padding_mode="zeros", padding=None):
    _, in_ch, out_ch, kernel, stride, pad, dilation, groups, bias = case
    torch.manual_seed(0)
    conv = nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=kernel,
        stride=stride,
        padding=pad if padding is None else padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
    )
    return conv.to(dtype).eval()


def make_input(case, dtype=torch.float32, batch=1, height=23, width=19):
    torch.manual_seed(1)
    return torch.randn(batch, case[1], height, width, dtype=dtype)


def force_sliced(conv, slice_dim, tile_size):
    """Wrapper with a pinned plan so the tiled code path always executes."""
    slicer = TensorSlicer(SlicingConfig(l1_cache_bytes=1, enabled=True))
    wrapper = SlicedConv2d(conv, slicer)
    wrapper.slice_dim = slice_dim
    wrapper.tile_size = tile_size
    wrapper.needs_slicing = True
    return wrapper


# ---------------------------------------------------------------------------
# 1. Equivalence with slicing forced on: SlicedConv2d spatial halo paths
# ---------------------------------------------------------------------------


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("slice_dim", [2, 3], ids=["slice_h", "slice_w"])
@pytest.mark.parametrize("case", CONV_CASES, ids=CASE_IDS)
def test_sliced_conv2d_halo_matches_reference(case, slice_dim, dtype):
    conv = make_conv(case, dtype=dtype)
    x = make_input(case, dtype=dtype)

    with torch.no_grad():
        expected = conv(x)
        actual = force_sliced(conv, slice_dim, tile_size=3)(x)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, **TOLERANCES[dtype])


@pytest.mark.cpu
@pytest.mark.parametrize("tile_size", [1, 4, 1000], ids=["unit", "uneven", "oversized"])
def test_sliced_conv2d_halo_tile_size_extremes(tile_size):
    """Unit tiles, tiles that do not divide the output, and tiles larger than
    the output must all reproduce the reference result."""
    case = CONV_CASES[1]  # k3s2p1
    conv = make_conv(case)
    x = make_input(case)

    with torch.no_grad():
        expected = conv(x)
        actual = force_sliced(conv, 2, tile_size=tile_size)(x)

    assert torch.allclose(actual, expected, **TOLERANCES[torch.float32])


@pytest.mark.cpu
@pytest.mark.parametrize("case", CONV_CASES, ids=CASE_IDS)
def test_sliced_conv2d_batch_slicing_matches_reference(case):
    """Batch tiling (dim 0) is exact for convolution: samples are independent."""
    conv = make_conv(case)
    x = make_input(case, batch=4, height=17, width=23)

    with torch.no_grad():
        expected = conv(x)
        actual = force_sliced(conv, 0, tile_size=1)(x)

    assert torch.allclose(actual, expected, **TOLERANCES[torch.float32])


@pytest.mark.cpu
def test_sliced_conv2d_halo_preserves_gradients():
    """The tiled path is built from differentiable ops; input and weight
    gradients must match the reference convolution."""
    case = CONV_CASES[0]
    conv_ref = make_conv(case)
    conv_tiled = make_conv(case)  # same seed → identical weights

    x_ref = make_input(case).requires_grad_(True)
    x_tiled = x_ref.detach().clone().requires_grad_(True)

    conv_ref(x_ref).sum().backward()
    force_sliced(conv_tiled, 2, tile_size=3)(x_tiled).sum().backward()

    tol = TOLERANCES[torch.float32]
    assert torch.allclose(x_tiled.grad, x_ref.grad, **tol)
    assert torch.allclose(conv_tiled.weight.grad, conv_ref.weight.grad, **tol)


@pytest.mark.cpu
def test_sliced_conv2d_low_precision_dtype_if_supported():
    """bfloat16 goes through the same code path; skip only if this CPU/torch
    build cannot run bf16 convolutions at all."""
    case = CONV_CASES[0]
    try:
        conv = make_conv(case, dtype=torch.bfloat16)
        x = make_input(case, dtype=torch.bfloat16)
        with torch.no_grad():
            expected = conv(x)
    except (RuntimeError, NotImplementedError) as exc:
        pytest.skip(f"bfloat16 conv unsupported on this build: {exc}")

    with torch.no_grad():
        actual = force_sliced(conv, 2, tile_size=3)(x)
    assert torch.allclose(
        actual.float(), expected.float(), rtol=2e-2, atol=2e-2
    )


# ---------------------------------------------------------------------------
# 2. Equivalence through the public forward() with a real planning pass
# ---------------------------------------------------------------------------


@pytest.mark.cpu
@pytest.mark.parametrize(
    "case", [CONV_CASES[0], CONV_CASES[1], CONV_CASES[10]], ids=lambda c: c[0]
)
def test_sliced_conv2d_public_forward_plans_and_matches(case):
    """With a budget far below the activation size the planner must engage
    slicing, and the sliced result must match the reference."""
    conv = make_conv(case)
    slicer = TensorSlicer(SlicingConfig(l1_cache_bytes=16 * 1024, enabled=True))
    wrapper = SlicedConv2d(conv, slicer)
    x = make_input(case, height=96, width=96)

    with torch.no_grad():
        expected = conv(x)
        actual = wrapper(x)

    assert torch.allclose(actual, expected, **TOLERANCES[torch.float32])
    # The contract of THIS test is that slicing actually engaged; if the
    # planner rejected the op the equivalence above proved nothing new.
    assert wrapper.needs_slicing, (
        f"planner did not slice (rejected: {wrapper.plan_rejected_reason})"
    )


# ---------------------------------------------------------------------------
# 3. Fallback contract: unsupported configs delegate, never corrupt
# ---------------------------------------------------------------------------


@pytest.mark.cpu
@pytest.mark.parametrize("padding_mode", ["reflect", "replicate", "circular"])
def test_sliced_conv2d_nonzero_padding_mode_falls_back_exactly(padding_mode):
    case = CONV_CASES[0]
    conv = make_conv(case, padding_mode=padding_mode)
    x = make_input(case)

    with torch.no_grad():
        expected = conv(x)
        actual = force_sliced(conv, 2, tile_size=3)(x)

    # Fallback delegates to the very same module: exact equality, no tolerance.
    assert torch.equal(actual, expected)


@pytest.mark.cpu
def test_sliced_conv2d_string_padding_falls_back_exactly():
    case = CONV_CASES[0]
    conv = make_conv(case, padding="same")
    x = make_input(case)

    with torch.no_grad():
        expected = conv(x)
        actual = force_sliced(conv, 2, tile_size=3)(x)

    assert torch.equal(actual, expected)


@pytest.mark.cpu
def test_sliced_conv2d_unbatched_input_falls_back_exactly():
    case = CONV_CASES[0]
    conv = make_conv(case)
    x = make_input(case)[0]  # (C, H, W)

    with torch.no_grad():
        expected = conv(x)
        actual = force_sliced(conv, 2, tile_size=3)(x)

    assert torch.equal(actual, expected)


# ---------------------------------------------------------------------------
# 4. SpatialHaloSConv2d (SConv-style baseline) — same contract
# ---------------------------------------------------------------------------


def force_sconv(conv, tile_output_rows):
    config = SConvConfig(
        l1_cache_bytes=1, tile_output_rows=tile_output_rows, enabled=True
    )
    return SpatialHaloSConv2d(conv, config)


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("case", CONV_CASES, ids=CASE_IDS)
def test_spatial_halo_sconv_matches_reference(case, dtype):
    conv = make_conv(case, dtype=dtype)
    x = make_input(case, dtype=dtype)

    with torch.no_grad():
        expected = conv(x)
        actual = force_sconv(conv, tile_output_rows=3)(x)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, **TOLERANCES[dtype])


@pytest.mark.cpu
@pytest.mark.parametrize("padding_mode", ["reflect", "replicate", "circular"])
def test_spatial_halo_sconv_fallback_is_exact_and_reported(padding_mode):
    case = CONV_CASES[0]
    conv = make_conv(case, padding_mode=padding_mode)
    x = make_input(case)
    wrapper = force_sconv(conv, tile_output_rows=3)

    with torch.no_grad():
        expected = conv(x)
        actual = wrapper(x)

    assert torch.equal(actual, expected)
    assert wrapper.fallback_reason is not None


@pytest.mark.cpu
def test_spatial_halo_sconv_string_padding_fallback_is_exact():
    case = CONV_CASES[0]
    conv = make_conv(case, padding="same")
    x = make_input(case)
    wrapper = force_sconv(conv, tile_output_rows=3)

    with torch.no_grad():
        expected = conv(x)
        actual = wrapper(x)

    assert torch.equal(actual, expected)
    assert wrapper.fallback_reason is not None
