"""
PyTorch demo bypass tests — NOT the core MLIR compilation pipeline.

These tests exercise ``edgeslide.edgeslide_pytorch`` (runtime tile-and-concat
wrappers) for quick numerical sanity checks when MLIR tooling is unavailable.

For the proposal-aligned pipeline (DAG → Transform Dialect → JIT), see:
  - ``tests/test_cpu_pipeline.py``
  - ``tests/test_sliced_mlir_numerical_equivalence.py``
  - ``tests/run_cpu_tests.py`` (MLIR JIT optional test)

Do not treat passing tests here as validation of the MLIR end-to-end path.
"""
import copy

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for EdgeSlide module tests")
nn = torch.nn

from edgeslide.edgeslide_pytorch import (
    EdgeSlideTransform,
    MemoryProfiler,
    SlicedLinear,
    SlicingConfig,
    TensorSlicer,
)


@pytest.mark.cpu
def test_slicedconv2d_equivalence_small(cpu_device, deterministic_env):
    """Demo bypass: small tensors under L1 should match baseline (no slicing)."""
    conv = nn.Conv2d(3, 16, 3, padding=1).to(cpu_device).eval()
    x = torch.randn(1, 3, 32, 32, device=cpu_device)

    cfg = SlicingConfig(l1_cache_bytes=2 * 1024 * 1024, enabled=True)
    slicer = TensorSlicer(cfg)
    from edgeslide.edgeslide_pytorch import SlicedConv2d
    wrapped = SlicedConv2d(conv, slicer).to(cpu_device).eval()

    with torch.no_grad():
        y_ref = conv(x)
        y_sliced = wrapped(x)

    assert torch.allclose(y_ref, y_sliced, rtol=1e-4, atol=1e-4)


@pytest.mark.cpu
def test_slicedconv2d_forces_slicing_on_large_input(cpu_device, deterministic_env):
    """Demo bypass: large activations with small L1 should trigger slicing."""
    # k=1 conv: tile-and-concat is numerically exact on any slice dimension.
    conv = nn.Conv2d(16, 32, 1).to(cpu_device).eval()
    x = torch.randn(1, 16, 128, 512, device=cpu_device)

    cfg = SlicingConfig(l1_cache_bytes=256 * 1024, enabled=True, verbose=False)
    from edgeslide.edgeslide_pytorch import SlicedConv2d
    wrapped = SlicedConv2d(conv, TensorSlicer(cfg)).to(cpu_device).eval()

    with torch.no_grad():
        y_ref = conv(x)
        y_sliced = wrapped(x)

    assert wrapped.needs_slicing is True
    assert wrapped.slice_dim == 3
    assert wrapped.tile_size is not None and wrapped.tile_size < 512
    assert torch.allclose(y_ref, y_sliced, rtol=1e-3, atol=1e-3)


@pytest.mark.cpu
@pytest.mark.parametrize("bias", [True, False])
def test_slicedlinear_input_feature_reduction_is_equivalent(
    cpu_device, deterministic_env, bias
):
    """Feature tiles must be reduced, with bias applied exactly once.

    Shapes are sized so the tile cost model actually chooses to tile. The
    model aligns a unit tile up to a cache line and counts an input and an
    output tile, so a tensor smaller than that floor is never worth tiling:
    a 17-feature row (68 B) is returned whole at any budget, which is correct
    but exercises none of the reduction logic under test.
    """
    in_features = 512
    linear = nn.Linear(in_features, 7, bias=bias).to(cpu_device).eval()
    x = torch.randn(1, in_features, device=cpu_device)
    cfg = SlicingConfig(l1_cache_bytes=1024, enabled=True)
    wrapped = SlicedLinear(linear, TensorSlicer(cfg)).to(cpu_device).eval()

    y_ref = linear(x)
    y_sliced = wrapped(x)

    assert wrapped.needs_slicing is True
    assert wrapped.slice_dim == 1
    assert wrapped.tile_size is not None and wrapped.tile_size < in_features
    # More than one tile, so the partial-product reduction really runs and the
    # bias is not silently added once per tile.
    assert in_features // wrapped.tile_size > 1
    assert y_sliced.shape == y_ref.shape
    assert torch.allclose(y_ref, y_sliced, rtol=1e-5, atol=1e-5)


@pytest.mark.cpu
def test_slicedlinear_batch_tiles_are_concatenated(cpu_device, deterministic_env):
    """Batch tiles are concatenated, not reduced.

    Batch must be the larger axis: for a matmul the dimension heuristic ranks
    M and N by size, so a 9x17 input would be tiled along the 17-wide feature
    axis and take the reduction path instead of this one.
    """
    batch = 64
    linear = nn.Linear(17, 7).to(cpu_device).eval()
    x = torch.randn(batch, 17, device=cpu_device)
    cfg = SlicingConfig(l1_cache_bytes=1024, enabled=True)
    wrapped = SlicedLinear(linear, TensorSlicer(cfg)).to(cpu_device).eval()

    y_ref = linear(x)
    y_sliced = wrapped(x)

    assert wrapped.needs_slicing is True
    assert wrapped.slice_dim == 0
    assert wrapped.tile_size is not None and wrapped.tile_size < batch
    assert y_sliced.shape == y_ref.shape
    assert torch.allclose(y_ref, y_sliced, rtol=1e-5, atol=1e-5)


@pytest.mark.cpu
def test_slicedlinear_runs_whole_when_no_tile_can_fit(cpu_device, deterministic_env):
    """A budget below the tiling floor must fall back, not emit a bogus tile.

    When even a unit tile exceeds the budget the calculator raises
    TileBudgetError and the wrapper runs the op whole. Returning a tile of 1
    instead would add loop overhead while still missing cache on every access,
    and the result would be slower than not slicing at all.
    """
    linear = nn.Linear(17, 7).to(cpu_device).eval()
    x = torch.randn(1, 17, device=cpu_device)
    # 64 B cannot hold one cache-line-aligned input and output element pair.
    cfg = SlicingConfig(l1_cache_bytes=64, enabled=True)
    wrapped = SlicedLinear(linear, TensorSlicer(cfg)).to(cpu_device).eval()

    y_sliced = wrapped(x)

    assert wrapped.needs_slicing is False
    # Falling back must stay numerically exact, not merely close.
    assert torch.equal(linear(x), y_sliced)


@pytest.mark.cpu
def test_slicedlinear_feature_tiles_preserve_gradients(cpu_device, deterministic_env):
    baseline = nn.Linear(17, 7).to(cpu_device)
    sliced = copy.deepcopy(baseline)
    x_ref = torch.randn(1, 17, device=cpu_device, requires_grad=True)
    x_sliced = x_ref.detach().clone().requires_grad_(True)
    wrapped = SlicedLinear(
        sliced,
        TensorSlicer(SlicingConfig(l1_cache_bytes=64, enabled=True)),
    )

    baseline(x_ref).sum().backward()
    wrapped(x_sliced).sum().backward()

    assert torch.allclose(x_ref.grad, x_sliced.grad, rtol=1e-5, atol=1e-5)
    assert torch.allclose(baseline.weight.grad, sliced.weight.grad, rtol=1e-5, atol=1e-5)
    assert torch.allclose(baseline.bias.grad, sliced.bias.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.cpu
def test_slicedlinear_preserves_invalid_feature_shape_error(cpu_device):
    linear = nn.Linear(17, 7).to(cpu_device).eval()
    x = torch.randn(1, 16, device=cpu_device)
    wrapped = SlicedLinear(
        linear,
        TensorSlicer(SlicingConfig(l1_cache_bytes=32, enabled=True)),
    )

    with pytest.raises(RuntimeError):
        wrapped(x)


@pytest.mark.cpu
def test_edgeslide_transform_replaces_large_layers(cpu_device):
    """Demo bypass: Conv2d with out_channels >= 64 should be wrapped."""
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
    ).to(cpu_device).eval()

    cfg = SlicingConfig(l1_cache_bytes=64 * 1024, enabled=True)
    transformed = EdgeSlideTransform(cfg).transform_model(model)

    wrapped_count = sum(
        1 for m in transformed.modules() if m.__class__.__name__ == "SlicedConv2d"
    )
    assert wrapped_count >= 1


@pytest.mark.cpu
def test_edgeslide_forward_equivalence_on_medium_input(cpu_device, deterministic_env):
    """Demo bypass: multi-layer model with out_channels >= 64 vs baseline."""
    baseline = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
    ).to(cpu_device).eval()

    cfg = SlicingConfig(l1_cache_bytes=256 * 1024, enabled=True)
    model = copy.deepcopy(baseline)
    sliced_model = EdgeSlideTransform(cfg).transform_model(model)

    # Batch > 1 slices along the batch dimension, which is safe for padded
    # Conv2d in the PyTorch demo path. Spatial H/W slicing needs halo regions
    # and is handled by the MLIR path instead.
    x = torch.randn(4, 3, 256, 256, device=cpu_device)
    with torch.no_grad():
        y_ref = baseline(x)
        y_sliced = sliced_model(x)

    assert y_ref.shape == y_sliced.shape
    assert torch.allclose(y_ref, y_sliced, rtol=1e-3, atol=1e-3)


@pytest.mark.cpu
def test_memory_profiler_runs_on_cpu(cpu_device, tiny_conv_model, small_input):
    prof = MemoryProfiler(tiny_conv_model)
    mem = prof.profile(small_input.to(cpu_device))
    assert len(mem) > 0
    summary = prof.get_memory_summary()
    assert "Output (MB)" in summary
