"""Numerical checks for EdgeSlice PyTorch spatial halo slicing."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from edgeslide.edgeslide_pytorch import EdgeSlideTransform, SlicingConfig

pytestmark = pytest.mark.cpu


class _SingleImageConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_edgeslice_batch_one_spatial_halo_matches_padded_conv():
    torch.manual_seed(0)
    baseline = _SingleImageConvNet().eval()
    transformed = copy.deepcopy(baseline).eval()
    transform = EdgeSlideTransform(
        SlicingConfig(l1_cache_bytes=4 * 1024, enabled=True, verbose=False)
    )
    transform.transform_model(transformed)
    x = torch.randn(1, 3, 31, 37)

    with torch.no_grad():
        expected = baseline(x)
        actual = transformed(x)

    assert len(transform.transformed_layers) == 2
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
