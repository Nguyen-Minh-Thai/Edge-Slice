"""Numerical checks for the local SConv PyTorch baseline."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from edgeslide.sconv_pytorch import SConvConfig, SConvTransform, SpatialHaloSConv2d

pytestmark = pytest.mark.cpu


class _PaddedConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_spatial_halo_sconv_matches_padded_conv2d():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 5, kernel_size=3, padding=1).eval()
    wrapped = SpatialHaloSConv2d(
        copy.deepcopy(conv),
        SConvConfig(tile_output_rows=7, min_out_channels=1),
    ).eval()
    x = torch.randn(2, 3, 31, 37)

    with torch.no_grad():
        expected = conv(x)
        actual = wrapped(x)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_sconv_transform_preserves_model_output():
    torch.manual_seed(0)
    baseline = _PaddedConvNet().eval()
    transformed = copy.deepcopy(baseline).eval()
    transform = SConvTransform(
        SConvConfig(tile_output_rows=5, min_out_channels=1)
    )
    transform.transform_model(transformed)
    x = torch.randn(4, 3, 33, 35)

    with torch.no_grad():
        expected = baseline(x)
        actual = transformed(x)

    assert len(transform.transformed_layers) == 2
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
