"""Tests for real-image batch loading (batch=10)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for real-image batch tests")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.image_inputs import (  # noqa: E402
    DEFAULT_IMAGE_DIR,
    list_image_paths,
    load_image_batch,
    make_real_or_dummy_input,
)
from benchmarks.workloads import make_example_input  # noqa: E402

pytestmark = pytest.mark.cpu


def _require_images():
    if not list_image_paths(DEFAULT_IMAGE_DIR):
        pytest.skip(
            f"No real images in {DEFAULT_IMAGE_DIR}. "
            "Provide data/real_images or the nuImages mini samples."
        )


def test_list_image_paths_recurses_into_camera_directories(tmp_path):
    camera_dir = tmp_path / "samples" / "CAM_FRONT"
    camera_dir.mkdir(parents=True)
    expected = camera_dir / "frame.jpg"
    expected.write_bytes(b"not-decoded-by-this-test")

    assert list_image_paths(tmp_path / "samples") == [expected]


def test_load_image_batch_shape_10():
    _require_images()
    batch = load_image_batch(DEFAULT_IMAGE_DIR, 224, 224, batch_size=10, normalize="none")
    assert batch.shape == (10, 3, 224, 224)
    assert batch.dtype == torch.float32
    assert batch.device.type == "cpu"
    # Unnormalized pixels in [0, 1]
    assert float(batch.min()) >= 0.0
    assert float(batch.max()) <= 1.0


def test_load_image_batch_imagenet_normalize():
    _require_images()
    batch = load_image_batch(
        DEFAULT_IMAGE_DIR, 224, 224, batch_size=10, normalize="imagenet"
    )
    assert batch.shape == (10, 3, 224, 224)
    # After ImageNet norm, values typically leave [0, 1]
    assert float(batch.std()) > 0.01


def test_make_example_input_real_images_meta():
    _require_images()
    tensor, meta = make_example_input(
        "resnet50",
        224,
        224,
        batch_size=10,
        use_real_images=True,
        return_meta=True,
    )
    assert tensor.shape == (10, 3, 224, 224)
    assert meta["source"] == "real_images"
    assert len(meta["paths"]) == 10


def test_make_real_or_dummy_fallback(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    tensor, meta = make_real_or_dummy_input(
        64,
        64,
        batch_size=10,
        image_dir=empty,
        use_real_images=True,
        allow_dummy_fallback=True,
    )
    assert tensor.shape == (10, 3, 64, 64)
    assert meta["source"] == "dummy_randn"


def test_fixture_real_image_batch10(real_image_batch10):
    assert real_image_batch10.shape == (10, 3, 224, 224)
