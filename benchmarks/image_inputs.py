"""Load real RGB images as NCHW batches for EdgeSlice tests / evaluation.

Default dummy path in this repo is ``torch.randn(...)``. Use this module when
you want a fixed batch of real photos (e.g. batch=10) for tracing, peak-memory
analysis, or forward-pass smoke tests.

Directory layouts (searched in order)::

    data/real_images/
        00.jpg
        01.jpg
        ...

    data/nuimages-v1.0-mini/samples/
        CAM_FRONT/*.jpg
        CAM_BACK/*.jpg
        ...

Prepare samples with::

    python scripts/prepare_real_images.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# Re-exported so existing importers keep working; the definitions live in the
# stdlib-only module shared with scripts/prepare_real_images.py.
from benchmarks.image_files import IMAGE_EXTENSIONS, list_image_paths  # noqa: F401

# Project root: benchmarks/ -> repo root
ROOT = Path(__file__).resolve().parents[1]

_PREPARED_IMAGE_DIR = ROOT / "data" / "real_images"
_NUIMAGES_SAMPLE_DIR = ROOT / "data" / "nuimages-v1.0-mini" / "samples"
DEFAULT_IMAGE_DIR = (
    _PREPARED_IMAGE_DIR if _PREPARED_IMAGE_DIR.is_dir() else _NUIMAGES_SAMPLE_DIR
)

# ImageNet mean/std (RGB, 0–1 scale) — ResNet pretrained convention.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_pil():
    try:
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to load real images. Install with: pip install pillow"
        ) from exc
    from PIL import Image

    return Image


def load_image_tensor(
    path: Union[str, Path],
    height: int,
    width: int,
    *,
    normalize: str = "none",
) -> torch.Tensor:
    """
    Load one image as float32 tensor ``[3, H, W]`` in RGB order.

    Args:
        path: Image file path.
        height, width: Resize target (bilinear via PIL).
        normalize:
            - ``"none"``: pixels in [0, 1]
            - ``"imagenet"``: (x - mean) / std for torchvision ResNet
    """
    Image = _require_pil()
    with Image.open(path) as img:
        img = img.convert("RGB").resize((width, height), Image.BILINEAR)
        # HWC uint8 -> CHW float
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32))
    tensor = arr.permute(2, 0, 1).contiguous() / 255.0

    if normalize == "imagenet":
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
    elif normalize != "none":
        raise ValueError(f"Unknown normalize mode: {normalize!r}")

    return tensor


def load_image_batch(
    image_dir: Union[str, Path],
    height: int,
    width: int,
    batch_size: int = 10,
    *,
    normalize: str = "none",
    paths: Optional[Sequence[Union[str, Path]]] = None,
) -> torch.Tensor:
    """
    Build a batch tensor ``[N, 3, H, W]`` from real images.

    If fewer than ``batch_size`` images exist, images are cycled (repeated)
    to fill the batch. If more exist, the first ``batch_size`` (sorted) are used.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if paths is None:
        found = list_image_paths(image_dir)
    else:
        found = [Path(p) for p in paths]

    if not found:
        raise FileNotFoundError(
            f"No images found in {image_dir!s}. "
            "Run: python scripts/prepare_real_images.py"
        )

    selected: List[Path] = []
    for i in range(batch_size):
        selected.append(found[i % len(found)])

    tiles = [
        load_image_tensor(p, height, width, normalize=normalize) for p in selected
    ]
    batch = torch.stack(tiles, dim=0).to(dtype=torch.float32, device="cpu")
    assert batch.shape == (batch_size, 3, height, width)
    return batch


def make_real_or_dummy_input(
    height: int,
    width: int,
    batch_size: int = 10,
    *,
    image_dir: Optional[Union[str, Path]] = None,
    use_real_images: bool = True,
    normalize: str = "none",
    allow_dummy_fallback: bool = False,
    seed: int = 0,
) -> Tuple[torch.Tensor, dict]:
    """
    Prefer real images; optionally fall back to ``torch.randn``.

    Returns:
        (tensor, meta) where meta describes the source of the batch.
    """
    directory = Path(
        image_dir
        or os.environ.get("EDGESLIDE_IMAGE_DIR", str(DEFAULT_IMAGE_DIR))
    )
    meta = {
        "source": "dummy_randn",
        "image_dir": str(directory),
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "normalize": normalize,
        "paths": [],
    }

    if use_real_images:
        try:
            paths = list_image_paths(directory)
            batch = load_image_batch(
                directory,
                height,
                width,
                batch_size=batch_size,
                normalize=normalize,
                paths=paths,
            )
            # Paths actually used (cycled)
            used = [str(paths[i % len(paths)]) for i in range(batch_size)]
            meta.update({"source": "real_images", "paths": used, "num_files": len(paths)})
            return batch, meta
        except FileNotFoundError:
            if not allow_dummy_fallback:
                raise

    torch.manual_seed(seed)
    dummy = torch.randn(batch_size, 3, height, width, device="cpu")
    meta["source"] = "dummy_randn"
    return dummy, meta


def normalize_mode_for_model(model_name: str) -> str:
    """Default pixel normalization per workload."""
    name = (model_name or "").lower()
    if name in ("resnet50", "mobilenet_v2", "efficientnet_b0"):
        # All three load real ImageNet-pretrained weights (workloads.py) and
        # were trained with ImageNet mean/std normalization.
        return "imagenet"
    return "none"
