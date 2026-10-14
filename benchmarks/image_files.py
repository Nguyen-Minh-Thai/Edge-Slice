"""Shared image-file discovery for the real-image workflow.

Single source of truth for which files count as images and how they are
enumerated, used by both the loader (``benchmarks/image_inputs.py``) and the
preparer (``scripts/prepare_real_images.py``). Deliberately stdlib-only so the
preparer does not drag in torch/numpy just to copy files.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_image_paths(image_dir: Union[str, Path]) -> List[Path]:
    """Return sorted image paths under *image_dir*, including subdirectories."""
    directory = Path(image_dir)
    if not directory.is_dir():
        return []
    return [
        p
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
