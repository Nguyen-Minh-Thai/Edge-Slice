#!/usr/bin/env python3
"""Populate ``data/real_images/`` with a deterministic set of real photos.

This is the script referenced by the ``--real-images`` workflow
(``benchmarks/run_evaluation.py``, ``benchmarks/image_inputs.py``). It copies
the first N images (sorted, recursive) from a source directory — by default
the nuImages mini samples — into ``data/real_images/`` with stable names
(``00.jpg``, ``01.jpg``, ...), so every run sees the same batch. Previously
prepared files are removed first: the loader globs the whole directory, so
leftovers from an earlier run (different --count or --source) would silently
mix into the batch.

Usage:
    python scripts/prepare_real_images.py
    python scripts/prepare_real_images.py --source ~/Pictures --count 20
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.image_files import IMAGE_EXTENSIONS, list_image_paths  # noqa: E402

DEFAULT_SOURCE = ROOT / "data" / "nuimages-v1.0-mini" / "samples"
DEFAULT_OUTPUT = ROOT / "data" / "real_images"

# Names this script itself creates (``00.jpg``, ``01.png``, ...); only these
# are cleared on re-run so user files in the directory are never deleted.
_PREPARED_NAME_RE = re.compile(r"^\d{2,}\.[a-z0-9]+$")


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive count, got {value}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy a deterministic batch of real photos into data/real_images/"
    )
    parser.add_argument(
        "--source",
        type=lambda text: Path(text).expanduser(),
        default=DEFAULT_SOURCE,
        help=f"Directory to copy images from (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=lambda text: Path(text).expanduser(),
        default=DEFAULT_OUTPUT,
        help=f"Destination directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--count",
        type=_positive_int,
        default=10,
        help="Number of images to copy (default: 10)",
    )
    args = parser.parse_args()

    if not args.source.is_dir():
        print(
            f"Source directory not found: {args.source}\n"
            "Pass --source <dir with .jpg/.png files>, or download the "
            "nuImages v1.0-mini samples into data/nuimages-v1.0-mini/.",
            file=sys.stderr,
        )
        return 2

    output_dir = args.output.resolve()
    # Exclude anything under the output directory: with output ⊆ source the
    # scan would otherwise pick up previously prepared copies (or copy a file
    # onto itself).
    found = [
        p
        for p in list_image_paths(args.source)
        if not p.resolve().is_relative_to(output_dir)
    ]
    if not found:
        print(f"No images under {args.source}", file=sys.stderr)
        return 2
    if len(found) < args.count:
        print(
            f"Only {len(found)} image(s) available; copying all of them "
            f"(requested {args.count})."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    stale = [
        p
        for p in args.output.iterdir()
        if p.is_file() and _PREPARED_NAME_RE.match(p.name)
    ]
    for path in stale:
        path.unlink()
    if stale:
        print(f"Removed {len(stale)} previously prepared image(s) from {args.output}")
    leftovers = [
        p.name for p in args.output.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if leftovers:
        print(
            f"Warning: {len(leftovers)} unmanaged image(s) remain in {args.output} "
            f"and will also be picked up by the loader: {sorted(leftovers)}",
            file=sys.stderr,
        )

    copied = []
    for index, path in enumerate(found[: args.count]):
        target = args.output / f"{index:02d}{path.suffix.lower()}"
        shutil.copyfile(path, target)
        copied.append(target)

    print(f"Copied {len(copied)} image(s) to {args.output}:")
    for target in copied:
        print(f"  {target.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
