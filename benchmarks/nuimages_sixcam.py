"""Strict synchronized six-camera nuImages batch loading.

The benchmark in this repository must not fabricate six-camera batches from
single-camera nuImages mini samples.  This module therefore only returns a
batch when metadata proves that each selected sample has all six camera files.

Output tensor convention for model benchmarks is::

    [batch_count, 6, 3, height, width]

Use :func:`flatten_sixcam_tensor` to feed single-image CNNs as
``[batch_count * 6, 3, height, width]`` while preserving the selected manifest.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NUIMAGES_ROOT = ROOT / "data" / "nuimages-v1.0-mini"

# Clockwise order around the vehicle.  Keep this stable so manifests are
# reproducible and downstream models always see the same camera axis order.
CAMERA_CHANNELS: Tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)


@dataclass(frozen=True)
class SixCameraSample:
    """One synchronized six-camera sample selected from nuImages metadata."""

    sample_token: str
    sample_timestamp: int
    paths: Dict[str, str]
    sample_data_tokens: Dict[str, str]
    camera_timestamps: Dict[str, int]

    @property
    def max_time_delta_us(self) -> int:
        timestamps = list(self.camera_timestamps.values())
        if not timestamps:
            return 0
        return max(timestamps) - min(timestamps)

    def to_manifest_item(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["max_time_delta_us"] = self.max_time_delta_us
        return payload


def _metadata_dir(nuimages_root: Union[str, Path], version: str) -> Path:
    root = Path(nuimages_root)
    direct = root / version
    if (direct / "sample.json").is_file():
        return direct
    if (root / "sample.json").is_file():
        return root
    candidates = sorted(p for p in root.glob("v*.*-*") if (p / "sample.json").is_file())
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not find nuImages metadata under {root}. Expected {direct}/sample.json"
    )


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def _channel_maps(meta_dir: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    sensors = _read_json_list(meta_dir / "sensor.json")
    calibrated = _read_json_list(meta_dir / "calibrated_sensor.json")
    sensor_channel = {row["token"]: row["channel"] for row in sensors}
    calibrated_channel = {
        row["token"]: sensor_channel[row["sensor_token"]] for row in calibrated
    }
    return sensor_channel, calibrated_channel


def list_synchronized_sixcam_samples(
    nuimages_root: Union[str, Path] = DEFAULT_NUIMAGES_ROOT,
    *,
    version: str = "v1.0-mini",
    channels: Sequence[str] = CAMERA_CHANNELS,
    require_key_frame: bool = True,
) -> List[SixCameraSample]:
    """Return complete six-camera samples proven by nuImages metadata.

    This function is intentionally strict: no filename guessing, no image
    cycling, and no mixing samples from different timestamps.  If a nuImages
    split only has one camera per sample, as ``v1.0-mini`` commonly does, the
    result is an empty list.
    """

    root = Path(nuimages_root)
    meta_dir = _metadata_dir(root, version)
    required = tuple(channels)

    samples = _read_json_list(meta_dir / "sample.json")
    sample_data = _read_json_list(meta_dir / "sample_data.json")
    _, calibrated_channel = _channel_maps(meta_dir)

    by_sample: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in sample_data:
        if require_key_frame and not row.get("is_key_frame", False):
            continue
        channel = calibrated_channel.get(row.get("calibrated_sensor_token"))
        if channel not in required:
            continue
        filename = row.get("filename")
        if not filename or not (root / filename).is_file():
            continue
        by_sample[row["sample_token"]][channel].append(row)

    complete: List[SixCameraSample] = []
    for sample in sorted(samples, key=lambda row: (row.get("timestamp", 0), row["token"])):
        grouped = by_sample.get(sample["token"], {})
        if not all(grouped.get(channel) for channel in required):
            continue

        sample_ts = int(sample.get("timestamp", 0))
        paths: Dict[str, str] = {}
        data_tokens: Dict[str, str] = {}
        camera_timestamps: Dict[str, int] = {}
        for channel in required:
            rows = sorted(
                grouped[channel],
                key=lambda row: abs(int(row.get("timestamp", 0)) - sample_ts),
            )
            selected = rows[0]
            paths[channel] = str(root / selected["filename"])
            data_tokens[channel] = selected["token"]
            camera_timestamps[channel] = int(selected["timestamp"])

        complete.append(
            SixCameraSample(
                sample_token=sample["token"],
                sample_timestamp=sample_ts,
                paths=paths,
                sample_data_tokens=data_tokens,
                camera_timestamps=camera_timestamps,
            )
        )

    return complete


def summarize_sixcam_availability(
    nuimages_root: Union[str, Path] = DEFAULT_NUIMAGES_ROOT,
    *,
    version: str = "v1.0-mini",
    channels: Sequence[str] = CAMERA_CHANNELS,
    require_key_frame: bool = True,
    max_examples: int = 5,
) -> Dict[str, Any]:
    """Summarize why a strict six-camera selection can or cannot be made."""

    root = Path(nuimages_root)
    meta_dir = _metadata_dir(root, version)
    required = tuple(channels)

    samples = _read_json_list(meta_dir / "sample.json")
    sample_data = _read_json_list(meta_dir / "sample_data.json")
    _, calibrated_channel = _channel_maps(meta_dir)

    by_sample: Dict[str, set] = defaultdict(set)
    per_channel = Counter()
    missing_files = 0
    for row in sample_data:
        if require_key_frame and not row.get("is_key_frame", False):
            continue
        channel = calibrated_channel.get(row.get("calibrated_sensor_token"))
        if channel not in required:
            continue
        filename = row.get("filename")
        if not filename or not (root / filename).is_file():
            missing_files += 1
            continue
        by_sample[row["sample_token"]].add(channel)
        per_channel[channel] += 1

    channel_count_hist = Counter(len(by_sample.get(row["token"], set())) for row in samples)
    incomplete_examples: List[Dict[str, Any]] = []
    for row in sorted(samples, key=lambda item: (item.get("timestamp", 0), item["token"])):
        present = sorted(by_sample.get(row["token"], set()))
        missing = [channel for channel in required if channel not in present]
        if missing:
            incomplete_examples.append(
                {
                    "sample_token": row["token"],
                    "timestamp": row.get("timestamp"),
                    "present_channels": present,
                    "missing_channels": missing,
                }
            )
        if len(incomplete_examples) >= max_examples:
            break

    complete = sum(1 for row in samples if set(required).issubset(by_sample.get(row["token"], set())))
    return {
        "nuimages_root": str(root),
        "metadata_dir": str(meta_dir),
        "required_channels": list(required),
        "require_key_frame": require_key_frame,
        "sample_count": len(samples),
        "complete_sixcam_samples": complete,
        "per_sample_channel_count_histogram": dict(sorted(channel_count_hist.items())),
        "per_channel_keyframes_with_files": dict(sorted(per_channel.items())),
        "missing_file_rows": missing_files,
        "incomplete_examples": incomplete_examples,
    }


def select_sixcam_batches(
    nuimages_root: Union[str, Path] = DEFAULT_NUIMAGES_ROOT,
    *,
    version: str = "v1.0-mini",
    batch_count: int = 10,
    channels: Sequence[str] = CAMERA_CHANNELS,
    require_key_frame: bool = True,
) -> List[SixCameraSample]:
    """Select exactly ``batch_count`` complete six-camera samples.

    Raises a detailed ``ValueError`` when real synchronized data is unavailable.
    """

    if batch_count < 1:
        raise ValueError("batch_count must be >= 1")

    samples = list_synchronized_sixcam_samples(
        nuimages_root,
        version=version,
        channels=channels,
        require_key_frame=require_key_frame,
    )
    if len(samples) < batch_count:
        summary = summarize_sixcam_availability(
            nuimages_root,
            version=version,
            channels=channels,
            require_key_frame=require_key_frame,
        )
        raise ValueError(
            f"Need {batch_count} synchronized six-camera batches, found {len(samples)}. "
            "This loader does not repeat or synthesize images. Availability: "
            f"{json.dumps(summary, indent=2)}"
        )
    return samples[:batch_count]


def load_sixcam_batch_tensor(
    nuimages_root: Union[str, Path] = DEFAULT_NUIMAGES_ROOT,
    *,
    version: str = "v1.0-mini",
    batch_count: int = 10,
    height: int = 224,
    width: int = 224,
    normalize: str = "imagenet",
    channels: Sequence[str] = CAMERA_CHANNELS,
    selected: Optional[Sequence[SixCameraSample]] = None,
):
    """Load real synchronized camera images as ``[B, 6, 3, H, W]``.

    ``torch`` and ``numpy`` are imported lazily so metadata inspection works on
    machines that have not installed ML dependencies yet.
    """

    import torch

    from benchmarks.image_inputs import load_image_tensor

    chosen = list(selected) if selected is not None else select_sixcam_batches(
        nuimages_root,
        version=version,
        batch_count=batch_count,
        channels=channels,
    )
    if len(chosen) != batch_count:
        raise ValueError(f"Expected {batch_count} selected samples, got {len(chosen)}")

    per_batch = []
    for sample in chosen:
        per_camera = [
            load_image_tensor(sample.paths[channel], height, width, normalize=normalize)
            for channel in channels
        ]
        per_batch.append(torch.stack(per_camera, dim=0))

    tensor = torch.stack(per_batch, dim=0).to(dtype=torch.float32, device="cpu")
    expected = (batch_count, len(channels), 3, height, width)
    if tuple(tensor.shape) != expected:
        raise RuntimeError(f"Unexpected six-camera tensor shape {tuple(tensor.shape)}")

    meta = build_manifest(
        chosen,
        nuimages_root=nuimages_root,
        version=version,
        height=height,
        width=width,
        normalize=normalize,
        channels=channels,
    )
    return tensor, meta


def flatten_sixcam_tensor(tensor):
    """Convert ``[B, 6, 3, H, W]`` to CNN-ready ``[B*6, 3, H, W]``."""

    if tensor.ndim != 5:
        raise ValueError(f"Expected rank-5 [B,6,3,H,W] tensor, got {tuple(tensor.shape)}")
    batch_count, camera_count, channels, height, width = tensor.shape
    return tensor.reshape(batch_count * camera_count, channels, height, width).contiguous()


def build_manifest(
    samples: Sequence[SixCameraSample],
    *,
    nuimages_root: Union[str, Path],
    version: str,
    height: int,
    width: int,
    normalize: str,
    channels: Sequence[str] = CAMERA_CHANNELS,
) -> Dict[str, Any]:
    return {
        "source": "nuimages_synchronized_sixcam",
        "nuimages_root": str(Path(nuimages_root)),
        "version": version,
        "batch_count": len(samples),
        "camera_count": len(channels),
        "flattened_image_count": len(samples) * len(channels),
        "height": height,
        "width": width,
        "normalize": normalize,
        "channels": list(channels),
        "samples": [sample.to_manifest_item() for sample in samples],
    }


def write_manifest(manifest: Mapping[str, Any], path: Union[str, Path]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def read_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("source") != "nuimages_synchronized_sixcam":
        raise ValueError(f"Not a synchronized six-camera manifest: {path}")
    return manifest


def samples_from_manifest(manifest: Mapping[str, Any]) -> List[SixCameraSample]:
    samples = []
    for item in manifest.get("samples", []):
        samples.append(
            SixCameraSample(
                sample_token=item["sample_token"],
                sample_timestamp=int(item["sample_timestamp"]),
                paths=dict(item["paths"]),
                sample_data_tokens=dict(item["sample_data_tokens"]),
                camera_timestamps={
                    channel: int(timestamp)
                    for channel, timestamp in item["camera_timestamps"].items()
                },
            )
        )
    return samples


def load_sixcam_tensor_from_manifest(path: Union[str, Path]):
    manifest = read_manifest(path)
    samples = samples_from_manifest(manifest)
    return load_sixcam_batch_tensor(
        manifest["nuimages_root"],
        version=manifest.get("version", "v1.0-mini"),
        batch_count=int(manifest["batch_count"]),
        height=int(manifest["height"]),
        width=int(manifest["width"]),
        normalize=manifest.get("normalize", "imagenet"),
        channels=tuple(manifest.get("channels", CAMERA_CHANNELS)),
        selected=samples,
    )
