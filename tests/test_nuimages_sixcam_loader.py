"""Tests for strict synchronized six-camera nuImages selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.nuimages_sixcam import (
    CAMERA_CHANNELS,
    build_manifest,
    select_sixcam_batches,
    summarize_sixcam_availability,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _make_fake_nuimages(root: Path, *, sample_count: int, complete: bool) -> None:
    meta = root / "v1.0-mini"
    channels = CAMERA_CHANNELS if complete else (CAMERA_CHANNELS[0],)

    sensors = [
        {"token": f"sensor_{channel}", "channel": channel, "modality": "camera"}
        for channel in CAMERA_CHANNELS
    ]
    calibrated = [
        {
            "token": f"calibrated_{channel}",
            "sensor_token": f"sensor_{channel}",
            "translation": [0.0, 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "camera_intrinsic": [],
            "camera_distortion": [],
        }
        for channel in CAMERA_CHANNELS
    ]
    samples = []
    sample_data = []
    for index in range(sample_count):
        sample_token = f"sample_{index:02d}"
        timestamp = 1_700_000_000_000_000 + index * 100_000
        samples.append(
            {
                "token": sample_token,
                "timestamp": timestamp,
                "log_token": "log",
                "key_camera_token": f"data_{index:02d}_{channels[0]}",
            }
        )
        for camera_index, channel in enumerate(channels):
            rel = Path("samples") / channel / f"sample_{index:02d}__{channel}__.jpg"
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake-jpeg-placeholder")
            sample_data.append(
                {
                    "token": f"data_{index:02d}_{channel}",
                    "sample_token": sample_token,
                    "ego_pose_token": "ego",
                    "calibrated_sensor_token": f"calibrated_{channel}",
                    "filename": rel.as_posix(),
                    "fileformat": "jpg",
                    "width": 1600,
                    "height": 900,
                    "timestamp": timestamp + camera_index,
                    "is_key_frame": True,
                    "prev": "",
                    "next": "",
                }
            )

    _write_json(meta / "sensor.json", sensors)
    _write_json(meta / "calibrated_sensor.json", calibrated)
    _write_json(meta / "sample.json", samples)
    _write_json(meta / "sample_data.json", sample_data)


def test_selects_exactly_10_complete_sixcam_batches(tmp_path):
    _make_fake_nuimages(tmp_path, sample_count=12, complete=True)

    selected = select_sixcam_batches(tmp_path, batch_count=10)
    assert len(selected) == 10
    assert all(set(sample.paths) == set(CAMERA_CHANNELS) for sample in selected)
    assert all(sample.max_time_delta_us == 5 for sample in selected)

    manifest = build_manifest(
        selected,
        nuimages_root=tmp_path,
        version="v1.0-mini",
        height=224,
        width=224,
        normalize="imagenet",
    )
    assert manifest["batch_count"] == 10
    assert manifest["camera_count"] == 6
    assert manifest["flattened_image_count"] == 60


def test_incomplete_sixcam_data_fails_without_repeating(tmp_path):
    _make_fake_nuimages(tmp_path, sample_count=12, complete=False)

    summary = summarize_sixcam_availability(tmp_path)
    assert summary["complete_sixcam_samples"] == 0
    assert summary["per_sample_channel_count_histogram"] == {1: 12}

    with pytest.raises(ValueError, match="does not repeat or synthesize"):
        select_sixcam_batches(tmp_path, batch_count=10)

