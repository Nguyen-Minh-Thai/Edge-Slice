"""
Regression tests for NuScenesLoader against a fake devkit.

The real nuscenes-devkit and the mini split are both optional, so every case
here drives a stand-in ``NuScenes`` object. That is enough to pin the
correctness bugs this suite exists for:

  * annotation counts came from ``len(anns[0]['translation'])`` — the length of
    an (x, y, z) vector — so statistics reported 3 boxes for every sample
    regardless of the truth, and a sample with no annotations raised IndexError;
  * a failed ``load()`` returned False, was ignored, and the next line
    dereferenced ``self.nusc`` as None;
  * ``load_sample`` populated only CAM_FRONT and CAM_FRONT_LEFT, leaving four
    of the six cameras silently empty.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy is required for the fake devkit")

pytestmark = pytest.mark.cpu

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


class FakeNuScenes:
    """Minimal stand-in for the devkit's NuScenes object.

    Implements only ``get``/``scene``/``sample``, which is the whole surface
    NuScenesLoader touches.
    """

    def __init__(self, samples, tables):
        self.sample = samples
        self.scene = [{"token": "scene_0", "name": "scene-0001"}]
        self._tables = tables

    def get(self, table: str, token: str):
        return self._tables[table][token]


def _build_fake(
    *,
    annotation_count: int,
    image_files: dict | None = None,
    lidar_file: str | None = None,
):
    """Fake dataset with one sample carrying *annotation_count* annotations."""
    ann_tokens = [f"ann_{i}" for i in range(annotation_count)]
    annotations = {
        token: {
            # A 3-element vector: measuring *this* is what produced the
            # constant "3 boxes" the old statistics reported.
            "translation": [1.0, 2.0, 3.0],
            "category_name": "vehicle.car" if index % 2 == 0 else "human.pedestrian",
        }
        for index, token in enumerate(ann_tokens)
    }

    sample = {
        "token": "sample_0",
        "timestamp": 1_500_000_000,
        "scene_token": "scene_0",
        "anns": ann_tokens,
        "data": {cam: f"{cam}_data" for cam in CAMERAS},
    }
    sample["data"]["LIDAR_TOP"] = "lidar_data"

    sample_data = {
        f"{cam}_data": {
            "filename": (image_files or {}).get(cam, f"images/{cam}.png"),
            "height": 900,
            "width": 1600,
            "calibrated_sensor_token": f"calib_{cam}",
        }
        for cam in CAMERAS
    }
    sample_data["lidar_data"] = {
        "filename": lidar_file if lidar_file is not None else "lidar/points.bin",
        "height": 0,
        "width": 0,
        "calibrated_sensor_token": "calib_lidar",
    }

    tables = {
        "sample": {"sample_0": sample},
        "scene": {"scene_0": {"token": "scene_0", "name": "scene-0001"}},
        "sample_annotation": annotations,
        "sample_data": sample_data,
        "calibrated_sensor": {
            f"calib_{cam}": {"camera_intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
            for cam in CAMERAS
        },
    }
    return FakeNuScenes([sample], tables)


@pytest.fixture
def loader_module(monkeypatch):
    """Import the loader with the devkit's absence patched out.

    The module refuses to construct a loader when nuscenes-devkit is missing,
    which is correct in production and useless for these tests.
    """
    if "nuscenes" not in sys.modules:
        stub = types.ModuleType("nuscenes")
        nuscenes_mod = types.ModuleType("nuscenes.nuscenes")
        nuscenes_mod.NuScenes = object
        utils_mod = types.ModuleType("nuscenes.utils")
        data_classes = types.ModuleType("nuscenes.utils.data_classes")
        data_classes.LidarPointCloud = object
        data_classes.RadarPointCloud = object
        monkeypatch.setitem(sys.modules, "nuscenes", stub)
        monkeypatch.setitem(sys.modules, "nuscenes.nuscenes", nuscenes_mod)
        monkeypatch.setitem(sys.modules, "nuscenes.utils", utils_mod)
        monkeypatch.setitem(sys.modules, "nuscenes.utils.data_classes", data_classes)

    import edgeslide.dataset.nuscenes_loader as module

    monkeypatch.setattr(module, "NUSCENES_AVAILABLE", True, raising=False)
    return module


def _loader_with(loader_module, fake, dataroot: Path):
    loader = loader_module.NuScenesLoader(dataroot=str(dataroot))
    loader.nusc = fake
    loader._loaded = True
    return loader


# ---------------------------------------------------------------------------
# Annotation counting (0, 1, many)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("annotation_count", [0, 1, 2, 7, 23])
def test_num_boxes_matches_the_annotation_list(
    loader_module, tmp_path, annotation_count
):
    """num_boxes is len(anns) — never the width of a translation vector."""
    loader = _loader_with(
        loader_module, _build_fake(annotation_count=annotation_count), tmp_path
    )

    sample = loader.load_sample("sample_0", load_images=False, load_lidar=False)

    assert sample.num_boxes == annotation_count


@pytest.mark.parametrize("annotation_count", [0, 1, 2, 7, 23])
def test_statistics_box_count_matches_the_annotation_list(
    loader_module, tmp_path, annotation_count
):
    """The old code reported 3 for every sample; pin the real count."""
    loader = _loader_with(
        loader_module, _build_fake(annotation_count=annotation_count), tmp_path
    )

    stats = loader.get_statistics()

    assert stats["annotation_boxes"]["mean"] == float(annotation_count)
    assert stats["annotation_boxes"]["total"] == annotation_count


def test_statistics_box_count_is_not_hardcoded_to_three(loader_module, tmp_path):
    """The precise regression: 5 annotations must not report 3."""
    loader = _loader_with(loader_module, _build_fake(annotation_count=5), tmp_path)

    assert loader.get_statistics()["annotation_boxes"]["mean"] == 5.0


def test_sample_with_no_annotations_does_not_raise(loader_module, tmp_path):
    """anns[0] on an empty list used to raise IndexError."""
    loader = _loader_with(loader_module, _build_fake(annotation_count=0), tmp_path)

    sample = loader.load_sample("sample_0", load_images=False, load_lidar=False)
    stats = loader.get_statistics()

    assert sample.num_boxes == 0
    assert sample.categories == []
    assert stats["annotation_boxes"]["mean"] == 0.0


def test_categories_are_deduplicated_and_sorted(loader_module, tmp_path):
    loader = _loader_with(loader_module, _build_fake(annotation_count=4), tmp_path)

    sample = loader.load_sample("sample_0", load_images=False, load_lidar=False)

    assert sample.categories == ["human.pedestrian", "vehicle.car"]


# ---------------------------------------------------------------------------
# All six cameras
# ---------------------------------------------------------------------------


def _write_images(root: Path, size=(4, 6)):
    """Write a distinct PNG per camera so each can be told apart."""
    Image = pytest.importorskip("PIL.Image", reason="pillow is required")
    files = {}
    for index, cam in enumerate(CAMERAS):
        rel = f"images/{cam}.png"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.full((size[0], size[1], 3), index * 10, dtype=np.uint8)
        Image.fromarray(array).save(path)
        files[cam] = rel
    return files


def test_all_six_cameras_are_populated(loader_module, tmp_path):
    """Four of six were left as `# ... etc` and never assigned."""
    files = _write_images(tmp_path)
    loader = _loader_with(
        loader_module, _build_fake(annotation_count=1, image_files=files), tmp_path
    )

    sample = loader.load_sample("sample_0", load_images=True, load_lidar=False)

    for attribute in (
        "camera_front",
        "camera_front_left",
        "camera_front_right",
        "camera_back",
        "camera_back_left",
        "camera_back_right",
    ):
        image = getattr(sample, attribute)
        assert image is not None, f"{attribute} was never populated"
        assert image.shape == (4, 6, 3)


def test_each_camera_receives_its_own_image(loader_module, tmp_path):
    """Guards against every attribute being assigned the same frame."""
    files = _write_images(tmp_path)
    loader = _loader_with(
        loader_module, _build_fake(annotation_count=1, image_files=files), tmp_path
    )

    sample = loader.load_sample("sample_0", load_images=True, load_lidar=False)

    observed = [
        int(getattr(sample, attr)[0, 0, 0])
        for attr in (
            "camera_front",
            "camera_front_left",
            "camera_front_right",
            "camera_back",
            "camera_back_left",
            "camera_back_right",
        )
    ]
    assert len(set(observed)) == len(CAMERAS)


def test_camera_resolution_is_taken_from_cam_front(loader_module, tmp_path):
    files = _write_images(tmp_path, size=(8, 12))
    loader = _loader_with(
        loader_module, _build_fake(annotation_count=1, image_files=files), tmp_path
    )

    sample = loader.load_sample("sample_0", load_images=True, load_lidar=False)

    assert sample.camera_resolution == (8, 12)


def test_images_are_closed_after_loading(loader_module, tmp_path):
    """PIL holds the descriptor open until closed; a sweep exhausts the limit."""
    files = _write_images(tmp_path)
    loader = _loader_with(
        loader_module, _build_fake(annotation_count=1, image_files=files), tmp_path
    )

    opened = []
    Image = pytest.importorskip("PIL.Image", reason="pillow is required")
    real_open = Image.open

    def tracking_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        opened.append(handle)
        return handle

    import PIL.Image as pil_image

    original = pil_image.open
    pil_image.open = tracking_open
    try:
        loader.load_sample("sample_0", load_images=True, load_lidar=False)
    finally:
        pil_image.open = original

    assert len(opened) == len(CAMERAS)
    for handle in opened:
        assert getattr(handle, "fp", None) is None, "image file left open"


# ---------------------------------------------------------------------------
# Missing files
# ---------------------------------------------------------------------------


def test_missing_image_leaves_the_camera_unset_without_raising(
    loader_module, tmp_path
):
    files = _write_images(tmp_path)
    files["CAM_BACK"] = "images/does_not_exist.png"
    loader = _loader_with(
        loader_module, _build_fake(annotation_count=1, image_files=files), tmp_path
    )

    sample = loader.load_sample("sample_0", load_images=True, load_lidar=False)

    assert sample.camera_back is None
    # The other five still load: one missing file must not abort the sweep.
    assert sample.camera_front is not None
    assert sample.camera_back_left is not None


def test_missing_lidar_file_leaves_points_unset(loader_module, tmp_path):
    loader = _loader_with(
        loader_module,
        _build_fake(annotation_count=1, lidar_file="lidar/absent.bin"),
        tmp_path,
    )

    sample = loader.load_sample("sample_0", load_images=False, load_lidar=True)

    assert sample.lidar_points is None
    assert sample.num_boxes == 1


def test_missing_lidar_is_skipped_in_statistics(loader_module, tmp_path):
    loader = _loader_with(
        loader_module,
        _build_fake(annotation_count=2, lidar_file="lidar/absent.bin"),
        tmp_path,
    )

    stats = loader.get_statistics()

    assert stats["lidar_point_count"]["mean"] == 0
    assert stats["annotation_boxes"]["mean"] == 2.0


# ---------------------------------------------------------------------------
# Load failure
# ---------------------------------------------------------------------------


def test_load_failure_raises_typed_error_chaining_the_cause(
    loader_module, monkeypatch, tmp_path
):
    """Not a False return that the caller then ignores."""
    def exploding_nuscenes(*args, **kwargs):
        raise FileNotFoundError("category.json not found")

    monkeypatch.setattr(loader_module, "NuScenes", exploding_nuscenes)
    loader = loader_module.NuScenesLoader(dataroot=str(tmp_path))

    with pytest.raises(loader_module.NuScenesLoadError) as excinfo:
        loader.load()

    assert isinstance(excinfo.value.__cause__, FileNotFoundError)
    assert "category.json" in str(excinfo.value)
    assert str(tmp_path) in str(excinfo.value)


def test_statistics_after_failed_load_raises_not_attribute_error(
    loader_module, monkeypatch, tmp_path
):
    """The old path reached `self.nusc.sample` with self.nusc still None."""
    def exploding_nuscenes(*args, **kwargs):
        raise FileNotFoundError("category.json not found")

    monkeypatch.setattr(loader_module, "NuScenes", exploding_nuscenes)
    loader = loader_module.NuScenesLoader(dataroot=str(tmp_path))

    with pytest.raises(loader_module.NuScenesLoadError):
        loader.get_statistics()


def test_load_sample_after_failed_load_raises_typed_error(
    loader_module, monkeypatch, tmp_path
):
    def exploding_nuscenes(*args, **kwargs):
        raise RuntimeError("corrupt table")

    monkeypatch.setattr(loader_module, "NuScenes", exploding_nuscenes)
    loader = loader_module.NuScenesLoader(dataroot=str(tmp_path))

    with pytest.raises(loader_module.NuScenesLoadError):
        loader.load_sample("sample_0")


def test_loader_state_stays_unloaded_after_a_failure(
    loader_module, monkeypatch, tmp_path
):
    """A half-loaded loader would let a later call skip the retry and crash."""
    def exploding_nuscenes(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(loader_module, "NuScenes", exploding_nuscenes)
    loader = loader_module.NuScenesLoader(dataroot=str(tmp_path))

    with pytest.raises(loader_module.NuScenesLoadError):
        loader.load()

    assert loader.nusc is None
    assert loader._loaded is False


# ---------------------------------------------------------------------------
# Default dataroot
# ---------------------------------------------------------------------------


def test_default_dataroot_is_not_an_absolute_personal_path(loader_module, monkeypatch):
    monkeypatch.delenv(loader_module.DEFAULT_DATAROOT_ENV, raising=False)

    loader = loader_module.NuScenesLoader()

    assert loader.dataroot == loader_module.DEFAULT_DATAROOT
    assert not Path(loader.dataroot).is_absolute()


def test_dataroot_env_override(loader_module, monkeypatch, tmp_path):
    monkeypatch.setenv(loader_module.DEFAULT_DATAROOT_ENV, str(tmp_path))

    assert loader_module.NuScenesLoader().dataroot == str(tmp_path)


def test_explicit_dataroot_wins_over_env(loader_module, monkeypatch, tmp_path):
    monkeypatch.setenv(loader_module.DEFAULT_DATAROOT_ENV, "/env/path")

    assert loader_module.NuScenesLoader(dataroot=str(tmp_path)).dataroot == str(tmp_path)
