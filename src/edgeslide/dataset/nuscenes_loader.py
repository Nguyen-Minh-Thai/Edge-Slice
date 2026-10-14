"""nuScenes dataset loader for EdgeSlide benchmarking."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# NuScenes devkit
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    NUSCENES_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the local install
    NUSCENES_AVAILABLE = False
    logger.debug(
        "nuscenes-devkit not installed; install with: pip install nuscenes-devkit"
    )

# Resolved when no dataroot is given. A relative path keeps the default working
# on any checkout; $NUSCENES_DATAROOT overrides it without code changes.
DEFAULT_DATAROOT_ENV = "NUSCENES_DATAROOT"
DEFAULT_DATAROOT = "data/v1.0-mini"


class NuScenesLoaderError(RuntimeError):
    """Base class for loader failures."""


class NuScenesUnavailableError(NuScenesLoaderError):
    """nuscenes-devkit is not installed."""


class NuScenesLoadError(NuScenesLoaderError):
    """The dataset could not be opened.

    Raised instead of returning a falsy value so a caller cannot proceed to
    touch ``self.nusc`` and get an ``AttributeError`` on ``None`` that says
    nothing about the real cause. The originating exception is chained.
    """


@dataclass
class NuScenesSample:
    """A single sample from nuScenes dataset."""
    token: str
    timestamp: int
    scene_name: str

    # Camera data (6 cameras)
    camera_front: Optional[np.ndarray] = None  # (H, W, 3)
    camera_front_left: Optional[np.ndarray] = None
    camera_front_right: Optional[np.ndarray] = None
    camera_back: Optional[np.ndarray] = None
    camera_back_left: Optional[np.ndarray] = None
    camera_back_right: Optional[np.ndarray] = None

    # Camera metadata
    camera_resolution: Tuple[int, int] = (900, 1600)

    # LiDAR data
    lidar_points: Optional[np.ndarray] = None  # (N, 4) - x, y, z, intensity

    # Radar data (5 radars)
    radar_points: Optional[List[np.ndarray]] = None  # List of (N, 18) arrays

    # Annotations
    num_boxes: int = 0
    categories: List[str] = field(default_factory=list)


class NuScenesLoader:
    """Loader for nuScenes dataset."""

    def __init__(
        self,
        dataroot: Optional[str] = None,
        version: str = "v1.0-mini",
    ):
        """Initialize loader with the dataset path.

        Args:
            dataroot: Path to the nuScenes dataset. When omitted, falls back to
                ``$NUSCENES_DATAROOT`` and then to the repo-relative
                ``data/v1.0-mini``. There is deliberately no absolute default:
                one baked to a developer's home directory silently resolves to
                nothing on every other machine.
            version: Dataset version
        """
        if not NUSCENES_AVAILABLE:
            raise NuScenesUnavailableError(
                "nuscenes-devkit is required. Install with: pip install nuscenes-devkit"
            )

        self.dataroot = dataroot or os.environ.get(
            DEFAULT_DATAROOT_ENV, DEFAULT_DATAROOT
        )
        self.version = version
        self.nusc: Optional[NuScenes] = None
        self._loaded = False

        # Camera names
        self.camera_names = [
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]

        # Channel -> NuScenesSample attribute. Keeping the mapping in one place
        # is what stops load_sample() from silently populating only the two
        # channels someone happened to write an elif for.
        self.camera_attributes = {
            'CAM_FRONT': 'camera_front',
            'CAM_FRONT_LEFT': 'camera_front_left',
            'CAM_FRONT_RIGHT': 'camera_front_right',
            'CAM_BACK': 'camera_back',
            'CAM_BACK_LEFT': 'camera_back_left',
            'CAM_BACK_RIGHT': 'camera_back_right',
        }

        # Radar names
        self.radar_names = [
            'RADAR_FRONT',
            'RADAR_FRONT_LEFT',
            'RADAR_FRONT_RIGHT',
            'RADAR_BACK_LEFT',
            'RADAR_BACK_RIGHT',
        ]

    def load(self) -> bool:
        """Load nuScenes dataset.

        Returns:
            True if loaded successfully
        """
        if self._loaded:
            return True

        logger.info("Loading nuScenes %s from %s", self.version, self.dataroot)
        try:
            self.nusc = NuScenes(
                version=self.version, dataroot=self.dataroot, verbose=False
            )
        except Exception as exc:
            # Chained, not swallowed: the devkit's message ("no such file",
            # "missing table") is the only thing that says what to fix.
            self.nusc = None
            self._loaded = False
            raise NuScenesLoadError(
                f"Failed to load nuScenes {self.version} from {self.dataroot}: {exc}"
            ) from exc

        self._loaded = True
        logger.info(
            "Loaded %d scenes, %d samples",
            len(self.nusc.scene),
            len(self.nusc.sample),
        )
        return True

    def _ensure_loaded(self) -> "NuScenes":
        """Return the devkit handle, loading on first use.

        Every method goes through here so none of them can reach a ``None``
        ``self.nusc``: the previous code called ``load()`` and ignored its
        boolean, then dereferenced the handle anyway.
        """
        if not self._loaded or self.nusc is None:
            self.load()
        if self.nusc is None:  # pragma: no cover - load() raises first
            raise NuScenesLoadError("nuScenes dataset is not loaded")
        return self.nusc

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        nusc = self._ensure_loaded()

        # Analyze a subset of samples for statistics
        num_samples_to_analyze = min(50, len(nusc.sample))

        camera_sizes = []
        lidar_counts = []
        box_counts = []

        for i in range(num_samples_to_analyze):
            sample = nusc.sample[i]

            # Camera data
            cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            camera_sizes.append((cam_front_data['height'], cam_front_data['width']))

            # LiDAR data
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_file = os.path.join(self.dataroot, lidar_data['filename'])
            if os.path.exists(lidar_file):
                pc = LidarPointCloud.from_file(lidar_file)
                lidar_counts.append(pc.points.shape[1])
            else:
                logger.warning("LiDAR sweep missing, skipping: %s", lidar_file)

            # Annotations. The count is simply how many annotation tokens the
            # sample carries. The previous expression took the *first*
            # annotation and measured its 'translation' vector, which is an
            # (x, y, z) triple — so every sample reported exactly 3 boxes, and
            # a sample with no annotations raised IndexError.
            box_counts.append(len(sample['anns']))

        if not camera_sizes:
            raise NuScenesLoadError(
                f"No samples found in nuScenes {self.version} at {self.dataroot}"
            )

        return {
            "num_scenes": len(nusc.scene),
            "num_samples": len(nusc.sample),
            "camera_resolution": {
                "height": int(np.mean([s[0] for s in camera_sizes])),
                "width": int(np.mean([s[1] for s in camera_sizes])),
            },
            "camera_resolution_range": {
                "min_height": min(s[0] for s in camera_sizes),
                "max_height": max(s[0] for s in camera_sizes),
                "min_width": min(s[1] for s in camera_sizes),
                "max_width": max(s[1] for s in camera_sizes),
            },
            "lidar_point_count": {
                "mean": float(np.mean(lidar_counts)) if lidar_counts else 0,
                "min": float(np.min(lidar_counts)) if lidar_counts else 0,
                "max": float(np.max(lidar_counts)) if lidar_counts else 0,
            },
            "annotation_boxes": {
                "mean": float(np.mean(box_counts)) if box_counts else 0.0,
                "min": int(np.min(box_counts)) if box_counts else 0,
                "max": int(np.max(box_counts)) if box_counts else 0,
                "total": int(np.sum(box_counts)) if box_counts else 0,
            },
        }

    def load_sample(self, sample_token: str, load_images: bool = False,
                    load_lidar: bool = True, load_radar: bool = False) -> NuScenesSample:
        """Load a single sample from the dataset.

        Args:
            sample_token: Sample token
            load_images: Whether to load actual image data (memory intensive)
            load_lidar: Whether to load LiDAR point clouds
            load_radar: Whether to load radar data

        Returns:
            NuScenesSample with loaded data
        """
        nusc = self._ensure_loaded()

        sample = nusc.get('sample', sample_token)
        scene = nusc.get('scene', sample['scene_token'])

        nusc_sample = NuScenesSample(
            token=sample_token,
            timestamp=sample['timestamp'],
            scene_name=scene['name'],
            categories=[],
        )

        # Load camera data — every channel in camera_attributes, not just the
        # first two.
        if load_images:
            from PIL import Image

            for cam_name in self.camera_names:
                attribute = self.camera_attributes.get(cam_name)
                if attribute is None:
                    logger.warning("No sample attribute for camera %s", cam_name)
                    continue

                cam_token = sample['data'].get(cam_name)
                if cam_token is None:
                    logger.warning(
                        "Sample %s has no %s entry", sample_token, cam_name
                    )
                    continue

                cam_data = nusc.get('sample_data', cam_token)
                image_path = os.path.join(self.dataroot, cam_data['filename'])
                if not os.path.exists(image_path):
                    logger.warning(
                        "Image missing for %s, leaving unset: %s",
                        cam_name,
                        image_path,
                    )
                    continue

                # Context manager: PIL keeps the file handle open until the
                # image is closed, and a six-camera sweep over a full split
                # exhausts the descriptor limit otherwise.
                with Image.open(image_path) as handle:
                    image = np.array(handle)

                setattr(nusc_sample, attribute, image)
                if cam_name == 'CAM_FRONT':
                    nusc_sample.camera_resolution = (image.shape[0], image.shape[1])

        # Load LiDAR data
        if load_lidar:
            lidar_token = sample['data'].get('LIDAR_TOP')
            if lidar_token is None:
                logger.warning("Sample %s has no LIDAR_TOP entry", sample_token)
            else:
                lidar_data = nusc.get('sample_data', lidar_token)
                lidar_path = os.path.join(self.dataroot, lidar_data['filename'])
                if os.path.exists(lidar_path):
                    pc = LidarPointCloud.from_file(lidar_path)
                    # Transpose to (N, 4)
                    nusc_sample.lidar_points = pc.points.T  # x, y, z, intensity
                else:
                    logger.warning(
                        "LiDAR sweep missing, leaving unset: %s", lidar_path
                    )

        # Load annotations. An empty 'anns' list is a valid sample, not an error.
        annotations = [nusc.get('sample_annotation', ann_token)
                       for ann_token in sample['anns']]
        nusc_sample.num_boxes = len(annotations)
        nusc_sample.categories = sorted({ann['category_name'] for ann in annotations})

        return nusc_sample

    def iterate_samples(self, max_samples: Optional[int] = None,
                       scene_filter: Optional[List[str]] = None):
        """Iterate through samples in the dataset.

        Args:
            max_samples: Maximum number of samples to yield
            scene_filter: List of scene names to filter by

        Yields:
            NuScenesSample objects
        """
        nusc = self._ensure_loaded()

        count = 0
        for sample in nusc.sample:
            if max_samples and count >= max_samples:
                break

            scene = nusc.get('scene', sample['scene_token'])

            # Filter by scene
            if scene_filter and scene['name'] not in scene_filter:
                continue

            # Load sample (without images to save memory)
            nusc_sample = self.load_sample(
                sample['token'],
                load_images=False,
                load_lidar=True,
                load_radar=False,
            )

            yield nusc_sample
            count += 1

    def get_model_input_statistics(self, model_type: str) -> Dict[str, Any]:
        """Get realistic input statistics for a specific model type.

        Args:
            model_type: One of 'pointpillars', 'centerpoint', 'bevfusion'

        Returns:
            Dictionary with input tensor shapes and statistics
        """
        stats = self.get_statistics()

        if model_type == "pointpillars":
            # PointPillars: Voxelized point cloud
            return {
                "model_type": "PointPillars",
                "input_type": "voxelized_lidar",
                "voxel_size": (0.1, 0.1, 0.1),  # meters
                "grid_size": (512, 512, 1),  # x, y, z
                "point_cloud_range": (-51.2, -51.2, -5.0, 51.2),
                "input_shape": (1, 64, 512, 512),  # (batch, channels, height, width)
                "num_points_mean": stats["lidar_point_count"]["mean"],
                "num_points_max": stats["lidar_point_count"]["max"],
            }

        elif model_type == "centerpoint":
            # CenterPoint: Similar but different resolution
            return {
                "model_type": "CenterPoint",
                "input_type": "voxelized_lidar",
                "voxel_size": (0.075, 0.075, 0.2),
                "grid_size": (512, 512, 1),
                "point_cloud_range": (-51.2, -51.2, -5.0, 51.2),
                "input_shape": (1, 128, 256, 256),
                "num_points_mean": stats["lidar_point_count"]["mean"],
            }

        elif model_type == "bevfusion":
            # BEVFusion: Multi-modal
            return {
                "model_type": "BEVFusion",
                "input_type": "multi_modal",
                "camera_resolution": stats["camera_resolution"],
                "camera_input_shape": (1, 6, 3, stats["camera_resolution"]["height"],
                                      stats["camera_resolution"]["width"]),
                "lidar_voxel_shape": (1, 64, 512, 512),
                "lidar_points_mean": stats["lidar_point_count"]["mean"],
            }

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def process_batch_for_model(self, model_type: str, batch_size: int = 1,
                               max_samples: int = 10) -> Dict[str, np.ndarray]:
        """Process actual nuScenes data to create input batch for a model.

        Args:
            model_type: Model type to process for
            batch_size: Batch size
            max_samples: Maximum samples to process

        Returns:
            Dictionary with processed input tensors
        """
        input_stats = self.get_model_input_statistics(model_type)

        if model_type == "pointpillars":
            # Create voxelized point cloud batch
            # In real implementation, would use actual voxelization
            batch = {
                "voxel_features": np.random.randn(
                    batch_size, 64, 512, 512
                ).astype(np.float32),
                "coordinates": np.random.randint(
                    0, 512, size=(batch_size * 100, 3)
                ).astype(np.int32),
                "num_points": np.array([input_stats["num_points_mean"]] * batch_size),
            }

        elif model_type == "centerpoint":
            batch = {
                "voxel_features": np.random.randn(
                    batch_size, 128, 256, 256
                ).astype(np.float32),
            }

        elif model_type == "bevfusion":
            h, w = input_stats["camera_resolution"]["height"], input_stats["camera_resolution"]["width"]
            batch = {
                "camera_images": np.random.randn(
                    batch_size, 6, 3, h, w
                ).astype(np.float32),
                "lidar_voxels": np.random.randn(
                    batch_size, 64, 512, 512
                ).astype(np.float32),
            }

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return batch


def _demo() -> None:
    """Manual smoke run against a real dataset. Not collected by pytest.

    Named with a leading underscore on purpose: as ``test_nuscenes_loader`` it
    was collected as a test, and it called ``RealNuScenesLoader``, a class that
    does not exist in this module — so it raised NameError whenever collected.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    loader = NuScenesLoader(version="v1.0-mini")

    try:
        loader.load()
    except NuScenesLoaderError as exc:
        logger.error("%s", exc)
        return

    stats = loader.get_statistics()
    logger.info("Dataset statistics:")
    for key, value in stats.items():
        logger.info("  %s: %s", key, value)

    for model_type in ("pointpillars", "centerpoint", "bevfusion"):
        model_stats = loader.get_model_input_statistics(model_type)
        logger.info(
            "%s input_shape=%s", model_type, model_stats.get("input_shape", "N/A")
        )

    for index, sample in enumerate(loader.iterate_samples(max_samples=5), start=1):
        lidar_shape = (
            sample.lidar_points.shape if sample.lidar_points is not None else "N/A"
        )
        logger.info(
            "  sample %d: scene=%s lidar=%s boxes=%d",
            index,
            sample.scene_name,
            lidar_shape,
            sample.num_boxes,
        )


if __name__ == "__main__":
    _demo()
