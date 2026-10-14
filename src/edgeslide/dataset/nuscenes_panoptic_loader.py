"""nuScenes-panoptic Dataset Loader for EdgeSlide Benchmarking.

This module loads and processes the actual nuScenes-panoptic-v1.0-mini dataset
for realistic benchmarking of 3D object detection models.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# NuScenes devkit
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    logger.debug("nuscenes-devkit not installed.")


@dataclass
class NuScenesPanopticSample:
    """A single sample from nuScenes-panoptic dataset."""
    token: str
    timestamp: int
    scene_name: str

    # LiDAR data (primary for 3D detection)
    lidar_points: Optional[np.ndarray] = None  # (N, 5) - x, y, z, intensity, ring_index

    # Radar data (5 radars)
    radar_points: Optional[List[np.ndarray]] = None

    # Metadata
    num_lidar_points: int = 0
    lidar_file_path: Optional[str] = None


@dataclass
class ProcessedBatch:
    """Processed batch ready for model input."""
    voxel_features: torch.Tensor  # Voxelized features
    coordinates: torch.Tensor     # Voxel coordinates
    num_points: torch.Tensor      # Number of points per voxel
    batch_size: int
    metadata: Dict[str, Any]


class NuScenesPanopticLoader:
    """Loader for nuScenes-panoptic v1.0-mini dataset."""

    # PointPillars voxelization config
    POINTPILLARS_CONFIG: Dict[str, Any] = {
        "voxel_size": (0.16, 0.16, 4.0),  # meters
        "point_cloud_range": (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        "max_voxels": (16000, 40000),  # (max voxels per batch, max voxels per sample)
        "max_num_points": 32,  # max points per voxel
    }

    # CenterPoint voxelization config
    CENTERPOINT_CONFIG: Dict[str, Any] = {
        "voxel_size": (0.075, 0.075, 0.2),
        "point_cloud_range": (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        "max_voxels": (16000, 40000),
        "max_num_points": 10,
    }

    def __init__(self, dataroot: str, version: str = "v1.0-mini", verbose: bool = True):
        """Initialize the loader.

        Args:
            dataroot: Path to nuScenes dataset (e.g., /path/to/nuScenes-panoptic-v1.0-mini/nuScenes-panoptic-v1.0-mini)
            version: Dataset version
            verbose: Print loading info
        """
        self.dataroot = dataroot
        self.version = version
        self.verbose = verbose
        self.nusc: Optional[NuScenes] = None
        self._loaded = False

    def load(self) -> bool:
        """Load the nuScenes dataset.

        Returns:
            True if loaded successfully
        """
        if self._loaded:
            return True

        if not NUSCENES_AVAILABLE:
            raise RuntimeError(
                "nuscenes-devkit is required. Install with: pip install nuscenes-devkit"
            )

        try:
            if self.verbose:
                logger.info("Loading nuScenes %s from %s", self.version, self.dataroot)

            self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)
            self._loaded = True

            if self.verbose:
                logger.info("Loaded %d scenes, %d samples", len(self.nusc.scene), len(self.nusc.sample))

            return True

        except Exception as e:
            logger.error("Failed to load dataset from %s: %s", self.dataroot, e)
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        if not self._loaded:
            self.load()
        if self.nusc is None:
            raise RuntimeError(
                f"nuScenes dataset is not loaded (dataroot={self.dataroot})"
            )

        # Analyze a subset of samples for statistics
        num_samples_to_analyze = min(50, len(self.nusc.sample))

        lidar_counts = []
        sample_tokens = []

        for i in range(num_samples_to_analyze):
            sample = self.nusc.sample[i]
            sample_tokens.append(sample['token'])

            # LiDAR data
            lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            lidar_file = os.path.join(self.dataroot, lidar_data['filename'])

            if os.path.exists(lidar_file):
                try:
                    pc = LidarPointCloud.from_file(lidar_file)
                    lidar_counts.append(pc.points.shape[1])  # points.shape is (4, N)
                except Exception as e:
                    if self.verbose:
                        logger.warning("Could not load %s: %s", lidar_file, e)
            else:
                if self.verbose:
                    logger.warning("LiDAR file not found: %s", lidar_file)

        return {
            "num_scenes": len(self.nusc.scene),
            "num_samples": len(self.nusc.sample),
            "sample_tokens": sample_tokens,
            "lidar_point_count": {
                "mean": float(np.mean(lidar_counts)) if lidar_counts else 0,
                "min": float(np.min(lidar_counts)) if lidar_counts else 0,
                "max": float(np.max(lidar_counts)) if lidar_counts else 0,
                "std": float(np.std(lidar_counts)) if lidar_counts else 0,
            },
        }

    def load_sample(self, sample_token: str) -> NuScenesPanopticSample:
        """Load a single sample from the dataset.

        Args:
            sample_token: Sample token

        Returns:
            NuScenesPanopticSample with loaded data
        """
        if not self._loaded:
            self.load()
        if self.nusc is None:
            raise RuntimeError(
                f"nuScenes dataset is not loaded (dataroot={self.dataroot})"
            )

        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])

        nusc_sample = NuScenesPanopticSample(
            token=sample_token,
            timestamp=sample['timestamp'],
            scene_name=scene['name'],
        )

        # Load LiDAR data
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_path = os.path.join(self.dataroot, lidar_data['filename'])

        if os.path.exists(lidar_path):
            try:
                pc = LidarPointCloud.from_file(lidar_path)
                # Transpose to (N, 4) - x, y, z, intensity
                nusc_sample.lidar_points = pc.points.T
                nusc_sample.num_lidar_points = pc.points.shape[1]
                nusc_sample.lidar_file_path = lidar_path
            except Exception as e:
                if self.verbose:
                    logger.warning("Could not load LiDAR %s: %s", lidar_path, e)

        return nusc_sample

    def iterate_samples(self, max_samples: Optional[int] = None):
        """Iterate through samples in the dataset.

        Args:
            max_samples: Maximum number of samples to yield

        Yields:
            NuScenesPanopticSample objects
        """
        if not self._loaded:
            self.load()
        if self.nusc is None:
            raise RuntimeError(
                f"nuScenes dataset is not loaded (dataroot={self.dataroot})"
            )

        count = 0
        for sample in self.nusc.sample:
            if max_samples and count >= max_samples:
                break

            yield self.load_sample(sample['token'])
            count += 1

    def voxelize_pointpillars(self, points: np.ndarray) -> ProcessedBatch:
        """Voxelize point cloud for PointPillars model.

        Args:
            points: (N, 4+) array of lidar points

        Returns:
            ProcessedBatch with voxelized data
        """
        cfg = self.POINTPILLARS_CONFIG

        # Filter points by range
        pc_range = cfg["point_cloud_range"]
        mask = (
            (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
            (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) &
            (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
        )
        points = points[mask]

        # Convert to voxel coordinates
        voxel_size = cfg["voxel_size"]
        pc_range = cfg["point_cloud_range"]

        # Compute voxel indices
        voxel_coords = np.floor(
            (points[:, :3] - np.array(pc_range[:3])) / np.array(voxel_size)
        ).astype(np.int32)

        # Create voxels
        max_num_points = cfg["max_num_points"]
        max_voxels = cfg["max_voxels"][1]

        # Initialize voxel arrays
        voxel_features = np.zeros((max_voxels, max_num_points, 9), dtype=np.float32)  # x, y, z, r, xc, yc, zc, xp, yp
        voxel_coords_array = np.zeros((max_voxels, 3), dtype=np.int32)
        num_points_per_voxel = np.zeros((max_voxels,), dtype=np.int32)

        # Group points by voxel
        unique_coords, inverse_indices = np.unique(
            voxel_coords, axis=0, return_inverse=True
        )

        num_voxels = min(len(unique_coords), max_voxels)

        for i, coord in enumerate(unique_coords[:num_voxels]):
            mask = inverse_indices == i
            voxel_points = points[mask]

            # Limit points per voxel
            if len(voxel_points) > max_num_points:
                voxel_points = voxel_points[:max_num_points]

            num_points = len(voxel_points)

            # Pillar feature augmentation (from PointPillars paper)
            # Features: [x, y, z, r, xc, yc, zc, xp, yp]
            # where (xc, yc, zc) is center of pillar, (xp, yp) is offset from center

            # Center of pillar
            center = np.mean(voxel_points[:, :3], axis=0)

            # Augmented features
            features = np.zeros((num_points, 9), dtype=np.float32)
            features[:, :4] = voxel_points[:, :4]  # x, y, z, intensity
            features[:, 4:7] = center  # xc, yc, zc
            features[:, 7:9] = voxel_points[:, :2] - center[:2]  # xp, yp

            voxel_features[i, :num_points] = features
            voxel_coords_array[i] = coord
            num_points_per_voxel[i] = num_points

        # Trim to actual number of voxels
        voxel_features = voxel_features[:num_voxels]
        voxel_coords_array = voxel_coords_array[:num_voxels]
        num_points_per_voxel = num_points_per_voxel[:num_voxels]

        return ProcessedBatch(
            voxel_features=torch.from_numpy(voxel_features).unsqueeze(0),  # Add batch dim
            coordinates=torch.from_numpy(voxel_coords_array).unsqueeze(0),
            num_points=torch.from_numpy(num_points_per_voxel).unsqueeze(0),
            batch_size=1,
            metadata={"num_voxels": num_voxels}
        )

    def voxelize_centerpoint(self, points: np.ndarray) -> ProcessedBatch:
        """Voxelize point cloud for CenterPoint model.

        Args:
            points: (N, 4+) array of lidar points

        Returns:
            ProcessedBatch with voxelized data
        """
        cfg = self.CENTERPOINT_CONFIG

        # Filter points by range
        pc_range = cfg["point_cloud_range"]
        mask = (
            (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
            (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) &
            (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
        )
        points = points[mask]

        # Convert to voxel coordinates
        voxel_size = cfg["voxel_size"]

        voxel_coords = np.floor(
            (points[:, :3] - np.array(pc_range[:3])) / np.array(voxel_size)
        ).astype(np.int32)

        # Create sparse voxels
        max_num_points = cfg["max_num_points"]
        max_voxels = cfg["max_voxels"][1]

        # Initialize voxel arrays (5 features: x, y, z, intensity, timestamp)
        voxel_features = np.zeros((max_voxels, max_num_points, 5), dtype=np.float32)
        voxel_coords_array = np.zeros((max_voxels, 3), dtype=np.int32)
        num_points_per_voxel = np.zeros((max_voxels,), dtype=np.int32)

        # Group points by voxel
        unique_coords, inverse_indices = np.unique(
            voxel_coords, axis=0, return_inverse=True
        )

        num_voxels = min(len(unique_coords), max_voxels)

        for i, coord in enumerate(unique_coords[:num_voxels]):
            mask = inverse_indices == i
            voxel_points = points[mask]

            num_points = min(len(voxel_points), max_num_points)
            voxel_features[i, :num_points, :4] = voxel_points[:num_points, :4]
            voxel_coords_array[i] = coord
            num_points_per_voxel[i] = num_points

        # Trim to actual number of voxels
        voxel_features = voxel_features[:num_voxels]
        voxel_coords_array = voxel_coords_array[:num_voxels]
        num_points_per_voxel = num_points_per_voxel[:num_voxels]

        return ProcessedBatch(
            voxel_features=torch.from_numpy(voxel_features).unsqueeze(0),
            coordinates=torch.from_numpy(voxel_coords_array).unsqueeze(0),
            num_points=torch.from_numpy(num_points_per_voxel).unsqueeze(0),
            batch_size=1,
            metadata={"num_voxels": num_voxels}
        )

    def get_model_input(self, model_type: str, max_samples: int = 10) -> List[ProcessedBatch]:
        """Get processed input batches for a specific model type.

        Args:
            model_type: One of 'pointpillars' or 'centerpoint'
            max_samples: Maximum number of samples to process

        Returns:
            List of ProcessedBatch objects
        """
        if not self._loaded:
            self.load()
        if self.nusc is None:
            raise RuntimeError(
                f"nuScenes dataset is not loaded (dataroot={self.dataroot})"
            )

        batches = []

        for i, sample in enumerate(self.nusc.sample[:max_samples]):
            nusc_sample = self.load_sample(sample['token'])

            if nusc_sample.lidar_points is None:
                continue

            if model_type.lower() == "pointpillars":
                batch = self.voxelize_pointpillars(nusc_sample.lidar_points)
            elif model_type.lower() == "centerpoint":
                batch = self.voxelize_centerpoint(nusc_sample.lidar_points)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            batch.metadata["sample_token"] = sample['token']
            batch.metadata["num_lidar_points"] = nusc_sample.num_lidar_points
            batches.append(batch)

        return batches


# Repository root, derived from this file's location:
# src/edgeslide/dataset/<this>.py
REPO_ROOT = Path(__file__).resolve().parents[3]

# Overrides the search base without touching code.
PANOPTIC_DATAROOT_ENV = "NUSCENES_PANOPTIC_DATAROOT"


def find_nuscenes_dataset(base_path: Optional[str] = None) -> Optional[str]:
    """Find the nuScenes-panoptic dataset directory.

    Args:
        base_path: Base path to search from. Defaults to
            ``$NUSCENES_PANOPTIC_DATAROOT`` and then to the repository root.
            The previous default was an absolute path under one developer's
            home directory, which resolved to nothing anywhere else and made
            the failure look like a missing dataset rather than a wrong base.

    Returns:
        Path to the nuScenes dataset, or None if not found.
    """
    if base_path is not None:
        base = Path(base_path)
    else:
        env_base = os.environ.get(PANOPTIC_DATAROOT_ENV)
        base = Path(env_base) if env_base else REPO_ROOT

    # Common locations (panoptic nests a v1.0-mini directory inside)
    search_paths = [
        base / "nuScenes-panoptic-v1.0-mini" / "v1.0-mini",
        base / "nuScenes-panoptic-v1.0-mini" / "nuScenes-panoptic-v1.0-mini" / "v1.0-mini",
        base / "data" / "nuScenes-panoptic-v1.0-mini" / "v1.0-mini",
        base / "data" / "v1.0-mini",
        base / "data" / "nuscenes",
    ]

    for path in search_paths:
        if path.exists() and (path / "category.json").exists():
            return str(path)

    logger.warning(
        "No nuScenes-panoptic dataset found under %s; set $%s to its location",
        base,
        PANOPTIC_DATAROOT_ENV,
    )
    return None


def _demo() -> None:
    """Manual smoke run against a real dataset. Not collected by pytest.

    Renamed from ``test_nuscenes_panoptic_loader``: the ``test_`` prefix made
    pytest collect this as a test case, where it either errored on the missing
    dataset or ran a multi-second voxelization sweep inside the unit suite.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataroot = find_nuscenes_dataset()
    if dataroot is None:
        logger.error(
            "Could not find the nuScenes-panoptic dataset; pass a path or set $%s",
            PANOPTIC_DATAROOT_ENV,
        )
        return

    logger.info("Found dataset at: %s", dataroot)

    loader = NuScenesPanopticLoader(dataroot=dataroot, verbose=True)
    if not loader.load():
        logger.error("Failed to load dataset")
        return

    logger.info("Dataset statistics:")
    for key, value in loader.get_statistics().items():
        if key != "sample_tokens":
            logger.info("  %s: %s", key, value)

    logger.info("Voxelization smoke test:")
    for index, sample in enumerate(loader.iterate_samples(max_samples=3), start=1):
        logger.info(
            "  sample %d: scene=%s lidar_points=%s",
            index,
            sample.scene_name,
            sample.num_lidar_points,
        )
        if sample.lidar_points is None:
            continue

        for label, voxelize in (
            ("PointPillars", loader.voxelize_pointpillars),
            ("CenterPoint", loader.voxelize_centerpoint),
        ):
            try:
                batch = voxelize(sample.lidar_points)
            except Exception as exc:
                logger.warning("    %s voxelization failed: %s", label, exc)
                continue
            logger.info(
                "    %s voxels=%s features=%s",
                label,
                batch.metadata["num_voxels"],
                batch.voxel_features.shape,
            )


if __name__ == "__main__":
    _demo()
