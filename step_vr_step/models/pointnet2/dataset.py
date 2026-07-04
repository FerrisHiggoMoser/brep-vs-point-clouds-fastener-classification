"""Dataset loader for PointNet++ fastener classification.

Supports:
  - ModelNet40 format (.txt files with N lines of x y z nx ny nz)
  - NumPy format (.npy files with Nx6 arrays)
  - Configurable number of points and augmentation
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FastenerPointCloudDataset(Dataset):
    """Point cloud dataset for fastener classification.

    Directory structure (ModelNet-style):
        root/
          class_0/
            model_001.txt
            model_002.npy
          class_1/
            ...
    """

    def __init__(
        self,
        root: str | Path,
        num_points: int = 2048,
        use_normals: bool = True,
        split: str = "train",
        train_ratio: float = 0.8,
        augment: bool = True,
    ):
        self.root = Path(root)
        self.num_points = num_points
        self.use_normals = use_normals
        self.augment = augment and (split == "train")

        # Check if dataset uses split directories (train/val/test)
        split_dir = self.root / split
        if split_dir.exists():
            # New format: root/train/class_name/*.npy
            data_root = split_dir
            use_ratio_split = False
        else:
            # Legacy format: root/class_name/*.npy (split by ratio)
            data_root = self.root
            use_ratio_split = True

        # Discover classes from directory names
        self.classes = sorted([
            d.name for d in data_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Collect all samples
        self.samples: list[tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_dir = data_root / cls_name
            files = sorted([
                f for f in cls_dir.iterdir()
                if f.suffix in (".txt", ".npy") and not f.name.startswith(".")
            ])
            cls_idx = self.class_to_idx[cls_name]
            for f in files:
                self.samples.append((f, cls_idx))

        # Legacy: train/val split by ratio
        if use_ratio_split:
            n = len(self.samples)
            split_idx = int(n * train_ratio)
            if split == "train":
                self.samples = self.samples[:split_idx]
            else:
                self.samples = self.samples[split_idx:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]

        # Load point cloud
        if filepath.suffix == ".npy":
            data = np.load(filepath)
        else:
            data = np.loadtxt(filepath, delimiter=",").astype(np.float32)

        # Ensure at least 6 columns (xyz + normals)
        if data.shape[1] < 6:
            normals = np.zeros((data.shape[0], 3), dtype=np.float32)
            data = np.concatenate([data[:, :3], normals], axis=1)

        # Subsample or pad to num_points
        if len(data) >= self.num_points:
            choice = np.random.choice(len(data), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(data), self.num_points, replace=True)
        data = data[choice]

        points = data[:, :3]
        normals = data[:, 3:6]

        # Normalize to unit sphere
        centroid = points.mean(axis=0)
        points = points - centroid
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 1e-12:
            points = points / scale

        # Augmentation
        if self.augment:
            points, normals = self._augment(points, normals)

        if self.use_normals:
            features = np.concatenate([points, normals], axis=1)  # (N, 6)
        else:
            features = points  # (N, 3)

        return torch.from_numpy(features.astype(np.float32)), label

    @staticmethod
    def _augment(
        points: np.ndarray, normals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random augmentation: rotation, jitter, scale."""
        # Random rotation around Y axis
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([
            [cos_t,  0, sin_t],
            [0,      1, 0],
            [-sin_t, 0, cos_t],
        ], dtype=np.float32)
        points = points @ rot.T
        normals = normals @ rot.T

        # Random jitter
        jitter = np.clip(np.random.normal(0, 0.01, size=points.shape), -0.05, 0.05)
        points = points + jitter.astype(np.float32)

        # Random scale
        scale = np.random.uniform(0.8, 1.2)
        points = points * scale

        return points, normals

    @property
    def num_classes(self) -> int:
        return len(self.classes)
