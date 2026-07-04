"""Dataset loader for BRepFormer training.

Handles variable-size B-Rep graphs by padding to the maximum number of
faces/edges in each batch. Each sample is a pre-processed directory
containing face UV-grids, edge curves, and topology distance matrices.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BRepDataset(Dataset):
    """B-Rep dataset for classification.

    Directory structure:
        root/
          class_0/
            model_001/
              face_grids.npy    # (Nf, 10, 10, 7)
              edge_curves.npy   # (Ne, 10, 12)
              topo_distances.npz  # face_shortest, face_centroid, face_angular, edge_path
            model_002/
              ...
          class_1/
            ...
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        train_ratio: float = 0.8,
    ):
        self.root = Path(root)

        # If root/<split>/ exists, use that (new format). Otherwise fall back to legacy
        # ratio-split mode on the root (for single-tree datasets).
        split_dir = self.root / split
        if split_dir.exists() and split_dir.is_dir():
            data_root = split_dir
            use_ratio_split = False
        else:
            data_root = self.root
            use_ratio_split = True

        # Detect layout: flat (no class subfolders, for segmentation) vs hierarchical
        # (root/split/class/model, for classification).
        immediate_dirs = [d for d in data_root.iterdir()
                         if d.is_dir() and not d.name.startswith(".")]
        flat_layout = (
            len(immediate_dirs) > 0 and
            all((d / "face_grids.npy").exists() for d in immediate_dirs[:5])
        )

        self.samples: list[tuple[Path, int]] = []

        if flat_layout:
            # Segmentation style: every immediate subdir is a model with face_labels.npy.
            # Assign a placeholder class label — real labels are per-face in face_labels.npy.
            self.classes = ["_flat"]
            self.class_to_idx = {"_flat": 0}
            for model_dir in sorted(immediate_dirs):
                if (model_dir / "face_grids.npy").exists():
                    self.samples.append((model_dir, 0))
        else:
            self.classes = sorted([d.name for d in immediate_dirs])
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            for cls_name in self.classes:
                cls_dir = data_root / cls_name
                models = sorted([
                    d for d in cls_dir.iterdir()
                    if d.is_dir() and (d / "face_grids.npy").exists()
                ])
                cls_idx = self.class_to_idx[cls_name]
                for model_dir in models:
                    self.samples.append((model_dir, cls_idx))

        # Legacy fallback: ratio split on a single tree
        if use_ratio_split:
            n = len(self.samples)
            split_idx = int(n * train_ratio)
            if split == "train":
                self.samples = self.samples[:split_idx]
            else:
                self.samples = self.samples[split_idx:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        model_dir, part_label = self.samples[idx]

        face_grids = np.load(model_dir / "face_grids.npy").astype(np.float32)
        edge_curves = np.load(model_dir / "edge_curves.npy").astype(np.float32)

        topo_data = np.load(model_dir / "topo_distances.npz")
        topo_distances = {
            key: topo_data[key].astype(np.float32)
            for key in ["face_shortest", "face_centroid", "face_angular", "edge_path"]
            if key in topo_data
        }

        # Per-face labels override part label if face_labels.npy exists (segmentation datasets)
        face_labels_path = model_dir / "face_labels.npy"
        if face_labels_path.exists():
            face_labels = np.load(face_labels_path).astype(np.int64)
            label = torch.from_numpy(face_labels)  # (Nf,)
        else:
            # Part-level label broadcast to all faces (for segmentation training on part-level data)
            label = part_label

        return {
            "face_grids": torch.from_numpy(face_grids),
            "edge_curves": torch.from_numpy(edge_curves),
            "topo_distances": {k: torch.from_numpy(v) for k, v in topo_distances.items()},
            "label": label,
            "num_faces": face_grids.shape[0],
            "num_edges": edge_curves.shape[0],
        }

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def brep_collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for variable-size B-Rep graphs.

    Pads face_grids, edge_curves, and topology distances to the maximum
    sizes in the batch and creates boolean masks.
    """
    max_faces = max(item["num_faces"] for item in batch)
    max_edges = max(item["num_edges"] for item in batch)
    B = len(batch)

    # Per-face labels mode if any item has a tensor label (segmentation datasets)
    is_seg = any(torch.is_tensor(item["label"]) for item in batch)

    face_grids = torch.zeros(B, max_faces, 10, 10, 7)
    edge_curves = torch.zeros(B, max_edges, 10, 12)
    face_mask = torch.zeros(B, max_faces, dtype=torch.bool)
    edge_mask = torch.zeros(B, max_edges, dtype=torch.bool)

    if is_seg:
        labels = torch.zeros(B, max_faces, dtype=torch.long)
    else:
        labels = torch.zeros(B, dtype=torch.long)

    topo_keys = ["face_shortest", "face_centroid", "face_angular", "edge_path"]
    topo_distances = {key: torch.zeros(B, max_faces, max_faces) for key in topo_keys}

    for i, item in enumerate(batch):
        nf = item["num_faces"]
        ne = item["num_edges"]

        face_grids[i, :nf] = item["face_grids"]
        edge_curves[i, :ne] = item["edge_curves"]
        face_mask[i, :nf] = True
        edge_mask[i, :ne] = True

        if is_seg:
            lbl = item["label"]
            if torch.is_tensor(lbl):
                labels[i, :nf] = lbl[:nf]
            else:
                # Broadcast part label to all faces
                labels[i, :nf] = int(lbl)
        else:
            labels[i] = item["label"]

        for key in topo_keys:
            if key in item["topo_distances"]:
                topo_distances[key][i, :nf, :nf] = item["topo_distances"][key]

    return {
        "face_grids": face_grids,
        "edge_curves": edge_curves,
        "topo_distances": topo_distances,
        "face_mask": face_mask,
        "edge_mask": edge_mask,
        "labels": labels,
    }
