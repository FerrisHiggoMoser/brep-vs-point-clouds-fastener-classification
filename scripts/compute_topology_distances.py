"""Compute face topology distance matrices from STEP files for BRepFormer training.

Usage:
    python scripts/compute_topology_distances.py \
        --input_dir data/fastener_library/ \
        --output_dir data/fasteners_brep/
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def process_step(step_path: Path, output_dir: Path):
    """Extract BRepFormer features from a single STEP file."""
    from OCC.Core.STEPControl import STEPControl_Reader

    from step_vr_step.models.brepformer.feature_extractor import (
        extract_face_uv_grids,
        extract_edge_curves,
        compute_topology_distances,
    )

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != 1:
        raise ValueError(f"Failed to read {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    face_grids = extract_face_uv_grids(shape)
    edge_curves = extract_edge_curves(shape)
    topo_dists = compute_topology_distances(shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "face_grids.npy", face_grids)
    np.save(output_dir / "edge_curves.npy", edge_curves)
    np.savez(
        output_dir / "topo_distances.npz",
        **topo_dists,
    )

    logger.info(
        f"Processed {step_path.name}: "
        f"{face_grids.shape[0]} faces, {edge_curves.shape[0]} edges"
    )


def main():
    parser = argparse.ArgumentParser(description="Compute B-Rep topology distances")
    parser.add_argument("--input_dir", required=True, help="Directory with STEP files")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    step_extensions = {".step", ".stp", ".STEP", ".STP"}

    for step_file in sorted(input_dir.rglob("*")):
        if step_file.suffix not in step_extensions:
            continue

        rel_path = step_file.relative_to(input_dir)
        out_path = output_dir / rel_path.with_suffix("")

        try:
            process_step(step_file, out_path)
        except Exception as e:
            logger.error(f"Failed to process {step_file}: {e}")


if __name__ == "__main__":
    main()
