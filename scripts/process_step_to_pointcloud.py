"""Batch convert STEP files to point clouds for PointNet++ training.

Usage:
    python scripts/process_step_to_pointcloud.py \
        --input_dir data/fastener_library/ \
        --output_dir data/fasteners_pointcloud/ \
        --num_points 10000
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def step_to_pointcloud(step_path: Path, num_points: int = 10000) -> np.ndarray:
    """Convert a STEP file to a point cloud with normals.

    Returns: (num_points, 6) array of [x, y, z, nx, ny, nz].
    """
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != 1:
        raise ValueError(f"Failed to read STEP file: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    # Tessellate
    mesh = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)
    mesh.Perform()

    all_verts = []
    all_normals = []

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)

        if tri is not None:
            for i in range(1, tri.NbNodes() + 1):
                pnt = tri.Node(i)
                if not loc.IsIdentity():
                    pnt = pnt.Transformed(loc.Transformation())
                all_verts.append([pnt.X(), pnt.Y(), pnt.Z()])

                if tri.HasNormals():
                    n = tri.Normal(i)
                    all_normals.append([n.X(), n.Y(), n.Z()])
                else:
                    all_normals.append([0, 0, 1])

        exp.Next()

    if not all_verts:
        raise ValueError(f"No geometry found in {step_path}")

    verts = np.array(all_verts, dtype=np.float32)
    normals = np.array(all_normals, dtype=np.float32)

    # Subsample or oversample to num_points
    n = len(verts)
    if n >= num_points:
        choice = np.random.choice(n, num_points, replace=False)
    else:
        choice = np.random.choice(n, num_points, replace=True)

    points = np.concatenate([verts[choice], normals[choice]], axis=1)
    return points


def main():
    parser = argparse.ArgumentParser(description="Convert STEP files to point clouds")
    parser.add_argument("--input_dir", required=True, help="Directory with STEP files (class subdirs)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--num_points", type=int, default=10000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    step_extensions = {".step", ".stp", ".STEP", ".STP"}

    for step_file in sorted(input_dir.rglob("*")):
        if step_file.suffix not in step_extensions:
            continue

        rel_path = step_file.relative_to(input_dir)
        out_path = output_dir / rel_path.with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            points = step_to_pointcloud(step_file, args.num_points)
            np.savetxt(out_path, points, delimiter=",", fmt="%.6f")
            logger.info(f"Converted {step_file.name} -> {out_path} ({len(points)} points)")
        except Exception as e:
            logger.error(f"Failed to convert {step_file}: {e}")


if __name__ == "__main__":
    main()
