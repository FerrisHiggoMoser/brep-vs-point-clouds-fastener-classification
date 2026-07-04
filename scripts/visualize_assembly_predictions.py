"""Re-decompose a STEP assembly and export a colored .glb where each solid is colored
by its prediction from classify_assembly.py's predictions.csv.

Fasteners -> red. Non-fasteners -> light grey. Missing predictions -> blue (shouldn't happen).

Usage:
    python backend/scripts/visualize_assembly_predictions.py \
        --step cad-bidirectional-poc/data/input/isispace_1uplt_type_b_2023-04-20.stp \
        --predictions logs/pointnet2_finetune/isispace/predictions.csv \
        --output logs/pointnet2_finetune/isispace/classified.glb
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import trimesh
from tqdm import tqdm


def extract_solids_as_meshes(step_path: Path) -> list[trimesh.Trimesh | None]:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods

    print(f"Loading STEP (238 MB, this takes 1-3 minutes)...")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != 1:
        raise ValueError("STEP read failed")
    reader.TransferRoots()
    shape = reader.OneShape()

    solids = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        solids.append(topods.Solid(exp.Current()))
        exp.Next()
    print(f"Found {len(solids)} solids")

    meshes: list[trimesh.Trimesh | None] = []
    for solid in tqdm(solids, desc="Tessellating"):
        try:
            BRepMesh_IncrementalMesh(solid, 0.1, False, 0.5, True).Perform()
            verts: list[np.ndarray] = []
            faces: list[np.ndarray] = []
            v_offset = 0
            fexp = TopExp_Explorer(solid, TopAbs_FACE)
            while fexp.More():
                face = topods.Face(fexp.Current())
                loc = TopLoc_Location()
                tri = BRep_Tool.Triangulation(face, loc)
                if tri is not None:
                    t = loc.Transformation()
                    n_nodes = tri.NbNodes()
                    fv = np.empty((n_nodes, 3), dtype=np.float64)
                    for k in range(1, n_nodes + 1):
                        p = tri.Node(k).Transformed(t)
                        fv[k - 1] = (p.X(), p.Y(), p.Z())
                    n_tris = tri.NbTriangles()
                    ff = np.empty((n_tris, 3), dtype=np.int64)
                    for k in range(1, n_tris + 1):
                        a, b, c = tri.Triangle(k).Get()
                        ff[k - 1] = (a - 1 + v_offset, b - 1 + v_offset, c - 1 + v_offset)
                    verts.append(fv)
                    faces.append(ff)
                    v_offset += n_nodes
                fexp.Next()
            if verts:
                meshes.append(trimesh.Trimesh(vertices=np.vstack(verts), faces=np.vstack(faces), process=False))
            else:
                meshes.append(None)
        except Exception:
            meshes.append(None)
    return meshes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", required=True, type=Path)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    predictions: dict[int, dict] = {}
    with args.predictions.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            predictions[int(row["solid_idx"])] = row

    meshes = extract_solids_as_meshes(args.step)

    FASTENER_COLOR = np.array([220, 40, 40, 255], dtype=np.uint8)       # red
    NON_FASTENER_COLOR = np.array([180, 180, 180, 120], dtype=np.uint8)  # translucent grey
    MISSING_COLOR = np.array([40, 80, 220, 255], dtype=np.uint8)        # blue fallback

    scene = trimesh.Scene()
    n_fast = n_non = n_missing = 0
    for idx, mesh in enumerate(meshes):
        if mesh is None or len(mesh.faces) == 0:
            continue
        pred = predictions.get(idx)
        if pred is None:
            color = MISSING_COLOR
            n_missing += 1
        elif pred["prediction"] == "fastener":
            color = FASTENER_COLOR
            n_fast += 1
        else:
            color = NON_FASTENER_COLOR
            n_non += 1
        mesh.visual.face_colors = np.tile(color, (len(mesh.faces), 1))
        scene.add_geometry(mesh, node_name=f"solid_{idx}_{pred['prediction'] if pred else 'missing'}")

    print(f"\nScene composition: {n_fast} fasteners (red), {n_non} non-fasteners (grey), {n_missing} missing predictions (blue)")
    print(f"Exporting {args.output}...")
    scene.export(args.output)
    print(f"Done. Open in VS Code glTF viewer, Blender, or https://gltf-viewer.donmccurdy.com/")


if __name__ == "__main__":
    main()
