"""Decompose a STEP assembly preserving part names, classify each part with Stage 2 PointNet++,
and export a colored .glb where fasteners are red and non-fasteners are translucent grey.

Uses STEPCAFControl_Reader (XDE) to preserve product names from the STEP assembly.

Usage:
    python backend/scripts/classify_assembly.py \
        --step cad-bidirectional-poc/data/input/isispace_1uplt_type_b_2023-04-20.stp \
        --checkpoint logs/pointnet2_finetune/best_model.pth \
        --output_dir logs/pointnet2_finetune/isispace
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm

from step_vr_step.models.pointnet2.pointnet2_cls_msg import PointNet2ClsMSG


def load_assembly_with_names(step_path: Path):
    """Walk every placed solid via STEPControl_Reader (correct world transforms, ~640 real parts).
    Separately use pythonocc's DataExchange helper to get named solid prototypes (~1013 including
    duplicates and markers). Match by geometric SIGNATURE (face count + bbox dimensions rounded to
    0.1mm) — identical parts share a signature regardless of which reader produced the TopoDS_Shape.
    This works around pythonocc 7.9's broken TShape-identity matching."""
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    from OCC.Extend.DataExchange import read_step_file_with_names_colors

    def count_faces(shape) -> int:
        n = 0
        e = TopExp_Explorer(shape, TopAbs_FACE)
        while e.More():
            n += 1
            e.Next()
        return n

    def signature(shape) -> tuple:
        """Geometric fingerprint: (n_faces, bbox_x, bbox_y, bbox_z) with bbox sides rounded."""
        bb = Bnd_Box()
        try:
            brepbndlib.Add(shape, bb)
            x1, y1, z1, x2, y2, z2 = bb.Get()
            dx = round(x2 - x1, 1)
            dy = round(y2 - y1, 1)
            dz = round(z2 - z1, 1)
            # Sort bbox sides so orientation doesn't break equality
            dims = tuple(sorted([dx, dy, dz]))
        except Exception:
            dims = (0.0, 0.0, 0.0)
        return (count_faces(shape),) + dims

    print(f"Loading STEP (this takes 1-3 min)...")
    shape_name_map = read_step_file_with_names_colors(str(step_path))

    # Build signature -> name lookup from helper's named shapes
    sig_to_name: dict[tuple, str] = {}
    for shape, info in shape_name_map.items():
        name = info[0] if info else ""
        if not name:
            continue
        e = TopExp_Explorer(shape, TopAbs_SOLID)
        while e.More():
            s = topods.Solid(e.Current())
            sig = signature(s)
            sig_to_name.setdefault(sig, name)
            e.Next()
    print(f"Built signature -> name lookup ({len(sig_to_name)} unique signatures)")

    # Read the whole assembly via STEPControl_Reader — these are the placed instances
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != 1:
        raise ValueError("STEP read failed")
    reader.TransferRoots()
    whole = reader.OneShape()

    from tqdm import tqdm as _tqdm
    all_instances = []
    e = TopExp_Explorer(whole, TopAbs_SOLID)
    while e.More():
        all_instances.append(topods.Solid(e.Current()))
        e.Next()
    print(f"Found {len(all_instances)} placed solid instances, matching by signature...")

    out: list[tuple[str, object]] = []
    name_counts: dict[str, int] = {}
    for idx, solid in enumerate(_tqdm(all_instances, desc="Matching names")):
        sig = signature(solid)
        base_name = sig_to_name.get(sig, f"unnamed_solid_{idx}")
        n = name_counts.get(base_name, 0)
        name_counts[base_name] = n + 1
        unique_name = f"{base_name}#{n}" if n > 0 else base_name
        out.append((unique_name, solid))
    named = sum(1 for n, _ in out if not n.startswith("unnamed_solid_"))
    print(f"Total solids: {len(out)} ({named} named, {len(out)-named} unnamed)")
    return out


def shape_to_mesh(shape) -> trimesh.Trimesh | None:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods

    try:
        BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True).Perform()
    except Exception:
        return None

    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    v_offset = 0
    fexp = TopExp_Explorer(shape, TopAbs_FACE)
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

    if not verts:
        return None
    return trimesh.Trimesh(vertices=np.vstack(verts), faces=np.vstack(faces), process=False)


def mesh_to_features(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    pts, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    normals = mesh.face_normals[face_idx]
    pts = pts.astype(np.float32)
    pts = pts - pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 1e-12:
        pts = pts / scale
    return np.concatenate([pts, normals.astype(np.float32)], axis=1)


def apply_color(mesh: trimesh.Trimesh, rgb: tuple[int, int, int], alpha: int) -> None:
    """Apply a PBR material to the mesh — this is what glb/gltf viewers actually render."""
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, alpha / 255.0],
        metallicFactor=0.2,
        roughnessFactor=0.7,
        alphaMode="BLEND" if alpha < 255 else "OPAQUE",
        doubleSided=True,
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=np.float32),
        material=material,
    )


# Keywords that, if present in a part name, indicate the part IS a fastener
FASTENER_KEYWORDS = (
    "bolt", "screw", "nut", "washer", "stud", "rivet",
    "fastener", "threaded", "dowel", "anchor", "clip", "circlip",
    # DIN standards (fastener families)
    "din125", "din126", "din127", "din128",                           # washers
    "din912", "din913", "din914", "din915", "din916",                 # socket screws
    "din931", "din933", "din934", "din935", "din936",                 # hex bolts/nuts
    "din7984", "din7985", "din7991", "din6912", "din6921",            # other screws
    "din985", "din980", "din982",                                      # lock nuts
    # ISO standards (fastener families)
    "iso4014", "iso4017", "iso4026", "iso4027", "iso4028", "iso4029", # hex / set screws
    "iso4032", "iso4033", "iso4034", "iso4035", "iso4036",            # nuts
    "iso4762",                                                         # socket cap screws
    "iso7089", "iso7090", "iso7091", "iso7092", "iso7093", "iso7094", # washers
    "iso7380", "iso7381",                                              # button head
    "iso10642", "iso14580", "iso14581", "iso14583", "iso14584",       # countersunk etc
    "iso7045", "iso7046", "iso7047",                                   # Phillips
    # Common CAD naming
    "hex_head", "hex head", "cap_screw", "cap screw", "lag_bolt",
    "socket_head", "pan_head", "flat_head", "button_head", "countersunk",
)
# A regex that matches metric-screw size codes embedded in part names.
# Catches: M3X12, M2.5X12, M2_5X12, M6X20, M10X50, etc.
_METRIC_SCREW_RE = re.compile(r"m\d+(?:[._]\d+)?x\d+", re.IGNORECASE)

# Tokens that override a fastener match (e.g. "mounting pin" is not a canonical fastener).
# Use with care — these only trigger when no explicit fastener keyword/regex matches.
NON_FASTENER_OVERRIDES = (
    "housing", "bracket_asm", "panel_asm", "assembly", "_asm", "_sub",
    "pcb_", "pcba_",
)


def name_based_label(name: str) -> str:
    """Return 'fastener', 'non-fastener', or 'unknown' based on keyword match on the part name."""
    if not name or name.startswith("unnamed_solid_"):
        return "unknown"
    lower = name.lower()
    # Uninformative names — can't judge
    if lower.startswith("compound"):
        return "unknown"
    has_fast_kw = any(kw in lower for kw in FASTENER_KEYWORDS)
    has_metric = bool(_METRIC_SCREW_RE.search(lower))
    has_override = any(tok in lower for tok in NON_FASTENER_OVERRIDES)
    if (has_fast_kw or has_metric) and not has_override:
        return "fastener"
    return "non-fastener"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--no_glb", action="store_true", help="Skip glb export")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    leaves = load_assembly_with_names(args.step)

    tessellated: list[tuple[str, trimesh.Trimesh]] = []
    for name, shape in tqdm(leaves, desc="Tessellating"):
        m = shape_to_mesh(shape)
        if m is not None and len(m.faces) > 0:
            tessellated.append((name, m))

    print(f"Tessellated {len(tessellated)} / {len(leaves)} parts")

    features = np.stack([mesh_to_features(m, args.num_points) for _, m in tqdm(tessellated, desc="Sampling")])

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    model = PointNet2ClsMSG(num_classes=ckpt["num_classes"], use_normals=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds = []
    probs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(features), args.batch_size), desc="Classifying"):
            batch = torch.from_numpy(features[start:start + args.batch_size]).to(device)
            logits, _ = model(batch.transpose(1, 2))
            p = F.softmax(logits, dim=-1).cpu().numpy()
            preds.extend(p.argmax(axis=1).tolist())
            probs.extend(p.tolist())

    rows: list[dict] = []
    scene = trimesh.Scene()
    fasteners_scene = trimesh.Scene()
    truth_scene = trimesh.Scene()

    # TP green, FP orange, FN purple, TN grey, unknown pale blue
    color_by_status = {
        "TP": (60, 200, 60),
        "FP": (255, 140, 40),
        "FN": (140, 60, 200),
        "TN": (180, 180, 180),
        "UNK": (80, 140, 220),
    }

    for (name, mesh), pred_idx, prob in zip(tessellated, preds, probs):
        pred = class_names[pred_idx]
        truth = name_based_label(name)
        if truth == "unknown":
            status = "UNK"
        elif truth == "fastener" and pred == "fastener":
            status = "TP"
        elif truth == "non-fastener" and pred == "non-fastener":
            status = "TN"
        elif truth == "non-fastener" and pred == "fastener":
            status = "FP"
        else:
            status = "FN"

        rows.append({
            "part_name": name,
            "n_faces": len(mesh.faces),
            "prediction": pred,
            "p_fastener": float(prob[0]),
            "p_non_fastener": float(prob[1]),
            "name_based_truth": truth,
            "status": status,
        })

        if not args.no_glb:
            safe_name = f'{name.replace("/", "__").replace(" ", "_")[:180]}__{pred}__{status}'

            # classified.glb: red fasteners, translucent grey non-fasteners
            m1 = mesh.copy()
            apply_color(m1, (220, 40, 40) if pred == "fastener" else (180, 180, 180),
                        255 if pred == "fastener" else 100)
            m1.metadata["name"] = safe_name
            scene.add_geometry(m1, node_name=safe_name, geom_name=safe_name)

            # fasteners_only.glb: only predicted fasteners, solid red
            if pred == "fastener":
                m2 = mesh.copy()
                apply_color(m2, (220, 40, 40), 255)
                m2.metadata["name"] = safe_name
                fasteners_scene.add_geometry(m2, node_name=safe_name, geom_name=safe_name)

            # ground_truth_check.glb: color by TP/FP/FN/TN/UNK vs name-based truth
            m3 = mesh.copy()
            apply_color(m3, color_by_status[status], 255 if status in ("TP","FP","FN") else 80)
            m3.metadata["name"] = safe_name
            truth_scene.add_geometry(m3, node_name=safe_name, geom_name=safe_name)

    csv_path = args.output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if not args.no_glb:
        print(f"Exporting classified.glb...")
        scene.export(args.output_dir / "classified.glb")
        if len(fasteners_scene.geometry) > 0:
            print(f"Exporting fasteners_only.glb ({len(fasteners_scene.geometry)} parts)...")
            fasteners_scene.export(args.output_dir / "fasteners_only.glb")
        print(f"Exporting ground_truth_check.glb...")
        truth_scene.export(args.output_dir / "ground_truth_check.glb")

    n_fast = sum(1 for r in rows if r["prediction"] == "fastener")
    print(f"\n=== Classification summary ===")
    print(f"Parts classified: {len(rows)}")
    print(f"Predicted fasteners: {n_fast} ({n_fast/len(rows)*100:.1f}%)")

    # Validation against name-based ground truth
    tp = sum(1 for r in rows if r["status"] == "TP")
    fp = sum(1 for r in rows if r["status"] == "FP")
    fn = sum(1 for r in rows if r["status"] == "FN")
    tn = sum(1 for r in rows if r["status"] == "TN")
    unk = sum(1 for r in rows if r["status"] == "UNK")
    graded = tp + fp + fn + tn

    print(f"\n=== Validation vs name-based ground truth ===")
    print(f"(Ground truth = keyword match on STEP part name: bolt/screw/nut/washer/...)")
    print(f"Parts with extractable name-based truth: {graded} / {len(rows)} (unknown/unnamed: {unk})")
    print(f"  True positive  (model=fast, name=fast): {tp}")
    print(f"  False positive (model=fast, name=non):  {fp}")
    print(f"  False negative (model=non,  name=fast): {fn}")
    print(f"  True negative  (model=non,  name=non):  {tn}")
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"  Fastener precision: {precision:.3f}")
    print(f"  Fastener recall:    {recall:.3f}")
    print(f"  Fastener F1:        {f1:.3f}")

    print(f"\nTop 15 confident fastener predictions (with ground-truth status):")
    for r in sorted([r for r in rows if r["prediction"] == "fastener"], key=lambda r: -r["p_fastener"])[:15]:
        print(f"  [{r['status']:3s}] p={r['p_fastener']:.3f}  truth={r['name_based_truth']:14s}  {r['part_name'][:100]}")

    if fn > 0:
        print(f"\nMissed fasteners (name says fastener, model said non-fastener):")
        for r in sorted([r for r in rows if r["status"] == "FN"], key=lambda r: -r["p_non_fastener"])[:15]:
            print(f"  p(non)={r['p_non_fastener']:.3f}  {r['part_name'][:120]}")

    print(f"\nCSV: {csv_path}")
    if not args.no_glb:
        print(f"glb files: classified.glb (all, red=fast), fasteners_only.glb (predicted fasteners), ground_truth_check.glb (TP=green FP=orange FN=purple TN=grey UNK=blue)")


if __name__ == "__main__":
    main()
