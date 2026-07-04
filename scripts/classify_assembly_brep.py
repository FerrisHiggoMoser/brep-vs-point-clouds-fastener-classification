"""BRepFormer version of classify_assembly.py.

Decomposes a STEP assembly into solids, extracts B-Rep features per solid, runs BRepFormer
inference, and exports glb visualizations matching the PointNet++ output format.

Usage:
    python backend/scripts/classify_assembly_brep.py \
        --step cad-bidirectional-poc/data/input/isispace_1uplt_type_b_2023-04-20.stp \
        --checkpoint logs/brepformer_finetune/best_model.pth \
        --output_dir logs/brepformer_finetune/isispace
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm

from step_vr_step.models.brepformer.brepformer import BRepFormer
from step_vr_step.models.brepformer.feature_extractor import (
    extract_face_uv_grids, extract_edge_curves, compute_topology_distances,
)

# Reuse the classify_assembly helpers for decomposition, mesh extraction, naming, colors
from scripts.classify_assembly import (
    load_assembly_with_names, shape_to_mesh, apply_color, name_based_label,
)

MAX_FACES = 600  # same cap as preprocessing — parts above this can't be batched sensibly


def extract_brep_features(shape, num_points_unused: int = 0):
    """Extract BRepFormer features from a TopoDS_Shape solid."""
    face_grids = extract_face_uv_grids(shape)
    if face_grids.shape[0] == 0 or face_grids.shape[0] > MAX_FACES:
        return None
    edge_curves = extract_edge_curves(shape)
    topo = compute_topology_distances(shape)
    return {
        "face_grids": face_grids.astype(np.float32),
        "edge_curves": edge_curves.astype(np.float32),
        "topo": {k: v.astype(np.float32) for k, v in topo.items()},
    }


def pad_and_batch(items: list[dict], device) -> dict:
    """Pad a list of per-part feature dicts into a single batched tensor dict."""
    max_nf = max(x["face_grids"].shape[0] for x in items)
    max_ne = max(x["edge_curves"].shape[0] for x in items)
    B = len(items)

    fg = torch.zeros(B, max_nf, 10, 10, 7)
    ec = torch.zeros(B, max_ne, 10, 12)
    fm = torch.zeros(B, max_nf, dtype=torch.bool)
    em = torch.zeros(B, max_ne, dtype=torch.bool)
    topo = {k: torch.zeros(B, max_nf, max_nf) for k in ("face_shortest", "face_centroid", "face_angular", "edge_path")}

    for i, x in enumerate(items):
        nf = x["face_grids"].shape[0]
        ne = x["edge_curves"].shape[0]
        fg[i, :nf] = torch.from_numpy(x["face_grids"])
        ec[i, :ne] = torch.from_numpy(x["edge_curves"])
        fm[i, :nf] = True
        em[i, :ne] = True
        for k in topo:
            if k in x["topo"]:
                topo[k][i, :nf, :nf] = torch.from_numpy(x["topo"][k])
    return {
        "face_grids": fg.to(device),
        "edge_curves": ec.to(device),
        "face_mask": fm.to(device),
        "edge_mask": em.to(device),
        "topo": {k: v.to(device) for k, v in topo.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--no_glb", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load BRepFormer
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt.get("class_names", ["fastener", "non-fastener"])
    num_classes = ckpt.get("num_classes", 2)
    print(f"Loading BRepFormer: classes={class_names}  num_classes={num_classes}")
    model = BRepFormer(num_classes=num_classes, dim=256, num_layers=8, head_mode="classification").to(device)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()

    # Decompose assembly (uses classify_assembly helpers — same as PointNet++ run)
    leaves = load_assembly_with_names(args.step)

    # Extract BRep features + cache meshes for glb
    print("Extracting BRep features per solid...")
    feats_list: list[tuple[str, object, dict]] = []
    skipped = 0
    for name, solid in tqdm(leaves):
        feat = extract_brep_features(solid)
        if feat is None:
            skipped += 1
            continue
        mesh = shape_to_mesh(solid)
        if mesh is None or len(mesh.faces) == 0:
            skipped += 1
            continue
        feats_list.append((name, mesh, feat))
    print(f"Usable parts: {len(feats_list)} (skipped {skipped}: too-complex or untessellable)")

    # Run inference in batches
    rows: list[dict] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(feats_list), args.batch_size), desc="Classifying"):
            batch_items = feats_list[start:start + args.batch_size]
            batch = pad_and_batch([f for _, _, f in batch_items], device)
            logits = model(batch["face_grids"], batch["edge_curves"], batch["topo"],
                           mask=batch["face_mask"], edge_mask=batch["edge_mask"])
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)
            for (name, mesh, _), pred_idx, prob in zip(batch_items, preds, probs):
                truth = name_based_label(name)
                if truth == "unknown":
                    status = "UNK"
                elif truth == "fastener" and class_names[pred_idx] == "fastener":
                    status = "TP"
                elif truth == "non-fastener" and class_names[pred_idx] == "non-fastener":
                    status = "TN"
                elif truth == "non-fastener" and class_names[pred_idx] == "fastener":
                    status = "FP"
                else:
                    status = "FN"
                rows.append({
                    "part_name": name,
                    "n_faces": len(mesh.faces),
                    "prediction": class_names[pred_idx],
                    "p_fastener": float(prob[0]),
                    "p_non_fastener": float(prob[1]),
                    "name_based_truth": truth,
                    "status": status,
                    "_mesh": mesh,  # temporary, for glb export
                })

    # Build glbs
    if not args.no_glb:
        scene = trimesh.Scene()
        fast_scene = trimesh.Scene()
        truth_scene = trimesh.Scene()
        color_by_status = {
            "TP": (60, 200, 60),
            "FP": (255, 140, 40),
            "FN": (140, 60, 200),
            "TN": (180, 180, 180),
            "UNK": (80, 140, 220),
        }
        for r in rows:
            mesh = r.pop("_mesh")
            pred = r["prediction"]
            safe = f'{r["part_name"].replace("/", "__").replace(" ", "_")[:180]}__{pred}__{r["status"]}'
            m1 = mesh.copy()
            apply_color(m1, (220, 40, 40) if pred == "fastener" else (180, 180, 180),
                        255 if pred == "fastener" else 100)
            m1.metadata["name"] = safe
            scene.add_geometry(m1, node_name=safe, geom_name=safe)
            if pred == "fastener":
                m2 = mesh.copy()
                apply_color(m2, (220, 40, 40), 255)
                m2.metadata["name"] = safe
                fast_scene.add_geometry(m2, node_name=safe, geom_name=safe)
            m3 = mesh.copy()
            apply_color(m3, color_by_status[r["status"]], 255 if r["status"] in ("TP", "FP", "FN") else 80)
            m3.metadata["name"] = safe
            truth_scene.add_geometry(m3, node_name=safe, geom_name=safe)
        print(f"Exporting classified.glb, fasteners_only.glb, ground_truth_check.glb ...")
        scene.export(args.output_dir / "classified.glb")
        if len(fast_scene.geometry) > 0:
            fast_scene.export(args.output_dir / "fasteners_only.glb")
        truth_scene.export(args.output_dir / "ground_truth_check.glb")
    else:
        for r in rows:
            r.pop("_mesh", None)

    # Save CSV
    csv_path = args.output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0].keys() if k != "_mesh"])
        w.writeheader()
        w.writerows(rows)

    # Summary
    tp = sum(1 for r in rows if r["status"] == "TP")
    fp = sum(1 for r in rows if r["status"] == "FP")
    fn = sum(1 for r in rows if r["status"] == "FN")
    tn = sum(1 for r in rows if r["status"] == "TN")
    unk = sum(1 for r in rows if r["status"] == "UNK")
    n_fast = sum(1 for r in rows if r["prediction"] == "fastener")
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    print(f"\n=== Assembly classification summary (BRepFormer) ===")
    print(f"Parts classified: {len(rows)}  (skipped: {skipped})")
    print(f"Predicted fasteners: {n_fast} ({n_fast/len(rows)*100:.1f}%)")
    print(f"\n=== Validation vs name-based ground truth ===")
    print(f"Graded parts: {tp+fp+fn+tn} (unknown/unnamed: {unk})")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Fastener precision: {prec:.3f}")
    print(f"  Fastener recall:    {rec:.3f}")
    print(f"  Fastener F1:        {f1:.3f}")

    print(f"\nTop 15 confident fastener predictions (w/ status):")
    for r in sorted([r for r in rows if r["prediction"] == "fastener"], key=lambda r: -r["p_fastener"])[:15]:
        print(f"  [{r['status']:3s}] p={r['p_fastener']:.3f}  truth={r['name_based_truth']:14s}  {r['part_name'][:100]}")


if __name__ == "__main__":
    main()
