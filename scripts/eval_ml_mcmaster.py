"""ML-only McMaster evaluation: load the trained BRepFormer checkpoint
directly, run on McMaster test split, report binary fastener vs non-
fastener accuracy. Goal: match the journal's val_acc 0.865 baseline.

Outputs:
  - Binary confusion matrix (collapse 13-class to fastener / non_fastener)
  - 13-class prediction histogram (so you can see WHICH subtype the
    model voted for on each correctly-classified fastener)
  - CSV with per-file ground-truth + prediction + confidence
"""
from __future__ import annotations

import sys
import time
import csv
from collections import Counter
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from step_vr_step.readers.step_reader import read_step
from step_vr_step.config import DetectionConfig
from step_vr_step.detection.ml_classifier import FastenerClassifier, is_ml_available
from step_vr_step.models.brepformer.feature_extractor import (
    extract_face_uv_grids, extract_edge_curves, compute_topology_distances,
)


# Class names in the order the bf_subtype13 model was trained.
BF_CLASS_NAMES = [
    "anchors", "keys", "nails", "non_fastener", "nuts", "pins",
    "retaining-rings", "rivets", "screws", "spacers",
    "threaded-inserts", "threaded-rods", "washers",
]


def predict_one(classifier: FastenerClassifier, shape) -> tuple[str, float]:
    """Run BRepFormer on a single TopoDS_Shape. Returns (class_name, confidence)."""
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    n_faces = 0
    while exp.More():
        n_faces += 1
        exp.Next()
    if n_faces == 0:
        return "non_fastener", 0.0
    if n_faces > 600:
        # Outside training distribution — return low-confidence non_fastener
        return "non_fastener", 0.0

    face_grids = extract_face_uv_grids(shape)
    edge_curves = extract_edge_curves(shape)
    topo = compute_topology_distances(shape)
    # Drop any None distance matrices (classifier zeros them anyway).
    topo = {k: v for k, v in topo.items() if v is not None}

    label = classifier.classify_brep(
        face_grids=face_grids,
        edge_grids=edge_curves,
        topo_distances=topo,
        class_names=BF_CLASS_NAMES,
    )
    # `label.fastener_type` may be 'foo' or 'likely_foo' depending on threshold.
    raw = label.fastener_type.replace("likely_", "").replace("possible_", "")
    return raw, float(label.confidence)


def evaluate(folder: Path, expected_fastener: bool, classifier, limit: int):
    files = sorted(folder.iterdir())[:limit]
    rows = []
    correct = wrong = errors = 0
    for f in files:
        if f.suffix.lower() not in (".step", ".stp"):
            continue
        try:
            doc, _, _ = read_step(str(f), return_shapes=True)
            # Largest solid wins (most McMaster files are single-part anyway)
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_SOLID
            from OCC.Core.BRepGProp import brepgprop_VolumeProperties
            from OCC.Core.GProp import GProp_GProps
            exp = TopExp_Explorer(doc, TopAbs_SOLID)
            solids = []
            while exp.More():
                solids.append(exp.Current())
                exp.Next()
            if not solids:
                errors += 1
                continue
            def _vol(s):
                g = GProp_GProps()
                brepgprop_VolumeProperties(s, g)
                return abs(g.Mass())
            shape = max(solids, key=_vol)
            cls, conf = predict_one(classifier, shape)
        except Exception:
            errors += 1
            continue

        # Binary collapse: any non_fastener prediction = non-fastener
        predicted_pos = (cls != "non_fastener" and cls != "unclassified")
        is_correct = (predicted_pos == expected_fastener)
        if is_correct:
            correct += 1
        else:
            wrong += 1
        rows.append({
            "file": f.stem,
            "gt": "fastener" if expected_fastener else "non_fastener",
            "pred_class": cls,
            "pred_binary": "fastener" if predicted_pos else "non_fastener",
            "confidence": round(conf, 3),
            "correct": int(is_correct),
        })
    return rows, correct, wrong, errors


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="Path to trained BRepFormer checkpoint.")
    p.add_argument("--limit", type=int, default=9999)
    p.add_argument("--root",
        default=str(_BACKEND_ROOT.parent / "training_data" / "mcmaster_binary" / "test"))
    p.add_argument("--out-csv", default="ml_mcmaster_predictions.csv")
    p.add_argument("--ml-threshold", type=float, default=0.0,
                   help="Override ML confidence threshold (0 = raw argmax).")
    args = p.parse_args()

    if not is_ml_available():
        print("ERROR: PyTorch not installed in this Python environment.", file=sys.stderr)
        return 2

    print(f"Loading BRepFormer from: {args.ckpt}", file=sys.stderr)
    cfg = DetectionConfig(
        enable_ml=True,
        brepformer_weights=args.ckpt,
        ml_confidence_threshold=args.ml_threshold,
    )
    classifier = FastenerClassifier(cfg)
    if classifier.brepformer_model is None:
        print("ERROR: BRepFormer failed to load.", file=sys.stderr)
        return 2

    root = Path(args.root)
    print(f"Evaluating: {root}  limit={args.limit}", file=sys.stderr)

    t0 = time.perf_counter()
    f_rows, f_corr, f_wrong, f_err = evaluate(root / "fastener", True, classifier, args.limit)
    print(f"fastener/    correct={f_corr}  wrong={f_wrong}  err={f_err}  "
          f"({time.perf_counter()-t0:.0f}s)", file=sys.stderr)

    t0 = time.perf_counter()
    nf_rows, nf_corr, nf_wrong, nf_err = evaluate(root / "non_fastener", False, classifier, args.limit)
    print(f"non_fastener/ correct={nf_corr}  wrong={nf_wrong}  err={nf_err}  "
          f"({time.perf_counter()-t0:.0f}s)", file=sys.stderr)

    # Binary confusion matrix
    TP = f_corr     # fastener correctly predicted fastener
    FN = f_wrong    # fastener predicted non_fastener
    TN = nf_corr    # non_fastener correctly predicted non_fastener
    FP = nf_wrong   # non_fastener predicted fastener
    tot = TP + FP + TN + FN
    acc = (TP + TN) / max(1, tot)
    prec = TP / max(1, TP + FP)
    rec = TP / max(1, TP + FN)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    print()
    print("=" * 70)
    print(f"McMaster test split — BRepFormer 13-class collapsed to binary")
    print("=" * 70)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Files: {tot} ({f_corr+f_wrong} fastener + {nf_corr+nf_wrong} non_fastener; "
          f"{f_err+nf_err} read errors)")
    print()
    print(f"                       predicted fastener   predicted not")
    print(f"  actual fastener      TP = {TP:>6}            FN = {FN:>6}")
    print(f"  actual non-fastener  FP = {FP:>6}            TN = {TN:>6}")
    print()
    print(f"  Accuracy   = {acc*100:>5.1f}%  (journal baseline: 86.5%)")
    print(f"  Precision  = {prec*100:>5.1f}%")
    print(f"  Recall     = {rec*100:>5.1f}%")
    print(f"  F1         = {f1*100:>5.1f}%")
    print()

    # 13-class histogram on the fastener side (which subtypes were predicted)
    print("13-class predictions on fastener/ folder:")
    hist = Counter(r["pred_class"] for r in f_rows)
    for cls, n in hist.most_common():
        print(f"  {cls:<20} {n:>4}")

    # Write CSV
    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file", "gt", "pred_class", "pred_binary", "confidence", "correct",
        ])
        w.writeheader()
        for r in f_rows + nf_rows:
            w.writerow(r)
    print(f"\nWrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
