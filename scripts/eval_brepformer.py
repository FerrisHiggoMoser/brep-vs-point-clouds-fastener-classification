"""Evaluate a fine-tuned BRepFormer classification checkpoint on the held-out test set.

Produces:
  - Overall accuracy + per-class precision/recall/F1
  - Confusion matrix (binary → simple print)
  - predictions.csv with per-sample probabilities

Usage:
    python backend/scripts/eval_brepformer.py \
        --checkpoint logs/brepformer_finetune/best_model.pth \
        --data_dir training_data/organized_brep \
        --output_dir logs/brepformer_finetune/eval
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from step_vr_step.models.brepformer.brepformer import BRepFormer
from step_vr_step.models.brepformer.dataset import BRepDataset, brep_collate_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--split", default="test")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=8)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt.get("class_names", ["fastener", "non-fastener"])
    num_classes = ckpt.get("num_classes", len(class_names))

    dataset = BRepDataset(root=args.data_dir, split=args.split)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=brep_collate_fn,
    )
    print(f"Test set: {len(dataset)} samples  classes={class_names}")

    model = BRepFormer(
        num_classes=num_classes, dim=args.dim, num_layers=args.num_layers,
        head_mode="classification",
    ).to(device)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            face_grids = batch["face_grids"].to(device)
            edge_curves = batch["edge_curves"].to(device)
            topo = {k: v.to(device) for k, v in batch["topo_distances"].items()}
            face_mask = batch["face_mask"].to(device)
            edge_mask = batch["edge_mask"].to(device)
            logits = model(face_grids, edge_curves, topo, face_mask, edge_mask)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(probs.argmax(axis=1).tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0)

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_samples": int(len(y_true)),
        "overall_accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "per_class_report.txt").write_text(report)

    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true_label_idx", "pred_label_idx", "true_class", "pred_class"] + [f"p_{c}" for c in class_names])
        for i, (t, p, probs) in enumerate(zip(y_true, y_pred, all_probs)):
            w.writerow([i, int(t), int(p), class_names[t], class_names[p]] + [f"{v:.6f}" for v in probs])

    print(f"\n=== Overall ===")
    print(f"  overall_accuracy: {acc:.4f}")
    print(f"  n_samples: {len(y_true)}")
    print(f"\n=== Confusion matrix ===")
    print(f"  {'':20s}  {' '.join(f'{c[:10]:>10s}' for c in class_names)}")
    for i, cname in enumerate(class_names):
        print(f"  true {cname:15s} {' '.join(f'{v:>10d}' for v in cm[i])}")
    print(f"\n=== Per-class ===")
    print(report)
    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
