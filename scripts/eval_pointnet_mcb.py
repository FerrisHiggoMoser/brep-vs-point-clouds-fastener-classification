"""Evaluate a trained PointNet++ checkpoint on the held-out MCB test set.

Produces:
  - Overall accuracy + macro/weighted precision, recall, F1
  - Per-class precision/recall/F1/support table
  - Confusion matrix PNG
  - Top-20 most-confused class pairs
  - Raw predictions CSV for later analysis

Usage:
    python backend/scripts/eval_pointnet_mcb.py \
        --checkpoint logs/pointnet2_mcb/best_model.pth \
        --data_path training_data/mcb/npy \
        --output_dir logs/pointnet2_mcb/eval
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from step_vr_step.models.pointnet2.dataset import FastenerPointCloudDataset
from step_vr_step.models.pointnet2.pointnet2_cls_msg import PointNet2ClsMSG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data_path", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--split", default="test")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    num_classes = ckpt["num_classes"]
    print(f"Checkpoint epoch {ckpt['epoch']}, val_acc {ckpt['val_acc']:.4f}, classes {num_classes}")

    dataset = FastenerPointCloudDataset(
        root=args.data_path,
        num_points=args.num_points,
        use_normals=True,
        split=args.split,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Test set: {len(dataset)} samples")

    model = PointNet2ClsMSG(num_classes=num_classes, use_normals=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for points, labels in tqdm(loader, desc="Evaluating"):
            points = points.transpose(1, 2).to(device)
            logits, _ = model(points)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Overall metrics
    overall_acc = float((y_true == y_pred).mean())
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_samples": len(y_true),
        "overall_accuracy": overall_acc,
        "macro_precision": float(p_mac),
        "macro_recall": float(r_mac),
        "macro_f1": float(f_mac),
        "weighted_precision": float(p_w),
        "weighted_recall": float(r_w),
        "weighted_f1": float(f_w),
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_acc": ckpt["val_acc"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Overall ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Per-class report
    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=3, zero_division=0)
    (args.output_dir / "per_class_report.txt").write_text(report_txt)
    print("\n=== Per-class (saved to per_class_report.txt) ===")
    print(report_txt)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        cm_norm,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Normalized frequency"},
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"MCB-A test — row-normalized confusion matrix (acc={overall_acc:.3f})")
    plt.xticks(rotation=90, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    np.save(args.output_dir / "confusion_matrix.npy", cm)

    # Top confused pairs (off-diagonal only)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and off[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(off[i, j]), int(cm[i].sum())))
    pairs.sort(key=lambda x: x[2], reverse=True)
    with (args.output_dir / "top_confused_pairs.txt").open("w") as f:
        f.write("true_class  ->  predicted_class  |  count / support  (fraction)\n")
        f.write("-" * 90 + "\n")
        for t, p, c, support in pairs[:30]:
            frac = c / support if support else 0
            f.write(f"{t:45s} -> {p:45s} | {c:4d}/{support:<5d} ({frac:.2%})\n")

    # Raw predictions
    with (args.output_dir / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_label_idx", "pred_label_idx", "true_class", "pred_class"])
        for t, p in zip(y_true, y_pred):
            w.writerow([int(t), int(p), class_names[t], class_names[p]])

    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
