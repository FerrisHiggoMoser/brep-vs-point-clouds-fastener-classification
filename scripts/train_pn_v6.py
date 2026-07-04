"""Train PointNet++ MSG 13-class on the v6 point-cloud twin dataset.

Matches prior PN++ runs (batch 16, 4096 points, xyz+normals, Adam lr 0.001
wd 1e-4, StepLR(20, 0.7)) with the v6 lessons applied:

  - NATURAL class distribution: plain CrossEntropyLoss, no class weights,
    no balanced sampling (v6's key lesson — reweighting destabilizes training).
  - Early stopping on val accuracy, patience 15; hard cap 60 epochs
    (dataset is ~10x the McMaster runs).
  - Checkpoints carry the val_loss in the FILENAME
    (best-epoch={E}-val_loss={L}.pth); downstream selection is by the
    val_loss token, never mtime (2026-05-09 bug).
  - Resume-safe: last.pth (model+optimizer+scheduler+epoch+history) is
    written every epoch; relaunching continues from the next epoch.

Usage (anaconda stepvrstep env):
    python backend/scripts/train_pn_v6.py
    python backend/scripts/train_pn_v6.py --epochs 60 --patience 15
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from step_vr_step.models.pointnet2.pointnet2_cls_msg import PointNet2ClsMSG
from step_vr_step.models.pointnet2.dataset import FastenerPointCloudDataset

RB_TD = Path(r"D:\step-vr-step-thesis\reproducible-build\training_data")
DATA_ROOT = RB_TD / "pn_v6_features"
RUN_DIR = RB_TD / "pn_v6_run"
CKPT_DIR = RUN_DIR / "checkpoints"
LOG_TSV = RUN_DIR / "train_log.tsv"


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for points, labels in loader:
            points = points.transpose(1, 2).to(device)
            labels = labels.to(device)
            logits, _ = model(points)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15,
                   help="early-stop if val_acc hasn't improved for N epochs")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_points", type=int, default=4096)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = FastenerPointCloudDataset(
        root=DATA_ROOT, num_points=args.num_points, use_normals=True,
        split="train", augment=True)
    val_ds = FastenerPointCloudDataset(
        root=DATA_ROOT, num_points=args.num_points, use_normals=True,
        split="val", augment=False)
    assert train_ds.classes == val_ds.classes, "split class mismatch"
    print(f"classes ({train_ds.num_classes}): {train_ds.classes}", flush=True)
    print(f"train={len(train_ds)} val={len(val_ds)}", flush=True)

    counts = {c: 0 for c in train_ds.classes}
    for _, lab in train_ds.samples:
        counts[train_ds.classes[lab]] += 1
    print(f"train class counts (NATURAL, unweighted): {counts}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            persistent_workers=args.num_workers > 0)

    model = PointNet2ClsMSG(num_classes=train_ds.num_classes, use_normals=True).to(device)
    # v6 lesson: natural distribution — NO class weights in the loss.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    start_epoch = 1
    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_since_improve = 0

    last_path = CKPT_DIR / "last.pth"
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val_acc = ck["best_val_acc"]
        best_val_loss = ck["best_val_loss"]
        epochs_since_improve = ck["epochs_since_improve"]
        print(f"RESUMED from epoch {ck['epoch']} "
              f"(best_val_acc={best_val_acc:.4f})", flush=True)

    new_log = not LOG_TSV.exists()
    logf = LOG_TSV.open("a", encoding="utf-8", newline="")
    logw = csv.writer(logf, delimiter="\t")
    if new_log:
        logw.writerow(["epoch", "train_loss", "train_acc", "val_loss",
                       "val_acc", "lr", "minutes"])
        logf.flush()

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = correct = total = 0
        for points, labels in train_loader:
            points = points.transpose(1, 2).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(points)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        mins = (time.time() - t0) / 60
        lr_now = scheduler.get_last_lr()[0]
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f}  val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f}  lr={lr_now:.6f}  {mins:.1f}min", flush=True)
        logw.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                       f"{val_loss:.6f}", f"{val_acc:.6f}",
                       f"{lr_now:.6g}", f"{mins:.2f}"])
        logf.flush()

        improved_acc = val_acc > best_val_acc
        if improved_acc:
            best_val_acc = val_acc
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ck_path = CKPT_DIR / f"best-epoch={epoch}-val_loss={val_loss:.4f}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "num_classes": train_ds.num_classes,
                "class_names": train_ds.classes,
            }, ck_path)
            print(f"  saved {ck_path.name}", flush=True)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "epochs_since_improve": epochs_since_improve,
            "num_classes": train_ds.num_classes,
            "class_names": train_ds.classes,
        }, last_path)

        if epochs_since_improve >= args.patience:
            print(f"EARLY STOP: val_acc no improvement for {args.patience} "
                  f"epochs (best {best_val_acc:.4f})", flush=True)
            break

    logf.close()
    summary = {
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "stopped_epoch": epoch,
        "config": vars(args),
    }
    (RUN_DIR / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"training complete: best_val_acc={best_val_acc:.4f} "
          f"best_val_loss={best_val_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
