"""PointNet++ MSG training script.

Usage:
    python -m step_vr_step.models.pointnet2.train \
        --data_path data/modelnet40/ \
        --epochs 200 \
        --log_dir logs/pointnet2

Transfer learning:
    python -m step_vr_step.models.pointnet2.train \
        --data_path data/fasteners/ \
        --pretrained_path checkpoints/pretrain_modelnet40.pth \
        --freeze_encoder \
        --epochs 100 \
        --lr 0.0001 \
        --log_dir logs/pointnet2_finetune
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .pointnet2_cls_msg import PointNet2ClsMSG
from .dataset import FastenerPointCloudDataset

logger = logging.getLogger(__name__)


def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # --- Data ---
    train_dataset = FastenerPointCloudDataset(
        root=args.data_path,
        num_points=args.num_points,
        use_normals=args.use_normals,
        split="train",
        augment=True,
    )
    val_dataset = FastenerPointCloudDataset(
        root=args.data_path,
        num_points=args.num_points,
        use_normals=args.use_normals,
        split="val",
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    num_classes = train_dataset.num_classes
    logger.info(f"Classes: {train_dataset.classes} ({num_classes})")
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # --- Model ---
    model = PointNet2ClsMSG(
        num_classes=num_classes,
        use_normals=args.use_normals,
    ).to(device)

    # Load pretrained weights for transfer learning
    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False)
        # Load weights, ignoring final FC layer size mismatch
        state_dict = checkpoint["model_state_dict"]
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained layers")

    # Freeze encoder layers for fine-tuning
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if name.startswith("sa1") or name.startswith("sa2") or name.startswith("sa3"):
                param.requires_grad = False
        logger.info("Encoder layers frozen (SA1, SA2, SA3)")

    # --- Training ---
    # Compute class weights to handle imbalance (fastener << non-fastener).
    # Read labels from .samples to skip the load+augment cost of __getitem__.
    class_counts = torch.zeros(num_classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    class_weights = class_counts.sum() / (num_classes * class_counts.clamp(min=1))
    class_weights = class_weights.to(device)
    logger.info(f"Class weights: {dict(zip(train_dataset.classes, class_weights.tolist()))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # TensorBoard
    writer = None
    if args.log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_path = Path(args.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(log_path))
        except ImportError:
            logger.warning("TensorBoard not available; skipping logging")

    best_val_acc = 0.0
    save_dir = Path(args.log_dir or "checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for points, labels in pbar:
            # points: (B, N, C) -> (B, C, N)
            points = points.transpose(1, 2).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(points)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{train_correct/train_total:.3f}")

        scheduler.step()

        train_acc = train_correct / max(train_total, 1)
        avg_train_loss = train_loss / max(train_total, 1)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for points, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False):
                points = points.transpose(1, 2).to(device)
                labels = labels.to(device)

                logits, _ = model(points)
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(val_total, 1)
        avg_val_loss = val_loss / max(val_total, 1)

        logger.info(
            f"Epoch {epoch}/{args.epochs} — "
            f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} — "
            f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        if writer:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = save_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
                "class_names": train_dataset.classes,
            }, save_path)
            logger.info(f"Saved best model (val_acc={val_acc:.4f}) to {save_path}")

    if writer:
        writer.close()

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train PointNet++ MSG classifier")
    parser.add_argument("--data_path", required=True, help="Dataset root directory")
    parser.add_argument("--num_points", type=int, default=2048, help="Points per sample")
    parser.add_argument("--use_normals", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_dir", default="logs/pointnet2")
    parser.add_argument("--pretrained_path", default=None, help="Path to pretrained checkpoint")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze SA layers for fine-tuning")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(args)


if __name__ == "__main__":
    main()
