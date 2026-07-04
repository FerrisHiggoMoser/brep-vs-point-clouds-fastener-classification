"""BRepFormer training script using PyTorch Lightning.

Usage:
    python -m step_vr_step.models.brepformer.train \
        --data_dir data/fusion360/processed/ \
        --epochs 200 \
        --batch_size 64 \
        --log_dir logs/brepformer
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    _LIGHTNING_AVAILABLE = True
except ImportError:
    _LIGHTNING_AVAILABLE = False

from torch.utils.data import DataLoader, WeightedRandomSampler

from .brepformer import BRepFormer
from .dataset import BRepDataset, brep_collate_fn


class BRepFormerLightning(pl.LightningModule if _LIGHTNING_AVAILABLE else nn.Module):
    """PyTorch Lightning wrapper for BRepFormer training."""

    def __init__(
        self,
        num_classes: int = 10,
        dim: int = 256,
        num_layers: int = 8,
        lr: float = 0.001,
        warmup_steps: int = 5000,
        weight_decay: float = 0.01,
        dropout: float = 0.0,
        class_weights: list[float] | None = None,
        head_mode: str = "classification",
    ):
        super().__init__()
        if _LIGHTNING_AVAILABLE:
            self.save_hyperparameters()

        self.head_mode = head_mode
        self.model = BRepFormer(
            num_classes=num_classes,
            dim=dim,
            num_layers=num_layers,
            dropout=dropout,
            head_mode=head_mode,
        )
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", w)
            self.criterion = nn.CrossEntropyLoss(weight=w)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

    def forward(self, batch):
        return self.model(
            face_grids=batch["face_grids"],
            edge_curves=batch["edge_curves"],
            topo_distances=batch["topo_distances"],
            mask=batch.get("face_mask"),
            edge_mask=batch.get("edge_mask"),
        )

    def _step(self, batch, stage: str):
        logits = self(batch)
        if self.head_mode == "segmentation":
            # logits: (B, Nf, C), labels: (B, Nf), face_mask: (B, Nf)
            labels = batch["labels"]
            fm = batch.get("face_mask")
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_labels = labels.reshape(-1)
            if fm is not None:
                flat_mask = fm.reshape(-1)
                loss = self.criterion(flat_logits[flat_mask], flat_labels[flat_mask])
                acc = (flat_logits[flat_mask].argmax(-1) == flat_labels[flat_mask]).float().mean()
            else:
                loss = self.criterion(flat_logits, flat_labels)
                acc = (flat_logits.argmax(-1) == flat_labels).float().mean()
        else:
            loss = self.criterion(logits, batch["labels"])
            acc = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # Linear warmup
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = self.lr * lr_scale

        if optimizer_closure is not None:
            optimizer.step(closure=optimizer_closure)
        else:
            optimizer.step()


def train(args):
    if not _LIGHTNING_AVAILABLE:
        raise RuntimeError(
            "PyTorch Lightning is required for BRepFormer training. "
            "Install with: pip install pytorch-lightning"
        )

    # Data
    train_dataset = BRepDataset(root=args.data_dir, split="train")
    val_dataset = BRepDataset(root=args.data_dir, split="val")

    num_classes = args.num_classes if args.num_classes else train_dataset.num_classes
    logger.info(f"Classes: {train_dataset.classes} (effective num_classes={num_classes})")
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Per-sample weights for oversampling — each minority-class sample drawn as often
    # as each majority-class sample on average. Irrelevant for segmentation (per-face labels),
    # so skipped in that mode.
    if args.head_mode == "segmentation":
        class_counts = None
        class_weights_arg = None
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=brep_collate_fn,
        )
        logger.info("Segmentation mode: no class-level weighting/sampling (per-face labels)")
    elif args.balanced_sampling:
        class_counts = torch.zeros(len(train_dataset.classes))
        for _, label in train_dataset.samples:
            class_counts[label] += 1
        logger.info(f"Class counts: {dict(zip(train_dataset.classes, class_counts.tolist()))}")
        # Mild oversampling — sqrt of inverse-frequency instead of full inverse.
        # Fully-balanced sampling (50/50) causes reverse collapse: model predicts all-fastener
        # because val distribution is 91/9, not 50/50. sqrt keeps the class prior partially intact.
        per_class_weight = 1.0 / torch.sqrt(class_counts.clamp(min=1))
        sample_weights = torch.tensor([per_class_weight[label].item() for _, label in train_dataset.samples])
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_dataset), replacement=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, collate_fn=brep_collate_fn,
        )
        # Keep mild class weighting in the loss as well
        cw = torch.sqrt(class_counts.sum() / (num_classes * class_counts.clamp(min=1)))
        class_weights_arg = cw.tolist()
        logger.info(f"Using sqrt-oversampling + sqrt-class-weights: {dict(zip(train_dataset.classes, cw.tolist()))}")
    else:
        class_counts = torch.zeros(len(train_dataset.classes))
        for _, label in train_dataset.samples:
            class_counts[label] += 1
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=brep_collate_fn,
        )
        if len(train_dataset.classes) > 1:
            cw = class_counts.sum() / (len(train_dataset.classes) * class_counts.clamp(min=1))
            class_weights_arg = cw.tolist()
            logger.info(f"Class weights: {dict(zip(train_dataset.classes, cw.tolist()))}")
        else:
            class_weights_arg = None

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=brep_collate_fn,
    )

    # Model
    model = BRepFormerLightning(
        num_classes=num_classes,
        dim=args.dim,
        num_layers=args.num_layers,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        dropout=args.dropout,
        class_weights=class_weights_arg,
        head_mode=args.head_mode,
    )

    # Optional: load pretrained encoder weights for transfer learning
    if args.pretrained_path:
        ckpt = torch.load(args.pretrained_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        # Strip possible "model." prefix (from Lightning checkpoints)
        sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in sd.items()}
        own_sd = model.model.state_dict()
        loadable = {k: v for k, v in sd.items() if k in own_sd and v.shape == own_sd[k].shape}
        own_sd.update(loadable)
        model.model.load_state_dict(own_sd)
        logger.info(f"Loaded {len(loadable)}/{len(own_sd)} pretrained layers from {args.pretrained_path}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.log_dir) / "checkpoints",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Logger
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name="brepformer")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    trainer.fit(model, train_loader, val_loader)

    # Save final model in a format compatible with ml_classifier.py
    best_path = checkpoint_callback.best_model_path
    if best_path:
        ckpt = torch.load(best_path, weights_only=False)
        save_path = Path(args.log_dir) / "best_model.pth"
        torch.save({
            "model_state_dict": model.model.state_dict(),
            "num_classes": num_classes,
            "class_names": train_dataset.classes,
        }, save_path)
        logger.info(f"Saved inference-ready model to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train BRepFormer classifier")
    parser.add_argument("--data_dir", required=True, help="Preprocessed B-Rep dataset directory")
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--balanced_sampling", action="store_true",
                        help="Use WeightedRandomSampler for balanced class batches")
    parser.add_argument("--head_mode", choices=["classification", "segmentation"],
                        default="classification",
                        help="Output head mode: classification (part-level) or segmentation (per-face)")
    parser.add_argument("--pretrained_path", default=None,
                        help="Path to pretrained model checkpoint (loads encoder weights for transfer)")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Override number of output classes (for segmentation datasets)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_dir", default="logs/brepformer")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Microbatch accumulation. effective_batch = batch_size * this.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(args)


if __name__ == "__main__":
    main()
