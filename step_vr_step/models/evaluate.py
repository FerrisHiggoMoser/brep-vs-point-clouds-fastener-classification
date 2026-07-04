"""Evaluate and compare PointNet++ vs BRepFormer on the test set.

Usage:
    python -m step_vr_step.models.evaluate \
        --data_path training_data/pointclouds \
        --pointnet_checkpoint logs/pointnet2/best_model.pth \
        --brepformer_checkpoint logs/brepformer/best_model.pth

    # Or evaluate just one model:
    python -m step_vr_step.models.evaluate \
        --data_path training_data/pointclouds \
        --pointnet_checkpoint logs/pointnet2/best_model.pth

    # Run on an unlabeled STEP assembly (holdout test):
    python -m step_vr_step.models.evaluate \
        --assembly_path training_data/holdout_test/isispace_1uplt_type_b_2023-04-20.stp \
        --pointnet_checkpoint logs/pointnet2/best_model.pth
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def evaluate_classifier(model, data_loader, device, class_names):
    """Run a classifier on the test set and return detailed metrics."""
    model.eval()

    all_preds = []
    all_labels = []
    all_confs = []

    with torch.no_grad():
        for points, labels in data_loader:
            points = points.transpose(1, 2).to(device)
            labels = labels.to(device)

            logits, _ = model(points)
            probs = torch.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs = np.array(all_confs)

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()

    # Per-class metrics
    per_class = {}
    for i, name in enumerate(class_names):
        mask_true = all_labels == i
        mask_pred = all_preds == i

        tp = (mask_true & mask_pred).sum()
        fp = (~mask_true & mask_pred).sum()
        fn = (mask_true & ~mask_pred).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        per_class[name] = {
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1': round(float(f1), 4),
            'support': int(mask_true.sum()),
            'avg_confidence': round(float(all_confs[mask_pred].mean()) if mask_pred.any() else 0, 4),
        }

    # Confusion matrix
    n_classes = len(class_names)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        confusion[true][pred] += 1

    return {
        'accuracy': round(float(accuracy), 4),
        'per_class': per_class,
        'confusion_matrix': confusion.tolist(),
        'class_names': class_names,
        'total_samples': len(all_labels),
    }


def print_results(name: str, results: dict):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy: {results['accuracy']:.4f}  ({results['total_samples']} samples)")
    print()
    print(f"  {'Class':25s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Support':>8s} {'Conf':>8s}")
    print(f"  {'-'*65}")
    for cls_name, metrics in results['per_class'].items():
        print(f"  {cls_name:25s} {metrics['precision']:8.4f} {metrics['recall']:8.4f} "
              f"{metrics['f1']:8.4f} {metrics['support']:8d} {metrics['avg_confidence']:8.4f}")
    print()

    # Confusion matrix
    names = results['class_names']
    cm = results['confusion_matrix']
    print("  Confusion Matrix:")
    header = "  " + " " * 15 + "".join(f"{n[:8]:>10s}" for n in names)
    print(header)
    for i, row in enumerate(cm):
        line = f"  {names[i]:15s}" + "".join(f"{v:10d}" for v in row)
        print(line)
    print()


def evaluate_on_assembly(assembly_path: str, checkpoint_path: str, model_type: str = "pointnet"):
    """Run detection on a full STEP assembly and print what was detected.

    This is the real-world test: load a satellite assembly,
    split it into parts, classify each part.
    """
    from ..detection.detect import detect_fasteners
    from ..schema import Manifest, Part
    from ..config import DetectionConfig
    from .preprocess import load_step_shape, tessellate_shape, sample_points

    from OCC.Extend.TopologyUtils import TopologyExplorer

    print(f"\n{'='*60}")
    print(f"  Assembly Test: {Path(assembly_path).name}")
    print(f"  Model: {model_type} ({checkpoint_path})")
    print(f"{'='*60}")

    # Load assembly
    print("  Loading STEP assembly...", flush=True)
    shape = load_step_shape(assembly_path)
    topo = TopologyExplorer(shape)
    solids = list(topo.solids())
    print(f"  Found {len(solids)} solid bodies", flush=True)

    # Classify each solid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", ["fastener", "non-fastener"])
    num_classes = checkpoint.get("num_classes", len(class_names))

    if model_type == "pointnet":
        from .pointnet2.pointnet2_cls_msg import PointNet2ClsMSG
        model = PointNet2ClsMSG(num_classes=num_classes, use_normals=True)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    fastener_count = 0
    non_fastener_count = 0
    results = []

    for i, solid in enumerate(solids):
        try:
            verts, norms = tessellate_shape(solid, linear_deflection=0.1)
            if len(verts) < 10:
                continue

            pc = sample_points(verts, norms, 2048)
            tensor = torch.from_numpy(pc).unsqueeze(0).to(device)  # (1, 2048, 6)
            tensor = tensor.transpose(1, 2)  # (1, 6, 2048)

            with torch.no_grad():
                logits, _ = model(tensor)
                probs = torch.softmax(logits, dim=-1)
                conf, pred = probs.max(dim=-1)

            cls_name = class_names[int(pred[0])]
            confidence = float(conf[0])

            if cls_name == "fastener":
                fastener_count += 1
                label = "FASTENER"
            else:
                non_fastener_count += 1
                label = "structural"

            results.append({
                'part_index': i,
                'prediction': cls_name,
                'confidence': round(confidence, 4),
                'num_vertices': len(verts),
            })

            if confidence > 0.7:
                print(f"  Part {i:4d}: {label:12s} (conf={confidence:.3f}, verts={len(verts)})", flush=True)

        except Exception as e:
            logger.debug(f"Error on solid {i}: {e}")
            continue

    print(f"\n  Summary:")
    print(f"    Fasteners detected:     {fastener_count}")
    print(f"    Non-fastener parts:     {non_fastener_count}")
    print(f"    Total parts processed:  {len(results)}")
    print(f"    Fastener ratio:         {fastener_count/max(len(results),1):.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fastener detection models")
    parser.add_argument("--data_path", help="Point cloud dataset directory")
    parser.add_argument("--pointnet_checkpoint", help="PointNet++ checkpoint path")
    parser.add_argument("--brepformer_checkpoint", help="BRepFormer checkpoint path")
    parser.add_argument("--assembly_path", help="STEP assembly for holdout test")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output", default="eval_results.json", help="Save results as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}

    # --- Evaluate on test set ---
    if args.data_path:
        from .pointnet2.dataset import FastenerPointCloudDataset

        test_dataset = FastenerPointCloudDataset(
            root=args.data_path,
            num_points=2048,
            use_normals=True,
            split="test",
            augment=False,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
        )
        class_names = test_dataset.classes

        # PointNet++
        if args.pointnet_checkpoint:
            from .pointnet2.pointnet2_cls_msg import PointNet2ClsMSG
            checkpoint = torch.load(args.pointnet_checkpoint, map_location=device, weights_only=False)
            model = PointNet2ClsMSG(
                num_classes=checkpoint.get("num_classes", len(class_names)),
                use_normals=True,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)

            results = evaluate_classifier(model, test_loader, device, class_names)
            print_results("PointNet++ MSG", results)
            all_results["pointnet2"] = results

        # BRepFormer
        if args.brepformer_checkpoint:
            # BRepFormer needs different data loading — B-Rep features not point clouds
            logger.info("BRepFormer evaluation requires B-Rep feature dataset (not implemented yet)")

    # --- Holdout assembly test ---
    if args.assembly_path and args.pointnet_checkpoint:
        assembly_results = evaluate_on_assembly(
            args.assembly_path, args.pointnet_checkpoint, "pointnet"
        )
        all_results["assembly_test"] = assembly_results

    # --- Compare models ---
    if "pointnet2" in all_results and "brepformer" in all_results:
        print(f"\n{'='*60}")
        print(f"  MODEL COMPARISON")
        print(f"{'='*60}")
        p = all_results["pointnet2"]
        b = all_results["brepformer"]
        print(f"  {'Metric':20s} {'PointNet++':>12s} {'BRepFormer':>12s} {'Winner':>10s}")
        print(f"  {'-'*56}")
        for metric in ['accuracy']:
            pv, bv = p[metric], b[metric]
            winner = "PointNet++" if pv > bv else "BRepFormer" if bv > pv else "Tie"
            print(f"  {metric:20s} {pv:12.4f} {bv:12.4f} {winner:>10s}")
        # Per-class F1
        for cls in p['per_class']:
            pf1 = p['per_class'][cls]['f1']
            bf1 = b['per_class'][cls]['f1']
            winner = "PointNet++" if pf1 > bf1 else "BRepFormer" if bf1 > pf1 else "Tie"
            print(f"  F1({cls[:12]}){' '*(8-len(cls[:12]))} {pf1:12.4f} {bf1:12.4f} {winner:>10s}")

    # Save
    output_path = Path(args.output)
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
