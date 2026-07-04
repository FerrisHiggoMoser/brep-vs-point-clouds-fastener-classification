"""Analyze Stage 2 model's errors on the GrabCAD test set.

For each misclassified sample, records the source sub-category (hex-nut, snap-ring, anchor, etc.)
by resolving the XSym stub in training_data/organized/ back to training_data/grabcad/<category>/<name>.

Outputs:
  - errors.csv           one row per test sample with label, prediction, softmax probs, source category
  - misses.txt           the 6 missed fasteners grouped by source sub-category
  - false_positives.txt  the ~20 falsely-flagged non-fasteners grouped by source sub-category
  - errors_summary.md    markdown summary you can paste into the thesis

Usage:
    python backend/scripts/error_analysis.py \
        --checkpoint logs/pointnet2_finetune/best_model.pth \
        --npy_root training_data/organized_npy \
        --stub_root training_data/organized \
        --output_dir logs/pointnet2_finetune/error_analysis
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from step_vr_step.models.pointnet2.dataset import FastenerPointCloudDataset
from step_vr_step.models.pointnet2.pointnet2_cls_msg import PointNet2ClsMSG

MAC_PREFIX = "/Volumes/Uncle Sam/GitHub/step-vr-step/"


def resolve_stub_category(stub_path: Path) -> str:
    """Read the XSym stub and return the sub-category (e.g. 'hex-nut') from the Mac path.
    Returns 'unknown' if stub can't be resolved."""
    try:
        with stub_path.open("rb") as f:
            head = f.read(4)
            if head != b"XSym":
                return "non-symlink"
            f.seek(0)
            lines = f.read(1200).decode("utf-8", errors="replace").splitlines()
        if len(lines) < 4:
            return "unknown"
        target = lines[3].rstrip("\x00").strip()
        if target.startswith(MAC_PREFIX):
            rel = target[len(MAC_PREFIX):]
            parts = rel.split("/")
            # training_data/<collection>/<category>/<file>
            if len(parts) >= 3 and parts[0] == "training_data":
                return f"{parts[1]}/{parts[2]}"
    except OSError:
        pass
    return "unknown"


def find_stub(npy_path: Path, npy_root: Path, stub_root: Path) -> Path | None:
    """Find the XSym stub corresponding to a .npy test file by matching train/val/test/class/name."""
    rel = npy_path.relative_to(npy_root).with_suffix("")
    for ext in (".stp", ".STEP", ".STP", ".step"):
        candidate = stub_root / rel.parent / f"{rel.name}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--npy_root", required=True, type=Path)
    p.add_argument("--stub_root", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    assert class_names == ["fastener", "non-fastener"], f"unexpected classes: {class_names}"

    dataset = FastenerPointCloudDataset(
        root=args.npy_root,
        num_points=args.num_points,
        use_normals=True,
        split="test",
        augment=False,
    )
    # We need ordered access to the file paths — DataLoader with shuffle=False preserves .samples order.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = PointNet2ClsMSG(num_classes=2, use_normals=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_probs: list[tuple[float, float]] = []
    with torch.no_grad():
        for points, _ in tqdm(loader, desc="Inference"):
            points = points.transpose(1, 2).to(device)
            logits, _ = model(points)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                all_probs.append((float(row[0]), float(row[1])))

    rows = []
    for (npy_path, label), (p_fast, p_non) in zip(dataset.samples, all_probs):
        pred = 0 if p_fast > p_non else 1
        stub = find_stub(npy_path, args.npy_root, args.stub_root)
        category = resolve_stub_category(stub) if stub else "missing-stub"
        rows.append({
            "npy_path": str(npy_path),
            "stub_path": str(stub) if stub else "",
            "true_label": class_names[label],
            "pred_label": class_names[pred],
            "correct": label == pred,
            "p_fastener": p_fast,
            "p_non_fastener": p_non,
            "source_category": category,
        })

    with (args.output_dir / "errors.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    misses = [r for r in rows if r["true_label"] == "fastener" and r["pred_label"] == "non-fastener"]
    fps = [r for r in rows if r["true_label"] == "non-fastener" and r["pred_label"] == "fastener"]
    tps = [r for r in rows if r["true_label"] == "fastener" and r["pred_label"] == "fastener"]
    tns = [r for r in rows if r["true_label"] == "non-fastener" and r["pred_label"] == "non-fastener"]

    # Missed fasteners (false negatives)
    miss_by_cat: dict[str, list[dict]] = defaultdict(list)
    for m in misses:
        miss_by_cat[m["source_category"]].append(m)
    with (args.output_dir / "misses.txt").open("w", encoding="utf-8") as f:
        f.write(f"MISSED FASTENERS ({len(misses)} total)\n")
        f.write("=" * 80 + "\n\n")
        for cat in sorted(miss_by_cat.keys()):
            items = miss_by_cat[cat]
            f.write(f"[{cat}] ({len(items)} files)\n")
            for m in sorted(items, key=lambda r: -r["p_non_fastener"]):
                name = Path(m["npy_path"]).stem
                f.write(
                    f"  {name:60s}  p(non-fastener)={m['p_non_fastener']:.3f}\n"
                )
            f.write("\n")

    # False positives
    fp_by_cat: dict[str, list[dict]] = defaultdict(list)
    for fp in fps:
        fp_by_cat[fp["source_category"]].append(fp)
    with (args.output_dir / "false_positives.txt").open("w", encoding="utf-8") as f:
        f.write(f"FALSE POSITIVES — non-fasteners flagged as fasteners ({len(fps)} total)\n")
        f.write("=" * 80 + "\n\n")
        for cat in sorted(fp_by_cat.keys()):
            items = fp_by_cat[cat]
            f.write(f"[{cat}] ({len(items)} files)\n")
            for fp in sorted(items, key=lambda r: -r["p_fastener"]):
                name = Path(fp["npy_path"]).stem
                f.write(
                    f"  {name:60s}  p(fastener)={fp['p_fastener']:.3f}\n"
                )
            f.write("\n")

    # Markdown summary
    miss_cat_counts = Counter(m["source_category"] for m in misses)
    fp_cat_counts = Counter(fp["source_category"] for fp in fps)

    md = []
    md.append("# Stage 2 PointNet++ Error Analysis on GrabCAD Test Set\n")
    md.append(f"**Checkpoint:** `{args.checkpoint}`  ")
    md.append(f"**Test set size:** {len(rows)} samples ({sum(r['true_label']=='fastener' for r in rows)} fasteners, {sum(r['true_label']=='non-fastener' for r in rows)} non-fasteners)\n")
    md.append("## Confusion breakdown\n")
    md.append("| | pred: fastener | pred: non-fastener |")
    md.append("|---|---:|---:|")
    md.append(f"| **true: fastener** | {len(tps)} (TP) | {len(misses)} (FN / miss) |")
    md.append(f"| **true: non-fastener** | {len(fps)} (FP) | {len(tns)} (TN) |")
    md.append("")
    md.append("## Missed fasteners by source category")
    md.append("| sub-category | count |")
    md.append("|---|---:|")
    for cat, n in miss_cat_counts.most_common():
        md.append(f"| {cat} | {n} |")
    md.append("")
    md.append("## False positives by source category")
    md.append("| sub-category | count |")
    md.append("|---|---:|")
    for cat, n in fp_cat_counts.most_common():
        md.append(f"| {cat} | {n} |")
    md.append("")
    md.append("See `misses.txt` and `false_positives.txt` for per-file details and model confidence.")
    (args.output_dir / "errors_summary.md").write_text("\n".join(md), encoding="utf-8")

    # Console summary
    print(f"\nWrote: {args.output_dir}/{{errors.csv, misses.txt, false_positives.txt, errors_summary.md}}")
    print(f"\n=== MISSED FASTENERS ({len(misses)}) — grouped by source category ===")
    for cat, n in miss_cat_counts.most_common():
        print(f"  {cat}: {n}")
    print(f"\n=== FALSE POSITIVES ({len(fps)}) — grouped by source category ===")
    for cat, n in fp_cat_counts.most_common():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
