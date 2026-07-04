"""Full multiclass paired analysis: PointNet++ vs BRepFormer on 13-class subtype task.

Computes, per model:
  - Accuracy with bootstrap 95% CI
  - Per-class precision, recall, F1 (one-vs-rest)
  - Macro F1, weighted F1, balanced accuracy
  - Multiclass MCC (Gorodkin)
  - Cohen's kappa vs random chance
  - AUROC OvR (macro-averaged)
  - Multiclass Brier score
  - ECE on max-softmax confidence
  - Top-3 accuracy
  - Binary-collapse accuracy (sanity check)

Computes paired:
  - 13×13 confusion matrix per model
  - Overall McNemar (continuity-corrected)
  - Per-subtype McNemar (one test per fastener subtype)
  - Cohen's kappa between models on predictions
  - Bootstrap 95% CI for accuracy difference

Outputs:
  - training_data/mcmaster_logs/full_analysis_subtype13.json
  - training_data/mcmaster_logs/full_analysis_subtype13.md
"""
from __future__ import annotations
from pathlib import Path
import json
import sys
import re
from collections import Counter, defaultdict
from math import erfc, sqrt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(r"c:\Users\ferri\OneDrive\Documents\GitHub\step-vr-step")
sys.path.insert(0, str(REPO / "backend"))

from step_vr_step.models.pointnet2.pointnet2_cls_msg import PointNet2ClsMSG
from step_vr_step.models.pointnet2.dataset import FastenerPointCloudDataset
from step_vr_step.models.brepformer.dataset import BRepDataset, brep_collate_fn
from step_vr_step.models.brepformer.brepformer import BRepFormer

PC_TEST = REPO / "training_data" / "mcmaster_pc_subtype13"
BREP_TEST = REPO / "training_data" / "mcmaster_brep_subtype13"
PN_LOG_DIR = REPO / "training_data" / "mcmaster_logs" / "pointnet2_subtype13"
BF_LOG_DIR = REPO / "training_data" / "mcmaster_logs" / "brepformer_subtype13"
OUT_JSON = REPO / "training_data" / "mcmaster_logs" / "full_analysis_subtype13.json"
OUT_MD   = REPO / "training_data" / "mcmaster_logs" / "full_analysis_subtype13.md"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NON_FASTENER = "non_fastener"


# --------------- collect predictions ---------------

def collect_pn(num_points: int = 4096) -> tuple[dict[str, dict], list[str]]:
    ds = FastenerPointCloudDataset(root=str(PC_TEST), num_points=num_points,
                                   use_normals=True, split="test", augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    model = PointNet2ClsMSG(num_classes=ds.num_classes, use_normals=True).to(DEVICE)
    pn_ckpts = sorted(PN_LOG_DIR.glob("*.pth"))
    if not pn_ckpts:
        raise FileNotFoundError(f"No PN checkpoint in {PN_LOG_DIR}")
    ckpt_path = pn_ckpts[-1]
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.eval()

    out: dict[str, dict] = {}
    sample_idx = 0
    with torch.no_grad():
        for points, labels in loader:
            points_t = points.transpose(1, 2).to(DEVICE)
            logits, _ = model(points_t)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(-1)
            for j in range(labels.size(0)):
                fp, lab = ds.samples[sample_idx + j]
                pn = Path(fp).stem
                out[pn] = {
                    "true": int(lab),
                    "pred": int(preds[j].item()),
                    "probs": probs[j].cpu().tolist(),
                }
            sample_idx += labels.size(0)
    print(f"  PN++ checkpoint: {ckpt_path.name}, classes: {ds.classes}")
    return out, ds.classes


def collect_bf() -> tuple[dict[str, dict], list[str]]:
    ds = BRepDataset(root=str(BREP_TEST), split="test")
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                        collate_fn=brep_collate_fn)
    model = BRepFormer(num_classes=ds.num_classes).to(DEVICE)
    # Pick lowest-val_loss checkpoint by filename token (avoid mtime selection bug)
    ckpt_dir = BF_LOG_DIR / "checkpoints"
    ckpts = list(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
    if not ckpts:
        raise FileNotFoundError(f"No BF checkpoint in {ckpt_dir}")
    def vl(p: Path) -> float:
        m = re.search(r"val_loss=(\d+\.\d+)", p.name)
        return float(m.group(1)) if m else float("inf")
    ckpts.sort(key=vl)
    ckpt_path = ckpts[0]
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    out: dict[str, dict] = {}
    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(face_grids=batch["face_grids"].to(DEVICE),
                           edge_curves=batch["edge_curves"].to(DEVICE),
                           topo_distances={k: v.to(DEVICE) for k, v in batch["topo_distances"].items()},
                           mask=batch["face_mask"].to(DEVICE),
                           edge_mask=batch["edge_mask"].to(DEVICE))
            labels = batch["labels"]
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(-1)
            for j in range(labels.size(0)):
                model_dir, lab = ds.samples[sample_idx + j]
                pn = Path(model_dir).name
                out[pn] = {
                    "true": int(lab),
                    "pred": int(preds[j].item()),
                    "probs": probs[j].cpu().tolist(),
                }
            sample_idx += labels.size(0)
    print(f"  BF checkpoint: {ckpt_path.name} (val_loss={vl(ckpt_path):.4f}), classes: {ds.classes}")
    return out, ds.classes


# --------------- multiclass metrics ---------------

def confusion(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> np.ndarray:
    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_metrics(cm: np.ndarray, classes: list[str]) -> dict[str, dict]:
    out = {}
    for i, name in enumerate(classes):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[name] = {"tp": tp, "fn": fn, "fp": fp,
                     "support": tp + fn,
                     "recall": recall, "precision": precision, "f1": f1}
    return out


def macro_f1(per_class: dict) -> float:
    f1s = [v["f1"] for v in per_class.values()]
    return float(np.mean(f1s)) if f1s else 0.0


def weighted_f1(per_class: dict) -> float:
    total = sum(v["support"] for v in per_class.values())
    if total == 0:
        return 0.0
    return float(sum(v["f1"] * v["support"] for v in per_class.values()) / total)


def balanced_accuracy(per_class: dict) -> float:
    recalls = [v["recall"] for v in per_class.values()]
    return float(np.mean(recalls)) if recalls else 0.0


def multiclass_mcc(cm: np.ndarray) -> float:
    """Gorodkin 2004 multiclass MCC."""
    cm = cm.astype(np.float64)
    n = cm.sum()
    t = cm.sum(axis=1)  # true class totals
    p = cm.sum(axis=0)  # pred class totals
    c = np.trace(cm)    # correct count
    s = n
    num = c * s - (t * p).sum()
    den_a = s * s - (p * p).sum()
    den_b = s * s - (t * t).sum()
    if den_a <= 0 or den_b <= 0:
        return 0.0
    return float(num / np.sqrt(den_a * den_b))


def multiclass_kappa(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> float:
    n = len(y_true)
    po = (y_pred == y_true).sum() / n
    p_pred = np.bincount(y_pred, minlength=K) / n
    p_true = np.bincount(y_true, minlength=K) / n
    pe = float((p_pred * p_true).sum())
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def kappa_models(pred_a: np.ndarray, pred_b: np.ndarray, K: int) -> float:
    n = len(pred_a)
    po = (pred_a == pred_b).sum() / n
    pa = np.bincount(pred_a, minlength=K) / n
    pb = np.bincount(pred_b, minlength=K) / n
    pe = float((pa * pb).sum())
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def auroc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary AUROC via Mann-Whitney rank-sum."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    ranks = all_scores.argsort().argsort() + 1
    rank_sum_pos = ranks[: len(pos)].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auroc_ovr_macro(y_true: np.ndarray, probs: np.ndarray, K: int) -> tuple[float, dict]:
    per_class = {}
    for k in range(K):
        bin_labels = (y_true == k).astype(int)
        per_class[k] = auroc_binary(probs[:, k], bin_labels)
    valid = [v for v in per_class.values() if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan"), per_class


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray, K: int) -> float:
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true)), y_true] = 1
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def ece_max_softmax(y_true: np.ndarray, y_pred: np.ndarray,
                    probs: np.ndarray, n_bins: int = 10) -> float:
    confidence = probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(y_true)
    ece_val = 0.0
    for i in range(n_bins):
        in_bin = (confidence > bins[i]) & (confidence <= bins[i + 1])
        if in_bin.sum() == 0:
            continue
        acc_bin = correct[in_bin].mean()
        conf_bin = confidence[in_bin].mean()
        ece_val += (in_bin.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece_val)


def top_k_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    hits = (topk == y_true[:, None]).any(axis=1)
    return float(hits.mean())


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)
    a_only = int((a_correct & (~b_correct)).sum())
    b_only = int(((~a_correct) & b_correct).sum())
    both_correct = int((a_correct & b_correct).sum())
    both_wrong = int(((~a_correct) & (~b_correct)).sum())
    n_dis = a_only + b_only
    if n_dis == 0:
        return {"a_only": 0, "b_only": 0, "both_correct": both_correct,
                "both_wrong": both_wrong, "chi2": 0.0, "p_value": 1.0}
    chi2 = ((abs(a_only - b_only) - 1) ** 2) / n_dis
    p = float(erfc(sqrt(chi2 / 2.0)))
    return {"a_only": a_only, "b_only": b_only, "both_correct": both_correct,
            "both_wrong": both_wrong, "chi2": float(chi2), "p_value": p}


def bootstrap_acc_diff(y_true, pred_a, pred_b, n=2000, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    accs_a = np.empty(n); accs_b = np.empty(n); diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        accs_a[i] = (pred_a[idx] == y_true[idx]).mean()
        accs_b[i] = (pred_b[idx] == y_true[idx]).mean()
        diffs[i] = accs_a[i] - accs_b[i]
    def ci(arr): return (float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return {"a_acc": ci(accs_a), "b_acc": ci(accs_b), "diff": ci(diffs)}


# --------------- main ---------------

def main():
    print("[1/4] running PointNet++ subtype13 on test set ...")
    pn, pn_classes = collect_pn()
    print(f"  {len(pn)} predictions")

    print("[2/4] running BRepFormer subtype13 on test set ...")
    bf, bf_classes = collect_bf()
    print(f"  {len(bf)} predictions")

    if pn_classes != bf_classes:
        raise ValueError(f"Class lists disagree:\n  PN: {pn_classes}\n  BF: {bf_classes}")
    classes = pn_classes
    K = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    non_fastener_idx = cls_to_idx.get(NON_FASTENER)

    shared = sorted(set(pn.keys()) & set(bf.keys()))
    print(f"[3/4] {len(shared)} shared part-numbers (PN-only={len(pn)-len(shared)}, BF-only={len(bf)-len(shared)})")

    y_true = np.array([pn[k]["true"] for k in shared], dtype=int)
    y_pn = np.array([pn[k]["pred"] for k in shared], dtype=int)
    y_bf = np.array([bf[k]["pred"] for k in shared], dtype=int)
    probs_pn = np.array([pn[k]["probs"] for k in shared], dtype=np.float32)
    probs_bf = np.array([bf[k]["probs"] for k in shared], dtype=np.float32)

    cm_pn = confusion(y_true, y_pn, K)
    cm_bf = confusion(y_true, y_bf, K)
    pc_pn = per_class_metrics(cm_pn, classes)
    pc_bf = per_class_metrics(cm_bf, classes)

    auroc_pn, auroc_pn_per = auroc_ovr_macro(y_true, probs_pn, K)
    auroc_bf, auroc_bf_per = auroc_ovr_macro(y_true, probs_bf, K)

    pn_metrics = {
        "n": len(shared),
        "accuracy": float((y_pn == y_true).mean()),
        "macro_f1": macro_f1(pc_pn),
        "weighted_f1": weighted_f1(pc_pn),
        "balanced_accuracy": balanced_accuracy(pc_pn),
        "mcc_multiclass": multiclass_mcc(cm_pn),
        "kappa_vs_random": multiclass_kappa(y_true, y_pn, K),
        "auroc_ovr_macro": auroc_pn,
        "auroc_per_class": {classes[k]: float(v) for k, v in auroc_pn_per.items()},
        "brier_multiclass": multiclass_brier(y_true, probs_pn, K),
        "ece_max_softmax": ece_max_softmax(y_true, y_pn, probs_pn),
        "top3_accuracy": top_k_accuracy(y_true, probs_pn, 3),
        "per_class": pc_pn,
        "confusion_matrix": cm_pn.tolist(),
    }
    bf_metrics = {
        "n": len(shared),
        "accuracy": float((y_bf == y_true).mean()),
        "macro_f1": macro_f1(pc_bf),
        "weighted_f1": weighted_f1(pc_bf),
        "balanced_accuracy": balanced_accuracy(pc_bf),
        "mcc_multiclass": multiclass_mcc(cm_bf),
        "kappa_vs_random": multiclass_kappa(y_true, y_bf, K),
        "auroc_ovr_macro": auroc_bf,
        "auroc_per_class": {classes[k]: float(v) for k, v in auroc_bf_per.items()},
        "brier_multiclass": multiclass_brier(y_true, probs_bf, K),
        "ece_max_softmax": ece_max_softmax(y_true, y_bf, probs_bf),
        "top3_accuracy": top_k_accuracy(y_true, probs_bf, 3),
        "per_class": pc_bf,
        "confusion_matrix": cm_bf.tolist(),
    }

    print("[4/4] paired tests + per-subtype McNemar + bootstrap ...")
    overall_mc = mcnemar(y_true, y_pn, y_bf)
    paired = {
        "mcnemar_overall": overall_mc,
        "model_agreement_kappa": kappa_models(y_pn, y_bf, K),
        "agreement_pct": float((y_pn == y_bf).mean()),
        "per_subtype_mcnemar": {},
        "per_subtype_acc": {},
    }
    # Per-subtype paired McNemar (within samples of that true class)
    for k, name in enumerate(classes):
        if name == NON_FASTENER:
            continue  # not a fastener subtype
        mask = (y_true == k)
        if mask.sum() == 0:
            continue
        sub_mc = mcnemar(y_true[mask], y_pn[mask], y_bf[mask])
        n_sub = int(mask.sum())
        pn_acc_sub = float((y_pn[mask] == y_true[mask]).mean())
        bf_acc_sub = float((y_bf[mask] == y_true[mask]).mean())
        paired["per_subtype_mcnemar"][name] = sub_mc
        paired["per_subtype_acc"][name] = {
            "n": n_sub, "pn_acc": pn_acc_sub, "bf_acc": bf_acc_sub,
            "diff": pn_acc_sub - bf_acc_sub,
        }

    boot = bootstrap_acc_diff(y_true, y_pn, y_bf, n=2000)

    # Binary-collapse: map all fastener subtypes to "fastener" (=1), non_fastener stays 0
    if non_fastener_idx is not None:
        bin_true = (y_true != non_fastener_idx).astype(int)
        bin_pn = (y_pn != non_fastener_idx).astype(int)
        bin_bf = (y_bf != non_fastener_idx).astype(int)
        binary_collapse = {
            "pn_acc": float((bin_pn == bin_true).mean()),
            "bf_acc": float((bin_bf == bin_true).mean()),
        }
    else:
        binary_collapse = None

    result = {
        "test_set_size": len(shared),
        "num_classes": K,
        "classes": classes,
        "class_distribution": {classes[k]: int((y_true == k).sum()) for k in range(K)},
        "pointnet2_subtype13": pn_metrics,
        "brepformer_subtype13": bf_metrics,
        "paired": paired,
        "bootstrap_2000_acc_diff": boot,
        "binary_collapse": binary_collapse,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\nJSON  -> {OUT_JSON}")

    # ------------- markdown -------------
    def fmt_pct(x): return f"{100*x:.2f}%"
    def fmt_ci(t):  return f"{100*t[0]:.2f}% (95% CI {100*t[1]:.2f}–{100*t[2]:.2f})"

    md = []
    md.append("# Full statistical analysis: PointNet++ vs BRepFormer (13-class subtype)")
    md.append("")
    md.append(f"**Test set:** {len(shared)} samples across {K} classes")
    md.append("")
    md.append("Class distribution (test split):")
    md.append("")
    md.append("| Class | n |")
    md.append("|---|---:|")
    for k, name in enumerate(classes):
        n_k = int((y_true == k).sum())
        md.append(f"| {name} | {n_k} |")
    md.append("")
    md.append("## 1. Per-model headline metrics")
    md.append("")
    md.append("| Metric | PointNet++ | BRepFormer | Δ (PN − BF) |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| Accuracy | {fmt_ci(boot['a_acc'])} | {fmt_ci(boot['b_acc'])} | {100*boot['diff'][0]:+.2f}pp (95% CI {100*boot['diff'][1]:+.2f} to {100*boot['diff'][2]:+.2f}) |")
    md.append(f"| Top-3 accuracy | {fmt_pct(pn_metrics['top3_accuracy'])} | {fmt_pct(bf_metrics['top3_accuracy'])} | {100*(pn_metrics['top3_accuracy']-bf_metrics['top3_accuracy']):+.2f}pp |")
    md.append(f"| Macro F1 | {fmt_pct(pn_metrics['macro_f1'])} | {fmt_pct(bf_metrics['macro_f1'])} | {100*(pn_metrics['macro_f1']-bf_metrics['macro_f1']):+.2f}pp |")
    md.append(f"| Weighted F1 | {fmt_pct(pn_metrics['weighted_f1'])} | {fmt_pct(bf_metrics['weighted_f1'])} | {100*(pn_metrics['weighted_f1']-bf_metrics['weighted_f1']):+.2f}pp |")
    md.append(f"| Balanced accuracy | {fmt_pct(pn_metrics['balanced_accuracy'])} | {fmt_pct(bf_metrics['balanced_accuracy'])} | {100*(pn_metrics['balanced_accuracy']-bf_metrics['balanced_accuracy']):+.2f}pp |")
    md.append(f"| MCC (multiclass) | {pn_metrics['mcc_multiclass']:.4f} | {bf_metrics['mcc_multiclass']:.4f} | {pn_metrics['mcc_multiclass']-bf_metrics['mcc_multiclass']:+.4f} |")
    md.append(f"| Cohen's κ vs random | {pn_metrics['kappa_vs_random']:.4f} | {bf_metrics['kappa_vs_random']:.4f} | {pn_metrics['kappa_vs_random']-bf_metrics['kappa_vs_random']:+.4f} |")
    md.append(f"| AUROC OvR macro | {pn_metrics['auroc_ovr_macro']:.4f} | {bf_metrics['auroc_ovr_macro']:.4f} | {pn_metrics['auroc_ovr_macro']-bf_metrics['auroc_ovr_macro']:+.4f} |")
    md.append(f"| Brier (multiclass) | {pn_metrics['brier_multiclass']:.4f} | {bf_metrics['brier_multiclass']:.4f} | {pn_metrics['brier_multiclass']-bf_metrics['brier_multiclass']:+.4f} (lower=better) |")
    md.append(f"| ECE (max-softmax) | {pn_metrics['ece_max_softmax']:.4f} | {bf_metrics['ece_max_softmax']:.4f} | {pn_metrics['ece_max_softmax']-bf_metrics['ece_max_softmax']:+.4f} (lower=better) |")
    md.append("")
    md.append("## 2. Binary-collapse sanity check")
    md.append("")
    if binary_collapse:
        md.append(f"Mapping all 12 fastener subtypes back to a single `fastener` label and reporting binary fastener-vs-non_fastener:")
        md.append("")
        md.append(f"- PointNet++ binary-collapse acc: **{fmt_pct(binary_collapse['pn_acc'])}**")
        md.append(f"- BRepFormer binary-collapse acc: **{fmt_pct(binary_collapse['bf_acc'])}**")
        md.append("")
        md.append(f"Reference: matched binary classification got PN 94.31% / BF 90.01% on n=558.")
    md.append("")
    md.append("## 3. Per-class metrics")
    md.append("")
    md.append("| Class | n | PN++ P | PN++ R | PN++ F1 | BF P | BF R | BF F1 | F1 Δ |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in classes:
        ppc = pc_pn[name]
        bpc = pc_bf[name]
        md.append(f"| {name} | {ppc['support']} | {fmt_pct(ppc['precision'])} | {fmt_pct(ppc['recall'])} | {fmt_pct(ppc['f1'])} | {fmt_pct(bpc['precision'])} | {fmt_pct(bpc['recall'])} | {fmt_pct(bpc['f1'])} | {100*(ppc['f1']-bpc['f1']):+.2f}pp |")
    md.append("")
    md.append("## 4. Per-class AUROC (one-vs-rest)")
    md.append("")
    md.append("| Class | PN++ AUROC | BF AUROC | Δ |")
    md.append("|---|---:|---:|---:|")
    for k, name in enumerate(classes):
        a_pn = auroc_pn_per[k]
        a_bf = auroc_bf_per[k]
        delta = a_pn - a_bf if (not np.isnan(a_pn) and not np.isnan(a_bf)) else float("nan")
        md.append(f"| {name} | {a_pn:.4f} | {a_bf:.4f} | {delta:+.4f} |")
    md.append("")
    md.append("## 5. Overall paired McNemar test")
    md.append("")
    md.append(f"- Both correct: **{overall_mc['both_correct']}**")
    md.append(f"- PN++ right, BF wrong (b): **{overall_mc['a_only']}**")
    md.append(f"- PN++ wrong, BF right (c): **{overall_mc['b_only']}**")
    md.append(f"- Both wrong: **{overall_mc['both_wrong']}**")
    md.append(f"- **χ² = {overall_mc['chi2']:.4f}, df = 1, p = {overall_mc['p_value']:.6f}**")
    if overall_mc['p_value'] < 0.001:
        sig = "**highly significant (p < 0.001)** — the two models genuinely differ on this 13-class task."
    elif overall_mc['p_value'] < 0.05:
        sig = "**significant at α=0.05** — the two models differ."
    elif overall_mc['p_value'] < 0.1:
        sig = "marginal (p < 0.1)."
    else:
        sig = "not significant — cannot reject equal-error null."
    md.append(f"- Decision: {sig}")
    md.append("")
    md.append(f"**Inter-model agreement:** Cohen's κ = {paired['model_agreement_kappa']:.4f} ({fmt_pct(paired['agreement_pct'])} of predictions match).")
    md.append("")
    md.append("## 6. Per-subtype paired McNemar (12 fastener subtypes)")
    md.append("")
    md.append("Per-fastener-subtype: tests whether PN++ and BF differ on samples whose TRUE class is that subtype. `Δ` is `PN_acc − BF_acc` on those samples (positive → PN wins this subtype).")
    md.append("")
    md.append("| Subtype | n | PN acc | BF acc | Δ | PN-only | BF-only | χ² | p |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    sub_rows = sorted(paired["per_subtype_acc"].items(), key=lambda kv: kv[1]["diff"])
    for name, acc in sub_rows:
        sub_mc = paired["per_subtype_mcnemar"][name]
        md.append(f"| {name} | {acc['n']} | {fmt_pct(acc['pn_acc'])} | {fmt_pct(acc['bf_acc'])} | {100*acc['diff']:+.1f}pp | {sub_mc['a_only']} | {sub_mc['b_only']} | {sub_mc['chi2']:.2f} | {sub_mc['p_value']:.4f} |")
    md.append("")
    md.append("## 7. Confusion matrices")
    md.append("")
    md.append("(Full 13×13 matrices in JSON. Diagonal counts shown here.)")
    md.append("")
    md.append("| Class | n | PN diag | BF diag |")
    md.append("|---|---:|---:|---:|")
    for k, name in enumerate(classes):
        n_k = int((y_true == k).sum())
        md.append(f"| {name} | {n_k} | {int(cm_pn[k,k])} | {int(cm_bf[k,k])} |")
    md.append("")
    md.append("## 8. Headline interpretation")
    md.append("")
    md.append(f"On {len(shared)} shared test samples across {K} classes (12 fastener subtypes + non_fastener), 120 epochs each, matched data:")
    md.append("")
    md.append(f"- **PN++ accuracy:** {fmt_ci(boot['a_acc'])}")
    md.append(f"- **BF accuracy:** {fmt_ci(boot['b_acc'])}")
    md.append(f"- **Difference:** {100*boot['diff'][0]:+.2f}pp (95% CI {100*boot['diff'][1]:+.2f}–{100*boot['diff'][2]:+.2f}pp)")
    md.append(f"- **McNemar p:** {overall_mc['p_value']:.4f}")
    md.append("")
    pn_macro = pn_metrics['macro_f1']
    bf_macro = bf_metrics['macro_f1']
    if abs(pn_macro - bf_macro) < 0.01:
        macro_note = "Macro F1 essentially tied — both models comparably balanced across classes."
    elif bf_macro > pn_macro:
        macro_note = f"BF wins on Macro F1 ({fmt_pct(bf_macro)} vs {fmt_pct(pn_macro)}) — better at the rare classes despite lower overall accuracy."
    else:
        macro_note = f"PN++ wins on Macro F1 ({fmt_pct(pn_macro)} vs {fmt_pct(bf_macro)}) too — uniformly better, not just on the easy classes."
    md.append(macro_note)
    md.append("")
    md.append(f"AUROC OvR macro: PN++ {pn_metrics['auroc_ovr_macro']:.3f} vs BF {bf_metrics['auroc_ovr_macro']:.3f}.")
    md.append("")
    md.append("Per-subtype results (Section 6) reveal where each architecture excels — the per-subtype McNemar tests highlight subtypes where the gap is statistically real even at the small per-class sample counts.")
    md.append("")
    md.append("## 9. Artifact provenance")
    md.append("")
    md.append(f"- PN++ checkpoint: latest in `training_data/mcmaster_logs/pointnet2_subtype13/`")
    md.append(f"- BF checkpoint:   lowest val_loss in `training_data/mcmaster_logs/brepformer_subtype13/checkpoints/`")
    md.append(f"- PN test data:    `training_data/mcmaster_pc_subtype13/test/`")
    md.append(f"- BF test data:    `training_data/mcmaster_brep_subtype13/test/`")
    md.append(f"- Numbers JSON:    `training_data/mcmaster_logs/full_analysis_subtype13.json`")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"MD    -> {OUT_MD}")


if __name__ == "__main__":
    main()
