"""Full statistical analysis: PointNet++ matched vs BRepFormer Plan C'.

Both models evaluated on the same 559-sample held-out test set
(mcmaster_pc_breponly/test/  ↔  mcmaster_brep/test/  — identical part-numbers).

Computes, per model:
  - Accuracy with bootstrap 95% CI
  - Per-class precision, recall, F1
  - Macro F1, balanced accuracy
  - Matthews Correlation Coefficient (MCC) — robust to class imbalance
  - Cohen's kappa vs random
  - AUROC, Brier score, Expected Calibration Error (ECE) — needs softmax probs

Computes, paired:
  - McNemar's test with continuity correction (χ² + p-value)
  - Cohen's kappa between the two models (model-model agreement)
  - 4-cell agreement matrix (both right, only A right, only B right, both wrong)
  - Bootstrap 95% CI for the accuracy difference

Per-category breakdown (uses dataset/{fastener,non_fastener}/<cat>/<leaf>/<pn>.step):
  - Per-category accuracy for each model

Outputs:
  - training_data/mcmaster_logs/full_analysis.json (machine-readable)
  - training_data/mcmaster_logs/full_analysis.md (human-readable report)
"""
from __future__ import annotations
from pathlib import Path
import json
import sys
from collections import Counter, defaultdict
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

DATASET_ROOT = REPO / "fastener_labeling" / "dataset"
PC_TEST = REPO / "training_data" / "mcmaster_pc_breponly"
BREP_TEST = REPO / "training_data" / "mcmaster_brep"
PN_CKPT = REPO / "training_data" / "mcmaster_logs" / "pointnet2_breponly" / "best_model.pth"
BF_CKPT = REPO / "training_data" / "mcmaster_logs" / "bf_planc_prime_best.ckpt"
OUT_JSON = REPO / "training_data" / "mcmaster_logs" / "full_analysis.json"
OUT_MD   = REPO / "training_data" / "mcmaster_logs" / "full_analysis.md"

CLASSES = ["fastener", "non_fastener"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------- predictions ---------------

def collect_pn(num_points: int = 4096) -> dict[str, dict]:
    """Run PN++ matched on test set; return {pn: {true, pred, p1, klass}}."""
    ds = FastenerPointCloudDataset(root=str(PC_TEST), num_points=num_points,
                                   use_normals=True, split="test", augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    model = PointNet2ClsMSG(num_classes=ds.num_classes, use_normals=True).to(DEVICE)
    ck = torch.load(PN_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.eval()

    out: dict[str, dict] = {}
    sample_idx = 0
    with torch.no_grad():
        for points, labels in loader:
            points_t = points.transpose(1, 2).to(DEVICE)
            labels_dev = labels.to(DEVICE)
            logits, _ = model(points_t)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(-1)
            for j in range(labels.size(0)):
                fp, lab = ds.samples[sample_idx + j]
                pn = Path(fp).stem
                out[pn] = {
                    "true": int(lab),
                    "pred": int(preds[j].item()),
                    "p1":   float(probs[j, 1].item()),
                    "klass": ds.classes[int(lab)],
                }
            sample_idx += labels.size(0)
    return out


def collect_bf() -> dict[str, dict]:
    """Run BF Plan C' on test set; return {pn: {true, pred, p1, klass}}."""
    ds = BRepDataset(root=str(BREP_TEST), split="test")
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                        collate_fn=brep_collate_fn)
    model = BRepFormer(num_classes=ds.num_classes).to(DEVICE)
    ck = torch.load(BF_CKPT, map_location=DEVICE, weights_only=False)
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
            B = labels.size(0)
            for j in range(B):
                model_dir, lab = ds.samples[sample_idx + j]
                pn = Path(model_dir).name
                out[pn] = {
                    "true": int(lab),
                    "pred": int(preds[j].item()),
                    "p1":   float(probs[j, 1].item()),
                    "klass": ds.classes[int(lab)],
                }
            sample_idx += B
    return out


# --------------- metrics ---------------

def confusion(y_true, y_pred):
    return {
        "tp_fast": int(((y_true == 0) & (y_pred == 0)).sum()),  # fastener (0) recalled
        "fn_fast": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tp_non":  int(((y_true == 1) & (y_pred == 1)).sum()),
        "fn_non":  int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def per_class_metrics(y_true, y_pred):
    cm = confusion(y_true, y_pred)
    out = {}
    for ki, kname in enumerate(CLASSES):
        if kname == "fastener":
            tp, fn, fp = cm["tp_fast"], cm["fn_fast"], cm["fn_non"]
        else:
            tp, fn, fp = cm["tp_non"], cm["fn_non"], cm["fn_fast"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[kname] = {"tp": tp, "fn": fn, "fp": fp,
                      "recall": recall, "precision": precision, "f1": f1}
    return out


def mcc(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def cohen_kappa_vs_random(y_true, y_pred):
    n = len(y_true)
    po = (y_pred == y_true).sum() / n
    p_pred1 = y_pred.mean()
    p_true1 = y_true.mean()
    pe = p_pred1 * p_true1 + (1 - p_pred1) * (1 - p_true1)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def cohen_kappa_models(pred_a, pred_b):
    n = len(pred_a)
    po = (pred_a == pred_b).sum() / n
    pa1 = pred_a.mean(); pb1 = pred_b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def auroc(y_true, p1):
    """ROC-AUC via Mann-Whitney rank-sum (no sklearn dependency)."""
    pos = p1[y_true == 1]
    neg = p1[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_scores = np.concatenate([pos, neg])
    ranks = all_scores.argsort().argsort() + 1
    rank_sum_pos = ranks[: len(pos)].sum()
    auc = (rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def brier(y_true, p1):
    return float(((p1 - y_true) ** 2).mean())


def ece(y_true, y_pred, p_pred_class, n_bins=10):
    """Expected calibration error: |accuracy − confidence| per bin, weighted.

    `p_pred_class` should be the model's confidence in its own prediction
    (= max softmax). Per bin we compare the empirical correctness rate to
    the average reported confidence.
    """
    correct = (y_pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(y_true)
    ece_val = 0.0
    for i in range(n_bins):
        in_bin = (p_pred_class > bins[i]) & (p_pred_class <= bins[i + 1])
        if in_bin.sum() == 0:
            continue
        acc_bin = correct[in_bin].mean()
        conf_bin = p_pred_class[in_bin].mean()
        ece_val += (in_bin.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece_val)


def mcnemar(y_true, pred_a, pred_b):
    """Paired-classifier test with continuity correction."""
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)
    a_only = int((a_correct & (~b_correct)).sum())   # A right, B wrong
    b_only = int(((~a_correct) & b_correct).sum())   # A wrong, B right
    both_correct = int((a_correct & b_correct).sum())
    both_wrong = int(((~a_correct) & (~b_correct)).sum())
    n_disagree = a_only + b_only
    if n_disagree == 0:
        return {"a_only": 0, "b_only": 0, "both_correct": both_correct,
                "both_wrong": both_wrong, "chi2": 0.0, "p_value": 1.0}
    chi2 = ((abs(a_only - b_only) - 1) ** 2) / n_disagree
    # 1-df chi2 survival via series approximation (no scipy needed):
    # P(X > t) = erfc(sqrt(t/2)) for chi2_1
    from math import erfc, sqrt
    p_value = float(erfc(sqrt(chi2 / 2.0)))
    return {"a_only": a_only, "b_only": b_only, "both_correct": both_correct,
            "both_wrong": both_wrong, "chi2": float(chi2), "p_value": p_value}


def bootstrap_ci(values_pn, values_bf, y_true, n=2000, seed=42):
    """Bootstrap distribution of (acc_pn, acc_bf, acc_pn - acc_bf)."""
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    accs_pn = np.empty(n); accs_bf = np.empty(n); diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        accs_pn[i] = (values_pn[idx] == y_true[idx]).mean()
        accs_bf[i] = (values_bf[idx] == y_true[idx]).mean()
        diffs[i] = accs_pn[i] - accs_bf[i]
    def ci(arr):
        return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return {
        "pn_acc": ci(accs_pn),
        "bf_acc": ci(accs_bf),
        "diff":   ci(diffs),
    }


# --------------- per-category index ---------------

def build_pn_to_category() -> dict[str, dict]:
    """Walk dataset/{fastener,non_fastener}/<cat>/<leaf>/<pn>.step to map pn -> (class, cat, leaf)."""
    idx = {}
    for klass_dir in ("fastener", "non_fastener"):
        root = DATASET_ROOT / klass_dir
        if not root.exists():
            continue
        for cat_dir in root.iterdir():
            if not cat_dir.is_dir():
                continue
            for leaf in cat_dir.rglob("*.step"):
                if leaf.name.startswith("._"):
                    continue
                idx[leaf.stem] = {"class": klass_dir, "category": cat_dir.name,
                                  "leaf": leaf.parent.name}
    return idx


# --------------- main ---------------

def main():
    print("[1/5] running PointNet++ matched on test set ...")
    pn = collect_pn()
    print(f"  {len(pn)} predictions")

    print("[2/5] running BRepFormer Plan C' on test set ...")
    bf = collect_bf()
    print(f"  {len(bf)} predictions")

    shared = sorted(set(pn.keys()) & set(bf.keys()))
    print(f"[3/5] {len(shared)} shared part-numbers (PN-only={len(pn)-len(shared)}, BF-only={len(bf)-len(shared)})")

    y_true = np.array([pn[k]["true"] for k in shared])
    y_pn = np.array([pn[k]["pred"] for k in shared])
    y_bf = np.array([bf[k]["pred"] for k in shared])
    p1_pn = np.array([pn[k]["p1"]   for k in shared])
    p1_bf = np.array([bf[k]["p1"]   for k in shared])

    # Per-model
    pn_metrics = {
        "n": len(shared),
        "accuracy": float((y_pn == y_true).mean()),
        "mcc": mcc(y_true, y_pn),
        "cohen_kappa_vs_random": cohen_kappa_vs_random(y_true, y_pn),
        "balanced_accuracy": ((y_pn[y_true == 0] == 0).mean() + (y_pn[y_true == 1] == 1).mean()) / 2,
        "per_class": per_class_metrics(y_true, y_pn),
        "auroc": auroc(y_true, p1_pn),
        "brier_score": brier(y_true, p1_pn),
        "ece": ece(y_true, y_pn, np.where(y_pn == 1, p1_pn, 1 - p1_pn)),
    }
    bf_metrics = {
        "n": len(shared),
        "accuracy": float((y_bf == y_true).mean()),
        "mcc": mcc(y_true, y_bf),
        "cohen_kappa_vs_random": cohen_kappa_vs_random(y_true, y_bf),
        "balanced_accuracy": ((y_bf[y_true == 0] == 0).mean() + (y_bf[y_true == 1] == 1).mean()) / 2,
        "per_class": per_class_metrics(y_true, y_bf),
        "auroc": auroc(y_true, p1_bf),
        "brier_score": brier(y_true, p1_bf),
        "ece": ece(y_true, y_bf, np.where(y_bf == 1, p1_bf, 1 - p1_bf)),
    }
    pn_metrics["macro_f1"] = (pn_metrics["per_class"]["fastener"]["f1"]
                              + pn_metrics["per_class"]["non_fastener"]["f1"]) / 2
    bf_metrics["macro_f1"] = (bf_metrics["per_class"]["fastener"]["f1"]
                              + bf_metrics["per_class"]["non_fastener"]["f1"]) / 2

    # Paired
    print("[4/5] paired tests ...")
    mc = mcnemar(y_true, y_pn, y_bf)
    paired = {
        "mcnemar": mc,
        "model_agreement_kappa": cohen_kappa_models(y_pn, y_bf),
        "agreement_pct": float((y_pn == y_bf).mean()),
    }
    boot = bootstrap_ci(y_pn, y_bf, y_true, n=2000)

    # Per-category
    print("[5/5] per-category breakdown ...")
    pn_to_cat = build_pn_to_category()
    by_cat: dict[str, dict] = {}
    for k in shared:
        cat = pn_to_cat.get(k, {}).get("category", "_unknown")
        klass = pn_to_cat.get(k, {}).get("class", "_unknown")
        by_cat.setdefault(cat, {"klass": klass, "n": 0, "pn_correct": 0, "bf_correct": 0})
        by_cat[cat]["n"] += 1
        if pn[k]["pred"] == pn[k]["true"]:
            by_cat[cat]["pn_correct"] += 1
        if bf[k]["pred"] == bf[k]["true"]:
            by_cat[cat]["bf_correct"] += 1
    for cat, d in by_cat.items():
        d["pn_acc"] = d["pn_correct"] / max(d["n"], 1)
        d["bf_acc"] = d["bf_correct"] / max(d["n"], 1)
        d["diff_pn_minus_bf"] = d["pn_acc"] - d["bf_acc"]

    result = {
        "test_set_size": len(shared),
        "class_distribution": {
            "fastener":     int((y_true == 0).sum()),
            "non_fastener": int((y_true == 1).sum()),
        },
        "pointnet2_matched": pn_metrics,
        "brepformer_planc_prime": bf_metrics,
        "paired": paired,
        "bootstrap_2000": boot,
        "per_category": by_cat,
        "checkpoints": {
            "pointnet2": str(PN_CKPT),
            "brepformer": str(BF_CKPT),
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\nJSON  -> {OUT_JSON}")

    # ---------- markdown report ----------
    def fmt_pct(x):  return f"{100*x:.2f}%"
    def fmt_ci(t):   return f"{100*t[0]:.2f}% (95% CI {100*t[1]:.2f}–{100*t[2]:.2f})"

    md = []
    md.append("# Full statistical analysis: PointNet++ matched vs BRepFormer Plan C'")
    md.append("")
    md.append(f"**Test set:** {len(shared)} samples — fasteners: {(y_true==0).sum()}, non-fasteners: {(y_true==1).sum()}")
    md.append("")
    md.append("Both models evaluated on the **same** held-out test set (shared part-numbers across "
              "`mcmaster_pc_breponly/test/` and `mcmaster_brep/test/`). Identical 559-part comparison; "
              "no PN++ data-availability advantage in this analysis.")
    md.append("")
    md.append("## 1. Per-model metrics (with bootstrap 95% CI on accuracy)")
    md.append("")
    md.append("| Metric | PointNet++ | BRepFormer | Δ (PN − BF) |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| Accuracy | {fmt_ci(boot['pn_acc'])} | {fmt_ci(boot['bf_acc'])} | "
              f"{100*boot['diff'][0]:+.2f}pp (95% CI {100*boot['diff'][1]:+.2f} to {100*boot['diff'][2]:+.2f}) |")
    md.append(f"| Balanced accuracy | {fmt_pct(pn_metrics['balanced_accuracy'])} "
              f"| {fmt_pct(bf_metrics['balanced_accuracy'])} "
              f"| {100*(pn_metrics['balanced_accuracy']-bf_metrics['balanced_accuracy']):+.2f}pp |")
    md.append(f"| Macro F1 | {fmt_pct(pn_metrics['macro_f1'])} | {fmt_pct(bf_metrics['macro_f1'])} "
              f"| {100*(pn_metrics['macro_f1']-bf_metrics['macro_f1']):+.2f}pp |")
    md.append(f"| MCC | {pn_metrics['mcc']:.4f} | {bf_metrics['mcc']:.4f} "
              f"| {pn_metrics['mcc']-bf_metrics['mcc']:+.4f} |")
    md.append(f"| Cohen's κ vs random | {pn_metrics['cohen_kappa_vs_random']:.4f} "
              f"| {bf_metrics['cohen_kappa_vs_random']:.4f} "
              f"| {pn_metrics['cohen_kappa_vs_random']-bf_metrics['cohen_kappa_vs_random']:+.4f} |")
    md.append(f"| AUROC | {pn_metrics['auroc']:.4f} | {bf_metrics['auroc']:.4f} "
              f"| {pn_metrics['auroc']-bf_metrics['auroc']:+.4f} |")
    md.append(f"| Brier score (lower=better) | {pn_metrics['brier_score']:.4f} | {bf_metrics['brier_score']:.4f} "
              f"| {pn_metrics['brier_score']-bf_metrics['brier_score']:+.4f} |")
    md.append(f"| ECE (lower=better) | {pn_metrics['ece']:.4f} | {bf_metrics['ece']:.4f} "
              f"| {pn_metrics['ece']-bf_metrics['ece']:+.4f} |")
    md.append("")
    md.append("## 2. Per-class metrics")
    md.append("")
    md.append("| Class | Metric | PN++ | BF | Δ |")
    md.append("|---|---|---:|---:|---:|")
    for k in CLASSES:
        for met in ("recall", "precision", "f1"):
            v_pn = pn_metrics["per_class"][k][met]
            v_bf = bf_metrics["per_class"][k][met]
            md.append(f"| {k} | {met} | {fmt_pct(v_pn)} | {fmt_pct(v_bf)} | {100*(v_pn-v_bf):+.2f}pp |")
    md.append("")
    md.append("## 3. Paired comparison — McNemar's test with continuity correction")
    md.append("")
    md.append(f"- **Both correct**: {mc['both_correct']}")
    md.append(f"- **PN++ right, BF wrong** (b): {mc['a_only']}")
    md.append(f"- **PN++ wrong, BF right** (c): {mc['b_only']}")
    md.append(f"- **Both wrong**: {mc['both_wrong']}")
    md.append(f"- **χ² = {mc['chi2']:.4f}, df = 1, p = {mc['p_value']:.6f}**")
    if mc['p_value'] < 0.001:
        sig = "**highly significant (p < 0.001)** — the two models genuinely differ."
    elif mc['p_value'] < 0.05:
        sig = "**significant at α=0.05** — the two models differ."
    elif mc['p_value'] < 0.1:
        sig = "marginal (p < 0.1) — suggestive but not conclusive."
    else:
        sig = "not significant — cannot reject the null that the two models perform equally."
    md.append(f"- Decision: {sig}")
    md.append("")
    md.append(f"**Inter-model agreement (Cohen's κ on predictions):** {paired['model_agreement_kappa']:.4f}  "
              f"({fmt_pct(paired['agreement_pct'])} of predictions match between PN++ and BF)")
    md.append("")
    md.append("## 4. Per-McMaster-category breakdown")
    md.append("")
    md.append("Sorted by sample count. `Δ = PN_acc − BF_acc` (positive = PN++ wins on this category).")
    md.append("")
    md.append("| Class | Category | n | PN++ acc | BF acc | Δ |")
    md.append("|---|---|---:|---:|---:|---:|")
    rows = sorted(by_cat.items(), key=lambda kv: -kv[1]["n"])
    for cat, d in rows:
        md.append(f"| {d['klass']} | {cat} | {d['n']} | {fmt_pct(d['pn_acc'])} "
                  f"| {fmt_pct(d['bf_acc'])} | {100*d['diff_pn_minus_bf']:+.1f}pp |")
    md.append("")
    md.append("## 5. Headline interpretation")
    md.append("")
    md.append(f"On **identical 559-sample** test set, identical 5,710-sample training set, identical 120 epochs and matched hyperparameters:")
    md.append("")
    md.append(f"- PointNet++ matched: {fmt_ci(boot['pn_acc'])}")
    md.append(f"- BRepFormer Plan C': {fmt_ci(boot['bf_acc'])}")
    md.append(f"- **Difference: {100*boot['diff'][0]:+.2f}pp** with 95% CI {100*boot['diff'][1]:+.2f}–{100*boot['diff'][2]:+.2f}pp")
    md.append("")
    md.append(f"The 95% CI for the difference {'excludes' if boot['diff'][1] > 0 or boot['diff'][2] < 0 else 'includes'} zero, "
              f"and the McNemar test gives p={mc['p_value']:.4f}. The result is {sig.replace('**', '')}")
    md.append("")
    md.append(f"Inter-model agreement is {fmt_pct(paired['agreement_pct'])} (κ={paired['model_agreement_kappa']:.3f}). "
              f"They disagree on {mc['a_only']+mc['b_only']} of {len(shared)} samples; on those disagreements, "
              f"PN++ is correct in {mc['a_only']} cases and BF is correct in {mc['b_only']} cases.")
    md.append("")
    md.append("## 6. Artifact provenance")
    md.append("")
    md.append(f"- PN++ checkpoint: `{PN_CKPT.relative_to(REPO)}`")
    md.append(f"- BF checkpoint:   `{BF_CKPT.relative_to(REPO)}`")
    md.append(f"- PN++ test data:  `training_data/mcmaster_pc_breponly/test/`")
    md.append(f"- BF test data:    `training_data/mcmaster_brep/test/`")
    md.append(f"- Numbers JSON:    `{OUT_JSON.relative_to(REPO)}`")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"MD    -> {OUT_MD}")


if __name__ == "__main__":
    main()
