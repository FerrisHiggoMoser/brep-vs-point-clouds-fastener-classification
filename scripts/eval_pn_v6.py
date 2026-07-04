"""Evaluate PointNet++ v6 (architecture comparison vs BRepFormer v6).

Phases (run separately; each is resumable / idempotent):

  bf-dump   Run the FROZEN bf_subtype13_v6.ckpt (inference only, never
            retrained) over the v6 test split (max_faces=300, the stored
            eval's filter) and write per-sample predictions — the stored
            v6 eval only saved aggregates, and McNemar needs pairs.
            Sanity-checked against the stored eval_summary.json accuracy.

  test      PointNet++ on pn_v6_features/test: 13-class + binary-collapse
            metrics, per-sample predictions CSV, McNemar + paired bootstrap
            vs the bf-dump on shared samples (name-mismatch samples
            excluded from pairing — solid identity unverified).

  paddle    PointNet++ on the PADDLE STEAMER 350-part name-labeled holdout
            (names are ground truth ONLY, never model input): overall +
            per-class, McNemar vs bf_v6_run/paddle_steamer_v6/predictions.csv
            matched by part_name.

  report    Merge everything into eval_report.md + eval_summary.json.

Checkpoint selection is by the val_loss token in the FILENAME (lowest),
never by mtime (2026-05-09 bug).

Usage (anaconda stepvrstep env):
    python backend/scripts/eval_pn_v6.py --phase bf-dump
    python backend/scripts/eval_pn_v6.py --phase test
    python backend/scripts/eval_pn_v6.py --phase paddle
    python backend/scripts/eval_pn_v6.py --phase report
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings

warnings.filterwarnings("ignore")
from collections import defaultdict
from math import erfc, sqrt
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import numpy as np

REPO = _BACKEND_ROOT.parent
D_THESIS = Path(r"D:\step-vr-step-thesis")
RB_TD = D_THESIS / "reproducible-build" / "training_data"

V6_FEATURES = RB_TD / "bf_v2_features"
PN_FEATURES = RB_TD / "pn_v6_features"
RUN_DIR = RB_TD / "pn_v6_run"
CKPT_DIR = RUN_DIR / "checkpoints"
BF_V6_CKPT = D_THESIS / "reproducible-build" / "models" / "bf_subtype13_v6.ckpt"
BF_V6_EVAL_SUMMARY = RB_TD / "bf_v6_run" / "eval_summary.json"
BF_PADDLE_CSV = RB_TD / "bf_v6_run" / "paddle_steamer_v6" / "predictions.csv"
PADDLE_STEP = D_THESIS / "fastener_labeling" / "files" / "PADDLE STEAMER.STEP"
PADDLE_HOLDOUT = RB_TD / "paddle_steamer_holdout"

BF_DUMP_CSV = RUN_DIR / "bf_v6_test_predictions.csv"
PN_TEST_CSV = RUN_DIR / "pn_v6_test_predictions.csv"
PADDLE_CSV = RUN_DIR / "paddle_predictions.csv"
PADDLE_PC_DIR = RUN_DIR / "paddle_pc"

V6_MAX_FACES = 300
NUM_POINTS = 4096

CLASSES13 = [
    "anchors", "keys", "nails", "non_fastener", "nuts", "pins",
    "retaining-rings", "rivets", "screws", "spacers",
    "threaded-inserts", "threaded-rods", "washers",
]
FASTENER_CLASSES = [c for c in CLASSES13 if c != "non_fastener"]


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:80]


# ---------------------------------------------------------------------------
# Stats (same math as full_analysis.py / eval_rule_based.py)
# ---------------------------------------------------------------------------

def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    both_c = int((correct_a & correct_b).sum())
    both_w = int((~correct_a & ~correct_b).sum())
    nd = a_only + b_only
    if nd == 0:
        return {"a_only": 0, "b_only": 0, "both_correct": both_c,
                "both_wrong": both_w, "chi2": 0.0, "p_value": 1.0}
    chi2 = ((abs(a_only - b_only) - 1) ** 2) / nd
    return {"a_only": a_only, "b_only": b_only, "both_correct": both_c,
            "both_wrong": both_w, "chi2": float(chi2),
            "p_value": float(erfc(sqrt(chi2 / 2.0)))}


def paired_bootstrap_diff(correct_a: np.ndarray, correct_b: np.ndarray,
                          n_boot: int = 2000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()
    return {"mean": float(diffs.mean()),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


def multi13_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES13, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES13)
    return {
        "n": len(y_true),
        "accuracy": acc,
        "macro_f1": float(f1.mean()),
        "per_class": [
            {"class": CLASSES13[i], "precision": float(prec[i]),
             "recall": float(rec[i]), "f1": float(f1[i]),
             "support": int(support[i])}
            for i in range(len(CLASSES13))],
        "confusion_matrix": cm.tolist(),
    }


def binary_collapse_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    bt = ["fastener" if t != "non_fastener" else "non_fastener" for t in y_true]
    bp = ["fastener" if p != "non_fastener" else "non_fastener" for p in y_pred]
    n = len(bt)
    acc = sum(t == p for t, p in zip(bt, bp)) / max(n, 1)
    out = {"n": n, "accuracy": acc, "per_class": {}}
    for cls in ("fastener", "non_fastener"):
        idx = [i for i, t in enumerate(bt) if t == cls]
        tp = sum(1 for i in idx if bp[i] == cls)
        pred_cls = sum(1 for p_ in bp if p_ == cls)
        out["per_class"][cls] = {
            "support": len(idx),
            "recall": tp / max(len(idx), 1),
            "precision": tp / max(pred_cls, 1),
        }
    return out


# ---------------------------------------------------------------------------
# Phase: bf-dump
# ---------------------------------------------------------------------------

def phase_bf_dump(batch_size: int = 4):
    import torch
    from torch.utils.data import DataLoader, Subset
    from step_vr_step.models.brepformer.brepformer import BRepFormer
    from step_vr_step.models.brepformer.dataset import BRepDataset, brep_collate_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(BF_V6_CKPT, map_location=device, weights_only=False)
    assert "model_state_dict" in ckpt, "expected v6 dict-format checkpoint"
    class_names = list(ckpt["class_names"])
    assert class_names == CLASSES13, f"ckpt class order mismatch: {class_names}"

    model = BRepFormer(num_classes=13, dim=256, num_layers=8,
                       head_mode="classification").to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"WARN load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    ds = BRepDataset(root=V6_FEATURES, split="test")
    assert ds.classes == CLASSES13, f"dataset class order mismatch: {ds.classes}"
    # repo BRepDataset has no max_faces arg; apply the exact v6 filter here
    keep = []
    for i, (model_dir, _) in enumerate(ds.samples):
        try:
            if np.load(model_dir / "face_grids.npy", mmap_mode="r").shape[0] <= V6_MAX_FACES:
                keep.append(i)
        except Exception:
            pass
    print(f"v6 test: {len(keep)} of {len(ds.samples)} dirs pass the "
          f"max_faces={V6_MAX_FACES} filter", flush=True)

    loader = DataLoader(Subset(ds, keep), batch_size=batch_size, shuffle=False,
                        num_workers=2, collate_fn=brep_collate_fn)
    preds, labels = [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            face_grids = batch["face_grids"].to(device)
            edge_curves = batch["edge_curves"].to(device)
            topo = {k: v.to(device) for k, v in batch["topo_distances"].items()}
            face_mask = batch["face_mask"].to(device)
            edge_mask = batch["edge_mask"].to(device)
            logits = model(face_grids, edge_curves, topo, face_mask, edge_mask)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
            if (bi + 1) % 200 == 0:
                print(f"  batch {bi + 1}/{len(loader)}", flush=True)

    rows = []
    for j, i in enumerate(keep):
        model_dir, cls_idx = ds.samples[i]
        sample_id = f"test/{model_dir.parent.name}/{model_dir.name}"
        rows.append({"sample_id": sample_id,
                     "gt_subtype13": CLASSES13[labels[j]],
                     "pred_subtype13": CLASSES13[preds[j]]})
        assert CLASSES13[cls_idx] == model_dir.parent.name

    with BF_DUMP_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "gt_subtype13", "pred_subtype13"])
        w.writeheader()
        w.writerows(rows)

    acc = sum(r["gt_subtype13"] == r["pred_subtype13"] for r in rows) / len(rows)
    print(f"BF v6 test accuracy (re-run): {acc:.4f} on n={len(rows)}", flush=True)
    stored = json.loads(BF_V6_EVAL_SUMMARY.read_text(encoding="utf-8"))
    for key, blob in stored.items():
        if "test" in key and "full" in key:
            print(f"stored eval [{key}]: acc={blob['accuracy']:.4f} n={blob['n']}", flush=True)
            if abs(blob["accuracy"] - acc) > 0.005:
                print("WARN: re-run accuracy deviates >0.5pp from stored eval — "
                      "check filter/order before trusting the pairing!", flush=True)
    print(f"wrote {BF_DUMP_CSV}", flush=True)


# ---------------------------------------------------------------------------
# PN++ inference helpers
# ---------------------------------------------------------------------------

def find_best_checkpoint() -> Path:
    """Lowest val_loss by FILENAME token (never mtime)."""
    pat = re.compile(r"best-epoch=(\d+)-val_loss=([0-9.]+)\.pth$")
    best, best_loss = None, float("inf")
    for f in CKPT_DIR.glob("best-epoch=*-val_loss=*.pth"):
        m = pat.match(f.name)
        if not m:
            continue
        loss = float(m.group(2).rstrip("."))
        if loss < best_loss:
            best, best_loss = f, loss
    if best is None:
        raise FileNotFoundError(f"no best-epoch=*-val_loss=*.pth in {CKPT_DIR}")
    print(f"checkpoint (lowest val_loss token): {best.name}", flush=True)
    return best


def load_pn_model(device):
    import torch
    from step_vr_step.models.pointnet2.pointnet2_cls_msg import PointNet2ClsMSG

    ckpt_path = find_best_checkpoint()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = list(ckpt["class_names"])
    assert class_names == CLASSES13, f"ckpt class order mismatch: {class_names}"
    model = PointNet2ClsMSG(num_classes=len(class_names), use_normals=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt_path


def normalize_pc(data: np.ndarray) -> np.ndarray:
    """FastenerPointCloudDataset eval-time transform: center + unit sphere,
    keep normals. Input/output (N, 6)."""
    points = data[:, :3].astype(np.float32)
    normals = data[:, 3:6].astype(np.float32)
    points = points - points.mean(axis=0)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 1e-12:
        points = points / scale
    return np.concatenate([points, normals], axis=1)


def predict_batch(model, arrs: list[np.ndarray], device) -> tuple[list[int], list[float]]:
    import torch
    x = torch.from_numpy(np.stack(arrs)).transpose(1, 2).to(device)
    with torch.no_grad():
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
    return pred.cpu().tolist(), conf.cpu().tolist()


# ---------------------------------------------------------------------------
# Phase: test
# ---------------------------------------------------------------------------

def load_name_mismatch_ids() -> set[str]:
    """Samples whose extracted solid name didn't match the feature-dir name
    (same-named source file ambiguity) — excluded from PAIRED analysis."""
    out = set()
    log = RUN_DIR / "extraction_log.csv"
    if log.exists():
        with log.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("name_mismatch") in ("1", "True"):
                    out.add(r["sample_id"])
    return out


def phase_test(batch_size: int = 32):
    import torch
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_path = load_pn_model(device)

    files = []
    for cls in CLASSES13:
        cls_dir = PN_FEATURES / "test" / cls
        if not cls_dir.exists():
            continue
        for f in sorted(cls_dir.glob("*.npy")):
            files.append((f"test/{cls}/{f.stem}", cls, f))
    print(f"PN v6 test: {len(files)} samples", flush=True)

    rows = []
    buf, meta = [], []

    def flush():
        nonlocal buf, meta
        if not buf:
            return
        preds, confs = predict_batch(model, buf, device)
        for (sid, cls), p, cf in zip(meta, preds, confs):
            rows.append({"sample_id": sid, "gt_subtype13": cls,
                         "pred_subtype13": CLASSES13[p],
                         "confidence": round(cf, 4)})
        buf, meta = [], []

    for sid, cls, f in files:
        try:
            data = np.load(f)
        except Exception as e:
            rows.append({"sample_id": sid, "gt_subtype13": cls,
                         "pred_subtype13": "", "confidence": "",
                         })
            print(f"  WARN unreadable {sid}: {e}", flush=True)
            continue
        buf.append(normalize_pc(data))
        meta.append((sid, cls))
        if len(buf) >= batch_size:
            flush()
        if len(rows) % 1000 < batch_size and rows:
            pass
    flush()

    with PN_TEST_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "gt_subtype13",
                                          "pred_subtype13", "confidence"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {PN_TEST_CSV}", flush=True)

    scored = [r for r in rows if r["pred_subtype13"]]
    y_true = [r["gt_subtype13"] for r in scored]
    y_pred = [r["pred_subtype13"] for r in scored]
    result = {
        "checkpoint": ckpt_path.name,
        "multi13": multi13_metrics(y_true, y_pred),
        "binary_collapse": binary_collapse_metrics(y_true, y_pred),
    }
    print(f"PN++ v6 test: acc={result['multi13']['accuracy']:.4f} "
          f"macro_f1={result['multi13']['macro_f1']:.4f} "
          f"binary={result['binary_collapse']['accuracy']:.4f} "
          f"n={len(scored)}", flush=True)

    # ---- paired comparison vs BF dump ----
    if BF_DUMP_CSV.exists():
        bf = {}
        with BF_DUMP_CSV.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                bf[r["sample_id"]] = r
        mism = load_name_mismatch_ids()
        pn_c, bf_c = [], []
        n_mism_excluded = 0
        for r in scored:
            b = bf.get(r["sample_id"])
            if b is None:
                continue
            if r["sample_id"] in mism:
                n_mism_excluded += 1
                continue
            pn_c.append(r["pred_subtype13"] == r["gt_subtype13"])
            bf_c.append(b["pred_subtype13"] == b["gt_subtype13"])
        pn_c = np.array(pn_c, dtype=bool)
        bf_c = np.array(bf_c, dtype=bool)
        result["paired_vs_bf_v6"] = {
            "n_shared": int(len(pn_c)),
            "n_name_mismatch_excluded": n_mism_excluded,
            "pn_accuracy_shared": float(pn_c.mean()),
            "bf_accuracy_shared": float(bf_c.mean()),
            "mcnemar_pn_vs_bf": mcnemar(pn_c, bf_c),
            "bootstrap_diff_pn_minus_bf": paired_bootstrap_diff(pn_c, bf_c),
        }
        m = result["paired_vs_bf_v6"]
        print(f"paired n={m['n_shared']}: PN {m['pn_accuracy_shared']:.4f} "
              f"vs BF {m['bf_accuracy_shared']:.4f}  "
              f"McNemar chi2={m['mcnemar_pn_vs_bf']['chi2']:.2f} "
              f"p={m['mcnemar_pn_vs_bf']['p_value']:.2e}", flush=True)
    else:
        print("WARN: bf-dump CSV missing — run --phase bf-dump for McNemar", flush=True)

    (RUN_DIR / "test_eval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {RUN_DIR / 'test_eval.json'}", flush=True)


# ---------------------------------------------------------------------------
# Phase: paddle
# ---------------------------------------------------------------------------

def build_paddle_tasks() -> list[dict]:
    """From paddle_steamer_holdout dirs: class, solid index, iso350 flag.
    Names are ground truth ONLY — the model never sees them."""
    tasks = []
    for cls_dir in sorted(PADDLE_HOLDOUT.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in CLASSES13:
            continue
        for d in sorted(cls_dir.iterdir()):
            parts = d.name.split("__")
            if len(parts) < 4:  # realcad__PADDLE_STEAMER__<solid>__<idx>
                continue
            solid_basename = "__".join(parts[2:-1])
            idx = int(parts[-1])
            in350 = not solid_basename.startswith("PADDLE_MASTER_ROD_PIVOT_BOLT")
            tasks.append({
                "sample_id": f"{cls_dir.name}/{d.name}",
                "solid_idx": idx,
                "expected_solid_name": solid_basename,
                "gt_subtype13": cls_dir.name,
                "subset": "iso350" if in350 else "keyword_only",
            })
    return tasks


def phase_paddle():
    import torch
    torch.manual_seed(42)
    np.random.seed(42)
    from prepare_pn_v6_dataset import load_assembly_with_names, shape_to_pointcloud

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_path = load_pn_model(device)

    tasks = build_paddle_tasks()
    n350 = sum(1 for t in tasks if t["subset"] == "iso350")
    print(f"PADDLE holdout: {len(tasks)} GT solids ({n350} in iso350)", flush=True)

    PADDLE_PC_DIR.mkdir(parents=True, exist_ok=True)
    need_decompose = any(
        not (PADDLE_PC_DIR / f"{safe_name(t['sample_id'].replace('/', '_'))}.npy").exists()
        for t in tasks)
    solids = None
    if need_decompose:
        print("decomposing PADDLE STEAMER.STEP (one pass) ...", flush=True)
        solids = load_assembly_with_names(PADDLE_STEP)
        print(f"  {len(solids)} solids", flush=True)

    rows = []
    for t in tasks:
        cache = PADDLE_PC_DIR / f"{safe_name(t['sample_id'].replace('/', '_'))}.npy"
        row = dict(t)
        row["part_name"] = ""
        row["name_mismatch"] = 0
        row["error"] = ""
        try:
            if cache.exists():
                pc = np.load(cache)
                meta = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8")) \
                    if cache.with_suffix(".json").exists() else {}
                row["part_name"] = meta.get("part_name", "")
                row["name_mismatch"] = meta.get("name_mismatch", 0)
            else:
                idx = t["solid_idx"]
                if idx >= len(solids):
                    raise RuntimeError(f"solid_idx {idx} >= {len(solids)}")
                name, solid = solids[idx]
                row["part_name"] = name
                row["name_mismatch"] = int(safe_name(name) != t["expected_solid_name"])
                pc = shape_to_pointcloud(solid, NUM_POINTS)
                np.save(cache, pc)
                cache.with_suffix(".json").write_text(json.dumps(
                    {"part_name": name, "name_mismatch": row["name_mismatch"]}),
                    encoding="utf-8")
            preds, confs = predict_batch(model, [normalize_pc(pc)], device)
            row["pred_subtype13"] = CLASSES13[preds[0]]
            row["confidence"] = round(confs[0], 4)
            row["correct13"] = int(row["pred_subtype13"] == t["gt_subtype13"])
        except Exception as e:
            row["pred_subtype13"] = ""
            row["confidence"] = ""
            row["correct13"] = ""
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)

    fields = ["sample_id", "part_name", "expected_solid_name", "solid_idx",
              "subset", "gt_subtype13", "pred_subtype13", "confidence",
              "correct13", "name_mismatch", "error"]
    with PADDLE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {PADDLE_CSV}", flush=True)

    iso = [r for r in rows if r["subset"] == "iso350" and r["pred_subtype13"]]
    n_fail = sum(1 for r in rows if r["error"])
    result = {"checkpoint": ckpt_path.name, "n_gt_solids": len(rows),
              "n_extraction_failures": n_fail}
    per_class = {}
    for cls in ("screws", "nuts", "rivets", "pins"):
        sub = [r for r in iso if r["gt_subtype13"] == cls]
        per_class[cls] = {"n": len(sub),
                          "correct": sum(r["correct13"] for r in sub),
                          "accuracy": sum(r["correct13"] for r in sub) / max(len(sub), 1)}
    result["iso350"] = {
        "n": len(iso),
        "correct": sum(r["correct13"] for r in iso),
        "accuracy": sum(r["correct13"] for r in iso) / max(len(iso), 1),
        "per_class": per_class,
    }
    print(f"PN++ PADDLE iso350: {result['iso350']['correct']}/{result['iso350']['n']} "
          f"= {result['iso350']['accuracy']:.3f}", flush=True)
    for cls, c in per_class.items():
        print(f"  {cls}: {c['correct']}/{c['n']} = {c['accuracy']:.3f}", flush=True)

    # ---- McNemar vs BF v6 paddle predictions ----
    # Join by SOLID INDEX, not part_name: the BF predictions.csv is one row
    # per solid in canonical decomposition order, and identical-named ISO
    # part instances (e.g. many "socket head cap screw ... M5 x 10") would
    # otherwise be paired to the wrong instance. The holdout dir name encodes
    # the canonical solid index; BF row[idx] is that solid's prediction. A
    # safe_name(part_name) sanity check guards the alignment.
    bf_rows = list(csv.DictReader(BF_PADDLE_CSV.open("r", newline="", encoding="utf-8")))
    pn_c, bf_c = [], []
    n_unmatched = n_name_guard_fail = 0
    for r in iso:
        idx = int(r["solid_idx"])
        if idx >= len(bf_rows):
            n_unmatched += 1
            continue
        b = bf_rows[idx]
        if safe_name(b["part_name"]) != r["expected_solid_name"]:
            n_name_guard_fail += 1
        pn_c.append(bool(r["correct13"]))
        bf_c.append(b["prediction"] == r["gt_subtype13"])
    pn_c = np.array(pn_c, dtype=bool)
    bf_c = np.array(bf_c, dtype=bool)
    result["paired_vs_bf_v6"] = {
        "join": "solid_index",
        "n_shared": int(len(pn_c)),
        "n_unmatched_by_index": n_unmatched,
        "n_name_guard_fail": n_name_guard_fail,
        "pn_accuracy_shared": float(pn_c.mean()) if len(pn_c) else None,
        "bf_accuracy_shared": float(bf_c.mean()) if len(bf_c) else None,
        "mcnemar_pn_vs_bf": mcnemar(pn_c, bf_c),
        "bootstrap_diff_pn_minus_bf": paired_bootstrap_diff(pn_c, bf_c),
    }
    m = result["paired_vs_bf_v6"]
    print(f"paired vs BF (index-join): n={m['n_shared']} "
          f"(unmatched={n_unmatched}, name_guard_fail={n_name_guard_fail})  "
          f"PN {m['pn_accuracy_shared']:.3f} vs BF {m['bf_accuracy_shared']:.3f}  "
          f"McNemar chi2={m['mcnemar_pn_vs_bf']['chi2']:.2f} "
          f"p={m['mcnemar_pn_vs_bf']['p_value']:.4e}", flush=True)
    if m["bf_accuracy_shared"] is not None and abs(m["bf_accuracy_shared"] - 0.64) > 0.02:
        print("WARN: BF paddle accuracy on shared set deviates from the stored "
              "64.0% — check part_name pairing!", flush=True)

    (RUN_DIR / "paddle_eval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {RUN_DIR / 'paddle_eval.json'}", flush=True)


# ---------------------------------------------------------------------------
# Phase: report
# ---------------------------------------------------------------------------

def phase_report():
    def jload(p: Path):
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    test = jload(RUN_DIR / "test_eval.json")
    paddle = jload(RUN_DIR / "paddle_eval.json")
    train = jload(RUN_DIR / "train_summary.json")
    retention = jload(RUN_DIR / "retention.json")
    stored_bf = jload(BF_V6_EVAL_SUMMARY)

    lines = ["# PointNet++ v6 evaluation report", ""]
    if train:
        lines += [f"- best val_acc {train['best_val_acc']:.4f}, "
                  f"best val_loss {train['best_val_loss']:.4f}, "
                  f"stopped at epoch {train['stopped_epoch']}", ""]
    if retention:
        lines += ["## Retention (PC twin vs BF v6 sample lists)", "",
                  "| split | BF v6 n | PC n | retention |", "|---|---|---|---|"]
        for split in ("train", "val", "test"):
            if split in retention:
                t = retention[split]["TOTAL"]
                lines.append(f"| {split} | {t['bf_v6']} | {t['pc']} | "
                             f"{100.0 * t['pc'] / max(t['bf_v6'], 1):.1f}% |")
        lines.append("")
    if test:
        m13 = test["multi13"]
        lines += ["## v6 test split", "",
                  f"- PN++ 13-class: acc {m13['accuracy']:.4f}, "
                  f"macro-F1 {m13['macro_f1']:.4f}, n={m13['n']}",
                  f"- PN++ binary-collapse: acc {test['binary_collapse']['accuracy']:.4f}", ""]
        if "paired_vs_bf_v6" in test:
            p = test["paired_vs_bf_v6"]
            mc = p["mcnemar_pn_vs_bf"]
            lines += [f"- paired (n={p['n_shared']}; "
                      f"{p['n_name_mismatch_excluded']} name-mismatch excluded): "
                      f"PN {p['pn_accuracy_shared']:.4f} vs BF {p['bf_accuracy_shared']:.4f}",
                      f"- McNemar: chi2={mc['chi2']:.2f}, p={mc['p_value']:.3e}, "
                      f"PN-only={mc['a_only']}, BF-only={mc['b_only']}",
                      f"- bootstrap diff (PN-BF): {p['bootstrap_diff_pn_minus_bf']['mean']:+.4f} "
                      f"CI95 {p['bootstrap_diff_pn_minus_bf']['ci95']}", ""]
        lines += ["| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
        for c in m13["per_class"]:
            lines.append(f"| {c['class']} | {c['precision']:.3f} | {c['recall']:.3f} "
                         f"| {c['f1']:.3f} | {c['support']} |")
        lines.append("")
    if paddle:
        iso = paddle["iso350"]
        lines += ["## PADDLE STEAMER holdout (iso350)", "",
                  f"- PN++: {iso['correct']}/{iso['n']} = {iso['accuracy']:.3f}",
                  "", "| class | n | PN++ correct | PN++ acc |", "|---|---|---|---|"]
        for cls, c in iso["per_class"].items():
            lines.append(f"| {cls} | {c['n']} | {c['correct']} | {c['accuracy']:.3f} |")
        p = paddle.get("paired_vs_bf_v6")
        if p:
            mc = p["mcnemar_pn_vs_bf"]
            lines += ["",
                      f"- paired vs BF v6 (n={p['n_shared']}): "
                      f"PN {p['pn_accuracy_shared']:.3f} vs BF {p['bf_accuracy_shared']:.3f}",
                      f"- McNemar: chi2={mc['chi2']:.2f}, p={mc['p_value']:.4f}, "
                      f"PN-only={mc['a_only']}, BF-only={mc['b_only']}"]
        lines.append("")

    summary = {"test": test, "paddle": paddle, "train": train,
               "retention": retention,
               "bf_v6_stored_eval_present": stored_bf is not None}
    (RUN_DIR / "eval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (RUN_DIR / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RUN_DIR / 'eval_report.md'} and eval_summary.json", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True,
                   choices=["bf-dump", "test", "paddle", "report"])
    args = p.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    {"bf-dump": phase_bf_dump, "test": phase_test,
     "paddle": phase_paddle, "report": phase_report}[args.phase]()


if __name__ == "__main__":
    main()
