"""Experiment 1 (WORKFLOW.md Phase 6): rule-based detector STANDALONE baseline.

Runs the existing rule-based fastener classifier (step_vr_step/detection/
rule_based.py + brep_signature.py + iso_tables.py) with ML fully disabled
(no --ml, no ensemble, no checkpoints) on the same held-out test sets the ML
models were scored on, so the thesis has a complete 3-way comparison:
rules vs PointNet++ vs BRepFormer.

The rule engine is benchmarked AS-IS with frozen DetectionConfig() defaults
(rule_confidence_threshold=0.60, dimension_tolerance_mm=0.1). No rule logic
or thresholds were modified for this evaluation.

Test sets (--testset):

  matched558   The matched McMaster test set used by full_analysis.py: the
               intersection of training_data/mcmaster_pc_breponly/test/ and
               training_data/mcmaster_brep/test/ part numbers (binary task).
               Source STEPs: training_data/mcmaster_binary/test/<class>/<pn>.step.
               Protocol per file = eval_ml_mcmaster.py: largest solid by volume.

  mcmaster649  Full McMaster test split (301 fastener / 348 non_fastener STEP
               files in training_data/mcmaster_binary/test/). Binary AND
               13-class subtype metrics (subtype ground truth from the
               McMaster dataset tree on D:).

  paddle       PADDLE STEAMER held-out assembly (D:\\step-vr-step-thesis\\
               fastener_labeling\\files\\PADDLE STEAMER.STEP). Ground truth =
               name-derived labels recorded in the paddle_steamer_holdout
               feature dirs (solid index -> class). The rule engine sees ONLY
               geometry; names are used exclusively for scoring.

  v6test       Held-out test split of the v6 diversified dataset
               (reproducible-build/training_data/bf_v2_features/test/) with
               the same max_faces=300 filter the stored v6 eval used
               (n=7,286). Each sample dir holds only BF features, so the
               source geometry is re-resolved per source prefix:
                 mcmaster__<pn>        -> D:/.../fastener_labeling/dataset tree
                 synthno__<stem>       -> reproducible-build/training_data/synthetic_no_threads
                 synthrm__<stem>       -> reproducible-build/training_data/synthetic_remixed
                 realcad__fusion__...__<uuid> -> D:/.../fusion360_assembly_raw/<assy>/<uuid>.step
                 realcad__<file>__<solid>__<idx> -> decompose D:/.../fastener_labeling/files/<file>
                 grabcad__<file>__<solid>__<idx> -> decompose D:/.../grabcad_kept/<file>
               Assembly decompositions replicate load_assembly_with_names()
               ordering (vendored below); a name sanity check flags any
               solid-index drift.

Rule-output -> subtype13 label mapping (documented in RESEARCH_JOURNAL.md):

  hex_bolt, socket_head_cap_screw, button_head_screw,
  countersunk_socket_screw, set_screw, wood_screw   -> screws
  hex_nut, thin_hex_nut, square_nut                 -> nuts
  flat_washer, chamfered_washer                     -> washers
  dowel_pin                                         -> pins
  threaded_stud                                     -> threaded-rods
  threaded_insert                                   -> threaded-inserts
  unclassified                                      -> non_fastener
  (the rules have NO output for: rivets, spacers, anchors, nails,
   retaining-rings, keys -> recall on those classes is 0 by construction)

Binary collapse follows detect.py step 5: prediction is "fastener" iff the
final fastener_type (after the confidence threshold inside classify_part)
is not "unclassified". "possible_X" counts as fastener.

Outputs per test set (default out root
training_data/mcmaster_logs/rule_based_eval/<testset>/):
  predictions.csv        per-sample rows, written incrementally (resume-safe:
                         rows already present are skipped on restart)
  summary.json           metrics + statistical tests + provenance
  confusion_matrix.csv   binary and (where applicable) 13-class confusion

Statistical tests on matched558 / mcmaster649:
  - exact McNemar (continuity-corrected) + paired bootstrap (2000, seed 42)
    vs the stored per-sample BRepFormer predictions in
    backend/ml_mcmaster_predictions.csv (bf_subtype13_best.ckpt collapsed to
    binary — the ONLY per-sample ML predictions saved to disk).
  - conservative-bound McNemar vs PointNet++ matched and BF Plan C' on
    matched558: their per-sample predictions were never saved (only the
    aggregates in full_analysis.json), so we bound the discordant-pair count
    from the marginals. b - c is fixed by the marginals; b + c is maximised
    for the most conservative chi^2. If even the bound is significant the
    conclusion holds under ANY possible pairing.

Run (miniconda CPU env is sufficient — no GPU, no torch):
  python backend/scripts/eval_rule_based.py --testset matched558
  python backend/scripts/eval_rule_based.py --testset mcmaster649
  python backend/scripts/eval_rule_based.py --testset paddle
  python backend/scripts/eval_rule_based.py --testset v6test --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from collections import Counter, defaultdict
from math import erfc, sqrt
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import numpy as np

REPO = _BACKEND_ROOT.parent
D_THESIS = Path(r"D:\step-vr-step-thesis")
RB_TD = D_THESIS / "reproducible-build" / "training_data"

MCMASTER_BINARY_TEST = REPO / "training_data" / "mcmaster_binary" / "test"
PC_BREPONLY_TEST = REPO / "training_data" / "mcmaster_pc_breponly" / "test"
BREP_TEST = REPO / "training_data" / "mcmaster_brep" / "test"
# Original 30-category scrape tree (subtype ground truth; full_analysis.py's
# DATASET_ROOT). The D: copy was reorganized for the v2+ datasets and only
# keeps 5 non_fastener subdirs, so the repo tree is primary for GT.
MCMASTER_DATASET_REPO = REPO / "fastener_labeling" / "dataset"
# The tree prepare_bf_v2_dataset.py actually read its mcmaster__ samples from
# (used for v6 source-geometry resolution).
MCMASTER_DATASET = D_THESIS / "fastener_labeling" / "dataset"
PADDLE_STEP = D_THESIS / "fastener_labeling" / "files" / "PADDLE STEAMER.STEP"
PADDLE_HOLDOUT = RB_TD / "paddle_steamer_holdout"
V6_TEST = RB_TD / "bf_v2_features" / "test"
V6_EVAL_SUMMARY = RB_TD / "bf_v6_run" / "eval_summary.json"
FULL_ANALYSIS_JSON = REPO / "training_data" / "mcmaster_logs" / "full_analysis.json"
BF_V1_PREDICTIONS_CSV = _BACKEND_ROOT / "ml_mcmaster_predictions.csv"
SYNTH_NO_THREADS = RB_TD / "synthetic_no_threads"
SYNTH_REMIXED = RB_TD / "synthetic_remixed"
FUSION_RAW = D_THESIS / "fusion360_assembly_raw"
REALCAD_FILES = D_THESIS / "fastener_labeling" / "files"
# GrabCAD sources are scattered: round-2 files live in grabcad_kept, a few
# remain in the dump, round-1 hand-curated files are in the repo's
# training_data/grabcad tree. Some sources were deleted after feature
# extraction and are genuinely unrecoverable (documented in summary.json).
GRABCAD_ROOTS = [
    D_THESIS / "grabcad_kept",
    D_THESIS / "grabcad_dump",
    D_THESIS / "grabcad_trash",
    REPO / "training_data" / "grabcad",
]
DEFAULT_OUT_ROOT = REPO / "training_data" / "mcmaster_logs" / "rule_based_eval"

V6_MAX_FACES = 300  # matches the stored v6 eval (n=7,286 of 7,898 test dirs)

CLASSES13 = [
    "anchors", "keys", "nails", "non_fastener", "nuts", "pins",
    "retaining-rings", "rivets", "screws", "spacers",
    "threaded-inserts", "threaded-rods", "washers",
]

# Rule-engine output label -> subtype13 class (see module docstring).
RULE_TO_SUBTYPE13 = {
    "hex_bolt": "screws",
    "socket_head_cap_screw": "screws",
    "button_head_screw": "screws",
    "countersunk_socket_screw": "screws",
    "set_screw": "screws",
    "wood_screw": "screws",
    "hex_nut": "nuts",
    "thin_hex_nut": "nuts",
    "square_nut": "nuts",
    "flat_washer": "washers",
    "chamfered_washer": "washers",
    "dowel_pin": "pins",
    "threaded_stud": "threaded-rods",
    "threaded_insert": "threaded-inserts",
    "unclassified": "non_fastener",
}

MCMASTER_NON_FASTENER_SUBDIRS = (
    "brackets", "hinges", "mounting-plates", "pcbs", "t-slotted-framing",
)

CSV_FIELDS = [
    "sample_id", "source", "source_path", "solid_idx",
    "gt_binary", "gt_subtype13", "subset",
    "rule_type_raw", "rule_confidence", "pred_binary", "pred_subtype13",
    "n_faces", "name_mismatch", "error",
]


def safe_name(s: str) -> str:
    """Identical to extract_real_cad_fasteners.safe_name (used to build the
    v6 sample dir names) — needed to reverse-map dirs to source files."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:80]


# ---------------------------------------------------------------------------
# Rule-based classification of a single OCC shape
# ---------------------------------------------------------------------------

def classify_shape(shape) -> dict:
    """Run the frozen rule-based pipeline on one TopoDS_Shape."""
    from step_vr_step.config import DetectionConfig
    from step_vr_step.detection.geometric_features import extract_brep_features
    from step_vr_step.detection.rule_based import classify_part

    feat = extract_brep_features(shape)
    label = classify_part(feat, DetectionConfig(), repetition_count=1)
    base = label.fastener_type.replace("possible_", "")
    pred_binary = "fastener" if base != "unclassified" else "non_fastener"
    pred_subtype13 = RULE_TO_SUBTYPE13.get(base, "non_fastener")
    return {
        "rule_type_raw": label.fastener_type,
        "rule_confidence": round(float(label.confidence), 4),
        "pred_binary": pred_binary,
        "pred_subtype13": pred_subtype13,
        "n_faces": feat.num_faces,
    }


# ---------------------------------------------------------------------------
# STEP loading helpers (OCC imports stay inside functions: Windows spawn-safe)
# ---------------------------------------------------------------------------

def _read_step_oneshape(step_path: str):
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone

    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != IFSelect_RetDone:
        raise RuntimeError("STEP read failed")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        raise RuntimeError("OneShape() returned null")
    return shape


def _solids_of(shape) -> list:
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopoDS import topods

    out = []
    e = TopExp_Explorer(shape, TopAbs_SOLID)
    while e.More():
        out.append(topods.Solid(e.Current()))
        e.Next()
    return out


def _largest_solid(shape):
    """eval_ml_mcmaster.py protocol: largest solid by volume."""
    from OCC.Core.BRepGProp import brepgprop_VolumeProperties
    from OCC.Core.GProp import GProp_GProps

    solids = _solids_of(shape)
    if not solids:
        raise RuntimeError("no solids in STEP")

    def vol(s):
        g = GProp_GProps()
        brepgprop_VolumeProperties(s, g)
        return abs(g.Mass())

    return max(solids, key=vol)


def load_assembly_with_names(step_path: Path):
    """Vendored verbatim (minus tqdm) from scripts/classify_assembly.py so the
    solid ordering matches the one used to build the realcad/grabcad/paddle
    feature dirs (sample ids embed the solid index of this enumeration)."""
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    from OCC.Extend.DataExchange import read_step_file_with_names_colors

    def count_faces(shape) -> int:
        n = 0
        e = TopExp_Explorer(shape, TopAbs_FACE)
        while e.More():
            n += 1
            e.Next()
        return n

    def signature(shape) -> tuple:
        bb = Bnd_Box()
        try:
            brepbndlib.Add(shape, bb)
            x1, y1, z1, x2, y2, z2 = bb.Get()
            dims = tuple(sorted([round(x2 - x1, 1), round(y2 - y1, 1), round(z2 - z1, 1)]))
        except Exception:
            dims = (0.0, 0.0, 0.0)
        return (count_faces(shape),) + dims

    shape_name_map = read_step_file_with_names_colors(str(step_path))

    sig_to_name: dict[tuple, str] = {}
    for shape, info in shape_name_map.items():
        name = info[0] if info else ""
        if not name:
            continue
        e = TopExp_Explorer(shape, TopAbs_SOLID)
        while e.More():
            s = topods.Solid(e.Current())
            sig_to_name.setdefault(signature(s), name)
            e.Next()

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != 1:
        raise ValueError("STEP read failed")
    reader.TransferRoots()
    whole = reader.OneShape()

    all_instances = _solids_of(whole)
    out: list[tuple[str, object]] = []
    name_counts: dict[str, int] = {}
    for idx, solid in enumerate(all_instances):
        base_name = sig_to_name.get(signature(solid), f"unnamed_solid_{idx}")
        n = name_counts.get(base_name, 0)
        name_counts[base_name] = n + 1
        out.append((f"{base_name}#{n}" if n > 0 else base_name, solid))
    return out


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def _row_base(task: dict) -> dict:
    row = {k: "" for k in CSV_FIELDS}
    row.update({
        "sample_id": task["sample_id"],
        "source": task.get("source", ""),
        "source_path": task.get("source_path", ""),
        "solid_idx": task.get("solid_idx", ""),
        "gt_binary": task.get("gt_binary", ""),
        "gt_subtype13": task.get("gt_subtype13", ""),
        "subset": task.get("subset", ""),
        "name_mismatch": 0,
        "error": "",
    })
    return row


def worker_single_step(task: dict) -> dict:
    """One STEP file -> one prediction. task['mode'] selects the solid:
    'largest' (eval_ml_mcmaster protocol), 'whole' (prepare_bf_v2_dataset
    protocol: OneShape), 'first' (fusion per-body protocol: first solid)."""
    row = _row_base(task)
    try:
        shape = _read_step_oneshape(task["source_path"])
        mode = task.get("mode", "largest")
        if mode == "largest":
            shape = _largest_solid(shape)
        elif mode == "first":
            solids = _solids_of(shape)
            if not solids:
                raise RuntimeError("no solids in STEP")
            shape = solids[0]
        row.update(classify_shape(shape))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def worker_assembly(group: tuple[str, list[dict]]) -> list[dict]:
    """Decompose one assembly once; classify every requested solid index."""
    asm_path, tasks = group
    rows = []
    try:
        solids = load_assembly_with_names(Path(asm_path))
    except Exception as e:
        for t in tasks:
            row = _row_base(t)
            row["error"] = f"decompose: {type(e).__name__}: {e}"
            rows.append(row)
        return rows

    for t in tasks:
        row = _row_base(t)
        idx = int(t["solid_idx"])
        if idx >= len(solids):
            row["error"] = f"solid_idx {idx} >= {len(solids)} solids"
            rows.append(row)
            continue
        name, solid = solids[idx]
        expected = t.get("expected_solid_name")
        if expected and safe_name(name) != expected:
            row["name_mismatch"] = 1
        try:
            row.update(classify_shape(solid))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Incremental CSV (resume-safe)
# ---------------------------------------------------------------------------

def load_done(csv_path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done[r["sample_id"]] = r
    return done


class IncrementalCSV:
    def __init__(self, path: Path):
        self.path = path
        new = not path.exists()
        self.f = path.open("a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            self.w.writeheader()
            self.f.flush()

    def write(self, row: dict):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------
# Metrics (protocol of full_analysis.py)
# ---------------------------------------------------------------------------

def binary_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    t = np.array([0 if y == "fastener" else 1 for y in y_true])
    p = np.array([0 if y == "fastener" else 1 for y in y_pred])
    cm = {
        "tp_fast": int(((t == 0) & (p == 0)).sum()),
        "fn_fast": int(((t == 0) & (p == 1)).sum()),
        "tp_non": int(((t == 1) & (p == 1)).sum()),
        "fn_non": int(((t == 1) & (p == 0)).sum()),
    }
    per_class = {}
    for kname in ("fastener", "non_fastener"):
        if kname == "fastener":
            tp, fn, fp = cm["tp_fast"], cm["fn_fast"], cm["fn_non"]
        else:
            tp, fn, fp = cm["tp_non"], cm["fn_non"], cm["fn_fast"]
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[kname] = {"tp": tp, "fn": fn, "fp": fp,
                            "recall": rec, "precision": prec, "f1": f1}
    tp_, tn_ = cm["tp_fast"], cm["tp_non"]
    fp_, fn_ = cm["fn_non"], cm["fn_fast"]   # fp_ = non_fastener called fastener
    denom = sqrt(float((tp_ + fp_) * (tp_ + fn_) * (tn_ + fp_) * (tn_ + fn_)))
    mcc = float((tp_ * tn_ - fp_ * fn_) / denom) if denom else 0.0
    n = len(t)
    bal = 0.0
    if (t == 0).any() and (t == 1).any():
        bal = ((p[t == 0] == 0).mean() + (p[t == 1] == 1).mean()) / 2
    return {
        "n": n,
        "accuracy": float((t == p).mean()),
        "balanced_accuracy": float(bal),
        "macro_f1": (per_class["fastener"]["f1"] + per_class["non_fastener"]["f1"]) / 2,
        "mcc": mcc,
        "per_class": per_class,
        "confusion": cm,
    }


def multi13_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    ix = {c: i for i, c in enumerate(CLASSES13)}
    t = np.array([ix[y] for y in y_true])
    p = np.array([ix.get(y, ix["non_fastener"]) for y in y_pred])
    K = len(CLASSES13)
    cm = np.zeros((K, K), dtype=int)
    for a, b in zip(t, p):
        cm[a, b] += 1
    per_class = []
    f1s = []
    for i, c in enumerate(CLASSES13):
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class.append({"class": c, "precision": prec, "recall": rec,
                          "f1": f1, "support": int(cm[i].sum())})
        f1s.append(f1)
    return {
        "n": int(len(t)),
        "accuracy": float((t == p).mean()),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def bootstrap_acc_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(correct)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        accs[i] = correct[idx].mean()
    return {"mean": float(accs.mean()),
            "ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]}


def paired_bootstrap_diff(correct_a: np.ndarray, correct_b: np.ndarray,
                          n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap CI for acc(a) - acc(b) on paired samples (full_analysis protocol)."""
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()
    return {"mean": float(diffs.mean()),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Continuity-corrected McNemar on paired correctness vectors
    (a = rules, b = the ML model). Same math as full_analysis.py."""
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


def mcnemar_bounds(rules_correct: np.ndarray, model_n_correct: int, model_n: int) -> dict:
    """Bound the McNemar statistic when only the model's AGGREGATE correct
    count is known (per-sample predictions were never saved). With
    b = model right & rules wrong, c = model wrong & rules right:
      b - c = model_correct - rules_correct   (fixed by marginals)
      b + c in [|b - c|, min(e_rules, c_model) + min(e_model, c_rules)]
    chi2 = (|b-c|-1)^2/(b+c) is decreasing in b+c, so the max-discordance
    case gives the most conservative (largest) p-value."""
    n = len(rules_correct)
    c_r = int(rules_correct.sum())
    e_r = n - c_r
    # Scale the model's correct count if its n differs by a sample or two
    # (e.g. full_analysis scored 558 of our 559 — report both ns).
    c_m = model_n_correct
    e_m = model_n - c_m
    diff = abs(c_m - c_r)
    d_max = min(e_r, c_m) + min(e_m, c_r)
    d_min = max(diff, 1)
    chi2_min = ((diff - 1) ** 2) / d_max if d_max > 0 else 0.0
    chi2_max = ((diff - 1) ** 2) / d_min
    return {
        "rules_correct": c_r, "rules_n": n,
        "model_correct": c_m, "model_n": model_n,
        "b_minus_c": c_m - c_r,
        "discordant_range": [diff, d_max],
        "chi2_range": [float(chi2_min), float(chi2_max)],
        "p_value_worst_case": float(erfc(sqrt(chi2_min / 2.0))),
        "p_value_best_case": float(erfc(sqrt(chi2_max / 2.0))),
        "note": ("Per-sample predictions for this model were never saved; "
                 "p_value_worst_case holds under ANY possible pairing of errors."),
    }


# ---------------------------------------------------------------------------
# Ground-truth indexes
# ---------------------------------------------------------------------------

def build_mcmaster_subtype_index() -> dict[str, str]:
    """Map McMaster part-number stem -> subtype13 class. Primary source is
    the repo's fastener_labeling/dataset (the original full 30-category tree
    full_analysis.py used); the reorganized D: copy is the fallback.
    fastener/<cat>/... -> cat; non_fastener/** -> non_fastener."""
    idx: dict[str, str] = {}
    for dataset_root in (MCMASTER_DATASET_REPO, MCMASTER_DATASET):
        fast_root = dataset_root / "fastener"
        if fast_root.exists():
            for cat_dir in fast_root.iterdir():
                if not cat_dir.is_dir() or cat_dir.name not in CLASSES13:
                    continue
                for f in cat_dir.rglob("*"):
                    if f.is_file() and f.suffix.lower() in (".step", ".stp") \
                            and not f.name.startswith("._"):
                        idx.setdefault(f.stem, cat_dir.name)
        nf_root = dataset_root / "non_fastener"
        if nf_root.exists():
            for f in nf_root.rglob("*"):
                if f.is_file() and f.suffix.lower() in (".step", ".stp") \
                        and not f.name.startswith("._"):
                    idx.setdefault(f.stem, "non_fastener")
    return idx


def enumerate_mcmaster_test() -> list[dict]:
    sub_idx = build_mcmaster_subtype_index()
    tasks = []
    n_missing_sub = 0
    for cls in ("fastener", "non_fastener"):
        for f in sorted((MCMASTER_BINARY_TEST / cls).iterdir()):
            if f.suffix.lower() not in (".step", ".stp"):
                continue
            sub = sub_idx.get(f.stem)
            if sub is None:
                n_missing_sub += 1
                sub = "non_fastener" if cls == "non_fastener" else "unknown"
            tasks.append({
                "sample_id": f.stem,
                "source": "mcmaster_binary_test",
                "source_path": str(f),
                "gt_binary": cls,
                "gt_subtype13": sub,
                "mode": "largest",
            })
    if n_missing_sub:
        print(f"  WARN: {n_missing_sub} part numbers not found in the dataset "
              f"subtype tree (gt_subtype13 = 'unknown'/'non_fastener' fallback)")
    return tasks


def matched_part_numbers() -> dict[str, set[str]]:
    """full_analysis.py intersection: PC stems x complete BRep sample dirs."""
    need = ("face_grids.npy", "edge_curves.npy", "topo_distances.npz")
    out = {}
    for cls in ("fastener", "non_fastener"):
        pc = {p.stem for p in (PC_BREPONLY_TEST / cls).glob("*.npy")}
        br = {d.name for d in (BREP_TEST / cls).iterdir()
              if d.is_dir() and all((d / f).exists() for f in need)}
        out[cls] = pc & br
    return out


# ---------------------------------------------------------------------------
# Test-set builders
# ---------------------------------------------------------------------------

def build_tasks_matched558() -> tuple[list[dict], list]:
    matched = matched_part_numbers()
    keep = matched["fastener"] | matched["non_fastener"]
    tasks = [t for t in enumerate_mcmaster_test() if t["sample_id"] in keep]
    print(f"matched test set: {len(tasks)} samples "
          f"(fastener={len(matched['fastener'])}, non_fastener={len(matched['non_fastener'])})")
    return tasks, []


def build_tasks_mcmaster649() -> tuple[list[dict], list]:
    tasks = enumerate_mcmaster_test()
    print(f"full McMaster test split: {len(tasks)} samples")
    return tasks, []


def build_tasks_paddle() -> tuple[list[dict], list]:
    """One assembly group covering every solid that has name-derived GT in
    paddle_steamer_holdout. subset='iso350' marks the 350-part set the
    journal's v1..v6 numbers were computed on (everything except the 62
    keyword-only PADDLE_MASTER_ROD_PIVOT_BOLT parts)."""
    if not PADDLE_STEP.exists():
        raise FileNotFoundError(PADDLE_STEP)
    gt_tasks = []
    for cls_dir in sorted(PADDLE_HOLDOUT.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in CLASSES13:
            continue
        for d in sorted(cls_dir.iterdir()):
            parts = d.name.split("__")
            # realcad__PADDLE_STEAMER__<solid_basename>__<idx>
            if len(parts) < 4:
                continue
            solid_basename = "__".join(parts[2:-1])
            idx = int(parts[-1])
            in350 = not solid_basename.startswith("PADDLE_MASTER_ROD_PIVOT_BOLT")
            gt_tasks.append({
                "sample_id": f"{cls_dir.name}/{d.name}",
                "source": "paddle_holdout",
                "source_path": str(PADDLE_STEP),
                "solid_idx": idx,
                "expected_solid_name": solid_basename,
                "gt_binary": "fastener",
                "gt_subtype13": cls_dir.name,
                "subset": "iso350" if in350 else "keyword_only",
            })
    print(f"PADDLE holdout: {len(gt_tasks)} ground-truth solids "
          f"({sum(1 for t in gt_tasks if t['subset'] == 'iso350')} in iso350 subset)")
    return [], [(str(PADDLE_STEP), gt_tasks)]


def build_tasks_v6test() -> tuple[list[dict], list]:
    """Enumerate bf_v2_features/test with the max_faces=300 filter, then
    resolve source geometry per prefix."""
    # ---- indexes ----
    print("building source indexes ...")
    mc_idx: dict[str, Path] = {}
    fast_root = MCMASTER_DATASET / "fastener"
    roots = []
    if fast_root.exists():
        roots += [fast_root / c for c in CLASSES13 if c != "non_fastener"]
    roots += [MCMASTER_DATASET / "non_fastener" / s for s in MCMASTER_NON_FASTENER_SUBDIRS]
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if f.is_file() and f.suffix.lower() in (".step", ".stp") \
                    and not f.name.startswith("._"):
                mc_idx.setdefault(safe_name(f.stem), f)

    def index_flat(root: Path) -> dict[tuple[str, str], Path]:
        out = {}
        if not root.exists():
            return out
        for cls_dir in root.iterdir():
            if not cls_dir.is_dir():
                continue
            for f in cls_dir.iterdir():
                if f.is_file() and f.suffix.lower() in (".step", ".stp"):
                    out.setdefault((cls_dir.name, safe_name(f.stem)), f)
        return out

    synthno_idx = index_flat(SYNTH_NO_THREADS)
    synthrm_idx = index_flat(SYNTH_REMIXED)

    def index_assemblies(root: Path) -> dict[str, Path]:
        out = {}
        if not root.exists():
            return out
        for f in root.rglob("*"):
            if f.is_file() and f.suffix.lower() in (".step", ".stp") \
                    and not f.name.startswith("._"):
                out.setdefault(safe_name(f.stem), f)
        return out

    realcad_idx = index_assemblies(REALCAD_FILES)
    grabcad_idx: dict[str, Path] = {}
    for root in GRABCAD_ROOTS:
        for k, v in index_assemblies(root).items():
            grabcad_idx.setdefault(k, v)
    print(f"  mcmaster={len(mc_idx)} synthno={len(synthno_idx)} "
          f"synthrm={len(synthrm_idx)} realcad_files={len(realcad_idx)} "
          f"grabcad_files={len(grabcad_idx)}")

    def resolve_assembly(body: str, fidx: dict[str, Path]):
        """body = '<file_base>__<solid_base>__<idx>' where file_base and
        solid_base may themselves contain '__' (safe_name doubles spaced
        punctuation). Find the LONGEST '__'-joined prefix present in the
        file index; the remainder is the solid name."""
        head, sep, tail = body.rpartition("__")
        if not sep or not tail.isdigit():
            return None, None, None
        idx = int(tail)
        stem_parts = head.split("__")
        if len(stem_parts) < 2:
            return None, None, idx
        for i in range(len(stem_parts) - 1, 0, -1):
            cand = "__".join(stem_parts[:i])
            p = fidx.get(cand)
            if p is not None:
                return p, "__".join(stem_parts[i:]), idx
        return None, None, idx

    # ---- enumerate test dirs ----
    singles: list[dict] = []
    asm_groups: dict[str, list[dict]] = defaultdict(list)
    unresolved: Counter = Counter()
    n_over_cap = 0
    for cls_dir in sorted(V6_TEST.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in CLASSES13:
            continue
        cls = cls_dir.name
        for d in sorted(cls_dir.iterdir()):
            fg = d / "face_grids.npy"
            if not fg.exists():
                continue
            try:
                if np.load(fg, mmap_mode="r").shape[0] > V6_MAX_FACES:
                    n_over_cap += 1
                    continue
            except Exception:
                unresolved[f"{cls}:bad_face_grids"] += 1
                continue
            name = d.name
            sample_id = f"{cls}/{name}"
            base = {"sample_id": sample_id, "gt_binary":
                    "non_fastener" if cls == "non_fastener" else "fastener",
                    "gt_subtype13": cls}

            if name.startswith("mcmaster__"):
                key = name[len("mcmaster__"):]
                p = mc_idx.get(key)
                if p is None:
                    unresolved["mcmaster"] += 1
                    continue
                singles.append({**base, "source": "mcmaster",
                                "source_path": str(p), "mode": "whole"})
            elif name.startswith("synthno__"):
                p = synthno_idx.get((cls, name[len("synthno__"):]))
                if p is None:
                    unresolved["synthno"] += 1
                    continue
                singles.append({**base, "source": "synthno",
                                "source_path": str(p), "mode": "whole"})
            elif name.startswith("synthrm__"):
                p = synthrm_idx.get((cls, name[len("synthrm__"):]))
                if p is None:
                    unresolved["synthrm"] += 1
                    continue
                singles.append({**base, "source": "synthrm",
                                "source_path": str(p), "mode": "whole"})
            elif name.startswith("realcad__fusion__"):
                uuid_stem = name.split("__")[-1]
                # fusion per-body file: <FUSION_RAW>/<assy_id>/<uuid>.step
                assy_id = name.split("__")[2]
                p = FUSION_RAW / assy_id / f"{uuid_stem}.step"
                if not p.exists():
                    hits = list(FUSION_RAW.glob(f"*/{uuid_stem}.step"))
                    p = hits[0] if hits else None
                if p is None:
                    unresolved["fusion"] += 1
                    continue
                singles.append({**base, "source": "fusion",
                                "source_path": str(p), "mode": "first"})
            elif name.startswith(("realcad__", "grabcad__")):
                src = "realcad" if name.startswith("realcad__") else "grabcad"
                body = name[len(src) + 2:]
                fidx = realcad_idx if src == "realcad" else grabcad_idx
                p, solid_base, idx = resolve_assembly(body, fidx)
                if idx is None:
                    unresolved[f"{src}:bad_name"] += 1
                    continue
                if p is None:
                    unresolved[src] += 1
                    continue
                asm_groups[str(p)].append({**base, "source": src,
                                           "source_path": str(p),
                                           "solid_idx": idx,
                                           "expected_solid_name": solid_base})
            else:
                unresolved["unknown_prefix"] += 1

    n_asm = sum(len(v) for v in asm_groups.values())
    print(f"v6 test: {len(singles)} single-file samples + {n_asm} assembly "
          f"samples across {len(asm_groups)} assemblies "
          f"(skipped {n_over_cap} over the {V6_MAX_FACES}-face cap)")
    if unresolved:
        print(f"  UNRESOLVED sources (excluded, reported in summary): {dict(unresolved)}")
    groups = sorted(asm_groups.items())
    # stash for the summary
    build_tasks_v6test.unresolved = dict(unresolved)
    build_tasks_v6test.n_over_cap = n_over_cap
    return singles, groups


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tasks(singles: list[dict], groups: list, out_csv: Path, workers: int):
    done = load_done(out_csv)
    singles_todo = [t for t in singles if t["sample_id"] not in done]
    groups_todo = []
    for path, tasks in groups:
        todo = [t for t in tasks if t["sample_id"] not in done]
        if todo:
            groups_todo.append((path, todo))
    n_total = len(singles_todo) + sum(len(t) for _, t in groups_todo)
    print(f"{len(done)} rows already done; {n_total} to compute "
          f"({len(singles_todo)} singles, {len(groups_todo)} assemblies)")
    if n_total == 0:
        return

    sink = IncrementalCSV(out_csv)
    t0 = time.time()
    n_done = 0

    def progress():
        rate = n_done / max(time.time() - t0, 1e-6)
        eta = (n_total - n_done) / max(rate, 1e-6)
        print(f"  {n_done}/{n_total}  rate={rate:.2f}/s  eta={eta/60:.1f}min",
              flush=True)

    try:
        if workers <= 1:
            for t in singles_todo:
                sink.write(worker_single_step(t))
                n_done += 1
                if n_done % 50 == 0:
                    progress()
            for g in groups_todo:
                for row in worker_assembly(g):
                    sink.write(row)
                    n_done += 1
                progress()
        else:
            with multiprocessing.Pool(processes=workers) as pool:
                if singles_todo:
                    for row in pool.imap_unordered(worker_single_step,
                                                   singles_todo, chunksize=8):
                        sink.write(row)
                        n_done += 1
                        if n_done % 100 == 0:
                            progress()
                if groups_todo:
                    for rows in pool.imap_unordered(worker_assembly,
                                                    groups_todo, chunksize=1):
                        for row in rows:
                            sink.write(row)
                            n_done += 1
                        progress()
    finally:
        sink.close()
    print(f"done in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def rows_ok(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not r.get("error")]


def summarize_mcmaster(rows: list[dict], testset: str, out_dir: Path) -> dict:
    ok = rows_ok(rows)
    errs = [r for r in rows if r.get("error")]
    y_true = [r["gt_binary"] for r in ok]
    y_pred = [r["pred_binary"] for r in ok]
    binm = binary_metrics(y_true, y_pred)
    correct_rules = np.array([a == b for a, b in zip(y_true, y_pred)])
    summary = {
        "testset": testset,
        "protocol": "eval_ml_mcmaster.py per-file protocol (largest solid by "
                    "volume); rules frozen at DetectionConfig() defaults; ML disabled.",
        "n_scored": len(ok),
        "n_errors": len(errs),
        "errors": [{"sample_id": r["sample_id"], "error": r["error"]} for r in errs],
        "binary": binm,
        "binary_accuracy_bootstrap": bootstrap_acc_ci(correct_rules),
        "rule_label_histogram": dict(Counter(
            r["rule_type_raw"].replace("possible_", "") for r in ok)),
    }

    # subtype13 metrics (only meaningful where gt_subtype13 is known)
    sub_rows = [r for r in ok if r["gt_subtype13"] in CLASSES13]
    if sub_rows:
        summary["subtype13"] = multi13_metrics(
            [r["gt_subtype13"] for r in sub_rows],
            [r["pred_subtype13"] for r in sub_rows])
        summary["subtype13"]["n_excluded_unknown_gt"] = len(ok) - len(sub_rows)

    # --- exact paired tests vs the stored BF per-sample predictions ---
    if BF_V1_PREDICTIONS_CSV.exists():
        bf = {}
        with BF_V1_PREDICTIONS_CSV.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                bf[r["file"]] = r
        shared = [r for r in ok if r["sample_id"] in bf]
        if shared:
            rc = np.array([r["gt_binary"] == r["pred_binary"] for r in shared])
            bc = np.array([int(bf[r["sample_id"]]["correct"]) == 1 for r in shared])
            summary["vs_bf_subtype13_binary_stored"] = {
                "model": "bf_subtype13_best.ckpt collapsed to binary "
                         "(backend/ml_mcmaster_predictions.csv) — the only "
                         "per-sample ML predictions saved to disk",
                "n_paired": len(shared),
                "rules_acc": float(rc.mean()),
                "bf_acc": float(bc.mean()),
                "mcnemar": mcnemar(rc, bc),
                "bootstrap_diff_rules_minus_bf": paired_bootstrap_diff(rc, bc),
            }

    # --- conservative-bound tests vs the matched binary models (aggregates only) ---
    if testset == "matched558" and FULL_ANALYSIS_JSON.exists():
        fa = json.loads(FULL_ANALYSIS_JSON.read_text())
        out = {}
        for key, label in (("pointnet2_matched", "PointNet++ matched"),
                           ("brepformer_planc_prime", "BRepFormer Plan C'")):
            m = fa[key]
            pc = m["per_class"]
            model_correct = pc["fastener"]["tp"] + pc["non_fastener"]["tp"]
            out[key] = {
                "model": label,
                "model_accuracy_stored": m["accuracy"],
                "model_n_stored": m["n"],
                "mcnemar_bounds": mcnemar_bounds(correct_rules, model_correct, m["n"]),
                "accuracy_delta_rules_minus_model":
                    float(correct_rules.mean()) - m["accuracy"],
            }
        out["note"] = (
            "full_analysis.json stores only aggregate metrics for these two "
            "models (their per-sample predictions were never written to disk), "
            "and re-running the checkpoints was out of scope for this "
            "experiment. McNemar is therefore bounded from the marginals; "
            "the worst-case p-value holds under any pairing of errors. "
            "full_analysis scored n=558 (one sample dropped at ML load time); "
            "the rules scored the full 559-part intersection.")
        summary["vs_matched_binary_models_bounds"] = out

    _write_confusions(summary, out_dir)
    return summary


def summarize_paddle(rows: list[dict], out_dir: Path) -> dict:
    ok = rows_ok(rows)
    errs = [r for r in rows if r.get("error")]
    out = {
        "testset": "paddle",
        "protocol": "PADDLE STEAMER.STEP decomposed with the vendored "
                    "load_assembly_with_names (same solid ordering as the "
                    "holdout feature extraction). Rules see geometry only; "
                    "names were used solely to derive ground truth.",
        "n_scored": len(ok),
        "n_errors": len(errs),
        "n_name_mismatch": sum(int(r.get("name_mismatch") or 0) for r in ok),
        "errors": [{"sample_id": r["sample_id"], "error": r["error"]} for r in errs],
    }
    for subset_name, rs in (("iso350", [r for r in ok if r["subset"] == "iso350"]),
                            ("all_holdout", ok)):
        per_class = {}
        for cls in ("screws", "nuts", "rivets", "pins"):
            crs = [r for r in rs if r["gt_subtype13"] == cls]
            n_corr = sum(1 for r in crs if r["pred_subtype13"] == cls)
            per_class[cls] = {"n": len(crs), "correct": n_corr,
                              "accuracy": n_corr / len(crs) if crs else 0.0}
        n = len(rs)
        n_corr = sum(1 for r in rs if r["pred_subtype13"] == r["gt_subtype13"])
        n_fast = sum(1 for r in rs if r["pred_binary"] == "fastener")
        out[subset_name] = {
            "n": n,
            "subtype_accuracy": n_corr / n if n else 0.0,
            "subtype_correct": n_corr,
            "binary_fastener_recall": n_fast / n if n else 0.0,
            "per_class": per_class,
            "prediction_histogram": dict(Counter(r["pred_subtype13"] for r in rs)),
        }
    out["journal_reference"] = {
        "bf_v1_iso350_accuracy": 0.031, "bf_v6_iso350_accuracy": 0.640,
        "source": "RESEARCH_JOURNAL.md 2026-05-21 -> 05-25 per-class table (n=350)",
    }
    return out


def summarize_v6test(rows: list[dict], out_dir: Path) -> dict:
    ok = rows_ok(rows)
    errs = [r for r in rows if r.get("error")]
    # Name-mismatch rows: the solid at the recorded index had a different
    # name than the sample dir encodes — for generic file stems ("Assembly")
    # the prefix resolver can pick a different file with the same safe_name,
    # so the scored solid's identity is unverified. Exclude from metrics,
    # report separately (audit 2026-06-12: 23 rows, all grabcad, scoring
    # them or not moves 13-class accuracy by ~0.1pp).
    mismatched = [r for r in ok if str(r.get("name_mismatch")) == "1"]
    ok = [r for r in ok if str(r.get("name_mismatch")) != "1"]
    sub = multi13_metrics([r["gt_subtype13"] for r in ok],
                          [r["pred_subtype13"] for r in ok])
    binm = binary_metrics([r["gt_binary"] for r in ok],
                          [r["pred_binary"] for r in ok])
    correct = np.array([r["gt_subtype13"] == r["pred_subtype13"] for r in ok])
    summary = {
        "testset": "v6test",
        "protocol": f"bf_v2_features/test with max_faces={V6_MAX_FACES} "
                    "(reproduces the stored v6 eval set, n=7286). Source "
                    "geometry re-resolved per prefix; assemblies decomposed "
                    "with the vendored load_assembly_with_names.",
        "n_scored": len(ok),
        "n_errors": len(errs),
        "n_excluded_name_mismatch": len(mismatched),
        "excluded_name_mismatch_ids": [r["sample_id"] for r in mismatched],
        "unresolved_sources": getattr(build_tasks_v6test, "unresolved", None),
        "subtype13": sub,
        "subtype13_accuracy_bootstrap": bootstrap_acc_ci(correct),
        "binary_collapse": binm,
        "per_source_accuracy": {},
        "error_histogram": dict(Counter(
            (r["error"] or "")[:60] for r in errs).most_common(10)),
    }
    by_src = defaultdict(list)
    for r in ok:
        by_src[r["source"]].append(r["gt_subtype13"] == r["pred_subtype13"])
    for src, v in sorted(by_src.items()):
        summary["per_source_accuracy"][src] = {
            "n": len(v), "subtype13_accuracy": float(np.mean(v))}

    # stored v6 reference numbers + binary collapse from its stored confusion
    if V6_EVAL_SUMMARY.exists():
        v6 = json.loads(V6_EVAL_SUMMARY.read_text())["v2_full_test"]
        cm = np.array(v6["confusion_matrix"])
        nf = CLASSES13.index("non_fastener")
        fast_ix = [i for i in range(len(CLASSES13)) if i != nf]
        tp_f = cm[np.ix_(fast_ix, fast_ix)].sum()
        fn_f = cm[fast_ix, nf].sum()
        fp_f = cm[nf, fast_ix].sum()
        tn_f = cm[nf, nf]
        summary["stored_v6_reference"] = {
            "checkpoint": "bf_subtype13_v6.ckpt",
            "n": v6["n"], "accuracy": v6["accuracy"], "macro_f1": v6["macro_f1"],
            "binary_collapse_accuracy": float((tp_f + tn_f) / cm.sum()),
            "source": str(V6_EVAL_SUMMARY),
        }
    _write_confusions(summary, out_dir)
    return summary


def _write_confusions(summary: dict, out_dir: Path):
    lines = []
    b = summary.get("binary") or summary.get("binary_collapse")
    if b:
        cm = b["confusion"]
        lines.append("binary confusion (rows=actual, cols=predicted)")
        lines.append(",pred_fastener,pred_non_fastener")
        lines.append(f"actual_fastener,{cm['tp_fast']},{cm['fn_fast']}")
        lines.append(f"actual_non_fastener,{cm['fn_non']},{cm['tp_non']}")
        lines.append("")
    s = summary.get("subtype13")
    if s and "confusion_matrix" in s:
        lines.append("subtype13 confusion (rows=actual, cols=predicted)")
        lines.append("," + ",".join(CLASSES13))
        for cname, row in zip(CLASSES13, s["confusion_matrix"]):
            lines.append(cname + "," + ",".join(str(v) for v in row))
    if lines:
        (out_dir / "confusion_matrix.csv").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BUILDERS = {
    "matched558": build_tasks_matched558,
    "mcmaster649": build_tasks_mcmaster649,
    "paddle": build_tasks_paddle,
    "v6test": build_tasks_v6test,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testset", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--limit", type=int, default=0,
                    help="Debug: cap the number of samples (0 = all)")
    ap.add_argument("--summary-only", action="store_true",
                    help="Skip computation; recompute summary from the existing CSV")
    args = ap.parse_args()

    out_dir = args.out_root / args.testset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "predictions.csv"

    singles, groups = BUILDERS[args.testset]()
    if args.limit:
        singles = singles[: args.limit]
        groups = groups[: max(1, args.limit // 50)] if groups else groups

    if not args.summary_only:
        run_tasks(singles, groups, out_csv, args.workers)

    # Re-read everything from CSV (the resume-safe source of truth), keep only
    # rows that belong to the current task list (guards against stale rows).
    wanted = {t["sample_id"] for t in singles}
    for _, ts in groups:
        wanted |= {t["sample_id"] for t in ts}
    all_rows = [r for r in load_done(out_csv).values() if r["sample_id"] in wanted]
    print(f"summarizing {len(all_rows)} rows")

    if args.testset in ("matched558", "mcmaster649"):
        summary = summarize_mcmaster(all_rows, args.testset, out_dir)
    elif args.testset == "paddle":
        summary = summarize_paddle(all_rows, out_dir)
    else:
        summary = summarize_v6test(all_rows, out_dir)

    summary["provenance"] = {
        "script": "backend/scripts/eval_rule_based.py",
        "rule_engine": "step_vr_step/detection/{rule_based,brep_signature,iso_tables}.py "
                       "(frozen, DetectionConfig defaults, no ML)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "predictions_csv": str(out_csv),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(f"summary -> {out_dir / 'summary.json'}")

    # console headline
    if "binary" in summary:
        b = summary["binary"]
        print(f"\nBINARY  acc={b['accuracy']*100:.2f}%  macroF1={b['macro_f1']*100:.2f}%  "
              f"MCC={b['mcc']:.3f}  n={b['n']}")
    if "subtype13" in summary and "accuracy" in summary.get("subtype13", {}):
        s = summary["subtype13"]
        print(f"SUBTYPE13  acc={s['accuracy']*100:.2f}%  macroF1={s['macro_f1']*100:.2f}%  n={s['n']}")
    if args.testset == "paddle":
        i = summary["iso350"]
        print(f"PADDLE iso350  acc={i['subtype_accuracy']*100:.1f}%  "
              f"({i['subtype_correct']}/{i['n']})")
        for c, d in i["per_class"].items():
            print(f"  {c:8s} {d['correct']:>3}/{d['n']:<3} = {d['accuracy']*100:.1f}%")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
