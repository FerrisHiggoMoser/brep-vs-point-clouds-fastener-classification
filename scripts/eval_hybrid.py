"""Experiments 4/5: GEOMETRY-ONLY hybrid cascade (rules -> BRepFormer v6).

Two-stage cascade, no part names anywhere in the prediction path:

  Stage 1  Rule-based pre-pass (step_vr_step/detection/{rule_based,
           brep_signature,iso_tables}.py, frozen DetectionConfig() defaults,
           repetition_count=1 — the exact Experiment-1 protocol). classify_part
           applies rule_confidence_threshold=0.60 internally, so a part is
           "claimed" by Stage 1 iff its final fastener_type != "unclassified".
           A claimed part's label is FINAL.
  Stage 2  BRepFormer v6 (models/bf_subtype13_v6.ckpt, dict format) raw-argmax
           fallback for everything Stage 1 left unclassified.
           Variant: Stage 2 with the production geometry-aware top-K recovery
           filter (_ml_with_topk_recovery in ml_classifier.py — the function
           is IMPORTED, not re-implemented).

Part names are GROUND TRUTH ONLY. A name-regex prediction stage would be
circular (the ground truth is itself name-derived) and is deliberately NOT
implemented. The only name-related output is a coverage statistic (fraction
of parts carrying a parseable ISO/DIN name), reported as context.

Configurations (--config):
  rules        Stage 1 alone (unclaimed -> non_fastener). No torch needed.
  ml           BRepFormer v6 alone (raw argmax, no thresholds).
  hybrid       Stage 1, else Stage 2 (plain).
  hybrid_topk  Stage 1, else Stage 2 with top-K geometry recovery.
  all          All four + pairwise McNemar tests (default).

Test sets (--testset):
  v6test     bf_v2_features/test with max_faces=300 (the stored v6 eval set,
             n=7,286). Stage-1 labels are REUSED verbatim from the
             Experiment-1 run (training_data/mcmaster_logs/rule_based_eval/
             v6test/predictions.csv, n=5,678 resolvable samples, 0 errors).
             ML runs on the precomputed feature tensors (all 7,286). The
             four-config comparison uses the 5,678-sample common subset;
             ML-alone is additionally reported on the full 7,286.
             Geometry for the top-K gate is re-extracted ONLY for samples
             that fall through to Stage 2 (resumable, multiprocess).
  paddle     PADDLE STEAMER held-out assembly. GT = 412 name-labeled solids
             in paddle_steamer_holdout/ (by solid index; iso350 subset =
             the journal's 350-part set). Stage-1 labels reused from
             Experiment 1; ML runs on the precomputed holdout tensors.
  satellite  ISIS CubeSat (training_data/holdout_test/isispace_*.stp).
             GT = explicit name->class table (SATELLITE_NAME_GT below)
             reconstructing the 2026-05-19 named-part protocol. Everything
             (rules + features + ML) computed in one in-process pass
             (XDE doc is not picklable), resumable per part.

Per-stage attribution is reported for every cascade configuration: how many
parts Stage 1 claimed and its accuracy on them; how many fell through to
Stage 2 and its accuracy there.

Frozen everything: no retraining, no threshold tuning. DetectionConfig()
defaults for rules; raw argmax for ML; production top-K filter as-is.

Outputs under training_data/hybrid_eval/<testset>/:
  ml_predictions.csv    per-sample BF v6 top-3 (incremental, resume-safe)
  geom_rules.csv        per-sample rule label + geometry summary (incremental)
  predictions.csv       combined per-sample table, all configs + attribution
  summary.json          metrics, attribution, McNemar, name coverage
  confusion_matrix_<config>.csv

Run (anaconda GPU env for ml/hybrid*; miniconda CPU env is enough for rules):
  python backend/scripts/eval_hybrid.py --testset paddle    --config all
  python backend/scripts/eval_hybrid.py --testset satellite --config all
  python backend/scripts/eval_hybrid.py --testset v6test    --config all --workers 8
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
from types import SimpleNamespace

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

import numpy as np

REPO = _BACKEND_ROOT.parent
D_THESIS = Path(r"D:\step-vr-step-thesis")
RB_TD = D_THESIS / "reproducible-build" / "training_data"

V6_CKPT = D_THESIS / "reproducible-build" / "models" / "bf_subtype13_v6.ckpt"
V6_TEST = RB_TD / "bf_v2_features" / "test"
V6_EVAL_SUMMARY = RB_TD / "bf_v6_run" / "eval_summary.json"
PADDLE_STEP = D_THESIS / "fastener_labeling" / "files" / "PADDLE STEAMER.STEP"
PADDLE_HOLDOUT = RB_TD / "paddle_steamer_holdout"
PADDLE_V6_STORED = RB_TD / "bf_v6_run" / "paddle_steamer_v6" / "predictions.csv"
SATELLITE_STEP = REPO / "training_data" / "holdout_test" / "isispace_1uplt_type_b_2023-04-20.stp"
RULES_EVAL_ROOT = REPO / "training_data" / "mcmaster_logs" / "rule_based_eval"
DEFAULT_OUT_ROOT = REPO / "training_data" / "hybrid_eval"

# v6test source-geometry roots (for the top-K geometry pass only)
MCMASTER_DATASET = D_THESIS / "fastener_labeling" / "dataset"
SYNTH_NO_THREADS = RB_TD / "synthetic_no_threads"
SYNTH_REMIXED = RB_TD / "synthetic_remixed"
FUSION_RAW = D_THESIS / "fusion360_assembly_raw"
REALCAD_FILES = D_THESIS / "fastener_labeling" / "files"
GRABCAD_ROOTS = [
    D_THESIS / "grabcad_kept",
    D_THESIS / "grabcad_dump",
    D_THESIS / "grabcad_trash",
    REPO / "training_data" / "grabcad",
]

V6_MAX_FACES = 300        # matches the stored v6 eval (n=7,286)
SAT_FACE_CAP = 600        # eval_ml_mcmaster predict_one deployment protocol

CLASSES13 = [
    "anchors", "keys", "nails", "non_fastener", "nuts", "pins",
    "retaining-rings", "rivets", "screws", "spacers",
    "threaded-inserts", "threaded-rods", "washers",
]

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

# Satellite name -> subtype13 ground truth. Reconstruction of the 2026-05-19
# named-part protocol (the exact 168-part GT set was never saved to disk).
# Names are matched against the instance-suffix-stripped part name.
#   iso_din          unambiguous catalog standard in the name
#   journal_verified ISIS-internal part numbers whose class was manually
#                    verified in the 2026-05-19 journal entry
SATELLITE_NAME_GT: list[tuple[str, str, str, str]] = [
    (r"^DIN912[_-]", "screws", "iso_din", "DIN 912 = ISO 4762 socket head cap screw"),
    (r"^DIN125", "washers", "iso_din", "DIN 125 flat washer"),
    (r"^ISO14581[_-]", "screws", "iso_din", "ISO 14581 hexalobular countersunk screw"),
    (r"^ISO14583[_-]", "screws", "iso_din", "ISO 14583 hexalobular pan head screw"),
    (r"^M1_NUT", "nuts", "iso_din", "M1 hex nut"),
    (r"^IOBC_DB_SPACER", "spacers", "journal_verified",
     "explicitly named spacer (journal 2026-05-19 lists 'IOBC_DB_SPACER called bolts' as an error)"),
    (r"^761_M05", "washers", "journal_verified",
     "ISIS 761-series Al ring washer/spacer; journal 2026-05-19 records "
     "'761_M0506/M0504 (x24): generic washers -> flat_washer (correct)'"),
]
_SAT_GT_COMPILED = [(re.compile(p), cls, src, note) for p, cls, src, note in SATELLITE_NAME_GT]

ISO_DIN_NAME_RE = re.compile(r"(?<![a-z])(iso|din)[\s_-]*\d+", re.IGNORECASE)

GEOM_FIELDS = [
    "sample_id", "source", "source_path", "solid_idx",
    "rule_type_raw", "rule_confidence", "n_faces", "geom_json",
    "name_mismatch", "error",
]
ML_FIELDS = [
    "sample_id", "n_faces", "face_cap",
    "top1", "p1", "top2", "p2", "top3", "p3", "error",
]
FINAL_FIELDS = [
    "sample_id", "source", "gt_subtype13", "gt_binary", "subset", "gt_source",
    "rule_type_raw", "rule_confidence", "pred_rules",
    "ml_top1", "ml_p1", "ml_top2", "ml_p2", "ml_top3", "ml_p3", "ml_face_cap",
    "pred_ml",
    "pred_hybrid", "stage_hybrid",
    "pred_hybrid_topk", "stage_hybrid_topk", "topk_action",
]


def safe_name(s: str) -> str:
    """Identical to extract_real_cad_fasteners.safe_name (sample-dir naming)."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:80]


# ---------------------------------------------------------------------------
# Geometry: rules + serializable summary for the top-K gate
# ---------------------------------------------------------------------------

def geom_summary(feat) -> dict:
    """Serialize exactly the GeometricFeatures fields that
    _passes_geometry_check / _cluster_radii consume."""
    return {
        "bbox_min": list(feat.bbox_min),
        "bbox_max": list(feat.bbox_max),
        "aspect_ratio": float(feat.aspect_ratio),
        "ext_radii": [float(c.radius) for c in feat.cylinders if not c.is_internal],
        "int_radii": [float(c.radius) for c in feat.cylinders if c.is_internal],
        "face_type_counts": dict(feat.face_type_counts),
        "head_diameter": (float(feat.head_diameter)
                          if feat.head_diameter is not None else None),
    }


def geom_proxy(d: dict):
    """Rebuild a features stand-in for _passes_geometry_check from the
    serialized summary."""
    cyls = ([SimpleNamespace(radius=r, is_internal=False) for r in d["ext_radii"]]
            + [SimpleNamespace(radius=r, is_internal=True) for r in d["int_radii"]])
    return SimpleNamespace(
        bbox_min=tuple(d["bbox_min"]),
        bbox_max=tuple(d["bbox_max"]),
        aspect_ratio=d["aspect_ratio"],
        cylinders=cyls,
        face_type_counts=d["face_type_counts"],
        head_diameter=d["head_diameter"],
    )


def classify_shape(shape) -> dict:
    """Frozen Experiment-1 rule protocol + geometry summary, one extraction."""
    from step_vr_step.config import DetectionConfig
    from step_vr_step.detection.geometric_features import extract_brep_features
    from step_vr_step.detection.rule_based import classify_part

    feat = extract_brep_features(shape)
    label = classify_part(feat, DetectionConfig(), repetition_count=1)
    return {
        "rule_type_raw": label.fastener_type,
        "rule_confidence": round(float(label.confidence), 4),
        "n_faces": feat.num_faces,
        "geom_json": json.dumps(geom_summary(feat)),
    }


def apply_topk(top3: list[tuple[str, float]], geom: dict | None) -> tuple[str, str]:
    """Run the PRODUCTION top-K recovery on a top-3 list + geometry summary.
    Returns (final_subtype13, action)."""
    from step_vr_step.detection.ml_classifier import _ml_with_topk_recovery
    from step_vr_step.schema import DetectionLabel

    top1, p1 = top3[0]
    if geom is None:
        return (top1 if top1 in CLASSES13 else "non_fastener"), "no_geom"
    label = DetectionLabel(
        fastener_type=top1, confidence=round(float(p1), 4),
        method="ml_brepformer",
        detected_dimensions={"ml_top3": [(c, float(p)) for c, p in top3]},
    )
    out = _ml_with_topk_recovery(label, geom_proxy(geom))
    final = out.fastener_type.replace("likely_", "").replace("possible_", "")
    if final == "unclassified":
        return "non_fastener", "demoted"
    action = "recovered" if (out.detected_dimensions or {}).get("ml_topk_recovered") \
        else "top1_pass"
    return (final if final in CLASSES13 else "non_fastener"), action


# ---------------------------------------------------------------------------
# STEP loading helpers (OCC imports inside functions: Windows spawn-safe)
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
    """Vendored verbatim from eval_rule_based.py (originally
    scripts/classify_assembly.py) so solid ordering matches the holdout/v6
    feature extraction runs."""
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
# Geometry-pass workers
# ---------------------------------------------------------------------------

def _geom_row_base(task: dict) -> dict:
    row = {k: "" for k in GEOM_FIELDS}
    row.update({
        "sample_id": task["sample_id"],
        "source": task.get("source", ""),
        "source_path": task.get("source_path", ""),
        "solid_idx": task.get("solid_idx", ""),
        "name_mismatch": 0,
        "error": "",
    })
    return row


def worker_single_geom(task: dict) -> dict:
    row = _geom_row_base(task)
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


def worker_assembly_geom(group: tuple[str, list[dict]]) -> list[dict]:
    asm_path, tasks = group
    rows = []
    try:
        solids = load_assembly_with_names(Path(asm_path))
    except Exception as e:
        for t in tasks:
            row = _geom_row_base(t)
            row["error"] = f"decompose: {type(e).__name__}: {e}"
            rows.append(row)
        return rows

    for t in tasks:
        row = _geom_row_base(t)
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
    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        new = not path.exists()
        self.f = path.open("a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore")
        if new:
            self.w.writeheader()
            self.f.flush()

    def write(self, row: dict):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


def run_geom_tasks(singles: list[dict], groups: list, out_csv: Path, workers: int):
    done = load_done(out_csv)
    singles_todo = [t for t in singles if t["sample_id"] not in done]
    groups_todo = []
    for path, tasks in groups:
        todo = [t for t in tasks if t["sample_id"] not in done]
        if todo:
            groups_todo.append((path, todo))
    n_total = len(singles_todo) + sum(len(t) for _, t in groups_todo)
    print(f"  geom pass: {len(done)} rows done; {n_total} to compute "
          f"({len(singles_todo)} singles, {len(groups_todo)} assemblies)")
    if n_total == 0:
        return

    sink = IncrementalCSV(out_csv, GEOM_FIELDS)
    t0 = time.time()
    n_done = 0

    def progress():
        rate = n_done / max(time.time() - t0, 1e-6)
        eta = (n_total - n_done) / max(rate, 1e-6)
        print(f"    {n_done}/{n_total}  rate={rate:.2f}/s  eta={eta/60:.1f}min",
              flush=True)

    try:
        if workers <= 1:
            for t in singles_todo:
                sink.write(worker_single_geom(t))
                n_done += 1
                if n_done % 50 == 0:
                    progress()
            for g in groups_todo:
                for row in worker_assembly_geom(g):
                    sink.write(row)
                    n_done += 1
                progress()
        else:
            with multiprocessing.Pool(processes=workers) as pool:
                if singles_todo:
                    for row in pool.imap_unordered(worker_single_geom,
                                                   singles_todo, chunksize=8):
                        sink.write(row)
                        n_done += 1
                        if n_done % 100 == 0:
                            progress()
                if groups_todo:
                    for rows in pool.imap_unordered(worker_assembly_geom,
                                                    groups_todo, chunksize=1):
                        for row in rows:
                            sink.write(row)
                            n_done += 1
                        progress()
    finally:
        sink.close()
    print(f"  geom pass done in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# ML phase: BF v6 batched inference on precomputed feature dirs
# ---------------------------------------------------------------------------

def load_v6_model(device):
    """Dict-format v6 checkpoint loader (protocol of eval_bf_v2.load_model)."""
    import torch
    from step_vr_step.models.brepformer.brepformer import BRepFormer

    ckpt = torch.load(V6_CKPT, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise RuntimeError(f"{V6_CKPT} is not a dict-format checkpoint")
    class_names = list(ckpt.get("class_names") or [])
    num_classes = int(ckpt.get("num_classes") or len(class_names))
    if class_names != CLASSES13:
        raise RuntimeError(f"ckpt class_names != canonical 13: {class_names}")
    model = BRepFormer(num_classes=num_classes, dim=256, num_layers=8,
                       head_mode="classification").to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"  WARN v6 load: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model, class_names


def _load_feature_dir(d: Path):
    face_grids = np.load(d / "face_grids.npy").astype(np.float32)
    edge_curves = np.load(d / "edge_curves.npy").astype(np.float32)
    topo_data = np.load(d / "topo_distances.npz")
    topo = {k: topo_data[k].astype(np.float32)
            for k in ("face_shortest", "face_centroid", "face_angular", "edge_path")
            if k in topo_data}
    return face_grids, edge_curves, topo


def _forward_batch(model, device, items: list[tuple[str, Path]]) -> list[dict]:
    """items: [(sample_id, feature_dir)] -> per-sample top-3 rows.
    Padding + masks replicate brep_collate_fn exactly."""
    import torch

    loaded = []
    rows = []
    for sid, d in items:
        try:
            loaded.append((sid, *_load_feature_dir(d)))
        except Exception as e:
            rows.append({"sample_id": sid, "n_faces": "", "face_cap": 0,
                         "top1": "", "p1": "", "top2": "", "p2": "",
                         "top3": "", "p3": "",
                         "error": f"{type(e).__name__}: {e}"})
    if not loaded:
        return rows

    B = len(loaded)
    max_f = max(fg.shape[0] for _, fg, _, _ in loaded)
    max_e = max(max(ec.shape[0], 1) for _, _, ec, _ in loaded)
    face_grids = torch.zeros(B, max_f, 10, 10, 7)
    edge_curves = torch.zeros(B, max_e, 10, 12)
    face_mask = torch.zeros(B, max_f, dtype=torch.bool)
    edge_mask = torch.zeros(B, max_e, dtype=torch.bool)
    topo_keys = ("face_shortest", "face_centroid", "face_angular", "edge_path")
    topo = {k: torch.zeros(B, max_f, max_f) for k in topo_keys}
    for i, (sid, fg, ec, td) in enumerate(loaded):
        nf, ne = fg.shape[0], ec.shape[0]
        face_grids[i, :nf] = torch.from_numpy(fg)
        if ne:
            edge_curves[i, :ne] = torch.from_numpy(ec)
            edge_mask[i, :ne] = True
        face_mask[i, :nf] = True
        for k in topo_keys:
            if k in td:
                topo[k][i, :nf, :nf] = torch.from_numpy(td[k])

    with torch.no_grad():
        logits = model(face_grids.to(device), edge_curves.to(device),
                       {k: v.to(device) for k, v in topo.items()},
                       face_mask.to(device), edge_mask.to(device))
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

    for i, (sid, fg, ec, td) in enumerate(loaded):
        p = probs[i]
        order = np.argsort(-p)[:3]
        rows.append({
            "sample_id": sid, "n_faces": fg.shape[0], "face_cap": 0,
            "top1": CLASSES13[order[0]], "p1": round(float(p[order[0]]), 6),
            "top2": CLASSES13[order[1]], "p2": round(float(p[order[1]]), 6),
            "top3": CLASSES13[order[2]], "p3": round(float(p[order[2]]), 6),
            "error": "",
        })
    return rows


def run_ml_phase(samples: list[tuple[str, Path]], out_csv: Path,
                 batch_size: int = 8):
    """samples: [(sample_id, feature_dir)]. Resumable."""
    done = load_done(out_csv)
    todo = [(sid, d) for sid, d in samples if sid not in done]
    print(f"  ml phase: {len(done)} rows done; {len(todo)} to compute")
    if not todo:
        return
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    model, _ = load_v6_model(device)

    # Sort by face count to minimize padding waste.
    def nf_of(d: Path) -> int:
        try:
            return int(np.load(d / "face_grids.npy", mmap_mode="r").shape[0])
        except Exception:
            return 0

    todo.sort(key=lambda t: nf_of(t[1]))
    sink = IncrementalCSV(out_csv, ML_FIELDS)
    t0 = time.time()
    n_done = 0
    try:
        for i in range(0, len(todo), batch_size):
            for row in _forward_batch(model, device, todo[i:i + batch_size]):
                sink.write(row)
                n_done += 1
            if (i // batch_size) % 50 == 0:
                rate = n_done / max(time.time() - t0, 1e-6)
                eta = (len(todo) - n_done) / max(rate, 1e-6)
                print(f"    {n_done}/{len(todo)}  rate={rate:.1f}/s  "
                      f"eta={eta/60:.1f}min", flush=True)
    finally:
        sink.close()
    print(f"  ml phase done in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# Metrics (protocol of full_analysis.py / eval_rule_based.py)
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
    return {
        "n": len(t),
        "accuracy": float((t == p).mean()),
        "macro_f1": (per_class["fastener"]["f1"] + per_class["non_fastener"]["f1"]) / 2,
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


# ---------------------------------------------------------------------------
# Test-set builders
# ---------------------------------------------------------------------------

def enumerate_v6_feature_dirs() -> list[dict]:
    """All bf_v2_features/test samples under the 300-face cap (n=7,286)."""
    out = []
    for cls_dir in sorted(V6_TEST.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in CLASSES13:
            continue
        for d in sorted(cls_dir.iterdir()):
            fg = d / "face_grids.npy"
            if not fg.exists():
                continue
            try:
                nf = int(np.load(fg, mmap_mode="r").shape[0])
            except Exception:
                continue
            if nf > V6_MAX_FACES:
                continue
            out.append({"sample_id": f"{cls_dir.name}/{d.name}",
                        "gt_subtype13": cls_dir.name,
                        "feature_dir": d, "n_faces": nf})
    return out


def build_v6_geom_tasks(wanted_ids: set[str]) -> tuple[list[dict], list]:
    """Source-resolution for v6test samples (vendored from eval_rule_based
    build_tasks_v6test), restricted to wanted_ids."""
    MCMASTER_NON_FASTENER_SUBDIRS = (
        "brackets", "hinges", "mounting-plates", "pcbs", "t-slotted-framing",
    )
    print("  building source indexes ...")
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

    def resolve_assembly(body: str, fidx: dict[str, Path]):
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

    singles: list[dict] = []
    asm_groups: dict[str, list[dict]] = defaultdict(list)
    unresolved: Counter = Counter()
    for sid in sorted(wanted_ids):
        cls, name = sid.split("/", 1)
        base = {"sample_id": sid}
        if name.startswith("mcmaster__"):
            p = mc_idx.get(name[len("mcmaster__"):])
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
            if idx is None or p is None:
                unresolved[src] += 1
                continue
            asm_groups[str(p)].append({**base, "source": src,
                                       "source_path": str(p),
                                       "solid_idx": idx,
                                       "expected_solid_name": solid_base})
        else:
            unresolved["unknown_prefix"] += 1
    if unresolved:
        print(f"  geom-pass unresolved sources (excluded): {dict(unresolved)}")
    return singles, sorted(asm_groups.items())


def paddle_gt_tasks() -> list[dict]:
    """412 name-labeled holdout solids; iso350 marks the journal's 350-set."""
    tasks = []
    for cls_dir in sorted(PADDLE_HOLDOUT.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in CLASSES13:
            continue
        for d in sorted(cls_dir.iterdir()):
            parts = d.name.split("__")
            if len(parts) < 4:
                continue
            solid_basename = "__".join(parts[2:-1])
            idx = int(parts[-1])
            in350 = not solid_basename.startswith("PADDLE_MASTER_ROD_PIVOT_BOLT")
            tasks.append({
                "sample_id": f"{cls_dir.name}/{d.name}",
                "source": "paddle_holdout",
                "source_path": str(PADDLE_STEP),
                "solid_idx": idx,
                "expected_solid_name": solid_basename,
                "gt_subtype13": cls_dir.name,
                "gt_binary": "fastener",
                "subset": "iso350" if in350 else "keyword_only",
                "feature_dir": d,
            })
    return tasks


def satellite_gt_for_name(part_name: str):
    """(gt_class, gt_source) from the explicit table, or (None, None)."""
    base = re.sub(r"_\d+$", "", part_name)  # strip read_step instance suffix
    for pat, cls, src, _note in _SAT_GT_COMPILED:
        if pat.search(base) or pat.search(part_name):
            return cls, src
    return None, None


# ---------------------------------------------------------------------------
# Satellite: single in-process pass (rules + features + ML per named part)
# ---------------------------------------------------------------------------

def run_satellite(out_dir: Path, need_ml: bool, limit: int = 0) -> dict:
    """Returns name-coverage stats. Writes geom_rules.csv (+ ml_predictions.csv
    when need_ml) for every part with name-derived GT."""
    from step_vr_step.readers.step_reader import read_step

    geom_csv = out_dir / "geom_rules.csv"
    ml_csv = out_dir / "ml_predictions.csv"
    geom_done = load_done(geom_csv)
    ml_done = load_done(ml_csv) if need_ml else {}

    print(f"  reading satellite STEP ({SATELLITE_STEP.stat().st_size/1e6:.0f} MB) ...")
    t0 = time.time()
    doc, manifest, shapes = read_step(str(SATELLITE_STEP), return_shapes=True)
    print(f"  read in {(time.time()-t0)/60:.1f} min, {len(manifest.parts)} parts, "
          f"{len(shapes)} shapes")

    # --- name coverage over ALL parts (context statistic, not accuracy) ---
    n_parts = len(manifest.parts)
    n_with_shape = 0
    n_named = 0
    n_iso_din = 0
    n_gt = 0
    gt_hist: Counter = Counter()
    scored: list[dict] = []
    name_counts: Counter = Counter()
    for part in manifest.parts:
        uid = str(part.uuid)
        name = part.name or ""
        has_shape = uid in shapes
        if has_shape:
            n_with_shape += 1
        base = re.sub(r"_\d+$", "", name)
        generic = (not name) or re.fullmatch(r"Part_?\d+|unnamed.*|SOLID\d*", base, re.IGNORECASE)
        if not generic:
            n_named += 1
        if ISO_DIN_NAME_RE.search(base):
            n_iso_din += 1
        gt_cls, gt_src = satellite_gt_for_name(name)
        if gt_cls and has_shape:
            n_gt += 1
            gt_hist[gt_cls] += 1
            k = name_counts[name]
            name_counts[name] += 1
            uname = f"{name}#{k}" if k else name  # names can repeat per instance
            scored.append({"uuid": uid, "name": uname,
                           "gt_subtype13": gt_cls, "gt_source": gt_src})
    coverage = {
        "n_parts_total": n_parts,
        "n_parts_with_shape": n_with_shape,
        "n_parts_named_nongeneric": n_named,
        "n_parts_iso_din_parseable": n_iso_din,
        "n_parts_name_derived_gt": n_gt,
        "gt_class_histogram": dict(gt_hist),
        "note": ("Name-derived GT reconstructs the 2026-05-19 named-part "
                 "protocol from an explicit name->class table (the original "
                 "168-part selection was never saved). Names are NEVER seen "
                 "by any prediction stage."),
    }
    print(f"  scored set: {n_gt} parts with name-derived GT  {dict(gt_hist)}")
    if limit:
        scored = scored[:limit]

    todo_geom = [s for s in scored if f"sat/{s['name']}" not in geom_done]
    todo_ml = [s for s in scored if f"sat/{s['name']}" not in ml_done] if need_ml else []
    print(f"  geom todo: {len(todo_geom)}   ml todo: {len(todo_ml)}")

    model = device = None
    if todo_ml:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_v6_model(device)
        print(f"  ml device: {device}")

    geom_sink = IncrementalCSV(geom_csv, GEOM_FIELDS) if todo_geom else None
    ml_sink = IncrementalCSV(ml_csv, ML_FIELDS) if todo_ml else None
    todo_ml_ids = {f"sat/{s['name']}" for s in todo_ml}
    todo_geom_ids = {f"sat/{s['name']}" for s in todo_geom}

    from step_vr_step.models.brepformer.feature_extractor import (
        extract_face_uv_grids, extract_edge_curves, compute_topology_distances,
    )
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE

    t0 = time.time()
    for i, s in enumerate(scored):
        sid = f"sat/{s['name']}"
        shape = shapes[s["uuid"]]

        if geom_sink and sid in todo_geom_ids:
            row = {k: "" for k in GEOM_FIELDS}
            row.update({"sample_id": sid, "source": "satellite",
                        "source_path": str(SATELLITE_STEP), "solid_idx": "",
                        "name_mismatch": 0, "error": ""})
            try:
                row.update(classify_shape(shape))
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
            geom_sink.write(row)

        if ml_sink and sid in todo_ml_ids:
            import torch
            exp = TopExp_Explorer(shape, TopAbs_FACE)
            nf = 0
            while exp.More():
                nf += 1
                exp.Next()
            row = {"sample_id": sid, "n_faces": nf, "face_cap": 0,
                   "top1": "", "p1": "", "top2": "", "p2": "",
                   "top3": "", "p3": "", "error": ""}
            if nf == 0 or nf > SAT_FACE_CAP:
                # eval_ml_mcmaster predict_one protocol: outside training
                # distribution -> non_fastener at zero confidence.
                row.update({"face_cap": 1, "top1": "non_fastener", "p1": 0.0,
                            "top2": "", "p2": "", "top3": "", "p3": ""})
            else:
                try:
                    fg = extract_face_uv_grids(shape).astype(np.float32)
                    ec = extract_edge_curves(shape).astype(np.float32)
                    td = compute_topology_distances(shape)
                    td = {k: v.astype(np.float32) for k, v in td.items()
                          if v is not None}
                    face_t = torch.from_numpy(fg).unsqueeze(0).to(device)
                    edge_t = torch.from_numpy(ec).unsqueeze(0).to(device)
                    topo_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                              for k, v in td.items()}
                    with torch.no_grad():
                        logits = model(face_t, edge_t, topo_t)
                        probs = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
                    order = np.argsort(-probs)[:3]
                    row.update({
                        "top1": CLASSES13[order[0]], "p1": round(float(probs[order[0]]), 6),
                        "top2": CLASSES13[order[1]], "p2": round(float(probs[order[1]]), 6),
                        "top3": CLASSES13[order[2]], "p3": round(float(probs[order[2]]), 6),
                    })
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"
            ml_sink.write(row)

        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(scored)}  ({(time.time()-t0)/60:.1f} min)",
                  flush=True)

    if geom_sink:
        geom_sink.close()
    if ml_sink:
        ml_sink.close()

    # persist the scored-set meta for the combine step
    meta = {"coverage": coverage,
            "scored": [{"sample_id": f"sat/{s['name']}",
                        "gt_subtype13": s["gt_subtype13"],
                        "gt_source": s["gt_source"]} for s in scored]}
    (out_dir / "satellite_meta.json").write_text(json.dumps(meta, indent=2),
                                                 encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# Paddle name coverage (cached; needs one decomposition)
# ---------------------------------------------------------------------------

def paddle_name_coverage(out_dir: Path) -> dict:
    cache = out_dir / "name_coverage.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    print("  computing PADDLE name coverage (one decomposition) ...")
    solids = load_assembly_with_names(PADDLE_STEP)
    n_total = len(solids)
    n_named = sum(1 for name, _ in solids if not name.startswith("unnamed_solid_"))
    n_iso_din = sum(1 for name, _ in solids if ISO_DIN_NAME_RE.search(name))
    cov = {
        "n_solids_total": n_total,
        "n_solids_named": n_named,
        "n_solids_iso_din_parseable": n_iso_din,
        "n_holdout_gt": 412,
        "n_iso350_subset": 350,
        "note": ("Coverage of parseable names in the assembly, reported as "
                 "context only. The detector never sees a name."),
    }
    cache.write_text(json.dumps(cov, indent=2), encoding="utf-8")
    return cov


# ---------------------------------------------------------------------------
# Combine: build per-sample predictions for all configs + summaries
# ---------------------------------------------------------------------------

def rule_base_of(raw: str) -> str:
    return (raw or "unclassified").replace("possible_", "").replace("likely_", "")


def combine(tasks: list[dict], rules_rows: dict[str, dict],
            ml_rows: dict[str, dict], geom_rows: dict[str, dict],
            configs: list[str]) -> list[dict]:
    """tasks carry sample_id/gt/subset; returns FINAL_FIELDS rows for samples
    scoreable under every requested config (common subset)."""
    out = []
    for t in tasks:
        sid = t["sample_id"]
        rr = rules_rows.get(sid)
        mr = ml_rows.get(sid)
        need_rules = any(c in configs for c in ("rules", "hybrid", "hybrid_topk"))
        need_ml = any(c in configs for c in ("ml", "hybrid", "hybrid_topk"))
        if need_rules and (rr is None or rr.get("error")):
            continue
        if need_ml and (mr is None or mr.get("error") or not mr.get("top1")):
            continue

        row = {k: "" for k in FINAL_FIELDS}
        row.update({
            "sample_id": sid,
            "source": t.get("source", "") or (rr or {}).get("source", ""),
            "gt_subtype13": t["gt_subtype13"],
            "gt_binary": t.get("gt_binary") or
                ("non_fastener" if t["gt_subtype13"] == "non_fastener" else "fastener"),
            "subset": t.get("subset", ""),
            "gt_source": t.get("gt_source", ""),
        })

        if rr is not None:
            base = rule_base_of(rr["rule_type_raw"])
            row["rule_type_raw"] = rr["rule_type_raw"]
            row["rule_confidence"] = rr["rule_confidence"]
            row["pred_rules"] = RULE_TO_SUBTYPE13.get(base, "non_fastener")
            claimed = base != "unclassified"
        else:
            claimed = False

        if mr is not None:
            top3 = [(mr["top1"], float(mr["p1"] or 0))]
            if mr.get("top2"):
                top3.append((mr["top2"], float(mr["p2"] or 0)))
            if mr.get("top3"):
                top3.append((mr["top3"], float(mr["p3"] or 0)))
            row.update({"ml_top1": mr["top1"], "ml_p1": mr["p1"],
                        "ml_top2": mr.get("top2", ""), "ml_p2": mr.get("p2", ""),
                        "ml_top3": mr.get("top3", ""), "ml_p3": mr.get("p3", ""),
                        "ml_face_cap": mr.get("face_cap", 0)})
            row["pred_ml"] = mr["top1"] if mr["top1"] in CLASSES13 else "non_fastener"

        if "hybrid" in configs and rr is not None and mr is not None:
            if claimed:
                row["pred_hybrid"] = row["pred_rules"]
                row["stage_hybrid"] = "stage1"
            else:
                row["pred_hybrid"] = row["pred_ml"]
                row["stage_hybrid"] = "stage2"

        if "hybrid_topk" in configs and rr is not None and mr is not None:
            if claimed:
                row["pred_hybrid_topk"] = row["pred_rules"]
                row["stage_hybrid_topk"] = "stage1"
                row["topk_action"] = ""
            else:
                gr = geom_rows.get(sid)
                geom = None
                if gr is not None and not gr.get("error") and gr.get("geom_json"):
                    geom = json.loads(gr["geom_json"])
                if int(mr.get("face_cap") or 0):
                    # face-capped parts have no top-3 distribution to recover from
                    pred, action = row["pred_ml"], "face_cap"
                else:
                    pred, action = apply_topk(top3, geom)
                row["pred_hybrid_topk"] = pred
                row["stage_hybrid_topk"] = "stage2"
                row["topk_action"] = action

        out.append(row)
    return out


def config_summary(rows: list[dict], pred_key: str, stage_key: str | None) -> dict:
    y_true = [r["gt_subtype13"] for r in rows]
    y_pred = [r[pred_key] for r in rows]
    correct = np.array([a == b for a, b in zip(y_true, y_pred)])
    out = {
        "subtype13": multi13_metrics(y_true, y_pred),
        "subtype13_accuracy_bootstrap": bootstrap_acc_ci(correct),
        "binary_collapse": binary_metrics(
            [r["gt_binary"] for r in rows],
            ["non_fastener" if p == "non_fastener" else "fastener" for p in y_pred]),
    }
    if stage_key:
        att = {}
        for stage in ("stage1", "stage2"):
            sub = [r for r in rows if r[stage_key] == stage]
            n_corr = sum(1 for r in sub if r[pred_key] == r["gt_subtype13"])
            att[stage] = {"n": len(sub), "n_correct": n_corr,
                          "accuracy": n_corr / len(sub) if sub else None}
        out["stage_attribution"] = att
    return out


def write_confusion(out_dir: Path, config: str, sub: dict):
    lines = ["," + ",".join(CLASSES13)]
    for cname, row in zip(CLASSES13, sub["confusion_matrix"]):
        lines.append(cname + "," + ",".join(str(v) for v in row))
    (out_dir / f"confusion_matrix_{config}.csv").write_text(
        "\n".join(lines), encoding="utf-8")


def summarize(testset: str, out_dir: Path, tasks: list[dict],
              rules_rows: dict, ml_rows: dict, geom_rows: dict,
              configs: list[str], extra: dict) -> dict:
    final = combine(tasks, rules_rows, ml_rows, geom_rows, configs)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FINAL_FIELDS)
        w.writeheader()
        for r in final:
            w.writerow(r)

    pred_keys = {"rules": ("pred_rules", None), "ml": ("pred_ml", None),
                 "hybrid": ("pred_hybrid", "stage_hybrid"),
                 "hybrid_topk": ("pred_hybrid_topk", "stage_hybrid_topk")}
    summary: dict = {
        "testset": testset,
        "n_common_scored": len(final),
        "configs": {},
        "protocol": {
            "stage1": "rule_based.classify_part, frozen DetectionConfig() "
                      "defaults (rule_confidence_threshold=0.60), "
                      "repetition_count=1; claimed iff fastener_type != "
                      "'unclassified' (the Experiment-1 protocol)",
            "stage2": f"BRepFormer v6 ({V6_CKPT}), raw argmax, no thresholds",
            "topk": "production _ml_with_topk_recovery imported from "
                    "step_vr_step.detection.ml_classifier (geometry summary "
                    "round-tripped through geom_json)",
            "names": "ground truth + coverage statistic ONLY — never in any "
                     "prediction path",
        },
    }
    for cfg in configs:
        pk, sk = pred_keys[cfg]
        s = config_summary(final, pk, sk)
        summary["configs"][cfg] = s
        write_confusion(out_dir, cfg, s["subtype13"])
        if cfg == "hybrid_topk":
            s["topk_action_histogram"] = dict(Counter(
                r["topk_action"] for r in final if r["stage_hybrid_topk"] == "stage2"))

    # subsets (paddle iso350; satellite gt_source splits)
    subset_vals = sorted({r["subset"] for r in final if r["subset"]})
    if subset_vals:
        summary["subsets"] = {}
        for sv in subset_vals:
            sub_rows = [r for r in final if r["subset"] == sv]
            summary["subsets"][sv] = {
                cfg: config_summary(sub_rows, pred_keys[cfg][0], pred_keys[cfg][1])
                for cfg in configs}
            summary["subsets"][sv]["n"] = len(sub_rows)
    gtsrc_vals = sorted({r["gt_source"] for r in final if r["gt_source"]})
    if len(gtsrc_vals) > 1:
        summary["by_gt_source"] = {}
        for gv in gtsrc_vals:
            sub_rows = [r for r in final if r["gt_source"] == gv]
            summary["by_gt_source"][gv] = {
                "n": len(sub_rows),
                **{cfg: {"accuracy": float(np.mean([
                    r[pred_keys[cfg][0]] == r["gt_subtype13"] for r in sub_rows]))}
                   for cfg in configs},
            }

    # pairwise McNemar on the common subset
    if len(configs) >= 2:
        summary["mcnemar_pairwise"] = {}
        cvec = {cfg: np.array([r[pred_keys[cfg][0]] == r["gt_subtype13"]
                               for r in final]) for cfg in configs}
        for i, a in enumerate(configs):
            for b in configs[i + 1:]:
                summary["mcnemar_pairwise"][f"{a}_vs_{b}"] = {
                    f"{a}_acc": float(cvec[a].mean()),
                    f"{b}_acc": float(cvec[b].mean()),
                    "mcnemar": mcnemar(cvec[a], cvec[b]),
                }
    summary.update(extra)
    summary["provenance"] = {
        "script": "backend/scripts/eval_hybrid.py",
        "v6_ckpt": str(V6_CKPT),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(f"summary -> {out_dir / 'summary.json'}")

    # console headline
    for cfg in configs:
        s = summary["configs"][cfg]["subtype13"]
        line = f"  {cfg:<12} acc={s['accuracy']*100:6.2f}%  macroF1={s['macro_f1']*100:6.2f}%  n={s['n']}"
        att = summary["configs"][cfg].get("stage_attribution")
        if att:
            s1, s2 = att["stage1"], att["stage2"]
            line += (f"   [stage1 {s1['n_correct']}/{s1['n']}"
                     f" | stage2 {s2['n_correct']}/{s2['n']}]")
        print(line)
    return summary


# ---------------------------------------------------------------------------
# Per-testset drivers
# ---------------------------------------------------------------------------

def needed_configs(arg: str) -> list[str]:
    return ["rules", "ml", "hybrid", "hybrid_topk"] if arg == "all" else [arg]


def drive_v6test(out_dir: Path, configs: list[str], workers: int, batch: int,
                 limit: int, summary_only: bool):
    samples = enumerate_v6_feature_dirs()
    if limit:
        samples = samples[:limit]
    print(f"v6test: {len(samples)} feature samples (max_faces={V6_MAX_FACES})")

    # Stage-1 labels: reuse Experiment-1 per-sample predictions verbatim.
    rules_csv = RULES_EVAL_ROOT / "v6test" / "predictions.csv"
    rules_rows = load_done(rules_csv)
    print(f"  reusing Experiment-1 rules rows: {len(rules_rows)} from {rules_csv}")

    need_ml = any(c in configs for c in ("ml", "hybrid", "hybrid_topk"))
    need_geom = "hybrid_topk" in configs

    if need_ml and not summary_only:
        run_ml_phase([(s["sample_id"], s["feature_dir"]) for s in samples],
                     out_dir / "ml_predictions.csv", batch)
    ml_rows = load_done(out_dir / "ml_predictions.csv")

    geom_rows: dict[str, dict] = {}
    if need_geom:
        # geometry only for samples that fall through to Stage 2
        fallback = {sid for sid, r in rules_rows.items()
                    if not r.get("error")
                    and rule_base_of(r["rule_type_raw"]) == "unclassified"}
        fallback &= {s["sample_id"] for s in samples}
        if not summary_only:
            singles, groups = build_v6_geom_tasks(fallback)
            run_geom_tasks(singles, groups, out_dir / "geom_rules.csv", workers)
        geom_rows = load_done(out_dir / "geom_rules.csv")
        # sanity: re-extracted rules should agree with the cached Experiment-1
        # labels (same frozen code) — report any drift
        n_drift = sum(
            1 for sid, gr in geom_rows.items()
            if sid in rules_rows and not gr.get("error")
            and rule_base_of(gr["rule_type_raw"]) != rule_base_of(rules_rows[sid]["rule_type_raw"]))
        if n_drift:
            print(f"  WARN: {n_drift} rule-label drift(s) between cached and "
                  f"re-extracted geometry pass")

    tasks = [{"sample_id": s["sample_id"], "gt_subtype13": s["gt_subtype13"],
              "source": s["sample_id"].split("/", 1)[1].split("__")[0]}
             for s in samples]

    extra: dict = {"name_coverage": {
        "note": "Not applicable: v6test is a per-part dataset; labels come "
                "from the dataset class tree, not from in-assembly names."}}
    # ML-alone reference on the FULL 7,286 (vs stored v6 eval)
    ml_ok = [s for s in samples
             if s["sample_id"] in ml_rows and not ml_rows[s["sample_id"]].get("error")
             and ml_rows[s["sample_id"]].get("top1")]
    if ml_ok:
        y_t = [s["gt_subtype13"] for s in ml_ok]
        y_p = [ml_rows[s["sample_id"]]["top1"] for s in ml_ok]
        extra["ml_full_testset"] = {
            "n": len(ml_ok),
            "subtype13": multi13_metrics(y_t, y_p),
        }
        if V6_EVAL_SUMMARY.exists():
            v6 = json.loads(V6_EVAL_SUMMARY.read_text())["v2_full_test"]
            extra["stored_v6_reference"] = {
                "n": v6["n"], "accuracy": v6["accuracy"],
                "macro_f1": v6["macro_f1"], "source": str(V6_EVAL_SUMMARY)}
    summarize("v6test", out_dir, tasks, rules_rows, ml_rows, geom_rows,
              configs, extra)


def drive_paddle(out_dir: Path, configs: list[str], workers: int, batch: int,
                 limit: int, summary_only: bool):
    tasks = paddle_gt_tasks()
    if limit:
        tasks = tasks[:limit]
    print(f"paddle: {len(tasks)} holdout solids "
          f"({sum(1 for t in tasks if t['subset'] == 'iso350')} iso350)")

    rules_csv = RULES_EVAL_ROOT / "paddle" / "predictions.csv"
    rules_rows = load_done(rules_csv)
    print(f"  reusing Experiment-1 rules rows: {len(rules_rows)} from {rules_csv}")

    need_ml = any(c in configs for c in ("ml", "hybrid", "hybrid_topk"))
    need_geom = "hybrid_topk" in configs

    if need_ml and not summary_only:
        run_ml_phase([(t["sample_id"], t["feature_dir"]) for t in tasks],
                     out_dir / "ml_predictions.csv", batch)
    ml_rows = load_done(out_dir / "ml_predictions.csv")

    geom_rows: dict[str, dict] = {}
    if need_geom:
        fallback = [t for t in tasks
                    if t["sample_id"] in rules_rows
                    and rule_base_of(rules_rows[t["sample_id"]]["rule_type_raw"])
                    == "unclassified"]
        if not summary_only and fallback:
            groups = [(str(PADDLE_STEP), [
                {"sample_id": t["sample_id"], "source": "paddle_holdout",
                 "source_path": str(PADDLE_STEP), "solid_idx": t["solid_idx"],
                 "expected_solid_name": t["expected_solid_name"]}
                for t in fallback])]
            run_geom_tasks([], groups, out_dir / "geom_rules.csv", workers=1)
        geom_rows = load_done(out_dir / "geom_rules.csv")

    extra = {
        "name_coverage": paddle_name_coverage(out_dir),
        "journal_reference": {
            "bf_v1_iso350": 0.031, "bf_v6_iso350": 0.640,
            "rules_iso350_experiment1": 0.626,
            "source": "RESEARCH_JOURNAL.md 2026-05-21->05-25 and 2026-06-12 Experiment 1",
        },
    }
    # cross-check our recomputed ML against the stored v6 paddle run
    summarize("paddle", out_dir, tasks, rules_rows, ml_rows, geom_rows,
              configs, extra)


def drive_satellite(out_dir: Path, configs: list[str], workers: int, batch: int,
                    limit: int, summary_only: bool):
    need_ml = any(c in configs for c in ("ml", "hybrid", "hybrid_topk"))
    meta_path = out_dir / "satellite_meta.json"
    if not summary_only:
        meta = run_satellite(out_dir, need_ml, limit)
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    rules_rows = load_done(out_dir / "geom_rules.csv")
    ml_rows = load_done(out_dir / "ml_predictions.csv")
    geom_rows = rules_rows  # same artifact carries the geometry summaries

    tasks = [{"sample_id": s["sample_id"], "gt_subtype13": s["gt_subtype13"],
              "gt_binary": "fastener", "source": "satellite",
              "gt_source": s["gt_source"]} for s in meta["scored"]]
    extra = {
        "name_coverage": meta["coverage"],
        "gt_table": [{"pattern": p, "class": c, "source": s, "note": n}
                     for p, c, s, n in SATELLITE_NAME_GT],
        "journal_reference": {
            "ensemble_topk_2026_05_19": 0.857,
            "raw_bf_subtype13_2026_05_19": 0.54,
            "n_named_2026_05_19": 168,
            "note": ("2026-05-19 numbers used the production ensemble "
                     "(rule threshold 0.35, ensemble_merge, repetition "
                     "bonus) with bf_subtype13_best (v1) — protocol and "
                     "checkpoint differ; indicative comparison only."),
        },
    }
    summarize("satellite", out_dir, tasks, rules_rows, ml_rows, geom_rows,
              configs, extra)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DRIVERS = {"v6test": drive_v6test, "paddle": drive_paddle,
           "satellite": drive_satellite}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testset", required=True, choices=sorted(DRIVERS))
    ap.add_argument("--config", default="all",
                    choices=["rules", "ml", "hybrid", "hybrid_topk", "all"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true",
                    help="Skip computation; rebuild summary from existing CSVs")
    args = ap.parse_args()

    out_dir = args.out_root / args.testset
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = needed_configs(args.config)
    print(f"testset={args.testset}  configs={configs}  out={out_dir}")
    DRIVERS[args.testset](out_dir, configs, args.workers, args.batch_size,
                          args.limit, args.summary_only)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
