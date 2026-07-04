"""Re-extract the McMaster subtype-13 point clouds with REAL surface normals.

The 2026-05-11 matched subtype-13 comparison (BF 89.98% vs PN 75.64%, BF
+14.34pp — the thesis's load-bearing pro-B-rep result) trained PointNet++ on
`mcmaster_pc_subtype13`, whose every sample carries **degenerate constant
(0,0,1) normals** (pipeline.py's triangulation-normal fallback fired on every
point). This script rebuilds the *identical* splits/part-numbers/classes with
genuine per-face normals (trimesh surface sampling, the step_to_npy protocol),
so the only variable that changes vs the original PN++ run is the normals.

Input  (defines the exact split membership): training_data/mcmaster_pc_subtype13/{split}/{cls}/<pn>.npy
Source geometry: fastener_labeling/dataset/{fastener,non_fastener}/**/<pn>.step (whole OneShape, all faces)
Output: training_data/mcmaster_pc_subtype13_normals/{split}/{cls}/<pn>.npy  float32 (4096, 6)

Resume-safe (skip existing), multiprocessing with __main__ guard (2026-05-09
Windows spawn-cascade lesson), per-split retention logged.

Usage (anaconda stepvrstep env): python backend/scripts/prepare_pn_subtype13_normals.py --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import sys
import time
import warnings

warnings.filterwarnings("ignore")
from collections import Counter
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import numpy as np

# Reuse the v6 sampling worker (OCC tessellation -> trimesh surface sampling)
from prepare_pn_v6_dataset import shape_to_pointcloud, _read_step_oneshape

REPO = _BACKEND_ROOT.parent
DATASET = REPO / "fastener_labeling" / "dataset"
PC_SRC = REPO / "training_data" / "mcmaster_pc_subtype13"
PC_DST = REPO / "training_data" / "mcmaster_pc_subtype13_normals"
RUN_DIR = REPO / "training_data" / "mcmaster_logs"
NUM_POINTS = 4096
SPLITS = ("train", "val", "test")


def build_pn_to_step() -> dict[str, Path]:
    """Map McMaster part-number stem -> source STEP path (same tree
    relabel_subtype_13.build_pn_to_label walked)."""
    idx: dict[str, Path] = {}
    for klass_dir in ("fastener", "non_fastener"):
        root = DATASET / klass_dir
        if not root.exists():
            continue
        for step_file in root.rglob("*.step"):
            if step_file.name.startswith("._"):
                continue
            idx.setdefault(step_file.stem, step_file)
    return idx


def _worker(task: tuple[str, str, str]) -> tuple[str, str, str]:
    """(out_path, step_path, sample_id) -> (sample_id, status, error)."""
    out_path_str, step_path_str, sample_id = task
    out_path = Path(out_path_str)
    if out_path.exists():
        return (sample_id, "skip", "")
    try:
        shape = _read_step_oneshape(step_path_str)
        pc = shape_to_pointcloud(shape, NUM_POINTS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, pc)
        return (sample_id, "ok", "")
    except Exception as e:
        return (sample_id, "fail", f"{type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    pn_to_step = build_pn_to_step()
    print(f"source STEP index: {len(pn_to_step)} part numbers", flush=True)

    tasks: list[tuple[str, str, str]] = []
    unresolved: list[str] = []
    per_split = Counter()
    for split in SPLITS:
        for cls_dir in sorted((PC_SRC / split).iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            for f in sorted(cls_dir.glob("*.npy")):
                stem = f.stem
                sid = f"{split}/{cls}/{stem}"
                step = pn_to_step.get(stem)
                if step is None:
                    unresolved.append(sid)
                    continue
                per_split[split] += 1
                out = PC_DST / split / cls / f"{stem}.npy"
                tasks.append((str(out), str(step), sid))

    print(f"resolved {len(tasks)} samples "
          f"({', '.join(f'{s}={per_split[s]}' for s in SPLITS)}); "
          f"unresolved {len(unresolved)}", flush=True)
    if unresolved[:5]:
        print(f"  unresolved e.g.: {unresolved[:5]}", flush=True)
    if args.dry_run:
        return

    log_path = RUN_DIR / "pn_subtype13_normals_extract.csv"
    n_ok = n_skip = n_fail = 0
    t0 = time.time()
    with multiprocessing.Pool(args.workers) as pool, \
         log_path.open("w", newline="", encoding="utf-8") as logf:
        w = csv.writer(logf)
        w.writerow(["sample_id", "status", "error"])
        for i, (sid, status, err) in enumerate(
                pool.imap_unordered(_worker, tasks, chunksize=8)):
            w.writerow([sid, status, err])
            n_ok += status == "ok"
            n_skip += status == "skip"
            n_fail += status == "fail"
            if (i + 1) % 500 == 0:
                logf.flush()
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(f"  {i+1}/{len(tasks)}  ok={n_ok} skip={n_skip} "
                      f"fail={n_fail}  {rate:.1f}/s", flush=True)

    # retention vs source PC tree
    ret = {}
    for split in SPLITS:
        src_n = sum(1 for _ in (PC_SRC / split).rglob("*.npy"))
        dst_n = sum(1 for _ in (PC_DST / split).rglob("*.npy"))
        ret[split] = {"src": src_n, "dst": dst_n}
    (RUN_DIR / "pn_subtype13_normals_retention.json").write_text(
        json.dumps(ret, indent=2), encoding="utf-8")
    print(f"done in {(time.time()-t0)/60:.1f} min  ok={n_ok} skip={n_skip} "
          f"fail={n_fail}", flush=True)
    for split in SPLITS:
        print(f"  {split}: {ret[split]['dst']}/{ret[split]['src']}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
