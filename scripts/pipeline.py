"""Binary fastener-classifier training pipeline (PointNet++ + BRepFormer).

Phases:
  1. stage    - hash-split dataset/{fastener,non_fastener}/.../<pn>.step into
                mcmaster_binary/{train,val,test}/{fastener,non_fastener}/<pn>.step
  2. pc       - STEP -> point cloud .npy (parallel)
  3. brep     - STEP -> face_grids/edge_curves/topo_distances (parallel via existing step_to_brep.py)
  4. train_pn - train PointNet++
  5. train_bf - train BRepFormer
  6. eval     - run both checkpoints on test, write metrics

Writes pipeline_state.json so the terminal dashboard can render bars.

Usage:
    python backend/scripts/pipeline.py --start stage
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------- paths ----------
REPO = Path(r"c:\Users\ferri\OneDrive\Documents\GitHub\step-vr-step")
BACKEND = REPO / "backend"
DATASET = REPO / "fastener_labeling" / "dataset"
TRAIN_ROOT = REPO / "training_data" / "mcmaster_binary"
PC_ROOT = REPO / "training_data" / "mcmaster_pc"
BREP_ROOT = REPO / "training_data" / "mcmaster_brep"
LOGS = REPO / "training_data" / "mcmaster_logs"
STATE = REPO / "training_data" / "pipeline_state.json"
SPLIT_SEED = "mcmaster-binary-2026-05-09"
VAL_FRAC = 0.10
TEST_FRAC = 0.10
NUM_POINTS = 4096

sys.path.insert(0, str(BACKEND))

# ---------- state helpers ----------
PHASE_NAMES = ["stage", "pc", "brep", "train_pn", "train_bf", "eval"]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {"started": _now(), "phases": {p: {"status": "pending"} for p in PHASE_NAMES}}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE)


def update_phase(name: str, **kwargs) -> None:
    state = load_state()
    state["phases"].setdefault(name, {})
    state["phases"][name].update(kwargs)
    save_state(state)


# ---------- phase: stage ----------
def split_for(pn: str) -> str:
    h = int(hashlib.md5(f"{SPLIT_SEED}:{pn}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < TEST_FRAC:
        return "test"
    if h < TEST_FRAC + VAL_FRAC:
        return "val"
    return "train"


def phase_stage() -> None:
    update_phase("stage", status="running", start=_now())
    counts: dict[str, int] = {}
    written = 0
    skipped = 0
    for klass in ("fastener", "non_fastener"):
        src_root = DATASET / klass
        if not src_root.exists():
            continue
        for step in src_root.rglob("*.step"):
            if step.name.startswith("._"):
                continue
            split = split_for(step.stem)
            dst_dir = TRAIN_ROOT / split / klass
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{step.stem}.step"
            if dst.exists():
                skipped += 1
            else:
                shutil.copy2(step, dst)
                written += 1
            counts[f"{split}/{klass}"] = counts.get(f"{split}/{klass}", 0) + 1
            if (written + skipped) % 200 == 0:
                update_phase("stage", written=written, skipped=skipped, counts=counts)
    update_phase("stage", status="done", end=_now(), written=written, skipped=skipped, counts=counts)
    print(f"[stage] done — wrote {written}, skipped {skipped}")
    print(f"        " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))


# ---------- phase: pc (parallel STEP -> .npy) ----------
def step_to_pointcloud(step_path: Path, num_points: int = NUM_POINTS) -> np.ndarray:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != 1:
        raise ValueError("read fail")
    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True).Perform()

    verts: list[list[float]] = []
    normals: list[list[float]] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            has_n = tri.HasNormals()
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i)
                if not loc.IsIdentity():
                    p = p.Transformed(loc.Transformation())
                verts.append([p.X(), p.Y(), p.Z()])
                if has_n:
                    n = tri.Normal(i)
                    normals.append([n.X(), n.Y(), n.Z()])
                else:
                    normals.append([0.0, 0.0, 1.0])
        exp.Next()
    if not verts:
        raise ValueError("empty mesh")
    v = np.asarray(verts, dtype=np.float32)
    n = np.asarray(normals, dtype=np.float32)
    idx = np.random.choice(len(v), num_points, replace=(len(v) < num_points))
    return np.concatenate([v[idx], n[idx]], axis=1)


def _pc_worker(args: tuple[Path, Path]) -> tuple[str, str]:
    src, dst = args
    if dst.exists():
        return ("skip", str(src))
    try:
        import warnings
        warnings.filterwarnings("ignore")
        pts = step_to_pointcloud(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, pts)
        return ("ok", str(src))
    except Exception as e:
        return (f"err:{e.__class__.__name__}", str(src))


def phase_pc() -> None:
    update_phase("pc", status="running", start=_now())
    jobs: list[tuple[Path, Path]] = []
    for split in ("train", "val", "test"):
        for klass in ("fastener", "non_fastener"):
            src_dir = TRAIN_ROOT / split / klass
            if not src_dir.exists():
                continue
            for step in src_dir.glob("*.step"):
                dst = PC_ROOT / split / klass / f"{step.stem}.npy"
                jobs.append((step, dst))
    total = len(jobs)
    update_phase("pc", total=total, done=0, errors=0)
    if total == 0:
        update_phase("pc", status="done", end=_now())
        return

    workers = max(1, (os.cpu_count() or 4) - 1)
    done = errs = 0
    with mp.Pool(workers) as pool:
        for status, _src in pool.imap_unordered(_pc_worker, jobs, chunksize=4):
            if status.startswith("err"):
                errs += 1
            else:
                done += 1
            if (done + errs) % 25 == 0:
                update_phase("pc", done=done, errors=errs)
    update_phase("pc", status="done", end=_now(), done=done, errors=errs, total=total)
    print(f"[pc] {done} ok, {errs} errors of {total}")


# ---------- phase: brep ----------
def phase_brep() -> None:
    update_phase("brep", status="running", start=_now())
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / "brep_convert.log"
    cmd = [
        sys.executable, str(BACKEND / "scripts" / "step_to_brep.py"),
        "--src", str(TRAIN_ROOT),
        "--dst", str(BREP_ROOT),
        "--workers", str(max(1, (os.cpu_count() or 4) - 1)),
    ]
    print(f"[brep] {' '.join(cmd)}")
    with log.open("ab") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(BACKEND))
        # Poll: count produced face_grids.npy files for progress
        expected = sum(
            1 for split in ("train", "val", "test") for klass in ("fastener", "non_fastener")
            for _ in (TRAIN_ROOT / split / klass).glob("*.step") if (TRAIN_ROOT / split / klass).exists()
        )
        update_phase("brep", expected=expected)
        while proc.poll() is None:
            done = sum(1 for _ in BREP_ROOT.rglob("face_grids.npy")) if BREP_ROOT.exists() else 0
            update_phase("brep", done=done, expected=expected)
            time.sleep(15)
    done = sum(1 for _ in BREP_ROOT.rglob("face_grids.npy")) if BREP_ROOT.exists() else 0
    update_phase("brep", status="done", end=_now(), done=done, expected=expected, exit_code=proc.returncode)
    print(f"[brep] done — {done}/{expected} samples, exit {proc.returncode}")


# ---------- phase: train_pn ----------
def phase_train_pn(epochs: int) -> None:
    update_phase("train_pn", status="running", start=_now(), epochs=epochs, current_epoch=0)
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / "train_pn.log"
    log.write_text("")  # truncate
    cmd = [
        sys.executable, "-u", "-m", "step_vr_step.models.pointnet2.train",
        "--data_path", str(PC_ROOT),
        "--epochs", str(epochs),
        "--batch_size", "16",
        "--num_points", str(NUM_POINTS),
        "--num_workers", "2",
        "--use_normals",
        "--lr", "0.001",
        "--log_dir", str(LOGS / "pointnet2"),
    ]
    print(f"[train_pn] {' '.join(cmd)}")
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"
    with log.open("ab") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(BACKEND), env=env)
        # Tail-parse for "Epoch X/Y" lines
        last_pos = 0
        import re
        ep_re = re.compile(r"Epoch (\d+)/(\d+)")
        while proc.poll() is None:
            try:
                with log.open("rb") as rf:
                    rf.seek(last_pos)
                    chunk = rf.read().decode("utf-8", errors="replace")
                    last_pos = rf.tell()
                eps = ep_re.findall(chunk)
                if eps:
                    e, t = eps[-1]
                    update_phase("train_pn", current_epoch=int(e), epochs=int(t))
            except Exception:
                pass
            time.sleep(10)
    update_phase("train_pn", status="done" if proc.returncode == 0 else "error",
                 end=_now(), exit_code=proc.returncode)
    print(f"[train_pn] exit {proc.returncode}")


# ---------- phase: train_bf ----------
def phase_train_bf(epochs: int) -> None:
    update_phase("train_bf", status="running", start=_now(), epochs=epochs, current_epoch=0)
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / "train_bf.log"
    log.write_text("")
    cmd = [
        sys.executable, "-u", "-m", "step_vr_step.models.brepformer.train",
        "--data_dir", str(BREP_ROOT),
        "--epochs", str(epochs),
        "--batch_size", "8",
        "--num_workers", "2",
        "--lr", "0.001",
        "--log_dir", str(LOGS / "brepformer"),
        "--num_classes", "2",
    ]
    print(f"[train_bf] {' '.join(cmd)}")
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"
    with log.open("ab") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(BACKEND), env=env)
        last_pos = 0
        import re
        ep_re = re.compile(r"[Ee]poch[ =:]?\s*(\d+)\s*/\s*(\d+)")
        while proc.poll() is None:
            try:
                with log.open("rb") as rf:
                    rf.seek(last_pos)
                    chunk = rf.read().decode("utf-8", errors="replace")
                    last_pos = rf.tell()
                eps = ep_re.findall(chunk)
                if eps:
                    e, t = eps[-1]
                    update_phase("train_bf", current_epoch=int(e), epochs=int(t))
            except Exception:
                pass
            time.sleep(10)
    update_phase("train_bf", status="done" if proc.returncode == 0 else "error",
                 end=_now(), exit_code=proc.returncode)
    print(f"[train_bf] exit {proc.returncode}")


# ---------- phase: eval ----------
def phase_eval() -> None:
    update_phase("eval", status="running", start=_now())
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / "eval.log"
    out_metrics = LOGS / "metrics.json"
    metrics: dict = {}

    # PointNet++ test
    try:
        import torch
        from torch.utils.data import DataLoader
        from step_vr_step.models.pointnet2.pointnet2_cls_msg import PointNet2ClsMSG
        from step_vr_step.models.pointnet2.dataset import FastenerPointCloudDataset
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_dir = LOGS / "pointnet2"
        ckpts = sorted(ckpt_dir.glob("*.pth")) if ckpt_dir.exists() else []
        if ckpts:
            ckpt_path = ckpts[-1]
            test_ds = FastenerPointCloudDataset(root=str(PC_ROOT), num_points=NUM_POINTS,
                                                use_normals=True, split="test", augment=False)
            loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
            model = PointNet2ClsMSG(num_classes=test_ds.num_classes, use_normals=True).to(device)
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck.get("model_state_dict", ck))
            model.eval()
            correct = total = 0
            from collections import Counter
            cm = Counter()
            with torch.no_grad():
                for points, labels in loader:
                    points = points.transpose(1, 2).to(device)
                    labels = labels.to(device)
                    logits, _ = model(points)
                    preds = logits.argmax(-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    for p, l in zip(preds.tolist(), labels.tolist()):
                        cm[(l, p)] += 1
            metrics["pointnet2"] = {"acc": correct / max(total, 1), "n": total,
                                     "checkpoint": str(ckpt_path),
                                     "confusion": {f"{l}->{p}": c for (l, p), c in cm.items()},
                                     "classes": test_ds.classes}
    except Exception as e:
        metrics["pointnet2"] = {"error": f"{e.__class__.__name__}: {e}", "trace": traceback.format_exc()}

    # BRepFormer test (best checkpoint from lightning)
    try:
        import torch
        from torch.utils.data import DataLoader
        from step_vr_step.models.brepformer.dataset import BRepDataset, brep_collate_fn
        from step_vr_step.models.brepformer.brepformer import BRepFormer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_glob = list((LOGS / "brepformer").rglob("*.ckpt")) if (LOGS / "brepformer").exists() else []
        if ckpt_glob:
            ckpt_path = sorted(ckpt_glob, key=lambda p: p.stat().st_mtime)[-1]
            test_ds = BRepDataset(root=str(BREP_ROOT), split="test")
            loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2,
                                collate_fn=brep_collate_fn)
            model = BRepFormer(num_classes=test_ds.num_classes).to(device)
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = ck.get("state_dict", ck)
            sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
            model.eval()
            correct = total = 0
            from collections import Counter
            cm = Counter()
            with torch.no_grad():
                for batch in loader:
                    out = model(face_grids=batch["face_grids"].to(device),
                                edge_curves=batch["edge_curves"].to(device),
                                topo_distances={k: v.to(device) for k, v in batch["topo_distances"].items()},
                                mask=batch["face_mask"].to(device),
                                edge_mask=batch["edge_mask"].to(device))
                    labels = batch["labels"].to(device)
                    preds = out.argmax(-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    for p, l in zip(preds.tolist(), labels.tolist()):
                        cm[(l, p)] += 1
            metrics["brepformer"] = {"acc": correct / max(total, 1), "n": total,
                                      "checkpoint": str(ckpt_path),
                                      "confusion": {f"{l}->{p}": c for (l, p), c in cm.items()},
                                      "classes": test_ds.classes}
    except Exception as e:
        metrics["brepformer"] = {"error": f"{e.__class__.__name__}: {e}", "trace": traceback.format_exc()}

    out_metrics.write_text(json.dumps(metrics, indent=2))
    update_phase("eval", status="done", end=_now(), metrics=metrics)
    print(f"[eval] metrics -> {out_metrics}")
    print(json.dumps({k: (v.get("acc", v.get("error"))) for k, v in metrics.items()}, indent=2))


# ---------- driver ----------
PHASES = {
    "stage": phase_stage, "pc": phase_pc, "brep": phase_brep,
    "train_pn": lambda: phase_train_pn(epochs=120),
    "train_bf": lambda: phase_train_bf(epochs=120),
    "eval": phase_eval,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="stage", choices=PHASE_NAMES,
                   help="run from this phase to the end")
    p.add_argument("--only", choices=PHASE_NAMES, help="run only this phase")
    p.add_argument("--skip", nargs="*", default=[], choices=PHASE_NAMES, help="skip these phases")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.only:
        plan = [args.only]
    else:
        i = PHASE_NAMES.index(args.start)
        plan = [n for n in PHASE_NAMES[i:] if n not in args.skip]
    print(f"pipeline plan: {' -> '.join(plan)}")
    for ph in plan:
        try:
            PHASES[ph]()
        except Exception as e:
            update_phase(ph, status="error", end=_now(), error=f"{e.__class__.__name__}: {e}",
                         trace=traceback.format_exc())
            print(f"[{ph}] FAILED: {e}", file=sys.stderr)
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
