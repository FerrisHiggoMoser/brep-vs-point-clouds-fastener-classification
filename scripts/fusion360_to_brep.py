"""Preprocess Fusion 360 Gallery (s2.0.0) into BRepFormer segmentation format.

Input:   <extracted>/step/*.stp  +  <extracted>/seg/*.seg  +  train_test.json
Output:  <dst>/{train,val,test}/<stem>/face_grids.npy, edge_curves.npy, topo_distances.npz,
                                       face_labels.npy

Each .seg file: one integer per line, label per face in the order TopologyExplorer.faces() yields.
We carve 10% of train into val (deterministic seed).

Usage:
    python backend/scripts/fusion360_to_brep.py \
        --src training_data/fusion360/extracted \
        --dst training_data/fusion360_brep \
        --workers 4
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

MAX_FACES = 600


def process_one(args_tuple: tuple[Path, Path, Path]) -> tuple[Path, bool, str]:
    import warnings
    warnings.filterwarnings("ignore")

    step_path, seg_path, out_dir = args_tuple
    required = ["face_grids.npy", "edge_curves.npy", "topo_distances.npz", "face_labels.npy"]
    if all((out_dir / f).exists() for f in required):
        return step_path, True, "skip"

    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from step_vr_step.models.brepformer.feature_extractor import (
            extract_face_uv_grids, extract_edge_curves, compute_topology_distances,
        )

        # Load STEP
        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_path)) != 1:
            return step_path, False, "read_failed"
        reader.TransferRoots()
        shape = reader.OneShape()
        if shape.IsNull():
            return step_path, False, "null_shape"

        face_grids = extract_face_uv_grids(shape)
        Nf = face_grids.shape[0]
        if Nf == 0:
            return step_path, False, "no_faces"
        if Nf > MAX_FACES:
            return step_path, False, f"too_many_faces:{Nf}"

        edge_curves = extract_edge_curves(shape)
        topo = compute_topology_distances(shape)

        # Load seg file and check alignment
        labels = np.loadtxt(seg_path, dtype=np.int64).reshape(-1)
        if labels.shape[0] != Nf:
            return step_path, False, f"label_mismatch:Nf={Nf},labels={labels.shape[0]}"

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "face_grids.npy", face_grids)
        np.save(out_dir / "edge_curves.npy", edge_curves)
        np.savez(out_dir / "topo_distances.npz", **topo)
        np.save(out_dir / "face_labels.npy", labels.astype(np.int64))
        return step_path, True, f"ok_nf={Nf}"
    except Exception as e:
        return step_path, False, f"err:{type(e).__name__}"


def split_assignments(src: Path, val_fraction: float = 0.1, seed: str = "f360-brepformer"):
    """Read train_test.json and carve val_fraction from train deterministically."""
    with (src / "train_test.json").open() as f:
        split = json.load(f)
    train, test = split["train"], split["test"]

    def hash01(stem: str) -> float:
        h = hashlib.md5(f"{seed}:{stem}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    new_train, new_val = [], []
    for stem in train:
        if hash01(stem) < val_fraction:
            new_val.append(stem)
        else:
            new_train.append(stem)
    return {"train": new_train, "val": new_val, "test": test}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--max_per_split", type=int, default=None,
                   help="If set, process only this many files per split (for quick tests)")
    args = p.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    dst.mkdir(parents=True, exist_ok=True)

    splits = split_assignments(src, args.val_fraction)
    print(f"Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    tasks = []
    for split_name, stems in splits.items():
        if args.max_per_split:
            stems = stems[:args.max_per_split]
        for stem in stems:
            step_path = src / "step" / f"{stem}.stp"
            seg_path = src / "seg" / f"{stem}.seg"
            out_dir = dst / split_name / stem
            if not step_path.exists() or not seg_path.exists():
                continue
            tasks.append((step_path, seg_path, out_dir))

    print(f"Processing {len(tasks)} parts with {args.workers} workers (maxtasksperchild=10)...")
    ok = failed = skipped = 0
    failures: list[tuple[Path, str]] = []
    with mp.Pool(args.workers, maxtasksperchild=10) as pool:
        for path, success, msg in tqdm(
            pool.imap_unordered(process_one, tasks, chunksize=1),
            total=len(tasks),
        ):
            if msg == "skip":
                skipped += 1
            elif success:
                ok += 1
            else:
                failed += 1
                failures.append((path, msg))

    log = dst / "failures.log"
    with log.open("w", encoding="utf-8") as f:
        for pth, msg in failures:
            f.write(f"{msg}\t{pth}\n")

    print(f"\nDone. ok={ok} skipped={skipped} failed={failed}")
    if failures[:10]:
        print("First 10 failures:")
        for pth, msg in failures[:10]:
            print(f"  {msg}: {pth}")


if __name__ == "__main__":
    main()
