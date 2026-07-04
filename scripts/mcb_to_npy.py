"""Convert MCB-A OBJ meshes to .npy point clouds matching FastenerPointCloudDataset layout.

Input:   training_data/mcb/dataset_org_norm/{train,test}/<class>/*.obj
Output:  training_data/mcb/npy/{train,test}/<class>/*.npy   shape (N, 6) = [xyz, nx, ny, nz]

Usage:
    python backend/scripts/mcb_to_npy.py \
        --src training_data/mcb/dataset_org_norm \
        --dst training_data/mcb/npy \
        --num_points 2048 \
        --workers 8
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm


def convert_one(args: tuple[Path, Path, int]) -> tuple[Path, bool, str]:
    obj_path, out_path, num_points = args
    if out_path.exists():
        return obj_path, True, "skip"
    try:
        mesh = trimesh.load(obj_path, process=False, force="mesh")
        if mesh.is_empty or len(mesh.faces) == 0:
            return obj_path, False, "empty"
        pts, face_idx = trimesh.sample.sample_surface(mesh, num_points)
        normals = mesh.face_normals[face_idx]
        data = np.concatenate([pts, normals], axis=1).astype(np.float32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, data)
        return obj_path, True, "ok"
    except Exception as e:
        return obj_path, False, f"err:{type(e).__name__}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = p.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()

    tasks = []
    for split in ("train", "test"):
        split_dir = src / split
        if not split_dir.is_dir():
            continue
        for obj in split_dir.rglob("*.obj"):
            rel = obj.relative_to(src).with_suffix(".npy")
            tasks.append((obj, dst / rel, args.num_points))

    print(f"Converting {len(tasks)} files with {args.workers} workers...")
    ok = failed = skipped = 0
    failures: list[tuple[Path, str]] = []
    with mp.Pool(args.workers) as pool:
        for path, success, msg in tqdm(
            pool.imap_unordered(convert_one, tasks, chunksize=16),
            total=len(tasks),
        ):
            if msg == "skip":
                skipped += 1
            elif success:
                ok += 1
            else:
                failed += 1
                failures.append((path, msg))

    print(f"\nDone. ok={ok} skipped={skipped} failed={failed}")
    if failures:
        print("First 20 failures:")
        for pth, msg in failures[:20]:
            print(f"  {msg}: {pth}")


if __name__ == "__main__":
    main()
