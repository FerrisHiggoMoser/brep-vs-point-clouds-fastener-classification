"""Carve a deterministic val split out of train/ for datasets that ship only train/test.

Moves ~val_ratio of files from each class in train/ into val/, seeded so reruns
are stable. Test is left untouched.
"""

import argparse
import hashlib
import shutil
from pathlib import Path


def file_hash_fraction(path: Path, seed: str) -> float:
    h = hashlib.md5(f"{seed}:{path.name}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path, help="dataset root (contains train/)")
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", default="mcb-pretrain-2026-04-21")
    args = p.parse_args()

    train = args.root / "train"
    val = args.root / "val"
    if val.exists() and any(val.rglob("*.npy")):
        print(f"val/ already has files — aborting to avoid double-splits")
        return

    moved = 0
    for cls_dir in sorted(train.iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.glob("*.npy"))
        to_move = [f for f in files if file_hash_fraction(f, args.seed) < args.val_ratio]
        if not to_move:
            continue
        dest_cls = val / cls_dir.name
        dest_cls.mkdir(parents=True, exist_ok=True)
        for f in to_move:
            shutil.move(str(f), str(dest_cls / f.name))
        moved += len(to_move)
        print(f"  {cls_dir.name}: {len(to_move)}/{len(files)} -> val")
    print(f"\nMoved {moved} files to val/")


if __name__ == "__main__":
    main()
