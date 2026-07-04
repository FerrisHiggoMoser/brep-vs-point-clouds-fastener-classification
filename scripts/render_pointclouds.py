"""Render point cloud .npy files as multi-view PNGs for visual inspection.

Given a list of (name, npy_path) pairs, produces one PNG per sample with 4 views:
isometric, top (XY), front (XZ), side (YZ). Used for eyeballing test-set errors.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def render(npy_path: Path, out_path: Path, title: str) -> None:
    data = np.load(npy_path)
    pts = data[:, :3]
    # normalize to unit sphere for consistent viewing
    pts = pts - pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 1e-9:
        pts = pts / scale

    fig = plt.figure(figsize=(12, 3))
    fig.suptitle(title, fontsize=10)
    views = [
        ("isometric", (30, 45)),
        ("top (XY)", (90, 0)),
        ("front (XZ)", (0, 0)),
        ("side (YZ)", (0, 90)),
    ]
    for i, (name, (elev, azim)) in enumerate(views, 1):
        ax = fig.add_subplot(1, 4, i, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.2, c=pts[:, 2], cmap="viridis")
        ax.set_title(name, fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True, type=Path,
                   help="text file with lines: <label>\\t<npy_path>")
    p.add_argument("--output_dir", required=True, type=Path)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines = [ln.strip() for ln in args.pairs.read_text(encoding="utf-8").splitlines() if ln.strip()]
    tasks: list[tuple[str, Path]] = []
    for ln in lines:
        label, path = ln.split("\t", 1)
        tasks.append((label, Path(path)))

    for label, path in tqdm(tasks, desc="Rendering"):
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)[:120]
        out = args.output_dir / f"{safe}.png"
        render(path, out, label)

    print(f"Wrote {len(tasks)} PNGs to {args.output_dir}")


if __name__ == "__main__":
    main()
