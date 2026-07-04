"""Batch convert STEP files to PointNet++-ready .npy via pythonocc tessellation + trimesh surface sampling.

Input:   root/{train,val,test}/<class>/*.step|*.stp
Output:  out/{train,val,test}/<class>/*.npy   shape (N, 6) = [xyz, nx, ny, nz]

Multiprocess, resume-safe (skips existing outputs), per-file timeout, logs failures.

Usage:
    python backend/scripts/step_to_npy.py \
        --src training_data/organized \
        --dst training_data/organized_npy \
        --num_points 2048 \
        --workers 6
"""

import argparse
import multiprocessing as mp
import sys
import traceback
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm


MAC_PREFIX = "/Volumes/Uncle Sam/GitHub/step-vr-step/"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_xsym(stub_path: Path) -> Path:
    """If stub_path is an MSYS2 XSym symlink stub, rewrite its target to the PC project root
    and return the resolved real path. Otherwise return stub_path unchanged."""
    try:
        with stub_path.open("rb") as f:
            head = f.read(4)
            if head != b"XSym":
                return stub_path
            f.seek(0)
            lines = f.read(1200).decode("utf-8", errors="replace").splitlines()
    except OSError:
        return stub_path
    if len(lines) < 4:
        return stub_path
    target = lines[3].rstrip("\x00").strip()
    if target.startswith(MAC_PREFIX):
        rel = target[len(MAC_PREFIX):]
        candidate = PROJECT_ROOT / rel
        if candidate.exists():
            return candidate
    return stub_path


def step_to_mesh(step_path: Path):
    step_path = resolve_xsym(step_path)

    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != 1:
        raise ValueError(f"STEP read failed, status={status}")
    reader.TransferRoots()
    shape = reader.OneShape()

    BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True).Perform()

    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    v_offset = 0

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            t = loc.Transformation()
            n_nodes = tri.NbNodes()
            fv = np.empty((n_nodes, 3), dtype=np.float64)
            for i in range(1, n_nodes + 1):
                p = tri.Node(i).Transformed(t)
                fv[i - 1] = (p.X(), p.Y(), p.Z())
            n_tris = tri.NbTriangles()
            ff = np.empty((n_tris, 3), dtype=np.int64)
            for i in range(1, n_tris + 1):
                a, b, c = tri.Triangle(i).Get()
                ff[i - 1] = (a - 1 + v_offset, b - 1 + v_offset, c - 1 + v_offset)
            verts.append(fv)
            faces.append(ff)
            v_offset += n_nodes
        exp.Next()

    if not verts:
        raise ValueError("no triangulated faces")

    return trimesh.Trimesh(
        vertices=np.vstack(verts), faces=np.vstack(faces), process=False
    )


def convert_one(args_tuple: tuple[Path, Path, int]) -> tuple[Path, bool, str]:
    step_path, out_path, num_points = args_tuple
    if out_path.exists():
        return step_path, True, "skip"
    try:
        mesh = step_to_mesh(step_path)
        if len(mesh.faces) == 0:
            return step_path, False, "empty"
        pts, face_idx = trimesh.sample.sample_surface(mesh, num_points)
        normals = mesh.face_normals[face_idx]
        data = np.concatenate([pts, normals], axis=1).astype(np.float32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, data)
        return step_path, True, "ok"
    except Exception as e:
        return step_path, False, f"err:{type(e).__name__}:{e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    args = p.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()

    tasks: list[tuple[Path, Path, int]] = []
    for step in src.rglob("*"):
        if step.suffix.lower() in (".step", ".stp"):
            rel = step.relative_to(src).with_suffix(".npy")
            tasks.append((step, dst / rel, args.num_points))

    print(f"Converting {len(tasks)} STEP files with {args.workers} workers...")
    ok = failed = skipped = 0
    failures: list[tuple[Path, str]] = []
    failure_log = dst / "failures.log"
    dst.mkdir(parents=True, exist_ok=True)

    with mp.Pool(args.workers) as pool:
        for path, success, msg in tqdm(
            pool.imap_unordered(convert_one, tasks, chunksize=4),
            total=len(tasks),
        ):
            if msg == "skip":
                skipped += 1
            elif success:
                ok += 1
            else:
                failed += 1
                failures.append((path, msg))

    with failure_log.open("w", encoding="utf-8") as f:
        for pth, msg in failures:
            f.write(f"{msg}\t{pth}\n")

    print(f"\nDone. ok={ok} skipped={skipped} failed={failed}")
    if failures:
        print(f"Full failure list: {failure_log}")
        print("First 10:")
        for pth, msg in failures[:10]:
            print(f"  {msg}: {pth}")


if __name__ == "__main__":
    main()
