"""Batch convert STEP files to BRepFormer's B-Rep feature format.

Input:   root/{train,val,test}/<class>/*.step|*.stp
Output:  out/{train,val,test}/<class>/<stem>/face_grids.npy, edge_curves.npy, topo_distances.npz

Multiprocess, resume-safe (skips existing outputs), handles macOS XSym symlink stubs.

Usage:
    python backend/scripts/step_to_brep.py \
        --src training_data/organized \
        --dst training_data/organized_brep \
        --workers 4
"""

import argparse
import concurrent.futures
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Resolve project-root paths so workers can import step_vr_step and resolve XSym stubs
_BACKEND = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _BACKEND.parent
sys.path.insert(0, str(_BACKEND))

MAC_PREFIX = "/Volumes/Uncle Sam/GitHub/step-vr-step/"


def resolve_xsym(stub_path: Path) -> Path:
    """Dereference macOS XSym symlink stubs that got copied as files during Mac→PC transfer."""
    try:
        with stub_path.open("rb") as f:
            if f.read(4) != b"XSym":
                return stub_path
            f.seek(0)
            lines = f.read(1200).decode("utf-8", errors="replace").splitlines()
    except OSError:
        return stub_path
    if len(lines) < 4:
        return stub_path
    target = lines[3].rstrip("\x00").strip()
    if target.startswith(MAC_PREFIX):
        candidate = _PROJECT_ROOT / target[len(MAC_PREFIX):]
        if candidate.exists():
            return candidate
    return stub_path


MAX_STEP_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB — anything bigger is satellite-scale and will hang workers
MAX_FACES = 600  # restored to baseline cap after MAX_FACES=1500 run had chaotic optimization (huge-sample gradient noise)


def process_one(args_tuple: tuple[Path, Path, int]) -> tuple[Path, bool, str]:
    """Process one STEP file.

    args_tuple = (step_path, out_dir, class_idx)
    class_idx: if >= 0, write face_labels.npy filled with this index; if < 0, skip face labels.
    """
    import warnings
    warnings.filterwarnings("ignore")

    step_path, out_dir, class_idx = args_tuple
    needs_labels = class_idx >= 0
    required_files = ["face_grids.npy", "edge_curves.npy", "topo_distances.npz"]
    if needs_labels:
        required_files.append("face_labels.npy")
    if all((out_dir / f).exists() for f in required_files):
        return step_path, True, "skip"

    try:
        step_path = resolve_xsym(step_path)
        # Defensive: skip giant STEP files that hang workers
        if step_path.exists() and step_path.stat().st_size > MAX_STEP_SIZE_BYTES:
            return step_path, False, f"too_large:{step_path.stat().st_size // 1024 // 1024}MB"
        from OCC.Core.STEPControl import STEPControl_Reader
        from step_vr_step.models.brepformer.feature_extractor import (
            extract_face_uv_grids, extract_edge_curves, compute_topology_distances,
        )
        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_path)) != 1:
            return step_path, False, "read_failed"
        reader.TransferRoots()
        shape = reader.OneShape()
        if shape.IsNull():
            return step_path, False, "null_shape"

        face_grids = extract_face_uv_grids(shape)
        if face_grids.shape[0] == 0:
            return step_path, False, "no_faces"
        if face_grids.shape[0] > MAX_FACES:
            return step_path, False, f"too_many_faces:{face_grids.shape[0]}"
        edge_curves = extract_edge_curves(shape)
        topo = compute_topology_distances(shape)

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "face_grids.npy", face_grids)
        np.save(out_dir / "edge_curves.npy", edge_curves)
        np.savez(out_dir / "topo_distances.npz", **topo)
        if needs_labels:
            Nf = face_grids.shape[0]
            face_labels = np.full((Nf,), class_idx, dtype=np.int64)
            np.save(out_dir / "face_labels.npy", face_labels)
        return step_path, True, f"ok_nf={face_grids.shape[0]}_ne={edge_curves.shape[0]}"
    except Exception as e:
        return step_path, False, f"err:{type(e).__name__}:{e}"


def class_from_path(step_path: Path, src_root: Path) -> str | None:
    """Derive a class name from the STEP path, assuming layout:
        <src>/<split>/<class>/<stem>.stp
    Returns the class folder name, or None if the path doesn't match that pattern.
    """
    try:
        rel = step_path.resolve().relative_to(src_root.resolve())
    except ValueError:
        return None
    parts = rel.parts
    # parts: (split, class, stem.stp)
    if len(parts) >= 3:
        return parts[1]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    p.add_argument("--timeout", type=int, default=120, help="Per-file timeout in seconds")
    p.add_argument("--face_labels_from_part", action="store_true",
                   help="Also write face_labels.npy per sample. Every face in a part gets the part's "
                        "class label (derived from <src>/<split>/<class>/<stem>.stp layout). "
                        "Used to train BRepFormer in segmentation mode on part-level labels.")
    args = p.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    dst.mkdir(parents=True, exist_ok=True)

    # Build sorted class index if requested — one lookup shared by all workers
    class_to_idx: dict[str, int] = {}
    if args.face_labels_from_part:
        # Discover classes from the src tree: walk depth-2 dirs (split/class/)
        classes: set[str] = set()
        for split_dir in src.iterdir():
            if split_dir.is_dir() and not split_dir.name.startswith("."):
                for cls_dir in split_dir.iterdir():
                    if cls_dir.is_dir() and not cls_dir.name.startswith("."):
                        classes.add(cls_dir.name)
        class_to_idx = {c: i for i, c in enumerate(sorted(classes))}
        if not class_to_idx:
            raise SystemExit(
                "--face_labels_from_part requires the src layout <src>/<split>/<class>/<stem>.stp; "
                f"no class folders found under {src}"
            )
        # Persist the class map so downstream eval/dataset can use it
        import json
        with (dst / "classes.json").open("w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, indent=2)
        print(f"Class index map written to {dst / 'classes.json'}: {class_to_idx}")

    tasks: list[tuple[Path, Path, int]] = []
    for step in src.rglob("*"):
        if step.suffix.lower() in (".step", ".stp"):
            rel = step.relative_to(src)
            out_dir = dst / rel.parent / rel.stem
            if args.face_labels_from_part:
                cls_name = class_from_path(step, src)
                if cls_name is None or cls_name not in class_to_idx:
                    # Skip files that don't fit <split>/<class>/ layout
                    continue
                class_idx = class_to_idx[cls_name]
            else:
                class_idx = -1
            tasks.append((step, out_dir, class_idx))

    print(f"Processing {len(tasks)} STEP files with {args.workers} workers, timeout={args.timeout}s per file...")
    ok = failed = skipped = 0
    failures: list[tuple[Path, str]] = []

    # mp.Pool with maxtasksperchild: workers are recycled after N tasks.
    # Prevents pythonocc memory accumulation that killed the pool last time.
    # Size/face guards inside process_one handle pathological meshes early.
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
    if failures:
        print(f"Failure log: {log}")
        print("First 10:")
        for pth, msg in failures[:10]:
            print(f"  {msg}: {pth}")


if __name__ == "__main__":
    main()
