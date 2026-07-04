"""Preprocess STEP files into point cloud .npy files for PointNet++ training.

Reads STEP files from the organized dataset directory, tessellates them,
samples point clouds with normals, and saves as .npy files.

Usage:
    python -m step_vr_step.models.preprocess \
        --input training_data/organized \
        --output training_data/pointclouds \
        --num_points 2048

Directory structure expected:
    input/
      train/
        fastener/
          bolt_m8.step
        non-fastener/
          bracket.step
      val/
        ...
      test/
        ...

Output:
    output/
      train/
        fastener/
          bolt_m8.npy      (2048, 6) float32
        non-fastener/
          bracket.npy
      val/
        ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

logger = logging.getLogger(__name__)


def load_step_shape(step_path: str):
    """Load a STEP file and return a TopoDS_Shape."""
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone

    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {step_path}")
    reader.TransferRoots()
    return reader.OneShape()


def tessellate_shape(shape, linear_deflection: float = 0.1, angular_deflection: float = 0.5):
    """Tessellate an OCC shape and return vertices + normals arrays."""
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.gp import gp_Vec

    # Tessellate
    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)

    vertices = []
    normals = []

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None:
            explorer.Next()
            continue

        trsf = loc.Transformation()
        n_nodes = triangulation.NbNodes()
        n_tris = triangulation.NbTriangles()

        for i in range(1, n_tris + 1):
            tri = triangulation.Triangle(i)
            n1, n2, n3 = tri.Get()

            p1 = triangulation.Node(n1).Transformed(trsf)
            p2 = triangulation.Node(n2).Transformed(trsf)
            p3 = triangulation.Node(n3).Transformed(trsf)

            # Face normal
            v1 = gp_Vec(p1, p2)
            v2 = gp_Vec(p1, p3)
            normal = v1.Crossed(v2)
            if normal.Magnitude() > 1e-12:
                normal.Normalize()
                nx, ny, nz = normal.X(), normal.Y(), normal.Z()
            else:
                nx, ny, nz = 0, 0, 1

            for p in (p1, p2, p3):
                vertices.append([p.X(), p.Y(), p.Z()])
                normals.append([nx, ny, nz])

        explorer.Next()

    if not vertices:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    return np.array(vertices, dtype=np.float32), np.array(normals, dtype=np.float32)


def sample_points(vertices: np.ndarray, normals: np.ndarray, num_points: int) -> np.ndarray:
    """Sample num_points from the tessellated mesh, returning (num_points, 6)."""
    if len(vertices) == 0:
        return np.zeros((num_points, 6), dtype=np.float32)

    if len(vertices) >= num_points:
        idx = np.random.choice(len(vertices), num_points, replace=False)
    else:
        idx = np.random.choice(len(vertices), num_points, replace=True)

    pts = vertices[idx]
    nrm = normals[idx]

    # Normalize to unit sphere
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-12:
        pts = pts / scale

    return np.concatenate([pts, nrm], axis=1).astype(np.float32)


def process_one_file(args: tuple) -> tuple[str, bool, str]:
    """Process a single STEP file. Returns (output_path, success, message)."""
    import signal
    import resource

    step_path, output_path, num_points = args

    # Limit memory per worker to 2GB to prevent OOM crashes
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, hard))
    except (ValueError, OSError):
        pass

    # Skip files larger than 50MB — they're likely full assemblies that will OOM
    try:
        file_size = Path(step_path).stat().st_size
        if file_size > 50 * 1024 * 1024:
            return str(output_path), False, f"Skipped: too large ({file_size // (1024*1024)}MB)"
    except OSError:
        pass

    try:
        shape = load_step_shape(str(step_path))
        verts, norms = tessellate_shape(shape)

        if len(verts) < 10:
            return str(output_path), False, f"Too few vertices ({len(verts)})"

        point_cloud = sample_points(verts, norms, num_points)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, point_cloud)
        return str(output_path), True, f"{len(verts)} verts -> {num_points} points"

    except MemoryError:
        return str(output_path), False, "Out of memory"
    except Exception as e:
        return str(output_path), False, str(e)


def preprocess_dataset(input_dir: Path, output_dir: Path, num_points: int, max_workers: int):
    """Preprocess all STEP files in the organized dataset."""
    tasks = []

    for split in ['train', 'val', 'test']:
        split_dir = input_dir / split
        if not split_dir.exists():
            continue

        for label_dir in split_dir.iterdir():
            if not label_dir.is_dir():
                continue

            label = label_dir.name
            for step_file in label_dir.iterdir():
                if step_file.suffix.lower() not in ('.step', '.stp'):
                    continue
                if step_file.name.startswith('._'):
                    continue
                # Resolve symlinks
                real_path = step_file.resolve()
                out_path = output_dir / split / label / (step_file.stem + '.npy')
                tasks.append((real_path, out_path, num_points))

    total = len(tasks)
    logger.info(f"Processing {total} STEP files with {max_workers} workers...")

    success = 0
    failed = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc="Preprocessing", unit="file")
    except ImportError:
        pbar = None

    # Process in batches to survive worker crashes
    batch_size = 100
    for batch_start in range(0, total, batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_one_file, t): t for t in batch}
                for future in as_completed(futures):
                    try:
                        out_path, ok, msg = future.result(timeout=120)
                        if ok:
                            success += 1
                        else:
                            failed += 1
                            logger.debug(f"  Failed: {Path(out_path).name} — {msg}")
                    except Exception as e:
                        failed += 1
                        logger.debug(f"  Worker error: {e}")
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix(ok=success, fail=failed)
        except Exception as e:
            # Pool crashed — count remaining batch as failed, continue
            remaining = len(batch) - (success + failed - batch_start)
            failed += max(remaining, 0)
            logger.warning(f"  Worker pool crashed at batch {batch_start}: {e}. Continuing...")
            if pbar:
                pbar.update(max(remaining, 0))

    if pbar:
        pbar.close()
    logger.info(f"Done. Success: {success}, Failed: {failed}")

    # Write class names
    classes = sorted([
        d.name for d in (output_dir / 'train').iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
    (output_dir / 'classes.txt').write_text('\n'.join(classes))
    logger.info(f"Classes: {classes}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess STEP files to point clouds")
    parser.add_argument("--input", required=True, help="Organized dataset directory")
    parser.add_argument("--output", required=True, help="Output directory for .npy files")
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    preprocess_dataset(Path(args.input), Path(args.output), args.num_points, args.workers)


if __name__ == "__main__":
    main()
