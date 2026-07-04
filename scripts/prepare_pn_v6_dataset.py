"""Build the PointNet++ twin of the BRepFormer v6 dataset.

For every sample BRepFormer v6 actually trained/validated/tested on
(bf_v2_features/{train,val,test}/<cls>/<sample_dir> with face_grids.npy
readable and <= 300 faces — the exact BRepDataset(max_faces=300) filter),
re-resolve the source geometry and sample a (4096, 6) xyz+normals point
cloud from the SAME solid:

    mcmaster__<stem>                      whole OneShape of the source STEP
    synthno__<stem> / synthrm__<stem>     whole OneShape
    realcad__fusion__<assy>__<uuid>       first solid of the per-body file
    realcad__<file>__<solid>__<idx>       solid <idx> of the decomposed assembly
    grabcad__<file>__<solid>__<idx>       solid <idx> of the decomposed assembly

Assembly decomposition vendors classify_assembly.load_assembly_with_names
(same enumeration that built the feature dirs, so solid indices line up);
source resolution mirrors eval_rule_based.build_tasks_v6test verbatim.

Point sampling mirrors step_to_npy.py: OCC tessellation
(BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5, True)) -> trimesh
surface sampling with per-face normals. NOTE this differs from
pipeline.py's vertex-node sampling, whose triangulation-normal fallback
produced constant (0,0,1) normals in every stored McMaster PC sample —
surface sampling gives PointNet++ genuine normals.

Output:
    pn_v6_features/<split>/<cls>/<sample_dir_name>.npy   float32 (4096, 6)
    pn_v6_run/manifest.csv          every v6 sample + resolution status
    pn_v6_run/extraction_log.csv    per-sample ok/skip/fail (resume index)
    pn_v6_run/retention.json        per-split per-source retention counts

Resume-safe: existing .npy outputs are skipped; samples already recorded
as failed in extraction_log.csv are not retried (use --retry-failed to
override). Safe to kill and relaunch at any point.

Usage (anaconda stepvrstep env — OCC + trimesh):
    python backend/scripts/prepare_pn_v6_dataset.py --dry-run
    python backend/scripts/prepare_pn_v6_dataset.py --workers 8
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

warnings.filterwarnings("ignore")
from collections import Counter, defaultdict
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import numpy as np

REPO = _BACKEND_ROOT.parent
D_THESIS = Path(r"D:\step-vr-step-thesis")
RB_TD = D_THESIS / "reproducible-build" / "training_data"

V6_FEATURES = RB_TD / "bf_v2_features"
OUT_FEATURES = RB_TD / "pn_v6_features"
RUN_DIR = RB_TD / "pn_v6_run"

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

V6_MAX_FACES = 300
NUM_POINTS = 4096
SPLITS = ("train", "val", "test")

CLASSES13 = [
    "anchors", "keys", "nails", "non_fastener", "nuts", "pins",
    "retaining-rings", "rivets", "screws", "spacers",
    "threaded-inserts", "threaded-rods", "washers",
]

MCMASTER_NON_FASTENER_SUBDIRS = (
    "brackets", "hinges", "mounting-plates", "pcbs", "t-slotted-framing",
)

MANIFEST_FIELDS = [
    "sample_id", "split", "class", "source", "source_path",
    "mode", "solid_idx", "expected_solid_name", "n_faces", "resolution",
]
LOG_FIELDS = ["sample_id", "status", "name_mismatch", "error"]


def safe_name(s: str) -> str:
    """Identical to extract_real_cad_fasteners.safe_name (used to build the
    v6 sample dir names) — needed to reverse-map dirs to source files."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:80]


# ---------------------------------------------------------------------------
# OCC helpers (imports stay inside functions: Windows spawn-safe)
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


def load_assembly_with_names(step_path: Path):
    """Vendored verbatim (minus tqdm) from scripts/classify_assembly.py so the
    solid ordering matches the one used to build the realcad/grabcad feature
    dirs (sample ids embed the solid index of this enumeration)."""
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


def shape_to_pointcloud(shape, num_points: int = NUM_POINTS) -> np.ndarray:
    """OCC tessellation -> trimesh surface sampling, (num_points, 6) float32.
    Mirrors step_to_npy.step_to_mesh + convert_one, parameterized on shape."""
    import trimesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods

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
            trsf = loc.Transformation()
            identity = loc.IsIdentity()
            fv = np.empty((tri.NbNodes(), 3), dtype=np.float64)
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i)
                if not identity:
                    p = p.Transformed(trsf)
                fv[i - 1] = (p.X(), p.Y(), p.Z())
            ft = np.empty((tri.NbTriangles(), 3), dtype=np.int64)
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                ft[i - 1] = (a - 1, b - 1, c - 1)
            verts.append(fv)
            faces.append(ft + v_offset)
            v_offset += len(fv)
        exp.Next()

    if not verts:
        raise RuntimeError("no triangulated faces")

    mesh = trimesh.Trimesh(
        vertices=np.vstack(verts), faces=np.vstack(faces), process=False
    )
    if len(mesh.faces) == 0:
        raise RuntimeError("empty mesh")
    pts, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float64)
    # Sanitize normals: degenerate/zero-area faces yield zero or non-finite
    # normals; replace those with a safe unit default so they never poison
    # downstream training (a single non-finite value NaNs the whole run).
    ln = np.linalg.norm(normals, axis=1)
    bad = ~np.isfinite(normals).all(axis=1) | (ln < 1e-9)
    if bad.any():
        normals[bad] = (0.0, 0.0, 1.0)
    data = np.concatenate([pts, normals], axis=1).astype(np.float32)
    # Hard guard: never emit a non-finite point cloud. Raising here means the
    # caller logs a failure (and skips) instead of writing garbage to disk.
    if not np.isfinite(data).all():
        raise RuntimeError("non-finite values in sampled point cloud")
    return data


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def worker_single(task: dict) -> list[dict]:
    """One single-file sample -> one .npy."""
    out_path = Path(task["out_path"])
    row = {"sample_id": task["sample_id"], "status": "", "name_mismatch": 0, "error": ""}
    try:
        shape = _read_step_oneshape(task["source_path"])
        if task["mode"] == "first":
            solids = _solids_of(shape)
            if not solids:
                raise RuntimeError("no solids in STEP")
            shape = solids[0]
        pc = shape_to_pointcloud(shape)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, pc)
        row["status"] = "ok"
    except Exception as e:
        row["status"] = "fail"
        row["error"] = f"{type(e).__name__}: {e}"
    return [row]


def worker_assembly(group: tuple[str, list[dict]]) -> list[dict]:
    """Decompose one assembly once; extract every requested solid index."""
    asm_path, tasks = group
    rows = []
    try:
        solids = load_assembly_with_names(Path(asm_path))
    except Exception as e:
        for t in tasks:
            rows.append({"sample_id": t["sample_id"], "status": "fail",
                         "name_mismatch": 0,
                         "error": f"decompose: {type(e).__name__}: {e}"})
        return rows

    for t in tasks:
        row = {"sample_id": t["sample_id"], "status": "", "name_mismatch": 0, "error": ""}
        idx = int(t["solid_idx"])
        if idx >= len(solids):
            row["status"] = "fail"
            row["error"] = f"solid_idx {idx} >= {len(solids)} solids"
            rows.append(row)
            continue
        name, solid = solids[idx]
        expected = t.get("expected_solid_name")
        if expected and safe_name(name) != expected:
            row["name_mismatch"] = 1
        try:
            pc = shape_to_pointcloud(solid)
            out_path = Path(t["out_path"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, pc)
            row["status"] = "ok"
        except Exception as e:
            row["status"] = "fail"
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Enumeration + source resolution (mirrors eval_rule_based.build_tasks_v6test)
# ---------------------------------------------------------------------------

def build_indexes():
    print("building source indexes ...", flush=True)
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
    print(f"  mcmaster={len(mc_idx)} synthno={len(synthno_idx)} "
          f"synthrm={len(synthrm_idx)} realcad_files={len(realcad_idx)} "
          f"grabcad_files={len(grabcad_idx)}", flush=True)
    return mc_idx, synthno_idx, synthrm_idx, realcad_idx, grabcad_idx


def resolve_assembly(body: str, fidx: dict[str, Path]):
    """body = '<file_base>__<solid_base>__<idx>'; find the LONGEST '__'-joined
    prefix present in the file index; the remainder is the solid name."""
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


def enumerate_v6_samples(splits) -> tuple[list[dict], Counter]:
    """Every bf_v2_features sample that passed the v6 max_faces=300 filter,
    with source resolution. Returns manifest rows."""
    mc_idx, synthno_idx, synthrm_idx, realcad_idx, grabcad_idx = build_indexes()

    rows: list[dict] = []
    stats: Counter = Counter()
    for split in splits:
        split_dir = V6_FEATURES / split
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name not in CLASSES13:
                continue
            cls = cls_dir.name
            for d in sorted(cls_dir.iterdir()):
                fg = d / "face_grids.npy"
                if not d.is_dir() or not fg.exists():
                    continue
                name = d.name
                # PADDLE STEAMER must never appear in any split
                if "PADDLE_STEAMER" in name or "PADDLE STEAMER" in name:
                    raise RuntimeError(
                        f"PADDLE STEAMER contamination in {split}/{cls}/{name}")
                try:
                    n_faces = int(np.load(fg, mmap_mode="r").shape[0])
                except Exception:
                    stats[f"{split}:bad_face_grids"] += 1
                    continue
                if n_faces > V6_MAX_FACES:
                    stats[f"{split}:over_cap"] += 1
                    continue

                row = {
                    "sample_id": f"{split}/{cls}/{name}",
                    "split": split, "class": cls, "source": "",
                    "source_path": "", "mode": "", "solid_idx": "",
                    "expected_solid_name": "", "n_faces": n_faces,
                    "resolution": "ok",
                }
                if name.startswith("mcmaster__"):
                    row["source"] = "mcmaster"
                    p = mc_idx.get(name[len("mcmaster__"):])
                    if p is None:
                        row["resolution"] = "unresolved"
                    else:
                        row.update(source_path=str(p), mode="whole")
                elif name.startswith("synthno__"):
                    row["source"] = "synthno"
                    p = synthno_idx.get((cls, name[len("synthno__"):]))
                    if p is None:
                        row["resolution"] = "unresolved"
                    else:
                        row.update(source_path=str(p), mode="whole")
                elif name.startswith("synthrm__"):
                    row["source"] = "synthrm"
                    p = synthrm_idx.get((cls, name[len("synthrm__"):]))
                    if p is None:
                        row["resolution"] = "unresolved"
                    else:
                        row.update(source_path=str(p), mode="whole")
                elif name.startswith("realcad__fusion__"):
                    row["source"] = "fusion"
                    # The solid-name part can end with '___<uuid>' (triple
                    # underscore), so split("__")[-1] leaves a leading '_'
                    # on the uuid (the unresolved-fusion bug in
                    # eval_rule_based.build_tasks_v6test). Match the
                    # trailing UUID pattern directly instead.
                    m = re.search(
                        r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                        r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$", name)
                    uuid_stem = m.group(1) if m else name.split("__")[-1]
                    assy_id = name.split("__")[2]
                    p = FUSION_RAW / assy_id / f"{uuid_stem}.step"
                    if not p.exists():
                        hits = list(FUSION_RAW.glob(f"*/{uuid_stem}.step"))
                        p = hits[0] if hits else None
                    if p is None:
                        row["resolution"] = "unresolved"
                    else:
                        row.update(source_path=str(p), mode="first")
                elif name.startswith(("realcad__", "grabcad__")):
                    src = "realcad" if name.startswith("realcad__") else "grabcad"
                    row["source"] = src
                    body = name[len(src) + 2:]
                    fidx = realcad_idx if src == "realcad" else grabcad_idx
                    p, solid_base, idx = resolve_assembly(body, fidx)
                    if idx is None:
                        row["resolution"] = "bad_name"
                    elif p is None:
                        row["resolution"] = "unresolved"
                    else:
                        row.update(source_path=str(p), mode="assembly",
                                   solid_idx=idx, expected_solid_name=solid_base)
                else:
                    row["source"] = "unknown"
                    row["resolution"] = "unknown_prefix"

                stats[f"{split}:{row['source']}:{row['resolution']}"] += 1
                rows.append(row)
    return rows, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_log(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done[r["sample_id"]] = r
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--dry-run", action="store_true",
                   help="enumerate + resolve + write manifest only")
    p.add_argument("--retry-failed", action="store_true",
                   help="retry samples recorded as failed in extraction_log.csv")
    p.add_argument("--limit", type=int, default=0,
                   help="cap number of extractions this run (smoke test)")
    args = p.parse_args()

    splits = [s for s in args.splits.split(",") if s in SPLITS]
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    rows, stats = enumerate_v6_samples(splits)

    manifest_path = RUN_DIR / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"manifest: {manifest_path} ({len(rows)} rows)", flush=True)

    print("--- enumeration stats ---", flush=True)
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}", flush=True)

    resolved = [r for r in rows if r["resolution"] == "ok"]
    by_split = Counter(r["split"] for r in resolved)
    print(f"resolved: {len(resolved)} / {len(rows)}  "
          f"({', '.join(f'{s}={by_split.get(s, 0)}' for s in splits)})", flush=True)

    if args.dry_run:
        return

    # ---- build work lists (resume-aware) ----
    log_path = RUN_DIR / "extraction_log.csv"
    done = load_log(log_path)
    if args.retry_failed:
        done = {k: v for k, v in done.items() if v["status"] == "ok"}

    singles: list[dict] = []
    asm_groups: dict[str, list[dict]] = defaultdict(list)
    n_exist = 0
    for r in resolved:
        out_path = OUT_FEATURES / r["split"] / r["class"] / f"{r['sample_id'].split('/')[-1]}.npy"
        if out_path.exists():
            n_exist += 1
            continue
        if r["sample_id"] in done and done[r["sample_id"]]["status"] == "fail":
            continue  # known-bad, don't retry
        task = {"sample_id": r["sample_id"], "source_path": r["source_path"],
                "mode": r["mode"], "out_path": str(out_path),
                "solid_idx": r["solid_idx"],
                "expected_solid_name": r["expected_solid_name"]}
        if r["mode"] == "assembly":
            asm_groups[r["source_path"]].append(task)
        else:
            singles.append(task)

    if args.limit:
        singles = singles[: args.limit]
        capped: dict[str, list[dict]] = {}
        n = 0
        for k, v in asm_groups.items():
            if n >= args.limit:
                break
            capped[k] = v[: args.limit - n]
            n += len(capped[k])
        asm_groups = capped

    n_todo = len(singles) + sum(len(v) for v in asm_groups.values())
    n_known_fail = sum(1 for v in done.values() if v["status"] == "fail")
    print(f"{n_exist} outputs already exist; {n_known_fail} known failures "
          f"skipped; {n_todo} to extract "
          f"({len(singles)} singles, {sum(len(v) for v in asm_groups.values())} "
          f"assembly solids in {len(asm_groups)} assemblies)", flush=True)
    if n_todo == 0:
        write_retention(rows)
        return

    # ---- extraction ----
    new_log = not log_path.exists()
    logf = log_path.open("a", newline="", encoding="utf-8")
    logw = csv.DictWriter(logf, fieldnames=LOG_FIELDS, extrasaction="ignore")
    if new_log:
        logw.writeheader()
        logf.flush()

    n_done = n_ok = n_fail = 0
    t0 = time.time()

    def progress():
        rate = n_done / max(time.time() - t0, 1e-6)
        eta_min = (n_todo - n_done) / max(rate, 1e-6) / 60
        print(f"  {n_done}/{n_todo}  ok={n_ok} fail={n_fail}  "
              f"rate={rate:.2f}/s  eta={eta_min:.1f}min", flush=True)

    groups = sorted(asm_groups.items())
    try:
        with multiprocessing.Pool(processes=args.workers) as pool:
            if singles:
                for rows_out in pool.imap_unordered(worker_single, singles, chunksize=8):
                    for row in rows_out:
                        logw.writerow(row)
                        n_done += 1
                        n_ok += row["status"] == "ok"
                        n_fail += row["status"] == "fail"
                    logf.flush()
                    if n_done % 200 < len(rows_out):
                        progress()
            if groups:
                for rows_out in pool.imap_unordered(worker_assembly, groups, chunksize=1):
                    for row in rows_out:
                        logw.writerow(row)
                        n_done += 1
                        n_ok += row["status"] == "ok"
                        n_fail += row["status"] == "fail"
                    logf.flush()
                    progress()
    finally:
        logf.close()

    print(f"extraction done in {(time.time() - t0) / 60:.1f} min  "
          f"ok={n_ok} fail={n_fail}", flush=True)
    write_retention(rows)


def write_retention(manifest_rows: list[dict]):
    """Per-split per-source: v6 BF samples vs PC outputs on disk."""
    ret: dict = {}
    for r in manifest_rows:
        split, src = r["split"], r["source"]
        cell = ret.setdefault(split, {}).setdefault(src, {"bf_v6": 0, "pc": 0})
        cell["bf_v6"] += 1
        out_path = OUT_FEATURES / split / r["class"] / f"{r['sample_id'].split('/')[-1]}.npy"
        if r["resolution"] == "ok" and out_path.exists():
            cell["pc"] += 1
    for split, by_src in ret.items():
        tot = {"bf_v6": sum(c["bf_v6"] for c in by_src.values()),
               "pc": sum(c["pc"] for c in by_src.values())}
        by_src["TOTAL"] = tot
    out = RUN_DIR / "retention.json"
    out.write_text(json.dumps(ret, indent=2), encoding="utf-8")
    print(f"retention: {out}", flush=True)
    for split in SPLITS:
        if split in ret:
            t = ret[split]["TOTAL"]
            pct = 100.0 * t["pc"] / max(t["bf_v6"], 1)
            print(f"  {split}: PC {t['pc']} / BF {t['bf_v6']} ({pct:.1f}%)", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
