"""Harvest real-world CAD assembly archives into BRepFormer-ready STEP splits.

This script is intentionally source-agnostic. It can:

  * download direct archive/CAD URLs from a user-provided manifest,
  * ingest already-downloaded ZIP archives from a folder,
  * safely extract STEP/IGES/Parasolid-like B-Rep files while skipping meshes/docs,
  * auto-label strongly named fasteners,
  * decompose large/named STEP assemblies into placed solids and stage named parts,
  * optionally run the existing STEP -> BRepFormer feature converter.

It does not crawl CAD-library sites. Some sites, including GrabCAD, restrict
automated collection/downloads in their terms. Use direct URLs only when you
have permission to automate access.

Example:
    python backend/scripts/harvest_real_cad_assemblies.py \
        --url-manifest training_data/real_cad_sources.txt \
        --archives-dir D:/cad_zips \
        --work-dir training_data/real_cad_pool \
        --stage-root training_data/real_cad_binary \
        --run-brep --brep-root training_data/real_cad_brep
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"
DEFAULT_WORK_DIR = REPO / "training_data" / "real_cad_pool"
DEFAULT_STAGE_ROOT = REPO / "training_data" / "real_cad_binary"
DEFAULT_BREP_ROOT = REPO / "training_data" / "real_cad_brep"

BREP_EXTENSIONS = {".step", ".stp", ".iges", ".igs", ".x_t", ".x_b"}
STEP_EXTENSIONS = {".step", ".stp"}
ARCHIVE_EXTENSIONS = {".zip"}
NOISE_EXTENSIONS = {
    ".stl", ".obj", ".ply", ".3mf",
    ".sldprt", ".sldasm", ".ipt", ".iam", ".catpart", ".catproduct",
    ".dwg", ".dxf", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp",
    ".txt", ".md", ".doc", ".docx", ".xls", ".xlsx",
}
RESTRICTED_DOMAINS = {"grabcad.com"}

SPLIT_SEED = "real-cad-assemblies-2026-05-22"
VAL_FRAC = 0.10
TEST_FRAC = 0.10


@dataclass(frozen=True)
class NameLabel:
    label: str
    reason: str
    subtype: str = ""


@dataclass
class HarvestStats:
    downloaded: int = 0
    download_skipped: int = 0
    archives_seen: int = 0
    members_seen: int = 0
    brep_members: int = 0
    staged_fasteners: int = 0
    staged_non_fasteners: int = 0
    assemblies_decomposed: int = 0
    decomposed_parts: int = 0
    errors: int = 0


FASTENER_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\b(?:iso|din)\s*[-_ ]?\s*(?:4762|912|7984)\b", re.I), "socket_screw", "standard"),
    (re.compile(r"\b(?:iso|din)\s*[-_ ]?\s*(?:4014|4017|931|933|6921)\b", re.I), "hex_bolt", "standard"),
    (re.compile(r"\b(?:iso|din)\s*[-_ ]?\s*(?:10642|7991|1458[0-4]|7046|7047)\b", re.I), "countersunk_screw", "standard"),
    (re.compile(r"\b(?:iso|din)\s*[-_ ]?\s*(?:7380|7381|7985|7045)\b", re.I), "cap_screw", "standard"),
    (re.compile(r"\b(?:iso|din)\s*[-_ ]?\s*(?:4026|4027|4028|4029|913|914|915|916)\b", re.I), "set_screw", "standard"),
    (re.compile(r"\b(?:iso|din)\s*[-_ ]?\s*(?:4032|4033|4034|4035|4036|934|935|936|985|980|982)\b", re.I), "hex_nut", "standard"),
    (re.compile(r"\b(?:iso|din)\s*[-_ ]?\s*(?:7089|7090|7091|7092|7093|7094|125|126|127|128)\b", re.I), "washer", "standard"),
    (re.compile(r"\b(?:bolt|hex[-_ ]?bolt|carriage[-_ ]?bolt|lag[-_ ]?bolt)\b", re.I), "hex_bolt", "keyword"),
    (re.compile(r"\b(?:socket[-_ ]?head|socket[-_ ]?cap|cap[-_ ]?screw|allen[-_ ]?screw|shcs)\b", re.I), "socket_screw", "keyword"),
    (re.compile(r"\b(?:countersunk|flat[-_ ]?head|csk)\b", re.I), "countersunk_screw", "keyword"),
    (re.compile(r"\b(?:set[-_ ]?screw|grub[-_ ]?screw)\b", re.I), "set_screw", "keyword"),
    (re.compile(r"\b(?:button[-_ ]?head|pan[-_ ]?head|machine[-_ ]?screw|screw)\b", re.I), "cap_screw", "keyword"),
    (re.compile(r"\b(?:hex[-_ ]?nut|lock[-_ ]?nut|locknut|nylock|wing[-_ ]?nut|nut)\b", re.I), "hex_nut", "keyword"),
    (re.compile(r"\b(?:flat[-_ ]?washer|plain[-_ ]?washer|spring[-_ ]?washer|lock[-_ ]?washer|washer)\b", re.I), "washer", "keyword"),
    (re.compile(r"\b(?:rivet|blind[-_ ]?rivet|solid[-_ ]?rivet)\b", re.I), "rivet", "keyword"),
    (re.compile(r"\b(?:dowel[-_ ]?pin|taper[-_ ]?pin|roll[-_ ]?pin|split[-_ ]?pin|cotter[-_ ]?pin)\b", re.I), "pin", "keyword"),
    (re.compile(r"\b(?:stud|threaded[-_ ]?rod)\b", re.I), "stud", "keyword"),
    (re.compile(r"\b(?:retaining[-_ ]?ring|snap[-_ ]?ring|circlip|e[-_ ]?ring)\b", re.I), "snap_ring", "keyword"),
    (re.compile(r"\banchor\b", re.I), "anchor", "keyword"),
]
METRIC_FASTENER_RE = re.compile(r"\bm\s*\d+(?:[.,_]\d+)?\s*(?:x|X|by)\s*\d+(?:[.,_]\d+)?\b")

NON_FASTENER_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:bracket|mounting[-_ ]?plate|plate|panel|cover|lid)\b", re.I), "structural"),
    (re.compile(r"\b(?:housing|case|body|base|frame|chassis|support|stand)\b", re.I), "structural"),
    (re.compile(r"\b(?:gear|wheel|pulley|sprocket|shaft|axle|lever|arm|linkage)\b", re.I), "mechanism"),
    (re.compile(r"\b(?:pipe|tube|manifold|flange|fitting|valve|nozzle)\b", re.I), "fluid"),
    (re.compile(r"\b(?:bearing|bushing|spacer|standoff|shim)\b", re.I), "hardware_nonfastener"),
    (re.compile(r"\b(?:pcb|pcba|circuit[-_ ]?board)\b", re.I), "electronics"),
]

ASSEMBLY_HINT_RE = re.compile(r"\b(?:assembly|assemblage|assy|asm|subassembly|sub[-_ ]?asm)\b", re.I)
FASTENER_OVERRIDE_RE = re.compile(r"\b(?:screwdriver|nutcracker|pinion|piniongear|washerfluid)\b", re.I)


def normalize_name(text: str) -> str:
    text = text.replace("\\", "/")
    stem = PurePosixPath(text).stem
    stem = stem.replace("#", " ")
    return re.sub(r"[_\-+.]+", " ", stem)


def classify_name(name: str) -> NameLabel:
    normalized = normalize_name(name)
    if not normalized or normalized.lower().startswith("unnamed solid"):
        return NameLabel("unknown", "unnamed")

    if FASTENER_OVERRIDE_RE.search(normalized):
        return NameLabel("unknown", "fastener_override")

    for pattern, subtype, reason in FASTENER_RULES:
        if pattern.search(normalized):
            return NameLabel("fastener", reason, subtype)

    if METRIC_FASTENER_RE.search(normalized):
        return NameLabel("fastener", "metric_size", "metric_fastener")

    if ASSEMBLY_HINT_RE.search(normalized):
        return NameLabel("assembly", "assembly_hint")

    for pattern, reason in NON_FASTENER_RULES:
        if pattern.search(normalized):
            return NameLabel("non_fastener", reason)

    return NameLabel("unknown", "no_rule")


def split_for(key: str) -> str:
    h = int(hashlib.md5(f"{SPLIT_SEED}:{key}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < TEST_FRAC:
        return "test"
    if h < TEST_FRAC + VAL_FRAC:
        return "val"
    return "train"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def slugify(text: str, max_len: int = 90) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalize_name(text)).strip("._-")
    return (slug or "cad_part")[:max_len]


def safe_archive_rel(member_name: str) -> PurePosixPath | None:
    raw = member_name.replace("\\", "/")
    pure = PurePosixPath(raw)
    if pure.is_absolute():
        return None
    if any(part in {"", ".", ".."} for part in pure.parts):
        return None
    if pure.parts and re.match(r"^[A-Za-z]:$", pure.parts[0]):
        return None
    return pure


def ensure_within(child: Path, parent: Path) -> None:
    child_resolved = child.resolve()
    parent_resolved = parent.resolve()
    try:
        child_resolved.relative_to(parent_resolved)
    except ValueError as exc:
        raise ValueError(f"unsafe path outside {parent_resolved}: {child_resolved}") from exc


def read_url_manifest(path: Path) -> list[str]:
    if not path or not path.exists():
        return []
    urls: list[str] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "url" in {h.lower() for h in reader.fieldnames}:
                url_key = next(h for h in reader.fieldnames if h.lower() == "url")
                for row in reader:
                    url = (row.get(url_key) or "").strip()
                    if url and not url.startswith("#"):
                        urls.append(url)
                return urls
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line.split(",", 1)[0].strip())
    return urls


def is_restricted_url(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    host = host.lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in RESTRICTED_DOMAINS)


def filename_from_response(url: str, response: urllib.response.addinfourl) -> str:
    disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', disposition, re.I)
    if match:
        return urllib.parse.unquote(match.group(1)).strip()
    path_name = Path(urllib.parse.urlparse(response.geturl()).path).name
    if path_name:
        return urllib.parse.unquote(path_name)
    return f"download_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}.zip"


def download_url(url: str, out_dir: Path, args: argparse.Namespace, rows: list[dict], stats: HarvestStats) -> Path | None:
    if is_restricted_url(url) and not args.i_have_permission_for_restricted_sites:
        stats.download_skipped += 1
        rows.append({
            "kind": "download",
            "source_url": url,
            "status": "skipped_restricted_site",
            "notes": "pass --i-have-permission-for-restricted-sites only if you have written permission",
        })
        print(f"[skip] restricted site URL needs explicit permission flag: {url}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "step-vr-step-real-cad-harvester/1.0",
            "Accept": "*/*",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=args.download_timeout) as response:
            name = filename_from_response(url, response)
            suffix = Path(name).suffix.lower()
            if suffix not in ARCHIVE_EXTENSIONS | BREP_EXTENSIONS:
                name = f"{slugify(name)}.zip"
            dst = out_dir / name
            if dst.exists() and not args.force_download:
                stats.download_skipped += 1
                rows.append({
                    "kind": "download",
                    "source_url": url,
                    "status": "exists",
                    "archive_path": str(dst),
                    "notes": "",
                })
                return dst

            tmp = dst.with_suffix(dst.suffix + ".part")
            max_bytes = args.max_archive_mb * 1024 * 1024
            written = 0
            with tmp.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        tmp.unlink(missing_ok=True)
                        raise ValueError(f"download exceeded --max-archive-mb={args.max_archive_mb}")
                    f.write(chunk)
            tmp.replace(dst)
            stats.downloaded += 1
            rows.append({
                "kind": "download",
                "source_url": url,
                "status": "ok",
                "archive_path": str(dst),
                "bytes": written,
                "notes": "",
            })
            if args.sleep_between_downloads > 0:
                time.sleep(args.sleep_between_downloads)
            return dst
    except (urllib.error.URLError, OSError, ValueError) as exc:
        stats.errors += 1
        rows.append({
            "kind": "download",
            "source_url": url,
            "status": "error",
            "notes": f"{type(exc).__name__}: {exc}",
        })
        print(f"[error] download failed: {url} ({exc})")
        return None


def collect_input_files(args: argparse.Namespace, rows: list[dict], stats: HarvestStats) -> list[Path]:
    files: list[Path] = []
    if args.url_manifest:
        urls = read_url_manifest(args.url_manifest)
        if args.limit:
            urls = urls[: args.limit]
        for url in urls:
            downloaded = download_url(url, args.work_dir / "archives", args, rows, stats)
            if downloaded:
                files.append(downloaded)

    for archives_dir in args.archives_dir:
        if not archives_dir.exists():
            print(f"[warn] archives dir does not exist: {archives_dir}")
            continue
        for path in sorted(archives_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in ARCHIVE_EXTENSIONS | BREP_EXTENSIONS:
                files.append(path)

    deduped: dict[Path, Path] = {}
    for path in files:
        try:
            deduped[path.resolve()] = path
        except OSError:
            deduped[path] = path
    return list(deduped.values())


def stage_step_file(
    src: Path,
    label: NameLabel,
    key: str,
    original_name: str,
    source_url: str,
    archive_path: str,
    inner_path: str,
    rows: list[dict],
    stats: HarvestStats,
    args: argparse.Namespace,
    part_name: str = "",
) -> Path | None:
    if label.label == "non_fastener" and not args.stage_name_negatives:
        return None
    if label.label not in {"fastener", "non_fastener"}:
        return None

    split = split_for(key)
    label_dir = "non_fastener" if label.label == "non_fastener" else "fastener"
    unique = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    dst_name = f"{slugify(original_name)}__{unique}.step"
    dst = args.stage_root / split / label_dir / dst_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or args.force_stage:
        shutil.copy2(src, dst)

    if label.label == "fastener":
        stats.staged_fasteners += 1
    else:
        stats.staged_non_fasteners += 1

    rows.append({
        "kind": "staged_step",
        "source_url": source_url,
        "archive_path": archive_path,
        "inner_path": inner_path,
        "part_name": part_name,
        "classification": label.label,
        "subtype": label.subtype,
        "reason": label.reason,
        "split": split,
        "staged_path": str(dst),
        "status": "ok",
        "bytes": src.stat().st_size if src.exists() else "",
        "notes": "",
    })
    return dst


def should_decompose_step(path: Path, inner_name: str, label: NameLabel, args: argparse.Namespace) -> bool:
    if args.no_decompose_assemblies:
        return False
    if path.suffix.lower() not in STEP_EXTENSIONS:
        return False
    if label.label == "fastener":
        return False
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
    except OSError:
        return False
    if size_mb > args.max_assembly_mb:
        return False
    if args.decompose_all_steps:
        return True
    if label.label == "assembly":
        return True
    return size_mb >= args.decompose_min_mb


def count_faces_and_signature_helpers():
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer

    def count_faces(shape) -> int:
        n = 0
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            n += 1
            exp.Next()
        return n

    def signature(shape) -> tuple:
        box = Bnd_Box()
        try:
            brepbndlib.Add(shape, box)
            x1, y1, z1, x2, y2, z2 = box.Get()
            dims = tuple(sorted([round(x2 - x1, 1), round(y2 - y1, 1), round(z2 - z1, 1)]))
        except Exception:
            dims = (0.0, 0.0, 0.0)
        return (count_faces(shape),) + dims

    return signature


def load_step_solids_with_names(step_path: Path) -> list[tuple[str, object]]:
    """Return placed solid instances with best-effort names from a STEP assembly."""
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    signature = count_faces_and_signature_helpers()
    sig_to_name: dict[tuple, str] = {}

    try:
        from OCC.Extend.DataExchange import read_step_file_with_names_colors

        shape_name_map = read_step_file_with_names_colors(str(step_path))
        for shape, info in shape_name_map.items():
            name = info[0] if info else ""
            if not name:
                continue
            exp = TopExp_Explorer(shape, TopAbs_SOLID)
            while exp.More():
                solid = topods.Solid(exp.Current())
                sig_to_name.setdefault(signature(solid), name)
                exp.Next()
    except Exception:
        sig_to_name = {}

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != 1:
        raise ValueError("STEP read failed")
    reader.TransferRoots()
    whole = reader.OneShape()
    if whole.IsNull():
        raise ValueError("STEP loaded null shape")

    solids: list[object] = []
    exp = TopExp_Explorer(whole, TopAbs_SOLID)
    while exp.More():
        solids.append(topods.Solid(exp.Current()))
        exp.Next()

    name_counts: dict[str, int] = {}
    named: list[tuple[str, object]] = []
    for idx, solid in enumerate(solids):
        base_name = sig_to_name.get(signature(solid), f"unnamed_solid_{idx}")
        count = name_counts.get(base_name, 0)
        name_counts[base_name] = count + 1
        name = f"{base_name}#{count}" if count else base_name
        named.append((name, solid))
    return named


def write_shape_step(shape, out_path: Path, product_name: str) -> None:
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.product.name", product_name)
    status = writer.Transfer(shape, STEPControl_AsIs)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP transfer failed: status={status}")
    status = writer.Write(str(out_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: status={status}")


def decompose_and_stage(
    step_path: Path,
    source_key: str,
    source_url: str,
    archive_path: str,
    inner_path: str,
    rows: list[dict],
    stats: HarvestStats,
    args: argparse.Namespace,
) -> None:
    try:
        solids = load_step_solids_with_names(step_path)
        stats.assemblies_decomposed += 1
    except Exception as exc:
        stats.errors += 1
        rows.append({
            "kind": "decompose",
            "source_url": source_url,
            "archive_path": archive_path,
            "inner_path": inner_path,
            "status": "error",
            "notes": f"{type(exc).__name__}: {exc}",
        })
        print(f"[error] assembly decomposition failed: {step_path} ({exc})")
        return

    out_dir = args.work_dir / "decomposed" / hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:12]
    exported = 0
    for idx, (name, solid) in enumerate(solids):
        stats.decomposed_parts += 1
        label = classify_name(name)
        if label.label == "non_fastener" and not args.stage_name_negatives:
            continue
        if label.label not in {"fastener", "non_fastener"}:
            continue
        if exported >= args.max_exported_parts_per_assembly:
            rows.append({
                "kind": "decompose",
                "source_url": source_url,
                "archive_path": archive_path,
                "inner_path": inner_path,
                "status": "truncated",
                "notes": f"hit --max-exported-parts-per-assembly={args.max_exported_parts_per_assembly}",
            })
            break

        part_key = f"{source_key}::{idx}::{name}"
        part_file = out_dir / label.label / f"{idx:05d}_{slugify(name)}.step"
        try:
            if not part_file.exists() or args.force_stage:
                write_shape_step(solid, part_file, name)
            stage_step_file(
                part_file,
                label,
                part_key,
                name,
                source_url,
                archive_path,
                inner_path,
                rows,
                stats,
                args,
                part_name=name,
            )
            exported += 1
        except Exception as exc:
            stats.errors += 1
            rows.append({
                "kind": "decomposed_part",
                "source_url": source_url,
                "archive_path": archive_path,
                "inner_path": inner_path,
                "part_name": name,
                "classification": label.label,
                "subtype": label.subtype,
                "reason": label.reason,
                "status": "error",
                "notes": f"{type(exc).__name__}: {exc}",
            })

    rows.append({
        "kind": "decompose",
        "source_url": source_url,
        "archive_path": archive_path,
        "inner_path": inner_path,
        "status": "ok",
        "notes": f"solids={len(solids)} exported={exported}",
    })


def process_cad_file(
    cad_path: Path,
    source_key: str,
    source_url: str,
    archive_path: str,
    inner_path: str,
    rows: list[dict],
    stats: HarvestStats,
    args: argparse.Namespace,
) -> None:
    ext = cad_path.suffix.lower()
    label = classify_name(inner_path or cad_path.name)
    rows.append({
        "kind": "cad_file",
        "source_url": source_url,
        "archive_path": archive_path,
        "inner_path": inner_path,
        "classification": label.label,
        "subtype": label.subtype,
        "reason": label.reason,
        "status": "seen",
        "bytes": cad_path.stat().st_size if cad_path.exists() else "",
        "notes": "",
    })

    if ext in STEP_EXTENSIONS:
        stage_step_file(
            cad_path,
            label,
            source_key,
            inner_path or cad_path.name,
            source_url,
            archive_path,
            inner_path,
            rows,
            stats,
            args,
        )
        if should_decompose_step(cad_path, inner_path or cad_path.name, label, args):
            decompose_and_stage(cad_path, source_key, source_url, archive_path, inner_path, rows, stats, args)


def process_zip_archive(path: Path, rows: list[dict], stats: HarvestStats, args: argparse.Namespace, source_url: str = "") -> None:
    stats.archives_seen += 1
    archive_hash = file_sha256(path)
    archive_id = f"{slugify(path.stem)}__{archive_hash[:10]}"
    extract_root = args.work_dir / "extracted_brep" / archive_id
    max_member_bytes = args.max_member_mb * 1024 * 1024
    max_total_bytes = args.max_total_mb * 1024 * 1024
    total = 0

    try:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                stats.members_seen += 1
                rel = safe_archive_rel(info.filename)
                if rel is None:
                    stats.errors += 1
                    rows.append({
                        "kind": "archive_member",
                        "source_url": source_url,
                        "archive_path": str(path),
                        "archive_sha256": archive_hash,
                        "inner_path": info.filename,
                        "status": "skipped_unsafe_path",
                        "notes": "",
                    })
                    continue

                ext = Path(rel.name).suffix.lower()
                if ext in NOISE_EXTENSIONS or ext not in BREP_EXTENSIONS:
                    continue
                if info.file_size > max_member_bytes:
                    rows.append({
                        "kind": "archive_member",
                        "source_url": source_url,
                        "archive_path": str(path),
                        "archive_sha256": archive_hash,
                        "inner_path": str(rel),
                        "status": "skipped_member_too_large",
                        "bytes": info.file_size,
                        "notes": "",
                    })
                    continue
                total += info.file_size
                if total > max_total_bytes:
                    rows.append({
                        "kind": "archive_member",
                        "source_url": source_url,
                        "archive_path": str(path),
                        "archive_sha256": archive_hash,
                        "inner_path": str(rel),
                        "status": "skipped_archive_total_limit",
                        "bytes": info.file_size,
                        "notes": "",
                    })
                    continue

                out_path = extract_root / Path(*rel.parts)
                ensure_within(out_path, extract_root)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if not out_path.exists() or args.force_extract:
                    with zf.open(info) as src, out_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                stats.brep_members += 1
                source_key = f"{archive_hash}:{rel}"
                process_cad_file(out_path, source_key, source_url, str(path), str(rel), rows, stats, args)
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        stats.errors += 1
        rows.append({
            "kind": "archive",
            "source_url": source_url,
            "archive_path": str(path),
            "status": "error",
            "notes": f"{type(exc).__name__}: {exc}",
        })
        print(f"[error] archive failed: {path} ({exc})")


def process_direct_cad(path: Path, rows: list[dict], stats: HarvestStats, args: argparse.Namespace, source_url: str = "") -> None:
    archive_hash = file_sha256(path)
    stats.brep_members += 1
    process_cad_file(path, archive_hash, source_url, "", path.name, rows, stats, args)


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind", "source_url", "archive_path", "archive_sha256", "inner_path", "part_name",
        "classification", "subtype", "reason", "split", "staged_path", "status",
        "bytes", "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_run_brep_conversion(args: argparse.Namespace) -> int:
    if not args.run_brep:
        return 0
    cmd = [
        sys.executable,
        str(BACKEND / "scripts" / "step_to_brep.py"),
        "--src",
        str(args.stage_root),
        "--dst",
        str(args.brep_root),
        "--workers",
        str(args.workers),
    ]
    print("[brep] " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(BACKEND), check=False).returncode


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-manifest", type=Path, help="Text or CSV file containing one direct URL per row. CSV may have a 'url' column.")
    parser.add_argument("--archives-dir", action="append", default=[], type=Path, help="Folder of already available ZIP/direct CAD files. May be repeated.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT, help="BRepFormer-ready STEP layout: root/{train,val,test}/{fastener,non_fastener}.")
    parser.add_argument("--manifest-out", type=Path, help="CSV audit manifest. Defaults to <work-dir>/manifest.csv.")
    parser.add_argument("--brep-root", type=Path, default=DEFAULT_BREP_ROOT)
    parser.add_argument("--run-brep", action="store_true", help="After staging STEP files, run backend/scripts/step_to_brep.py.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--limit", type=int, help="Limit URL manifest entries.")
    parser.add_argument("--stage-name-negatives", action="store_true", help="Also stage strongly named non-fastener STEP parts as non_fastener.")
    parser.add_argument("--no-decompose-assemblies", action="store_true", help="Do not split large/named STEP assemblies into solids.")
    parser.add_argument("--decompose-all-steps", action="store_true", help="Try decomposing every non-fastener STEP file, not just assembly-like or large files.")
    parser.add_argument("--decompose-min-mb", type=float, default=2.0, help="Decompose non-fastener/unknown STEP files at least this large.")
    parser.add_argument("--max-assembly-mb", type=float, default=250.0)
    parser.add_argument("--max-exported-parts-per-assembly", type=int, default=5000)
    parser.add_argument("--max-archive-mb", type=int, default=1024)
    parser.add_argument("--max-member-mb", type=int, default=512)
    parser.add_argument("--max-total-mb", type=int, default=4096)
    parser.add_argument("--download-timeout", type=int, default=120)
    parser.add_argument("--sleep-between-downloads", type=float, default=2.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-stage", action="store_true")
    parser.add_argument("--fail-on-errors", action="store_true", help="Exit nonzero if any downloads, archives, or part exports failed.")
    parser.add_argument(
        "--i-have-permission-for-restricted-sites",
        action="store_true",
        help="Allow direct URLs from sites with known automated-collection restrictions. Do not use unless you have permission.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.url_manifest and not args.archives_dir:
        raise SystemExit("Provide --url-manifest, --archives-dir, or both.")

    args.work_dir = args.work_dir.resolve()
    args.stage_root = args.stage_root.resolve()
    args.brep_root = args.brep_root.resolve()
    args.manifest_out = (args.manifest_out or (args.work_dir / "manifest.csv")).resolve()
    args.archives_dir = [p.resolve() for p in args.archives_dir]

    rows: list[dict] = []
    stats = HarvestStats()
    input_files = collect_input_files(args, rows, stats)
    print(f"[input] {len(input_files)} archive/CAD files to process")

    for path in input_files:
        suffix = path.suffix.lower()
        if suffix in ARCHIVE_EXTENSIONS:
            print(f"[zip] {path}")
            process_zip_archive(path, rows, stats, args)
        elif suffix in BREP_EXTENSIONS:
            print(f"[cad] {path}")
            process_direct_cad(path, rows, stats, args)

    write_manifest(rows, args.manifest_out)
    print(f"[manifest] {args.manifest_out}")
    print(
        "[summary] "
        f"downloaded={stats.downloaded} "
        f"download_skipped={stats.download_skipped} "
        f"archives={stats.archives_seen} "
        f"members={stats.members_seen} "
        f"brep_members={stats.brep_members} "
        f"assemblies_decomposed={stats.assemblies_decomposed} "
        f"decomposed_parts={stats.decomposed_parts} "
        f"staged_fasteners={stats.staged_fasteners} "
        f"staged_non_fasteners={stats.staged_non_fasteners} "
        f"errors={stats.errors}"
    )
    if stats.staged_fasteners == 0:
        print("[warn] no fastener STEP samples were staged. Check naming rules, permissions, or source quality.")

    brep_exit = maybe_run_brep_conversion(args)
    return 1 if stats.errors and args.fail_on_errors else brep_exit


if __name__ == "__main__":
    raise SystemExit(main())
