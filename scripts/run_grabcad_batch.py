"""Run one GrabCAD-style dump batch end to end.

This orchestrates the compliant part of the workflow:

  1. optionally downloads direct/permitted URLs into D:\\step-vr-step-thesis\\grabcad_dump
  2. runs backend\\scripts\\process_dump_folder.py
  3. counts staged fastener feature directories
  4. writes training_data\\grabcad_run\\batch_summary.json

It deliberately does not crawl GrabCAD Library pages or automate their search
and download flow. Feed it direct file URLs you are allowed to automate, or run
it with --skip-download to process whatever is already in the dump folder.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"
PYTHON = Path(r"C:\Users\ferri\anaconda3\envs\stepvrstep\python.exe")

DEFAULT_DUMP = Path(r"D:\step-vr-step-thesis\grabcad_dump")
DEFAULT_KEPT = Path(r"D:\step-vr-step-thesis\grabcad_kept")
DEFAULT_TRASH = Path(r"D:\step-vr-step-thesis\grabcad_trash")
DEFAULT_RUN_DIR = REPO / "training_data" / "grabcad_run"
DEFAULT_STAGING = REPO / "training_data" / "grabcad_features_staged"
DEFAULT_URL_MANIFEST = DEFAULT_RUN_DIR / "direct_urls.txt"
DEFAULT_TARGET_QUERIES = DEFAULT_RUN_DIR / "target_queries.txt"


TARGET_QUERY_TEXT = """# Priority search list for finding direct/permitted CAD archive URLs.
# This file is documentation/input for a compliant source of URLs; this script
# does not crawl GrabCAD Library pages.

[highest_priority_rivet_dense]
aircraft fuselage
aircraft wing assembly
riveted sheet metal
aluminum boat hull
truss riveted
bus body frame
trailer body
riveted bracket

[high_priority_nut_dense]
engine assembly
gearbox assembly
bench vise
pipe flange assembly
suspension bushing
trailer hitch assembly
clamp assembly
skateboard truck

[high_priority_pin_dense]
hinge assembly
knuckle joint
clevis pin
door latch mechanism
excavator arm joint
pin and yoke
chain link assembly
cotter pin assembly

[medium_retaining_rings]
bearing housing assembly
shaft assembly with bearings
drill chuck assembly
gearbox internals

[medium_threaded_rods_studs]
turnbuckle assembly
tie rod adjustable
engine head stud
clevis turnbuckle
plumbing manifold

[medium_anchors]
anchor bolt
eye bolt assembly
lifting eye
expansion anchor
bridge anchor

[skip]
plain screw
plain bolt
tutorial first design
3d printed art
decorative model
"""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_seed_files(run_dir: Path, url_manifest: Path, target_queries: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if not url_manifest.exists():
        url_manifest.write_text(
            "# One direct, permitted CAD/archive URL per line.\n"
            "# Examples:\n"
            "# https://example.edu/open-cad/gearbox_assembly.zip\n"
            "# https://manufacturer.example/downloads/bracket_with_fasteners.step\n",
            encoding="utf-8",
        )
    if not target_queries.exists():
        target_queries.write_text(TARGET_QUERY_TEXT, encoding="utf-8")


def count_dump_files(dump: Path) -> int:
    if not dump.exists():
        return 0
    return sum(1 for p in dump.rglob("*") if p.is_file())


def count_staged(staging: Path) -> dict:
    per_class: dict[str, int] = {}
    total = 0
    if not staging.exists():
        return {"total": 0, "per_class": per_class}
    for fg in staging.rglob("face_grids.npy"):
        model_dir = fg.parent
        if not (model_dir / "edge_curves.npy").exists():
            continue
        if not (model_dir / "topo_distances.npz").exists():
            continue
        cls = model_dir.parent.name
        per_class[cls] = per_class.get(cls, 0) + 1
        total += 1
    return {"total": total, "per_class": dict(sorted(per_class.items()))}


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    print("[run] " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env)
    return proc.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-manifest", type=Path, default=DEFAULT_URL_MANIFEST)
    parser.add_argument("--url", action="append", default=[], help="Direct/permitted URL. May be repeated.")
    parser.add_argument("--skip-download", action="store_true", help="Only run the sorter on files already in the dump folder.")
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--kept", type=Path, default=DEFAULT_KEPT)
    parser.add_argument("--trash", type=Path, default=DEFAULT_TRASH)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--target-queries", type=Path, default=DEFAULT_TARGET_QUERIES)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target-count", type=int, default=1000)
    parser.add_argument("--download-limit", type=int)
    parser.add_argument("--download-max-mb", type=int, default=1024)
    parser.add_argument("--download-sleep", type=float, default=1.0)
    parser.add_argument("--delete-empty", action="store_true")
    parser.add_argument("--fail-below-target", action="store_true")
    parser.add_argument(
        "--i-have-permission-for-restricted-sites",
        action="store_true",
        help="Pass through to downloader for direct URLs from restricted domains; use only with permission.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    args.url_manifest = args.url_manifest.resolve()
    args.target_queries = args.target_queries.resolve()
    args.dump = args.dump.resolve()
    args.kept = args.kept.resolve()
    args.trash = args.trash.resolve()
    args.staging = args.staging.resolve()

    ensure_seed_files(args.run_dir, args.url_manifest, args.target_queries)
    for folder in (args.dump, args.kept, args.trash, args.staging):
        folder.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(BACKEND)

    before = count_staged(args.staging)
    dump_before = count_dump_files(args.dump)
    print(f"[status] staged before: {before['total']} fastener solids")
    print(f"[status] dump files before: {dump_before}")
    print(f"[queries] {args.target_queries}")
    print(f"[urls]    {args.url_manifest}")

    download_exit = 0
    if not args.skip_download:
        has_manifest_urls = any(
            line.strip() and not line.lstrip().startswith("#")
            for line in args.url_manifest.read_text(encoding="utf-8").splitlines()
        )
        if has_manifest_urls or args.url:
            download_cmd = [
                str(args.python),
                str(BACKEND / "scripts" / "download_to_grabcad_dump.py"),
                "--dump",
                str(args.dump),
                "--run-dir",
                str(args.run_dir),
                "--max-mb",
                str(args.download_max_mb),
                "--sleep",
                str(args.download_sleep),
            ]
            if has_manifest_urls:
                download_cmd.extend(["--url-manifest", str(args.url_manifest)])
            for url in args.url:
                download_cmd.extend(["--url", url])
            if args.download_limit:
                download_cmd.extend(["--limit", str(args.download_limit)])
            if args.i_have_permission_for_restricted_sites:
                download_cmd.append("--i-have-permission-for-restricted-sites")
            download_exit = run_command(download_cmd, REPO, env)
        else:
            print("[download] no URLs supplied; skipping download step")

    if download_exit != 0:
        print(f"[error] downloader exited {download_exit}; sorter not started")
        return download_exit

    dump_after_download = count_dump_files(args.dump)
    if dump_after_download == 0:
        print("[sorter] dump folder is empty; nothing to process")
    else:
        sorter_cmd = [
            str(args.python),
            str(BACKEND / "scripts" / "process_dump_folder.py"),
            "--dump",
            str(args.dump),
            "--kept",
            str(args.kept),
            "--trash",
            str(args.trash),
            "--staging",
            str(args.staging),
            "--run-dir",
            str(args.run_dir),
            "--workers",
            str(args.workers),
        ]
        if args.delete_empty:
            sorter_cmd.append("--delete-empty")
        sorter_exit = run_command(sorter_cmd, REPO, env)
        if sorter_exit != 0:
            print(f"[error] sorter exited {sorter_exit}")
            return sorter_exit

    after = count_staged(args.staging)
    added = after["total"] - before["total"]
    summary = {
        "updated": now_iso(),
        "staging": str(args.staging),
        "target_count": args.target_count,
        "staged_before": before,
        "staged_after": after,
        "added_this_batch": added,
        "dump_files_before": dump_before,
        "dump_files_after": count_dump_files(args.dump),
        "kept_files": count_dump_files(args.kept),
        "trash_files": count_dump_files(args.trash),
        "url_manifest": str(args.url_manifest),
        "target_queries": str(args.target_queries),
    }
    summary_path = args.run_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[summary] staged now: {after['total']} / {args.target_count}")
    print(f"[summary] added this batch: {added}")
    for cls, count in after["per_class"].items():
        print(f"[summary] {cls}: {count}")
    print(f"[summary] {summary_path}")

    if args.fail_below_target and after["total"] < args.target_count:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
