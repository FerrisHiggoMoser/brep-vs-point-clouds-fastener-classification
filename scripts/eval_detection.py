"""Run fastener detection across a directory of STEP files and report
accuracy + performance metrics.

This is the practical "does the matcher actually work on my real CAD?"
test. It walks every .step/.stp under --data-dir, runs the full detection
pipeline on each, then writes a per-file CSV plus a console summary.

Usage:
    py -3.12 scripts/eval_detection.py
    py -3.12 scripts/eval_detection.py --data-dir ../fastener_labeling/files
    py -3.12 scripts/eval_detection.py --ml --detail --emit-usd
"""
from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

# Make `step_vr_step` importable regardless of CWD when invoked as
# `py scripts/eval_detection.py`.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# Some fixture filenames contain Unicode that Windows' default cp1252
# console encoding can't print. Force UTF-8 stdout so the script never
# fails on a `print(filename)` mid-run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

logger = logging.getLogger("eval_detection")


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    step_path: Path,
    *,
    use_ml: bool = False,
    threshold: float = 0.60,
    emit_usd: bool = False,
    usd_dir: Path | None = None,
) -> dict:
    """Run the pipeline on a single STEP file and return a row dict."""
    from step_vr_step.readers.step_reader import read_step
    from step_vr_step.detection import detect_fasteners
    from step_vr_step.config import DetectionConfig

    row: dict = {
        "file": step_path.name,
        "parts_total": 0, "parts_unique": 0,
        "fasteners_detected": 0,
        "fasteners_with_match": 0, "fasteners_without_match": 0,
        "screwedInto_arcs": 0, "contained_in_arcs": 0,
        "fit_taps": 0, "fit_slips": 0, "fit_clearances": 0,
        "hole_through": 0, "hole_blind": 0,
        "hole_counterbore": 0, "hole_countersink": 0,
        "time_read_s": 0.0, "time_detect_s": 0.0,
        "time_usd_s": 0.0, "time_total_s": 0.0,
        "parts_per_sec": 0.0, "match_rate": 0.0,
        "error": "",
        "unmatched_fasteners": [],
    }

    t_total = time.perf_counter()
    try:
        t = time.perf_counter()
        doc, manifest, shapes = read_step(str(step_path), return_shapes=True)
        row["time_read_s"] = round(time.perf_counter() - t, 3)

        # Unique geometries (after topology-hash dedup the pipeline does).
        hashes = {
            p.fingerprint.topology_hash for p in manifest.parts
            if p.fingerprint and p.fingerprint.topology_hash
            not in ("root", "empty", "")
        }
        row["parts_total"] = len(manifest.parts)
        row["parts_unique"] = len(hashes)

        t = time.perf_counter()
        manifest = detect_fasteners(
            manifest,
            shapes=shapes,
            config=DetectionConfig(
                enable_ml=use_ml,
                rule_confidence_threshold=threshold,
            ),
        )
        row["time_detect_s"] = round(time.perf_counter() - t, 3)

        # Tally per-part labels.
        fastener_uuids: set[str] = set()
        for p in manifest.parts:
            if p.detection and p.detection.fastener_type not in (
                "unclassified", "possible_unclassified",
            ):
                fastener_uuids.add(str(p.uuid))
        row["fasteners_detected"] = len(fastener_uuids)

        # Tally relationships.
        with_match: set[str] = set()
        for rel in manifest.relationships:
            if rel.kind == "fastener":
                row["screwedInto_arcs"] += 1
                with_match.add(str(rel.subject_uuid))
                fit = (rel.params or {}).get("fit_class")
                if fit == "tap":
                    row["fit_taps"] += 1
                elif fit == "slip":
                    row["fit_slips"] += 1
                elif fit == "clearance":
                    row["fit_clearances"] += 1
                hk = (rel.params or {}).get("hole_kind")
                if hk == "through":
                    row["hole_through"] += 1
                elif hk == "blind":
                    row["hole_blind"] += 1
                elif hk == "counterbore":
                    row["hole_counterbore"] += 1
                elif hk == "countersink":
                    row["hole_countersink"] += 1
            elif rel.kind == "contained_in":
                row["contained_in_arcs"] += 1

        row["fasteners_with_match"] = len(with_match)
        row["fasteners_without_match"] = (
            row["fasteners_detected"] - row["fasteners_with_match"]
        )
        if row["fasteners_detected"] > 0:
            row["match_rate"] = round(
                row["fasteners_with_match"] / row["fasteners_detected"], 3,
            )

        # Capture unmatched fasteners for the triage list.
        unmatched = fastener_uuids - with_match
        for p in manifest.parts:
            if str(p.uuid) in unmatched and p.detection:
                row["unmatched_fasteners"].append(
                    f"{p.name}({p.detection.fastener_type}/{p.detection.variant})"
                )

        if emit_usd:
            from step_vr_step.exporters import write_usd, USD_AVAILABLE
            if USD_AVAILABLE:
                out_path = (usd_dir or step_path.parent) / (step_path.stem + ".usdc")
                t = time.perf_counter()
                write_usd(manifest, out_path, glb_path=None)
                row["time_usd_s"] = round(time.perf_counter() - t, 3)
            else:
                logger.warning("usd-core not installed; skipping --emit-usd")

    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Failed on %s", step_path)

    row["time_total_s"] = round(time.perf_counter() - t_total, 3)
    if row["time_total_s"] > 0 and row["parts_total"] > 0:
        row["parts_per_sec"] = round(row["parts_total"] / row["time_total_s"], 1)
    return row


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize(rows: list[dict]) -> None:
    """Print a console summary across all files."""
    n = len(rows)
    failed = [r for r in rows if r["error"]]
    ok = [r for r in rows if not r["error"]]

    print()
    print("=" * 70)
    print(f"Evaluated {n} files  |  ok: {len(ok)}  |  failed: {len(failed)}")
    print("=" * 70)

    if not ok:
        print("(no successful runs)")
        if failed:
            for r in failed:
                print(f"  ! {r['file']}: {r['error']}")
        return

    total_fasteners = sum(r["fasteners_detected"] for r in ok)
    total_matched = sum(r["fasteners_with_match"] for r in ok)
    aggregate_match_rate = (
        total_matched / total_fasteners if total_fasteners else 0.0
    )
    total_arcs = sum(r["screwedInto_arcs"] for r in ok)
    total_contained = sum(r["contained_in_arcs"] for r in ok)

    print(f"Fasteners detected:    {total_fasteners}")
    print(f"Fasteners with match:  {total_matched}  ({aggregate_match_rate:.1%})")
    print(f"screwedInto arcs:      {total_arcs}  "
          f"(>{total_matched} means some bolts thread through multiple plates)")
    print(f"contained_in arcs:     {total_contained}")

    fit_total = sum(r["fit_taps"] + r["fit_slips"] + r["fit_clearances"] for r in ok)
    if fit_total:
        taps = sum(r["fit_taps"] for r in ok)
        slips = sum(r["fit_slips"] for r in ok)
        clears = sum(r["fit_clearances"] for r in ok)
        print(f"Fit-class breakdown:   "
              f"tap {taps} ({taps/fit_total:.0%})  "
              f"slip {slips} ({slips/fit_total:.0%})  "
              f"clearance {clears} ({clears/fit_total:.0%})")

    hk_total = sum(
        r["hole_through"] + r["hole_blind"]
        + r["hole_counterbore"] + r["hole_countersink"] for r in ok
    )
    if hk_total:
        thr = sum(r["hole_through"] for r in ok)
        bli = sum(r["hole_blind"] for r in ok)
        cb = sum(r["hole_counterbore"] for r in ok)
        cs = sum(r["hole_countersink"] for r in ok)
        print(f"Hole-kind breakdown:   "
              f"through {thr}  blind {bli}  counterbore {cb}  countersink {cs}")

    times = sorted(r["time_total_s"] for r in ok)
    pps = [r["parts_per_sec"] for r in ok if r["parts_per_sec"] > 0]
    print(f"Time per file:         "
          f"median {statistics.median(times):.2f}s  "
          f"max {max(times):.2f}s")
    if pps:
        print(f"Parts/sec throughput:  median {statistics.median(pps):.0f}  "
              f"min {min(pps):.0f}")

    # Bottom 3 by match rate.
    by_match = sorted(
        [r for r in ok if r["fasteners_detected"] > 0],
        key=lambda r: r["match_rate"],
    )[:3]
    if by_match:
        print()
        print("Lowest match-rate files (triage these first):")
        for r in by_match:
            print(f"  {r['match_rate']:>5.0%}  {r['file']}  "
                  f"({r['fasteners_with_match']}/{r['fasteners_detected']} matched)")

    # Slowest 3.
    by_time = sorted(ok, key=lambda r: -r["time_total_s"])[:3]
    print()
    print("Slowest files:")
    for r in by_time:
        print(f"  {r['time_total_s']:>6.2f}s  "
              f"{r['parts_per_sec']:>6.0f} parts/s  {r['file']}")

    if failed:
        print()
        print("Failed files:")
        for r in failed:
            print(f"  ! {r['file']}: {r['error']}")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = [k for k in rows[0].keys() if k != "unmatched_fasteners"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[2] / "fastener_labeling" / "files"),
        help="Directory containing .step / .stp files to evaluate.",
    )
    p.add_argument(
        "--out-csv", default="eval_results.csv",
        help="Per-file metrics CSV output path.",
    )
    p.add_argument("--ml", action="store_true", help="Enable ML detection (needs PyTorch).")
    p.add_argument(
        "--threshold", type=float, default=0.60,
        help="rule_confidence_threshold passed to DetectionConfig. "
             "Lower (e.g. 0.45) to surface borderline fasteners.",
    )
    p.add_argument(
        "--detail", action="store_true",
        help="Print the list of unmatched fastener names for each file.",
    )
    p.add_argument(
        "--emit-usd", action="store_true",
        help="Also write a .usdc next to each input STEP and time it.",
    )
    p.add_argument(
        "--usd-dir", default=None,
        help="Directory for emitted .usdc files (default: alongside source).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error("--data-dir does not exist: %s", data_dir)
        return 2
    files = sorted(
        f for f in data_dir.iterdir()
        if f.suffix.lower() in (".step", ".stp")
    )
    if not files:
        logger.error("No .step/.stp files under %s", data_dir)
        return 2

    logger.info("Found %d files in %s", len(files), data_dir)
    usd_dir = Path(args.usd_dir) if args.usd_dir else None
    if usd_dir:
        usd_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i, f in enumerate(files, 1):
        logger.info("[%d/%d] %s", i, len(files), f.name)
        row = evaluate_one(
            f, use_ml=args.ml, threshold=args.threshold,
            emit_usd=args.emit_usd, usd_dir=usd_dir,
        )
        rows.append(row)
        # One-line per-file echo so you see progress on long runs.
        print(
            f"  parts={row['parts_total']:>5} ({row['parts_unique']} unique)"
            f"  fasteners={row['fasteners_detected']:>4}"
            f"  matched={row['fasteners_with_match']:>4} ({row['match_rate']*100:>3.0f}%)"
            f"  arcs={row['screwedInto_arcs']:>4}"
            f"  contained_in={row['contained_in_arcs']:>4}"
            f"  t={row['time_total_s']:>5.1f}s"
            + (f"  ERR: {row['error']}" if row['error'] else "")
        )
        if args.detail and row["unmatched_fasteners"]:
            for name in row["unmatched_fasteners"][:10]:
                print(f"     · unmatched: {name}")
            if len(row["unmatched_fasteners"]) > 10:
                print(f"     · ... {len(row['unmatched_fasteners']) - 10} more")

    write_csv(rows, Path(args.out_csv))
    summarize(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
