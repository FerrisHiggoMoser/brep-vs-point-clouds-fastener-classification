"""Per-part disagreement audit for the BRepFormer McMaster eval.

Reads the predictions CSV produced by `eval_ml_mcmaster.py`, finds every
file where the model's prediction disagreed with the McMaster ground-
truth folder, and prints a table the assistant (or a human) can review:

    file                        | gt           | pred         | conf  | reason
    1894N12_Precision Alignment | fastener     | non_fastener | 0.78  | <verdict>

Also re-reads each disagreement file to pull the McMaster descriptive
name (which is encoded as the OCC label name inside the STEP). That
name reveals whether the model was actually right (e.g. "Ball Bearing"
mis-filed as a fastener) or wrong (a real bolt missed).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


def descriptive_name(step_path: Path) -> str:
    """Extract the descriptive name from a McMaster STEP file.

    McMaster STEPs encode the descriptive name as the first arg of a
    `PRODUCT ( '...' )` entry. The entry is usually a few KB into the
    file (after the header) but in some files can be 100KB+ in. Scan
    iteratively rather than slurping the whole file.
    """
    try:
        import re
        pat = re.compile(rb"PRODUCT\s*\(\s*'([^']+)'", re.IGNORECASE)
        with step_path.open("rb") as f:
            # Stream up to 2 MB looking for the first PRODUCT entry
            chunk_size = 256 * 1024
            buf = b""
            for _ in range(8):  # up to 2 MB total
                blk = f.read(chunk_size)
                if not blk:
                    break
                buf += blk
                m = pat.search(buf)
                if m:
                    return m.group(1).decode("utf-8", errors="replace")
    except Exception:
        pass
    return step_path.stem


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="ml_mcmaster_predictions.csv",
                   help="CSV produced by eval_ml_mcmaster.py")
    p.add_argument("--root",
        default=str(_BACKEND_ROOT.parent / "training_data" / "mcmaster_binary" / "test"))
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run eval_ml_mcmaster.py first.", file=sys.stderr)
        return 2

    root = Path(args.root)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    disagreements = [r for r in rows if int(r["correct"]) == 0]

    print(f"Total: {len(rows)}, disagreements: {len(disagreements)}", file=sys.stderr)

    # Categorize: FN (missed real fasteners) vs FP (non-fasteners called fasteners)
    fn = [r for r in disagreements if r["gt"] == "fastener"]
    fp = [r for r in disagreements if r["gt"] == "non_fastener"]

    print()
    print("=" * 100)
    print(f"FALSE NEGATIVES — {len(fn)} fasteners the model missed")
    print("=" * 100)
    print(f"{'SKU':<18} {'GT':<14} {'PRED':<18} {'CONF':<6}  descriptive name")
    print("-" * 100)
    for r in fn:
        # Look up the descriptive name from the STEP file
        step_file = root / "fastener" / f"{r['file']}.step"
        if not step_file.exists():
            step_file = root / "fastener" / f"{r['file']}.stp"
        desc = descriptive_name(step_file) if step_file.exists() else "(file not found)"
        print(f"{r['file'][:18]:<18} {r['gt']:<14} {r['pred_class']:<18} {r['confidence']:<6}  {desc[:60]}")

    print()
    print("=" * 100)
    print(f"FALSE POSITIVES — {len(fp)} non-fasteners the model called fasteners")
    print("=" * 100)
    print(f"{'SKU':<18} {'GT':<14} {'PRED':<18} {'CONF':<6}  descriptive name")
    print("-" * 100)
    for r in fp:
        step_file = root / "non_fastener" / f"{r['file']}.step"
        if not step_file.exists():
            step_file = root / "non_fastener" / f"{r['file']}.stp"
        desc = descriptive_name(step_file) if step_file.exists() else "(file not found)"
        print(f"{r['file'][:18]:<18} {r['gt']:<14} {r['pred_class']:<18} {r['confidence']:<6}  {desc[:60]}")

    print()
    print(f"Summary: {len(fn)} FN, {len(fp)} FP, {len(rows) - len(disagreements)} correct of {len(rows)} total")


if __name__ == "__main__":
    sys.exit(main())
