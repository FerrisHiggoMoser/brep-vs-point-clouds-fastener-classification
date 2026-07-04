"""Evaluate the BRep signature classifier on McMaster-Carr ground truth.

mcmaster_binary/test/ contains:
  - fastener/<SKU>.step       — should classify as SOME fastener type
  - non_fastener/<SKU>.step   — should classify as unclassified

This is a real binary evaluation: TP, FP, TN, FN at the
"is this part a fastener?" level, plus a histogram of detected types on
the fastener set so you can see the type breakdown.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from step_vr_step.readers.step_reader import read_step
from step_vr_step.detection import detect_fasteners
from step_vr_step.config import DetectionConfig


def best_type(manifest) -> tuple[str, float]:
    """Return (type, confidence) of the highest-confidence part. 'unclassified'
    means nothing in the file was identified as a fastener."""
    best_t, best_c = "unclassified", 0.0
    for p in manifest.parts:
        d = p.detection
        if not d:
            continue
        t = d.fastener_type.replace("possible_", "").replace("likely_", "")
        if t == "unclassified":
            continue
        if d.confidence > best_c:
            best_c = d.confidence
            best_t = t
    return best_t, best_c


def evaluate(folder: Path, expected_fastener: bool, threshold: float, limit: int) -> dict:
    files = sorted(folder.iterdir())[:limit]
    out = {
        "n": 0, "correct": 0, "wrong": 0, "errors": 0,
        "types": Counter(),
        "wrong_samples": [],
    }
    for f in files:
        if f.suffix.lower() not in (".step", ".stp"):
            continue
        out["n"] += 1
        try:
            doc, manifest, shapes = read_step(str(f), return_shapes=True)
            manifest = detect_fasteners(
                manifest, shapes=shapes,
                config=DetectionConfig(rule_confidence_threshold=threshold),
            )
            t, c = best_type(manifest)
            out["types"][t] += 1
            is_classified = (t != "unclassified")
            if expected_fastener and is_classified:
                out["correct"] += 1
            elif (not expected_fastener) and (not is_classified):
                out["correct"] += 1
            else:
                out["wrong"] += 1
                if len(out["wrong_samples"]) < 5:
                    out["wrong_samples"].append(f"{f.stem[:20]} -> {t}({c:.2f})")
        except Exception as e:
            out["errors"] += 1
    return out


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=str(_BACKEND_ROOT.parent / "training_data" / "mcmaster_binary" / "test"),
    )
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--limit", type=int, default=100,
                   help="Max files per class (default 100; use 9999 for all).")
    args = p.parse_args()

    root = Path(args.root)
    print(f"Evaluating: {root}  threshold={args.threshold}  limit={args.limit}", file=sys.stderr)

    print(file=sys.stderr); print(f"[fastener/]   ground truth: IS a fastener", file=sys.stderr)
    t0 = time.perf_counter()
    fst = evaluate(root / "fastener", True, args.threshold, args.limit)
    fst_t = time.perf_counter() - t0

    print(file=sys.stderr); print(f"[non_fastener/]   ground truth: NOT a fastener", file=sys.stderr)
    t0 = time.perf_counter()
    nf = evaluate(root / "non_fastener", False, args.threshold, args.limit)
    nf_t = time.perf_counter() - t0

    # Binary confusion matrix.
    TP = fst["correct"]                  # is fastener, predicted fastener
    FN = fst["wrong"]                    # is fastener, predicted not
    TN = nf["correct"]                   # not fastener, predicted not
    FP = nf["wrong"]                     # not fastener, predicted fastener

    tot = TP + TN + FP + FN
    acc = (TP + TN) / max(1, tot)
    prec = TP / max(1, TP + FP)
    rec = TP / max(1, TP + FN)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    print()
    print("=" * 70)
    print(f"McMaster test split — binary fastener detection")
    print("=" * 70)
    print(f"Confusion matrix (n = {tot}):")
    print(f"                       predicted fastener   predicted not")
    print(f"  actual fastener      TP = {TP:>6}            FN = {FN:>6}")
    print(f"  actual not-fastener  FP = {FP:>6}            TN = {TN:>6}")
    print()
    print(f"  Accuracy   = {acc*100:>5.1f}%   (overall correctness)")
    print(f"  Precision  = {prec*100:>5.1f}%   (of predicted fasteners, how many real)")
    print(f"  Recall     = {rec*100:>5.1f}%   (of actual fasteners, how many caught)")
    print(f"  F1         = {f1*100:>5.1f}%")
    print()
    print(f"Type histogram (on fastener/ folder, only correct/wrong tells you "
          f"that something was found):")
    for t, c in fst["types"].most_common(10):
        print(f"  {t:<30}  {c:>4}")
    if fst["wrong_samples"]:
        print(f"\nA few fasteners we MISSED:")
        for s in fst["wrong_samples"][:5]:
            print(f"  · {s}")
    if nf["wrong_samples"]:
        print(f"\nA few non-fasteners we MIS-classified:")
        for s in nf["wrong_samples"][:5]:
            print(f"  · {s}")
    print()
    print(f"Read+detect time: fastener {fst_t:.1f}s ({fst['n']} files), "
          f"non_fastener {nf_t:.1f}s ({nf['n']} files); errors {fst['errors']}+{nf['errors']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
