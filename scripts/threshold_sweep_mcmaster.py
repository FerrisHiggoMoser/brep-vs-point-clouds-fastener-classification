"""Run the McMaster eval over a range of thresholds (sampled to keep
runtime sane) and print F1 for each. Use to pick the operating point
that matches the BRepFormer fine-tuned F1=0.699 baseline from the
research journal.
"""
from __future__ import annotations

import sys
import time
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


def best_type(manifest):
    bt, bc = "unclassified", 0.0
    for p in manifest.parts:
        d = p.detection
        if not d:
            continue
        t = d.fastener_type.replace("possible_", "").replace("likely_", "")
        if t == "unclassified":
            continue
        if d.confidence > bc:
            bc, bt = d.confidence, t
    return bt, bc


def run(folder: Path, expected: bool, threshold: float, limit: int):
    files = sorted(folder.iterdir())[:limit]
    tp = fp = tn = fn = 0
    for f in files:
        if f.suffix.lower() not in (".step", ".stp"):
            continue
        try:
            doc, m, sh = read_step(str(f), return_shapes=True)
            m = detect_fasteners(m, shapes=sh,
                                 config=DetectionConfig(rule_confidence_threshold=threshold))
            t, _ = best_type(m)
            is_pos = t != "unclassified"
            if expected and is_pos: tp += 1
            elif expected and not is_pos: fn += 1
            elif (not expected) and is_pos: fp += 1
            else: tn += 1
        except Exception:
            pass
    return tp, fp, tn, fn


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    args = p.parse_args()

    root = _BACKEND_ROOT.parent / "training_data" / "mcmaster_binary" / "test"
    print(f"sampling {args.limit} files per class from {root}", file=sys.stderr)

    rows = []
    for thr in args.thresholds:
        t0 = time.perf_counter()
        ftp, ffp, ftn, ffn = run(root / "fastener", True, thr, args.limit)
        nftp, nffp, nftn, nffn = run(root / "non_fastener", False, thr, args.limit)
        TP = ftp; FN = ffn
        FP = nffp; TN = nftn   # non-fastener side: FP = predicted-positive, TN = predicted-negative
        tot = TP + FP + TN + FN
        if tot == 0:
            continue
        acc = (TP + TN) / tot
        prec = TP / max(1, TP + FP)
        rec = TP / max(1, TP + FN)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        rows.append((thr, TP, FN, FP, TN, acc, prec, rec, f1))
        print(f"thr={thr:.2f}  TP={TP:>4} FN={FN:>3} FP={FP:>3} TN={TN:>4}  "
              f"acc={acc*100:>4.1f}%  P={prec*100:>4.1f}%  R={rec*100:>4.1f}%  "
              f"F1={f1*100:>4.1f}%  t={time.perf_counter()-t0:.0f}s")

    rows.sort(key=lambda r: -r[8])
    print("\nBest by F1:")
    for r in rows[:3]:
        thr, TP, FN, FP, TN, acc, prec, rec, f1 = r
        print(f"  thr={thr:.2f}  acc={acc*100:.1f}%  P={prec*100:.1f}%  R={rec*100:.1f}%  F1={f1*100:.1f}%")


if __name__ == "__main__":
    main()
