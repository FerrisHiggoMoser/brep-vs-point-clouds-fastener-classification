"""Zero-shot deployment of the trained BRepFormer on the ISIS-Space
CubeSat structural panel. Journal expectation: BF binary F1 ~0.40
(worse than PN++ at 0.52 due to balanced-sampling-induced OOD bias);
with the 13-class subtype model, expected F1 0.55-0.65.

Runs the FULL detect_fasteners pipeline (rules + ML ensemble + hole
matcher + contained_in arcs) on the satellite STEP, then prints:

  - How many parts the model classified as each fastener subtype
  - How many `screwedInto` (fastener-to-host) relationships emitted
  - How many `contained_in` (housing) relationships emitted
  - The first few classified fasteners with descriptive names
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


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--step",
        default=str(_BACKEND_ROOT.parent / "training_data" / "holdout_test"
                    / "isispace_1uplt_type_b_2023-04-20.stp"))
    p.add_argument("--ckpt", required=True,
                   help="Trained BRepFormer checkpoint")
    p.add_argument("--threshold", type=float, default=0.35,
                   help="Rule-based confidence threshold")
    p.add_argument("--ml-threshold", type=float, default=0.50,
                   help="ML confidence threshold")
    args = p.parse_args()

    step_path = Path(args.step)
    if not step_path.exists():
        print(f"ERROR: {step_path} not found", file=sys.stderr)
        return 2

    print(f"Loading satellite STEP: {step_path.name}  ({step_path.stat().st_size / 1e6:.1f} MB)",
          file=sys.stderr)
    t0 = time.perf_counter()
    doc, manifest, shapes = read_step(str(step_path), return_shapes=True)
    print(f"  read in {time.perf_counter()-t0:.1f}s   parts={len(manifest.parts)}",
          file=sys.stderr)

    cfg = DetectionConfig(
        enable_ml=True,
        brepformer_weights=args.ckpt,
        rule_confidence_threshold=args.threshold,
        ml_confidence_threshold=args.ml_threshold,
    )

    print(f"Running detect_fasteners with ML…", file=sys.stderr)
    t0 = time.perf_counter()
    manifest = detect_fasteners(manifest, shapes=shapes, config=cfg)
    print(f"  detection done in {time.perf_counter()-t0:.1f}s", file=sys.stderr)

    # Tally classifications
    types = Counter()
    confs = []
    classified_parts = []
    for part in manifest.parts:
        d = part.detection
        if not d:
            continue
        t = d.fastener_type.replace("possible_", "").replace("likely_", "")
        types[t] += 1
        if t != "unclassified":
            confs.append(d.confidence)
            classified_parts.append((part.name, t, d.confidence, d.method))

    n_fast = sum(c for t, c in types.items() if t != "unclassified")
    fasteners_arcs = sum(1 for r in manifest.relationships if r.kind == "fastener")
    contained_arcs = sum(1 for r in manifest.relationships if r.kind == "contained_in")

    print()
    print("=" * 70)
    print(f"ISIS Satellite zero-shot — BRepFormer + rules ensemble")
    print("=" * 70)
    print(f"Checkpoint:        {args.ckpt}")
    print(f"Parts total:       {len(manifest.parts)}")
    print(f"Fasteners found:   {n_fast}  ({n_fast / max(1, len(manifest.parts)) * 100:.1f}%)")
    print(f"screwedInto arcs:  {fasteners_arcs}")
    print(f"contained_in arcs: {contained_arcs}")
    if confs:
        import statistics
        print(f"Confidence (mean): {statistics.mean(confs):.2f}   "
              f"(median {statistics.median(confs):.2f}, "
              f"min {min(confs):.2f}, max {max(confs):.2f})")

    print()
    print("Predicted-type histogram:")
    for t, n in types.most_common(15):
        print(f"  {t:<30}  {n:>5}")

    # Full per-part dump for manual verification
    import csv
    out_csv = Path("satellite_classifications.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["part_name", "predicted_type", "confidence", "method"])
        for name, t, c, method in classified_parts:
            w.writerow([name, t, f"{c:.3f}", method])

    print()
    print(f"Wrote full per-part dump to {out_csv}  ({len(classified_parts)} classified parts)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
