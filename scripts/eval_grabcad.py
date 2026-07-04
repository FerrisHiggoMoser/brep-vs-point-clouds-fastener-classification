"""Per-class accuracy eval against the labeled grabcad training data.

Each subdirectory under training_data/grabcad/ is a ground-truth fastener
class (hex-bolt/, hex-nut/, flat-washer/, ...). This script runs the
BRep signature classifier on every file and reports per-class accuracy
plus a confusion summary.

Folder → expected type mapping is intentionally lenient: hex-bolt accepts
hex_bolt OR socket_head_cap_screw (both are "bolt-family externally
threaded"). non-fastener and other unsupported classes are expected to
return "unclassified" — false positives there penalize precision.
"""
from __future__ import annotations

import sys
import time
import traceback
from collections import Counter, defaultdict
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


# Folder name → set of acceptable detected types.
ACCEPTABLE: dict[str, set[str]] = {
    "hex-bolt":             {"hex_bolt", "socket_head_cap_screw"},
    "hex-nut":              {"hex_nut", "thin_hex_nut"},
    "flat-washer":          {"flat_washer"},
    "socket-head-cap-screw":{"socket_head_cap_screw", "hex_bolt"},
    "set-screw":            {"set_screw", "threaded_stud"},
    "stud-bolt":            {"threaded_stud", "hex_bolt"},
    "carriage-bolt":        {"hex_bolt"},
    "flange-bolt":          {"hex_bolt"},
    "eye-bolt":             {"hex_bolt"},
    "machine-screw":        {"hex_bolt", "socket_head_cap_screw", "wood_screw"},
    "cap-screw":            {"socket_head_cap_screw", "hex_bolt"},
    "shoulder-screw":       {"socket_head_cap_screw", "hex_bolt"},
    "nylon-lock-nut":       {"hex_nut"},
    "spring-washer":        {"flat_washer"},   # close-enough
    "snap-ring":            {"flat_washer"},   # disc-shaped
    "dowel-pin":            {"threaded_stud", "set_screw"},
    "cotter-pin":           set(),              # no class; expect unclassified
    "rivet":                set(),
    "insert":               set(),
    "anchor":               {"hex_bolt", "socket_head_cap_screw", "threaded_stud", "wood_screw"},
    "non-fastener":         set(),
    "wing-nut":             {"hex_nut"},
    "u-bolt":               {"hex_bolt"},
}


def evaluate_folder(folder: Path, expected: set[str], threshold: float, limit: int = 30) -> dict:
    files = sorted(f for f in folder.iterdir()
                   if f.suffix.lower() in (".step", ".stp"))[:limit]
    out = {"folder": folder.name, "n": len(files), "correct": 0,
           "false_pos": 0, "false_neg": 0, "errors": 0,
           "detected": Counter(), "samples": []}

    for f in files:
        try:
            doc, manifest, shapes = read_step(str(f), return_shapes=True)
            manifest = detect_fasteners(
                manifest, shapes=shapes,
                config=DetectionConfig(rule_confidence_threshold=threshold),
            )
            # Pick the "main" classification: the highest-confidence non-
            # unclassified detection in the file.
            best_type = "unclassified"
            best_conf = 0.0
            for p in manifest.parts:
                d = p.detection
                if not d:
                    continue
                t = d.fastener_type.replace("possible_", "").replace("likely_", "")
                if t == "unclassified":
                    continue
                if d.confidence > best_conf:
                    best_conf = d.confidence
                    best_type = t
            out["detected"][best_type] += 1
            is_correct = (
                (not expected and best_type == "unclassified")  # negative class
                or (best_type in expected)
            )
            if is_correct:
                out["correct"] += 1
            elif best_type == "unclassified" and expected:
                out["false_neg"] += 1
            elif not expected and best_type != "unclassified":
                out["false_pos"] += 1
            else:
                # Wrong type (classified, but not in expected set)
                out["false_pos"] += 1
            if len(out["samples"]) < 3:
                out["samples"].append(f"{f.name[:30]} -> {best_type}({best_conf:.2f})")
        except Exception as e:
            out["errors"] += 1
    return out


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root",
        default=str(_BACKEND_ROOT.parent / "training_data" / "grabcad"))
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--limit", type=int, default=30,
                   help="Max files per class folder.")
    args = p.parse_args()

    root = Path(args.root)
    rows = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        expected = ACCEPTABLE.get(sub.name, set())
        n = sum(1 for f in sub.iterdir()
                if f.suffix.lower() in (".step", ".stp"))
        if n == 0:
            continue
        print(f"[{sub.name}] expected={expected or 'unclassified'}  files={n}",
              file=sys.stderr, flush=True)
        result = evaluate_folder(sub, expected, args.threshold, args.limit)
        rows.append(result)

    print()
    print("=" * 90)
    print(f"{'class':<25} {'n':>4} {'corr':>5} {'%':>5}  detected histogram")
    print("=" * 90)
    total_n = total_corr = 0
    for r in rows:
        pct = 100.0 * r["correct"] / max(1, r["n"])
        total_n += r["n"]
        total_corr += r["correct"]
        hist = ", ".join(f"{t}:{c}" for t, c in r["detected"].most_common(4))
        print(f"{r['folder']:<25} {r['n']:>4} {r['correct']:>5} "
              f"{pct:>4.0f}%  {hist}")
        for s in r["samples"][:2]:
            print(f"    · {s}")
    print("=" * 90)
    overall = 100.0 * total_corr / max(1, total_n)
    print(f"OVERALL: {total_corr}/{total_n} = {overall:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
