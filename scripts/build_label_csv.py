"""Build a CSV + STEP-file folder for manual fastener sub-classification.

Walks every fastener-class STEP under <src>, copies the actual file
(dereferencing XSym stubs from the Mac→PC transfer), and writes a CSV
with one row per file:
    file, split, proposed_class, confidence, final_class

`final_class` is left blank for manual fill. `proposed_class` is auto-derived
from filename heuristics (DIN/ISO codes, M-size patterns, common keywords).
`confidence` is "high" if a strong rule matched, "low" otherwise.

Usage:
    python backend/scripts/build_label_csv.py \
        --src training_data/organized \
        --grabcad_root training_data/grabcad \
        --dst <some local folder>

Output:
    <dst>/labels_to_review.csv
    <dst>/files/<original_filename>.STEP    (real, dereferenced)
"""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _BACKEND.parent
sys.path.insert(0, str(_BACKEND))

from scripts.step_to_brep import resolve_xsym  # reuses the XSym dereferencer

CLASSES = (
    "hex_bolt", "cap_screw", "socket_screw", "countersunk_screw", "set_screw",
    "hex_nut", "lock_nut", "wing_nut",
    "flat_washer", "spring_washer", "lock_washer",
    "rivet", "pin", "snap_ring", "stud", "anchor",
    "_skip", "_unknown",  # control labels
)

# Strong signals: filename patterns → class
RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"din\s*933", re.I), "hex_bolt"),
    (re.compile(r"din\s*931", re.I), "hex_bolt"),
    (re.compile(r"din\s*912", re.I), "socket_screw"),
    (re.compile(r"din\s*7984", re.I), "socket_screw"),
    (re.compile(r"din\s*7991", re.I), "countersunk_screw"),
    (re.compile(r"iso\s*4762", re.I), "socket_screw"),
    (re.compile(r"iso\s*4014", re.I), "hex_bolt"),
    (re.compile(r"iso\s*4017", re.I), "hex_bolt"),
    (re.compile(r"iso\s*10642", re.I), "countersunk_screw"),
    (re.compile(r"iso\s*1458[0-4]", re.I), "countersunk_screw"),
    (re.compile(r"iso\s*7380", re.I), "cap_screw"),
    (re.compile(r"din\s*934", re.I), "hex_nut"),
    (re.compile(r"din\s*985", re.I), "lock_nut"),
    (re.compile(r"iso\s*4032", re.I), "hex_nut"),
    (re.compile(r"din\s*125", re.I), "flat_washer"),
    (re.compile(r"din\s*127", re.I), "spring_washer"),
    (re.compile(r"din\s*5406", re.I), "lock_washer"),
    (re.compile(r"iso\s*7089", re.I), "flat_washer"),
    (re.compile(r"iso\s*7091", re.I), "flat_washer"),
    (re.compile(r"\bcountersunk\b", re.I), "countersunk_screw"),
    (re.compile(r"\bcsk\b", re.I), "countersunk_screw"),
    (re.compile(r"flat[\s_-]*head", re.I), "countersunk_screw"),
    (re.compile(r"\bcap[\s_-]*screw\b", re.I), "cap_screw"),
    (re.compile(r"\bsocket[\s_-]*head\b|\ballen\b", re.I), "socket_screw"),
    (re.compile(r"set[\s_-]*screw|grub", re.I), "set_screw"),
    (re.compile(r"\bbolt\b", re.I), "hex_bolt"),
    (re.compile(r"\bhex[\s_-]*head\b", re.I), "hex_bolt"),
    (re.compile(r"\bcarriage\b", re.I), "hex_bolt"),
    (re.compile(r"\bhex[\s_-]*nut\b|\bhexagon[\s_-]*nut\b", re.I), "hex_nut"),
    (re.compile(r"\block[\s_-]*nut\b|nylock|locknut", re.I), "lock_nut"),
    (re.compile(r"\bwing[\s_-]*nut\b|wingnut", re.I), "wing_nut"),
    (re.compile(r"\bspring[\s_-]*washer\b", re.I), "spring_washer"),
    (re.compile(r"\block[\s_-]*washer\b", re.I), "lock_washer"),
    (re.compile(r"flat[\s_-]*washer|\bplain[\s_-]*washer\b|\bwasher\b", re.I), "flat_washer"),
    (re.compile(r"\brivet\b", re.I), "rivet"),
    (re.compile(r"\bcirclip\b|snap[\s_-]*ring|retain", re.I), "snap_ring"),
    (re.compile(r"\bstud\b", re.I), "stud"),
    (re.compile(r"\bthreaded[\s_-]*rod\b", re.I), "stud"),
    (re.compile(r"\banchor\b", re.I), "anchor"),
    (re.compile(r"\bdowel\b|cylindrical[\s_-]*pin|taper[\s_-]*pin|roll[\s_-]*pin|split[\s_-]*pin", re.I), "pin"),
    (re.compile(r"_pin\b|^pin_", re.I), "pin"),
    (re.compile(r"\bscrew\b", re.I), "cap_screw"),  # generic fallback for "screw"
    (re.compile(r"\bnut\b", re.I), "hex_nut"),       # generic fallback for "nut"
]

# Things that look like assemblies / not single fasteners → _skip
SKIP_PATTERNS = (
    re.compile(r"\bassem(bly)?\b|_asm\b", re.I),
    re.compile(r"compound", re.I),
)


def propose_class(name: str) -> tuple[str, str]:
    """Return (proposed_class, confidence). confidence is 'high' / 'low'."""
    if any(p.search(name) for p in SKIP_PATTERNS):
        return "_skip", "high"
    for pat, cls in RULES:
        if pat.search(name):
            return cls, "high"
    return "_unknown", "low"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path,
                   help="organized/ tree (split/class/file.stp)")
    p.add_argument("--dst", required=True, type=Path,
                   help="output folder for CSV + STEP copies")
    args = p.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    files_dir = dst / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for split_dir in sorted([d for d in src.iterdir() if d.is_dir()]):
        cls_dir = split_dir / "fastener"
        if not cls_dir.is_dir():
            continue
        for stub in sorted(cls_dir.iterdir()):
            if stub.suffix.lower() not in (".step", ".stp"):
                continue
            real_path = resolve_xsym(stub)
            if not real_path.exists():
                continue
            try:
                shutil.copy2(real_path, files_dir / stub.name)
            except OSError:
                continue
            proposed, conf = propose_class(stub.name)
            rows.append({
                "file": stub.name,
                "split": split_dir.name,
                "proposed_class": proposed,
                "confidence": conf,
                "final_class": proposed if conf == "high" else "",
            })

    csv_path = dst / "labels_to_review.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "split", "proposed_class", "confidence", "final_class"])
        w.writeheader()
        w.writerows(rows)

    classes_path = dst / "VALID_CLASSES.txt"
    classes_path.write_text(
        "Valid values for the 'final_class' column:\n\n" +
        "\n".join(f"  {c}" for c in CLASSES) +
        "\n\nLeave a row's final_class blank ONLY if you skipped it; otherwise pick exactly one of these.\n",
        encoding="utf-8",
    )

    n_high = sum(1 for r in rows if r["confidence"] == "high")
    n_low = len(rows) - n_high
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"  high-confidence auto-labels: {n_high}  (final_class pre-filled)")
    print(f"  low-confidence (NEEDS REVIEW): {n_low}  (final_class blank)")
    print(f"STEP files copied (dereferenced): {len(rows)} → {files_dir}")
    print(f"Class reference: {classes_path}")


if __name__ == "__main__":
    main()
