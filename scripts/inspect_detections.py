"""Run detection on every STEP fixture and dump a readable per-file report
of every classified part and every emitted relationship.

The output is meant to be read by a human (or by the assistant) for
ground-truth grading: does each detection match the file's descriptive
name and the part's own name? Are the relationships sensible?
"""
from __future__ import annotations

import sys
import traceback
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


def inspect(step_path: Path, threshold: float = 0.45) -> str:
    out: list[str] = []
    out.append(f"\n{'='*78}")
    out.append(f"FILE: {step_path.name}")
    out.append(f"{'='*78}")
    try:
        doc, manifest, shapes = read_step(str(step_path), return_shapes=True)
    except Exception as e:
        out.append(f"  READ FAILED: {type(e).__name__}: {e}")
        return "\n".join(out)

    try:
        manifest = detect_fasteners(
            manifest, shapes=shapes,
            config=DetectionConfig(rule_confidence_threshold=threshold),
        )
    except Exception as e:
        out.append(f"  DETECT FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return "\n".join(out)

    by_uuid = {str(p.uuid): p for p in manifest.parts}
    real_parts = [p for p in manifest.parts if p.parent_uuid is not None]
    out.append(f"  parts: {len(manifest.parts)}  (non-root: {len(real_parts)})")

    out.append("\n  PARTS:")
    for p in real_parts:
        det = p.detection
        if det:
            tag = f"{det.fastener_type}({det.confidence:.2f})"
            if det.variant:
                tag += f" [{det.standard or ''} {det.variant}]"
        else:
            tag = "—"
        out.append(f"    {p.name[:40]:<40}  -> {tag}")

    fastener_rels = [r for r in manifest.relationships if r.kind == "fastener"]
    contained_rels = [r for r in manifest.relationships if r.kind == "contained_in"]
    out.append(f"\n  RELATIONSHIPS: {len(fastener_rels)} fastener arcs, {len(contained_rels)} contained_in arcs")

    if fastener_rels:
        out.append("    screwedInto:")
        for r in fastener_rels:
            subj = by_uuid.get(str(r.subject_uuid))
            tgt = by_uuid.get(str(r.target_uuid))
            sname = subj.name if subj else "?"
            tname = tgt.name if tgt else "?"
            params = r.params or {}
            out.append(
                f"      {sname[:25]:<25} -> {tname[:25]:<25}  "
                f"fit={params.get('fit_class','?'):<10} "
                f"hole={params.get('hole_kind','?'):<12} "
                f"d={params.get('hole_diameter','?')}mm "
                f"bolt_order={params.get('bolt_order','?')}"
            )

    if contained_rels:
        # Don't dump 100+ contained_in lines — just count by parent.
        from collections import Counter
        parents = Counter(str(r.target_uuid) for r in contained_rels)
        out.append("    contained_in (by parent):")
        for parent_uid, count in parents.most_common(5):
            pname = (by_uuid.get(parent_uid) or {})
            pname = pname.name if hasattr(pname, "name") else parent_uid[:8]
            out.append(f"      {count:>3}× -> {pname[:50]}")

    return "\n".join(out)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--substr", action="append", default=[],
                   help="Only inspect files whose name contains this substring "
                        "(can be repeated; case-insensitive).")
    p.add_argument("--exclude", action="append", default=[],
                   help="Skip files matching this substring.")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--out", default="inspection_report.txt")
    args = p.parse_args()

    data_dir = Path(__file__).resolve().parents[2] / "fastener_labeling" / "files"
    files = sorted(
        f for f in data_dir.iterdir()
        if f.suffix.lower() in (".step", ".stp")
    )
    if args.substr:
        keep = []
        for f in files:
            n = f.name.lower()
            if any(s.lower() in n for s in args.substr):
                keep.append(f)
        files = keep
    if args.exclude:
        files = [f for f in files if not any(s.lower() in f.name.lower() for s in args.exclude)]

    report_path = _BACKEND_ROOT / args.out
    chunks = []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}", file=sys.stderr, flush=True)
        chunks.append(inspect(f, threshold=args.threshold))
    report = "\n".join(chunks)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nWrote {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
