"""Diagnostic: read one STEP, extract B-Rep features per part, print them.

This is for figuring out *why* the rule-based classifier finds nothing on
real fixtures — if cylindrical_surface_ratio or aspect_ratio look wrong,
the issue is in feature extraction; if they look right, the gates in
rule_based.classify_part are too tight.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step_vr_step.readers.step_reader import read_step
from step_vr_step.detection.geometric_features import extract_brep_features
from step_vr_step.detection.rule_based import classify_part
from step_vr_step.config import DetectionConfig


def run(step_path: str) -> None:
    doc, manifest = read_step(step_path)
    config = DetectionConfig()
    print(f"\n=== {step_path} ===")
    print(f"parts: {len(manifest.parts)}")

    # The reader returns the OCC compound as `doc`; pull individual solids.
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID
    solids = []
    exp = TopExp_Explorer(doc, TopAbs_SOLID)
    while exp.More():
        solids.append(exp.Current())
        exp.Next()
    print(f"solids in compound: {len(solids)}")

    # Only non-root parts have geometry in the compound.
    real_parts = [p for p in manifest.parts if p.parent_uuid is not None]
    print(f"non-root parts: {len(real_parts)}")
    for i, (part, solid) in enumerate(zip(real_parts, solids)):
        feat = extract_brep_features(solid)
        label = classify_part(feat, config)
        radii_sorted = sorted({round(r, 2) for r in feat.cylindrical_face_radii})
        print(
            f"  [{i}] {part.name[:30]:<30}"
            f"  faces={feat.num_faces:>3}  "
            f"cyl_ratio={feat.cylindrical_surface_ratio:>5.2f}  "
            f"vol={feat.volume:>7.0f}  "
            f"aspect={feat.aspect_ratio:>5.2f}  "
            f"int_cyl={sum(1 for c in feat.cylinders if c.is_internal):>2}  "
            f"ext_cyl={sum(1 for c in feat.cylinders if not c.is_internal):>2}  "
            f"radii={radii_sorted[:5]}  "
            f"-> {label.fastener_type}({label.confidence:.2f})"
        )


if __name__ == "__main__":
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    targets = sys.argv[1:] or [
        "../fastener_labeling/files/hex_bolt with hex_nut2.step",
        "../fastener_labeling/files/multiple socket_screw.stp",
    ]
    for t in targets:
        try:
            run(t)
        except Exception as e:
            print(f"FAIL on {t}: {type(e).__name__}: {e}")
