"""Run BRep feature extraction on every fastener that the classifier
currently MISSES (returns unclassified) in the McMaster test set, and
print their features so we can see what they look like geometrically.

Output is grouped by feature signature so common patterns emerge:
  - "long-and-thin no head" → likely a stud/rivet/pin
  - "internal cylinder + external short" → likely an insert/coupling
  - "two ext clusters, no internal" → likely a shouldered rivet
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from step_vr_step.readers.step_reader import read_step
from step_vr_step.detection import detect_fasteners
from step_vr_step.detection.geometric_features import extract_brep_features
from step_vr_step.detection.brep_signature import classify_by_signature
from step_vr_step.config import DetectionConfig


def main():
    mc_root = _BACKEND_ROOT.parent / "training_data" / "mcmaster_binary" / "test" / "fastener"
    files = sorted(mc_root.iterdir())
    limit = 80   # sample
    found = 0
    for f in files[:limit * 2]:
        if found >= limit:
            break
        if f.suffix.lower() not in (".step", ".stp"):
            continue
        try:
            doc, manifest, shapes = read_step(str(f), return_shapes=True)
            m = detect_fasteners(
                manifest, shapes=shapes,
                config=DetectionConfig(rule_confidence_threshold=0.45),
            )
            # Was any part classified as a fastener?
            classified = any(
                p.detection and p.detection.fastener_type != "unclassified"
                and "unclassified" not in p.detection.fastener_type.replace("possible_", "")
                for p in m.parts
            )
            if classified:
                continue   # we want misses only
        except Exception:
            continue

        # Re-extract features for the largest solid
        try:
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_SOLID
            from OCC.Core.BRepGProp import brepgprop_VolumeProperties
            from OCC.Core.GProp import GProp_GProps
        except ImportError:
            return
        exp = TopExp_Explorer(doc, TopAbs_SOLID)
        solids = []
        while exp.More():
            solids.append(exp.Current())
            exp.Next()
        if not solids:
            continue
        # Largest by volume
        def _vol(s):
            g = GProp_GProps()
            brepgprop_VolumeProperties(s, g)
            return abs(g.Mass())
        best = max(solids, key=_vol)
        feat = extract_brep_features(best)
        sig = classify_by_signature(feat)
        n_int = sum(1 for c in feat.cylinders if c.is_internal)
        n_ext = sum(1 for c in feat.cylinders if not c.is_internal)
        n_plane = feat.face_type_counts.get("plane", 0)
        n_cone = feat.face_type_counts.get("cone", 0)
        n_torus = feat.face_type_counts.get("torus", 0)
        n_sphere = feat.face_type_counts.get("sphere", 0)
        ext_radii = sorted({round(c.radius, 2) for c in feat.cylinders if not c.is_internal})
        int_radii = sorted({round(c.radius, 2) for c in feat.cylinders if c.is_internal})
        dims = sorted([feat.bbox_max[i] - feat.bbox_min[i] for i in range(3)])
        print(
            f"{f.stem[:14]:<14} aspect={feat.aspect_ratio:>5.2f} "
            f"vol={feat.volume:>8.0f} "
            f"bbox={dims[0]:>5.1f}x{dims[1]:>5.1f}x{dims[2]:>5.1f}  "
            f"cyl_r={feat.cylindrical_surface_ratio:.2f} "
            f"faces={feat.num_faces:>3}(p{n_plane},c{n_cone},t{n_torus},s{n_sphere}) "
            f"ext={n_ext:>2}{ext_radii} int={n_int}{int_radii} "
            f"sig={sig.fastener_type if sig else '-'}"
        )
        found += 1


if __name__ == "__main__":
    main()
