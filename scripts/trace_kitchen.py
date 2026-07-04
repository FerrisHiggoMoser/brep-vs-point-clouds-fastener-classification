"""Trace the score every hole gets against bolt_M3-tap_6 in the kitchen test."""
import sys
sys.path.insert(0, ".")
import numpy as np

from step_vr_step.readers.step_reader import read_step
from step_vr_step.detection.geometric_features import extract_brep_features
from step_vr_step.detection.holes import detect_holes
from step_vr_step.detection.detect import _principal_shaft, _score_hole_match, _unit


doc, manifest, shapes = read_step("synthetic_tests/test3_kitchen_sink.step", return_shapes=True)
plate = max(manifest.parts, key=lambda p: p.fingerprint.volume_mm3 if p.fingerprint else 0)
plate_shape = shapes[str(plate.uuid)]
plate_feat = extract_brep_features(plate_shape)
all_holes = detect_holes(str(plate.uuid), plate_feat)

# Find bolt_M3-tap
for p in manifest.parts:
    if "M3-tap" not in p.name:
        continue
    shape = shapes.get(str(p.uuid))
    if not shape:
        continue
    feat = extract_brep_features(shape)
    shaft = _principal_shaft(feat)
    print(f"\n{p.name}")
    print(f"  shaft: r={shaft.radius:.2f} origin={shaft.axis_origin}  length={shaft.length:.2f}")
    f_origin = np.asarray(shaft.axis_origin, dtype=np.float64)
    f_dir = _unit(np.asarray(shaft.axis_direction, dtype=np.float64))
    shaft_d = 2 * shaft.radius
    print(f"  scoring against {len(all_holes)} holes:")
    for hole in all_holes:
        score, info = _score_hole_match(f_origin, f_dir, shaft_d, shaft.length, hole)
        loc = tuple(round(x, 1) for x in hole.axis_origin)
        print(f"    hole d={hole.diameter:.2f} at {loc}: score={score:.3f}  info={info}")
    break
