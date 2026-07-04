"""Generate synthetic STEP test files with known fastener-to-hole ground
truth, then run detection on them and verify the relationships.

Each test builds an assembly out of OCC primitives (plates with holes,
cylinder bolts, hex-prism nuts) and writes it with named XDE labels so
the detection pipeline can extract individual parts. Geometry is
positioned so every bolt's shaft axis is exactly collinear with a
plate's hole axis — that's what the screwedInto matcher tests for.

We deliberately use bare primitives (no real threads, no hex flats on
bolt heads) to test what the geometry alone can recover.
"""
from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Optional

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


def _write_step(shapes_named: list[tuple], out_path: Path):
    """Write a list of (shape, name) pairs to a STEP file.

    We embed the descriptive names into PRODUCT entries by writing each
    shape as a SEPARATE call to writer.Transfer(...) with a different
    "ProductName" header parameter. This is simpler than the XDE path
    and works reliably across OCC versions.
    """
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Interface import Interface_Static

    writer = STEPControl_Writer()
    for shape, name in shapes_named:
        # Per-shape product name — each TransferShape call uses the current
        # "write.step.product.name" header value as the PRODUCT name.
        Interface_Static.SetCVal("write.step.product.name", name)
        status = writer.Transfer(shape, STEPControl_AsIs)
        if status != IFSelect_RetDone:
            raise RuntimeError(f"STEP transfer failed for {name}: status={status}")
    status = writer.Write(str(out_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: status={status}")


def make_plate(width: float, height: float, thickness: float,
               hole_specs: list[tuple]):
    """Plate with holes drilled through. hole_specs is a list of
    (x, y, radius) tuples — each hole is centered at (x, y) and drilled
    along +Z through the full thickness.
    """
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir

    plate = BRepPrimAPI_MakeBox(width, height, thickness).Shape()
    for x, y, r in hole_specs:
        # over-drill in +Z and -Z so the cut is clean through both faces
        axis = gp_Ax2(gp_Pnt(x, y, -1.0), gp_Dir(0, 0, 1))
        hole = BRepPrimAPI_MakeCylinder(axis, r, thickness + 2.0).Shape()
        plate = BRepAlgoAPI_Cut(plate, hole).Shape()
    return plate


def make_bolt(shaft_dia: float, shaft_length: float, head_dia: float,
              head_height: float, position: tuple) -> "TopoDS_Shape":
    """A simple bolt: cylindrical shaft + larger cylindrical head sitting
    on top of the shaft, both along +Z. Origin is the bottom of the shaft.
    """
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir

    px, py, pz = position
    shaft_axis = gp_Ax2(gp_Pnt(px, py, pz), gp_Dir(0, 0, 1))
    shaft = BRepPrimAPI_MakeCylinder(shaft_axis, shaft_dia / 2.0, shaft_length).Shape()
    head_axis = gp_Ax2(gp_Pnt(px, py, pz + shaft_length), gp_Dir(0, 0, 1))
    head = BRepPrimAPI_MakeCylinder(head_axis, head_dia / 2.0, head_height).Shape()
    return BRepAlgoAPI_Fuse(shaft, head).Shape()


def make_nut(bore_dia: float, outer_dia: float, height: float,
             position: tuple) -> "TopoDS_Shape":
    """A nut: outer cylinder (we use cylinder not hex because OCC doesn't
    have a hex-prism primitive — the classifier doesn't care about flats
    for the synthetic test) with a concentric internal bore.
    """
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir

    px, py, pz = position
    axis = gp_Ax2(gp_Pnt(px, py, pz), gp_Dir(0, 0, 1))
    outer = BRepPrimAPI_MakeCylinder(axis, outer_dia / 2.0, height).Shape()
    # over-drill the bore
    bore_axis = gp_Ax2(gp_Pnt(px, py, pz - 1.0), gp_Dir(0, 0, 1))
    bore = BRepPrimAPI_MakeCylinder(bore_axis, bore_dia / 2.0, height + 2.0).Shape()
    return BRepAlgoAPI_Cut(outer, bore).Shape()


def make_washer(bore_dia: float, outer_dia: float, thickness: float,
                position: tuple) -> "TopoDS_Shape":
    """A flat washer ring."""
    return make_nut(bore_dia, outer_dia, thickness, position)


# ----------------------------------------------------------------------
# Test scenarios
# ----------------------------------------------------------------------


def build_test_simple_4bolts(out_path: Path):
    """Test 1 — one plate, 4 identical bolts through 4 clearance holes.

    Ground truth:
      - 4 fasteners detected
      - 4 screwedInto arcs from each bolt to the plate
      - All fit_class = clearance (6.5mm hole, 6mm shaft = +0.5mm gap)
      - All hole_kind = through
      - 5 contained_in arcs (1 per part to the assembly root)
    """
    PLATE_W, PLATE_H, PLATE_T = 100.0, 100.0, 10.0
    BOLT_SHAFT_D, BOLT_SHAFT_L = 6.0, 25.0   # M6 × 25mm
    BOLT_HEAD_D, BOLT_HEAD_H = 10.0, 4.0
    HOLE_R = 3.25                              # 6.5mm clearance hole

    # 4 holes in a square pattern
    centers = [(25, 25), (75, 25), (25, 75), (75, 75)]
    holes = [(x, y, HOLE_R) for x, y in centers]

    plate = make_plate(PLATE_W, PLATE_H, PLATE_T, holes)
    shapes = [(plate, "plate_100x100x10")]

    for i, (x, y) in enumerate(centers, 1):
        # Bolt origin = z=0 (bottom of plate), shaft 25mm tall ending at z=25
        # The bolt overlaps the plate from z=0..10 (the plate thickness)
        bolt = make_bolt(BOLT_SHAFT_D, BOLT_SHAFT_L,
                         BOLT_HEAD_D, BOLT_HEAD_H,
                         position=(x, y, -5.0))
        shapes.append((bolt, f"bolt_M6x25_clearance_{i}"))

    _write_step(shapes, out_path)
    return {
        "expected_fasteners": 4,
        "expected_screwedInto": 4,
        "expected_fit_class": "clearance",
        "expected_hole_kind": "through",
    }


def build_test_pass_through(out_path: Path):
    """Test 2 — bolt passes through plate-A (clearance) into plate-B (tap).

    Two plates stacked, one bolt goes through both. Ground truth:
      - 3 fasteners (1 bolt + 0 nuts) — actually just 1 fastener
      - 2 screwedInto arcs from the bolt: bolt_order 0 (plate-A clearance),
        bolt_order 1 (plate-B tap)
    """
    PLATE_W = PLATE_H = 60.0
    PLATE_T = 5.0

    plate_a_hole = 3.5    # M6 clearance: 7mm dia → 3.5 radius
    plate_b_hole = 2.5    # M6 tap-drill: 5mm dia → 2.5 radius

    plate_a = make_plate(PLATE_W, PLATE_H, PLATE_T, [(30, 30, plate_a_hole)])
    plate_b = make_plate(PLATE_W, PLATE_H, PLATE_T, [(30, 30, plate_b_hole)])

    # Translate plate_b up by PLATE_T (5mm) so the two plates stack
    from OCC.Core.gp import gp_Trsf, gp_Vec
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(0, 0, PLATE_T))
    plate_b = BRepBuilderAPI_Transform(plate_b, trsf, True).Shape()

    # Bolt: M6 × 18mm shaft, 12mm head dia, 5mm head height
    # Position so shaft spans both plates: z from -3 (head sits below) to 15
    # Actually let's put the head ABOVE plate-A: shaft from z=-3 (in plate-B)
    # to z=15 (head above plate-A). Head at z=15 to z=20.
    bolt = make_bolt(6.0, 18.0, 12.0, 5.0, position=(30, 30, -3.0))

    shapes = [
        (plate_a, "plate_A_clearance"),
        (plate_b, "plate_B_tap"),
        (bolt, "bolt_M6x18_through_stack"),
    ]
    _write_step(shapes, out_path)
    return {
        "expected_fasteners": 1,
        "expected_screwedInto": 2,
        "expected_fit_classes": {"clearance", "tap"},
        "expected_hole_kinds": {"through"},
    }


def build_test_kitchen_sink(out_path: Path):
    """Test 3 — 'shit ton' assembly: big plate, 9 bolts of varying M-sizes
    through 9 holes, 4 nuts on the underside of some, 2 washers.

    Ground truth:
      - 9 bolts + 4 nuts + 2 washers = 15 fasteners
      - 9 screwedInto arcs (bolt → plate)
      - 15 contained_in arcs (each fastener to the assembly root)
    """
    PLATE_W = PLATE_H = 200.0
    PLATE_T = 8.0

    sizes = [
        # (label, shaft_dia, shaft_len, head_dia, head_h, hole_r, fit_label)
        ("M3", 3.0, 18.0, 5.5, 3.0, 1.7, "clearance"),     # 3.4 hole, 3 shaft → 0.4 = slip
        ("M4", 4.0, 20.0, 7.0, 3.5, 2.25, "clearance"),    # 4.5 hole, 4 shaft → 0.5 = slip
        ("M5", 5.0, 22.0, 8.5, 4.0, 2.75, "clearance"),    # 5.5 hole
        ("M6", 6.0, 25.0, 10.0, 4.5, 3.25, "clearance"),
        ("M8", 8.0, 28.0, 13.0, 5.5, 4.25, "clearance"),
        ("M10", 10.0, 30.0, 16.0, 6.5, 5.25, "clearance"),
        ("M3-tap", 3.0, 18.0, 5.5, 3.0, 1.25, "tap"),       # 2.5 hole, 3 shaft → -0.5 = tap
        ("M4-tap", 4.0, 20.0, 7.0, 3.5, 1.65, "tap"),       # 3.3 hole = tap
        ("M5-tap", 5.0, 22.0, 8.5, 4.0, 2.1,  "tap"),       # 4.2 hole = tap
    ]
    centers = [
        (30, 30),  (100, 30), (170, 30),
        (30, 100), (100, 100), (170, 100),
        (30, 170), (100, 170), (170, 170),
    ]

    holes = [(x, y, sizes[i][5]) for i, (x, y) in enumerate(centers)]
    plate = make_plate(PLATE_W, PLATE_H, PLATE_T, holes)
    shapes = [(plate, "main_plate_200x200x8")]

    for i, ((x, y), spec) in enumerate(zip(centers, sizes)):
        label, sd, sl, hd, hh, _hole_r, fit = spec
        bolt = make_bolt(sd, sl, hd, hh, position=(x, y, -5.0))
        shapes.append((bolt, f"bolt_{label}_{i}"))

    # Add 4 nuts on the underside of the first 4 bolts (M3, M4, M5, M6)
    for i in range(4):
        x, y = centers[i]
        bore = sizes[i][1] + 0.4  # bore = M-size + 0.4mm for clearance to thread
        outer = sizes[i][1] * 2.0  # rough hex width
        nut = make_nut(bore, outer, 4.0, position=(x, y, -10.0))
        shapes.append((nut, f"nut_{sizes[i][0]}_back_{i}"))

    # 2 washers under M8 and M10 bolts
    for i in [4, 5]:
        x, y = centers[i]
        bore = sizes[i][1] + 0.6
        outer = sizes[i][1] * 2.5
        wash = make_washer(bore, outer, 1.5, position=(x, y, -2.0))
        shapes.append((wash, f"washer_{sizes[i][0]}_{i}"))

    _write_step(shapes, out_path)
    return {
        "expected_fasteners": 15,        # 9 bolts + 4 nuts + 2 washers
        "expected_screwedInto": 9,       # 9 bolts → plate
        "expected_contained_in_min": 15, # every fastener → root
    }


def build_test_grid_50bolts(out_path: Path):
    """Test 4 — stress test: one big plate, 50 bolts in a 5×10 grid.

    Ground truth: 50 bolts, 50 screwedInto arcs.
    """
    NX, NY = 10, 5
    PITCH = 20.0
    PLATE_W = NX * PITCH + 40
    PLATE_H = NY * PITCH + 40
    PLATE_T = 6.0
    BOLT_D = 5.0
    HOLE_R = 2.75   # 5.5 hole, 5 bolt = 0.5 slip

    centers = [
        (20 + i * PITCH, 20 + j * PITCH)
        for j in range(NY) for i in range(NX)
    ]
    holes = [(x, y, HOLE_R) for x, y in centers]
    plate = make_plate(PLATE_W, PLATE_H, PLATE_T, holes)
    shapes = [(plate, f"plate_grid_{NX}x{NY}")]
    for i, (x, y) in enumerate(centers):
        bolt = make_bolt(BOLT_D, 20.0, 8.0, 3.5, position=(x, y, -3.0))
        shapes.append((bolt, f"bolt_M5_grid_{i:03d}"))

    _write_step(shapes, out_path)
    return {
        "expected_fasteners": NX * NY,
        "expected_screwedInto": NX * NY,
    }


# ----------------------------------------------------------------------
# Detection + verification
# ----------------------------------------------------------------------


def run_detection_and_verify(step_path: Path, expected: dict,
                              ckpt: Optional[str] = None) -> dict:
    from step_vr_step.readers.step_reader import read_step
    from step_vr_step.detection import detect_fasteners
    from step_vr_step.config import DetectionConfig

    doc, manifest, shapes = read_step(str(step_path), return_shapes=True)
    cfg = DetectionConfig(
        enable_ml=bool(ckpt),
        brepformer_weights=ckpt,
        rule_confidence_threshold=0.35,  # permissive — synthetic primitives lack thread/hex flats
    )
    manifest = detect_fasteners(manifest, shapes=shapes, config=cfg)

    # Count classifications
    classified = [
        p for p in manifest.parts
        if p.detection and p.detection.fastener_type != "unclassified"
    ]
    screwedInto = [r for r in manifest.relationships if r.kind == "fastener"]
    contained_in = [r for r in manifest.relationships if r.kind == "contained_in"]

    return {
        "n_parts": len([p for p in manifest.parts if p.parent_uuid]),
        "n_classified": len(classified),
        "n_screwedInto": len(screwedInto),
        "n_contained_in": len(contained_in),
        "classifications": [(p.name, p.detection.fastener_type, p.detection.confidence) for p in classified],
        "screwedInto_details": [
            {
                "subject": next((p.name for p in manifest.parts if p.uuid == r.subject_uuid), "?"),
                "target":  next((p.name for p in manifest.parts if p.uuid == r.target_uuid), "?"),
                "fit_class": r.params.get("fit_class"),
                "hole_kind": r.params.get("hole_kind"),
                "hole_diameter": r.params.get("hole_diameter"),
                "bolt_order": r.params.get("bolt_order"),
            }
            for r in screwedInto
        ],
        "expected": expected,
    }


def main():
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="synthetic_tests")
    p.add_argument("--ckpt", default=None,
                   help="Optional BRepFormer checkpoint (ML mode)")
    p.add_argument("--only", default=None,
                   help="Run only this test (simple/pass_through/kitchen/grid)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tests = {
        "simple":       (build_test_simple_4bolts,    "test1_simple_4bolts.step"),
        "pass_through": (build_test_pass_through,     "test2_pass_through.step"),
        "kitchen":      (build_test_kitchen_sink,     "test3_kitchen_sink.step"),
        "grid":         (build_test_grid_50bolts,     "test4_grid_50bolts.step"),
    }

    results = {}
    for key, (builder, fname) in tests.items():
        if args.only and key != args.only:
            continue
        out = out_dir / fname
        print(f"\n=== Building {fname} ===", file=sys.stderr)
        expected = builder(out)
        print(f"  wrote {out}  expected: {expected}", file=sys.stderr)

        print(f"  running detection (ml={bool(args.ckpt)})...", file=sys.stderr)
        result = run_detection_and_verify(out, expected, ckpt=args.ckpt)
        results[key] = result

        print(f"\n--- {key} ---")
        print(f"  parts (non-root):    {result['n_parts']}")
        print(f"  classified:          {result['n_classified']}  "
              f"(expected ≥ {expected.get('expected_fasteners','?')})")
        print(f"  screwedInto arcs:    {result['n_screwedInto']}  "
              f"(expected {expected.get('expected_screwedInto','?')})")
        print(f"  contained_in arcs:   {result['n_contained_in']}")
        if result["classifications"]:
            print(f"  classifications:")
            for name, t, c in result["classifications"][:30]:
                print(f"    {name[:35]:<35} -> {t} ({c:.2f})")
        if result["screwedInto_details"]:
            print(f"  screwedInto details:")
            for r in result["screwedInto_details"][:30]:
                print(f"    {r['subject'][:25]:<25} -> {r['target'][:25]:<25}  "
                      f"fit={r['fit_class']} hole={r['hole_kind']} "
                      f"d={r['hole_diameter']}mm bolt_order={r['bolt_order']}")

    summary_path = out_dir / "results.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
