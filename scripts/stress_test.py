"""Stress test the detection pipeline with progressively harder synthetic
STEP assemblies. Each scenario reports: part count, runtime, classification
accuracy, relationship counts, and per-fastener verdicts.
"""
from __future__ import annotations

import sys
import math
import time
from pathlib import Path
from typing import Optional

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# Reuse the synthetic builders
from scripts.generate_synthetic_tests import (
    make_plate, make_bolt, make_nut, make_washer, _write_step,
)


def make_bolt_angled(shaft_dia, shaft_length, head_dia, head_height,
                     position, axis_dir):
    """Like make_bolt but with an arbitrary axis direction."""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir, gp_Vec

    px, py, pz = position
    dx, dy, dz = axis_dir
    direction = gp_Dir(dx, dy, dz)
    shaft_axis = gp_Ax2(gp_Pnt(px, py, pz), direction)
    shaft = BRepPrimAPI_MakeCylinder(shaft_axis, shaft_dia / 2.0, shaft_length).Shape()
    head_origin = gp_Pnt(
        px + shaft_length * dx,
        py + shaft_length * dy,
        pz + shaft_length * dz,
    )
    head_axis = gp_Ax2(head_origin, direction)
    head = BRepPrimAPI_MakeCylinder(head_axis, head_dia / 2.0, head_height).Shape()
    return BRepAlgoAPI_Fuse(shaft, head).Shape()


def make_plate_angled(width, height, thickness, hole_specs, axis_dir=(0,0,1),
                      origin=(0,0,0)):
    """Plate with holes drilled along an arbitrary axis."""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir, gp_Trsf, gp_Vec
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

    plate = BRepPrimAPI_MakeBox(width, height, thickness).Shape()
    direction = gp_Dir(*axis_dir)
    for x, y, r in hole_specs:
        axis = gp_Ax2(gp_Pnt(x, y, -1.0), gp_Dir(0, 0, 1))
        hole = BRepPrimAPI_MakeCylinder(axis, r, thickness + 2.0).Shape()
        plate = BRepAlgoAPI_Cut(plate, hole).Shape()
    if origin != (0, 0, 0) or axis_dir != (0, 0, 1):
        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec(*origin))
        plate = BRepBuilderAPI_Transform(plate, trsf, True).Shape()
    return plate


# ----------------------------------------------------------------------
# Stress scenarios
# ----------------------------------------------------------------------


def stress_scale_100_bolts(out_path: Path):
    """100 bolts in a 10x10 grid, varying M-sizes (M3, M5, M6, M8, M10).

    Ground truth:
      - 100 bolts detected as fasteners
      - 100 screwedInto arcs, each to the plate, hole=through
    """
    NX = NY = 10
    PITCH = 25.0
    PLATE_W = NX * PITCH + 40
    PLATE_H = NY * PITCH + 40
    PLATE_T = 8.0

    # Cycle through M-sizes to make it variable
    size_cycle = [
        # (shaft_d, shaft_l, head_d, head_h, hole_r)
        (3.0, 18.0, 5.5, 3.0, 1.7),    # M3 clearance
        (5.0, 22.0, 8.5, 4.0, 2.75),   # M5 clearance
        (6.0, 25.0, 10.0, 4.5, 3.25),  # M6 clearance
        (8.0, 28.0, 13.0, 5.5, 4.25),  # M8 clearance
        (10.0, 30.0, 16.0, 6.5, 5.25), # M10 clearance
    ]

    centers = [(20 + i * PITCH, 20 + j * PITCH) for j in range(NY) for i in range(NX)]
    holes = []
    for k, (x, y) in enumerate(centers):
        spec = size_cycle[k % len(size_cycle)]
        holes.append((x, y, spec[4]))

    plate = make_plate(PLATE_W, PLATE_H, PLATE_T, holes)
    shapes = [(plate, f"main_plate_grid_{NX}x{NY}")]

    for k, (x, y) in enumerate(centers):
        sd, sl, hd, hh, _ = size_cycle[k % len(size_cycle)]
        bolt = make_bolt(sd, sl, hd, hh, position=(x, y, -3.0))
        shapes.append((bolt, f"bolt_{k:03d}_M{int(sd)}"))

    _write_step(shapes, out_path)
    return {
        "expected_fasteners": NX * NY,
        "expected_screwedInto": NX * NY,
        "name": f"stress_scale_{NX*NY}bolts",
    }


def stress_angled_bolts(out_path: Path):
    """16 bolts on the same plate but with their axes at different angles.
    Tests whether the axis-aware matcher works when bolts aren't z-aligned.

    Ground truth:
      - 16 bolts detected
      - 16 screwedInto arcs, each to plate
    """
    from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Ax1, gp_Dir, gp_Pnt
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

    PLATE_W = PLATE_H = 200.0
    PLATE_T = 10.0
    HOLE_R = 3.25  # M6 clearance

    centers = [(40 + (k % 4) * 40, 40 + (k // 4) * 40) for k in range(16)]
    holes = [(x, y, HOLE_R) for x, y in centers]
    plate = make_plate(PLATE_W, PLATE_H, PLATE_T, holes)
    shapes = [(plate, "plate_200x200_16_holes")]

    # Bolts at various tilts; the bolt axis is +Z initially, then rotated
    # We rotate around the hole axis intersection so the bolt still passes
    # through the hole (just at a slight angle). The matcher's axis_align
    # gate is > 0.95 (within ~18°) so tilts up to ~15° should still match.
    for k, (x, y) in enumerate(centers):
        # Build bolt vertically first
        bolt = make_bolt(6.0, 25.0, 10.0, 4.5, position=(0, 0, 0))
        # Apply a small tilt (0° to 10° depending on k) about the X axis
        # The tilt is centered at (x, y, PLATE_T/2) so the bolt still
        # roughly passes through the hole
        tilt_deg = (k % 5) * 2.0    # 0, 2, 4, 6, 8 degrees
        tilt_rad = math.radians(tilt_deg)
        trsf = gp_Trsf()
        # First translate to hole center then rotate then translate to final pos
        # Simpler: translate so bolt is at (x, y, -3), keep vertical
        # The tilt: rotate around an axis at the hole center
        if tilt_deg > 0:
            rot_axis = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0))
            trsf.SetRotation(rot_axis, tilt_rad)
            bolt = BRepBuilderAPI_Transform(bolt, trsf, True).Shape()
        # Now translate to position
        trsf2 = gp_Trsf()
        trsf2.SetTranslation(gp_Vec(x, y, -3.0))
        bolt = BRepBuilderAPI_Transform(bolt, trsf2, True).Shape()
        shapes.append((bolt, f"bolt_M6_tilt{tilt_deg}deg_{k:02d}"))

    _write_step(shapes, out_path)
    return {
        "expected_fasteners": 16,
        "expected_screwedInto_min": 12,    # tilts of 8° might fail axis_align
        "name": "stress_angled_bolts",
    }


def stress_deep_stack(out_path: Path):
    """One M8 bolt passing through 5 stacked plates with progressively
    smaller holes. Bottom plate has tap-drill hole; rest are clearance.

    Ground truth:
      - 1 fastener
      - 5 screwedInto arcs from same bolt, bolt_order 0..4
      - Last arc fit=tap, rest fit=clearance/slip
    """
    from OCC.Core.gp import gp_Trsf, gp_Vec
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

    PLATE_W = PLATE_H = 60.0
    PLATE_T = 4.0

    hole_radii = [4.5, 4.3, 4.2, 4.1, 4.0]   # progressively tighter; last is tap
    plates = []
    for i, hr in enumerate(hole_radii):
        plate = make_plate(PLATE_W, PLATE_H, PLATE_T, [(30, 30, hr)])
        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec(0, 0, i * PLATE_T))
        plate = BRepBuilderAPI_Transform(plate, trsf, True).Shape()
        plates.append((plate, f"plate_layer_{i}_holeR{hr}"))

    # Bolt goes from below plate 0 to above plate 4
    # Total stack height = 5 * 4 = 20mm. Bolt shaft 30mm, head 8mm dia, 5mm h
    bolt = make_bolt(8.0, 30.0, 14.0, 5.0, position=(30, 30, -5.0))
    shapes = plates + [(bolt, "bolt_M8x30_thru_5_plates")]

    _write_step(shapes, out_path)
    return {
        "expected_fasteners": 1,
        "expected_screwedInto": 5,
        "name": "stress_deep_stack",
    }


def stress_adversarial(out_path: Path):
    """Mix of REAL fasteners and STRUCTURAL parts that LOOK like fasteners.
    Stress-tests false positives.

    Real fasteners (should classify):
      - 4 M6 bolts
      - 4 M6 nuts
      - 4 flat washers

    Decoy structures (should NOT classify):
      - 2 long shafts (axles) — long cylinders with no head
      - 2 sleeve bearings — int + ext cylinders, tube shape
      - 2 hex prism structural blocks — hex shape but no thread
      - 1 thin plate (the host)
    """
    PLATE_W = PLATE_H = 150.0
    PLATE_T = 8.0
    HOLE_R = 3.25

    centers = [(30, 30), (60, 30), (90, 30), (120, 30)]
    holes = [(x, y, HOLE_R) for x, y in centers]
    plate = make_plate(PLATE_W, PLATE_H, PLATE_T, holes)
    shapes = [(plate, "host_plate")]

    # Real fasteners
    for k, (x, y) in enumerate(centers):
        bolt = make_bolt(6.0, 22.0, 10.0, 4.0, position=(x, y, -3.0))
        shapes.append((bolt, f"real_bolt_M6_{k}"))
        nut = make_nut(6.4, 12.0, 5.0, position=(x, y, -12.0))
        shapes.append((nut, f"real_nut_M6_{k}"))
        wash = make_washer(6.4, 12.0, 1.0, position=(x, y, -4.5))
        shapes.append((wash, f"real_washer_M6_{k}"))

    # Decoys: long shafts (no head)
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir
    for k in range(2):
        ax = gp_Ax2(gp_Pnt(30 + k * 50, 80, 20), gp_Dir(0, 0, 1))
        shaft = BRepPrimAPI_MakeCylinder(ax, 4.0, 80.0).Shape()  # 8mm dia × 80mm
        shapes.append((shaft, f"decoy_axle_{k}"))

    # Decoys: sleeve bearings (tube)
    for k in range(2):
        sleeve = make_nut(10.0, 16.0, 30.0, position=(30 + k * 50, 110, 20))
        shapes.append((sleeve, f"decoy_sleeve_bearing_{k}"))

    # Decoys: heavy structural cylindrical blocks (might look like bolt heads)
    for k in range(2):
        ax = gp_Ax2(gp_Pnt(30 + k * 50, 140, 20), gp_Dir(0, 0, 1))
        block = BRepPrimAPI_MakeCylinder(ax, 20.0, 20.0).Shape()  # 40mm dia × 20mm
        shapes.append((block, f"decoy_block_{k}"))

    _write_step(shapes, out_path)
    return {
        "expected_fasteners": 12,  # 4 bolts + 4 nuts + 4 washers
        "expected_screwedInto": 4,  # only bolts → plate
        "expected_decoys_rejected": 6,
        "decoys": ["decoy_axle_0", "decoy_axle_1",
                   "decoy_sleeve_bearing_0", "decoy_sleeve_bearing_1",
                   "decoy_block_0", "decoy_block_1"],
        "name": "stress_adversarial",
    }


# ----------------------------------------------------------------------
# Iter 2 — harder scenarios
# ----------------------------------------------------------------------


def stress_mega_500_bolts(out_path: Path):
    """500 bolts in a 25×20 grid. Stress scaling: does the matcher finish?
    Does the read+detect time stay reasonable?
    """
    NX, NY = 25, 20
    PITCH = 15.0
    PLATE_W = NX * PITCH + 30
    PLATE_H = NY * PITCH + 30
    PLATE_T = 6.0
    HOLE_R = 2.75   # M5 clearance: 5.5mm hole

    centers = [(15 + i * PITCH, 15 + j * PITCH) for j in range(NY) for i in range(NX)]
    holes = [(x, y, HOLE_R) for x, y in centers]

    plate = make_plate(PLATE_W, PLATE_H, PLATE_T, holes)
    shapes = [(plate, f"mega_plate_{NX}x{NY}")]
    for k, (x, y) in enumerate(centers):
        bolt = make_bolt(5.0, 18.0, 8.5, 3.5, position=(x, y, -3.0))
        shapes.append((bolt, f"m5_bolt_{k:04d}"))

    _write_step(shapes, out_path)
    return {
        "expected_fasteners": NX * NY,
        "expected_screwedInto": NX * NY,
        "name": f"stress_mega_{NX*NY}bolts",
    }


def stress_extreme_tilt(out_path: Path):
    """20 bolts at increasing tilt: 0°, 5°, 10°, 12°, 15°, 18°, 20°, 25°, 30°.
    Tests where the axis_align gate breaks. Gate is > 0.95, so cos⁻¹(0.95)
    = ~18.2°. Bolts ≤ 18° should match, ≥ 19° should NOT.
    """
    from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Ax1, gp_Dir, gp_Pnt
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

    PLATE_W = PLATE_H = 250.0
    PLATE_T = 12.0
    # 7mm dia hole for 6mm M6 bolt → 1mm gap (coarse clearance, within
    # the matcher's 2.0mm gate)
    HOLE_R = 3.5

    tilts = [0, 5, 10, 12, 15, 17, 18, 19, 20, 22, 25, 30]   # degrees
    centers = [(40 + (k % 4) * 60, 40 + (k // 4) * 60) for k in range(len(tilts))]

    holes = [(x, y, HOLE_R) for x, y in centers]
    plate = make_plate(PLATE_W, PLATE_H, PLATE_T, holes)
    shapes = [(plate, "tilt_test_plate")]

    for k, ((x, y), tilt_deg) in enumerate(zip(centers, tilts)):
        bolt = make_bolt(6.0, 25.0, 10.0, 4.0, position=(0, 0, 0))
        if tilt_deg > 0:
            tilt_rad = math.radians(tilt_deg)
            trsf = gp_Trsf()
            trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), tilt_rad)
            bolt = BRepBuilderAPI_Transform(bolt, trsf, True).Shape()
        # Move into position
        trsf2 = gp_Trsf()
        trsf2.SetTranslation(gp_Vec(x, y, -4.0))
        bolt = BRepBuilderAPI_Transform(bolt, trsf2, True).Shape()
        shapes.append((bolt, f"bolt_M6_tilt{tilt_deg:02d}deg_{k:02d}"))

    _write_step(shapes, out_path)
    # Bolts with tilt <= 18° should match (cos(18°) = 0.951 > 0.95)
    # Bolts with tilt >= 19° should NOT match
    return {
        "expected_fasteners": len(tilts),     # all bolts detected as fasteners
        "expected_screwedInto_max": 7,        # tilts 0/5/10/12/15/17/18 (≤18°)
        "expected_screwedInto_min": 5,        # at least 0/5/10/12/15
        "tilts": tilts,
        "name": "stress_extreme_tilt",
    }


def stress_bolt_nut_sandwich(out_path: Path):
    """The classic bolted-joint sandwich: bolt + washer + plate + washer + nut.
    8 such stacks on one big plate.

    Each stack (in z order, bolt going +Z):
      - Top of bolt head: z = -7 (head is 3mm tall + 4mm above plate)
      - Plate top: z = 0
      - Plate bottom: z = -10
      - Washer below: z = -10 to -11
      - Nut below: z = -11 to -16
      - Bolt shaft bottom: z = -25 (so shaft goes from -25 up through plate and out the top)

    Wait — let's flip it: bolt head ON TOP, shaft goes DOWN through plate, then washer, then nut on the bottom of the shaft.

    Ground truth: 8 bolts + 8 washers + 8 nuts = 24 fasteners. 8 screwedInto
    arcs (bolt → plate, one per bolt). The nut+washer don't need their own
    screwedInto since they don't have axes of their own engaging a hole.
    """
    PLATE_W = PLATE_H = 200.0
    PLATE_T = 10.0
    BOLT_SHAFT_D, BOLT_SHAFT_L = 6.0, 30.0
    BOLT_HEAD_D, BOLT_HEAD_H = 10.0, 4.0
    HOLE_R = 3.25

    centers = [(40 + (k % 4) * 40, 40 + (k // 4) * 40) for k in range(8)]
    holes = [(x, y, HOLE_R) for x, y in centers]
    plate = make_plate(PLATE_W, PLATE_H, PLATE_T, holes)
    shapes = [(plate, "sandwich_plate")]

    for k, (x, y) in enumerate(centers):
        # Bolt shaft goes from z=-15 to z=15, head from z=15 to z=19.
        # Plate occupies z=0 to z=10. So shaft passes through plate.
        bolt = make_bolt(BOLT_SHAFT_D, BOLT_SHAFT_L, BOLT_HEAD_D, BOLT_HEAD_H,
                         position=(x, y, -15.0))
        shapes.append((bolt, f"sandwich_bolt_M6_{k}"))
        # Washer below the plate: z = -3 to -4 (1mm thick washer)
        wash = make_washer(6.5, 14.0, 1.0, position=(x, y, -3.0))
        shapes.append((wash, f"sandwich_washer_M6_{k}"))
        # Nut below the washer: z = -10 to -15 (5mm tall nut)
        nut = make_nut(6.4, 12.0, 5.0, position=(x, y, -8.0))
        shapes.append((nut, f"sandwich_nut_M6_{k}"))

    _write_step(shapes, out_path)
    return {
        "expected_fasteners": 24,       # 8 bolts + 8 nuts + 8 washers
        "expected_screwedInto": 8,
        "name": "stress_bolt_nut_sandwich",
    }


# ----------------------------------------------------------------------
# Iter 3 — orthogonal axes, real SHCS, micro-fasteners, curved host
# ----------------------------------------------------------------------


def stress_orthogonal_axes(out_path: Path):
    """A rectangular box with bolts going in from BOTH the top face (+Z)
    AND the side face (+X). Tests the matcher's ability to handle multiple
    hole-axis directions in one assembly.

    Ground truth:
      - 4 bolts on top (axis +Z)
      - 3 bolts on side (axis +X)
      - 7 bolts total, 7 screwedInto arcs
    """
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir

    BOX_W, BOX_H, BOX_T = 100.0, 60.0, 30.0

    block = BRepPrimAPI_MakeBox(BOX_W, BOX_H, BOX_T).Shape()
    # 4 top holes along +Z (drill direction)
    top_holes = [(20, 30), (40, 30), (60, 30), (80, 30)]
    for x, y in top_holes:
        ax = gp_Ax2(gp_Pnt(x, y, -1), gp_Dir(0, 0, 1))
        hole = BRepPrimAPI_MakeCylinder(ax, 3.25, BOX_T + 2).Shape()
        block = BRepAlgoAPI_Cut(block, hole).Shape()
    # 3 side holes along +X (going from x=0 face into the block)
    side_holes_yz = [(15, 15), (15, 22), (45, 15)]
    for y, z in side_holes_yz:
        ax = gp_Ax2(gp_Pnt(-1, y, z), gp_Dir(1, 0, 0))
        hole = BRepPrimAPI_MakeCylinder(ax, 2.75, BOX_W + 2).Shape()
        block = BRepAlgoAPI_Cut(block, hole).Shape()

    shapes = [(block, "block_with_holes_2_directions")]

    # Top bolts: M6, going +Z
    for k, (x, y) in enumerate(top_holes):
        bolt = make_bolt(6.0, 25.0, 10.0, 4.0, position=(x, y, -5.0))
        shapes.append((bolt, f"top_bolt_M6_{k}"))

    # Side bolts: M5 along +X. Use the angled bolt helper.
    for k, (y, z) in enumerate(side_holes_yz):
        bolt = make_bolt_angled(5.0, 25.0, 8.5, 3.5,
                                position=(-5.0, y, z),
                                axis_dir=(1.0, 0.0, 0.0))
        shapes.append((bolt, f"side_bolt_M5_{k}"))

    _write_step(shapes, out_path)
    return {
        "expected_fasteners": 7,
        "expected_screwedInto": 7,
        "name": "stress_orthogonal_axes",
    }


def stress_real_shcs(out_path: Path):
    """6 bolts that are GEOMETRICALLY realistic socket head cap screws —
    cylindrical head with internal hex-socket recess (modeled as a cylinder).
    This tests the SHCS signature path which expects n_int >= 1 in the bolt.
    """
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir

    PLATE_W = PLATE_H = 100.0
    PLATE_T = 8.0
    HOLE_R = 3.25

    centers = [(20 + (k % 3) * 30, 20 + (k // 3) * 30) for k in range(6)]
    holes = [(x, y, HOLE_R) for x, y in centers]
    plate = make_plate(PLATE_W, PLATE_H, PLATE_T, holes)
    shapes = [(plate, "shcs_test_plate")]

    for k, (x, y) in enumerate(centers):
        # Build a real SHCS:
        # - shaft (cylindrical, M6 = 6mm dia, 20mm long)
        # - head (cylindrical, 10mm dia, 4mm tall)
        # - internal hex socket recess in the head (cylinder, 4mm dia × 3mm deep)
        bolt = make_bolt(6.0, 20.0, 10.0, 4.0, position=(x, y, -3.0))
        # Subtract the socket recess from the top of the head
        head_top_z = -3.0 + 20.0 + 4.0   # = 21.0
        recess_axis = gp_Ax2(gp_Pnt(x, y, head_top_z - 3.0), gp_Dir(0, 0, 1))
        recess = BRepPrimAPI_MakeCylinder(recess_axis, 2.0, 4.0).Shape()
        bolt = BRepAlgoAPI_Cut(bolt, recess).Shape()
        shapes.append((bolt, f"real_shcs_M6_{k}"))

    _write_step(shapes, out_path)
    return {
        "expected_fasteners": 6,
        "expected_screwedInto": 6,
        "name": "stress_real_shcs",
    }


def stress_micro_fasteners(out_path: Path):
    """8 micro fasteners: M1.6 and M2 sizes — the smallest end of common ISO.
    Tests precision at sub-2mm shaft diameters where rounding errors and OCC
    tolerances might matter.

    Ground truth:
      - 8 bolts (4 M1.6 + 4 M2)
      - 8 screwedInto arcs
    """
    PLATE_W = PLATE_H = 60.0
    PLATE_T = 3.0

    sizes = [
        # (label, shaft_d, shaft_l, head_d, head_h, hole_r)
        ("M1.6", 1.6, 8.0, 3.0, 1.5, 0.9),   # 1.8mm hole, 1.6 shaft = 0.2mm
        ("M2",   2.0, 10.0, 3.8, 1.8, 1.1),  # 2.2mm hole, 2 shaft = 0.2mm
    ]
    centers = [(10 + (k % 4) * 12, 15 + (k // 4) * 30) for k in range(8)]

    holes = []
    for k, (x, y) in enumerate(centers):
        spec = sizes[k % 2]
        holes.append((x, y, spec[5]))
    plate = make_plate(PLATE_W, PLATE_H, PLATE_T, holes)
    shapes = [(plate, "micro_plate_60x60x3")]

    for k, (x, y) in enumerate(centers):
        label, sd, sl, hd, hh, _ = sizes[k % 2]
        bolt = make_bolt(sd, sl, hd, hh, position=(x, y, -2.0))
        shapes.append((bolt, f"micro_bolt_{label}_{k}"))

    _write_step(shapes, out_path)
    return {
        "expected_fasteners": 8,
        "expected_screwedInto": 8,
        "name": "stress_micro_fasteners",
    }


def stress_curved_host(out_path: Path):
    """4 bolts threaded radially into a cylindrical tube/hub (host is curved,
    not flat). Real automotive/mechanical scenario: a wheel hub with lug
    bolts. The "hole" in a cylindrical host is a radial cylindrical bore.
    """
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir

    # Hub: outer cylinder Ø50mm × 40mm tall, axial bore Ø20mm
    hub_axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
    hub = BRepPrimAPI_MakeCylinder(hub_axis, 25.0, 40.0).Shape()
    bore_axis = gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1))
    bore = BRepPrimAPI_MakeCylinder(bore_axis, 10.0, 42.0).Shape()
    hub = BRepAlgoAPI_Cut(hub, bore).Shape()

    # Drill 4 radial holes at z=20, 90° apart, going inward
    n_radial = 4
    for k in range(n_radial):
        angle = math.radians(90.0 * k)
        # Position the drill axis to start from outside the hub
        cx = math.cos(angle) * 30.0  # start at radius 30 (outside the OD 25)
        cy = math.sin(angle) * 30.0
        # Direction points toward center (inward)
        dx = -math.cos(angle)
        dy = -math.sin(angle)
        # Drill 12mm deep (just past the outer wall, not all the way through)
        ax = gp_Ax2(gp_Pnt(cx, cy, 20.0), gp_Dir(dx, dy, 0))
        hole = BRepPrimAPI_MakeCylinder(ax, 2.75, 15.0).Shape()
        hub = BRepAlgoAPI_Cut(hub, hole).Shape()

    shapes = [(hub, "cylindrical_hub")]

    # 4 bolts coming in radially. Bolt shaft points from outside toward center.
    for k in range(n_radial):
        angle = math.radians(90.0 * k)
        # Bolt origin = at radius 32 (outside the hub OD), pointing inward
        bx = math.cos(angle) * 32.0
        by = math.sin(angle) * 32.0
        dx = -math.cos(angle)
        dy = -math.sin(angle)
        bolt = make_bolt_angled(5.0, 15.0, 8.5, 3.5,
                                position=(bx, by, 20.0),
                                axis_dir=(dx, dy, 0.0))
        shapes.append((bolt, f"radial_bolt_M5_{k}"))

    _write_step(shapes, out_path)
    return {
        "expected_fasteners": 4,
        "expected_screwedInto": 4,
        "name": "stress_curved_host",
    }


def stress_huge_1000_bolts(out_path: Path):
    """1000 bolts in a 40×25 grid. Final scale stress test — verifies the
    O(F×H) matcher doesn't blow up quadratically (1000² = 1M score ops).
    """
    NX, NY = 40, 25
    PITCH = 12.0
    PLATE_W = NX * PITCH + 30
    PLATE_H = NY * PITCH + 30
    PLATE_T = 5.0
    HOLE_R = 2.25   # M4 clearance: 4.5mm hole

    centers = [(15 + i * PITCH, 15 + j * PITCH) for j in range(NY) for i in range(NX)]
    holes = [(x, y, HOLE_R) for x, y in centers]

    plate = make_plate(PLATE_W, PLATE_H, PLATE_T, holes)
    shapes = [(plate, f"huge_plate_{NX}x{NY}")]
    for k, (x, y) in enumerate(centers):
        bolt = make_bolt(4.0, 14.0, 7.0, 3.0, position=(x, y, -3.0))
        shapes.append((bolt, f"m4_bolt_{k:04d}"))

    _write_step(shapes, out_path)
    return {
        "expected_fasteners": NX * NY,
        "expected_screwedInto": NX * NY,
        "name": f"stress_huge_{NX*NY}bolts",
    }


# ----------------------------------------------------------------------
# Run + verify
# ----------------------------------------------------------------------


def run_detection(step_path: Path, ckpt: Optional[str] = None) -> dict:
    from step_vr_step.readers.step_reader import read_step
    from step_vr_step.detection import detect_fasteners
    from step_vr_step.config import DetectionConfig

    t_read = time.perf_counter()
    doc, manifest, shapes = read_step(str(step_path), return_shapes=True)
    t_read = time.perf_counter() - t_read

    t_detect = time.perf_counter()
    cfg = DetectionConfig(
        enable_ml=bool(ckpt),
        brepformer_weights=ckpt,
        rule_confidence_threshold=0.35,
    )
    manifest = detect_fasteners(manifest, shapes=shapes, config=cfg)
    t_detect = time.perf_counter() - t_detect

    classified = [
        p for p in manifest.parts
        if p.detection and "unclassified" not in p.detection.fastener_type
    ]
    screwedInto = [r for r in manifest.relationships if r.kind == "fastener"]
    contained_in = [r for r in manifest.relationships if r.kind == "contained_in"]

    return {
        "n_parts": len([p for p in manifest.parts if p.parent_uuid]),
        "n_classified": len(classified),
        "n_screwedInto": len(screwedInto),
        "n_contained_in": len(contained_in),
        "t_read_s": round(t_read, 2),
        "t_detect_s": round(t_detect, 2),
        "classifications": [(p.name, p.detection.fastener_type, round(p.detection.confidence, 2)) for p in classified],
        "screwedInto": [
            {
                "subject": next((p.name for p in manifest.parts if p.uuid == r.subject_uuid), "?"),
                "target": next((p.name for p in manifest.parts if p.uuid == r.target_uuid), "?"),
                "d": r.params.get("hole_diameter"),
                "fit": r.params.get("fit_class"),
                "kind": r.params.get("hole_kind"),
                "bolt_order": r.params.get("bolt_order"),
            }
            for r in screwedInto
        ],
    }


def main():
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="stress_tests")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--only", default=None,
                   help="scale / angled / deep_stack / adversarial")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "scale":       (stress_scale_100_bolts, "stress1_scale_100bolts.step"),
        "angled":      (stress_angled_bolts,    "stress2_angled.step"),
        "deep_stack":  (stress_deep_stack,      "stress3_deep_stack.step"),
        "adversarial": (stress_adversarial,     "stress4_adversarial.step"),
        # Iter 2 — harder
        "mega":        (stress_mega_500_bolts,  "stress5_mega_500bolts.step"),
        "extreme_tilt":(stress_extreme_tilt,    "stress6_extreme_tilt.step"),
        "sandwich":    (stress_bolt_nut_sandwich, "stress7_sandwich.step"),
        # Iter 3 — orthogonal axes, real SHCS topology, micro-fasteners,
        # curved host
        "orthogonal":  (stress_orthogonal_axes, "stress8_orthogonal.step"),
        "real_shcs":   (stress_real_shcs,        "stress9_real_shcs.step"),
        "micro":       (stress_micro_fasteners,  "stress10_micro.step"),
        "curved_host": (stress_curved_host,      "stress11_curved.step"),
        # Iter 4 — scale test
        "huge":        (stress_huge_1000_bolts,  "stress12_huge_1000.step"),
    }

    results = {}
    for key, (builder, fname) in scenarios.items():
        if args.only and key != args.only:
            continue
        out = out_dir / fname

        print(f"\n{'='*70}", file=sys.stderr)
        print(f"=== STRESS: {key} ===", file=sys.stderr)
        print('=' * 70, file=sys.stderr)

        t_build = time.perf_counter()
        expected = builder(out)
        t_build = time.perf_counter() - t_build
        print(f"  build: {t_build:.2f}s  expected: {expected}", file=sys.stderr)

        result = run_detection(out, ckpt=args.ckpt)
        result["expected"] = expected
        result["t_build_s"] = round(t_build, 2)
        results[key] = result

        print(f"\n  parts={result['n_parts']}  classified={result['n_classified']}  "
              f"screwedInto={result['n_screwedInto']}  contained_in={result['n_contained_in']}")
        print(f"  read={result['t_read_s']}s  detect={result['t_detect_s']}s  "
              f"build={result['t_build_s']}s")

        # Print up to 10 sample classifications
        if result["classifications"]:
            print(f"  classifications (sample of {len(result['classifications'])}):")
            for name, t, c in result["classifications"][:8]:
                print(f"    {name[:40]:<40} -> {t} ({c})")
            if len(result["classifications"]) > 8:
                print(f"    ... +{len(result['classifications']) - 8} more")

        # Print up to 10 sample screwedInto arcs
        if result["screwedInto"]:
            print(f"  screwedInto (sample of {len(result['screwedInto'])}):")
            for r in result["screwedInto"][:8]:
                print(f"    {r['subject'][:30]:<30} -> {r['target'][:25]:<25}  "
                      f"d={r['d']} fit={r['fit']} kind={r['kind']} order={r['bolt_order']}")
            if len(result["screwedInto"]) > 8:
                print(f"    ... +{len(result['screwedInto']) - 8} more")

        # For adversarial: check decoys
        if "decoys" in expected:
            decoys_classified = [
                name for name, t, c in result["classifications"]
                if any(d in name for d in expected["decoys"])
            ]
            print(f"  decoys CORRECTLY rejected: {len(expected['decoys']) - len(decoys_classified)}/{len(expected['decoys'])}")
            if decoys_classified:
                print(f"  decoys MIS-CLASSIFIED: {decoys_classified}")

    with (out_dir / "stress_results.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {out_dir}/stress_results.json")


if __name__ == "__main__":
    main()
