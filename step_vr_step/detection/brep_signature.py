"""Pure-BRep classifier that decides *what kind of fastener* a part is from
its topology signature, before any ISO size matching.

The original rule_based.py only ranks parts against ISO dimension tables and
ignores the strongest BRep signal we already extract — namely the ratio of
internal vs external cylindrical faces (a bolt's shaft is external, a nut's
threaded bore is internal). That's why the test fixtures show bolts
labelled as "thin_hex_nut" and nuts as "unclassified". This module fixes
that by deciding the *type* first, with high-confidence gates, and lets the
ISO scoring contribute the *variant/size* on top.

The return value is (fastener_type, signature_confidence, evidence_dict).
A signature_confidence of 0 means "no strong opinion — fall back to ISO
scoring as before."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometric_features import CylindricalFeature, GeometricFeatures


@dataclass
class Signature:
    fastener_type: str
    confidence: float
    shaft_diameter: float
    head_diameter: float
    length: float
    evidence: str


def _disc_aspect(feat: GeometricFeatures) -> float:
    """Aspect ratio robust to disc-shaped parts.

    GeometricFeatures.aspect_ratio is computed from a sorted bounding-box:
    largest dim / mean(two smaller dims). For a flat washer (24×24×2.5)
    that gives 24 / 13.25 = 1.81 — useless. This helper computes
    min(bbox) / max(bbox), which gives 2.5/24 = 0.10 for the washer and
    still gives 0.10 for a long thin bolt (5/50), so it cleanly
    distinguishes flat vs. long. Returns 1.0 for cube-like parts.
    """
    bbox_dims = sorted([feat.bbox_max[i] - feat.bbox_min[i] for i in range(3)])
    if bbox_dims[2] < 1e-9:
        return 1.0
    return bbox_dims[0] / bbox_dims[2]


def _is_disc(feat: GeometricFeatures, ratio: float = 0.25) -> bool:
    """A disc has one bbox dim much smaller than the other two (which are
    roughly equal): washer, gasket, spacer-disc, large flat ring.
    """
    dims = sorted([feat.bbox_max[i] - feat.bbox_min[i] for i in range(3)])
    if dims[2] < 1e-9 or dims[1] < 1e-9:
        return False
    flatness = dims[0] / dims[2]
    squareness = dims[1] / dims[2]
    return flatness <= ratio and squareness >= 0.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cluster_radii(radii: list[float], rel_tol: float = 0.05) -> list[tuple[float, int]]:
    """Group radii into clusters whose relative spread is within `rel_tol`.
    Returns (median_radius, count) per cluster, sorted by radius ascending.
    """
    if not radii:
        return []
    sorted_r = sorted(radii)
    clusters: list[list[float]] = [[sorted_r[0]]]
    for r in sorted_r[1:]:
        ref = clusters[-1][0]
        if (r - ref) / max(ref, 1e-9) <= rel_tol:
            clusters[-1].append(r)
        else:
            clusters.append([r])
    return [(float(np.median(c)), len(c)) for c in clusters]


def _external_radii(feat: GeometricFeatures) -> list[float]:
    return [c.radius for c in feat.cylinders if not c.is_internal]


def _internal_radii(feat: GeometricFeatures) -> list[float]:
    return [c.radius for c in feat.cylinders if c.is_internal]


def _max_internal_radius(feat: GeometricFeatures) -> float:
    rs = _internal_radii(feat)
    return max(rs) if rs else 0.0


def _has_head_profile(feat: GeometricFeatures) -> bool:
    """A bolt/screw head shows up as cone, torus, sphere, OR multiple
    radially distinct external cylinders, OR a populated head_diameter
    (set by the original head heuristic in extract_brep_features)."""
    n_cone = feat.face_type_counts.get("cone", 0)
    n_torus = feat.face_type_counts.get("torus", 0)
    n_sphere = feat.face_type_counts.get("sphere", 0)
    if n_cone + n_torus + n_sphere >= 1:
        return True
    if feat.head_diameter is not None and feat.head_diameter > 0:
        return True
    ext_clusters = _cluster_radii(_external_radii(feat))
    if len(ext_clusters) >= 2:
        small = ext_clusters[0][0]
        big = ext_clusters[-1][0]
        if big >= 1.3 * small:
            return True
    return False


# ---------------------------------------------------------------------------
# Main signature classifier
# ---------------------------------------------------------------------------

def classify_by_signature(feat: GeometricFeatures) -> Optional[Signature]:
    """Return a Signature if the part has a strong BRep signature; None
    otherwise (so the caller can fall back to plain ISO scoring).

    Confidence is set to make sense after ISO size matching: this is the
    *type* confidence. The caller multiplies by the size-match score to
    get the final score.
    """
    n_ext = sum(1 for c in feat.cylinders if not c.is_internal)
    n_int = sum(1 for c in feat.cylinders if c.is_internal)
    ext_clusters = _cluster_radii(_external_radii(feat))
    int_clusters = _cluster_radii(_internal_radii(feat))
    n_plane = feat.face_type_counts.get("plane", 0)
    n_cone = feat.face_type_counts.get("cone", 0)
    n_torus = feat.face_type_counts.get("torus", 0)
    n_sphere = feat.face_type_counts.get("sphere", 0)
    aspect_bbox = feat.aspect_ratio          # length/diameter from sorted bbox
    aspect_flat = _disc_aspect(feat)          # min/max bbox — robust for discs
    has_head = _has_head_profile(feat)

    # Hard global size guards — real fasteners are unlikely to exceed these.
    #   - Length > 500mm: rules out structural shafts, rods, brake arms.
    #     (allows M30+ bolts up to 300mm long, threaded rods up to 500mm.)
    #   - Volume > 500 cm³: rules out structural housings/levers/axles.
    #   - Shaft cylinder radius > 30mm: rules out M60+ (very uncommon).
    bbox_dims = [feat.bbox_max[i] - feat.bbox_min[i] for i in range(3)]
    if max(bbox_dims) > 500.0:
        return None
    if feat.volume > 500_000.0:
        return None
    if _external_radii(feat) and min(_external_radii(feat)) > 30.0:
        return None

    # ------------- FLAT WASHER (including circlips / spring washers) -----
    # Disc shape with concentric internal+external cylinders. Strict
    # face-count cap (most washers have ≤20 faces); chamfered washers
    # with up to ~60 faces still OK. Require explicit disc shape AND a
    # planar-area majority so we don't catch o-rings / gaskets that look
    # disc-ish but have torus profiles.
    n_plane_local = feat.face_type_counts.get("plane", 0)
    if (
        _is_disc(feat, ratio=0.4)
        and n_int >= 1 and n_ext >= 1
        and n_plane >= 2
        and feat.num_faces <= 60
        and n_plane_local >= 2
        and n_torus <= 4    # rules out o-rings / gaskets
    ):
        bore = min(_internal_radii(feat)) * 2.0
        outer = max(_external_radii(feat)) * 2.0
        if outer > bore * 1.15:
            return Signature(
                fastener_type="flat_washer",
                confidence=0.90,
                shaft_diameter=bore,
                head_diameter=outer,
                length=feat.bounding_cylinder_length,
                evidence=f"disc + int/ext concentric, flat={aspect_flat:.2f}",
            )

    # ------------- HEX NUT -----------------
    # Short and wide, internal threaded bore, hex flats. Hex nut has 6
    # planar side faces + 2 top/bottom = 8 planes total (or 14-16 with
    # chamfers). Use aspect_flat so the test works for short-but-square nuts.
    # The bore = median of internal radii (most stable; chamfers can produce
    # smaller or larger internal cylinders).
    if (
        0.2 <= aspect_bbox <= 1.8
        and n_int >= 1
        and n_plane >= 6
        and _max_internal_radius(feat) > 0
    ):
        int_r = _internal_radii(feat)
        bore_r = float(np.median(int_r))
        # Make sure no external cylinder is the dominant feature — if there
        # is one, this looks more like a bolt with a hex head.
        ext_dominant = any(r > bore_r * 1.2 for r in _external_radii(feat))
        if not ext_dominant or aspect_bbox < 0.8:
            bore = bore_r * 2.0
            # Across-flats = the largest bbox horizontal dim
            dims = sorted([feat.bbox_max[i] - feat.bbox_min[i] for i in range(3)])
            outer = max(dims[1], bore * 1.5)
            return Signature(
                fastener_type="hex_nut",
                confidence=0.90,
                shaft_diameter=bore,
                head_diameter=outer,
                length=feat.bounding_cylinder_length,
                evidence=f"aspect={aspect_bbox:.2f}, {n_int} int cyl, {n_plane} planes",
            )

    # ------------- SOCKET HEAD CAP SCREW -----------------
    # External shaft + larger external head + internal hex drive recess.
    # Shaft = smallest external cluster, head = largest, drive = internal.
    if (
        1.0 <= aspect_bbox <= 15.0
        and n_ext >= 2
        and n_int >= 1
        and len(ext_clusters) >= 2
    ):
        shaft_r = ext_clusters[0][0]
        head_r = ext_clusters[-1][0]
        drive_r = _max_internal_radius(feat)
        if (
            head_r >= 1.3 * shaft_r        # head wider than shaft
            and drive_r > 0
            and drive_r < shaft_r * 1.1     # drive smaller than shaft
            and shaft_r <= 30.0             # not a structural shaft
        ):
            return Signature(
                fastener_type="socket_head_cap_screw",
                confidence=0.88,
                shaft_diameter=shaft_r * 2.0,
                head_diameter=head_r * 2.0,
                length=feat.bounding_cylinder_length,
                evidence=(
                    f"shaft r={shaft_r:.1f}, head r={head_r:.1f}, "
                    f"drive r={drive_r:.1f}, aspect={aspect_bbox:.2f}"
                ),
            )

    # ------------- HEX BOLT -----------------
    # External shaft + head profile (≥6 hex flats OR torus/cone head).
    # Shaft = most-common (dominant) external radius cluster, not min:
    # threaded bolts have thread tip cylinders at smaller radii.
    if (
        1.2 <= aspect_bbox <= 15.0
        and n_ext >= 1
        and has_head
        and n_plane >= 6
        and (
            n_int == 0
            or _max_internal_radius(feat) < min(_external_radii(feat)) * 0.6
        )
    ):
        # Pick the cluster with the highest count of cylinder faces — that
        # corresponds to the bolt body (thread peaks repeat many times).
        dominant = max(ext_clusters, key=lambda c: (c[1], c[0]))
        shaft_r = dominant[0]
        head_r = max(r for r, _ in ext_clusters)
        if head_r <= shaft_r * 1.1:
            head_r = feat.bounding_cylinder_diameter / 2.0
        if shaft_r > 30.0:        # bigger than M30 — structural, not a fastener
            return None
        return Signature(
            fastener_type="hex_bolt",
            confidence=0.85,
            shaft_diameter=shaft_r * 2.0,
            head_diameter=head_r * 2.0,
            length=feat.bounding_cylinder_length,
            evidence=(
                f"aspect={aspect_bbox:.2f}, head + {n_plane} planes, "
                f"dominant shaft r={shaft_r:.1f} ({dominant[1]} faces)"
            ),
        )

    # ------------- WOOD / SELF-TAPPING SCREW -----------------
    # Must be LONG and slender (real wood screws have aspect_bbox ≥ 5
    # and bbox_flat ≤ 0.2 — narrow stick). Cone tip + many cyl + drive.
    if (
        aspect_bbox >= 5.0
        and aspect_flat <= 0.25
        and n_cone >= 1
        and n_ext >= 3
        and n_int >= 1
    ):
        shaft_r = float(np.median(_external_radii(feat)))
        if shaft_r > 15.0:    # > M30 wood screw is unrealistic
            return None
        head_r = max(_external_radii(feat))
        return Signature(
            fastener_type="wood_screw",
            confidence=0.75,
            shaft_diameter=shaft_r * 2.0,
            head_diameter=head_r * 2.0,
            length=feat.bounding_cylinder_length,
            evidence=f"slender + cone tip + drive recess",
        )

    # ------------- STUD (threaded rod, possibly with end flats) -----
    # Stud bolts can have hex flats at one end for installation, so allow
    # up to ~10 planar faces (one hex = 8 with two end caps).
    if (
        aspect_bbox >= 3.0
        and aspect_flat <= 0.35
        and n_ext >= 1
        and not has_head
        and n_plane <= 10
        and (
            n_int == 0
            or _max_internal_radius(feat) < min(_external_radii(feat) or [1]) * 0.5
        )
    ):
        shaft_r = float(np.median(_external_radii(feat)))
        if shaft_r > 30.0:
            return None
        return Signature(
            fastener_type="threaded_stud",
            confidence=0.80,
            shaft_diameter=shaft_r * 2.0,
            head_diameter=shaft_r * 2.0,
            length=feat.bounding_cylinder_length,
            evidence=f"long+slender, no head, {n_plane} planes",
        )

    # ------------- DOWEL PIN / CLEVIS PIN -----------------
    # Plain cylindrical pin: external-only, optionally with a chamfer cone
    # at each end. Either 1 cluster (uniform pin) or 2 clusters (shouldered
    # / clevis pin). No internal cylinders (or very small for cross-hole).
    # Real dowel pins have L/D ≤ 10 (per DIN 7 / ISO 2338); longer cylinders
    # without heads are typically shafts/axles/rods, not fasteners.
    if (
        0.4 <= aspect_bbox <= 10.0       # was 30.0 — long shafts excluded
        and aspect_flat <= 0.5
        and n_ext >= 1
        and not has_head
        and n_int == 0
        and n_plane <= 4
    ):
        ext_r = _external_radii(feat)
        shaft_r = float(np.median(ext_r))
        # Stock dowel pins go up to ~M16 (8mm radius); larger is structural.
        if 0.4 <= shaft_r <= 10.0:
            return Signature(
                fastener_type="dowel_pin",
                confidence=0.78,
                shaft_diameter=shaft_r * 2.0,
                head_diameter=shaft_r * 2.0,
                length=feat.bounding_cylinder_length,
                evidence=f"plain cyl pin aspect={aspect_bbox:.2f} r={shaft_r:.1f}",
            )

    # Shouldered dowel / clevis pin: 2 ext clusters, no internal, no head.
    if (
        0.4 <= aspect_bbox <= 30.0
        and aspect_flat <= 0.5
        and n_ext >= 2
        and len(ext_clusters) >= 2
        and n_int == 0
        and n_plane <= 6
    ):
        shaft_r = ext_clusters[0][0]
        head_r = ext_clusters[-1][0]
        if (0.4 <= shaft_r <= 30.0
                and 1.1 <= head_r / shaft_r <= 3.0):
            return Signature(
                fastener_type="dowel_pin",
                confidence=0.78,
                shaft_diameter=shaft_r * 2.0,
                head_diameter=head_r * 2.0,
                length=feat.bounding_cylinder_length,
                evidence=f"shouldered pin r={shaft_r:.1f}/{head_r:.1f}",
            )

    # ------------- SQUARE NUT / T-NUT -----------------
    # Cube-ish bbox, an internal threaded bore, 4-5 planar faces (sides +
    # top/bottom of a square body, fewer than hex's 6). Distinct from
    # hex_nut which requires 6+ planar faces.
    if (
        0.5 <= aspect_bbox <= 2.5
        and 0.4 <= aspect_flat <= 1.0           # not flat, not very long
        and n_int >= 1
        and 4 <= n_plane <= 6
        and _max_internal_radius(feat) > 0
    ):
        int_r = _internal_radii(feat)
        bore_r = float(np.median(int_r))
        if 1.0 <= bore_r <= 20.0:
            dims = sorted([feat.bbox_max[i] - feat.bbox_min[i] for i in range(3)])
            return Signature(
                fastener_type="square_nut",
                confidence=0.82,
                shaft_diameter=bore_r * 2.0,
                head_diameter=dims[1],
                length=feat.bounding_cylinder_length,
                evidence=f"cube-shape + int bore, planes={n_plane}",
            )

    # ------------- THREADED INSERT (Helicoil-style) -----------------
    # Real threaded inserts have a distinctive geometry:
    #   - SHORT aspect (0.7-2.2)
    #   - Knurled external = many ext cylinders all at the SAME radius
    #     (one tight cluster), so len(ext_clusters)==1
    #   - Clean internal threaded bore = one dominant internal cluster
    #   - Chamfered ends (cones)
    #   - At least 6 external thread cylinders (knurl pattern)
    if (
        0.7 <= aspect_bbox <= 2.2
        and aspect_flat <= 0.7
        and n_int >= 3
        and n_ext >= 6
        and len(ext_clusters) == 1                # uniform knurl
        and len(int_clusters) <= 3                # mostly one internal bore
        and _max_internal_radius(feat) > 0
        and n_cone >= 2                           # chamfers on both ends
        and n_plane >= 2
    ):
        int_r = _internal_radii(feat)
        bore_r = float(np.median(int_r))
        knurl_r = ext_clusters[0][0]
        # Knurl radius should be reasonably larger than bore (wall thickness).
        if 1.5 <= bore_r <= 15.0 and knurl_r > bore_r * 1.15:
            return Signature(
                fastener_type="threaded_insert",
                confidence=0.78,
                shaft_diameter=bore_r * 2.0,
                head_diameter=knurl_r * 2.0,
                length=feat.bounding_cylinder_length,
                evidence=f"uniform knurl + int bore + 2 chamfers",
            )

    # ------------- SET SCREW / GRUB SCREW -----------------
    if (
        0.5 <= aspect_bbox <= 4.0
        and aspect_flat <= 0.4              # not a disc (excludes washers)
        and n_ext >= 1
        and not has_head
        and n_int >= 1
        and _max_internal_radius(feat) > 0
        and _max_internal_radius(feat) < min(_external_radii(feat) or [1]) * 0.9
    ):
        shaft_r = float(np.median(_external_radii(feat)))
        if shaft_r > 30.0:
            return None
        return Signature(
            fastener_type="set_screw",
            confidence=0.75,
            shaft_diameter=shaft_r * 2.0,
            head_diameter=shaft_r * 2.0,
            length=feat.bounding_cylinder_length,
            evidence=f"no head + internal drive, aspect={aspect_bbox:.2f}",
        )

    return None
