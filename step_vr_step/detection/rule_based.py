"""Rule-based fastener classification using ISO dimension tables.

Pipeline:
  1. BRep signature classifier (brep_signature.py) decides what *kind* of
     fastener the part is, using internal/external cylinder topology +
     face-type histogram. This catches bolt-vs-nut role-flips that the
     ISO scorer misses, and adds classes the ISO tables don't cover
     (set_screw, threaded_stud, wood_screw).
  2. ISO scoring picks the *variant/size* by matching against tables.

If the signature classifier returns a strong opinion, we restrict the ISO
scoring to tables of that type only, so the size match doesn't drift to a
neighbouring type (e.g. a hex_bolt being given a hex_nut variant label).
"""

import logging
from typing import Optional

from ..schema import DetectionLabel
from ..config import DetectionConfig
from .geometric_features import GeometricFeatures
from .iso_tables import (
    ALL_STANDARDS, BOLT_TYPES, NUT_TYPES, WASHER_TYPES,
)
from .brep_signature import classify_by_signature, Signature

logger = logging.getLogger(__name__)


# Map signature type → set of fastener_type values in iso_tables that
# represent that class. Lets us filter the ISO scoring to matching tables.
# For types that have no direct ISO equivalent (dowel_pin, square_nut,
# threaded_insert), use the closest family and let the signature confidence
# drive the final score.
_TYPE_FAMILIES: dict[str, set[str]] = {
    "hex_bolt":     set(BOLT_TYPES),
    "socket_head_cap_screw": set(BOLT_TYPES),
    "hex_nut":      set(NUT_TYPES),
    "flat_washer":  set(WASHER_TYPES),
    "threaded_stud": set(BOLT_TYPES),
    "set_screw":    set(BOLT_TYPES),
    "wood_screw":   set(BOLT_TYPES),
    "dowel_pin":    set(BOLT_TYPES),     # no real ISO match; sig confidence drives
    "square_nut":   set(NUT_TYPES),
    "threaded_insert": set(NUT_TYPES),
}


def classify_part(
    features: GeometricFeatures,
    config: DetectionConfig,
    repetition_count: int = 1,
) -> DetectionLabel:
    """Classify a single part against ISO fastener dimension tables.

    Returns a DetectionLabel with confidence thresholds:
        >0.90  → assign type directly
        0.60–0.90 → "possible_{type}"
        <0.60  → "unclassified"
    """
    # Quick rejection: not cylindrical enough. Threshold loosened from
    # 0.25 → 0.15 because spring washers, snap rings, and some stud bolts
    # have lots of planar/torus/cone area and a relatively small
    # cylindrical share but are still valid fasteners.
    if features.cylindrical_surface_ratio < 0.15:
        return DetectionLabel(fastener_type="unclassified", confidence=0.0)

    # Pre-filter: need cylindrical features AND reasonable aspect ratio or torus
    has_torus = features.face_type_counts.get("torus", 0) > 0
    if features.aspect_ratio < 0.02 and not has_torus:
        return DetectionLabel(fastener_type="unclassified", confidence=0.0)

    # Step A: BRep signature — what kind of fastener is this?
    sig = classify_by_signature(features)

    # Step B: ISO scoring restricted to the signature's type family
    # (or all tables if no signature).
    allowed_types = _TYPE_FAMILIES.get(sig.fastener_type) if sig else None

    best_label = DetectionLabel(fastener_type="unclassified", confidence=0.0)
    best_score = 0.0

    for std_name, std_info in ALL_STANDARDS.items():
        fastener_type = std_info["type"]
        if allowed_types is not None and fastener_type not in allowed_types:
            continue
        table = std_info["table"]
        for variant_name, dims in table.items():
            score = _score_match(features, fastener_type, dims, config, sig)
            if score > best_score:
                best_score = score
                detected_dims = _collect_detected_dims(features)
                best_label = DetectionLabel(
                    fastener_type=fastener_type,
                    standard=std_name,
                    variant=variant_name,
                    confidence=score,
                    method="rule_based",
                    detected_dimensions=detected_dims,
                )

    # If the signature is confident but no ISO entry scored well, still
    # return the signature's type — we just won't have an ISO variant.
    if sig and sig.confidence >= 0.75 and best_score < 0.40:
        best_label = DetectionLabel(
            fastener_type=sig.fastener_type,
            standard=None,
            variant=None,
            confidence=sig.confidence * 0.85,  # discount: no size match
            method="rule_based",
            detected_dimensions=_collect_detected_dims(features),
        )
        best_score = best_label.confidence
    elif sig and best_score >= 0.40:
        # Signature found a type AND ISO confirmed a size — combine.
        best_label.fastener_type = sig.fastener_type
        best_label.confidence = min(1.0, 0.4 * sig.confidence + 0.6 * best_score + 0.2)
    elif sig is None:
        # No signature: be strict about what we accept from ISO-only
        # scoring. Reject:
        #  - any part > 250mm or > 200000 mm³ (structural, not a fastener)
        #  - a hex_bolt label that lacks strict head evidence: a real bolt
        #    has cone/torus/sphere faces OR two distinct ext cylinder
        #    clusters (shaft + head). A plain rod/rope/axle has neither.
        bbox_max = max(features.bbox_max[i] - features.bbox_min[i] for i in range(3))
        if bbox_max > 500.0 or features.volume > 500_000.0:
            best_label = DetectionLabel(fastener_type="unclassified", confidence=0.0)
            best_score = 0.0
        elif best_label.fastener_type in BOLT_TYPES:
            n_cone = features.face_type_counts.get("cone", 0)
            n_torus = features.face_type_counts.get("torus", 0)
            n_sphere = features.face_type_counts.get("sphere", 0)
            # Strict head test. A real bolt has at least one of:
            #   - explicit cone/torus/sphere face (countersunk, pan, button)
            #   - two distinct ext cyl clusters where the bigger ≥ 1.3× smaller
            #   - head_diameter populated and ≥ 1.3× the bounding shaft diameter
            #     (the existing head-detection heuristic only fires when a clearly
            #     larger cylinder cluster is present, which rules out ropes)
            from .brep_signature import _cluster_radii
            ext_r = [c.radius for c in features.cylinders if not c.is_internal]
            ext_clusters = _cluster_radii(ext_r)
            shaft_est = features.bounding_cylinder_diameter
            has_strict_head = (
                (n_cone + n_torus + n_sphere) >= 1
                or (len(ext_clusters) >= 2 and ext_clusters[-1][0] >= 1.3 * ext_clusters[0][0])
                or (features.head_diameter is not None
                    and shaft_est > 0
                    and features.head_diameter >= 1.3 * shaft_est)
            )
            if not has_strict_head:
                best_label = DetectionLabel(
                    fastener_type="unclassified", confidence=0.0,
                )
                best_score = 0.0
    # When the size score is essentially zero, drop the spurious variant —
    # we shouldn't display "M4" if the M4 entry only matched by accident.
    if best_score < 0.30:
        best_label.standard = None
        best_label.variant = None

    # Repetition bonus
    if repetition_count >= config.min_repetition_count:
        best_label.confidence = min(best_label.confidence + 0.10, 1.0)

    # Apply confidence thresholds
    if best_label.confidence < config.rule_confidence_threshold:
        best_label.fastener_type = "unclassified"
        best_label.standard = None
        best_label.variant = None
    elif best_label.confidence < 0.90:
        best_label.fastener_type = f"possible_{best_label.fastener_type}"

    return best_label


def _score_match(
    features: GeometricFeatures,
    fastener_type: str,
    dims: dict,
    config: DetectionConfig,
    sig: Optional[Signature] = None,
) -> float:
    """Score how well features match a specific ISO table entry."""
    import numpy as np

    score = 0.0
    tol = config.dimension_tolerance_mm

    # Shaft diameter (weight: 0.30). When the signature gives us a
    # confident shaft (which excludes the head from the measurement),
    # trust ONLY that — otherwise the head-sized bbox dimension matches
    # the wrong ISO size (e.g. an M3 pan-cross-head has a 6mm head that
    # accidentally matches the M6 shaft entry).
    #
    # For nuts the cylindrical face we measure is the THREAD MINOR
    # diameter, but ISO entries key on the THREAD MAJOR (the bolt shaft
    # that screws in). The gap is roughly the thread depth: 1.3–1.7 mm
    # for M8–M16 metric coarse. Add a +1.5 mm candidate for nut-class
    # signatures so an M14 wheel nut (12.5mm bore) matches the M14 entry
    # (14mm shaft) instead of the M12 entry (12mm shaft).
    ref_shaft = dims.get("shaft_diameter", dims.get("bore_diameter", 0.0))
    if ref_shaft > 0:
        shaft_candidates: list[float] = []
        if sig and sig.shaft_diameter > 0:
            shaft_candidates.append(sig.shaft_diameter)
            if sig.fastener_type == "hex_nut":
                # add thread-major estimate
                shaft_candidates.append(sig.shaft_diameter + 1.5)
        else:
            if features.bounding_cylinder_diameter > 0:
                shaft_candidates.append(features.bounding_cylinder_diameter)
            if features.cylindrical_face_radii:
                shaft_candidates.append(
                    float(np.median(features.cylindrical_face_radii)) * 2.0
                )
        if shaft_candidates:
            diff = min(abs(c - ref_shaft) for c in shaft_candidates)
            if diff <= tol:
                score += 0.30
            elif diff <= tol * 3:
                score += 0.15 * (1 - diff / (tol * 3))

    # Head width / head diameter (weight: 0.20). Prefer signature head;
    # fall back to feat.head_diameter if available.
    ref_head = dims.get("head_width", dims.get("head_diameter",
                dims.get("width_across_flats", dims.get("outer_diameter", 0.0))))
    if ref_head > 0:
        head_candidates: list[float] = []
        if sig and sig.head_diameter > 0:
            head_candidates.append(sig.head_diameter)
        if features.head_diameter is not None:
            head_candidates.append(features.head_diameter)
        if head_candidates:
            diff = min(abs(c - ref_head) for c in head_candidates)
            if diff <= tol:
                score += 0.20
            elif diff <= tol * 3:
                score += 0.10 * (1 - diff / (tol * 3))

    # Head height / thickness (weight: 0.15)
    ref_height = dims.get("head_height", dims.get("height", dims.get("thickness", 0.0)))
    if ref_height > 0 and features.head_height is not None:
        diff = abs(features.head_height - ref_height)
        if diff <= tol:
            score += 0.15
        elif diff <= tol * 3:
            score += 0.07 * (1 - diff / (tol * 3))

    # Thread pitch (weight: 0.10)
    ref_pitch = dims.get("thread_pitch", 0.0)
    if ref_pitch > 0 and features.has_thread:
        score += 0.05  # bonus for having thread features

    # Aspect ratio range check (weight: 0.10)
    if fastener_type in BOLT_TYPES and 1.5 <= features.aspect_ratio <= 15.0:
        score += 0.10
    elif fastener_type in NUT_TYPES and 0.3 <= features.aspect_ratio <= 1.5:
        score += 0.10
    elif fastener_type in WASHER_TYPES and 0.05 <= features.aspect_ratio <= 0.5:
        score += 0.10

    # Cylindrical dominance bonus (weight: 0.05)
    if features.cylindrical_surface_ratio > 0.5:
        score += 0.05

    return min(score, 1.0)


def _collect_detected_dims(features: GeometricFeatures) -> dict:
    """Collect measured dimensions for the detection label."""
    dims = {
        "shaft_dia": round(features.bounding_cylinder_diameter, 3),
        "length": round(features.bounding_cylinder_length, 3),
        "aspect_ratio": round(features.aspect_ratio, 3),
        "volume": round(features.volume, 3),
        "surface_area": round(features.surface_area, 3),
    }
    if features.head_diameter is not None:
        dims["head_dia"] = round(features.head_diameter, 3)
    if features.head_height is not None:
        dims["head_ht"] = round(features.head_height, 3)
    return dims


def detect_repetitions(
    all_features: list[tuple[str, GeometricFeatures]],
    tolerance: float = 0.02,
) -> dict[str, list[str]]:
    """Group geometrically identical parts.

    Returns dict mapping group_id -> list of part UUIDs.
    """
    groups: dict[str, list[str]] = {}
    assigned: set[str] = set()

    for i, (id_a, feat_a) in enumerate(all_features):
        if id_a in assigned:
            continue

        group = [id_a]
        assigned.add(id_a)

        for j in range(i + 1, len(all_features)):
            id_b, feat_b = all_features[j]
            if id_b in assigned:
                continue

            if _features_match(feat_a, feat_b, tolerance):
                group.append(id_b)
                assigned.add(id_b)

        if len(group) > 1:
            group_id = f"group_{len(groups)}"
            groups[group_id] = group

    return groups


def _features_match(a: GeometricFeatures, b: GeometricFeatures, tol: float) -> bool:
    """Check if two feature sets are geometrically similar within tolerance."""
    if a.bounding_cylinder_diameter < 1e-12 or b.bounding_cylinder_diameter < 1e-12:
        return False

    checks = [
        (a.bounding_cylinder_diameter, b.bounding_cylinder_diameter),
        (a.bounding_cylinder_length, b.bounding_cylinder_length),
    ]
    if a.volume > 0 and b.volume > 0:
        checks.append((a.volume, b.volume))

    for va, vb in checks:
        if va > 0 and abs(va - vb) / va > tol:
            return False

    return True
