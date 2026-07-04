"""Main fastener detection entry point.

Orchestrates rule-based classification, optional ML inference, and
writes detection results back into the manifest.
"""

import logging
from typing import Any, Optional
from uuid import UUID

from ..schema import Manifest, DetectionLabel, Relationship
from ..config import DetectionConfig
from .geometric_features import (
    CylindricalFeature, GeometricFeatures, extract_brep_features, extract_mesh_features,
)
from .holes import HoleCandidate, detect_holes
from .rule_based import classify_part, detect_repetitions

logger = logging.getLogger(__name__)


def _looks_like_screw(feat) -> bool:
    """Pure-BRep heuristic for 'this part is a threaded screw'.

    Designed to catch screws that BRepFormer's McMaster-trained head
    mis-labels as washers / pins / threaded-rods / non_fastener.
    Thresholds are tuned against ISO14581 countersunk torx screws in
    the ISIS satellite assembly but generalize to any screw with a
    cylindrical body and a head with cone / torus / sphere surfaces.

    A part is a screw if:
      - bounding cylinder diameter in [1.0, 30.0] mm  (M1.5 .. M16 thread)
      - bounding cylinder length in [2.0, 80.0] mm
      - cylindrical surfaces dominate (cyl_ratio >= 0.40) — the shaft
      - aspect ratio in [0.30, 12.0] — not flat (washer), not a rod
      - at least one cone OR torus OR sphere face — the head profile
        (countersunk = cone, button head = sphere, pan/cheese head = torus)
      - at least 3 planar faces — head + drive recess flats
      - face count between 8 and 100 — not trivial, not absurd
    """
    if not feat:
        return False
    d = feat.bounding_cylinder_diameter
    L = feat.bounding_cylinder_length
    if not (1.0 <= d <= 30.0):
        return False
    if not (2.0 <= L <= 80.0):
        return False
    if feat.cylindrical_surface_ratio < 0.40:
        return False
    if not (0.30 <= feat.aspect_ratio <= 12.0):
        return False
    if not (8 <= feat.num_faces <= 100):
        return False
    n_plane = feat.face_type_counts.get("plane", 0)
    n_cone = feat.face_type_counts.get("cone", 0)
    n_torus = feat.face_type_counts.get("torus", 0)
    n_sphere = feat.face_type_counts.get("sphere", 0)
    if n_plane < 3:
        return False
    if (n_cone + n_torus + n_sphere) < 1:
        return False
    return True


def detect_fasteners(
    manifest: Manifest,
    shapes: Optional[dict[str, Any]] = None,
    meshes: Optional[dict[str, tuple]] = None,
    config: Optional[DetectionConfig] = None,
) -> Manifest:
    """Run fastener detection on all parts in the manifest.

    Args:
        manifest: The Manifest with parts to analyse.
        shapes: Optional dict mapping part UUID string → OCC TopoDS_Shape
                 for B-Rep feature extraction.
        meshes: Optional dict mapping part UUID string →
                 (vertices_Nx3, normals_Nx3, faces_Mx3) ndarrays.
        config: Detection configuration. Uses defaults when None.

    Returns:
        Updated Manifest with ``part.detection`` populated and new
        ``Relationship`` entries for detected fastener assemblies.
    """
    if config is None:
        config = DetectionConfig()

    shapes = shapes or {}
    meshes = meshes or {}

    # --- Step 0: Group parts by topology hash so identical instances share
    # all expensive work (rule-based feature extraction + BRepFormer inference).
    # ISISpace satellite STEPs contain ~80k shape instances but only ~80 unique
    # geometries — running detection once per instance burns hours; once per
    # unique shape takes seconds.
    hash_to_canonical_uid: dict[str, str] = {}  # topology_hash -> canonical part uuid
    canonical_for: dict[str, str] = {}          # part uuid -> canonical part uuid
    for part in manifest.parts:
        uid = str(part.uuid)
        h = part.fingerprint.topology_hash if part.fingerprint else None
        if not h or h in ("root", "empty"):
            canonical_for[uid] = uid
            continue
        if h not in hash_to_canonical_uid:
            hash_to_canonical_uid[h] = uid
        canonical_for[uid] = hash_to_canonical_uid[h]

    n_unique = len(set(canonical_for.values()))
    logger.info(
        "Dedup: %d parts -> %d unique geometries (by topology_hash). "
        "Detection cost will scale with unique count.",
        len(manifest.parts), n_unique,
    )

    # --- Step 1: Extract features per part ---
    # We extract once per UID (not per canonical) because feature.axis_origin
    # is in world coordinates — two parts with identical topology at
    # different world positions need separate feature records, otherwise the
    # screwedInto matcher conflates them. Classification (steps 3-4) still
    # benefits from canonical-based caching since classification is
    # transform-invariant, but features themselves must be per-instance.
    feature_map: dict[str, GeometricFeatures] = {}
    for part in manifest.parts:
        uid = str(part.uuid)
        if uid in shapes:
            feat = extract_brep_features(shapes[uid])
        elif uid in meshes:
            verts, norms, faces = meshes[uid]
            feat = extract_mesh_features(verts, norms, faces)
        else:
            feat = GeometricFeatures()
        feature_map[uid] = feat

    # --- Step 2: Detect repetition groups ---
    features_list = [(uid, feat) for uid, feat in feature_map.items()]
    repetition_groups = detect_repetitions(features_list)

    # Build reverse lookup: part_id → group size
    rep_count: dict[str, int] = {}
    for group_parts in repetition_groups.values():
        for pid in group_parts:
            rep_count[pid] = len(group_parts)

    # --- Step 3: Rule-based classification, once per unique canonical part ---
    labels: dict[str, DetectionLabel] = {}
    canonical_labels: dict[str, DetectionLabel] = {}
    for part in manifest.parts:
        uid = str(part.uuid)
        canonical = canonical_for[uid]
        if canonical in canonical_labels:
            labels[uid] = canonical_labels[canonical]
            continue
        feat = feature_map[uid]
        count = rep_count.get(uid, 1)
        if config.enable_rule_based:
            label = classify_part(feat, config, repetition_count=count)
        else:
            label = DetectionLabel(fastener_type="unclassified", confidence=0.0)
        canonical_labels[canonical] = label
        labels[uid] = label

    # --- Step 4: Optional ML classification ---
    if config.enable_ml:
        from .ml_classifier import is_ml_available, FastenerClassifier, ensemble_merge

        if is_ml_available():
            try:
                classifier = FastenerClassifier(config)
                # 13-class label order — MUST match training order (alphabetical).
                # Verified against full_analysis_subtype13.json#classes.
                bf_class_names = [
                    "anchors", "keys", "nails", "non_fastener", "nuts", "pins",
                    "retaining-rings", "rivets", "screws", "spacers",
                    "threaded-inserts", "threaded-rods", "washers",
                ]
                # Cap from training: face matrices are O(N²); skip BF on huge parts.
                MAX_FACES_BF = 600
                # Dedup-aware merged label cache: once we've computed the
                # ensemble label for a canonical shape, every instance reuses it.
                canonical_merged: dict[str, DetectionLabel] = {}
                for part in manifest.parts:
                    uid = str(part.uuid)
                    rule_label = labels[uid]
                    canonical = canonical_for[uid]

                    # If a sibling instance already produced a merged label,
                    # short-circuit (this is the 99% case on ISISpace-style files).
                    if canonical in canonical_merged:
                        labels[uid] = canonical_merged[canonical]
                        continue

                    # Pull per-part features for the override gates below —
                    # without this, `feat` would leak from step 3's last
                    # iteration and override decisions would be made against
                    # an arbitrary other part's geometry.
                    feat = feature_map[uid]

                    # Only run ML on parts that are uncertain or unclassified
                    if rule_label.confidence < 0.90:
                        ml_label = DetectionLabel(
                            fastener_type="unclassified", confidence=0.0
                        )

                        # Try BRepFormer first if shape available (per the
                        # thesis result it beats PointNet++ on subtype id by
                        # 14.3pp). Falls back to PN++ on shapes with too
                        # many faces (matches the training-time cap) or on
                        # any feature-extraction failure.
                        bf_ran = False
                        if uid in shapes and classifier.brepformer_model is not None:
                            try:
                                from ..models.brepformer.feature_extractor import (
                                    extract_face_uv_grids, extract_edge_curves,
                                    compute_topology_distances,
                                )
                                from OCC.Core.TopExp import TopExp_Explorer
                                from OCC.Core.TopAbs import TopAbs_FACE

                                shape = shapes[uid]
                                exp = TopExp_Explorer(shape, TopAbs_FACE)
                                n_faces = 0
                                while exp.More():
                                    n_faces += 1
                                    exp.Next()
                                if n_faces == 0:
                                    raise RuntimeError("shape has 0 faces")
                                if n_faces > MAX_FACES_BF:
                                    raise RuntimeError(
                                        f"face count {n_faces} > BF cap {MAX_FACES_BF}; "
                                        f"will fall back to PointNet++"
                                    )
                                face_grids = extract_face_uv_grids(shape)
                                edge_curves = extract_edge_curves(shape)
                                topo = compute_topology_distances(shape)
                                # ml_classifier expects all 4 distance matrices.
                                for key in ("face_shortest", "face_centroid",
                                            "face_angular", "edge_path"):
                                    topo.setdefault(key, None)
                                # Drop any None entries — the classifier zeros them.
                                topo = {k: v for k, v in topo.items() if v is not None}
                                ml_label = classifier.classify_brep(
                                    face_grids=face_grids,
                                    edge_grids=edge_curves,
                                    topo_distances=topo,
                                    class_names=bf_class_names,
                                )
                                bf_ran = True
                            except Exception as bf_exc:
                                logger.info(
                                    "BF failed for %s (%s); will try PN++ if available",
                                    part.name, bf_exc,
                                )

                        # Fall back to PointNet++ if BF didn't produce a usable
                        # label or wasn't tried (e.g. too many faces).
                        if not bf_ran and uid in meshes and classifier.pointnet_model is not None:
                            verts, norms, faces = meshes[uid]
                            ml_label = classifier.classify_pointcloud(verts, norms)

                        merged = ensemble_merge(rule_label, ml_label, features=feat)

                        # Geometric screw recovery: BF was trained on
                        # McMaster-Carr screws (mostly hex socket / pan head /
                        # socket cap) and is OOD on ISO14581-style
                        # countersunk torx screws — which the ISIS satellite
                        # is full of. To BF, short countersunk torx screws
                        # look like washers (flat flush head, no protruding
                        # shaft); long ones look like pins (thin cylindrical
                        # body) or non_fastener. Whenever BF assigns one of
                        # those mis-classes but the BRep features match a
                        # screw profile (M-size shaft, cone/torus head
                        # features, plane-dominant drive recess), override
                        # to `likely_screws`. This recovers OOD screws
                        # without retraining BF.
                        merged_base = merged.fastener_type.replace(
                            "possible_", "").replace("likely_", "")
                        if (merged_base in ("non_fastener", "washers",
                                            "pins", "threaded-rods",
                                            "unclassified")
                                and _looks_like_screw(feat)
                                and rule_label.fastener_type == "unclassified"):
                            n_cone = feat.face_type_counts.get("cone", 0)
                            n_torus = feat.face_type_counts.get("torus", 0)
                            logger.info(
                                "Geometric screw override on %s "
                                "(BF=%s, bcyl_d=%.2fmm, bcyl_l=%.2fmm, "
                                "cyl_ratio=%.2f, cones=%d, tori=%d) -> likely_screws",
                                part.name, merged_base,
                                feat.bounding_cylinder_diameter,
                                feat.bounding_cylinder_length,
                                feat.cylindrical_surface_ratio, n_cone, n_torus,
                            )
                            ml_top3 = (ml_label.detected_dimensions or {}).get("ml_top3") or []
                            merged = DetectionLabel(
                                fastener_type="likely_screws",
                                confidence=0.60,
                                method="ensemble",
                                detected_dimensions={
                                    "ml_top3": ml_top3,
                                    "bf_original": merged_base,
                                    "ml_engine": "brepformer+geom_override",
                                    "override": "geometric_screw",
                                },
                            )

                        # Symmetric BF non_fastener gate: BF was trained on the
                        # McMaster-Carr catalog only and is overconfident on
                        # OOD inputs (ISIS-satellite ISO14581 screws, GrabCAD
                        # parts with unusual scale, etc.). When BF labels a
                        # part as `non_fastener` but the part's geometry is
                        # fastener-like — cylindrical-surface dominant OR has
                        # torus surfaces (threads/seats/chamfered heads) — we
                        # try to recover a real fastener label from BF's top-3
                        # distribution (rank-2 often holds the right answer
                        # when rank-1 collapses to non_fastener). If no
                        # fastener class in the top-3 has enough mass we fall
                        # back to `unclassified` so the name-substring rescue
                        # in sidecar_server.py can still match ISO/DIN parts.
                        merged_base = merged.fastener_type.replace(
                            "possible_", "").replace("likely_", "")
                        if (merged_base == "non_fastener"
                                and merged.method in ("ensemble", "ml_brepformer")
                                and rule_label.fastener_type == "unclassified"):
                            has_torus = feat.face_type_counts.get("torus", 0) > 0
                            cyl_dominant = feat.cylindrical_surface_ratio >= 0.50
                            if has_torus or cyl_dominant:
                                top3 = (ml_label.detected_dimensions or {}).get("ml_top3") or []
                                rescued = None
                                for cls, prob in top3:
                                    if cls in ("non_fastener", "non-fastener"):
                                        continue
                                    if prob >= 0.20:
                                        rescued = (cls, prob)
                                        break
                                if rescued is not None:
                                    logger.info(
                                        "Recovering BF top-2 fastener for %s: "
                                        "%s @ %.3f (cyl_ratio=%.2f, torus=%d)",
                                        part.name, rescued[0], rescued[1],
                                        feat.cylindrical_surface_ratio,
                                        feat.face_type_counts.get("torus", 0),
                                    )
                                    merged = DetectionLabel(
                                        fastener_type=f"likely_{rescued[0]}",
                                        confidence=round(rescued[1], 4),
                                        method=merged.method,
                                        detected_dimensions=merged.detected_dimensions,
                                    )
                                else:
                                    logger.info(
                                        "Demoting BF non_fastener on %s to "
                                        "unclassified (no fastener in top-3; "
                                        "cyl_ratio=%.2f, torus=%d)",
                                        part.name, feat.cylindrical_surface_ratio,
                                        feat.face_type_counts.get("torus", 0),
                                    )
                                    merged = DetectionLabel(
                                        fastener_type="unclassified",
                                        confidence=0.0,
                                        method=merged.method,
                                        detected_dimensions=merged.detected_dimensions,
                                    )

                        # Size-sanity gate: fasteners in BF's training set
                        # are bounded — anything > 200mm in any axis or
                        # < 1mm overall is almost certainly not a fastener,
                        # regardless of what the model says.
                        bmin = part.fingerprint.bbox_min
                        bmax = part.fingerprint.bbox_max
                        max_dim = max(bmax[i] - bmin[i] for i in range(3))
                        if max_dim > 200.0 or max_dim < 1.0:
                            merged_base = merged.fastener_type.replace(
                                "possible_", "").replace("likely_", "")
                            if merged_base != "unclassified" and merged_base != "non_fastener":
                                logger.info(
                                    "Demoting %s: %s -> unclassified "
                                    "(max_dim=%.1fmm outside fastener range)",
                                    part.name, merged.fastener_type, max_dim,
                                )
                                merged = DetectionLabel(
                                    fastener_type="unclassified",
                                    confidence=0.0,
                                    method=merged.method,
                                )

                        labels[uid] = merged
                        canonical_merged[canonical] = merged
                    else:
                        # Rule label was already strong; cache it as-is for siblings.
                        canonical_merged[canonical] = rule_label
            except Exception:
                logger.exception("ML classification failed; using rule-based results only")
        else:
            logger.warning(
                "ML detection requested but PyTorch not installed; "
                "using rule-based results only"
            )

    # --- Step 5: Write labels back to manifest ---
    fastener_uuids: list[str] = []
    non_fastener_uuids: list[str] = []

    for part in manifest.parts:
        uid = str(part.uuid)
        label = labels[uid]
        part.detection = label

        base_type = label.fastener_type.replace("possible_", "").replace("likely_", "")
        if base_type != "unclassified" and label.confidence >= config.rule_confidence_threshold:
            fastener_uuids.append(uid)
        else:
            non_fastener_uuids.append(uid)

    # --- Step 6: Infer fastener relationships (axis-aware hole matching) ---
    new_relationships = _infer_fastener_relationships(
        manifest, fastener_uuids, non_fastener_uuids, feature_map
    )
    manifest.relationships.extend(new_relationships)

    # --- Step 7: Emit explicit `contained_in` relationships for housing ---
    housing_relationships = _infer_housing_relationships(manifest, fastener_uuids)
    manifest.relationships.extend(housing_relationships)

    # --- Summary ---
    total = len(manifest.parts)
    detected = len(fastener_uuids)
    logger.info(
        "Detection complete: %d/%d parts classified as fasteners; "
        "%d fastener->host arcs, %d contained_in arcs",
        detected, total, len(new_relationships), len(housing_relationships),
    )

    return manifest


def _infer_fastener_relationships(
    manifest: Manifest,
    fastener_uuids: list[str],
    structural_uuids: list[str],
    feature_map: dict[str, GeometricFeatures],
) -> list[Relationship]:
    """Infer which hole on which host part each fastener is screwed into.

    For every fastener:
      1. Pick the principal shaft axis (longest external cylinder).
      2. Score every HoleCandidate across structural parts:
         - axis_align (require > 0.95)
         - radial_offset (require < 0.5 mm)
         - diameter_gap (require in [-0.05, 2.0] mm; classify fit_class)
         - axial_overlap (require > 0)
      3. Emit one Relationship per matched host. Pass-through bolts yield
         multiple relationships ordered by bolt_order along the shaft.
    """
    import numpy as np

    relationships: list[Relationship] = []
    if not fastener_uuids or not structural_uuids:
        return relationships

    # Gather all hole candidates from EVERY part that has internal
    # cylindrical features, not just "structural" parts. Real assemblies
    # have parts that are simultaneously fastener-shaped AND act as hosts
    # (press-fit inserts, captive panel screws, hubs with radial bolt
    # holes, threaded couplings). The matcher's axis/radial/diameter/
    # overlap gates still filter out spurious matches; relaxing the
    # candidate pool only adds true positives.
    all_holes: list[HoleCandidate] = []
    fastener_set = set(fastener_uuids)
    for part in manifest.parts:
        uid = str(part.uuid)
        feat = feature_map.get(uid)
        if not feat or feat.volume <= 0:
            continue
        # Skip self-loops: don't let a fastener be a hole-host for itself.
        # We allow a fastener-classified part to host OTHER fasteners.
        all_holes.extend(detect_holes(uid, feat))

    if not all_holes:
        # No holes detected anywhere; falling back to bbox-centroid would be
        # noisy and is now the responsibility of the higher-level pipeline.
        return relationships

    for f_uid in fastener_uuids:
        feat = feature_map.get(f_uid)
        if not feat:
            continue
        shaft = _principal_shaft(feat)
        if shaft is None:
            continue
        f_axis_origin = np.asarray(shaft.axis_origin, dtype=np.float64)
        f_axis_dir = _unit(np.asarray(shaft.axis_direction, dtype=np.float64))
        shaft_d = 2.0 * shaft.radius
        shaft_len = shaft.length

        scored: list[tuple[float, HoleCandidate, dict]] = []
        for hole in all_holes:
            # Self-loop guard: don't match a fastener against a hole in
            # itself (e.g. the hex socket recess in an SHCS bolt's own head)
            if hole.host_uuid == f_uid:
                continue
            score, info = _score_hole_match(
                f_axis_origin, f_axis_dir, shaft_d, shaft_len, hole,
            )
            if score > 0:
                scored.append((score, hole, info))

        if not scored:
            continue

        # Order matched holes along the shaft so bolt_order makes sense
        # (entry plate first, then any clamped layers, then the tapped one).
        scored.sort(key=lambda s: s[2]["axis_t"])

        part = next((p for p in manifest.parts if str(p.uuid) == f_uid), None)
        det = part.detection if part else None
        for bolt_order, (score, hole, info) in enumerate(scored):
            relationships.append(Relationship(
                kind="fastener",
                subject_uuid=UUID(f_uid),
                target_uuid=UUID(hole.host_uuid),
                params={
                    "hole_axis_origin": [float(x) for x in hole.axis_origin],
                    "hole_axis_dir": [float(x) for x in hole.axis_direction],
                    "hole_diameter": round(hole.diameter, 3),
                    "hole_kind": hole.kind,
                    "fit_class": info["fit_class"],
                    "diameter_gap_mm": round(info["diameter_gap"], 3),
                    "bolt_order": bolt_order,
                    "score": round(score, 3),
                    "fastener_type": det.fastener_type if det else "unknown",
                    "variant": det.variant if det else None,
                },
                inferred=True,
                confidence=float(min(1.0, score * (det.confidence if det else 1.0))),
            ))

    return relationships


def _infer_housing_relationships(
    manifest: Manifest, fastener_uuids: list[str],
) -> list[Relationship]:
    """Emit `contained_in` arcs from each fastener to its assembly parent.

    Parent info comes from the STEP XDE tree (PartEntry.parent_uuid) so the
    relationship is structural fact, not heuristic — `inferred=False`.
    """
    fastener_set = set(fastener_uuids)
    out: list[Relationship] = []
    for part in manifest.parts:
        uid = str(part.uuid)
        if uid not in fastener_set or part.parent_uuid is None:
            continue
        out.append(Relationship(
            kind="contained_in",
            subject_uuid=part.uuid,
            target_uuid=part.parent_uuid,
            params={"depth": 1},
            inferred=False,
            confidence=1.0,
        ))
    return out


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _unit(v):
    import numpy as np
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _principal_shaft(feat: GeometricFeatures) -> Optional[CylindricalFeature]:
    """Pick the fastener's shaft = the longest external cylindrical face.

    Falls back to the longest cylinder of either polarity if every face is
    flagged internal (rare but happens with malformed STEP imports).
    """
    if not feat.cylinders:
        return None
    external = [c for c in feat.cylinders if not c.is_internal]
    pool = external or feat.cylinders
    return max(pool, key=lambda c: c.length)


def _score_hole_match(
    f_origin, f_dir, shaft_diameter: float, shaft_length: float,
    hole: HoleCandidate,
) -> tuple[float, dict]:
    """Return (composite_score, info) — score is 0 if any hard gate fails."""
    import numpy as np

    # Axis alignment gate.
    axis_align = abs(float(np.dot(f_dir, hole.axis_direction)))
    if axis_align < 0.95:
        return 0.0, {}

    # Radial offset gate (perpendicular distance from fastener axis origin
    # to the hole's axis line).
    delta = f_origin - hole.axis_origin
    proj = float(np.dot(delta, hole.axis_direction))
    perp = delta - proj * hole.axis_direction
    radial_offset = float(np.linalg.norm(perp))
    if radial_offset > 0.5:
        return 0.0, {}

    # Diameter fit gate.
    #   - tap / threaded engagement: gap in [-1.5, 0.2] mm
    #     (CAD hole = tap-drill minor diameter, ≈ bolt OD minus the thread
    #      depth; e.g. M6 bolt vs M6 tap drill = 6.0 vs 5.0 → gap −1.0)
    #   - slip / medium fit:         gap in (0.2, 0.6] mm
    #   - clearance / coarse fit:    gap in (0.6, 2.0] mm
    diameter_gap = hole.diameter - shaft_diameter
    if not (-1.5 <= diameter_gap <= 2.0):
        return 0.0, {}
    if diameter_gap <= 0.2:
        fit_class = "tap"
    elif diameter_gap <= 0.6:
        fit_class = "slip"
    else:
        fit_class = "clearance"

    # Axial overlap gate: project both the shaft segment (f_origin →
    # f_origin + shaft_length·f_dir) and the hole segment (hole.top →
    # hole.bottom) onto the hole's axis and intersect.
    shaft_start = f_origin
    shaft_end = f_origin + shaft_length * f_dir
    t_start = float(np.dot(shaft_start - hole.axis_origin, hole.axis_direction))
    t_end = float(np.dot(shaft_end - hole.axis_origin, hole.axis_direction))
    shaft_t0, shaft_t1 = (t_start, t_end) if t_start <= t_end else (t_end, t_start)

    hole_t0 = float(np.dot(hole.top - hole.axis_origin, hole.axis_direction))
    hole_t1 = float(np.dot(hole.bottom - hole.axis_origin, hole.axis_direction))
    if hole_t1 < hole_t0:
        hole_t0, hole_t1 = hole_t1, hole_t0

    overlap = max(0.0, min(shaft_t1, hole_t1) - max(shaft_t0, hole_t0))
    if overlap <= 0:
        return 0.0, {}

    # Composite score (each term normalized to [0, 1]).
    # fit_score: plateau 1.0 across the canonical tap/slip/coarse-clearance
    # band, tapering linearly to 0 at the outer gates.
    if -1.2 <= diameter_gap <= 0.8:
        fit_score = 1.0
    elif diameter_gap < -1.2:
        fit_score = max(0.0, (diameter_gap - (-1.5)) / 0.3)
    else:  # diameter_gap > 0.8
        fit_score = max(0.0, (2.0 - diameter_gap) / 1.2)
    score = (
        0.40 * axis_align
        + 0.30 * (1.0 - radial_offset / 0.5)
        + 0.20 * fit_score
        + 0.10 * (overlap / max(shaft_length, 1e-6))
    )
    return float(score), {
        "axis_align": axis_align,
        "radial_offset": radial_offset,
        "diameter_gap": diameter_gap,
        "fit_class": fit_class,
        "overlap": overlap,
        # Position along the shaft (used for bolt_order). Project hole's
        # entry point onto fastener axis so it's monotonic along the shaft.
        "axis_t": float(np.dot(hole.top - f_origin, f_dir)),
    }
