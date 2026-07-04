"""Detect hole candidates on host parts by clustering co-axial internal cylinders.

A "hole" is one or more concave (internal) cylindrical faces that share an
axis. A single cylinder = simple through/blind hole. A larger-radius cylinder
sitting axially on top of a smaller one = counterbore. A cone face adjacent
to a cylinder = countersink (cone is checked via face-type counts on the
parent feature set, since the cone's geometry isn't stored per-face yet).

The output is consumed by the axis-aware fastener-to-hole matcher in
detect.py — every fastener is scored against every HoleCandidate across all
non-fastener parts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .geometric_features import CylindricalFeature, GeometricFeatures


HoleKind = Literal["through", "blind", "counterbore", "countersink"]


@dataclass
class HoleCandidate:
    host_uuid: str
    axis_origin: np.ndarray       # one point on the axis
    axis_direction: np.ndarray    # unit vector
    diameter: float               # smallest (innermost) diameter in the stack
    top: np.ndarray               # entry point along +axis_direction
    bottom: np.ndarray            # exit / floor point
    kind: HoleKind
    stack: list[CylindricalFeature]


def _perp_distance_point_to_line(
    point: np.ndarray, line_origin: np.ndarray, line_dir: np.ndarray,
) -> float:
    """Distance from `point` to the infinite line through `line_origin`/`line_dir`."""
    v = point - line_origin
    proj = float(np.dot(v, line_dir))
    closest = line_origin + proj * line_dir
    return float(np.linalg.norm(point - closest))


def _same_axis(
    a: CylindricalFeature, b: CylindricalFeature,
    angle_cos_tol: float = 0.999, radial_tol_mm: float = 0.05,
) -> bool:
    """Two cylinders share an axis if their directions are (anti-)parallel
    and one cylinder's origin lies on the other's axis line."""
    da = np.asarray(a.axis_direction, dtype=np.float64)
    db = np.asarray(b.axis_direction, dtype=np.float64)
    if abs(float(np.dot(da, db))) < angle_cos_tol:
        return False
    pa = np.asarray(a.axis_origin, dtype=np.float64)
    pb = np.asarray(b.axis_origin, dtype=np.float64)
    return _perp_distance_point_to_line(pa, pb, db) < radial_tol_mm


def _cluster_coaxial(internal: list[CylindricalFeature]) -> list[list[CylindricalFeature]]:
    """Group internal cylinders that share an axis line."""
    clusters: list[list[CylindricalFeature]] = []
    for c in internal:
        placed = False
        for cluster in clusters:
            if _same_axis(c, cluster[0]):
                cluster.append(c)
                placed = True
                break
        if not placed:
            clusters.append([c])
    return clusters


def _axial_extent(
    cluster: list[CylindricalFeature],
    axis_origin: np.ndarray, axis_dir: np.ndarray,
) -> tuple[float, float]:
    """Return (min_t, max_t) — the cluster's extent projected onto the axis,
    measured as signed distance from `axis_origin` along `axis_dir`."""
    ts: list[float] = []
    for cf in cluster:
        origin = np.asarray(cf.axis_origin, dtype=np.float64)
        t0 = float(np.dot(origin - axis_origin, axis_dir))
        ts.extend([t0, t0 + cf.length])
    return min(ts), max(ts)


def _classify_kind(
    cluster: list[CylindricalFeature],
    feat: GeometricFeatures,
    axis_origin: np.ndarray, axis_dir: np.ndarray,
    bbox_min: np.ndarray, bbox_max: np.ndarray,
) -> HoleKind:
    """Pick one of through / blind / counterbore / countersink.

    - 2+ cylinders of clearly different radii (>15% delta) => counterbore.
    - At least one cone face on the part AND the cluster is a single cylinder
      => countersink (we can't yet bind the cone to *this* cluster — see
      Phase-2 follow-up note in the plan).
    - Otherwise: through if both axial endpoints lie outside or on the
      bounding box of the part; blind if only one does.
    """
    if len(cluster) >= 2:
        radii = sorted(c.radius for c in cluster)
        if (radii[-1] - radii[0]) / max(radii[0], 1e-9) > 0.15:
            return "counterbore"

    n_cone = feat.face_type_counts.get("cone", 0)
    if n_cone >= 1 and len(cluster) == 1:
        return "countersink"

    # Compare the hole's axial length to the part's extent along the same
    # axis. A hole spanning ≥ 90% of the part's thickness in that direction
    # is "through"; otherwise "blind". This is robust to where OCC anchors
    # the cylinder's parameter origin (which can be outside the part's
    # bbox after a Boolean cut operation).
    cluster_length = max(cf.length for cf in cluster)
    projected_extent = sum(
        abs(axis_dir[i]) * (bbox_max[i] - bbox_min[i]) for i in range(3)
    )
    if cluster_length >= 0.9 * projected_extent:
        return "through"
    return "blind"


def detect_holes(part_uuid: str, feat: GeometricFeatures) -> list[HoleCandidate]:
    """Return all hole candidates on a single part.

    Skips parts that have no internal cylindrical features (e.g. solid blocks,
    pure assemblies). The returned list is empty for those.
    """
    internal = [c for c in feat.cylinders if c.is_internal]
    if not internal:
        return []

    bbox_min = np.asarray(feat.bbox_min, dtype=np.float64)
    bbox_max = np.asarray(feat.bbox_max, dtype=np.float64)

    candidates: list[HoleCandidate] = []
    for cluster in _cluster_coaxial(internal):
        # Use the longest cylinder's axis as the representative axis.
        anchor = max(cluster, key=lambda c: c.length)
        axis_dir = np.asarray(anchor.axis_direction, dtype=np.float64)
        norm = np.linalg.norm(axis_dir)
        if norm < 1e-9:
            continue
        axis_dir = axis_dir / norm
        axis_origin = np.asarray(anchor.axis_origin, dtype=np.float64)

        t_min, t_max = _axial_extent(cluster, axis_origin, axis_dir)
        top = axis_origin + t_min * axis_dir
        bottom = axis_origin + t_max * axis_dir

        diameter = 2.0 * min(c.radius for c in cluster)
        kind = _classify_kind(cluster, feat, axis_origin, axis_dir, bbox_min, bbox_max)

        candidates.append(HoleCandidate(
            host_uuid=part_uuid,
            axis_origin=axis_origin,
            axis_direction=axis_dir,
            diameter=float(diameter),
            top=top,
            bottom=bottom,
            kind=kind,
            stack=cluster,
        ))
    return candidates
