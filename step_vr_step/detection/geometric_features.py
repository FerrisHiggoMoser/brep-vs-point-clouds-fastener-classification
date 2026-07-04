"""Extract geometric features from B-Rep shapes or tessellated meshes for fastener detection."""

from dataclasses import dataclass, field
from typing import Optional
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CylindricalFeature:
    """One cylindrical face on a part, with its axis line in world coordinates.

    Used by the hole detector and the axis-aware fastener-to-hole matcher.
    A face whose OCCT orientation is REVERSED is concave (an internal bore /
    hole); FORWARD is convex (an external shaft).
    """
    axis_origin: tuple[float, float, float]
    axis_direction: tuple[float, float, float]  # unit vector
    radius: float
    length: float
    is_internal: bool
    face_index: int


@dataclass
class GeometricFeatures:
    """Geometric feature vector for a single part."""
    # Bounding box
    bbox_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Global properties
    volume: float = 0.0
    surface_area: float = 0.0

    # Face type histogram (B-Rep)
    face_type_counts: dict[str, int] = field(default_factory=dict)

    # Cylindrical features
    cylindrical_face_radii: list[float] = field(default_factory=list)
    cylindrical_face_lengths: list[float] = field(default_factory=list)
    cylindrical_surface_ratio: float = 0.0

    # Per-cylinder axis info (B-Rep only). Populated by extract_brep_features.
    # Holes are detected by clustering the entries with is_internal=True.
    cylinders: list[CylindricalFeature] = field(default_factory=list)

    # Topology counts
    num_faces: int = 0
    num_edges: int = 0
    num_vertices: int = 0

    # Derived
    aspect_ratio: float = 0.0
    has_thread: bool = False
    bounding_cylinder_diameter: float = 0.0
    bounding_cylinder_length: float = 0.0

    # Head detection (if applicable)
    head_diameter: Optional[float] = None
    head_height: Optional[float] = None


# ---------------------------------------------------------------------------
# B-Rep feature extraction (requires PythonOCC)
# ---------------------------------------------------------------------------

def extract_brep_features(shape) -> GeometricFeatures:
    """Extract geometric features directly from an OpenCascade TopoDS_Shape.

    Uses OCCT surface type classification, BRepGProp for mass properties,
    and topology exploration for face/edge/vertex counts.
    """
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_VolumeProperties
        from OCC.Core.BRepLProp import BRepLProp_SLProps
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.GeomAbs import (
            GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
            GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
            GeomAbs_BezierSurface, GeomAbs_SurfaceOfRevolution,
            GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface,
            GeomAbs_OtherSurface,
        )
        from OCC.Core.TopAbs import TopAbs_REVERSED
        from OCC.Core.gp import gp_Vec, gp_Pnt
        from OCC.Extend.TopologyUtils import TopologyExplorer
    except ImportError:
        logger.warning("PythonOCC not available; cannot extract B-Rep features")
        return GeometricFeatures()

    feat = GeometricFeatures()
    topo = TopologyExplorer(shape)

    # --- Bounding box ---
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    if not bbox.IsVoid():
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        feat.bbox_min = (xmin, ymin, zmin)
        feat.bbox_max = (xmax, ymax, zmax)

    # --- Volume and surface area ---
    props_vol = GProp_GProps()
    brepgprop_VolumeProperties(shape, props_vol)
    feat.volume = abs(props_vol.Mass())

    props_srf = GProp_GProps()
    brepgprop_SurfaceProperties(shape, props_srf)
    feat.surface_area = props_srf.Mass()

    # --- Topology counts ---
    faces = list(topo.faces())
    feat.num_faces = len(faces)
    feat.num_edges = topo.number_of_edges()
    feat.num_vertices = topo.number_of_vertices()

    # --- Face type classification ---
    TYPE_MAP = {
        GeomAbs_Plane: "plane",
        GeomAbs_Cylinder: "cylinder",
        GeomAbs_Cone: "cone",
        GeomAbs_Sphere: "sphere",
        GeomAbs_Torus: "torus",
    }
    NURBS_TYPES = {
        GeomAbs_BSplineSurface, GeomAbs_BezierSurface,
        GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion,
        GeomAbs_OffsetSurface, GeomAbs_OtherSurface,
    }

    type_counts: dict[str, int] = {
        "plane": 0, "cylinder": 0, "cone": 0,
        "sphere": 0, "torus": 0, "nurbs": 0,
    }
    total_srf_area = max(feat.surface_area, 1e-12)
    cyl_area = 0.0
    cyl_radii = []
    cyl_lengths = []
    cylinders: list[CylindricalFeature] = []

    for face_idx, face in enumerate(faces):
        adaptor = BRepAdaptor_Surface(face, True)
        stype = adaptor.GetType()

        name = TYPE_MAP.get(stype)
        if name:
            type_counts[name] += 1
        elif stype in NURBS_TYPES:
            type_counts["nurbs"] += 1

        if stype == GeomAbs_Cylinder:
            cyl = adaptor.Cylinder()
            radius = cyl.Radius()
            cyl_radii.append(radius)
            # Estimate cylinder length from UV bounds
            u1, u2 = adaptor.FirstUParameter(), adaptor.LastUParameter()
            v1, v2 = adaptor.FirstVParameter(), adaptor.LastVParameter()
            length = abs(v2 - v1)
            cyl_lengths.append(length)
            cyl_area += 2 * math.pi * radius * length

            # Capture the cylinder's axis line in world coordinates.
            ax = cyl.Axis()
            origin = ax.Location()
            direction = ax.Direction()

            # Determine concave (hole) vs convex (shaft). Face Orientation
            # alone is unreliable because STEP exporters use different
            # conventions, so we compute the face's outward normal at a
            # sample point and compare it to the radial direction:
            #   external shaft  → face normal points radially outward (dot > 0)
            #   internal hole   → face normal points radially inward  (dot < 0)
            is_internal = _is_face_concave(
                face, adaptor, u1, u2, v1, v2, origin, direction,
                BRepLProp_SLProps, gp_Vec, TopAbs_REVERSED,
            )

            cylinders.append(CylindricalFeature(
                axis_origin=(origin.X(), origin.Y(), origin.Z()),
                axis_direction=(direction.X(), direction.Y(), direction.Z()),
                radius=float(radius),
                length=float(length),
                is_internal=bool(is_internal),
                face_index=face_idx,
            ))

    feat.face_type_counts = type_counts
    feat.cylindrical_face_radii = cyl_radii
    feat.cylindrical_face_lengths = cyl_lengths
    feat.cylinders = cylinders
    feat.cylindrical_surface_ratio = cyl_area / total_srf_area

    # --- Bounding cylinder approximation ---
    dims = [
        feat.bbox_max[i] - feat.bbox_min[i]
        for i in range(3)
    ]
    dims.sort()
    # Assume two smallest dims form the diameter, largest is length
    feat.bounding_cylinder_diameter = (dims[0] + dims[1]) / 2.0
    feat.bounding_cylinder_length = dims[2]
    if feat.bounding_cylinder_diameter > 1e-12:
        feat.aspect_ratio = feat.bounding_cylinder_length / feat.bounding_cylinder_diameter
    else:
        feat.aspect_ratio = 0.0

    # --- Thread heuristic ---
    # Many small cylindrical faces with similar radii suggest threads
    if len(cyl_radii) > 6:
        unique_radii = set(round(r, 2) for r in cyl_radii)
        if len(cyl_radii) / max(len(unique_radii), 1) > 3:
            feat.has_thread = True

    # --- Head detection ---
    if cyl_radii and feat.cylindrical_surface_ratio > 0.3:
        sorted_radii = sorted(cyl_radii, reverse=True)
        median_radius = float(np.median(cyl_radii))
        # Largest cylindrical face radius significantly bigger than median → head
        if sorted_radii[0] > median_radius * 1.3:
            feat.head_diameter = sorted_radii[0] * 2.0
            # Estimate head height from corresponding cylinder length
            head_idx = cyl_radii.index(sorted_radii[0])
            if head_idx < len(cyl_lengths):
                feat.head_height = cyl_lengths[head_idx]

    return feat


def _is_face_concave(
    face, adaptor, u1, u2, v1, v2, origin, direction,
    BRepLProp_SLProps, gp_Vec, TopAbs_REVERSED,
) -> bool:
    """Return True if the cylindrical face bounds a void (hole), False if
    it bounds material (shaft).

    Compares the face's *outward* normal at a sample point with the radial
    direction from the cylinder axis to that point. The face orientation
    flag alone is unreliable across STEP exporters — face_normal vs radial
    is geometrically unambiguous.
    """
    u_mid = (u1 + u2) / 2.0
    v_mid = (v1 + v2) / 2.0
    try:
        props = BRepLProp_SLProps(adaptor, u_mid, v_mid, 1, 1e-6)
        if not props.IsNormalDefined():
            return False
        n = props.Normal()
        p = props.Value()
    except Exception:
        return False

    # Face outward normal = surface normal, flipped if the face is REVERSED
    # relative to the underlying surface.
    nx, ny, nz = n.X(), n.Y(), n.Z()
    if face.Orientation() == TopAbs_REVERSED:
        nx, ny, nz = -nx, -ny, -nz

    # Radial direction at p: project (p - axis_origin) perpendicular to axis.
    dx, dy, dz = direction.X(), direction.Y(), direction.Z()
    ex, ey, ez = p.X() - origin.X(), p.Y() - origin.Y(), p.Z() - origin.Z()
    t = ex * dx + ey * dy + ez * dz
    rx, ry, rz = ex - t * dx, ey - t * dy, ez - t * dz
    rnorm = (rx * rx + ry * ry + rz * rz) ** 0.5
    if rnorm < 1e-9:
        return False
    rx, ry, rz = rx / rnorm, ry / rnorm, rz / rnorm

    # dot < 0 means outward normal points opposite to radial-outward
    # (i.e. inward toward the axis) → the cylinder bounds a void → hole.
    return (nx * rx + ny * ry + nz * rz) < 0.0


# ---------------------------------------------------------------------------
# Mesh-based feature extraction (fallback when no B-Rep available)
# ---------------------------------------------------------------------------

def extract_mesh_features(
    vertices: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray,
) -> GeometricFeatures:
    """Extract geometric features from a tessellated mesh (Nx3 arrays).

    Adapted from the POC geometric_analyzer.py.
    """
    feat = GeometricFeatures()

    if len(vertices) < 4 or len(faces) < 2:
        return feat

    # Bounding box
    feat.bbox_min = tuple(float(x) for x in vertices.min(axis=0))
    feat.bbox_max = tuple(float(x) for x in vertices.max(axis=0))

    # Volume and surface area via signed tetrahedra
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = np.linalg.norm(cross, axis=1) * 0.5
    feat.surface_area = float(face_areas.sum())
    feat.volume = float(abs(np.sum(
        v0[:, 0] * (v1[:, 1] * v2[:, 2] - v1[:, 2] * v2[:, 1]) +
        v0[:, 1] * (v1[:, 2] * v2[:, 0] - v1[:, 0] * v2[:, 2]) +
        v0[:, 2] * (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    ) / 6.0))

    feat.num_faces = len(faces)
    feat.num_vertices = len(vertices)

    # Bounding cylinder from PCA
    center = vertices.mean(axis=0)
    centered = vertices - center
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    axis = eigenvectors[:, -1]  # principal axis

    # Project onto axis for length
    projections = centered @ axis
    feat.bounding_cylinder_length = float(projections.max() - projections.min())

    # Radial distances for diameter
    proj_along = np.outer(centered @ axis, axis)
    radial = centered - proj_along
    radial_dists = np.linalg.norm(radial, axis=1)
    feat.bounding_cylinder_diameter = float(radial_dists.max() * 2.0)

    if feat.bounding_cylinder_diameter > 1e-12:
        feat.aspect_ratio = feat.bounding_cylinder_length / feat.bounding_cylinder_diameter

    # Cylindrical surface ratio (normal alignment heuristic)
    if normals is not None and len(normals) == len(vertices):
        # For cylindrical surfaces, normals are perpendicular to the axis
        dot_products = np.abs(normals @ axis)
        # Normals perpendicular to axis have small dot product
        cylindrical_mask = dot_products < 0.3
        if len(faces) > 0:
            # Weight by face area
            face_normals_dot = np.zeros(len(faces))
            for i, face_idx in enumerate(faces):
                avg_dot = dot_products[face_idx].mean()
                face_normals_dot[i] = avg_dot
            cyl_face_mask = face_normals_dot < 0.3
            cyl_area = face_areas[cyl_face_mask].sum()
            feat.cylindrical_surface_ratio = float(cyl_area / max(feat.surface_area, 1e-12))

    return feat
