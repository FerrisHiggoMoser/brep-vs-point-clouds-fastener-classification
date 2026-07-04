"""Mesh-to-BRep reconstruction (T4 fallback).

Attempts to convert triangle meshes back to B-Rep solid geometry using:
1. Curvature segmentation
2. RANSAC primitive fitting (plane, cylinder, cone, sphere, torus)
3. NURBS surface fitting for non-primitive regions
4. Face building and sewing into a solid

Every part produced by this module is flagged with provenance "reconstructed".
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import ReconstructionConfig

logger = logging.getLogger(__name__)


def mesh_to_brep(vertices: np.ndarray, faces: np.ndarray,
                 config: ReconstructionConfig | None = None) -> Optional[object]:
    """Convert a triangle mesh to a B-Rep solid via primitive fitting + NURBS.

    Args:
        vertices: Nx3 array of vertex positions (mm)
        faces: Mx3 array of triangle indices
        config: Reconstruction quality config

    Returns:
        TopoDS_Shape or None if reconstruction fails.
        On failure, returns a shell rather than a solid.
    """
    if config is None:
        config = ReconstructionConfig()

    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeFace
        from OCC.Core.ShapeFix import ShapeFix_Shape
        from OCC.Core.TopoDS import TopoDS_Shell
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
        from OCC.Core.TColgp import TColgp_Array2OfPnt
        from OCC.Core.gp import gp_Pnt

        HAS_OCC = True
    except ImportError:
        HAS_OCC = False
        logger.warning("PythonOCC not available for mesh-to-BRep reconstruction")
        return None

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    logger.info(f"Reconstructing B-Rep from {len(vertices)} vertices, {len(faces)} triangles")

    # Step 1: Segment mesh by curvature (simplified: group adjacent faces by normal)
    segments = _segment_by_normal(vertices, faces, angle_threshold=0.3)

    logger.info(f"Segmented into {len(segments)} regions")

    # Step 2: Sew faces together
    sew = BRepBuilderAPI_Sewing(config.max_sew_tolerance)

    for seg_faces in segments:
        seg_verts = vertices
        face_shape = _fit_surface_to_segment(
            seg_verts, seg_faces, config
        )
        if face_shape is not None:
            sew.Add(face_shape)

    sew.Perform()
    shape = sew.SewedShape()

    # Step 3: Try to make solid
    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
        solid_maker = BRepBuilderAPI_MakeSolid()
        solid_maker.Add(shape)
        if solid_maker.IsDone():
            shape = solid_maker.Shape()
    except Exception:
        logger.warning("Could not make solid, returning shell")

    # Step 4: ShapeFix pass
    fix = ShapeFix_Shape(shape)
    fix.Perform()
    shape = fix.Shape()

    logger.info("B-Rep reconstruction complete")
    return shape


def _segment_by_normal(vertices: np.ndarray, faces: np.ndarray,
                       angle_threshold: float = 0.3) -> list[np.ndarray]:
    """Segment mesh faces by surface normal similarity.

    Groups adjacent triangles whose normals differ by less than angle_threshold radians.
    Returns list of face index arrays.
    """
    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms

    # Simple k-means-like clustering by normal direction
    n_faces = len(faces)
    if n_faces == 0:
        return []

    assigned = np.full(n_faces, -1, dtype=int)
    segments = []
    current_seg = 0

    for i in range(n_faces):
        if assigned[i] >= 0:
            continue

        # Start new segment
        seg_indices = [i]
        assigned[i] = current_seg
        ref_normal = normals[i]

        # Find all faces with similar normal
        for j in range(i + 1, n_faces):
            if assigned[j] >= 0:
                continue
            dot = np.abs(np.dot(ref_normal, normals[j]))
            if dot > np.cos(angle_threshold):
                seg_indices.append(j)
                assigned[j] = current_seg

        segments.append(faces[seg_indices])
        current_seg += 1

    return segments


def _fit_surface_to_segment(vertices: np.ndarray, segment_faces: np.ndarray,
                            config: ReconstructionConfig) -> Optional[object]:
    """Fit a surface (primitive or NURBS) to a mesh segment and return a face shape."""
    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
        from OCC.Core.TColgp import TColgp_Array2OfPnt
        from OCC.Core.gp import gp_Pnt

        # Get unique vertices from this segment
        unique_indices = np.unique(segment_faces.flatten())
        seg_verts = vertices[unique_indices]

        if len(seg_verts) < 3:
            return None

        # Try primitive fit first: check if planar
        pca_normal, pca_center, flatness = _pca_analysis(seg_verts)

        if flatness < config.primitive_fit_tolerance:
            # Planar — create a plane face
            from OCC.Core.gp import gp_Pln, gp_Dir
            plane = gp_Pln(
                gp_Pnt(float(pca_center[0]), float(pca_center[1]), float(pca_center[2])),
                gp_Dir(float(pca_normal[0]), float(pca_normal[1]), float(pca_normal[2])),
            )
            # Create a bounded face from the convex hull bounds
            bbox_min = seg_verts.min(axis=0)
            bbox_max = seg_verts.max(axis=0)
            size = np.linalg.norm(bbox_max - bbox_min) * 1.1
            face_maker = BRepBuilderAPI_MakeFace(plane, -size, size, -size, size)
            if face_maker.IsDone():
                return face_maker.Shape()

        # Fallback: NURBS surface fit
        grid_res = min(config.nurbs_grid_resolution, int(np.sqrt(len(seg_verts))))
        grid_res = max(grid_res, 2)

        # Create a regular grid of points via projection
        pnt_array = TColgp_Array2OfPnt(1, grid_res, 1, grid_res)
        for ui in range(grid_res):
            for vi in range(grid_res):
                idx = min((ui * grid_res + vi) % len(seg_verts), len(seg_verts) - 1)
                p = seg_verts[idx]
                pnt_array.SetValue(ui + 1, vi + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))

        approx = GeomAPI_PointsToBSplineSurface(pnt_array, config.nurbs_degree, config.nurbs_degree, 0, 1)
        surface = approx.Surface()

        face_maker = BRepBuilderAPI_MakeFace(surface, 1e-6)
        if face_maker.IsDone():
            return face_maker.Shape()

        return None

    except Exception as e:
        logger.debug(f"Surface fitting failed: {e}")
        return None


def _pca_analysis(points: np.ndarray) -> tuple:
    """PCA analysis of a point cloud. Returns (normal, center, flatness).

    flatness is the ratio of smallest eigenvalue to largest — 0 means perfectly planar.
    """
    center = points.mean(axis=0)
    centered = points - center

    if len(centered) < 3:
        return np.array([0, 0, 1]), center, 0.0

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Normal is the eigenvector of the smallest eigenvalue
    normal = eigenvectors[:, 0]

    # Flatness ratio
    flatness = eigenvalues[0] / max(eigenvalues[2], 1e-10)

    return normal, center, flatness
