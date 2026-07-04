"""Geometric fingerprinting for shape identity and comparison.

Computes bounding box, volume, surface area, and a topology hash
that survives re-import. Used for fallback matching when UUIDs are stripped.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def fingerprint_shape(shape) -> Optional[dict]:
    """Compute fingerprint from an OCCT TopoDS_Shape.

    Returns dict with bbox_min, bbox_max, volume_mm3, surface_area_mm2,
    topology_hash, vertex_count, face_count. Returns None if shape is invalid.
    """
    try:
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX

        if shape is None or shape.IsNull():
            return None

        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        vprops = GProp_GProps()
        brepgprop.VolumeProperties(shape, vprops)
        volume = abs(vprops.Mass())

        sprops = GProp_GProps()
        brepgprop.SurfaceProperties(shape, sprops)
        surface_area = abs(sprops.Mass())

        face_count = 0
        face_areas = []
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face_count += 1
            fp = GProp_GProps()
            brepgprop.SurfaceProperties(exp.Current(), fp)
            face_areas.append(abs(fp.Mass()))
            exp.Next()

        edge_count = 0
        edge_lengths = []
        exp = TopExp_Explorer(shape, TopAbs_EDGE)
        while exp.More():
            edge_count += 1
            ep = GProp_GProps()
            brepgprop.LinearProperties(exp.Current(), ep)
            edge_lengths.append(abs(ep.Mass()))
            exp.Next()

        vertex_count = 0
        exp = TopExp_Explorer(shape, TopAbs_VERTEX)
        while exp.More():
            vertex_count += 1
            exp.Next()

        topo_hash = canonical_topology_hash(
            face_count, edge_count, vertex_count,
            face_areas, edge_lengths,
        )

        return {
            "bbox_min": (xmin, ymin, zmin),
            "bbox_max": (xmax, ymax, zmax),
            "volume_mm3": volume,
            "surface_area_mm2": surface_area,
            "topology_hash": topo_hash,
            "vertex_count": vertex_count,
            "face_count": face_count,
        }
    except Exception as e:
        logger.warning(f"Failed to compute shape fingerprint: {e}")
        return None


def canonical_topology_hash(face_count: int, edge_count: int, vertex_count: int,
                            face_areas: list[float] | None = None,
                            edge_lengths: list[float] | None = None) -> str:
    """Compute a canonical topology hash.

    Walks the shape in a deterministic order (face-by-area, edges-by-length)
    and produces a SHA256 hash. Same shape = same hash even after re-import.
    """
    parts = [f"F{face_count}", f"E{edge_count}", f"V{vertex_count}"]

    if face_areas:
        sorted_areas = sorted(face_areas, reverse=True)
        area_sig = ",".join(f"{a:.4f}" for a in sorted_areas[:50])
        parts.append(f"FA:{area_sig}")

    if edge_lengths:
        sorted_lengths = sorted(edge_lengths, reverse=True)
        length_sig = ",".join(f"{l:.4f}" for l in sorted_lengths[:50])
        parts.append(f"EL:{length_sig}")

    signature = "|".join(parts)
    return hashlib.sha256(signature.encode()).hexdigest()


def fingerprint_mesh(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """Compute fingerprint from mesh arrays. Wrapper around uuid_registry function."""
    from ..uuid_registry import compute_fingerprint_from_mesh
    fp = compute_fingerprint_from_mesh(vertices, faces)
    return fp.model_dump()


def compare_fingerprints(fp1: dict, fp2: dict, volume_tol: float = 0.001) -> dict:
    """Compare two fingerprints and return similarity metrics.

    Returns dict with: volume_diff, area_diff, bbox_diff, topo_match, confidence.
    """
    vol1, vol2 = fp1["volume_mm3"], fp2["volume_mm3"]
    area1, area2 = fp1["surface_area_mm2"], fp2["surface_area_mm2"]

    vol_diff = abs(vol1 - vol2) / max(vol1, vol2, 1e-10)
    area_diff = abs(area1 - area2) / max(area1, area2, 1e-10)

    topo_match = fp1["topology_hash"] == fp2["topology_hash"]

    # Compute confidence
    if topo_match and vol_diff <= volume_tol:
        confidence = 0.95
    elif topo_match:
        confidence = 0.80
    elif vol_diff <= volume_tol and area_diff <= volume_tol * 2:
        confidence = 0.85
    elif vol_diff <= volume_tol * 5:
        confidence = 0.60
    else:
        confidence = 0.0

    bbox_diff = max(
        abs(fp1["bbox_max"][i] - fp1["bbox_min"][i] - fp2["bbox_max"][i] + fp2["bbox_min"][i])
        / max(abs(fp1["bbox_max"][i] - fp1["bbox_min"][i]), 1e-10)
        for i in range(3)
    )

    return {
        "volume_diff": vol_diff,
        "area_diff": area_diff,
        "bbox_diff": bbox_diff,
        "topo_match": topo_match,
        "confidence": confidence,
    }
