"""Pre-built low-poly proxy mesh library for common fastener types.

Generates parametric meshes (cylinders, hexagonal prisms, discs) at
each LOD level. Meshes are returned as (vertices, faces) tuples of
NumPy arrays.
"""

import math
from typing import Optional

import numpy as np

from .lod_levels import LODLevel, LOD_POLY_TARGETS


def get_proxy(
    fastener_type: str,
    diameter_mm: float = 6.0,
    length_mm: float = 20.0,
    lod_level: LODLevel = LODLevel.L1,
) -> tuple[np.ndarray, np.ndarray]:
    """Get or generate a proxy mesh for the given fastener type and size.

    Args:
        fastener_type: one of "hex_bolt", "socket_head_cap_screw", "hex_nut",
                       "flat_washer", "lock_washer", "rivet", etc.
        diameter_mm: shaft/bore diameter in mm.
        length_mm: overall length in mm.
        lod_level: desired LOD level.

    Returns:
        (vertices_Nx3, faces_Mx3) NumPy arrays (float32 / int32).
    """
    base_type = fastener_type.replace("possible_", "").replace("likely_", "")

    generators = {
        "hex_bolt": _generate_hex_bolt,
        "socket_head_cap_screw": _generate_cap_screw,
        "button_head_screw": _generate_cap_screw,
        "countersunk_socket_screw": _generate_cap_screw,
        "hex_nut": _generate_hex_nut,
        "thin_hex_nut": _generate_hex_nut,
        "flat_washer": _generate_washer,
        "chamfered_washer": _generate_washer,
        "lock_washer": _generate_washer,
        "rivet": _generate_rivet,
    }

    gen = generators.get(base_type, _generate_cylinder)
    return gen(diameter_mm, length_mm, lod_level)


# ---------------------------------------------------------------------------
# Parametric mesh generators
# ---------------------------------------------------------------------------

def _generate_hex_bolt(dia: float, length: float, lod: LODLevel):
    """Hex bolt: hexagonal prism head + cylindrical shaft."""
    if lod == LODLevel.L2:
        return _cylinder(dia / 2, length, segments=6)

    head_width = dia * 1.7
    head_height = dia * 0.7

    head_v, head_f = _hexagonal_prism(head_width / 2, head_height, top_z=length)
    shaft_segs = 12 if lod == LODLevel.L0 else 6
    shaft_v, shaft_f = _cylinder(dia / 2, length - head_height, segments=shaft_segs)

    return _merge_meshes([(head_v, head_f), (shaft_v, shaft_f)])


def _generate_cap_screw(dia: float, length: float, lod: LODLevel):
    """Socket head cap screw: cylindrical head + shaft."""
    if lod == LODLevel.L2:
        return _cylinder(dia / 2, length, segments=6)

    head_dia = dia * 1.6
    head_height = dia * 0.9
    segs = 16 if lod == LODLevel.L0 else 8

    head_v, head_f = _cylinder(head_dia / 2, head_height, segments=segs,
                                offset_z=length - head_height)
    shaft_v, shaft_f = _cylinder(dia / 2, length - head_height, segments=segs)

    return _merge_meshes([(head_v, head_f), (shaft_v, shaft_f)])


def _generate_hex_nut(dia: float, length: float, lod: LODLevel):
    """Hex nut: hexagonal prism with center bore."""
    width = dia * 1.7
    height = dia * 0.8
    if lod == LODLevel.L2:
        return _hexagonal_prism(width / 2, height)
    return _hexagonal_prism(width / 2, height)


def _generate_washer(dia: float, length: float, lod: LODLevel):
    """Washer: flat disc."""
    outer_r = dia
    thickness = dia * 0.2
    segs = 16 if lod == LODLevel.L0 else 6
    return _cylinder(outer_r, thickness, segments=segs)


def _generate_rivet(dia: float, length: float, lod: LODLevel):
    """Rivet: shaft + dome head."""
    if lod == LODLevel.L2:
        return _cylinder(dia / 2, length, segments=6)

    head_dia = dia * 1.8
    head_height = dia * 0.4
    segs = 12 if lod == LODLevel.L0 else 6

    head_v, head_f = _cylinder(head_dia / 2, head_height, segments=segs,
                                offset_z=length - head_height)
    shaft_v, shaft_f = _cylinder(dia / 2, length - head_height, segments=segs)

    return _merge_meshes([(head_v, head_f), (shaft_v, shaft_f)])


def _generate_cylinder(dia: float, length: float, lod: LODLevel):
    """Fallback: simple cylinder."""
    segs = 12 if lod == LODLevel.L0 else 6
    return _cylinder(dia / 2, length, segments=segs)


# ---------------------------------------------------------------------------
# Primitive builders
# ---------------------------------------------------------------------------

def _cylinder(
    radius: float, height: float, segments: int = 12, offset_z: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a capped cylinder mesh."""
    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False)

    # Bottom ring + top ring + 2 center vertices
    verts = []
    # Bottom ring
    for a in angles:
        verts.append([radius * math.cos(a), radius * math.sin(a), offset_z])
    # Top ring
    for a in angles:
        verts.append([radius * math.cos(a), radius * math.sin(a), offset_z + height])
    # Bottom center, top center
    bottom_center = len(verts)
    verts.append([0, 0, offset_z])
    top_center = len(verts)
    verts.append([0, 0, offset_z + height])

    verts = np.array(verts, dtype=np.float32)
    n = segments
    faces = []

    # Side quads (as triangles)
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, j + n])
        faces.append([i, j + n, i + n])

    # Bottom cap
    for i in range(n):
        j = (i + 1) % n
        faces.append([bottom_center, j, i])

    # Top cap
    for i in range(n):
        j = (i + 1) % n
        faces.append([top_center, i + n, j + n])

    return verts, np.array(faces, dtype=np.int32)


def _hexagonal_prism(
    radius: float, height: float, top_z: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a hexagonal prism mesh."""
    return _cylinder(radius, height, segments=6, offset_z=top_z)


def _merge_meshes(
    meshes: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Merge multiple (vertices, faces) into one mesh."""
    all_verts = []
    all_faces = []
    offset = 0

    for verts, faces in meshes:
        all_verts.append(verts)
        all_faces.append(faces + offset)
        offset += len(verts)

    return (
        np.concatenate(all_verts, axis=0),
        np.concatenate(all_faces, axis=0),
    )
