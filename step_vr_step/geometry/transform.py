"""Coordinate system conversions between Unreal Engine and STEP/CAD.

UE: Left-handed, Z-up, centimeters
STEP: Right-handed, Z-up, millimeters (our convention)
glTF: Right-handed, Y-up, meters

Key transforms:
- UE → STEP: negate Y (handedness), scale *10 (cm→mm)
- STEP → UE: negate Y (handedness), scale /10 (mm→cm)
- glTF → STEP: swap Y↔Z, scale *1000 (m→mm)
"""
from __future__ import annotations

import numpy as np


def ue_to_step_matrix() -> np.ndarray:
    """4x4 matrix converting UE coordinates to STEP coordinates.

    UE (LH Z-up cm) → STEP (RH Z-up mm):
    - X: *10 (cm→mm)
    - Y: *-10 (negate for handedness, cm→mm)
    - Z: *10 (cm→mm)
    """
    return np.array([
        [10,  0,  0, 0],
        [ 0, -10, 0, 0],
        [ 0,  0, 10, 0],
        [ 0,  0,  0, 1],
    ], dtype=np.float64)


def step_to_ue_matrix() -> np.ndarray:
    """4x4 matrix converting STEP coordinates to UE coordinates.

    STEP (RH Z-up mm) → UE (LH Z-up cm):
    - X: /10
    - Y: /-10 (negate for handedness, mm→cm)
    - Z: /10
    """
    return np.array([
        [0.1,  0,    0,   0],
        [0,   -0.1,  0,   0],
        [0,    0,    0.1, 0],
        [0,    0,    0,   1],
    ], dtype=np.float64)


def gltf_to_step_matrix() -> np.ndarray:
    """4x4 matrix converting glTF coordinates to STEP coordinates.

    glTF (RH Y-up m) → STEP (RH Z-up mm):
    - X → X * 1000
    - Y → Z * 1000 (glTF Y → STEP Z)
    - Z → -Y * 1000 (glTF Z → STEP -Y)
    """
    return np.array([
        [1000,  0,     0,    0],
        [0,     0,    -1000, 0],
        [0,     1000,  0,    0],
        [0,     0,     0,    1],
    ], dtype=np.float64)


def step_to_gltf_matrix() -> np.ndarray:
    """4x4 matrix converting STEP coordinates to glTF coordinates."""
    return np.array([
        [0.001,  0,      0,     0],
        [0,      0,      0.001, 0],
        [0,     -0.001,  0,     0],
        [0,      0,      0,     1],
    ], dtype=np.float64)


def transform_vertices(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform matrix to Nx3 vertices."""
    n = len(vertices)
    homogeneous = np.ones((n, 4), dtype=np.float64)
    homogeneous[:, :3] = vertices
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :3]


def transform_point(point: tuple[float, float, float],
                    matrix: np.ndarray) -> tuple[float, float, float]:
    """Transform a single 3D point."""
    p = np.array([point[0], point[1], point[2], 1.0])
    result = matrix @ p
    return (float(result[0]), float(result[1]), float(result[2]))


def flip_triangle_winding(faces: np.ndarray) -> np.ndarray:
    """Reverse triangle winding order (needed after handedness flip).

    Swaps column 1 and 2 of Mx3 face array to flip normals.
    """
    flipped = faces.copy()
    flipped[:, 1], flipped[:, 2] = faces[:, 2].copy(), faces[:, 1].copy()
    return flipped


def decompose_matrix(matrix: np.ndarray) -> tuple:
    """Decompose a 4x4 matrix into (translation, quaternion_xyzw, scale)."""
    translation = (float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]))

    rot = matrix[:3, :3].copy()
    sx = float(np.linalg.norm(rot[:, 0]))
    sy = float(np.linalg.norm(rot[:, 1]))
    sz = float(np.linalg.norm(rot[:, 2]))
    scale = (sx, sy, sz)

    if sx > 0: rot[:, 0] /= sx
    if sy > 0: rot[:, 1] /= sy
    if sz > 0: rot[:, 2] /= sz

    quat = _matrix_to_quaternion(rot)
    return translation, quat, scale


def _matrix_to_quaternion(m: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return (float(x), float(y), float(z), float(w))
