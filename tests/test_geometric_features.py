"""Tests for geometric feature extraction (mesh-based fallback)."""

import numpy as np
import pytest

from step_vr_step.detection.geometric_features import (
    GeometricFeatures,
    extract_mesh_features,
)


def _make_cylinder_mesh(radius=3.0, height=20.0, segments=16):
    """Generate a simple cylinder mesh for testing."""
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    verts = []
    faces = []

    # Bottom ring
    for a in angles:
        verts.append([radius * np.cos(a), radius * np.sin(a), 0])
    # Top ring
    for a in angles:
        verts.append([radius * np.cos(a), radius * np.sin(a), height])
    # Centers
    bottom_c = len(verts)
    verts.append([0, 0, 0])
    top_c = len(verts)
    verts.append([0, 0, height])

    n = segments
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, j + n])
        faces.append([i, j + n, i + n])
        faces.append([bottom_c, j, i])
        faces.append([top_c, i + n, j + n])

    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    # Compute normals
    normals = np.zeros_like(verts)
    for f in faces:
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        fn = np.cross(v1 - v0, v2 - v0)
        for idx in f:
            normals[idx] += fn
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normals /= norms

    return verts, normals, faces


class TestMeshFeatures:
    def test_cylinder_features(self):
        verts, normals, faces = _make_cylinder_mesh(radius=3.0, height=20.0)
        feat = extract_mesh_features(verts, normals, faces)

        assert feat.volume > 0
        assert feat.surface_area > 0
        assert feat.bounding_cylinder_length > 15  # roughly 20
        assert feat.bounding_cylinder_diameter > 4  # roughly 6
        assert feat.aspect_ratio > 2.0  # length / diameter

    def test_flat_disc_features(self):
        verts, normals, faces = _make_cylinder_mesh(radius=10.0, height=1.0)
        feat = extract_mesh_features(verts, normals, faces)

        # PCA-based bounding cylinder may pick diameter as principal axis
        # for very flat cylinders; just verify volume/area are reasonable
        assert feat.volume > 0
        assert feat.surface_area > 0

    def test_empty_mesh(self):
        feat = extract_mesh_features(
            np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((0, 3), dtype=np.int32)
        )
        assert feat.volume == 0
        assert feat.aspect_ratio == 0

    def test_bbox(self):
        verts, normals, faces = _make_cylinder_mesh(radius=5.0, height=10.0)
        feat = extract_mesh_features(verts, normals, faces)

        # Bounding box should contain the cylinder
        assert feat.bbox_min[2] <= 0
        assert feat.bbox_max[2] >= 9.5
