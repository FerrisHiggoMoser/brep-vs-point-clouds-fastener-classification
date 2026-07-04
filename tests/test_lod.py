"""Tests for LOD substitution system."""

import numpy as np
import pytest

from step_vr_step.lod.lod_levels import LODLevel, LOD_POLY_TARGETS, LOD_SCREEN_SIZES
from step_vr_step.lod.proxy_library import get_proxy


class TestLODLevels:
    def test_levels_ordered(self):
        assert LODLevel.L0 < LODLevel.L1 < LODLevel.L2

    def test_poly_targets_decrease(self):
        assert LOD_POLY_TARGETS[LODLevel.L0] > LOD_POLY_TARGETS[LODLevel.L1]
        assert LOD_POLY_TARGETS[LODLevel.L1] > LOD_POLY_TARGETS[LODLevel.L2]

    def test_screen_sizes_decrease(self):
        assert LOD_SCREEN_SIZES[LODLevel.L0] > LOD_SCREEN_SIZES[LODLevel.L1]
        assert LOD_SCREEN_SIZES[LODLevel.L1] > LOD_SCREEN_SIZES[LODLevel.L2]


class TestProxyLibrary:
    @pytest.mark.parametrize("fastener_type", [
        "hex_bolt", "socket_head_cap_screw", "hex_nut",
        "flat_washer", "rivet",
    ])
    def test_generates_valid_mesh(self, fastener_type):
        verts, faces = get_proxy(fastener_type, diameter_mm=6.0, length_mm=20.0)
        assert isinstance(verts, np.ndarray)
        assert isinstance(faces, np.ndarray)
        assert verts.shape[1] == 3
        assert faces.shape[1] == 3
        assert len(verts) > 0
        assert len(faces) > 0

    def test_l2_has_fewer_polys_than_l0(self):
        v0, f0 = get_proxy("hex_bolt", 8.0, 30.0, LODLevel.L0)
        v2, f2 = get_proxy("hex_bolt", 8.0, 30.0, LODLevel.L2)
        assert len(f2) <= len(f0)

    def test_face_indices_valid(self):
        verts, faces = get_proxy("hex_bolt", 6.0, 20.0, LODLevel.L1)
        assert faces.max() < len(verts), "Face index exceeds vertex count"
        assert faces.min() >= 0, "Negative face index"

    def test_scales_with_diameter(self):
        v_small, _ = get_proxy("hex_bolt", 3.0, 10.0)
        v_large, _ = get_proxy("hex_bolt", 12.0, 40.0)

        extent_small = v_small.max(axis=0) - v_small.min(axis=0)
        extent_large = v_large.max(axis=0) - v_large.min(axis=0)

        # Larger bolt should have larger extent in at least 2 dimensions
        assert sum(extent_large > extent_small) >= 2

    def test_unknown_type_falls_back_to_cylinder(self):
        verts, faces = get_proxy("unknown_fastener_type", 6.0, 20.0)
        assert len(verts) > 0
        assert len(faces) > 0

    def test_possible_prefix_handled(self):
        v1, f1 = get_proxy("possible_hex_bolt", 6.0, 20.0)
        v2, f2 = get_proxy("hex_bolt", 6.0, 20.0)
        # Same geometry since prefix is stripped
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(f1, f2)
