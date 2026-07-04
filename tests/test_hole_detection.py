"""Tests for hole clustering, axis-aware fastener matching, and the USD writer.

These tests build synthetic GeometricFeatures so they run without OCC and
without an actual STEP file. They cover the new logic added in
backend/step_vr_step/detection/holes.py and the rewritten
_infer_fastener_relationships in detect.py.
"""
from __future__ import annotations

import datetime
import uuid

import numpy as np
import pytest

from step_vr_step.config import DetectionConfig
from step_vr_step.detection.detect import (
    _infer_fastener_relationships,
    _infer_housing_relationships,
)
from step_vr_step.detection.geometric_features import (
    CylindricalFeature, GeometricFeatures,
)
from step_vr_step.detection.holes import detect_holes
from step_vr_step.schema import (
    BundleMetadata, DetectionLabel, Fingerprint, Manifest,
    PartEntry, PBRMaterial, ProvenanceRecord, Transform, UnrealSpecific,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _features_for_plate(
    bore_diameter: float,
    *,
    plate_thickness: float = 5.0,
    plate_size: float = 50.0,
    z_offset: float = 0.0,
) -> GeometricFeatures:
    """A flat plate with one internal cylindrical hole along +Z."""
    radius = bore_diameter / 2.0
    return GeometricFeatures(
        bbox_min=(-plate_size / 2, -plate_size / 2, z_offset),
        bbox_max=(plate_size / 2, plate_size / 2, z_offset + plate_thickness),
        volume=plate_size * plate_size * plate_thickness,
        surface_area=2 * plate_size * plate_size + 4 * plate_size * plate_thickness,
        face_type_counts={"plane": 6, "cylinder": 1, "cone": 0, "sphere": 0, "torus": 0, "nurbs": 0},
        cylindrical_face_radii=[radius],
        cylindrical_face_lengths=[plate_thickness],
        cylindrical_surface_ratio=0.10,
        num_faces=7, num_edges=18, num_vertices=12,
        aspect_ratio=plate_thickness / plate_size,
        has_thread=False,
        bounding_cylinder_diameter=plate_size,
        bounding_cylinder_length=plate_thickness,
        cylinders=[CylindricalFeature(
            axis_origin=(0.0, 0.0, z_offset),
            axis_direction=(0.0, 0.0, 1.0),
            radius=radius, length=plate_thickness,
            is_internal=True, face_index=0,
        )],
    )


def _features_for_bolt(shaft_diameter: float, length: float) -> GeometricFeatures:
    """A bolt along +Z with the shaft midpoint at the origin."""
    radius = shaft_diameter / 2.0
    return GeometricFeatures(
        bbox_min=(-radius, -radius, -length / 2),
        bbox_max=(radius, radius, length / 2),
        volume=3.14 * radius * radius * length,
        surface_area=2 * 3.14 * radius * length,
        face_type_counts={"plane": 2, "cylinder": 1, "cone": 0, "sphere": 0, "torus": 1, "nurbs": 0},
        cylindrical_face_radii=[radius],
        cylindrical_face_lengths=[length],
        cylindrical_surface_ratio=0.75,
        num_faces=5, num_edges=10, num_vertices=20,
        aspect_ratio=length / shaft_diameter,
        has_thread=True,
        bounding_cylinder_diameter=shaft_diameter,
        bounding_cylinder_length=length,
        head_diameter=shaft_diameter * 1.6,
        head_height=shaft_diameter * 0.7,
        cylinders=[CylindricalFeature(
            axis_origin=(0.0, 0.0, -length / 2),
            axis_direction=(0.0, 0.0, 1.0),
            radius=radius, length=length,
            is_internal=False, face_index=0,
        )],
    )


def _part(uid: str, name: str, parent: str | None = None) -> PartEntry:
    return PartEntry(
        uuid=uuid.UUID(uid),
        step_entity_id="#1",
        name=name,
        parent_uuid=uuid.UUID(parent) if parent else None,
        transform=Transform(translation=(0, 0, 0), rotation_quat=(0, 0, 0, 1)),
        local_transform=Transform(translation=(0, 0, 0), rotation_quat=(0, 0, 0, 1)),
        fingerprint=Fingerprint(
            bbox_min=(0, 0, 0), bbox_max=(1, 1, 1),
            volume_mm3=1.0, surface_area_mm2=6.0,
            topology_hash=name, vertex_count=8, face_count=6,
        ),
        material=PBRMaterial(name="m", base_color=(0.5, 0.5, 0.5, 1.0)),
        provenance=ProvenanceRecord(
            source_type="original_step",
            import_timestamp=datetime.datetime.now(datetime.timezone.utc),
        ),
        unreal=UnrealSpecific(),
    )


# ---------------------------------------------------------------------------
# detect_holes
# ---------------------------------------------------------------------------

class TestDetectHoles:
    def test_single_through_hole_in_plate(self):
        feat = _features_for_plate(bore_diameter=6.0)
        holes = detect_holes("plate-uuid", feat)

        assert len(holes) == 1
        h = holes[0]
        assert h.host_uuid == "plate-uuid"
        assert abs(h.diameter - 6.0) < 1e-6
        assert h.kind == "through"
        np.testing.assert_allclose(h.axis_direction, [0, 0, 1], atol=1e-6)

    def test_no_holes_when_no_internal_cylinders(self):
        feat = _features_for_plate(bore_diameter=6.0)
        # Flip every cylinder to external
        for c in feat.cylinders:
            c.is_internal = False
        assert detect_holes("plate-uuid", feat) == []

    def test_counterbore_classified(self):
        """A larger-radius cylinder axially stacked on a smaller one → counterbore."""
        plate = _features_for_plate(bore_diameter=6.0, plate_thickness=10.0)
        # Add a larger-radius cylinder at the top of the existing one,
        # same axis line.
        plate.cylinders.append(CylindricalFeature(
            axis_origin=(0.0, 0.0, 0.0),
            axis_direction=(0.0, 0.0, 1.0),
            radius=5.0,    # noticeably larger than the 3.0 bore radius
            length=3.0,
            is_internal=True,
            face_index=1,
        ))
        holes = detect_holes("plate-uuid", plate)
        assert len(holes) == 1
        assert holes[0].kind == "counterbore"
        # The reported diameter should be the inner (smaller) one.
        assert abs(holes[0].diameter - 6.0) < 1e-6


# ---------------------------------------------------------------------------
# Axis-aware matcher
# ---------------------------------------------------------------------------

class TestFastenerMatcher:
    def test_bolt_threads_through_two_plates(self):
        """M6×20 bolt: coarse clearance hole in plate-A (7.0mm, ISO 273 coarse),
        tap-drill hole in plate-B (5.0mm — minor diameter for M6×1.0 threads)."""
        # Plates stacked along +Z.
        plate_a = _features_for_plate(bore_diameter=7.0, plate_thickness=5.0, z_offset=0.0)
        plate_b = _features_for_plate(bore_diameter=5.0, plate_thickness=5.0, z_offset=5.0)

        # Bolt centered between the two plates' midpoint, axis along +Z.
        bolt = _features_for_bolt(shaft_diameter=6.0, length=20.0)
        # Shift the bolt's cylinder origin so its midpoint is at z=5.0
        # (i.e. between the two plates).
        bolt.cylinders[0].axis_origin = (0.0, 0.0, -5.0)
        bolt.cylinders[0].radius = 3.0  # shaft 6mm

        manifest = Manifest(
            meta=BundleMetadata(
                created=datetime.datetime.now(datetime.timezone.utc),
                created_by="test", app_version="t",
                source_format="step",
            ),
            parts=[
                _part("11111111-1111-1111-1111-111111111111", "plateA"),
                _part("22222222-2222-2222-2222-222222222222", "plateB"),
                _part("33333333-3333-3333-3333-333333333333", "bolt"),
            ],
            relationships=[],
        )
        manifest.parts[2].detection = DetectionLabel(
            fastener_type="hex_bolt", standard="ISO 4014", variant="M6x20",
            confidence=0.9, method="rule_based",
        )

        feature_map = {
            "11111111-1111-1111-1111-111111111111": plate_a,
            "22222222-2222-2222-2222-222222222222": plate_b,
            "33333333-3333-3333-3333-333333333333": bolt,
        }
        fastener_uuids = ["33333333-3333-3333-3333-333333333333"]
        structural_uuids = [
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
        ]

        rels = _infer_fastener_relationships(
            manifest, fastener_uuids, structural_uuids, feature_map,
        )

        assert len(rels) == 2, f"expected 2 host arcs, got {len(rels)}: {rels}"
        # Both arcs come from the same fastener.
        assert {str(r.subject_uuid) for r in rels} == set(fastener_uuids)
        # And point at distinct hosts.
        assert {str(r.target_uuid) for r in rels} == set(structural_uuids)
        # Bolt_order spans 0 and 1.
        orders = sorted(r.params["bolt_order"] for r in rels)
        assert orders == [0, 1]
        # The plate with the smaller-clearance hole is the tap fit.
        by_host = {str(r.target_uuid): r for r in rels}
        assert by_host["11111111-1111-1111-1111-111111111111"].params["fit_class"] == "clearance"
        assert by_host["22222222-2222-2222-2222-222222222222"].params["fit_class"] == "tap"

    def test_misaligned_axis_yields_no_match(self):
        plate = _features_for_plate(bore_diameter=6.5, plate_thickness=5.0)
        bolt = _features_for_bolt(shaft_diameter=6.0, length=20.0)
        # Rotate bolt axis 30° away from Z — should fail axis_align > 0.95.
        bolt.cylinders[0].axis_direction = (np.sin(np.deg2rad(30)), 0.0, np.cos(np.deg2rad(30)))

        manifest = Manifest(
            meta=BundleMetadata(
                created=datetime.datetime.now(datetime.timezone.utc),
                created_by="test", app_version="t", source_format="step",
            ),
            parts=[
                _part("11111111-1111-1111-1111-111111111111", "plate"),
                _part("33333333-3333-3333-3333-333333333333", "bolt"),
            ],
            relationships=[],
        )
        feature_map = {
            "11111111-1111-1111-1111-111111111111": plate,
            "33333333-3333-3333-3333-333333333333": bolt,
        }
        rels = _infer_fastener_relationships(
            manifest,
            ["33333333-3333-3333-3333-333333333333"],
            ["11111111-1111-1111-1111-111111111111"],
            feature_map,
        )
        assert rels == []


# ---------------------------------------------------------------------------
# Housing relationships
# ---------------------------------------------------------------------------

class TestHousingRelationships:
    def test_contained_in_emitted_for_fastener_with_parent(self):
        root_uid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        bolt_uid = "33333333-3333-3333-3333-333333333333"
        manifest = Manifest(
            meta=BundleMetadata(
                created=datetime.datetime.now(datetime.timezone.utc),
                created_by="test", app_version="t", source_format="step",
            ),
            parts=[
                _part(root_uid, "Assembly"),
                _part(bolt_uid, "bolt", parent=root_uid),
            ],
            relationships=[],
        )
        out = _infer_housing_relationships(manifest, [bolt_uid])
        assert len(out) == 1
        rel = out[0]
        assert rel.kind == "contained_in"
        assert str(rel.subject_uuid) == bolt_uid
        assert str(rel.target_uuid) == root_uid
        assert rel.inferred is False
        assert rel.params == {"depth": 1}

    def test_no_arc_when_fastener_has_no_parent(self):
        bolt_uid = "33333333-3333-3333-3333-333333333333"
        manifest = Manifest(
            meta=BundleMetadata(
                created=datetime.datetime.now(datetime.timezone.utc),
                created_by="test", app_version="t", source_format="step",
            ),
            parts=[_part(bolt_uid, "bolt")],
            relationships=[],
        )
        assert _infer_housing_relationships(manifest, [bolt_uid]) == []


# ---------------------------------------------------------------------------
# USD writer smoke test
# ---------------------------------------------------------------------------

class TestUsdEmitter:
    def test_writes_usd_with_relationships(self, tmp_path):
        from step_vr_step.exporters import USD_AVAILABLE
        if not USD_AVAILABLE:
            pytest.skip("usd-core not installed")
        from step_vr_step.exporters import write_usd
        from step_vr_step.schema import Relationship

        bolt_uid = "33333333-3333-3333-3333-333333333333"
        plate_uid = "11111111-1111-1111-1111-111111111111"
        root_uid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

        manifest = Manifest(
            meta=BundleMetadata(
                created=datetime.datetime.now(datetime.timezone.utc),
                created_by="test", app_version="t", source_format="step",
            ),
            parts=[
                _part(root_uid, "Assembly"),
                _part(plate_uid, "plate", parent=root_uid),
                _part(bolt_uid, "bolt", parent=root_uid),
            ],
            relationships=[
                Relationship(
                    kind="fastener",
                    subject_uuid=uuid.UUID(bolt_uid),
                    target_uuid=uuid.UUID(plate_uid),
                    params={"fit_class": "tap", "hole_diameter": 5.1, "bolt_order": 0},
                    inferred=True, confidence=0.92,
                ),
                Relationship(
                    kind="contained_in",
                    subject_uuid=uuid.UUID(bolt_uid),
                    target_uuid=uuid.UUID(root_uid),
                    params={"depth": 1},
                    inferred=False, confidence=1.0,
                ),
            ],
        )
        # Set a detection label so vrs:fastenerType is written.
        manifest.parts[2].detection = DetectionLabel(
            fastener_type="hex_bolt", standard="ISO 4014", variant="M6x20",
            confidence=0.9, method="rule_based",
        )

        out = tmp_path / "out.usda"
        write_usd(manifest, out, glb_path=None, fmt="usda")

        # Re-open the stage and check structure.
        from pxr import Usd, Sdf
        stage = Usd.Stage.Open(str(out))
        assert stage is not None
        # Three parts means three prims under /Root.
        bolt_prim = next(
            (p for p in stage.Traverse() if p.GetName() == "bolt"),
            None,
        )
        assert bolt_prim is not None, "bolt prim not found in USD"
        ftype = bolt_prim.GetAttribute("vrs:fastenerType")
        assert ftype.IsValid() and ftype.Get() == "hex_bolt"
        screwed_into = bolt_prim.GetRelationship("vrs:screwedInto")
        assert screwed_into.IsValid()
        targets = [str(t) for t in screwed_into.GetTargets()]
        assert any("plate" in t for t in targets)
        contained_in = bolt_prim.GetRelationship("vrs:containedIn")
        assert contained_in.IsValid()
