"""Tests for Pydantic schema models."""

import uuid
import datetime
import json

from step_vr_step.schema import (
    Manifest, PartEntry, Transform, Fingerprint, PBRMaterial,
    ProvenanceRecord, UnrealSpecific, BundleMetadata,
)


def test_transform_defaults():
    t = Transform(translation=(1, 2, 3), rotation_quat=(0, 0, 0, 1))
    assert t.scale == (1.0, 1.0, 1.0)


def test_fingerprint_fields():
    fp = Fingerprint(
        bbox_min=(0, 0, 0), bbox_max=(10, 10, 10),
        volume_mm3=1000, surface_area_mm2=600,
        topology_hash="test_hash", vertex_count=8, face_count=12,
    )
    assert fp.volume_mm3 == 1000
    assert fp.topology_hash == "test_hash"


def test_manifest_serialization(sample_manifest):
    data = sample_manifest.model_dump(mode="json")
    assert isinstance(data, dict)
    assert len(data["parts"]) == 3
    assert len(data["relationships"]) == 1

    # Round-trip through JSON
    json_str = json.dumps(data, default=str)
    loaded = json.loads(json_str)
    assert loaded["parts"][0]["name"] == "Bracket_A"


def test_part_entry_complete(sample_part):
    assert str(sample_part.uuid) == "11111111-1111-1111-1111-111111111111"
    assert sample_part.name == "Bracket_A"
    assert sample_part.provenance.source_type == "original_step"
    assert "structural" in sample_part.unreal.tags


def test_material_defaults():
    mat = PBRMaterial(name="default", base_color=(0.5, 0.5, 0.5, 1.0))
    assert mat.metallic == 0.0
    assert mat.roughness == 0.5
    assert mat.alpha_mode == "opaque"
