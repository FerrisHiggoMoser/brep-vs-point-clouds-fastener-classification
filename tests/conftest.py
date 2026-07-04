"""Shared test fixtures for step-vr-step backend tests."""

import uuid
import datetime
import pytest
import numpy as np

from step_vr_step.schema import (
    Manifest, PartEntry, Transform, Fingerprint, PBRMaterial,
    ProvenanceRecord, UnrealSpecific, BundleMetadata, Relationship, PMIEntry,
)


@pytest.fixture
def sample_fingerprint():
    return Fingerprint(
        bbox_min=(0, 0, 0),
        bbox_max=(50, 30, 10),
        volume_mm3=15000,
        surface_area_mm2=6200,
        topology_hash="abc123def456",
        vertex_count=8,
        face_count=12,
    )


@pytest.fixture
def sample_transform():
    return Transform(
        translation=(100, 200, 300),
        rotation_quat=(0, 0, 0, 1),
        scale=(1, 1, 1),
    )


@pytest.fixture
def sample_material():
    return PBRMaterial(
        name="Steel_Brushed",
        base_color=(0.8, 0.8, 0.8, 1.0),
        metallic=0.9,
        roughness=0.3,
    )


@pytest.fixture
def sample_part(sample_fingerprint, sample_transform, sample_material):
    return PartEntry(
        uuid=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        step_entity_id="#1",
        name="Bracket_A",
        transform=sample_transform,
        local_transform=sample_transform,
        fingerprint=sample_fingerprint,
        material=sample_material,
        provenance=ProvenanceRecord(
            source_type="original_step",
            import_timestamp=datetime.datetime.now(datetime.timezone.utc),
        ),
        unreal=UnrealSpecific(
            tags=["structural", "bracket"],
            data_layers=["Engineering"],
        ),
    )


@pytest.fixture
def sample_manifest(sample_part):
    part2 = sample_part.model_copy(update={
        "uuid": uuid.UUID("22222222-2222-2222-2222-222222222222"),
        "name": "Cover_B",
        "transform": Transform(translation=(200, 0, 0), rotation_quat=(0, 0, 0, 1)),
        "local_transform": Transform(translation=(200, 0, 0), rotation_quat=(0, 0, 0, 1)),
    })
    part3 = sample_part.model_copy(update={
        "uuid": uuid.UUID("33333333-3333-3333-3333-333333333333"),
        "name": "Bolt_C",
        "fingerprint": Fingerprint(
            bbox_min=(0, 0, 0), bbox_max=(5, 5, 20),
            volume_mm3=50, surface_area_mm2=120,
            topology_hash="bolt_hash", vertex_count=24, face_count=40,
        ),
    })
    return Manifest(
        meta=BundleMetadata(
            created=datetime.datetime.now(datetime.timezone.utc),
            created_by="test",
            app_version="1.0.0",
            source_format="step",
        ),
        parts=[sample_part, part2, part3],
        relationships=[
            Relationship(
                kind="fastener",
                subject_uuid=uuid.UUID("33333333-3333-3333-3333-333333333333"),
                target_uuid=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                confidence=0.95,
            ),
        ],
    )


@pytest.fixture
def box_mesh():
    """A simple box mesh for testing."""
    vertices = np.array([
        [0,0,0],[10,0,0],[10,10,0],[0,10,0],
        [0,0,10],[10,0,10],[10,10,10],[0,10,10],
    ], dtype=np.float64)
    faces = np.array([
        [0,1,2],[0,2,3],[4,5,6],[4,6,7],
        [0,1,5],[0,5,4],[2,3,7],[2,7,6],
        [0,3,7],[0,7,4],[1,2,6],[1,6,5],
    ], dtype=np.int32)
    return vertices, faces
