"""Tests for the reconciliation engine."""

import uuid
import datetime

from step_vr_step.schema import (
    Manifest, PartEntry, Transform, Fingerprint, PBRMaterial,
    ProvenanceRecord, UnrealSpecific, BundleMetadata, Relationship,
)
from step_vr_step.reconciliation.matcher import match
from step_vr_step.reconciliation.diff import build_diff
from step_vr_step.reconciliation.relationships import revalidate_relationships


def _make_part(uid, name, tx=0, vol=1000, topo="hash1"):
    now = datetime.datetime.now(datetime.timezone.utc)
    return PartEntry(
        uuid=uid, step_entity_id="#1", name=name,
        transform=Transform(translation=(tx, 0, 0), rotation_quat=(0, 0, 0, 1)),
        local_transform=Transform(translation=(tx, 0, 0), rotation_quat=(0, 0, 0, 1)),
        fingerprint=Fingerprint(
            bbox_min=(0, 0, 0), bbox_max=(10, 10, 10),
            volume_mm3=vol, surface_area_mm2=600,
            topology_hash=topo, vertex_count=8, face_count=12,
        ),
        material=PBRMaterial(name="mat", base_color=(0.7, 0.7, 0.7, 1)),
        provenance=ProvenanceRecord(source_type="original_step", import_timestamp=now),
        unreal=UnrealSpecific(tags=["test"]),
    )


def _make_manifest(*parts, rels=None):
    now = datetime.datetime.now(datetime.timezone.utc)
    return Manifest(
        meta=BundleMetadata(created=now, created_by="test", app_version="1.0.0", source_format="step"),
        parts=list(parts),
        relationships=rels or [],
    )


def test_uuid_matching():
    """Parts with same UUID are matched with confidence 1.0."""
    uid = uuid.UUID("11111111-1111-1111-1111-111111111111")
    old = _make_manifest(_make_part(uid, "Bracket"))
    new = _make_manifest(_make_part(uid, "Bracket"))

    result = match(new, old)
    assert len(result.matches) == 1
    assert result.matches[0][2] == 1.0
    assert len(result.new_parts) == 0
    assert len(result.deleted_parts) == 0


def test_detect_moved_part():
    """Part with same UUID but different transform is classified as 'moved'."""
    uid = uuid.UUID("11111111-1111-1111-1111-111111111111")
    old = _make_manifest(_make_part(uid, "Bracket", tx=0))
    new = _make_manifest(_make_part(uid, "Bracket", tx=50))

    result = match(new, old)
    diff = build_diff(new, old, result)

    assert diff["summary"]["moved"] == 1


def test_detect_reshaped_part():
    """Part with same UUID but different geometry is 'reshaped'."""
    uid = uuid.UUID("11111111-1111-1111-1111-111111111111")
    old = _make_manifest(_make_part(uid, "Bracket", vol=1000, topo="hash1"))
    new = _make_manifest(_make_part(uid, "Bracket", vol=800, topo="hash2"))

    result = match(new, old)
    diff = build_diff(new, old, result)

    assert diff["summary"]["reshaped"] == 1


def test_detect_added_deleted():
    """New parts appear as added, missing parts as deleted."""
    uid1 = uuid.UUID("11111111-1111-1111-1111-111111111111")
    uid2 = uuid.UUID("22222222-2222-2222-2222-222222222222")
    uid3 = uuid.UUID("33333333-3333-3333-3333-333333333333")

    old = _make_manifest(_make_part(uid1, "A"), _make_part(uid2, "B", vol=500, topo="hash_b"))
    new = _make_manifest(_make_part(uid1, "A"), _make_part(uid3, "C", vol=2000, topo="hash_c"))

    result = match(new, old)
    diff = build_diff(new, old, result)

    assert diff["summary"]["added"] == 1
    assert diff["summary"]["deleted"] == 1


def test_relationship_revalidation():
    """Fastener relationship stays valid when parts are close."""
    uid1 = uuid.UUID("11111111-1111-1111-1111-111111111111")
    uid2 = uuid.UUID("22222222-2222-2222-2222-222222222222")

    old = _make_manifest(
        _make_part(uid1, "Bracket"),
        _make_part(uid2, "Bolt"),
        rels=[Relationship(kind="fastener", subject_uuid=uid2, target_uuid=uid1, confidence=0.95)],
    )
    new = _make_manifest(
        _make_part(uid1, "Bracket"),
        _make_part(uid2, "Bolt"),
    )

    result = match(new, old)
    revalidate_relationships(old, new, result)

    assert not old.relationships[0].broken
