"""End-to-end round-trip tests."""

import json
import os
import tempfile
import datetime
import uuid as uuid_mod

from step_vr_step.schema import *
from step_vr_step.writers.step_writer import write_step_bundle
from step_vr_step.writers.gltf_writer import write_gltf
from step_vr_step.writers.datasmith_writer import write_datasmith
from step_vr_step.sidecar.manifest import read_manifest, write_manifest
from step_vr_step.sidecar.bundle import create_bundle, find_sibling_bundle


def _make_manifest_with_parts(n=3):
    now = datetime.datetime.now(datetime.timezone.utc)
    parts = []
    for i in range(n):
        parts.append(PartEntry(
            uuid=uuid_mod.uuid4(),
            step_entity_id=f"#{i+1}",
            name=f"Part_{i}",
            transform=Transform(translation=(i * 100, 0, 0), rotation_quat=(0, 0, 0, 1)),
            local_transform=Transform(translation=(i * 100, 0, 0), rotation_quat=(0, 0, 0, 1)),
            fingerprint=Fingerprint(
                bbox_min=(0, 0, 0), bbox_max=(10, 10, 10),
                volume_mm3=1000 + i * 100, surface_area_mm2=600,
                topology_hash=f"hash_{i}", vertex_count=8, face_count=12,
            ),
            material=PBRMaterial(name=f"mat_{i}", base_color=(0.5 + i * 0.1, 0.5, 0.5, 1.0)),
            provenance=ProvenanceRecord(source_type="original_step", import_timestamp=now),
            unreal=UnrealSpecific(tags=[f"tag_{i}"]),
        ))
    return Manifest(
        meta=BundleMetadata(created=now, created_by="test", app_version="1.0.0", source_format="step"),
        parts=parts, relationships=[],
    )


def test_manifest_roundtrip():
    """Manifest survives write -> read cycle."""
    manifest = _make_manifest_with_parts(5)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "manifest.json")
        write_manifest(manifest, path)
        loaded = read_manifest(path)

        assert len(loaded.parts) == 5
        assert loaded.parts[0].name == "Part_0"
        assert loaded.meta.app_version == "1.0.0"


def test_bundle_creation_and_open():
    """Bundle create -> open preserves manifest."""
    manifest = _make_manifest_with_parts(3)

    with tempfile.TemporaryDirectory() as td:
        bundle_dir = create_bundle(os.path.join(td, "test.bundle"), manifest)

        from step_vr_step.sidecar.bundle import open_bundle
        opened_dir, loaded = open_bundle(bundle_dir)

        assert len(loaded.parts) == 3


def test_step_to_datasmith_roundtrip():
    """STEP bundle -> Datasmith output preserves UUIDs."""
    manifest = _make_manifest_with_parts(2)
    original_uuids = {str(p.uuid) for p in manifest.parts}

    with tempfile.TemporaryDirectory() as td:
        # Write STEP bundle
        bundle_dir = write_step_bundle(None, manifest, os.path.join(td, "test.bundle"))

        # Write Datasmith
        ds_path = write_datasmith(None, manifest, os.path.join(td, "out.udatasmith"))

        # Check UUIDs preserved in Datasmith XML
        content = ds_path.read_bytes().decode("utf-8")
        for uid in original_uuids:
            assert uid in content, f"UUID {uid} not found in Datasmith output"


def test_sibling_bundle_detection():
    """find_sibling_bundle detects bundle folder next to STEP file."""
    with tempfile.TemporaryDirectory() as td:
        # Create a STEP file and sibling bundle
        step_path = os.path.join(td, "model.step")
        with open(step_path, "w") as f:
            f.write("ISO-10303-21;")

        bundle_path = os.path.join(td, "model.bundle")
        os.makedirs(bundle_path)
        with open(os.path.join(bundle_path, "manifest.json"), "w") as f:
            json.dump({"meta": {}, "parts": [], "relationships": []}, f)

        found = find_sibling_bundle(step_path)
        assert found is not None
        assert "model.bundle" in str(found)
