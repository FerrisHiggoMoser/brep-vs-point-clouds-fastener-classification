"""Tests for the 7-check validation harness."""

import os
import tempfile
import json
import datetime

from step_vr_step.schema import *
from step_vr_step.writers.step_writer import write_step_bundle
from step_vr_step.validation.report import run_full_validation


def _make_test_bundle(tmp_dir):
    """Create a test bundle for validation."""
    import uuid as uuid_mod
    now = datetime.datetime.now(datetime.timezone.utc)

    part = PartEntry(
        uuid=uuid_mod.uuid4(), step_entity_id="#1", name="TestPart",
        transform=Transform(translation=(0, 0, 0), rotation_quat=(0, 0, 0, 1)),
        local_transform=Transform(translation=(0, 0, 0), rotation_quat=(0, 0, 0, 1)),
        fingerprint=Fingerprint(
            bbox_min=(0, 0, 0), bbox_max=(10, 10, 10),
            volume_mm3=1000, surface_area_mm2=600,
            topology_hash="test", vertex_count=8, face_count=12,
        ),
        material=PBRMaterial(name="default", base_color=(0.7, 0.7, 0.7, 1)),
        provenance=ProvenanceRecord(source_type="original_step", import_timestamp=now),
        unreal=UnrealSpecific(),
    )
    manifest = Manifest(
        meta=BundleMetadata(created=now, created_by="test", app_version="1.0.0", source_format="step"),
        parts=[part], relationships=[],
    )

    return write_step_bundle(None, manifest, os.path.join(tmp_dir, "test.bundle"))


def test_validation_runs_all_checks():
    """Validation harness runs all 7 checks."""
    with tempfile.TemporaryDirectory() as td:
        bundle_dir = _make_test_bundle(td)
        report = run_full_validation(bundle_dir)

        assert len(report.checks) == 7
        assert report.timestamp != ""


def test_validation_report_written():
    """Validation report is written to bundle directory."""
    with tempfile.TemporaryDirectory() as td:
        bundle_dir = _make_test_bundle(td)
        run_full_validation(bundle_dir)

        report_path = bundle_dir / "validation_report.json"
        assert report_path.exists()

        with open(report_path) as f:
            data = json.load(f)
        assert "checks" in data
        assert "all_passed" in data


def test_schema_validation_detects_bad_step():
    """Schema validation fails on invalid STEP file."""
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "bad.bundle"))
        # Write invalid STEP
        with open(os.path.join(td, "bad.bundle", "part.step"), "w") as f:
            f.write("NOT A VALID STEP FILE")
        # Write valid manifest
        with open(os.path.join(td, "bad.bundle", "manifest.json"), "w") as f:
            json.dump({"meta": {}, "parts": [], "relationships": []}, f)

        from step_vr_step.validation.schema import check_schema
        from pathlib import Path
        result = check_schema(Path(td) / "bad.bundle")
        assert not result.passed
