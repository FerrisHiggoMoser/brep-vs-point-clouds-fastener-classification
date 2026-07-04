"""Tests for all file writers."""

import os
import tempfile
import json

from step_vr_step.writers.step_writer import write_step_bundle
from step_vr_step.writers.gltf_writer import write_gltf
from step_vr_step.writers.datasmith_writer import write_datasmith


def test_step_bundle_structure(sample_manifest):
    """STEP bundle creates correct folder structure."""
    with tempfile.TemporaryDirectory() as td:
        bundle_dir = write_step_bundle(None, sample_manifest, os.path.join(td, "test.bundle"))

        assert (bundle_dir / "part.step").exists()
        assert (bundle_dir / "manifest.json").exists()
        assert (bundle_dir / "textures").is_dir()
        assert (bundle_dir / ".history").is_dir()
        assert (bundle_dir / "source_refs.json").exists()
        assert (bundle_dir / "README.txt").exists()


def test_step_file_contains_uuids(sample_manifest):
    """STEP file contains UUID property definitions."""
    with tempfile.TemporaryDirectory() as td:
        bundle_dir = write_step_bundle(None, sample_manifest, os.path.join(td, "test.bundle"))
        content = (bundle_dir / "part.step").read_text()
        assert "step_vr_step/part_uuid" in content


def test_gltf_writer(sample_manifest):
    """glTF writer produces valid file."""
    with tempfile.TemporaryDirectory() as td:
        path = write_gltf(None, sample_manifest, os.path.join(td, "test.glb"))
        assert path.exists()
        assert path.stat().st_size > 0


def test_datasmith_writer(sample_manifest):
    """Datasmith writer produces XML with UUID metadata."""
    with tempfile.TemporaryDirectory() as td:
        ds_path = write_datasmith(None, sample_manifest, os.path.join(td, "out.udatasmith"))
        content = ds_path.read_bytes().decode("utf-8")

        assert "step_vr_step_uuid" in content
        assert "Bracket_A" in content

        content_dir = ds_path.parent / f"{ds_path.stem}.udatasmith_content"
        assert content_dir.is_dir()
        assert (content_dir / "Geometries").is_dir()
