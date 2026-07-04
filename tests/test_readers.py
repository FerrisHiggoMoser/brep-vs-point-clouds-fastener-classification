"""Tests for all file readers."""

import json
import os
import tempfile

from step_vr_step.readers.step_reader import read_step
from step_vr_step.readers.gltf_reader import read_gltf
from step_vr_step.readers.datasmith_reader import read_datasmith
from step_vr_step.readers.unreal_bundle_reader import read_unreal_bundle


def test_step_reader_fallback():
    """STEP reader produces valid manifest in fallback mode."""
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
        f.write(b"ISO-10303-21; test")
        path = f.name

    try:
        doc, manifest = read_step(path)
        assert len(manifest.parts) >= 1
        assert manifest.meta.source_format == "step"
        assert manifest.parts[0].provenance.source_type == "original_step"
    finally:
        os.unlink(path)


def test_unreal_bundle_reader():
    """Unreal bundle reader parses all fields correctly."""
    with tempfile.TemporaryDirectory() as td:
        bundle = {
            "meta": {"engine_version": "5.3"},
            "parts": [{
                "name": "TestBracket",
                "uuid": "12345678-1234-1234-1234-123456789abc",
                "transform": {"translation": [10, 20, 30], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
                "materials": [{"name": "Steel", "base_color": [0.8, 0.8, 0.8, 1.0], "metallic": 0.9, "roughness": 0.3}],
                "class": "StaticMeshActor",
                "tags": ["structural"],
                "data_layers": ["Engineering"],
                "custom_properties": {"weight_kg": "2.5"},
            }],
        }
        path = os.path.join(td, "unreal_bundle.json")
        with open(path, "w") as f:
            json.dump(bundle, f)

        _, manifest = read_unreal_bundle(td)

        assert len(manifest.parts) == 1
        p = manifest.parts[0]
        assert p.name == "TestBracket"
        assert str(p.uuid) == "12345678-1234-1234-1234-123456789abc"
        assert p.unreal.actor_class == "StaticMeshActor"
        assert "structural" in p.unreal.tags
        assert p.material.metallic == 0.9


def test_datasmith_reader_with_provenance():
    """Datasmith reader extracts source provenance from metadata."""
    with tempfile.TemporaryDirectory() as td:
        xml = '''<?xml version="1.0"?>
<DatasmithUnrealScene Version="5.3">
  <StaticMeshActor Name="Panel" Label="Front Panel" Layer="Structure" Folder="Satellite/Body">
    <Transform tx="100" ty="200" tz="300" qx="0" qy="0" qz="0" qw="1" sx="1" sy="1" sz="1" />
    <Tag Value="thermal" />
    <MetaData Key="step_vr_step_uuid" Value="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" />
    <MetaData Key="Datasmith_SourceCADPath" Value="/archive/panel.step" />
  </StaticMeshActor>
</DatasmithUnrealScene>'''
        path = os.path.join(td, "test.udatasmith")
        with open(path, "w") as f:
            f.write(xml)

        _, manifest = read_datasmith(path)

        assert len(manifest.parts) == 1
        p = manifest.parts[0]
        assert p.name == "Front Panel"
        assert str(p.uuid) == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert p.provenance.source_type == "original_step"
        assert p.provenance.original_step_path == "/archive/panel.step"
