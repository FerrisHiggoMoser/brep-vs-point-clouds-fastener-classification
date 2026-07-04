"""Datasmith .udatasmith writer for Unreal Engine import.

Writes Datasmith format:
- .udatasmith XML file with actors, materials, metadata
- .udatasmith_content/ folder with mesh files (.udsmesh) and textures
- UUID embedded as custom metadata for round-trip tracking
"""
from __future__ import annotations

import logging
import shutil
import struct
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from xml.dom import minidom

import numpy as np

logger = logging.getLogger(__name__)


def write_datasmith(xde_doc, manifest, output_path: str | Path,
                    source_bundle_dir: str | Path | None = None) -> Path:
    """Write a Datasmith .udatasmith file + content directory.

    Args:
        xde_doc: XDE document handle (may be None)
        manifest: Manifest with all parts
        output_path: Output .udatasmith file path (or directory)
        source_bundle_dir: Optional source bundle for texture copying

    Returns:
        Path to the .udatasmith file
    """
    output_path = Path(output_path)

    if output_path.suffix != ".udatasmith":
        # If directory given, create file inside
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "scene.udatasmith"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Content directory alongside the .udatasmith file
    content_dir = output_path.parent / f"{output_path.stem}.udatasmith_content"
    content_dir.mkdir(parents=True, exist_ok=True)
    (content_dir / "Geometries").mkdir(exist_ok=True)
    (content_dir / "Textures").mkdir(exist_ok=True)

    logger.info(f"Writing Datasmith: {output_path}")

    # Build XML
    root = ET.Element("DatasmithUnrealScene")
    root.set("Version", manifest.meta.unreal_engine_version or "5.3")
    root.set("SDKVersion", "4.27")
    root.set("Application", "step-vr-step")
    root.set("ApplicationVersion", manifest.meta.app_version)

    # Write materials
    material_map = {}  # part_uuid → material name
    written_materials = set()

    for part in manifest.parts:
        mat = part.material
        mat_name = f"M_{mat.name}".replace(" ", "_")

        if mat_name not in written_materials:
            _write_material_xml(root, mat, mat_name)
            written_materials.add(mat_name)

        material_map[str(part.uuid)] = mat_name

    # Write actors
    for part in manifest.parts:
        mat_name = material_map.get(str(part.uuid), "M_default")
        mesh_ref = f"SM_{part.name}".replace(" ", "_")

        # Write mesh file
        _write_udsmesh_placeholder(content_dir / "Geometries" / f"{mesh_ref}.udsmesh", part)

        # Write actor XML
        actor = ET.SubElement(root, "StaticMeshActor")
        actor.set("Name", part.name.replace(" ", "_"))
        actor.set("Label", part.name)
        actor.set("Mesh", mesh_ref)

        if part.unreal.data_layers:
            actor.set("Layer", part.unreal.data_layers[0] if part.unreal.data_layers else "")
        if part.unreal.outliner_folder:
            actor.set("Folder", part.unreal.outliner_folder)

        # Transform — convert from STEP (RH Z-up mm) to UE (LH Z-up cm)
        t = part.transform
        trans_elem = ET.SubElement(actor, "Transform")
        trans_elem.set("tx", str(t.translation[0] / 10.0))    # mm → cm
        trans_elem.set("ty", str(-t.translation[1] / 10.0))   # handedness flip
        trans_elem.set("tz", str(t.translation[2] / 10.0))
        trans_elem.set("qx", str(t.rotation_quat[0]))
        trans_elem.set("qy", str(t.rotation_quat[1]))
        trans_elem.set("qz", str(t.rotation_quat[2]))
        trans_elem.set("qw", str(t.rotation_quat[3]))
        trans_elem.set("sx", str(t.scale[0]))
        trans_elem.set("sy", str(t.scale[1]))
        trans_elem.set("sz", str(t.scale[2]))

        # Material reference
        mat_ref = ET.SubElement(actor, "Material")
        mat_ref.set("Name", mat_name)

        # Tags
        for tag in part.unreal.tags:
            tag_elem = ET.SubElement(actor, "Tag")
            tag_elem.set("Value", tag)

        # Custom metadata — always include UUID for round-trip
        _add_metadata(actor, "step_vr_step_uuid", str(part.uuid))
        _add_metadata(actor, "step_vr_step_fingerprint", part.fingerprint.topology_hash)
        _add_metadata(actor, "step_vr_step_provenance", part.provenance.source_type)

        # Preserve original Datasmith provenance if available
        if part.provenance.original_step_path:
            _add_metadata(actor, "Datasmith_SourceCADPath", part.provenance.original_step_path)
        if part.provenance.original_entity_id:
            _add_metadata(actor, "Datasmith_SourceEntity", part.provenance.original_entity_id)
        if part.provenance.original_brep_hash:
            _add_metadata(actor, "Datasmith_SourceHash", part.provenance.original_brep_hash)

        # Pass through custom properties
        for key, value in part.unreal.custom_properties.items():
            if not key.startswith("step_vr_step_"):
                _add_metadata(actor, key, str(value))

    # Copy textures from source bundle if available
    if source_bundle_dir:
        src_textures = Path(source_bundle_dir) / "textures"
        if src_textures.exists():
            for tex_file in src_textures.rglob("*"):
                if tex_file.is_file():
                    dest = content_dir / "Textures" / tex_file.name
                    shutil.copy2(tex_file, dest)

    # Write XML
    xml_str = ET.tostring(root, encoding="unicode")
    # Pretty print
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ", encoding="utf-8")

    output_path.write_bytes(pretty_xml)

    logger.info(f"Wrote Datasmith with {len(manifest.parts)} actors to {output_path}")
    return output_path


def _write_material_xml(root: ET.Element, mat, mat_name: str) -> None:
    """Write a <Material> element to the Datasmith XML."""
    mat_elem = ET.SubElement(root, "Material")
    mat_elem.set("Name", mat_name)
    mat_elem.set("Label", mat.name)

    # Base color
    bc = mat.base_color
    _add_property(mat_elem, "BaseColor", f"({bc[0]:.4f},{bc[1]:.4f},{bc[2]:.4f},{bc[3]:.4f})")
    _add_property(mat_elem, "Metallic", f"{mat.metallic:.4f}")
    _add_property(mat_elem, "Roughness", f"{mat.roughness:.4f}")

    em = mat.emissive
    _add_property(mat_elem, "EmissiveColor", f"({em[0]:.4f},{em[1]:.4f},{em[2]:.4f})")

    # Texture references
    if mat.albedo_texture:
        _add_property(mat_elem, "DiffuseTexture", mat.albedo_texture)
    if mat.normal_texture:
        _add_property(mat_elem, "NormalTexture", mat.normal_texture)
    if mat.roughness_texture:
        _add_property(mat_elem, "RoughnessTexture", mat.roughness_texture)
    if mat.metallic_texture:
        _add_property(mat_elem, "MetallicTexture", mat.metallic_texture)


def _add_property(parent: ET.Element, name: str, value: str) -> None:
    """Add a <Property> element."""
    prop = ET.SubElement(parent, "Property")
    prop.set("Name", name)
    prop.set("Value", value)


def _add_metadata(parent: ET.Element, key: str, value: str) -> None:
    """Add a <MetaData> element."""
    meta = ET.SubElement(parent, "MetaData")
    meta.set("Key", key)
    meta.set("Value", value)


def _write_udsmesh_placeholder(mesh_path: Path, part) -> None:
    """Write a placeholder .udsmesh file.

    In production, this would contain the actual tessellated geometry.
    Format: vertex_count(u32) + index_count(u32) + vertices(f32*3) + indices(u32)
    """
    mesh_path.parent.mkdir(parents=True, exist_ok=True)

    # Write minimal placeholder with bounding box corners as vertices
    bb = part.fingerprint
    if bb.vertex_count > 0:
        # Write a simple box from bounding box
        vertices = _bbox_to_vertices(bb.bbox_min, bb.bbox_max)
        indices = np.array([
            0,1,2, 0,2,3,  # front
            4,5,6, 4,6,7,  # back
            0,4,7, 0,7,3,  # left
            1,5,6, 1,6,2,  # right
            3,2,6, 3,6,7,  # top
            0,1,5, 0,5,4,  # bottom
        ], dtype=np.uint32)
    else:
        vertices = np.zeros((0, 3), dtype=np.float32)
        indices = np.zeros(0, dtype=np.uint32)

    with open(mesh_path, "wb") as f:
        f.write(struct.pack("<I", len(vertices)))
        f.write(struct.pack("<I", len(indices)))
        f.write(vertices.astype(np.float32).tobytes())
        f.write(indices.tobytes())


def _bbox_to_vertices(bbox_min, bbox_max) -> np.ndarray:
    """Generate 8 box corner vertices from bounding box."""
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    return np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
