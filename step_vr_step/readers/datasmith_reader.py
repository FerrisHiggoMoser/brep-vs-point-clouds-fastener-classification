"""Datasmith .udatasmith reader for Unreal Engine scene import.

Parses the XML-based Datasmith format with companion mesh files (.udsmesh).
Extracts actors, materials, textures, metadata, and assembly hierarchy.
"""
from __future__ import annotations

import logging
import struct
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def read_datasmith(filepath: str | Path) -> tuple:
    """Read a Datasmith .udatasmith file and return (None, Manifest).

    Also reads companion mesh/texture files from the sibling content directory.
    """
    from ..schema import (
        Manifest, PartEntry, Transform, Fingerprint, PBRMaterial,
        ProvenanceRecord, UnrealSpecific, BundleMetadata, Relationship,
    )
    from ..uuid_registry import UUIDRegistry, compute_fingerprint_from_mesh

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Datasmith file not found: {filepath}")

    logger.info(f"Reading Datasmith file: {filepath}")

    # Content directory is typically <name>.udatasmith_content/
    content_dir = filepath.parent / f"{filepath.stem}.udatasmith_content"

    # Parse XML
    tree = ET.parse(str(filepath))
    root = tree.getroot()

    registry = UUIDRegistry()
    parts = []
    materials_map = {}  # name -> PBRMaterial

    # Parse materials first
    for mat_elem in root.iter("Material"):
        mat = _parse_material(mat_elem)
        if mat:
            materials_map[mat.name] = mat

    for mat_elem in root.iter("MasterMaterial"):
        mat = _parse_master_material(mat_elem)
        if mat:
            materials_map[mat.name] = mat

    # Parse actors
    for actor_elem in root.iter("Actor"):
        entry = _parse_actor(
            actor_elem, content_dir, registry, materials_map
        )
        if entry:
            parts.append(entry)

    # Also check for StaticMeshActor elements
    for actor_elem in root.iter("StaticMeshActor"):
        entry = _parse_actor(
            actor_elem, content_dir, registry, materials_map
        )
        if entry:
            parts.append(entry)

    # Build manifest
    ue_version = root.get("Version", None)

    meta = BundleMetadata(
        created=datetime.now(timezone.utc),
        created_by="step-vr-step",
        app_version="1.0.0",
        source_format="datasmith",
        unreal_engine_version=ue_version,
        coordinate_system="LH_Z_up_cm",
        units="mm",  # We convert to mm during import
    )

    manifest = Manifest(
        meta=meta,
        parts=parts,
        relationships=[],
    )

    logger.info(f"Read {len(parts)} actors from {filepath}")
    return None, manifest


def _parse_actor(actor_elem, content_dir, registry, materials_map) -> Optional[PartEntry]:
    """Parse a single actor element into a PartEntry."""
    from ..schema import (
        PartEntry, Transform, Fingerprint, PBRMaterial,
        ProvenanceRecord, UnrealSpecific,
    )
    from ..uuid_registry import compute_fingerprint_from_mesh

    name = actor_elem.get("Name", actor_elem.get("Label", "Unknown"))
    label = actor_elem.get("Label", name)
    layer = actor_elem.get("Layer", "")
    folder = actor_elem.get("Folder", "")

    # Extract transform
    transform = _parse_transform(actor_elem)

    # Extract mesh reference and compute fingerprint
    mesh_ref = actor_elem.get("Mesh", "")
    fp = Fingerprint(
        bbox_min=(0, 0, 0), bbox_max=(0, 0, 0),
        volume_mm3=0, surface_area_mm2=0,
        topology_hash="no_mesh", vertex_count=0, face_count=0,
    )

    if mesh_ref and content_dir and content_dir.exists():
        mesh_data = _read_udsmesh(content_dir, mesh_ref)
        if mesh_data is not None:
            verts, faces = mesh_data
            # Convert UE coords (LH Z-up cm) to RH Z-up mm
            verts = _ue_to_step_coords(verts)
            fp = compute_fingerprint_from_mesh(verts, faces)

    # Extract material
    material = PBRMaterial(name=f"mat_{name}", base_color=(0.7, 0.7, 0.7, 1.0))
    mat_refs = actor_elem.findall(".//Material")
    if mat_refs:
        mat_name = mat_refs[0].get("Name", "")
        if mat_name in materials_map:
            material = materials_map[mat_name]

    # Extract tags
    tags = []
    for tag_elem in actor_elem.findall(".//Tag"):
        tag_val = tag_elem.get("Value", tag_elem.text or "")
        if tag_val:
            tags.append(tag_val)

    # Extract custom metadata (DatasmithMetaData)
    custom_props = {}
    source_provenance = {}
    for meta_elem in actor_elem.findall(".//MetaData"):
        key = meta_elem.get("Key", meta_elem.get("Name", ""))
        value = meta_elem.get("Value", meta_elem.text or "")
        if key:
            custom_props[key] = value
            # Check for Datasmith source provenance
            if key.startswith("Datasmith_Source"):
                source_provenance[key] = value

    # Also check parent-level metadata elements
    for meta_elem in actor_elem.findall(".//Property"):
        key = meta_elem.get("Name", "")
        value = meta_elem.get("Value", meta_elem.text or "")
        if key:
            custom_props[key] = value

    # Check for existing UUID from previous round-trip
    existing_uuid = None
    uuid_str = custom_props.get("step_vr_step_uuid", "")
    if uuid_str:
        try:
            existing_uuid = uuid.UUID(uuid_str)
        except ValueError:
            pass

    part_uuid = existing_uuid or uuid.uuid4()

    # Determine provenance
    prov_type = "unreal_native"
    if "Datasmith_SourceCADPath" in source_provenance:
        prov_type = "original_step"

    original_step_path = source_provenance.get("Datasmith_SourceCADPath")
    original_entity_id = source_provenance.get("Datasmith_SourceEntity")
    original_brep_hash = source_provenance.get("Datasmith_SourceHash")

    registry.register(
        name=name,
        fingerprint=fp,
        source_type=prov_type,
        existing_uuid=part_uuid,
        original_brep_hash=original_brep_hash,
    )

    entry = PartEntry(
        uuid=part_uuid,
        step_entity_id=f"ds_{name}",
        name=label or name,
        parent_uuid=None,  # Datasmith hierarchy is flat; folders indicate structure
        transform=transform,
        local_transform=transform,
        fingerprint=fp,
        material=material,
        provenance=ProvenanceRecord(
            source_type=prov_type,
            original_step_path=original_step_path,
            original_entity_id=original_entity_id,
            original_brep_hash=original_brep_hash,
            import_timestamp=datetime.now(timezone.utc),
        ),
        unreal=UnrealSpecific(
            tags=tags,
            data_layers=[layer] if layer else [],
            outliner_folder=folder if folder else None,
            custom_properties=custom_props,
        ),
    )

    return entry


def _parse_transform(actor_elem) -> Transform:
    """Parse transform from actor XML element."""
    from ..schema import Transform

    # Datasmith stores transforms as child elements or attributes
    trans_elem = actor_elem.find("Transform")
    if trans_elem is not None:
        tx = float(trans_elem.get("tx", "0"))
        ty = float(trans_elem.get("ty", "0"))
        tz = float(trans_elem.get("tz", "0"))

        qx = float(trans_elem.get("qx", "0"))
        qy = float(trans_elem.get("qy", "0"))
        qz = float(trans_elem.get("qz", "0"))
        qw = float(trans_elem.get("qw", "1"))

        sx = float(trans_elem.get("sx", "1"))
        sy = float(trans_elem.get("sy", "1"))
        sz = float(trans_elem.get("sz", "1"))

        # Convert cm to mm
        return Transform(
            translation=(tx * 10, ty * 10, tz * 10),
            rotation_quat=(qx, qy, qz, qw),
            scale=(sx, sy, sz),
        )

    # Try alternative format: separate Translation/Rotation/Scale elements
    translation = (0.0, 0.0, 0.0)
    rotation = (0.0, 0.0, 0.0, 1.0)
    scale = (1.0, 1.0, 1.0)

    t = actor_elem.find("Translation")
    if t is not None:
        translation = (
            float(t.get("X", "0")) * 10,  # cm -> mm
            float(t.get("Y", "0")) * 10,
            float(t.get("Z", "0")) * 10,
        )

    r = actor_elem.find("Rotation")
    if r is not None:
        rotation = (
            float(r.get("X", "0")),
            float(r.get("Y", "0")),
            float(r.get("Z", "0")),
            float(r.get("W", "1")),
        )

    s = actor_elem.find("Scale")
    if s is not None:
        scale = (
            float(s.get("X", "1")),
            float(s.get("Y", "1")),
            float(s.get("Z", "1")),
        )

    return Transform(translation=translation, rotation_quat=rotation, scale=scale)


def _parse_material(mat_elem) -> Optional[PBRMaterial]:
    """Parse a <Material> element into PBRMaterial."""
    from ..schema import PBRMaterial

    name = mat_elem.get("Name", "unknown")

    base_color = (0.7, 0.7, 0.7, 1.0)
    metallic = 0.0
    roughness = 0.5

    # Look for color properties
    for prop in mat_elem.findall(".//Property"):
        pname = prop.get("Name", "")
        pvalue = prop.get("Value", prop.text or "")

        if pname == "BaseColor" and pvalue:
            parts = pvalue.strip("()").split(",")
            if len(parts) >= 3:
                base_color = tuple(float(p) for p in parts[:3]) + (1.0,)
        elif pname == "Metallic":
            metallic = float(pvalue) if pvalue else 0.0
        elif pname == "Roughness":
            roughness = float(pvalue) if pvalue else 0.5

    return PBRMaterial(
        name=name,
        base_color=base_color,
        metallic=metallic,
        roughness=roughness,
    )


def _parse_master_material(mat_elem) -> Optional[PBRMaterial]:
    """Parse a <MasterMaterial> element into PBRMaterial."""
    return _parse_material(mat_elem)


def _read_udsmesh(content_dir: Path, mesh_ref: str) -> Optional[tuple]:
    """Read a .udsmesh binary mesh file.

    Format: little-endian, header (magic, version, vertex_count, index_count),
    followed by vertex positions, indices, normals, UVs.

    Returns (vertices_Nx3, faces_Mx3) or None.
    """
    # Look for mesh file
    mesh_path = None
    candidates = [
        content_dir / f"{mesh_ref}.udsmesh",
        content_dir / "Geometries" / f"{mesh_ref}.udsmesh",
    ]

    for candidate in candidates:
        if candidate.exists():
            mesh_path = candidate
            break

    if mesh_path is None:
        logger.debug(f"Mesh file not found for ref: {mesh_ref}")
        return None

    try:
        with open(mesh_path, "rb") as f:
            data = f.read()

        if len(data) < 16:
            return None

        # Read header
        offset = 0

        # Try to detect format -- different Datasmith versions vary
        # Common format: uint32 vertex_count, uint32 index_count, then data
        vertex_count = struct.unpack_from("<I", data, 0)[0]
        index_count = struct.unpack_from("<I", data, 4)[0]
        offset = 8

        # Sanity check
        if vertex_count > 10_000_000 or index_count > 30_000_000:
            # Try with magic header skip
            offset = 16
            vertex_count = struct.unpack_from("<I", data, 8)[0]
            index_count = struct.unpack_from("<I", data, 12)[0]

        if vertex_count == 0 or vertex_count > 10_000_000:
            return None

        # Read vertices (3 floats each)
        vert_size = vertex_count * 3 * 4
        if offset + vert_size > len(data):
            return None
        vertices = np.frombuffer(data, dtype=np.float32, count=vertex_count * 3, offset=offset)
        vertices = vertices.reshape(-1, 3)
        offset += vert_size

        # Read indices (uint32)
        idx_size = index_count * 4
        if offset + idx_size > len(data):
            return None
        indices = np.frombuffer(data, dtype=np.uint32, count=index_count, offset=offset)
        faces = indices.reshape(-1, 3)

        return vertices, faces

    except Exception as e:
        logger.warning(f"Failed to read udsmesh {mesh_path}: {e}")
        return None


def _ue_to_step_coords(vertices: np.ndarray) -> np.ndarray:
    """Convert UE coordinates (LH Z-up cm) to STEP (RH Z-up mm).

    Handedness flip: negate Y. Unit scale: cm -> mm (*10).
    """
    converted = np.zeros_like(vertices)
    converted[:, 0] = vertices[:, 0] * 10    # X * 10
    converted[:, 1] = -vertices[:, 1] * 10   # -Y * 10 (handedness)
    converted[:, 2] = vertices[:, 2] * 10    # Z * 10
    return converted
