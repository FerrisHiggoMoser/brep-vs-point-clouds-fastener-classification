"""Reader for the Unreal plugin's native bundle format.

The Unreal plugin (extractor.py) produces a folder with:
- unreal_bundle.json: JSON with parts array and metadata
- meshes/<name>.bin: binary mesh buffers (vertices, indices, normals, UVs)
- textures/<name>.<ext>: texture files

This is the simplest input format -- zero ambiguity.
"""
from __future__ import annotations

import json
import logging
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def read_unreal_bundle(filepath: str | Path) -> tuple:
    """Read an Unreal bundle (from the UE plugin extractor) and return (None, Manifest).

    Args:
        filepath: Path to unreal_bundle.json or the directory containing it
    """
    from ..schema import (
        Manifest, PartEntry, Transform, Fingerprint, PBRMaterial,
        ProvenanceRecord, UnrealSpecific, BundleMetadata,
    )
    from ..uuid_registry import UUIDRegistry, compute_fingerprint_from_mesh

    filepath = Path(filepath)
    if filepath.is_dir():
        filepath = filepath / "unreal_bundle.json"

    if not filepath.exists():
        raise FileNotFoundError(f"Unreal bundle not found: {filepath}")

    bundle_dir = filepath.parent

    logger.info(f"Reading Unreal bundle: {filepath}")

    with open(filepath, "r") as f:
        bundle_data = json.load(f)

    registry = UUIDRegistry()
    parts = []

    meta_info = bundle_data.get("meta", {})
    parts_data = bundle_data.get("parts", [])

    for part_data in parts_data:
        entry = _parse_part(part_data, bundle_dir, registry)
        if entry:
            parts.append(entry)

    # Build manifest
    meta = BundleMetadata(
        created=datetime.now(timezone.utc),
        created_by="step-vr-step",
        app_version="1.0.0",
        source_format="unreal",
        unreal_engine_version=meta_info.get("engine_version"),
        coordinate_system="LH_Z_up_cm",
        units="mm",
    )

    manifest = Manifest(
        meta=meta,
        parts=parts,
        relationships=[],
    )

    logger.info(f"Read {len(parts)} parts from Unreal bundle")
    return None, manifest


def _parse_part(part_data: dict, bundle_dir: Path, registry) -> Optional[PartEntry]:
    """Parse a single part entry from the bundle JSON."""
    from ..schema import (
        PartEntry, Transform, Fingerprint, PBRMaterial,
        ProvenanceRecord, UnrealSpecific,
    )
    from ..uuid_registry import compute_fingerprint_from_mesh

    name = part_data.get("name", "Unknown")

    # UUID (may come from previous round-trip or be freshly assigned by extractor)
    existing_uuid = None
    uuid_str = part_data.get("uuid", "")
    if uuid_str:
        try:
            existing_uuid = uuid.UUID(uuid_str)
        except ValueError:
            pass
    part_uuid = existing_uuid or uuid.uuid4()

    # Transform
    transform_data = part_data.get("transform", {})
    t = transform_data.get("translation", [0, 0, 0])
    r = transform_data.get("rotation", [0, 0, 0, 1])
    s = transform_data.get("scale", [1, 1, 1])

    transform = Transform(
        translation=(float(t[0]) * 10, -float(t[1]) * 10, float(t[2]) * 10),  # UE->STEP
        rotation_quat=(float(r[0]), float(r[1]), float(r[2]), float(r[3])),
        scale=(float(s[0]), float(s[1]), float(s[2])),
    )

    # Mesh and fingerprint
    fp = Fingerprint(
        bbox_min=(0, 0, 0), bbox_max=(0, 0, 0),
        volume_mm3=0, surface_area_mm2=0,
        topology_hash="no_mesh", vertex_count=0, face_count=0,
    )

    mesh_ref = part_data.get("mesh_ref", "")
    if mesh_ref:
        mesh_data = _read_mesh_bin(bundle_dir / mesh_ref)
        if mesh_data is not None:
            verts, faces = mesh_data
            # Convert UE coords to STEP coords
            converted = np.zeros_like(verts)
            converted[:, 0] = verts[:, 0] * 10
            converted[:, 1] = -verts[:, 1] * 10
            converted[:, 2] = verts[:, 2] * 10
            fp = compute_fingerprint_from_mesh(converted, faces)

    # Materials
    materials_data = part_data.get("materials", [])
    material = PBRMaterial(name=f"mat_{name}", base_color=(0.7, 0.7, 0.7, 1.0))
    if materials_data:
        mat = materials_data[0]
        material = PBRMaterial(
            name=mat.get("name", f"mat_{name}"),
            base_color=tuple(mat.get("base_color", [0.7, 0.7, 0.7, 1.0])),
            metallic=float(mat.get("metallic", 0.0)),
            roughness=float(mat.get("roughness", 0.5)),
            emissive=tuple(mat.get("emissive", [0.0, 0.0, 0.0])),
            albedo_texture=mat.get("albedo_texture"),
            normal_texture=mat.get("normal_texture"),
            roughness_texture=mat.get("roughness_texture"),
            metallic_texture=mat.get("metallic_texture"),
        )

    # Provenance
    source_prov = part_data.get("source_provenance", {})
    prov_type = "unreal_native"
    if source_prov.get("Datasmith_SourceCADPath"):
        prov_type = "original_step"

    # Unreal-specific metadata
    unreal = UnrealSpecific(
        actor_class=part_data.get("class", "StaticMeshActor"),
        blueprint_path=part_data.get("blueprint_path"),
        tags=part_data.get("tags", []),
        data_layers=part_data.get("data_layers", []),
        outliner_folder=part_data.get("outliner_folder"),
        mobility=part_data.get("mobility", "Static"),
        lod_count=part_data.get("lod_count", 1),
        custom_properties=part_data.get("custom_properties", {}),
    )

    # Collision
    collision_data = part_data.get("collision", {})
    if collision_data:
        unreal.collision_geometry = collision_data.get("primitives", [])
        unreal.collision_profile = collision_data.get("profile")

    registry.register(
        name=name,
        fingerprint=fp,
        source_type=prov_type,
        existing_uuid=part_uuid,
        original_brep_hash=source_prov.get("Datasmith_SourceHash"),
    )

    entry = PartEntry(
        uuid=part_uuid,
        step_entity_id=f"ue_{name}",
        name=name,
        parent_uuid=None,
        transform=transform,
        local_transform=transform,
        fingerprint=fp,
        material=material,
        provenance=ProvenanceRecord(
            source_type=prov_type,
            original_step_path=source_prov.get("Datasmith_SourceCADPath"),
            original_entity_id=source_prov.get("Datasmith_SourceEntity"),
            original_brep_hash=source_prov.get("Datasmith_SourceHash"),
            import_timestamp=datetime.now(timezone.utc),
        ),
        unreal=unreal,
    )

    return entry


def _read_mesh_bin(mesh_path: Path) -> Optional[tuple]:
    """Read a binary mesh file produced by the UE extractor.

    Format: magic(4) + version(4) + vertex_count(4) + index_count(4)
            + vertices(float32 * vertex_count * 3)
            + indices(uint32 * index_count)
            + normals(float32 * vertex_count * 3)
            + uvs(float32 * vertex_count * 2)
    """
    if not mesh_path.exists():
        return None

    try:
        with open(mesh_path, "rb") as f:
            data = f.read()

        if len(data) < 16:
            return None

        # Read header
        magic = struct.unpack_from("<4s", data, 0)[0]
        version = struct.unpack_from("<I", data, 4)[0]
        vertex_count = struct.unpack_from("<I", data, 8)[0]
        index_count = struct.unpack_from("<I", data, 12)[0]
        offset = 16

        if vertex_count == 0 or vertex_count > 10_000_000:
            return None

        # Read vertices
        vert_size = vertex_count * 3 * 4
        vertices = np.frombuffer(data, dtype=np.float32, count=vertex_count * 3, offset=offset)
        vertices = vertices.reshape(-1, 3)
        offset += vert_size

        # Read indices
        idx_size = index_count * 4
        indices = np.frombuffer(data, dtype=np.uint32, count=index_count, offset=offset)
        faces = indices.reshape(-1, 3)

        return vertices, faces

    except Exception as e:
        logger.warning(f"Failed to read mesh bin {mesh_path}: {e}")
        return None
