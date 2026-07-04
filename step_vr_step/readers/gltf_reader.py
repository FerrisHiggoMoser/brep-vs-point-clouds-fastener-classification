"""glTF/GLB reader for mesh-based scene import.

Parses glTF 2.0 files via pygltflib, extracting:
- Scene hierarchy from node tree
- Mesh triangles from accessors/buffers
- PBR materials (base color, metallic, roughness, textures)
- extras metadata for round-trip UUID preservation
- Coordinate conversion: glTF (RH Y-up meters) → STEP (RH Z-up mm)
"""
from __future__ import annotations

import logging
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def read_gltf(filepath: str | Path) -> tuple:
    """Read a glTF/GLB file and return (compound_or_None, Manifest).

    Args:
        filepath: Path to .gltf or .glb file

    Returns:
        Tuple of (OCC compound if OCC available else None, Manifest)
    """
    import pygltflib
    from ..schema import (
        Manifest, PartEntry, Transform, Fingerprint, PBRMaterial,
        ProvenanceRecord, UnrealSpecific, BundleMetadata,
    )
    from ..uuid_registry import UUIDRegistry, compute_fingerprint_from_mesh

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"glTF file not found: {filepath}")

    logger.info(f"Reading glTF file: {filepath}")

    gltf = pygltflib.GLTF2().load(str(filepath))

    registry = UUIDRegistry()
    parts = []

    # Get the binary blob for buffer access
    # For .glb files, binary_blob() returns the embedded data.
    # For .gltf files with external .bin, we need to load it manually.
    blob = gltf.binary_blob()
    if blob is None and gltf.buffers and gltf.buffers[0].uri:
        bin_path = filepath.parent / gltf.buffers[0].uri
        if bin_path.exists():
            blob = bin_path.read_bytes()
            logger.info(f"Loaded external binary buffer: {bin_path} ({len(blob)} bytes)")

    # Build node index → parent mapping
    parent_map = {}
    for node_idx, node in enumerate(gltf.nodes):
        if node.children:
            for child_idx in node.children:
                parent_map[child_idx] = node_idx

    # Track node index → UUID mapping for parent references
    node_uuid_map = {}

    # Process all scenes
    scene = gltf.scenes[gltf.scene] if gltf.scene is not None else gltf.scenes[0]

    # Process nodes (first pass: assign UUIDs)
    for node_idx, node in enumerate(gltf.nodes):
        # Check extras for existing UUID from previous round-trip
        existing_uuid = None
        provenance_type = "unreal_native"
        extras = node.extras or {}

        if isinstance(extras, dict):
            uuid_str = extras.get("step_vr_step_uuid") or extras.get("part_uuid")
            if uuid_str:
                try:
                    existing_uuid = uuid.UUID(str(uuid_str))
                except ValueError:
                    pass

            prov = extras.get("provenance_type") or extras.get("source_type")
            if prov:
                provenance_type = prov

        part_uuid = existing_uuid or uuid.uuid4()
        node_uuid_map[node_idx] = part_uuid

    # Collect mesh data per node name for OCC shape building
    mesh_data_by_name: dict[str, tuple] = {}

    # Second pass: build parts
    for node_idx, node in enumerate(gltf.nodes):
        node_name = node.name or f"Node_{node_idx}"
        part_uuid = node_uuid_map[node_idx]

        # Parent UUID
        parent_uuid = None
        if node_idx in parent_map:
            parent_uuid = node_uuid_map.get(parent_map[node_idx])

        # Extract transform
        transform = _extract_node_transform(node)

        # Extract mesh and compute fingerprint
        fp = Fingerprint(
            bbox_min=(0, 0, 0), bbox_max=(0, 0, 0),
            volume_mm3=0, surface_area_mm2=0,
            topology_hash="no_mesh", vertex_count=0, face_count=0,
        )

        if node.mesh is not None:
            raw_mesh = _extract_mesh_data(gltf, node.mesh, blob)
            if raw_mesh is not None:
                verts, faces = raw_mesh
                # Convert coordinates: glTF Y-up meters → Z-up mm
                verts = _gltf_to_step_coords(verts)
                fp = compute_fingerprint_from_mesh(verts, faces)
                # Store for OCC shape building later
                mesh_data_by_name[node_name] = (verts, faces)

        # Extract material
        material = PBRMaterial(
            name=f"mat_{node_name}",
            base_color=(0.7, 0.7, 0.7, 1.0),
        )
        if node.mesh is not None and gltf.meshes[node.mesh].primitives:
            prim = gltf.meshes[node.mesh].primitives[0]
            if prim.material is not None and prim.material < len(gltf.materials):
                material = _extract_material(gltf, prim.material)

        # Extract extras metadata
        extras = node.extras or {}
        unreal_specific = UnrealSpecific()
        if isinstance(extras, dict):
            if "tags" in extras:
                unreal_specific.tags = extras["tags"]
            if "actor_class" in extras:
                unreal_specific.actor_class = extras["actor_class"]
            if "blueprint_path" in extras:
                unreal_specific.blueprint_path = extras["blueprint_path"]
            if "data_layers" in extras:
                unreal_specific.data_layers = extras["data_layers"]
            if "outliner_folder" in extras:
                unreal_specific.outliner_folder = extras["outliner_folder"]

        # Determine provenance
        prov_type = "unreal_native"
        if isinstance(extras, dict):
            prov_type = extras.get("source_type", "unreal_native")

        # Register UUID
        registry.register(
            name=node_name,
            fingerprint=fp,
            source_type=prov_type,
            existing_uuid=part_uuid,
        )

        entry = PartEntry(
            uuid=part_uuid,
            step_entity_id=f"gltf_node_{node_idx}",
            name=node_name,
            parent_uuid=parent_uuid,
            transform=transform,
            local_transform=transform,
            fingerprint=fp,
            material=material,
            provenance=ProvenanceRecord(
                source_type=prov_type,
                import_timestamp=datetime.now(timezone.utc),
            ),
            unreal=unreal_specific,
        )
        parts.append(entry)

    # Build manifest
    meta = BundleMetadata(
        created=datetime.now(timezone.utc),
        created_by="step-vr-step",
        app_version="1.0.0",
        source_format="gltf",
        coordinate_system="RH_Y_up_mm",
        units="mm",
    )

    manifest = Manifest(
        meta=meta,
        parts=parts,
        relationships=[],
    )

    logger.info(f"Read {len(parts)} parts from {filepath} ({len(mesh_data_by_name)} with meshes)")

    # Build OCC compound from mesh data so STEP writer can use it
    compound = _build_compound_from_meshes(mesh_data_by_name)

    return compound, manifest


def _build_compound_from_meshes(mesh_data_by_name: dict) -> object | None:
    """Build an OCC TopoDS_Compound from mesh triangle data.

    Uses the same mesh_to_brep approach as the POC's reverse pipeline:
    sew triangular faces into shells, then combine into a compound.
    """
    if not mesh_data_by_name:
        return None

    try:
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.TopoDS import TopoDS_Compound
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
        from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    except ImportError:
        logger.debug("OCC not available, cannot build compound from meshes")
        return None

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    built = 0

    for name, (vertices, faces) in mesh_data_by_name.items():
        try:
            sew = BRepBuilderAPI_Sewing(1.0)  # 1mm tolerance for mesh gaps

            for tri_idx in range(len(faces)):
                i0 = int(faces[tri_idx][0])
                i1 = int(faces[tri_idx][1])
                i2 = int(faces[tri_idx][2])
                p0 = gp_Pnt(float(vertices[i0][0]), float(vertices[i0][1]), float(vertices[i0][2]))
                p1 = gp_Pnt(float(vertices[i1][0]), float(vertices[i1][1]), float(vertices[i1][2]))
                p2 = gp_Pnt(float(vertices[i2][0]), float(vertices[i2][1]), float(vertices[i2][2]))

                if p0.IsEqual(p1, 1e-6) or p1.IsEqual(p2, 1e-6) or p0.IsEqual(p2, 1e-6):
                    continue

                try:
                    wire = BRepBuilderAPI_MakePolygon(p0, p1, p2, True)
                    if wire.IsDone():
                        face = BRepBuilderAPI_MakeFace(wire.Wire())
                        if face.IsDone():
                            sew.Add(face.Shape())
                except Exception:
                    continue

            sew.Perform()
            shape = sew.SewedShape()

            if shape is not None and not shape.IsNull():
                # Unify coplanar faces for cleaner geometry
                try:
                    unify = ShapeUpgrade_UnifySameDomain(shape)
                    unify.Build()
                    shape = unify.Shape()
                except Exception:
                    pass

                builder.Add(compound, shape)
                built += 1
                logger.debug(f"Built OCC shape for '{name}': {len(faces)} tris")

        except Exception as e:
            logger.warning(f"Failed to build OCC shape for '{name}': {e}")

    if built > 0:
        logger.info(f"Built OCC compound: {built}/{len(mesh_data_by_name)} meshes")
        return compound

    logger.warning("No OCC shapes could be built from mesh data")
    return None


def _extract_node_transform(node) -> Transform:
    """Extract Transform from a glTF node's TRS or matrix."""
    from ..schema import Transform

    if node.matrix is not None and node.matrix != [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]:
        m = np.array(node.matrix).reshape(4, 4).T  # glTF uses column-major
        translation = (float(m[0, 3]), float(m[1, 3]), float(m[2, 3]))
        # Extract rotation as quaternion from the 3x3 part
        rot_mat = m[:3, :3]
        # Remove scale
        sx = np.linalg.norm(rot_mat[:, 0])
        sy = np.linalg.norm(rot_mat[:, 1])
        sz = np.linalg.norm(rot_mat[:, 2])
        rot_mat[:, 0] /= sx if sx > 0 else 1
        rot_mat[:, 1] /= sy if sy > 0 else 1
        rot_mat[:, 2] /= sz if sz > 0 else 1
        quat = _rotation_matrix_to_quaternion(rot_mat)
        scale = (float(sx), float(sy), float(sz))

        # Convert translation from meters to mm
        translation = (translation[0] * 1000, translation[1] * 1000, translation[2] * 1000)

        return Transform(translation=translation, rotation_quat=quat, scale=scale)

    # TRS
    t = node.translation or [0, 0, 0]
    r = node.rotation or [0, 0, 0, 1]  # xyzw in glTF
    s = node.scale or [1, 1, 1]

    # Convert translation from meters to mm
    translation = (float(t[0]) * 1000, float(t[1]) * 1000, float(t[2]) * 1000)
    rotation = (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
    scale = (float(s[0]), float(s[1]), float(s[2]))

    return Transform(translation=translation, rotation_quat=rotation, scale=scale)


def _extract_mesh_data(gltf, mesh_idx, blob) -> Optional[tuple]:
    """Extract vertices and faces from a glTF mesh.

    Returns (vertices_Nx3, faces_Mx3) as numpy arrays, or None.
    """
    import pygltflib

    mesh = gltf.meshes[mesh_idx]
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for prim in mesh.primitives:
        if prim.mode is not None and prim.mode != 4:  # 4 = TRIANGLES
            continue

        # Get position accessor
        pos_accessor_idx = prim.attributes.POSITION
        if pos_accessor_idx is None:
            continue

        vertices = _read_accessor(gltf, pos_accessor_idx, blob)
        if vertices is None:
            continue

        # Get indices
        if prim.indices is not None:
            indices = _read_accessor(gltf, prim.indices, blob)
            if indices is not None:
                faces = indices.reshape(-1, 3) + vertex_offset
                all_faces.append(faces)

        all_vertices.append(vertices.reshape(-1, 3))
        vertex_offset += len(vertices.reshape(-1, 3))

    if not all_vertices:
        return None

    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0) if all_faces else np.zeros((0, 3), dtype=np.int32)

    return vertices, faces


def _read_accessor(gltf, accessor_idx, blob) -> Optional[np.ndarray]:
    """Read data from a glTF accessor."""
    import pygltflib

    accessor = gltf.accessors[accessor_idx]
    buffer_view = gltf.bufferViews[accessor.bufferView]

    # Component type mapping
    comp_types = {
        pygltflib.BYTE: (np.int8, 1),
        pygltflib.UNSIGNED_BYTE: (np.uint8, 1),
        pygltflib.SHORT: (np.int16, 2),
        pygltflib.UNSIGNED_SHORT: (np.uint16, 2),
        pygltflib.UNSIGNED_INT: (np.uint32, 4),
        pygltflib.FLOAT: (np.float32, 4),
    }

    if accessor.componentType not in comp_types:
        return None

    dtype, comp_size = comp_types[accessor.componentType]

    # Type to component count
    type_counts = {
        pygltflib.SCALAR: 1,
        pygltflib.VEC2: 2,
        pygltflib.VEC3: 3,
        pygltflib.VEC4: 4,
        pygltflib.MAT4: 16,
    }

    count = type_counts.get(accessor.type, 1)

    offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    length = accessor.count * count * comp_size

    if blob is None:
        return None

    data = np.frombuffer(blob, dtype=dtype, count=accessor.count * count, offset=offset)
    return data


def _extract_material(gltf, material_idx) -> PBRMaterial:
    """Extract PBR material from glTF."""
    from ..schema import PBRMaterial

    mat = gltf.materials[material_idx]
    name = mat.name or f"material_{material_idx}"

    base_color = (0.7, 0.7, 0.7, 1.0)
    metallic = 0.0
    roughness = 0.5
    emissive = (0.0, 0.0, 0.0)

    if mat.pbrMetallicRoughness:
        pbr = mat.pbrMetallicRoughness
        if pbr.baseColorFactor:
            bc = pbr.baseColorFactor
            base_color = (float(bc[0]), float(bc[1]), float(bc[2]), float(bc[3]) if len(bc) > 3 else 1.0)
        if pbr.metallicFactor is not None:
            metallic = float(pbr.metallicFactor)
        if pbr.roughnessFactor is not None:
            roughness = float(pbr.roughnessFactor)

    if mat.emissiveFactor:
        ef = mat.emissiveFactor
        emissive = (float(ef[0]), float(ef[1]), float(ef[2]))

    alpha_mode = "opaque"
    if mat.alphaMode:
        alpha_mode = mat.alphaMode.lower()

    return PBRMaterial(
        name=name,
        base_color=base_color,
        metallic=metallic,
        roughness=roughness,
        emissive=emissive,
        alpha_mode=alpha_mode,
        double_sided=bool(mat.doubleSided),
    )


def _gltf_to_step_coords(vertices: np.ndarray) -> np.ndarray:
    """Convert glTF coordinates (RH Y-up meters) to STEP (RH Z-up mm)."""
    converted = np.zeros_like(vertices)
    converted[:, 0] = vertices[:, 0] * 1000   # X stays, m → mm
    converted[:, 1] = -vertices[:, 2] * 1000   # glTF -Z → STEP Y
    converted[:, 2] = vertices[:, 1] * 1000    # glTF Y → STEP Z
    return converted


def _rotation_matrix_to_quaternion(m: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return (float(x), float(y), float(z), float(w))
