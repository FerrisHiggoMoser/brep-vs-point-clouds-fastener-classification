"""glTF 2.0 writer for mesh export with round-trip metadata.

Writes glTF/GLB files from manifest data, preserving:
- Assembly hierarchy as node tree
- Tessellated mesh geometry from OCC shapes
- PBR materials mapped to glTF PBR metallic-roughness
- Full PartEntry metadata in node extras for round-trip preservation
- UUIDs, provenance, Unreal-specific data in extras
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def write_gltf(xde_doc, manifest, output_path: str | Path,
               shapes: dict | None = None,
               tessellation_config=None) -> Path:
    """Write a glTF/GLB file from XDE document and manifest.

    Args:
        xde_doc: OCC shape (Compound) or XDE doc, or None
        manifest: Manifest with all parts and metadata
        output_path: Output .gltf or .glb file path
        shapes: Optional dict mapping part name -> TopoDS_Shape for tessellation
        tessellation_config: Optional TessellationConfig for mesh quality control

    Returns:
        Path to written file
    """
    import pygltflib

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing glTF file: {output_path}")

    # If we got an OCC compound but no shapes dict, extract individual shapes
    if shapes is None and xde_doc is not None:
        shapes = _extract_shapes_from_compound(xde_doc, manifest)

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[])],
        nodes=[],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[],
        materials=[],
    )

    binary_blob = bytearray()

    # Track UUID -> node index for hierarchy
    uuid_to_node = {}

    # Create default material
    gltf.materials.append(pygltflib.Material(
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[0.7, 0.7, 0.7, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.5,
        ),
        name="default",
    ))

    # First pass: create all nodes with meshes
    for idx, part in enumerate(manifest.parts):
        node = pygltflib.Node(name=part.name)

        # Shapes are tessellated with world transforms baked into vertices,
        # so node transforms stay identity. This avoids double-transform issues.

        # Create material for this part
        mat = part.material
        gltf_mat = pygltflib.Material(
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=list(mat.base_color),
                metallicFactor=mat.metallic,
                roughnessFactor=mat.roughness,
            ),
            emissiveFactor=list(mat.emissive),
            name=mat.name,
            doubleSided=mat.double_sided,
        )
        if mat.alpha_mode != "opaque":
            gltf_mat.alphaMode = mat.alpha_mode.upper()

        mat_index = len(gltf.materials)
        gltf.materials.append(gltf_mat)

        # Tessellate shape and add mesh if available
        shape = shapes.get(part.name) if shapes else None
        if shape is not None:
            mesh_index = _add_tessellated_mesh(
                gltf, binary_blob, shape, part.name, mat_index,
                tessellation_config=tessellation_config,
            )
            if mesh_index is not None:
                node.mesh = mesh_index

        # Set extras with full round-trip metadata
        node.extras = _build_extras(part)

        node_index = len(gltf.nodes)
        gltf.nodes.append(node)
        uuid_to_node[str(part.uuid)] = node_index

    # Second pass: set up hierarchy
    root_nodes = []
    for part in manifest.parts:
        node_index = uuid_to_node.get(str(part.uuid))
        if node_index is None:
            continue

        if part.parent_uuid is not None:
            parent_index = uuid_to_node.get(str(part.parent_uuid))
            if parent_index is not None:
                pnode = gltf.nodes[parent_index]
                if pnode.children is None:
                    pnode.children = []
                pnode.children.append(node_index)
                continue

        root_nodes.append(node_index)

    gltf.scenes[0].nodes = root_nodes

    # Set buffer
    if binary_blob:
        gltf.buffers.append(pygltflib.Buffer(byteLength=len(binary_blob)))
        gltf.set_binary_blob(bytes(binary_blob))

    # Save
    gltf.save(str(output_path))

    mesh_count = sum(1 for n in gltf.nodes if n.mesh is not None)
    logger.info(f"Wrote glTF with {len(manifest.parts)} nodes ({mesh_count} with meshes) to {output_path}")
    return output_path


def _extract_shapes_from_compound(xde_doc, manifest) -> dict:
    """Extract individual shapes from an OCC compound, mapped by part name."""
    shapes = {}

    try:
        from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_COMPOUND

        if not isinstance(xde_doc, (TopoDS_Compound, TopoDS_Shape)):
            return shapes

        # Extract solids from the compound
        solids = []
        exp = TopExp_Explorer(xde_doc, TopAbs_SOLID)
        while exp.More():
            solids.append(exp.Current())
            exp.Next()

        # If no solids, try shells
        if not solids:
            exp = TopExp_Explorer(xde_doc, TopAbs_SHELL)
            while exp.More():
                solids.append(exp.Current())
                exp.Next()

        # Map shapes to manifest parts (skip the root assembly node)
        part_names = [p.name for p in manifest.parts if p.parent_uuid is not None]
        for i, shape in enumerate(solids):
            if i < len(part_names):
                shapes[part_names[i]] = shape

        logger.info(f"Extracted {len(shapes)} shapes from compound for glTF tessellation")
    except ImportError:
        logger.debug("OCC not available for shape extraction")
    except Exception as e:
        logger.warning(f"Failed to extract shapes from compound: {e}")

    return shapes


def _add_tessellated_mesh(gltf, binary_blob: bytearray, shape,
                          name: str, material_index: int,
                          tessellation_config=None) -> int | None:
    """Tessellate an OCC shape and add it as a glTF mesh. Returns mesh index or None."""
    import pygltflib
    from ..config import TessellationConfig

    if tessellation_config is None:
        tessellation_config = TessellationConfig()

    try:
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.BRepTools import breptools
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCC.Core.TopLoc import TopLoc_Location
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopoDS import topods

        # Clear any cached triangulation so we re-mesh at the requested density.
        # Without this, OCC reuses whatever mesh was baked in during read/import
        # and ignores our deflection parameters entirely.
        breptools.Clean(shape)

        mesh = BRepMesh_IncrementalMesh(
            shape,
            tessellation_config.linear_deflection,
            tessellation_config.relative,
            tessellation_config.angular_deflection,
            tessellation_config.parallel,
        )
        mesh.Perform()
        if not mesh.IsDone():
            return None

        all_verts = []
        all_normals = []
        all_indices = []
        vert_offset = 0

        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = topods.Face(exp.Current())
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation(face, loc)

            if tri is None:
                exp.Next()
                continue

            reversed_face = face.Orientation() == TopAbs_REVERSED
            nb_nodes = tri.NbNodes()
            nb_tris = tri.NbTriangles()

            # Extract vertices — apply location transform, keep in STEP coords (mm)
            for i in range(1, nb_nodes + 1):
                pnt = tri.Node(i)
                if not loc.IsIdentity():
                    pnt = pnt.Transformed(loc.Transformation())
                # Convert STEP (Z-up, mm) to glTF (Y-up, meters)
                all_verts.append([pnt.X() / 1000.0, pnt.Z() / 1000.0, -pnt.Y() / 1000.0])

            # Extract normals — flip for reversed faces
            has_normals = tri.HasNormals()
            for i in range(1, nb_nodes + 1):
                if has_normals:
                    n = tri.Normal(i)
                    if not loc.IsIdentity():
                        n = n.Transformed(loc.Transformation())
                    nx, ny, nz = n.X(), n.Z(), -n.Y()  # to glTF coords
                    if reversed_face:
                        nx, ny, nz = -nx, -ny, -nz
                    all_normals.append([nx, ny, nz])
                else:
                    all_normals.append([0, 1, 0])

            # Extract triangles — flip winding for reversed faces
            for i in range(1, nb_tris + 1):
                t = tri.Triangle(i)
                n1, n2, n3 = t.Get()
                if reversed_face:
                    n2, n3 = n3, n2
                all_indices.extend([n1 - 1 + vert_offset, n2 - 1 + vert_offset, n3 - 1 + vert_offset])

            vert_offset += nb_nodes
            exp.Next()

        if not all_verts:
            return None

        vertices = np.array(all_verts, dtype=np.float32)
        normals = np.array(all_normals, dtype=np.float32)
        indices = np.array(all_indices, dtype=np.uint32)

        # Recompute normals if they're all placeholder zeros
        if not np.any(normals != 0):
            normals = _compute_vertex_normals(vertices, indices.reshape(-1, 3))

        # Pad blob to 4-byte boundary
        while len(binary_blob) % 4 != 0:
            binary_blob.append(0)

        # Write indices
        idx_offset = len(binary_blob)
        idx_bytes = indices.tobytes()
        binary_blob.extend(idx_bytes)

        while len(binary_blob) % 4 != 0:
            binary_blob.append(0)

        # Write vertices
        vert_offset_blob = len(binary_blob)
        vert_bytes = vertices.tobytes()
        binary_blob.extend(vert_bytes)

        while len(binary_blob) % 4 != 0:
            binary_blob.append(0)

        # Write normals
        norm_offset = len(binary_blob)
        norm_bytes = normals.tobytes()
        binary_blob.extend(norm_bytes)

        # Buffer views
        idx_bv = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=idx_offset, byteLength=len(idx_bytes),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        ))
        vert_bv = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=vert_offset_blob, byteLength=len(vert_bytes),
            byteStride=12, target=pygltflib.ARRAY_BUFFER,
        ))
        norm_bv = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=norm_offset, byteLength=len(norm_bytes),
            byteStride=12, target=pygltflib.ARRAY_BUFFER,
        ))

        # Accessors
        v_min = vertices.min(axis=0).tolist()
        v_max = vertices.max(axis=0).tolist()

        idx_acc = len(gltf.accessors)
        gltf.accessors.append(pygltflib.Accessor(
            bufferView=idx_bv, componentType=pygltflib.UNSIGNED_INT,
            count=len(indices), type=pygltflib.SCALAR,
            max=[int(indices.max())], min=[int(indices.min())],
        ))
        vert_acc = len(gltf.accessors)
        gltf.accessors.append(pygltflib.Accessor(
            bufferView=vert_bv, componentType=pygltflib.FLOAT,
            count=len(vertices), type=pygltflib.VEC3,
            max=v_max, min=v_min,
        ))
        norm_acc = len(gltf.accessors)
        gltf.accessors.append(pygltflib.Accessor(
            bufferView=norm_bv, componentType=pygltflib.FLOAT,
            count=len(normals), type=pygltflib.VEC3,
        ))

        # Mesh
        gltf_mesh = pygltflib.Mesh(
            name=name,
            primitives=[pygltflib.Primitive(
                attributes=pygltflib.Attributes(
                    POSITION=vert_acc,
                    NORMAL=norm_acc,
                ),
                indices=idx_acc,
                material=material_index,
            )],
        )
        mesh_index = len(gltf.meshes)
        gltf.meshes.append(gltf_mesh)

        logger.debug(f"Tessellated '{name}': {len(vertices)} verts, {len(indices)//3} tris")
        return mesh_index

    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"Failed to tessellate '{name}': {e}")
        return None


def _build_extras(part) -> dict:
    """Build the extras dict for a glTF node, carrying full round-trip metadata."""
    extras = {
        "step_vr_step_uuid": str(part.uuid),
        "step_entity_id": part.step_entity_id,
        "source_type": part.provenance.source_type,
        "fingerprint_hash": part.fingerprint.topology_hash,
    }

    if part.unreal:
        extras["actor_class"] = part.unreal.actor_class
        extras["blueprint_path"] = part.unreal.blueprint_path
        extras["tags"] = part.unreal.tags
        extras["data_layers"] = part.unreal.data_layers
        extras["outliner_folder"] = part.unreal.outliner_folder
        extras["collision_profile"] = part.unreal.collision_profile
        extras["mobility"] = part.unreal.mobility
        extras["lod_count"] = part.unreal.lod_count
        extras["custom_properties"] = part.unreal.custom_properties

    extras["provenance"] = {
        "source_type": part.provenance.source_type,
        "original_step_path": part.provenance.original_step_path,
        "original_entity_id": part.provenance.original_entity_id,
        "original_brep_hash": part.provenance.original_brep_hash,
        "import_timestamp": str(part.provenance.import_timestamp),
    }

    if part.detection is not None:
        extras["detection"] = part.detection.model_dump()

    if part.pmi:
        extras["pmi"] = [
            {"kind": p.kind, "value": p.value, "tolerance": p.tolerance,
             "target_face_id": p.target_face_id}
            for p in part.pmi
        ]

    return extras


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute smooth per-vertex normals by averaging adjacent face normals."""
    normals = np.zeros_like(vertices)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normals /= norms

    return normals.astype(np.float32)
