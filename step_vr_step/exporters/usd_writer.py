"""USD emitter — writes a Maya-importable .usd from a Pydantic Manifest.

The Pydantic ``Manifest`` is the single source of truth. This writer transcribes
it into a USD stage so the relationships graph travels into DCC tools
(Maya's ``mayaUsdPlugin`` reads ``UsdRelationship`` directly into the
node-attribute editor).

Mesh geometry is pulled from the previously-written ``.glb`` (or skipped if
no glb is supplied) to avoid duplicating tessellation logic. The geometry
contract is: glTF nodes are 1:1 with PartEntry, keyed by ``node.name == PartEntry.name``.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from ..schema import Manifest, PartEntry, Relationship

logger = logging.getLogger(__name__)


# Detect usd-core at import time so callers can gate cleanly.
try:
    from pxr import Usd, UsdGeom, Gf, Sdf, Vt  # noqa: F401
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False


# `kind` → USD relationship name. Prefix `vrs` keeps our attributes namespaced
# so they don't collide with stock USD or other DCC extensions.
_REL_NAMES = {
    "fastener": "vrs:screwedInto",
    "contained_in": "vrs:containedIn",
    "mate": "vrs:mate",
    "contact": "vrs:contact",
    "weld": "vrs:weld",
    "bond": "vrs:bond",
    "custom": "vrs:custom",
}


def write_usd(
    manifest: Manifest,
    output_path: str | Path,
    glb_path: Optional[str | Path] = None,
    *,
    fmt: Literal["usda", "usdc"] = "usdc",
) -> Path:
    """Write ``manifest`` to ``output_path`` as a USD file.

    Args:
        manifest: the Pydantic Manifest (same one fed to the glTF writer).
        output_path: target ``.usd`` / ``.usda`` / ``.usdc`` path. The extension
            is normalized to match ``fmt`` if it doesn't already.
        glb_path: optional path to the .glb that contains tessellated meshes.
            When supplied, mesh data is copied into UsdGeom.Mesh prims so the
            file is self-contained for Maya. When None, only hierarchy +
            relationships + custom attrs are written (geometry-less skeleton).
        fmt: ``"usdc"`` (binary, compact) or ``"usda"`` (ASCII, diff-friendly).

    Returns:
        Path to the written file.
    """
    if not USD_AVAILABLE:
        raise RuntimeError(
            "usd-core is not installed. Install it with `pip install usd-core` "
            "to enable USD export."
        )

    from pxr import Usd, UsdGeom, Gf, Sdf, Vt

    output_path = _normalize_ext(Path(output_path), fmt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a name → (positions, indices, normals) lookup from the glb.
    mesh_lookup: dict[str, _MeshData] = {}
    if glb_path is not None:
        mesh_lookup = _load_glb_meshes(Path(glb_path))
        logger.info("Loaded %d meshes from %s", len(mesh_lookup), glb_path)

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)  # mm

    # Allocate each PartEntry a stable, USD-legal prim path.
    uuid_to_path = _allocate_prim_paths(manifest)
    root_path = Sdf.Path("/Root")
    stage.DefinePrim(root_path, "Xform")
    stage.SetDefaultPrim(stage.GetPrimAtPath(root_path))

    # Pass 1: create Xform / Mesh prims and metadata.
    for part in manifest.parts:
        prim_path = uuid_to_path[str(part.uuid)]
        xform_prim = UsdGeom.Xform.Define(stage, prim_path)
        _apply_transform(xform_prim, part)
        _write_part_attrs(xform_prim.GetPrim(), part)

        mesh = mesh_lookup.get(part.name)
        if mesh is not None:
            _write_mesh_child(stage, prim_path, mesh)

    # Pass 2: write relationship arcs. We do this after every prim exists so
    # targets always resolve.
    for rel in manifest.relationships:
        _write_relationship(stage, uuid_to_path, rel)

    stage.GetRootLayer().Save()
    logger.info(
        "Wrote USD: %s (%d parts, %d relationships)",
        output_path, len(manifest.parts), len(manifest.relationships),
    )
    return output_path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _normalize_ext(path: Path, fmt: str) -> Path:
    target_ext = "." + fmt
    if path.suffix.lower() in (".usd", ".usda", ".usdc"):
        return path.with_suffix(target_ext)
    return path.with_suffix(target_ext)


_NAME_SCRUB = re.compile(r"[^A-Za-z0-9_]")


def _sanitize(name: str) -> str:
    """USD prim names must match [A-Za-z_][A-Za-z0-9_]*."""
    cleaned = _NAME_SCRUB.sub("_", name)
    if not cleaned or not cleaned[0].isalpha() and cleaned[0] != "_":
        cleaned = "p_" + cleaned
    return cleaned


def _allocate_prim_paths(manifest: Manifest) -> dict[str, "Sdf.Path"]:
    """Map every PartEntry.uuid to a USD prim path that mirrors the assembly tree.

    Falls back to a flat layout under /Root if parent_uuid chains are missing.
    """
    from pxr import Sdf

    by_uuid: dict[str, PartEntry] = {str(p.uuid): p for p in manifest.parts}

    # First, pick a unique prim name per uuid (collisions happen on sanitized
    # names because OCC sometimes repeats identical names across the tree).
    used: dict[str, int] = {}
    name_for: dict[str, str] = {}
    for uid, part in by_uuid.items():
        base = _sanitize(part.name or f"part_{uid[:8]}")
        suffix = used.get(base, 0)
        if suffix:
            name = f"{base}_{suffix}"
        else:
            name = base
        used[base] = suffix + 1
        name_for[uid] = name

    # Walk parent_uuid chains to build the path.
    paths: dict[str, Sdf.Path] = {}

    def _resolve(uid: str) -> Sdf.Path:
        if uid in paths:
            return paths[uid]
        part = by_uuid[uid]
        if part.parent_uuid is not None and str(part.parent_uuid) in by_uuid:
            parent_path = _resolve(str(part.parent_uuid))
        else:
            parent_path = Sdf.Path("/Root")
        p = parent_path.AppendChild(name_for[uid])
        paths[uid] = p
        return p

    for uid in by_uuid:
        _resolve(uid)
    return paths


def _apply_transform(xform_prim, part: PartEntry) -> None:
    """Translate + rotate (quat xyzw → quat wxyz) + scale, in that order."""
    from pxr import Gf, UsdGeom

    tx = part.local_transform
    op = UsdGeom.XformCommonAPI(xform_prim)
    op.SetTranslate(Gf.Vec3d(*tx.translation))

    # Convert quaternion xyzw → wxyz for USD, then to euler.
    x, y, z, w = tx.rotation_quat
    quat = Gf.Quatd(w, Gf.Vec3d(x, y, z))
    rot = Gf.Rotation(quat)
    euler = rot.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())
    op.SetRotate(Gf.Vec3f(float(euler[0]), float(euler[1]), float(euler[2])))

    op.SetScale(Gf.Vec3f(*tx.scale))


def _write_part_attrs(prim, part: PartEntry) -> None:
    """Stamp the PartEntry's metadata onto the prim as namespaced attributes.

    Maya's mayaUsdPlugin exposes these in the Attribute Editor under a
    collapsible 'Extra Attributes' section keyed by namespace.
    """
    from pxr import Sdf

    prim.CreateAttribute("vrs:uuid", Sdf.ValueTypeNames.String).Set(str(part.uuid))
    prim.CreateAttribute("vrs:stepEntityId", Sdf.ValueTypeNames.String).Set(
        part.step_entity_id or ""
    )

    if part.detection is not None:
        det = part.detection
        prim.CreateAttribute("vrs:fastenerType", Sdf.ValueTypeNames.String).Set(
            det.fastener_type
        )
        if det.standard:
            prim.CreateAttribute("vrs:standard", Sdf.ValueTypeNames.String).Set(det.standard)
        if det.variant:
            prim.CreateAttribute("vrs:variant", Sdf.ValueTypeNames.String).Set(det.variant)
        prim.CreateAttribute("vrs:confidence", Sdf.ValueTypeNames.Float).Set(
            float(det.confidence)
        )
        prim.CreateAttribute("vrs:detectionMethod", Sdf.ValueTypeNames.String).Set(
            det.method
        )

    # Full payload as customData so callers can recover everything if needed.
    # USD's VtDictionary doesn't accept None, so route through _jsonable which
    # drops null entries.
    prim.SetCustomDataByKey("step_vr_step:fingerprint",
                            _jsonable(part.fingerprint.model_dump(mode="json")))
    if part.detection is not None:
        prim.SetCustomDataByKey("step_vr_step:detection",
                                _jsonable(part.detection.model_dump(mode="json")))


def _write_relationship(stage, uuid_to_path: dict, rel: Relationship) -> None:
    """Add one `UsdRelationship` arc on the subject prim pointing at the target."""
    from pxr import Sdf

    subj = uuid_to_path.get(str(rel.subject_uuid))
    tgt = uuid_to_path.get(str(rel.target_uuid))
    if subj is None or tgt is None:
        logger.debug("Skipping relationship with unresolved prim: %s", rel)
        return

    rel_name = _REL_NAMES.get(rel.kind, "vrs:custom")
    prim = stage.GetPrimAtPath(subj)
    if not prim or not prim.IsValid():
        return

    # If we've already created this relationship (e.g. multiple bolt_order
    # entries on the same fastener), append the new target instead of clobbering.
    usd_rel = prim.GetRelationship(rel_name) or prim.CreateRelationship(rel_name)
    targets = list(usd_rel.GetTargets())
    targets.append(Sdf.Path(tgt))
    usd_rel.SetTargets(targets)

    # Per-arc params go into customData under a per-target key. Maya can
    # script-read this via the Python USD API; humans see it as JSON in the
    # Attribute Editor's customData section.
    if rel.params:
        slot = f"step_vr_step:rel:{rel.kind}:{rel.target_uuid}"
        prim.SetCustomDataByKey(slot, _jsonable(rel.params))


def _jsonable(obj):
    """Recursively coerce to USD-customData-friendly Python types.

    USD's VtDictionary rejects Python None entirely, so we drop keys whose
    values resolve to None (a dropped key is semantically equivalent for our
    consumers — model_dump emits None for unset Optional fields).
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            cleaned = _jsonable(v)
            if cleaned is None:
                continue
            out[str(k)] = cleaned
        return out
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj if v is not None]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)):
        return obj
    if obj is None:
        return None
    return str(obj)


# ---------------------------------------------------------------------------
# Mesh import from glb
# ---------------------------------------------------------------------------


from dataclasses import dataclass


@dataclass
class _MeshData:
    positions: np.ndarray  # (N, 3) float32
    indices: np.ndarray    # (M,) uint32
    normals: Optional[np.ndarray]  # (N, 3) float32 or None


def _load_glb_meshes(glb_path: Path) -> dict[str, _MeshData]:
    """Extract per-node mesh data from a glb. Maps node.name → _MeshData."""
    import pygltflib

    gltf = pygltflib.GLTF2().load(str(glb_path))
    if gltf is None or not gltf.nodes:
        return {}

    blob = gltf.binary_blob()
    if blob is None:
        return {}

    out: dict[str, _MeshData] = {}
    for node in gltf.nodes:
        if node.mesh is None:
            continue
        mesh = gltf.meshes[node.mesh]
        if not mesh.primitives:
            continue
        prim = mesh.primitives[0]  # one primitive per part in our writer

        positions = _read_accessor(gltf, blob, prim.attributes.POSITION)
        normals = (
            _read_accessor(gltf, blob, prim.attributes.NORMAL)
            if prim.attributes.NORMAL is not None else None
        )
        indices = _read_accessor(gltf, blob, prim.indices) if prim.indices is not None else None

        if positions is None or indices is None:
            continue

        out[node.name] = _MeshData(
            positions=positions.reshape(-1, 3).astype(np.float32),
            indices=indices.astype(np.uint32),
            normals=normals.reshape(-1, 3).astype(np.float32) if normals is not None else None,
        )
    return out


def _read_accessor(gltf, blob: bytes, accessor_index: int) -> Optional[np.ndarray]:
    """Decode one glTF accessor to a flat numpy array."""
    import pygltflib

    if accessor_index is None or accessor_index < 0:
        return None
    acc = gltf.accessors[accessor_index]
    bv = gltf.bufferViews[acc.bufferView]

    # glTF component-type code → numpy dtype.
    dtype_map = {
        5120: np.int8, 5121: np.uint8,
        5122: np.int16, 5123: np.uint16,
        5125: np.uint32, 5126: np.float32,
    }
    dtype = dtype_map.get(acc.componentType, np.float32)

    # glTF type code → component count per element.
    type_map = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    components = type_map.get(acc.type, 1)

    offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    count = acc.count * components
    raw = np.frombuffer(blob, dtype=dtype, count=count, offset=offset)
    return raw.copy()


def _write_mesh_child(stage, parent_path, mesh_data: _MeshData) -> None:
    """Create a UsdGeom.Mesh as a child of the Xform prim."""
    from pxr import UsdGeom, Sdf, Vt

    mesh_path = parent_path.AppendChild("mesh")
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    positions = mesh_data.positions
    indices = mesh_data.indices

    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(positions))
    n_tris = len(indices) // 3
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(n_tris, 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices.astype(np.int32)))

    if mesh_data.normals is not None and len(mesh_data.normals) == len(positions):
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(mesh_data.normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
