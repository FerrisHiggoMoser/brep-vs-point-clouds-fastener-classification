"""STEP file reader using PythonOCC XDE for full metadata extraction.

Reads STEP files via STEPCAFControl_Reader with XDE document, extracting:
- Assembly hierarchy (labels + sub-labels)
- Transforms (TopLoc_Location -> translation/rotation/scale)
- Colors (XCAFDoc_ColorTool)
- Names (TDataStd_Name)
- Custom properties including step_vr_step UUIDs
- Geometry fingerprints (bounding box, volume, topology hash)
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def read_step(filepath: str | Path, assign_uuids: bool = True,
              return_shapes: bool = False) -> tuple:
    """Read a STEP file and return (xde_doc_handle, Manifest).

    Args:
        filepath: Path to the STEP file
        assign_uuids: If True, assign UUIDs to parts that don't have them
        return_shapes: If True, also return ``shapes_by_uuid``, a dict
            mapping str(part_uuid) → TopoDS_Shape. Required by downstream
            BRepFormer / B-Rep feature extraction. Default False to keep
            existing 2-tuple callers working.

    Returns:
        (doc, manifest) by default, or (doc, manifest, shapes_by_uuid) when
        return_shapes is True.
    """
    from ..schema import (
        Manifest, PartEntry, Transform, Fingerprint, PBRMaterial,
        ProvenanceRecord, UnrealSpecific, BundleMetadata,
    )
    from ..uuid_registry import UUIDRegistry, compute_topology_hash

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"STEP file not found: {filepath}")

    logger.info(f"Reading STEP file: {filepath}")

    try:
        from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
        from OCC.Core.TDocStd import TDocStd_Document
        from OCC.Core.XCAFApp import XCAFApp_Application
        from OCC.Core.XCAFDoc import (
            XCAFDoc_DocumentTool,
            XCAFDoc_ColorTool,
            XCAFDoc_ShapeTool,
            XCAFDoc_ColorGen,
            XCAFDoc_ColorSurf,
            XCAFDoc_ColorCurv,
        )
        from OCC.Core.TDF import TDF_LabelSequence, TDF_Label
        from OCC.Core.TDataStd import TDataStd_Name
        from OCC.Core.TCollection import TCollection_ExtendedString
        from OCC.Core.Quantity import Quantity_Color
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.TopLoc import TopLoc_Location
        from OCC.Core.gp import gp_Trsf, gp_XYZ, gp_Quaternion
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopoDS import TopoDS_Shape
        from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE
        from OCC.Core.TopExp import TopExp_Explorer

        HAS_OCC = True
    except ImportError:
        HAS_OCC = False
        logger.warning("PythonOCC not available, using fallback STEP reader")

    registry = UUIDRegistry()
    parts = []
    shapes_by_uuid: dict = {}  # populated by readers that have per-part shapes

    if HAS_OCC:
        # The XDE/STEPCAFControl path can preserve real assembly hierarchy
        # (parent_uuid → child) but on some OCCT builds it produces a
        # **native crash** (stack overrun / Standard_NullObject) when called
        # on certain STEP files — Python try/except cannot catch that.
        # Default to the proven-stable flat names_colors reader.
        # To opt-in to hierarchy preservation set STEP_VR_STEP_XDE=1; the
        # reader still falls back to flat on any Python-level error.
        import os as _os
        doc = None
        if _os.environ.get("STEP_VR_STEP_XDE") == "1":
            try:
                doc = _read_step_with_xde_hierarchy(filepath, registry, parts, assign_uuids)
                if not parts or len(parts) == 1:
                    raise RuntimeError("XDE produced no parts; falling back to flat reader")
                logger.info(f"XDE hierarchy reader succeeded: {len(parts)} parts")
            except Exception as e:
                logger.warning(f"XDE hierarchy reader failed ({e}); falling back to flat reader")
                parts.clear()
                doc = None
        if doc is None:
            doc = _read_step_with_names_colors(
                filepath, registry, parts, assign_uuids,
                shapes_out=shapes_by_uuid,
            )
    else:
        # No OCCT at all — minimal fallback
        doc = _fallback_read_step(filepath, registry, parts)

    # Build manifest
    meta = BundleMetadata(
        created=datetime.now(timezone.utc),
        created_by="step-vr-step",
        app_version="1.0.0",
        source_format="step",
        coordinate_system="RH_Z_up_mm",
        units="mm",
    )

    manifest = Manifest(
        meta=meta,
        parts=parts,
        relationships=[],
    )

    logger.info(f"Read {len(parts)} parts from {filepath}")
    final_doc = doc if HAS_OCC else None
    if return_shapes:
        return final_doc, manifest, shapes_by_uuid
    return final_doc, manifest


def _read_step_with_names_colors(filepath, registry, parts, assign_uuids,
                                  shapes_out: dict | None = None):
    """Read STEP using the stable OCC.Extend.DataExchange API.

    This avoids the XCAFApp_Application crash that happens on some OCCT builds.
    Uses the same approach as the working POC (cad-bidirectional-poc).

    If ``shapes_out`` is provided, fills it with ``str(part_uuid) -> TopoDS_Shape``
    for every non-root part so downstream B-Rep / BRepFormer feature
    extraction can run on the actual geometry.
    """
    from ..schema import (
        PartEntry, Transform, Fingerprint, PBRMaterial,
        ProvenanceRecord, UnrealSpecific,
    )
    from ..uuid_registry import compute_topology_hash
    from OCC.Extend.DataExchange import read_step_file_with_names_colors
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX

    logger.info(f"Reading STEP with names/colors API: {filepath}")

    result = read_step_file_with_names_colors(str(filepath))

    # Build a compound for the doc handle
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    # Root assembly node
    root_uuid = uuid.uuid4()
    root_fp = Fingerprint(
        bbox_min=(0, 0, 0), bbox_max=(0, 0, 0),
        volume_mm3=0, surface_area_mm2=0,
        topology_hash="root", vertex_count=0, face_count=0,
    )
    registry.register(name=filepath.stem, fingerprint=root_fp,
                      source_type="original_step", existing_uuid=root_uuid)
    parts.append(PartEntry(
        uuid=root_uuid,
        step_entity_id="#0",
        name=filepath.stem,
        parent_uuid=None,
        transform=Transform(translation=(0, 0, 0), rotation_quat=(0, 0, 0, 1)),
        local_transform=Transform(translation=(0, 0, 0), rotation_quat=(0, 0, 0, 1)),
        fingerprint=root_fp,
        material=PBRMaterial(name="default", base_color=(0.7, 0.7, 0.7, 1.0)),
        provenance=ProvenanceRecord(source_type="original_step",
                                     import_timestamp=datetime.now(timezone.utc)),
        unreal=UnrealSpecific(),
    ))

    # OCC's read_step_file_with_names_colors returns EVERY named/colored
    # entity in the XDE assembly tree — including loose faces, CSYS markers,
    # PMI annotations, axis arrows. On industrial CAD (ISIS, satellites) this
    # can be tens of thousands of entries, only a small fraction of which are
    # actual parts. A "part" in STEP-native terms is a shape that contains at
    # least one TopAbs_SOLID. Match the POC behaviour and skip the rest.
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID
    skipped_no_solid = 0

    seen_names: dict[str, int] = {}

    # OCC returns these default placeholder strings when the STEP file's
    # PRODUCT entity has no meaningful name. Treat them as missing so we
    # fall back to a `Part_NNN` index instead of showing every part as
    # "Open CASCADE STEP translator-N".
    OCC_DEFAULT_NAME_FRAGMENTS = (
        "open cascade step translator",
        "opencascade step translator",
        "step translator",
        "translator-",
    )
    def _is_occ_default_name(n: str) -> bool:
        if not n:
            return True
        ln = n.strip().lower()
        if not ln:
            return True
        return any(frag in ln for frag in OCC_DEFAULT_NAME_FRAGMENTS)

    for shape, (name, occ_color) in result.items():
        solid_exp = TopExp_Explorer(shape, TopAbs_SOLID)
        if not solid_exp.More():
            skipped_no_solid += 1
            continue

        builder.Add(compound, shape)

        # Unique name (with OCC-default-name filter)
        if _is_occ_default_name(name):
            part_name = f"Part_{len(parts):03d}"
        else:
            part_name = name
        if part_name in seen_names:
            seen_names[part_name] += 1
            part_name = f"{part_name}_{seen_names[part_name]}"
        else:
            seen_names[part_name] = 0

        # Color
        base_color = (0.7, 0.7, 0.7, 1.0)
        if occ_color is not None:
            try:
                base_color = (occ_color.Red(), occ_color.Green(), occ_color.Blue(), 1.0)
            except Exception:
                pass

        # Transform from shape location
        transform = _label_transform(shape)

        # Fingerprint
        fp = _compute_shape_fingerprint(shape)
        if fp is None:
            fp = Fingerprint(
                bbox_min=(0, 0, 0), bbox_max=(0, 0, 0),
                volume_mm3=0, surface_area_mm2=0,
                topology_hash="empty", vertex_count=0, face_count=0,
            )

        part_uuid = uuid.uuid4() if assign_uuids else uuid.uuid4()
        registry.register(name=part_name, fingerprint=fp,
                          source_type="original_step", existing_uuid=part_uuid)

        parts.append(PartEntry(
            uuid=part_uuid,
            step_entity_id=f"#{len(parts)}",
            name=part_name,
            parent_uuid=root_uuid,
            transform=transform,
            local_transform=transform,
            fingerprint=fp,
            material=PBRMaterial(name=f"mat_{part_name}", base_color=base_color),
            provenance=ProvenanceRecord(source_type="original_step",
                                         import_timestamp=datetime.now(timezone.utc)),
            unreal=UnrealSpecific(),
        ))
        if shapes_out is not None:
            shapes_out[str(part_uuid)] = shape

    logger.info(
        f"Read {len(parts) - 1} parts via names/colors API "
        f"(skipped {skipped_no_solid} non-solid XDE entries — annotations, axes, loose faces)"
    )
    return compound


def _read_step_with_xde_hierarchy(filepath, registry, parts, assign_uuids):
    """Read STEP using XCAFDoc_ShapeTool to preserve assembly hierarchy.

    Walks free shapes (top-level assemblies) recursively via _walk_label,
    populating parent_uuid on each child so the frontend PartTree can render
    a real tree instead of a flat list. May raise on older OCCT builds; the
    caller is expected to wrap this in try/except.
    """
    from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.XCAFApp import XCAFApp_Application
    from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
    from OCC.Core.TDF import TDF_LabelSequence
    from OCC.Core.TCollection import TCollection_ExtendedString
    from OCC.Core.IFSelect import IFSelect_RetDone

    app = XCAFApp_Application.GetApplication()
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    app.NewDocument(TCollection_ExtendedString("XmlOcaf"), doc)

    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    reader.SetLayerMode(True)

    status = reader.ReadFile(str(filepath))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEPCAFControl_Reader.ReadFile returned status={status}")

    if not reader.Transfer(doc):
        raise RuntimeError("STEPCAFControl_Reader.Transfer failed")

    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())

    free_shapes = TDF_LabelSequence()
    shape_tool.GetFreeShapes(free_shapes)
    n_free = free_shapes.Length()
    logger.info(f"XDE reader: {n_free} free (top-level) shape(s)")

    if n_free == 0:
        raise RuntimeError("XDE reader saw no free shapes")

    for i in range(n_free):
        root_label = free_shapes.Value(i + 1)
        _walk_label(
            root_label, shape_tool, color_tool, registry, parts,
            parent_uuid=None, depth=0, assign_uuids=assign_uuids,
        )

    # Populate `children` lists on each part by inverting parent_uuid.
    parent_to_children: dict = {}
    for p in parts:
        if p.parent_uuid is not None:
            parent_to_children.setdefault(p.parent_uuid, []).append(p.uuid)
    for p in parts:
        if p.uuid in parent_to_children:
            try:
                p.children = parent_to_children[p.uuid]
            except Exception:
                pass  # schema may not allow direct assignment; safe to skip

    return doc


def _walk_label(label, shape_tool, color_tool, registry, parts,
                parent_uuid, depth, assign_uuids):
    """Recursively walk XDE labels building PartEntry objects."""
    from ..schema import (
        PartEntry, Transform, Fingerprint, PBRMaterial,
        ProvenanceRecord, UnrealSpecific,
    )
    from OCC.Core.TDF import TDF_LabelSequence
    from OCC.Core.TDataStd import TDataStd_Name
    from OCC.Core.TCollection import TCollection_ExtendedString
    from OCC.Core.Quantity import Quantity_Color
    from OCC.Core.XCAFDoc import XCAFDoc_ColorSurf, XCAFDoc_ColorGen
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
    from ..uuid_registry import compute_topology_hash

    # Get name
    name_attr = TDataStd_Name()
    name = f"Part_{depth}"
    if label.FindAttribute(TDataStd_Name.GetID(), name_attr):
        name = name_attr.Get().ToExtString()

    # Get shape
    shape = shape_tool.GetShape(label)

    # Get or assign UUID
    existing_uuid = _extract_uuid_property(label)

    # Compute transform
    transform = _label_transform(shape)

    # Compute fingerprint
    fp = _compute_shape_fingerprint(shape)

    # Register in UUID registry
    part_uuid_val = registry.register(
        name=name,
        fingerprint=fp if fp else Fingerprint(
            bbox_min=(0, 0, 0), bbox_max=(0, 0, 0),
            volume_mm3=0, surface_area_mm2=0,
            topology_hash="empty", vertex_count=0, face_count=0,
        ),
        source_type="original_step",
        existing_uuid=existing_uuid if existing_uuid else (uuid.uuid4() if assign_uuids else None),
    )

    # Get color
    color = Quantity_Color()
    has_color = color_tool.GetColor(label, XCAFDoc_ColorSurf, color)
    if not has_color:
        has_color = color_tool.GetColor(label, XCAFDoc_ColorGen, color)

    base_color = (0.7, 0.7, 0.7, 1.0)
    if has_color:
        base_color = (color.Red(), color.Green(), color.Blue(), 1.0)

    # Build step entity ID from label tag
    step_entity_id = f"#{label.Tag()}"

    # Create PartEntry
    entry = PartEntry(
        uuid=part_uuid_val,
        step_entity_id=step_entity_id,
        name=name,
        parent_uuid=parent_uuid,
        transform=transform,
        local_transform=transform,  # Will be refined with parent transforms
        fingerprint=fp if fp else Fingerprint(
            bbox_min=(0, 0, 0), bbox_max=(0, 0, 0),
            volume_mm3=0, surface_area_mm2=0,
            topology_hash="empty", vertex_count=0, face_count=0,
        ),
        material=PBRMaterial(
            name=f"mat_{name}",
            base_color=base_color,
        ),
        provenance=ProvenanceRecord(
            source_type="original_step",
            import_timestamp=datetime.now(timezone.utc),
        ),
        unreal=UnrealSpecific(),
    )
    parts.append(entry)

    # Recurse into sub-labels (assembly components)
    sub_labels = TDF_LabelSequence()
    shape_tool.GetSubShapes(label, sub_labels)

    components = TDF_LabelSequence()
    shape_tool.GetComponents(label, components)

    for i in range(components.Length()):
        sub_label = components.Value(i + 1)
        ref_label = sub_label
        # Resolve reference if this is an instance
        if shape_tool.IsReference(sub_label):
            ref_label_out = sub_label  # placeholder
            shape_tool.GetReferredShape(sub_label, ref_label_out)
            ref_label = ref_label_out

        _walk_label(
            ref_label, shape_tool, color_tool, registry, parts,
            parent_uuid=part_uuid_val, depth=depth + 1,
            assign_uuids=assign_uuids,
        )


def _extract_uuid_property(label) -> Optional[uuid.UUID]:
    """Extract step_vr_step/part_uuid property from an XDE label."""
    # Properties are stored as XCAFDoc attributes
    # In practice, custom properties in STEP are property_definitions
    # We look for the specific named property
    try:
        from OCC.Core.TDataStd import TDataStd_Name
        from OCC.Core.TDF import TDF_ChildIterator

        # Walk child labels looking for our custom property
        it = TDF_ChildIterator(label)
        while it.More():
            child = it.Value()
            name_attr = TDataStd_Name()
            if child.FindAttribute(TDataStd_Name.GetID(), name_attr):
                attr_name = name_attr.Get().ToExtString()
                if "step_vr_step/part_uuid" in attr_name:
                    # Extract UUID value from the attribute
                    # The UUID is stored as part of the name or as a string attribute
                    uuid_str = attr_name.split("=")[-1].strip() if "=" in attr_name else None
                    if uuid_str:
                        try:
                            return uuid.UUID(uuid_str)
                        except ValueError:
                            pass
            it.Next()
    except Exception as e:
        logger.debug(f"Could not extract UUID property: {e}")

    return None


def _label_transform(shape) -> Transform:
    """Extract transform from a shape's location."""
    from ..schema import Transform

    try:
        loc = shape.Location()
        trsf = loc.Transformation()

        # Translation
        t = trsf.TranslationPart()
        translation = (t.X(), t.Y(), t.Z())

        # Rotation as quaternion
        # OCCT gp_Trsf stores rotation as a 3x3 matrix
        mat = trsf.VectorialPart()
        # Convert 3x3 rotation matrix to quaternion
        rot_matrix = np.array([
            [mat.Value(1, 1), mat.Value(1, 2), mat.Value(1, 3)],
            [mat.Value(2, 1), mat.Value(2, 2), mat.Value(2, 3)],
            [mat.Value(3, 1), mat.Value(3, 2), mat.Value(3, 3)],
        ])
        quat = _rotation_matrix_to_quaternion(rot_matrix)

        return Transform(
            translation=translation,
            rotation_quat=quat,
            scale=(1.0, 1.0, 1.0),
        )
    except Exception as e:
        logger.debug(f"Could not extract transform: {e}")
        return Transform(
            translation=(0.0, 0.0, 0.0),
            rotation_quat=(0.0, 0.0, 0.0, 1.0),
        )


def _compute_shape_fingerprint(shape) -> Optional[Fingerprint]:
    """Compute geometric fingerprint from an OCCT shape."""
    from ..schema import Fingerprint
    from ..uuid_registry import compute_topology_hash

    try:
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX

        if shape is None or shape.IsNull():
            return None

        # Bounding box
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        # Volume and surface area
        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        volume = abs(props.Mass())

        sprops = GProp_GProps()
        brepgprop.SurfaceProperties(shape, sprops)
        surface_area = abs(sprops.Mass())

        # Count topology elements
        face_count = 0
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        face_areas = []
        while exp.More():
            face_count += 1
            # Get individual face area
            face_props = GProp_GProps()
            brepgprop.SurfaceProperties(exp.Current(), face_props)
            face_areas.append(abs(face_props.Mass()))
            exp.Next()

        edge_count = 0
        exp = TopExp_Explorer(shape, TopAbs_EDGE)
        while exp.More():
            edge_count += 1
            exp.Next()

        vertex_count = 0
        exp = TopExp_Explorer(shape, TopAbs_VERTEX)
        while exp.More():
            vertex_count += 1
            exp.Next()

        topo_hash = compute_topology_hash(
            face_count, edge_count, vertex_count,
            face_areas=sorted(face_areas, reverse=True)[:50],
        )

        return Fingerprint(
            bbox_min=(xmin, ymin, zmin),
            bbox_max=(xmax, ymax, zmax),
            volume_mm3=volume,
            surface_area_mm2=surface_area,
            topology_hash=topo_hash,
            vertex_count=vertex_count,
            face_count=face_count,
        )
    except Exception as e:
        logger.warning(f"Could not compute fingerprint: {e}")
        return None


def _rotation_matrix_to_quaternion(m: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to quaternion (x, y, z, w)."""
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


def _fallback_read_step(filepath, registry, parts):
    """Fallback reader when PythonOCC is not available (for testing)."""
    from ..schema import (
        PartEntry, Transform, Fingerprint, PBRMaterial,
        ProvenanceRecord, UnrealSpecific,
    )

    logger.info("Using fallback STEP reader (no PythonOCC)")

    # Create a minimal part entry for the file
    part_uuid = uuid.uuid4()
    fp = Fingerprint(
        bbox_min=(0, 0, 0), bbox_max=(1, 1, 1),
        volume_mm3=1.0, surface_area_mm2=6.0,
        topology_hash="fallback", vertex_count=0, face_count=0,
    )

    registry.register(
        name=filepath.stem,
        fingerprint=fp,
        source_type="original_step",
        existing_uuid=part_uuid,
    )

    entry = PartEntry(
        uuid=part_uuid,
        step_entity_id="#1",
        name=filepath.stem,
        parent_uuid=None,
        transform=Transform(translation=(0, 0, 0), rotation_quat=(0, 0, 0, 1)),
        local_transform=Transform(translation=(0, 0, 0), rotation_quat=(0, 0, 0, 1)),
        fingerprint=fp,
        material=PBRMaterial(name="default", base_color=(0.7, 0.7, 0.7, 1.0)),
        provenance=ProvenanceRecord(
            source_type="original_step",
            import_timestamp=datetime.now(timezone.utc),
        ),
        unreal=UnrealSpecific(),
    )
    parts.append(entry)

    return None
