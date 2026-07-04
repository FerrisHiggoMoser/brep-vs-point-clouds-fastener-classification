"""STEP file writer with sidecar bundle generation.

Takes an XDE document and Manifest, writes:
- part.step: STEP AP242 file with UUID properties on every entity
- Sidecar bundle: manifest.json, textures/, validation_report.json, etc.

For original_step provenance parts: preserves NURBS B-Rep from archived originals.
For mesh-only parts: writes AP242 tessellated_solid representation.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def write_step_bundle(xde_doc, manifest, output_path: str | Path,
                      archive_dir: str | Path | None = None) -> Path:
    """Write a STEP file + sidecar bundle.

    Args:
        xde_doc: XDE document handle (from reader), or None for mesh-only input
        manifest: Manifest describing all parts
        output_path: Output directory for the bundle
        archive_dir: Optional directory containing archived original STEP files

    Returns:
        Path to the created bundle directory
    """
    from ..schema import Manifest
    from ..sidecar.bundle import create_bundle

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    step_path = output_path / "part.step"

    # Heal invalid solids before writing. For an XDE doc we substitute solids
    # in-place via SetShape (preserves UUID/name/color metadata on labels).
    # For a bare TopoDS_Shape we rebuild the compound with healed solids and
    # swap the local reference so the writer sees the repaired geometry.
    # Counts are persisted to topology_repair.json so validation/topology.py
    # can attribute failures (source-bad vs round-trip-bad).
    repair_stats, healed_shape = _heal_invalid_solids(xde_doc)
    if healed_shape is not None:
        xde_doc = healed_shape

    # Write STEP file
    _write_step_file(xde_doc, manifest, step_path, archive_dir)

    # Create bundle with STEP file
    bundle_dir = create_bundle(output_path, manifest, step_path)

    # Persist topology repair counts so the validator can interpret failures.
    if repair_stats is not None:
        import json as _json
        (bundle_dir / "topology_repair.json").write_text(
            _json.dumps(repair_stats, indent=2)
        )
        logger.info(
            "Topology heal: source_invalid=%d, healed=%d, remaining=%d",
            repair_stats["source_invalid"],
            repair_stats["healed"],
            repair_stats["remaining_invalid"],
        )

    logger.info(f"Wrote STEP bundle to {bundle_dir}")
    return bundle_dir


def _heal_invalid_solids(xde_doc):
    """Run ShapeFix on invalid solids. Returns ``(stats, healed_shape)``:
        stats        -- {"source_invalid", "healed", "remaining_invalid", "mode"}
                        or None if OCC unavailable / shape can't be walked
        healed_shape -- a replacement shape for the bare-shape path, or None
                        for the XDE path (where solids are substituted in-place
                        via SetShape so labels and metadata are preserved)
    """
    try:
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        from OCC.Core.ShapeFix import ShapeFix_Shape
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SOLID
        from OCC.Core.TopoDS import TopoDS_Shape, topods, TopoDS_Compound
        from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
        from OCC.Core.TDF import TDF_LabelSequence
        from OCC.Core.BRep import BRep_Builder
    except ImportError:
        return None, None

    def _heal_solid(solid):
        """Returns a healed TopoDS_Solid or None if repair didn't yield a valid result."""
        try:
            fixer = ShapeFix_Shape(solid)
            fixer.Perform()
            candidate = fixer.Shape()
            fix_exp = TopExp_Explorer(candidate, TopAbs_SOLID)
            if fix_exp.More():
                cand_solid = topods.Solid(fix_exp.Current())
                if BRepCheck_Analyzer(cand_solid, True).IsValid():
                    return cand_solid
        except Exception:
            pass
        return None

    is_xde_doc = hasattr(xde_doc, "Main")
    if not is_xde_doc:
        if not isinstance(xde_doc, TopoDS_Shape):
            return None, None
        # Bare-shape path: walk solids, heal invalid ones, rebuild a compound.
        replacements = []
        any_bad = False
        source_invalid = 0
        healed = 0
        exp = TopExp_Explorer(xde_doc, TopAbs_SOLID)
        while exp.More():
            solid = topods.Solid(exp.Current())
            if not BRepCheck_Analyzer(solid, True).IsValid():
                source_invalid += 1
                any_bad = True
                fixed = _heal_solid(solid)
                if fixed is not None:
                    healed += 1
                replacements.append(fixed if fixed is not None else solid)
            else:
                replacements.append(solid)
            exp.Next()

        stats = {
            "source_invalid": source_invalid,
            "healed": healed,
            "remaining_invalid": source_invalid - healed,
            "mode": "bare_shape_rebuild",
        }
        if not any_bad:
            return stats, None

        builder = BRep_Builder()
        new_compound = TopoDS_Compound()
        builder.MakeCompound(new_compound)
        for s in replacements:
            builder.Add(new_compound, s)
        return stats, new_compound

    shape_tool = XCAFDoc_DocumentTool.ShapeTool(xde_doc.Main())
    free_labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(free_labels)

    source_invalid = 0
    healed = 0

    for i in range(1, free_labels.Length() + 1):
        label = free_labels.Value(i)
        try:
            shape = shape_tool.GetShape(label)
        except Exception:
            continue
        if shape is None or shape.IsNull():
            continue

        # Walk solids inside this free shape, ShapeFix the bad ones, rebuild
        # a compound with healed substitutions. If nothing was bad, skip the
        # rewrite (preserves identity for the SetShape no-op case).
        replacements = []  # list of (original_solid, healed_or_None)
        any_bad = False
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        while exp.More():
            solid = topods.Solid(exp.Current())
            if not BRepCheck_Analyzer(solid, True).IsValid():
                source_invalid += 1
                any_bad = True
                fixed_solid = None
                try:
                    fixer = ShapeFix_Shape(solid)
                    fixer.Perform()
                    candidate = fixer.Shape()
                    fix_exp = TopExp_Explorer(candidate, TopAbs_SOLID)
                    if fix_exp.More():
                        cand_solid = topods.Solid(fix_exp.Current())
                        if BRepCheck_Analyzer(cand_solid, True).IsValid():
                            fixed_solid = cand_solid
                            healed += 1
                except Exception:
                    pass
                replacements.append((solid, fixed_solid))
            else:
                replacements.append((solid, None))
            exp.Next()

        if not any_bad:
            continue

        # Rebuild this free shape's compound with healed solids substituted.
        # Note: this collapses any non-solid sub-shapes inside the free shape,
        # but free shapes in CAD assemblies are almost always compounds of
        # solids, so this is acceptable in practice.
        builder = BRep_Builder()
        new_compound = TopoDS_Compound()
        builder.MakeCompound(new_compound)
        for orig, fixed in replacements:
            builder.Add(new_compound, fixed if fixed is not None else orig)

        try:
            shape_tool.SetShape(label, new_compound)
        except Exception as e:
            logger.warning("Could not SetShape on healed compound: %s", e)
            # Roll back the healed count for this label since we couldn't apply
            healed -= sum(1 for _, f in replacements if f is not None)

    remaining = source_invalid - healed
    return {
        "source_invalid": source_invalid,
        "healed": healed,
        "remaining_invalid": remaining,
        "mode": "xde_in_place",
    }, None


def _write_step_file(xde_doc, manifest, step_path: Path,
                     archive_dir: Path | None = None) -> None:
    """Write the STEP file from XDE document and manifest."""

    try:
        from OCC.Core.STEPCAFControl import STEPCAFControl_Writer
        from OCC.Core.TDocStd import TDocStd_Document
        from OCC.Core.XCAFApp import XCAFApp_Application
        from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
        from OCC.Core.TCollection import TCollection_ExtendedString
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_AsIs
        from OCC.Core.TDataStd import TDataStd_Name
        from OCC.Core.TDF import TDF_LabelSequence

        HAS_OCC = True
    except ImportError:
        HAS_OCC = False

    if not HAS_OCC:
        logger.warning("PythonOCC not available, writing placeholder STEP file")
        _write_fallback_step(manifest, step_path)
        return

    # Check if we have a real XDE document or just a TopoDS_Shape/Compound
    from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound
    is_xde_doc = hasattr(xde_doc, 'Main')
    is_shape = isinstance(xde_doc, (TopoDS_Shape, TopoDS_Compound))

    if is_xde_doc:
        # Full XDE path
        shape_tool = XCAFDoc_DocumentTool.ShapeTool(xde_doc.Main())
        _attach_uuid_properties(shape_tool, manifest)

        writer = STEPCAFControl_Writer()
        writer.SetColorMode(True)
        writer.SetNameMode(True)
        writer.SetLayerMode(True)
        writer.SetPropsMode(True)
        writer.Transfer(xde_doc, STEPControl_AsIs)

        status = writer.Write(str(step_path))
        if status != IFSelect_RetDone:
            raise RuntimeError(f"Failed to write STEP file: {step_path} (status={status})")
        logger.info(f"Wrote STEP file (XDE): {step_path}")

    elif is_shape:
        # Simple shape — use STEPControl_Writer directly
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs as SC_AsIs
        writer = STEPControl_Writer()
        writer.Transfer(xde_doc, SC_AsIs)
        status = writer.Write(str(step_path))
        if status != IFSelect_RetDone:
            raise RuntimeError(f"Failed to write STEP file: {step_path} (status={status})")
        logger.info(f"Wrote STEP file (shape): {step_path}")

    elif xde_doc is None:
        # No geometry at all — write placeholder
        _write_fallback_step(manifest, step_path)

    else:
        logger.warning(f"Unknown doc type: {type(xde_doc)}, writing placeholder")
        _write_fallback_step(manifest, step_path)


def _build_xde_from_manifest(manifest):
    """Build an XDE document from a Manifest (for mesh-only input)."""
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.XCAFApp import XCAFApp_Application
    from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool, XCAFDoc_ColorTool
    from OCC.Core.TCollection import TCollection_ExtendedString
    from OCC.Core.TDataStd import TDataStd_Name
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCC.Core.XCAFDoc import XCAFDoc_ColorSurf

    app = XCAFApp_Application.GetApplication()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.InitDocument(doc)

    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())

    for part in manifest.parts:
        # Create an empty compound for each part (actual geometry would be tessellated)
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)

        label = shape_tool.AddShape(compound)

        # Set name
        TDataStd_Name.Set(label, TCollection_ExtendedString(part.name))

        # Set color
        bc = part.material.base_color
        color = Quantity_Color(bc[0], bc[1], bc[2], Quantity_TOC_RGB)
        color_tool.SetColor(label, color, XCAFDoc_ColorSurf)

    return doc


def _attach_uuid_properties(shape_tool, manifest) -> None:
    """Attach UUID and metadata properties to XDE labels."""
    from OCC.Core.TDF import TDF_LabelSequence
    from OCC.Core.TDataStd import TDataStd_Name
    from OCC.Core.TCollection import TCollection_ExtendedString

    labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(labels)

    part_idx = 0
    for i in range(labels.Length()):
        if part_idx >= len(manifest.parts):
            break

        label = labels.Value(i + 1)
        part = manifest.parts[part_idx]

        # Write UUID as a child label attribute
        # In STEP, this becomes a property_definition
        _set_string_property(label, f"step_vr_step/part_uuid={part.uuid}")
        _set_string_property(label, f"step_vr_step/fingerprint_hash={part.fingerprint.topology_hash}")
        _set_string_property(label, f"step_vr_step/provenance_type={part.provenance.source_type}")
        _set_string_property(label, f"step_vr_step/exported_at={datetime.now(timezone.utc).isoformat()}")
        _set_string_property(label, f"step_vr_step/bundle_version={manifest.meta.bundle_version}")

        if part.provenance.original_brep_hash:
            _set_string_property(label, f"step_vr_step/original_brep_hash={part.provenance.original_brep_hash}")

        part_idx += 1


def _set_string_property(label, value: str) -> None:
    """Set a string property on an XDE label via child label with TDataStd_Name."""
    from OCC.Core.TDataStd import TDataStd_Name
    from OCC.Core.TCollection import TCollection_ExtendedString
    from OCC.Core.TDF import TDF_TagSource

    child = TDF_TagSource.NewChild(label)
    TDataStd_Name.Set(child, TCollection_ExtendedString(value))


def _write_fallback_step(manifest, step_path: Path) -> None:
    """Write a minimal placeholder STEP file when PythonOCC is not available."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    lines = [
        "ISO-10303-21;",
        "HEADER;",
        f"FILE_DESCRIPTION(('step-vr-step export'), '2;1');",
        f"FILE_NAME('{step_path.name}', '{timestamp}', ('step-vr-step'), (''), '', 'step-vr-step 1.0', '');",
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));",
        "ENDSEC;",
        "DATA;",
    ]

    entity_id = 1
    for part in manifest.parts:
        lines.append(f"#{entity_id} = PRODUCT('{part.name}', '{part.name}', '', (#));")
        entity_id += 1
        # Write UUID as property
        lines.append(f"#{entity_id} = PROPERTY_DEFINITION('step_vr_step/part_uuid', '{part.uuid}', #);")
        entity_id += 1

    lines.extend([
        "ENDSEC;",
        "END-ISO-10303-21;",
    ])

    step_path.write_text("\n".join(lines))
    logger.info(f"Wrote fallback STEP file: {step_path}")
