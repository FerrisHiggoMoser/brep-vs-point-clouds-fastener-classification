"""Reconciliation engine — matches, diffs, and merges CAD edits back into Unreal scenes.

This is the headline capability: detecting what changed between an original export
and an engineer's CAD edits, then applying those changes back with full metadata
preservation.
"""
from __future__ import annotations

from .matcher import match
from .diff import build_diff
from .relationships import revalidate_relationships


def reconcile(edited_step_path, original_bundle_path, output_path, emitter=None, req_id="reconcile"):
    """Full reconciliation pipeline.

    Per spec §11: 10-pass algorithm.
    """
    from pathlib import Path
    from ..readers.step_reader import read_step
    from ..sidecar.manifest import read_manifest
    from ..writers.datasmith_writer import write_datasmith
    from ..validation.report import run_full_validation
    from ..schema import UnrealSpecific
    from .conflict import collect_ambiguous, yield_conflict_events

    edited_step_path = Path(edited_step_path)
    original_bundle_path = Path(original_bundle_path)
    output_path = Path(output_path)

    if emitter:
        emitter.progress(req_id, "reconciliation_match", 0.1, "Reading edited STEP")

    # Pass 1-2: Read and match
    new_doc, new_manifest = read_step(edited_step_path)
    old_manifest = read_manifest(original_bundle_path)

    if emitter:
        emitter.progress(req_id, "reconciliation_match", 0.3, "Matching parts by UUID and fingerprint")

    match_result = match(new_manifest, old_manifest)

    if emitter:
        emitter.progress(req_id, "reconciliation_diff", 0.5, "Computing diff")

    # Pass 5: Handle ambiguities
    ambiguities = collect_ambiguous(match_result)
    if ambiguities and emitter:
        yield_conflict_events(ambiguities, emitter, req_id)

    # Pass 6: Build diff
    diff_report = build_diff(new_manifest, old_manifest, match_result)

    if emitter:
        emitter.emit({"evt": "diff", "req_id": req_id, "diff": diff_report})
        emitter.progress(req_id, "reconciliation_diff", 0.6, "Revalidating relationships")

    # Pass 7: Relationship revalidation
    revalidate_relationships(old_manifest, new_manifest, match_result)

    # Pass 8: Construct new manifest — transfer metadata from old to new
    for m in match_result.matches:
        new_uuid, old_uuid = m[0], m[1]
        old_part = next((p for p in old_manifest.parts if str(p.uuid) == str(old_uuid)), None)
        new_part = next((p for p in new_manifest.parts if str(p.uuid) == str(new_uuid)), None)
        if old_part and new_part:
            # Transfer Unreal metadata from old
            new_part.unreal = old_part.unreal
            new_part.material = old_part.material
            new_part.pmi = old_part.pmi

    # New parts get default metadata
    for new_uuid in match_result.new_parts:
        new_part = next((p for p in new_manifest.parts if str(p.uuid) == str(new_uuid)), None)
        if new_part:
            new_part.unreal = UnrealSpecific()

    # Record deletions
    for del_uuid in match_result.deleted_parts:
        new_manifest.unmatched_on_return.append({"uuid": str(del_uuid), "reason": "deleted_in_cad"})

    # Transfer relationships
    new_manifest.relationships = old_manifest.relationships

    if emitter:
        emitter.progress(req_id, "datasmith_write", 0.8, "Writing Datasmith output")

    # Pass 9: Write output
    write_datasmith(new_doc, new_manifest, output_path, source_bundle_dir=str(original_bundle_path))

    if emitter:
        emitter.progress(req_id, "validation", 0.9, "Validating output")

    # Pass 10: Validate
    report = run_full_validation(output_path, original_manifest=old_manifest)

    return {"diff": diff_report, "report": report.model_dump(), "output_path": str(output_path)}
