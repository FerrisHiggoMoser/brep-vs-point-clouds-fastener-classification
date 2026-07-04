"""Conflict detection and user prompt generation for ambiguous matches.

When reconciliation encounters uncertain cases (confidence < 0.8, splits,
merges), generates structured conflict events for the UI to prompt the user.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def collect_ambiguous(match_result) -> list[dict]:
    """Collect all cases requiring user resolution."""
    conflicts = []

    # Ambiguous matches (low confidence)
    for amb in match_result.ambiguous:
        conflicts.append({
            "type": "ambiguous_match",
            "new_uuid": amb["new_uuid"],
            "old_uuid": amb["old_uuid"],
            "confidence": amb["confidence"],
            "message": f"Uncertain match (confidence {amb['confidence']:.0%}). Is this the same part?",
            "options": ["accept", "reject", "manual_match"],
        })

    # Splits
    for split in match_result.splits:
        conflicts.append({
            "type": "split",
            "old_uuid": split["old_uuid"],
            "new_uuids": split["new_uuids"],
            "message": f"Part was split into {len(split['new_uuids'])} parts. Which keeps the original metadata?",
            "options": split["new_uuids"],
        })

    # Merges
    for merge in match_result.merges:
        conflicts.append({
            "type": "merge",
            "new_uuid": merge["new_uuid"],
            "old_uuids": merge["old_uuids"],
            "message": f"{len(merge['old_uuids'])} parts were merged. Which source provides metadata?",
            "options": merge["old_uuids"],
        })

    return conflicts


def yield_conflict_events(conflicts: list[dict], emitter, req_id: str) -> None:
    """Emit conflict events to the Tauri frontend for user resolution."""
    for conflict in conflicts:
        emitter.emit({
            "evt": "conflict",
            "req_id": req_id,
            "conflict": conflict,
            "needs_user": True,
        })


def apply_resolutions(match_result, resolutions: list[dict]) -> None:
    """Apply user resolutions to the match result.

    Resolutions format: [{"conflict_index": 0, "choice": "accept"}, ...]
    """
    for res in resolutions:
        idx = res.get("conflict_index", -1)
        choice = res.get("choice", "")

        if choice == "reject":
            # Remove the match
            if idx < len(match_result.ambiguous):
                amb = match_result.ambiguous[idx]
                match_result.matches = [
                    m for m in match_result.matches
                    if not (str(m[0]) == str(amb["new_uuid"]) and str(m[1]) == str(amb["old_uuid"]))
                ]
                match_result.new_parts.append(amb["new_uuid"])
                match_result.deleted_parts.append(amb["old_uuid"])
