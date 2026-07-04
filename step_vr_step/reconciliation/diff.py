"""Per-part change type computation after matching.

Given matched pairs, compute: unchanged, moved, reshaped, renamed,
recolored, metadata_only. Aggregates into a DiffReport.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_diff(new_manifest, old_manifest, match_result) -> dict:
    """Build a diff report from matched parts.

    Returns a dict consumable by the ReconciliationDiff UI component.
    """
    old_by_uuid = {str(p.uuid): p for p in old_manifest.parts}
    new_by_uuid = {str(p.uuid): p for p in new_manifest.parts}

    changes = []
    summary = {
        "unchanged": 0, "moved": 0, "reshaped": 0,
        "renamed": 0, "recolored": 0, "metadata_only": 0,
        "added": len(match_result.new_parts),
        "deleted": len(match_result.deleted_parts),
    }

    for new_uuid, old_uuid, confidence in match_result.matches:
        new_part = new_by_uuid.get(str(new_uuid))
        old_part = old_by_uuid.get(str(old_uuid))

        if not new_part or not old_part:
            continue

        change_type = _classify_change(new_part, old_part)
        summary[change_type] += 1

        changes.append({
            "new_uuid": str(new_uuid),
            "old_uuid": str(old_uuid),
            "name": new_part.name,
            "change_type": change_type,
            "confidence": confidence,
            "details": _change_details(new_part, old_part, change_type),
        })

    # Add new parts
    for uuid_str in match_result.new_parts:
        part = new_by_uuid.get(str(uuid_str))
        if part:
            changes.append({
                "new_uuid": str(uuid_str),
                "old_uuid": None,
                "name": part.name,
                "change_type": "added",
                "confidence": 1.0,
                "details": "New part added in CAD",
            })

    # Add deleted parts
    for uuid_str in match_result.deleted_parts:
        part = old_by_uuid.get(str(uuid_str))
        if part:
            changes.append({
                "new_uuid": None,
                "old_uuid": str(uuid_str),
                "name": part.name,
                "change_type": "deleted",
                "confidence": 1.0,
                "details": "Part deleted in CAD",
            })

    return {
        "summary": summary,
        "changes": changes,
        "splits": match_result.splits,
        "merges": match_result.merges,
        "total_parts_new": len(new_manifest.parts),
        "total_parts_old": len(old_manifest.parts),
    }


def _classify_change(new_part, old_part) -> str:
    """Classify the type of change between matched parts."""
    fp_same = new_part.fingerprint.topology_hash == old_part.fingerprint.topology_hash

    if fp_same:
        vol_diff = abs(new_part.fingerprint.volume_mm3 - old_part.fingerprint.volume_mm3)
        max_vol = max(new_part.fingerprint.volume_mm3, old_part.fingerprint.volume_mm3, 1e-10)

        if vol_diff / max_vol < 0.001:
            # Geometry identical — check transform
            if _transforms_equal(new_part.transform, old_part.transform):
                if new_part.name != old_part.name:
                    return "renamed"
                return "unchanged"
            else:
                return "moved"
        else:
            return "reshaped"
    else:
        return "reshaped"


def _transforms_equal(t1, t2, tolerance: float = 0.01) -> bool:
    """Check if two transforms are equal within tolerance."""
    for i in range(3):
        if abs(t1.translation[i] - t2.translation[i]) > tolerance:
            return False
    for i in range(4):
        if abs(t1.rotation_quat[i] - t2.rotation_quat[i]) > tolerance:
            return False
    return True


def _change_details(new_part, old_part, change_type: str) -> str:
    """Generate human-readable change details."""
    if change_type == "unchanged":
        return "No changes detected"
    elif change_type == "moved":
        t1 = new_part.transform.translation
        t2 = old_part.transform.translation
        dx = t1[0] - t2[0]
        dy = t1[1] - t2[1]
        dz = t1[2] - t2[2]
        import math
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        return f"Moved {dist:.2f}mm (dx={dx:.1f}, dy={dy:.1f}, dz={dz:.1f})"
    elif change_type == "reshaped":
        vol_old = old_part.fingerprint.volume_mm3
        vol_new = new_part.fingerprint.volume_mm3
        if vol_old > 0:
            pct = ((vol_new - vol_old) / vol_old) * 100
            return f"Geometry changed (volume {pct:+.1f}%)"
        return "Geometry changed"
    elif change_type == "renamed":
        return f"Renamed: '{old_part.name}' → '{new_part.name}'"
    return change_type
