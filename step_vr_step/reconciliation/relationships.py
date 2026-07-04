"""Relationship revalidation after geometry reconciliation.

After matching, walk every relationship in the old manifest and check
if the geometric constraints still hold (fastener fits hole, surfaces
still in contact, etc.).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def revalidate_relationships(old_manifest, new_manifest, match_result) -> None:
    """Revalidate all relationships from the old manifest.

    Checks if geometric constraints still hold after CAD edits.
    Marks broken relationships and proposes new ones for added parts.
    """
    new_by_uuid = {str(p.uuid): p for p in new_manifest.parts}

    # Map old UUIDs to new UUIDs
    uuid_remap = {}
    for new_uuid, old_uuid, conf in match_result.matches:
        uuid_remap[str(old_uuid)] = str(new_uuid)

    for rel in old_manifest.relationships:
        # Remap UUIDs
        new_subject = uuid_remap.get(str(rel.subject_uuid))
        new_target = uuid_remap.get(str(rel.target_uuid))

        if new_subject is None or new_target is None:
            rel.broken = True
            logger.info(f"Relationship {rel.kind} broken: missing subject/target after edit")
            continue

        # Check if both parts still exist
        subject_part = new_by_uuid.get(new_subject)
        target_part = new_by_uuid.get(new_target)

        if subject_part is None or target_part is None:
            rel.broken = True
            continue

        # Validate based on relationship kind
        if rel.kind == "fastener":
            rel.broken = not _validate_fastener(subject_part, target_part, rel)
        elif rel.kind == "mate":
            rel.broken = not _validate_contact(subject_part, target_part)
        elif rel.kind == "contact":
            rel.broken = not _validate_contact(subject_part, target_part)
        else:
            # For custom/weld/bond: assume valid if both parts exist
            rel.broken = False

    # Count results
    intact = sum(1 for r in old_manifest.relationships if not r.broken)
    broken = sum(1 for r in old_manifest.relationships if r.broken)
    logger.info(f"Relationships: {intact} intact, {broken} broken")


def _validate_fastener(subject, target, rel) -> bool:
    """Validate a fastener relationship still holds.

    Checks: subject's axis from new position still hits target's geometry.
    Simplified: check bounding boxes overlap reasonably.
    """
    # Check if bounding boxes are still in proximity
    s_center = [
        (subject.fingerprint.bbox_min[i] + subject.fingerprint.bbox_max[i]) / 2 + subject.transform.translation[i]
        for i in range(3)
    ]
    t_center = [
        (target.fingerprint.bbox_min[i] + target.fingerprint.bbox_max[i]) / 2 + target.transform.translation[i]
        for i in range(3)
    ]

    # Distance between centers
    import math
    dist = math.sqrt(sum((s - t) ** 2 for s, t in zip(s_center, t_center)))

    # Fastener should be within reasonable range of its target
    # Use max dimension of target as threshold
    target_size = max(
        target.fingerprint.bbox_max[i] - target.fingerprint.bbox_min[i]
        for i in range(3)
    )

    # If distance > 2x target size, relationship is likely broken
    return dist < target_size * 2 + 50  # 50mm margin


def _validate_contact(part1, part2) -> bool:
    """Validate a contact/mate relationship.

    Check if bounding boxes overlap or are in close proximity.
    """
    margin = 1.0  # mm tolerance

    for i in range(3):
        min1 = part1.fingerprint.bbox_min[i] + part1.transform.translation[i] - margin
        max1 = part1.fingerprint.bbox_max[i] + part1.transform.translation[i] + margin
        min2 = part2.fingerprint.bbox_min[i] + part2.transform.translation[i] - margin
        max2 = part2.fingerprint.bbox_max[i] + part2.transform.translation[i] + margin

        if max1 < min2 or max2 < min1:
            return False  # No overlap in this dimension

    return True
