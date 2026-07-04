"""Part matching for reconciliation — UUID, fingerprint, and name-based matching.

This is the critical matching engine that determines what changed between
the original export and the engineer's edited STEP file.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of the matching algorithm."""
    matches: list[tuple]  # (new_uuid, old_uuid, confidence)
    new_parts: list  # UUIDs of parts only in the new manifest
    deleted_parts: list  # UUIDs of parts only in the old manifest
    splits: list[dict] = field(default_factory=list)  # one old → multiple new
    merges: list[dict] = field(default_factory=list)  # multiple old → one new
    ambiguous: list[dict] = field(default_factory=list)  # uncertain cases


def match(new_manifest, old_manifest, strategy: str = "hybrid") -> MatchResult:
    """Match parts between new (edited) and old (original) manifests.

    Matching passes per spec §9.9:
    - Pass 1: Direct UUID match (confidence 1.0)
    - Pass 2: Fingerprint match for UUID-less parts
    - Pass 3: Name + position match for still-unmatched
    - Pass 4: Detect splits (one old → multiple new)
    - Pass 5: Detect merges (multiple old → one new)

    Returns MatchResult with all match types.
    """
    matches = []

    # Build lookup dicts
    new_by_uuid = {str(p.uuid): p for p in new_manifest.parts}
    old_by_uuid = {str(p.uuid): p for p in old_manifest.parts}

    unmatched_new = set(new_by_uuid.keys())
    unmatched_old = set(old_by_uuid.keys())

    # Pass 1: Direct UUID match
    for new_uuid in list(unmatched_new):
        if new_uuid in old_by_uuid:
            matches.append((new_uuid, new_uuid, 1.0))
            unmatched_new.discard(new_uuid)
            unmatched_old.discard(new_uuid)

    logger.info(f"Pass 1 (UUID): {len(matches)} direct matches")

    # Pass 2: Fingerprint match for unmatched parts
    if strategy in ("hybrid", "fingerprint_first"):
        fp_matches = _fingerprint_match(
            [new_by_uuid[u] for u in unmatched_new],
            [old_by_uuid[u] for u in unmatched_old],
        )
        for new_uuid, old_uuid, confidence in fp_matches:
            if confidence >= 0.85:
                matches.append((str(new_uuid), str(old_uuid), confidence))
                unmatched_new.discard(str(new_uuid))
                unmatched_old.discard(str(old_uuid))

    logger.info(f"Pass 2 (fingerprint): {len(matches)} total matches")

    # Pass 3: Name + position match
    name_matches = _name_position_match(
        [new_by_uuid[u] for u in unmatched_new],
        [old_by_uuid[u] for u in unmatched_old],
    )
    for new_uuid, old_uuid, confidence in name_matches:
        if confidence >= 0.70:
            matches.append((str(new_uuid), str(old_uuid), confidence))
            unmatched_new.discard(str(new_uuid))
            unmatched_old.discard(str(old_uuid))

    logger.info(f"Pass 3 (name+position): {len(matches)} total matches")

    # Pass 4: Detect splits
    splits = _detect_splits(matches, old_by_uuid)

    # Pass 5: Detect merges
    merges = _detect_merges(matches, new_by_uuid)

    # Collect ambiguous (confidence < 0.80)
    ambiguous = [
        {"new_uuid": m[0], "old_uuid": m[1], "confidence": m[2]}
        for m in matches if m[2] < 0.80
    ]

    return MatchResult(
        matches=matches,
        new_parts=list(unmatched_new),
        deleted_parts=list(unmatched_old),
        splits=splits,
        merges=merges,
        ambiguous=ambiguous,
    )


def _fingerprint_match(new_parts, old_parts) -> list[tuple]:
    """Match parts by geometric fingerprint."""
    matches = []
    used_old = set()

    for new_part in new_parts:
        best_match = None
        best_confidence = 0.0

        for old_part in old_parts:
            if str(old_part.uuid) in used_old:
                continue

            confidence = _compare_fingerprints(new_part.fingerprint, old_part.fingerprint)
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = old_part

        if best_match and best_confidence >= 0.60:
            matches.append((new_part.uuid, best_match.uuid, best_confidence))
            used_old.add(str(best_match.uuid))

    return matches


def _name_position_match(new_parts, old_parts) -> list[tuple]:
    """Match parts by name similarity and transform proximity."""
    matches = []
    used_old = set()

    for new_part in new_parts:
        best_match = None
        best_confidence = 0.0

        for old_part in old_parts:
            if str(old_part.uuid) in used_old:
                continue

            # Name match
            name_sim = _name_similarity(new_part.name, old_part.name)

            # Transform proximity
            trans_dist = _transform_distance(new_part.transform, old_part.transform)
            trans_sim = max(0, 1.0 - trans_dist / 1000.0)  # Normalize by 1000mm

            confidence = name_sim * 0.5 + trans_sim * 0.5

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = old_part

        if best_match and best_confidence >= 0.50:
            matches.append((new_part.uuid, best_match.uuid, min(best_confidence, 0.75)))
            used_old.add(str(best_match.uuid))

    return matches


def _compare_fingerprints(fp1, fp2) -> float:
    """Compare two fingerprints and return confidence 0-1."""
    # Topology hash match
    if fp1.topology_hash == fp2.topology_hash:
        max_vol = max(fp1.volume_mm3, fp2.volume_mm3, 1e-10)
        vol_diff = abs(fp1.volume_mm3 - fp2.volume_mm3) / max_vol
        if vol_diff <= 0.001:
            return 0.95
        elif vol_diff <= 0.01:
            return 0.85
        else:
            return 0.80

    # Bounding box + volume match
    max_vol = max(fp1.volume_mm3, fp2.volume_mm3, 1e-10)
    vol_diff = abs(fp1.volume_mm3 - fp2.volume_mm3) / max_vol

    bbox_sim = _bbox_similarity(fp1, fp2)

    if bbox_sim > 0.9 and vol_diff <= 0.001:
        return 0.85
    elif bbox_sim > 0.8 and vol_diff <= 0.01:
        return 0.70
    elif vol_diff <= 0.05:
        return 0.60

    return 0.0


def _bbox_similarity(fp1, fp2) -> float:
    """Compare bounding boxes, return 0-1 similarity."""
    size1 = [fp1.bbox_max[i] - fp1.bbox_min[i] for i in range(3)]
    size2 = [fp2.bbox_max[i] - fp2.bbox_min[i] for i in range(3)]

    diffs = []
    for i in range(3):
        max_size = max(abs(size1[i]), abs(size2[i]), 1e-10)
        diffs.append(abs(size1[i] - size2[i]) / max_size)

    avg_diff = sum(diffs) / 3.0
    return max(0, 1.0 - avg_diff)


def _name_similarity(name1: str, name2: str) -> float:
    """Simple name similarity score 0-1."""
    if name1 == name2:
        return 1.0
    n1, n2 = name1.lower().strip(), name2.lower().strip()
    if n1 == n2:
        return 0.95
    if n1 in n2 or n2 in n1:
        return 0.80
    # Count common words
    words1 = set(n1.replace("_", " ").replace("-", " ").split())
    words2 = set(n2.replace("_", " ").replace("-", " ").split())
    if not words1 or not words2:
        return 0.0
    common = words1 & words2
    return len(common) / max(len(words1), len(words2))


def _transform_distance(t1, t2) -> float:
    """Euclidean distance between two transforms (translation only, in mm)."""
    import math
    dx = t1.translation[0] - t2.translation[0]
    dy = t1.translation[1] - t2.translation[1]
    dz = t1.translation[2] - t2.translation[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def _detect_splits(matches, old_by_uuid) -> list[dict]:
    """Detect splits: one old UUID claimed by multiple new parts."""
    old_claim_count = {}
    for new_uuid, old_uuid, conf in matches:
        old_claim_count.setdefault(old_uuid, []).append((new_uuid, conf))

    splits = []
    for old_uuid, claimants in old_claim_count.items():
        if len(claimants) > 1:
            splits.append({
                "old_uuid": old_uuid,
                "new_uuids": [c[0] for c in claimants],
                "confidences": [c[1] for c in claimants],
            })

    return splits


def _detect_merges(matches, new_by_uuid) -> list[dict]:
    """Detect merges: multiple old UUIDs matched to one new part."""
    new_claim_count = {}
    for new_uuid, old_uuid, conf in matches:
        new_claim_count.setdefault(new_uuid, []).append((old_uuid, conf))

    merges = []
    for new_uuid, sources in new_claim_count.items():
        if len(sources) > 1:
            merges.append({
                "new_uuid": new_uuid,
                "old_uuids": [s[0] for s in sources],
                "confidences": [s[1] for s in sources],
            })

    return merges
