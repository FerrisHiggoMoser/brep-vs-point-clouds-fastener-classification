"""UUID registry and geometric fingerprinting for part identity tracking."""
from __future__ import annotations

import hashlib
import uuid
from typing import Optional

from pydantic import BaseModel

from .schema import Fingerprint


class RegistryEntry(BaseModel):
    """A single entry in the UUID registry."""
    part_uuid: uuid.UUID
    name: str
    fingerprint: Fingerprint
    source_type: str = "original_step"
    original_brep_hash: Optional[str] = None


def compute_topology_hash(face_count: int, edge_count: int, vertex_count: int,
                          face_areas: list[float] | None = None,
                          edge_lengths: list[float] | None = None) -> str:
    """Compute a deterministic topology hash from shape characteristics.

    The hash is built from face/edge/vertex counts plus sorted geometric
    measurements, producing a stable signature that survives re-import.
    """
    parts = [f"F{face_count}", f"E{edge_count}", f"V{vertex_count}"]

    if face_areas:
        # Sort by area (largest first) for deterministic ordering
        sorted_areas = sorted(face_areas, reverse=True)
        # Quantize to 4 decimal places to absorb tessellation noise
        area_sig = ",".join(f"{a:.4f}" for a in sorted_areas)
        parts.append(f"FA:{area_sig}")

    if edge_lengths:
        sorted_lengths = sorted(edge_lengths, reverse=True)
        length_sig = ",".join(f"{l:.4f}" for l in sorted_lengths)
        parts.append(f"EL:{length_sig}")

    signature = "|".join(parts)
    return hashlib.sha256(signature.encode()).hexdigest()


def compute_fingerprint_from_mesh(vertices, faces) -> Fingerprint:
    """Compute fingerprint from raw mesh data (numpy arrays).

    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of triangle indices
    """
    import numpy as np

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    bbox_min = tuple(vertices.min(axis=0).tolist())
    bbox_max = tuple(vertices.max(axis=0).tolist())

    # Compute triangle areas for volume and surface area
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    triangle_areas = np.linalg.norm(cross, axis=1) / 2.0
    surface_area = float(triangle_areas.sum())

    # Signed volume via divergence theorem
    volume = float(np.abs(np.sum(
        v0[:, 0] * cross[:, 0] +
        v0[:, 1] * cross[:, 1] +
        v0[:, 2] * cross[:, 2]
    ) / 6.0))

    # Sort face areas for topology hash
    sorted_face_areas = sorted(triangle_areas.tolist(), reverse=True)
    # Use only top-50 face areas to keep hash stable for large meshes
    top_areas = sorted_face_areas[:50]

    topo_hash = compute_topology_hash(
        face_count=len(faces),
        edge_count=0,  # Not readily available from triangle mesh
        vertex_count=len(vertices),
        face_areas=top_areas,
    )

    return Fingerprint(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        volume_mm3=volume,
        surface_area_mm2=surface_area,
        topology_hash=topo_hash,
        vertex_count=len(vertices),
        face_count=len(faces),
    )


class UUIDRegistry:
    """In-memory registry mapping UUIDs to fingerprints and metadata.

    Used during conversion to track part identity across round trips.
    Supports lookup by UUID (exact) or by fingerprint similarity (fallback).
    """

    # Custom property names written into STEP files
    PROP_UUID = "step_vr_step/part_uuid"
    PROP_FINGERPRINT = "step_vr_step/fingerprint_hash"
    PROP_PROVENANCE = "step_vr_step/provenance_type"
    PROP_BREP_HASH = "step_vr_step/original_brep_hash"
    PROP_BUNDLE_VERSION = "step_vr_step/bundle_version"
    PROP_EXPORTED_AT = "step_vr_step/exported_at"

    def __init__(self):
        self._entries: dict[uuid.UUID, RegistryEntry] = {}

    def register(self, name: str, fingerprint: Fingerprint,
                 source_type: str = "original_step",
                 existing_uuid: uuid.UUID | None = None,
                 original_brep_hash: str | None = None) -> uuid.UUID:
        """Register a part and return its UUID.

        If existing_uuid is provided (e.g., read from STEP properties),
        it is reused. Otherwise a new UUIDv4 is generated.
        """
        part_uuid = existing_uuid or uuid.uuid4()

        self._entries[part_uuid] = RegistryEntry(
            part_uuid=part_uuid,
            name=name,
            fingerprint=fingerprint,
            source_type=source_type,
            original_brep_hash=original_brep_hash,
        )

        return part_uuid

    def lookup(self, part_uuid: uuid.UUID) -> Optional[RegistryEntry]:
        """Exact lookup by UUID."""
        return self._entries.get(part_uuid)

    def find_by_fingerprint(self, target: Fingerprint,
                            volume_tolerance: float = 0.01,
                            candidates: list[uuid.UUID] | None = None) -> list[tuple[uuid.UUID, float]]:
        """Find parts matching a fingerprint, ranked by confidence.

        Returns list of (uuid, confidence) sorted by confidence descending.

        Matching rules from spec:
        - Topology hash exact match + volume within tolerance: confidence 0.95
        - Bounding box + volume match: confidence 0.85
        - Name match + transform tolerance: confidence 0.70
        """
        results = []
        search_entries = (
            {uid: self._entries[uid] for uid in candidates if uid in self._entries}
            if candidates else self._entries
        )

        for uid, entry in search_entries.items():
            fp = entry.fingerprint
            confidence = 0.0

            # Check topology hash match
            if fp.topology_hash == target.topology_hash:
                # Topology match — check volume
                if fp.volume_mm3 > 0 and target.volume_mm3 > 0:
                    vol_diff = abs(fp.volume_mm3 - target.volume_mm3) / max(fp.volume_mm3, target.volume_mm3)
                    if vol_diff <= volume_tolerance:
                        confidence = 0.95
                    else:
                        confidence = 0.80
                else:
                    confidence = 0.90
            else:
                # No topology match — try bbox + volume
                bbox_match = _bbox_similar(fp.bbox_min, fp.bbox_max,
                                           target.bbox_min, target.bbox_max,
                                           tolerance=volume_tolerance * 10)
                if bbox_match:
                    if fp.volume_mm3 > 0 and target.volume_mm3 > 0:
                        vol_diff = abs(fp.volume_mm3 - target.volume_mm3) / max(fp.volume_mm3, target.volume_mm3)
                        if vol_diff <= volume_tolerance:
                            confidence = 0.85
                        elif vol_diff <= volume_tolerance * 5:
                            confidence = 0.60

            if confidence > 0:
                results.append((uid, confidence))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def all_uuids(self) -> list[uuid.UUID]:
        """Return all registered UUIDs."""
        return list(self._entries.keys())

    def all_entries(self) -> dict[uuid.UUID, RegistryEntry]:
        """Return all entries."""
        return dict(self._entries)

    def remove(self, part_uuid: uuid.UUID) -> bool:
        """Remove a part from the registry."""
        if part_uuid in self._entries:
            del self._entries[part_uuid]
            return True
        return False

    def __len__(self) -> int:
        return len(self._entries)


def _bbox_similar(min1: tuple, max1: tuple, min2: tuple, max2: tuple,
                  tolerance: float = 0.1) -> bool:
    """Check if two bounding boxes are similar within tolerance."""
    for i in range(3):
        size1 = max1[i] - min1[i]
        size2 = max2[i] - min2[i]
        if size1 == 0 and size2 == 0:
            continue
        max_size = max(abs(size1), abs(size2), 1e-10)
        if abs(size1 - size2) / max_size > tolerance:
            return False
    return True
