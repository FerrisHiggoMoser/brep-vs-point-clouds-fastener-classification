"""Replace detected fastener meshes with LOD proxy meshes in glTF output."""

import logging
from typing import Optional

from ..schema import Manifest
from .lod_levels import LODLevel, LOD_SCREEN_SIZES
from .proxy_library import get_proxy

logger = logging.getLogger(__name__)


class LODSubstitutionResult:
    """Result of LOD substitution with stats."""

    def __init__(self):
        self.substituted_count: int = 0
        self.original_poly_count: int = 0
        self.proxy_poly_count: int = 0
        self.skipped_parts: list[str] = []

    @property
    def polygon_reduction(self) -> float:
        if self.original_poly_count == 0:
            return 0.0
        return 1.0 - (self.proxy_poly_count / self.original_poly_count)


def substitute_fasteners(
    manifest: Manifest,
    lod_level: LODLevel = LODLevel.L1,
    confidence_threshold: float = 0.60,
) -> tuple[dict[str, tuple], LODSubstitutionResult]:
    """Generate proxy meshes for detected fasteners.

    Args:
        manifest: Manifest with detection labels populated.
        lod_level: Default LOD level for proxies.
        confidence_threshold: Minimum confidence to substitute.

    Returns:
        proxy_meshes: dict mapping part UUID -> (vertices, faces) proxy mesh.
        result: Substitution statistics.
    """
    result = LODSubstitutionResult()
    proxy_meshes: dict[str, tuple] = {}

    for part in manifest.parts:
        det = part.detection
        if det is None:
            continue

        base_type = det.fastener_type.replace("possible_", "").replace("likely_", "")
        if base_type == "unclassified":
            continue

        if det.confidence < confidence_threshold:
            result.skipped_parts.append(str(part.uuid))
            continue

        # Get dimensions from detection or fingerprint
        dims = det.detected_dimensions
        diameter = dims.get("shaft_dia", dims.get("head_dia", 6.0))
        length = dims.get("length", 20.0)

        # Estimate original poly count from fingerprint
        original_polys = part.fingerprint.face_count
        result.original_poly_count += original_polys

        # Generate proxy mesh
        try:
            verts, faces = get_proxy(
                fastener_type=base_type,
                diameter_mm=diameter,
                length_mm=length,
                lod_level=lod_level,
            )
            proxy_meshes[str(part.uuid)] = (verts, faces)
            result.proxy_poly_count += len(faces)
            result.substituted_count += 1

            # Update detection label with proxy info
            det.lod_proxy = f"{base_type}_{lod_level.name}"

            logger.debug(
                "Substituted %s (%s) %s: %d -> %d polys",
                part.name, base_type, det.variant or "",
                original_polys, len(faces),
            )
        except Exception as e:
            logger.warning("Failed to generate proxy for %s: %s", part.name, e)
            result.skipped_parts.append(str(part.uuid))

    logger.info(
        "LOD substitution: %d parts, %.1f%% polygon reduction (%d -> %d polys)",
        result.substituted_count,
        result.polygon_reduction * 100,
        result.original_poly_count,
        result.proxy_poly_count,
    )

    return proxy_meshes, result
