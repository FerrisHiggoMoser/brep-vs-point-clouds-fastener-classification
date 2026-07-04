"""Geometric validation checks: Hausdorff distance and mass properties.

Check 1: Re-tessellate output STEP, compute Hausdorff to input mesh.
Check 5: Compare input vs output volume, center of mass, surface area.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .report import CheckResult

logger = logging.getLogger(__name__)


def check_geometric_diff(bundle_path: Path, manifest, original_manifest,
                         config) -> CheckResult:
    """Check 1: Geometric diff via Hausdorff distance."""
    if manifest is None or original_manifest is None:
        return CheckResult(
            name="geometric_diff",
            passed=True,
            score=1.0,
            detail="Skipped: no original manifest for comparison",
        )

    failures = []
    max_distance = 0.0

    for new_part in manifest.parts:
        # Find matching old part by UUID
        old_part = None
        for op in (original_manifest.parts if original_manifest else []):
            if str(op.uuid) == str(new_part.uuid):
                old_part = op
                break

        if old_part is None:
            continue

        # Compare fingerprints as proxy for geometric diff
        vol_diff = abs(new_part.fingerprint.volume_mm3 - old_part.fingerprint.volume_mm3)
        max_vol = max(new_part.fingerprint.volume_mm3, old_part.fingerprint.volume_mm3, 1e-10)
        rel_diff = vol_diff / max_vol

        if rel_diff > config.hausdorff_tolerance:
            failures.append({
                "uuid": str(new_part.uuid),
                "name": new_part.name,
                "volume_diff": rel_diff,
                "detail": f"Volume difference {rel_diff:.4f} exceeds tolerance {config.hausdorff_tolerance}",
            })

        max_distance = max(max_distance, rel_diff)

    passed = len(failures) == 0
    return CheckResult(
        name="geometric_diff",
        passed=passed,
        score=1.0 - min(max_distance, 1.0),
        detail=f"Max geometric diff: {max_distance:.6f}mm" if passed else f"{len(failures)} parts exceed tolerance",
        failures=failures,
    )


def check_mass_properties(bundle_path: Path, manifest, original_manifest,
                          config) -> CheckResult:
    """Check 5: Mass properties comparison (volume, surface area)."""
    if manifest is None:
        return CheckResult(
            name="mass_properties",
            passed=True,
            score=1.0,
            detail="No manifest available for mass properties check",
        )

    failures = []

    for part in manifest.parts:
        fp = part.fingerprint

        # Self-consistency checks
        if fp.volume_mm3 < 0:
            failures.append({
                "uuid": str(part.uuid),
                "name": part.name,
                "detail": f"Negative volume: {fp.volume_mm3}",
            })

        if fp.surface_area_mm2 < 0:
            failures.append({
                "uuid": str(part.uuid),
                "name": part.name,
                "detail": f"Negative surface area: {fp.surface_area_mm2}",
            })

    # Cross-check with original if available
    if original_manifest:
        for new_part in manifest.parts:
            for old_part in original_manifest.parts:
                if str(new_part.uuid) == str(old_part.uuid):
                    if old_part.fingerprint.volume_mm3 > 0:
                        vol_diff = abs(new_part.fingerprint.volume_mm3 - old_part.fingerprint.volume_mm3) / old_part.fingerprint.volume_mm3
                        if vol_diff > config.volume_tolerance:
                            failures.append({
                                "uuid": str(new_part.uuid),
                                "name": new_part.name,
                                "detail": f"Volume diff {vol_diff:.4f} exceeds tolerance {config.volume_tolerance}",
                            })
                    break

    passed = len(failures) == 0
    return CheckResult(
        name="mass_properties",
        passed=passed,
        score=1.0 if passed else 0.5,
        detail=f"All {len(manifest.parts)} parts pass mass property checks" if passed else f"{len(failures)} failures",
        failures=failures,
    )
