"""PMI audit validation.

Check 4: Walk every PMI entry in manifest, verify target faces still exist in output.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .report import CheckResult

logger = logging.getLogger(__name__)


def check_pmi(bundle_path: Path, manifest, original_manifest) -> CheckResult:
    """Check 4: PMI binding audit."""
    if manifest is None:
        return CheckResult(
            name="pmi_audit",
            passed=True,
            score=1.0,
            detail="No manifest for PMI check",
        )

    failures = []
    total_pmi = 0
    orphaned_pmi = 0

    # Check PMI entries on parts
    for part in manifest.parts:
        for pmi in part.pmi:
            total_pmi += 1

            # Verify target UUID exists in manifest
            target_found = any(str(p.uuid) == str(pmi.target_uuid) for p in manifest.parts)
            if not target_found:
                orphaned_pmi += 1
                failures.append({
                    "uuid": str(part.uuid),
                    "name": part.name,
                    "pmi_kind": pmi.kind,
                    "pmi_value": pmi.value,
                    "detail": f"PMI target UUID {pmi.target_uuid} not found in manifest",
                })

    # Check global PMI
    for pmi in manifest.global_pmi:
        total_pmi += 1
        target_found = any(str(p.uuid) == str(pmi.target_uuid) for p in manifest.parts)
        if not target_found:
            orphaned_pmi += 1
            failures.append({
                "pmi_kind": pmi.kind,
                "pmi_value": pmi.value,
                "detail": f"Global PMI target UUID {pmi.target_uuid} not found",
            })

    # Compare with original if available
    if original_manifest:
        original_pmi_count = sum(len(p.pmi) for p in original_manifest.parts) + len(original_manifest.global_pmi)
        if original_pmi_count > 0 and total_pmi < original_pmi_count:
            lost = original_pmi_count - total_pmi
            failures.append({
                "detail": f"{lost} PMI entries lost during conversion (had {original_pmi_count}, now {total_pmi})",
            })

    passed = len(failures) == 0
    return CheckResult(
        name="pmi_audit",
        passed=passed,
        score=1.0 if passed else max(0, 1.0 - orphaned_pmi / max(total_pmi, 1)),
        detail=f"All {total_pmi} PMI entries valid" if passed else f"{orphaned_pmi} orphaned PMI entries",
        failures=failures,
    )
