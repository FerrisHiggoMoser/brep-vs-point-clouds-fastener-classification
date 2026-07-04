"""Validation report assembly and full harness runner.

Runs all seven validation checks and assembles results into a ValidationReport.
"""
from __future__ import annotations

import logging
from typing import Optional
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CheckResult(BaseModel):
    """Result of a single validation check."""
    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    detail: str
    failures: list[dict] = []


class ValidationReport(BaseModel):
    """Complete validation report from the 7-check harness."""
    checks: list[CheckResult]
    all_passed: bool
    summary: str
    timestamp: str = ""


def run_full_validation(bundle_path: str | Path,
                        original_manifest=None,
                        config=None,
                        on_check_complete=None) -> ValidationReport:
    """Run the complete 7-check validation harness on a bundle.

    Args:
        bundle_path: Path to the output bundle directory
        original_manifest: Optional original manifest for comparison
        config: Optional ValidationConfig

    Returns:
        ValidationReport with all check results
    """
    from datetime import datetime, timezone
    from ..config import ValidationConfig
    from ..sidecar.manifest import read_manifest

    if config is None:
        config = ValidationConfig()

    bundle_path = Path(bundle_path)

    # Load manifest
    manifest = None
    try:
        manifest = read_manifest(bundle_path)
    except Exception as e:
        logger.warning(f"Could not load manifest: {e}")

    checks = []

    def _run_check(result: CheckResult) -> CheckResult:
        checks.append(result)
        if on_check_complete:
            on_check_complete(result, len(checks))
        return result

    # Check 1: Geometric diff
    from .geometric import check_geometric_diff
    _run_check(check_geometric_diff(bundle_path, manifest, original_manifest, config))

    # Check 2: Topology check
    from .topology import check_topology
    _run_check(check_topology(bundle_path, manifest))

    # Check 3: Schema validation
    from .schema import check_schema
    _run_check(check_schema(bundle_path))

    # Check 4: PMI audit
    from .pmi import check_pmi
    _run_check(check_pmi(bundle_path, manifest, original_manifest))

    # Check 5: Mass properties
    from .geometric import check_mass_properties
    _run_check(check_mass_properties(bundle_path, manifest, original_manifest, config))

    # Check 6: Round-trip stability
    from .roundtrip import check_roundtrip_stability
    if config.run_roundtrip_check:
        _run_check(check_roundtrip_stability(bundle_path, manifest))
    else:
        _run_check(CheckResult(name="round_trip_stability", passed=True, score=1.0, detail="Skipped (disabled in config)"))

    # Check 7: Visual diff
    from .visual import check_visual_diff
    if config.run_visual_diff:
        _run_check(check_visual_diff(bundle_path, manifest))
    else:
        _run_check(CheckResult(name="visual_diff", passed=True, score=1.0, detail="Skipped (disabled in config)"))

    # Check 8: Transfer losses
    _run_check(_check_transfer_losses(manifest, original_manifest))

    all_passed = all(c.passed for c in checks)
    passed_count = sum(1 for c in checks if c.passed)
    summary = f"{passed_count}/{len(checks)} checks passed"

    report = ValidationReport(
        checks=checks,
        all_passed=all_passed,
        summary=summary,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Write report to bundle
    _write_report(bundle_path, report)

    return report


def _check_transfer_losses(output_manifest, input_manifest) -> CheckResult:
    """Check what data was lost during the conversion."""
    failures = []

    if input_manifest is None or output_manifest is None:
        return CheckResult(
            name="transfer_losses",
            passed=True,
            score=1.0,
            detail="No input manifest to compare against",
        )

    in_parts = {str(p.uuid): p for p in input_manifest.parts}
    out_parts = {str(p.uuid): p for p in output_manifest.parts}

    # Check for lost parts
    for uid, p in in_parts.items():
        if uid not in out_parts:
            failures.append({"name": p.name, "detail": f"Part '{p.name}' was lost entirely"})

    # Check for lost materials (textures specifically)
    for uid, p in in_parts.items():
        op = out_parts.get(uid)
        if not op:
            continue
        if p.material.albedo_texture and not op.material.albedo_texture:
            failures.append({"name": p.name, "detail": f"Albedo texture lost on '{p.name}'"})
        if p.material.normal_texture and not op.material.normal_texture:
            failures.append({"name": p.name, "detail": f"Normal map lost on '{p.name}'"})

    # Check for lost PMI
    in_pmi = sum(len(p.pmi) for p in input_manifest.parts)
    out_pmi = sum(len(p.pmi) for p in output_manifest.parts)
    if in_pmi > 0 and out_pmi < in_pmi:
        failures.append({"name": "PMI", "detail": f"{in_pmi - out_pmi} PMI annotation(s) lost"})

    # Check for lost relationships
    in_rels = len(input_manifest.relationships)
    out_rels = len(output_manifest.relationships)
    if in_rels > 0 and out_rels < in_rels:
        failures.append({"name": "Relationships", "detail": f"{in_rels - out_rels} relationship(s) lost"})

    # Check for lost Unreal metadata
    for uid, p in in_parts.items():
        op = out_parts.get(uid)
        if not op:
            continue
        if p.unreal.tags and not op.unreal.tags:
            failures.append({"name": p.name, "detail": f"Unreal tags lost on '{p.name}'"})
        if p.unreal.blueprint_path and not op.unreal.blueprint_path:
            failures.append({"name": p.name, "detail": f"Blueprint reference lost on '{p.name}'"})
        if p.unreal.collision_geometry and not op.unreal.collision_geometry:
            failures.append({"name": p.name, "detail": f"Collision data lost on '{p.name}'"})

    # Check name preservation
    for uid, p in in_parts.items():
        op = out_parts.get(uid)
        if op and op.name != p.name:
            failures.append({"name": p.name, "detail": f"Name changed: '{p.name}' → '{op.name}'"})

    passed = len(failures) == 0
    if passed:
        detail = f"All {len(in_parts)} parts preserved — no data lost"
    else:
        detail = f"{len(failures)} item(s) lost or degraded during transfer"

    return CheckResult(
        name="transfer_losses",
        passed=passed,
        score=max(0, 1.0 - len(failures) / max(len(in_parts), 1)),
        detail=detail,
        failures=failures,
    )


def _write_report(bundle_path: Path, report: ValidationReport) -> None:
    """Write validation report JSON to the bundle directory."""
    import json
    report_path = bundle_path / "validation_report.json"
    try:
        with open(report_path, "w") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Could not write validation report: {e}")
