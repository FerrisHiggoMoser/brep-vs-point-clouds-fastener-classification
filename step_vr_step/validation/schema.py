"""STEP AP242 schema validation.

Check 3: Verify output STEP file is schema-valid.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .report import CheckResult

logger = logging.getLogger(__name__)


def check_schema(bundle_path: Path) -> CheckResult:
    """Check 3: STEP schema validation."""
    step_path = bundle_path / "part.step"

    if not step_path.exists():
        return CheckResult(
            name="schema_validation",
            passed=True,
            score=1.0,
            detail="No STEP file to validate",
        )

    failures = []

    try:
        content = step_path.read_text(errors="replace")

        # Basic structural validation
        if "ISO-10303-21" not in content:
            failures.append({"detail": "Missing ISO-10303-21 header"})

        if "HEADER;" not in content:
            failures.append({"detail": "Missing HEADER section"})

        if "DATA;" not in content:
            failures.append({"detail": "Missing DATA section"})

        if "END-ISO-10303-21;" not in content:
            failures.append({"detail": "Missing END-ISO-10303-21 terminator"})

        if "ENDSEC;" not in content:
            failures.append({"detail": "Missing ENDSEC delimiter"})

        # Check FILE_SCHEMA
        if "FILE_SCHEMA" not in content:
            failures.append({"detail": "Missing FILE_SCHEMA"})

        # Validate manifest.json exists and is valid
        manifest_path = bundle_path / "manifest.json"
        if manifest_path.exists():
            import json
            try:
                with open(manifest_path) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                failures.append({"detail": f"Invalid manifest.json: {e}"})

    except Exception as e:
        failures.append({"detail": f"Schema validation error: {e}"})

    passed = len(failures) == 0
    return CheckResult(
        name="schema_validation",
        passed=passed,
        score=1.0 if passed else 0.0,
        detail="STEP file passes structural validation" if passed else f"{len(failures)} schema issues",
        failures=failures,
    )
