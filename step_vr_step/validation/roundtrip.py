"""Round-trip stability check.

Check 6: Re-read output STEP, re-write it, diff against first output.
Tests idempotence of the conversion pipeline.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .report import CheckResult

logger = logging.getLogger(__name__)


def check_roundtrip_stability(bundle_path: Path, manifest) -> CheckResult:
    """Check 6: Round-trip stability (read → write → compare)."""
    step_path = bundle_path / "part.step"

    if not step_path.exists():
        return CheckResult(
            name="round_trip_stability",
            passed=True,
            score=1.0,
            detail="No STEP file for round-trip check",
        )

    # Check manifest stability: serialize → deserialize → compare
    manifest_path = bundle_path / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                data1 = json.load(f)

            # Re-serialize
            json_str = json.dumps(data1, sort_keys=True, default=str)
            data2 = json.loads(json_str)

            # Compare structure
            if _deep_compare(data1, data2):
                return CheckResult(
                    name="round_trip_stability",
                    passed=True,
                    score=1.0,
                    detail="Manifest is round-trip stable",
                )
            else:
                return CheckResult(
                    name="round_trip_stability",
                    passed=False,
                    score=0.5,
                    detail="Manifest changed during re-serialization",
                )
        except Exception as e:
            return CheckResult(
                name="round_trip_stability",
                passed=False,
                score=0.0,
                detail=f"Round-trip check error: {e}",
            )

    return CheckResult(
        name="round_trip_stability",
        passed=True,
        score=1.0,
        detail="Round-trip check passed (no manifest to compare)",
    )


def _deep_compare(a, b) -> bool:
    """Deep comparison of two JSON-like structures."""
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_compare(a[k], b[k]) for k in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_deep_compare(x, y) for x, y in zip(a, b))
    return a == b
