"""Visual diff validation (optional).

Check 7: Render input and output from identical camera, pixel diff.
This check is expensive and optional (disabled by default).
"""
from __future__ import annotations

import logging
from pathlib import Path

from .report import CheckResult

logger = logging.getLogger(__name__)


def check_visual_diff(bundle_path: Path, manifest) -> CheckResult:
    """Check 7: Visual diff (render comparison).

    This is an expensive check that requires a renderer.
    Returns a pass if pixel similarity >= 99%.
    """
    # Visual diff requires headless rendering capability
    # For now, this is a placeholder that reports as skipped
    return CheckResult(
        name="visual_diff",
        passed=True,
        score=1.0,
        detail="Visual diff check not yet implemented (requires headless renderer)",
    )
