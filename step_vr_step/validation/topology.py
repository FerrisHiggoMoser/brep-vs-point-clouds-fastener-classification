"""Topology validation: BRepCheck_Analyzer on every solid.

Check 2: Verify no free edges, self-intersections, or inconsistent orientations.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .report import CheckResult

logger = logging.getLogger(__name__)


def check_topology(bundle_path: Path, manifest) -> CheckResult:
    """Check 2: Topology check via BRepCheck_Analyzer."""
    try:
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
        HAS_OCC = True
    except ImportError:
        HAS_OCC = False

    step_path = bundle_path / "part.step"

    if not step_path.exists():
        return CheckResult(
            name="topology",
            passed=True,
            score=1.0,
            detail="No STEP file to check",
        )

    if not HAS_OCC:
        return CheckResult(
            name="topology",
            passed=True,
            score=0.5,
            detail="PythonOCC not available — topology check skipped",
        )

    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SOLID

        reader = STEPControl_Reader()
        status = reader.ReadFile(str(step_path))
        if status != IFSelect_RetDone:
            return CheckResult(name="topology", passed=False, score=0.0,
                             detail=f"Failed to read STEP file for topology check")

        reader.TransferRoots()
        shape = reader.OneShape()

        analyzer = BRepCheck_Analyzer(shape, True)
        is_valid = analyzer.IsValid()

        failures = []
        if not is_valid:
            exp = TopExp_Explorer(shape, TopAbs_SOLID)
            solid_idx = 0
            while exp.More():
                solid_analyzer = BRepCheck_Analyzer(exp.Current(), True)
                if not solid_analyzer.IsValid():
                    failures.append({
                        "solid_index": solid_idx,
                        "detail": "Solid failed topology check",
                    })
                solid_idx += 1
                exp.Next()

        # Read repair stats from the writer (if it ran heal pass).
        repair = _load_repair_stats(bundle_path)

        if is_valid:
            if repair and repair.get("healed", 0) > 0:
                detail = (f"All {solid_idx} solids pass topology check "
                          f"({repair['healed']} source-bad solid(s) auto-healed before write)")
            else:
                detail = "All solids pass topology check"
            return CheckResult(name="topology", passed=True, score=1.0, detail=detail)

        # Failures: try to attribute them between source and round-trip.
        n_failed = len(failures)
        if repair is not None:
            remaining_from_source = repair.get("remaining_invalid", 0)
            new_from_writer = max(0, n_failed - remaining_from_source)
            healed = repair.get("healed", 0)
            parts = []
            if remaining_from_source > 0:
                parts.append(f"{remaining_from_source} unhealable from source")
            if new_from_writer > 0:
                parts.append(f"{new_from_writer} introduced by conversion")
            if healed > 0:
                parts.append(f"{healed} source solid(s) auto-healed")
            attribution = "; ".join(parts) if parts else "source unknown"
            detail = f"{n_failed} solids failed ({attribution})"
        else:
            detail = f"{n_failed} solids failed"

        return CheckResult(
            name="topology",
            passed=False,
            score=0.0,
            detail=detail,
            failures=failures,
        )
    except Exception as e:
        return CheckResult(name="topology", passed=False, score=0.0,
                         detail=f"Topology check error: {e}")


def _load_repair_stats(bundle_path: Path) -> dict | None:
    """Read topology_repair.json written by step_writer, if present."""
    import json
    repair_path = bundle_path / "topology_repair.json"
    if not repair_path.exists():
        return None
    try:
        return json.loads(repair_path.read_text())
    except Exception:
        return None
