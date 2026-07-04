"""Re-run the EXACT 2026-05-11 subtype-13 paired analysis, but with the
corrected-normals PointNet++ (mcmaster_pc_subtype13_normals) instead of the
degenerate-normals one. BRepFormer is untouched (same frozen checkpoint, same
mcmaster_brep_subtype13) so its column must reproduce the stored 89.98% — a
built-in sanity check that the only thing that changed is the PN++ normals.

Reuses full_analysis_subtype13's protocol verbatim (McNemar, paired bootstrap,
per-class, AUROC, binary collapse) by redirecting its module globals, so the
output is directly comparable to the original full_analysis_subtype13.{json,md}.

Outputs: training_data/mcmaster_logs/full_analysis_subtype13_normals.{json,md}

Usage (anaconda stepvrstep env): python backend/scripts/eval_pn_subtype13_normals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import full_analysis_subtype13 as fa

REPO = _BACKEND_ROOT.parent
TD = REPO / "training_data"
LOGS = TD / "mcmaster_logs"


def main():
    # Redirect ONLY the PN++ side + the output paths. BF side stays frozen.
    fa.PC_TEST = TD / "mcmaster_pc_subtype13_normals"
    fa.PN_LOG_DIR = LOGS / "pointnet2_subtype13_normals"
    # BF unchanged: fa.BREP_TEST, fa.BF_LOG_DIR as defined in the module
    fa.OUT_JSON = LOGS / "full_analysis_subtype13_normals.json"
    fa.OUT_MD = LOGS / "full_analysis_subtype13_normals.md"
    print(f"PN++ (corrected normals): {fa.PC_TEST.name} / {fa.PN_LOG_DIR.name}")
    print(f"BF (frozen):             {fa.BREP_TEST.name} / {fa.BF_LOG_DIR.name}")
    print(f"out: {fa.OUT_JSON.name}, {fa.OUT_MD.name}\n")
    fa.main()


if __name__ == "__main__":
    main()
