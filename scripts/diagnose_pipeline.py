"""Run the FULL detect_fasteners pipeline on one file with a configurable
threshold, and print every part's final detection label + every emitted
relationship. The point: see end-to-end behavior, not just classify_part().
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step_vr_step.readers.step_reader import read_step
from step_vr_step.detection import detect_fasteners
from step_vr_step.config import DetectionConfig


def run(step_path: str, threshold: float) -> None:
    print(f"\n=== {step_path}  (threshold={threshold}) ===")
    doc, manifest = read_step(step_path)
    config = DetectionConfig(rule_confidence_threshold=threshold)
    manifest = detect_fasteners(manifest, config=config)

    print(f"\nParts ({len(manifest.parts)}):")
    for p in manifest.parts:
        det = p.detection
        det_str = f"{det.fastener_type}({det.confidence:.2f})" if det else "—"
        print(f"  {str(p.uuid)[:8]}  parent={str(p.parent_uuid)[:8] if p.parent_uuid else '—':<10}  {p.name[:35]:<35}  -> {det_str}")

    print(f"\nRelationships ({len(manifest.relationships)}):")
    for r in manifest.relationships:
        print(f"  [{r.kind}]  {str(r.subject_uuid)[:8]} -> {str(r.target_uuid)[:8]}  conf={r.confidence:.2f}  params={r.params}")


if __name__ == "__main__":
    targets = sys.argv[1:3] or [
        "../fastener_labeling/files/hex_bolt with hex_nut2.step",
    ]
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45
    for t in targets:
        try:
            run(t, thr)
        except Exception as e:
            import traceback
            print(f"FAIL on {t}: {type(e).__name__}: {e}")
            traceback.print_exc()
