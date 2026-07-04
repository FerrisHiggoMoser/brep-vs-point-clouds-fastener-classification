"""Delete BRep sample dirs where face_grids has >600 faces.

Restores the baseline-equivalent dataset after a MAX_FACES=1500 conversion run.
"""
from pathlib import Path
import numpy.lib.format as fmt
import shutil

ROOT = Path(r"c:\Users\ferri\OneDrive\Documents\GitHub\step-vr-step\training_data\mcmaster_brep")
LIMIT = 600


def npy_first_dim(path: Path) -> int:
    with open(path, "rb") as f:
        ver = fmt.read_magic(f)
        if ver == (1, 0):
            shape, _, _ = fmt.read_array_header_1_0(f)
        elif ver == (2, 0):
            shape, _, _ = fmt.read_array_header_2_0(f)
        else:
            shape, _, _ = fmt.read_array_header_3_0(f)
    return shape[0]


removed = kept = errored = 0
for split in ("train", "val", "test"):
    for klass in ("fastener", "non_fastener"):
        cls_dir = ROOT / split / klass
        if not cls_dir.exists():
            continue
        for sample_dir in list(cls_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            fg = sample_dir / "face_grids.npy"
            if not fg.exists():
                # Already filtered (no face_grids) — count as removed-already, skip.
                removed += 1
                continue
            try:
                n_faces = npy_first_dim(fg)
            except Exception as e:
                print(f"  ERR {sample_dir}: {e}")
                errored += 1
                continue
            if n_faces > LIMIT:
                # Just unlink face_grids.npy — dataset filter requires it to include the sample,
                # so removing it effectively "uninstalls" the sample. Leaves empty dirs (cheap).
                fg.unlink()
                removed += 1
            else:
                kept += 1
        kept_n = sum(1 for d in cls_dir.iterdir() if d.is_dir() and (d / "face_grids.npy").exists())
        print(f"  {split}/{klass}: {kept_n}  (was {kept_n + removed} before this pass)")

print(f"\nremoved={removed}  kept={kept}  errored={errored}")
