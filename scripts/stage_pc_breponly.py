"""Stage a filtered PC tree containing only the parts BF could process.

For each <split>/<class>/<pn>.npy in mcmaster_pc/, copy it to
mcmaster_pc_breponly/<split>/<class>/<pn>.npy iff
mcmaster_brep/<split>/<class>/<pn>/face_grids.npy exists.
"""
from pathlib import Path
import shutil

REPO = Path(r"c:\Users\ferri\OneDrive\Documents\GitHub\step-vr-step")
PC_SRC = REPO / "training_data" / "mcmaster_pc"
BREP_SRC = REPO / "training_data" / "mcmaster_brep"
PC_DST = REPO / "training_data" / "mcmaster_pc_breponly"

copied = skipped = missing_brep = 0
for split in ("train", "val", "test"):
    for klass in ("fastener", "non_fastener"):
        src_dir = PC_SRC / split / klass
        brep_dir = BREP_SRC / split / klass
        dst_dir = PC_DST / split / klass
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.exists():
            continue
        kept_here = 0
        for pc_file in src_dir.glob("*.npy"):
            pn = pc_file.stem
            brep_marker = brep_dir / pn / "face_grids.npy"
            if not brep_marker.exists():
                missing_brep += 1
                continue
            dst = dst_dir / pc_file.name
            if dst.exists():
                skipped += 1
            else:
                shutil.copy2(pc_file, dst)
                copied += 1
            kept_here += 1
        print(f"  {split}/{klass}: {kept_here} (was {len(list(src_dir.glob('*.npy')))} in PC tree)")

print(f"\ncopied={copied}  skipped={skipped}  missing_brep={missing_brep}")
print(f"target: {PC_DST}")
