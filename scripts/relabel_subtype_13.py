"""Reorganize existing PC + BREP trees into 13-class subtype layout.

Source layout (matched-binary trees):
  training_data/mcmaster_pc_breponly/{train,val,test}/{fastener,non_fastener}/<pn>.npy
  training_data/mcmaster_brep/{train,val,test}/{fastener,non_fastener}/<pn>/{face_grids.npy, edge_curves.npy, topo_distances.npz}

Target layout (13 classes: 12 fastener subtypes + non_fastener pooled):
  training_data/mcmaster_pc_subtype13/{train,val,test}/<label>/<pn>.npy
  training_data/mcmaster_brep_subtype13/{train,val,test}/<label>/<pn>/...

Label resolution: walk fastener_labeling/dataset/{fastener,non_fastener}/<category>/<leaf>/<pn>.step.
Categories under fastener/ become labels; everything under non_fastener/ collapses to one label.

Idempotent — skips destinations that already exist.
"""
from __future__ import annotations
from pathlib import Path
import json
import os
import shutil

REPO = Path(r"c:\Users\ferri\OneDrive\Documents\GitHub\step-vr-step")
DATASET = REPO / "fastener_labeling" / "dataset"
PC_SRC = REPO / "training_data" / "mcmaster_pc_breponly"
BREP_SRC = REPO / "training_data" / "mcmaster_brep"
PC_DST = REPO / "training_data" / "mcmaster_pc_subtype13"
BREP_DST = REPO / "training_data" / "mcmaster_brep_subtype13"
MANIFEST = REPO / "training_data" / "mcmaster_logs" / "relabel_manifest.subtype13.json"
UNRESOLVED = REPO / "training_data" / "mcmaster_logs" / "unresolved_pns.subtype13.txt"


def build_pn_to_label() -> dict[str, str]:
    """Walk dataset/ to map part_number -> 13-class label."""
    idx: dict[str, str] = {}
    for klass_dir in ("fastener", "non_fastener"):
        root = DATASET / klass_dir
        if not root.exists():
            continue
        for cat_dir in root.iterdir():
            if not cat_dir.is_dir() or cat_dir.name.startswith("._"):
                continue
            label = cat_dir.name if klass_dir == "fastener" else "non_fastener"
            for step_file in cat_dir.rglob("*.step"):
                if step_file.name.startswith("._"):
                    continue
                # If a pn shows up under multiple categories, last write wins —
                # but for non_fastener the label collapses anyway, and McMaster
                # part numbers are unique so this should not collide.
                idx[step_file.stem] = label
    return idx


def copy_dir_tree(src: Path, dst: Path) -> str:
    """Try directory junction, fall back to copytree (OneDrive-safe)."""
    if dst.exists():
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError):
        shutil.copytree(src, dst)
        return "copy"


def main() -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building pn -> label map from {DATASET} ...")
    pn_to_label = build_pn_to_label()
    print(f"  {len(pn_to_label)} part-numbers in dataset/")

    counts: dict[str, dict[str, int]] = {}
    unresolved: list[str] = []

    # PC: reorganize .npy files
    print(f"\nReorganizing PC tree ...")
    pc_copied = pc_skipped = 0
    for split in ("train", "val", "test"):
        for klass in ("fastener", "non_fastener"):
            src_dir = PC_SRC / split / klass
            if not src_dir.exists():
                continue
            for pc_file in src_dir.glob("*.npy"):
                pn = pc_file.stem
                label = pn_to_label.get(pn)
                if label is None:
                    unresolved.append(f"PC {split}/{klass}/{pn}")
                    continue
                dst_dir = PC_DST / split / label
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / pc_file.name
                if dst.exists():
                    pc_skipped += 1
                else:
                    shutil.copy2(pc_file, dst)
                    pc_copied += 1
                counts.setdefault(f"pc/{split}/{label}", {"n": 0})["n"] += 1
    print(f"  PC: copied {pc_copied}, skipped {pc_skipped}")

    # BREP: reorganize sample directories (each sample is its own dir of .npy files)
    print(f"\nReorganizing BREP tree ...")
    brep_added = brep_skipped = 0
    for split in ("train", "val", "test"):
        for klass in ("fastener", "non_fastener"):
            src_dir = BREP_SRC / split / klass
            if not src_dir.exists():
                continue
            for sample_dir in src_dir.iterdir():
                if not sample_dir.is_dir():
                    continue
                # Only consider samples that survived the MAX_FACES=600 filter
                if not (sample_dir / "face_grids.npy").exists():
                    continue
                pn = sample_dir.name
                label = pn_to_label.get(pn)
                if label is None:
                    unresolved.append(f"BREP {split}/{klass}/{pn}")
                    continue
                dst = BREP_DST / split / label / pn
                action = copy_dir_tree(sample_dir, dst)
                if action == "skip":
                    brep_skipped += 1
                else:
                    brep_added += 1
                counts.setdefault(f"brep/{split}/{label}", {"n": 0})["n"] += 1
    print(f"  BREP: added {brep_added}, skipped {brep_skipped}")

    # Per-split per-class summary
    print(f"\nPer-split per-class counts:")
    for prefix in ("pc", "brep"):
        for split in ("train", "val", "test"):
            row = {k.split("/")[2]: v["n"] for k, v in counts.items()
                   if k.startswith(f"{prefix}/{split}/")}
            total = sum(row.values())
            print(f"  {prefix}/{split:<5}  total={total:>5}  " +
                  "  ".join(f"{k}={v}" for k, v in sorted(row.items(), key=lambda kv: -kv[1])))

    MANIFEST.write_text(json.dumps({
        "pn_to_label_size": len(pn_to_label),
        "pc": {"copied": pc_copied, "skipped": pc_skipped},
        "brep": {"added": brep_added, "skipped": brep_skipped},
        "counts": counts,
        "unresolved_count": len(unresolved),
    }, indent=2))
    if unresolved:
        UNRESOLVED.write_text("\n".join(unresolved))
        print(f"\n  !! {len(unresolved)} unresolved part-numbers — see {UNRESOLVED.name}")
    else:
        if UNRESOLVED.exists():
            UNRESOLVED.unlink()
        print(f"\n  unresolved: 0 (all part-numbers mapped to a label)")
    print(f"\nManifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
