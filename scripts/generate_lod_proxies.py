"""Generate LOD proxy meshes from high-res STEP fastener files.

Usage:
    python scripts/generate_lod_proxies.py \
        --input_dir data/fastener_library/ \
        --output_dir data/lod_proxies/
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def generate_proxies(step_path: Path, output_dir: Path):
    """Generate L0/L1/L2 proxy meshes from a STEP file."""
    from step_vr_step.lod.lod_levels import LODLevel
    from step_vr_step.lod.proxy_library import get_proxy
    from step_vr_step.detection.geometric_features import extract_brep_features

    from OCC.Core.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != 1:
        raise ValueError(f"Failed to read {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    features = extract_brep_features(shape)

    for level in LODLevel:
        verts, faces = get_proxy(
            fastener_type="hex_bolt",  # infer from filename or features
            diameter_mm=features.bounding_cylinder_diameter,
            length_mm=features.bounding_cylinder_length,
            lod_level=level,
        )

        out_path = output_dir / f"{step_path.stem}_{level.name}.npz"
        np.savez(out_path, vertices=verts, faces=faces)
        logger.info(f"Generated {level.name}: {len(faces)} faces -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate LOD proxy meshes")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step_extensions = {".step", ".stp", ".STEP", ".STP"}

    for step_file in sorted(input_dir.rglob("*")):
        if step_file.suffix not in step_extensions:
            continue
        try:
            generate_proxies(step_file, output_dir)
        except Exception as e:
            logger.error(f"Failed to process {step_file}: {e}")


if __name__ == "__main__":
    main()
