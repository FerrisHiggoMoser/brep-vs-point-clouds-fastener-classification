"""LOD level definitions for fastener proxy meshes."""

from enum import IntEnum


class LODLevel(IntEnum):
    """Level of detail for fastener proxy meshes."""
    L0 = 0  # Original / close-up: 500–2000 polys, used < 2m in VR
    L1 = 1  # Medium distance: 50–100 polys, used 2–10m
    L2 = 2  # Far / hidden: 6–20 polys (simple primitive), used > 10m


# Screen-size thresholds for Unreal Engine LOD switching
# (fraction of screen height the bounding sphere occupies)
LOD_SCREEN_SIZES = {
    LODLevel.L0: 1.0,    # default / closest
    LODLevel.L1: 0.25,   # medium distance
    LODLevel.L2: 0.05,   # far away
}

# Target polygon counts per level
LOD_POLY_TARGETS = {
    LODLevel.L0: 1000,   # keep original or cap at 1000
    LODLevel.L1: 64,     # simplified mesh
    LODLevel.L2: 12,     # simple cylinder / box primitive
}
