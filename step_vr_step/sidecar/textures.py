"""PBR texture copy and reference management for sidecar bundles."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from uuid import UUID


def copy_texture_to_bundle(source_path: str | Path, bundle_dir: str | Path,
                           part_uuid: UUID, texture_type: str) -> str:
    """Copy a texture file into the bundle's textures/ directory.

    Args:
        source_path: Path to the source texture file
        bundle_dir: Path to the bundle directory
        part_uuid: UUID of the part this texture belongs to
        texture_type: One of: albedo, normal, roughness, metallic, ao, emissive

    Returns:
        Relative path within the bundle (e.g., "textures/a3f7-9b21-.../albedo.png")
    """
    source_path = Path(source_path)
    bundle_dir = Path(bundle_dir)

    # Use UUID prefix for organization
    uuid_prefix = str(part_uuid)[:13]  # e.g., "a3f7b9c1-9b21"
    texture_dir = bundle_dir / "textures" / uuid_prefix
    texture_dir.mkdir(parents=True, exist_ok=True)

    ext = source_path.suffix or ".png"
    dest = texture_dir / f"{texture_type}{ext}"

    if source_path.exists():
        shutil.copy2(source_path, dest)

    return str(dest.relative_to(bundle_dir))


def deduplicate_textures(bundle_dir: str | Path) -> dict[str, str]:
    """Find duplicate textures by content hash and deduplicate.

    Returns mapping of removed paths to their canonical replacement paths.
    """
    bundle_dir = Path(bundle_dir)
    textures_dir = bundle_dir / "textures"

    if not textures_dir.exists():
        return {}

    # Hash all texture files
    hash_to_path: dict[str, Path] = {}
    remap: dict[str, str] = {}

    for tex_file in sorted(textures_dir.rglob("*")):
        if not tex_file.is_file():
            continue

        file_hash = _hash_file(tex_file)

        if file_hash in hash_to_path:
            # Duplicate found — record the mapping and remove
            canonical = hash_to_path[file_hash]
            remap[str(tex_file.relative_to(bundle_dir))] = str(canonical.relative_to(bundle_dir))
            tex_file.unlink()
        else:
            hash_to_path[file_hash] = tex_file

    return remap


def validate_texture(path: str | Path) -> bool:
    """Validate a texture file by checking magic bytes.

    Prevents renamed executables from being included in bundles.
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        return False

    with open(path, "rb") as f:
        header = f.read(16)

    # Known image format magic bytes
    magic_checks = [
        (b"\x89PNG\r\n\x1a\n", "png"),
        (b"\xff\xd8\xff", "jpeg"),
        (b"GIF87a", "gif"),
        (b"GIF89a", "gif"),
        (b"RIFF", "webp"),  # RIFF header (WebP starts with RIFF)
        (b"BM", "bmp"),
        (b"\x00\x00\x00", "tga"),  # TGA (approximate)
    ]

    for magic, _fmt in magic_checks:
        if header.startswith(magic):
            return True

    # Check for TIFF (little-endian and big-endian)
    if header[:2] in (b"II", b"MM"):
        return True

    # EXR
    if header[:4] == b"\x76\x2f\x31\x01":
        return True

    return False


def _hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
