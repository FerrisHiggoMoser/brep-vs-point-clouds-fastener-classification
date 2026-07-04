"""Read/write manifest.json for sidecar bundles."""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import Manifest


def read_manifest(path: str | Path) -> Manifest:
    """Read a manifest.json file and return a Manifest object."""
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"

    with open(path, "r") as f:
        data = json.load(f)

    return Manifest.model_validate(data)


def write_manifest(manifest: Manifest, path: str | Path) -> Path:
    """Write a Manifest object to manifest.json.

    Args:
        manifest: The manifest to write
        path: File path or directory (manifest.json will be created inside)

    Returns:
        Path to the written file
    """
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"

    path.parent.mkdir(parents=True, exist_ok=True)

    data = manifest.model_dump(mode="json")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return path
