"""Sidecar bundle folder/zip operations.

A bundle is a folder containing:
    part.step, manifest.json, textures/, source_refs.json,
    validation_report.json, .history/, README.txt
"""
from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from ..schema import Manifest, BundleMetadata


def create_bundle(output_dir: str | Path, manifest: Manifest,
                  step_path: str | Path | None = None) -> Path:
    """Create a new sidecar bundle folder.

    Args:
        output_dir: Where to create the bundle folder
        manifest: The manifest describing the bundle contents
        step_path: Optional STEP file to copy into the bundle

    Returns:
        Path to the created bundle directory
    """
    bundle_dir = Path(output_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (bundle_dir / "textures").mkdir(exist_ok=True)
    (bundle_dir / ".history").mkdir(exist_ok=True)

    # Copy STEP file if provided
    if step_path:
        step_path = Path(step_path)
        dest = bundle_dir / "part.step"
        if step_path != dest:
            shutil.copy2(step_path, dest)

        # Save initial history snapshot
        history_dir = bundle_dir / ".history"
        version = _next_version(history_dir)
        shutil.copy2(dest, history_dir / f"{version}.step")

    # Write manifest
    from .manifest import write_manifest
    write_manifest(manifest, bundle_dir)

    # Save manifest history
    history_dir = bundle_dir / ".history"
    version = _next_version(history_dir, suffix=".manifest.json")
    from .manifest import write_manifest as _wm
    _wm(manifest, history_dir / f"{version}.manifest.json")

    # Write history log
    _append_history_log(history_dir, f"Bundle created")

    # Write source_refs.json (initially empty)
    source_refs = bundle_dir / "source_refs.json"
    if not source_refs.exists():
        source_refs.write_text(json.dumps({"refs": []}, indent=2))

    # Write README
    readme = bundle_dir / "README.txt"
    if not readme.exists():
        readme.write_text(
            f"STEP VR STEP Bundle\n"
            f"Created: {datetime.now(timezone.utc).isoformat()}\n"
            f"App version: {manifest.meta.app_version}\n"
            f"Source format: {manifest.meta.source_format}\n"
            f"Parts: {len(manifest.parts)}\n"
        )

    return bundle_dir


def open_bundle(bundle_path: str | Path) -> tuple[Path, Manifest]:
    """Open an existing bundle and return its path and manifest.

    Handles both folder bundles and zip bundles.
    """
    bundle_path = Path(bundle_path)

    if bundle_path.suffix == ".zip":
        return _open_zip_bundle(bundle_path)

    if bundle_path.is_dir():
        from .manifest import read_manifest
        manifest = read_manifest(bundle_path / "manifest.json")
        return bundle_path, manifest

    raise FileNotFoundError(f"Bundle not found: {bundle_path}")


def zip_bundle(bundle_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Compress a bundle folder into a zip file."""
    bundle_dir = Path(bundle_dir)
    output_path = Path(output_path) if output_path else bundle_dir.with_suffix(".zip")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in bundle_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(bundle_dir)
                zf.write(file_path, arcname)

    return output_path


def find_sibling_bundle(step_path: str | Path) -> Path | None:
    """Look for a sibling .bundle/ folder for a STEP file.

    Checks for: same_name.bundle/, same_name.step.bundle/, and .bundle/ in same dir.
    """
    step_path = Path(step_path)
    parent = step_path.parent
    stem = step_path.stem

    candidates = [
        parent / f"{stem}.bundle",
        parent / f"{stem}.step.bundle",
        parent / ".bundle",
    ]

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "manifest.json").exists():
            return candidate

    return None


def _open_zip_bundle(zip_path: Path) -> tuple[Path, Manifest]:
    """Extract a zip bundle to a temp directory and return it."""
    import tempfile
    extract_dir = Path(tempfile.mkdtemp(prefix="svs_bundle_"))

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    from .manifest import read_manifest
    manifest = read_manifest(extract_dir / "manifest.json")
    return extract_dir, manifest


def _next_version(history_dir: Path, suffix: str = ".step") -> str:
    """Get the next version number for history files."""
    existing = sorted(history_dir.glob(f"v*{suffix}"))
    if not existing:
        return "v0001"
    last = existing[-1].stem.split(".")[0]  # e.g., "v0001"
    num = int(last[1:]) + 1
    return f"v{num:04d}"


def _append_history_log(history_dir: Path, message: str) -> None:
    """Append a line to the history log."""
    log_file = history_dir / "history.log"
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
