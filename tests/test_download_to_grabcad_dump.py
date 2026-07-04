from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_to_grabcad_dump.py"
SPEC = importlib.util.spec_from_file_location("download_to_grabcad_dump", SCRIPT_PATH)
downloader = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = downloader
SPEC.loader.exec_module(downloader)


def test_restricted_url_detection():
    assert downloader.is_restricted_url("https://grabcad.com/library/foo-1")
    assert downloader.is_restricted_url("https://cad.grabcad.com/library/foo-1")
    assert not downloader.is_restricted_url("https://example.edu/open/foo.zip")


def test_sanitize_filename_keeps_cad_names_windows_safe():
    assert downloader.sanitize_filename("ISO 4762 M6x20.step", "x.bin") == "ISO 4762 M6x20.step"
    assert downloader.sanitize_filename("../bad/name?.zip", "x.bin") == "name_.zip"
    assert downloader.sanitize_filename("CON.step", "x.bin") == "download_CON.step"


def test_looks_like_html():
    assert downloader.looks_like_html("text/html; charset=utf-8", b"anything")
    assert downloader.looks_like_html("application/octet-stream", b"   <!doctype html><html>")
    assert not downloader.looks_like_html("application/zip", b"PK\x03\x04")


def test_read_url_manifest_text(tmp_path):
    manifest = tmp_path / "urls.txt"
    manifest.write_text(
        "# comment\n"
        "https://example.edu/a.zip\n"
        "https://example.edu/b.step, note ignored\n",
        encoding="utf-8",
    )
    assert downloader.read_url_manifest(manifest) == [
        "https://example.edu/a.zip",
        "https://example.edu/b.step",
    ]


def test_read_url_manifest_csv(tmp_path):
    manifest = tmp_path / "urls.csv"
    manifest.write_text(
        "name,url\n"
        "a,https://example.edu/a.zip\n"
        "b,https://example.edu/b.step\n",
        encoding="utf-8",
    )
    assert downloader.read_url_manifest(manifest) == [
        "https://example.edu/a.zip",
        "https://example.edu/b.step",
    ]
