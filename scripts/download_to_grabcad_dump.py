"""Download direct CAD/archive URLs into the GrabCAD dump folder.

This is the downloader half of the dump-folder sorter workflow. It does not
crawl CAD library pages or discover files by itself. Give it a text/CSV manifest
of direct, permitted download URLs and it writes the resulting files to:

    D:\\step-vr-step-thesis\\grabcad_dump

The sorter can then process that folder with:

    python backend\\scripts\\process_dump_folder.py

Examples:
    python backend/scripts/download_to_grabcad_dump.py \
        --url-manifest training_data/grabcad_run/direct_urls.txt

    python backend/scripts/download_to_grabcad_dump.py \
        --url https://example.edu/open-cad/gearbox.zip --limit 1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_DUMP = Path(r"D:\step-vr-step-thesis\grabcad_dump")
DEFAULT_RUN_DIR = Path(r"D:\step-vr-step-thesis\reproducible-build\training_data\grabcad_run")
RESTRICTED_DOMAINS = {"grabcad.com"}


@dataclass
class DownloadStats:
    ok: int = 0
    skipped_existing: int = 0
    skipped_restricted: int = 0
    skipped_html: int = 0
    failed: int = 0


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha1_text(text: str, chars: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:chars]


def is_restricted_url(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in RESTRICTED_DOMAINS)


def sanitize_filename(name: str, fallback: str) -> str:
    name = urllib.parse.unquote(name or "").strip().strip("\"'")
    if not name:
        name = fallback
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" ._")
    if not name:
        name = fallback
    reserved = {
        "con", "prn", "aux", "nul",
        "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
        "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
    }
    stem = Path(name).stem.lower()
    if stem in reserved:
        name = f"download_{name}"
    if len(name) > 180:
        suffix = Path(name).suffix[:20]
        name = f"{Path(name).stem[:150]}__{sha1_text(name)}{suffix}"
    return name


def filename_from_response(url: str, response: urllib.response.addinfourl) -> str:
    disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r"filename\*?=(?:UTF-8''|utf-8'')?\"?([^\";]+)\"?", disposition)
    if match:
        return sanitize_filename(match.group(1), f"download_{sha1_text(url)}.bin")

    path_name = Path(urllib.parse.urlparse(response.geturl()).path).name
    if path_name:
        return sanitize_filename(path_name, f"download_{sha1_text(url)}.bin")

    content_type = response.headers.get_content_type()
    suffix = mimetypes.guess_extension(content_type) or ".bin"
    return f"download_{sha1_text(url)}{suffix}"


def looks_like_html(content_type: str, first_chunk: bytes) -> bool:
    if "text/html" in content_type.lower():
        return True
    probe = first_chunk[:512].lstrip().lower()
    return probe.startswith(b"<!doctype html") or probe.startswith(b"<html")


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"urls": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("urls"), dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"urls": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_url_manifest(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                url_field = next((field for field in reader.fieldnames if field.lower() == "url"), None)
                if url_field:
                    return [
                        (row.get(url_field) or "").strip()
                        for row in reader
                        if (row.get(url_field) or "").strip()
                    ]

    urls: list[str] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line.split(",", 1)[0].strip())
    return urls


def collect_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = []
    for manifest in args.url_manifest:
        urls.extend(read_url_manifest(manifest))
    urls.extend(args.url)

    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    if args.limit:
        deduped = deduped[: args.limit]
    return deduped


def make_destination(dump: Path, preferred_name: str, url: str, force: bool) -> Path:
    base = sanitize_filename(preferred_name, f"download_{sha1_text(url)}.bin")
    dst = dump / base
    if force or not dst.exists():
        return dst
    suffix = Path(base).suffix
    stem = Path(base).stem
    return dump / f"{stem}__{sha1_text(url)}{suffix}"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(url: str, args: argparse.Namespace, state: dict, rows: list[dict], stats: DownloadStats) -> None:
    old = state["urls"].get(url)
    if old and old.get("status") == "ok" and Path(old.get("path", "")).exists() and not args.force:
        stats.skipped_existing += 1
        rows.append({"url": url, "status": "exists", "path": old["path"], "bytes": old.get("bytes", "")})
        print(f"[exists] {old['path']}")
        return

    if is_restricted_url(url) and not args.i_have_permission_for_restricted_sites:
        stats.skipped_restricted += 1
        note = "restricted domain; pass permission flag only if you have permission"
        state["urls"][url] = {"status": "skipped_restricted", "updated": now_iso(), "note": note}
        rows.append({"url": url, "status": "skipped_restricted", "path": "", "bytes": "", "note": note})
        print(f"[skip] restricted domain: {url}")
        return

    max_bytes = args.max_mb * 1024 * 1024
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "step-vr-step-dump-downloader/1.0",
            "Accept": "*/*",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError(f"remote file exceeds --max-mb={args.max_mb}")

            first_chunk = response.read(64 * 1024)
            content_type = response.headers.get("Content-Type", "")
            if looks_like_html(content_type, first_chunk) and not args.save_html:
                stats.skipped_html += 1
                note = f"HTML response ({content_type or 'unknown content type'}); likely not a direct file URL"
                state["urls"][url] = {"status": "skipped_html", "updated": now_iso(), "note": note}
                rows.append({"url": url, "status": "skipped_html", "path": "", "bytes": "", "note": note})
                print(f"[skip] HTML response, not a direct file: {url}")
                return

            args.dump.mkdir(parents=True, exist_ok=True)
            name = filename_from_response(url, response)
            dst = make_destination(args.dump, name, url, args.force)
            tmp = dst.with_suffix(dst.suffix + ".part")
            written = 0
            with tmp.open("wb") as f:
                if first_chunk:
                    written += len(first_chunk)
                    if written > max_bytes:
                        raise ValueError(f"download exceeds --max-mb={args.max_mb}")
                    f.write(first_chunk)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(f"download exceeds --max-mb={args.max_mb}")
                    f.write(chunk)
            tmp.replace(dst)
            digest = file_sha256(dst)
            stats.ok += 1
            state["urls"][url] = {
                "status": "ok",
                "updated": now_iso(),
                "path": str(dst),
                "bytes": written,
                "sha256": digest,
            }
            rows.append({"url": url, "status": "ok", "path": str(dst), "bytes": written, "sha256": digest})
            print(f"[ok] {written / 1024 / 1024:.1f} MB -> {dst}")
    except (OSError, urllib.error.URLError, ValueError) as exc:
        stats.failed += 1
        state["urls"][url] = {"status": "failed", "updated": now_iso(), "error": f"{type(exc).__name__}: {exc}"}
        rows.append({"url": url, "status": "failed", "path": "", "bytes": "", "note": f"{type(exc).__name__}: {exc}"})
        print(f"[fail] {url} ({exc})")


def write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["url", "status", "path", "bytes", "sha256", "note"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-manifest", action="append", default=[], type=Path, help="Text/CSV manifest. CSV may contain a 'url' column.")
    parser.add_argument("--url", action="append", default=[], help="Single direct URL. May be repeated.")
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--state", type=Path, help="Resume-state JSON. Defaults to <run-dir>/download_state.json.")
    parser.add_argument("--report", type=Path, help="Per-run CSV report. Defaults to <run-dir>/download_report.csv.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-mb", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between downloads in seconds.")
    parser.add_argument("--force", action="store_true", help="Re-download URLs even if state says they are already present.")
    parser.add_argument("--save-html", action="store_true", help="Save HTML responses too. Off by default because they are usually login/search pages.")
    parser.add_argument("--fail-on-errors", action="store_true")
    parser.add_argument(
        "--i-have-permission-for-restricted-sites",
        action="store_true",
        help="Allow direct URLs from restricted domains. Use only with permission from that site.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.url_manifest and not args.url:
        raise SystemExit("Provide --url-manifest, --url, or both.")

    args.dump = args.dump.resolve()
    args.run_dir = args.run_dir.resolve()
    args.state = (args.state or (args.run_dir / "download_state.json")).resolve()
    args.report = (args.report or (args.run_dir / "download_report.csv")).resolve()

    urls = collect_urls(args)
    if not urls:
        raise SystemExit("No URLs found.")

    state = load_state(args.state)
    rows: list[dict] = []
    stats = DownloadStats()
    print(f"[input] {len(urls)} URLs")
    print(f"[dump]  {args.dump}")

    for idx, url in enumerate(urls, start=1):
        print(f"[{idx}/{len(urls)}] {url}")
        download_one(url, args, state, rows, stats)
        save_state(args.state, state)
        if args.sleep > 0 and idx < len(urls):
            time.sleep(args.sleep)

    write_report(args.report, rows)
    print(f"[state]  {args.state}")
    print(f"[report] {args.report}")
    print(
        "[summary] "
        f"ok={stats.ok} "
        f"existing={stats.skipped_existing} "
        f"restricted={stats.skipped_restricted} "
        f"html={stats.skipped_html} "
        f"failed={stats.failed}"
    )
    return 1 if args.fail_on_errors and stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
