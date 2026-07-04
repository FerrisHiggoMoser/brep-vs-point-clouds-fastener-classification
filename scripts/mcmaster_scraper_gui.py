"""
McMaster-Carr STEP scraper using PyAutoGUI to drive a real Chrome window.

This bypasses McMaster's bot detection by literally being a real user:
OS-level mouse + keyboard events into your visible Chrome.

SETUP (one-time):
  1) In Chrome, Settings -> Downloads -> turn OFF "Ask where to save each file
     before downloading", and set the download folder to something like
     ~/Downloads/mcmaster (must be a folder this script knows).
  2) Take cropped screenshots of the McMaster part-page UI buttons and save them
     to backend/scripts/templates/:
        - cad_save_button.png   : the "Save" button under the 3D CAD viewer
        - step_format.png       : the "3-D STEP" item in the format dropdown
     Make tight crops — 100-200px wide, just the button itself with a few px
     border. Take them on the SAME monitor + browser zoom you'll run with.
  3) Build a URL list. One absolute McMaster part URL per line. You can produce
     this with mcmaster_scraper.py --dry-run (it'll print discovered families).
     Pass that file with --urls.

USAGE:
    python mcmaster_scraper_gui.py \\
        --urls part_urls.txt \\
        --download-dir ~/Downloads/mcmaster \\
        --out "/Volumes/Uncle Sam/step-vr-step-thesis/fastener_labeling/New Files/screws"

DURING THE RUN:
  - Don't move your mouse or type. The script needs the foreground.
  - Slam your mouse to the top-left corner to abort (PyAutoGUI failsafe).
"""
from __future__ import annotations

import argparse
import csv
import platform
import shutil
import sys
import time
from pathlib import Path

import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.4

IS_MAC = platform.system() == "Darwin"
MOD = "command" if IS_MAC else "ctrl"


def focus_address_bar() -> None:
    pyautogui.hotkey(MOD, "l")
    time.sleep(0.3)


def navigate(url: str) -> None:
    focus_address_bar()
    pyautogui.typewrite(url, interval=0.005)
    pyautogui.press("enter")


def click_template(template: Path, confidence: float = 0.85, timeout: float = 12.0) -> bool:
    """Find a template image on screen and click its center. Return True on success."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            box = pyautogui.locateOnScreen(str(template), confidence=confidence)
        except Exception:
            box = None
        if box:
            pyautogui.click(pyautogui.center(box))
            return True
        time.sleep(0.4)
    return False


def wait_for_new_file(download_dir: Path, before: set[str], timeout: float = 25.0) -> Path | None:
    """Watch the download dir for a new file (excluding .crdownload)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        now = {p.name for p in download_dir.iterdir() if not p.name.endswith(".crdownload")}
        new = now - before
        if new:
            # Take the most recently modified one
            candidates = [download_dir / n for n in new]
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
        time.sleep(0.4)
    return None


def part_number_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def append_manifest(manifest: Path, part_number: str, filename: str, url: str) -> None:
    new = not manifest.exists()
    with manifest.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["part_number", "filename", "source_url"])
        w.writerow([part_number, filename, url])


def load_already_done(out_root: Path) -> set[str]:
    done: set[str] = set()
    for csvf in out_root.rglob("_manifest.csv"):
        with csvf.open() as f:
            r = csv.DictReader(f)
            for row in r:
                done.add(row["part_number"])
    return done


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--urls", required=True, type=Path,
                   help="Text file: one part URL per line; lines starting with # are subcategory headers.")
    p.add_argument("--download-dir", required=True, type=Path,
                   help="Chrome's auto-download folder.")
    p.add_argument("--out", required=True, type=Path,
                   help="Where to move STEP files into (organized by subcategory).")
    p.add_argument("--templates", type=Path,
                   default=Path(__file__).parent / "templates",
                   help="Folder containing cad_save_button.png and step_format.png.")
    p.add_argument("--page-load-wait", type=float, default=6.0,
                   help="Seconds to wait after navigating before looking for the CAD button.")
    p.add_argument("--between-parts", type=float, default=4.0,
                   help="Idle seconds between parts (politeness).")
    args = p.parse_args()

    args.download_dir = args.download_dir.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)
    args.download_dir.mkdir(parents=True, exist_ok=True)

    cad_btn = args.templates / "cad_save_button.png"
    step_opt = args.templates / "step_format.png"
    for t in (cad_btn, step_opt):
        if not t.exists():
            print(f"missing template: {t}", file=sys.stderr)
            return 2

    done = load_already_done(args.out)
    print(f"already-downloaded: {len(done)} parts")

    print("starting in 5s — switch to Chrome and don't touch the keyboard/mouse.")
    time.sleep(5)

    current_subdir = args.out / "uncategorized"
    current_subdir.mkdir(exist_ok=True)
    consecutive_fails = 0

    for raw in args.urls.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            # Subcategory marker, e.g. "# socket-head-screws"
            sub = line.lstrip("#").strip().replace(" ", "_") or "uncategorized"
            current_subdir = args.out / sub
            current_subdir.mkdir(exist_ok=True)
            print(f"\n=== {sub} ===")
            continue

        url = line
        pn = part_number_from_url(url)
        if pn in done:
            print(f"  skip {pn} (already)")
            continue

        before = {p.name for p in args.download_dir.iterdir()}
        navigate(url)
        time.sleep(args.page_load_wait)

        if not click_template(cad_btn):
            print(f"  FAIL {pn}: CAD button not found")
            consecutive_fails += 1
            if consecutive_fails >= 5:
                print("5 fails in a row — aborting (page layout changed or throttled).")
                return 1
            continue
        time.sleep(0.6)

        if not click_template(step_opt):
            print(f"  FAIL {pn}: STEP option not found")
            consecutive_fails += 1
            continue
        # Some part pages need a final "Save" press; the dropdown click is enough
        # if the format dropdown is wired to download-on-select. If not, add a
        # third template "save_button.png" and click it here.

        new_file = wait_for_new_file(args.download_dir, before, timeout=30.0)
        if not new_file:
            print(f"  FAIL {pn}: download didn't appear")
            consecutive_fails += 1
            continue

        target = current_subdir / f"{pn}{new_file.suffix}"
        shutil.move(str(new_file), str(target))
        append_manifest(current_subdir / "_manifest.csv", pn, target.name, url)
        done.add(pn)
        consecutive_fails = 0
        print(f"  OK {pn} -> {target.relative_to(args.out)}")
        time.sleep(args.between_parts)

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
