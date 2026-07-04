"""
McMaster-Carr STEP file scraper.

Hierarchy: macro category (/products/screws/) -> shape family
(/products/screws/socket-head-screws-2~/) -> material leaf
(/products/screws/socket-head-screws-2~/steel-socket-head-screws~~/) -> parts.

For each shape family we pick ONE material leaf (steel preferred), since material
variants of the same shape are geometrically identical and would just inflate the
dataset with duplicates.

Usage:
    python mcmaster_scraper.py --out ./mcmaster_step
    python mcmaster_scraper.py --category screws --dry-run   # plan only, no downloads

Politeness: visible Chromium, sequential, 2-6s jitter between parts, abort on 3
consecutive download failures (likely throttle). Progress saved per-download.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import (
    Page,
    TimeoutError as PWTimeout,
    sync_playwright,
)

CATEGORY_URLS: dict[str, str] = {
    "screws": "https://www.mcmaster.com/products/screws/",
    "threaded-rods": "https://www.mcmaster.com/products/threaded-rods/",
    "nuts": "https://www.mcmaster.com/products/nuts/",
    "washers": "https://www.mcmaster.com/products/standard-washers/",
    "pins": "https://www.mcmaster.com/products/pins/",
    "anchors": "https://www.mcmaster.com/products/anchors/",
    "threaded-inserts": "https://www.mcmaster.com/products/threaded-inserts/",
    "spacers": "https://www.mcmaster.com/products/spacers/",
    "nails": "https://www.mcmaster.com/products/nails/",
    "rivets": "https://www.mcmaster.com/products/rivets/",
    "retaining-rings": "https://www.mcmaster.com/products/retaining-rings/",
    "keys": "https://www.mcmaster.com/products/machine-keys/",
    # Non-fastener structural categories for ML negatives (satellite brackets,
    # mounting plates, extruded framing — geometrically distinct from fasteners
    # but commonly co-located in satellite assemblies).
    "brackets": "https://www.mcmaster.com/products/brackets/",
    "mounting-plates": "https://www.mcmaster.com/products/mounting-plates/",
    "t-slotted-framing": "https://www.mcmaster.com/products/t-slotted-framing/",
    # PCBs + electronics — McMaster's circuit-boards macro page is the broadest;
    # subcategories include bare PCBs, dev boards, SBCs, enclosures, racks,
    # connectors, terminals, displays, relays, etc. — all great ML negatives.
    "pcbs": "https://www.mcmaster.com/products/circuit-boards/",
    # Tier 1 — structural / thermal (highest ML negative value)
    "hinges": "https://www.mcmaster.com/products/hinges/",
    "heat-sinks": "https://www.mcmaster.com/products/heat-sinks/",
    "springs": "https://www.mcmaster.com/products/springs/",
    "bearings": "https://www.mcmaster.com/products/bearings/",
    "gears": "https://www.mcmaster.com/products/gears/",
    "electrical-enclosures": "https://www.mcmaster.com/products/electrical-enclosures/",
    "circuit-board-hardware": "https://www.mcmaster.com/products/circuit-board-hardware/",
    # Tier 2 — mechanism / motion
    "pulleys": "https://www.mcmaster.com/products/pulleys/",
    "sprockets": "https://www.mcmaster.com/products/sprockets/",
    "shaft-couplings": "https://www.mcmaster.com/products/shaft-couplings/",
    "cams": "https://www.mcmaster.com/products/cams/",
    "latches": "https://www.mcmaster.com/products/latches/",
    "linear-slides": "https://www.mcmaster.com/products/linear-slides/",
    "linear-bearings": "https://www.mcmaster.com/products/linear-bearings/",
    # Tier 3 — electrical / cable
    "terminal-blocks": "https://www.mcmaster.com/products/terminal-blocks/",
    "connectors": "https://www.mcmaster.com/products/connectors/",
    "wire-connectors": "https://www.mcmaster.com/products/wire-connectors/",
    "cable-clips": "https://www.mcmaster.com/products/cable-clips/",
    "grommets": "https://www.mcmaster.com/products/grommets/",
    # Tier 4 — fluidic
    "tube-fittings": "https://www.mcmaster.com/products/tube-fittings/",
    "pipe-fittings": "https://www.mcmaster.com/products/pipe-fittings/",
    "manifolds": "https://www.mcmaster.com/products/manifolds/",
    "quick-disconnects": "https://www.mcmaster.com/products/quick-disconnects/",
    "valves": "https://www.mcmaster.com/products/valves/",
    # Tier 5 — small hardware
    "knobs": "https://www.mcmaster.com/products/knobs/",
    "handles": "https://www.mcmaster.com/products/handles/",
    "bumpers": "https://www.mcmaster.com/products/bumpers/",
    "plugs": "https://www.mcmaster.com/products/plugs/",
    "end-caps": "https://www.mcmaster.com/products/end-caps/",
    "feet": "https://www.mcmaster.com/products/feet/",
}

CATEGORY_QUOTAS: dict[str, int] = {
    "screws": 1500,        # bumped — newly added bolt + flat-head + drywall families
    "nuts": 1100,          # bumped — added jam, lug, push, slotted-round, t-slot etc.
    "washers": 400,
    "pins": 450,           # bumped — added hitch, weld, alignment, captive
    "threaded-rods": 130,
    "spacers": 60,
    "threaded-inserts": 120,
    "anchors": 100,
    "nails": 50,
    "rivets": 400,         # major aerospace/satellite structural fastener — was missing
    "retaining-rings": 200, # snap rings / circlips — common in mechanisms
    "keys": 80,            # shaft keys / keystock / woodruff
    "brackets": 250,       # negative class: angle/L/T/corner brackets, gussets
    "mounting-plates": 150, # negative class: base plates, panels
    "t-slotted-framing": 200, # negative class: extrusions + framing accessories
    "pcbs": 300,           # negative class: bare PCBs, dev boards, SBCs, enclosures
    # Tier 1
    "hinges": 200, "heat-sinks": 80, "springs": 200, "bearings": 200,
    "gears": 150, "electrical-enclosures": 150, "circuit-board-hardware": 150,
    # Tier 2
    "pulleys": 100, "sprockets": 120, "shaft-couplings": 100, "cams": 80,
    "latches": 200, "linear-slides": 100, "linear-bearings": 100,
    # Tier 3
    "terminal-blocks": 100, "connectors": 250, "wire-connectors": 150,
    "cable-clips": 100, "grommets": 100,
    # Tier 4
    "tube-fittings": 150, "pipe-fittings": 200, "manifolds": 100,
    "quick-disconnects": 100, "valves": 200,
    # Tier 5
    "knobs": 150, "handles": 200, "bumpers": 150, "plugs": 250,
    "end-caps": 100, "feet": 100,
}

# Material priority — when a shape family offers multiple material variants,
# we pick the highest-priority one. Same geometry = same shape to the model.
MATERIAL_PRIORITY = [
    "steel", "stainless-steel", "aluminum", "brass", "bronze",
    "titanium", "nickel", "plastic", "nylon",
]

# Shape-family allow-list per category. The scraper auto-discovers subcategory
# links on each macro page; only links whose URL slug contains one of these
# substrings are kept. Curated to maximise GEOMETRIC diversity, no near-duplicates.
# Within a category, families matching one of these substrings are downloaded
# FIRST (in this order). Anything else goes last, alphabetical. Lets us hit the
# most common / most important shape families before quotas run out.
SHAPE_PRIORITY: dict[str, list[str]] = {
    "screws": [
        "socket-head", "hex-head", "shoulder", "set-screws",
        "thumb-screws", "wood", "captive-panel", "jack",
    ],
    "nuts": ["hex-nuts", "locknuts", "flange-nuts", "cap-nuts"],
    "washers": ["general-purpose", "lock-washers", "flat-washers"],
    "pins": ["dowel", "clevis", "spring", "cotter"],
    "rivets": ["rivets-2", "binding-barrels", "conveyor-wear-strip", "screw-nails"],
    "retaining-rings": ["external-retaining-rings", "internal-retaining-rings"],
    "keys": ["machine-keys", "machine-key-stock", "fixture-keys"],
    "brackets": ["brackets-1", "shelf-brackets", "angle-plates"],
    "mounting-plates": ["mounting-plates", "panel-hanging-brackets", "u-bolt-plates"],
    "t-slotted-framing": ["t-slotted-framing-and-fittings", "framing-and-fittings"],
    "pcbs": ["circuit-boards", "development-boards", "single-board-computers"],
}


SHAPE_ALLOWLIST: dict[str, list[str]] = {
    "screws": [
        # ---- socket head family ----
        "socket-head-screws",
        "low-profile-socket-head",
        "ultra-low-profile-socket-head",
        "flanged-socket-head",
        "torx-plus-socket-head",
        "venting-socket-head",
        # ---- external-drive heads ----
        "hex-head-screws",
        "12-point-screws",
        "square-head-screws",
        "pentagon-head-screws",
        "shoulder-screws",
        # ---- cosmetic / panel head shapes ----
        "flat-head-screws",            # countersunk profile — MAJOR class we were missing
        "oval-head-screws",
        "rounded-head-screws",
        # ---- thumb / wing / captive ----
        "thumb-screws",
        "captive-panel-screws",
        # ---- drives without big heads ----
        "jack-screws",
        "set-screws",
        "thread-cutting-screws",
        "sheet-metal-screws",
        "wood-screws",
        "machine-screws",
        "drywall-screws",
        "masonry-and-concrete-screws",
        "dowel-screws",
        # ---- bolts (the user said "every type of bolt") ----
        "carriage-bolts",
        "eyebolts",
        "hanger-bolts",
        "plow-bolts",
        "elevator-bolts",
        "hold-down-bolts",
        "t-slot-bolts",
        # ---- aerospace / panel-mount additions ----
        "u-bolts",
        "j-bolts",
        "self-clinching-screws",
        "press-fit-screws",
        "tap-bolts",
        "flange-bolts",
        "lag-screws",
        "lag-bolts",
    ],
    "nuts": [
        "hex-nuts",
        "jam-nuts",
        "locknuts",
        "flange-nuts",
        "cap-nuts", "acorn-nuts",
        "thumb-nuts", "wing-nuts",
        "square-nuts",
        "coupling-nuts",
        "panel-nuts",
        "socket-nuts",
        "tamper-resistant-nuts",
        "sealing-nuts",
        "weld-nuts",
        "press-fit-nuts",
        "rivet-nuts",
        "clip-on-nuts",
        "lug-nuts",
        "handle-nuts",
        "push-nuts",
        "slotted-round-nuts",
        "split-nuts",
        "t-slot-nuts",
        "screw-mount-nuts",
        "rod-end-nuts",
    ],
    "washers": [
        "general-purpose-washers", "flat-washers",
        "lock-washers", "split-lock", "tooth-lock",
        "sealing-washers",
        "finishing-washers",
        "leveling-washers",
        "square-washers",
        "structural-washers",
        "dished-washers",
        "curved-washers",
        "belleville", "disc-springs",
    ],
    "pins": [
        "dowel-pins",
        "clevis-pins",
        "spring-pins",
        "cotter-pins",
        "quick-release-pins",
        "taper-pins",
        "linch-pins",
        "shear-pins",
        "hitch-pins",
        "weld-pins",
        "captive-pins",
        "alignment-pins",
        "spring-locating-pins",
    ],
    "anchors": [
        "wedge-anchors",
        "sleeve-anchors",
        "hook-anchors",
        "drive-rivet-anchors",
        "toggle",
        "drilling-anchors",
        "lag-anchors",
    ],
    "threaded-inserts": [
        "threaded-inserts",
        "rivet-nut",
        "press-fit",
    ],
    "spacers": [
        "round-spacers",
        "hex-spacers",
        "standoffs",
    ],
    "threaded-rods": [
        "threaded-rods",        # generic
        "left-hand-threaded",
        "studs",
    ],
    "nails": [
        "nails",
        "finishing-nails",
        "framing-nails",
        "brad-nails",
    ],
    "rivets": [
        # McMaster nests solid/blind/drive/semi-tubular/etc. under rivets-2~/.
        # The scraper auto-discovers those leaves and dedupes by geometry.
        "rivets-2",
        "binding-barrels",            # post-and-screw rivets
        "conveyor-wear-strip-rivets", # conveyor-specific shape
        "screw-nails",                # hybrid screw/nail/rivet
    ],
    "retaining-rings": [
        # external-/internal- prefixes catch every genuine retaining-ring
        # geometry (clip-on, crimp-on, high-load, mil-spec, push-on, side-mount,
        # spiral, tight-clearance, tight-hold-bowed, speared-ends, ...) without
        # pulling in socket-retaining-rings, push-nuts, or protective-caps.
        "external-retaining-rings",
        "internal-retaining-rings",
    ],
    "keys": [
        "machine-keys",
        "machine-key-stock",
        "fixture-keys",
    ],
    # ---- non-fastener structural categories (ML negatives) ----
    # Tokens chosen from live probe of each macro page's actual sub-slugs.
    # Maximise geometric diversity for negatives while excluding entries
    # that overlap fastener categories (weld-nuts, eyebolts already in nuts/).
    "brackets": [
        "brackets-1", "shelf-brackets", "conveyor-brackets",
        "conveyor-bracket-spacers", "conveyor-track-brackets",
        "din-rail-mounting-brackets", "panel-hanging-brackets",
        "fire-extinguisher-brackets", "sawhorse-brackets",
        "raceway-mounting-brackets", "micrometer-head-mounting-brackets",
        "coordinate-measuring-machine-angle-brackets",
        "angle-plates", "u-bolt-plates", "webbing-anchor-plates",
        "shelf-supports", "pipe-supports", "lid-supports",
        "monitor-mounts", "robot-tool-mounts", "robot-controller-mounts",
        "threaded-rod-mounts", "positioning-arm-mounts",
        "panel-clips", "panel-alignment-clamps",
        "hanging-hooks", "s-hooks", "tie-down-hooks",
    ],
    "mounting-plates": [
        "brackets-1", "panel-hanging-brackets",
        "micrometer-head-mounting-brackets",
        "locking-slotted-framing-brackets",
        "electrical-enclosure-panels", "strip-door-mounting-plates",
        "u-bolt-plates", "webbing-anchor-plates",
        "robot-mounts", "robot-pedestals", "robot-controller-mounts",
        "threaded-rod-mounts", "positioning-arm-mounts", "jack-mounts",
        "screw-jack-mounts", "hydraulic-cylinder-mounts",
        "leveling-mounts", "leveling-mount-inserts",
        "caster-sockets", "tie-down-rings",
        "coordinate-measuring-machine-plates-and-fixtures",
    ],
    "t-slotted-framing": [
        "t-slotted-framing-and-fittings",
        "bolt-together-framing-and-fittings",
        "locking-slotted-framing-and-fittings",
        "fixture-tables",
        "robot-pedestal-adapter-kits", "robot-tool-mounts",
        "mounts-for-vacuum-cups-and-lifters",
        "push-to-close-latches", "teach-pendant-holders",
    ],
    # PCB / electronics negatives. Tokens picked from live probe of
    # /products/circuit-boards/ to keep geometrically distinct things
    # (bare boards, dev boards, board-mounted components, enclosures, racks)
    # and skip clutter (reference-books, vises, light-bulbs, toggle-clamps).
    "pcbs": [
        # bare PCBs and populated boards
        "circuit-boards-3", "development-boards", "single-board-computers",
        # board housings / mechanical structure
        "circuit-board-enclosures", "circuit-board-racks",
        "circuit-board-holders", "development-board-holders",
        "potting-boxes",
        # board-edge / board-mount components (distinctive small geometries)
        "circuit-board-headers", "circuit-board-connectors",
        "circuit-board-terminals", "battery-holders", "battery-contacts",
        "d-sub-connectors", "molex-connectors", "amphenol-connectors",
        "rj45-sockets", "usb-connectors", "ix-industrial-sockets",
        "audio-receptacles", "power-receptacles",
        # discrete components with characteristic shapes
        "circuit-board-capacitors", "circuit-board-voltage-regulators",
        "inductors", "transformers", "relays",
        "diodes", "rectifiers", "varistors", "resistors",
        "thermoelectric-modules", "thermal-interface-pads",
        # interface / display
        "display-screens", "dip-switches", "snap-acting-switches",
        "push-button-switches", "magnetic-switches", "proximity-sensors",
        # adjacent electronics
        "terminal-blocks", "plcs", "decade-boxes", "load-cells",
        "logic-level-converters", "rfid-readers", "safety-relays",
        "pressure-transmitters",
    ],
    # ---- Tier 1-5 negatives — stem-token allowlists ----
    "hinges": ["hinges", "hinge"],
    "heat-sinks": ["heat-sinks", "heat-sink"],
    "springs": ["springs", "spring"],
    "bearings": ["bearings", "bearing"],
    "gears": ["gears", "gear"],
    "electrical-enclosures": ["enclosures", "enclosure"],
    "circuit-board-hardware": ["circuit-board"],
    "pulleys": ["pulleys", "pulley"],
    "sprockets": ["sprockets", "sprocket"],
    "shaft-couplings": ["couplings", "coupling"],
    "cams": ["cams", "cam-followers", "cam-tracks"],
    "latches": ["latches", "latch", "draw-locks", "barrel-bolts", "cam-locks"],
    "linear-slides": ["slides", "slide"],
    "linear-bearings": ["bearings", "bushings", "linear"],
    "terminal-blocks": ["terminal-blocks", "terminal", "blocks"],
    "connectors": [
        "connectors", "amphenol", "molex", "audio", "usb", "rj45",
        "d-sub", "ix-industrial", "amp-connectors",
    ],
    "wire-connectors": ["connectors", "splices", "lugs", "ferrules", "terminals"],
    "cable-clips": ["clips", "clamps"],
    "grommets": ["grommets", "grommet"],
    "tube-fittings": ["fittings", "fitting"],
    "pipe-fittings": ["fittings", "pipe"],
    "manifolds": ["manifolds", "manifold"],
    "quick-disconnects": ["disconnects", "couplings", "quick"],
    "valves": ["valves", "valve"],
    "knobs": ["knobs", "knob"],
    "handles": ["handles", "handle"],
    "bumpers": ["bumpers", "bumper"],
    "plugs": ["plugs", "plug"],
    "end-caps": ["caps", "cap"],
    "feet": ["feet", "foot"],
}

# Selectors — verify against a live part page before first run.
CAD_BUTTON_SELECTOR = "button:has-text('Save CAD')"
STEP_FORMAT_SELECTOR = "text=/3-?D\\s*STEP/i"

PART_NUMBER_RE = re.compile(r"/(\d+[A-Z]\d+[A-Z]?\d*)/?$")


@dataclass
class Progress:
    downloaded: dict[str, list[str]] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)

    def count(self, category: str) -> int:
        return len(self.downloaded.get(category, []))

    def add(self, category: str, part_number: str) -> None:
        self.downloaded.setdefault(category, []).append(part_number)

    def has(self, category: str, part_number: str) -> bool:
        return part_number in self.downloaded.get(category, [])


def load_progress(path: Path) -> Progress:
    if not path.exists():
        return Progress()
    data = json.loads(path.read_text())
    return Progress(downloaded=data.get("downloaded", {}), failed=data.get("failed", []))


def save_progress(path: Path, progress: Progress) -> None:
    path.write_text(json.dumps(
        {"downloaded": progress.downloaded, "failed": progress.failed},
        indent=2,
    ))


_MATERIAL_TOKENS = {
    "steel", "stainless", "stainless-steel", "aluminum", "brass", "bronze",
    "titanium", "nickel", "plastic", "nylon", "carbon", "silver", "iron",
    "copper", "monel", "inconel",
    # McMaster stainless grade prefixes (e.g. "18-8-stainless-steel-...",
    # "316-stainless-steel-..."). Each grade splits into its own dash tokens.
    "18", "8", "316", "304", "410", "440",
    # marketing prefixes that appear on rivet/blind-rivet slugs and don't
    # affect geometry — keep these collapsed into one canonical leaf.
    "highly", "corrosion", "resistant", "high", "strength",
    "wide", "thickness", "range",
}

# Trailing McMaster hierarchy markers like "-1", "-2" — not real variants.
_TRAILING_NUM_RE = re.compile(r"-\d+$")

# Slug patterns that indicate a multi-piece assembly OR bundle product
# (rod + nut, screw + washer, insert + tool kit, strips/coils of nails).
# The downloaded STEP file contains multiple pieces, which contaminates
# per-fastener training. Keep -with-toggle / -with-groove / -with-notch /
# -with-retainer / -with-rounded-head — those are integral features of
# one part, not separate items.
_ASSEMBLY_RE = re.compile(
    r"-with-([a-z-]+-)?washer|"
    r"-with-(hex-)?nuts?(-|$|/)|"
    r"-with-installation-tools?|"
    r"-with-cotter-pin|"
    r"-with-(d-ring|key)(-|$|/)|"
    r"for-use-with-cotter-pin|"
    # bundle-of-many-parts patterns (nail strips, ammo-style nail coils, etc.)
    r"strips?-of-|"
    r"coils?-of-|"
    r"packs?-of-|"
    r"rolls?-of-"
)


def slug_priority(slug: str) -> int:
    """Lower = higher priority. Used to pick one material per shape family."""
    s = slug.lower()
    for i, mat in enumerate(MATERIAL_PRIORITY):
        if mat in s:
            return i
    return len(MATERIAL_PRIORITY)


def geometry_key(leaf_url: str) -> str:
    """Strip material words from a leaf slug to group geometrically identical leaves.

    "/.../steel-socket-head-screws~~/"           -> "socket-head-screws"
    "/.../stainless-steel-socket-head-screws~~/" -> "socket-head-screws"
    "/.../low-profile-steel-socket-head-screws~~/" -> "low-profile-socket-head-screws"
    """
    slug = leaf_url.rstrip("/").split("/")[-1].rstrip("~")
    slug = _TRAILING_NUM_RE.sub("", slug)
    tokens = [t for t in slug.split("-") if t and t not in _MATERIAL_TOKENS]
    return "-".join(tokens)


def harvest_links(page: Page, url: str) -> list[str]:
    """Visit a McMaster page, return all in-site product links found in the DOM.

    Uses domcontentloaded + a manual delay rather than networkidle: when
    logged in McMaster keeps the network busy with heartbeats, so networkidle
    never resolves and the scraper would hang here.
    """
    page.goto(url, wait_until="domcontentloaded", timeout=20000)
    page.wait_for_timeout(3500)
    return page.eval_on_selector_all(
        "a[href]",
        "els => Array.from(new Set(els.map(e => e.href).filter(h => h.includes('/products/') || /\\/\\d+[A-Z]\\d+/.test(h))))",
    )


def discover_shape_families(page: Page, macro_url: str, category_key: str) -> list[str]:
    """From a macro category page, find sub-pages whose SLUG matches the allow-list.

    We must match against the slug (last URL segment), not the whole URL —
    otherwise a category like /products/threaded-rods/standoffs-2~/ matches the
    "threaded-rods" token via the parent path and slips past the filter.
    """
    allow = SHAPE_ALLOWLIST.get(category_key, [])
    links = harvest_links(page, macro_url)
    shape_family_links = []
    for l in links:
        if not l.endswith("~/"):
            continue
        slug = l.rstrip("/").split("/")[-1].rstrip("~").lower()
        if any(token in slug for token in allow):
            shape_family_links.append(l)
    return sorted(set(shape_family_links))


def discover_material_leaves(page: Page, shape_url: str) -> list[str]:
    """From a shape-family page, find material-variant leaf pages.

    Most categories use `~~/` for material leaves. A few (rivets, some nails)
    use a 2-tier structure where leaves are tilde-less paths one level below
    the shape URL. Try the standard pattern first; fall back to tilde-less.
    """
    links = harvest_links(page, shape_url)
    standard = sorted({l for l in links if l.endswith("~~/") and l.startswith(shape_url)})
    if standard:
        return standard
    base_depth = shape_url.rstrip("/").count("/")
    fallback: set[str] = set()
    for l in links:
        if not l.startswith(shape_url) or not l.endswith("/") or l == shape_url:
            continue
        if l.rstrip("/").count("/") != base_depth + 1:
            continue
        slug = l.rstrip("/").split("/")[-1]
        # Reject filter-state URLs (slug contains a tilde mid-string, e.g.
        # `color~black-1`, `diameter~0-088`, `dfars-specialty-metals~exempt`).
        # Real geometry-leaf slugs never contain `~`.
        if "~" in slug or PART_NUMBER_RE.search("/" + slug):
            continue
        # Skip color-pick / similar non-geometry leaves.
        if slug.startswith("choose-a-color"):
            continue
        fallback.add(l)
    return sorted(fallback)


def pick_canonical_material(leaves: list[str]) -> str | None:
    """DEPRECATED — superseded by dedup_by_geometry. Kept for compatibility."""
    if not leaves:
        return None
    return sorted(leaves, key=slug_priority)[0]


def dedup_by_geometry(leaves: list[str]) -> list[str]:
    """Group leaves by geometry-key (slug minus material tokens), keep one per group.

    The kept one is the steel/standard variant, since material is irrelevant to BRep
    geometry. Returns a list of one canonical leaf per geometric subtype.
    """
    groups: dict[str, list[str]] = {}
    for leaf in leaves:
        groups.setdefault(geometry_key(leaf), []).append(leaf)
    return [sorted(group, key=slug_priority)[0] for group in groups.values()]


def collect_part_links(page: Page, leaf_url: str, target: int) -> list[str]:
    """Visit a material-leaf page and harvest part-detail URLs (with scrolling)."""
    page.goto(leaf_url, wait_until="domcontentloaded", timeout=20000)
    page.wait_for_timeout(2500)

    seen: set[str] = set()
    stagnant = 0
    while len(seen) < target and stagnant < 4:
        links = page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => e.href).filter(h => /\\/\\d+[A-Z]\\d+\\/?$/.test(h))",
        )
        before = len(seen)
        seen.update(links)
        stagnant = stagnant + 1 if len(seen) == before else 0
        page.mouse.wheel(0, 5000)
        page.wait_for_timeout(1500)

    return sorted(seen)[:target]


def download_step_inline(page: Page, part_url: str, out_dir: Path,
                         chrome_dl_dir: Path | None = None) -> str | None:
    """Click the part-number link in the table to expand the inline order panel,
    open the React format dropdown, pick "3-D STEP", click the row's Download
    button, capture the file. Selectors based on inspecting the live DOM:

      - dropdown trigger:  button[class*='buttonDropdown']  (text default: "3-D Solidworks")
      - dropdown options:  li[class*='liEnabled']           (rendered in a portal at body)
      - download button:   button[class*='downloadButton']

    Assumes `page` is already on the leaf listing page that contains this part_url.
    Returns the saved filename or None on failure.

    For some categories (brackets, plates) McMaster's UI fires a direct browser
    download instead of one Playwright's expect_download can intercept. We fall
    back to scanning chrome_dl_dir (the Chrome-managed default download dir
    redirected to Uncle Sam) for a file matching this part number, then move
    it into out_dir under the canonical "<pn>.step" name.
    """
    pn = part_number(part_url)
    # Most categories use trailing-slash hrefs (`/products/.../12345A678/`).
    # Rivets and a few others use bare-pn hrefs (`/97447A105`). Match either.
    href_slash = part_url.replace("https://www.mcmaster.com", "").rstrip("/") + "/"
    href_bare = href_slash.rstrip("/")
    a_sel = f"a[href='{href_slash}'], a[href='{href_bare}']"
    try:
        # 1. Click the part-number link to expand the row's CAD panel.
        link = page.locator(a_sel).first
        link.scroll_into_view_if_needed(timeout=4000)
        link.click()
        page.wait_for_timeout(800)

        # 2. Find the inline-order panel for THIS part. McMaster renders the
        #    expansion as a SEPARATE colspan'd <tr> below the part-link row,
        #    not inside it — so row-scoped locators miss it. Each panel's
        #    container has a class fragment "...For{partnumber}" we can match.
        #    Fall back to page-level (works because close_inline_panel ensures
        #    only one panel is open at a time).
        panel = page.locator(f"tr[class*='For{pn}']")
        if panel.count() == 0:
            panel = page.locator("tr[class*='InLnOrdWebPart_CntntRow']").first

        # 3. Open the format dropdown. Assortments / kits have no CAD dropdown
        #    at all — return SKIP rather than treating that as a fail.
        dropdown_btn = panel.locator("button[class*='buttonDropdown']")
        if dropdown_btn.count() == 0:
            # Final fallback: any dropdown on the page (only one expansion open)
            dropdown_btn = page.locator("button[class*='buttonDropdown']")
        if dropdown_btn.count() == 0:
            return "SKIP_NO_STEP"
        try:
            dropdown_btn.first.click(timeout=4000)
        except PWTimeout:
            return "SKIP_NO_STEP"
        page.wait_for_timeout(400)

        # 4. Click the "3-D STEP" option. Some parts offer the dropdown but no
        #    STEP option (other kits) — also skip silently.
        step_opt = page.locator("li:has-text('3-D STEP')")
        if step_opt.count() == 0:
            page.keyboard.press("Escape")
            return "SKIP_NO_STEP"
        step_opt.first.click(timeout=5000)
        page.wait_for_timeout(500)

        # 5. Click the panel's Download button and capture the file.
        dl_btn = panel.locator("button[class*='downloadButton']")
        if dl_btn.count() == 0:
            dl_btn = page.locator("button[class*='downloadButton']")
        target = out_dir / f"{pn}.step"
        # 12s captures fast downloads; if Playwright's save_as fails (Chrome's
        # CDP downloadPath redirects the file and cancels playwright's managed
        # download), fall through to scanning chrome_dl_dir for the file.
        captured = False
        try:
            with page.expect_download(timeout=12000) as dl_info:
                dl_btn.first.click(timeout=5000)
            try:
                dl_info.value.save_as(target)
                captured = True
            except Exception as save_err:
                print(f"      save_as failed ({save_err}); using chrome_dl_dir fallback")
        except PWTimeout:
            pass  # no download event — fall through to fallback

        if not captured:
            # Look in chrome_dl_dir for a file containing this part number.
            # Skip partial .crdownload files AND macOS AppleDouble metadata
            # (`._<filename>`) which appear on SMB-mounted volumes.
            if chrome_dl_dir is None:
                return None
            found = False
            for _ in range(10):
                page.wait_for_timeout(500)
                matches = sorted(
                    p for p in chrome_dl_dir.iterdir()
                    if pn in p.name
                    and not p.name.endswith(".crdownload")
                    and not p.name.startswith("._")
                )
                if matches:
                    matches[0].replace(target)
                    for dup in matches[1:]:
                        dup.unlink(missing_ok=True)
                    found = True
                    break
            if not found:
                print(f"      fallback: no temp file for {pn}")
                return None
            print(f"      fallback claimed temp file for {pn}")
        # Cancelled / silently-truncated downloads land as zero-byte files —
        # before this check the scraper logged them as OK and 41% of one run
        # was useless. Treat empty as a fail so the part will be retried.
        if not target.exists() or target.stat().st_size == 0:
            target.unlink(missing_ok=True)
            return None
        # A few McMaster STEP exports are pathological — exact knurl tessellation
        # blows up to 50-200 MB for a single dowel pin. Anything over 40 MB is
        # genuinely unusable; under that we keep it. Skip-mark on overflow so we
        # don't retry the same gigantic part next run.
        if target.stat().st_size > 40 * 1024 * 1024:
            target.unlink(missing_ok=True)
            return "SKIP_NO_STEP"
        return target.name
    except PWTimeout as e:
        print(f"      pw timeout in click chain ({pn}): {e}")
        return None
    except Exception as e:
        print(f"      err ({pn}): {e}")
        return None


def close_inline_panel(page: Page) -> None:
    """Close the currently-open inline CAD panel so the next click expands cleanly."""
    try:
        # Press Escape — usually closes the inline expand on McMaster
        page.keyboard.press("Escape")
        page.wait_for_timeout(200)
    except Exception:
        pass


def append_manifest(manifest_path: Path, part_number: str, filename: str, source_url: str) -> None:
    """Append one row to a per-subcategory manifest CSV."""
    new = not manifest_path.exists()
    with manifest_path.open("a") as f:
        if new:
            f.write("part_number,filename,source_url\n")
        f.write(f"{part_number},{filename},{source_url}\n")


def part_number(url: str) -> str:
    m = PART_NUMBER_RE.search(url)
    return m.group(1) if m else url.rstrip("/").split("/")[-1]


def _family_priority_key(fam_url: str, key: str) -> tuple[int, str]:
    """Sort key: priority families first (in declared order), then alphabetical."""
    slug = fam_url.lower()
    for i, token in enumerate(SHAPE_PRIORITY.get(key, [])):
        if token in slug:
            return (i, slug)
    return (10_000, slug)


def plan_category(page: Page, key: str, url: str) -> list[tuple[str, str]]:
    """Return [(shape_family_slug, leaf_url), ...].

    Ordered by SHAPE_PRIORITY for the category, then alphabetical for the rest.
    Within each shape family, leaves are deduped by geometric subtype.
    Assortment / kit leaves are excluded — they're multi-part packages, not
    individual fasteners, so they're irrelevant for the fastener detection model.
    """
    families = discover_shape_families(page, url, key)
    # Some categories (e.g. retaining-rings) list leaves directly off the macro
    # page. Drop assortment kits here — they'd otherwise slip through as
    # (fam, fam) since they have no nested material variants.
    families = [f for f in families if "assortment" not in f.lower()]
    families = sorted(families, key=lambda f: _family_priority_key(f, key))
    plan: list[tuple[str, str]] = []
    for fam in families:
        # If the "family" URL is itself a `~~/` leaf (categories like hinges,
        # latches, springs list leaves directly off the macro page), don't waste
        # a network round-trip probing for nested material variants — there
        # aren't any. Treat it as its own leaf.
        if fam.endswith("~~/"):
            plan.append((fam, fam))
            continue
        leaves = discover_material_leaves(page, fam)
        if not leaves:
            plan.append((fam, fam))
            continue
        canonical_leaves = dedup_by_geometry(leaves)
        canonical_leaves = [l for l in canonical_leaves if "assortment" not in l.lower()]
        # Exclude leaves that ship as multi-piece assemblies (rod + nut, screw + washer,
        # insert + tool kit). The CAD bundles two parts as one model — bad for a
        # per-fastener classifier. Keep -with-toggle / -with-groove / etc. since
        # those are integral features of the same part.
        canonical_leaves = [
            l for l in canonical_leaves
            if not _ASSEMBLY_RE.search(l.lower())
        ]
        for leaf in canonical_leaves:
            plan.append((fam, leaf))
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    default_out = Path("/Volumes/Uncle Sam/step-vr-step-thesis/fastener_labeling/New Files")
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--progress", type=Path,
                        default=default_out / "mcmaster_progress.json")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--category", help="Only run one (key from CATEGORY_URLS).")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, no downloads.")
    parser.add_argument("--dump-urls", type=Path,
                        help="Discover part URLs and write them to this file (with # subfolder "
                             "markers), then exit. For feeding into mcmaster_scraper_gui.py.")
    parser.add_argument(
        "--cdp",
        default="http://localhost:9222",
        help="Attach to an already-running Chrome on this CDP endpoint. "
             "Set to empty string to launch a fresh browser instead (which McMaster blocks).",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    # Chrome's default download directory — used by download_step_inline as a
    # fallback file source when Playwright's expect_download misses an event.
    # Whatever lands here gets moved to the proper category leaf, then the
    # post-category hook empties anything we couldn't claim.
    chrome_dl_dir = Path.home() / "Downloads"
    progress = load_progress(args.progress)

    cats = {args.category: CATEGORY_URLS[args.category]} if args.category else CATEGORY_URLS

    with sync_playwright() as pw:
        if args.cdp:
            # Attach to a Chrome the user already started with --remote-debugging-port=9222.
            # This uses their real browser fingerprint, which McMaster trusts.
            print(f"connecting to existing Chrome at {args.cdp} …")
            browser = pw.chromium.connect_over_cdp(args.cdp)
            context = browser.contexts[0] if browser.contexts else browser.new_context(accept_downloads=True)
            page = context.pages[0] if context.pages else context.new_page()
            # If page[0] is on a deep leaf URL, page.goto to a macro URL
            # sometimes silently no-ops (returns immediately at networkidle
            # without re-rendering). Force a real navigation by bouncing through
            # about:blank first.
            try:
                page.goto("about:blank", timeout=5000)
            except Exception:
                pass
            # NOTE: previously we tried Browser.setDownloadBehavior with a custom
            # downloadPath here. That broke Playwright's expect_download capture
            # (Chrome handled the file directly, save_as raised "canceled"). Now
            # we leave Chrome's default flow alone — captured downloads go via
            # Playwright's temp + save_as as designed; un-captured downloads
            # leak to ~/Downloads and get swept by the post-category hook.
        else:
            # Fallback: launch a fresh browser (McMaster blocks this).
            browser = pw.chromium.launch(
                headless=args.headless,
                channel="chrome",
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            )
            context = browser.new_context(
                accept_downloads=True,
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1440, "height": 900},
                locale="en-US",
            )
            context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            )
            page = context.new_page()

        for key, url in cats.items():
            quota = CATEGORY_QUOTAS.get(key, 100)
            already = progress.count(key)
            if already >= quota:
                print(f"[{key}] quota met ({already}/{quota})")
                continue

            print(f"[{key}] planning…")
            plan = plan_category(page, key, url)
            print(f"[{key}] {len(plan)} distinct geometric subtypes:")
            for fam, leaf in plan:
                print(f"    {leaf.rstrip('/').split('/')[-1]}")

            if args.dump_urls:
                # Visit each leaf, harvest part URLs, write to file with # markers.
                quota = CATEGORY_QUOTAS.get(key, 100)
                per_family = max(1, quota // max(1, len(plan)))
                with args.dump_urls.open("a") as f:
                    for _fam, leaf in plan:
                        leaf_slug = leaf.rstrip("/").split("/")[-1].rstrip("~")
                        leaf_slug = _TRAILING_NUM_RE.sub("", leaf_slug)
                        f.write(f"\n# {key}/{leaf_slug}\n")
                        try:
                            urls = collect_part_links(page, leaf, target=per_family)
                        except Exception as e:
                            print(f"  [{leaf_slug}] err: {e}")
                            continue
                        for u in urls:
                            f.write(u + "\n")
                        print(f"  [{leaf_slug}] {len(urls)} parts")
                continue

            if args.dry_run:
                continue

            cat_dir = args.out / key
            cat_dir.mkdir(exist_ok=True)
            per_leaf = max(1, (quota - already) // max(1, len(plan)))

            for _fam_slug, leaf in plan:
                if progress.count(key) >= quota:
                    break

                # Subfolder named after the geometric subtype (the leaf slug).
                leaf_slug = leaf.rstrip("/").split("/")[-1].rstrip("~")
                leaf_slug = _TRAILING_NUM_RE.sub("", leaf_slug)
                subdir = cat_dir / leaf_slug
                subdir.mkdir(exist_ok=True)
                manifest = subdir / "_manifest.csv"

                print(f"  [{key}] {leaf_slug} (target {per_leaf})")
                # Navigate to leaf once; collect_part_links does the goto and harvest.
                part_urls = collect_part_links(page, leaf, target=per_leaf * 2)
                print(f"    {len(part_urls)} part links found")
                # Reset per-leaf so a problem leaf doesn't poison the next one.
                consecutive_fails = 0

                for purl in part_urls:
                    if progress.count(key) >= quota:
                        break
                    pn = part_number(purl)
                    if progress.has(key, pn):
                        continue
                    # Click inline — no navigation away from the leaf page.
                    fname = download_step_inline(page, purl, subdir, chrome_dl_dir)
                    if fname == "SKIP_NO_STEP":
                        # Part doesn't offer STEP (e.g. assortments). Mark as
                        # done so we don't retry, but don't count as failure.
                        progress.add(key, pn)
                        consecutive_fails = 0
                        print(f"    SKIP {pn} (no STEP available)")
                    elif fname:
                        progress.add(key, pn)
                        append_manifest(manifest, pn, fname, purl)
                        consecutive_fails = 0
                        print(f"    OK  {pn}  ({progress.count(key)}/{quota})")
                    else:
                        # Mark failed parts as done (in progress.downloaded) so a
                        # resume doesn't retry them and re-trigger the abort.
                        # The manifest stays clean — only successful downloads
                        # are appended there.
                        progress.add(key, pn)
                        progress.failed.append(pn)
                        consecutive_fails += 1
                        print(f"    FAIL {pn}")
                    close_inline_panel(page)
                    save_progress(args.progress, progress)
                    # 5-9s jitter — slightly more conservative than 4-7s to keep
                    # the authenticated session healthy.
                    time.sleep(random.uniform(5.0, 9.0))
                    if consecutive_fails >= 6:
                        # 6 fails in a single leaf usually means this leaf's
                        # parts don't have STEP (assortments, kits, weird shapes).
                        # Skip the rest of this leaf and move to the next one
                        # rather than killing the whole run.
                        print(f"  [{key}] {leaf_slug}: 6 fails — abandoning leaf")
                        break

        if not args.cdp:
            browser.close()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
