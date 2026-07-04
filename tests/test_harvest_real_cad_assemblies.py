from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "harvest_real_cad_assemblies.py"
SPEC = importlib.util.spec_from_file_location("harvest_real_cad_assemblies", SCRIPT_PATH)
harvest = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = harvest
SPEC.loader.exec_module(harvest)


def make_args(tmp_path: Path, **overrides):
    defaults = {
        "work_dir": tmp_path / "work",
        "stage_root": tmp_path / "staged",
        "stage_name_negatives": False,
        "no_decompose_assemblies": True,
        "decompose_all_steps": False,
        "decompose_min_mb": 2.0,
        "max_assembly_mb": 250.0,
        "max_exported_parts_per_assembly": 5000,
        "max_member_mb": 1,
        "max_total_mb": 4,
        "force_extract": False,
        "force_stage": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_classify_name_finds_standard_fasteners():
    label = harvest.classify_name("ISO 4762 M6x20 socket head cap screw.stp")
    assert label.label == "fastener"
    assert label.subtype == "socket_screw"

    nut = harvest.classify_name("DIN 934 - M8 hex nut.STEP")
    assert nut.label == "fastener"
    assert nut.subtype == "hex_nut"


def test_classify_name_keeps_strong_non_fasteners_out_of_fastener_bucket():
    label = harvest.classify_name("Left mounting bracket plate.step")
    assert label.label == "non_fastener"

    unknown = harvest.classify_name("Part_001.step")
    assert unknown.label == "unknown"


def test_safe_archive_rel_rejects_traversal():
    assert harvest.safe_archive_rel("../evil.step") is None
    assert harvest.safe_archive_rel("nested/../../evil.step") is None
    assert harvest.safe_archive_rel("/absolute/evil.step") is None
    assert str(harvest.safe_archive_rel("nested/ISO 4032 M6.stp")) == "nested/ISO 4032 M6.stp"


def test_process_zip_archive_stages_named_step_fastener(tmp_path):
    archive = tmp_path / "assembly.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Steam/ISO 4762 M6x20.stp", "ISO-10303-21;\nEND-ISO-10303-21;\n")
        zf.writestr("Steam/render.png", b"not cad")
        zf.writestr("../escape.step", "bad")

    rows = []
    stats = harvest.HarvestStats()
    args = make_args(tmp_path)
    harvest.process_zip_archive(archive, rows, stats, args)

    staged = list((tmp_path / "staged").rglob("*.step"))
    assert len(staged) == 1
    assert staged[0].parent.name == "fastener"
    assert stats.staged_fasteners == 1
    assert stats.errors == 1
    assert any(row["status"] == "skipped_unsafe_path" for row in rows)


def test_process_zip_archive_can_stage_strong_name_negatives(tmp_path):
    archive = tmp_path / "assembly.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Gearbox/base_plate.step", "ISO-10303-21;\nEND-ISO-10303-21;\n")

    rows = []
    stats = harvest.HarvestStats()
    args = make_args(tmp_path, stage_name_negatives=True)
    harvest.process_zip_archive(archive, rows, stats, args)

    staged = list((tmp_path / "staged").rglob("*.step"))
    assert len(staged) == 1
    assert staged[0].parent.name == "non_fastener"
    assert stats.staged_non_fasteners == 1
