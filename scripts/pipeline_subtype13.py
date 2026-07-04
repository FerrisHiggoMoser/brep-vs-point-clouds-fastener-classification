"""4-phase orchestrator for the 13-class fastener-subtype experiment.

Phases: relabel -> train_pn -> train_bf -> analysis
State written to training_data/pipeline_state.subtype13.json so the cron-fire
agent can read it and decide auto-restart actions.

Usage:
    python backend/scripts/pipeline_subtype13.py --start relabel
    python backend/scripts/pipeline_subtype13.py --start train_pn --skip relabel
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(r"c:\Users\ferri\OneDrive\Documents\GitHub\step-vr-step")
BACKEND = REPO / "backend"
PC_ROOT = REPO / "training_data" / "mcmaster_pc_subtype13"
BREP_ROOT = REPO / "training_data" / "mcmaster_brep_subtype13"
LOGS = REPO / "training_data" / "mcmaster_logs"
STATE = REPO / "training_data" / "pipeline_state.subtype13.json"

PN_LOG = LOGS / "train_pn_subtype13.log"
BF_LOG = LOGS / "train_bf_subtype13.log"
ANALYSIS_LOG = LOGS / "analysis_subtype13.log"

PHASE_NAMES = ["relabel", "train_pn", "train_bf", "analysis"]
EPOCHS = 120


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {
        "started": _now(),
        "phases": {p: {"status": "pending"} for p in PHASE_NAMES},
        "watchdog": {"restart_count_per_phase": {}, "last_action": None, "last_restart_at": None},
    }


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE)


def update_phase(name: str, **kwargs) -> None:
    state = load_state()
    state["phases"].setdefault(name, {})
    state["phases"][name].update(kwargs)
    save_state(state)


def phase_relabel() -> None:
    update_phase("relabel", status="running", start=_now())
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / "relabel_subtype13.log"
    cmd = [sys.executable, "-u", str(BACKEND / "scripts" / "relabel_subtype_13.py")]
    print(f"[relabel] {' '.join(cmd)}")
    with log.open("ab") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(BACKEND))
    if proc.returncode != 0:
        update_phase("relabel", status="error", end=_now(), exit_code=proc.returncode)
        raise RuntimeError(f"relabel failed: exit {proc.returncode}")
    # Sanity: count classes per split
    counts: dict = {}
    for tree, root in (("pc", PC_ROOT), ("brep", BREP_ROOT)):
        for split in ("train", "val", "test"):
            split_dir = root / split
            if not split_dir.exists():
                continue
            classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
            counts[f"{tree}/{split}"] = {"num_classes": len(classes), "classes": classes}
    update_phase("relabel", status="done", end=_now(), exit_code=0, counts=counts)
    print(f"[relabel] done")


def _spawn_training(cmd: list[str], log_path: Path, phase_name: str,
                    epoch_re: re.Pattern) -> int:
    """Run a training command, tail-parse for epoch progress, update state.

    Returns the subprocess exit code.
    """
    log_path.write_text("")  # truncate
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"
    print(f"[{phase_name}] {' '.join(cmd)}")
    with log_path.open("ab") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(BACKEND), env=env)
        last_pos = 0
        while proc.poll() is None:
            try:
                with log_path.open("rb") as rf:
                    rf.seek(last_pos)
                    chunk = rf.read().decode("utf-8", errors="replace")
                    last_pos = rf.tell()
                eps = epoch_re.findall(chunk)
                if eps:
                    last = eps[-1]
                    if isinstance(last, tuple):
                        e = int(last[0])
                        t = int(last[1]) if len(last) > 1 and last[1] else EPOCHS
                    else:
                        e = int(last); t = EPOCHS
                    update_phase(phase_name, current_epoch=e, epochs=t, log_path=str(log_path))
            except Exception:
                pass
            time.sleep(15)
    return proc.returncode


def phase_train_pn() -> None:
    update_phase("train_pn", status="running", start=_now(), epochs=EPOCHS, current_epoch=0)
    cmd = [
        sys.executable, "-u", "-m", "step_vr_step.models.pointnet2.train",
        "--data_path", str(PC_ROOT),
        "--epochs", str(EPOCHS),
        "--batch_size", "16",
        "--num_points", "4096",
        "--num_workers", "2",
        "--use_normals",
        "--lr", "0.001",
        "--log_dir", str(LOGS / "pointnet2_subtype13"),
    ]
    rc = _spawn_training(cmd, PN_LOG, "train_pn", re.compile(r"Epoch (\d+)/(\d+)"))
    update_phase("train_pn", status="done" if rc == 0 else "error",
                 end=_now(), exit_code=rc)
    if rc != 0:
        raise RuntimeError(f"train_pn failed: exit {rc}")
    print(f"[train_pn] done")


def phase_train_bf() -> None:
    update_phase("train_bf", status="running", start=_now(), epochs=EPOCHS, current_epoch=0)
    cmd = [
        sys.executable, "-u", "-m", "step_vr_step.models.brepformer.train",
        "--data_dir", str(BREP_ROOT),
        "--epochs", str(EPOCHS),
        "--batch_size", "8",
        "--num_workers", "2",
        "--lr", "0.001",
        "--log_dir", str(LOGS / "brepformer_subtype13"),
        "--num_classes", "13",
        "--balanced_sampling",
    ]
    rc = _spawn_training(cmd, BF_LOG, "train_bf",
                         re.compile(r"[Ee]poch[ =:]?\s*(\d+)\s*(?:/\s*(\d+))?"))
    update_phase("train_bf", status="done" if rc == 0 else "error",
                 end=_now(), exit_code=rc)
    if rc != 0:
        raise RuntimeError(f"train_bf failed: exit {rc}")
    print(f"[train_bf] done")


def phase_analysis() -> None:
    update_phase("analysis", status="running", start=_now())
    cmd = [sys.executable, "-u", str(BACKEND / "scripts" / "full_analysis_subtype13.py")]
    ANALYSIS_LOG.write_text("")
    print(f"[analysis] {' '.join(cmd)}")
    with ANALYSIS_LOG.open("ab") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(BACKEND))
    update_phase("analysis", status="done" if proc.returncode == 0 else "error",
                 end=_now(), exit_code=proc.returncode)
    if proc.returncode != 0:
        raise RuntimeError(f"analysis failed: exit {proc.returncode}")
    print(f"[analysis] done")


PHASES = {
    "relabel": phase_relabel,
    "train_pn": phase_train_pn,
    "train_bf": phase_train_bf,
    "analysis": phase_analysis,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="relabel", choices=PHASE_NAMES)
    p.add_argument("--only", choices=PHASE_NAMES, help="run only this phase")
    p.add_argument("--skip", nargs="*", default=[], choices=PHASE_NAMES, help="skip these phases")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.only:
        plan = [args.only]
    else:
        i = PHASE_NAMES.index(args.start)
        plan = [n for n in PHASE_NAMES[i:] if n not in args.skip]
    print(f"pipeline plan: {' -> '.join(plan)}")
    for ph in plan:
        try:
            PHASES[ph]()
        except Exception as e:
            update_phase(ph, status="error", end=_now(),
                         error=f"{e.__class__.__name__}: {e}",
                         trace=traceback.format_exc())
            print(f"[{ph}] FAILED: {e}", file=sys.stderr)
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
