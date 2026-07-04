"""Sidecar server bridging the Tauri frontend to the new backend.

Reads JSON commands from stdin (one per line), dispatches to the appropriate
backend pipeline, emits NDJSON progress events to stdout.

Supports both the legacy command format (action/input_path) and the new
RPC format (id/cmd/args).
"""
from __future__ import annotations

import json
import sys
import os
import logging
import traceback
from pathlib import Path

# Ensure the backend package is importable
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from step_vr_step.rpc.server import EventEmitter

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sidecar")

emitter = EventEmitter()


def handle_command(raw: dict) -> None:
    """Dispatch a command from the frontend."""
    # Detect format: legacy (has "action") or new RPC (has "cmd")
    if "cmd" in raw:
        handle_rpc_command(raw)
    elif "action" in raw:
        handle_legacy_command(raw)
    else:
        emitter.emit({"type": "error", "timestamp": _ts(),
                       "message": f"Unknown command format: {list(raw.keys())}"})


def handle_rpc_command(raw: dict) -> None:
    """Handle new RPC format: {id, cmd, args}."""
    req_id = raw.get("id", "req_0")
    cmd = raw.get("cmd", "")
    args = raw.get("args", {})

    try:
        if cmd == "open_file":
            _cmd_open_file(req_id, args)
        elif cmd == "convert":
            _cmd_convert(req_id, args)
        elif cmd == "reconcile":
            _cmd_reconcile(req_id, args)
        elif cmd == "validate":
            _cmd_validate(req_id, args)
        else:
            emitter.error(req_id, "UNKNOWN_COMMAND", f"Unknown command: {cmd}")
    except Exception as e:
        logger.error(traceback.format_exc())
        emitter.error(req_id, "HANDLER_ERROR", str(e))


def handle_legacy_command(raw: dict) -> None:
    """Handle legacy format: {action, input_path, output_dir, ...}."""
    action = raw.get("action", "")
    input_path = raw.get("input_path", "")
    output_dir = raw.get("output_dir", "data/output")
    config = raw.get("config", {})
    req_id = "legacy_0"

    try:
        if action in ("forward", "full_pipeline"):
            _run_full_pipeline(req_id, input_path, output_dir, config)
        elif action == "detect":
            # Detection is part of the full pipeline in the new backend
            _run_full_pipeline(req_id, input_path, output_dir, config)
        elif action == "reverse":
            _run_full_pipeline(req_id, input_path, output_dir, config)
        elif action == "reduce_lod":
            _run_reduce_lod(req_id, input_path, raw.get("config") or {})
        elif action == "ping":
            emitter.emit({"type": "result", "timestamp": _ts(),
                           "action": "pong", "message": "alive"})
        else:
            emitter.emit({"type": "error", "timestamp": _ts(),
                           "message": f"Unknown action: {action}"})
    except Exception as e:
        logger.error(traceback.format_exc())
        emitter.emit({"type": "error", "timestamp": _ts(),
                       "message": str(e)})


def _run_full_pipeline(req_id: str, input_path: str, output_dir: str, config: dict) -> None:
    """Run the full conversion pipeline on an input file."""
    from step_vr_step.readers.step_reader import read_step
    from step_vr_step.readers.gltf_reader import read_gltf
    from step_vr_step.writers.step_writer import write_step_bundle
    from step_vr_step.writers.gltf_writer import write_gltf
    from step_vr_step.validation.report import run_full_validation

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = input_path.suffix.lower()

    # Emit progress in legacy format for backward compat
    _emit_progress(req_id, "reading", 0.05, f"Reading {input_path.name}")

    # Check for sibling bundle (carries Unreal metadata from previous round-trip)
    from step_vr_step.sidecar.bundle import find_sibling_bundle
    bundle_path = find_sibling_bundle(input_path)
    original_manifest = None

    if bundle_path:
        _emit_log(req_id, "info", f"Found sidecar bundle: {bundle_path.name}/ — Unreal metadata will be preserved")
        try:
            from step_vr_step.sidecar.manifest import read_manifest
            original_manifest = read_manifest(bundle_path)
            _emit_log(req_id, "info", f"Bundle contains {len(original_manifest.parts)} parts, {len(original_manifest.relationships)} relationships")
            if any(p.unreal.tags for p in original_manifest.parts):
                tag_count = sum(len(p.unreal.tags) for p in original_manifest.parts)
                _emit_log(req_id, "info", f"Bundle preserves {tag_count} Unreal tag(s)")
            if any(p.unreal.blueprint_path for p in original_manifest.parts):
                bp_count = sum(1 for p in original_manifest.parts if p.unreal.blueprint_path)
                _emit_log(req_id, "info", f"Bundle preserves {bp_count} Blueprint reference(s)")
            if any(p.material.albedo_texture for p in original_manifest.parts):
                tex_count = sum(1 for p in original_manifest.parts if p.material.albedo_texture)
                _emit_log(req_id, "info", f"Bundle preserves {tex_count} PBR texture(s)")
        except Exception as e:
            _emit_log(req_id, "warning", f"Could not read bundle manifest: {e}")
            original_manifest = None
    else:
        _emit_log(req_id, "info", f"No sidecar bundle found — treating as fresh import (no Unreal metadata)")

    # 1. Read input — STEP also returns per-part TopoDS_Shape dict so
    # downstream BRepFormer / B-Rep feature extraction can actually run.
    shapes_by_uuid: dict = {}
    if ext in (".step", ".stp"):
        doc, manifest, shapes_by_uuid = read_step(str(input_path), return_shapes=True)
    elif ext in (".gltf", ".glb"):
        doc, manifest = read_gltf(str(input_path))
    else:
        emitter.emit({"type": "error", "timestamp": _ts(),
                       "message": f"Unsupported file type: {ext}"})
        return

    part_count = len(manifest.parts)
    _emit_progress(req_id, "building_xde", 0.2, f"Loaded {part_count} parts")

    # 2. Forward: write glTF for the viewer
    # Build tessellation config from frontend overrides. Two density knobs
    # come from the Sidebar:
    #   circular_density  — desired chord error (mm) on curved surfaces
    #                       (cylinders, spheres, cones, tori)
    #   irregular_density — desired chord error (mm) on freeform/NURBS/planar
    # OCC has a single linear_deflection that applies globally + an
    # angular_deflection that only matters on curved faces. We use the finer
    # of the two as the global linear constraint (so circles get circular
    # quality when finer; NURBS get irregular quality when finer) and derive
    # angular from circular_density.
    from step_vr_step.config import TessellationConfig
    tess_config = TessellationConfig()
    legacy_res = config.get("tessellation_resolution")
    circ = config.get("circular_density")
    irreg = config.get("irregular_density")
    if circ is None and legacy_res is not None:
        circ = legacy_res
    if irreg is None and legacy_res is not None:
        irreg = legacy_res
    if circ is not None or irreg is not None:
        circ_v = float(circ) if circ is not None else tess_config.linear_deflection
        irreg_v = float(irreg) if irreg is not None else tess_config.linear_deflection
        tess_config.linear_deflection = min(circ_v, irreg_v)
        # Angular deflection: smaller circular_density → finer angle.
        # 0.01mm→~0.003rad (very fine), 1mm→0.3rad, 5mm→1.5rad (clamp).
        tess_config.angular_deflection = max(0.01, min(1.5, circ_v * 0.3))
    # Tier breakpoints aligned with the UI sidebar's irregular_density slider
    # (Sidebar.tsx:403). The backend uses min(circ, irreg) for the actual
    # OCC linear_deflection, so the irregular tier is the relevant one for
    # the user-visible label.
    res = tess_config.linear_deflection
    if res <= 0.05:
        mesh_label = "ultra"
    elif res <= 0.15:
        mesh_label = "high"
    elif res <= 0.5:
        mesh_label = "medium"
    elif res <= 1.5:
        mesh_label = "low"
    elif res <= 3.0:
        mesh_label = "lowpoly"
    else:
        mesh_label = "potato"
    mesh_tag = f"_mesh-{mesh_label}"

    _emit_log(req_id, "info",
              f"Mesh density: {mesh_label} "
              f"(linear={tess_config.linear_deflection:.3f}mm, "
              f"angular={tess_config.angular_deflection:.2f}rad)")

    _emit_progress(req_id, "tessellation", 0.3, f"Generating glTF ({mesh_label} density)")
    gltf_path = write_gltf(doc, manifest,
                           str(output_dir / f"{input_path.stem}{mesh_tag}.gltf"),
                           tessellation_config=tess_config)
    _emit_progress(req_id, "tessellation", 0.5, f"glTF written: {gltf_path}")

    # Write manifest FIRST so when the frontend reacts to forward_complete
    # and loads manifest.json from disk, the file actually exists.
    from step_vr_step.sidecar.manifest import write_manifest
    write_manifest(manifest, output_dir / "manifest.json")

    # Now emit forward_complete for the UI
    emitter.emit({
        "type": "result",
        "timestamp": _ts(),
        "action": "forward_complete",
        "gltf_path": str(gltf_path),
        "manifest_path": str(output_dir / "manifest.json"),
        "part_count": part_count,
    })

    # 3. Detection — call the real rule-based detector. If the step_reader
    # didn't surface TopoDS_Shape per part we get empty features and the
    # geometric classifier returns "unclassified" for everything; we then
    # apply a name-substring fallback so common fastener-named parts still
    # get a sensible label.
    _emit_progress(req_id, "fingerprinting", 0.6, "Analyzing parts")
    try:
        from step_vr_step.detection.detect import detect_fasteners
        from step_vr_step.schema import DetectionLabel
        from step_vr_step.config import DetectionConfig

        # Wire the trained BRepFormer + PointNet++ checkpoints. Search
        # several known locations so this works in the dev repo, the
        # reproducible build layout, and the research-output tree.
        repo_root = Path(__file__).resolve().parents[2]
        def _find_ckpt(env_var: str, *candidates):
            override = os.environ.get(env_var)
            if override:
                return override
            for c in candidates:
                if c.exists():
                    return str(c)
            return str(candidates[0])
        bf_ckpt = _find_ckpt("BF_CKPT_PATH",
            repo_root / "models" / "bf_subtype13_best.ckpt",                            # build layout
            repo_root / "cad-bidirectional-poc" / "models" / "bf_subtype13_best.ckpt",   # dev repo
            repo_root / "training_data" / "mcmaster_logs" / "bf_subtype13_best.ckpt",    # research output
        )
        det_config = DetectionConfig(
            enable_rule_based=True,
            enable_ml=True,
            brepformer_weights=bf_ckpt if Path(bf_ckpt).exists() else None,
            # CLI parity: BRepFormer only, no PointNet++ in the ensemble.
            ml_confidence_threshold=0.40,
        )
        if det_config.brepformer_weights:
            _emit_log(req_id, "info", f"BRepFormer enabled: {Path(det_config.brepformer_weights).name}")
        else:
            _emit_log(req_id, "warning", "BRepFormer checkpoint not found; ML detection disabled")
            det_config.enable_ml = False

        manifest = detect_fasteners(manifest, shapes=shapes_by_uuid, config=det_config)
    except Exception as e:
        import traceback
        _emit_log(req_id, "warning", f"detect_fasteners failed ({e}); falling back to name heuristic only")
        logger.warning(traceback.format_exc())
        from step_vr_step.schema import DetectionLabel
        for p in manifest.parts:
            if p.detection is None:
                p.detection = DetectionLabel(fastener_type="unclassified", confidence=0.0)

    # No name-substring fallback. Detection is BRep-only by user requirement —
    # a part called "Bolt" gets the same answer as one called "part_47", only
    # the geometry decides.

    # Aggregate counts by type. Cover all 13 BF subtype labels plus the
    # legacy 5; map plural BF training labels (e.g. "screws") and rule-engine
    # variants (e.g. "hex_nut", "flat_washer") to singular buckets.
    BF_TO_SINGULAR = {
        # BF 13-class (plurals)
        "anchors": "anchor", "keys": "key", "nails": "nail", "nuts": "nut",
        "pins": "pin", "retaining-rings": "retaining_ring",
        "rivets": "rivet", "screws": "screw", "spacers": "spacer",
        "threaded-inserts": "threaded_insert", "threaded-rods": "threaded_rod",
        "washers": "washer",
        # Rule-engine outputs (from ISO/DIN dimension tables)
        "hex_bolt": "bolt", "socket_head_cap_screw": "screw",
        "hex_nut": "nut", "lock_nut": "nut", "wing_nut": "nut",
        "flat_washer": "washer", "lock_washer": "washer", "spring_washer": "washer",
        "pop_rivet": "rivet", "solid_rivet": "rivet",
    }
    FASTENER_KEYS = (
        "bolt", "screw", "nut", "washer", "rivet",
        "pin", "anchor", "key", "nail", "retaining_ring",
        "spacer", "threaded_insert", "threaded_rod",
    )
    type_counts: dict[str, int] = {k: 0 for k in FASTENER_KEYS}
    fastener_uuids: list[str] = []
    non_fastener_uuids: list[str] = []
    conf_sum = 0.0
    conf_count = 0
    for p in manifest.parts:
        d = p.detection
        if d is None:
            non_fastener_uuids.append(str(p.uuid))
            continue
        ftype = d.fastener_type.replace("possible_", "").replace("likely_", "")
        ftype = BF_TO_SINGULAR.get(ftype, ftype)
        if ftype in type_counts:
            type_counts[ftype] += 1
            fastener_uuids.append(str(p.uuid))
            conf_sum += float(d.confidence)
            conf_count += 1
        else:
            non_fastener_uuids.append(str(p.uuid))

    avg_conf = (conf_sum / conf_count) if conf_count > 0 else 0.0
    _emit_log(req_id, "info",
              f"Detection: {len(fastener_uuids)} fasteners, "
              f"{len(non_fastener_uuids)} non-fasteners, "
              f"{len(manifest.relationships)} relationships")

    # Re-write the manifest now that detection populated `part.detection`
    # and `manifest.relationships`.
    write_manifest(manifest, output_dir / "manifest.json")

    # 3b. Emit Maya-importable USD with the full relationship graph
    # (vrs:screwedInto / vrs:containedIn arcs per fastener prim).
    try:
        from step_vr_step.exporters import write_usd, USD_AVAILABLE
        if USD_AVAILABLE:
            usd_path = output_dir / f"{input_path.stem}.usdc"
            write_usd(manifest, usd_path, glb_path=None, fmt="usdc")
            _emit_log(req_id, "info", f"USD written: {usd_path.name}")
        else:
            _emit_log(req_id, "info", "usd-core not installed; skipping USD export")
    except Exception as e:
        _emit_log(req_id, "warning", f"USD export failed: {e}")

    emitter.emit({
        "type": "result",
        "timestamp": _ts(),
        "action": "detect_complete",
        "total_parts": part_count,
        "fasteners_found": len(fastener_uuids),
        "non_fasteners_found": len(non_fastener_uuids),
        # Legacy 5 (kept for backward compat with the UI's stat bars)
        "bolts": type_counts["bolt"],
        "screws": type_counts["screw"],
        "nuts": type_counts["nut"],
        "washers": type_counts["washer"],
        "rivets": type_counts["rivet"],
        # 8 additional BF subtypes
        "pins": type_counts["pin"],
        "anchors": type_counts["anchor"],
        "keys": type_counts["key"],
        "nails": type_counts["nail"],
        "retaining_rings": type_counts["retaining_ring"],
        "spacers": type_counts["spacer"],
        "threaded_inserts": type_counts["threaded_insert"],
        "threaded_rods": type_counts["threaded_rod"],
        "avg_confidence": avg_conf,
        "relationships": len(manifest.relationships),
    })

    # 4. Write standalone STEP file + sidecar bundle
    _emit_progress(req_id, "step_write", 0.7, "Writing STEP file")

    # Write the sidecar bundle (contains manifest, textures, history, etc.)
    bundle_dir = write_step_bundle(doc, manifest, str(output_dir / f"{input_path.stem}.bundle"))

    # Also copy the STEP file to the output dir root for easy access
    import shutil
    standalone_step = output_dir / f"{input_path.stem}{mesh_tag}.step"
    shutil.copy2(str(bundle_dir / "part.step"), str(standalone_step))
    _emit_log(req_id, "info", f"STEP written: {standalone_step.name}")

    emitter.emit({
        "type": "result",
        "timestamp": _ts(),
        "action": "reverse_complete",
        "step_path": str(standalone_step),
    })

    # 5. Validate (compare output against input manifest)
    _emit_progress(req_id, "validation", 0.85, "Running validation")

    total_checks = 8

    def on_check_done(check_result, check_index):
        """Emit each check result as it completes so the UI can show real-time progress."""
        emitter.emit({
            "type": "validation_check",
            "evt": "validation_check",
            "timestamp": _ts(),
            "req_id": req_id,
            "check": check_result.model_dump(),
            "index": check_index,
            "total": total_checks,
        })

    report = run_full_validation(
        bundle_dir, original_manifest=manifest, on_check_complete=on_check_done
    )

    # Emit validation report (new format)
    emitter.emit({
        "type": "validation",
        "evt": "validation",
        "timestamp": _ts(),
        "req_id": req_id,
        "report": report.model_dump(),
    })

    _emit_progress(req_id, "done", 1.0,
                   f"Complete: {part_count} parts, {report.summary}")

    emitter.emit({
        "type": "result",
        "evt": "result",
        "timestamp": _ts(),
        "action": "pipeline_complete",
        "req_id": req_id,
        "status": "ok",
        "output_path": str(output_dir),
    })


def _cmd_open_file(req_id: str, args: dict) -> None:
    """Handle open_file RPC command."""
    path = args.get("path", "")
    _run_full_pipeline(req_id, path, str(Path(path).parent / "output"), {})


def _cmd_convert(req_id: str, args: dict) -> None:
    """Handle convert RPC command."""
    source = args.get("source_path", "")
    target = args.get("target_path", "")
    _run_full_pipeline(req_id, source, target, args.get("options", {}))


def _cmd_reconcile(req_id: str, args: dict) -> None:
    """Handle reconcile RPC command."""
    from step_vr_step.reconciliation import reconcile

    edited = args.get("edited_step_path", "")
    bundle = args.get("original_bundle_path", "")
    output = args.get("output_path", "")

    result = reconcile(edited, bundle, output, emitter, req_id)
    emitter.result(req_id, "ok", output)


def _cmd_validate(req_id: str, args: dict) -> None:
    """Handle validate RPC command."""
    from step_vr_step.validation.report import run_full_validation

    bundle_path = args.get("bundle_path", "")
    report = run_full_validation(bundle_path)

    emitter.emit({
        "evt": "validation",
        "req_id": req_id,
        "report": report.model_dump(),
    })
    emitter.result(req_id, "ok" if report.all_passed else "error", bundle_path)


def _run_reduce_lod(req_id: str, manifest_path: str, config: dict) -> None:
    """Apply an LOD reduction to a single part and rewrite the manifest.

    Expected config keys:
        part_id:      str  — UUID of the part to reduce
        target_tier:  str  — one of "transform_only" | "metadata_only" |
                              "constrained_geo" | "freeform_mesh"

    Emits a `lod_reduced` result event with the updated part on success.
    """
    from step_vr_step.sidecar.manifest import read_manifest, write_manifest
    from step_vr_step.schema import DetectionLabel

    part_id = str(config.get("part_id", "") or "")
    target_tier = str(config.get("target_tier", "metadata_only") or "metadata_only")
    if not part_id:
        emitter.emit({"type": "error", "timestamp": _ts(),
                       "message": "reduce_lod: missing part_id"})
        return

    manifest_file = Path(manifest_path)
    if manifest_file.is_dir():
        manifest_file = manifest_file / "manifest.json"
    if not manifest_file.exists():
        emitter.emit({"type": "error", "timestamp": _ts(),
                       "message": f"reduce_lod: manifest not found at {manifest_file}"})
        return

    _emit_progress(req_id, "lod_reduce", 0.1, f"Loading manifest from {manifest_file.name}")
    manifest = read_manifest(manifest_file.parent)

    target = None
    for p in manifest.parts:
        if str(p.uuid) == part_id:
            target = p
            break
    if target is None:
        emitter.emit({"type": "error", "timestamp": _ts(),
                       "message": f"reduce_lod: part {part_id} not found"})
        return

    _emit_progress(req_id, "lod_reduce", 0.5, f"Reducing {target.name} → {target_tier}")
    # Record the LOD change as a tier_edit. The actual mesh-decimation can be
    # plugged in by a follow-up; the tier label propagates to the frontend's
    # `edit_tier` via the manifest loader's mapping.
    target.tier_edits.append({
        "kind": "lod_reduce",
        "target_tier": target_tier,
        "timestamp": _ts(),
    })
    # If the part has a detection label and is a known fastener, generate a
    # proxy mesh via the existing proxy library so the reduction is real.
    try:
        from step_vr_step.lod.proxy_library import get_proxy
        from step_vr_step.lod.lod_levels import LODLevel
        det = target.detection
        if det and det.fastener_type not in ("unclassified", "unknown"):
            tier_to_lod = {
                "transform_only": LODLevel.L0,
                "constrained_geo": LODLevel.L1,
                "metadata_only": LODLevel.L2,
                "freeform_mesh": LODLevel.L0,
            }
            lod = tier_to_lod.get(target_tier, LODLevel.L1)
            dims = det.detected_dimensions or {}
            dia = float(dims.get("shaft_dia", dims.get("head_dia", 6.0)))
            length = float(dims.get("length", 20.0))
            proxy = get_proxy(det.fastener_type, dia, length, lod)
            if proxy is not None:
                verts, _faces = proxy
                target.fingerprint.vertex_count = len(verts)
    except Exception as e:
        _emit_log(req_id, "warning", f"proxy generation skipped: {e}")

    _emit_progress(req_id, "lod_reduce", 0.85, "Writing manifest")
    write_manifest(manifest, manifest_file.parent)

    emitter.emit({
        "type": "result",
        "timestamp": _ts(),
        "action": "lod_reduced",
        "part_id": part_id,
        "target_tier": target_tier,
        "part": json.loads(target.model_dump_json()),
    })
    _emit_progress(req_id, "done", 1.0, f"LOD reduced for {target.name}")


def _emit_progress(req_id: str, stage: str, pct: float, msg: str) -> None:
    """Emit progress in both legacy and new format."""
    emitter.emit({
        "type": "progress",
        "evt": "progress",
        "timestamp": _ts(),
        "req_id": req_id,
        "stage": stage,
        "progress": pct * 100,
        "pct": pct,
        "detail": msg,
        "msg": msg,
    })


def _emit_log(req_id: str, level: str, msg: str) -> None:
    """Emit a log message to the frontend."""
    emitter.emit({
        "type": "log",
        "evt": "log",
        "timestamp": _ts(),
        "req_id": req_id,
        "level": level,
        "message": msg,
        "msg": msg,
    })


def _ts() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def main():
    logger.info("step-vr-step sidecar server started")
    logger.info(f"Python: {sys.executable}")
    logger.info(f"Backend: {backend_dir}")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            raw = json.loads(line)
            handle_command(raw)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {line[:100]}")
            emitter.emit({"type": "error", "timestamp": _ts(),
                           "message": f"Invalid JSON input"})
        except Exception as e:
            logger.error(f"Unhandled error: {traceback.format_exc()}")
            emitter.emit({"type": "error", "timestamp": _ts(),
                           "message": str(e)})

    logger.info("Sidecar server exiting (stdin closed)")


if __name__ == "__main__":
    main()
