"""Standalone CLI entry point for step-vr-step.

Usage:
    python -m step_vr_step.cli convert input.step output.step
    python -m step_vr_step.cli validate bundle_path/
    python -m step_vr_step.cli reconcile edited.step original.bundle/ output/
    python -m step_vr_step.cli serve   # start RPC server (used by Tauri sidecar)
"""

import argparse
import logging
import sys

logger = logging.getLogger("step_vr_step")


def cmd_convert(args):
    """Run a file conversion."""
    from .rpc.server import EventEmitter

    emitter = EventEmitter()
    req_id = "cli_convert"

    emitter.log(req_id, f"Converting {args.source} -> {args.target} (format: {args.format})")

    source = args.source
    target = args.target
    target_format = args.format

    # Detect input format
    ext = source.lower().rsplit(".", 1)[-1] if "." in source else ""

    doc = None
    manifest = None
    shapes = None

    if ext in ("step", "stp"):
        from .readers.step_reader import read_step
        emitter.progress(req_id, "reading", 0.1, f"Reading STEP file: {source}")
        doc, manifest, shapes = read_step(source, return_shapes=True)
    elif ext in ("gltf", "glb"):
        from .readers.gltf_reader import read_gltf
        emitter.progress(req_id, "reading", 0.1, f"Reading glTF file: {source}")
        doc, manifest = read_gltf(source)
    elif ext == "udatasmith":
        from .readers.datasmith_reader import read_datasmith
        emitter.progress(req_id, "reading", 0.1, f"Reading Datasmith file: {source}")
        doc, manifest = read_datasmith(source)
    else:
        emitter.error(req_id, "READ_UNSUPPORTED_FORMAT",
                      f"Unsupported input format: .{ext}")
        return 1

    # Optional fastener detection
    if getattr(args, "detect_fasteners", False) and manifest is not None:
        emitter.progress(req_id, "fastener_detection", 0.5, "Detecting fasteners...")
        from .detection import detect_fasteners
        from .config import DetectionConfig
        det_config = DetectionConfig(
            enable_ml=getattr(args, "ml", False),
            brepformer_weights=getattr(args, "brepformer_ckpt", None),
            pointnet2_weights=getattr(args, "pointnet_ckpt", None),
            ml_confidence_threshold=getattr(args, "ml_threshold", 0.50),
        )
        # `shapes` was set by the STEP reader above (or None for glTF/datasmith
        # inputs); detect_fasteners requires it for B-Rep feature extraction.
        manifest = detect_fasteners(manifest, shapes=shapes, config=det_config)
        detected = sum(
            1 for p in manifest.parts
            if p.detection and p.detection.fastener_type != "unclassified"
        )
        emitter.log(req_id, f"Detected {detected} fasteners in {len(manifest.parts)} parts")

    # Write output
    if target_format == "step_bundle":
        from .writers.step_writer import write_step_bundle
        emitter.progress(req_id, "step_write", 0.7, f"Writing STEP bundle: {target}")
        write_step_bundle(doc, manifest, target)
    elif target_format == "datasmith":
        from .writers.datasmith_writer import write_datasmith
        emitter.progress(req_id, "datasmith_write", 0.7, f"Writing Datasmith: {target}")
        write_datasmith(doc, manifest, target)
    elif target_format == "gltf":
        from .writers.gltf_writer import write_gltf
        emitter.progress(req_id, "gltf_write", 0.7, f"Writing glTF: {target}")
        write_gltf(doc, manifest, target)
    else:
        emitter.error(req_id, "WRITE_UNSUPPORTED_FORMAT",
                      f"Unsupported output format: {target_format}")
        return 1

    # Optional: emit a USD copy for Maya / DCC consumption.
    usd_target = getattr(args, "emit_usd", None)
    if usd_target:
        from .exporters import write_usd, USD_AVAILABLE
        if not USD_AVAILABLE:
            emitter.error(req_id, "USD_NOT_INSTALLED",
                          "usd-core not installed; cannot emit USD")
            return 1
        glb_for_usd = target if target_format == "gltf" else None
        emitter.progress(req_id, "usd_write", 0.9, f"Writing USD: {usd_target}")
        write_usd(manifest, usd_target, glb_path=glb_for_usd)

    emitter.progress(req_id, "done", 1.0, "Conversion complete")
    emitter.result(req_id, "ok", target)
    return 0


def cmd_validate(args):
    """Run the validation harness on a bundle."""
    from .rpc.server import EventEmitter

    emitter = EventEmitter()
    req_id = "cli_validate"
    emitter.log(req_id, f"Validating bundle: {args.bundle_path}")

    from .validation.report import run_full_validation
    report = run_full_validation(args.bundle_path)

    emitter.emit({"evt": "validation", "req_id": req_id, "report": report.model_dump()})

    all_passed = all(c.passed for c in report.checks)
    emitter.result(req_id, "ok" if all_passed else "error", args.bundle_path)
    return 0 if all_passed else 1


def cmd_reconcile(args):
    """Run reconciliation between edited STEP and original bundle."""
    from .rpc.server import EventEmitter

    emitter = EventEmitter()
    req_id = "cli_reconcile"
    emitter.log(req_id, f"Reconciling {args.edited_step} with {args.original_bundle}")

    from .reconciliation import reconcile
    result = reconcile(args.edited_step, args.original_bundle, args.output, emitter, req_id)

    emitter.result(req_id, "ok", args.output)
    return 0


def cmd_detect(args):
    """Run fastener detection on a STEP file."""
    from .rpc.server import EventEmitter

    emitter = EventEmitter()
    req_id = "cli_detect"
    emitter.log(req_id, f"Detecting fasteners in {args.input}")

    ext = args.input.lower().rsplit(".", 1)[-1] if "." in args.input else ""
    if ext not in ("step", "stp"):
        emitter.error(req_id, "READ_UNSUPPORTED_FORMAT",
                      "Fastener detection requires STEP input")
        return 1

    from .readers.step_reader import read_step
    emitter.progress(req_id, "reading", 0.1, f"Reading STEP file: {args.input}")
    doc, manifest, shapes = read_step(args.input, return_shapes=True)

    from .detection import detect_fasteners
    from .config import DetectionConfig
    det_config = DetectionConfig(
        enable_ml=args.ml,
        enable_lod_substitution=args.lod,
        rule_confidence_threshold=getattr(args, "threshold", 0.60),
        ml_confidence_threshold=getattr(args, "ml_threshold", 0.50),
        brepformer_weights=getattr(args, "brepformer_ckpt", None),
        pointnet2_weights=getattr(args, "pointnet_ckpt", None),
    )

    emitter.progress(req_id, "fastener_detection", 0.4, "Running detection...")
    manifest = detect_fasteners(manifest, shapes=shapes, config=det_config)

    detected = sum(
        1 for p in manifest.parts
        if p.detection and p.detection.fastener_type != "unclassified"
    )
    emitter.log(req_id, f"Detected {detected} fasteners in {len(manifest.parts)} parts")

    # Write manifest
    import json, os
    output = args.output or "manifest.json"
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, "w") as f:
        json.dump(manifest.model_dump(mode="json"), f, indent=2, default=str)

    # Optional: emit a Maya-friendly USD alongside the manifest.
    # USD is OPTIONAL — if usd-core isn't installed, warn and continue
    # (the manifest.json is the primary output).
    usd_target = getattr(args, "emit_usd", None)
    if usd_target:
        try:
            from .exporters import write_usd, USD_AVAILABLE
            if not USD_AVAILABLE:
                emitter.log(req_id, "WARN: usd-core not installed; skipping USD. "
                            "Run `pip install usd-core` to enable Maya export.")
            else:
                usd_dir = os.path.dirname(usd_target)
                if usd_dir:
                    os.makedirs(usd_dir, exist_ok=True)
                glb_for_usd = getattr(args, "glb_source", None)
                emitter.log(req_id, f"Writing USD: {usd_target}")
                write_usd(manifest, usd_target, glb_path=glb_for_usd)
        except Exception as exc:
            emitter.log(req_id, f"WARN: USD export failed ({exc}); manifest still written.")

    emitter.progress(req_id, "done", 1.0, "Detection complete")
    emitter.result(req_id, "ok", output)
    return 0


def cmd_serve(args):
    """Start the RPC server for Tauri sidecar mode."""
    from .rpc.server import RPCServer
    from . import _register_handlers

    server = RPCServer()
    _register_handlers(server)
    server.run()
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="step-vr-step",
        description="Bidirectional CAD <-> Unreal round-trip system",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    # convert
    p_conv = sub.add_parser("convert", help="Convert between formats")
    p_conv.add_argument("source", help="Input file path")
    p_conv.add_argument("target", help="Output file/folder path")
    p_conv.add_argument("-f", "--format", default="step_bundle",
                        choices=["step_bundle", "datasmith", "gltf"],
                        help="Output format")
    p_conv.add_argument("--detect-fasteners", action="store_true",
                        help="Run fastener detection during conversion")
    p_conv.add_argument("--ml", action="store_true",
                        help="Enable ML-based detection (requires PyTorch)")
    p_conv.add_argument("--brepformer-ckpt", dest="brepformer_ckpt", default=None,
                        help="Path to trained BRepFormer checkpoint")
    p_conv.add_argument("--pointnet-ckpt", dest="pointnet_ckpt", default=None,
                        help="Path to trained PointNet++ checkpoint")
    p_conv.add_argument("--ml-threshold", dest="ml_threshold", type=float, default=0.50,
                        help="ML confidence threshold")
    p_conv.add_argument("--emit-usd", dest="emit_usd", default=None,
                        help="Also write a Maya-importable .usd to this path")
    p_conv.set_defaults(func=cmd_convert)

    # validate
    p_val = sub.add_parser("validate", help="Validate a bundle")
    p_val.add_argument("bundle_path", help="Path to bundle folder")
    p_val.set_defaults(func=cmd_validate)

    # reconcile
    p_rec = sub.add_parser("reconcile", help="Reconcile edited STEP with original bundle")
    p_rec.add_argument("edited_step", help="Edited STEP file")
    p_rec.add_argument("original_bundle", help="Original bundle folder")
    p_rec.add_argument("output", help="Output folder")
    p_rec.set_defaults(func=cmd_reconcile)

    # detect
    p_det = sub.add_parser("detect", help="Detect fasteners in a STEP file")
    p_det.add_argument("input", help="Input STEP file path")
    p_det.add_argument("-o", "--output", default=None,
                       help="Output manifest path (default: manifest.json)")
    p_det.add_argument("--ml", action="store_true",
                       help="Enable ML-based detection (requires PyTorch)")
    p_det.add_argument("--lod", action="store_true",
                       help="Enable LOD substitution for detected fasteners")
    p_det.add_argument("--threshold", type=float, default=0.60,
                       help="Rule-based confidence threshold (default 0.60)")
    p_det.add_argument("--ml-threshold", dest="ml_threshold", type=float, default=0.50,
                       help="ML confidence threshold (default 0.50)")
    p_det.add_argument("--brepformer-ckpt", dest="brepformer_ckpt", default=None,
                       help="Path to trained BRepFormer checkpoint (.pth or .ckpt)")
    p_det.add_argument("--pointnet-ckpt", dest="pointnet_ckpt", default=None,
                       help="Path to trained PointNet++ checkpoint")
    p_det.add_argument("--emit-usd", dest="emit_usd", default=None,
                       help="Also write a Maya-importable .usd to this path")
    p_det.add_argument("--glb-source", dest="glb_source", default=None,
                       help="Optional glb file to copy mesh data from into the USD")
    p_det.set_defaults(func=cmd_detect)

    # serve
    p_srv = sub.add_parser("serve", help="Start RPC server (Tauri sidecar mode)")
    p_srv.set_defaults(func=cmd_serve)

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,  # Logs go to stderr; stdout is for RPC
    )

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
