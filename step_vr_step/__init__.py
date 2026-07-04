"""step-vr-step: Bidirectional CAD ↔ Unreal round-trip system."""
__version__ = "1.0.0"


def _register_handlers(server):
    """Register all RPC command handlers on the given RPCServer."""
    # Each handler is imported lazily to avoid pulling in heavy deps at startup.

    def handle_convert(req_id, args, emitter):
        from .cli import cmd_convert
        import argparse
        ns = argparse.Namespace(
            source=args.get("input_path", ""),
            target=args.get("output_path", ""),
            format=args.get("format", "step_bundle"),
            detect_fasteners=args.get("detect_fasteners", False),
            ml=args.get("ml", False),
        )
        cmd_convert(ns)

    def handle_validate(req_id, args, emitter):
        from .cli import cmd_validate
        import argparse
        ns = argparse.Namespace(bundle_path=args.get("bundle_path", ""))
        cmd_validate(ns)

    def handle_reconcile(req_id, args, emitter):
        from .cli import cmd_reconcile
        import argparse
        ns = argparse.Namespace(
            edited_step=args.get("edited_step", ""),
            original_bundle=args.get("original_bundle", ""),
            output=args.get("output", ""),
        )
        cmd_reconcile(ns)

    def handle_detect_fasteners(req_id, args, emitter):
        from .cli import cmd_detect
        import argparse
        ns = argparse.Namespace(
            input=args.get("input_path", ""),
            output=args.get("output_path"),
            ml=args.get("ml", False),
            lod=args.get("lod", False),
        )
        cmd_detect(ns)

    server.register("convert", handle_convert)
    server.register("validate", handle_validate)
    server.register("reconcile", handle_reconcile)
    server.register("detect_fasteners", handle_detect_fasteners)
