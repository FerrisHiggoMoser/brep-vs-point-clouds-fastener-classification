"""NDJSON RPC server for the Python sidecar process.

Reads commands from stdin (one JSON per line), dispatches to handlers,
writes events to stdout (one JSON per line). Tauri spawns this process
and communicates over stdin/stdout.
"""
from __future__ import annotations

import json
import sys
import traceback
import logging
from typing import Any, Callable

from .protocol import (
    CommandMessage,
    ProgressEvent,
    LogEvent,
    ResultEvent,
    ErrorEvent,
    ErrorCode,
    ERROR_MESSAGES,
)

logger = logging.getLogger("step_vr_step.rpc")


class EventEmitter:
    """Writes NDJSON events to stdout for the Tauri frontend."""

    def __init__(self, stream=None):
        self._stream = stream or sys.stdout

    def emit(self, event: dict[str, Any] | Any) -> None:
        """Write a single event as a JSON line to stdout."""
        if hasattr(event, "model_dump"):
            data = event.model_dump()
        elif isinstance(event, dict):
            data = event
        else:
            data = {"evt": "log", "msg": str(event)}

        line = json.dumps(data, default=str)
        self._stream.write(line + "\n")
        self._stream.flush()

    def progress(self, req_id: str, stage: str, pct: float, msg: str = "") -> None:
        self.emit(ProgressEvent(req_id=req_id, stage=stage, pct=pct, msg=msg))

    def log(self, req_id: str, msg: str, level: str = "info") -> None:
        self.emit(LogEvent(req_id=req_id, level=level, msg=msg))

    def result(self, req_id: str, status: str = "ok", output_path: str = "") -> None:
        self.emit(ResultEvent(req_id=req_id, status=status, output_path=output_path))

    def error(self, req_id: str, code: str, detail: str = "") -> None:
        msg = ERROR_MESSAGES.get(code, detail)
        self.emit(ErrorEvent(req_id=req_id, code=code, detail=detail or msg))


CommandHandler = Callable[[CommandMessage, EventEmitter], None]


class RPCServer:
    """NDJSON RPC server reading from stdin, writing to stdout."""

    def __init__(self):
        self._handlers: dict[str, CommandHandler] = {}
        self._emitter = EventEmitter()
        self._running = False
        self._active_requests: dict[str, bool] = {}  # req_id -> cancelled

    def register(self, cmd: str, handler: CommandHandler) -> None:
        """Register a handler for a command type."""
        self._handlers[cmd] = handler

    @property
    def emitter(self) -> EventEmitter:
        return self._emitter

    def run(self) -> None:
        """Main loop: read commands from stdin, dispatch to handlers."""
        self._running = True
        logger.info("RPC server started, waiting for commands on stdin")

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
                cmd = CommandMessage(**raw)
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Failed to parse command: {e}")
                self._emitter.error(
                    req_id="unknown",
                    code="PARSE_ERROR",
                    detail=f"Failed to parse command: {e}",
                )
                continue

            # Handle cancel specially
            if cmd.cmd == "cancel":
                target_id = cmd.args.get("req_id", "")
                if target_id in self._active_requests:
                    self._active_requests[target_id] = True  # Mark cancelled
                    self._emitter.log(cmd.id, f"Cancellation requested for {target_id}")
                continue

            handler = self._handlers.get(cmd.cmd)
            if not handler:
                self._emitter.error(
                    req_id=cmd.id,
                    code="UNKNOWN_COMMAND",
                    detail=f"Unknown command: {cmd.cmd}",
                )
                continue

            # Track active request
            self._active_requests[cmd.id] = False

            try:
                self._emitter.log(cmd.id, f"Starting command: {cmd.cmd}")
                handler(cmd, self._emitter)
            except Exception as e:
                logger.error(f"Handler error for {cmd.cmd}: {traceback.format_exc()}")
                self._emitter.error(
                    req_id=cmd.id,
                    code="HANDLER_ERROR",
                    detail=str(e),
                )
            finally:
                self._active_requests.pop(cmd.id, None)

        self._running = False
        logger.info("RPC server stopped (stdin closed)")

    def is_cancelled(self, req_id: str) -> bool:
        """Check if a request has been cancelled."""
        return self._active_requests.get(req_id, False)

    def stop(self) -> None:
        self._running = False
