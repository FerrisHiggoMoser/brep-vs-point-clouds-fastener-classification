"""RPC protocol message types for Tauri ↔ Python sidecar communication.

Communication is NDJSON over stdin/stdout. One JSON object per line.
Commands flow Tauri → Python; events flow Python → Tauri.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel


# --- Command messages (Tauri → Python) ---

class CommandMessage(BaseModel):
    """A command from the Tauri frontend to the Python sidecar."""
    id: str
    cmd: Literal[
        "open_file",
        "preview_scene",
        "convert",
        "reconcile",
        "validate",
        "detect_fasteners",
        "cancel",
    ]
    args: dict[str, Any] = {}


# --- Event messages (Python → Tauri) ---

ProgressStage = Literal[
    "reading",
    "parsing",
    "building_xde",
    "tessellation",
    "fingerprinting",
    "relationship_inference",
    "fastener_detection",
    "lod_substitution",
    "sidecar_write",
    "step_write",
    "validation",
    "reconciliation_match",
    "reconciliation_diff",
    "datasmith_write",
    "gltf_write",
    "done",
]

STAGE_COLORS: dict[str, str] = {
    "reading": "#888888",
    "parsing": "#5b9dd9",
    "building_xde": "#5b9dd9",
    "tessellation": "#6fcf97",
    "fingerprinting": "#6fcf97",
    "relationship_inference": "#c678dd",
    "fastener_detection": "#f5a623",
    "lod_substitution": "#f5a623",
    "sidecar_write": "#e0a050",
    "step_write": "#e0a050",
    "validation": "#5b9dd9",
    "reconciliation_match": "#c678dd",
    "reconciliation_diff": "#c678dd",
    "datasmith_write": "#e0a050",
    "gltf_write": "#e0a050",
    "done": "#6fcf97",
}


class ProgressEvent(BaseModel):
    evt: Literal["progress"] = "progress"
    req_id: str
    stage: ProgressStage
    pct: float  # 0.0 to 1.0
    msg: str = ""


class LogEvent(BaseModel):
    evt: Literal["log"] = "log"
    req_id: str
    level: Literal["debug", "info", "warning", "error"] = "info"
    msg: str = ""


class SceneTreeEvent(BaseModel):
    evt: Literal["scene_tree"] = "scene_tree"
    req_id: str
    tree: dict[str, Any]


class PreviewMeshEvent(BaseModel):
    evt: Literal["preview_mesh"] = "preview_mesh"
    req_id: str
    uuid: str
    mesh_url: str


class DiffEvent(BaseModel):
    evt: Literal["diff"] = "diff"
    req_id: str
    diff: dict[str, Any]


class ConflictEvent(BaseModel):
    evt: Literal["conflict"] = "conflict"
    req_id: str
    conflict: dict[str, Any]
    needs_user: bool = True


class ValidationEvent(BaseModel):
    evt: Literal["validation"] = "validation"
    req_id: str
    report: dict[str, Any]


class ResultEvent(BaseModel):
    evt: Literal["result"] = "result"
    req_id: str
    status: Literal["ok", "error", "partial"]
    output_path: str = ""


class DetectionEvent(BaseModel):
    evt: Literal["detection"] = "detection"
    req_id: str
    results: dict[str, Any]


class ErrorEvent(BaseModel):
    evt: Literal["error"] = "error"
    req_id: str
    code: str
    detail: str = ""


# Error codes from spec §16
class ErrorCode:
    READ_UNSUPPORTED_FORMAT = "READ_UNSUPPORTED_FORMAT"
    READ_CORRUPT = "READ_CORRUPT"
    WRITE_SCHEMA_FAIL = "WRITE_SCHEMA_FAIL"
    RECONCILE_NO_BUNDLE = "RECONCILE_NO_BUNDLE"
    RECONCILE_AMBIGUOUS = "RECONCILE_AMBIGUOUS"
    UNREAL_PLUGIN_MISSING = "UNREAL_PLUGIN_MISSING"
    VALIDATION_FAIL = "VALIDATION_FAIL"
    T4_EDIT_ON_PMI_FACE = "T4_EDIT_ON_PMI_FACE"


ERROR_MESSAGES: dict[str, str] = {
    ErrorCode.READ_UNSUPPORTED_FORMAT: "File type not recognized. Supported: STEP, glTF, Datasmith.",
    ErrorCode.READ_CORRUPT: "Input file appears corrupt. Details in log.",
    ErrorCode.WRITE_SCHEMA_FAIL: "Output failed STEP schema validation. See report.",
    ErrorCode.RECONCILE_NO_BUNDLE: "Cannot reconcile: original bundle not found. Expected at {path}.",
    ErrorCode.RECONCILE_AMBIGUOUS: "Reconciliation has unresolved ambiguities. Please complete prompts.",
    ErrorCode.UNREAL_PLUGIN_MISSING: "Unreal Engine installation not detected. Please set path in Settings.",
    ErrorCode.VALIDATION_FAIL: "Conversion completed but validation failed. See report before using output.",
    ErrorCode.T4_EDIT_ON_PMI_FACE: "This edit will destroy {count} PMI entries. Proceed / Cancel / Enter sculpt-mode anyway?",
}
