"""Pydantic models for the manifest.json sidecar bundle schema (spec §6.2)."""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from uuid import UUID
import datetime


class Transform(BaseModel):
    translation: tuple[float, float, float]
    rotation_quat: tuple[float, float, float, float]  # xyzw
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


class Fingerprint(BaseModel):
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    volume_mm3: float
    surface_area_mm2: float
    topology_hash: str  # SHA256 of face/edge/vertex count signature
    vertex_count: int
    face_count: int


class PBRMaterial(BaseModel):
    name: str
    base_color: tuple[float, float, float, float]  # RGBA 0-1
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    albedo_texture: Optional[str] = None       # relative path in textures/
    normal_texture: Optional[str] = None
    roughness_texture: Optional[str] = None
    metallic_texture: Optional[str] = None
    ao_texture: Optional[str] = None
    emissive_texture: Optional[str] = None
    alpha_mode: Literal["opaque", "mask", "blend"] = "opaque"
    double_sided: bool = False


class Relationship(BaseModel):
    kind: Literal["fastener", "mate", "contact", "weld", "bond", "contained_in", "custom"]
    subject_uuid: UUID
    target_uuid: UUID
    params: dict = {}   # e.g. {"hole_diameter": 6.0, "fastener_std": "MS20470"}
    inferred: bool = True
    confidence: float = 1.0
    broken: bool = False  # set by reconciliation if geometry no longer supports it


class PMIEntry(BaseModel):
    kind: Literal["dimension", "gdt", "surface_finish", "weld_symbol", "note"]
    target_uuid: UUID
    target_face_id: Optional[str] = None  # STEP face identifier
    value: str
    tolerance: Optional[str] = None
    annotation_plane: Optional[Transform] = None
    preserved_through_edit: bool = True


class UnrealSpecific(BaseModel):
    actor_class: str = "StaticMeshActor"   # or Blueprint class path
    blueprint_path: Optional[str] = None
    tags: list[str] = []
    data_layers: list[str] = []
    outliner_folder: Optional[str] = None
    collision_profile: Optional[str] = None
    collision_geometry: list[dict] = []    # simple primitives
    lod_count: int = 1
    lod_screen_sizes: list[float] = []
    mobility: Literal["Static", "Stationary", "Movable"] = "Static"
    variants: dict = {}
    custom_properties: dict = {}           # full DatasmithMetaData pass-through


class ProvenanceRecord(BaseModel):
    source_type: Literal["original_step", "unreal_native", "user_imported_mesh", "reconstructed", "library_instance"]
    original_step_path: Optional[str] = None
    original_entity_id: Optional[str] = None
    original_brep_hash: Optional[str] = None
    import_timestamp: datetime.datetime
    edit_log: list[dict] = []              # every edit gesture that touched this part
    reconstruction_note: Optional[str] = None  # populated for reconstructed parts


class DetectionLabel(BaseModel):
    fastener_type: str = "unclassified"
    standard: Optional[str] = None       # e.g. "ISO 4014"
    variant: Optional[str] = None        # e.g. "M6x30"
    confidence: float = 0.0
    method: Literal["rule_based", "ml_pointnet2", "ml_brepformer", "ensemble"] = "rule_based"
    detected_dimensions: dict = {}       # shaft_dia, head_dia, head_ht, length, etc.
    lod_proxy: Optional[str] = None      # key into proxy library


class PartEntry(BaseModel):
    uuid: UUID
    step_entity_id: str                    # the STEP #NNN reference
    name: str
    parent_uuid: Optional[UUID] = None     # None for root
    transform: Transform                   # world transform
    local_transform: Transform             # relative to parent
    fingerprint: Fingerprint
    material: PBRMaterial
    provenance: ProvenanceRecord
    unreal: UnrealSpecific
    pmi: list[PMIEntry] = []
    tier_edits: list[dict] = []            # editable T1-T4 params
    detection: Optional[DetectionLabel] = None
    tessellation: dict = {}                # tessellation params used
    notes: str = ""


class BundleMetadata(BaseModel):
    bundle_version: str = "1.0"
    created: datetime.datetime
    created_by: str
    app_version: str
    source_format: Literal["unreal", "step", "gltf", "datasmith"]
    unreal_engine_version: Optional[str] = None
    coordinate_system: Literal["RH_Y_up_mm", "RH_Z_up_mm", "LH_Z_up_cm"] = "RH_Y_up_mm"
    units: Literal["mm", "m", "in"] = "mm"


class Manifest(BaseModel):
    meta: BundleMetadata
    parts: list[PartEntry]
    relationships: list[Relationship]
    global_pmi: list[PMIEntry] = []
    unmatched_on_return: list[dict] = []   # populated by reconciliation
    conflict_log: list[dict] = []
