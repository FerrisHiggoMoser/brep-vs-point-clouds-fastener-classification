from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class TessellationConfig:
    linear_deflection: float = 0.1        # mm — chord height tolerance
    angular_deflection: float = 0.05      # rad — max normal angle between adjacent tris
    relative: bool = False
    parallel: bool = True

@dataclass
class ReconciliationConfig:
    match_strategy: Literal["uuid_first", "fingerprint_first", "hybrid"] = "hybrid"
    fingerprint_tolerance: float = 0.01   # relative volume
    prompt_on_ambiguous: bool = True
    auto_accept_unchanged: bool = True

@dataclass
class CoordinateConfig:
    source_system: Literal["RH_Y_up_mm", "RH_Z_up_mm", "LH_Z_up_cm"] = "LH_Z_up_cm"
    target_system: Literal["RH_Y_up_mm", "RH_Z_up_mm", "LH_Z_up_cm"] = "RH_Z_up_mm"
    unit_scale: float = 10.0              # cm → mm

@dataclass
class ValidationConfig:
    hausdorff_tolerance: float = 0.05     # mm
    volume_tolerance: float = 0.001       # relative
    run_roundtrip_check: bool = True
    run_visual_diff: bool = False
    fail_on_pmi_loss: bool = True

@dataclass
class ReconstructionConfig:
    primitive_fit_tolerance: float = 0.1
    ransac_min_support: int = 500
    nurbs_degree: int = 3
    nurbs_grid_resolution: int = 20
    max_sew_tolerance: float = 0.5

@dataclass
class DetectionConfig:
    enable_rule_based: bool = True
    enable_ml: bool = False
    rule_confidence_threshold: float = 0.60
    ml_confidence_threshold: float = 0.50
    dimension_tolerance_mm: float = 0.1
    thread_pitch_tolerance_mm: float = 0.05
    cylindrical_threshold: float = 0.50
    min_repetition_count: int = 3
    pointnet2_weights: Optional[str] = None
    brepformer_weights: Optional[str] = None
    enable_lod_substitution: bool = False
    lod_default_level: int = 1


@dataclass
class PipelineConfig:
    tessellation: TessellationConfig = None
    reconciliation: ReconciliationConfig = None
    coordinates: CoordinateConfig = None
    validation: ValidationConfig = None
    reconstruction: ReconstructionConfig = None
    detection: DetectionConfig = None

    def __post_init__(self):
        if self.tessellation is None:
            self.tessellation = TessellationConfig()
        if self.reconciliation is None:
            self.reconciliation = ReconciliationConfig()
        if self.coordinates is None:
            self.coordinates = CoordinateConfig()
        if self.validation is None:
            self.validation = ValidationConfig()
        if self.reconstruction is None:
            self.reconstruction = ReconstructionConfig()
        if self.detection is None:
            self.detection = DetectionConfig()
