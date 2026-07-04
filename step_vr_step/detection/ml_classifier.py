"""Optional ML-based fastener classifier wrapping PointNet++ and BRepFormer.

PyTorch is an optional dependency. All imports are guarded so the module
can be imported even when torch is not installed — ``is_ml_available()``
will simply return False.
"""

import logging
from typing import Optional

import numpy as np

from ..schema import DetectionLabel
from ..config import DetectionConfig

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def is_ml_available() -> bool:
    """Return True when PyTorch is installed and inference is possible."""
    return _TORCH_AVAILABLE


class FastenerClassifier:
    """Unified inference wrapper for PointNet++ and BRepFormer models.

    Raises ``RuntimeError`` at construction time if PyTorch is absent.
    """

    def __init__(self, config: DetectionConfig):
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. "
                "Install ML dependencies with: pip install step-vr-step[ml]"
            )

        self.config = config
        self.pointnet_model = None
        self.brepformer_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.pointnet2_weights:
            self.pointnet_model = self._load_pointnet(config.pointnet2_weights)
        if config.brepformer_weights:
            self.brepformer_model = self._load_brepformer(config.brepformer_weights)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_pointnet(self, checkpoint_path: str):
        """Load a trained PointNet++ MSG checkpoint."""
        from ..models.pointnet2.pointnet2_cls_msg import PointNet2ClsMSG

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        num_classes = checkpoint.get("num_classes", 10)
        model = PointNet2ClsMSG(num_classes=num_classes, use_normals=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        logger.info("Loaded PointNet++ from %s (%d classes)", checkpoint_path, num_classes)
        return model

    def _load_brepformer(self, checkpoint_path: str):
        """Load a trained BRepFormer checkpoint.

        Accepts two formats:
          - Lightning .ckpt: top-level key 'state_dict' with 'model.' prefix
          - Bare .pth: top-level key 'model_state_dict' or a raw state_dict
        """
        from ..models.brepformer.brepformer import BRepFormer

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Discover the state dict and number of classes.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            sd = checkpoint["model_state_dict"]
            num_classes = checkpoint.get("num_classes", 13)
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
            # Strip Lightning's "model." prefix.
            sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
                  for k, v in sd.items()}
            # Try to derive class count from the classifier head shape.
            num_classes = 13
            for key, tensor in sd.items():
                if key.endswith(("classifier.weight", "head.weight", "fc.weight")) and tensor.ndim == 2:
                    num_classes = tensor.shape[0]
                    break
        else:
            # Assume raw state_dict
            sd = checkpoint
            num_classes = 13

        model = BRepFormer(num_classes=num_classes)
        # strict=False — the head may have additional buffers / aux layers
        # used during training but not at inference.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning("BRepFormer load: %d missing keys (e.g. %s)", len(missing), missing[:2])
        if unexpected:
            logger.warning("BRepFormer load: %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:2])
        model.to(self.device)
        model.eval()
        logger.info("Loaded BRepFormer from %s (%d classes)", checkpoint_path, num_classes)
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify_pointcloud(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        class_names: Optional[list[str]] = None,
    ) -> DetectionLabel:
        """Classify from point cloud (Nx3 positions + Nx3 normals).

        Args:
            points: (N, 3) point positions.
            normals: (N, 3) point normals.
            class_names: ordered list mapping class index → name.

        Returns:
            DetectionLabel with confidence thresholds applied.
        """
        if self.pointnet_model is None:
            return DetectionLabel(fastener_type="unclassified", confidence=0.0, method="ml_pointnet2")

        # Normalize to unit sphere
        centroid = points.mean(axis=0)
        pts = points - centroid
        scale = np.max(np.linalg.norm(pts, axis=1))
        if scale > 1e-12:
            pts = pts / scale

        # Combine xyz + normals → (1, N, 6)
        features = np.concatenate([pts, normals], axis=1).astype(np.float32)
        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)  # (1, N, 6)
        tensor = tensor.transpose(1, 2)  # (1, 6, N) — PointNet++ expects (B, C, N)

        with torch.no_grad():
            logits, _ = self.pointnet_model(tensor)
            probs = torch.softmax(logits, dim=-1)
            confidence, pred_class = probs.max(dim=-1)

        conf = float(confidence[0])
        cls_idx = int(pred_class[0])
        cls_name = class_names[cls_idx] if class_names and cls_idx < len(class_names) else f"class_{cls_idx}"

        return self._apply_thresholds(cls_name, conf, "ml_pointnet2")

    def classify_brep(
        self,
        face_grids: np.ndarray,
        edge_grids: np.ndarray,
        topo_distances: dict[str, np.ndarray],
        class_names: Optional[list[str]] = None,
    ) -> DetectionLabel:
        """Classify from B-Rep features.

        Args:
            face_grids: (Nf, 10, 10, 7) face UV-grid features.
            edge_grids: (Ne, 10, 12) edge curve features.
            topo_distances: dict with 4 distance matrices, each (Nf, Nf).
            class_names: ordered list mapping class index → name.
        """
        if self.brepformer_model is None:
            return DetectionLabel(fastener_type="unclassified", confidence=0.0, method="ml_brepformer")

        face_t = torch.from_numpy(face_grids).float().unsqueeze(0).to(self.device)
        edge_t = torch.from_numpy(edge_grids).float().unsqueeze(0).to(self.device)

        topo_t = {}
        for key, mat in topo_distances.items():
            topo_t[key] = torch.from_numpy(mat).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.brepformer_model(face_t, edge_t, topo_t)
            probs = torch.softmax(logits, dim=-1)
            confidence, pred_class = probs.max(dim=-1)

        conf = float(confidence[0])
        cls_idx = int(pred_class[0])
        cls_name = class_names[cls_idx] if class_names and cls_idx < len(class_names) else f"class_{cls_idx}"

        # Capture the top-3 softmax distribution so ensemble_merge can spot
        # OOD inputs (high non_fastener probability) and demote BF when
        # rule-based said "unclassified".
        probs_np = probs[0].detach().cpu().numpy()
        top3_idx = np.argsort(-probs_np)[:3]
        top3 = [
            (class_names[i] if class_names and i < len(class_names) else f"class_{i}",
             float(probs_np[i]))
            for i in top3_idx
        ]

        label = self._apply_thresholds(cls_name, conf, "ml_brepformer")
        label.detected_dimensions = {"ml_top3": top3, "ml_engine": "brepformer"}
        return label

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_thresholds(self, cls_name: str, confidence: float, method: str) -> DetectionLabel:
        threshold = self.config.ml_confidence_threshold
        if confidence >= 0.85:
            ftype = cls_name
        elif confidence >= threshold:
            ftype = f"likely_{cls_name}"
        else:
            ftype = "unclassified"

        return DetectionLabel(
            fastener_type=ftype,
            confidence=round(confidence, 4),
            method=method,
        )


def _passes_geometry_check(fastener_type: str, features) -> bool:
    """Pure-BRep sanity check: does the predicted class match the part's
    measured geometry? Used to reject BRepFormer predictions that the
    model is confident about but that contradict the part's shape.

    Examples this catches on the ISIS satellite:
      - DIN912_M2_5X12 (long thin SHCS, aspect 4.8) → BF says "washers"
        with 0.96. A washer has aspect_flat ≤ 0.4. This rejects washer.
      - DIN125A_M2 (flat washer, aspect_flat 0.07) → BF says "pins" with
        0.97. A pin has aspect ≥ 1. This rejects pin.

    No name lookups, no McMaster-specific tricks — just shape sanity.
    Returns True if the type is consistent with the part's geometry,
    False if it's contradicted.
    """
    if features is None:
        return True

    # Disc-aware aspect ratio: min(bbox) / max(bbox). 0 = paper-thin,
    # 1 = cube-like.
    bbox_dims = sorted([features.bbox_max[i] - features.bbox_min[i] for i in range(3)])
    if bbox_dims[2] < 1e-9:
        return True
    aspect_flat = bbox_dims[0] / bbox_dims[2]
    aspect_bbox = features.aspect_ratio  # length / mean(diameter)
    n_int = sum(1 for c in features.cylinders if c.is_internal) if features.cylinders else 0
    n_ext = sum(1 for c in features.cylinders if not c.is_internal) if features.cylinders else 0
    n_cone = features.face_type_counts.get("cone", 0)
    n_torus = features.face_type_counts.get("torus", 0)
    n_sphere = features.face_type_counts.get("sphere", 0)
    n_plane = features.face_type_counts.get("plane", 0)
    has_head_shape = (n_cone + n_torus + n_sphere) >= 1 or (features.head_diameter is not None and features.head_diameter > 0)

    t = fastener_type.lower().replace("-", "_")

    # --- Washers (any flavor) ---
    if t in ("washers", "flat_washer", "spring_washer", "lock_washer"):
        # A washer is a flat disc:
        #   - thin in one dim (aspect_flat <= 0.35)
        #   - NOT elongated (aspect_bbox <= 1.5 — washers are short)
        # The bbox check rules out long screws/bolts that happen to have
        # small-radius shafts (e.g. DIN912_M2_5X12 with bbox 5x5x12 has
        # aspect_bbox = 2.4, far longer than any real washer).
        return aspect_flat <= 0.35 and aspect_bbox <= 1.5

    # --- Pins / dowels (cylindrical, no head) ---
    if t in ("pins", "cylindrical_pin", "dowel_pin"):
        # A pin is a slim cylinder with no head profile. Reject if the
        # part is short-and-flat (disc) OR has head features.
        if aspect_flat <= 0.3 and n_ext <= 2:
            # could be a washer — reject pin
            return False
        if has_head_shape and aspect_bbox < 2.0:
            # short with a head = bolt-class, not a pin
            return False
        return True

    # --- Screws / bolts (external thread + head) ---
    if t in ("screws", "hex_bolt", "socket_head_cap_screw",
             "button_head_screw", "countersunk_socket_screw",
             "wood_screw", "thin_hex_nut"):
        if aspect_flat <= 0.25 and aspect_bbox <= 1.5:
            return False        # pancake = washer
        if n_ext == 0 and n_int >= 1:
            return False        # all-internal = nut-like
        # Long shafts without a head profile are studs / spacers / pins —
        # NOT bolts. A real bolt has a torus/cone/sphere head OR multiple
        # distinct external cylinder clusters (shaft + cylindrical head).
        if aspect_bbox >= 2.0 and not has_head_shape:
            # extra check: two distinct ext cylinder clusters (shaft+head)?
            from .brep_signature import _cluster_radii
            ext_r = [c.radius for c in features.cylinders if not c.is_internal]
            clusters = _cluster_radii(ext_r) if ext_r else []
            has_shaft_head_clusters = (
                len(clusters) >= 2 and clusters[-1][0] >= 1.3 * clusters[0][0]
            )
            if not has_shaft_head_clusters:
                return False    # long tube/stud/spacer, not a bolt
        return True

    # --- Nuts ---
    if t in ("nuts", "hex_nut", "square_nut", "threaded_insert"):
        # Nuts are short and have an internal bore. Reject if very long
        # (aspect > 3) or no internal cylinder at all.
        if aspect_bbox > 3.0:
            return False
        if n_int == 0 and n_ext >= 1:
            # all-external = bolt/pin, not a nut
            return False
        return True

    # --- Threaded rods / studs ---
    if t in ("threaded_rods", "threaded_stud"):
        # Long, slender, no head. Reject if very flat (washer) or has head.
        if aspect_flat > 0.5 and aspect_bbox < 2.0:
            return False
        if has_head_shape and aspect_bbox < 5.0:
            return False
        return True

    # --- Rivets ---
    if t in ("rivets",):
        # Short to medium, with a head, no internal thread.
        if aspect_bbox > 8.0:
            return False
        return True

    # --- Spacers / standoffs ---
    if t in ("spacers",):
        # Tubular: external + internal cylinders of similar length
        if aspect_flat > 0.6 and aspect_bbox < 1.0:
            return False
        return True

    # Set-screws, anchors, keys, nails, retaining-rings: accept by default
    # (less common subtypes — let BF have the call).
    return True


def _ml_with_topk_recovery(ml_label: DetectionLabel, features):
    """If the model's top-1 prediction fails the geometry sanity check,
    look down the top-3 list and pick the highest-confidence candidate
    that DOES pass. Returns a (possibly substituted) DetectionLabel.

    The top-K probabilities are stashed in ml_label.detected_dimensions
    under key "ml_top3" by classify_brep.
    """
    top3 = (ml_label.detected_dimensions or {}).get("ml_top3")
    if not top3 or features is None:
        return ml_label

    top1_type = ml_label.fastener_type.replace("likely_", "").replace("possible_", "")
    if _passes_geometry_check(top1_type, features):
        return ml_label  # top-1 is OK, keep it
    logger.debug("topk_recovery: REJECTED top-1=%s", top1_type)

    # Top-1 disagrees with geometry. Try top-2, top-3.
    for cls, prob in top3[1:]:
        if cls in ("non_fastener", "non-fastener"):
            # If geometry rejects top-1 AND top-2/3 says non_fastener with
            # reasonable confidence, demote to unclassified rather than
            # forcing a guess.
            if prob >= 0.25:
                return DetectionLabel(
                    fastener_type="unclassified", confidence=0.0,
                    method=ml_label.method,
                    detected_dimensions=ml_label.detected_dimensions,
                )
            continue
        if _passes_geometry_check(cls, features):
            return DetectionLabel(
                fastener_type=cls,
                confidence=round(float(prob), 4),
                method=ml_label.method,
                detected_dimensions={
                    **(ml_label.detected_dimensions or {}),
                    "ml_topk_recovered": True,
                    "ml_top1_rejected": top1_type,
                },
            )

    # No candidate passed geometry checks — demote to unclassified
    return DetectionLabel(
        fastener_type="unclassified", confidence=0.0,
        method=ml_label.method,
        detected_dimensions=ml_label.detected_dimensions,
    )


def ensemble_merge(
    rule_label: DetectionLabel,
    ml_label: DetectionLabel,
    features=None,
) -> DetectionLabel:
    """Merge rule-based and ML detection labels.

    Strategy: take the label with higher confidence unless they agree,
    in which case boost confidence.

    IMPORTANT — `rule_label.confidence == 0.0` with `fastener_type=="unclassified"`
    does NOT mean rule-based is "confident the part is nothing". It means
    rule-based couldn't match the part against any known fastener standard.
    BRepFormer trained on the McMaster-Carr fastener catalog only knows 13
    classes and has no "unsure" output, so on out-of-distribution parts
    (gripper fingers, brackets, etc.) it confidently picks the closest
    training match. Without a stricter gate, BF's overconfident guess
    always beats rule-based's "I dunno". This gate fixes that.

    Geometry-aware top-K recovery: before merging, if the model's top-1
    prediction is contradicted by the part's measured BRep geometry
    (e.g. "washers" for a long thin shaft), substitute the highest-
    confidence top-3 candidate that DOES match the geometry.
    """
    # Apply geometry sanity to BF prediction before merging
    if features is not None:
        ml_label = _ml_with_topk_recovery(ml_label, features)

    # Both unclassified
    if rule_label.fastener_type == "unclassified" and ml_label.fastener_type == "unclassified":
        return DetectionLabel(fastener_type="unclassified", confidence=0.0, method="ensemble")

    # Strip prefixes for comparison
    rule_base = rule_label.fastener_type.replace("possible_", "").replace("likely_", "")
    ml_base = ml_label.fastener_type.replace("possible_", "").replace("likely_", "")

    if rule_base == ml_base and rule_base != "unclassified":
        # Agreement — boost confidence
        combined_conf = min(rule_label.confidence + ml_label.confidence * 0.3, 1.0)
        return DetectionLabel(
            fastener_type=rule_base,
            standard=rule_label.standard,
            variant=rule_label.variant,
            confidence=round(combined_conf, 4),
            method="ensemble",
            detected_dimensions=rule_label.detected_dimensions,
            lod_proxy=rule_label.lod_proxy,
        )

    # When rule-based said "unclassified" (which is the common case — it
    # only fires on parts whose dimensions exactly match an ISO/DIN table),
    # we still want to trust BF when its prediction is clearly dominant.
    # The previous gate (require conf > 0.95 AND non_fastener < 0.05) was
    # killing legitimate predictions on real screws where BF typically says
    # "screws: 0.7-0.85, non_fastener: 0.1-0.2". OOD overconfidence is
    # already handled by the per-part size-sanity check in detect.py
    # (parts > 200mm get demoted), so this gate can be much more permissive.
    if rule_label.fastener_type == "unclassified" and ml_base != "non_fastener":
        ml_top3 = (ml_label.detected_dimensions or {}).get("ml_top3") if ml_label.detected_dimensions else None
        nf_prob = 0.0
        for cls, prob in (ml_top3 or []):
            if cls in ("non_fastener", "non-fastener", "non_fasteners"):
                nf_prob = float(prob)
                break
        # Reject ONLY when BF is genuinely confused — its winning class
        # has lower probability than non_fastener. In every other case
        # trust BF's pick (the size-sanity gate further upstream will
        # demote obvious OOD parts).
        if ml_label.confidence < 0.40 or ml_label.confidence < nf_prob:
            return DetectionLabel(
                fastener_type="unclassified", confidence=0.0, method="ensemble",
            )

    # If ML says "non_fastener" but rule-based confidently identified a
    # specific fastener type, KEEP the rule-based label. McMaster-trained BF
    # has no concept of "unsure"; on out-of-distribution shapes it can confidently
    # vote non_fastener even when the part is plainly a bolt (geometry that
    # doesn't fit McMaster's catalogue distribution). Rule-based saying e.g.
    # "hex_bolt with confidence 0.84" is a positive signal we shouldn't discard
    # just because BF returned non_fastener at 0.99 — that's a classic
    # representational disagreement, not a precision disagreement.
    if ml_base in ("non_fastener", "non-fastener", "non_fasteners") and rule_base != "unclassified":
        if rule_label.confidence >= 0.50:
            return DetectionLabel(
                fastener_type=rule_label.fastener_type,
                standard=rule_label.standard,
                variant=rule_label.variant,
                confidence=rule_label.confidence,
                method="ensemble",
                detected_dimensions=rule_label.detected_dimensions,
                lod_proxy=rule_label.lod_proxy,
            )

    # Disagreement — take higher confidence
    if ml_label.confidence > rule_label.confidence:
        winner = ml_label
    else:
        winner = rule_label

    return DetectionLabel(
        fastener_type=winner.fastener_type,
        standard=winner.standard,
        variant=winner.variant,
        confidence=winner.confidence,
        method="ensemble",
        detected_dimensions=winner.detected_dimensions,
        lod_proxy=winner.lod_proxy,
    )
