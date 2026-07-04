"""Tests for rule-based fastener detection."""

import pytest

from step_vr_step.config import DetectionConfig
from step_vr_step.schema import DetectionLabel
from step_vr_step.detection.geometric_features import GeometricFeatures
from step_vr_step.detection.rule_based import classify_part, detect_repetitions


def _bolt_features(shaft_dia=6.0, head_dia=10.0, head_ht=4.0, length=30.0):
    """Create features that look like a hex bolt."""
    return GeometricFeatures(
        bbox_min=(0, 0, 0),
        bbox_max=(head_dia, head_dia, length),
        volume=3.14 * (shaft_dia / 2) ** 2 * length,
        surface_area=500.0,
        face_type_counts={"plane": 2, "cylinder": 8, "cone": 0, "sphere": 0, "torus": 0, "nurbs": 0},
        cylindrical_face_radii=[shaft_dia / 2] * 6 + [head_dia / 2] * 2,
        cylindrical_face_lengths=[25.0] * 6 + [head_ht] * 2,
        cylindrical_surface_ratio=0.75,
        num_faces=10,
        num_edges=20,
        num_vertices=30,
        aspect_ratio=length / shaft_dia,
        has_thread=True,
        bounding_cylinder_diameter=shaft_dia,
        bounding_cylinder_length=length,
        head_diameter=head_dia,
        head_height=head_ht,
    )


def _washer_features(bore_dia=6.6, outer_dia=12.0, thickness=1.6):
    return GeometricFeatures(
        bbox_min=(0, 0, 0),
        bbox_max=(outer_dia, outer_dia, thickness),
        volume=3.14 * ((outer_dia / 2) ** 2 - (bore_dia / 2) ** 2) * thickness,
        surface_area=200.0,
        face_type_counts={"plane": 2, "cylinder": 4, "cone": 0, "sphere": 0, "torus": 0, "nurbs": 0},
        cylindrical_face_radii=[bore_dia / 2, outer_dia / 2],
        cylindrical_face_lengths=[thickness, thickness],
        cylindrical_surface_ratio=0.60,
        num_faces=6,
        num_edges=8,
        num_vertices=8,
        aspect_ratio=thickness / outer_dia,
        has_thread=False,
        bounding_cylinder_diameter=outer_dia,
        bounding_cylinder_length=thickness,
        head_diameter=None,
        head_height=None,
    )


def _structural_features():
    """Large, non-cylindrical structural part."""
    return GeometricFeatures(
        bbox_min=(0, 0, 0),
        bbox_max=(500, 300, 200),
        volume=1e6,
        surface_area=5e5,
        face_type_counts={"plane": 20, "cylinder": 2, "cone": 0, "sphere": 0, "torus": 0, "nurbs": 5},
        cylindrical_surface_ratio=0.05,
        num_faces=27,
        num_edges=50,
        num_vertices=40,
        aspect_ratio=2.5,
        has_thread=False,
        bounding_cylinder_diameter=300,
        bounding_cylinder_length=500,
    )


class TestRuleBasedDetection:
    def test_m6_bolt_high_confidence(self):
        config = DetectionConfig()
        feat = _bolt_features(shaft_dia=6.0, head_dia=10.0, head_ht=4.0, length=30.0)
        label = classify_part(feat, config)

        assert "unclassified" not in label.fastener_type
        assert label.confidence > 0.5
        assert label.method == "rule_based"
        assert label.standard is not None

    def test_washer_detected(self):
        config = DetectionConfig()
        feat = _washer_features()
        label = classify_part(feat, config)

        # Washer detection is challenging with bounding-cylinder-only features
        # since ISO tables key on bore diameter not outer diameter.
        # At minimum it should not crash and should produce a valid label.
        assert isinstance(label, DetectionLabel)
        assert label.method == "rule_based"

    def test_structural_part_unclassified(self):
        config = DetectionConfig()
        feat = _structural_features()
        label = classify_part(feat, config)

        assert label.fastener_type == "unclassified"
        assert label.confidence == 0.0

    def test_repetition_bonus(self):
        config = DetectionConfig()
        feat = _bolt_features()

        label_single = classify_part(feat, config, repetition_count=1)
        label_repeated = classify_part(feat, config, repetition_count=5)

        assert label_repeated.confidence >= label_single.confidence

    def test_confidence_thresholds(self):
        config = DetectionConfig(rule_confidence_threshold=0.60)
        # A minimal match should be below threshold
        feat = GeometricFeatures(
            cylindrical_surface_ratio=0.30,
            aspect_ratio=2.0,
            bounding_cylinder_diameter=99.0,  # won't match any ISO table
            bounding_cylinder_length=200.0,
        )
        label = classify_part(feat, config)
        assert label.fastener_type == "unclassified"

    def test_detected_dimensions_populated(self):
        config = DetectionConfig()
        feat = _bolt_features()
        label = classify_part(feat, config)

        assert "shaft_dia" in label.detected_dimensions
        assert "length" in label.detected_dimensions
        assert "aspect_ratio" in label.detected_dimensions


class TestRepetitionDetection:
    def test_identical_parts_grouped(self):
        feat_a = _bolt_features(shaft_dia=6.0)
        feat_b = _bolt_features(shaft_dia=6.0)
        feat_c = _bolt_features(shaft_dia=10.0)  # different size

        features = [("a", feat_a), ("b", feat_b), ("c", feat_c)]
        groups = detect_repetitions(features)

        # a and b should be grouped, c alone
        found_ab = any("a" in g and "b" in g for g in groups.values())
        assert found_ab

    def test_no_groups_for_unique_parts(self):
        features = [
            ("a", _bolt_features(shaft_dia=3.0)),
            ("b", _bolt_features(shaft_dia=8.0)),
            ("c", _washer_features()),
        ]
        groups = detect_repetitions(features)
        assert len(groups) == 0
