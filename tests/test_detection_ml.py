"""Tests for ML classifier wrapper and ensemble merging."""

import pytest

from step_vr_step.schema import DetectionLabel
from step_vr_step.detection.ml_classifier import is_ml_available, ensemble_merge


class TestMLAvailability:
    def test_is_ml_available_returns_bool(self):
        result = is_ml_available()
        assert isinstance(result, bool)


class TestEnsembleMerge:
    def test_both_unclassified(self):
        rule = DetectionLabel(fastener_type="unclassified", confidence=0.0, method="rule_based")
        ml = DetectionLabel(fastener_type="unclassified", confidence=0.0, method="ml_pointnet2")
        result = ensemble_merge(rule, ml)
        assert result.fastener_type == "unclassified"
        assert result.method == "ensemble"

    def test_agreement_boosts_confidence(self):
        rule = DetectionLabel(
            fastener_type="hex_bolt", confidence=0.70, method="rule_based",
            standard="ISO 4014", variant="M6",
        )
        ml = DetectionLabel(
            fastener_type="hex_bolt", confidence=0.80, method="ml_pointnet2",
        )
        result = ensemble_merge(rule, ml)
        assert result.fastener_type == "hex_bolt"
        assert result.confidence > 0.70  # boosted
        assert result.method == "ensemble"
        assert result.standard == "ISO 4014"

    def test_disagreement_takes_higher_confidence(self):
        rule = DetectionLabel(
            fastener_type="hex_bolt", confidence=0.60, method="rule_based",
        )
        ml = DetectionLabel(
            fastener_type="socket_head_cap_screw", confidence=0.90, method="ml_brepformer",
        )
        result = ensemble_merge(rule, ml)
        assert result.fastener_type == "socket_head_cap_screw"
        assert result.confidence == 0.90

    def test_possible_prefix_stripped_for_comparison(self):
        rule = DetectionLabel(
            fastener_type="possible_hex_bolt", confidence=0.65, method="rule_based",
        )
        ml = DetectionLabel(
            fastener_type="likely_hex_bolt", confidence=0.75, method="ml_pointnet2",
        )
        result = ensemble_merge(rule, ml)
        assert result.fastener_type == "hex_bolt"  # agreement on base type
        assert result.confidence > 0.65

    def test_one_unclassified_takes_other(self):
        rule = DetectionLabel(
            fastener_type="hex_nut", confidence=0.85, method="rule_based",
        )
        ml = DetectionLabel(
            fastener_type="unclassified", confidence=0.0, method="ml_pointnet2",
        )
        result = ensemble_merge(rule, ml)
        assert result.fastener_type == "hex_nut"
