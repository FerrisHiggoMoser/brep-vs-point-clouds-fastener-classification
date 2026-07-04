"""Tests for ISO fastener dimension tables."""

from step_vr_step.detection.iso_tables import (
    ISO_4014_HEX_BOLTS,
    ISO_4762_SOCKET_HEAD_CAP_SCREWS,
    ISO_4032_HEX_NUTS,
    ISO_4035_THIN_HEX_NUTS,
    ISO_7089_PLAIN_WASHERS,
    ISO_7380_BUTTON_HEAD_SCREWS,
    ISO_10642_COUNTERSUNK_SCREWS,
    ALL_STANDARDS,
)


class TestISOTables:
    def test_hex_bolt_sizes(self):
        """All M3–M24 entries present with required fields."""
        expected_sizes = ["M3", "M4", "M5", "M6", "M8", "M10", "M12", "M16", "M20", "M24"]
        for size in expected_sizes:
            assert size in ISO_4014_HEX_BOLTS, f"Missing {size}"
            entry = ISO_4014_HEX_BOLTS[size]
            assert "shaft_diameter" in entry
            assert "head_width" in entry
            assert "head_height" in entry
            assert "thread_pitch" in entry

    def test_shaft_diameter_matches_size(self):
        """M6 should have shaft_diameter=6.0, etc."""
        for size, entry in ISO_4014_HEX_BOLTS.items():
            expected = float(size[1:])
            assert entry["shaft_diameter"] == expected

    def test_head_width_larger_than_shaft(self):
        """Head width should always be larger than shaft diameter."""
        for size, entry in ISO_4014_HEX_BOLTS.items():
            assert entry["head_width"] > entry["shaft_diameter"]

    def test_socket_head_cap_screws(self):
        assert "M6" in ISO_4762_SOCKET_HEAD_CAP_SCREWS
        m6 = ISO_4762_SOCKET_HEAD_CAP_SCREWS["M6"]
        assert m6["shaft_diameter"] == 6.0
        assert m6["head_diameter"] == 10.0
        assert "socket_size" in m6

    def test_hex_nuts(self):
        assert "M8" in ISO_4032_HEX_NUTS
        m8 = ISO_4032_HEX_NUTS["M8"]
        assert m8["bore_diameter"] == 8.0
        assert m8["width_across_flats"] == 13.0

    def test_thin_nuts_thinner_than_regular(self):
        for size in ISO_4035_THIN_HEX_NUTS:
            if size in ISO_4032_HEX_NUTS:
                assert ISO_4035_THIN_HEX_NUTS[size]["height"] < ISO_4032_HEX_NUTS[size]["height"]

    def test_washers(self):
        assert "M10" in ISO_7089_PLAIN_WASHERS
        m10 = ISO_7089_PLAIN_WASHERS["M10"]
        assert m10["bore_diameter"] > 10.0  # bore is slightly larger than bolt
        assert m10["outer_diameter"] > m10["bore_diameter"]

    def test_all_standards_unified(self):
        """ALL_STANDARDS maps every ISO standard to its table."""
        assert "ISO 4014" in ALL_STANDARDS
        assert "ISO 4762" in ALL_STANDARDS
        assert "ISO 4032" in ALL_STANDARDS
        assert "ISO 7089" in ALL_STANDARDS
        for name, info in ALL_STANDARDS.items():
            assert "type" in info
            assert "table" in info
            assert len(info["table"]) > 0

    def test_countersunk_screws(self):
        assert "M8" in ISO_10642_COUNTERSUNK_SCREWS
        m8 = ISO_10642_COUNTERSUNK_SCREWS["M8"]
        assert m8["head_diameter"] > m8["shaft_diameter"]

    def test_button_head_screws(self):
        assert "M6" in ISO_7380_BUTTON_HEAD_SCREWS
        m6 = ISO_7380_BUTTON_HEAD_SCREWS["M6"]
        assert m6["head_height"] < m6["head_diameter"]
