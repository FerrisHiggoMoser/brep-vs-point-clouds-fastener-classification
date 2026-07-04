"""ISO fastener dimension tables for rule-based detection.

All dimensions in millimetres. Sources:
  ISO 4014 / 4017  — Hex head bolts (partial / full thread)
  ISO 4762         — Socket head cap screws
  ISO 4032 / 4035  — Hex nuts (style 1) / thin hex nuts
  ISO 7089 / 7090  — Plain washers / chamfered washers
  ISO 7380         — Button head screws
  ISO 10642        — Countersunk socket head cap screws
"""

# ---------------------------------------------------------------------------
# Hex bolts  (ISO 4014 partial-thread / ISO 4017 full-thread)
# ---------------------------------------------------------------------------
ISO_4014_HEX_BOLTS = {
    "M3":  {"shaft_diameter": 3.0,  "head_width": 5.5,  "head_height": 2.0,  "thread_pitch": 0.5},
    "M4":  {"shaft_diameter": 4.0,  "head_width": 7.0,  "head_height": 2.8,  "thread_pitch": 0.7},
    "M5":  {"shaft_diameter": 5.0,  "head_width": 8.0,  "head_height": 3.5,  "thread_pitch": 0.8},
    "M6":  {"shaft_diameter": 6.0,  "head_width": 10.0, "head_height": 4.0,  "thread_pitch": 1.0},
    "M8":  {"shaft_diameter": 8.0,  "head_width": 13.0, "head_height": 5.3,  "thread_pitch": 1.25},
    "M10": {"shaft_diameter": 10.0, "head_width": 16.0, "head_height": 6.4,  "thread_pitch": 1.5},
    "M12": {"shaft_diameter": 12.0, "head_width": 18.0, "head_height": 7.5,  "thread_pitch": 1.75},
    "M16": {"shaft_diameter": 16.0, "head_width": 24.0, "head_height": 10.0, "thread_pitch": 2.0},
    "M20": {"shaft_diameter": 20.0, "head_width": 30.0, "head_height": 12.5, "thread_pitch": 2.5},
    "M24": {"shaft_diameter": 24.0, "head_width": 36.0, "head_height": 15.0, "thread_pitch": 3.0},
}
# ISO 4017 shares the same dimensions (difference is thread length = full shank)
ISO_4017_HEX_BOLTS = ISO_4014_HEX_BOLTS

# ---------------------------------------------------------------------------
# Socket head cap screws  (ISO 4762)
# ---------------------------------------------------------------------------
ISO_4762_SOCKET_HEAD_CAP_SCREWS = {
    "M3":  {"shaft_diameter": 3.0,  "head_diameter": 5.5,  "head_height": 3.0,  "socket_size": 2.5,  "thread_pitch": 0.5},
    "M4":  {"shaft_diameter": 4.0,  "head_diameter": 7.0,  "head_height": 4.0,  "socket_size": 3.0,  "thread_pitch": 0.7},
    "M5":  {"shaft_diameter": 5.0,  "head_diameter": 8.5,  "head_height": 5.0,  "socket_size": 4.0,  "thread_pitch": 0.8},
    "M6":  {"shaft_diameter": 6.0,  "head_diameter": 10.0, "head_height": 6.0,  "socket_size": 5.0,  "thread_pitch": 1.0},
    "M8":  {"shaft_diameter": 8.0,  "head_diameter": 13.0, "head_height": 8.0,  "socket_size": 6.0,  "thread_pitch": 1.25},
    "M10": {"shaft_diameter": 10.0, "head_diameter": 16.0, "head_height": 10.0, "socket_size": 8.0,  "thread_pitch": 1.5},
    "M12": {"shaft_diameter": 12.0, "head_diameter": 18.0, "head_height": 12.0, "socket_size": 10.0, "thread_pitch": 1.75},
    "M16": {"shaft_diameter": 16.0, "head_diameter": 24.0, "head_height": 16.0, "socket_size": 14.0, "thread_pitch": 2.0},
    "M20": {"shaft_diameter": 20.0, "head_diameter": 30.0, "head_height": 20.0, "socket_size": 17.0, "thread_pitch": 2.5},
    "M24": {"shaft_diameter": 24.0, "head_diameter": 36.0, "head_height": 24.0, "socket_size": 19.0, "thread_pitch": 3.0},
}

# ---------------------------------------------------------------------------
# Hex nuts  (ISO 4032 style 1)
# ---------------------------------------------------------------------------
ISO_4032_HEX_NUTS = {
    "M3":  {"bore_diameter": 3.0,  "width_across_flats": 5.5,  "height": 2.4,  "thread_pitch": 0.5},
    "M4":  {"bore_diameter": 4.0,  "width_across_flats": 7.0,  "height": 3.2,  "thread_pitch": 0.7},
    "M5":  {"bore_diameter": 5.0,  "width_across_flats": 8.0,  "height": 4.7,  "thread_pitch": 0.8},
    "M6":  {"bore_diameter": 6.0,  "width_across_flats": 10.0, "height": 5.2,  "thread_pitch": 1.0},
    "M8":  {"bore_diameter": 8.0,  "width_across_flats": 13.0, "height": 6.8,  "thread_pitch": 1.25},
    "M10": {"bore_diameter": 10.0, "width_across_flats": 16.0, "height": 8.4,  "thread_pitch": 1.5},
    "M12": {"bore_diameter": 12.0, "width_across_flats": 18.0, "height": 10.8, "thread_pitch": 1.75},
    "M16": {"bore_diameter": 16.0, "width_across_flats": 24.0, "height": 14.8, "thread_pitch": 2.0},
    "M20": {"bore_diameter": 20.0, "width_across_flats": 30.0, "height": 18.0, "thread_pitch": 2.5},
    "M24": {"bore_diameter": 24.0, "width_across_flats": 36.0, "height": 21.5, "thread_pitch": 3.0},
}

# ---------------------------------------------------------------------------
# Thin hex nuts  (ISO 4035)
# ---------------------------------------------------------------------------
ISO_4035_THIN_HEX_NUTS = {
    "M3":  {"bore_diameter": 3.0,  "width_across_flats": 5.5,  "height": 1.8,  "thread_pitch": 0.5},
    "M4":  {"bore_diameter": 4.0,  "width_across_flats": 7.0,  "height": 2.2,  "thread_pitch": 0.7},
    "M5":  {"bore_diameter": 5.0,  "width_across_flats": 8.0,  "height": 2.7,  "thread_pitch": 0.8},
    "M6":  {"bore_diameter": 6.0,  "width_across_flats": 10.0, "height": 3.2,  "thread_pitch": 1.0},
    "M8":  {"bore_diameter": 8.0,  "width_across_flats": 13.0, "height": 4.0,  "thread_pitch": 1.25},
    "M10": {"bore_diameter": 10.0, "width_across_flats": 16.0, "height": 5.0,  "thread_pitch": 1.5},
    "M12": {"bore_diameter": 12.0, "width_across_flats": 18.0, "height": 6.0,  "thread_pitch": 1.75},
    "M16": {"bore_diameter": 16.0, "width_across_flats": 24.0, "height": 8.0,  "thread_pitch": 2.0},
    "M20": {"bore_diameter": 20.0, "width_across_flats": 30.0, "height": 10.0, "thread_pitch": 2.5},
    "M24": {"bore_diameter": 24.0, "width_across_flats": 36.0, "height": 12.0, "thread_pitch": 3.0},
}

# ---------------------------------------------------------------------------
# Plain washers  (ISO 7089)
# ---------------------------------------------------------------------------
ISO_7089_PLAIN_WASHERS = {
    "M3":  {"bore_diameter": 3.4,  "outer_diameter": 7.0,   "thickness": 0.5},
    "M4":  {"bore_diameter": 4.5,  "outer_diameter": 9.0,   "thickness": 0.8},
    "M5":  {"bore_diameter": 5.5,  "outer_diameter": 10.0,  "thickness": 1.0},
    "M6":  {"bore_diameter": 6.6,  "outer_diameter": 12.0,  "thickness": 1.6},
    "M8":  {"bore_diameter": 8.4,  "outer_diameter": 16.0,  "thickness": 1.6},
    "M10": {"bore_diameter": 10.5, "outer_diameter": 20.0,  "thickness": 2.0},
    "M12": {"bore_diameter": 13.0, "outer_diameter": 24.0,  "thickness": 2.5},
    "M16": {"bore_diameter": 17.0, "outer_diameter": 30.0,  "thickness": 3.0},
    "M20": {"bore_diameter": 21.0, "outer_diameter": 37.0,  "thickness": 3.0},
    "M24": {"bore_diameter": 25.0, "outer_diameter": 44.0,  "thickness": 4.0},
}

# ISO 7090 chamfered washers share the same dimensions
ISO_7090_CHAMFERED_WASHERS = ISO_7089_PLAIN_WASHERS

# ---------------------------------------------------------------------------
# Button head screws  (ISO 7380)
# ---------------------------------------------------------------------------
ISO_7380_BUTTON_HEAD_SCREWS = {
    "M3":  {"shaft_diameter": 3.0,  "head_diameter": 5.7,  "head_height": 1.65, "socket_size": 2.0,  "thread_pitch": 0.5},
    "M4":  {"shaft_diameter": 4.0,  "head_diameter": 7.6,  "head_height": 2.20, "socket_size": 2.5,  "thread_pitch": 0.7},
    "M5":  {"shaft_diameter": 5.0,  "head_diameter": 9.5,  "head_height": 2.75, "socket_size": 3.0,  "thread_pitch": 0.8},
    "M6":  {"shaft_diameter": 6.0,  "head_diameter": 10.5, "head_height": 3.30, "socket_size": 4.0,  "thread_pitch": 1.0},
    "M8":  {"shaft_diameter": 8.0,  "head_diameter": 14.0, "head_height": 4.40, "socket_size": 5.0,  "thread_pitch": 1.25},
    "M10": {"shaft_diameter": 10.0, "head_diameter": 17.5, "head_height": 5.50, "socket_size": 6.0,  "thread_pitch": 1.5},
    "M12": {"shaft_diameter": 12.0, "head_diameter": 21.0, "head_height": 6.60, "socket_size": 8.0,  "thread_pitch": 1.75},
    "M16": {"shaft_diameter": 16.0, "head_diameter": 28.0, "head_height": 8.80, "socket_size": 10.0, "thread_pitch": 2.0},
}

# ---------------------------------------------------------------------------
# Countersunk socket head cap screws  (ISO 10642)
# ---------------------------------------------------------------------------
ISO_10642_COUNTERSUNK_SCREWS = {
    "M3":  {"shaft_diameter": 3.0,  "head_diameter": 6.72,  "head_height": 1.86, "socket_size": 2.0,  "thread_pitch": 0.5},
    "M4":  {"shaft_diameter": 4.0,  "head_diameter": 8.96,  "head_height": 2.48, "socket_size": 2.5,  "thread_pitch": 0.7},
    "M5":  {"shaft_diameter": 5.0,  "head_diameter": 11.20, "head_height": 3.10, "socket_size": 3.0,  "thread_pitch": 0.8},
    "M6":  {"shaft_diameter": 6.0,  "head_diameter": 13.44, "head_height": 3.72, "socket_size": 4.0,  "thread_pitch": 1.0},
    "M8":  {"shaft_diameter": 8.0,  "head_diameter": 17.92, "head_height": 4.96, "socket_size": 5.0,  "thread_pitch": 1.25},
    "M10": {"shaft_diameter": 10.0, "head_diameter": 22.40, "head_height": 6.20, "socket_size": 6.0,  "thread_pitch": 1.5},
    "M12": {"shaft_diameter": 12.0, "head_diameter": 26.88, "head_height": 7.44, "socket_size": 8.0,  "thread_pitch": 1.75},
    "M16": {"shaft_diameter": 16.0, "head_diameter": 33.60, "head_height": 8.80, "socket_size": 10.0, "thread_pitch": 2.0},
    "M20": {"shaft_diameter": 20.0, "head_diameter": 40.32, "head_height": 10.16, "socket_size": 12.0, "thread_pitch": 2.5},
}

# ---------------------------------------------------------------------------
# Unified lookup
# ---------------------------------------------------------------------------
ALL_STANDARDS: dict[str, dict] = {
    "ISO 4014": {"type": "hex_bolt",                    "table": ISO_4014_HEX_BOLTS},
    "ISO 4017": {"type": "hex_bolt",                    "table": ISO_4017_HEX_BOLTS},
    "ISO 4762": {"type": "socket_head_cap_screw",       "table": ISO_4762_SOCKET_HEAD_CAP_SCREWS},
    "ISO 4032": {"type": "hex_nut",                     "table": ISO_4032_HEX_NUTS},
    "ISO 4035": {"type": "thin_hex_nut",                "table": ISO_4035_THIN_HEX_NUTS},
    "ISO 7089": {"type": "flat_washer",                 "table": ISO_7089_PLAIN_WASHERS},
    "ISO 7090": {"type": "chamfered_washer",            "table": ISO_7090_CHAMFERED_WASHERS},
    "ISO 7380": {"type": "button_head_screw",           "table": ISO_7380_BUTTON_HEAD_SCREWS},
    "ISO 10642": {"type": "countersunk_socket_screw",   "table": ISO_10642_COUNTERSUNK_SCREWS},
}

# Fastener type → category mapping for feature-based pre-filtering
BOLT_TYPES = {"hex_bolt", "socket_head_cap_screw", "button_head_screw", "countersunk_socket_screw"}
NUT_TYPES = {"hex_nut", "thin_hex_nut"}
WASHER_TYPES = {"flat_washer", "chamfered_washer"}
