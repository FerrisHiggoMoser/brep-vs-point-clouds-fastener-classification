# Table (f) — Zero-shot deployment: ISIS satellite + relationships + stress test

## f.1 — ISIS satellite hybrid scoreboard (CSV: [f_satellite_hybrid_results.csv](f_satellite_hybrid_results.csv))

Geometry-only cascade on the ISIS-Space 1U CubeSat, 13-class. GT reconstructed from an explicit
name→class table (`SATELLITE_NAME_GT`); **names are never seen by any prediction stage**. 176
scored parts (92 screws, 72 washers, 8 nuts, 4 spacers). Source: `hybrid_eval/satellite/summary.json`.

| Config | All named (n=176) | ISO/DIN-named only (n=112) | journal-verified 761/IOBC (n=64) |
|---|---:|---:|---:|
| Rules | 65.91% | **100.0%** | 6.3% |
| v6 ML | 35.80% | 47.3% | 15.6% |
| **Hybrid** | **70.45%** | **100.0%** | 18.8% |
| Hybrid+topk | 65.91% | **100.0%** | 6.3% |

Stage-1 claims 116 parts (60 SHCS, 32 hex_bolt, 16 flat_washer, 8 hex_nut) — **all 116 correct.**
On the ISO/DIN subset the cascade is exact (112/112) while raw v6 manages 47.3% (McNemar b=0/c=59,
p=4.3e-14). Honest finding: **raw v6 (35.8%) is worse here than the 2026-05-19 raw v1 (~54%)** —
v2→v6 diversification did not transfer to M1–M3 micro-hardware. The top-K filter backfires on the
satellite (lands 56/60 Stage-2 rings on "keys", a default-accept class).

## f.2 — Relationship + zero-shot counts (CSV: [f_satellite_relationship_counts.csv](f_satellite_relationship_counts.csv))

| Metric | Value | Source |
|---|---:|---|
| screwedInto arcs (ISIS) | **161** | 2026-05-19 `_infer_fastener_relationships` |
| contained_in arcs (ISIS) | **592** | 2026-05-19 `_infer_housing_relationships` |
| classified fasteners (ISIS) | 244 | 2026-05-19 `eval_satellite.py` |
| fastener_labeling/files contained_in | 522 | 2026-05-19 `eval_detection.py` |
| fastener_labeling/files screwedInto | 69 | 2026-05-19 `eval_detection.py` |
| Day-4 PN++ zero-shot satellite F1 | 0.524 | 2026-04-22 Stage 2 isispace (551 GT parts) |
| Day-4 PN++ satellite precision / recall | 0.717 / 0.413 | 2026-04-22 |

Relationships are computed by **geometric algorithms on the OCC kernel** (axis/radial/diameter
gates), not by a learned model. Fit-class spread on `fastener_labeling/files/`: 36% tap / 35% slip
/ 29% clearance. The Day-4 satellite F1 (0.524) uses the binary Stage-2 PN++ model and a
name-keyword ground truth over 551 parts — a *different* GT and checkpoint from the f.1 cascade,
so the numbers are not directly comparable.

## f.3 — 12-scenario stress-test scoreboard (CSV: [f_stress_test_scoreboard.csv](f_stress_test_scoreboard.csv))

Synthetic STEPs through the full `detect_fasteners` pipeline (2026-05-19). All 12 pass.

| # | Scenario | Parts | Matched | Detect (s) |
|---|---|---:|---|---:|
| 1 | Scale 100-bolt grid | 101 | 100/100 | 0.17 |
| 2 | Angled bolts 0–8° | 17 | 16/16 | 0.02 |
| 3 | Deep stack (5 plates) | 6 | 5 arcs order 0–4 | 0.01 |
| 4 | Adversarial 12+6 decoys | 19 | 4/6 decoys rejected | 0.02 |
| 5 | Mega 500-bolt grid | 501 | 500/500 | 2.05 |
| 6 | Extreme tilt 0–30° | 13 | 7/12 (cos>0.95 boundary) | 0.01 |
| 7 | Bolt-nut sandwich | 25 | 24 arcs | 0.02 |
| 8 | Orthogonal axes | 8 | 7/7 | 0.01 |
| 9 | Real SHCS internal socket | 7 | 6/6 (self-loop guard) | 0.01 |
| 10 | M1.6/M2 micro-fasteners | 9 | 8/8 | 0.01 |
| 11 | Curved host hub | 5 | 4/4 | 0.01 |
| 12 | Huge 1000-bolt grid | 1001 | 1000/1000 | 7.32 |

Matcher scales linearly to 500 bolts; tilt gate behaves exactly to the degree (cos 18°=0.951 in,
cos 19°=0.946 out). Source: journal 2026-05-19, `backend/scripts/stress_test.py` →
`stress_results.json` (per-scenario verification; full JSON not separately consolidated here).
