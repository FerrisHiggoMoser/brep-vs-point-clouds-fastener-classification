# Table (g) — Dataset composition + all train/val/test split sizes

## g.1 — v6 training-set composition (CSV: [g_v6_training_composition.csv](g_v6_training_composition.csv))

~66,941 samples across train/val/test; real-CAD ≈ 39% (Fusion + GrabCAD; vs v1's 0%, v4's 12%). Source: journal
2026-05-21→05-25.

| Source | Approx count | Modeling DNA |
|---|---:|---|
| McMaster (raw) | 3,059 | mcmaster.com CAD export |
| synth_no_threads | 9,360 | OCC primitives, no threads |
| synth_remixed | 28,669 | McMaster + scale/stretch/shear/recess-imprint |
| Fusion 360 Assembly | 4,443 | 755 unique Fusion designers, real threads |
| GrabCAD round #1 | 2,629 | hand-curated GrabCAD downloads |
| GrabCAD round #2 | 18,781 | broader GrabCAD via dump-folder pipeline |
| **TOTAL (approx)** | **66,941** | ~39% real-CAD (Fusion + GrabCAD) |

Other datasets used upstream: **MCB-A** (Kim et al. ECCV 2020, 58,696 OBJ, 68 classes — PN++
pretrain); **Fusion 360 Gallery s2.0.0** (35,680 STEP, 8 per-face seg classes — BF pretrain);
**GrabCAD binary** (164 fastener + 399 non-fastener STEPs, Day-1/3 transfer-learning study);
**ABC dataset** chunk 0 (10k generic STEP negatives).

## g.2 — All split sizes (CSV: [g_split_sizes.csv](g_split_sizes.csv))

| Dataset | train | val | test |
|---|---|---|---|
| MCB-A (PN++ pretrain) | 39,822 | 7,157 | 11,716 |
| GrabCAD binary | 1,837 (161 f / 1,676 n) | 393 (34 f / 359 n) | 395 (35 f / 360 n) |
| Fusion 360 Gallery (BF pretrain) | — (35,680 total, 8 seg classes) | — | — |
| McMaster binary PN++ baseline | 5,201 (2,368 f / 2,833 n) | 669 (291 f / 378 n) | 649 (301 f / 348 n) |
| McMaster BRep (MAX_FACES=600) | 4,560 (2,283 f / 2,277 n) | 591 (278 f / 313 n) | 559 (290 f / 269 n) |
| **McMaster matched (shared) — Family I binary/subtype13** | 5,710 | — | **558** (289 f / 269 n) |
| McMaster subtype13 | 4,557 | 590 | 558 |
| v6 BF (max_faces=300) | 58,188 | 7,302 | 7,286 |
| v6 PN++ (retained ~81%) | 46,948 | 5,904 | 5,850 |

Notes:
- McMaster matched binary test n=558: the shared intersection (`full_analysis.json`), one part
  (7513K111) dual-filed and de-duplicated; the rule-based eval scored the full 559 then reports
  on 558. The subtype13 `relabel_manifest` records test non_fastener=270 (the dual-file counted),
  which becomes 269 at scoring time (see [../discrepancies.md](../discrepancies.md)).
- v6 PN++ retention is per-source in `pn_v6_run/eval_summary.json` → `retention`: grabcad drops to
  ~33% (deleted source STEPs), all other sources ~100%; this is a *data disadvantage for PN++*.
- McMaster subtype13 per-class train/val/test counts: `relabel_manifest.subtype13.json`.
