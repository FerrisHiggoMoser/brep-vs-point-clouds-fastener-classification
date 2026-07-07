# thesis_results/ — consolidated results package

BSc thesis (VU Amsterdam, Ferris Moser): **"BRepFormer vs. PointNet++ for Fastener Classification
in Satellite CAD."** Single consumable package for a thesis-writing pass. Every number is
re-derived from primary artifacts (`full_analysis*.json`, `eval_summary.json`, `test_eval.json`,
`summary.json`, `relabel_manifest*.json`) and cross-checked against RESEARCH_JOURNAL.md; where they
disagreed, **the artifact wins** and the mismatch is logged in [discrepancies.md](discrepancies.md).
No new training or evaluation was run.

## Two comparison families (keep separate — different training distributions)
- **Family I — McMaster matched comparison:** binary + 13-class subtype on the shared 558-sample
  test set; matched data/compute/hyperparameters.
- **Family II — v6 cross-CAD-tool generalization:** BF v2→v6 data-diversification study + PN++
  trained on the same v6 data; held out on PADDLE STEAMER and the ISIS satellite.
- The **rule-based baseline** appears in both (no training data).
- ⚠️ Family-I PN++ point clouds historically had **degenerate normals** (bug); Family-II PN++ has
  real normals. **Never pool numbers across families.**

## Headline results (corrected)

| Task | Family | PN++ | BF | Winner | p |
|---|---|---:|---:|---|---:|
| Binary (n=558) | I | 94.27% | 89.96% | PN++ +4.30pp | 0.006 |
| 13-class subtype (n=558, **corrected**) | I | 93.01% | 89.96% | tie (PN++ nominal) | 0.058 |
| 13-class v6 in-dist (n=5,827) | II | 95.33% | 88.78% | PN++ +6.55pp | 6.4e-58 |
| 13-class v6 PADDLE (n=350) | II | 84.86% | 64.00% | PN++ +20.9pp | 2.7e-13 |
| Rules floor (binary, n=558) | both | — | — | 65.05% (training-free) | bounds <1e-15 |

> ⚠️ **Retraction.** The 2026-05-11 "BRepFormer wins subtype +14.34pp" result was a PointNet++
> normals-extraction bug; corrected, the architectures are statistically **tied** on McMaster
> subtype13. The corrected numbers above are the thesis numbers. See [claims.md](claims.md) C2/C7.

## Contents
- **[tables/](tables/)** — clean Markdown + CSV for tables (a)–(h). Index: [tables/README.md](tables/README.md).
- **[figures/](figures/)** — confusion matrices (regenerated from JSON/npy), training error renders,
  reference images. Index: [figures/README.md](figures/README.md).
- **[claims.md](claims.md)** — every defensible thesis claim, in order, with backing artifacts, n,
  p-values/CIs, caveats, and retraction status.
- **[timeline.md](timeline.md)** — chronological experiment log.
- **[provenance.md](provenance.md)** — checkpoint inventory + number→script mapping.
- **[discrepancies.md](discrepancies.md)** — every journal-vs-artifact mismatch (artifact wins).
- **[MISSING_FIGURES.md](MISSING_FIGURES.md)** — figures the thesis still needs that can't be made
  from existing artifacts.
- **[_generate.py](_generate.py)** — regenerates all CSVs + confusion-matrix PNGs from the artifacts.

## What is NOT here (out of scope / not run)
- Exp 6 data-efficiency curves and Exp 7 attention-bias ablation (WORKFLOW Phase 6) — never run.
- Per-face segmentation head-to-head (B-rep's home turf) — pretrain exists, PN++ seg model does not
  (future work).
- Re-derivation of v2–v5 BF McMaster numbers (only v1 & v6 checkpoints were re-evaluated; v2–v5 are
  journal-prose-only — flagged in discrepancies.md D9).
