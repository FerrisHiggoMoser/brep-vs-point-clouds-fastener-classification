# Table (b) — Family I: 13-class fastener-subtype matched comparison

**Comparison family I.** Same 558-sample matched McMaster test set, 13 classes (12 fastener
subtypes + `non_fastener`). Matched data/compute/hyperparameters.

> ⚠️ **The headline result was corrected on 2026-06-13.** The original 2026-05-11 run trained
> PointNet++ on point clouds with **degenerate constant (0,0,1) normals** (a `pipeline.py`
> extraction bug). Re-extracting with real per-face normals and an otherwise byte-identical
> protocol — **BRepFormer's frozen checkpoint reproduces 89.96% exactly** — lifts PN++ from
> 75.63% to 93.01%. The "BF +14.34pp" win becomes a statistical tie. The corrected numbers are
> the thesis numbers; the flawed numbers are kept only as a documented retraction.

## Headline (CSV: [b_familyI_subtype13_headline.csv](b_familyI_subtype13_headline.csv))

| Model | n | Accuracy | Macro F1 | Weighted F1 | Bal. acc | MCC | Top-3 | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PointNet++ (CORRECTED real normals)** | 558 | **93.01%** | 89.65% | 93.50% | 93.37% | 0.909 | 99.64% | 0.034 | 0.107 |
| BRepFormer subtype13 (frozen) | 558 | **89.96%** | 86.14% | 90.21% | 88.95% | 0.865 | 98.92% | 0.038 | 0.143 |
| PointNet++ (FLAWED — RETRACTED) | 558 | 75.63% | 66.58% | 77.24% | 78.25% | 0.694 | 96.77% | 0.066 | 0.345 |
| Rules (frozen) | 558 | 49.82%* | 14.13% | — | — | — | — | — | — |

\* Rules subtype13 on the 558 matched set (per `rule_based_eval/matched558/summary.json`); 6 of
12 fastener classes are structural zeros (no signature exists). The journal's headline rules
subtype number is 52.62% on the full **649**-file set (table below / [b CSV](b_familyI_subtype13_headline.csv)).

McNemar (corrected, paired, n=558): **χ²=3.606, p=0.0576 → fail to reject equal-error null
(statistical tie).** Disagreement: both correct 475, PN-only 44, BF-only 27, both wrong 12.
Bootstrap Δacc(PN−BF) 95% CI [+0.18, +6.09]pp. κ(agreement)=0.815.

Flawed-run McNemar (RETRACTED): χ²=48.01, p=4.2e-12, BF-only 105 vs PN-only 25.

## Per-class F1 — CORRECTED (CSV: [b_familyI_subtype13_perclass_CORRECTED.csv](b_familyI_subtype13_perclass_CORRECTED.csv))

| Class | n | PN++ F1 | BF F1 | Δ (PN−BF) | subtype McNemar χ² | p |
|---|---:|---:|---:|---:|---:|---:|
| anchors | 8 | 0.933 | 0.750 | +0.183 | 0.0 | 1.0 |
| keys | 8 | 1.000 | 0.778 | +0.222 | 0.0 | 1.0 |
| nails | 5 | 0.889 | 1.000 | −0.111 | 0.0 | 1.0 |
| non_fastener | 269 | 0.935 | 0.927 | +0.008 | — | — |
| nuts | 51 | 0.971 | 0.940 | +0.031 | 2.25 | 0.134 |
| pins | 24 | 0.750 | 0.742 | +0.008 | 0.0 | 1.0 |
| retaining-rings | 18 | 1.000 | 0.973 | +0.027 | 0.0 | 1.0 |
| rivets | 37 | 0.987 | 0.935 | +0.052 | 0.0 | 1.0 |
| screws | 85 | 0.971 | 0.874 | +0.097 | **9.091** | **0.00257** |
| spacers | 7 | 0.923 | 0.750 | +0.173 | 0.5 | 0.480 |
| threaded-inserts | 10 | 0.762 | 0.632 | +0.130 | 0.25 | 0.617 |
| threaded-rods | 6 | 0.600 | 1.000 | −0.400 | 0.0 | 1.0 |
| washers | 30 | 0.933 | 0.897 | +0.036 | 0.5 | 0.480 |

Corrected: PN++ ≥ BF on 11/13 classes by F1; BF wins clearly only on `threaded-rods` (n=6) and
`nails` (n=5) — both tiny-n. The only individually-significant per-subtype McNemar is **screws**
(PN-only 11, BF-only 0; p=0.0026). Most per-class n are too small to reach significance alone.

### Rules 13-class row (full 649-file McMaster set, `rule_based_eval/mcmaster649/summary.json`)
Subtype13 accuracy **52.62%**, macro F1 **14.08%**. Non-zero classes only: screws (P 0.36 / R 0.61),
nuts (P 0.46 / R 0.58), washers (P 0.19 / R 0.17), non_fastener (P 0.67 / R 0.72). Six fastener
classes (anchors, keys, nails, pins, retaining-rings, rivets, spacers, threaded-inserts,
threaded-rods) = 0 recall by construction.

Confusion matrices: [../figures/cm_familyI_subtype13_pointnet2_CORRECTED.png](../figures/cm_familyI_subtype13_pointnet2_CORRECTED.png),
[../figures/cm_familyI_subtype13_brepformer.png](../figures/cm_familyI_subtype13_brepformer.png),
[../figures/cm_familyI_subtype13_pointnet2_FLAWED.png](../figures/cm_familyI_subtype13_pointnet2_FLAWED.png) (retracted).

Sources: `full_analysis_subtype13_normals.json` (corrected), `full_analysis_subtype13.json` (flawed).
