# Table (d) — Family II: v6 cross-CAD-tool generalization

**Comparison family II** (cross-CAD-tool generalization). Different training distribution from
Family I; PN++ here uses **real surface normals**. Holdout = PADDLE STEAMER (cross-tool),
plus the in-distribution v6 test split and the ISIS satellite.

## d.1 — v1→v6 PADDLE progression (CSV: [d_familyII_v1_v6_progression.csv](d_familyII_v1_v6_progression.csv))

BRepFormer only; data-scaling study (no architectural control until the PN++-on-v6 run).

| Ver | Training data | Class-weight | val_acc | McMaster test | PADDLE iso350 |
|---|---|---|---:|---:|---:|
| v1 | McMaster only (~10k) | unknown (pre-existing) | 0.865 | 0.916 | **3.1%** |
| v2 | +synth_no_threads +synth_remixed (~52k) | sqrt + WeightedSampler | 0.965 | 0.982 | **0.9%** |
| v3 | +380 real_cad | none | 0.931 | 0.961 | **10.9%** |
| v4 | +4,443 Fusion 360 Assembly | none | 0.897 | 0.952 | **52.3%** |
| v5 | +18,781 GrabCAD | sqrt | 0.552 (never converged) | 0.419 | **52%** |
| **v6** | same as v5 | none | **0.904** | **0.934** | **64.0%** |

Per-class PADDLE accuracy by version (CSV: [d_familyII_v1_v6_perclass_paddle.csv](d_familyII_v1_v6_perclass_paddle.csv)):

| Truth | n | v1 | v2 | v3 | v4 | v5 | v6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| screws | 203 | 0% | 1% | 8% | 90% | 86% | 81% |
| nuts | 70 | 0% | 0% | 31% | 1% | 11% | 83% |
| pins | 29 | 34% | 0% | 0% | 0% | 0% | 0% |
| rivets | 48 | 0% | 0% | 0% | 0% | 0% | 4% |
| **TOTAL** | 350 | 3.1% | 0.9% | 10.9% | 52.3% | 52% | 64.0% |

## d.2 — Four-config × three-test-set 13-class accuracy (CSV: [d_familyII_fourrow_by_testset.csv](d_familyII_fourrow_by_testset.csv))

Rows: rules / BF-v6 / PN++-v6 / hybrid(rules→v6) / hybrid+topk.

| Config | v6test (in-dist) | PADDLE iso350 (OOD) | Satellite named (OOD) |
|---|---:|---:|---:|
| Rules alone | 33.66% | 53.16%† | 65.91% |
| BF v6 alone | 89.15% | 64.00% (paired 64.0%) | 35.80% |
| **PN++ v6 alone** | **95.33%** | **84.86%** | — (not run) |
| Hybrid (rules→v6) | 73.92% | 68.45%† | 70.45% |
| Hybrid + top-K | 64.74% | 76.94%† | 65.91% |

† PADDLE values in this hybrid table are on the **n=412** full holdout (`hybrid_eval/paddle/summary.json`
top-level `configs`); the **iso350** 350-part comparable values are: rules 62.57%, BF-v6 64.00%,
hybrid 63.14%, hybrid+topk **73.14%** (see d.4 / journal 2026-06-12). PN++-v6 84.86% is on iso350.

**Verdicts:** in-distribution the hybrid is strictly worse than ML alone (−15.2pp, McNemar
p≈6.5e-163); on PADDLE the hybrid wins only **with** the top-K filter (+9.1pp over ML, p=0.0054);
on the satellite the hybrid wins outright (+34.7pp over ML, p=1.6e-14) and the top-K filter
gives the win back.

## d.3 — Per-stage attribution (CSV: [d_familyII_per_stage_attribution.csv](d_familyII_per_stage_attribution.csv))

| Test set | Stage1 claimed | Rules acc on claimed | (v6 would score) | Stage2 n | v6 acc there | +topk acc there |
|---|---:|---:|---:|---:|---:|---:|
| v6test | 2,476 (43.6%) | 55.2% | 90.1% | 3,202 | 88.4% | 72.1% |
| PADDLE iso350 | 269 (76.9%) | 81.4% | 82.5% | 81 | 2.5% | **45.7%** |
| Satellite | 116 (65.9%) | **100.0%** | 47.4% | 60 | 13.3% | 0.0% |

Stage-1 claim precision tracks catalog-likeness: 100% on the satellite's clean DIN/ISO hardware,
81% on PADDLE catalog parts, 55% on the v6 split (synthetic-remixed scale/shear defeats ISO
dimension matching). (The "v6 acc on the same claimed parts" figures are from the journal text.)

## d.4 — PN++ v6 vs BF v6, per-class on PADDLE (CSV: [d_familyII_paddle_pn_vs_bf_perclass.csv](d_familyII_paddle_pn_vs_bf_perclass.csv))

| Truth | n | PN++ v6 | BF v6 |
|---|---:|---:|---:|
| screws | 203 | **100%** (203/203) | 81% |
| nuts | 70 | 75.7% (53/70) | **83%** |
| rivets | 48 | **25%** (12/48) | 4% |
| pins | 29 | **100%** (29/29) | 0% |
| **TOTAL** | 350 | **84.86%** (297/350) | **64.0%** (224/350) |

PN++ vs BF McNemar (iso350, paired by solid index): χ²=53.44, p=2.7e-13; both correct 212,
PN-only 85, BF-only 12, both wrong 41. Standout: **pins PN++ 29/29 vs BF 0/29.**

## d.5 — v6 in-distribution PN++−BF gap by source (CSV: [d_familyII_v6_per_source_gap.csv](d_familyII_v6_per_source_gap.csv))

| Source | n | PN++ acc | BF acc | gap |
|---|---:|---:|---:|---:|
| mcmaster | 332 | 0.997 | 0.934 | +6.3pp |
| synth_no_threads | 1,170 | 0.897 | 0.865 | +3.2pp |
| synth_remixed | 3,192 | 0.993 | 0.940 | +5.3pp |
| **fusion (real)** | 444 | 0.806 | 0.678 | **+12.8pp** |
| **grabcad (real)** | 655 | 0.942 | 0.808 | **+13.4pp** |
| **realcad (real)** | 34 | 0.853 | 0.618 | **+23.5pp** |

The gap is smallest on synthetic/thread-stripped geometry and ~quadruples on genuine multi-tool
CAD — the in-distribution shadow of the cross-tool PADDLE result.

**Retention caveat:** PN++ trained on ~81% of BF's exact samples (deleted GrabCAD source files);
the data asymmetry runs *against* PN++, so the PN++ win is conservative on data volume. v6test PN++
paired n=5,827; full v6 BF test n=7,286 (89.27%).

Sources: `D:\…\bf_v6_run\eval_summary.json`, `D:\…\pn_v6_run\eval_summary.json`,
`training_data/hybrid_eval/{v6test,paddle,satellite}/summary.json`.
Confusion matrices: [../figures/cm_familyII_v6test_pointnet2.png](../figures/cm_familyII_v6test_pointnet2.png),
[../figures/cm_familyII_v6test_brepformer.png](../figures/cm_familyII_v6test_brepformer.png).
