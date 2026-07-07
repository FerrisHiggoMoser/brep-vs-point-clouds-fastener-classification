# provenance.md — checkpoint inventory + number→script mapping

## 1. Checkpoint inventory

Format key: **Lightning** = `.ckpt` with `model.` prefix; **dict** = `model_state_dict` +
`num_classes` + `class_names`; **pth** = bare state-dict. Classes: 2 = binary
[fastener, non_fastener]; 13 = subtype13 [anchors, keys, nails, non_fastener, nuts, pins,
retaining-rings, rivets, screws, spacers, threaded-inserts, threaded-rods, washers].

### Family I — McMaster matched (training_data/mcmaster_logs/)

| Checkpoint | Format | Classes | Training data | Used by |
|---|---|---|---|---|
| `pointnet2/best_model.pth` | pth | 2 | McMaster binary, 6,519 parts, 120 ep | PN++ baseline 92.14% (metrics.baseline.json) |
| `pointnet2_breponly/best_model.pth` | pth | 2 | McMaster, **5,710 matched** parts, 120 ep | **PN++ matched** 94.27% (full_analysis.json) |
| `pointnet2_breponly/best_model.bad.pth` | pth | 2 | (corrupted by spawn cascade) | evidence only — not used |
| `brepformer.baseline/…best-epoch=75-val_loss=0.3052.ckpt` | Lightning | 2 | McMaster BRep ≤600 faces, 80 ep | BF baseline 89.62% (metrics.baseline.json) |
| `bf_planc_prime_best.ckpt` (= brepformer/…best-epoch=108-val_loss=0.2780.ckpt) | Lightning | 2 | same data, 120 ep | **BF Plan C'** 89.96% (full_analysis.json) |
| `pn_subtype13_best.pth` | pth | 13 | McMaster subtype13 (degenerate normals) | ~~flawed PN++ subtype 75.64%~~ (full_analysis_subtype13.json) |
| `pointnet2_subtype13_normals/best_model.pth` | pth | 13 | McMaster subtype13, **real normals**, val_acc 0.9356 | **corrected PN++** 93.01% (full_analysis_subtype13_normals.json) |
| `brepformer_subtype13/…best-epoch=118-val_loss=0.2612.ckpt` | Lightning | 13 | McMaster subtype13 BRep | **BF subtype13** 89.96% (both subtype runs — frozen) |
| `bf_subtype13_best.ckpt` | Lightning | 13 | McMaster subtype13 (= v1) | production deploy; `ml_mcmaster_predictions.csv` (only per-sample ML on disk) |

### Family II — v6 (D:\step-vr-step-thesis\reproducible-build\)

| Checkpoint | Format | Classes | Training data | Used by |
|---|---|---|---|---|
| `models/bf_subtype13_best.ckpt` (v1) | Lightning | 13 | McMaster only (~10k) | v1 baseline; PADDLE 3.1% |
| `models/bf_subtype13_v{2,3,4,5}.ckpt` | dict | 13 | progressive (see table g) | v2–v5 progression (journal-only McMaster numbers — D9) |
| `models/bf_subtype13_v6.ckpt` | dict | 13 | ~67k, ~39% real-CAD | **BF v6** 89.27% / PADDLE 64.0% (bf_v6_run/eval_summary.json) |
| `pn_v6_run/checkpoints/best-epoch=53-val_loss=0.1429.pth` | pth | 13 | v6 PC, ~46,948 train (81% of BF) real normals | **PN++ v6** 95.21% / PADDLE 84.86% (pn_v6_run/eval_summary.json) |

### Pretrain checkpoints (logs/)
- `logs/pointnet2_mcb/best_model.pth` — pth, 68-class, MCB-A, test 93.43% (PN++ encoder source).
- `logs/pointnet2_finetune/best_model.pth` — pth, 2-class, GrabCAD Stage-2, F1 0.69.
- `logs/pointnet2_unfreeze/best_model.pth` — pth, 2-class, Stage-3 ablation, F1 0.63.
- `logs/brepformer_pretrain/best_model.pth` — Lightning, 8-class seg, Fusion 360 Gallery, val 0.9113.
- `logs/brepformer_finetune/best_model.pth` — Lightning, 2-class, GrabCAD binary, test 93.21%.
- `logs/brepformer_seg/best_model.pth` — segmentation experiment (Plan MAX Phase 1).

## 2. Which number came from which script / artifact

| Number(s) | Script | Primary artifact |
|---|---|---|
| MCB-A 93.43% (n=11,716) | `eval_pointnet_mcb.py` | `logs/pointnet2_mcb/eval/summary.json` |
| GrabCAD Stage-2/3 binary | `train.py` + eval | `logs/pointnet2_finetune/eval/summary.json`, `logs/pointnet2_unfreeze/eval/` |
| BF GrabCAD finetune 93.21% | BF train/eval | `logs/brepformer_finetune/eval/summary.json` |
| McMaster binary baseline (PN 92.14 / BF 89.62) | `pipeline.py` eval phase | `metrics.baseline.json` |
| BF Plan C' 89.98/89.96 | `pipeline.py` (best-by-val_loss fix) | `metrics.plan_c_prime.json` |
| PN++ matched 93.74/94.27 | `stage_pc_breponly.py` + `train_pn_breponly.py` | `metrics.matched_pn_breponly.json` |
| **Binary paired stats** (4.30pp, McNemar, CIs, per-category, calibration) | `full_analysis.py` | `full_analysis.json` / `.md` |
| ~~Subtype13 flawed~~ (14.34pp) | `relabel_subtype_13.py`, `pipeline_subtype13.py`, `full_analysis_subtype13.py` | `full_analysis_subtype13.json` |
| **Subtype13 corrected** (93.01% / tie) | `prepare_pn_subtype13_normals.py`, `eval_pn_subtype13_normals.py` | `full_analysis_subtype13_normals.json` |
| Rule-based (4 test sets) | `eval_rule_based.py` | `rule_based_eval/{matched558,mcmaster649,paddle,v6test}/summary.json` |
| Geometry-only hybrid (3 sets × 4 configs) | `eval_hybrid.py` | `hybrid_eval/{v6test,paddle,satellite}/summary.json` |
| BF v6 progression / eval | `prepare_bf_v2_dataset.py`, `eval_bf_v2.py`, `train.py` | `bf_v6_run/eval_summary.json` |
| PN++ on v6 (control) | `prepare_pn_v6_dataset.py`, `train_pn_v6.py`, `eval_pn_v6.py` | `pn_v6_run/eval_summary.json` |
| Relationships (161/592) + ensemble | `detect.py` (`_infer_fastener_relationships`, `_infer_housing_relationships`), `eval_satellite.py` | journal 2026-05-19 (no consolidated JSON) |
| Stress test (12 scenarios) | `stress_test.py` | `backend/stress_results.json` |
| ML-only McMaster benchmark + per-sample CSV | `eval_ml_mcmaster.py` | `backend/ml_mcmaster_predictions.csv` |

## 3. Reproduce this package
```
python thesis_results/_generate.py     # emits tables/*.csv + figures/*.png from the JSONs above
```
No training or evaluation is performed; the script only re-serialises stored numbers and renders
confusion matrices from stored matrices. (The PADDLE PN++ CM was generated by a one-off snippet
reading `pn_v6_run/paddle_predictions.csv`; see figures/README.md.)
