# figures/ — index

Confusion-matrix PNGs regenerated from stored `.json`/`.npy` matrices by
[`../_generate.py`](../_generate.py) (matplotlib, row-normalised, raw counts overlaid); other
plots copied verbatim from `logs/`. Each filename encodes family, task, model.

## Confusion matrices — regenerated from JSON

| File | Source artifact | n |
|---|---|---|
| `cm_familyI_binary_pointnet2.png` | full_analysis.json (per-class tp/fn) | 558 |
| `cm_familyI_binary_brepformer.png` | full_analysis.json | 558 |
| `cm_familyI_binary_rules.png` | rule_based_eval/matched558/summary.json | 558 |
| `cm_familyI_subtype13_pointnet2_CORRECTED.png` | full_analysis_subtype13_normals.json | 558 |
| `cm_familyI_subtype13_brepformer.png` | full_analysis_subtype13_normals.json | 558 |
| `cm_familyI_subtype13_pointnet2_FLAWED.png` ⚠️ RETRACTED | full_analysis_subtype13.json | 558 |
| `cm_familyII_v6test_pointnet2.png` | pn_v6_run/eval_summary.json (test.multi13) | 5,850 |
| `cm_familyII_v6test_brepformer.png` | bf_v6_run/eval_summary.json (v2_full_test) | 7,286 |
| `cm_familyII_paddle_pointnet2.png` | pn_v6_run/paddle_predictions.csv (iso350) | 350 |

## Copied verbatim from logs/

| File | Origin |
|---|---|
| `cm_mcb_a_pointnet2_68class_test.png` | logs/pointnet2_mcb/eval/confusion_matrix.png (MCB-A 68-class, 93.4%) |
| `cm_grabcad_binary_pointnet2_stage2.png` | logs/pointnet2_finetune/eval/ (GrabCAD binary, F1 0.69) |
| `cm_grabcad_binary_pointnet2_stage3_unfrozen_ablation.png` | logs/pointnet2_unfreeze/eval/ (Stage-3 ablation) |
| `ref_suchai_cubesat.png` | training_data/satellite_parts/CAD_SUCHAI_II/ (reference render) |
| `error_analysis_grabcad_stage2/` (15 PNGs) | logs/pointnet2_finetune/error_analysis/renders/ (FP/MISS 4-view renders) |

See [../MISSING_FIGURES.md](../MISSING_FIGURES.md) for figures the thesis still needs that
cannot be produced from existing artifacts (e.g. classified.glb screenshots, training curves,
satellite spatial relationship renders).
