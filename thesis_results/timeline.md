# timeline.md — chronological experiment log

Reconstructed from RESEARCH_JOURNAL.md. Columns: date · what was run · headline number · artifact.
Headline numbers verified against stored artifacts where one exists (see
[discrepancies.md](discrepancies.md) for the few that are journal-prose-only).

| Date | What was run | Headline | Artifact |
|---|---|---|---|
| 2026-04-20 | First PointNet++ MSG, 5 epochs, MPS (GrabCAD) | val 88.9% (ep4) — **not a held-out set** (corrected 04-22) | — (no test) |
| 2026-04-21/22 | PointNet++ MSG pretrain on MCB-A, 100 ep | **MCB-A test 93.43%**, macro F1 0.882, n=11,716 | `logs/pointnet2_mcb/best_model.pth`, `eval/summary.json` |
| 2026-04-22 | Stage-2 transfer fine-tune on GrabCAD binary (frozen encoder) | test acc 93.42%, fastener **F1 0.690**, recall 0.829, n=395 | `logs/pointnet2_finetune/`, `eval/summary.json` |
| 2026-04-22 | Stage-3 full fine-tune (unfrozen) — negative | fastener F1 0.634 (worse) | `logs/pointnet2_unfreeze/` (ablation) |
| 2026-04-22 | Error analysis (27 errors) + ISIS zero-shot demo | satellite F1 **0.524**, P 0.717, R 0.413 (551 GT parts) | `logs/pointnet2_finetune/{error_analysis,isispace}/` |
| 2026-04-23 | BRepFormer pretrain on Fusion 360 Gallery (8-class seg) | per-face val_acc **0.9113** | `logs/brepformer_pretrain/best_model.pth` |
| 2026-04-23 | BRepFormer fine-tune on GrabCAD binary | test acc 93.21%, fastener F1 0.699, n=368 | `logs/brepformer_finetune/`, `eval/summary.json` |
| 2026-05-09 | McMaster scrape (3,319) + clean → 7,041 (2,966 f / 4,075 n) | dataset built | `training_data/mcmaster_*` |
| 2026-05-09 | McMaster binary pipeline: PN++ 120ep, BF 80ep | PN++ **92.14%** (n=649), BF baseline **89.62%** (n=559) | `metrics.baseline.json` |
| 2026-05-10 | BF Plan C' (120 ep, MAX_FACES=600, best-by-val_loss ep108) | BF **89.98%** (n=559) / 89.96% (n=558) | `metrics.plan_c_prime.json`, `bf_planc_prime_best.ckpt` |
| 2026-05-10 | PN++ matched (5,710 parts = BF's set), 120 ep | PN++ **93.74%** (n=559) / 94.27% (n=558) | `metrics.matched_pn_breponly.json`, `pointnet2_breponly/best_model.pth` |
| 2026-05-10 | Full paired statistical analysis (binary, n=558) | PN++ 94.27% vs BF 89.96%, **+4.30pp p=0.006** | `full_analysis.json` / `.md` |
| 2026-05-11 | Subtype-13 multiclass (overnight) | ~~BF 89.98% vs PN++ 75.64% (+14.34pp BF)~~ **RETRACTED** | `full_analysis_subtype13.json` (flawed) |
| 2026-05-11 | BF subtype13 deployed into cad-bidirectional poc | 3/3 smoke test correct | `cad-bidirectional-poc/`, `bf_subtype13_best.ckpt` |
| 2026-05-19 | Production relationships + geometry-aware ensemble | McMaster 85.7%, ISIS 85.7%; **161 screwedInto / 592 contained_in** | `detect.py`, `eval_satellite.py` |
| 2026-05-19 | Stress test, 12 synthetic scenarios | all 12 pass, 1000-bolt in 7.32s | `backend/scripts/stress_test.py`, `stress_results.json` |
| 2026-05-21→25 | BRepFormer v2→v6 data diversification | PADDLE **3.1% (v1) → 64.0% (v6)** | `models/bf_subtype13_v{2..6}.ckpt`, `bf_v6_run/eval_summary.json` |
| 2026-06-12 | Exp 1: rule-based standalone (4 test sets) | binary **65.05%** (matched558); PADDLE 62.6%; v6test 33.8% | `rule_based_eval/*/summary.json` |
| 2026-06-12 | Exp 4/5: geometry-only hybrid (rules→v6, ±topk) | PADDLE+topk **73.1%**; satellite ISO/DIN 112/112; in-dist harmful | `hybrid_eval/*/summary.json` |
| 2026-06-12/13 | PN++ trained on v6 data (architectural control) | v6 in-dist **95.33%** (+6.55pp); PADDLE **84.86%** (+20.9pp) | `pn_v6_run/eval_summary.json` |
| 2026-06-13 | **Correction:** re-run PN++ subtype13 with real normals | PN++ **93.01%** vs BF 89.96% → **tie** (p=0.058) | `full_analysis_subtype13_normals.json` |
| 2026-06-13 | Synthesis: complementary representations | reframed thesis core | journal 2026-06-13 |
| 2026-06-13 | **This consolidation** → `thesis_results/` | results package | `thesis_results/` |

### Compute / environment notes
- All ML on a single **RTX 5070 Ti (16 GB, Blackwell sm_120)**, CUDA 12.8, torch 2.11.
- Two `stepvrstep` conda envs: anaconda (CUDA+Lightning+sklearn, training) and miniconda (CPU+OCC,
  extraction/rule-based). Rule-based eval ran CPU-only.
- Recurrent engineering lessons baked into scripts: `if __name__=="__main__"` guard (Windows spawn
  cascade, 2026-05-09); checkpoint selection by val_loss filename token, never mtime (2026-05-09);
  resumable skip-if-exists for any long run.
