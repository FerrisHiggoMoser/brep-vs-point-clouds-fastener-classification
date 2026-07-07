# tables/ — index

Every table is provided as clean Markdown (`.md`) + machine-readable CSV (`.csv`). All numbers
re-derived from primary artifacts (`full_analysis*.json`, `eval_summary.json`, `test_eval.json`,
`summary.json`, `relabel_manifest*.json`) by [`../_generate.py`](../_generate.py) and cross-checked
against the journal. Where journal prose and artifact disagreed, the artifact won — see
[../discrepancies.md](../discrepancies.md).

**Two comparison families are kept strictly separate** (different training distributions, and the
Family-I PN++ historically had degenerate normals while Family-II PN++ has real normals):

- **Family I** — McMaster matched comparison (tables a, b, c, e, h binary/subtype13 rows).
- **Family II** — v6 cross-CAD-tool generalization (table d, h v6 rows).
- The **rule-based baseline** appears in both (no training data).

| Table | Topic | Files |
|---|---|---|
| (a) | Family I 3-way binary headline (matched 558) | [a_familyI_binary_headline.md](a_familyI_binary_headline.md) · `.csv` + `a_familyI_binary_significance.csv` |
| (b) | Family I 13-class subtype matched (CORRECTED) | [b_familyI_subtype13.md](b_familyI_subtype13.md) · `b_familyI_subtype13_headline.csv` + `b_familyI_subtype13_perclass_CORRECTED.csv` |
| (c) | Task-dependent winner summary | [c_winner_swap.md](c_winner_swap.md) · `c_winner_swap_summary.csv` |
| (d) | Family II v1→v6 + rules/PN++/hybrid/topk | [d_familyII_generalization.md](d_familyII_generalization.md) · 6 CSVs |
| (e) | MAX_FACES sampling bias | [e_max_faces_bias.md](e_max_faces_bias.md) · `e_max_faces_sampling_bias.csv` |
| (f) | Zero-shot: ISIS + relationships + stress test | [f_zeroshot_deployment.md](f_zeroshot_deployment.md) · 3 CSVs |
| (g) | Dataset composition + split sizes | [g_dataset_composition.md](g_dataset_composition.md) · 2 CSVs |
| (h) | Calibration (ECE/Brier) + binary collapse | [h_calibration.md](h_calibration.md) · 2 CSVs |

⚠️ The 2026-05-11 "BRepFormer wins subtype +14.34pp" result is **RETRACTED** (PointNet++
normals-extraction bug). Corrected numbers are the thesis numbers; flawed numbers are kept only
as a documented retraction (struck-through / labelled FLAWED).
