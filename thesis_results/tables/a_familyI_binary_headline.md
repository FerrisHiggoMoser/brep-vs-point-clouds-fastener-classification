# Table (a) — Family I: 3-way binary headline (matched 558-sample McMaster test)

**Comparison family I** (McMaster matched comparison). Whole-part fastener-vs-non-fastener.
Test set = shared intersection of `mcmaster_pc_breponly/test/` and `mcmaster_brep/test/`
part numbers, **n = 558** (the one part 7513K111 filed under both class folders is de-duplicated).
Matched training data (5,710 parts), matched compute (120 epochs), matched hyperparameters.

Source artifacts: `training_data/mcmaster_logs/full_analysis.json` (PN++, BF — aggregate only,
per-sample predictions never saved) and `training_data/mcmaster_logs/rule_based_eval/matched558/summary.json` (rules).
CSV: [a_familyI_binary_headline.csv](a_familyI_binary_headline.csv)

| Model | n | Accuracy | Macro F1 | MCC | Bal. acc | AUROC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rules (frozen, no training) | 558 | **65.05%** | 65.00% | 0.308 | 65.30% | — | — | — |
| BRepFormer Plan C' | 558 | **89.96%** | 89.96% | 0.799 | 89.99% | 0.957 | 0.079 | 0.023 |
| PointNet++ matched | 558 | **94.27%** | 94.26% | 0.886 | 94.32% | 0.980 | 0.047 | 0.029 |

Accuracy 95% bootstrap CIs (2000 resamples, seed 42): PN++ [92.29, 96.24]; BF [87.46, 92.30];
rules [60.93, 68.99].

### Per-class precision / recall / F1

| Model | fast P | fast R | fast F1 | non P | non R | non F1 |
|---|---:|---:|---:|---:|---:|---:|
| Rules | 0.692 | 0.590 | 0.637 | 0.617 | 0.716 | 0.663 |
| BRepFormer Plan C' | 0.912 | 0.893 | 0.902 | 0.887 | 0.907 | 0.897 |
| PointNet++ matched | 0.961 | 0.927 | 0.944 | 0.925 | 0.959 | 0.942 |

### Significance (CSV: [a_familyI_binary_significance.csv](a_familyI_binary_significance.csv))

| Comparison | Test | Statistic | p-value | Δacc (pp) | 95% CI (pp) |
|---|---|---|---:|---:|---|
| PN++ vs BF (exact, per-sample) | McNemar cc | χ²=7.557 | **0.00598** | +4.30 | [+1.43, +7.17] |
| Rules vs BF-subtype13-binary (exact) | McNemar cc | χ²=68.53 | **1.2e-16** | −22.52 | [−27.42, −17.56] |
| Rules vs PN++ matched (bound) | McNemar bound | p_worst ≤ | **5.8e-27** | −29.21 | — |
| Rules vs BF Plan C' (bound) | McNemar bound | p_worst ≤ | **3.0e-18** | −24.91 | — |

PN++ vs BF disagreement matrix (n=558): both correct 479, PN-only 47, BF-only 23, both wrong 9.
κ(model agreement) = 0.749 (87.5% match).

**Caveats:** PN++ matched and BF Plan C' per-sample predictions were never written to disk, so
the rules-vs-ML McNemar tests are *conservative bounds from the marginals* (worst-case p holds
under any error pairing); only the rules-vs-stored-BF-subtype13-binary test is exact. The
PN++/BF +4.30pp McNemar (χ²=7.557, p=0.006) comes from `full_analysis.json`'s stored paired
counts. See [../discrepancies.md](../discrepancies.md).
