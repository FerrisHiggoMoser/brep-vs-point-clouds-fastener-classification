# discrepancies.md — journal text vs stored artifacts

Every mismatch found while re-deriving numbers from primary artifacts. **Rule: the artifact
wins.** Most are sub-1pp rounding or sample-count drift; two are material (the retracted subtype13
result, and the missing per-sample ML predictions). Severity: 🔴 material · 🟡 minor/rounding · 🟢 reconciled.

---

### 🔴 D1 — The 2026-05-11 subtype13 headline is an artifact (already corrected in-journal)
- **Journal (2026-05-11):** "BRepFormer wins multiclass by 14.34pp" — PN++ 75.64% vs BF 89.98%.
- **Artifact (`full_analysis_subtype13_normals.json`, 2026-06-13):** PN++ **93.01%**, BF 89.96%,
  gap +3.05pp PN++, McNemar **p=0.058 (tie)**.
- **Resolution:** the journal *itself* retracts this (2026-06-13 correction section) — degenerate
  (0,0,1) PC normals. Consolidation uses the corrected file as canonical; the flawed file is kept
  and labelled RETRACTED. Both confusion matrices are rendered. No action beyond labelling.

### 🔴 D2 — PN++ matched & BF Plan C' per-sample predictions were never saved
- **Journal (2026-06-12 Exp 1):** flags this explicitly.
- **Artifact:** `full_analysis.json` stores **only aggregates** for these two models; the only
  per-sample ML predictions on disk are `backend/ml_mcmaster_predictions.csv` (bf_subtype13_best
  collapsed to binary). Consequence: the rules-vs-PN++ and rules-vs-BF-Plan-C' McNemar tests in
  table (a) are **conservative bounds from the marginals**, not exact. The PN++-vs-BF +4.30pp
  McNemar (χ²=7.557, p=0.006) is exact (its paired counts a_only/b_only ARE stored in
  `full_analysis.json` → `paired.mcnemar`). Faithfully reported as bounds in table (a).

---

### 🟡 D3 — BF Plan C' binary accuracy: 89.98% (n=559) vs 89.96% (n=558)
- **Journal (2026-05-11 headline):** BF Plan C' = **89.98%**.
- **Artifact (`full_analysis.json`):** accuracy = 0.899641… = **89.96%** (n=558).
- **Cause:** the 89.98% is from `metrics.plan_c_prime.json` (test_acc 0.89982, **n=559**, the full
  intersection before the 1-sample ML-load drop). `full_analysis` scored 558 → 89.96%. Both are
  "correct" for their n. Tables use the `full_analysis.json` n=558 number (89.96%) for the matched
  comparison and note 89.98%/559 where the journal headline uses it.

### 🟡 D4 — PN++ matched binary: 93.74% vs 94.27%/94.31%
- **Journal:** "PN++ matched: **93.74%**" (`metrics.matched_pn_breponly.json`, test_acc 0.93739,
  **n=559**).
- **Artifact (`full_analysis.json`):** **94.27%** (n=558, the shared subset re-scored).
- **Cause:** same n=559→558 drift plus the shared-subset re-evaluation. The journal also quotes
  94.31% in places (0.943221 balanced acc / earlier full_analysis value). Canonical headline used:
  **94.27%** accuracy (n=558) from `full_analysis.json`. (The 94.31% appears as `balanced_accuracy`
  0.94322 and as an earlier-rounded accuracy in journal prose.)

### 🟡 D5 — matched test n: 558 vs 559
- **Journal:** uses both "558" and "559" depending on stage. **Artifacts:** `full_analysis*.json`
  = 558; `metrics.*` and `rule_based_eval/matched558` (which scores 559 then reports on 558) = 559
  intersection. **Cause:** part 7513K111 is filed under *both* class folders (a McMaster mislabel
  flagged 2026-05-19); intersection = 559 part numbers, 558 unique after de-dup, and ML-load drops
  one more in `full_analysis`. Documented in table (g). Not an error — a known dual-filing.

### 🟡 D6 — subtype13 test class_distribution: non_fastener 269 vs 270
- **Artifact A (`relabel_manifest.subtype13.json`):** pc/test/non_fastener = **270**, nuts = 51.
- **Artifact B (`full_analysis_subtype13_normals.json`):** non_fastener = **269**, nuts = 51 (sum 558).
- **Artifact C (`rule_based_eval/matched558` subtype13):** non_fastener support = **268**, nuts = **52**.
- **Cause:** the 7513K111 dual-file again — counted as non_fastener in the tree (270), one dropped
  at scoring (269 in full_analysis), and the rule eval's GT resolver assigns it to nuts (so 268
  non + 52 nuts). All three are internally consistent given where the duplicate lands. Headline
  per-class table (b) uses `full_analysis_subtype13_normals.json` (269/51).

### 🟡 D7 — Rules subtype13 on matched558: 49.82% (artifact) vs journal's 52.6%
- **Journal (2026-06-12):** leads with subtype13 **52.62%** — but that is the **649**-file set, not
  matched558. **Artifact (`rule_based_eval/matched558/summary.json`):** matched558 subtype13 =
  **49.82%**. No conflict once you match the test set; table (b) reports both with their n.

### 🟡 D8 — v6test rules n: journal "~6,700" / "5,655" / hybrid "5,678"
- **Journal:** Exp 1 reports rules scored on **5,655** verified v6test samples (of 7,286), prompt
  said "~6,700". The hybrid experiment (Exp 4/5) reports **5,678** common samples (the +23 from a
  fusion trailing-UUID resolver fix). **Artifacts confirm both:** `rule_based_eval/v6test`=5,655;
  `hybrid_eval/v6test`=5,678. Different resolver versions → different retained n. Both reported
  with their n in table (d); neither is wrong.

### 🟡 D9 — v6 BF McMaster test: 0.934 (eval) vs 0.916 (v1) vs journal "0.952"
- **Artifact (`bf_v6_run/eval_summary.json`):** v2_mcm_test (v6) = **0.9337**; v1_mcm_test = 0.9157.
- **Journal v6 table:** McMaster test 0.934 ✓ (v6). The journal's v5 "0.952" and v4 "0.952" are
  separate versions not independently stored here (only v1 and v6 checkpoints were re-evaluated in
  `eval_summary.json`). v2–v5 McMaster numbers are journal-only — flagged as **unverifiable
  against a stored eval** (see claims.md caveats).

### 🟡 D10 — PADDLE iso350 BF v6 = 64.0%: stored as accuracy, journal as 224/350
- 224/350 = 0.6400 ✓ exact. Reconciled. PN++ 297/350 = 0.84857 ✓.

---

### 🟢 D11 — Numbers verified to match exactly (no discrepancy)
- MCB-A test 93.43% (`logs/pointnet2_mcb/eval/summary.json` 0.93428) ✓
- GrabCAD Stage-2 test acc 93.42%, n=395, macro F1 0.827 ✓
- BF GrabCAD finetune test 93.21%, n=368 ✓
- BF baseline 89.62% / n=559, PN++ baseline 92.14% / n=649 (`metrics.baseline.json`) ✓
- Binary McNemar χ²=7.557 p=0.006; subtype13-corrected McNemar χ²=3.606 p=0.0576 ✓
- v6 PN++ test 95.21% (n=5850) / paired-vs-BF 95.33% (n=5827) ✓; PADDLE 84.86% ✓; McNemar
  p=2.7e-13 ✓
- v6 BF test 89.27% (n=7286), macro F1 85.38% ✓
- ISIS satellite hybrid: rules 65.91%, hybrid 70.45%, ISO/DIN-only 112/112 ✓
- All v1→v6 PADDLE progression values match the journal table ✓

### Unverifiable against any stored primary artifact (journal-prose only — flagged, not trusted)
- v2/v3/v4/v5 BRepFormer McMaster-test and val_acc numbers (only v1 & v6 checkpoints re-evaluated
  in `bf_v6_run/eval_summary.json`).
- The "v6 acc on the same Stage-1-claimed parts" column of table (d.3) (journal text; not in the
  stored hybrid summary, which stores only stage1/stage2 aggregate accuracy).
- The 2026-05-19 satellite "85.7% on 168 named parts" — the 168-part selection was never saved;
  the consolidated satellite numbers use the reconstructed 176-part GT (`satellite_meta.json`).
- Stress-test per-scenario detail beyond the journal table (`stress_results.json` not consolidated).
- Day-4 satellite F1 0.524 and the relationship arc counts (161/592/522/69) — journal-only;
  no machine-readable summary on disk in this consolidation pass.
