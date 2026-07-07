# claims.md — defensible thesis claims, in journal order, with backing

Each claim is quoted/paraphrased from a journal "Defensible thesis claim" block, followed by
**backing artifacts · sample sizes · p-values/CIs · caveats**. Claims superseded by the
2026-06-13 normals correction are marked **⚠️ RETRACTED / SUPERSEDED**. Universal caveats (apply
to all): name-derived ground-truth circularity on assemblies; label noise in source catalogs;
MAX_FACES sampling bias for B-rep; small per-class n; the two comparison families must not be pooled.

---

## C1 — Binary architectural comparison (Family I) · 2026-05-09 · **STANDS**

> "On the McMaster-Carr part library, PointNet++ outperforms BRepFormer for whole-part
> fastener-vs-non-fastener binary classification by **4.30 percentage points** (95% bootstrap CI
> [+1.43, +7.17]pp), under matched data, matched compute (120 epochs), and matched hyperparameters.
> Significant by McNemar (χ²=7.56, p=0.006). PN++ also wins MCC, Macro-F1, AUROC, Brier, with
> comparable calibration. The result holds even on the simpler ≤600-face subset BF was equipped for."

- **Backing:** `training_data/mcmaster_logs/full_analysis.json`; `…/metrics.matched_pn_breponly.json`;
  `…/metrics.plan_c_prime.json`. PN++ ckpt `pointnet2_breponly/best_model.pth`; BF ckpt
  `bf_planc_prime_best.ckpt`.
- **n:** 558 shared test (289 fastener / 269 non). Trained on 5,710 matched parts, 120 epochs.
- **Stats:** PN++ 94.27% vs BF 89.96%; McNemar χ²=7.557, p=0.005977; bootstrap Δ 95% CI [+1.43,+7.17]pp.
  MCC 0.886 vs 0.799; AUROC 0.980 vs 0.957; Brier 0.047 vs 0.079.
- **Caveats:** PN++/BF per-sample predictions never saved → rules-vs-ML tests are bounds (D2); the
  binary-task PN++ here used the *McMaster-chapter* point clouds (degenerate normals — D1) yet still
  won, so the binary win is robust to the normals bug (the bug hurts fine-grained tasks most).
  See [tables/a_familyI_binary_headline.md](tables/a_familyI_binary_headline.md).

## C2 — 13-class subtype, original · 2026-05-11 · **⚠️ RETRACTED**

> ~~"BRepFormer outperforms PointNet++ by 14.34pp on 13-class subtype (p<1e-6), is better
> calibrated, and benefits from multitask supervision; for fine-grained CAD reasoning BRepFormer
> is the architecturally appropriate choice."~~

- **Status:** RETRACTED by the 2026-06-13 correction (C7). The PN++ point clouds had degenerate
  constant (0,0,1) normals; corrected, the 14.34pp BF win becomes a statistical tie.
- **Flawed artifact (kept):** `full_analysis_subtype13.json`. Do **not** cite as a result.

## C3 — Production deployment of the architecture choice · 2026-05-11 · **STANDS (engineering)**

> "The architecture comparison is backed by an end-to-end deployment: the trained BF subtype13
> model classifies parts inside the actual STEP→glTF production pipeline, with labels flowing into
> the manifest, glTF node metadata, relationship inference, and color palette."

- **Backing:** `cad-bidirectional-poc/src/detection/bf_classifier.py`, `…/models/bf_subtype13_best.ckpt`,
  `scripts/smoke_test_bf.py` (3/3 correct on real McMaster STEPs through the production path).
- **Caveat:** the *checkpoint* deployed (bf_subtype13_best) was the model the retracted C2 used; the
  deployment claim is about operational viability, not the (now-retracted) accuracy ranking.

## C4 — Rule-based training-free floor (both families) · 2026-06-12 Exp 1 · **STANDS**

> "A frozen rule-based detector provides a constant training-free floor: 65.1% binary accuracy on
> the matched McMaster test set (vs 89.96% BF and 94.27% PN++, both gaps p<1e-15 under worst-case
> McNemar), 52.6% / 14.1% macro-F1 on 13-class subtype, 33.8% on the v6 split where BF v6 scores
> 89.3%. Learned representations deliver +25 to +56pp over engineered rules on every in-distribution
> task. The exception: on the PADDLE cross-CAD-tool holdout the rules score 62.6% — indistinguishable
> from BF v6's 64.0% — with complementary per-class strengths (rules 100% on ISO screws; v6 83% nuts)."

- **Backing:** `rule_based_eval/{matched558,mcmaster649,paddle,v6test}/summary.json`; stored ML
  refs `full_analysis.json`, `backend/ml_mcmaster_predictions.csv`, `bf_v6_run/eval_summary.json`.
- **n:** matched558=558; mcmaster=648; paddle iso350=350; v6test verified=5,655 (of 7,286).
- **Stats:** binary rules 65.05%; rules-vs-stored-BF-binary McNemar χ²=68.53 p=1.2e-16; rules-vs-PN++
  bound p_worst≤5.8e-27; rules-vs-BF-Plan-C' bound p_worst≤3.0e-18. PADDLE rules 62.57% vs BF 64.0%.
- **Caveats:** rules-vs-PN++/BF-Plan-C' are bounds (D2); v6test covers 78% of samples (worst-case
  bound still ≥52.4pp gap); 6/12 fastener classes are structural zeros (no signature).
  See [tables/a](tables/a_familyI_binary_headline.md), [tables/d](tables/d_familyII_generalization.md).

## C5 — Geometry-only hybrid cascade is distribution-dependent · 2026-06-12 Exp 4/5 · **STANDS**

> "A geometry-only two-stage cascade (frozen rules → BF v6) is not a free lunch: its value has the
> same sign as the distribution shift. In-distribution it is strictly harmful (73.9% vs 89.2%,
> p<1e-160). On PADDLE the cascade with the top-K filter reaches 73.1% vs the model's 64.0% (p=0.005),
> the filter recovering rivets 4%→77% by vetoing impossible 'pin' votes. On the satellite's ISO/DIN
> hardware the cascade is exact (112/112) where the model alone scores 47%. Gate the rule pre-pass on
> catalog-likeness; treat geometric sanity filters as assembly-specific, not universal."

- **Backing:** `training_data/hybrid_eval/{v6test,paddle,satellite}/summary.json`;
  `bf_subtype13_v6.ckpt`; top-K filter imported from `ml_classifier.py` (not re-implemented).
- **n:** v6test common=5,678; PADDLE iso350=350 (full 412 also reported); satellite=176 (ISO/DIN
  subset 112).
- **Stats:** in-dist ml≻hybrid McNemar p≈6.5e-163; PADDLE ml≺hybrid_topk p=0.0054; satellite
  ml≺hybrid p=1.6e-14, ISO/DIN-only b=0/c=59 p=4.3e-14.
- **Caveats:** satellite GT is a 176-part reconstruction (not the unsaved 168 of 05-19); the
  761-series "washers" GT is itself debatable (geometrically tube spacers) — lead with the 112-part
  ISO/DIN subset; PADDLE pins stay 0% in every config; rivet recovery rides on top-3 mass (n=48,
  one assembly). See [tables/d](tables/d_familyII_generalization.md), [tables/f](tables/f_zeroshot_deployment.md).

## C6 — PN++ on v6 (the architectural control) · 2026-06-12/13 · **STANDS (Family II)**

> "Trained on the same v6 diversified dataset (~81% of BF's exact samples — a data disadvantage)
> and evaluated on the same holdouts with sample-paired McNemar vs frozen BF v6, PointNet++
> outperforms BRepFormer on both: in-distribution by 6.55pp (95.33% vs 88.78%, p=6.4e-58, all 13
> classes) and on the PADDLE cross-CAD-tool holdout by 20.9pp (84.9% vs 64.0%, p=2.7e-13) — most
> strikingly recovering taper-pins from BF's 0/29 to 29/29. The in-distribution gap widens from
> synthetic (+3.2pp) to genuine multi-tool CAD (fusion +12.8, grabcad +13.4, realcad +23.5pp).
> The v6 study's 64% is a BRepFormer ceiling, not an ML ceiling."

- **Backing:** `pn_v6_run/eval_summary.json` (+ `test_eval.json`, `paddle_eval.json`); PN++ ckpt
  `pn_v6_run/checkpoints/best-epoch=53-val_loss=0.1429.pth`; BF ckpt `bf_subtype13_v6.ckpt` (frozen);
  `bf_v6_run/eval_summary.json`.
- **n:** v6 test paired=5,827 (of 5,850 PC / 7,286 BF); PADDLE iso350=350.
- **Stats:** in-dist McNemar χ²=257.38 p=6.4e-58, bootstrap Δ 95% CI [+5.75,+7.31]pp; PADDLE
  McNemar χ²=53.44 p=2.7e-13, bootstrap Δ 95% CI [+15.7,+25.7]pp.
- **Caveats:** **not** matched-sample (PN++ on 81% of BF's data — asymmetry favors BF, so PN++ win
  is conservative); PN++ here has real normals, McMaster-chapter PN++ did not → cross-family
  non-comparability (never pool with Family I); in-dist test is synth-heavy but PN++ wins every
  source incl. all 3 real-CAD; PADDLE is one assembly (per-class = one observation each).
  See [tables/c](tables/c_winner_swap.md), [tables/d](tables/d_familyII_generalization.md).

## C7 — The normals correction (supersedes C2) · 2026-06-13 · **STANDS**

> "An earlier experiment reported BRepFormer +14.34pp on 13-class McMaster subtype. Traced to a
> point-cloud bug writing degenerate constant normals. With corrected normals and an otherwise
> byte-identical protocol — BF's frozen checkpoint reproducing 89.96% exactly — PointNet++ scores
> 93.01%, turning the win into a statistical tie (PN++ +3.05pp, McNemar p=0.058), with PN++ also
> leading macro-F1 and calibration. Consistent with the v6 comparison. For whole-part fastener
> subtype classification, point-cloud and B-rep are at best on par, point clouds favored under
> distribution shift; B-rep's advantages, if any, lie in per-face segmentation and relationship
> tasks not measured here."

- **Backing:** `full_analysis_subtype13_normals.json` (corrected); `full_analysis_subtype13.json`
  (flawed, preserved); PN++ ckpt `pointnet2_subtype13_normals/best_model.pth` (val_acc 0.9356);
  BF ckpt `brepformer_subtype13/checkpoints/best-epoch=118-val_loss=0.2612.ckpt` (unchanged).
- **n:** 558 (splits identical to `mcmaster_pc_subtype13`, verified part-number-identical;
  `pn_subtype13_normals_retention.json` train 4,557 / val 590 / test 558, 0 unresolved).
- **Stats:** PN++ 93.01% vs BF 89.96%; McNemar χ²=3.606, **p=0.0576 (n.s.)**; bootstrap Δ 95% CI
  [+0.18,+6.09]pp; PN++ leads ECE (0.034 vs 0.038), Brier (0.107 vs 0.143), macro-F1 (89.65 vs 86.14).
- **Caveats:** BF wins clearly only on threaded-rods (n=6) and nails (n=5); the `.npy` files still
  contain constant normals (examiner-falsifiable — this is the *strength* of the correction).
  See [tables/b](tables/b_familyI_subtype13.md), [tables/h](tables/h_calibration.md).

## C8 — Reframed thesis core: complementary representations · 2026-06-13 · **STANDS (framing)**

> "For learned whole-part fastener *classification*, point clouds (PointNet++) match or beat B-rep
> (BRepFormer) across binary, fine-grained subtype, and cross-CAD-tool generalization. Conversely,
> *relationship inference* requires the B-rep representation's exact geometry, realized with
> geometric reasoning on the CAD kernel rather than a learned model. The contribution is a
> representation–task mapping — point clouds for classification, B-rep for structural/relational
> reasoning — not a single winning architecture."

- **Backing:** C1 (binary), C7 (subtype), C6 (v6 + cross-tool); relationship layer
  `backend/step_vr_step/detection/{detect.py,holes.py}` (161 screwedInto / 592 contained_in on ISIS;
  stress-tested to 1000 bolts, `stress_results.json`).
- **Key distinction:** B-rep the *representation* (exact for relationships) ≠ BRepFormer the *model*
  (no measured classification win). Relationships are computed by geometric algorithms, not a learned head.
- **Caveats:** the per-face segmentation head-to-head (B-rep's design task) is **untested** — listed
  as future work, not a claimed B-rep win (see [MISSING_FIGURES.md](MISSING_FIGURES.md) #8). The
  relationship layer is validated by construction/stress-test, not against a labeled relationship
  ground-truth dataset (none exists for these assemblies).

---

## Cross-family caution (applies whenever C1/C7 are cited next to C6)

Family I (McMaster) and Family II (v6) have **different training distributions** and **different
PC normals provenance** (Family-I PN++ historically degenerate, fixed for subtype13 in C7;
Family-II PN++ real throughout). The two are consistent (point clouds ≥ B-rep for classification,
gap grows under tool shift) but their absolute numbers are **not comparable**. Always state which
family a number comes from.
