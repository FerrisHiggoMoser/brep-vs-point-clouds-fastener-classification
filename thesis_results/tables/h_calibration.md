# Table (h) — Calibration (ECE / Brier) + binary-collapse cross-check

## h.1 — Calibration (CSV: [h_calibration.csv](h_calibration.csv))

Lower is better for ECE and Brier; higher for AUROC. Sources: `full_analysis.json`,
`full_analysis_subtype13_normals.json` (corrected), `full_analysis_subtype13.json` (flawed).

| Task | Model | ECE | Brier | AUROC |
|---|---|---:|---:|---:|
| Binary matched558 | PointNet++ | 0.029 | 0.047 | 0.980 |
| Binary matched558 | BRepFormer | 0.023 | 0.079 | 0.957 |
| Subtype13 CORRECTED | PointNet++ | **0.034** | **0.107** | 0.997 |
| Subtype13 | BRepFormer | 0.038 | 0.143 | 0.993 |
| ~~Subtype13 FLAWED~~ | ~~PointNet++~~ | ~~0.066~~ | ~~0.345~~ | ~~0.975~~ |

**Calibration reversal (corrected):** the flawed run reported BF better-calibrated than PN++
(ECE 0.038 vs 0.066, Brier 0.143 vs 0.345). With real normals, **PN++ is now better-calibrated**
on subtype13 (ECE 0.034 vs 0.038, Brier 0.107 vs 0.143) — directly reversing the 2026-05-11
"BF is better calibrated / benefits from multitask" finding. On the binary task BF remains
marginally better-calibrated (ECE 0.023 vs 0.029) despite lower accuracy.

The rules' confidence is **uncalibrated** — on mcmaster649 true-positive and false-positive
fastener calls share the same median confidence (0.75); it is a threshold artifact, not signal.

## h.2 — Binary-collapse cross-check (CSV: [h_binary_collapse_crosscheck.csv](h_binary_collapse_crosscheck.csv))

Collapse 13-class predictions → binary (any fastener subtype → "fastener"), compare to the
dedicated binary-trained model on the same 558 test set.

| Source model / task | Binary-collapse acc | Dedicated binary acc | Δ (pp) | Note |
|---|---:|---:|---:|---|
| PN++ subtype13 CORRECTED | 94.09% | 94.27% | −0.18 | collapse ≈ dedicated (corrected) |
| BF subtype13 | 93.19% | 89.96% | +3.23 | multitask helps BF binary |
| ~~PN++ subtype13 FLAWED~~ | ~~80.65%~~ | ~~94.27%~~ | ~~−13.66~~ | artifact of normals bug |
| PN++ v6 subtype13 | 99.54% | — | — | v6 test n=5,850 |
| BF v6 subtype13 | 98.63% | — | — | v6 test n=7,286 |

**Reversal:** the flawed run claimed "multitask training helps BF, hurts PN++" (PN++ collapse
−13.66pp). Corrected, PN++'s 13-class model collapses to binary at 94.09% — essentially equal to
its dedicated 94.27%, so the "multitask hurts PN++" finding does **not** survive. BF still gains
+3.18pp from multitask supervision on binary collapse.
