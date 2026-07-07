# Table (c) — Task-dependent winner summary (binary vs multiclass)

CSV: [c_winner_swap_summary.csv](c_winner_swap_summary.csv)

> **Note on the corrected story.** Before the 2026-06-13 normals correction, the thesis story
> was a "winner swap": PN++ wins binary, BF wins multiclass. **After correction that swap
> disappears** — PN++ matches or beats BF on every measured whole-part classification task.
> The table keeps the flawed row for transparency but it is RETRACTED. Treat the four
> non-retracted rows as the load-bearing summary.

| Task | Family | n | PN++ | BF | Winner | Δ (pp) | p-value |
|---|---|---:|---:|---:|---|---:|---:|
| Binary fastener-vs-not | I McMaster | 558 | 94.27% | 89.96% | **PN++** | +4.30 | 0.006 |
| 13-class subtype (CORRECTED) | I McMaster | 558 | 93.01% | 89.96% | tie (PN++ nominal) | +3.05 | 0.058 (n.s.) |
| ~~13-class subtype (FLAWED)~~ | I McMaster | 558 | ~~75.63%~~ | ~~89.96%~~ | ~~BF~~ (RETRACTED) | ~~−14.34~~ | ~~4.2e-12~~ |
| 13-class v6 in-distribution | II v6 | 5827 | 95.33% | 88.78% | **PN++** | +6.55 | 6.4e-58 |
| 13-class v6 PADDLE cross-tool | II v6 | 350 | 84.86% | 64.00% | **PN++** | +20.86 | 2.7e-13 |

### Interpretation (corrected synthesis, journal 2026-06-13)

- For **learned whole-part classification**, point clouds (PointNet++) match or beat B-rep
  (BRepFormer) across binary, fine-grained subtype, and cross-CAD-tool generalization. The
  fine-grained advantage B-rep was expected to hold **does not materialize once point clouds
  carry surface normals**, and it inverts under cross-tool distribution shift.
- The two comparison families must **stay separate** (different training distributions, and the
  Family-I PN++ point clouds historically had degenerate normals while Family-II PN++ has real
  normals). Never pool numbers across families.
- Where an honest B-rep case still lives (not a measured classification win): per-face
  segmentation (BF's design target, **untested head-to-head**) and topological relationship
  inference (done by geometric algorithms on the OCC kernel, not a learned model).
