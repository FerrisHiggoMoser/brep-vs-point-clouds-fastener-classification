# Table (e) — MAX_FACES sampling-bias (PC vs BRep retention by class/split)

CSV: [e_max_faces_sampling_bias.csv](e_max_faces_sampling_bias.csv)

`step_to_brep.py` caps at `MAX_FACES = 600` (topology-distance matrices are O(N²) and OOM on
large parts). Real-world non-fasteners (valves, manifolds, enclosures, complex brackets) have
600–4,000+ faces; fasteners are simple (<100). The cap silently drops the most distinctive
non-fastener shapes, biasing BRepFormer's training distribution. **Non-fasteners are dropped
~5× more often than fasteners.** (McMaster binary dataset, 2026-05-09.)

| split / class | PC n | BRep n | dropped |
|---|---:|---:|---:|
| train / fastener | 2,368 | 2,283 | 3.6% |
| train / non_fastener | 2,833 | 2,277 | **19.6%** |
| val / fastener | 291 | 278 | 4.5% |
| val / non_fastener | 378 | 313 | **17.2%** |
| test / fastener | 301 | 290 | 3.7% |
| test / non_fastener | 348 | 269 | **22.7%** |

This is the mechanism behind BF's `n=559` test set vs PN++'s `n=649` in the McMaster binary
baseline. The Family-I matched comparison neutralises it by training **both** models on the same
5,710 in-cap parts (and PN++ still wins). For Family II, the v6 study used `max_faces=300`
(drops ~14% outliers).

**Why the BRepFormer paper doesn't hit this:** their datasets (CBF, MFInstSeg, MFTRCAD) are
synthetic with bounded face counts and use per-face segmentation, so no face cap is needed.
Real industrial catalogs have a long face-count tail (3 → 4,000+).
