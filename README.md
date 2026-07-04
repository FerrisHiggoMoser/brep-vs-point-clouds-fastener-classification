# step-vr-step — thesis code

Code for the BSc thesis *Boundary Representation versus Point Clouds for Fastener
Classification in Aerospace CAD*. It contains the two learned models (a from-scratch
BRepFormer and PointNet++ MSG), a training-free rule baseline, the geometric
relationship layer, and the STEP→glTF deployment pipeline used in the thesis.

The datasets are **not** included here (see *Data* below). The numbers reported in
the thesis trace to stored evaluation artifacts, not to a re-run of this code.

## Layout

```
step_vr_step/            the Python package
  config.py, schema.py   configuration and data schema
  cli.py, sidecar_server.py, uuid_registry.py   entry points and identity
  detection/             fastener detection and classification
    rule_based.py          training-free ISO/DIN rule baseline
    iso_tables.py          ISO/DIN dimension tables the rules match against
    brep_signature.py      B-rep signatures (face-type counts, cylinders, dimensions)
    geometric_features.py  geometric feature extraction from a solid
    ml_classifier.py       learned-model inference, plus the top-K geometry-recovery filter
    holes.py, detect.py    hole detection and the relationship layer
                           (screwedInto / contained_in arcs)
  models/
    brepformer/          from-scratch BRepFormer
      feature_extractor.py   STEP -> face UV-grids, edge curves, 4 topology-distance matrices
      face_encoder.py, edge_encoder.py   2D/1D CNN encoders
      attention_bias.py, transformer.py, brepformer.py   the model
      dataset.py, train.py
    pointnet2/           PointNet++ MSG
      pointnet2_cls_msg.py, pointnet2_utils.py   the model
      dataset.py, train.py
    evaluate.py, preprocess.py
  geometry/              tessellation, fingerprinting, transforms, reconstruction
  reconciliation/        STEP<->glTF round-trip reconciliation (diff, matcher, conflict)
  readers/ writers/ exporters/   STEP / glTF / Datasmith / USD I/O
  lod/                   level-of-detail proxies and substitution
  sidecar/ rpc/ validation/   deployment plumbing (sidecar bundles, RPC, round-trip checks)

scripts/                 training, evaluation, and data-preparation entry points
tests/                   unit tests
requirements.txt / environment.yml / pyproject.toml   environment
```

## Key entry points, mapped to the thesis

**Training**
- `scripts/train_pn_breponly.py` — PointNet++ matched (Experiment 1)
- `scripts/train_pn_v6.py` — PointNet++ on the v6 diversified set (Experiment 2 control)
- `step_vr_step/models/brepformer/train.py` — BRepFormer
- `step_vr_step/models/pointnet2/train.py` — PointNet++

**Data preparation**
- `scripts/step_to_brep.py` — STEP → B-rep features (applies the `MAX_FACES` cap)
- `scripts/process_step_to_pointcloud.py`, `scripts/step_to_npy.py` — STEP → point clouds
- `scripts/compute_topology_distances.py` — the four topology-distance matrices
- `scripts/prepare_pn_v6_dataset.py`, `scripts/prepare_pn_subtype13_normals.py` — dataset assembly (the second re-extracts the corrected per-face normals)
- `scripts/relabel_subtype_13.py` — 13-class subtype relabelling
- `scripts/mcmaster_scraper.py` — McMaster-Carr catalogue scrape

**Evaluation**
- `scripts/full_analysis.py` — Experiment 1 binary paired statistics (McNemar, bootstrap)
- `scripts/eval_pn_subtype13_normals.py` — corrected 13-class subtype
- `scripts/eval_pn_v6.py` — Experiment 2 (in-distribution + steam-engine holdout)
- `scripts/eval_rule_based.py` — rule baseline on all four test sets
- `scripts/eval_hybrid.py` — the geometry-only rules→BRepFormer cascade
- `scripts/audit_split_leakage.py` — the within-family leakage audit
- `scripts/stress_test.py` — matcher stress test (up to 1000 bolts)

## Environment

```
conda env create -f environment.yml        # or: pip install -r requirements.txt
```

Two environments were used in the thesis: one with CUDA + PyTorch Lightning for
training, and one with CPU + pythonOCC for STEP extraction. **pythonOCC / Open
CASCADE is required** for the B-rep and point-cloud extraction steps.

## Data

The datasets are not distributed with this code. Sources and licensing are listed
in the thesis Data Provenance appendix: McMaster-Carr (per-SKU STEP exports),
GrabCAD, the Fusion 360 Gallery segmentation and assembly sets, the Mechanical
Components Benchmark (MCB), the ABC dataset, and the ISIS-Space CubeSat assembly.
Raw McMaster geometry and the ISIS assembly are not redistributable.
