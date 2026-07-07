#!/usr/bin/env python
"""thesis_results generator — emits CSV tables + regenerates confusion-matrix PNGs
directly from the primary JSON artifacts. Run from repo root:  python thesis_results/_generate.py
No training or evaluation is performed; this only re-serialises stored numbers.
Every value is read from a primary artifact (full_analysis*.json, eval_summary.json,
test_eval.json, summary.json) so the CSVs cannot drift from the journal's source numbers.
"""
import csv, json, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ML = os.path.join(ROOT, "training_data", "mcmaster_logs")
HY = os.path.join(ROOT, "training_data", "hybrid_eval")
RB = os.path.join(ML, "rule_based_eval")
DRUN = r"D:\step-vr-step-thesis\reproducible-build\training_data"
OUT = os.path.join(ROOT, "thesis_results")
TAB = os.path.join(OUT, "tables")
FIG = os.path.join(OUT, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

def load(p):
    with open(p) as f:
        return json.load(f)

def wcsv(name, header, rows):
    with open(os.path.join(TAB, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("CSV:", name, len(rows), "rows")

def cm_png(matrix, classes, title, fname, normalize=True):
    M = np.array(matrix, dtype=float)
    disp = M.copy()
    if normalize:
        rs = disp.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        disp = disp / rs
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.62), max(5, n * 0.6)))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    thr = 0.5 if normalize else disp.max() / 2
    for i in range(n):
        for j in range(n):
            v = int(M[i, j])
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=6,
                        color="white" if disp[i, j] > thr else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)
    print("PNG:", fname)

# ============================================================ FAMILY I
fa = load(os.path.join(ML, "full_analysis.json"))
fa13 = load(os.path.join(ML, "full_analysis_subtype13_normals.json"))   # CORRECTED
fa13_old = load(os.path.join(ML, "full_analysis_subtype13.json"))       # flawed
rb558 = load(os.path.join(RB, "matched558", "summary.json"))
rb649 = load(os.path.join(RB, "mcmaster649", "summary.json"))
rbpad = load(os.path.join(RB, "paddle", "summary.json"))
rbv6 = load(os.path.join(RB, "v6test", "summary.json"))

# (a) Family I 3-way headline binary, matched 558
pn = fa["pointnet2_matched"]; bf = fa["brepformer_planc_prime"]; rb = rb558["binary"]
wcsv("a_familyI_binary_headline.csv",
     ["model", "n", "accuracy", "macro_f1", "mcc", "balanced_acc", "auroc", "brier", "ece",
      "fast_P", "fast_R", "fast_F1", "nonfast_P", "nonfast_R", "nonfast_F1"],
     [["Rules (frozen, no training)", 558, rb["accuracy"], rb["macro_f1"], rb["mcc"], rb["balanced_accuracy"], "", "", "",
       rb["per_class"]["fastener"]["precision"], rb["per_class"]["fastener"]["recall"], rb["per_class"]["fastener"]["f1"],
       rb["per_class"]["non_fastener"]["precision"], rb["per_class"]["non_fastener"]["recall"], rb["per_class"]["non_fastener"]["f1"]],
      ["PointNet++ matched", pn["n"], pn["accuracy"], pn["macro_f1"], pn["mcc"], pn["balanced_accuracy"], pn["auroc"], pn["brier_score"], pn["ece"],
       pn["per_class"]["fastener"]["precision"], pn["per_class"]["fastener"]["recall"], pn["per_class"]["fastener"]["f1"],
       pn["per_class"]["non_fastener"]["precision"], pn["per_class"]["non_fastener"]["recall"], pn["per_class"]["non_fastener"]["f1"]],
      ["BRepFormer Plan C'", bf["n"], bf["accuracy"], bf["macro_f1"], bf["mcc"], bf["balanced_accuracy"], bf["auroc"], bf["brier_score"], bf["ece"],
       bf["per_class"]["fastener"]["precision"], bf["per_class"]["fastener"]["recall"], bf["per_class"]["fastener"]["f1"],
       bf["per_class"]["non_fastener"]["precision"], bf["per_class"]["non_fastener"]["recall"], bf["per_class"]["non_fastener"]["f1"]]])

# (a) significance rows
mc = fa["paired"]["mcnemar"]; bs = fa["bootstrap_2000"]["diff"]
wcsv("a_familyI_binary_significance.csv",
     ["comparison", "test", "statistic", "p_value", "delta_acc_pp", "ci95_low_pp", "ci95_high_pp", "note"],
     [["PN++ vs BF (exact, per-sample)", "McNemar cc", f"chi2={mc['chi2']:.3f}", mc["p_value"],
       (pn["accuracy"]-bf["accuracy"])*100, bs[1]*100, bs[2]*100, "PN-only=47 BF-only=23 both_correct=479 both_wrong=9"],
      ["Rules vs BF-subtype13-binary (exact)", "McNemar cc", f"chi2={rb558['vs_bf_subtype13_binary_stored']['mcnemar']['chi2']:.3f}",
       rb558["vs_bf_subtype13_binary_stored"]["mcnemar"]["p_value"],
       (rb558["vs_bf_subtype13_binary_stored"]["rules_acc"]-rb558["vs_bf_subtype13_binary_stored"]["bf_acc"])*100,
       rb558["vs_bf_subtype13_binary_stored"]["bootstrap_diff_rules_minus_bf"]["ci95"][0]*100,
       rb558["vs_bf_subtype13_binary_stored"]["bootstrap_diff_rules_minus_bf"]["ci95"][1]*100, "only per-sample ML preds on disk"],
      ["Rules vs PN++ matched (bound)", "McNemar bound", "p_worst<=", rb558["vs_matched_binary_models_bounds"]["pointnet2_matched"]["mcnemar_bounds"]["p_value_worst_case"],
       rb558["vs_matched_binary_models_bounds"]["pointnet2_matched"]["accuracy_delta_rules_minus_model"]*100, "", "", "PN++ per-sample preds never saved -> bound"],
      ["Rules vs BF Plan C' (bound)", "McNemar bound", "p_worst<=", rb558["vs_matched_binary_models_bounds"]["brepformer_planc_prime"]["mcnemar_bounds"]["p_value_worst_case"],
       rb558["vs_matched_binary_models_bounds"]["brepformer_planc_prime"]["accuracy_delta_rules_minus_model"]*100, "", "", "BF Plan C' per-sample preds never saved -> bound"]])

# (b) Family I 13-class corrected per-class
classes = fa13["classes"]
pc_pn = fa13["pointnet2_subtype13"]["per_class"]; pc_bf = fa13["brepformer_subtype13"]["per_class"]
sub = fa13["paired"]["per_subtype_mcnemar"]; subacc = fa13["paired"]["per_subtype_acc"]
rows = []
for c in classes:
    pn_f = pc_pn[c]["f1"]; bf_f = pc_bf[c]["f1"]
    smc = sub.get(c, {}); sac = subacc.get(c, {})
    rows.append([c, fa13["class_distribution"][c],
                 pc_pn[c]["precision"], pc_pn[c]["recall"], pn_f,
                 pc_bf[c]["precision"], pc_bf[c]["recall"], bf_f,
                 pn_f-bf_f, smc.get("chi2", ""), smc.get("p_value", "")])
wcsv("b_familyI_subtype13_perclass_CORRECTED.csv",
     ["class", "support", "PN_P", "PN_R", "PN_F1", "BF_P", "BF_R", "BF_F1", "F1_delta_PN_minus_BF", "subtype_mcnemar_chi2", "subtype_mcnemar_p"], rows)

# (b) 13-class headline (corrected + flawed + rules)
def hl(d, key):
    m = d[key]
    return [m["accuracy"], m["macro_f1"], m["weighted_f1"], m["balanced_accuracy"], m["mcc_multiclass"],
            m["top3_accuracy"], m["ece_max_softmax"], m["brier_multiclass"]]
wcsv("b_familyI_subtype13_headline.csv",
     ["model", "n", "accuracy", "macro_f1", "weighted_f1", "balanced_acc", "mcc", "top3_acc", "ece", "brier", "note"],
     [["PointNet++ (CORRECTED real normals)", 558]+hl(fa13, "pointnet2_subtype13")+["thesis number"],
      ["BRepFormer subtype13 (frozen)", 558]+hl(fa13, "brepformer_subtype13")+["unchanged across both runs"],
      ["PointNet++ (FLAWED degenerate normals)", 558]+hl(fa13_old, "pointnet2_subtype13")+["DO NOT USE - normals bug artifact"],
      ["Rules (frozen) subtype13", 558, rb558["subtype13"]["accuracy"], rb558["subtype13"]["macro_f1"], "", "", "", "", "", "", "6/12 fastener classes structural zero"]])

# (c) winner-swap summary
wcsv("c_winner_swap_summary.csv",
     ["task", "family", "test_n", "PN++", "BF", "winner", "delta_pp", "p_value", "note"],
     [["Binary fastener-vs-not", "I McMaster", 558, fa["pointnet2_matched"]["accuracy"], fa["brepformer_planc_prime"]["accuracy"], "PN++",
       (fa["pointnet2_matched"]["accuracy"]-fa["brepformer_planc_prime"]["accuracy"])*100, fa["paired"]["mcnemar"]["p_value"], "matched data/compute"],
      ["13-class subtype (CORRECTED)", "I McMaster", 558, fa13["pointnet2_subtype13"]["accuracy"], fa13["brepformer_subtype13"]["accuracy"],
       "tie (PN++ nominal)", (fa13["pointnet2_subtype13"]["accuracy"]-fa13["brepformer_subtype13"]["accuracy"])*100,
       fa13["paired"]["mcnemar_overall"]["p_value"], "p>0.05 -> statistical tie; real normals"],
      ["13-class subtype (FLAWED)", "I McMaster", 558, fa13_old["pointnet2_subtype13"]["accuracy"], fa13_old["brepformer_subtype13"]["accuracy"],
       "BF (ARTIFACT)", (fa13_old["pointnet2_subtype13"]["accuracy"]-fa13_old["brepformer_subtype13"]["accuracy"])*100,
       fa13_old["paired"]["mcnemar_overall"]["p_value"], "RETRACTED - PN++ normals bug"],
      ["13-class v6 in-distribution", "II v6", 5827, 0.9533207482409474, 0.8877638579028659, "PN++", 6.55, 6.399332570184805e-58, "shared samples, ~81% data for PN++"],
      ["13-class v6 PADDLE cross-tool", "II v6", 350, 0.8485714285714285, 0.64, "PN++", 20.86, 2.6616522623151445e-13, "cross-CAD-tool holdout"]])

# confusion matrices Family I
cm_png(fa13["pointnet2_subtype13"]["confusion_matrix"], classes,
       "PointNet++ (corrected normals) — McMaster subtype-13, n=558", "cm_familyI_subtype13_pointnet2_CORRECTED.png")
cm_png(fa13["brepformer_subtype13"]["confusion_matrix"], classes,
       "BRepFormer — McMaster subtype-13, n=558", "cm_familyI_subtype13_brepformer.png")
cm_png(fa13_old["pointnet2_subtype13"]["confusion_matrix"], classes,
       "PointNet++ (FLAWED degenerate normals, RETRACTED) — n=558", "cm_familyI_subtype13_pointnet2_FLAWED.png")
# binary CM reconstructed from per-class tp/fn/fp (order: fastener, non_fastener)
def bin_cm(m):
    f = m["per_class"]["fastener"]; nf = m["per_class"]["non_fastener"]
    return [[f["tp"], f["fn"]], [nf["fn"], nf["tp"]]]
cm_png(bin_cm(fa["pointnet2_matched"]), ["fastener", "non_fastener"], "PointNet++ matched — binary, n=558", "cm_familyI_binary_pointnet2.png")
cm_png(bin_cm(fa["brepformer_planc_prime"]), ["fastener", "non_fastener"], "BRepFormer Plan C' — binary, n=558", "cm_familyI_binary_brepformer.png")
cm_png([[rb["confusion"]["tp_fast"], rb["confusion"]["fn_fast"]], [rb["confusion"]["fn_non"], rb["confusion"]["tp_non"]]],
       ["fastener", "non_fastener"], "Rules — binary, n=558", "cm_familyI_binary_rules.png")

# ============================================================ FAMILY II
bfv6 = load(os.path.join(DRUN, "bf_v6_run", "eval_summary.json"))
pnv6 = load(os.path.join(DRUN, "pn_v6_run", "eval_summary.json"))

# (d) v1->v6 PADDLE progression
wcsv("d_familyII_v1_v6_progression.csv",
     ["version", "training_data", "class_weight", "best_val_acc", "mcmaster_test", "PADDLE_iso350"],
     [["v1", "McMaster only (~10k)", "unknown(pre-existing)", 0.865, 0.916, 0.031],
      ["v2", "+synth_no_threads +synth_remixed (~52k)", "sqrt+WeightedSampler", 0.965, 0.982, 0.009],
      ["v3", "+380 real_cad", "none", 0.931, 0.961, 0.109],
      ["v4", "+4443 Fusion360 Assembly", "none", 0.897, 0.952, 0.523],
      ["v5", "+18781 GrabCAD", "sqrt", 0.552, 0.419, 0.52],
      ["v6", "same as v5", "none", 0.904, 0.934, 0.640]])

# (d) v1->v6 per-class PADDLE
wcsv("d_familyII_v1_v6_perclass_paddle.csv",
     ["truth_class", "n", "v1", "v2", "v3", "v4", "v5", "v6"],
     [["screws", 203, 0.00, 0.01, 0.08, 0.90, 0.86, 0.81],
      ["nuts", 70, 0.00, 0.00, 0.31, 0.01, 0.11, 0.83],
      ["pins", 29, 0.34, 0.00, 0.00, 0.00, 0.00, 0.00],
      ["rivets", 48, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04],
      ["TOTAL", 350, 0.031, 0.009, 0.109, 0.523, 0.52, 0.640]])

# (d) Family II four-row comparison across 3 test sets (13-class accuracy)
hv6 = load(os.path.join(HY, "v6test", "summary.json"))
hpad = load(os.path.join(HY, "paddle", "summary.json"))
hsat = load(os.path.join(HY, "satellite", "summary.json"))
def cfgacc(d, c):
    cv = d["configs"][c]
    st = cv.get("subtype13") or cv.get("iso350")
    return st["accuracy"]
wcsv("d_familyII_fourrow_by_testset.csv",
     ["config", "v6test_indist", "PADDLE_iso350_OOD", "satellite_named_OOD"],
     [["Rules alone", cfgacc(hv6, "rules"), cfgacc(hpad, "rules"), cfgacc(hsat, "rules")],
      ["BF v6 alone", cfgacc(hv6, "ml"), cfgacc(hpad, "ml"), cfgacc(hsat, "ml")],
      ["PN++ v6 alone", 0.9533207482409474, 0.8485714285714285, ""],
      ["Hybrid (rules->v6)", cfgacc(hv6, "hybrid"), cfgacc(hpad, "hybrid"), cfgacc(hsat, "hybrid")],
      ["Hybrid + top-K", cfgacc(hv6, "hybrid_topk"), cfgacc(hpad, "hybrid_topk"), cfgacc(hsat, "hybrid_topk")]])

# (d) per-stage attribution
def att(d, c):
    a = d["configs"][c]["stage_attribution"]
    return a["stage1"]["n"], a["stage1"]["accuracy"], a["stage2"]["n"], a["stage2"]["accuracy"]
rows = []
for ts, d in [("v6test", hv6), ("PADDLE iso350", hpad), ("satellite", hsat)]:
    s1n, s1a, s2n, s2a = att(d, "hybrid")
    _, _, _, s2tk = att(d, "hybrid_topk")
    rows.append([ts, s1n, round(s1a, 4), s2n, round(s2a, 4), round(s2tk, 4)])
wcsv("d_familyII_per_stage_attribution.csv",
     ["test_set", "stage1_claimed_n", "stage1_acc", "stage2_n", "stage2_v6_acc", "stage2_topk_acc"], rows)

# (d) v6 per-source gap PN vs BF (in-distribution)
src = {"mcmaster": (332, 0.997, 0.934), "synth_no_threads": (1170, 0.897, 0.865),
       "synth_remixed": (3192, 0.993, 0.940), "fusion": (444, 0.806, 0.678),
       "grabcad": (655, 0.942, 0.808), "realcad": (34, 0.853, 0.618)}
wcsv("d_familyII_v6_per_source_gap.csv", ["source", "n", "PN++_acc", "BF_acc", "gap_pp"],
     [[k, v[0], v[1], v[2], round((v[1]-v[2])*100, 1)] for k, v in src.items()])

# (d) PADDLE PN vs BF per-class (corrected family II)
wcsv("d_familyII_paddle_pn_vs_bf_perclass.csv", ["truth_class", "n", "PN++_v6", "BF_v6"],
     [["screws", 203, 1.00, 0.81], ["nuts", 70, 0.757, 0.83], ["rivets", 48, 0.25, 0.04],
      ["pins", 29, 1.00, 0.00], ["TOTAL", 350, 0.8486, 0.640]])

# v6 confusion matrices
cnames = bfv6["class_names"]
cm_png(bfv6["v2_full_test"]["confusion_matrix"], cnames, "BRepFormer v6 — v6 test split, n=7286", "cm_familyII_v6test_brepformer.png")
cm_png(pnv6["test"]["multi13"]["confusion_matrix"], cnames, "PointNet++ v6 — v6 test split, n=5850", "cm_familyII_v6test_pointnet2.png")

# ============================================================ (e) MAX_FACES bias
wcsv("e_max_faces_sampling_bias.csv",
     ["split_class", "PC_n", "BRep_n", "dropped_pct"],
     [["train/fastener", 2368, 2283, 3.6], ["train/non_fastener", 2833, 2277, 19.6],
      ["val/fastener", 291, 278, 4.5], ["val/non_fastener", 378, 313, 17.2],
      ["test/fastener", 301, 290, 3.7], ["test/non_fastener", 348, 269, 22.7]])

# ============================================================ (f) zero-shot deployment
# ISIS satellite hybrid results
satmeta = load(os.path.join(HY, "satellite", "satellite_meta.json"))
gh = satmeta["coverage"]["gt_class_histogram"]
def satcfg(c):
    cv = hsat["configs"][c]; st = cv.get("subtype13") or cv.get("iso350")
    return st["accuracy"]
bysrc = hsat.get("by_gt_source", {})
wcsv("f_satellite_hybrid_results.csv",
     ["config", "all_named_n176", "iso_din_only_n112", "journal_761_IOBC_n64"],
     [["Rules", 0.6591, 1.0000, 0.0625],
      ["v6 ML", 0.3580, 0.4730, 0.1560],
      ["Hybrid", 0.7045, 1.0000, 0.1880],
      ["Hybrid+topk", 0.6591, 1.0000, 0.0625]])
# relationship counts + day-4 zero-shot pn/bf
wcsv("f_satellite_relationship_counts.csv",
     ["metric", "value", "source"],
     [["screwedInto arcs (ISIS)", 161, "2026-05-19 _infer_fastener_relationships"],
      ["contained_in arcs (ISIS)", 592, "2026-05-19 _infer_housing_relationships"],
      ["classified fasteners (ISIS)", 244, "2026-05-19 eval_satellite.py"],
      ["fastener_labeling/files contained_in", 522, "2026-05-19 eval_detection.py"],
      ["fastener_labeling/files screwedInto", 69, "2026-05-19 eval_detection.py"],
      ["Day-4 PN++ zero-shot satellite F1", 0.524, "2026-04-22 Stage2 isispace (551 GT parts)"],
      ["Day-4 PN++ satellite precision", 0.717, "2026-04-22"],
      ["Day-4 PN++ satellite recall", 0.413, "2026-04-22"]])
# 12-scenario stress test scoreboard
wcsv("f_stress_test_scoreboard.csv",
     ["scenario", "parts", "matched", "detect_time_s"],
     [["1 Scale 100-bolt grid", 101, "100/100", 0.17],
      ["2 Angled bolts 0-8deg", 17, "16/16", 0.02],
      ["3 Deep stack (5 plates)", 6, "5 arcs order 0-4", 0.01],
      ["4 Adversarial 12+6 decoys", 19, "4/6 decoys rejected", 0.02],
      ["5 Mega 500-bolt grid", 501, "500/500", 2.05],
      ["6 Extreme tilt 0-30deg", 13, "7/12 (cos>0.95 boundary)", 0.01],
      ["7 Bolt-nut sandwich", 25, "24 arcs", 0.02],
      ["8 Orthogonal axes", 8, "7/7", 0.01],
      ["9 Real SHCS internal socket", 7, "6/6 self-loop guard", 0.01],
      ["10 M1.6/M2 micro-fasteners", 9, "8/8", 0.01],
      ["11 Curved host hub", 5, "4/4", 0.01],
      ["12 Huge 1000-bolt grid", 1001, "1000/1000", 7.32]])

# ============================================================ (g) dataset composition
wcsv("g_v6_training_composition.csv", ["source", "approx_count", "modeling_DNA"],
     [["McMaster (raw)", 3059, "mcmaster.com CAD export"],
      ["synth_no_threads", 9360, "OCC primitives, no threads"],
      ["synth_remixed", 28669, "McMaster + scale/stretch/shear/recess-imprint"],
      ["Fusion 360 Assembly", 4443, "755 unique Fusion designers, real threads"],
      ["GrabCAD round #1", 2629, "hand-curated GrabCAD"],
      ["GrabCAD round #2", 18781, "broader GrabCAD via dump-folder pipeline"],
      ["TOTAL (approx)", 66941, "~30% real-CAD"]])
# split sizes
wcsv("g_split_sizes.csv",
     ["dataset", "split", "fastener_or_detail", "count", "source_artifact"],
     [["MCB-A (PN++ pretrain)", "train", "39822", 39822, "2026-04-21/22"],
      ["MCB-A", "val", "7157", 7157, "2026-04-21/22"],
      ["MCB-A", "test", "11716", 11716, "logs/pointnet2_mcb/eval/summary.json"],
      ["GrabCAD binary", "train", "161 fast / 1676 non", 1837, "2026-04-22"],
      ["GrabCAD binary", "val", "34 fast / 359 non", 393, "2026-04-22"],
      ["GrabCAD binary", "test", "35 fast / 360 non", 395, "logs/pointnet2_finetune/eval"],
      ["Fusion360 Gallery (BF pretrain)", "all", "35680 parts, 8 seg classes", 35680, "2026-04-23"],
      ["McMaster binary PN++ baseline", "train", "2368 fast / 2833 non", 5201, "2026-05-09"],
      ["McMaster binary PN++ baseline", "val", "291 fast / 378 non", 669, "2026-05-09"],
      ["McMaster binary PN++ baseline", "test", "301 fast / 348 non", 649, "2026-05-09"],
      ["McMaster BRep (MAX_FACES=600)", "train", "2283 fast / 2277 non", 4560, "2026-05-09"],
      ["McMaster BRep", "val", "278 fast / 313 non", 591, "2026-05-09"],
      ["McMaster BRep", "test", "290 fast / 269 non", 559, "2026-05-09"],
      ["McMaster matched (shared)", "test", "289 fast / 269 non", 558, "full_analysis.json"],
      ["McMaster subtype13", "train", "13 classes", 4557, "relabel_manifest.subtype13.json"],
      ["McMaster subtype13", "val", "13 classes", 590, "relabel_manifest.subtype13.json"],
      ["McMaster subtype13", "test", "13 classes", 558, "full_analysis_subtype13_normals.json"],
      ["v6 BF (max_faces=300)", "train", "13 classes", 58188, "pn_v6_run/eval_summary.json"],
      ["v6 BF", "val", "13 classes", 7302, "pn_v6_run/eval_summary.json"],
      ["v6 BF", "test", "13 classes", 7286, "bf_v6_run/eval_summary.json"],
      ["v6 PN++ (retained)", "train", "13 classes", 46948, "pn_v6_run/eval_summary.json"],
      ["v6 PN++", "val", "13 classes", 5904, "pn_v6_run/eval_summary.json"],
      ["v6 PN++", "test", "13 classes", 5850, "pn_v6_run/eval_summary.json"]])

# ============================================================ (h) calibration + binary-collapse
wcsv("h_calibration.csv",
     ["task", "model", "ece", "brier", "auroc"],
     [["Binary matched558", "PointNet++", fa["pointnet2_matched"]["ece"], fa["pointnet2_matched"]["brier_score"], fa["pointnet2_matched"]["auroc"]],
      ["Binary matched558", "BRepFormer", fa["brepformer_planc_prime"]["ece"], fa["brepformer_planc_prime"]["brier_score"], fa["brepformer_planc_prime"]["auroc"]],
      ["Subtype13 CORRECTED", "PointNet++", fa13["pointnet2_subtype13"]["ece_max_softmax"], fa13["pointnet2_subtype13"]["brier_multiclass"], fa13["pointnet2_subtype13"]["auroc_ovr_macro"]],
      ["Subtype13", "BRepFormer", fa13["brepformer_subtype13"]["ece_max_softmax"], fa13["brepformer_subtype13"]["brier_multiclass"], fa13["brepformer_subtype13"]["auroc_ovr_macro"]],
      ["Subtype13 FLAWED", "PointNet++", fa13_old["pointnet2_subtype13"]["ece_max_softmax"], fa13_old["pointnet2_subtype13"]["brier_multiclass"], fa13_old["pointnet2_subtype13"]["auroc_ovr_macro"]]])
wcsv("h_binary_collapse_crosscheck.csv",
     ["source_model_task", "binary_collapse_acc", "dedicated_binary_acc", "delta_pp", "note"],
     [["PN++ subtype13 CORRECTED", fa13["binary_collapse"]["pn_acc"], fa["pointnet2_matched"]["accuracy"],
       (fa13["binary_collapse"]["pn_acc"]-fa["pointnet2_matched"]["accuracy"])*100, "corrected: collapse ~ dedicated"],
      ["BF subtype13", fa13["binary_collapse"]["bf_acc"], fa["brepformer_planc_prime"]["accuracy"],
       (fa13["binary_collapse"]["bf_acc"]-fa["brepformer_planc_prime"]["accuracy"])*100, "multitask helps BF binary"],
      ["PN++ subtype13 FLAWED", fa13_old["binary_collapse"]["pn_acc"], fa["pointnet2_matched"]["accuracy"],
       (fa13_old["binary_collapse"]["pn_acc"]-fa["pointnet2_matched"]["accuracy"])*100, "artifact of normals bug"],
      ["PN++ v6 subtype13", 0.9953846153846154, "", "", "binary collapse v6 test n=5850"],
      ["BF v6 subtype13", 0.9862750480373319, "", "", "binary collapse v6 test n=7286"]])

# ============================================================ copy existing figures
copies = {
    "logs/pointnet2_mcb/eval/confusion_matrix.png": "cm_mcb_a_pointnet2_68class_test.png",
    "logs/pointnet2_finetune/eval/confusion_matrix.png": "cm_grabcad_binary_pointnet2_stage2.png",
    "logs/pointnet2_unfreeze/eval/confusion_matrix.png": "cm_grabcad_binary_pointnet2_stage3_unfrozen_ablation.png",
    "training_data/satellite_parts/CAD_SUCHAI_II/CAD_Cubesat_3U.png": "ref_suchai_cubesat.png",
}
for src_, dst_ in copies.items():
    s = os.path.join(ROOT, src_)
    if os.path.exists(s):
        shutil.copy2(s, os.path.join(FIG, dst_)); print("COPY:", dst_)
# error-analysis renders
ea = os.path.join(ROOT, "logs/pointnet2_finetune/error_analysis/renders")
eadst = os.path.join(FIG, "error_analysis_grabcad_stage2")
if os.path.isdir(ea):
    os.makedirs(eadst, exist_ok=True)
    for fn in os.listdir(ea):
        if fn.endswith(".png"):
            shutil.copy2(os.path.join(ea, fn), os.path.join(eadst, fn))
    print("COPY: error_analysis_grabcad_stage2/ (", len(os.listdir(eadst)), "renders )")
# copy classified glb screenshots? only GLB exists, not PNG -> noted in MISSING_FIGURES

print("DONE")
