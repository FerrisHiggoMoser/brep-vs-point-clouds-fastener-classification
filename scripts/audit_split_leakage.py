"""Audit train/test catalogue-family leakage in the Family-I McMaster splits.

Motivation: McMaster part numbers encode a product family in the leading
digits+letter block (e.g. 8822T912, 8822T921, ... are the same product in
different sizes). A random part-level train/test split can place near-duplicate
size variants of one family on both sides, which inflates absolute test accuracy
relative to a family-grouped split. This script quantifies the overlap.

It does NOT retrain anything; it only reads the .npy split directories and
compares part-number family stems. Output: a JSON summary per dataset.

Reproduce:
    python backend/scripts/audit_split_leakage.py
"""
import json
import os
import re
import glob
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "training_data")
DATASETS = ["mcmaster_pc_breponly", "mcmaster_pc_subtype13"]
OUT = os.path.join(ROOT, "mcmaster_logs", "split_leakage_audit.json")

# part number -> family stem: leading digits + letter block, dropping the size suffix.
FAM_RE = re.compile(r"^(\d+[A-Za-z]+)\d+$")


def family(pn):
    m = FAM_RE.match(pn)
    return m.group(1) if m else None  # None => unparseable; treated as its own singleton


def collect(split_dir):
    out = {}
    for f in glob.glob(os.path.join(split_dir, "*", "*.npy")):
        label = os.path.basename(os.path.dirname(f))
        pn = os.path.splitext(os.path.basename(f))[0]
        out[pn] = label
    return out


def audit(base):
    train = collect(os.path.join(base, "train"))
    test = collect(os.path.join(base, "test"))
    train_fams = defaultdict(list)
    for pn in train:
        fam = family(pn)
        if fam:
            train_fams[fam].append(pn)

    leaked, unparseable = [], 0
    for pn, label in test.items():
        fam = family(pn)
        if fam is None:
            unparseable += 1
            continue
        if fam in train_fams:
            leaked.append({"pn": pn, "family": fam, "class": label,
                           "siblings_in_train": len(train_fams[fam])})

    by_class = defaultdict(lambda: [0, 0])
    for pn, label in test.items():
        by_class[label][1] += 1
    for row in leaked:
        by_class[row["class"]][0] += 1

    n_test = len(test)
    return {
        "n_train": len(train),
        "n_test": n_test,
        "n_test_unparseable_partno": unparseable,
        "n_test_family_in_train": len(leaked),
        "leakage_rate": round(len(leaked) / n_test, 4) if n_test else None,
        "distinct_test_families_in_train": len(sorted({r["family"] for r in leaked})),
        "by_class": {k: {"leaked": v[0], "test": v[1],
                         "rate": round(v[0] / v[1], 4) if v[1] else None}
                     for k, v in sorted(by_class.items())},
        "examples_largest_train_family": sorted(
            leaked, key=lambda r: -r["siblings_in_train"])[:10],
    }


def main():
    result = {}
    for ds in DATASETS:
        base = os.path.join(ROOT, ds)
        if not os.path.isdir(base):
            result[ds] = {"error": f"not found at {base}"}
            continue
        result[ds] = audit(base)
        r = result[ds]
        print(f"### {ds}: train={r['n_train']} test={r['n_test']} "
              f"family-leakage={r['n_test_family_in_train']}/{r['n_test']} "
              f"= {r['leakage_rate'] * 100:.1f}%")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
