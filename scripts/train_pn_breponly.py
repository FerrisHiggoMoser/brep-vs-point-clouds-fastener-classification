"""Train PointNet++ on the BF-restricted dataset (5,710 parts ≤600 faces).

Then evaluate on the matched test split (559 samples) to give the cleanest
possible apples-to-apples architectural comparison vs Plan C' BRepFormer.

Writes:
  - log: training_data/mcmaster_logs/pointnet2_breponly/train.log
  - eval: training_data/mcmaster_logs/metrics.matched_pn_breponly.json

NOTE: The actual logic is inside `if __name__ == "__main__":` because PyTorch
DataLoader on Windows uses spawn-multiprocessing — workers re-execute this
script top-to-bottom. Without the guard, the eval-phase DataLoader's workers
would re-run subprocess.run() and start a runaway training cascade.
"""
from pathlib import Path
import json, os, subprocess, sys
from datetime import datetime
from collections import Counter

REPO = Path(r"c:\Users\ferri\OneDrive\Documents\GitHub\step-vr-step")
sys.path.insert(0, str(REPO / "backend"))

PC_DATA = REPO / "training_data" / "mcmaster_pc_breponly"
LOG_DIR = REPO / "training_data" / "mcmaster_logs" / "pointnet2_breponly"
METRICS_OUT = REPO / "training_data" / "mcmaster_logs" / "metrics.matched_pn_breponly.json"
EPOCHS = 120


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    train_log = LOG_DIR / "train.log"

    print(f"[{datetime.now():%H:%M:%S}] Training PN++ on matched data")
    print(f"  data:   {PC_DATA}")
    print(f"  log:    {train_log}")
    cmd = [
        sys.executable, "-u", "-m", "step_vr_step.models.pointnet2.train",
        "--data_path", str(PC_DATA),
        "--epochs", str(EPOCHS),
        "--batch_size", "16",
        "--num_points", "4096",
        "--num_workers", "2",
        "--use_normals",
        "--lr", "0.001",
        "--log_dir", str(LOG_DIR),
    ]
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"
    with train_log.open("ab") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              cwd=str(REPO / "backend"), env=env)
    print(f"[{datetime.now():%H:%M:%S}] training exit code {proc.returncode}")

    ckpts = sorted(LOG_DIR.glob("*.pth"))
    if not ckpts:
        print("no checkpoint found")
        sys.exit(1)
    ckpt_path = ckpts[-1]
    print(f"\n[{datetime.now():%H:%M:%S}] Evaluating {ckpt_path.name}")

    import torch
    from torch.utils.data import DataLoader
    from step_vr_step.models.pointnet2.pointnet2_cls_msg import PointNet2ClsMSG
    from step_vr_step.models.pointnet2.dataset import FastenerPointCloudDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = FastenerPointCloudDataset(root=str(PC_DATA), num_points=4096,
                                        use_normals=True, split="test", augment=False)
    # num_workers=0 here on purpose — keep eval single-process so we can't trigger
    # the spawn-cascade bug a second time even if the guard ever broke.
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    model = PointNet2ClsMSG(num_classes=test_ds.num_classes, use_normals=True).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.eval()

    correct = total = 0
    cm: Counter = Counter()
    with torch.no_grad():
        for points, labels in loader:
            points = points.transpose(1, 2).to(device)
            labels = labels.to(device)
            logits, _ = model(points)
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for p, l in zip(preds.tolist(), labels.tolist()):
                cm[(l, p)] += 1

    result = {
        "matched_pn_breponly": {
            "test_acc": correct / total,
            "n": total,
            "checkpoint": str(ckpt_path),
            "epochs": EPOCHS,
            "confusion": {f"{l}->{p}": c for (l, p), c in cm.items()},
            "classes": test_ds.classes,
        }
    }
    METRICS_OUT.write_text(json.dumps(result, indent=2))
    print(f"\n[{datetime.now():%H:%M:%S}] DONE")
    print(f"  test acc: {correct/total:.4f} ({correct}/{total})")
    print(f"  confusion: {dict(cm)}")
    print(f"  -> {METRICS_OUT}")


if __name__ == "__main__":
    main()
