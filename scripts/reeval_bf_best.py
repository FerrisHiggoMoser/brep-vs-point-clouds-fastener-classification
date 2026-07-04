"""Re-eval BRepFormer with the actual best-by-val_loss checkpoint
(filename like best-epoch=NN-val_loss=X.YYYY.ckpt — pick min val_loss)."""
from pathlib import Path
import re, json, sys
import torch
from collections import Counter
from torch.utils.data import DataLoader

REPO = Path(r"c:\Users\ferri\OneDrive\Documents\GitHub\step-vr-step")
sys.path.insert(0, str(REPO / "backend"))
from step_vr_step.models.brepformer.dataset import BRepDataset, brep_collate_fn
from step_vr_step.models.brepformer.brepformer import BRepFormer

ckpt_dir = REPO / "training_data" / "mcmaster_logs" / "brepformer" / "checkpoints"
ckpts = list(ckpt_dir.glob("*.ckpt"))
def vl(p):
    m = re.search(r"val_loss=(\d+\.\d+)", p.name)
    return float(m.group(1)) if m else float("inf")
ckpts.sort(key=vl)
best = ckpts[0]
print(f"=== chosen checkpoint (lowest val_loss) ===")
print(f"  {best.name}  val_loss={vl(best):.4f}")
print(f"  (alt: {[(c.name, vl(c)) for c in ckpts]})")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_ds = BRepDataset(root=str(REPO / "training_data" / "mcmaster_brep"), split="test")
loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=brep_collate_fn)
model = BRepFormer(num_classes=test_ds.num_classes).to(device)
ck = torch.load(best, map_location=device, weights_only=False)
sd = ck.get("state_dict", ck)
sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.eval()

correct = total = 0
cm = Counter()
with torch.no_grad():
    for batch in loader:
        out = model(face_grids=batch["face_grids"].to(device),
                    edge_curves=batch["edge_curves"].to(device),
                    topo_distances={k: v.to(device) for k, v in batch["topo_distances"].items()},
                    mask=batch["face_mask"].to(device),
                    edge_mask=batch["edge_mask"].to(device))
        labels = batch["labels"].to(device)
        preds = out.argmax(-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for p, l in zip(preds.tolist(), labels.tolist()):
            cm[(l, p)] += 1

print(f"\n=== test results (Plan C' best-by-val_loss) ===")
print(f"  test acc: {correct/total:.4f} ({correct}/{total})")
print(f"  classes: {test_ds.classes}")
print(f"  confusion (label->pred): {dict(cm)}")
result = {"plan_c_prime_best_by_val_loss": {
    "checkpoint": str(best), "val_loss": vl(best),
    "test_acc": correct/total, "n": total,
    "confusion": {f"{l}->{p}": c for (l,p), c in cm.items()},
    "classes": test_ds.classes,
}}
out_path = REPO / "training_data" / "mcmaster_logs" / "metrics.plan_c_prime.json"
out_path.write_text(json.dumps(result, indent=2))
print(f"\n  -> {out_path}")
