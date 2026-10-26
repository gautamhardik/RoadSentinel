import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import torch

ckpt = torch.load("experiments/phase5a_modeling/runs/phase5a_u_ablation/phase5a_training/yolo11s_vru_finetune/weights/best.pt", map_location="cpu", weights_only=False)
model = ckpt.get("model")
print("best.pt model names:", model.names if hasattr(model, "names") else ckpt.get("names"))

with open("experiments/phase5a_modeling/runs/phase5a_u_ablation/eval/phase4d_metrics.json", "r", encoding="utf-8") as f:
    d = json.load(f)
m = d["configurations"]["ActiveFoveation-960"]["non_pilot_confirmation_metrics"]
print("\nAll 10 classes in Phase 5A-U eval:")
for c in m["per_class_results"]:
    print(f"Class {c['class_name']}: TP={c['tp_50']}, FP={c['fp_50']}, FN={c['fn_50']}, Prec={c['precision_50']:.4f}, Rec={c['recall_50']:.4f}, AP50={c['ap_50']:.4f}")
