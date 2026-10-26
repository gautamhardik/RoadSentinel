import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

ckpt_base = torch.load("weights/bdd100k_yolo11s.pt", map_location="cpu", weights_only=False)
model_base = ckpt_base.get("model")
print("Base model names:", model_base.names if hasattr(model_base, "names") else ckpt_base.get("names"))

ckpt_finetune = torch.load("experiments/phase5a_modeling/runs/phase5a_u_ablation/phase5a_training/yolo11s_vru_finetune/weights/best.pt", map_location="cpu", weights_only=False)
model_finetune = ckpt_finetune.get("model")
print("Finetuned model names:", model_finetune.names if hasattr(model_finetune, "names") else ckpt_finetune.get("names"))
