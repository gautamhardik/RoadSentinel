import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

ckpt_base = torch.load("weights/bdd100k_yolo11s.pt", map_location="cpu", weights_only=False)
model_base = ckpt_base.get("model")
base_names = model_base.names if hasattr(model_base, "names") else ckpt_base.get("names")

CHECKPOINT_TO_GT_ID = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 9, 6: 6, 7: 5, 8: 7, 9: 8}
GT_CLASSES = {
    0: "person", 1: "rider", 2: "car", 3: "bus", 4: "truck",
    5: "bike", 6: "motor", 7: "traffic light", 8: "traffic sign", 9: "train"
}

print("Checking base model:")
for i in range(10):
    mapped_name = GT_CLASSES[CHECKPOINT_TO_GT_ID[i]]
    actual_name = base_names[i]
    assert mapped_name == actual_name, f"Mismatch at {i}: {mapped_name} vs {actual_name}"
print("EXACT MATCH! GT_CLASSES[CHECKPOINT_TO_GT_ID[i]] is ALWAYS IDENTICAL to model.names[i]!")

ckpt_finetune = torch.load("experiments/phase5a_modeling/runs/phase5a_u_ablation/phase5a_training/yolo11s_vru_finetune/weights/best.pt", map_location="cpu", weights_only=False)
model_ft = ckpt_finetune.get("model")
ft_names = model_ft.names if hasattr(model_ft, "names") else ckpt_finetune.get("names")
print("\nComparing with fine-tune model:")
for i in range(10):
    corrupted_name = GT_CLASSES[CHECKPOINT_TO_GT_ID[i]]
    true_ft_name = ft_names[i]
    print(f"Index {i}: True FT name='{true_ft_name}', but evaluator mapped it to '{corrupted_name}'")
