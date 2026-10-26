"""
Build Decoupled-Head Surgical Blend: RoadSentinel Ultimate Champion
Blending:
- Base A: weights/phase14_best.pt (76.53% Precision, 61.32% mAP50)
- Base B: weights/phase13_best.pt (56.61% Recall, 61.01% mAP50)
Target: weights/roadsentinel_ultimate_champion.pt
"""
import os
import copy
import hashlib
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

P14_PATH = Path("weights/phase14_best.pt")
P13_PATH = Path("weights/phase13_best.pt")
OUT_PATH = Path("weights/roadsentinel_ultimate_champion.pt")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 70)
    print("BUILDING ROADSENTINEL ULTIMATE CHAMPION (DECOUPLED-HEAD GRAFT)")
    print("=" * 70)

    print(f"Loading Phase 14: {P14_PATH}...")
    ckpt_p14 = torch.load(P14_PATH, map_location="cpu", weights_only=False)
    print(f"Loading Phase 13: {P13_PATH}...")
    ckpt_p13 = torch.load(P13_PATH, map_location="cpu", weights_only=False)

    model_p14 = ckpt_p14["model"]
    model_p13 = ckpt_p13["model"]

    state_p14 = model_p14.state_dict()
    state_p13 = model_p13.state_dict()

    new_state = {}
    counts = {"backbone_neck": 0, "box_reg": 0, "dfl": 0, "cls_head": 0, "buffers": 0}

    # Blending ratios:
    # Backbone & Neck (Layers 0..22): 0.50 P14 + 0.50 P13
    # Box Regression (23.cv2): 0.50 P14 + 0.50 P13
    # DFL (23.dfl): 0.50 P14 + 0.50 P13
    # Classification Head (23.cv3): 0.70 P14 + 0.30 P13 (Precision Lock Bias)

    for k, v14 in state_p14.items():
        if k not in state_p13:
            new_state[k] = v14
            continue
        v13 = state_p13[k]

        if not torch.is_floating_point(v14):
            new_state[k] = v14
            counts["buffers"] += 1
            continue

        if k.startswith("model.23.cv3."):
            # Classification logits
            new_state[k] = 0.70 * v14 + 0.30 * v13
            counts["cls_head"] += 1
        elif k.startswith("model.23.cv2."):
            # Bounding box regression
            new_state[k] = 0.50 * v14 + 0.50 * v13
            counts["box_reg"] += 1
        elif "dfl" in k:
            new_state[k] = 0.50 * v14 + 0.50 * v13
            counts["dfl"] += 1
        else:
            # Backbone / Neck / C2PSA
            new_state[k] = 0.50 * v14 + 0.50 * v13
            counts["backbone_neck"] += 1

    print(f"Grafted tensor counts: {counts}")

    # Create new checkpoint
    new_model = copy.deepcopy(model_p14)
    new_model.load_state_dict(new_state)

    new_ckpt = copy.deepcopy(ckpt_p14)
    new_ckpt["model"] = new_model
    if "ema" in new_ckpt and new_ckpt["ema"] is not None:
        new_ckpt["ema"] = None
    if "updates" in new_ckpt:
        new_ckpt["updates"] = None

    torch.save(new_ckpt, OUT_PATH)
    file_size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    sha_hash = sha256_file(OUT_PATH)

    print(f"SUCCESS: Saved {OUT_PATH}")
    print(f"File Size: {file_size_mb:.2f} MB")
    print(f"SHA-256: {sha_hash}")

if __name__ == "__main__":
    main()
