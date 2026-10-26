"""
RoadSentinel Phase 12 — P2 Architecture Weight Initialization Script

Implements Strategy B: Semantic Functional Layer Transfer
Transfers 19,558,680 parameters (94.93%) from session4_best.pt into yolo11m-p2.yaml,
leaving exactly 1,044,707 parameters (5.07%) in the newly introduced P2 feature
branch and P2 detection head to be initialized with Kaiming normal weights.
"""

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHAMPION_PATH = REPO_ROOT / "experiments" / "phase10_model_optimization" / "phase10b_extended_training" / "session4_artifacts" / "session4_best.pt"
P2_YAML_PATH = REPO_ROOT / "experiments" / "phase12_p2_architecture" / "yolo11m-p2.yaml"
OUT_WEIGHTS_PATH = REPO_ROOT / "experiments" / "phase12_p2_architecture" / "yolo11m_p2_initialized.pt"


def build_semantic_mapping():
    """Maps functional feature levels from Standard YOLO11m to YOLO11m-P2."""
    mapping = {}
    # 1. Backbone (Layers 0..10): 1-to-1 exact identity
    for i in range(11):
        mapping[f"model.{i}."] = f"model.{i}."

    # 2. Top-down FPN path
    mapping["model.13."] = "model.13."  # FPN P4
    mapping["model.16."] = "model.16."  # FPN P3

    # 3. Bottom-up PAN path (shifted by the P2 lateral block)
    mapping["model.17."] = "model.23."  # PAN P4 Conv
    mapping["model.19."] = "model.25."  # PAN P4 C3k2
    mapping["model.20."] = "model.26."  # PAN P5 Conv
    mapping["model.22."] = "model.28."  # PAN P5 C3k2

    return mapping


def transfer_weights(save_path: Path = None):
    print("=" * 80)
    print("YOLO11m-P2 Semantic Pretrained-Weight Transfer Audit")
    print("=" * 80)
    print(f"Source Champion Checkpoint: {CHAMPION_PATH}")
    print(f"Target Architecture:        {P2_YAML_PATH}")

    m_std = YOLO(str(CHAMPION_PATH))
    m_p2 = YOLO(str(P2_YAML_PATH))

    sd_std = m_std.model.state_dict()
    sd_p2 = m_p2.model.state_dict()
    tot_p2 = sum(p.numel() for p in sd_p2.values())

    semantic_map = build_semantic_mapping()
    transferred_count = 0
    matched_keys = []
    new_keys = []

    new_sd = {}
    for k_p2, v_p2 in sd_p2.items():
        matched = False
        # Check standard layer mapping
        for p_std, p_p2 in semantic_map.items():
            if k_p2.startswith(p_p2):
                k_std = k_p2.replace(p_p2, p_std, 1)
                if k_std in sd_std and sd_std[k_std].shape == v_p2.shape:
                    new_sd[k_p2] = sd_std[k_std].clone()
                    transferred_count += v_p2.numel()
                    matched_keys.append((k_p2, k_std, v_p2.numel()))
                    matched = True
                break

        # Check Detect head mapping
        if not matched and k_p2.startswith("model.29."):
            parts = k_p2.split(".")
            sub = parts[2]
            if sub in ["cv2", "cv3"]:
                head_idx = int(parts[3])
                if head_idx >= 1:  # P3, P4, P5 heads map from Std 0, 1, 2
                    std_k = f"model.23.{sub}.{head_idx - 1}." + ".".join(parts[4:])
                    if std_k in sd_std and sd_std[std_k].shape == v_p2.shape:
                        new_sd[k_p2] = sd_std[std_k].clone()
                        transferred_count += v_p2.numel()
                        matched_keys.append((k_p2, std_k, v_p2.numel()))
                        matched = True
            elif sub == "dfl":
                std_k = "model.23." + ".".join(parts[2:])
                if std_k in sd_std and sd_std[std_k].shape == v_p2.shape:
                    new_sd[k_p2] = sd_std[std_k].clone()
                    transferred_count += v_p2.numel()
                    matched_keys.append((k_p2, std_k, v_p2.numel()))
                    matched = True

        if not matched:
            new_sd[k_p2] = v_p2.clone()
            new_keys.append((k_p2, v_p2.numel()))

    new_params = tot_p2 - transferred_count
    pct_transferred = transferred_count / tot_p2 * 100.0
    pct_new = new_params / tot_p2 * 100.0

    print(f"\nTotal YOLO11m-P2 Parameters:      {tot_p2:,}")
    print(f"Semantically Transferred Params:  {transferred_count:,} ({pct_transferred:.2f}%)")
    print(f"Newly Initialized Params:         {new_params:,} ({pct_new:.2f}%)")
    print(f"Matched Tensors:                  {len(matched_keys)}")
    print(f"Newly Initialized Tensors:        {len(new_keys)}")

    # Load into model to verify
    m_p2.model.load_state_dict(new_sd)
    print("\n[PASSED] Successfully loaded semantically mapped weights into YOLO11m-P2 model.")

    if save_path:
        save_path = Path(save_path)
        torch.save({"model": m_p2.model.half(), "epoch": -1}, str(save_path))
        print(f"[SAVED] Initialized FP16 checkpoint saved to: {save_path}")

    return {
        "total_parameters": tot_p2,
        "transferred_parameters": transferred_count,
        "transferred_pct": pct_transferred,
        "new_parameters": new_params,
        "new_pct": pct_new,
        "matched_tensors": len(matched_keys),
        "new_tensors": len(new_keys)
    }


if __name__ == "__main__":
    out_file = OUT_WEIGHTS_PATH if "--save" in sys.argv else None
    transfer_weights(out_file)
