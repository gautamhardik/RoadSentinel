"""
RoadSentinel — Decoupled-Head Surgical Grafting Champion Builder
File: scripts/build_decoupled_surgical_champion.py

Architecture Context (YOLO11m Decoupled Detection Head):
- Layers 0..22: Backbone, Feature Pyramid Neck, and C2PSA Spatial Attention.
- Layer 23 (Detect):
    - cv2 (Box Regression / DFL localization branch): Preserves Phase 13's high-fidelity
      bounding box regression achieved via native-frame training and box=8.5 loss.
    - cv3 (Classification logit branch): Blends Session 4's strong hard-negative suppression
      bias with Phase 13's high-recall VRU class representations.
    - dfl (Distribution Focal Loss projection): Integral bin conv.

Blending Specification:
  - Backbone, Neck, and Box Regression (Layers 0..22, model.23.dfl.*, model.23.cv2.*):
      W_graft = 0.92 * W_phase13 + 0.08 * W_session4
  - Classification Head (model.23.cv3.*):
      W_graft = 0.68 * W_phase13 + 0.32 * W_session4
  - Non-floating tensors (e.g. num_batches_tracked): Retained from Phase 13.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import copy
import hashlib
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE13_PATH = REPO_ROOT / "weights" / "phase13_best.pt"
SESSION4_PATH = REPO_ROOT / "weights" / "session4_best.pt"
OUTPUT_PATH = REPO_ROOT / "weights" / "phase13_surgical_champion.pt"

# Blending ratios
ALPHA_BACKBONE_BOX = 0.92  # 0.92 * Phase13 + 0.08 * Session4
ALPHA_CLASSIFIER = 0.68    # 0.68 * Phase13 + 0.32 * Session4


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_surgical_champion(
    p13_path: Path = PHASE13_PATH,
    s4_path: Path = SESSION4_PATH,
    out_path: Path = OUTPUT_PATH,
    alpha_box: float = ALPHA_BACKBONE_BOX,
    alpha_cls: float = ALPHA_CLASSIFIER,
) -> Path:
    print("=" * 80)
    print("ROADSENTINEL: DECOUPLED-HEAD SURGICAL GRAFTING BUILDER")
    print("=" * 80)
    print(f"[input 1] Phase 13 Checkpoint : {p13_path}")
    print(f"          SHA-256             : {compute_sha256(p13_path)}")
    print(f"[input 2] Session 4 Checkpoint: {s4_path}")
    print(f"          SHA-256             : {compute_sha256(s4_path)}")
    print(f"[config]  Backbone/Neck/Box   : {alpha_box:.2f} * P13 + {1.0 - alpha_box:.2f} * S4")
    print(f"[config]  Classification Head : {alpha_cls:.2f} * P13 + {1.0 - alpha_cls:.2f} * S4")
    print("-" * 80)

    ckpt_p13 = torch.load(p13_path, map_location="cpu", weights_only=False)
    ckpt_s4 = torch.load(s4_path, map_location="cpu", weights_only=False)

    model_p13 = ckpt_p13["model"].float()
    model_s4 = ckpt_s4["model"].float()

    champion_model = copy.deepcopy(model_p13)
    state_p13 = model_p13.state_dict()
    state_s4 = model_s4.state_dict()
    champion_state = champion_model.state_dict()

    cv2_count = 0
    cv3_count = 0
    backbone_neck_count = 0
    dfl_count = 0
    tracked_count = 0

    for key in state_p13.keys():
        if key not in state_s4:
            print(f"[warning] Key {key} missing from Session 4; keeping Phase 13.")
            champion_state[key] = state_p13[key]
            continue

        t_p13 = state_p13[key]
        t_s4 = state_s4[key]

        if not t_p13.is_floating_point():
            # Integer buffer like num_batches_tracked
            champion_state[key] = t_p13
            tracked_count += 1
            continue

        if "23.cv3." in key:
            # Classification Head: Restore hard-negative FP rejection bias
            champion_state[key] = alpha_cls * t_p13 + (1.0 - alpha_cls) * t_s4
            cv3_count += 1
        elif "23.cv2." in key:
            # Box Regression / Localization Head: Retain high-precision localization
            champion_state[key] = alpha_box * t_p13 + (1.0 - alpha_box) * t_s4
            cv2_count += 1
        elif "23.dfl." in key:
            champion_state[key] = alpha_box * t_p13 + (1.0 - alpha_box) * t_s4
            dfl_count += 1
        else:
            # Backbone / Neck / C2PSA (Layers 0..22)
            champion_state[key] = alpha_box * t_p13 + (1.0 - alpha_box) * t_s4
            backbone_neck_count += 1

    champion_model.load_state_dict(champion_state)

    champion_ckpt = copy.deepcopy(ckpt_p13)
    champion_ckpt["model"] = champion_model
    # Clear any stale optimizer or epoch metadata
    champion_ckpt.pop("optimizer", None)
    champion_ckpt.pop("ema", None)
    champion_ckpt["epoch"] = -1
    champion_ckpt["surgical_grafting"] = {
        "alpha_backbone_box": alpha_box,
        "alpha_classifier": alpha_cls,
        "parent_phase13_sha256": compute_sha256(p13_path),
        "parent_session4_sha256": compute_sha256(s4_path),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(champion_ckpt, out_path)

    out_hash = compute_sha256(out_path)
    file_size_mb = out_path.stat().st_size / (1024 * 1024)

    print(f"[graft] Grafting Summary:")
    print(f"        Backbone/Neck Tensors Blended : {backbone_neck_count} (alpha={alpha_box:.2f})")
    print(f"        Box Reg (cv2) Tensors Blended : {cv2_count} (alpha={alpha_box:.2f})")
    print(f"        DFL Tensors Blended          : {dfl_count} (alpha={alpha_box:.2f})")
    print(f"        Classification Tensors (cv3) : {cv3_count} (alpha={alpha_cls:.2f})")
    print(f"        Tracked Int Buffers Retained : {tracked_count}")
    print(f"        Total Model Tensors Processed: {len(state_p13)}")
    print("-" * 80)
    print(f"[output] Successfully created: {out_path}")
    print(f"         Size   : {file_size_mb:.2f} MB")
    print(f"         SHA-256: {out_hash}")
    print("=" * 80)

    return out_path


if __name__ == "__main__":
    build_surgical_champion()
