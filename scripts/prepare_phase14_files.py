"""
RoadSentinel — Phase 14 Precision Lock Preparation Script
File: scripts/prepare_phase14_files.py

Sets up:
  1. kaggle_jobs/phase14_precision_lock/dataset_upload/
     - phase13_best.pt (100% weights for Step 0 transfer from 61.01% mAP50 / 56.61% Recall)
     - train_phase14_balanced.txt (28,554 VRU-positive + 12,000 non-VRU hard-negative background frames = 40,554 frames)
     - holdout_split.txt (exact 5,000 holdout split)
     - ultralytics-8.4.174-py3-none-any.whl (for offline pip install)
     - dataset-metadata.json
  2. kaggle_jobs/phase14_precision_lock/kernel/
     - train_phase14_precision_lock.py
     - kernel-metadata.json
"""

import os
import shutil
import random
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Paths
P13_BEST_LOCAL = REPO_ROOT / "weights" / "phase13_best.pt"
P13_DATASET_DIR = REPO_ROOT / "kaggle_jobs" / "phase13_vru_polish" / "dataset_upload"
TRAIN_SPLIT_FILE = REPO_ROOT / "experiments" / "phase6_session1" / "train_split.txt"
VRU_CURRICULUM_FILE = REPO_ROOT / "experiments" / "phase_13_output" / "train_vru_curriculum.txt"
HOLDOUT_FILE = REPO_ROOT / "experiments" / "phase_13_output" / "holdout_split.txt"

P14_DIR = REPO_ROOT / "kaggle_jobs" / "phase14_precision_lock"
P14_DATASET_DIR = P14_DIR / "dataset_upload"
P14_KERNEL_DIR = P14_DIR / "kernel"


def sha256_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("=" * 80)
    print("PREPARING PHASE 14 HARD-NEGATIVE PRECISION LOCK DATASET & KERNEL")
    print("=" * 80)

    P14_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    P14_KERNEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy base checkpoint: phase13_best.pt
    dest_p13 = P14_DATASET_DIR / "phase13_best.pt"
    if not dest_p13.exists() or dest_p13.stat().st_size != P13_BEST_LOCAL.stat().st_size:
        print(f"[copy] Copying {P13_BEST_LOCAL.name} -> {dest_p13}")
        shutil.copy2(P13_BEST_LOCAL, dest_p13)
    p13_hash = sha256_file(dest_p13)
    print(f"[check] phase13_best.pt SHA-256: {p13_hash}")

    # 2. Copy holdout_split.txt
    dest_holdout = P14_DATASET_DIR / "holdout_split.txt"
    shutil.copy2(HOLDOUT_FILE, dest_holdout)
    holdout_hash = sha256_file(dest_holdout)
    print(f"[check] holdout_split.txt SHA-256: {holdout_hash}")

    # 3. Copy ultralytics wheel
    wheel_src = P13_DATASET_DIR / "ultralytics-8.4.174-py3-none-any.whl"
    dest_wheel = P14_DATASET_DIR / "ultralytics-8.4.174-py3-none-any.whl"
    if wheel_src.exists() and not dest_wheel.exists():
        shutil.copy2(wheel_src, dest_wheel)
        print(f"[copy] Copied ultralytics wheel -> {dest_wheel.name}")

    # 4. Generate train_phase14_balanced.txt (28,554 VRU positive + 12,000 non-VRU hard negatives)
    with open(VRU_CURRICULUM_FILE, "r") as f:
        vru_lines = [l.strip() for l in f if l.strip()]

    with open(TRAIN_SPLIT_FILE, "r") as f:
        train_lines = [l.strip() for l in f if l.strip()]

    vru_basenames = {Path(p).name for p in vru_lines}
    non_vru_candidates = [p for p in train_lines if Path(p).name not in vru_basenames]

    random.seed(42)
    sampled_negatives = sorted(random.sample(non_vru_candidates, 12000))
    balanced_lines = sorted(vru_lines + sampled_negatives)

    dest_balanced = P14_DATASET_DIR / "train_phase14_balanced.txt"
    with open(dest_balanced, "w") as f:
        for p in balanced_lines:
            f.write(p + "\n")

    balanced_hash = sha256_file(dest_balanced)
    print(f"[curriculum] Balanced Curriculum Generated: {dest_balanced}")
    print(f"             VRU Positives   : {len(vru_lines):,}")
    print(f"             Hard Negatives  : {len(sampled_negatives):,}")
    print(f"             Total Frames    : {len(balanced_lines):,}")
    print(f"             SHA-256         : {balanced_hash}")

    # 5. Dataset metadata JSON
    dataset_meta = {
        "title": "roadsentinel-phase14-inputs",
        "id": "hardikgautam24119014/roadsentinel-phase14-inputs",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(P14_DATASET_DIR / "dataset-metadata.json", "w") as f:
        import json
        json.dump(dataset_meta, f, indent=2)
    print(f"[meta] dataset-metadata.json written for {dataset_meta['id']}")

    # 6. Kernel metadata JSON
    kernel_meta = {
        "id": "hardikgautam24119014/roadsentinel-phase-14-precision-lock",
        "title": "RoadSentinel Phase 14 Precision Lock",
        "code_file": "train_phase14_precision_lock.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "a7madmostafa/bdd100k-yolo",
            "hardikgautam24119014/roadsentinel-phase14-inputs"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    with open(P14_KERNEL_DIR / "kernel-metadata.json", "w") as f:
        import json
        json.dump(kernel_meta, f, indent=2)
    print(f"[meta] kernel-metadata.json written for {kernel_meta['id']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
