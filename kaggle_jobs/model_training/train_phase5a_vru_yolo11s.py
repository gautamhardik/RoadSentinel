"""
RoadSentinel Phase 5A — Candidate 1: Targeted Small-VRU Fine-Tuning
Model: YOLO11s
Dataset: BDD100K Train Split (70,000 images)
Validation: BDD100K Val Split (10,000 images) [Quarantined Test: 0 images accessed]
Resolution: 960x960
Duration: 8 Epochs (Controlled Fine-Tuning)
Target: Small VRU Recall >= 45.0%, VRU F1 >= 0.6500, Precision >= 67.03%
"""

import os
import sys
import time
import json
import hashlib
import platform
import subprocess
from pathlib import Path

# Ensure ultralytics is installed in Kaggle environment
try:
    import ultralytics
except ImportError:
    print("Installing ultralytics in Kaggle environment...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
    import ultralytics

import yaml
import torch
from ultralytics import YOLO

EXPECTED_BASE_SHA256 = "a4747292648f37f68ab997bea65a36391c41356f35e309a9f3d2eb1d3ea8d8e3"

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()

def discover_kaggle_paths():
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        # Fallback to local repo paths if testing locally
        local_root = Path(__file__).resolve().parent.parent.parent
        ckpt_path = local_root / "weights" / "bdd100k_yolo11s.pt"
        data_yaml = local_root / "bdd100k_dataset" / "data.yaml"
        return ckpt_path, data_yaml

    # 1. Discover checkpoint
    ckpt_path = None
    for p in input_root.rglob("bdd100k_yolo11s.pt"):
        if p.is_file():
            ckpt_path = p
            break
    if ckpt_path is None:
        raise FileNotFoundError("Could not find bdd100k_yolo11s.pt in /kaggle/input")

    # 2. Discover dataset root
    train_img_dir = None
    val_img_dir = None
    for p in input_root.rglob("images"):
        if p.parent.name == "train" and p.is_dir():
            train_img_dir = p
        elif p.parent.name == "val" and p.is_dir():
            val_img_dir = p

    if train_img_dir is None or val_img_dir is None:
        raise FileNotFoundError("Could not locate BDD100K train and val image directories.")

    # Generate a temporary Kaggle data.yaml
    work_dir = Path("/kaggle/working")
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_yaml = work_dir / "bdd100k_kaggle.yaml"

    yaml_content = {
        "path": str(train_img_dir.parent.parent),
        "train": str(train_img_dir),
        "val": str(val_img_dir),
        "names": {
            0: "person",
            1: "rider",
            2: "car",
            3: "bus",
            4: "truck",
            5: "bike",
            6: "motor",
            7: "traffic light",
            8: "traffic sign",
            9: "train",
        }
    }
    with open(temp_yaml, "w") as f:
        yaml.dump(yaml_content, f)

    return ckpt_path, temp_yaml

def main():
    print("=" * 75)
    print("ROADSENTINEL PHASE 5A — TARGETED SMALL-VRU MODEL FINE-TUNING")
    print("=" * 75)
    t_start = time.time()

    # Hardware & Environment Profiling
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu_name})")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"Platform: {platform.platform()}")

    # Path Discovery & Base Weight Integrity
    ckpt_path, data_yaml_path = discover_kaggle_paths()
    print(f"\nBase Checkpoint: {ckpt_path}")
    base_sha256 = compute_sha256(ckpt_path)
    print(f"SHA-256: {base_sha256}")
    if base_sha256 != EXPECTED_BASE_SHA256:
        print(f"WARNING: Checkpoint SHA-256 does not match expected canonical hash!")
    else:
        print("Base Checkpoint SHA-256 verified successfully.")

    print(f"Data YAML: {data_yaml_path}")

    # Initialize YOLO Model from Checkpoint
    model = YOLO(str(ckpt_path))

    # Attach explicit VRU class weighting to loss function
    # Order matches data.yaml: 0:person, 1:rider, 2:car, 3:bus, 4:truck, 5:bike, 6:motor, 7:light, 8:sign, 9:train
    class_weights_tensor = torch.tensor([1.2, 2.0, 1.0, 1.0, 1.0, 1.8, 2.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    def attach_class_weights(trainer):
        trainer.model.class_weights = class_weights_tensor
        if hasattr(trainer, "loss") and hasattr(trainer.loss, "class_weights"):
            trainer.loss.class_weights = class_weights_tensor.to(trainer.device).view(1, 1, -1)
            print(f"\n[ACTIVE VRU WEIGHTING] Applied class weights to v8DetectionLoss:")
            class_names = ["person", "rider", "car", "bus", "truck", "bike", "motor", "t-light", "t-sign", "train"]
            for cid, (cname, w) in enumerate(zip(class_names, class_weights_tensor.tolist())):
                print(f"  Class {cid} ({cname}): weight={w:.1f}")
            sys.stdout.flush()

    model.add_callback("on_pretrain_routine_end", attach_class_weights)

    # Pre-registered Phase 5A Training Hyperparameters
    output_dir = Path("/kaggle/working/phase5a_training") if Path("/kaggle/working").exists() else Path("runs/phase5a_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nInitiating Phase 5A Controlled Fine-Tuning...")
    print("Configuration: 8 Epochs | imgsz=960 | AdamW lr0=0.0003 -> lrf=0.05 | Copy-Paste=0.15 | Mosaic=0.8 (close=2)")
    sys.stdout.flush()

    train_results = model.train(
        data=str(data_yaml_path),
        epochs=8,
        imgsz=960,
        batch=16,
        optimizer="AdamW",
        lr0=0.0003,
        lrf=0.05,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        seed=42,
        deterministic=True,
        # Augmentation tailored for small VRU representation
        mosaic=0.8,
        mixup=0.0,
        copy_paste=0.15,
        scale=0.4,
        close_mosaic=2,
        # Loss gains prioritizing classification accuracy
        cls=1.2,
        box=7.5,
        dfl=1.5,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=8,
        project=str(output_dir),
        name="yolo11s_vru_finetune",
        save=True,
        plots=True,
        verbose=True,
    )

    t_end = time.time()
    total_training_sec = t_end - t_start
    print(f"\nTraining completed in {total_training_sec / 3600.0:.2f} hours ({total_training_sec:.1f} s).")

    # Record Provenance & Checkpoint Hashes
    best_pt = output_dir / "yolo11s_vru_finetune" / "weights" / "best.pt"
    last_pt = output_dir / "yolo11s_vru_finetune" / "weights" / "last.pt"

    best_sha256 = compute_sha256(best_pt) if best_pt.exists() else None
    last_sha256 = compute_sha256(last_pt) if last_pt.exists() else None

    print(f"\nTrained Weights:")
    print(f"  best.pt: {best_pt} (SHA-256: {best_sha256})")
    print(f"  last.pt: {last_pt} (SHA-256: {last_sha256})")

    manifest = {
        "experiment": "RoadSentinel Phase 5A — Small-VRU Fine-Tuning",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_checkpoint_sha256": base_sha256,
        "best_checkpoint_sha256": best_sha256,
        "last_checkpoint_sha256": last_sha256,
        "training_duration_seconds": round(total_training_sec, 2),
        "device": gpu_name,
        "parameters": {
            "epochs": 8,
            "imgsz": 960,
            "batch": 16,
            "optimizer": "AdamW",
            "lr0": 0.0003,
            "lrf": 0.05,
            "copy_paste": 0.15,
            "mosaic": 0.8,
            "close_mosaic": 2,
            "cls_gain": 1.2,
        },
        "status": "TRAINING_COMPLETE"
    }

    manifest_path = output_dir / "training_run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved run manifest to {manifest_path}")

if __name__ == "__main__":
    main()
