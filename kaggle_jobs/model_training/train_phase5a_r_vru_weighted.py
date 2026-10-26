"""
RoadSentinel Phase 5A-R — Corrected Class-Weighted VRU Fine-Tuning
Model: YOLO11s (base: weights/bdd100k_yolo11s.pt)
Dataset: BDD100K Train Split (70,000 images) | Validation: 10,000 images [Quarantined Test: 0 images]
Resolution: 960x960 | Epochs: 8 | Optimizer: AdamW lr0=0.0003 -> lrf=0.05
Augmentation: Mosaic=0.80 (close=2), Scale=0.40, Fliplr=0.50 (CopyPaste explicitly disabled: 0.0)
Class Weights (in data.yaml order):
  [person=1.2, rider=2.0, car=1.0, bus=1.0, truck=1.0, bike=1.8, motor=2.0, t-light=1.0, t-sign=1.0, train=1.0]
Runtime Proof: Fail-fast assertions enforce criterion.class_weights activation on every epoch.
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
TARGET_CLASS_WEIGHTS = torch.tensor([1.2, 2.0, 1.0, 1.0, 1.0, 1.8, 2.0, 1.0, 1.0, 1.0], dtype=torch.float32)
CLASS_NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def discover_kaggle_paths():
    input_root = Path("/kaggle/input")
    if not input_root.exists():
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

    # Generate Kaggle data.yaml
    work_dir = Path("/kaggle/working")
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_yaml = work_dir / "bdd100k_kaggle_phase5ar.yaml"

    yaml_content = {
        "path": str(train_img_dir.parent.parent),
        "train": str(train_img_dir),
        "val": str(val_img_dir),
        "names": {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    with open(temp_yaml, "w") as f:
        yaml.dump(yaml_content, f)

    return ckpt_path, temp_yaml


def main():
    print("=" * 80)
    print("ROADSENTINEL PHASE 5A-R — CORRECTED CLASS-WEIGHTED VRU FINE-TUNING")
    print("=" * 80)
    t_start = time.time()

    # Hardware & Telemetry
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu_name})")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"Platform: {platform.platform()}")

    # Checkpoint Integrity
    ckpt_path, data_yaml_path = discover_kaggle_paths()
    print(f"\nBase Checkpoint: {ckpt_path}")
    base_sha256 = compute_sha256(ckpt_path)
    print(f"SHA-256: {base_sha256}")
    assert base_sha256 == EXPECTED_BASE_SHA256, (
        f"FATAL: Checkpoint SHA-256 mismatch! Expected {EXPECTED_BASE_SHA256}, got {base_sha256}"
    )
    print("Base Checkpoint SHA-256 verified successfully.")

    # Load Model
    model = YOLO(str(ckpt_path))

    # Instrument Fail-Fast Callbacks for Runtime Loss-Weight Enforcement
    def enforce_loss_weights_pretrain(trainer):
        # 1. Attach to trainer.model (DetectionModel)
        m = getattr(trainer.model, "module", trainer.model)
        m.class_weights = TARGET_CLASS_WEIGHTS.to(trainer.device)
        if getattr(m, "criterion", None) is None:
            m.criterion = m.init_criterion()
        else:
            m.criterion.class_weights = TARGET_CLASS_WEIGHTS.to(trainer.device).view(1, 1, -1)
        if hasattr(trainer, "loss") and hasattr(trainer.loss, "class_weights"):
            trainer.loss.class_weights = TARGET_CLASS_WEIGHTS.to(trainer.device).view(1, 1, -1)

        print("\n[HOOK: on_pretrain_routine_end] Attached target class weights and initialized criterion:")
        for cid, (cname, w) in enumerate(zip(CLASS_NAMES, TARGET_CLASS_WEIGHTS.tolist())):
            print(f"  Class {cid:02d} ({cname:<14}): weight = {w:.1f}")
        sys.stdout.flush()

    def assert_active_loss_weights_epoch(trainer):
        # Verify criterion possesses active class_weights
        m = getattr(trainer.model, "module", trainer.model)
        criterion = getattr(m, "criterion", None) or getattr(trainer, "loss", None)
        assert criterion is not None, "FATAL: Criterion not found on trainer or model!"
        assert criterion.class_weights is not None, "FATAL: criterion.class_weights is None during training!"
        assert criterion.class_weights.shape == (1, 1, 10), (
            f"FATAL: Unexpected class_weights shape: {criterion.class_weights.shape}"
        )
        cw_cpu = criterion.class_weights.squeeze().cpu()
        assert torch.allclose(cw_cpu, TARGET_CLASS_WEIGHTS, atol=1e-4), (
            f"FATAL: criterion.class_weights tensor mismatch! Got {cw_cpu.tolist()}"
        )
        print(f"\n[RUNTIME ASSERTION PASSED - Epoch {trainer.epoch + 1}/{trainer.epochs}] Active criterion.class_weights verified on {trainer.device}.")
        sys.stdout.flush()

    model.add_callback("on_pretrain_routine_end", enforce_loss_weights_pretrain)
    model.add_callback("on_train_epoch_start", assert_active_loss_weights_epoch)

    output_dir = Path("/kaggle/working/phase5a_r_training") if Path("/kaggle/working").exists() else Path("runs/phase5a_r_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nInitiating Phase 5A-R Controlled Class-Weighted Fine-Tuning...")
    print("Hyperparameters: 8 Epochs | imgsz=960 | AdamW lr0=0.0003 -> lrf=0.05 | Mosaic=0.8 (close=2) | Scale=0.4 | CopyPaste=0.0")
    sys.stdout.flush()

    # Train
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
        # Multi-scale augmentation without ineffective copy-paste
        mosaic=0.8,
        mixup=0.0,
        copy_paste=0.0,
        scale=0.4,
        close_mosaic=2,
        # Loss gains
        cls=1.2,
        box=7.5,
        dfl=1.5,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=8,
        project=str(output_dir),
        name="yolo11s_vru_weighted_finetune",
        save=True,
        plots=True,
        verbose=True,
    )

    t_end = time.time()
    total_training_sec = t_end - t_start
    print(f"\nTraining completed in {total_training_sec / 3600.0:.2f} hours ({total_training_sec:.1f} s).")

    # Record Provenance & Checkpoint Hashes
    best_pt = output_dir / "yolo11s_vru_weighted_finetune" / "weights" / "best.pt"
    last_pt = output_dir / "yolo11s_vru_weighted_finetune" / "weights" / "last.pt"

    best_sha256 = compute_sha256(best_pt) if best_pt.exists() else None
    last_sha256 = compute_sha256(last_pt) if last_pt.exists() else None

    print(f"\nTrained Weights:")
    print(f"  best.pt: {best_pt} (SHA-256: {best_sha256})")
    print(f"  last.pt: {last_pt} (SHA-256: {last_sha256})")

    manifest = {
        "experiment": "RoadSentinel Phase 5A-R — Corrected Class-Weighted VRU Fine-Tuning",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_checkpoint_sha256": base_sha256,
        "best_checkpoint_sha256": best_sha256,
        "last_checkpoint_sha256": last_sha256,
        "training_duration_seconds": round(total_training_sec, 2),
        "device": gpu_name,
        "target_class_weights": {cname: w for cname, w in zip(CLASS_NAMES, TARGET_CLASS_WEIGHTS.tolist())},
        "parameters": {
            "epochs": 8,
            "imgsz": 960,
            "batch": 16,
            "optimizer": "AdamW",
            "lr0": 0.0003,
            "lrf": 0.05,
            "copy_paste": 0.0,
            "mosaic": 0.8,
            "close_mosaic": 2,
            "scale": 0.4,
            "cls_gain": 1.2,
        },
        "runtime_loss_weighting_verified": True,
        "status": "TRAINING_COMPLETE"
    }

    manifest_path = output_dir / "training_run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved run manifest to {manifest_path}")


if __name__ == "__main__":
    main()
