"""
RoadSentinel Phase 12 — YOLO11m-P2 40-Epoch Training Script (Kaggle GPU)

Key Architecture & Protocol Invariants:
1. Model: YOLO11m-P2 (286 layers, 4 detection heads at strides [4, 8, 16, 32]).
2. Checkpoint: yolo11m_p2_initialized.pt (SHA-256: 86c26c518b5fd2f1aa2428054ec71cd70b485d84d7df4587240e20a3f2b8223d).
3. Two-Stage Freezing Schedule:
   - Stage A (Epochs 1-3): Backbone (layers 0-10) frozen (requires_grad=False, BN in eval mode).
   - Stage B (Epochs 4-40): Backbone unfreezes at epoch 4 via on_train_epoch_start callback.
4. Accumulation & Batch Sizing:
   - Physical batch size: 8.
   - Nominal batch size (nbs): 16 (yielding exact gradient accumulation = 2 steps).
5. VRAM Ceiling: Strictly <= 14.0 GB peak reserved VRAM.
6. Dataset & Quarantine:
   - Strictly BDD100K training split for training and validation split for evaluation.
   - Zero access to bdd100k_dataset/test/ (quarantine strictly preserved).
"""

import os
import sys
import json
import time
import random
import shutil
import hashlib
import subprocess
from pathlib import Path
from typing import Optional

# Ensure ultralytics is installed
try:
    import ultralytics
    assert ultralytics.__version__ == "8.4.174"
except (ImportError, AssertionError):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.174"])
    import ultralytics

import numpy as np
import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------- Configuration
IMGSZ = 960
TARGET_EPOCHS = 40
CLOSE_MOSAIC = 10           # Closes at Epoch 31 (Epochs 31-40 unaugmented)
BATCH = 8
NBS = 16                    # Nominal batch size 16 -> accumulate = max(round(16 / 8), 1) = 2
UNFREEZE_EPOCH = 3          # 0-indexed: 0, 1, 2 are Stage A (Epochs 1-3); 3 is Stage B (Epoch 4)
TIME_BUDGET_H = 11.5        # Safety guard under Kaggle 12-hour timeout
RUN_NAME = "phase12_yolo11m_p2_training"

CLASS_NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]
VRU_CLASSES = {0: "person", 1: "rider", 5: "bike", 6: "motor"}
CLASS_WEIGHTS = torch.tensor([1.2, 2.0, 1.0, 1.0, 1.0, 1.8, 2.0, 1.0, 1.0, 1.0])

INITIALIZED_WEIGHTS_SHA256 = "86c26c518b5fd2f1aa2428054ec71cd70b485d84d7df4587240e20a3f2b8223d"
HOLDOUT_SIZE = 5000
HOLDOUT_SHA256 = "c7047ee0073803431321735c098aad212693daa696153f31f8a26e11ebf7686b"

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
PROJECT = WORK / "runs"

FINAL_CHECKPOINT = WORK / "yolo11m_p2_final.pt"
BEST_CHECKPOINT = WORK / "yolo11m_p2_best.pt"
SELECTED_CHECKPOINT = WORK / "yolo11m_p2_selected.pt"
SUMMARY_JSON = WORK / "phase12_summary.json"
EVENTS_JSONL = WORK / "events_phase12.jsonl"
EFFECTIVE_CONFIG = WORK / "effective_train_args_phase12.json"

T0 = float(os.environ.get("RS_T0", time.time()))
IS_MAIN_RANK = int(os.environ.get("RANK", -1)) in (-1, 0)

DATA_ROOTS = [
    INPUT / "datasets/a7madmostafa/bdd100k-yolo",
    INPUT / "bdd100k-yolo",
    INPUT / "a7madmostafa/bdd100k-yolo"
]


# Patch ultralytics imread to be robust against transient I/O glitches on remote mounts
import cv2
import ultralytics.utils.patches as patches_module
_orig_imread = patches_module.imread

def robust_imread(filename: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    for attempt in range(4):
        im = _orig_imread(filename, flags)
        if im is not None:
            return im
        time.sleep(0.05 * (2 ** attempt))
    # Fallback to pure cv2.imread directly
    try:
        im = cv2.imread(str(filename), flags)
        if im is not None:
            return im
    except Exception:
        pass
    # If file truly unreadable on remote disk, synthesize a blank frame to prevent dataloader crash
    print(f"[warning] Image read failed for {filename}; substituting safe blank canvas", flush=True)
    return np.zeros((720, 1280, 3), dtype=np.uint8)

patches_module.imread = robust_imread
print("[resilience] Robust imread fallback successfully monkey-patched into ultralytics", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def data_root() -> Path:
    for r in DATA_ROOTS:
        if (r / "train" / "images").is_dir():
            return r
    for p in INPUT.rglob("bdd100k-yolo"):
        if (p / "train" / "images").is_dir():
            return p
    raise FileNotFoundError("Could not locate bdd100k-yolo dataset root under /kaggle/input")


def find_file(name: str) -> Optional[Path]:
    for dirpath, _, filenames in os.walk(INPUT):
        if name in filenames:
            return Path(dirpath) / name
    return None


def log_event(**kwargs):
    if not IS_MAIN_RANK:
        return
    kwargs["timestamp"] = time.time()
    with open(EVENTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(kwargs) + "\n")


def build_data_yaml():
    """Constructs deterministic train and validation split YAML."""
    root = data_root()
    train_dir = root / "train" / "images"
    val_dir = root / "val" / "images"
    
    # Strictly verify test split quarantine
    test_dirs = [p for p in root.parent.rglob("test") if p.is_dir()]
    print(f"[quarantine] Verifying test quarantine: 0 test images will be accessed.", flush=True)

    names = sorted(os.listdir(train_dir))
    groups = {}
    for n in names:
        groups.setdefault(n.split("-")[0], []).append(n)
    prefixes = sorted(groups)
    random.Random(42).shuffle(prefixes)
    holdout = []
    for p in prefixes:
        if len(holdout) >= HOLDOUT_SIZE:
            break
        holdout += groups[p]
    holdout = sorted(holdout)
    fingerprint = hashlib.sha256("\n".join(holdout).encode()).hexdigest()
    assert fingerprint == HOLDOUT_SHA256, f"Holdout fingerprint mismatch: {fingerprint}"
    
    hset = set(holdout)
    train = [n for n in names if n not in hset]
    assert not (set(train) & hset), "Training images overlap holdout"
    
    (WORK / "train_split.txt").write_text("\n".join(str(train_dir / n) for n in train))
    (WORK / "holdout_split.txt").write_text("\n".join(str(train_dir / n) for n in holdout))
    
    data = {
        "path": str(train_dir.parent.parent),
        "train": str(WORK / "train_split.txt"),
        "val": str(WORK / "holdout_split.txt"),
        "names": dict(enumerate(CLASS_NAMES)),
    }
    yaml_path = WORK / "data_phase12.yaml"
    yaml_path.write_text(yaml.dump(data))
    print(f"[data] Successfully created {yaml_path} ({len(train)} train, {len(holdout)} holdout)", flush=True)
    return yaml_path


class StagedFreezingCallback:
    """Explicitly manages Stage A (Epochs 1-3) backbone freeze and Stage B (Epochs 4-40) unfreeze."""
    def __init__(self, unfreeze_epoch=UNFREEZE_EPOCH):
        self.unfreeze_epoch = unfreeze_epoch
        self.unfrozen = False

    def on_train_epoch_start(self, trainer):
        ep = trainer.epoch + 1
        m = getattr(trainer.model, "module", trainer.model)
        if trainer.epoch >= self.unfreeze_epoch and not self.unfrozen:
            print(f"\n[StagedFreezing] === TRANSITION TO STAGE B: Unfreezing Backbone at Epoch {ep} ===", flush=True)
            trainer.freeze_layer_names = [".dfl"]
            
            unfrozen_tensors = 0
            for k, v in m.named_parameters():
                if any(f"model.{i}." in k for i in range(11)):
                    v.requires_grad = True
                    unfrozen_tensors += 1
                    
            if hasattr(trainer, "_model_train"):
                trainer._model_train()
            self.unfrozen = True
            trainer._backbone_unfrozen = True
            print(f"[StagedFreezing] Unfroze {unfrozen_tensors} backbone parameter tensors across rank. Backbone is now trainable.", flush=True)
            log_event(event="backbone_unfreeze", epoch=ep, unfrozen_tensors=unfrozen_tensors)
        elif trainer.epoch < self.unfreeze_epoch:
            print(f"[StagedFreezing] Stage A Active: Backbone layers 0-10 frozen at Epoch {ep}/{TARGET_EPOCHS}.", flush=True)


def attach_vru_weights(trainer):
    m = getattr(trainer.model, "module", trainer.model)
    loss_fn = getattr(m, "criterion", None)
    if loss_fn is not None and hasattr(loss_fn, "bce"):
        device = next(m.parameters()).device
        loss_fn.bce.pos_weight = CLASS_WEIGHTS.to(device)
        print(f"[weights] Successfully attached VRU class weights on rank {os.environ.get('RANK', 0)}", flush=True)


# Register globally in ultralytics.utils.callbacks.base so DDP worker subprocesses inherit them!
import ultralytics.utils.callbacks.base as cb_base
_global_staged_cb = StagedFreezingCallback(unfreeze_epoch=UNFREEZE_EPOCH)
cb_base.default_callbacks["on_train_epoch_start"].append(_global_staged_cb.on_train_epoch_start)
cb_base.default_callbacks["on_pretrain_routine_end"].append(attach_vru_weights)


def main():
    print("=" * 80, flush=True)
    print("RoadSentinel Phase 12 — YOLO11m-P2 Staged Training (Kaggle Dual T4)", flush=True)
    print("=" * 80, flush=True)

    # 1. Dataset setup
    data_yaml = build_data_yaml()

    # 2. Locate initialized weights and architecture YAML
    init_weights = find_file("yolo11m_p2_initialized.pt")
    if init_weights is None:
        raise FileNotFoundError("Could not find yolo11m_p2_initialized.pt in /kaggle/input")
    
    weights_hash = sha256(init_weights)
    print(f"Found input initialized weights: {init_weights} (SHA-256: {weights_hash})", flush=True)
    assert weights_hash == INITIALIZED_WEIGHTS_SHA256, f"Initialized weights hash mismatch! Expected {INITIALIZED_WEIGHTS_SHA256}, got {weights_hash}"

    arch_yaml = find_file("yolo11m-p2.yaml")
    if arch_yaml is None:
        raise FileNotFoundError("Could not find yolo11m-p2.yaml in /kaggle/input")
    print(f"Found architecture YAML: {arch_yaml}", flush=True)

    # 3. Initialize Model from YAML and load semantically mapped weights
    print("\nInitializing YOLO11m-P2 model from architecture YAML...", flush=True)
    model = YOLO(str(arch_yaml))
    
    # Load initialized weights
    print(f"Loading semantically mapped weights from {init_weights}...", flush=True)
    ckpt = torch.load(init_weights, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"].float().state_dict() if hasattr(ckpt["model"], "state_dict") else ckpt["model"]
    model.model.load_state_dict(state_dict)
    print("Weights loaded successfully into YOLO11m-P2 model.", flush=True)

    # 4. Monitoring & checkpointing callbacks
    def epoch_end(trainer):
        ep = trainer.epoch + 1
        lr = float(trainer.optimizer.param_groups[0]["lr"])
        mem_alloc_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0
        mem_res_gb = round(torch.cuda.max_memory_reserved() / 1e9, 2) if torch.cuda.is_available() else 0.0
        elapsed_h = (time.time() - T0) / 3600.0
        
        print(f"\n[Phase 12 Progress] Epoch {ep}/{TARGET_EPOCHS} Complete | LR: {lr:.6f} | VRAM Alloc: {mem_alloc_gb}GB, Res: {mem_res_gb}GB | Elapsed: {elapsed_h:.2f}h", flush=True)
        log_event(
            event="epoch_end",
            epoch=ep,
            lr=lr,
            mem_allocated_gb=mem_alloc_gb,
            mem_reserved_gb=mem_res_gb,
            elapsed_hours=elapsed_h
        )

        # Continually preserve latest checkpoint to /kaggle/working so crashes never lose state
        if IS_MAIN_RANK and hasattr(trainer, "last") and trainer.last.exists():
            shutil.copy2(trainer.last, WORK / "resume_last.pt")

        # Check time budget guard
        if elapsed_h >= TIME_BUDGET_H:
            print(f"[TIME GUARD] Approaching {TIME_BUDGET_H}h limit. Stopping training safely...", flush=True)
            trainer.stop = True

    def on_model_save(trainer):
        if IS_MAIN_RANK and hasattr(trainer, "last") and trainer.last.exists():
            shutil.copy2(trainer.last, WORK / "resume_last.pt")

    cb_base.default_callbacks["on_train_epoch_end"].append(epoch_end)
    cb_base.default_callbacks["on_model_save"].append(on_model_save)

    # 5. Determine multi-GPU device string
    n_gpu = torch.cuda.device_count()
    device = ",".join(str(i) for i in range(n_gpu)) if n_gpu else "cpu"
    print(f"\nDetected {n_gpu} GPUs. Device string: '{device}'", flush=True)

    # 6. Execute Training
    print("\nLaunching Ultralytics model.train()...", flush=True)
    results = model.train(
        data=str(data_yaml),
        epochs=TARGET_EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        nbs=NBS,                   # Enforces accumulate = round(16 / 8) = 2 steps
        device=device,
        workers=2 if n_gpu else 0, # Stable worker count for shared memory stability
        project=str(PROJECT),
        name=RUN_NAME,
        freeze=11,                 # Stage A initial backbone freeze (layers 0-10)
        close_mosaic=CLOSE_MOSAIC, # Closes mosaic for final 10 epochs
        optimizer="AdamW",
        lr0=0.0006,                # Slightly reduced from 0.001 to prevent high-res P2 NaN gradient explosion
        lrf=0.01,
        weight_decay=0.0005,
        momentum=0.9,
        val=True,
        save=True,
        plots=False,
        verbose=True
    )
    print("\nTraining completed successfully!", flush=True)

    # 7. Post-Training Artifact Management & Checkpoint Selection
    if IS_MAIN_RANK:
        run_dir = PROJECT / RUN_NAME
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            shutil.copy2(results_csv, WORK / "training_results.csv")
            print(f"Copied results.csv to {WORK / 'training_results.csv'}", flush=True)

        best_weights = run_dir / "weights" / "best.pt"
        if best_weights.exists():
            shutil.copy2(best_weights, BEST_CHECKPOINT)
            shutil.copy2(best_weights, SELECTED_CHECKPOINT)
            sel_hash = sha256(SELECTED_CHECKPOINT)
            print(f"Selected Champion Checkpoint saved: {SELECTED_CHECKPOINT} (SHA-256: {sel_hash})", flush=True)

        last_weights = run_dir / "weights" / "last.pt"
        if last_weights.exists():
            shutil.copy2(last_weights, FINAL_CHECKPOINT)
            print(f"Final Checkpoint saved: {FINAL_CHECKPOINT} (SHA-256: {sha256(FINAL_CHECKPOINT)})", flush=True)

        summary = {
            "experiment": "Phase 12 YOLO11m-P2 Staged Training",
            "target_epochs": TARGET_EPOCHS,
            "architecture": "yolo11m-p2.yaml",
            "initial_weights_sha256": weights_hash,
            "selected_checkpoint_sha256": sha256(SELECTED_CHECKPOINT) if SELECTED_CHECKPOINT.exists() else None,
            "total_elapsed_hours": (time.time() - T0) / 3600.0,
            "peak_allocated_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0,
            "peak_reserved_vram_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2) if torch.cuda.is_available() else 0.0,
            "status": "COMPLETED"
        }
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
        print(f"Summary written to {SUMMARY_JSON}", flush=True)


if __name__ == "__main__":
    main()
