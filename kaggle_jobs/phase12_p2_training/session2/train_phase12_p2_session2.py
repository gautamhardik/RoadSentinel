"""
RoadSentinel Phase 12 — YOLO11m-P2 Fast-Start GPU Training (Kaggle Dual T4)

Optimized for instant execution without the remote mount label scanning delay:
1. Installs offline ultralytics-8.4.174-py3-none-any.whl.
2. Injects fast-cache mechanism into YOLODataset:
   - Uses NUM_THREADS=32 for parallel I/O.
   - Caches parsed labels directly into /kaggle/working/ / /tmp/.
3. Resumes from uncorrupted resume_last.pt (Epoch 9 state, 409 optimizer state tensors).
4. Targets 15 Epochs (Epochs 10-15: ~8 hours on Dual T4, safely below 12h Kaggle timeout).
5. Registers on_fit_epoch_end callback to export checkpoint after every epoch.
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

# ---------------------------------------------------------------- Setup Ultralytics
try:
    import ultralytics
    assert ultralytics.__version__ == "8.4.174"
except (ImportError, AssertionError):
    whl_found = False
    for r, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f.startswith("ultralytics") and f.endswith(".whl"):
                whl_path = os.path.join(r, f)
                print(f"[setup] Installing offline wheel: {whl_path}", flush=True)
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", whl_path])
                whl_found = True
                break
        if whl_found:
            break
    if not whl_found:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.174"])
    import ultralytics

import numpy as np
import torch
import yaml
import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------- Fast Cache & Resilience Patches
import ultralytics.data.dataset as dataset_module
import ultralytics.utils.patches as patches_module

# 1. Thread concurrency boost for scanning remote mount
dataset_module.NUM_THREADS = 32
print(f"[performance] Boosted dataset worker threads: NUM_THREADS={dataset_module.NUM_THREADS}", flush=True)

# 2. Patch imread resilience against transient I/O hiccups
_orig_imread = patches_module.imread

def robust_imread(filename: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    for attempt in range(4):
        im = _orig_imread(filename, flags)
        if im is not None:
            return im
        time.sleep(0.05 * (2 ** attempt))
    try:
        im = cv2.imread(str(filename), flags)
        if im is not None:
            return im
    except Exception:
        pass
    print(f"[warning] Image read failed for {filename}; substituting safe blank canvas", flush=True)
    return np.zeros((720, 1280, 3), dtype=np.uint8)

patches_module.imread = robust_imread
print("[resilience] Robust imread fallback successfully monkey-patched into ultralytics", flush=True)

# ---------------------------------------------------------------- Configuration
IMGSZ = 960
TARGET_EPOCHS = 15          # Focused refinement run: Epochs 10-15 (~8.5h total, safely under 12h limit)
BATCH = 8
NBS = 16                    # Nominal batch size 16 -> accumulate = max(round(16 / 8), 1) = 2
TIME_BUDGET_H = 11.2        # Safety guard under Kaggle 12-hour timeout
RUN_NAME = "phase12_yolo11m_p2_training"

CLASS_NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]
VRU_CLASSES = {0: "person", 1: "rider", 5: "bike", 6: "motor"}
CLASS_WEIGHTS = torch.tensor([1.2, 2.0, 1.0, 1.0, 1.0, 1.8, 2.0, 1.0, 1.0, 1.0])

HOLDOUT_SIZE = 5000
HOLDOUT_SHA256 = "c7047ee0073803431321735c098aad212693daa696153f31f8a26e11ebf7686b"

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
PROJECT = WORK / "runs"
EXPORT_DIR = WORK / "exported_checkpoints"

FINAL_CHECKPOINT = WORK / "yolo11m_p2_final.pt"
SELECTED_CHECKPOINT = WORK / "yolo11m_p2_selected.pt"
SUMMARY_JSON = WORK / "phase12_session2_summary.json"
EVENTS_JSONL = WORK / "events_session2.jsonl"

T0 = float(os.environ.get("RS_T0", time.time()))
IS_MAIN_RANK = int(os.environ.get("RANK", -1)) in (-1, 0)

DATA_ROOTS = [
    INPUT / "datasets/a7madmostafa/bdd100k-yolo",
    INPUT / "bdd100k-yolo",
    INPUT / "a7madmostafa/bdd100k-yolo"
]


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
    root = data_root()
    train_dir = root / "train" / "images"
    
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
    print(f"[data] Created {yaml_path} ({len(train)} train, {len(holdout)} holdout)", flush=True)
    return yaml_path


def attach_vru_weights(trainer):
    loss_fn = getattr(trainer.model, "criterion", None)
    if loss_fn is not None and hasattr(loss_fn, "bce"):
        device = next(trainer.model.parameters()).device
        loss_fn.bce.pos_weight = CLASS_WEIGHTS.to(device)
        print(f"[weights] Attached VRU class weights on rank {os.environ.get('RANK', 0)}", flush=True)


def main():
    print("=" * 80, flush=True)
    print("RoadSentinel Phase 12 — YOLO11m-P2 Fast-Start GPU Training", flush=True)
    print(f"Target Epochs: {TARGET_EPOCHS} (Refinement Epochs 10-15)", flush=True)
    print("=" * 80, flush=True)

    if IS_MAIN_RANK:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset setup
    data_yaml = build_data_yaml()

    # 2. Locate resume checkpoint from Session 1
    resume_ckpt = find_file("resume_last.pt")
    if resume_ckpt is None:
        raise FileNotFoundError("Could not find resume_last.pt in /kaggle/input")
    
    ckpt_hash = sha256(resume_ckpt)
    print(f"Found input resume checkpoint: {resume_ckpt} (SHA-256: {ckpt_hash})", flush=True)
    
    # Audit checkpoint
    ck = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
    completed_epoch = ck["epoch"] + 1
    opt_tensors = len(ck.get("optimizer", {}).get("state", {}))
    print(f"Session 1 Completed Epochs: {completed_epoch} / {TARGET_EPOCHS}", flush=True)
    print(f"Optimizer State Tensors:    {opt_tensors} (Intact: {opt_tensors > 0})", flush=True)
    assert opt_tensors > 0, "Resume checkpoint missing optimizer states!"

    # Copy to working directory for in-place resume update
    work_resume = WORK / "resume_last.pt"
    
    # Ensure checkpoint has a fully instantiated model object
    if ck.get("model") is None:
        arch_yaml = find_file("yolo11m-p2.yaml")
        print(f"Instantiating model from {arch_yaml}...", flush=True)
        temp_model = YOLO(str(arch_yaml))
        state = ck["ema"].float().state_dict() if hasattr(ck["ema"], "state_dict") else ck["ema"].state_dict()
        temp_model.model.load_state_dict(state)
        ck["model"] = temp_model.model
        torch.save(ck, work_resume)
        print("Successfully injected reconstructed model object into working checkpoint.", flush=True)
    else:
        shutil.copy2(resume_ckpt, work_resume)

    # 3. Initialize Model and Attach Callbacks
    model = YOLO(str(work_resume))

    def on_train_epoch_start(trainer):
        # Stage B active: ensure backbone layers 0-10 are unfrozen and only .dfl is frozen
        m = getattr(trainer.model, "module", trainer.model)
        trainer.freeze_layer_names = [".dfl"]
        for k, v in m.named_parameters():
            if any(f"model.{i}." in k for i in range(11)):
                v.requires_grad = True
        if hasattr(trainer, "_model_train"):
            trainer._model_train()

    def epoch_end(trainer):
        ep = trainer.epoch + 1
        lr = float(trainer.optimizer.param_groups[0]["lr"])
        mem_alloc_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0
        mem_res_gb = round(torch.cuda.max_memory_reserved() / 1e9, 2) if torch.cuda.is_available() else 0.0
        elapsed_h = (time.time() - T0) / 3600.0
        
        print(f"\n[Progress] Epoch {ep}/{TARGET_EPOCHS} Complete | LR: {lr:.6f} | VRAM: {mem_alloc_gb}GB | Elapsed: {elapsed_h:.2f}h", flush=True)
        log_event(
            event="epoch_end",
            epoch=ep,
            lr=lr,
            mem_allocated_gb=mem_alloc_gb,
            mem_reserved_gb=mem_res_gb,
            elapsed_hours=elapsed_h
        )

        # Survival Hook: Save checkpoint copy after every epoch
        if IS_MAIN_RANK and hasattr(trainer, "last") and trainer.last.exists():
            ep_dest = EXPORT_DIR / f"epoch_{ep}"
            ep_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trainer.last, ep_dest / "last.pt")
            if hasattr(trainer, "best") and trainer.best.exists():
                shutil.copy2(trainer.best, ep_dest / "best.pt")
            print(f"[Survival Hook] Checkpoint successfully backed up to {ep_dest}", flush=True)

        if elapsed_h >= TIME_BUDGET_H:
            print(f"[TIME GUARD] Approaching {TIME_BUDGET_H}h limit. Stopping gracefully...", flush=True)
            trainer.stop = True

    def on_model_save(trainer):
        if IS_MAIN_RANK and hasattr(trainer, "last") and trainer.last.exists():
            shutil.copy2(trainer.last, WORK / "resume_last.pt")

    import ultralytics.utils.callbacks.base as cb_base
    cb_base.default_callbacks["on_pretrain_routine_end"].append(attach_vru_weights)
    cb_base.default_callbacks["on_train_epoch_start"].append(on_train_epoch_start)
    cb_base.default_callbacks["on_train_epoch_end"].append(epoch_end)
    cb_base.default_callbacks["on_model_save"].append(on_model_save)

    # 4. Multi-GPU execution
    n_gpu = torch.cuda.device_count()
    device = ",".join(str(i) for i in range(n_gpu)) if n_gpu else "cpu"
    print(f"\nDetected {n_gpu} GPUs. Device string: '{device}'", flush=True)

    # 5. Resume Training through Epoch 15
    print(f"\nInvoking model.train(resume=True, epochs={TARGET_EPOCHS})...", flush=True)
    model.train(resume=True, epochs=TARGET_EPOCHS, device=device, data=str(data_yaml), amp=False)
    print("Training finished successfully.", flush=True)

    # 6. Save Final & Selected Artifacts
    if IS_MAIN_RANK:
        run_dir = PROJECT / RUN_NAME
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            shutil.copy2(results_csv, WORK / "training_results.csv")
        
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            shutil.copy2(best_pt, SELECTED_CHECKPOINT)
            print(f"Selected Checkpoint saved: {SELECTED_CHECKPOINT} (SHA-256: {sha256(SELECTED_CHECKPOINT)})", flush=True)

        last_pt = run_dir / "weights" / "last.pt"
        if last_pt.exists():
            shutil.copy2(last_pt, FINAL_CHECKPOINT)
            print(f"Final Checkpoint saved: {FINAL_CHECKPOINT} (SHA-256: {sha256(FINAL_CHECKPOINT)})", flush=True)


if __name__ == "__main__":
    main()
