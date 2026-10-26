"""
RoadSentinel Phase 14 — Hard-Negative Precision Lock Training Script
Base Checkpoint: phase13_best.pt (SHA-256: bbac96e571881b294c96020756ae59140bc72a97229d1d42d703904bb61354ce)
Target: phase14_best.pt (mAP50 > 63% | Precision > 76% | Recall > 56%) on Kaggle Dual Tesla T4 GPUs

Core Engineering Architecture:
1. Base Initialization: 100.0% weight transfer from phase13_best.pt (starts Epoch 1 Step 0 at 61.01% mAP50 / 56.61% Recall).
2. Hard-Negative Balanced Curriculum: 40,554 frames (28,554 VRU-positive frames + 12,000 non-VRU hard-negative background frames).
3. Validation: Strict holdout on exact 5,000-image holdout_split.txt.
4. Hyperparameters:
   - epochs = 8
   - imgsz = 1024 (P3 stride-8 feature map expands from 120x120 to 128x128, +14% pixels per pedestrian)
   - batch = 16 across device=[0, 1] Dual T4 DDP
   - optimizer = "AdamW", lr0 = 0.0001 (1e-4 ultra-gentle polish), lrf = 0.05, cos_lr = True, warmup_epochs = 0.5
   - mosaic = 0.0 (100% undistorted native road frames for all 8 epochs)
   - box = 9.0 (tight IoU localization loss)
   - cls = 0.45 (penalizes false-positive activations on trees/poles/signs to recover precision)
   - dfl = 1.8
5. Fast-Cache Hook: 32 worker threads streaming to /tmp/bdd100k_fast/ NVMe.
6. Survival Callback: Saves best.pt, last.pt, results.csv, and SHA-256 hashes after every epoch.
"""

import os
import sys
import json
import time
import shutil
import hashlib
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

# Step 0: Ensure ultralytics is installed
try:
    import ultralytics
    print(f"[setup] Ultralytics {ultralytics.__version__} already installed.", flush=True)
except ImportError:
    whl_candidates = list(Path("/kaggle/input").rglob("ultralytics*.whl"))
    if whl_candidates:
        print(f"[setup] Installing local wheel: {whl_candidates[0]}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", str(whl_candidates[0])])
    else:
        print("[setup] Installing ultralytics via pip...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.174"])
    import ultralytics

import numpy as np
import torch
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------- Hyperparameter Configuration
IMGSZ = 1024                  # 1024px expands P3 stride-8 grid to 128x128 (+14% pixels for small VRUs)
TARGET_EPOCHS = 8             # 8 epochs (~3.5 hours on Dual T4)
BATCH = 16                   # Dual T4 DDP: 8 per GPU
LR0 = 0.0001                 # 1e-4 ultra-gentle learning rate prevents catastrophic forgetting
LRF = 0.05                   # Cosine annealing final factor
WARMUP_EPOCHS = 0.5
MOSAIC = 0.0                 # 0.0: 100% native undistorted road frames for all 8 epochs
SCALE = 0.5                  # Gentle scale jitter
BOX_LOSS = 9.0               # Maximum bounding-box IoU tightness
CLS_LOSS = 0.45              # Penalize false-positive activations to drive Precision from 65.7% -> 76%+
DFL_LOSS = 1.8               # Distribution focal loss for sub-pixel boundary tightness
NUM_CACHE_THREADS = 32
TIME_BUDGET_H = 8.5          # Run budget well within Kaggle 12-hour ceiling
RUN_NAME = "phase14_precision_lock"

CLASS_NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]
VRU_CLASSES = {0: "person", 1: "rider", 5: "bike", 6: "motor"}

EXPECTED_P13_SHA256 = "bbac96e571881b294c96020756ae59140bc72a97229d1d42d703904bb61354ce"

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
FAST_CACHE = Path("/tmp/bdd100k_fast")
PROJECT = WORK / "runs"

FINAL_CHECKPOINT = WORK / "phase14_final.pt"
BEST_CHECKPOINT = WORK / "phase14_best.pt"
LAST_CHECKPOINT = WORK / "phase14_last.pt"
RESULTS_CSV_OUT = WORK / "phase14_results.csv"
SUMMARY_JSON = WORK / "phase14_summary.json"
EVENTS_JSONL = WORK / "events_phase14.jsonl"

T0 = float(os.environ.get("RS_T0", time.time()))
IS_MAIN_RANK = int(os.environ.get("RANK", -1)) in (-1, 0)

# Patch ultralytics imread for transient I/O resilience
import cv2
import ultralytics.utils.patches as patches_module
_orig_imread = patches_module.imread

def robust_imread(filename: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    for attempt in range(4):
        im = _orig_imread(filename, flags)
        if im is not None:
            return im
        time.sleep(0.04 * (2 ** attempt))
    try:
        im = cv2.imread(str(filename), flags)
        if im is not None:
            return im
    except Exception:
        pass
    return None

patches_module.imread = robust_imread


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def log_event(event_type: str, data: dict):
    if not IS_MAIN_RANK:
        return
    payload = {
        "timestamp": time.time(),
        "elapsed_s": round(time.time() - T0, 2),
        "event": event_type,
        **data
    }
    with open(EVENTS_JSONL, "a") as f:
        f.write(json.dumps(payload) + "\n")
    print(f"[{event_type.upper()}] {json.dumps(data)}", flush=True)


def locate_input_files():
    print("\n" + "=" * 80)
    print("ROADSENTINEL PHASE 14 — HARD-NEGATIVE PRECISION LOCK")
    print("=" * 80)

    # 1. Base weights (phase13_best.pt)
    weight_candidates = list(INPUT.rglob("phase13_best.pt"))
    if not weight_candidates:
        raise FileNotFoundError("phase13_best.pt not found in /kaggle/input")
    base_weight_path = weight_candidates[0]
    base_hash = sha256_file(base_weight_path)
    print(f"[input] Base Checkpoint : {base_weight_path} (SHA-256: {base_hash})")

    # 2. Balanced Curriculum manifest
    curr_candidates = list(INPUT.rglob("train_phase14_balanced.txt"))
    if not curr_candidates:
        raise FileNotFoundError("train_phase14_balanced.txt not found in /kaggle/input")
    curr_manifest_path = curr_candidates[0]
    print(f"[input] Balanced Curriculum : {curr_manifest_path}")

    # 3. Holdout split manifest
    holdout_candidates = list(INPUT.rglob("holdout_split.txt"))
    if not holdout_candidates:
        raise FileNotFoundError("holdout_split.txt not found in /kaggle/input")
    holdout_manifest_path = holdout_candidates[0]
    print(f"[input] Holdout Split       : {holdout_manifest_path}")

    # 4. BDD100K raw dataset root
    bdd_roots = list(INPUT.rglob("bdd100k-yolo"))
    if not bdd_roots:
        bdd_roots = list(INPUT.rglob("images"))
        if bdd_roots:
            bdd_root = bdd_roots[0].parent
        else:
            raise FileNotFoundError("BDD100K dataset root not found in /kaggle/input")
    else:
        bdd_root = bdd_roots[0]
    print(f"[input] BDD100K Root Path   : {bdd_root}")

    return base_weight_path, curr_manifest_path, holdout_manifest_path, bdd_root


def build_fast_cache(curr_manifest: Path, holdout_manifest: Path, bdd_root: Path):
    """
    Copies images and labels to local fast NVMe /tmp/bdd100k_fast/ for 0-latency I/O.
    """
    if not IS_MAIN_RANK:
        return FAST_CACHE / "train" / "images", FAST_CACHE / "val" / "images"

    print("\n[cache] Initializing NVMe Fast Cache at /tmp/bdd100k_fast/...")
    train_img_dst = FAST_CACHE / "train" / "images"
    train_lbl_dst = FAST_CACHE / "train" / "labels"
    val_img_dst = FAST_CACHE / "val" / "images"
    val_lbl_dst = FAST_CACHE / "val" / "labels"

    for d in (train_img_dst, train_lbl_dst, val_img_dst, val_lbl_dst):
        d.mkdir(parents=True, exist_ok=True)

    # Read curriculum lines
    with open(curr_manifest) as f:
        train_basenames = [Path(line.strip()).name for line in f if line.strip()]

    with open(holdout_manifest) as f:
        val_basenames = [Path(line.strip()).name for line in f if line.strip()]

    print(f"[cache] Total Training Images: {len(train_basenames):,}")
    print(f"[cache] Total Holdout Images : {len(val_basenames):,}")

    # Index source files
    print("[cache] Indexing source files from /kaggle/input/...")
    src_images = {}
    src_labels = {}
    for p in bdd_root.rglob("*.jpg"):
        src_images[p.name] = p
    for p in bdd_root.rglob("*.txt"):
        src_labels[p.stem] = p

    print(f"[cache] Indexed {len(src_images):,} source images, {len(src_labels):,} label files.")

    def copy_pair(item):
        fname, dst_img_dir, dst_lbl_dir = item
        stem = Path(fname).stem
        if fname in src_images:
            dst_img = dst_img_dir / fname
            if not dst_img.exists():
                shutil.copyfile(src_images[fname], dst_img)
        if stem in src_labels:
            dst_lbl = dst_lbl_dir / f"{stem}.txt"
            if not dst_lbl.exists():
                shutil.copyfile(src_labels[stem], dst_lbl)

    tasks = []
    for fn in train_basenames:
        tasks.append((fn, train_img_dst, train_lbl_dst))
    for fn in val_basenames:
        tasks.append((fn, val_img_dst, val_lbl_dst))

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=NUM_CACHE_THREADS) as executor:
        list(executor.map(copy_pair, tasks))
    el = time.time() - t_start

    print(f"[cache] Fast Cache population complete in {el:.2f}s ({len(tasks)/el:.1f} files/s).")
    return train_img_dst, val_img_dst


def generate_data_yaml(train_img_dir: Path, val_img_dir: Path) -> Path:
    yaml_dict = {
        "path": str(FAST_CACHE),
        "train": "train/images",
        "val": "val/images",
        "nc": 10,
        "names": CLASS_NAMES
    }
    yaml_path = WORK / "data_phase14.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)
    print(f"[dataset] Generated data YAML: {yaml_path}")
    return yaml_path


def setup_survival_callback(model: YOLO):
    """Saves checkpoints, metrics, and logs after every single epoch."""
    def on_train_epoch_end(trainer):
        try:
            # In PyTorch DDP, only rank 0 / -1 performs logging and checkpointing
            rank = getattr(trainer, "rank", -1)
            if rank not in (-1, 0):
                return

            ep = getattr(trainer, "epoch", 0) + 1
            print(f"\n[survival] Epoch {ep}/{TARGET_EPOCHS} completed. Archiving survival checkpoints...", flush=True)

            run_dir = Path(trainer.save_dir) if hasattr(trainer, "save_dir") else None
            if run_dir and run_dir.exists():
                best_pt = run_dir / "weights" / "best.pt"
                last_pt = run_dir / "weights" / "last.pt"
                res_csv = run_dir / "results.csv"

                if best_pt.exists():
                    shutil.copyfile(best_pt, BEST_CHECKPOINT)
                    b_hash = sha256_file(BEST_CHECKPOINT)
                    print(f"[survival] phase14_best.pt updated (SHA-256: {b_hash})", flush=True)

                if last_pt.exists():
                    shutil.copyfile(last_pt, LAST_CHECKPOINT)

                if res_csv.exists():
                    shutil.copyfile(res_csv, RESULTS_CSV_OUT)

            raw_metrics = getattr(trainer, "metrics", None)
            if isinstance(raw_metrics, dict):
                metrics_dict = {str(k): float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))}
            else:
                metrics_dict = {}

            log_event("epoch_end", {
                "epoch": ep,
                "metrics": metrics_dict
            })
        except Exception as exc:
            print(f"[survival] Non-fatal callback warning: {exc}", flush=True)

    model.add_callback("on_train_epoch_end", on_train_epoch_end)


def main():
    base_weight_path, curr_manifest, holdout_manifest, bdd_root = locate_input_files()

    train_img_dst, val_img_dst = build_fast_cache(curr_manifest, holdout_manifest, bdd_root)
    data_yaml = generate_data_yaml(train_img_dst, val_img_dst)

    devices = [0, 1] if torch.cuda.device_count() >= 2 else (0 if torch.cuda.is_available() else "cpu")
    print(f"[train] Training on devices: {devices} (Count: {torch.cuda.device_count()})")

    model = YOLO(str(base_weight_path))
    setup_survival_callback(model)

    print("\n" + "=" * 80)
    print(f"LAUNCHING PHASE 14 PRECISION LOCK: {TARGET_EPOCHS} EPOCHS @ {IMGSZ}px")
    print(f"Hyperparameters: AdamW, lr0={LR0}, lrf={LRF}, mosaic={MOSAIC}, box={BOX_LOSS}, cls={CLS_LOSS}, dfl={DFL_LOSS}")
    print("=" * 80 + "\n")

    t_train_start = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=TARGET_EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=devices,
        optimizer="AdamW",
        lr0=LR0,
        lrf=LRF,
        cos_lr=True,
        warmup_epochs=WARMUP_EPOCHS,
        mosaic=MOSAIC,
        scale=SCALE,
        box=BOX_LOSS,
        cls=CLS_LOSS,
        dfl=DFL_LOSS,
        val=True,
        save=True,
        save_period=1,
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=True,
        plots=False,
        verbose=True
    )
    total_train_hours = (time.time() - t_train_start) / 3600.0

    print(f"\n[train] Training completed in {total_train_hours:.2f} hours.")

    rank = int(os.environ.get("RANK", -1))
    if rank in (-1, 0):
        if BEST_CHECKPOINT.exists():
            final_best_hash = sha256_file(BEST_CHECKPOINT)
            print(f"[complete] Final Champion: {BEST_CHECKPOINT} (SHA-256: {final_best_hash})")

        summary = {
            "experiment": "Phase 14 — Hard-Negative Precision Lock",
            "base_checkpoint": "phase13_best.pt",
            "base_sha256": EXPECTED_P13_SHA256,
            "target_epochs": TARGET_EPOCHS,
            "imgsz": IMGSZ,
            "curriculum_size": 40554,
            "holdout_size": 5000,
            "total_elapsed_hours": total_train_hours,
            "best_checkpoint_sha256": sha256_file(BEST_CHECKPOINT) if BEST_CHECKPOINT.exists() else None,
            "status": "COMPLETED"
        }
        with open(SUMMARY_JSON, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
