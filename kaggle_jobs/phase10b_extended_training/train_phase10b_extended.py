"""
RoadSentinel Phase 10B — Controlled YOLO11m Extended Training: Session 4
Trains Epochs 39 through 50 inclusive (12 epochs) on Kaggle dual Tesla T4 GPUs.

Key Invariants:
  - Resumes from Session 3 unstripped checkpoint (Epoch 38 completed, SHA-256: 718a975a...).
  - Completes the full 50-epoch total schedule.
  - Mosaic augmentation closes at Epoch 41 (close_mosaic=10), so Epochs 41-50 run unaugmented.
  - Preserves custom VRU loss weighting and deterministic training/holdout split.
  - Keeps frozen Phase 6 baseline unchanged.
  - Saves unstripped session4_final_checkpoint_ep50.pt with complete optimizer state, plus best checkpoint.
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

try:
    import ultralytics
    assert ultralytics.__version__ == "8.4.174"
    import cloudpickle
except (ImportError, AssertionError):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.174", "cloudpickle>=3.1.1"])
    import ultralytics
    import cloudpickle

import numpy as np
import torch
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------- Configuration
IMGSZ = 960
TARGET_EPOCHS = 50          # Scheduler targets 50 total epochs
SESSION_TARGET_EPOCH = 50   # Session 4 completes at Epoch 50 (12 epochs: 39..50)
START_EPOCH = 38            # 0-indexed completed epochs in resume checkpoint (start at 39)
CLOSE_MOSAIC = 10           # Mosaic closes at Epoch 41 (epochs 41-50 unaugmented)
BATCH = 16
TIME_BUDGET_H = 10.5        # Hard safety ceiling under Kaggle 12-hour limit
RUN_NAME = "phase10b_yolo11m_960_session4"
CLASS_NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]
VRU_NAMES = ("person", "rider", "bike", "motor")
CLASS_WEIGHTS = torch.tensor([1.2, 2.0, 1.0, 1.0, 1.0, 1.8, 2.0, 1.0, 1.0, 1.0])

HOLDOUT_SIZE = 5000
HOLDOUT_SHA256 = "c7047ee0073803431321735c098aad212693daa696153f31f8a26e11ebf7686b"

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
PROJECT = WORK / "runs"
RESUME_SOURCE = WORK / "resume_last.pt"
RESUME_PREPARED = WORK / "resume_last_prepared_50.pt"
SESSION4_FINAL_CHECKPOINT = WORK / "session4_final_checkpoint_ep50.pt"
SESSION4_BEST_CHECKPOINT = WORK / "session4_best.pt"
SESSION4_SUMMARY = WORK / "session4_summary.json"
EVENTS_SESSION4 = WORK / "events_session4.jsonl"
EFFECTIVE_CONFIG = WORK / "effective_train_args_session4.json"
BEST_RECORD = WORK / "best_checkpoint.json"

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


def find_one(name: str) -> Optional[Path]:
    skip = {str(r) for r in DATA_ROOTS}
    for dirpath, dirnames, filenames in os.walk(INPUT):
        dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) not in skip]
        if name in filenames:
            return Path(dirpath) / name
    return None


def log_event(**kv):
    if IS_MAIN_RANK:
        with open(EVENTS_SESSION4, "a") as f:
            f.write(json.dumps({"t": time.time(), **kv}) + "\n")


def build_data_yaml() -> dict:
    """Deterministic train/holdout split identical to Phase 6 Sessions 1 & 2."""
    root = data_root()
    train_dir, val_dir = root / "train" / "images", root / "val" / "images"
    dataset_names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    dataset_names = list(dataset_names.values()) if isinstance(dataset_names, dict) else list(dataset_names)
    assert dataset_names == CLASS_NAMES, f"Dataset class order mismatch: {dataset_names}"
    
    names = sorted(p.name for p in train_dir.glob("*.jpg"))
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
    (WORK / "data_phase6.yaml").write_text(yaml.dump(data))
    print(f"[data] Successfully created /kaggle/working/data_phase6.yaml ({len(train)} train, {len(holdout)} holdout)", flush=True)
    return data


def prepare_resumable_checkpoint(input_ckpt: Path, target_epochs: int, close_mosaic: int) -> Path:
    """Modifies checkpoint metadata so Ultralytics resume accepts epochs=target_epochs."""
    print(f"Preparing resume checkpoint: {input_ckpt} for {target_epochs} total epochs...", flush=True)
    ckpt = torch.load(input_ckpt, map_location="cpu", weights_only=False)
    assert "epoch" in ckpt, "Checkpoint missing epoch key"
    current_completed = ckpt["epoch"] + 1
    print(f"Current completed epochs: {current_completed}. Target total: {target_epochs}", flush=True)
    assert current_completed == START_EPOCH, f"Expected {START_EPOCH} completed epochs, got {current_completed}"

    train_args = dict(ckpt["train_args"]) if isinstance(ckpt["train_args"], dict) else vars(ckpt["train_args"]).copy()
    train_args["epochs"] = target_epochs
    train_args["close_mosaic"] = close_mosaic
    train_args["data"] = str(WORK / "data_phase6.yaml")
    train_args["project"] = str(PROJECT)
    train_args["name"] = RUN_NAME
    ckpt["train_args"] = train_args

    torch.save(ckpt, RESUME_PREPARED)
    hash_val = sha256(RESUME_PREPARED)
    print(f"Saved prepared checkpoint to: {RESUME_PREPARED} (SHA-256: {hash_val})", flush=True)
    return RESUME_PREPARED


def attach_vru_weights(trainer):
    """Hooks custom VRU loss weighting into the loss criterion."""
    loss_fn = getattr(trainer.model, "criterion", None)
    if loss_fn is not None and hasattr(loss_fn, "bce"):
        device = next(trainer.model.parameters()).device
        loss_fn.bce.pos_weight = CLASS_WEIGHTS.to(device)
        print(f"[weights] Successfully attached VRU class weights on rank {os.environ.get('RANK', 0)}", flush=True)


def add_callbacks(model: YOLO):
    def on_pretrain_routine_end(trainer):
        attach_vru_weights(trainer)

    def epoch_end(trainer):
        ep = trainer.epoch + 1
        lr = float(trainer.optimizer.param_groups[0]["lr"])
        mem_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0
        elapsed_h = (time.time() - T0) / 3600.0
        
        print(f"[Session 4 Progress] Completed Epoch {ep}/{TARGET_EPOCHS} (Session Epoch {ep - START_EPOCH}/{SESSION_TARGET_EPOCH - START_EPOCH}) | LR: {lr:.6f} | Mem: {mem_gb} GB | Elapsed: {elapsed_h:.2f}h", flush=True)
        
        log_event(
            event="epoch_end",
            epoch=ep,
            session_epoch=ep - START_EPOCH,
            lr=lr,
            mem_alloc_gb=mem_gb,
            elapsed_hours=elapsed_h
        )

    def on_model_save(trainer):
        # Save unstripped last.pt with complete optimizer state on each epoch
        shutil.copy2(trainer.last, WORK / "resume_last.pt")

    def fit_end(trainer):
        ep = trainer.epoch + 1
        is_best = bool(trainer.best_fitness == trainer.fitness)
        
        if IS_MAIN_RANK and is_best and trainer.best.exists():
            shutil.copy2(trainer.best, SESSION4_BEST_CHECKPOINT)
            ck = torch.load(trainer.best, map_location="cpu", weights_only=False)
            best_trace = {
                "epoch": ep,
                "date": ck.get("date"),
                "file_sha256": sha256(SESSION4_BEST_CHECKPOINT),
                "fitness": float(ck["train_metrics"]["fitness"])
            }
            BEST_RECORD.write_text(json.dumps(best_trace, indent=2))
            print(f"[BEST] New best fitness {trainer.fitness:.5f} at epoch {ep} saved to {SESSION4_BEST_CHECKPOINT}", flush=True)

        log_event(
            event="fit_end",
            epoch=ep,
            session_epoch=ep - START_EPOCH,
            fitness=float(trainer.fitness or 0),
            best_fitness=float(trainer.best_fitness or 0),
            is_best=is_best,
            metrics={k: float(v) for k, v in (trainer.metrics or {}).items()}
        )

        # Check if Session 4 completed schedule target (Epoch 50)
        if ep >= SESSION_TARGET_EPOCH:
            print(f"\n[SESSION 4 COMPLETE] Target Epoch {ep} reached! Preserving final unstripped checkpoint and artifacts...", flush=True)
            if IS_MAIN_RANK:
                # Copy unstripped checkpoint from WORK / resume_last.pt (saved during on_model_save)
                unstripped_src = WORK / "resume_last.pt"
                if unstripped_src.exists():
                    shutil.copy2(unstripped_src, SESSION4_FINAL_CHECKPOINT)
                else:
                    shutil.copy2(trainer.last, SESSION4_FINAL_CHECKPOINT)
                ckpt_hash = sha256(SESSION4_FINAL_CHECKPOINT)
                
                saved_ck = torch.load(SESSION4_FINAL_CHECKPOINT, map_location="cpu", weights_only=False)
                opt_tensors = len(saved_ck["optimizer"]["state"]) if saved_ck.get("optimizer") else 0
                param_groups = len(saved_ck["optimizer"]["param_groups"]) if saved_ck.get("optimizer") else 0
                
                summary = {
                    "session": 4,
                    "target_completed_epoch": ep,
                    "start_epoch": START_EPOCH + 1,
                    "epochs_trained_in_session": ep - START_EPOCH,
                    "final_checkpoint_file": "session4_final_checkpoint_ep50.pt",
                    "final_checkpoint_sha256": ckpt_hash,
                    "optimizer_intact": bool(opt_tensors > 0),
                    "optimizer_state_tensors": opt_tensors,
                    "optimizer_param_groups": param_groups,
                    "best_fitness": float(trainer.best_fitness or 0),
                    "final_metrics": {k: float(v) for k, v in (trainer.metrics or {}).items()},
                    "elapsed_hours": (time.time() - T0) / 3600.0
                }
                SESSION4_SUMMARY.write_text(json.dumps(summary, indent=2))
                
                # Save effective config
                eff_args = dict(trainer.args) if isinstance(trainer.args, dict) else vars(trainer.args).copy()
                EFFECTIVE_CONFIG.write_text(json.dumps({k: str(v) for k, v in eff_args.items()}, indent=2))
                
                print(f"[SAVED] Final Checkpoint: {SESSION4_FINAL_CHECKPOINT} (SHA-256: {ckpt_hash}, {opt_tensors} optimizer tensors)", flush=True)
            
            trainer.stop = True
            return

        # Time budget check
        elapsed = time.time() - T0
        per_epoch = elapsed / max(1, ep - START_EPOCH)
        if elapsed + 1.2 * per_epoch > TIME_BUDGET_H * 3600:
            print(f"[time_guard] {elapsed/3600:.2f}h elapsed; pausing cleanly after epoch {ep}", flush=True)
            if IS_MAIN_RANK:
                recov_ckpt = WORK / f"session4_recovery_ep{ep}.pt"
                shutil.copy2(WORK / "resume_last.pt", recov_ckpt)
                print(f"[RECOVERY SAVED] {recov_ckpt} (SHA-256: {sha256(recov_ckpt)})", flush=True)
            trainer.stop = True

    model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    model.add_callback("on_train_epoch_end", epoch_end)
    model.add_callback("on_model_save", on_model_save)
    model.add_callback("on_fit_epoch_end", fit_end)


def main():
    print("=" * 70, flush=True)
    print("RoadSentinel Phase 10B — Gate C: Kaggle Session 4 Execution", flush=True)
    print("=" * 70, flush=True)
    print(f"Target schedule: Epochs {START_EPOCH + 1} through {SESSION_TARGET_EPOCH} (Total target: {TARGET_EPOCHS})", flush=True)

    # 1. Setup deterministic dataset split
    build_data_yaml()

    # 2. Locate resume_last.pt in /kaggle/input
    source_ckpt = find_one("resume_last.pt")
    if source_ckpt is None:
        raise FileNotFoundError("Could not find resume_last.pt in /kaggle/input")
    
    source_hash = sha256(source_ckpt)
    print(f"Found input resume checkpoint: {source_ckpt} (SHA-256: {source_hash})", flush=True)
    assert source_hash == "718a975aeec3c259df96f138d75d9e6298efc6b47fd63017a95a1b9b557b5fb1", f"Resume hash mismatch: {source_hash}"

    # 2b. Locate and verify frozen baseline best.pt
    baseline_ckpt = find_one("best.pt")
    if baseline_ckpt is not None:
        baseline_hash = sha256(baseline_ckpt)
        print(f"Verified frozen Phase 6 baseline: {baseline_ckpt} (SHA-256: {baseline_hash})", flush=True)
        assert baseline_hash == "42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e", f"Baseline hash mismatch: {baseline_hash}"

    # 3. Prepare checkpoint with target epochs
    prepared = prepare_resumable_checkpoint(source_ckpt, TARGET_EPOCHS, CLOSE_MOSAIC)

    # 4. Train with Ultralytics native multi-GPU execution
    n_gpu = torch.cuda.device_count()
    device = ",".join(str(i) for i in range(n_gpu)) if n_gpu else "cpu"
    print(f"Detected {n_gpu} GPUs. Target training device string: '{device}'", flush=True)

    model = YOLO(str(prepared))
    add_callbacks(model)
    print("Invoking model.train(resume=True)...", flush=True)
    model.train(resume=True, device=device)
    print("Training call finished successfully.", flush=True)

    if IS_MAIN_RANK:
        run_dir = PROJECT / RUN_NAME
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            shutil.copy2(results_csv, WORK / "session4_results.csv")
            shutil.copy2(results_csv, WORK / "training_results.csv")
            print(f"Copied {results_csv} to /kaggle/working/session4_results.csv and training_results.csv", flush=True)
        
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists() and not SESSION4_BEST_CHECKPOINT.exists():
            shutil.copy2(best_pt, SESSION4_BEST_CHECKPOINT)
        if SESSION4_BEST_CHECKPOINT.exists():
            print(f"Session 4 Best Checkpoint verified: {SESSION4_BEST_CHECKPOINT} (SHA-256: {sha256(SESSION4_BEST_CHECKPOINT)})", flush=True)

        if SESSION4_FINAL_CHECKPOINT.exists():
            print(f"Session 4 Final Checkpoint verified: {SESSION4_FINAL_CHECKPOINT} (SHA-256: {sha256(SESSION4_FINAL_CHECKPOINT)})", flush=True)


if __name__ == "__main__":
    main()
