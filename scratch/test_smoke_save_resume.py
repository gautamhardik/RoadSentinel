import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

def run_smoke_test():
    print("=" * 60)
    print("SMOKE TEST: SAVE CHECKPOINT AND RESUME VERIFICATION")
    print("=" * 60)
    
    smoke_dir = Path("scratch/smoke_test_run").resolve()
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir, ignore_errors=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    
    saved_epoch_checkpoint = smoke_dir / "epoch_saved_with_opt.pt"
    
    def on_model_save(trainer):
        # Callback identical to production script: preserve last with full optimizer
        print(f"[callback] Saving unstripped checkpoint at epoch {trainer.epoch + 1}")
        shutil.copy2(trainer.last, saved_epoch_checkpoint)

    # 1. Train a miniature 11n detector for 2 epochs on coco8
    print("\nPhase A: Training initial 2 epochs on coco8...")
    model = YOLO("yolo11n.pt")
    model.add_callback("on_model_save", on_model_save)
    model.train(
        data="coco8.yaml",
        epochs=2,
        batch=4,
        imgsz=320,
        device="cpu",
        project=str(smoke_dir),
        name="initial_run",
        plots=False,
        save=True,
        verbose=False
    )
    
    assert saved_epoch_checkpoint.exists(), f"Saved checkpoint missing at {saved_epoch_checkpoint}"
    ck = torch.load(saved_epoch_checkpoint, map_location="cpu", weights_only=False)
    print(f"Phase A Complete: Saved checkpoint has epoch index {ck['epoch']} ({ck['epoch'] + 1} completed)")
    assert ck["optimizer"] is not None, "Optimizer state is missing!"
    print(f"  Optimizer state tensors count: {len(ck['optimizer']['state'])}")
    print(f"  Optimizer param groups: {len(ck['optimizer']['param_groups'])}")
    
    # 2. Patch checkpoint to extend total epochs from 2 to 4
    print("\nPhase B: Applying extended training patch (epochs: 2 -> 4)...")
    ck_args = dict(ck["train_args"]) if isinstance(ck["train_args"], dict) else vars(ck["train_args"]).copy()
    ck_args["epochs"] = 4
    ck_args["close_mosaic"] = 1
    ck["train_args"] = ck_args
    
    patched_ckpt_path = smoke_dir / "patched_for_resume.pt"
    torch.save(ck, patched_ckpt_path)
    print(f"Patched checkpoint saved to {patched_ckpt_path}")
    
    # 3. Resume training from patched checkpoint to epoch 4
    print("\nPhase C: Resuming from patched checkpoint to 4 total epochs...")
    resume_model = YOLO(str(patched_ckpt_path))
    resume_model.add_callback("on_model_save", on_model_save)
    resume_model.train(
        resume=True,
        device="cpu"
    )
    
    # Inspect final checkpoint
    final_ck = torch.load(saved_epoch_checkpoint, map_location="cpu", weights_only=False)
    print(f"\nPhase C Complete:")
    print(f"  Final completed epoch: {final_ck['epoch'] + 1}")
    print(f"  Final target epochs: {final_ck['train_args']['epochs']}")
    print(f"  Optimizer state tensors: {len(final_ck['optimizer']['state'])}")
    
    assert final_ck["epoch"] == 3, f"Expected final completed epoch 3 (4 total epochs), got {final_ck['epoch']}"
    print("\n[SMOKE TEST 100% PASS] Successfully trained, saved, patched, resumed, and completed extended training!")

if __name__ == "__main__":
    run_smoke_test()
