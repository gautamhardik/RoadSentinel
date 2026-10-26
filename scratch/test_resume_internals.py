import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import copy
import shutil
import tempfile
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

def test_resume_mechanics():
    print("=" * 60)
    print("TEST 1: RESUME CHECKPOINT STATE & SCHEDULER VERIFICATION")
    print("=" * 60)
    
    ckpt_path = Path("experiments/phase6_session2_error/resume_last.pt")
    assert ckpt_path.exists(), f"Missing {ckpt_path}"
    
    # Load raw checkpoint
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_epoch = raw_ckpt.get("epoch")
    print(f"1. Raw checkpoint 'epoch' index: {raw_epoch} (0-indexed, meaning {raw_epoch + 1} epochs completed)")
    
    # Check optimizer state in checkpoint
    raw_opt = raw_ckpt.get("optimizer")
    assert raw_opt is not None, "Missing optimizer in checkpoint"
    saved_lrs = [g["lr"] for g in raw_opt["param_groups"]]
    saved_momentums = [g.get("momentum") for g in raw_opt["param_groups"]]
    print(f"2. Saved optimizer parameter groups count: {len(raw_opt['param_groups'])}")
    print(f"   Saved LRs: {saved_lrs}")
    print(f"   Saved momentums: {saved_momentums}")
    
    # Verify patch for 50 epochs
    patched_ckpt = copy.deepcopy(raw_ckpt)
    train_args = dict(patched_ckpt["train_args"]) if isinstance(patched_ckpt["train_args"], dict) else vars(patched_ckpt["train_args"]).copy()
    train_args["epochs"] = 50
    train_args["close_mosaic"] = 10
    patched_ckpt["train_args"] = train_args
    
    # Save temporary patched checkpoint
    tmp_dir = Path("scratch/test_resume_patch")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    patched_path = tmp_dir / "resume_last_prepared_50.pt"
    torch.save(patched_ckpt, patched_path)
    print(f"3. Prepared 50-epoch checkpoint saved to: {patched_path}")
    
    # Now instantiate DetectionTrainer with overrides
    cfg = copy.deepcopy(DEFAULT_CFG)
    cfg.model = str(patched_path)
    cfg.data = "bdd100k_dataset/bdd100k.yaml" if Path("bdd100k_dataset/bdd100k.yaml").exists() else "coco8.yaml"
    cfg.epochs = 50
    cfg.close_mosaic = 10
    cfg.batch = 16
    cfg.imgsz = 960
    cfg.device = "cpu"
    cfg.resume = str(patched_path)
    
    trainer = DetectionTrainer(overrides=dict(cfg))
    
    # Check check_resume
    trainer.check_resume(overrides={"epochs": 50, "close_mosaic": 10})
    print(f"4. After check_resume: trainer.args.epochs = {trainer.args.epochs}, trainer.resume = {trainer.resume}")
    
    # Setup model and resume state
    ckpt = trainer.setup_model()
    trainer.set_model_attributes()
    
    # Build optimizer and scheduler
    # In DetectionTrainer, build_optimizer creates 3 groups
    trainer.optimizer = trainer.build_optimizer(
        model=trainer.model,
        name=trainer.args.optimizer,
        lr=trainer.args.lr0,
        momentum=trainer.args.momentum,
        decay=trainer.args.weight_decay,
        iterations=1000
    )
    trainer._setup_scheduler()
    
    print(f"5. Before resume: optimizer groups = {len(trainer.optimizer.param_groups)}, initial base_lrs = {trainer.scheduler.base_lrs}")
    
    # Setup scaler and stopper as BaseTrainer._setup_train does
    from ultralytics.utils.torch_utils import EarlyStopping
    trainer.stopper = EarlyStopping(patience=trainer.args.patience)
    trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
    trainer.ema = None
    
    # Now call resume_training(ckpt)
    trainer.resume_training(ckpt)
    trainer.scheduler.last_epoch = trainer.start_epoch - 1
    
    print(f"6. After resume_training:")
    print(f"   trainer.start_epoch = {trainer.start_epoch} (Epoch {trainer.start_epoch + 1} 1-indexed)")
    assert trainer.start_epoch == 26, f"Expected start_epoch=26, got {trainer.start_epoch}"
    print(f"   [PASS] Resumes at epoch 27 (start_epoch=26)")
    
    # Verify optimizer state restored
    loaded_momentums = [g.get("momentum") for g in trainer.optimizer.param_groups]
    print(f"7. Loaded momentums in trainer.optimizer: {loaded_momentums}")
    for m in loaded_momentums:
        assert abs(m - 0.936988) < 1e-4, f"Momentum not preserved! Got {m}"
    print(f"   [PASS] Optimizer momentum preserved ({loaded_momentums[0]:.6f})")
    
    # Check scheduler step to start_epoch (Epoch 27)
    print(f"8. Stepping scheduler from last_epoch={trainer.scheduler.last_epoch} to epoch 26 (start of Epoch 27):")
    trainer.scheduler.step()
    resumed_lrs = [g["lr"] for g in trainer.optimizer.param_groups]
    print(f"   Active LRs at Epoch 27 start: {resumed_lrs}")
    
    # Calculate exact expected LR under linear schedule to 50 epochs:
    # lf = max(1 - 26 / 50, 0) * (1 - 0.01) + 0.01 = 0.48 * 0.99 + 0.01 = 0.4852
    expected_lr_group0 = trainer.args.lr0 * (0.48 * (1.0 - trainer.args.lrf) + trainer.args.lrf)
    print(f"   Theoretical expected LR for Epoch 27 of 50: {expected_lr_group0:.6f}")
    assert abs(resumed_lrs[0] - expected_lr_group0) < 1e-6, f"LR mismatch: {resumed_lrs[0]} vs {expected_lr_group0}"
    print(f"   [PASS] Actual learning rate at resumption follows intended 50-epoch linear schedule ({resumed_lrs[0]:.6f})")
    
    # Verify entire trajectory from Epoch 27 to 50
    print("\n9. Full LR trajectory over extended schedule (Epochs 27 to 50):")
    trajectory = []
    current_epoch = trainer.start_epoch
    for ep in range(current_epoch, trainer.epochs):
        lr_val = trainer.optimizer.param_groups[0]["lr"]
        mosaic_active = (ep < trainer.epochs - trainer.args.close_mosaic)
        trajectory.append((ep + 1, lr_val, mosaic_active))
        trainer.scheduler.step()
    
    print(f"   Epoch 27: LR = {trajectory[0][1]:.6f}, Mosaic = {trajectory[0][2]}")
    print(f"   Epoch 38 (End Session 3): LR = {trajectory[11][1]:.6f}, Mosaic = {trajectory[11][2]}")
    print(f"   Epoch 39 (Start Session 4): LR = {trajectory[12][1]:.6f}, Mosaic = {trajectory[12][2]}")
    print(f"   Epoch 40: LR = {trajectory[13][1]:.6f}, Mosaic = {trajectory[13][2]}")
    print(f"   Epoch 41 (close_mosaic): LR = {trajectory[14][1]:.6f}, Mosaic = {trajectory[14][2]}")
    print(f"   Epoch 50 (Final): LR = {trajectory[-1][1]:.6f}, Mosaic = {trajectory[-1][2]}")
    
    # Check close_mosaic
    # For epochs 41..50 (10 epochs), mosaic should be False
    mosaic_off_epochs = [ep for ep, lr, mos in trajectory if not mos]
    print(f"10. Epochs where mosaic is OFF: {mosaic_off_epochs}")
    assert mosaic_off_epochs == list(range(41, 51)), f"Expected epochs 41..50 to have mosaic off, got {mosaic_off_epochs}"
    print("   [PASS] Mosaic augmentation turns off for exactly the final 10 epochs (Epochs 41 to 50)")

if __name__ == "__main__":
    test_resume_mechanics()
