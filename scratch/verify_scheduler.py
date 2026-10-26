import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.optim as optim
from ultralytics.utils.torch_utils import one_cycle

# 1. Inspect checkpoint train_args to see cos_lr and lrf used in Phase 6
ckpt_path = "experiments/phase6_session2_error/resume_last.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("=== CHECKPOINT AUDIT ===")
train_args = ckpt["train_args"]
if hasattr(train_args, "__dict__"):
    train_args = vars(train_args)
print(f"Recorded epochs: {train_args.get('epochs')}")
print(f"Recorded cos_lr: {train_args.get('cos_lr')}")
print(f"Recorded lr0: {train_args.get('lr0')}")
print(f"Recorded lrf: {train_args.get('lrf')}")
print(f"Recorded warmup_epochs: {train_args.get('warmup_epochs')}")
print(f"Checkpoint completed epoch index: {ckpt.get('epoch')}")

# Optimizer state
opt_state = ckpt.get("optimizer")
print(f"Saved optimizer param groups: {len(opt_state['param_groups'])}")
for i, g in enumerate(opt_state["param_groups"]):
    print(f"  Group {i} saved lr: {g['lr']}, momentum: {g.get('momentum')}")

# 2. Simulate Ultralytics BaseTrainer scheduler mechanics
epochs_50 = 50
cos_lr = train_args.get("cos_lr", False)
lr0 = train_args.get("lr0", 0.01)
lrf = train_args.get("lrf", 0.01)

# Formula for lf:
if cos_lr:
    lf_50 = one_cycle(1, lrf, epochs_50)
    lf_26 = one_cycle(1, lrf, 26)
else:
    lf_50 = lambda x: max(1 - x / epochs_50, 0) * (1.0 - lrf) + lrf
    lf_26 = lambda x: max(1 - x / 26, 0) * (1.0 - lrf) + lrf

print("\n=== SCHEDULER MATHEMATICS ===")
print(f"Using cos_lr = {cos_lr}, lr0 = {lr0}, lrf = {lrf}")
print("If schedule was 26 epochs total:")
print(f"  lf_26(25) = {lf_26(25):.6f} -> lr = {lr0 * lf_26(25):.6f}")
print(f"  lf_26(26) = {lf_26(26):.6f} -> lr = {lr0 * lf_26(26):.6f}")

print("\nIf schedule is extended to 50 epochs total:")
print(f"  lf_50(25) (end of epoch 26)   = {lf_50(25):.6f} -> lr = {lr0 * lf_50(25):.6f}")
print(f"  lf_50(26) (start of epoch 27) = {lf_50(26):.6f} -> lr = {lr0 * lf_50(26):.6f}")
print(f"  lf_50(38) (end of session 3)  = {lf_50(38):.6f} -> lr = {lr0 * lf_50(38):.6f}")
print(f"  lf_50(49) (end of epoch 50)   = {lf_50(49):.6f} -> lr = {lr0 * lf_50(49):.6f}")

# 3. Simulate exact PyTorch optimizer + scheduler step sequence on resume
model = torch.nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.937)
# Build scheduler before resume loading (as BaseTrainer._build_train_pipeline does)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf_50)

print("\n=== RESUME STEP SIMULATION ===")
print("Step A: Optimizer built with lr0:", optimizer.param_groups[0]["lr"])
print("Step B: Scheduler initialized with base_lrs:", scheduler.base_lrs)

# Step C: BaseTrainer._load_checkpoint_state loads checkpoint optimizer state
optimizer.load_state_dict(opt_state)
print("Step C: After load_state_dict, optimizer lr in param group:", optimizer.param_groups[0]["lr"])

# Step D: BaseTrainer sets scheduler.last_epoch = start_epoch - 1
start_epoch = ckpt["epoch"] + 1  # 25 + 1 = 26
scheduler.last_epoch = start_epoch - 1  # 25
print(f"Step D: scheduler.last_epoch set to: {scheduler.last_epoch}")

# Step E: BaseTrainer._do_train calls scheduler.step() at start of loop
scheduler.step()
print(f"Step E: After scheduler.step(), scheduler.last_epoch = {scheduler.last_epoch}")
print(f"Step E: Active optimizer LR for Epoch 27 = {optimizer.param_groups[0]['lr']:.6f}")
print(f"Does it match theoretical lf_50(26)*lr0? {abs(optimizer.param_groups[0]['lr'] - lr0 * lf_50(26)) < 1e-7}")
