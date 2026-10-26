import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from pathlib import Path
from ultralytics import YOLO

p = Path('experiments/phase10_model_optimization/phase10b_extended_training/session3_artifacts/session3_checkpoint_ep38_unstripped.pt')
ck = torch.load(p, map_location='cpu', weights_only=False)

print('--- CHECKPOINT RESUME AUDIT ---')
completed_ep = ck['epoch'] + 1
print(f"Completed epoch index: {ck['epoch']} (Epoch {completed_ep} completed)")
print(f"Target total epochs in train_args: {ck['train_args']['epochs']}")
print(f"Optimizer present: {ck['optimizer'] is not None}")
print(f"Optimizer state tensors count: {len(ck['optimizer']['state'])}")
print(f"Optimizer param groups: {len(ck['optimizer']['param_groups'])}")
print(f"Current lr in optimizer: {ck['optimizer']['param_groups'][0]['lr']}")
print(f"EMA present: {ck['ema'] is not None}")

# Verify YOLO load
model = YOLO(str(p))
print('YOLO model loaded cleanly without error!')
