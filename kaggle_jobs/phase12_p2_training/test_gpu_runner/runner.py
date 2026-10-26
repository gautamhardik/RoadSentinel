import os
import sys
import torch

print("=" * 60)
print("GPU HARDWARE ACCELERATION AUDIT")
print("=" * 60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU [{i}]: {torch.cuda.get_device_name(i)} ({round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2)} GB)")
    # Quick tensor compute test
    x = torch.randn(1000, 1000, device='cuda:0')
    y = x @ x
    print("Tensor computation on CUDA:0 PASSED successfully!")
else:
    print("NO CUDA ACCELERATION FOUND!")
print("=" * 60)
