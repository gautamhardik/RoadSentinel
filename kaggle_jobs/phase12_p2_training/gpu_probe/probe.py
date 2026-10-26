import torch
import subprocess
import sys

print("=" * 60)
print("GPU PROBE DIAGNOSTIC")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

try:
    smi = subprocess.check_output(["nvidia-smi"], text=True)
    print("\n--- NVIDIA-SMI OUTPUT ---")
    print(smi)
except Exception as e:
    print(f"\nnvidia-smi failed: {e}")
