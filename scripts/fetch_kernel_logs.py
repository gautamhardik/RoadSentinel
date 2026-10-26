import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
kernel_ref = 'hardikgautam24119014/roadsentinel-phase-8-official-val-eval'
status = api.kernels_status(kernel_ref)
print(f"Status: {status}")

out_dir = os.path.abspath('scratch/kernel_debug')
os.makedirs(out_dir, exist_ok=True)
try:
    api.kernels_output(kernel_ref, path=out_dir)
    print(f"Output files in {out_dir}:", os.listdir(out_dir))
    for f in os.listdir(out_dir):
        fp = os.path.join(out_dir, f)
        if os.path.isfile(fp):
            print(f"--- File: {f} ({os.path.getsize(fp)} bytes) ---")
            if f.endswith('.log') or f.endswith('.txt'):
                with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                    lines = fh.readlines()
                    print(f"Total lines: {len(lines)}")
                    print("Tail:")
                    for line in lines[-25:]:
                        print(line.rstrip())
except Exception as e:
    print(f"Failed to fetch output: {e}")
