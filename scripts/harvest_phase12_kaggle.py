"""
RoadSentinel - Harvest Kaggle Phase 12 (YOLO11m-P2) Outputs

Monitors the running Kaggle training kernel:
  hardikgautam24119014/roadsentinel-phase-12-p2-training-gpu
Downloads exported checkpoints, training curves, and provenance ledger.
"""

import sys
import subprocess
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL_SLUG = "hardikgautam24119014/roadsentinel-phase-12-p2-training-gpu"
DEST_DIR = ROOT / "experiments/phase12_p2_architecture/harvested_outputs"


def check_status() -> str:
    cmd = ["kaggle", "kernels", "status", KERNEL_SLUG]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout.strip()
    print(f"[status] {out}")
    return out


def harvest_outputs():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[harvest] Checking status for {KERNEL_SLUG}...")
    st = check_status()

    print(f"[download] Fetching outputs into {DEST_DIR}...")
    cmd = ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(DEST_DIR)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(f"[stderr] {res.stderr}")

    files = list(DEST_DIR.glob("**/*"))
    print(f"[harvest] Retrieved {len(files)} files/directories:")
    for f in files:
        if f.is_file():
            print(f"  - {f.relative_to(DEST_DIR)} ({f.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--status-only":
        check_status()
    else:
        harvest_outputs()
