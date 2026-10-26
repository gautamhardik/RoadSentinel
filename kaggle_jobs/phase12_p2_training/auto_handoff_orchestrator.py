"""
RoadSentinel Phase 12 — Automated Session Handoff Monitor
Watches Session 1 until completion, verifies resume_last.pt, and automatically launches Session 2.
"""

import time
import subprocess
import json
import hashlib
from pathlib import Path

SESSION1_SLUG = "hardikgautam24119014/roadsentinel-phase-12-p2-training"
SESSION2_DIR = Path("kaggle_jobs/phase12_p2_training/session2")
OUTPUT_DIR = Path("kaggle_jobs/phase12_p2_training/session1_output")

def run(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.stdout.strip(), p.stderr.strip(), p.returncode

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

def check_status():
    out, _, _ = run(f"kaggle kernels status {SESSION1_SLUG}")
    return out

def main():
    print("=" * 80)
    print("RoadSentinel Phase 12 — Automated Session 1 -> Session 2 Orchestrator")
    print("=" * 80)
    
    poll_count = 0
    while True:
        status = check_status()
        poll_count += 1
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{now}] Poll #{poll_count}: {status}")
        
        if "KernelWorkerStatus.RUNNING" not in status:
            print(f"\n[ALERT] Session 1 status changed: {status}")
            break
            
        time.sleep(120)  # Check every 2 minutes
        
    print("\n[Step 1/3] Session 1 execution has concluded. Downloading outputs...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out, err, code = run(f"kaggle kernels output {SESSION1_SLUG} --path {OUTPUT_DIR}")
    print(f"Download output: {out}\n{err}")
    
    ckpt_path = OUTPUT_DIR / "resume_last.pt"
    if not ckpt_path.exists():
        found = list(OUTPUT_DIR.rglob("resume_last.pt"))
        if found:
            ckpt_path = found[0]
            
    if not ckpt_path.exists():
        print(f"[ERROR] Could not find resume_last.pt in {OUTPUT_DIR}! Aborting automatic launch.")
        return
        
    ckpt_hash = sha256(ckpt_path)
    file_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"\n[Step 2/3] Verified Session 1 Checkpoint:")
    print(f"  Path:     {ckpt_path}")
    print(f"  Size:     {file_size_mb:.2f} MB")
    print(f"  SHA-256:  {ckpt_hash}")
    
    print("\n[Step 3/3] Automatically launching Session 2 on Kaggle...")
    out, err, code = run(f"kaggle kernels push -p {SESSION2_DIR}")
    print(f"Push result:\n{out}\n{err}")
    
    time.sleep(10)
    s2_status, _, _ = run("kaggle kernels status hardikgautam24119014/roadsentinel-phase-12-p2-session2")
    print(f"\nSession 2 Initial Status: {s2_status}")
    print("=" * 80)
    print("Automation handoff complete! Session 2 is now running.")
    print("=" * 80)

if __name__ == "__main__":
    main()
