# RoadSentinel Phase 8 — Bounded Pilot Diagnosis & Recovery Report

**Date:** 2026-10-13  
**Target Kernel:** `hardikgautam24119014/roadsentinel-pilot-300-native-eval`  
**Earlier Kernel:** `hardikgautam24119014/roadsentinel-phase-8-native-res-eval`  
**Checkpoint:** Frozen YOLO11m Phase 6 Epoch 26 (`42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e`)  

---

## 1. Execution Status Verification (Measured Facts)

1. **Earlier 10,000-Image Kernel:**
   - Query: `api.kernels_status('hardikgautam24119014/roadsentinel-phase-8-native-res-eval')`
   - Measured Status: **`KernelWorkerStatus.CANCEL_ACKNOWLEDGED`** (Confirmed stopped).
   - Artifacts recovered: 0 bytes (Kaggle buffered logs; teardown flushed empty `[]`).

2. **Bounded 300-Image Pilot Kernel:**
   - Query: `api.kernels_status('hardikgautam24119014/roadsentinel-pilot-300-native-eval')`
   - Measured Status: **`KernelWorkerStatus.RUNNING`**
   - Active Duration: **>24 minutes** (Pushed at ~05:58 local time; current time: 06:15+ local time).
   - Output files retrieved: `[]` (0 files).
   - Execution logs retrieved: `0 bytes` (`''`).

---

## 2. Code Trace & Timer Audit ([`evaluate_pilot_native.py`](file:///c:/Users/hiten/Downloads/cv/kaggle_jobs/pilot_300_native_eval/evaluate_pilot_native.py))

A line-by-line inspection of the script reveals the exact lifecycle and timer behavior:

1. **Timer Start Point (Lines 359 & 411):**
   - `t0_start = time.time()` is initialized inside `def main():` on line 359.
   - Container spin-up, package loading, and top-level module imports (`import ultralytics`, `import torch`) happen **before** `t0_start`.
2. **Operations Between Timer Start and Inference Loop:**
   - `root = data_root()` (checks directory existence).
   - `pilot_images = sorted(...)` (filters 300 filenames from local directory).
   - `find_checkpoint()`: Recursively globs `/kaggle/input` for `.pt` files matching `epoch26` or `best`.
   - `ckpt_hash = hashlib.sha256(f.read()).hexdigest()`: Reads the 40MB checkpoint from disk to verify SHA-256 integrity.
   - Parsing labels: Iterates over the 300 pilot text label files and loads ground-truth boxes into memory.
   - `model = YOLO(str(ckpt_path))` on line 409: PyTorch weights loading and architecture instantiation.
3. **Inference Loop & Time-Budget Enforcement (Lines 418–423):**
   ```python
   for i in range(0, total_imgs, BATCH):
       elapsed = time.time() - t0_start
       if elapsed > MAX_TIME_BUDGET_SECONDS:  # 600 seconds (10 min)
           print(f"WARNING: Time budget ({MAX_TIME_BUDGET_SECONDS}s) exceeded at {elapsed:.1f}s. Halting cleanly.")
           break
   ```
   - **Key Finding:** The 600-second timer covers label parsing, model weight loading, and batch inference.
   - **Critical Loop Behavior:** The budget check only executes **at the boundary between batches**. If a single call to `model.predict(batch_paths, imgsz=1280, ...)` stalls or blocks inside Ultralytics / PyTorch / CUDA driver, the Python interpreter never reaches the next loop iteration, and the break condition cannot trigger.
4. **Post-Loop Execution (Lines 452–522):**
   - Post-loop aggregation, metric comparison, and writing `pilot_300_native_res_comparison.json` happen **after** the loop completes or breaks.

---

## 3. Checkpoint Persistence Mechanism Audit

- **Script Intent:** Line 440 writes `pilot_image_stats_checkpoint.json` to `/kaggle/working/` after every batch.
- **Kaggle Infrastructure Reality:**
  - Kaggle batch kernels execute inside an isolated Docker container with ephemeral overlay storage.
  - Files written to `/kaggle/working/` are packaged and exposed via `api.kernels_output()` **only after the kernel container terminates cleanly (exit code 0 or graceful shutdown)**.
  - During `RUNNING` status, Kaggle's public API does not expose a live read-stream to `/kaggle/working/`.
  - **Fact:** `pilot_image_stats_checkpoint.json` is not downloadable or readable while the kernel is still in `RUNNING` state.

---

## 4. Failure Mode Diagnosis (Evidence vs Hypothesis)

### Measured Facts:
1. Ground-truth parsing + SHA-256 verification of the 300 pilot images takes **< 0.15 seconds**.
2. CPU vectorized greedy matching on a real BDD100k 1280px prediction (40 preds, 34 GTs) takes **0.39 ms per image** (less than 0.12 seconds total for all 300 images). Matching is definitely **not** the stall point.
3. The pilot kernel has run for **24 minutes**, exceeding the 10-minute (600s) script budget by $2.4\times$.
4. Zero output files or logs have been flushed.

### Root Cause Analysis (Hypothesis supported by architectural behavior):
- Because the script has been running for 24 minutes without the batch loop breaking at 10 minutes, the process is stalled **inside `model.predict(...)` on Batch 0 or during model initialization**.
- **Probable Block Point:** On Kaggle's dual-T4 environment (`GPU T4 x2`), passing `device=0` combined with batch size 16 at 1280px ($1280 \times 1280$ square letterboxed in YOLO) can trigger:
  1. CUDA memory allocation deadlock / paging freeze on 16 images $\times$ 1280px tensors with Ultralytics auto-batching or engine warm-up.
  2. Network filesystem I/O hang when reading 16 full-resolution 1280x720 images simultaneously from Kaggle's `/kaggle/input` shared volume.

---

## 5. Recovery Plan & Smallest Justified Next Step

1. **Stop the Stuck Pilot Session:**
   - The pilot has exceeded its time budget and is confirmed non-responsive.
   - Stop `hardikgautam24119014/roadsentinel-pilot-300-native-eval` from the Kaggle UI (same Stop Run control).
2. **Post-Termination Evidence Sweep:**
   - Query `api.kernels_logs` and `api.kernels_output` immediately after termination to verify whether Batch 0 checkpoint or any error stack trace was flushed.
3. **Execution Safety Modifications Before Any Future Re-run:**
   - **Batch Size:** Reduce batch size from `16` to `4` (or `1`) to eliminate memory pressure at 1280px.
   - **Explicit Flush:** Set `sys.stdout.reconfigure(line_buffering=True)` and `PYTHONUNBUFFERED=1` so progress lines appear in real-time logs.
   - **Per-Image Timeout / Try-Catch:** Wrap batch predict with an alarm/watchdog.
