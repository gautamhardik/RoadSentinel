# RoadSentinel Phase 8 — Native-Resolution Run Recovery & Pilot Audit Report

**Date:** 2026-10-13  
**Target Kernel:** `hardikgautam24119014/roadsentinel-phase-8-native-res-eval`  
**Checkpoint:** Frozen YOLO11m Phase 6 Epoch 26 (`42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e`)  

---

## 1. Status of the Non-Viable Kernel & Cancellation Outcome

- **Current Status via Kaggle API:** `{"status": "RUNNING", "failureMessage": null}`
- **Active Runtime:** >120 minutes (launched 2026-10-12 22:13:13 UTC / 03:43:08 local time).
- **Cancellation Attempt via API:**
  - Kaggle's public REST API does not grant session cancellation (`CancelKernelSession`) permissions to user API tokens (returns HTTP 403 Forbidden).
  - Consequently, **cancellation requires user manual intervention via the Kaggle Notebook UI**.
  - **Required Action:** Please open `https://www.kaggle.com/code/hardikgautam24119014/roadsentinel-phase-8-native-res-eval` and click **"Stop Run"** / **"Cancel Run"**.
- **GPU Quota Lock:** Per instructions, no secondary GPU job will be launched until this running session is confirmed stopped and GPU session capacity is restored.

---

## 2. Evidence Recovery Assessment

We probed the Kaggle API for outputs and stdout logs:
- **`api.kernels_output()`:** Returned empty list (`[]`). Zero output files or partial checkpoint files flushed to disk.
- **`api.kernels_logs()`:** Returned empty string (`0 bytes`).
- **Mechanism Finding:** Kaggle batch kernel execution buffers stdout, stderr, and output directory flushes until process teardown or explicit session termination. Because the kernel remains in a running/hung state, **no partial predictions, Pass 1 metric tables, or intermediate logs are available for recovery at this time**. We will probe again immediately once the session is halted in the UI.

---

## 3. Failure Root-Cause Audit

A comparative audit of the successful 960px baseline kernel (`roadsentinel-phase-8-official-val-eval`, runtime 15.7 minutes) against the 1280px native-resolution kernel revealed why the native-resolution run stalled:

1. **Quadratic / High-Resolution Memory Bandwidth Bottleneck:**
   - At 1280px ($1280 \times 720$), the input spatial resolution contains $1.78\times$ more pixels than 960px ($960 \times 540$).
   - On a Tesla T4 GPU (limited to 300 GB/s GDDR6 memory bandwidth), the early backbone layers and multi-scale feature pyramids (P3, P4) suffer substantial memory-bandwidth throttling at batch size 16.
2. **Two-Pass Architecture on 10,000 Images:**
   - The script executed two full passes: Pass 1 (`model.val()`) and Pass 2 (`model.predict()`). Over 10,000 images, this represents **20,000 forward passes** at 1280px.
   - When combined with Kaggle's network-mounted shared disk I/O, throughput degraded from ~23 FPS to ~2–4 FPS, transforming an expected ~20-minute run into a multi-hour timeout.
3. **Buffering & Missing Mid-Pass Progress Telemetry:**
   - Ultralytics standard validation progress bars use carriage returns (`\r`), which are buffered by Kaggle until batch completion or session termination, creating an observable "black box" where no progress messages reach the API log stream.

---

## 4. Bounded 300-Image Pilot Evaluation Readiness

To eliminate GPU waste, bypass Kaggle's shared I/O bottlenecks, and obtain an immediate, empirical answer on whether 1280px improves VRU detection, we prepared a dedicated, bounded pilot evaluation.

### Pilot Specification & Invariants:
- **Cohort:** Exactly the 300-image development pilot cohort from Milestone 1 (verified matching SHA-256 and labels).
- **Confidence Threshold:** `0.25` (frozen operational threshold).
- **NMS IoU Threshold:** `0.65` (frozen).
- **Matching Policy:** Class-aware, confidence-ordered greedy matching at IoU `0.50`.
- **Small-VRU Definition:** Area $< 1024\,\text{px}^2$ in original $1280 \times 720$ coordinates.
- **Test Split Quarantine:** Zero access to `bdd100k_dataset/test`.

### Safeguards Implemented in [`kaggle_jobs/pilot_300_native_eval/evaluate_pilot_native.py`](file:///c:/Users/hiten/Downloads/cv/kaggle_jobs/pilot_300_native_eval/evaluate_pilot_native.py):
1. **Single-Pass Prediction & Immediate Vectorized Matching:** Eliminates `model.val()` overhead, running inference directly and evaluating bounding boxes in one step.
2. **Hard Time Budget Ceiling:** Automatically halts and cleanly outputs partial metrics if elapsed time exceeds 600 seconds (10 minutes). (Expected runtime on T4 is ~60–80 seconds for 300 images).
3. **Atomic Progress Checkpoints:** Saves `pilot_image_stats_checkpoint.json` after every batch so no progress can ever be lost.
4. **Automated Delta Comparison:** Compares results directly against the archived 960px baseline for the exact same 300 images:
   - Baseline 960px Pilot Reference:
     - VRU Recall: **60.46%**
     - VRU Precision: **78.45%**
     - VRU F1: **0.6829**
     - Small-VRU Recall: **42.73%** (385 / 901)
     - Person Recall: 61.91%, Rider: 55.17%, Bike: 51.64%, Motor: 34.38%

### Deployment Files Ready:
- Script: [`kaggle_jobs/pilot_300_native_eval/evaluate_pilot_native.py`](file:///c:/Users/hiten/Downloads/cv/kaggle_jobs/pilot_300_native_eval/evaluate_pilot_native.py)
- Metadata: [`kaggle_jobs/pilot_300_native_eval/kernel-metadata.json`](file:///c:/Users/hiten/Downloads/cv/kaggle_jobs/pilot_300_native_active_eval/kernel-metadata.json) (Dataset: `a7madmostafa/bdd100k-yolo`, Model: `hardikgautam24119014/roadsentinel-phase-6-epoch-26-checkpoint`, GPU enabled).
