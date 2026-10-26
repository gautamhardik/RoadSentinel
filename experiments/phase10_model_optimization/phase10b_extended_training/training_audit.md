# RoadSentinel Phase 10B — Training Audit & Continuation Protocol

**Document Version:** 1.0 (Audited Pre-Execution Protocol)  
**Date:** 2026-10-13  
**Auditor:** RoadSentinel Lead Perception Auditor  
**Experiment Target:** Controlled YOLO11m Extended Training (Epoch 26 $\to$ 50)  
**Frozen Baseline Checkpoint:** `scratch/kaggle_upload_weights/best.pt`  
**Frozen Baseline SHA-256:** `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` (Verified Intact)  
**Resumable Training Checkpoint:** `experiments/phase6_session2_error/resume_last.pt`  
**Resumable Checkpoint SHA-256:** `db64f798410ed392e2af35217661ef1b8ecab32e4a6f431600468d3c44915305` (Verified Intact)  
**Quarantine Status:** `bdd100k_dataset/test` strictly preserved; zero access.

---

## 1. Executive Summary & Audit Mandate

Phase 10B evaluates whether extending the training schedule of the domain-adapted YOLO11m model from **26 to 50 total epochs** produces measurable gains in Vulnerable Road User (VRU) detection, specifically small-VRU recall ($< 32^2\text{ px}$).

Before initiating compute jobs, Stage 1 mandates a complete pre-execution audit:
1. Reconcile the exact training history and provenance of the Phase 6 sessions.
2. Confirm the exact cryptographic hashes and weight states of both the deployment weights (`best.pt`) and the full resumable training state (`resume_last.pt`).
3. Audit the optimizer, EMA, and learning-rate scheduler states.
4. Investigate the resume semantics of Ultralytics `8.4.174` to guarantee that resuming to 50 epochs does not restart training from epoch 1 or add 50 additional epochs.
5. Determine empirical throughput from raw telemetry to budget execution under Kaggle session limits.

---

## 2. Forensic Audit of Phase 6 Training Sessions

A forensic audit of `experiments/phase6_session1/` and `experiments/phase6_session2_error/` establishes the complete training timeline:

```
[Phase 6 Session 1]
Schedule: Epochs 1 to 13 (50% of 26-epoch plan)
Hardware: 2x Tesla T4 DDP (Batch 16 total, 8 per GPU)
Completion: Epoch 13 completed cleanly (fitness: 0.29889)
Output Handoff: resume_last.pt (SHA-256: 7fcb6d8a28f92cc42c764dcd87e9295bd881f3fad469dbce1645cbb1e8317c4b)
       |
       v
[Phase 6 Session 2]
Input: Restored handoff from Session 1
Schedule: Epochs 14 to 26 (100% of 26-epoch plan)
Completion: Epoch 26 completed at 2026-10-11T20:41:28Z (fitness: 0.31074)
Outputs Produced:
  - best.pt: Deployment weights (optimizer stripped, 40.5 MB, SHA-256: 42d4b385...)
  - resume_last.pt: Full resumable state (optimizer intact, 80.8 MB, SHA-256: db64f798...)
```

### 2.1 Checkpoint Cryptographic & Architectural Audit

| Attribute | Deployment Weights (`best.pt`) | Resumable State (`resume_last.pt`) | Audit Assessment |
| :--- | :--- | :--- | :--- |
| **Location** | `scratch/kaggle_upload_weights/best.pt` | `experiments/phase6_session2_error/resume_last.pt` | Both verified in repository. |
| **File Size** | **40,539,943 bytes** (~40.5 MB) | **80,805,199 bytes** (~80.8 MB) | Exact 2× size reflects full optimizer state. |
| **SHA-256 Hash** | `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` | `db64f798410ed392e2af35217661ef1b8ecab32e4a6f431600468d3c44915305` | Verified against recorded baseline. |
| **Epoch Field** | `-1` (Stripped by final evaluation) | **`25`** (0-indexed $\implies$ Epoch 26 completed) | Direct resume pointer intact. |
| **Timestamp** | `2026-10-11T20:41:28.381141+00:00` | `2026-10-11T20:41:28.381141+00:00` | Exact millisecond identity confirms common origin. |
| **Best Fitness** | Recorded in `best_checkpoint.json`: `0.31074` | `0.31074` | Perfect metric alignment. |
| **Optimizer State** | `None` (Stripped) | **Present (330 tensors, 3 param groups)** | Full momentum and gradient state preserved. |
| **EMA State** | `None` | **Present** | Exponential moving average weights intact. |
| **GradScaler** | `None` | **Present** (`scale = 2048.0`) | FP16 dynamic loss scale preserved. |
| **Total Updates** | `29,287` | `29,287` | Step counter intact. |

---

## 3. Ultralytics Resume Semantics & Technical Solution

### 3.1 The Hidden Ultralytics Resume Bug
Inspection of the installed Ultralytics engine source code (`BaseTrainer.check_resume()`) reveals a critical behavior:

```python
# In ultralytics/engine/trainer.py -> check_resume()
allowed = {
    "imgsz", "batch", "device", "close_mosaic", "augmentations",
    "save_period", "workers", "cache", "patience", "time", "freeze",
    "val", "plots", "channels_last", "distill_model", "save_dir"
}
for k, v in overrides.items():
    if k in allowed:
        setattr(self.args, k, v)
    elif k not in {"model", "data", "mode", "resume", "pretrained"} and v != getattr(self.args, k, None):
        ignored.append(k)
if ignored:
    LOGGER.warning(f"Resume ignores {ignored}, using checkpoint values. Start a new run to change them.")
```

**Auditor Finding:** `"epochs"` is **strictly excluded** from `allowed`!  
If a user executes `model.train(resume=True, epochs=50)` on an existing 26-epoch checkpoint:
1. Ultralytics ignores `epochs=50` and restores `self.args.epochs = 26` from `ckpt['train_args']`.
2. When `self.resume_training()` runs:
   ```python
   start_epoch = ckpt.get("epoch", -1) + 1  # start_epoch = 26
   assert 0 < start_epoch < self.epochs     # 26 < 26 evaluates to FALSE!
   ```
3. The execution crashes immediately with:
   `AssertionError: path/to/last.pt training to 26 epochs is finished, nothing to resume. Start a new training without resuming!`

### 3.2 The Validated Phase 10B Resolution
To resume seamlessly without restarting or adding 50 additional epochs:
1. We load `resume_last.pt` prior to training launch:
   ```python
   ckpt = torch.load("resume_last.pt", map_location="cpu", weights_only=False)
   ckpt["train_args"]["epochs"] = 50
   ckpt["train_args"]["close_mosaic"] = 10
   torch.save(ckpt, "resume_last_prepared_50.pt")
   ```
2. When Ultralytics initializes `check_resume()` from `resume_last_prepared_50.pt`, it reads `self.args.epochs = 50`.
3. `start_epoch` is calculated as $25 + 1 = 26$.
4. The assertion $26 < 50$ passes cleanly. Training begins at **Epoch 27** and runs to **Epoch 50** (exactly 24 continuation epochs).

---

## 4. `close_mosaic=10` Compatibility Analysis

- **What it does:** In YOLO training, mosaic data augmentation (stitching 4 images together) is critical for spatial diversity early in training. However, it distorts natural image scales and boundary labels. Disabling mosaic for the final epochs fine-tunes the network on true driving scene geometry.
- **Schedule Impact:**
  - `close_mosaic = 10` activates at epoch $\text{total\_epochs} - 10 = 50 - 10 = 40$.
  - **Epochs 27 to 40 (14 epochs):** Active mosaic augmentation (`p = 1.0`).
  - **Epochs 41 to 50 (10 epochs):** Mosaic disabled (`p = 0.0`), standard augmentation pipeline active.
- **Compatibility:** Fully verified. Ultralytics dynamic dataset transforms dynamically adjust `dataset.transforms` based on `epoch >= self.epochs - self.args.close_mosaic`.

---

## 5. Measured Throughput & Kaggle Session Partitioning

From the raw timestamps recorded in `experiments/phase6_session2_error/events.jsonl`:
- **Training Epoch Duration:** $1,949.7\text{ seconds}$ ($32.50\text{ minutes}$) across 65,000 images on 2x Tesla T4.
- **Holdout Validation Duration:** $51.6\text{ seconds}$ ($0.86\text{ minutes}$) across 5,000 images.
- **Combined Per-Epoch Cycle:** **33.36 minutes** ($0.556\text{ hours}$).
- **Total Duration for 24 Epochs:**
  $$24 \times 33.36\text{ min} = 800.64\text{ min} = \mathbf{13.34\text{ hours}}$$

### Kaggle Budget Constraint
Kaggle batch kernels enforce a hard timeout of **12.0 hours (43,200 seconds)**. A single session will abort if it attempts 24 epochs continuously.

**Controlled Multi-Session Schedule:**
1. **Session 3 (Phase 10B-1):**
   - Target: Epochs 27 to 38 (12 epochs).
   - Expected Duration: $12 \times 33.36\text{ min} = 400.3\text{ min} \approx \mathbf{6.67\text{ hours}}$.
   - Handoff Output: `handoff_phase10b_session3/resume_last.pt` (Epoch 38 completed).
2. **Session 4 (Phase 10B-2):**
   - Target: Epochs 39 to 50 (12 epochs, with `close_mosaic` active for Epochs 41–50).
   - Expected Duration: $12 \times 33.36\text{ min} \approx \mathbf{6.67\text{ hours}}$.
   - Final Output: `best.pt` and `final_selection.json`.
3. **Time Guard Callback:** A hard software timeout triggers at $10.5\text{ hours}$, ensuring that if any GPU throttling occurs, the session saves `resume_last.pt` cleanly before the container terminates.

---

## 6. Gate A — Preflight Verification: Scheduler & Resume Smoke Test

Before dispatching the multi-epoch job to Kaggle GPU, empirical tests were executed directly in the local Ultralytics `8.4.174` environment to verify scheduler continuity, momentum preservation, and checkpoint recoverability.

### 6.1 Scheduler Trajectory Analysis
In Ultralytics `8.4.174`, `BaseTrainer._setup_scheduler()` defines linear decay via:
$$\lambda(x) = \max\left(1 - \frac{x}{\text{epochs}}, 0\right) \times (1.0 - \text{lrf}) + \text{lrf}$$
where $x$ is the 0-indexed epoch counter, $\text{lr0} = 0.01$, and $\text{lrf} = 0.01$.

- **In the 26-Epoch Schedule:**
  - At Epoch 26 ($x = 25$): $\lambda(25) = (1 - 25/26) \times 0.99 + 0.01 = 0.048077 \implies \text{LR} = \mathbf{0.000481}$. This exactly matches the saved learning rate in `resume_last.pt`.
- **In the Extended 50-Epoch Schedule:**
  - Extending total epochs to 50 repositions Epoch 27 ($x = 26$) as the midpoint of the decay trajectory:
    $$\lambda(26) = \left(1 - \frac{26}{50}\right) \times 0.99 + 0.01 = 0.48 \times 0.99 + 0.01 = 0.485200 \implies \text{LR} = \mathbf{0.004852}$$
  - **Empirical Check (`scratch/test_resume_internals.py`):**
    - `trainer.start_epoch`: Exactly 26 (Epoch 27 1-indexed) — **PASSED**.
    - Saved optimizer momentums: Preserved at `[0.936989, 0.936989, 0.936989]` — **PASSED**.
    - Optimizer state: All 3 parameter groups and 330 tensors restored — **PASSED**.
    - Active LR after `scheduler.step()`: Exactly `0.004852` — **PASSED**.
    - Trajectory to Epoch 50:
      - Epoch 27: $\text{LR} = 0.004852$ (Mosaic ON)
      - Epoch 38 (End Session 3): $\text{LR} = 0.002674$ (Mosaic ON)
      - Epoch 39 (Start Session 4): $\text{LR} = 0.002476$ (Mosaic ON)
      - Epoch 40: $\text{LR} = 0.002278$ (Mosaic ON)
      - Epoch 41 (`close_mosaic` trigger): $\text{LR} = 0.002080$ (Mosaic OFF)
      - Epoch 50 (Final): $\text{LR} = 0.000298$ (Mosaic OFF)
    - Epochs with mosaic disabled: Exactly `[41, 42, 43, 44, 45, 46, 47, 48, 49, 50]` (10 epochs) — **PASSED**.

### 6.2 End-to-End Save and Resume Smoke Test
An end-to-end cycle test (`scratch/test_smoke_save_resume.py`) was executed to confirm checkpoint saving, patching, and resumption without data or optimizer corruption:
1. **Initial Run:** Trained 2 epochs on `coco8`, capturing unstripped checkpoint with optimizer state via `on_model_save` callback.
2. **Patching:** Checkpoint `train_args['epochs']` extended from 2 to 4, `close_mosaic` set to 1.
3. **Resumption:** Resumed cleanly from epoch 3:
   `Resuming training ... from epoch 3 to 4 total epochs`
4. **Augmentation Shift:** Logged `Closing dataloader mosaic` before epoch 4.
5. **Final Evaluation:** Completed epoch 4 with 255 optimizer state tensors intact.
- **Verdict:** **100% PASS** — Checkpoint recovery and scheduler continuity confirmed.

---

## 7. Baseline Metrics Reconciliation

Earlier RoadSentinel reports recorded different metrics depending on whether evaluation was conducted across the full 10,000-image BDD100K validation set or the frozen 300-image development pilot cohort (`PILOT_300_FILENAMES`). The reconciliation below unifies these benchmarks:

### 7.1 Protocol Definitions
- **All-Val Split (10,000 images):** Evaluates mean average precision across all thresholds ($0.001 \le \text{conf} \le 1.0$) and IoU thresholds ($0.50 \dots 0.95$).
- **Pilot Cohort (300 images):** Evaluates operational detection performance at a fixed threshold ($\text{conf} = 0.25$, $\text{NMS IoU} = 0.65$, matching $\text{IoU} = 0.50$) using class-aware, confidence-ordered greedy matching.
- **Small-VRU Definition:** Bounding box area $< 1024\text{ px}^2$ ($< 32^2$) in original $1280 \times 720$ coordinates.

### 7.2 Reconciled Baseline Values
| Evaluation Cohort | Resolution | Metric | Measured Baseline | Context / Origin |
| :--- | :--- | :--- | :--- | :--- |
| **All-Val Split** | 960px | All-Class mAP50 | **61.31%** (0.6131) | Phase 6 Session 2 final evaluation |
| | 960px | All-Class mAP50–95 | **35.41%** (0.3541) | Phase 6 Session 2 final evaluation |
| | 960px | VRU Macro mAP50 | **60.25%** (0.6025) | Person, Rider, Bike, Motor mean |
| | 960px | VRU Macro mAP50–95 | **31.07%** (0.3107) | Person, Rider, Bike, Motor mean |
| **300 Pilot Images** | **960px Native Baseline** | Small-VRU Recall | **42.73%** (385 / 901) | **True 960px baseline** (`BASELINE_960_PILOT`) |
| | **1280px Native Reference**| Small-VRU Recall | **45.17%** (407 / 901) | High-resolution native reference |
| | 960px Native Baseline | VRU Precision | **78.45%** (1052 / 1341) | Operational conf = 0.25 |
| | 960px Native Baseline | VRU Recall | **60.46%** (1052 / 1740) | Operational conf = 0.25 |
| | 960px Native Baseline | VRU F1 Score | **0.6829** | Operational conf = 0.25 |

### 7.3 Clarification on Acceptance Targets (Absolute vs Relative)
All percentage gains are defined as **absolute percentage-point improvements (+pp)**:
- **Baseline 960px Target:** $+2.00\text{ pp}$ over $42.73\% = \mathbf{\ge 44.73\%}$ (recovering $\ge 403 / 901$ small VRUs).
- **1280px Reference Target (Stretch Goal):** $+2.00\text{ pp}$ over $45.17\% = \mathbf{\ge 47.17\%}$ (recovering $\ge 425 / 901$ small VRUs). Beating $45.17\%$ proves 960px extended training exceeds native 1280px inference without extra inference compute.

---

## 8. Multi-Stage Execution Gates

- **Gate A — Preflight (PASSED):** Checkpoint SHA-256 verified, optimizer/EMA state confirmed, scheduler math verified, and end-to-end smoke test passed.
- **Gate B — Kaggle Session 3:** Train Epochs 27–38 on dual-T4 GPUs. Save verified `session3_checkpoint_ep38.pt`, `events_session3.jsonl`, and hashes.
- **Gate C — Kaggle Session 4:** Resume from Epoch 38, train Epochs 39–50 with `close_mosaic=10` active for Epochs 41–50. Save final `best.pt` and `last.pt`.
- **Gate D — Champion Selection:** Evaluate baseline vs extended checkpoints on identical validation images and protocol. Adopt extended checkpoint only if all gates pass.

**Audit Verdict:** **GATE A PREFLIGHT PASSED — PROVENANCE, SCHEDULER, AND RESUME VERIFIED. READY FOR KAGGLE SESSION 3.**
