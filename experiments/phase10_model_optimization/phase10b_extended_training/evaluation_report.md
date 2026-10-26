# RoadSentinel Phase 10B — Controlled YOLO11m Extended Training Evaluation & Implementation Report

**Document ID:** `RS-PHASE10B-EVAL-001`  
**Date:** 2026-10-13  
**Status:** GATE A PASSED — READY FOR KAGGLE SESSION 3  
**Experiment Path:** `experiments/phase10_model_optimization/phase10b_extended_training/`

---

## 1. Executive Summary

RoadSentinel Phase 10B determines whether extending the existing 26-epoch YOLO11m training run to a full 50-epoch total schedule improves Vulnerable Road User (VRU) detection—specifically small-VRU recall (< 1024 px²)—without destabilizing precision or exceeding Kaggle compute limits.

Following initial pre-execution auditing, two critical execution issues identified during review were empirically addressed under **Gate A (Preflight)**:
1. **Scheduler-Level Verification & Resume Mechanics:** We established that extending `train_args['epochs']` from 26 to 50 repositions Epoch 27 as the exact midpoint of the linear decay schedule ($\text{LR} = 0.004852$), preserves all 330 optimizer state tensors and momentum ($0.936989$), and disables mosaic augmentation for exactly the final 10 epochs (Epochs 41–50). An end-to-end save-patch-resume smoke test in the installed Ultralytics `8.4.174` environment completed with 100% pass rate.
2. **Reconciliation of Baseline Metrics:** We reconciled the historical 45.17% small-VRU metric (which represents the **1280px native reference** on the 300-image development cohort) with the **960px baseline** on the same cohort (**42.73%**). Acceptance criteria are now formally defined as **absolute percentage-point improvements (+pp)**.

---

## 2. Checkpoint Provenance and Integrity Verification

Forensic SHA-256 validation confirmed that the frozen Phase 6 baseline is intact and that the resume candidate retains complete optimizer history:

| Checkpoint Name | File Path | Size (Bytes) | SHA-256 Hash | Status / Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Frozen Best** | `scratch/kaggle_upload_weights/best.pt` | 40,539,943 | `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` | **VERIFIED MATCH** (Frozen Phase 6 reference) |
| **Resume Checkpoint** | `experiments/phase6_session2_error/resume_last.pt` | 80,805,199 | `db64f798410ed392e2af35217661ef1b8ecab32e4a6f431600468d3c44915305` | **VERIFIED RESUME** (Epoch 25 index = 26 completed epochs) |
| **Corrupted Target** | `experiments/phase6_session2_error/best.pt` | 3,132 | `9622d057a66978434311ce63a8a071efbeafb5a03e1cfcf0d367e9f3b5ba540d` | **REJECTED** (Truncated JSON artifact) |

---

## 3. Gate A — Preflight Verification (Scheduler & Resume Mechanics)

### 3.1 Learning Rate Scheduler Behavior on Resume
In Ultralytics `8.4.174`, `BaseTrainer._setup_scheduler()` defines linear decay:
$$\lambda(x) = \max\left(1 - \frac{x}{\text{epochs}}, 0\right) \times (1.0 - \text{lrf}) + \text{lrf}$$
where $\text{lr0} = 0.01$ and $\text{lrf} = 0.01$.

- At Epoch 26 ($x = 25$ of 26): $\text{LR} = 0.01 \times (0.048077) = \mathbf{0.000481}$ (matches recorded state in `resume_last.pt`).
- At Epoch 27 ($x = 26$ of 50): Because total epochs are extended to 50, Epoch 27 is the midpoint of the decay trajectory:
  $$\lambda(26) = \left(1 - \frac{26}{50}\right) \times 0.99 + 0.01 = 0.48 \times 0.99 + 0.01 = 0.485200 \implies \mathbf{\text{LR} = 0.004852}$$
- **Empirical Execution Verification (`scratch/test_resume_internals.py`):**
  - Training resumes cleanly at **Epoch 27** (`trainer.start_epoch = 26`).
  - Optimizer momentums loaded: `[0.936989, 0.936989, 0.936989]` (fully preserved).
  - Optimizer state tensors loaded: 330 tensors across 3 parameter groups.
  - Active learning rate at start of Epoch 27: Exactly `0.004852`.
  - Augmentation: Mosaic remains active from Epoch 27 through Epoch 40; turns off automatically at Epoch 41 (`close_mosaic=10`), running unaugmented for Epochs 41–50 (exactly 10 epochs).

### 3.2 End-to-End Save/Resume Smoke Test
An end-to-end cycle test (`scratch/test_smoke_save_resume.py`) verified that:
1. Checkpoints saved via the production `on_model_save` callback retain unstripped optimizer buffers.
2. Checkpoint patching (`train_args['epochs']: 2 -> 4`, `close_mosaic: 1`) bypasses `check_resume()` assertion errors.
3. Resumed training runs seamlessly across the boundary and successfully saves recoverable subsequent checkpoints.

---

## 4. Reconciled Baseline Performance & Evaluation Protocol

### 4.1 Evaluation Protocol Specifications
To ensure 100% apples-to-apples comparability between the baseline and extended models:
- **Cohort:** The frozen 300-image development pilot cohort (`PILOT_300_FILENAMES`).
- **Image Size:** 960 × 960 px letterboxed (`imgsz=960`).
- **Operational Confidence Threshold:** `conf = 0.25`.
- **NMS Configuration:** `iou = 0.65`.
- **Matching IoU Threshold:** `iou = 0.50` (Class-aware, confidence-ordered greedy matching).
- **Small-Object Definition:** Bounding box area $< 1024\text{ px}^2$ ($< 32^2$) in original $1280 \times 720$ coordinates.
- **VRU Classes:** Person, Rider, Bike, Motor.

### 4.2 Reconciled Baseline Metrics

| Metric | 960px Frozen Baseline (Conf A) | 1280px Native Reference (Conf B) | Evaluation Scope |
| :--- | :--- | :--- | :--- |
| **All-Class mAP50** | **61.31%** (0.6131) | — | BDD100K Full Val Split |
| **All-Class mAP50–95** | **35.41%** (0.3541) | — | BDD100K Full Val Split |
| **VRU Macro mAP50** | **60.25%** (0.6025) | — | BDD100K Full Val Split |
| **VRU Macro mAP50–95** | **31.07%** (0.3107) | — | BDD100K Full Val Split |
| **Small-VRU Recall (<1024 px²)** | **42.73%** (385 / 901 TP) | **45.17%** (407 / 901 TP) | 300 Pilot Images @ conf=0.25 |
| **VRU Precision** | **78.45%** (1052 / 1341) | **78.20%** (1076 / 1376) | 300 Pilot Images @ conf=0.25 |
| **VRU Recall** | **60.46%** (1052 / 1740) | **61.84%** (1076 / 1740) | 300 Pilot Images @ conf=0.25 |
| **VRU F1 Score** | **0.6829** | **0.6906** | 300 Pilot Images @ conf=0.25 |
| **All-Class Recall** | **71.88%** (4954 / 6892) | **73.64%** (5075 / 6892) | 300 Pilot Images @ conf=0.25 |
| **All-Class Precision** | **77.93%** (4954 / 6357) | **77.68%** (5075 / 6533) | 300 Pilot Images @ conf=0.25 |

### 4.3 Per-Class Baseline (300 Pilot Images @ conf=0.25)
- **Person:** Baseline Recall = 61.91% (946 TP), Precision = 79.90% | Native 1280px Recall = 63.22% (966 TP)
- **Rider:** Baseline Recall = 55.17% (32 TP), Precision = 69.57% | Native 1280px Recall = 56.90% (33 TP)
- **Bike:** Baseline Recall = 51.64% (63 TP), Precision = 67.74% | Native 1280px Recall = 55.74% (68 TP)
- **Motor:** Baseline Recall = 34.38% (11 TP), Precision = 61.11% | Native 1280px Recall = 28.12% (9 TP)

---

## 5. Execution Plan & Kaggle Session Partitioning

Throughput measurement: **2,001.3 seconds (33.36 minutes) per epoch** on dual Tesla T4 GPUs.
24 remaining epochs = **13.34 hours**, exceeding Kaggle's 12.0-hour hard limit.

```
+---------------------------------------------------------------------------------------+
| GATE A: Preflight (Complete)                                                          |
| - Verify SHA-256 hashes, test resume mechanics, verify scheduler math, run smoke test  |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| GATE B: Kaggle Session 3 (Epochs 27 to 38)                                            |
| - Runtime: ~6.92 hours (under 12h cutoff)                                             |
| - Generates session3_checkpoint_ep38.pt, events_session3.jsonl                        |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| GATE C: Kaggle Session 4 (Epochs 39 to 50)                                            |
| - Runtime: ~6.92 hours                                                                |
| - close_mosaic=10 active for Epochs 41 to 50                                          |
| - Generates best.pt, last.pt, and training_results.csv                                |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| GATE D: Champion Selection                                                            |
| - Evaluate extended best.pt against frozen baseline under identical protocol          |
| - Determine adoption or baseline retention                                            |
+---------------------------------------------------------------------------------------+
```

---

## 6. Champion Selection Acceptance Gates (Gate D)

All criteria are defined as **absolute percentage-point (+pp)** improvements:

| Gate | Target Metric | Baseline Reference | Required Target | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **GATE 1 (Core)** | Small-VRU Recall (<1024 px²) | **42.73%** (960px baseline) | **$\ge$ 44.73% (+2.00 pp)** | Core research priority |
| **GATE 1 (Stretch)**| Small-VRU Recall (<1024 px²) | **45.17%** (1280px native) | **$\ge$ 47.17% (+2.00 pp)** | Exceeds native 1280px reference |
| **GATE 2** | VRU Macro mAP50 | **60.25%** | **$\ge$ 61.75% (+1.50 pp)** | Balanced VRU performance |
| **GATE 3** | VRU Precision | **78.45%** | **$\ge$ 75.00% (-3.45 pp max)** | Anti-hallucination constraint |
| **GATE 4** | All-Class mAP50 | **61.31%** | **$\ge$ 61.31% (+0.00 pp)** | Zero catastrophic forgetting |
| **GATE 5** | Checkpoint Integrity | Valid PyTorch | **Clean FP16 & SHA-256** | Production export integrity |

---

## 7. Actionable Recommendation

1. **Gate A is officially PASSED.** All scheduler, momentum, checkpoint recovery, and baseline consistency questions have been empirically answered and validated.
2. **Proceed to Gate B (Kaggle Session 3):** Launch `kaggle_jobs/phase10b_extended_training/` targeting Epochs 27–38.
3. The reported 50-epoch gains remain explicit target gates, not promises. The Phase 6 baseline remains the active production champion until Gate D evaluation confirms that all acceptance thresholds are met.
