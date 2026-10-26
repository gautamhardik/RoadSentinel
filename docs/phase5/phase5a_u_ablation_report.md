# RoadSentinel Phase 5A-U — Unweighted Fine-Tuning Ablation Report

> **CORRECTION (2026-10-11, Phase 6 Stage 0).** Superseded by the [Stage 0 matched-policy evaluation](../phase6/stage0_matched_evaluation_report.md). Under a matched comparison (same evaluator, same 9,700 images, threshold swept), the original YOLO11s has equal or higher VRU recall and small-VRU recall than both Phase 5A-U and Phase 5A-R at every precision level from 0.55 to 0.75. Recomputed at the standard confidence floor (Stage 0b, conf 0.001), raw VRU mAP50 is 0.5157 (original) vs 0.5077 (5A-U) / 0.5122 (5A-R); only rider AP improved with class weighting (0.4598 vs 0.4413). The fixed-threshold (tau = 0.25) recall gains reported for the fine-tuned models reflect higher confidence scores, not a better detector. These are validation-based results; the original checkpoint's training provenance is uncertain.
> The run described here completed; its status line above is stale.

**Status:** ACTIVE REMOTE EXECUTION ON KAGGLE GPU  
**Experiment Identifier:** Phase 5A-U (Unweighted Fine-Tuning Ablation)  
**Remote Kernel ID:** `hardikgautam25/roadsentinel-phase-5a-vru-model-training`  
**Remote Worker Status:** `KernelWorkerStatus.RUNNING`  
**Execution Device:** NVIDIA Tesla T4 (16GB) / P100 (16GB)  
**Launch Timestamp:** 2026-10-10 00:13:53 UTC (05:43:53 IST)  
**Test Split Quarantine:** Maintained ($0$ test images accessed; $20,000$ images strictly quarantined)  

---

## 1. Executive Summary & Experiment Reclassification

The remote training job submitted in Phase 5A (`hardikgautam25/roadsentinel-phase-5a-vru-model-training`) is actively running on a remote Kaggle GPU container. A thorough forensic audit of the submitted script against the runtime mechanics of Ultralytics 8.4.174 revealed that the planned per-class loss weights were not active in the submitted code due to criterion initialization timing.

In strict adherence to scientific integrity:
> **"Do not treat an unweighted run as evidence for the class-weighted method. Label this run accurately as Phase 5A-U: Unweighted Fine-Tuning Ablation. The label is retrospective; do not modify historical records or pretend the original configuration was class-weighted."**

This job has been formally classified as **Phase 5A-U: Unweighted Fine-Tuning Ablation**. Rather than terminating the job and discarding GPU compute, the run is permitted to finish naturally to serve as an indispensable ablation baseline measuring the isolated effects of:
1. Increased global classification loss gain (`cls=1.2` vs default `0.5`).
2. High-resolution input training ($960\times 960$).
3. Multi-scale Mosaic ($0.80$) and Scale jitter ($0.40$).
4. AdamW optimization ($3 \times 10^{-4}$ initial LR, cosine schedule to $1.5 \times 10^{-5}$).

---

## 2. Remote Job Monitoring & Telemetry

* **Status Query Command:**
  ```bash
  kaggle kernels status hardikgautam25/roadsentinel-phase-5a-vru-model-training
  ```
* **Execution Progress:** The job was initiated at 05:43:53 IST. Total estimated duration for 8 epochs on 70,000 images is ~3.5 to 4.5 hours.
* **Kaggle CLI Output Stream:** The public Kaggle CLI does not stream real-time standard output while a batch kernel is running; execution output and checkpoints are made available upon kernel termination (`COMPLETE` or `FAILED`).
* **Non-Duplication Guarantee:** In accordance with Phase 1 instructions, no duplicate job has been submitted and no premature termination has been issued.

---

## 3. Effective Training Configuration (Phase 5A-U)

| Parameter | Submitted Value | Effective Runtime Behavior |
|---|---|---|
| **Base Checkpoint** | `weights/bdd100k_yolo11s.pt` | Verified SHA-256: `A4747292648F37F68AB997BEA65A36391C41356F35E309A9F3D2EB1D3EA8D8E3` |
| **Input Resolution** | $960 \times 960$ | Active |
| **Batch Size** | 16 | Active (gradient accumulation $= 1$) |
| **Epochs** | 8 | Active |
| **Optimizer** | AdamW | Active ($\text{lr}_0 = 3\times 10^{-4}, \text{lr}_f = 1.5\times 10^{-5}$, weight decay $5\times 10^{-4}$) |
| **Class Loss Weights** | Specified in YAML | **INACTIVE (Unweighted: all classes $= 1.0$)** |
| **Classification Gain** | `cls = 1.2` | **Active** (scales BCE loss by $1.2\times$ globally) |
| **Copy-Paste** | `copy_paste = 0.15` | **INACTIVE** (bypassed due to empty polygon contours) |
| **Mosaic / Scale** | `mosaic = 0.8, scale = 0.4` | **Active** (closed in final 2 epochs) |

---

## 4. Post-Completion Retrieval & Audit Protocol

When the remote kernel transitions to `KernelWorkerStatus.COMPLETE`:
1. Retrieve artifacts into dedicated directory:
   ```bash
   kaggle kernels output hardikgautam25/roadsentinel-phase-5a-vru-model-training -p experiments/phase5a_modeling/runs/phase5a_u_ablation/
   ```
2. Verify checkpoint existence, non-zero file size, and SHA-256 hash.
3. Validate checkpoint loading via PyTorch/Ultralytics.
4. Evaluate on the canonical 9,700-image confirmation cohort using the hardened ActiveFoveation-960 evaluator.
