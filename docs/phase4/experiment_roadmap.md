# RoadSentinel Phase 4A — Staged Experimentation Roadmap

**Date:** October 10, 2026  
**Auditor:** Principal Computer Vision Researcher & Experimental ML Scientist  
**Status:** COMPLETE — Controlled Stage-Gate Experiment Specification  
**Governing Principle:** Do not advance to higher-cost stages until lower-cost diagnostic and post-processing stages have established clear empirical ceilings.

---

## Roadmap Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 0: Evaluation Integrity (Complete — Verified Hash, Manifest & Metrics) │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Failure Atlas & Diagnostic Extraction (300 Stratified Images)       │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Post-Processing & Fusion Optimization (Zero Retraining Cost)        │
│          Exp 2.1: Dual-Threshold Crop Calibration                            │
│          Exp 2.2: Class-Specific Threshold Sweeps                            │
│          Exp 2.3: Weighted Box Fusion (WBF) vs Greedy NMS                     │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Crop-Policy & Spatial Geometry Optimization                         │
│          Exp 3.1: Horizon-Prioritized Spatial Uncertainty                    │
│          Exp 3.2: Crop Aspect Ratio & Context Padding Sweeps                 │
│          Exp 3.3: Crop Mini-Batching for Real-Time Latency                   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 4: Targeted Detector Fine-Tuning (BDD100K 70k Train Split)            │
│          Exp 4.1: Sliced Fine-Tuning (SFT) with Full+Crop Mixtures           │
│          Exp 4.2: Class-Balanced Copy-Paste Augmentation for Rare VRUs        │
│          Exp 4.3: Native 1280px Fine-Tuning Comparison                       │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 5: Architecture Scaling & Engine Acceleration                          │
│          Exp 5.1: YOLO11m Capacity Diagnostic                                │
│          Exp 5.2: TensorRT FP16 Compilation & Edge Latency Benchmark         │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 6: Temporal Tracking Formulation (Video Split Feasibility Diagnostic)  │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 7: Final Multi-Metric Model Selection & Quarantined Test Evaluation    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1 — Failure Atlas & Diagnostic Extraction

### Experiment 1.1: Stratified Error Extraction, Visual Audit & Prediction Collection
- **Hypothesis:** Visual analysis of $300$ stratified failure cases will demonstrate that $>80\%$ of crop false positives are isolated unconfirmed background textures and $>50\%$ of persistent small misses are below human visual recognizability.
- **Independent Variables:** Sampling stratum (Recovered Small VRUs, Persistent Misses, High-Conf FPs, Rare VRUs, Boundary Truncations, Domain Shifts).
- **Controlled Variables:** Base checkpoint (`weights/bdd100k_yolo11s.pt`), evaluation threshold ($\tau = 0.25$), IoU match ($0.50$).
- **Dataset Split:** BDD100K Validation Split ($10,000$ images, selecting $300$ stratified images).
- **Compute Requirements:** Local CPU/GPU, runtime $< 5\text{ minutes}$.
- **Effort Level:** Low ($2 - 3$ hours).
- **Required Artifacts:** [experiments/phase3_foveation/runs/kaggle_run_v2/per_image_records.jsonl](file:///c:/Users/hiten/Downloads/cv/experiments/phase3_foveation/runs/kaggle_run_v2/per_image_records.jsonl), validation images/labels.
- **Output Artifacts:**
  1. `experiments/phase4_diagnostics/cohort_metadata.json` and $300$ diagnostic side-by-side PNGs.
  2. `experiments/phase4_diagnostics/cohort_raw_predictions.json` (stores complete candidate bounding-box coordinates, continuous confidence floats, classes, and source tags for full-frame and crop passes to support offline threshold sweeps).
- **Confounding Risks:** Subjective human interpretation bias during inspection.
- **Acceptance Criteria:** Produces a verified error atlas classifying all 300 cases into the error taxonomy and persists complete candidate predictions for the cohort.
- **Stopping Rule:** Complete review of all 300 cases.
- **Dependencies:** Stage 0 integrity sign-off (Completed).

---

## Stage 2 — Threshold, Score Calibration & Fusion Optimization

### Experiment 2.1: Dual-Threshold Crop-Source Score Calibration (Pilot & Sweep)
- **Hypothesis (Unvalidated):** Setting a higher confidence threshold ($\tau_{\text{isolated}} \ge 0.40$) for crop detections that lack an overlapping full-frame candidate, while maintaining $\tau_{\text{confirmed}} \ge 0.20$ for confirming detections, will eliminate $>50\%$ of crop false positives while preserving $>92\%$ of recovered small VRU true positives.
- **Hypothesized Target Goal:** Tests whether VRU Precision can be lifted toward $\ge 68.0-70.0\%$ with VRU F1 $\ge 0.6550$. *Note: This figure is an experimental target to be tested, not a promised outcome or justification for unverified approval.*
- **Execution Workflow:**
  1. *Step A (Pilot):* Run calibration grid search on `cohort_raw_predictions.json` gathered in Exp 1.1 to measure empirical precision/recall curves.
  2. *Step B (Full Val):* If the pilot confirms the hypothesis, execute the full 10,000-image evaluation on Kaggle GPU.
- **Independent Variables:**
  - $\tau_{\text{isolated}} \in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]$
  - $\tau_{\text{confirmed}} \in [0.15, 0.20, 0.25]$
  - Overlap association threshold $\text{IoU}_{\text{assoc}} \in [0.20, 0.30, 0.40]$
- **Controlled Variables:** Model weights, crop generation policy, validation manifest ($10,000$ images).
- **Dataset Split:** BDD100K Validation Split ($10,000$ images).
- **Compute Requirements:** Single Kaggle/Colab T4 GPU, runtime $\approx 45\text{ minutes}$.
- **Effort Level:** Low ($3 - 4$ hours code & execution).
- **Required Artifacts:** `evaluate_phase3_foveation.py` with dual-threshold parameterization, `cohort_raw_predictions.json`.
- **Metrics Evaluated:** VRU Precision, VRU Recall, VRU F1, Small VRU Recall, Overall mAP50.
- **Confounding Risks:** Overfitting threshold grid to validation split.
- **Acceptance Criteria:**
  - Empirical VRU Precision improves to $\ge 68.0\%$ (baseline: $62.03\%$).
  - VRU F1 increases from $0.6237$ to $\ge 0.6500$.
  - Small VRU Recall does not drop below $44.5\%$ (baseline: $46.83\%$).
- **Stopping Rule:** Terminate if precision increases by $< 2\%$ at the cost of $> 5\%$ small VRU recall.
- **Dependencies:** Experiment 1.1 visual confirmation and raw prediction capture.

### Experiment 2.2: Class-Specific Threshold Optimization
- **Hypothesis:** Decoupling the single global threshold $\tau=0.25$ into class-specific thresholds $(\tau_{\text{person}}, \tau_{\text{rider}}, \tau_{\text{bike}}, \tau_{\text{motor}})$ calibrated to maximize per-class F1 will lift macro VRU metrics without degrading dominant classes.
- **Independent Variables:** $\tau_c \in [0.15, 0.40]$ independently optimized per class.
- **Controlled Variables:** Model weights, foveation policy, fusion method.
- **Dataset Split:** $50\%$ validation split for tuning ($5,000$ images), $50\%$ validation split for validation ($5,000$ images).
- **Compute Requirements:** Local CPU post-processing grid search, runtime $< 15\text{ minutes}$.
- **Effort Level:** Low ($2 - 3$ hours).
- **Acceptance Criteria:** VRU Macro mAP50 increases by $\ge +0.015$, `rider` and `bike` recall increase by $\ge +4.0\%$.
- **Dependencies:** Experiment 2.1.

### Experiment 2.3: Weighted Box Fusion (WBF) vs. Greedy NMS
- **Hypothesis:** Merging multi-scale candidates via Weighted Box Fusion with confidence consensus will achieve higher coordinate accuracy and penalize isolated crop hallucinations compared to greedy NMS deduplication.
- **Independent Variables:** Fusion algorithm (Greedy NMS vs Soft-NMS vs WBF); fusion weights $(w_{\text{ff}}, w_{\text{crop}})$.
- **Controlled Variables:** Candidate predictions from Exp 2.1 and 2.2.
- **Dataset Split:** BDD100K Validation Split ($10,000$ images).
- **Compute Requirements:** Single GPU, runtime $\approx 30\text{ minutes}$.
- **Effort Level:** Medium ($4 - 6$ hours).
- **Acceptance Criteria:** Overall mAP50 improves over Active Foveation baseline ($0.5403 \rightarrow \ge 0.5500$).
- **Dependencies:** Experiment 2.1 and 2.2.

---

## Stage 3 — Crop-Policy & Spatial Geometry Optimization

### Experiment 3.1: Horizon-Prioritized Spatial Uncertainty Policy
- **Hypothesis:** Incorporating a vertical road-corridor spatial prior ($y \in [0.35H, 0.70H]$) into the uncertainty scoring function will suppress irrelevant crops (sky, car hood) by $\ge 15\%$, improving throughput and small-VRU crop density.
- **Independent Variables:** Spatial corridor vertical bounds $[y_{\min}, y_{\max}]$, Gaussian vertical decay bandwidth $\sigma_y$.
- **Controlled Variables:** Model checkpoint, Stage 2 optimized post-processing thresholds.
- **Dataset Split:** BDD100K Validation Split ($10,000$ images).
- **Compute Requirements:** Single GPU, runtime $\approx 60\text{ minutes}$.
- **Effort Level:** Medium.
- **Acceptance Criteria:** Crop trigger rate drops from $48.37\%$ to $\le 42.0\%$, throughput increases from $27.98\text{ FPS}$ to $\ge 31.0\text{ FPS}$, Small VRU Recall remains $\ge 46.0\%$.
- **Dependencies:** Stage 2 completion.

### Experiment 3.2: Crop Mini-Batching Execution
- **Hypothesis:** Stacking $K=2$ crops into a single tensor `(2, 3, 640, 640)` for a single forward pass will reduce P95 pipeline latency from $39.72\text{ ms}$ to $< 30.0\text{ ms}$.
- **Independent Variables:** Crop execution mode (Sequential $2 \times (1, 3, 640, 640)$ vs Batched $(2, 3, 640, 640)$).
- **Controlled Variables:** Hardware (Tesla T4 GPU), model weights, image inputs.
- **Dataset Split:** $50$-image isolated latency benchmark.
- **Compute Requirements:** GPU runtime $< 5\text{ minutes}$.
- **Effort Level:** Low ($1 - 2$ hours).
- **Acceptance Criteria:** Pipeline P95 latency $< 30.0\text{ ms}$; mean pipeline latency $< 20.0\text{ ms}$.
- **Dependencies:** None (can be implemented in parallel).

---

## Stage 4 — Targeted Detector Fine-Tuning (BDD100K Train Split)

### Experiment 4.1: Sliced Fine-Tuning (SFT) with Full+Crop Mixtures
- **Hypothesis:** Fine-tuning `weights/bdd100k_yolo11s.pt` on a dataset mixture of $80\%$ full-frame images ($960\times 960$) and $20\%$ cropped patches ($640\times 640$) for 15 epochs will align the detector's internal representations to both scales, reducing crop-induced false alarms.
- **Independent Variables:** Fine-tuning dataset composition (Full-frame only vs $80/20$ Full+Crop mixture); learning rate ($lr_0 = 10^{-4}$ with cosine decay).
- **Controlled Variables:** Pretrained weights (`bdd100k_yolo11s.pt`), optimizer (AdamW), batch size (8).
- **Dataset Split:** BDD100K Training Split ($70,000$ images).
- **Compute Requirements:** Cloud GPU (Tesla T4 or A100), $12 - 16\text{ GPU-hours}$.
- **Effort Level:** High ($1 - 2$ days).
- **Acceptance Criteria:** Standalone FullFrame-960 mAP50 increases from $0.5544$ to $\ge 0.5650$; crop-induced FP rate drops by $\ge 25\%$.
- **Dependencies:** Stage 2 and Stage 3 completion.

### Experiment 4.2: Class-Balanced Copy-Paste Augmentation for Rare VRUs
- **Hypothesis:** Applying Copy-Paste augmentation to paste instances of rare VRUs (`rider`, `motor`, `bike`) into empty or background road scenes during training will overcome the $30:1$ class imbalance, lifting `rider` and `motor` recall by $\ge +8\%$.
- **Independent Variables:** Copy-paste frequency ($p = 0.3$), rare-class sampling weights.
- **Controlled Variables:** SFT training schedule from Exp 4.1.
- **Dataset Split:** BDD100K Training Split ($70,000$ images).
- **Compute Requirements:** $12 - 16\text{ GPU-hours}$.
- **Effort Level:** High.
- **Acceptance Criteria:** VRU Macro mAP50 lifts by $\ge +0.025$; `rider` recall $\ge 52\%$, `motor` recall $\ge 50\%$.
- **Dependencies:** Experiment 4.1.

---

## Stage 5 — Architecture Scaling & Engine Acceleration

### Experiment 5.1: YOLO11m Capacity Diagnostic
- **Hypothesis:** Training a medium-scale YOLO11m detector ($20.1\text{M}$ params) at $imgsz=960$ will demonstrate whether feature capacity is an insurmountable ceiling for small VRUs.
- **Independent Variables:** Model architecture (`YOLO11s` vs `YOLO11m`).
- **Controlled Variables:** BDD100K train split, 50 epochs, imgsz=960.
- **Dataset Split:** BDD100K Train ($70,000$ images) & Val ($10,000$ images).
- **Compute Requirements:** $25 - 35\text{ GPU-hours}$.
- **Effort Level:** High.
- **Acceptance Criteria:** Overall mAP50 improvement of $\ge +0.030$ and Small VRU Recall gain $\ge +4.0\%$ over YOLO11s.
- **Stopping Rule:** If mAP50 gain is $< +0.015$, reject YOLO11m due to $2.5\times$ latency penalty.
- **Dependencies:** Stage 4 completion.

### Experiment 5.2: TensorRT FP16 Engine Compilation
- **Hypothesis:** Compiling the fine-tuned YOLO11s model into a TensorRT FP16 engine with CUDA stream management will achieve $> 50\text{ FPS}$ full-pipeline throughput and $< 10\text{ ms}$ batch-1 latency.
- **Independent Variables:** Runtime engine (PyTorch Native CUDA vs TensorRT FP16).
- **Controlled Variables:** Checkpoint weights, input resolution, GPU hardware (Tesla T4).
- **Compute Requirements:** Local GPU, runtime $< 30\text{ minutes}$.
- **Effort Level:** Medium ($4 - 6$ hours).
- **Acceptance Criteria:** Pipeline P50 latency $\le 10.0\text{ ms}$; throughput $\ge 50.0\text{ FPS}$.
- **Dependencies:** Final model checkpoint freeze.

---

## Stage 6 — Temporal Tracking Formulation (Video Split Feasibility)

### Experiment 6.1: Temporal Persistence Feasibility Study
- **Hypothesis:** When evaluated on continuous video sequences (BDD100K video tracking split), tracking-assisted foveation will maintain persistent tracks of VRUs across consecutive frames with $< 20\%$ crop trigger rate.
- **Scope Note:** Strictly separate from single-image keyframe benchmarks. Conducted only on BDD100K tracking sequences.
- **Dependencies:** Single-image detector freeze.

---

## Stage 7 — Final Model Selection & Quarantined Test Split Evaluation

### Experiment 7.1: Final Candidate Selection & Frozen Test Evaluation
- **Protocol:**
  1. Freeze top $2$ candidate configurations evaluated strictly on the $10,000$-image validation split using the multi-metric policy (Section 9).
  2. Perform code freeze: lock all weights, thresholds, configs, and random seeds.
  3. Single-pass evaluation on the quarantined **$20,000$-image BDD100K Test Split**.
  4. Full transparency reporting: report all metrics, failure cases, and limitations. Zero post-hoc adjustments.
