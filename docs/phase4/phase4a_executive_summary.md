# RoadSentinel Phase 4A — Master Executive Summary & Optimization Strategy

**Date:** October 10, 2026  
**Auditor:** Principal Computer Vision Researcher, Autonomous Perception Engineer & ML Scientist  
**Status:** COMPLETE — Authoritative Research Strategy & Roadmap  
**Document Series:** Phase 4A Master Technical Report  

---

## 1. Current Verified Baseline

All metrics below are derived directly from genuine Kaggle Tesla T4 GPU evaluation records ([experiments/phase3_foveation/runs/kaggle_run_v2/raw_metrics.json](file:///c:/Users/hiten/Downloads/cv/experiments/phase3_foveation/runs/kaggle_run_v2/raw_metrics.json) and [per_image_records.jsonl](file:///c:/Users/hiten/Downloads/cv/experiments/phase3_foveation/runs/kaggle_run_v2/per_image_records.jsonl)), audited and verified in [docs/phase4/phase4a_repository_audit.md](file:///c:/Users/hiten/Downloads/cv/docs/phase4/phase4a_repository_audit.md):

| Configuration | Type | Imgsz | Crops / Frame | Overall mAP@50 | VRU Macro mAP@50 | VRU Precision ($\tau=0.25$) | VRU Recall ($\tau=0.25$) | VRU F1 ($\tau=0.25$) | Small VRU Recall ($N=7029$) | Pipeline P50 Latency | Pipeline P95 Latency | Throughput (FPS) | Peak Tensor VRAM |
|:---|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **FullFrame-640** | Full Frame | 640 | 0 | 0.4819 | 0.4428 | 0.7319 | 0.4836 | 0.5824 | 0.2325 | 13.73 ms | 15.30 ms | 40.77 | 166.6 MB |
| **FullFrame-960** | Full Frame | 960 | 0 | 0.5544 | 0.5262 | 0.7524 | 0.5692 | 0.6481 | 0.3639 | 17.44 ms | 22.02 ms | 44.40 | 196.8 MB |
| **FullFrame-1280** | Full Frame | 1280 | 0 | 0.5638 | 0.5367 | 0.7617 | 0.5807 | 0.6590 | 0.4033 | 23.23 ms | 24.56 ms | 33.00 | 240.2 MB |
| **FixedGrid-960** | Fixed Grid | 960 | 2 | 0.5032 | 0.4800 | 0.5643 | 0.6343 | 0.5972 | 0.4629 | 37.80 ms | 39.23 ms | 16.94 | 196.8 MB |
| **RandomCrop-960** | Random Crop | 960 | 2 | 0.5048 | 0.4935 | 0.6158 | 0.6183 | 0.6171 | 0.4446 | 39.71 ms | 40.90 | 17.47 | 196.1 MB |
| **ActiveFoveation-960** | Active Foveation | 960 | 2 | **0.5403** | **0.5120** | **0.6203** | **0.6271** | **0.6237** | **0.4683** | **18.56 ms** | **39.72 ms** | **27.98** | **196.8 MB** |

### Verified Physical Quantities:
- **Small VRU True Positives:** $3,292$ (Active Foveation) vs $2,558$ (FullFrame-960) $\rightarrow$ **$+734$ small VRUs recovered ($+28.7\%$ relative gain, $+10.44\%$ absolute recall gain)**.
- **Total VRU True Positives:** $9,641$ (Active Foveation) vs $8,750$ (FullFrame-960) $\rightarrow$ **$+891$ total VRUs recovered**.
- **The Core Bottleneck:** Active Foveation generates **$5,901$ False Positives** vs $2,879$ in FullFrame-960 ($\mathbf{+3,022\text{ FPs}}$), dropping precision from $75.24\%$ to $62.03\%$.
- **Crop Budget Efficiency:** Triggered crops on **$48.37\%$ of frames** ($4,837$ frames); executed **$7,666$ total crop passes** out of $20,000$ possible $\rightarrow$ **$61.67\%$ crop compute savings**.

---

## 2. Claims That Remain Unverified

Prior to Phase 4 experimentation, the following claims must remain explicitly marked as **UNVERIFIED**:
1. **UNVERIFIED: Optical Limit of 720p Sensor.** Whether the $5,142$ VRUs persistently missed across all resolutions are optically resolvable by a human observer or represent physical sub-pixel blur / label noise. *(Resolved in Stage 1 Failure Atlas).*
2. **UNVERIFIED: Transferability of Sliced Fine-Tuning Gains to BDD100K.** SFT literature gains ($+3-5\text{ mAP}$) were established on aerial images (VisDrone/DOTA); its exact quantitative impact on perspective driving scenes with road clutter remains unverified.
3. **UNVERIFIED: Standalone Capacity Ceiling of YOLO11s.** Whether the feature extractor in YOLO11s is incapable of representing distant VRUs, or if its bottleneck is purely post-processing and training resolution. *(Resolved in Stage 5).*
4. **UNVERIFIED: Generalizability to Quarantined Test Set.** Performance on the $20,000$-image test split has not been measured and must remain quarantined.

---

## 3. Top Ten Improvement Opportunities (Ranked)

| Rank | Priority | Opportunity | Expected Primary Mechanism | Target Metric Impact | Effort & Compute |
|:---:|:---:|:---|:---|:---|:---:|
| **1** | **P1** | **Dual-Threshold Crop Score Calibration** | Require $\tau_{\text{isolated}} \ge 0.40$ for unconfirmed crop detections; keep $\tau \ge 0.20$ for confirming detections. | **VRU Precision $\uparrow +8 - 12\%$**, F1 $\uparrow +0.03 - 0.05$ | Low / $< 1$ GPU-hr |
| **2** | **P1** | **Class-Specific Thresholds ($\tau_c$)** | Decouple $\tau_{\text{person}} \approx 0.30$ from $\tau_{\text{rider}}, \tau_{\text{bike}}, \tau_{\text{motor}} \approx 0.18-0.22$. | **VRU Macro mAP50 $\uparrow +0.015 - 0.025$** | Low / $< 30$ mins |
| **3** | **P1** | **Weighted Box Fusion (WBF)** | Blend full-frame and crop boxes with consensus score boosting; penalize single-scale outliers. | **Overall mAP50 $\uparrow +0.010 - 0.020$** | Low-Med / $< 1$ GPU-hr |
| **4** | **P1** | **Horizon-Prioritized Spatial Prior** | Constrain crop proposals to optical vanishing corridor ($y \in [0.35H, 0.70H]$); eliminate sky/hood crops. | **Throughput $\uparrow +15\%$**, Crop FPs $\downarrow 20\%$ | Low / $< 1$ GPU-hr |
| **5** | **P1** | **Crop Mini-Batch Forward Pass** | Stack $K=2$ crops into single tensor `(2, 3, 640, 640)` for concurrent GPU Tensor Core execution. | **P95 Latency $\downarrow 30\%$ (to $<30\text{ ms}$)** | Low / $< 1$ hr |
| **6** | **P2** | **Sliced Fine-Tuning (SFT)** | Fine-tune YOLO11s on $80\%$ full-frame + $20\%$ cropped patches from BDD100K 70k train split. | **Crop False Alarms $\downarrow 40\%$**, Small Rec $\uparrow +3\%$ | Med / $12-16$ GPU-hrs |
| **7** | **P2** | **Class-Balanced Copy-Paste Augmentation** | Apply Copy-Paste on rare classes (`rider`, `motor`, `bike`) during training to overcome $30:1$ imbalance. | **Rare VRU Recall $\uparrow +8 - 12\%$**, Macro mAP $\uparrow$ | Med / $12-16$ GPU-hrs |
| **8** | **P2** | **TensorRT FP16 Engine Compilation** | Compile YOLO11s checkpoint into TensorRT FP16 engine with dynamic shapes. | **In-Memory Latency $\downarrow 50\%$ (to $\approx 6\text{ ms}$)**, FPS $>50$ | Med / $1$ GPU-hr |
| **9** | **P2** | **YOLO11m Backbone Scaling Diagnostic** | Train YOLO11m ($20.1\text{M}$ params) at $imgsz=960$ to measure backbone capacity ceiling. | **Overall mAP50 $\uparrow +0.02$**, Latency $\uparrow 2.5\times$ | Med-High / $30$ GPU-hrs |
| **10** | **P3** | **Addition of P2 (Stride-4) High-Res Head** | Modify YOLO11 neck to output stride-4 feature maps for tiny vanishing-point objects. | **Small VRU Recall $\uparrow +5 - 8\%$**, FLOPs $\uparrow 40\%$ | High / $35$ GPU-hrs |

---

---

## 4. Best Three Initial Experiments & Scientific Rationale

### Experiment 1 (Stage 1): Stratified Failure Atlas & Prediction Collection (300 Images)
- **Why First:** We must not alter code or evaluate hypotheses on assumptions alone. Furthermore, existing artifacts only store binary matching outcomes at $\tau=0.25$, not raw candidate coordinates and confidences. A targeted validation run on $300$ stratified cases ($60$ recovered small VRUs, $60$ persistent misses, $60$ high-conf FPs, $40$ rare VRUs, $40$ boundary truncations, $40$ weather/night shifts) visually confirms false-alarm mechanisms and simultaneously collects the raw candidate predictions required for threshold calibration.
- **Cost:** $< 5$ minutes compute, zero risk.

### Experiment 2 (Stage 2.1): Dual-Threshold Crop-Source Score Calibration (Pilot Sweep)
- **Why Second:** The primary demonstrated bottleneck is Active Foveation's precision drop to $62.03\%$ driven by $+3,022$ crop false alarms. Imposing a stricter threshold ($\tau_{\text{isolated}} \ge 0.40$) on unconfirmed crop boxes tests whether precision can be recovered without model retraining.
- **Cost:** $< 1$ GPU-hour using raw predictions gathered from the targeted validation run.
- **Hypothesized Goal (Unvalidated):** Tests the engineering hypothesis of whether precision can be lifted toward the $\ge 68.0-70.0\%$ target floor while retaining small VRU recall. This number is an experimental acceptance target to be tested, not a promised outcome.

### Experiment 3 (Stage 2.2): Class-Specific Threshold Calibration ($\tau_c$)
- **Why Third:** BDD100K has extreme class imbalance ($13,265$ pedestrians vs $452$ motorcycles). Decoupling thresholds allows rare-class recall to be tested without flooding the scene with pedestrian false alarms.
- **Cost:** $< 30$ minutes local CPU grid search on saved candidate tensors.
- **Hypothesized Goal (Unvalidated):** Tests whether VRU Macro mAP@50 can improve by $+0.015$ to $+0.025$.

---

## 5. Unified First Step for Implementation

To resolve any ambiguity between diagnostic inspection and calibration sweeps, Phase 4B begins with a single, unified execution step:

1. **Check Available Artifacts (Completed):** Audit confirmed that `per_image_records.jsonl` stores matched ground-truth indices at $\tau=0.25$, but lacks continuous candidate prediction boxes across $\tau \in [0.01, 0.50]$.
2. **Execute Targeted Diagnostic & Prediction Gathering Run (The Immediate Next Step):** Run a targeted inference script on the 300-image stratified cohort that simultaneously:
   - (a) Produces the visual **Failure Atlas** (visual inspection of crop hallucinations, boundary truncations, and persistent misses).
   - (b) Serializes **raw candidate predictions** (`boxes`, `classes`, `confidences`, `sources: full_frame vs crop`) for all 300 images.
3. **Execute Pilot Calibration Sweep:** Use the saved candidate predictions from the 300-image cohort to empirically test the dual-threshold hypothesis before spending compute on a full 10,000-image evaluation.

---

## 6. What Should NOT Be Attempted Yet

1. **DO NOT treat hypothetical precision targets as guaranteed gains:** A projection of $\ge 70\%$ precision is an unvalidated hypothesis. Approval of any method requires empirical validation on genuine benchmark execution, not hypothetical projection.
2. **DO NOT attempt off-line threshold sweeps on existing Phase 3.4 files alone:** Because existing files lack raw unassigned bounding boxes and confidence floats, attempting to sweep thresholds without a prediction-gathering run will produce incomplete or flawed results.
3. **DO NOT retrain large models (YOLO11x, RT-DETR):** Retraining massive models without understanding the failure modes wastes GPU budget ($>50$ GPU-hours) and destroys real-time edge feasibility.
4. **DO NOT evaluate the 20,000-image test set:** The test split must remain strictly quarantined until final candidate selection is frozen.
5. **DO NOT attempt temporal video models on static keyframes:** The validation benchmark consists of disconnected keyframes; applying tracking algorithms to static frames is scientifically invalid.

---

## 7. Compute & Effort Estimates

| Stage | Activity | Compute Required | Calendar / Engineering Effort |
|:---|:---|:---:|:---:|
| **Stage 1** | Failure Atlas & Targeted Raw Prediction Dump ($300$ imgs) | Local CPU / GPU ($< 5\text{ mins}$) | $0.5\text{ days}$ |
| **Stage 2** | Threshold & Fusion Optimization (Pilot + Full Val) | $1 - 2$ Kaggle GPU-hours | $1.0\text{ days}$ |
| **Stage 3** | Crop-Policy & Geometry Optimization | $2 - 3$ Kaggle GPU-hours | $1.5\text{ days}$ |
| **Stage 4** | Sliced Fine-Tuning & Data Augmentation | $25 - 35$ Cloud GPU-hours | $3.0\text{ days}$ |
| **Stage 5** | Architecture Diagnostic & TensorRT Export | $15 - 20$ Cloud GPU-hours | $2.0\text{ days}$ |
| **Stage 7** | Frozen Model Selection & Test Pass | $2$ Kaggle GPU-hours | $1.0\text{ days}$ |
| **Total** | **Full Phase 4 Execution Roadmap** | **$\approx 50 - 65$ GPU-hours** | **$\approx 9$ working days** |

---

## 8. Risks & Dependencies

1. **Risk: Prediction Artifact Absence for Sweeps.** Existing Phase 3.4 artifacts lack continuous prediction coordinates.
   - *Mitigation:* Explicitly include raw prediction logging in the Stage 1 diagnostic run.
2. **Risk: Threshold Overfitting.** Sweeping post-processing thresholds on all 10,000 validation images could overfit to that specific split.
   - *Mitigation:* Split validation set into $5,000$ tuning images and $5,000$ validation images, verifying threshold stability before final frozen evaluation.
3. **Risk: Training Divergence in SFT.** Fine-tuning on crops could destabilize full-frame feature representations.
   - *Mitigation:* Use low learning rate ($lr_0 = 10^{-4}$), cosine decay, and retain $80\%$ full-frame images in training mini-batches.

---

## 9. Explicit Go / No-Go Criteria

- **Proceed from Stage 1 to Stage 2 (GO):** If failure atlas confirms that crop false positives exhibit separable confidence distributions or lack full-frame confirmation, and raw predictions are successfully captured.
- **Proceed from Stage 2 to Stage 3 (GO):** If empirical Dual-Threshold Calibration validates precision improvement to $\ge 68.0\%$ while maintaining small VRU recall $\ge 45.0\%$.
- **Proceed from Stage 3 to Stage 4 (GO):** If post-processing optimizations have reached their empirical ceiling and small VRU recall remains capped at $< 48\%$.
- **NO-GO on YOLO11m / Architecture Replacement:** If YOLO11m achieves $< +0.015\text{ mAP}$ gain over fine-tuned YOLO11s while incurring a $2.5\times$ latency penalty.

---

## 10. Exact Instructions for the Next Implementation Phase (Phase 4B)

1. **Step 1:** Implement diagnostic script `scripts/extract_diagnostic_failure_cohort.py` to extract the 300-image failure cohort and save both visual comparison frames and raw detection candidate JSONs (`boxes`, `scores`, `classes`, `sources`).
2. **Step 2:** Review the visual failure atlas to inspect false-positive patterns.
3. **Step 3:** Pilot the dual-threshold calibration sweep on the saved 300-image candidate records to measure actual empirical precision and recall changes.
4. **Step 4:** If the pilot demonstrates empirical precision gains, scale the dual-threshold evaluation to the full 10,000-image validation run on Kaggle GPU.

---

## 11. Authoritative Source List

1. **Akyon, F. C., Altinuc, S. O., & Temizel, A. (2022).** *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection.* IEEE ICIP 2022. [arXiv:2202.06934](https://arxiv.org/abs/2202.06934).  
   *Application:* Informs the mechanism of high-resolution crop slicing and sliced fine-tuning (SFT) to bridge the domain gap between full frames and crops.
2. **Solovyev, R., Wang, W., & Gabruseva, T. (2021).** *Weighted Boxes Fusion: Ensembling Methods for Object Detection.* Image and Vision Computing, 107, 104117. [arXiv:1910.13302](https://arxiv.org/abs/1910.13302).  
   *Application:* Informs the multi-scale fusion strategy to replace greedy NMS deduplication with consensus-based confidence weighting.
3. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** *On Calibration of Modern Neural Networks.* ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).  
   *Application:* Informs confidence discounting and dual-threshold score calibration for crop-originating candidate detections.
4. **Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).** *Focal Loss for Dense Object Detection.* ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002).  
   *Application:* Informs loss reweighting during fine-tuning to mitigate the 30:1 class imbalance between `person` and `motor`/`rider`.
5. **Ghiasi, G., Cui, Y., Srinivas, A., et al. (2021).** *Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation.* CVPR 2021. [arXiv:2012.07177](https://arxiv.org/abs/2012.07177).  
   *Application:* Informs targeted data augmentation to inject rare VRU instances into road training frames.
6. **Yu, F., Chen, H., Wang, X., et al. (2020).** *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning.* CVPR 2020. [arXiv:2005.04610](https://arxiv.org/abs/2005.04610).  
   *Application:* Primary dataset source establishing class definitions, label syntax, and vanishing point perspective geometry.
7. **Zhang, Y., Sun, P., Dong, Y., et al. (2022).** *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022. [arXiv:2110.06864](https://arxiv.org/abs/2110.06864).  
   *Application:* Informs the future Stage 6 formulation for tracking-assisted low-confidence detection persistence in video sequences.
8. **NVIDIA TensorRT Developer Guide (v10.x).** NVIDIA Corporation.  
   *Application:* Informs the FP16 engine compilation protocol and CUDA stream crop batching for real-time edge execution.

---

## 12. Concluding Single-Experiment Recommendation

> **If we have to start with just one experiment:**  
> **Unified Experiment 1: 300-Image Diagnostic Atlas & Raw Candidate Prediction Collection.**
>
> - **Why it is the true first choice:** It bridges the artifact gap identified in the audit. Existing files do not contain the continuous prediction coordinates and confidences needed to calibrate thresholds or inspect false alarms. This experiment runs a lightweight, targeted inference pass on $300$ stratified failure images to simultaneously produce the visual Failure Atlas AND save the raw candidate bounding-box tensors (`boxes`, `scores`, `classes`, `source: full_frame vs crop`).
> - **Exact artifacts it will consume:** [kaggle_jobs/foveation_benchmark/evaluate_phase3_foveation.py](file:///c:/Users/hiten/Downloads/cv/kaggle_jobs/foveation_benchmark/evaluate_phase3_foveation.py), checkpoint [weights/bdd100k_yolo11s.pt](file:///c:/Users/hiten/Downloads/cv/weights/bdd100k_yolo11s.pt), and validation split [bdd100k_dataset/val/](file:///c:/Users/hiten/Downloads/cv/bdd100k_dataset/val/).
> - **Exact artifacts it will produce:**
>   1. Visual Failure Atlas ($300$ color-coded side-by-side diagnostic images).
>   2. Raw candidate predictions JSON (`cohort_raw_predictions.json`) containing all candidate detections with confidence scores and source tags.
> - **Evidence that determines whether we proceed:**
>   1. Visual confirmation that crop false positives are predominantly unconfirmed background patterns rather than true VRUs.
>   2. An empirical pilot sweep on `cohort_raw_predictions.json` demonstrating that dual-threshold filtering increases precision by $\ge +5.0\%$ with $\le 2.0\%$ true-positive loss. If confirmed, we immediately scale the calibration to the full 10,000-image benchmark. If disproven, we skip post-processing grid searches and proceed directly to Stage 4 model fine-tuning.
