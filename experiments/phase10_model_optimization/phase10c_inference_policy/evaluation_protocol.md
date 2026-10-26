# RoadSentinel Phase 10C — Evaluation Protocol: Adaptive Thresholding vs Active Foveation

**Document ID:** `RS-PHASE10C-PROTOCOL-001`  
**Date:** 2026-10-14  
**Author:** Antigravity Autonomous System  
**Status:** **ACTIVE / FROZEN PROTOCOL**  
**Experiment Root:** `experiments/phase10_model_optimization/phase10c_inference_policy/`

---

## 1. Objective & Scope

The objective of Phase 10C is to empirically resolve the trade-off between **Adaptive Confidence Thresholding** and **Active Foveation** when deployed on the **50-epoch YOLO11m Champion Model** (`session4_best.pt`), specifically targeting small Vulnerable Road User (VRU) detection (<1024 px² / sub-32x32 px in 1280x720).

### Key Questions Addressed:
1. Does Adaptive Thresholding (`conf=0.20` for VRUs) recover small VRUs with acceptable precision without adding any compute latency?
2. Does Active Foveation (960px Full-Frame + 640px uncertainty-guided crops) achieve superior small-VRU recall that cannot be matched by thresholding alone?
3. What is the false-positive profile and confidence distribution across both approaches?
4. How do these policies generalize beyond the development pilot cohort to an unseen, non-test validation cohort?

---

## 2. Model Protection & SHA-256 Provenance

Both models are strictly frozen and protected from modification. No training or fine-tuning will be performed.

| Model Role | File Path | File Size | SHA-256 Checksum | Audit Status |
| :--- | :--- | :--- | :--- | :--- |
| **Operational Champion** | `experiments/phase10_model_optimization/phase10b_extended_training/session4_artifacts/session4_best.pt` | 40,539,500 B | `4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd` | **VERIFIED 50-EPOCH CHAMPION** |
| **Frozen Phase 6 Baseline** | `scratch/kaggle_upload_weights/best.pt` | 40,539,943 B | `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` | **MATCHED & FROZEN** |

---

## 3. Evaluated Policy Configurations

All configurations use identical image resolutions, NMS settings, and matching criteria:

| Configuration ID | Policy Name | Operational VRU Conf ($\tau_{\text{vru}}$) | Non-VRU Conf ($\tau_{\text{other}}$) | Multi-Pass Crop Engine | Forward Passes / Image |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Config A** | Champion Baseline | **0.25** | 0.25 | Disabled (Full-Frame only) | Exactly 1 |
| **Config B** | Adaptive Thresholding | **0.20** | 0.25 | Disabled (Full-Frame only) | Exactly 1 (Zero Latency Overhead) |
| **Config C** | Active Foveation (Standard) | **0.25** | 0.25 | Enabled ($K \le 2$ crops, 640px) | $1 + K$ passes (1.91 avg) |
| **Config D** | Adaptive Threshold + Foveation | **0.20** | 0.25 | Enabled ($K \le 2$ crops, 640px) | $1 + K$ passes (1.91 avg) |

### Controlled Parameters Across All Configurations:
- **Base Full-Frame Resolution:** 960 × 960 pixels.
- **Crop Resolution (Configs C & D):** 640 × 640 pixels (max 2 crops per image).
- **Foveation Uncertainty Trigger Range:** $[0.10, 0.35]$.
- **Small-Box Uncertainty Threshold:** Area $\le 1024\text{ px}^2$ ($32^2$).
- **Cluster IoU Threshold:** 0.35.
- **Context Margin Ratio:** 0.25 (25% boundary expansion).
- **Spatial Fusion IoU Threshold:** 0.50 (Greedy NMS/Fusion with boundary suppression margin 8px).
- **Evaluation IoU Matching Threshold:** $\text{IoU} \ge 0.50$ (Class-aware greedy matching).
- **VRU Class Definition:** `person`, `rider`, `bike`, `motor`.

---

## 4. Evaluation Cohorts & Validation Discipline

To avoid presenting pilot over-fitting as generalization evidence, Phase 10C enforces a dual-cohort protocol:

### Cohort 1: Development Pilot Cohort (`PILOT_300_FILENAMES`)
- **Size:** 300 images from `bdd100k_dataset/val/images`.
- **Ground Truth:** 6,892 total objects, 1,740 VRUs, **901 small VRUs (<1024 px²)**.
- **Purpose:** Direct comparison against historical Phase 8 and Phase 10B benchmarks.

### Cohort 2: Generalization Validation Cohort (`HOLDOUT_300_FILENAMES`)
- **Size:** 300 distinct, non-pilot images from `bdd100k_dataset/val/images`.
- **Ground Truth:** 5,416 total objects, 367 VRUs, **151 small VRUs (<1024 px²)**.
- **Purpose:** Validates that the threshold and foveation gains hold on completely unseen validation scenes without hyperparameter tuning.
- **Quarantine Guarantee:** The 20,000-image official test split (`bdd100k_dataset/test`) has **0 accesses** and remains strictly quarantined.

---

## 5. Metrics & Attribution Specifications

For each configuration across both cohorts, the evaluation records:
1. **Detection Performance:**
   - Small-VRU Recall, True Positives (TP), False Negatives (FN).
   - Overall VRU Precision, Recall, F1-Score, TP, FP, FN.
   - Per-Class Metrics for `person`, `rider`, `bike`, and `motor`.
2. **False Positive Stratification:**
   - FP counts binned by confidence interval: $[0.15, 0.20)$, $[0.20, 0.25)$, $[0.25, 0.50)$, and $[0.50, 1.00]$.
3. **Foveation Detection Attribution:**
   - Every true positive and false positive attributed to its origin: `full_frame` (detected by base pass), `crop_only` (recovered solely by crop pass), or `fused` (refined by spatial fusion).
4. **End-to-End Latency & Resource Profiling:**
   - Median ($p50$), Mean, and $95^{\text{th}}$ percentile ($p95$) latency in milliseconds.
   - Discrete stage profiling: Preprocessing (image read, resize, normalize), Inference (full-frame + crop forward passes), Postprocessing & Fusion.
   - Mean number of crops executed per image.
