# RoadSentinel Phase 4A — Failure Atlas & Diagnostic Error Analysis Plan

**Date:** October 10, 2026  
**Auditor:** Principal Computer Vision Researcher & Evaluation Specialist  
**Status:** COMPLETE — Stratified Diagnostic Methodology  
**Target:** Rigorous taxonomy and sampling methodology to dissect detection failures across FullFrame-960, FullFrame-1280, and ActiveFoveation-960 before model modification or retraining.

---

## 1. Objective & Capabilities of Existing Artifacts

### 1.1 What Existing Artifacts Can Answer Directly
From [per_image_records.jsonl](file:///c:/Users/hiten/Downloads/cv/experiments/phase3_foveation/runs/kaggle_run_v2/per_image_records.jsonl) and [bdd100k_dataset/val/labels/](file:///c:/Users/hiten/Downloads/cv/bdd100k_dataset/val/labels/):
1. **Per-Object True Positives & False Negatives:** For all $15,373$ VRU ground-truth objects, we know exactly which objects were detected ($\text{TP}$) and which were missed ($\text{FN}$) by every configuration.
2. **Size-Group Miss Breakdown:** We can partition misses into Small ($< 1,024\text{ px}^2$), Medium ($1,024 - 9,216\text{ px}^2$), and Large ($\ge 9,216\text{ px}^2$).
3. **Cross-Configuration Differential Outcomes:**
   - **Crop-Recovered VRUs:** Ground-truth VRUs missed by FullFrame-960 but detected by ActiveFoveation-960 ($N = 1,173$ objects).
   - **Persistent Multi-Resolution Misses:** VRUs missed by FullFrame-640, FullFrame-960, FullFrame-1280, and ActiveFoveation-960 ($N = 5,142$ objects).
   - **Resolution-Sensitives:** VRUs missed by 960 but detected by 1280 ($N = 612$ objects).
   - **Foveation-Degraded Detections:** Objects detected by FullFrame-960 but lost during foveation fusion due to crop boundary clipping or duplicate suppression ($N = 282$ objects).

### 1.2 What Requires Targeted Diagnostic Dump
1. **Granular False Positive Bounding Boxes:** As established in the repository audit, `per_image_records.jsonl` stores *matched ground-truth indices*, but did not serialize the pixel bounding-box coordinates of unassigned false positives.
2. **Requirement for Phase 4 Stage 1:** A lightweight diagnostic script must extract the false positive bounding boxes and crop proposals for a stratified sample of $300$ validation images to visually inspect false-alarm mechanisms.

---

## 2. Comprehensive Error Taxonomy

Every error in RoadSentinel is classified into one of six distinct failure modes:

```
                    ┌───────────────────────────────┐
                    │  RoadSentinel Failure Modes   │
                    └───────────────┬───────────────┘
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
┌───────────────────┐                               ┌───────────────────┐
│  False Negatives  │                               │  False Positives  │
│  (Missed VRUs)    │                               │  (False Alarms)   │
└─────────┬─────────┘                               └─────────┬─────────┘
          │                                                   │
  ├─ FN-1: Tiny Vanishing-Point Scale (< 16 px height)        ├─ FP-1: Crop Hallucination (Textures/Poles)
  ├─ FN-2: Occlusion / Crowding (Overlap > 50%)               ├─ FP-2: Semantic Confusion (Rider vs Person)
  ├─ FN-3: Adverse Lighting / Glare (Night/Rain blur)         ├─ FP-3: Scale Artifact (Reflections/Shadows)
  ├─ FN-4: Crop Boundary Truncation                           ├─ FP-4: Boundary Splitting / Duplicate
  └─ FN-5: Score Suppression (Confidence < 0.25)              └─ FP-5: Full-Frame Misdetection
```

### 2.1 False Negative Categories
1. **FN-1: Sub-Resolution Vanishing Point:**
   - *Definition:* Distant pedestrians or cyclists where box height $h < 16\text{ px}$ (area $< 256\text{ px}^2$). At $960\times 960$, they span $\le 2$ feature pixels in stride-8 ($P3$).
   - *Diagnostic Question:* Does a $640\times 640$ crop provide sufficient magnification to recover the object, or is the original optical resolution blurred?
2. **FN-2: Severe Occlusion / Dense Crowds:**
   - *Definition:* VRUs occluded $>50\%$ by vehicles, guardrails, or other pedestrians.
   - *Diagnostic Question:* Does NMS suppress the occluded pedestrian because of IoU overlap with a foreground vehicle or companion pedestrian?
3. **FN-3: Domain Shift / Adverse Lighting:**
   - *Definition:* Night scenes, oncoming headlight glare, or rain droplets on the camera lens.
   - *Diagnostic Question:* Are features washed out by illumination extremes, causing low model activation?
4. **FN-4: Crop Boundary Truncation:**
   - *Definition:* A VRU located on the periphery of an uncertainty crop that gets severed by the crop edge ($cx_1, cy_1, cx_2, cy_2$).
   - *Diagnostic Question:* Was the proposal margin too narrow, causing the detector to discard the partial torso?
5. **FN-5: Marginal Score Suppression:**
   - *Definition:* Genuine VRU detected with confidence $\tau \in [0.10, 0.24]$, suppressed by the rigid $\tau = 0.25$ operational threshold.

### 2.2 False Positive Categories
1. **FP-1: High-Frequency Crop Hallucination:**
   - *Definition:* Road signs, fire hydrants, tree trunks, or roadside foliage that resemble human limbs when magnified in an isolated crop without context.
2. **FP-2: Semantic Class Confusion:**
   - *Definition:* A cyclist detected as both `person` and `bike`, or a scooter rider detected as `motor` and `rider`.
3. **FP-3: Environmental Reflection / Shadow:**
   - *Definition:* Wet pavement reflections or elongated streetlamp shadows detected as upright VRUs.
4. **FP-4: Duplicate Detection / Boundary Splitting:**
   - *Definition:* A pedestrian straddling the full-frame and crop coordinate systems where IoU is $<0.50$, resulting in double-counting.

---

## 3. Stratified Diagnostic Sampling Plan

To avoid wasteful manual review of all $10,000$ validation images, we specify a **stratified, reproducible diagnostic cohort of exactly 300 validation images** ($3.0\%$ of the split).

### 3.1 Stratification Matrix ($N = 300$ Images)

| Cohort ID | Focus Category | Image Count | Selection Criteria | Objective |
|:---:|:---|:---:|:---|:---|
| **C1** | **Crop-Recovered Small VRUs** | **60 images** | Images where Active Foveation detected a Small VRU missed by FullFrame-960. | Analyze the visual mechanism of successful foveation recovery. |
| **C2** | **Persistent Small VRU Misses** | **60 images** | Images with Small VRUs missed by all configurations (640, 960, 1280, Foveation). | Determine the optical resolution floor and label quality limits. |
| **C3** | **High-Confidence Crop False Positives** | **60 images** | Images where Active Foveation generated $\ge 2$ FPs with confidence $\ge 0.40$. | Identify background patterns driving crop hallucinations. |
| **C4** | **Rare VRU Semantic Confusion** | **40 images** | Images containing ground-truth `rider`, `bike`, or `motor`. | Analyze `rider` vs `person` and `bike` vs `motor` misclassification. |
| **C5** | **Crop Boundary Truncation** | **40 images** | Images where full-frame detected a VRU but crop foveation lost it. | Measure boundary padding failures and fusion suppression errors. |
| **C6** | **Environmental Domain Shift** | **40 images** | Night ($20$) and Rain/Fog ($20$) scenes containing VRUs. | Diagnose domain degradation in adverse weather and lighting. |
| **Total** | **Diagnostic Audit Cohort** | **300 images** | Deterministic selection using sorted image manifests. | Comprehensive error atlas. |

### 3.2 Reproducible Sample Extraction Protocol
1. **Script Path:** `scripts/extract_diagnostic_failure_cohort.py` (to be executed in Stage 1).
2. **Deterministic Seed:** Seed `42`.
3. **Artifact Outputs:**
   - Diagnostic metadata JSON: `experiments/phase4_diagnostics/cohort_metadata.json`
   - Annotated visual side-by-side images ($1280 \times 720$) with color-coded bounding boxes:
     - **Green:** Ground Truth
     - **Blue:** FullFrame-960 Prediction
     - **Magenta:** Crop Proposal Bounding Box
     - **Yellow:** Active Foveation Final Prediction
     - **Red:** False Positive Detection
     - **Cyan Dashed:** False Negative (Missed Ground Truth)

---

## 4. Key Questions the Error Atlas Must Settle

Before approving any model retraining or fine-tuning in Stage 4, the visual inspection of this 300-image cohort must answer:

1. **What fraction of persistent small-VRU misses are visually recognizable to a human?**
   - If $>70\%$ are heavily blurred or sub-pixel ($<8\text{ px}$ height), they represent label noise or the physical optical limit of the 720p sensor, meaning retraining will overfit to noise.
2. **What fraction of crop false positives are isolated vs. full-frame confirmed?**
   - If $>85\%$ of crop false positives are completely absent in the full-frame pass, this provides definitive proof that the Rank 1 Dual-Threshold policy will successfully eliminate them.
3. **What is the spatial distribution of false-positive triggers?**
   - Confirming whether $>20\%$ of false crops occur in the upper third ($y < 0.35H$) or bottom sixth ($y > 0.85H$) will validate the Rank 4 Horizon-Prioritized crop policy.
4. **Is `rider` failure caused by localization or misclassification?**
   - Determine whether riders are missed entirely or classified as `person`. If classified as `person`, it is a semantic head calibration issue solvable by loss re-weighting, not an optical failure.
