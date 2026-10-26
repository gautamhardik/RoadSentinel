# RoadSentinel Phase 10E — Policy Confirmation and Controlled Ablation Protocol

## 1. Executive Objective

Phase 10E establishes a controlled confirmation and ablation study for the inference policy of the 50-epoch YOLO11m champion model (`session4_best.pt`).

Following the Phase 10D review, the criteria are explicitly locked **before** evaluating any new data:
- **Primary Operating Objective:** **Balanced / Higher Precision Floor ($\ge 72.0\%$)**. Maximize small-VRU recall among policies that strictly clear this threshold.
- **Decision Rule:** If neither adaptive policy ($\tau=0.20$ or $\tau=0.22$) clears the $72.0\%$ precision floor on fresh validation data, the baseline policy ($\tau=0.25$) is retained as the conservative operational policy.
- **No Training / Checkpoint Lock:** Checkpoints remain immutable (`session4_best.pt` SHA-256 `4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd`; `best.pt` SHA-256 `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e`).
- **Test Split Quarantine:** The official BDD100K test split (`bdd100k_dataset/test/`) remains 100% quarantined (0 accesses).

---

## 2. Controlled Staged Workflow

```
[Stage A: Threshold Confirmation] (Fresh 1,000-image Cohort 4)
  - Compare Flat 0.25 vs Adaptive 0.22 vs Adaptive 0.20 (IoU fixed at 0.65)
  - Clustered Bootstrap 95% Confidence Intervals
  - Select candidate meeting >= 72.0% precision gate
        |
        v
[Stage B: Duplicate Suppression NMS Ablation] (Cohort 4)
  - Test VRU NMS IoU in {0.65, 0.60, 0.55} on candidate policy
  - Measure both FPs eliminated and TPs lost (collateral damage)
  - Decide whether tighter NMS improves precision without harming recall
        |
        v
[Stage D: Independent Policy Confirmation] (Fresh Untouched 1,000-image Cohort 5)
  - Validate finalized policy against Baseline Flat 0.25 on an untouched slice
  - Final operational decision report
```

---

## 3. Cohort Partitioning Scheme

All cohorts are drawn deterministically from sorted filenames in `bdd100k_dataset/val/images/` ($N = 10,000$ images total):

1. **Previously Evaluated (Quarantined from Phase 10E):**
   - Cohort 1: Pilot (300 images, indices non-contiguous)
   - Cohort 2: Generalization Holdout (300 images, indices 0–299 of non-pilot pool)
   - Cohort 3: Phase 10D Validation (1,000 images, indices 300–1299 of non-pilot pool)
2. **Phase 10E Fresh Cohorts:**
   - **Cohort 4 (Stage A & B Confirmation):** Indices 1300–2299 of non-pilot pool ($N = 1,000$ images).
     * Ground Truth: 18,794 objects, 1,615 VRUs, 651 Small VRUs.
   - **Cohort 5 (Stage D Untouched Confirmation):** Indices 2300–3299 of non-pilot pool ($N = 1,000$ images).
     * Ground Truth: 18,548 objects, 1,300 VRUs, 628 Small VRUs.

All 5 cohorts are verified to be strictly pairwise disjoint. Full file paths and ground truth counts are documented in `cohort_manifest.csv`.

---

## 4. Evaluation Metrics & Statistical Rigor

1. **Clustered Bootstrap Confidence Intervals:**
   - Because objects are clustered within images (violating IID assumptions), uncertainty estimates are computed using **image-level bootstrap resampling** ($B = 1,000$ iterations).
   - Report 95% percentile confidence intervals for Small-VRU Recall and VRU Precision: $[\text{CI}_{2.5\%}, \text{CI}_{97.5\%}]$.
2. **Detection Matching Protocol:**
   - BDD100K $1280 \times 720$ ground truth coordinates.
   - Per-class greedy matching with $\text{IoU} \ge 0.50$.
   - COCO size categories: Small ($< 32^2 = 1024\text{ px}^2$), Medium ($[32^2, 96^2)$), Large ($\ge 96^2$).
3. **VRU Classes:** `person`, `rider`, `bike`, `motor`.
4. **Non-VRU Classes:** `car`, `bus`, `truck`, `traffic light`, `traffic sign`, `train` (kept fixed at $\tau = 0.25$).
