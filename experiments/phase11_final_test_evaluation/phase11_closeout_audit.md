# RoadSentinel Phase 11 — Closeout and Evidence Audit

**Document Status:** FINAL CLOSEOUT RECORD  
**Audit Date:** October 15, 2026  
**Auditor:** Antigravity Autonomous Coding Agent  
**Audit Target:** `experiments/phase11_final_test_evaluation/`  
**Governing Decision:** Phase 10F Pre-Declared Locked Evaluation Protocol  

---

## 1. Executive Summary & Audit Scope

This closeout audit establishes the permanent record for **RoadSentinel Phase 11**, the one-time blind evaluation of the frozen candidate policy (**Config E**) against the **Controlled Baseline Policy** on the official BDD100K test split (`bdd100k_dataset/test/`).

### Invariant Verification
- **Test Set Quarantine & Re-Inference Prohibited:** No inference was rerun on the test set during this audit. Test images and labels were not accessed.
- **Zero Post-Hoc Tuning:** Checkpoint weights, confidence thresholds, NMS parameters, and matching IoUs were not adjusted.
- **Checkpoint Immutability:** Checkpoint SHA-256 digests remain identical to their Phase 6 and Phase 10B states.
- **Final Binding Verdict:** **FAIL — 2 of 4 Pre-Declared Acceptance Gates Passed.**

---

## 2. Checkpoint Integrity Verification (Task 3)

The cryptographic SHA-256 hashes of the frozen models were audited in place without modification:

| Checkpoint Identifier | Physical File Path | Pre-Declared / Frozen Hash | Audited SHA-256 Hash | Verification Status |
|---|---|---|---|---|
| **Champion Model** | `experiments/phase10_model_optimization/phase10b_extended_training/session4_artifacts/session4_best.pt` | `4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd` | `4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd` | **MATCH (VERIFIED)** |
| **Phase 6 Reference Baseline** | `scratch/kaggle_upload_weights/best.pt` | `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` | `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` | **MATCH (VERIFIED)** |

Both checkpoints are completely intact. Neither file was modified during Phase 11 execution or during this audit.

---

## 3. Artifact Consistency Matrix (Tasks 1 & 2)

All six Phase 11 evaluation artifacts were cross-checked for internal numerical consistency across image counts, ground-truth annotations, confusion matrix counts, point metrics, confidence intervals, and bootstrap probabilities:

1. `test_cohort_manifest.csv`
2. `final_test_policy_comparison.csv`
3. `final_test_per_class_metrics.csv`
4. `final_test_statistical_results.json`
5. `final_test_statistical_report.md`
6. `final_test_decision_record.md`

### Consistency Verification Table

| Metric / Parameter | Value in JSON | Value in CSVs | Value in Statistical Report | Value in Decision Record | Cross-Check Status |
|---|---|---|---|---|---|
| **Test Image Count** | 2,000 | 2,000 (manifest) | 2,000 | 2,000 | **EXACT MATCH** |
| **Total GT Objects** | 37,173 | 37,173 (manifest sum) | 37,173 | 37,173 | **EXACT MATCH** |
| **VRU GT Objects** | 2,813 | 2,813 (manifest sum) | 2,813 | 2,813 | **EXACT MATCH** |
| **Small VRU GT (<32²)** | 1,237 | 1,237 (manifest sum) | 1,237 | 1,237 | **EXACT MATCH** |
| **Medium VRU GT [32²–96²)** | 1,362 | 1,362 (manifest sum) | 1,362 | 1,362 | **EXACT MATCH** |
| **Large VRU GT (≥96²)** | 214 | 214 (manifest sum) | 214 | 214 | **EXACT MATCH** |
| **Baseline VRU (TP, FP, FN)** | 1841, 521, 972 | 1841, 521, 972 | 1841, 521, 972 | 1841, 521, 972 | **EXACT MATCH (VRU-only)** |
| **Config E VRU (TP, FP, FN)** | 1875, 557, 938 | 1875, 557, 938 | 1875, 557, 938 | 1875, 557, 938 | **EXACT MATCH (VRU-only)** |
| **Baseline Small Recall** | 45.51% (563/1237) | 45.51% | 45.51% (563/1237) | 45.51% | **EXACT MATCH** |
| **Config E Small Recall** | 47.29% (585/1237) | 47.29% | 47.29% (585/1237) | 47.29% | **EXACT MATCH** |
| **Small Recall Delta ($\Delta$)** | +1.78 pp | +1.78 pp (derived) | +1.78 pp | +1.78 pp | **EXACT MATCH** |
| **$\Delta$ Small Recall 95% CI** | [+0.99, +2.62] pp | [+0.99, +2.62] pp | [+0.99, +2.62] pp | [+0.99, +2.62] pp | **EXACT MATCH** |
| **Small Recall $P(\Delta > 0)$** | 100.0% | 100.0% | 100.0% | 100.0% | **EXACT MATCH** |
| **Baseline VRU Precision** | 77.94% | 77.94% | 77.94% | 77.94% | **EXACT MATCH (VRU-only)** |
| **Config E VRU Precision** | 77.10% | 77.10% | 77.10% | 77.10% | **EXACT MATCH (VRU-only)** |
| **Config E Precision 95% CI** | [75.13, 79.03]% | [75.13, 79.03]% | [75.13, 79.03]% | [75.13, 79.03]% | **EXACT MATCH (VRU-only)** |
| **Baseline VRU F1** | 0.7115 | 0.7115 | 0.7115 | 0.7115 | **EXACT MATCH (VRU-only)** |
| **Config E VRU F1** | 0.7150 | 0.7150 | 0.7150 | 0.7150 | **EXACT MATCH (VRU-only)** |
| **VRU F1 Delta ($\Delta$)** | +0.0035 | +0.0035 (derived) | +0.0035 | +0.0035 | **EXACT MATCH (VRU-only)** |
| **$\Delta$ VRU F1 95% CI** | [−0.0010, +0.0080] | [−0.0010, +0.0080] | [−0.0010, +0.0080] | [−0.0010, +0.0080] | **EXACT MATCH (VRU-only)** |
| **VRU F1 $P(\Delta > 0)$** | 93.5% | 93.5% | 93.5% | 93.5% | **EXACT MATCH (VRU-only)** |
| **Baseline VRU FP / Image** | 0.261 (0.2605) | 0.261 | 0.261 | 0.261 | **EXACT MATCH (VRU-only)** |
| **Config E VRU FP / Image** | 0.279 (0.2785) | 0.279 | 0.279 | 0.279 | **EXACT MATCH (VRU-only)** |
| **$\Delta$ VRU FP / Image** | +0.018 | +0.018 | +0.018 | +0.018 | **EXACT MATCH (VRU-only)** |
| **Forward Passes / Frame** | 1.00 | 1.00 | 1.00 | 1.00 | **EXACT MATCH** |

### Clarification on Metric Scope: VRU-Specific vs. All-Class Detection Counts
To prevent any ambiguity in evaluation reporting, detection metrics must explicitly distinguish between **VRU-supercategory counts** and **full-detector all-class counts**:
- **VRU-Specific Metrics (`person`, `rider`, `bike`, `motor`):**
  * The reported 521 false positives (Baseline) and 557 false positives (Config E), and the corresponding precisions (77.94% and 77.10%), are **strictly VRU-level counts**. They exclude vehicle and signage false detections.
  * Mean FP/image values of 0.261 and 0.279 represent **VRU false alarms per image**.
- **All-Class Full-Detector Metrics (All 10 Categories):**
  * Baseline All-Class: $\text{TP} = 27,594$, $\text{FP} = 7,740$, $\text{FN} = 9,579 \implies \text{Precision} = \mathbf{78.09\%}$, $\text{Recall} = \mathbf{74.23\%}$, $\text{F1} = \mathbf{0.7611}$. Mean All-Class FP/image = $3.870$.
  * Config E All-Class: $\text{TP} = 27,628$, $\text{FP} = 7,776$, $\text{FN} = 9,545 \implies \text{Precision} = \mathbf{78.04\%}$, $\text{Recall} = \mathbf{74.32\%}$, $\text{F1} = \mathbf{0.7613}$. Mean All-Class FP/image = $3.888$.
  * All-Class Deltas: $\Delta \text{TP} = +34$, $\Delta \text{FP} = +36$, $\Delta \text{FN} = -34$.
  * **Key Finding:** The net $+34$ true positives gained and $+36$ false positives incurred by Config E across the entire detector correspond **1-to-1 with the VRU supercategory changes**, demonstrating that Config E had zero impact on vehicle and signage detection.

### Per-Class Summation Audit
Cross-checking `final_test_per_class_metrics.csv` against aggregate metrics:
- **Baseline VRU Sums:**
  * $\text{TP} = 1634 \text{ (person)} + 59 \text{ (rider)} + 106 \text{ (bike)} + 42 \text{ (motor)} = \mathbf{1,841}$ (Matches aggregate)
  * $\text{FP} = 431 \text{ (person)} + 22 \text{ (rider)} + 51 \text{ (bike)} + 17 \text{ (motor)} = \mathbf{521}$ (Matches aggregate)
  * $\text{FN} = 808 \text{ (person)} + 55 \text{ (rider)} + 68 \text{ (bike)} + 41 \text{ (motor)} = \mathbf{972}$ (Matches aggregate)
- **Config E VRU Sums:**
  * $\text{TP} = 1667 \text{ (person)} + 61 \text{ (rider)} + 103 \text{ (bike)} + 44 \text{ (motor)} = \mathbf{1,875}$ (Matches aggregate)
  * $\text{FP} = 463 \text{ (person)} + 22 \text{ (rider)} + 53 \text{ (bike)} + 19 \text{ (motor)} = \mathbf{557}$ (Matches aggregate)
  * $\text{FN} = 775 \text{ (person)} + 53 \text{ (rider)} + 71 \text{ (bike)} + 39 \text{ (motor)} = \mathbf{938}$ (Matches aggregate)
- **Non-VRU Invariance:** All 6 vehicle/signage categories (`car`, `bus`, `truck`, `traffic light`, `traffic sign`, `train`) have identical TP, FP, and FN across Baseline and Config E, confirming zero unintended side effects outside VRU classes.

Zero internal discrepancies were identified across all six artifacts.

---

## 4. Execution Trace & Access Distinctions Audit (Tasks 4 & 5)

Inspection of `run_final_test_evaluation.py` and execution logs (`task-7666.log` and `task-7684.log`) establishes the exact sequence of data operations executed during Phase 11. To ensure precision, we explicitly distinguish between initial file access, model inference, and downstream cached evaluation:

1. **Initial Physical File Read (Single-Pass Access):**
   - In `task-7666`, the 2,000 test cohort image files (`cabc30fc-e7726578.jpg` to `cf8fb4a6-36c46511.jpg`) and corresponding label files in `bdd100k_dataset/test/` were opened and read in read-only mode to load pixel arrays for batch generation and parse ground-truth bounding box coordinates.
   - This represents the **single physical read pass** of the underlying BDD100K test image and annotation files.
2. **Single-Pass Inference Execution:**
   - Model inference was executed across the 2,000 images in batches of 4 on host CPU, completing in 651.1s (325.6 ms/img).
   - Exactly **one forward pass per image** was performed through the champion detector at $960 \times 960$ letterbox.
   - All raw detector outputs (projected bounding boxes, raw confidences, class IDs) were immediately serialized and cached to `test_2000_raw_preds.json` (8.18 MB).
3. **Subsequent Access to Cached Predictions (Zero Re-Inference):**
   - In `task-7666`, following inference, matching, and bootstrap computation, the script encountered an unhandled `KeyError: 'vru_size_stats'` at line 542 (the evaluation helper returned `vru_by_size`).
   - The key was corrected to `vru_by_size` in `run_final_test_evaluation.py`.
   - In `task-7684`, the script restarted and **loaded the cached predictions directly from `test_2000_raw_preds.json` and the manifest**, completing all JSON/CSV generation and visual overlay rendering in ~15 seconds.
   - **Critical Audit Distinction:** Neither `task-7684` nor this closeout audit re-opened, re-read, or re-inferred on the test image or label files in `bdd100k_dataset/test/`. All subsequent evaluation was performed strictly on the cached prediction artifact.
4. **Bootstrap Implementation Fidelity (Task 5):**
   - **Sampling Unit:** Paired image-level resampling (`rng.randint(0, N, size=(B, N))`). The identical sampled image indices were applied to both Baseline and Config E arrays (`np.take(arr_b, boot_indices)` and `np.take(arr_e, boot_indices)`).
   - **Replicate Count:** Exactly $B = 2,000$.
   - **RNG Seed:** Fixed `seed = 42`.
   - **Metric Computation:** Within each bootstrap replicate, counts are aggregated across sampled images first, followed by ratio calculations ($\text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}}$, $\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}}$, $\text{F1} = \frac{2 P R}{P + R}$), and paired difference $\Delta = \text{Metric}_E - \text{Metric}_{\text{Base}}$.
   - **Intervals:** 2.5th and 97.5th percentiles of empirical replicate distribution.

The bootstrap implementation is mathematically sound, correctly paired at the image level, and adheres strictly to the pre-declared protocol.

---

## 5. Acceptance Gate Audit & Binding Verdict (Tasks 6 & 7)

| Gate | Pre-Declared Criterion | Official Test Split Result | Gate Status |
|---|---|---|---|
| **Gate 1: Small-VRU Recall Gain** | $\Delta \ge +2.50\text{ pp}$ and $P(\Delta > 0) \ge 99.0\%$ | $\Delta = \mathbf{+1.78\text{ pp}}$ ($45.51\% \to 47.29\%$), $P(\Delta > 0) = \mathbf{100.0\%}$ | **FAIL** |
| **Gate 2: Precision Safety Floor** | Config E Precision $\ge 72.0\%$, CI lower bound $\ge 70.0\%$ | Precision = $\mathbf{77.10\%}$, 95% CI Lower = $\mathbf{75.13\%}$ | **PASS** |
| **Gate 3: Net F1 Superiority** | $\Delta \text{F1} > 0.000$ and $P(\Delta > 0) \ge 95.0\%$ | $\Delta \text{F1} = \mathbf{+0.0035}$ ($0.7115 \to 0.7150$), $P(\Delta > 0) = \mathbf{93.5\%}$ | **FAIL** |
| **Gate 4: Deterministic Latency** | Exactly 1 forward pass per frame | $\mathbf{1.00}$ passes per frame ($960 \times 960$ letterbox) | **PASS** |

### **Binding Verdict: FAIL (2 of 4 Gates Passed)**

The failure is recorded as binding:
- **Gate 1 failed on effect magnitude:** Although 100% of bootstrap resamples showed a positive gain ($P=100\%$, zero overlap with 0), the effect size of $+1.78\text{ pp}$ was below the $+2.50\text{ pp}$ contract hurdle.
- **Gate 3 failed on statistical threshold:** The F1 improvement point estimate was positive ($+0.0035$), but the 95% confidence interval slightly crossed zero ($[−0.0010, +0.0080]$), yielding $93.5\%$ probability of positive gain versus the required $95.0\%$.

---

## 6. Distinguishing Verified Facts vs. Unresolved Limitations (Task 8)

### Verified Facts (Supported by Direct Evidence)
1. **Positive Small-VRU Gain:** Config E detected +22 additional small VRUs (563 $\to$ 585) and +34 total VRUs (1841 $\to$ 1875) across the 2,000 test images. The small-VRU recall gain (+1.78 pp) has a 95% CI of $[+0.99, +2.62]\text{ pp}$ with $P(\Delta>0)=100.0\%$.
2. **Preserved Precision:** Config E achieved $77.10\%$ VRU precision, comfortably above the $72.0\%$ point floor and with its 95% CI lower bound ($75.13\%$) well above the $70.0\%$ limit.
3. **Controlled False Alarms:** Config E added 36 VRU false alarms across 2,000 images (+0.018 VRU FP/image, or 1 extra false alarm every 55 frames), yielding an exchange ratio of 1.06 false alarms per additional VRU true positive.
4. **Deterministic Single-Pass Latency:** Exactly 1 forward pass per frame at 960px.
5. **Bicycle Over-Suppression:** Bicycles experienced a slight regression (−3 TP, 106 $\to$ 103, recall $-1.72\text{ pp}$) due to tighter NMS ($0.55$) in dense geometry.

### Unresolved Scientific Limitations
1. **Absolute Small-VRU Detection Deficit:** Over half of all small ground-truth VRUs (**52.71%**, or 652 out of 1,237) remain undetected under Config E.
2. **Post-Processing Saturation:** Threshold lowering ($\tau=0.22$) only recovers candidates where the model already produced raw confidence in the $[0.22, 0.25)$ window. It cannot recover false negatives where confidence is near zero or where features were omitted entirely by the backbone.
3. **Hardware Latency Benchmarking:** While algorithmic cycle count is locked at 1.00 pass/frame, target automotive hardware (e.g. NVIDIA Orin / Jetson) benchmarking remains outstanding.

---

## 7. Cohort Consumption & Future Development Boundary (Task 9)

### A. Consumption of Test Cohort & Quarantine Status
- **Test Cohort (Images 1–2,000):** The deterministic 2,000-image cohort (`cabc30fc-e7726578.jpg` to `cf8fb4a6-36c46511.jpg`) is formally marked **CONSUMED**. It must **never** be reused for model training, validation, hyperparameter sweeps, or candidate selection in any future phase.
- **Remaining Test Split (Images 2,001–20,000):** Based on the repository execution audit, script inspections, and manifest range definition (`all_test_imgs[:2000]`), the remaining 18,000 images in `bdd100k_dataset/test/` show zero access records and are treated as quarantined and unvisited.
  * *Audit Caveat:* As noted in the audit scope, this quarantine status is established from the execution logs and script inspection rather than being an independently provable mathematical guarantee deduced solely from the fact that the closeout audit script avoided the test directory.

### B. Separation of Future Proposals (Non-Executed)
Future gains beyond 50% small-VRU recall cannot be achieved by post-processing tuning. Future development must occur strictly on training and validation splits.

#### Proposed Development Pathways for Future Phases (Do Not Execute in Phase 11):
1. **Proposal 1 — P2 Micro-Detection Head Architecture (Candidate Hypothesis):**
   - Explore adding a high-resolution P2 feature pyramid layer (stride 4, $240 \times 240$ feature map) to the YOLO11 backbone to retain spatial resolution for sub-32px bounding boxes.
   - *Audit Caveat:* The P2 head is a candidate architectural hypothesis to be tested and validated—not an assumption that it will automatically improve results or resolve the small-VRU problem without latency/memory trade-offs.
   - Train and validate exclusively on BDD100K `train` and `val` splits.
2. **Proposal 2 — Small-Object Loss Reweighting & Targeted Augmentation:**
   - Investigate focal loss scaling inversely proportional to bounding box area during training.
   - Test targeted small-VRU copy-paste augmentation during training on validation benchmarks.
3. **Proposal 3 — Independent Test Validation:**
   - Reserve an independent second slice from the quarantined 18,000 test images (e.g., Test Cohort 2, images 2,001–4,000) for a one-time final test only after a new architecture demonstrates verified superiority on the validation set under an approved, locked protocol.

---

## 8. Final Audit Sign-Off

- **Audit Completion:** PASSED. All artifacts and metrics verified consistent.
- **Evidence Integrity:** 100% traceable. Zero data loss, zero post-hoc tuning, zero test split re-evaluations.
- **Evaluation Status:** Phase 11 is formally **CLOSED**.
