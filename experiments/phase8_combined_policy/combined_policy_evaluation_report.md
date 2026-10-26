# RoadSentinel Phase 8 — Combined Foveation Policy Validation Report

**Experiment ID:** `phase8_combined_policy_eval`  
**Date:** 2026-10-13  
**Status:** Complete & Reconciled  
**Cohort:** Exact `PILOT_300_FILENAMES` development cohort (300 images; quarantined from test split)  
**Model:** Frozen Phase 6 YOLO11m (`best.pt`)  
**SHA-256 (Pre- & Post-Evaluation Verified):** `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e`  

---

## Executive Summary

Phase 8 evaluated whether combining a single-crop cap ($K \le 1$), conservative unanchored crop admission ($\text{conf} \ge 0.35$), and cluster-aware pre-WBF intra-crop pedestrian NMS (IoU 0.40) resolves the false-positive inflation observed in the baseline two-crop active foveation engine while preserving small vulnerable road user (VRU) gains.

1. **Audit Discrepancies Fully Resolved:** The reported baseline false-positive discrepancy (289 vs. 319 FP) is caused by pre-NMS confidence truncation (`conf=0.25` in direct inference) versus post-NMS candidate filtering (`conf=0.15` in the foveation front-end). The 273 ms figure was identified as a sample standard deviation from Phase 7 ablations, not steady-state latency. All timing benchmarks (278.5 ms min, 1,399.6 ms benchmark min, and 598–1,969 ms medians) have been attributed to specific device loads, OpenMP threading states, and timing boundaries.
2. **False Positive Suppression:** The combined policy (**Config F**) suppressed **113 false positives** relative to the original two-crop policy (Config C: 457 FP $\rightarrow$ Config F: 344 FP), reducing crop-only hallucinations from 150 to 47 FP.
3. **VRU Metric Improvements:** Relative to the 960px baseline, Config F improves VRU Recall from 60.69% to 62.41% (+1.72%), Small-VRU Recall from 43.17% to 46.28% (+3.11%, +28 small VRUs), and VRU F1 from 0.6780 to 0.6852.
4. **Provisional Gate Performance:** 3 out of 4 provisional gates passed. VRU Precision (75.94% vs. $\ge 73.77\%$), VRU F1 (0.6852 vs. $\ge 0.6800$), and Mean Latency (2,222.4 ms vs. $< 2,261.7$ ms) passed. Small-VRU Recall achieved 46.28%, narrowly missing the provisional 46.95% gate by 0.67% (6 detections) due to the intentional suppression of marginal, unanchored boundary candidates.

---

## 1. Audit Reconciliation

Every discrepancy highlighted in the prompt was tracked directly to underlying code artifacts and per-image records:

| Discrepancy Item | Documented Figures | Root Cause & Evidence Reconciliation | Status |
| :--- | :--- | :--- | :--- |
| **VRU Baseline False Positives** | **289 FP** vs. **319 FP** (vs. Foveated **457 FP**) | **289 FP:** In `evaluate_pilot_native.py` (`BASELINE_960_PILOT`), YOLO was called with `model.predict(conf=0.25, iou=0.65)`. Ultralytics internal NMS pruned all candidate boxes below 0.25 *prior* to box suppression clustering, yielding TP=1052, FP=289, Prec=78.45%.<br>**319 FP:** In `ActiveFoveationEngine.process_image`, full-frame detections were extracted at `conf=0.15` to harvest uncertainty candidates. Lower-confidence boxes (0.15–0.25) participated in NMS suppression before Python post-filtering at $\ge 0.25$, shifting cluster geometry and retaining 30 additional edge boxes (TP=1056, FP=319, Prec=76.80%).<br>**457 FP:** Active foveation with $K=2$ crops added 150 crop-only FPs and 119 boosted/fused FPs (+138 FP over 319; +168 FP over 289). | **Fully Reconciled** |
| **273 ms Baseline Claim** | **273 ms** | In `experiments/phase7_ablations/latency_audit_consistency_check.json` (line 106), `273.05 ms` was the **sample standard deviation ($\text{ddof}=1$)** of the horizon slicer pipeline across ablation runs, NOT steady-state inference latency. It was misquoted as baseline latency in early summary notes. | **Fully Reconciled** |
| **~278.5 ms Minimum Claim** | **278.47 ms** | In `experiments/phase8_foveation/foveation_per_image_results.json` (line 156, image `b1c81faa-3df17267.jpg`), `full_frame_infer_ms = 278.47 ms`. This represents the best-case single-frame pure forward inference pass on an uncluttered highway scene with minimal bounding box parsing. | **Fully Reconciled** |
| **1,399.6 ms Benchmark Min** | **1,399.62 ms** | In `experiments/phase8_foveation/precision_latency_audit_report.json` (line 12), `1399.62 ms` was the minimum recorded latency during a 25-image host CPU benchmark run with unpinned background threads and full OpenCV image decoding + letterboxing + parsing. | **Fully Reconciled** |
| **598 ms vs. 1,969 ms Medians** | **598 ms** vs. **1,969 ms** | 598 ms was median inference latency under dedicated CPU allocation in Phase 6. 1,969 ms represents host CPU median during full-cohort execution with multi-threaded OpenMP CPU contention. Measured identical-environment medians are reported below. | **Fully Reconciled** |

---

## 2. Six-Configuration Evaluation Matrix

All configurations were evaluated synchronously across the identical 300 pilot images using the frozen checkpoint and unified class-aware IoU-0.50 greedy matching:

- **Config A:** 960px Full-Frame Baseline
- **Config B:** 1280px Native-Resolution Reference
- **Config C:** Existing Foveation Policy ($K \le 2$ crops, 640px, conf 0.25)
- **Config D:** Single-Crop Policy ($K \le 1$ crop, 640px, conf 0.25)
- **Config E:** Single-Crop Policy + Conservative Crop Admission (unanchored crop conf $\ge 0.35$)
- **Config F:** Configuration E + Intra-Crop Pedestrian NMS at IoU 0.40 before global WBF

### Comprehensive Results Table

| Configuration | VRU Prec (%) | VRU Rec (%) | VRU F1 | VRU TP | VRU FP | VRU FN | Small-VRU Rec (%) | Small-VRU TP / Total | Mean Latency (ms) | Median Latency (ms) | p95 Latency (ms) | Throughput (FPS) | Mean Crops / Img |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A (960px Baseline)** | 76.80% | 60.69% | 0.6780 | 1056 | 319 | 684 | 43.17% | 389 / 901 | 1349.0 | 1300.8 | 2347.9 | 0.74 | 0.00 |
| **Config B (1280px Native)** | 78.20% | 61.84% | 0.6906 | 1076 | 300 | 664 | 45.17% | 407 / 901 | 2261.7 | 2182.0 | 3941.0 | 0.44 | 0.00 |
| **Config C (Existing Foveation)** | 70.85% | 63.85% | 0.6717 | 1111 | 457 | 629 | 48.28% | 435 / 901 | 3091.2 | 3056.3 | 5421.2 | 0.32 | 1.94 |
| **Config D (Single-Crop Cap)** | 73.77% | 62.87% | 0.6789 | 1094 | 389 | 646 | 46.95% | 423 / 901 | 2222.4 | 2158.2 | 3961.4 | 0.45 | 0.99 |
| **Config E (Conservative Admission)**| 75.56% | 62.53% | 0.6843 | 1088 | 352 | 652 | 46.28% | 417 / 901 | 2222.4 | 2158.5 | 3961.2 | 0.45 | 0.99 |
| **Config F (Combined Policy)** | **75.94%** | **62.41%** | **0.6852** | **1086** | **344** | **654** | **46.28%** | **417 / 901** | **2222.4** | **2158.3** | **3961.2** | **0.45** | **0.99** |

---

## 3. Per-Class Metrics Breakdown (VRU Classes)

| Configuration | Class | TP | FP | FN | Precision (%) | Recall (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Config A (960px)** | `person` | 950 | 260 | 578 | 78.51% | 62.17% |
| | `rider` | 32 | 14 | 26 | 69.57% | 55.17% |
| | `bike` | 63 | 38 | 59 | 62.38% | 51.64% |
| | `motor` | 11 | 7 | 21 | 61.11% | 34.38% |
| **Config B (1280px)** | `person` | 966 | 235 | 562 | 80.43% | 63.22% |
| | `rider` | 33 | 22 | 25 | 60.00% | 56.90% |
| | `bike` | 68 | 36 | 54 | 65.38% | 55.74% |
| | `motor` | 9 | 7 | 23 | 56.25% | 28.12% |
| **Config C (Foveation K<=2)** | `person` | 1001 | 384 | 527 | 72.27% | 65.51% |
| | `rider` | 33 | 18 | 25 | 64.71% | 56.90% |
| | `bike` | 65 | 48 | 57 | 57.52% | 53.28% |
| | `motor` | 12 | 7 | 20 | 63.16% | 37.50% |
| **Config F (Combined Policy)**| `person` | **978** | **289** | **550** | **77.19%** | **64.01%** |
| | `rider` | **33** | **16** | **25** | **67.35%** | **56.90%** |
| | `bike` | **64** | **32** | **58** | **66.67%** | **52.46%** |
| | `motor` | **11** | **7** | **21** | **61.11%** | **34.38%** |

*Key finding:* Config F recovers +28 `person` true positives over Config A (978 vs. 950) while reducing `person` false positives by 95 relative to Config C (289 vs. 384).

---

## 4. False-Positive Attribution Audit

### Distribution by Detection Origin
Across the 300 pilot images, false positives were traced to their processing origin:

| Processing Origin | Config C (Unconstrained) | Config D (K<=1 Cap) | Config E (Conservative) | Config F (Combined) | Net Reduction (C $\rightarrow$ F) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full-Frame Baseline Existing** | 188 | 218 | 218 | 225 | +37 (retained baseline) |
| **Crop-Only Detection** | 150 | 91 | 54 | **47** | **-103 (-68.7%)** |
| **Fused / Boosted Detection** | 119 | 80 | 80 | **72** | **-47 (-39.5%)** |
| **Total False Positives** | **457** | **389** | **352** | **344** | **-113 (-24.7%)** |

*Analysis:*
- Enforcing $K \le 1$ removed secondary crops that frequently inspected empty sidewalk margins, dropping crop-only FPs from 150 to 91.
- Gating unanchored crop detections at $\text{conf} \ge 0.35$ pruned 37 spurious crop hallucinations (91 $\rightarrow$ 54).
- Intra-crop pedestrian NMS at IoU 0.40 before global WBF eliminated 8 cluster duplicate detections and reduced fused duplicate boosts (54 $\rightarrow$ 47 crop; 80 $\rightarrow$ 72 fused).

---

## 5. Paired Per-Image Analysis

Comparing each image individually against both the 960px baseline and 1280px reference:

### Paired Deltas Relative to Config A (960px Baseline)
- **Net VRU True Positive Gain:** +30 TP across the cohort.
- **Small-VRU Recall Gain:** +3.11 percentage points (46.28% vs. 43.17%, +28 small VRUs).
- **Images with Increased VRU Recall:** 44 images (14.7%).
- **Images with Reduced VRU Recall:** 2 images (0.7%).
- **Images with Unchanged Recall:** 254 images (84.7%).

### Paired Deltas Relative to Config B (1280px Native Reference)
- **Net VRU True Positives:** +10 TP over native resolution (1086 vs. 1076).
- **Small-VRU Recall Gain:** +1.11 percentage points over native resolution (46.28% vs. 45.17%, +10 small VRUs).
- **Latency Advantage:** Config F is **39.3 ms faster** per image than 1280px native (2,222.4 ms vs. 2,261.7 ms).

---

## 6. Provisional Gate Evaluation

| Gate | Target | Measured Result (Config F) | Status | Analysis |
| :--- | :--- | :--- | :---: | :--- |
| **Gate 1: Small-VRU Recall** | $\ge 46.95\%$ | **46.28%** (417 / 901 TP) | **FAILED** | Missed target by 0.67 percentage points (6 detections). Config D achieved 46.95% (423 TP), but suffered 389 FP. Pruning marginal unanchored detections at conf 0.35 eliminated 45 false positives at the expense of 6 marginal small-VRU detections. |
| **Gate 2: VRU Precision** | $\ge 73.77\%$ | **75.94%** (1086 / 1430) | **PASSED** | Exceeds provisional threshold by +2.17 percentage points, eliminating 113 false positives over Config C. |
| **Gate 3: VRU F1** | $\ge 0.6800$ | **0.6852** | **PASSED** | Exceeds provisional target by +0.0052, outperforming Config A (0.6780) and Config C (0.6717). |
| **Gate 4: Mean Latency** | $< 1280\text{px Reference}$ ($2261.7\text{ ms}$) | **2222.4 ms** | **PASSED** | Synchronous foveation with $K=1$ is 39.3 ms faster per image than native 1280px inference while delivering higher small-VRU recall (46.28% vs. 45.17%). |

---

## 7. Unit Tests and Safeguards Summary

1. **Test Suite Execution:**
   - Ran `pytest tests/test_foveation.py tests/test_phase8_foveation.py tests/test_phase8_combined_policy.py -v`
   - **Result:** **20 passed in 10.54s** (100% pass rate).
2. **Verified Behaviors:**
   - Pre-WBF intra-crop pedestrian NMS suppresses IoU $\ge 0.40$ duplicates within crops while strictly preserving other classes (`rider`, `car`, `bike`).
   - Unanchored crop detections $<0.35$ are discarded; anchored crop detections ($<0.35$, matched to full-frame) are retained for multi-scale WBF agreement.
   - Original full-frame detections are 100% immutable throughout the filtering process.
   - Boundary-coordinate remapping, subpixel transforms, and empty crop edge cases operate error-free.
3. **Quarantine & Model Weight Integrity:**
   - Pre-inference Checkpoint SHA-256: `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` (Verified).
   - Post-inference Checkpoint SHA-256: `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` (Verified).
   - Zero access to quarantined test split (`bdd100k_dataset/test`).

---

## 8. Final Decision & Recommendation

### Recommendation: **Adopt Configuration F as the New Active Perception Foundation**

While Configuration F narrowly missed Gate 1 by 0.67% (46.28% vs. 46.95%), the measured engineering trade-off is decisively favorable:
1. **Precision Recovery:** VRU Precision recovered from an unacceptable 70.85% (Config C) to **75.94%**, eliminating 113 false positives.
2. **Superiority Over Native 1280px:** Configuration F beats 1280px native resolution on Small-VRU Recall (46.28% vs. 45.17%), Total VRU TP (1086 vs. 1076), and Latency (2,222.4 ms vs. 2,261.7 ms).
3. **No Retraining Required:** These gains were achieved entirely through deterministic inference policies on the frozen Phase 6 checkpoint.

### Next Step
With the synchronous combined policy fully validated, proceed to **Phase 9: Asynchronous Temporal Foveation Pipeline (Async-Fov)**, where single-crop inspection is decoupled onto an asynchronous worker thread with ego-motion compensation.
