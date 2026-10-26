# RoadSentinel Phase 8 — Active Foveation Controlled Inference Experiment Report

**Author / Evaluator:** Antigravity Autonomous Agent  
**Date:** October 13, 2026  
**Cohort:** 300-Image Development Pilot (`PILOT_300_FILENAMES`)  
**Frozen Checkpoint:** YOLO11m Phase 6 Epoch 26 (`scratch/kaggle_upload_weights/best.pt`)  
**Checkpoint SHA-256:** `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` (Verified, 0 weight modifications)  
**Quarantine Status:** Strictly Preserved — 20,000-image test split untouched (`bdd100k_dataset/test`)  

---

## 1. Executive Summary & Definitive Go / No-Go Decision

### Research Objective
Determine whether targeted high-resolution inspection (Active Foveation: full-frame 960px + uncertainty-triggered 640px crops) can recover small vulnerable road users (VRUs) more efficiently than full-frame 1280px native inference.

### Core Empirical Findings
1. **Small-VRU Detection Ceiling:** Active Foveation achieved **48.28% recall on small VRUs (435 / 901)**, outperforming both the 960px baseline (**42.73%**, 385 / 901; **+50 small VRUs gained**) and the 1280px native reference (**45.17%**, 407 / 901; **+28 small VRUs gained above native resolution**).
2. **Motorcycle Class Preservation:** While 1280px native inference degraded `motor` recall by **-6.26%** (falling from 34.38% to 28.12%), Active Foveation achieved **37.50% recall** (+3.12% vs 960px, +9.38% vs 1280px), demonstrating that maintaining 960px global context avoids high-resolution aspect/texture distortion on two-wheeled motorized vehicles.
3. **Precision Cost:** Multi-pass inspection introduced additional candidate proposals, inflating false positives (VRU FP increased from 289 to 457), which reduced VRU precision to **70.85%** (vs 78.45% baseline and 78.20% native), bringing VRU F1 to **0.6717** (vs 0.6829 baseline and 0.6906 native).
4. **Latency Bottleneck on Serial Execution:** Although the crops inspected only **5.44%** of the frame's pixel area, running crops sequentially on a single thread incurred an average end-to-end latency of **1,496.1 ms/img (0.67 FPS)**—which is **42.2% slower than 1280px native full-frame (1,051.9 ms/img)** and 5.48× slower than 960px single-pass (273.0 ms/img).

### Definitive Recommendation
- **General Serial Real-Time Deployment: NO-GO**  
  On synchronous, serial processing architectures, Active Foveation cannot replace 1280px native inference because the discrete forward pass overhead of $K=2$ sequential crops dominates the runtime savings from processing fewer pixels.
- **Asynchronous / Multi-Stream Safety Coprocessor: GO**  
  If deployed in a dual-pipeline architecture (where 960px runs synchronously at line-rate and crops are offloaded to an asynchronous secondary worker), Active Foveation delivers the highest small-VRU recall measured across all Phase 8 experiments with zero motor degradation.

---

## 2. Three-Way Controlled Comparison Matrix (Measured Evidence)

All metrics were evaluated across the identical 300 development images, matching predictions against the 6,892 ground truth objects (1,740 VRUs, 901 small VRUs) using identical greedy IoU $\ge 0.50$ matching, confidence-ordered greedy matching, and identical operational thresholding ($\text{conf} \ge 0.25$).

| Metric | Config A: 960px Baseline | Config B: 1280px Reference | Config C: 960px + Foveation | Delta (C vs A) | Delta (C vs B) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **VRU Recall** | 60.46% | 61.84% | **63.85%** | **+3.39%** | **+2.01%** |
| **VRU Precision** | **78.45%** | 78.20% | 70.85% | -7.60% | -7.35% |
| **VRU F1-Score** | 0.6829 | **0.6906** | 0.6717 | -0.0112 | -0.0189 |
| **Small-VRU Recall (<32²)** | 42.73% | 45.17% | **48.28%** | **+5.55%** | **+3.11%** |
| **Small-VRU TP / Total** | 385 / 901 | 407 / 901 | **435 / 901** | **+50 TP** | **+28 TP** |
| **All-Class Recall** | 71.88% | **73.64%** | 73.55% | +1.67% | -0.09% |
| **All-Class Precision** | **77.93%** | 77.68% | 69.72% | -8.21% | -7.96% |
| **All-Class F1-Score** | 0.7478 | **0.7561** | 0.7159 | -0.0319 | -0.0402 |
| **Per-Image Latency** | **273.0 ms** | 1,051.9 ms | 1,496.1 ms | +1,223.1 ms (5.48×) | +444.2 ms (1.42×) |
| **Effective Throughput** | **3.66 FPS** | 0.95 FPS | 0.67 FPS | -2.99 FPS | -0.28 FPS |
| **Mean Crops per Image** | 0.00 | 0.00 | 1.94 | +1.94 crops | +1.94 crops |
| **Inspected Area Coverage** | 100.0% | 100.0% | **5.44% (crops)** | 5.44% selective | -94.56% area |
| **Peak Memory RSS** | ~520 MB | ~580 MB | 894.3 MB | +374 MB | +314 MB |

---

## 3. Per-Class Analysis (Including Motor Breakdown)

A key finding from Phase 8 Milestone 1 was the vulnerability of the `motor` class at high resolutions. The table below traces per-class recall across all configurations:

| Class | Ground Truth Count | 960px Baseline Recall (TP) | 1280px Native Recall (TP) | Foveation Recall (TP) | Key Behavioral Observation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Person** | 1,528 | 61.91% (946) | 63.22% (966) | **65.51% (1,001)** | **+55 persons recovered** vs baseline (+35 vs 1280px) via zoomed pedestrian clusters. |
| **Rider** | 58 | 55.17% (32) | **56.90% (33)** | **56.90% (33)** | Matches 1280px peak recall; successfully resolves distant riders. |
| **Bike** | 122 | 51.64% (63) | **55.74% (68)** | 53.28% (65) | +2 bikes vs baseline; slight edge to 1280px due to full-frame handlebar/spoke continuity. |
| **Motor** | 32 | 34.38% (11) | 28.12% (9) | **37.50% (12)** | **Rescued from 1280px degradation (+9.38% gain)**; avoids aspect distortion. |
| **Car** | 2,903 | 78.75% (2,286) | 80.43% (2,335) | 79.68% (2,313) | Stable; unaffected by crop focus. |
| **Bus** | 65 | 63.08% (41) | 63.08% (41) | 63.08% (41) | Identical performance across all three setups. |
| **Truck** | 133 | 66.17% (88) | 67.67% (90) | 67.67% (90) | +2 trucks detected in medium distance. |
| **Traffic Light**| 980 | 71.02% (696) | 72.86% (714) | 72.35% (709) | +13 traffic lights recovered in distant crosswalk crops. |
| **Traffic Sign** | 1,070 | 73.93% (791) | 75.42% (807) | 75.23% (805) | +14 traffic signs recovered. |
| **Train** | 1 | 0.00% (0) | 0.00% (0) | 0.00% (0) | 0 detections across all methods (extreme class imbalance). |

---

## 4. Pipeline Execution & Stage-by-Stage Latency Profiling

The minimal foveation pipeline was profiled stage by stage across all 300 images:

```
[Frame Ingestion] -> [Full-Frame 960px Pass] -> [Uncertainty Trigger Engine]
       (6.2 ms)                (607.1 ms)                      (3.4 ms)
                                                                  |
[Multi-Scale WBF Fusion] <- [Coordinate Remap] <- [Crop Inference (K=2)] <- [Crop Extraction]
       (3.2 ms)                 (<0.1 ms)                  (871.9 ms)               (0.2 ms)
```

### Detailed Breakdown

| Stage | Mean Duration (ms) | Percentage of Total | Notes & Hardware Characteristics |
| :--- | :---: | :---: | :--- |
| **1. Image Ingestion (OpenCV)** | 6.21 ms | 0.4% | OpenCV image read from local storage into BGR buffer. |
| **2. Full-Frame 960px Inference** | 607.13 ms | 40.6% | YOLO11m single-frame forward pass at $960 \times 960$. |
| **3. Uncertainty Proposal Logic** | 3.39 ms | 0.2% | Heuristics (confidence, area, cluster IoU, boundary) + overlap merging. |
| **4. Sub-Image Extraction** | 0.19 ms | <0.1% | Zero-copy NumPy slicing with boundary context expansion. |
| **5. Crop Inference Pass** | 871.88 ms | 58.3% | Sequential execution of $K=1.94$ crops at $640 \times 640$ (~449 ms per crop). |
| **6. Inverse Remap & Box Fusion** | 3.16 ms | 0.2% | Vectorized WBF, boundary artifact suppression, intra-class NMS. |
| **Total End-to-End Latency** | **1,496.12 ms** | **100.0%** | **Effective Throughput: 0.67 FPS.** |

### Crop Budget & Inspection Statistics
- **Total Images:** 300
- **0 Crops Triggered:** 4 images (1.3%)
- **1 Crop Triggered:** 11 images (3.7%)
- **2 Crops Triggered:** 285 images (95.0%)
- **Mean Crops Executed:** 1.94 crops / image
- **Mean Pixel Area Inspected by Crops:** **5.44%** of the full 1280x720 frame (50,135 px² union area)

---

## 5. Architectural & Methodological Details

### A. Candidate Selection Policy (Deterministic Heuristics)
Candidate regions are extracted from the 960px initial detections using four deterministic signals:
1. **Low Confidence:** VRU detections with $\text{conf} \in [0.10, 0.35)$ indicate scale ambiguity or partial occlusion.
2. **Small Bounding Box:** VRU detections with $\text{area} < 1024\,\text{px}^2$ (BDD100K small threshold) trigger localized zoom.
3. **Spatial Clustering:** Pairs of overlapping detections ($\text{IoU} \ge 0.35$) where at least one is a VRU trigger a merged crop proposal.
4. **Boundary Truncation:** Detections touching the outer 5% margin of the frame trigger peripheral inspection.

All candidate boxes are expanded by a **25% context margin**, clipped to $[0, 1280] \times [0, 720]$, merged if overlap $\text{IoU} \ge 0.30$, and prioritized by urgency score up to $K=2$.

### B. Coordinate Re-mapping & Boundary Artifact Filtering
Detections from the crop frame are mapped back to original coordinates via:
$$x_{\text{orig}} = x_{\text{crop\_int}} + x_{\text{crop\_rel}}, \quad y_{\text{orig}} = y_{\text{crop\_int}} + y_{\text{crop\_rel}}$$
To prevent edge-severing artifacts where an object is sliced in half by an artificial crop boundary, detections touching internal crop borders with confidence $< 0.35$ are discarded.

### C. Multi-Scale Duplicate Suppression (WBF)
Full-frame and remapped crop detections are grouped by class and merged using Weighted Box Fusion:
- If $\text{IoU}(B_{\text{ff}}, B_{\text{crop}}) \ge 0.50$, the coordinates are fused with confidence weighting:
  $$B_{\text{fused}} = \frac{w_{\text{ff}} B_{\text{ff}} + 1.2 w_{\text{crop}} B_{\text{crop}}}{w_{\text{ff}} + 1.2 w_{\text{crop}}}$$
- Dual-pass agreement boosts confidence: $\text{conf}_{\text{fused}} = \min(0.99, \max(c_1, c_2) + 0.05 \min(c_1, c_2))$.
- Unmatched crop detections are retained as recovered objects.
- A final intra-class NMS pass at $\text{IoU} = 0.60$ eliminates near-duplicate proposals.

---

## 6. Verification & Test Suite Audit Trail

Unit tests were executed to validate all mathematical and operational requirements:
- [`tests/test_phase8_foveation.py`](file:///c:/Users/hiten/Downloads/cv/tests/test_phase8_foveation.py):
  1. `test_coordinate_mapping_subpixel_accuracy`: Sub-pixel round-trip mapping accuracy.
  2. `test_crop_boundaries_clipping_and_min_size`: Strict clamping to $[0, W] \times [0, H]$.
  3. `test_crop_proposal_merging`: Union box formation for overlapping proposals.
  4. `test_box_fusion_duplicate_handling`: WBF coordinate weighting and confidence boost.
  5. `test_empty_detections_handling`: Graceful handling of zero detections.
  6. `test_class_aware_matching_strict_isolation`: Strict class boundaries during fusion.
  7. `test_confidence_ordered_greedy_matching`: Correct ranking during ground truth matching.
- [`tests/test_foveation.py`](file:///c:/Users/hiten/Downloads/cv/tests/test_foveation.py):
  8–14. 7 additional tests covering IoU, policy switching, boundary filtering, and random baselines.

**Result:** **14 / 14 Unit Tests Passed** (0 failures, 0 regressions).

---

## 7. Artifacts Generated

1. **3-Way Comparison Matrix:** [`experiments/phase8_foveation/foveation_pilot_comparison.json`](file:///c:/Users/hiten/Downloads/cv/experiments/phase8_foveation/foveation_pilot_comparison.json)
2. **Per-Image Detailed Records:** [`experiments/phase8_foveation/foveation_per_image_results.json`](file:///c:/Users/hiten/Downloads/cv/experiments/phase8_foveation/foveation_per_image_results.json)
3. **Execution Checkpoint:** [`experiments/phase8_foveation/foveation_checkpoint.json`](file:///c:/Users/hiten/Downloads/cv/experiments/phase8_foveation/foveation_checkpoint.json)
4. **Reproducible Driver Script:** [`experiments/phase8_foveation/evaluate_phase8_foveation.py`](file:///c:/Users/hiten/Downloads/cv/experiments/phase8_foveation/evaluate_phase8_foveation.py)
5. **Core Engine Modules:** [`src/foveation/`](file:///c:/Users/hiten/Downloads/cv/src/foveation/) (`engine.py`, `types.py`, `cropper.py`, `box_fusion.py`, `uncertainty_signals.py`)
