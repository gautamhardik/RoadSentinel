# RoadSentinel Phase 8 — Bounded 300-Image Native-Resolution Evaluation Report

**Date:** 2026-10-13  
**Cohort:** Exact 300-image development pilot cohort from Milestone 1  
**Checkpoint:** Frozen YOLO11m Phase 6 Epoch 26 (`42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e`)  
**Matching Rules:** Class-aware, confidence-ordered greedy matching at IoU 0.50  
**Quarantine:** Zero access to `bdd100k_dataset/test` (strictly preserved)  
**Primary Artifacts:**
- [`experiments/phase8_pilot_300_native_res_comparison.json`](file:///c:/Users/hiten/Downloads/cv/experiments/phase8_pilot_300_native_res_comparison.json)
- [`experiments/phase8_pilot_300_native_res_checkpoint.json`](file:///c:/Users/hiten/Downloads/cv/experiments/phase8_pilot_300_native_res_checkpoint.json)

---

## 1. Kernel Status & Queue Reconciliation

- **Earlier 10,000-Image Native-Res Kernel (`roadsentinel-phase-8-native-res-eval`):** `CANCEL_ACKNOWLEDGED` (Stopped).
- **Pilot Kernel (`roadsentinel-pilot-300-native-eval`):** Both Version #1 and Version #2 are `CANCEL_ACKNOWLEDGED` (Stopped).
- **GPU Queue State:** Free. Zero concurrent GPU jobs running.

---

## 2. Reconciling Throughput & Latency

### Measured Benchmarks:
1. **Cold Start vs Warmed-Up Inference:**
   - Model loading + PyTorch instantiation: **0.10s**.
   - Cold-start inference on Image 0: ~4.17s (includes initial CUDA/OpenMP context creation and graph tracing).
   - Warmed-up steady-state per-image inference: **0.45s – 0.48s** (effective steady-state ~2.2 FPS on host CPU).
2. **Matching & Checkpointing Overhead:**
   - Vectorized greedy matching across all classes: **1.47 milliseconds** per image.
   - Atomic disk checkpointing: **< 0.01s** per image.
3. **Full 300-Image Cohort Execution:**
   - Total evaluated images: **300 / 300** (100% complete).
   - Total wall-clock time: **315.57 seconds** (5.25 minutes).
   - Measured average throughput: **0.95 FPS** (1,051.9 ms/img effective end-to-end, including disk I/O, GT decoding, inference, and matching).

---

## 3. Empirical Results: 1280px Native vs 960px Baseline (Identical 300 Images)

All operational thresholds remained strictly identical (`conf=0.25`, `nms_iou=0.65`, `matching_iou=0.50`):

### Aggregate Metrics Comparison

| Metric | 960px Baseline (Milestone 1) | 1280px Native Resolution | Empirical Delta ($\Delta$) | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **VRU Recall** | **60.46%** (1,052 / 1,740) | **61.84%** (1,076 / 1,740) | **+1.38%** | **+24 VRU True Positives gained** |
| **VRU Precision** | **78.45%** | **78.20%** | **-0.25%** | Precision remains rock-solid (negligible FP drift) |
| **VRU F1-Score** | **0.6829** | **0.6906** | **+0.0077** | Net positive F1 trade-off |
| **Small-VRU Recall** | **42.73%** (385 / 901) | **45.17%** (407 / 901) | **+2.44%** | **+22 Small VRU detections recovered** |
| **All-Class Recall** | 71.88% (4,954 / 6,892) | 73.64% (5,075 / 6,892) | **+1.76%** | +121 objects recovered overall |
| **All-Class Precision** | 77.93% | 77.68% | -0.25% | Consistent precision across categories |
| **All-Class F1-Score** | 0.7478 | 0.7561 | **+0.0083** | Broad macro improvement |

---

## 4. Detailed Breakdown: By Size & Per-Class

### VRU Detection by Size Cohort
- **Small VRUs ($< 32^2 = 1,024\,\text{px}^2$):** Recall improved from **42.73%** (385 TP) to **45.17%** (407 TP) — **+22 critical targets recovered**.
- **Medium VRUs ($32^2 \le \text{area} < 96^2$):** Recall improved from **78.15%** (583 TP) to **78.42%** (585 TP) — +2 TP.
- **Large VRUs ($\ge 96^2$):** Recall remained identical at **90.32%** (84 TP / 93).

### Per-Class VRU Recall

| Class | 960px Recall | 1280px Recall | Delta Recall | 960px TP | 1280px TP | TP Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Person** | 61.91% | **63.22%** | **+1.31%** | 946 | 966 | **+20** |
| **Rider** | 55.17% | **56.90%** | **+1.73%** | 32 | 33 | **+1** |
| **Bike** | 51.64% | **55.74%** | **+4.10%** | 63 | 68 | **+5** |
| **Motor** | 34.38% | **28.12%** | -6.26% | 11 | 9 | -2 |

*Note: Bike recall showed the largest relative gain (+4.10%), while Motor lost 2 detections out of 32 total instances.*

---

## 5. Artifact Verification & Evidence Integrity

- **Artifacts Present & Validated:**
  1. `experiments/phase8_pilot_300_native_res_comparison.json` (complete structured report with raw counts and deltas).
  2. `experiments/phase8_pilot_300_native_res_checkpoint.json` (atomic image-by-image record containing all 300 evaluations).
- **Test-Split Integrity:** All 20,000 images in `bdd100k_dataset/test` remained untouched.

---

## 6. Clear Go / No-Go Decision Recommendation

### Decision: **CONDITIONAL NO-GO for Full 10,000-Image Native-Resolution Rerun; GO for Active Foveation / Retraining**

#### Rationale:
1. **Marginal Small-VRU Benefit vs Compute Cost:**
   - While 1280px native inference proves that additional pixel density improves small-VRU recall (+2.44% gain, +22 small VRUs found without harming precision), a +2.44% improvement is relatively modest given the compute penalty.
   - At 1280px, memory requirements and tensor operations increase quadratically, causing batch-16 timeouts on cloud GPUs and dropping throughput significantly compared to 960px (22.96 FPS).
2. **Resource Allocation:**
   - Running the full 10,000-image validation split at 1280px would consume ~2.5–3 hours of GPU compute simply to confirm a ~2% recall gain on frozen weights.
3. **Strategic Next Step:**
   - Rather than spending GPU quota on brute-force 10k native-resolution inference, the evidence strongly supports directing compute toward **retraining with higher resolution** or **Active Foveation / Multi-scale inference** where the model's receptive fields are explicitly optimized for high-resolution small objects.
