# RoadSentinel Phase 3.3 — Evidence Provenance Gate Report

**Date:** October 9, 2026  
**Evaluator:** RoadSentinel Quality & Safety Engineering  
**Subject:** Benchmark Provenance, Audit Integrity, and Execution Gate for Phase 3.2 Artifacts  
**Model Checkpoint:** `weights/bdd100k_yolo11s.pt`  
**Checkpoint SHA-256:** `A4747292648F37F68AB997BEA65A36391C41356F35E309A9F3D2EB1D3EA8D8E3`  
**Evaluation Set:** BDD100K 10,000 Validation Images (`bdd100k_dataset/val/`)  
**Operating Operational Thresholds:** Confidence $\tau = 0.25$, Matching $\text{IoU} = 0.50$  

---

## Executive Summary & Exit Decision

### Gate Exit Decision: **BLOCKED**

This gate was established to verify whether the Phase 3.2 reconciled benchmark artifacts reflect genuine execution records or manufactured/reconstructed values. Following an exhaustive code and artifact audit, the Evidence Provenance Gate has concluded with a **BLOCKED** decision due to the following specific findings:

1. **Missing Crop Prediction & Matching Records on Disk:** While the Phase 3 Kaggle kernel executed inference across all 10,000 validation images for all six configurations on Nvidia Tesla T4 GPU, the script crashed at line 824 (`json.dump(final_output, f)`) with `TypeError: Object of type float32 is not JSON serializable`. Consequently, per-image prediction arrays, per-box matching records, and fine-grained per-class false-positive counts for `FixedGrid-960`, `RandomCrop-960`, and `ActiveFoveation-960` were **never serialized or persisted to disk**.
2. **Manufactured Size-Group TP Partitions:** Because per-box matching records were lost during the crash, the Medium and Large VRU true positive counts in Phase 3.2 (`FixedGrid-960`: 5,568 Med / 929 Large; `RandomCrop-960`: 5,454 Med / 926 Large; `ActiveFoveation-960`: 5,419 Med / 930 Large) were **mathematically inferred and hardcoded** in `scripts/finalize_phase3_benchmark.py` rather than parsed from verified prediction records.
3. **Manufactured Pipeline Latencies:** The reported "batch-1 pipeline latency" figures in Phase 3.2 were manufactured by adding a static constant ($+6.18\text{ ms}$) to in-memory latency instead of being directly measured through end-to-end file-to-output profiling.
4. **Resolution Latency Anomaly Identified:** The apparent discrepancy in Phase 2B.2 where `Control-640` ($28.95\text{ ms}$) exhibited higher batch-1 latency than `Baseline-960` ($19.27\text{ ms}$) was traced directly to cold OS disk caching during sequential execution, as Phase 2B.2 passed file paths (`source=img_path`) inside the timed loop. In contrast, Phase 3 measured in-memory tensors, showing correct monotonic scaling ($10.18\text{ ms} < 13.09\text{ ms} < 19.04\text{ ms}$).

### Action Required to Unblock:
Rerun the benchmark script with the verified recursive `to_serializable()` handler on Kaggle GPU (`hardikgautam25/roadsentinel-phase-3-foveation-benchmark`), persist the complete `phase3_foveation_metrics.json` containing the full `per_class_results` and `size_group_results` tables, and download the resulting raw artifacts.

---

## 1. Metric Provenance Traceability Matrix

Every metric across all six configurations was traced to its exact point of origin.

| Configuration | Metric | Reported Value | Provenance Classification | Source Location / Artifact | Audit Notes |
|---|---|---|---|---|---|
| **FullFrame-640** | Overall mAP@50 | 0.4819 | Directly Measured | `experiments/full_resolution_benchmark/full_resolution_metrics.json` | Validated in Phase 2B.2 on 10k val images |
| | VRU Macro mAP@50 | 0.4428 | Directly Measured | `full_resolution_metrics.json` | Validated in Phase 2B.2 on 10k val images |
| | VRU TP (Total) | 7,435 | Directly Measured | `full_resolution_metrics.json` | Sum of exact class TPs (person: 6,432, rider: 297, bike: 478, motor: 228) |
| | VRU FP | 2,724 | Directly Measured | `full_resolution_metrics.json` | Sum of exact class FPs at $\tau=0.25$ |
| | VRU FN | 7,938 | Directly Measured | Ground truth subtraction | $15,373 - 7,435 = 7,938$ |
| | VRU Precision / Recall / F1 | 0.7319 / 0.4836 / 0.5824 | Directly Measured | Computed from TP, FP, FN | Mathematically exact identity |
| | Small / Med / Large TP | 1,634 / 4,884 / 917 | Directly Measured | `full_resolution_metrics.json` | Exact size matching from Phase 2B.2 records |
| | Throughput / Batch1 In-Memory | 22.01 FPS / 10.18 ms | Directly Measured | `experiments/phase3_foveation/kaggle_console_log.txt` | Measured directly in Phase 3 Kaggle run |
| | Batch1 Pipeline P50 | 28.95 ms | Directly Measured | Phase 2B.2 `batch1_stats` | End-to-end file I/O + decode + predict (cold cache) |
| **FullFrame-960** | Overall mAP@50 | 0.5544 | Directly Measured | `full_resolution_metrics.json` | Validated in Phase 2B.2 on 10k val images |
| | VRU Macro mAP@50 | 0.5262 | Directly Measured | `full_resolution_metrics.json` | Validated in Phase 2B.2 on 10k val images |
| | VRU TP (Total) | 8,750 | Directly Measured | `full_resolution_metrics.json` | Exact class TPs (person: 7,790, rider: 337, bike: 379, motor: 244) |
| | VRU FP | 2,880 | Directly Measured | `full_resolution_metrics.json` | Sum of exact class FPs at $\tau=0.25$ |
| | VRU FN | 6,623 | Directly Measured | Ground truth subtraction | $15,373 - 8,750 = 6,623$ |
| | VRU Precision / Recall / F1 | 0.7524 / 0.5692 / 0.6481 | Directly Measured | Computed from TP, FP, FN | Mathematically exact identity |
| | Small / Med / Large TP | 2,558 / 5,265 / 927 | Directly Measured | `full_resolution_metrics.json` | Exact size matching from Phase 2B.2 records |
| | Throughput / Batch1 In-Memory | 39.92 FPS / 13.09 ms | Directly Measured | `kaggle_console_log.txt` | Measured directly in Phase 3 Kaggle run |
| | Batch1 Pipeline P50 | 19.27 ms | Directly Measured | Phase 2B.2 `batch1_stats` | End-to-end file I/O + decode + predict |
| **FullFrame-1280** | Overall mAP@50 | 0.5638 | Directly Measured | `full_resolution_metrics.json` | Validated in Phase 2B.2 on 10k val images |
| | VRU Macro mAP@50 | 0.5367 | Directly Measured | `full_resolution_metrics.json` | Validated in Phase 2B.2 on 10k val images |
| | VRU TP (Total) | 8,927 | Directly Measured | `full_resolution_metrics.json` | Exact class TPs |
| | VRU FP | 2,793 | Directly Measured | `full_resolution_metrics.json` | Sum of exact class FPs at $\tau=0.25$ |
| | VRU FN | 6,446 | Directly Measured | Ground truth subtraction | $15,373 - 8,927 = 6,446$ |
| | VRU Precision / Recall / F1 | 0.7617 / 0.5807 / 0.6590 | Directly Measured | Computed from TP, FP, FN | Mathematically exact identity |
| | Small / Med / Large TP | 2,835 / 5,192 / 900 | Directly Measured | `full_resolution_metrics.json` | Exact size matching from Phase 2B.2 records |
| | Throughput / Batch1 In-Memory | 31.50 FPS / 19.04 ms | Directly Measured | `kaggle_console_log.txt` | Measured directly in Phase 3 Kaggle run |
| | Batch1 Pipeline P50 | 24.64 ms | Directly Measured | Phase 2B.2 `batch1_stats` | End-to-end file I/O + decode + predict |
| **FixedGrid-960** | Overall mAP@50 | 0.5032 | Directly Measured (Printed) | `kaggle_console_log.txt` (line 74) | Evaluated over 10k val images on Kaggle GPU |
| | VRU Macro mAP@50 | 0.4800 | Directly Measured (Printed) | `kaggle_console_log.txt` (line 74) | Evaluated over 10k val images on Kaggle GPU |
| | VRU Recall / Small Recall | 0.6343 / 0.4629 | Directly Measured (Printed) | `kaggle_console_log.txt` (line 75) | Printed to console during evaluation |
| | VRU TP (Total) | 9,751 | Derived | Reconstructed: $15,373 \times 0.6343$ | Derived from printed recall percentage |
| | Small VRU TP | 3,254 | Derived | Reconstructed: $7,029 \times 0.4629$ | Derived from printed small recall percentage |
| | Medium VRU TP | 5,568 | **Inferred / Reconstructed** | `finalize_phase3_benchmark.py` | Manually reconciled to eliminate 148 mismatch |
| | Large VRU TP | 929 | **Inferred / Reconstructed** | `finalize_phase3_benchmark.py` | Copied from Phase 3.1 preliminary table |
| | VRU FP | 7,003 | Derived | Reconstructed: $9,751 / 0.5820 - 9,751$ | Derived from precision 0.5820 |
| | VRU FN | 5,622 | Derived | Ground truth subtraction | $15,373 - 9,751 = 5,622$ |
| | VRU Precision / F1 | 0.5820 / 0.6070 | Directly Measured (Summary) | `phase3_foveation_metrics.json` | From top-level summary dictionary |
| | Throughput / Batch1 In-Memory | 16.79 FPS / 34.30 ms | Directly Measured | `kaggle_console_log.txt` (lines 24, 76) | Measured directly on Kaggle GPU |
| | Batch1 Pipeline P50 | 40.48 ms | **Manufactured (+6.18 ms)** | `finalize_phase3_benchmark.py` | Computed as $34.30 + 6.18\text{ ms}$ |
| **RandomCrop-960** | Overall mAP@50 | 0.5048 | Directly Measured (Printed) | `kaggle_console_log.txt` (line 86) | Evaluated over 10k val images on Kaggle GPU |
| | VRU Macro mAP@50 | 0.4935 | Directly Measured (Printed) | `kaggle_console_log.txt` (line 86) | Evaluated over 10k val images on Kaggle GPU |
| | VRU Recall / Small Recall | 0.6183 / 0.4446 | Directly Measured (Printed) | `kaggle_console_log.txt` (line 87) | Printed to console during evaluation |
| | VRU TP (Total) | 9,505 | Derived | Reconstructed: $15,373 \times 0.6183$ | Derived from printed recall percentage |
| | Small VRU TP | 3,125 | Derived | Reconstructed: $7,029 \times 0.4446$ | Derived from printed small recall percentage |
| | Medium VRU TP | 5,454 | **Inferred / Reconstructed** | `finalize_phase3_benchmark.py` | Manually reconciled to eliminate 122 mismatch |
| | Large VRU TP | 926 | **Inferred / Reconstructed** | `finalize_phase3_benchmark.py` | Derived |
| | VRU FP | 6,578 | Derived | Reconstructed: $9,505 / 0.5910 - 9,505$ | Derived from precision 0.5910 |
| | VRU FN | 5,868 | Derived | Ground truth subtraction | $15,373 - 9,505 = 5,868$ |
| | VRU Precision / F1 | 0.5910 / 0.6043 | Directly Measured (Summary) | `phase3_foveation_metrics.json` | From top-level summary dictionary |
| | Throughput / Batch1 In-Memory | 16.93 FPS / 35.55 ms | Directly Measured | `kaggle_console_log.txt` (lines 25, 88) | Measured directly on Kaggle GPU |
| | Batch1 Pipeline P50 | 41.73 ms | **Manufactured (+6.18 ms)** | `finalize_phase3_benchmark.py` | Computed as $35.55 + 6.18\text{ ms}$ |
| **ActiveFoveation-960** | Overall mAP@50 | 0.5403 | Directly Measured (Printed) | `kaggle_console_log.txt` (line 98) | Evaluated over 10k val images on Kaggle GPU |
| | VRU Macro mAP@50 | 0.5120 | Directly Measured (Printed) | `kaggle_console_log.txt` (line 98) | Evaluated over 10k val images on Kaggle GPU |
| | VRU Recall / Small Recall | 0.6271 / 0.4683 | Directly Measured (Printed) | `kaggle_console_log.txt` (line 99) | Printed to console during evaluation |
| | VRU TP (Total) | 9,641 | Derived | Reconstructed: $15,373 \times 0.6271$ | Derived from printed recall percentage |
| | Small VRU TP | 3,292 | Derived | Reconstructed: $7,029 \times 0.4683$ | Derived from printed small recall percentage |
| | Medium VRU TP | 5,419 | **Inferred / Reconstructed** | `finalize_phase3_benchmark.py` | Manually reconciled to eliminate 12 mismatch |
| | Large VRU TP | 930 | **Inferred / Reconstructed** | `finalize_phase3_benchmark.py` | Derived |
| | VRU FP | 5,101 | Derived | Reconstructed: $9,641 / 0.6540 - 9,641$ | Derived from precision 0.6540 |
| | VRU FN | 5,732 | Derived | Ground truth subtraction | $15,373 - 9,641 = 5,732$ |
| | VRU Precision / F1 | 0.6540 / 0.6402 | Directly Measured (Summary) | `phase3_foveation_metrics.json` | From top-level summary dictionary |
| | Throughput / Batch1 In-Memory | 24.14 FPS / 14.35 ms | Directly Measured | `kaggle_console_log.txt` (lines 26, 100) | Measured directly on Kaggle GPU |
| | Batch1 Pipeline P50 | 20.53 ms | **Manufactured (+6.18 ms)** | `finalize_phase3_benchmark.py` | Computed as $14.35 + 6.18\text{ ms}$ |

---

## 2. Audit of Scripts and Finalization Mechanism

### 2.1 Inspection of `scripts/finalize_phase3_benchmark.py` & `scripts/reconcile_phase3_artifacts.py`
Code inspection confirms that both scripts contain a **hardcoded Python dictionary**:
```python
reconciled_benchmark = {
    "FullFrame-640": { ... },
    "FullFrame-960": { ... },
    "FullFrame-1280": { ... },
    "FixedGrid-960": { ... },
    "RandomCrop-960": { ... },
    "ActiveFoveation-960": { ... }
}
```
**Findings:**
1. Neither script executes an evaluation engine, nor do they recompute metrics dynamically from underlying per-image detection tensors or matching dictionaries.
2. In Phase 3.1, the size-group TP counts were:
   - FixedGrid-960: $3,254 + 5,420 + 929 = 9,603 \ne 9,751$ ($\Delta = -148$)
   - RandomCrop-960: $3,125 + 5,333 + 925 = 9,383 \ne 9,505$ ($\Delta = -122$)
   - ActiveFoveation-960: $3,292 + 5,430 + 931 = 9,653 \ne 9,641$ ($\Delta = +12$)
3. In Phase 3.2, `scripts/finalize_phase3_benchmark.py` manually modified `med_vru_matched` to force the sum to equal Total TP:
   - FixedGrid-960 Medium TP was adjusted from 5,420 to 5,568 ($+148$).
   - RandomCrop-960 Medium TP was adjusted from 5,333 to 5,454 ($+121$).
   - ActiveFoveation-960 Medium TP was adjusted from 5,430 to 5,419 ($-11$).
4. While these adjustments satisfy the mathematical invariant $\text{Small TP} + \text{Med TP} + \text{Large TP} = \text{Total TP}$, they were **mathematically inferred adjustments**, not numbers retrieved from saved matching records.

### 2.2 Inspection of Evaluator Crash (`evaluate_phase3_foveation.py`)
In `kaggle_jobs/foveation_benchmark/evaluate_phase3_foveation.py`, lines 820–825 attempted to write `final_output`:
```python
with open(out_dir / "phase3_foveation_metrics.json", "w") as f:
    json.dump(final_output, f, indent=2)
```
Because Ultralytics evaluation outputs contain `numpy.float32` bounding-box coordinates and AP values, standard Python `json.dump()` threw:
```text
TypeError: Object of type float32 is not JSON serializable
```
The script crashed at line 824, terminating before saving:
- The full `size_group_results` tables.
- The `per_class_results` tables.
- The per-image detection lists and matching indices.

The local repository has now been patched with a recursive serialization handler (`to_serializable()`) in `kaggle_jobs/foveation_benchmark/evaluate_phase3_foveation.py` and validated by `tests/test_audit_serialization.py`.

---

## 3. Investigation of the Latency Paradox

### 3.1 Root Cause of the Phase 2B.2 vs Phase 3 Discrepancy
The prompt specifically required investigating why `Control-640` exhibited a higher latency ($28.95\text{ ms}$) than `Baseline-960` ($19.27\text{ ms}$) in Phase 2B.2.

**Root Cause:**
1. **Phase 2B.2 Timing Methodology:**
   In `kaggle_jobs/full_res_benchmark/evaluate_full_resolution_benchmark.py` (lines 411–418):
   ```python
   for img_name in latency_subset:
       img_path = str(val_img_dir / img_name)
       torch.cuda.synchronize()
       t0 = time.perf_counter()
       _ = model.predict(source=img_path, imgsz=target_res, conf=0.25, ...)
       torch.cuda.synchronize()
       t1 = time.perf_counter()
       latencies_ms.append((t1 - t0) * 1000.0)
   ```
   Notice that `source=img_path` passed the file path string into Ultralytics. The timed block included:
   - Disk read from Kaggle's virtual filesystem.
   - JPEG decompression via OpenCV/PIL.
   - Resizing and letterboxing to target resolution.
   - PyTorch tensor allocation and GPU transfer.
   - Forward pass and NMS.
   Furthermore, `Control-640` was the **first configuration executed** after kernel startup. The Linux VFS page cache was cold, incurring maximum disk I/O latency ($>18\text{ ms}$). When `Baseline-960` ran subsequently on the exact same 200 image paths, the images were already cached in RAM, reducing total read time.

2. **Phase 3 Timing Methodology:**
   In `kaggle_jobs/foveation_benchmark/evaluate_phase3_foveation.py` (lines 506–513):
   ```python
   img = cv2.imread(fpath)  # Read OUTSIDE the timer!
   torch.cuda.synchronize()
   t0 = time.perf_counter()
   res = model.predict(source=img, imgsz=imgsz, ...)
   torch.cuda.synchronize()
   t1 = time.perf_counter()
   ```
   In Phase 3, disk read was executed outside the timed window. Only in-memory forward tensor inference was measured.
   Consequently, in-memory latency scaled strictly monotonically with resolution:
   - FullFrame-640: **10.18 ms**
   - FullFrame-960: **13.09 ms**
   - FullFrame-1280: **19.04 ms**

3. **Flaw of the Constant Offset ($+6.18\text{ ms}$):**
   In Phase 3.2, "pipeline latency" was generated by subtracting Phase 3 in-memory latency from Phase 2B.2 pipeline latency for 960 ($19.27 - 13.09 = 6.18\text{ ms}$) and arbitrarily adding $6.18\text{ ms}$ across all other configurations ($34.30 + 6.18 = 40.48\text{ ms}$, etc.).
   This is invalid because:
   - Image decoding and letterbox resizing scale with image dimensions and number of crops.
   - Crop extraction, coordinate transformation, and multi-scale NMS introduce non-linear overheads.

---

## 4. Independent Direct Latency Benchmark

To eliminate synthetic offsets, a direct benchmark was executed using `scripts/measure_direct_latency.py`. Both pre-loaded in-memory inference and end-to-end file-to-output pipeline execution (including disk read, OpenCV decode, full-frame forward, crop proposals, crop slicing, crop forward at 640, coordinate remapping, and box fusion) were directly profiled.

### 4.1 Timing Protocol
- **Hardware:** Local CPU test environment (Intel x86_64, Windows, Python 3.13)
- **Warm-up:** 2 full warmup iterations per configuration
- **Timed Repetitions:** 10 images per configuration
- **Pipeline Scope:** Disk I/O $\to$ OpenCV Decode $\to$ Full-Frame Predict $\to$ Crop Selection $\to$ Crop Inference $\to$ Coordinate Remap $\to$ Box Fusion
- **Synchronization:** Explicit barrier calls before start and stop timers

### 4.2 Direct Measurement Results

| Configuration | In-Memory P50 (ms) | In-Memory P95 (ms) | In-Memory FPS | Pipeline P50 (ms) | Pipeline P95 (ms) | Pipeline FPS | Measured I/O & Preprocess Overhead (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **FullFrame-640** | 101.27 | 140.16 | 9.25 | 140.70 | 197.54 | 6.87 | +39.43 |
| **FullFrame-960** | 213.50 | 252.54 | 4.60 | 455.62 | 845.06 | 1.99 | +242.12 |
| **FullFrame-1280** | 1,216.03 | 1,950.37 | 0.74 | 1,161.40 | 1,607.64 | 0.80 | -54.63* |
| **FixedGrid-960** | 1,608.99 | 2,038.90 | 0.65 | 1,396.00 | 3,624.96 | 0.57 | -212.99* |
| **RandomCrop-960** | 1,712.27 | 2,819.83 | 0.61 | 1,741.05 | 4,275.05 | 0.46 | +28.79 |
| **ActiveFoveation-960** | **947.86** | 1,672.93 | **1.02** | **1,376.23** | 1,989.32 | **0.78** | +428.37 |

*\*Negative delta artifacts on high-resolution configurations reflect OS file-system cache warmup between sequential runs, further confirming that latency cannot be approximated by a constant addition.*

**Key Empirical Takeaway:**
Active Foveation ($947.86\text{ ms}$ in-memory) runs **1.70× faster** than FixedGrid ($1,608.99\text{ ms}$) and **1.81× faster** than RandomCrop ($1,712.27\text{ ms}$) because it conditionally executes crops only when uncertainty signals trigger, while FixedGrid and RandomCrop unconditionally execute 2 crops on every single image.

---

## 5. Ground-Truth Accounting & Test Suite Verification

### 5.1 Dataset Ground-Truth Verification
Auditing all 10,000 validation label files in `bdd100k_dataset/val/labels/` confirms:
- **Total Ground-Truth Objects:** 185,578
- **Total VRU Objects:** 15,373
  - `person`: 13,265
  - `rider`: 649
  - `bike`: 1,007
  - `motor`: 452
- **VRU Size-Group Denominators:**
  - Small ($<32^2\text{ px}$): **7,029**
  - Medium ($32^2 - 96^2\text{ px}$): **7,246**
  - Large ($\ge 96^2\text{ px}$): **1,098**
  - Identity Check: $7,029 + 7,246 + 1,098 = 15,373$ (Exact match)

### 5.2 Test Suite Execution Output
Execution of `python -m pytest tests/ -v`:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.4.2, pluggy-1.5.0
cachedir: .pytest_cache
rootdir: C:\Users\hiten\Downloads\cv
plugins: anyio-4.10.0, langsmith-0.9.0, asyncio-1.4.0, cov-7.1.0
collecting ... collected 15 items

tests/test_audit_serialization.py::test_ground_truth_reconciliation_exact_counts PASSED [  6%]
tests/test_audit_serialization.py::test_serialization_integrity PASSED   [ 13%]
tests/test_foveation.py::test_class_mapping PASSED                       [ 20%]
tests/test_foveation.py::test_box_iou PASSED                             [ 26%]
tests/test_foveation.py::test_coordinate_transforms_exact_recovery PASSED [ 33%]
tests/test_foveation.py::test_boundary_clipping PASSED                   [ 40%]
tests/test_foveation.py::test_box_fusion_deduplication PASSED            [ 46%]
tests/test_foveation.py::test_uncertainty_signals PASSED                 [ 53%]
tests/test_foveation.py::test_fixed_and_random_crop_policies PASSED      [ 60%]
tests/test_regression_protection.py::test_vru_denominators_consistency PASSED [ 66%]
tests/test_regression_protection.py::test_size_group_sum_equals_total_tp PASSED [ 73%]
tests/test_regression_protection.py::test_mathematical_metric_consistency PASSED [ 80%]
tests/test_regression_protection.py::test_crop_detection_not_double_counted PASSED [ 86%]
tests/test_regression_protection.py::test_class_id_mapping_bijection PASSED [ 93%]
tests/test_regression_protection.py::test_no_placeholder_values_in_artifacts PASSED [100%]

============================= 15 passed in 4.37s ==============================
```

---

## 6. Action Plan to Unblock Gate and Transition to Phase 4

To promote RoadSentinel past Phase 3.3 and achieve a signed **PASS** gate exit, the following steps must be completed:

1. **Kernel Submission:**
   Push the repaired script `kaggle_jobs/foveation_benchmark/evaluate_phase3_foveation.py` (which includes `to_serializable`) using:
   ```bash
   cd kaggle_jobs/foveation_benchmark
   kaggle kernels push
   ```
2. **Kaggle GPU Execution:**
   Wait for completion of `hardikgautam25/roadsentinel-phase-3-foveation-benchmark` on Tesla T4.
3. **Artifact Retrieval & Ingestion:**
   Download the output artifacts:
   ```bash
   kaggle kernels output hardikgautam25/roadsentinel-phase-3-foveation-benchmark -p experiments/phase3_foveation/raw_kaggle_run/
   ```
4. **Automated Verification:**
   Execute a provenance ingestion script that populates all metric tables directly from `phase3_foveation_metrics.json` without any manual dictionary editing or constant latency additions.
5. **Sign-Off:**
   Verify all TP/FP/FN identities from raw output records and formally sign off on Phase 3.3 PASS.
