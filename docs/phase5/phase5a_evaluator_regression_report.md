# RoadSentinel Phase 5A — Evaluator Hardening and Regression Report

**Status:** COMPLETE  
**Evaluator Status:** HARDENED AND PROTECTED  
**Regression Test Suite Status:** `38 passed in 44.72s` (Exit Code: 0)  
**Historical Files Protected:** Preserved unchanged in `experiments/phase3_foveation/`, `experiments/phase4c_validation/`, `experiments/phase4d_validation/`

---

## 1. Executive Summary

Phase 5A Workstream A permanently resolves the baseline evaluation defect identified in Phase 4D-R. The ActiveFoveation-960 baseline fusion engine has been restored to strict canonical Phase 3.4/Phase 4C behavior, ensuring that sub-threshold detections down to $\text{conf}=0.01$ participate in fusion and coordinate blending prior to the operational confidence gate.

Comprehensive regression tests have been added to the test suite, proving:
1. Low-confidence detections participate in fusion and can boost across operational thresholds as designed.
2. Accidental pre-filtering cannot recur without failing automated test assertions.
3. The canonical aggregate counts ($9,641\text{ TP}, 5,901\text{ FP}, 5,732\text{ FN}$, $62.03\%$ Precision, $62.71\%$ Recall, $46.83\%$ Small-VRU Recall) reproduce exactly from retained execution records.

---

## 2. Evaluator Correction Details

### The Evaluator Defect
In `kaggle_jobs/calibration_benchmark/evaluate_phase4d_adaptive_area.py`, lines 194–204 previously implemented:
```python
def baseline_fuse_and_deduplicate(
    full_frame_dets: List[Dict[str, Any]],
    crop_dets: List[Dict[str, Any]],
    iou_thresh: float = 0.50,
) -> List[Dict[str, Any]]:
    # DEFECT: Hard confidence filter at 0.25 prior to fusion
    ff_filt = [d for d in full_frame_dets if d["conf"] >= 0.25]
    cr_filt = [d for d in crop_dets if d["conf"] >= 0.25]

    fused: List[Dict[str, Any]] = []
    ff_by_class = defaultdict(list)
    cr_by_class = defaultdict(list)

    for d in ff_filt: ff_by_class[d["class_name"]].append(d)
    for d in cr_filt: cr_by_class[d["class_name"]].append(d)
    ...
```

### The Permanent Evaluator Fix
In `kaggle_jobs/calibration_benchmark/evaluate_phase4d_adaptive_area.py`, the function was permanently corrected to:
```python
def baseline_fuse_and_deduplicate(
    full_frame_dets: List[Dict[str, Any]],
    crop_dets: List[Dict[str, Any]],
    iou_thresh: float = 0.50,
) -> List[Dict[str, Any]]:
    # Canonical Phase 3.4/4C baseline fusion: all detections down to conf=0.01
    # participate in fusion; operational threshold (0.25) is applied post-fusion.
    fused: List[Dict[str, Any]] = []
    ff_by_class = defaultdict(list)
    cr_by_class = defaultdict(list)

    for d in full_frame_dets: ff_by_class[d["class_name"]].append(d)
    for d in crop_dets: cr_by_class[d["class_name"]].append(d)
    ...
```

### Invariance of Calibrated Candidates
During this correction:
- `adaptive_area_wbf_consensus_fusion` was **not altered**.
- Calibrated thresholds ($\tau_{\text{isolated}} = 0.40 / 0.28 / 0.30 / 0.32$, $\tau_{\text{confirmed}} = 0.20$, $\tau_{\text{ff}} = 0.25$) were **not altered**.
- WBF consensus parameters ($\beta = +0.08, 1.2\times$ crop weight, $\text{IoU} \ge 0.50$) were **not altered**.

---

## 3. Regression Protection & Test Assertions

A dedicated regression suite was created in [tests/test_phase5a_evaluator_hardening.py](file:///c:/Users/hiten/Downloads/cv/tests/test_phase5a_evaluator_hardening.py) containing four automated test cases:

### A. Sub-Threshold Fusion Participation Test (`test_baseline_fusion_subthreshold_participation`)
Constructs a synthetic full-frame detection ($c_{\text{ff}} = 0.24$) and a crop detection ($c_{\text{crop}} = 0.22$) sharing high IoU ($0.95$).
- Proves both Phase 4D and Phase 4C baseline fusion functions produce identical fused detections with confidence:
  $$c_{\text{fused}} = 0.24 + 0.05 \times 0.22 = 0.2510 \ge 0.25$$
- Asserts that both coordinate arrays and confidences match within numerical precision ($10^{-4}$).

### B. Pre-Filtering Guard Test (`test_prefiltering_regression_guard`)
Proves that if an operational pre-filter ($c \ge 0.25$) were introduced prior to fusion, both sub-threshold boxes would be discarded, whereas canonical fusion retains them.

### C. Exact Canonical Aggregate Count Assertions (`test_canonical_aggregate_reconstruction_from_records`)
Directly loads and parses `experiments/phase3_foveation/runs/kaggle_run_v2/per_image_records.jsonl` ($60,000$ lines) and raw ground-truth annotations from `bdd100k_dataset/val/labels/` across all 10,000 validation images.

Asserts exact agreement:
- **Total VRU Ground Truth:** $15,373$ ($7,029$ Small, $7,246$ Medium, $1,098$ Large).
- **VRU True Positives (TP):** Exactly **9,641**.
- **VRU False Negatives (FN):** Exactly **5,732** ($15,373 - 9,641$).
- **VRU Recall:** Exactly **62.71%** ($9,641 / 15,373 = 0.627138$).
- **Small-VRU TP:** Exactly **3,292** (Recall: **46.83%** = $3,292 / 7,029$).
- **Medium-VRU TP:** Exactly **5,414** (Recall: **74.72%** = $5,414 / 7,246$).
- **Large-VRU TP:** Exactly **935** (Recall: **85.15%** = $935 / 1,098$).

### D. Partition Disjointness Test (`test_partition_membership_disjointness`)
Asserts that:
- Pilot cohort contains exactly $300$ images.
- Non-pilot confirmation partition contains exactly $9,700$ images.
- $\text{Pilot} \cap \text{Non-Pilot} = \emptyset$ (strictly disjoint).
- $\text{Pilot} \cup \text{Non-Pilot} = 10,000$ images.

---

## 4. Documentation of Metric Provenance & Reconstruction Limits

In accordance with scientific rigor, we explicitly distinguish metrics that can be independently recomputed from retained records from those requiring new inference:

| Metric | Source of Truth | Recomputation Status | Provenance & Limits |
|---|---|---|---|
| **VRU True Positives (TP)** | `per_image_records.jsonl` (P3.4) + BDD100K Labels | **100% Independently Verified** | Exactly 9,641. Ground-truth match indices (`gt_idx`) are fully preserved for all 10,000 images. |
| **VRU False Negatives (FN)** | Formula: $\text{FN} = \text{GT} - \text{TP}$ | **100% Independently Verified** | Exactly 5,732. Directly follows from 15,373 GT objects minus 9,641 TPs. |
| **VRU Recall** | Formula: $\text{Rec} = \frac{\text{TP}}{\text{GT}}$ | **100% Independently Verified** | Exactly 62.7139% (rounds to 62.71%). |
| **Small-VRU Recall** | `per_image_records.jsonl` + Label Box Sizes | **100% Independently Verified** | Exactly 46.8345% ($3,292 / 7,029$). Reconstructed from label areas $< 1024\text{ px}^2$. |
| **Medium-VRU Recall** | `per_image_records.jsonl` + Label Box Sizes | **100% Independently Verified** | Exactly 74.7171% ($5,414 / 7,246$). Reconstructed from label areas $[1024, 9216)\text{ px}^2$. |
| **Large-VRU Recall** | `per_image_records.jsonl` + Label Box Sizes | **100% Independently Verified** | Exactly 85.1548% ($935 / 1,098$). Reconstructed from label areas $\ge 9216\text{ px}^2$. |
| **VRU False Positives (FP)** | `phase3_foveation_metrics.json` | **Audited from Metric Artifact** | Exactly 5,901. `per_image_records.jsonl` does not store false-positive bounding box coordinates. |
| **VRU Precision** | Formula: $\text{Prec} = \frac{\text{TP}}{\text{TP} + \text{FP}}$ | **Audited from Metric Artifact** | Exactly 62.0319% ($9,641 / 15,542$). Follows directly from 9,641 TP and 5,901 FP. |
| **Overall & VRU mAP@50** | `phase3_foveation_metrics.json` | **Audited from Metric Artifact** | $0.5403$ and $0.5120$. Requires full raw prediction score arrays down to $0.01$ across all 10,000 images. |

> [!IMPORTANT]
> **Methodological Boundary Statement:** While exact agreement with canonical aggregate counts ($9,641\text{ TP}, 5,901\text{ FP}, 5,732\text{ FN}$, $46.83\%\text{ Small Recall}$) is mathematically proven from execution traces and labels, this does not imply that every prediction coordinate or continuous AP curve is identical across historical inference executions.

---

## 5. Test Suite Execution Output

The regression suite was executed across the entire repository with the following results:

```
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.4.2, pluggy-1.5.0
rootdir: C:\Users\hiten\Downloads\cv
plugins: anyio-4.10.0, langsmith-0.9.0, asyncio-1.4.0, cov-7.1.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 38 items

tests\test_audit_serialization.py ..                                     [  5%]
tests\test_foveation.py .......                                          [ 23%]
tests\test_phase3_4_reconciliation.py .                                  [ 26%]
tests\test_phase4b_pilot.py ....                                         [ 36%]
tests\test_phase4c_evaluator.py ......                                   [ 52%]
tests\test_phase4d_adaptive_area.py ........                             [ 73%]
tests\test_phase5a_evaluator_hardening.py ....                           [ 84%]
tests\test_regression_protection.py ......                               [100%]

============================= 38 passed in 44.72s =============================
```

- **Total Tests Collected:** 38
- **Total Tests Passed:** 38
- **Total Tests Failed:** 0
- **Exit Code:** `0`
