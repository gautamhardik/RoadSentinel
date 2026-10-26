# RoadSentinel Phase 4D-R — Baseline Reconciliation and Evaluator Audit

**Status:** COMPLETE  
**Audit Verdict:** `RECONCILED`  
**Target Investigation:** Discrepancy between Phase 3.4, Phase 4C, and Phase 4D ActiveFoveation-960 Baseline Metrics  
**Investigation Mode:** Read-only inspection, mathematical verification, code-path diffing, independent record recomputation  
**Test Split Quarantine:** Maintained (0 images accessed; BDD100K test split untouched)

---

## Executive Summary

Phase 4D-R resolves the material discrepancy identified between the historical ActiveFoveation-960 baseline reported in Phase 3.4 / Phase 4C and the baseline reported in Phase 4D:

| Metric | Phase 3.4 & Phase 4C Baseline | Phase 4D Baseline | Discrepancy ($\Delta$) | Root Cause Category |
|---|---:|---:|---:|---|
| **VRU True Positives (TP)** | 9,641 | 9,555 | **$-86$** | Pre-filter in `baseline_fuse_and_deduplicate` |
| **VRU False Positives (FP)** | 5,901 | 5,357 | **$-544$** | Pre-filter in `baseline_fuse_and_deduplicate` |
| **VRU False Negatives (FN)** | 5,732 | 5,818 | **$+86$** | Mathematical identity ($\text{FN} = \text{GT} - \text{TP}$) |
| **VRU Precision** | 62.03% | 64.08% | **$+2.05\text{ pp}$** | Elimination of 544 sub-threshold FP fusions |
| **VRU Recall** | 62.71% | 62.15% | **$-0.56\text{ pp}$** | Loss of 86 sub-threshold TP fusions |
| **Small-VRU Recall** | 46.83% | 45.87% | **$-0.96\text{ pp}$** | Loss of 68 small-VRU TP fusions |
| **Overall mAP@50** | 0.5403 | 0.4698 | **$-0.0705$** | Confidence truncation at 0.25 before COCO PR integration |
| **VRU Macro mAP@50** | 0.5120 | 0.4379 | **$-0.0741$** | Confidence truncation at 0.25 before COCO PR integration |

### Core Findings of the Audit

1. **Exact Mechanism Identified:** In `evaluate_phase4d_adaptive_area.py`, the baseline fusion function (`baseline_fuse_and_deduplicate`, lines 194–196) introduced a hard confidence pre-filter:
   ```python
   ff_filt = [d for d in full_frame_dets if d["conf"] >= 0.25]
   cr_filt = [d for d in crop_dets if d["conf"] >= 0.25]
   ```
   In contrast, Phase 3.4 (`evaluate_phase3_foveation.py`) and Phase 4C (`evaluate_phase4c_wbf.py`) passed all detector predictions down to $\text{conf}=0.01$ into fusion, allowing sub-threshold full-frame detections (e.g., $0.24$) and sub-threshold crop detections (e.g., $0.22$) to fuse and boost across the operational threshold ($0.24 + 0.05 \times 0.22 = 0.251 \ge 0.25$).
2. **Phase 4C Baseline Authenticity:** The Phase 4C evaluation script executed a live joint evaluation that reproduced the Phase 3.4 metrics to the exact integer ($9,641\text{ TP}, 5,901\text{ FP}, 5,732\text{ FN}$).
3. **Calibrated Reference Invariance:** The calibrated policy (`Calibrated-ActiveFoveation-960`) was **completely unaffected** by this change. Its metrics in Phase 4D match Phase 4C **bit-for-bit to every single digit** across all partitions (Full Val: $9,359\text{ TP}, 3,437\text{ FP}$; Non-Pilot: $8,281\text{ TP}, 3,084\text{ FP}$).
4. **Phase 4D Verdict Stability:** Re-evaluating Phase 4D Candidates A, B, and C against the true historical Phase 3.4 / Phase 4C baseline reveals that **all three candidates still fail Gate 3** (Candidate A: $44.52\% < 45.00\%$, Candidate B: $44.34\% < 45.00\%$, Candidate C: $44.08\% < 45.00\%$). The final verdict of `FAIL — one or more mandatory gates failed` is **100% robust and unchanged**.

---

## 1. Artifact & Checkpoint Inventory

Every relevant artifact from Phase 3.4, Phase 4C, and Phase 4D was audited and verified using SHA-256 checksums:

| Component | Artifact Path | SHA-256 Checksum |
|---|---|---|
| **Model Weights** | `weights/bdd100k_yolo11s.pt` | `A4747292648F37F68AB997BEA65A36391C41356F35E309A9F3D2EB1D3EA8D8E3` |
| **Phase 3.4 Script** | `kaggle_jobs/foveation_benchmark/evaluate_phase3_foveation.py` | `E8071E911B568555712E22338DF8FC5205313C1C906EC6AE1E67CB679B438B72` |
| **Phase 3.4 Metrics** | `experiments/phase3_foveation/runs/kaggle_run_v2/phase3_foveation_metrics.json` | `471BE320086FBCF489AAE361558BF49937961A58A6D9BE162C65EACB95219626` |
| **Phase 3.4 Records** | `experiments/phase3_foveation/runs/kaggle_run_v2/per_image_records.jsonl` | `83FE1A6C237F0F033503D0BE6F69D9206AC2B1763F73D6B5EE73429DF0CECCB3` |
| **Phase 4C Script** | `kaggle_jobs/calibration_benchmark/evaluate_phase4c_wbf.py` | `AD6D0E84C7F53C4E57943CD1C73919FB3B781EF48912DF59873925CA19BFB894` |
| **Phase 4C Metrics** | `experiments/phase4c_validation/phase4c_metrics.json` | `7ABC5410A6D558BD1138466C10C7D811F253A90B7053EA457FD2A84C074FBB91` |
| **Phase 4C Console** | `experiments/phase4c_validation/console.log` | `8BF903274291B77B907BE5B2F90833C174FE2C81DFE164FF6972230303DAF003` |
| **Phase 4D Script** | `kaggle_jobs/calibration_benchmark/evaluate_phase4d_adaptive_area.py` | `031326DAB734DBF54E95C4A0DF22057A75E99A9C57D875D17C7A12FE5D6A8729` |
| **Phase 4D Config** | `experiments/phase4d_validation/config.json` | `A1954DCEE45EF028CE8F485E62C8B58513F322E5E90F219344E1063E9914C865` |
| **Phase 4D Metrics** | `experiments/phase4d_validation/phase4d_metrics.json` | `26E02B948DB60F3ED78A718DFCC3D6F08C845DD26A47897252D526E38B5A70C1` |
| **Phase 4D Records** | `experiments/phase4d_validation/per_image_records.jsonl` | `E5C3F5E80800BCC5B117ABDC65C844807C33E53165F31F47F50F231E234E58BE` |
| **Phase 4D Console** | `experiments/phase4d_validation/console.log` | `4753D8E98C46D50C00A7115FA90FE85024A3586CE290CC795BD8BA76F683F472` |

---

## 2. Evaluation Implementation Diff Analysis

An exhaustive code audit comparing `evaluate_phase3_foveation.py`, `evaluate_phase4c_wbf.py`, and `evaluate_phase4d_adaptive_area.py` revealed the exact divergence point.

### A. The Baseline Fusion Code Diff

In `kaggle_jobs/calibration_benchmark/evaluate_phase4c_wbf.py` (lines 296–354) and `kaggle_jobs/foveation_benchmark/evaluate_phase3_foveation.py` (lines 297–359):
```python
def baseline_fuse_and_deduplicate(
    full_frame_dets: List[Dict[str, Any]],
    crop_dets: List[Dict[str, Any]],
    iou_thresh: float = 0.50,
) -> List[Dict[str, Any]]:
    fused: List[Dict[str, Any]] = []
    ff_by_class = defaultdict(list)
    cr_by_class = defaultdict(list)

    # NO PRE-FILTERING: All detections down to conf=0.01 enter fusion
    for d in full_frame_dets: ff_by_class[d["class_name"]].append(d)
    for d in crop_dets: cr_by_class[d["class_name"]].append(d)
    ...
```

In `kaggle_jobs/calibration_benchmark/evaluate_phase4d_adaptive_area.py` (lines 189–204):
```python
def baseline_fuse_and_deduplicate(
    full_frame_dets: List[Dict[str, Any]],
    crop_dets: List[Dict[str, Any]],
    iou_thresh: float = 0.50,
) -> List[Dict[str, Any]]:
    # DIVERGENCE: Baseline filters at tau=0.25 on both full frame and crops BEFORE fusion
    ff_filt = [d for d in full_frame_dets if d["conf"] >= 0.25]
    cr_filt = [d for d in crop_dets if d["conf"] >= 0.25]

    fused: List[Dict[str, Any]] = []
    ff_by_class = defaultdict(list)
    cr_by_class = defaultdict(list)

    for d in ff_filt: ff_by_class[d["class_name"]].append(d)
    for d in cr_filt: cr_by_class[d["class_name"]].append(d)
    ...
```

### B. Mathematical Impact of Pre-Filtering on Baseline Inference

1. **Sub-Threshold Confidence Boosting:**
   In Phase 3.4 and Phase 4C, the fusion formula was:
   $$\text{conf}_{\text{fused}} = \min(0.99, \max(c_{\text{ff}}, c_{\text{crop}}) + 0.05 \times \min(c_{\text{ff}}, c_{\text{crop}}))$$
   When an object had a marginal full-frame detection ($c_{\text{ff}} = 0.24$) and a marginal crop detection ($c_{\text{crop}} = 0.22$):
   $$\text{conf}_{\text{fused}} = 0.24 + 0.05 \times 0.22 = 0.2510 \ge 0.25$$
   In Phase 3.4 and Phase 4C, this boosted detection survived the final $0.25$ operational threshold. In Phase 4D, both detections were discarded prior to fusion.
2. **Sub-Threshold Coordinate Blending:**
   When an object had a coarse full-frame detection ($c_{\text{ff}} = 0.20 < 0.25$) matching a confident crop detection ($c_{\text{crop}} = 0.26 \ge 0.25$):
   In Phase 3.4/4C, the full-frame box blended with the crop box ($w_{\text{ff}} = 0.20, w_{\text{crop}} = 0.26 \times 1.2 = 0.312$), yielding a better-aligned bounding box and confidence $0.26 + 0.05 \times 0.20 = 0.270 \ge 0.25$.
   In Phase 4D, the full-frame detection was discarded, preventing coordinate blending and boosting.
3. **Clutter Elimination:**
   The pre-filter in Phase 4D similarly prevented sub-threshold clutter detections from combining and crossing the $0.25$ threshold, directly eliminating $544$ false positives ($5,901 \to 5,357$).
4. **COCO mAP@50 Depression:**
   In Phase 3.4 and Phase 4C, the full list of fused predictions down to $\text{conf}=0.01$ was passed into `compute_coco_ap_for_class`, evaluating the complete Precision-Recall curve.
   In Phase 4D, because `baseline_fuse_and_deduplicate` pre-filtered at $0.25$, zero predictions below $0.25$ existed in the output. The PR curve was truncated at recall corresponding to $\text{conf}=0.25$, causing the computed overall mAP@50 to fall from $0.5403$ to $0.4698$.

---

## 3. Independent Image-by-Image and Record-Level Recomputation

To verify that historical Phase 3.4 metrics were genuine direct measurements and not synthetic reconstructions, we independently parsed the $7.94\text{ MB}$ Kaggle execution artifact `experiments/phase3_foveation/runs/kaggle_run_v2/per_image_records.jsonl` ($60,000$ lines) and matched the recorded ground-truth indices (`gt_idx`) against the raw BDD100K validation labels.

### Independent Recomputation Results (Phase 3.4 Records vs Raw Labels)

```
========================================================================================
INDEPENDENT RECORD RECOMPUTATION (N = 10,000 Validation Images)
========================================================================================
Ground Truth Totals:
  - Small VRU (<32²):     7,029
  - Medium VRU [32²-96²): 7,246
  - Large VRU (≥96²):     1,098
  - Total VRU Ground Truth: 15,373

ActiveFoveation-960 Recomputed Recall:
  - Full Validation VRU TP: 9,641 / 15,373  ==> Recall = 62.71% (Exact match)
      * Small VRU TP:       3,292 / 7,029   ==> Recall = 46.83% (Exact match)
      * Medium VRU TP:      5,414 / 7,246   ==> Recall = 74.72% (Exact match)
      * Large VRU TP:         935 / 1,098   ==> Recall = 85.15% (Exact match)

  - Pilot Cohort (300 Images):
      * Pilot VRU TP:       1,119 / 1,740   ==> Recall = 64.31% (Exact match)
      * Small VRU TP:         458 / 901     ==> Recall = 50.83% (Exact match)
      * Medium VRU TP:        579 / 746     ==> Recall = 77.61% (Exact match)
      * Large VRU TP:          82 / 93      ==> Recall = 88.17% (Exact match)

  - Non-Pilot Confirmation (9,700 Images):
      * Non-Pilot VRU TP:   8,522 / 13,633  ==> Recall = 62.51% (Exact match)
      * Small VRU TP:       2,834 / 6,128   ==> Recall = 46.25% (Exact match)
      * Medium VRU TP:      4,835 / 6,500   ==> Recall = 74.38% (Exact match)
      * Large VRU TP:         853 / 1,005   ==> Recall = 84.88% (Exact match)
========================================================================================
```

### Partition Additivity Proof
- $\text{Pilot TP } (1,119) + \text{Non-Pilot TP } (8,522) = 9,641\text{ Full Val TP}$.
- $\text{Small TP } (458 + 2,834 = 3,292)$.
- $\text{Medium TP } (579 + 4,835 = 5,414)$.
- $\text{Large TP } (82 + 853 = 935)$.
- $\text{Pilot GT } (1,740) + \text{Non-Pilot GT } (13,633) = 15,373\text{ Full Val GT}$.

**Conclusion:** The Phase 3.4 and Phase 4C baseline numbers ($9,641\text{ TP}, 5,901\text{ FP}, 5,732\text{ FN}$) are genuine, reproducible evaluation metrics derived directly from the un-prefiltered baseline fusion code.

---

## 4. Comprehensive Mathematical Verification Across All Phases

We verified every mathematical identity across all partitions for Phase 3.4, Phase 4C, and Phase 4D:

| Configuration & Cohort | Reported TP | Reported FP | Reported FN | GT Sum ($\text{TP}+\text{FN}$) | Expected GT | Calculated Precision | Reported Precision | Calculated Recall | Reported Recall | Calculated F1 | Reported F1 | Identity Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **P3.4 ActiveFoveation (Full Val)** | 9,641 | 5,901 | 5,732 | 15,373 | 15,373 | 62.0319% | 62.03% | 62.7139% | 62.71% | 0.6237 | 0.6237 | **VERIFIED** |
| **P4C ActiveFoveation (Full Val)** | 9,641 | 5,901 | 5,732 | 15,373 | 15,373 | 62.0319% | 62.03% | 62.7139% | 62.71% | 0.6237 | 0.6237 | **VERIFIED** |
| **P4C ActiveFoveation (Pilot 300)** | 1,119 | 685 | 621 | 1,740 | 1,740 | 62.0288% | 62.03% | 64.3103% | 64.31% | 0.6315 | 0.6315 | **VERIFIED** |
| **P4C ActiveFoveation (Non-Pilot 9.7k)** | 8,522 | 5,216 | 5,111 | 13,633 | 13,633 | 62.0323% | 62.03% | 62.5097% | 62.51% | 0.6227 | 0.6227 | **VERIFIED** |
| **P4C Calibrated (Full Val)** | 9,359 | 3,437 | 6,014 | 15,373 | 15,373 | 73.1396% | 73.14% | 60.8795% | 60.88% | 0.6645 | 0.6645 | **VERIFIED** |
| **P4C Calibrated (Pilot 300)** | 1,078 | 353 | 662 | 1,740 | 1,740 | 75.3319% | 75.33% | 61.9540% | 61.95% | 0.6799 | 0.6799 | **VERIFIED** |
| **P4C Calibrated (Non-Pilot 9.7k)** | 8,281 | 3,084 | 5,352 | 13,633 | 13,633 | 72.8641% | 72.86% | 60.7423% | 60.74% | 0.6625 | 0.6625 | **VERIFIED** |
| **P4D ActiveFoveation (Full Val)** | 9,555 | 5,357 | 5,818 | 15,373 | 15,373 | 64.0759% | 64.08% | 62.1544% | 62.15% | 0.6310 | 0.6310 | **VERIFIED** |
| **P4D ActiveFoveation (Pilot 300)** | 1,102 | 609 | 638 | 1,740 | 1,740 | 64.4068% | 64.41% | 63.3333% | 63.33% | 0.6387 | 0.6387 | **VERIFIED** |
| **P4D ActiveFoveation (Non-Pilot 9.7k)** | 8,453 | 4,748 | 5,180 | 13,633 | 13,633 | 64.0330% | 64.03% | 62.0039% | 62.00% | 0.6300 | 0.6300 | **VERIFIED** |
| **P4D Calibrated (Full Val)** | 9,359 | 3,437 | 6,014 | 15,373 | 15,373 | 73.1396% | 73.14% | 60.8795% | 60.88% | 0.6645 | 0.6645 | **VERIFIED** |
| **P4D Calibrated (Pilot 300)** | 1,078 | 353 | 662 | 1,740 | 1,740 | 75.3319% | 75.33% | 61.9540% | 61.95% | 0.6799 | 0.6799 | **VERIFIED** |
| **P4D Calibrated (Non-Pilot 9.7k)** | 8,281 | 3,084 | 5,352 | 13,633 | 13,633 | 72.8641% | 72.86% | 60.7423% | 60.74% | 0.6625 | 0.6625 | **VERIFIED** |
| **P4D Candidate A (Full Val)** | 9,441 | 3,734 | 5,932 | 15,373 | 15,373 | 71.6584% | 71.66% | 61.4129% | 61.41% | 0.6614 | 0.6614 | **VERIFIED** |
| **P4D Candidate A (Pilot 300)** | 1,090 | 387 | 650 | 1,740 | 1,740 | 73.8000% | 73.80% | 62.6437% | 62.64% | 0.6776 | 0.6776 | **VERIFIED** |
| **P4D Candidate A (Non-Pilot 9.7k)** | 8,351 | 3,347 | 5,282 | 13,633 | 13,633 | 71.3883% | 71.39% | 61.2558% | 61.26% | 0.6594 | 0.6594 | **VERIFIED** |
| **P4D Candidate B (Full Val)** | 9,426 | 3,655 | 5,947 | 15,373 | 15,373 | 72.0587% | 72.06% | 61.3153% | 61.32% | 0.6625 | 0.6625 | **VERIFIED** |
| **P4D Candidate B (Pilot 300)** | 1,087 | 377 | 653 | 1,740 | 1,740 | 74.2486% | 74.25% | 62.4713% | 62.47% | 0.6785 | 0.6785 | **VERIFIED** |
| **P4D Candidate B (Non-Pilot 9.7k)** | 8,339 | 3,278 | 5,294 | 13,633 | 13,633 | 71.7827% | 71.78% | 61.1678% | 61.17% | 0.6605 | 0.6605 | **VERIFIED** |
| **P4D Candidate C (Full Val)** | 9,406 | 3,600 | 5,967 | 15,373 | 15,373 | 72.3205% | 72.32% | 61.1852% | 61.19% | 0.6629 | 0.6629 | **VERIFIED** |
| **P4D Candidate C (Pilot 300)** | 1,083 | 369 | 657 | 1,740 | 1,740 | 74.5868% | 74.59% | 62.2414% | 62.24% | 0.6786 | 0.6786 | **VERIFIED** |
| **P4D Candidate C (Non-Pilot 9.7k)** | 8,323 | 3,231 | 5,310 | 13,633 | 13,633 | 72.0357% | 72.04% | 61.0504% | 61.05% | 0.6609 | 0.6609 | **VERIFIED** |

---

## 5. Audit of Phase 4D Candidate Comparisons and Gate Sensitivity

A critical requirement of this audit is determining whether the baseline evaluation discrepancy altered any research conclusion or quality gate verdict in Phase 4D.

### A. Candidate Difference Isolation
We verified that Candidates A, B, and C differ strictly by their pre-registered isolated crop thresholds:
- Full-frame threshold: $\tau_{\text{ff}} = 0.25$ (identical).
- Confirmed crop threshold: $\tau_{\text{confirmed}} = 0.20$ (identical).
- Full-frame coarse confirmation floor: $\tau_{\text{coarse}} = 0.10, \text{IoU} \ge 0.30$ (identical).
- WBF consensus boost: $\beta = +0.08, 1.2\times$ crop weight (identical).
- Large isolated crop threshold ($\ge 1024\text{ px}^2$): $\tau_{\text{isolated}} = 0.45$ (identical).
- **Sole variation:** Small isolated crop threshold ($< 1024\text{ px}^2$): Candidate A = $0.28$, Candidate B = $0.30$, Candidate C = $0.32$.

### B. Quality Gate Sensitivity Audit (True Baseline vs Phase 4D Baseline)

The table below recomputes the Phase 4D quality gates on the 9,700-image confirmation partition using:
1. The **Phase 4D Baseline** (as executed in `evaluate_phase4d_adaptive_area.py`).
2. The **True Phase 3.4 / Phase 4C Baseline** (reconciled from historical records).

| Quality Gate | Gate Requirement | Under Phase 4D Baseline ($P=64.03\%, R=62.00\%$) | Under True Historical Baseline ($P=62.03\%, R=62.51\%$) | Gate Sensitivity Impact |
|---|---|---|---|:---:|
| **Gate 1: Precision Gain** | $\Delta \text{Prec} \ge +5.0\text{ pp}$ | Cand A: $+7.36\text{ pp}$ [PASS]<br>Cand B: $+7.75\text{ pp}$ [PASS]<br>Cand C: $+8.01\text{ pp}$ [PASS] | Cand A: $+9.36\text{ pp}$ [PASS]<br>Cand B: $+9.75\text{ pp}$ [PASS]<br>Cand C: $+10.01\text{ pp}$ [PASS] | **No change:** Passed under both baselines |
| **Gate 2: Recall Retention** | $\Delta \text{Rec} \ge -2.0\text{ pp}$ | Cand A: $-0.74\text{ pp}$ [PASS]<br>Cand B: $-0.83\text{ pp}$ [PASS]<br>Cand C: $-0.95\text{ pp}$ [PASS] | Cand A: $-1.25\text{ pp}$ [PASS]<br>Cand B: $-1.34\text{ pp}$ [PASS]<br>Cand C: $-1.46\text{ pp}$ [PASS] | **No change:** Passed under both baselines |
| **Gate 3: Small-VRU Recall** | Small Recall $\ge 45.00\%$ | Cand A: $44.52\%$ [FAIL]<br>Cand B: $44.34\%$ [FAIL]<br>Cand C: $44.08\%$ [FAIL] | Cand A: $44.52\%$ [FAIL]<br>Cand B: $44.34\%$ [FAIL]<br>Cand C: $44.08\%$ [FAIL] | **No change:** Absolute gate failed by all candidates |
| **Gate 4: VRU Micro F1** | F1 $\ge 0.6500$ | Cand A: $0.6594$ [PASS]<br>Cand B: $0.6605$ [PASS]<br>Cand C: $0.6609$ [PASS] | Cand A: $0.6594$ [PASS]<br>Cand B: $0.6605$ [PASS]<br>Cand C: $0.6609$ [PASS] | **No change:** Passed under both baselines |
| **Overall Verdict** | **All 4 Must Pass** | **FAIL** | **FAIL** | **100% Invariant** |

**Conclusion:** The baseline evaluation discrepancy had **zero impact** on the final decision or candidate selection. Under both baselines, all three candidates fail Gate 3, and no candidate qualifies for adoption.

---

## 6. Artifact Status & Missing Information Audit

| Artifact | Location | Size | Status / Missing Data | Impact on Audit |
|---|---|---|---|---|
| **Phase 3.4 Records** | `experiments/phase3_foveation/runs/kaggle_run_v2/per_image_records.jsonl` | 7.94 MB | Contains full 10k image TP match records (`gt_idx`), but omits FP coordinates and confidence scores. | Permits 100% reconstruction of TP, FN, and size-group recall. |
| **Phase 4C Records** | `experiments/phase4c_validation/per_image_records.jsonl` | 0 B | File was initialized but not written to during the Kaggle run loop. | Full metrics JSON is intact and matches Phase 3.4; no raw prediction coordinates saved. |
| **Phase 4D Records** | `experiments/phase4d_validation/per_image_records.jsonl` | 113 KB | Contains 500 representative images with candidate detection counts; omits full 10k per-box coordinates. | Full metrics JSON is intact; candidate counts verify policy differentiation. |

---

## 7. Audit Verdict & Corrective Actions

### Formal Verdict: **`RECONCILED`**

1. **Discrepancy Explained:** The difference between Phase 4C ($9,641\text{ TP}, 5,901\text{ FP}, 5,732\text{ FN}$) and Phase 4D ($9,555\text{ TP}, 5,357\text{ FP}, 5,818\text{ FN}$) is completely traced to the pre-filter introduced in `baseline_fuse_and_deduplicate` in `evaluate_phase4d_adaptive_area.py`.
2. **Canonical Baseline Established:** The canonical, authoritative baseline for ActiveFoveation-960 across the 10,000-image BDD100K validation set is:
   $$\text{TP} = 9,641, \quad \text{FP} = 5,901, \quad \text{FN} = 5,732$$
   $$\text{Precision} = 62.03\%, \quad \text{Recall} = 62.71\%, \quad \text{Small-VRU Recall} = 46.83\%$$
   $$\text{Overall mAP@50} = 0.5403, \quad \text{VRU Macro mAP@50} = 0.5120$$
3. **Corrective Action for Future Benchmark Scripts:**
   - Any future evaluation of ActiveFoveation-960 must remove lines 195–196 of `evaluate_phase4d_adaptive_area.py` (`ff_filt = ...; cr_filt = ...`) to preserve historical comparability.
   - When computing COCO mAP, predictions down to $\text{conf}=0.01$ must be retained so that Precision-Recall curves are not truncated at operational operating points.
4. **Research Decision:** The Phase 4D conclusion stands firm: adaptive area post-processing does not achieve $\ge 45.0\%$ small-VRU recall on the confirmation partition. Modeling improvements must be pursued at the architectural or feature representation level.
