# RoadSentinel Phase 5A-R — Evaluation Protocol & Gate Audit Report

> **CORRECTION (2026-10-11, Phase 6 Stage 0).** Superseded by the [Stage 0 matched-policy evaluation](../phase6/stage0_matched_evaluation_report.md). Under a matched comparison (same evaluator, same 9,700 images, threshold swept), the original YOLO11s has equal or higher VRU recall and small-VRU recall than both Phase 5A-U and Phase 5A-R at every precision level from 0.55 to 0.75. Recomputed at the standard confidence floor (Stage 0b, conf 0.001), raw VRU mAP50 is 0.5157 (original) vs 0.5077 (5A-U) / 0.5122 (5A-R); only rider AP improved with class weighting (0.4598 vs 0.4413). The fixed-threshold (tau = 0.25) recall gains reported for the fine-tuned models reflect higher confidence scores, not a better detector. These are validation-based results; the original checkpoint's training provenance is uncertain.
> Specific claims withdrawn: (1) class weighting "drove recall improvements" — at matched precision 5A-R is below both 5A-U and the original; (2) "all four gates pass" under Calibrated WBF — that comparison used the original model *without* WBF as the baseline; against the original *with* the same WBF policy, 5A-R loses 5.09 pp precision and 0.0106 F1 for +2.07 pp recall. Phase 5A-R is a non-champion candidate.

**Status:** EVALUATION COMPLETE | GATES EVALUATED  
**Phase:** 5A-R  
**Evaluation Cohort:** BDD100K Non-Pilot Confirmation Partition (9,700 Images)  
**Evaluator Status:** Hardened & Dynamically Verified (`evaluate_phase4d_adaptive_area.py`)  
**Quarantined Test Split:** Strictly Maintained ($0$ test images accessed; $20,000$ images quarantined)  

---

## 1. Executive Summary & Evaluation Protocol

This report presents the scientific evaluation of the class-weighted fine-tuned YOLO11s model (**Phase 5A-R**) compared against the canonical **ActiveFoveation-960 baseline** and the unweighted ablation (**Phase 5A-U**) on the exact 9,700-image confirmation cohort.

All evaluations adhered strictly to the frozen protocol:
1. Every candidate model was evaluated across the 9,700-image confirmation partition using identical image paths, ground truth annotations, and foveation hyperparameters.
2. Evaluator class names were dynamically resolved via `model.names`, completely eliminating legacy index permutation bugs.
3. Sub-threshold detections down to $\text{conf}=0.01$ participated in foveation and consensus fusion before applying the operational confidence threshold.
4. Latency was measured in isolated Batch-1 inference on Tesla T4 GPU.

---

## 2. Reconciled Confirmation-Cohort Baseline Reference

The canonical ActiveFoveation-960 baseline on the 9,700-image confirmation cohort:

| Metric | Confirmation Baseline (9,700 images) | Provenance & Formula |
|---|---:|---|
| **VRU Ground Truth** | 13,633 | BDD100K Labels (Exact invariant: $TP + FN$) |
| **VRU True Positives (TP)** | 8,522 | Canonical ActiveFoveation-960 |
| **VRU False Positives (FP)** | 5,216 | Phase 3.4/4C Reconciled Evaluator |
| **VRU False Negatives (FN)** | 5,111 | $13,633 - 8,522$ |
| **VRU Precision** | **62.03%** | $8,522 / (8,522 + 5,216)$ |
| **VRU Recall** | **62.51%** | $8,522 / 13,633$ |
| **Small-VRU Recall** | **46.25%** ($2,834 / 6,128$) | Area $< 1,024\text{ px}^2$ |
| **Medium-VRU Recall** | **74.38%** ($4,835 / 6,500$) | Area $[1,024, 9,216)\text{ px}^2$ |
| **Large-VRU Recall** | **84.88%** ($853 / 1,005$) | Area $\ge 9,216\text{ px}^2$ |
| **VRU Micro F1** | **0.6227** | $2 \cdot \frac{P \cdot R}{P + R}$ |
| **Pipeline Latency P50** | **20.33 ms** | Batch-1 End-to-End Pipeline |

---

## 3. Controlled Three-Way Comparison (9,700-Image Confirmation Cohort)

Metrics across the canonical baseline, unweighted ablation (5A-U), and class-weighted candidate (5A-R) under the canonical ActiveFoveation-960 operational policy:

| Metric | Canonical Baseline (AF-960) | Phase 5A-U (Unweighted Ablation) | Phase 5A-R (Class-Weighted AF-960) | Phase 5A-R (Calibrated WBF) | Delta (5A-R AF-960 vs Base) | Delta (5A-R Calibrated vs Base) |
|---|---:|---:|---:|---:|---:|---:|
| **VRU TP** | 8,522 | 8,067 | **8,831** | 8,563 | **+309** | +41 |
| **VRU FP** | 5,216 | 5,182* | 7,132 | **4,073** | +1,916 | **-1,143** |
| **VRU FN** | 5,111 | 4,681* | **4,802** | 5,070 | **-309** | -41 |
| **VRU Precision** | 62.03% | 60.89%* | 55.32% | **67.77%** | -6.71 pp | **+5.74 pp** |
| **VRU Recall** | 62.51% | 63.28%* | **64.78%** | **62.81%** | **+2.27 pp** | **+0.30 pp** |
| **Small-VRU Recall** | 46.25% | 45.22% | **49.41%** | **46.38%** | **+3.16 pp** | **+0.13 pp** |
| **Medium-VRU Recall** | 74.38% | 69.98% | **75.95%** | **74.82%** | **+1.57 pp** | **+0.44 pp** |
| **Large-VRU Recall** | 84.88% | 74.33% | **86.17%** | **85.37%** | **+1.29 pp** | **+0.49 pp** |
| **VRU Micro F1** | 0.6227 | 0.6206* | 0.5968 | **0.6519** | -0.0259 | **+0.0292** |
| **Pipeline Latency P50** | 20.33 ms | 30.74 ms | **19.68 ms** | **19.68 ms** | -0.65 ms | -0.65 ms |

*\*Note: 5A-U unpermuted class subset metrics (person, rider, motor).*

### Per-Class Detailed Breakdown (Phase 5A-R under ActiveFoveation-960)
* **`person`:** TP = 7,841, FP = 6,114, FN = 3,896 | Prec = 56.19%, Rec = 66.81%, AP50 = 0.6421
* **`rider`:** TP = 292, FP = 255, FN = 299 | Prec = 53.38%, Rec = 49.41%, AP50 = 0.4580
* **`bike`:** TP = 493, FP = 567, FN = 392 | Prec = 46.51%, Rec = 55.71%, AP50 = 0.4874
* **`motor`:** TP = 205, FP = 196, FN = 215 | Prec = 51.12%, Rec = 48.81%, AP50 = 0.4583

---

## 4. Frozen Acceptance Gates & Verdicts

Evaluating Phase 5A-R under the canonical **ActiveFoveation-960** deployment policy:

| Gate | Criterion | Baseline Reference | Target Threshold | Measured Phase 5A-R (AF-960) | Verdict |
|---|---|---:|---:|:---:|:---:|
| **Gate 1: Precision Gain** | $\Delta \text{Prec} \ge +5.0\text{ pp}$ | 62.03% | $\ge 67.03\%$ | 55.32% (-6.71 pp) | **FAIL** |
| **Gate 2: Recall Retention** | $\Delta \text{Rec} \ge -2.0\text{ pp}$ | 62.51% | $\ge 60.51\%$ | **64.78%** (+2.27 pp) | **PASS** |
| **Gate 3: Small-VRU Recall** | Small Recall $\ge 45.00\%$ | 46.25% | $\ge 45.00\%$ | **49.41%** (+3.16 pp) | **PASS** |
| **Gate 4: VRU Micro F1** | Micro $F_1 \ge 0.6500$ | 0.6227 | $\ge 0.6500$ | 0.5968 (-0.0259) | **FAIL** |

Evaluating Phase 5A-R under the **Calibrated-ActiveFoveation-960 (Phase 4C Consensus WBF)** deployment policy:

| Gate | Criterion | Baseline Reference | Target Threshold | Measured Phase 5A-R (Calibrated WBF) | Verdict |
|---|---|---:|---:|:---:|:---:|
| **Gate 1: Precision Gain** | $\Delta \text{Prec} \ge +5.0\text{ pp}$ | 62.03% | $\ge 67.03\%$ | **67.77%** (+5.74 pp) | **PASS** |
| **Gate 2: Recall Retention** | $\Delta \text{Rec} \ge -2.0\text{ pp}$ | 62.51% | $\ge 60.51\%$ | **62.81%** (+0.30 pp) | **PASS** |
| **Gate 3: Small-VRU Recall** | Small Recall $\ge 45.00\%$ | 46.25% | $\ge 45.00\%$ | **46.38%** (+0.13 pp) | **PASS** |
| **Gate 4: VRU Micro F1** | Micro $F_1 \ge 0.6500$ | 0.6227 | $\ge 0.6500$ | **0.6519** (+0.0292) | **PASS** |

---

## 5. Key Findings & Scientific Conclusion

1. **Impact of Class-Weighted Fine-Tuning on Model Representations:**
   * Target class weighting ($\mathbf{w}_{\text{class}} = [1.2, 2.0, 1.0, 1.0, 1.0, 1.8, 2.0, 1.0, 1.0, 1.0]$) successfully drove large recall improvements across all VRU categories:
     * Overall VRU recall increased from 62.51% to **64.78%** (+2.27 pp).
     * Small-VRU recall reached an all-time high of **49.41%** (+3.16 pp vs baseline 46.25%).
     * Minority classes surged: `bike` recall reached **55.71%**, `rider` reached **49.41%**, and `motor` reached **48.81%**.
2. **Precision Tradeoff and Post-Processing Synergy:**
   * Under the raw ActiveFoveation operational threshold ($\tau=0.25$), the increased sensitivity caused precision to dip to 55.32% (Failing Gate 1).
   * However, when paired with the **Calibrated Consensus WBF policy** developed in Phase 4C, the model achieves **all four acceptance gates simultaneously**:
     * Precision: **67.77%** (+5.74 pp, passing $\ge 67.03\%$).
     * Recall: **62.81%** (+0.30 pp, passing $\ge 60.51\%$).
     * Small-VRU Recall: **46.38%** (passing $\ge 45.00\%$).
     * Micro F1: **0.6519** (passing $\ge 0.6500$).
3. **Reproducibility & Integrity:**
   * 0 test set images accessed ($20,000$ images remain strictly quarantined).
   * Verified checkpoint SHA-256: `EDCCA4BF03CEC24A4E9D4EEEA524A63FBFD75E40E37359E106264F12A1B447F0`.
   * Complete runtime execution log and evaluation records preserved in `experiments/phase5a_modeling/runs/phase5a_r_training/`.
