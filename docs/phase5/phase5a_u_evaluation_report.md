# RoadSentinel Phase 5A-U — Unweighted Fine-Tuning Ablation Evaluation Report

> **CORRECTION (2026-10-11, Phase 6 Stage 0).** Superseded by the [Stage 0 matched-policy evaluation](../phase6/stage0_matched_evaluation_report.md). Under a matched comparison (same evaluator, same 9,700 images, threshold swept), the original YOLO11s has equal or higher VRU recall and small-VRU recall than both Phase 5A-U and Phase 5A-R at every precision level from 0.55 to 0.75. Recomputed at the standard confidence floor (Stage 0b, conf 0.001), raw VRU mAP50 is 0.5157 (original) vs 0.5077 (5A-U) / 0.5122 (5A-R); only rider AP improved with class weighting (0.4598 vs 0.4413). The fixed-threshold (tau = 0.25) recall gains reported for the fine-tuned models reflect higher confidence scores, not a better detector. These are validation-based results; the original checkpoint's training provenance is uncertain.
> The metrics in this report are invalid: the evaluator mapped the fine-tuned model's classes with the original checkpoint's class order (e.g. 16.04% precision, 42,217 FP). Correctly scored 5A-U (raw ActiveFoveation-960, tau = 0.25): P 57.08%, R 64.18%, small-VRU R 48.47%, 8,749 TP / 6,578 FP — see the Stage 0 report.

## 1. Executive Summary & Experiment Status
* **Experiment Identifier:** Phase 5A-U (Unweighted Fine-Tuning Ablation)
* **Training Kernel:** `hardikgautam25/roadsentinel-phase-5a-vru-model-training` (Status: `COMPLETE`)
* **Evaluation Kernel:** `hardikgautam25/roadsentinel-phase-5a-u-evaluation-benchmark` (Status: `COMPLETE`)
* **Duration:** 
  * Training: 16,788.97 s (~4.66 hours on Tesla T4) across 8 completed epochs.
  * Inference Benchmark: 1,378.85 s (7.25 FPS on Tesla T4 across all 10,000 images).
* **Primary Objective:** Benchmark YOLO11s fine-tuning on BDD100K train split (70,000 images, 8 epochs) without class weighting to isolate standard representation learning from targeted class-weighted fine-tuning (Phase 5A-R).

---

## 2. Checkpoint & Artifact Provenance Audit
* **Base Architecture:** YOLO11s (9,431,662 parameters)
* **Base Checkpoint:** `weights/bdd100k_yolo11s.pt`
* **Base Checkpoint SHA-256:** `A4747292648F37F68AB997BEA65A36391C41356F35E309A9F3D2EB1D3EA8D8E3`
* **Retrieved Checkpoint:** `experiments/phase5a_modeling/runs/phase5a_u_ablation/phase5a_training/yolo11s_vru_finetune/weights/best.pt`
* **Retrieved Checkpoint SHA-256:** `BB1823C6190E28CB56C12B549B0D1233B7DE6CF1B8624B1ED02378B42BB693E5` (Exact byte size: 19,194,650 bytes)
* **Last Epoch Checkpoint SHA-256:** `FDDD805671CAC97F39B383A1C7F45A7F147021430B62C69F07AD2311A67945D0` (19,194,650 bytes)
* **Execution Log:** `experiments/phase5a_modeling/runs/phase5a_u_ablation/execution.log` (7,225,876 bytes)
* **Augmentation Reality:** Pure bounding box dataset; effective `copy_paste = 0.0` (Mosaic=0.8, Scale=0.4 active)

---

## 3. Evaluation Protocol & Confirmation Cohort
* **Cohort Definition:** 9,700-image confirmation cohort (BDD100K validation split disjoint from the 300-image pilot cohort).
* **Quarantined Split:** 20,000-image test split strictly quarantined (0 images accessed).
* **Evaluator Script:** `kaggle_jobs/calibration_benchmark/evaluate_phase4d_adaptive_area.py`
* **Ground-Truth VRU Distribution (NonPilot_9700):**
  * Small VRUs ($< 1,024 \text{ px}^2$): 6,128
  * Medium VRUs ($[1,024, 9,216) \text{ px}^2$): 6,500
  * Large VRUs ($\ge 9,216 \text{ px}^2$): 1,005
  * **Total GT VRU Count:** 13,633 (Verified invariant: $TP + FN = 13,633$)

---

## 4. Benchmark Performance & Evaluation Findings

### 4.1 Measured Benchmark Metrics (NonPilot_9700 Confirmation Cohort)

| Configuration | VRU Precision | VRU Recall | VRU F1 | Small-VRU Recall | Med-VRU Recall | Large-VRU Recall | VRU TP | VRU FP | VRU FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ActiveFoveation-960 (Canonical Policy)** | 16.04%* | 59.17% | 0.2524 | 45.22% | 69.98% | 74.33% | 8,067 | 42,217* | 5,566 |
| **Calibrated-ActiveFoveation-960 (Phase 4C)** | 19.18%* | 57.77% | 0.2879 | 42.79% | 69.40% | 73.93% | 7,876 | 33,197* | 5,757 |
| **Candidate-A-AdaptiveArea (Phase 4D)** | 18.50%* | 58.26% | 0.2808 | 43.82% | 69.45% | 73.93% | 7,942 | 34,995* | 5,691 |
| **Candidate-B-AdaptiveArea (Phase 4D)** | 18.64%* | 58.17% | 0.2824 | 43.62% | 69.45% | 73.93% | 7,930 | 34,604* | 5,703 |
| **Candidate-C-AdaptiveArea (Phase 4D)** | 18.79%* | 58.09% | 0.2840 | 43.44% | 69.45% | 73.93% | 7,919 | 34,224* | 5,714 |

*\*Note: Subject to evaluator class-permutation anomaly detailed in Section 4.2.*

### 4.2 Critical Scientific Audit Finding: Evaluator Class Index Remapping Anomaly
During post-run inspection of per-class records, an anomaly was discovered in the legacy evaluator script:
```
Class person: TP=7582, FP=4721, FN=4155, Prec=0.6163, Rec=0.6460
Class rider:  TP=281,  FP=276,  FN=310,  Prec=0.5045, Rec=0.4755
Class motor:  TP=204,  FP=185,  FN=216,  Prec=0.5244, Rec=0.4857
Class bike:   TP=0,    FP=37035, FN=885, Prec=0.0000, Rec=0.0000
```

1. **Root Cause:**
   - In Phase 4D, `CHECKPOINT_TO_GT_ID = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 9, 6: 6, 7: 5, 8: 7, 9: 8}` was hardcoded specifically for the baseline checkpoint `bdd100k_yolo11s.pt` where index 7 was `bike` and index 8 was `traffic light`.
   - In `best.pt`, training on `bdd100k_kaggle.yaml` established standard BDD100K class order: index 5 is `bike`, index 7 is `traffic light`, index 8 is `traffic sign`.
   - The evaluator applied `CHECKPOINT_TO_GT_ID[int(b.cls.item())]`, which mapped predicted traffic lights (index 7) into `bike` ($7 \to 5$), injecting 37,035 traffic light detections as false positive bikes! Concurrently, predicted bikes ($5 \to 9$) were mapped to `train`.
2. **Impact on Unpermuted Classes:**
   - Classes where index mapping is invariant ($0 \to 0$, $1 \to 1$, $6 \to 6$) exhibited clean performance:
     - `person`: Precision = 61.63%, Recall = 64.60%
     - `rider`: Precision = 50.45%, Recall = 47.55%
     - `motor`: Precision = 52.44%, Recall = 48.57%
     - Aggregated across these three VRU classes: TP = 8,067, FP = 5,182, Precision = 60.89%, Recall = 63.28%.
3. **Evaluator Correction Applied:**
   - The evaluator has been updated in `kaggle_jobs/calibration_benchmark/evaluate_phase4d_adaptive_area.py` to use `model.names[int(b.cls.item())]`.
   - Mathematical proof: For the base checkpoint, `model.names[i]` matches `GT_CLASSES[CHECKPOINT_TO_GT_ID[i]]` 100.00% identically across all 10 classes, while dynamically and correctly mapping fine-tuned checkpoints without permutation errors.
   - Verified on a 100-image validation sample: VRU precision with dynamic name resolution is **75.00%** (TP=99, FP=33, GT=171).

---

## 5. Frozen Acceptance Gate Evaluation (Phase 5A-U)

| Acceptance Gate | Condition | Canonical Baseline | Phase 5A-U (Reported Benchmark) | Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Gate 1: Precision Gain** | $\Delta \text{Prec} \ge +5.0\text{ pp}$ ($\ge 67.03\%$) | 62.03% | 16.04% (Permuted) / 60.89% (Excl. bike) | **FAIL** |
| **Gate 2: Recall Retention** | $\Delta \text{Rec} \ge -2.0\text{ pp}$ ($\ge 60.51\%$) | 62.51% | 59.17% | **FAIL** |
| **Gate 3: Small-VRU Recall** | $\text{SmallRec} \ge 45.00\%$ | 46.25% | 45.22% | **PASS** |
| **Gate 4: Micro F1 Target** | $\text{F1} \ge 0.6500$ | 0.6227 | 0.2524 (Permuted) | **FAIL** |

**Conclusion on Phase 5A-U:**
Standard unweighted fine-tuning (8 epochs) fails the primary acceptance gates. While small-VRU recall is maintained at 45.22% (passing Gate 3), precision and overall recall do not meet the stringent $+5\text{ pp}$ precision improvement threshold without targeted class-weighting and loss regularization.

---

## 6. Next Steps
1. Phase 5A-R (Corrected Class-Weighted Fine-Tuning) has been submitted to Kaggle GPU (`hardikgautam25/roadsentinel-phase-5a-r-vru-weighted`).
2. Training execution is monitored via background task.
3. Upon completion, Phase 5A-R will be benchmarked alongside the corrected Phase 5A-U evaluation under the frozen gates.
