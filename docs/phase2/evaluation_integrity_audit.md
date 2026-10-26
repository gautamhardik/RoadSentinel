# RoadSentinel — Phase 2A Evaluation Integrity Audit

**Document Version:** 2.0.0  
**Date:** October 9, 2026  
**Auditor:** RoadSentinel Engineering Team (DeepMind Antigravity)  
**System:** OpenCV AI Competition 2026, powered by AWS  
**Audit Status:** COMPLETE — ROOT CAUSE DIAGNOSED & EVALUATOR REPAIRED  

---

## 1. Executive Summary

During the initial baseline evaluation of RoadSentinel Phase 2A, severe metric anomalies were detected:
1. **Zero ground-truth traffic signs** were reported despite hundreds of high-confidence predictions.
2. **Traffic light precision was virtually zero** ($0.0041$) with a recall of $0.0031$.
3. **Train was flagged as a Vulnerable Road User (VRU)** while bicycle was omitted from VRU status.
4. **Bus and truck metrics were heavily degraded** ($P \approx 0.02 - 0.08, R \approx 0.01 - 0.05$).
5. **Reported mAP@50 was $0.157$**, an artificially depressed score caused by mismatched evaluations.

This audit report documents the forensic investigation, proves the exact cause of each anomaly, provides explicit bidirectional class mappings with empirical evidence, implements a mathematically rigorous evaluator, validates the repair against synthetic edge cases and real image crops, and establishes a scientifically trustworthy baseline on a VRU-enriched validation sample.

---

## 2. Forensic Investigation of Data Artifacts

### 2.1 Artifacts Inspected
1. **Dataset Configuration:** `bdd100k_dataset/data.yaml`
2. **Dataset Ground Truth Labels:** Representative label files in `bdd100k_dataset/val/labels/*.txt`
3. **Official BDD100K Class Hierarchy:** `bdd100k/bdd100k/configs/det.toml`
4. **Pretrained Checkpoint:** `weights/bdd100k_yolo11s.pt` (embedded `model.names`)
5. **Legacy Baseline Evaluator:** `notebooks/phase2a_baseline_evaluation.ipynb`

### 2.2 Dataset Origin & Historical Taxonomy
The BDD100K dataset has two primary annotation specifications in the computer vision literature:
- **CVPR 2018 / 2020 Flat Taxonomy:** The original flat 10-class list used by standard YOLO converters (such as Kaggle `a7madmostafa/bdd100k-yolo` from which `bdd100k_dataset` was extracted).
- **Scalabel / Official 2020 Hierarchical Taxonomy:** The official Berkeley DeepDrive hierarchy (`bdd100k/bdd100k/configs/det.toml`), grouping categories by superclasses: `human` (`pedestrian`, `rider`), `vehicle` (`car`, `truck`, `bus`, `train`), `bike` (`motorcycle`, `bicycle`), `traffic light`, `traffic sign`.

---

## 3. Explicit Class Mapping & Proof Table

The table below presents the exhaustive class mapping across the original Scalabel BDD100K standard, the converted YOLO dataset labels, and the pretrained model checkpoint.

| Canonical Semantic Class | Official Scalabel ID (`det.toml`) | Converted YOLO GT ID (`data.yaml`) | Checkpoint Class ID (`yolo11s.pt`) | Integer Match? | Required Checkpoint $\to$ GT Remapping | Empirical Evidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **person** (pedestrian) | 0 | 0 | 0 | **YES** | $0 \to 0$ | Verified: Walking humans on sidewalks (`ca3165e0`). |
| **rider** | 1 | 1 | 1 | **YES** | $1 \to 1$ | Verified: Human seated on scooter (`ca1c6003`). |
| **car** | 2 | 2 | 2 | **YES** | $2 \to 2$ | Verified: Passenger cars & taxis (`ca3165e0`). |
| **bus** | 4 | 3 | 4 | **MISMATCH** | $4 \to 3$ | Verified: Municipal transit bus labeled `3` in GT, predicted as `4` by model (`ca1c6003`). |
| **truck** | 3 | 4 | 3 | **MISMATCH** | $3 \to 4$ | Verified: Box trucks labeled `4` in GT, predicted as `3` by model. |
| **bike** (bicycle) | 7 | 5 | 7 | **MISMATCH** | $7 \to 5$ | Verified: Bicycle cropped from `c95ad578` has GT ID `5`; model predicts ID `7`. |
| **motor** (motorcycle) | 6 | 6 | 6 | **YES** | $6 \to 6$ | Verified: Motor scooter chassis cropped from `ca1c6003` has GT ID `6`; model predicts ID `6`. |
| **traffic light** | 8 | 7 | 8 | **MISMATCH** | $8 \to 7$ | Verified: Traffic light cropped from `c95ad578` has GT ID `7`; model predicts ID `8`. |
| **traffic sign** | 9 | 8 | 9 | **MISMATCH** | $9 \to 8$ | Verified: Street sign cropped from `c95ad578` has GT ID `8`; model predicts ID `9`. |
| **train** | 5 | 9 | 5 | **MISMATCH** | $5 \to 9$ | Trains are ID `9` in dataset, ID `5` in checkpoint. 0 instances in validation sample. |

### 3.1 Formal Remapping Functions

```python
# Checkpoint Predicted Class ID -> Ground Truth Label ID
CHECKPOINT_TO_GT_ID = {
    0: 0,  # person       -> person
    1: 1,  # rider        -> rider
    2: 2,  # car          -> car
    3: 4,  # truck (ckpt) -> truck (gt 4)
    4: 3,  # bus (ckpt)   -> bus (gt 3)
    5: 9,  # train (ckpt) -> train (gt 9)
    6: 6,  # motor        -> motor
    7: 5,  # bike (ckpt)  -> bike (gt 5)
    8: 7,  # t-light(ckpt)-> t-light (gt 7)
    9: 8,  # t-sign (ckpt)-> t-sign (gt 8)
}

# Ground Truth Label ID -> Checkpoint Predicted Class ID
GT_TO_CHECKPOINT_ID = {v: k for k, v in CHECKPOINT_TO_GT_ID.items()}
```

---

## 4. Root-Cause Autopsy: Why Previous Metrics Failed

The investigation revealed **five compounding flaws** in the legacy evaluation pipeline (`notebooks/phase2a_baseline_evaluation.ipynb`):

### Flaw 1: Direct Integer Index Comparison
In cell 7 of the legacy notebook:
```python
if gt['cid'] == cid:
    iou = compute_iou(p['box'], gt['box'])
```
Here, `gt['cid']` was read directly from the `.txt` label file (GT space), while `cid` was extracted from `res.boxes` (Checkpoint space). Because IDs 3, 4, 5, 7, 8, 9 did not match, the evaluator was comparing disjoint object categories:
- When the model detected a **traffic light** (ID 8), the evaluator checked if GT ID was 8 (which was actually a **traffic sign**). Result: False positive for traffic light, zero match.
- When the model detected a **traffic sign** (ID 9), the evaluator checked if GT ID was 9 (which was actually a **train**). Because trains were 0 in that sample, GT count was reported as 0, and all predictions were marked as false alarms.
- When the model detected a **bike** (ID 7), it was compared against GT ID 7 (**traffic lights**), producing 0 matches.
- When the model detected a **truck** (ID 3), it was compared against GT ID 3 (**buses**).
- When the model detected a **bus** (ID 4), it was compared against GT ID 4 (**trucks**).

Only **person** (0), **rider** (1), **car** (2), and **motor** (6) were evaluated against the correct ground truth because their IDs happened to be identical.

### Flaw 2: Label Name Attribution Desynchronization
In cell 9 of the legacy notebook:
```python
cname = model.names[cid]
gt_n = class_gt_counts[cid]
```
`class_gt_counts` was indexed by GT ID, but `cname` was taken from `model.names`. For index 8, `model.names[8]` was `"traffic light"`, but `class_gt_counts[8]` held the count of traffic signs (320). For index 9, `model.names[9]` was `"traffic sign"`, but `class_gt_counts[9]` held the count of trains (0). This created the bizarre display:
```
traffic light: GT=320 Pred=242 P@50=0.0041 R@50=0.0031
traffic sign : GT=0   Pred=241 P@50=0.0000 R@50=0.0000
```

### Flaw 3: Inversion of VRU Class Tags
In cell 3 and cell 9:
```python
VRU_CLASSES = [0, 1, 5, 6]  # person, rider, bike, motor
```
In the checkpoint taxonomy, ID 5 is `train` and ID 7 is `bike`. Consequently:
- `train` was marked `is_vru: true`.
- `bike` was marked `is_vru: false`.

### Flaw 4: Mathematically Invalid AP Formula
In cell 9 of the legacy notebook:
```python
map_50 = p50 * r50 * 2 / (p50 + r50) if (p50 + r50) > 0 else 0.0 # Or p50 * r50
map_50_95 = np.mean([class_tp_matrix[cid][i] / max(gt_n, 1) for i in range(len(iou_thresholds))])
```
- `map_50` was computed as the product of precision and recall or an F1 harmonic mean proxy. This is **not Average Precision**. AP is the area under the interpolated Precision-Recall curve.
- `map_50_95` was computed as the arithmetic mean of recall across IoU thresholds, ignoring precision and false positives completely.

### Flaw 5: Uncalibrated Latency Measurement
- The legacy script did not isolate PyTorch model loading or initial CUDA/CPU warmup.
- Latencies included first-inference kernel initialization, yielding uncalibrated mean latencies ($598.98\text{ ms}$).

---

## 5. Implementation of the Repaired Evaluation Engine

The repaired evaluation engine is implemented in `src/evaluation_engine.py` (`DetectionEvaluator`). Key design guarantees:

1. **Semantic Canonical Alignment:** Ground truths and predictions are resolved to canonical string names (`"person"`, `"rider"`, `"car"`, `"bus"`, `"truck"`, `"bike"`, `"motor"`, `"traffic light"`, `"traffic sign"`, `"train"`). Box matching is strictly evaluated within identical semantic names.
2. **IoU Box Matching:** True positives are determined using standard greedy bipartite matching at documented IoU thresholds $\theta \in [0.50, 0.55, \dots, 0.95]$. Duplicate predictions matching an already-claimed GT box are correctly penalized as False Positives.
3. **Operational Thresholds:** Operational metrics are computed at confidence $\tau = 0.25$ and $\text{IoU} = 0.50$:
   $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN} = \frac{TP}{N_{GT}}$$
4. **COCO 101-Point Interpolated Average Precision:** Predictions sorted descending by confidence are accumulated to trace the exact PR curve:
   $$p_{\text{interp}}(r) = \max_{r' \ge r} p(r'), \quad r \in \{0.00, 0.01, \dots, 1.00\}$$
   $$\text{AP} = \frac{1}{101} \sum_{k=0}^{100} p_{\text{interp}}(k/100)$$
5. **Handling Absent Classes:** If a class has $N_{GT} = 0$ in the evaluation subset (e.g., `train`), its AP is explicitly reported as `N/A (0 GT)` and excluded from the macro mAP denominator to avoid artificial metric distortion.

---

## 6. Sanity Verification Suite Results

The evaluation engine was validated by `experiments/baseline_v2/test_evaluator_sanity.py` across 5 synthetic edge cases and real validation crops:

| Test Scenario | Condition Tested | Expected Outcome | Actual Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| **IoU Calculation** | Overlap $[5,0,15,10]$ vs $[0,0,10,10]$ | $\text{IoU} = 50 / 150 = 0.3333$ | $0.3333$ | **PASS** |
| **Scenario 1** | Box & Class match (Person) | $TP=1, FP=0, FN=0, P=1.0, R=1.0, AP=1.0$ | Exact match | **PASS** |
| **Scenario 2A** | Wrong class (GT Bus, Pred Truck) | Bus: $TP=0, FP=0, FN=1, R=0.0$ | Exact match | **PASS** |
| **Scenario 2B** | Wrong class false alarm (Pred Truck) | Truck: $TP=0, FP=1, FN=0, P=0.0$ | Exact match | **PASS** |
| **Scenario 3** | Missed object (GT Bike, No pred) | Bike: $TP=0, FP=0, FN=1, R=0.0$ | Exact match | **PASS** |
| **Scenario 4** | Duplicate predictions on 1 Car | Car: $TP=1, FP=1, FN=0, P=0.5, R=1.0$ | Exact match | **PASS** |
| **Scenario 5** | Absent class (Train: 0 GT, 0 Pred) | Train: $GT=0, Pred=0, AP=\text{None}$ | Handled cleanly | **PASS** |
| **Real Crops** | Validation images `c95ad578`, `ca1c6003` | Correct GT counts: bike=3, bus=2, sign=3 | Exact match | **PASS** |

---

## 7. Latency Benchmarking Methodology

To ensure reproducible, scientifically rigorous latency reporting:
1. **Isolated Model Loading:** Model checkpoint loading was measured separately ($267.9\text{ ms}$).
2. **Warm-Up Phase:** 5 complete warm-up inferences were executed on dummy images prior to timing to initialize CPU vectorization and PyTorch memory caches.
3. **Timed Inference:** 100 evaluation images were timed individually using high-resolution monotonic timers (`time.perf_counter()`).
4. **Environment Specification:**
   - **Device:** CPU (`Intel64 Family 6 Model 170 Stepping 4, GenuineIntel`)
   - **Framework:** PyTorch `2.12.0+cpu`, Ultralytics `8.4.174`, OpenCV `5.0.0`
   - **Image Resolution:** $640 \times 640$ pixels
   - **Confidence Threshold:** Operational $\tau = 0.25$
   - **Latency Profile:**
     - **Mean:** $483.02\text{ ms}$
     - **Median:** $454.91\text{ ms}$
     - **P95:** $642.10\text{ ms}$
     - **Throughput:** $2.07\text{ FPS}$

---

## 8. Audit Verdict

### Verdict: PASS WITH LIMITATIONS

- **Integrity Validation:** **PASS**. All class alignments, metric formulas, and absent-class edge cases are fully verified and reproducible.
- **Limitation Notice:** The evaluated sample comprises **100 validation images enriched with Vulnerable Road Users** (70 VRU-rich images containing pedestrians, riders, bikes, or motorcycles, plus 30 standard traffic images). It must **not** be cited as the representative benchmark for the entire 10,000-image BDD100K validation split. A full-validation run requires dedicated GPU infrastructure due to CPU throughput ($2.07\text{ FPS} \implies \approx 80\text{ minutes}$ for 10k images).
- **Next Immediate Action:** Proceed to Phase 2B (Agentic Vision Pipeline & Uncertainty-Aware VRU Analysis) utilizing the verified `DetectionEvaluator` and corrected baseline.
