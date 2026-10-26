# RoadSentinel Phase 11 — Official BDD100K Test Split Statistical Report

**Evaluation Split:** Official BDD100K Test Split (`bdd100k_dataset/test/`)  
**Sample Cohort:** Deterministic 2,000 images (`cabc30fc-e7726578.jpg` to `cf8fb4a6-36c46511.jpg`)  
**Evaluation Mode:** One-Time Blind Evaluation (Zero model selection or post-hoc threshold tuning)  
**Evaluator:** Greedy IoU matching ($\text{IoU} \ge 0.50$), native $1280 \times 720$ coordinate projection  

---

## 1. Executive Summary & Core Results

The one-time blind final evaluation of **Config E** (Adaptive confidence threshold $\tau=0.22$ for VRUs / $0.25$ others; NMS IoU $0.55$ for VRUs / $0.65$ others) against the **Controlled Baseline Policy** (Flat $\tau=0.25$, NMS IoU $0.65$) was executed across a deterministic 2,000-image cohort of the official BDD100K test split.

The test cohort contains **37,173 total ground-truth objects**, including **2,813 VRUs** (1,237 Small, 1,362 Medium, 214 Large).

### Official Test Split Performance Summary

| Metric | Controlled Baseline | Candidate Policy (Config E) | Paired Difference ($\Delta$) | 95% Paired Bootstrap CI | Bootstrap $P(\Delta > 0)$ |
|---|---|---|---|---|---|
| **Small-VRU Recall** | 45.51% (563/1237) | **47.29%** (585/1237) | **+1.78 pp** | **[+0.99, +2.62] pp** | **100.0%** |
| **Overall VRU Recall** | 65.45% (1841/2813) | **66.65%** (1875/2813) | **+1.21 pp** | **[+0.72, +1.75] pp** | **100.0%** |
| **VRU Precision** | 77.94% (1841/2362) | **77.10%** (1875/2432) | **−0.85 pp** | **[−1.60, −0.13] pp** | — |
| **Overall VRU F1** | 0.7115 | **0.7150** | **+0.0035** | **[−0.0010, +0.0080]** | **93.5%** |
| **False Positives / Image** | 0.261 | **0.279** | **+0.018** | **[+0.007, +0.029]** | — |
| **Forward Passes / Frame** | 1.00 | **1.00** | **0.00** | [1.00, 1.00] | — |

---

## 2. Integrity & Quota Audit

Prior to loading test data, cryptographic SHA-256 digests were verified:

1. **Champion Model Checkpoint (`session4_best.pt`):**
   - SHA-256: `4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd` (**VERIFIED**)
2. **Phase 6 Reference Baseline Checkpoint (`best.pt`):**
   - SHA-256: `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` (**VERIFIED**)
3. **Quarantine Invariant:** The test images and labels were accessed strictly once in read-only batch mode. No thresholds, weights, or algorithms were altered after inference started.

---

## 3. Paired Image-Level Bootstrap Analysis ($B = 2,000$, Seed 42)

To account for image-level scene clustering and avoid the error of treating bounding boxes as independent observations, 2,000 paired image-level resamples were computed on the exact same 2,000 test images:

1. **Small-VRU Recall Difference ($\Delta$):**
   - Point Estimate: **+1.78 pp**
   - Empirical 95% CI: **[+0.99, +2.62] percentage points**
   - Probability of Positive Gain: **100.0%** (2,000 out of 2,000 replicates exhibited positive delta)
   - *Interpretation:* The detection improvement for small VRUs is genuine, reproducible, and non-zero across unvisited test scenes. However, the magnitude (+1.78 pp) contracted from the validation cohort (+3.98 pp).

2. **Overall VRU Recall Difference ($\Delta$):**
   - Point Estimate: **+1.21 pp**
   - Empirical 95% CI: **[+0.72, +1.75] percentage points**
   - Probability of Positive Gain: **100.0%**

3. **VRU Precision Safety:**
   - Point Estimate: **77.10%** (Baseline 77.94%)
   - Empirical 95% CI: **[75.13%, 79.03%]**
   - Safety Margin: The lower bound ($75.13\%$) remains **+5.13 pp above the 70.0% minimum operating bound**, and the point estimate has a **+5.10 pp buffer above the 72.0% floor**. 100% of bootstrap resamples exceeded 72.0%.

4. **VRU F1 Difference ($\Delta$):**
   - Point Estimate: **+0.0035**
   - Empirical 95% CI: **[−0.0010, +0.0080]**
   - Probability of Positive Gain: **93.5%** (1,870 out of 2,000 replicates exhibited positive delta)
   - *Interpretation:* While the net balance of true positives vs false alarms tilts positive, the confidence interval slightly crosses zero at the lower 2.5th percentile.

5. **VRU False Alarm Trade-off:**
   - Baseline VRU FP rate: 0.261 VRU FP/image (521 total VRU false alarms across 2,000 images).
   - Config E VRU FP rate: 0.279 VRU FP/image (557 total VRU false alarms across 2,000 images).
   - Net increase: **+0.018 VRU FP/image** (+36 VRU false alarms total, or 1 extra false alarm every 55 frames).
   - Net VRU True Positives gained: **+34 VRUs** (including +22 small VRUs).
   - Exchange ratio: **1.06 VRU false alarms per additional VRU recovered**.
   - *(Note: All-class detector false alarms were 7,740 for Baseline and 7,776 for Config E; the net +36 false alarms across the entire detector were 100% localized to VRU classes).*

---

## 4. Size-Stratified VRU Performance

| Object Size Category (COCO Area) | Ground Truth Objects | Baseline Recall | Config E Recall | Delta ($\Delta$) | True Positives (Base $\to$ Config E) |
|---|---|---|---|---|---|
| **Small (<32² px)** | 1,237 | 45.51% | **47.29%** | **+1.78 pp** | 563 $\to$ 585 (+22) |
| **Medium [32²–96² px)** | 1,362 | 79.52% | **80.47%** | **+0.95 pp** | 1,083 $\to$ 1,096 (+13) |
| **Large (≥96² px)** | 214 | 91.12% | **90.65%** | **−0.47 pp** | 195 $\to$ 194 (−1) |

- **Small VRUs:** Demonstrated the highest relative recovery (+22 true positives, representing 65% of all gained VRU detections).
- **Medium VRUs:** Also benefited positively (+13 true positives).
- **Large VRUs:** Experienced a single true positive suppression (195 $\to$ 194) attributable to the tighter NMS IoU threshold ($0.55$) in dense occluded groupings.

---

## 5. Per-Class Performance Breakdown

| Class Name | Supercategory | Baseline Precision | Config E Precision | Baseline Recall | Config E Recall | Baseline F1 | Config E F1 | TP Gain | FP Gain |
|---|---|---|---|---|---|---|---|---|---|
| **`person`** | VRU | 79.13% | **78.26%** | 66.91% | **68.26%** | 0.7251 | **0.7292** | +33 | +32 |
| **`rider`** | VRU | 72.84% | **73.49%** | 51.75% | **53.51%** | 0.6051 | **0.6193** | +2 | 0 |
| **`motor`** | VRU | 71.19% | **69.84%** | 50.60% | **53.01%** | 0.5915 | **0.6027** | +2 | +2 |
| **`bike`** | VRU | 67.52% | **66.03%** | 60.92% | **59.20%** | 0.6405 | **0.6243** | −3 | +2 |
| `car` | Vehicle | 81.80% | 81.80% | 78.11% | 78.11% | 0.7991 | 0.7991 | 0 | 0 |
| `bus` | Vehicle | 63.95% | 63.95% | 57.67% | 57.67% | 0.6065 | 0.6065 | 0 | 0 |
| `truck` | Vehicle | 69.80% | 69.80% | 63.65% | 63.65% | 0.6658 | 0.6658 | 0 | 0 |
| `traffic light` | Signage | 72.52% | 72.52% | 69.75% | 69.75% | 0.7111 | 0.7111 | 0 | 0 |
| `traffic sign` | Signage | 72.98% | 72.98% | 71.61% | 71.61% | 0.7229 | 0.7229 | 0 | 0 |
| `train` | Vehicle | 0.00% | 0.00% | 0.00% | 0.00% | 0.0000 | 0.0000 | 0 | 0 |

### Observations:
- **`person`:** The dominant VRU class accounted for +33 additional true positives with an F1 gain of $+0.0041$.
- **`rider` & `motor`:** Both demonstrated substantial recall increases (+1.76 pp and +2.41 pp) with net F1 gains.
- **`bike` Regression:** Bicycles exhibited a minor regression (-3 TP, 106 $\to$ 103, recall $-1.72\text{ pp}$). This indicates that bicycle ground truths with low aspect-ratio or overlapping bicycle racks were slightly over-suppressed by the tighter $0.55$ NMS threshold.
- **Non-VRU Invariance:** All 6 non-VRU categories exhibited mathematically identical performance, confirming zero cross-class side effects.

---

## 6. Predeclared Acceptance Gates Evaluation

| Gate Identifier | Pre-Declared Requirement | Measured Value on Official Test Split | Status |
|---|---|---|---|
| **Gate 1: Small-VRU Recall Gain** | $\Delta \ge +2.50\text{ pp}$ and $P(\Delta > 0) \ge 99.0\%$ | $\Delta = +1.78\text{ pp}$, $P(\Delta > 0) = 100.0\%$ | **FAIL** (Effect size $+1.78\text{ pp} < +2.50\text{ pp}$) |
| **Gate 2: Precision Safety Floor** | Config E Precision $\ge 72.0\%$, CI lower bound $\ge 70.0\%$ | Precision = $77.10\%$, CI lower bound = $75.13\%$ | **PASS** (+5.10 pp above floor) |
| **Gate 3: Net F1 Superiority** | $\Delta \text{Overall VRU F1} > 0.000$ and $P(\Delta > 0) \ge 95.0\%$ | $\Delta \text{F1} = +0.0035$, $P(\Delta > 0) = 93.5\%$ | **FAIL** ($93.5\% < 95.0\%$ requirement) |
| **Gate 4: Computational Cycle** | Exactly 1 forward pass per frame ($O(1)$ deterministic) | Exactly 1.00 forward pass per frame | **PASS** |

### Overall Gate Outcome: **FAIL (2 of 4 Gates Passed)**
Although Config E demonstrates statistically verified small-VRU recall gains ($P=100\%$) and exceeds precision safety standards, the magnitude of the recall improvement on unvisited test scenes did not reach the ambitious $+2.50\text{ pp}$ contract hurdle, and F1 statistical power fell just short of the 95% threshold.
