# RoadSentinel Phase 10E — False-Positive Analysis and Ablation Report

## 1. Executive Summary

Phase 10E executed a controlled, pre-locked evaluation to resolve the precision-headroom tension identified in Phase 10D. Under the locked **Balanced / Higher Precision Floor ($\ge 72.0\%$)**, 2,000 fresh, previously unvisited validation images were evaluated across two independent stages:
- **Cohort 4 (Confirmation Cohort, 1,000 images):** Used for Stage A (Threshold Confirmation) and Stage B (NMS Duplicate Suppression Ablation).
- **Cohort 5 (Untouched Confirmation Cohort, 1,000 images):** Used for Stage D (Independent Policy Validation).

Both cohorts are strictly pairwise disjoint from each other and from all 1,600 images evaluated in prior phases.

---

## 2. Stage A — Threshold Effect on Cohort 4 (1,000 Images)

In Stage A, the baseline and two candidate adaptive thresholds were evaluated under identical NMS ($\text{IoU} = 0.65$):

| Policy Candidate | VRU Confidence Threshold | Small-VRU Recall (95% CI) | Overall VRU Recall (95% CI) | VRU Precision (95% CI) | VRU F1 | Total VRU TP | Total VRU FP | Mean FP / Img | $\ge 72.0\%$ Gate |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline (Flat 0.25)** | 0.25 | 45.47% [40.40, 50.45%] | 65.02% [61.89, 68.44%] | **75.70%** [72.98, 78.34%] | 0.6995 | 1,050 | **337** | 0.337 | **PASS** |
| **Adaptive VRU 0.22** | 0.22 | **48.54%** [43.40, 53.62%] | **67.18%** [64.11, 70.66%] | **72.24%** [69.46, 74.81%] | **0.6962** | 1,085 (+35) | **417** (+80) | 0.417 | **PASS** |
| **Adaptive VRU 0.20** | 0.20 | 50.23% [45.40, 55.22%] | 68.36% [65.43, 71.79%] | **69.83%** [67.21, 72.44%] | 0.6909 | 1,104 (+54) | **477** (+140)| 0.477 | **FAIL (<72%)** |

### Crucial Finding from Stage A:
The Phase 10D assessment's warning was empirically vindicated:
On this fresh 1,000-image cohort, **Adaptive 0.20 fell to $69.83\%$ precision**, breaching both the locked $72.0\%$ floor and the project's historical $70.0\%$ line.
In contrast, **Adaptive 0.22 successfully cleared the $72.0\%$ precision gate ($72.24\%$)** while capturing a $+3.07\text{ pp}$ small-VRU recall improvement ($45.47\% \to 48.54\%$, $+20$ small VRU TPs).

---

## 3. Stage B — Duplicate Suppression NMS Ablation on Cohort 4

Phase 10D identified that $21.4\%$ of added false alarms were duplicate or loose secondary bounding boxes on already detected pedestrians. Stage B tested whether surgical per-class NMS tightening on VRUs could eliminate duplicates without suppressing nearby pedestrians:

| Policy | VRU Conf | VRU NMS IoU | VRU TP | VRU FP | Small-VRU Recall | VRU Recall | VRU Precision | VRU F1 | TPs Lost vs 0.65 | FPs Eliminated vs 0.65 | Net Result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Adaptive 0.22** | 0.22 | **0.65** | 1,085 | 417 | 48.54% | 67.18% | 72.24% | 0.6962 | 0 | 0 | Baseline |
| **Adaptive 0.22** | 0.22 | **0.60** | 1,083 | 393 | 48.39% | 67.06% | 73.37% | 0.7007 | -2 | **+24** | Win |
| **Adaptive 0.22** | 0.22 | **0.55** | 1,082 | 381 | **48.39%** | **67.00%** | **73.96%** | **0.7031** | -3 | **+36** | **Optimal Win** |
| Adaptive 0.20 | 0.20 | 0.65 | 1,104 | 477 | 50.23% | 68.36% | 69.83% | 0.6909 | 0 | 0 | Fails 72% |
| Adaptive 0.20 | 0.20 | 0.60 | 1,102 | 448 | 50.08% | 68.24% | 71.10% | 0.6964 | -2 | +29 | Fails 72% |
| Adaptive 0.20 | 0.20 | 0.55 | 1,099 | 429 | 49.77% | 68.05% | 71.92% | 0.6993 | -5 | +48 | Fails 72% |

### Engineering Insight from Stage B:
Tightening VRU NMS IoU from $0.65$ to $0.55$ eliminated **36 duplicate false-positive bounding boxes** while losing only **3 true positives** (a 12:1 beneficial ratio).
This lifted the precision of Adaptive 0.22 from **$72.24\% \to 73.96\%$** ($+1.72\text{ pp}$ safety headroom) and boosted overall VRU F1 to **$0.7031$**, with essentially zero sacrifice in small-VRU recall ($48.54\% \to 48.39\%$).

---

## 4. Stage D — Final Independent Confirmation on Untouched Cohort 5 (1,000 Images)

The finalized policy (**Adaptive VRU $\tau = 0.22$, VRU NMS $\text{IoU} = 0.55$**) was evaluated on the completely untouched Cohort 5 ($1,000$ images, $18,548$ GT objects, $1,300$ VRUs, $628$ Small VRUs) alongside the Baseline and the 0.20 reference:

| Policy Configuration | Small-VRU Recall (95% CI) | Overall VRU Recall (95% CI) | VRU Precision (95% CI) | VRU F1 | VRU TP | VRU FP | VRU FN | Mean FP / Img | $\ge 72.0\%$ Gate |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline (Flat 0.25, IoU 0.65)** | 45.22% [40.18, 50.16%] | 63.62% [59.66, 67.20%] | **77.58%** [73.49, 81.24%] | 0.6991 | 827 | **239** | 473 | 0.239 | **PASS** |
| **Selected Policy ($\tau=0.22$, IoU 0.55)** | **49.20%** [44.14, 54.14%] | **66.08%** [62.18, 69.43%] | **77.04%** [73.15, 80.69%] | **0.7114** | **859** (+32) | **256** (+17) | 441 | **0.256** | **PASS (+5.04 pp margin)** |
| **Adaptive 0.20 Reference ($\tau=0.20$, IoU 0.65)**| 51.59% [46.29, 56.61%] | 67.46% [63.50, 70.90%] | 72.24% [68.30, 76.17%] | 0.6977 | 877 (+50) | 337 (+98) | 423 | 0.337 | PASS (Narrow, CI dips to 68.3%) |

### Per-Class Metric Confirmation on Cohort 5:
- **Person:** Precision maintained at **$77.29\%$** (vs $77.61\%$ baseline), while Recall increased by **$+2.74\text{ pp}$** ($64.31\% \to 67.05\%$, $+31$ persons recovered). Person F1 rose to **$0.7181$** (highest in project).
- **Rider:** Precision $69.44\%$, Recall $54.35\%$, F1 $0.6098$.
- **Bike:** Precision $77.33\%$, Recall $71.60\%$, F1 $0.7435$.
- **Motor:** Precision $77.27\%$, Recall $41.46\%$, F1 $0.5396$.
- **Vehicles & Signage (Non-VRU):** Precision $82.14\%$, Recall $78.83\%$, F1 $0.8045$ (100% bit-for-bit preserved).

---

## 5. Statistical Rigor: Clustered Bootstrap Confidence Intervals

Because road traffic objects are clustered within scenes, independent sample assumptions fail. The 1,000-iteration image-level bootstrap reveals:
- **Baseline 0.25:** VRU Precision 95% CI is $[73.49\%, 81.24\%]$.
- **Selected Policy (0.22 + 0.55):** VRU Precision 95% CI is **$[73.15\%, 80.69\%]$**. Crucially, the entire $95\%$ confidence interval lies strictly above the $72.0\%$ precision requirement, with a lower bound of $73.15\%$.
- **Adaptive 0.20 Policy:** VRU Precision 95% CI is $[68.30\%, 76.17\%]$. The lower tail extends to $68.30\%$, demonstrating an unacceptable risk of violating production precision limits under real-world distribution variance.
