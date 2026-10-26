# RoadSentinel Phase 4A — Metric, Acceptance & Selection Policy

**Date:** October 10, 2026  
**Auditor:** Principal Computer Vision & Safety Evaluation Specialist  
**Status:** COMPLETE — Authoritative Multi-Metric Evaluation Standard  
**Purpose:** Prevent single-metric gaming, enforce mathematical consistency, and provide a multi-dimensional selection policy for autonomous road user detection.

---

## 1. Core Evaluation Principles

1. **Anti-Gaming Principle:** No candidate system will be approved based on a single metric. Improving Small VRU Recall by flooding the scene with low-confidence detections and exploding False Positives is strictly prohibited.
2. **Deterministic Paired Evaluation:** All candidate comparisons must be evaluated against the exact same $10,000$ validation image manifest in identical sequential order using deterministic random seeds.
3. **Explicit Operating Points:** Ranking metrics (COCO 101-point mAP@50 and mAP@50–95) must always be reported alongside operational point metrics (Precision, Recall, F1, TP, FP, FN) evaluated at declared operational thresholds.
4. **Conservation Invariants:** For any valid evaluation run, size-group counts and ground-truth totals must satisfy:
   $$\text{Small TP} + \text{Medium TP} + \text{Large TP} = \text{Total VRU TP}$$
   $$\text{Total VRU TP} + \text{Total VRU FN} = 15,373$$

---

## 2. Complete Metric Hierarchy

### 2.1 Primary Detection Metrics
- **Overall mAP@50:** COCO-style 101-point mean Average Precision across all 10 semantic classes at $\text{IoU} = 0.50$.
- **Overall mAP@50–95:** Mean Average Precision averaged over 10 IoU thresholds $[0.50, 0.55, \dots, 0.95]$.
- **VRU Macro mAP@50:** Unweighted arithmetic mean of Average Precision across the four VRU classes:
  $$\text{mAP}_{\text{VRU}} = \frac{\text{AP}_{\text{person}} + \text{AP}_{\text{rider}} + \text{AP}_{\text{bike}} + \text{AP}_{\text{motor}}}{4}$$
- **VRU Macro mAP@50–95:** Unweighted arithmetic mean of AP@50–95 across the four VRU classes.

### 2.2 Operational Point Metrics (Evaluated at $\tau_{\text{op}}$ and $\text{IoU} = 0.50$)
- **VRU Precision:** $\frac{\text{VRU TP}}{\text{VRU TP} + \text{VRU FP}}$
- **VRU Recall:** $\frac{\text{VRU TP}}{15,373}$
- **VRU F1-Score:** $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Small-VRU Recall:** $\frac{\text{Small VRU TP}}{7,029}$ (where bounding box area $< 1,024\text{ px}^2$)
- **Medium-VRU Recall:** $\frac{\text{Medium VRU TP}}{7,246}$ (where $1,024\text{ px}^2 \le \text{area} < 9,216\text{ px}^2$)
- **Large-VRU Recall:** $\frac{\text{Large VRU TP}}{1,098}$ (where $\text{area} \ge 9,216\text{ px}^2$)
- **False Positive Count ($\text{FP}$):** Total unassigned detections at $\tau_{\text{op}}$.
- **False Negative Count ($\text{FN}$):** Total missed ground-truth VRUs ($15,373 - \text{TP}$).

### 2.3 Runtime, Efficiency & Resource Metrics
- **Throughput (FPS):** Total evaluated validation images ($10,000$) divided by total continuous wall-clock evaluation loop time (seconds) on specified GPU hardware.
- **Batch-1 Pipeline Latency P50 (ms):** Median end-to-end single-frame processing time (disk read $\rightarrow$ preprocessing $\rightarrow$ full frame $\rightarrow$ crops $\rightarrow$ fusion $\rightarrow$ outputs) on isolated streaming frames at $\tau_{\text{op}}$.
- **Batch-1 Pipeline Latency P95 (ms):** 95th percentile pipeline latency representing worst-case frame delay.
- **Peak VRAM Allocated (MB):** Peak GPU memory allocated to PyTorch tensors (`torch.cuda.max_memory_allocated`), measured at batch size 1.
- **Crop Trigger Rate ($\%$):** Fraction of validation images that executed at least one crop pass.
- **Total Crops Executed:** Total number of crop forward passes across the $10,000$ validation images.

---

## 3. Operational Operating Regimes

RoadSentinel defines two formal operating regimes:

### Regime A: Primary Deployment Operating Point (Balanced Precision & Recall)
- **Primary Threshold:** $\tau_{\text{op}} = 0.25$ (or optimized class-specific thresholds $\vec{\tau}_c \approx [0.20, 0.30]$).
- **Matching IoU:** $0.50$.
- **Design Target:** Real-time in-vehicle perception where false alarms must be controlled to prevent phantom braking or alert fatigue.
- **Precision Floor:** $\text{VRU Precision} \ge 70.0\%$.

### Regime B: High-Recall Safety-Enquiry Point (Audit & Pre-Collision Warning)
- **Secondary Threshold:** $\tau_{\text{safety}} = 0.15$.
- **Design Target:** Downstream risk assessment and safety envelope monitoring where missing a human has catastrophic cost, and downstream trackers or temporal models filter transient false alarms.
- **Recall Target:** $\text{Small VRU Recall} \ge 52.0\%$, $\text{Overall VRU Recall} \ge 68.0\%$.

---

## 4. Multi-Metric Gate & Acceptance Criteria

To be accepted into the candidate pool for final model selection, any proposed optimization must clear the following **Hard Gates**:

```
                                  CANDIDATE OPTIMIZATION
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Gate 1: Denominator Check │
                               │ Total GT == 15,373        │
                               │ Tier Sum == Total TP      │
                               └─────────────┬─────────────┘
                                             │ PASS
                                             ▼
                               ┌───────────────────────────┐
                               │ Gate 2: Precision Floor   │
                               │ VRU Precision >= 68.0%    │
                               │ (Baseline was 62.03%)     │
                               └─────────────┬─────────────┘
                                             │ PASS
                                             ▼
                               ┌───────────────────────────┐
                               │ Gate 3: Recall Non-Degrad │
                               │ Small Recall >= 45.0%     │
                               │ Total VRU Recall >= 61.5% │
                               └─────────────┬─────────────┘
                                             │ PASS
                                             ▼
                               ┌───────────────────────────┐
                               │ Gate 4: Latency Envelope  │
                               │ Pipeline P50 <= 25.0 ms   │
                               │ Pipeline P95 <= 42.0 ms   │
                               │ Throughput >= 25.0 FPS    │
                               └─────────────┬─────────────┘
                                             │ PASS
                                             ▼
                                     ACCEPTED CANDIDATE
```

### Quantitative Acceptance Thresholds:

| Metric | Phase 3.4 Baseline (Active Foveation) | Minimum Required Gate | Target Optimization Goal |
|---|---:|---:|---:|
| **Overall mAP@50** | 0.5403 | $\ge 0.5450$ | **$\ge 0.5600$** |
| **VRU Macro mAP@50** | 0.5120 | $\ge 0.5180$ | **$\ge 0.5350$** |
| **VRU Precision ($\tau_{\text{op}}$)** | 0.6203 (62.03%) | **$\ge 0.6800$ (68.0%)** | **$\ge 0.7200$ (72.0%)** |
| **VRU Recall ($\tau_{\text{op}}$)** | 0.6271 (62.71%) | $\ge 0.6150$ (61.5%) | **$\ge 0.6400$ (64.0%)** |
| **VRU F1-Score ($\tau_{\text{op}}$)** | 0.6237 | **$\ge 0.6500$** | **$\ge 0.6750$** |
| **Small VRU Recall** | 0.4683 (46.83%) | **$\ge 0.4500$ (45.0%)** | **$\ge 0.4850$ (48.5%)** |
| **Batch-1 Pipeline P50 Latency** | 18.56 ms | $\le 25.0\text{ ms}$ | **$\le 15.0\text{ ms}$** |
| **Batch-1 Pipeline P95 Latency** | 39.72 ms | $\le 42.0\text{ ms}$ | **$\le 28.0\text{ ms}$** |
| **Throughput (FPS)** | 27.98 FPS | $\ge 25.0\text{ FPS}$ | **$\ge 35.0\text{ FPS}$** |
| **Crop Compute Savings vs FixedGrid** | 61.67% | $\ge 55.0\%$ | **$\ge 65.0\%$** |

---

## 5. System Role Distinctions in Final Reporting

RoadSentinel explicitly distinguishes between four distinct system configurations in all final reporting:

1. **Best Standalone Detector:**
   - *Definition:* The single full-frame model operating without any cropping, tiling, or agentic proposals.
   - *Current Baseline:* `FullFrame-960` (or `FullFrame-1280`).
   - *Role:* Baseline reference establishing the ceiling of conventional passive detection.
2. **Best Active Foveation System:**
   - *Definition:* The highest-accuracy uncertainty-guided foveation system, prioritizing Small VRU Recall and Macro VRU AP while satisfying the minimum precision floor.
   - *Role:* Primary research contribution demonstrating the power of active visual enquiry.
3. **Best Accuracy / Compute Trade-Off System:**
   - *Definition:* The configuration maximizing the Efficiency Ratio:
     $$\text{Efficiency Score} = \frac{\text{VRU Macro F1} \times \text{Small VRU Recall}}{\text{Amortized Latency (ms)} / 10.0}$$
   - *Role:* Demonstrates superior compute efficiency over uniform grid slicing.
4. **Best Deployable Real-Time System:**
   - *Definition:* The configuration meeting strict embedded automotive constraints:
     - Streaming latency P95 $\le 33.3\text{ ms}$ ($30\text{ FPS}$ hard deadline)
     - Peak VRAM $\le 500\text{ MB}$
     - Precision $\ge 72.0\%$
   - *Role:* Production-ready edge pipeline for demonstrator applications.

---

## 6. Statistical Significance & Paired Comparison Protocol

To ensure that reported improvements are statistically robust and not random fluctuations:
1. **Paired McNemar's Test for Detection Binary Outcomes:**
   For any paired comparison between Baseline ($A$) and Candidate ($B$) across the $15,373$ VRU ground-truth objects:
   $$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$$
   where $n_{01}$ is VRUs missed by $A$ but detected by $B$, and $n_{10}$ is VRUs detected by $A$ but missed by $B$. A gain is declared significant only if $p < 0.01$.
2. **Bootstrap Confidence Intervals for Precision, Recall, and F1:**
   Compute $95\%$ bootstrap confidence intervals using $1,000$ resamples with replacement over the validation image manifest.
