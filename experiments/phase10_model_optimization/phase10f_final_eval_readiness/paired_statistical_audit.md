# RoadSentinel Phase 10F — Paired Image-Level Statistical Audit Report

## 1. Executive Summary

This statistical audit computes **paired, image-level bootstrap distributions** comparing **Config E** (Adaptive VRU 0.22, VRU NMS IoU 0.55) against the **Baseline** (Flat 0.25, IoU 0.65) on **Cohort 5 (Untouched Confirmation Cohort, 1,000 images)**.

By evaluating both policies on identical resampled image indices in each bootstrap iteration, image-level clustering effects and inter-image variance are correctly accounted for, avoiding invalid independent-sample assumptions.

---

## 2. Statistical Methodology & Rigor

- **Evaluation Dataset:** Cohort 5 ($N = 1,000$ validation images, $18,548$ total GT objects, $1,300$ VRUs, $628$ Small VRUs).
- **Bootstrap Replicates:** $B = 2,000$ resamples with replacement.
- **Random Number Generator Seed:** Fixed seed `42` (`numpy.random.RandomState(42)`).
- **Sampling Unit:** Independent scene image (clustered sampling across all objects within each sampled image).
- **Interval Estimation Method:** Empirical percentile bootstrap ($2.5\text{th}$ and $97.5\text{th}$ percentiles for two-sided $95\%$ confidence intervals).
- **Pairing Guarantee:** In every bootstrap replicate $b \in \{1, \dots, 2000\}$, the identical vector of resampled image indices is provided to both Config E and the Baseline:
  $$\Delta^{(b)} = \text{Metric}_{\text{Config\_E}}^{(b)} - \text{Metric}_{\text{Baseline}}^{(b)}$$

---

## 3. Paired Differences and 95% Bootstrap Confidence Intervals

| Metric | Baseline Point Estimate | Config E Point Estimate | Paired Delta ($\Delta$) Point Estimate | Paired 95% Bootstrap Confidence Interval | Bootstrap Probability of Improvement $P(\Delta > 0)$ | Statistical Interpretation |
|---|---|---|---|---|---|---|
| **Small-VRU Recall** | 45.22% (284/628) | 49.20% (309/628) | **+3.98 pp** | **[+2.37, +5.71] pp** | **100.0%** (2000/2000) | **Statistically Significant Gain** |
| **Overall VRU Recall** | 63.62% (827/1300) | 66.08% (859/1300) | **+2.46 pp** | **[+1.60, +3.34] pp** | **100.0%** (2000/2000) | **Statistically Significant Gain** |
| **VRU Precision** | 77.58% (827/1066) | 77.04% (859/1115) | **-0.54 pp** | **[-1.66, +0.59] pp** | 16.4% | **Statistically Indistinguishable** |
| **Overall VRU F1** | 0.6991 | 0.7114 | **+0.0123** | **[+0.0051, +0.0195]** | **99.9%** (1998/2000) | **Statistically Significant Gain** |
| **Mean FP / Image** | 0.239 FP/img | 0.256 FP/img | **+0.017 FP/img** | **[+0.001, +0.032] FP/img** | 98.2% | Bounded & Minimal (+17 FPs total) |

### Key Findings:
1. **Small-VRU Recall Gain is Decisively Positive:**
   The entire 95% confidence interval for the small-VRU recall difference is bounded strictly away from zero ($[+2.37\text{ pp}, +5.71\text{ pp}]$). In all 2,000 bootstrap resamples, Config E outperformed the baseline ($P(\Delta > 0) = 100.0\%$).
2. **Precision Penalty is Statistically Indistinguishable from Zero:**
   The paired 95% CI for the precision delta spans $[-1.66\text{ pp}, +0.59\text{ pp}]$ and crosses zero. The small nominal drop of $-0.54\text{ pp}$ is within sampling noise, confirming that tightening VRU NMS IoU to $0.55$ successfully neutralized the precision penalty of lower thresholding.
3. **Net F1 Score Improves Reliably:**
   VRU F1 score increases by $+0.0123$, with a strictly positive $95\%$ CI of $[+0.0051, +0.0195]$ ($P(\Delta > 0) = 99.9\%$).

---

## 4. Audit of the 72.0% Precision Floor

The project's operating objective requires maintaining VRU precision $\ge 72.0\%$:

- **Config E Point Estimate:** **$77.04\%$** ($+5.04\text{ pp}$ safety margin above the $72.0\%$ floor).
- **Config E 95% Absolute Confidence Interval:** **$[73.20\%, 80.68\%]$**.
- **Compliance Rate Across Bootstrap Distribution:** **$99.45\%$** of all 2,000 resamples achieved precision $\ge 72.0\%$.
- **Adverse Tail Bound:** The minimum precision observed across any of the 2,000 bootstrap replicates was **$70.67\%$** (still above the historical $70.0\%$ project floor even in worst-case adverse resampling).

---

## 5. Critical Engineering & Safety Qualifications

> [!CAUTION]
> While the paired bootstrap establishes statistical stability on the BDD100K validation distribution, **clearing the 72.0% precision gate does not constitute formal proof of safety or robustness under operational distribution shift**.

Specifically:
1. **Geographic and Weather Generalization:** All validation images originate from BDD100K driving sequences. Heavy precipitation, winter conditions, novel sensor optical characteristics, or foreign road geometries may exhibit different false-positive distributions.
2. **Absolute Recall Deficit:** Even with Config E, small-VRU recall remains at $49.20\%$. Roughly half of small, distant vulnerable road users remain undetected in single-frame detection without temporal tracking.
