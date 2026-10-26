# RoadSentinel Phase 12 — Predeclared Validation and Statistical Protocol

**Status:** FROZEN & RATIFIED (Pre-Training Execution Rule)  
**Target Architecture:** YOLO11m-P2 (`experiments/phase12_p2_architecture/yolo11m-p2.yaml`)  
**Evaluation Baseline:** Frozen Baseline Checkpoint (`best.pt`, SHA-256: `01bd55ec8075942b12ec2d5f5b9f243d4b6ab43eb4a3aa0910946f3edcaf45d1`)  
**Designated Validation Cohort:** BDD100K Official Validation Cohort (Quarantined Test Cohort Strictly Preserved)  

---

## 1. Acceptance Gates & Performance Targets

The selected YOLO11m-P2 candidate checkpoint must satisfy all five acceptance gates simultaneously against the predeclared validation cohort to earn certification:

| Gate | Acceptance Criterion | Evaluation Standard | Failure Implication |
| :--- | :--- | :--- | :--- |
| **Gate 1: Small-VRU Recall** | $\ge 52.0\%$ | Point estimate on designated validation cohort | Candidate fails primary objective of Phase 12 |
| **Gate 2: VRU Precision** | $\ge 72.0\%$ (Point Estimate)<br>$\ge 70.0\%$ (95% CI Lower Bound) | Non-parametric bootstrap resampling ($B = 2,000$) | Safety contract violation (excessive false positives) |
| **Gate 3: Overall VRU F1** | $\ge 0.7250$ AND<br>$\Delta \text{F1} > 0$ vs Baseline with $P(\Delta > 0) \ge 95\%$ | Paired image-level bootstrap comparison | Failure to achieve statistically defensible gain |
| **Gate 4: Bike Recall** | $\ge 60.0\%$ | Point estimate on designated validation cohort | Specific VRU vulnerability unaddressed |
| **Gate 5: Compute & VRAM** | 1 forward pass per frame AND<br>Peak training VRAM $\le 14.0\text{ GB}$ | Architectural verification & memory profiling | Violation of edge deployment & GPU safety budgets |

> [!NOTE]
> The single-forward-pass criterion in Gate 5 specifies architectural efficiency (native end-to-end multi-scale detection without foveation cropping or secondary passes). Target-hardware TensorRT latency on Jetson Orin will be evaluated post-training and is formally tracked as **UNASSESSED** until hardware benchmarking occurs.

---

## 2. Statistical Methodology

### 2.1 Paired Image-Level Bootstrap Resampling
To ensure valid inference and correct standard errors, all statistical tests comparing the selected YOLO11m-P2 model against the frozen baseline will use **paired image-level bootstrap resampling**:

1. **Unit of Resampling:** Whole images (not individual bounding boxes), preserving intra-image spatial and class correlation structures.
2. **Replicate Count:** $B = 2,000$ independent resamples drawn with replacement.
3. **Fixed Random Seed:** `SEED = 42` recorded and hardcoded in the evaluation script to ensure exact bit-for-bit reproducibility.
4. **Paired Coupling:** For each bootstrap replicate $b \in \{1, \dots, B\}$, the identical set of resampled image indices is used to compute metrics for both the baseline model and the candidate model.
5. **Delta Computation:** For each replicate:
   $$\Delta \text{Metric}^{(b)} = \text{Metric}_{\text{P2}}^{(b)} - \text{Metric}_{\text{Baseline}}^{(b)}$$
6. **Confidence Intervals:** Empirical percentile method ($2.5^{\text{th}}$ percentile and $97.5^{\text{th}}$ percentile across $B=2,000$ replicates).
7. **Empirical Bootstrap Probability:**
   $$P(\Delta > 0) = \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}\left(\Delta \text{Metric}^{(b)} > 0\right)$$
   *Formal Rule:* A finite bootstrap probability of $100.0\%$ ($2,000 / 2,000$) represents strong sample evidence but is **NOT** interpreted as a theoretical guarantee of future improvement.

### 2.2 Boundary Case Handling
- **Images with Zero Ground-Truth Objects:**
  - Predictions on empty images are strictly evaluated as False Positives ($FP$).
  - They contribute to the total FP count and precision denominator.
- **Images with Zero Predictions:**
  - Ground truth objects on images with no predictions are strictly evaluated as False Negatives ($FN$).
  - They contribute to the total FN count and recall denominator.
- **Undefined Precision / Recall Edge Cases in Replicates:**
  - If a bootstrap replicate contains zero ground truth objects for a rare class, that replicate's class recall is omitted from the distribution with an explicit warning count logged. Overall VRU metrics always possess non-zero denominators.

---

## 3. Discrepancy Reconciliation

- **Prior Research Plan vs Validation Protocol:**
  The preliminary preflight draft referenced a candidate threshold of $50.0\%$ for small-VRU recall. The binding Phase 12 validation protocol ratifies the definitive target of **$\ge 52.0\%$**, directly addressing the Phase 11 gap where Config E achieved $47.29\%$ on the test cohort.
- **Strict Quarantine of Official Test Split:**
  Under no circumstances will the 20,000 official test images in `bdd100k_dataset/test/` be evaluated during this phase. If all five validation gates pass, the candidate model is declared **READY FOR OFFICIAL EVALUATION GOVERNANCE**, but test performance claims are strictly prohibited until a separately authorized evaluation is held.
