# RoadSentinel Phase 11 — Final Test Evaluation Decision Record

**Evaluation Date:** October 14, 2026  
**Artifact Directory:** `experiments/phase11_final_test_evaluation/`  
**Candidate Policy:** Config E (Adaptive VRU $\tau=0.22$, VRU NMS IoU $0.55$)  
**Baseline Policy:** Controlled Baseline (Flat $\tau=0.25$, NMS IoU $0.65$)  
**Model Checkpoint:** `session4_best.pt` (SHA-256: `4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd`)  
**Evaluation Cohort:** Deterministic 2,000-image official test sample (`cabc30fc-e7726578.jpg` to `cf8fb4a6-36c46511.jpg`)  

---

## 1. Final Decision Verdict

### **VERDICT: FORMAL GATES FAILED (2 of 4 Gates Passed)**

In strict adherence to the pre-declared protocol and governance invariants:
- **No silent re-tuning:** Thresholds and parameters will **not** be adjusted post-hoc to retroactively force a passing result.
- **No data contamination:** The remaining 18,000 images of the official BDD100K test split remain quarantined and untouched.
- **Transparent reporting:** The evaluation is documented as a gate failure against the pre-declared $+2.50\text{ pp}$ recall improvement threshold.

---

## 2. Gate-by-Gate Compliance Audit

| Gate | Criterion | Pre-Declared Standard | Measured Test Outcome | Result | Failure Analysis |
|---|---|---|---|---|---|
| **Gate 1** | Small-VRU Recall Gain | $\Delta \ge +2.50\text{ pp}$ and $P(\Delta > 0) \ge 99.0\%$ | $\mathbf{\Delta = +1.78\text{ pp}}$, $P(\Delta > 0) = \mathbf{100.0\%}$ (95% CI: $[+0.99, +2.62]\text{ pp}$) | **FAIL** | While the gain is statistically indisputable (100% of 2,000 bootstrap resamples showed a positive gain), the effect size contracted from $+3.98\text{ pp}$ on the validation cohort to $+1.78\text{ pp}$ on the test split, falling below the $+2.50\text{ pp}$ contractual hurdle. |
| **Gate 2** | Precision Safety Floor | Point estimate $\ge 72.0\%$, 95% CI lower bound $\ge 70.0\%$ | Point estimate = $\mathbf{77.10\%}$, 95% CI lower bound = $\mathbf{75.13\%}$ | **PASS** | Passed with a substantial safety margin (+5.10 pp above point floor; +5.13 pp above CI lower bound floor). |
| **Gate 3** | Net F1 Superiority | $\Delta \text{F1} > 0.000$ and $P(\Delta > 0) \ge 95.0\%$ | $\mathbf{\Delta \text{F1} = +0.0035}$, $P(\Delta > 0) = \mathbf{93.5\%}$ (95% CI: $[−0.0010, +0.0080]$) | **FAIL** | The point estimate of F1 improved from 0.7115 to 0.7150, but the bootstrap confidence interval slightly crossed zero ($[−0.0010, +0.0080]$), reaching $93.5\%$ probability of gain rather than the required $95.0\%$. |
| **Gate 4** | Deterministic Latency | Exactly 1.00 forward pass per frame | $\mathbf{1.00}$ passes ($960 \times 960$ letterbox) | **PASS** | Zero multi-crop overhead; deterministic $O(1)$ cycle time maintained. |

---

## 3. Root Cause Analysis: Why Did the Recall Gain Contract on Test Data?

Across validation experiments (Cohorts 4 & 5), Config E produced small-VRU recall gains between $+3.5\text{ pp}$ and $+4.0\text{ pp}$. On the official test split, the measured gain was $+1.78\text{ pp}$.

Three primary factors account for this delta:

1. **Test Split Scene Distribution:**
   - The official BDD100K test split contains a higher diversity of extreme-distance VRUs and adverse night/glare scenarios. For extreme-distance pedestrians where the model outputs confidence scores $<0.15$ or misses the feature representation entirely, lowering the threshold from $0.25$ to $0.22$ has zero effect.
2. **Post-Processing Ceilings:**
   - Threshold adjustments can only recover "near-threshold" false negatives (scores between $0.22$ and $0.25$). The $+1.78\text{ pp}$ gain recovered 22 small VRUs out of 1,237. The remaining 652 small VRUs missed by the model have confidence scores below $0.22$ or were not proposed as candidates by the network.
3. **Bicycle Subcategory Suppression:**
   - Tightening VRU NMS from $0.65$ to $0.55$ successfully suppressed duplicate pedestrian false alarms, but resulted in 3 lost bicycle true positives (106 $\to$ 103, recall drop of $-1.72\text{ pp}$). Bicycles frequently have elongated aspect ratios and overlapping geometry with riders or bike racks, making them sensitive to tighter IoU suppression.

---

## 4. What This Evaluation Firmly Establishes

1. **Config E is Defensibly Superior to the Flat Baseline:**
   - On the official test split, Config E improved small-VRU recall by $+1.78\text{ pp}$ with **100% bootstrap probability of positive gain**.
   - Net VRU detections increased by **+34 individuals** (+22 small, +13 medium, −1 large).
   - False alarms increased by only **+0.018 FP/image** (1 additional false alarm per 55 frames).
   - Precision remained high at **77.10%** (only 0.85 pp below baseline).
2. **Post-Processing Optimization Has Reached Its Fundamental Limit:**
   - No further gain can be extracted from non-parametric post-processing (thresholds or NMS) without severely compromising precision.
   - The absolute small-VRU recall on the test split remains **47.29%** (52.71% of small VRUs remain undetected).
3. **Integrity and Reproducibility Are Intact:**
   - The evaluation was conducted cleanly and transparently without tuning to test data.
   - All outputs, manifests, and predictions are permanently cached and reproducible.

---

## 5. Next Steps & Recommended Direction

1. **Freeze Test Split:**
   - Maintain strict quarantine over `bdd100k_dataset/test/`. No further runs on the test set should be executed until a genuinely new model architecture is trained and validated.
2. **Operational Deployment Recommendation:**
   - For immediate operational deployment under the current YOLO11m weights, **Config E remains the recommended inference configuration over the flat baseline**, because its measured gain is positive with 100% statistical probability ($P(\Delta > 0) = 1.0$) and its precision ($77.10\%$) easily meets safety requirements.
   - However, it must be acknowledged that the contractual $+2.50\text{ pp}$ gate was not satisfied.
3. **Architecture-Level Recommendations for Next Development Cycle:**
   - Further meaningful improvements in small-VRU recall ($>55\%$) cannot be achieved by post-processing tuning. They will require:
     * Architectural feature enhancement (e.g., P2 micro-detection head for tiny bounding boxes).
     * High-resolution targeted training or patch-based multi-scale feature pyramids.
     * Focal loss reweighting prioritizing small-area bounding box regression.
