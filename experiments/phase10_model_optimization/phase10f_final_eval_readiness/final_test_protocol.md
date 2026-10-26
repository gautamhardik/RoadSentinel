# RoadSentinel Phase 10F — Final Official Test Split Evaluation Protocol

## 1. Governance & Strict Non-Contamination Invariants

This protocol specifies the locked rules for the **one-time, blind final evaluation** on the official BDD100K test split (`bdd100k_dataset/test/`).

### Invariants:
1. **Zero Prior Exposure:** The test split has recorded exactly **0 accesses** through Phases 1 through 10F.
2. **One-Time Evaluation:** The test split is evaluated exactly once. It must **never** be used for:
   - Threshold selection or parameter sweeps.
   - NMS IoU tuning.
   - Post-hoc error debugging or retraining triggers.
   - Model selection.
3. **Deficiency Protocol:** If the final test evaluation reveals a performance shortfall or gate failure, the result must be recorded transparently. Any subsequent iteration or model architecture must be developed exclusively on training and validation splits.
4. **Execution Prohibition in Phase 10F:** This protocol is locked during Phase 10F. **Execution on the test split during Phase 10F is strictly prohibited.**

---

## 2. Frozen Models & Exact Inference Policies

### A. Evaluated Model Checkpoints
- **Primary Champion Checkpoint:** `experiments/phase10_model_optimization/phase10b_extended_training/session4_artifacts/session4_best.pt`
  * Mandatory SHA-256: `4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd`
- **Reference Baseline Checkpoint:** `scratch/kaggle_upload_weights/best.pt`
  * Mandatory SHA-256: `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e`

### B. Evaluated Policy Configurations
Both policies will be evaluated across the identical test cohort:

1. **Candidate Production Policy (Config E):**
   - Confidence threshold: $\tau = 0.22$ for `person, rider, bike, motor`; $\tau = 0.25$ for all other classes.
   - NMS IoU threshold: $\text{IoU} = 0.55$ for `person, rider, bike, motor`; $\text{IoU} = 0.65$ for all other classes.
   - Forward passes: 1 pass per frame ($960 \times 960$ letterbox).
2. **Controlled Baseline Policy:**
   - Confidence threshold: $\tau = 0.25$ for all classes.
   - NMS IoU threshold: $\text{IoU} = 0.65$ for all classes.
   - Forward passes: 1 pass per frame ($960 \times 960$ letterbox).

---

## 3. Evaluation Environment & Matching Rules

1. **Coordinate & Resolution Conventions:**
   - Input resolution: $960 \times 960$ letterbox (stride 32).
   - Coordinate space: Detector boxes natively projected to original BDD100K image dimensions ($1280 \times 720$).
2. **Object Size Definitions (COCO Area Brackets):**
   - Small: $\text{Area} < 1024\text{ px}^2$ ($< 32^2$)
   - Medium: $1024 \le \text{Area} < 9216\text{ px}^2$ ($[32^2, 96^2)$)
   - Large: $\text{Area} \ge 9216\text{ px}^2$ ($\ge 96^2$)
3. **Matching Engine:**
   - `match_single_image` greedy IoU matching at threshold $\text{IoU} \ge 0.50$.
   - Same-class matching strictly enforced.
4. **Statistical Uncertainty Requirements:**
   - Paired image-level bootstrap resampling ($B = 2,000$ iterations, seed 42) computing empirical $95\%$ confidence intervals for paired deltas:
     $$\Delta = \text{Metric}_{\text{Config\_E}} - \text{Metric}_{\text{Baseline}}$$

---

## 4. Predeclared Acceptance Gates for Final Evaluation

For Config E to be declared formally validated on the official test split, the following pre-declared gates must be met:

| Gate Identifier | Metric Criterion | Pre-Declared Requirement | Rationale |
|---|---|---|---|
| **Gate 1: Small-VRU Recall Gain** | $\Delta \text{Small-VRU Recall}$ | $\ge +2.50\text{ pp}$ and $P(\Delta > 0) \ge 99.0\%$ | Verifies that small-VRU recovery is reproducible on unvisited test scenes. |
| **Gate 2: Precision Safety Floor** | Config E VRU Precision | $\ge 72.0\%$ point estimate, and $95\%\text{ CI lower bound} \ge 70.0\%$ | Preserves operating precision and limits false-positive braking events. |
| **Gate 3: Net F1 Superiority** | $\Delta \text{Overall VRU F1}$ | $> 0.000$ and $P(\Delta > 0) \ge 95.0\%$ | Confirms that recall gains exceed precision trade-offs. |
| **Gate 4: Computational Cycle** | Forward Passes per Frame | Exactly $1.00$ ($O(1)$ deterministic) | Guarantees zero multi-crop foveation jitter. |

---

## 5. Execution Logging & Artifact Storage

When explicitly authorized by the user, final test evaluation artifacts will be written to:
`experiments/phase11_final_test_evaluation/`

Required artifacts:
- `test_cohort_manifest.csv`
- `final_test_policy_comparison.csv`
- `final_test_per_class_metrics.csv`
- `final_test_statistical_report.md`
- `final_test_decision_record.md`
- Complete terminal execution log with verified input/output SHA-256 hashes.
