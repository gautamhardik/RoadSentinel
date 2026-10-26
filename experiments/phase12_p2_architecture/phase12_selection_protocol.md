# RoadSentinel Phase 12 — Checkpoint Selection Protocol

**Status:** FROZEN & RATIFIED (Pre-Training Execution Rule)  
**Target Architecture:** YOLO11m-P2 (`experiments/phase12_p2_architecture/yolo11m-p2.yaml`)  
**Designated Evaluation Split:** BDD100K Official Validation Cohort (10,000 images, quarantined test split remains untouched)  
**Quarantine Invariant:** The official BDD100K test split (`bdd100k_dataset/test/`) is strictly prohibited. Under no circumstances may test images or labels be accessed, preprocessed, or used to evaluate or break ties among candidate checkpoints.

---

## 1. Objective & Governance

This document establishes the binding, immutable protocol for evaluating and selecting the champion checkpoint from the 40-epoch YOLO11m-P2 training run. To prevent researcher degrees of freedom, post-hoc metric cherry-picking, and overfitting to the validation cohort, all selection criteria, eligibility thresholds, tie-breaking hierarchies, and statistical bounds defined herein are frozen prior to model training.

---

## 2. Standardized Evaluation Conditions

Every candidate checkpoint (saved at epoch checkpoints or periodic evaluation intervals) must be evaluated under identical, standardized conditions:

1. **Resolution & Preprocessing:** Native $960 \times 960$ letterbox input with aspect-ratio preservation and identical bilinear interpolation.
2. **Class Mapping:** Standard BDD100K 10-class taxonomy:
   - `0: person`, `1: rider`, `2: car`, `3: bus`, `4: truck`, `5: bike`, `6: motor`, `7: traffic light`, `8: traffic sign`, `9: train`.
3. **VRU Class Subset:** The four vulnerable road user classes: `person` (0), `rider` (1), `bike` (5), and `motor` (6).
4. **Size Categorization (Ground Truth Bounding Box Area):**
   - **Small:** $\text{Area} < 32^2 = 1,024\text{ pixels}^2$
   - **Medium:** $32^2 \le \text{Area} < 96^2 = 9,216\text{ pixels}^2$
   - **Large:** $\text{Area} \ge 96^2 = 9,216\text{ pixels}^2$
5. **Detection Policy (Config E Standards):**
   - Confidence threshold: **0.22** for VRU classes (`person`, `rider`, `bike`, `motor`).
   - Confidence threshold: **0.25** for all vehicle and infrastructure classes.
   - NMS IoU threshold: **0.55** for VRU classes.
   - NMS IoU threshold: **0.65** for vehicle and infrastructure classes.
6. **Matching IoU:** Standard PASCAL VOC matching criterion of $\text{IoU} \ge 0.50$ between prediction and ground-truth bounding boxes.

---

## 3. Metric Definitions & Reporting Requirements

For every evaluated checkpoint, the following metrics must be computed and recorded in tabular format:
- **Small-VRU Recall:** $\frac{\text{TP}_{\text{VRU, small}}}{\text{GT}_{\text{VRU, small}}}$
- **Overall VRU Precision:** $\frac{\text{TP}_{\text{VRU}}}{\text{TP}_{\text{VRU}} + \text{FP}_{\text{VRU}}}$
- **Overall VRU Recall:** $\frac{\text{TP}_{\text{VRU}}}{\text{GT}_{\text{VRU}}}$
- **Overall VRU F1:** $\frac{2 \cdot \text{Precision}_{\text{VRU}} \cdot \text{Recall}_{\text{VRU}}}{\text{Precision}_{\text{VRU}} + \text{Recall}_{\text{VRU}}}$
- **Per-Class VRU Metrics:** Precision, Recall, and F1 reported individually for `person`, `rider`, `bike`, and `motor`.
- **Vehicle & Infrastructure Metrics:** Monitored to confirm non-VRU invariance.

---

## 4. Checkpoint Eligibility Criteria (Hard Gates)

A candidate checkpoint is declared **ELIGIBLE** if and only if it satisfies both precision constraints simultaneously:

$$\text{VRU Precision Point Estimate} \ge 72.0\%$$
$$\text{VRU Precision 95\% CI Lower Bound} \ge 70.0\%$$

*Where the 95% Confidence Interval is derived via 2,000 image-level bootstrap replicates.*

> [!CRITICAL]
> **No Relaxation Rule:** If no checkpoint satisfies both eligibility criteria across all 40 epochs, the selection gate fails. Thresholds will NOT be relaxed, adjusted, or post-hoc modified.

---

## 5. Candidate Selection Hierarchy

Among all candidate checkpoints that achieve **ELIGIBLE** status:

1. **Primary Objective (Small-VRU Recall):**
   Select the checkpoint with the highest **Small-VRU Recall**.
2. **Secondary Objective (Tie-Break Level 1):**
   If two or more eligible checkpoints share identical Small-VRU Recall (within $0.05\text{ pp}$), select the candidate with the higher **Overall VRU F1-score**.
3. **Tertiary Objective (Tie-Break Level 2):**
   If still tied, select the candidate with the higher **Overall VRU Precision**.
4. **Final Tie-Break (Epoch Index):**
   If still tied, select the checkpoint from the earlier training epoch (favoring compute efficiency and avoiding late-epoch over-fitting).

---

## 6. Artifact Serialization & Immutability

1. **Output Artifact Path:** The winning checkpoint must be saved to:
   `experiments/phase12_p2_architecture/yolo11m_p2_selected.pt`
2. **Hash Recording:** Compute and record the SHA-256 checksum of the selected checkpoint in `phase12_training_report.md`.
3. **Preservation of Baselines:**
   - Under no circumstances will `experiments/phase10_model_optimization/phase10b_extended_training/session4_artifacts/session4_best.pt` be overwritten.
   - Under no circumstances will `experiments/phase6_session1/handoff/best.pt` be overwritten.
   - Under no circumstances will `experiments/phase12_p2_architecture/yolo11m_p2_initialized.pt` be overwritten.
4. **Traceability:** The winning checkpoint's source epoch, training step loss, validation curves, and full metric breakdown must be explicitly detailed in the closeout report.
