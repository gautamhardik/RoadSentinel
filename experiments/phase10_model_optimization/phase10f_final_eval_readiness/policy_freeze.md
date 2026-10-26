# RoadSentinel Phase 10F — Operational Policy Freeze Specification

**Policy Designation:** RoadSentinel Candidate Inference Policy — Config E  
**Freeze Status:** **LOCKED & FROZEN** (No further tuning permitted)  
**Effective Date:** Phase 10F Pre-Final Evaluation Gate  

---

## 1. Frozen Artifacts & Checksums

The model weights and architecture are completely immutable. No training or fine-tuning is permitted:

| Component | Repository Path | Immutable SHA-256 Checksum |
|---|---|---|
| **Champion Model Checkpoint** | `experiments/phase10_model_optimization/phase10b_extended_training/session4_artifacts/session4_best.pt` | `4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd` |
| **Baseline Checkpoint (Phase 6)** | `scratch/kaggle_upload_weights/best.pt` | `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` |

---

## 2. Frozen Inference Parameters (Config E)

Config E is strictly defined by the following non-parametric postprocessing rules applied to the 960px champion detector:

### A. Class-Specific Confidence Thresholds ($\tau$)
$$\tau(c) = \begin{cases} 0.22 & \text{if } c \in \{\text{person, rider, bike, motor}\} \\ 0.25 & \text{if } c \in \{\text{car, bus, truck, traffic light, traffic sign, train}\} \end{cases}$$

### B. Class-Specific Greedy NMS IoU Thresholds
$$\text{NMS\_IoU}(c) = \begin{cases} 0.55 & \text{if } c \in \{\text{person, rider, bike, motor}\} \\ 0.65 & \text{if } c \in \{\text{car, bus, truck, traffic light, traffic sign, train}\} \end{cases}$$

### C. Geometric & Resolution Parameters
- **Input Resolution:** $960 \times 960$ letterbox (Bilinear interpolation, stride 32, zero-padding to square aspect ratio).
- **Forward Passes:** Exactly 1 forward pass per frame (Deterministic $O(1)$ cycle time).
- **Secondary Active Foveation / Crops:** **PROHIBITED & EXCLUDED**.
- **Geometric Horizon / Elevation Hard Cutoffs:** **PROHIBITED & EXCLUDED**.

---

## 3. Freeze Governance & Enforcement

1. **Parameter Lock:** Confidence thresholds ($0.22 / 0.25$) and NMS IoU thresholds ($0.55 / 0.65$) are permanently locked. No post-hoc sweep or adjustment is permitted during or after test-split evaluation.
2. **Evaluator Lock:** Greedy IoU matching threshold is locked at $\text{IoU} \ge 0.50$ per class. Class IDs and ground-truth bounding box coordinate spaces ($1280 \times 720$) are fixed.
3. **Quarantine Lock:** The official BDD100K test split (`bdd100k_dataset/test/`) remains strictly quarantined and unread until the one-time final execution script is launched.
