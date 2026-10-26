# RoadSentinel Phase 12 — Research Plan: P2 Stride-4 Micro-Detection Head

**Phase Designation:** Phase 12 Architectural Investigation  
**Status:** DRAFT SPECIFICATION & RESEARCH PROTOCOL (Validation-Only)  
**Effort:** High (Architecture design, transfer training, validation audit, compute profiling)  
**Primary Hypothesis:** Adding a stride-4 ($240 \times 240$ @ 960px) P2 feature pyramid tap will preserve spatial high-frequency details for sub-32px VRU objects that are lost in standard stride-8/16/32 backbones.  
**Strict Constraint:** Validation-only exploration. No access to the official BDD100K test split.  

---

## 1. Architectural Design & Initial Compute Profile

The architecture definition is implemented in [`yolo11m-p2.yaml`](file:///c:/Users/hiten/Downloads/cv/experiments/phase12_p2_architecture/yolo11m-p2.yaml).

### Comparative Architecture & Resource Profile (at 960x960 Input Resolution)

| Specification / Resource | Standard YOLO11m (`session4_best.pt`) | YOLO11m-P2 (`yolo11m-p2.yaml`) | Delta ($\Delta$) | Architectural Implication |
|---|---|---|---|---|
| **Feature Pyramid Strides** | P3/8, P4/16, P5/32 | **P2/4, P3/8, P4/16, P5/32** | +1 Detection Scale | Adds stride-4 high-resolution prediction grid |
| **P2 Feature Map Dimensions** | None ($120 \times 120$ max) | **$240 \times 240$ px** | $4\times$ spatial area | Direct spatial alignment with 8px–32px objects |
| **Total Model Layers** | 231 layers | **286 layers** | +55 layers | Additional lateral skip & bottom-up PAN stages |
| **Model Parameters** | 20,060,718 | **20,555,000** | **+494,282 (+2.46%)** | Highly compact parameter expansion |
| **Computational Complexity** | 154.1 GFLOPs | **199.3 GFLOPs** | **+45.2 GFLOPs (+29.3%)** | Higher computation on $240 \times 240$ feature map |
| **Anchor/Candidate Density** | 18,900 candidates/frame | **76,500 candidates/frame** | **$4.05\times$ candidates** | $4\times$ more spatial cells evaluated per frame |

---

## 2. Theoretical Rationale & Failure Mode Anticipation

### A. Why P2 Is Justified
In standard YOLO11 ($960 \times 960$ letterbox for $1280 \times 720$ BDD100K frames):
- A small VRU at distance (e.g. $16 \times 28$ pixels in raw image space) is scaled to approximately $12 \times 21$ pixels at 960px input.
- At stride 8 (P3), this object is represented by only $1.5 \times 2.6$ feature cells.
- At stride 16 (P4), it collapses to $<1$ feature cell, causing feature blending with background asphalt or vehicles.
- In contrast, at **stride 4 (P2)**, the same object occupies **$3 \times 5.25$ feature cells**—retaining distinct aspect ratio and silhouette boundaries before pooling.

### B. Identified Risks & Trade-offs
1. **Compute & Memory Overhead:** The $+29.3\%$ GFLOPs increase directly impacts inference latency and training VRAM.
2. **False Alarm Proliferation:** Because the P2 grid evaluates $57,600$ additional candidate cells per image, background clutter (textures, road cracks, vertical signposts) may trigger false alarms, putting downward pressure on precision.
3. **Bicycle Subcategory Geometry:** Bicycles frequently overlap with riders. The multi-scale head must be trained carefully so that dense P2 detections do not exacerbate NMS suppression artifacts.

---

## 3. Pre-Declared Phase 12 Acceptance Criteria (Validation Split)

Prior to initiating model training or hyperparameter searches, acceptance gates are established strictly on validation data:

| Gate | Metric / Criterion | Threshold Requirement | Rationale |
|---|---|---|---|
| **Gate 1** | Small-VRU Recall (Validation) | $\mathbf{\ge 52.0\%}$ (Point Estimate) | Requires a meaningful $+4.7\text{ pp}$ breakthrough beyond the test-set plateau ($47.29\%$). |
| **Gate 2** | VRU Precision Safety | $\mathbf{\ge 72.0\%}$ Point, $95\%\text{ CI lower bound} \ge 70.0\%$ | Maintains the non-negotiable automotive precision floor. |
| **Gate 3** | Overall VRU F1 Superiority | $\mathbf{\ge 0.7250}$ with $P(\Delta > 0) \ge 95.0\%$ | Confirms net detection quality improvement over champion ($0.7150$). |
| **Gate 4** | Class Stability (`bike`) | $\text{Recall}_{\text{bike}} \ge \mathbf{60.0\%}$ | Prevents recurrence of bicycle suppression seen in Phase 11. |
| **Gate 5** | Latency & Hardware Feasibility | Single forward pass, Peak VRAM $\le 14\text{ GB}$ | Ensures trainable on standard Kaggle GPU (T4/P100) and deployable. |

---

## 4. Phase 12 Execution Roadmap (4 Stages)

### Stage 1: Weight Surgery & Pre-training Alignment
- Initialize backbone from pre-trained `session4_best.pt` weights.
- Initialize new P2 head layers with He (Kaiming) normal initialization.
- Freeze backbone for the first 3 epochs to stabilize the newly introduced P2 detection heads, then unfreeze for joint end-to-end training.

### Stage 2: Training on BDD100K `train` Split (Kaggle GPU Session)
- Train for 30–50 epochs using SGD with cosine learning rate decay ($1\times 10^{-2} \to 1\times 10^{-4}$).
- Input size: $960 \times 960$.
- Batch size: 8 (or 16 if memory permits on dual T4).
- Loss weights: Box 7.5, Cls 0.5, DFL 1.5.

### Stage 3: Controlled Validation Evaluation
- Evaluate on the standardized validation cohort (disjoint from training).
- Compute paired image-level bootstrap intervals ($B=2,000$ replicates, seed 42) comparing YOLO11m-P2 against the frozen champion `session4_best.pt`.
- Perform per-class error audit, particularly on `bike`, `rider`, and small pedestrians.

### Stage 4: Comprehensive Compute & Profiling Report
- Measure exact inference latency (ms/frame) across batch sizes 1, 4, 8 on CPU and GPU.
- Profile peak memory consumption during forward passes.
- Evaluate Gates 1–5 and generate formal decision record.

---

## 5. Governance Invariants

1. **Test Split Quarantine:** The remaining 18,000 test images remain quarantined. The 2,000 test images used in Phase 11 are retired and will never be used for validation or training.
2. **Checkpoint Immutability:** `session4_best.pt` and `best.pt` remain unchanged.
3. **No Premature Claims:** The P2 head is treated as an empirical hypothesis to be rigorously tested, with failure documented transparently if gates are not met.
