# RoadSentinel Phase 5A — Small-VRU Model Experiment Design Report

> **CORRECTION (2026-10-11).** Actual BDD100K val per-class VRU counts are person 13,265 / rider 649 / bike 1,007 / motor 452 (total 15,373), per [docs/phase6/stage1_split_audit.md](../phase6/stage1_split_audit.md); the per-class figures below are wrong, the total is right. Outcome of this design: see the [Stage 0 matched-policy evaluation](../phase6/stage0_matched_evaluation_report.md) — neither Phase 5A candidate improved the detector at matched precision.

**Status:** COMPLETE & FROZEN  
**Phase:** 5A  
**Mode:** Controlled Model Research Experiment Design  
**Selected First Experiment:** **Candidate 1 — Targeted Small-VRU Fine-Tuning**  
**Deferred Experiment:** Candidate 2 — Higher-Resolution P2 Detection Head  
**Test Split Quarantine:** Maintained (0 images accessed; 20,000-image BDD100K test split quarantined)  
**Machine-Readable Config:** [experiments/phase5a_modeling/training_config.yaml](file:///c:/Users/hiten/Downloads/cv/experiments/phase5a_modeling/training_config.yaml)  
**Experiment Manifest:** [experiments/phase5a_modeling/run_manifest.json](file:///c:/Users/hiten/Downloads/cv/experiments/phase5a_modeling/run_manifest.json)

---

## 1. Executive Summary & Objective

In Phases 4B, 4C, and 4D, we established through rigorous benchmark experiments that post-processing calibration (source-specific thresholding, concurrence routing, and Weighted Box Fusion) has reached its empirical Pareto boundary. While Phase 4C significantly reduced false alarms ($+8.83\text{ pp}$ precision gain), post-hoc confidence adjustment on a frozen detector backbone cannot recover small Vulnerable Road Users ($< 32\times 32\text{ px}$) whose raw sigmoid activations never exceed $0.20$.

Phase 5A initiates the modeling phase to directly enhance feature representation and confidence calibration for small VRUs. This report inspects the existing perception pipeline, compares two research directions, justifies the selection of **Candidate 1 (Targeted Fine-Tuning)** as the first controlled experiment, and predeclares frozen acceptance criteria.

---

## 2. Inspection of Current Perception Pipeline & Checkpoint

### A. Checkpoint Architecture
Inspection of the production checkpoint `weights/bdd100k_yolo11s.pt` (`A4747292648F37F68AB997BEA65A36391C41356F35E309A9F3D2EB1D3EA8D8E3`) reveals:
- **Architecture:** Ultralytics YOLO11s (Single-stage anchor-free detector).
- **Parameter Count:** $9,431,662$ parameters (~9.43M).
- **Feature Pyramid:** Standard Path Aggregation Network (PAN) with 3 detection heads:
  - **P3 Head (Stride 8):** For small objects (feature map: $120 \times 120$ at 960px input, $80 \times 80$ at 640px).
  - **P4 Head (Stride 16):** For medium objects (feature map: $60 \times 60$ at 960px input, $40 \times 40$ at 640px).
  - **P5 Head (Stride 32):** For large objects (feature map: $30 \times 30$ at 960px input, $20 \times 20$ at 640px).
- **Classes:** 10 BDD100K classes (`person`, `rider`, `car`, `bus`, `truck`, `bike`, `motor`, `traffic light`, `traffic sign`, `train`).

### B. Dataset Class Distribution & Small-Object Imbalance
Analysis of the BDD100K dataset manifests reveals severe distribution skews:
1. **Total Annotations:** The dataset contains 70,000 training images, 10,000 validation images, and 20,000 quarantined test images.
2. **Category Imbalance:** Vehicles heavily dominate the training distribution ($>100,000$ car instances in validation vs $11,737$ person, $885$ bike, $591$ rider, $420$ motor instances). Minority VRUs (rider, bike, motor) represent less than $2\%$ of all bounding boxes.
3. **Scale Imbalance:** In the validation set, $45.72\%$ of all VRUs ($7,029 / 15,373$) are Small ($< 32\times 32\text{ px}$). Many distant pedestrians in highway and suburban scenes span only $10 \times 20\text{ px}$.
4. **The Stride-8 Representation Bottleneck:** At stride 8, a $10 \times 20\text{ px}$ pedestrian occupies only $1.25 \times 2.5$ feature cells in P3. In the off-the-shelf checkpoint, the network's loss function was dominated by high-confidence vehicle boxes, resulting in undertrained feature representations and underconfident predictions for small VRUs.

### C. Available Hardware Environment
Target execution runs in a standardized GPU container (Kaggle GPU with Tesla T4 16GB or P100 16GB, PyTorch 2.1.2, CUDA 12.1).

---

## 3. Comparison of Candidate Research Directions

| Evaluation Dimension | Candidate 1: Targeted Fine-Tuning | Candidate 2: Higher-Resolution P2 Feature Head |
|---|---|---|
| **Core Mechanism** | Retain YOLO11s architecture; fine-tune with VRU class weighting, small-object copy-paste, and 960px resolution. | Modify network architecture to add a stride-4 (P2) detection head to the feature pyramid. |
| **Expected Benefit** | Elevates feature activation and calibration for minority and small VRUs without architectural disruption. | Quadruples spatial feature density for tiny VRUs ($< 32\text{ px}$), providing 16x more feature cells for 16px objects. |
| **Engineering Risk** | **Low.** 100% compatible with existing weights, inference pipeline, WBF consensus engine, and deployment runtimes. | **High.** Requires custom YOLO11 YAML configuration, uninitialized random weights in the new head, and risk of gradient shock. |
| **Latency & Memory Impact** | **Zero.** Retains exact production latency ($16.23\text{ ms}$ P50) and VRAM ($159.9\text{ MB}$). | **Severe.** Increases inference latency by $+35\%\text{ to }+50\%$ ($>28\text{ ms}$ P50) and increases peak VRAM by $+50\%$. |
| **Compute Cost** | **Low to Moderate:** ~3.5 GPU hours on T4 for 8 controlled epochs. | **High:** ~20–30 GPU hours (requires extensive re-training to train the uninitialized head). |
| **Vehicle Regression Risk** | **Low:** Low learning rate ($3 \times 10^{-4}$) and cosine decay prevent catastrophic forgetting. | **Moderate to High:** New multi-scale gradient routing can degrade medium and large vehicle AP. |
| **Evaluation Complexity** | Directly evaluated on established foveation benchmark pipeline. | Requires re-benchmarking latency, throughput, and memory budgets alongside accuracy. |

---

## 4. Experiment Selection Rationale

### Selection: **Candidate 1 (Targeted Fine-Tuning)** is selected as the first experiment.

**Scientific Justification:**
1. **Principle of Isolated Variables:** Modifying the architecture *and* changing the training recipe simultaneously would confound the source of any observed performance delta. If metrics degrade or improve, it would be impossible to isolate whether the architecture or the hyperparameters caused the change.
2. **Efficiency & Risk Minimization:** Candidate 1 directly tests whether the small-VRU recall deficit is an **optimization/data representation issue** that can be resolved within the existing computational footprint.
3. **Automotive Latency Budget:** ActiveFoveation-960 currently achieves a batch-1 in-memory median latency of $16.23\text{ ms}$ ($20.33\text{ ms}$ pipeline P50). Candidate 2 threatens to push P95 latency beyond automotive safety limits ($>100\text{ ms}$).
4. **Staged Progression:** If Candidate 1 fails to achieve $\ge 45.0\%$ small-VRU recall, it establishes definitive empirical evidence that the stride-8 architecture is fundamentally resolution-limited, fully justifying the substantial compute and latency investment of Candidate 2 in Phase 5B.

---

## 5. Candidate 1 Experimental Specification

The machine-readable configuration is frozen in `experiments/phase5a_modeling/training_config.yaml`:

### A. Training Setup & Hyperparameters
- **Base Checkpoint:** `weights/bdd100k_yolo11s.pt` (`A4747292648F37F68AB997BEA65A36391C41356F35E309A9F3D2EB1D3EA8D8E3`).
- **Input Resolution:** $960 \times 960$ (matches production full-frame inference).
- **Training Duration:** 8 epochs (controlled fine-tuning to prevent overfitting and preserve vehicle representations).
- **Batch Size:** 16 (gradient accumulation $= 1$).
- **Optimizer:** AdamW with initial learning rate $\text{lr}_0 = 3 \times 10^{-4}$ (1/10th of scratch training) and cosine decay to $\text{lr}_f = 1.5 \times 10^{-5}$.
- **Weight Decay:** $5 \times 10^{-4}$.
- **Warmup:** 1.0 epoch with bias learning rate warmup.
- **Random Seed:** 42 (deterministic execution).

### B. Targeted Small-VRU Loss & Augmentation Recipe
- **Class Loss Weighting:**
  - `person`: $1.2\times$
  - `rider`: $2.0\times$ (compensates for severe $0.05\%$ class rarity)
  - `bike`: $1.8\times$
  - `motor`: $2.0\times$
  - Vehicles (`car`, `bus`, `truck`, `train`): $1.0\times$
- **Classification Loss Gain:** Elevated from $0.5$ to $1.2$ to penalize false negative VRUs.
- **Targeted Augmentation:**
  - `copy_paste = 0.15`: Targeted copy-paste of small VRU instances into varied road scenes.
  - `mosaic = 0.80`: Multi-scale exposure; closed during the final 2 epochs (`close_mosaic = 2`) for clean feature alignment.
  - `scale = 0.40`: Scale jitter ($0.6\times$ to $1.4\times$) prioritizing small feature learning.
  - `mixup = 0.0`: Disabled to prevent fuzzy, low-contrast boundaries on distant pedestrians.

---

## 6. Predeclared Acceptance Criteria & Evaluation Rules

All evaluation will be executed using the hardened benchmark evaluator across the established 10,000-image validation set, reporting metrics separately for the Full Validation set (10,000 images), Pilot Cohort (300 images), and Non-Pilot Confirmation Partition (9,700 images).

### Acceptance Criteria (Evaluated on 9,700 Confirmation Partition)

| Acceptance Gate | Predeclared Requirement | Canonical Baseline Reference | Target Value |
|---|---|---:|---:|
| **Gate 1: Precision Improvement** | Evaluated against canonical baseline | 62.03% | $\ge 67.03\%$ ($\Delta \ge +5.0\text{ pp}$) |
| **Gate 2: Recall Retention** | $\Delta \text{Recall} \ge -2.0\text{ pp}$ vs canonical baseline | 62.71% | $\ge 60.71\%$ |
| **Gate 3: Small-VRU Recall** | Small-VRU Recall $\ge 45.00\%$ | 46.83% (Canonical) | **$\ge 45.00\%$** |
| **Gate 4: VRU Micro F1** | Micro $F_1 \ge 0.6500$ | 0.6237 (Canonical) | **$\ge 0.6500$** |

### Evaluation Protocol
1. **References:** Compare against both the canonical ActiveFoveation-960 baseline ($62.03\%$ Prec, $62.71\%$ Rec, $46.83\%$ Small Rec) and the Phase 4C Calibrated Reference ($72.86\%$ Prec, $60.74\%$ Rec, $43.36\%$ Small Rec).
2. **Metric Reporting:** Report overall and VRU mAP@50, VRU precision, recall, F1, TP, FP, FN, size-group recall (Small, Medium, Large), per-class metrics (Person, Rider, Bike, Motor), throughput (FPS), batch-1 P50/P95 latency, and peak VRAM.
3. **Quarantine Enforcement:** Zero access to the BDD100K test split.
4. **Exploratory Designation:** Outcomes on the 9,700-image partition will be explicitly documented as exploratory validation.
