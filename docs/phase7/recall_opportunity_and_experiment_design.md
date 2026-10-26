# RoadSentinel Phase 7 — Recall Opportunity Audit & Experiment Design

**Date:** 2026-10-12  
**Status:** DESIGN & AUDIT COMPLETE (AWAITING REVIEW BEFORE EXECUTION)  
**Target Checkpoint:** Frozen Epoch 26 YOLO11m (`experiments/phase6_session2_error/runs/phase6_yolo11m_960_session1/weights/best.pt`)  
**Holdout Selection Fitness:** `0.31074`  
**Data Boundaries:**  
- 300 Pilot Images: Development / Tuning / Pilot Evaluation  
- 9,700 Validation Images: Confirmatory Validation Partition (Untouched)  
- 20,000 Test Images: Strictly Quarantined (0 Touches)  

---

## 1. Executive Summary & Objective

The objective of this design is to identify the largest measurable sources of missed Vulnerable Road Users (VRUs) and specify controlled, attributed experiments across four candidate levers:
1. **Lower Detector Proposal Threshold:** Quantifying the low-confidence candidate pool.
2. **Autonomous Horizon-Corridor Slicing:** Recovering distant VRUs missed by full-frame downsampling.
3. **Live Vision-Language Model (VLM) Deliberation:** Epistemic verification via NVIDIA Nemotron Nano 2 VL.
4. **Native-Resolution Inference & Test-Time Augmentation (TTA):** Disentangling multi-scale representation gains.

Each lever is designed as an isolated ablation with strict attribution, pre-declared compute budgets, deterministic fail-safes, and zero data leakage.

---

## 2. Task 1 — Data Partition Audit & Boundaries

| Cohort Name | Image Count | GT Annotations | Permitted Use | Quarantine Status |
|---|---|---|---|---|
| **Training Holdout** | 5,000 | ~92,500 | Checkpoint selection only (`fitness=0.31074`). | Frozen. Completed in Phase 6. |
| **Pilot Development Set** | 300 | 1,740 VRUs (5,139 total) | Parameter calibration, threshold sweeps, prototype slicing, development ablations. | Open for development. Used in Phase 4B/4C and Phase 6. |
| **Confirmatory Validation** | 9,700 | 13,633 VRUs (~170,000 total) | One-shot confirmatory evaluation of frozen pipeline. | **STRICTLY UNTOUCHED**. Zero threshold tuning permitted. |
| **Official Validation** | 10,000 | 15,373 VRUs (185,578 total) | Standalone official detector evaluation baseline (`evaluate_official_val.py`). | **PENDING GPU QUOTA RESET**. |
| **Test Split** | 20,000 | ~370,000 | Final competition evaluation. | **100% QUARANTINED (0 TOUCHES)**. |

---

## 3. Task 2 — Empirical Low-Confidence Proposal Opportunity Audit

Evaluated on the 300-image development cohort using the frozen Epoch 26 YOLO11m checkpoint across $\tau \in [0.05, 0.30]$ at $\text{IoU} = 0.50$ (measured via [`scratch/audit_threshold_sweep.py`](../../scratch/audit_threshold_sweep.py)):

| Operating Threshold ($\tau$) | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | VRU Recall | F1-Score | Small-VRU TP (<32²) | Small-VRU Recall |
|---|---|---|---|---|---|---|---|---|
| **$\tau = 0.30$** | 988 | 212 | 752 | 82.33% | 56.78% | 0.6721 | 338 | 37.51% |
| **$\tau = 0.25$ (Current Baseline)** | **1,056** | **319** | **684** | **76.80%** | **60.69%** | **0.6780** | **389** | **43.17%** |
| **$\tau = 0.20$** | 1,140 (+84) | 510 (+191) | 600 | 69.09% | 65.52% (+4.83 pp) | 0.6726 | 445 (+56) | 49.39% (+6.22 pp) |
| **$\tau = 0.15$** | 1,213 (+157) | 772 (+453) | 527 | 61.11% | 69.71% (+9.02 pp) | 0.6513 | 496 (+107) | 55.05% (+11.88 pp) |
| **$\tau = 0.10$** | 1,281 (+225) | 1,219 (+900) | 459 | 51.24% | 73.62% (+12.93 pp) | 0.6042 | 551 (+162) | 61.15% (+17.98 pp) |
| **$\tau = 0.05$** | 1,381 (+325) | 2,259 (+1940)| 359 | 37.94% | 79.37% (+18.68 pp) | 0.5134 | 621 (+232) | 68.92% (+25.75 pp) |

### Key Findings on Proposal Ceiling:
1. **Raw Proposal Headroom:** The detector already predicts up to **79.37% VRU recall** and **68.92% small-VRU recall** at $\tau = 0.05$. There are **325 true VRUs** existing in the proposal pool between $\tau = 0.05$ and $0.25$.
2. **False Alarm Trade-Off:** Lowering the raw operating point without filtering increases false alarms from 319 to 2,259 (precision collapses from 76.80% to 37.94%).
3. **The Verification Target:** The core opportunity is not retraining the detector, but **epistemic verification**: evaluating proposals in the window $\tau \in [0.10, 0.25]$ via autonomous slicing or VLM deliberation to recover true VRUs while filtering false positives.

---

## 4. Task 3 — Autonomous Horizon-Slice Ablation Specification

### Scientific Motivation
Current Active Foveation is purely *reactive*: it only crops if the full-frame pass already produced a detection. Distant pedestrians ($<20\text{ px}$ tall) that are completely missed by full-frame 960 downsampling never receive a crop. An *autonomous horizon slice* evaluates the central driving horizon at native pixel resolution regardless of initial detector output.

### Technical Specification:
1. **Crop Geometry:**
   - Frame Dimensions: $1280 \times 720$.
   - Horizon Band Crop: $x_1 = 160$, $y_1 = 200$, $x_2 = 1120$, $y_2 = 520$ (Width: 960 px, Height: 320 px).
   - Covers central 75% horizontal corridor and the vertical horizon range $[0.28 \cdot H, 0.72 \cdot H]$ where distant road users concentrate.
2. **Inference Resolution:**
   - Sliced patch is inferred at $960 \times 320$ (or padded to $960 \times 640$) at $\text{conf} = 0.10$.
3. **Coordinate Transformation:**
   - Forward: Extract patch `frame[y1:y2, x1:x2]`.
   - Inverse: Remap detected box $[rx_1, ry_1, rx_2, ry_2]$ to global image:
     $$gx_1 = x_1 + rx_1, \quad gy_1 = y_1 + ry_1, \quad gx_2 = x_1 + rx_2, \quad gy_2 = y_1 + ry_2$$
4. **Boundary Artifact Suppression:**
   - Discard patch detections with edge touching within 8 px of patch border if confidence $< 0.30$.
5. **Fusion Rules & Duplicate Handling:**
   - Feed global full-frame detections and horizon slice detections into Calibrated WBF:
     - Horizon slice detections receive source tag `"horizon_slice"` with weight $1.1\times$.
     - $\text{IoU}_{\text{match}} = 0.45$. Consensus boost $\beta = +0.06$ when supported by full-frame.
6. **Incremental Latency:**
   - Adds exactly 1 forward pass per frame at $960 \times 320$. Measured CPU stage latency: $641.4\text{ ms}$ slice inference ($647.6\text{ ms}$ including extraction and WBF fusion); projected GPU latency (unverified projection): $\approx +6\text{ ms}$.

---

## 5. Task 4 — Live VLM Deliberation Ablation Specification

### Model & Endpoint
- **Model:** `nvidia/nemotron-nano-12b-v2-vl` (Nebius Token Factory, OpenAI-compatible).
- **Endpoint:** `https://api.tokenfactory.nebius.com/v1/chat/completions`.

### Failure Modes & Deterministic Guardrails (VLM as Untrusted Hypothesis)
The VLM is treated as an untrusted verification module. It must **not** be described as a verified safety filter before evaluation.

| Failure Event | Trigger Condition | System Guardrail / Behavior |
|---|---|---|
| **Network Timeout** | Response time $> 2.0\text{ seconds}$ | Abort call; engage deterministic fallback; preserve detector prior; log latency violation. |
| **API Error / 5xx / 4xx**| HTTP status $\ne 200$ or connection refused | Catch exception; log error; retain detector prior; assert fail-safe escalation. |
| **Malformed Response**| Non-JSON text or schema violation | Catch JSON decode error; discard text; retain detector prior; flag as unresolved. |
| **Uncertain Answer** | `"epistemic_confidence" < 0.70` | **Safety Override:** Reject background dismissal; keep detector's VRU prior; set `"safety_escalation": true`. |
| **Hallucinated Class** | Class not in `["person", "rider", "bike", "motor", "background_false_positive"]` | Reject verdict; retain detector prior. |
| **Hallucinated Box** | VLM attempts to define bounding coordinates | Ignored; bounding box coordinates are strictly locked to detector output. |

---

## 6. Task 5 — Native-Resolution & TTA Experiment Specification

To prevent conflation and ensure clear scientific attribution, resolution and test-time augmentation must be evaluated independently:

### Experiment 5A: FullFrame-1280 Native Resolution Inference
- **Input:** Full frame resized to $1280 \times 1280$ (native width, letterboxed height $720 \rightarrow 1280$).
- **Model:** Frozen Epoch 26 YOLO11m (`imgsz=1280`).
- **Control:** Standalone evaluation; NO foveation crops, NO horizon slice, NO VLM.
- **Hypothesis:** Higher pixel density ($1.77\times$ pixel area vs 960) improves small-object feature extraction.

### Experiment 5B: Test-Time Augmentation (TTA)
- **Input:** $960 \times 960$ with horizontal flip and multi-scale factors $[0.83, 1.0, 1.15]$.
- **Control:** Standalone evaluation; NO foveation crops, NO VLM.
- **Hypothesis:** Ensembling multi-scale augmented passes stabilizes bounding box jitter and improves recall near the threshold boundary.

---

## 7. Task 6 — Experiment Manifest, Metrics & Reconciled Acceptance Decisions

### Pre-Declared Acceptance Gates vs Measured Empirical Results (300 Development Images)

| Experiment ID | Lever Tested | Primary Metric | Acceptance Target | Measured Result | Gate Decision | Engineering Rationale |
|---|---|---|---|---|---|---|
| **EXP-7A** | Lower Proposal Threshold ($\tau=0.15$) | Operational VRU Recall & F1 | Recall $\ge 68.0\%$ with $\text{F1} \ge 0.650$ | **Recall: 69.71%**, **F1: 0.6513** (TP: 1,213, FP: 772) | **CONDITIONAL / EXPLORATORY PASS** | Recovers +157 TP over baseline, but at the cost of +453 FP (precision drops from 76.80% to 61.11%). Cannot be deployed without filtering. |
| **EXP-7B** | Autonomous Horizon Slice + WBF | Small VRU Recall (<32²) | Small Recall $\ge 55.0\%$ (+3.0 pp over Foveation) | **Small Recall: 48.50%** (Small TP: 437) | **FAIL (GATE NOT MET)** | Improves on raw baseline (43.17% $\rightarrow$ 48.50%, +5.33 pp), but misses 55.0% target by -6.50 pp and trails Active Foveation (51.94%, 468 TP). |
| **EXP-7C** | Live VLM Deliberation (Nemotron Nano 2 VL) | Gated FP Reduction | Reduce ego FP by $\ge 20\%$, 0 safety dismissals | Untested (Pending API key) | **PENDING CREDENTIALS** | Deterministic prior-preservation fail-safe verified; live validation awaiting `NEBIUS_API_KEY`. |
| **EXP-7D** | FullFrame-1280 Native Inference | Small VRU Recall & mAP@50 | mAP@50 $\ge 0.540$ (+1.2 pp over 960 baseline) | Untested (Specification Complete) | **PENDING GPU RESET** | Requires native resolution inference on GPU. |
| **EXP-7E** | Test-Time Augmentation (TTA) | VRU mAP@50-95 | mAP@50-95 $\ge 0.305$ (+1.2 pp over 960 baseline) | Untested (Specification Complete) | **PENDING GPU RESET** | Requires 3 augmented multi-scale passes on GPU. |

### Reconciliation Details for EXP-7B:
- **Baseline Comparison:** Raw YOLO11m baseline achieved 43.17% small-VRU recall (389 TP). Autonomous Horizon Slicing increased this to 48.50% (437 TP), delivering a genuine +5.33 pp gain (+48 small VRUs).
- **Target Comparison:** The pre-declared target was $\ge 55.00\%$. The measured result of 48.50% fails this gate by -6.50 pp.
- **Foveation Benchmark Comparison:** Reactive Active Foveation achieved 51.94% small-VRU recall (468 TP). Horizon Slicing trails Active Foveation by -3.44 pp (31 fewer small VRUs), because fixed horizon cropping misses lateral road boundaries ($x < 160$ or $x > 1120$) where reactive crops can dynamically steer.

---

## 8. Task 7 — Execution Feasibility & Compute Pre-Requisites

| Experiment | CPU Feasible? | GPU Required? | Live API Key Required? | Estimated Runtime (300 Pilot Images) |
|---|---|---|---|---|
| **EXP-7A (Threshold Sweep + WBF)** | **YES** | No | No | ~2 minutes |
| **EXP-7B (Autonomous Horizon Slice)** | **YES** | Recommended for speed | No | ~4 minutes (CPU) / ~15s (GPU) |
| **EXP-7C (Live VLM Deliberation)** | **YES** (inference) | No | **YES (`NEBIUS_API_KEY`)** | ~1 minute (bounded to 18–30 calls) |
| **EXP-7D (FullFrame-1280)** | **YES** (slow) | **YES (Recommended)** | No | ~12 minutes (CPU) / ~20s (GPU) |
| **EXP-7E (Test-Time Augmentation)** | **YES** (slow) | **YES (Recommended)** | No | ~15 minutes (CPU) / ~30s (GPU) |
| **Official 10,000 Validation Baseline** | Marginal (~40m) | **YES (Required for quota reset)** | No | ~40 min (CPU) / ~18 min (GPU) |

---

## 9. Statement on Checkpoint & Quarantine Integrity

1. **Frozen Checkpoint:** The Epoch 26 YOLO11m checkpoint (`experiments/phase6_session2_error/runs/phase6_yolo11m_960_session1/weights/best.pt`) remains bit-for-bit identical to its frozen state dict fingerprint (`706008333c...f59e`).
2. **Confirmatory Partition (9,700 images):** Remains 100% untouched. All exploratory sweeps above were executed strictly on the 300 development images.
3. **Test Split (20,000 images):** Strictly quarantined with zero touches.
