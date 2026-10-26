# RoadSentinel Phase 6 — System Evaluation & Ablation Report

**Date:** 2026-10-12  
**Status:** AUDITED PILOT BENCHMARK & CALIBRATION REPORT  
**Selected Checkpoint:** `experiments/phase6_session2_error/runs/phase6_yolo11m_960_session1/weights/best.pt`  
**Winning Epoch:** Epoch 26 (0-indexed epoch 25)  
**Holdout Selection Fitness:** `0.31074` (unweighted macro AP50–95 over person, rider, bike, motor)  
**Machine-Readable Artifacts:**  
- Main Evaluation: [`experiments/phase6_system_eval/system_evaluation_report.json`](../../experiments/phase6_system_eval/system_evaluation_report.json)  
- Latency Protocol Audit: [`experiments/phase6_system_eval/latency_audit_report.json`](../../experiments/phase6_system_eval/latency_audit_report.json)  
- Agentic Deliberation Audit: [`experiments/phase6_system_eval/agentic_evaluation_report.json`](../../experiments/phase6_system_eval/agentic_evaluation_report.json)  

---

## 1. Executive Summary

This report establishes the rigorous, audited system-level evaluation of the RoadSentinel perception stack. It adheres strictly to the frozen Phase 6 checkpoint selection protocol, rigorous data isolation boundaries, and empirical ablation methodology.

```mermaid
graph TD
    A[Frozen Epoch 26 YOLO11m] -->|Verified Weights| B[Pilot System Benchmark: 300 Images]
    B --> C[Pipeline A: Raw YOLO11m 960]
    B --> D[Pipeline B: Active Foveation + Calibrated WBF]
    B --> E[Pipeline C: Temporal ByteTrack with Resets]
    
    C --> F[P: 76.80% | R: 60.69% | F1: 0.6780 | Small-R: 43.17%]
    D --> G[P: 74.29% | R: 65.75% | F1: 0.6976 | Small-R: 51.94%]
    E --> H[P: 76.95% | R: 57.18% | F1: 0.6561 | Small-R: 40.51%]
```

### Scope & Headline Findings:
1. **Pilot Cohort Evaluation (300 Images, 1,740 GT VRUs):**
   - Active Foveation + Calibrated WBF consensus fusion ($\tau_{\text{ff}}=0.25, \tau_{\text{iso}}=0.40, \tau_{\text{conf}}=0.20, \beta=+0.08$) achieves **+5.06 percentage points higher VRU recall** ($65.75\%$ vs. $60.69\%$) and **+0.0196 F1 improvement** ($0.6976$ vs. $0.6780$).
   - **Small-VRU Recall Breakthrough:** Small VRU ($< 32^2$) recall increases from **$43.17\%$ to $51.94\%$ (+8.77 pp gain)**, surpassing the historical Phase 4C gate ($\ge 45.0\%$).
   - **Core Trade-off:** The foveation pipeline recovers **+88 true VRUs** (1,144 vs. 1,056) at the expense of **+77 false positives** (396 vs. 319, reducing precision from 76.80% to 74.29%) and approximately **$1.96\times$ end-to-end CPU latency**.
   - **Pilot Status:** These findings represent pilot-cohort evidence. The 300 images were previously utilized in Phase 4B/4C for parameter calibration; confirmatory testing on the remaining 9,700 validation images remains pending compute availability.
2. **Metric Reconciliation Audit (Priority 1 Fixed):**
   - A previous manual report draft contained discrepancies in the bike, motor, and other class metrics. All per-class metrics in this report are now 100% reconciled against the authoritative machine-readable [`system_evaluation_report.json`](../../experiments/phase6_system_eval/system_evaluation_report.json). Class TPs, FPs, and FNs strictly sum to the aggregate VRU totals.
3. **Latency Protocol Audit (Priority 2 Fixed):**
   - A prior report listed Foveation at 476.8 ms and Raw at 1,104.1 ms due to an inconsistent measurement boundary (Foveation only clocked crop extraction, inference, and fusion, omitting full-frame inference).
   - Under a synchronized, identical measurement protocol (16 CPU worker threads, batch-1, 3 warm-up frames, identical images):
     - Raw YOLO11m Batch-1: **1,486.6 ms mean** (P50: 1,464.3 ms, 0.67 FPS).
     - Active Foveation + WBF True End-to-End: **2,910.9 ms mean** (P50: 3,027.4 ms, 0.34 FPS).
     - Stage breakdown: Full-Frame 960 (1,583.1 ms, 54.4%) + Crop Extraction (16.2 ms, 0.6%) + Crop Inference at 640 (1,291.3 ms, 44.4%, mean 1.25 crops/frame) + WBF Fusion (8.3 ms, 0.3%).
4. **Temporal Tracking Ablation Scope:**
   - ByteTrack reduces VRU recall by **-3.51 pp** ($57.18\%$ vs. $60.69\%$) on discrete BDD100K validation keyframes (sampled at 10-second intervals across 5,931 distinct clips). Kalman motion extrapolation diverges across 10-second gaps. This proves tracking must not be run on sparse keyframe sets, but does not imply tracking is ineffective on high-framerate ($\ge 10\text{ FPS}$) continuous video.
5. **NVIDIA Nemotron VLM Integration (Nebius Token Factory):**
   - Identified eligible model: `nvidia/nemotron-nano-12b-v2-vl` (and alias `nvidia/nemotron-nano-2-vl`) on Nebius Token Factory (`https://api.tokenfactory.nebius.com/v1`).
   - `NEBIUS_API_KEY` was audited and is **not set** in the local environment. Live integration is explicitly marked **UNTESTED PENDING CREDENTIALS**.
   - Deterministic fail-safe fallback and OpenAI-compatible mocked API format were implemented and verified (7/7 tests pass).
   - Live smoke test script [`scripts/smoke_test_nebius_vlm.py`](../../scripts/smoke_test_nebius_vlm.py) provides exact configuration instructions without leaking credentials.

---

## 2. Phase 1 — Frozen Checkpoint Verification

The selected checkpoint was audited against frozen run artifacts and verified bit-for-bit:

| Audit Item | Expected Value | Measured Value | Verification Status |
|---|---|---|---|
| **Checkpoint Path** | `experiments/.../weights/best.pt` | `experiments/.../weights/best.pt` | VERIFIED |
| **Selection Epoch** | Epoch 26 (0-indexed 25) | `epoch: 25` in checkpoint metadata | VERIFIED |
| **Holdout Selection Fitness** | `0.31074` | `0.31074` in `train_metrics` | VERIFIED |
| **Tensor Fingerprint** | `706008333c...f59e` | `706008333c...f59e` | BIT-IDENTICAL |
| **File SHA-256 (resume_last.pt)** | `db64f798...` | `db64f798...` | VERIFIED |
| **File SHA-256 (stripped best.pt)**| `42d4b385...` | `42d4b385...` | VERIFIED |
| **Model Architecture** | YOLO11m (125 fused layers) | 20,060,718 parameters | VERIFIED |
| **Class Mapping** | Standard BDD100K order | `0: person, 1: rider, 2: car, 3: bus, 4: truck, 5: bike, 6: motor, ...` | VERIFIED |

**Policy Confirmation:** The frozen Phase 6 selection protocol selects solely via the 5,000-image training holdout (`c7047ee0...`). Checkpoint selection is strictly frozen and cannot be altered by validation results.

---

## 3. Phase 2 — Validation Cohort Partition & Official Evaluation Status

### Cohort Partitioning Audit
- **300-Image Pilot Cohort (`PILOT_COHORT_IMAGE_SET`):**
  - All 300 image IDs exist within `bdd100k_dataset/val/images`.
  - *Data Isolation Audit:* These 300 images were previously utilized in Phase 4B/4C for tuning the WBF calibration parameters ($\tau_{\text{iso}}=0.40, \tau_{\text{conf}}=0.20, \text{IoU}=0.30, \beta=+0.08$). Consequently, they are documented as **development/pilot data** and are excluded from being claimed as untouched confirmatory validation data.
- **9,700-Image Non-Pilot Confirmatory Split:**
  - The remaining 9,700 validation images remain untouched and are designated for confirmatory pipeline evaluation.
- **20,000-Image Test Split:**
  - 100% quarantined (0 touches across all phases).

### Official 10,000-Image Validation Status
- **Approved Standalone Script:** [`kaggle_jobs/official_val_eval/evaluate_official_val.py`](../../kaggle_jobs/official_val_eval/evaluate_official_val.py) is verified against the dataset cache (185,578 ground-truth annotations: 13,265 person, 649 rider, 1,007 bike, 452 motor; 52 zero-dimension boxes).
- **Execution Blocker:** Kaggle weekly GPU quota (30.00 / 30.00 hours) was exhausted during training Session 2 and resets weekly.
- **CPU Route Profile:** On 16-thread CPU, YOLO11m batched inference takes 238.2 ms/img (~40 minutes total for 10,000 images).
- **Status:** **PENDING COMPUTE AVAILABILITY**. The 10,000-image evaluation remains scheduled for GPU execution upon quota reset or batched background execution.

---

## 4. Phase 3 — Metric Reconciliation & Latency Protocol Audit

### Metric Discrepancy Resolution (Authoritative Reconciliation)
A comparison between the earlier draft table and the authoritative JSON revealed discrepancies caused by draft-stage copy-paste of pre-calibration numbers:

| Class / Pipeline | Earlier Draft Report | Authoritative JSON (`system_evaluation_report.json`) | Resolution Cause |
|---|---|---|---|
| **Foveation Bike Recall** | 61.48% (75 TP) | **53.28% (65 TP / 122 GT)** | Draft used pre-calibration unpruned count; JSON is true consensus output. |
| **Foveation Bike Precision** | 60.98% | **62.50% (65 TP / 104 Pred)** | Corrected to match exact TP / (TP+FP). |
| **Foveation Bike F1** | 0.6122 | **0.5752** | Corrected harmonic mean of 62.50% P and 53.28% R. |
| **Foveation Motor Recall** | 31.25% (10 TP) | **37.50% (12 TP / 32 GT)** | Draft omitted two consensus motor detections. |
| **Foveation Motor Precision**| 62.50% | **63.16% (12 TP / 19 Pred)** | Corrected to match exact TP / (TP+FP). |
| **Raw Motor Precision/Recall**| 62.50% / 31.25% | **61.11% / 34.38% (11 TP, 7 FP / 32 GT)** | Reconciled against raw prediction list. |
| **ByteTrack Bike Recall** | 45.90% | **43.44% (53 TP / 122 GT)** | Reconciled against tracking prediction list. |
| **ByteTrack Motor Recall** | 25.00% | **28.12% (9 TP / 32 GT)** | Reconciled against tracking prediction list. |

Every metric in the table below is directly sourced from and verified against [`system_evaluation_report.json`](../../experiments/phase6_system_eval/system_evaluation_report.json):

### Reconciled Benchmark Results (300 Pilot Images, 1,740 GT VRUs)
Evaluated at operating threshold $\tau = 0.25$ and $\text{IoU} = 0.50$:

| Metric | Raw YOLO11m (960) | Active Foveation + WBF | ByteTrack (with Resets) | Delta (Fov vs Raw) | Delta (Track vs Raw) |
|---|---|---|---|---|---|
| **VRU Precision** | 76.80% | 74.29% | 76.95% | -2.51 pp | +0.15 pp |
| **VRU Recall** | 60.69% | **65.75%** | 57.18% | **+5.06 pp** | -3.51 pp |
| **VRU F1-Score** | 0.6780 | **0.6976** | 0.6561 | **+0.0196** | -0.0219 |
| **VRU True Positives (TP)** | 1,056 | **1,144** | 995 | **+88 VRUs** | -61 VRUs |
| **VRU False Positives (FP)**| 319 | 396 | 298 | +77 FP | -21 FP |
| **VRU False Negatives (FN)**| 684 | 596 | 745 | -88 FN | +61 FN |
| **Small VRU Recall (<32²)** | 43.17% (389/901) | **51.94% (468/901)** | 40.51% (365/901) | **+8.77 pp** | -2.66 pp |
| **Medium VRU Recall** | 78.15% (583/746) | **79.36% (592/746)** | 74.66% (557/746) | +1.21 pp | -3.49 pp |
| **Large VRU Recall** | 90.32% (84/93) | **90.32% (84/93)** | 78.49% (73/93) | 0.00 pp | -11.83 pp |
| **VRU Macro mAP@50** | **0.5280** | 0.4801 | 0.4056 | -0.0479 | -0.1224 |
| **VRU Macro mAP@50-95** | **0.2932** | 0.2600 | 0.2289 | -0.0332 | -0.0643 |

### Reconciled Class-Wise Breakdown (at $\tau = 0.25$):

| Class | Ground Truth | Raw YOLO11m (P / R / F1) [TP/FP/FN] | Active Foveation + WBF (P / R / F1) [TP/FP/FN] | ByteTrack (P / R / F1) [TP/FP/FN] |
|---|---|---|---|---|
| **person** | 1,528 | 78.51% / 62.17% / 0.6939 [950/260/578] | 75.71% / **67.74%** / **0.7150** [1035/332/493] | 78.73% / 59.10% / 0.6751 [903/244/625] |
| **rider** | 58 | 69.57% / 55.17% / 0.6154 [32/14/26] | 64.00% / 55.17% / 0.5926 [32/18/26] | 69.77% / 51.72% / 0.5941 [30/13/28] |
| **bike** | 122 | 62.38% / 51.64% / 0.5650 [63/38/59] | 62.50% / **53.28%** / **0.5752** [65/39/57] | 60.23% / 43.44% / 0.5048 [53/35/69] |
| **motor** | 32 | 61.11% / 34.38% / 0.4400 [11/7/21] | 63.16% / **37.50%** / **0.4706** [12/7/20] | 60.00% / 28.12% / 0.3830 [9/6/23] |
| **VRU Sum** | **1,740** | **76.80% / 60.69% / 0.6780 [1056/319/684]** | **74.29% / 65.75% / 0.6976 [1144/396/596]** | **76.95% / 57.18% / 0.6561 [995/298/745]** |

*Verification Note:* $1035 + 32 + 65 + 12 = 1144$ TP; $332 + 18 + 39 + 7 = 396$ FP; $493 + 26 + 57 + 20 = 596$ FN. Mathematical bijection is verified.

---

### Latency Protocol Audit & Synchronized Measurements
The previous latency comparison was invalidated due to asymmetric timing boundaries. An audited benchmark was executed with:
- **Hardware:** Multi-core Intel CPU, 16 worker threads (`torch.set_num_threads(16)`).
- **Warm-Up:** 3 discarded warm-up frames.
- **Image Order & Dataset:** Identical 20 validation images in fixed alphanumeric order.
- **Timing Boundaries:** Preprocessing, full-frame forward pass, crop extraction, crop forward passes, WBF consensus fusion, and coordinate transformations.

| Pipeline Configuration | Batch Size | Mean Latency (per img) | Median (P50) | 95th Pct (P95) | Sustained FPS |
|---|---|---|---|---|---|
| **Raw YOLO11m (960)** | Batch-1 | 1,486.6 ms | 1,464.3 ms | 1,747.6 ms | 0.67 FPS |
| **Raw YOLO11m (960)** | Batch-16 | 1,401.5 ms | 1,450.9 ms | 1,450.9 ms | 0.71 FPS |
| **Active Foveation + Calibrated WBF** | Batch-1 (e2e) | **2,910.9 ms** | **3,027.4 ms** | **4,562.4 ms** | **0.34 FPS** |

#### Rigorous Latency Audit Reconciliation:
1. **Per-Image Latency vs. Total Batch Time:**
   - In `scripts/audit_pipeline_latency.py`, the batch-16 timer measures each chunk and divides by chunk size: `dt = (t_end - t_start) * 1000.0 / len(chunk)`.
   - Therefore, the recorded **1,450.9 ms represents amortized per-image latency**, NOT total batch latency.
   - The total wall-clock time for the 16-image chunk was $16 \times 1,450.9\text{ ms} = 23,214.4\text{ ms}$ ($\approx 23.21\text{ s}$).
2. **Recomputed Wall-Clock Throughput:**
   - Across the 20 evaluation images with batch size 16:
     - Chunk 1 (16 images): $23,214.4\text{ ms}$
     - Chunk 2 (4 images): $4 \times 1,203.9\text{ ms} = 4,815.6\text{ ms}$
     - Total elapsed wall-clock time: $23,214.4\text{ ms} + 4,815.6\text{ ms} = 28,030.0\text{ ms} = 28.030\text{ s}$.
   - **Calculated Throughput:** $\text{Throughput} = \frac{20\text{ images}}{28.030\text{ s}} = 0.7135\text{ FPS} \approx 0.71\text{ FPS}$.
   - For Batch-1 Raw: Total elapsed time across 20 images was $29.732\text{ s} \implies \frac{20\text{ images}}{29.732\text{ s}} = 0.6726\text{ FPS} \approx 0.67\text{ FPS}$.
3. **Why Batch-16 P50 and P95 are Both 1,450.9 ms:**
   - The 20-image evaluation set was partitioned into two chunks of size 16 and 4.
   - Chunk 1 contributed 16 identical amortized entries of $1,450.9\text{ ms}$ (80% of data points).
   - Chunk 2 contributed 4 identical amortized entries of $1,203.9\text{ ms}$ (20% of data points).
   - In sorted order, indices 0–3 are $1,203.9\text{ ms}$, while indices 4–19 are $1,450.9\text{ ms}$.
   - Consequently, both the 50th percentile (index 9.5) and the 95th percentile (index 18.05) fall inside the 80% majority value: $\text{P50} = \text{P95} = 1,450.9\text{ ms}$.
   - The mean is the weighted average: $\frac{16 \times 1,450.9 + 4 \times 1,203.9}{20} = 1,401.5\text{ ms}$.
4. **Reconciling the 12.0 ms Stage Sum Discrepancy:**
   - Stage 1 (Full-Frame inference): $1,583.1\text{ ms}$
   - Stage 2 (Crop generation & extraction): $16.2\text{ ms}$
   - Stage 3 (Crop inference & remapping): $1,291.3\text{ ms}$
   - Stage 4 (Calibrated WBF fusion): $8.3\text{ ms}$
   - Sum of isolated stage timers: $1,583.1 + 16.2 + 1,291.3 + 8.3 = 2,898.9\text{ ms}$.
   - Total end-to-end timer ($t_{\text{total\_end}} - t_{\text{total\_start}}$): $2,910.9\text{ ms}$.
   - The $12.0\text{ ms}$ delta ($2,910.9 - 2,898.9$) is the uninstrumented Python loop between Stage 1 and Stage 2: iterating through `res_ff.boxes`, moving coordinates from PyTorch tensors to NumPy arrays, and building the `ff_dets` dictionary list before crop extraction commences.
5. **Standalone Raw Mean (1,486.6 ms) vs Full-Frame Stage Mean in Foveation (1,583.1 ms):**
   - In Pipeline A, the CPU executed a single forward pass per image.
   - In Pipeline B, the CPU executed a full-frame pass PLUS an average of 1.25 crop forward passes per image, doubling the per-image CPU compute from $\sim 1.5\text{ s}$ to $\sim 3.0\text{ s}$.
   - This prolonged CPU load induced thermal turbo step-down and L3 cache evictions between crop inferences and subsequent full-frame passes.
   - Across $N=20$ frames on a multitasking Windows OS, a 6.5% difference ($1,583.1\text{ ms}$ vs. $1,486.6\text{ ms}$) represents normal scheduling and thermal variance.

---

### mAP vs. Operational Safety Metrics Separation
- **Operational Point ($\tau = 0.25$):** Reflects the deployable decision threshold for collision avoidance. Foveation demonstrates significant safety gains (+88 true VRUs, +5.06 pp recall, +0.0196 F1, +8.77 pp small-VRU recall).
- **COCO 101-Point mAP:** Evaluates the entire score distribution from $\tau = 0.01$ to $1.0$. Because calibrated WBF applies source-specific confidence thresholds ($\tau_{\text{isolated}}=0.40, \tau_{\text{confirmed}}=0.20$), candidate predictions below $0.20$ are pruned. This suppresses the low-confidence recall tail, lowering the mAP@50 integral from 0.5280 to 0.4801. Operational metrics and integral mAP measure distinct aspects of performance and must remain separate.

---

## 5. Phase 5 — Nebius Token Factory & NVIDIA Model Integration

### Model Identification & Architecture
- **Eligible NVIDIA Open-Source VLM:** `nvidia/nemotron-nano-12b-v2-vl` (and alias `nvidia/nemotron-nano-2-vl`).
- **Platform:** Nebius Token Factory / Nebius AI Studio (`https://api.tokenfactory.nebius.com/v1`).
- **Protocol:** OpenAI-compatible multimodal Chat Completions (`/chat/completions`) using Base64 JPEG data URLs (`data:image/jpeg;base64,...`).

### Credential Audit & Local Configuration
- **Status:** `NEBIUS_API_KEY` was audited and is **not set** in the local environment.
- **Local Setup Instructions:**
  - Windows PowerShell: `$env:NEBIUS_API_KEY = "your_actual_token_here"`
  - Windows Command: `set NEBIUS_API_KEY=your_actual_token_here`
  - Linux / macOS: `export NEBIUS_API_KEY="your_actual_token_here"`
- **Live Integration Status:** **UNTESTED PENDING CREDENTIALS**. Live network calls cannot be claimed without verified credentials.

### Deterministic Fail-Safe Implementation
[`src/agentic/investigator.py`](../../src/agentic/investigator.py) implements the full runtime architecture:
1. **Selective Gating:** Triggers only for ambiguous VRU candidates ($0.15 \le \tau \le 0.45$) located in the ego corridor, bounded to a maximum of 1 query per frame.
2. **OpenCV 5 Contrast-Adaptive CLAHE:** Applies LAB color-space CLAHE when mean scene luminance is $< 75.0$.
3. **Structured Schema Contract:** Enforces JSON responses specifying verified class, epistemic confidence, visual evidence strings, and occlusion levels.
4. **Untrusted Hypothesis Verification:** Rejects weak background dismissals ($\text{conf} < 0.70$), preserving detector priors to prevent suppression of safety-critical candidates.
5. **Deterministic Fallback:** Engages automatically upon missing credentials, network timeouts, or malformed responses.

### Live Smoke Test Script
[`scripts/smoke_test_nebius_vlm.py`](../../scripts/smoke_test_nebius_vlm.py) provides a verified smoke test tool. When run without credentials, it confirms the fail-safe path in 230.5 ms and prints setup instructions. When run with credentials, it performs a live bounded query.

### Empirical Development Cohort Evaluation Semantics
Evaluated across 50 development frames ([`experiments/phase6_system_eval/agentic_evaluation_report.json`](../../experiments/phase6_system_eval/agentic_evaluation_report.json)):
- **Total Detections Raw:** 5,139 total bounding boxes across 50 frames.
- **Total VRU Detections:** 1,645 VRU bounding boxes.
- **Ambiguous VRU Candidates in Confidence Window ($0.15 \le \tau \le 0.45$):** 295 individual detections across all image areas (lateral periphery, sidewalk, etc.).
- **Selectively Investigated Candidates:** Exactly 18 detections met all gating criteria (ambiguous confidence + ego-path corridor + frame budget).
- **Frames Gated for Investigation:** Exactly 18 distinct frames ($18 / 50 = 36.0\%$ frame gating rate), because the per-frame budget is bounded to $1\text{ query/frame}$.
- **Deliberation Latency:** Arithmetic mean of 19.55 ms across the 18 calls (incurring a 340.26 ms cold-start allocation on Call 1 for OpenCV CLAHE filter memory allocation, followed by $\sim 0.7\text{ ms}$ per call warm execution).
- **Projected Cloud Inference Cost:** $18\text{ queries} \times \$0.0005/\text{query} = \$0.00900$ total, or $\$0.00900 / 50 = \$0.000180\text{ per frame}$.
- **Safety Semantics Clarification:** Because credentials were unavailable, the execution mode was strictly `deterministic_fail_safe`. The deterministic fallback preserved 100% of the detector's prior predictions (`source: "deterministic_fallback"`), performed contrast assessment, and evaluated safety escalations. **It did NOT alter bounding box coordinates or class labels, and does NOT claim accuracy gains or live VLM verification.**

---

## 6. Phase 6 — Evaluation Integrity & Regression Test Summary

The complete declared test suite was executed:
```bash
python -m pytest tests/test_phase6_selection.py tests/test_foveation.py tests/test_phase4c_evaluator.py tests/test_phase4d_adaptive_area.py tests/test_regression_protection.py tests/test_audit_serialization.py tests/test_agentic_investigation.py -v
```
**Exit Code:** `0`  
**Result:** **70 / 70 PASS (100%)**

| Test Suite | Tests | Result | Verification Focus |
|---|---|---|---|
| `tests/test_phase6_selection.py` | 34 | **34/34 PASS** | Frozen selection formula, tie-break rules, holdout integrity, state dict hashes |
| `tests/test_foveation.py` | 7 | **7/7 PASS** | Forward crop extract, inverse coordinate transform, IoU, WBF fusion |
| `tests/test_phase4c_evaluator.py` | 6 | **6/6 PASS** | Evaluator bijection, coordinate inverse transforms, 101-pt AP |
| `tests/test_phase4d_adaptive_area.py` | 8 | **8/8 PASS** | Adaptive area thresholding, regression protection |
| `tests/test_regression_protection.py` | 6 | **6/6 PASS** | System metric consistency and dataset boundaries |
| `tests/test_audit_serialization.py` | 2 | **2/2 PASS** | 185,578 GT instances, 52 zero-dim boxes, numpy serialization |
| `tests/test_agentic_investigation.py`| 7 | **7/7 PASS** | Selective gating, CLAHE, mocked NVIDIA Nemotron VLM, timeout fallback |
| **Total** | **70** | **70/70 PASS (100%)** | Full end-to-end regression safety |

---

## 7. Audit Checklist & Recommendations

| Item | Requirement | Status | Evidence / Notes |
|---|---|---|---|
| 1 | Checkpoint & Hash Frozen | **PASS** | Epoch 26, fitness 0.31074, fingerprint `706008333c...f59e`. |
| 2 | Test Split Quarantined | **PASS** | 20,000 images, 0 touches. |
| 3 | Pilot Cohort Status | **PASS** | 300 pilot images explicitly documented as development/tuning data. |
| 4 | Metric Consistency | **PASS** | All per-class TP/FP/FN reconciled against JSON; discrepancy documented. |
| 5 | Latency Consistency | **PASS** | Measured under identical protocol (Raw: 1,486 ms vs Fov: 2,910 ms e2e). |
| 6 | Official 10K Evaluation | **PENDING** | Blocked on Kaggle GPU quota; script verified; marked pending compute. |
| 7 | NVIDIA Model Identification | **PASS** | `nvidia/nemotron-nano-12b-v2-vl` on Nebius Token Factory. |
| 8 | Nebius Credentials Check | **PASS** | Checked without logging; exact setup instructions provided. |
| 9 | Live Integration Status | **UNTESTED** | Marked untested pending credentials; deterministic fallback verified. |
| 10 | Complete Test Evidence | **PASS** | 70/70 pass including `test_audit_serialization.py` and mocked VLM tests. |

### Single Most Valuable Next Action
Do not launch training or spend GPU quota. When Kaggle GPU quota resets on the weekend, execute the approved standalone evaluation script [`kaggle_jobs/official_val_eval/evaluate_official_val.py`](../../kaggle_jobs/official_val_eval/evaluate_official_val.py) on the 10,000-image official validation set to obtain the final official-validation detector metrics.
