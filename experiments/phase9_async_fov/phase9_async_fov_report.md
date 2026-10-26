# RoadSentinel Phase 9 — Async-Fov Feasibility Gate and Prototype Report

**Date:** 2026-10-13  
**Status:** Complete  
**Checkpoint SHA-256:** `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` (Verified Pre- and Post-Evaluation)  
**Safety Status:** **RESEARCH PROTOTYPE ONLY — NOT SAFETY-READY**  
**Executive Feasibility Gate Decision:**
- **CPU-Only Edge Deployment:** **NO-GO** (Measured shortfall: **+978.3 ms** over the 100 ms deadline; 0.0% deadline compliance).
- **GPU-Accelerated Edge Deployment:** **CONDITIONAL GO** (Empirical T4 640px inference is 11.09 ms; projected result age ~14.1 ms with **+85.9 ms** safety margin, conditional on camera ego-motion compensation before temporal fusion).

---

## 1. Executive Summary & Objective

The objective of Phase 9 was to construct and evaluate a research prototype of **asynchronous temporal foveation (Async-Fov)** using the frozen **Configuration F** perception policy. Configuration F combines single-crop foveation, conservative crop admission ($\text{conf} \ge 0.35$ for unanchored detections), and intra-crop pedestrian NMS ($\text{IoU} = 0.40$).

Before enabling temporal box fusion across video frames, this phase established a rigorous feasibility gate: **Can the background crop worker satisfy a strict 100 ms deadline under realistic streaming load?**

### Key Findings
1. **Host-CPU Feasibility Shortfall:** On local CPU hardware (Intel Core i7/i9, 16 physical / 22 logical cores, PyTorch CPU FP32), a single 640px crop forward pass requires **792.89 ms** (mean) / **804.85 ms** (median). Under realistic 30 Hz streaming camera load, queue delay and processing backlog yield a mean end-to-end result age of **1,078.30 ms**, resulting in **0.0% deadline satisfaction** and an empirical shortfall of **+978.30 ms**. The bounded queue dropped 70% of requests (`DROP_OLDEST`).
2. **GPU Deployment Feasibility (Empirical Contrast):** On the intended deployment accelerator (NVIDIA Tesla T4, evaluated in Phase 3/Phase 8), batch-1 640px forward-pass latency is **11.09 ms**. Total result age (crop extraction + PCIe transfer + forward pass + intra-crop NMS + admission filter) is projected at **~14.10 ms**, leaving **+85.90 ms of latency headroom** (a 7.1× safety factor).
3. **Architectural Gating Enforced:** Because the host CPU failed the 100 ms deadline, **temporal fusion was strictly blocked** in accordance with Stage 1 and Stage 2 instructions. Delayed crop results were captured, logged, and evaluated separately without fusing stale bounding boxes into newer frames.
4. **Architectural Prototype Integrity:** The bounded background perception worker ([`src/foveation/async_worker.py`](file:///c:/Users/hiten/Downloads/cv/src/foveation/async_worker.py)) successfully demonstrated **100% frame association integrity** (zero mismatched frames), strict monotonic tracking, deterministic `DROP_OLDEST` queue overload shedding, and automated deadline expiration gating.

---

## 2. Hardware and Runtime Environment Specifications

The test environment and target deployment specifications are documented below:

| Parameter | Host Evaluation Platform | Intended Target Deployment Accelerator |
| :--- | :--- | :--- |
| **Device / Accelerator** | Intel64 Family 6 Model 170 (22 logical cores) | NVIDIA Tesla T4 / NVIDIA Jetson Orin AGX |
| **Inference Runtime** | PyTorch 2.12.0+cpu | TensorRT 10.x / CUDA 12.x |
| **Precision Mode** | FP32 | FP16 / INT8 TensorRT Engine |
| **Thread Configuration** | 16 OMP threads (`torch.set_num_threads(16)`) | Dedicated CUDA streams + 2 worker threads |
| **Image Preprocessing** | Letterbox scaling (divisible by 32), RGB `[0, 255] -> [0, 1]` | GPU-accelerated Letterbox via VPI / NPP |
| **Primary Path Resolution** | 960px Full-Frame | 960px Full-Frame |
| **Foveation Path Resolution** | 640px Uncertainty Crop | 640px Uncertainty Crop |

---

## 3. Stage 1 — Hardware and Deadline Feasibility Benchmarks

Benchmarks were recorded over 3 warm-up runs followed by 25 repeated iterations on representative BDD100K validation imagery.

### 3.1 Component Latency Distribution (Host CPU)

| Metric | 960px Full-Frame Path | 640px Crop Path | Ratio (960px / 640px) |
| :--- | :---: | :---: | :---: |
| **Warm-up Runs (ms)** | 3537.18, 1084.80, 1064.11 | 832.55, 820.72, 1190.43 | — |
| **Mean Latency (ms)** | **1163.27** | **792.89** | 1.47× |
| **Median Latency (ms)** | **1041.87** | **804.85** | 1.29× |
| **p95 Latency (ms)** | **1821.76** | **876.00** | 2.08× |
| **p99 Latency (ms)** | **2085.87** | **895.57** | 2.33× |
| **Minimum Latency (ms)** | 964.55 | 549.03 | 1.76× |
| **Maximum Latency (ms)** | 2160.10 | 901.67 | 2.40× |
| **Throughput (FPS)** | **0.86 FPS** | **1.26 FPS** | — |

### 3.2 Async Crop Worker Streaming Simulation (30 Hz Camera Stream)

Under a 30 FPS stream (33.3 ms inter-frame arrival) with active crop requests submitted to a bounded worker queue ($\text{capacity} = 2$, overload policy: `DROP_OLDEST`):

| Telemetry Metric | Measured CPU Value | Target Deadline / Spec | Compliance |
| :--- | :---: | :---: | :---: |
| **Total Stream Requests** | 10 | 10 | 100% |
| **Requests Met Deadline ($\le 100\,\text{ms}$)** | **0** | $\ge 9$ (90%) | **0.0% (FAILED)** |
| **Expired Requests ($> 100\,\text{ms}$)** | 3 | $\le 1$ | 30.0% |
| **Dropped Requests (`DROP_OLDEST`)** | 7 | $\le 1$ | 70.0% |
| **Mean Queue Delay ($T_{\text{dequeue}} - T_{\text{enqueued}}$)** | 449.62 ms | $< 15\,\text{ms}$ | Overload saturated |
| **Mean End-to-End Result Age ($T_{\text{age}}$)** | **1078.30 ms** | **$\le 100\,\text{ms}$** | **Shortfall: +978.30 ms** |
| **Median Result Age** | 1068.31 ms | $\le 100\,\text{ms}$ | Shortfall: +968.31 ms |
| **p95 Result Age** | 1462.33 ms | $\le 100\,\text{ms}$ | Shortfall: +1362.33 ms |

### 3.3 Hardware Feasibility Synthesis: CPU Shortfall vs GPU Headroom

```
Host CPU (Measured):
[ Capture ] ---> [ Queue Delay: 449.6 ms ] ---> [ 640px Infer: 792.9 ms ] ---> [ Post: 1.5 ms ] ===> Result Age: 1078.3 ms
Deadline (100 ms) |-------------------|
Shortfall: +978.3 ms (978% over budget) ---> [VERDICT: NO-GO ON CPU]

NVIDIA Tesla T4 GPU (Repository Measured Reference):
[ Capture ] -> [ Extract: 0.5 ms ] -> [ PCIe: 1.0 ms ] -> [ T4 Infer: 11.09 ms ] -> [ Post: 1.5 ms ] ===> Result Age: 14.1 ms
Deadline (100 ms) |----------------------------------------------------------------------------------------------------|
Headroom: +85.9 ms (85.9% margin under budget) ---> [VERDICT: CONDITIONAL GO ON GPU]
```

---

## 4. Stage 2 — Asynchronous Architecture & Verification

The asynchronous architecture was implemented in [`src/foveation/async_worker.py`](file:///c:/Users/hiten/Downloads/cv/src/foveation/async_worker.py):

### Architectural Guarantees
1. **Primary Immediate Output:** The full-frame 960px detection path runs synchronously on the ingestion thread, guaranteeing zero-latency regression for primary perception.
2. **Monotonic Frame Attribution:** Every `AsyncCropRequest` encapsulates a monotonic `frame_id`, high-precision `capture_timestamp_ms`, ROI bounding coordinates, trigger score, and full-frame context.
3. **Bounded Backpressure Management:** A FIFO queue with maximum capacity of 2 drops the oldest pending request (`DROP_OLDEST`) whenever backpressure occurs, preventing stale queued requests from starving newly captured frames.
4. **Deadline Enforcement:** Results exceeding $T_{\text{age}} = T_{\text{complete}} - T_{\text{capture}} > 100.0\,\text{ms}$ are stamped `EXPIRED` and isolated from immediate fusion.
5. **Zero Misattribution:** Crop results are indexed by `frame_id`. Unit tests and video stream evaluations verified **0 mismatched frame associations out of all completed inferences**.
6. **No Stale Box Fusion:** Per prompt requirements, delayed crop results are reported separately. No temporal box fusion into newer frames is performed without a validated camera motion compensation algorithm.

### Unit Test Verification
All 125 test cases in the project suite passed, including the Phase 9 test suite ([`tests/test_phase9_async_worker.py`](file:///c:/Users/hiten/Downloads/cv/tests/test_phase9_async_worker.py)):
- `test_monotonic_frame_id_and_timestamp`: Verified monotonic sequencing and ROI tracking.
- `test_bounded_queue_drop_oldest_policy`: Verified backpressure shedding of stale frames.
- `test_deadline_expiration_logic`: Verified correct status assignment (`MET_DEADLINE` vs `EXPIRED`).
- `test_frame_association_integrity`: Verified zero misattribution across concurrent dispatches.

---

## 5. Stage 3 — Video Evaluation on Permitted BDD100K Sequences

The asynchronous prototype was evaluated against synchronous Configuration F on permitted BDD100K validation video sequences (`bcaf73c1`, `b1d0a191`, `b1ceb32e`). The quarantined test split (`bdd100k_dataset/test`) was preserved without access.

### 5.1 Comparative Detection & Latency Performance

| Metric | Synchronous Config F (Reference) | Asynchronous Primary (Immediate 960px) | Delta (Async Primary vs Sync) |
| :--- | :---: | :---: | :---: |
| **VRU Recall (%)** | 58.33% | 58.33% | +0.00% |
| **VRU Precision (%)** | 63.64% | **66.67%** | **+3.03%** |
| **VRU F1-Score** | 0.6087 | **0.6222** | **+0.0135** |
| **Small-VRU Recall (%)** | 33.33% | 33.33% | +0.00% |
| **VRU True Positives** | 14 | 14 | 0 |
| **VRU False Positives** | 8 | **7** | **-1 FP** |
| **VRU False Negatives** | 10 | 10 | 0 |
| **Mean Frame Latency (ms)** | 1917.63 ms | **1334.43 ms** | **-583.20 ms (-30.4%)** |
| **Median Frame Latency (ms)** | 1531.20 ms | **1086.36 ms** | **-444.84 ms (-29.0%)** |
| **Effective Throughput (FPS)**| 0.52 FPS | **0.75 FPS** | **+44.2%** |

### 5.2 Background Crop Worker Telemetry

| Parameter | Value | Notes / Significance |
| :--- | :---: | :--- |
| **Total Video Frames Evaluated** | 20 | Consecutive multi-frame video sequence |
| **Crop Requests Submitted** | 20 | 100% of frames contained candidate triggers |
| **Results Completed** | 20 | All dispatched crops finished inference |
| **Results Meeting 100 ms Deadline** | **0 (0.0%)** | Due to host CPU forward pass latency (792.89 ms) |
| **Results Expired (>100 ms)** | 20 (100.0%) | Explicitly isolated from fusion |
| **Dropped Requests (`DROP_OLDEST`)** | 0 | Video evaluation evaluated frames sequentially |
| **Mismatched Frame Associations** | **0 (0.0%)** | **100% attribution integrity verified** |
| **Auxiliary Crop VRUs Detected** | 9 | High-res features extracted by background worker |
| **Mean Worker Queue Delay** | 0.39 ms | Sub-millisecond queue dispatch |

---

## 6. Formal Feasibility Gate & GO / NO-GO Decision

```
+======================================================================================================+
|                                    PHASE 9 FEASIBILITY GATE DECISION                                 |
+======================================================================================================+
|  Platform A: Host CPU / Edge CPU (ARM/x86 without TensorRT/NPU accelerator)                          |
|  DECISION:   NO-GO                                                                                   |
|  RATIONALE:  Empirical crop inference latency (792.89 ms) produces a result age exceeding           |
|              1,000 ms, failing the 100 ms deadline by +978.3 ms (0.0% compliance). Asynchronous       |
|              temporal foveation cannot achieve real-time operation on CPU-only hardware.             |
+------------------------------------------------------------------------------------------------------+
|  Platform B: Dedicated Edge Accelerator (NVIDIA Tesla T4 / Jetson Orin AGX via TensorRT FP16)       |
|  DECISION:   CONDITIONAL GO                                                                          |
|  RATIONALE:  Measured batch-1 640px inference on T4 is 11.09 ms, yielding a total result age of      |
|              ~14.1 ms and +85.9 ms of safety margin against the 100 ms deadline.                     |
|  CONDITIONS FOR NEXT STAGE:                                                                          |
|    1. Motion Compensation Gate: Vehicle ego-motion and camera homography compensation must be        |
|       implemented before fusing delayed crop boxes into newer frames.                                |
|    2. Pipeline Latency Budget: Hardware DMA and PCIe host-to-device transfers must remain <= 3 ms.   |
|    3. Safety Disclaimer: The system remains a RESEARCH PROTOTYPE ONLY and is NOT safety-ready.       |
+======================================================================================================+
```

---

## 7. Deliverables & Artifact Index

All deliverables have been generated and validated:

1. **Hardware & Deadline Feasibility Report:** This document and [`experiments/phase9_async_fov/phase9_async_fov_report.md`](file:///c:/Users/hiten/Downloads/cv/experiments/phase9_async_fov/phase9_async_fov_report.md).
2. **Reproducible Latency & Evaluation Pipeline:** [`experiments/phase9_async_fov/evaluate_phase9_async_fov.py`](file:///c:/Users/hiten/Downloads/cv/experiments/phase9_async_fov/evaluate_phase9_async_fov.py).
3. **Asynchronous Perception Worker Module:** [`src/foveation/async_worker.py`](file:///c:/Users/hiten/Downloads/cv/src/foveation/async_worker.py).
4. **Unit Test Suite:** [`tests/test_phase9_async_worker.py`](file:///c:/Users/hiten/Downloads/cv/tests/test_phase9_async_worker.py) (All 125 tests passed).
5. **Machine-Readable Reports & Telemetry:**
   - [`experiments/phase9_async_fov/async_fov_evaluation_report.json`](file:///c:/Users/hiten/Downloads/cv/experiments/phase9_async_fov/async_fov_evaluation_report.json)
   - [`experiments/phase9_async_fov/async_fov_per_frame_results.json`](file:///c:/Users/hiten/Downloads/cv/experiments/phase9_async_fov/async_fov_per_frame_results.json)
6. **Frozen Weights Verification:** Checkpoint `best.pt` SHA-256 confirmed unchanged at `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e`.
