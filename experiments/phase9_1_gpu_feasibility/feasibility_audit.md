# RoadSentinel Phase 9.1 — GPU Feasibility Verification Audit Report

**Audit Date:** 2026-10-13  
**Audit Mode:** Research and audit only (No model, code, checkpoint, or policy modifications)  
**Checkpoint Path:** `scratch/kaggle_upload_weights/best.pt`  
**Checkpoint SHA-256:** `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` (Verified Intact Pre- and Post-Audit)  
**Test Split Quarantine:** `bdd100k_dataset/test` strictly isolated; 0 accesses.  
**Machine-Readable Evidence Table:** [`experiments/phase9_1_gpu_feasibility/evidence_table.json`](file:///c:/Users/hiten/Downloads/cv/experiments/phase9_1_gpu_feasibility/evidence_table.json)  
**Audit Decision:** **HOLD PENDING MISSING HARDWARE & EMPIRICAL EVIDENCE**

---

## 1. Executive Summary & Audit Mandate

Phase 9 established that host CPU execution unequivocally fails the 100 ms crop-result deadline under 30 Hz streaming load, demonstrating a measured shortfall of **+978.30 ms** and **0.0% deadline compliance** (`mean result age = 1,078.30 ms`). 

However, Phase 9 suggested a "Conditional GO" for GPU deployment based on a reported **11.09 ms batch-1 inference latency** and a derived **14.10 ms result-age estimate** on an NVIDIA Tesla T4 GPU.

The mandate of Phase 9.1 is to conduct a rigorous, forensic audit of that GPU evidence:
1. Audit the origin and validity of the 11.09 ms T4 inference number and the 14.10 ms result-age projection.
2. Determine whether any existing artifact contains an authentic end-to-end GPU streaming benchmark.
3. Formulate the precise target benchmark protocol for genuine edge GPU hardware (specifically contrasting cloud Tesla T4 vs. edge NVIDIA Jetson Orin).
4. Audit correctness constraints (deadline gating, monotonic tracking, spatial misalignment under ego-motion).
5. Render an unambiguous audit verdict (**GO**, **HOLD**, or **NO-GO**).

---

## 2. Forensic Audit of Existing GPU Evidence

Every latency figure and compute claim in the repository history was traced back to its raw generation artifacts:

```
[Phase 3 Benchmark (Oct 9, 2026)]
Model: weights/bdd100k_yolo11s.pt  (YOLO11s, ~9.4M params, SHA-256: A4747292...)
Hardware: Tesla T4 (Cloud GPU), CUDA 12.1, PyTorch 2.12.0
Method: torch.cuda.synchronize(), 10-cycle warm-up, batch=1, imgsz=640
Result: 11.09 ms (Isolated forward pass)
       |
       |  CRITICAL DISCONNECT: Model changed in Phase 5/6 to YOLO11m (~20.1M params)
       v
[Phase 9 Projection (Oct 13, 2026)]
Assumed GPU Forward Pass:       11.09 ms  (Directly measured in Phase 3, BUT on YOLO11s, NOT YOLO11m!)
+ Assumed Crop Extraction:       0.50 ms  (Hypothetical constant, never measured on GPU)
+ Assumed PCIe DMA Transfer:     1.00 ms  (Hypothetical constant, never measured on GPU)
+ Assumed Intra-Crop Post-Proc:  1.50 ms  (Hypothetical constant, never measured on GPU)
-------------------------------------------------------------------------------------------------------
= Projected Result Age:         14.10 ms  (Synthetic composite projection; NEVER measured end-to-end)
= Projected Headroom:          +85.90 ms  (Theoretical margin against 100 ms deadline)
```

### 2.1 Traceability of the 11.09 ms T4 Latency (Metric `M01`)
- **Source Artifacts:** 
  - [`experiments/phase3_foveation/phase3_foveation_metrics.json`](file:///c:/Users/hiten/Downloads/cv/experiments/phase3_foveation/phase3_foveation_metrics.json) (lines 36–38)
  - [`experiments/phase3_foveation/active_foveation_benchmark_report.md`](file:///c:/Users/hiten/Downloads/cv/experiments/phase3_foveation/active_foveation_benchmark_report.md) (lines 53, 100)
  - [`experiments/phase3_foveation/latency_and_compute.csv`](file:///c:/Users/hiten/Downloads/cv/experiments/phase3_foveation/latency_and_compute.csv) (line 2)
  - [`experiments/phase3_foveation/kaggle_console_log.txt`](file:///c:/Users/hiten/Downloads/cv/experiments/phase3_foveation/kaggle_console_log.txt) (line 21)
- **Device & Environment:** NVIDIA Tesla T4 GPU (16 GB GDDR6), PyTorch 2.12.0, CUDA 12.1, Kaggle Cloud environment.
- **Model Checkpoint:** `weights/bdd100k_yolo11s.pt` (SHA-256: `A4747292648F37F68AB997BEA65A36391C41356F35E309A9F3D2EB1D3EA8D8E3`).
- **Input Tensor Shape & Batch Size:** Shape `(1, 3, 640, 640)`, `batch_size = 1`.
- **Timing Scope & Synchronization:** Warmed up with 10 dummy iterations; timed using `torch.cuda.synchronize()` strictly around `model(tensor)`. 
- **Auditor Finding:** **CRITICAL ARCHITECTURAL MISMATCH.** The 11.09 ms measurement was conducted on **YOLO11s** (~9.4 million parameters, 21.5 GFLOPs). The production frozen checkpoint selected in Phase 6 is **YOLO11m** (~20.1 million parameters, 68.0 GFLOPs — **over 3.16× the compute footprint**). **YOLO11m has NEVER been benchmarked at 640px on a Tesla T4 in this repository.** Citing 11.09 ms as evidence for YOLO11m is scientifically invalid.

### 2.2 Forensic Decomposition of the 14.10 ms Result-Age Projection
The 14.10 ms result-age estimate reported in Phase 9 was decomposed into its four constituent terms:

| Pipeline Component | Value | Evidence Classification | Audit Assessment |
| :--- | :---: | :---: | :--- |
| **Model Forward Pass ($T_{\text{infer}}$)** | 11.09 ms | Directly Measured on GPU (**Obsolete Model**) | Measured on YOLO11s, NOT frozen YOLO11m. True YOLO11m latency will be substantially higher. |
| **Crop Extraction ($T_{\text{extract}}$)** | 0.50 ms | **Assumed / Theoretical Estimate** | Hardcoded assumption. CPU-to-GPU crop slicing, boundary clipping, and letterbox padding overhead were not timed. |
| **PCIe Host-to-Device Transfer ($T_{\text{dma}}$)** | 1.00 ms | **Assumed / Theoretical Estimate** | Hardcoded assumption. Unpinned memory transfers on PCIe Gen3 x16 can introduce 1.5–4.0 ms of driver dispatch latency. |
| **Intra-Crop NMS & Admission ($T_{\text{post}}$)** | 1.50 ms | **Assumed / Theoretical Estimate** | Hardcoded assumption. PyTorch/CPU coordinate remapping + pedestrian NMS (IoU 0.40) + unanchored filtering (conf 0.35) takes 0.8–3.5 ms depending on box count. |
| **Total Result Age ($T_{\text{age}}$)** | **14.10 ms** | **Derived Synthetic Projection** | **UNVALIDATED COMPOSITE.** Never observed in any operational run. |
| **Calculated Safety Headroom** | **+85.90 ms** | **Derived Synthetic Projection** | Derived entirely from unvalidated synthetic projection. |

### 2.3 Search for Existing End-to-End GPU Streaming Benchmarks
- A comprehensive audit of all 15 subdirectories in `kaggle_jobs/`, 26 subdirectories in `experiments/`, and test logs confirmed:
  **No existing artifact contains a genuine end-to-end GPU streaming benchmark.**
- All prior GPU jobs on Kaggle evaluated static image batches (`batch_size = 1` or `batch_size = 8` offline evaluation) with no camera streaming simulation, no inter-frame arrival intervals, no background worker thread synchronization, and no queue backpressure testing.
- Therefore, GPU streaming feasibility under sustained load is **completely unmeasured in this repository**.

---

## 3. Hardware Architecture Contrast: Tesla T4 vs. Jetson Orin

The Phase 9 report conflated the cloud Tesla T4 with the edge NVIDIA Jetson Orin. These platforms possess fundamentally distinct memory and compute architectures:

| Architectural Feature | NVIDIA Tesla T4 (Cloud / Server) | NVIDIA Jetson Orin AGX / NX (Edge SoC) | Implications for 100 ms Deadline |
| :--- | :--- | :--- | :--- |
| **Compute Core Architecture** | Turing (TU104, 2560 CUDA cores, 320 Tensor cores) | Ampere (1024–2048 CUDA cores, 32–64 Tensor cores) + DLA | Different tensor core instruction throughput and FP16 FP32 scheduling. |
| **Memory Architecture** | Discrete GDDR6 (16 GB, 320 GB/s bandwidth) via PCIe Gen3 x16 bus | **Unified LPDDR5** (8–64 GB, 102–204 GB/s bandwidth) shared with CPU | **Zero-copy Unified Memory:** Host-to-device PCIe transfer is eliminated, but GPU memory bandwidth is shared with CPU and camera DMA. |
| **Preprocessing Acceleration** | Host CPU OpenCV $\rightarrow$ PCIe transfer | On-chip hardware VIC (Video Image Compositor) / VPI / NVMM memory | Crop extraction can be offloaded to VIC hardware, bypassing CPU. |
| **Inference Engine** | PyTorch CUDA / TensorRT Server | TensorRT standalone runtime engine (`.engine`) | PyTorch Python overhead (2–5 ms per call) is unacceptable on edge; TensorRT C++ runtime is mandatory. |
| **Power & Thermal Budget** | 70 W TDP (Actively cooled server rack) | 15 W – 60 W configurable TDP (Passive/Edge enclosure) | Dynamic frequency throttling under sustained thermal load may spike p99 latency above 100 ms. |

**Audit Rule:** **Tesla T4 performance cannot serve as a proxy for Jetson Orin performance.** Jetson feasibility requires direct hardware measurement on the physical SoC running TensorRT.

---

## 4. Specification of the Target GPU Benchmark Protocol

To establish genuine GPU feasibility, the benchmark must be executed on actual GPU hardware following this auditable protocol:

### 4.1 Strict Timing Boundaries ($T_{\text{age}}$ Definition)
The benchmark must instrument discrete wall-clock timestamps ($T$ in milliseconds, microsecond precision via `std::chrono::high_resolution_clock` or `time.perf_counter` synchronized with CUDA events):

$$\begin{aligned}
T_{\text{age}} &= T_{\text{publish}} - T_{\text{capture}} \\
&= (T_{\text{enqueue}} - T_{\text{capture}}) + (T_{\text{dequeue}} - T_{\text{enqueue}}) + T_{\text{extract}} + T_{\text{dma}} + T_{\text{infer}} + T_{\text{post}} + (T_{\text{publish}} - T_{\text{post\_done}})
\end{aligned}$$

Where:
1. $T_{\text{capture}}$: Timestamp when camera frame DMA is completed.
2. $T_{\text{enqueue}}$: Timestamp when full-frame detection identifies an uncertainty trigger and enqueues crop request.
3. $T_{\text{dequeue}}$: Timestamp when background GPU crop worker picks request from queue.
4. $T_{\text{extract}}$: Sub-image extraction and letterbox padding.
5. $T_{\text{dma}}$: Memory transfer to GPU engine input tensor buffer (or zero-copy pointer mapping on Jetson).
6. $T_{\text{infer}}$: TensorRT engine execution (`cudaEventRecord(start)` to `cudaEventRecord(stop)`).
7. $T_{\text{post}}$: Intra-crop pedestrian NMS ($\text{IoU} = 0.40$) and conservative admission filter ($\text{conf} \ge 0.35$).
8. $T_{\text{publish}}$: Timestamp when verified crop result is placed in the completed results registry.

### 4.2 Workload & Streaming Conditions
- **Input Stream:** Continuous sequence of 1,000 frames from permitted BDD100K validation videos (`bdd100k_dataset/val/images`), streamed at a fixed interval of **$\Delta t = 33.33\,\text{ms}$ (30.0 Hz)**.
- **Worker Configuration:**
  - Dedicated background worker thread bound to an independent CUDA stream.
  - Queue capacity: **Strictly bounded to $N = 2$**.
  - Overload policy: **`DROP_OLDEST`** (drops pending item in queue when new item arrives to preserve temporal freshness).
  - Deadline gating: **$T_{\text{age}} \le 100.0\,\text{ms}$**. Any result finishing after 100.0 ms must be stamped `EXPIRED`.
- **Model Checkpoint:** Frozen Phase 6 YOLO11m (`best.pt`, SHA-256 `42d4b385...`). **NO YOLO11s.**
- **Precision Modes to Evaluate:**
  1. PyTorch CUDA FP16 (Baseline).
  2. TensorRT FP16 optimized engine (Target deployment).
  3. TensorRT INT8 calibrated engine (Low-power edge deployment).

### 4.3 Mandatory Reporting Schema
The benchmark must collect and publish:
- **Result Age Distribution:** Mean, p50, p90, p95, p99, and maximum $T_{\text{age}}$ (ms).
- **Queue Delay Distribution:** Mean, p50, p95 queue wait time (ms).
- **Deadline Satisfaction Rate:** Percentage of submitted requests completing with $T_{\text{age}} \le 100.0\,\text{ms}$. (Acceptance threshold: $\ge 98.0\%$).
- **Overload Drop Rate:** Percentage of requests discarded by `DROP_OLDEST`.
- **Resource Profiling:** Average and peak GPU compute utilization (%), VRAM allocation (MB), and GPU temperature / thermal throttling flags.

---

## 5. Audit of Correctness Requirements

### 5.1 Deadline Expiration & Isolation
- **Verification in Code:** [`src/foveation/async_worker.py`](file:///c:/Users/hiten/Downloads/cv/src/foveation/async_worker.py) lines 234–240:
  ```python
  if total_age <= req.deadline_ms:
      status = "MET_DEADLINE"
  else:
      status = "EXPIRED"
  ```
- **Auditor Finding:** When a crop result is marked `EXPIRED`, it is isolated in the telemetry record. Because temporal fusion into subsequent frames is currently disabled, expired results cannot contaminate immediate full-frame perception. Unit test `test_deadline_expiration_logic` verifies this behavior.

### 5.2 Monotonic Frame Identification & Provenance Tracking
- Every request encapsulates `frame_id`, `capture_timestamp_ms`, integer crop box, proposal metadata, and source detections.
- Completed results are indexed strictly by originating `frame_id`.
- Video evaluation over 20 consecutive frames verified **0 mismatched frame associations (100% integrity)**.

### 5.3 Spatial Misalignment & Camera Ego-Motion Under Delay
When a high-resolution crop is triggered on frame $t$ and completes at frame $t+k$ ($\Delta t \approx 30\text{--}90\,\text{ms}$):
1. **Vehicle Ego-Motion:** At an urban speed of $50\,\text{km/h}$ ($13.9\,\text{m/s}$), the vehicle translates forward by **$0.42\text{--}1.25\,\text{meters}$** between capture and completion. Road pitch and roll further displace the camera optical center by $10\text{--}40\text{ pixels}$.
2. **Target VRU Motion:** A crossing pedestrian ($1.4\,\text{m/s}$) shifts by $4\text{--}13\text{ cm}$ in physical space, moving several pixels across the sensor array.
3. **The Stale-Box Problem:** If crop bounding boxes from frame $t$ are projected directly into frame $t+k$ without compensation, the IoU overlap with the actual object drops below $0.50$, creating **false double detections** and tracking jitter.

**Mandatory Motion-Compensation Evaluation Protocol (Before Any Future Fusion):**
- Compute ego-motion displacement using camera IMU / vehicle CAN odometry or Lucas-Kanade sparse optical flow / planar homography $H_{t \to t+k}$.
- Transform crop box coordinates: $B_{t+k} = H_{t \to t+k}(B_t)$.
- Validate alignment accuracy: Centroid error must be $\le 10\text{ pixels}$ and IoU with ground-truth at frame $t+k$ must remain $\ge 0.70$.
- **Gating Rule:** **Delayed-result fusion must remain strictly DISABLED** until motion compensation is empirically validated.

---

## 6. Formal Feasibility Audit Decision

```
+======================================================================================================+
|                                  PHASE 9.1 AUDIT VERDICT & GATE DECISION                             |
+======================================================================================================+
|  AUDIT DECISION:  HOLD PENDING MISSING HARDWARE & EMPIRICAL EVIDENCE                                 |
|                                                                                                      |
|  RATIONALE:                                                                                          |
|  1. The 11.09 ms inference baseline was measured on YOLO11s in Phase 3. The frozen production         |
|     checkpoint is YOLO11m (over 3.16x compute footprint). No measurement of YOLO11m on T4 exists.    |
|  2. The 14.10 ms result age was a synthetic composite that combined an obsolete forward pass with    |
|     three unmeasured constants (0.5 ms crop, 1.0 ms PCIe, 1.5 ms postprocessing).                     |
|  3. No artifact in the repository contains a continuous 30 Hz streaming benchmark on GPU.            |
|  4. Target deployment hardware (Jetson Orin) has never been tested in this repository.               |
|  5. Local host environment is CPU-only (CUDA available: False). Feasibility remains unverified.      |
|                                                                                                      |
|  PROHIBITION: Projecting a successful GPU result without empirical measurement is strictly rejected.  |
+======================================================================================================+
```

---

## 7. Unresolved Blockers & Explicit Acceptance Criteria for Phase 9.2

### 7.1 Unresolved Blockers
1. **Model Disconnect Blocker:** Absence of isolated batch-1 640px latency measurements for the frozen **YOLO11m** checkpoint on Tesla T4 or Jetson Orin.
2. **Hardware Environment Blocker:** Local development machine lacks an active CUDA device (CPU execution only); all edge GPU benchmarking requires access to a target hardware testbench or cloud GPU runner.
3. **End-to-End Pipeline Blocker:** Absence of an integrated streaming script that synchronizes camera frame capture, background worker threading, TensorRT execution, and telemetry collection on GPU.
4. **Motion Compensation Blocker:** Absence of a validated homography/odometry motion-compensation algorithm to prevent spatial misattribution.

### 7.2 Explicit Acceptance Criteria for Advancing to Phase 9.2 (GPU Execution)
To lift the **HOLD** and advance to implementation, the next phase must satisfy all five criteria:

1. **Criterion 1 (Frozen Model Fidelity):** The benchmark must execute exclusively using `scratch/kaggle_upload_weights/best.pt` (SHA-256: `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e`).
2. **Criterion 2 (Empirical 30 Hz Streaming Test):** Execute a continuous 1,000-frame video stream at 30.0 Hz ($\Delta t = 33.33\,\text{ms}$) on target GPU hardware with queue capacity 2 and `DROP_OLDEST` overload policy.
3. **Criterion 3 (Measured 100 ms Deadline Compliance):** Achieve $\ge 95.0\%$ deadline compliance ($T_{\text{age}} \le 100.0\,\text{ms}$) across all executed crops under sustained 30 Hz streaming.
4. **Criterion 4 (Zero Misattribution Guarantee):** Maintain 100% frame association integrity with 0 mismatched frame IDs.
5. **Criterion 5 (Fusion Freeze Preserved):** Temporal box fusion into newer frames must remain disabled until an independent motion-compensation audit is conducted.
