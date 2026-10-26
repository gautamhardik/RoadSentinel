# RoadSentinel Phase 9.2 — Frozen YOLO11m GPU Benchmark Report

**Date:** 2026-10-13  
**Status:** **HOLD — TARGET HARDWARE UNAVAILABLE IN CURRENT ENVIRONMENT**  
**Checkpoint Path:** `scratch/kaggle_upload_weights/best.pt`  
**Checkpoint SHA-256:** `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` (Verified Pre- and Post-Audit)  
**Test Split Quarantine:** `bdd100k_dataset/test` strictly isolated (0 accesses).  
**Machine-Readable Diagnosis:** [`experiments/phase9_2_gpu_benchmark/benchmark_results.json`](file:///c:/Users/hiten/Downloads/cv/experiments/phase9_2_gpu_benchmark/benchmark_results.json)  
**Telemetry File Schema:** [`experiments/phase9_2_gpu_benchmark/per_request_telemetry.jsonl`](file:///c:/Users/hiten/Downloads/cv/experiments/phase9_2_gpu_benchmark/per_request_telemetry.jsonl)  
**Reproducible Benchmark Harness:** [`experiments/phase9_2_gpu_benchmark/run_gpu_benchmark.py`](file:///c:/Users/hiten/Downloads/cv/experiments/phase9_2_gpu_benchmark/run_gpu_benchmark.py)

---

## 1. Executive Summary & Objective

The objective of Phase 9.2 is to replace unsupported GPU feasibility projections with empirical, reproducible measurements of the frozen Phase 6 YOLO11m checkpoint under sustained 30 Hz streaming load with a bounded worker queue ($N = 2$) and `DROP_OLDEST` overload policy.

### Core Non-Negotiable Constraint
Per Phase 9.2 instructions:
> *"If the current environment has no CUDA device, stop before claiming a GPU benchmark was completed. Produce only the environment diagnosis and a reproducible run procedure. Do not label estimated latency as measured latency."*

Accordingly, this report delivers:
1. An empirical hardware and runtime environment diagnosis confirming the absence of an active CUDA GPU in the local runtime.
2. Bit-exact verification of the frozen checkpoint SHA-256 and preservation of the test-split quarantine.
3. A complete, self-contained, reproducible GPU streaming benchmark harness ([`run_gpu_benchmark.py`](file:///c:/Users/hiten/Downloads/cv/experiments/phase9_2_gpu_benchmark/run_gpu_benchmark.py)) configured to run on accelerated targets (e.g., NVIDIA Tesla T4 or NVIDIA Jetson Orin).
4. The formal decision: **HOLD**, pending execution on an active GPU testbed.

---

## 2. Environment Diagnosis & Hardware Probe

The local execution environment was probed programmatically:

```python
import torch
torch.__version__            # '2.12.0+cpu'
torch.cuda.is_available()     # False
torch.cuda.device_count()    # 0
```

### Detailed Environment Breakdown

| Parameter | Current Host Environment | Required GPU Benchmark Environment | Status |
| :--- | :--- | :--- | :---: |
| **Operating System** | Windows 11 (Build 26100) | Linux Ubuntu 22.04 LTS or Jetpack 6.x | Non-matching |
| **CPU Architecture** | Intel64 Family 6 Model 170 (16 physical / 22 logical cores) | Host CPU or Jetson ARM Cortex-A78AE | Compatible |
| **System RAM** | 31.43 GB DDR5 | $\ge 16$ GB RAM (or $\ge 16$ GB Unified on Jetson) | Compatible |
| **PyTorch Runtime** | **PyTorch 2.12.0+cpu** | **PyTorch 2.12.0+cu121 or TensorRT 10.x** | **MISSING CUDA** |
| **GPU Accelerator** | **None detected (`device_count = 0`)** | **NVIDIA Tesla T4 (16 GB) or Jetson Orin AGX** | **UNAVAILABLE** |
| **CUDA Driver Version** | Not installed / Not bound | CUDA 12.1+ / NVIDIA Driver 535+ | **UNAVAILABLE** |

**Forensic Finding:** The local development workstation does not possess an active NVIDIA GPU accelerator accessible to PyTorch. Under the Phase 9.2 instructions, **it is prohibited to fabricate, extrapolate, or report simulated GPU timings as measured latency.**

---

## 3. Checkpoint Integrity & Quarantine Audit

1. **Pre-Audit Checkpoint SHA-256:**
   `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e`
2. **Post-Audit Checkpoint SHA-256:**
   `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e`
3. **Weight Integrity:** 100% bit-exact match. Zero weights modified, zero retraining, zero fine-tuning.
4. **BDD100K Test Split Quarantine:**
   `bdd100k_dataset/test` remained completely untouched with 0 accesses.
5. **Temporal Fusion Status:**
   Temporal box fusion into newer frames remains **strictly disabled**. No stale bounding boxes are merged into current-frame perception.

---

## 4. Reproducible GPU Benchmark Procedure

To enable immediate, reproducible execution on target GPU hardware, a dedicated benchmark script was authored: [`experiments/phase9_2_gpu_benchmark/run_gpu_benchmark.py`](file:///c:/Users/hiten/Downloads/cv/experiments/phase9_2_gpu_benchmark/run_gpu_benchmark.py).

### 4.1 Benchmark Harness Architecture

```
Camera Stream (1,000 frames @ 30.0 Hz, Delta_t = 33.33 ms)
       |
       v
[ Frame Ingestion & Capture Timestamp T_capture ]
       |
       +---> [ 960px Full-Frame Primary Path ] ---> Immediate Output (Frame t)
       |
       v  (If uncertainty trigger fires)
[ Enqueue Crop Request (Frame ID, T_capture, ROI) ]
       |
       v
[ Bounded Worker Queue: maxsize = 2 ]
       |
       +---> [ Overload Policy: DROP_OLDEST ] ---> Log DROPPED_OVERLOAD
       |
       v
[ Background GPU Stream: torch.cuda.Stream() ]
       |
       +---> Sub-image extraction & letterbox (T_extract)
       +---> Host-to-device memory transfer (T_dma)
       +---> YOLO11m 640x640 batch-1 inference: cudaEventRecord() (T_infer)
       +---> Intra-crop pedestrian NMS (IoU 0.40) + admission filter (conf >= 0.35) (T_post)
       |
       v
[ Result Publication (T_publish) ]
       |
       v
Compute T_age = T_publish - T_capture
       |
       +---> If T_age <= 100.0 ms:  status = MET_DEADLINE
       +---> If T_age >  100.0 ms:  status = EXPIRED (isolated from immediate perception)
       |
       v
Append to per_request_telemetry.jsonl
```

### 4.2 Step-by-Step Procedure to Run on GPU Runner (Tesla T4 or Jetson Orin)

1. **Environment Setup on Target Machine:**
   ```bash
   # Ensure CUDA drivers and PyTorch with CUDA are active
   python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
   ```

2. **Verify Checkpoint:**
   ```bash
   sha256sum scratch/kaggle_upload_weights/best.pt
   # Must output: 42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e
   ```

3. **Launch the Benchmark:**
   ```bash
   python experiments/phase9_2_gpu_benchmark/run_gpu_benchmark.py \
       --checkpoint scratch/kaggle_upload_weights/best.pt \
       --output-dir experiments/phase9_2_gpu_benchmark \
       --stream-fps 30.0 \
       --num-frames 1000
   ```

4. **Telemetry Outputs Generated:**
   - Summary JSON: `experiments/phase9_2_gpu_benchmark/benchmark_results.json`
   - Raw JSONL: `experiments/phase9_2_gpu_benchmark/per_request_telemetry.jsonl`

---

## 5. Formal Feasibility Decision

```
+======================================================================================================+
|                                  PHASE 9.2 AUDIT VERDICT & GATE DECISION                             |
+======================================================================================================+
|  DECISION:  HOLD                                                                                     |
|                                                                                                      |
|  RATIONALE:                                                                                          |
|  1. Target GPU hardware is unavailable in the local environment (PyTorch 2.12.0+cpu, CUDA: False).   |
|  2. In strict adherence to non-negotiable constraints, execution was stopped before claiming         |
|     a GPU benchmark was completed.                                                                   |
|  3. Estimated or projected latencies were NOT labeled as measured latency.                          |
|  4. A reproducible benchmark harness (run_gpu_benchmark.py) was delivered to execute on GPU.         |
|                                                                                                      |
|  NEXT STEP: Deploy run_gpu_benchmark.py to an active CUDA testbed (Tesla T4 / Jetson Orin) to       |
|             collect physical measurements and evaluate the 95% deadline compliance threshold.         |
+======================================================================================================+
```

---

## 6. Deliverables Index

All required Phase 9.2 deliverables have been created and verified:

1. **Benchmark Report:** This document and [`experiments/phase9_2_gpu_benchmark/benchmark_report.md`](file:///c:/Users/hiten/Downloads/cv/experiments/phase9_2_gpu_benchmark/benchmark_report.md).
2. **Machine-Readable Environment Diagnosis & Results:** [`experiments/phase9_2_gpu_benchmark/benchmark_results.json`](file:///c:/Users/hiten/Downloads/cv/experiments/phase9_2_gpu_benchmark/benchmark_results.json).
3. **Telemetry Log Schema & Status Header:** [`experiments/phase9_2_gpu_benchmark/per_request_telemetry.jsonl`](file:///c:/Users/hiten/Downloads/cv/experiments/phase9_2_gpu_benchmark/per_request_telemetry.jsonl).
4. **Reproducible GPU Execution Script:** [`experiments/phase9_2_gpu_benchmark/run_gpu_benchmark.py`](file:///c:/Users/hiten/Downloads/cv/experiments/phase9_2_gpu_benchmark/run_gpu_benchmark.py).
5. **Frozen Checkpoint Integrity:** Bit-exact hash `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` preserved.
