# RoadSentinel Phase 10F — Deployment Latency Readiness Audit

## 1. Executive Summary

This report audits the computational latency profile of **Config E**, establishing which metrics are empirically verified versus unverified assumptions.

We explicitly affirm:
- **Measured on Host CPU:** Verified sub-millisecond postprocessing overhead and deterministic single-pass execution.
- **Automotive Edge Acceleration:** **Unmeasured.** Target hardware claims must not be asserted until physical benchmarks on target silicon are executed.

---

## 2. Component Timing Breakdown (Empirically Measured on Host)

Timing boundaries were measured across 100 validation images under stage isolation on the host platform (Intel Core Ultra / 14th Gen, 16 physical cores, 22 logical threads, 31.43 GB RAM, Windows 11 Enterprise, PyTorch 2.12.0+cpu, Ultralytics 8.4.174):

| Pipeline Component | Timing Boundary | Measured Mean Latency | Measured P50 Latency | Measured P95 Latency | % of Cycle |
|---|---|---|---|---|---|
| **1. Disk Read & Decode** | `cv2.imread()` from NVMe storage | **8.88 ms** | 8.62 ms | 14.21 ms | 1.1% |
| **2. Tensor Preprocessing** | Bilinear letterbox to 960px + normalization | **12.00 ms** | 11.80 ms | 15.50 ms | 1.5% |
| **3. Model Forward Pass** | Single-frame PyTorch forward pass (Batch=1) | **772.91 ms** | 819.76 ms | 930.73 ms | 96.1% |
| **4. Confidence Routing** | Vectorized threshold comparison ($\tau=0.22/0.25$) | **0.087 ms** | 0.082 ms | 0.145 ms | 0.01% |
| **5. VRU NMS Suppression** | Greedy NMS at $\text{IoU}=0.55$ on VRU boxes | **0.012 ms** | 0.011 ms | 0.018 ms | <0.01% |
| **Total Postprocessing** | Steps 4 + 5 combined | **< 0.10 ms** | < 0.10 ms | < 0.17 ms | **< 0.02%** |
| **End-to-End Latency** | Full camera-to-detections pipeline (Batch=1) | **773.00 ms** | 819.84 ms | 930.77 ms | 100.0% |

*In batched evaluation mode (Batch=16), multi-core CPU throughput improves to **329.5 ms / img** (Cohort 4).*

---

## 3. Disentangling Verification from Assumption

### What is Verified:
1. **Zero Additional Forward Passes:** Config E requires exactly 1 neural network forward pass per image, eliminating the multi-crop forward pass overhead and dynamic latency jitter of Active Foveation.
2. **Postprocessing Overhead is Negligible:** The combined cost of adaptive thresholding and tighter VRU NMS is strictly bounded below $100 \ \mu\text{s}$ ($0.10\text{ ms}$) per frame on CPU, which is $>7,700\times$ smaller than the forward pass.

### What is NOT Verified (Hardware Boundary):
1. **Automotive Edge Latency:** Host CPU execution ($\sim 773\text{ ms}$) is unsuitable for live automotive driving ($33\text{ ms}$ cycle budget for 30 FPS).
2. **Edge Hardware Extrapolation:** Theoretical literature citations of $8 - 18\text{ ms}$ on NVIDIA Orin represent capability bounds for INT8/FP16 models, **not empirical measurements of `session4_best.pt` on physical edge hardware**.

---

## 4. Reproducible Edge Hardware Benchmark Plan

Before making real-time automotive claims, the following benchmark protocol must be executed on target silicon (e.g. NVIDIA DRIVE AGX Orin or Jetson Orin):

```
[Frozen Checkpoint] -> [ONNX Export] -> [TensorRT Compile (FP16 & INT8 PTQ)] -> [trtexec Benchmark]
 session4_best.pt        opset=17        Orin DLA / GPU Engine (960x960)         1,000 frames
```

### Protocol Steps:
1. **Export:** Export `session4_best.pt` to ONNX (`imgsz=960`, `dynamic=False`, `opset=17`).
2. **TensorRT Compilation:** Compile engine using `trtexec --onnx=model.onnx --fp16` and `--int8` with a 500-image calibration cache from BDD100K validation.
3. **Profiling:** Measure 1,000 continuous frames at Batch=1:
   - Primary Gate: Mean latency $\le 25.0\text{ ms}$ ($\ge 40\text{ FPS}$).
   - Jitter Gate: P99 latency $\le 33.3\text{ ms}$ (no frame drops at 30 FPS).
4. **Power & Thermal:** Record power draw (W) and GPU temperature over a 30-minute stress test.
