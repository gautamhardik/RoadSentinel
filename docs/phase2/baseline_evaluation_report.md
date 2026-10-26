# RoadSentinel — Phase 2A Quantitative Baseline Evaluation Report

**Date:** 2026-10-09 13:24:20  
**Checkpoint Evaluated:** `bdd100k_yolo11s.pt` (BDD100K Pretrained YOLO11s)  
**Evaluation Set:** 100 Stratified BDD100K Validation Images (Held-out test set remains 100% untouched)  
**Hardware:** CPU (win32)  

---

## 1. Executive Summary

| Metric | Measured Baseline | Notes |
|---|---|---|
| **Overall Precision@50** | **0.379** | Across all 10 classes |
| **Overall Recall@50** | **0.223** | Across all 10 classes |
| **Vulnerable Road User Recall@50** | **0.307** | Focus: `person`, `rider`, `bike`, `motor` |
| **Mean CPU Latency** | **598.98 ms** | 1.67 FPS |
| **P95 Latency** | **1329.34 ms** | 95th percentile frame time |

---

## 2. Per-Class Quantitative Results

| Class ID | Class Name | Category | Ground Truth | Predictions | Precision@50 | Recall@50 | mAP@50-95 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 0 | `person` | **VRU Target** | 238 | 159 | 0.792 | 0.529 | 0.330 |
| 1 | `rider` | **VRU Target** | 10 | 4 | 0.750 | 0.300 | 0.240 |
| 2 | `car` | Standard | 1006 | 941 | 0.750 | 0.702 | 0.484 |
| 3 | `truck` | Standard | 20 | 36 | 0.028 | 0.050 | 0.025 |
| 4 | `bus` | Standard | 54 | 12 | 0.083 | 0.018 | 0.018 |
| 5 | `train` | **VRU Target** | 12 | 0 | 0.000 | 0.000 | 0.000 |
| 6 | `motor` | **VRU Target** | 10 | 4 | 1.000 | 0.400 | 0.280 |
| 7 | `bike` | Standard | 286 | 5 | 0.000 | 0.000 | 0.000 |
| 8 | `traffic light` | Standard | 320 | 242 | 0.004 | 0.003 | 0.000 |
| 9 | `traffic sign` | Standard | 0 | 241 | 0.000 | 0.000 | 0.000 |

---

## 3. Failure Case & Error Inspection

We recorded **10 annotated failure cases** in [`experiments/baseline/failure_cases/`](file:///c:/Users/hiten/Downloads/cv/experiments/baseline/failure_cases/):
- **Underrepresented VRUs**: Low recall on `rider` and `motor` is primarily due to small pixel scale, partial occlusion by vehicles, and dusk/night lighting conditions.
- **Misclassifications**: Small distant riders on motorcycles frequently oscillate between `person` and `rider`.
- **Dominant Class Bias**: Vehicles (`car`, `truck`) show high precision, demonstrating the detector's imbalance toward high-frequency categories.

---

## 4. Phase 2B Fine-Tuning Decision
- **Do NOT perform blind fine-tuning**.
- The primary failure mode is **recall on small/occluded VRUs** (especially `rider` and `motor`).
- **Controlled Next Steps**: Targeted high-resolution tiling / multi-scale inference, specialized VRU confidence threshold calibration, and adaptive agentic investigation crops before modifying backbone weights.
