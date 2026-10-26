# RoadSentinel — Corrected Baseline Evaluation Report (Phase 2A)

**Evaluation Version:** 2.0.0  
**Date:** October 9, 2026  
**Model Weights:** `weights/bdd100k_yolo11s.pt` (YOLO11s Pretrained on BDD100K)  
**Evaluation Protocol:** Class-Aligned, 101-Point COCO Interpolation  
**Dataset Split:** BDD100K Validation — 100-Image VRU-Enriched Sample (70 VRU-Rich + 30 Standard)  
**Hardware Environment:** Intel64 Family 6 Model 170 (CPU) | PyTorch 2.12.0+cpu | OpenCV 5.0.0  

---

## 1. Benchmark Specification & Disclaimer

> [!IMPORTANT]
> **Sample Characterization Disclaimer:**  
> This evaluation was performed on a **100-image VRU-enriched validation subset** (70 images with pedestrian/cyclist/rider annotations and 30 standard traffic images). It is designed specifically to audit evaluation integrity and benchmark baseline performance on Vulnerable Road Users (VRUs). **This is NOT a representative sample of the full 10,000-image BDD100K validation split.** Full validation results must be computed separately on GPU infrastructure.

### Operating Parameters
- **Inference Resolution:** $640 \times 640$ pixels
- **Operational Confidence Threshold ($\tau$):** $0.25$
- **Operational Box Matching IoU ($\theta$):** $0.50$
- **AP Evaluation Protocol:** Standard COCO 101-point interpolated PR curve evaluated across 10 IoU thresholds $\theta \in [0.50 : 0.05 : 0.95]$

---

## 2. Corrected Per-Class Performance Table

The table below reports the verified ground-truth counts, operational predictions, confusion matrix components ($TP, FP, FN$), operational Precision/Recall at IoU 0.50, and official COCO Average Precision ($AP@50$ and $AP@50\text{--}95$).

| Semantic Class | Category | GT Count | Pred Count ($\tau \ge 0.25$) | TP@50 | FP@50 | FN@50 | Precision@50 | Recall@50 | F1@50 | AP@50 | AP@50–95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **person** | **VRU** | 238 | 159 | 126 | 33 | 112 | 0.7925 | 0.5294 | 0.6348 | 0.6242 | 0.3330 |
| **rider** | **VRU** | 10 | 4 | 3 | 1 | 7 | 0.7500 | 0.3000 | 0.4286 | 0.3069 | 0.2432 |
| **car** | Standard | 1006 | 941 | 706 | 235 | 300 | 0.7503 | 0.7018 | 0.7252 | 0.7438 | 0.4781 |
| **bus** | Standard | 20 | 12 | 10 | 2 | 10 | 0.8333 | 0.5000 | 0.6250 | 0.5727 | 0.4796 |
| **truck** | Standard | 54 | 36 | 29 | 7 | 25 | 0.8056 | 0.5370 | 0.6444 | 0.6014 | 0.4560 |
| **bike** | **VRU** | 12 | 5 | 2 | 3 | 10 | 0.4000 | 0.1667 | 0.2353 | 0.1840 | 0.1008 |
| **motor** | **VRU** | 10 | 4 | 4 | 0 | 6 | 1.0000 | 0.4000 | 0.5714 | 0.4767 | 0.3236 |
| **traffic light** | Standard | 286 | 242 | 165 | 77 | 121 | 0.6818 | 0.5769 | 0.6250 | 0.5903 | 0.2221 |
| **traffic sign** | Standard | 320 | 241 | 171 | 70 | 149 | 0.7095 | 0.5344 | 0.6096 | 0.5973 | 0.3090 |
| **train** | Standard | 0 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | *N/A (0 GT)* | *N/A (0 GT)* |

---

## 3. Discrepancy Autopsy: Legacy vs Corrected Baseline

A direct comparison demonstrates that the previously reported failure of traffic signs, traffic lights, buses, and trucks was an artifact of integer ID misalignment, not model failure.

| Class | Legacy GT | Corrected GT | Legacy Pred | Corrected Pred | Legacy P@50 | Corrected P@50 | Legacy R@50 | Corrected R@50 | Legacy AP@50 | Corrected AP@50 | Status / Cause |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **person** | 238 | 238 | 159 | 159 | 0.793 | 0.793 | 0.529 | 0.529 | 0.420 | 0.624 | Coincidentally matched ID 0. Repaired COCO AP. |
| **rider** | 10 | 10 | 4 | 4 | 0.750 | 0.750 | 0.300 | 0.300 | 0.225 | 0.307 | Coincidentally matched ID 1. Repaired COCO AP. |
| **car** | 1006 | 1006 | 941 | 941 | 0.750 | 0.750 | 0.702 | 0.702 | 0.527 | 0.744 | Coincidentally matched ID 2. Repaired COCO AP. |
| **bus** | 54 | **20** | 12 | 12 | 0.083 | **0.833** | 0.019 | **0.500** | 0.002 | **0.573** | Was matched against GT trucks (ID 4). **REPAIRED**. |
| **truck** | 20 | **54** | 36 | 36 | 0.028 | **0.806** | 0.050 | **0.537** | 0.001 | **0.601** | Was matched against GT buses (ID 3). **REPAIRED**. |
| **bike** | 286 | **12** | 5 | 5 | 0.000 | **0.400** | 0.000 | **0.167** | 0.000 | **0.184** | Was matched against GT traffic lights (ID 7). **REPAIRED**. |
| **motor** | 10 | 10 | 4 | 4 | 1.000 | 1.000 | 0.400 | 0.400 | 0.400 | 0.477 | Coincidentally matched ID 6. Repaired COCO AP. |
| **traffic light**| 320 | **286** | 242 | 242 | 0.004 | **0.682** | 0.003 | **0.577** | 0.000 | **0.590** | Was matched against GT traffic signs (ID 8). **REPAIRED**. |
| **traffic sign** | 0 | **320** | 241 | 241 | 0.000 | **0.710** | 0.000 | **0.534** | 0.000 | **0.597** | Was matched against GT trains (ID 9). **REPAIRED**. |
| **train** | 12 | **0** | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **N/A** | Was reading GT bikes (ID 5). Now cleanly handled. |

---

## 4. Vulnerable Road User (VRU) Aggregate Performance

VRUs represent the safety-critical core of RoadSentinel. We report both micro-averaged (pooled instance-level) and macro-averaged (class-level) metrics for the 4 VRU categories: `person`, `rider`, `bike`, `motor`.

### 4.1 Micro-Averaged VRU Performance (Pooled Instances)
Micro-averaging pools all true positives, false positives, and false negatives across all VRU categories into a single aggregate metric:
- $\text{VRU } TP = 126 + 3 + 2 + 4 = 135$
- $\text{VRU } FP = 33 + 1 + 3 + 0 = 37$
- $\text{VRU } FN = 112 + 7 + 10 + 6 = 135$
- $\text{Total VRU Ground Truth} = 238 + 10 + 12 + 10 = 270$
- $\text{Total VRU Operational Predictions} = 159 + 4 + 5 + 4 = 172$

$$\text{VRU Micro Precision@50} = \frac{135}{172} = \mathbf{0.7849} \quad (78.49\%)$$
$$\text{VRU Micro Recall@50} = \frac{135}{270} = \mathbf{0.5000} \quad (50.00\%)$$
$$\text{VRU Micro F1@50} = \frac{2 \times 0.7849 \times 0.5000}{0.7849 + 0.5000} = \mathbf{0.6109} \quad (61.09\%)$$

### 4.2 Macro-Averaged VRU Performance (Unweighted Class Mean)
Macro-averaging calculates the arithmetic mean across the 4 VRU classes, giving equal weight to each user type regardless of instance frequency:
$$\text{VRU Macro mAP@50} = \frac{0.6242 + 0.3069 + 0.1840 + 0.4767}{4} = \mathbf{0.3980} \quad (39.80\%)$$
$$\text{VRU Macro mAP@50–95} = \frac{0.3330 + 0.2432 + 0.1008 + 0.3236}{4} = \mathbf{0.2501} \quad (25.01\%)$$
$$\text{VRU Macro Precision@50} = \frac{0.7925 + 0.7500 + 0.4000 + 1.0000}{4} = \mathbf{0.7356} \quad (73.56\%)$$
$$\text{VRU Macro Recall@50} = \frac{0.5294 + 0.3000 + 0.1667 + 0.4000}{4} = \mathbf{0.3490} \quad (34.90\%)$$

---

## 5. Overall Dataset Summary Metrics

Calculated over all 9 present classes (excluding `train` which had 0 ground truth instances in the evaluated subset):
- **Overall mAP@50:** $\mathbf{0.5219}$ ($52.19\%$) — *(vs $0.157$ reported erroneously in legacy)*
- **Overall mAP@50–95:** $\mathbf{0.3273}$ ($32.73\%$)
- **Overall Macro Precision@50:** $\mathbf{0.7470}$ ($74.70\%$)
- **Overall Macro Recall@50:** $\mathbf{0.4718}$ ($47.18\%$)

---

## 6. Inference Latency & System Benchmark

All latency measurements adhered to rigorous benchmarking standards:
1. Model weights loaded into memory prior to timing (load duration: $267.9\text{ ms}$).
2. 5 warm-up iterations executed to initialize PyTorch CPU kernels and thread pools.
3. Timed inference across 100 images recorded with `time.perf_counter()`.

```json
{
  "device": "cpu",
  "processor": "Intel64 Family 6 Model 170 Stepping 4, GenuineIntel",
  "input_resolution": "640x640",
  "num_timed_images": 100,
  "mean_latency_ms": 483.02,
  "median_latency_ms": 454.91,
  "std_latency_ms": 78.45,
  "p95_latency_ms": 642.10,
  "min_latency_ms": 328.40,
  "max_latency_ms": 712.30,
  "throughput_fps": 2.07
}
```

---

## 7. Vulnerable Road User Failure Mode Analysis

Inspection of the 15 visual failure cases saved in `experiments/baseline_v2/failure_cases/` highlights the primary challenges facing single-pass YOLO detection:

1. **Small-Scale Distant Pedestrians (Scale Ambiguity):**
   - In urban streetscapes (e.g., `failure_vru_01_b1c66a42`, `failure_vru_06_b1ceb32e`), pedestrians occupying fewer than $20 \times 40$ pixels are frequently missed ($FN$) by the standard $640 \times 640$ feature pyramid.
2. **Cyclist-Bicycle Occlusion:**
   - Cyclists riding in traffic are frequently classified as `person` while the `bike` bounding box is missed, or the combined rider/bike entity is partially merged into a single detection.
3. **Low-Illumination & Rainy Night Conditions:**
   - Pedestrians walking in unlit street corners or under windshield rain reflections exhibit low feature contrast, resulting in confidence scores falling below the operational threshold $\tau = 0.25$.

These failure modes establish the clear scientific motivation for RoadSentinel's upcoming **Agentic Vision Pipeline**: active ROI re-examination, uncertainty-aware zoom, and multi-scale context verification will directly target these small, occluded, and low-contrast VRU instances.
