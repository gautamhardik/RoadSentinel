# RoadSentinel Phase 4A — Ranked Improvement Opportunity Matrix

**Date:** October 10, 2026  
**Auditor:** Principal Computer Vision Researcher & ML Experimentalist  
**Status:** COMPLETE — Authoritative Research & Experiment Ranking  

---

## Priority Framework Definition

- **P0 — Integrity Blockers:** Fundamental defects in metrics, dataset splits, or class mappings that invalidate comparison. *(Status: Cleared in Phase 3.4 & Phase 4A audit).*
- **P1 — High-Value, Low-Cost Experiments:** Rapid, high-leverage optimizations utilizing the existing checkpoint, local validation data, and post-processing/inference pipeline adjustments without expensive multi-GPU retraining.
- **P2 — Substantial Experiments:** Controlled fine-tuning, training data augmentation, crop-policy redesign, or engine export requiring GPU training on the BDD100K 70k training split.
- **P3 — Advanced Research:** Architecture redesigns, multi-scale extra heads, temporal multi-frame models, or ensembles requiring significant compute and complexity.

---

## 1. Comprehensive Ranked Opportunity Matrix

| Rank | Priority | Area | Proposed Change | Demonstrated Bottleneck | Expected Mechanism | Primary Metric Impact | Secondary Impact | Compute Cost | Effort | Overfitting Risk | Evidence Strength |
|:---:|:---:|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|
| **1** | **P1** | **Post-Processing** | **Dual-Threshold Crop Score Calibration** | Active Foveation adds $+3,022$ FPs ($5,901$ vs $2,879$) due to uncalibrated crop hallucinations. | Require higher confidence ($\tau_{\text{crop, isolated}} \ge 0.40$) for unconfirmed crop detections, while allowing lower threshold ($\tau \ge 0.20$) for crops confirming full-frame candidates. | **VRU Precision ($\uparrow +8 - 12\%$)** | **VRU F1 ($\uparrow +0.03 - 0.05$), mAP50 ($\uparrow$)** | Minimal ($< 1$ GPU-hr) | Low | Low | **Strong (Demonstrated in audit)** |
| **2** | **P1** | **Post-Processing** | **Class-Specific Threshold Optimization ($\tau_c$)** | FullFrame and Foveation use uniform $\tau=0.25$. Rare classes (`rider`: $44.2\%$ rec, `bike`: $48.4\%$ rec) suffer score compression. | Decouple thresholds: raise $\tau_{\text{person}} \approx 0.30$ to prune person FPs, lower $\tau_{\text{rider}}, \tau_{\text{bike}} \approx 0.18-0.22$ to lift rare recalls. | **VRU Macro mAP50 ($\uparrow +0.015 - 0.025$)** | **VRU Macro Recall ($\uparrow$), F1 ($\uparrow$)** | Minimal ($< 30$ mins) | Low | Low (Cross-val split) | **Strong (Established CV practice)** |
| **3** | **P1** | **Detection Fusion** | **Weighted Box Fusion (WBF) with Scale Consensus** | Naive greedy deduplication randomly keeps higher score or discards overlapping boxes; misses mutual confirmation. | Blend overlapping boxes from full-frame and crop passes; boost confidence when both scales agree, penalize isolated boxes. | **Overall mAP50 ($\uparrow +0.010 - 0.020$)** | **Localization IoU ($\uparrow$), VRU Precision ($\uparrow$)** | Minimal ($< 1$ GPU-hr) | Low-Med | Low | **Strong (Solovyev et al., 2021)** |
| **4** | **P1** | **Active Perception** | **Horizon & Road-Corridor Spatial Prior** | Uncertainty proposals currently fire in sky, tree foliage, and vehicle hood, generating false alarms. | Restrict crop candidate generation to optical horizon corridor ($y \in [0.35H, 0.70H]$) where vanishing-point VRUs exist. | **Throughput ($\uparrow +15\%$), FP Reduction ($\downarrow 20\%$)** | **Small VRU Precision ($\uparrow$)** | Minimal ($< 1$ GPU-hr) | Low | Very Low | **Strong (Perspective geometry)** |
| **5** | **P1** | **Runtime Efficiency** | **Batched Crop Forward Pass** | Currently executes $K=2$ crops sequentially via two independent PyTorch calls. | Stack crop patches into single tensor `(K, 3, 640, 640)` for a single batched inference call. | **Batch-1 Pipeline Latency ($\downarrow 30 - 35\%$)** | **Throughput ($\uparrow 32 - 35\text{ FPS}$)** | None | Low | Zero | **Direct PyTorch / GPU optimization** |
| **6** | **P2** | **Model Training** | **Sliced Fine-Tuning (SFT) on BDD100K Train** | Existing checkpoint `bdd100k_yolo11s.pt` was trained purely on full-frame images; suffers domain shift on crops. | Fine-tune YOLO11s on mixture of $80\%$ full-frame images + $20\%$ cropped patches with ground-truth VRUs. | **Crop False Positives ($\downarrow 40\%$)** | **Small VRU Recall ($\uparrow +3 - 5\%$)** | Medium ($10 - 15$ GPU-hrs) | Med | Low (70k images) | **Strong (Akyon et al., ICIP 2022)** |
| **7** | **P2** | **Model Training** | **Class-Balanced VRU Augmentation (Copy-Paste)** | Extreme class imbalance: $13,265$ `person` vs $452$ `motor` and $649$ `rider` ($30:1$ ratio). | Apply targeted Copy-Paste and Mosaic augmentation on rare VRU classes during fine-tuning. | **Rare VRU Recall (`rider`, `motor` $\uparrow +8 - 12\%$)** | **VRU Macro mAP50 ($\uparrow +0.030$)** | Medium ($10 - 15$ GPU-hrs) | Med | Low | **Strong (Ghiasi et al., CVPR 2021)** |
| **8** | **P2** | **Runtime Engine** | **TensorRT FP16 Engine Compilation** | Local PyTorch inference runs unoptimized kernels, limiting throughput to $27.98\text{ FPS}$. | Export YOLO11s to TensorRT FP16 engine with dynamic shapes for full-frame and crop passes. | **In-Memory Latency ($\downarrow 50 - 60\%$, to $\approx 6\text{ ms}$)** | **Pipeline Throughput ($\uparrow > 50\text{ FPS}$)** | Low ($1$ GPU-hr) | Med | Zero | **Strong (NVIDIA TensorRT standards)** |
| **9** | **P2** | **Model Selection** | **YOLO11m Backbone Scaling Diagnostic** | Standalone YOLO11s feature capacity may limit small-object feature richness. | Evaluate pretrained or fine-tuned YOLO11m ($20.1\text{M}$ params) at $imgsz=960$ to measure capacity ceiling. | **Overall mAP50 ($\uparrow +0.02 - 0.03$)** | **Latency ($\uparrow 2.5\times$, drops to $\approx 18\text{ FPS}$)** | High ($20 - 30$ GPU-hrs) | Med-High | Low | **Moderate (Capacity vs FLOPs trade-off)** |
| **10** | **P3** | **Architecture** | **Addition of P2 (Stride-4) High-Res Head** | YOLO11s smallest head is $P3$ (stride 8); a $16\text{ px}$ pedestrian is only $2\text{ px}$ in feature space. | Modify YOLO11 neck to fuse $P2$ feature map with an additional tiny-object prediction head. | **Small VRU Recall ($\uparrow +5 - 8\%$)** | **FLOPs ($\uparrow 40\%$), Latency ($\uparrow 35\%$)** | High ($30 - 40$ GPU-hrs) | High | Med | **Moderate (Requires custom head code)** |
| **11** | **P3** | **Temporal Tracking** | **Multi-Frame ByteTrack Association** | Single-frame detection misses occluded or blurred VRUs across time. | Associate low-confidence detections with multi-frame Kalman filter tracks across video sequences. | **Temporal Consistency, Miss Recovery** | **Requires video split (not in val keyframes)** | High | High | Med | **Strong (Video only, invalid on keyframes)** |

---

## 2. Granular Profiles of Top Opportunities

### Opportunity 1: Dual-Threshold Crop Score Calibration (Rank 1, Priority P1)
- **Bottleneck Addressed:** In Phase 3.4, Active Foveation recovered $+891$ true positives over FullFrame-960, but admitted $+3,022$ false positives, degrading precision from $75.24\%$ to $62.03\%$.
- **Hypothesis:** Detections originating from high-resolution crop sub-images have a higher rate of false alarms on background textures unless confirmed by a coarse full-frame detection. Applying a dual-threshold rule:
  $$\tau = \begin{cases} 0.20 & \text{if crop detection overlaps a full-frame candidate } (\text{IoU} \ge 0.30) \\ 0.40 & \text{if crop detection is completely isolated} \end{cases}$$
  will eliminate $>50\%$ of the crop false positives ($>1,500\text{ FPs}$) while retaining $>92\%$ of recovered true positives.
- **Experiment Design:** Run parameter sweep over $(\tau_{\text{isolated}} \in [0.30, 0.50], \tau_{\text{confirmed}} \in [0.15, 0.25])$ on validation split predictions.
- **Acceptance Criteria:** VRU Precision improves to $\ge 70.0\%$, VRU F1 improves to $\ge 0.6550$, Small VRU Recall remains $\ge 45.0\%$.
- **Recommendation:** **GO — Execute immediately in Stage 2.**

### Opportunity 2: Class-Specific Threshold Optimization (Rank 2, Priority P1)
- **Bottleneck Addressed:** Uniform $\tau=0.25$ forces common and rare classes into the same operating point. `rider` ($44.2\%$ rec), `bike` ($48.4\%$ rec), and `motor` ($41.1\%$ rec) are severely suppressed.
- **Hypothesis:** Optimizing individual thresholds $\vec{\tau} = (\tau_{\text{person}}, \tau_{\text{rider}}, \tau_{\text{bike}}, \tau_{\text{motor}})$ by maximizing class-wise F1 score on a 50/50 validation split will lift VRU Macro Recall without precision collapse.
- **Experiment Design:** Grid search $\tau_c \in [0.15, 0.40]$ with step $0.02$ independently per class. Evaluate cross-validation stability.
- **Acceptance Criteria:** VRU Macro mAP50 improves by $\ge +0.010$, VRU Macro F1 improves by $\ge +0.020$.
- **Recommendation:** **GO — Execute in Stage 2.**

### Opportunity 3: Weighted Box Fusion with Scale Consensus (Rank 3, Priority P1)
- **Bottleneck Addressed:** In the current fusion implementation, full-frame and crop boxes undergo greedy NMS deduplication. When two detections overlap, the lower score is dropped without contributing coordinate precision or confidence reinforcement.
- **Hypothesis:** Multi-scale agreement provides strong evidence of a true positive. Weighted Box Fusion (WBF) that boosts candidate confidence when both full-frame and crop detect the object, and downweights single-scale marginal candidates, will directly improve mAP50 and F1.
- **Experiment Design:** Integrate WBF with IoU match threshold $0.50$ and score combination weights $w_{\text{ff}} = 1.0, w_{\text{crop}} = 0.8$.
- **Acceptance Criteria:** Overall mAP50 improves over ActiveFoveation baseline ($0.5403 \rightarrow \ge 0.5500$).
- **Recommendation:** **GO — Execute in Stage 2.**

### Opportunity 4: Horizon-Prioritized Spatial Uncertainty (Rank 4, Priority P1)
- **Bottleneck Addressed:** $12\%$ of generated crop proposals in Phase 3.4 were placed in irrelevant image regions (overhead road signs, tree foliage, car hood).
- **Hypothesis:** Camera perspective geometry in autonomous driving confines distant vulnerable road users to a horizontal corridor centered on the vanishing point ($y \in [0.35H, 0.70H]$). Weighting uncertainty proposals with a vertical Gaussian prior centered at $0.50H$:
  $$U_{\text{spatial}}(x, y) = U_{\text{det}}(x, y) \cdot \exp\left(-\frac{(y - 0.50H)^2}{2(0.15H)^2}\right)$$
  will suppress false-trigger crops and concentrate zoom budget on genuine distant road users.
- **Acceptance Criteria:** Reduces crop trigger rate by $\ge 10\%$ while maintaining or improving Small VRU Recall.
- **Recommendation:** **GO — Execute in Stage 3.**

### Opportunity 5: Crop Forward-Pass Batching (Rank 5, Priority P1)
- **Bottleneck Addressed:** Active Foveation executes crop forward passes sequentially, causing latency to jump from $17.44\text{ ms}$ (uncropped) to $39.72\text{ ms}$ (P95 cropped).
- **Hypothesis:** Combining the $K \le 2$ crops into a single mini-batch tensor `(K, 3, 640, 640)` executes both crops concurrently on GPU Tensor Cores, cutting crop pass latency from $18\text{ ms}$ to $\approx 11\text{ ms}$.
- **Acceptance Criteria:** Batch-1 Pipeline P95 latency drops from $39.72\text{ ms}$ to $< 30.0\text{ ms}$; throughput increases from $27.98\text{ FPS}$ to $> 32.0\text{ FPS}$.
- **Recommendation:** **GO — Implement in Stage 3.**

---

## 3. High-Cost / High-Risk Items Deferred

1. **Retraining Large Models (YOLO11x / RT-DETR):** Deferred. Massive training cost ($>50$ GPU-hours) with low probability of meeting real-time 30 FPS edge latency budgets.
2. **Temporal Multi-Frame Tracking on Current Split:** Deferred. The BDD100K 10,000 validation split consists of isolated single keyframes; evaluating tracking on disconnected static frames is scientifically invalid.
3. **Complex Transformer Ensembles:** Deferred. Ensembles destroy the core research thesis of selective active perception by multiplying compute uniformly.
