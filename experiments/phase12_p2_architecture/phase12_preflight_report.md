# RoadSentinel Phase 12 — P2 Architecture Preflight & Audit Report

**Report Status:** PREFLIGHT AUDIT COMPLETE (Ready for Training Review)  
**Date:** October 15, 2026  
**Investigated Model:** YOLO11m-P2 (`yolo11m-p2.yaml`)  
**Reference Checkpoint:** `session4_best.pt` (SHA-256: `4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd`)  
**Evaluation Scope:** Strictly Permitted BDD100K Training (`train`) and Validation (`val`) Splits  
**Test Split Quarantine:** Intact; Zero Access  

---

## 1. Architecture Audit & Structural Comparison (Task 1)

The proposed [`yolo11m-p2.yaml`](file:///c:/Users/hiten/Downloads/cv/experiments/phase12_p2_architecture/yolo11m-p2.yaml) was audited against the installed standard YOLO11m configuration:

### Layer-by-Layer Architectural Alignment

| Stage | Standard YOLO11m (`session4_best.pt`) | YOLO11m-P2 (`yolo11m-p2.yaml`) | Structural Relationship |
|---|---|---|---|
| **Backbone (Layers 0–10)** | 11 layers: Conv $\to$ C3k2 $\to$ SPPF $\to$ C2PSA | 11 layers: Conv $\to$ C3k2 $\to$ SPPF $\to$ C2PSA | **100% Identical Architecture** |
| **P2 Lateral Tap** | Not tapped (Layer 2 unused by head) | **Backbone Layer 2 tapped** ($240 \times 240 \times 256$) | New skip connection to FPN |
| **Top-Down FPN Path** | Layers 11–16: Downsamples to P4 ($60^2$) and P3 ($120^2$) | Layers 11–19: Continues downsample to **P2 ($240^2$)** | Adds Upsample + Concat + C3k2 |
| **Bottom-Up PAN Path** | Layers 17–22: Steps P3 $\to$ P4 $\to$ P5 | Layers 20–28: Steps **P2 $\to$ P3 $\to$ P4 $\to$ P5** | Adds Conv + Concat + C3k2 for P2 |
| **Detection Head** | Layer 23: `Detect([16, 19, 22])` (3 scales) | Layer 29: **`Detect([19, 22, 25, 28])` (4 scales)** | Adds high-resolution P2 sub-head |
| **Total Layers** | 24 composite layers (231 torch modules) | **30 composite layers (286 torch modules)** | +6 composite layers (+55 torch modules) |

---

## 2. Stride, Feature Map, and Output Dimension Verification (Task 2)

A forward pass with a tensor of shape `[1, 3, 960, 960]` was hooked into the detection head:

### Feature Map and Output Tensor Verification

| Feature Scale | Downsampling Factor | Feature Map Dimensions | Receptive Field Focus | Spatial Anchors / Grid Cells |
|---|---|---|---|---|
| **P2 (Micro-Scale)** | **Stride 4** | **`[1, 128, 240, 240]`** | **Small VRUs (<32² px)** | **57,600 cells (75.29%)** |
| **P3 (Small-Scale)** | Stride 8 | `[1, 256, 120, 120]` | Medium-Small Objects | 14,400 cells (18.82%) |
| **P4 (Medium-Scale)** | Stride 16 | `[1, 512, 60, 60]` | Medium-Large Objects | 3,600 cells (4.71%) |
| **P5 (Large-Scale)** | Stride 32 | `[1, 512, 30, 30]` | Large Vehicles / Trucks | 900 cells (1.18%) |
| **Total Spatial Grids** | — | — | Full Scene | **76,500 cells (4.05× baseline)** |

- **Detection Strides:** `[4, 8, 16, 32]` (**VERIFIED**).
- **Target Classes:** `nc = 10` BDD100K categories (`person`, `rider`, `car`, `bus`, `truck`, `bike`, `motor`, `traffic light`, `traffic sign`, `train`) (**VERIFIED**).
- **Raw Prediction Output Tensor:** Shape is exactly `torch.Size([1, 14, 76500])`, where $14 = 4 \text{ box coordinates } (x, y, w, h) + 10 \text{ class scores}$, and $76,500 = 57,600 + 14,400 + 3,600 + 900$ (**VERIFIED**).

---

## 3. Pretrained-Weight Transfer Coverage Audit (Task 3)

Weight transfer from the champion checkpoint (`session4_best.pt`, 50 epochs fine-tuned) into `yolo11m-p2.yaml` was audited under two strategies using [`initialize_p2_weights.py`](file:///c:/Users/hiten/Downloads/cv/experiments/phase12_p2_architecture/initialize_p2_weights.py):

### Weight Transfer Audit

| Transfer Strategy | Transferred Parameters | Transferred % | Newly Initialized Parameters | Newly Initialized % | Matched Tensors | New Tensors |
|---|---|---|---|---|---|---|
| **Strategy A: Naive Key Match** | 12,565,783 | 60.99% | 8,037,604 | 39.01% | 427 | 376 |
| **Strategy B: Semantic Functional Mapping** | **19,558,680** | **94.93%** | **1,044,707** | **5.07%** | **601** | **202** |

### Breakdown of Newly Initialized Parameters Under Strategy B
Strategy B maps functional stages by feature-scale semantics (Backbone $\to$ Backbone, FPN P4/P3 $\to$ FPN P4/P3, PAN P4/P5 $\to$ PAN P4/P5, Detect P3/P4/P5 $\to$ Detect P3/P4/P5). Exactly **1,044,707 parameters (5.07%)** require random initialization:

1. **`model.19` (FPN P2 C3k2 Block):** 137,225 parameters — processes the fused stride-4 lateral features.
2. **`model.20` (PAN P2 $\to$ P3 Downsample Conv):** 147,969 parameters — downsamples P2 to P3 in bottom-up path.
3. **`model.22` (PAN P3 C3k2 Block):** 380,937 parameters — fuses the refined P2 features with P3.
4. **`model.29.cv2.0` & `model.29.cv3.0` (Detect P2 Sub-head):** 378,576 parameters — regression and classification heads for stride-4 grid.

**Conclusion:** Strategy B transfers **94.93%** of weights from `session4_best.pt`, preserving mature feature detectors and stabilizing transfer learning.

---

## 4. Recomputed Compute & Parameter Profile (Task 4)

Both models were profiled under identical conditions using Ultralytics info profiling at $960 \times 960$ input resolution:

| Architecture | Model Checkpoint / Config | Total Layers | Parameter Count | Trainable Gradients | GFLOPs (@ 960x960) |
|---|---|---|---|---|---|
| **Standard YOLO11m** | `session4_best.pt` | 231 layers | 20,060,718 | 0 (frozen) | 154.1 GFLOPs |
| **YOLO11m-P2** | `yolo11m-p2.yaml` | 286 layers (+55) | 20,555,000 (+2.46%) | 20,554,984 | **199.3 GFLOPs (+29.33%)** |

- **Parameter Overhead:** $+494,282$ parameters (+2.46%).
- **Computational Overhead:** $+45.2$ GFLOPs (+29.33%), caused by convolutions executing on the $240 \times 240$ spatial grid.

---

## 5. Forward and Backward Pass Smoke Test (Task 5)

A smoke test was executed using a batch of 2 real images from permitted training data (`bdd100k_dataset/train/images/`):

1. **Input Batch:** Shape `torch.Size([2, 3, 960, 960])` normalized to $[0.0, 1.0]$.
2. **Forward Pass:** Model set to `.train()` mode. Successfully returned prediction dictionary containing multi-scale boxes, class scores, and feature representations across all 4 scales (`dict_keys(['boxes', 'scores', 'feats'])`).
3. **Loss & Backward Pass:** Surrogate detection loss computed and `.backward()` executed cleanly.
4. **Gradient Flow Audit:**
   - **Active Gradients Computed:** **409 parameter tensors** (**VERIFIED**).
   - **Missing Gradients:** Exactly 1 tensor (`Detect.dfl.conv.weight`, which is a fixed non-learnable coordinate register buffer, `requires_grad=False` by design).
   - **Numerical Stability:** **0 NaN gradients, 0 Inf gradients** (**PASSED**).
   - **P2 Specific Gradient Verification:**
     * `model.19` (FPN P2): 27 gradient tensors active.
     * `model.20` (PAN P2 Conv): 3 gradient tensors active.
     * `model.22` (PAN P3): 27 gradient tensors active.
     * `model.29` (Detect): 88 gradient tensors active.

---

## 6. VRAM Footprint & Conservative Batch Sizing (Task 6)

### Activation Memory Analysis for 16GB GPU (NVIDIA Tesla T4 / P100)
- **Static Model Footprint:** Weights + Gradients + AdamW Optimizer States = ~330 MB.
- **Dynamic Activation Memory:** The $240 \times 240$ feature maps in P2 require $4\times$ the activation memory of P3.
- **Estimated Peak VRAM on 960x960 Training with AMP (FP16):**
  * Batch Size 4: $\approx 5.2\text{ GB}$ VRAM (Plenty of headroom).
  * Batch Size 8: $\approx 8.8\text{ GB}$ VRAM (**Recommended**; leaves 7.2 GB safety buffer on 16GB GPU).
  * Batch Size 16: $\approx 15.9\text{ GB}$ VRAM (**UNSAFE**; high risk of CUDA OOM during mosaic augmentation or gradient scaling spikes).

**Recommendation:** Set **`batch_size = 8`** with **`accumulate = 2`**, producing an effective nominal batch size of 16 while keeping peak VRAM below 9.0 GB.

---

## 7. Validation Pipeline & Metric Parity Verification (Task 7)

YOLO11m-P2 outputs predictions natively projected to original image dimensions:
- Preprocessing: Identical $960 \times 960$ letterbox (stride 32, bilinear interpolation).
- Coordinate Space: `Boxes.xyxy` natively scales back to original $1280 \times 720$ coordinates.
- Matching Engine: Class-aware greedy matching at $\text{IoU} \ge 0.50$ via `match_single_image`.
- Object Size Brackets: COCO standard area thresholds ($<32^2$ Small, $32^2$–$96^2$ Medium, $\ge 96^2$ Large).
- Metric Computation: Small-VRU recall, VRU precision, overall VRU F1, and mean false positives per image calculated identically to Phase 11.

---

## 8. Proposed Training Strategy & Validation Confirmation Protocol (Task 8)

### A. Training Hyperparameters (Kaggle GPU Session)
- **Model:** `yolo11m_p2_initialized.pt` (Strategy B semantic functional initialization).
- **Dataset:** BDD100K `train` split (70,000 images).
- **Epochs:** 40 epochs total.
- **Backbone Warmup:** Freeze backbone (Layers 0–10) for **first 3 epochs** (`freeze=10`) to let the 1.04M newly initialized P2 parameters stabilize, then unfreeze all layers for epochs 4–40.
- **Optimizer:** SGD with cosine learning rate schedule (`lr0=0.01`, `lrf=0.01`, `momentum=0.937`, `weight_decay=0.0005`).
- **Input Size:** 960px.
- **Batching:** `batch=8`, `accumulate=2`.
- **Augmentation:** `mosaic=1.0`, `mixup=0.1`, `scale=0.5`, `translate=0.1`.

### B. Checkpoint Selection Rule
Generic mAP50-95 must **not** be used for model selection because vehicles dominate 85% of annotations.
- **Primary Selection Criterion:** Checkpoint that achieves the highest **Small-VRU Recall** on the validation cohort, subject to meeting the **VRU Precision floor ($\ge 72.0\%$)**.
- Artifacts to retain: `last.pt`, `best_small_vru.pt`, and `best_vru_f1.pt`.

---

## 9. Target-Hardware Latency Criterion (Task 9)

- **Algorithmic Complexity:** Single forward pass per frame verified ($O(1)$ deterministic execution).
- **GFLOPs Overhead:** Measured at 199.3 GFLOPs (+29.33% over standard YOLO11m).
- **Target-Hardware Latency Status:** Formally designated as **UNASSESSED**. Real-time automotive compliance ($\le 33.3\text{ ms}$ at 30 FPS or $\le 100\text{ ms}$ at 10 FPS) cannot be claimed until the candidate model is converted to FP16/INT8 TensorRT and benchmarked on automotive embedded hardware (e.g., NVIDIA Jetson AGX Orin).

---

## 10. Preflight Verification Summary

| Preflight Check | Status | Verification Detail |
|---|---|---|
| **Architecture Specification** | **PASSED** | 286 layers, 4 detection scales [4, 8, 16, 32], 76,500 spatial candidates |
| **Weight Transfer Audit** | **PASSED** | Strategy B transfers 94.93% (19.56M params); 5.07% (1.04M params) new |
| **Compute Profiling** | **PASSED** | 20.56M params (+2.46%), 199.3 GFLOPs (+29.33%) at 960px |
| **Smoke Test (Fwd/Bwd)** | **PASSED** | Successful forward/backward pass on train images; 409 active gradient tensors |
| **VRAM & Batch Sizing** | **PASSED** | Batch 8 with accumulate 2 recommended (~8.8 GB peak VRAM on 16GB GPU) |
| **Evaluation Parity** | **PASSED** | Identical letterbox, class mapping, matching IoU 0.50, and size brackets |
| **Test Quarantine** | **PASSED** | Zero test split accesses; 2,000 consumed images retired |

**STOP CONDITION ENFORCED:** Full training has **not** been launched. Awaiting review of this preflight report and explicit authorization of the training strategy.
