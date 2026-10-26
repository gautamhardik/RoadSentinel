# RoadSentinel Phase 10F — Implementation and Reproducibility Audit

## 1. Executive Summary

This audit independently inspects the software pipeline, tensor coordinate mappings, class dictionaries, NMS routines, and ground-truth evaluation engines to verify that **Config E** can be deployed and evaluated with zero implementation ambiguity.

---

## 2. Model Checkpoint & Weight Integrity

Both the champion and baseline checkpoints were verified on the local filesystem using cryptographically secure SHA-256 hashes:

| Checkpoint | Path | Expected SHA-256 | Observed SHA-256 | Verification Status |
|---|---|---|---|---|
| **Champion (Ep 50)** | `experiments/phase10_model_optimization/phase10b_extended_training/session4_artifacts/session4_best.pt` | `4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd` | `4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd` | **PASSED (Bit-for-Bit Identical)** |
| **Baseline (Phase 6)** | `scratch/kaggle_upload_weights/best.pt` | `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` | `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e` | **PASSED (Bit-for-Bit Identical)** |

*Zero weight modification occurred. Checkpoints are 100% frozen.*

---

## 3. Class ID Mapping and Partitioning

The BDD100K detector uses a 10-class ontology. The partition between VRU and non-VRU classes was verified against `CLASS_NAMES` in `evaluate_pilot_native.py`:

```python
CLASS_NAMES = [
    "person",        # Class ID 0  --> VRU (tau = 0.22, NMS IoU = 0.55)
    "rider",         # Class ID 1  --> VRU (tau = 0.22, NMS IoU = 0.55)
    "car",           # Class ID 2  --> Non-VRU (tau = 0.25, NMS IoU = 0.65)
    "bus",           # Class ID 3  --> Non-VRU (tau = 0.25, NMS IoU = 0.65)
    "truck",         # Class ID 4  --> Non-VRU (tau = 0.25, NMS IoU = 0.65)
    "bike",          # Class ID 5  --> VRU (tau = 0.22, NMS IoU = 0.55)
    "motor",         # Class ID 6  --> VRU (tau = 0.22, NMS IoU = 0.55)
    "traffic light", # Class ID 7  --> Non-VRU (tau = 0.25, NMS IoU = 0.65)
    "traffic sign",  # Class ID 8  --> Non-VRU (tau = 0.25, NMS IoU = 0.65)
    "train"          # Class ID 9  --> Non-VRU (tau = 0.25, NMS IoU = 0.65)
]
VRU_CLASS_IDS = {0, 1, 5, 6}
```
- **Auditor Verification:** Class IDs $0, 1, 5, 6$ map strictly to vulnerable road user classes. No vehicle or static signage class is mapped to VRU status.

---

## 4. Coordinate Transformation & Resolution Handling

The coordinate system alignment between detector predictions and ground truth annotations was audited empirically:
1. **Raw Ground Truth:** Stored as normalized YOLO coordinates $[x_{\text{center}}, y_{\text{center}}, w, h]$ in `bdd100k_dataset/val/labels/`.
2. **Evaluation Ground Truth:** De-normalized using native image dimensions $1280 \times 720$:
   $$x_1 = (x_{\text{center}} - w/2) \times 1280, \quad y_1 = (y_{\text{center}} - h/2) \times 720$$
   $$x_2 = (x_{\text{center}} + w/2) \times 1280, \quad y_2 = (y_{\text{center}} + h/2) \times 720$$
3. **Detector Input & Output:**
   - Image is letterboxed to $960 \times 960$ (scaling factor $960 / 1280 = 0.75$, with $720 \times 0.75 = 540\text{ px}$ height, padded vertically with $210\text{ px}$ top/bottom).
   - Ultralytics `Results.boxes.xyxy` automatically inverts letterbox padding and projects box coordinates back to the original $1280 \times 720$ space.
   - **Empirical Check:** Maximum coordinates measured on validation outputs: $\max(x_2) = 1280.0\text{ px}$, $\max(y_2) = 720.0\text{ px}$. Coordinates natively match the ground truth coordinate space.

---

## 5. NMS Mechanics & Class-Specific Routing Audit

The dual-threshold, dual-NMS execution pipeline was verified:
1. **Model Forward Pass:** Called with base `conf=0.10, iou=0.65, agnostic=False`.
   - YOLO discards candidates with $\text{conf} < 0.10$.
   - Standard greedy NMS runs per-class at $\text{IoU}=0.65$. Under greedy NMS, a box with $\text{conf} \in [0.10, 0.22)$ cannot suppress a box with $\text{conf} \ge 0.22$.
2. **Confidence Filtering:** Vectorized mask separates detections by class:
   $$\text{mask}_i = (\text{score}_i \ge 0.22) \quad \text{if } \text{cls}_i \in \text{VRU} \quad \text{else} \quad (\text{score}_i \ge 0.25)$$
3. **Secondary VRU NMS Suppression:**
   - Surviving VRU detections undergo per-class greedy NMS at $\text{IoU}=0.55$.
   - Any secondary box with $\text{IoU} \in [0.55, 0.65)$ relative to a higher-confidence box of the same VRU class is suppressed.
   - Non-VRU detections remain untouched at $\text{IoU}=0.65$.
4. **Matching Engine (`match_single_image`):**
   - Greedy IoU matching at threshold $\text{IoU} \ge 0.50$.
   - Same-class matching strictly enforced.
   - Area definition: $\text{Area} = (x_2 - x_1) \times (y_2 - y_1)$. Small VRU: $\text{Area} < 1024\text{ px}^2$.

---

## 6. Workspace Preservation & Quarantine Status

- **Phase 10D Outputs:** Preserved intact under `experiments/phase10_model_optimization/phase10d_adaptive_threshold_validation/`.
- **Phase 10E Outputs:** Preserved intact under `experiments/phase10_model_optimization/phase10e_policy_confirmation/`.
- **Test Split Quarantine:** The directory `bdd100k_dataset/test/` has recorded **zero accesses**.

**Auditor Conclusion:** The software implementation matches the Phase 10E protocol without discrepancies. The policy is reproducible from standard code without modifying checkpoint weights.
