# RoadSentinel Phase 10D — Adaptive Threshold Validation and Error Audit Protocol

## 1. Executive Objective

Phase 10D serves as the rigorous, empirical validation and error audit for **Config B (Adaptive VRU Confidence Thresholding)** identified as the winning operational candidate from Phase 10C.

The core objectives are:
1. **Reproducibility:** Verify that Config A (Flat 0.25) and Config B (Adaptive VRU 0.20) reproduce the exact reported Phase 10C performance metrics on the 300-image development pilot cohort and the 300-image generalization holdout cohort.
2. **False-Positive Error Audit:** Dissect every false positive added by lowering the VRU threshold from 0.25 to 0.20. Quantify false positives per image, categorize error modes (foliage, shadows, pavement texture, background clutter, signage, class confusion, or uncertain), and inspect visual overlays.
3. **Larger Non-Test Validation:** Evaluate Config A and Config B on an independent, 1,000-image deterministic validation cohort disjoint from all previous cohorts to confirm statistical significance and check threshold sensitivity.
4. **Latency & Computational Audit:** Isolate preprocessing, neural network inference, and post-processing latency to provide measured empirical latency and throughput figures.
5. **Operational Decision:** Formulate an uncompromised production recommendation based on whether Config B satisfies the $\ge 70.0\%$ VRU precision floor across all cohorts without latency penalties.

---

## 2. Invariants and Quarantine Rules

1. **Model Checkpoint Immutability:**
   - Champion Checkpoint: `experiments/phase10_model_optimization/phase10b_extended_training/session4_artifacts/session4_best.pt`
   - Mandatory SHA-256 Checksum: `4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd`
   - Frozen Baseline Checkpoint: `scratch/kaggle_upload_weights/best.pt`
   - Mandatory Baseline SHA-256: `42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e`
   - Zero modifications to weights. No training runs.
2. **Quarantined Test Split:**
   - Directory `bdd100k_dataset/test/` must remain completely untouched (0 file reads, 0 evaluations, 0 tuning).
3. **Ground Truth Consistency:**
   - BDD100K labels scaled to $1280 \times 720$ resolution.
   - Size definitions:
     * Small: $\text{Area} < 1024 \text{ px}^2$ ($< 32^2$)
     * Medium: $1024 \le \text{Area} < 9216 \text{ px}^2$ ($[32^2, 96^2)$)
     * Large: $\text{Area} \ge 9216 \text{ px}^2$ ($\ge 96^2$)
   - VRU classes: `person`, `rider`, `bike`, `motor`.
   - Non-VRU classes: `car`, `bus`, `truck`, `traffic light`, `traffic sign`, `train`.

---

## 3. Policy Specification & Architecture Verification

### Policies Under Evaluation
- **Config A (Baseline):** Flat confidence threshold $0.25$ applied uniformly across all 10 classes.
- **Config B (Adaptive VRU):** 
  $$\tau(c) = \begin{cases} 0.20 & \text{if } c \in \{\text{person, rider, bike, motor}\} \\ 0.25 & \text{if } c \in \{\text{car, bus, truck, traffic light, traffic sign, train}\} \end{cases}$$

### Confidence Filtering and NMS Equivalence
In Ultralytics YOLO11, model execution is called via:
```python
results = model.predict(source=batch_images, imgsz=960, conf=0.10, iou=0.65)
```
1. **Internal Candidate Filtering:** YOLO discards anchor predictions with $\text{conf} < 0.10$.
2. **Per-Class Greedy NMS:** Standard NMS suppresses lower-confidence overlapping boxes with higher-confidence boxes independently within each class (`agnostic=False`). Under greedy NMS, a box with $\text{conf} \in [0.10, 0.20)$ can *never* suppress a box with $\text{conf} \ge 0.20$.
3. **Postprocessing Thresholding:** Applying class-specific filtering $\text{conf} \ge \tau(c)$ on the surviving detections from $\text{conf}=0.10$ produces mathematically identical detections to running NMS with candidates $\ge \tau(c)$, while permitting vectorized multi-threshold routing without additional neural network passes.
4. **Matching Engine:** The evaluator (`match_single_image`) strictly sorts surviving predictions by confidence descending and performs greedy IoU matching ($\text{IoU} \ge 0.50$) against ground truth.

---

## 4. Cohort Partitioning Scheme

All cohorts are deterministically partitioned from `bdd100k_dataset/val/images/` ($N = 10,000$ total validation images):

1. **Cohort 1 — Development Pilot (300 images):**
   - The established `PILOT_300_FILENAMES` set curated in Phase 8 for dense pedestrian and VRU interaction scenes.
   - Ground truth: 6,892 total objects, 1,740 VRUs, 901 small VRUs.
2. **Cohort 2 — Phase 10C Generalization Holdout (300 images):**
   - The first 300 non-pilot images from sorted validation filenames.
   - Ground truth: 5,416 total objects, 367 VRUs, 151 small VRUs.
3. **Cohort 3 — Larger Non-Test Validation Cohort (1,000 images):**
   - The next 1,000 non-pilot, non-Cohort-2 images from sorted validation filenames.
   - Strictly disjoint from Cohorts 1 and 2.
   - Ground truth: 18,230 total objects, 1,495 VRUs, 695 small VRUs.

The complete manifest mapping every image to its cohort and object counts is recorded in `cohort_manifest.csv`.

---

## 5. Decision Criteria

- **Safety-Critical Precision Floor:** VRU Precision must remain $\ge 70.0\%$ across all cohorts.
- **Small-VRU Recall Target:** Config B must demonstrate a positive, reproducible recall increase on small VRUs ($\ge +3.0\text{ pp}$) on both Pilot and Holdout cohorts.
- **Computational Overhead Ceiling:** Thresholding routing must add $< 1.0\text{ ms}$ processing time per frame with zero added forward passes.
- **Failure Fallback:** If precision degrades below $70.0\%$ on the larger 1,000-image cohort, Config B must be rejected and Config A retained as the fallback.
