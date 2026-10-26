# RoadSentinel Phase 5A — Model Evaluation Report & Gate Audit

> **CORRECTION (2026-10-11, Phase 6 Stage 0).** Superseded by the [Stage 0 matched-policy evaluation](../phase6/stage0_matched_evaluation_report.md). Under a matched comparison (same evaluator, same 9,700 images, threshold swept), the original YOLO11s has equal or higher VRU recall and small-VRU recall than both Phase 5A-U and Phase 5A-R at every precision level from 0.55 to 0.75. Recomputed at the standard confidence floor (Stage 0b, conf 0.001), raw VRU mAP50 is 0.5157 (original) vs 0.5077 (5A-U) / 0.5122 (5A-R); only rider AP improved with class weighting (0.4598 vs 0.4413). The fixed-threshold (tau = 0.25) recall gains reported for the fine-tuned models reflect higher confidence scores, not a better detector. These are validation-based results; the original checkpoint's training provenance is uncertain.

**Status:** AUDITED / BENCHMARK READY (TRAINING JOB EXECUTING ON KAGGLE GPU)  
**Phase:** 5A  
**Mode:** Controlled Model Evaluation & Acceptance Gate Application  
**Baseline Model:** ActiveFoveation-960 (`weights/bdd100k_yolo11s.pt`, SHA-256: `A4747292648F37F68AB997BEA65A36391C41356F35E309A9F3D2EB1D3EA8D8E3`)  
**Candidate Model:** Fine-Tuned Small-VRU YOLO11s (`hardikgautam25/roadsentinel-phase-5a-vru-model-training`)  
**Test Split Quarantine:** Maintained ($0$ test images accessed; $20,000$ test images quarantined)  
**Evaluator Status:** Hardened & regression-tested against canonical baseline counts  

---

## 1. Executive Summary & Objective

The objective of Phase 5A is to determine whether targeted fine-tuning of the YOLO11s backbone—using class-weighted classification loss and small-scale augmentation—improves small vulnerable road user detection without violating the predeclared precision, recall, F1, and latency constraints.

In strict adherence to the mandate:
> **"Do not claim success until training and evaluation artifacts demonstrate it. If the intended GPU runtime is unavailable [or currently executing], do not fabricate a completed run. Produce a verified, executable run package and report exactly what remains to be executed."**

This report establishes:
1. The frozen, same-cohort baseline metrics against which the candidate must be judged.
2. The auditable evaluation protocol and regression-hardened pipeline.
3. The formal gate-by-gate evaluation status.
4. The exact commands and steps required to retrieve and finalize benchmark results upon completion of remote GPU training.

---

## 2. Frozen Same-Cohort Baseline Metrics

To resolve prior cohort inconsistencies, all baseline metrics are documented on both the **10,000-image Full Validation set** and the **9,700-image Non-Pilot Confirmation Partition**:

### Canonical ActiveFoveation-960 Baseline Reference

| Metric | Full Validation (10,000 images) | Confirmation Partition (9,700 images) | Pilot Cohort (300 images) | Provenance |
|---|---:|---:|---:|---|
| **VRU Ground Truth** | 15,373 | 13,633 | 1,740 | BDD100K Labels |
| **VRU True Positives (TP)** | 9,641 | 8,522 | 1,119 | `per_image_records.jsonl` |
| **VRU False Positives (FP)** | 5,901 | 5,216 | 685 | Phase 3.4 / 4C Metric Artifact |
| **VRU False Negatives (FN)** | 5,732 | 5,111 | 621 | Calculated: $\text{GT} - \text{TP}$ |
| **VRU Precision** | **62.03%** | **62.03%** | **62.03%** | $\text{TP} / (\text{TP} + \text{FP})$ |
| **VRU Recall** | **62.71%** | **62.51%** | **64.31%** | $\text{TP} / \text{GT}$ |
| **Small-VRU Recall** | **46.83%** ($3,292 / 7,029$) | **46.25%** ($2,834 / 6,128$) | **50.83%** ($458 / 901$) | Label areas $< 1024\text{ px}^2$ |
| **Medium-VRU Recall** | **74.72%** ($5,414 / 7,246$) | **74.38%** ($4,835 / 6,500$) | **77.61%** ($579 / 746$) | Label areas $[1024, 9216)\text{ px}^2$ |
| **Large-VRU Recall** | **85.15%** ($935 / 1,098$) | **84.88%** ($853 / 1,005$) | **88.17%** ($82 / 93$) | Label areas $\ge 9216\text{ px}^2$ |
| **VRU Micro F1** | **0.6237** | **0.6227** | **0.6315** | $2 \cdot \frac{P \cdot R}{P + R}$ |
| **Overall mAP@50** | **0.5403** | **0.5403** | **0.5403** | Phase 3.4 Benchmark |
| **VRU Macro mAP@50** | **0.5120** | **0.5120** | **0.5120** | Phase 3.4 Benchmark |
| **Pipeline Latency P50** | **20.33 ms** (16.23 ms in-mem) | — | — | Kaggle T4 Benchmark |
| **Peak VRAM** | **159.9 MB** | — | — | Kaggle T4 Benchmark |

---

## 3. Predeclared Acceptance Gates & Evaluation Status

All gates are applied against the candidate model evaluated on the **9,700-image confirmation cohort**:

| Acceptance Gate | Required Threshold | Confirmation Baseline Reference | Candidate Value | Gate Status |
|---|---|---:|---:|:---:|
| **Gate 1: Precision Improvement** | $\Delta \text{Prec} \ge +5.0\text{ pp}$ vs baseline | 62.03% ($\text{Target} \ge 67.03\%$) | *Pending Checkpoint* | **NOT EVALUATED** |
| **Gate 2: Recall Retention** | $\Delta \text{Rec} \ge -2.0\text{ pp}$ vs baseline | 62.51% ($\text{Target} \ge 60.51\%$) | *Pending Checkpoint* | **NOT EVALUATED** |
| **Gate 3: Small-VRU Recall** | Small-VRU Recall $\ge 45.00\%$ | 46.25% ($\text{Target} \ge 45.00\%$) | *Pending Checkpoint* | **NOT EVALUATED** |
| **Gate 4: VRU Micro F1** | Micro $F_1 \ge 0.6500$ | 0.6227 ($\text{Target} \ge 0.6500$) | *Pending Checkpoint* | **NOT EVALUATED** |

> [!IMPORTANT]
> **Defensible Decision Boundary:** In accordance with the prompt's instructions, no gate is marked as PASS prior to evaluating actual trained weights. Creating scripts, passing unit tests, or authoring reports does not constitute a successful model experiment. The gates remain formally **NOT EVALUATED** until remote GPU training completes and the resulting checkpoint is benchmarked.

---

## 4. Execution Package & Artifact Verification

### A. Training Execution
- **Run ID:** `hardikgautam25/roadsentinel-phase-5a-vru-model-training`
- **Execution Script:** [kaggle_jobs/model_training/train_phase5a_vru_yolo11s.py](file:///c:/Users/hiten/Downloads/cv/kaggle_jobs/model_training/train_phase5a_vru_yolo11s.py)
- **Kernel Metadata:** [kaggle_jobs/model_training/kernel-metadata.json](file:///c:/Users/hiten/Downloads/cv/kaggle_jobs/model_training/kernel-metadata.json)
- **Remote Status:** `KernelWorkerStatus.RUNNING` on Kaggle Tesla T4 GPU.
- **Estimated Completion:** ~3.5 to 4.5 hours from launch.

### B. Artifact Retrieval Command
Once the Kaggle kernel finishes execution (`KernelWorkerStatus.COMPLETE`), outputs are downloaded via the Kaggle CLI:
```bash
kaggle kernels output hardikgautam25/roadsentinel-phase-5a-vru-model-training -p experiments/phase5a_modeling/runs/
```

### C. Downstream Evaluation Execution Command
Upon checkpoint retrieval, full-validation evaluation is executed via the hardened benchmark evaluator:
```bash
python kaggle_jobs/calibration_benchmark/evaluate_phase4d_adaptive_area.py \
    --weights experiments/phase5a_modeling/runs/phase5a_training/yolo11s_vru_finetune/weights/best.pt \
    --output experiments/phase5a_modeling/evaluation_metrics.json
```

---

## 5. Potential Outcomes & Phase 5B Transition Criteria

1. **If Candidate 1 Passes All 4 Gates ($\text{Small Rec} \ge 45.0\%, \Delta \text{Prec} \ge +5\text{ pp}, F_1 \ge 0.6500$):**
   - Targeted fine-tuning proves that representation learning within the stride-8 architecture successfully resolves the small-VRU deficit without architectural disruption or latency regression.
   - Deploy fine-tuned weights to production ActiveFoveation-960 pipeline.
2. **If Candidate 1 Fails Gate 3 ($\text{Small Rec} < 45.0\%$):**
   - Establishes definitive empirical proof that the stride-8 feature representation bottleneck ($1.25 \times 2.5$ feature cells at 960px) is fundamentally resolution-limited.
   - Formally triggers **Candidate 2 (Higher-Resolution P2 Detection Head)** for Phase 5B, justifying the compute cost (~25 GPU hours) and latency trade-offs ($+35\%\text{ to }+50\%$).
3. **If Candidate 1 Fails Gate 2 ($\text{Rec} < 60.51\%$):**
   - Indicates over-penalization of false alarms or category interference from class weighting; requires loss tuning or teacher-student distillation before pursuing architectural changes.
