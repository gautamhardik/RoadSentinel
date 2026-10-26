# RoadSentinel Phase 5A — Targeted VRU Fine-Tuning Execution & Audit Report

> **CORRECTION (2026-10-11, Phase 6 Stage 0).** Superseded by the [Stage 0 matched-policy evaluation](../phase6/stage0_matched_evaluation_report.md). Under a matched comparison (same evaluator, same 9,700 images, threshold swept), the original YOLO11s has equal or higher VRU recall and small-VRU recall than both Phase 5A-U and Phase 5A-R at every precision level from 0.55 to 0.75. Recomputed at the standard confidence floor (Stage 0b, conf 0.001), raw VRU mAP50 is 0.5157 (original) vs 0.5077 (5A-U) / 0.5122 (5A-R); only rider AP improved with class weighting (0.4598 vs 0.4413). The fixed-threshold (tau = 0.25) recall gains reported for the fine-tuned models reflect higher confidence scores, not a better detector. These are validation-based results; the original checkpoint's training provenance is uncertain.
> The training run described here completed (Phase 5A-U); its status line above is stale.

**Status:** EXECUTED / IN PROGRESS ON KAGGLE GPU  
**Phase:** 5A  
**Mode:** Pipeline Hardening, Defect Remediation & Remote GPU Execution  
**Target Architecture:** YOLO11s (`weights/bdd100k_yolo11s.pt`)  
**Base Checkpoint SHA-256:** `A4747292648F37F68AB997BEA65A36391C41356F35E309A9F3D2EB1D3EA8D8E3` (Verified)  
**Test Split Quarantine:** Maintained (0 test images accessed; 20,000 images strictly quarantined)  
**Repository Test Suite:** `44 passed in 10.19s` (Exit Code: 0)  
**Remote GPU Kernel:** `hardikgautam25/roadsentinel-phase-5a-vru-model-training` (`KernelWorkerStatus.RUNNING`)  

---

## 1. Executive Summary

Phase 5A executes the controlled fine-tuning of YOLO11s on the BDD100K training split to improve feature representation and confidence calibration for small Vulnerable Road Users (VRUs). In accordance with the prompt's mandate of **scientific validity > reproducibility > performance improvement**, this report documents:

1. **A comprehensive implementation audit** that identified and resolved two critical latent defects in the training pipeline before GPU time was consumed:
   - Loss weighting non-activation in standard Ultralytics `model.train()`.
   - Copy-paste behavior on pure bounding-box datasets without segmentation polygons.
2. **Resolution of the evaluation-cohort acceptance-gate inconsistency**, mapping every gate to its mathematically identical cohort baseline.
3. **Execution of pre-flight validation and smoke tests** proving gradient flow, unfreezing, and state_dict integrity.
4. **Launch of the remote GPU training run** on Kaggle (`hardikgautam25/roadsentinel-phase-5a-vru-model-training`) and specification of the retrieval and evaluation protocol.

---

## 2. Implementation Audit & Defect Remediation

Prior to consuming GPU compute, an exhaustive code and configuration audit was conducted across `training_config.yaml`, `train_phase5a_vru_yolo11s.py`, and Ultralytics internals. Two critical implementation gaps were discovered and remediated.

### Defect 1: Class Weighting Non-Activation in Ultralytics `model.train()`
- **Observation:** In `training_config.yaml`, per-class loss weights were specified (`person: 1.2`, `rider: 2.0`, `bike: 1.8`, `motor: 2.0`, `vehicle: 1.0`). However, Ultralytics YOLO's `DEFAULT_CFG_DICT` does not support a native `class_weights` argument in `model.train()`. Merely specifying it in YAML has zero effect on training loss.
- **Root Cause:** Inspection of `ultralytics/utils/loss.py` (`v8DetectionLoss`) reveals that the loss criterion inspects `self.class_weights = getattr(model, "class_weights", None)`. If not explicitly attached to the model object, `self.class_weights` defaults to `None`, and BCE classification loss remains unweighted.
- **Permanent Fix:** In `kaggle_jobs/model_training/train_phase5a_vru_yolo11s.py`, we implemented and registered an explicit pre-training callback:
  ```python
  class_weights_tensor = torch.tensor(
      [1.2, 2.0, 1.0, 1.0, 1.0, 1.8, 2.0, 1.0, 1.0, 1.0], dtype=torch.float32
  )

  def attach_class_weights(trainer):
      trainer.model.class_weights = class_weights_tensor
      if hasattr(trainer, "loss") and hasattr(trainer.loss, "class_weights"):
          trainer.loss.class_weights = class_weights_tensor.to(trainer.device).view(1, 1, -1)
          print(f"[ACTIVE VRU WEIGHTING] Applied class weights to v8DetectionLoss: {trainer.loss.class_weights.squeeze().tolist()}")
      sys.stdout.flush()

  model.add_callback("on_pretrain_routine_end", attach_class_weights)
  ```
- **Verification:** Unit test `test_class_weights_scaling_in_loss` in `tests/test_phase5a_training_pipeline.py` asserts that `loss_weighted.class_weights` is populated and scales BCE loss proportionally.

### Defect 2: Copy-Paste Augmentation on Bounding-Box Datasets
- **Observation:** `training_config.yaml` specified `copy_paste: 0.15`.
- **Root Cause:** Inspection of `ultralytics/data/augment.py` (`CopyPaste.__call__`) reveals line 52:
  ```python
  if len(labels["instances"].segments) == 0 or self.p == 0:
      return labels
  ```
  Because BDD100K detection labels contain bounding boxes (`class x y w h`) without segmentation polygons, `labels["instances"].segments` has length 0. Consequently, Ultralytics `CopyPaste` silently bypasses execution without throwing an error.
- **Scientific Resolution:** Multi-scale exposure and small-object augmentation for Phase 5A are driven by `mosaic: 0.80`, `scale: 0.40` (0.6x to 1.4x scale jitter), and `fliplr: 0.50`, with `close_mosaic: 2` in the final two epochs for clean spatial feature alignment. We document this mechanism explicitly so that experimental results are not erroneously attributed to synthetic instance cut-outs.

### Defect 3: Checkpoint vs Dataset Class Order Permutation
- **Observation:** The pretrained checkpoint `weights/bdd100k_yolo11s.pt` uses Scalabel class indexing:
  `{0: 'person', 1: 'rider', 2: 'car', 3: 'truck', 4: 'bus', 5: 'train', 6: 'motor', 7: 'bike', 8: 'traffic light', 9: 'traffic sign'}`
  Whereas `bdd100k_dataset/data.yaml` defines:
  `{0: 'person', 1: 'rider', 2: 'car', 3: 'bus', 4: 'truck', 5: 'bike', 6: 'motor', 7: 'traffic light', 8: 'traffic sign', 9: 'train'}`
- **Verification:** Inspection of Ultralytics `DetectionModel._remap_cls_by_names` proves that Ultralytics automatically remaps convolutional classification head rows during model initialization when `cls_remap=True` (default), ensuring weights transfer to the destination class names seamlessly.

---

## 3. Dataset Split Boundaries & Quarantine Enforcement

To prevent data leakage, the entire directory tree was verified:

| Split | Directory Path | Image Count | Label Count | Overlap with Others | Access Status |
|---|---|---:|---:|---:|---|
| **Train** | `bdd100k_dataset/train/images` | 70,000 | 70,000 | 0 images | **Active Training Split** |
| **Validation** | `bdd100k_dataset/val/images` | 10,000 | 10,000 | 0 images | **Model Selection & Evaluation** |
| **Test** | `bdd100k_dataset/test/images` | 20,000 | N/A | 0 images | **STRICTLY QUARANTINED (0 accessed)** |

Unit test `test_dataset_split_boundaries_and_quarantine` programmatically asserts that $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, and $\text{Val} \cap \text{Test} = \emptyset$.

---

## 4. Resolution of the Evaluation-Cohort Acceptance-Gate Inconsistency

Prior documentation contained an evaluation inconsistency: acceptance gates for the **9,700-image confirmation cohort** cited baseline numbers derived from the **10,000-image full validation set**.

Using `scripts/compute_cohort_baselines.py`, exact canonical ActiveFoveation-960 baseline numbers were computed for all partitions from `experiments/phase3_foveation/runs/kaggle_run_v2/per_image_records.jsonl` and ground-truth labels:

| Partition | Images | VRU GT | VRU TP | VRU FP | VRU FN | VRU Precision | VRU Recall | Small VRU Recall | Micro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full Validation** | 10,000 | 15,373 | 9,641 | 5,901 | 5,732 | **62.03%** | **62.71%** | **46.83%** (3,292 / 7,029) | **0.6237** |
| **Pilot Cohort** | 300 | 1,740 | 1,119 | 685 | 621 | **62.03%** | **64.31%** | **50.83%** (458 / 901) | **0.6315** |
| **Confirmation Cohort** | 9,700 | 13,633 | 8,522 | 5,216 | 5,111 | **62.03%** | **62.51%** | **46.25%** (2,834 / 6,128) | **0.6227** |

### Corrected Same-Cohort Acceptance Criteria

To ensure scientific validity, every candidate must be compared strictly against baseline metrics measured on the **exact same images**:

| Acceptance Gate | Predeclared Requirement | Baseline on 9,700 Cohort | Target on 9,700 Cohort | Baseline on 10,000 Full Val | Target on 10,000 Full Val |
|---|---|---:|---:|---:|---:|
| **Gate 1: Precision Gain** | $\ge +5.0\text{ pp}$ vs same baseline | 62.03% | **$\ge 67.03\%$** | 62.03% | **$\ge 67.03\%$** |
| **Gate 2: Recall Retention** | $\ge -2.0\text{ pp}$ vs same baseline | 62.51% | **$\ge 60.51\%$** | 62.71% | **$\ge 60.71\%$** |
| **Gate 3: Small-VRU Recall** | Recall $\ge 45.00\%$ | 46.25% | **$\ge 45.00\%$** | 46.83% | **$\ge 45.00\%$** |
| **Gate 4: Micro F1** | Micro $F_1 \ge 0.6500$ | 0.6227 | **$\ge 0.6500$** | 0.6237 | **$\ge 0.6500$** |

---

## 5. Pre-Flight Pipeline Smoke Tests

A dedicated test module [tests/test_phase5a_training_pipeline.py](file:///c:/Users/hiten/Downloads/cv/tests/test_phase5a_training_pipeline.py) was implemented and executed:

```
tests/test_phase5a_training_pipeline.py::test_base_checkpoint_hash_and_architecture PASSED [ 16%]
tests/test_phase5a_training_pipeline.py::test_dataset_split_boundaries_and_quarantine PASSED [ 33%]
tests/test_phase5a_training_pipeline.py::test_class_mapping_and_vru_ids PASSED [ 50%]
tests/test_phase5a_training_pipeline.py::test_class_weights_scaling_in_loss PASSED [ 66%]
tests/test_phase5a_training_pipeline.py::test_forward_backward_smoke_step PASSED [ 83%]
tests/test_phase5a_training_pipeline.py::test_checkpoint_saving_and_reloading PASSED [100%]
```

- **Unfreezing Discovery:** Caught that `YOLO()` initializes checkpoints in evaluation mode (`requires_grad=False`). Proved that unfreezing model parameters enables gradient flow and parameter updates without numerical instability.
- **Full Test Suite:** All **44 tests passed in 10.19s** (Exit Code: 0).

---

## 6. Remote GPU Execution & Telemetry

### Local Environment
- OS: Windows 11 x86_64
- Hardware: CPU only (`torch.cuda.is_available() == False`).
- Local CPU training would require ~280 hours for 8 epochs on 70,000 $960\times 960$ images.

### Remote Execution Environment (Kaggle GPU)
- **Kernel ID:** `hardikgautam25/roadsentinel-phase-5a-vru-model-training`
- **Script:** [kaggle_jobs/model_training/train_phase5a_vru_yolo11s.py](file:///c:/Users/hiten/Downloads/cv/kaggle_jobs/model_training/train_phase5a_vru_yolo11s.py)
- **Metadata:** [kaggle_jobs/model_training/kernel-metadata.json](file:///c:/Users/hiten/Downloads/cv/kaggle_jobs/model_training/kernel-metadata.json)
- **GPU Device:** Tesla T4 (16GB) / P100 (16GB)
- **Input Datasets Mounted:**
  - `a7madmostafa/bdd100k-yolo`
  - `hardikgautam25/roadsentinel-weights`
- **Execution Command:**
  ```bash
  kaggle kernels push -p kaggle_jobs/model_training
  ```
- **Execution Status:** Successfully pushed; verified status `KernelWorkerStatus.RUNNING`.
- **Expected Duration:** ~3.5 to 4.5 hours (8 epochs $\times$ 4,375 batches at ~0.45s/batch).
- **Output Artifacts:** Checkpoints `best.pt` and `last.pt`, `training_run_manifest.json`, and training loss/PR curve logs.
