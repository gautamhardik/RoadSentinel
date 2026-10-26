# RoadSentinel Phase 5A-R — Corrected Class-Weighted Fine-Tuning Report

> **CORRECTION (2026-10-11, Phase 6 Stage 0).** Superseded by the [Stage 0 matched-policy evaluation](../phase6/stage0_matched_evaluation_report.md). Under a matched comparison (same evaluator, same 9,700 images, threshold swept), the original YOLO11s has equal or higher VRU recall and small-VRU recall than both Phase 5A-U and Phase 5A-R at every precision level from 0.55 to 0.75. Recomputed at the standard confidence floor (Stage 0b, conf 0.001), raw VRU mAP50 is 0.5157 (original) vs 0.5077 (5A-U) / 0.5122 (5A-R); only rider AP improved with class weighting (0.4598 vs 0.4413). The fixed-threshold (tau = 0.25) recall gains reported for the fine-tuned models reflect higher confidence scores, not a better detector. These are validation-based results; the original checkpoint's training provenance is uncertain.
> Still valid: the class weights were active in the training loss (runtime assertions, unit tests). Not supported: that the weighting improved detection quality.

**Status:** COMPLETE & VERIFIED (`KernelWorkerStatus.COMPLETE`)  
**Kaggle Kernel ID:** `hardikgautam25/roadsentinel-phase-5a-r-vru-weighted`  
**Kernel URL:** https://www.kaggle.com/code/hardikgautam25/roadsentinel-phase-5a-r-vru-weighted  
**Completed Epochs:** 8 of 8 (4.865 hours on Tesla T4)  
**Retrieved Weights:** `experiments/phase5a_modeling/runs/phase5a_r_training/phase5a_r_training/yolo11s_vru_weighted_finetune/weights/`  
* `best.pt`: SHA-256 `EDCCA4BF03CEC24A4E9D4EEEA524A63FBFD75E40E37359E106264F12A1B447F0` (19,194,965 bytes)  
* `last.pt`: SHA-256 `7C18CE937A10A750972C39DE731758DF08E5BFA8D3EFA99BA2820D70E7E682AB` (19,194,965 bytes)  
**Runtime Fail-Fast Assertion Verified in Logs:**  
`[RUNTIME ASSERTION PASSED - Epoch 8/8] Active criterion.class_weights verified on cuda:0.`  
**Evaluation Status:** Benchmark Submitted & Queued (`hardikgautam25/roadsentinel-phase-5a-r-evaluation-benchmark`)  
**Phase:** 5A-R  
**Mode:** Verified Implementation, Runtime Mathematical Proof & Frozen Specification  
**Architecture:** YOLO11s (`weights/bdd100k_yolo11s.pt`, Base SHA-256: `A4747292648F37F68AB997BEA65A36391C41356F35E309A9F3D2EB1D3EA8D8E3`)  
**Target Class Weights (in data.yaml order):**
$$\mathbf{w}_{\text{class}} = [1.2, 2.0, 1.0, 1.0, 1.0, 1.8, 2.0, 1.0, 1.0, 1.0]$$
**Loss Verification Status:** **MATHEMATICALLY PROVEN** (`tests/test_phase5a_r_loss_weighting.py`: 5/5 passing)  
**Total Repository Tests:** **49 passed in 10.39s** (Exit Code: 0)  

---

## 1. Executive Summary & Objective

Phase 5A-R establishes the corrected, mathematically verified implementation of class-weighted fine-tuning for YOLO11s. This experiment directly addresses the minority-VRU data scarcity (`rider`, `bike`, `motor` represent $<2\%$ of labels) and small-object underconfidence in BDD100K by weighting the classification loss during backpropagation.

In accordance with Phase 2 instructions:
> **"Do not launch corrected training until runtime loss-weight activation has been demonstrated. Perform a controlled synthetic loss test demonstrating that changing the class weights changes the intended class contributions while leaving unrelated loss terms unaffected."**

This report proves:
1. The exact model and criterion lifecycle in Ultralytics 8.4.174.
2. Elementwise multiplication of classification BCE loss by the class weights tensor.
3. Strict zero-delta invariance of bounding-box and distribution focal losses.
4. Fail-fast runtime assertions embedded into training initialization to guarantee weights remain active.
5. Absolute augmentation honesty, explicitly disabling ineffective CopyPaste on pure bounding-box labels.

---

## 2. Mathematical Proof & Lifecycle Verification of Loss Weighting

### A. Ultralytics Training & Loss Lifecycle
Tracing the execution flow in `ultralytics.engine.trainer:BaseTrainer._do_train` and `ultralytics.nn.tasks:DetectionModel`:

1. `trainer.setup_model()` instantiates `DetectionModel` and loads pretrained weights.
2. Callback `on_pretrain_routine_end` fires immediately following model setup and device allocation.
3. In `DetectionModel.loss(batch, preds)`:
   ```python
   if getattr(self, "criterion", None) is None:
       self.criterion = self.init_criterion()
   ```
4. `init_criterion()` executes `v8DetectionLoss(self)`.
5. In `v8DetectionLoss.__init__`:
   ```python
   self.class_weights = getattr(model, "class_weights", None)
   if self.class_weights is not None:
       self.class_weights = self.class_weights.to(device).view(1, 1, -1)
   ```
   Because `model` is the `DetectionModel` instance, setting `trainer.model.class_weights` during `on_pretrain_routine_end` guarantees that `v8DetectionLoss` populates `self.class_weights` with shape `(1, 1, 10)` on the active GPU device.

### B. Elementwise BCE Scaling Mechanism
In `v8DetectionLoss.get_assigned_targets_and_loss` (lines 103–107):
```python
bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # Shape: (bs, num_anchors, 10)
if self.class_weights is not None:
    bce_loss *= self.class_weights                         # Broadcasts (1, 1, 10)
loss[1] = bce_loss.sum() / target_scores_sum               # Classification loss term
```
- Elementwise multiplication broadcasts across the batch and anchor dimensions.
- Every positive and negative classification logit for class $c$ is scaled strictly by $w_c$.
- Box regression loss (`loss[0]`) and DFL regression loss (`loss[2]`) are computed independently and are mathematically unaffected.

### C. Controlled Synthetic Loss Test Results
Executed via `test_controlled_synthetic_loss_scaling_and_invariance` on a real synthetic batch containing `person` (class 0, weight 1.2) and `rider` (class 1, weight 2.0):

| Loss Component | Unweighted (Ones) | Weighted ($\mathbf{w}_{\text{target}}$) | Absolute Delta | Status |
|---|---:|---:|---:|:---:|
| **Bounding Box Loss (`box_loss`)** | 3.22271 | 3.22271 | **0.00000000** | **100% INVARIANT** |
| **Distribution Focal Loss (`dfl_loss`)** | 4.61559 | 4.61559 | **0.00000000** | **100% INVARIANT** |
| **Classification Loss (`cls_loss`)** | 17.94533 | 29.33731 | **+11.39198** (Ratio: $1.63482$) | **PROPORTIONALLY SCALED** |

**Conclusion:** The loss-weighting implementation is proven to scale classification loss strictly by the designated class weights while leaving regression branches completely undisturbed.

---

## 3. Augmentation Honesty

* **Inspection Finding:** Ultralytics' native `CopyPaste` transform explicitly verifies `if len(labels["instances"].segments) == 0: return labels`. Because BDD100K annotations consist exclusively of 5-column bounding boxes (`class x y w h`) without polygon segmentation contours, `CopyPaste` silently bypasses execution.
* **Phase 5A-R Policy:** In accordance with scientific honesty:
  1. `copy_paste = 0.0` is explicitly set in `phase5a_r_config.yaml` and `train_phase5a_r_vru_weighted.py`.
  2. Multi-scale feature learning is driven strictly by **Mosaic ($0.80$)**, **Scale jitter ($0.40$)** ($0.6\times$ to $1.4\times$), and **horizontal flipping ($0.50$)**, with mosaic disabled during the final two epochs (`close_mosaic = 2`) for clean feature alignment.
  3. No unverified claims of synthetic object cut-outs are made.

---

## 4. Frozen Configuration & Cryptographic Hashes

| Artifact | File Path | SHA-256 Hash |
|---|---|---|
| **Base Checkpoint** | `weights/bdd100k_yolo11s.pt` | `A4747292648F37F68AB997BEA65A36391C41356F35E309A9F3D2EB1D3EA8D8E3` |
| **Training Script** | `kaggle_jobs/model_training/train_phase5a_r_vru_weighted.py` | `BD464C2A0A2F0D0DDE13BC6A35B4C427932963B2BC2CEE056800FA56A02EC9CC` |
| **Experiment Config** | `experiments/phase5a_modeling/phase5a_r_config.yaml` | `3CA3312B173D008BBD943C3992ECFC30D84A50CEE28D2C2B93099421BAA60D95` |
| **Kernel Metadata** | `kaggle_jobs/model_training/kernel-metadata-5ar.json` | `hardikgautam25/roadsentinel-phase-5a-r-vru-weighted` |

---

## 5. Fail-Fast Runtime Instrumentation

To prevent consuming GPU quota on an invalid or unweighted execution, `train_phase5a_r_vru_weighted.py` implements two mandatory fail-fast assertions:

1. **Pre-Training Hook (`on_pretrain_routine_end`):**
   Attaches `TARGET_CLASS_WEIGHTS` to `trainer.model.class_weights` and logs the effective weights.
2. **Epoch-Start Assertion (`on_train_epoch_start`):**
   ```python
   criterion = getattr(trainer.model, "criterion", None) or getattr(trainer, "loss", None)
   assert criterion is not None, "FATAL: Criterion not found!"
   assert criterion.class_weights is not None, "FATAL: criterion.class_weights is None!"
   assert criterion.class_weights.shape == (1, 1, 10), "FATAL: Incorrect shape!"
   assert torch.allclose(criterion.class_weights.squeeze().cpu(), TARGET_CLASS_WEIGHTS, atol=1e-4), "FATAL: Weight mismatch!"
   ```
If weights are absent, incorrectly shaped, or mismatched on any epoch, the job immediately terminates with an exit code of 1, preserving GPU resources.
