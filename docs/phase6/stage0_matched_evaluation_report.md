# RoadSentinel Phase 6 — Stage 0 Matched-Policy Evaluation

**Date:** 2026-10-11
**Status:** COMPLETE
**Kaggle kernel:** `hardikgautam25/roadsentinel-stage-0-matched-evaluation` (Tesla T4, ~62 min, COMPLETE)
**Evaluator:** `evaluate_phase4d_adaptive_area.py` embedded byte-for-byte, sha256 `B6C6D7E2…6088` (the version that produced the Phase 5A-R numbers), ultralytics 8.4.174
**Cohort:** BDD100K val, 9,700-image non-pilot partition (13,633 VRU GT). Test split untouched.
**Artifacts:** [experiments/stage0_matched_eval/stage0/](../../experiments/stage0_matched_eval/stage0/) — per-model metrics, raw detections (conf ≥ 0.01), `operating_points.json/.txt`
**Scripts:** [kaggle_jobs/stage0_matched_eval/build_kernel.py](../../kaggle_jobs/stage0_matched_eval/build_kernel.py), [scripts/compare_stage0.py](../../scripts/compare_stage0.py), [scripts/stage0_operating_points.py](../../scripts/stage0_operating_points.py)

| Model | Checkpoint sha256 |
|---|---|
| Original YOLO11s | `A4747292…D8E3` |
| Phase 5A-U (fine-tuned, class weights inactive) | `BB1823C6…93E5` |
| Phase 5A-R (fine-tuned, class weights active) | `EDCCA4BF…47F0` |

## 1. Reproducibility

| Check | Result |
|---|---|
| Original, raw ActiveFoveation-960 | 8,522 TP / 5,216 FP — reproduces the Phase 3.4 canonical baseline exactly |
| 5A-R, raw and Phase 4C WBF | 8,831 / 7,132 and 8,563 / 4,073 — reproduces the Phase 5A-R evaluation exactly |
| CPU re-fusion from saved raw detections at τ = 0.25 | Within ±2 boxes of the kernel for all models (rounding of saved coordinates) |

## 2. Fixed-threshold results (τ = 0.25, as in the frozen gates)

| Policy | Model | P | R | F1 | Small R | VRU mAP50 |
|---|---|---:|---:|---:|---:|---:|
| Raw AF-960 | Original | 62.03% | 62.51% | 0.6227 | 46.25% | **0.5156** |
| | 5A-U | 57.08% | 64.18% | 0.6042 | 48.47% | 0.5078 |
| | 5A-R | 55.32% | 64.78% | 0.5968 | 49.41% | 0.5115 |
| Phase 4C WBF | Original | **72.86%** | 60.74% | **0.6625** | 43.36% | — |
| | 5A-U | 69.04% | 62.16% | 0.6542 | 45.35% | — |
| | 5A-R | 67.77% | 62.81% | 0.6519 | 46.38% | — |

VRU mAP50 is only meaningful under the raw policy (WBF output is pre-thresholded). Even raw-policy AP is computed from predictions with conf ≥ 0.01 (max 300 per call); §6 shows this truncation can hide up to 0.044 VRU mAP50, more than the gaps between models, so the AP column is not a valid ranking until recomputed at a lower floor.

## 3. Matched-precision results (raw AF-960, threshold swept on CPU)

Highest recall among thresholds (0.01 steps) whose precision meets the target:

| Precision ≥ | Original R / Small R | 5A-U R / Small R | 5A-R R / Small R |
|---:|---|---|---|
| 0.55 | **65.38% / 49.98%** | 64.83% / 49.28% | 64.78% / 49.43% |
| 0.60 | **63.27% / 47.19%** | 62.09% / 45.66% | 61.89% / 45.45% |
| 0.65 | **60.62% / 43.75%** | 59.41% / 42.31% | 59.21% / 41.92% |
| 0.70 | **57.50% / 39.75%** | 56.65% / 38.90% | 55.55% / 37.40% |
| 0.75 | **53.74% / 35.43%** | 52.97% / 34.02% | 51.82% / 32.85% |

Best-F1 threshold per model (selected on val, exploratory): original 0.6323 (τ 0.30) > 5A-U 0.6288 (τ 0.36) > 5A-R 0.6226 (τ 0.36).

## 4. Conclusions

1. **Phase 5A did not improve the detector at any tested operating point.** At every matched precision the original checkpoint has equal or higher overall and small-VRU recall than both fine-tuned models. The fixed-τ recall gains reported for 5A-R come from higher confidence scores (the operating point moves), not from a better precision–recall curve.
2. **Class weighting did not help at matched precision.** 5A-R ≤ 5A-U ≤ original. Weighting raised confidences further (bigger fixed-τ recall gain, bigger precision loss).
3. **The Phase 5A-R "all gates pass" claim does not hold under a matched comparison.** Against the original with the same WBF policy, 5A-R loses 5.09 pp precision and 0.0106 F1 for +2.07 pp recall.
4. **Current best system: original YOLO11s + Phase 4C WBF** (P 72.86%, R 60.74%, F1 0.6625). It still fails the small-VRU gate (43.36% < 45%).
5. **Caveat — alternative explanation:** if the third-party original checkpoint saw val during training, it would be favoured on val, and fine-tuning on train only would dilute that. Val alone cannot separate "fine-tuning hurt" from "original was leak-advantaged". A model trained only on our 65,000-image split (Phase 6) evaluated on val gives the first leakage-free reference.
6. **Latency:** all three ≈ 20–21 ms pipeline P50 on T4; the original's P95 (89 ms) is an outlier, probably warm-up, versus 41–43 ms for the others.

## 5. Implications for Phase 6

- Judge candidates by recall at matched precision and threshold-free AP, not by recall at τ = 0.25.
- Retune thresholds per model (fixed τ = 0.25 favours whichever model is most confident).
- The saved raw detections let all further threshold and fusion work for these three models run on CPU.

## 6. Record closure (2026-10-11)

Script: [scripts/close_stage0_record.py](../../scripts/close_stage0_record.py) → `experiments/stage0_matched_eval/stage0_record.json`.

| Item | Result |
|---|---|
| A1 Archive | sha256 + size of every Stage 0 artifact, the generated kernel source, builder and reports (`A1_archive_manifest`) |
| A2 Reconciliation | **585 checks, 0 failures**: TP+FN = GT (15,373 / 1,740 / 13,633), P/R/F1 re-derived from counts, per-class and size-group sums, pilot + non-pilot additivity, for 3 models × 5 policies × 3 partitions |
| A3 Confidence floor | Raw AP used predictions with conf ≥ 0.01 (truncated). Recall at the floor: person 0.88, bike 0.79–0.81, rider 0.62–0.64, motor 0.65–0.67 |
| A5 Checkpoints | Original, 5A-U best/last, 5A-R best/last: sha256 recorded, files set read-only; Kaggle copies remain in the source dataset/kernel outputs |

**A3 detail — VRU mAP50 bounds** (true untruncated value lies in [floor, upper]; upper assumes every GT missed at the floor is recovered by pure true positives ranked right after it):

| Model | mAP50 at floor 0.01 | Upper bound | Max hidden |
|---|---:|---:|---:|
| Original | 0.5156 | 0.5599 | 0.044 |
| 5A-U | 0.5078 | 0.5336 | 0.026 |
| 5A-R | 0.5113 | 0.5385 | 0.027 |

The between-model AP gaps (≤ 0.008) are smaller than the possible truncation, so **the raw-AP ranking is inconclusive**. The matched-precision conclusions (§3) use thresholds ≥ 0.20 and are unaffected. A lower-floor recompute (conf 0.001, max_det 300; same crops as the canonical pipeline) is prepared as Stage 0b: [kaggle_jobs/stage0_matched_eval/build_kernel_floor.py](../../kaggle_jobs/stage0_matched_eval/build_kernel_floor.py) + [scripts/stage0b_ap_recompute.py](../../scripts/stage0b_ap_recompute.py). The recompute script reproduces the Stage 0 kernel AP from Stage 0 raw detections (0.5156 / 0.5078 / 0.5113 vs 0.5156 / 0.5078 / 0.5115).

**Quarantine disclosure:** the Stage 0 wrapper located checkpoints with a recursive search of `/kaggle/input`, which enumerated directory entries of the dataset's `test/images` folder (file names only; no test image or label was opened, read or scored). Earlier Phase 5A kernels used the same pattern. All new kernels resolve dataset paths explicitly and prune the dataset directory from searches.

## 7. Stage 0b — AP at the standard confidence floor (2026-10-11)

Kernel `hardikgautam25/roadsentinel-stage-0b-floor-dump` (T4, ≈ 35 min, COMPLETE): same models, same ActiveFoveation-960 inference, conf floor 0.001, max_det 300; crops chosen from detections ≥ 0.01 as in the canonical pipeline. Crop counts match Stage 0 (7,666 / 8,240 / 8,171 vs 7,666 / 8,240 / 8,172). Scored on CPU with the evaluator's own fusion and AP functions ([scripts/stage0b_ap_recompute.py](../../scripts/stage0b_ap_recompute.py)); outputs in `experiments/stage0b_floor_dump/stage0b/`.

| Model | VRU mAP50 | VRU mAP50-95 | Residual upper bound (mAP50) | person | rider | bike | motor |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | **0.5157** | **0.2621** | 0.5306 | **0.6542** | 0.4413 | **0.4987** | **0.4687** |
| 5A-U | 0.5077 | 0.2581 | 0.5142 | 0.6490 | 0.4391 | 0.4902 | 0.4524 |
| 5A-R | 0.5122 | 0.2599 | 0.5199 | 0.6415 | **0.4598** | 0.4880 | 0.4597 |

1. Lowering the floor from 0.01 to 0.001 changed VRU mAP50 by ≤ 0.001 for every model; the truncation bound in §6 was loose, not the measurement.
2. The original's measured AP (0.5157) exceeds 5A-U's upper bound (0.5142): original > 5A-U is established. Original vs 5A-R: measured gap +0.0035 mAP50 / +0.0022 mAP50-95 in the original's favour; 5A-R's optimistic bound (0.5199) still overlaps, so that ordering is strongly indicated, not strictly proven.
3. Class weighting helped exactly one class: rider AP50 +1.85 pp (weight 2.0). Person and bike AP fell; motor (weight 2.0) fell.
4. Riders remain the weakest class: only 65–68% of rider ground truth is ever detected above conf 0.001.

These AP results agree with the matched-precision results in §3. Validation-based; the original checkpoint's provenance is uncertain.
