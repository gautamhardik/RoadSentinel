# RoadSentinel Phase 6 — Checkpoint Selection Protocol and Session Handoff

**Date:** 2026-10-11 · **Status:** implemented and verified on CPU; Session 1 not launched
**Code:** [kaggle_jobs/model_training/train_phase6_vru.py](../../kaggle_jobs/model_training/train_phase6_vru.py) · **Tests:** [tests/test_phase6_selection.py](../../tests/test_phase6_selection.py)

## Fitness (selection metric)

score = round₅( (AP50-95_person + AP50-95_rider + AP50-95_bike + AP50-95_motor) / 4 )

- AP50-95 = mean AP over IoU 0.50:0.05:0.95 (ultralytics `all_ap` row mean), on the fixed 5,000-image training-only holdout (sha256 `c7047ee0…686b`).
- Classes resolved by **name**; data.yaml (dataset), the generated data yaml, `trainer.data` and `model.names` must agree index-for-index. Verified mapping: person 0, rider 1, bike 5, motor 6.
- Holdout contains all four classes (asserted before training). Non-finite AP → 0.0; no AP rows at all → 0.0; a VRU class missing while others present → error.
- round₅: ultralytics rounds every validator metric to 5 decimals; ties at that resolution go to the later epoch.
- Installed only inside training processes (`DetMetrics.fitness`); official-val evaluation keeps stock ultralytics behaviour and is never used for selection.
- Secondary (reported only): all-class mAP50-95, VRU mAP50, per-class VRU AP50 and AP50-95.

## best.pt rules (ultralytics 8.4.174, unchanged)

best_fitness updates only on a strictly higher score; best.pt is rewritten when score == best_fitness (later epoch wins ties); best_fitness is restored from the checkpoint on resume. `best_checkpoint.json` records the epoch, score, file sha256 and weights fingerprint (sha256 of model tensors; survives ultralytics' end-of-run strip) each time best.pt is written. ultralytics' stopper `best_epoch` is not used (it keeps the first tied epoch).

## Session handoff

Session 1 stops after epoch 13 and writes `handoff/`: `resume_last.pt`, `best.pt`, `best_checkpoint.json`, `handoff_manifest.json` (sha256s, best fitness, best epoch, weights fingerprint). Session 2 verifies sha256s, fitness equality (resume checkpoint = best.pt = manifest), fingerprint and record before restoring, then trains epochs 14–26. A lower-scoring Session 2 cannot overwrite best.pt.

## Verification (CPU, real ultralytics trainer, YOLO11n/160 px, 8-epoch schedule split 4 + 4, Kaggle-identical paths)

| Scenario | Scores | Expected winner | Result |
|---|---|---|---|
| A — Session 2 lower | .10 .20 .30 .50 \| .40 .30 .45 .20 | epoch 4 (S1) | PASS: epoch 4, score 0.5, fingerprint `bcb83ea3…` identical before and after the boundary; Session 2 never wrote best.pt |
| B — ties and higher in S2 | .10 .20 .50 .30 \| .50 .40 .60 .60 | epoch 8 | PASS: best.pt rewritten at 5 (tie), 7, 8 (tie); final fingerprint `ecd633ad…`, score 0.6, best_epoch 8 |
| C — real VRU fitness | real | last flagged best (epoch 3) | PASS: fingerprint `ce12b902…` persisted; score = round₅(mean of 4 logged VRU AP50-95) at all 8 epochs |

All three: epochs 1–8 complete, resume at epoch 5, first-batch LR equals the scheduled value every epoch, mosaic on epochs 1–3 and off 4–8, final-eval callbacks tagged separately, official val ran once after selection. Test suite: see final report.
