# RoadSentinel Phase 6 — Mosaic Speed Probe: Plan for Review

**Date:** 2026-10-11
**Status:** PREPARED — not pushed. Awaiting approval.
**Kernel folder:** `kaggle_jobs/phase6_probe/` (`hardikgautam25/roadsentinel-phase-6-mosaic-probe`)
**Code:** [kaggle_jobs/model_training/train_phase6_vru.py](../../kaggle_jobs/model_training/train_phase6_vru.py) with `MODE="probe"` (same file as the smoke test; smoke behaviour unchanged)
**Estimate script (local, CPU):** [scripts/phase6_schedule_estimate.py](../../scripts/phase6_schedule_estimate.py)

## 1. Why

The smoke test measured 27.4 img/s, but with only 2 epochs and `close_mosaic=5` ultralytics disabled mosaic for the whole run. A full run uses mosaic in all but the last 5 epochs. Mosaic builds each sample from 4 images, on a 4-vCPU machine with a slow dataset mount, so its throughput must be measured before an epoch budget is fixed.

## 2. Configuration (identical to the smoke test except mosaic)

| Item | Value |
|---|---|
| Hardware | Kaggle 2× T4, DDP via torchrun (`machine_shape: NvidiaTeslaT4`) |
| Model / init | YOLO11m, official COCO weights |
| Training data | Same deterministic 10% (6,500 images, seed 7) of the 65,000-image split; holdout overlap asserted empty before training |
| Holdout | 5,000 images, sha256 `c7047ee0…686b` (validation only, never in gradient updates) |
| Epochs | 2 |
| Mosaic | `mosaic=1.0`, **`close_mosaic=0`** (ultralytics never closes mosaic) |
| Other augmentation | Same as planned full run: scale 0.5, translate 0.1, fliplr 0.5, HSV (0.015, 0.7, 0.4), mixup 0, copy-paste 0 |
| Optimizer | SGD lr0 0.01, lrf 0.01, warmup (capped by ultralytics at epochs − 1 = 1), AMP, batch 16 (8/GPU), workers 4 |
| Class weights | `[1.2, 2.0, 1, 1, 1, 1.8, 2.0, 1, 1, 1]` (as in the smoke test; the full-run choice is separate) |
| Test split / official val | Not read. Dataset paths explicit; no recursive walk of the dataset tree |

## 3. What it records

| Measurement | How |
|---|---|
| Mosaic actually active | (a) effective trainer args `mosaic=1.0`, `close_mosaic=0`; (b) at the end of every epoch: dataset `augment=True`, `rect=False`, and the live pipeline contains `Mosaic(p=1.0)`; (c) the log never contains "Closing dataloader mosaic" |
| Steady-state throughput | Per-batch timings on **every rank**, CUDA-synchronized; first 20 steps per epoch excluded; global img/s over 25-step windows → p10 / p50 / p90 |
| Data-loading overhead | Time each step waits on the dataloader vs compute time; reported as a fraction plus median wait per step |
| GPU utilization / memory | `nvidia-smi` every 2 s, both GPUs, inside training windows only |
| CPU pressure | Host CPU % every 2 s (psutil) and core count |
| Validation overhead | Holdout validation + checkpoint save time per epoch |
| Launch overhead | Startup, dataset scan, final evaluation |
| Provenance | `run_config.json`, `effective_train_args.json`, `augmentation_pipeline.txt`, `events.jsonl`, `gpu_util.csv`, `cpu_util.csv`, per-rank batch timings, logs, sha256 of `last.pt`, `best.pt` and config files |

The kernel reports **measurements only** (`probe_report.json`). Extrapolation to 65,000 images is done separately and labelled as such.

## 4. Hard limits

| Limit | Value |
|---|---|
| Kaggle session timeout | `--timeout 2700` (45 min) |
| Launcher kill switch for the training process | 35 min |
| In-training time guard | 0.6 h |
| Expected duration | ≈ 15–30 min, depending on the mosaic slowdown |
| Expected quota use | ≤ 0.75 h (13.71 h available, resets 2026-10-17) |

## 5. Pre-run verification (CPU dry run, scaled down: YOLO11n, imgsz 160, 195 images)

| Check | Result |
|---|---|
| End-to-end run, report written, no warnings | PASS |
| Effective args show mosaic 1.0 / close_mosaic 0; no close in log | PASS |
| Live pipeline contains `Mosaic(p=1.0)` both epochs | PASS (after fixing the runtime check — see below) |
| Negative control: `close_mosaic=1` must be detected | PASS (all three mosaic checks FAIL as intended; epoch 2 shows `Mosaic p=0.0`) |
| Batch timing captured, steady-state computed, monitors summarized | PASS |
| GPU checks | Not testable on CPU (expected FAIL on CPU) |

Fix found by the dry run: ultralytics 8.4.174 has no runtime `dataset.mosaic` flag, so the first version of the check read it as False. The check now uses `augment`, `rect` and the live `Mosaic` transform probability, which is what ultralytics itself uses.

## 6. How the full-run budget will be computed (after the probe)

All inputs are measured; only the 6,500 → 65,000-image scaling is an extrapolation.

| Scenario | Mosaic epochs | Close-mosaic epochs |
|---|---|---|
| Optimistic | 65,000 / p90 steady-state rate | fastest measured no-mosaic rate (smoke) |
| Typical | 65,000 / p50 steady-state rate | median no-mosaic rate |
| Conservative | 65,000 / min(p10 steady-state, slowest epoch-level rate) | slowest no-mosaic rate |

Each epoch adds the slowest measured holdout validation + checkpoint time. Each session subtracts launch overhead and a final holdout evaluation of `best.pt`; session 2 also subtracts the official-val report (≈ 2× holdout validation). Usable budget per session = 11 h × 0.85. **The fixed total epoch budget is taken from the conservative scenario.**

## 7. Proposed two-session schedule (structure fixed now; numbers filled from the probe)

| Element | Proposal |
|---|---|
| Total epochs E | Fixed before session 1 = conservative two-session capacity (§6) |
| Split | Session 1: epochs 1…E1; session 2: epochs E1+1…E, at a fixed epoch boundary (`SESSION_STOP_EPOCH = E1`), not time-based |
| LR behaviour | SGD lr0 0.01, 3 warmup epochs (bias LR 0.1 → lr0, momentum 0.8 → 0.937), then linear decay lf(e) = (1 − e/E)(1 − 0.01) + 0.01 over the **total** E. Session 2 resumes the same schedule; the smoke test verified that the first post-resume LR equals the expected value exactly |
| State across resume | EMA, optimizer, AMP scaler, epoch, best fitness restored (verified). Not restored: rank-1/worker RNG (augmentation order differs from an uninterrupted run — documented) |
| Mosaic | On for epochs 1…E−5, off for the last 5 (all in session 2) |
| Checkpoint selection | Fixed protocol: after every epoch ultralytics validates the EMA weights on the 5,000-image training-only holdout; fitness = **all-class mAP50-95** (ultralytics 8.4.174 weights [P, R, mAP50, mAP50-95] = [0, 0, 0, 1]); `best.pt` = the epoch with the highest holdout fitness. The selected checkpoint is then evaluated once on official val, for the record only — never used for selection or tuning. Note: the holdout fitness is all-class, not VRU-specific |
| Stop: planned | Session 1 stops after epoch E1 and saves `resume_last.pt`; session 2 stops after epoch E |
| Stop: time | If the projected next epoch would exceed the session budget, stop early and save `resume_last.pt`; E is **not** silently changed — the shortfall goes to review |
| Stop: hard | Kaggle `--timeout` 11.5 h per session; Kaggle's 12 h limit is never relied on |
| Stop: abort and report | Operational failures only: unrecovered NaN/Inf loss, DDP/NCCL error, or a crashed/timed-out process. No metric-based automatic abort (holdout fitness is noisy epoch to epoch; any such rule would have to be justified and fixed in advance). Ultralytics early stopping (`patience=100`) cannot trigger within this schedule |
| Review gate between sessions | Holdout curve, GPU/CPU stats, `resume_last.pt` hash and contents; session 2 only after explicit approval (expected after the quota reset on 2026-10-17) |
| Not automatic | Neither session starts without approval; session 2 never auto-launches |

Decision (2026-10-11): the planned Phase 6 experiment keeps the existing class weights `[1.2, 2.0, 1, 1, 1, 1.8, 2.0, 1, 1, 1]`. The training configuration is not changed based on the probe.

## 8. Negative-control result

| Run | mosaic_in_effective_args | mosaic_active_every_epoch | mosaic_never_closed | Live `Mosaic p` (epoch 1, 2) |
|---|---|---|---|---|
| As configured (`close_mosaic=0`) | PASS | PASS | PASS | 1.0, 1.0 |
| Negative control (`close_mosaic=1`) | FAIL | FAIL | FAIL | 1.0, 0.0 |

The verification distinguishes active from closed mosaic.
