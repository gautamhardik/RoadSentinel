# RoadSentinel Phase 6 — Smoke Test Report

**Date:** 2026-10-11
**Kaggle kernel:** `hardikgautam25/roadsentinel-phase-6-smoke-test` v1 — COMPLETE (hard stop 80 min; GPU quota used ≈ 0.28 h)
**Hardware/software:** 2× Tesla T4 (15,360 MiB each), torch 2.11.0+cu128, ultralytics 8.4.174, DDP via torchrun
**Code:** [kaggle_jobs/model_training/train_phase6_vru.py](../../kaggle_jobs/model_training/train_phase6_vru.py) (`MODE="smoke"`), identical copy in `kaggle_jobs/phase6_smoke/`
**Artifacts:** [experiments/phase6_smoke/](../../experiments/phase6_smoke/) — `smoke_report.json`, `events.jsonl`, `gpu_util.csv`, phase logs, checkpoints, `rng_state_rank0.pt`, `run_config.json`
**Quarantine:** dataset paths resolved explicitly; the dataset tree (including `test/`) was never walked. Official val was not evaluated.

## 1. Configuration

| Item | Value |
|---|---|
| Model / init | YOLO11m from official COCO weights (`yolo11m.pt`) |
| Training data | 6,500 images = deterministic 10% (seed 7) of the 65,000-image train split; list sha256 `e73df84c…4430` |
| Selection holdout | 5,000 images, sha256 `c7047ee0…686b` (matches the audited manifest; overlap with training asserted empty) |
| Schedule | 2 epochs, imgsz 960, batch 16 total (8 per GPU), SGD lr0 0.01, AMP on, class weights `[1.2, 2.0, 1, 1, 1, 1.8, 2.0, 1, 1, 1]` |
| Resume test | Phase A trained epoch 1 and stopped (simulated session end); phase B resumed from the phase-A checkpoint for epoch 2, sequentially in the same kernel |

## 2. Operational checks

| Check | Evidence | Result |
|---|---|---|
| Two GPUs visible | `['Tesla T4', 'Tesla T4']` | **PASS** |
| DDP world size 2 | trainer `world_size = 2`, 8 images per GPU | **PASS** |
| Both GPUs genuinely utilized | GPU0 mean 89.6% (p10 91.4%), GPU1 mean 90.8% (p10 92.0%); 95 samples each during training | **PASS** |
| GPU memory | Peak 9,641 / 9,557 MiB of 15,360 (≈ 63%); torch max allocated 9.1 GB | **PASS** (≈ 5.7 GB headroom per GPU) |
| Phase A completed | exit 0, 591 s | **PASS** |
| Phase B completed | exit 0, 408 s | **PASS** |
| Class weights active in the real loss | Same 4 images through the live criterion: cls loss 2.2166 weighted vs 2.1374 unweighted; box (1.4283) and DFL (1.1078) identical | **PASS** |
| Checkpoint contents | `ema`, `optimizer`, `scaler`, `epoch`, `updates`, `train_args`, `train_metrics`, `best_fitness` all present | **PASS** |
| Resume starts at epoch 2 | log: "Resuming … from epoch 2 to 2 total epochs"; `start_epoch = 1` | **PASS** |
| Resume restores EMA | EMA updates at phase-B start = 198 = saved value | **PASS** |
| Resume restores optimizer | optimizer state non-empty at phase-B start | **PASS** |
| LR schedule exact after resume | first-batch LR 0.00505 = expected 0.00505 (lr0 × lf(1)) | **PASS** |
| No DDP errors | 0 NCCL errors, tracebacks or timeouts | **PASS** |

Warnings (non-blocking): ultralytics "slow image access" on `/kaggle/input` (≈ 7 MB/s read probe); label cache not writable on the read-only dataset mount (re-scanned each run); a harmless ray-tune `get_trial_id` UserWarning.

**State that cannot be resumed exactly:** scheduler state is not saved (rebuilt from the epoch index; verified exact above); RNG state of rank 1 and dataloader workers is not saved; ultralytics reseeds every rank on resume, so augmentation order after a resume differs from an uninterrupted run. Rank-0 RNG is saved to `rng_state_rank0.pt` for the record.

## 3. Measured timing

| Epoch | Phase | Train | Holdout val (5,000 img) | Train throughput |
|---|---|---:|---:|---:|
| 1 | A | 233.5 s | 57.7 s | 27.8 img/s |
| 2 | B | 237.4 s | 62.3 s | 27.4 img/s |

Startup + dataset scan + final evaluation overhead: ≈ 291 s per launch.

Holdout fitness after 2 epochs (not a quality claim): 0.134 → 0.190.

## 4. Full-session estimate

| Quantity | Value |
|---|---:|
| Train time per full epoch (65,000 img, slowest smoke epoch scaled linearly) | ≈ 2,374 s |
| + holdout validation | ≈ 62 s |
| = per epoch | **≈ 2,436 s (40.6 min)** |
| Session budget | 11 h = 39,600 s |
| − 15% margin, − startup/final eval (291 s), − official-val report (≈ 125 s) | 33,244 s usable |
| **Epochs that fit one session** | **13** |

**Caveat — this estimate is not yet conservative.** With only 2 epochs and `close_mosaic=5`, ultralytics disabled mosaic for the whole smoke run ("Closing dataloader mosaic"), and warmup was capped at 1 epoch instead of 3. A full run uses mosaic in every epoch except the last 5. Mosaic assembles each sample from 4 images, which multiplies image reads and decoding on a 4-vCPU machine whose dataset mount already triggers ultralytics' slow-access warning. GPUs ran at ≈ 90% without mosaic; with mosaic the run may become input-bound. The size of that slowdown was not measured, so **13 epochs is an upper bound**.

## 5. Recommendation (smoke test only)

The training stack is operationally ready: 2-GPU DDP, class weighting, checkpointing and exact-schedule resume all verified on the target hardware. The full-session epoch count should not be fixed from this run alone. See the go/no-go section in the final Stage 0b/Phase 6 summary.
