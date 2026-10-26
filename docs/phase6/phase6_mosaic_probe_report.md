# RoadSentinel Phase 6 — Mosaic Speed Probe Report and Full-Run Proposal

**Date:** 2026-10-11
**Kaggle kernel:** `hardikgautam25/roadsentinel-phase-6-mosaic-probe` v1 — COMPLETE (hard stop 45 min; GPU quota used 0.21 h; 13.50 h remaining until 2026-10-17)
**Plan:** [phase6_mosaic_probe_plan.md](phase6_mosaic_probe_plan.md)
**Artifacts:** [experiments/phase6_probe/](../../experiments/phase6_probe/) — `probe_report.json`, `effective_train_args.json`, `augmentation_pipeline.txt`, per-rank `batch_timing_*.json`, `gpu_util.csv`, `cpu_util.csv`, `events.jsonl`, logs, checkpoints; estimate: `schedule_estimate.json`
**Quarantine:** explicit dataset paths; test split never read; official val not evaluated.

## 1. Operational checks

| Check | Evidence | Result |
|---|---|---|
| Two GPUs, DDP world size 2 | 2× Tesla T4, 8 images per GPU | PASS |
| Both GPUs utilized | GPU0 mean 92.5% (p10 91.0%), GPU1 mean 92.8% (p10 92.0%), 198 samples each | PASS |
| Mosaic in effective args | `mosaic=1.0`, `close_mosaic=0` | PASS |
| Mosaic active during both epochs | end of epoch 1 and 2: `augment=True`, `rect=False`, live pipeline `Mosaic(p=1.0)` | PASS |
| Mosaic never closed | no "Closing dataloader mosaic" in log | PASS |
| Holdout excluded from gradient updates | overlap asserted empty before training | PASS |
| Run completed, no DDP errors | exit 0, 750 s; only slow-mount, read-only cache and ray-tune warnings | PASS |

## 2. Measured values (6,500-image subset, mosaic on)

| Quantity | Epoch 1 | Epoch 2 |
|---|---:|---:|
| Train time | 203.1 s | 195.3 s |
| Holdout validation + checkpoint save | 54.3 s | 47.3 s |
| **Epoch duration incl. validation and checkpoint** | **257.4 s** | **242.6 s** |
| Epoch-level throughput | 32.0 img/s | 33.3 img/s |
| Steady-state global throughput p10 / p50 / p90 (rank 0) | 33.9 / 34.2 / 35.7 | 32.9 / 33.3 / 33.7 |
| Median step | 0.464 s | 0.482 s |
| Median data wait per step | 0.3 ms | 0.4 ms |
| Peak GPU memory (torch allocated) | 9.25 GB | 9.28 GB |

All ranks and epochs (1,544 steps after excluding the first 20 per epoch): **32.9 / 33.8 / 35.2 img/s** (p10 / p50 / p90); data wait = **0.1%** of step time. GPU memory peak 9,631 / 9,537 MiB of 15,360. Host CPU (4 cores) mean 66%, p90 69%, load average 2.9. Launch overhead (startup, dataset scan, final holdout evaluation): 250 s.

**Interpretation (measured):** the run is GPU-bound, not input-bound. Mosaic did not slow training. Mosaic-on epochs here (32–33 img/s) were faster than the mosaic-off smoke epochs in a different session (27.4–27.8 img/s); since data loading is negligible in both, the difference is most plausibly session-to-session hardware variance on Kaggle, not augmentation cost. This variance is carried into the conservative estimate.

## 3. Extrapolation to the full run (linear scaling 6,500 → 65,000 images; labelled estimate)

| Scenario | Rate used (mosaic / last-5 no-mosaic) | Full epoch incl. val + checkpoint | Session 1 epochs | Session 2 epochs | Total |
|---|---|---:|---:|---:|---:|
| Optimistic | 35.2 / 27.8 img/s | 0.53 h | 17 | 16 | 33 |
| Typical | 33.8 / 27.8 img/s | 0.55 h | 16 | 15 | 31 |
| **Conservative** | **27.4 / 27.4 img/s** (slowest rate measured in any session, applied to every epoch) | **0.68 h** | **13** | **13** | **26** |

Session model: usable budget 11 h × 0.85 = 9.35 h; minus 291 s launch overhead (largest measured) per session; minus ≈ 125 s official-val report in session 2; per-epoch validation = slowest measured (62.3 s); last 5 epochs costed at the no-mosaic rate. The 3-epoch warmup changes the learning rate and gradient accumulation only, not throughput, so it needs no extra time.

## 4. Proposed full Phase 6 run (for review — not launched)

| Item | Proposal |
|---|---|
| Model / data | YOLO11m, COCO init, full 65,000-image training split; 5,000-image training-only holdout (`c7047ee0…686b`) |
| Configuration | Unchanged from the smoke/probe runs: imgsz 960, batch 16 (8/GPU), SGD lr0 0.01, lrf 0.01, AMP, mosaic 1.0, close_mosaic 5, scale 0.5, fliplr 0.5, mixup 0, copy-paste 0, class weights `[1.2, 2.0, 1, 1, 1, 1.8, 2.0, 1, 1, 1]` |
| **Total epochs** | **26 (fixed)** |
| **Session split** | **Session 1: epochs 1–13. Session 2: epochs 14–26** (fixed boundary, not time-based) |
| LR schedule | Epochs 1–3 warmup (bias LR 0.1 → lr0, others 0 → lr0; momentum 0.8 → 0.937; computed from the global iteration, so entirely in session 1). Then linear decay lr = 0.01 × [(1 − e/26) × 0.99 + 0.01] for epoch index e = 0…25: 0.0100 at e = 0, 0.00505 at e = 13 (first epoch of session 2), ≈ 4.8e-4 in the last epoch. Session 2 continues the same 26-epoch schedule |
| Mosaic | On for epochs 1–21, off for epochs 22–26 (session 2) |
| **Estimated quota** | Session 1 ≈ 7.2 h (typical) to 8.9 h (conservative), run this week (13.50 h available). Session 2 ≈ 7.3–8.9 h after the reset on 2026-10-17. Total ≈ 14.5–17.8 h |
| Hard stops | Kaggle `--timeout 41400` (11.5 h) per session; in-training time guard at 11 h saves `resume_last.pt` and stops early if the next epoch would not fit (any shortfall goes to review; E is never changed silently) |
| Abort and report | Operational failures only: unrecovered NaN/Inf loss, DDP/NCCL error, process crash or timeout. No metric-based automatic abort |
| **Checkpoint selection** | **Revised 2026-10-11 (before session 1):** VRU macro mAP50-95 on the training-only holdout = unweighted mean of AP50-95 for person, rider, bike, motor (classes resolved by name), rounded to 5 decimals by ultralytics; `best.pt` = highest score across both sessions, later epoch on ties. All-class mAP50-95, VRU mAP50 and per-class VRU AP are reported only. Official val evaluated once on the selected checkpoint, for the record. Full protocol: [phase6_selection_protocol.md](phase6_selection_protocol.md) |
| Review gate | Session 2 only after you review session-1 outputs (holdout curve, GPU stats, checkpoint hashes) and approve |

## 5. Does the checkpoint/resume path preserve the schedule across sessions?

| Property | Status |
|---|---|
| Resume starts at the next epoch; EMA, optimizer, AMP scaler, best fitness restored | **Verified at runtime** (smoke test) |
| First post-resume LR equals the scheduled value | **Verified at runtime** (smoke test: 0.00505 = expected) |
| Total epochs (26) and lr schedule carried in the checkpoint's `train_args` | Verified in code (resume reloads `train_args`) |
| Warmup computed from global iteration, so it never repeats after resume | Verified in code |
| Mosaic closes on entry to epoch 22 even after a resume at epoch 14 | Verified in code (ultralytics re-evaluates `epoch ≥ epochs − close_mosaic` each epoch); at runtime the smoke resume closed mosaic correctly inside the final window |
| Augmentation RNG | **Not preserved** for rank 1 and dataloader workers; ultralytics reseeds on resume. Augmentation order differs from an uninterrupted run (documented, not a schedule change) |

**Two code changes required before session 1** (training configuration unchanged):

1. **Fixed session boundary:** add `SESSION_STOP_EPOCH = 13` to full mode (same stop-and-save mechanism verified as smoke phase A).
2. **Carry `best.pt` into session 2:** ultralytics restores `best_fitness` on resume but session 2 starts with an empty weights folder. If session 2 never beats session 1's best, it would write no `best.pt`. Session 1's `best.pt` must be published with `resume_last.pt` and restored before session 2, with both hashes recorded.

Both would be CPU-dry-run tested before asking for session-1 approval.
