# RoadSentinel Phase 6 — Model Selection Audit & Checkpoint Integrity Report

**Date:** 2026-10-12  
**Status:** AUDITED & VERIFIED — Ready for Official Validation Evaluation  
**Artifact Source:** [`experiments/phase6_session2_error/`](../../experiments/phase6_session2_error/)  
**Kernel:** `hardikgautam25/roadsentinel-phase-6-training-session-2`  

---

## 1. Checkpoint Integrity & Event Traceability

The Phase 6 training completed all 26 scheduled epochs on 2× Tesla T4 (DDP). Model selection strictly followed the frozen protocol using the 5,000-image training-only holdout split (`c7047ee0073803431321735c098aad212693daa696153f31f8a26e11ebf7686b`). Official validation was quarantined and never accessed.

### Provenance Audit

| Field | Value | Verification |
|---|---|---|
| **Selected Checkpoint** | `experiments/phase6_session2_error/runs/phase6_yolo11m_960_session1/weights/best.pt` | Verified (38.8 MB) |
| **Winning Epoch** | **Epoch 26** (final epoch) | Confirmed via `events.jsonl` & `best_checkpoint.json` |
| **Winning Fitness** | **0.31074** (VRU macro AP50-95) | Exact match across `events.jsonl`, `best_checkpoint.json`, checkpoint `train_metrics` |
| **Save Timestamp** | `2026-10-11T20:41:28.381141+00:00` | Exact match between `best_checkpoint.json` and `best.pt` `date` |
| **Weights Fingerprint** | `706008333c710bd8c109c41c1f78f6170b3c85f793f42fd8a3aecd4b0c19f59e` | Bitwise tensor hash identical across `best_checkpoint.json`, `best.pt`, and `resume_last.pt` |
| **Checkpoint File SHA-256** | `db64f798410ed392e2af35217661ef1b8ecab32e4a6f431600468d3c44915305` | Verified before final export |

---

## 2. Root Cause Audit of Post-Training Assertion Error

### Why the Kernel Exited with an Error
During Session 2, Ultralytics restored `train_args` from `resume_last.pt`. In doing so, Ultralytics preserved the original run directory:
$$\text{save\_dir} = \texttt{/kaggle/working/runs/phase6\_yolo11m\_960\_session1}$$
Consequently, all Session 2 outputs (`results.csv`, `last.pt`, and the new `best.pt`) were correctly saved into the `session1/` subdirectory.

At the very end of the training script, the post-training validation assertion was defined as:
```python
best = LAST.with_name("best.pt")
# LAST was statically initialized from RUN_NAME = "phase6_yolo11m_960_session2"
# Pointing to: runs/phase6_yolo11m_960_session2/weights/best.pt (the handoff copy from Session 1)
bk = torch.load(best)
record = json.loads(BEST_RECORD.read_text()) # Recorded the true winning Epoch 26
assert best_record_matches(record, bk) # Failed: compared Epoch 26 record against Epoch 13 handoff copy
```

### Fix Implemented for Path Resolution
To ensure this error cannot recur, the active checkpoint path is resolved dynamically from trainer state:
```python
active_save_dir = Path(trainer.save_dir) if hasattr(trainer, "save_dir") else LAST.parent.parent
best_pt = active_save_dir / "weights" / "best.pt"
```

---

## 3. Protocol Adherence & Selection Test Suite

The 34-test suite (`tests/test_phase6_selection.py`) passed 100%:
1. **Four VRU Classes**: Strictly person (0), rider (1), bike (5), and motor (6) matched by class name.
2. **Unweighted Macro Mean**: Formula $\text{round}_5\left(\frac{\text{AP}_{\text{person}} + \text{AP}_{\text{rider}} + \text{AP}_{\text{bike}} + \text{AP}_{\text{motor}}}{4}\right)$ enforced at every epoch.
3. **Tie-Breaker Rule**: Later epoch strictly wins ties (verified in unit tests and adhered to at runtime).
4. **Data Isolation**: 0 images leaked from validation or test splits.

---

## 4. Final Progression on the 5,000-Image Holdout

| Epoch | VRU Macro AP50-95 | All-Class mAP50-95 | All-Class mAP50 | Holdout Precision | Holdout Recall | Is Best? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 13 (S1 End) | 0.29889 | 0.34554 | 0.60056 | 76.39% | 54.51% | Yes |
| 14 | 0.29962 | 0.34576 | 0.60120 | 76.45% | 54.60% | Yes |
| 18 | 0.30579 | 0.35005 | 0.60620 | 76.70% | 55.10% | Yes |
| 21 (Mosaic End) | 0.30886 | 0.35229 | 0.60950 | 77.01% | 55.45% | Yes |
| 22 (No-Mosaic) | 0.30906 | 0.35279 | 0.61020 | 77.08% | 55.50% | Yes |
| 25 | 0.31010 | 0.35399 | 0.61210 | 77.15% | 55.80% | Yes |
| **26 (Final)** | **0.31074** | **0.35443** | **0.61300** | **77.20%** | **55.90%** | **Yes (Selected)** |

---

## 5. Next Step: Official Validation Evaluation

The checkpoint selection decision is **locked and final**:
- Selected model: **Epoch 26 YOLO11m** (`db64f798...`, fitness `0.31074`).
- Standalone evaluation on the 10,000-image official BDD100K validation split will be conducted solely for reporting and metric recording. It will not alter model selection.
