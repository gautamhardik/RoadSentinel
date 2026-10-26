# RoadSentinel Phase 6 — Stage 1 Dataset Split Audit

**Date:** 2026-10-11
**Status:** COMPLETE — split defensible; Phase 6 holdout fixed
**Script:** [scripts/audit_phase6_splits.py](../../scripts/audit_phase6_splits.py) (CPU only)
**Artifacts:** [experiments/phase6_audit/](../../experiments/phase6_audit/) — `split_audit.json`, `holdout_manifest.txt`, near-duplicate pair lists
**Test split:** quarantined — never listed, opened or hashed by this audit

---

## 1. Scope

Dataset: Kaggle `a7madmostafa/bdd100k-yolo` ("BDD100K converted to YOLO format"; the dataset page documents no split changes). Local copy `bdd100k_dataset/`, train and val only.

## 2. Results

| Check | Train | Val | Verdict |
|---|---:|---:|---|
| Images / label files | 70,000 / 70,000 | 10,000 / 10,000 | PASS |
| Images without label file / with no boxes | 0 / 0 | 0 / 0 | PASS |
| Resolution | 1280×720 (all) | 1280×720 (all) | PASS |
| Files with zero-area boxes | 387 (395 boxes, 33 VRU) | 51 (52 boxes, 3 VRU) | Negligible: ≤3 unmatchable VRU GT in val |
| Byte-identical groups within split | 1 | 2 | Negligible |
| VRU instances / small VRU (<1024 px²) | 106,186 / 48,514 | 15,373 / 7,029 | Matches prior reports |

| Cross-split check (train vs val) | Result |
|---|---|
| Filename overlap | 0 |
| Byte-identical files | 0 |
| Near-duplicates (dHash64 ≤ 6, then 32×32 thumbnail corr ≥ 0.97) | 0 pairs |
| Shared 8-hex name prefixes (recording session) | 1 prefix, 5 val images |
| Name structure | Train names span `0000…`–`a…` (+713 `b…`, 112 `f…`); val is a contiguous block `b1c66a42…`–`c…` (+17 `f…`) |

**Finding:** train and val are disjoint by name, by bytes and by visual near-duplication. Val begins at `b1c66a42-6f7d68ca`, matching the structure of the official BDD100K split (contiguous val block). The split is consistent with the official one and shows no cross-split contamination.

**Not resolved:** whether the third-party checkpoint `weights/bdd100k_yolo11s.pt` was trained on val. This audit validates the dataset, not that checkpoint's training history.

**Documentation correction:** actual val per-class VRU counts are person 13,265 / rider 649 / bike 1,007 / motor 452 (total 15,373). The per-class figures in `docs/phase5/phase5a_experiment_design.md` (11,737 / 591 / 885 / 420) do not match the label files; the total does.

## 3. Phase 6 training holdout

A first holdout drawn as 5,000 independently sampled images leaked: 8 holdout images had 14 near-duplicates in the train split, **all** sharing the same 8-hex name prefix (same recording session). The rule was changed to sample whole prefix groups.

| Property | Value |
|---|---|
| Rule | Sorted unique 8-hex prefixes, `random.Random(42).shuffle`, take whole groups until ≥ 5,000 images |
| Holdout / train split | 5,000 / 65,000 |
| Disjoint; union = train | Yes; yes |
| Shared prefixes holdout ↔ train split | 0 |
| Near-duplicates holdout ↔ train split | 0 |
| Fingerprint (sha256 of sorted names) | `c7047ee0073803431321735c098aad212693daa696153f31f8a26e11ebf7686b` |

`kaggle_jobs/model_training/train_phase6_vru.py` uses the same rule and refuses to train unless it reproduces this fingerprint.

**Limitation:** the holdout is clean only for models trained on the 65,000-image split. The original YOLO11s and the Phase 5A models trained on all 70,000 train images, so cross-family comparisons must use val.
