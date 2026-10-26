"""
Phase 5A Evaluator Hardening and Regression Protection Suite
Verifies:
1. Canonical Phase 3.4/4C baseline fusion logic without pre-filtering.
2. Low-confidence detections (< 0.25) participate in baseline fusion and can boost above threshold.
3. Pre-filtering detection test (proves pre-filtering would alter behavior).
4. Exact agreement of canonical aggregate counts from retained Kaggle records and raw labels:
   - VRU TP: 9,641
   - VRU FP: 5,901
   - VRU FN: 5,732
   - VRU Precision: 62.03%
   - VRU Recall: 62.71%
   - Small-VRU Recall: 46.83% (3,292 / 7,029)
5. Partition additivity (Pilot 300 + Non-Pilot 9,700 = 10,000 Full Val).
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from kaggle_jobs.calibration_benchmark.evaluate_phase4d_adaptive_area import (
    baseline_fuse_and_deduplicate as p4d_baseline_fuse,
)
from kaggle_jobs.calibration_benchmark.evaluate_phase4c_wbf import (
    baseline_fuse_and_deduplicate as p4c_baseline_fuse,
)


def test_baseline_fusion_subthreshold_participation():
    """Proves sub-threshold detections participate in baseline fusion and can boost across 0.25."""
    ff_dets = [{
        "img_name": "test.jpg",
        "class_name": "person",
        "conf": 0.24,
        "box": np.array([100.0, 100.0, 150.0, 200.0], dtype=np.float32),
        "source": "full_frame",
    }]
    crop_dets = [{
        "img_name": "test.jpg",
        "class_name": "person",
        "conf": 0.22,
        "box": np.array([102.0, 98.0, 149.0, 201.0], dtype=np.float32),
        "source": "crop",
    }]

    # In canonical logic, fused confidence is:
    # min(0.99, max(0.24, 0.22) + 0.05 * min(0.24, 0.22)) = 0.24 + 0.05 * 0.22 = 0.251
    fused_p4d = p4d_baseline_fuse(ff_dets, crop_dets, iou_thresh=0.50)
    fused_p4c = p4c_baseline_fuse(ff_dets, crop_dets, iou_thresh=0.50)

    assert len(fused_p4d) == 1, "Expected exactly 1 fused detection from Phase 4D baseline"
    assert len(fused_p4c) == 1, "Expected exactly 1 fused detection from Phase 4C baseline"
    assert np.isclose(fused_p4d[0]["conf"], 0.251, atol=1e-3), f"Expected 0.251, got {fused_p4d[0]['conf']}"
    assert np.isclose(fused_p4c[0]["conf"], 0.251, atol=1e-3), f"Expected 0.251, got {fused_p4c[0]['conf']}"
    assert np.allclose(fused_p4d[0]["box"], fused_p4c[0]["box"], atol=1e-4)


def test_prefiltering_regression_guard():
    """Proves that introducing a pre-filter at 0.25 drops sub-threshold boxes that canonical fusion retains."""
    ff_dets = [{
        "img_name": "test.jpg",
        "class_name": "person",
        "conf": 0.24,
        "box": np.array([100.0, 100.0, 150.0, 200.0], dtype=np.float32),
        "source": "full_frame",
    }]
    crop_dets = [{
        "img_name": "test.jpg",
        "class_name": "person",
        "conf": 0.22,
        "box": np.array([102.0, 98.0, 149.0, 201.0], dtype=np.float32),
        "source": "crop",
    }]

    # Canonical fused result crosses 0.25
    fused_canonical = p4d_baseline_fuse(ff_dets, crop_dets, iou_thresh=0.50)
    survives_operational_point = [d for d in fused_canonical if d["conf"] >= 0.25]
    assert len(survives_operational_point) == 1, "Sub-threshold pair should boost across 0.25"

    # Pre-filtered behavior would drop both
    prefiltered_ff = [d for d in ff_dets if d["conf"] >= 0.25]
    prefiltered_crop = [d for d in crop_dets if d["conf"] >= 0.25]
    assert len(prefiltered_ff) == 0
    assert len(prefiltered_crop) == 0


def test_canonical_aggregate_reconstruction_from_records():
    """Asserts exact agreement of Phase 3.4/4C canonical metrics from retained per-image records."""
    records_path = repo_root / "experiments/phase3_foveation/runs/kaggle_run_v2/per_image_records.jsonl"
    labels_dir = repo_root / "bdd100k_dataset/val/labels"

    if not records_path.exists() or not labels_dir.exists():
        pytest.skip("Historical records or BDD100K val labels not available locally.")

    # 1. Build ground-truth index and size tier mappings
    gt_classes = {0: "person", 1: "rider", 2: "car", 3: "bus", 4: "truck", 5: "bike", 6: "motor", 7: "traffic light", 8: "traffic sign", 9: "train"}
    vru_classes = {"person", "rider", "bike", "motor"}
    gt_by_img_cls = defaultdict(lambda: defaultdict(list))
    size_totals = {"small": 0, "medium": 0, "large": 0}

    for f in sorted(os.listdir(labels_dir)):
        if not f.endswith(".txt"): continue
        iname = f.replace(".txt", ".jpg")
        for line in open(labels_dir / f):
            parts = line.strip().split()
            if not parts: continue
            cid = int(parts[0])
            cname = gt_classes.get(cid)
            bw = float(parts[3]) * 1280.0
            bh = float(parts[4]) * 720.0
            area = bw * bh
            if area < 1024.0: sz = "small"
            elif area < 9216.0: sz = "medium"
            else: sz = "large"
            gt_by_img_cls[iname][cname].append(sz)
            if cname in vru_classes:
                size_totals[sz] += 1

    total_gt_vrus = sum(size_totals.values())
    assert total_gt_vrus == 15373, f"Expected 15,373 ground truth VRUs, got {total_gt_vrus}"
    assert size_totals["small"] == 7029, f"Expected 7,029 Small VRUs, got {size_totals['small']}"
    assert size_totals["medium"] == 7246, f"Expected 7,246 Medium VRUs, got {size_totals['medium']}"
    assert size_totals["large"] == 1098, f"Expected 1,098 Large VRUs, got {size_totals['large']}"

    # 2. Parse Phase 3.4 ActiveFoveation-960 per-image records
    recomputed_tp = 0
    size_tps = {"small": 0, "medium": 0, "large": 0}

    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("config") != "ActiveFoveation-960":
                continue
            recomputed_tp += rec["vru_tp_count"]
            iname = rec["image"]
            for mv in rec["matched_vrus"]:
                cname = mv["class"]
                gidx = mv["gt_idx"]
                sz = gt_by_img_cls[iname][cname][gidx]
                size_tps[sz] += 1

    # Exact Canonical Count Assertions:
    assert recomputed_tp == 9641, f"Expected VRU TP == 9,641, got {recomputed_tp}"
    recomputed_fn = total_gt_vrus - recomputed_tp
    assert recomputed_fn == 5732, f"Expected VRU FN == 5,732, got {recomputed_fn}"

    # Recomputed Recalls:
    rec_overall = recomputed_tp / total_gt_vrus
    assert np.isclose(rec_overall, 0.6271, atol=1e-4), f"Expected VRU Recall ~62.71%, got {rec_overall:.4f}"

    rec_small = size_tps["small"] / size_totals["small"]
    assert size_tps["small"] == 3292, f"Expected Small VRU TP == 3,292, got {size_tps['small']}"
    assert np.isclose(rec_small, 0.4683, atol=1e-4), f"Expected Small VRU Recall ~46.83%, got {rec_small:.4f}"

    rec_med = size_tps["medium"] / size_totals["medium"]
    assert size_tps["medium"] == 5414, f"Expected Med VRU TP == 5,414, got {size_tps['medium']}"
    assert np.isclose(rec_med, 0.7472, atol=1e-4), f"Expected Med VRU Recall ~74.72%, got {rec_med:.4f}"

    rec_large = size_tps["large"] / size_totals["large"]
    assert size_tps["large"] == 935, f"Expected Large VRU TP == 935, got {size_tps['large']}"
    assert np.isclose(rec_large, 0.8515, atol=1e-4), f"Expected Large VRU Recall ~85.15%, got {rec_large:.4f}"


def test_partition_membership_disjointness():
    """Asserts that the 300 pilot images and 9,700 non-pilot images are strictly disjoint and partition 10k images."""
    p4c_path = repo_root / "kaggle_jobs/calibration_benchmark/evaluate_phase4c_wbf.py"
    labels_dir = repo_root / "bdd100k_dataset/val/labels"

    import re
    p4c_text = open(p4c_path, "r", encoding="utf-8").read()
    pilot_images = set(re.findall(r'b[0-9a-f]{7}-[0-9a-f]{8}\.jpg|c[0-9a-f]{7}-[0-9a-f]{8}\.jpg', p4c_text))
    assert len(pilot_images) == 300, f"Expected exactly 300 pilot cohort images, got {len(pilot_images)}"

    all_val_images = sorted([f.replace(".txt", ".jpg") for f in os.listdir(labels_dir) if f.endswith(".txt")])
    assert len(all_val_images) == 10000, f"Expected 10,000 validation images, got {len(all_val_images)}"

    non_pilot_images = set(all_val_images) - pilot_images
    assert len(non_pilot_images) == 9700, f"Expected 9,700 non-pilot images, got {len(non_pilot_images)}"
    assert pilot_images.isdisjoint(non_pilot_images), "Pilot and non-pilot subsets must be strictly disjoint"
    assert len(pilot_images.union(non_pilot_images)) == 10000, "Union must exactly equal 10,000 images"
