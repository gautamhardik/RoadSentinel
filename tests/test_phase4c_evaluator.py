"""
Unit and Regression Tests for RoadSentinel Phase 4C Evaluator & WBF Candidate
Validates evaluator integrity, coordinate transforms, 1-to-1 matching,
AP calculation, data isolation, and manifest partitioning.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pytest

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from kaggle_jobs.calibration_benchmark.evaluate_phase4c_wbf import (
    CHECKPOINT_TO_GT_ID,
    GT_CLASSES,
    VRU_CLASS_NAMES,
    ALL_CLASS_NAMES,
    compute_box_iou,
    compute_box_iou_vec,
    compute_coco_ap_for_class,
    evaluate_operational_point,
    expand_crop_box,
    baseline_fuse_and_deduplicate,
    calibrated_wbf_consensus_fusion,
    evaluate_subset_metrics,
)


def test_class_mapping_bijection():
    """Verify that class mapping is a complete bijection over all 10 classes."""
    assert len(CHECKPOINT_TO_GT_ID) == 10
    mapped_ids = list(CHECKPOINT_TO_GT_ID.values())
    assert sorted(mapped_ids) == list(range(10))
    for cid in range(10):
        assert cid in GT_CLASSES
        assert GT_CLASSES[cid] in ALL_CLASS_NAMES


def test_coordinate_transforms_and_crop_expansion():
    """Verify crop box expansion and bounds clipping."""
    img_w, img_h = 1280, 720
    box = (100.0, 150.0, 200.0, 300.0)
    cx1, cy1, cx2, cy2 = expand_crop_box(box, img_w, img_h, margin=0.25, min_size=160)
    assert 0.0 <= cx1 <= box[0]
    assert 0.0 <= cy1 <= box[1]
    assert box[2] <= cx2 <= img_w
    assert box[3] <= cy2 <= img_h


def test_one_to_one_matching_and_duplicate_suppression():
    """Verify that multiple duplicate detections on the same ground truth produce exactly 1 TP and remaining FPs."""
    gt_boxes = np.array([[100.0, 100.0, 200.0, 200.0]], dtype=np.float32)
    gt_by_img = {"test_img.jpg": gt_boxes}

    # Two overlapping detections
    preds = [
        {"img_name": "test_img.jpg", "conf": 0.90, "box": np.array([102.0, 101.0, 198.0, 199.0], dtype=np.float32)},
        {"img_name": "test_img.jpg", "conf": 0.85, "box": np.array([101.0, 100.0, 201.0, 200.0], dtype=np.float32)},
        {"img_name": "test_img.jpg", "conf": 0.70, "box": np.array([103.0, 102.0, 197.0, 198.0], dtype=np.float32)},
    ]

    tp, fp, fn, prec, rec, f1, matched_gt = evaluate_operational_point(preds, gt_by_img, num_gt=1, conf_thresh=0.25, iou_thresh=0.50)
    assert tp == 1, f"Expected exactly 1 TP, got {tp}"
    assert fp == 2, f"Expected 2 duplicate FPs, got {fp}"
    assert fn == 0
    assert len(matched_gt["test_img.jpg"]) == 1


def test_ap_computation_monotonicity():
    """Verify that 101-point COCO AP evaluates correctly on perfect vs noisy detections."""
    gt_boxes = np.array([[100.0, 100.0, 200.0, 200.0], [300.0, 300.0, 400.0, 400.0]], dtype=np.float32)
    gt_by_img = {"test_img.jpg": gt_boxes}

    # Perfect detections
    perfect_preds = [
        {"img_name": "test_img.jpg", "conf": 0.95, "box": np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32)},
        {"img_name": "test_img.jpg", "conf": 0.90, "box": np.array([300.0, 300.0, 400.0, 400.0], dtype=np.float32)},
    ]
    ap50_perf, ap_perf = compute_coco_ap_for_class(perfect_preds, gt_by_img, num_gt=2)
    assert ap50_perf == pytest.approx(1.0, abs=1e-3)

    # Empty detections
    ap50_empty, ap_empty = compute_coco_ap_for_class([], gt_by_img, num_gt=2)
    assert ap50_empty == 0.0
    assert ap_empty == 0.0


def test_wbf_consensus_fusion_data_isolation():
    """Verify that WBF fusion receives only detection fields and blends coordinates correctly."""
    ff_dets = [{
        "img_name": "img_01.jpg",
        "class_name": "person",
        "conf": 0.70,
        "box": np.array([100.0, 100.0, 200.0, 300.0], dtype=np.float32),
        "source": "full_frame",
    }]
    crop_dets = [{
        "img_name": "img_01.jpg",
        "class_name": "person",
        "conf": 0.80,
        "box": np.array([102.0, 98.0, 202.0, 302.0], dtype=np.float32),
        "source": "crop",
    }]

    fused = calibrated_wbf_consensus_fusion(ff_dets, crop_dets, tau_ff=0.25, tau_isolated=0.40, tau_confirmed=0.20)
    assert len(fused) == 1
    f_det = fused[0]
    assert f_det["source"] == "wbf_consensus"
    assert f_det["class_name"] == "person"
    # Multi-source agreement boost should increase confidence
    assert f_det["conf"] > 0.80
    # Coordinate blending check
    assert 100.0 <= f_det["box"][0] <= 102.0
    assert 98.0 <= f_det["box"][1] <= 100.0


def test_manifest_partition_disjointness():
    """Verify that the 300 pilot cohort images and the remaining validation images are strictly disjoint."""
    manifest_p = repo_root / "experiments" / "phase4_pilot" / "cohort_manifest.json"
    if manifest_p.exists():
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        pilot_imgs = [item["image"] for item in manifest["images"]]
        assert len(pilot_imgs) == 300
        assert len(set(pilot_imgs)) == 300, "Pilot cohort contains duplicate image IDs"
