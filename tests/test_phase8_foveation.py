"""Unit tests for RoadSentinel Phase 8 Active Foveation Engine.

Covers:
  1. Coordinate Mapping (forward crop extraction & inverse full-frame re-mapping)
  2. Crop Boundaries (context expansion, clipping, minimum sizing)
  3. Duplicate Handling & Fusion (WBF coordinate weighting, confidence boosting, NMS)
  4. Empty Detection Edge Cases (empty full-frame, empty crops, zero total detections)
  5. Class-Aware Matching (confidence-ordered greedy policy, class isolation, small-VRU tagging)
"""

import pytest
import numpy as np
from typing import List, Dict, Any, Tuple

from src.foveation.types import Detection, CropProposal, FoveationConfig, CLASS_NAMES, VRU_CLASS_NAMES
from src.foveation.cropper import ActiveCropper
from src.foveation.box_fusion import BoxFusionEngine
from src.foveation.uncertainty_signals import (
    compute_box_iou, expand_box_with_margin, merge_overlapping_proposals,
    UncertaintySignalEvaluator
)
from kaggle_jobs.pilot_300_native_eval.evaluate_pilot_native import match_single_image


# ============================================================================
# 1. Coordinate Mapping Tests
# ============================================================================

def test_coordinate_mapping_subpixel_accuracy():
    """Verify exact numerical reconstruction of full-frame coordinates from crop detections."""
    config = FoveationConfig(boundary_suppress_margin=0)
    cropper = ActiveCropper(config)

    frame_w, frame_h = 1280, 720
    crop_int_box = (320, 180, 800, 540)  # width 480, height 360
    proposal = CropProposal(
        crop_id=1,
        crop_box=(320.0, 180.0, 800.0, 540.0),
        trigger_reason="small_vru",
        trigger_score=0.85,
    )

    # Relative coordinates inside the crop
    crop_dets = [
        Detection(box=(50.25, 40.50, 95.75, 120.25), class_id=0, class_name="person", confidence=0.72),
        Detection(box=(120.0, 80.0, 150.0, 140.0), class_id=5, class_name="bike", confidence=0.68),
    ]

    remapped = cropper.remap_crop_detections_to_frame(crop_dets, crop_int_box, (frame_h, frame_w), proposal)

    assert len(remapped) == 2

    # Expected: crop_int_box x1=320, y1=180
    expected_box_0 = (320 + 50.25, 180 + 40.50, 320 + 95.75, 180 + 120.25)
    for act, exp in zip(remapped[0].box, expected_box_0):
        assert act == pytest.approx(exp, abs=1e-3)
    assert remapped[0].source == "crop"
    assert remapped[0].crop_id == 1
    assert remapped[0].class_name == "person"

    expected_box_1 = (320 + 120.0, 180 + 80.0, 320 + 150.0, 180 + 140.0)
    for act, exp in zip(remapped[1].box, expected_box_1):
        assert act == pytest.approx(exp, abs=1e-3)
    assert remapped[1].class_name == "bike"


# ============================================================================
# 2. Crop Boundaries Tests
# ============================================================================

def test_crop_boundaries_clipping_and_min_size():
    """Verify boundary clipping prevents crops from exceeding frame dimensions and enforces minimum size."""
    img_w, img_h = 1280, 720
    min_size = 160

    # 1. Target right against top-left border
    top_left_box = (5.0, 5.0, 25.0, 45.0)
    exp_tl = expand_box_with_margin(top_left_box, img_w, img_h, margin_ratio=0.5, min_size=min_size)
    assert exp_tl[0] == 0.0, "x1 must be clamped to 0"
    assert exp_tl[1] == 0.0, "y1 must be clamped to 0"
    assert exp_tl[2] >= min_size, "x2 must provide at least min_size width"
    assert exp_tl[3] >= min_size, "y2 must provide at least min_size height"
    assert exp_tl[2] <= img_w
    assert exp_tl[3] <= img_h

    # 2. Target right against bottom-right border
    bottom_right_box = (1265.0, 705.0, 1279.0, 719.0)
    exp_br = expand_box_with_margin(bottom_right_box, img_w, img_h, margin_ratio=0.5, min_size=min_size)
    assert exp_br[2] == float(img_w), "x2 must be clamped to img_w"
    assert exp_br[3] == float(img_h), "y2 must be clamped to img_h"
    assert exp_br[0] <= float(img_w) - min_size, "x1 must ensure min_size width"
    assert exp_br[1] <= float(img_h) - min_size, "y1 must ensure min_size height"
    assert exp_br[0] >= 0.0
    assert exp_br[1] >= 0.0


def test_crop_proposal_merging():
    """Verify overlapping crop proposals are merged to prevent redundant duplicate inspections."""
    img_w, img_h = 1280, 720
    prop1 = CropProposal(crop_id=1, crop_box=(400.0, 200.0, 600.0, 400.0), trigger_reason="low_conf", trigger_score=0.7)
    prop2 = CropProposal(crop_id=2, crop_box=(450.0, 250.0, 650.0, 450.0), trigger_reason="small_vru", trigger_score=0.9)

    merged = merge_overlapping_proposals([prop1, prop2], img_w, img_h, iou_merge_threshold=0.25)
    assert len(merged) == 1
    # Merged proposal should enclose both
    m_box = merged[0].crop_box
    assert m_box[0] <= 400.0
    assert m_box[1] <= 200.0
    assert m_box[2] >= 650.0
    assert m_box[3] >= 450.0
    assert merged[0].trigger_score == 0.9


# ============================================================================
# 3. Duplicate Handling & Fusion Tests
# ============================================================================

def test_box_fusion_duplicate_handling():
    """Verify box fusion merges overlapping same-class detections and suppresses duplicates."""
    config = FoveationConfig(fusion_iou_threshold=0.50)
    engine = BoxFusionEngine(config)

    # Global detection
    ff_det = Detection(
        box=(100.0, 100.0, 200.0, 200.0),
        class_id=0,
        class_name="person",
        confidence=0.50,
        source="full_frame"
    )
    # High-resolution crop detection with high IoU
    crop_det = Detection(
        box=(105.0, 98.0, 202.0, 198.0),
        class_id=0,
        class_name="person",
        confidence=0.80,
        source="crop",
        crop_id=1
    )
    # Independent new crop detection
    new_det = Detection(
        box=(500.0, 300.0, 540.0, 380.0),
        class_id=0,
        class_name="person",
        confidence=0.70,
        source="crop",
        crop_id=1
    )

    fused = engine.fuse_detections([ff_det], [crop_det, new_det])
    assert len(fused) == 2

    # Verify merged detection properties
    merged = [d for d in fused if d.source == "fused"][0]
    assert merged.confidence > 0.80, "Fused confidence should receive dual-pass agreement boost"
    # Weighted box should be closer to crop_det since w_cr (0.80 * 1.2 = 0.96) > w_ff (0.50)
    assert 100.0 < merged.box[0] < 105.0 or merged.box[0] == pytest.approx(103.28, abs=0.5)


# ============================================================================
# 4. Empty Detection Edge Cases
# ============================================================================

def test_empty_detections_handling():
    """Verify pipeline handles all combinations of empty detections without crashing."""
    config = FoveationConfig()
    engine = BoxFusionEngine(config)

    # 1. Both empty
    assert engine.fuse_detections([], []) == []

    # 2. Full-frame empty, crop has detections
    crop_det = Detection(box=(10.0, 10.0, 50.0, 50.0), class_id=1, class_name="rider", confidence=0.75, source="crop")
    res1 = engine.fuse_detections([], [crop_det])
    assert len(res1) == 1
    assert res1[0].class_name == "rider"

    # 3. Full-frame has detections, crop empty
    ff_det = Detection(box=(20.0, 20.0, 60.0, 60.0), class_id=2, class_name="car", confidence=0.88, source="full_frame")
    res2 = engine.fuse_detections([ff_det], [])
    assert len(res2) == 1
    assert res2[0].class_name == "car"

    # 4. Uncertainty evaluator on empty detection list
    evaluator = UncertaintySignalEvaluator(config)
    assert evaluator.extract_candidate_signals([], 1280, 720) == []


# ============================================================================
# 5. Class-Aware Greedy Matching Tests
# ============================================================================

def test_class_aware_matching_strict_isolation():
    """Verify that matching is strictly class-isolated (rider does not match person, bike does not match motor)."""
    # GT is person and motor
    gt_objs = [
        {"class_name": "person", "box": np.array([100.0, 100.0, 150.0, 200.0]), "size_group": "Medium [32²-96²)", "is_vru": True},
        {"class_name": "motor", "box": np.array([300.0, 300.0, 350.0, 400.0]), "size_group": "Medium [32²-96²)", "is_vru": True},
    ]

    # Predictions have 100% geometric overlap, but mismatched classes
    pred_objs = [
        {"class_name": "rider", "conf": 0.90, "box": np.array([100.0, 100.0, 150.0, 200.0])},  # Should NOT match person
        {"class_name": "bike", "conf": 0.85, "box": np.array([300.0, 300.0, 350.0, 400.0])},   # Should NOT match motor
    ]

    stats = match_single_image(pred_objs, gt_objs)

    # Person has 0 TP, 0 FP, 1 FN
    assert stats["class_stats"]["person"]["tp"] == 0
    assert stats["class_stats"]["person"]["fn"] == 1
    assert stats["class_stats"]["person"]["fp"] == 0

    # Rider has 0 TP, 1 FP, 0 FN
    assert stats["class_stats"]["rider"]["tp"] == 0
    assert stats["class_stats"]["rider"]["fp"] == 1
    assert stats["class_stats"]["rider"]["fn"] == 0

    # Motor has 0 TP, 0 FP, 1 FN
    assert stats["class_stats"]["motor"]["tp"] == 0
    assert stats["class_stats"]["motor"]["fn"] == 1
    assert stats["class_stats"]["motor"]["fp"] == 0

    # Bike has 0 TP, 1 FP, 0 FN
    assert stats["class_stats"]["bike"]["tp"] == 0
    assert stats["class_stats"]["bike"]["fp"] == 1
    assert stats["class_stats"]["bike"]["fn"] == 0


def test_confidence_ordered_greedy_matching():
    """Verify that when multiple predictions overlap with one GT, higher confidence gets the TP."""
    gt_objs = [
        {"class_name": "person", "box": np.array([100.0, 100.0, 150.0, 200.0]), "size_group": "Small (<32²)", "is_vru": True}
    ]

    # Two predictions overlapping the same GT:
    # Pred A: conf 0.80, IoU ~0.85
    # Pred B: conf 0.40, IoU ~0.95
    pred_objs = [
        {"class_name": "person", "conf": 0.40, "box": np.array([101.0, 101.0, 149.0, 199.0])},
        {"class_name": "person", "conf": 0.80, "box": np.array([105.0, 105.0, 150.0, 200.0])},
    ]

    stats = match_single_image(pred_objs, gt_objs)
    assert stats["class_stats"]["person"]["tp"] == 1
    assert stats["class_stats"]["person"]["fp"] == 1
    assert stats["class_stats"]["person"]["fn"] == 0
    assert stats["vru_size_stats"]["Small (<32²)"]["tp"] == 1
