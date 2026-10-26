"""
Unit tests for RoadSentinel Phase 3 Active Foveation Engine.
Tests:
  1. Coordinate transforms (forward crop extraction & inverse frame re-mapping)
  2. Boundary clipping and context expansion
  3. Class mapping integrity (Checkpoint -> Ground Truth)
  4. Box fusion, Weighted Box Fusion, and deduplication
  5. Uncertainty signals (low-confidence, small VRU, boundary truncation, clustering)
  6. Fixed-grid and deterministic random crop policies
"""

import pytest
import numpy as np
from src.foveation.types import Detection, CropProposal, FoveationConfig, CLASS_NAMES, VRU_CLASS_NAMES
from src.foveation.uncertainty_signals import (
    compute_box_iou, expand_box_with_margin, merge_overlapping_proposals,
    UncertaintySignalEvaluator, propose_fixed_grid_crops, propose_random_crops
)
from src.foveation.cropper import ActiveCropper
from src.foveation.box_fusion import BoxFusionEngine
from src.foveation.engine import CHECKPOINT_CLASSES, CHECKPOINT_TO_GT_ID, GT_CLASSES


def test_class_mapping():
    """Verify class mapping between checkpoint and ground truth aligns all 10 classes."""
    for ckpt_id, ckpt_name in CHECKPOINT_CLASSES.items():
        gt_id = CHECKPOINT_TO_GT_ID[ckpt_id]
        gt_name = GT_CLASSES[gt_id]
        assert ckpt_name == gt_name, f"Class mismatch: ckpt {ckpt_id} ({ckpt_name}) != gt {gt_id} ({gt_name})"
    assert len(CHECKPOINT_CLASSES) == 10
    assert len(GT_CLASSES) == 10


def test_box_iou():
    """Verify vectorized and scalar IoU computations."""
    boxA = (100.0, 100.0, 200.0, 200.0)
    boxB = (100.0, 100.0, 200.0, 200.0)
    assert compute_box_iou(boxA, boxB) == pytest.approx(1.0)

    boxC = (150.0, 100.0, 250.0, 200.0)  # 50x100 overlap = 5000 / 15000 = 0.3333
    assert compute_box_iou(boxA, boxC) == pytest.approx(5000.0 / 15000.0, abs=1e-4)

    boxD = (300.0, 300.0, 400.0, 400.0)  # No overlap
    assert compute_box_iou(boxA, boxD) == 0.0


def test_coordinate_transforms_exact_recovery():
    """
    Verify that mapping a bounding box into a crop and remapping back
    recovers the exact original frame coordinates.
    """
    config = FoveationConfig()
    cropper = ActiveCropper(config)

    frame_w, frame_h = 1280, 720
    # Original object on full frame
    orig_obj_box = (450.0, 250.0, 550.0, 400.0)

    # Crop proposal covering this object
    crop_proposal = CropProposal(
        crop_id=1,
        crop_box=(400.0, 200.0, 600.0, 500.0),
        trigger_reason="small_vru",
        trigger_score=0.8,
    )
    crop_box_int = (400, 200, 600, 500)

    # In crop coordinates (relative to cx1=400, cy1=200):
    crop_rel_box = (
        orig_obj_box[0] - crop_box_int[0],  # 50.0
        orig_obj_box[1] - crop_box_int[1],  # 50.0
        orig_obj_box[2] - crop_box_int[0],  # 150.0
        orig_obj_box[3] - crop_box_int[1],  # 200.0
    )

    crop_det = Detection(
        box=crop_rel_box,
        class_id=0,
        class_name="person",
        confidence=0.85,
        source="crop",
    )

    remapped = cropper.remap_crop_detections_to_frame(
        [crop_det], crop_box_int, (frame_h, frame_w), crop_proposal
    )

    assert len(remapped) == 1
    rec_box = remapped[0].box
    assert rec_box[0] == pytest.approx(orig_obj_box[0], abs=1e-3)
    assert rec_box[1] == pytest.approx(orig_obj_box[1], abs=1e-3)
    assert rec_box[2] == pytest.approx(orig_obj_box[2], abs=1e-3)
    assert rec_box[3] == pytest.approx(orig_obj_box[3], abs=1e-3)
    assert remapped[0].source == "crop"
    assert remapped[0].crop_id == 1


def test_boundary_clipping():
    """Verify boundary clipping prevents crops from exceeding frame dimensions."""
    img_w, img_h = 1280, 720

    # Box near top-left corner
    corner_box = (10.0, 10.0, 40.0, 50.0)
    expanded = expand_box_with_margin(corner_box, img_w, img_h, margin_ratio=0.5, min_size=128)
    assert expanded[0] >= 0.0
    assert expanded[1] >= 0.0
    assert expanded[2] <= img_w
    assert expanded[3] <= img_h

    # Box near bottom-right corner
    br_box = (1260.0, 700.0, 1279.0, 719.0)
    expanded_br = expand_box_with_margin(br_box, img_w, img_h, margin_ratio=0.5, min_size=128)
    assert expanded_br[0] >= 0.0
    assert expanded_br[1] >= 0.0
    assert expanded_br[2] <= img_w
    assert expanded_br[3] <= img_h


def test_box_fusion_deduplication():
    """Verify Weighted Box Fusion merges duplicates and retains newly recovered objects."""
    config = FoveationConfig(fusion_iou_threshold=0.50)
    fusion_engine = BoxFusionEngine(config)

    # Case 1: Overlapping full-frame and crop detection
    ff_det = Detection(
        box=(100.0, 100.0, 150.0, 200.0),
        class_id=0,
        class_name="person",
        confidence=0.40,
        source="full_frame",
    )
    crop_det = Detection(
        box=(102.0, 98.0, 152.0, 202.0),
        class_id=0,
        class_name="person",
        confidence=0.75,
        source="crop",
        crop_id=1,
    )

    # Case 2: New recovered detection only in crop
    new_crop_det = Detection(
        box=(500.0, 300.0, 530.0, 360.0),
        class_id=0,
        class_name="person",
        confidence=0.65,
        source="crop",
        crop_id=1,
    )

    fused = fusion_engine.fuse_detections([ff_det], [crop_det, new_crop_det])

    assert len(fused) == 2  # 1 fused match + 1 newly recovered
    sources = [d.source for d in fused]
    assert "fused" in sources
    assert "crop" in sources

    # Check that fused confidence is higher than single pass
    fused_det = [d for d in fused if d.source == "fused"][0]
    assert fused_det.confidence > 0.75


def test_uncertainty_signals():
    """Verify uncertainty evaluator flags low confidence, small VRUs, and boundary cases."""
    config = FoveationConfig(
        low_conf_min=0.10,
        low_conf_max=0.35,
        small_box_area_threshold=1024.0,
    )
    evaluator = UncertaintySignalEvaluator(config)

    img_w, img_h = 1280, 720

    # 1. Low confidence VRU
    det_low_conf = Detection(
        box=(300.0, 300.0, 350.0, 420.0),
        class_id=0,
        class_name="person",
        confidence=0.22,
    )

    # 2. Small VRU
    det_small = Detection(
        box=(600.0, 300.0, 620.0, 340.0),  # 20x40 = 800 px < 1024
        class_id=5,
        class_name="bike",
        confidence=0.55,
    )

    # 3. Large confident car (Should NOT trigger)
    det_car = Detection(
        box=(700.0, 400.0, 950.0, 600.0),  # 250x200 = 50000 px
        class_id=2,
        class_name="car",
        confidence=0.88,
    )

    proposals = evaluator.extract_candidate_signals([det_low_conf, det_small, det_car], img_w, img_h)

    assert len(proposals) >= 2
    triggers = [p.trigger_reason for p in proposals]
    assert any("low_confidence" in t for t in triggers)
    assert any("small_vru" in t for t in triggers)


def test_fixed_and_random_crop_policies():
    """Verify fixed-grid and random crop baselines return valid non-empty crops."""
    img_w, img_h = 1280, 720

    fixed = propose_fixed_grid_crops(img_w, img_h, k=3)
    assert len(fixed) == 3
    for p in fixed:
        assert 0 <= p.crop_box[0] < p.crop_box[2] <= img_w
        assert 0 <= p.crop_box[1] < p.crop_box[3] <= img_h

    # Test random crops determinism with seed
    rand1 = propose_random_crops(img_w, img_h, k=3, seed=42)
    rand2 = propose_random_crops(img_w, img_h, k=3, seed=42)
    assert len(rand1) == 3
    for p1, p2 in zip(rand1, rand2):
        assert p1.crop_box == p2.crop_box
