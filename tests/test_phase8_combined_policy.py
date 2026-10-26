"""
RoadSentinel Phase 8 — Combined Policy Unit Tests
Tests:
- Intra-crop pedestrian NMS (IoU 0.40) and class isolation
- Unanchored crop-only confidence gate (0.35) and multi-scale anchor retention
- Full-frame detections immutability
- Empty crop / full-frame edge cases
- Cross-crop boxes, overlapping pedestrian boxes, boundary coordinate remapping
"""

import pytest
import copy
from typing import List

from src.foveation.types import Detection, CropProposal, FoveationConfig
from src.foveation.cropper import ActiveCropper
from src.foveation.policy_filters import (
    apply_intra_crop_pedestrian_nms,
    filter_unanchored_crop_detections,
)


def test_intra_crop_pedestrian_nms_suppression():
    """Verify overlapping pedestrian boxes (IoU >= 0.40) within a crop are suppressed."""
    dets = [
        # Two highly overlapping pedestrians (IoU ~ 0.60)
        Detection(box=(100.0, 100.0, 150.0, 200.0), class_id=0, class_name="person", confidence=0.80),
        Detection(box=(105.0, 102.0, 152.0, 198.0), class_id=0, class_name="person", confidence=0.45),
        # Distinct pedestrian (IoU < 0.40)
        Detection(box=(200.0, 100.0, 250.0, 200.0), class_id=0, class_name="person", confidence=0.70),
    ]

    filtered = apply_intra_crop_pedestrian_nms(dets, target_class="person", iou_threshold=0.40)
    assert len(filtered) == 2
    confs = [d.confidence for d in filtered]
    assert 0.80 in confs
    assert 0.70 in confs
    assert 0.45 not in confs


def test_intra_crop_pedestrian_nms_class_isolation():
    """Verify intra-crop NMS only suppresses target class (person) and leaves other classes untouched."""
    dets = [
        # Person and Rider overlapping heavily
        Detection(box=(100.0, 100.0, 150.0, 200.0), class_id=0, class_name="person", confidence=0.80),
        Detection(box=(102.0, 101.0, 149.0, 199.0), class_id=1, class_name="rider", confidence=0.60),
        # Two overlapping cars
        Detection(box=(300.0, 300.0, 450.0, 450.0), class_id=2, class_name="car", confidence=0.85),
        Detection(box=(305.0, 305.0, 448.0, 448.0), class_id=2, class_name="car", confidence=0.55),
    ]

    filtered = apply_intra_crop_pedestrian_nms(dets, target_class="person", iou_threshold=0.40)
    assert len(filtered) == 4, "Non-person detections must not be suppressed by pedestrian intra-crop NMS"


def test_unanchored_crop_confidence_filtering():
    """Verify unanchored crop detections below 0.35 are discarded while anchored ones are kept."""
    full_frame = [
        # Full frame person at (100, 100, 200, 200)
        Detection(box=(100.0, 100.0, 200.0, 200.0), class_id=0, class_name="person", confidence=0.30, source="full_frame"),
    ]

    crop_dets = [
        # 1. Anchored person (IoU ~ 0.85 with full-frame), conf = 0.28 (< 0.35) -> MUST BE KEPT
        Detection(box=(102.0, 101.0, 198.0, 199.0), class_id=0, class_name="person", confidence=0.28, source="crop"),
        # 2. Unanchored person at (500, 500, 600, 600), conf = 0.28 (< 0.35) -> MUST BE DISCARDED
        Detection(box=(500.0, 500.0, 600.0, 600.0), class_id=0, class_name="person", confidence=0.28, source="crop"),
        # 3. Unanchored person at (700, 700, 800, 800), conf = 0.42 (>= 0.35) -> MUST BE KEPT
        Detection(box=(700.0, 700.0, 800.0, 800.0), class_id=0, class_name="person", confidence=0.42, source="crop"),
        # 4. Spatially overlapping box but different class (rider vs person), conf = 0.28 -> UNANCHORED -> DISCARDED
        Detection(box=(101.0, 101.0, 199.0, 199.0), class_id=1, class_name="rider", confidence=0.28, source="crop"),
    ]

    filtered_crop = filter_unanchored_crop_detections(
        full_frame_dets=full_frame,
        crop_dets=crop_dets,
        unanchored_conf_threshold=0.35,
        anchoring_iou_threshold=0.50,
    )

    assert len(filtered_crop) == 2
    # Verify exactly #1 and #3 survived
    kept_confs = sorted([d.confidence for d in filtered_crop])
    assert pytest.approx(kept_confs[0]) == 0.28
    assert pytest.approx(kept_confs[1]) == 0.42
    assert filtered_crop[0].class_name == "person"
    assert filtered_crop[1].class_name == "person"


def test_full_frame_detections_immutability():
    """Verify full-frame detections are never modified or pruned by crop filtering."""
    orig_ff = [
        Detection(box=(10.0, 20.0, 30.0, 40.0), class_id=0, class_name="person", confidence=0.18, source="full_frame"),
        Detection(box=(50.0, 60.0, 70.0, 80.0), class_id=2, class_name="car", confidence=0.95, source="full_frame"),
    ]
    ff_copy = copy.deepcopy(orig_ff)
    crop_dets = [
        Detection(box=(10.0, 20.0, 30.0, 40.0), class_id=0, class_name="person", confidence=0.15, source="crop")
    ]

    _ = filter_unanchored_crop_detections(
        full_frame_dets=orig_ff,
        crop_dets=crop_dets,
        unanchored_conf_threshold=0.35,
        anchoring_iou_threshold=0.50,
    )

    assert len(orig_ff) == len(ff_copy)
    for d_curr, d_exp in zip(orig_ff, ff_copy):
        assert d_curr.box == d_exp.box
        assert d_curr.confidence == d_exp.confidence
        assert d_curr.class_id == d_exp.class_id
        assert d_curr.source == d_exp.source


def test_empty_crops_and_edge_cases():
    """Verify behavior with empty detection lists."""
    assert apply_intra_crop_pedestrian_nms([]) == []
    assert filter_unanchored_crop_detections(full_frame_dets=[], crop_dets=[]) == []

    # Non-empty full frame, empty crop
    ff = [Detection(box=(10.0, 20.0, 30.0, 40.0), class_id=0, class_name="person", confidence=0.5)]
    assert filter_unanchored_crop_detections(full_frame_dets=ff, crop_dets=[]) == []

    # Empty full frame, crop with unanchored detections
    cr = [
        Detection(box=(10.0, 20.0, 30.0, 40.0), class_id=0, class_name="person", confidence=0.25),
        Detection(box=(50.0, 60.0, 70.0, 80.0), class_id=0, class_name="person", confidence=0.40),
    ]
    res = filter_unanchored_crop_detections(full_frame_dets=[], crop_dets=cr, unanchored_conf_threshold=0.35)
    assert len(res) == 1
    assert res[0].confidence == 0.40


def test_crop_boundary_remapping_and_cross_crop_separation():
    """Verify that boundary coordinate remapping preserves spatial coordinates across different crops."""
    config = FoveationConfig(boundary_suppress_margin=0)
    cropper = ActiveCropper(config)

    frame_size = (720, 1280)
    # Crop 1: left half
    c1_box = (0, 100, 400, 500)
    prop1 = CropProposal(crop_id=1, crop_box=(0.0, 100.0, 400.0, 500.0), trigger_reason="small_vru", trigger_score=0.8)
    dets1 = [Detection(box=(50.0, 50.0, 100.0, 120.0), class_id=0, class_name="person", confidence=0.75)]
    remapped1 = cropper.remap_crop_detections_to_frame(dets1, c1_box, frame_size, prop1)

    # Crop 2: right half
    c2_box = (600, 100, 1000, 500)
    prop2 = CropProposal(crop_id=2, crop_box=(600.0, 100.0, 1000.0, 500.0), trigger_reason="small_vru", trigger_score=0.8)
    dets2 = [Detection(box=(50.0, 50.0, 100.0, 120.0), class_id=0, class_name="person", confidence=0.75)]
    remapped2 = cropper.remap_crop_detections_to_frame(dets2, c2_box, frame_size, prop2)

    # In remapped coords, Crop 1 x1 is 50.0; Crop 2 x1 is 650.0
    assert remapped1[0].box[0] == pytest.approx(50.0)
    assert remapped2[0].box[0] == pytest.approx(650.0)
    # Intra-crop NMS on crop 1 does not suppress crop 2
    nms1 = apply_intra_crop_pedestrian_nms(remapped1, target_class="person", iou_threshold=0.40)
    nms2 = apply_intra_crop_pedestrian_nms(remapped2, target_class="person", iou_threshold=0.40)
    assert len(nms1) == 1
    assert len(nms2) == 1
