"""
Unit tests for Autonomous Horizon-Corridor Slicer Module (EXP-7B readiness).
Tests:
1. Geometry calculation and pixel boundary clamping.
2. Coordinate round-trip consistency (local to global remapping).
3. Boundary artifact suppression for truncated edge boxes.
4. Robustness against empty, None, and degraded inputs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np

from src.foveation.horizon_slicer import HorizonSlicer


def test_horizon_slicer_geometry():
    slicer = HorizonSlicer()
    # Test on standard BDD100K 1280x720 resolution
    x1, y1, x2, y2 = slicer.get_crop_box(img_w=1280, img_h=720)
    assert x1 == 160
    assert y1 == 200
    assert x2 == 1120
    assert y2 == 520
    assert (x2 - x1) == 960
    assert (y2 - y1) == 320


def test_coordinate_round_trip():
    slicer = HorizonSlicer()
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    slice_img, slice_coords = slicer.extract_slice(frame)
    cx1, cy1, cx2, cy2 = slice_coords

    # Inject a local detection in the slice
    local_box = (100.0, 50.0, 160.0, 180.0)
    conf = 0.85
    remapped = slicer.remap_detection(local_box, slice_coords, frame.shape[:2], conf=conf)

    assert remapped is not None
    assert remapped[0] == pytest.approx(cx1 + 100.0)
    assert remapped[1] == pytest.approx(cy1 + 50.0)
    assert remapped[2] == pytest.approx(cx1 + 160.0)
    assert remapped[3] == pytest.approx(cy1 + 180.0)
    # Check width and height preservation
    assert (remapped[2] - remapped[0]) == pytest.approx(60.0)
    assert (remapped[3] - remapped[1]) == pytest.approx(130.0)


def test_boundary_box_suppression():
    slicer = HorizonSlicer(boundary_margin_px=8.0, boundary_suppress_conf=0.30)
    slice_coords = (160, 200, 1120, 520)
    frame_shape = (720, 1280)

    # Box 1: Abuts left boundary (rx1 = 2.0 <= 8.0) with low confidence (0.20) -> MUST BE SUPPRESSED
    boundary_box_low = (2.0, 40.0, 50.0, 120.0)
    res_low = slicer.remap_detection(boundary_box_low, slice_coords, frame_shape, conf=0.20)
    assert res_low is None, "Low confidence boundary box must be suppressed"

    # Box 2: Abuts left boundary with high confidence (0.80) -> MUST BE PRESERVED
    res_high = slicer.remap_detection(boundary_box_low, slice_coords, frame_shape, conf=0.80)
    assert res_high is not None, "High confidence boundary box must be preserved"

    # Box 3: Interior box (not touching boundary) with low confidence (0.20) -> MUST BE PRESERVED
    interior_box = (40.0, 40.0, 90.0, 120.0)
    res_int = slicer.remap_detection(interior_box, slice_coords, frame_shape, conf=0.20)
    assert res_int is not None, "Interior low confidence box must be preserved"


def test_empty_and_degraded_inputs():
    slicer = HorizonSlicer()

    # None frame
    sub_none, coords_none = slicer.extract_slice(None)
    assert coords_none == (0, 0, 0, 0)
    assert sub_none.shape[0] > 0

    # Empty frame
    empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
    sub_empty, coords_empty = slicer.extract_slice(empty_frame)
    assert coords_empty == (0, 0, 0, 0)

    # Tiny frame
    tiny_frame = np.full((8, 8, 3), 50, dtype=np.uint8)
    sub_tiny, coords_tiny = slicer.extract_slice(tiny_frame)
    assert coords_tiny == (0, 0, 0, 0)

    # Remap on invalid coords
    res_invalid = slicer.remap_detection((10, 10, 20, 20), (0, 0, 0, 0), (720, 1280))
    assert res_invalid is None
