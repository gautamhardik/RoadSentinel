"""
Unit tests for RoadSentinel Phase 9 Asynchronous Crop Worker (Async-Fov)
Tests:
- Monotonic frame ID tracking and timestamp propagation
- Bounded queue overload behavior (DROP_OLDEST)
- 100 ms deadline evaluation (MET_DEADLINE vs. EXPIRED)
- Frame association integrity (no cross-frame misattribution)
- Graceful worker start, execution, and shutdown
"""

import time
import pytest
import numpy as np
from unittest.mock import MagicMock

from src.foveation.types import Detection, CropProposal, FoveationConfig
from src.foveation.cropper import ActiveCropper
from src.foveation.async_worker import AsyncCropWorker, AsyncCropRequest, AsyncCropResult


class MockYOLOModel:
    def __init__(self):
        self.names = {0: "person", 1: "rider", 2: "car"}

    def predict(self, source, imgsz, conf, device, verbose=False):
        mock_res = MagicMock()
        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0
        mock_box.conf.item.return_value = 0.45
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu().numpy().tolist.return_value = [10.0, 10.0, 50.0, 100.0]
        mock_res.boxes = [mock_box]
        return [mock_res]


def test_async_crop_request_creation():
    """Verify AsyncCropRequest captures monotonic frame IDs, capture timestamps, and ROIs."""
    prop = CropProposal(crop_id=1, crop_box=(100.0, 100.0, 300.0, 300.0), trigger_reason="small_vru", trigger_score=0.8)
    req = AsyncCropRequest(
        frame_id=42,
        capture_timestamp_ms=1000.0,
        crop_image=np.zeros((200, 200, 3), dtype=np.uint8),
        crop_int_box=(100, 100, 300, 300),
        full_frame_shape=(720, 1280),
        proposal=prop,
        full_frame_detections=[],
        deadline_ms=100.0,
    )
    assert req.frame_id == 42
    assert req.capture_timestamp_ms == 1000.0
    assert req.deadline_ms == 100.0
    assert req.proposal.trigger_reason == "small_vru"


def test_bounded_queue_drop_oldest_overload():
    """Verify bounded queue drops the oldest request when overloaded (DROP_OLDEST policy)."""
    cfg = FoveationConfig()
    cropper = ActiveCropper(cfg)
    model = MockYOLOModel()

    # Worker with max queue size = 2 (not started so queue fills up)
    worker = AsyncCropWorker(model, cropper, max_queue_size=2, deadline_ms=100.0)

    prop = CropProposal(crop_id=1, crop_box=(100.0, 100.0, 300.0, 300.0), trigger_reason="small_vru", trigger_score=0.8)
    dummy_crop = np.zeros((200, 200, 3), dtype=np.uint8)

    # Submit 3 requests into queue of size 2
    worker.submit_crop_request(frame_id=1, capture_timestamp_ms=100.0, crop_image=dummy_crop, crop_int_box=(0, 0, 10, 10), full_frame_shape=(720, 1280), proposal=prop, full_frame_detections=[])
    worker.submit_crop_request(frame_id=2, capture_timestamp_ms=133.0, crop_image=dummy_crop, crop_int_box=(0, 0, 10, 10), full_frame_shape=(720, 1280), proposal=prop, full_frame_detections=[])
    # 3rd submission must push out frame 1
    worker.submit_crop_request(frame_id=3, capture_timestamp_ms=166.0, crop_image=dummy_crop, crop_int_box=(0, 0, 10, 10), full_frame_shape=(720, 1280), proposal=prop, full_frame_detections=[])

    assert worker.total_requests_dropped_overload == 1
    # Check dropped result telemetry
    res_1 = worker.get_result_for_frame(1)
    assert res_1 is not None
    assert res_1.status == "DROPPED_OVERLOAD"
    assert res_1.frame_id == 1

    # Remaining items in queue should be frame 2 and frame 3
    remaining = [worker.queue.get_nowait().frame_id for _ in range(2)]
    assert remaining == [2, 3]


def test_deadline_enforcement_and_expiration():
    """Verify worker evaluates result age against 100 ms deadline and marks EXPIRED if late."""
    cfg = FoveationConfig()
    cropper = ActiveCropper(cfg)
    model = MockYOLOModel()

    worker = AsyncCropWorker(model, cropper, max_queue_size=2, deadline_ms=100.0)
    worker.start()

    prop = CropProposal(crop_id=1, crop_box=(100.0, 100.0, 300.0, 300.0), trigger_reason="small_vru", trigger_score=0.8)
    dummy_crop = np.zeros((200, 200, 3), dtype=np.uint8)

    # 1. Fresh request with current timestamp (should meet deadline)
    now_ms = time.perf_counter() * 1000.0
    worker.submit_crop_request(frame_id=10, capture_timestamp_ms=now_ms, crop_image=dummy_crop, crop_int_box=(100, 100, 300, 300), full_frame_shape=(720, 1280), proposal=prop, full_frame_detections=[])

    # 2. Stale request simulated with capture timestamp 200 ms ago (should expire)
    stale_ms = now_ms - 200.0
    worker.submit_crop_request(frame_id=11, capture_timestamp_ms=stale_ms, crop_image=dummy_crop, crop_int_box=(100, 100, 300, 300), full_frame_shape=(720, 1280), proposal=prop, full_frame_detections=[])

    # Wait for processing
    time.sleep(0.3)
    worker.stop()

    res_10 = worker.get_result_for_frame(10)
    res_11 = worker.get_result_for_frame(11)

    assert res_10 is not None
    assert res_10.status == "MET_DEADLINE"
    assert res_10.total_result_age_ms <= 100.0

    assert res_11 is not None
    assert res_11.status == "EXPIRED"
    assert res_11.total_result_age_ms > 100.0


def test_frame_association_integrity():
    """Verify crop results maintain strict 1:1 frame ID association and never attach to wrong frames."""
    cfg = FoveationConfig()
    cropper = ActiveCropper(cfg)
    model = MockYOLOModel()

    worker = AsyncCropWorker(model, cropper, max_queue_size=5, deadline_ms=500.0)
    worker.start()

    dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    for f_id in range(100, 105):
        prop = CropProposal(crop_id=f_id, crop_box=(10.0, 10.0, 50.0, 50.0), trigger_reason=f"trig_{f_id}", trigger_score=0.7)
        now_ms = time.perf_counter() * 1000.0
        worker.submit_crop_request(frame_id=f_id, capture_timestamp_ms=now_ms, crop_image=dummy_crop, crop_int_box=(0, 0, 50, 50), full_frame_shape=(720, 1280), proposal=prop, full_frame_detections=[])

    time.sleep(0.3)
    worker.stop()

    for f_id in range(100, 105):
        res = worker.get_result_for_frame(f_id)
        assert res is not None
        assert res.frame_id == f_id
        assert res.proposal.trigger_reason == f"trig_{f_id}"
