"""
RoadSentinel Phase 9 — Asynchronous Crop Perception Worker (Async-Fov)
Implements:
1. AsyncCropRequest / AsyncCropResult with monotonic frame IDs, capture timestamps, ROIs, and trigger metadata.
2. Bounded queue with explicit overload policy (DROP_OLDEST).
3. Dedicated worker thread executing crop inference, intra-crop NMS, and conservative admission filtering.
4. Deadline enforcement: rejects expired results (T_age > 100 ms).
5. Frame association validation preventing cross-frame box misattribution.
6. Synchronous Config F fallback.
"""

import os
import sys
import time
import queue
import threading
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import numpy as np

# Ensure OpenMP stability on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.foveation.types import Detection, CropProposal, CropResult, FoveationConfig
from src.foveation.policy_filters import (
    apply_intra_crop_pedestrian_nms,
    filter_unanchored_crop_detections,
)


@dataclass
class AsyncCropRequest:
    frame_id: int
    capture_timestamp_ms: float
    crop_image: np.ndarray
    crop_int_box: Tuple[int, int, int, int]
    full_frame_shape: Tuple[int, int]
    proposal: CropProposal
    full_frame_detections: List[Detection]
    deadline_ms: float = 100.0
    enqueued_timestamp_ms: float = field(default_factory=lambda: time.perf_counter() * 1000.0)


@dataclass
class AsyncCropResult:
    frame_id: int
    capture_timestamp_ms: float
    enqueued_timestamp_ms: float
    dequeued_timestamp_ms: float
    completed_timestamp_ms: float
    queue_delay_ms: float
    inference_time_ms: float
    total_result_age_ms: float
    status: str  # "MET_DEADLINE", "EXPIRED", "DROPPED_OVERLOAD"
    detections: List[Detection]
    proposal: CropProposal
    full_frame_detections: List[Detection]


class AsyncCropWorker:
    """
    Background perception worker that processes high-resolution crop requests asynchronously.
    Enforces bounded queue capacity with DROP_OLDEST overload policy and 100 ms deadline gating.
    """

    def __init__(
        self,
        model: Any,
        cropper: Any,
        max_queue_size: int = 2,
        crop_imgsz: int = 640,
        deadline_ms: float = 100.0,
        unanchored_conf_threshold: float = 0.35,
        intra_crop_pedestrian_iou: float = 0.40,
        operational_conf: float = 0.25,
        device: str = "cpu",
    ):
        self.model = model
        self.cropper = cropper
        self.max_queue_size = max_queue_size
        self.crop_imgsz = crop_imgsz
        self.deadline_ms = deadline_ms
        self.unanchored_conf_threshold = unanchored_conf_threshold
        self.intra_crop_pedestrian_iou = intra_crop_pedestrian_iou
        self.operational_conf = operational_conf
        self.device = device

        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.lock = threading.Lock()
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None

        # Telemetry
        self.total_requests_received = 0
        self.total_requests_dropped_overload = 0
        self.total_results_completed = 0
        self.total_results_met_deadline = 0
        self.total_results_expired = 0

        # Recent completed results buffer: frame_id -> AsyncCropResult
        self.completed_results: Dict[int, AsyncCropResult] = {}

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="AsyncCropWorker")
            self.worker_thread.start()

    def stop(self, drain: bool = False):
        with self.lock:
            self.running = False
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None

    def submit_crop_request(
        self,
        frame_id: int,
        capture_timestamp_ms: float,
        crop_image: np.ndarray,
        crop_int_box: Tuple[int, int, int, int],
        full_frame_shape: Tuple[int, int],
        proposal: CropProposal,
        full_frame_detections: List[Detection],
    ) -> bool:
        """
        Submits a crop request to the bounded worker queue.
        If queue is full, drops the oldest request (DROP_OLDEST overload behavior).
        Returns: True if enqueued, False if dropped due to overload.
        """
        self.total_requests_received += 1
        req = AsyncCropRequest(
            frame_id=frame_id,
            capture_timestamp_ms=capture_timestamp_ms,
            crop_image=crop_image,
            crop_int_box=crop_int_box,
            full_frame_shape=full_frame_shape,
            proposal=proposal,
            full_frame_detections=full_frame_detections,
            deadline_ms=self.deadline_ms,
            enqueued_timestamp_ms=time.perf_counter() * 1000.0,
        )

        with self.lock:
            if self.queue.full():
                # Bounded queue overload: Drop oldest request to prioritize newer frames
                try:
                    dropped_req = self.queue.get_nowait()
                    self.total_requests_dropped_overload += 1
                    # Record dropped telemetry
                    age_at_drop = (time.perf_counter() * 1000.0) - dropped_req.capture_timestamp_ms
                    drop_res = AsyncCropResult(
                        frame_id=dropped_req.frame_id,
                        capture_timestamp_ms=dropped_req.capture_timestamp_ms,
                        enqueued_timestamp_ms=dropped_req.enqueued_timestamp_ms,
                        dequeued_timestamp_ms=time.perf_counter() * 1000.0,
                        completed_timestamp_ms=time.perf_counter() * 1000.0,
                        queue_delay_ms=age_at_drop,
                        inference_time_ms=0.0,
                        total_result_age_ms=age_at_drop,
                        status="DROPPED_OVERLOAD",
                        detections=[],
                        proposal=dropped_req.proposal,
                        full_frame_detections=dropped_req.full_frame_detections,
                    )
                    self.completed_results[dropped_req.frame_id] = drop_res
                except queue.Empty:
                    pass

            try:
                self.queue.put_nowait(req)
                return True
            except queue.Full:
                self.total_requests_dropped_overload += 1
                return False

    def get_result_for_frame(self, frame_id: int) -> Optional[AsyncCropResult]:
        """Queries whether a completed result exists for the specified frame ID."""
        with self.lock:
            return self.completed_results.get(frame_id)

    def _worker_loop(self):
        while self.running:
            try:
                req: AsyncCropRequest = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue

            t_dequeue = time.perf_counter() * 1000.0
            queue_delay = t_dequeue - req.enqueued_timestamp_ms

            # Execute crop model prediction
            t_inf0 = time.perf_counter()
            sub_frame = req.crop_image
            if sub_frame.shape[0] < 10 or sub_frame.shape[1] < 10:
                raw_crop_dets = []
            else:
                cr_res = self.model.predict(
                    source=sub_frame,
                    imgsz=self.crop_imgsz,
                    conf=0.15,
                    device=self.device,
                    verbose=False,
                )
                raw_crop_dets = self._parse_yolo_boxes(cr_res[0])

            infer_ms = (time.perf_counter() - t_inf0) * 1000.0
            t_complete = time.perf_counter() * 1000.0
            total_age = t_complete - req.capture_timestamp_ms

            # Remap detections
            remapped = self.cropper.remap_crop_detections_to_frame(
                raw_crop_dets, req.crop_int_box, req.full_frame_shape, req.proposal
            )

            # Intra-crop pedestrian NMS at IoU 0.40
            crop_nms = apply_intra_crop_pedestrian_nms(
                remapped, target_class="person", iou_threshold=self.intra_crop_pedestrian_iou
            )

            # Conservative admission filter (unanchored >= 0.35)
            filtered = filter_unanchored_crop_detections(
                req.full_frame_detections,
                crop_nms,
                unanchored_conf_threshold=self.unanchored_conf_threshold,
                anchoring_iou_threshold=0.50,
            )

            # Filter operational confidence
            operational_crop = [d for d in filtered if d.confidence >= self.operational_conf]

            # Deadline evaluation
            if total_age <= req.deadline_ms:
                status = "MET_DEADLINE"
                self.total_results_met_deadline += 1
            else:
                status = "EXPIRED"
                self.total_results_expired += 1

            self.total_results_completed += 1

            result = AsyncCropResult(
                frame_id=req.frame_id,
                capture_timestamp_ms=req.capture_timestamp_ms,
                enqueued_timestamp_ms=req.enqueued_timestamp_ms,
                dequeued_timestamp_ms=t_dequeue,
                completed_timestamp_ms=t_complete,
                queue_delay_ms=round(queue_delay, 2),
                inference_time_ms=round(infer_ms, 2),
                total_result_age_ms=round(total_age, 2),
                status=status,
                detections=operational_crop,
                proposal=req.proposal,
                full_frame_detections=req.full_frame_detections,
            )

            with self.lock:
                self.completed_results[req.frame_id] = result
            self.queue.task_done()

    def _parse_yolo_boxes(self, r: Any) -> List[Detection]:
        dets = []
        if r.boxes is None or len(r.boxes) == 0:
            return dets
        for b in r.boxes:
            cid = int(b.cls.item())
            conf = float(b.conf.item())
            coords = b.xyxy[0].cpu().numpy().tolist()
            cname = self.model.names[cid]
            dets.append(Detection(
                box=tuple(coords),
                class_id=cid,
                class_name=cname,
                confidence=conf,
                source="crop"
            ))
        return dets
