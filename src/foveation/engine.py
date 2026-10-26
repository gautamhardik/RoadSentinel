"""
RoadSentinel Phase 3 — Active Foveation Engine: Core Orchestrator
Coordinates OpenCV 5 image ingestion, full-frame detection, uncertainty-triggered crop generation,
crop-level high-resolution inference, coordinate transformation, and multi-scale box fusion.
"""

import time
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.foveation.types import (
    Detection, CropProposal, CropResult, FoveationTiming, FoveationTrace,
    FoveationConfig, CLASS_NAMES, VRU_CLASS_NAMES
)
from src.foveation.uncertainty_signals import (
    UncertaintySignalEvaluator, propose_fixed_grid_crops, propose_random_crops
)
from src.foveation.cropper import ActiveCropper
from src.foveation.box_fusion import BoxFusionEngine


# Pretrained Checkpoint classes vs Ground Truth IDs
CHECKPOINT_CLASSES: Dict[int, str] = {
    0: "person", 1: "rider", 2: "car", 3: "truck", 4: "bus",
    5: "train", 6: "motor", 7: "bike", 8: "traffic light", 9: "traffic sign"
}
CHECKPOINT_TO_GT_ID: Dict[int, int] = {
    0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 9, 6: 6, 7: 5, 8: 7, 9: 8
}
GT_CLASSES: Dict[int, str] = {
    0: "person", 1: "rider", 2: "car", 3: "bus", 4: "truck",
    5: "bike", 6: "motor", 7: "traffic light", 8: "traffic sign", 9: "train"
}


class ActiveFoveationEngine:
    """End-to-end active perception engine for uncertainty-aware VRU detection."""

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[FoveationConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or FoveationConfig()
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = YOLO(checkpoint_path)
        self.model.to(self.device)

        self.signal_evaluator = UncertaintySignalEvaluator(self.config)
        self.cropper = ActiveCropper(self.config)
        self.fusion_engine = BoxFusionEngine(self.config)

        # Build dynamic class mapping based on model.names to ensure correct GT IDs
        gt_name_to_id = {v: k for k, v in GT_CLASSES.items()}
        if hasattr(self.model, "names") and isinstance(self.model.names, dict):
            self.checkpoint_to_gt_id = {
                int(k): gt_name_to_id.get(v, int(k))
                for k, v in self.model.names.items()
            }
        else:
            self.checkpoint_to_gt_id = CHECKPOINT_TO_GT_ID

    def _parse_yolo_boxes(self, yolo_res, offset_x: float = 0.0, offset_y: float = 0.0) -> List[Detection]:
        """Converts Ultralytics prediction boxes into Detection objects."""
        dets = []
        for b in yolo_res.boxes:
            ckpt_cid = int(b.cls.item())
            conf = float(b.conf.item())
            coords = b.xyxy[0].cpu().numpy().tolist()

            gt_cid = self.checkpoint_to_gt_id.get(ckpt_cid, ckpt_cid)
            cname = GT_CLASSES.get(gt_cid, "unknown")

            x1 = float(coords[0]) + offset_x
            y1 = float(coords[1]) + offset_y
            x2 = float(coords[2]) + offset_x
            y2 = float(coords[3]) + offset_y

            dets.append(Detection(
                box=(x1, y1, x2, y2),
                class_id=gt_cid,
                class_name=cname,
                confidence=conf,
                source="full_frame",
            ))
        return dets

    def process_image(
        self,
        image_input: Any,  # file path str, Path, or np.ndarray (OpenCV BGR)
        image_id: Optional[str] = None,
        conf_threshold: float = 0.01,
    ) -> Tuple[List[Detection], FoveationTrace]:
        """
        Executes the full active foveation perception pipeline on a single frame.
        Returns:
            fused_detections: List of final deduplicated detections
            trace: Full per-image execution trace
        """
        timing = FoveationTiming()
        t_total_start = time.perf_counter()

        # 1. Image Ingestion via OpenCV 5
        t_read_start = time.perf_counter()
        if isinstance(image_input, (str, Path)):
            img_path = str(image_input)
            img_id = image_id or Path(img_path).name
            frame = cv2.imread(img_path)
            if frame is None:
                raise FileNotFoundError(f"Failed to read image at {img_path}")
        elif isinstance(image_input, np.ndarray):
            frame = image_input
            img_id = image_id or f"frame_{int(time.time()*1000)}"
        else:
            raise TypeError("image_input must be a file path or numpy.ndarray")
        timing.read_ms = (time.perf_counter() - t_read_start) * 1000.0

        img_h, img_w = frame.shape[:2]

        # 2. Full-Frame Detection Pass
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        t_ff_start = time.perf_counter()
        ff_results = self.model.predict(
            source=frame,
            imgsz=self.config.full_frame_imgsz,
            conf=conf_threshold,
            device=self.device,
            verbose=False,
        )
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        timing.full_frame_infer_ms = (time.perf_counter() - t_ff_start) * 1000.0

        full_frame_dets = self._parse_yolo_boxes(ff_results[0])

        # If policy is full-frame only, bypass crop proposals
        if self.config.policy == "full_frame_only" or self.config.max_crops_per_image <= 0:
            timing.total_ms = (time.perf_counter() - t_total_start) * 1000.0
            trace = FoveationTrace(
                image_id=img_id,
                image_shape=(img_h, img_w),
                policy=self.config.policy,
                full_frame_detections=full_frame_dets,
                crop_proposals=[],
                crop_results=[],
                fused_detections=full_frame_dets,
                timing=timing,
                num_crops_executed=0,
                notes="Bypassed crop extraction per policy.",
            )
            return full_frame_dets, trace

        # 3. Crop Proposal Generation
        t_prop_start = time.perf_counter()
        if self.config.policy == "fixed_grid":
            proposals = propose_fixed_grid_crops(img_w, img_h, k=self.config.max_crops_per_image)
        elif self.config.policy == "random":
            proposals = propose_random_crops(img_w, img_h, k=self.config.max_crops_per_image)
        else:  # 'uncertainty_active'
            proposals = self.signal_evaluator.extract_candidate_signals(full_frame_dets, img_w, img_h)
        timing.uncertainty_proposal_ms = (time.perf_counter() - t_prop_start) * 1000.0

        # 4. Active Crop Inference Pass
        crop_results: List[CropResult] = []
        all_remapped_crop_dets: List[Detection] = []

        t_crop_extract_total = 0.0
        t_crop_infer_total = 0.0

        for prop in proposals:
            t_ext_0 = time.perf_counter()
            sub_frame, crop_box_int = self.cropper.extract_crop_image(frame, prop.crop_box)
            t_crop_extract_total += (time.perf_counter() - t_ext_0) * 1000.0

            if sub_frame.shape[0] < 10 or sub_frame.shape[1] < 10:
                continue

            if torch.cuda.is_available() and "cuda" in self.device:
                torch.cuda.synchronize()
            t_c_inf_0 = time.perf_counter()
            sub_res = self.model.predict(
                source=sub_frame,
                imgsz=self.config.crop_imgsz,
                conf=conf_threshold,
                device=self.device,
                verbose=False,
            )
            if torch.cuda.is_available() and "cuda" in self.device:
                torch.cuda.synchronize()
            c_inf_time = (time.perf_counter() - t_c_inf_0) * 1000.0
            t_crop_infer_total += c_inf_time

            raw_crop_dets = self._parse_yolo_boxes(sub_res[0])
            remapped = self.cropper.remap_crop_detections_to_frame(
                raw_crop_dets, crop_box_int, (img_h, img_w), prop
            )

            crop_results.append(CropResult(
                proposal=prop,
                crop_shape=(sub_frame.shape[0], sub_frame.shape[1]),
                raw_crop_detections=raw_crop_dets,
                remapped_detections=remapped,
                inference_time_ms=c_inf_time,
            ))
            all_remapped_crop_dets.extend(remapped)

        timing.crop_extraction_ms = t_crop_extract_total
        timing.crop_infer_ms = t_crop_infer_total

        # 5. Multi-Scale Box Fusion
        t_fus_start = time.perf_counter()
        fused_dets = self.fusion_engine.fuse_detections(full_frame_dets, all_remapped_crop_dets)
        timing.fusion_ms = (time.perf_counter() - t_fus_start) * 1000.0

        timing.total_ms = (time.perf_counter() - t_total_start) * 1000.0

        trace = FoveationTrace(
            image_id=img_id,
            image_shape=(img_h, img_w),
            policy=self.config.policy,
            full_frame_detections=full_frame_dets,
            crop_proposals=proposals,
            crop_results=crop_results,
            fused_detections=fused_dets,
            timing=timing,
            num_crops_executed=len(crop_results),
            notes=f"Active foveation completed with {len(crop_results)} crops.",
        )

        return fused_dets, trace
