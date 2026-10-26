"""
RoadSentinel Phase 3 — Phase 1 Reference Pipeline: Deterministic Full-Frame 960px
Decodes frames via OpenCV 5, runs YOLO11s at native 960px, logs timing and detections,
and operates completely independently of the active foveation crop pipeline.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.foveation.types import Detection, CLASS_NAMES, VRU_CLASS_NAMES


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


class ReferenceFullFramePipeline:
    """Independent deterministic full-frame 960px reference pipeline."""

    def __init__(self, checkpoint_path: str, imgsz: int = 960, device: Optional[str] = None):
        self.imgsz = imgsz
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(checkpoint_path)
        self.model.to(self.device)

    def process_frame(
        self,
        image_path: str,
        conf_threshold: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Processes a single image frame deterministically.
        Returns:
            Dictionary containing original dimensions, model input size, detections, and timing.
        """
        # 1. Decode frame using OpenCV 5
        t_decode_0 = time.perf_counter()
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
        t_decode_1 = time.perf_counter()
        decode_ms = (t_decode_1 - t_decode_0) * 1000.0

        orig_h, orig_w = img.shape[:2]

        # 2. Synchronized inference
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        t_inf_0 = time.perf_counter()
        results = self.model.predict(
            source=img,
            imgsz=self.imgsz,
            conf=conf_threshold,
            device=self.device,
            verbose=False,
        )
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        t_inf_1 = time.perf_counter()
        infer_ms = (t_inf_1 - t_inf_0) * 1000.0

        # 3. Extract detections in original image coordinates
        detections: List[Dict[str, Any]] = []
        for det in results[0].boxes:
            ckpt_cid = int(det.cls.item())
            conf = float(det.conf.item())
            coords = det.xyxy[0].cpu().numpy().tolist()

            gt_cid = CHECKPOINT_TO_GT_ID[ckpt_cid]
            cname = GT_CLASSES[gt_cid]

            detections.append({
                "class_id": gt_cid,
                "class_name": cname,
                "confidence": round(conf, 4),
                "box": [round(float(v), 2) for v in coords],
                "is_vru": cname in VRU_CLASS_NAMES,
            })

        return {
            "image_path": str(image_path),
            "original_dimensions": [int(orig_w), int(orig_h)],
            "model_input_dimensions": [int(self.imgsz), int(self.imgsz)],
            "decode_time_ms": round(decode_ms, 2),
            "inference_time_ms": round(infer_ms, 2),
            "total_detections": len(detections),
            "detections": detections,
        }
