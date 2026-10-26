"""
RoadSentinel Phase 3 — Active Foveation Engine: OpenCV 5 Image Cropper & Coordinate Transforms
Handles sub-image extraction with OpenCV 5, context expansion, boundary clipping,
and forward/inverse coordinate re-mapping.
"""

from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
from src.foveation.types import Detection, CropProposal, CropResult, FoveationConfig


class ActiveCropper:
    """Manages OpenCV 5 frame extraction, cropping, and coordinate transformations."""

    def __init__(self, config: FoveationConfig):
        self.config = config

    def extract_crop_image(
        self,
        frame: np.ndarray,
        crop_box: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extracts sub-image from frame using OpenCV slicing.
        Returns:
            cropped_image: np.ndarray (H_crop, W_crop, C)
            int_coords: (x1, y1, x2, y2) integer pixel bounds on original frame
        """
        img_h, img_w = frame.shape[:2]
        x1 = max(0, min(img_w - 1, int(round(crop_box[0]))))
        y1 = max(0, min(img_h - 1, int(round(crop_box[1]))))
        x2 = max(x1 + 1, min(img_w, int(round(crop_box[2]))))
        y2 = max(y1 + 1, min(img_h, int(round(crop_box[3]))))

        crop = frame[y1:y2, x1:x2].copy()
        return crop, (x1, y1, x2, y2)

    def remap_crop_detections_to_frame(
        self,
        crop_detections: List[Detection],
        crop_box_int: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int],
        proposal: CropProposal,
    ) -> List[Detection]:
        """
        Transforms bounding box coordinates from crop-relative coordinates
        back into full original-image coordinates.
        Suppresses false-positive edge artifacts where objects are severed by crop boundaries.
        """
        cx1, cy1, cx2, cy2 = crop_box_int
        frame_h, frame_w = frame_shape
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        remapped: List[Detection] = []

        for d in crop_detections:
            # Crop-relative coords
            rx1, ry1, rx2, ry2 = d.box

            # Boundary artifact filter:
            # If box touches a crop border that is inside the original image (not a frame boundary),
            # check if it's truncated/severed.
            margin = self.config.boundary_suppress_margin
            touches_internal_left = (rx1 <= margin) and (cx1 > margin)
            touches_internal_right = (rx2 >= crop_w - margin) and (cx2 < frame_w - margin)
            touches_internal_top = (ry1 <= margin) and (cy1 > margin)
            touches_internal_bottom = (ry2 >= crop_h - margin) and (cy2 < frame_h - margin)

            # If heavily truncated at an artificial crop slice, discard or penalize
            if touches_internal_left or touches_internal_right or touches_internal_top or touches_internal_bottom:
                # Discard low-confidence boundary fragments
                if d.confidence < 0.35:
                    continue

            # Forward map to original frame coordinates
            ox1 = max(0.0, min(float(frame_w), cx1 + rx1))
            oy1 = max(0.0, min(float(frame_h), cy1 + ry1))
            ox2 = max(0.0, min(float(frame_w), cx1 + rx2))
            oy2 = max(0.0, min(float(frame_h), cy1 + ry2))

            if (ox2 - ox1) < 2.0 or (oy2 - oy1) < 2.0:
                continue

            remapped.append(Detection(
                box=(float(ox1), float(oy1), float(ox2), float(oy2)),
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                source="crop",
                crop_id=proposal.crop_id,
                trigger_reason=proposal.trigger_reason,
                uncertainty_score=proposal.trigger_score,
            ))

        return remapped
