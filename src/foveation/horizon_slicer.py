"""
RoadSentinel Phase 7 — Autonomous Horizon-Corridor Slicer Module
Implements fixed horizon-band extraction, coordinate transformations,
boundary artifact suppression, and crop validation for autonomous driving scenes.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import cv2


class HorizonSlicer:
    """
    Extracts and maps a central horizon corridor patch from driving camera frames.
    
    Default Geometry (1280x720 input):
    - x1=160, y1=200, x2=1120, y2=520 (Width=960, Height=320)
    - Covers central 75% horizontal corridor and horizon elevation range [0.278, 0.722].
    """

    def __init__(
        self,
        rel_x1: float = 0.125,  # 160 / 1280
        rel_y1: float = 0.2778, # 200 / 720
        rel_x2: float = 0.875,  # 1120 / 1280
        rel_y2: float = 0.7222, # 520 / 720
        boundary_margin_px: float = 8.0,
        boundary_suppress_conf: float = 0.30,
    ):
        self.rel_x1 = rel_x1
        self.rel_y1 = rel_y1
        self.rel_x2 = rel_x2
        self.rel_y2 = rel_y2
        self.boundary_margin_px = boundary_margin_px
        self.boundary_suppress_conf = boundary_suppress_conf

    def get_crop_box(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Calculates pixel coordinates of the horizon band."""
        x1 = max(0, int(round(self.rel_x1 * img_w)))
        y1 = max(0, int(round(self.rel_y1 * img_h)))
        x2 = min(img_w, int(round(self.rel_x2 * img_w)))
        y2 = min(img_h, int(round(self.rel_y2 * img_h)))
        return (x1, y1, x2, y2)

    def extract_slice(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extracts the horizon slice from the input frame.
        Handles empty or degraded inputs gracefully.
        """
        if frame is None or frame.size == 0:
            return np.zeros((32, 32, 3), dtype=np.uint8), (0, 0, 0, 0)

        img_h, img_w = frame.shape[:2]
        if img_h < 16 or img_w < 16:
            return np.zeros((32, 32, 3), dtype=np.uint8), (0, 0, 0, 0)

        x1, y1, x2, y2 = self.get_crop_box(img_w, img_h)
        sub_img = frame[y1:y2, x1:x2].copy()
        if sub_img.size == 0 or sub_img.shape[0] < 4 or sub_img.shape[1] < 4:
            return np.zeros((32, 32, 3), dtype=np.uint8), (0, 0, 0, 0)

        return sub_img, (x1, y1, x2, y2)

    def remap_detection(
        self,
        local_box: Tuple[float, float, float, float],
        slice_coords: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int],
        conf: float = 1.0,
    ) -> Optional[np.ndarray]:
        """
        Inverse coordinate mapping from local slice to global frame.
        Applies boundary artifact suppression and coordinate clamping.
        Returns np.ndarray [gx1, gy1, gx2, gy2] or None if suppressed.
        """
        img_h, img_w = frame_shape[:2]
        cx1, cy1, cx2, cy2 = slice_coords
        sub_w = cx2 - cx1
        sub_h = cy2 - cy1

        if sub_w <= 0 or sub_h <= 0:
            return None

        rx1, ry1, rx2, ry2 = local_box

        # Boundary artifact check
        margin = self.boundary_margin_px
        is_boundary = (
            (rx1 <= margin and cx1 > 0)
            or (rx2 >= sub_w - margin and cx2 < img_w)
            or (ry1 <= margin and cy1 > 0)
            or (ry2 >= sub_h - margin and cy2 < img_h)
        )
        if is_boundary and conf < self.boundary_suppress_conf:
            return None

        gx1 = max(0.0, min(float(img_w), float(cx1 + rx1)))
        gy1 = max(0.0, min(float(img_h), float(cy1 + ry1)))
        gx2 = max(0.0, min(float(img_w), float(cx1 + rx2)))
        gy2 = max(0.0, min(float(img_h), float(cy1 + ry2)))

        # Ensure valid area
        if (gx2 - gx1) < 2.0 or (gy2 - gy1) < 2.0:
            return None

        return np.array([gx1, gy1, gx2, gy2], dtype=np.float32)
