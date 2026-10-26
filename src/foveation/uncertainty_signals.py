"""
RoadSentinel Phase 3 — Active Foveation Engine: Uncertainty Signals & Crop Proposals
Implements candidate uncertainty heuristics, spatial clustering, boundary truncation,
and baseline comparison policies (fixed-grid and random crops).
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from src.foveation.types import Detection, CropProposal, FoveationConfig, VRU_CLASS_NAMES


def compute_box_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Computes IoU between two bounding boxes [x1, y1, x2, y2]."""
    xi1 = max(boxA[0], boxB[0])
    yi1 = max(boxA[1], boxB[1])
    xi2 = min(boxA[2], boxB[2])
    yi2 = min(boxA[3], boxB[3])

    iw = max(0.0, xi2 - xi1)
    ih = max(0.0, yi2 - yi1)
    inter = iw * ih

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    if union <= 1e-8:
        return 0.0
    return inter / union


def expand_box_with_margin(
    box: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    margin_ratio: float = 0.25,
    min_size: int = 128,
) -> Tuple[float, float, float, float]:
    """
    Expands a bounding box by a context margin ratio while enforcing minimum size
    and clipping strictly to [0, img_w] x [0, img_h].
    """
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    pad_w = max(bw * margin_ratio, (min_size - bw) / 2.0 if bw < min_size else 0.0)
    pad_h = max(bh * margin_ratio, (min_size - bh) / 2.0 if bh < min_size else 0.0)

    cx1 = max(0.0, x1 - pad_w)
    cy1 = max(0.0, y1 - pad_h)
    cx2 = min(float(img_w), x2 + pad_w)
    cy2 = min(float(img_h), y2 + pad_h)

    # Ensure min size if image boundaries permit
    curr_w = cx2 - cx1
    curr_h = cy2 - cy1
    if curr_w < min_size and img_w >= min_size:
        needed = min_size - curr_w
        cx1 = max(0.0, cx1 - needed / 2.0)
        cx2 = min(float(img_w), cx1 + min_size)
        if cx2 >= float(img_w):
            cx1 = max(0.0, float(img_w) - min_size)

    if curr_h < min_size and img_h >= min_size:
        needed = min_size - curr_h
        cy1 = max(0.0, cy1 - needed / 2.0)
        cy2 = min(float(img_h), cy1 + min_size)
        if cy2 >= float(img_h):
            cy1 = max(0.0, float(img_h) - min_size)

    return (float(cx1), float(cy1), float(cx2), float(cy2))


def merge_overlapping_proposals(
    proposals: List[CropProposal],
    img_w: int,
    img_h: int,
    iou_merge_threshold: float = 0.30,
) -> List[CropProposal]:
    """
    Merges overlapping crop proposals into enclosing bounding boxes
    to prevent redundant duplicate crops over the same image region.
    """
    if not proposals:
        return []

    # Sort proposals descending by trigger score
    sorted_props = sorted(proposals, key=lambda p: p.trigger_score, reverse=True)
    merged: List[CropProposal] = []

    for prop in sorted_props:
        matched = False
        for idx, m in enumerate(merged):
            if compute_box_iou(prop.crop_box, m.crop_box) > iou_merge_threshold:
                # Merge into union box
                ux1 = min(prop.crop_box[0], m.crop_box[0])
                uy1 = min(prop.crop_box[1], m.crop_box[1])
                ux2 = max(prop.crop_box[2], m.crop_box[2])
                uy2 = max(prop.crop_box[3], m.crop_box[3])
                new_score = max(prop.trigger_score, m.trigger_score)
                merged[idx] = CropProposal(
                    crop_id=m.crop_id,
                    crop_box=(float(ux1), float(uy1), float(ux2), float(uy2)),
                    trigger_reason=f"{m.trigger_reason}+{prop.trigger_reason}",
                    trigger_score=new_score,
                    source_detection_idx=m.source_detection_idx,
                    context_margin=m.context_margin,
                )
                matched = True
                break
        if not matched:
            merged.append(prop)

    # Renumber crop_ids cleanly
    for i, m in enumerate(merged):
        m.crop_id = i + 1

    return merged


class UncertaintySignalEvaluator:
    """Evaluates transparent, measurable uncertainty heuristics on full-frame detections."""

    def __init__(self, config: FoveationConfig):
        self.config = config

    def extract_candidate_signals(
        self,
        detections: List[Detection],
        img_w: int,
        img_h: int,
    ) -> List[CropProposal]:
        """Evaluates candidate uncertainty triggers across all full-frame detections."""
        candidates: List[CropProposal] = []

        # 1. Low Confidence & Small Box Signals
        for idx, det in enumerate(detections):
            # Prioritize VRUs or objects with high likelihood of being missed road users
            vru_mult = 1.3 if det.is_vru else 1.0

            # Signal A: Low Confidence
            if self.config.low_conf_min <= det.confidence < self.config.low_conf_max:
                score = (self.config.low_conf_max - det.confidence) / (self.config.low_conf_max - self.config.low_conf_min)
                score = min(1.0, score * vru_mult)
                crop_box = expand_box_with_margin(
                    det.box, img_w, img_h,
                    margin_ratio=self.config.context_margin_ratio,
                    min_size=self.config.min_crop_size
                )
                candidates.append(CropProposal(
                    crop_id=len(candidates) + 1,
                    crop_box=crop_box,
                    trigger_reason="low_confidence",
                    trigger_score=score,
                    source_detection_idx=idx,
                    context_margin=self.config.context_margin_ratio,
                ))

            # Signal B: Small Bounding Box (High scale ambiguity)
            if det.area < self.config.small_box_area_threshold and det.is_vru:
                area_ratio = 1.0 - (det.area / self.config.small_box_area_threshold)
                score = min(1.0, area_ratio * vru_mult)
                crop_box = expand_box_with_margin(
                    det.box, img_w, img_h,
                    margin_ratio=self.config.context_margin_ratio * 1.5,
                    min_size=self.config.min_crop_size
                )
                candidates.append(CropProposal(
                    crop_id=len(candidates) + 1,
                    crop_box=crop_box,
                    trigger_reason="small_vru",
                    trigger_score=score,
                    source_detection_idx=idx,
                    context_margin=self.config.context_margin_ratio * 1.5,
                ))

            # Signal C: Boundary Truncation
            is_at_border = (
                det.x1 <= self.config.boundary_margin_ratio * img_w or
                det.x2 >= (1.0 - self.config.boundary_margin_ratio) * img_w or
                det.y1 <= self.config.boundary_margin_ratio * img_h or
                det.y2 >= (1.0 - self.config.boundary_margin_ratio) * img_h
            )
            if is_at_border and det.is_vru:
                crop_box = expand_box_with_margin(
                    det.box, img_w, img_h,
                    margin_ratio=self.config.context_margin_ratio * 1.5,
                    min_size=self.config.min_crop_size
                )
                candidates.append(CropProposal(
                    crop_id=len(candidates) + 1,
                    crop_box=crop_box,
                    trigger_reason="boundary_truncation",
                    trigger_score=0.6 * vru_mult,
                    source_detection_idx=idx,
                    context_margin=self.config.context_margin_ratio,
                ))

        # 2. Signal D: Spatial Clustering / Overlapping Detections
        n_dets = len(detections)
        for i in range(n_dets):
            for j in range(i + 1, n_dets):
                d_i = detections[i]
                d_j = detections[j]
                if not (d_i.is_vru or d_j.is_vru):
                    continue
                iou = compute_box_iou(d_i.box, d_j.box)
                if iou >= self.config.cluster_iou_threshold:
                    union_box = (
                        min(d_i.x1, d_j.x1),
                        min(d_i.y1, d_j.y1),
                        max(d_i.x2, d_j.x2),
                        max(d_i.y2, d_j.y2)
                    )
                    crop_box = expand_box_with_margin(
                        union_box, img_w, img_h,
                        margin_ratio=self.config.context_margin_ratio,
                        min_size=self.config.min_crop_size
                    )
                    candidates.append(CropProposal(
                        crop_id=len(candidates) + 1,
                        crop_box=crop_box,
                        trigger_reason="cluster_occlusion",
                        trigger_score=min(1.0, 0.7 + iou * 0.3),
                        source_detection_idx=i,
                        context_margin=self.config.context_margin_ratio,
                    ))

        # Merge overlapping proposals to avoid redundant execution
        merged = merge_overlapping_proposals(candidates, img_w, img_h)
        # Cap at configured budget
        return merged[:self.config.max_crops_per_image]


def propose_fixed_grid_crops(
    img_w: int,
    img_h: int,
    k: int = 3,
) -> List[CropProposal]:
    """
    Proposes a fixed non-uncertainty crop baseline covering primary road regions
    (e.g., center forward road, left sidewalk/curb, right sidewalk/curb).
    """
    proposals: List[CropProposal] = []
    # Standard driving scene partitions on 1280x720:
    # 1. Forward Center Path: [320, 200, 960, 680]
    # 2. Right Crosswalk / Sidewalk: [700, 240, 1280, 680]
    # 3. Left Sidewalk / Oncoming: [0, 240, 580, 680]
    fixed_boxes = [
        (0.25 * img_w, 0.28 * img_h, 0.75 * img_w, 0.94 * img_h),  # Center road
        (0.55 * img_w, 0.33 * img_h, 1.00 * img_w, 0.94 * img_h),  # Right curb
        (0.00 * img_w, 0.33 * img_h, 0.45 * img_w, 0.94 * img_h),  # Left curb
        (0.35 * img_w, 0.30 * img_h, 0.65 * img_w, 0.70 * img_h),  # Distant horizon
    ]
    for i in range(min(k, len(fixed_boxes))):
        fb = fixed_boxes[i]
        proposals.append(CropProposal(
            crop_id=i + 1,
            crop_box=(float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3])),
            trigger_reason="fixed_grid",
            trigger_score=1.0 - (0.1 * i),
            context_margin=0.0,
        ))
    return proposals


def propose_random_crops(
    img_w: int,
    img_h: int,
    k: int = 3,
    seed: Optional[int] = None,
    min_size: int = 256,
    max_size: int = 512,
) -> List[CropProposal]:
    """
    Proposes a random crop baseline with an identical budget to test
    whether uncertainty-guided selection outperforms random spatial sampling.
    """
    rng = np.random.RandomState(seed)
    proposals: List[CropProposal] = []
    for i in range(k):
        cw = rng.randint(min_size, max(min_size + 1, min(max_size, img_w)))
        ch = rng.randint(min_size, max(min_size + 1, min(max_size, img_h)))
        cx1 = rng.randint(0, max(1, img_w - cw))
        cy1 = rng.randint(int(0.2 * img_h), max(int(0.2 * img_h) + 1, img_h - ch))  # Bias towards road zone
        cx2 = cx1 + cw
        cy2 = cy1 + ch
        proposals.append(CropProposal(
            crop_id=i + 1,
            crop_box=(float(cx1), float(cy1), float(cx2), float(cy2)),
            trigger_reason="random",
            trigger_score=0.5,
            context_margin=0.0,
        ))
    return proposals
