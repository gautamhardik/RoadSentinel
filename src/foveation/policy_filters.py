"""
RoadSentinel Phase 8 — Combined Policy Filters
Implements isolated, testable modules for:
1. Intra-crop pedestrian NMS (IoU 0.40) applied prior to multi-scale WBF fusion.
2. Unanchored crop-only confidence gating (conf >= 0.35) while strictly preserving
   full-frame detections and anchored multi-scale detections.
"""

from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import numpy as np

from src.foveation.types import Detection, FoveationConfig, VRU_CLASS_NAMES
from src.foveation.uncertainty_signals import compute_box_iou


def apply_intra_crop_pedestrian_nms(
    crop_detections: List[Detection],
    target_class: str = "person",
    iou_threshold: float = 0.40,
) -> List[Detection]:
    """
    Applies strict intra-crop Non-Maximum Suppression to the specified class (default: 'person')
    before global Weighted Box Fusion to suppress crowd-cluster duplicate inflation.
    
    Other classes (rider, bike, motor, vehicles) are preserved without modification.
    Detections are sorted descending by confidence, and overlapping boxes with IoU >= iou_threshold
    are suppressed.
    """
    if not crop_detections:
        return []

    target_dets = [d for d in crop_detections if d.class_name == target_class]
    other_dets = [d for d in crop_detections if d.class_name != target_class]

    if len(target_dets) <= 1:
        return list(crop_detections)

    # Sort descending by confidence
    sorted_targets = sorted(target_dets, key=lambda d: d.confidence, reverse=True)
    kept_targets: List[Detection] = []
    suppressed_indices = set()

    for i in range(len(sorted_targets)):
        if i in suppressed_indices:
            continue
        curr = sorted_targets[i]
        kept_targets.append(curr)

        for j in range(i + 1, len(sorted_targets)):
            if j in suppressed_indices:
                continue
            cand = sorted_targets[j]
            if compute_box_iou(curr.box, cand.box) >= iou_threshold:
                suppressed_indices.add(j)

    return kept_targets + other_dets


def filter_unanchored_crop_detections(
    full_frame_dets: List[Detection],
    crop_dets: List[Detection],
    unanchored_conf_threshold: float = 0.35,
    anchoring_iou_threshold: float = 0.50,
) -> List[Detection]:
    """
    Filters crop detections based on whether they are anchored by full-frame evidence:
    - Anchored: Has a matching full-frame detection (same class, IoU >= anchoring_iou_threshold).
      Retained even if confidence < unanchored_conf_threshold, allowing multi-scale WBF agreement.
    - Unanchored: No matching full-frame detection. Must satisfy confidence >= unanchored_conf_threshold
      to prevent spurious ungrounded crop hallucinations.

    Note: Original full_frame_dets are NOT modified or filtered by this function.
    Returns: Filtered list of crop detections to be passed to WBF.
    """
    if not crop_dets:
        return []

    # Group full-frame detections by class for fast lookup
    ff_by_class: Dict[int, List[Detection]] = defaultdict(list)
    for d in full_frame_dets:
        ff_by_class[d.class_id].append(d)

    surviving_crop_dets: List[Detection] = []

    for cr in crop_dets:
        # Check if cr matches any full-frame detection of the same class
        is_anchored = False
        matching_ff = ff_by_class.get(cr.class_id, [])
        for ff in matching_ff:
            if compute_box_iou(cr.box, ff.box) >= anchoring_iou_threshold:
                is_anchored = True
                break

        if is_anchored:
            # Anchored by full-frame context: retain for WBF fusion
            surviving_crop_dets.append(cr)
        else:
            # Unanchored detection: enforce stricter confidence gate
            if cr.confidence >= unanchored_conf_threshold:
                surviving_crop_dets.append(cr)

    return surviving_crop_dets
