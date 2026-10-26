"""
RoadSentinel Phase 3 — Active Foveation Engine: Box Fusion & Deduplication
Fuses full-frame and crop detections using Weighted Box Fusion (WBF) and Non-Maximum Suppression (NMS),
preserving provenance metadata and source origins.
"""

from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import numpy as np
from src.foveation.types import Detection, FoveationConfig
from src.foveation.uncertainty_signals import compute_box_iou


class BoxFusionEngine:
    """Fuses multi-scale detections from full-frame and selective crop passes."""

    def __init__(self, config: FoveationConfig):
        self.config = config

    def fuse_detections(
        self,
        full_frame_dets: List[Detection],
        crop_dets: List[Detection],
    ) -> List[Detection]:
        """
        Deduplicates and merges full-frame and crop detections class-by-class.
        - High IoU matches: Fuses boxes using confidence weighting, updates source to 'fused'.
        - Unmatched crop detections: Kept as newly recovered objects (source='crop').
        - Unmatched full-frame detections: Retained (source='full_frame').
        """
        fused_results: List[Detection] = []

        # Group by class
        ff_by_class: Dict[int, List[Detection]] = defaultdict(list)
        crop_by_class: Dict[int, List[Detection]] = defaultdict(list)

        for d in full_frame_dets:
            ff_by_class[d.class_id].append(d)

        for d in crop_dets:
            crop_by_class[d.class_id].append(d)

        all_classes = set(ff_by_class.keys()).union(crop_by_class.keys())

        for cid in all_classes:
            ff_list = ff_by_class[cid]
            crop_list = crop_by_class[cid]

            matched_crop_indices = set()

            for ff_d in ff_list:
                best_match_idx = -1
                best_iou = 0.0

                for c_idx, cr_d in enumerate(crop_list):
                    if c_idx in matched_crop_indices:
                        continue
                    iou = compute_box_iou(ff_d.box, cr_d.box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = c_idx

                if best_iou >= self.config.fusion_iou_threshold and best_match_idx >= 0:
                    # Match found: Fuse with confidence weighting
                    matched_crop_indices.add(best_match_idx)
                    cr_d = crop_list[best_match_idx]

                    w_ff = ff_d.confidence
                    w_cr = cr_d.confidence * 1.2  # Slight bias towards higher-resolution crop localization
                    w_sum = max(w_ff + w_cr, 1e-6)

                    fused_box = (
                        (ff_d.x1 * w_ff + cr_d.x1 * w_cr) / w_sum,
                        (ff_d.y1 * w_ff + cr_d.y1 * w_cr) / w_sum,
                        (ff_d.x2 * w_ff + cr_d.x2 * w_cr) / w_sum,
                        (ff_d.y2 * w_ff + cr_d.y2 * w_cr) / w_sum,
                    )

                    # Boost confidence from dual-pass agreement
                    fused_conf = min(0.99, max(ff_d.confidence, cr_d.confidence) + 0.05 * min(ff_d.confidence, cr_d.confidence))

                    fused_results.append(Detection(
                        box=fused_box,
                        class_id=cid,
                        class_name=ff_d.class_name,
                        confidence=fused_conf,
                        source="fused",
                        crop_id=cr_d.crop_id,
                        trigger_reason=cr_d.trigger_reason,
                        uncertainty_score=cr_d.uncertainty_score,
                    ))
                else:
                    # Unmatched full-frame detection: retain
                    fused_results.append(ff_d)

            # Newly recovered detections from crop that had no matching full-frame detection
            for c_idx, cr_d in enumerate(crop_list):
                if c_idx not in matched_crop_indices:
                    fused_results.append(cr_d)

        # Final intra-class NMS to remove any redundant duplicates
        final_dets = self._apply_nms(fused_results, iou_thresh=0.60)
        return final_dets

    def _apply_nms(self, detections: List[Detection], iou_thresh: float = 0.60) -> List[Detection]:
        """Applies non-maximum suppression per class."""
        if not detections:
            return []

        dets_by_class: Dict[int, List[Detection]] = defaultdict(list)
        for d in detections:
            dets_by_class[d.class_id].append(d)

        kept: List[Detection] = []
        for cid, d_list in dets_by_class.items():
            # Sort descending by confidence
            sorted_dets = sorted(d_list, key=lambda d: d.confidence, reverse=True)
            suppressed = set()

            for i in range(len(sorted_dets)):
                if i in suppressed:
                    continue
                d_i = sorted_dets[i]
                kept.append(d_i)

                for j in range(i + 1, len(sorted_dets)):
                    if j in suppressed:
                        continue
                    d_j = sorted_dets[j]
                    if compute_box_iou(d_i.box, d_j.box) >= iou_thresh:
                        suppressed.add(j)

        return kept
