"""
Evaluation Engine for RoadSentinel - Phase 2A
Provides verified, class-aligned detection evaluation, IoU matching,
101-point interpolated COCO AP, and VRU metric aggregation.
"""

import time
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


# ============================================================================
# Class Configuration & Alignments
# ============================================================================

# Ground truth label definitions (bdd100k_dataset/data.yaml / Kaggle yolo conversion)
GT_CLASSES: Dict[int, str] = {
    0: "person",
    1: "rider",
    2: "car",
    3: "bus",
    4: "truck",
    5: "bike",
    6: "motor",
    7: "traffic light",
    8: "traffic sign",
    9: "train",
}

# Checkpoint model.names (weights/bdd100k_yolo11s.pt / BDD100K 2020 Scalabel hierarchy)
CHECKPOINT_CLASSES: Dict[int, str] = {
    0: "person",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motor",
    7: "bike",
    8: "traffic light",
    9: "traffic sign",
}

# Explicit mapping: Checkpoint Class ID -> Ground Truth Class ID
CHECKPOINT_TO_GT_ID: Dict[int, int] = {
    0: 0,  # person -> person
    1: 1,  # rider -> rider
    2: 2,  # car -> car
    3: 4,  # truck (ckpt 3) -> truck (gt 4)
    4: 3,  # bus (ckpt 4) -> bus (gt 3)
    5: 9,  # train (ckpt 5) -> train (gt 9)
    6: 6,  # motor (ckpt 6) -> motor (gt 6)
    7: 5,  # bike (ckpt 7) -> bike (gt 5)
    8: 7,  # traffic light (ckpt 8) -> traffic light (gt 7)
    9: 8,  # traffic sign (ckpt 9) -> traffic sign (gt 8)
}

# Ground Truth Class ID -> Checkpoint Class ID
GT_TO_CHECKPOINT_ID: Dict[int, int] = {v: k for k, v in CHECKPOINT_TO_GT_ID.items()}

# All 10 canonical class names
ALL_CLASS_NAMES: List[str] = [
    "person",
    "rider",
    "car",
    "bus",
    "truck",
    "bike",
    "motor",
    "traffic light",
    "traffic sign",
    "train",
]

# Vulnerable Road User (VRU) classes
VRU_CLASS_NAMES: List[str] = ["person", "rider", "bike", "motor"]


# ============================================================================
# Geometry & Matching Utilities
# ============================================================================

def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) of two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter

    return float(inter / union) if union > 0.0 else 0.0


def compute_ap_coco(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Computes Average Precision (AP) using the standard 101-point COCO interpolation.
    Evaluates interpolated precision at 101 recall points: [0.00, 0.01, ..., 1.00].
    """
    if len(recalls) == 0:
        return 0.0

    recall_thresholds = np.linspace(0.0, 1.0, 101)
    interpolated_precisions = np.zeros(101)

    for i, r in enumerate(recall_thresholds):
        matching_precisions = precisions[recalls >= r]
        if len(matching_precisions) > 0:
            interpolated_precisions[i] = np.max(matching_precisions)
        else:
            interpolated_precisions[i] = 0.0

    return float(np.mean(interpolated_precisions))


# ============================================================================
# Core Evaluator
# ============================================================================

class DetectionEvaluator:
    """
    Class-aligned evaluator for object detection.
    Evaluates True Positives, False Positives, False Negatives,
    Precision, Recall, AP@50, and AP@50-95 with explicit alignment.
    """

    def __init__(
        self,
        iou_thresholds: Optional[List[float]] = None,
        operational_conf: float = 0.25,
        operational_iou: float = 0.50,
    ):
        if iou_thresholds is None:
            self.iou_thresholds = [round(x, 2) for x in np.linspace(0.50, 0.95, 10)]
        else:
            self.iou_thresholds = iou_thresholds

        self.operational_conf = operational_conf
        self.operational_iou = operational_iou

        # Ground truth storage: image_id -> list of {'name': str, 'box': [x1, y1, x2, y2]}
        self.ground_truths: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Predictions storage: list of {'image_id': str, 'name': str, 'conf': float, 'box': [x1, y1, x2, y2]}
        self.predictions: List[Dict[str, Any]] = []

    def add_ground_truth(self, image_id: str, class_name: str, box: List[float]):
        """Add a single ground truth annotation."""
        self.ground_truths[image_id].append({
            "name": class_name,
            "box": [float(b) for b in box],
        })

    def add_ground_truth_yolo(
        self,
        image_id: str,
        class_id: int,
        xc: float,
        yc: float,
        bw: float,
        bh: float,
        img_w: int,
        img_h: int,
    ):
        """Add a ground truth annotation from normalized YOLO coordinates [0, 1]."""
        if class_id not in GT_CLASSES:
            raise ValueError(f"Unknown ground truth class ID: {class_id}")
        class_name = GT_CLASSES[class_id]

        x1 = (xc - bw / 2.0) * img_w
        y1 = (yc - bh / 2.0) * img_h
        x2 = (xc + bw / 2.0) * img_w
        y2 = (yc + bh / 2.0) * img_h
        self.add_ground_truth(image_id, class_name, [x1, y1, x2, y2])

    def add_prediction(
        self,
        image_id: str,
        class_name: str,
        confidence: float,
        box: List[float],
    ):
        """Add a single detection prediction."""
        self.predictions.append({
            "image_id": image_id,
            "name": class_name,
            "conf": float(confidence),
            "box": [float(b) for b in box],
        })

    def add_prediction_from_checkpoint(
        self,
        image_id: str,
        checkpoint_class_id: int,
        confidence: float,
        box: List[float],
    ):
        """
        Add a prediction emitted by checkpoint, automatically mapping
        the checkpoint class ID to the correct semantic class name.
        """
        if checkpoint_class_id not in CHECKPOINT_CLASSES:
            raise ValueError(f"Unknown checkpoint class ID: {checkpoint_class_id}")
        class_name = CHECKPOINT_CLASSES[checkpoint_class_id]
        self.add_prediction(image_id, class_name, confidence, box)

    def evaluate(self) -> Dict[str, Any]:
        """
        Performs full evaluation across all classes and IoU thresholds.
        Returns detailed per-class metrics, aggregate VRU metrics, and dataset summary.
        """
        # Count ground truths per class and per image
        gt_by_class: Dict[str, Dict[str, List[List[float]]]] = {
            c: defaultdict(list) for c in ALL_CLASS_NAMES
        }
        gt_counts: Dict[str, int] = {c: 0 for c in ALL_CLASS_NAMES}

        for img_id, gts in self.ground_truths.items():
            for gt in gts:
                cname = gt["name"]
                gt_by_class[cname][img_id].append(gt["box"])
                gt_counts[cname] += 1

        # Group predictions by class
        preds_by_class: Dict[str, List[Dict[str, Any]]] = {
            c: [] for c in ALL_CLASS_NAMES
        }
        for pred in self.predictions:
            cname = pred["name"]
            preds_by_class[cname].append(pred)

        # Sort predictions per class by confidence descending
        for cname in ALL_CLASS_NAMES:
            preds_by_class[cname].sort(key=lambda x: x["conf"], reverse=True)

        per_class_results = []

        # Evaluate each class
        for cname in ALL_CLASS_NAMES:
            n_gt = gt_counts[cname]
            preds = preds_by_class[cname]
            n_pred = len(preds)

            # Operational metrics (at operational_conf and operational_iou)
            op_preds = [p for p in preds if p["conf"] >= self.operational_conf]
            n_op_preds = len(op_preds)

            # Match for operational threshold
            op_tp = 0
            op_fp = 0
            matched_gt_op: Dict[str, List[bool]] = {
                img_id: [False] * len(boxes)
                for img_id, boxes in gt_by_class[cname].items()
            }

            for p in op_preds:
                img_id = p["image_id"]
                cand_boxes = gt_by_class[cname].get(img_id, [])
                best_iou = 0.0
                best_gt_idx = -1
                for idx, gt_box in enumerate(cand_boxes):
                    iou = compute_box_iou(p["box"], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou >= self.operational_iou and best_gt_idx != -1 and not matched_gt_op[img_id][best_gt_idx]:
                    op_tp += 1
                    matched_gt_op[img_id][best_gt_idx] = True
                else:
                    op_fp += 1

            op_fn = n_gt - op_tp
            op_prec = (op_tp / n_op_preds) if n_op_preds > 0 else 0.0
            op_rec = (op_tp / n_gt) if n_gt > 0 else 0.0
            op_f1 = (2 * op_prec * op_rec / (op_prec + op_rec)) if (op_prec + op_rec) > 0 else 0.0

            # AP computation across IoU thresholds
            ap_at_iou: Dict[float, float] = {}

            if n_gt == 0:
                # Class absent in ground truth
                ap_50 = 0.0 if n_pred > 0 else 0.0
                ap_50_95 = 0.0
                is_absent = True
            else:
                is_absent = False
                for iou_thr in self.iou_thresholds:
                    matched_gt_iou: Dict[str, List[bool]] = {
                        img_id: [False] * len(boxes)
                        for img_id, boxes in gt_by_class[cname].items()
                    }
                    tp_list = np.zeros(n_pred)
                    fp_list = np.zeros(n_pred)

                    for i, p in enumerate(preds):
                        img_id = p["image_id"]
                        cand_boxes = gt_by_class[cname].get(img_id, [])
                        best_iou = 0.0
                        best_gt_idx = -1
                        for idx, gt_box in enumerate(cand_boxes):
                            iou = compute_box_iou(p["box"], gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = idx

                        if best_iou >= iou_thr and best_gt_idx != -1 and not matched_gt_iou[img_id][best_gt_idx]:
                            tp_list[i] = 1.0
                            matched_gt_iou[img_id][best_gt_idx] = True
                        else:
                            fp_list[i] = 1.0

                    cum_tp = np.cumsum(tp_list)
                    cum_fp = np.cumsum(fp_list)
                    rec = cum_tp / n_gt
                    prec = cum_tp / (cum_tp + cum_fp)

                    ap = compute_ap_coco(rec, prec)
                    ap_at_iou[iou_thr] = ap

                ap_50 = ap_at_iou.get(0.50, 0.0)
                ap_50_95 = float(np.mean([ap_at_iou[t] for t in self.iou_thresholds]))

            class_metric = {
                "name": cname,
                "is_vru": cname in VRU_CLASS_NAMES,
                "is_absent_in_gt": is_absent,
                "gt_count": n_gt,
                "pred_total_count": n_pred,
                "pred_operational_count": n_op_preds,
                "tp_50": op_tp,
                "fp_50": op_fp,
                "fn_50": op_fn,
                "precision_50": round(op_prec, 4),
                "recall_50": round(op_rec, 4),
                "f1_50": round(op_f1, 4),
                "ap_50": round(ap_50, 4) if not is_absent else None,
                "ap_50_95": round(ap_50_95, 4) if not is_absent else None,
            }
            per_class_results.append(class_metric)

        # Aggregate metrics
        present_classes = [c for c in per_class_results if not c["is_absent_in_gt"]]
        
        # Macro averages across present classes
        macro_ap50 = float(np.mean([c["ap_50"] for c in present_classes])) if present_classes else 0.0
        macro_ap50_95 = float(np.mean([c["ap_50_95"] for c in present_classes])) if present_classes else 0.0
        macro_prec50 = float(np.mean([c["precision_50"] for c in present_classes])) if present_classes else 0.0
        macro_rec50 = float(np.mean([c["recall_50"] for c in present_classes])) if present_classes else 0.0

        # VRU specific metrics
        vru_present = [c for c in present_classes if c["is_vru"]]
        
        # Micro VRU (pooled TP, FP, FN)
        vru_total_tp = sum(c["tp_50"] for c in vru_present)
        vru_total_fp = sum(c["fp_50"] for c in vru_present)
        vru_total_fn = sum(c["fn_50"] for c in vru_present)
        vru_total_gt = sum(c["gt_count"] for c in vru_present)
        vru_total_preds = sum(c["pred_operational_count"] for c in vru_present)
        
        vru_micro_prec = (vru_total_tp / vru_total_preds) if vru_total_preds > 0 else 0.0
        vru_micro_rec = (vru_total_tp / vru_total_gt) if vru_total_gt > 0 else 0.0
        vru_micro_f1 = (2 * vru_micro_prec * vru_micro_rec / (vru_micro_prec + vru_micro_rec)) if (vru_micro_prec + vru_micro_rec) > 0 else 0.0

        # Macro VRU
        vru_macro_ap50 = float(np.mean([c["ap_50"] for c in vru_present])) if vru_present else 0.0
        vru_macro_ap50_95 = float(np.mean([c["ap_50_95"] for c in vru_present])) if vru_present else 0.0
        vru_macro_prec = float(np.mean([c["precision_50"] for c in vru_present])) if vru_present else 0.0
        vru_macro_rec = float(np.mean([c["recall_50"] for c in vru_present])) if vru_present else 0.0

        return {
            "config": {
                "operational_confidence_threshold": self.operational_conf,
                "operational_iou_threshold": self.operational_iou,
                "iou_thresholds": self.iou_thresholds,
                "evaluation_method": "COCO 101-point interpolated AP with explicit class alignment",
            },
            "summary": {
                "evaluated_images_count": len(self.ground_truths),
                "total_ground_truth_boxes": sum(gt_counts.values()),
                "total_predicted_boxes": len(self.predictions),
                "classes_present_count": len(present_classes),
                "classes_absent_count": len(ALL_CLASS_NAMES) - len(present_classes),
                "overall_mAP_50": round(macro_ap50, 4),
                "overall_mAP_50_95": round(macro_ap50_95, 4),
                "overall_macro_precision_50": round(macro_prec50, 4),
                "overall_macro_recall_50": round(macro_rec50, 4),
            },
            "vru_aggregates": {
                "vru_classes_evaluated": [c["name"] for c in vru_present],
                "vru_ground_truth_total": vru_total_gt,
                "vru_predictions_operational_total": vru_total_preds,
                "vru_tp_50": vru_total_tp,
                "vru_fp_50": vru_total_fp,
                "vru_fn_50": vru_total_fn,
                "vru_micro_precision_50": round(vru_micro_prec, 4),
                "vru_micro_recall_50": round(vru_micro_rec, 4),
                "vru_micro_f1_50": round(vru_micro_f1, 4),
                "vru_macro_mAP_50": round(vru_macro_ap50, 4),
                "vru_macro_mAP_50_95": round(vru_macro_ap50_95, 4),
                "vru_macro_precision_50": round(vru_macro_prec, 4),
                "vru_macro_recall_50": round(vru_macro_rec, 4),
            },
            "per_class_metrics": per_class_results,
        }
