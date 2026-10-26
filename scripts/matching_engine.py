import numpy as np
import time

def evaluate_partition_original(predictions_by_img: dict, gt_by_img: dict, image_subset: set, class_names, vru_names):
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in class_names}
    size_stats = {
        "Small (<32²)": {"tp": 0, "fn": 0, "total": 0},
        "Medium [32²-96²)": {"tp": 0, "fn": 0, "total": 0},
        "Large (≥96²)": {"tp": 0, "fn": 0, "total": 0},
    }
    vru_size_stats = {
        "Small (<32²)": {"tp": 0, "fn": 0, "total": 0},
        "Medium [32²-96²)": {"tp": 0, "fn": 0, "total": 0},
        "Large (≥96²)": {"tp": 0, "fn": 0, "total": 0},
    }

    total_predictions = 0
    total_gt_objects = 0

    def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
        xa1, ya1, xa2, ya2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
        xb1, yb1, xb2, yb2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

        inter_x1 = np.maximum(xa1[:, None], xb1[None, :])
        inter_y1 = np.maximum(ya1[:, None], yb1[None, :])
        inter_x2 = np.minimum(xa2[:, None], xb2[None, :])
        inter_y2 = np.minimum(ya2[:, None], yb2[None, :])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        union_area = area_a[:, None] + area_b[None, :] - inter_area
        return np.where(union_area > 0, inter_area / union_area, 0.0)

    for img_name in image_subset:
        gt_objs = gt_by_img.get(img_name, [])
        pred_objs = predictions_by_img.get(img_name, [])

        total_gt_objects += len(gt_objs)
        total_predictions += len(pred_objs)

        # Match per class
        for cname in class_names:
            c_gt = [g for g in gt_objs if g["class_name"] == cname]
            c_pred = [p for p in pred_objs if p["class_name"] == cname]

            if not c_gt and not c_pred:
                continue

            if not c_gt and c_pred:
                class_stats[cname]["fp"] += len(c_pred)
                continue

            if c_gt and not c_pred:
                class_stats[cname]["fn"] += len(c_gt)
                for g in c_gt:
                    sz = g["size_group"]
                    size_stats[sz]["fn"] += 1
                    size_stats[sz]["total"] += 1
                    if g["is_vru"]:
                        vru_size_stats[sz]["fn"] += 1
                        vru_size_stats[sz]["total"] += 1
                continue

            # Both present: sort predictions by confidence descending
            c_pred = sorted(c_pred, key=lambda x: x["conf"], reverse=True)
            pred_boxes = np.array([p["box"] for p in c_pred], dtype=np.float32)
            gt_boxes = np.array([g["box"] for g in c_gt], dtype=np.float32)

            iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)
            gt_matched = set()

            for p_idx in range(len(c_pred)):
                best_gt_idx = -1
                best_iou = 0.50  # matching threshold
                for g_idx in range(len(c_gt)):
                    if g_idx in gt_matched:
                        continue
                    if iou_mat[p_idx, g_idx] >= best_iou:
                        best_iou = iou_mat[p_idx, g_idx]
                        best_gt_idx = g_idx

                if best_gt_idx >= 0:
                    gt_matched.add(best_gt_idx)
                    class_stats[cname]["tp"] += 1
                    sz = c_gt[best_gt_idx]["size_group"]
                    size_stats[sz]["tp"] += 1
                    size_stats[sz]["total"] += 1
                    if c_gt[best_gt_idx]["is_vru"]:
                        vru_size_stats[sz]["tp"] += 1
                        vru_size_stats[sz]["total"] += 1
                else:
                    class_stats[cname]["fp"] += 1

            for g_idx in range(len(c_gt)):
                if g_idx not in gt_matched:
                    class_stats[cname]["fn"] += 1
                    sz = c_gt[g_idx]["size_group"]
                    size_stats[sz]["fn"] += 1
                    size_stats[sz]["total"] += 1
                    if c_gt[g_idx]["is_vru"]:
                        vru_size_stats[sz]["fn"] += 1
                        vru_size_stats[sz]["total"] += 1

    vru_tp = sum(class_stats[c]["tp"] for c in vru_names)
    vru_fp = sum(class_stats[c]["fp"] for c in vru_names)
    vru_fn = sum(class_stats[c]["fn"] for c in vru_names)
    vru_prec = vru_tp / (vru_tp + vru_fp) if (vru_tp + vru_fp) > 0 else 0.0
    vru_rec = vru_tp / (vru_tp + vru_fn) if (vru_tp + vru_fn) > 0 else 0.0
    vru_f1 = 2 * vru_prec * vru_rec / (vru_prec + vru_rec) if (vru_prec + vru_rec) > 0 else 0.0

    all_tp = sum(class_stats[c]["tp"] for c in class_names)
    all_fp = sum(class_stats[c]["fp"] for c in class_names)
    all_fn = sum(class_stats[c]["fn"] for c in class_names)
    all_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    all_rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    all_f1 = 2 * all_prec * all_rec / (all_prec + all_rec) if (all_prec + all_rec) > 0 else 0.0

    small_vru_tot = vru_size_stats["Small (<32²)"]["total"]
    small_vru_tp = vru_size_stats["Small (<32²)"]["tp"]
    small_vru_recall = small_vru_tp / small_vru_tot if small_vru_tot > 0 else 0.0

    return {
        "image_count": len(image_subset),
        "total_gt_objects": total_gt_objects,
        "total_predictions": total_predictions,
        "vru_metrics": {
            "tp": vru_tp, "fp": vru_fp, "fn": vru_fn,
            "precision": round(vru_prec * 100.0, 2),
            "recall": round(vru_rec * 100.0, 2),
            "f1_score": round(vru_f1, 4),
            "small_vru_ground_truth": small_vru_tot,
            "small_vru_true_positives": small_vru_tp,
            "small_vru_recall": round(small_vru_recall * 100.0, 2),
        },
        "all_class_metrics": {
            "tp": all_tp, "fp": all_fp, "fn": all_fn,
            "precision": round(all_prec * 100.0, 2),
            "recall": round(all_rec * 100.0, 2),
            "f1_score": round(all_f1, 4),
        },
        "per_class": {
            c: {
                "tp": class_stats[c]["tp"], "fp": class_stats[c]["fp"], "fn": class_stats[c]["fn"],
                "precision": round(class_stats[c]["tp"] / max(class_stats[c]["tp"] + class_stats[c]["fp"], 1) * 100.0, 2),
                "recall": round(class_stats[c]["tp"] / max(class_stats[c]["tp"] + class_stats[c]["fn"], 1) * 100.0, 2),
            } for c in class_names
        },
        "vru_by_size": {
            sz: {
                "tp": vru_size_stats[sz]["tp"], "fn": vru_size_stats[sz]["fn"], "total": vru_size_stats[sz]["total"],
                "recall": round(vru_size_stats[sz]["tp"] / max(vru_size_stats[sz]["total"], 1) * 100.0, 2),
            } for sz in ["Small (<32²)", "Medium [32²-96²)", "Large (≥96²)"]
        }
    }


def match_single_image(pred_objs: list, gt_objs: list, class_names: list, vru_names: set) -> dict:
    """Matches a single image once and returns raw count summaries."""
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in class_names}
    vru_size_stats = {
        "Small (<32²)": {"tp": 0, "fn": 0, "total": 0},
        "Medium [32²-96²)": {"tp": 0, "fn": 0, "total": 0},
        "Large (≥96²)": {"tp": 0, "fn": 0, "total": 0},
    }

    n_gt = len(gt_objs)
    n_pred = len(pred_objs)

    for cname in class_names:
        c_gt = [g for g in gt_objs if g["class_name"] == cname]
        c_pred = [p for p in pred_objs if p["class_name"] == cname]

        if not c_gt and not c_pred:
            continue

        if not c_gt and c_pred:
            class_stats[cname]["fp"] += len(c_pred)
            continue

        if c_gt and not c_pred:
            class_stats[cname]["fn"] += len(c_gt)
            for g in c_gt:
                sz = g["size_group"]
                if g["is_vru"]:
                    vru_size_stats[sz]["fn"] += 1
                    vru_size_stats[sz]["total"] += 1
            continue

        # Sort predictions descending by confidence
        c_pred.sort(key=lambda x: x["conf"], reverse=True)
        pred_boxes = np.array([p["box"] for p in c_pred], dtype=np.float32)
        gt_boxes = np.array([g["box"] for g in c_gt], dtype=np.float32)

        # Vectorized IoU
        xa1, ya1, xa2, ya2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        xb1, yb1, xb2, yb2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

        inter_x1 = np.maximum(xa1[:, None], xb1[None, :])
        inter_y1 = np.maximum(ya1[:, None], yb1[None, :])
        inter_x2 = np.minimum(xa2[:, None], xb2[None, :])
        inter_y2 = np.minimum(ya2[:, None], yb2[None, :])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        union_area = area_a[:, None] + area_b[None, :] - inter_area
        iou_mat = np.where(union_area > 0, inter_area / union_area, 0.0)

        gt_matched = set()
        for p_idx in range(len(c_pred)):
            best_gt_idx = -1
            best_iou = 0.50
            for g_idx in range(len(c_gt)):
                if g_idx in gt_matched:
                    continue
                if iou_mat[p_idx, g_idx] >= best_iou:
                    best_iou = iou_mat[p_idx, g_idx]
                    best_gt_idx = g_idx

            if best_gt_idx >= 0:
                gt_matched.add(best_gt_idx)
                class_stats[cname]["tp"] += 1
                sz = c_gt[best_gt_idx]["size_group"]
                if c_gt[best_gt_idx]["is_vru"]:
                    vru_size_stats[sz]["tp"] += 1
                    vru_size_stats[sz]["total"] += 1
            else:
                class_stats[cname]["fp"] += 1

        for g_idx in range(len(c_gt)):
            if g_idx not in gt_matched:
                class_stats[cname]["fn"] += 1
                sz = c_gt[g_idx]["size_group"]
                if c_gt[g_idx]["is_vru"]:
                    vru_size_stats[sz]["fn"] += 1
                    vru_size_stats[sz]["total"] += 1

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "class_stats": class_stats,
        "vru_size_stats": vru_size_stats,
    }


def aggregate_partition_from_image_stats(image_stats: dict, image_subset: set, class_names: list, vru_names: set) -> dict:
    """Aggregates pre-computed per-image stats for any partition in O(N_subset) time."""
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in class_names}
    vru_size_stats = {
        "Small (<32²)": {"tp": 0, "fn": 0, "total": 0},
        "Medium [32²-96²)": {"tp": 0, "fn": 0, "total": 0},
        "Large (≥96²)": {"tp": 0, "fn": 0, "total": 0},
    }
    total_gt = 0
    total_pred = 0

    for img_name in image_subset:
        st = image_stats.get(img_name)
        if not st:
            continue
        total_gt += st["n_gt"]
        total_pred += st["n_pred"]
        for c in class_names:
            class_stats[c]["tp"] += st["class_stats"][c]["tp"]
            class_stats[c]["fp"] += st["class_stats"][c]["fp"]
            class_stats[c]["fn"] += st["class_stats"][c]["fn"]
        for sz in ["Small (<32²)", "Medium [32²-96²)", "Large (≥96²)"]:
            vru_size_stats[sz]["tp"] += st["vru_size_stats"][sz]["tp"]
            vru_size_stats[sz]["fn"] += st["vru_size_stats"][sz]["fn"]
            vru_size_stats[sz]["total"] += st["vru_size_stats"][sz]["total"]

    vru_tp = sum(class_stats[c]["tp"] for c in vru_names)
    vru_fp = sum(class_stats[c]["fp"] for c in vru_names)
    vru_fn = sum(class_stats[c]["fn"] for c in vru_names)
    vru_prec = vru_tp / (vru_tp + vru_fp) if (vru_tp + vru_fp) > 0 else 0.0
    vru_rec = vru_tp / (vru_tp + vru_fn) if (vru_tp + vru_fn) > 0 else 0.0
    vru_f1 = 2 * vru_prec * vru_rec / (vru_prec + vru_rec) if (vru_prec + vru_rec) > 0 else 0.0

    all_tp = sum(class_stats[c]["tp"] for c in class_names)
    all_fp = sum(class_stats[c]["fp"] for c in class_names)
    all_fn = sum(class_stats[c]["fn"] for c in class_names)
    all_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    all_rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    all_f1 = 2 * all_prec * all_rec / (all_prec + all_rec) if (all_prec + all_rec) > 0 else 0.0

    small_vru_tot = vru_size_stats["Small (<32²)"]["total"]
    small_vru_tp = vru_size_stats["Small (<32²)"]["tp"]
    small_vru_recall = small_vru_tp / small_vru_tot if small_vru_tot > 0 else 0.0

    return {
        "image_count": len(image_subset),
        "total_gt_objects": total_gt,
        "total_predictions": total_pred,
        "vru_metrics": {
            "tp": vru_tp, "fp": vru_fp, "fn": vru_fn,
            "precision": round(vru_prec * 100.0, 2),
            "recall": round(vru_rec * 100.0, 2),
            "f1_score": round(vru_f1, 4),
            "small_vru_ground_truth": small_vru_tot,
            "small_vru_true_positives": small_vru_tp,
            "small_vru_recall": round(small_vru_recall * 100.0, 2),
        },
        "all_class_metrics": {
            "tp": all_tp, "fp": all_fp, "fn": all_fn,
            "precision": round(all_prec * 100.0, 2),
            "recall": round(all_rec * 100.0, 2),
            "f1_score": round(all_f1, 4),
        },
        "per_class": {
            c: {
                "tp": class_stats[c]["tp"], "fp": class_stats[c]["fp"], "fn": class_stats[c]["fn"],
                "precision": round(class_stats[c]["tp"] / max(class_stats[c]["tp"] + class_stats[c]["fp"], 1) * 100.0, 2),
                "recall": round(class_stats[c]["tp"] / max(class_stats[c]["tp"] + class_stats[c]["fn"], 1) * 100.0, 2),
            } for c in class_names
        },
        "vru_by_size": {
            sz: {
                "tp": vru_size_stats[sz]["tp"], "fn": vru_size_stats[sz]["fn"], "total": vru_size_stats[sz]["total"],
                "recall": round(vru_size_stats[sz]["tp"] / max(vru_size_stats[sz]["total"], 1) * 100.0, 2),
            } for sz in ["Small (<32²)", "Medium [32²-96²)", "Large (≥96²)"]
        }
    }
