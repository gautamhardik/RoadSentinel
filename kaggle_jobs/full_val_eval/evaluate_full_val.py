import os
import sys
import subprocess

# Ensure ultralytics is installed in the Kaggle environment
try:
    import ultralytics
except ImportError:
    print("Installing ultralytics package in Kaggle environment...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
    import ultralytics

import time
import json
import platform
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# ============================================================================
# Class Configuration & Mapping
# ============================================================================

# Kaggle YOLO Dataset Ground Truth label classes (val/labels/*.txt)
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

# Pretrained Checkpoint classes (weights/bdd100k_yolo11s.pt / Scalabel hierarchy)
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

# Explicit Checkpoint Class ID -> Ground Truth Class ID
CHECKPOINT_TO_GT_ID: Dict[int, int] = {
    0: 0,  # person       -> person
    1: 1,  # rider        -> rider
    2: 2,  # car          -> car
    3: 4,  # truck (ckpt) -> truck (gt 4)
    4: 3,  # bus (ckpt)   -> bus (gt 3)
    5: 9,  # train (ckpt) -> train (gt 9)
    6: 6,  # motor        -> motor
    7: 5,  # bike (ckpt)  -> bike (gt 5)
    8: 7,  # t-light(ckpt)-> t-light (gt 7)
    9: 8,  # t-sign (ckpt)-> t-sign (gt 8)
}

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

VRU_CLASS_NAMES: List[str] = ["person", "rider", "bike", "motor"]

IOU_THRESHOLDS = [round(x, 2) for x in np.linspace(0.50, 0.95, 10)]


# ============================================================================
# Vectorized IoU & AP Calculation
# ============================================================================

def compute_box_iou_vec(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Computes 1-to-N IoU between box [4] and boxes [N, 4].
    Format: [x1, y1, x2, y2].
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)

    xi1 = np.maximum(box[0], boxes[:, 0])
    yi1 = np.maximum(box[1], boxes[:, 1])
    xi2 = np.minimum(box[2], boxes[:, 2])
    yi2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, xi2 - xi1) * np.maximum(0.0, yi2 - yi1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter

    return np.where(union > 0, inter / union, 0.0)


def compute_ap_coco(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Computes Average Precision using standard 101-point COCO interpolation:
    p_interp(r) = max_{r' >= r} p(r') evaluated at r in [0.00, 0.01, ..., 1.00].
    """
    if len(recalls) == 0:
        return 0.0

    recall_thresholds = np.linspace(0.0, 1.0, 101)
    interpolated_precisions = np.zeros(101)

    for i, r in enumerate(recall_thresholds):
        matching = precisions[recalls >= r]
        if len(matching) > 0:
            interpolated_precisions[i] = np.max(matching)
        else:
            interpolated_precisions[i] = 0.0

    return float(np.mean(interpolated_precisions))


# ============================================================================
# Dataset and Checkpoint Auto-Discovery
# ============================================================================

def discover_paths():
    input_root = Path("/kaggle/input")
    print(f"Scanning {input_root} for dataset and weights...")

    # 1. Locate checkpoint
    ckpt_path = None
    for p in input_root.rglob("bdd100k_yolo11s.pt"):
        if p.is_file():
            ckpt_path = p
            break
    if ckpt_path is None:
        raise FileNotFoundError("Could not find bdd100k_yolo11s.pt in /kaggle/input")
    print(f"  Found Checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # 2. Locate validation split
    val_img_dir = None
    val_lbl_dir = None
    for p in input_root.rglob("images"):
        if p.parent.name == "val" and p.is_dir():
            val_img_dir = p
            candidate_lbl = p.parent / "labels"
            if candidate_lbl.exists():
                val_lbl_dir = candidate_lbl
            break

    if val_img_dir is None or val_lbl_dir is None:
        # Fallback search for any val images
        for p in input_root.rglob("*val*"):
            if (p / "images").exists() and (p / "labels").exists():
                val_img_dir = p / "images"
                val_lbl_dir = p / "labels"
                break

    if val_img_dir is None or val_lbl_dir is None:
        raise FileNotFoundError("Could not find validation images/labels in /kaggle/input")

    print(f"  Found Validation Images: {val_img_dir}")
    print(f"  Found Validation Labels: {val_lbl_dir}")
    return ckpt_path, val_img_dir, val_lbl_dir


# ============================================================================
# Main Execution Pipeline
# ============================================================================

def main():
    print("=" * 75)
    print("ROADSENTINEL — FULL BDD100K VALIDATION BASELINE EVALUATION (KAGGLE GPU)")
    print("=" * 75)
    start_time_all = time.time()

    # 1. Hardware Profiling
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)"
    print(f"Compute Device : {device} ({gpu_name})")
    print(f"PyTorch Version: {torch.__version__} | CUDA: {torch.version.cuda}")
    print(f"Platform       : {platform.platform()} ({platform.processor()})")

    # 2. Paths Discovery
    ckpt_path, val_img_dir, val_lbl_dir = discover_paths()

    # Output directories
    out_dir = Path("/kaggle/working")
    failures_dir = out_dir / "failure_cases"
    failures_dir.mkdir(parents=True, exist_ok=True)

    # 3. Model Loading & Provenance
    t_load_0 = time.perf_counter()
    model = YOLO(str(ckpt_path))
    model.to(device)
    t_load_1 = time.perf_counter()
    load_time_ms = (t_load_1 - t_load_0) * 1000
    print(f"Model loaded onto {device} in {load_time_ms:.1f} ms")

    # Record checkpoint training arguments & metadata
    raw_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    provenance = {
        "weights_file": ckpt_path.name,
        "date_trained": raw_ckpt.get("date", "Unknown"),
        "ultralytics_version": raw_ckpt.get("version", "Unknown"),
        "license": raw_ckpt.get("license", "Unknown"),
        "train_imgsz": raw_ckpt.get("train_args", {}).get("imgsz", 960),
        "train_epochs": raw_ckpt.get("train_args", {}).get("epochs", 50),
        "train_optimizer": raw_ckpt.get("train_args", {}).get("optimizer", "SGD"),
        "model_architecture": raw_ckpt.get("train_args", {}).get("model", "yolo11s.pt"),
    }
    del raw_ckpt  # Free memory
    print(f"Checkpoint Provenance: Trained {provenance['date_trained']} with {provenance['model_architecture']} at imgsz={provenance['train_imgsz']}")

    # 4. Enumerate All 10,000 Validation Images
    image_files = sorted([f for f in os.listdir(val_img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    total_images_count = len(image_files)
    print(f"\nDiscovered {total_images_count} validation images to evaluate.")
    if total_images_count < 9500:
        print(f"WARNING: Expected ~10,000 validation images, but found {total_images_count}.")

    # 5. GPU Warm-up
    print("Executing 10 GPU warm-up cycles (excluded from inference benchmark)...")
    dummy_path = val_img_dir / image_files[0]
    dummy_img = cv2.imread(str(dummy_path))
    for _ in range(10):
        _ = model.predict(source=dummy_img, imgsz=640, conf=0.25, device=device, verbose=False)
    torch.cuda.synchronize()
    print("GPU warm-up complete.")

    # 6. Load Ground Truth Annotations
    print("\nLoading ground truth annotations from label files...")
    # gt_by_class_img[class_name][img_name] = np.ndarray of shape [N, 4]
    gt_boxes_raw = defaultdict(lambda: defaultdict(list))
    total_gt_boxes_count = 0

    for img_name in image_files:
        lbl_name = Path(img_name).stem + ".txt"
        lbl_path = val_lbl_dir / lbl_name
        if not lbl_path.exists():
            continue

        # Standard BDD100K 720x1280 image shape
        img_w, img_h = 1280.0, 720.0
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                if cid not in GT_CLASSES:
                    continue
                cname = GT_CLASSES[cid]
                xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                x1 = (xc - bw / 2.0) * img_w
                y1 = (yc - bh / 2.0) * img_h
                x2 = (xc + bw / 2.0) * img_w
                y2 = (yc + bh / 2.0) * img_h
                gt_boxes_raw[cname][img_name].append([x1, y1, x2, y2])
                total_gt_boxes_count += 1

    # Convert to NumPy arrays for vectorized IoU
    gt_by_class_img: Dict[str, Dict[str, np.ndarray]] = {c: {} for c in ALL_CLASS_NAMES}
    gt_counts: Dict[str, int] = {c: 0 for c in ALL_CLASS_NAMES}

    for cname in ALL_CLASS_NAMES:
        for img_name, boxes in gt_boxes_raw[cname].items():
            arr = np.array(boxes, dtype=np.float32)
            gt_by_class_img[cname][img_name] = arr
            gt_counts[cname] += len(arr)

    print(f"Loaded {total_gt_boxes_count} total ground truth boxes across {len(image_files)} images.")

    # 7. Batched Timed Inference
    print(f"\nStarting GPU inference on {total_images_count} images (Batch size = 32, conf = 0.01)...")
    batch_size = 32
    all_preds_by_class = defaultdict(list)
    failure_cases = []

    t_infer_start = time.perf_counter()
    num_processed = 0

    for i in range(0, total_images_count, batch_size):
        chunk_names = image_files[i : i + batch_size]
        chunk_paths = [str(val_img_dir / fname) for fname in chunk_names]

        # Batch prediction on GPU
        results = model.predict(
            source=chunk_paths,
            imgsz=640,
            conf=0.01,  # Raw low threshold to capture full precision-recall curve
            device=device,
            verbose=False,
        )

        for img_name, res in zip(chunk_names, results):
            num_processed += 1
            op_preds_img = []
            
            for det in res.boxes:
                ckpt_cid = int(det.cls.item())
                conf = float(det.conf.item())
                box = det.xyxy[0].cpu().numpy()
                
                # Semantic class resolution
                cname = CHECKPOINT_CLASSES[ckpt_cid]
                all_preds_by_class[cname].append({
                    "img_name": img_name,
                    "conf": conf,
                    "box": box,
                })
                
                if conf >= 0.25:
                    op_preds_img.append({"name": cname, "conf": conf, "box": box})

            # Log top 15 failure cases for VRUs at operational threshold
            if len(failure_cases) < 15:
                # Check for missed GT VRUs
                for vru_cname in VRU_CLASS_NAMES:
                    gt_arr = gt_by_class_img[vru_cname].get(img_name, None)
                    if gt_arr is not None and len(gt_arr) > 0:
                        vru_preds = [p for p in op_preds_img if p["name"] == vru_cname]
                        if len(vru_preds) == 0:  # Complete miss
                            img_vis = cv2.imread(str(val_img_dir / img_name))
                            for gb in gt_arr:
                                cv2.rectangle(img_vis, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])), (0, 0, 255), 2)
                                cv2.putText(img_vis, f"MISSED {vru_cname}", (int(gb[0]), max(15, int(gb[1]) - 4)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                            fail_fname = f"vru_miss_{len(failure_cases)+1:02d}_{img_name}"
                            cv2.imwrite(str(failures_dir / fail_fname), img_vis)
                            failure_cases.append({
                                "image": img_name,
                                "missed_class": vru_cname,
                                "missed_count": len(gt_arr),
                                "artifact": fail_fname,
                            })
                            break

        if num_processed % 1000 == 0 or num_processed == total_images_count:
            elapsed = time.perf_counter() - t_infer_start
            fps = num_processed / elapsed if elapsed > 0 else 0
            print(f"  [{num_processed:05d}/{total_images_count}] Processed | Elapsed: {elapsed:.1f}s | Speed: {fps:.1f} FPS")

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_infer_end = time.perf_counter()
    total_infer_time_s = t_infer_end - t_infer_start
    mean_lat_ms = (total_infer_time_s / total_images_count) * 1000
    throughput_fps = total_images_count / total_infer_time_s

    print(f"\nInference complete! Processed {total_images_count} images in {total_infer_time_s:.2f} s ({throughput_fps:.1f} FPS, {mean_lat_ms:.2f} ms/image).")

    # 8. Evaluation Metrics Calculation
    print("\nComputing class-aligned detection metrics and 101-point COCO AP...")
    t_eval_start = time.perf_counter()

    per_class_results = []
    
    for cname in ALL_CLASS_NAMES:
        n_gt = gt_counts[cname]
        preds = all_preds_by_class[cname]
        n_preds_total = len(preds)

        # Operational metrics at fixed confidence tau = 0.25 and IoU = 0.50
        op_preds = [p for p in preds if p["conf"] >= 0.25]
        n_op_preds = len(op_preds)

        # Group operational predictions by image
        op_preds_by_img = defaultdict(list)
        for p in op_preds:
            op_preds_by_img[p["img_name"]].append(p)

        op_tp = 0
        op_fp = 0

        for img_name, img_op_preds in op_preds_by_img.items():
            gts = gt_by_class_img[cname].get(img_name, None)
            if gts is None or len(gts) == 0:
                op_fp += len(img_op_preds)
                continue

            matched_gt = np.zeros(len(gts), dtype=bool)
            img_op_preds.sort(key=lambda x: x["conf"], reverse=True)

            for p in img_op_preds:
                ious = compute_box_iou_vec(p["box"], gts)
                best_idx = np.argmax(ious)
                best_iou = ious[best_idx]

                if best_iou >= 0.50 and not matched_gt[best_idx]:
                    op_tp += 1
                    matched_gt[best_idx] = True
                else:
                    op_fp += 1

        # Operational false negatives
        op_fn = n_gt - op_tp
        op_prec = (op_tp / n_op_preds) if n_op_preds > 0 else 0.0
        op_rec = (op_tp / n_gt) if n_gt > 0 else 0.0
        op_f1 = (2 * op_prec * op_rec / (op_prec + op_rec)) if (op_prec + op_rec) > 0 else 0.0

        # Full AP Calculation across IoU thresholds [0.50 : 0.05 : 0.95]
        if n_gt == 0:
            is_absent = True
            ap_50 = None
            ap_50_95 = None
        else:
            is_absent = False
            # Sort ALL predictions descending by confidence
            preds.sort(key=lambda x: x["conf"], reverse=True)
            ap_by_iou = {}

            for iou_thr in IOU_THRESHOLDS:
                # Track matched GTs per image
                matched_gts_per_img = {
                    img_name: np.zeros(len(gts), dtype=bool)
                    for img_name, gts in gt_by_class_img[cname].items()
                }

                tp_vec = np.zeros(n_preds_total, dtype=np.float32)
                fp_vec = np.zeros(n_preds_total, dtype=np.float32)

                for idx, p in enumerate(preds):
                    img_name = p["img_name"]
                    gts = gt_by_class_img[cname].get(img_name, None)
                    if gts is None or len(gts) == 0:
                        fp_vec[idx] = 1.0
                        continue

                    ious = compute_box_iou_vec(p["box"], gts)
                    best_idx = np.argmax(ious)
                    best_iou = ious[best_idx]

                    if best_iou >= iou_thr and not matched_gts_per_img[img_name][best_idx]:
                        tp_vec[idx] = 1.0
                        matched_gts_per_img[img_name][best_idx] = True
                    else:
                        fp_vec[idx] = 1.0

                cum_tp = np.cumsum(tp_vec)
                cum_fp = np.cumsum(fp_vec)
                rec = cum_tp / n_gt
                prec = cum_tp / (cum_tp + cum_fp)

                ap_by_iou[iou_thr] = compute_ap_coco(rec, prec)

            ap_50 = ap_by_iou.get(0.50, 0.0)
            ap_50_95 = float(np.mean([ap_by_iou[t] for t in IOU_THRESHOLDS]))

        per_class_results.append({
            "name": cname,
            "is_vru": cname in VRU_CLASS_NAMES,
            "is_absent_in_gt": is_absent,
            "gt_count": n_gt,
            "pred_total_count": n_preds_total,
            "pred_operational_count": n_op_preds,
            "tp_50": op_tp,
            "fp_50": op_fp,
            "fn_50": op_fn,
            "precision_50": round(op_prec, 4),
            "recall_50": round(op_rec, 4),
            "f1_50": round(op_f1, 4),
            "ap_50": round(ap_50, 4) if ap_50 is not None else None,
            "ap_50_95": round(ap_50_95, 4) if ap_50_95 is not None else None,
        })

    # Summary aggregations
    present_classes = [c for c in per_class_results if not c["is_absent_in_gt"]]
    macro_ap50 = float(np.mean([c["ap_50"] for c in present_classes])) if present_classes else 0.0
    macro_ap50_95 = float(np.mean([c["ap_50_95"] for c in present_classes])) if present_classes else 0.0
    macro_prec50 = float(np.mean([c["precision_50"] for c in present_classes])) if present_classes else 0.0
    macro_rec50 = float(np.mean([c["recall_50"] for c in present_classes])) if present_classes else 0.0

    # VRU aggregates
    vru_present = [c for c in present_classes if c["is_vru"]]
    vru_tp = sum(c["tp_50"] for c in vru_present)
    vru_fp = sum(c["fp_50"] for c in vru_present)
    vru_fn = sum(c["fn_50"] for c in vru_present)
    vru_gt = sum(c["gt_count"] for c in vru_present)
    vru_op_preds = sum(c["pred_operational_count"] for c in vru_present)

    vru_micro_p = (vru_tp / vru_op_preds) if vru_op_preds > 0 else 0.0
    vru_micro_r = (vru_tp / vru_gt) if vru_gt > 0 else 0.0
    vru_micro_f1 = (2 * vru_micro_p * vru_micro_r / (vru_micro_p + vru_micro_r)) if (vru_micro_p + vru_micro_r) > 0 else 0.0

    vru_macro_ap50 = float(np.mean([c["ap_50"] for c in vru_present])) if vru_present else 0.0
    vru_macro_ap50_95 = float(np.mean([c["ap_50_95"] for c in vru_present])) if vru_present else 0.0
    vru_macro_p = float(np.mean([c["precision_50"] for c in vru_present])) if vru_present else 0.0
    vru_macro_r = float(np.mean([c["recall_50"] for c in vru_present])) if vru_present else 0.0

    eval_duration_s = time.perf_counter() - t_eval_start
    print(f"Metrics evaluation finished in {eval_duration_s:.2f} s.")

    # Latency summary
    latency_summary = {
        "device": device,
        "gpu_model": gpu_name,
        "input_resolution": "640x640",
        "batch_size": batch_size,
        "num_timed_images": total_images_count,
        "total_inference_time_s": round(total_infer_time_s, 2),
        "mean_latency_ms": round(mean_lat_ms, 2),
        "throughput_fps": round(throughput_fps, 2),
    }

    # Final Payload
    payload = {
        "evaluation_name": "RoadSentinel Full BDD100K Validation Baseline",
        "benchmark_type": "Full 10,000-Image Representative Validation Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_environment": {
            "device": device,
            "gpu_name": gpu_name,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "model_load_time_ms": round(load_time_ms, 2),
        },
        "provenance": provenance,
        "latency_benchmark": latency_summary,
        "evaluation_config": {
            "operational_confidence_threshold": 0.25,
            "operational_iou_threshold": 0.50,
            "iou_thresholds": IOU_THRESHOLDS,
            "ap_interpolation": "COCO 101-point interpolated AP",
            "class_alignment": "Explicit semantic mapping from Scalabel to 2018 YOLO labels",
        },
        "summary": {
            "evaluated_images_count": total_images_count,
            "total_ground_truth_boxes": total_gt_boxes_count,
            "classes_present_count": len(present_classes),
            "classes_absent_count": len(ALL_CLASS_NAMES) - len(present_classes),
            "overall_mAP_50": round(macro_ap50, 4),
            "overall_mAP_50_95": round(macro_ap50_95, 4),
            "overall_macro_precision_50": round(macro_prec50, 4),
            "overall_macro_recall_50": round(macro_rec50, 4),
        },
        "vru_aggregates": {
            "vru_classes": [c["name"] for c in vru_present],
            "vru_ground_truth_total": vru_gt,
            "vru_operational_predictions_total": vru_op_preds,
            "vru_tp_50": vru_tp,
            "vru_fp_50": vru_fp,
            "vru_fn_50": vru_fn,
            "vru_micro_precision_50": round(vru_micro_p, 4),
            "vru_micro_recall_50": round(vru_micro_r, 4),
            "vru_micro_f1_50": round(vru_micro_f1, 4),
            "vru_macro_mAP_50": round(vru_macro_ap50, 4),
            "vru_macro_mAP_50_95": round(vru_macro_ap50_95, 4),
            "vru_macro_precision_50": round(vru_macro_p, 4),
            "vru_macro_recall_50": round(vru_macro_r, 4),
        },
        "per_class_metrics": per_class_results,
        "failure_cases": failure_cases,
    }

    # Save JSON metrics
    json_path = out_dir / "full_val_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved metrics JSON to: {json_path}")

    # Print Formatted Table
    print("\n" + "=" * 95)
    print("ROADSENTINEL — FULL 10,000-IMAGE BDD100K VALIDATION RESULTS")
    print("=" * 95)
    print(f"Overall mAP@50     : {macro_ap50:.4f} ({macro_ap50*100:.2f}%)")
    print(f"Overall mAP@50-95  : {macro_ap50_95:.4f} ({macro_ap50_95*100:.2f}%)")
    print(f"Overall Macro P@50 : {macro_prec50:.4f}")
    print(f"Overall Macro R@50 : {macro_rec50:.4f}")
    print("-" * 95)
    print("VRU PERFORMANCE (person, rider, bike, motor):")
    print(f"VRU Macro mAP@50   : {vru_macro_ap50:.4f} ({vru_macro_ap50*100:.2f}%)")
    print(f"VRU Macro mAP@50-95: {vru_macro_ap50_95:.4f} ({vru_macro_ap50_95*100:.2f}%)")
    print(f"VRU Micro P@50     : {vru_micro_p:.4f} | VRU Micro R@50: {vru_micro_r:.4f} | VRU Micro F1: {vru_micro_f1:.4f}")
    print("-" * 95)
    print(f"GPU Latency        : {mean_lat_ms:.2f} ms/image ({throughput_fps:.1f} FPS) on {gpu_name}")
    print("=" * 95)
    print(f"{'Class Name':<15} {'GT':<7} {'Pred(Op)':<10} {'TP@50':<8} {'FP@50':<8} {'FN@50':<8} {'P@50':<8} {'R@50':<8} {'AP@50':<8} {'AP@50-95':<8}")
    print("-" * 95)
    for m in per_class_results:
        vru_tag = "* " if m["is_vru"] else "  "
        cname = vru_tag + m["name"]
        ap50_s = f"{m['ap_50']:.4f}" if m["ap_50"] is not None else "N/A"
        ap5095_s = f"{m['ap_50_95']:.4f}" if m["ap_50_95"] is not None else "N/A"
        print(f"{cname:<15} {m['gt_count']:<7} {m['pred_operational_count']:<10} {m['tp_50']:<8} {m['fp_50']:<8} {m['fn_50']:<8} {m['precision_50']:<8.4f} {m['recall_50']:<8.4f} {ap50_s:<8} {ap5095_s:<8}")
    print("=" * 95)

    total_duration = time.time() - start_time_all
    print(f"Total job duration: {total_duration / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
