"""
RoadSentinel Phase 2B.2 — Full-Validation Resolution Benchmark
Evaluates YOLO11s on all 10,000 BDD100K validation images across 640, 960, and 1280.
Measures:
  - COCO-style 101-point AP (mAP@50, mAP@50-95, VRU macro AP)
  - Operational metrics at tau=0.25, IoU=0.50 (Precision, Recall, F1, TP, FP, FN)
  - Size group analysis on original 1280x720 area: Small (<1024), Medium [1024, 9216), Large (>=9216)
  - Latency: Batched throughput (FPS) & amortized ms, plus isolated batch-1 latency (mean, std, p50, p95)
  - Peak GPU memory allocation (VRAM MB)
  - Representative visual examples (recovered VRUs and persistent misses)
"""

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
import hashlib
import platform
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================================
# Class Configuration & Mapping
# ============================================================================

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
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]

VRU_CLASS_NAMES: List[str] = ["person", "rider", "bike", "motor"]

IOU_THRESHOLDS = [round(x, 2) for x in np.linspace(0.50, 0.95, 10)]


# ============================================================================
# Vectorized IoU & Matching Utilities
# ============================================================================

def compute_box_iou_vec(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Computes 1-to-N IoU between box [4] and boxes [N, 4] in [x1, y1, x2, y2]."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)

    xi1 = np.maximum(box[0], boxes[:, 0])
    yi1 = np.maximum(box[1], boxes[:, 1])
    xi2 = np.minimum(box[2], boxes[:, 2])
    yi2 = np.minimum(box[3], boxes[:, 3])

    iw = np.maximum(0.0, xi2 - xi1)
    ih = np.maximum(0.0, yi2 - yi1)
    inter = iw * ih

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    union = np.maximum(union, 1e-8)
    return inter / union


def compute_coco_ap_for_class(
    preds: List[Dict[str, Any]],
    gt_by_img: Dict[str, np.ndarray],
    num_gt: int,
) -> Tuple[float, float, Dict[float, float]]:
    """
    Computes COCO-style 101-point interpolated AP across IoU thresholds [0.50:0.05:0.95].
    Predictions are sorted globally by confidence in descending order.
    """
    if num_gt == 0:
        return 0.0, 0.0, {t: 0.0 for t in IOU_THRESHOLDS}
    if not preds:
        return 0.0, 0.0, {t: 0.0 for t in IOU_THRESHOLDS}

    # Sort globally by confidence
    preds_sorted = sorted(preds, key=lambda x: x["conf"], reverse=True)
    n_preds = len(preds_sorted)

    ap_per_iou: Dict[float, float] = {}

    for iou_thresh in IOU_THRESHOLDS:
        matched_gt: Dict[str, set] = defaultdict(set)
        tp = np.zeros(n_preds, dtype=np.float32)
        fp = np.zeros(n_preds, dtype=np.float32)

        for idx, p in enumerate(preds_sorted):
            img_name = p["img_name"]
            gt_boxes = gt_by_img.get(img_name)
            if gt_boxes is None or len(gt_boxes) == 0:
                fp[idx] = 1.0
                continue

            ious = compute_box_iou_vec(p["box"], gt_boxes)
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])

            if best_iou >= iou_thresh and best_idx not in matched_gt[img_name]:
                tp[idx] = 1.0
                matched_gt[img_name].add(best_idx)
            else:
                fp[idx] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / float(num_gt)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

        # 101-point COCO interpolation
        rec_levels = np.linspace(0.0, 1.0, 101)
        prec_levels = np.zeros(101, dtype=np.float32)
        for r_idx, r in enumerate(rec_levels):
            prec_candidates = precisions[recalls >= r]
            if len(prec_candidates) > 0:
                prec_levels[r_idx] = np.max(prec_candidates)
            else:
                prec_levels[r_idx] = 0.0

        ap_per_iou[iou_thresh] = float(np.mean(prec_levels))

    ap_50 = ap_per_iou[0.50]
    ap_50_95 = float(np.mean([ap_per_iou[t] for t in IOU_THRESHOLDS]))
    return ap_50, ap_50_95, ap_per_iou


def evaluate_operational_point(
    preds: List[Dict[str, Any]],
    gt_by_img: Dict[str, np.ndarray],
    num_gt: int,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
) -> Tuple[int, int, int, float, float, float, Dict[str, set]]:
    """Evaluates TP, FP, FN, Precision, Recall, and F1 at fixed operational point."""
    filt_preds = [p for p in preds if p["conf"] >= conf_thresh]
    filt_preds.sort(key=lambda x: x["conf"], reverse=True)

    matched_gt: Dict[str, set] = defaultdict(set)
    tp = 0
    fp = 0

    for p in filt_preds:
        img_name = p["img_name"]
        gt_boxes = gt_by_img.get(img_name)
        if gt_boxes is None or len(gt_boxes) == 0:
            fp += 1
            continue

        ious = compute_box_iou_vec(p["box"], gt_boxes)
        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])

        if best_iou >= iou_thresh and best_idx not in matched_gt[img_name]:
            tp += 1
            matched_gt[img_name].add(best_idx)
        else:
            fp += 1

    fn = num_gt - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(num_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return tp, fp, fn, precision, recall, f1, matched_gt


# ============================================================================
# Main Execution Pipeline
# ============================================================================

def main():
    print("=" * 80)
    print("RoadSentinel Phase 2B.2 — Full-Validation Multi-Resolution Benchmark")
    print("Configurations: Control-640 (b32), Candidate-960 (b16), Candidate-1280 (b8)")
    print("=" * 80)

    # 1. Hardware & Environment Check
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu_name})")
    print(f"PyTorch Version: {torch.__version__} | CUDA: {torch.version.cuda}")
    print(f"Platform: {platform.platform()}")

    # 2. Dataset & Checkpoint Discovery
    out_dir = Path("/kaggle/working")
    out_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = out_dir / "visual_examples"
    visual_dir.mkdir(parents=True, exist_ok=True)

    # Fast direct dataset resolution
    candidate_ckpt_paths = [
        Path("/kaggle/input/datasets/hardikgautam25/roadsentinel-weights/bdd100k_yolo11s.pt"),
        Path("/kaggle/input/roadsentinel-weights/bdd100k_yolo11s.pt"),
        Path("weights/bdd100k_yolo11s.pt"),
    ]
    ckpt_path = None
    for p in candidate_ckpt_paths:
        if p.exists():
            ckpt_path = p
            break
    if ckpt_path is None:
        matches = list(Path("/kaggle/input").rglob("bdd100k_yolo11s.pt"))
        if matches:
            ckpt_path = matches[0]
        else:
            raise FileNotFoundError("Could not locate weights/bdd100k_yolo11s.pt")

    candidate_img_dirs = [
        Path("/kaggle/input/datasets/a7madmostafa/bdd100k-yolo/val/images"),
        Path("/kaggle/input/bdd100k-yolo/val/images"),
        Path("bdd100k_dataset/val/images"),
    ]
    val_img_dir = None
    for p in candidate_img_dirs:
        if p.exists():
            val_img_dir = p
            break
    if val_img_dir is None:
        matches = list(Path("/kaggle/input").rglob("val/images"))
        if matches:
            val_img_dir = matches[0]
        else:
            raise FileNotFoundError("Could not locate val/images directory")

    val_lbl_dir = val_img_dir.parent / "labels"
    print(f"Using Checkpoint: {ckpt_path}")
    print(f"Using Validation Images: {val_img_dir}")
    print(f"Using Validation Labels: {val_lbl_dir}")

    # Checkpoint SHA-256 and metadata audit
    ckpt_bytes = ckpt_path.read_bytes()
    ckpt_sha256 = hashlib.sha256(ckpt_bytes).hexdigest().upper()
    print(f"Checkpoint SHA-256: {ckpt_sha256}")

    raw_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    train_args = raw_ckpt.get("train_args", {})
    trained_imgsz = train_args.get("imgsz", 960) if isinstance(train_args, dict) else 960
    trained_epochs = train_args.get("epochs", 50) if isinstance(train_args, dict) else 50
    trained_batch = train_args.get("batch", 8) if isinstance(train_args, dict) else 8
    trained_opt = train_args.get("optimizer", "SGD") if isinstance(train_args, dict) else "SGD"
    trained_date = raw_ckpt.get("date", "Unknown")
    trained_version = raw_ckpt.get("version", "Unknown")
    del raw_ckpt

    print(f"Checkpoint Provenance Audit:")
    print(f"  Trained date: {trained_date}")
    print(f"  Ultralytics version: {trained_version}")
    print(f"  Native training resolution (imgsz): {trained_imgsz}")
    print(f"  Training epochs: {trained_epochs}, batch: {trained_batch}, optimizer: {trained_opt}")

    # 3. Enumerate and Verify 10,000 Validation Images
    image_files = sorted([f for f in os.listdir(val_img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    total_images_count = len(image_files)
    print(f"\nDiscovered {total_images_count} validation images to evaluate.")

    # Save validation manifest
    manifest_path = out_dir / "validation_manifest_10000.txt"
    with open(manifest_path, "w") as f:
        for fname in image_files:
            f.write(f"{fname}\n")
    print(f"Saved validation manifest to {manifest_path}")

    # 4. Load Ground Truth Annotations and Classify Size Groups
    print("\nLoading ground truth annotations and size group categories...")
    # BDD100K original image dimensions: 1280 x 720
    img_w, img_h = 1280.0, 720.0

    # gt_by_class_img[class_name][img_name] = np.ndarray [N, 4]
    gt_boxes_raw = defaultdict(lambda: defaultdict(list))
    # gt_info_list: list of dicts with img_name, class_name, idx_in_img, box, area, size_group, is_vru
    gt_objects = []
    total_gt_count = 0

    for img_name in image_files:
        lbl_path = val_lbl_dir / (Path(img_name).stem + ".txt")
        if not lbl_path.exists():
            continue

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
                
                box = [x1, y1, x2, y2]
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if area < 1024.0:
                    size_grp = "Small (<32²)"
                elif area < 9216.0:
                    size_grp = "Medium [32²-96²)"
                else:
                    size_grp = "Large (≥96²)"

                gt_idx = len(gt_boxes_raw[cname][img_name])
                gt_boxes_raw[cname][img_name].append(box)
                
                gt_objects.append({
                    "img_name": img_name,
                    "class_name": cname,
                    "gt_idx": gt_idx,
                    "box": box,
                    "area": area,
                    "size_group": size_grp,
                    "is_vru": (cname in VRU_CLASS_NAMES),
                })
                total_gt_count += 1

    gt_by_class_img: Dict[str, Dict[str, np.ndarray]] = {c: {} for c in ALL_CLASS_NAMES}
    gt_counts: Dict[str, int] = {c: 0 for c in ALL_CLASS_NAMES}
    for cname in ALL_CLASS_NAMES:
        for img_name, boxes in gt_boxes_raw[cname].items():
            arr = np.array(boxes, dtype=np.float32)
            gt_by_class_img[cname][img_name] = arr
            gt_counts[cname] += len(arr)

    print(f"Loaded {total_gt_count} total ground truth boxes across {len(image_files)} images.")
    for cname in ALL_CLASS_NAMES:
        print(f"  {cname:<14}: {gt_counts[cname]:>6} boxes")

    # 5. Configurations to Evaluate
    CONFIGS = [
        {"name": "Control-640", "imgsz": 640, "batch_size": 32},
        {"name": "Candidate-960", "imgsz": 960, "batch_size": 16},
        {"name": "Candidate-1280", "imgsz": 1280, "batch_size": 8},
    ]

    # Benchmark Batch-1 Latency on fixed representative subset (first 200 images)
    LATENCY_SUBSET_COUNT = 200
    latency_subset = image_files[:LATENCY_SUBSET_COUNT]
    print(f"\nBenchmarking Batch-1 Latency on fixed subset of {LATENCY_SUBSET_COUNT} images...")

    batch1_results: Dict[str, Dict[str, float]] = {}

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        target_res = cfg["imgsz"]
        print(f"\n--- Batch-1 Latency Benchmark: {cfg_name} (imgsz={target_res}) ---")
        model = YOLO(str(ckpt_path))
        model.to(device)

        # Warm-up (10 images batch-1)
        for warm_idx in range(10):
            w_img = cv2.imread(str(val_img_dir / latency_subset[warm_idx]))
            _ = model.predict(source=w_img, imgsz=target_res, conf=0.25, device=device, verbose=False)
        torch.cuda.synchronize()

        # Timed sequential inference
        latencies_ms = []
        for img_name in latency_subset:
            img_path = str(val_img_dir / img_name)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.predict(source=img_path, imgsz=target_res, conf=0.25, device=device, verbose=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        lat_arr = np.array(latencies_ms, dtype=np.float32)
        batch1_stats = {
            "mean_ms": float(np.mean(lat_arr)),
            "std_ms": float(np.std(lat_arr)),
            "median_p50_ms": float(np.median(lat_arr)),
            "p95_ms": float(np.percentile(lat_arr, 95)),
            "min_ms": float(np.min(lat_arr)),
            "max_ms": float(np.max(lat_arr)),
        }
        batch1_results[cfg_name] = batch1_stats
        print(f"  Batch-1 Mean Latency: {batch1_stats['mean_ms']:.2f} ± {batch1_stats['std_ms']:.2f} ms")
        print(f"  Batch-1 Median (p50): {batch1_stats['median_p50_ms']:.2f} ms | p95: {batch1_stats['p95_ms']:.2f} ms")

        del model
        torch.cuda.empty_cache()

    # 6. Full 10,000-Image Batched Inference & Metric Evaluation
    all_config_results: Dict[str, Any] = {}
    config_matched_gts: Dict[str, Dict[str, Dict[str, set]]] = {}
    config_op_preds: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        target_res = cfg["imgsz"]
        batch_size = cfg["batch_size"]
        actual_batch_size = batch_size
        oom_occurred = False

        print("\n" + "=" * 60)
        print(f"Running Full Validation: {cfg_name} | imgsz={target_res} | Initial Batch={batch_size}")
        print("=" * 60)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        model = YOLO(str(ckpt_path))
        model.to(device)

        # Batched warm-up (3 batches)
        warm_paths = [str(val_img_dir / image_files[k]) for k in range(min(batch_size * 3, total_images_count))]
        for w_start in range(0, len(warm_paths), batch_size):
            _ = model.predict(source=warm_paths[w_start : w_start + batch_size], imgsz=target_res, conf=0.25, device=device, verbose=False)
        torch.cuda.synchronize()

        all_preds_by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        op_preds_by_img: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        t_infer_start = time.perf_counter()
        i = 0
        while i < total_images_count:
            chunk_names = image_files[i : i + actual_batch_size]
            chunk_paths = [str(val_img_dir / fname) for fname in chunk_names]

            try:
                results = model.predict(
                    source=chunk_paths,
                    imgsz=target_res,
                    conf=0.01,
                    device=device,
                    verbose=False,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
                    oom_occurred = True
                    actual_batch_size = max(1, actual_batch_size // 2)
                    print(f"WARNING: CUDA OOM at batch {actual_batch_size * 2}. Reduced batch size to {actual_batch_size}.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            for img_name, res in zip(chunk_names, results):
                for det in res.boxes:
                    ckpt_cid = int(det.cls.item())
                    conf = float(det.conf.item())
                    box = det.xyxy[0].cpu().numpy().tolist()
                    class_name = CHECKPOINT_CLASSES[ckpt_cid]

                    all_preds_by_class[class_name].append({
                        "img_name": img_name,
                        "conf": conf,
                        "box": np.array(box, dtype=np.float32),
                    })
                    if conf >= 0.25:
                        op_preds_by_img[img_name].append({
                            "class_name": class_name,
                            "conf": conf,
                            "box": np.array(box, dtype=np.float32),
                        })

            i += actual_batch_size

            if (i % 2000 == 0) or (i >= total_images_count):
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t_infer_start
                curr_fps = i / max(elapsed, 1e-5)
                print(f"  Processed {min(i, total_images_count)}/{total_images_count} images ({curr_fps:.1f} FPS)...")

        torch.cuda.synchronize()
        t_infer_end = time.perf_counter()
        total_infer_time_s = t_infer_end - t_infer_start
        amortized_ms = (total_infer_time_s * 1000.0) / total_images_count
        batched_fps = total_images_count / total_infer_time_s
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)

        print(f"Inference Complete: {total_infer_time_s:.2f} s | {amortized_ms:.2f} ms/img | {batched_fps:.2f} FPS | Peak VRAM: {peak_vram_mb:.1f} MB")

        # 7. Evaluate Metrics
        print("Calculating COCO-style 101-point AP and Operational Metrics...")
        per_class_results = []
        matched_gts_for_cfg: Dict[str, Dict[str, set]] = {}

        for cls_name in ALL_CLASS_NAMES:
            c_preds = all_preds_by_class[cls_name]
            c_gt = gt_by_class_img[cls_name]
            n_gt = gt_counts[cls_name]
            is_vru = (cls_name in VRU_CLASS_NAMES)

            # 101-point AP
            ap_50, ap_50_95, _ = compute_coco_ap_for_class(c_preds, c_gt, n_gt)

            # Operational point (tau=0.25, IoU=0.50)
            tp, fp, fn, prec, rec, f1, matched_gt = evaluate_operational_point(c_preds, c_gt, n_gt, 0.25, 0.50)
            matched_gts_for_cfg[cls_name] = matched_gt

            per_class_results.append({
                "class_name": cls_name,
                "is_vru": is_vru,
                "gt_count": n_gt,
                "total_preds": len(c_preds),
                "operational_preds": tp + fp,
                "tp_50": tp,
                "fp_50": fp,
                "fn_50": fn,
                "precision_50": round(prec, 4),
                "recall_50": round(rec, 4),
                "f1_50": round(f1, 4),
                "ap_50": round(ap_50, 4),
                "ap_50_95": round(ap_50_95, 4),
            })

        config_matched_gts[cfg_name] = matched_gts_for_cfg
        config_op_preds[cfg_name] = op_preds_by_img

        # Aggregates
        active_classes = [r for r in per_class_results if r["gt_count"] > 0]
        overall_mAP_50 = float(np.mean([r["ap_50"] for r in active_classes]))
        overall_mAP_50_95 = float(np.mean([r["ap_50_95"] for r in active_classes]))

        vru_classes = [r for r in per_class_results if r["is_vru"]]
        vru_macro_mAP_50 = float(np.mean([r["ap_50"] for r in vru_classes]))
        vru_macro_mAP_50_95 = float(np.mean([r["ap_50_95"] for r in vru_classes]))

        vru_total_tp = sum(r["tp_50"] for r in vru_classes)
        vru_total_fp = sum(r["fp_50"] for r in vru_classes)
        vru_total_fn = sum(r["fn_50"] for r in vru_classes)
        vru_micro_prec = vru_total_tp / max(vru_total_tp + vru_total_fp, 1)
        vru_micro_rec = vru_total_tp / max(vru_total_tp + vru_total_fn, 1)
        vru_micro_f1 = 2 * vru_micro_prec * vru_micro_rec / max(vru_micro_prec + vru_micro_rec, 1e-8)

        # Size Group Analysis
        size_tier_counts = {
            "Large (≥96²)": {"total_gt": 0, "matched_gt": 0, "vru_gt": 0, "vru_matched": 0},
            "Medium [32²-96²)": {"total_gt": 0, "matched_gt": 0, "vru_gt": 0, "vru_matched": 0},
            "Small (<32²)": {"total_gt": 0, "matched_gt": 0, "vru_gt": 0, "vru_matched": 0},
        }

        for gt_obj in gt_objects:
            grp = gt_obj["size_group"]
            c_name = gt_obj["class_name"]
            i_name = gt_obj["img_name"]
            g_idx = gt_obj["gt_idx"]
            is_vru = gt_obj["is_vru"]

            size_tier_counts[grp]["total_gt"] += 1
            if is_vru:
                size_tier_counts[grp]["vru_gt"] += 1

            # Check if matched in matched_gts_for_cfg
            if g_idx in matched_gts_for_cfg[c_name].get(i_name, set()):
                size_tier_counts[grp]["matched_gt"] += 1
                if is_vru:
                    size_tier_counts[grp]["vru_matched"] += 1

        size_group_results = {}
        for grp, cnts in size_tier_counts.items():
            all_rec = cnts["matched_gt"] / max(cnts["total_gt"], 1)
            v_rec = cnts["vru_matched"] / max(cnts["vru_gt"], 1)
            size_group_results[grp] = {
                "total_gt": cnts["total_gt"],
                "matched_gt": cnts["matched_gt"],
                "overall_recall": round(all_rec, 4),
                "vru_gt": cnts["vru_gt"],
                "vru_matched": cnts["vru_matched"],
                "vru_recall": round(v_rec, 4),
            }

        cfg_summary = {
            "name": cfg_name,
            "imgsz": target_res,
            "initial_batch_size": batch_size,
            "actual_batch_size": actual_batch_size,
            "oom_occurred": oom_occurred,
            "latency_and_throughput": {
                "num_images": total_images_count,
                "total_infer_time_s": round(total_infer_time_s, 3),
                "amortized_ms_per_image": round(amortized_ms, 2),
                "batched_throughput_fps": round(batched_fps, 2),
                "peak_vram_mb": round(peak_vram_mb, 1),
                "batch1_stats": batch1_results[cfg_name],
            },
            "overall_aggregates": {
                "overall_mAP_50": round(overall_mAP_50, 4),
                "overall_mAP_50_95": round(overall_mAP_50_95, 4),
            },
            "vru_aggregates": {
                "vru_macro_mAP_50": round(vru_macro_mAP_50, 4),
                "vru_macro_mAP_50_95": round(vru_macro_mAP_50_95, 4),
                "vru_micro_precision_50": round(vru_micro_prec, 4),
                "vru_micro_recall_50": round(vru_micro_rec, 4),
                "vru_micro_f1_50": round(vru_micro_f1, 4),
            },
            "size_group_results": size_group_results,
            "per_class_results": per_class_results,
        }

        all_config_results[cfg_name] = cfg_summary

        del model
        del all_preds_by_class
        torch.cuda.empty_cache()

    # 8. Extract Representative Comparison Cases (Visual Artifacts)
    print("\nExtracting representative visual cases (Recovered VRUs & Persistent Misses)...")
    ref_matched = config_matched_gts["Control-640"]
    high_matched = config_matched_gts["Candidate-960"]
    max_matched = config_matched_gts["Candidate-1280"]

    recovered_cases = []
    persistent_miss_cases = []

    for gt_obj in gt_objects:
        if not gt_obj["is_vru"]:
            continue
        c_name = gt_obj["class_name"]
        i_name = gt_obj["img_name"]
        g_idx = gt_obj["gt_idx"]
        box = gt_obj["box"]

        m_ref = g_idx in ref_matched[c_name].get(i_name, set())
        m_high = g_idx in high_matched[c_name].get(i_name, set())
        m_max = g_idx in max_matched[c_name].get(i_name, set())

        # Recovered VRU: Missed at 640, detected at 960 or 1280
        if not m_ref and (m_high or m_max) and len(recovered_cases) < 10:
            rec_method = "960 & 1280" if (m_high and m_max) else ("960" if m_high else "1280")
            recovered_cases.append({
                "type": "Recovered VRU",
                "img_name": i_name,
                "class_name": c_name,
                "box": box,
                "recovered_by": rec_method,
            })

        # Persistent Miss: Missed across all three resolutions
        if not m_ref and not m_high and not m_max and len(persistent_miss_cases) < 10:
            persistent_miss_cases.append({
                "type": "Persistent Miss",
                "img_name": i_name,
                "class_name": c_name,
                "box": box,
                "notes": "Missed at 640, 960, and 1280",
            })

        if len(recovered_cases) >= 10 and len(persistent_miss_cases) >= 10:
            break

    # Render Annotated Visual Examples
    visual_records = []
    for idx, c in enumerate(recovered_cases):
        img_path = val_img_dir / c["img_name"]
        img = cv2.imread(str(img_path))
        b = [int(v) for v in c["box"]]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.putText(img, f"RECOVERED {c['class_name']} ({c['recovered_by']})", (b[0], max(20, b[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out_name = f"recovered_vru_{idx+1:02d}_{c['class_name']}_{c['img_name']}"
        cv2.imwrite(str(visual_dir / out_name), img)
        visual_records.append({
            "category": "Recovered VRU",
            "image": c["img_name"],
            "class": c["class_name"],
            "file": out_name,
            "details": f"Missed at 640px, recovered by {c['recovered_by']}",
        })

    for idx, c in enumerate(persistent_miss_cases):
        img_path = val_img_dir / c["img_name"]
        img = cv2.imread(str(img_path))
        b = [int(v) for v in c["box"]]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cv2.putText(img, f"PERSISTENT MISS: {c['class_name']}", (b[0], max(20, b[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        out_name = f"persistent_miss_{idx+1:02d}_{c['class_name']}_{c['img_name']}"
        cv2.imwrite(str(visual_dir / out_name), img)
        visual_records.append({
            "category": "Persistent Miss",
            "image": c["img_name"],
            "class": c["class_name"],
            "file": out_name,
            "details": "Missed across 640px, 960px, and 1280px; candidate for active foveation",
        })

    print(f"Rendered {len(visual_records)} comparison images in {visual_dir}")

    # 9. Output Generation: JSON, CSVs, Run Log
    print("\nWriting auditable output artifacts...")

    # A. Machine-readable JSON
    final_output = {
        "benchmark_name": "RoadSentinel Phase 2B.2 — Full-Validation Multi-Resolution Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": {
            "file": ckpt_path.name,
            "sha256": ckpt_sha256,
            "native_train_imgsz": trained_imgsz,
            "train_epochs": trained_epochs,
            "train_date": trained_date,
        },
        "hardware": {
            "device": device,
            "gpu_name": gpu_name,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "platform": platform.platform(),
        },
        "validation_dataset": {
            "num_images": total_images_count,
            "total_gt_boxes": total_gt_count,
            "vru_gt_boxes": sum(gt_counts[c] for c in VRU_CLASS_NAMES),
        },
        "configurations": all_config_results,
        "visual_comparison_examples": visual_records,
    }

    json_path = out_dir / "full_resolution_metrics.json"
    with open(json_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"Wrote JSON metrics to {json_path}")

    # B. Per-Class Comparison CSV
    csv_per_class = out_dir / "per_class_comparison.csv"
    with open(csv_per_class, "w") as f:
        f.write("Class,Target,GT_Count,Ref640_AP50,HighRes960_AP50,MaxRes1280_AP50,Ref640_Rec50,HighRes960_Rec50,MaxRes1280_Rec50,Ref640_Prec50,HighRes960_Prec50,MaxRes1280_Prec50,Ref640_F1,HighRes960_F1,MaxRes1280_F1\n")
        for i, cname in enumerate(ALL_CLASS_NAMES):
            target_str = "VRU" if cname in VRU_CLASS_NAMES else "Standard"
            gt_c = gt_counts[cname]
            r640 = all_config_results["Control-640"]["per_class_results"][i]
            r960 = all_config_results["Candidate-960"]["per_class_results"][i]
            r1280 = all_config_results["Candidate-1280"]["per_class_results"][i]
            f.write(f"{cname},{target_str},{gt_c},{r640['ap_50']},{r960['ap_50']},{r1280['ap_50']},"
                    f"{r640['recall_50']},{r960['recall_50']},{r1280['recall_50']},"
                    f"{r640['precision_50']},{r960['precision_50']},{r1280['precision_50']},"
                    f"{r640['f1_50']},{r960['f1_50']},{r1280['f1_50']}\n")
    print(f"Wrote Per-Class CSV to {csv_per_class}")

    # C. Size Group Analysis CSV
    csv_size = out_dir / "size_group_analysis.csv"
    with open(csv_size, "w") as f:
        f.write("SizeGroup,Total_GT,VRU_GT,Ref640_VRU_Recall,HighRes960_VRU_Recall,MaxRes1280_VRU_Recall,Ref640_All_Recall,HighRes960_All_Recall,MaxRes1280_All_Recall\n")
        for grp in ["Large (≥96²)", "Medium [32²-96²)", "Small (<32²)"]:
            sg640 = all_config_results["Control-640"]["size_group_results"][grp]
            sg960 = all_config_results["Candidate-960"]["size_group_results"][grp]
            sg1280 = all_config_results["Candidate-1280"]["size_group_results"][grp]
            f.write(f'"{grp}",{sg640["total_gt"]},{sg640["vru_gt"]},'
                    f'{sg640["vru_recall"]},{sg960["vru_recall"]},{sg1280["vru_recall"]},'
                    f'{sg640["overall_recall"]},{sg960["overall_recall"]},{sg1280["overall_recall"]}\n')
    print(f"Wrote Size Group CSV to {csv_size}")

    # D. Latency & Memory CSV
    csv_lat = out_dir / "latency_and_memory.csv"
    with open(csv_lat, "w") as f:
        f.write("Configuration,Resolution,BatchSize,ActualBatchSize,ThroughputFPS,AmortizedMS,Batch1_MeanMS,Batch1_StdMS,Batch1_P50MS,Batch1_P95MS,PeakVRAM_MB\n")
        for cname in ["Control-640", "Candidate-960", "Candidate-1280"]:
            cf = all_config_results[cname]
            lt = cf["latency_and_throughput"]
            b1 = lt["batch1_stats"]
            f.write(f"{cname},{cf['imgsz']},{cf['initial_batch_size']},{cf['actual_batch_size']},"
                    f"{lt['batched_throughput_fps']},{lt['amortized_ms_per_image']},"
                    f"{b1['mean_ms']:.2f},{b1['std_ms']:.2f},{b1['median_p50_ms']:.2f},{b1['p95_ms']:.2f},"
                    f"{lt['peak_vram_mb']}\n")
    print(f"Wrote Latency & Memory CSV to {csv_lat}")

    # E. Execution Run Log
    log_path = out_dir / "run_log.txt"
    with open(log_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ROADSENTINEL PHASE 2B.2 — FULL-VALIDATION BENCHMARK RUN LOG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {final_output['timestamp']}\n")
        f.write(f"Hardware: {gpu_name} ({device}) | PyTorch {torch.__version__} | CUDA {torch.version.cuda}\n")
        f.write(f"Checkpoint SHA-256: {ckpt_sha256}\n")
        f.write(f"Checkpoint Provenance: Native imgsz={trained_imgsz}, epochs={trained_epochs}, date={trained_date}\n")
        f.write(f"Images Evaluated: {total_images_count} | Total GT: {total_gt_count}\n\n")

        for cname in ["Control-640", "Candidate-960", "Candidate-1280"]:
            cf = all_config_results[cname]
            lt = cf["latency_and_throughput"]
            b1 = lt["batch1_stats"]
            ov = cf["overall_aggregates"]
            vru = cf["vru_aggregates"]
            f.write(f"CONFIGURATION: {cname} (imgsz={cf['imgsz']})\n")
            f.write(f"  Batch Size: {cf['actual_batch_size']} (initial: {cf['initial_batch_size']}, OOM: {cf['oom_occurred']})\n")
            f.write(f"  Batched Throughput: {lt['batched_throughput_fps']} FPS ({lt['amortized_ms_per_image']} ms/img)\n")
            f.write(f"  Batch-1 Latency: Mean={b1['mean_ms']:.2f}ms, P50={b1['median_p50_ms']:.2f}ms, P95={b1['p95_ms']:.2f}ms\n")
            f.write(f"  Peak VRAM: {lt['peak_vram_mb']} MB\n")
            f.write(f"  Overall mAP@50: {ov['overall_mAP_50']} | mAP@50-95: {ov['overall_mAP_50_95']}\n")
            f.write(f"  VRU Macro mAP@50: {vru['vru_macro_mAP_50']} | mAP@50-95: {vru['vru_macro_mAP_50_95']}\n")
            f.write(f"  VRU Operational Point: Prec={vru['vru_micro_precision_50']}, Rec={vru['vru_micro_recall_50']}, F1={vru['vru_micro_f1_50']}\n\n")
    print(f"Wrote Run Log to {log_path}")

    print("\n" + "=" * 80)
    print("RoadSentinel Phase 2B.2 Full Validation Benchmark Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
