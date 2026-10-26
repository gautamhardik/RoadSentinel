"""
RoadSentinel - Phase 2A Corrected Baseline Evaluation
Runs comprehensive, class-aligned evaluation on the VRU-enriched validation subset.
Saves metrics to experiments/baseline_v2/baseline_metrics.json
Saves annotated failure cases to experiments/baseline_v2/failure_cases/
"""

import os
import sys
import time
import json
import platform
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_engine import (
    DetectionEvaluator,
    GT_CLASSES,
    CHECKPOINT_CLASSES,
    CHECKPOINT_TO_GT_ID,
    ALL_CLASS_NAMES,
    VRU_CLASS_NAMES,
    compute_box_iou,
)

# Paths
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "bdd100k_yolo11s.pt"
VAL_IMG_DIR = PROJECT_ROOT / "bdd100k_dataset" / "val" / "images"
VAL_LBL_DIR = PROJECT_ROOT / "bdd100k_dataset" / "val" / "labels"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "baseline_v2"
FAILURES_DIR = OUTPUT_DIR / "failure_cases"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FAILURES_DIR.mkdir(parents=True, exist_ok=True)


def build_evaluation_dataset() -> List[str]:
    """
    Selects the 100-image VRU-enriched validation subset
    (70 VRU-rich images + 30 standard traffic images).
    """
    # GT class IDs for VRUs in dataset: 0 (person), 1 (rider), 5 (bike), 6 (motor)
    GT_VRU_IDS = {0, 1, 5, 6}

    vru_rich_files = []
    standard_files = []

    for fname in sorted(os.listdir(VAL_LBL_DIR)):
        if not fname.endswith(".txt"):
            continue
        with open(VAL_LBL_DIR / fname, "r") as f:
            classes = [int(line.split()[0]) for line in f if line.strip()]

        img_name = fname.replace(".txt", ".jpg")
        if not (VAL_IMG_DIR / img_name).exists():
            continue

        if any(c in GT_VRU_IDS for c in classes):
            vru_rich_files.append(img_name)
        else:
            standard_files.append(img_name)

    eval_set = vru_rich_files[:70] + standard_files[:30]
    print(f"Loaded evaluation subset: {len(eval_set)} images (70 VRU-rich + 30 standard traffic).")
    return eval_set


def measure_hardware_and_load_model():
    """Measures model loading latency and system environment."""
    t_load_start = time.perf_counter()
    model = YOLO(str(WEIGHTS_PATH))
    t_load_end = time.perf_counter()
    load_ms = (t_load_end - t_load_start) * 1000

    hw_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "model_weights": WEIGHTS_PATH.name,
        "model_load_time_ms": round(load_ms, 2),
    }
    return model, hw_info


def run_warmup(model: YOLO, sample_img_path: Path, num_warmup: int = 5):
    """Executes isolated warm-up inferences to prime caches and runtime."""
    print(f"Executing {num_warmup} warm-up cycles (excluded from latency timing)...")
    img = cv2.imread(str(sample_img_path))
    for i in range(num_warmup):
        _ = model.predict(source=img, imgsz=640, conf=0.25, verbose=False)
    print("Warm-up complete.")


def run_evaluation():
    print("=" * 70)
    print("RoadSentinel - Phase 2A Evaluation Integrity Audit & Repair")
    print("=" * 70)

    # 1. Dataset selection
    eval_images = build_evaluation_dataset()

    # 2. Hardware and Model Loading
    model, hw_info = measure_hardware_and_load_model()
    print(f"Device: {hw_info['device']} | Architecture: {hw_info['processor']}")
    print(f"Model loaded in {hw_info['model_load_time_ms']} ms")

    # 3. Warm-up
    sample_img_path = VAL_IMG_DIR / eval_images[0]
    run_warmup(model, sample_img_path, num_warmup=5)

    # 4. Evaluation Loop
    evaluator = DetectionEvaluator(
        operational_conf=0.25,
        operational_iou=0.50,
    )

    inference_latencies_ms = []
    failure_cases = []

    print(f"\nRunning evaluation on {len(eval_images)} images...")

    for idx, img_name in enumerate(eval_images, 1):
        img_path = VAL_IMG_DIR / img_name
        lbl_path = VAL_LBL_DIR / img_name.replace(".jpg", ".txt")

        img = cv2.imread(str(img_path))
        h, w, _ = img.shape

        # Load GT
        gt_boxes_img = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                    evaluator.add_ground_truth_yolo(img_name, cid, xc, yc, bw, bh, w, h)
                    
                    # Store for failure case annotation
                    cname = GT_CLASSES[cid]
                    x1 = (xc - bw / 2.0) * w
                    y1 = (yc - bh / 2.0) * h
                    x2 = (xc + bw / 2.0) * w
                    y2 = (yc + bh / 2.0) * h
                    gt_boxes_img.append({"name": cname, "box": [x1, y1, x2, y2], "matched": False})

        # Timed Inference
        t0 = time.perf_counter()
        # Use low conf (0.01) so full PR curve can be calculated, but evaluator handles operational_conf 0.25
        res = model.predict(source=img, imgsz=640, conf=0.01, verbose=False)[0]
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000
        inference_latencies_ms.append(lat_ms)

        # Record Predictions
        pred_boxes_img = []
        for det in res.boxes:
            ckpt_cid = int(det.cls.item())
            conf = float(det.conf.item())
            box = det.xyxy[0].tolist()
            evaluator.add_prediction_from_checkpoint(img_name, ckpt_cid, conf, box)
            
            cname = CHECKPOINT_CLASSES[ckpt_cid]
            pred_boxes_img.append({"name": cname, "conf": conf, "box": box})

        # Failure Case Logging (save visual annotations for VRU misses/errors at conf >= 0.25)
        op_preds = [p for p in pred_boxes_img if p["conf"] >= 0.25]
        
        # Match for this image to identify misses
        for p in op_preds:
            best_iou = 0.0
            best_gt = None
            for gt in gt_boxes_img:
                if gt["name"] == p["name"]:
                    iou = compute_box_iou(p["box"], gt["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt
            if best_gt is not None and best_iou >= 0.50 and not best_gt["matched"]:
                best_gt["matched"] = True

        # Find unmatched VRU GTs (False Negatives)
        missed_vrus = [gt for gt in gt_boxes_img if gt["name"] in VRU_CLASS_NAMES and not gt["matched"]]
        if len(missed_vrus) > 0 and len(failure_cases) < 15:
            vis = img.copy()
            # Draw Missed GTs in RED
            for m in missed_vrus:
                x1, y1, x2, y2 = [int(v) for v in m["box"]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis, f"FN (Missed): {m['name']}", (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            # Draw Operational Detections in GREEN (or CYAN for VRUs)
            for p in op_preds:
                x1, y1, x2, y2 = [int(v) for v in p["box"]]
                is_vru = p["name"] in VRU_CLASS_NAMES
                color = (255, 255, 0) if is_vru else (0, 255, 0)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                cv2.putText(vis, f"{p['name']} {p['conf']:.2f}", (x1, max(12, y1 - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

            fail_filename = f"failure_vru_{len(failure_cases) + 1:02d}_{img_name}"
            fail_filepath = FAILURES_DIR / fail_filename
            cv2.imwrite(str(fail_filepath), vis)
            failure_cases.append({
                "image_name": img_name,
                "missed_vru_count": len(missed_vrus),
                "missed_classes": [m["name"] for m in missed_vrus],
                "saved_artifact": str(fail_filepath.relative_to(PROJECT_ROOT)),
            })

        if idx % 20 == 0 or idx == len(eval_images):
            mean_so_far = np.mean(inference_latencies_ms)
            print(f"  [{idx:03d}/{len(eval_images)}] Processed {img_name} | Latency: {lat_ms:.1f}ms (Mean: {mean_so_far:.1f}ms)")

    # 5. Evaluate Metrics
    results = evaluator.evaluate()

    # Latency statistics
    lat_arr = np.array(inference_latencies_ms)
    latency_stats = {
        "device": hw_info["device"],
        "processor": hw_info["processor"],
        "num_timed_images": len(inference_latencies_ms),
        "image_size": "640x640",
        "mean_ms": round(float(np.mean(lat_arr)), 2),
        "median_ms": round(float(np.median(lat_arr)), 2),
        "std_ms": round(float(np.std(lat_arr)), 2),
        "min_ms": round(float(np.min(lat_arr)), 2),
        "max_ms": round(float(np.max(lat_arr)), 2),
        "p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
        "fps": round(1000.0 / float(np.mean(lat_arr)), 2),
    }

    # Assemble complete metrics output
    final_payload = {
        "evaluation_name": "RoadSentinel Phase 2A Corrected Baseline Evaluation",
        "evaluation_type": "VRU-enriched validation subset (NOT representative full validation)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_environment": hw_info,
        "latency_methodology": {
            "model_loading_isolated": True,
            "warmup_cycles": 5,
            "timed_inference_images": len(eval_images),
            "stats": latency_stats,
        },
        "evaluation_config": results["config"],
        "summary": results["summary"],
        "vru_aggregates": results["vru_aggregates"],
        "per_class_metrics": results["per_class_metrics"],
        "failure_cases": failure_cases,
    }

    # Save to baseline_metrics.json
    metrics_json_path = OUTPUT_DIR / "baseline_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print("\n" + "=" * 70)
    print("CORRECTED BASELINE EVALUATION RESULTS")
    print("=" * 70)
    print(f"Evaluated Images       : {results['summary']['evaluated_images_count']} (70 VRU-rich, 30 standard)")
    print(f"Overall mAP@50         : {results['summary']['overall_mAP_50']:.4f}")
    print(f"Overall mAP@50-95      : {results['summary']['overall_mAP_50_95']:.4f}")
    print(f"Overall Macro P@50     : {results['summary']['overall_macro_precision_50']:.4f}")
    print(f"Overall Macro R@50     : {results['summary']['overall_macro_recall_50']:.4f}")
    print("-" * 70)
    print("VRU AGGREGATE PERFORMANCE (person, rider, bike, motor):")
    print(f"VRU Macro mAP@50       : {results['vru_aggregates']['vru_macro_mAP_50']:.4f}")
    print(f"VRU Macro mAP@50-95    : {results['vru_aggregates']['vru_macro_mAP_50_95']:.4f}")
    print(f"VRU Micro Precision@50 : {results['vru_aggregates']['vru_micro_precision_50']:.4f}")
    print(f"VRU Micro Recall@50    : {results['vru_aggregates']['vru_micro_recall_50']:.4f}")
    print(f"VRU Micro F1@50        : {results['vru_aggregates']['vru_micro_f1_50']:.4f}")
    print("-" * 70)
    print(f"Inference Latency (CPU): Mean={latency_stats['mean_ms']}ms | Median={latency_stats['median_ms']}ms | FPS={latency_stats['fps']}")
    print("=" * 70)
    print(f"{'Class Name':<15} {'GT':<6} {'Pred(Op)':<10} {'TP@50':<8} {'FP@50':<8} {'FN@50':<8} {'P@50':<8} {'R@50':<8} {'AP@50':<8} {'AP@50-95':<8}")
    print("-" * 95)
    for m in results["per_class_metrics"]:
        vru_mark = "* " if m["is_vru"] else "  "
        cname = vru_mark + m["name"]
        ap50_str = f"{m['ap_50']:.4f}" if m["ap_50"] is not None else "N/A (0 GT)"
        ap5095_str = f"{m['ap_50_95']:.4f}" if m["ap_50_95"] is not None else "N/A (0 GT)"
        print(f"{cname:<15} {m['gt_count']:<6} {m['pred_operational_count']:<10} {m['tp_50']:<8} {m['fp_50']:<8} {m['fn_50']:<8} {m['precision_50']:<8.4f} {m['recall_50']:<8.4f} {ap50_str:<8} {ap5095_str:<8}")
    print("=" * 95)
    print(f"Metrics saved to: {metrics_json_path}")
    print(f"Logged {len(failure_cases)} annotated VRU failure images in: {FAILURES_DIR}")
    return final_payload


if __name__ == "__main__":
    run_evaluation()
