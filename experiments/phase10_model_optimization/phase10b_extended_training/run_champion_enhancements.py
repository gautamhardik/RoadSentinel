"""
RoadSentinel Phase 10B.1 — Champion Enhancement Evaluation
Evaluates the New 50-Epoch Champion (session4_best.pt) under:
1. Adaptive Confidence Thresholding (VRU conf in [0.15, 0.18, 0.20, 0.25])
2. Active Foveation (960px Full-Frame + 640px Uncertainty Crops)
across the exact 300-image development pilot cohort (PILOT_300_FILENAMES).
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import cv2

# Ensure OpenMP stability on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from ultralytics import YOLO

from src.foveation.engine import ActiveFoveationEngine
from src.foveation.types import FoveationConfig
from kaggle_jobs.pilot_300_native_eval.evaluate_pilot_native import (
    PILOT_300_FILENAMES, CLASS_NAMES, VRU_NAMES,
    match_single_image, aggregate_partition
)
from experiments.phase8_foveation.evaluate_phase8_foveation import load_ground_truth

CHAMPION_PATH = REPO_ROOT / "experiments" / "phase10_model_optimization" / "phase10b_extended_training" / "session4_artifacts" / "session4_best.pt"
VAL_IMG_DIR = REPO_ROOT / "bdd100k_dataset" / "val" / "images"
VAL_LBL_DIR = REPO_ROOT / "bdd100k_dataset" / "val" / "labels"
OUT_DIR = REPO_ROOT / "experiments" / "phase10_model_optimization" / "phase10b_extended_training"


def evaluate_threshold_sweep(pilot_subset: list, gt_by_image: dict, raw_predictions: dict) -> dict:
    """Evaluates recall/precision across varying VRU confidence thresholds."""
    thresholds = [0.15, 0.18, 0.20, 0.25]
    sweep_results = {}

    for tau_vru in thresholds:
        image_stats = {}
        for name in pilot_subset:
            # Filter predictions: tau_vru for VRU, 0.25 for vehicles/others
            filtered_preds = []
            for p in raw_predictions[name]:
                thresh = tau_vru if p["is_vru"] else 0.25
                if p["conf"] >= thresh:
                    filtered_preds.append(p)
            image_stats[name] = match_single_image(filtered_preds, gt_by_image[name])

        agg = aggregate_partition(image_stats, set(pilot_subset))
        vru = agg["vru_metrics"]
        all_c = agg["all_class_metrics"]
        sweep_results[f"vru_conf_{tau_vru:.2f}"] = {
            "vru_threshold": tau_vru,
            "non_vru_threshold": 0.25,
            "small_vru_recall": vru["small_vru_recall"],
            "small_vru_tp": vru["small_vru_true_positives"],
            "vru_recall": vru["recall"],
            "vru_precision": vru["precision"],
            "vru_f1": vru["f1_score"],
            "all_class_recall": all_c["recall"],
            "all_class_precision": all_c["precision"]
        }
    return sweep_results


def evaluate_champion_foveation(pilot_subset: list, gt_by_image: dict) -> dict:
    """Evaluates the 50-epoch champion inside the Active Foveation Engine."""
    print("\n--- Initializing Active Foveation with 50-Epoch Champion ---", flush=True)
    cfg = FoveationConfig(
        full_frame_imgsz=960,
        crop_imgsz=640,
        max_crops_per_image=2,
        policy="uncertainty_active",
        low_conf_min=0.10,
        low_conf_max=0.35,
        small_box_area_threshold=1024.0,
        cluster_iou_threshold=0.35,
        context_margin_ratio=0.25,
        min_crop_size=128,
        fusion_iou_threshold=0.50,
        boundary_suppress_margin=8,
        operational_conf_threshold=0.25,
    )

    engine = ActiveFoveationEngine(
        checkpoint_path=str(CHAMPION_PATH),
        config=cfg,
        device="cpu"
    )

    image_stats = {}
    total_time = 0.0
    crop_counts = []

    for idx, name in enumerate(pilot_subset, start=1):
        img_p = VAL_IMG_DIR / name
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        t0 = time.perf_counter()
        fused_dets, trace = engine.process_image(img, conf_threshold=0.15)
        dt = (time.perf_counter() - t0)
        total_time += dt
        crop_counts.append(trace.num_crops_executed)

        preds = []
        for d in fused_dets:
            if d.confidence >= 0.25:
                preds.append({
                    "box": list(d.box),
                    "conf": float(d.confidence),
                    "class_name": d.class_name,
                    "is_vru": d.class_name in VRU_NAMES
                })

        st = match_single_image(preds, gt_by_image[name])
        image_stats[name] = st

        if idx % 50 == 0 or idx == len(pilot_subset):
            avg_crops = np.mean(crop_counts)
            print(f"  Foveation: {idx} / {len(pilot_subset)} images done (avg crops: {avg_crops:.2f})...", flush=True)

    agg = aggregate_partition(image_stats, set(pilot_subset))
    agg["avg_latency_ms"] = round((total_time / len(pilot_subset)) * 1000.0, 1)
    agg["mean_crops_per_image"] = round(float(np.mean(crop_counts)), 2)
    return agg


def main():
    print("=" * 75, flush=True)
    print("RoadSentinel Phase 10B.1 — Champion Enhancement Evaluation", flush=True)
    print("=" * 75, flush=True)

    pilot_subset = sorted(list(PILOT_300_FILENAMES))
    gt_by_image = load_ground_truth(PILOT_300_FILENAMES)

    # 1. Pre-generate raw champion predictions down to conf=0.10 for fast threshold tuning
    print("\n--- Pass 1: Batched Champion Prediction (conf=0.10) for Threshold Sweeps ---", flush=True)
    model = YOLO(str(CHAMPION_PATH))
    raw_predictions = {}
    batch_size = 16

    for i in range(0, len(pilot_subset), batch_size):
        b_names = pilot_subset[i:i + batch_size]
        b_imgs = [cv2.imread(str(VAL_IMG_DIR / n)) for n in b_names]
        results = model.predict(source=b_imgs, imgsz=960, conf=0.10, iou=0.65, batch=batch_size, device="cpu", verbose=False)
        for name, r in zip(b_names, results):
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            preds = []
            for b, s, c in zip(boxes, scores, cls_ids):
                cname = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
                preds.append({
                    "box": b.tolist(),
                    "conf": float(s),
                    "class_name": cname,
                    "is_vru": cname in VRU_NAMES
                })
            raw_predictions[name] = preds
        processed = min(i + batch_size, len(pilot_subset))
        print(f"  Processed {processed} / {len(pilot_subset)} images...", flush=True)

    # 2. Threshold Sweep Evaluation
    print("\nEvaluating Confidence Threshold Sweeps...", flush=True)
    sweep_results = evaluate_threshold_sweep(pilot_subset, gt_by_image, raw_predictions)

    # 3. Active Foveation Evaluation with 50-Epoch Champion
    foveation_results = evaluate_champion_foveation(pilot_subset, gt_by_image)

    # 4. Synthesize Comparison Matrix
    comparison = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "champion_checkpoint": str(CHAMPION_PATH),
        "threshold_sweep": sweep_results,
        "champion_active_foveation": {
            "small_vru_recall": foveation_results["vru_metrics"]["small_vru_recall"],
            "small_vru_tp": foveation_results["vru_metrics"]["small_vru_true_positives"],
            "vru_recall": foveation_results["vru_metrics"]["recall"],
            "vru_precision": foveation_results["vru_metrics"]["precision"],
            "vru_f1": foveation_results["vru_metrics"]["f1_score"],
            "all_class_recall": foveation_results["all_class_metrics"]["recall"],
            "all_class_precision": foveation_results["all_class_metrics"]["precision"],
            "avg_latency_ms": foveation_results["avg_latency_ms"],
            "mean_crops_per_image": foveation_results["mean_crops_per_image"],
            "per_class": foveation_results["per_class"]
        }
    }

    out_file = OUT_DIR / "champion_enhancement_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[SAVED] Enhancement comparison saved to: {out_file}", flush=True)

    # Print summary
    print("\n" + "=" * 80)
    print("CHAMPION ENHANCEMENT BENCHMARK SUMMARY (300-IMAGE PILOT)")
    print("=" * 80)
    print(f"{'Configuration':<35} | {'Small-VRU Rec':<14} | {'VRU Rec':<10} | {'VRU Prec':<10} | {'VRU F1':<8}")
    print("-" * 80)
    print(f"{'Baseline 960px (26 ep, conf=0.25)':<35} | {'42.73% (385)':<14} | {'60.46%':<10} | {'78.45%':<10} | {'0.6829':<8}")
    print(f"{'Champion 960px (50 ep, conf=0.25)':<35} | {'42.95% (387)':<14} | {'61.09%':<10} | {'78.62%':<10} | {'0.6876':<8}")
    
    for k, v in sweep_results.items():
        tau = v['vru_threshold']
        lbl = f"Champion (VRU conf={tau:.2f}, other=0.25)"
        rec_str = f"{v['small_vru_recall']:.2f}% ({v['small_vru_tp']})"
        print(f"{lbl:<35} | {rec_str:<14} | {v['vru_recall']:5.2f}%    | {v['vru_precision']:5.2f}%    | {v['vru_f1']:6.4f}")
    
    fov_vru = foveation_results["vru_metrics"]
    fov_rec_str = f"{fov_vru['small_vru_recall']:.2f}% ({fov_vru['small_vru_true_positives']})"
    print(f"{'Champion + Active Foveation':<35} | {fov_rec_str:<14} | {fov_vru['recall']:5.2f}%    | {fov_vru['precision']:5.2f}%    | {fov_vru['f1_score']:6.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
