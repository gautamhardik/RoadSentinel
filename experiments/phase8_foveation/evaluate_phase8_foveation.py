"""
RoadSentinel Phase 8 — Active Foveation Controlled Inference Experiment
Evaluates Configuration C (960px Full-Frame + Active Foveation) across the identical
300-image development pilot cohort and compares against:
- Configuration A: 960px Full-Frame Baseline
- Configuration B: 1280px Native Resolution Reference

Records:
- Crop selection count (total, mean, per-image distribution)
- Inspected crop area percentage
- Per-stage latency profiling (read, full-frame infer, proposal, crop extraction, crop infer, fusion, total)
- Peak memory usage (RSS MB and tracemalloc)
- Detection metrics: VRU precision/recall/F1, small-VRU recall, per-class breakdown (including motor)
"""

import os
import sys
import time
import json
import hashlib
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import cv2

# Ensure OpenMP stability on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.foveation.engine import ActiveFoveationEngine
from src.foveation.types import FoveationConfig, Detection, FoveationTrace
from kaggle_jobs.pilot_300_native_eval.evaluate_pilot_native import (
    PILOT_300_FILENAMES, CLASS_NAMES, VRU_NAMES, VRU_IDS,
    EXPECTED_SHA256, BASELINE_960_PILOT, match_single_image
)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


CHECKPOINT_PATH = REPO_ROOT / "scratch" / "kaggle_upload_weights" / "best.pt"
VAL_IMG_DIR = REPO_ROOT / "bdd100k_dataset" / "val" / "images"
VAL_LBL_DIR = REPO_ROOT / "bdd100k_dataset" / "val" / "labels"
OUT_DIR = REPO_ROOT / "experiments" / "phase8_foveation"
CHECKPOINT_JSON = OUT_DIR / "foveation_checkpoint.json"
PER_IMAGE_JSON = OUT_DIR / "foveation_per_image_results.json"
COMPARISON_JSON = OUT_DIR / "foveation_pilot_comparison.json"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_ground_truth(pilot_subset: set) -> Dict[str, List[Dict[str, Any]]]:
    """Loads ground truth bounding boxes in 1280x720 coordinates."""
    gt_by_image = {}
    for img_name in pilot_subset:
        gt_by_image[img_name] = []
        lbl_path = VAL_LBL_DIR / (Path(img_name).stem + ".txt")
        if not lbl_path.exists():
            continue
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                x1 = (xc - w / 2.0) * 1280.0
                y1 = (yc - h / 2.0) * 720.0
                x2 = (xc + w / 2.0) * 1280.0
                y2 = (yc + h / 2.0) * 720.0
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                
                if area < 1024.0:
                    sz_grp = "Small (<32²)"
                elif area < 9216.0:
                    sz_grp = "Medium [32²-96²)"
                else:
                    sz_grp = "Large (≥96²)"

                cname = CLASS_NAMES[cid]
                gt_by_image[img_name].append({
                    "class_name": cname,
                    "box": [x1, y1, x2, y2],
                    "size_group": sz_grp,
                    "is_vru": cname in VRU_NAMES,
                    "area": area,
                })
    return gt_by_image


def compute_crop_union_area(proposals: List[Any], img_w: int = 1280, img_h: int = 720) -> float:
    """Computes exact pixel area covered by the union of crop boxes."""
    if not proposals:
        return 0.0
    # Create a 2D boolean mask to calculate exact union area without double-counting overlap
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for p in proposals:
        b = p.crop_box if hasattr(p, "crop_box") else p["crop_box"]
        x1 = max(0, min(img_w, int(round(b[0]))))
        y1 = max(0, min(img_h, int(round(b[1]))))
        x2 = max(0, min(img_w, int(round(b[2]))))
        y2 = max(0, min(img_h, int(round(b[3]))))
        mask[y1:y2, x1:x2] = 1
    return float(np.sum(mask))


def run_evaluation(
    max_crops: int = 2,
    crop_imgsz: int = 640,
    operational_conf: float = 0.25,
    trigger_conf: float = 0.15,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RoadSentinel Phase 8 — Active Foveation Controlled Evaluation")
    print("=" * 80)

    # 1. Checkpoint Verification
    print(f"Target checkpoint: {CHECKPOINT_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    actual_sha = compute_sha256(CHECKPOINT_PATH)
    print(f"Actual SHA-256:   {actual_sha}")
    print(f"Expected SHA-256: {EXPECTED_SHA256}")
    if actual_sha != EXPECTED_SHA256:
        raise ValueError("CRITICAL: SHA-256 checksum mismatch on frozen checkpoint!")
    print("SUCCESS: Checkpoint SHA-256 verified.")

    # 2. Dataset Cohort Setup
    pilot_subset = sorted(list(PILOT_300_FILENAMES))
    if limit is not None and limit > 0:
        pilot_subset = pilot_subset[:limit]
        print(f"NOTE: Subset limit applied: running on {len(pilot_subset)} images.")
    n_images = len(pilot_subset)
    print(f"\nCohort: {n_images} development images strictly quarantined from test split.")
    gt_by_image = load_ground_truth(PILOT_300_FILENAMES)
    total_gt_all = sum(len(objs) for objs in gt_by_image.values())
    total_gt_vru = sum(sum(1 for o in objs if o["is_vru"]) for objs in gt_by_image.values())
    total_gt_small_vru = sum(sum(1 for o in objs if o["is_vru"] and o["size_group"] == "Small (<32²)") for objs in gt_by_image.values())
    print(f"Ground Truth: {total_gt_all} total objects ({total_gt_vru} VRUs, {total_gt_small_vru} small VRUs).")

    # 3. Initialize Active Foveation Engine
    foveation_cfg = FoveationConfig(
        full_frame_imgsz=960,
        crop_imgsz=crop_imgsz,
        max_crops_per_image=max_crops,
        policy="uncertainty_active",
        low_conf_min=0.10,
        low_conf_max=0.35,
        small_box_area_threshold=1024.0,
        cluster_iou_threshold=0.35,
        context_margin_ratio=0.25,
        min_crop_size=128,
        fusion_iou_threshold=0.50,
        boundary_suppress_margin=8,
        operational_conf_threshold=operational_conf,
    )
    print(f"\nEngine Config: Full-frame={foveation_cfg.full_frame_imgsz}px, Crop={foveation_cfg.crop_imgsz}px, MaxCrops={max_crops}")
    print(f"Confidence Thresholds: Trigger={trigger_conf}, Operational={operational_conf}")

    tracemalloc.start()
    t_init_start = time.perf_counter()
    engine = ActiveFoveationEngine(
        checkpoint_path=str(CHECKPOINT_PATH),
        config=foveation_cfg,
        device="cpu",
    )
    init_ms = (time.perf_counter() - t_init_start) * 1000.0
    print(f"Model initialized on device={engine.device} in {init_ms:.1f} ms.")

    # Warm-up pass
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, _ = engine.process_image(dummy_frame, conf_threshold=trigger_conf)
    print("Warm-up pass completed.")

    # Tracking data structures
    per_image_results = {}
    stage_timings_accum = {
        "read_ms": 0.0,
        "full_frame_infer_ms": 0.0,
        "uncertainty_proposal_ms": 0.0,
        "crop_extraction_ms": 0.0,
        "crop_infer_ms": 0.0,
        "fusion_ms": 0.0,
        "total_ms": 0.0,
    }
    crop_counts = []
    inspected_area_pcts = []
    
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
    vru_size_stats = {
        "Small (<32²)": {"tp": 0, "fn": 0, "total": 0},
        "Medium [32²-96²)": {"tp": 0, "fn": 0, "total": 0},
        "Large (≥96²)": {"tp": 0, "fn": 0, "total": 0},
    }
    total_preds_count = 0

    process = psutil.Process() if HAS_PSUTIL else None
    peak_rss_mb = 0.0

    print("\nStarting inference across 300 images...")
    t_eval_start = time.perf_counter()

    for idx, img_name in enumerate(pilot_subset, start=1):
        img_path = VAL_IMG_DIR / img_name
        if not img_path.exists():
            continue

        # Process image through Foveation Perception Engine
        fused_dets, trace = engine.process_image(str(img_path), conf_threshold=trigger_conf)

        # Operational thresholding
        operational_preds = [
            d for d in fused_dets if d.confidence >= operational_conf
        ]
        total_preds_count += len(operational_preds)

        # Measure inspected area
        crop_proposals = trace.crop_proposals
        union_crop_area = compute_crop_union_area(crop_proposals, 1280, 720)
        inspected_pct = (union_crop_area / (1280.0 * 720.0)) * 100.0

        num_crops = trace.num_crops_executed
        crop_counts.append(num_crops)
        inspected_area_pcts.append(inspected_pct)

        # Stage timings
        t_dict = trace.timing.to_dict()
        for k in stage_timings_accum:
            stage_timings_accum[k] += t_dict.get(k, 0.0)

        # Match against ground truth
        pred_dicts = [
            {"class_name": d.class_name, "conf": d.confidence, "box": list(d.box)}
            for d in operational_preds
        ]
        match_res = match_single_image(pred_dicts, gt_by_image[img_name])

        for c in CLASS_NAMES:
            class_stats[c]["tp"] += match_res["class_stats"][c]["tp"]
            class_stats[c]["fp"] += match_res["class_stats"][c]["fp"]
            class_stats[c]["fn"] += match_res["class_stats"][c]["fn"]

        for sz in ["Small (<32²)", "Medium [32²-96²)", "Large (≥96²)"]:
            vru_size_stats[sz]["tp"] += match_res["vru_size_stats"][sz]["tp"]
            vru_size_stats[sz]["fn"] += match_res["vru_size_stats"][sz]["fn"]
            vru_size_stats[sz]["total"] += match_res["vru_size_stats"][sz]["total"]

        # Track memory
        current_rss = (process.memory_info().rss / (1024.0 * 1024.0)) if process else 0.0
        if current_rss > peak_rss_mb:
            peak_rss_mb = current_rss

        # Record per-image detail
        per_image_results[img_name] = {
            "num_crops_executed": num_crops,
            "crop_proposals": [p.to_dict() for p in trace.crop_proposals],
            "inspected_area_pct": round(inspected_pct, 2),
            "timing_ms": t_dict,
            "num_full_frame_dets": len(trace.full_frame_detections),
            "num_fused_dets": len(fused_dets),
            "num_operational_preds": len(operational_preds),
            "match_stats": match_res,
        }

        # Periodic checkpointing
        if idx % 25 == 0 or idx == n_images:
            elapsed = time.perf_counter() - t_eval_start
            avg_ms = (elapsed / idx) * 1000.0
            print(f"[{idx:3d}/{n_images}] {elapsed:.1f}s elapsed | Avg: {avg_ms:.1f} ms/img | Crops/img: {np.mean(crop_counts):.2f} | Mem: {current_rss:.1f} MB", flush=True)

            # Atomic checkpoint write
            ckpt_data = {
                "images_processed": idx,
                "total_images": n_images,
                "elapsed_seconds": round(elapsed, 2),
                "crop_counts_mean": round(float(np.mean(crop_counts)), 2),
                "inspected_area_pct_mean": round(float(np.mean(inspected_area_pcts)), 2),
                "current_rss_mb": round(current_rss, 1),
                "class_stats": class_stats,
                "vru_size_stats": vru_size_stats,
            }
            tmp_ckpt = CHECKPOINT_JSON.with_suffix(".tmp")
            with open(tmp_ckpt, "w") as f:
                json.dump(ckpt_data, f, indent=2)
            tmp_ckpt.replace(CHECKPOINT_JSON)

    total_eval_time = time.perf_counter() - t_eval_start
    current_tracemalloc_peak = tracemalloc.get_traced_memory()[1] / (1024.0 * 1024.0)
    tracemalloc.stop()

    # Aggregate Final Metrics
    vru_tp = sum(class_stats[c]["tp"] for c in VRU_NAMES)
    vru_fp = sum(class_stats[c]["fp"] for c in VRU_NAMES)
    vru_fn = sum(class_stats[c]["fn"] for c in VRU_NAMES)
    vru_prec = (vru_tp / max(vru_tp + vru_fp, 1)) * 100.0
    vru_rec = (vru_tp / max(vru_tp + vru_fn, 1)) * 100.0
    vru_f1 = (2.0 * vru_prec * vru_rec) / max(vru_prec + vru_rec, 1e-6) / 100.0

    all_tp = sum(class_stats[c]["tp"] for c in CLASS_NAMES)
    all_fp = sum(class_stats[c]["fp"] for c in CLASS_NAMES)
    all_fn = sum(class_stats[c]["fn"] for c in CLASS_NAMES)
    all_prec = (all_tp / max(all_tp + all_fp, 1)) * 100.0
    all_rec = (all_tp / max(all_tp + all_fn, 1)) * 100.0
    all_f1 = (2.0 * all_prec * all_rec) / max(all_prec + all_rec, 1e-6) / 100.0

    small_vru_tp = vru_size_stats["Small (<32²)"]["tp"]
    small_vru_tot = vru_size_stats["Small (<32²)"]["total"]
    small_vru_rec = (small_vru_tp / max(small_vru_tot, 1)) * 100.0

    per_class_metrics = {}
    for c in CLASS_NAMES:
        tp = class_stats[c]["tp"]
        fp = class_stats[c]["fp"]
        fn = class_stats[c]["fn"]
        p = (tp / max(tp + fp, 1)) * 100.0
        r = (tp / max(tp + fn, 1)) * 100.0
        f1 = (2.0 * p * r) / max(p + r, 1e-6) / 100.0
        per_class_metrics[c] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 2),
            "recall": round(r, 2),
            "f1_score": round(f1, 4),
        }

    avg_stage_timings = {k: round(v / n_images, 2) for k, v in stage_timings_accum.items()}
    avg_total_ms = avg_stage_timings["total_ms"]
    throughput_fps = 1000.0 / max(avg_total_ms, 1e-3)

    # Load 1280px reference data from phase8_pilot_300_native_res_comparison.json
    ref_path = REPO_ROOT / "experiments" / "phase8_pilot_300_native_res_comparison.json"
    with open(ref_path, "r") as f:
        ref_data = json.load(f)

    ref_1280 = ref_data["full_report"]
    ref_1280_vru = ref_1280["vru_metrics"]

    # Build 3-Way Side-by-Side Comparison Matrix
    comparison = {
        "cohort": "300-Image Development Pilot",
        "checkpoint_sha256": actual_sha,
        "evaluation_scope": "Identical 300-image development cohort, zero test-split leakage",
        "configurations": {
            "A_baseline_960px": {
                "name": "Full-Frame 960px Baseline",
                "resolution": 960,
                "strategy": "Single-pass full-frame inference",
                "latency_per_img_ms": 273.0,
                "throughput_fps": 3.66,
                "mean_crops_per_image": 0.0,
                "inspected_area_pct": 100.0,
                "vru_recall": 60.46,
                "vru_precision": 78.45,
                "vru_f1": 0.6829,
                "small_vru_recall": 42.73,
                "small_vru_tp": 385,
                "small_vru_total": 901,
                "all_class_recall": 71.88,
                "all_class_precision": 77.93,
                "all_class_f1": 0.7478,
                "per_class_recall": {
                    "person": 61.91,
                    "rider": 55.17,
                    "bike": 51.64,
                    "motor": 34.38,
                }
            },
            "B_reference_1280px": {
                "name": "Full-Frame 1280px Reference",
                "resolution": 1280,
                "strategy": "Single-pass high-resolution inference",
                "latency_per_img_ms": 1051.9,
                "throughput_fps": 0.95,
                "mean_crops_per_image": 0.0,
                "inspected_area_pct": 100.0,
                "vru_recall": 61.84,
                "vru_precision": 78.20,
                "vru_f1": 0.6906,
                "small_vru_recall": 45.17,
                "small_vru_tp": 407,
                "small_vru_total": 901,
                "all_class_recall": 73.64,
                "all_class_precision": 77.68,
                "all_class_f1": 0.7561,
                "per_class_recall": {
                    "person": 63.22,
                    "rider": 56.90,
                    "bike": 55.74,
                    "motor": 28.12,
                }
            },
            "C_active_foveation_960px": {
                "name": "Active Foveation (960px + Crops)",
                "resolution": "960px base + 640px crops",
                "strategy": "Uncertainty-triggered targeted inspection (K<=2 crops)",
                "latency_per_img_ms": avg_total_ms,
                "throughput_fps": round(throughput_fps, 2),
                "mean_crops_per_image": round(float(np.mean(crop_counts)), 2),
                "crop_distribution": {
                    "0_crops": int(sum(1 for c in crop_counts if c == 0)),
                    "1_crop": int(sum(1 for c in crop_counts if c == 1)),
                    "2_crops": int(sum(1 for c in crop_counts if c == 2)),
                },
                "mean_inspected_crop_area_pct": round(float(np.mean(inspected_area_pcts)), 2),
                "stage_timings_ms": avg_stage_timings,
                "peak_memory_rss_mb": round(peak_rss_mb, 1),
                "tracemalloc_peak_mb": round(current_tracemalloc_peak, 1),
                "vru_recall": round(vru_rec, 2),
                "vru_precision": round(vru_prec, 2),
                "vru_f1": round(vru_f1, 4),
                "small_vru_recall": round(small_vru_rec, 2),
                "small_vru_tp": small_vru_tp,
                "small_vru_total": small_vru_tot,
                "all_class_recall": round(all_rec, 2),
                "all_class_precision": round(all_prec, 2),
                "all_class_f1": round(all_f1, 4),
                "per_class_recall": {
                    c: per_class_metrics[c]["recall"] for c in ["person", "rider", "bike", "motor"]
                },
                "per_class_full": per_class_metrics,
            }
        },
        "deltas_vs_960px_baseline": {
            "vru_recall_delta": round(vru_rec - 60.46, 2),
            "vru_precision_delta": round(vru_prec - 78.45, 2),
            "vru_f1_delta": round(vru_f1 - 0.6829, 4),
            "small_vru_recall_delta": round(small_vru_rec - 42.73, 2),
            "small_vru_tp_gain": small_vru_tp - 385,
            "latency_multiplier": round(avg_total_ms / 273.0, 2),
        },
        "deltas_vs_1280px_reference": {
            "vru_recall_gap": round(vru_rec - 61.84, 2),
            "small_vru_recall_gap": round(small_vru_rec - 45.17, 2),
            "small_vru_tp_gap": small_vru_tp - 407,
            "latency_reduction_factor": round(1051.9 / max(avg_total_ms, 1e-3), 2),
            "latency_savings_pct": round((1051.9 - avg_total_ms) / 1051.9 * 100.0, 1),
        }
    }

    # Save outputs
    print("\nSaving per-image results and 3-way comparison matrix...")
    with open(PER_IMAGE_JSON, "w") as f:
        json.dump(per_image_results, f, indent=2)
    with open(COMPARISON_JSON, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"Artifact saved: {PER_IMAGE_JSON}")
    print(f"Artifact saved: {COMPARISON_JSON}")

    # Print Final Summary Table
    print("\n" + "=" * 80)
    print("PHASE 8 ACTIVE FOVEATION EVALUATION: SUMMARY COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<28} | {'960px Baseline':<15} | {'1280px Reference':<16} | {'960px + Foveation':<17}")
    print("-" * 80)
    print(f"{'VRU Recall':<28} | {'60.46%':<15} | {'61.84%':<16} | {vru_rec:.2f}%")
    print(f"{'VRU Precision':<28} | {'78.45%':<15} | {'78.20%':<16} | {vru_prec:.2f}%")
    print(f"{'VRU F1-Score':<28} | {'0.6829':<15} | {'0.6906':<16} | {vru_f1:.4f}")
    print(f"{'Small-VRU Recall':<28} | {'42.73%':<15} | {'45.17%':<16} | {small_vru_rec:.2f}%")
    print(f"{'Small-VRU TP / Total':<28} | {'385 / 901':<15} | {'407 / 901':<16} | {small_vru_tp} / {small_vru_tot}")
    print(f"{'All-Class Recall':<28} | {'71.88%':<15} | {'73.64%':<16} | {all_rec:.2f}%")
    print(f"{'All-Class Precision':<28} | {'77.93%':<15} | {'77.68%':<16} | {all_prec:.2f}%")
    print(f"{'All-Class F1-Score':<28} | {'0.7478':<15} | {'0.7561':<16} | {all_f1:.4f}")
    print("-" * 80)
    print(f"{'Person Recall':<28} | {'61.91%':<15} | {'63.22%':<16} | {per_class_metrics['person']['recall']:.2f}%")
    print(f"{'Rider Recall':<28} | {'55.17%':<15} | {'56.90%':<16} | {per_class_metrics['rider']['recall']:.2f}%")
    print(f"{'Bike Recall':<28} | {'51.64%':<15} | {'55.74%':<16} | {per_class_metrics['bike']['recall']:.2f}%")
    print(f"{'Motor Recall':<28} | {'34.38%':<15} | {'28.12%':<16} | {per_class_metrics['motor']['recall']:.2f}%")
    print("-" * 80)
    print(f"{'Latency per Image (ms)':<28} | {'273.0 ms':<15} | {'1051.9 ms':<16} | {avg_total_ms:.1f} ms")
    print(f"{'Throughput (FPS)':<28} | {'3.66 fps':<15} | {'0.95 fps':<16} | {throughput_fps:.2f} fps")
    print(f"{'Mean Crops per Image':<28} | {'0.00':<15} | {'0.00':<16} | {np.mean(crop_counts):.2f}")
    print(f"{'Inspected Crop Area (%)':<28} | {'0.0%':<15} | {'100.0% (full)':<16} | {np.mean(inspected_area_pcts):.1f}%")
    print(f"{'Peak Memory RSS (MB)':<28} | {'N/A':<15} | {'~580 MB':<16} | {peak_rss_mb:.1f} MB")
    print("=" * 80)

    return comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Phase 8 Active Foveation Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for dry-run testing")
    parser.add_argument("--max_crops", type=int, default=2, help="Max crops per image")
    parser.add_argument("--crop_imgsz", type=int, default=640, help="Crop resolution")
    parser.add_argument("--operational_conf", type=float, default=0.25, help="Operational confidence threshold")
    parser.add_argument("--trigger_conf", type=float, default=0.15, help="Trigger confidence threshold")
    args = parser.parse_args()

    # Pass limit if provided
    run_evaluation(
        max_crops=args.max_crops,
        crop_imgsz=args.crop_imgsz,
        operational_conf=args.operational_conf,
        trigger_conf=args.trigger_conf,
        limit=args.limit
    )
