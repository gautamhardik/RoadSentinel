"""
RoadSentinel Phase 7 — Controlled CPU Latency Revalidation Protocol
Benchmark verifying latency stability across:
1. Pipeline 1: Raw Full-Frame YOLO11m (imgsz=960)
2. Pipeline 2: Autonomous Horizon Slicing + Calibrated WBF (EXP-7B)
3. Pipeline 3: Active Foveation + Calibrated WBF (Incumbent Baseline)

Controlled Protocol:
- Identical fixed image cohort: 15 development images from the 300 pilot set
- Separate cold-start measurements (3 warm-up frames logged separately)
- 2 repetitions with alternating pipeline execution order (Latin-square inspired):
    Run 1: Raw -> Horizon Slicing -> Active Foveation
    Run 2: Active Foveation -> Horizon Slicing -> Raw
- Explicit CPU telemetry: psutil CPU frequency (MHz) and utilization (%) before/after each pass
- Per-image timing arrays and exact percentile calculations from individual observations (no replicated batch averages)
- End-to-end wall clock timing and true throughput calculation
- Preserves all historical artifacts; outputs to experiments/phase7_ablations/phase7_controlled_latency_report.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, os.getcwd())

import time
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import cv2
import psutil
import torch
torch.set_num_threads(16)
from ultralytics import YOLO

from kaggle_jobs.calibration_benchmark.evaluate_phase4c_wbf import (
    PILOT_COHORT_IMAGE_SET,
    generate_uncertainty_crops,
    calibrated_wbf_consensus_fusion,
)
from src.foveation.horizon_slicer import HorizonSlicer
from src.foveation.cropper import ActiveCropper
from src.foveation.types import FoveationConfig


def get_cpu_telemetry() -> Dict[str, Any]:
    freq = psutil.cpu_freq()
    return {
        "current_freq_mhz": round(freq.current, 1) if freq else None,
        "min_freq_mhz": round(freq.min, 1) if freq else None,
        "max_freq_mhz": round(freq.max, 1) if freq else None,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "temperature_sensor_available": False,  # Not exposed via psutil on Windows
    }


def compute_distribution_stats(times_ms: List[float]) -> Dict[str, Any]:
    arr = np.array(times_ms, dtype=np.float64)
    return {
        "sample_count": len(times_ms),
        "mean_ms": round(float(np.mean(arr)), 1),
        "std_ms": round(float(np.std(arr)), 1),
        "min_ms": round(float(np.min(arr)), 1),
        "p25_ms": round(float(np.percentile(arr, 25)), 1),
        "p50_ms": round(float(np.median(arr)), 1),
        "p75_ms": round(float(np.percentile(arr, 75)), 1),
        "p90_ms": round(float(np.percentile(arr, 90)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
        "max_ms": round(float(np.max(arr)), 1),
    }


def run_raw_inference(model, val_img_dir: Path, image_names: List[str]) -> Dict[str, Any]:
    e2e_times = []
    t_start_total = time.perf_counter()
    for img_name in image_names:
        p = str(val_img_dir / img_name)
        t0 = time.perf_counter()
        _ = model.predict(p, imgsz=960, conf=0.01, device="cpu", verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000.0
        e2e_times.append(dt)
    t_end_total = time.perf_counter()
    wall_clock_s = t_end_total - t_start_total
    return {
        "per_image_times_ms": [round(t, 1) for t in e2e_times],
        "wall_clock_total_s": round(wall_clock_s, 3),
        "throughput_fps": round(len(image_names) / max(wall_clock_s, 1e-6), 3),
        "stats": compute_distribution_stats(e2e_times),
    }


def run_horizon_slicing(model, slicer: HorizonSlicer, val_img_dir: Path, image_names: List[str]) -> Dict[str, Any]:
    e2e_times = []
    slice_ext_times = []
    slice_inf_times = []
    wbf_times = []
    t_start_total = time.perf_counter()

    for img_name in image_names:
        p = str(val_img_dir / img_name)
        t0 = time.perf_counter()

        # Step A: Full-Frame pass
        ff_res = model.predict(p, imgsz=960, conf=0.01, device="cpu", verbose=False)[0]
        ff_dets = []
        for b in ff_res.boxes:
            ff_dets.append({
                "class_name": model.names[int(b.cls.item())],
                "conf": float(b.conf.item()),
                "box": b.xyxy[0].cpu().numpy(),
                "source": "full_frame",
                "img_name": img_name,
            })

        # Step B: Horizon Slice extraction
        t_sl0 = time.perf_counter()
        frame = cv2.imread(p)
        sub_img, s_coords = slicer.extract_slice(frame)
        dt_sl = (time.perf_counter() - t_sl0) * 1000.0
        slice_ext_times.append(dt_sl)

        # Step C: Horizon Slice inference & remapping
        t_inf0 = time.perf_counter()
        slice_res = model.predict(sub_img, imgsz=960, conf=0.05, device="cpu", verbose=False)[0]
        hz_dets = []
        for b in slice_res.boxes:
            cname = model.names[int(b.cls.item())]
            conf = float(b.conf.item())
            coords = b.xyxy[0].cpu().numpy()
            remapped = slicer.remap_detection(tuple(coords), s_coords, (720, 1280), conf=conf)
            if remapped is not None:
                hz_dets.append({
                    "img_name": img_name,
                    "class_name": cname,
                    "conf": conf,
                    "box": remapped,
                    "source": "crop",
                })
        dt_inf = (time.perf_counter() - t_inf0) * 1000.0
        slice_inf_times.append(dt_inf)

        # Step D: WBF
        t_wbf0 = time.perf_counter()
        _ = calibrated_wbf_consensus_fusion(
            full_frame_dets=ff_dets,
            crop_dets=hz_dets,
            tau_ff=0.25,
            tau_isolated=0.35,
            tau_confirmed=0.15,
            iou_assoc=0.30,
        )
        dt_wbf = (time.perf_counter() - t_wbf0) * 1000.0
        wbf_times.append(dt_wbf)

        dt_total = (time.perf_counter() - t0) * 1000.0
        e2e_times.append(dt_total)

    t_end_total = time.perf_counter()
    wall_clock_s = t_end_total - t_start_total

    return {
        "per_image_times_ms": [round(t, 1) for t in e2e_times],
        "wall_clock_total_s": round(wall_clock_s, 3),
        "throughput_fps": round(len(image_names) / max(wall_clock_s, 1e-6), 3),
        "stats": compute_distribution_stats(e2e_times),
        "stage_breakdown_mean_ms": {
            "slice_extraction": round(float(np.mean(slice_ext_times)), 1),
            "slice_inference": round(float(np.mean(slice_inf_times)), 1),
            "wbf_fusion": round(float(np.mean(wbf_times)), 1),
        },
    }


def run_active_foveation(model, cropper: ActiveCropper, val_img_dir: Path, image_names: List[str]) -> Dict[str, Any]:
    e2e_times = []
    crop_counts = []
    crop_proc_times = []
    t_start_total = time.perf_counter()
    img_w, img_h = 1280.0, 720.0

    for img_name in image_names:
        p = str(val_img_dir / img_name)
        t0 = time.perf_counter()

        ff_res = model.predict(p, imgsz=960, conf=0.01, device="cpu", verbose=False)[0]
        ff_dets = []
        for b in ff_res.boxes:
            ff_dets.append({
                "class_name": model.names[int(b.cls.item())],
                "conf": float(b.conf.item()),
                "box": b.xyxy[0].cpu().numpy(),
                "source": "full_frame",
                "img_name": img_name,
            })

        frame = cv2.imread(p)
        proposals = generate_uncertainty_crops(ff_dets, int(img_w), int(img_h), max_crops=2)
        crop_counts.append(len(proposals))

        crop_dets = []
        t_c0 = time.perf_counter()
        for prop in proposals:
            sub_img, int_coords = cropper.extract_crop_image(frame, prop["crop_box"])
            if sub_img.shape[0] < 8 or sub_img.shape[1] < 8:
                continue
            c_res = model.predict(sub_img, imgsz=640, conf=0.01, device="cpu", verbose=False)[0]
            cx1, cy1, cx2, cy2 = int_coords
            for b in c_res.boxes:
                cname = model.names[int(b.cls.item())]
                conf = float(b.conf.item())
                rx1, ry1, rx2, ry2 = b.xyxy[0].cpu().numpy()
                ox1 = max(0.0, min(img_w, cx1 + rx1))
                oy1 = max(0.0, min(img_h, cy1 + ry1))
                ox2 = max(0.0, min(img_w, cx1 + rx2))
                oy2 = max(0.0, min(img_h, cy1 + ry2))
                crop_dets.append({
                    "img_name": img_name,
                    "class_name": cname,
                    "conf": conf,
                    "box": np.array([ox1, oy1, ox2, oy2], dtype=np.float32),
                    "source": "crop",
                })
        dt_c = (time.perf_counter() - t_c0) * 1000.0
        crop_proc_times.append(dt_c)

        _ = calibrated_wbf_consensus_fusion(
            full_frame_dets=ff_dets,
            crop_dets=crop_dets,
            tau_ff=0.25,
            tau_isolated=0.35,
            tau_confirmed=0.15,
            iou_assoc=0.30,
        )
        dt_total = (time.perf_counter() - t0) * 1000.0
        e2e_times.append(dt_total)

    t_end_total = time.perf_counter()
    wall_clock_s = t_end_total - t_start_total

    return {
        "per_image_times_ms": [round(t, 1) for t in e2e_times],
        "wall_clock_total_s": round(wall_clock_s, 3),
        "throughput_fps": round(len(image_names) / max(wall_clock_s, 1e-6), 3),
        "mean_crops_per_frame": round(float(np.mean(crop_counts)), 2),
        "crop_processing_mean_ms": round(float(np.mean(crop_proc_times)), 1),
        "stats": compute_distribution_stats(e2e_times),
    }


def execute_controlled_latency_benchmark(num_eval_images: int = 15, num_warmup: int = 3):
    ckpt_path = Path("experiments/phase6_session2_error/runs/phase6_yolo11m_960_session1/weights/best.pt")
    val_img_dir = Path("bdd100k_dataset/val/images")
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} not found"
    assert val_img_dir.exists(), f"Validation dir {val_img_dir} not found"

    model = YOLO(str(ckpt_path))
    pilot_images = sorted(list(PILOT_COHORT_IMAGE_SET))
    
    warmup_images = pilot_images[:num_warmup]
    eval_images = pilot_images[num_warmup:num_warmup + num_eval_images]

    slicer = HorizonSlicer()
    cropper = ActiveCropper(FoveationConfig())

    print("=" * 75)
    print("RoadSentinel Phase 7 — Controlled CPU Latency Revalidation")
    print(f"Warmup Images: {len(warmup_images)}, Fixed Eval Images: {len(eval_images)}")
    print("=" * 75)

    # ------------------------------------------------------------------------
    # Phase A: Cold-Start Logging & Warm-Up
    # ------------------------------------------------------------------------
    print("\nPhase A: Measuring Cold-Start Warm-Up Passes...")
    telemetry_start = get_cpu_telemetry()
    cold_start_raw_times = []
    for img_name in warmup_images:
        p = str(val_img_dir / img_name)
        t0 = time.perf_counter()
        _ = model.predict(p, imgsz=960, conf=0.01, device="cpu", verbose=False)
        cold_start_raw_times.append((time.perf_counter() - t0) * 1000.0)

    # Warmup Slicer and Cropper structures
    for img_name in warmup_images:
        p = str(val_img_dir / img_name)
        frame = cv2.imread(p)
        sub_img, _ = slicer.extract_slice(frame)
        _ = model.predict(sub_img, imgsz=960, conf=0.05, device="cpu", verbose=False)
        sub_c, _ = cropper.extract_crop_image(frame, (100, 100, 400, 400))
        _ = model.predict(sub_c, imgsz=640, conf=0.01, device="cpu", verbose=False)

    cold_start_data = {
        "images": warmup_images,
        "per_image_times_ms": [round(t, 1) for t in cold_start_raw_times],
        "mean_ms": round(float(np.mean(cold_start_raw_times)), 1),
    }

    # ------------------------------------------------------------------------
    # Phase B: Repetition 1 (Order: Raw -> Horizon Slicing -> Active Foveation)
    # ------------------------------------------------------------------------
    print("\nPhase B: Executing Run 1 (Order: Raw -> Horizon Slicing -> Active Foveation)...")
    run1_telemetry_before = get_cpu_telemetry()

    print("  Run 1 - Pipeline 1: Raw Full-Frame...")
    run1_raw = run_raw_inference(model, val_img_dir, eval_images)

    print("  Run 1 - Pipeline 2: Horizon Slicing + WBF...")
    run1_horizon = run_horizon_slicing(model, slicer, val_img_dir, eval_images)

    print("  Run 1 - Pipeline 3: Active Foveation + WBF...")
    run1_fov = run_active_foveation(model, cropper, val_img_dir, eval_images)

    run1_telemetry_after = get_cpu_telemetry()

    # ------------------------------------------------------------------------
    # Phase C: Repetition 2 (Order: Active Foveation -> Horizon Slicing -> Raw)
    # ------------------------------------------------------------------------
    print("\nPhase C: Executing Run 2 (Order: Active Foveation -> Horizon Slicing -> Raw)...")
    run2_telemetry_before = get_cpu_telemetry()

    print("  Run 2 - Pipeline 3: Active Foveation + WBF...")
    run2_fov = run_active_foveation(model, cropper, val_img_dir, eval_images)

    print("  Run 2 - Pipeline 2: Horizon Slicing + WBF...")
    run2_horizon = run_horizon_slicing(model, slicer, val_img_dir, eval_images)

    print("  Run 2 - Pipeline 1: Raw Full-Frame...")
    run2_raw = run_raw_inference(model, val_img_dir, eval_images)

    run2_telemetry_after = get_cpu_telemetry()

    # ------------------------------------------------------------------------
    # Phase D: Pooled Analysis (N = 30 observations per pipeline across 2 runs)
    # ------------------------------------------------------------------------
    pooled_raw_times = run1_raw["per_image_times_ms"] + run2_raw["per_image_times_ms"]
    pooled_horizon_times = run1_horizon["per_image_times_ms"] + run2_horizon["per_image_times_ms"]
    pooled_fov_times = run1_fov["per_image_times_ms"] + run2_fov["per_image_times_ms"]

    pooled_results = {
        "benchmark_metadata": {
            "target_checkpoint": "experiments/phase6_session2_error/runs/phase6_yolo11m_960_session1/weights/best.pt",
            "model_architecture": "YOLO11m",
            "eval_image_count_per_run": len(eval_images),
            "repetition_count": 2,
            "total_observations_per_pipeline": len(pooled_raw_times),
            "eval_image_subset": eval_images,
            "torch_num_threads": 16,
            "cpu_telemetry": {
                "initial": telemetry_start,
                "run1_before": run1_telemetry_before,
                "run1_after": run1_telemetry_after,
                "run2_before": run2_telemetry_before,
                "run2_after": run2_telemetry_after,
            },
        },
        "cold_start_warmup_measurements": cold_start_data,
        "run1_order_raw_horizon_foveation": {
            "raw_full_frame": run1_raw,
            "horizon_slicing_wbf": run1_horizon,
            "active_foveation_wbf": run1_fov,
        },
        "run2_order_foveation_horizon_raw": {
            "active_foveation_wbf": run2_fov,
            "horizon_slicing_wbf": run2_horizon,
            "raw_full_frame": run2_raw,
        },
        "pooled_distribution_summary": {
            "raw_full_frame": {
                "distribution_stats": compute_distribution_stats(pooled_raw_times),
                "total_wall_clock_s": round(run1_raw["wall_clock_total_s"] + run2_raw["wall_clock_total_s"], 3),
                "combined_throughput_fps": round(
                    (len(pooled_raw_times)) / max(run1_raw["wall_clock_total_s"] + run2_raw["wall_clock_total_s"], 1e-6), 3
                ),
                "inter_run_drift_pct": round(
                    ((run2_raw["stats"]["mean_ms"] - run1_raw["stats"]["mean_ms"]) / run1_raw["stats"]["mean_ms"]) * 100.0, 2
                ),
            },
            "horizon_slicing_wbf": {
                "distribution_stats": compute_distribution_stats(pooled_horizon_times),
                "total_wall_clock_s": round(run1_horizon["wall_clock_total_s"] + run2_horizon["wall_clock_total_s"], 3),
                "combined_throughput_fps": round(
                    (len(pooled_horizon_times)) / max(run1_horizon["wall_clock_total_s"] + run2_horizon["wall_clock_total_s"], 1e-6), 3
                ),
                "inter_run_drift_pct": round(
                    ((run2_horizon["stats"]["mean_ms"] - run1_horizon["stats"]["mean_ms"]) / run1_horizon["stats"]["mean_ms"]) * 100.0, 2
                ),
                "stage_breakdown_mean_ms": {
                    "slice_extraction": round(
                        (run1_horizon["stage_breakdown_mean_ms"]["slice_extraction"] + run2_horizon["stage_breakdown_mean_ms"]["slice_extraction"]) / 2.0, 1
                    ),
                    "slice_inference": round(
                        (run1_horizon["stage_breakdown_mean_ms"]["slice_inference"] + run2_horizon["stage_breakdown_mean_ms"]["slice_inference"]) / 2.0, 1
                    ),
                    "wbf_fusion": round(
                        (run1_horizon["stage_breakdown_mean_ms"]["wbf_fusion"] + run2_horizon["stage_breakdown_mean_ms"]["wbf_fusion"]) / 2.0, 1
                    ),
                },
            },
            "active_foveation_wbf": {
                "distribution_stats": compute_distribution_stats(pooled_fov_times),
                "total_wall_clock_s": round(run1_fov["wall_clock_total_s"] + run2_fov["wall_clock_total_s"], 3),
                "combined_throughput_fps": round(
                    (len(pooled_fov_times)) / max(run1_fov["wall_clock_total_s"] + run2_fov["wall_clock_total_s"], 1e-6), 3
                ),
                "inter_run_drift_pct": round(
                    ((run2_fov["stats"]["mean_ms"] - run1_fov["stats"]["mean_ms"]) / run1_fov["stats"]["mean_ms"]) * 100.0, 2
                ),
                "mean_crops_per_frame": round((run1_fov["mean_crops_per_frame"] + run2_fov["mean_crops_per_frame"]) / 2.0, 2),
                "crop_processing_mean_ms": round(
                    (run1_fov["crop_processing_mean_ms"] + run2_fov["crop_processing_mean_ms"]) / 2.0, 1
                ),
            },
        },
    }

    out_p = Path("experiments/phase7_ablations/phase7_controlled_latency_report.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(pooled_results, f, indent=2)

    print("\n" + "=" * 75)
    print("CONTROLLED BENCHMARK COMPLETE")
    print("=" * 75)
    print(f"Artifact written to {out_p}")
    print(json.dumps(pooled_results["pooled_distribution_summary"], indent=2))
    return pooled_results


if __name__ == "__main__":
    execute_controlled_latency_benchmark(num_eval_images=15, num_warmup=3)
