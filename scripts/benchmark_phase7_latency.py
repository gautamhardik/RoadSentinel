"""
RoadSentinel Phase 7 — Matched Latency Benchmark
Compares end-to-end CPU latency under identical conditions for:
1. Pipeline 1: Raw Full-Frame YOLO11m (imgsz=960)
2. Pipeline 2: Autonomous Horizon Slicing + Calibrated WBF (EXP-7B)
3. Pipeline 3: Active Foveation + Calibrated WBF (Uncertainty Crops)

Hardware: Multi-core CPU (16 threads, PyTorch oneDNN)
Warm-up: 2 frames discarded
Timing: High-precision perf_counter around all preprocessing, inference, remapping, and WBF
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, os.getcwd())

import time
import json
from pathlib import Path
import numpy as np
import cv2
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


def run_matched_latency_benchmark(num_eval_images: int = 10, num_warmup: int = 2):
    ckpt_path = Path("experiments/phase6_session2_error/runs/phase6_yolo11m_960_session1/weights/best.pt")
    val_img_dir = Path("bdd100k_dataset/val/images")
    assert ckpt_path.exists()
    assert val_img_dir.exists()

    model = YOLO(str(ckpt_path))
    pilot_images = sorted(list(PILOT_COHORT_IMAGE_SET))
    sample_images = pilot_images[:num_warmup + num_eval_images]

    slicer = HorizonSlicer()
    cropper = ActiveCropper(FoveationConfig())
    img_w, img_h = 1280.0, 720.0

    print(f"Phase 7 Latency Benchmark: {num_warmup} warm-up, {num_eval_images} eval frames.")

    # 1. Warm-up
    for w_img in sample_images[:num_warmup]:
        p = str(val_img_dir / w_img)
        _ = model.predict(p, imgsz=960, conf=0.01, device="cpu", verbose=False)
        frame = cv2.imread(p)
        sub_img, _ = slicer.extract_slice(frame)
        _ = model.predict(sub_img, imgsz=960, conf=0.01, device="cpu", verbose=False)

    eval_images = sample_images[num_warmup:num_warmup + num_eval_images]

    # Benchmark 1: Raw Full-Frame
    print("Benchmarking 1: Raw Full-Frame...")
    raw_times = []
    raw_preds_dict = {}
    for img_name in eval_images:
        p = str(val_img_dir / img_name)
        t0 = time.perf_counter()
        res = model.predict(p, imgsz=960, conf=0.01, device="cpu", verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000.0
        raw_times.append(dt)

        preds = []
        for b in res.boxes:
            preds.append({
                "class_name": model.names[int(b.cls.item())],
                "conf": float(b.conf.item()),
                "box": b.xyxy[0].cpu().numpy(),
                "source": "full_frame",
                "img_name": img_name,
            })
        raw_preds_dict[img_name] = preds

    # Benchmark 2: Autonomous Horizon Slicing + Calibrated WBF (EXP-7B)
    print("Benchmarking 2: Autonomous Horizon Slicing + WBF...")
    horizon_e2e_times = []
    horizon_slice_times = []
    horizon_inf_times = []
    horizon_wbf_times = []

    for img_name in eval_images:
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
        t_sl_start = time.perf_counter()
        frame = cv2.imread(p)
        sub_img, s_coords = slicer.extract_slice(frame)
        t_sl_end = time.perf_counter()
        horizon_slice_times.append((t_sl_end - t_sl_start) * 1000.0)

        # Step C: Horizon Slice inference
        t_inf_start = time.perf_counter()
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
        t_inf_end = time.perf_counter()
        horizon_inf_times.append((t_inf_end - t_inf_start) * 1000.0)

        # Step D: WBF
        t_wbf_start = time.perf_counter()
        _ = calibrated_wbf_consensus_fusion(
            full_frame_dets=ff_dets,
            crop_dets=hz_dets,
            tau_ff=0.25,
            tau_isolated=0.35,
            tau_confirmed=0.15,
            iou_assoc=0.30,
        )
        t_wbf_end = time.perf_counter()
        horizon_wbf_times.append((t_wbf_end - t_wbf_start) * 1000.0)

        dt_total = (time.perf_counter() - t0) * 1000.0
        horizon_e2e_times.append(dt_total)

    # Benchmark 3: Active Foveation + Calibrated WBF (Baseline)
    print("Benchmarking 3: Active Foveation + WBF...")
    fov_e2e_times = []
    fov_crop_times = []
    crops_count = []

    for img_name in eval_images:
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
        crops_count.append(len(proposals))

        crop_dets = []
        t_c_start = time.perf_counter()
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
        t_c_end = time.perf_counter()
        fov_crop_times.append((t_c_end - t_c_start) * 1000.0)

        _ = calibrated_wbf_consensus_fusion(
            full_frame_dets=ff_dets,
            crop_dets=crop_dets,
            tau_ff=0.25,
            tau_isolated=0.35,
            tau_confirmed=0.15,
            iou_assoc=0.30,
        )
        dt_total = (time.perf_counter() - t0) * 1000.0
        fov_e2e_times.append(dt_total)

    results = {
        "hardware": "Local multi-core CPU (16 threads, PyTorch oneDNN)",
        "num_evaluated_frames": num_eval_images,
        "warmup_frames": num_warmup,
        "raw_full_frame": {
            "mean_ms": round(float(np.mean(raw_times)), 1),
            "p50_ms": round(float(np.median(raw_times)), 1),
            "p95_ms": round(float(np.percentile(raw_times, 95)), 1),
            "fps": round(1000.0 / float(np.mean(raw_times)), 2),
        },
        "horizon_slicing_wbf": {
            "mean_ms": round(float(np.mean(horizon_e2e_times)), 1),
            "p50_ms": round(float(np.median(horizon_e2e_times)), 1),
            "p95_ms": round(float(np.percentile(horizon_e2e_times, 95)), 1),
            "fps": round(1000.0 / float(np.mean(horizon_e2e_times)), 2),
            "slice_extraction_mean_ms": round(float(np.mean(horizon_slice_times)), 1),
            "slice_inference_mean_ms": round(float(np.mean(horizon_inf_times)), 1),
            "wbf_mean_ms": round(float(np.mean(horizon_wbf_times)), 1),
        },
        "active_foveation_wbf": {
            "mean_ms": round(float(np.mean(fov_e2e_times)), 1),
            "p50_ms": round(float(np.median(fov_e2e_times)), 1),
            "p95_ms": round(float(np.percentile(fov_e2e_times, 95)), 1),
            "fps": round(1000.0 / float(np.mean(fov_e2e_times)), 2),
            "mean_crops_per_frame": round(float(np.mean(crops_count)), 2),
            "crop_processing_mean_ms": round(float(np.mean(fov_crop_times)), 1),
        },
    }

    out_p = Path("experiments/phase7_ablations/phase7_latency_comparison.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 7 MATCHED LATENCY BENCHMARK RESULTS")
    print("=" * 70)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_matched_latency_benchmark(num_eval_images=10, num_warmup=2)
