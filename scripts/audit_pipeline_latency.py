"""
RoadSentinel Phase 6 — Latency Protocol Audit
Measures true end-to-end timing for:
1. Raw YOLO11m (Batch-1, imgsz=960)
2. Active Foveation + Calibrated WBF (Batch-1, true end-to-end: FullFrame + Crops + WBF)
3. Batched Raw YOLO11m (Batch-16, imgsz=960)

Uses:
- Identical hardware: Local multi-core CPU (16 threads)
- Explicit warm-up: 3 images discarded
- Identical image order: sorted validation subset
- Strict stage-by-stage timing boundaries
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
from src.foveation.cropper import ActiveCropper
from src.foveation.types import FoveationConfig


def run_latency_audit(num_eval_images: int = 30, num_warmup: int = 3):
    ckpt_path = Path("experiments/phase6_session2_error/runs/phase6_yolo11m_960_session1/weights/best.pt")
    val_img_dir = Path("bdd100k_dataset/val/images")
    assert ckpt_path.exists()
    assert val_img_dir.exists()

    model = YOLO(str(ckpt_path))
    pilot_images = sorted(list(PILOT_COHORT_IMAGE_SET))
    sample_images = pilot_images[:num_warmup + num_eval_images]

    foveation_config = FoveationConfig()
    cropper = ActiveCropper(foveation_config)
    img_w, img_h = 1280.0, 720.0

    print(f"Starting Latency Audit: {num_warmup} warm-up frames, {num_eval_images} evaluation frames.")

    # -------------------------------------------------------------
    # 1. Warm-up
    # -------------------------------------------------------------
    print("Executing warm-up passes...")
    for w_img in sample_images[:num_warmup]:
        p = str(val_img_dir / w_img)
        _ = model.predict(p, imgsz=960, conf=0.01, device="cpu", verbose=False)
        frame = cv2.imread(p)
        _ = model.predict(frame[100:400, 200:600], imgsz=640, conf=0.01, device="cpu", verbose=False)

    eval_images = sample_images[num_warmup:num_warmup + num_eval_images]

    # -------------------------------------------------------------
    # 2. Pipeline A: Raw YOLO11m Batch-1 Sequential
    # -------------------------------------------------------------
    print("Benchmarking Pipeline A: Raw YOLO11m Batch-1 Sequential...")
    raw_b1_times = []
    raw_b1_preds = {}

    for img_name in eval_images:
        p = str(val_img_dir / img_name)
        t0 = time.perf_counter()
        res = model.predict(p, imgsz=960, conf=0.01, device="cpu", verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000.0
        raw_b1_times.append(dt)

        preds = []
        for b in res.boxes:
            preds.append({
                "class_name": model.names[int(b.cls.item())],
                "conf": float(b.conf.item()),
                "box": b.xyxy[0].cpu().numpy(),
                "source": "full_frame",
                "img_name": img_name,
            })
        raw_b1_preds[img_name] = preds

    # -------------------------------------------------------------
    # 3. Pipeline B: Active Foveation + Calibrated WBF Batch-1 End-to-End
    # -------------------------------------------------------------
    print("Benchmarking Pipeline B: Active Foveation + Calibrated WBF True End-to-End...")
    fov_stage_ff_times = []
    fov_stage_cropgen_times = []
    fov_stage_cropinf_times = []
    fov_stage_wbf_times = []
    fov_total_e2e_times = []
    crops_per_frame = []

    for img_name in eval_images:
        p = str(val_img_dir / img_name)
        t_total_start = time.perf_counter()

        # Stage 1: Full-Frame Inference
        t_ff_start = time.perf_counter()
        res_ff = model.predict(p, imgsz=960, conf=0.01, device="cpu", verbose=False)[0]
        t_ff_end = time.perf_counter()
        fov_stage_ff_times.append((t_ff_end - t_ff_start) * 1000.0)

        ff_dets = []
        for b in res_ff.boxes:
            ff_dets.append({
                "class_name": model.names[int(b.cls.item())],
                "conf": float(b.conf.item()),
                "box": b.xyxy[0].cpu().numpy(),
                "source": "full_frame",
                "img_name": img_name,
            })

        # Stage 2: Uncertainty Crop Generation & Extraction
        t_cg_start = time.perf_counter()
        frame = cv2.imread(p)
        crop_proposals = generate_uncertainty_crops(ff_dets, int(img_w), int(img_h), max_crops=2)
        crops_per_frame.append(len(crop_proposals))

        crop_dets_remapped = []
        t_ci_start = time.perf_counter()
        fov_stage_cropgen_times.append((t_ci_start - t_cg_start) * 1000.0)

        # Stage 3: Crop Inference & Remapping
        for prop in crop_proposals:
            crop_b = prop["crop_box"]
            sub_img, int_coords = cropper.extract_crop_image(frame, crop_b)
            if sub_img.shape[0] < 8 or sub_img.shape[1] < 8:
                continue

            crop_res = model.predict(sub_img, imgsz=640, conf=0.01, device="cpu", verbose=False)[0]
            cx1, cy1, cx2, cy2 = int_coords

            for b in crop_res.boxes:
                cname = model.names[int(b.cls.item())]
                conf = float(b.conf.item())
                rx1, ry1, rx2, ry2 = b.xyxy[0].cpu().numpy()

                margin = foveation_config.boundary_suppress_margin
                sub_w = cx2 - cx1
                sub_h = cy2 - cy1
                is_boundary = (rx1 <= margin and cx1 > margin) or \
                              (rx2 >= sub_w - margin and cx2 < img_w - margin) or \
                              (ry1 <= margin and cy1 > margin) or \
                              (ry2 >= sub_h - margin and cy2 < img_h - margin)
                if is_boundary and conf < 0.35:
                    continue

                ox1 = max(0.0, min(img_w, cx1 + rx1))
                oy1 = max(0.0, min(img_h, cy1 + ry1))
                ox2 = max(0.0, min(img_w, cx1 + rx2))
                oy2 = max(0.0, min(img_h, cy1 + ry2))

                if (ox2 - ox1) >= 2.0 and (oy2 - oy1) >= 2.0:
                    crop_dets_remapped.append({
                        "img_name": img_name,
                        "class_name": cname,
                        "conf": conf,
                        "box": np.array([ox1, oy1, ox2, oy2], dtype=np.float32),
                        "source": "crop",
                    })
        t_ci_end = time.perf_counter()
        fov_stage_cropinf_times.append((t_ci_end - t_ci_start) * 1000.0)

        # Stage 4: Calibrated WBF Fusion
        t_wbf_start = time.perf_counter()
        fused = calibrated_wbf_consensus_fusion(
            full_frame_dets=ff_dets,
            crop_dets=crop_dets_remapped,
            tau_ff=0.25,
            tau_isolated=0.40,
            tau_confirmed=0.20,
            iou_assoc=0.30,
        )
        t_wbf_end = time.perf_counter()
        fov_stage_wbf_times.append((t_wbf_end - t_wbf_start) * 1000.0)

        t_total_end = time.perf_counter()
        fov_total_e2e_times.append((t_total_end - t_total_start) * 1000.0)

    # -------------------------------------------------------------
    # 4. Pipeline A Batched (batch_size=16)
    # -------------------------------------------------------------
    print("Benchmarking Pipeline A Batched (batch_size=16)...")
    batch_paths = [str(val_img_dir / img_name) for img_name in eval_images]
    batched_times = []
    b_size = 16
    for i in range(0, len(batch_paths), b_size):
        chunk = batch_paths[i:i+b_size]
        t0 = time.perf_counter()
        _ = model.predict(chunk, imgsz=960, conf=0.01, device="cpu", verbose=False)
        dt = (time.perf_counter() - t0) * 1000.0 / len(chunk)
        batched_times.extend([dt] * len(chunk))

    results = {
        "num_evaluated_frames": num_eval_images,
        "raw_yolo11m_batch1": {
            "p50_ms": round(float(np.median(raw_b1_times)), 1),
            "p95_ms": round(float(np.percentile(raw_b1_times, 95)), 1),
            "mean_ms": round(float(np.mean(raw_b1_times)), 1),
            "fps": round(1000.0 / float(np.mean(raw_b1_times)), 2),
        },
        "raw_yolo11m_batched_16": {
            "p50_ms": round(float(np.median(batched_times)), 1),
            "p95_ms": round(float(np.percentile(batched_times, 95)), 1),
            "mean_ms": round(float(np.mean(batched_times)), 1),
            "fps": round(1000.0 / float(np.mean(batched_times)), 2),
        },
        "foveation_wbf_true_e2e": {
            "p50_ms": round(float(np.median(fov_total_e2e_times)), 1),
            "p95_ms": round(float(np.percentile(fov_total_e2e_times, 95)), 1),
            "mean_ms": round(float(np.mean(fov_total_e2e_times)), 1),
            "fps": round(1000.0 / float(np.mean(fov_total_e2e_times)), 2),
            "mean_crops_per_frame": round(float(np.mean(crops_per_frame)), 2),
            "stage_breakdown_mean_ms": {
                "stage1_full_frame_960": round(float(np.mean(fov_stage_ff_times)), 1),
                "stage2_crop_generation_cv": round(float(np.mean(fov_stage_cropgen_times)), 1),
                "stage3_crop_inference_640": round(float(np.mean(fov_stage_cropinf_times)), 1),
                "stage4_wbf_consensus_fusion": round(float(np.mean(fov_stage_wbf_times)), 1),
            },
            "isolated_foveation_addon_mean_ms": round(
                float(np.mean(fov_stage_cropgen_times) + np.mean(fov_stage_cropinf_times) + np.mean(fov_stage_wbf_times)), 1
            ),
        },
    }

    print("\n" + "=" * 70)
    print("LATENCY AUDIT RESULTS (IDENTICAL PROTOCOL)")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    out_p = Path("experiments/phase6_system_eval/latency_audit_report.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved latency audit to {out_p}")


if __name__ == "__main__":
    run_latency_audit(num_eval_images=20, num_warmup=3)
