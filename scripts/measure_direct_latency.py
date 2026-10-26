"""
RoadSentinel Phase 3.3 — Direct Latency and Pipeline Benchmark
Directly measures both:
  1. In-memory forward inference latency (preloaded frames)
  2. End-to-end file-to-output pipeline latency (disk read, OpenCV 5 decode, full-frame predict,
     crop proposal, crop predict, coordinate remap, box fusion)

Measures directly on real BDD100K validation images with explicit warm-up and timing repetitions.
DOES NOT add a constant offset.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch

from src.foveation.engine import ActiveFoveationEngine
from src.foveation.types import FoveationConfig

def run_direct_latency_benchmark(num_images: int = 15, num_warmup: int = 3):
    val_img_dir = Path("bdd100k_dataset/val/images")
    ckpt_path = Path("weights/bdd100k_yolo11s.pt")
    out_dir = Path("experiments/phase3_foveation")
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(val_img_dir.glob("*.jpg")))[:num_images]
    assert len(image_paths) >= num_images, f"Not enough validation images in {val_img_dir}"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("RoadSentinel Phase 3.3 — Independent Direct Latency Benchmark")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Timed images: {num_images} | Warm-up runs: {num_warmup}")

    # Pre-load in-memory frames
    in_memory_images = [cv2.imread(str(p)) for p in image_paths]

    configs = [
        {
            "name": "FullFrame-640",
            "config": FoveationConfig(policy="full_frame_only", full_frame_imgsz=640, max_crops_per_image=0),
        },
        {
            "name": "FullFrame-960",
            "config": FoveationConfig(policy="full_frame_only", full_frame_imgsz=960, max_crops_per_image=0),
        },
        {
            "name": "FullFrame-1280",
            "config": FoveationConfig(policy="full_frame_only", full_frame_imgsz=1280, max_crops_per_image=0),
        },
        {
            "name": "FixedGrid-960",
            "config": FoveationConfig(policy="fixed_grid", full_frame_imgsz=960, max_crops_per_image=2),
        },
        {
            "name": "RandomCrop-960",
            "config": FoveationConfig(policy="random", full_frame_imgsz=960, max_crops_per_image=2),
        },
        {
            "name": "ActiveFoveation-960",
            "config": FoveationConfig(policy="uncertainty_active", full_frame_imgsz=960, max_crops_per_image=2),
        },
    ]

    results: Dict[str, Any] = {}

    for cfg_item in configs:
        name = cfg_item["name"]
        f_config = cfg_item["config"]
        print(f"\n--- Benchmarking {name} (Policy: {f_config.policy}, FullFrame Imgsz: {f_config.full_frame_imgsz}) ---")

        engine = ActiveFoveationEngine(checkpoint_path=str(ckpt_path), config=f_config, device=device)

        # ---------------------------------------------------------------------
        # Part A: Warm-up
        # ---------------------------------------------------------------------
        for w_idx in range(num_warmup):
            _ = engine.process_image(in_memory_images[w_idx % len(in_memory_images)], conf_threshold=0.25)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()

        # ---------------------------------------------------------------------
        # Part B: In-Memory Latency (Pre-loaded numpy ndarrays)
        # ---------------------------------------------------------------------
        in_mem_times = []
        in_mem_breakdowns = []
        for img in in_memory_images:
            if torch.cuda.is_available() and "cuda" in device:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, trace = engine.process_image(img, conf_threshold=0.25)
            if torch.cuda.is_available() and "cuda" in device:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            in_mem_times.append((t1 - t0) * 1000.0)
            in_mem_breakdowns.append({
                "ff_infer_ms": trace.timing.full_frame_infer_ms,
                "crop_proposal_ms": trace.timing.uncertainty_proposal_ms,
                "crop_extract_ms": trace.timing.crop_extraction_ms,
                "crop_infer_ms": trace.timing.crop_infer_ms,
                "fusion_ms": trace.timing.fusion_ms,
                "total_ms": trace.timing.total_ms,
                "crops_executed": trace.num_crops_executed,
            })

        # ---------------------------------------------------------------------
        # Part C: Pipeline Latency (Disk File-to-Output with I/O and Decode)
        # ---------------------------------------------------------------------
        # Pipeline warm-up
        for w_idx in range(num_warmup):
            _ = engine.process_image(str(image_paths[w_idx % len(image_paths)]), conf_threshold=0.25)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()

        pipeline_times = []
        pipeline_breakdowns = []
        for p in image_paths:
            if torch.cuda.is_available() and "cuda" in device:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, trace = engine.process_image(str(p), conf_threshold=0.25)
            if torch.cuda.is_available() and "cuda" in device:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            pipeline_times.append((t1 - t0) * 1000.0)
            pipeline_breakdowns.append({
                "read_ms": trace.timing.read_ms,
                "ff_infer_ms": trace.timing.full_frame_infer_ms,
                "crop_proposal_ms": trace.timing.uncertainty_proposal_ms,
                "crop_extract_ms": trace.timing.crop_extraction_ms,
                "crop_infer_ms": trace.timing.crop_infer_ms,
                "fusion_ms": trace.timing.fusion_ms,
                "total_ms": trace.timing.total_ms,
                "crops_executed": trace.num_crops_executed,
            })

        in_mem_arr = np.array(in_mem_times, dtype=np.float32)
        pipe_arr = np.array(pipeline_times, dtype=np.float32)

        results[name] = {
            "policy": f_config.policy,
            "full_frame_imgsz": f_config.full_frame_imgsz,
            "crop_imgsz": f_config.crop_imgsz,
            "max_crops": f_config.max_crops_per_image,
            "num_timed_images": num_images,
            "num_warmup": num_warmup,
            "in_memory": {
                "mean_ms": float(np.mean(in_mem_arr)),
                "median_p50_ms": float(np.median(in_mem_arr)),
                "p95_ms": float(np.percentile(in_mem_arr, 95)),
                "min_ms": float(np.min(in_mem_arr)),
                "max_ms": float(np.max(in_mem_arr)),
                "throughput_fps": float(1000.0 / np.mean(in_mem_arr)),
            },
            "pipeline": {
                "mean_ms": float(np.mean(pipe_arr)),
                "median_p50_ms": float(np.median(pipe_arr)),
                "p95_ms": float(np.percentile(pipe_arr, 95)),
                "min_ms": float(np.min(pipe_arr)),
                "max_ms": float(np.max(pipe_arr)),
                "throughput_fps": float(1000.0 / np.mean(pipe_arr)),
            },
            "measured_io_and_decode_overhead_ms": float(np.median(pipe_arr) - np.median(in_mem_arr)),
        }

        print(f"  In-Memory Median P50 : {results[name]['in_memory']['median_p50_ms']:.2f} ms (p95: {results[name]['in_memory']['p95_ms']:.2f} ms, FPS: {results[name]['in_memory']['throughput_fps']:.2f})")
        print(f"  Pipeline  Median P50 : {results[name]['pipeline']['median_p50_ms']:.2f} ms (p95: {results[name]['pipeline']['p95_ms']:.2f} ms, FPS: {results[name]['pipeline']['throughput_fps']:.2f})")
        print(f"  Direct Measured I/O Overhead: {results[name]['measured_io_and_decode_overhead_ms']:.2f} ms")

        del engine
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.empty_cache()

    out_file = out_dir / "directly_measured_latency.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 70)
    print(f"Direct latency measurements successfully saved to {out_file}")
    print("=" * 70)
    return results

if __name__ == "__main__":
    run_direct_latency_benchmark(num_images=10, num_warmup=2)
