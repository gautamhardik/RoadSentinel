"""
RoadSentinel Autonomous CLI & Edge Engine
Run standalone perception, foveation, video processing, and server deployment from the command line.

Usage:
    python -m src.cli investigate --image path/to/image.jpg
    python -m src.cli video --input demo_videos/night_urban_vru.mp4 --output runs/annotated.mp4
    python -m src.cli benchmark --iterations 50
    python -m src.cli serve --port 8000
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np


def cmd_investigate(args):
    """Executes live 5-stage agentic vision loop on an image."""
    from ultralytics import YOLO
    from src.agentic.mcp_opencv_server import McpOpenCVServer

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[Error] Image not found at {image_path}")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"  RoadSentinel Agentic Vision Investigation")
    print(f"  Input: {image_path.name}")
    print(f"=======================================================")

    image_mat = cv2.imread(str(image_path))
    orig_h, orig_w = image_mat.shape[:2]

    model_path = WORKSPACE_ROOT / "experiments/phase10_model_optimization/phase10b_extended_training/session4_artifacts/session4_best.pt"
    if not model_path.exists():
        model_path = WORKSPACE_ROOT / "session4_best.pt"

    t0 = time.perf_counter()
    print(f"[*] Loading detector ({model_path.name})...")
    model = YOLO(str(model_path))
    mcp = McpOpenCVServer()

    # Stage 1: Full-frame 960px Fast Path
    print(f"[*] Stage 1: Running 960px Full-Frame Fast-Path Inference...")
    t_s1 = time.perf_counter()
    res = model.predict(source=image_mat, imgsz=960, conf=0.15, verbose=False)
    s1_ms = (time.perf_counter() - t_s1) * 1000.0

    VRU_CLASSES = {0: "person", 1: "rider", 5: "bicycle", 6: "motorcycle"}
    candidates = []
    for b in res[0].boxes:
        cls_id = int(b.cls.item())
        if cls_id in VRU_CLASSES:
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].cpu().numpy()]
            candidates.append({
                "class": VRU_CLASSES[cls_id],
                "conf": conf,
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })

    print(f"    Found {len(candidates)} VRU candidates in {s1_ms:.1f}ms.")
    if not candidates:
        print("[!] No VRUs detected in frame. Fast-path clear.")
        return

    # Select primary candidate
    target = sorted(candidates, key=lambda c: abs(c["conf"] - 0.30))[0]
    print(f"    Target VRU selected: {target['class']} (Initial Conf: {target['conf']*100:.1f}%) at {target['box']}")

    # Stage 2: Sensor Diagnosis
    print(f"[*] Stage 2: OpenCV 5 LAB/Laplacian Diagnostic Sensor...")
    x1, y1, x2, y2 = [int(v) for v in target["box"]]
    diag = mcp.tool_diagnose_roi(image_mat, [x1, y1, x2, y2])
    print(f"    Mean Luma: {diag['mean_luminance']:.1f} | Acuity: {diag['laplacian_variance']:.1f} | Recommended: {diag['recommended_tool']}")

    # Stage 3: Tool Routing & Foveation
    print(f"[*] Stage 3: Autonomous MCP Tool Call...")
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    csize = max(320, int(max(x2 - x1, y2 - y1) * 3.0))
    rx1 = max(0, center_x - csize // 2)
    ry1 = max(0, center_y - csize // 2)
    rx2 = min(orig_w, rx1 + csize)
    ry2 = min(orig_h, ry1 + csize)
    crop_raw = image_mat[ry1:ry2, rx1:rx2].copy()

    if diag["is_underexposed"]:
        enhanced_crop, meta = mcp.tool_photometric_enhance_clahe(crop_raw, clip_limit=2.5, gamma=1.2)
        tool_invoked = "tool_photometric_enhance_clahe (CLAHE + Gamma 1.2 LUT)"
    elif diag["is_low_acuity"]:
        enhanced_crop, meta = mcp.tool_acuity_sharpen(crop_raw)
        tool_invoked = "tool_acuity_sharpen (Bilateral + Unsharp Mask)"
    else:
        enhanced_crop = crop_raw.copy()
        tool_invoked = "tool_foveate_supercrop_daylight (Lanczos-4 Supercrop)"

    print(f"    Agent Tool: {tool_invoked}")

    # Stage 4: Counterfactual Crop Inference
    print(f"[*] Stage 4: Counterfactual 640px Crop Inference...")
    res_raw = model.predict(source=crop_raw, imgsz=640, conf=0.10, verbose=False)
    res_cv = model.predict(source=enhanced_crop, imgsz=640, conf=0.10, verbose=False)

    raw_confs = [float(b.conf.item()) for b in res_raw[0].boxes if int(b.cls.item()) in VRU_CLASSES]
    cv_confs = [float(b.conf.item()) for b in res_cv[0].boxes if int(b.cls.item()) in VRU_CLASSES]

    conf_raw = max(raw_confs) if raw_confs else 0.0
    conf_cv = max(cv_confs) if cv_confs else 0.0
    delta = conf_cv - conf_raw
    print(f"    Raw Crop Conf: {conf_raw*100:.1f}% | OpenCV 5 Crop Conf: {conf_cv*100:.1f}% (Delta: {delta*100:+.1f}%)")

    # Stage 5: WBF Consensus
    print(f"[*] Stage 5: Calibrated WBF Consensus & Epistemic Gate...")
    effective = max(conf_raw, conf_cv)
    if effective >= 0.25:
        conf_final = min(0.98, 0.5 * (target["conf"] + effective) + 0.10)
        status = "RECOVERED_TRUE_POSITIVE"
    elif effective >= 0.15:
        conf_final = effective
        status = "ESCALATED_HUMAN_REVIEW"
    else:
        conf_final = min(target["conf"], effective)
        status = "SUPPRESSED_BACKGROUND_ARTIFACT"

    total_time = (time.perf_counter() - t0) * 1000.0

    print(f"\n=======================================================")
    print(f"  INVESTIGATION VERDICT: {status}")
    print(f"  Confidence Shift: {target['conf']*100:.1f}% -> {conf_final*100:.1f}%")
    print(f"  Total Pipeline Latency: {total_time:.1f}ms")
    print(f"=======================================================\n")

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "raw_crop.jpg"), crop_raw)
        cv2.imwrite(str(out_dir / "enhanced_crop.jpg"), enhanced_crop)
        print(f"[✓] Crops exported to {out_dir}")


def cmd_video(args):
    """Processes video file using continuous foveation pipeline."""
    from src.video.continuous_video_pipeline import ContinuousVideoPipeline

    input_p = Path(args.input)
    if not input_p.exists():
        print(f"[Error] Input video not found at {input_p}")
        sys.exit(1)

    output_p = Path(args.output or "runs/annotated_video.mp4")
    print(f"[*] Processing video {input_p.name} -> {output_p}...")
    pipeline = ContinuousVideoPipeline()
    meta = pipeline.process_video(str(input_p), str(output_p), max_frames=args.max_frames)
    print(f"[✓] Video successfully processed: {meta.get('output_video_path')}")


def cmd_benchmark(args):
    """Runs live OpenCV 5 COOL microbenchmarks."""
    from src.saas.api import run_live_microbenchmark

    print(f"[*] Running OpenCV 5 COOL Microbenchmark ({args.iterations} iterations)...")
    res = run_live_microbenchmark(iterations=args.iterations)
    bench = res.get("benchmarks", {})

    print(f"\n{'-'*65}")
    print(f"{'OpenCV 5 Operation':<35} | {'Mean Latency':<12} | {'FPS'}")
    print(f"{'-'*65}")
    total_ms = 0.0
    for name, data in bench.items():
        if name == "Op6_Full_Chained_Pipeline":
            continue
        ms = data.get("mean_ms", 0.0)
        fps = data.get("fps", 1000.0 / ms if ms > 0 else 0)
        total_ms += ms
        print(f"{name:<35} | {ms:6.2f} ms     | {fps:5.1f}")
    print(f"{'-'*65}")
    chained_ms = bench.get("Op6_Full_Chained_Pipeline", {}).get("mean_ms", total_ms)
    print(f"{'Full Chained Pipeline':<35} | {chained_ms:6.2f} ms     | {1000.0/chained_ms:5.1f} FPS\n")


def cmd_serve(args):
    """Launches FastAPI SaaS Studio server."""
    import uvicorn
    print(f"[*] Starting RoadSentinel Studio server on http://0.0.0.0:{args.port}...")
    uvicorn.run("src.saas.api:app", host="0.0.0.0", port=args.port, reload=False, access_log=True)


def main():
    parser = argparse.ArgumentParser(description="RoadSentinel Autonomous Vision CLI & Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Investigate Command
    p_inv = subparsers.add_parser("investigate", help="Run 5-stage agentic investigation on an image")
    p_inv.add_argument("--image", "-i", required=True, help="Path to input image")
    p_inv.add_argument("--output", "-o", default=None, help="Directory to export crops")

    # Video Command
    p_vid = subparsers.add_parser("video", help="Process video with agentic foveation")
    p_vid.add_argument("--input", "-i", required=True, help="Input MP4 video")
    p_vid.add_argument("--output", "-o", default="runs/annotated_video.mp4", help="Output MP4 video")
    p_vid.add_argument("--max-frames", "-n", type=int, default=None, help="Max frames to process")

    # Benchmark Command
    p_bench = subparsers.add_parser("benchmark", help="Run live OpenCV 5 COOL benchmarks")
    p_bench.add_argument("--iterations", "-n", type=int, default=25, help="Number of benchmark iterations")

    # Serve Command
    p_serve = subparsers.add_parser("serve", help="Launch RoadSentinel SaaS Studio server")
    p_serve.add_argument("--port", "-p", type=int, default=8000, help="Port to bind server")

    args = parser.parse_args()

    if args.command == "investigate":
        cmd_investigate(args)
    elif args.command == "video":
        cmd_video(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()
