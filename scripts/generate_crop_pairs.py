"""
Generate Raw vs OpenCV Enhanced Crop Pairs for Golden 12 Traces
Creates:
- artifacts/golden_traces/trace_{id:02d}_raw.jpg
- artifacts/golden_traces/trace_{id:02d}_opencv.jpg
- artifacts/golden_traces/trace_{id:02d}_full.jpg
Updates golden_10_verified_traces.json with exact relative URLs for frontend display.
"""

import os
import sys
import json
from pathlib import Path

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.agentic.mcp_opencv_server import McpOpenCVServer

TRACES_JSON = WORKSPACE_ROOT / "experiments/agentic_traces/golden_10_verified_traces.json"
OUT_DIR = WORKSPACE_ROOT / "artifacts/golden_traces"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_crops():
    if not TRACES_JSON.exists():
        print(f"Error: {TRACES_JSON} not found")
        return

    with open(TRACES_JSON, "r") as f:
        traces = json.load(f)

    server = McpOpenCVServer()
    print(f"Generating crop pairs for {len(traces)} golden traces...")

    for t in traces:
        tid = t["trace_id"]
        img_p = WORKSPACE_ROOT / t["image_path"]
        if not img_p.exists():
            print(f"Warning: {img_p} not found")
            continue

        frame = cv2.imread(str(img_p))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        box = t["gt_box"]
        bx1, by1, bx2, by2 = [int(v) for v in box]

        # Extract context-preserving crop (min 320x320)
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
        bw, bh = bx2 - bx1, by2 - by1
        context_sz = max(320, int(max(bw, bh) * 3.5))

        x1_c = max(0, cx - context_sz // 2)
        y1_c = max(0, cy - context_sz // 2)
        x2_c = min(w, x1_c + context_sz)
        y2_c = min(h, y1_c + context_sz)

        if x2_c - x1_c < context_sz and x1_c > 0:
            x1_c = max(0, x2_c - context_sz)
        if y2_c - y1_c < context_sz and y1_c > 0:
            y1_c = max(0, y2_c - context_sz)

        raw_crop = frame[y1_c:y2_c, x1_c:x2_c].copy()

        # Apply the exact MCP transformation indicated by the trace
        tool_name = t["mcp_tool_selected"]
        if "photometric" in tool_name or t["diagnostic_sensor"]["is_underexposed"]:
            enhanced_crop, _ = server.tool_photometric_enhance_clahe(raw_crop, clip_limit=2.5, gamma=1.2)
        elif "acuity" in tool_name or t["diagnostic_sensor"]["is_low_acuity"]:
            enhanced_crop, _ = server.tool_acuity_sharpen(raw_crop)
        else:
            enhanced_crop = cv2.resize(raw_crop, (raw_crop.shape[1], raw_crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        # Draw reticle on full frame copy
        full_annotated = frame.copy()
        # Draw tactical corner ticks on full frame
        cw_box = (bx2 - bx1)
        ch_box = (by2 - by1)
        tick = min(15, min(cw_box, ch_box) // 3)
        # Green reticle ticks
        col = (16, 185, 129) if t["status"] == "RECOVERED_TRUE_POSITIVE" else (11, 158, 245)
        # Top-left
        cv2.line(full_annotated, (bx1, by1), (bx1 + tick, by1), col, 2)
        cv2.line(full_annotated, (bx1, by1), (bx1, by1 + tick), col, 2)
        # Top-right
        cv2.line(full_annotated, (bx2, by1), (bx2 - tick, by1), col, 2)
        cv2.line(full_annotated, (bx2, by1), (bx2, by1 + tick), col, 2)
        # Bottom-left
        cv2.line(full_annotated, (bx1, by2), (bx1 + tick, by2), col, 2)
        cv2.line(full_annotated, (bx1, by2), (bx1, by2 - tick), col, 2)
        # Bottom-right
        cv2.line(full_annotated, (bx2, by2), (bx2 - tick, by2), col, 2)
        cv2.line(full_annotated, (bx2, by2), (bx2, by2 - tick), col, 2)

        # Tag
        cv2.putText(
            full_annotated,
            f"{t['gt_class'].upper()} | WBF: {t['conf_wbf_final']:.2f}",
            (bx1, max(22, by1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            col,
            2,
        )

        raw_path = OUT_DIR / f"trace_{tid:02d}_raw.jpg"
        enh_path = OUT_DIR / f"trace_{tid:02d}_opencv.jpg"
        full_path = OUT_DIR / f"trace_{tid:02d}_full.jpg"

        cv2.imwrite(str(raw_path), raw_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(enh_path), enhanced_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(full_path), full_annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        t["raw_crop_path"] = f"artifacts/golden_traces/trace_{tid:02d}_raw.jpg"
        t["opencv_crop_path"] = f"artifacts/golden_traces/trace_{tid:02d}_opencv.jpg"
        t["annotated_full_path"] = f"artifacts/golden_traces/trace_{tid:02d}_full.jpg"

        print(f"Trace #{tid:02d}: Saved raw, enhanced, and full images.")

    with open(TRACES_JSON, "w") as f:
        json.dump(traces, f, indent=2)

    print(f"\n[success] All crop pairs generated and {TRACES_JSON} updated!")

if __name__ == "__main__":
    generate_crops()
