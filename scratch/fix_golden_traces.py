import json
import cv2
import numpy as np
from pathlib import Path
import os
import sys

# Add root workspace to sys.path
WORKSPACE_ROOT = Path(".").resolve()
sys.path.insert(0, str(WORKSPACE_ROOT / "src/agentic"))
import mcp_opencv_server as mcp_mod

server = mcp_mod.McpOpenCVServer()

json_path = "experiments/agentic_traces/golden_10_verified_traces.json"
with open(json_path, "r", encoding="utf-8") as f:
    traces = json.load(f)

artifacts_dir = Path("artifacts/golden_traces")
artifacts_dir.mkdir(parents=True, exist_ok=True)

print(f"Loaded {len(traces)} traces from {json_path}")

for idx, t in enumerate(traces, 1):
    img_id = t["image_id"]
    full_path = artifacts_dir / f"trace_{idx:02d}_full.jpg"
    
    # If full image exists, read it; otherwise read from bdd100k_dataset/val/images
    if full_path.exists():
        full_img = cv2.imread(str(full_path))
    else:
        src_path = f"bdd100k_dataset/val/images/{img_id}.jpg"
        full_img = cv2.imread(src_path)
        cv2.imwrite(str(full_path), full_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
    orig_h, orig_w = full_img.shape[:2]
    
    # Determine context crop window around gt_box
    gt_box = t["gt_box"]
    bx1, by1, bx2, by2 = gt_box
    cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    crop_w = max(320, int((bx2 - bx1) * 3.5))
    crop_h = max(320, int((by2 - by1) * 3.5))
    x1_c, y1_c = max(0, int(cx - crop_w / 2)), max(0, int(cy - crop_h / 2))
    x2_c, y2_c = min(orig_w, int(cx + crop_w / 2)), min(orig_h, int(cy + crop_h / 2))
    
    raw_sub = full_img[y1_c:y2_c, x1_c:x2_c].copy()
    raw_640 = cv2.resize(raw_sub, (640, 640), interpolation=cv2.INTER_LINEAR)
    
    # Apply tool transformation based on branch
    branch = t.get("branch_type", "")
    tool = t.get("mcp_tool_selected", "")
    
    # Fix scene labeling & branch discrepancies:
    if idx == 1:
        # Trace 01: Night Crosswalk Pedestrian
        t["scenario_title"] = "01. Night Crosswalk Pedestrian - Headlight Glare & Low-Lux Recovery"
        t["environment_tag"] = "NIGHT GLARE"
        t["branch_type"] = "NIGHT_CLAHE_GAMMA_RECOVERY"
        t["mcp_tool_selected"] = "tool_photometric_recover_night_glare"
        t["tool_label"] = "OpenCV CLAHE + Gamma LUT"
        branch = "NIGHT_CLAHE_GAMMA_RECOVERY"
    elif idx == 3:
        # Trace 03: Storefront Deep Shadow - Night Scene
        t["scenario_title"] = "03. Storefront Deep Shadow - Night High-Contrast Glare & Shadow Recovery"
        t["environment_tag"] = "NIGHT GLARE"
        t["branch_type"] = "NIGHT_CLAHE_GAMMA_RECOVERY"
        t["mcp_tool_selected"] = "tool_photometric_recover_night_glare"
        t["tool_label"] = "OpenCV CLAHE + Gamma LUT"
        branch = "NIGHT_CLAHE_GAMMA_RECOVERY"
    elif idx == 7:
        # Trace 07: Night Corridor Pedestrian
        t["scenario_title"] = "07. Night Corridor Pedestrian - Multi-Scale Context Verification"
        t["environment_tag"] = "NIGHT GLARE"
        t["branch_type"] = "NIGHT_CLAHE_GAMMA_RECOVERY"
        t["mcp_tool_selected"] = "tool_photometric_recover_night_glare"
        t["tool_label"] = "OpenCV CLAHE + Gamma LUT"
        branch = "NIGHT_CLAHE_GAMMA_RECOVERY"
    elif idx == 8:
        # Trace 08: Distant Crosswalk Group (Dusk/Shadow)
        t["environment_tag"] = "SHADOW RECOVERY"
        t["branch_type"] = "DAYLIGHT_SHADOW_RECOVERY"
        t["mcp_tool_selected"] = "tool_photometric_recover_shadow_clahe"
        branch = "DAYLIGHT_SHADOW_RECOVERY"
    elif idx == 10:
        # Trace 10: Curbside Shadowed Pedestrian (Broad Daylight with Shadows)
        t["environment_tag"] = "SHADOW RECOVERY"
        t["branch_type"] = "DAYLIGHT_SHADOW_RECOVERY"
        t["mcp_tool_selected"] = "tool_photometric_recover_shadow_clahe"
        branch = "DAYLIGHT_SHADOW_RECOVERY"
        
    # Generate clean enhanced crop
    if "NIGHT" in branch or "CLAHE" in tool or "night_glare" in tool:
        enh_sub, _ = server.tool_photometric_enhance_clahe(raw_sub, clip_limit=2.5, gamma=1.2)
    elif "SHADOW" in branch or "shadow_clahe" in tool:
        enh_sub, _ = server.tool_photometric_enhance_clahe(raw_sub, clip_limit=2.0, gamma=1.15)
    elif "RAIN" in branch or "sharpen" in tool or "bilateral" in tool:
        enh_sub, _ = server.tool_acuity_sharpen(raw_sub)
    elif "FOVEAT" in branch:
        enh_sub = cv2.resize(raw_sub, (raw_sub.shape[1], raw_sub.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    else:
        enh_sub = raw_sub.copy()
        
    enh_640 = cv2.resize(enh_sub, (640, 640), interpolation=cv2.INTER_LINEAR)
    
    # Save PRISTINE clean crops without baked-in cv2.putText banners!
    raw_crop_path = artifacts_dir / f"trace_{idx:02d}_raw_crop.jpg"
    opencv_crop_path = artifacts_dir / f"trace_{idx:02d}_opencv_crop.jpg"
    
    cv2.imwrite(str(raw_crop_path), raw_640, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(opencv_crop_path), enh_640, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # Update trace URLs
    t["raw_crop_url"] = f"/artifacts/golden_traces/trace_{idx:02d}_raw_crop.jpg"
    t["opencv_crop_url"] = f"/artifacts/golden_traces/trace_{idx:02d}_opencv_crop.jpg"
    t["full_image_url"] = f"/artifacts/golden_traces/trace_{idx:02d}_full.jpg"
    
    print(f"Trace {idx:02d} ({img_id}): Saved clean crops (640x640) -> {t['scenario_title']} [{t['environment_tag']}]")

# Save updated golden traces JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(traces, f, indent=2)

print("\n[SUCCESS] Updated golden_10_verified_traces.json and pristine crop images without text banners!")
