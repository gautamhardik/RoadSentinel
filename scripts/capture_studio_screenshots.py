"""
Capture high-resolution screenshots of RoadSentinel with the new Linear + Apple Pro
deep obsidian slate aesthetic, glowing jewel tones, and refined typography.
"""

import os
import time
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = r"C:\Users\hiten\.gemini\antigravity-ide\brain\77772665-0f72-4191-a352-b84d09a69c0a"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def capture_screenshots():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # 1920x1080 high-DPI viewport
        context = browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page = context.new_page()

        print("[1] Loading RoadSentinel on http://localhost:8000 ...")
        page.goto("http://localhost:8000", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1500)

        # Shot 1: Perception Studio (2D Camera View)
        shot1_path = os.path.join(ARTIFACT_DIR, "studio_obsidian_palette_2d.png")
        page.screenshot(path=shot1_path, full_page=True)
        print(f"  Captured Shot 1: {shot1_path}")

        # Shot 2: Split Crop Compare
        print("[2] Switching to Split Crop Compare view...")
        page.click("#canvas-tab-crop")
        page.wait_for_timeout(1000)
        shot2_path = os.path.join(ARTIFACT_DIR, "studio_split_crop_view.png")
        page.screenshot(path=shot2_path, full_page=True)
        print(f"  Captured Shot 2: {shot2_path}")

        # Shot 3: 3D Spatial Corridor
        print("[3] Switching to 3D Spatial Corridor view...")
        page.click("#canvas-tab-three")
        page.wait_for_timeout(1500)
        shot3_path = os.path.join(ARTIFACT_DIR, "studio_3d_spatial_corridor.png")
        page.screenshot(path=shot3_path, full_page=True)
        print(f"  Captured Shot 3: {shot3_path}")

        # Shot 4: Fleet Video Stream
        print("[4] Switching to Fleet Video Stream workspace...")
        page.click("#nav-btn-video")
        page.wait_for_timeout(1500)
        shot4_path = os.path.join(ARTIFACT_DIR, "studio_fleet_video_workspace.png")
        page.screenshot(path=shot4_path, full_page=True)
        print(f"  Captured Shot 4: {shot4_path}")

        # Shot 5: Scientific Evaluation Ledger
        print("[5] Switching to Scientific Evaluation Ledger workspace...")
        page.click("#nav-btn-ledger")
        page.wait_for_timeout(1500)
        shot5_path = os.path.join(ARTIFACT_DIR, "studio_scientific_ledger_workspace.png")
        page.screenshot(path=shot5_path, full_page=True)
        print(f"  Captured Shot 5: {shot5_path}")

        browser.close()
        print("All forensic screenshots captured successfully!")

if __name__ == "__main__":
    capture_screenshots()
