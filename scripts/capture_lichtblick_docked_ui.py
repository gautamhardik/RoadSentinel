"""
RoadSentinel — Capture Lichtblick Docked IDE Layout Screenshots
File: scripts/capture_lichtblick_docked_ui.py
"""

import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ARTIFACTS_DIR = Path("C:/Users/hiten/.gemini/antigravity-ide/brain/77772665-0f72-4191-a352-b84d09a69c0a")

def capture():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Full HD desktop viewport
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        print("[capture] Navigating to http://localhost:8000/ ...")
        page.goto("http://localhost:8000/", wait_until="networkidle")
        time.sleep(2.0)

        # 1. Main Docked IDE Stage (Perception Studio)
        shot1 = ARTIFACTS_DIR / "lichtblick_docked_studio_main.png"
        page.screenshot(path=str(shot1), full_page=False)
        print(f"[OK] Captured Main Studio: {shot1}")

        # 2. Trigger Deliberation to populate Deliberation Stream and Inspector
        page.evaluate("if (window.triggerLiveInvestigation) window.triggerLiveInvestigation();")
        time.sleep(2.0)
        shot2 = ARTIFACTS_DIR / "lichtblick_docked_studio_deliberation.png"
        page.screenshot(path=str(shot2), full_page=False)
        print(f"[OK] Captured Deliberation: {shot2}")

        # 3. Switch to Scientific Ledger
        page.click("#nav-btn-ledger")
        time.sleep(1.0)
        shot3 = ARTIFACTS_DIR / "lichtblick_docked_ledger.png"
        page.screenshot(path=str(shot3), full_page=False)
        print(f"[OK] Captured Ledger: {shot3}")

        # 4. Switch to Fleet Video
        page.click("#nav-btn-video")
        time.sleep(1.0)
        shot4 = ARTIFACTS_DIR / "lichtblick_docked_video.png"
        page.screenshot(path=str(shot4), full_page=False)
        print(f"[OK] Captured Fleet Video: {shot4}")

        browser.close()
        print("[capture] Complete!")

if __name__ == "__main__":
    capture()
