import asyncio
import os
from playwright.async_api import async_playwright

ARTIFACT_DIR = r"C:\Users\hiten\.gemini\antigravity-ide\brain\77772665-0f72-4191-a352-b84d09a69c0a"

async def capture_all():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Use high resolution desktop viewport
        context = await browser.new_context(viewport={"width": 1600, "height": 950})
        page = await context.new_page()

        print("[*] Navigating to http://localhost:8000 ...")
        await page.goto("http://localhost:8000", wait_until="networkidle")
        await page.wait_for_timeout(2000)

        # 1. Capture Workspace 1 (Perception Workbench)
        ws1_path = os.path.join(ARTIFACT_DIR, "product_showcase_ws1_workbench.png")
        await page.screenshot(path=ws1_path)
        print(f"[OK] Captured Workspace 1: {ws1_path}")

        # 2. Switch to Workspace 2 (Fleet Video & 3D HUD)
        print("[*] Switching to Workspace 2...")
        await page.click("#nav-btn-workspace2")
        await page.wait_for_timeout(1500)
        ws2_path = os.path.join(ARTIFACT_DIR, "product_showcase_ws2_video.png")
        await page.screenshot(path=ws2_path)
        print(f"[OK] Captured Workspace 2: {ws2_path}")

        # 3. Switch to Workspace 3 (AWS COOL & Evidence Ledger)
        print("[*] Switching to Workspace 3...")
        await page.click("#nav-btn-workspace3")
        await page.wait_for_timeout(1500)
        ws3_path = os.path.join(ARTIFACT_DIR, "product_showcase_ws3_ledger.png")
        await page.screenshot(path=ws3_path)
        print(f"[OK] Captured Workspace 3: {ws3_path}")

        await browser.close()
        print("[OK] All product screenshots captured successfully!")

if __name__ == "__main__":
    asyncio.run(capture_all())
