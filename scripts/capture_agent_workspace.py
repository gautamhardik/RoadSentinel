import asyncio
import os
from playwright.async_api import async_playwright

ARTIFACT_DIR = r"C:\Users\hiten\.gemini\antigravity-ide\brain\77772665-0f72-4191-a352-b84d09a69c0a"

async def capture_agent_workspace():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1600, "height": 920})
        page = await context.new_page()

        print("[*] Navigating to http://localhost:8000 ...")
        await page.goto("http://localhost:8000", wait_until="networkidle")
        await page.wait_for_timeout(2000)

        # 1. Capture Main Agent Workspace with Camera Stage
        p1 = os.path.join(ARTIFACT_DIR, "ai_agent_workspace_camera.png")
        await page.screenshot(path=p1)
        print(f"[OK] Captured Camera Viewport: {p1}")

        # 2. Trigger Deliberation so the Agent Stream populates live with thought cards and tool execution
        print("[*] Triggering Live Deliberation in Agent Console...")
        await page.click("#stage-deliberate-btn")
        await page.wait_for_timeout(3500)
        p2 = os.path.join(ARTIFACT_DIR, "ai_agent_deliberation_active.png")
        await page.screenshot(path=p2)
        print(f"[OK] Captured Active Deliberation Stream: {p2}")

        # 3. Switch to 3D Ego-Corridor
        print("[*] Switching to 3D Ego-Corridor...")
        await page.click("#tab-btn-three")
        await page.wait_for_timeout(1500)
        p3 = os.path.join(ARTIFACT_DIR, "ai_agent_3d_corridor.png")
        await page.screenshot(path=p3)
        print(f"[OK] Captured 3D Ego-Corridor: {p3}")

        # 4. Switch to Crop Comparator
        print("[*] Switching to Crop Comparator...")
        await page.click("#tab-btn-loupe")
        await page.wait_for_timeout(1000)
        p4 = os.path.join(ARTIFACT_DIR, "ai_agent_crop_comparator.png")
        await page.screenshot(path=p4)
        print(f"[OK] Captured Crop Comparator: {p4}")

        await browser.close()
        print("[OK] All AI Agent Workspace captures completed!")

if __name__ == "__main__":
    asyncio.run(capture_agent_workspace())
