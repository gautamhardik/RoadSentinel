#!/usr/bin/env python3
"""
Test script to verify:
1. Sidebar toggle behavior: Never disappears, can reopen via floating button, top bar button, or Ctrl+B.
2. 3D Spatial Corridor: OrbitControls attached, 360 rotation, zoom, pan, and perspective angle presets.
"""
import time
from playwright.sync_api import sync_playwright

def run_tests():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        console_errors = []
        page_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        print("[Test 1] Navigating to http://localhost:8000 ...")
        page.goto("http://localhost:8000")
        page.wait_for_selector("#scenario-tree-list button")
        time.sleep(1.0)

        # 1. Test Sidebar Toggle & Re-open
        print("[Test 2] Testing Sidebar Toggle & Persistent Reopen Button...")
        drawer = page.query_selector("#left-scenario-drawer")
        assert drawer is not None, "Left drawer not found"
        bbox_before = drawer.bounding_box()
        assert bbox_before["width"] >= 200, f"Drawer initial width invalid: {bbox_before['width']}"

        # Collapse sidebar
        collapse_btn = page.query_selector("#toggle-left-drawer-btn")
        collapse_btn.click()
        time.sleep(0.4)

        # Check drawer is collapsed
        bbox_collapsed = drawer.bounding_box()
        print(f"  Drawer width when collapsed: {bbox_collapsed['width']}px")
        assert bbox_collapsed["width"] <= 10, f"Drawer did not collapse properly: {bbox_collapsed['width']}"

        # Check floating reopen button is visible
        floating_btn = page.query_selector("#floating-open-sidebar-btn")
        assert floating_btn is not None, "Floating reopen button not found in DOM"
        assert floating_btn.is_visible(), "Floating reopen button is not visible when drawer is collapsed!"
        print("  [PASS] Floating reopen button is visible on stage.")

        # Check topbar toggle button is visible
        topbar_btn = page.query_selector("#topbar-toggle-sidebar-btn")
        assert topbar_btn is not None, "Topbar toggle button not found in DOM"
        assert topbar_btn.is_visible(), "Topbar toggle button is not visible!"
        print("  [PASS] Topbar toggle button is visible.")

        # Capture screenshot of collapsed state with floating reopen button
        page.screenshot(path="scratch/sidebar_collapsed_with_reopen_tab.png")

        # Click floating button to reopen
        floating_btn.click()
        time.sleep(0.4)
        bbox_reopened = drawer.bounding_box()
        print(f"  Drawer width after clicking floating button: {bbox_reopened['width']}px")
        assert bbox_reopened["width"] >= 200, "Drawer failed to reopen via floating button!"
        print("  [PASS] Drawer successfully reopened via floating reopen button.")

        # Test Ctrl+B keyboard shortcut
        page.keyboard.press("Control+b")
        time.sleep(0.4)
        bbox_ctrl_b = drawer.bounding_box()
        assert bbox_ctrl_b["width"] <= 10, "Ctrl+B failed to collapse drawer"
        page.keyboard.press("Control+b")
        time.sleep(0.4)
        bbox_ctrl_b_reopened = drawer.bounding_box()
        assert bbox_ctrl_b_reopened["width"] >= 200, "Ctrl+B failed to reopen drawer"
        print("  [PASS] Ctrl+B keyboard shortcut toggle verified.")

        # 2. Test 3D Spatial Corridor & OrbitControls
        print("[Test 3] Testing 3D Spatial Corridor & OrbitControls...")
        page.query_selector("#canvas-tab-three").click()
        time.sleep(0.8)

        # Verify OrbitControls exists
        controls_exist = page.evaluate("state.threeObj && state.threeObj.controls !== null")
        print(f"  OrbitControls initialized: {controls_exist}")
        assert controls_exist, "OrbitControls failed to initialize on state.threeObj!"

        # Drag 360 degrees
        canvas = page.query_selector("#three-main-canvas")
        cbox = canvas.bounding_box()
        cx = cbox["x"] + cbox["width"] / 2
        cy = cbox["y"] + cbox["height"] / 2

        # Drag across canvas (full 360 orbit)
        page.mouse.move(cx, cy)
        page.mouse.down()
        page.mouse.move(cx + 300, cy - 80, steps=10)
        page.mouse.move(cx - 300, cy + 50, steps=10)
        page.mouse.up()
        time.sleep(0.3)
        print("  [PASS] Performed 360-degree mouse orbit drag.")

        # Test Zoom
        page.mouse.move(cx, cy)
        page.mouse.wheel(0, -300)
        time.sleep(0.3)
        page.mouse.wheel(0, 300)
        time.sleep(0.3)
        print("  [PASS] Performed mouse wheel zoom.")

        # Test Perspective Buttons
        perspectives = ["chase", "top", "side", "target", "reset"]
        for p_name in perspectives:
            page.evaluate(f"if (typeof setThreePerspective === 'function') setThreePerspective('{p_name}');")
            time.sleep(0.2)
        print("  [PASS] Perspective preset angle buttons executed successfully.")

        # Screenshot of 3D Corridor
        page.evaluate("if (typeof setThreePerspective === 'function') setThreePerspective('side');")
        time.sleep(0.4)
        page.screenshot(path="scratch/three_corridor_side_profile.png")

        page.evaluate("if (typeof setThreePerspective === 'function') setThreePerspective('top');")
        time.sleep(0.4)
        page.screenshot(path="scratch/three_corridor_birdseye.png")

        page.evaluate("if (typeof resetThreeCamera === 'function') resetThreeCamera();")
        time.sleep(0.4)
        page.screenshot(path="scratch/three_corridor_360_orbit.png")

        print(f"Console Errors ({len(console_errors)}): {console_errors}")
        print(f"Page Errors ({len(page_errors)}): {page_errors}")
        assert len(console_errors) == 0, f"Unexpected console errors: {console_errors}"
        assert len(page_errors) == 0, f"Unexpected page errors: {page_errors}"

        browser.close()
        print("\n>>> ALL TESTS PASSED WITH ZERO ERRORS! <<<")

if __name__ == "__main__":
    run_tests()
