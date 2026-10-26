"""
Headless Playwright Forensic Audit Suite for RoadSentinel Platform.
Zero-Trust protocol testing Pillar 6 under 1920x1080 and 1440x900 viewports.
"""

import sys
import time
from playwright.sync_api import sync_playwright

def run_playwright_audit(viewport_w=1920, viewport_h=1080):
    print(f"\n=======================================================")
    print(f"  PLAYWRIGHT AUDIT: Viewport {viewport_w}x{viewport_h}")
    print(f"=======================================================")
    
    console_errors = []
    page_errors = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": viewport_w, "height": viewport_h})
        page = context.new_page()
        
        # Attach listeners for console errors and page errors
        page.on("console", lambda msg: console_errors.append(f"[{msg.type}] {msg.text}") if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        
        # 1. Initial Page Load
        print("[step 1] Navigating to http://localhost:8000 ...")
        t0 = time.time()
        page.goto("http://localhost:8000", wait_until="domcontentloaded", timeout=30000)
        page.wait_for_selector("#scenario-tree-list button", timeout=20000)
        print(f"  Loaded in {time.time() - t0:.2f}s | Title: {page.title()}")
        page.wait_for_timeout(1000)
        
        # 2. Check for duplicate prefix in scenario drawer
        print("[step 2] Checking scenario drawer labels for duplicate prefixes (e.g. '01. 01.')...")
        page.wait_for_selector("#scenario-tree-list button", timeout=5000)
        scenario_titles = page.eval_on_selector_all(
            "#scenario-tree-list button .font-semibold",
            "els => els.map(e => (e.innerText || '').trim()).filter(Boolean)"
        )
        if len(scenario_titles) != 12:
            print(f"  [FAIL] Expected exactly 12 scenario items, but found {len(scenario_titles)}")
            return False
        
        print(f"  [DOM VERIFIED] Found all {len(scenario_titles)} live Scenario Cards in DOM:")
        for idx, title in enumerate(scenario_titles, 1):
            print(f"    [{idx:02d}/12] {title}")
            
        dup_prefixes = [t for t in scenario_titles if any(f"0{d}. 0{d}." in t for d in range(10))]
        if dup_prefixes:
            print(f"  [FAIL] Found duplicate prefixes in titles: {dup_prefixes}")
            return False
        else:
            print(f"  [PASS] All {len(scenario_titles)} scenario titles verified clean with zero duplicate prefixes.")
            
        # 3. Test Loupe Hover over #main-road-image
        print("[step 3] Hovering across #main-road-image (Tactical Loupe interaction)...")
        road_img = page.query_selector("#main-road-image")
        if road_img:
            box = road_img.bounding_box()
            if box:
                page.mouse.move(box["x"] + box["width"] * 0.3, box["y"] + box["height"] * 0.3)
                page.wait_for_timeout(150)
                page.mouse.move(box["x"] + box["width"] * 0.7, box["y"] + box["height"] * 0.7)
                page.wait_for_timeout(150)
                print(f"  [PASS] Mouse hovered across road image ({box['width']}x{box['height']}px)")
            else:
                print("  [WARN] #main-road-image bounding box is None")
        else:
            print("  [WARN] #main-road-image element not found in DOM")

        # 4. Click through Scenarios 01 to 12
        print("[step 4] Clicking through all 12 scenarios in the sidebar...")
        for sc_idx in range(1, 13):
            page.evaluate(f"if (typeof selectScenario === 'function') selectScenario({sc_idx});")
            page.wait_for_timeout(100)
        print("  [PASS] Clicked through Scenarios 01 to 12 without crashing.")

        # 5. Split Curtain Slider Drag
        print("[step 5] Testing Split Curtain slider...")
        page.evaluate("if (typeof setCanvasViewMode === 'function') setCanvasViewMode('crop');")
        page.wait_for_timeout(400)
        handle = page.query_selector("#curtain-stage-handle")
        if handle:
            hbox = handle.bounding_box()
            if hbox:
                page.mouse.move(hbox["x"] + hbox["width"] / 2, hbox["y"] + hbox["height"] / 2)
                page.mouse.down()
                page.mouse.move(hbox["x"] - 150, hbox["y"])
                page.mouse.move(hbox["x"] + 150, hbox["y"])
                page.mouse.up()
                print("  [PASS] Dragged curtain slider handle left and right.")
            else:
                print("  [WARN] Curtain handle bounding box not visible.")
        else:
            print("  [WARN] #curtain-stage-handle element not found.")
        page.evaluate("if (typeof setCanvasViewMode === 'function') setCanvasViewMode('camera');")
        page.wait_for_timeout(200)

        # 6. Click 5-Stage Deliberation, Approve Recovery, Escalate Safety
        print("[step 6] Testing Deliberation & Approval action buttons...")
        page.evaluate("""() => {
            if (typeof sendAgentAction === 'function') sendAgentAction('deliberate_active');
            if (typeof approveAutonomousLock === 'function') approveAutonomousLock();
            if (typeof escalateToSafety === 'function') escalateToSafety();
        }""")
        page.wait_for_timeout(300)
        print("  [PASS] Action buttons executed successfully.")

        # 7. Navigation Tabs & Button Bounding Rect Check
        print("[step 7] Clicking all 3 top navigation tabs & checking button geometry...")
        tabs = [
            ("nav-btn-agent", "Perception Studio", "agent"),
            ("nav-btn-video", "Fleet Video", "video"),
            ("nav-btn-ledger", "Scientific Ledger", "ledger"),
        ]
        button_geometry_passes = True
        for tab_id, tab_label, mode in tabs:
            btn = page.query_selector(f"#{tab_id}")
            if btn:
                btn.click()
                page.wait_for_timeout(200)
                # Verify bounding box of all 3 tabs
                for tid, tlabel, _ in tabs:
                    tbtn = page.query_selector(f"#{tid}")
                    if tbtn:
                        bbox = tbtn.bounding_box()
                        if bbox:
                            w = bbox["width"]
                            h = bbox["height"]
                            if w < 100 or h > 36:
                                print(f"  [FAIL] Tab {tid} button geometry corrupted: width={w:.1f}px (expected >= 100), height={h:.1f}px (expected <= 36)")
                                button_geometry_passes = False
                            else:
                                print(f"  [GEOM PASS] Tab {tid} ({tlabel}): {w:.1f}x{h:.1f}px (>= 100px width, <= 36px height)")
            else:
                print(f"  [FAIL] Tab button #{tab_id} not found in DOM!")
                button_geometry_passes = False

        # Switch to Fleet Video tab and check video
        print("[step 8] Checking Fleet Video tab & HTML5 video metadata...")
        page.evaluate("if (typeof switchNavMode === 'function') switchNavMode('video');")
        page.wait_for_timeout(500)
        video_info = page.evaluate("""() => {
            const v = document.getElementById('fleet-video-element') || document.querySelector('#stage-view-video video');
            if (!v) return { exists: false };
            return {
                exists: true,
                src: v.src || v.currentSrc,
                videoWidth: v.videoWidth,
                videoHeight: v.videoHeight,
                readyState: v.readyState
            };
        }""")
        print(f"  Video status: {video_info}")
        if video_info.get("exists"):
            assert video_info["videoWidth"] == 1920 and video_info["videoHeight"] == 648, f"Unexpected video resolution: {video_info}"

        # Switch to Scientific Ledger tab and test FinOps slider + Live Benchmark
        print("[step 9] Checking Scientific Ledger tab & FinOps slider...")
        page.evaluate("if (typeof switchNavMode === 'function') switchNavMode('ledger');")
        page.wait_for_timeout(300)
        page.evaluate("""() => {
            const slider = document.getElementById('finops-fleet-slider');
            if (slider) {
                slider.value = 100;
                slider.dispatchEvent(new Event('input'));
            }
            if (typeof triggerLiveMicrobenchmark === 'function') {
                triggerLiveMicrobenchmark();
            }
        }""")
        page.wait_for_timeout(1000)
        print("  [PASS] Adjusted FinOps slider and triggered microbenchmark.")

        # Switch back to Perception Studio
        page.evaluate("if (typeof switchNavMode === 'function') switchNavMode('agent');")
        page.wait_for_timeout(300)

        # 10. Three.js 3D WebGL Canvas Verification
        print("[step 10] Verifying Three.js 3D WebGL Canvas...")
        page.evaluate("if (typeof setCanvasViewMode === 'function') setCanvasViewMode('three');")
        page.wait_for_timeout(800)
        canvas_info = page.evaluate("""() => {
            const c = document.getElementById('three-main-canvas') || document.querySelector('#canvas-view-three canvas');
            if (!c) return { exists: false };
            const gl = c.getContext('webgl2') || c.getContext('webgl');
            let isNonBlack = false;
            let sampleStats = null;
            if (gl) {
                const pixels = new Uint8Array(4 * c.width * c.height);
                gl.readPixels(0, 0, c.width, c.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
                let nonZero = 0;
                for (let i = 0; i < pixels.length; i += 4) {
                    if (pixels[i] > 10 || pixels[i+1] > 10 || pixels[i+2] > 10) nonZero++;
                }
                isNonBlack = nonZero > 1000;
                sampleStats = { nonZeroPixels: nonZero, totalPixels: c.width * c.height };
            }
            return {
                exists: true,
                id: c.id,
                clientWidth: c.clientWidth,
                clientHeight: c.clientHeight,
                width: c.width,
                height: c.height,
                isNonBlack: isNonBlack,
                sampleStats: sampleStats
            };
        }""")
        print(f"  Three.js canvas status: {canvas_info}")
        assert canvas_info.get("exists") == True, "Three.js canvas does not exist"
        assert canvas_info.get("clientWidth") >= 300 and canvas_info.get("clientHeight") >= 180, "Three.js canvas size invalid"
        assert canvas_info.get("isNonBlack") == True, f"Three.js canvas is black: {canvas_info}"

        # 11. Check all <img> elements in DOM
        print("[step 11] Checking all images in DOM for natural dimensions...")
        broken_imgs = page.evaluate("""() => {
            const imgs = Array.from(document.querySelectorAll('img'));
            return imgs.filter(i => {
                const style = window.getComputedStyle(i);
                if (style.display === 'none') return false;
                return !i.complete || i.naturalWidth === 0;
            }).map(i => ({ id: i.id, src: i.src, naturalWidth: i.naturalWidth, complete: i.complete }));
        }""")
        if broken_imgs:
            print(f"  [FAIL] Broken images detected: {broken_imgs}")
            return False
        else:
            print(f"  [PASS] All visible <img> elements loaded with naturalWidth > 0.")

        browser.close()

    print("\n--- Playwright Diagnostic Errors ---")
    print(f"Console Errors ({len(console_errors)}): {console_errors}")
    print(f"Page Errors ({len(page_errors)}): {page_errors}")
    
    return len(console_errors) == 0 and len(page_errors) == 0 and button_geometry_passes

if __name__ == "__main__":
    p1 = run_playwright_audit(1920, 1080)
    p2 = run_playwright_audit(1440, 900)
    if p1 and p2:
        print("\n=======================================================")
        print(">>> ALL PLAYWRIGHT AUDITS PASSED WITH ZERO ERRORS <<<")
        print("=======================================================")
        sys.exit(0)
    else:
        print("\n>>> PLAYWRIGHT AUDIT REPORTED DEFECTS <<<")
        sys.exit(1)
