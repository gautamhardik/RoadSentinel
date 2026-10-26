from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={'width': 1920, 'height': 1080})
    page.goto('http://localhost:8000')
    page.wait_for_timeout(1000)
    
    # Check tabs
    for tid in ['nav-btn-agent', 'nav-btn-video', 'nav-btn-ledger']:
        btn = page.query_selector(f'#{tid}')
        box = btn.bounding_box()
        w = box['width']
        h = box['height']
        print(f"Tab {tid}: visible={btn.is_visible()}, width={w:.1f}px, height={h:.1f}px")
        
    # Switch canvas view mode to crop
    page.evaluate('setCanvasViewMode("crop");')
    page.wait_for_timeout(500)
    handle = page.query_selector('#curtain-stage-handle')
    box = handle.bounding_box()
    print(f"Curtain handle: visible={handle.is_visible()}, box={box}")
    
    # Switch canvas view mode to three
    page.evaluate('setCanvasViewMode("three");')
    page.wait_for_timeout(800)
    three_canvas = page.query_selector('#three-main-canvas')
    tbox = three_canvas.bounding_box()
    print(f"Three canvas: visible={three_canvas.is_visible()}, box={tbox}")

    # Read WebGL canvas pixels
    gl_status = page.evaluate("""() => {
        const c = document.getElementById('three-main-canvas');
        const gl = c.getContext('webgl2') || c.getContext('webgl');
        if (!gl) return { error: 'no gl' };
        const pixels = new Uint8Array(4 * c.width * c.height);
        gl.readPixels(0, 0, c.width, c.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        let nonZero = 0;
        for (let i = 0; i < pixels.length; i += 4) {
            if (pixels[i] > 10 || pixels[i+1] > 10 || pixels[i+2] > 10) nonZero++;
        }
        return { width: c.width, height: c.height, nonZeroPixels: nonZero };
    }""")
    print("WebGL Pixel Check:", gl_status)
    
    browser.close()
