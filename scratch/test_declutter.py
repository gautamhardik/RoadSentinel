import time
from playwright.sync_api import sync_playwright

def test_declutter():
    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        page.on("console", lambda msg: console_errors.append(f"[{msg.type}] {msg.text}") if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        print("Navigating to http://localhost:8000 ...")
        page.goto("http://localhost:8000", wait_until="domcontentloaded", timeout=15000)
        page.wait_for_selector("#scenario-tree-list button", timeout=10000)
        page.wait_for_timeout(1000)

        # 1. Verify Top Bar Header
        header = page.locator("header")
        header_text = header.inner_text()
        print("Header Text:", repr(header_text))

        assert "STUDIO v3.2" not in header_text, "Error: STUDIO v3.2 still visible in header!"
        assert "roadsentinel_ultimate_champion" not in header_text, "Error: Champion model filename still visible in header!"
        print("[PASS] Top header free of version badge and raw model filename.")

        # 2. Verify Bottom Footer
        footer = page.locator("footer")
        footer_text = footer.inner_text()
        print("Footer Text:", repr(footer_text))

        assert "SHA-256" not in footer_text, "Error: SHA-256 still visible in footer!"
        assert "MCP PROTOCOL ONLINE" not in footer_text, "Error: MCP PROTOCOL ONLINE still visible in footer!"
        assert "ASYNC_WBF_ACTIVE" not in footer_text, "Error: ASYNC_WBF_ACTIVE still visible in footer!"
        assert "System Online" in footer_text, "Error: System Online status dot missing!"
        print("[PASS] Bottom footer free of SHA-256 hash, time stamp, and protocol flags.")

        # 3. Capture Screenshots
        header.screenshot(path="scratch/decluttered_top_bar.png")
        footer.screenshot(path="scratch/decluttered_footer.png")
        page.screenshot(path="scratch/decluttered_studio_1440.png")
        print("[PASS] Screenshots saved to scratch/")

        browser.close()

    print(f"Console Errors: {len(console_errors)}")
    print(f"Page Errors: {len(page_errors)}")
    assert len(console_errors) == 0, f"Console errors found: {console_errors}"
    assert len(page_errors) == 0, f"Page errors found: {page_errors}"
    print("ALL DECLUTTER CHECKS PASSED WITH ZERO ERRORS!")

if __name__ == "__main__":
    test_declutter()
