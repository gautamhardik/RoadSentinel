#!/usr/bin/env python3
import time
from playwright.sync_api import sync_playwright

def capture_button_views():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto("http://localhost:8000")
        page.wait_for_selector("#scenario-tree-list button")
        time.sleep(1.5)

        # 1. Capture Full Dashboard
        page.screenshot(path="scratch/better_buttons_studio_1440.png")

        # 2. Capture Top Bar
        top_bar = page.query_selector("header")
        if top_bar:
            top_bar.screenshot(path="scratch/better_buttons_top_bar.png")

        # 3. Capture Deliberation Console Action Panel
        action_bar = page.query_selector("#agent-thought-stream + div")
        if action_bar:
            action_bar.screenshot(path="scratch/better_buttons_deliberation.png")

        # 4. Capture View Switcher
        switcher = page.query_selector(".absolute.top-3.left-1\\/2")
        if switcher:
            switcher.screenshot(path="scratch/better_buttons_switcher.png")

        # 5. Capture Bottom Footer
        footer = page.query_selector("footer")
        if footer:
            footer.screenshot(path="scratch/better_buttons_footer.png")

        browser.close()
        print("All button screenshots successfully captured.")

if __name__ == "__main__":
    capture_button_views()
