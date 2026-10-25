"""Optional real-browser regression for the Atlas list's scroll boundary."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import threading
import unittest

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

ROOT = Path(__file__).resolve().parents[1]
CHROMIUM = shutil.which("chromium") or shutil.which("google-chrome")


@unittest.skipUnless(sync_playwright and CHROMIUM,
                     "Optional Playwright and local Chromium are required")
class AtlasListFocusGeometry(unittest.TestCase):
    def test_keyboard_focus_ring_fits_inside_the_list_scroll_boundary(self):
        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0),
                                     partial(QuietHandler, directory=str(ROOT / "docs")))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(executable_path=CHROMIUM, headless=True)
                try:
                    page = browser.new_page(viewport={"width": 390, "height": 844},
                                            reduced_motion="reduce")
                    page.goto(f"http://127.0.0.1:{server.server_port}/site/dependencies.html")
                    page.wait_for_function("() => !document.querySelector('#atlas-search').disabled")
                    self.assertEqual(page.locator("#atlas-view").input_value(), "list")
                    page.locator("#atlas-view").focus()
                    for _ in range(12):
                        page.keyboard.press("Tab")
                        if page.evaluate("() => document.activeElement === document.querySelector('#atlas-list button')"):
                            break
                    self.assertTrue(page.evaluate("() => document.activeElement === document.querySelector('#atlas-list button')"))
                    page.locator("#atlas-list").scroll_into_view_if_needed()
                    measure = """() => {
                        const e=document.activeElement,s=getComputedStyle(e),r=e.getBoundingClientRect();
                        const list=document.querySelector('#atlas-list'),b=list.getBoundingClientRect();
                        return {ring:parseFloat(s.outlineWidth)+parseFloat(s.outlineOffset),
                            top:r.top-b.top-list.clientTop,left:r.left-b.left-list.clientLeft,
                            right:b.left+list.clientLeft+list.clientWidth-r.right,
                            bottom:b.top+list.clientTop+list.clientHeight-r.bottom,
                            focused:e.matches(':focus-visible'),outline:s.outlineStyle,
                            documentWidth:document.documentElement.scrollWidth};
                    }"""
                    first = page.evaluate(measure)
                    evidence = os.environ.get("ATLAS_FOCUS_EVIDENCE")
                    if evidence:
                        destination = Path(evidence)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        page.screenshot(path=str(destination.with_suffix(".png")))
                        destination.with_suffix(".json").write_text(json.dumps(first, indent=2) + "\n")
                    self.assertTrue(first["focused"])
                    self.assertNotEqual(first["outline"], "none")
                    self.assertGreater(first["ring"], 0)
                    for edge in ("top", "left", "right"):
                        self.assertGreaterEqual(first[edge], first["ring"],
                                                f"First node focus ring clipped at {edge}: {first}")
                    for _ in range(page.locator("#atlas-list button").count() - 1):
                        page.keyboard.press("Tab")
                    last = page.evaluate(measure)
                    for edge in ("left", "right", "bottom"):
                        self.assertGreaterEqual(last[edge], last["ring"],
                                                f"Last node focus ring clipped at {edge}: {last}")
                    self.assertLessEqual(last["documentWidth"], 390)
                    if evidence:
                        destination.with_suffix(".json").write_text(
                            json.dumps({"first": first, "last": last}, indent=2) + "\n")
                        page.screenshot(path=str(destination.with_name(destination.stem + "-last").with_suffix(".png")))
                finally:
                    browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
