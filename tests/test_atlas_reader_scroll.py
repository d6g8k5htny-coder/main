"""Optional browser regression for selecting a claim after a long source read."""
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
BROWSER = os.environ.get("ATLAS_BROWSER_EXECUTABLE") or shutil.which("google-chrome") or shutil.which("chromium")


@unittest.skipUnless(sync_playwright and BROWSER,
                     "Optional Playwright and a local Chromium browser are required")
class AtlasReaderScroll(unittest.TestCase):
    def test_new_selection_after_a_long_read_keeps_its_heading_visible(self):
        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0),
                                     partial(QuietHandler, directory=str(ROOT / "docs")))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        rows = []
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(executable_path=BROWSER,
                                                     headless=True, chromium_sandbox=True)
                try:
                    for width, height in [(1440, 1000), (390, 844)]:
                        with self.subTest(width=width):
                            page = browser.new_page(viewport={"width": width, "height": height},
                                                    reduced_motion="reduce")
                            try:
                                page.goto(f"http://127.0.0.1:{server.server_port}/site/dependencies.html?atlas-view=list")
                                page.wait_for_function("() => !document.querySelector('#atlas-search').disabled")
                                if width < 721:
                                    page.locator("#atlas-open-reader").click()
                                reader = page.locator(".atlas-reader")
                                measure = """e => {
                                    const h=e.querySelector('#atlas-inspector h3'),r=h.getBoundingClientRect();
                                    const s=getComputedStyle(e);
                                    return {scrollTop:e.scrollTop,pageScroll:scrollY,heading:h.innerText,
                                        headingTop:r.top,headingBottom:r.bottom,viewport:innerHeight,
                                        overflow:s.overflowY,maxHeight:s.maxHeight,
                                        documentWidth:document.documentElement.scrollWidth};
                                }"""
                                row = {"width": width, "browser": browser.version,
                                       "initial": reader.evaluate(measure)}
                                rows.append(row)
                                reader.hover()
                                before_wheel = reader.evaluate(measure)
                                page.mouse.wheel(0, 1500)
                                page.wait_for_function(
                                    "before => document.querySelector('.atlas-reader').scrollTop > before.scrollTop || scrollY > before.pageScroll",
                                    arg=before_wheel)
                                row["after_read"] = reader.evaluate(measure)
                                if width < 721:
                                    page.locator("#atlas-browse-link").click()
                                page.locator("#atlas-search").fill("math.d5-component.punctured-pin-proof")
                                page.locator("#atlas-list button").filter(
                                    has_text="math.d5-component.punctured-pin-proof").first.click()
                                page.wait_for_function("() => document.querySelector('#atlas-inspector h3').textContent === 'math.d5-component.punctured-pin-proof'")
                                row["new_selection"] = reader.evaluate(measure)
                                row["selection_focus"] = page.evaluate("() => document.activeElement.id")
                                evidence = os.environ.get("ATLAS_READER_EVIDENCE")
                                if evidence:
                                    destination = Path(evidence)
                                    destination.parent.mkdir(parents=True, exist_ok=True)
                                    page.screenshot(path=str(destination.with_name(destination.name + f"-{width}.png")))
                                    destination.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
                                selected = row["new_selection"]
                                self.assertGreaterEqual(selected["headingTop"], 0, row)
                                self.assertLessEqual(selected["headingBottom"], height, row)
                                self.assertEqual(row["selection_focus"], "atlas-inspector")
                                self.assertLessEqual(selected["documentWidth"], width)
                                page.keyboard.press("Tab")
                                source = page.evaluate("""() => {
                                    const e=document.activeElement,r=e.getBoundingClientRect();
                                    return {tag:e.tagName,href:e.href,visible:r.top>=0&&r.bottom<=innerHeight,
                                        inside:document.querySelector('#atlas-inspector').contains(e)};
                                }""")
                                row["keyboard_source"] = source
                                self.assertEqual(source["tag"], "A")
                                self.assertTrue(source["inside"] and source["visible"], source)
                                if width < 721:
                                    self.assertEqual(selected["overflow"], "visible")
                                    self.assertEqual(selected["maxHeight"], "none")
                                if evidence:
                                    destination.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
                            finally:
                                page.close()
                finally:
                    browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
