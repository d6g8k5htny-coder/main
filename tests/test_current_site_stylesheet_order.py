"""Live generated node pages load style.css before brand.css (F2: brand.css dark/reduced-motion rules must win)."""
import json
import re
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SITE = ROOT / "docs/site"
LINK = re.compile(r'<link rel="stylesheet" href="(?:\.\./)*([a-z]+\.css)')


class StylesheetOrder(unittest.TestCase):
    def order(self, path):
        return LINK.findall(path.read_text(encoding="utf-8"))

    def test_live_node_pages_load_style_before_brand(self):
        pages = sorted((SITE / "nodes").glob("*.html"))
        self.assertTrue(pages)
        for page in pages:
            names = self.order(page)
            with self.subTest(page=page.name):
                self.assertIn("style.css", names)
                self.assertIn("brand.css", names)
                self.assertLess(names.index("style.css"), names.index("brand.css"))

    def test_generators_emit_style_before_brand(self):
        for tool, marker in (("current_site_build.py", "def static_page"), ("current_site_history.py", "def _live_v2_static_page")):
            text = (ROOT / "tools" / tool).read_text(encoding="utf-8")
            body = text[text.index(marker):]
            body = body[:body.index("</head>")]
            with self.subTest(tool=tool):
                self.assertLess(body.index("style.css"), body.index("brand.css"))

    def test_removed_node_live_route_loads_style_before_brand_and_archive_stays(self):
        # A later observation that drops a v2 node republishes nodes/<id>.html
        # through render_static_v2(..., archived=False). That convenience route
        # must follow the live order. The immutable snapshot keeps brand.css first.
        import test_current_site_build as build_controls
        import current_site_history as history

        controls = build_controls.CurrentSiteBuildTests()
        with TemporaryDirectory() as tmp:
            site, payload = controls.prepare(Path(tmp))
            first = controls.run_build(site, payload)
            self.assertEqual(first.returncode, 0, first.stderr)
            digest = history.mathematical_digest(build_controls.fixture())
            archive = site / "snapshots" / digest
            archived_before = {p.relative_to(archive).as_posix(): p.read_bytes()
                               for p in archive.rglob("*") if p.is_file()}
            self.assertEqual(self.order(archive / "nodes/beta.html"), ["brand.css", "style.css", "current.css"])
            data = build_controls.fixture()
            data["observed_at"] = "2026-10-13T13:00:00Z"
            data["graph"]["data"]["nodes"].pop("beta")
            data["graph"]["data"]["edges"] = [
                edge for edge in data["graph"]["data"]["edges"]
                if edge["from"] != "beta" and edge["to"] != "beta"]
            payload.write_bytes(json.dumps(data, ensure_ascii=False, indent=2).encode() + b"\n")
            second = controls.run_build(site, payload)
            self.assertEqual(second.returncode, 0, second.stderr)
            live = site / "nodes/beta.html"
            self.assertIn("Historical removed-node static route", live.read_text(encoding="utf-8"))
            self.assertEqual(self.order(live), ["style.css", "brand.css", "current.css"])
            self.assertEqual(self.order(archive / "nodes/beta.html"), ["brand.css", "style.css", "current.css"])
            archived_after = {p.relative_to(archive).as_posix(): p.read_bytes()
                              for p in archive.rglob("*") if p.is_file()}
            self.assertEqual(archived_before, archived_after)


if __name__ == "__main__":
    unittest.main()
