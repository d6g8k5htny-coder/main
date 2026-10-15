"""Live generated node pages load style.css before brand.css (F2: brand.css dark/reduced-motion rules must win)."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


if __name__ == "__main__":
    unittest.main()
