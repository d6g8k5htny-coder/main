"""Pages that load current.mjs must permit its same-origin fetch of current.json."""
import re
from pathlib import Path
import unittest

SITE = Path(__file__).resolve().parents[1] / "docs/site"
CSP = re.compile(r'<meta\s+http-equiv="Content-Security-Policy"\s+content="([^"]*)"', re.I)
LOADS_CURRENT = re.compile(r'<script[^>]+src="current\.mjs(?:\?[^"]*)?"', re.I)


def connect_sources(policy):
    directives = {}
    for part in policy.split(";"):
        words = part.split()
        if words:
            directives.setdefault(words[0].lower(), words[1:])
    return directives.get("connect-src", directives.get("default-src", []))


class CurrentModuleCspTests(unittest.TestCase):
    def test_pages_loading_current_module_allow_same_origin_connect(self):
        pages = sorted(p for p in SITE.glob("*.html") if LOADS_CURRENT.search(p.read_text(encoding="utf-8")))
        self.assertTrue(pages, "no docs/site page loads current.mjs")
        for page in pages:
            with self.subTest(page=page.name):
                policies = CSP.findall(page.read_text(encoding="utf-8"))
                self.assertEqual(len(policies), 1, "expected exactly one CSP meta")
                sources = connect_sources(policies[0])
                self.assertNotIn("'none'", sources)
                self.assertIn("'self'", sources, "current.mjs fetches same-origin current.json")


if __name__ == "__main__":
    unittest.main()
