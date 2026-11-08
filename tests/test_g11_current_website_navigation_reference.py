"""Repository integration control for the current navigation browser CLI."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "cbz"))
import g11_census as g11  # noqa: E402


class CurrentWebsiteNavigationReference(unittest.TestCase):
    def test_website_navigation_browser_check_has_static_execution_reference(self):
        paths = (
            "tools/website_navigation_browser_check.py",
            ".github/workflows/public-shop.yml",
        )
        files = {path: (ROOT / path).read_bytes() for path in paths}
        report = g11.census(files, sorted(files))["tools"]
        self.assertTrue(
            report["tools/website_navigation_browser_check.py"]["referenced"]
        )


if __name__ == "__main__":
    unittest.main()
