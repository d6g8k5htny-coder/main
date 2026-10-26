from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RouteWindowWorkflowTest(unittest.TestCase):
    def test_workflow_is_read_only_pinned_and_fail_closed(self) -> None:
        text = (ROOT / ".github/workflows/current-site-route-window.yml").read_text(encoding="utf-8")
        self.assertIn("runs-on: ubuntu-24.04", text)
        self.assertIn("timeout-minutes: 10", text)
        self.assertGreaterEqual(text.count("contents: read"), 2)
        self.assertIn("persist-credentials: false", text)
        self.assertIn("actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", text)
        self.assertIn("actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97", text)
        self.assertIn("actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9", text)
        self.assertIn("--require-hashes --only-binary=:all:", text)
        self.assertNotIn("playwright install", text)
        self.assertNotIn("secrets.", text)
        self.assertNotIn("contents: write", text)
        self.assertNotIn("deploy", text.lower())
        self.assertLess(
            text.index("Preserve the exact generated dependency page"),
            text.index("Require the checked-in dependency page"),
        )
        self.assertIn("if: always()", text)

    def test_browser_contract_uses_admitted_payload_and_exact_budget(self) -> None:
        text = (ROOT / "tools/current_site_route_window_browser_check.py").read_text(encoding="utf-8")
        self.assertIn("SIZE = 10_000", text)
        self.assertIn("LIMIT = 150", text)
        self.assertIn("data-current-digest", text)
        self.assertIn("data-current-bytes", text)
        self.assertIn("50", str(sum(range(2, 10_001))))
        self.assertIn("#atlas-route-list [data-route-node]", text)
        self.assertIn("#atlas-route-list [data-route-edge]", text)
        self.assertIn("page.reload()", text)
        self.assertIn("page.go_back()", text)


if __name__ == "__main__":
    unittest.main()
