"""Repository integration control for the current Atlas formal bridge CLI."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "cbz"))
import g11_census as g11  # noqa: E402


class CurrentAtlasFormalBridgeReference(unittest.TestCase):
    def test_atlas_formal_source_bridge_has_static_execution_reference(self):
        paths = (
            "tools/atlas_formal_source_bridge.py",
            "tests/test_atlas_formal_source_bridge.py",
        )
        files = {path: (ROOT / path).read_bytes() for path in paths}
        report = g11.census(files, sorted(files))["tools"]
        self.assertTrue(
            report["tools/atlas_formal_source_bridge.py"]["referenced"]
        )


if __name__ == "__main__":
    unittest.main()
