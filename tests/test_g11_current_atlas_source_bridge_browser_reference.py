"""Repository integration control for the Atlas source-bridge browser CLI."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "cbz"))
import g11_census as g11  # noqa: E402


class CurrentAtlasSourceBridgeBrowserReference(unittest.TestCase):
    paths = (
        "tools/atlas_source_bridge_browser_check.py",
        ".github/workflows/public-shop.yml",
    )

    def census(self, replacements=None):
        replacements = replacements or {}
        files = {
            path: replacements.get(path, (ROOT / path).read_bytes())
            for path in self.paths
        }
        return g11.census(files, sorted(files))["tools"]

    def test_source_bridge_browser_check_has_static_execution_reference(self):
        report = self.census()
        self.assertTrue(
            report["tools/atlas_source_bridge_browser_check.py"]["referenced"]
        )

    def test_removing_workflow_invocation_is_detected(self):
        workflow = (ROOT / self.paths[1]).read_bytes()
        command = b"python -B tools/atlas_source_bridge_browser_check.py"
        self.assertEqual(workflow.count(command), 1)
        report = self.census(
            {self.paths[1]: workflow.replace(command, b"python -B")}
        )
        self.assertFalse(
            report["tools/atlas_source_bridge_browser_check.py"]["referenced"]
        )


if __name__ == "__main__":
    unittest.main()
