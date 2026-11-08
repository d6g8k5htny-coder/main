"""Repository integration control for the current Atlas accessibility CLI."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "cbz"))
import g11_census as g11  # noqa: E402


class CurrentAtlasAccessibilityRegressionReference(unittest.TestCase):
    paths = (
        "tools/atlas_accessibility_regression.py",
        ".github/workflows/public-shop.yml",
    )

    def census(self, replacements=None):
        replacements = replacements or {}
        files = {
            path: replacements.get(path, (ROOT / path).read_bytes())
            for path in self.paths
        }
        return g11.census(files, sorted(files))["tools"]

    def test_atlas_accessibility_regression_has_static_execution_reference(self):
        report = self.census()
        self.assertTrue(
            report["tools/atlas_accessibility_regression.py"]["referenced"]
        )

    def test_removing_workflow_invocation_is_detected(self):
        workflow = (ROOT / self.paths[1]).read_bytes()
        command = b"python -B tools/atlas_accessibility_regression.py"
        self.assertEqual(workflow.count(command), 1)
        report = self.census({self.paths[1]: workflow.replace(command, b"python -B")})
        self.assertFalse(
            report["tools/atlas_accessibility_regression.py"]["referenced"]
        )


if __name__ == "__main__":
    unittest.main()
