"""Repository integration controls for current architecture CLI references."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "cbz"))
import g11_census as g11  # noqa: E402


ARCHITECTURE_TOOLS = (
    "tools/architecture_evidence_adapter.py",
    "tools/architecture_graph_artifact_readback.py",
    "tools/artifact_index_export.py",
)

REFERENCE_SOURCES = (
    "tests/test_architecture_evidence_adapter.py",
    "tests/test_architecture_graph_artifact_readback.py",
    "tests/test_architecture_artifact_index.py",
    "tests/test_architecture_cli_boundaries.py",
)


class CurrentArchitectureReferences(unittest.TestCase):
    def test_architecture_clis_have_static_execution_references(self):
        paths = (*ARCHITECTURE_TOOLS, *REFERENCE_SOURCES)
        files = {path: (ROOT / path).read_bytes() for path in paths}
        report = g11.census(files, sorted(files))["tools"]
        self.assertEqual(
            {path: report[path]["referenced"] for path in ARCHITECTURE_TOOLS},
            {
                "tools/architecture_evidence_adapter.py": True,
                "tools/architecture_graph_artifact_readback.py": True,
                "tools/artifact_index_export.py": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
