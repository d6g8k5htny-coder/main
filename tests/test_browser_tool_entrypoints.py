import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BrowserToolEntrypointContract(unittest.TestCase):
    """Exercise parser/import boundaries, not browser scenario semantics."""

    def run_help(self, relative_path):
        tool = ROOT / relative_path
        mode = (["-OO"] if sys.flags.optimize == 2 else
                (["-O"] if sys.flags.optimize == 1 else []))
        environment = os.environ.copy()
        environment.pop("PYTHONOPTIMIZE", None)
        return subprocess.run(
            [sys.executable, "-B", *mode, str(tool), "--help"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def assert_help(self, relative_path):
        result = self.run_help(relative_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertTrue(result.stdout.startswith("usage:"), result.stdout)
        self.assertIn(Path(relative_path).name, result.stdout.splitlines()[0])

    def test_atlas_browser_entrypoints_advertise_their_actual_cli(self):
        self.assert_help("tools/atlas_accessibility_audit.py")
        self.assert_help("tools/atlas_accessibility_regression.py")
        self.assert_help("tools/atlas_accessible_list_browser_check.py")
        self.assert_help("tools/atlas_approachable_browser_check.py")
        self.assert_help("tools/atlas_claim_summary_browser_check.py")
        self.assert_help("tools/atlas_formal_evidence_browser_check.py")
        self.assert_help("tools/atlas_live_controls_browser_check.py")
        self.assert_help("tools/atlas_refusal_browser_check.py")
        self.assert_help("tools/atlas_relationships_browser_check.py")
        self.assert_help("tools/atlas_resilience_browser_check.py")
        self.assert_help("tools/atlas_source_bridge_browser_check.py")
        self.assert_help("tools/atlas_spatial_browser_check.py")
        self.assert_help("tools/atlas_spatial_labels_browser_check.py")
        self.assert_help("tools/atlas_webgl_fallback_browser_check.py")

    def test_reader_browser_entrypoints_advertise_their_actual_cli(self):
        self.assert_help("tools/compare_experience_browser_check.py")
        self.assert_help("tools/figure_png_browser_check.py")
        self.assert_help("tools/library_discovery_browser_check.py")
        self.assert_help("tools/library_source_search_browser_check.py")
        self.assert_help("tools/persistent_status_browser_check.py")
        self.assert_help("tools/reader_jump_browser_check.py")
        self.assert_help("tools/reading_list_citation_browser_check.py")
        self.assert_help("tools/research_desk_browser_check.py")
        self.assert_help("tools/research_desk_continuity_browser_check.py")
        self.assert_help("tools/research_desk_undo_browser_check.py")
        self.assert_help("tools/research_reading_browser_check.py")
        self.assert_help("tools/snapshot_compare_browser_check.py")
        self.assert_help("tools/source_list_continuity_browser_check.py")
        self.assert_help("tools/website_navigation_browser_check.py")

    def test_missing_tool_cannot_pass_the_help_contract(self):
        result = self.run_help("tools/THIS_BROWSER_TOOL_DOES_NOT_EXIST.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertNotEqual(result.stderr, "")

    def test_help_refusal_cannot_pass_the_help_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "refuses_help.py"
            script.write_text(textwrap.dedent("""\
                import sys
                if "--help" in sys.argv:
                    raise SystemExit(7)
            """))
            result = self.run_help(script)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
