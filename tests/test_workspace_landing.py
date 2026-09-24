"""Synthetic controls for the landing checker, using only disposable files."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "landing", Path(__file__).resolve().parents[1] / "tools/workspace_landing_check.py")
landing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(landing)


class LandingChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "docs").mkdir()
        (self.root / "README.md").write_text("[Guide](docs/guide.md)\n")
        (self.root / "docs/guide.md").write_text("[Home](../README.md)\n")
        self.original = b"historical bytes\n"
        (self.root / "old.txt").write_bytes(self.original)
        digest = hashlib.sha1(b"blob 17\0" + self.original).hexdigest()
        self.config = {"scope": "LANDING_LINKS_AND_HISTORICAL_IDENTITY_ONLY",
                       "pages": ["README.md", "docs/guide.md"],
                       "historical_files": [{"path": "old.txt", "bytes": 17,
                                             "git_blob_sha1": digest}]}
        self.save()

    def save(self):
        (self.root / "docs/LANDING_MANIFEST.json").write_text(json.dumps(self.config))

    def test_valid_and_no_writes(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = landing.check(self.root)
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["local_links"], 2)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_broken_local_link(self):
        (self.root / "README.md").write_text("[Missing](not_here.md)\n")
        self.assertIn("not_here", landing.check(self.root)["problems"][0])

    def test_modified_archive(self):
        (self.root / "old.txt").write_bytes(self.original + b"x")
        self.assertIn("identity mismatch", landing.check(self.root)["problems"][0])

    def test_missing_page(self):
        (self.root / "docs/guide.md").unlink()
        with self.assertRaises(ValueError):
            landing.check(self.root)

    def test_parent_escape(self):
        (self.root / "README.md").write_text("[Bad](../../outside.txt)\n")
        self.assertTrue(landing.check(self.root)["problems"])

    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "source.txt"
            target.write_text("outside")
            (self.root / "link.txt").symlink_to(target)
            (self.root / "README.md").write_text("[Bad](link.txt)\n")
            self.assertTrue(landing.check(self.root)["problems"])

    def test_bool_not_byte_count(self):
        self.config["historical_files"][0]["bytes"] = True
        self.save()
        with self.assertRaises(ValueError):
            landing.check(self.root)

    def test_duplicate_json_key(self):
        (self.root / "docs/LANDING_MANIFEST.json").write_text('{"pages":[],"pages":[]}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            landing.check(self.root)

    def test_duplicate_page(self):
        self.config["pages"].append("README.md")
        self.save()
        with self.assertRaises(ValueError):
            landing.check(self.root)

    def test_remote_and_fenced_examples_not_fetched(self):
        (self.root / "README.md").write_text(
            "[Remote](https://example.invalid/no-network)\n[Anchor](#heading)\n"
            "```sh\n[Example](missing.md)\n```\n[Guide](docs/guide.md)\n")
        result = landing.check(self.root)
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["local_links"], 2)

    def test_invalid_schema_cli(self):
        self.config["scope"] = "not-science"
        self.save()
        self.assertEqual(landing.main(["--root", str(self.root)]), 2)

    def test_cli_corruption_is_failure(self):
        (self.root / "old.txt").write_bytes(b"corrupt")
        self.assertEqual(landing.main(["--root", str(self.root)]), 1)

    def test_batch248_workspace_landing_workflow_path_split(self):
        """Landing checks must not share .github/workflows/ci.yml with hardening CI.

        Evidence (Batch 248): workflow_id for path ci.yml was named
        workspace-landing on default main, so 12–22m hardening verifies
        appeared under the workspace-landing Actions filter.
        """
        root = Path(__file__).resolve().parents[1]
        wl = (root / ".github/workflows/workspace-landing.yml").read_text(encoding="utf-8")
        ci = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("name: workspace-landing", wl)
        self.assertIn("Landing checker tests", wl)
        self.assertIn("tools/workspace_landing_check.py", wl)
        self.assertIn("name: ci", ci)
        self.assertNotIn("Landing checker tests", ci)
        self.assertNotIn("workspace_landing_check.py", ci)
        self.assertIn("workflow_dispatch", ci)
        self.assertIn("path holder", ci.lower().replace("-", " "))


if __name__ == "__main__":
    unittest.main()
