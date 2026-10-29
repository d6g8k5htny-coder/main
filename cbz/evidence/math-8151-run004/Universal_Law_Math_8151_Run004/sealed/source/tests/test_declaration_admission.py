"""Regression tests for RF-GATE-01 declaration admission.

These tests exercise the real source-only gate in a copied formal tree. They do
not run Lean; hosted execute-time controls remain separate.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

FORMAL = Path(__file__).resolve().parents[1]
MODULE = "ResearchFormalCoreR1/AlgebraV2.lean"
PREFIX = "ResearchFormalCoreR1."


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DeclarationAdmissionRegressionTests(unittest.TestCase):
    def fixture(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name) / "formal"
        shutil.copytree(
            FORMAL,
            root,
            ignore=shutil.ignore_patterns(".lake", "__pycache__", "*.pyc"),
        )
        return td, root

    def write_module(self, root: Path, extra: str):
        p = root / MODULE
        text = p.read_text()
        marker = "\nend ResearchFormalCoreR1\n"
        self.assertIn(marker, text)
        p.write_text(text.replace(marker, "\n" + extra + "\n" + marker, 1))
        mpath = root / "manifest.json"
        manifest = json.loads(mpath.read_text())
        manifest["files"][MODULE] = sha256(p)
        mpath.write_text(json.dumps(manifest, indent=2) + "\n")

    def run_gate(self, root: Path):
        return subprocess.run(
            [sys.executable, "gate.py"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )

    def assert_rejected(self, extra: str):
        td, root = self.fixture()
        try:
            self.write_module(root, extra)
            result = self.run_gate(root)
            self.assertNotEqual(
                result.returncode,
                0,
                "source gate admitted an unregistered Lean declaration:\n"
                + extra
                + "\nstdout:\n"
                + result.stdout
                + "\nstderr:\n"
                + result.stderr,
            )
        finally:
            td.cleanup()

    def test_hidden_declaration_forms_are_rejected(self):
        cases = (
            "  theorem rf_gate_indented : False := by sorry",
            "private theorem rf_gate_private : False := by sorry",
            "@[simp] theorem rf_gate_attributed : False := by sorry",
            "def rf_gate_unused : False := by sorry",
            "opaque rf_gate_opaque : False",
            "axiom rf_gate_axiom : False",
        )
        for source in cases:
            with self.subTest(source=source):
                self.assert_rejected(source)

    def test_comment_text_is_not_inventoried_as_a_declaration(self):
        td, root = self.fixture()
        try:
            self.write_module(
                root,
                "/-\ntheorem rf_gate_comment_ghost : False := by sorry\n-/",
            )
            result = self.run_gate(root)
            self.assertEqual(
                result.returncode,
                0,
                "comment text was treated as a package declaration:\n"
                + result.stdout
                + result.stderr,
            )
        finally:
            td.cleanup()

    def test_duplicate_target_list_is_rejected_source_side(self):
        td, root = self.fixture()
        try:
            extra = "theorem ec005_fold_gap (s : ℝ) : True := by trivial"
            self.write_module(root, extra)
            mpath = root / "manifest.json"
            manifest = json.loads(mpath.read_text())
            name = PREFIX + "ec005_fold_gap"
            insert_at = manifest["targets"].index(PREFIX + "ec014_contact_power") + 1
            manifest["targets"].insert(insert_at, name)
            mpath.write_text(json.dumps(manifest, indent=2) + "\n")
            result = self.run_gate(root)
            self.assertNotEqual(
                result.returncode,
                0,
                "duplicate source/manifest target escaped source-side validation",
            )
        finally:
            td.cleanup()

    def test_nested_namespace_cannot_be_misprefixed(self):
        td, root = self.fixture()
        try:
            extra = (
                "namespace Nested\n"
                "theorem rf_gate_nested : True := by trivial\n"
                "end Nested"
            )
            self.write_module(root, extra)
            mpath = root / "manifest.json"
            manifest = json.loads(mpath.read_text())
            insert_at = manifest["targets"].index(PREFIX + "ec014_contact_power") + 1
            manifest["targets"].insert(insert_at, PREFIX + "rf_gate_nested")
            mpath.write_text(json.dumps(manifest, indent=2) + "\n")
            result = self.run_gate(root)
            self.assertNotEqual(
                result.returncode,
                0,
                "nested declaration escaped with a fabricated flat namespace name",
            )
        finally:
            td.cleanup()


if __name__ == "__main__":
    unittest.main()
