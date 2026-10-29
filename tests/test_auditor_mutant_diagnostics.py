"""Fail-closed source and execution controls for the mutation diagnostic collector."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/cbz/auditor_mutant_diagnostics.py"


class DiagnosticsTests(unittest.TestCase):
    def diagnostic(self):
        self.assertTrue(TOOL.is_file(), "mutation diagnostic source is required")
        module = types.ModuleType("auditor_mutation_diagnostic_control")
        module.__file__ = str(TOOL)
        exec(compile(TOOL.read_bytes(), str(TOOL), "exec"), module.__dict__)
        return module

    def test_literal_registry_preserves_every_mutant_and_target(self):
        mod = self.diagnostic()
        text = (
            "MUTANTS = [('first', 'old', 'new')]\n"
            "MUTANTS.extend([('second', 'left', 'right')])\n"
            "EXPECTED_FAILURE = {'first': 'test_one'}\n"
            "EXPECTED_FAILURE.update({'second': 'test_two'})\n")
        mutants, targets = mod.parse_registry(text)
        self.assertEqual(mutants, [("first", "old", "new"), ("second", "left", "right")])
        self.assertEqual(targets, {"first": "test_one", "second": "test_two"})

    def test_registry_refuses_nonliteral_duplicate_and_unmapped_cases(self):
        mod = self.diagnostic()
        cases = [
            "MUTANTS = dangerous()\nEXPECTED_FAILURE = {}\n",
            "MUTANTS = [('a','x','y'),('a','z','q')]\nEXPECTED_FAILURE = {'a':'test_a'}\n",
            "MUTANTS = [('a','x','y')]\nEXPECTED_FAILURE = {}\n",
            "MUTANTS = []\nMUTANTS.append(('a','x','y'))\nEXPECTED_FAILURE = {}\n",
            "MUTANTS = [('a','x','y')]\nMUTANTS[0] = ('a','x','z')\nEXPECTED_FAILURE = {'a':'test_a'}\n",
            "MUTANTS = [('a','x','y')]\nEXPECTED_FAILURE = {'a':'test_a','a':'test_b'}\n",
        ]
        for source in cases:
            with self.subTest(source=source), self.assertRaises(ValueError):
                mod.parse_registry(source)

    def test_control_resolution_requires_one_exact_test(self):
        mod = self.diagnostic()
        samples = {
            "tests.one": "class C:\n def test_a(self): pass\n",
            "tests.two": "class D:\n def test_b(self): pass\n",
        }
        self.assertEqual(mod.resolve_target(samples, "test_a"), "tests.one.C.test_a")
        with self.assertRaises(ValueError):
            mod.resolve_target(samples, "test_missing")
        samples["tests.two"] = "class D:\n def test_a(self): pass\n"
        with self.assertRaises(ValueError):
            mod.resolve_target(samples, "test_a")

    def test_mutation_requires_exactly_one_source_anchor(self):
        mod = self.diagnostic()
        self.assertEqual(mod.make_mutant("one old two", "old", "new"), "one new two")
        for source in ("none", "old old"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                mod.make_mutant(source, "old", "new")

    def test_source_mutations_expose_the_exact_guards(self):
        self.diagnostic()
        source = TOOL.read_text()
        mutations = (
            ("len(set(names)) != len(names)", "False", "duplicate"),
            ("source.count(old) != 1", "False", "anchor"),
        )
        for old, new, kind in mutations:
            with self.subTest(kind=kind):
                self.assertEqual(source.count(old), 1)
                module = types.ModuleType("diagnostic_mutant")
                module.__file__ = str(TOOL)
                exec(compile(source.replace(old, new), str(TOOL), "exec"), module.__dict__)
                if kind == "duplicate":
                    value = ("MUTANTS=[('a','x','y'),('a','z','q')]\n"
                             "EXPECTED_FAILURE={'a':'test_a'}\n")
                    result, _ = module.parse_registry(value)
                    self.assertEqual(len(result), 2)
                else:
                    self.assertEqual(module.make_mutant("old old", "old", "new"), "new old")

    def test_actual_nonzero_child_and_both_streams_are_preserved(self):
        mod = self.diagnostic()
        result = mod.observe(
            [sys.executable, "-B", "-S", "-c",
             "import sys; print('OUT'); print('ERR', file=sys.stderr); sys.exit(7)"],
            ROOT, dict(os.environ), timeout=10)
        self.assertEqual(result["kind"], "COMPLETED")
        self.assertEqual(result["returncode"], 7)
        self.assertEqual(result["stdout"], "OUT\n")
        self.assertEqual(result["stderr"], "ERR\n")
        self.assertNotIn("passed", result)
        self.assertNotIn("killed", result)

    def test_launch_failure_is_recorded_not_a_mutation_kill(self):
        mod = self.diagnostic()
        with tempfile.TemporaryDirectory() as directory:
            result = mod.observe([str(Path(directory) / "missing-executable")],
                                 ROOT, dict(os.environ), timeout=10)
        self.assertEqual(result["kind"], "LAUNCH_ERROR")
        self.assertIsNone(result["returncode"])
        self.assertNotIn("passed", result)
        self.assertNotIn("killed", result)


if __name__ == "__main__":
    unittest.main()
