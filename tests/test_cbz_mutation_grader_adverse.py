"""Adverse controls for CBZ scanner mutation graders and source-bound test loaders."""
import importlib._bootstrap_external
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    path = Path(path)
    raw = path.read_bytes()
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), mod.__dict__)
    return mod


class GraderAdverseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graders = [
            load(ROOT / "tests/test_cbz_layer_a_scan_mutants.py", "old_lexer_grader"),
            load(ROOT / "tests/test_cbz_a3_float_census.py", "old_a3_grader"),
        ]

    def sample(self, *, code=1, summary="failures=1", extra="", tests=1):
        text = ("FAIL: test_expected (fixture.Case)\n" + extra
                + "\nRan " + str(tests) + (" test" if tests == 1 else " tests")
                + " in 0.001s\nFAILED (" + summary + ")\n")
        return subprocess.CompletedProcess([], code, stdout="", stderr=text)

    def test_abnormal_exit_does_not_kill(self):
        for grader in self.graders:
            with self.subTest(grader=grader.__name__):
                self.assertFalse(grader.killed(self.sample(code=86), "test_expected"))

    def test_mixed_foreign_failure_does_not_kill(self):
        for grader in self.graders:
            with self.subTest(grader=grader.__name__):
                process = self.sample(summary="failures=2", extra="FAIL: test_foreign (fixture.Case)\n", tests=2)
                self.assertFalse(grader.killed(process, "test_expected"))

    def test_skips_and_inconsistent_failure_counts_do_not_kill(self):
        for grader in self.graders:
            for summary in ("failures=1, skipped=1", "failures=2"):
                with self.subTest(grader=grader.__name__, summary=summary):
                    self.assertFalse(grader.killed(self.sample(summary=summary), "test_expected"))

    def test_expected_assertion_without_specific_observation_does_not_kill(self):
        for grader in self.graders:
            with self.subTest(grader=grader.__name__):
                self.assertFalse(grader.killed(self.sample(), "test_expected"))

    def test_unnamed_target_does_not_kill(self):
        for grader in self.graders:
            with self.subTest(grader=grader.__name__):
                self.assertFalse(grader.killed(self.sample()))


class SourceLoaderAdverseTests(unittest.TestCase):
    def test_adverse_loader_ignores_poisoned_bytecode(self):
        self.assertIsNone(sys.pycache_prefix, "control requires the default cache location")
        with tempfile.TemporaryDirectory(prefix="cbz-a8-loader-control-") as directory:
            path = Path(directory) / "probe.py"
            path.write_text("GOOD_SOURCE = True\n")
            stat = path.stat()
            suffix = ".opt-" + str(sys.flags.optimize) if sys.flags.optimize else ""
            cache = path.parent / "__pycache__" / ("probe." + sys.implementation.cache_tag + suffix + ".pyc")
            cache.parent.mkdir()
            poisoned = compile("POISON_EXECUTED = True\n", str(path), "exec")
            cache.write_bytes(importlib._bootstrap_external._code_to_timestamp_pyc(
                poisoned, int(stat.st_mtime), stat.st_size))
            probe = load(path, "target")
            self.assertTrue(getattr(probe, "GOOD_SOURCE", False))
            self.assertFalse(getattr(probe, "POISON_EXECUTED", False))

    def test_grader_children_do_not_execute_poisoned_test_cache(self):
        for grader, module in (
            (self.load_grader("tests/test_cbz_layer_a_scan_mutants.py", "child_lexer"),
             "test_cbz_layer_a_scan"),
            (self.load_grader("tests/test_cbz_a3_float_census.py", "child_a3"),
             "test_cbz_a3_float_census"),
        ):
            with self.subTest(grader=grader.__name__), tempfile.TemporaryDirectory(prefix="cbz-a8-child-") as directory:
                root = Path(directory)
                package = root / "tests"
                package.mkdir()
                (package / "__init__.py").write_text("")
                source = package / (module + ".py")
                source.write_text("import unittest\nclass TestFixture(unittest.TestCase):\n def test_source(self): print('CBZ_CHILD_SOURCE')\n")
                (package / "test_cbz_auditor_hardening.py").write_text(
                    "import unittest\nclass TestHardening(unittest.TestCase):\n def test_source(self): pass\n")
                stat = source.stat()
                suffix = ".opt-" + str(sys.flags.optimize) if sys.flags.optimize else ""
                cache = package / "__pycache__" / (module + "." + sys.implementation.cache_tag + suffix + ".pyc")
                cache.parent.mkdir()
                poisoned = compile("print('CBZ_CHILD_POISON'); raise RuntimeError('poison')\n", str(source), "exec")
                cache.write_bytes(importlib._bootstrap_external._code_to_timestamp_pyc(
                    poisoned, int(stat.st_mtime), stat.st_size))
                original = grader.WORKSPACE
                grader.WORKSPACE = str(root)
                try:
                    result = grader.run_suite(str(root / "irrelevant.py"))
                finally:
                    grader.WORKSPACE = original
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("CBZ_CHILD_SOURCE", result.stdout)
                self.assertNotIn("CBZ_CHILD_POISON", result.stdout + result.stderr)

    def test_grader_children_ignore_inherited_poison_cache_prefix(self):
        from unittest.mock import patch
        for grader, module in (
            (self.load_grader("tests/test_cbz_layer_a_scan_mutants.py", "prefix_lexer"),
             "test_cbz_layer_a_scan"),
            (self.load_grader("tests/test_cbz_a3_float_census.py", "prefix_a3"),
             "test_cbz_a3_float_census"),
        ):
            with self.subTest(grader=grader.__name__), tempfile.TemporaryDirectory(prefix="cbz-a8-prefix-") as directory:
                root = Path(directory) / "workspace"
                package = root / "tests"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text("")
                source = package / (module + ".py")
                source.write_text("import unittest\nclass TestFixture(unittest.TestCase):\n def test_source(self): print('CBZ_CHILD_SOURCE')\n")
                (package / "test_cbz_auditor_hardening.py").write_text(
                    "import unittest\nclass TestHardening(unittest.TestCase):\n def test_source(self): pass\n")
                stat = source.stat()
                inherited_cache = Path(directory) / "inherited-cache"
                with patch.object(sys, "pycache_prefix", str(inherited_cache)):
                    cache = Path(importlib.util.cache_from_source(
                        str(source), optimization=str(sys.flags.optimize) if sys.flags.optimize else ""))
                cache.parent.mkdir(parents=True)
                poisoned = compile("print('CBZ_CHILD_POISON'); raise RuntimeError('poison')\n", str(source), "exec")
                cache.write_bytes(importlib._bootstrap_external._code_to_timestamp_pyc(
                    poisoned, int(stat.st_mtime), stat.st_size))
                original = grader.WORKSPACE
                grader.WORKSPACE = str(root)
                try:
                    with patch.dict(os.environ, {"PYTHONPYCACHEPREFIX": str(inherited_cache)}):
                        result = grader.run_suite(str(root / "irrelevant.py"))
                finally:
                    grader.WORKSPACE = original
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("CBZ_CHILD_SOURCE", result.stdout)
                self.assertNotIn("CBZ_CHILD_POISON", result.stdout + result.stderr)

    @staticmethod
    def load_grader(path, name):
        return load(ROOT / path, name)

    def test_poisoned_cached_bytecode_does_not_execute(self):
        sources = [
            ("lexer", ROOT / "tools/cbz/layer_a_scan.py"),
            ("a3", ROOT / "tools/cbz/a3_float_census.py"),
            ("hardening", ROOT / "tools/cbz/layer_a_scan.py"),
        ]
        for kind, source in sources:
            with self.subTest(loader=kind), tempfile.TemporaryDirectory(prefix="cbz-a8-cache-") as directory:
                path = Path(directory) / "probe.py"
                path.write_bytes(source.read_bytes())
                stat = path.stat()
                suffix = ".opt-" + str(sys.flags.optimize) if sys.flags.optimize else ""
                cache = path.parent / "__pycache__" / ("probe." + sys.implementation.cache_tag + suffix + ".pyc")
                cache.parent.mkdir()
                poisoned = compile("POISON_EXECUTED = True\n", str(path), "exec")
                cache.write_bytes(importlib._bootstrap_external._code_to_timestamp_pyc(
                    poisoned, int(stat.st_mtime), stat.st_size))
                if kind == "lexer":
                    entry = "import tests.test_cbz_layer_a_scan as case; probe=case.M\n"
                elif kind == "a3":
                    entry = "import tests.test_cbz_a3_float_census as case; probe=case.M\n"
                else:
                    entry = "import tests.test_cbz_auditor_hardening as case; probe=case.load(Path(sys.argv[1]), 'target')\n"
                program = ("import sys\nfrom pathlib import Path\n" + entry
                           + "print('CBZ_BYTES_MATCH' if Path(probe.__file__).read_bytes()==Path(sys.argv[1]).read_bytes() else 'CBZ_BYTES_DIFFER')\n"
                           + "print('CBZ_POISONED' if getattr(probe, 'POISON_EXECUTED', False) else 'CBZ_SOURCE_ONLY')\n")
                env = {key: value for key, value in os.environ.items()
                       if key not in ("PYTHONPATH", "PYTHONHOME", "PYTHONPYCACHEPREFIX",
                                      "CBZ_LAYER_A_SCAN", "CBZ_A3_CENSUS")}
                if kind == "lexer":
                    env["CBZ_LAYER_A_SCAN"] = str(path)
                elif kind == "a3":
                    env["CBZ_A3_CENSUS"] = str(path)
                result = subprocess.run(
                    [sys.executable, *(["-O"] if sys.flags.optimize else []), "-B", "-S",
                     "-c", program, str(path)],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("CBZ_BYTES_MATCH", result.stdout)
                self.assertIn("CBZ_SOURCE_ONLY", result.stdout)


if __name__ == "__main__":
    unittest.main()
