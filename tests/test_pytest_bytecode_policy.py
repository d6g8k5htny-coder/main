"""Root pytest policy regressions; only synthetic tests and verify_closure run.

Run directly with Python/unittest, normally and with -O. Installed pytest is
required. No evaluator, real fixture, numerical control, or parent proof runs.
Set ASI_BYTECODE_EVIDENCE_DIR to a NEW directory to retain all child evidence.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = "experiments/board_capacity_probe_20261022/test_timestamp_offsets.py"
PREFIX = "experiments/reactor_v0/"
VALIDATOR_SHA = "2d39c1bd844f49d43aa380443fa384f8e3a0aeb9df6173a3fcf9eeaeb150bef3"
VALIDATOR_BLOB = "fb4d8419b638b5619f8d8c6f685a036b5a6b8a61"
MEMBERSHIP_ERROR = "package file set differs from frozen closure"

# This file replaces, rather than executes, the real Reactor test in each fixture.
SYNTHETIC_TEST = r'''import hashlib, importlib.util, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUTSIDE = HERE.parents[1]

def snapshot():
    result = {}
    for p in sorted(HERE.rglob("*")):
        if p.is_file():
            raw = p.read_bytes()
            result[p.relative_to(HERE).as_posix()] = {
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return result

def test_synthetic_frozen_membership():
    source = HERE / "validator.py"
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "2d39c1bd844f49d43aa380443fa384f8e3a0aeb9df6173a3fcf9eeaeb150bef3":
        raise AssertionError("wrong validator before import")
    observation = {"before_validator_import": snapshot(),
                   "dont_write_bytecode": sys.dont_write_bytecode,
                   "optimize": sys.flags.optimize, "python": sys.version}
    calls = {}
    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_filename == str(source):
            name = frame.f_code.co_name
            calls[name] = calls.get(name, 0) + 1
    try:
        sys.setprofile(profile)
        spec = importlib.util.spec_from_file_location("asi_synthetic_validator", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        observation["after_validator_import"] = snapshot()
        closure = json.loads((HERE / "closure.json").read_text())
        observation["result"] = module.verify_closure(HERE, closure)
        observation["error_type"] = None
        observation["error"] = None
    except Exception as error:
        observation["error_type"] = type(error).__name__
        observation["error"] = str(error)
        raise
    finally:
        sys.setprofile(None)
        observation["after_validation"] = snapshot()
        observation["validator_calls"] = calls
        (OUTSIDE / "observation.json").write_text(json.dumps(observation, sort_keys=True))
    if observation["result"] is not True:
        raise AssertionError("validator did not return True")
'''


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@unittest.skipUnless(importlib.util.find_spec("pytest"), "installed pytest required")
class PytestBytecodePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = (ROOT / "conftest.py").read_bytes()
        cls.validator = (ROOT / PREFIX / "validator.py").read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(cls.validator)).encode() + b"\0" + cls.validator).hexdigest()
        if len(cls.validator) != 10770 or sha(cls.validator) != VALIDATOR_SHA or blob != VALIDATOR_BLOB:
            raise AssertionError("pinned validator identity changed; requalify the fixture")
        syntax = ast.parse(cls.validator)
        assignments = [n for n in syntax.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "REQUIRED_CLOSURE"
                               for t in n.targets)]
        if len(assignments) != 1:
            raise AssertionError("ambiguous REQUIRED_CLOSURE")
        cls.paths = sorted(ast.literal_eval(assignments[0].value))
        cls.package_names = {p.removeprefix(PREFIX) for p in cls.paths if p.startswith(PREFIX)} | {"closure.json"}
        evidence = os.environ.get("ASI_BYTECODE_EVIDENCE_DIR")
        cls.retained = Path(evidence).resolve() if evidence else None
        if cls.retained:
            cls.retained.mkdir(parents=True, exist_ok=False)

    def setUp(self):
        if self.retained:
            self.home = self.retained / self._testMethodName
            self.home.mkdir()
        else:
            temporary = tempfile.TemporaryDirectory(prefix="pytest-bytecode-policy-")
            self.addCleanup(temporary.cleanup)
            self.home = Path(temporary.name)
        self.sequence = 0

    def make_root(self, name="fixture"):
        root = self.home / name
        root.mkdir()
        (root / "conftest.py").write_bytes(self.policy)
        return root

    def frozen_fixture(self):
        root = self.make_root()
        closure = {}
        for relative in self.paths:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative == PREFIX + "validator.py":
                payload = self.validator
            elif relative == PREFIX + "test_reactor_v0.py":
                payload = SYNTHETIC_TEST.encode()
            else:
                payload = ("SYNTHETIC INERT PLACEHOLDER: " + relative + "\n").encode()
            target.write_bytes(payload)
            closure[relative] = sha(payload)
        package = root / PREFIX
        write_json(package / "closure.json", closure)
        return root, package

    def run_pytest(self, root, *, no_bytecode=False, args=()):
        self.sequence += 1
        log = self.home / ("run-%02d" % self.sequence)
        log.mkdir()
        command = [sys.executable, "-I"]
        if sys.flags.optimize:
            command.append("-O")
        if no_bytecode:
            command.append("-B")
        command += ["-m", "pytest", "-q", "-p", "no:cacheprovider", "--color=no", *args]
        environment = {"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                       "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
        before = {p.relative_to(root).as_posix(): sha(p.read_bytes())
                  for p in root.rglob("*") if p.is_file()}
        completed = subprocess.run(command, cwd=root, env=environment, capture_output=True, timeout=15)
        (log / "stdout.bin").write_bytes(completed.stdout)
        (log / "stderr.bin").write_bytes(completed.stderr)
        after = {p.relative_to(root).as_posix(): sha(p.read_bytes())
                 for p in root.rglob("*") if p.is_file()}
        observation_path = root / "observation.json"
        observation = json.loads(observation_path.read_text()) if observation_path.exists() else None
        if observation is not None:
            write_json(log / "observation.json", observation)
        write_json(log / "receipt.json", {
            "command": command, "cwd": str(root), "environment": environment,
            "returncode": completed.returncode, "stdout_sha256": sha(completed.stdout),
            "stderr_sha256": sha(completed.stderr), "before": before, "after": after,
            "policy_sha256": sha((root / "conftest.py").read_bytes()), "observation": observation})
        self.assertEqual(completed.stderr, b"", "unexpected child stderr")
        # Imports may add cache files; they must not edit existing inputs.
        for name, digest in before.items():
            self.assertEqual(after.get(name), digest, "child edited/deleted input: " + name)
        if observation is not None:
            calls = observation["validator_calls"]
            self.assertEqual(calls.get("verify_closure"), 1)
            self.assertTrue(set(calls) <= {"<module>", "verify_closure", "_require", "<setcomp>", "<genexpr>"}, calls)
            self.assertEqual(observation["after_validator_import"], observation["after_validation"])
        return completed, observation

    def assert_pass(self, completed, observation):
        self.assertEqual(completed.returncode, 0, completed.stdout.decode())
        self.assertIn(b"1 passed", completed.stdout)
        self.assertIsNotNone(observation)
        self.assertIs(observation.get("result"), True)
        self.assertIsNone(observation["error"])
        self.assertEqual(set(observation["after_validation"]), self.package_names)

    def assert_refusal(self, completed, observation, message=MEMBERSHIP_ERROR):
        self.assertEqual(completed.returncode, 1, completed.stdout.decode())
        self.assertIn(b"1 failed", completed.stdout)
        self.assertIsNotNone(observation)
        self.assertEqual((observation["error_type"], observation["error"]), ("ValueError", message))

    def test_clean_root_run_prevents_package_bytecode(self):
        root, _ = self.frozen_fixture()
        completed, observed = self.run_pytest(root)
        self.assert_pass(completed, observed)
        self.assertTrue(observed["dont_write_bytecode"])

    def test_explicit_B_baseline_works(self):
        root, _ = self.frozen_fixture()
        self.assert_pass(*self.run_pytest(root, no_bytecode=True))

    def test_dirty_bytecode_is_preserved_and_rejected(self):
        root, package = self.frozen_fixture()
        # Seed real import-created caches with the unamended discovery policy.
        (root / "conftest.py").write_text("collect_ignore = " + repr([EXCLUDED]) + "\n")
        first, observed = self.run_pytest(root)
        self.assert_refusal(first, observed)
        caches = {p.relative_to(package).as_posix(): p.read_bytes()
                  for p in package.rglob("*.pyc")}
        self.assertTrue(caches, "control never produced real bytecode")
        (root / "conftest.py").write_bytes(self.policy)
        # Remove only the previous measurement, outside the preserved package.
        (root / "observation.json").unlink()
        self.assert_refusal(*self.run_pytest(root, no_bytecode=True))
        for name, raw in caches.items():
            self.assertEqual((package / name).read_bytes(), raw)

    def test_empty_cache_directory_is_accepted(self):
        root, package = self.frozen_fixture()
        (package / "__pycache__").mkdir()
        self.assert_pass(*self.run_pytest(root))

    def test_extra_file_is_not_hidden(self):
        root, package = self.frozen_fixture()
        (package / "extra.txt").write_text("retain this unexpected evidence\n")
        self.assert_refusal(*self.run_pytest(root, no_bytecode=True))

    def test_missing_file_is_not_hidden(self):
        root, package = self.frozen_fixture()
        (package / "README.md").unlink()
        self.assert_refusal(*self.run_pytest(root, no_bytecode=True))

    def test_changed_byte_is_not_hidden(self):
        root, package = self.frozen_fixture()
        path = package / "README.md"
        path.write_bytes(path.read_bytes() + b"changed\n")
        completed, observed = self.run_pytest(root, no_bytecode=True)
        self.assert_refusal(completed, observed, "closure hash mismatch: experiments/reactor_v0/README.md")

    def test_exclusion_is_exact(self):
        syntax = ast.parse(self.policy)
        assignments = [n for n in syntax.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "collect_ignore" for t in n.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(ast.literal_eval(assignments[0].value), [EXCLUDED])

    def discovery_fixture(self, failing=False):
        root = self.make_root()
        excluded = root / EXCLUDED
        excluded.parent.mkdir(parents=True)
        excluded.write_text("raise RuntimeError('EXCLUDED_TARGET_WAS_IMPORTED')\n")
        (excluded.parent / "test_neighbour.py").write_text(
            "def test_neighbour():\n    " +
            ("raise AssertionError('NEIGHBOUR_FAILURE_RETAINED')\n" if failing else "return None\n"))
        other = root / "other"
        other.mkdir()
        (other / "test_timestamp_offsets.py").write_text("def test_same_basename():\n    return None\n")
        (root / "test_ordinary.py").write_text("def test_ordinary():\n    return None\n")
        return root

    def test_discovery_keeps_neighbours_and_same_basename(self):
        completed, _ = self.run_pytest(self.discovery_fixture(), args=("--collect-only",))
        self.assertEqual(completed.returncode, 0, completed.stdout.decode())
        self.assertIn(b"3 tests collected", completed.stdout)
        for node in (b"test_neighbour.py::test_neighbour", b"other/test_timestamp_offsets.py::test_same_basename", b"test_ordinary.py::test_ordinary"):
            self.assertIn(node, completed.stdout)
        self.assertNotIn(b"EXCLUDED_TARGET_WAS_IMPORTED", completed.stdout)

    def test_neighbour_failures_are_not_suppressed(self):
        completed, _ = self.run_pytest(self.discovery_fixture(failing=True))
        self.assertEqual(completed.returncode, 1, completed.stdout.decode())
        self.assertIn(b"1 failed, 2 passed", completed.stdout)
        self.assertIn(b"NEIGHBOUR_FAILURE_RETAINED", completed.stdout)
        self.assertNotIn(b"EXCLUDED_TARGET_WAS_IMPORTED", completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
