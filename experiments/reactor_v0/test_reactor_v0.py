import copy
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


class ReactorFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evaluator import evaluate_package
        from validator import load_and_validate

        cls.evaluate_package = staticmethod(evaluate_package)
        cls.load_and_validate = staticmethod(load_and_validate)
        cls.package = load_and_validate(HERE)

    def test_exposed_control_is_exact_five_sixteenths(self):
        result = self.evaluate_package(self.package)
        self.assertEqual(result["expected_short_bar_count"], Fraction(5, 16))
        self.assertEqual(result["finite_bars"], ((1, 0), (8, 0)))
        self.assertEqual(result["essential_birth"], 27)
        self.assertEqual(result["included_lifetimes"], (1, 8))

    def test_positive_integer_amplitude_scaling_scales_barcode_endpoints(self):
        from exact_helpers import exact_periodic_h0

        fixture = self.package["fixture"]
        barcode = exact_periodic_h0(fixture["values"], fixture["grid_side"])
        for scale in (2, 3, 5, 7, 11):
            with self.subTest(scale=scale):
                scaled = exact_periodic_h0(
                    [scale * value for value in fixture["values"]],
                    fixture["grid_side"],
                )
                self.assertEqual(
                    scaled["intervals"],
                    [[scale * birth, scale * death]
                     for birth, death in barcode["intervals"]],
                )
                self.assertEqual(
                    scaled["essential"],
                    [scale * birth for birth in barcode["essential"]],
                )

    def test_tampered_expected_control_is_rejected(self):
        package = copy.deepcopy(self.package)
        package["control"]["expected_control"] = {"numerator": 1, "denominator": 2}
        with self.assertRaisesRegex(ValueError, "declared expected control"):
            self.evaluate_package(package)

    def test_threshold_is_inclusive_and_essential_class_is_excluded(self):
        from evaluator import short_bar_lifetimes

        barcode = {"intervals": [[3, 1], [8, 0]], "essential": [27]}
        self.assertEqual(short_bar_lifetimes(barcode, Fraction(2)), (2,))

    def test_unsupported_exact_root_is_refused(self):
        from exact_helpers import rational_power_exact

        with self.assertRaisesRegex(ValueError, "rational root"):
            rational_power_exact(Fraction(1, 2), Fraction(2, 3))

    def test_schema_rejects_missing_extra_wrong_type_and_status_effect(self):
        from validator import validate_document

        good = json.loads((HERE / "control.json").read_text(encoding="utf-8"))
        mutations = []
        missing = copy.deepcopy(good)
        del missing["scientific_status_effect"]
        mutations.append(missing)
        extra = copy.deepcopy(good)
        extra["unexpected"] = False
        mutations.append(extra)
        wrong_type = copy.deepcopy(good)
        wrong_type["threshold"] = 1.0
        mutations.append(wrong_type)
        wrong_status = copy.deepcopy(good)
        wrong_status["scientific_status_effect"] = "ACCEPTED"
        mutations.append(wrong_status)
        for document in mutations:
            with self.subTest(document=document), self.assertRaises(ValueError):
                validate_document(document)

    def test_source_manifest_separates_raw_hash_from_git_membership(self):
        sources = self.package["sources"]
        self.assertEqual(len(sources), 2)
        for source in sources:
            self.assertEqual(source["repository"], "d6g8k5htny-coder/main")
            self.assertEqual(len(source["commit"]), 40)
            self.assertEqual(len(source["raw_sha256"]), 64)
            self.assertEqual(len(source["git_blob_sha"]), 40)
            self.assertFalse(source["theorem_conclusion_consumed"])

    def test_consumed_closure_hashes_match_exact_bytes(self):
        from validator import verify_closure

        self.assertTrue(verify_closure(HERE, self.package["closure"]))

    def test_optimized_child_process_runs_validator_and_evaluator(self):
        completed = subprocess.run(
            [sys.executable, "-O", "-B", str(HERE / "run_control.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["expected_short_bar_count"], "5/16")
        self.assertTrue(payload["validated"])


class ValidatorNegativeTests(unittest.TestCase):
    def setUp(self):
        self.control = json.loads((HERE / "control.json").read_text(encoding="utf-8"))
        self.fixture = json.loads((HERE / "fixture.json").read_text(encoding="utf-8"))
        self.sources = json.loads(
            (HERE / "source_manifest.json").read_text(encoding="utf-8")
        )

    def test_control_rejects_wrong_constants_types_and_local_names(self):
        from validator import validate_document

        mutations = []
        for value in ("PROMOTED", "none", None, ["NONE"]):
            document = copy.deepcopy(self.control)
            document["scientific_status_effect"] = value
            mutations.append(document)
        for key, value in (
                ("schema_version", True),
                ("scope", "OTHER"),
                ("campaign_status", "RAN")):
            document = copy.deepcopy(self.control)
            document[key] = value
            mutations.append(document)
        for value in ("a/b", ".", ".."):
            document = copy.deepcopy(self.control)
            document["fixture"] = value
            mutations.append(document)
        document = copy.deepcopy(self.control)
        document["alpha"]["numerator"] = True
        mutations.append(document)
        document = copy.deepcopy(self.control)
        document["alpha"]["denominator"] = 0
        mutations.append(document)
        for document in mutations:
            with self.subTest(document=document), self.assertRaises(ValueError):
                validate_document(document)

    def test_control_rejects_out_of_domain_rationals(self):
        from validator import validate_document

        cases = (
            ("alpha", {"numerator": 0, "denominator": 1}),
            ("alpha", {"numerator": -1, "denominator": 1}),
            ("amplitude_scale", {"numerator": 0, "denominator": 1}),
            ("amplitude_scale", {"numerator": -2, "denominator": 1}),
            ("threshold", {"numerator": -1, "denominator": 4}),
            ("expected_control", {"numerator": -1, "denominator": 16}),
        )
        for key, value in cases:
            document = copy.deepcopy(self.control)
            document[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_document(document)

    def test_fixture_rejects_each_wrong_contract_field(self):
        from validator import _validate_fixture

        mutations = []
        for key, value in (
                ("schema_version", True),
                ("grid_side", 5),
                ("values", [0] * 15),
                ("expected_barcode", {"intervals": [[1, 0]], "essential": [27]}),
                ("exclusions", [])):
            document = copy.deepcopy(self.fixture)
            document[key] = value
            mutations.append(document)
        for document in mutations:
            with self.subTest(document=document), self.assertRaises(ValueError):
                _validate_fixture(document)

    def test_source_manifest_rejects_malformed_or_rebound_identities(self):
        from validator import _validate_sources

        mutations = []
        replacements = (
            ("repository", "other/repository"),
            ("commit", "z" * 40),
            ("raw_sha256", "z" * 64),
            ("git_blob_sha", "z" * 40),
            ("theorem_conclusion_consumed", True),
            ("theorem_conclusion_consumed", 0),
            ("theorem_conclusion_consumed", "false"),
            ("source_status", "MERGED"),
        )
        for key, value in replacements:
            document = copy.deepcopy(self.sources)
            document["sources"][0][key] = value
            mutations.append(document)
        swapped = copy.deepcopy(self.sources)
        swapped["sources"][0]["pull_request"] = 334
        swapped["sources"][1]["pull_request"] = 330
        mutations.append(swapped)
        duplicate = copy.deepcopy(self.sources)
        duplicate["sources"][1] = copy.deepcopy(duplicate["sources"][0])
        mutations.append(duplicate)
        for value in ([], {}):
            unhashable = copy.deepcopy(self.sources)
            unhashable["sources"][0]["pull_request"] = value
            mutations.append(unhashable)
        widened = copy.deepcopy(self.sources)
        widened["sources"].append(copy.deepcopy(widened["sources"][0]))
        mutations.append(widened)
        for document in mutations:
            with self.subTest(document=document), self.assertRaises(ValueError):
                _validate_sources(document)

    def test_duplicate_json_key_is_rejected(self):
        from validator import _load_json

        with tempfile.TemporaryDirectory(prefix="reactor-duplicate-") as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                _load_json(path)

    def test_source_object_verifier_checks_bytes_length_hash_and_git_blob(self):
        from validator import verify_source_objects

        with tempfile.TemporaryDirectory(prefix="reactor-source-git-") as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Reactor Test"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "reactor@example.invalid"],
                cwd=root,
                check=True,
            )
            source_path = root / "source.txt"
            raw = b"source-bound bytes\n"
            source_path.write_bytes(raw)
            subprocess.run(["git", "add", "source.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "source fixture"], cwd=root, check=True)
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            blob = subprocess.check_output(
                ["git", "rev-parse", "HEAD:source.txt"], cwd=root, text=True
            ).strip()
            record = {
                "commit": commit,
                "path": "source.txt",
                "byte_length": len(raw),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob_sha": blob,
            }
            self.assertTrue(verify_source_objects(root, [record]))
            for key, value in (
                    ("byte_length", len(raw) + 1),
                    ("raw_sha256", "0" * 64),
                    ("git_blob_sha", "0" * 40)):
                changed = copy.deepcopy(record)
                changed[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    verify_source_objects(root, [changed])


class ClosureAndWorkflowTests(unittest.TestCase):
    def _copy_frozen_tree(self, root):
        from validator import REQUIRED_CLOSURE

        for relative in REQUIRED_CLOSURE:
            source = ROOT / relative
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        closure = root / "experiments/reactor_v0/closure.json"
        closure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / "closure.json", closure)
        return closure.parent

    def _workflow_guard(self):
        workflow = (ROOT / ".github/workflows/reactor-v0-controls.yml").read_text(
            encoding="utf-8"
        )
        lines = workflow.splitlines()
        start = next(
            index for index, line in enumerate(lines)
            if line.strip().startswith("python3 ") and "<<'PY'" in line
        )
        end = next(
            index for index in range(start + 1, len(lines))
            if lines[index].strip() == "PY"
        )
        command = shlex.split(lines[start].strip().split("<<", 1)[0])
        command[0] = sys.executable
        script = textwrap.dedent("\n".join(lines[start + 1:end])) + "\n"
        return workflow, command, script

    def _run_workflow_guard(self, root, base):
        _, command, script = self._workflow_guard()
        environment = os.environ.copy()
        environment["BASE_SHA"] = base
        return subprocess.run(
            command,
            cwd=root,
            env=environment,
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_tampered_closure_byte_is_rejected_directly_and_by_loader(self):
        from validator import load_and_validate, verify_closure

        with tempfile.TemporaryDirectory(prefix="reactor-closure-") as temporary:
            root = Path(temporary)
            package = self._copy_frozen_tree(root)
            readme = package / "README.md"
            readme.write_bytes(readme.read_bytes() + b"tampered\n")
            closure = json.loads((package / "closure.json").read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "closure hash mismatch"):
                verify_closure(package, closure)
            with self.assertRaisesRegex(ValueError, "closure hash mismatch"):
                load_and_validate(package)

    def test_extra_package_file_is_rejected(self):
        from validator import verify_closure

        with tempfile.TemporaryDirectory(prefix="reactor-extra-") as temporary:
            root = Path(temporary)
            package = self._copy_frozen_tree(root)
            (package / "fractions.py").write_text("raise RuntimeError('shadowed')\n")
            closure = json.loads((package / "closure.json").read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "package file set"):
                verify_closure(package, closure)

    def test_package_symlinks_are_rejected(self):
        from validator import load_and_validate, verify_closure

        with tempfile.TemporaryDirectory(prefix="reactor-symlink-") as temporary:
            root = Path(temporary)
            outside = root / "outside"
            outside.mkdir()
            package = self._copy_frozen_tree(root / "copy")
            closure = json.loads((package / "closure.json").read_text(encoding="utf-8"))

            (package / "extra-directory").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                verify_closure(package, closure)
            (package / "extra-directory").unlink()

            package_link = root / "package-link"
            package_link.symlink_to(package, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                load_and_validate(package_link)

            readme = package / "README.md"
            outside_readme = outside / "README.md"
            outside_readme.write_bytes(readme.read_bytes())
            readme.unlink()
            readme.symlink_to(outside_readme)
            with self.assertRaisesRegex(ValueError, "symlink"):
                verify_closure(package, closure)

    def test_workflow_hardens_actions_sources_and_freeze_boundary(self):
        workflow = (ROOT / ".github/workflows/reactor-v0-controls.yml").read_text(
            encoding="utf-8"
        )
        required = (
            "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
            "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
            "persist-credentials: false",
            "timeout-minutes:",
            "REACTOR_VERIFY_SOURCES: '1'",
            "git fetch --no-tags --depth=1 origin",
            "git status --porcelain --untracked-files=all --ignored",
            "github.event.pull_request.base.sha",
            "is_symlink()",
            "closure.json",
        )
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, workflow)
        self.assertLess(
            workflow.index("is_symlink()"),
            workflow.index("Base has no Reactor A freeze"),
        )

    def test_workflow_guard_and_unittests_use_isolated_python(self):
        workflow, command, _ = self._workflow_guard()
        with tempfile.TemporaryDirectory(prefix="reactor-root-shadow-") as temporary:
            root = Path(temporary)
            package = self._copy_frozen_tree(root)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.name", "Reactor Test"], cwd=root, check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "reactor@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "frozen base"], cwd=root, check=True)
            base = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()

            evaluator = package / "evaluator.py"
            evaluator.write_bytes(evaluator.read_bytes() + b"# repinned edit\n")
            closure_file = package / "closure.json"
            closure = json.loads(closure_file.read_text(encoding="utf-8"))
            closure["experiments/reactor_v0/evaluator.py"] = hashlib.sha256(
                evaluator.read_bytes()
            ).hexdigest()
            closure_file.write_text(
                json.dumps(closure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (root / "subprocess.py").write_text(
                "PIPE = -1\n"
                "DEVNULL = -3\n"
                "class Result:\n"
                "    returncode = 128\n"
                "    stdout = b''\n"
                "def run(*args, **kwargs):\n"
                "    return Result()\n",
                encoding="utf-8",
            )
            completed = self._run_workflow_guard(root, base)
            self.assertNotEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
        self.assertIn("-I", command)
        self.assertIn(
            "python3 -I -B -m unittest discover -s experiments/reactor_v0",
            workflow,
        )
        self.assertIn(
            "python3 -I -B -O -m unittest discover -s experiments/reactor_v0",
            workflow,
        )

    def test_workflow_guard_fails_closed_for_invalid_base(self):
        workflow, _, _ = self._workflow_guard()
        with tempfile.TemporaryDirectory(prefix="reactor-invalid-base-") as temporary:
            root = Path(temporary)
            self._copy_frozen_tree(root)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            completed = self._run_workflow_guard(root, "0" * 40)
            self.assertNotEqual(
                completed.returncode,
                0,
                completed.stdout + completed.stderr,
            )
        self.assertIn('["git", "cat-file", "-e", base + "^{commit}"]', workflow)
        self.assertIn('["git", "ls-tree", "-z", base, "--", closure_path]', workflow)


if __name__ == "__main__":
    unittest.main()
