"""Real Git/subprocess contracts for the bounded prepared Node runner."""
import hashlib
import importlib.util
import io
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("CBZ_NODE_RUNNER_SOURCE", ROOT / "tools/cbz_node_regression.py"))


def runner():
    if not SOURCE.is_file():
        raise AssertionError("prepared Node runner is missing")
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("cbz_node_regression_tested", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NodeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.module = runner()
        self.temporary = tempfile.TemporaryDirectory(prefix="cbz-node-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.repo = self.directory / "main"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "CBZ fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        (self.repo / "data.txt").write_bytes(b"exact input\n")
        self.commit()
        self.head = self.git("rev-parse", "HEAD").strip()
        self.roots = {"d6g8k5htny-coder/main": self.repo}

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args],
            stderr=subprocess.STDOUT, text=True)

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")

    def descriptor(self, raw=b"exact input\n"):
        return dict(repository="d6g8k5htny-coder/main", commit=self.head,
                    path="data.txt", bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    blob=hashlib.sha1(b"blob " + str(len(raw)).encode()
                                      + b"\0" + raw).hexdigest())

    def test_ambient_git_redirect_cannot_select_another_object_database(self):
        before = os.environ.get("GIT_DIR")
        os.environ["GIT_DIR"] = str(self.directory / "absent.git")
        try:
            try:
                raw = self.module.read_pinned(self.roots, self.descriptor())
            except ValueError as error:
                self.fail("ambient Git redirection affected pinned read: " + str(error))
            self.assertEqual(raw, b"exact input\n")
        finally:
            if before is None:
                os.environ.pop("GIT_DIR", None)
            else:
                os.environ["GIT_DIR"] = before

    def test_loaded_sources_must_belong_to_selected_main(self):
        self.assertTrue(hasattr(self.module, "assert_loaded_sources"),
                        "loaded helper identity guard is missing")
        with self.assertRaisesRegex(ValueError, "loaded source"):
            self.module.assert_loaded_sources(self.repo)

    def test_loaded_source_and_imported_helper_cannot_drift(self):
        self.assertTrue(hasattr(self.module, "assert_loaded_sources"),
                        "loaded helper identity guard is missing")
        tools = self.repo / "tools"
        tools.mkdir()
        (tools / "cbz_node_regression.py").write_bytes(SOURCE.read_bytes())
        (tools / "museum_data.py").write_bytes((ROOT / "tools/museum_data.py").read_bytes())
        for name in ("cbz_node_regression.py", "museum_data.py"):
            script = (
                "import sys\nfrom pathlib import Path\n"
                "sys.path.insert(0, sys.argv[1])\n"
                "import cbz_node_regression as recorder\n"
                "root=Path(sys.argv[1]).parent\n"
                "recorder.assert_loaded_sources(root)\n"
                "print('CONTROL_IMPORT_PASS')\n"
                "path=Path(sys.argv[1])/sys.argv[2]\n"
                "path.write_bytes(path.read_bytes()+b'\\n# changed after import\\n')\n"
                "try:\n recorder.assert_loaded_sources(root)\n"
                "except ValueError:\n print('EXPECTED_SOURCE_REJECTION')\n"
                "else:\n raise AssertionError('changed loaded source was accepted')\n")
            result = subprocess.run(
                [sys.executable, *(["-O"] if sys.flags.optimize else []), "-B", "-S",
                 "-c", script, str(tools), name],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("CONTROL_IMPORT_PASS", result.stdout)
            self.assertIn("EXPECTED_SOURCE_REJECTION", result.stdout)

    def test_tracked_source_symlink_cannot_bind_an_external_dependency(self):
        external = self.directory / "outside.txt"
        external.write_text("unbound external source\n")
        (self.repo / "dependency").symlink_to(external)
        self.commit()
        with self.assertRaisesRegex(ValueError, "unsupported tracked Git entry"):
            self.module.source_snapshot(self.repo)

    def test_selected_node_test_symlink_is_rejected(self):
        (self.repo / "tests").mkdir()
        external = self.directory / "outside.mjs"
        external.write_text("/* external test code */")
        (self.repo / "tests/external.mjs").symlink_to(external)
        self.commit()
        with self.assertRaisesRegex(ValueError, "regular.*Node"):
            self.module.node_command(self.repo, "node")

    def make_fixtures(self):
        output = self.directory / "finite-fixtures"
        paths = {"shop/Math-/data.txt": b"original\n",
                 "museum/verified-inputs.json": b"{}\n"}
        expected = {}
        for relative, raw in paths.items():
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            expected[relative] = dict(bytes=len(raw),
                                      sha256=hashlib.sha256(raw).hexdigest())
        return output, expected

    def test_child_fixture_mutation_is_a_failed_postflight(self):
        self.assertTrue(hasattr(self.module, "verify_fixture_tree"),
                        "fixture postflight guard is missing")
        self.assertIsNotNone(shutil.which("node"), "Node is required for this suite")
        output, expected = self.make_fixtures()
        self.module.verify_fixture_tree(output, expected)
        (self.repo / "tests").mkdir()
        (self.repo / "tests/mutate.mjs").write_text(
            "import test from 'node:test'; import fs from 'node:fs';"
            "test('mutation',()=>fs.writeFileSync("
            "process.env.SHOP_MATH_FIXTURE+'/data.txt','mutated\\n'));")
        self.commit()
        environment = dict(os.environ, SHOP_MATH_FIXTURE=str(output / "shop/Math-"))
        path = self.directory / "mutation.tap"
        with path.open("wb") as log:
            with self.assertRaisesRegex(ValueError, "fixture") as failure:
                self.module.execute_node(
                    self.repo, self.module.node_command(self.repo, "node"),
                    environment, log, output, expected)
        self.assertEqual(failure.exception.exit_code, 0)
        self.assertEqual(self.module.tap_summary(path.read_text())["pass"], 1)
        self.assertEqual((output / "shop/Math-/data.txt").read_bytes(), b"mutated\n")

    def test_fixture_file_directory_and_type_set_is_exact(self):
        self.assertTrue(hasattr(self.module, "verify_fixture_tree"),
                        "fixture postflight guard is missing")
        output, expected = self.make_fixtures()
        self.module.verify_fixture_tree(output, expected)
        path = output / "shop/Math-/data.txt"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "fixture"):
            self.module.verify_fixture_tree(output, expected)
        path.write_bytes(b"original\n")
        extra = output / "shop/extra.txt"
        extra.write_text("new input")
        with self.assertRaisesRegex(ValueError, "fixture"):
            self.module.verify_fixture_tree(output, expected)
        extra.unlink()
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, "fixture"):
            self.module.verify_fixture_tree(output, expected)
        extra.rmdir()
        path.unlink()
        external = self.directory / "outside-fixture"
        external.write_bytes(b"original\n")
        path.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "fixture"):
            self.module.verify_fixture_tree(output, expected)

    def test_reads_immutable_blob_instead_of_dirty_working_copy(self):
        (self.repo / "data.txt").write_bytes(b"changed")
        self.assertEqual(self.module.read_pinned(self.roots, self.descriptor()),
                         b"exact input\n")

    def test_missing_historical_object_does_not_fall_back_to_worktree(self):
        descriptor = self.descriptor()
        descriptor["commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "Git command failed"):
            self.module.read_pinned(self.roots, descriptor)

    def test_all_byte_identities_are_required(self):
        for field, wrong in (("bytes", 7), ("sha256", "0" * 64),
                             ("blob", "0" * 40)):
            with self.subTest(field=field):
                descriptor = self.descriptor()
                descriptor[field] = wrong
                with self.assertRaisesRegex(ValueError, "source identity mismatch"):
                    self.module.read_pinned(self.roots, descriptor)

    def test_invalid_descriptor_domains_refuse(self):
        for field, wrong in (
                ("repository", "attacker/main"), ("commit", "main"),
                ("path", "../data.txt"), ("path", "/data.txt"),
                ("path", "a//b"), ("path", "a/./b"), ("path", "a\\b"),
                ("bytes", True), ("bytes", 0), ("sha256", 9),
                ("blob", True)):
            with self.subTest(field=field, wrong=wrong):
                descriptor = self.descriptor()
                descriptor[field] = wrong
                with self.assertRaises(ValueError):
                    self.module.read_pinned(self.roots, descriptor)

    def test_url_alias_is_rejected(self):
        descriptor = self.descriptor()
        descriptor["url"] = "https://example.invalid/latest/data.txt"
        with self.assertRaisesRegex(ValueError, "URL/identity"):
            self.module.read_pinned(self.roots, descriptor)

    def test_symlink_is_not_admitted_as_a_regular_fixture(self):
        (self.repo / "link").symlink_to("data.txt")
        self.commit()
        descriptor = self.descriptor(b"data.txt")
        descriptor.update(path="link", commit=self.git("rev-parse", "HEAD").strip())
        with self.assertRaisesRegex(ValueError, "not a regular Git blob"):
            self.module.read_pinned(self.roots, descriptor)

    def test_materialization_checks_every_input_before_writing(self):
        good = self.descriptor()
        bad = dict(good, sha256="0" * 64)
        output = self.directory / "fixtures"
        with self.assertRaisesRegex(ValueError, "source identity mismatch"):
            self.module.materialize([good, bad], self.roots, output)
        self.assertFalse(output.exists())

    def test_materialization_preserves_exact_bytes(self):
        output = self.directory / "fixtures"
        verified = self.module.materialize([self.descriptor()], self.roots, output)
        self.assertEqual(len(verified), 1)
        self.assertEqual((output / "main/data.txt").read_bytes(), b"exact input\n")

    def test_duplicate_config_keys_refuse(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            json.loads('{"pin": 1, "pin": 2}',
                       object_pairs_hook=self.module.no_duplicates)

    def test_snapshot_detects_assume_unchanged_source_changes(self):
        self.git("update-index", "--assume-unchanged", "data.txt")
        (self.repo / "data.txt").write_bytes(b"hidden mutation\n")
        self.assertEqual(self.git("status", "--porcelain"), "")
        with self.assertRaisesRegex(ValueError, "tracked bytes differ"):
            self.module.source_snapshot(self.repo)

    def test_snapshot_detects_index_and_untracked_changes(self):
        (self.repo / "untracked").write_text("unexpected source")
        with self.assertRaisesRegex(ValueError, "not clean"):
            self.module.source_snapshot(self.repo)
        (self.repo / "untracked").unlink()
        (self.repo / "data.txt").write_text("changed")
        self.git("add", "data.txt")
        with self.assertRaisesRegex(ValueError, "not clean"):
            self.module.source_snapshot(self.repo)

    def test_snapshot_detects_hidden_executable_mode_changes(self):
        self.git("config", "core.fileMode", "false")
        (self.repo / "data.txt").chmod(0o755)
        with self.assertRaisesRegex(ValueError, "executable mode changed"):
            self.module.source_snapshot(self.repo)

    def test_clean_snapshot_binds_head(self):
        self.assertEqual(self.module.source_snapshot(self.repo), self.head)

    def test_node_command_enumerates_every_tracked_test(self):
        (self.repo / "tests/nested").mkdir(parents=True)
        for path in ("tests/a.mjs", "tests/z.mjs", "tests/nested/b.mjs"):
            (self.repo / path).write_text("/* fixture */")
        self.commit()
        command = self.module.node_command(self.repo, "node")
        self.assertEqual(command[:3], ["node", "--test", "--test-reporter=tap"])
        self.assertEqual(command[3:],
                         ["tests/a.mjs", "tests/nested/b.mjs", "tests/z.mjs"])

    def test_zero_tracked_node_tests_refuses(self):
        with self.assertRaisesRegex(ValueError, "no tracked Node tests"):
            self.module.node_command(self.repo, "node")

    def test_tap_requires_nonzero_complete_all_passed_summary(self):
        good = "# tests 3\n# pass 3\n# fail 0\n# cancelled 0\n# skipped 0\n# todo 0\n"
        self.assertEqual(self.module.tap_summary(good)["tests"], 3)
        cases = [good.replace("# tests 3", "# tests 0").replace("# pass 3", "# pass 0"),
                 good.replace("# pass 3", "# pass 2")]
        for field in ("fail", "cancelled", "skipped", "todo"):
            cases.append(good.replace("# " + field + " 0", "# " + field + " 1"))
        for value in cases:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "failed.*skipped/todo"):
                    self.module.tap_summary(value)

    def test_tap_missing_or_duplicate_fields_refuse(self):
        good = "# tests 1\n# pass 1\n# fail 0\n# cancelled 0\n# skipped 0\n# todo 0\n"
        for field in ("tests", "pass", "fail", "cancelled", "skipped", "todo"):
            line = next(line for line in good.splitlines()
                        if line.startswith("# " + field + " "))
            for text in (good.replace(line + "\n", ""), good + line + "\n"):
                with self.subTest(field=field, text=text):
                    with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
                        self.module.tap_summary(text)

    def test_subprocess_exit_and_output_are_preserved(self):
        log_path = self.directory / "process.log"
        with log_path.open("wb") as log:
            code = self.module.execute_node(
                self.repo, [sys.executable, "-c",
                            "print('real stdout'); raise SystemExit(7)"],
                dict(os.environ), log, *self.make_fixtures())
        self.assertEqual(code, 7)
        self.assertEqual(log_path.read_text(), "real stdout\n")

    def test_missing_executable_is_an_execution_error(self):
        with (self.directory / "missing.log").open("wb") as log:
            with self.assertRaises(FileNotFoundError):
                self.module.execute_node(
                    self.repo, [str(self.directory / "absent")],
                    dict(os.environ), log, *self.make_fixtures())

    def test_real_node_skips_are_not_a_green_suite(self):
        self.assertIsNotNone(shutil.which("node"), "Node is required for this suite")
        (self.repo / "tests").mkdir()
        (self.repo / "tests/skip.mjs").write_text(
            "import test from 'node:test'; test('retained skip', {skip:true}, ()=>{});")
        self.commit()
        path = self.directory / "node.tap"
        with path.open("wb") as log:
            code = self.module.execute_node(
                self.repo, self.module.node_command(self.repo, "node"),
                dict(os.environ), log, *self.make_fixtures())
        self.assertEqual(code, 0)
        with self.assertRaisesRegex(ValueError, "skipped/todo"):
            self.module.tap_summary(path.read_text())

    def test_report_is_atomic_and_keeps_running_distinct_from_pass(self):
        path = self.directory / "result.json"
        self.module.write_report(path, {"status": "RUNNING"})
        self.assertEqual(json.loads(path.read_text()), {"status": "RUNNING"})
        self.assertFalse(path.with_suffix(".tmp").exists())

    def test_output_in_any_source_root_refuses(self):
        for root in (self.repo, self.directory / "Math-", self.directory / "query-"):
            root.mkdir(exist_ok=True)
            with self.subTest(root=root):
                with self.assertRaisesRegex(ValueError, "outside every source root"):
                    self.module.run(
                        self.repo, self.directory / "Math-",
                        self.directory / "query-", root / "evidence")

    def test_preflight_failure_produces_fail_report(self):
        result = self.module.run(
            self.repo, self.directory / "Math-", self.directory / "query-",
            self.directory / "report")
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["mathematical_certification"])
        self.assertFalse(result["final_cbz_certification"])
        self.assertEqual(json.loads(
            (self.directory / "report/result.json").read_text())["status"], "FAIL")

    def test_existing_output_is_never_overwritten(self):
        output = self.directory / "report"
        output.mkdir()
        (output / "retained.txt").write_text("old failure")
        with self.assertRaises(FileExistsError):
            self.module.run(
                self.repo, self.directory / "Math-", self.directory / "query-", output)
        self.assertEqual((output / "retained.txt").read_text(), "old failure")


def assertion_kill(process, expected):
    text = process.stdout + process.stderr
    summary = re.search(r"^FAILED \(([^)]*)\)$", text, re.MULTILINE)
    if re.search(r"(?:ImportError|ModuleNotFoundError|SyntaxError|IndentationError|FileNotFoundError):", text):
        return False
    if process.returncode == 0 or not summary or "errors=" in summary.group(1):
        return False
    if not re.search(r"^Ran [1-9][0-9]* tests?", text, re.MULTILINE):
        return False
    failures = re.findall(r"^FAIL: (test_[a-zA-Z0-9_]+) \(", text, re.MULTILINE)
    return expected in failures


class MutationHarnessTests(unittest.TestCase):
    def result(self, code=1, expected="test_expected", summary="failures=1"):
        return subprocess.CompletedProcess(
            [], code, "",
            "FAIL: " + expected + " (fixture.Contract)\n"
            "AssertionError: intended mutation\nRan 1 test in 0.001s\n"
            "FAILED (" + summary + ")\n")

    def test_intended_assertion_counts(self):
        self.assertTrue(assertion_kill(self.result(), "test_expected"))

    def test_success_exit_cannot_kill(self):
        self.assertFalse(assertion_kill(self.result(code=0), "test_expected"))

    def test_errors_cannot_kill(self):
        self.assertFalse(assertion_kill(
            self.result(summary="failures=1, errors=1"), "test_expected"))

    def test_foreign_assertion_cannot_kill(self):
        self.assertFalse(assertion_kill(self.result(expected="test_foreign"),
                                        "test_expected"))

    def test_no_tests_cannot_kill(self):
        result = self.result()
        result.stderr = result.stderr.replace("Ran 1 test", "Ran 0 tests")
        self.assertFalse(assertion_kill(result, "test_expected"))

    def test_import_trace_inside_assertion_cannot_kill(self):
        result = self.result()
        result.stderr += "\nImportError: child module could not load\n"
        self.assertFalse(assertion_kill(result, "test_expected"))

    def test_launch_failure_cannot_kill(self):
        result = subprocess.CompletedProcess([], 1, "", "SyntaxError: broken module")
        self.assertFalse(assertion_kill(result, "test_expected"))


class SourceMutationTests(unittest.TestCase):
    MUTATIONS = [
        ("loaded-root", "set(_LOADED_SOURCES) != expected", "False",
         "test_loaded_sources_must_belong_to_selected_main"),
        ("loaded-bytes", "path.read_bytes() != raw", "False",
         "test_loaded_source_and_imported_helper_cannot_drift"),
        ("selected-symlink", 'mode not in (b"100644", b"100755") or stage != b"0"',
         "False", "test_selected_node_test_symlink_is_rejected"),
        ("external-source-symlink",
         '        else:\n            raise ValueError("main contains an unsupported tracked Git entry")',
         '        elif mode == b"120000" and kind == b"blob":\n            raw = os.fsencode(os.readlink(path))\n        else:\n            raise ValueError("main contains an unsupported tracked Git entry")',
         "test_tracked_source_symlink_cannot_bind_an_external_dependency"),
        ("fixture-domain", "if actual_files != expected or actual_dirs != expected_dirs:",
         "if False:", "test_child_fixture_mutation_is_a_failed_postflight"),
        ("fixture-postflight",
         "    try:\n        verify_fixture_tree(fixture_root, expected_fixtures)\n    except (OSError, ValueError) as error:\n        raise FixturePostflightError(error, process.returncode) from error\n    return process.returncode",
         "    return process.returncode",
         "test_child_fixture_mutation_is_a_failed_postflight"),

        ("ambient-git", 'if not key.startswith("GIT_")', "if True",
         "test_ambient_git_redirect_cannot_select_another_object_database"),
        ("length", "len(raw) != length or ", "",
         "test_all_byte_identities_are_required"),
        ("sha256", "hashlib.sha256(raw).hexdigest() != digest", "False",
         "test_all_byte_identities_are_required"),
        ("declared-blob", "(blob is not None and actual_blob != blob)", "False",
         "test_all_byte_identities_are_required"),
        ("regular-blob", 'mode not in (b"100644", b"100755") or kind != b"blob"', 'False or kind != b"blob"',
         "test_symlink_is_not_admitted_as_a_regular_fixture"),
        ("url", 'descriptor["url"] != expected_url', "False",
         "test_url_alias_is_rejected"),
        ("skip-inputs", "for descriptor in descriptors]", "for descriptor in descriptors[:1]]",
         "test_materialization_checks_every_input_before_writing"),
        ("hidden-bytes", 'if actual.encode("ascii") != oid:', "if False:",
         "test_snapshot_detects_assume_unchanged_source_changes"),
        ("hidden-mode", 'if executable != (mode == b"100755"):', "if False:",
         "test_snapshot_detects_hidden_executable_mode_changes"),
        ("zero-tests", 'values["tests"] > 0', 'values["tests"] >= 0',
         "test_tap_requires_nonzero_complete_all_passed_summary"),
        ("skip-summary",
         'values["pass"] == values["tests"]\n            and all(values[name] == 0 for name in _SUMMARY_FIELDS[2:])',
         'values["pass"] <= values["tests"]\n            and all(values[name] == 0 for name in _SUMMARY_FIELDS[2:4])',
         "test_real_node_skips_are_not_a_green_suite"),
        ("duplicate-summary", "if len(matches) != 1:", "if not matches:",
         "test_tap_missing_or_duplicate_fields_refuse"),
        ("exit", "return process.returncode", "return 0",
         "test_subprocess_exit_and_output_are_preserved"),
        ("empty-discovery", "if not paths:", "if False:",
         "test_zero_tracked_node_tests_refuses"),
        ("false-green-report", 'record["status"] = "FAIL"', 'record["status"] = "PASS"',
         "test_preflight_failure_produces_fail_report"),
    ]

    def invoke(self, path, name):
        environment = dict(os.environ, CBZ_NODE_RUNNER_SOURCE=str(path),
                           PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run(
            [sys.executable, *(["-O"] if sys.flags.optimize else []), "-B", "-S",
             "-m", "unittest", "-v",
             "tests.test_cbz_node_regression.NodeRunnerTests" + ("." + name if name else "")],
            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)

    def test_unmodified_copied_source_passes(self):
        with tempfile.TemporaryDirectory(prefix="cbz-node-baseline-") as directory:
            path = Path(directory) / "runner.py"
            path.write_bytes(SOURCE.read_bytes())
            result = self.invoke(path, None)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_every_source_mutation_has_its_intended_assertion(self):
        source = SOURCE.read_text()
        with tempfile.TemporaryDirectory(prefix="cbz-node-mutants-") as directory:
            for name, old, new, expected in self.MUTATIONS:
                with self.subTest(name=name):
                    self.assertEqual(source.count(old), 1, "mutation anchor drift: " + name)
                    path = Path(directory) / (name + ".py")
                    path.write_text(source.replace(old, new))
                    result = self.invoke(path, expected)
                    self.assertTrue(assertion_kill(result, expected),
                                    name + "\n" + result.stdout + result.stderr)

    def test_harness_mutations_are_rejected_by_named_controls(self):
        full_source = Path(__file__).read_text()
        source, suffix = full_source.split("\nclass SourceMutationTests", 1)
        mutations = [
            ("import",
             '    if re.search(r"(?:ImportError|ModuleNotFoundError|SyntaxError|IndentationError|FileNotFoundError):", text):\n        return False\n',
             "", "test_import_trace_inside_assertion_cannot_kill"),

            ("exit", "process.returncode == 0 or not summary", "not summary",
             "test_success_exit_cannot_kill"),
            ("errors", ' or "errors=" in summary.group(1)', "",
             "test_errors_cannot_kill"),
            ("foreign", "return expected in failures", "return bool(failures)",
             "test_foreign_assertion_cannot_kill"),
        ]
        with tempfile.TemporaryDirectory(prefix="cbz-node-harness-mutants-") as directory:
            for name, old, new, expected in mutations:
                with self.subTest(name=name):
                    self.assertEqual(source.count(old), 1, "harness anchor drift")
                    path = Path(directory) / ("test-" + name + ".py")
                    path.write_text(source.replace(old, new) + "\nclass SourceMutationTests" + suffix)
                    result = subprocess.run(
                        [sys.executable, *(["-O"] if sys.flags.optimize else []), "-B", "-S",
                         str(path), "MutationHarnessTests." + expected, "-v"],
                        env=dict(os.environ), capture_output=True, text=True, timeout=120)
                    self.assertTrue(assertion_kill(result, expected),
                                    result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
