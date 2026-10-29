"""Real Git/subprocess contracts for the bounded prepared Node runner."""
import hashlib
import importlib.util
import importlib._bootstrap_external
import io
import inspect
import secrets
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("CBZ_NODE_RUNNER_SOURCE", ROOT / "tools/cbz_node_regression.py"))


def runner():
    if not SOURCE.is_file():
        raise AssertionError("prepared Node runner is missing")
    sys.path.insert(0, str(ROOT / "tools"))
    module = types.ModuleType("cbz_node_regression_tested")
    module.__file__ = str(SOURCE)
    exec(compile(SOURCE.read_bytes(), str(SOURCE), "exec"), module.__dict__)
    return module


class _ObservedRegex:
    def __init__(self, inner, case, pattern, frame):
        self.inner, self.case, self.pattern, self.frame = inner, case, pattern, frame

    def __enter__(self):
        return self.inner.__enter__()

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.case.observe_missing_rejection(self.pattern, self.frame.f_locals)
        return self.inner.__exit__(exc_type, exc, traceback)


class NodeRunnerTests(unittest.TestCase):
    def mark_bypass(self, event):
        nonce = os.environ.get("CBZ_NODE_MUTANT_NONCE")
        path = os.environ.get("CBZ_NODE_MUTANT_RECEIPT")
        wanted = os.environ.get("CBZ_NODE_MUTANT_NAME")
        if nonce and path and wanted == event:
            with open(path, "x", encoding="utf-8") as output:
                json.dump({"nonce": nonce, "mutation": event,
                           "test": self._testMethodName,
                           "observation": "intended_bypass_observed"}, output)

    def assertRaisesRegex(self, expected_exception, expected_regex, *args, **kwargs):
        inner = super().assertRaisesRegex(expected_exception, expected_regex, *args, **kwargs)
        if args or kwargs or not os.environ.get("CBZ_NODE_MUTANT_RECEIPT"):
            return inner
        return _ObservedRegex(inner, self, expected_regex, inspect.currentframe().f_back)

    def observe_missing_rejection(self, pattern, variables):
        mutation = os.environ.get("CBZ_NODE_MUTANT_NAME")
        allowed = {
            "loaded-root": ("test_loaded_sources_must_belong_to_selected_main", "loaded source"),
            "selected-symlink": ("test_selected_node_test_symlink_is_rejected", "regular.*Node"),
            "external-source-symlink": ("test_tracked_source_symlink_cannot_bind_an_external_dependency", "unsupported tracked Git entry"),
            "fixture-domain": ("test_child_fixture_mutation_is_a_failed_postflight", "fixture"),
            "fixture-postflight": ("test_child_fixture_mutation_is_a_failed_postflight", "fixture"),
            "length": ("test_all_byte_identities_are_required", "source identity mismatch"),
            "sha256": ("test_all_byte_identities_are_required", "source identity mismatch"),
            "declared-blob": ("test_all_byte_identities_are_required", "source identity mismatch"),
            "regular-blob": ("test_symlink_is_not_admitted_as_a_regular_fixture", "not a regular Git blob"),
            "url": ("test_url_alias_is_rejected", "URL/identity"),
            "skip-inputs": ("test_materialization_checks_every_input_before_writing", "source identity mismatch"),
            "hidden-bytes": ("test_snapshot_detects_assume_unchanged_source_changes", "tracked bytes differ"),
            "hidden-mode": ("test_snapshot_detects_hidden_executable_mode_changes", "executable mode changed"),
            "zero-tests": ("test_tap_requires_nonzero_complete_all_passed_summary", "failed.*skipped/todo"),
            "skip-summary": ("test_real_node_skips_are_not_a_green_suite", "skipped/todo"),
            "duplicate-summary": ("test_tap_missing_or_duplicate_fields_refuse", "missing or ambiguous"),
            "empty-discovery": ("test_zero_tracked_node_tests_refuses", "no tracked Node tests"),
        }
        if allowed.get(mutation) != (self._testMethodName, pattern):
            return
        fields = {"length": "bytes", "sha256": "sha256", "declared-blob": "blob"}
        if mutation in fields and variables.get("field") != fields[mutation]:
            return
        if mutation == "zero-tests" and ("# tests 0" not in variables.get("value", "") or "# pass 0" not in variables.get("value", "")):
            return
        if mutation == "duplicate-summary" and (variables.get("field") != "tests" or variables.get("text", "").count("# tests ") != 2):
            return
        if mutation == "skip-summary":
            text = variables["path"].read_text()
            if "# tests 1" not in text or "# skipped 1" not in text:
                return
        if mutation == "skip-inputs":
            fixture = variables["output"] / "main/data.txt"
            if not fixture.is_file() or fixture.read_bytes() != b"exact input\n":
                return
        if mutation in ("fixture-domain", "fixture-postflight"):
            if variables.get("observed_exit") != 0:
                return
            fixture = variables["output"] / "shop/Math-/data.txt"
            transcript = variables["path"]
            if fixture.read_bytes() != b"mutated\n" or not one_pass_tap(transcript.read_text()):
                return
        self.mark_bypass(mutation)

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
                if str(error).startswith("Git command failed: ls-tree -z "):
                    self.mark_bypass("ambient-git")
                self.fail("ambient Git redirection affected pinned read: " + str(error))
            self.assertEqual(raw, b"exact input\n")
        finally:
            if before is None:
                os.environ.pop("GIT_DIR", None)
            else:
                os.environ["GIT_DIR"] = before

    def test_poisoned_runner_bytecode_cannot_execute_in_test_loader(self):
        self.assertIsNone(sys.pycache_prefix, "bytecode-cache control requires the default cache location")
        tools = self.repo / "tools"
        tools.mkdir()
        runner_path = tools / "cbz_node_regression.py"
        runner_path.write_bytes(SOURCE.read_bytes())
        (tools / "museum_data.py").write_bytes((ROOT / "tools/museum_data.py").read_bytes())
        stat = runner_path.stat()
        suffix = ".opt-" + str(sys.flags.optimize) if sys.flags.optimize else ""
        cache = tools / "__pycache__" / ("cbz_node_regression." + sys.implementation.cache_tag + suffix + ".pyc")
        cache.parent.mkdir()
        forged = compile("POISON_RUNNER = True\n", str(runner_path), "exec")
        cache.write_bytes(importlib._bootstrap_external._code_to_timestamp_pyc(
            forged, int(stat.st_mtime), stat.st_size))
        old_source = SOURCE
        try:
            globals()["SOURCE"] = runner_path
            module = runner()
        finally:
            globals()["SOURCE"] = old_source
        self.assertFalse(getattr(module, "POISON_RUNNER", False))

    def test_poisoned_museum_bytecode_cannot_pass_source_binding(self):
        tools = self.repo / "tools"
        tools.mkdir()
        runner_path = tools / "cbz_node_regression.py"
        museum_path = tools / "museum_data.py"
        runner_path.write_bytes(SOURCE.read_bytes())
        museum_path.write_bytes((ROOT / "tools/museum_data.py").read_bytes())
        stat = museum_path.stat()
        suffix = ".opt-" + str(sys.flags.optimize) if sys.flags.optimize else ""
        cache = tools / "__pycache__" / ("museum_data." + sys.implementation.cache_tag + suffix + ".pyc")
        cache.parent.mkdir()
        forged = compile("POISON_EXECUTED = True\n", str(museum_path), "exec")
        cache.write_bytes(importlib._bootstrap_external._code_to_timestamp_pyc(
            forged, int(stat.st_mtime), stat.st_size))
        script = (
            "import sys\nfrom pathlib import Path\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import cbz_node_regression as recorder\n"
            "recorder.assert_loaded_sources(Path(sys.argv[1]).parent)\n"
            "print('CBZ_POISONED' if getattr(recorder.museum_data, 'POISON_EXECUTED', False) else 'CBZ_SOURCE_ONLY')\n")
        environment = {key: value for key, value in os.environ.items()
                       if key not in ("PYTHONPATH", "PYTHONHOME", "PYTHONPYCACHEPREFIX")}
        result = subprocess.run(
            [sys.executable, *(["-O"] if sys.flags.optimize else []), "-B", "-S",
             "-c", script, str(tools)], env=environment, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and "CBZ_POISONED" in result.stdout:
            self.mark_bypass("museum-bytecode")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CBZ_SOURCE_ONLY", result.stdout)

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
            if loaded_bypass_observed(result):
                self.mark_bypass("loaded-bytes")
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
                observed_exit = self.module.execute_node(
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
        if code == 0 and log_path.read_bytes() == b"real stdout\n":
            self.mark_bypass("exit")
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
        if (result["status"] == "PASS" and result.get("error") == "loaded source root identity mismatch"
                and not result["mathematical_certification"] and not result["final_cbz_certification"]):
            self.mark_bypass("false-green-report")
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


def one_pass_tap(text):
    expected = {"tests": 1, "pass": 1, "fail": 0, "cancelled": 0, "skipped": 0, "todo": 0}
    lines = text.splitlines()
    for field, value in expected.items():
        matching = [line for line in lines if line.startswith("# " + field)]
        if matching != [f"# {field} {value}"]:
            return False
    return True


def loaded_bypass_observed(result):
    return (result.returncode == 1
            and result.stdout.splitlines().count("CONTROL_IMPORT_PASS") == 1
            and result.stderr.splitlines().count(
                "AssertionError: changed loaded source was accepted") == 1)


def assertion_kill(process, expected):
    text = process.stdout + process.stderr
    summary = re.search(r"^FAILED \(([^)]*)\)$", text, re.MULTILINE)
    if re.search(r"(?:ImportError|ModuleNotFoundError|SyntaxError|IndentationError|FileNotFoundError):", text):
        return False
    if process.returncode != 1 or not summary or not re.fullmatch(r"failures=[1-9][0-9]*", summary.group(1)):
        return False
    runs = re.findall(r"^Ran [^\n]*$", text, re.MULTILINE)
    if len(runs) != 1 or not re.fullmatch(r"Ran 1 test in [0-9]+(?:\.[0-9]+)?s", runs[0]):
        return False
    failures = re.findall(r"^FAIL: (test_[a-zA-Z0-9_]+) \(", text, re.MULTILINE)
    count = int(re.match(r"failures=([0-9]+)", summary.group(1)).group(1))
    return len(failures) == count and all(name == expected for name in failures)


def receipt_kill(process, expected, receipt, nonce, mutation):
    required = {"nonce": nonce, "mutation": mutation, "test": expected,
                "observation": "intended_bypass_observed"}
    return process.returncode == 1 and assertion_kill(process, expected) and receipt == required


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

    def test_expected_and_foreign_failures_cannot_kill(self):
        result = self.result(summary="failures=2")
        result.stderr += "\nFAIL: test_foreign (fixture.Contract)\n"
        self.assertFalse(assertion_kill(result, "test_expected"))

    def test_skips_and_expected_failures_cannot_kill(self):
        for summary in ("failures=1, skipped=1",
                        "failures=1, expected failures=1",
                        "failures=1, unexpected successes=1"):
            with self.subTest(summary=summary):
                self.assertFalse(assertion_kill(self.result(summary=summary), "test_expected"))

    def test_abnormal_test_process_exit_cannot_kill(self):
        self.assertFalse(assertion_kill(self.result(code=86), "test_expected"))

    def test_mismatched_failure_count_cannot_kill(self):
        self.assertFalse(assertion_kill(self.result(summary="failures=2"), "test_expected"))

    def test_no_tests_cannot_kill(self):
        result = self.result()
        result.stderr = result.stderr.replace("Ran 1 test", "Ran 0 tests")
        self.assertFalse(assertion_kill(result, "test_expected"))

    def test_import_trace_inside_assertion_cannot_kill(self):
        result = self.result()
        result.stderr += "\nImportError: child module could not load\n"
        self.assertFalse(assertion_kill(result, "test_expected"))

    def test_receipt_accepts_only_the_matching_observation(self):
        receipt = {"nonce": "fresh", "mutation": "m", "test": "test_expected",
                   "observation": "intended_bypass_observed"}
        self.assertTrue(receipt_kill(self.result(), "test_expected", receipt, "fresh", "m"))
        for change in (None, {}, dict(receipt, nonce="old"),
                       dict(receipt, mutation="other"),
                       dict(receipt, test="test_foreign"),
                       dict(receipt, observation="wrong_refusal")):
            self.assertFalse(receipt_kill(self.result(), "test_expected", change, "fresh", "m"))
        self.assertFalse(receipt_kill(self.result(code=86), "test_expected", None, "fresh", "m"))
        self.assertFalse(receipt_kill(self.result(code=86), "test_expected", receipt, "fresh", "m"))
        self.assertFalse(receipt_kill(self.result(expected="test_foreign"), "test_expected", receipt, "fresh", "m"))

    def test_control_receipt_requires_the_exact_subject(self):
        with tempfile.TemporaryDirectory(prefix="cbz-receipt-control-") as directory:
            receipt = Path(directory) / "receipt.json"
            names = ("CBZ_NODE_MUTANT_NONCE", "CBZ_NODE_MUTANT_RECEIPT", "CBZ_NODE_MUTANT_NAME")
            previous = {name: os.environ.get(name) for name in names}
            def restore():
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
            self.addCleanup(restore)
            os.environ.update(CBZ_NODE_MUTANT_NONCE="fresh",
                              CBZ_NODE_MUTANT_RECEIPT=str(receipt),
                              CBZ_NODE_MUTANT_NAME="length")
            case = NodeRunnerTests("test_all_byte_identities_are_required")
            case.mark_bypass("foreign-event")
            self.assertFalse(receipt.exists())
            case.observe_missing_rejection("source identity mismatch", {"field": "sha256"})
            case.observe_missing_rejection("wrong refusal", {"field": "bytes"})
            self.assertFalse(receipt.exists())
            os.environ.pop("CBZ_NODE_MUTANT_NONCE")
            case.observe_missing_rejection("source identity mismatch", {"field": "bytes"})
            self.assertFalse(receipt.exists())
            os.environ["CBZ_NODE_MUTANT_NONCE"] = "fresh"
            case.observe_missing_rejection("source identity mismatch", {"field": "bytes"})
            self.assertEqual(json.loads(receipt.read_text()), {
                "nonce": "fresh", "mutation": "length",
                "test": "test_all_byte_identities_are_required",
                "observation": "intended_bypass_observed"})

    def test_fixture_receipt_requires_successful_node_exit(self):
        with tempfile.TemporaryDirectory(prefix="cbz-fixture-observation-") as directory:
            root = Path(directory)
            fixture = root / "shop/Math-/data.txt"
            fixture.parent.mkdir(parents=True)
            fixture.write_bytes(b"mutated\n")
            transcript = root / "node.tap"
            transcript.write_text("# tests 1\n# pass 1\n# fail 0\n# cancelled 0\n# skipped 0\n# todo 0\n")
            names = ("CBZ_NODE_MUTANT_NONCE", "CBZ_NODE_MUTANT_RECEIPT", "CBZ_NODE_MUTANT_NAME")
            previous = {name: os.environ.get(name) for name in names}
            def restore():
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
            self.addCleanup(restore)
            case = NodeRunnerTests("test_child_fixture_mutation_is_a_failed_postflight")
            for mutation in ("fixture-domain", "fixture-postflight"):
                receipt = root / (mutation + ".json")
                os.environ.update(CBZ_NODE_MUTANT_NONCE="fresh",
                                  CBZ_NODE_MUTANT_RECEIPT=str(receipt),
                                  CBZ_NODE_MUTANT_NAME=mutation)
                inputs = {"output": root, "path": transcript, "observed_exit": 1}
                case.observe_missing_rejection("fixture", inputs)
                self.assertFalse(receipt.exists(), mutation)
                inputs["observed_exit"] = 0
                case.observe_missing_rejection("fixture", inputs)
                self.assertTrue(receipt.exists(), mutation)

    def test_selected_control_must_run_exactly_one_test(self):
        for kind in ("two-tests", "duplicate-summary"):
            with self.subTest(kind=kind):
                result = self.result()
                if kind == "two-tests":
                    result.stderr = result.stderr.replace("Ran 1 test", "Ran 2 tests")
                else:
                    result.stderr += "Ran 1 test in 0.002s\n"
                self.assertFalse(assertion_kill(result, "test_expected"))

    def test_fixture_receipt_rejects_ambiguous_or_incomplete_tap(self):
        with tempfile.TemporaryDirectory(prefix="cbz-fixture-tap-control-") as directory:
            root = Path(directory)
            fixture = root / "shop/Math-/data.txt"
            fixture.parent.mkdir(parents=True)
            fixture.write_bytes(b"mutated\n")
            transcript = root / "node.tap"
            case = NodeRunnerTests("test_child_fixture_mutation_is_a_failed_postflight")
            valid = "# tests 1\n# pass 1\n# fail 0\n# cancelled 0\n# skipped 0\n# todo 0\n"
            examples = ("# pass 10\n", "# pass 1\n",
                        valid + "# pass 1\n",
                        valid.replace("# fail 0\n", "# fail 1\n"),
                        valid.replace("# skipped 0\n", "# skipped 1\n"),
                        valid.replace("# cancelled 0\n", "# cancelled 1\n"),
                        valid.replace("# pass 1\n", "# pass 10\n"))
            for mutation in ("fixture-domain", "fixture-postflight"):
                for index, value in enumerate(examples):
                    with self.subTest(mutation=mutation, index=index):
                        receipt = root / (mutation + str(index) + ".json")
                        transcript.write_text(value)
                        with patch.dict(os.environ, {"CBZ_NODE_MUTANT_NONCE": "fresh",
                                                    "CBZ_NODE_MUTANT_RECEIPT": str(receipt),
                                                    "CBZ_NODE_MUTANT_NAME": mutation}):
                            case.observe_missing_rejection("fixture",
                                {"output": root, "path": transcript, "observed_exit": 0})
                        self.assertFalse(receipt.exists(), (mutation, value))

    def test_loaded_bytes_observation_requires_exact_failure(self):
        observer = globals().get("loaded_bypass_observed")
        self.assertTrue(callable(observer), "loaded-source observation must be explicit and testable")
        good = subprocess.CompletedProcess([], 1, "CONTROL_IMPORT_PASS\n",
                                           "Traceback (most recent call last):\n  File \"<string>\", line 9, in <module>\nAssertionError: changed loaded source was accepted\n")
        self.assertTrue(observer(good))
        for candidate in (
            subprocess.CompletedProcess([], 86, good.stdout, good.stderr),
            subprocess.CompletedProcess([], 0, good.stdout, good.stderr),
            subprocess.CompletedProcess([], 1, "CONTROL_IMPORT_PASS_EXTRA\n", good.stderr),
            subprocess.CompletedProcess([], 1, good.stdout, "crash: changed loaded source was accepted\n"),
            subprocess.CompletedProcess([], 1, good.stdout, good.stderr + good.stderr),
        ):
            with self.subTest(candidate=candidate):
                self.assertFalse(observer(candidate))

    def test_mutant_child_does_not_execute_poisoned_test_cache(self):
        for use_prefix in (False, True):
            with self.subTest(inherited_prefix=use_prefix), tempfile.TemporaryDirectory(prefix="cbz-node-child-cache-") as directory:
                root = Path(directory) / "workspace"
                package = root / "tests"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text("")
                source = package / "test_cbz_node_regression.py"
                source.write_text("import unittest\nclass NodeRunnerTests(unittest.TestCase):\n def test_child(self): print('CBZ_CHILD_SOURCE')\n")
                stat = source.stat()
                prefix = str(Path(directory) / "inherited") if use_prefix else None
                with patch.object(sys, "pycache_prefix", prefix):
                    cache = Path(importlib.util.cache_from_source(
                        str(source), optimization=str(sys.flags.optimize) if sys.flags.optimize else ""))
                cache.parent.mkdir(parents=True)
                poisoned = compile("print('CBZ_CHILD_POISON'); raise RuntimeError('poison')\n", str(source), "exec")
                cache.write_bytes(importlib._bootstrap_external._code_to_timestamp_pyc(
                    poisoned, int(stat.st_mtime), stat.st_size))
                (root / "tools").mkdir()
                (root / "tools/museum_data.py").write_text("")
                target = root / "mutant.py"
                target.write_text("")
                original = ROOT
                globals()["ROOT"] = root
                try:
                    with patch.dict(os.environ):
                        if prefix is None:
                            os.environ.pop("PYTHONPYCACHEPREFIX", None)
                        else:
                            os.environ["PYTHONPYCACHEPREFIX"] = prefix
                        result = SourceMutationTests("test_unmodified_copied_source_passes").invoke(target, "test_child")
                finally:
                    globals()["ROOT"] = original
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("CBZ_CHILD_SOURCE", result.stdout)
                self.assertNotIn("CBZ_CHILD_POISON", result.stdout + result.stderr)

    def test_outer_control_launcher_ignores_poisoned_test_cache(self):
        launcher = Path(os.environ.get("CBZ_NODE_CONTROL_LAUNCHER", ROOT / "tools/cbz_node_controls.py"))
        self.assertTrue(launcher.is_file(), "source-only control launcher is required")
        with tempfile.TemporaryDirectory(prefix="cbz-node-outer-cache-") as directory:
            root = Path(directory) / "workspace"
            package = root / "tests"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            source = package / "test_cbz_node_regression.py"
            source.write_text("import unittest\nclass TestOuter(unittest.TestCase):\n def test_source(self): print('CBZ_OUTER_SOURCE')\n")
            stat = source.stat()
            inherited = Path(directory) / "inherited"
            cache_paths = []
            for prefix in (None, str(inherited)):
                with patch.object(sys, "pycache_prefix", prefix):
                    cache_paths.append(Path(importlib.util.cache_from_source(
                        str(source), optimization=str(sys.flags.optimize) if sys.flags.optimize else "")))
            for cache in cache_paths:
                cache.parent.mkdir(parents=True, exist_ok=True)
                poisoned = compile("print('CBZ_OUTER_POISON'); raise RuntimeError('poison')\n", str(source), "exec")
                cache.write_bytes(importlib._bootstrap_external._code_to_timestamp_pyc(
                    poisoned, int(stat.st_mtime), stat.st_size))
            for prefix in (None, str(inherited)):
                with self.subTest(prefix=prefix), patch.dict(os.environ):
                    if prefix is None:
                        os.environ.pop("PYTHONPYCACHEPREFIX", None)
                    else:
                        os.environ["PYTHONPYCACHEPREFIX"] = prefix
                    result = subprocess.run(
                        [sys.executable, *(["-O"] if sys.flags.optimize else []), "-I", "-B", "-S",
                         str(launcher), "--root", str(root)],
                        capture_output=True, text=True, env=dict(os.environ), timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("CBZ_OUTER_SOURCE", result.stdout)
                    self.assertNotIn("CBZ_OUTER_POISON", result.stdout + result.stderr)

    def test_outer_control_launcher_refuses_empty_and_skipped_suites(self):
        launcher = Path(os.environ.get("CBZ_NODE_CONTROL_LAUNCHER", ROOT / "tools/cbz_node_controls.py"))
        self.assertTrue(launcher.is_file(), "source-only control launcher is required")
        with tempfile.TemporaryDirectory(prefix="cbz-node-empty-controls-") as directory:
            root = Path(directory)
            package = root / "tests"
            package.mkdir()
            (package / "__init__.py").write_text("")
            source = package / "test_cbz_node_regression.py"
            examples = ("", "import unittest\n@unittest.skip('not exercised')\nclass TestSkipped(unittest.TestCase):\n def test_skipped(self): pass\n")
            for text in examples:
                with self.subTest(source=text):
                    source.write_text(text)
                    result = subprocess.run(
                        [sys.executable, *(["-O"] if sys.flags.optimize else []), "-I", "-B", "-S",
                         str(launcher), "--root", str(root)],
                        capture_output=True, text=True, timeout=30)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_source_loader_ignores_poisoned_bytecode(self):
        self.assertIsNone(sys.pycache_prefix, "bytecode-cache control requires the default cache location")
        with tempfile.TemporaryDirectory(prefix="cbz-loader-control-") as directory:
            path = Path(directory) / "probe.py"
            path.write_text("GOOD_SOURCE = True\n")
            stat = path.stat()
            suffix = ".opt-" + str(sys.flags.optimize) if sys.flags.optimize else ""
            cache = path.parent / "__pycache__" / ("probe." + sys.implementation.cache_tag + suffix + ".pyc")
            cache.parent.mkdir()
            forged = compile("POISON_RUNNER = True\n", str(path), "exec")
            cache.write_bytes(importlib._bootstrap_external._code_to_timestamp_pyc(
                forged, int(stat.st_mtime), stat.st_size))
            old_source = SOURCE
            try:
                globals()["SOURCE"] = path
                module = runner()
            finally:
                globals()["SOURCE"] = old_source
            self.assertFalse(getattr(module, "POISON_RUNNER", False))
            self.assertTrue(getattr(module, "GOOD_SOURCE", False))

    def test_launch_failure_cannot_kill(self):
        result = subprocess.CompletedProcess([], 1, "", "SyntaxError: broken module")
        self.assertFalse(assertion_kill(result, "test_expected"))


class SourceMutationTests(unittest.TestCase):
    MUTATIONS = [
        ("museum-bytecode", "_MUSEUM_PATH = Path(__file__).resolve().with_name(\"museum_data.py\")\n_MUSEUM_SOURCE = _MUSEUM_PATH.read_bytes()\nmuseum_data = types.ModuleType(\"_cbz_museum_data\")\nmuseum_data.__file__ = str(_MUSEUM_PATH)\nexec(compile(_MUSEUM_SOURCE, str(_MUSEUM_PATH), \"exec\"), museum_data.__dict__)\n", "import museum_data\n_MUSEUM_PATH = Path(museum_data.__file__).resolve()\n_MUSEUM_SOURCE = _MUSEUM_PATH.read_bytes()\n",
         "test_poisoned_museum_bytecode_cannot_pass_source_binding"),
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

    def invoke(self, path, name, receipt=None, nonce=None, mutation=None):
        path.with_name("museum_data.py").write_bytes((ROOT / "tools/museum_data.py").read_bytes())
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("PYTHON")}
        environment["CBZ_NODE_RUNNER_SOURCE"] = str(path)
        if receipt is not None:
            environment.update(CBZ_NODE_MUTANT_RECEIPT=str(receipt), CBZ_NODE_MUTANT_NONCE=nonce, CBZ_NODE_MUTANT_NAME=mutation)
        target = "tests.test_cbz_node_regression.NodeRunnerTests" + ("." + name if name else "")
        bootstrap = ("import sys, unittest\n"
                     "sys.path.insert(0, sys.argv[1])\n"
                     "unittest.main(module=None, argv=['unittest', '-v', sys.argv[2]])\n")
        with tempfile.TemporaryDirectory(prefix="cbz-node-child-pycache-") as cache:
            return subprocess.run(
                [sys.executable, *(["-O"] if sys.flags.optimize else []), "-I", "-B", "-S",
                 "-X", f"pycache_prefix={cache}", "-c", bootstrap, str(ROOT), target],
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
                    receipt_path = Path(directory) / (name + ".receipt.json")
                    nonce = secrets.token_hex(16)
                    result = self.invoke(path, expected, receipt_path, nonce, name)
                    try:
                        receipt = json.loads(receipt_path.read_text())
                    except (FileNotFoundError, ValueError):
                        receipt = None
                    self.assertTrue(receipt_kill(result, expected, receipt, nonce, name),
                                    name + "\n" + result.stdout + result.stderr)

    def test_harness_mutations_are_rejected_by_named_controls(self):
        full_source = Path(__file__).read_text()
        source, suffix = full_source.split("\nclass SourceMutationTests", 1)
        mutations = [
            ("runner-loader", "    module = types.ModuleType(\"cbz_node_regression_tested\")\n    module.__file__ = str(SOURCE)\n    exec(compile(SOURCE.read_bytes(), str(SOURCE), \"exec\"), module.__dict__)\n", "    spec = importlib.util.spec_from_file_location(\"cbz_node_regression_tested\", SOURCE)\n    module = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(module)\n",
             "test_source_loader_ignores_poisoned_bytecode"),
            ("fixture-exit", 'if variables.get("observed_exit") != 0:', "if False:",
             "test_fixture_receipt_requires_successful_node_exit"),
            ("selected-test-count",
             'len(runs) != 1 or not re.fullmatch(r"Ran 1 test in [0-9]+(?:\.[0-9]+)?s", runs[0])',
             "False", "test_selected_control_must_run_exactly_one_test"),
            ("fixture-tap", "not one_pass_tap(transcript.read_text())", "False",
             "test_fixture_receipt_rejects_ambiguous_or_incomplete_tap"),
            ("loaded-exit", "result.returncode == 1", "result.returncode != 0",
             "test_loaded_bytes_observation_requires_exact_failure"),
            ("loaded-exact", 'result.stderr.splitlines().count(\n                "AssertionError: changed loaded source was accepted") == 1',
             '"changed loaded source was accepted" in result.stderr',
             "test_loaded_bytes_observation_requires_exact_failure"),
            ("count", "len(failures) == count", "True",
             "test_mismatched_failure_count_cannot_kill"),
            ("receipt", "receipt == required", "True",
             "test_receipt_accepts_only_the_matching_observation"),
            ("subject", 'if mutation in fields and variables.get("field") != fields[mutation]:',
             "if False:", "test_control_receipt_requires_the_exact_subject"),
            ("nonce", "if nonce and path and wanted == event:",
             "if path and wanted == event:", "test_control_receipt_requires_the_exact_subject"),
            ("event", "if nonce and path and wanted == event:",
             "if nonce and path:", "test_control_receipt_requires_the_exact_subject"),
            ("import",
             '    if re.search(r"(?:ImportError|ModuleNotFoundError|SyntaxError|IndentationError|FileNotFoundError):", text):\n        return False\n',
             "", "test_import_trace_inside_assertion_cannot_kill"),

            ("exit", "process.returncode != 1 or not summary", "not summary",
             "test_success_exit_cannot_kill"),
            ("errors", ' or not re.fullmatch(r"failures=[1-9][0-9]*", summary.group(1))', "",
             "test_errors_cannot_kill"),
            ("foreign", "all(name == expected for name in failures)", "bool(failures)",
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


    def test_child_cache_guard_mutation_is_rejected(self):
        full_source = Path(__file__).read_text()
        prefix, suffix = full_source.split("\nclass SourceMutationTests", 1)
        old = '"-X", f"pycache_prefix={cache}",'
        before, after = suffix.split("\n    def test_child_cache_guard_mutation_is_rejected", 1)
        self.assertEqual(before.count(old), 1)
        with tempfile.TemporaryDirectory(prefix="cbz-node-child-mutant-") as directory:
            test_path = Path(directory) / "test-child-mutant.py"
            test_path.write_text(prefix + "\nclass SourceMutationTests" + before.replace(old, "", 1)
                                 + "\n    def test_child_cache_guard_mutation_is_rejected" + after)
            result = subprocess.run(
                [sys.executable, *(["-O"] if sys.flags.optimize else []), "-I", "-B", "-S",
                 str(test_path), "MutationHarnessTests.test_mutant_child_does_not_execute_poisoned_test_cache", "-v"],
                capture_output=True, text=True, timeout=120)
            self.assertTrue(assertion_kill(result, "test_mutant_child_does_not_execute_poisoned_test_cache"),
                            result.stdout + result.stderr)
            self.assertIn("CBZ_CHILD_POISON", result.stdout + result.stderr)
            self.assertIn("RuntimeError: poison", result.stdout + result.stderr)

    def test_outer_empty_guard_mutation_is_rejected(self):
        launcher = ROOT / "tools/cbz_node_controls.py"
        source = launcher.read_text()
        old = 'not re.fullmatch(r"Ran [1-9][0-9]* tests? in [0-9]+(?:\.[0-9]+)?s", runs[0])'
        self.assertEqual(source.count(old), 1)
        with tempfile.TemporaryDirectory(prefix="cbz-node-empty-mutant-") as directory:
            mutant = Path(directory) / "controls-mutant.py"
            mutant.write_text(source.replace(old, "False", 1))
            environment = dict(os.environ, CBZ_NODE_CONTROL_LAUNCHER=str(mutant))
            result = subprocess.run(
                [sys.executable, *(["-O"] if sys.flags.optimize else []), "-I", "-B", "-S",
                 str(Path(__file__).resolve()),
                 "MutationHarnessTests.test_outer_control_launcher_refuses_empty_and_skipped_suites", "-v"],
                env=environment, cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertTrue(assertion_kill(result, "test_outer_control_launcher_refuses_empty_and_skipped_suites"),
                            result.stdout + result.stderr)
            self.assertIn("Ran 0 tests", result.stdout + result.stderr)

    def test_outer_cache_guard_mutation_is_rejected(self):
        launcher = ROOT / "tools/cbz_node_controls.py"
        source = launcher.read_text()
        old = '"-X", f"pycache_prefix={cache}",'
        self.assertEqual(source.count(old), 1)
        with tempfile.TemporaryDirectory(prefix="cbz-node-outer-mutant-") as directory:
            mutant = Path(directory) / "controls-mutant.py"
            mutant.write_text(source.replace(old, "", 1))
            environment = dict(os.environ, CBZ_NODE_CONTROL_LAUNCHER=str(mutant))
            result = subprocess.run(
                [sys.executable, *(["-O"] if sys.flags.optimize else []), "-I", "-B", "-S",
                 str(Path(__file__).resolve()),
                 "MutationHarnessTests.test_outer_control_launcher_ignores_poisoned_test_cache", "-v"],
                env=environment, cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertTrue(assertion_kill(result, "test_outer_control_launcher_ignores_poisoned_test_cache"),
                            result.stdout + result.stderr)
            self.assertIn("CBZ_OUTER_POISON", result.stdout + result.stderr)
            self.assertIn("RuntimeError: poison", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
