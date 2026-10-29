"""Real Git/subprocess contracts for the bounded prepared Node runner."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/cbz_node_regression.py"


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
        cases = [good.replace("# tests 3", "# tests 0"),
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
                dict(os.environ), log)
        self.assertEqual(code, 7)
        self.assertEqual(log_path.read_text(), "real stdout\n")

    def test_missing_executable_is_an_execution_error(self):
        with (self.directory / "missing.log").open("wb") as log:
            with self.assertRaises(FileNotFoundError):
                self.module.execute_node(
                    self.repo, [str(self.directory / "absent")],
                    dict(os.environ), log)

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
                dict(os.environ), log)
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


if __name__ == "__main__":
    unittest.main()
