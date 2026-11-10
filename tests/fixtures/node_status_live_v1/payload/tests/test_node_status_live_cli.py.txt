"""Offline CLI/source-context contracts for Q-9006; no token or live credit.

Owned local Git fixtures provide real immutable byte bindings. A child harness
forbids Python HTTP/socket connection attempts. The authenticated/live path is
not executed by this module. The existing 23 helper controls are unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SELECTED = (
    "tools/node_status_live_check.py",
    "tools/node_status_refresh.py",
    "tools/validate_node_status.py",
    "architecture/node-status.schema.json",
    "docs/site/current.json",
    "records/node_status/eng.d0-packet-allowlist-fix.json",
    "records/node_status/eng.hard-gate.json",
    "records/node_status/math.lifetime-remainder.json",
    "records/node_status/math.side24-coefficient.json",
)
HARNESS = r'''
from pathlib import Path
import runpy
import socket
import sys
import urllib.request
attempts = []
def forbidden(*args, **kwargs):
    attempts.append(True)
    raise AssertionError("OFFLINE_CLI_NETWORK_ATTEMPT")
urllib.request.urlopen = forbidden
socket.create_connection = forbidden
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(sys.argv[0]).parent))
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    if attempts:
        raise SystemExit("OFFLINE_CLI_NETWORK_ATTEMPT")
'''


# Separate fake-auth import probe. The original no-token HARNESS is unchanged.
IMPORT_PROBE_HARNESS = r'''
import json
from pathlib import Path
import runpy
import socket
import sys
import urllib.request

tool = Path(sys.argv[1]).resolve()
checkpoint = Path(sys.argv[2])
attempts = []

def forbidden(*args, **kwargs):
    attempts.append(True)
    raise AssertionError("OFFLINE_IMPORT_PROBE_NETWORK_ATTEMPT")

def deny_network(event, args):
    if event.startswith("socket.") or event.startswith("urllib."):
        forbidden()

urllib.request.urlopen = forbidden
socket.create_connection = forbidden
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
sys.addaudithook(deny_network)

def stop_at_bound_live_entry(frame, event, arg):
    if (event == "call" and frame.f_code.co_name == "run_live"
            and frame.f_globals.get("__name__") == "__main__"
            and Path(frame.f_code.co_filename).resolve() == tool):
        checkpoint.write_text(json.dumps({
            "phase": "BOUND_RUN_LIVE_ENTRY",
            "filename": str(tool),
            "root": str(Path(frame.f_locals["root"]).resolve()),
            "identity": frame.f_locals["identity"],
            "python_optimize": sys.flags.optimize,
        }), encoding="utf-8")
        raise RuntimeError("SYNTHETIC_TRACE_STOP_BEFORE_LIVE_BODY")
    return stop_at_bound_live_entry

sys.argv = [str(tool), *sys.argv[3:]]
sys.path.insert(0, str(tool.parent))
sys.settrace(stop_at_bound_live_entry)
try:
    runpy.run_path(str(tool), run_name="__main__")
finally:
    sys.settrace(None)
    if attempts:
        raise SystemExit("OFFLINE_IMPORT_PROBE_NETWORK_ATTEMPT")
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


class NodeStatusLiveCLIContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="q9006-cli-contract-")
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.hooks = self.home / "empty-hooks"
        self.hooks.mkdir()
        self.root = self.home / "selected"
        self.commit = self.make_repository(self.root)
        self.tool = self.root / "tools/node_status_live_check.py"
        self.output = self.home / "report.json"

    def git_env(self):
        return {
            "PATH": os.environ.get("PATH", os.defpath),
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
            "GIT_AUTHOR_NAME": "Synthetic CLI fixture",
            "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
            "GIT_COMMITTER_NAME": "Synthetic CLI fixture",
            "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
        }

    def git(self, root, *args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], env=self.git_env(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        return result.stdout.decode().strip()

    def make_repository(self, root):
        root.mkdir()
        for relative in SELECTED:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        self.git(root, "-c", "init.defaultBranch=main", "init", "-q")
        for key, value in (
            ("user.name", "Synthetic CLI fixture"),
            ("user.email", "fixture@example.invalid"),
            ("commit.gpgsign", "false"),
            ("core.hooksPath", str(self.hooks)),
            ("core.attributesFile", os.devnull),
            ("core.fsmonitor", "false"),
            ("core.autocrlf", "false"),
        ):
            self.git(root, "config", key, value)
        self.git(root, "add", "--all")
        self.git(root, "commit", "-q", "-m", "Synthetic source-context fixture")
        return self.git(root, "rev-parse", "HEAD")

    def environment(self, overrides=None):
        env = self.git_env()
        env.update({
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": "d6g8k5htny-coder/main",
            "GITHUB_SHA": self.commit,
            "GITHUB_RUN_ID": "123456",
            "GITHUB_RUN_ATTEMPT": "1",
        })
        for key, value in (overrides or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        self.assertNotIn("GITHUB_TOKEN", env)
        return env

    def invoke(self, arguments, *, overrides=None, optimized=None, cwd=None):
        mode = sys.flags.optimize if optimized is None else int(optimized)
        command = [sys.executable, "-B", "-S"]
        if mode:
            command.append("-O")
        command += ["-c", HARNESS, str(self.tool), *map(str, arguments)]
        return subprocess.run(
            command, cwd=self.home if cwd is None else cwd,
            env=self.environment(overrides), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30,
        )

    def run_cli(self, *, selected=None, output=None, include_root=True,
                overrides=None, optimized=None, extra=()):
        arguments = ["--output", self.output if output is None else output]
        if include_root:
            arguments = ["--root", self.root if selected is None else selected, *arguments]
        return self.invoke([*arguments, *extra], overrides=overrides, optimized=optimized)

    def refused(self, result, code):
        # A missing CLI currently exits zero. Fail in the test body before JSON parsing.
        self.assertNotEqual(result.returncode, 0, "CLI silently succeeded without the required refusal")
        self.assertEqual(result.stderr, b"", "CLI refusal must be structured, not a crash")
        self.assertTrue(result.stdout.strip(), "CLI omitted its structured refusal")
        try:
            report = json.loads(result.stdout)
        except (ValueError, UnicodeError) as error:
            self.fail("CLI refusal is not JSON: " + str(error))
        self.assertEqual(report["schema"], "cbz.node-status-live-cli.v1")
        self.assertEqual(report["status"], "REFUSED")
        self.assertEqual(report["error_code"], code)
        self.assertEqual(report["scientific_effect"], "NONE")
        self.assertIs(report["cbz_green"], False)
        return report

    def test_help_is_offline_and_describes_the_actual_root_output_interface(self):
        result = self.invoke(["--help"], overrides={
            "GITHUB_ACTIONS": None, "GITHUB_REPOSITORY": None, "GITHUB_SHA": None,
            "GITHUB_RUN_ID": None, "GITHUB_RUN_ATTEMPT": None,
        })
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, b"")
        self.assertIn(b"--root", result.stdout)
        self.assertIn(b"--output", result.stdout)
        self.assertFalse(self.output.exists())

    def test_missing_required_output_argument_is_not_success(self):
        result = self.invoke(["--root", self.root])
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"--output", result.stderr)
        self.assertFalse(self.output.exists())

    def test_missing_native_run_context_refuses(self):
        result = self.run_cli(overrides={"GITHUB_ACTIONS": None, "GITHUB_RUN_ID": None})
        self.refused(result, "RUN_CONTEXT_INVALID")
        self.assertFalse(self.output.exists())

    def test_wrong_repository_context_refuses(self):
        result = self.run_cli(overrides={"GITHUB_REPOSITORY": "other/repository"})
        self.refused(result, "RUN_CONTEXT_INVALID")
        self.assertFalse(self.output.exists())

    def test_context_commit_must_equal_the_actual_selected_commit(self):
        result = self.run_cli(overrides={"GITHUB_SHA": "f" * 40})
        self.refused(result, "CHECKED_COMMIT_MISMATCH")
        self.assertFalse(self.output.exists())

    def test_invalid_native_run_and_attempt_values_refuse(self):
        for change in (
            {"GITHUB_RUN_ID": "0"}, {"GITHUB_RUN_ID": "-1"},
            {"GITHUB_RUN_ATTEMPT": "0"}, {"GITHUB_RUN_ATTEMPT": "true"},
            {"GITHUB_SHA": "main"},
        ):
            with self.subTest(change=change):
                self.refused(self.run_cli(overrides=change), "RUN_CONTEXT_INVALID")
                self.assertFalse(self.output.exists())

    def test_selected_bytes_must_match_git_even_when_assume_unchanged(self):
        # Catches a source-identity check that trusts Git's stat/index shortcuts.
        for relative in SELECTED:
            with self.subTest(relative=relative):
                path = self.root / relative
                raw = path.read_bytes()
                self.git(self.root, "update-index", "--assume-unchanged", "--", relative)
                path.write_bytes(raw + b"\n")
                try:
                    self.refused(self.run_cli(), "SOURCE_BYTES_CHANGED")
                    self.assertFalse(self.output.exists())
                finally:
                    path.write_bytes(raw)
                    self.git(self.root, "update-index", "--no-assume-unchanged", "--", relative)

    def test_changed_dependency_cannot_execute_before_source_binding_refusal(self):
        path = self.root / "tools/validate_node_status.py"
        marker = self.home / "unbound-dependency-executed"
        raw = path.read_bytes()
        path.write_bytes(
            raw + b"\nfrom pathlib import Path\nPath(" + repr(str(marker)).encode()
            + b").write_text('UNBOUND_DEPENDENCY_EXECUTED')\n"
        )
        result = self.run_cli()
        self.assertFalse(marker.exists(), "CLI executed a changed dependency before binding source bytes")
        self.refused(result, "SOURCE_BYTES_CHANGED")

    def test_caller_cannot_select_another_checkout_even_with_identical_source(self):
        other = self.home / "other"
        other_commit = self.make_repository(other)
        self.refused(self.run_cli(selected=other, overrides={"GITHUB_SHA": other_commit}),
                     "ROOT_MISMATCH")
        self.assertFalse(self.output.exists())

    def test_default_root_is_the_executed_checkout_not_callers_cwd(self):
        report = self.refused(self.run_cli(include_root=False), "AUTH_REQUIRED")
        self.assertEqual(report["identity"]["checked_commit"], self.commit)

    def test_existing_output_is_not_overwritten(self):
        self.output.write_bytes(b"ORIGINAL_OUTPUT")
        self.refused(self.run_cli(), "OUTPUT_INVALID")
        self.assertEqual(self.output.read_bytes(), b"ORIGINAL_OUTPUT")

    def test_output_inside_the_selected_source_is_refused(self):
        output = self.root / "live-report.json"
        self.refused(self.run_cli(output=output), "OUTPUT_INVALID")
        self.assertFalse(output.exists())

    def test_output_symlink_is_refused_without_touching_its_target(self):
        target = self.home / "existing-target"
        target.write_bytes(b"KEEP_TARGET")
        self.output.symlink_to(target)
        self.refused(self.run_cli(), "OUTPUT_INVALID")
        self.assertEqual(target.read_bytes(), b"KEEP_TARGET")

    def test_output_parent_symlink_is_refused(self):
        target = self.home / "real-output-directory"
        target.mkdir()
        link = self.home / "linked-output-directory"
        link.symlink_to(target, target_is_directory=True)
        self.refused(self.run_cli(output=link / "report.json"), "OUTPUT_INVALID")
        self.assertFalse((target / "report.json").exists())

    def test_missing_token_retains_real_source_and_run_identity_without_live_credit(self):
        report = self.refused(self.run_cli(), "AUTH_REQUIRED")
        self.assertEqual(json.loads(self.output.read_bytes()), report)
        self.assertEqual(report["identity"], {
            "repository": "d6g8k5htny-coder/main",
            "checked_commit": self.commit, "run_id": "123456", "run_attempt": "1",
            "python_optimize": sys.flags.optimize,
        })
        self.assertEqual(set(report["source_bindings"]), set(SELECTED))
        for relative in SELECTED:
            raw = (self.root / relative).read_bytes()
            self.assertEqual(report["source_bindings"][relative]["git_blob"], blob(raw))
            self.assertEqual(report["source_bindings"][relative]["sha256"], sha(raw))
        self.assertNotIn("source_observations", report)
        self.assertNotIn("fresh_checked", report)

    def test_cli_reports_actual_interpreter_mode_in_each_child(self):
        for mode in (False, True):
            with self.subTest(mode=mode):
                output = self.home / ("mode-" + str(int(mode)) + ".json")
                report = self.refused(self.run_cli(output=output, optimized=mode), "AUTH_REQUIRED")
                self.assertIs(type(report["identity"]["python_optimize"]), int)
                self.assertEqual(report["identity"]["python_optimize"], int(mode))

    def test_ambient_git_path_redirects_cannot_replace_the_selected_checkout(self):
        other = self.home / "redirect"
        self.make_repository(other)
        (other / "unrelated.txt").write_text("OTHER OWNED FIXTURE\n")
        self.git(other, "add", "--all")
        self.git(other, "commit", "-q", "-m", "Different synthetic commit")
        report = self.refused(self.run_cli(overrides={
            "GIT_DIR": str(other / ".git"), "GIT_WORK_TREE": str(other),
        }), "AUTH_REQUIRED")
        self.assertEqual(report["identity"]["checked_commit"], self.commit)

    def test_api_or_transcript_override_is_not_a_supported_cli_input(self):
        for option in ("--api", "--transcript", "--observer"):
            with self.subTest(option=option):
                result = self.run_cli(extra=(option, "synthetic-never-live.json"))
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.output.exists())


    def _unbound_dependency_wrapper(self, source, marker):
        # Preserve the real dependency API after the controlled adverse side effect.
        # Only an owned, untracked package or bytecode file contains this wrapper.
        return (
            "from pathlib import Path\n"
            "Path(" + repr(str(marker)) + ").write_text("
            "'UNBOUND_DEPENDENCY_IMPORT_EXECUTED', encoding='utf-8')\n"
            "__file__ = " + repr(str(source)) + "\n"
            "exec(compile(" + repr(source.read_bytes())
            + ", __file__, 'exec', dont_inherit=True), globals())\n"
        )

    def _invoke_import_probe(self, label):
        checkpoint = self.home / (label + ".checkpoint.json")
        output = self.home / (label + ".report.json")
        environment = self.environment()
        environment["GITHUB_TOKEN"] = "SYNTHETIC_IMPORT_PROBE_NOT_A_CREDENTIAL"
        command = [sys.executable, "-B", "-S"]
        if sys.flags.optimize:
            command.append("-O")
        command += [
            "-c", IMPORT_PROBE_HARNESS, str(self.tool), str(checkpoint),
            "--root", str(self.root), "--output", str(output),
        ]
        result = subprocess.run(
            command, cwd=self.home, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        # Validate the real post-import phase before testing marker absence:
        # an unrelated import/context/network failure cannot satisfy this control.
        self.assertTrue(checkpoint.is_file(), "probe did not reach the bound run_live entry")
        reached = json.loads(checkpoint.read_bytes())
        self.assertEqual(reached["phase"], "BOUND_RUN_LIVE_ENTRY")
        self.assertEqual(reached["filename"], str(self.tool.resolve()))
        self.assertEqual(reached["root"], str(self.root.resolve()))
        self.assertEqual(reached["identity"]["checked_commit"], self.commit)
        self.assertEqual(reached["python_optimize"], sys.flags.optimize)
        report = self.refused(result, "FRESHNESS_CHECK_FAILED")
        self.assertEqual(json.loads(output.read_bytes()), report)
        self.assertEqual(report["checked_commit_before"], self.commit)
        self.assertEqual(report["checked_commit_after"], self.commit)
        self.assertEqual(set(report["source_bindings"]), set(SELECTED))
        self.assertEqual(report["source_bindings"], report["source_bindings_after"])
        self.assertNotIn("freshness", report)
        self.assertNotIn("source_observations", report)
        self.assertNotIn("fresh_checked", report)

    def test_unbound_package_cannot_replace_either_bound_project_dependency(self):
        for dependency in ("node_status_refresh", "validate_node_status"):
            with self.subTest(dependency=dependency):
                source = self.root / "tools" / (dependency + ".py")
                package = self.root / "tools" / dependency
                marker = self.home / (dependency + ".package-executed")
                package.mkdir()
                (package / "__init__.py").write_text(
                    self._unbound_dependency_wrapper(source, marker), encoding="utf-8",
                )
                try:
                    self._invoke_import_probe(dependency + "-package")
                    self.assertFalse(
                        marker.exists(),
                        "CLI executed an unbound package for " + dependency,
                    )
                finally:
                    shutil.rmtree(package)

    def test_timestamp_valid_bytecode_cannot_replace_either_bound_project_dependency(self):
        import importlib.util
        import marshal
        import struct

        for dependency in ("node_status_refresh", "validate_node_status"):
            with self.subTest(dependency=dependency):
                source = self.root / "tools" / (dependency + ".py")
                marker = self.home / (dependency + ".bytecode-executed")
                code = compile(
                    self._unbound_dependency_wrapper(source, marker), str(source),
                    "exec", dont_inherit=True, optimize=sys.flags.optimize,
                )
                metadata = source.stat()
                header = importlib.util.MAGIC_NUMBER + struct.pack(
                    "<III", 0, int(metadata.st_mtime) & 0xffffffff,
                    metadata.st_size & 0xffffffff,
                )
                cache = Path(importlib.util.cache_from_source(
                    str(source),
                    optimization="" if not sys.flags.optimize else str(sys.flags.optimize),
                ))
                cache.parent.mkdir(exist_ok=True)
                cache.write_bytes(header + marshal.dumps(code))
                try:
                    self._invoke_import_probe(dependency + "-bytecode")
                    self.assertFalse(
                        marker.exists(),
                        "CLI executed unbound timestamp-valid bytecode for " + dependency,
                    )
                finally:
                    shutil.rmtree(cache.parent)


if __name__ == "__main__":
    unittest.main()
