"""Isolated runner negative controls, never recursive runs of the research suite.

The fixtures substitute trivial checker files and tiny real pytest suites. These
tests establish orchestration behavior, not the correctness of any research claim.
"""
from datetime import datetime
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verification_runner", ROOT / "tools/run_checks.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def workflow(commands=None):
    commands = runner.REQUIRED_COMMANDS if commands is None else commands
    return runner.CI_PREFIX + "\n" + "".join(
        f"      - name: Fixture check {number}\n        run: {command}\n"
        for number, command in enumerate(commands))


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".github/workflows").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "tests").mkdir()
    (root / runner.WORKFLOW).write_text(workflow())
    (root / ".gitignore").write_text("ignored-input.dat\n__pycache__/\n.pytest_cache/\n")
    for command in runner.REQUIRED_COMMANDS:
        if command.startswith("python tools/"):
            (root / command.split()[1]).write_text("print('fixture checker executed')\n")
    shutil.copyfile(ROOT / "tools/run_checks.py", root / "tools/run_checks.py")
    (root / "tests/test_tiny.py").write_text(
        "import pathlib, subprocess, sys\n"
        "def test_assertions_are_enabled():\n"
        "    assert __debug__\n"
        "def test_literal_python_uses_current_interpreter():\n"
        "    actual = subprocess.check_output(['python', '-c', 'import sys; print(sys.executable)'], text=True).strip()\n"
        "    assert pathlib.Path(actual).resolve() == pathlib.Path(sys.executable).resolve()\n")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
    return root


def invoke(root, output, *args, python_flags=(), env=None):
    child_env = os.environ.copy()
    child_env.update(PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    if env:
        child_env.update(env)
    result = subprocess.run([sys.executable, *python_flags, str(root / "tools/run_checks.py"),
                             "--output-dir", str(output), *args], cwd=root.parent,
                            env=child_env, capture_output=True, text=True, timeout=30)
    path = output / "report.json"
    return result, json.loads(path.read_text()) if path.is_file() else None


def first_checker(root):
    return root / runner.REQUIRED_COMMANDS[0].split()[1]


def test_current_ci_is_exactly_covered():
    steps = runner.parse_workflow((ROOT / runner.WORKFLOW).read_text())
    assert len(steps) >= 25  # all 24 original check commands plus pytest
    assert {step["ci_command"] for step in steps} == set(runner.REQUIRED_COMMANDS)


@pytest.mark.parametrize("mutation", ["empty", "omit_checker", "omit_pytest", "duplicate", "shell",
                                     "dynamic", "multiline", "conditional", "environment", "header"])
def test_workflow_parser_fails_closed(mutation):
    source = workflow()
    if mutation == "empty":
        source = runner.CI_PREFIX
    elif mutation == "omit_checker":
        source = workflow(runner.REQUIRED_COMMANDS[1:])
    elif mutation == "omit_pytest":
        source = workflow(runner.REQUIRED_COMMANDS[:-1])
    elif mutation == "duplicate":
        source = workflow((*runner.REQUIRED_COMMANDS[:-1], runner.REQUIRED_COMMANDS[0], runner.REQUIRED_COMMANDS[-1]))
    elif mutation == "shell":
        source = source.replace(runner.REQUIRED_COMMANDS[0], runner.REQUIRED_COMMANDS[0] + " || true")
    elif mutation == "dynamic":
        source = source.replace(runner.REQUIRED_COMMANDS[0], "${{ inputs.check }}")
    elif mutation == "multiline":
        source = source.replace("run: " + runner.REQUIRED_COMMANDS[0], "run: |\n          " + runner.REQUIRED_COMMANDS[0])
    elif mutation == "conditional":
        source = source.replace("        run: ", "        continue-on-error: true\n        run: ", 1)
    elif mutation == "environment":
        source = source.replace("        run: ", "        env: {PYTHONOPTIMIZE: 1}\n        run: ", 1)
    elif mutation == "header":
        source = source.replace('python-version: "3.11"', 'python-version: "3.12"')
    with pytest.raises(ValueError):
        runner.parse_workflow(source)


def test_complete_run_binds_inputs_logs_environment_and_real_test_counts(repository, tmp_path):
    (repository / "ignored-input.dat").write_text("ignored but still an input\n")
    output = tmp_path / "complete"
    result, report = invoke(repository, output, env={"PYTEST_ADDOPTS": "--collect-only"})
    assert result.returncode == 0, (result.stdout, result.stderr, report)
    assert report["status"] == "PASS"
    assert report["before"] == report["after"]
    assert report["before"]["dirty"] is False
    assert "ignored-input.dat" in report["before"]["inputs"]
    assert report["before"]["commit"] == git(repository, "rev-parse", "HEAD").strip()
    assert report["before"]["tree"] == git(repository, "rev-parse", "HEAD^{tree}").strip()
    assert report["pytest"]["executed"] == 2
    assert report["environment"]["pytest_version"]
    assert report["environment"]["python_version"].startswith("3.11.")
    assert report["environment"]["lock_claimed"] is False
    assert len(report["steps"]) == len(runner.REQUIRED_COMMANDS)
    assert datetime.fromisoformat(report["start_utc"]) <= datetime.fromisoformat(report["end_utc"])
    for step in report["steps"]:
        assert step["argv"][0] == sys.executable
        assert step["cwd"] == str(repository)
        assert step["returncode"] == 0 and step["passed"]
        assert datetime.fromisoformat(step["start_utc"]) <= datetime.fromisoformat(step["end_utc"])
        assert step["argv_sha256"] == hashlib.sha256(runner.canonical_bytes(step["argv"])).hexdigest()
        assert step["log_identity"]["sha256"] == hashlib.sha256((output / step["log"]).read_bytes()).hexdigest()
    assert not list(repository.rglob("__pycache__"))
    assert not (repository / ".pytest_cache").exists()
    assert not (output.stat().st_mode & 0o222)
    assert not ((output / "report.json").stat().st_mode & 0o222)
    original = (output / "report.json").read_bytes()
    rerun, _ = invoke(repository, output)
    assert rerun.returncode != 0
    assert (output / "report.json").read_bytes() == original


@pytest.mark.parametrize("failure", ["exit", "missing", "malformed_workflow"])
def test_failure_is_preserved_with_no_stale_pass(repository, tmp_path, failure):
    if failure == "exit":
        first_checker(repository).write_text("print('specific checker failure'); raise SystemExit(7)\n")
    elif failure == "missing":
        first_checker(repository).unlink()
    else:
        (repository / runner.WORKFLOW).write_text(workflow(runner.REQUIRED_COMMANDS[1:]))
    result, report = invoke(repository, tmp_path / failure)
    assert result.returncode == 1 and report["status"] == "FAIL"
    assert report["before"] == report["after"]
    assert report["failures"]
    if failure == "malformed_workflow":
        assert report["steps"] == []
    else:
        assert len(report["steps"]) == 1
        assert report["steps"][0]["returncode"] == (7 if failure == "exit" else 2)
        assert (tmp_path / failure / "01.log").read_text()


@pytest.mark.parametrize("test_source,expected_count,expected_failure", [
    ("# no tests\n", 0, 0),
    ("import pytest\ndef test_skip():\n    pytest.skip('all skipped')\n", 0, 0),
    ("def test_failure():\n    assert 2 + 2 == 5\n", 1, 1),
])
def test_zero_execution_and_real_pytest_failure_cannot_pass(repository, tmp_path, test_source,
                                                          expected_count, expected_failure):
    (repository / "tests/test_tiny.py").write_text(test_source)
    result, report = invoke(repository, tmp_path / "pytest-failure")
    assert result.returncode == 1 and report["status"] == "FAIL"
    assert report["pytest"]["executed"] == expected_count
    assert report["pytest"]["failures"] == expected_failure
    assert "pytest.xml" in report["artifacts"]
    if expected_failure:
        assert "assert (2 + 2) == 5" in (tmp_path / "pytest-failure" / report["steps"][-1]["log"]).read_text()


def test_timeout_kills_step_and_retains_output(repository, tmp_path):
    first_checker(repository).write_text("import time\nprint('before timeout', flush=True)\ntime.sleep(30)\n")
    result, report = invoke(repository, tmp_path / "timeout", "--timeout-seconds", "0.25")
    assert result.returncode == 1
    step = report["steps"][0]
    assert step["timed_out"] and step["returncode"] != 0
    assert not step["passed"]
    assert step["elapsed_seconds"] < 5
    assert "before timeout" in (tmp_path / "timeout/01.log").read_text()


def test_exhausted_total_budget_cannot_pass(repository, tmp_path):
    result, report = invoke(repository, tmp_path / "budget", "--budget-seconds", "0.000001")
    assert result.returncode == 1 and report["status"] == "FAIL"
    assert report["steps"] == []
    assert any("budget exhausted" in reason for reason in report["failures"])


def test_total_budget_bounds_a_running_step(repository, tmp_path):
    first_checker(repository).write_text("import time\ntime.sleep(30)\n")
    result, report = invoke(repository, tmp_path / "running-budget", "--timeout-seconds", "20",
                            "--budget-seconds", "0.4")
    assert result.returncode == 1 and report["status"] == "FAIL"
    assert report["steps"][0]["timed_out"]
    assert report["steps"][0]["budget_exhausted"]
    assert report["steps"][0]["timeout_seconds"] < 0.4


@pytest.mark.parametrize("flag,value", [("--timeout-seconds", "nan"), ("--budget-seconds", "inf"),
                                      ("--timeout-seconds", "0"), ("--budget-seconds", "-1")])
def test_invalid_time_limits_produce_valid_failed_receipts(repository, tmp_path, flag, value):
    result, report = invoke(repository, tmp_path / "invalid-time", flag, value)
    assert result.returncode == 1 and report["status"] == "FAIL"
    assert report["steps"] == []
    json.loads((tmp_path / "invalid-time/report.json").read_text(),
               parse_constant=lambda value: pytest.fail(f"nonstandard JSON {value}"))


@pytest.mark.parametrize("path", ["new-input.dat", "ignored-input.dat", "tests/test_tiny.py"])
def test_any_input_mutation_prevents_pass(repository, tmp_path, path):
    first_checker(repository).write_text(
        f"from pathlib import Path\np=Path({path!r})\np.write_text(p.read_text()+'\\n# mutation\\n' if p.exists() else 'mutation\\n')\n")
    result, report = invoke(repository, tmp_path / "changed")
    assert result.returncode == 1 and report["status"] == "FAIL"
    assert all(step["passed"] for step in report["steps"])
    assert report["before"]["inputs_sha256"] != report["after"]["inputs_sha256"]
    assert any("changed during checks" in reason for reason in report["failures"])


@pytest.mark.parametrize("flags,env", [(("-O",), {}), ((), {"PYTHONOPTIMIZE": "1"})])
def test_optimized_execution_is_rejected_before_checks(repository, tmp_path, flags, env):
    result, report = invoke(repository, tmp_path / "optimized", python_flags=flags, env=env)
    assert result.returncode == 1 and report["status"] == "FAIL"
    assert report["steps"] == []
    assert any("optimized interpreter" in reason for reason in report["failures"])


@pytest.mark.parametrize("placement", ["inside", "symlink", "existing"])
def test_unsafe_or_reused_output_directory_is_rejected(repository, tmp_path, placement):
    if placement == "inside":
        output = repository / "report-output"
    elif placement == "symlink":
        link = tmp_path / "link"
        link.symlink_to(repository, target_is_directory=True)
        output = link / "report-output"
    else:
        output = tmp_path / "already-there"
        output.mkdir()
    result, report = invoke(repository, output)
    assert result.returncode == 2 and report is None


def test_source_symlink_is_rejected(repository, tmp_path):
    (repository / "untracked-link").symlink_to(repository / "tests/test_tiny.py")
    result, report = invoke(repository, tmp_path / "symlink-input")
    assert result.returncode == 1 and report["status"] == "FAIL"
    assert report["steps"] == []


def test_excessive_checker_output_is_bounded_and_fails(repository, tmp_path):
    first_checker(repository).write_text("import sys\nsys.stdout.write('x' * (9 * 1024 * 1024))\n")
    result, report = invoke(repository, tmp_path / "large-log")
    assert result.returncode == 1 and report["status"] == "FAIL"
    step = report["steps"][0]
    assert step["log_limit_exceeded"]
    assert (tmp_path / "large-log/01.log").stat().st_size == runner.MAX_LOG_BYTES


def test_forged_junit_summary_cannot_hide_missing_cases(tmp_path):
    path = tmp_path / "forged.xml"
    path.write_text('<testsuites><testsuite tests="1" skipped="0" errors="0" failures="0"/></testsuites>')
    with pytest.raises(ValueError, match="disagree"):
        runner.pytest_results(path)


def test_later_checker_cannot_rewrite_an_earlier_log_and_pass(repository, tmp_path):
    checker = repository / runner.REQUIRED_COMMANDS[1].split()[1]
    checker.write_text(
        "import os\nfrom pathlib import Path\n"
        "output = Path(os.environ['PYTHONPYCACHEPREFIX']).parent\n"
        "(output / '01.log').write_text('earlier evidence overwritten\\n')\n")
    result, report = invoke(repository, tmp_path / "overwritten-log")
    assert result.returncode == 1 and report["status"] == "FAIL"
    assert all(step["passed"] for step in report["steps"])
    assert report["pytest"]["executed"] == 2
    assert report["before"] == report["after"]
    assert report["steps"][0]["log_identity"] != report["artifacts"]["01.log"]
    assert any("step log identity changed" in reason for reason in report["failures"])


@pytest.mark.parametrize("mutation", [None, "missing_log", "step_hash", "missing_junit", "junit_hash",
                                     "extra_artifact", "duplicate_log", "missing_junit_result"])
def test_complete_artifact_binding_cannot_omit_or_replace_evidence(mutation):
    identity = {"sha256": hashlib.sha256(b"output").hexdigest(), "bytes": 6}
    report = {"steps": [
        {"ci_command": runner.REQUIRED_COMMANDS[0], "log": "01.log", "log_identity": copy.deepcopy(identity), "passed": True},
        {"ci_command": "python -m pytest -q", "log": "02.log", "log_identity": copy.deepcopy(identity), "passed": True}],
        "pytest": {"file": "pytest.xml", "identity": copy.deepcopy(identity), "executed": 1},
        "artifacts": {name: copy.deepcopy(identity) for name in ("01.log", "02.log", "pytest.xml")}}
    if mutation == "missing_log":
        report["artifacts"].pop("01.log")
    elif mutation == "step_hash":
        report["steps"][0]["log_identity"]["sha256"] = "f" * 64
    elif mutation == "missing_junit":
        report["artifacts"].pop("pytest.xml")
    elif mutation == "junit_hash":
        report["pytest"]["identity"]["sha256"] = "f" * 64
    elif mutation == "extra_artifact":
        report["artifacts"]["extra.log"] = identity
    elif mutation == "duplicate_log":
        report["steps"].append(report["steps"][0])
    elif mutation == "missing_junit_result":
        report.pop("pytest")
    if mutation is None:
        runner.validate_artifact_bindings(report)
    else:
        with pytest.raises(ValueError):
            runner.validate_artifact_bindings(report)
