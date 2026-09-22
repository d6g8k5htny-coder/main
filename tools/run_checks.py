#!/usr/bin/env python3
"""Run every explicitly supported CI check and preserve a snapshot-bound receipt.

A PASS records execution only: no mathematical certification, claim promotion,
gate closure, organizational independence, authorization, or deployment follows.
This is a read-only orchestration aid, not an execution sandbox or an environment
lock. Run on a stable checkout; before/after hashes cannot detect change-and-revert.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import selectors
import signal
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/ci.yml"
MAX_LOG_BYTES = 8 * 1024 * 1024
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
SCOPE = ("Execution record only; no mathematical certification, status promotion, "
         "gate closure, independence credit, authorization, or deployment. "
         "Not an environment lock or a security sandbox; concurrent change-and-revert "
         "between input snapshots is not detected.")

# Deliberate duplication of the reviewed minimum set, not a second workflow.
# CI supplies order and names. Any addition/removal/refactor requires reviewing
# this allowlist as well; unfamiliar commands never silently disappear.
REQUIRED_COMMANDS = (
    "python tools/registers_import.py --check",
    "python tools/registers_check.py",
    "python tools/provenance_check.py",
    "python tools/claims_check.py",
    "python tools/quarantine_check.py",
    "python tools/verify_manifests.py",
    "python tools/manifest_integrity_check.py --coverage .github/manifest-coverage.json",
    "python tools/reviews_check.py",
    "python tools/recovery_check.py",
    "python tools/collision_proposal_check.py",
    "python tools/collision_proposal_check.py --proposal registers/collision_proposal_2026-09-19.json",
    "python tools/collision_proposal_check.py --proposal registers/collision_proposal_2026-09-19b.json",
    "python tools/consumers_check.py",
    "python tools/carriers_verify.py",
    "python tools/lanes_check.py",
    "python tools/slack_check.py",
    "python tools/receipts_check.py",
    "python tools/frozen_check.py",
    "python tools/bridge_check.py",
    "python tools/math_status_check.py",
    "python tools/operations_check.py",
    "python tools/mirror_quotes_check.py",
    "python tools/mirrors_index_check.py",
    "python tools/drive_index.py stats",
    "python tools/drive_coverage.py --json",
    "python tools/hermite_envelope_report.py --check research/bands/candidates/hermite_gaussian_20260919.json",
    "python tools/drive_reconcile.py",
    "python tools/native_export_check.py --verify-containers",
    "python tools/rn_moment_report.py --check research/rn/candidates/affine_moments_20260920.json",
    "python tools/rn_certificate.py check-candidates",
    "python tools/rn_side24_check.py",
    "python tools/rn_side24_density_check.py",
    "python tools/rn_side24_spatial_check.py",
    "python tools/parallel_math_check.py",
    "python tools/twelve_project_check.py",
    "python tools/h3_rn_n6_check.py",
    "python tools/rn_inner_wedge_check.py",
    "python tools/rn_bernstein_sharp_check.py",
    "python tools/research_frontier.py self-check",
    "python tools/lpw_amplitude_check.py",
    "python tools/closure_pipeline.py check-plan",
    "python -m pytest -q",
)
CI_PREFIX = '''name: ci
on:
  push:
  pull_request:
permissions:
  contents: read

defaults:
  run:
    shell: bash

jobs:
  verify:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          persist-credentials: false
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install test dependencies
        run: python -m pip install --upgrade pip pytest'''


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()


def digest_file(path: Path) -> dict:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    after = path.lstat()
    if (info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_mode):
        raise ValueError(f"file changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "bytes": size,
            "mode": stat.S_IMODE(info.st_mode)}


def artifact_identity(path: Path) -> dict:
    identity = digest_file(path)
    identity.pop("mode")  # artifacts become read-only after report publication
    return identity


def parse_workflow(text: str) -> list[dict]:
    """Accept only the checked-in header and simple name/run pairs, never YAML or shell."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    prefix = [line for line in CI_PREFIX.splitlines() if line.strip()]
    if lines[:len(prefix)] != prefix:
        raise ValueError("unsupported CI header/setup; review the restricted parser")
    lines = lines[len(prefix):]
    if not lines or len(lines) % 2:
        raise ValueError("empty or malformed CI check steps")
    if not REQUIRED_COMMANDS or len(set(REQUIRED_COMMANDS)) != len(REQUIRED_COMMANDS):
        raise ValueError("runner's required command policy is empty or duplicated")
    steps = []
    for number in range(0, len(lines), 2):
        name_line, command_line = lines[number:number + 2]
        if (not name_line.startswith("      - name: ") or
                not command_line.startswith("        run: ")):
            raise ValueError("unsupported CI step form; only a name and one literal run are allowed")
        name = name_line[len("      - name: "):]
        command = command_line[len("        run: "):]
        if not name or command not in REQUIRED_COMMANDS:
            raise ValueError(f"unsupported or dynamic CI command: {command!r}")
        steps.append({"name": name, "ci_command": command})
    actual = [step["ci_command"] for step in steps]
    if len(set(actual)) != len(actual):
        raise ValueError("duplicate CI command")
    missing = sorted(set(REQUIRED_COMMANDS) - set(actual))
    if missing:
        raise ValueError(f"missing required CI commands: {missing}")
    if actual[-1] != "python -m pytest -q":
        raise ValueError("the complete pytest suite must be the final CI check")
    return steps


def git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def snapshot(root: Path) -> dict:
    """Bind HEAD, index, Git dirtiness, and actual non-cache bytes, including ignored files."""
    if Path(os.fsdecode(git(root, "rev-parse", "--show-toplevel")).strip()).resolve() != root:
        raise ValueError("runner root is not the Git repository root")
    head = git(root, "rev-parse", "HEAD").decode().strip()
    tree = git(root, "rev-parse", "HEAD^{tree}").decode().strip()
    status = git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    index = git(root, "ls-files", "--stage", "-z")
    inputs = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_DIRS)
        for name in dirs:
            if (base / name).is_symlink():
                raise ValueError(f"source directory symlink: {base / name}")
        for name in sorted(files):
            path = base / name
            if name == ".git" or path.suffix in EXCLUDED_SUFFIXES:
                continue
            inputs[path.relative_to(root).as_posix()] = digest_file(path)
    if not inputs:
        raise ValueError("empty input snapshot")
    # Catch concurrent Git/index changes during the file pass as well.
    if (head != git(root, "rev-parse", "HEAD").decode().strip() or
            status != git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all") or
            index != git(root, "ls-files", "--stage", "-z")):
        raise ValueError("Git identity changed while taking snapshot")
    return {"commit": head, "tree": tree, "dirty": bool(status),
            "status_porcelain_z": os.fsdecode(status),
            "index_sha256": hashlib.sha256(index).hexdigest(),
            "input_count": len(inputs), "inputs_sha256": hashlib.sha256(canonical_bytes(inputs)).hexdigest(),
            "inputs": inputs}


def prepare_output(root: Path, requested: Path) -> Path:
    output = requested.absolute()
    for part in (output, *output.parents):
        if part.is_symlink():
            raise ValueError(f"output path contains a symlink: {part}")
    output = output.resolve()
    if output.is_relative_to(root):
        raise ValueError("output directory must be outside the repository's source inputs")
    if not output.parent.is_dir():
        raise ValueError("output parent must already exist")
    output.mkdir(mode=0o700)  # exclusive: even an empty existing directory is rejected
    return output


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)  # never replace an existing report
    temporary.unlink()


def child_environment(output: Path) -> dict:
    env = os.environ.copy()
    for key in ("PYTHONOPTIMIZE", "PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONPYCACHEPREFIX=str(output / "unused-pycache"))
    # Tests that invoke literal `python` must use this environment's interpreter.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def execute_step(root: Path, output: Path, index: int, step: dict,
                 timeout: float, env: dict) -> dict:
    argv = [sys.executable, *step["ci_command"].split()[1:]]
    if step["ci_command"] == "python -m pytest -q":
        argv += ["-p", "no:cacheprovider", "-o", "addopts=", f"--junitxml={output / 'pytest.xml'}"]
    log = output / f"{index:02d}.log"
    result = {**step, "argv": argv, "argv_sha256": hashlib.sha256(canonical_bytes(argv)).hexdigest(),
              "cwd": str(root), "start_utc": utc_now(), "timeout_seconds": timeout,
              "returncode": None, "timed_out": False, "log_limit_exceeded": False,
              "log": log.name, "output_bytes_observed": 0}
    started = time.monotonic()
    process = None
    try:
        with log.open("xb") as stream, selectors.DefaultSelector() as selector:
            process = subprocess.Popen(argv, cwd=root, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            selector.register(process.stdout, selectors.EVENT_READ)
            killed_at = None
            while selector.get_map() or process.poll() is None:
                now = time.monotonic()
                if killed_at is None and now - started >= timeout:
                    result["timed_out"] = True
                    kill_group(process)
                    killed_at = now
                if killed_at is not None and now - killed_at > 5:
                    raise RuntimeError("process pipes did not close after termination")
                for key, _ in selector.select(timeout=0.05):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    remaining = max(0, MAX_LOG_BYTES - stream.tell())
                    stream.write(data[:remaining])
                    result["output_bytes_observed"] += len(data)
                    if len(data) > remaining and killed_at is None:
                        result["log_limit_exceeded"] = True
                        kill_group(process)
                        killed_at = time.monotonic()
            result["returncode"] = process.wait(timeout=5)
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if process is not None:
            kill_group(process)
            result["returncode"] = process.wait(timeout=5)
    finally:
        if process is not None and process.stdout is not None:
            process.stdout.close()
    result["end_utc"] = utc_now()
    result["elapsed_seconds"] = time.monotonic() - started
    if log.exists():
        result["log_identity"] = artifact_identity(log)
    result["passed"] = (result["returncode"] == 0 and not result["timed_out"] and
                        not result["log_limit_exceeded"] and "error" not in result)
    return result


def pytest_results(path: Path) -> dict:
    identity = artifact_identity(path)
    if identity["bytes"] > 16 * 1024 * 1024:
        raise ValueError("pytest JUnit report exceeds 16 MiB")
    document = ET.fromstring(path.read_bytes())
    if document.tag != "testsuites" or not list(document):
        raise ValueError("missing pytest JUnit testsuites")
    cases = document.findall(".//testcase")
    counts = {"testcases": len(cases), "skipped": 0, "errors": 0, "failures": 0}
    for case in cases:
        for tag, field in (("skipped", "skipped"), ("error", "errors"), ("failure", "failures")):
            if case.find(tag) is not None:
                counts[field] += 1
    counts["executed"] = counts["testcases"] - counts["skipped"] - counts["errors"]
    declared = {key: 0 for key in ("tests", "skipped", "errors", "failures")}
    for suite in document:
        if suite.tag != "testsuite":
            raise ValueError("unexpected pytest JUnit element")
        for key in declared:
            value = int(suite.attrib[key])
            if value < 0:
                raise ValueError("negative pytest JUnit count")
            declared[key] += value
    if declared != {"tests": counts["testcases"], **{key: counts[key] for key in ("skipped", "errors", "failures")}}:
        raise ValueError("pytest JUnit counts disagree with testcase outcomes")
    return {**counts, "file": path.name, "identity": identity}


def validate_artifact_bindings(report: dict) -> None:
    """Require a complete artifact map agreeing with each execution-time identity.

    The caller must hash/read back the actual files as well. This cross-check
    prevents a later step from rewriting an earlier log and silently replacing
    its digest in the final aggregate artifact map.
    """
    artifacts = report["artifacts"]
    required = set()
    ran_pytest = False
    for step in report["steps"]:
        name = step.get("log")
        if (not isinstance(name, str) or not name or name in (".", "..") or
                Path(name).name != name or name in required):
            raise ValueError("invalid or duplicate step log name")
        required.add(name)
        if not step.get("log_identity") or artifacts.get(name) != step["log_identity"]:
            raise ValueError(f"step log identity changed or omitted: {name}")
        ran_pytest |= step["ci_command"] == "python -m pytest -q"
    summary = report.get("pytest")
    if ran_pytest:
        required.add("pytest.xml")
        if summary is not None and (summary.get("file") != "pytest.xml" or
                                    summary.get("identity") != artifacts.get("pytest.xml")):
            raise ValueError("pytest artifact identity changed or omitted")
        if summary is None and report["steps"][-1].get("passed"):
            raise ValueError("successful pytest step has no parsed result identity")
    elif summary is not None:
        raise ValueError("pytest summary exists without pytest execution")
    if set(artifacts) != required:
        raise ValueError("artifact set does not exactly cover executed step logs and pytest output")


def run_checks(root: Path, requested_output: Path, timeout_seconds: float = 1800,
               budget_seconds: float = 1800) -> dict:
    root = root.resolve()
    output = prepare_output(root, requested_output)
    started = time.monotonic()
    report = {"schema_version": 1, "status": "FAIL", "start_utc": utc_now(),
              "scope": SCOPE, "root": str(root), "output_directory": str(output),
              "budget_seconds": budget_seconds if math.isfinite(budget_seconds) else str(budget_seconds),
              "per_step_timeout_seconds": timeout_seconds if math.isfinite(timeout_seconds) else str(timeout_seconds),
              "failures": [], "steps": [], "before": None, "after": None,
              "snapshot_exclusions": {"directory_names": sorted(EXCLUDED_DIRS),
                                      "file_suffixes": sorted(EXCLUDED_SUFFIXES),
                                      "git_worktree_pointer": ".git"},
              "environment": {"python_executable": sys.executable, "python_version": sys.version,
                              "implementation": platform.python_implementation(),
                              "platform": platform.platform(), "optimize": sys.flags.optimize,
                              "PYTHONOPTIMIZE": os.environ.get("PYTHONOPTIMIZE"),
                              "lock_claimed": False},
              "child_environment_policy": {
                  "python_path": "interpreter directory prepended to PATH",
                  "cleared": ["PYTHONOPTIMIZE", "PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"],
                  "bytecode": "disabled; external empty pycache prefix",
                  "pytest": "plugin autoload and cache disabled; config addopts overridden"}}
    try:
        if os.name != "posix":
            raise ValueError("runner requires POSIX process-group termination")
        if sys.version_info[:2] != (3, 11):
            raise ValueError("project checks require Python 3.11")
        if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE"):
            raise ValueError("optimized interpreter or PYTHONOPTIMIZE environment is forbidden")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout must be finite and positive")
        if not math.isfinite(budget_seconds) or budget_seconds <= 0:
            raise ValueError("budget must be finite and positive")
        report["environment"]["pytest_version"] = importlib.metadata.version("pytest")
        report["environment"]["python_binary"] = digest_file(Path(sys.executable).resolve())
        report["before"] = snapshot(root)
        workflow = root / WORKFLOW
        report["workflow"] = {"path": WORKFLOW, **digest_file(workflow)}
        steps = parse_workflow(workflow.read_text())
        report["required_commands"] = list(REQUIRED_COMMANDS)
        report["planned_steps"] = steps
        env = child_environment(output)
        for index, step in enumerate(steps, start=1):
            remaining = budget_seconds - (time.monotonic() - started)
            if remaining <= 0:
                report["failures"].append("run budget exhausted before every required check executed")
                break
            result = execute_step(root, output, index, step, min(timeout_seconds, remaining), env)
            result["budget_remaining_at_start"] = remaining
            result["budget_exhausted"] = result["timed_out"] and remaining <= timeout_seconds
            report["steps"].append(result)
            print(f"{index}/{len(steps)} {'PASS' if result['passed'] else 'FAIL'} {step['ci_command']}", flush=True)
            if not result["passed"]:
                report["failures"].append(f"step {index} failed: {step['ci_command']}")
                break
        if len(report["steps"]) != len(steps):
            report["failures"].append("not every required check executed")
        if report["steps"] and report["steps"][-1]["ci_command"] == "python -m pytest -q":
            report["pytest"] = pytest_results(output / "pytest.xml")
            counts = report["pytest"]
            if counts["executed"] <= 0 or counts["errors"] or counts["failures"]:
                report["failures"].append(f"pytest must execute tests without failures or errors: {counts}")
        else:
            report["failures"].append("complete pytest suite did not execute")
    except (Exception, KeyboardInterrupt) as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if report["before"] is not None:
            try:
                report["after"] = snapshot(root)
                if report["after"] != report["before"]:
                    report["failures"].append("repository identity or input bytes changed during checks")
            except (Exception, KeyboardInterrupt) as exc:
                report["failures"].append(f"final snapshot failed: {type(exc).__name__}: {exc}")
        report["artifacts"] = {}
        try:
            for path in output.iterdir():
                report["artifacts"][path.name] = artifact_identity(path)
            validate_artifact_bindings(report)
        except (Exception, KeyboardInterrupt) as exc:
            report["failures"].append(f"artifact binding failed: {type(exc).__name__}: {exc}")
        if (not report["failures"] and report["before"] is not None and
                report["after"] == report["before"] and report.get("pytest", {}).get("executed", 0) > 0):
            report["status"] = "PASS"
        report["end_utc"] = utc_now()
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output / "report.json", report)
        for path in output.iterdir():
            if path.is_file() and not path.is_symlink():
                path.chmod(0o444)
        output.chmod(0o555)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new, absent directory outside the repository; parent must exist")
    parser.add_argument("--timeout-seconds", type=float, default=1800,
                        help="per-check timeout (default: 1800 seconds)")
    parser.add_argument("--budget-seconds", type=float, default=1800,
                        help="total check budget; final identity capture still runs (default: 1800 seconds)")
    args = parser.parse_args(argv)
    try:
        report = run_checks(ROOT, args.output_dir, args.timeout_seconds, args.budget_seconds)
    except (OSError, ValueError) as exc:
        print(f"FAIL run_checks: {exc}", file=sys.stderr)
        return 2
    print(f"{report['status']} run_checks: {len(report['steps'])} steps; {args.output_dir / 'report.json'}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
