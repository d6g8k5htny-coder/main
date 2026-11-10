#!/usr/bin/env python3
"""Explicit Task A+B+C+D hosted contracts; Task D additions remain test first.

Fifteen unittest methods declare 67 subcases: 61 Lean and 6 Python_controller.
Separate output admission has 2 methods/6 routes outside that census.
Original Task A remains six/39. Absence, compile/setup failure and unrun subcases
are evidence categories, never semantic rejections or passing coverage.
Run explicitly; this file is outside ordinary offline test_*.py discovery.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import ctypes
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sdga_component"
BRIDGE = ROOT / "tools" / "sdga" / "Component.lean"
CONTROLLER = ROOT / "tools" / "sdga_component.py"
RELEASE = "5045d0056413266e57c625dcd7c365b10e377c52"
DESIGN_BLOB = "3487f54476b5e6a283839902797bc05aae2ce623"
PLAN_BLOB = "9eefcb144247aa3f358b735956e580777e2b8945"
CASES = {
    "C01": ("anon-roundtrip", "anon-identity-reject", "numeric-vs-string",
            "empty-component", "dot-component", "slash-component",
            "unicode-normalization-distinct", "nonbmp", "large-nat",
            "uppercase-hex-reject", "leading-zero-reject", "invalid-utf8-reject"),
    "C02": ("raw-tables", "foreign-owner", "explicit-private", "module-private",
            "generated-present", "dependency-excluded", "final-rows-owner-census"),
    "C03": ("proof-def-prop", "prop-valued-data", "nat-data", "malformed-type-block"),
    "C12": ("table-length", "table-identity", "duplicate-name",
            "missing-final-row", "earlier-subset"),
    "C13": ("missing-server", "reordered-parts", "truncated-private"),
    "C15": ("wrong-resolved-path", "unknown-mode", "unknown-field",
            "malformed-transcript", "deps-json-error-row", "duplicate-vertex",
            "missing-local-edge", "cycle"),
}
TASK_A_CASES = dict(CASES)
CASES.update({
    "C04": ("type-route", "value-route", "opaque-route"),
    "C05": ("missing-type", "missing-value", "missing-link"),
    "C06": ("imports-empty", "exact-owned-propext", "owned-axiom-block"),
    "C07": ("unsafe-info", "partial-info", "unsafe-replay-skip", "partial-replay-skip"),
})
TASK_B_FIXTURES = ("BadDependency", "AxiomUse", "Skipped", "OwnedAxiom")
TASK_B_CASES = {case: names for case, names in CASES.items() if case not in TASK_A_CASES}
CASES.update({
    "C08": ("clean-base-absence", "full-safe-replay"),
    "C09": ("actual-source-ci", "singleton-resolved-refs", "clean-base-absence",
            "kernel-type-rejection"),
    "C10": ("good-base-setup", "singleton-resolved-refs", "duplicate-base-guard"),
})
CASES.update({
    "C11": ("raw-union-preserved", "raw-extra-block"),
    "C14": ("actual-ir-binding", "raw-names-preserved", "ci-backed-accounted",
            "residue-retained-blocked"),
})
PURE_C15 = {"malformed-transcript", "deps-json-error-row", "duplicate-vertex",
            "missing-local-edge", "cycle"}
STATE = None

# This fixed process exercise is cleanup evidence only, outside all Lean/control cases.
CLEANUP_CHILD = (
    "import os, signal, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "print('READY', os.getpid(), os.getpgrp(), flush=True)\n"
    "time.sleep(60)\n"
)
CLEANUP_PARENT = (
    "import json, os, signal, subprocess, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "child = subprocess.Popen([sys.executable, '-B', '-S', '-c', "
    + repr(CLEANUP_CHILD) + "], stdout=subprocess.PIPE, text=True)\n"
    "ready = child.stdout.readline().strip().split()\n"
    "if len(ready) != 3 or ready[0] != 'READY':\n"
    "    raise RuntimeError('descendant readiness handshake failed')\n"
    "print(json.dumps({'record': 'cleanup-ready', 'parent_pid': os.getpid(), "
    "'parent_pgid': os.getpgrp(), 'descendant_pid': int(ready[1]), "
    "'descendant_pgid': int(ready[2])}), flush=True)\n"
    "time.sleep(60)\n"
)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                               indent=2) + "\n", encoding="utf-8")


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("no module loader: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def measure_objects(paths):
    """Measure actual bytes; only FileNotFoundError denotes an absent path."""
    rows = {}
    for path in sorted({Path(path).resolve() for path in paths}, key=str):
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            rows[str(path)] = {"exists": False}
        else:
            rows[str(path)] = {"exists": True, "bytes": len(raw), "sha256": digest(raw)}
    return rows


@contextlib.contextmanager
def object_custody(paths, expected_manifest, evidence_path):
    """Check immutable expectations before/after, preserving any original body error."""
    paths = tuple(paths)
    expected = copy.deepcopy(expected_manifest)
    evidence_path = Path(evidence_path)
    if evidence_path.exists():
        raise ValueError("CUSTODY_EVIDENCE_REUSE: " + str(evidence_path))
    before = after = None
    pending = None
    entered = False
    record = {"record": "object-custody", "expected": expected,
              "before": None, "after": None, "drift": [], "status": "fail",
              "precheck_error": None, "observation_error": None,
              "measurement_errors": []}
    try:
        try:
            before = measure_objects(paths)
        except BaseException as exc:
            record["measurement_errors"].append("before: " + type(exc).__name__ + ": " + str(exc))
            raise
        if before != expected:
            raise RuntimeError("OBJECT_DRIFT: before observation")
        entered = True
        yield
    except BaseException as exc:
        pending = exc
        key = "observation_error" if entered else "precheck_error"
        record[key] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        try:
            after = measure_objects(paths)
        except BaseException as exc:
            record["measurement_errors"].append("after: " + type(exc).__name__ + ": " + str(exc))
        names = set(expected)
        for measured in (before, after):
            if measured is not None:
                names.update(measured)
        drift = sorted(name for name in names if any(
            measured is not None and measured.get(name) != expected.get(name)
            for measured in (before, after)))
        record.update(before=before, after=after, drift=drift,
                      status=("pass" if not drift and pending is None
                              and not record["measurement_errors"] else "fail"))
        try:
            json_write(evidence_path, record)
        except BaseException as write_error:
            if pending is not None:
                raise pending from write_error
            raise
        # Preserve the first body/precheck exception; custody failure is independently retained.
        if pending is None:
            if record["measurement_errors"]:
                raise RuntimeError("OBJECT_MEASUREMENT: after observation")
            if drift:
                raise RuntimeError("OBJECT_DRIFT: after observation")


class Evidence:
    def __init__(self, output):
        if output.exists():
            raise ValueError("OUTPUT_NOT_FRESH: " + str(output))
        output.mkdir(parents=True)
        self.output = output.resolve()
        self.owned = self.output / "owned"
        self.owned.mkdir()
        self.native = self.output / "native"
        self.native.mkdir()
        self.commands = []
        self.custody = self.output / "object-custody"
        self.custody.mkdir()
        self.object_expected = {}
        self.request_objects = {}
        self.request_bindings = {}
        self.object_observations = []
        self.object_failures = []
        self.object_final = None
        self.scheduling_blocked = False
        self.subreaper = None
        self.cleanup_preflight = None
        self.records = {}
        self.bootstrap_errors = []
        self.task_b_errors = {}
        self.task_b_dependency_request = self.output / "task-b-BadDependency-request.json"
        self.cli_replay = {
            "record": "supplemental-cli-replay", "probe_id": "cli-safe-replay",
            "case_id": "C08", "executor": "Lean_cli", "status": "not_run",
            "outside_declared_method_and_subcase_census": True,
            "command": None, "stdout_hex": None, "stderr_hex": None, "receipt": None,
        }
        self.inventory = None
        self.inventory_text = None
        self.request = None
        self.request_path = self.output / "request.json"
        self.scope = None
        self.controller = None
        self.controller_error = None
        self.task1 = None
        self.identities = {
            "record": "source-identities", "observed_utc": utc(),
            "scientific_effect": "NONE", "qualification": "BLOCKED",
            "classification_status": "BLOCKED", "execution_authenticated": False,
            "design_blob": DESIGN_BLOB, "plan_blob": PLAN_BLOB,
            "expected_lean_release": RELEASE,
            "python": sys.version, "python_optimized": bool(sys.flags.optimize),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "sources": {}, "absent_production": [],
        }
        for case_id, names in CASES.items():
            for subcase in names:
                executor = ("Python_controller" if
                            (case_id == "C15" and subcase in PURE_C15)
                            or (case_id == "C06" and subcase == "owned-axiom-block")
                            else "Lean")
                self.records[(case_id, subcase)] = {
                    "record": "subcase", "case_id": case_id,
                    "subcase_id": subcase, "executor": executor,
                    "status": "not_run", "reason": "not yet invoked",
                }
        self.capture_sources()
        self.flush()

    def capture_sources(self):
        paths = [Path(__file__).resolve(), BRIDGE, CONTROLLER,
                 ROOT / "tests" / "sdga_object_custody_contract.py",
                 ROOT / "tests" / "sdga_ir_evidence_contract.py",
                 ROOT / "tools" / "sdga_ir_evidence.py",
                 ROOT / "tools" / "sdga_scope.py",
                 ROOT / ".github" / "workflows" / "sdga-component-contract.yml",
                 ROOT / "tests" / "fixtures" / "sdga_scope" / "scope-f532.json"]
        paths.extend(sorted(p for p in FIXTURE.rglob("*") if p.is_file()
                            and ".lake" not in p.parts))
        for path in paths:
            relative = str(path.relative_to(ROOT))
            if not path.is_file():
                self.identities["sources"][relative] = {"exists": False}
                if path in (BRIDGE, CONTROLLER):
                    self.identities["absent_production"].append(relative)
                continue
            raw = path.read_bytes()
            self.identities["sources"][relative] = {
                "exists": True, "bytes": len(raw), "sha256": digest(raw),
                "git_blob": hashlib.sha1(
                    b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest(),
            }
        self.identities["declared_cases"] = {
            case: list(names) for case, names in CASES.items()}
        self.identities["original_task_a_methods"] = 6
        self.identities["original_task_a_subcases"] = 39
        self.identities["original_task_a_cases"] = {k: list(v) for k, v in TASK_A_CASES.items()}
        self.identities["task_b_fixed_fixtures"] = list(TASK_B_FIXTURES)
        self.identities["declared_python_methods"] = 15
        self.identities["declared_subcases"] = 67
        self.identities["task_b_methods"] = 4
        self.identities["task_b_subcases"] = 13
        self.identities["task_c_methods"] = 3
        self.identities["task_c_subcases"] = 9
        self.identities["lean_subcases"] = 61
        self.identities["python_controller_subcases"] = 6
        self.identities["separate_python_custody_regression_included"] = False
        profile = self.identities["sources"]["tests/fixtures/sdga_scope/scope-f532.json"]
        self.identities["fixed_profile_matches"] = (
            profile.get("git_blob") == "4c6b4ea59eb07a17ebfed536b07688c53c01eb64")

    def enable_subreaper(self):
        if sys.platform != "linux":
            raise RuntimeError("PROCESS_CLEANUP_UNAVAILABLE: Linux subreaper required")
        libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER")
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")
        current = ctypes.c_int()
        if libc.prctl(37, ctypes.byref(current), 0, 0, 0) != 0 or current.value != 1:
            raise RuntimeError("PROCESS_CLEANUP_UNAVAILABLE: subreaper not enabled")
        self.subreaper = {"previous": previous.value, "active": current.value}
        self.identities["process_subreaper"] = self.subreaper

    def cleanup_group(self, process):
        pgid = process.pid
        cleanup = {"pgid": pgid, "signals": [], "reaped_descendants": [],
                   "leader_reaped": False, "group_absent": False,
                   "confirmed": False, "errors": []}
        if pgid <= 1 or pgid == os.getpgrp():
            cleanup["errors"].append("owned group identity guard failed")
            return cleanup

        def exists():
            try:
                os.killpg(pgid, 0)
                return True
            except ProcessLookupError:
                return False

        def settle(seconds):
            deadline = time.monotonic() + seconds
            while True:
                # Popen consumes the direct leader before scoped waitpid can run.
                cleanup["leader_reaped"] = process.poll() is not None
                if cleanup["leader_reaped"]:
                    while True:
                        try:
                            pid, status = os.waitpid(-pgid, os.WNOHANG)
                        except ChildProcessError:
                            break
                        except InterruptedError:
                            continue
                        if pid == 0:
                            break
                        cleanup["reaped_descendants"].append(
                            {"pid": pid, "wait_status": status})
                        if time.monotonic() >= deadline:
                            break
                cleanup["group_absent"] = not exists()
                if cleanup["leader_reaped"] and cleanup["group_absent"]:
                    return True
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.02)

        try:
            if settle(0):
                cleanup["confirmed"] = True
                return cleanup
            for sig, grace in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 2.0)):
                try:
                    os.killpg(pgid, sig)
                    cleanup["signals"].append(sig.name)
                except ProcessLookupError:
                    pass
                if settle(grace):
                    cleanup["confirmed"] = True
                    return cleanup
            cleanup["errors"].append("owned group did not disappear within cleanup bound")
        except BaseException as exc:
            cleanup["errors"].append(type(exc).__name__ + ": " + str(exc))
        return cleanup

    def command(self, label, args, *, cwd=FIXTURE, env_extra=None, timeout=180,
                _cleanup_only_timeout=False):
        if _cleanup_only_timeout and (
                label != "cleanup-only-preflight" or cwd != ROOT or timeout != 3
                or args != [sys.executable, "-B", "-S", "-c", CLEANUP_PARENT]):
            raise ValueError("EXPECTED_TIMEOUT_SCOPE: fixed cleanup preflight only")
        if self.scheduling_blocked or self.subreaper is None:
            raise RuntimeError("PROCESS_SCHEDULING_BLOCKED: cleanup unavailable or uncertain")
        index = len(self.commands) + 1
        prefix = self.native / ("%02d-%s" % (index, label))
        env = os.environ.copy()
        env["LEAN_NUM_THREADS"] = "2"
        if env_extra:
            env.update(env_extra)
        entry = {
            "label": label, "argv": [str(x) for x in args], "cwd": str(cwd),
            "started_utc": utc(), "started_monotonic_ns": time.monotonic_ns(),
            "declared_env": dict(env_extra or {}), "timeout_seconds": timeout,
            "exit_code": None, "timed_out": False, "launch_error": None,
            "pid": None, "pgid": None, "cleanup": None,
            "timeout_expected_cleanup_only": _cleanup_only_timeout,
        }
        process = None
        pending_error = None
        # Files preserve original bytes without PIPE descendants keeping communicate alive.
        with prefix.with_suffix(".stdout").open("wb") as out_stream, (
                prefix.with_suffix(".stderr").open("wb")) as err_stream:
            try:
                process = subprocess.Popen(
                    [str(x) for x in args], cwd=cwd, env=env,
                    stdout=out_stream, stderr=err_stream, start_new_session=True)
                entry["pid"] = entry["pgid"] = process.pid
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    entry["timed_out"] = True
            except OSError as exc:
                entry["launch_error"] = type(exc).__name__ + ": " + str(exc)
                err_stream.write(entry["launch_error"].encode("utf-8"))
            except BaseException as exc:
                pending_error = exc
                entry["interruption"] = type(exc).__name__ + ": " + str(exc)
            finally:
                if process is not None:
                    entry["cleanup"] = self.cleanup_group(process)
                    entry["exit_code"] = process.returncode
                    if not entry["cleanup"]["confirmed"]:
                        self.scheduling_blocked = True
        stdout = prefix.with_suffix(".stdout").read_bytes()
        stderr = prefix.with_suffix(".stderr").read_bytes()
        entry.update(
            finished_utc=utc(), finished_monotonic_ns=time.monotonic_ns(),
            stdout_path=str(prefix.with_suffix(".stdout")),
            stderr_path=str(prefix.with_suffix(".stderr")),
            stdout_sha256=digest(stdout), stderr_sha256=digest(stderr),
            stdout_bytes=len(stdout), stderr_bytes=len(stderr),
            cumulative_child_maxrss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        )
        # ru_maxrss is cumulative across observed children, not a per-process peak.
        self.commands.append(entry)
        json_write(self.output / "commands.json", self.commands)
        if pending_error is not None:
            raise pending_error
        if self.scheduling_blocked:
            raise RuntimeError("PROCESS_CLEANUP_UNCERTAIN: scheduling permanently blocked")
        if entry["timed_out"] and not _cleanup_only_timeout:
            raise RuntimeError("NATIVE_TIMEOUT: original exit/streams/cleanup retained: " + label)
        return entry, stdout, stderr

    def cleanup_only_preflight(self):
        report = {
            "record": "cleanup-only-preflight", "executor": "Python_process_cleanup_only",
            "status": "fail", "outside_lean_method_and_subcase_census": True,
            "scientific_effect": "NONE", "qualification": "BLOCKED",
            "classification_status": "BLOCKED", "execution_authenticated": False,
        }
        try:
            command, out, err = self.command(
                "cleanup-only-preflight", [sys.executable, "-B", "-S", "-c", CLEANUP_PARENT],
                cwd=ROOT, timeout=3, _cleanup_only_timeout=True)
            report["native_label"] = command["label"]
            rows = out.decode("utf-8").splitlines()
            if len(rows) != 1 or err:
                raise ValueError("cleanup preflight readiness output differs")
            ready = json.loads(rows[0])
            pgid = command["pgid"]
            cleanup = command["cleanup"]
            descendant = ready.get("descendant_pid")
            if (ready.get("record") != "cleanup-ready"
                    or ready.get("parent_pid") != command["pid"]
                    or ready.get("parent_pgid") != pgid
                    or ready.get("descendant_pgid") != pgid
                    or type(descendant) is not int or descendant <= 1
                    or descendant == command["pid"]):
                raise ValueError("cleanup preflight did not establish the owned descendant")
            if (not command["timed_out"] or command["exit_code"] != -signal.SIGKILL
                    or cleanup["signals"] != ["SIGTERM", "SIGKILL"]
                    or not cleanup["confirmed"] or not cleanup["leader_reaped"]
                    or not cleanup["group_absent"]
                    or descendant not in [row["pid"] for row in cleanup["reaped_descendants"]]):
                raise ValueError("genuine timeout did not prove owned group TERM/KILL/reap")
            report.update(status="pass", ready=ready, cleanup=cleanup,
                          timed_out=True, leader_exit_code=command["exit_code"])
        except BaseException as exc:
            report["error"] = type(exc).__name__ + ": " + str(exc)
            self.scheduling_blocked = True
            self.cleanup_preflight = report
            json_write(self.output / "cleanup-only-preflight.json", report)
            raise
        self.cleanup_preflight = report
        json_write(self.output / "cleanup-only-preflight.json", report)

    def lean(self, label, lean_args, *, search_root=None, env_extra=None):
        selected = search_root or self.owned
        return self.command(
            label, ["lake", "env", "env", "LEAN_PATH=" + str(selected),
                    "lean", "--threads=2", *lean_args], env_extra=env_extra)

    def bind_request(self, path, request, *, intentionally_missing=()):
        """Bind each path once, after any explicitly declared adverse-copy mutation."""
        path = Path(path).resolve()
        raw = path.read_bytes()
        if json.loads(raw.decode("utf-8")) != request:
            raise ValueError("OBJECT_REQUEST_BYTES: request differs from supplied record")
        parts = request["olean_parts"] + request["ir_parts"]
        paths = tuple(sorted({path, *(Path(part["path"]).resolve() for part in parts)}, key=str))
        snapshot = measure_objects(paths)
        if snapshot[str(path)] != {"exists": True, "bytes": len(raw), "sha256": digest(raw)}:
            raise RuntimeError("OBJECT_DRIFT: request changed while binding")
        missing = {str(Path(name).resolve()) for name in intentionally_missing}
        actual_missing = set()
        declared = {}
        for part in parts:
            name = str(Path(part["path"]).resolve())
            declared[name] = part["sha256"]
            state = snapshot[name]
            if not state["exists"]:
                actual_missing.add(name)
            elif state["sha256"] != part["sha256"]:
                raise RuntimeError("OBJECT_DRIFT: declared part digest differs: " + name)
        if actual_missing != missing:
            raise RuntimeError("OBJECT_MISSING: unexplained or incorrect intentional absence")
        for name, state in snapshot.items():
            if name in self.object_expected and self.object_expected[name] != state:
                raise RuntimeError("OBJECT_DRIFT: cannot reset expected state: " + name)
        for name, state in snapshot.items():
            if name not in self.object_expected:
                self.object_expected[name] = copy.deepcopy(state)
        name = str(path)
        if name not in self.request_objects:
            self.request_objects[name] = paths
            self.request_bindings[name] = {
                "paths": [str(item) for item in paths],
                "request_sha256": digest(raw), "declared_part_sha256": declared,
                "intentionally_missing": sorted(missing),
                "expected": copy.deepcopy(snapshot),
            }
        self.flush()

    def observe_lean(self, label, lean_args, *, requests, search_root=None, env_extra=None):
        if self.object_failures:
            raise RuntimeError("OBJECT_CUSTODY_BLOCKED: earlier custody failure")
        names = [str(Path(path).resolve()) for path in requests]
        if any(name not in self.request_objects for name in names):
            raise ValueError("OBJECT_REQUEST_UNBOUND: fixed observation request")
        paths = tuple(sorted({path for name in names for path in self.request_objects[name]},
                             key=str))
        expected = {str(path): copy.deepcopy(self.object_expected[str(path)]) for path in paths}
        record_path = self.custody / ("%02d-%s.json" %
                                     (len(self.object_observations) + 1, label))
        pending = None
        try:
            with object_custody(paths, expected, record_path):
                return self.lean(label, lean_args, search_root=search_root, env_extra=env_extra)
        except BaseException as exc:
            pending = exc
            raise
        finally:
            try:
                if record_path.is_file():
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                else:
                    record = {"status": "fail", "error": "custody record not produced"}
                command = self.commands[-1] if self.commands and self.commands[-1]["label"] == label else None
                record["native_label"] = label
                record["native_outcome"] = None if command is None else {
                    key: command[key] for key in
                    ("exit_code", "timed_out", "launch_error", "stdout_sha256", "stderr_sha256")
                }
                json_write(record_path, record)
                self.object_observations.append(record)
                if record["status"] != "pass":
                    self.object_failures.append(str(record_path))
                self.flush()
            except BaseException as custody_error:
                self.object_failures.append(str(record_path) + ": " + str(custody_error))
                if pending is not None:
                    raise pending from custody_error
                raise

    def finalize_objects(self):
        after = None
        error = None
        try:
            after = measure_objects(tuple(self.object_expected))
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        drift = [] if after is None else sorted(
            name for name in set(self.object_expected) | set(after)
            if after.get(name) != self.object_expected.get(name))
        self.object_final = {
            "after": after, "drift": drift, "measurement_error": error,
            "observation_count": len(self.object_observations),
            "request_count": len(self.request_bindings),
            "path_count": len(self.object_expected),
            "expected_observation_count": 28,
            "expected_request_count": 10,
            "expected_path_count": 55,
            "status": ("pass" if error is None and not drift and not self.object_failures
                       and len(self.object_observations) == 28
                       and len(self.request_bindings) == 10
                       and len(self.object_expected) == 55 else "fail"),
        }
        self.flush()
        return self.object_final["status"] == "pass"

    def bootstrap(self):
        try:
            self.enable_subreaper()
            self.cleanup_only_preflight()
            if not self.identities["fixed_profile_matches"]:
                self.bootstrap_errors.append("FROZEN_PROFILE_CHANGED")
            for expression, key in (("HEAD^{commit}", "main_commit"), ("HEAD^{tree}", "main_tree")):
                observed, raw, _ = self.command(
                    key, ["git", "rev-parse", "--verify", expression], cwd=ROOT)
                self.identities[key] = raw.decode("utf-8", errors="replace").strip()
                if observed["exit_code"] != 0:
                    self.bootstrap_errors.append("MAIN_SOURCE_IDENTITY")
            self.task1 = load_module(ROOT / "tools" / "sdga_scope.py", "sdga_scope_task_a")
            self.scope = {
                "scope_id": "sdga-task-a-fixture",
                "modules": [{"module": "Fixture.Safe", "path": "Fixture/Safe.lean",
                             "role": "production"}],
                "targets": ["Fixture.Safe.proof", "Fixture.Safe.propValue",
                            "Fixture.Safe.natValue"],
                "expected_counts": {"public_targets": 3,
                                    "proposition_targets": 1, "data_targets": 2},
            }
            try:
                self.controller = load_module(CONTROLLER, "sdga_component_task_a")
            except Exception as exc:
                self.controller_error = type(exc).__name__ + ": " + str(exc)
                (self.output / "controller-import.txt").write_text(
                    traceback.format_exc(), encoding="utf-8")
            tool, out, _ = self.command("lean-githash", ["lean", "--githash"])
            self.identities["actual_lean_githash"] = out.decode("utf-8", errors="replace").strip()
            if tool["exit_code"] != 0 or self.identities["actual_lean_githash"] != RELEASE:
                self.bootstrap_errors.append("TOOLCHAIN_IDENTITY")
            # Fixed sequential compile order; B never accepts arbitrary module names.
            task_b_dir = self.owned / "Fixture"
            task_b_dir.mkdir(parents=True)
            dependency, _, _ = self.lean(
                "fixture-bad-dependency-compile",
                ["--root=" + str(FIXTURE), "-o", str(task_b_dir / "BadDependency.olean"),
                 "-i", str(task_b_dir / "BadDependency.ilean"),
                 str(FIXTURE / "Fixture" / "BadDependency.lean")])
            if dependency["exit_code"] != 0:
                self.task_b_errors["AxiomUse"] = ["BAD_DEPENDENCY_COMPILE_FAILED"]
            else:
                # Binding only: zero inventory, observation or subcase credit.
                dependency_scope = {
                    "scope_id": "sdga-task-b-BadDependency",
                    "modules": [{"module": "Fixture.BadDependency",
                                 "path": "Fixture/BadDependency.lean", "role": "production"}],
                    "targets": [],
                    "expected_counts": {"public_targets": 0, "proposition_targets": 0,
                                        "data_targets": 0},
                }
                dependency_base = task_b_dir / "BadDependency.olean"
                dependency_request = {
                    "schema": 1, "scope_id": dependency_scope["scope_id"],
                    "profile_sha256": digest(json.dumps(
                        dependency_scope, sort_keys=True).encode("utf-8")),
                    "module": self.task1.encode_name(("Fixture", "BadDependency")),
                    "olean_parts": [
                        {"level": level, "path": str(path.resolve()),
                         "sha256": digest(path.read_bytes())}
                        for level, path in (
                            ("exported", dependency_base),
                            ("server", Path(str(dependency_base) + ".server")),
                            ("private", Path(str(dependency_base) + ".private")))],
                    "ir_parts": [
                        {"level": level, "path": str(path.resolve()),
                         "sha256": digest(path.read_bytes())}
                        for level, path in (
                            ("ir.sig", dependency_base.with_suffix(".ir.sig")),
                            ("ir", dependency_base.with_suffix(".ir")))],
                }
                json_write(self.task_b_dependency_request, dependency_request)
                self.bind_request(self.task_b_dependency_request, dependency_request)
            target = self.owned / "Fixture" / "Safe.olean"
            target.parent.mkdir(parents=True, exist_ok=True)
            safe, _, _ = self.lean(
                "fixture-safe-compile",
                ["--root=" + str(FIXTURE), "-o", str(target),
                 "-i", str(target.with_suffix(".ilean")),
                 str(FIXTURE / "Fixture" / "Safe.lean")])
            if safe["exit_code"] != 0:
                self.bootstrap_errors.append("FIXTURE_COMPILE_FAILED")
            for module in TASK_B_FIXTURES[1:]:
                if self.task_b_errors.get(module):
                    continue  # AxiomUse is not scheduled after dependency failure.
                compiled, _, _ = self.lean(
                    "fixture-" + module.lower() + "-compile",
                    ["--root=" + str(FIXTURE), "-o", str(task_b_dir / (module + ".olean")),
                     "-i", str(task_b_dir / (module + ".ilean")),
                     str(FIXTURE / "Fixture" / (module + ".lean"))])
                if compiled["exit_code"] != 0:
                    self.task_b_errors[module] = ["FIXTURE_COMPILE_FAILED"]
            bridge, _, _ = self.lean(
                "bridge-compile",
                ["--root=" + str(BRIDGE.parent), "-o", str(self.owned / "Component.olean"),
                 "-i", str(self.owned / "Component.ilean"), str(BRIDGE)])
            if bridge["exit_code"] != 0:
                self.bootstrap_errors.append(
                    "IMPLEMENTATION_ABSENCE: tools/sdga/Component.lean"
                    if not BRIDGE.is_file() else "BRIDGE_COMPILE_FAILED")
            if not self.bootstrap_errors:
                self.request = self.request_for(self.owned)
                json_write(self.request_path, self.request)
                self.bind_request(self.request_path, self.request)
                command, out, err = self.observe_lean(
                    "inventory", ["--run", str(BRIDGE), "inventory", str(self.request_path)],
                    requests=(self.request_path,))
                (self.output / "inventory.jsonl").write_bytes(out)
                if command["exit_code"] != 0:
                    self.bootstrap_errors.append("INVENTORY_NATIVE_FAILED")
                else:
                    self.inventory_text = out.decode("utf-8")
                    self.inventory = self.task1.parse_inventory(self.inventory_text)
                    result = self.task1.classify_inventory(self.inventory, self.scope)
                    json_write(self.output / "fixture-classification.json", result)
                    if result["classification_status"] != "BLOCKED":
                        self.bootstrap_errors.append("CLASSIFICATION_NOT_BLOCKED")
                    counts = result.get("counts", {})
                    if any(counts.get(key) != expected for key, expected in
                           self.scope["expected_counts"].items()):
                        self.bootstrap_errors.append("FIXTURE_TARGET_COUNTS")
                    if result.get("findings") != []:
                        self.bootstrap_errors.append("UNEXPECTED_FIXTURE_FINDINGS")
        except Exception:
            self.bootstrap_errors.append("BOOTSTRAP_ERROR")
            (self.output / "bootstrap-error.txt").write_text(
                traceback.format_exc(), encoding="utf-8")
        self.identities["bootstrap_errors"] = self.bootstrap_errors
        self.identities["task_b_fixture_errors"] = self.task_b_errors
        self.identities["controller_import_error"] = self.controller_error
        self.identities["process_scheduling_blocked"] = self.scheduling_blocked
        self.identities["cleanup_only_preflight"] = self.cleanup_preflight
        self.flush()

    def request_for(self, owned):
        base = owned / "Fixture" / "Safe.olean"
        parts = []
        for level, path in (
                ("exported", base), ("server", Path(str(base) + ".server")),
                ("private", Path(str(base) + ".private"))):
            raw = path.read_bytes()
            parts.append({"level": level, "path": str(path.resolve()), "sha256": digest(raw)})
        ir_parts = []
        for level, path in (
                ("ir.sig", base.with_suffix(".ir.sig")), ("ir", base.with_suffix(".ir"))):
            if path.is_file():
                ir_parts.append({"level": level, "path": str(path.resolve()),
                                 "sha256": digest(path.read_bytes())})
        return {
            "schema": 1, "scope_id": "sdga-task-a-fixture",
            "profile_sha256": digest(json.dumps(self.scope, sort_keys=True).encode("utf-8")),
            "module": self.task1.encode_name(("Fixture", "Safe")),
            "olean_parts": parts, "ir_parts": ir_parts,
        }

    def flush(self):
        self.identities["cli_replay"] = self.cli_replay
        self.identities["object_custody"] = {
            "expected": self.object_expected, "request_bindings": self.request_bindings,
            "observations": self.object_observations, "failures": self.object_failures,
            "final": self.object_final, "outside_lean_method_and_subcase_census": True,
        }
        json_write(self.output / "object-custody.json", self.identities["object_custody"])
        json_write(self.output / "source-identities.json", self.identities)
        with (self.output / "subcases.jsonl").open("w", encoding="utf-8") as stream:
            for key in self.records:
                stream.write(json.dumps(self.records[key], sort_keys=True,
                                        ensure_ascii=False) + "\n")
        statuses = {}
        for row in self.records.values():
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        json_write(self.output / "census.json", {
            "declared_python_methods": 15, "declared_subcases": 67,
            "original_task_a_methods": 6, "original_task_a_subcases": 39,
            "actual_subcase_statuses": statuses,
            "scientific_effect": "NONE", "qualification": "BLOCKED",
            "classification_status": "BLOCKED", "execution_authenticated": False,
        })


class TaskAContracts(unittest.TestCase):
    def require_bridge(self):
        if STATE.bootstrap_errors:
            reason = "; ".join(STATE.bootstrap_errors)
            self.fail(reason + " — no semantic or mutation credit")
        self.assertIsNotNone(STATE.inventory)

    def accept_rows(self, case, expected_subcases, command, stdout):
        rows = []
        protocol_errors = []
        setup_rows = []
        for line in stdout.decode("utf-8").splitlines():
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as exc:
                protocol_errors.append("non-JSON native output: " + str(exc))
                continue
            if row.get("record") == "setup_error":
                setup_rows.append(row)
                continue
            if (row.get("record") != "subcase" or row.get("case_id") != case
                    or row.get("executor") != "Lean"
                    or row.get("subcase_id") not in expected_subcases
                    or row.get("status") not in ("pass", "fail")):
                protocol_errors.append("unexpected native record: " + repr(row))
                continue
            key = (case, row["subcase_id"])
            if STATE.records[key]["status"] != "not_run":
                protocol_errors.append("duplicate native subcase: " + repr(key))
                continue
            rows.append(row)
            STATE.records[key] = row
        STATE.flush()
        json_write(STATE.output / ("control-observation-" + command["label"] + ".json"),
                   {"native_label": command["label"], "rows": rows,
                    "setup_rows": setup_rows, "protocol_errors": protocol_errors,
                    "exit_code": command["exit_code"]})
        self.assertEqual(command["exit_code"], 0,
                         "native control failed; original rows and streams retained")
        self.assertFalse(setup_rows, "setup rejection is not a semantic negative")
        self.assertFalse(protocol_errors, "; ".join(protocol_errors))
        self.assertEqual([row["subcase_id"] for row in rows], list(expected_subcases),
                         "missing, reordered, duplicate or unexpected native subcase")
        self.assertTrue(all(row["status"] == "pass" for row in rows),
                        "a native assertion failed; no unrelated-error rejection credit")
        return rows

    def native_case(self, case, *, subcase=None, alternate=None, search_root=None):
        self.require_bridge()
        extra = {"SDGA_CONTROL_CASE": case,
                 "SDGA_CONTROL_REQUEST": str(STATE.request_path)}
        expected = CASES[case]
        if subcase is not None:
            extra["SDGA_CONTROL_SUBCASE"] = subcase
            expected = (subcase,)
        if alternate is not None:
            extra["SDGA_CONTROL_ALTERNATE_REQUEST"] = str(alternate)
        command, out, _ = STATE.observe_lean(
            "control-" + case + ("-" + subcase if subcase else ""),
            ["--root=" + str(FIXTURE), str(FIXTURE / "Controls.lean")],
            requests=(STATE.request_path,) + ((alternate,) if alternate is not None else ()),
            search_root=search_root, env_extra=extra)
        return self.accept_rows(case, expected, command, out)

    def record_python(self, subcase, action):
        try:
            action()
        except Exception as exc:
            STATE.records[("C15", subcase)] = {
                "record": "subcase", "case_id": "C15", "subcase_id": subcase,
                "executor": "Python_controller", "status": "fail",
                "detail": type(exc).__name__ + ": " + str(exc)}
            STATE.flush()
            raise
        STATE.records[("C15", subcase)] = {
            "record": "subcase", "case_id": "C15", "subcase_id": subcase,
            "executor": "Python_controller", "status": "pass"}
        STATE.flush()

    def expected_controller_error(self, code, action):
        try:
            action()
        except ValueError as exc:
            self.assertTrue(str(exc).startswith(code + ":"),
                            "unexpected controller failure cause: " + str(exc))
        else:
            self.fail("expected controller refusal " + code)

    def test_C01_name_wire(self):
        self.native_case("C01")

    def test_C02_owned_population(self):
        rows = self.native_case("C02")
        detail = rows[-1].get("detail")
        self.assertIsInstance(detail, dict)
        actual = detail.get("final_names")
        self.assertIsInstance(actual, list)
        emitted = [row["name"] for row in STATE.inventory["declarations"]]
        self.assertEqual(len(actual), len(set(actual)), "duplicate raw final name")
        self.assertEqual(set(emitted), set(actual), "inventory omitted actual final constants")
        self.assertEqual(len(emitted), len(actual))

        # Compare the actual CLI fields with independently emitted raw/core oracles.
        # The shared observeConstant helper is still asserted by Controls separately.
        fields = ("name", "module", "kind", "is_prop", "is_private", "unsafe", "partial")
        oracle = detail.get("oracle_rows")
        self.assertIsInstance(oracle, list)
        self.assertEqual(len(oracle), len(actual), "incomplete raw/core oracle census")
        self.assertTrue(all(type(row) is dict and set(row) == set(fields) for row in oracle))
        self.assertEqual([row["name"] for row in oracle], actual,
                         "raw/core oracle identities differ from final raw order")
        self.assertEqual(len({row["name"] for row in oracle}), len(oracle))
        observed = {row["name"]: row for row in STATE.inventory["declarations"]}
        for row in oracle:
            with self.subTest(oracle_name=row["name"]):
                self.assertEqual({key: observed[row["name"]][key] for key in fields}, row,
                                 "actual inventory differs from independent core observation")
        raw_parts = detail.get("raw_parts")
        self.assertIsInstance(raw_parts, list)
        self.assertEqual([part["level"] for part in raw_parts],
                         ["exported", "server", "private"])
        self.assertEqual(len(STATE.inventory["modules"]), 1)
        module = STATE.inventory["modules"][0]
        self.assertEqual(STATE.inventory["scope_id"], STATE.scope["scope_id"])
        self.assertEqual(module["name"], STATE.request["module"])
        self.assertEqual(module["const_count"], len(actual))
        self.assertEqual(len(module["object_parts"]), len(raw_parts))
        extras = set()
        for raw, emitted_part, requested in zip(
                raw_parts, module["object_parts"], STATE.request["olean_parts"]):
            self.assertEqual(set(raw), {"level", "const_names", "constant_names",
                                        "extra_const_names"})
            self.assertEqual(
                emitted_part,
                {"level": raw["level"], "sha256": requested["sha256"],
                 "const_names": raw["const_names"],
                 "constant_names": raw["constant_names"]},
                "actual inventory part fields differ from raw tables/measured request")
            self.assertEqual(raw["const_names"], raw["constant_names"])
            extras.update(raw["extra_const_names"])
        self.assertEqual(set(module["extra_const_names"]), extras)
        classification = STATE.task1.classify_inventory(STATE.inventory, STATE.scope)
        self.assertEqual(classification["counts"],
                         {"owned_declarations": len(actual), "public_targets": 3,
                          "proposition_targets": 1, "data_targets": 2})
        self.assertEqual(classification["findings"], [],
                         "safe fixture emitted unexpected blocking findings")
        self.assertEqual(classification["classification_status"], "BLOCKED")

    def test_C03_actual_prop_data(self):
        self.native_case("C03")

    def test_C12_real_part_table_adverses(self):
        self.native_case("C12")

    def test_C13_real_part_file_adverses(self):
        self.require_bridge()
        for subcase in CASES["C13"]:
            with self.subTest(subcase=subcase):
                case_dir = STATE.output / ("case-" + subcase)
                shutil.copytree(STATE.owned, case_dir)
                request = STATE.request_for(case_dir)
                if subcase == "missing-server":
                    Path(request["olean_parts"][1]["path"]).unlink()
                elif subcase == "reordered-parts":
                    request["olean_parts"][0], request["olean_parts"][1] = (
                        request["olean_parts"][1], request["olean_parts"][0])
                else:
                    private = Path(request["olean_parts"][2]["path"])
                    raw = private.read_bytes()
                    private.write_bytes(raw[:max(1, len(raw) // 4)])
                    request["olean_parts"][2]["sha256"] = digest(private.read_bytes())
                request_path = STATE.output / (subcase + "-request.json")
                json_write(request_path, request)
                missing = ((Path(request["olean_parts"][1]["path"]),)
                           if subcase == "missing-server" else ())
                STATE.bind_request(request_path, request, intentionally_missing=missing)
                self.native_case("C13", subcase=subcase, alternate=request_path,
                                 search_root=case_dir)

    def test_C15_identity_and_controller_contracts(self):
        self.require_bridge()
        copy_dir = STATE.output / "case-wrong-resolved-path"
        shutil.copytree(STATE.owned, copy_dir)
        wrong = STATE.output / "wrong-path-request.json"
        wrong_request = STATE.request_for(copy_dir)
        json_write(wrong, wrong_request)
        STATE.bind_request(wrong, wrong_request)
        self.native_case("C15", subcase="wrong-resolved-path", alternate=wrong)
        for subcase, args, code in (
                ("unknown-mode", ["not-a-mode", str(STATE.request_path)], "CLI_MODE"),
                ("unknown-field", ["inventory", str(STATE.output / "unknown-field.json")],
                 "REQUEST_SCHEMA")):
            if subcase == "unknown-field":
                altered = copy.deepcopy(STATE.request)
                altered["unapproved"] = True
                json_write(STATE.output / "unknown-field.json", altered)
                STATE.bind_request(STATE.output / "unknown-field.json", altered)
            command, out, err = STATE.observe_lean(
                "cli-" + subcase, ["--run", str(BRIDGE), *args],
                requests=(STATE.request_path, Path(args[-1])))
            combined = (out + b"\n" + err).decode("utf-8", errors="strict")
            self.assertNotEqual(command["exit_code"], 0)
            self.assertEqual(out, b"", "native CLI refusal must not emit an inventory or compiler output")
            error_lines = err.decode("utf-8").splitlines()
            self.assertEqual(len(error_lines), 1, "unrelated native diagnostics cannot satisfy refusal")
            self.assertTrue(error_lines[0].startswith("SDGA_ERROR " + code + ":"),
                            "unrelated setup/absence failure cannot satisfy native refusal")
            STATE.records[("C15", subcase)] = {
                "record": "subcase", "case_id": "C15", "subcase_id": subcase,
                "executor": "Lean", "status": "pass", "error_code": code,
                "native_label": command["label"]}
            STATE.flush()

        def malformed_transcript():
            with self.assertRaises(STATE.task1.InventoryProtocolError):
                STATE.task1.parse_inventory(STATE.inventory_text + '{"record":"extra"}\n')
        self.record_python("malformed-transcript", malformed_transcript)
        if STATE.controller is None:
            self.fail("IMPLEMENTATION_ABSENCE: tools/sdga_component.py — no controller semantics credit")

        malformed = STATE.output / "malformed-header.lean"
        malformed.write_text("module\npublic import\n", encoding="utf-8")
        command, out, _ = STATE.lean(
            "deps-json-malformed-header", ["--deps-json", str(malformed)])
        self.assertEqual(command["exit_code"], 0,
                         "this control must expose per-file error despite process success")
        payload = out.decode("utf-8")
        parsed = json.loads(payload)
        self.assertEqual(len(parsed["imports"]), 1)
        self.assertTrue(parsed["imports"][0].get("errors"),
                        "fixture did not exercise the expected header error")
        self.record_python(
            "deps-json-error-row",
            lambda: self.expected_controller_error(
                "DEPENDENCY_HEADERS",
                lambda: STATE.controller.parse_dependency_headers(payload, (str(malformed),))))
        self.record_python(
            "duplicate-vertex",
            lambda: self.expected_controller_error(
                "DUPLICATE_MODULE",
                lambda: STATE.controller.topological_modules({"A": ()}, ("A", "A"))))
        self.record_python(
            "missing-local-edge",
            lambda: self.expected_controller_error(
                "MISSING_LOCAL_EDGE",
                lambda: STATE.controller.topological_modules({"A": ("B",)}, ("A",))))
        self.record_python(
            "cycle",
            lambda: self.expected_controller_error(
                "DEPENDENCY_CYCLE",
                lambda: STATE.controller.topological_modules(
                    {"A": ("B",), "B": ("A",)}, ("A", "B"))))

    def task_b_inventory(self, module):
        """Retain the actual complete native inventory before grading Task B."""
        self.require_bridge()
        self.assertIn(module, ("AxiomUse", "OwnedAxiom", "Skipped"))
        self.assertFalse(STATE.task_b_errors.get(module),
                         repr(STATE.task_b_errors.get(module)))
        scope = {
            "scope_id": "sdga-task-b-" + module,
            "modules": [{"module": "Fixture." + module,
                         "path": "Fixture/" + module + ".lean", "role": "production"}],
            "targets": [],
            "expected_counts": {"public_targets": 0, "proposition_targets": 0, "data_targets": 0},
        }
        base = STATE.owned / "Fixture" / (module + ".olean")
        request = {
            "schema": 1, "scope_id": scope["scope_id"],
            "profile_sha256": digest(json.dumps(scope, sort_keys=True).encode("utf-8")),
            "module": STATE.task1.encode_name(("Fixture", module)),
            "olean_parts": [
                {"level": level, "path": str(path.resolve()), "sha256": digest(path.read_bytes())}
                for level, path in (("exported", base), ("server", Path(str(base) + ".server")),
                                    ("private", Path(str(base) + ".private")))],
            "ir_parts": [
                {"level": level, "path": str(path.resolve()), "sha256": digest(path.read_bytes())}
                for level, path in (("ir.sig", base.with_suffix(".ir.sig")),
                                    ("ir", base.with_suffix(".ir")))],
        }
        request_path = STATE.output / ("task-b-" + module + "-request.json")
        json_write(request_path, request)
        STATE.bind_request(request_path, request)
        requests = (request_path,) + (
            (STATE.task_b_dependency_request,) if module == "AxiomUse" else ())
        command, out, _ = STATE.observe_lean(
            "task-b-" + module + "-inventory",
            ["--run", str(BRIDGE), "inventory", str(request_path)], requests=requests)
        (STATE.output / ("task-b-" + module + "-inventory.jsonl")).write_bytes(out)
        self.assertEqual(command["exit_code"], 0,
                         "Task B inventory failed; partial transcript is blocking evidence")
        inventory = STATE.task1.parse_inventory(out.decode("utf-8"))
        classification = STATE.task1.classify_inventory(inventory, scope)
        json_write(STATE.output / ("task-b-" + module + "-classification.json"), classification)
        self.assertEqual(classification["classification_status"], "BLOCKED")
        return request_path, inventory, classification

    def task_b_native(self, case, request_path=None, subcase=None):
        extra = {"SDGA_CONTROL_CASE": case, "SDGA_CONTROL_REQUEST": str(STATE.request_path)}
        if request_path is not None:
            extra["SDGA_CONTROL_TASK_B_REQUEST"] = str(request_path)
        if subcase is not None:
            extra["SDGA_CONTROL_SUBCASE"] = subcase
        requests = (STATE.request_path,) if case == "C05" else (request_path,)
        if case == "C04":
            requests += (STATE.task_b_dependency_request,)
        command, out, _ = STATE.observe_lean(
            "control-" + case + ("-" + subcase if subcase else ""),
            ["--root=" + str(FIXTURE), str(FIXTURE / "Controls.lean")],
            requests=requests, env_extra=extra)
        expected = (subcase,) if subcase else (
            CASES[case][:2] if case == "C06" else CASES[case])
        try:
            return self.accept_rows(case, expected, command, out)
        finally:
            # Failed native assertions retain this provenance too.
            for subcase_id in expected:
                row = STATE.records[(case, subcase_id)]
                if row["status"] != "not_run":
                    row["native_label"] = command["label"]
            STATE.flush()

    def reconcile_task_b_rows(self, inventory, rows):
        observed = {row["name"]: row for row in inventory["declarations"]}
        fields = ("name", "module", "kind", "is_prop", "is_private", "unsafe", "partial")
        for row in rows:
            oracle = row.get("detail", {}).get("oracle_row")
            if oracle is not None:
                self.assertIn(oracle["name"], observed, "native inventory omitted actual CI")
                self.assertEqual({key: observed[oracle["name"]][key] for key in fields}, oracle)

    def test_C04_actual_axiom_routes(self):
        request, inventory, classification = self.task_b_inventory("AxiomUse")
        rows = self.task_b_native("C04", request)
        bad = STATE.task1.encode_name(("Fixture", "BadDependency", "BadProof"))
        self.assertIn({"code": "FORBIDDEN_AXIOM", "name": bad}, classification["findings"])
        observed = {row["name"]: row for row in inventory["declarations"]}
        for source in ("typeOnly", "valueOnly", "opaqueOnly"):
            name = STATE.task1.encode_name(("Fixture", "AxiomUse", source))
            self.assertIn(name, observed)
            self.assertIn(bad, observed[name]["axioms"])
        self.assertEqual(len(rows), 3)

    def test_C05_actual_missing_lookup(self):
        self.require_bridge()
        self.task_b_native("C05")

    def test_C06_actual_owned_allowlist_axiom(self):
        request, inventory, classification = self.task_b_inventory("OwnedAxiom")
        rows = self.task_b_native("C06", request)
        self.reconcile_task_b_rows(inventory, rows)
        # This subcase's executor is the actual unchanged Python Task 1 classifier.
        key = ("C06", "owned-axiom-block")
        try:
            name = STATE.task1.encode_name(("propext",))
            observed = [row for row in inventory["declarations"] if row["name"] == name]
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0]["kind"], "axiom")
            self.assertIn(name, observed[0]["axioms"])
            self.assertIn({"code": "OWNED_AXIOM", "name": name}, classification["findings"])
            self.assertNotIn({"code": "FORBIDDEN_AXIOM", "name": name}, classification["findings"])
        except Exception as exc:
            STATE.records[key] = {
                "record": "subcase", "case_id": key[0], "subcase_id": key[1],
                "executor": "Python_controller", "status": "fail",
                "detail": type(exc).__name__ + ": " + str(exc)}
            STATE.flush()
            raise
        STATE.records[key] = {
            "record": "subcase", "case_id": key[0], "subcase_id": key[1],
            "executor": "Python_controller", "status": "pass",
            "detail": {"expected_code": "OWNED_AXIOM", "name": name,
                       "source": "unchanged-Task1-classify_inventory",
                       "native_label": "task-b-OwnedAxiom-inventory"}}
        STATE.flush()

    def test_C07_actual_safety_and_shared_guard(self):
        request, inventory, classification = self.task_b_inventory("Skipped")
        for subcase in CASES["C07"]:
            with self.subTest(subcase=subcase):
                rows = self.task_b_native("C07", request, subcase)
                self.reconcile_task_b_rows(inventory, rows)
                if subcase in ("unsafe-info", "partial-info"):
                    name = rows[0]["detail"]["oracle_row"]["name"]
                    code = "UNSAFE_REPLAY_SKIP" if subcase == "unsafe-info" else "PARTIAL_REPLAY_SKIP"
                    self.assertIn({"code": code, "name": name}, classification["findings"])



    def test_C08_actual_complete_safe_kernel_replay(self):
        with self.subTest(route="shared-helper"):
            rows = self.native_case("C08")
            detail = rows[-1]["detail"]
            self.assertEqual(detail["helper"], "SDGA.replayOwned")
            self.assertEqual(detail["owned_count"], len(STATE.inventory["declarations"]))
            self.assertEqual(detail["skipped_names"], [])
            self.assertIn(STATE.task1.encode_name(("Fixture", "Safe", "Switch")),
                          detail["inductives"])
            self.assertGreaterEqual(len(detail["constructors"]), 2)
            self.assertTrue(detail["recursors"])

        with self.subTest(route="public-cli"):
            self.require_bridge()
            STATE.cli_replay["status"] = "running"
            STATE.flush()
            command = None
            try:
                command, out, err = STATE.observe_lean(
                    "cli-safe-replay",
                    ["--run", str(BRIDGE), "replay", str(STATE.request_path)],
                    requests=(STATE.request_path,))
                STATE.cli_replay.update(
                    command=copy.deepcopy(command),
                    stdout_hex=out.hex(), stderr_hex=err.hex())
                self.assertEqual(command["exit_code"], 0,
                                 "public replay CLI must succeed; raw native outcome retained")
                self.assertEqual(err, b"", "public replay CLI success emitted stderr")
                lines = out.decode("utf-8").splitlines()
                self.assertEqual(len(lines), 1,
                                 "public replay CLI must emit exactly one success receipt")
                def receipt_object(pairs):
                    value = dict(pairs)
                    self.assertEqual(len(pairs), len(value),
                                     "public replay receipt repeats a JSON field")
                    return value
                receipt = json.loads(lines[0], object_pairs_hook=receipt_object)
                self.assertIs(type(receipt), dict)
                native_names = [row["name"] for row in STATE.inventory["declarations"]]
                expected = {
                    "schema": "cbz.sdga.kernel-replay.v1",
                    "module": STATE.request["module"],
                    "scope_id": STATE.request["scope_id"],
                    "profile_sha256": STATE.request["profile_sha256"],
                    "helper": "SDGA.replayOwned",
                    "owned_count": len(native_names),
                    "replayed_names": sorted(native_names),
                    "skipped_names": [],
                    "scientific_effect": "NONE",
                    "qualification": "BLOCKED",
                    "classification_status": "BLOCKED",
                    "execution_authenticated": False,
                }
                self.assertEqual(set(receipt), set(expected))
                self.assertIs(type(receipt["owned_count"]), int)
                self.assertIs(receipt["execution_authenticated"], False)
                self.assertIs(type(receipt["replayed_names"]), list)
                self.assertTrue(all(type(name) is str for name in receipt["replayed_names"]))
                self.assertEqual(len(receipt["replayed_names"]),
                                 len(set(receipt["replayed_names"])),
                                 "public replay receipt repeats an owned Name")
                normalized = dict(receipt)
                normalized["replayed_names"] = sorted(receipt["replayed_names"])
                self.assertEqual(normalized, expected,
                                 "public replay receipt differs from actual request/inventory")
                STATE.cli_replay.update(status="pass", receipt=receipt)
            except BaseException as exc:
                if command is None and STATE.commands and STATE.commands[-1]["label"] == "cli-safe-replay":
                    STATE.cli_replay["command"] = copy.deepcopy(STATE.commands[-1])
                STATE.cli_replay.update(status="fail", error=type(exc).__name__ + ": " + str(exc))
                try:
                    STATE.flush()
                except BaseException as record_error:
                    raise exc from record_error
                raise
            else:
                STATE.flush()

    def test_C09_actual_malformed_theorem_kernel_rejection(self):
        rows = self.native_case("C09")
        name = STATE.task1.encode_name(("Fixture", "Safe", "good"))
        references = {STATE.task1.encode_name(("True",)),
                      STATE.task1.encode_name(("Nat", "zero"))}
        self.assertEqual(rows[0]["detail"]["name"], name)
        self.assertEqual(rows[0]["detail"]["altered_field"], "value")
        self.assertEqual(rows[1]["detail"]["singleton_names"], [name])
        self.assertEqual(set(rows[1]["detail"]["resolved_references"]), references)
        self.assertTrue(rows[2]["detail"]["target_absent"])
        self.assertTrue(rows[2]["detail"]["all_owned_absent"])
        rejection = rows[3]["detail"]
        self.assertEqual(rejection["helper"], "SDGA.replayOwned")
        self.assertEqual(rejection["kernel_cause"], "declTypeMismatch")
        self.assertEqual(rejection["singleton_names"], [name])
        self.assertTrue(rejection["native_error"].startswith(
            "while replaying declaration 'Fixture.Safe.good':\n"))
        self.assertIn("(kernel) declaration type mismatch", rejection["native_error"])

    def test_C10_actual_good_replay_then_duplicate_base_guard(self):
        rows = self.native_case("C10")
        name = STATE.task1.encode_name(("Fixture", "Safe", "good"))
        setup = rows[0]["detail"]
        self.assertEqual(setup["helper"], "SDGA.replayOwned")
        self.assertEqual(setup["owned_count"], len(STATE.inventory["declarations"]))
        self.assertEqual(setup["skipped_names"], [])
        self.assertEqual(rows[1]["detail"]["singleton_names"], [name])
        guard = rows[2]["detail"]
        self.assertEqual(guard["helper"], "SDGA.replayOwned")
        self.assertEqual(guard["expected_code"], "DUPLICATE_BASE")
        self.assertEqual(guard["singleton_names"], [name])
        self.assertEqual(guard["native_error"], "SDGA_ERROR DUPLICATE_BASE: " + name)
        self.assertEqual(guard["contamination"], "successful-original-complete-map-replay")


    # Task D assertions are scoped here. Baseline C bootstrap never imports or
    # requires the absent I/O module or sidecars at the test-first checkpoint.
    def d_json(self, path):
        def unique(pairs):
            value = {}
            for key, item in pairs:
                self.assertNotIn(key, value, "duplicate artifact JSON field")
                value[key] = item
            return value
        self.assertTrue(path.is_file(), "IMPLEMENTATION_ABSENCE: required IR artifact " + str(path))
        return json.loads(path.read_bytes().decode("utf-8"), object_pairs_hook=unique)

    def d_keys(self, value, fields):
        self.assertIs(type(value), dict)
        self.assertEqual(set(value), set(fields.split()))

    def d_nat(self, value):
        self.assertIs(type(value), int)
        self.assertGreaterEqual(value, 0)

    def d_hash(self, value, length=64):
        self.assertIs(type(value), str)
        self.assertEqual(len(value), length)
        self.assertTrue(all(char in "0123456789abcdef" for char in value))

    def d_wire(self, value, anonymous=False):
        self.assertIs(type(value), str)
        decoded = STATE.task1.decode_name(value)
        self.assertEqual(STATE.task1.encode_name(decoded), value)
        self.assertTrue(anonymous or value != "")

    def d_state(self, value, nullable=False):
        if nullable and value is None:
            return
        self.assertIs(type(value), dict)
        self.assertIs(type(value.get("exists")), bool)
        if value["exists"]:
            self.d_keys(value, "exists bytes sha256")
            self.d_nat(value["bytes"])
            self.d_hash(value["sha256"])
        else:
            self.d_keys(value, "exists")

    def d_policy(self, policy):
        self.d_keys(policy, "olean_extras ir_residue combined")
        self.assertTrue(all(value in ("clear", "blocked") for value in policy.values()))
        self.assertEqual(policy["combined"], "blocked" if "blocked" in (
            policy["olean_extras"], policy["ir_residue"]) else "clear")

    def d_core(self, core):
        self.d_keys(core, "olean_parts olean_extra_union ir_parts ir_occurrences "
                    "ci_backed_names unexplained_residue policy")
        for family in ("olean_parts", "ir_parts"):
            self.assertIs(type(core[family]), list)
            self.assertEqual(len(core[family]), len(STATE.request[family]))
            for part, requested in zip(core[family], STATE.request[family]):
                self.d_keys(part, "level path declared_sha256 is_module const_names "
                            "constant_names extra_const_names extension_entries")
                self.assertEqual({key: part[key] for key in ("level", "path")},
                                 {key: requested[key] for key in ("level", "path")})
                self.assertEqual(part["declared_sha256"], requested["sha256"])
                self.assertIs(type(part["is_module"]), bool)
                for field in ("const_names", "constant_names", "extra_const_names"):
                    self.assertIs(type(part[field]), list)
                    for name in part[field]:
                        self.d_wire(name)
                self.assertIs(type(part["extension_entries"]), list)
                for extension in part["extension_entries"]:
                    self.d_keys(extension, "name entry_count")
                    self.d_wire(extension["name"], anonymous=True)
                    self.d_nat(extension["entry_count"])
        expected_extra = list(dict.fromkeys(
            name for part in core["olean_parts"] for name in part["extra_const_names"]))
        self.assertEqual(core["olean_extra_union"], expected_extra)
        owned = {row["name"] for row in STATE.inventory["declarations"]}
        occurrences = []
        backed, residue = [], []
        for part_index, part in enumerate(core["ir_parts"]):
            for field in ("const_names", "constant_names", "extra_const_names"):
                for index, name in enumerate(part[field]):
                    is_owned = name in owned
                    occurrences.append({"part_index": part_index, "field": field, "index": index,
                                        "name": name, "ci_backed": is_owned})
                    population = backed if is_owned else residue
                    if name not in population:
                        population.append(name)
        self.assertIs(type(core["ir_occurrences"]), list)
        for occurrence in core["ir_occurrences"]:
            self.d_keys(occurrence, "part_index field index name ci_backed")
            self.d_nat(occurrence["part_index"])
            self.d_nat(occurrence["index"])
            self.assertIs(type(occurrence["ci_backed"]), bool)
            self.d_wire(occurrence["name"])
        self.assertEqual(core["ir_occurrences"], occurrences)
        self.assertEqual(core["ci_backed_names"], backed)
        self.assertEqual(core["unexplained_residue"], residue)
        self.d_policy(core["policy"])
        self.assertEqual(core["policy"], {
            "olean_extras": "blocked" if expected_extra else "clear",
            "ir_residue": "blocked" if residue else "clear",
            "combined": "blocked" if expected_extra or residue else "clear"})
        return core

    def d_manifest(self, mode, core):
        path = Path(str(STATE.request_path.resolve()) + "." + mode + ".ir-manifest.json")
        record = self.d_json(path)
        fields = ("record schema mode request_path scope_id profile_sha256 module hash_semantics "
                  "completeness olean_parts olean_extra_union ir_parts ir_occurrences ci_backed_names "
                  "unexplained_residue policy scientific_effect qualification classification_status "
                  "execution_authenticated")
        self.d_keys(record, fields)
        self.assertIs(type(record["schema"]), int)
        self.assertEqual(record["schema"], 1)
        self.assertEqual(record["record"], "sdga-ir-manifest")
        self.assertEqual(record["mode"], mode)
        self.assertEqual(record["request_path"], str(STATE.request_path.resolve()))
        for name in ("scope_id", "profile_sha256", "module"):
            self.assertEqual(record[name], STATE.request[name])
        self.assertEqual(record["hash_semantics"], "request-declared")
        self.assertEqual(record["completeness"], "complete")
        for name, wanted in (("scientific_effect", "NONE"), ("qualification", "BLOCKED"),
                             ("classification_status", "BLOCKED")):
            self.assertEqual(record[name], wanted)
        self.assertIs(record["execution_authenticated"], False)
        actual_core = {key: record[key] for key in core}
        self.assertEqual(self.d_core(actual_core), core)
        for part in STATE.request["olean_parts"] + STATE.request["ir_parts"]:
            measured = STATE.object_expected[part["path"]]
            self.assertTrue(measured["exists"])
            self.assertEqual(measured["sha256"], part["sha256"])
            self.assertEqual(digest(Path(part["path"]).read_bytes()), measured["sha256"])
        return path

    def d_envelope(self, label, mode, manifest_path):
        root = STATE.output / "ir-evidence"
        self.assertTrue(root.is_dir(), "IMPLEMENTATION_ABSENCE: production IR retention envelopes")
        matches = []
        for path in sorted(root.glob("*.json")):
            record = self.d_json(path)
            self.d_keys(record, "record schema mode request_path request_measurement manifest_path "
                        "manifest_before manifest_after manifest_completeness registration producer_label "
                        "native_outcome observation_error input_custody source_bindings policy validation_status "
                        "errors scientific_effect qualification classification_status execution_authenticated")
            self.assertEqual(record["record"], "sdga-ir-envelope")
            self.assertIs(type(record["schema"]), int)
            self.assertEqual(record["schema"], 1)
            self.assertIn(record["mode"], ("inventory", "replay"))
            self.assertIs(type(record["request_path"]), str)
            self.assertIs(type(record["manifest_path"]), str)
            for key in ("request_measurement", "manifest_before", "manifest_after"):
                self.d_state(record[key], nullable=True)
            self.assertIn(record["manifest_completeness"], ("complete", "missing", "invalid", "unreadable"))
            self.assertIn(record["registration"], ("new", "preexisting", "none"))
            for key in ("producer_label", "observation_error"):
                self.assertTrue(record[key] is None or type(record[key]) is str)
            self.assertIn(record["validation_status"], ("valid", "invalid"))
            self.assertIs(type(record["errors"]), list)
            for error in record["errors"]:
                self.d_keys(error, "stage code message")
                self.assertIn(error["stage"], ("admission", "before", "native", "after",
                                               "schema", "binding", "retention", "final"))
                self.assertIs(type(error["code"]), str)
                self.assertIs(type(error["message"]), str)
            if record["policy"] is not None:
                self.d_policy(record["policy"])
            custody = record["input_custody"]
            self.d_keys(custody, "path status sha256")
            self.assertIs(type(custody["path"]), str)
            self.assertIn(custody["status"], ("pass", "fail", "not_produced"))
            if custody["sha256"] is not None:
                self.d_hash(custody["sha256"])
            self.assertIs(type(record["source_bindings"]), list)
            source_names = []
            for source in record["source_bindings"]:
                self.d_keys(source, "path exists bytes sha256 git_blob")
                self.assertIs(type(source["path"]), str)
                self.assertIs(type(source["exists"]), bool)
                source_names.append(source["path"])
                if source["exists"]:
                    self.d_nat(source["bytes"])
                    self.d_hash(source["sha256"])
                    self.d_hash(source["git_blob"], 40)
                else:
                    self.assertIsNone(source["bytes"])
                    self.assertIsNone(source["sha256"])
                    self.assertIsNone(source["git_blob"])
            self.assertEqual(source_names, sorted(set(source_names)))
            for key, wanted in (("scientific_effect", "NONE"), ("qualification", "BLOCKED"),
                                ("classification_status", "BLOCKED")):
                self.assertEqual(record[key], wanted)
            self.assertIs(record["execution_authenticated"], False)
            native = record["native_outcome"]
            if native is not None:
                self.d_keys(native, "label argv cwd exit_code timed_out launch_error interruption "
                            "cleanup_confirmed stdout stderr")
                for key in ("label", "cwd"):
                    self.assertIs(type(native[key]), str)
                self.assertIs(type(native["argv"]), list)
                self.assertTrue(all(type(item) is str for item in native["argv"]))
                self.assertTrue(native["exit_code"] is None or type(native["exit_code"]) is int)
                self.assertIs(type(native["timed_out"]), bool)
                for key in ("launch_error", "interruption"):
                    self.assertTrue(native[key] is None or type(native[key]) is str)
                self.assertTrue(native["cleanup_confirmed"] is None
                                or type(native["cleanup_confirmed"]) is bool)
                for field in ("stdout", "stderr"):
                    stream = native[field]
                    self.d_keys(stream, "path bytes sha256")
                    self.assertIs(type(stream["path"]), str)
                    self.d_nat(stream["bytes"])
                    self.d_hash(stream["sha256"])
                if native["label"] == label:
                    matches.append((path, record))
        self.assertEqual(len(matches), 1, "missing/duplicate actual observation envelope")
        path, record = matches[0]
        native = record["native_outcome"]
        command = next(item for item in STATE.commands if item["label"] == label)
        self.assertEqual(record["mode"], mode)
        self.assertEqual(record["request_path"], str(STATE.request_path.resolve()))
        self.assertEqual(record["manifest_path"], str(manifest_path))
        self.assertEqual(record["request_measurement"], STATE.object_expected[str(STATE.request_path)])
        self.assertEqual(record["manifest_completeness"], "complete")
        self.assertEqual(record["validation_status"], "valid")
        if command["exit_code"] == 0:
            self.assertEqual(record["errors"], [])
        else:
            self.assertLessEqual(len(record["errors"]), 1,
                                 "collision envelope allows only zero or one native error")
            self.assertEqual(label, "cli-ir-sidecar-collision")
            for error in record["errors"]:
                self.assertEqual(error["stage"], "native")
                self.assertEqual(error["code"], "SIDECAR_COLLISION")
                self.assertEqual(error["message"],
                                 Path(command["stderr_path"]).read_bytes().decode("utf-8").strip())
        self.assertEqual(record["input_custody"]["status"], "pass")
        self.assertEqual(record["policy"], self.d_json(manifest_path)["policy"])
        self.assertIsNone(record["observation_error"])
        custody_path = Path(record["input_custody"]["path"])
        self.assertEqual(custody_path.resolve(), custody_path)
        self.assertEqual(custody_path.parent, STATE.custody)
        custody_bytes = custody_path.read_bytes()
        self.assertEqual(digest(custody_bytes), record["input_custody"]["sha256"])
        self.assertEqual(json.loads(custody_bytes)["native_label"], label)
        for key in ("argv", "cwd", "exit_code", "timed_out", "launch_error"):
            self.assertEqual(native[key], command[key])
        self.assertEqual(native["interruption"], command.get("interruption"))
        self.assertEqual(native["cleanup_confirmed"], command["cleanup"]["confirmed"])
        for field in ("stdout", "stderr"):
            self.assertEqual(native[field], {
                "path": command[field + "_path"], "bytes": command[field + "_bytes"],
                "sha256": command[field + "_sha256"]})
            raw = Path(native[field]["path"]).read_bytes()
            self.assertEqual((len(raw), digest(raw)),
                             (native[field]["bytes"], native[field]["sha256"]))
        expected_sources = STATE.identities["sources"]
        self.assertEqual({source["path"] for source in record["source_bindings"]}, set(expected_sources))
        for source in record["source_bindings"]:
            expected = expected_sources[source["path"]]
            self.assertEqual(source["exists"], expected["exists"])
            for key in ("bytes", "sha256", "git_blob"):
                self.assertEqual(source[key], expected.get(key))
        actual = manifest_path.read_bytes()
        self.assertEqual(record["manifest_after"],
                         {"exists": True, "bytes": len(actual), "sha256": digest(actual)})
        return record


    def d_base_files(self, rows):
        expected = []
        for family in ("olean", "ir"):
            for index, part in enumerate(STATE.request[family + "_parts"]):
                expected.append({"family": family, "part_index": index,
                                 "level": part["level"], "path": part["path"],
                                 "declared_sha256": part["sha256"]})
        self.assertEqual(rows, expected)

    def test_C11_actual_raw_olean_extra_refusal(self):
        rows = self.native_case("C11")
        detail = rows[0]["detail"]
        raw = detail["raw_setup"]["olean_parts"]
        extras = list(dict.fromkeys(name for part in raw for name in part["extra_const_names"]))
        self.assertEqual(detail["original_union"], extras)
        self.assertEqual(STATE.inventory["modules"][0]["extra_const_names"], extras)
        mutation = detail["mutation"]
        self.assertEqual(mutation["metadata_origin"], "altered_in_memory")
        self.assertEqual(mutation["base_request_path"], str(STATE.request_path))
        self.d_base_files(mutation["base_files"])
        change = mutation["alteration"]
        self.assertEqual(change["family"], "olean")
        self.assertEqual(change["part_index"], 2)
        self.assertEqual(change["field"], "extra_const_names")
        self.assertEqual(change["operation"], "append")
        self.assertEqual(mutation["after"], mutation["before"] + [change["appended_name"]])
        self.assertEqual(change["old_length"], len(mutation["before"]))
        self.assertEqual(rows[1]["detail"]["expected_code"], "UNSUPPORTED_EXTRAS")
        self.assertEqual(rows[1]["detail"]["reported_name"], change["appended_name"])

    def test_C14_actual_ir_provenance_and_residue(self):
        rows = self.native_case("C14")
        core = self.d_core(rows[0]["detail"]["core"])
        self.d_base_files(rows[0]["detail"]["base_files"])
        self.assertEqual(rows[1]["detail"]["core"], core)
        witness = rows[2]["detail"]
        self.assertEqual((witness["part_index"], witness["field"]), (1, "extra_const_names"))
        self.assertIs(witness["ci_backed"], True)
        self.assertEqual(core["ir_parts"][1]["extra_const_names"][witness["index"]], witness["name"])
        self.assertIn(witness["name"], {row["name"] for row in STATE.inventory["declarations"]})
        residue = rows[3]["detail"]
        for field in ("dependency_present", "owned_absent", "raw_absent"):
            self.assertIs(residue[field], True)
        self.assertEqual(residue["name"], STATE.task1.encode_name(("Nat",)))
        self.assertEqual(residue["expected_code"], "UNSUPPORTED_IR_RESIDUE")
        self.assertIn(residue["name"], residue["core"]["unexplained_residue"])
        self.assertNotIn(residue["name"], residue["core"]["ci_backed_names"])
        self.assertEqual(residue["core"]["policy"]["ir_residue"], "blocked")
        mutation = residue["mutation"]
        self.assertEqual(mutation["metadata_origin"], "altered_in_memory")
        self.assertEqual(mutation["base_request_path"], str(STATE.request_path))
        self.d_base_files(mutation["base_files"])
        self.assertEqual(mutation["after"], mutation["before"] + [residue["name"]])
        self.assertEqual(mutation["alteration"], {
            "family": "ir", "part_index": 1, "field": "extra_const_names", "operation": "append",
            "old_length": len(mutation["before"]), "appended_name": residue["name"]})
        inventory_path = self.d_manifest("inventory", core)
        replay_path = self.d_manifest("replay", core)
        self.assertNotEqual(inventory_path, replay_path)
        inventory_envelope = self.d_envelope("inventory", "inventory", inventory_path)
        replay_envelope = self.d_envelope("cli-safe-replay", "replay", replay_path)
        self.assertEqual(inventory_envelope["registration"], "new")
        self.assertEqual(replay_envelope["registration"], "new")
        self.assertEqual(inventory_envelope["manifest_before"], {"exists": False})
        self.assertEqual(replay_envelope["manifest_before"], {"exists": False})
        self.assertEqual(inventory_envelope["producer_label"], "inventory")
        self.assertEqual(replay_envelope["producer_label"], "cli-safe-replay")
        before = inventory_path.read_bytes()
        relative_request = os.path.relpath(STATE.request_path, FIXTURE)
        self.assertFalse(Path(relative_request).is_absolute())
        command, out, err = STATE.observe_lean(
            "cli-ir-sidecar-collision",
            ["--run", str(BRIDGE), "inventory", relative_request],
            requests=(STATE.request_path,))
        self.assertNotEqual(command["exit_code"], 0)
        self.assertEqual(out, b"", "collision must precede successful inventory/end")
        lines = err.decode("utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("SDGA_ERROR SIDECAR_COLLISION:"))
        self.assertEqual(inventory_path.read_bytes(), before, "native sidecar was overwritten")
        collision = self.d_envelope("cli-ir-sidecar-collision", "inventory", inventory_path)
        self.assertEqual(collision["registration"], "preexisting")
        self.assertEqual(collision["producer_label"], inventory_envelope["producer_label"])
        self.assertEqual(collision["manifest_before"], collision["manifest_after"])
        json_write(STATE.output / "task-d-sidecar-observation.json", {
            "record": "task-d-sidecar-observation", "native_collision": command,
            "relative_request_argument": relative_request,
            "inventory_path": str(inventory_path), "replay_path": str(replay_path),
            "inventory_bytes": len(before), "inventory_sha256": digest(before),
            "global_qualification": "BLOCKED", "outside_67_supplemental_collision": True})


class Result(unittest.TextTestResult):
    def startTestRun(self):
        self.method_rows = []
        super().startTestRun()

    def startTest(self, test):
        self.method_rows.append({"method": test.id(), "status": "running"})
        super().startTest(test)

    def note(self, test, status):
        for row in self.method_rows:
            if row["method"] == test.id():
                row["status"] = status
                return
        raise AssertionError("result without a started method")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.note(test, "failure")

    def addError(self, test, err):
        super().addError(test, err)
        self.note(test, "error")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.note(test, "failure" if issubclass(err[0], test.failureException) else "error")

    def addSuccess(self, test):
        super().addSuccess(test)
        self.note(test, "success")


def main():
    global STATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    STATE = Evidence(args.output)
    STATE.bootstrap()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TaskAContracts)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1,
                                    resultclass=Result).run(suite)
    for row in STATE.records.values():
        if row["status"] == "not_run":
            row["reason"] = (
                "implementation absence or failed prerequisite; body not executed"
                if STATE.bootstrap_errors or STATE.controller is None
                else "body not reached; inspect failed method")
    source_after = {}
    drift = []
    for relative, before in STATE.identities["sources"].items():
        path = ROOT / relative
        after = {"exists": path.is_file()}
        if after["exists"]:
            after["sha256"] = digest(path.read_bytes())
        source_after[relative] = after
        if before["exists"] != after["exists"] or (
                before["exists"] and before["sha256"] != after.get("sha256")):
            drift.append(relative)
    objects_complete = STATE.finalize_objects()
    STATE.identities["source_after"] = source_after
    STATE.identities["source_drift"] = drift
    STATE.flush()
    final = {
        "record": "task-a-b-c-d-result", "methods_run": result.testsRun,
        "method_results": result.method_rows,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "declared_subcases": 67,
        "original_task_a_methods": 6, "original_task_a_subcases": 39,
        "observed_subcases": sum(row["status"] != "not_run" for row in STATE.records.values()),
        "subcase_pass": sum(row["status"] == "pass" for row in STATE.records.values()),
        "subcase_fail": sum(row["status"] == "fail" for row in STATE.records.values()),
        "subcase_not_run": sum(row["status"] == "not_run" for row in STATE.records.values()),
        "scientific_effect": "NONE", "qualification": "BLOCKED",
        "classification_status": "BLOCKED", "execution_authenticated": False,
        "implementation_absence": STATE.identities["absent_production"],
        "source_drift": drift,
        "object_custody": STATE.object_final,
        "object_custody_failures": STATE.object_failures,
        "cli_replay": STATE.cli_replay,
        "cleanup_only_preflight": STATE.cleanup_preflight,
        "process_scheduling_blocked": STATE.scheduling_blocked,
    }
    json_write(STATE.output / "result.json", final)
    print(json.dumps(final, sort_keys=True), flush=True)
    complete = (
        result.wasSuccessful() and result.testsRun == 15 and not result.skipped
        and final["subcase_pass"] == 67 and final["subcase_fail"] == 0
        and final["subcase_not_run"] == 0 and not drift and objects_complete
        and final["cli_replay"]["status"] == "pass"
        and STATE.cleanup_preflight is not None
        and STATE.cleanup_preflight["status"] == "pass" and not STATE.scheduling_blocked)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
