#!/usr/bin/env python3
"""Explicit Task A hosted contracts. No production bridge/controller is supplied here.

Six unittest methods declare 39 subcases. Absence, compile/setup failure and unrun
subcases are evidence categories, never semantic rejections or passing coverage.
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
                executor = "Python_controller" if case_id == "C15" and subcase in PURE_C15 else "Lean"
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
        self.identities["declared_python_methods"] = 6
        self.identities["declared_subcases"] = 39
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
            "status": ("pass" if error is None and not drift and not self.object_failures
                       and len(self.object_observations) == 11
                       and len(self.request_bindings) == 6 else "fail"),
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
            target = self.owned / "Fixture" / "Safe.olean"
            target.parent.mkdir(parents=True)
            safe, _, _ = self.lean(
                "fixture-safe-compile",
                ["--root=" + str(FIXTURE), "-o", str(target),
                 "-i", str(target.with_suffix(".ilean")),
                 str(FIXTURE / "Fixture" / "Safe.lean")])
            if safe["exit_code"] != 0:
                self.bootstrap_errors.append("FIXTURE_COMPILE_FAILED")
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
            "declared_python_methods": 6, "declared_subcases": 39,
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
        "record": "task-a-result", "methods_run": result.testsRun,
        "method_results": result.method_rows,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "declared_subcases": 39,
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
        "cleanup_only_preflight": STATE.cleanup_preflight,
        "process_scheduling_blocked": STATE.scheduling_blocked,
    }
    json_write(STATE.output / "result.json", final)
    print(json.dumps(final, sort_keys=True), flush=True)
    complete = (
        result.wasSuccessful() and result.testsRun == 6 and not result.skipped
        and final["subcase_pass"] == 39 and final["subcase_fail"] == 0
        and final["subcase_not_run"] == 0 and not drift and objects_complete
        and STATE.cleanup_preflight is not None
        and STATE.cleanup_preflight["status"] == "pass" and not STATE.scheduling_blocked)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
