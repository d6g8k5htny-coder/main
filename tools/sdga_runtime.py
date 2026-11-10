"""Production-owned SDGA acquisition, custody and sequential process runtime.

No test module is imported here; fixture assertions remain in their test adapter.
"""
from __future__ import annotations

import copy
import contextlib
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sdga_component"
BRIDGE = ROOT / "tools" / "sdga" / "Component.lean"
CONTROLLER = ROOT / "tools" / "sdga_component.py"
RELEASE = "5045d0056413266e57c625dcd7c365b10e377c52"
DESIGN_BLOB = "3487f54476b5e6a283839902797bc05aae2ce623"
PLAN_BLOB = "9eefcb144247aa3f358b735956e580777e2b8945"
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
    """Execute captured canonical bytes; never resolve a package or cached bytecode."""
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("SOURCE_PATH: expected canonical source")
    raw = path.read_bytes()
    module = types.ModuleType(module_name)
    module.__file__, module.__package__ = str(path), None
    sys.modules[module_name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    if path.read_bytes() != raw:
        raise RuntimeError("SOURCE_DRIFT: source changed during captured execution")
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



def _load_ir_io_subject(expected=None):
    """Execute only captured bytes from the exact canonical production file."""
    subject = ROOT / "tools" / "sdga_ir_evidence.py"
    canonical = subject.resolve(strict=True)
    if canonical != subject.absolute():
        raise RuntimeError("IR_SOURCE_PATH: production helper alias is not admitted")
    raw = canonical.read_bytes()
    binding = {
        "path": str(canonical), "bytes": len(raw), "sha256": digest(raw),
        "git_blob": hashlib.sha1(
            b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest(),
    }
    if expected is not None and binding != expected:
        raise RuntimeError("IR_SOURCE_DRIFT: production helper changed after capture")
    if canonical.read_bytes() != raw:
        raise RuntimeError("IR_SOURCE_DRIFT: helper changed before captured execution")
    namespace = {"__name__": "sdga_ir_io_captured_subject",
                 "__file__": str(canonical), "__package__": None}
    exec(compile(raw, str(canonical), "exec"), namespace)
    api = {}
    for name in ("admit_output_root", "retain_ir_manifest"):
        if not callable(namespace.get(name)):
            raise RuntimeError("IR_SOURCE_API: missing " + name)
        api[name] = namespace[name]
    if canonical.read_bytes() != raw:
        raise RuntimeError("IR_SOURCE_DRIFT: helper changed during captured execution")
    return api, binding



def _append_ir_cause(error, prior):
    """Preserve existing evidence-error causes before attaching an earlier failure."""
    if prior is None or prior is error:
        return error
    seen = set()
    tail = error
    while tail.__cause__ is not None:
        if id(tail) in seen:
            error.add_note("Earlier evidence failure: " + type(prior).__name__ + ": " + str(prior))
            return error
        seen.add(id(tail))
        if tail.__cause__ is prior:
            return error
        tail = tail.__cause__
    seen.add(id(tail))
    node = prior
    prior_seen = set()
    while node is not None and id(node) not in prior_seen:
        if node is error or id(node) in seen:
            error.add_note("Earlier evidence failure: " + type(prior).__name__ + ": " + str(prior))
            return error
        prior_seen.add(id(node))
        node = node.__cause__
    tail.__cause__ = prior
    return error



_RUNTIMES = {}


class Runtime:
    def __init__(self, output):
        subject, self.ir_subject_binding = _load_ir_io_subject()
        self.source_roots = (ROOT.resolve(strict=True),)
        self.output = subject["admit_output_root"](output, self.source_roots)
        self.ir_expected = {}
        self.ir_contexts = []
        self.ir_observations = []
        self.ir_failures = []
        self.ir_final = None
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
        self.identities = {
            "record": "source-identities", "observed_utc": utc(),
            "scientific_effect": "NONE", "qualification": "BLOCKED",
            "classification_status": "BLOCKED", "execution_authenticated": False,
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"), "sources": {},
        }
        self.runtime_sources = {}
        source_paths = [Path(__file__).resolve(), CONTROLLER, BRIDGE,
                        ROOT / "tools/sdga_scope.py", ROOT / "tools/sdga_ir_evidence.py"]
        source_paths.extend(sorted(path for path in FIXTURE.rglob("*")
                                   if path.is_file() and ".lake" not in path.parts))
        for path in source_paths:
            if path.resolve(strict=True) != path:
                raise ValueError("SOURCE_PATH: acquisition source alias")
            raw = path.read_bytes()
            row = {"exists": True, "bytes": len(raw), "sha256": digest(raw),
                   "git_blob": hashlib.sha1(
                       b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()}
            relative = str(path.relative_to(ROOT))
            self.runtime_sources[relative] = row
            self.identities["sources"][relative] = copy.deepcopy(row)
        self.compiled_expected = {}
        self.compiled_producers = {}
        self.compiled_failures = []
        self.source_failures = []
        _RUNTIMES[str(self.output)] = self

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
        if self.ir_failures:
            raise RuntimeError("IR_EVIDENCE_BLOCKED: earlier generated-output failure")
        self.check_sources()
        self.check_compiled()
        self._check_generated("before-command-" + label)
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
        compile_source = None
        compile_outputs = ()
        if "lean" in entry["argv"] and "-o" in entry["argv"]:
            target = Path(entry["argv"][entry["argv"].index("-o") + 1])
            compile_source = Path(entry["argv"][-1])
            compile_outputs = (
                target, Path(str(target) + ".server"), Path(str(target) + ".private"),
                target.with_suffix(".ir.sig"), target.with_suffix(".ir"),
                target.with_suffix(".ilean"))
            entry["compile_source_path"] = str(compile_source)
            entry["compile_source_before"] = measure_objects((compile_source,))[str(compile_source)]
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
        if compile_source is not None:
            try:
                entry["compile_source_after"] = measure_objects((compile_source,))[str(compile_source)]
                entry["compile_outputs"] = measure_objects(compile_outputs)
                if entry["compile_source_after"] != entry["compile_source_before"]:
                    raise RuntimeError("SOURCE_DRIFT: native compile source changed")
            except BaseException as measurement_error:
                entry["compile_measurement_error"] = type(measurement_error).__name__ + ": " + str(measurement_error)
                if pending_error is None:
                    pending_error = measurement_error
                else:
                    _append_ir_cause(pending_error, measurement_error)
        # ru_maxrss is cumulative across observed children, not a per-process peak.
        self.commands.append(entry)
        json_write(self.output / "commands.json", self.commands)
        try:
            self.check_sources()
            self.check_compiled()
        except BaseException as source_error:
            if pending_error is not None:
                _append_ir_cause(pending_error, source_error)
                raise pending_error
            raise
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
        self.request_bindings[name]["part_producers"] = {
            part["path"]: self.producer_for(part["path"])
            for part in parts
            if any(part["path"] == str(candidate)
                   for runtime in _RUNTIMES.values()
                   for command in runtime.commands
                   if command["exit_code"] == 0 and "-o" in command["argv"]
                   for target in (Path(command["argv"][command["argv"].index("-o") + 1]),)
                   for candidate in (target, Path(str(target) + ".server"),
                                     Path(str(target) + ".private"),
                                     target.with_suffix(".ir.sig"), target.with_suffix(".ir")))
        }
        self.flush()

    def observe_lean(self, label, lean_args, *, requests, search_root=None, env_extra=None):
        if self.object_failures:
            raise RuntimeError("OBJECT_CUSTODY_BLOCKED: earlier custody failure")
        if self.ir_failures:
            raise RuntimeError("IR_EVIDENCE_BLOCKED: earlier generated-output failure")
        names = [str(Path(path).resolve()) for path in requests]
        if any(name not in self.request_objects for name in names):
            raise ValueError("OBJECT_REQUEST_UNBOUND: fixed observation request")
        paths = tuple(sorted({path for name in names for path in self.request_objects[name]},
                             key=str))
        expected = {str(path): copy.deepcopy(self.object_expected[str(path)]) for path in paths}
        record_path = self.custody / ("%02d-%s.json" %
                                     (len(self.object_observations) + 1, label))
        pending = None
        context = None
        try:
            context = self._prepare_ir(label, lean_args, names)
            self._check_generated("before-" + label)
            with object_custody(paths, expected, record_path):
                return self.lean(label, lean_args, search_root=search_root, env_extra=env_extra)
        except BaseException as error:
            pending = error
            raise
        finally:
            secondary = None
            command = self.commands[-1] if self.commands and self.commands[-1]["label"] == label else None
            try:
                if record_path.is_file():
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                else:
                    record = {"status": "fail", "error": "custody record not produced"}
                record["native_label"] = label
                record["native_outcome"] = None if command is None else {
                    key: command[key] for key in
                    ("exit_code", "timed_out", "launch_error", "stdout_sha256", "stderr_sha256")
                }
                json_write(record_path, record)
                self.object_observations.append(record)
                if record["status"] != "pass":
                    self.object_failures.append(str(record_path))
            except BaseException as error:
                self.object_failures.append(str(record_path) + ": " + str(error))
                secondary = error
            try:
                # A before-measurement error still has its attempted CLI context.
                if context is None and self.ir_contexts and self.ir_contexts[-1]["label"] == label:
                    context = self.ir_contexts[-1]
                if context is not None:
                    original = pending if pending is not None else secondary
                    self._retain_ir(
                        context, command, record_path,
                        None if original is None else type(original).__name__ + ": " + str(original))
            except BaseException as error:
                if not any(row.get("native_label") == label for row in self.ir_failures):
                    self.ir_failures.append({
                        "stage": "retention-finally", "native_label": label,
                        "error": type(error).__name__ + ": " + str(error)})
                _append_ir_cause(error, secondary)
                secondary = error
            try:
                self._check_generated("after-" + label)
            except BaseException as error:
                _append_ir_cause(error, secondary)
                secondary = error
            try:
                self.flush()
            except BaseException as error:
                self.ir_failures.append({"stage": "metadata", "native_label": label,
                                         "error": type(error).__name__ + ": " + str(error)})
                _append_ir_cause(error, secondary)
                secondary = error
            if secondary is not None:
                if pending is not None:
                    _append_ir_cause(pending, secondary)
                    raise pending
                raise secondary

    def _generated_state(self, path):
        path = Path(path)
        if not path.is_absolute() or not path.is_relative_to(self.output):
            raise OSError("IR_OUTPUT_PATH: expected canonical output child")
        try:
            info = path.lstat()
        except FileNotFoundError:
            return {"exists": False}
        if not stat.S_ISREG(info.st_mode) or path.resolve(strict=True) != path:
            raise OSError("IR_OUTPUT_TYPE: expected a regular canonical artifact")
        raw = path.read_bytes()
        return {"exists": True, "bytes": len(raw), "sha256": digest(raw)}

    def _register_generated(self, path, label, kind):
        path = Path(path)
        state = self._generated_state(path)
        if not state["exists"]:
            return
        name = str(path)
        prior = self.ir_expected.get(name)
        if prior is not None:
            if state != prior["measurement"]:
                raise RuntimeError("IR_OUTPUT_DRIFT: cannot replace expected bytes: " + name)
            return
        self.ir_expected[name] = {
            "measurement": copy.deepcopy(state), "producer_label": label, "kind": kind}

    def _check_generated(self, stage):
        after = {}
        drift = []
        measurement_errors = []
        for name, expected in self.ir_expected.items():
            try:
                after[name] = self._generated_state(Path(name))
                if after[name] != expected["measurement"]:
                    drift.append(name)
            except OSError as error:
                after[name] = None
                measurement_errors.append(name + ": " + type(error).__name__ + ": " + str(error))
        if drift or measurement_errors:
            self.ir_failures.append({"stage": stage, "drift": drift,
                                     "measurement_errors": measurement_errors})
            raise RuntimeError("IR_OUTPUT_DRIFT: " + stage)
        return after

    def _prepare_ir(self, label, lean_args, requested_names):
        if (len(lean_args) != 4 or lean_args[0] != "--run"
                or Path(lean_args[1]).resolve() != BRIDGE
                or lean_args[2] not in ("inventory", "replay")):
            return None
        argument = Path(lean_args[3])
        if not argument.is_absolute():
            # lake/lean runs from FIXTURE, including the frozen relative probe.
            argument = FIXTURE / argument
        origin = argument.resolve()
        if str(origin) not in requested_names or str(origin) not in self.request_objects:
            raise ValueError("IR_REQUEST_UNBOUND: CLI request differs from guarded inputs")
        if not origin.is_relative_to(self.output):
            raise ValueError("IR_REQUEST_PATH: request is outside the admitted output root")
        mode = lean_args[2]
        manifest = Path(str(origin) + "." + mode + ".ir-manifest.json")
        ordinal = len(self.object_observations) + 1
        envelope = self.output / "ir-evidence" / ("%02d-%s.json" % (ordinal, label))
        prior = self.ir_expected.get(str(manifest))
        context = {
            "label": label, "ordinal": ordinal, "mode": mode,
            "request_path": origin, "manifest_path": manifest, "envelope_path": envelope,
            "manifest_before": None, "manifest_before_error": None,
            "producer_label": None if prior is None else prior["producer_label"],
            "requires_manifest": False,
        }
        self.ir_contexts.append(context)
        try:
            context["manifest_before"] = self._generated_state(manifest)
        except OSError as error:
            context["manifest_before_error"] = type(error).__name__ + ": " + str(error)
            raise
        return context

    def _retain_ir(self, context, command, custody_path, observation_error):
        pending = None
        envelope = None
        transport = {
            "label": context["label"], "ordinal": context["ordinal"],
            "command": copy.deepcopy(command),
            "manifest_before": copy.deepcopy(context["manifest_before"]),
            "manifest_before_error": context["manifest_before_error"],
            "producer_label": context["producer_label"],
        }
        context["requires_manifest"] = command is not None and command["exit_code"] == 0
        try:
            subject, _ = _load_ir_io_subject(self.ir_subject_binding)
            envelope = subject["retain_ir_manifest"](
                mode=context["mode"], request_path=context["request_path"],
                output_root=self.output, source_roots=self.source_roots,
                native_outcome=transport, custody_record=custody_path,
                source_bindings=copy.deepcopy(self.identities["sources"]),
                observation_error=observation_error)
            self.ir_observations.append({
                "native_label": context["label"], "mode": context["mode"],
                "manifest_path": str(context["manifest_path"]),
                "envelope_path": str(context["envelope_path"]),
                "validation_status": envelope["validation_status"],
                "manifest_completeness": envelope["manifest_completeness"],
                "registration": envelope["registration"],
            })
            if envelope["validation_status"] != "valid":
                raise RuntimeError("IR_EVIDENCE_INVALID: " + context["label"])
        except BaseException as error:
            pending = error
            self.ir_failures.append({"stage": "retention", "native_label": context["label"],
                                     "error": type(error).__name__ + ": " + str(error)})
            raise
        finally:
            try:
                # Preserve even failed/partial bytes, without replacing a first expectation.
                manifest_label = (envelope.get("producer_label") if envelope is not None
                                  else context["producer_label"])
                if context["manifest_before"] == {"exists": False}:
                    manifest_label = context["label"]
                self._register_generated(context["manifest_path"], manifest_label, "manifest")
                self._register_generated(context["envelope_path"], context["label"], "envelope")
            except BaseException as registration_error:
                self.ir_failures.append({
                    "stage": "registration", "native_label": context["label"],
                    "error": type(registration_error).__name__ + ": " + str(registration_error)})
                if pending is not None:
                    _append_ir_cause(pending, registration_error)
                    raise pending
                raise

    def flush(self):
        self.identities["ir_evidence"] = {
            "expected_generated_outputs": self.ir_expected,
            "contexts": [{
                "native_label": row["label"], "ordinal": row["ordinal"], "mode": row["mode"],
                "request_path": str(row["request_path"]),
                "manifest_path": str(row["manifest_path"]),
                "envelope_path": str(row["envelope_path"]),
                "manifest_before": row["manifest_before"],
                "manifest_before_error": row["manifest_before_error"],
                "previous_producer_label": row["producer_label"],
                "requires_manifest": row["requires_manifest"],
            } for row in self.ir_contexts],
            "observations": self.ir_observations, "failures": self.ir_failures,
            "final": self.ir_final, "outside_method_and_subcase_census": True,
        }
        json_write(self.output / "ir-evidence-registry.json", self.identities["ir_evidence"])
        self.identities["object_custody"] = {
            "expected": self.object_expected, "request_bindings": self.request_bindings,
            "observations": self.object_observations, "failures": self.object_failures,
            "final": self.object_final, "outside_lean_method_and_subcase_census": True,
        }
        json_write(self.output / "object-custody.json", self.identities["object_custody"])
        json_write(self.output / "source-identities.json", self.identities)


    def check_sources(self):
        failures = []
        try:
            for relative, before in self.runtime_sources.items():
                path = ROOT / relative
                if path.resolve(strict=True) != path:
                    failures.append(relative + ": noncanonical source")
                    continue
                actual = measure_objects((path,))[str(path)]
                expected = {key: before[key] for key in ("exists", "bytes", "sha256")}
                if actual != expected:
                    failures.append(relative)
        except OSError as error:
            self.source_failures.append({
                "stage": "measurement", "paths": failures,
                "error": type(error).__name__ + ": " + str(error)})
            try:
                json_write(self.output / "source-failures.json", self.source_failures)
            except BaseException as retention_error:
                _append_ir_cause(error, retention_error)
            raise
        if failures:
            self.source_failures.append({"stage": "native-use", "paths": failures})
            json_write(self.output / "source-failures.json", self.source_failures)
            raise RuntimeError("SOURCE_DRIFT: production acquisition inputs")
        if self.source_failures:
            raise RuntimeError("SOURCE_DRIFT: earlier production source failure")

    def check_compiled(self):
        try:
            if self.compiled_expected:
                actual = {name: self._generated_state(name) for name in self.compiled_expected}
                if actual != self.compiled_expected:
                    self.compiled_failures.append("compiled output bytes changed")
        except OSError as error:
            self.compiled_failures.append(
                "measurement: " + type(error).__name__ + ": " + str(error))
            try:
                json_write(self.output / "compiled-output-failures.json", self.compiled_failures)
            except BaseException as retention_error:
                _append_ir_cause(error, retention_error)
            raise
        if self.compiled_failures:
            json_write(self.output / "compiled-output-failures.json", self.compiled_failures)
            raise RuntimeError("COMPILED_OUTPUT_DRIFT: " + "; ".join(self.compiled_failures))

    def producer_for(self, path):
        """Admit one complete healthy origin; failed origins are never filtered away."""
        path = Path(path)
        origins = []
        for runtime in _RUNTIMES.values():
            for command in runtime.commands:
                argv = command["argv"]
                if "-o" not in argv or "lean" not in argv:
                    continue
                target = Path(argv[argv.index("-o") + 1])
                outputs = (target, Path(str(target) + ".server"),
                           Path(str(target) + ".private"), target.with_suffix(".ir.sig"),
                           target.with_suffix(".ir"), target.with_suffix(".ilean"))
                if path in outputs:
                    origins.append((runtime, command, outputs))
        if len(origins) != 1:
            raise ValueError("PART_PRODUCER: expected exactly one native origin: "
                             + str(path) + "; origins="
                             + repr([(str(row.output), cmd["label"])
                                     for row, cmd, _ in origins]))
        runtime, command, outputs = origins[0]

        def refuse(reason):
            runtime.compiled_failures.append("producer: " + reason)
            raise RuntimeError("PART_PRODUCER_FAILED: " + reason)

        if (runtime.scheduling_blocked or runtime.identities.get("setup_error")
                or runtime.source_failures or runtime.compiled_failures
                or runtime.object_failures or runtime.ir_failures):
            refuse("origin context already failed: " + str(runtime.output))
        runtime.check_sources()
        runtime.check_compiled()
        cleanup = command.get("cleanup")
        if (type(command.get("exit_code")) is not int or command["exit_code"] != 0
                or command.get("timed_out") is not False
                or command.get("launch_error") is not None
                or command.get("interruption") is not None
                or command.get("compile_measurement_error") is not None
                or type(cleanup) is not dict or cleanup.get("confirmed") is not True
                or cleanup.get("leader_reaped") is not True
                or cleanup.get("group_absent") is not True or cleanup.get("errors") != []):
            refuse("native outcome/cleanup incomplete: " + command["label"])
        source_path = command.get("compile_source_path")
        source_expected = command.get("compile_source_before")
        expected = command.get("compile_outputs")
        source = Path(source_path) if isinstance(source_path, str) else None
        if source == BRIDGE:
            # Ordinary Component has no module header: two products, four absences.
            present = {str(outputs[0]), str(outputs[5])}
        elif source in {FIXTURE / "Fixture" / (leaf + ".lean")
                        for leaf in ("BadDependency", "Safe", "AxiomUse",
                                     "Skipped", "OwnedAxiom")}:
            present = set(map(str, outputs))
        else:
            refuse("source is outside the fixed bridge/fixture families")
        if (source_path != command["argv"][-1]
                or type(source_expected) is not dict
                or source_expected.get("exists") is not True
                or command.get("compile_source_after") != source_expected
                or type(expected) is not dict or set(expected) != set(map(str, outputs))
                or str(path) not in present
                or any(type(value) is not dict
                       or value.get("exists") is not (name in present)
                       for name, value in expected.items())):
            refuse("complete fixed source/output capture absent: " + command["label"])
        try:
            if (source.resolve(strict=True) != source
                    or any(output.resolve() != output for output in outputs)):
                refuse("source/output alias: " + command["label"])
            current_source = measure_objects((source,))[str(source)]
            current_outputs = measure_objects(outputs)
        except OSError as error:
            runtime.compiled_failures.append(
                "producer measurement: " + type(error).__name__ + ": " + str(error))
            raise
        if current_source != source_expected or current_outputs != expected:
            refuse("native source/six-output bytes differ: " + command["label"])
        return {
            "evidence_root": str(runtime.output), "command_label": command["label"],
            "command": copy.deepcopy(command), "source_path": str(source),
            "source_measurement": copy.deepcopy(source_expected),
            "output_measurement": copy.deepcopy(expected[str(path)]),
        }

def acquisition_runtime(evidence_root):
    """One lazy production runtime per admitted public evidence root."""
    root = Path(evidence_root)
    if not root.is_absolute() or root.absolute() != root:
        raise ValueError("OUTPUT_ROOT: absolute canonical evidence root required")
    existing = _RUNTIMES.get(str(root))
    if existing is not None:
        if root.resolve(strict=True) != root:
            raise ValueError("OUTPUT_ROOT: evidence alias")
        if existing.object_final is not None or existing.ir_final is not None:
            raise ValueError("ACQUISITION_FINALIZED: no observation may extend a closed registry")
        if existing.identities.get("setup_error"):
            raise RuntimeError("ACQUISITION_SETUP: earlier initialization failure")
        existing.check_sources()
        existing.check_compiled()
        return existing
    state = Runtime(root)
    state.flush()
    try:
        state.enable_subreaper()
        state.cleanup_only_preflight()
        for expression, key in (("HEAD^{commit}", "main_commit"), ("HEAD^{tree}", "main_tree")):
            command, out, err = state.command(
                key, ["git", "rev-parse", "--verify", expression], cwd=ROOT)
            if command["exit_code"] != 0 or err:
                raise RuntimeError("MAIN_SOURCE_IDENTITY: " + key)
            value = out.decode("ascii").strip()
            if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("MAIN_SOURCE_IDENTITY: malformed native Git identity")
            state.identities[key] = value
        command, out, err = state.command("lean-githash", ["lean", "--githash"])
        state.identities["actual_lean_githash"] = out.decode("ascii").strip()
        if command["exit_code"] != 0 or err or state.identities["actual_lean_githash"] != RELEASE:
            raise RuntimeError("TOOLCHAIN_IDENTITY: pinned Lean release differs")
    except BaseException as error:
        state.identities["setup_error"] = type(error).__name__ + ": " + str(error)
        state.scheduling_blocked = True
        raise
    finally:
        state.flush()
    return state
