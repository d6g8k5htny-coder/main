"""Main registered unittest census, original-suite execution and inert evidence join.

Native/API-shaped dictionaries establish consistency only, never HTTP authority.
This trusted-process recorder is not hostile-test containment. This module has
no acquisition, dependency installation, browser/Lean setup or scientific gate.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
import zlib

MIB = 1024 * 1024
SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROTECTED = {
    ".github/workflows/research-architecture.yml": "69ad3b47537ab8e4f8936affa3e09fccbdcfd1e5",
    "tools/unittest_timing_probe.py": "05fb481b5538c3f517174b00948de45d2849387a",
    "tools/architecture_run_receipt.py": "a6062afa9261958b8d8b85853a9c43031385b2c1",
    "architecture/graph_artifact_readback.py": "c39ffd0fdcde1fd5f895c57df5f770a582918a8a",
    "tools/architecture_check_binding.py": "53ea34e4fffc1087770584744939904844a41cff",
    "tests/test_unittest_full_qualification.py": "2fe09abce9a8ef9356fd0af85780dcde75179386",
    ".github/workflows/full-unittest-qualification.yml": "ff8226cdeea57ee2ddb41c58c8c93e644343557d",
}
CORE_MEMBERS = {
    "source-before.json": 8 * MIB, "source-after.json": 8 * MIB,
    "full-census.json": 4 * MIB, "architecture-census.json": 4 * MIB,
    "events.jsonl": 16 * MIB, "stdout.bin": 16 * MIB,
    "stderr.bin": 16 * MIB, "core-result.json": MIB,
    "ARTIFACT_SHA256SUMS": 16 * 1024,
}
_CONTEXT_KEYS = {"repository", "checked_commit", "checked_tree", "run_id", "run_attempt"}


def _fail(reason):
    raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def _encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _json(raw, limit):
    if type(raw) is not bytes or len(raw) > limit:
        _fail("INVALID_RECORD")

    def unique(rows):
        result = {}
        for key, value in rows:
            if key in result:
                _fail("INVALID_RECORD")
            result[key] = value
        return result

    def bad(value):
        _fail("INVALID_RECORD")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=bad)
        def bounded(item, depth=0):
            if depth > 64:
                _fail("INVALID_RECORD")
            if type(item) is float and not math.isfinite(item):
                _fail("INVALID_RECORD")
            if type(item) is dict:
                for key, child in item.items():
                    bounded(child, depth + 1)
            elif type(item) is list:
                for child in item:
                    bounded(child, depth + 1)
        bounded(value)
        return value
    except (UnicodeError, json.JSONDecodeError, RecursionError, OverflowError):
        _fail("INVALID_RECORD")


def _context(value):
    if type(value) is not dict or set(value) != _CONTEXT_KEYS:
        _fail("INVALID_RECORD")
    if any(type(item) is not str for item in value.values()):
        _fail("INVALID_RECORD")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value["repository"]):
        _fail("INVALID_RECORD")
    if any(part in (".", "..") for part in value["repository"].split("/")):
        _fail("INVALID_RECORD")
    for name in ("checked_commit", "checked_tree"):
        if not re.fullmatch(r"[0-9a-f]{40}", value[name]):
            _fail("INVALID_RECORD")
    for name in ("run_id", "run_attempt"):
        if not re.fullmatch(r"[1-9][0-9]*", value[name]):
            _fail("INVALID_RECORD")
    return dict(value)


def _git(root, *args):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                       GIT_NO_REPLACE_OBJECTS="1", GIT_OPTIONAL_LOCKS="0")
    try:
        return subprocess.check_output(
            ["git", "--no-replace-objects", "--literal-pathspecs", "-c",
             "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
             "-C", str(root), *args], env=environment, stderr=subprocess.PIPE,
            timeout=20)
    except (OSError, subprocess.SubprocessError):
        _fail("SOURCE_MISMATCH")


def _directory(path):
    path = Path(path).absolute()
    if ".." in path.parts or path != path.resolve() or not path.is_dir():
        _fail("SOURCE_MISMATCH")
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor /= part
        if not stat.S_ISDIR(cursor.lstat().st_mode):
            _fail("SOURCE_MISMATCH")
    return path


def _read(path, limit):
    path = Path(path).absolute()
    _directory(path.parent)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                    or before.st_size > limit):
                _fail("INVALID_RECORD")
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
            fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                      "st_mtime_ns", "st_ctime_ns")
            if (len(raw) != before.st_size or len(raw) > limit
                    or any(getattr(before, name) != getattr(after, name) for name in fields)):
                _fail("INVALID_RECORD")
            return raw
    except OSError:
        _fail("INCOMPLETE")


def _write(path, raw):
    _directory(Path(path).parent)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()


def _source(root, context):
    context = _context(context)
    root = _directory(root)
    if (_git(root, "rev-parse", "--show-toplevel").decode().strip() != str(root)
            or _git(root, "rev-parse", "--verify", "HEAD").decode().strip()
            != context["checked_commit"]
            or _git(root, "rev-parse", "--verify", "HEAD^{tree}").decode().strip()
            != context["checked_tree"]):
        _fail("SOURCE_MISMATCH")
    files = []
    for row in _git(root, "ls-tree", "-r", "--full-tree", "-z",
                    context["checked_commit"]).split(b"\0"):
        if not row:
            continue
        metadata, name = row.split(b"\t", 1)
        mode, kind, identity = metadata.decode("ascii").split()
        relative = name.decode("utf-8")
        if kind != "blob" or mode not in ("100644", "100755"):
            _fail("SOURCE_MISMATCH")
        path = root / relative
        raw = _read(path, 256 * MIB)
        if (_blob(raw) != identity
                or bool(path.stat().st_mode & stat.S_IXUSR) != (mode == "100755")):
            _fail("SOURCE_MISMATCH")
        files.append({"path": relative, "mode": mode, "git_blob": identity,
                      "bytes": len(raw), "sha256": _sha(raw)})
    if (not files or _git(root, "ls-files", "--others", "--exclude-standard", "-z")
            or _git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")):
        _fail("SOURCE_MISMATCH")
    return {"schema": "main-unittest.source.v1", "context": context, "files": files}


def _project_snapshot():
    root = _directory(SOURCE_ROOT)
    if _git(root, "rev-parse", "--show-toplevel").decode().strip() != str(root):
        _fail("SOURCE_MISMATCH")
    if (_git(root, "ls-files", "--others", "--exclude-standard", "-z")
            or _git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")):
        _fail("SOURCE_MISMATCH")
    paths = set(PROTECTED) | {"tools/unittest_full_qualification.py",
                              "architecture/__init__.py"}
    entries = {}
    for row in _git(root, "ls-tree", "-r", "--full-tree", "-z", "HEAD",
                    "--", *sorted(paths)).split(b"\0"):
        if row:
            metadata, path = row.split(b"\t", 1)
            mode, kind, identity = metadata.decode("ascii").split()
            entries[path.decode()] = (mode, kind, identity)
    if set(entries) != paths:
        _fail("SOURCE_MISMATCH")
    records = []
    for relative in sorted(paths):
        mode, kind, identity = entries[relative]
        raw = _read(root / relative, 4 * MIB)
        if (kind != "blob" or mode not in ("100644", "100755") or _blob(raw) != identity
                or relative in PROTECTED and identity != PROTECTED[relative]
                or bool((root / relative).stat().st_mode & stat.S_IXUSR) != (mode == "100755")):
            _fail("SOURCE_MISMATCH")
        records.append((relative, mode, identity, len(raw), _sha(raw)))
    for name, relative in (("tools", "tools"), ("architecture", "architecture")):
        module = sys.modules.get(name)
        if module is not None:
            if not getattr(module, "__path__", None):
                _fail("SOURCE_MISMATCH")
            if any(Path(item).resolve() != root / relative for item in module.__path__):
                _fail("SOURCE_MISMATCH")
            if name == "architecture" and Path(module.__file__).resolve() != root / "architecture/__init__.py":
                _fail("SOURCE_MISMATCH")
    for name, relative in (
            ("architecture.graph_artifact_readback", "architecture/graph_artifact_readback.py"),
            ("tools.unittest_timing_probe", "tools/unittest_timing_probe.py")):
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve() != root / relative:
            _fail("SOURCE_MISMATCH")
    return (_git(root, "rev-parse", "--verify", "HEAD").decode().strip(),
            _git(root, "rev-parse", "--verify", "HEAD^{tree}").decode().strip(),
            records)


_PROJECT_STAMP = _project_snapshot()


def _load_project(relative, name):
    raw = _read(SOURCE_ROOT / relative, 4 * MIB)
    if _blob(raw) != PROTECTED[relative]:
        _fail("SOURCE_MISMATCH")
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE_ROOT / relative)
    module.__package__ = ""
    exec(compile(raw, module.__file__, "exec", dont_inherit=True), module.__dict__)
    return module


_RECORDER = _load_project("tools/unittest_timing_probe.py", "_main_unittest_native_recorder")
_READER = _load_project("architecture/graph_artifact_readback.py", "_main_unittest_archive_reader")
Journal = _RECORDER.Journal
runner_type = _RECORDER.runner_type
read_graph_archive = _READER.read_graph_archive


def _captured_source_guard():
    if (_project_snapshot() != _PROJECT_STAMP
            or Journal is not _RECORDER.Journal or runner_type is not _RECORDER.runner_type
            or read_graph_archive is not _READER.read_graph_archive):
        _fail("SOURCE_MISMATCH")


def _runtime():
    result = {
        "implementation": sys.implementation.name, "python": sys.version,
        "executable_sha256": _sha(Path(sys.executable).resolve().read_bytes()),
        "optimize": sys.flags.optimize, "no_site": sys.flags.no_site,
        "dont_write_bytecode": sys.dont_write_bytecode,
    }
    if (result["implementation"] != "cpython" or type(result["optimize"]) is not int
            or result["optimize"] not in (0, 1) or result["no_site"] != 1
            or result["dont_write_bytecode"] is not True or sys.flags.isolated):
        _fail("INVALID_RECORD")
    return result


def _outside(destination, root):
    destination = Path(destination).absolute()
    _directory(destination.parent)
    if ".." in destination.parts or destination.exists() or destination.is_symlink():
        _fail("INVALID_RECORD")
    for source in (_directory(root), SOURCE_ROOT):
        if destination == source or source in destination.parents:
            _fail("SOURCE_MISMATCH")
    return destination


def _flags():
    _runtime()
    return ["-B"] + (["-O"] if sys.flags.optimize else []) + ["-S"]


def _children(root, context, operation, destination, architecture=None):
    with tempfile.TemporaryDirectory(prefix="main-unittest-control-") as raw:
        control = Path(raw) / "control.json"
        _write(control, _encode({"context": context, "architecture": architecture}))
        command = [sys.executable, *_flags(), str(SOURCE_ROOT / "tools/unittest_full_qualification.py"),
                   operation, str(root), str(control), str(destination)]
        environment = dict(os.environ)
        for name in ("PYTHONHOME", "PYTHONOPTIMIZE"):
            environment.pop(name, None)
        environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        stdout_context = ((destination / "stdout.bin").open("xb")
                          if operation == "_core_child" else tempfile.TemporaryFile())
        stderr_context = ((destination / "stderr.bin").open("xb")
                          if operation == "_core_child" else tempfile.TemporaryFile())
        with stdout_context as stdout:
            with stderr_context as stderr:
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=stdout,
                                           stderr=stderr, start_new_session=True)
                interrupted = False
                try:
                    code = process.wait(timeout=240)
                except subprocess.TimeoutExpired:
                    interrupted = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                    code = process.returncode
        return code, interrupted


def _suite_rows(suite, loader, root, context, source_sha256, profile):
    if loader.errors or type(suite) is not unittest.TestSuite:
        _fail("INVENTORY_MISMATCH")
    rows, cases, branches, identities = [], [], [], set()

    def walk(node, index):
        if type(node) is unittest.TestSuite:
            if getattr(node.run, "__func__", None) is not unittest.TestSuite.run:
                _fail("UNSAFE_PARTITION")
            result = []
            for child in node:
                result.extend(walk(child, index))
            return result
        if not isinstance(node, unittest.TestCase) or type(node).__name__ == "_FailedTest":
            _fail("UNSAFE_PARTITION")
        identity = node.id()
        if type(identity) is not str or not identity or len(identity.encode("utf-8")) > 1024 or identity in identities:
            _fail("INVENTORY_MISMATCH")
        identities.add(identity)
        module = sys.modules.get(type(node).__module__)
        if module is None or not getattr(module, "__file__", None):
            _fail("SOURCE_MISMATCH")
        path = Path(module.__file__).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            _fail("SOURCE_MISMATCH")
        if not path.is_file() or path.suffix != ".py":
            _fail("SOURCE_MISMATCH")
        rows.append({"id": identity, "module": type(node).__module__,
                     "class": type(node).__module__ + "." + type(node).__qualname__,
                     "source_path": relative, "branch": index})
        cases.append(node)
        return [node]

    for index, child in enumerate(suite):
        branches.append((child, walk(child, index)))
    if not rows:
        _fail("INVENTORY_MISMATCH")
    return ({"schema": "main-unittest.inventory.v1", "context": context,
             "source_sha256": source_sha256, "mode": "optimized" if sys.flags.optimize else "normal",
             "profile": profile, "discovery": {"start_dir": "tests",
             "pattern": "test_architecture_*.py" if profile == "architecture" else "test*.py",
             "top_level_dir": None}, "runtime": _runtime(), "tests": rows,
             "count": len(rows), "loader_errors": []}, branches, cases)


def _discover(root, context, source_sha256, profile):
    os.chdir(root)
    sys.path[0] = str(root)
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(1, str(SOURCE_ROOT))
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_architecture_*.py"
                            if profile == "architecture" else "test*.py", top_level_dir=None)
    return _suite_rows(suite, loader, root, context, source_sha256, profile)


def _partition(full, branches, cases, architecture):
    if type(architecture) is not dict:
        _fail("INVALID_RECORD")
    shared = ("id", "module", "class", "source_path")
    actual = architecture.get("tests")
    if type(actual) is not list or not actual:
        _fail("INVENTORY_MISMATCH")
    full_by_id = {row["id"]: row for row in full["tests"]}
    arch_ids = [row["id"] for row in actual]
    if len(set(arch_ids)) != len(arch_ids):
        _fail("INVENTORY_MISMATCH")
    for row in actual:
        match = full_by_id.get(row["id"])
        if match is None or any(row[key] != match[key] for key in shared):
            _fail("INVENTORY_MISMATCH")
    selected_arch = set(arch_ids)
    if [row["id"] for row in full["tests"] if row["id"] in selected_arch] != arch_ids:
        _fail("INVENTORY_MISMATCH")
    module_profiles, class_profiles = {}, {}
    for row, case in zip(full["tests"], cases):
        profile = row["id"] in selected_arch
        module_profiles.setdefault(type(case).__module__, set()).add(profile)
        class_profiles.setdefault(type(case), set()).add(profile)
    if any(len(value) != 1 for value in [*module_profiles.values(), *class_profiles.values()]):
        _fail("UNSAFE_PARTITION")
    retained, selected = [], []
    for child, leaves in branches:
        flags = {case.id() in selected_arch for case in leaves}
        if len(flags) > 1:
            _fail("UNSAFE_PARTITION")
        if flags == {False}:
            retained.append(child)
            selected.extend(leaves)
    positions = {id(case): index for index, case in enumerate(cases)}
    for left, right in zip(selected, selected[1:]):
        if (positions[id(right)] > positions[id(left)] + 1
                and type(left).__module__ == type(right).__module__):
            _fail("UNSAFE_PARTITION")
    if not selected:
        _fail("INVENTORY_MISMATCH")
    return unittest.TestSuite(retained), [case.id() for case in selected]


def architecture_inventory(root, context, destination):
    """Retain native architecture-pattern population from a fresh process."""
    _captured_source_guard()
    context, root = _context(context), _directory(root)
    before = _source(root, context)
    destination = _outside(destination, root)
    with tempfile.TemporaryDirectory(prefix="main-unittest-inventory-") as raw:
        target = Path(raw) / "architecture.json"
        code, interrupted = _children(root, context, "_architecture_child", target)
        if interrupted or code != 0 or not target.is_file():
            _fail("INVENTORY_MISMATCH")
        value = _json(_read(target, 4 * MIB), 4 * MIB)
    if (_source(root, context) != before or value["source_sha256"] != _sha(_encode(before))
            or value["context"] != context or value["runtime"] != _runtime()):
        _fail("SOURCE_MISMATCH")
    _captured_source_guard()
    value = _j_inventory(_encode(value), before, _sha(_encode(before)), context,
                         "optimized" if sys.flags.optimize else "normal", "architecture")
    _write(destination, _encode(value))
    return value


def run_core(root, context, architecture_inventory, destination):
    """Run original full-discovery complement and seal available originals."""
    _captured_source_guard()
    context, root = _context(context), _directory(root)
    before = _source(root, context)
    destination = _outside(destination, root)
    destination.mkdir(mode=0o700)
    _write(destination / "source-before.json", _encode(before))
    _write(destination / "architecture-census.json", _encode(architecture_inventory))
    code, interrupted = _children(root, context, "_core_child", destination, architecture_inventory)
    complete = False
    selected_ids = []
    try:
        after = _source(root, context)
        _write(destination / "source-after.json", _encode(after))
        _captured_source_guard()
        full = _json(_read(destination / "full-census.json", 4 * MIB), 4 * MIB)
        arch_ids = {row["id"] for row in architecture_inventory["tests"]}
        selected_ids = [row["id"] for row in full["tests"] if row["id"] not in arch_ids]
        rows = [_json(line, 16 * MIB) for line in
                _read(destination / "events.jsonl", 16 * MIB).splitlines()]
        complete = (not interrupted and before == after and rows
                    and rows[-1].get("event") == "invocation_exit"
                    and rows[-1].get("exit_code") == code)
    except (ValueError, OSError, KeyError, TypeError):
        after = None
    names = set(CORE_MEMBERS) - {"core-result.json", "ARTIFACT_SHA256SUMS"}
    if all((destination / name).is_file() for name in names):
        bodies = {name: _read(destination / name, CORE_MEMBERS[name]) for name in names}
        result = {"schema": "main-unittest.core.v1", "context": context,
                  "mode": "optimized" if sys.flags.optimize else "normal",
                  "profile": "core", "purpose": "MAIN_DISCOVERY",
                  "source_before_sha256": _sha(bodies["source-before.json"]),
                  "source_after_sha256": _sha(bodies["source-after.json"]),
                  "full_inventory_sha256": _sha(bodies["full-census.json"]),
                  "architecture_inventory_sha256": _sha(bodies["architecture-census.json"]),
                  "selected_ids": selected_ids,
                  "journal": {"bytes": len(bodies["events.jsonl"]), "sha256": _sha(bodies["events.jsonl"])},
                  "stdout": {"bytes": len(bodies["stdout.bin"]), "sha256": _sha(bodies["stdout.bin"])},
                  "stderr": {"bytes": len(bodies["stderr.bin"]), "sha256": _sha(bodies["stderr.bin"])},
                  "exit_code": code, "complete": bool(complete)}
        _write(destination / "core-result.json", _encode(result))
        bodies["core-result.json"] = _read(destination / "core-result.json", MIB)
        _write(destination / "ARTIFACT_SHA256SUMS", "".join(
            _sha(raw) + "  " + name + "\n" for name, raw in sorted(bodies.items())).encode("ascii"))
    return code if code else (0 if complete else 1)


def _worker(operation, root, control, destination):
    _captured_source_guard()
    root = _directory(root)
    value = _json(_read(control, 8 * MIB), 8 * MIB)
    if type(value) is not dict or set(value) != {"context", "architecture"}:
        _fail("INVALID_RECORD")
    context = _context(value["context"])
    before = _source(root, context)
    profile = "architecture" if operation == "_architecture_child" else "full"
    full, branches, cases = _discover(root, context, _sha(_encode(before)), profile)
    if _source(root, context) != before:
        _fail("SOURCE_MISMATCH")
    _captured_source_guard()
    if operation == "_architecture_child":
        _write(destination, _encode(full))
        return 0
    _write(destination / "full-census.json", _encode(full))
    full = _j_inventory(_encode(full), before, full["source_sha256"], context,
                        full["mode"], "full")
    arch = _j_inventory(_encode(value["architecture"]), before, full["source_sha256"],
                        context, full["mode"], "architecture")
    if (type(arch) is not dict or arch.get("context") != context
            or arch.get("source_sha256") != full["source_sha256"]
            or arch.get("runtime") != full["runtime"]):
        _fail("INVENTORY_MISMATCH")
    expected_selected, expected_architecture = _j_partition(full, arch)
    suite, selected = _partition(full, branches, cases, arch)
    if selected != expected_selected:
        _fail("INVENTORY_MISMATCH")
    with (destination / "events.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        journal = Journal(stream)
        journal.emit("invocation_start", argv=list(sys.argv), python=sys.version,
                     optimize=sys.flags.optimize, no_site=sys.flags.no_site,
                     isolated=sys.flags.isolated)
        result = runner_type(journal)(verbosity=2).run(suite)
        code = 0 if result.wasSuccessful() else 1
        journal.emit("invocation_exit", exit_code=code)
    return code


# Inert evidence verification. No supplied N/P dictionary authenticates HTTP.
# Uses existing helpers: _fail, _json, _encode, _sha, _context, _source,
# _runtime, _captured_source_guard, and unchanged read_graph_archive.

_J_MIB = 1024 * 1024
_J_MEMBERS = CORE_MEMBERS
_J_BINDING = {
    "checked_commit", "repository", "run_id", "run_attempt",
    "receipt_sha256", "artifact_id", "artifact_sha256", "artifact_name",
}
_J_REASONS = {
    "INVALID_RECORD", "SOURCE_MISMATCH", "INVENTORY_MISMATCH",
    "UNSAFE_PARTITION", "NON_SUCCESS", "STALE_NATIVE",
    "ARCHIVE_MISMATCH", "INCOMPLETE", "PILOT_SCOPE",
}


def _j_require(value, reason="INVALID_RECORD"):
    if not value:
        _fail(reason)


def _j_fields(value, keys):
    _j_require(type(value) is dict and set(value) == set(keys))
    return value


def _j_int(value, minimum=0):
    _j_require(type(value) is int and value >= minimum)
    return value


def _j_text(value):
    _j_require(type(value) is str and 0 < len(value) <= 4096 and "\0" not in value)
    return value


def _j_digest(value):
    _j_require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)
    return value


def _j_decimal(value):
    _j_require(type(value) is str and re.fullmatch(r"[1-9][0-9]*", value) is not None)
    return value


def _j_equal_context(value, context):
    _context(value)
    _j_require(_encode(value) == _encode(context), "STALE_NATIVE")


def _j_stats(value, raw):
    _j_fields(value, {"bytes", "sha256"})
    _j_require(_j_int(value["bytes"]) == len(raw), "ARCHIVE_MISMATCH")
    _j_require(_j_digest(value["sha256"]) == _sha(raw), "ARCHIVE_MISMATCH")


def _j_record(raw, name):
    record = _json(raw, _J_MEMBERS[name])
    _j_require(_encode(record) == raw)
    return record


def _j_core_members(raw):
    """Read only the fixed core layout; never extract or execute members."""
    _j_require(type(raw) is bytes and 22 <= len(raw) <= 64 * _J_MIB, "ARCHIVE_MISMATCH")
    _j_require(raw[:4] == b"PK\x03\x04" and raw[-22:-18] == b"PK\x05\x06",
               "ARCHIVE_MISMATCH")
    end = struct.unpack_from("<4s4H2LH", raw, len(raw) - 22)
    _j_require(end[1:5] == (0, 0, len(_J_MEMBERS), len(_J_MEMBERS))
               and end[5] + end[6] == len(raw) - 22 and end[7] == 0,
               "ARCHIVE_MISMATCH")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            _j_require(len(names) == len(set(names)) == len(_J_MEMBERS)
                       and set(names) == set(_J_MEMBERS), "ARCHIVE_MISMATCH")
            cursor = 0
            for info in sorted(infos, key=lambda item: item.header_offset):
                _j_require(info.header_offset == cursor and cursor + 30 <= end[6],
                           "ARCHIVE_MISMATCH")
                local = struct.unpack_from("<4s5H3I2H", raw, cursor)
                _j_require(local[0] == b"PK\x03\x04" and local[2] == info.flag_bits
                           and local[3] == info.compress_type and local[10] == 0,
                           "ARCHIVE_MISMATCH")
                start = cursor + 30 + local[9]
                _j_require(raw[cursor + 30:start] == info.filename.encode("ascii")
                           and start + info.compress_size <= end[6],
                           "ARCHIVE_MISMATCH")
                declared = (info.CRC, info.compress_size, info.file_size)
                if info.flag_bits & 8:
                    _j_require(all(value in (0, actual)
                                   for value, actual in zip(local[6:9], declared)),
                               "ARCHIVE_MISMATCH")
                else:
                    _j_require(local[6:9] == declared, "ARCHIVE_MISMATCH")
                cursor = start + info.compress_size
                if info.flag_bits & 8:
                    if raw[cursor:cursor + 4] == b"PK\x07\x08":
                        cursor += 4
                    _j_require(cursor + 12 <= end[6]
                               and struct.unpack_from("<3I", raw, cursor) == declared,
                               "ARCHIVE_MISMATCH")
                    cursor += 12
            _j_require(cursor == end[6], "ARCHIVE_MISMATCH")
            result, total, offsets = {}, 0, set()
            for info in infos:
                limit = _J_MEMBERS[info.filename]
                kind = stat.S_IFMT(info.external_attr >> 16)
                _j_require(not info.is_dir() and kind in (0, stat.S_IFREG)
                           and not info.flag_bits & 1
                           and info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                           and not info.extra and not info.comment
                           and 0 <= info.file_size <= limit
                           and 0 <= info.compress_size <= len(raw)
                           and 0 <= info.header_offset < end[6]
                           and info.header_offset not in offsets,
                           "ARCHIVE_MISMATCH")
                offsets.add(info.header_offset)
                with archive.open(info, "r") as stream:
                    body = stream.read(limit + 1)
                _j_require(len(body) == info.file_size <= limit, "ARCHIVE_MISMATCH")
                total += len(body)
                _j_require(total <= 64 * _J_MIB, "ARCHIVE_MISMATCH")
                result[info.filename] = body
            _j_require(min(offsets) == 0 and archive.start_dir == end[6],
                       "ARCHIVE_MISMATCH")
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError,
            EOFError, zlib.error):
        _fail("ARCHIVE_MISMATCH")
    manifest = result["ARTIFACT_SHA256SUMS"]
    try:
        seen = {}
        for line in manifest.decode("ascii").splitlines():
            match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
            _j_require(match is not None and match[2] not in seen, "ARCHIVE_MISMATCH")
            seen[match[2]] = match[1]
        _j_require(set(seen) == set(_J_MEMBERS) - {"ARTIFACT_SHA256SUMS"},
                   "ARCHIVE_MISMATCH")
        _j_require(all(seen[name] == _sha(result[name]) for name in seen),
                   "ARCHIVE_MISMATCH")
    except UnicodeError:
        _fail("ARCHIVE_MISMATCH")
    return result


def _j_inventory(raw, source, source_hash, context, mode, profile):
    inventory = _j_record(raw, "full-census.json" if profile == "full"
                          else "architecture-census.json")
    _j_fields(inventory, {
        "schema", "context", "source_sha256", "mode", "profile", "discovery",
        "runtime", "tests", "count", "loader_errors",
    })
    _j_require(inventory["schema"] == "main-unittest.inventory.v1")
    _j_equal_context(inventory["context"], context)
    _j_require(inventory["mode"] == mode and inventory["profile"] == profile)
    _j_require(_j_digest(inventory["source_sha256"]) == source_hash, "SOURCE_MISMATCH")
    discovery = _j_fields(inventory["discovery"], {"start_dir", "pattern", "top_level_dir"})
    pattern = "test*.py" if profile == "full" else "test_architecture_*.py"
    _j_require(discovery == {"start_dir": "tests", "pattern": pattern, "top_level_dir": None})
    runtime = _j_fields(inventory["runtime"], {
        "implementation", "python", "executable_sha256", "optimize",
        "no_site", "dont_write_bytecode",
    })
    expected_runtime = dict(_runtime())
    expected_runtime.update(optimize=int(mode == "optimized"), no_site=1,
                            dont_write_bytecode=True)
    _j_int(runtime["optimize"])
    _j_int(runtime["no_site"])
    _j_require(type(runtime["dont_write_bytecode"]) is bool)
    _j_digest(runtime["executable_sha256"])
    _j_text(runtime["implementation"])
    _j_text(runtime["python"])
    _j_require(_encode(runtime) == _encode(expected_runtime), "SOURCE_MISMATCH")
    rows = inventory["tests"]
    _j_require(type(rows) is list and rows and inventory["loader_errors"] == [],
               "INVENTORY_MISMATCH")
    _j_require(type(inventory["loader_errors"]) is list)
    _j_require(_j_int(inventory["count"], 1) == len(rows), "INVENTORY_MISMATCH")
    sources = {row["path"] for row in source["files"]}
    identities, previous = set(), -1
    for row in rows:
        _j_fields(row, {"id", "module", "class", "source_path", "branch"})
        identity = _j_text(row["id"])
        module = _j_text(row["module"])
        owner = _j_text(row["class"])
        path = _j_text(row["source_path"])
        branch = _j_int(row["branch"])
        _j_require(identity not in identities and path in sources
                   and owner.startswith(module + ".") and branch >= previous,
                   "INVENTORY_MISMATCH")
        identities.add(identity)
        previous = branch
    return inventory


def _j_partition(full, architecture):
    """Compare actual discovery populations, never ID prefixes or fixed totals."""
    full_rows, arch_rows = full["tests"], architecture["tests"]
    by_id = {row["id"]: row for row in full_rows}
    arch_ids = [row["id"] for row in arch_rows]
    selected = set(arch_ids)
    _j_require(selected <= set(by_id), "INVENTORY_MISMATCH")

    def descriptor(row):
        return {key: row[key] for key in ("id", "module", "class", "source_path")}

    _j_require([descriptor(row) for row in full_rows if row["id"] in selected]
               == [descriptor(row) for row in arch_rows], "INVENTORY_MISMATCH")
    branches, modules, classes = {}, {}, {}
    for row in full_rows:
        profile = row["id"] in selected
        branches.setdefault(row["branch"], set()).add(profile)
        modules.setdefault(row["module"], set()).add(profile)
        classes.setdefault(row["class"], set()).add(profile)
    _j_require(all(len(kinds) == 1 for table in (branches, modules, classes)
                   for kinds in table.values()), "UNSAFE_PARTITION")
    core_rows = [row for row in full_rows if row["id"] not in selected]
    for key in ("module", "class"):
        def transitions(rows):
            result = []
            for row in rows:
                value = row[key]
                if not result or result[-1] != value:
                    result.append(value)
            return result
        core_values = {row[key] for row in core_rows}
        _j_require([value for value in transitions(full_rows) if value in core_values]
                   == transitions(core_rows), "UNSAFE_PARTITION")
    return [row["id"] for row in core_rows], arch_ids


def _j_journal(raw, selected, mode, runtime, root):
    _j_require(type(raw) is bytes and raw and raw.endswith(b"\n")
               and len(raw) <= _J_MEMBERS["events.jsonl"], "INCOMPLETE")
    rows = [_json(line, _J_MEMBERS["events.jsonl"]) for line in raw.splitlines()]
    _j_require(len(rows) >= 4, "INCOMPLETE")
    common = {"schema", "seq", "pid", "monotonic_ns", "event"}
    extras = {
        "invocation_start": {"argv", "python", "optimize", "no_site", "isolated"},
        "run_start": set(), "test_start": {"test_id"},
        "test_stop": {"test_id", "elapsed_ns"},
        "run_end": {"tests_run", "failures", "errors", "skipped",
                    "expected_failures", "unexpected_successes", "successful"},
        "invocation_exit": {"exit_code"},
    }
    pid, clock, active, successful, completed = None, -1, None, False, 0
    for index, row in enumerate(rows):
        _j_require(type(row) is dict and "event" in row)
        event = row["event"]
        if event == "outcome":
            outcome = row.get("outcome")
            extra = {"test_id", "outcome"}
            if type(outcome) is str and outcome.startswith("subtest_"):
                extra.add("parent_test_id")
            _j_fields(row, common | extra)
        else:
            _j_require(event in extras)
            _j_fields(row, common | extras[event])
        _j_require(row["schema"] == "unittest-timing/v1")
        _j_require(_j_int(row["seq"]) == index, "INCOMPLETE")
        current_pid = _j_int(row["pid"], 1)
        timestamp = _j_int(row["monotonic_ns"])
        _j_require(timestamp >= clock and (pid is None or pid == current_pid), "INCOMPLETE")
        pid, clock = current_pid, timestamp
        if event == "invocation_start":
            _j_require(index == 0 and type(row["argv"]) is list and row["argv"])
            _j_require(all(type(item) is str and item for item in row["argv"]))
            argv = row["argv"]
            _j_require(len(argv) == 5 and argv[1] == "_core_child"
                       and Path(argv[0]).resolve() == SOURCE_ROOT / "tools/unittest_full_qualification.py"
                       and Path(argv[2]).resolve() == root, "SOURCE_MISMATCH")
            for original in (argv[3], argv[4]):
                path = Path(original)
                _j_require(path.is_absolute() and ".." not in path.parts
                           and path != root and root not in path.parents
                           and path != SOURCE_ROOT and SOURCE_ROOT not in path.parents,
                           "SOURCE_MISMATCH")
            _j_require(Path(argv[3]).name == "control.json", "SOURCE_MISMATCH")
            _j_require(row["python"] == runtime["python"]
                       and _j_int(row["optimize"]) == int(mode == "optimized")
                       and _j_int(row["no_site"]) == 1
                       and _j_int(row["isolated"]) == 0, "SOURCE_MISMATCH")
        elif event == "run_start":
            _j_require(index == 1, "INCOMPLETE")
        elif event == "test_start":
            _j_require(1 < index < len(rows) - 2 and active is None
                       and completed < len(selected)
                       and _j_text(row["test_id"]) == selected[completed],
                       "INVENTORY_MISMATCH")
            active, successful = row["test_id"], False
        elif event == "outcome":
            outcome = row["outcome"]
            _j_require(type(outcome) is str)
            _j_require(outcome in ("success", "subtest_success"), "NON_SUCCESS")
            if outcome == "success":
                _j_require(active is not None and row["test_id"] == active
                           and not successful, "INVENTORY_MISMATCH")
                successful = True
            else:
                _j_text(row["test_id"])
                _j_require(active is not None and row["parent_test_id"] == active
                           and not successful, "INVENTORY_MISMATCH")
        elif event == "test_stop":
            _j_require(active is not None and row["test_id"] == active and successful,
                       "INCOMPLETE")
            _j_int(row["elapsed_ns"])
            completed += 1
            active, successful = None, False
        elif event == "run_end":
            _j_require(index == len(rows) - 2 and active is None
                       and completed == len(selected), "INCOMPLETE")
            _j_require(_j_int(row["tests_run"]) == len(selected), "INVENTORY_MISMATCH")
            for key in ("failures", "errors", "skipped", "expected_failures",
                        "unexpected_successes"):
                _j_require(_j_int(row[key]) == 0, "NON_SUCCESS")
            _j_require(row["successful"] is True, "NON_SUCCESS")
        elif event == "invocation_exit":
            _j_require(index == len(rows) - 1 and _j_int(row["exit_code"]) == 0,
                       "INCOMPLETE")
    _j_require(rows[0]["event"] == "invocation_start"
               and rows[1]["event"] == "run_start"
               and rows[-2]["event"] == "run_end"
               and rows[-1]["event"] == "invocation_exit", "INCOMPLETE")


def _j_native(context, producer, native_api, cores, architecture_raw):
    """Check supplied original/projection consistency; authenticate no transport."""
    _j_fields(producer, {"core_result", "architecture_result", "architecture_outputs"})
    _j_require(producer["core_result"] == producer["architecture_result"] == "success",
               "NON_SUCCESS")
    binding = _j_fields(producer["architecture_outputs"], _J_BINDING)
    for key in ("repository", "checked_commit", "run_id", "run_attempt"):
        _j_require(type(binding[key]) is str and binding[key] == context[key], "STALE_NATIVE")
    for key in ("receipt_sha256", "artifact_sha256"):
        _j_digest(binding[key])
    _j_decimal(binding["artifact_id"])
    _j_require(binding["artifact_name"] == "research-architecture-"
               + context["run_id"] + "-" + context["run_attempt"], "STALE_NATIVE")
    _j_require(type(native_api) is tuple and len(native_api) == 2)
    native, originals = native_api
    _j_fields(native, {"schema", "context", "producer", "api_originals", "artifacts"})
    _j_require(native["schema"] == "main-unittest.native.v1")
    _j_equal_context(native["context"], context)
    _j_require(_encode(native["producer"]) == _encode(producer))
    _j_require(type(originals) is dict and len(originals) == 4)
    rows = native["api_originals"]
    _j_require(type(rows) is list and len(rows) == 4)
    base = "https://api.github.com/repos/" + context["repository"]
    run_url = base + "/actions/runs/" + context["run_id"] + "/attempts/" + context["run_attempt"]
    responses, names = {}, []
    for row in rows:
        _j_fields(row, {"path", "url", "bytes", "sha256"})
        path = _j_text(row["path"])
        _j_require(re.fullmatch(r"[A-Za-z0-9_.-]+\.json", path) is not None)
        url = _j_text(row["url"])
        _j_require(path in originals and type(originals[path]) is bytes
                   and path not in names and url not in responses)
        _j_stats({"bytes": row["bytes"], "sha256": row["sha256"]}, originals[path])
        responses[url] = _json(originals[path], 2 * _J_MIB)
        names.append(path)
    _j_require(names == sorted(names) and set(names) == set(originals))
    _j_require(run_url in responses, "STALE_NATIVE")
    run = responses[run_url]
    _j_require(type(run) is dict
               and str(_j_int(run["id"], 1)) == context["run_id"]
               and str(_j_int(run["run_attempt"], 1)) == context["run_attempt"]
               and run["head_sha"] == context["checked_commit"]
               and type(run["repository"]) is dict
               and run["repository"].get("full_name") == context["repository"]
               and type(run["head_repository"]) is dict
               and run["head_repository"].get("full_name") == context["repository"]
               and run["path"] == ".github/workflows/full-unittest-qualification.yml"
               and run["status"] in ("in_progress", "completed"), "STALE_NATIVE")
    artifact_rows = native["artifacts"]
    _j_require(type(artifact_rows) is list and len(artifact_rows) == 3)
    expected = {
        ("architecture", "both"): (binding["artifact_name"], architecture_raw),
        ("core", "normal"): ("main-unittest-core-normal-" + context["run_id"]
                            + "-" + context["run_attempt"], cores["normal"]),
        ("core", "optimized"): ("main-unittest-core-optimized-" + context["run_id"]
                               + "-" + context["run_attempt"], cores["optimized"]),
    }
    admitted, identities = {}, set()
    for row in artifact_rows:
        _j_fields(row, {"profile", "mode", "id", "name", "run_id", "head_sha",
                        "expired", "digest", "archive_bytes"})
        key = (row["profile"], row["mode"])
        _j_require(key in expected and key not in admitted)
        identity = _j_decimal(row["id"])
        _j_require(identity not in identities)
        identities.add(identity)
        name, raw = expected[key]
        _j_require(row["name"] == name and row["run_id"] == context["run_id"]
                   and row["head_sha"] == context["checked_commit"]
                   and row["expired"] is False, "STALE_NATIVE")
        _j_require(_j_digest(row["digest"]) == _sha(raw)
                   and _j_int(row["archive_bytes"]) == len(raw), "ARCHIVE_MISMATCH")
        url = base + "/actions/artifacts/" + identity
        _j_require(url in responses, "STALE_NATIVE")
        original = responses[url]
        _j_require(type(original) is dict
                   and str(_j_int(original["id"], 1)) == identity
                   and original["name"] == name
                   and _j_int(original["size_in_bytes"]) == len(raw)
                   and original["expired"] is False
                   and original["digest"] == "sha256:" + row["digest"]
                   and original["url"] == url
                   and original["archive_download_url"] == url + "/zip",
                   "ARCHIVE_MISMATCH")
        workflow = original["workflow_run"]
        _j_require(type(workflow) is dict
                   and str(_j_int(workflow["id"], 1)) == context["run_id"]
                   and workflow["head_sha"] == context["checked_commit"], "STALE_NATIVE")
        if key == ("architecture", "both"):
            _j_require(identity == binding["artifact_id"]
                       and row["digest"] == binding["artifact_sha256"], "ARCHIVE_MISMATCH")
        admitted[key] = row
    _j_require(set(admitted) == set(expected)
               and set(responses) == {run_url} | {
                   base + "/actions/artifacts/" + row["id"] for row in artifact_rows})
    return native, binding, admitted


def _j_core(raw, context, source, source_hash, mode, root):
    members = _j_core_members(raw)
    before = _j_record(members["source-before.json"], "source-before.json")
    after = _j_record(members["source-after.json"], "source-after.json")
    for record in (before, after):
        _j_fields(record, {"schema", "context", "files"})
        _j_require(record["schema"] == "main-unittest.source.v1")
        _j_equal_context(record["context"], context)
        _j_require(type(record["files"]) is list)
        for row in record["files"]:
            _j_fields(row, {"path", "mode", "git_blob", "bytes", "sha256"})
            _j_text(row["path"])
            _j_require(row["mode"] in ("100644", "100755"))
            _j_require(type(row["git_blob"]) is str
                       and re.fullmatch(r"[0-9a-f]{40}", row["git_blob"]) is not None)
            _j_int(row["bytes"])
            _j_digest(row["sha256"])
        _j_require(_encode(record) == _encode(source), "SOURCE_MISMATCH")
    _j_require(_sha(members["source-before.json"]) == source_hash
               and members["source-before.json"] == members["source-after.json"],
               "SOURCE_MISMATCH")
    full = _j_inventory(members["full-census.json"], source, source_hash, context, mode, "full")
    architecture = _j_inventory(members["architecture-census.json"], source, source_hash,
                                context, mode, "architecture")
    selected, architecture_ids = _j_partition(full, architecture)
    result = _j_record(members["core-result.json"], "core-result.json")
    _j_fields(result, {
        "schema", "context", "mode", "profile", "purpose", "source_before_sha256",
        "source_after_sha256", "full_inventory_sha256", "architecture_inventory_sha256",
        "selected_ids", "journal", "stdout", "stderr", "exit_code", "complete",
    })
    _j_require(result["schema"] == "main-unittest.core.v1")
    _j_equal_context(result["context"], context)
    _j_require(result["mode"] == mode and result["profile"] == "core")
    if result["purpose"] == "ARCHITECTURE_PYTEST_PILOT":
        _fail("PILOT_SCOPE")
    _j_require(result["purpose"] == "MAIN_DISCOVERY")
    _j_require(result["complete"] is True, "INCOMPLETE")
    _j_require(_j_int(result["exit_code"]) == 0, "NON_SUCCESS")
    _j_require(type(result["selected_ids"]) is list and result["selected_ids"] == selected,
               "INVENTORY_MISMATCH")
    for member, field in (
        ("source-before.json", "source_before_sha256"),
        ("source-after.json", "source_after_sha256"),
        ("full-census.json", "full_inventory_sha256"),
        ("architecture-census.json", "architecture_inventory_sha256"),
    ):
        _j_require(_j_digest(result[field]) == _sha(members[member]), "ARCHIVE_MISMATCH")
    for member, field in (
        ("events.jsonl", "journal"), ("stdout.bin", "stdout"), ("stderr.bin", "stderr"),
    ):
        _j_stats(result[field], members[member])
    _j_journal(members["events.jsonl"], selected, mode, full["runtime"], root)
    return members, full, architecture, selected, architecture_ids


def qualify_join(root, context, producer_outputs, native_api, core_archives,
                 architecture_archive):
    """Verify inert supplied evidence. Never authenticate acquisition or award G5."""
    try:
        _captured_source_guard()
        context = _context(context)
        _j_require(type(core_archives) is dict and set(core_archives) == {"normal", "optimized"})
        _j_require(type(architecture_archive) is bytes)
        _j_require(all(type(raw) is bytes for raw in core_archives.values()))
        root = Path(root).resolve()
        source = _source(root, context)
        source_hash = _sha(_encode(source))
        native, binding, artifacts = _j_native(
            context, producer_outputs, native_api, core_archives, architecture_archive)
        admitted = {}
        for mode in ("normal", "optimized"):
            admitted[mode] = _j_core(core_archives[mode], context, source, source_hash, mode, root)
        normal, optimized = admitted["normal"], admitted["optimized"]
        _j_require(normal[1]["tests"] == optimized[1]["tests"]
                   and normal[2]["tests"] == optimized[2]["tests"],
                   "INVENTORY_MISMATCH")
        try:
            readback = read_graph_archive(
                architecture_archive, binding,
                (root / "docs/site/dependency-source/GRAPH.json").read_bytes())
        except (ValueError, OSError, UnicodeError, RecursionError, OverflowError):
            _fail("ARCHIVE_MISMATCH")
        strict_modes = readback["receipt"]["producer_receipt"]["test_modes"]
        strict = {row["mode"]: row for row in strict_modes}
        _j_require(set(strict) == {"normal", "optimized"} and len(strict_modes) == 2,
                   "INVENTORY_MISMATCH")
        core_rows, coverage = [], []
        for mode in ("normal", "optimized"):
            members, full, architecture, selected, architecture_ids = admitted[mode]
            _j_require(strict[mode]["test_ids"] == sorted(architecture_ids)
                       and strict[mode]["test_count"] == len(architecture_ids),
                       "INVENTORY_MISMATCH")
            full_ids = [row["id"] for row in full["tests"]]
            _j_require(not (set(selected) & set(architecture_ids))
                       and set(selected) | set(architecture_ids) == set(full_ids)
                       and len(selected) + len(architecture_ids) == len(full_ids),
                       "INVENTORY_MISMATCH")
            artifact = artifacts[("core", mode)]
            core_rows.append({
                "mode": mode, "artifact_id": artifact["id"],
                "artifact_sha256": _sha(core_archives[mode]),
                "result_sha256": _sha(members["core-result.json"]),
                "full_inventory_sha256": _sha(members["full-census.json"]),
                "architecture_inventory_sha256": _sha(members["architecture-census.json"]),
            })
            coverage.append({
                "mode": mode, "full_count": len(full_ids),
                "core_count": len(selected), "architecture_count": len(architecture_ids),
            })
        _j_require(_encode(_source(root, context)) == _encode(source), "SOURCE_MISMATCH")
        _captured_source_guard()
        return {
            "schema": "main-unittest.join.v1", "context": context,
            "scope": "MAIN_REGISTERED_UNITTEST_DISCOVERY_ONLY", "status": "QUALIFIED",
            "scientific_effect": "NONE", "g5_claim": False, "cbz_claim": False,
            "source_sha256": source_hash, "native_sha256": _sha(_encode(native)),
            "core": core_rows,
            "architecture": {"binding": dict(binding),
                             "readback_sha256": _sha(_encode(readback))},
            "coverage": coverage,
        }
    except ValueError as error:
        if str(error) in _J_REASONS:
            raise
        _fail("INVALID_RECORD")
    except (TypeError, KeyError, IndexError, AttributeError, OSError, UnicodeError,
            OverflowError, RecursionError, struct.error):
        _fail("INVALID_RECORD")

def main(argv=None):
    """Fixed file-based operations only; no HTTP acquisition or test filters."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if not arguments:
            _fail("INVALID_RECORD")
        operation = arguments[0]
        if operation in ("_architecture_child", "_core_child") and len(arguments) == 4:
            return _worker(operation, Path(arguments[1]), Path(arguments[2]), Path(arguments[3]))
        if operation == "architecture-inventory" and len(arguments) == 4:
            architecture_inventory(
                Path(arguments[1]), _json(_read(arguments[2], MIB), MIB), Path(arguments[3]))
            return 0
        if operation == "core" and len(arguments) == 5:
            return run_core(
                Path(arguments[1]), _json(_read(arguments[2], MIB), MIB),
                _json(_read(arguments[3], 4 * MIB), 4 * MIB), Path(arguments[4]))
        if operation == "join" and len(arguments) == 10:
            root, context_path, producer_path, native_path, original_dir = map(Path, arguments[1:6])
            context = _json(_read(context_path, MIB), MIB)
            producer = _json(_read(producer_path, MIB), MIB)
            native = _json(_read(native_path, 4 * MIB), 4 * MIB)
            _directory(original_dir)
            originals = {}
            for row in native["api_originals"]:
                name = row["path"]
                if type(name) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", name):
                    _fail("INVALID_RECORD")
                if name in originals:
                    _fail("INVALID_RECORD")
                originals[name] = _read(original_dir / name, 2 * MIB)
            cores = {"normal": _read(arguments[6], 64 * MIB),
                     "optimized": _read(arguments[7], 64 * MIB)}
            architecture = _read(arguments[8], 64 * MIB)
            result = qualify_join(root, context, producer, (native, originals), cores, architecture)
            _write(_outside(Path(arguments[9]), root), _encode(result))
            return 0
        _fail("INVALID_RECORD")
    except ValueError as error:
        reason = str(error) if str(error) in _J_REASONS else "INVALID_RECORD"
        sys.stderr.write(reason + "\n")
        return 1
    except (OSError, TypeError, KeyError, IndexError, AttributeError, UnicodeError,
            RecursionError, OverflowError):
        sys.stderr.write("INVALID_RECORD\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
