"""Check supplied Main unittest acquisition bytes and print admitted originals.

Only verify_capture and write_original_frames are public APIs. Neither performs
HTTP, reads credentials, starts producers or authenticates a supplied record.
The unchanged checked-source helper and reader supply the actual admission.
Project code is loaded from its pinned original bytes, not a cached checker.
The data root may be a synthetic Git fixture; it is not the project import root.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import types
from typing import BinaryIO
import zipfile

__all__ = ("verify_capture", "write_original_frames")
_MIB = 1024 * 1024
_API_LIMIT = 2 * _MIB
_SOURCE_LIMIT = 4 * _MIB
_ARCHIVE_LIMIT = 64 * _MIB
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SELF_PATH = _PROJECT_ROOT / "tools/unittest_native_acquisition.py"
_HELPER_PATH = _PROJECT_ROOT / "tools/unittest_full_qualification.py"
_READER_PATH = _PROJECT_ROOT / "architecture/graph_artifact_readback.py"
_HELPER_BLOB = "02dc2f3222aaf3e3d4d024659fffff22d6fb5e66"
_READER_BLOB = "c39ffd0fdcde1fd5f895c57df5f770a582918a8a"
_REF = "refs/heads/codex/g5-native-qualification-20261110"
_BRANCH = _REF.removeprefix("refs/heads/")
_WORKFLOW = ".github/workflows/full-unittest-qualification.yml"
_AUTHORITY = "NONE_FROM_SUPPLIED_BYTES"
_CONTEXT = {"repository", "checked_commit", "checked_tree", "run_id", "run_attempt"}
_CAPTURE = {"schema", "context", "native", "needs_sha256", "responses", "downloads"}
_NATIVE = {"event_name", "ref", "repository", "sha", "run_id", "run_attempt",
           "workflow_ref", "workflow_sha"}
_ROW = {"path", "url", "bytes", "sha256", "exit_code", "timed_out", "complete"}
_DOWNLOAD = _ROW | {"profile", "mode", "id", "name"}
_BINDING = {"repository", "checked_commit", "run_id", "run_attempt",
            "receipt_sha256", "artifact_id", "artifact_sha256", "artifact_name"}
_CORE_OUTPUT = {"artifact_id", "artifact_name", "artifact_sha256"}
_NEEDS = {"architecture", "core_normal", "core_optimized"}
_ARCHIVES = {"architecture.zip", "core-normal.zip", "core-optimized.zip"}
_PRODUCERS = {"architecture / research-architecture",
              "Core full discovery (normal)", "Core full discovery (optimized)"}
_JOBS = _PRODUCERS | {"Stage 1 targeted contracts (normal)",
                      "Stage 1 targeted contracts (optimized)",
                      "Acquire original Main unittest evidence"}
_REASONS = {
    "INVALID_CAPTURE", "STALE_CONTEXT", "PRODUCER_NOT_COMPLETE", "ARCHIVE_MISMATCH",
    "SOURCE_MISMATCH", "INVENTORY_MISMATCH", "NON_SUCCESS", "INCOMPLETE",
    "PILOT_SCOPE", "INVALID_RECORD", "UNSAFE_PARTITION", "STALE_NATIVE",
}


def _fail(reason):
    raise ValueError(reason)


def _require(condition, reason="INVALID_CAPTURE"):
    if not condition:
        _fail(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def _encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _source_bytes(path):
    """Read a bounded original project source, without following its components."""
    try:
        relative = path.relative_to(_PROJECT_ROOT)
        current = _PROJECT_ROOT
        for part in relative.parts:
            current = current / part
            _require(not current.is_symlink(), "SOURCE_MISMATCH")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            _require(stat.S_ISREG(before.st_mode) and before.st_size <= _SOURCE_LIMIT,
                     "SOURCE_MISMATCH")
            raw = stream.read(_SOURCE_LIMIT + 1)
            after = os.fstat(stream.fileno())
        identity = lambda info: (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
                                 info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        _require(len(raw) == before.st_size <= _SOURCE_LIMIT
                 and identity(before) == identity(after), "SOURCE_MISMATCH")
        return raw
    except (OSError, TypeError, ValueError):
        _fail("SOURCE_MISMATCH")


# The fresh private module executes only the exact original pinned helper.
# Its own loader reads the exact pinned recorder/reader originals. A cached
# sys.modules object that merely advertises a plausible __file__ is never used
# as an admission checker here.
_HELPER_RAW = _source_bytes(_HELPER_PATH)
_require(_blob(_HELPER_RAW) == _HELPER_BLOB, "SOURCE_MISMATCH")
_require(_blob(_source_bytes(_READER_PATH)) == _READER_BLOB, "SOURCE_MISMATCH")
_FULL = types.ModuleType("_unittest_acquisition_checked_full")
_FULL.__file__ = str(_HELPER_PATH)
_FULL.__package__ = ""
exec(compile(_HELPER_RAW, str(_HELPER_PATH), "exec", dont_inherit=True), _FULL.__dict__)
_ADMIT = _FULL.qualify_join


def _own_origin():
    """Bind this new module to its actual tracked bytes/mode and project HEAD."""
    try:
        _require(Path(__file__).resolve() == _SELF_PATH, "SOURCE_MISMATCH")
        head = _FULL._git(_PROJECT_ROOT, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
        tree = _FULL._git(_PROJECT_ROOT, "rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip()
        raw = _source_bytes(_SELF_PATH)
        entries = _FULL._git(_PROJECT_ROOT, "ls-tree", "--full-tree", "-z", head,
                             "--", "tools/unittest_native_acquisition.py").split(b"\0")
        _require(len(entries) == 2 and entries[-1] == b"", "SOURCE_MISMATCH")
        metadata, name = entries[0].split(b"\t", 1)
        mode, kind, identity = metadata.decode("ascii").split()
        _require(name == b"tools/unittest_native_acquisition.py" and kind == "blob"
                 and mode in ("100644", "100755") and _blob(raw) == identity
                 and bool(_SELF_PATH.stat().st_mode & stat.S_IXUSR) == (mode == "100755"),
                 "SOURCE_MISMATCH")
        return head, tree, mode, identity, len(raw), _sha(raw)
    except (OSError, TypeError, ValueError, UnicodeError):
        _fail("SOURCE_MISMATCH")


_OWN_ORIGIN = _own_origin()


def _project_guard():
    _require(_own_origin() == _OWN_ORIGIN, "SOURCE_MISMATCH")
    _require(_blob(_source_bytes(_HELPER_PATH)) == _HELPER_BLOB
             and _blob(_source_bytes(_READER_PATH)) == _READER_BLOB, "SOURCE_MISMATCH")
    _require(_FULL.qualify_join is _ADMIT
             and type(_ADMIT) is types.FunctionType
             and _ADMIT.__globals__ is _FULL.__dict__
             and _ADMIT.__code__.co_filename == str(_HELPER_PATH), "SOURCE_MISMATCH")
    _FULL._captured_source_guard()


def _fields(value, keys):
    _require(type(value) is dict and set(value) == keys)
    return value


def _integer(value, reason="INVALID_CAPTURE"):
    _require(type(value) is int and value >= 0, reason)
    return value


def _positive(value, reason="INVALID_CAPTURE"):
    _require(type(value) is int and value > 0, reason)
    return value


def _decimal(value):
    _require(type(value) is str and re.fullmatch(r"[1-9][0-9]*", value) is not None)
    return value


def _digest(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)
    return value


def _context(value):
    _fields(value, _CONTEXT)
    _require(all(type(item) is str for item in value.values()))
    _require(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value["repository"]) is not None)
    _require(all(part not in (".", "..") for part in value["repository"].split("/")))
    for key in ("checked_commit", "checked_tree"):
        _require(re.fullmatch(r"[0-9a-f]{40}", value[key]) is not None)
    for key in ("run_id", "run_attempt"):
        _decimal(value[key])
    return dict(value)


def _json(raw):
    _require(type(raw) is bytes and len(raw) <= _API_LIMIT)

    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result)
            result[key] = value
        return result

    def constant(_):
        _fail("INVALID_CAPTURE")

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)

    def check(item, depth=0):
        _require(depth <= 64)
        if type(item) is dict:
            for key, child in item.items():
                _require(type(key) is str)
                check(child, depth + 1)
        elif type(item) is list:
            for child in item:
                check(child, depth + 1)
        elif type(item) is float:
            _require(math.isfinite(item))
        else:
            _require(item is None or type(item) in (str, bool, int))
    check(value)
    return value


def _process_row(row):
    _require(type(row["exit_code"]) is int
             and type(row["timed_out"]) is bool and type(row["complete"]) is bool)
    _require(row["exit_code"] == 0 and row["timed_out"] is False
             and row["complete"] is True, "INCOMPLETE")
    _integer(row["bytes"])
    _digest(row["sha256"])
    _require(type(row["path"]) is str and type(row["url"]) is str)


def _stats(row, raw, reason):
    _require(row["bytes"] == len(raw) and row["sha256"] == _sha(raw), reason)


def _needs(needs, context):
    _fields(needs, _NEEDS)
    for row in needs.values():
        _fields(row, {"result", "outputs"})
        _require(type(row["result"]) is str and row["result"] == "success", "PRODUCER_NOT_COMPLETE")
    architecture = _fields(needs["architecture"]["outputs"], _BINDING)
    _require(all(type(value) is str for value in architecture.values()))
    for key in ("repository", "checked_commit", "run_id", "run_attempt"):
        _require(architecture[key] == context[key], "STALE_CONTEXT")
    for key in ("receipt_sha256", "artifact_sha256"):
        _digest(architecture[key])
    specs = {}
    names = {
        "architecture.zip": "research-architecture-" + context["run_id"] + "-" + context["run_attempt"],
        "core-normal.zip": "main-unittest-core-normal-" + context["run_id"] + "-" + context["run_attempt"],
        "core-optimized.zip": "main-unittest-core-optimized-" + context["run_id"] + "-" + context["run_attempt"],
    }
    for path, profile, mode, output in (
        ("architecture.zip", "architecture", "both", architecture),
        ("core-normal.zip", "core", "normal", _fields(needs["core_normal"]["outputs"], _CORE_OUTPUT)),
        ("core-optimized.zip", "core", "optimized", _fields(needs["core_optimized"]["outputs"], _CORE_OUTPUT)),
    ):
        identity = _decimal(output["artifact_id"])
        digest = _digest(output["artifact_sha256"])
        _require(type(output["artifact_name"]) is str)
        _require(output["artifact_name"] == names[path], "ARCHIVE_MISMATCH")
        specs[path] = dict(profile=profile, mode=mode, id=identity,
                           name=names[path], sha256=digest)
    _require(len({row["id"] for row in specs.values()}) == 3, "ARCHIVE_MISMATCH")
    return specs, dict(architecture)


def _run(run, context):
    reason = "STALE_CONTEXT"
    _require(type(run) is dict, reason)
    for key in ("repository", "head_repository"):
        _require(type(run.get(key)) is dict
                 and run[key].get("full_name") == context["repository"], reason)
    repository_id = _positive(run["repository"].get("id"), reason)
    _require(_positive(run["head_repository"].get("id"), reason) == repository_id, reason)
    _require(str(_positive(run.get("id"), reason)) == context["run_id"]
             and str(_positive(run.get("run_attempt"), reason)) == context["run_attempt"]
             and run.get("head_sha") == context["checked_commit"]
             and run.get("event") == "push" and run.get("head_branch") == _BRANCH
             and run.get("path") == _WORKFLOW, reason)
    status = run.get("status")
    _require(status in ("in_progress", "completed"), "INCOMPLETE")
    _require((status == "in_progress" and run.get("conclusion") is None)
             or (status == "completed" and run.get("conclusion") == "success"), "NON_SUCCESS")
    return repository_id, status


def _jobs(value, context):
    reason = "INCOMPLETE"
    _require(type(value) is dict and type(value.get("jobs")) is list, reason)
    jobs = value["jobs"]
    _require(_integer(value.get("total_count"), reason) == len(jobs) <= 100, reason)
    names, identities = set(), set()
    for row in jobs:
        _require(type(row) is dict and type(row.get("name")) is str, reason)
        identity = _positive(row.get("id"), reason)
        _require(row["name"] in _JOBS and row["name"] not in names
                 and identity not in identities
                 and str(_positive(row.get("run_id"), reason)) == context["run_id"]
                 and str(_positive(row.get("run_attempt"), reason)) == context["run_attempt"]
                 and row.get("head_sha") == context["checked_commit"], reason)
        names.add(row["name"])
        identities.add(identity)
    _require(names == _JOBS, reason)
    for row in jobs:
        if row["name"] in _PRODUCERS:
            _require(row.get("status") == "completed" and row.get("conclusion") == "success",
                     "PRODUCER_NOT_COMPLETE")


def _artifact(value, spec, context, repository_id, base, raw):
    _require(type(value) is dict, "ARCHIVE_MISMATCH")
    url = base + "/actions/artifacts/" + spec["id"]
    _require(str(_positive(value.get("id"), "ARCHIVE_MISMATCH")) == spec["id"]
             and value.get("name") == spec["name"]
             and _integer(value.get("size_in_bytes"), "ARCHIVE_MISMATCH") == len(raw)
             and value.get("expired") is False
             and value.get("digest") == "sha256:" + spec["sha256"]
             and value.get("url") == url
             and value.get("archive_download_url") == url + "/zip", "ARCHIVE_MISMATCH")
    workflow = value.get("workflow_run")
    reason = "STALE_CONTEXT"
    _require(type(workflow) is dict, reason)
    _require(str(_positive(workflow.get("id"), reason)) == context["run_id"]
             and _positive(workflow.get("repository_id"), reason) == repository_id
             and _positive(workflow.get("head_repository_id"), reason) == repository_id
             and workflow.get("head_sha") == context["checked_commit"]
             and workflow.get("head_branch") == _BRANCH, reason)


def _verify(root, context, needs, capture, originals, archives):
    _project_guard()
    context = _context(context)
    captured = _fields(_json(capture), _CAPTURE)
    _require(captured["schema"] == "main-unittest.capture.v1")
    inner_context = _context(captured["context"])
    _require(inner_context == context, "STALE_CONTEXT")
    native = _fields(captured["native"], _NATIVE)
    expected_native = dict(event_name="push", ref=_REF, repository=context["repository"],
        sha=context["checked_commit"], run_id=context["run_id"], run_attempt=context["run_attempt"],
        workflow_ref=context["repository"] + "/" + _WORKFLOW + "@" + _REF,
        workflow_sha=context["checked_commit"])
    _require(all(type(native[key]) is str and native[key] == expected_native[key]
                 for key in _NATIVE), "STALE_CONTEXT")
    specs, binding = _needs(needs, context)
    _require(_digest(captured["needs_sha256"]) == _sha(_encode(needs)))
    _require(type(archives) is dict and set(archives) == _ARCHIVES
             and all(type(raw) is bytes for raw in archives.values()))
    archives = dict(archives)
    base = "https://api.github.com/repos/" + context["repository"]
    downloads = captured["downloads"]
    _require(type(downloads) is list and len(downloads) == 3)
    download_paths = []
    for row in downloads:
        _fields(row, _DOWNLOAD)
        _process_row(row)
        _require(all(type(row[key]) is str for key in ("profile", "mode", "id", "name")))
        _decimal(row["id"])
        path = row["path"]
        _require(path in specs and path not in download_paths)
        download_paths.append(path)
        spec, raw = specs[path], archives[path]
        _require((row["profile"], row["mode"], row["id"], row["name"])
                 == (spec["profile"], spec["mode"], spec["id"], spec["name"]), "ARCHIVE_MISMATCH")
        _require(row["url"] == base + "/actions/artifacts/" + spec["id"] + "/zip", "STALE_CONTEXT")
        _require(22 <= len(raw) <= _ARCHIVE_LIMIT, "ARCHIVE_MISMATCH")
        _stats(row, raw, "ARCHIVE_MISMATCH")
        _require(row["sha256"] == spec["sha256"], "ARCHIVE_MISMATCH")
    _require(download_paths == sorted(_ARCHIVES))
    urls = {
        "api-run-attempt.json": base + "/actions/runs/" + context["run_id"] + "/attempts/" + context["run_attempt"],
        "api-attempt-jobs.json": base + "/actions/runs/" + context["run_id"] + "/attempts/" + context["run_attempt"] + "/jobs?per_page=100",
        "api-artifacts.json": base + "/actions/runs/" + context["run_id"] + "/artifacts?per_page=100",
        "api-git-commit.json": base + "/git/commits/" + context["checked_commit"],
    }
    for spec in specs.values():
        urls["api-artifact-" + spec["id"] + ".json"] = base + "/actions/artifacts/" + spec["id"]
    _require(type(originals) is dict and set(originals) == set(urls)
             and all(type(raw) is bytes for raw in originals.values()))
    originals = dict(originals)
    responses = captured["responses"]
    _require(type(responses) is list and len(responses) == 7)
    response_paths, api = [], {}
    for row in responses:
        _fields(row, _ROW)
        _process_row(row)
        path = row["path"]
        _require(type(path) is str and path in urls and path not in response_paths)
        response_paths.append(path)
        _require(row["url"] == urls[path], "STALE_CONTEXT")
        raw = originals[path]
        _stats(row, raw, "INVALID_CAPTURE")
        api[path] = _json(raw)
    _require(response_paths == sorted(urls))
    repository_id, run_status = _run(api["api-run-attempt.json"], context)
    _jobs(api["api-attempt-jobs.json"], context)
    commit = api["api-git-commit.json"]
    _require(type(commit) is dict and commit.get("sha") == context["checked_commit"]
             and type(commit.get("tree")) is dict
             and commit["tree"].get("sha") == context["checked_tree"], "STALE_CONTEXT")
    listing = api["api-artifacts.json"]
    _require(type(listing) is dict and type(listing.get("artifacts")) is list, "INCOMPLETE")
    listed = listing["artifacts"]
    _require(_integer(listing.get("total_count"), "INCOMPLETE") == len(listed) <= 100, "INCOMPLETE")
    identities = set()
    for item in listed:
        _require(type(item) is dict and type(item.get("name")) is str, "INCOMPLETE")
        identity = _positive(item.get("id"), "INCOMPLETE")
        _require(identity not in identities, "INCOMPLETE")
        identities.add(identity)
    for path, spec in specs.items():
        individual = api["api-artifact-" + spec["id"] + ".json"]
        _artifact(individual, spec, context, repository_id, base, archives[path])
        matches = [item for item in listed if item["name"] == spec["name"]]
        _require(len(matches) == 1, "ARCHIVE_MISMATCH")
        _artifact(matches[0], spec, context, repository_id, base, archives[path])
    producer = dict(core_result="success", architecture_result="success", architecture_outputs=binding)
    selected_paths = {"api-run-attempt.json"} | {
        "api-artifact-" + spec["id"] + ".json" for spec in specs.values()}
    four = [{key: row[key] for key in ("path", "url", "bytes", "sha256")}
            for row in responses if row["path"] in selected_paths]
    normalized = dict(schema="main-unittest.native.v1", context=context, producer=producer,
        api_originals=four,
        artifacts=[dict(profile=row["profile"], mode=row["mode"], id=row["id"], name=row["name"],
                        run_id=context["run_id"], head_sha=context["checked_commit"], expired=False,
                        digest=row["sha256"], archive_bytes=row["bytes"]) for row in downloads])
    # The genuine unchanged helper performs core and strict ZIP admission.
    # No zipfile reader has touched any supplied archive before this call.
    try:
        data_root = Path(root).resolve()
    except (TypeError, OSError, RuntimeError):
        _fail("SOURCE_MISMATCH")
    joined = _ADMIT(data_root, context, producer,
                    (normalized, {path: originals[path] for path in selected_paths}),
                    {mode: archives["core-" + mode + ".zip"] for mode in ("normal", "optimized")},
                    archives["architecture.zip"])
    _project_guard()
    result = dict(
        schema="main-unittest.acquisition-check.v1", context=context, status="INERT_CAPTURE_VERIFIED",
        transport_authority=_AUTHORITY, completion_authority=_AUTHORITY,
        supplied_run_status=run_status, native_completion_required=True,
        scientific_effect="NONE", g5_claim=False, cbz_claim=False, capture_sha256=_sha(capture),
        api_originals=[{key: row[key] for key in ("path", "url", "bytes", "sha256")} for row in responses],
        archives=[{key: row[key] for key in ("path", "profile", "mode", "id", "name", "bytes", "sha256")}
                  for row in downloads], join=joined)
    return result, data_root, context, capture, originals, archives


def _checked(root, context, needs, capture, originals, archives):
    try:
        return _verify(root, context, needs, capture, originals, archives)
    except ValueError as error:
        if str(error) in _REASONS:
            raise
        _fail("INVALID_CAPTURE")
    except (TypeError, KeyError, IndexError, AttributeError, OSError, UnicodeError,
            OverflowError, RecursionError, zipfile.BadZipFile):
        _fail("INVALID_CAPTURE")


def verify_capture(root: Path, context: dict, needs: dict, capture: bytes,
                   originals: dict[str, bytes], archives: dict[str, bytes]) -> dict:
    """Verify supplied-byte consistency; award no HTTP or completion authority."""
    return _checked(root, context, needs, capture, originals, archives)[0]


def _write(output, raw):
    written = output.write(raw)
    if type(written) is not int or written != len(raw):
        raise OSError("original frame write incomplete")


def write_original_frames(root: Path, context: dict, needs: dict, capture: bytes,
                          originals: dict[str, bytes], archives: dict[str, bytes],
                          output: BinaryIO) -> dict:
    """Admit first, then write the complete original member bytes as inert frames."""
    _require(callable(getattr(output, "write", None)))
    check, data_root, context, capture, originals, archives = _checked(
        root, context, needs, capture, originals, archives)
    try:
        bodies = {"capture.json": capture}
        bodies.update({"api/" + path: raw for path, raw in originals.items()})
        for path, raw in archives.items():
            # This is original-byte readback AFTER the actual existing helper
            # and reader admitted exact geometry, member inventory and bounds.
            # It is not a second ZIP security parser and extracts no files.
            with zipfile.ZipFile(io.BytesIO(raw)) as stream:
                for name in stream.namelist():
                    bodies[path.removesuffix(".zip") + "/" + name] = stream.read(name)
        # Bind the retained GRAPH original to the source census actually
        # admitted by the real join, rather than trust a later path read alone.
        census_raw = bodies["core-normal/source-before.json"]
        _require(_sha(census_raw) == check["join"]["source_sha256"], "SOURCE_MISMATCH")
        census = json.loads(census_raw.decode("ascii"))
        graph_rows = [row for row in census["files"]
                      if row["path"] == "docs/site/dependency-source/GRAPH.json"]
        _require(len(graph_rows) == 1, "SOURCE_MISMATCH")
        graph = _FULL._read(data_root / "docs/site/dependency-source/GRAPH.json", _SOURCE_LIMIT)
        _require(len(graph) == graph_rows[0]["bytes"]
                 and _sha(graph) == graph_rows[0]["sha256"], "SOURCE_MISMATCH")
        bodies["source/GRAPH.json"] = graph
        frames, rows = [], []
        for path, raw in sorted(bodies.items()):
            try:
                raw.decode("utf-8")
                encoding, body = "utf-8", raw
            except UnicodeDecodeError:
                encoding, body = "base64", base64.b64encode(raw)
            header = dict(path=path, original_bytes=len(raw), sha256=_sha(raw), encoding=encoding)
            footer = dict(path=path, printed_original_bytes=len(raw), sha256=_sha(raw))
            frames.append((body, b"NATIVE_ORIGINAL_BEGIN " + _encode(header),
                           b"NATIVE_ORIGINAL_END " + _encode(footer)))
            rows.append(dict(path=path, original_bytes=len(raw), sha256=_sha(raw),
                             encoding=encoding, printed_original_bytes=len(raw)))
        while True:
            nonce = os.urandom(24).hex()
            terminal = b"::" + nonce.encode("ascii") + b"::"
            if all(terminal not in body for body, _, _ in frames):
                break
        _project_guard()
    except ValueError as error:
        if str(error) in _REASONS:
            raise
        _fail("INVALID_CAPTURE")
    except (TypeError, KeyError, OSError, UnicodeError, zipfile.BadZipFile, RuntimeError):
        _fail("ARCHIVE_MISMATCH")
    # Everything is admitted and staged before the first output byte.
    _write(output, b"::stop-commands::" + nonce.encode("ascii") + b"\n")
    for body, header, footer in frames:
        _write(output, header)
        _write(output, body)
        _write(output, b"\n")
        _write(output, footer)
    _write(output, terminal + b"\n")
    flush = getattr(output, "flush", None)
    if callable(flush):
        flush()
    _project_guard()
    return dict(schema="main-unittest.original-frames.v1", context=context,
                capture_sha256=check["capture_sha256"], transport_authority=_AUTHORITY,
                complete=True, members=rows)
