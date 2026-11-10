#!/usr/bin/env python3
"""Q-9006 real-record freshness helper and source-bound live companion.

Offline helper controls never earn live credit. The CLI binds its actual
checkout before importing project dependencies and calls only run_live's
authenticated default production HTTPS observer. Scientific admission is outside
this entrypoint's scope.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit

def _load_project_dependencies(source_bytes=None, root=None):
    # Ordinary helper imports retain their existing public API. The CLI instead
    # executes only bytes captured while matching these exact files to HEAD.
    global refresh, ContentObserver, validator
    if source_bytes is None:
        import node_status_refresh as refresh
        from node_status_refresh import ContentObserver
        import validate_node_status as validator
        return
    from types import ModuleType
    for name in ("validate_node_status", "node_status_refresh"):
        relative = "tools/" + name + ".py"
        module = ModuleType(name)
        module.__file__ = str(root / relative)
        module.__package__ = ""
        # The refresh module's validator import must reuse this bound module.
        sys.modules[name] = module
        exec(compile(source_bytes[relative], module.__file__, "exec",
                     dont_inherit=True), module.__dict__)
    validator = sys.modules["validate_node_status"]
    refresh = sys.modules["node_status_refresh"]
    ContentObserver = refresh.ContentObserver


if __name__ != "__main__" and not globals().get("_CAPTURE_BOOTSTRAP", False):
    _load_project_dependencies()

RECORD_NAMES = (
    "eng.d0-packet-allowlist-fix.json",
    "eng.hard-gate.json",
    "math.lifetime-remainder.json",
    "math.side24-coefficient.json",
)
METADATA = ("architecture/node-status.schema.json", "docs/site/current.json")
IDENTITY_FIELDS = {"repository", "checked_commit", "run_id", "run_attempt", "python_optimize"}
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
POSITIVE = re.compile(r"[1-9][0-9]*\Z")


class LiveCheckError(ValueError):
    """A required input or actual freshness comparison could not be accepted."""


def require(condition, message):
    if not condition:
        raise LiveCheckError(message)


def _identity(value):
    require(isinstance(value, dict) and set(value) == IDENTITY_FIELDS,
            "invalid live-check identity fields")
    require(value["repository"] == "d6g8k5htny-coder/main", "wrong source repository")
    require(isinstance(value["checked_commit"], str)
            and HEX40.fullmatch(value["checked_commit"]), "immutable checked commit required")
    for key in ("run_id", "run_attempt"):
        require(isinstance(value[key], str) and POSITIVE.fullmatch(value[key]),
                "positive native " + key + " required")
    require(type(value["python_optimize"]) is int and value["python_optimize"] in (0, 1),
            "interpreter mode must be integer 0 or 1")
    return dict(value)


def _path(root, relative):
    path = root
    for part in Path(relative).parts:
        require(part not in ("", ".", ".."), "noncanonical selected input path")
        path = path / part
        require(not path.is_symlink(), "symlink selected input refused")
    return path


def _read(root, relative, limit):
    path = _path(root, relative)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= limit,
            "selected input is not a bounded regular file: " + relative)
    raw = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(tuple(getattr(before, name) for name in fields)
            == tuple(getattr(after, name) for name in fields)
            and len(raw) == before.st_size, "selected input changed during read: " + relative)
    return raw


def _capture(root):
    require(root.is_dir() and not root.is_symlink(), "selected root is missing or symlink")
    directory = _path(root, "records/node_status")
    require(directory.is_dir(), "mandatory records directory missing")
    require({path.name for path in directory.iterdir()} == set(RECORD_NAMES),
            "selected four-record input set differs")
    captured = {
        "records/node_status/" + name: _read(root, "records/node_status/" + name, validator.MAX_BYTES)
        for name in RECORD_NAMES
    }
    for relative in METADATA:
        limit = 2_000_000 if relative.endswith("current.json") else validator.MAX_BYTES
        captured[relative] = _read(root, relative, limit)
    return captured


def _validated_records(captured):
    # Production validation observes the same captured bytes, not a second
    # mutable read from the selected checkout. Only this owned temporary is written.
    with tempfile.TemporaryDirectory(prefix="node-status-live-input-") as temporary:
        root = Path(temporary)
        for relative, raw in captured.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        problems = validator.validate_directory(root, root / "records/node_status")
        require(not problems, "production record/schema validation refused: " + "; ".join(problems))
        return [
            validator.load_json(root / "records/node_status" / name)
            for name in RECORD_NAMES
        ]


def _observation_report(observer, requested, validator_module=None):
    checked_validator = validator if validator_module is None else validator_module
    observations = getattr(observer, "observations", None)
    heads = getattr(observer, "heads", None)
    require(isinstance(observations, list) and isinstance(heads, dict),
            "observer did not retain observations and heads")
    require(all(isinstance(row, dict) and isinstance(row.get("url"), str)
                for row in observations), "malformed source observation")
    require(Counter(row["url"] for row in observations) == Counter(url for url, _ in requested),
            "source observation counters or URL multiplicities differ")
    git_repositories = set()
    for row in observations:
        url = row["url"]
        if checked_validator.PINNED_URL.fullmatch(url):
            parts = urlsplit(url).path.split("/")
            repository, kind = parts[2], parts[3]
            git_repositories.add(repository)
            head = heads.get(repository)
            require(isinstance(head, str) and HEX40.fullmatch(head)
                    and row.get("head") == head, "observation differs from captured repository head")
            before, after = row.get("pinned_object"), row.get("current_object")
            require(isinstance(before, (tuple, list)) and isinstance(after, (tuple, list))
                    and len(before) == len(after) == 3, "incomplete Git object observation")
            require(before[0] == after[0] == kind
                    and before[1] == after[1]
                    and before[1] in (("040000",) if kind == "tree" else ("100644", "100755"))
                    and isinstance(before[2], str) and HEX40.fullmatch(before[2])
                    and tuple(before) == tuple(after), "cited Git object is not unchanged")
        else:
            require(isinstance(row.get("updated_at"), str)
                    and checked_validator.OBSERVED.fullmatch(row["updated_at"]),
                    "discussion timestamp observation missing")
            expected_revisions = {revision for source, revision in requested if source == url}
            require(expected_revisions == {row["updated_at"]},
                    "retained discussion timestamp differs from captured source revision")
    require(set(heads) == git_repositories, "captured repository head coverage differs")
    # Freeze returned evidence values, so a later caller change cannot rewrite this report.
    return copy.deepcopy(heads), copy.deepcopy(observations)


def evaluate_fresh_records(root, *, observer, identity):
    """Check real captured inputs using the supplied source-observation boundary.

    A supplied observer or identity is not proof of actual HTTPS or hosted
    execution. The separately qualified live entrypoint must establish those.
    """
    try:
        identity = _identity(identity)
        root = Path(root).absolute()
        captured = _capture(root)
        records = _validated_records(captured)
        fresh = [record for record in records if record["status"] == "fresh"]
        require(fresh, "no real declared-fresh candidate; live coverage is unexercised")
        require(callable(observer) and getattr(observer, "observations", None) == [],
                "a fresh observation boundary is required for this mode")
        requested = []
        checked = []
        for record in fresh:
            requested.extend((source["url"], source["rev"]) for source in record["sources"])
            result = refresh.refresh_record(record, observer)
            require(result.get("status") == "fresh" and not result.get("refresh_error"),
                    "declared-fresh source is stale, unknown or unavailable: " + record["node_id"])
            require(result.get("source_rev") == record["source_rev"]
                    and result.get("sources") == record["sources"],
                    "immutable source pins changed during observation")
            checked.append(record["node_id"])
        require(len(checked) == len(fresh), "fresh cohort was not completely checked")
        heads, observations = _observation_report(observer, requested)
        require(_capture(root) == captured, "selected record/schema/graph bytes changed during observation")
        return {
            "schema": "cbz.node-status-live-check.v1",
            "scope": "NODE_RECORD_FRESHNESS_ONLY",
            "records_total": len(records),
            "fresh_candidates": len(fresh),
            "fresh_checked": len(checked),
            "fresh_nodes": sorted(checked),
            "nonfresh_records": sorted(
                ({"node_id": record["node_id"], "declared_status": record["status"]}
                 for record in records if record["status"] != "fresh"),
                key=lambda row: row["node_id"],
            ),
            "input_sha256": {
                name: hashlib.sha256(captured["records/node_status/" + name]).hexdigest()
                for name in RECORD_NAMES
            },
            "schema_sha256": hashlib.sha256(captured[METADATA[0]]).hexdigest(),
            "graph_sha256": hashlib.sha256(captured[METADATA[1]]).hexdigest(),
            "identity": identity,
            "observed_heads": heads,
            "source_observation_count": len(observations),
            "source_observations": observations,
            "scientific_effect": "NONE",
            "cbz_green": False,
        }
    except LiveCheckError:
        raise
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise LiveCheckError("freshness check refused: " + str(error)) from error


def run_live(root, *, identity):
    """Require authentication and construct exactly one default HTTPS observer.

    There is deliberately no observer/API/transcript parameter on this wrapper,
    no anonymous fallback and no retry.
    """
    token = os.environ.get("GITHUB_TOKEN")
    require(isinstance(token, str) and bool(token.strip()), "GITHUB_TOKEN is required")
    return evaluate_fresh_records(root, observer=ContentObserver(), identity=identity)


SELECTED_SOURCE_FILES = (
    "tools/node_status_live_check.py",
    "tools/node_status_refresh.py",
    "tools/validate_node_status.py",
    *METADATA,
    *("records/node_status/" + name for name in RECORD_NAMES),
)


class _CLIRefusal(Exception):
    def __init__(self, code):
        self.code = code


def _cli_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _cli_git(root, *arguments):
    # Only this child environment is changed. Credential/proxy settings and
    # repository/global configuration are neither rewritten nor reported.
    environment = dict(os.environ)
    selectors = {
        "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE", "GIT_CEILING_DIRECTORIES",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM", "GIT_PREFIX", "GIT_SHALLOW_FILE",
        "GIT_REPLACE_REF_BASE", "GIT_CONFIG", "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
    }
    for key in list(environment):
        if key in selectors or re.fullmatch(r"GIT_CONFIG_(KEY|VALUE)_[0-9]+", key):
            environment.pop(key)
    environment.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0",
                       GIT_NO_REPLACE_OBJECTS="1", GIT_NO_LAZY_FETCH="1")
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", "--literal-pathspecs",
             "-c", "core.fsmonitor=false", "-C", str(root), *arguments],
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise _CLIRefusal("SOURCE_BYTES_CHANGED") from None
    if result.returncode:
        raise _CLIRefusal("SOURCE_BYTES_CHANGED")
    return result.stdout


def _cli_root(selected):
    entrypoint = Path(os.path.abspath(__file__))
    root = entrypoint.parent.parent
    candidate = root if selected is None else Path(os.path.abspath(selected))
    if candidate != root:
        raise _CLIRefusal("ROOT_MISMATCH")
    for component in (root, *root.parents):
        if component.is_symlink() or not component.is_dir():
            raise _CLIRefusal("ROOT_MISMATCH")
    top = _cli_git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    if Path(top).resolve() != root:
        raise _CLIRefusal("ROOT_MISMATCH")
    return root


def _cli_context():
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise _CLIRefusal("RUN_CONTEXT_INVALID")
    value = {
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "checked_commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "python_optimize": sys.flags.optimize,
    }
    try:
        return _identity(value)
    except LiveCheckError:
        raise _CLIRefusal("RUN_CONTEXT_INVALID") from None


def _cli_bind(root):
    try:
        commit = _cli_git(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
        if not HEX40.fullmatch(commit):
            raise _CLIRefusal("SOURCE_BYTES_CHANGED")
        tree = _cli_git(root, "rev-parse", "--verify", commit + "^{tree}").decode("ascii").strip()
        if not HEX40.fullmatch(tree):
            raise _CLIRefusal("SOURCE_BYTES_CHANGED")
        bindings = {}
        source_bytes = {}
        for relative in SELECTED_SOURCE_FILES:
            entry = _cli_git(root, "ls-tree", "-z", commit, "--", relative)
            metadata, name = entry.rstrip(b"\0").split(b"\t", 1)
            mode, kind, oid = metadata.decode("ascii").split(" ")
            if (name != relative.encode("utf-8") or kind != "blob"
                    or mode not in ("100644", "100755") or not HEX40.fullmatch(oid)):
                raise _CLIRefusal("SOURCE_BYTES_CHANGED")
            committed = _cli_git(root, "cat-file", "blob", oid)
            raw = _read(root, relative, 2_000_000)
            if raw != committed:
                raise _CLIRefusal("SOURCE_BYTES_CHANGED")
            if relative in ("tools/validate_node_status.py", "tools/node_status_refresh.py"):
                source_bytes[relative] = raw
            bindings[relative] = {
                "git_blob": oid, "git_mode": mode,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        if _cli_git(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip() != commit:
            raise _CLIRefusal("SOURCE_BYTES_CHANGED")
        return {"checked_commit": commit, "checked_tree": tree, "files": bindings,
                "source_bytes": source_bytes}
    except (OSError, UnicodeError, ValueError, LiveCheckError):
        raise _CLIRefusal("SOURCE_BYTES_CHANGED") from None


def _cli_output(output, root):
    # Walk the original absolute parent with no-follow directory descriptors;
    # resolving the path first would hide an inadmissible symlink.
    path = Path(os.path.abspath(output))
    if path.is_relative_to(root):
        raise _CLIRefusal("OUTPUT_INVALID")
    descriptor = None
    accepted = False
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open(path.anchor, flags)
        for component in path.parent.parts[1:]:
            following = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        try:
            os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            accepted = True
            return descriptor, path.name
        raise _CLIRefusal("OUTPUT_INVALID")
    except (OSError, ValueError):
        raise _CLIRefusal("OUTPUT_INVALID") from None
    finally:
        if descriptor is not None and not accepted:
            os.close(descriptor)


def _cli_write(descriptor, name, raw):
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=descriptor)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        raise _CLIRefusal("OUTPUT_INVALID") from None


# Current v2 is an explicit native-input boundary. The historical helpers above
# retain their four-record semantics; no missing-manifest fallback is provided.
V2_CODE_PATHS = (
    "tools/node_status_live_check.py", "tools/node_status_refresh.py",
    "tools/node_status_v1.py", "tools/validate_node_status.py",
)
V2_SCHEMA_PATHS = (
    "architecture/node-status.schema.json", "architecture/node-status-v1.schema.json",
)
V2_SCHEMA_BLOBS = (
    "2bad95d30bcb89ea686d1ac5a50d52113dd730be",
    "05f5dbf110497bbc89fd6030df2c21c1d8ae52ac",
)
V2_MANIFEST = "architecture/node-status-inputs.json"
V2_SNAPSHOT = "docs/site/current.json"
V2_RAW_GRAPH = "docs/site/dependency-source/GRAPH.json"
V2_ARTIFACTS = (
    "cbz/evidence/hard-gate-source-refresh-20261110/README.md",
    "cbz/evidence/hard-gate-source-refresh-20261110/current-SOURCE_FILES.json",
    "cbz/evidence/hard-gate-source-refresh-20261110/observation.json",
)
V2_SEEDS = tuple(name[:-5] for name in RECORD_NAMES)
V2_MATH = "d6g8k5htny-coder/Math-"
V2_PIN = "dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde"
V2_GRAPH_PATH = "frontiers/downstream_gate_20260925/GRAPH.json"
V2_GRAPH_BLOB = "220b56867f24c5ef6f55a3cce70490e9777cf3af"
V2_GRAPH_BYTES = 38753
V2_GRAPH_SHA256 = "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09"


def _v2_git(root, *arguments):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                       GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0",
                       GIT_NO_REPLACE_OBJECTS="1", GIT_NO_LAZY_FETCH="1")
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", "--literal-pathspecs", "-c",
             "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
             "-C", str(root), *arguments], env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30, check=False)
        require(result.returncode == 0, "native Git source operation refused")
        return result.stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise LiveCheckError("native Git source operation unavailable") from error


def _v2_root(root):
    root = Path(os.path.abspath(root))
    require(all(not item.is_symlink() and item.is_dir()
                for item in (root, *root.parents)), "noncanonical native source root")
    top = _v2_git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    require(Path(top).resolve(strict=True) == root, "native Git top-level differs")
    return root


def _v2_cut(root):
    commit = _v2_git(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    require(HEX40.fullmatch(commit), "native commit identity required")
    tree = _v2_git(root, "rev-parse", "--verify", commit + "^{tree}").decode("ascii").strip()
    require(HEX40.fullmatch(tree), "native tree identity required")
    return commit, tree


def _v2_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def _v2_native_entry(root, commit, relative):
    row = _v2_git(root, "ls-tree", "--full-tree", "-z", commit, "--", relative)
    require(row.count(b"\0") == 1 and row.endswith(b"\0"), "native input path missing")
    metadata, name = row[:-1].split(b"\t", 1)
    mode, kind, oid = metadata.decode("ascii").split(" ")
    require(name == relative.encode("ascii") and kind == "blob"
            and mode in ("100644", "100755") and HEX40.fullmatch(oid),
            "native input path, kind or mode differs")
    return mode, oid


def _v2_read(root, relative, limit, expected_mode):
    require(type(relative) is str and relative
            and all(part not in ("", ".", "..") for part in relative.split("/"))
            and not relative.startswith("/"), "noncanonical selected path")
    descriptor = None
    opened = None
    try:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        parts = relative.split("/")
        for part in parts[:-1]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        opened = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        before = os.fstat(opened)
        require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= limit,
                "selected file is not bounded regular data")
        require(bool(before.st_mode & 0o111) == (expected_mode == "100755"),
                "physical executable mode differs from native Git")
        with os.fdopen(opened, "rb") as stream:
            opened = None
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(all(getattr(before, key) == getattr(after, key) for key in fields)
                and len(raw) == before.st_size and len(raw) <= limit,
                "selected input changed during bounded capture")
        current = os.stat(parts[-1], dir_fd=descriptor, follow_symlinks=False)
        require(all(getattr(after, key) == getattr(current, key) for key in fields),
                "selected pathname changed during capture")
        return raw
    finally:
        if opened is not None:
            os.close(opened)
        if descriptor is not None:
            os.close(descriptor)


def _v2_file(root, commit, relative, limit):
    mode, oid = _v2_native_entry(root, commit, relative)
    size = _v2_git(root, "cat-file", "-s", oid).decode("ascii").strip()
    require(size.isdecimal() and int(size) <= limit, "native input object exceeds bound")
    raw = _v2_read(root, relative, limit, mode)
    require(_v2_blob(raw) == oid and _v2_git(root, "cat-file", "blob", oid) == raw,
            "working bytes differ from the actual native object")
    return raw, dict(git_blob=oid, git_mode=mode, sha256=hashlib.sha256(raw).hexdigest())


def _v2_json(raw, limit):
    import math
    require(type(raw) is bytes and 0 < len(raw) <= limit, "bounded original JSON required")
    text = raw.decode("utf-8", errors="strict")
    depth, quoted, escaped = 0, False, False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            require(depth <= 64, "JSON depth exceeds current input bound")
        elif character in "]}":
            depth -= 1
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate original JSON key")
            result[key] = value
        return result
    def floating(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite original JSON number")
        return result
    def constant(_value):
        raise LiveCheckError("nonfinite original JSON constant")
    return json.loads(text, object_pairs_hook=pairs, parse_float=floating,
                      parse_constant=constant)


def _v2_same(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_v2_same(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(_v2_same(a, b) for a, b in zip(left, right))
    return left == right


def _v2_manifest(raw):
    value = _v2_json(raw, 65536)
    keys = {"schema", "profile", "records", "schema_paths", "code_paths", "graph", "artifact_paths"}
    require(type(value) is dict and set(value) == keys, "current manifest fields differ")
    require(type(value["schema"]) is str and value["schema"] == "node-status-inputs.v2"
            and type(value["profile"]) is str
            and value["profile"] in ("legacy-four-v0", "graph-closure-v1"),
            "unsupported current manifest schema or profile")
    require(type(value["code_paths"]) is list and value["code_paths"] == list(V2_CODE_PATHS),
            "manifest cannot select executable project paths")
    schemas = value["schema_paths"]
    require(type(schemas) is list and len(schemas) == 2, "closed schema path list required")
    for version, (row, path) in enumerate(zip(schemas, V2_SCHEMA_PATHS)):
        require(type(row) is dict and set(row) == {"record_version", "path"}
                and type(row["record_version"]) is int and row["record_version"] == version
                and type(row["path"]) is str and row["path"] == path, "closed schema path policy differs")
    require(type(value["graph"]) is dict and value["graph"] == dict(
        snapshot_path=V2_SNAPSHOT, raw_path=V2_RAW_GRAPH), "fixed Graph paths required")
    expected_artifacts = [] if value["profile"] == "legacy-four-v0" else list(V2_ARTIFACTS)
    require(type(value["artifact_paths"]) is list
            and value["artifact_paths"] == expected_artifacts, "fixed artifact path policy differs")
    rows = value["records"]
    require(type(rows) is list and 0 < len(rows) <= 256, "bounded current records list required")
    nodes = []
    for row in rows:
        require(type(row) is dict and set(row) == {"node_id", "path", "record_version"},
                "closed current record entry required")
        node = row["node_id"]
        require(type(node) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", node)
                and type(row["path"]) is str
                and row["path"] == "records/node_status/" + node + ".json"
                and type(row["record_version"]) is int and row["record_version"] in (0, 1),
                "invalid current record identity, path or primitive version")
        nodes.append(node)
    require(nodes == sorted(set(nodes)), "duplicate or unsorted current record entries")
    return value


def _v2_graph(raw, snapshot_raw):
    graph = _v2_json(raw, 2_000_000)
    snapshot = _v2_json(snapshot_raw, 2_000_000)
    require(type(graph) is dict and type(graph.get("nodes")) is dict
            and type(graph.get("edges")) is list, "invalid captured Graph")
    selected = snapshot.get("graph") if type(snapshot) is dict else None
    require(type(selected) is dict and set(selected) == {"identity", "data"},
            "current snapshot Graph representation differs")
    identity = selected["identity"]
    expected = dict(repository=V2_MATH, commit=V2_PIN, path=V2_GRAPH_PATH,
                    blob=V2_GRAPH_BLOB, bytes=V2_GRAPH_BYTES, sha256=V2_GRAPH_SHA256)
    require(_v2_same(identity, expected), "snapshot cannot select another pinned Math Graph")
    require(len(raw) == V2_GRAPH_BYTES and _v2_blob(raw) == V2_GRAPH_BLOB
            and hashlib.sha256(raw).hexdigest() == V2_GRAPH_SHA256,
            "raw Graph differs from the reviewed pinned object")
    require(_v2_same(selected["data"], graph), "full embedded Graph data differs from captured raw Graph")
    return graph, dict(repository=V2_MATH, commit=V2_PIN, path=V2_GRAPH_PATH,
                       git_blob=V2_GRAPH_BLOB, bytes=len(raw), sha256=V2_GRAPH_SHA256)


def _v2_cohort(graph, profile):
    nodes, edges = graph["nodes"], graph["edges"]
    require(all(type(node) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", node)
                and type(value) is dict for node, value in nodes.items())
            and set(V2_SEEDS) <= nodes.keys(), "fixed seed nodes absent from Graph")
    outgoing, seen = {node: [] for node in nodes}, set()
    for edge in edges:
        require(type(edge) is dict and set(edge) == {"from", "to", "relation", "required"}
                and type(edge["from"]) is str and edge["from"] in nodes
                and type(edge["to"]) is str and edge["to"] in nodes
                and type(edge["relation"]) is str and bool(edge["relation"])
                and type(edge["required"]) is bool, "invalid required Graph edge")
        key = (edge["from"], edge["to"], edge["relation"], edge["required"])
        require(key not in seen, "duplicate Graph edge")
        seen.add(key)
        if edge["required"]:
            outgoing[edge["from"]].append(edge["to"])
    cohort, pending = set(), list(V2_SEEDS)
    while pending:
        node = pending.pop()
        if node not in cohort:
            cohort.add(node)
            if profile == "graph-closure-v1":
                pending.extend(outgoing[node])
    return sorted(cohort)


def _v2_directory(root, record_paths):
    directory = _path(root, "records/node_status")
    require(directory.is_dir() and not directory.is_symlink(), "current records directory missing")
    names = set()
    for item in directory.iterdir():
        require(len(names) < 256, "current record directory exceeds bounded population")
        names.add(item.name)
    require(names == {path.rsplit("/", 1)[1] for path in record_paths},
            "complete current record directory differs from independent cohort")


def _v2_modules(root, raw):
    from types import ModuleType
    names = ("node_status_v1", "validate_node_status", "node_status_refresh")
    sentinel = object()
    previous = {name: sys.modules.get(name, sentinel) for name in names}
    modules = {}
    try:
        for name in names:
            relative = "tools/" + name + ".py"
            module = ModuleType(name)
            module.__file__ = str(root / relative)
            module.__package__ = ""
            sys.modules[name] = module
            exec(compile(raw[relative], module.__file__, "exec", dont_inherit=True), module.__dict__)
            modules[name] = module
    finally:
        for name in names:
            if previous[name] is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]
    return modules


def _v2_artifacts(root, commit, raw, files, records):
    evidence = records["eng.hard-gate"].get("artifact_evidence")
    require(type(evidence) is list and len(evidence) == 3, "complete hard-gate artifact evidence required")
    frames = []
    for path, claim in zip(V2_ARTIFACTS, evidence):
        require(type(claim) is dict and type(claim.get("commit")) is str
                and HEX40.fullmatch(claim["commit"]), "immutable historical artifact commit required")
        historical = claim["commit"]
        require(_v2_git(root, "rev-parse", "--verify", historical + "^{commit}").decode("ascii").strip()
                == historical, "historical artifact commit differs")
        _v2_git(root, "merge-base", "--is-ancestor", historical, commit)
        mode, oid = _v2_native_entry(root, historical, path)
        original = raw[path]
        require(mode == files[path]["git_mode"] and oid == files[path]["git_blob"]
                and _v2_blob(original) == oid, "historical artifact native path object differs")
        size = _v2_git(root, "cat-file", "-s", oid).decode("ascii").strip()
        require(size.isdecimal() and int(size) <= 2_000_000
                and _v2_git(root, "cat-file", "blob", oid) == original,
                "historical artifact bytes differ")
        frames.append(dict(identity=dict(path=path, repository="d6g8k5htny-coder/main",
            commit=historical, git_mode=mode, git_blob=oid, bytes=len(original),
            sha256=hashlib.sha256(original).hexdigest()), raw=original))
    return frames


def _capture_v2(root):
    """Capture and locally validate the closed native current population; no network."""
    try:
        root = _v2_root(root)
        commit, tree = _v2_cut(root)
        raw, files = {}, {}
        bootstrap = (*V2_CODE_PATHS, *V2_SCHEMA_PATHS, V2_MANIFEST, V2_SNAPSHOT, V2_RAW_GRAPH)
        for path in bootstrap:
            raw[path], files[path] = _v2_file(
                root, commit, path, 65536 if path == V2_MANIFEST else 2_000_000)
        if globals().get("_CAPTURE_BOOTSTRAP"):
            require(globals().get("_CAPTURE_BOOTSTRAP_RAW") == raw[V2_CODE_PATHS[0]],
                    "executed native bootstrap differs from captured live source")
        manifest = _v2_manifest(raw[V2_MANIFEST])
        graph, graph_identity = _v2_graph(raw[V2_RAW_GRAPH], raw[V2_SNAPSHOT])
        cohort = _v2_cohort(graph, manifest["profile"])
        version = 0 if manifest["profile"] == "legacy-four-v0" else 1
        expected_records = [dict(node_id=node, path="records/node_status/" + node + ".json",
                                 record_version=version) for node in cohort]
        require(_v2_same(manifest["records"], expected_records),
                "manifest records differ from independently derived current cohort")
        record_paths = [row["path"] for row in expected_records]
        _v2_directory(root, record_paths)
        for path in (*record_paths, *manifest["artifact_paths"]):
            raw[path], files[path] = _v2_file(
                root, commit, path, 65536 if path in record_paths else 2_000_000)
        require(_v2_cut(root) == (commit, tree), "native source cut changed before project execution")
        for path, expected_blob in zip(V2_SCHEMA_PATHS, V2_SCHEMA_BLOBS):
            require(files[path]["git_blob"] == expected_blob
                    and type(_v2_json(raw[path], 2_000_000)) is dict,
                    "reviewed closed record schema changed")
        records = {}
        for row in expected_records:
            record = _v2_json(raw[row["path"]], 65536)
            require(type(record) is dict and type(record.get("schema_version")) is int
                    and record["schema_version"] == version, "current record version/profile differs")
            records[row["node_id"]] = record
        source_inputs = dict(schema="node-status-v1.source-inputs.v1",
                             graph=dict(identity=graph_identity, raw=raw[V2_RAW_GRAPH]), artifacts=[])
        if version == 1:
            source_inputs["artifacts"] = _v2_artifacts(root, commit, raw, files, records)
        modules = _v2_modules(root, raw)
        if version == 0:
            for row in expected_records:
                problems = modules["validate_node_status"].check_record(
                    records[row["node_id"]], row["node_id"] + ".json", set(graph["nodes"]))
                require(not problems, "current v0 record refused: " + "; ".join(problems))
        else:
            records = modules["node_status_v1"].check_v1_population(
                [(row["node_id"] + ".json", raw[row["path"]]) for row in expected_records],
                source_inputs)
        captured = dict(checked_commit=commit, checked_tree=tree, files=files, raw=raw,
            manifest=manifest, records=records, source_inputs=source_inputs, modules=modules,
            profile=manifest["profile"], record_paths=record_paths)
        _recapture_v2(root, captured)
        return captured
    except LiveCheckError:
        raise
    except Exception as error:
        raise LiveCheckError("current native input validation refused: " + str(error)) from error


def _recapture_v2(root, captured):
    """Recheck only inert native bytes/context; execute no newly read module."""
    try:
        root = _v2_root(root)
        expected_cut = (captured["checked_commit"], captured["checked_tree"])
        require(_v2_cut(root) == expected_cut, "native source cut changed before final capture")
        _v2_directory(root, captured["record_paths"])
        raw, files = {}, {}
        for path in captured["raw"]:
            limit = 65536 if path == V2_MANIFEST or path in captured["record_paths"] else 2_000_000
            raw[path], files[path] = _v2_file(root, expected_cut[0], path, limit)
        require(raw == captured["raw"] and files == captured["files"]
                and _v2_cut(root) == expected_cut, "native current inputs changed")
        return dict(checked_commit=expected_cut[0], checked_tree=expected_cut[1], files=files, raw=raw)
    except LiveCheckError:
        raise
    except Exception as error:
        raise LiveCheckError("current native input recheck refused: " + str(error)) from error


def _pinned_graph_binding_v2(captured, observer):
    """Join fixed Math pinned Git object to local originals; never record counters."""
    try:
        require(callable(getattr(observer, "object", None)), "native pinned Graph resolver required")
        native = observer.object("Math-", V2_PIN, V2_GRAPH_PATH, "blob")
        require(tuple(native) == ("blob", "100644", V2_GRAPH_BLOB),
                "native pinned Math Graph differs from the captured original")
        raw = captured["raw"][V2_RAW_GRAPH]
        require(len(raw) == V2_GRAPH_BYTES and _v2_blob(raw) == native[2]
                and hashlib.sha256(raw).hexdigest() == V2_GRAPH_SHA256,
                "native pinned Math Graph does not join local raw bytes")
        return dict(scope="PINNED_MATH_GRAPH_OBJECT_IDENTITY_ONLY", repository=V2_MATH,
            commit=V2_PIN, path=V2_GRAPH_PATH, git_kind=native[0], git_mode=native[1],
            git_blob=native[2], bytes=len(raw), sha256=V2_GRAPH_SHA256,
            snapshot_path=V2_SNAPSHOT, raw_path=V2_RAW_GRAPH,
            snapshot_git_blob=captured["files"][V2_SNAPSHOT]["git_blob"],
            raw_git_blob=captured["files"][V2_RAW_GRAPH]["git_blob"],
            full_data_equal=True, complete=True)
    except LiveCheckError:
        raise
    except Exception as error:
        raise LiveCheckError("native pinned Math Graph lookup refused") from error


def _evaluate_capture_v2(root, captured, observer, identity):
    identity = _identity(identity)
    require(identity["checked_commit"] == captured["checked_commit"]
            and identity["python_optimize"] == sys.flags.optimize,
            "current identity differs from actual checked source or interpreter")
    require(callable(observer) and getattr(observer, "observations", None) == [],
            "a new empty source observer is required")
    pinned = _pinned_graph_binding_v2(captured, observer)
    _recapture_v2(root, captured)
    records = list(captured["records"].values())
    fresh = [record for record in records if record["status"] == "fresh"]
    require(fresh, "no declared-fresh current candidate; coverage is unexercised")
    requested, checked = [], []
    for record in fresh:
        requested.extend((source["url"], source["rev"]) for source in record["sources"])
        result = captured["modules"]["node_status_refresh"].refresh_record(record, observer)
        require(result.get("status") == "fresh" and not result.get("refresh_error"),
                "declared-fresh current source is stale, unknown or unavailable")
        require(result.get("sources") == record["sources"]
                and result.get("source_rev") == record["source_rev"], "immutable current sources changed")
        checked.append(record["node_id"])
    heads, observations = _observation_report(
        observer, requested, validator_module=captured["modules"]["validate_node_status"])
    _recapture_v2(root, captured)
    manifest_raw = captured["raw"][V2_MANIFEST]
    return dict(schema="cbz.node-status-live-check.v2", scope="NODE_RECORD_FRESHNESS_ONLY",
        records_total=len(records), record_versions={r["node_id"]: r["schema_version"] for r in records},
        fresh_candidates=len(fresh), fresh_checked=len(checked), fresh_nodes=sorted(checked),
        nonfresh_records=sorted((dict(node_id=r["node_id"], declared_status=r["status"])
            for r in records if r["status"] != "fresh"), key=lambda row: row["node_id"]),
        source_bindings=copy.deepcopy(captured["files"]),
        manifest_binding=dict(profile=captured["profile"], path=V2_MANIFEST,
            bytes=len(manifest_raw), **captured["files"][V2_MANIFEST]),
        input_sha256={path.rsplit("/", 1)[1]: hashlib.sha256(captured["raw"][path]).hexdigest()
                      for path in captured["record_paths"]},
        identity=identity, observed_heads=heads, source_observation_count=len(observations),
        source_observations=observations, pinned_graph_binding=pinned,
        scientific_effect="NONE", cbz_green=False)


def evaluate_fresh_records_v2(root, *, observer, identity):
    """Injected consistency boundary; supplied replies/context do not authenticate HTTP."""
    try:
        captured = _capture_v2(root)
        return _evaluate_capture_v2(root, captured, observer, identity)
    except LiveCheckError:
        raise
    except Exception as error:
        raise LiveCheckError("current freshness refused: " + str(error)) from error


def run_live_v2(root, *, identity):
    """One default observer after local admission; no provider/transcript override."""
    token = os.environ.get("GITHUB_TOKEN")
    require(type(token) is str and bool(token.strip()), "GITHUB_TOKEN is required")
    captured = _capture_v2(root)
    return _evaluate_capture_v2(root, captured, ContentObserver(), identity)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check the mandatory current native manifest population using live sources.",
        allow_abbrev=False)
    parser.add_argument("--root", help="executed tool checkout (default: this tool's checkout)")
    parser.add_argument("--output", required=True, help="new JSON file outside the selected checkout")
    arguments = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    started_at, started_clock = _cli_now(), time.monotonic()
    report = dict(schema="cbz.node-status-live-cli.v2", status="REFUSED",
                  scope="NODE_RECORD_FRESHNESS_ONLY", scientific_effect="NONE",
                  cbz_green=False, started_at=started_at)
    before, output, code = None, None, None
    try:
        root = _cli_root(arguments.root)
        identity = _cli_context()
        try:
            before = _capture_v2(root)
        except LiveCheckError:
            raise _CLIRefusal("VALIDATION_REFUSED") from None
        if before["checked_commit"] != identity["checked_commit"]:
            raise _CLIRefusal("CHECKED_COMMIT_MISMATCH")
        report.update(identity=identity, source_bindings=before["files"],
            checked_commit_before=before["checked_commit"], checked_tree_before=before["checked_tree"],
            source_bound_at=_cli_now())
        output = _cli_output(arguments.output, root)
        token = os.environ.get("GITHUB_TOKEN")
        if type(token) is not str or not token.strip():
            raise _CLIRefusal("AUTH_REQUIRED")
        global validator, refresh, ContentObserver
        validator = before["modules"]["validate_node_status"]
        refresh = before["modules"]["node_status_refresh"]
        ContentObserver = refresh.ContentObserver
        report["observation_started_at"] = _cli_now()
        try:
            freshness = run_live_v2(root, identity=identity)
        finally:
            report["observation_finished_at"] = _cli_now()
        report.update(freshness=freshness, status="FRESH")
    except _CLIRefusal as error:
        code = error.code
    except Exception:
        code = "FRESHNESS_CHECK_FAILED"
    if before is not None:
        try:
            after = _recapture_v2(root, before)
            report.update(source_bindings_after=after["files"],
                checked_commit_after=after["checked_commit"], checked_tree_after=after["checked_tree"],
                source_rechecked_at=_cli_now())
        except Exception:
            code = "SOURCE_BYTES_CHANGED"
            if output is not None:
                os.close(output[0])
                output = None
    if code:
        report.update(status="REFUSED", error_code=code)
        report.pop("freshness", None)
    report.update(finished_at=_cli_now(),
        elapsed_seconds=round(max(0.0, time.monotonic() - started_clock), 6))
    raw = json.dumps(report, sort_keys=True, indent=2) + "\n"
    try:
        if output is not None:
            try:
                _cli_write(*output, raw)
            except _CLIRefusal as error:
                code = error.code
                report.update(status="REFUSED", error_code=code)
                report.pop("freshness", None)
                raw = json.dumps(report, sort_keys=True, indent=2) + "\n"
    finally:
        if output is not None:
            os.close(output[0])
    sys.stdout.write(raw)
    return 1 if code else 0


if __name__ == "__main__":
    raise SystemExit(main())
