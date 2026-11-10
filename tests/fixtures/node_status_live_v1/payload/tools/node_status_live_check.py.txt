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


if __name__ != "__main__":
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


def _observation_report(observer, requested):
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
        if validator.PINNED_URL.fullmatch(url):
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
                    and validator.OBSERVED.fullmatch(row["updated_at"]),
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check the four actual node records using authenticated live source observations.",
        allow_abbrev=False,
    )
    parser.add_argument("--root", help="executed tool checkout (default: this tool's checkout)")
    parser.add_argument("--output", required=True, help="new JSON file outside the selected checkout")
    arguments = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    started_at, started_clock = _cli_now(), time.monotonic()
    report = {
        "schema": "cbz.node-status-live-cli.v1", "status": "REFUSED",
        "scope": "NODE_RECORD_FRESHNESS_ONLY", "scientific_effect": "NONE",
        "cbz_green": False, "started_at": started_at,
    }
    before = None
    output = None
    code = None
    try:
        root = _cli_root(arguments.root)
        identity = _cli_context()
        before = _cli_bind(root)
        if before["checked_commit"] != identity["checked_commit"]:
            raise _CLIRefusal("CHECKED_COMMIT_MISMATCH")
        report.update(
            identity=identity, source_bindings=before["files"],
            checked_commit_before=before["checked_commit"],
            checked_tree_before=before["checked_tree"], source_bound_at=_cli_now(),
        )
        output = _cli_output(arguments.output, root)
        token = os.environ.get("GITHUB_TOKEN")
        if not isinstance(token, str) or not token.strip():
            raise _CLIRefusal("AUTH_REQUIRED")
        _load_project_dependencies(before["source_bytes"], root)
        report["observation_started_at"] = _cli_now()
        try:
            freshness = run_live(root, identity=identity)
        finally:
            report["observation_finished_at"] = _cli_now()
        report["freshness"] = freshness
        report["status"] = "FRESH"
    except _CLIRefusal as error:
        code = error.code
    except Exception:
        # Raw transport/import errors are never echoed; they can contain request
        # or environment details. An error grants no freshness credit.
        code = "FRESHNESS_CHECK_FAILED"
    if before is not None:
        try:
            after = _cli_bind(root)
            report.update(
                source_bindings_after=after["files"],
                checked_commit_after=after["checked_commit"],
                checked_tree_after=after["checked_tree"], source_rechecked_at=_cli_now(),
            )
            if after != before:
                raise _CLIRefusal("SOURCE_BYTES_CHANGED")
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
