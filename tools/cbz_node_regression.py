"""Run all tracked Node tests with byte-verified historical display fixtures.

Operational test evidence only; no mathematical or final CBZ certification.
The report records a real subprocess attempt and endpoint source checks.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from datetime import datetime, timezone
from urllib.parse import quote

import museum_data

ROOT = Path(__file__).resolve().parents[1]
_LOADED_SOURCE = Path(__file__).read_bytes()
_LOADED_SOURCES = {
    Path(__file__).resolve(): _LOADED_SOURCE,
    Path(museum_data.__file__).resolve(): Path(museum_data.__file__).read_bytes(),
}
_REPOSITORIES = {
    "d6g8k5htny-coder/main",
    "d6g8k5htny-coder/Math-",
    "d6g8k5htny-coder/query-",
}
_SUMMARY_FIELDS = ("tests", "pass", "fail", "cancelled", "skipped", "todo")


def assert_loaded_sources(main_root):
    expected = {main_root.resolve() / "tools/cbz_node_regression.py",
                main_root.resolve() / "tools/museum_data.py"}
    if set(_LOADED_SOURCES) != expected:
        raise ValueError("loaded source root identity mismatch")
    for path, raw in _LOADED_SOURCES.items():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError("loaded source bytes changed")


def git_env():
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_NO_REPLACE_OBJECTS="1", GIT_NO_LAZY_FETCH="1",
               GIT_OPTIONAL_LOCKS="0")
    return env


def git(root, *args):
    process = subprocess.run(["git", "-C", str(root), *args],
                             capture_output=True, env=git_env())
    if process.returncode:
        raise ValueError("Git command failed: " + " ".join(args))
    return process.stdout


def no_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key: " + key)
        value[key] = item
    return value


def read_pinned(roots, descriptor):
    """Read a regular immutable Git blob and verify all declared identities."""
    if not isinstance(descriptor, dict):
        raise ValueError("invalid source descriptor")
    repository = descriptor.get("repository")
    commit = descriptor.get("commit")
    path = descriptor.get("path")
    length = descriptor.get("bytes")
    digest = descriptor.get("sha256")
    if (repository not in _REPOSITORIES or repository not in roots
            or not isinstance(commit, str)
            or not re.fullmatch(r"[0-9a-f]{40}", commit)
            or not isinstance(path, str) or not path
            or PurePosixPath(path).is_absolute()
            or any(part in ("", ".", "..") for part in path.split("/"))
            or "\0" in path or "\\" in path
            or type(length) is not int or length <= 0
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("invalid source descriptor")
    expected_url = ("https://raw.githubusercontent.com/" + repository + "/"
                    + commit + "/" + quote(path, safe="/"))
    if "url" in descriptor and descriptor["url"] != expected_url:
        raise ValueError("pin URL/identity mismatch")
    blob = descriptor.get("blob")
    if blob is not None and (not isinstance(blob, str)
                            or not re.fullmatch(r"[0-9a-f]{40}", blob)):
        raise ValueError("invalid Git blob identity")
    root = roots[repository]
    tree = git(root, "ls-tree", "-z", commit, "--", path).split(b"\0")
    if len(tree) != 2 or tree[1]:
        raise ValueError("missing or ambiguous pinned Git blob")
    metadata, actual_path = tree[0].split(b"\t", 1)
    mode, kind, oid = metadata.split()
    if (mode not in (b"100644", b"100755") or kind != b"blob"
            or actual_path.decode("utf-8") != path):
        raise ValueError("pinned source is not a regular Git blob")
    raw = git(root, "cat-file", "blob", commit + ":" + path)
    actual_blob = hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    if (len(raw) != length or hashlib.sha256(raw).hexdigest() != digest
            or actual_blob != oid.decode("ascii")
            or (blob is not None and actual_blob != blob)):
        raise ValueError("source identity mismatch: " + path)
    return raw


def materialize(descriptors, roots, destination):
    """Verify every descriptor before writing any source fixture."""
    verified = [(descriptor, read_pinned(roots, descriptor))
                for descriptor in descriptors]
    paths = {}
    for descriptor, raw in verified:
        key = (descriptor["repository"].rsplit("/", 1)[1], descriptor["path"])
        if key in paths and paths[key] != raw:
            raise ValueError("conflicting fixture identities")
        paths[key] = raw
    for (repository, relative), raw in paths.items():
        target = destination / repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return verified


def source_snapshot(root):
    """Check owned worktree bytes even when Git's index hides their changes."""
    head = git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    if git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("main source worktree or index is not clean")
    if git(root, "diff", "--cached", "--name-only", "HEAD", "--"):
        raise ValueError("main index differs from HEAD")
    for entry in git(root, "ls-tree", "-r", "-z", "HEAD").split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        path = root / os.fsdecode(raw_path)
        if mode in (b"100644", b"100755") and kind == b"blob":
            if path.is_symlink() or not path.is_file():
                raise ValueError("main tracked file missing or changed type")
            raw = path.read_bytes()
            executable = bool(path.stat().st_mode & 0o111)
            if executable != (mode == b"100755"):
                raise ValueError("main tracked executable mode changed")
        else:
            raise ValueError("main contains an unsupported tracked Git entry")
        actual = hashlib.sha1(
            b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        if actual.encode("ascii") != oid:
            raise ValueError("main tracked bytes differ from HEAD: "
                             + os.fsdecode(raw_path))
    return head


def tap_summary(text):
    values = {}
    for name in _SUMMARY_FIELDS:
        matches = re.findall(r"^# " + name + r" ([0-9]+)\s*$", text, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError("missing or ambiguous Node TAP summary: " + name)
        values[name] = int(matches[0])
    if not (values["tests"] > 0 and values["pass"] == values["tests"]
            and all(values[name] == 0 for name in _SUMMARY_FIELDS[2:])):
        raise ValueError("Node tests failed, were cancelled, or were skipped/todo")
    return values


def node_command(root, node):
    entries = git(root, "ls-files", "--stage", "-z", "--", "tests/*.mjs")
    paths = []
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        if mode not in (b"100644", b"100755") or stage != b"0":
            raise ValueError("regular stage-zero Node test files are required")
        paths.append(os.fsdecode(path))
    if not paths:
        raise ValueError("no tracked Node tests")
    return [node, "--test", "--test-reporter=tap", *sorted(paths)]


def verify_fixture_tree(output, expected):
    """Require the exact regular fixture/manifest file and directory domain."""
    expected_dirs = {"shop", "museum"}
    for relative in expected:
        path = PurePosixPath(relative)
        if (path.is_absolute() or path.parts[0] not in expected_dirs
                or any(part in ("", ".", "..") for part in relative.split("/"))):
            raise ValueError("invalid fixture expectation")
        expected_dirs.update(str(parent) for parent in path.parents
                             if str(parent) != ".")
    def walk_error(error):
        raise error

    actual_files = {}
    actual_dirs = set()
    for label in ("shop", "museum"):
        root = output / label
        if root.is_symlink() or not root.is_dir():
            raise ValueError("fixture root missing or changed type")
        for directory, directories, files in os.walk(root, followlinks=False, onerror=walk_error):
            parent = Path(directory)
            actual_dirs.add(parent.relative_to(output).as_posix())
            for name in directories:
                actual_dirs.add((parent / name).relative_to(output).as_posix())
                if (parent / name).is_symlink():
                    raise ValueError("fixture directory is a symlink")
            for name in files:
                path = parent / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError("fixture file missing or changed type")
                raw = path.read_bytes()
                actual_files[path.relative_to(output).as_posix()] = dict(
                    bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    if actual_files != expected or actual_dirs != expected_dirs:
        raise ValueError("fixture set or bytes changed")


class FixturePostflightError(ValueError):
    def __init__(self, error, exit_code):
        super().__init__(str(error))
        self.exit_code = exit_code


def execute_node(root, command, environment, log, fixture_root, expected_fixtures):
    verify_fixture_tree(fixture_root, expected_fixtures)
    process = subprocess.run(command, cwd=root, env=environment,
                             stdout=log, stderr=subprocess.STDOUT)
    try:
        verify_fixture_tree(fixture_root, expected_fixtures)
    except (OSError, ValueError) as error:
        raise FixturePostflightError(error, process.returncode) from error
    return process.returncode


def write_report(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(main_root, math_root, query_root, output, node="node"):
    """Execute the whole tracked Node suite, preserving failures and output."""
    roots = {
        "d6g8k5htny-coder/main": main_root.resolve(),
        "d6g8k5htny-coder/Math-": math_root.resolve(),
        "d6g8k5htny-coder/query-": query_root.resolve(),
    }
    output = output.resolve()
    if any(output == root or output.is_relative_to(root)
           for root in roots.values()):
        raise ValueError("output must be outside every source root")
    output.mkdir(parents=True, exist_ok=False)
    record = dict(schema="cbz.node-regression.v1", status="RUNNING",
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  mathematical_certification=False, final_cbz_certification=False,
                  runtime=dict(python=sys.version, implementation=sys.implementation.name,
                               platform=sys.platform, machine=platform.machine()))
    report = output / "result.json"
    write_report(report, record)
    log = output / "node.tap"
    try:
        assert_loaded_sources(main_root)
        record["source_before"] = source_snapshot(main_root)
        config = json.loads((main_root / "docs/site/config.json").read_bytes(),
                            object_pairs_hook=no_duplicates)
        shop = [config[name] for name in ("coefficient", "proof", "imports", "query")]
        shop_verified = materialize(shop, roots, output / "shop")
        museum = materialize(list(museum_data.SOURCES.values()), roots,
                             output / "museum")
        mapping = {
            museum_data.descriptor(descriptor["path"])["url"]:
                base64.b64encode(raw).decode("ascii")
            for descriptor, raw in museum
        }
        manifest = output / "museum/verified-inputs.json"
        manifest_raw = (json.dumps(mapping, sort_keys=True) + "\n").encode("utf-8")
        manifest.write_bytes(manifest_raw)
        expected_fixtures = {}
        for label, verified in (("shop", shop_verified), ("museum", museum)):
            for descriptor, raw in verified:
                repository = descriptor["repository"].rsplit("/", 1)[1]
                relative = label + "/" + repository + "/" + descriptor["path"]
                expected_fixtures[relative] = dict(
                    bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        expected_fixtures["museum/verified-inputs.json"] = dict(
            bytes=len(manifest_raw), sha256=hashlib.sha256(manifest_raw).hexdigest())
        verify_fixture_tree(output, expected_fixtures)
        record["fixture_files"] = len(expected_fixtures)
        record["fixture_tree_sha256"] = hashlib.sha256(json.dumps(
            expected_fixtures, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        museum_data.check_export(
            main_root / "docs/site/museum.json",
            museum_data.build(output / "museum/main", output / "museum/Math-"))
        environment = {key: value for key, value in git_env().items()
                       if key not in ("NODE_OPTIONS", "NODE_PATH")}
        environment.update(
            SHOP_MATH_FIXTURE=str(output / "shop/Math-"),
            SHOP_QUERY_FIXTURE=str(output / "shop/query-"),
            MUSEUM_FIXTURE=str(manifest),
        )
        command = node_command(main_root, node)
        record["node_version"] = subprocess.check_output([node, "--version"], env=environment, text=True).strip()
        record["command"] = command
        record["fixture_environment"] = {
            key: environment[key]
            for key in ("SHOP_MATH_FIXTURE", "SHOP_QUERY_FIXTURE", "MUSEUM_FIXTURE")
        }
        record["fixtures"] = [dict(repository=d["repository"], commit=d["commit"],
                                   path=d["path"], bytes=len(raw),
                                   sha256=hashlib.sha256(raw).hexdigest())
                              for d, raw in shop_verified
                              ] + [dict(repository=d["repository"], commit=d["commit"],
                                        path=d["path"], bytes=len(raw),
                                        sha256=hashlib.sha256(raw).hexdigest())
                                   for d, raw in museum]
        write_report(report, record)
        with log.open("wb") as stream:
            record["node_exit"] = execute_node(
                main_root, command, environment, stream, output, expected_fixtures)
        record["log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
        assert_loaded_sources(main_root)
        record["source_after"] = source_snapshot(main_root)
        if record["source_after"] != record["source_before"]:
            raise ValueError("main HEAD changed during Node execution")
        if record["node_exit"] != 0:
            raise ValueError("Node exited unsuccessfully")
        record["summary"] = tap_summary(log.read_text(encoding="utf-8"))
        record["status"] = "PASS"
    except Exception as error:
        record["status"] = "FAIL"
        record["error"] = str(error)
        if isinstance(error, FixturePostflightError):
            record["node_exit"] = error.exit_code
        if log.is_file():
            record["log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(report, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-root", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if Path(__file__).read_bytes() != _LOADED_SOURCE:
        parser.error("loaded runner differs from its source bytes")
    try:
        record = run(ROOT, args.math_root, args.query_root, args.output)
    except (OSError, ValueError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps({key: record[key] for key in ("status", "schema")}))
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
