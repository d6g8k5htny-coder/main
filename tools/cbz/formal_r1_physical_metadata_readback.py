"""Fixed Phase2 readback of 56 original formal R1 physical records.
Only the two native workflow keys select their closed original cohorts.
No Lean, trace parser, formal helper, policy or fixture is executed.
Large snapshots/censuses stay unread; original R1 HOLD is unchanged.
"""
import base64
import binascii
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import secrets
import selectors
import signal
import stat
import subprocess
import sys
import time
import traceback
import zipfile

PURPOSE = "DIAGNOSTIC_EXISTING_FORMAL_R1_ONLY"
REPOSITORY = "d6g8k5htny-coder/main"
BRANCH = "refs/heads/codex/g5-formal-r1-evidence-readback-20261110"
RUN, ATTEMPT, JOB, ARTIFACT = 42803771651, 1, 129787665051, 13758500307
CHECKED = "b1b8267f6774f4f8c536ee8c54a00bcbcd5271b1"
TREE = "50cf08822e0e642b9f21aa7efb8c62fa6fc7859e"
PARENTS = ["90093a2e61cf53271570db5707162e1c3176d773"]
SUBJECT = "02973f3975a1c4cc91b7c51d372f65717c9b3c5c"
SUBJECT_TREE = "5d3160d982b4513f17848de45e0f4569233acd4e"
ZIP_SIZE = 16104541
ZIP_SHA256 = "a18fd4260fb3dc8f694ea24744a87a4933a21a3b48c5cafa0a8cd9d56d7ee2d0"
MIB = 1024 * 1024
OUTPUT_LIMIT = 2 * MIB
ORIGINAL_BUDGET = 3 * MIB // 2
METHODS = [
    "tests.test_formal_gate.RepositoryControls.test_real_package_executes_with_receipt",
    "tests.test_formal_compiled_inventory.PinnedLeanInventoryTests.test_actual_clean_and_omitted_constants",
]
PINS = {
    "tests/test_formal_gate.py": "aa8acd86661183a4d0b97704efaf025ee385bc3e",
    "tests/test_formal_compiled_inventory.py": "ce3b77bebd084c711acf5509ca56410ea3e56071",
    "tools/formal_gate_check.py": "aa2f90376f349ec6fca3d35a115990829d3d3f54",
    "tools/formal_compiled_inventory.py": "c037a8c3cdab114089a37f67aeb5d109277083ea",
    "formal/manifest.json": "7bd51fe201ade413fcd18314854c40ee4aae3df6",
    "formal/lake-manifest.json": "fc7655fad5ede55ad51ac99d0801a13a21f78dd7",
    "formal/lakefile.toml": "d986aa8d776636f3f47deaa6d17d41e79eb5b0d3",
    "formal/lean-toolchain": "ba8ebf2dbaf6a668cd2a0e086186d6d569b69ff5",
}
SMALL = ("probe.json", "cleanup.json", "control.json", "runtime-binaries.json",
         "observer-error.log") + tuple(
    mode + "/" + name for mode in ("normal", "optimized")
    for name in ("observation.json", "stdout.bin", "stderr.bin", "observer-error.log"))
SNAPSHOTS = tuple(mode + "/" + name for mode in ("normal", "optimized")
                  for name in ("source-preparation.json", "source-before.json", "source-after.json"))
CENSUSES = ("normal/write-census.json", "optimized/write-census.json")
REQUESTS = (
    ("original-attempt.json", "repos/d6g8k5htny-coder/main/actions/runs/42803771651/attempts/1", 128 * 1024),
    ("original-jobs.json", "repos/d6g8k5htny-coder/main/actions/runs/42803771651/attempts/1/jobs?per_page=100", 128 * 1024),
    ("original-artifact.json", "repos/d6g8k5htny-coder/main/actions/artifacts/13758500307", 64 * 1024),
    ("original-controller.json", "repos/d6g8k5htny-coder/main/git/commits/b1b8267f6774f4f8c536ee8c54a00bcbcd5271b1", 64 * 1024),
    ("original-artifact.zip", "repos/d6g8k5htny-coder/main/actions/artifacts/13758500307/zip", ZIP_SIZE),
)


PHASE_SCOPE = "ORIGINAL_FORMAL_R1_PHYSICAL_METADATA_PHASE2_ONLY"
UTILITY_PATH = "tools/cbz/formal_r1_physical_metadata_readback.py"
WORKFLOW_PATH = ".github/workflows/formal-r1-physical-metadata-readback.yml"
CONTROLLER_PINS = {
    ".github/workflows/research-architecture.yml": "69ad3b47537ab8e4f8936affa3e09fccbdcfd1e5",
    "tools/unittest_timing_probe.py": "05fb481b5538c3f517174b00948de45d2849387a",
    "tools/architecture_run_receipt.py": "a6062afa9261958b8d8b85853a9c43031385b2c1",
    "architecture/graph_artifact_readback.py": "c39ffd0fdcde1fd5f895c57df5f770a582918a8a",
    "tools/architecture_check_binding.py": "53ea34e4fffc1087770584744939904844a41cff",
    "tools/unittest_full_qualification.py": "02dc2f3222aaf3e3d4d024659fffff22d6fb5e66",
    "tests/test_unittest_full_qualification.py": "2fe09abce9a8ef9356fd0af85780dcde75179386",
    ".github/workflows/full-unittest-qualification.yml": "ff8226cdeea57ee2ddb41c58c8c93e644343557d"
}
GROUP_MEMBERS = {
    "root-bindings": (
        ("bind-0.json", 473, 2686347395),
        ("bind-0.stderr.bin", 0, 0),
        ("bind-0.stdout.bin", 0, 0),
        ("bind-1.json", 559, 3809019906),
        ("bind-1.stderr.bin", 0, 0),
        ("bind-1.stdout.bin", 0, 0),
        ("bind-2.json", 565, 4078398837),
        ("bind-2.stderr.bin", 0, 0),
        ("bind-2.stdout.bin", 0, 0),
        ("bind-3.json", 579, 1252192405),
        ("bind-3.stderr.bin", 0, 0),
        ("bind-3.stdout.bin", 0, 0),
        ("bind-4.json", 571, 2070444718),
        ("bind-4.stderr.bin", 0, 0),
        ("bind-4.stdout.bin", 0, 0),
        ("bind-5.json", 584, 1727103956),
        ("bind-5.stderr.bin", 0, 0),
        ("bind-5.stdout.bin", 0, 0),
        ("exec-scratch-normal.json", 485, 3903588992),
        ("exec-scratch-normal.stderr.bin", 0, 0),
        ("exec-scratch-normal.stdout.bin", 0, 0),
        ("exec-scratch-optimized.json", 491, 3903450239),
        ("exec-scratch-optimized.stderr.bin", 0, 0),
        ("exec-scratch-optimized.stdout.bin", 0, 0),
        ("seal-1.json", 452, 1513435214),
        ("seal-1.stderr.bin", 0, 0),
        ("seal-1.stdout.bin", 0, 0),
        ("seal-2.json", 495, 2209885371),
        ("seal-2.stderr.bin", 0, 0),
        ("seal-2.stdout.bin", 0, 0),
        ("seal-3.json", 498, 2930899557),
        ("seal-3.stdout.bin", 0, 0),
        ("seal-3.stderr.bin", 0, 0),
        ("seal-5.json", 501, 1249936189),
        ("seal-5.stderr.bin", 0, 0),
        ("seal-5.stdout.bin", 0, 0),
        ("worktree-normal.json", 596, 2286950063),
        ("worktree-normal.stderr.bin", 1239, 3250050599),
        ("worktree-normal.stdout.bin", 111, 975072424),
        ("worktree-optimized.json", 597, 2041474490),
        ("worktree-optimized.stderr.bin", 44, 2081881607),
        ("worktree-optimized.stdout.bin", 111, 975072424),
    ),
    "poststate": (
        ("normal/formal-evidence/receipt.json", 12938, 4150319522),
        ("normal/generated-after.json", 22730, 3007644920),
        ("normal/mounts-after.json", 4549, 3924997965),
        ("normal/mounts-before.json", 4549, 3924997965),
        ("normal/retention-cleanup-1.json", 8632, 1246506967),
        ("normal/retention-cleanup-2.json", 8632, 1246506967),
        ("normal/scratch-after.json", 272, 2626406170),
        ("optimized/formal-evidence/receipt.json", 12938, 1329515730),
        ("optimized/generated-after.json", 22733, 98261878),
        ("optimized/mounts-before.json", 4549, 3924997965),
        ("optimized/mounts-after.json", 4549, 3924997965),
        ("optimized/retention-cleanup-1.json", 8632, 2048079539),
        ("optimized/retention-cleanup-2.json", 8632, 2048079539),
        ("optimized/scratch-after.json", 275, 3386255626),
    ),
}
GROUP_COUNTS = {"root-bindings": 42, "poststate": 14}
GROUP_RAW_BYTES = {"root-bindings": 8951, "poststate": 124610}
GROUP_ENCODED_BYTES = {"root-bindings": 12062, "poststate": 167566}
EXPECTED_ZIP_ENTRIES = 247
SUMMARY_LIMIT = 131072
FRAMING_LIMIT = 16384
CONTROLLER_LIMIT = 1572864
SOURCE_GIT_LIMIT = 16 * MIB
ROOT = None
out = None
observation = None
context = None

def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")

def require(condition, reason):
    if not condition:
        raise ValueError(reason)

def request(index):
    # The request table above is closed; no CLI argument, API response, or
    # archive member can supply a command, endpoint, host, or output path.
    name, endpoint, limit = REQUESTS[index]
    environment = os.environ.copy()
    environment.pop("GH_TOKEN", None)
    environment.pop("GH_DEBUG", None)
    environment.update(GH_HOST="github.com", GH_PROMPT_DISABLED="1", GH_PAGER="cat",
                       NO_COLOR="1", CLICOLOR="0")
    require(bool(environment.get("GITHUB_TOKEN")), "native GITHUB_TOKEN is absent")
    command = ["gh", "api", "--method", "GET", "--hostname", "github.com", endpoint]
    chunks = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": limit, "stderr": 64 * 1024}
    record = {"name": name, "endpoint": endpoint, "method": "GET",
              "status": "STARTED", "stdout_complete": False, "stderr_complete": False}
    observation["requests"].append(record)
    process = None
    error = None
    selector = selectors.DefaultSelector()
    deadline = time.monotonic() + 90
    try:
        process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        for channel, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, channel)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("fixed gh GET exceeded 90 seconds")
            for key, _ in selector.select(min(remaining, 1.0)):
                channel = key.data
                allowance = limits[channel] + 1 - len(chunks[channel])
                data = os.read(key.fileobj.fileno(), min(64 * 1024, allowance))
                if not data:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    record[channel + "_complete"] = True
                else:
                    chunks[channel].extend(data)
                    if len(chunks[channel]) > limits[channel]:
                        raise ValueError("fixed gh GET " + channel + " exceeds byte bound")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("fixed gh GET exceeded 90 seconds")
        record["exit_code"] = process.wait(timeout=remaining)
        require(record["exit_code"] == 0, "fixed gh GET returned nonzero native exit")
        record["status"] = "COMPLETE"
    except Exception as caught:
        error = caught
        record["status"] = "NATIVE_ERROR"
        record["error_type"] = type(caught).__name__
        record["error"] = str(caught)
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        selector.close()
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
            record.setdefault("exit_code", process.returncode)
        for channel in ("stdout", "stderr"):
            raw = bytes(chunks[channel])
            record[channel + "_bytes_retained"] = len(raw)
            record[channel + "_sha256"] = hashlib.sha256(raw).hexdigest()
            # Partial/error responses have their explicit completeness flags.
            # They are never passed to metadata or archive acceptance.
            suffix = "" if channel == "stdout" and record["status"] == "COMPLETE" else ".gh-" + channel + ".bin"
            (out / (name + suffix)).write_bytes(raw)
        (out / "observation.json").write_bytes(encoded(observation))
    if error is not None:
        raise error
    return bytes(chunks["stdout"])

def parse(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate metadata key")
            value[key] = item
        return value
    def reject_constant(value):
        raise ValueError("non-finite metadata number")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                       parse_constant=reject_constant)
    require(type(value) is dict, "metadata is not an object")
    return value

def verify_api(attempt, jobs, artifact, checked):
    require(attempt.get("id") == RUN and attempt.get("run_attempt") == ATTEMPT
            and attempt.get("head_sha") == CHECKED
            and attempt.get("head_commit", {}).get("tree_id") == TREE
            and attempt.get("repository", {}).get("id") == 1117225690
            and attempt.get("repository", {}).get("full_name") == REPOSITORY
            and attempt.get("head_repository", {}).get("id") == 1117225690
            and attempt.get("event") == "push"
            and attempt.get("head_branch") == "codex/g5-formal-runtime-feasibility-20261110"
            and attempt.get("path") == ".github/workflows/g5-formal-runtime-feasibility.yml"
            and attempt.get("status") == "completed" and attempt.get("conclusion") == "failure",
            "fixed R1 attempt binding mismatch")
    require(jobs.get("total_count") == 1 and type(jobs.get("jobs")) is list
            and len(jobs["jobs"]) == 1, "fixed R1 job population mismatch")
    job = jobs["jobs"][0]
    require(job.get("id") == JOB and job.get("run_id") == RUN
            and job.get("run_attempt") == ATTEMPT and job.get("head_sha") == CHECKED
            and job.get("name") == "Observe the two unchanged real Lean methods"
            and job.get("status") == "completed" and job.get("conclusion") == "failure",
            "fixed R1 job binding mismatch")
    steps = job.get("steps")
    require(type(steps) is list, "fixed R1 steps missing")
    for number, name, conclusion in (
        (4, "Observe the fixed original methods under the initial observation limit", "failure"),
        (5, "Stop fixed probe processes and release its mounts", "success"),
        (6, "Print retained original material without replay", "success"),
        (7, "Retain actual observations and failures", "success"),
    ):
        rows = [row for row in steps if row.get("number") == number]
        require(len(rows) == 1 and rows[0].get("name") == name
                and rows[0].get("status") == "completed"
                and rows[0].get("conclusion") == conclusion, "fixed R1 step binding mismatch")
    native = artifact.get("workflow_run", {})
    require(artifact.get("id") == ARTIFACT
            and artifact.get("name") == "formal-runtime-feasibility-42803771651-1"
            and artifact.get("size_in_bytes") == ZIP_SIZE
            and artifact.get("digest") == "sha256:" + ZIP_SHA256
            and artifact.get("expired") is False
            and native.get("id") == RUN and native.get("repository_id") == 1117225690
            and native.get("head_repository_id") == 1117225690
            and native.get("head_sha") == CHECKED, "fixed R1 artifact binding mismatch")
    require(checked.get("sha") == CHECKED and checked.get("tree", {}).get("sha") == TREE
            and [row.get("sha") for row in checked.get("parents", [])] == PARENTS
            and checked.get("verification", {}).get("verified") is False
            and checked.get("verification", {}).get("reason") == "unsigned",
            "fixed R1 controller commit binding mismatch")

def member_bytes(source, item, limit):
    require(item.file_size <= limit, "selected original exceeds fixed read ceiling")
    with source.open(item, "r") as stream:
        raw = stream.read(limit + 1)
        require(len(raw) <= limit and stream.read(1) == b"", "selected original extent exceeded")
    require(len(raw) == item.file_size
            and (binascii.crc32(raw) & 0xffffffff) == item.CRC,
            "selected original extent or CRC mismatch")
    return raw

def descriptor(name, raw):
    return {"name": name, "original_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}

def snapshot_projection(name, raw):
    doc = parse(raw)
    rows = doc.get("files")
    require(type(rows) is list and len(rows) == 6183
            and doc.get("subject_commit") == SUBJECT and doc.get("subject_tree") == SUBJECT_TREE,
            "fixed original snapshot subject or count mismatch")
    by_path = {}
    for row in rows:
        require(type(row) is dict and type(row.get("path")) is str
                and row["path"] not in by_path, "snapshot row population mismatch")
        by_path[row["path"]] = row
    require(all(by_path.get(name, {}).get("git_blob") == blob for name, blob in PINS.items()),
            "fixed eight snapshot pins mismatch")
    return {**descriptor(name, raw), "kind": "PROJECTION_NOT_ORIGINAL_SNAPSHOT",
            "tracked_count": len(rows), "subject_commit": SUBJECT, "subject_tree": SUBJECT_TREE,
            "known_original_pins": {name: by_path[name]["git_blob"] for name in PINS},
            "full_snapshot_printed_or_model_reviewed": False}

def census_projection(name, raw):
    doc = parse(raw)
    rows, concerns = doc.get("records"), doc.get("hold_concerns")
    require(type(rows) is list and type(concerns) is list, "census arrays absent")
    first = []
    for concern in concerns[:3]:
        require(type(concern) is dict and type(concern.get("record_index")) is int,
                "census concern index malformed")
        index = concern["record_index"]
        require(0 <= index < len(rows) and type(rows[index]) is dict,
                "census concern row absent")
        row = rows[index]
        paths = row.get("absolute_paths", [])
        require(type(paths) is list and all(type(path) is str for path in paths),
                "census path population malformed")
        first.append({
            "record_index": index, "operation": row.get("operation"),
            "trace_line_number": row.get("trace_line_number"),
            "trace_byte_offset": row.get("trace_byte_offset"),
            "trace_line_sha256": row.get("trace_line_sha256"),
            "native_failure": row.get("native_failure"),
            "unresolved_or_incomplete": row.get("unresolved_or_incomplete"),
            "parse_error_present": row.get("parse_error") is not None,
            "absolute_paths_first_three_prefix512": [path[:512] for path in paths[:3]],
            "paths_are_bounded_projection_not_complete_originals": True,
        })
    return {**descriptor(name, raw), "kind": "BOUNDED_PROJECTION_NOT_ORIGINAL_CENSUS",
            "status": doc.get("status"), "partial_reason": doc.get("partial_reason"),
            "producer_declared_raw_trace_bytes": doc.get("raw_trace_bytes"),
            "producer_declared_raw_trace_sha256": doc.get("raw_trace_sha256"),
            "producer_declared_raw_trace_lines": doc.get("raw_trace_lines"),
            "indexed_trace_prefix_bytes": doc.get("indexed_trace_prefix_bytes"),
            "all_trace_lines_considered_by_original_producer": doc.get("all_trace_lines_considered"),
            "unresolved_write_path_lines": doc.get("unresolved_write_path_lines"),
            "producer_review_index_only": doc.get("review_index_only"),
            "producer_manual_write_review_required": doc.get("manual_write_review_required"),
            "record_count": len(rows), "concern_count": len(concerns),
            "first_three_concern_projection": first, "full_census_or_raw_trace_reviewed": False}

def _identity(value):
    # Reading source legitimately changes atime; it is not source identity.
    return {"device": value.st_dev, "inode": value.st_ino, "mode": value.st_mode,
            "links": value.st_nlink, "bytes": value.st_size,
            "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def _stable_bytes(path, nonalias=True):
    require(path.is_absolute() and path.resolve(strict=True) == path,
            "regular input path alias")
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and (not nonalias or before.st_nlink == 1),
            "regular nonalias input required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as handle:
        opened = os.fstat(handle.fileno())
        require(_identity(before) == _identity(opened), "input changed before read")
        raw = handle.read(opened.st_size + 1)
        require(len(raw) == opened.st_size and handle.read(1) == b""
                and _identity(opened) == _identity(os.fstat(handle.fileno()))
                == _identity(path.lstat()), "input changed during read")
    return raw, {"path": str(path), **_identity(opened),
                 "sha256": hashlib.sha256(raw).hexdigest()}


def _git(*arguments):
    # Only fixed native Git reads; inherited overrides/config hooks cannot supply code.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                       GIT_OPTIONAL_LOCKS="0", GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")
    failure = None
    try:
        result = subprocess.run(["git", "--no-replace-objects", "--literal-pathspecs",
                                 "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
                                 "-C", str(ROOT), *arguments], env=environment,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=30, check=False)
        stdout, stderr, code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr, code = error.stdout or b"", error.stderr or b"", None
        failure = error
    row = {"arguments": list(arguments), "exit_code": code, "timed_out": failure is not None,
           "stdout_complete": failure is None, "stderr_complete": failure is None,
           "stdout_bytes": len(stdout), "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
           "stderr_bytes": len(stderr), "stderr_sha256": hashlib.sha256(stderr).hexdigest()}
    observation["source_git_requests"].append(row)
    number = len(observation["source_git_requests"])
    for name, raw in (("stdout", stdout), ("stderr", stderr)):
        (out / ("source-git-" + str(number) + "-" + name + ".bin")).write_bytes(raw)
    if failure is not None:
        raise failure
    require(len(stdout) <= SOURCE_GIT_LIMIT and len(stderr) <= 64 * 1024,
            "fixed local Git output ceiling")
    require(code == 0, "fixed local Git returned nonzero")
    return stdout


def _git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def _tracked_bytes(relative, expected_blob=None):
    require(relative in (UTILITY_PATH, WORKFLOW_PATH), "unfixed reader source path")
    raw_entry = _git("ls-tree", "-z", "HEAD", "--", relative)
    fields = raw_entry.rstrip(b"\0").split(b"\t")
    require(raw_entry.endswith(b"\0") and raw_entry.count(b"\0") == 1 and len(fields) == 2
            and fields[1] == relative.encode("ascii"), "fixed source entry population")
    metadata = fields[0].split(b" ")
    require(len(metadata) == 3 and metadata[:2] == [b"100644", b"blob"]
            and re.fullmatch(rb"[0-9a-f]{40}", metadata[2]), "fixed source mode/type/blob")
    blob = metadata[2].decode("ascii")
    require(expected_blob is None or blob == expected_blob, "fixed source Git blob mismatch")
    original = _git("cat-file", "blob", blob)
    working, identity = _stable_bytes(ROOT / relative)
    require(working == original and _git_blob(working) == blob and not identity["mode"] & 0o111,
            "fixed100644 source Git/cat-file/working bytes or nonexec mode mismatch")
    return working, {"path": relative, "git_mode": "100644", "git_blob": blob,
                     **identity}


def source_snapshot():
    require(_git("rev-parse", "--show-toplevel").decode("utf-8").rstrip("\n") == str(ROOT),
            "native checked Git root mismatch")
    head = _git("rev-parse", "--verify", "HEAD").decode("ascii").strip()
    tree = _git("rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip()
    require(head == context["GITHUB_SHA"] and re.fullmatch(r"[0-9a-f]{40}", tree),
            "native checked HEAD/tree mismatch")
    require(_git("status", "--porcelain=v1", "-z", "--untracked-files=all",
                 "--ignored=matching") == b"", "changed/untracked/ignored source input")
    entries = _git("ls-tree", "-rz", "--full-tree", "HEAD")
    require(entries.endswith(b"\0"), "native complete Git tree missing terminator")
    rows, seen = [], set()
    for item in entries[:-1].split(b"\0"):
        metadata, separator, name = item.partition(b"\t")
        fields = metadata.split(b" ")
        require(separator and len(fields) == 3 and fields[1] == b"blob"
                and fields[0] in (b"100644", b"100755")
                and re.fullmatch(rb"[0-9a-f]{40}", fields[2]), "unsupported native source entry")
        relative = name.decode("utf-8", errors="strict")
        path = PurePosixPath(relative)
        require(relative and not path.is_absolute() and "\\" not in relative
                and all(part not in ("", ".", "..") for part in relative.split("/"))
                and relative not in seen, "native source path population mismatch")
        seen.add(relative)
        raw, identity = _stable_bytes(ROOT / relative)
        mode, blob = fields[0].decode("ascii"), fields[2].decode("ascii")
        require(_git_blob(raw) == blob
                and bool(identity["mode"] & 0o111) == (mode == "100755"),
                "native working source bytes/mode mismatch")
        rows.append({"path": relative, "git_mode": mode, "git_blob": blob,
                     "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                     "working_identity": {key: value for key, value in identity.items()
                                          if key not in ("path", "sha256")}})
    by_path = {row["path"]: row for row in rows}
    require(all(by_path.get(name, {}).get("git_blob") == pin
                for name, pin in CONTROLLER_PINS.items()), "fixed controller pins mismatch")
    require(UTILITY_PATH in by_path and WORKFLOW_PATH in by_path
            and by_path[UTILITY_PATH]["git_mode"] == "100644"
            and by_path[WORKFLOW_PATH]["git_mode"] == "100644",
            "actual native utility/workflow entry missing")
    source_raw, original = _tracked_bytes(UTILITY_PATH, by_path[UTILITY_PATH]["git_blob"])
    workflow_raw, workflow_source = _tracked_bytes(WORKFLOW_PATH, by_path[WORKFLOW_PATH]["git_blob"])
    copied_path = Path(__file__)
    expected_copy = Path(os.environ["RUNNER_TEMP"]) / (
        "formal-r1-physical-bootstrap-" + observation["case"]) / "formal_r1_physical_metadata_readback.py"
    require(copied_path == expected_copy and copied_path.is_absolute(),
            "utility was not executed from the fixed external copy")
    copied_raw, copied = _stable_bytes(copied_path)
    require(copied_raw == source_raw
            and (copied["device"], copied["inode"]) != (original["device"], original["inode"]),
            "executed copy/native source byte or identity mismatch")
    require(_git("rev-parse", "--verify", "HEAD").decode("ascii").strip() == head
            and _git("rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip() == tree
            and not _git("status", "--porcelain=v1", "-z", "--untracked-files=all",
                         "--ignored=matching"), "native source changed during inert snapshot")
    return {"head": head, "tree": tree, "tracked_count": len(rows), "files": rows,
            "utility": {"git_blob": by_path[UTILITY_PATH]["git_blob"],
                        "source": original, "executed_copy": copied},
            "workflow_git_blob": by_path[WORKFLOW_PATH]["git_blob"],
            "workflow_source": workflow_source,
            "controller_pins": dict(CONTROLLER_PINS)}


def reader_runtime():
    require(sys.version_info[:2] == (3, 11) and sys.flags.isolated == 1
            and sys.flags.no_site == 1 and sys.flags.dont_write_bytecode == 1
            and sys.flags.optimize == 0, "Python3.11 isolated runtime flags mismatch")
    executable = Path(sys.executable).resolve(strict=True)
    raw, identity = _stable_bytes(executable, nonalias=False)
    flags = ("debug", "inspect", "interactive", "optimize", "dont_write_bytecode",
             "no_user_site", "no_site", "ignore_environment", "verbose", "bytes_warning",
             "quiet", "hash_randomization", "isolated", "dev_mode", "utf8_mode", "safe_path")
    images = {key: os.environ.get(key) for key in
              ("ImageOS", "ImageVersion", "RUNNER_OS", "RUNNER_ARCH")}
    require(all(type(value) is str and value for value in images.values()),
            "native runner image metadata absent")
    return {"implementation": platform.python_implementation(), "sys_version": sys.version,
            "version_info": list(sys.version_info), "flags": {key: getattr(sys.flags, key) for key in flags},
            "sys_executable": sys.executable, "resolved_executable": str(executable),
            "executable": identity, "image": images}


def _snapshot_descriptor(name, document):
    raw = encoded(document)
    return {**descriptor(name, raw), "head": document["head"], "tree": document["tree"],
            "tracked_count": document["tracked_count"],
            "utility_git_blob": document["utility"]["git_blob"],
            "utility_source_sha256": document["utility"]["source"]["sha256"],
            "utility_copy_sha256": document["utility"]["executed_copy"]["sha256"],
            "workflow_git_blob": document["workflow_git_blob"],
            "workflow_source_sha256": document["workflow_source"]["sha256"],
            "workflow_source_bytes": document["workflow_source"]["bytes"],
            "both_reader_sources_native100644_nonexec_catfile_verified": True,
            "full_snapshot_model_reviewed": False}



def frame(summary, records):
    summary_raw = encoded(summary)
    require(len(summary_raw) <= SUMMARY_LIMIT, "complete Phase2 summary ceiling")
    pieces, facts = [], []
    for name, raw, fields in records:
        plain = base64.b64encode(raw)
        body = b"".join(plain[index:index + 120] + b"\n"
                        for index in range(0, len(plain), 120))
        expected = 4 * ((len(raw) + 2) // 3)
        expected += (expected + 119) // 120 if expected else 0
        require(len(body) == expected, "whole original base64 extent mismatch")
        metadata = {"name": name, "raw_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "encoded_body_bytes": len(body), "full_raw_record_printed": True,
                    **fields}
        facts.append(encoded(metadata))
        pieces.append((metadata, body))
    if summary.get("capture_complete"):
        key = summary["fixed_group"]
        require(len(pieces) == GROUP_COUNTS[key]
                and sum(len(body) for _, body in pieces) == GROUP_ENCODED_BYTES[key],
                "complete fixed group encoded census mismatch")
    nonce = secrets.token_hex(32)
    collision_material = summary_raw + b"".join(facts) + b"".join(body for _, body in pieces)
    while nonce.encode("ascii") in collision_material:
        nonce = secrets.token_hex(32)
    prefix = ("::stop-commands::" + nonce + "\n").encode("ascii") + b"PHYSICAL_R1_SUMMARY "
    frames, body_bytes, framing = [], len(summary_raw), len(prefix)
    for metadata, body in pieces:
        before = b"PHYSICAL_R1_RECORD " + encoded(metadata)
        after = b"PHYSICAL_R1_RECORD_END " + encoded({"name": metadata["name"]})
        frames.extend((before, body, after))
        framing += len(before) + len(after)
        body_bytes += len(body)
    ending = ("::" + nonce + "::\n").encode("ascii")
    framing += len(ending)
    require(framing <= FRAMING_LIMIT, "all Phase2 framing/nonce ceiling")
    stream = prefix + summary_raw + b"".join(frames) + ending
    require(len(stream) == body_bytes + framing and len(stream) <= CONTROLLER_LIMIT,
            "complete Phase2 controller stdout ceiling")
    return stream


def record_error(stage, error):
    raw = traceback.format_exc().encode("utf-8")
    name = "reader-error-" + str(len(observation["errors"]) + 1) + ".log"
    (out / name).write_bytes(raw)
    row = {**descriptor(name, raw), "stage": stage, "error_type": type(error).__name__,
           "error": str(error), "original_error_retained": True}
    if len(raw) <= 64 * 1024:
        row["original_error_utf8"] = raw.decode("utf-8", errors="strict")
    observation["errors"].append(row)


def main():
    global ROOT, out, observation, context
    require(len(sys.argv) == 1, "fixed Phase2 reader accepts no CLI arguments")
    keys = ("GITHUB_REPOSITORY", "GITHUB_EVENT_NAME", "GITHUB_REF", "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT", "GITHUB_SHA", "GITHUB_JOB")
    context = {key: os.environ.get(key) for key in keys}
    require(context["GITHUB_REPOSITORY"] == REPOSITORY and context["GITHUB_EVENT_NAME"] == "push"
            and context["GITHUB_REF"] == "refs/heads/codex/formal-r1-physical-metadata-readback-20261110"
            and context["GITHUB_JOB"] == "physical-metadata-phase2", "fixed native context mismatch")
    require(all(re.fullmatch(r"[1-9][0-9]{0,19}", context[key] or "")
                for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"))
            and re.fullmatch(r"[0-9a-f]{40}", context["GITHUB_SHA"] or ""), "native ID syntax mismatch")
    fixed_group = os.environ.get("FORMAL_R1_GROUP")
    require(type(fixed_group) is str and fixed_group in GROUP_MEMBERS,
            "native workflow fixed group missing or unknown")
    specifications = GROUP_MEMBERS[fixed_group]
    require(len(specifications) == GROUP_COUNTS[fixed_group]
            and len({name for name, _, _ in specifications}) == len(specifications),
            "closed fixed group member population mismatch")
    ROOT, temporary = Path(os.environ.get("GITHUB_WORKSPACE", "")), Path(os.environ.get("RUNNER_TEMP", ""))
    require(all(path.is_absolute() and path.resolve(strict=True) == path
                and stat.S_ISDIR(path.lstat().st_mode) for path in (ROOT, temporary))
            and ROOT != temporary and ROOT not in temporary.parents and temporary not in ROOT.parents,
            "checked root or genuine external RUNNER_TEMP refused")
    case = context["GITHUB_RUN_ID"] + "-" + context["GITHUB_RUN_ATTEMPT"] + "-" + fixed_group
    out = temporary / ("formal-r1-physical-metadata-" + case)
    out.mkdir(mode=0o700, exist_ok=False)
    require(out.resolve(strict=True) == out and not out.is_symlink(), "output root alias")
    observation = {
        "scope": PHASE_SCOPE, "purpose": PURPOSE, "status": "STARTED",
        "case": case, "fixed_group": fixed_group, "native_context": context,
        "expected_group_count": GROUP_COUNTS[fixed_group],
        "expected_group_raw_bytes": GROUP_RAW_BYTES[fixed_group],
        "expected_group_encoded_bytes": GROUP_ENCODED_BYTES[fixed_group],
        "transport": {"encoding": "RFC4648_BASE64", "columns": 120, "line_ending": "LF",
                      "complete_original_bytes_encoded_not_literal_raw": True},
        "workspace": str(ROOT), "output_root": str(out),
        "historical": {"run": RUN, "attempt": ATTEMPT, "job": JOB, "artifact": ARTIFACT,
                       "controller": CHECKED, "tree": TREE, "ordered_parents": PARENTS,
                       "subject": SUBJECT, "subject_tree": SUBJECT_TREE,
                       "ZIP_bytes": ZIP_SIZE, "ZIP_sha256": ZIP_SHA256,
                       "original_R1_PARTIAL_HOLD_preserved": True},
        "requests": [], "source_git_requests": [], "members": [], "errors": [],
        "capture_complete": False, "source_unchanged": False,
        "no_trace_body_read_or_parse": True, "no_formal_helper_or_Lean_execution": True,
        "large_snapshot_bodies_model_read": False, "old_census_bodies_model_read": False,
        "physical_admission": False, "no_physical_SAFE_classification": True,
        "full_Main_G5_CBZ_or_scientific_credit": False,
        "actual_completed_native_joblog_UTF8_limit": OUTPUT_LIMIT,
        "actual_native_log_limit_requires_independent_postcompletion_check": True,
    }
    (out / "observation.json").write_bytes(encoded(observation))
    before = None
    records = []
    success = False
    try:
        runtime = reader_runtime()
        (out / "runtime.json").write_bytes(encoded(runtime))
        observation["reader_runtime"] = runtime
        before = source_snapshot()
        (out / "source-before.json").write_bytes(encoded(before))
        observation["source_before"] = _snapshot_descriptor("source-before.json", before)
        proof_raw, proof_identity = _stable_bytes(temporary / (
            "formal-r1-physical-bootstrap-" + case) / "bootstrap.json")
        proof = parse(proof_raw)
        require(proof.get("status") == "VERIFIED_COPY_COMPILED"
                and proof.get("context") == context and proof.get("case") == case
                and proof.get("fixed_group") == fixed_group
                and proof.get("head") == before["head"] and proof.get("tree") == before["tree"]
                and proof.get("expected_utility") == {
                    "path": UTILITY_PATH, "git_blob": before["utility"]["git_blob"],
                    "bytes": before["utility"]["source"]["bytes"],
                    "sha256": before["utility"]["source"]["sha256"]},
                "actual bootstrap/source binding mismatch")
        for relative, actual in ((UTILITY_PATH, before["utility"]["source"]),
                                 (WORKFLOW_PATH, before["workflow_source"])):
            declared = proof.get("sources", {}).get(relative, {})
            require(declared.get("git_blob") == actual["git_blob"]
                    and declared.get("git_mode") == "100644"
                    and declared.get("sha256") == actual["sha256"]
                    and declared.get("bytes") == actual["bytes"]
                    and not actual["mode"] & 0o111,
                    "both native reader sources/bootstrap physical binding mismatch")
        observation["bootstrap_proof"] = proof_identity
        responses = [request(index) for index in range(len(REQUESTS))]
        verify_api(*[parse(raw) for raw in responses[:4]])
        raw_zip = responses[4]
        require(len(raw_zip) == ZIP_SIZE and hashlib.sha256(raw_zip).hexdigest() == ZIP_SHA256,
                "fixed original raw ZIP extent or digest mismatch")
        observation.update(raw_zip_verified_by_reader=True, raw_zip_bytes=len(raw_zip),
                           raw_zip_sha256=hashlib.sha256(raw_zip).hexdigest())
        with zipfile.ZipFile(io.BytesIO(raw_zip), "r") as source:
            infos = source.infolist()
            require(0 < len(infos) <= 2048 and len({row.filename for row in infos}) == len(infos),
                    "original ZIP member population mismatch")
            require(sum(row.file_size for row in infos) <= 512 * MIB,
                    "original ZIP total declared member bound exceeded")
            members = {}
            for item in infos:
                name = item.filename
                relative = PurePosixPath(name)
                kind = stat.S_IFMT(item.external_attr >> 16)
                require(item.orig_filename == name and name and "\\" not in name
                        and ":" not in name and not relative.is_absolute()
                        and all(part not in ("", ".", "..") for part in name.rstrip("/").split("/"))
                        and not item.flag_bits & 1
                        and item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        and 0 <= item.file_size <= 128 * MIB
                        and 0 <= item.compress_size <= ZIP_SIZE
                        and 0 <= item.header_offset < ZIP_SIZE
                        and kind in ((0, stat.S_IFDIR) if item.is_dir() else (0, stat.S_IFREG)),
                        "original ZIP member layout or declared bounds refused")
                if not item.is_dir():
                    members[name] = item
            require(len(infos) == EXPECTED_ZIP_ENTRIES, "original observed complete ZIP inventory count mismatch")
            observation["original_zip_member_count"] = len(infos)
            selected = {}
            selected_root = out / "original-physical"
            selected_root.mkdir(mode=0o700)
            for name, expected_bytes, expected_crc in specifications:
                require(name in members, "required fixed original absent: " + name)
                item = members[name]
                require(item.file_size == expected_bytes and item.CRC == expected_crc,
                        "fixed original directory extent/CRC mismatch: " + name)
                raw = member_bytes(source, item, expected_bytes)
                require(len(raw) == expected_bytes,
                        "selected fixed original whole extent mismatch: " + name)
                target = selected_root / name
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    require(stream.write(raw) == len(raw), "short fixed-original retention")
                characters = 4 * ((len(raw) + 2) // 3)
                wrapped_bytes = characters + ((characters + 119) // 120 if characters else 0)
                row = {**descriptor(name, raw), "CRC32_declared": item.CRC,
                       "CRC32_actual_verified": True, "encoded_body_bytes": wrapped_bytes,
                       "status": "FULL_SELECTED_ORIGINAL"}
                observation["members"].append(row)
                selected[name] = row
                records.append((name, raw, {"origin": "ORIGINAL_R1_ARCHIVE_MEMBER",
                                            "CRC32_declared": item.CRC, "CRC32_actual_verified": True}))
            require(len(selected) == GROUP_COUNTS[fixed_group]
                    and sum(row["original_bytes"] for row in selected.values()) == GROUP_RAW_BYTES[fixed_group]
                    and sum(row["encoded_body_bytes"] for row in selected.values()) == GROUP_ENCODED_BYTES[fixed_group],
                    "complete fixed group original census mismatch")
            observation["selected_original_count"] = len(selected)
            observation["selected_original_bytes"] = GROUP_RAW_BYTES[fixed_group]
            observation["selected_encoded_body_bytes"] = GROUP_ENCODED_BYTES[fixed_group]
        success = True
    except Exception as error:
        record_error("SOURCE_CUSTODY_FIXED_PHYSICAL_ORIGINALS", error)
    try:
        after = source_snapshot()
        (out / "source-after.json").write_bytes(encoded(after))
        observation["source_after"] = _snapshot_descriptor("source-after.json", after)
        observation["source_unchanged"] = before is not None and before == after
        require(observation["source_unchanged"], "checked source/copy changed or before absent")
    except Exception as error:
        success = False
        record_error("INERT_SOURCE_AFTER", error)
    observation.update(status="PHASE2_COMPLETE_ORIGINAL_READBACK_NO_ADMISSION" if success else "PHASE2_HOLD",
                       capture_complete=success, declared_native_exit=0 if success else 1)
    (out / "observation.json").write_bytes(encoded(observation))
    try:
        stream = frame(observation, records if success else [])
    except Exception as error:
        record_error("COMPLETE_FRAMING", error)
        success = False
        observation.update(status="PHASE2_VISIBILITY_HOLD", capture_complete=False, declared_native_exit=1)
        full = encoded(observation)
        (out / "observation.json").write_bytes(full)
        failure = {"scope": PHASE_SCOPE, "status": "PHASE2_VISIBILITY_HOLD", "capture_complete": False,
                   "native_context": context, "case": case, "fixed_group": fixed_group,
                   "declared_native_exit": 1,
                   "full_observation_printed": False, "full_observation": descriptor("observation.json", full),
                   "last_error": observation["errors"][-1], "historical": observation["historical"]}
        stream = frame(failure, [])
    (out / "readback-stream.bin").write_bytes(stream)
    require(sys.stdout.buffer.write(stream) == len(stream), "short native stdout write")
    sys.stdout.buffer.flush()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
