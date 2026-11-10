"""Fixed, authenticated derived syntax readback of the historical formal R1.
No original methods run. Historical HOLD and all admission decisions stay unchanged.
Only the reviewed bootstrap executes this fixed utility; no CLI selectors exist.
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
TRACE_IDENTITIES = {
    "normal": {"bytes": 124150681, "lines": 278262,
               "sha256": "4c78906cfb1b65d98e0702af43dac084faaae1a8b921fa68d77528c68e470a07"},
    "optimized": {"bytes": 124923553, "lines": 277657,
                  "sha256": "be0e917478f37611af72127ef26dc45db313b8246d091f55d9b6f960d88361f7"},
}
TRACE_ANCHORS = {
    "normal": ((1, 323, 111300, "9a1afa083ccab12a69ac67833683e076f70faf59ed5d66d2ed8c083e796a9818"),
               (2, 350, 117413, "72602f015a1816487de279c60817c2fe3d7348efd47d2306ea09755a3f2e0323"),
               (3, 383, 125406, "b4fb5a5c11aa546fed18e782c60eedc51c6978efef4c6eb9aa6df9985db42d57")),
    "optimized": ((1, 425, 145373, "f1c558008f941569287917d38d48fb3199b241849420095afac36a305e822121"),
                  (2, 452, 151594, "a211eb87d6e33e885200f3f94860bd4b72a26b035cb6a1d0636f7805f0eef3ef"),
                  (3, 485, 159743, "67f87c336e5ec5bc4278be83ab957c52eacd3d0f1a05e8c92537c2cb528376fa")),
}
TRACE_SLICES = {}
REQUESTS = (
    ("original-attempt.json", "repos/d6g8k5htny-coder/main/actions/runs/42803771651/attempts/1", 128 * 1024),
    ("original-jobs.json", "repos/d6g8k5htny-coder/main/actions/runs/42803771651/attempts/1/jobs?per_page=100", 128 * 1024),
    ("original-artifact.json", "repos/d6g8k5htny-coder/main/actions/artifacts/13758500307", 64 * 1024),
    ("original-controller.json", "repos/d6g8k5htny-coder/main/git/commits/b1b8267f6774f4f8c536ee8c54a00bcbcd5271b1", 64 * 1024),
    ("original-artifact.zip", "repos/d6g8k5htny-coder/main/actions/artifacts/13758500307/zip", ZIP_SIZE),
)

# Historical selector constants above are retained literally for source custody.
# They never select this utility's analyzer input or produce excerpt output.
DERIVED_SCOPE = "DERIVED_FORMAL_R1_SYNTAX_ONLY"
UTILITY_PATH = "tools/cbz/formal_r1_derived_readback.py"
WORKFLOW_PATH = ".github/workflows/g5-formal-r1-evidence-readback.yml"
HELPER_PATH = "tools/formal_trace_reanalysis.py"
HELPER_BLOB = "cb56b7015afac4b963257a050a5e77b9b03cebbc"
HELPER_BYTES = 30469
HELPER_SHA256 = "2163be1ebfc94912f8570506eae1cffbc7ffa8cec1182cf9e928f7c44b95bf2e"
CONTROLLER_PINS = {
    ".github/workflows/research-architecture.yml": "69ad3b47537ab8e4f8936affa3e09fccbdcfd1e5",
    "tools/unittest_timing_probe.py": "05fb481b5538c3f517174b00948de45d2849387a",
    "tools/architecture_run_receipt.py": "a6062afa9261958b8d8b85853a9c43031385b2c1",
    "architecture/graph_artifact_readback.py": "c39ffd0fdcde1fd5f895c57df5f770a582918a8a",
    "tools/architecture_check_binding.py": "53ea34e4fffc1087770584744939904844a41cff",
    "tools/unittest_full_qualification.py": "02dc2f3222aaf3e3d4d024659fffff22d6fb5e66",
    "tests/test_unittest_full_qualification.py": "2fe09abce9a8ef9356fd0af85780dcde75179386",
    ".github/workflows/full-unittest-qualification.yml": "ff8226cdeea57ee2ddb41c58c8c93e644343557d",
    "tests/test_formal_trace_reanalysis.py": "e201a20984a4c16c0424f7e374b27a2d4105b008",
    ".github/workflows/formal-trace-parser-contract.yml": "d36aa6906607ea8a64505ee230729fbba4fb4687",
    "tests/fixtures/formal_trace_r1_lines_20261110.json": "e087f661d8494f7fb780e2f32ce04bee1aeb8f3c",
    HELPER_PATH: HELPER_BLOB,
}
PART_BYTES = 1048576
REPORT_LIMIT = 4194304
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
    # This private local Git reader is called only with fixed commands below.
    failure = None
    try:
        result = subprocess.run(["git", "--no-optional-locks", "-C", str(ROOT), *arguments],
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
    require(relative in (UTILITY_PATH, HELPER_PATH), "unfixed executable source path")
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
    require(working == original and _git_blob(working) == blob,
            "fixed source Git/cat-file/working byte mismatch")
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
    copied_path = Path(__file__)
    expected_copy = Path(os.environ["RUNNER_TEMP"]) / (
        "formal-derived-r1-bootstrap-" + observation["case"]) / "formal_r1_derived_readback.py"
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
            "controller_pins": dict(CONTROLLER_PINS)}


def analyzer_runtime():
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
            "full_snapshot_model_reviewed": False}


def analyze_complete_trace(source, members, trace_mode):
    require(trace_mode in ("normal", "optimized"), "unfixed historical trace mode")
    expected = TRACE_IDENTITIES[trace_mode]
    name = trace_mode + "/file-syscalls.log"
    require(name in members, "selected complete historical trace absent")
    raw = member_bytes(source, members[name], 128 * MIB)
    require(len(raw) == expected["bytes"]
            and hashlib.sha256(raw).hexdigest() == expected["sha256"]
            and len(raw.splitlines(keepends=True)) == expected["lines"],
            "selected complete historical trace identity mismatch")
    observation["selected_trace"] = {"mode": trace_mode, "member": name, **expected,
                                     "CRC_and_extent_verified": True}
    helper_raw, helper = _tracked_bytes(HELPER_PATH, HELPER_BLOB)
    require(len(helper_raw) == HELPER_BYTES
            and hashlib.sha256(helper_raw).hexdigest() == HELPER_SHA256,
            "cold analyzer bytes/extent/SHA mismatch")
    observation["analyzer_source"] = helper
    namespace = {"__name__": "_fixed_formal_trace_reanalysis", "__file__": str(ROOT / HELPER_PATH),
                 "__package__": None, "__spec__": None, "__builtins__": __builtins__}
    compiled = compile(helper_raw, str(ROOT / HELPER_PATH), "exec", dont_inherit=True)
    exec(compiled, namespace, namespace)
    require(callable(namespace.get("_analyze_trace")), "fixed private analyzer absent")
    started = time.monotonic()
    report = namespace["_analyze_trace"](raw)
    observation["analysis_seconds_instrumented"] = time.monotonic() - started
    require(type(report) is dict and report.get("scope") == "TRACE_SYNTAX_DIAGNOSTIC_ONLY"
            and report.get("manual_review_required") is True
            and report.get("input") == {"bytes": expected["bytes"],
                                      "sha256": expected["sha256"], "line_count": expected["lines"]}
            and report.get("status") in ("COMPLETE_BOUNDED_DIAGNOSTIC", "HOLD", "PARTIAL_HOLD"),
            "unchanged analyzer result/input contract mismatch")
    serialized = encoded(report)
    require(len(serialized) <= REPORT_LIMIT, "entire derived report plus LF exceeds unchanged ceiling")
    (out / "derived-report.json").write_bytes(serialized)
    return report, serialized


def encode_part(report_bytes, part_index):
    require(type(part_index) is int and part_index in (0, 1, 2, 3),
            "unfixed derived report part")
    length = len(report_bytes)
    require(length <= REPORT_LIMIT, "entire derived report ceiling")
    start, end = min(part_index * PART_BYTES, length), min((part_index + 1) * PART_BYTES, length)
    raw = report_bytes[start:end]
    encoded_raw = base64.b64encode(raw)
    body = b"".join(encoded_raw[position:position + 120] + b"\n"
                    for position in range(0, len(encoded_raw), 120))
    expected_chars = 4 * ((len(raw) + 2) // 3)
    expected_lines = (expected_chars + 119) // 120 if raw else 0
    require(len(encoded_raw) == expected_chars and len(body) == expected_chars + expected_lines
            and b"\r" not in body, "fixed canonical base64 extent mismatch")
    metadata = {"mode": observation["mode"], "part": part_index,
                "status": "DERIVED_SYNTAX_REPORT_PART_BASE64",
                "raw_offset": start, "raw_end": end, "raw_bytes": len(raw),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "whole_report_bytes": length, "whole_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "base64_characters": expected_chars, "encoded_body_bytes": len(body),
                "encoded_line_count": expected_lines, "columns": 120, "LF_only": True,
                "independently_padded_RFC4648": True, "framing_LF_in_raw_hash": False}
    (out / "part-metadata.json").write_bytes(encoded(metadata))
    (out / "part-base64.bin").write_bytes(body)
    return metadata, body


def frame(summary, part_metadata, base64_body):
    summary_bytes = encoded(summary)
    require(len(summary_bytes) <= SUMMARY_LIMIT, "complete derived summary ceiling")
    nonce = secrets.token_hex(32)
    metadata_bytes = encoded(part_metadata)
    while nonce.encode("ascii") in summary_bytes + metadata_bytes + base64_body:
        nonce = secrets.token_hex(32)
    prefix = (("::stop-commands::" + nonce + "\n").encode("ascii")
              + b"FORMAL_R1_DERIVED_SUMMARY ")
    middle = b"FORMAL_R1_DERIVED_PART_BEGIN " + metadata_bytes
    ending = (b"FORMAL_R1_DERIVED_PART_END " + encoded({
        "mode": summary["mode"], "part": summary["part"], "status": part_metadata["status"]})
        + ("::" + nonce + "::\n").encode("ascii"))
    framing_bytes = len(prefix) + len(middle) + len(ending)
    require(framing_bytes <= FRAMING_LIMIT, "all derived framing/nonce ceiling")
    stream = prefix + summary_bytes + middle + base64_body + ending
    require(len(stream) <= CONTROLLER_LIMIT, "complete derived controller stdout ceiling")
    return stream

def _record_error(label, error):
    raw = traceback.format_exc().encode("utf-8")
    name = "reader-error-" + str(len(observation["errors"]) + 1) + ".log"
    (out / name).write_bytes(raw)
    row = {**descriptor(name, raw), "stage": label, "error_type": type(error).__name__,
           "error": str(error), "full_error_text_printed": len(raw) <= 64 * 1024}
    if row["full_error_text_printed"]:
        row["original_error_utf8"] = raw.decode("utf-8", errors="strict")
    observation["errors"].append(row)


def main():
    global ROOT, out, observation, context
    require(len(sys.argv) == 1, "this fixed utility accepts no CLI arguments")
    mode, part_text = os.environ.get("FORMAL_R1_MODE"), os.environ.get("FORMAL_R1_PART")
    require(mode in ("normal", "optimized") and part_text in ("0", "1", "2", "3"),
            "unfixed historical trace/part matrix")
    keys = ("GITHUB_REPOSITORY", "GITHUB_EVENT_NAME", "GITHUB_REF", "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT", "GITHUB_SHA", "GITHUB_JOB")
    context = {key: os.environ.get(key) for key in keys}
    require(context["GITHUB_REPOSITORY"] == REPOSITORY
            and context["GITHUB_EVENT_NAME"] == "push" and context["GITHUB_REF"] == BRANCH
            and context["GITHUB_JOB"] == "derived-formal-r1-report", "native derived context mismatch")
    require(all(re.fullmatch(r"[1-9][0-9]{0,19}", context[name] or "")
                for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"))
            and re.fullmatch(r"[0-9a-f]{40}", context["GITHUB_SHA"] or ""),
            "native derived IDs absent or malformed")
    ROOT = Path(os.environ.get("GITHUB_WORKSPACE", ""))
    temporary = Path(os.environ.get("RUNNER_TEMP", ""))
    require(all(path.is_absolute() and path.resolve(strict=True) == path
                and stat.S_ISDIR(path.lstat().st_mode) for path in (ROOT, temporary))
            and ROOT != temporary and ROOT not in temporary.parents and temporary not in ROOT.parents,
            "native checked root or genuine external RUNNER_TEMP refused")
    case = mode + "-part-" + part_text + "-" + context["GITHUB_RUN_ID"] + "-" + context["GITHUB_RUN_ATTEMPT"]
    out = temporary / ("formal-derived-r1-" + case)
    out.mkdir(mode=0o700, exist_ok=False)
    require(out.resolve(strict=True) == out and not out.is_symlink(), "derived output alias")
    observation = {
        "scope": DERIVED_SCOPE, "purpose": PURPOSE, "status": "STARTED",
        "mode": mode, "part": int(part_text), "case": case, "native_context": context,
        "workspace": str(ROOT), "output_root": str(out),
        "historical": {"run": RUN, "attempt": ATTEMPT, "job": JOB, "artifact": ARTIFACT,
                       "controller": CHECKED, "tree": TREE, "ordered_parents": PARENTS,
                       "subject": SUBJECT, "subject_tree": SUBJECT_TREE,
                       "ZIP_bytes": ZIP_SIZE, "ZIP_sha256": ZIP_SHA256,
                       "original_R1_HOLD_preserved": True},
        "requests": [], "source_git_requests": [], "members": [], "snapshot_summaries": [],
        "raw_zip_verified_by_reader": False, "errors": [], "report": None,
        "capture_complete": False, "analysis_gate": "NOT_OBSERVED", "source_unchanged": False,
        "no_original_methods_executed": True, "full_trace_model_read": False,
        "full_archive_or_snapshot_model_read": False, "admission_or_runtime_feasibility_claim": False,
        "full_Main_G5_CBZ_or_scientific_credit": False,
        "transport": {"raw_part_limit": PART_BYTES, "whole_report_including_LF_limit": REPORT_LIMIT,
                      "summary_limit": SUMMARY_LIMIT, "all_framing_limit": FRAMING_LIMIT,
                      "controller_stdout_limit": CONTROLLER_LIMIT,
                      "native_wrapper_planning_reserve": OUTPUT_LIMIT - CONTROLLER_LIMIT,
                      "actual_native_joblog_limit": OUTPUT_LIMIT,
                      "actual_native_joblog_must_be_checked_after_completion": True},
    }
    (out / "observation.json").write_bytes(encoded(observation))
    before = None
    report = None
    report_bytes = None
    part_metadata = None
    body = b""
    success = False
    native_code = 1
    try:
        runtime = analyzer_runtime()
        (out / "runtime.json").write_bytes(encoded(runtime))
        observation["analyzer_runtime"] = runtime
        before = source_snapshot()
        (out / "source-before.json").write_bytes(encoded(before))
        observation["source_before"] = _snapshot_descriptor("source-before.json", before)
        bootstrap_directory = temporary / ("formal-derived-r1-bootstrap-" + case)
        proof_raw, proof_identity = _stable_bytes(bootstrap_directory / "bootstrap.json")
        proof = parse(proof_raw)
        require(proof.get("status") == "VERIFIED_COPY_COMPILED"
                and proof.get("context") == context and proof.get("case") == case
                and proof.get("head") == before["head"] and proof.get("tree") == before["tree"]
                and proof.get("expected_utility") == {
                    "path": UTILITY_PATH, "git_blob": before["utility"]["git_blob"],
                    "bytes": before["utility"]["source"]["bytes"],
                    "sha256": before["utility"]["source"]["sha256"]},
                "actual verified bootstrap/source binding mismatch")
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
            require("probe.json" in members, "original probe record was not produced")
            probe_context = None
            for name in SMALL:
                item = members.get(name)
                if item is None:
                    observation["members"].append({"name": name, "status": "NOT_PRODUCED"})
                    continue
                raw = member_bytes(source, item, 128 * MIB)
                raw.decode("utf-8", errors="strict")
                if name == "probe.json":
                    doc = parse(raw)
                    probe_context = doc.get("native_controller")
                    require(probe_context == {
                        "GITHUB_EVENT_NAME": "push",
                        "GITHUB_REF": "refs/heads/codex/g5-formal-runtime-feasibility-20261110",
                        "GITHUB_REPOSITORY": REPOSITORY, "GITHUB_RUN_ATTEMPT": "1",
                        "GITHUB_RUN_ID": str(RUN), "GITHUB_SHA": CHECKED,
                    } and doc.get("fixed_original_subject", {}).get("commit") == SUBJECT
                        and doc.get("fixed_original_subject", {}).get("tree") == SUBJECT_TREE
                        and doc.get("fixed_original_subject", {}).get("pins") == PINS
                        and doc.get("methods") == METHODS, "original probe source/run binding mismatch")
                if name in ("normal/observation.json", "optimized/observation.json"):
                    doc = parse(raw)
                    require(doc.get("subject_commit") == SUBJECT and doc.get("subject_tree") == SUBJECT_TREE
                            and doc.get("expected_original_ids") == METHODS
                            and doc.get("native_context") == probe_context,
                            "original mode observation source/run binding mismatch")
                observation["members"].append({**descriptor(name, raw),
                    "status": "RETAINED_IN_VERIFIED_ORIGINAL_ZIP_NOT_PRINTED",
                    "model_member_read_claim": False})
            for name in SNAPSHOTS:
                item = members.get(name)
                require(item is not None, "fixed original subject snapshot absent")
                observation["snapshot_summaries"].append(
                    snapshot_projection(name, member_bytes(source, item, 8 * MIB)))
            report, report_bytes = analyze_complete_trace(source, members, mode)
            require(type(report.get("indexed_prefix_bytes")) is int
                    and 0 <= report["indexed_prefix_bytes"] <= TRACE_IDENTITIES[mode]["bytes"]
                    and type(report.get("all_lines_considered")) is bool
                    and type(report.get("counts")) is dict and type(report.get("events")) is list
                    and len(report["events"]) <= 4096 and type(report.get("concerns")) is list,
                    "bounded derived report coverage/count shape mismatch")
            if report["status"] == "PARTIAL_HOLD":
                require(report["all_lines_considered"] is False
                        and report.get("partial_reason") in ("LINE_LIMIT", "RECORD_LIMIT", "SERIALIZED_LIMIT"),
                        "partial report did not declare an unchanged capacity refusal")
            else:
                require(report["all_lines_considered"] is True and report.get("partial_reason") is None
                        and report["indexed_prefix_bytes"] == TRACE_IDENTITIES[mode]["bytes"],
                        "nonpartial report did not cover the complete selected trace")
            part_metadata, body = encode_part(report_bytes, int(part_text))
            observation["report"] = {**descriptor("derived-report.json", report_bytes),
                "status": report["status"], "indexed_prefix_bytes": report["indexed_prefix_bytes"],
                "all_lines_considered": report["all_lines_considered"],
                "partial_reason": report["partial_reason"], "counts": report["counts"],
                "event_count": len(report["events"]), "concern_count": len(report["concerns"]),
                "manual_review_required": report["manual_review_required"],
                "full_report_model_review_requires_all_four_parts": True}
            observation["part_metadata"] = part_metadata
            observation["original_zip_member_count"] = len(infos)
        success = True
    except Exception as error:
        _record_error("CUSTODY_SOURCE_ANALYSIS_OR_PART", error)
    # Inert source-after collection never imports or evaluates the analyzer.
    try:
        after = source_snapshot()
        (out / "source-after.json").write_bytes(encoded(after))
        observation["source_after"] = _snapshot_descriptor("source-after.json", after)
        observation["source_unchanged"] = before is not None and before == after
        require(observation["source_unchanged"], "actual checked source/copy changed or before snapshot absent")
    except Exception as error:
        success = False
        _record_error("INERT_SOURCE_AFTER", error)
    if success:
        status = report["status"]
        observation["capture_complete"] = True
        observation["status"] = "COMPLETE_DERIVED_REPORT_PART_CAPTURE"
        observation["analysis_gate"] = ("COVERAGE_REFUSED" if status == "PARTIAL_HOLD"
                                        else "MANUAL_REVIEW_HOLD" if status == "HOLD"
                                        else "SYNTAX_ONLY_MANUAL_REVIEW_REQUIRED")
        native_code = 1 if status == "PARTIAL_HOLD" else 0
        observation["declared_native_exit"] = native_code
        observation["expected_native_nonzero_reason"] = "COVERAGE_REFUSED" if native_code else None
    else:
        observation.update(status="DERIVED_READER_HOLD", capture_complete=False,
                           analysis_gate="NOT_QUALIFIED", declared_native_exit=1,
                           expected_native_nonzero_reason=None)
        part_metadata = {"mode": mode, "part": int(part_text), "status": "NO_REPORT_CAPTURE",
                         "report_produced_but_capture_refused": report_bytes is not None,
                         "encoded_body_bytes": 0, "encoded_line_count": 0}
        body = b""
    # All complete metadata/report/source files remain outside the checkout.
    (out / "observation.json").write_bytes(encoded(observation))
    try:
        stream = frame(observation, part_metadata, body)
    except Exception as error:
        _record_error("COMPLETE_FRAMING", error)
        native_code = 1
        observation.update(status="DERIVED_FRAMING_HOLD", capture_complete=False,
                           analysis_gate="NOT_QUALIFIED", declared_native_exit=1,
                           expected_native_nonzero_reason=None)
        full_observation = encoded(observation)
        (out / "observation.json").write_bytes(full_observation)
        # This is an explicit failure manifest, never a replacement derived report.
        failure = {"scope": DERIVED_SCOPE, "mode": mode, "part": int(part_text),
                   "status": "DERIVED_FRAMING_HOLD", "capture_complete": False,
                   "declared_native_exit": 1, "analysis_gate": "NOT_QUALIFIED",
                   "original_complete_observation": descriptor("observation.json", full_observation),
                   "full_observation_printed": False, "last_actual_error": observation["errors"][-1],
                   "native_context": context, "historical": observation["historical"]}
        part_metadata = {"mode": mode, "part": int(part_text), "status": "NO_REPORT_CAPTURE",
                         "encoded_body_bytes": 0, "encoded_line_count": 0}
        stream = frame(failure, part_metadata, b"")
    (out / "readback-stream.bin").write_bytes(stream)
    (out / "readback-stream-descriptor.json").write_bytes(encoded(descriptor("readback-stream.bin", stream)))
    require(sys.stdout.buffer.write(stream) == len(stream), "short derived controller stdout write")
    sys.stdout.buffer.flush()
    return native_code


if __name__ == "__main__":
    raise SystemExit(main())
