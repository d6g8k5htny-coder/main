"""Read exactly eight Site grading originals; never execute or grade their bytes."""
import base64
import binascii
import hashlib
import io
import json
import os
from pathlib import Path
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

REPOSITORY = "d6g8k5htny-coder/main"
BRANCH = "refs/heads/codex/node-status-grader-v2-artifact-readback-20261110"
UTILITY = "tools/cbz/node_status_grader_artifact_readback.py"
WORKFLOW = ".github/workflows/node-status-grader-artifact-readback.yml"
RUN, ATTEMPT, JOB, ARTIFACT = 42867828892, 1, 130010808861, 13787984129
HEAD = "96e85eecc69cfc56b5edb15b12e676133b392051"
CHECKED = "8999da2e0480d6f513e353964fc6320a063e1522"
TREE = "c1a5ee039e641ce9a3125ab7b2747ce9621d6051"
PARENTS = ["e7515c02ccabba29d8a4c310af003c54d0f08785", HEAD]
ZIP_SIZE = 28472
ZIP_SHA256 = "ab9cea79a46e4d533668cedd84cb961dab9b75f923aacc1f29d927d84f10ef80"
MEMBERS = (
    "controls-normal.stdout", "controls-normal.stderr",
    "controls-optimized.stdout", "controls-optimized.stderr",
    "mutations-normal.json", "mutations-normal.stderr",
    "mutations-optimized.json", "mutations-optimized.stderr",
)
NOMINAL_EMPTY = ("mutations-normal.stderr", "mutations-optimized.stderr")
ORIGINAL_JOB_INVENTORY = [[130010808861,"controls","success",[[1,"Set up job","success"],[2,"Run actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1","success"],[3,"Offline source and reader admission controls","success"],[4,"Verify freshness mutation provenance","success"],[5,"Preserve freshness mutation evidence","success"],[10,"Post Run actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1","success"],[11,"Complete job","success"]]],[130010809227,"node-status-refresh","success",[[1,"Set up job","success"],[2,"Run actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1","success"],[3,"Offline freshness tests and mutation controls","success"],[4,"Report-only freshness refresh (read-only GitHub API)","skipped"],[8,"Post Run actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1","success"],[9,"Complete job","success"]]],[130014793982,"build","skipped",[]],[130014795903,"deploy","skipped",[]]]
RAW_LIMIT, SUMMARY_LIMIT, FRAME_LIMIT = 1048576, 32768, 8192
STDOUT_LIMIT, JOBLOG_LIMIT = 1572864, 2097152
REQUESTS = (
    ("original-attempt.json", "repos/d6g8k5htny-coder/main/actions/runs/42867828892/attempts/1", 131072),
    ("original-jobs.json", "repos/d6g8k5htny-coder/main/actions/runs/42867828892/attempts/1/jobs?per_page=100", 131072),
    ("original-artifact.json", "repos/d6g8k5htny-coder/main/actions/artifacts/13787984129", 65536),
    ("original-controller.json", "repos/d6g8k5htny-coder/main/git/commits/8999da2e0480d6f513e353964fc6320a063e1522", 65536),
    ("original-artifact.zip", "repos/d6g8k5htny-coder/main/actions/artifacts/13787984129/zip", ZIP_SIZE),
)
ROOT = None
out = None
observation = None

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

def verify_native(attempt, jobs, artifact, checked):
    require(attempt.get("id") == RUN and attempt.get("run_attempt") == ATTEMPT
            and attempt.get("event") == "pull_request" and attempt.get("head_sha") == HEAD
            and attempt.get("head_commit", {}).get("tree_id") == TREE
            and attempt.get("path") == ".github/workflows/current-site.yml"
            and attempt.get("head_branch") == "codex/node-status-v2-test-first-r0-20261110"
            and attempt.get("repository", {}).get("id") == 1117225690
            and attempt.get("repository", {}).get("full_name") == REPOSITORY
            and attempt.get("head_repository", {}).get("id") == 1117225690
            and attempt.get("status") == "completed" and attempt.get("conclusion") == "success",
            "fixed successful producer attempt binding mismatch")
    prs = attempt.get("pull_requests")
    require(type(prs) is list and len(prs) == 1 and prs[0].get("number") == 816
            and prs[0].get("head", {}).get("sha") == HEAD, "fixed old PR/head mismatch")
    # The run's PR-base field is not substituted for the actual checked merge parents.
    require(jobs.get("total_count") == 4 and type(jobs.get("jobs")) is list
            and len(jobs["jobs"]) == 4, "fixed old job population mismatch")
    require(len({row.get("id") for row in jobs["jobs"]}) == 4, "duplicate original job ID")
    for job_id, name, conclusion, expected_steps in ORIGINAL_JOB_INVENTORY:
        found_jobs = [row for row in jobs["jobs"] if row.get("id") == job_id]
        require(len(found_jobs) == 1, "fixed original job inventory missing row")
        row = found_jobs[0]
        require(row.get("name") == name and row.get("run_id") == RUN
                and row.get("run_attempt") == ATTEMPT and row.get("head_sha") == HEAD
                and row.get("status") == "completed" and row.get("conclusion") == conclusion,
                "fixed original four-job outcome mismatch")
        actual_steps = row.get("steps")
        require(type(actual_steps) is list and len(actual_steps) == len(expected_steps),
                "fixed original step population mismatch")
        for number, step_name, result in expected_steps:
            matches = [step for step in actual_steps if step.get("number") == number]
            require(len(matches) == 1 and matches[0].get("name") == step_name
                    and matches[0].get("status") == "completed"
                    and matches[0].get("conclusion") == result, "fixed original step outcome mismatch")
    rows = [row for row in jobs["jobs"] if row.get("id") == JOB]
    require(len(rows) == 1, "fixed original controls job absent")
    job = rows[0]
    require(job.get("name") == "controls" and job.get("run_id") == RUN
            and job.get("run_attempt") == ATTEMPT and job.get("head_sha") == HEAD
            and job.get("status") == "completed" and job.get("conclusion") == "success",
            "fixed successful controls job binding mismatch")
    for number, name, conclusion in (
        (2, "Run actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", "success"),
        (3, "Offline source and reader admission controls", "success"),
        (4, "Verify freshness mutation provenance", "success"),
        (5, "Preserve freshness mutation evidence", "success"),
    ):
        found = [row for row in job.get("steps", []) if row.get("number") == number]
        require(len(found) == 1 and found[0].get("name") == name
                and found[0].get("status") == "completed"
                and found[0].get("conclusion") == conclusion, "fixed original step binding mismatch")
    native = artifact.get("workflow_run", {})
    require(artifact.get("id") == ARTIFACT
            and artifact.get("name") == "node-status-grader-42867828892-1"
            and artifact.get("size_in_bytes") == ZIP_SIZE
            and artifact.get("digest") == "sha256:" + ZIP_SHA256 and artifact.get("expired") is False
            and native.get("id") == RUN and native.get("head_sha") == HEAD
            and native.get("repository_id") == 1117225690
            and native.get("head_repository_id") == 1117225690
            and native.get("head_branch") == "codex/node-status-v2-test-first-r0-20261110",
            "fixed old artifact binding mismatch")
    require(checked.get("sha") == CHECKED and checked.get("tree", {}).get("sha") == TREE
            and [row.get("sha") for row in checked.get("parents", [])] == PARENTS
            and checked.get("verification", {}).get("verified") is True
            and checked.get("verification", {}).get("reason") == "valid",
            "fixed signed checked-merge binding mismatch")


def identity(value):
    return {"device": value.st_dev, "inode": value.st_ino, "mode": value.st_mode,
            "links": value.st_nlink, "bytes": value.st_size,
            "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def stable(path):
    require(path.is_absolute() and path.resolve(strict=True) == path, "file path alias")
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "regular single-link file required")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        opened = os.fstat(handle.fileno())
        require(identity(before) == identity(opened), "file changed before read")
        raw = handle.read(opened.st_size + 1)
        require(len(raw) == opened.st_size
                and identity(opened) == identity(os.fstat(handle.fileno())) == identity(path.lstat()),
                "file changed during read")
    return raw, {"path": str(path), **identity(opened), "sha256": hashlib.sha256(raw).hexdigest()}


def git(*args):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                       GIT_OPTIONAL_LOCKS="0", GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")
    try:
        done = subprocess.run(["git", "--no-replace-objects", "--literal-pathspecs",
                               "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
                               "-C", str(ROOT), *args], env=environment,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=30, check=False)
    except subprocess.TimeoutExpired as error:
        (out / "git-timeout.stdout").write_bytes(error.stdout or b"")
        (out / "git-timeout.stderr").write_bytes(error.stderr or b"")
        raise
    if done.returncode:
        (out / "git-error.stdout").write_bytes(done.stdout)
        (out / "git-error.stderr").write_bytes(done.stderr)
    require(len(done.stdout) <= 262144 and len(done.stderr) <= 65536
            and done.returncode == 0, "fixed source Git read failed or exceeded bounds")
    return done.stdout


def source_origin(context, case):
    require(git("rev-parse", "--show-toplevel").decode().rstrip("\n") == str(ROOT),
            "reader native Git root mismatch")
    head = git("rev-parse", "--verify", "HEAD").decode("ascii").strip()
    tree = git("rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip()
    require(head == context["GITHUB_SHA"] and re.fullmatch(r"[0-9a-f]{40}", tree)
            and not git("status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching"),
            "reader changed/untracked/ignored native source")
    entries = {}
    for relative in (UTILITY, WORKFLOW):
        raw = git("ls-tree", "-z", "HEAD", "--", relative)
        prefix, separator, name = raw.rstrip(b"\0").partition(b"\t")
        fields = prefix.split(b" ")
        require(raw.endswith(b"\0") and raw.count(b"\0") == 1 and separator
                and name == relative.encode("ascii") and len(fields) == 3
                and fields[:2] == [b"100644", b"blob"]
                and re.fullmatch(rb"[0-9a-f]{40}", fields[2]), "fixed reader source entry mismatch")
        entries[relative] = fields[2].decode("ascii")
    original = git("cat-file", "blob", entries[UTILITY])
    working, source = stable(ROOT / UTILITY)
    workflow_raw, workflow_source = stable(ROOT / WORKFLOW)
    require(not source["mode"] & 0o111 and not workflow_source["mode"] & 0o111,
            "native Git100644 reader sources must be nonexecutable")
    require(git("cat-file", "blob", entries[WORKFLOW]) == workflow_raw
            and hashlib.sha1(b"blob " + str(len(workflow_raw)).encode("ascii") + b"\0"
                             + workflow_raw).hexdigest() == entries[WORKFLOW],
            "workflow native Git/cat-file/physical byte mismatch")
    copied_path = Path(os.environ["RUNNER_TEMP"]) / (
        "node-status-grader-v2-readback-bootstrap-" + case) / "node_status_grader_artifact_readback.py"
    require(Path(__file__) == copied_path, "reader not executing the fixed external copy")
    copied, copy = stable(copied_path)
    require(original == working == copied
            and hashlib.sha1(b"blob " + str(len(working)).encode("ascii") + b"\0" + working).hexdigest()
                == entries[UTILITY]
            and (source["device"], source["inode"]) != (copy["device"], copy["inode"]),
            "reader Git/working/verified-copy mismatch")
    require(git("rev-parse", "--verify", "HEAD").decode().strip() == head
            and git("rev-parse", "--verify", "HEAD^{tree}").decode().strip() == tree,
            "reader source changed during origin check")
    return {"head": head, "tree": tree, "utility_git_blob": entries[UTILITY],
            "workflow_git_blob": entries[WORKFLOW], "source": {"git_mode": "100644", **source},
            "workflow_source": {"git_mode": "100644", "git_blob": entries[WORKFLOW], **workflow_source},
            "executed_copy": copy}


def runtime():
    require(sys.version_info[:2] == (3, 11) and sys.flags.isolated == 1
            and sys.flags.no_site == 1 and sys.flags.dont_write_bytecode == 1
            and sys.flags.optimize == 0, "isolated Python3.11 flags mismatch")
    executable = Path(sys.executable).resolve(strict=True)
    require(stat.S_ISREG(executable.lstat().st_mode), "resolved Python executable not regular")
    raw = executable.read_bytes()
    return {"sys_version": sys.version, "version_info": list(sys.version_info),
            "flags": {key: getattr(sys.flags, key) for key in
                      ("isolated", "no_site", "dont_write_bytecode", "optimize", "ignore_environment")},
            "executable": str(executable), "executable_bytes": len(raw),
            "executable_sha256": hashlib.sha256(raw).hexdigest(),
            "runner": {key: os.environ.get(key) for key in
                       ("ImageOS", "ImageVersion", "RUNNER_OS", "RUNNER_ARCH")}}


def frame(summary, originals):
    summary_raw = encoded(summary)
    require(len(summary_raw) <= SUMMARY_LIMIT, "complete fixed-reader summary ceiling")
    chunks = []
    for name, raw in originals:
        b64 = base64.b64encode(raw)
        body = b"".join(b64[i:i + 120] + b"\n" for i in range(0, len(b64), 120))
        try:
            raw.decode("utf-8", errors="strict")
            utf8 = True
        except UnicodeError:
            utf8 = False
        metadata = {**descriptor(name, raw), "origin": "ORIGINAL_SITE_GRADING_ARCHIVE_MEMBER",
                    "full_original_bytes_encoded": True, "utf8_valid": utf8,
                    "encoding": "RFC4648_STANDARD_PADDED_120_COLUMNS_LF",
                    "encoded_body_bytes": len(body), "encoded_characters": len(b64),
                    "encoded_lines": (len(b64) + 119) // 120 if b64 else 0}
        chunks.append((b"NODE_GRADER_ORIGINAL_BEGIN " + encoded(metadata), body,
                       b"NODE_GRADER_ORIGINAL_END " + encoded({"name": name})))
    nonce = secrets.token_hex(32)
    while nonce.encode("ascii") in summary_raw + b"".join(b"".join(row) for row in chunks):
        nonce = secrets.token_hex(32)
    opening = ("::stop-commands::" + nonce + "\n").encode("ascii") + b"NODE_GRADER_READBACK_SUMMARY "
    closing = ("::" + nonce + "::\n").encode("ascii")
    framing = len(opening) + len(closing) + sum(len(first) + len(last) for first, _, last in chunks)
    require(framing <= FRAME_LIMIT, "complete fixed-reader framing/nonce ceiling")
    stream = opening + summary_raw + b"".join(b"".join(row) for row in chunks) + closing
    require(len(stream) <= STDOUT_LIMIT, "complete fixed-reader stdout ceiling")
    return stream


def main():
    global ROOT, out, observation
    require(len(sys.argv) == 1, "this fixed reader accepts no CLI arguments")
    keys = ("GITHUB_REPOSITORY", "GITHUB_EVENT_NAME", "GITHUB_REF", "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT", "GITHUB_SHA", "GITHUB_JOB")
    context = {key: os.environ.get(key) for key in keys}
    require(context["GITHUB_REPOSITORY"] == REPOSITORY and context["GITHUB_EVENT_NAME"] == "push"
            and context["GITHUB_REF"] == BRANCH and context["GITHUB_JOB"] == "read-old-artifact"
            and re.fullmatch(r"[0-9a-f]{40}", context["GITHUB_SHA"] or "")
            and all(re.fullmatch(r"[1-9][0-9]{0,19}", context[key] or "")
                    for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")), "reader native context mismatch")
    ROOT, temporary = Path(os.environ.get("GITHUB_WORKSPACE", "")), Path(os.environ.get("RUNNER_TEMP", ""))
    require(all(path.is_absolute() and path.resolve(strict=True) == path
                and stat.S_ISDIR(path.lstat().st_mode) for path in (ROOT, temporary))
            and ROOT != temporary and ROOT not in temporary.parents and temporary not in ROOT.parents,
            "reader output is not genuine external RUNNER_TEMP")
    case = context["GITHUB_RUN_ID"] + "-" + context["GITHUB_RUN_ATTEMPT"]
    out = temporary / ("node-status-grader-v2-readback-" + case)
    out.mkdir(mode=0o700, exist_ok=False)
    observation = {"scope": "READ_EXISTING_SITE_GRADING_ORIGINALS_ONLY", "status": "STARTED",
                   "reader_context": context, "case": case, "requests": [], "members": [],
                   "original_run": RUN, "original_attempt": ATTEMPT, "original_job": JOB,
                   "original_artifact": ARTIFACT, "original_head": HEAD,
                   "original_checked": CHECKED, "original_tree": TREE, "original_parents": PARENTS,
                   "original_job_conclusion": "success", "grader_pass_claim": False,
                   "original_grader_or_tests_replayed": False, "original_results_rewritten": False,
                   "expected_members": list(MEMBERS), "capture_complete": False,
                   "nominal_empty_members": list(NOMINAL_EMPTY), "nominal_expectation_discrepancies": [],
                   "producer_python_version": None,
                   "producer_python_version_status": "NOT_OBSERVED_OR_INFERRED_FROM_READER",
                   "producer_command_source_fact": "Site68db invokes system python3",
                   "source_unchanged": False, "G5_CBZ_scientific_credit": False,
                   "actual_complete_native_joblog_limit": JOBLOG_LIMIT,
                   "actual_joblog_limit_checked_after_completion_not_assumed": True}
    originals, before, code = [], None, 1
    try:
        observation["reader_runtime"] = runtime()
        before = source_origin(context, case)
        observation["source_before"] = before
        (out / "source-before.json").write_bytes(encoded(before))
        responses = [request(i) for i in range(5)]
        verify_native(*[parse(raw) for raw in responses[:4]])
        raw_zip = responses[4]
        require(len(raw_zip) == ZIP_SIZE and hashlib.sha256(raw_zip).hexdigest() == ZIP_SHA256,
                "original ZIP byte extent/SHA mismatch")
        observation["original_ZIP_verified"] = descriptor("original-artifact.zip", raw_zip)
        with zipfile.ZipFile(io.BytesIO(raw_zip), "r") as source:
            infos = source.infolist()
            require(len(infos) == 8 and {item.filename for item in infos} == set(MEMBERS)
                    and len({item.header_offset for item in infos}) == 8,
                    "fixed eight-member original ZIP layout mismatch")
            require(sum(item.file_size for item in infos) <= RAW_LIMIT, "whole original eight-file aggregate byte ceiling")
            by_name = {}
            for item in infos:
                require(item.orig_filename == item.filename and item.filename in MEMBERS
                        and not item.is_dir() and stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG)
                        and not item.flag_bits & 1
                        and item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        and 0 <= item.file_size <= RAW_LIMIT and 0 <= item.compress_size <= ZIP_SIZE
                        and 0 <= item.header_offset < ZIP_SIZE, "fixed original ZIP member refused")
                by_name[item.filename] = item
            for name in MEMBERS:
                raw = member_bytes(source, by_name[name], RAW_LIMIT)
                originals.append((name, raw))
                expected_empty, actual_empty = name in NOMINAL_EMPTY, len(raw) == 0
                observation["members"].append({**descriptor(name, raw), "CRC_verified": True,
                    "nominal_expected_empty": expected_empty, "actual_empty": actual_empty,
                    "nominal_expectation_matched": expected_empty == actual_empty})
                if expected_empty != actual_empty:
                    observation["nominal_expectation_discrepancies"].append({
                        "name": name, "nominal_expected_empty": expected_empty,
                        "actual_empty": actual_empty, "actual_bytes": len(raw)})
                # Nominal content expectations never discard or replace original bytes.
        observation["capture_complete"] = len(originals) == 8
        code = 0
    except Exception as error:
        raw_error = traceback.format_exc().encode("utf-8")
        (out / "reader-error.log").write_bytes(raw_error)
        observation.update(status="READER_HOLD", error_type=type(error).__name__, error=str(error),
                           error_record=descriptor("reader-error.log", raw_error))
    try:
        after = source_origin(context, case)
        observation["source_after"] = after
        (out / "source-after.json").write_bytes(encoded(after))
        require(before is not None and before == after, "reader source/copy changed")
        observation["source_unchanged"] = True
    except Exception as error:
        (out / "source-after-error.log").write_bytes(traceback.format_exc().encode("utf-8"))
        observation.update(source_after_error_type=type(error).__name__, source_after_error=str(error),
                           capture_complete=False)
        code = 1
    if code == 0:
        observation["status"] = "EIGHT_COMPLETE_ORIGINALS_CAPTURED_NO_GRADER_CREDIT"
    observation["declared_reader_exit"] = code
    (out / "observation.json").write_bytes(encoded(observation))
    try:
        stream = frame(observation, originals)
    except Exception as error:
        (out / "framing-error.log").write_bytes(traceback.format_exc().encode("utf-8"))
        observation.update(status="FRAMING_HOLD", capture_complete=False, declared_reader_exit=1,
                           framing_error_type=type(error).__name__, framing_error=str(error))
        (out / "observation.json").write_bytes(encoded(observation))
        raise
    (out / "readback-stream.bin").write_bytes(stream)
    require(sys.stdout.buffer.write(stream) == len(stream), "short fixed-reader stdout write")
    sys.stdout.buffer.flush()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
