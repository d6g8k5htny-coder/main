#!/usr/bin/env python3
"""Fixed read-only inventory of one retained E artifact; no tests or member execution."""
import binascii
import codecs
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import selectors
import signal
import stat
import subprocess
import sys
import time
import traceback
import zipfile

PURPOSE = "INVENTORY_EXISTING_SDGA_E_ARTIFACT_ONLY"
REPOSITORY = "d6g8k5htny-coder/main"
BRANCH = "refs/heads/codex/cbz-sdga-e-original-inventory-20261110"
RUN, ATTEMPT, JOB, ARTIFACT = 42812650934, 1, 129817709245, 13764261877
HEAD = "21f5f75b2f96ffca01368b3597a977e0ee0364d1"
CHECKED = "3888fd73af76b515dc74f8381121ef1ce060117a"
TREE = "23b78cfe06577d34ae8b0ce9b944f1859d2c9f83"
PARENTS = ["d09f3252e4f9eb91d624f42357e81e3e43a658e5", HEAD]
HEAD_PARENTS = ["d09f3252e4f9eb91d624f42357e81e3e43a658e5",
                "2b8db33557a30759680fce18b9fae39103c7b083"]
ZIP_SIZE = 27670946
ZIP_SHA256 = "d8e875884c165480131c4db6f022318655a1048a582c06c042dbb3985e0656c9"
MIB = 1024 * 1024
OUTPUT_LIMIT = 3 * MIB // 2
INVENTORY_LIMIT = 768 * 1024
SUMMARY_LIMIT = 64 * 1024
REQUESTS = (
    ("original-attempt.json", "repos/d6g8k5htny-coder/main/actions/runs/42812650934/attempts/1", 128 * 1024),
    ("original-jobs.json", "repos/d6g8k5htny-coder/main/actions/runs/42812650934/attempts/1/jobs?per_page=100", 128 * 1024),
    ("original-artifact.json", "repos/d6g8k5htny-coder/main/actions/artifacts/13764261877", 64 * 1024),
    ("original-checked.json", "repos/d6g8k5htny-coder/main/git/commits/3888fd73af76b515dc74f8381121ef1ce060117a", 64 * 1024),
    ("original-head.json", "repos/d6g8k5htny-coder/main/git/commits/21f5f75b2f96ffca01368b3597a977e0ee0364d1", 64 * 1024),
    ("original-artifact.zip", "repos/d6g8k5htny-coder/main/actions/artifacts/13764261877/zip", ZIP_SIZE),
)
STEPS = (
    (5, "Observe original D and E API contracts normally"),
    (6, "Observe original D and E API contracts with optimized Python"),
    (7, "Observe pure dependency contracts normally"),
    (8, "Observe pure dependency contracts with optimized Python"),
    (9, "Observe Python-only object custody normally"),
    (10, "Observe Python-only object custody with optimized Python"),
    (11, "Observe Task D output admission normally"),
    (12, "Observe Task D output admission with optimized Python"),
    (13, "Observe all eight fixed source mutations normally"),
    (14, "Observe all eight fixed source mutations with optimized Python"),
    (15, "Print original Python-only object-custody records"),
    (16, "Print original critical records and native stream identities"),
    (17, "Print original E contracts and mutation evidence"),
    (18, "Retain original source, cases, objects and failures"),
)

def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")

def require(condition, reason):
    if not condition:
        raise ValueError(reason)


require(len(sys.argv) == 1, "fixed inventory reader accepts no selector or argument")
require(os.environ.get("GITHUB_REPOSITORY") == REPOSITORY
        and os.environ.get("GITHUB_EVENT_NAME") == "push"
        and os.environ.get("GITHUB_REF") == BRANCH
        and os.environ.get("GITHUB_RUN_ATTEMPT") == "1"
        and os.environ.get("GITHUB_JOB") == "original-e-inventory",
        "fixed reader native context mismatch")
temporary = Path(os.environ["RUNNER_TEMP"])
workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)
require(temporary.is_absolute() and temporary == temporary.resolve(strict=True)
        and not temporary.is_symlink() and not temporary.is_relative_to(workspace),
        "reader output root is not genuine external RUNNER_TEMP")
out = temporary / "sdga-e-original-inventory-42812650934-1"
out.mkdir(mode=0o700, exist_ok=False)
source_paths = (workspace / "tools/cbz/sdga_task_e_original_inventory_20261110.py",
                workspace / ".github/workflows/sdga-task-e-original-inventory.yml")
source_before = {}
for path in source_paths:
    require(stat.S_ISREG(path.lstat().st_mode) and path.resolve(strict=True) == path,
            "reader source path is not a canonical ordinary file")
    raw = path.read_bytes()
    source_before[str(path.relative_to(workspace))] = {
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob": hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()}
observation = {
    "purpose": PURPOSE, "status": "STARTED", "original_run": RUN,
    "original_attempt": ATTEMPT, "original_job": JOB, "original_artifact": ARTIFACT,
    "original_head": HEAD, "original_checked": CHECKED, "original_tree": TREE,
    "original_ordered_parents": PARENTS, "reader_sources_before": source_before,
    "reader_context": {name: os.environ.get(name) for name in
                       ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA", "GITHUB_REF", "GITHUB_JOB")},
    "no_original_tests_executed": True, "no_original_Lean_or_mutation_replay": True,
    "raw_zip_verified_by_reader": False, "all_member_extent_CRC_SHA_verified": False,
    "complete_inventory_printed": False, "model_original_member_bodies_read": False,
    "semantic_qualification": False, "Green_or_kill_credit": False,
    "scientific_effect": "NONE", "qualification": "BLOCKED",
    "execution_authenticated": False, "requests": [],
}
(out / "observation.json").write_bytes(encoded(observation))

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


def verify_api(attempt, jobs, artifact, checked, head):
    require(attempt.get("id") == RUN and attempt.get("run_attempt") == ATTEMPT
            and attempt.get("head_sha") == HEAD
            and attempt.get("head_commit", {}).get("tree_id") == TREE
            and attempt.get("repository", {}).get("id") == 1117225690
            and attempt.get("repository", {}).get("full_name") == REPOSITORY
            and attempt.get("head_repository", {}).get("id") == 1117225690
            and attempt.get("event") == "pull_request"
            and attempt.get("head_branch") == "codex/cbz-sdga-controller-task-e-20261110"
            and attempt.get("path") == ".github/workflows/sdga-component-contract.yml"
            and attempt.get("status") == "completed" and attempt.get("conclusion") == "success",
            "fixed original E attempt binding mismatch")
    require(jobs.get("total_count") == 1 and type(jobs.get("jobs")) is list
            and len(jobs["jobs"]) == 1, "fixed original E job population mismatch")
    job = jobs["jobs"][0]
    require(job.get("id") == JOB and job.get("run_id") == RUN
            and job.get("run_attempt") == ATTEMPT and job.get("head_sha") == HEAD
            and job.get("name") == "task-a" and job.get("status") == "completed"
            and job.get("conclusion") == "success", "fixed original E job binding mismatch")
    steps = job.get("steps")
    require(type(steps) is list, "original E steps absent")
    for number, name in STEPS:
        rows = [row for row in steps if row.get("number") == number]
        require(len(rows) == 1 and rows[0].get("name") == name
                and rows[0].get("status") == "completed"
                and rows[0].get("conclusion") == "success", "fixed original E step binding mismatch")
    native = artifact.get("workflow_run", {})
    require(artifact.get("id") == ARTIFACT
            and artifact.get("name") == "sdga-task-a-42812650934-1"
            and artifact.get("size_in_bytes") == ZIP_SIZE
            and artifact.get("digest") == "sha256:" + ZIP_SHA256
            and artifact.get("expired") is False
            and native.get("id") == RUN and native.get("repository_id") == 1117225690
            and native.get("head_repository_id") == 1117225690
            and native.get("head_sha") == HEAD, "fixed original E artifact binding mismatch")
    require(checked.get("sha") == CHECKED and checked.get("tree", {}).get("sha") == TREE
            and [row.get("sha") for row in checked.get("parents", [])] == PARENTS
            and checked.get("verification", {}).get("verified") is True
            and checked.get("verification", {}).get("reason") == "valid",
            "fixed original E signed checked commit mismatch")
    require(head.get("sha") == HEAD and head.get("tree", {}).get("sha") == TREE
            and [row.get("sha") for row in head.get("parents", [])] == HEAD_PARENTS
            and head.get("verification", {}).get("verified") is False
            and head.get("verification", {}).get("reason") == "unsigned",
            "fixed original E feature commit mismatch")


def inventory_member(source, item, deadline):
    count = 0
    crc = 0
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    utf8 = True
    with source.open(item, "r") as stream:
        while True:
            require(time.monotonic() < deadline, "fixed archive scan exceeds120 seconds")
            chunk = stream.read(MIB)
            if not chunk:
                break
            count += len(chunk)
            require(count <= item.file_size and count <= 128 * MIB,
                    "original member exceeds declared extent")
            crc = binascii.crc32(chunk, crc)
            digest.update(chunk)
            if utf8:
                try:
                    decoder.decode(chunk, final=False)
                except UnicodeDecodeError:
                    utf8 = False
        if utf8:
            try:
                decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                utf8 = False
    require(count == item.file_size and (crc & 0xffffffff) == item.CRC,
            "original member extent/CRC mismatch")
    return {"name": item.filename, "bytes": count, "sha256": digest.hexdigest(),
            "crc32": format(item.CRC, "08x"), "compressed_bytes": item.compress_size,
            "header_offset": item.header_offset, "compression": item.compress_type,
            "directory": item.is_dir(), "utf8_valid": utf8,
            "body_printed_or_model_reviewed": False}


def descriptor(name, raw):
    return {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


records = []
code = 1
try:
    responses = [request(index) for index in range(len(REQUESTS))]
    for index, raw in enumerate(responses[:5]):
        records.append((REQUESTS[index][0], raw, "ACTUAL_COMPLETE_GITHUB_API_RESPONSE"))
    verify_api(*[parse(raw) for raw in responses[:5]])
    raw_zip = responses[5]
    require(len(raw_zip) == ZIP_SIZE and hashlib.sha256(raw_zip).hexdigest() == ZIP_SHA256,
            "fixed original raw ZIP extent or SHA mismatch")
    observation.update(raw_zip_verified_by_reader=True, raw_zip_bytes=len(raw_zip),
                       raw_zip_sha256=hashlib.sha256(raw_zip).hexdigest())
    deadline = time.monotonic() + 120
    rows = []
    with zipfile.ZipFile(io.BytesIO(raw_zip), "r") as source:
        infos = source.infolist()
        require(0 < len(infos) <= 4096 and len({item.filename for item in infos}) == len(infos),
                "original ZIP member population refused")
        require(sum(item.file_size for item in infos) <= 512 * MIB,
                "original ZIP total declared bytes exceed512MiB")
        for item in infos:
            name = item.filename
            relative = PurePosixPath(name)
            kind = stat.S_IFMT(item.external_attr >> 16)
            require(item.orig_filename == name and name and len(name.encode("utf-8")) <= 1024
                    and "\\" not in name and ":" not in name and not relative.is_absolute()
                    and all(part not in ("", ".", "..") for part in name.rstrip("/").split("/"))
                    and not item.flag_bits & 1
                    and item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    and 0 <= item.file_size <= 128 * MIB
                    and 0 <= item.compress_size <= ZIP_SIZE
                    and 0 <= item.header_offset < ZIP_SIZE
                    and item.header_offset + item.compress_size <= ZIP_SIZE
                    and kind in ((0, stat.S_IFDIR) if item.is_dir() else (0, stat.S_IFREG)),
                    "original ZIP member layout or declared bounds refused")
        for ordinal, item in enumerate(infos):
            observation["current_original_member"] = item.filename
            rows.append({"ordinal": ordinal, **inventory_member(source, item, deadline)})
    inventory = {
        "record": "sdga-e-retained-archive-member-inventory", "schema": 1,
        "purpose": PURPOSE, "original_run": RUN, "original_attempt": ATTEMPT,
        "original_job": JOB, "original_artifact": ARTIFACT,
        "original_head": HEAD, "original_checked": CHECKED, "original_tree": TREE,
        "raw_zip_bytes": ZIP_SIZE, "raw_zip_sha256": ZIP_SHA256,
        "member_count": len(rows), "total_member_bytes": sum(row["bytes"] for row in rows),
        "members_in_original_central_directory_order": rows,
        "all_member_extent_CRC_SHA_verified_by_reader": True,
        "archive_member_bodies_printed_or_model_reviewed": False,
        "no_semantic_qualification": True,
    }
    raw_inventory = encoded(inventory)
    (out / "member-inventory.json").write_bytes(raw_inventory)
    observation.update(
        all_member_extent_CRC_SHA_verified=True, member_count=len(rows),
        total_member_bytes=inventory["total_member_bytes"],
        inventory=descriptor("member-inventory.json", raw_inventory))
    require(len(raw_inventory) <= INVENTORY_LIMIT, "complete inventory exceeds768KiB output admission")
    records.append(("member-inventory.json", raw_inventory, "DERIVED_READER_INVENTORY_NOT_MEMBER_BODY"))
    observation["complete_inventory_printed"] = True
    observation["status"] = "INVENTORY_COMPLETE_ORIGINAL_MEMBER_BODIES_UNREAD"
    code = 0
except Exception as error:
    observation.update(status="READER_HOLD", error_type=type(error).__name__, error=str(error),
                       complete_inventory_printed=False)
    raw_error = traceback.format_exc().encode("utf-8")
    (out / "reader-error.log").write_bytes(raw_error)
    if len(raw_error) <= 64 * 1024:
        records.append(("reader-error.log", raw_error, "ACTUAL_READER_ERROR"))
    else:
        observation["error_visibility"] = {**descriptor("reader-error.log", raw_error),
                                           "full_error_printed": False}
    # Completed prior API responses remain originals; incomplete responses stay
    # explicitly partial native files and are never accepted as metadata.
    known = {name for name, _, _ in records}
    for row in observation["requests"]:
        name = row["name"]
        if name.endswith(".json") and row["status"] == "COMPLETE" and name not in known:
            records.append((name, (out / name).read_bytes(), "ACTUAL_COMPLETE_GITHUB_API_RESPONSE"))
finally:
    source_after = {}
    try:
        for path in source_paths:
            require(stat.S_ISREG(path.lstat().st_mode) and path.resolve(strict=True) == path,
                    "reader source path changed type or alias")
            raw = path.read_bytes()
            source_after[str(path.relative_to(workspace))] = {
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob": hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()}
        require(source_after == source_before, "reader source bytes changed during readback")
    except Exception as error:
        code = 1
        observation.update(status="READER_HOLD_SOURCE", complete_inventory_printed=False,
                           source_error_type=type(error).__name__, source_error=str(error))
    observation["reader_sources_after"] = source_after
    # Transport stderr stays byte-exact in the retained reader files. Print its
    # extent/hash/completeness above, never unknown redirect/error text.
    observation["native_gh_stderr_bodies_printed"] = False
    nonce = secrets.token_hex(32)
    while any(nonce.encode("ascii") in raw for _, raw, _ in records):
        nonce = secrets.token_hex(32)
    require(len(records) <= 16, "fixed reader file-frame population exceeds16")
    frames = []
    framed_bytes = 0
    for name, raw, origin in records:
        try:
            raw.decode("utf-8", errors="strict")
            header = {**descriptor(name, raw), "origin": origin, "full_reader_file_printed": True,
                      "archive_member_body": False,
                      "framing_newline_after": bool(raw and not raw.endswith(b"\n"))}
            frame = b"SDGA_E_INVENTORY_FILE_BEGIN " + encoded(header) + raw
            if raw and not raw.endswith(b"\n"):
                frame += b"\n"
            frame += b"SDGA_E_INVENTORY_FILE_END " + encoded({"name": name})
            require(framed_bytes + len(frame) + SUMMARY_LIMIT + 16 * 1024 <= OUTPUT_LIMIT,
                    "whole file frame would exceed fixed reader output admission")
        except (UnicodeError, ValueError):
            code = 1
            observation.update(status="READER_HOLD_OUTPUT_VISIBILITY", complete_inventory_printed=False)
            frame = b"SDGA_E_INVENTORY_NOT_FULL " + encoded({
                **descriptor(name, raw), "origin": origin, "full_reader_file_printed": False,
                "archive_member_body": False, "retained_unprinted_file": True})
        frames.append(frame)
        framed_bytes += len(frame)
    summary = encoded(observation)
    require(len(summary) <= SUMMARY_LIMIT, "fixed reader summary exceeds64KiB")
    stream = (("::stop-commands::" + nonce + "\n").encode("ascii")
              + b"SDGA_E_INVENTORY_SUMMARY " + summary + b"".join(frames)
              + ("::" + nonce + "::\n").encode("ascii"))
    require(len(stream) <= OUTPUT_LIMIT, "fixed reader-controlled stdout exceeds1.5MiB")
    (out / "readback-stream.bin").write_bytes(stream)
    (out / "observation.json").write_bytes(encoded(observation))
    require(sys.stdout.buffer.write(stream) == len(stream), "short reader stdout write")
    sys.stdout.buffer.flush()
raise SystemExit(code)
