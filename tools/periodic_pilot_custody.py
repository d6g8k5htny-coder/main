"""Bounded custody for the hosted periodic H0 pilot, without scientific promotion.

Receipts contain runner-claimed GitHub identity and observed bytes. They are not
native API attestations, an independent review, or proof of archive readback.
No source, earlier output or custody directory is deleted or overwritten.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import sys
from datetime import datetime, timezone

CUSTODY_FAILURE = 74
LOG_LIMIT = 1024 * 1024
OUTPUT_LIMIT = 16 * 1024 * 1024
SOURCE_LIMIT = 2 * 1024 * 1024
OUTPUT_NAMES = ("observations.json", "RUN.json")
SOURCE_PATHS = (
    ".github/workflows/periodic-h0-pilot.yml",
    "tools/periodic_pilot_custody.py",
    "experiments/periodic_h0/run_pilot.py",
    "experiments/periodic_h0/experiment.py",
    "experiments/periodic_h0/controls.py",
    "experiments/periodic_h0/pilot_config.json",
    "experiments/periodic_h0/requirements.txt",
)


class CustodyError(Exception):
    """A refused custody operation; no automatic retry is appropriate."""


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    except OSError as exc:
        raise CustodyError("exclusive receipt write refused: " + str(path)) from exc


def _same(a, b):
    return (a.st_dev, a.st_ino, a.st_mode, a.st_nlink, a.st_size,
            a.st_mtime_ns, a.st_ctime_ns) == (
            b.st_dev, b.st_ino, b.st_mode, b.st_nlink, b.st_size,
            b.st_mtime_ns, b.st_ctime_ns)


def _read_file(path, limit):
    """Read once without following a final symlink; reject changing/hardlinked files."""
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise CustodyError("nonregular or linked file: " + str(path))
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as source:
            opened = os.fstat(source.fileno())
            if not _same(before, opened):
                raise CustodyError("file changed before read: " + str(path))
            raw = source.read(limit + 1)
            after = os.fstat(source.fileno())
        if not _same(before, after) or not _same(after, path.lstat()):
            raise CustodyError("file changed during read: " + str(path))
        return raw[:limit], len(raw) > limit, before.st_size
    except OSError as exc:
        raise CustodyError("file read refused: " + str(path)) from exc


def _read_json(path):
    raw, overflow, unused_size = _read_file(path, 128 * 1024)
    if overflow:
        raise CustodyError("oversized custody record: " + str(path))
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise CustodyError("invalid custody JSON: " + str(path)) from exc


def _directory(path):
    try:
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or path.resolve() != path.absolute():
            raise CustodyError("directory is not a direct absolute path: " + str(path))
        return info
    except OSError as exc:
        raise CustodyError("directory unavailable: " + str(path)) from exc


def _present(path):
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def context_from_env(env):
    expected = {"GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "github-hosted",
                "GITHUB_REPOSITORY": "d6g8k5htny-coder/main",
                "GITHUB_SERVER_URL": "https://github.com", "GITHUB_JOB": "pilot"}
    if any(env.get(key) != value for key, value in expected.items()):
        raise CustodyError("not the declared GitHub-hosted periodic pilot context")
    for name in ("GITHUB_SHA", "GITHUB_WORKFLOW_SHA"):
        if not re.fullmatch(r"[0-9a-f]{40}", env.get(name, "")):
            raise CustodyError("invalid " + name)
    for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
        if not re.fullmatch(r"[1-9][0-9]{0,19}", env.get(name, "")):
            raise CustodyError("invalid " + name)
    workflow_ref = env.get("GITHUB_WORKFLOW_REF", "")
    prefix = "d6g8k5htny-coder/main/.github/workflows/periodic-h0-pilot.yml@refs/"
    if not workflow_ref.startswith(prefix) or "\n" in workflow_ref or "\r" in workflow_ref:
        raise CustodyError("invalid workflow ref")
    return {
        "repository": expected["GITHUB_REPOSITORY"],
        "server_url": expected["GITHUB_SERVER_URL"], "source_sha": env["GITHUB_SHA"],
        "workflow_sha": env["GITHUB_WORKFLOW_SHA"], "workflow_ref": workflow_ref,
        "run_id": env["GITHUB_RUN_ID"], "run_attempt": env["GITHUB_RUN_ATTEMPT"],
        "job": "pilot", "runner_environment": "github-hosted",
        "attestation": "RUNNER_CLAIMED_IDENTITY_NOT_NATIVE_API_VERIFICATION",
    }


def _root(temp, identity):
    return temp / ("periodic-h0-custody-" + identity["run_id"] + "-" + identity["run_attempt"])


def _sources(checkout):
    _directory(checkout)
    result = {}
    for name in SOURCE_PATHS:
        path = checkout / name
        for parent in path.parents:
            if parent == checkout:
                break
            _directory(parent)
        raw, overflow, size = _read_file(path, SOURCE_LIMIT)
        if overflow:
            raise CustodyError("source exceeds capture bound: " + name)
        result[name] = {"bytes": size, "sha256": hashlib.sha256(raw).hexdigest()}
    return result


def _bind(root, temp, env):
    identity = context_from_env(env)
    _directory(temp)
    if root != _root(temp, identity):
        raise CustodyError("custody directory does not match run/attempt")
    info = _directory(root)
    owner = _read_json(root / "ownership.json")
    if owner["identity"] != identity or owner["directory"] != [info.st_dev, info.st_ino]:
        raise CustodyError("custody ownership/native identity mismatch")
    for name in ("logs", "outputs"):
        _directory(root / name)
    return identity


def initialize(checkout, temp, env):
    identity = context_from_env(env)
    _directory(temp)
    root = _root(temp, identity)
    try:
        root.mkdir(mode=0o700)
    except OSError as exc:
        raise CustodyError("new custody directory required; existing bytes preserved") from exc
    info = root.lstat()
    for name in ("logs", "outputs"):
        (root / name).mkdir(mode=0o700)
    _write_json(root / "ownership.json", {
        "schema": "periodic-h0-custody-owner-v1", "identity": identity,
        "directory": [info.st_dev, info.st_ino], "created_utc": _utc(),
    })
    try:
        source_rows = _sources(checkout)
    except CustodyError as exc:
        _write_json(root / "initialization-error.json", {"error": str(exc), "utc": _utc()})
        raise
    _write_json(root / "sources-before.json", source_rows)
    return root


def capture_command(argv, directory, prefix, limit=LOG_LIMIT, cwd=None):
    """Drain both pipes; bounded raw prefixes plus hashes/counts of the full streams."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", prefix) or not 0 < limit <= LOG_LIMIT:
        raise CustodyError("invalid bounded capture request")
    _directory(directory)
    streams = {}
    started = _utc()
    with contextlib.ExitStack() as stack:
        files = {}
        for name in ("stdout", "stderr"):
            path = directory / (prefix + "." + name + ".bin")
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except OSError as exc:
                raise CustodyError("new raw stream file required") from exc
            files[name] = stack.enter_context(os.fdopen(fd, "wb"))
            streams[name] = {"bytes": 0, "retained_bytes": 0}
        hashes = {name: hashlib.sha256() for name in streams}
        retained_hashes = {name: hashlib.sha256() for name in streams}
        process = None
        spawn_error = None
        try:
            process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with selectors.DefaultSelector() as selector:
                for name in streams:
                    pipe = getattr(process, name)
                    stack.callback(pipe.close)
                    selector.register(pipe, selectors.EVENT_READ, name)
                while selector.get_map():
                    for key, unused_events in selector.select():
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        row = streams[key.data]
                        row["bytes"] += len(chunk)
                        hashes[key.data].update(chunk)
                        keep = chunk[:max(0, limit - row["retained_bytes"])]
                        files[key.data].write(keep)
                        retained_hashes[key.data].update(keep)
                        row["retained_bytes"] += len(keep)
            returncode = process.wait()
        except OSError as exc:
            if process is not None:
                process.kill()
                process.wait()
                raise CustodyError("capture I/O failed; partial raw files preserved") from exc
            returncode, spawn_error = 127, type(exc).__name__ + ": " + str(exc)
        except BaseException:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            raise
        for name, row in streams.items():
            files[name].flush()
            os.fsync(files[name].fileno())
            row.update(sha256=hashes[name].hexdigest(),
                       retained_sha256=retained_hashes[name].hexdigest(),
                       truncated=row["bytes"] > row["retained_bytes"])
    return {"phase": prefix, "argv": argv, "started_utc": started, "ended_utc": _utc(),
            "returncode": returncode, "spawn_error": spawn_error, "streams": streams,
            "overflow": any(row["truncated"] for row in streams.values())}


def _shell_exit(returncode):
    return returncode if returncode >= 0 else 128 - returncode


def run_pilot(root, checkout, temp, env):
    _bind(root, temp, env)
    if _present(root / "execution-start.json") or _present(root / "receipt.json"):
        raise CustodyError("execution already attempted or sealed; no retry")
    new_output_absent = not _present(temp / "periodic-h0-replay")
    _write_json(root / "execution-start.json", {"utc": _utc(),
                                               "new_output_absent": new_output_absent})
    result = {"commands": [], "verify": "NOT_REACHED", "precondition_error": None}
    try:
        before = _read_json(root / "sources-before.json")
        try:
            current = _sources(checkout)
        except CustodyError as exc:
            result["precondition_error"] = "source_read_error"
            result["error"] = str(exc)
            current = None
        if current is not None and current != before:
            result["precondition_error"] = "source_drift"
        if result["precondition_error"] is None and not new_output_absent:
            result["precondition_error"] = "preexisting_output"
        if result["precondition_error"] is not None:
            result["source_observation"] = current
            result["exit"] = CUSTODY_FAILURE
        else:
            script = "experiments/periodic_h0/run_pilot.py"
            output = temp / "periodic-h0-replay"
            pilot = capture_command([sys.executable, "-B", script, "--output", str(output)],
                                    root / "logs", "pilot", cwd=checkout)
            result["commands"].append(pilot)
            if pilot["returncode"] != 0:
                result["verify"] = "SKIPPED_PILOT_FAILURE"
                result["exit"] = _shell_exit(pilot["returncode"])
            else:
                verify = capture_command([sys.executable, "-B", script, "--verify",
                                          str(output / "observations.json")],
                                         root / "logs", "verify", cwd=checkout)
                result["commands"].append(verify)
                result["verify"] = "COMPLETED"
                result["exit"] = _shell_exit(verify["returncode"])
                if result["exit"] == 0 and any(x["overflow"] for x in result["commands"]):
                    result["exit"] = CUSTODY_FAILURE
    except CustodyError as exc:
        result["error"] = str(exc)
        result["exit"] = CUSTODY_FAILURE
    _write_json(root / "execution.json", result)
    return result["exit"]


def _snapshot_outputs(root, output):
    rows = {}
    unexpected = []
    holds = []
    if not _present(output):
        return {name: {"state": "missing"} for name in OUTPUT_NAMES}, unexpected, holds
    try:
        _directory(output)
    except CustodyError:
        return {name: {"state": "unsafe"} for name in OUTPUT_NAMES}, [], ["unsafe_output_directory"]
    try:
        with os.scandir(output) as entries:
            names = []
            for index, entry in enumerate(entries):
                if index == 64:
                    holds.append("output_directory_inventory_overflow")
                    break
                names.append(entry.name)
    except OSError as exc:
        raise CustodyError("output directory inventory failed: " + str(output)) from exc
    unexpected = sorted(set(names) - set(OUTPUT_NAMES))
    if unexpected:
        holds.append("unexpected_output_members")
    for name in OUTPUT_NAMES:
        source = output / name
        if not _present(source):
            rows[name] = {"state": "missing"}
            continue
        try:
            raw, overflow, size = _read_file(source, OUTPUT_LIMIT)
            target = root / "outputs" / name
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as destination:
                destination.write(raw)
                destination.flush()
                os.fsync(destination.fileno())
            rows[name] = {"state": "oversized" if overflow else "captured",
                          "observed_bytes": size, "retained_bytes": len(raw),
                          "sha256": None if overflow else hashlib.sha256(raw).hexdigest(),
                          "retained_sha256": hashlib.sha256(raw).hexdigest()}
            if overflow:
                holds.append("output_overflow:" + name)
        except (CustodyError, OSError) as exc:
            rows[name] = {"state": "unsafe", "error": str(exc)}
            holds.append("unsafe_output:" + name)
    return rows, unexpected, holds



def _stream_inventory(root, execution):
    """Observe known raw files once and admit only matched bounded bytes to upload."""
    expected = {}
    if execution:
        for command in execution["commands"]:
            for stream_name, stream_info in command["streams"].items():
                expected[command["phase"] + "." + stream_name + ".bin"] = stream_info
    inventory, problems = {}, []
    for phase in ("pilot", "verify"):
        for channel in ("stdout", "stderr"):
            name = phase + "." + channel + ".bin"
            path = root / "logs" / name
            if not _present(path):
                inventory[name] = {"state": "missing" if name in expected else "not_created"}
                if name in expected:
                    problems.append("missing_stream:" + name)
                continue
            try:
                raw, overflow, observed_size = _read_file(path, LOG_LIMIT)
                digest = hashlib.sha256(raw).hexdigest()
                prior = expected.get(name)
                if overflow or (prior and
                        (prior["retained_bytes"] != observed_size or
                         prior["retained_sha256"] != digest)):
                    inventory[name] = {"state": "rejected"}
                    problems.append("changed_stream:" + name)
                    continue
                inventory[name] = {
                    "state": "captured" if prior else "interrupted_prefix",
                    "retained_bytes": len(raw), "retained_sha256": digest,
                    "full_stream": prior if prior else None,
                    "full_stream_identity": "OBSERVED" if prior else "UNKNOWN_INTERRUPTED",
                }
                if prior is None:
                    problems.append("interrupted_stream:" + name)
            except CustodyError:
                inventory[name] = {"state": "unsafe"}
                problems.append("unsafe_stream:" + name)
    return inventory, problems


def _prepare_artifact(root, receipt):
    """Publish a fresh allowlisted stage, never a recursive walk of raw workspace."""
    stage = root / "upload"
    try:
        stage.mkdir(mode=0o700)
    except OSError as exc:
        raise CustodyError("artifact stage already exists or cannot be created") from exc
    choices = []
    metadata = ("ownership.json", "sources-before.json", "initialization-error.json",
                "execution-start.json", "execution.json", "seal-start.json", "receipt.json")
    for name in metadata:
        if _present(root / name):
            choices.append((name, 128 * 1024, None))
    for name, item in receipt["raw_streams"].items():
        if item["state"] in ("captured", "interrupted_prefix"):
            choices.append(("logs/" + name, LOG_LIMIT, item))
    for name, item in receipt["outputs"].items():
        if item["state"] in ("captured", "oversized"):
            choices.append(("outputs/" + name, OUTPUT_LIMIT, item))
    members = []
    for relative, limit, expected in choices:
        raw, overflow, size = _read_file(root / relative, limit)
        digest = hashlib.sha256(raw).hexdigest()
        if overflow or (expected is not None and
                        (expected["retained_bytes"] != size or
                         expected["retained_sha256"] != digest)):
            raise CustodyError("upload-stage readback refused; safe partial stage preserved")
        target = stage / relative
        if target.parent != stage and not _present(target.parent):
            try:
                target.parent.mkdir(mode=0o700)  # exactly one known logs/outputs component
            except OSError as exc:
                raise CustodyError("artifact stage subdirectory cannot be created: "
                                   + str(target.parent)) from exc
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as destination:
            destination.write(raw)
            destination.flush()
            os.fsync(destination.fileno())
        members.append({"path": relative, "bytes": len(raw), "sha256": digest})
    _write_json(stage / "artifact-manifest.json", {
        "schema": "periodic-h0-prepared-artifact-v1", "identity": receipt["identity"],
        "members": members,
        "complete_member_names": sorted([item["path"] for item in members] +
                                        ["artifact-manifest.json"]),
        "manifest_self_exclusion": "artifact-manifest.json",
        "qualification": "Prepared local runner bytes only; upload and archive readback separate.",
    })


def seal(root, checkout, temp, env, job_status, pilot_outcome):
    identity = _bind(root, temp, env)
    if _present(root / "receipt.json") or _present(root / "seal-start.json"):
        raise CustodyError("sealing already attempted; no retry or replacement")
    _write_json(root / "seal-start.json", {"utc": _utc()})
    execution = _read_json(root / "execution.json") if _present(root / "execution.json") else None
    before = _read_json(root / "sources-before.json") if _present(root / "sources-before.json") else None
    precondition = execution.get("precondition_error") if execution else None
    holds, after = [], None
    if before is None:
        holds.append("source_initialization_incomplete")
    elif precondition in ("source_read_error", "source_drift"):
        # Reuse the first refusal observation; do not acquire that source again.
        after = execution.get("source_observation")
        holds.append(precondition)
    else:
        try:
            after = _sources(checkout)
            if after != before:
                holds.append("source_drift")
        except CustodyError as exc:
            holds.append("source_after_read_error:" + str(exc))
    raw_streams, stream_problems = _stream_inventory(root, execution)
    holds.extend(stream_problems)
    journal = _read_json(root / "execution-start.json") if _present(root / "execution-start.json") else None
    claimed_output = bool(journal and journal.get("new_output_absent") is True and not precondition)
    if claimed_output:
        outputs, unexpected, output_problems = _snapshot_outputs(root, temp / "periodic-h0-replay")
        holds.extend(output_problems)
    else:
        outputs = {name: {"state": "not_owned_not_read"} for name in OUTPUT_NAMES}
        unexpected = []
    if execution is None:
        holds.append("execution_not_complete")
    else:
        if precondition:
            holds.append(precondition)
        if any(command["overflow"] for command in execution["commands"]):
            holds.append("stream_overflow")
        if execution.get("error") and not precondition:
            holds.append("capture_or_execution_error")
    if not claimed_output:
        completeness = "not_owned"
    elif all(item["state"] == "captured" for item in outputs.values()):
        completeness = "complete"
    elif all(item["state"] == "missing" for item in outputs.values()):
        completeness = "absent"
    else:
        completeness = "partial"
    if execution and execution["exit"] == 0 and completeness != "complete":
        holds.append("successful_execution_missing_complete_outputs")
    pilot_success = bool(execution and execution["exit"] == 0 and
                         len(execution["commands"]) == 2 and not holds and
                         completeness == "complete" and job_status == "success" and
                         pilot_outcome == "success")
    receipt = {
        "schema": "periodic-h0-hosted-custody-v1", "sealed_utc": _utc(),
        "identity": identity, "source_before": before, "source_after": after,
        "execution": execution if execution else "NOT_STARTED_OR_INTERRUPTED",
        "job_status_before_seal": job_status, "workflow_pilot_outcome": pilot_outcome,
        "raw_streams": raw_streams, "outputs": outputs, "unexpected_members": unexpected,
        "output_completeness": completeness, "pilot_success": pilot_success,
        "holds": sorted(set(holds)), "custody_exit": CUSTODY_FAILURE if holds else 0,
        "limits": {"raw_prefix_bytes_per_stream": LOG_LIMIT,
                   "raw_bytes_per_output_member": OUTPUT_LIMIT, "directory_entries": 64},
        "scientific_effect": "NONE",
        "source_identity": "Runner observations plus claimed GitHub context; not native attestation.",
        "scope": "Raw bounded custody; no archive readback or continuum/lifetime/theorem verdict.",
        "cleanup": "No deletion; retain original output and custody until hosted ephemeral teardown.",
        "interruptions": "Cancellation, timeout, runner/storage loss can prevent sealing/upload.",
        "partial_observations": "No partial realization arrays inferred from a pre-write failure.",
    }
    _write_json(root / "receipt.json", receipt)
    _prepare_artifact(root, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("init", "run", "seal"))
    parser.add_argument("--job-status", choices=("success", "failure", "cancelled"), default="failure")
    parser.add_argument("--pilot-outcome", choices=("success", "failure", "cancelled", "skipped"), default="skipped")
    args = parser.parse_args()
    try:
        checkout = Path(os.environ["GITHUB_WORKSPACE"])
        temp = Path(os.environ["RUNNER_TEMP"])
        identity = context_from_env(os.environ)
        root = _root(temp, identity)
        if args.operation == "init":
            initialize(checkout, temp, os.environ)
            print(json.dumps({"initialized": str(root), "identity": identity}))
            return 0
        if args.operation == "run":
            return run_pilot(root, checkout, temp, os.environ)
        receipt = seal(root, checkout, temp, os.environ, args.job_status, args.pilot_outcome)
        print(json.dumps({"receipt": str(root / "receipt.json"),
                          "pilot_success": receipt["pilot_success"], "holds": receipt["holds"]}))
        return receipt["custody_exit"]
    except (CustodyError, KeyError) as exc:
        print("periodic pilot custody refused: " + str(exc), file=sys.stderr)
        return CUSTODY_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
