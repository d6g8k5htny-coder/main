"""Bounded binary capture for the hosted site-graph control lane.

This is output custody, not a sandbox or a hard cumulative-emission guarantee.
It retains bounded original prefixes, counts/hashes observed bytes, and refuses
overflow, timeout or incomplete collection even if the child exits zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import time


FILE_LIMIT = 256 * 1024
AGGREGATE_LIMIT = 512 * 1024
METADATA_LIMIT = 32 * 1024
PACKAGE_LIMIT = 1024 * 1024


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _write_new(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        if stream.write(raw) != len(raw):
            raise OSError("short capture metadata write")
        stream.flush()
        os.fsync(stream.fileno())


def _signal_group(process, number):
    try:
        os.killpg(process.pid, number)
    except ProcessLookupError:
        pass


def run_capture(command, output_dir, *, context=None, wall_seconds=90,
                file_limit=FILE_LIMIT, aggregate_limit=AGGREGATE_LIMIT,
                terminate_seconds=2, drain_seconds=3):
    """Capture one trusted argv on a hosted runner; never replay raw output."""
    for name, value, maximum in (("wall", wall_seconds, 90),
                                 ("terminate", terminate_seconds, 2),
                                 ("drain", drain_seconds, 3)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError("invalid capture " + name + " bound")
    for name, value, maximum in (("file", file_limit, FILE_LIMIT),
                                 ("aggregate", aggregate_limit, AGGREGATE_LIMIT)):
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError("invalid capture " + name + " bound")
    if (type(command) is not list or not command
            or any(type(x) is not str or not x or "\x00" in x for x in command)
            or len(_json_bytes(command)) > 16 * 1024):
        raise ValueError("invalid bounded argv")
    context_raw = _json_bytes({"schema_version": 1, "custody": "runner_claimed_only",
                              "scientific_effect": "NONE", "context": context or {}})
    if len(context_raw) > METADATA_LIMIT:
        raise ValueError("capture context exceeds bound")
    output = Path(output_dir)
    os.mkdir(output, 0o700)
    _write_new(output / "context.json", context_raw)
    files = {}
    rows = {}
    for name in ("stdout", "stderr"):
        fd = os.open(output / (name + ".bin"),
                     os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        files[name] = os.fdopen(fd, "wb")
        rows[name] = {"observed_bytes": 0, "retained_bytes": 0,
                      "observed_hash": hashlib.sha256(), "retained_hash": hashlib.sha256()}
    process = None
    selector = selectors.DefaultSelector()
    start = time.monotonic()
    deadline = start + wall_seconds
    stop_at = None
    drain_deadline = None
    killed = False
    overflow = timed_out = incomplete = False
    controller_error = None
    total = 0
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
        for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        while selector.get_map():
            now = time.monotonic()
            returncode = process.poll()
            if returncode is not None and drain_deadline is None:
                drain_deadline = now + drain_seconds
            if returncode is None and now >= deadline and not timed_out:
                timed_out = True
                stop_at = now if stop_at is None else stop_at
                _signal_group(process, signal.SIGTERM)
                drain_deadline = now + terminate_seconds + drain_seconds
            if stop_at is not None and now >= stop_at + terminate_seconds and not killed:
                _signal_group(process, signal.SIGKILL)
                killed = True
            if drain_deadline is not None and now >= drain_deadline:
                incomplete = bool(selector.get_map())
                _signal_group(process, signal.SIGKILL)
                break
            for key, _mask in selector.select(0.02):
                try:
                    block = os.read(key.fileobj.fileno(), 64 * 1024)
                except BlockingIOError:
                    continue
                if not block:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                row = rows[key.data]
                row["observed_bytes"] += len(block)
                row["observed_hash"].update(block)
                keep = min(len(block), file_limit - row["retained_bytes"],
                           aggregate_limit - total)
                if keep:
                    retained = block[:keep]
                    if files[key.data].write(retained) != keep:
                        raise OSError("short original capture write")
                    row["retained_bytes"] += keep
                    row["retained_hash"].update(retained)
                    total += keep
                if keep < len(block):
                    overflow = True
                    if stop_at is None:
                        stop_at = time.monotonic()
                        _signal_group(process, signal.SIGTERM)
                        drain_deadline = stop_at + terminate_seconds + drain_seconds
        try:
            process.wait(timeout=max(0.001, deadline - time.monotonic()) if stop_at is None
                         else terminate_seconds + drain_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _signal_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=terminate_seconds)
            except subprocess.TimeoutExpired:
                _signal_group(process, signal.SIGKILL)
                process.wait(timeout=drain_seconds)
    except (OSError, subprocess.SubprocessError) as error:
        controller_error = type(error).__name__
        incomplete = True
        if process is not None:
            _signal_group(process, signal.SIGKILL)
            try:
                process.wait(timeout=drain_seconds)
            except subprocess.SubprocessError:
                pass
    finally:
        for key in list(selector.get_map().values()):
            try:
                selector.unregister(key.fileobj)
            finally:
                key.fileobj.close()
        selector.close()
        for stream in files.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
    child_status = None if process is None else process.returncode
    complete = not (overflow or timed_out or incomplete or controller_error)
    if controller_error:
        outcome, status = "CONTROLLER_ERROR", 70
    elif overflow:
        outcome, status = "OUTPUT_OVERFLOW", 74
    elif timed_out:
        outcome, status = "CHILD_TIMEOUT", 124
    elif incomplete or child_status is None:
        outcome, status = "INCOMPLETE_COLLECTION", 75
    elif child_status != 0:
        outcome, status = "CHILD_FAILED", child_status if child_status > 0 else 128 - child_status
    else:
        outcome, status = "COMPLETE_CHILD_SUCCESS", 0
    streams = {}
    for name, row in rows.items():
        streams[name] = {
            "path": name + ".bin", "observed_bytes": row["observed_bytes"],
            "retained_bytes": row["retained_bytes"],
            "observed_sha256": row["observed_hash"].hexdigest(),
            "retained_sha256": row["retained_hash"].hexdigest(),
        }
    result = {
        "schema_version": 1, "scientific_effect": "NONE", "custody": "runner_claimed_only",
        "command": command, "child_returncode": child_status,
        "controller_outcome": outcome, "controller_exit": status,
        "controller_error_type": controller_error, "overflow": overflow,
        "timed_out": timed_out, "incomplete": incomplete,
        "original_streams_complete": complete, "streams": streams,
        "retained_total_bytes": total, "observed_scope": "READ_BYTES_ONLY_NOT_ALL_EMITTED",
        "bounds": {"file": file_limit, "aggregate": aggregate_limit,
                   "metadata_file": METADATA_LIMIT, "package": PACKAGE_LIMIT,
                   "child_seconds": wall_seconds, "terminate_seconds": terminate_seconds,
                   "drain_seconds": drain_seconds},
        "elapsed_seconds": time.monotonic() - start,
        "context_sha256": hashlib.sha256(context_raw).hexdigest(),
    }
    encoded = _json_bytes(result)
    if len(encoded) > METADATA_LIMIT:
        raise ValueError("capture result exceeds bound")
    _write_new(output / "result.json", encoded)
    wanted = {"stdout.bin", "stderr.bin", "context.json", "result.json"}
    if {x.name for x in output.iterdir()} != wanted:
        raise ValueError("capture inventory refused")
    package_bytes = 0
    for path in output.iterdir():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("capture member refused")
        maximum = file_limit if path.name.endswith(".bin") else METADATA_LIMIT
        if info.st_size > maximum:
            raise ValueError("capture file exceeds bound")
        package_bytes += info.st_size
    if package_bytes > PACKAGE_LIMIT:
        raise ValueError("capture package exceeds bound")
    return result


def inert_summary(result):
    return ("SITE_GRAPH_CAPTURE outcome=" + result["controller_outcome"]
            + " child=" + str(result["child_returncode"])
            + " retained=" + str(result["retained_total_bytes"])
            + " complete=" + str(result["original_streams_complete"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("normal", "optimized"), required=True)
    args = parser.parse_args(argv)
    command = [sys.executable, "-B"]
    if args.mode == "optimized":
        command.append("-O")
    command += ["-S", "-m", "unittest", "discover", "-s", "tests",
                "-p", "test_site_graph*.py", "-v"]
    context = {key: os.environ.get(key, "") for key in
               ("GITHUB_SHA", "GITHUB_REPOSITORY", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")}
    context.update({"mode": args.mode, "python": sys.version.split()[0],
                    "identity_scope": "RUNNER_CLAIMED_ONLY"})
    try:
        result = run_capture(command, args.output, context=context)
    except (ValueError, OSError):
        print("SITE_GRAPH_CAPTURE outcome=CONTROLLER_ERROR")
        return 70
    print(inert_summary(result))
    return result["controller_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
