#!/usr/bin/env python3
"""Closed TEST routing CLI. No transport, result printing or scientific authority."""
from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from architecture.routing_broker import (Broker, REFUSAL, REGISTRY_LIMIT,
                                             decode, digest, fields, require)
except Exception:
    sys.stderr.buffer.write(b"ARCHITECTURE_ROUTING_BROKER_REFUSED\n")
    raise SystemExit(1) from None


ACTIONS = {
    "reserve": ("reserve", ("request",)),
    "claim-dispatch": ("claim_dispatch", ("task_id", "request_sha256")),
    "complete": ("complete", ("attempt_id", "outcome", "usage")),
    "settle": ("settle", ("attempt_id", "settlement_id", "usage", "accounting_evidence")),
    "confirm-not-dispatched": ("confirm_not_dispatched", ("task_id", "request_sha256", "reason")),
    "mark-unknown": ("mark_unknown", ("attempt_id", "reason")),
    "read": ("read", ("task_id", "request_sha256")),
}


def arguments(argv):
    require(len(argv) == 6)
    values = {}
    allowed = {"--test-ledger-root", "--test-registry", "--action"}
    for index in range(0, 6, 2):
        option, value = argv[index:index + 2]
        require(option in allowed and option not in values and value and not value.startswith("--"))
        values[option] = value
    require(set(values) == allowed and values["--action"] in ACTIONS)
    return values


def registry_file(path):
    target = Path(path)
    require(target.is_absolute() and ".." not in target.parts)
    for parent in target.parents:
        require(stat.S_ISDIR(parent.lstat().st_mode))
    info = target.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == os.geteuid())
    require(stat.S_IMODE(info.st_mode) == 0o600 and info.st_size <= REGISTRY_LIMIT)
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        opened = os.fstat(fd)
        require((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
                == (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(REGISTRY_LIMIT + 1)
        after = os.fstat(fd)
        require((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns))
        require(len(raw) == opened.st_size)
        # Both byte and canonical object identities are computed locally; no
        # caller-provided hash field serves as authority or file authentication.
        original_sha256 = digest(raw)
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(fd, "rb", closefd=False) as stream:
            observed = stream.read(REGISTRY_LIMIT + 1)
        require(observed == raw and digest(observed) == original_sha256)
        return decode(raw, REGISTRY_LIMIT)
    finally:
        os.close(fd)


def main(argv=None):
    broker = None
    try:
        options = arguments(sys.argv[1:] if argv is None else argv)
        action = options["--action"]
        limit = 512 * 1024 if action == "complete" else 64 * 1024
        packet = decode(sys.stdin.buffer.read(limit + 1), limit)
        method, names = ACTIONS[action]
        fields(packet, names)
        registry = registry_file(options["--test-registry"])
        broker = Broker(options["--test-ledger-root"], registry,
                        clock=lambda: time.time_ns() // 1000000)
        receipt = getattr(broker, method)(*(packet[name] for name in names))
        broker.close()
        broker = None
        sys.stdout.buffer.write(receipt)
        sys.stdout.buffer.flush()
        return 0
    except Exception:
        if broker is not None:
            broker.close()
        sys.stderr.buffer.write((REFUSAL + "\n").encode("ascii"))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
