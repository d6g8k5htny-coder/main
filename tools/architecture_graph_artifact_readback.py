#!/usr/bin/env python3
"""Read one bounded graph artifact; emit one validated stdout-only unit."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from architecture.graph_artifact_readback import (  # noqa: E402
    RAW_LIMIT, _encoded, read_graph_archive, validate_expected,
)

REFUSAL = "ARCHITECTURE_GRAPH_READBACK_REFUSED\n"
SOURCE_LIMIT = 16 * 1024 * 1024


class OpaqueParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(REFUSAL)

    def exit(self, status=0, message=None):
        raise ValueError(REFUSAL)


class Once(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest) is not None:
            parser.error("duplicate option")
        setattr(namespace, self.dest, values)


def _metadata(info) -> tuple:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_regular(value: str | Path, limit: int) -> bytes:
    path = Path(value)
    path = path if path.is_absolute() else Path.cwd() / path
    if ".." in path.parts or not path.name:
        raise ValueError(REFUSAL)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:-1]:
            following = os.open(component, flags, dir_fd=directory)
            os.close(directory)
            directory = following
        leaf = os.open(path.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                       dir_fd=directory)
        with os.fdopen(leaf, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= limit:
                raise ValueError(REFUSAL)
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
            if len(raw) > limit or len(raw) != before.st_size or _metadata(before) != _metadata(after):
                raise ValueError(REFUSAL)
            return raw
    finally:
        os.close(directory)


def main(argv: list[str] | None = None) -> int:
    parser = OpaqueParser(allow_abbrev=False, add_help=False)
    parser.add_argument("--archive", required=True, action=Once)
    aliases = {
        "checked_commit": "commit", "repository": "repository", "run_id": "run-id",
        "run_attempt": "run-attempt", "receipt_sha256": "receipt-sha256",
        "artifact_id": "artifact-id", "artifact_sha256": "artifact-sha256",
        "artifact_name": "artifact-name",
    }
    for field, flag in aliases.items():
        parser.add_argument("--expected-" + flag, dest=field, required=True, action=Once)
    try:
        arguments = parser.parse_args(argv)
        expected = validate_expected({field: getattr(arguments, field) for field in aliases})
        archive = _read_regular(arguments.archive, RAW_LIMIT)
        source = _read_regular(ROOT / "docs/site/dependency-source/GRAPH.json", SOURCE_LIMIT)
        unit = read_graph_archive(archive, expected, source)
        encoded = _encoded(unit)
    except (ValueError, OSError, UnicodeError, RecursionError, OverflowError):
        sys.stderr.write(REFUSAL)
        return 1
    try:
        written = sys.stdout.buffer.write(encoded)
        if written != len(encoded):
            raise OSError(REFUSAL)
        sys.stdout.buffer.flush()
    except OSError:
        sys.stderr.write(REFUSAL)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
