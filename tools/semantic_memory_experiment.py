#!/usr/bin/env python3
"""Task1 deterministic fixture reports only. Execute in the hosted isolated lane."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import types


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("arguments")


def _product():
    # Select the actual adjacent reviewed file, never a CWD/PYTHONPATH package.
    script = Path(__file__).resolve(strict=True)
    root = script.parents[1]
    directory = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for name in ("experiments", "semantic_memory"):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open("benchmark.py", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=directory)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > 131072 or before.st_nlink != 1:
                raise ValueError("product source")
            raw = b""
            while len(raw) <= 131072:
                block = os.read(fd, min(65536, 131073 - len(raw)))
                if not block:
                    break
                raw += block
            after = os.fstat(fd)
            fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
            if len(raw) != before.st_size or any(getattr(before, key) != getattr(after, key) for key in fields):
                raise ValueError("product source")
        finally:
            os.close(fd)
    finally:
        os.close(directory)
    module = types.ModuleType("_semantic_memory_task1")
    module.__file__ = str(root / "experiments/semantic_memory/benchmark.py")
    exec(compile(raw, module.__file__, "exec"), module.__dict__)
    return module


def main(argv=None):
    try:
        parser = _Parser(description=__doc__, add_help=False, allow_abbrev=False)
        parser.add_argument("mode", choices=("prepare-report", "verify-report"))
        parser.add_argument("--fixtures", required=True, action="append")
        parser.add_argument("--report", action="append")
        arguments = parser.parse_args(argv)
        if len(arguments.fixtures) != 1:
            raise ValueError("arguments")
        if arguments.mode == "prepare-report":
            if arguments.report is not None:
                raise ValueError("arguments")
        elif arguments.report is None or len(arguments.report) != 1:
            raise ValueError("arguments")
        product = _product()
        if arguments.mode == "prepare-report":
            raw = product.canonical_report(product.validate_fixtures(arguments.fixtures[0]))
        else:
            raw = product.verify_report(arguments.fixtures[0], arguments.report[0])
        # Direct descriptor writes leave no buffered success data to flush after
        # an output error. A truncated write never returns success.
        view = memoryview(raw)
        while view:
            written = os.write(1, view)
            if written <= 0:
                raise OSError("output")
            view = view[written:]
        return 0
    except Exception:
        try:
            os.write(2, b"semantic memory fixtures refused: invalid input or output\n")
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
