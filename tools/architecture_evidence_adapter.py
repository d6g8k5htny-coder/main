#!/usr/bin/env python3
"""Read one inert evidence packet on stdin and emit a conservative JSON report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architecture.evidence_adapter import MAX_PACKET_BYTES, adapt_packet, parse_packet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    fields = ("repository", "commit", "run-id", "run-attempt")
    for field in fields:
        parser.add_argument("--expected-" + field, required=True, action="append")
    arguments = parser.parse_args(argv)
    expected = {}
    for field in fields:
        values = getattr(arguments, "expected_" + field.replace("-", "_"))
        if len(values) != 1:
            parser.error("--expected-" + field + " must occur exactly once")
        expected[field] = values[0]
    try:
        raw = sys.stdin.buffer.read(MAX_PACKET_BYTES + 1)
        packet = parse_packet(raw)
        report = adapt_packet(packet, expected)
        output = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    except (ValueError, OSError, UnicodeError, RecursionError, TypeError, OverflowError) as exc:
        print("architecture evidence adapter refused: " + str(exc), file=sys.stderr)
        return 1
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
