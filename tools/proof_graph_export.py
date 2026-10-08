#!/usr/bin/env python3
"""Export source-bound proof graph metadata; execute only in the sandbox lane."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


# The CLI can run from outside the checkout and does not require installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architecture.graph_export import DEFAULT_PROVENANCE_SHA256, export_graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-provenance-sha256", default=DEFAULT_PROVENANCE_SHA256,
        help="trusted caller pin for raw provenance bytes; cannot change the executable gate pin",
    )
    arguments = parser.parse_args(argv)
    try:
        destination = export_graph(
            arguments.source_dir, arguments.output,
            expected_provenance_sha256=arguments.expected_provenance_sha256,
        )
    except (ValueError, OSError, UnicodeError, RecursionError) as exc:
        parser.exit(1, f"graph export refused: {exc}\n")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
