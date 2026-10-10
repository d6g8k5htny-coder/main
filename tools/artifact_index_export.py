#!/usr/bin/env python3
"""Emit a pinned Artifact Index public projection from stdin only."""
from __future__ import annotations

import os
import sys

# Keep the CLI's own imports read-only even if a caller omits Python's -B flag.
sys.dont_write_bytecode = True
# Resolve only the trusted source location lexically, never a caller path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture.artifact_index import (
    ArtifactIndexRefused, SHA256, STDIN_LIMIT, canonical_json, parse_packet,
    project_artifact_index,
)


FLAGS = {
    "--expected-capture-sha256": "capture_sha256",
    "--expected-metadata-sha256": "metadata_sha256",
    "--expected-policy-sha256": "policy_sha256",
    "--expected-bindings-sha256": "bindings_sha256",
    "--expected-graph-nodes-sha256": "graph_nodes_sha256",
}


def _arguments(arguments):
    if len(arguments) != 2 * len(FLAGS):
        raise ArtifactIndexRefused()
    expected = {}
    for index in range(0, len(arguments), 2):
        flag, value = arguments[index:index + 2]
        name = FLAGS.get(flag)
        if name is None or name in expected or SHA256.fullmatch(value) is None:
            raise ArtifactIndexRefused()
        expected[name] = value
    return expected


def main():
    try:
        expected = _arguments(sys.argv[1:])
        raw = sys.stdin.buffer.read(STDIN_LIMIT + 1)
        packet = parse_packet(raw)
        projection = project_artifact_index(packet, expected)
        output = canonical_json(projection) + b"\n"
    except Exception:
        sys.stderr.buffer.write(b"ARTIFACT_INDEX_REFUSED\n")
        return 1
    sys.stdout.buffer.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
