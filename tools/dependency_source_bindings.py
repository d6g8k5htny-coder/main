"""Record the pinned source identities behind the dependency atlas's node fingerprints.

Scientific effect: NONE. A digest match establishes byte identity only, never proof
validity, review independence or scientific acceptance.

The public atlas (docs/site/dependencies.html) renders the downstream hard-gate graph
captured in docs/site/dependency-source/GRAPH.json from the Math- commit named in
docs/site/dependency-source/PROVENANCE.json. Some graph nodes carry a fingerprint
that is the SHA-256 of a Math- file (one node carries the file's Git blob id) and a
``source`` path. This tool reads those files from a Math- checkout at the pinned
commit and writes a fixture that tests/test_dependency_source_binding.py compares
with GRAPH.json and with the committed inventory in docs/site/current.json.

Usage:
    python3 tools/dependency_source_bindings.py --math /path/to/Math- \
        [--output tests/fixtures/dependency_source_bindings.json]

The Math- checkout must contain the pinned commit (a shallow clone may not). Only
Git plumbing commands that read objects are run; nothing is written to Math-.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "docs" / "site" / "dependency-source" / "GRAPH.json"
PROVENANCE_PATH = ROOT / "docs" / "site" / "dependency-source" / "PROVENANCE.json"
DEFAULT_OUTPUT = ROOT / "tests" / "fixtures" / "dependency_source_bindings.json"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BLOB_RE = re.compile(r"^[0-9a-f]{40}$")


def fingerprint_kind(fingerprint: object) -> str:
    """Classify a node fingerprint: sha256, blob40, ABSENT, symbolic or no_field."""
    if fingerprint is None:
        return "no_field"
    if not isinstance(fingerprint, str):
        return "symbolic"
    if fingerprint == "ABSENT":
        return "ABSENT"
    if SHA256_RE.match(fingerprint):
        return "sha256"
    if BLOB_RE.match(fingerprint):
        return "blob40"
    return "symbolic"


def is_repository_path(source: object) -> bool:
    """A source that names a path inside the Math- repository (not a URL or label)."""
    return isinstance(source, str) and bool(source) and "://" not in source and ":" not in source


def _git(math_root: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(math_root), *args],
        capture_output=True, timeout=60, check=False)
    if proc.returncode:
        raise SystemExit("REFUSED: git failed: " + proc.stderr.decode(errors="replace").strip())
    return proc.stdout


def build_fixture(math_root: Path, graph: dict, provenance: dict, graph_bytes: bytes) -> dict:
    commit = provenance["commit"]
    _git(math_root, "cat-file", "-e", commit + "^{commit}")
    nodes = graph["nodes"]
    census: dict[str, int] = {}
    file_nodes: dict[str, dict] = {}
    directory_sources: dict[str, dict] = {}
    record_only: list[str] = []
    for node_id in sorted(nodes):
        node = nodes[node_id]
        kind = fingerprint_kind(node.get("fingerprint"))
        census[kind] = census.get(kind, 0) + 1
        source = node.get("source")
        if kind in ("sha256", "blob40"):
            if not is_repository_path(source):
                raise SystemExit("REFUSED: digest node without a repository path: " + node_id)
            blob = _git(math_root, "rev-parse", f"{commit}:{source}").decode().strip()
            data = _git(math_root, "cat-file", "blob", blob)
            file_nodes[node_id] = {
                "source": source,
                "kind": kind,
                "fingerprint": node["fingerprint"],
                "blob": blob,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        elif kind == "symbolic" and source is None:
            record_only.append(node_id)
        if is_repository_path(source) and source.endswith("/"):
            listing = _git(math_root, "ls-tree", "-r", "-z", commit, "--", source).decode()
            files = {}
            for entry in listing.split("\0"):
                if not entry:
                    continue
                meta, _, path = entry.partition("\t")
                _mode, _kind, blob = meta.split(" ", 2)
                files[path] = blob
            tree = _git(math_root, "rev-parse", f"{commit}:{source.rstrip('/')}").decode().strip()
            directory_sources[node_id] = {"source": source, "tree": tree, "files": files}
    return {
        "object": "DEPENDENCY-SOURCE-BINDINGS-v1",
        "scientific_effect": "NONE",
        "meaning": ("Pinned path/blob/bytes/sha256 per digest-fingerprinted atlas node at the PROVENANCE commit, "
                    "plus the files under each directory-bound source. Byte identity only; not proof validity, "
                    "review independence or scientific acceptance."),
        "pinned_commit": commit,
        "graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
        "graph_bytes": len(graph_bytes),
        "node_count": len(nodes),
        "edge_count": len(graph["edges"]),
        "fingerprint_census": dict(sorted(census.items())),
        "file_nodes": file_nodes,
        "directory_sources": directory_sources,
        "record_only_nodes": record_only,
    }


def render(fixture: dict) -> str:
    return json.dumps(fixture, indent=1, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--math", required=True, type=Path, help="Math- checkout containing the pinned commit")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="compare with the existing fixture instead of writing")
    args = parser.parse_args(argv)
    graph_bytes = GRAPH_PATH.read_bytes()
    fixture = build_fixture(args.math, json.loads(graph_bytes), json.loads(PROVENANCE_PATH.read_text(encoding="utf-8")), graph_bytes)
    text = render(fixture)
    if args.check:
        existing = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if existing != text:
            print("DEPENDENCY_SOURCE_BINDINGS_DIFFER: " + str(args.output), file=sys.stderr)
            return 1
        print("DEPENDENCY_SOURCE_BINDINGS_MATCH: " + str(args.output))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(json.dumps({"written": str(args.output), "file_nodes": len(fixture["file_nodes"]),
                      "directory_sources": len(fixture["directory_sources"]),
                      "record_only_nodes": len(fixture["record_only_nodes"]),
                      "pinned_commit": fixture["pinned_commit"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
