#!/usr/bin/env python3
"""GB-6 proposal: fail on unbound graph nodes and unbound artifacts.

Engineering hygiene only; scientific_effect NONE. A passing check means every
declared graph node has a binding to an existing artifact whose bytes match
the pinned sha256, and every in-scope artifact is bound to some node. It does
not claim review, acceptance, custody or any mathematical status.

Inputs (all explicit; no network, no writes):
  --root DIR            repository root that artifact paths are relative to
  --bindings FILE       JSON list of {"node_id", "artifact_path", "sha256"}
  --graph-nodes FILE    JSON list of node IDs (repeatable), and/or
  --site-nodes DIR      derive node IDs from DIR/*.html file stems
  --scope GLOB          in-scope artifact glob relative to root (repeatable)
  --report-only         always exit 0 after printing the report

Exit codes: 0 all bound; 1 unbound/stale items found; 2 malformed input.
The report is deterministic JSON on stdout (sorted keys, one trailing newline).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

NODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:\-]{0,191}")
HEX64 = re.compile(r"[0-9a-f]{64}")
BINDING_KEYS = {"node_id", "artifact_path", "sha256"}
LIMIT = 16 * 1024 * 1024


class InputError(ValueError):
    pass


def _strict_load(path: Path):
    raw = path.read_bytes()
    if len(raw) > LIMIT:
        raise InputError(f"{path.name}: input too large")

    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise InputError(f"{path.name}: duplicate key {key!r}")
            out[key] = value
        return out

    def bad_constant(token):
        raise InputError(f"{path.name}: nonfinite number {token}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=bad_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"{path.name}: invalid JSON ({exc.__class__.__name__})")


def _safe_relpath(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise InputError("artifact_path must be a nonempty POSIX string")
    p = PurePosixPath(value)
    if p.is_absolute() or any(part in ("..", "") for part in value.split("/")):
        raise InputError(f"artifact_path not repo-relative: {value!r}")
    return str(p)


def load_nodes(files, site_dirs):
    nodes = set()
    for f in files:
        data = _strict_load(Path(f))
        if not isinstance(data, list):
            raise InputError(f"{Path(f).name}: graph nodes must be a JSON list")
        for n in data:
            if not isinstance(n, str) or not NODE_ID.fullmatch(n):
                raise InputError(f"invalid node id {n!r}")
            if n in nodes:
                raise InputError(f"duplicate node id {n!r}")
            nodes.add(n)
    for d in site_dirs:
        for html in sorted(Path(d).glob("*.html")):
            n = html.name[:-len(".html")]
            if not NODE_ID.fullmatch(n):
                raise InputError(f"invalid site node id {n!r}")
            nodes.add(n)
    return nodes


def load_bindings(path):
    data = _strict_load(Path(path))
    if not isinstance(data, list):
        raise InputError("bindings must be a JSON list")
    seen = set()
    out = []
    for b in data:
        if not isinstance(b, dict) or set(b) != BINDING_KEYS:
            raise InputError(f"binding must have exactly keys {sorted(BINDING_KEYS)}")
        node, rel, sha = b["node_id"], _safe_relpath(b["artifact_path"]), b["sha256"]
        if not isinstance(node, str) or not NODE_ID.fullmatch(node):
            raise InputError(f"invalid binding node id {node!r}")
        if not isinstance(sha, str) or not HEX64.fullmatch(sha):
            raise InputError(f"sha256 must be lower-case 64 hex for {rel}")
        key = (node, rel)
        if key in seen:
            raise InputError(f"duplicate binding {node} -> {rel}")
        seen.add(key)
        out.append({"node_id": node, "artifact_path": rel, "sha256": sha})
    return out


def check(root: Path, nodes, bindings, scopes):
    root = root.resolve()
    unknown_node, missing, stale, outside = [], [], [], []
    bound_nodes, bound_paths = set(), set()
    for b in sorted(bindings, key=lambda x: (x["node_id"], x["artifact_path"])):
        if b["node_id"] not in nodes:
            unknown_node.append(b)
            continue
        target = (root / b["artifact_path"])
        try:
            resolved = target.resolve()
            resolved.relative_to(root)
        except ValueError:
            outside.append(b)
            continue
        if not resolved.is_file():
            missing.append(b)
            continue
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != b["sha256"]:
            stale.append(dict(b, actual_sha256=actual))
            continue
        bound_nodes.add(b["node_id"])
        bound_paths.add(b["artifact_path"])
    in_scope = set()
    for g in scopes:
        for p in root.glob(g):
            if p.is_file():
                in_scope.add(p.relative_to(root).as_posix())
    report = {
        "schema_version": 1,
        "scientific_effect": "NONE",
        "counts": {
            "graph_nodes": len(nodes),
            "bindings": len(bindings),
            "in_scope_artifacts": len(in_scope),
        },
        "unbound_nodes": sorted(nodes - bound_nodes),
        "unbound_artifacts": sorted(in_scope - bound_paths),
        "bindings_unknown_node": unknown_node,
        "bindings_missing_artifact": missing,
        "bindings_stale_sha": stale,
        "bindings_outside_root": outside,
    }
    failing = any(report[k] for k in (
        "unbound_nodes", "unbound_artifacts", "bindings_unknown_node",
        "bindings_missing_artifact", "bindings_stale_sha", "bindings_outside_root"))
    report["status"] = "UNBOUND_ITEMS" if failing else "ALL_BOUND"
    return report, failing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--bindings", required=True)
    ap.add_argument("--graph-nodes", action="append", default=[])
    ap.add_argument("--site-nodes", action="append", default=[])
    ap.add_argument("--scope", action="append", default=[])
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    try:
        if not args.graph_nodes and not args.site_nodes:
            raise InputError("need --graph-nodes and/or --site-nodes")
        nodes = load_nodes(args.graph_nodes, args.site_nodes)
        bindings = load_bindings(args.bindings)
        report, failing = check(Path(args.root), nodes, bindings, args.scope)
    except (InputError, OSError) as exc:
        print(f"GRAPH_BINDING_INPUT_ERROR: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(report, sort_keys=True, indent=1) + "\n")
    return 0 if (args.report_only or not failing) else 1


if __name__ == "__main__":
    sys.exit(main())
