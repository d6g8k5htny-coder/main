#!/usr/bin/env python3
"""Pinned, nonauthoritative graph 1-skeleton topology / snapshot-change diagnostic.

Only exact graph-theoretic invariants of a labelled undirected multigraph are
reported. Directions, edge relations, evidence status and review roles are NOT
interpreted as theorem validity. In particular, beta1 is NOT a proof-gap count.
This is opt-in engineering instrumentation; it does not execute scientific code,
fetch the network, change the graph, or alter required gates.

Example:
  python -B -S tools/evidence_topology_diagnostic.py \
    --graph GRAPH.json --sha256 <exact-raw-source-digest>
  python -B -S tools/evidence_topology_diagnostic.py \
    --graph old.json --sha256 <old-digest> \
    --compare new.json --compare-sha256 <new-digest>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path

MAX_INPUT = 4 * 1024 * 1024
MAX_NODES = 4096
MAX_EDGES = 16384
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
EDGE_FIELDS = frozenset(("from", "to", "required", "relation"))
REFUSAL = "EVIDENCE_TOPOLOGY_REFUSED\n"


class Refused(ValueError):
    pass


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Refused("duplicate JSON key")
        result[key] = value
    return result


def _constant(_value):
    raise Refused("nonfinite JSON")


def _float(text):
    value = float(text)
    if not math.isfinite(value):
        raise Refused("nonfinite JSON")
    return value


def _json(raw):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                          parse_constant=_constant, parse_float=_float)
    except (UnicodeError, json.JSONDecodeError, RecursionError, OverflowError,
            ValueError) as exc:
        raise Refused("invalid JSON") from exc


def _raw(path):
    """Read only a regular file, disallowing a final-component symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_INPUT:
            raise Refused("source is not a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(MAX_INPUT + 1)
        if len(raw) > MAX_INPUT or len(raw) != info.st_size:
            raise Refused("source bytes are too large or changed during read")
        return raw
    finally:
        os.close(fd)


def _identifier(value):
    return type(value) is str and IDENTIFIER.fullmatch(value) is not None


def load_graph(path, digest):
    if type(digest) is not str or DIGEST.fullmatch(digest) is None:
        raise Refused("invalid declared source digest")
    raw = _raw(path)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise Refused("graph source digest mismatch")
    document = _json(raw)
    if not isinstance(document, dict):
        raise Refused("graph must be an object")
    nodes = document.get("nodes")
    edges = document.get("edges")
    if type(nodes) is not dict or not nodes or len(nodes) > MAX_NODES:
        raise Refused("node map missing, empty, or oversized")
    if type(edges) is not list or len(edges) > MAX_EDGES:
        raise Refused("edge list missing or oversized")
    for node, attributes in nodes.items():
        if not _identifier(node) or type(attributes) is not dict:
            raise Refused("invalid node identifier or metadata")
    unique, normal = set(), []
    for edge in edges:
        if type(edge) is not dict or set(edge) != EDGE_FIELDS:
            raise Refused("edge must carry from/to/relation/required exactly")
        src, dst, relation, required = (edge["from"], edge["to"],
                                        edge["relation"], edge["required"])
        if (not _identifier(src) or not _identifier(dst) or not _identifier(relation)
                or type(required) is not bool or src not in nodes or dst not in nodes):
            raise Refused("invalid typed edge or unknown endpoint")
        identity = (src, dst, relation, required)
        if identity in unique:
            raise Refused("duplicate typed edge")
        unique.add(identity)
        normal.append(identity)
    normal.sort()
    return {"identity": digest, "nodes": nodes, "edges": normal,
            "metadata": {k: v for k, v in document.items() if k not in {"nodes", "edges"}}}


def topology(node_ids, edges):
    """Exact ranks H_0 and H_1 of the UNDIRECTED one-dimensional CW complex.

    Every typed, directed edge contributes exactly one 1-cell, including
    parallel and loop edges. No 2-cells are inferred. H1 rank = E - V + H0.
    """
    vertices = sorted(node_ids)
    parent = {v: v for v in vertices}
    sizes = {v: 1 for v in vertices}
    incidence = {v: 0 for v in vertices}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for src, dst, _relation, _required in edges:
        incidence[src] += 1
        incidence[dst] += 1
        left, right = find(src), find(dst)
        if left != right:
            if sizes[left] < sizes[right] or (sizes[left] == sizes[right] and left > right):
                left, right = right, left
            parent[right] = left
            sizes[left] += sizes[right]

    groups = {}
    for node in vertices:
        groups.setdefault(find(node), []).append(node)
    components = sorted(groups.values(), key=lambda group: (-len(group), group))
    beta0, e, v = len(components), len(edges), len(vertices)
    return {
        "vertices": v,
        "edges": e,
        "beta0": beta0,
        "beta1": e - v + beta0,
        "component_sizes": [len(group) for group in components],
        "isolated_nodes": [node for node in vertices if incidence[node] == 0],
        "components": components,
    }


def summarize(graph):
    edges = graph["edges"]
    return {
        "raw_sha256": graph["identity"],
        "all_edges": topology(graph["nodes"], edges),
        "required_edges_only": topology(graph["nodes"], [e for e in edges if e[3]]),
    }


def _edge_record(edge):
    src, dst, relation, required = edge
    return {"from": src, "to": dst, "relation": relation, "required": required}


def _json_equal(left, right):
    """Compare parsed values without Python's bool/number coercion.

    Preserve existing numeric equality (1 == 1.0, 0 == -0.0) and parser
    precision. This is not lexical or lossless-decimal comparison; raw source
    digests retain byte identity. Object order is immaterial; array order is
    significant. Use an explicit stack to avoid a new recursion-depth limit.
    """
    pending = [(left, right)]
    while pending:
        a, b = pending.pop()
        if type(a) is bool or type(b) is bool:
            if type(a) is not type(b) or a != b:
                return False
        elif type(a) is dict and type(b) is dict:
            if a.keys() != b.keys():
                return False
            pending.extend((value, b[key]) for key, value in a.items())
        elif type(a) is list and type(b) is list:
            if len(a) != len(b):
                return False
            pending.extend(zip(a, b))
        elif a != b:
            return False
    return True


def compare(first, second):
    left, right = first["nodes"], second["nodes"]
    old, new = set(first["edges"]), set(second["edges"])
    old_ids, new_ids = set(left), set(right)
    a, b = summarize(first), summarize(second)
    return {
        "nodes_added": sorted(new_ids - old_ids),
        "nodes_removed": sorted(old_ids - new_ids),
        "node_attributes_changed": sorted(node for node in old_ids & new_ids
                                          if not _json_equal(left[node], right[node])),
        "typed_edges_added": [_edge_record(e) for e in sorted(new - old)],
        "typed_edges_removed": [_edge_record(e) for e in sorted(old - new)],
        "other_metadata_changed": not _json_equal(first["metadata"], second["metadata"]),
        "beta_delta": {
            subset: {key: b[subset][key] - a[subset][key] for key in ("beta0", "beta1")}
            for subset in ("all_edges", "required_edges_only")
        },
    }


def run(graph_path, graph_sha, compare_path=None, compare_sha=None):
    if (compare_path is None) != (compare_sha is None):
        raise Refused("comparison path and digest must both be specified")
    before = load_graph(graph_path, graph_sha)
    result = {
        "schema_version": 1,
        "scientific_effect": "NONE",
        "scientific_status_authority": False,
        "scope": "undirected-multigraph-1-skeleton-of-directed-typed-edges",
        "interpretation": "diagnostic-topological-ranks-only-not-proof-status-or-logical-gap",
        "before": summarize(before),
    }
    if compare_path is not None:
        after = load_graph(compare_path, compare_sha)
        result["after"] = summarize(after)
        result["delta"] = compare(before, after)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.splitlines()[0])
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--compare-sha256")
    args = parser.parse_args(argv)
    try:
        output = run(args.graph, args.sha256, args.compare, args.compare_sha256)
    except (Refused, OSError, TypeError, RecursionError, OverflowError):
        sys.stderr.write(REFUSAL)
        return 2
    sys.stdout.write(json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
