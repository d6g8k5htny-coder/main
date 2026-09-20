#!/usr/bin/env python3
"""Derive a bounded, unsigned research checkpoint; never write source/status data.

This is a view of recorded dependencies and outcomes, not another governing
register, proof verifier, trusted timestamp, or complete history of refusals.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

# The command is read-only even when the caller did not set this environment flag.
sys.dont_write_bytecode = True
TOOL_ROOT = Path(__file__).resolve().parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))
from engine import receipt as R  # noqa: E402
from engine.operations import trial as T  # noqa: E402
from tools import operations_check as O  # noqa: E402

SCHEMA = "q0.research-frontier/v1"
GRAPH = "claims/graph.json"
CATALOG = "engine/operations/REGISTRY.json"
PATTERNS = ("engine/operations/trials/*.json", "engine/receipts/*/*.json")
IMPLEMENTATION = (
    "tools/research_frontier.py",
    "engine/receipt.py", "engine/operations/__init__.py",
    "engine/operations/trial.py", "tools/operations_check.py",
)
LAYERS = ("status_frozen_v2_2", "status_register_note")
EDGE_TYPES = ("depends_on", "sub_obligations")
UNRESOLVED = ("OPEN", "NOT_CLOSED")
NODE_FIELDS = (
    "track", "grade", *LAYERS, "technical_status", "register_status",
    "requires_independent_verdict", "independent_review_state",
    "independence_credit", "source", "exact_object", "body_bytes", "body_sha256",
    "proposed_layer_verbatim", "proposed_layer_source",
)
SCOPE = {
    "required_files": [GRAPH, CATALOG], "record_globs": list(PATTERNS),
    "node_projection_fields": list(NODE_FIELDS), "edge_types": list(EDGE_TYPES),
    "unresolved_status_tokens": list(UNRESOLVED),
    "exclusions": [
        "Drive live state and governing Work Events; no new dispositions",
        "vault, legacy and quarantine bodies", "Git history and deleted records",
        "bridge receipts, nested receipt directories and non-JSON records",
        "unpersisted exceptions, rejected attempts and all other non-actions",
        "slack registry, numerical certificates and unprojected graph fields",
        "historical operation catalogs: only the current pinned catalog is supported",
    ],
}
LIMITS = [
    "Unsigned derived view; a recomputable hash is not authentication or authority.",
    "Local observation time is untrusted; this is not a trusted timestamp.",
    "No claim, premise, grade or gate is moved; coauthor work earns zero independence credit.",
    "Paths mean recorded reachability only: no AND/OR proof composition or minimal cut is inferred.",
    "Only OPEN and NOT_CLOSED premise tokens are selected, separately in each layer; other tokens are not declared satisfied.",
    "No all-history or research-wide completeness; absence of a diagnostic means only none in the selected records.",
    "Git tree identifies HEAD; dirty/status metadata does not identify all uncommitted content. Only listed source bytes are pinned.",
    "Source membership and bytes are rechecked during capture, without an atomic filesystem or remote lock.",
    "Record validators check existing receipt/trial formats; graph checking here covers structure and projected types, not every claim firewall or any mathematics.",
]
PAYLOAD_KEYS = {
    "observed_at_local_utc", "git", "implementation", "scope", "sources", "nodes",
    "edges", "unresolved_dependency_paths", "recorded_diagnostics", "limits",
    "authority", "independence_credit",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class Refused(ValueError):
    """Invalid, unsupported or stale input; no partial checkpoint is emitted."""


def require(ok, message):
    if not ok:
        raise Refused(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise Refused(f"non-finite JSON number: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def source_paths(root):
    paths = {GRAPH, CATALOG}
    for pattern in PATTERNS:
        paths.update(p.relative_to(root).as_posix() for p in root.glob(pattern))
    return sorted(paths)


def read_source(root, relative):
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
            f"source symlink or escape refused: {relative}")
    # Refuse symlinked ancestors too, including links to other permitted-looking data.
    require(all(not p.is_symlink() for p in path.parents if p != root.parent),
            f"source ancestor symlink refused: {relative}")
    data = path.read_bytes()
    return data, {"path": relative, "sha256": digest(data), "bytes": len(data)}


def git_identity(root):
    def git(*args):
        run = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *args],
                             capture_output=True, check=False)
        require(run.returncode == 0, "Git identity unavailable: " + run.stderr.decode(errors="replace").strip())
        return run.stdout

    require(Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() == root.resolve(),
            "--root must be the repository root")
    head = git("rev-parse", "HEAD").decode().strip()
    tree = git("rev-parse", "HEAD^{tree}").decode().strip()
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    require(re.fullmatch(r"[0-9a-f]{40,64}", head) and re.fullmatch(r"[0-9a-f]{40,64}", tree),
            "invalid Git object identity")
    return {"commit": head, "head_tree": tree, "dirty": bool(status),
            "status_porcelain_sha256": digest(status)}


def graph_view(graph):
    require(isinstance(graph, dict), "graph must be an object")
    require(set(graph) == {"_comment", "as_of", "tracks", "premises", "claims", "firewalls"},
            "unsupported graph top-level fields")
    require(isinstance(graph["tracks"], dict) and isinstance(graph["firewalls"], list),
            "invalid tracks/firewalls shape")
    nodes, edges = {}, []
    for collection in ("premises", "claims"):
        require(isinstance(graph[collection], dict), f"{collection} must be an object")
        for name, node in sorted(graph[collection].items()):
            require(name and name not in nodes and isinstance(node, dict),
                    f"invalid/duplicate graph node: {name}")
            require(node.get("track") in graph["tracks"], f"unknown track on {name}")
            source = node.get("source")
            require((isinstance(source, str) and source.strip()) or
                    (isinstance(source, dict) and source and all(isinstance(v, str) and v.strip() for v in source.values())),
                    f"{name}: missing/invalid source")
            required = LAYERS if collection == "premises" else ("grade",)
            for field in required:
                require(isinstance(node.get(field), str) and node[field].strip(),
                        f"{name}: missing/invalid {field}")
            for field in NODE_FIELDS:
                if field not in node:
                    continue
                if field == "source":
                    continue
                if field == "requires_independent_verdict":
                    require(type(node[field]) is bool, f"{name}: invalid {field}")
                elif field in ("body_bytes", "independence_credit"):
                    require(type(node[field]) is int and node[field] >= 0, f"{name}: invalid {field}")
                else:
                    require(isinstance(node[field], str), f"{name}: invalid {field}")
            if "body_sha256" in node:
                require(HEX64.fullmatch(node["body_sha256"]) is not None, f"{name}: invalid body_sha256")
            require(not any(k.startswith("depends_") and k not in EDGE_TYPES for k in node),
                    f"{name}: unsupported dependency semantics")
            nodes[name] = {"kind": collection[:-1] if collection == "claims" else "premise",
                           **{k: node[k] for k in NODE_FIELDS if k in node}}
            for edge_type in EDGE_TYPES:
                refs = node.get(edge_type, [])
                require(isinstance(refs, list) and all(isinstance(v, str) for v in refs),
                        f"{name}: invalid {edge_type}")
                require(len(refs) == len(set(refs)), f"{name}: repeated {edge_type} reference")
                edges.extend({"from": name, "to": target, "type": edge_type} for target in sorted(refs))
    require(0 < len(nodes) <= 1000, "graph node limit exceeded or graph empty")
    adjacency = {name: [] for name in nodes}
    for edge in edges:
        require(edge["to"] in nodes, f"missing graph reference: {edge['from']} -> {edge['to']}")
        adjacency[edge["from"]].append(edge)
    # Kahn's algorithm avoids accepting cycles with no reachable open premise.
    incoming = {name: 0 for name in nodes}
    for edge in edges:
        incoming[edge["to"]] += 1
    ready = [name for name in nodes if incoming[name] == 0]
    visited = 0
    while ready:
        name = ready.pop()
        visited += 1
        for edge in adjacency[name]:
            incoming[edge["to"]] -= 1
            if incoming[edge["to"]] == 0:
                ready.append(edge["to"])
    require(visited == len(nodes), "graph dependency cycle")
    paths = []
    # Preserve every typed route, including diamonds; no shortest-path collapse.
    traversals = 0
    for origin in sorted(nodes):
        stack = [(origin, [origin], [])]
        while stack:
            name, route, types = stack.pop()
            traversals += 1
            require(traversals <= 100000, "dependency path budget exceeded; no partial view")
            if len(route) > 1 and nodes[name]["kind"] == "premise":
                for layer in LAYERS:
                    if nodes[name][layer] in UNRESOLVED:
                        paths.append({"nodes": route, "edge_types": types,
                                      "layer": layer, "recorded_status": nodes[name][layer]})
            for edge in reversed(adjacency[name]):
                stack.append((edge["to"], route + [edge["to"]], types + [edge["type"]]))
    return nodes, sorted(edges, key=lambda e: (e["from"], e["type"], e["to"])), paths


def diagnostics(relative, obj):
    """Only structured outcomes and exact runner-emitted refusal prefixes qualify."""
    out = []

    def add(field, outcome, detail):
        out.append({"source": relative, "field": field, "outcome": outcome, "detail": detail})

    if relative.startswith("engine/operations/trials/"):
        result, evidence = obj["Verified result"], obj["Run evidence"]
        if result == "NOT_RUN":
            add("Verified result", result, evidence["reason_not_run"])
        elif result == "IDENTITY_NOT_REPRODUCED":
            add("Verified result", result, {"failure": evidence["failure"],
                "failed_steps": [s for s in evidence["steps"] if s["equal"] is False]})
    else:
        if obj["outcome"] != "RAN":
            add("outcome", obj["outcome"], {"error": obj["error"], "notes": obj["notes"]})
        for index, note in enumerate(obj["notes"]):
            if note.startswith(("total() refused: ", "certified_enclosure() refused: ")):
                add(f"notes/{index}", "RECORDED_REFUSAL", note)
    return out


def derive(root):
    paths = source_paths(root)
    raw, sources, objects = {}, [], {}
    for relative in paths:
        data, identity = read_source(root, relative)
        raw[relative] = data
        sources.append(identity)
        objects[relative] = strict_json(data)
    nodes, edges, unresolved = graph_view(objects[GRAPH])
    catalog = objects[CATALOG]
    require(isinstance(catalog, dict) and isinstance(catalog.get("entries"), list), "invalid operation catalog")
    entries = T.registry_entries(catalog)
    require(len(entries) == len(catalog["entries"]), "duplicate operation catalog ID")
    for key, entry in entries.items():
        require(isinstance(key, str) and isinstance(entry.get("cells"), dict)
                and set(entry["cells"]) == set(T.REGISTER_COLUMNS)
                and isinstance(entry.get("git_side"), dict), "invalid operation catalog entry")
        row = [entry["cells"][column] for column in T.REGISTER_COLUMNS]
        require(all(isinstance(value, str) for value in row), "invalid operation catalog cells")
        require(entry.get("row_sha256") == T.row_sha256(row), "operation row hash mismatch")
    records, seen = [], set()
    for relative in paths:
        if relative in (GRAPH, CATALOG):
            continue
        obj = objects[relative]
        if relative.startswith("engine/operations/trials/"):
            require(isinstance(obj, dict), f"{relative}: trial must be an object")
            require(obj.get("Library / catalog SHA-256") == digest(raw[CATALOG]),
                    f"{relative}: stale/unsupported catalog identity")
            problems, _ = O.check_trial(str(root / relative), entries, digest(raw[CATALOG]), str(root), lambda _: None)
            record_id = obj.get("Trial ID")
        else:
            problems = R.validate_receipt_object(obj)
            record_id = obj.get("receipt_id") if isinstance(obj, dict) else None
            require(Path(relative).stem == record_id, f"{relative}: receipt filename/ID mismatch")
        require(not problems, f"{relative}: invalid record: {'; '.join(problems)}")
        require(record_id not in seen, f"duplicate record identity: {record_id}")
        seen.add(record_id)
        records.extend(diagnostics(relative, obj))
    # Existing check_trial reads the file itself. Verify it and all other inputs
    # still contain the exact bytes initially parsed, and catch new/deleted rows.
    require(paths == source_paths(root), "source membership changed during capture")
    for relative in paths:
        require(read_source(root, relative)[0] == raw[relative], f"source changed during capture: {relative}")
    return {"sources": sources, "nodes": nodes, "edges": edges,
            "unresolved_dependency_paths": unresolved, "recorded_diagnostics": records}


def snapshot(root=TOOL_ROOT):
    root = Path(root).resolve()
    before = git_identity(root)
    implementation = [read_source(TOOL_ROOT, path)[1] for path in IMPLEMENTATION]
    view = derive(root)
    require(git_identity(root) == before, "Git identity changed during capture")
    require(implementation == [read_source(TOOL_ROOT, path)[1] for path in IMPLEMENTATION],
            "implementation changed during capture")
    payload = {"observed_at_local_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "git": before, "implementation": implementation, "scope": strict_json(canonical(SCOPE)),
               **view, "limits": list(LIMITS), "authority": "NONE", "independence_credit": 0}
    return {"schema": SCHEMA, "payload": payload, "payload_sha256": digest(canonical(payload))}


def validate_checkpoint(obj):
    require(isinstance(obj, dict) and set(obj) == {"schema", "payload", "payload_sha256"},
            "invalid checkpoint envelope")
    require(obj["schema"] == SCHEMA, "unsupported checkpoint schema")
    payload = obj["payload"]
    require(isinstance(payload, dict) and set(payload) == PAYLOAD_KEYS, "invalid checkpoint fields")
    require(obj["payload_sha256"] == digest(canonical(payload)), "checkpoint payload hash mismatch")
    require(payload["scope"] == SCOPE and payload["limits"] == LIMITS
            and payload["authority"] == "NONE" and type(payload["independence_credit"]) is int
            and payload["independence_credit"] == 0, "checkpoint scope/authority mismatch")
    require(isinstance(payload["nodes"], dict), "invalid checkpoint node map")
    for field in ("edges", "unresolved_dependency_paths", "recorded_diagnostics"):
        require(isinstance(payload[field], list), f"invalid checkpoint {field}")
    for field in ("sources", "implementation"):
        require(isinstance(payload[field], list), f"invalid checkpoint {field}")
        names = set()
        for entry in payload[field]:
            require(isinstance(entry, dict) and set(entry) == {"path", "sha256", "bytes"}, "invalid source identity")
            path = entry["path"]
            require(isinstance(path, str) and path not in names and not Path(path).is_absolute()
                    and ".." not in Path(path).parts, "invalid/duplicate source path")
            require(isinstance(entry["sha256"], str) and HEX64.fullmatch(entry["sha256"])
                    and type(entry["bytes"]) is int and entry["bytes"] >= 0, "invalid source hash/bytes")
            names.add(path)
    require([item["path"] for item in payload["implementation"]] == list(IMPLEMENTATION),
            "checkpoint implementation scope mismatch")
    source_names = [item["path"] for item in payload["sources"]]
    require(source_names == sorted(source_names) and {GRAPH, CATALOG} <= set(source_names),
            "checkpoint source scope mismatch")
    from fnmatch import fnmatchcase
    require(all(path in (GRAPH, CATALOG) or
                (fnmatchcase(path, PATTERNS[0]) and len(Path(path).parts) == 4) or
                (fnmatchcase(path, PATTERNS[1]) and len(Path(path).parts) == 4)
                for path in source_names), "checkpoint source outside declared scope")
    git = payload["git"]
    require(isinstance(git, dict) and set(git) == {"commit", "head_tree", "dirty", "status_porcelain_sha256"},
            "invalid checkpoint Git fields")
    require(type(git["dirty"]) is bool and all(isinstance(git[k], str) and
            re.fullmatch(r"[0-9a-f]{40,64}", git[k]) for k in ("commit", "head_tree"))
            and isinstance(git["status_porcelain_sha256"], str)
            and HEX64.fullmatch(git["status_porcelain_sha256"]), "invalid checkpoint Git identity")
    stamp = dt.datetime.fromisoformat(payload["observed_at_local_utc"])
    require(stamp.utcoffset() == dt.timedelta(0), "checkpoint time must name UTC (still untrusted)")
    # Rebuild only the closed projection to check its types, routes and layering.
    projected = {"_comment": "projection", "as_of": "untrusted", "tracks": {},
                 "claims": {}, "premises": {}, "firewalls": []}
    for name, node in payload["nodes"].items():
        require(isinstance(node, dict) and "kind" in node
                and set(node) <= {"kind", *NODE_FIELDS}, "invalid projected node fields")
        require(node["kind"] in ("claim", "premise"), "invalid projected node kind")
        require(isinstance(node.get("track"), str), "invalid projected track")
        projected["tracks"][node["track"]] = "transcribed"
        projected[node["kind"] + "s"][name] = {k: v for k, v in node.items() if k != "kind"}
    for edge in payload["edges"]:
        require(isinstance(edge, dict) and set(edge) == {"from", "to", "type"}
                and edge["type"] in EDGE_TYPES and edge["from"] in payload["nodes"]
                and edge["to"] in payload["nodes"], "invalid projected dependency")
        node = payload["nodes"][edge["from"]]
        projected[node["kind"] + "s"][edge["from"]].setdefault(edge["type"], []).append(edge["to"])
    nodes, edges, paths = graph_view(projected)
    require(nodes == payload["nodes"] and edges == payload["edges"]
            and paths == payload["unresolved_dependency_paths"], "checkpoint dependency projection mismatch")
    for record in payload["recorded_diagnostics"]:
        require(isinstance(record, dict) and set(record) == {"source", "field", "outcome", "detail"}
                and record["source"] in source_names and isinstance(record["field"], str)
                and record["outcome"] in {"NOT_RUN", "IDENTITY_NOT_REPRODUCED", "RECORDED_REFUSAL",
                                          *set(R.OUTCOMES) - {"RAN"}}, "invalid recorded diagnostic")
        if record["source"].startswith("engine/operations/trials/"):
            require(record["field"] == "Verified result" and record["outcome"] in
                    {"NOT_RUN", "IDENTITY_NOT_REPRODUCED"}, "invalid trial diagnostic selector")
        else:
            require(record["source"].startswith("engine/receipts/"), "invalid diagnostic source")
            if record["outcome"] == "RECORDED_REFUSAL":
                require(re.fullmatch(r"notes/[0-9]+", record["field"]) and
                        isinstance(record["detail"], str) and record["detail"].startswith(
                            ("total() refused: ", "certified_enclosure() refused: ")),
                        "invalid recorded refusal selector")
            else:
                require(record["field"] == "outcome" and record["outcome"] in set(R.OUTCOMES) - {"RAN"},
                        "invalid receipt diagnostic selector")
    require(len({canonical(record) for record in payload["recorded_diagnostics"]}) ==
            len(payload["recorded_diagnostics"]), "duplicate recorded diagnostic")
    return payload


def check(checkpoint, root=TOOL_ROOT):
    payload = validate_checkpoint(checkpoint)
    current = snapshot(root)["payload"]
    # Time differs by definition. Everything else must replay against current inputs.
    for key in sorted(PAYLOAD_KEYS - {"observed_at_local_utc"}):
        require(payload[key] == current[key], f"stale or altered checkpoint field: {key}")
    return payload


def semantic_diff(old, new):
    left, right = validate_checkpoint(old), validate_checkpoint(new)
    changes = {}
    node_changes = {}
    for name in sorted(set(left["nodes"]) | set(right["nodes"])):
        before, after = left["nodes"].get(name), right["nodes"].get(name)
        if before != after:
            node_changes[name] = {"before": before, "after": after}
    if node_changes:
        changes["nodes"] = node_changes
    for field in ("edges", "unresolved_dependency_paths", "recorded_diagnostics"):
        before = {canonical(item): item for item in left[field]}
        after = {canonical(item): item for item in right[field]}
        if before != after:
            changes[field] = {"removed": [before[k] for k in sorted(before.keys() - after.keys())],
                              "added": [after[k] for k in sorted(after.keys() - before.keys())]}
    identities = {}
    for field in ("git", "sources", "implementation"):
        if left[field] != right[field]:
            identities[field] = {"before": left[field], "after": right[field]}
    return {"schema": "q0.research-frontier-diff/v1", "known_field_changes": changes,
            "identity_changes": identities, "authority": "NONE", "independence_credit": 0,
            "does_not_establish": "Compares unsigned checkpoint assertions and payload integrity only; no historical source replay, mathematical equivalence, promotion or completeness. Unprojected source changes appear only as identity changes."}


def write_snapshot(obj, output, root=TOOL_ROOT):
    validate_checkpoint(obj)
    path = Path(output)
    require(not path.resolve().is_relative_to(Path(root).resolve())
            and not path.resolve().is_relative_to(TOOL_ROOT),
            "checkpoint output must be outside the repository and implementation tree")
    # No mkdir, overwrite or source write. O_EXCL rejects an existing symlink too.
    with path.open("xb") as stream:
        stream.write(canonical(obj) + b"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=TOOL_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("snapshot")
    capture.add_argument("--output", type=Path)
    sub.add_parser("self-check")
    verify = sub.add_parser("check")
    verify.add_argument("checkpoint", type=Path)
    compare = sub.add_parser("diff")
    compare.add_argument("old", type=Path)
    compare.add_argument("new", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in ("snapshot", "self-check"):
            obj = snapshot(args.root)
            if args.command == "self-check":
                p = validate_checkpoint(obj)
                print(f"research_frontier: OK sources={len(p['sources'])} nodes={len(p['nodes'])} edges={len(p['edges'])} unresolved_paths={len(p['unresolved_dependency_paths'])} recorded_diagnostics={len(p['recorded_diagnostics'])}; authority=NONE")
            elif args.output:
                write_snapshot(obj, args.output, args.root)
                print(f"research_frontier: wrote {args.output}; payload_sha256={obj['payload_sha256']}; UNSIGNED")
            else:
                print(canonical(obj).decode())
        elif args.command == "check":
            check(strict_json(args.checkpoint.read_bytes()), args.root)
            print("research_frontier: OK current-source replay; UNSIGNED, authority=NONE")
        else:
            print(canonical(semantic_diff(strict_json(args.old.read_bytes()),
                                          strict_json(args.new.read_bytes()))).decode())
        return 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
        print(f"research_frontier: REFUSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
