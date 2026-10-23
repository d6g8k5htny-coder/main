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

v2 mode (GB-5 contract in docs/plans/2026-10-08-artifact-index-contract.md):
  --v2 FILE             v2 root {schema_version:2,node_source,scope,bindings};
                        records use kind/repository/artifact_path/commit/
                        blob_sha/sha256 and refs owner/repo:path@commit#blob.
                        REPORT ONLY: exit 0 on well-formed input, 2 otherwise.

Exit codes (v1): 0 all bound; 1 unbound/stale items or a scope matching nothing;
2 malformed input, including an empty node set, a missing --site-nodes
directory, or an absolute/'..' --scope (fail closed; --report-only does not
override exit 2).
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
        if not Path(d).is_dir():
            raise InputError(f"--site-nodes is not a directory: {Path(d).name!r}")
        for html in sorted(Path(d).glob("*.html")):
            n = html.name[:-len(".html")]
            if not NODE_ID.fullmatch(n):
                raise InputError(f"invalid site node id {n!r}")
            nodes.add(n)
    if not nodes:
        # F1: an empty node set must fail closed, never read as ALL_BOUND.
        raise InputError("graph node set is empty")
    return nodes


def validate_scope(pattern: str) -> str:
    """F2/F3: scope globs must be repo-relative; no absolute, '..' or empty parts."""
    if (not isinstance(pattern, str) or not pattern or "\\" in pattern
            or "\x00" in pattern or pattern.startswith("/")
            or any(part in ("..", "") for part in pattern.split("/"))):
        raise InputError(f"--scope not repo-relative: {pattern!r}")
    return pattern


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
    in_scope, empty_scopes, scope_outside = set(), [], []
    for g in scopes:
        matched = False
        for p in root.glob(validate_scope(g)):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            try:
                p.resolve().relative_to(root)
            except ValueError:
                scope_outside.append(rel)  # symlink escaping root
                continue
            matched = True
            in_scope.add(rel)
        if not matched:
            empty_scopes.append(g)
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
        "empty_scopes": sorted(empty_scopes),
        "scope_outside_root": sorted(scope_outside),
    }
    failing = any(report[k] for k in (
        "unbound_nodes", "unbound_artifacts", "bindings_unknown_node",
        "bindings_missing_artifact", "bindings_stale_sha", "bindings_outside_root",
        "empty_scopes", "scope_outside_root"))
    report["status"] = "UNBOUND_ITEMS" if failing else "ALL_BOUND"
    return report, failing


# ---------------------------------------------------------------------------
# v2 report mode (GB-5 contract, main e1d1ebc2). REPORT ONLY: exits 0 on any
# well-formed input and 2 on malformed input; there is no enforce mode.
# ---------------------------------------------------------------------------
V2_ROOT_KEYS = {"schema_version", "node_source", "scope", "bindings"}
V2_NODE_SOURCE_KEYS = {"kind", "repository", "path", "commit"}
V2_NODE_SOURCE_KINDS = {"graph-readback", "site-projection"}
V2_KINDS = {"file", "tree", "historical", "absent", "external_unverified"}
V2_ROLES = {"source", "review", "kernel", "computation", "alignment", "teaching-source"}
V2_REQUIRED = {"node_id", "kind", "roles"}
V2_KIND_FIELDS = {
    "file": ({"repository", "artifact_path", "commit", "blob_sha", "sha256"}, set()),
    "historical": ({"repository", "artifact_path", "commit", "blob_sha", "sha256"}, set()),
    "tree": ({"repository", "artifact_path", "commit", "blob_sha"}, set()),
    "absent": ({"repository", "artifact_path", "reason", "last_seen_commit"}, set()),
    "external_unverified": ({"url"}, set()),
}
V2_OPTIONAL = {"artifact_id"}
HEX40 = re.compile(r"[0-9a-f]{40}")
REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def v2_ref(repository: str, path: str, commit: str, blob: str) -> str:
    """Compact reference form from the contract: owner/repo:path@commit#blob."""
    return f"{repository}:{path}@{commit}#{blob}"


def _v2_path(value) -> str:
    if (not isinstance(value, str) or not value or len(value.encode("utf-8")) > 1024
            or value.startswith("/") or "\\" in value or CONTROL.search(value)
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise InputError(f"v2 artifact_path not repo-relative: {value!r}")
    return value


def _hex(value, pattern, name):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise InputError(f"v2 {name} must be full lower-case hex")
    return value


def load_v2(path):
    doc = _strict_load(Path(path))
    if not isinstance(doc, dict) or set(doc) != V2_ROOT_KEYS:
        raise InputError(f"v2 root must have exactly keys {sorted(V2_ROOT_KEYS)}")
    if type(doc["schema_version"]) is not int or doc["schema_version"] != 2:
        raise InputError("v2 schema_version must be integer 2")
    ns = doc["node_source"]
    if not isinstance(ns, dict) or set(ns) != V2_NODE_SOURCE_KEYS:
        raise InputError("v2 node_source must have exactly kind, repository, path, commit")
    if ns["kind"] not in V2_NODE_SOURCE_KINDS:
        raise InputError("v2 node_source.kind unknown")
    if not isinstance(ns["repository"], str) or not REPO.fullmatch(ns["repository"]):
        raise InputError("v2 node_source.repository must be owner/name")
    _v2_path(ns["path"])
    _hex(ns["commit"], HEX40, "node_source.commit")
    scope = doc["scope"]
    if (not isinstance(scope, list) or scope != sorted(set(scope))
            or not all(isinstance(g, str) for g in scope)):
        raise InputError("v2 scope must be a sorted unique list of strings")
    for g in scope:
        validate_scope(g)
    recs = doc["bindings"]
    if not isinstance(recs, list):
        raise InputError("v2 bindings must be a list")
    seen, keys = set(), []
    for r in recs:
        if not isinstance(r, dict) or not V2_REQUIRED <= set(r):
            raise InputError("v2 binding missing node_id/kind/roles")
        kind = r["kind"]
        if kind not in V2_KINDS:
            raise InputError(f"v2 binding kind unknown: {kind!r}")
        need, _ = V2_KIND_FIELDS[kind]
        allowed = V2_REQUIRED | need | V2_OPTIONAL
        if not need <= set(r) or not set(r) <= allowed:
            raise InputError(f"v2 {kind} binding fields must be {sorted(V2_REQUIRED | need)} (+artifact_id)")
        if not isinstance(r["node_id"], str) or not NODE_ID.fullmatch(r["node_id"]):
            raise InputError("v2 invalid node_id")
        roles = r["roles"]
        if (not isinstance(roles, list) or not roles or roles != sorted(set(roles))
                or not set(roles) <= V2_ROLES):
            raise InputError("v2 roles must be a nonempty sorted unique subset of the v1 roles")
        if "artifact_id" in r and (not isinstance(r["artifact_id"], str)
                                   or not NODE_ID.fullmatch(r["artifact_id"])):
            raise InputError("v2 invalid artifact_id")
        if "repository" in r and (not isinstance(r["repository"], str)
                                  or not REPO.fullmatch(r["repository"])):
            raise InputError("v2 repository must be owner/name")
        if "artifact_path" in r:
            _v2_path(r["artifact_path"])
        for f in ("commit", "blob_sha", "last_seen_commit"):
            if f in r:
                _hex(r[f], HEX40, f)
        if "sha256" in r:
            _hex(r["sha256"], HEX64, "sha256")
        for f in ("reason", "url"):
            if f in r and (not isinstance(r[f], str) or not r[f] or CONTROL.search(r[f])):
                raise InputError(f"v2 {f} must be a nonempty string")
        key = (r["node_id"], r.get("repository", ""), r.get("artifact_path", r.get("url", "")))
        if key in seen:
            raise InputError(f"v2 duplicate binding {key}")
        seen.add(key)
        keys.append(key)
    if keys != sorted(keys):
        raise InputError("v2 bindings must sort by node_id, repository, artifact_path")
    return doc


def _v2_nodes(root: Path, ns):
    """Node universe only from the declared node_source file (graph.data.nodes,
    or a top-level nodes dict/list). Returns {node_id: node_object}."""
    target = (root / ns["path"]).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise InputError("v2 node_source.path escapes root")
    if not target.is_file():
        raise InputError("v2 node_source.path missing at checked ref")
    data = _strict_load(target)
    nodes = data
    if isinstance(nodes, dict) and "graph" in nodes:
        nodes = nodes["graph"]
    if isinstance(nodes, dict) and "data" in nodes:
        nodes = nodes["data"]
    if isinstance(nodes, dict) and "nodes" in nodes:
        nodes = nodes["nodes"]
    if isinstance(nodes, list):
        nodes = {n.get("id") if isinstance(n, dict) else n:
                 (n if isinstance(n, dict) else {}) for n in nodes}
    if not isinstance(nodes, dict) or not nodes:
        raise InputError("v2 node_source has no nodes")  # F1: fail closed
    for k in nodes:
        if not isinstance(k, str) or not NODE_ID.fullmatch(k):
            raise InputError(f"v2 invalid node id in node_source {k!r}")
    return nodes


def _git_tree_sha(root: Path, rel: str):
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", f"HEAD:{rel}"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and HEX40.fullmatch(sha) else None


def check_v2(root: Path, doc, repository: str):
    root = root.resolve()
    nodes = _v2_nodes(root, doc["node_source"])
    st = {k: [] for k in ("bound", "stale", "missing", "historical", "absent",
                          "external_unverified", "other_repository",
                          "tree_unverifiable", "unknown_node")}
    nodes_with_record, file_bound_paths = set(), set()
    for r in doc["bindings"]:
        item = {"node_id": r["node_id"], "kind": r["kind"]}
        if "artifact_path" in r and "repository" in r:
            item["artifact_path"] = r["artifact_path"]
        if r["kind"] in ("file", "historical", "tree"):
            item["ref"] = v2_ref(r["repository"], r["artifact_path"], r["commit"], r["blob_sha"])
        if r["node_id"] not in nodes:
            st["unknown_node"].append(item)
            continue
        nodes_with_record.add(r["node_id"])
        kind = r["kind"]
        if kind in ("historical", "absent", "external_unverified"):
            if kind == "external_unverified":
                item["url"] = r["url"]
            st[kind].append(item)
            continue
        if r["repository"] != repository:
            st["other_repository"].append(item)
            continue
        target = root / r["artifact_path"]
        try:
            target.resolve().relative_to(root)
        except ValueError:
            st["missing"].append(dict(item, detail="escapes_root"))
            continue
        if kind == "tree":
            if not target.is_dir():
                st["missing"].append(item)
                continue
            actual = _git_tree_sha(root, r["artifact_path"])
            if actual is None:
                st["tree_unverifiable"].append(item)
            elif actual == r["blob_sha"]:
                st["bound"].append(item)
            else:
                st["stale"].append(dict(item, actual_blob_sha=actual))
            continue
        if not target.is_file():
            st["missing"].append(item)
            continue
        data = target.resolve().read_bytes()
        blob, digest = git_blob_sha(data), hashlib.sha256(data).hexdigest()
        if blob == r["blob_sha"] and digest == r["sha256"]:
            st["bound"].append(item)
            file_bound_paths.add(r["artifact_path"])
        else:
            st["stale"].append(dict(item, actual_blob_sha=blob, actual_sha256=digest))
    in_scope, empty_scopes = set(), []
    for g in doc["scope"]:
        matched = False
        for p in root.glob(g):
            if p.is_file():
                try:
                    p.resolve().relative_to(root)
                except ValueError:
                    continue
                matched = True
                in_scope.add(p.relative_to(root).as_posix())
        if not matched:
            empty_scopes.append(g)

    # Node `source` rules from the GB-5 contract, fail-closed answers (e1d1ebc2
    # docs/plans/2026-10-08-artifact-index-contract.md). Report only.
    #  Q2 (L101) directory source: "Bind each file, or use kind: tree"; a tree
    #      record is evaluated per L87, and files under it stay unbound
    #      artifacts unless a `file` record names them (L94).
    #  Q3 (L102, L92) URL/external: source: external_unverified, "Never bound".
    #  hist.* nodes: CoS ruling 2026-10-23 12:24Z (475), fail-closed, in force
    #      until Dylan or the GB-5 owner changes it: a hist.* node may carry a
    #      historical record pinned to an old commit, but per L90 that never
    #      counts as bound -> reported as unbound_historical, never bound.
    hist_bound = [i for i in st["bound"] if i["node_id"].startswith("hist.")]
    st["bound"] = [i for i in st["bound"] if not i["node_id"].startswith("hist.")]
    kinds_by_node = {}
    bound_nodes = set()
    for r in doc["bindings"]:
        kinds_by_node.setdefault(r["node_id"], set()).add(r["kind"])
    for item in st["bound"]:
        bound_nodes.add(item["node_id"])
    dir_rule, ext_rule, hist_rule = [], [], []
    for nid in sorted(nodes):
        src = nodes[nid].get("source") if isinstance(nodes[nid], dict) else None
        kinds = sorted(kinds_by_node.get(nid, ()))
        if isinstance(src, str) and (src.startswith(("http://", "https://"))
                                     or src.startswith("external:")):
            ext_rule.append({
                "node_id": nid, "source": src, "record_kinds": kinds,
                "has_external_unverified_record": "external_unverified" in kinds,
                "bound": nid in bound_nodes})
        elif isinstance(src, str) and (src.endswith("/") or (root / src).is_dir()):
            ok = bool(set(kinds) & {"tree", "file"})
            dir_rule.append({
                "node_id": nid, "source": src, "record_kinds": kinds,
                "status": "pinned_by_tree_or_files" if ok else "loose_directory_unbound",
                "bound": nid in bound_nodes})
        if nid.startswith("hist."):
            hist_rule.append({
                "node_id": nid, "source": src, "record_kinds": kinds,
                "has_historical_record": "historical" in kinds,
                "status": "unbound_historical", "bound": False,
                "demoted_from_bound": sum(1 for i in hist_bound if i["node_id"] == nid)})
    for k in st:
        st[k].sort(key=lambda x: (x["node_id"], x.get("artifact_path", ""), x["kind"]))
    return {
        "schema_version": 2,
        "mode": "report",
        "scientific_effect": "NONE",
        "node_source": doc["node_source"],
        "counts": {"graph_nodes": len(nodes), "bindings": len(doc["bindings"]),
                   "in_scope_artifacts": len(in_scope),
                   **{k: len(v) for k, v in st.items()}},
        **st,
        "unbound_nodes": sorted(set(nodes) - nodes_with_record),
        "unbound_artifacts": sorted(in_scope - file_bound_paths),
        "empty_scopes": empty_scopes,
        "source_rules": {
            "contract": "docs/plans/2026-10-08-artifact-index-contract.md@e1d1ebc2",
            "directory_sources": dir_rule,         # Q2, L101 + L87 + L94
            "url_or_external_sources": ext_rule,   # Q3, L102 + L92
            "hist_nodes": hist_rule,               # CoS ruling 475 12:24Z + L90
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--bindings")
    ap.add_argument("--v2", help="v2 bindings file (report mode only)")
    ap.add_argument("--repository", default="d6g8k5htny-coder/main",
                    help="owner/name of --root, for v2 records")
    ap.add_argument("--graph-nodes", action="append", default=[])
    ap.add_argument("--site-nodes", action="append", default=[])
    ap.add_argument("--scope", action="append", default=[])
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    if args.v2 is not None:
        try:
            if args.bindings or args.graph_nodes or args.site_nodes or args.scope:
                raise InputError("--v2 takes node source and scope from the v2 file only")
            report = check_v2(Path(args.root), load_v2(args.v2), args.repository)
        except (InputError, OSError) as exc:
            print(f"GRAPH_BINDING_INPUT_ERROR: {exc}", file=sys.stderr)
            return 2
        sys.stdout.write(json.dumps(report, sort_keys=True, indent=1) + "\n")
        return 0  # v2 is report-only by contract until Dylan approves enforcement
    try:
        if not args.bindings:
            raise InputError("--bindings is required without --v2")
        if not args.graph_nodes and not args.site_nodes:
            raise InputError("need --graph-nodes and/or --site-nodes")
        for g in args.scope:
            validate_scope(g)
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
