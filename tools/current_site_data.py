#!/usr/bin/env python3
"""Observe immutable public research sources; never execute them or infer status."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote, urlsplit
import urllib.request


REPOSITORIES = ("d6g8k5htny-coder/main", "d6g8k5htny-coder/Math-")
GRAPH_PATH = "frontiers/downstream_gate_20260925/GRAPH.json"
EQUATION_PATHS = tuple(f"experiments/universality/{name}/PROOF.md" for name in (
    "fold_structural_universality", "periodized_gaussian_h0", "iid_short_bar_process"))
REFERENCE_PATHS = (
    (REPOSITORIES[0], "STATUS.md", "Current status source"),
    (REPOSITORIES[0], "docs/RESEARCH_INDEX.md", "Current research guide"),
    (REPOSITORIES[1], "PROOF_INDEX.md", "Current proof index"),
)
MAX_BYTES = 16 * 1024 * 1024
MAX_TREE_ENTRIES = 100_000
MAX_GRAPH_NODES = 10_000
MAX_GRAPH_EDGES = 50_000
MAX_EQUATIONS = 2_000
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_json(raw):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ValueError("nonfinite JSON value")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid JSON source") from error


def identity(value, kind):
    if not isinstance(value, str) or not HEX40.fullmatch(value):
        raise ValueError("invalid " + kind + " identity")
    return value


def safe_path(path):
    if (not isinstance(path, str) or not path or len(path) > 4096
            or path.startswith("/") or "\\" in path or "%" in path
            or any(ord(char) < 32 or ord(char) == 127 for char in path)
            or any(part in ("", ".", "..") for part in path.split("/"))):
        raise ValueError("unsafe source path")
    return path


def public_inventory_path(path):
    """Filter copied/private operational material, without judging scientific scope."""
    parts = path.lower().split("/")
    if any(any(token in part for token in ("legacy", "quarantine", "recovery", "scratch"))
           for part in parts):
        return False
    if parts[0] in {"audits", "history", "operational", "operations", "private"}:
        return False
    if any(part.startswith(("drive_", "drive-", "drive ", "dropbox_", "dropbox-"))
           or part in {"drive", "dropbox", "private", "operational"} for part in parts):
        return False
    return True


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ValueError("source redirect refused")


class GitHubReader:
    """Bounded HTTPS reads; an optional API token is only an HTTP header."""
    def __init__(self, *, opener=None, token=None, timeout=30, max_bytes=MAX_BYTES,
                 max_requests=16, max_total_bytes=64 * 1024 * 1024):
        if not 0 < timeout <= 30 or not 0 < max_bytes <= MAX_BYTES:
            raise ValueError("invalid source read bounds")
        if type(max_requests) is not int or not 0 < max_requests <= 32:
            raise ValueError("invalid source request bound")
        if type(max_total_bytes) is not int or not 0 < max_total_bytes <= 64 * 1024 * 1024:
            raise ValueError("invalid total byte bound")
        self.opener = opener if opener is not None else urllib.request.build_opener(NoRedirect())
        self.token = os.environ.get("GITHUB_TOKEN") if token is None else token
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_requests = max_requests
        self.max_total_bytes = max_total_bytes
        self.requests = 0
        self.total_bytes = 0

    def get_bytes(self, url):
        parsed = urlsplit(url)
        api = parsed.hostname == "api.github.com"
        prefix = "/repos/" if api else "/"
        if (parsed.scheme != "https" or parsed.netloc not in {
                "api.github.com", "raw.githubusercontent.com"}
                or parsed.fragment or not any(parsed.path.startswith(prefix + repo + "/")
                                              for repo in REPOSITORIES)):
            raise ValueError("nonpublic source URL refused")
        if self.requests >= self.max_requests:
            raise ValueError("source request bound exceeded")
        headers = {"User-Agent": "research-current-site-source-observation",
                   "Accept": "application/vnd.github+json" if api else "application/octet-stream"}
        if api and self.token:
            headers["Authorization"] = "Bearer " + self.token
        self.requests += 1
        limit = min(self.max_bytes, self.max_total_bytes - self.total_bytes)
        if limit < 0:
            raise ValueError("total source byte bound exceeded")
        with self.opener.open(urllib.request.Request(url, headers=headers), timeout=self.timeout) as response:
            raw = response.read(limit + 1)
        self.total_bytes += len(raw)
        if len(raw) > limit:
            raise ValueError("source byte bound exceeded")
        return raw

    def get_json(self, url):
        return parse_json(self.get_bytes(url))


def read_repository(reader, repository, pin):
    base = "https://api.github.com/repos/" + repository
    response = reader.get_json(base + "/commits/" + (pin or "HEAD"))
    if not isinstance(response, dict):
        raise ValueError("invalid commit response")
    commit = identity(response.get("sha"), "commit")
    if pin is not None and commit != pin:
        raise ValueError("commit differs from requested pin")
    try:
        tree = identity(response["commit"]["tree"]["sha"], "tree")
    except (TypeError, KeyError) as error:
        raise ValueError("missing commit tree identity") from error
    response = reader.get_json(base + "/git/trees/" + tree + "?recursive=1")
    entries = validate_tree(response, tree)
    return {"repository": repository, "commit": commit, "tree": tree}, entries


def validate_tree(response, expected_tree):
    if not isinstance(response, dict) or response.get("sha") != expected_tree:
        raise ValueError("tree response identity mismatch")
    if response.get("truncated") is not False:
        raise ValueError("tree missing completeness marker or truncated")
    rows = response.get("tree")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_TREE_ENTRIES:
        raise ValueError("invalid tree entry inventory")
    entries = {}
    children = {"": []}
    modes = {"040000": "tree", "100644": "blob", "100755": "blob",
             "120000": "blob", "160000": "commit"}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid tree entry")
        path = safe_path(row.get("path"))
        if path in entries:
            raise ValueError("duplicate tree path")
        identity(row.get("sha"), "tree entry")
        if row.get("mode") not in modes or row.get("type") != modes[row["mode"]]:
            raise ValueError("invalid tree entry mode or type")
        if row["type"] == "blob" and (type(row.get("size")) is not int or row["size"] < 0):
            raise ValueError("missing or invalid blob size")
        entries[path] = row
        if row["type"] == "tree":
            children[path] = []
    for path, row in entries.items():
        parent, _, name = path.rpartition("/")
        if parent not in children:
            raise ValueError("incomplete tree parent inventory")
        children[parent].append((name.encode("utf-8"), row))
    # Reconstruct every Git tree object from the complete recursive response.
    # This also catches omissions when an upstream response says truncated=false.
    for path in sorted(children, key=lambda part: part.count("/") + bool(part), reverse=True):
        body = b""
        for name, row in sorted(children[path], key=lambda child:
                                child[0] + (b"/" if child[1]["type"] == "tree" else b"")):
            mode = row["mode"].lstrip("0").encode("ascii")
            body += mode + b" " + name + b"\0" + bytes.fromhex(row["sha"])
        computed = hashlib.sha1(b"tree " + str(len(body)).encode("ascii") + b"\0" + body).hexdigest()
        wanted = entries[path]["sha"] if path else expected_tree
        if computed != wanted:
            raise ValueError("Git tree integrity mismatch")
    return entries


def read_source(reader, repository, entries, path):
    safe_path(path)
    row = entries.get(path)
    if row is None:
        raise ValueError("required source missing: " + path)
    if row["type"] != "blob" or row["mode"] not in {"100644", "100755"}:
        raise ValueError("required source is not a regular file: " + path)
    if row["size"] > MAX_BYTES:
        raise ValueError("required source exceeds byte bound")
    url = ("https://raw.githubusercontent.com/" + repository["repository"] + "/"
           + repository["commit"] + "/" + quote(path, safe="/"))
    raw = reader.get_bytes(url)
    if not isinstance(raw, bytes) or len(raw) != row["size"]:
        raise ValueError("source byte size mismatch: " + path)
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    if blob != row["sha"]:
        raise ValueError("source Git blob integrity mismatch: " + path)
    source = {"repository": repository["repository"], "commit": repository["commit"],
              "path": path, "blob": blob, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return source, raw


def validate_graph(graph):
    if (not isinstance(graph, dict) or type(graph.get("schema_version")) is not int
            or graph["schema_version"] != 1):
        raise ValueError("invalid graph schema")
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if (not isinstance(nodes, dict) or not 1 <= len(nodes) <= MAX_GRAPH_NODES
            or not all(isinstance(key, str) and key and isinstance(value, dict)
                       for key, value in nodes.items())):
        raise ValueError("invalid graph nodes")
    for node in nodes.values():
        for field in ("classification", "kind", "layer"):
            if not isinstance(node.get(field), str) or not node[field]:
                raise ValueError("missing typed graph node metadata: " + field)
    if not isinstance(edges, list) or len(edges) > MAX_GRAPH_EDGES:
        raise ValueError("invalid graph edges")
    indegrees = dict.fromkeys(nodes, 0)
    adjacency = {node: [] for node in nodes}
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("invalid graph edge")
        if type(edge.get("required")) is not bool:
            raise ValueError("graph edge required field must be boolean")
        source, target = edge.get("from"), edge.get("to")
        if not isinstance(source, str) or not isinstance(target, str) or source not in nodes or target not in nodes:
            raise ValueError("graph edge refers to absent node")
        if not isinstance(edge.get("relation"), str) or not edge["relation"]:
            raise ValueError("missing graph edge relation")
        if edge["required"]:
            adjacency[source].append(target)
            indegrees[target] += 1
    queue = deque(node for node, count in indegrees.items() if count == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for target in adjacency[node]:
            indegrees[target] -= 1
            if indegrees[target] == 0:
                queue.append(target)
    if visited != len(nodes):
        raise ValueError("required graph edge cycle")


def equation_blocks(raw, source):
    """Retain original indented display blocks and their exact inclusive line span."""
    try:
        lines = raw.decode("utf-8").splitlines(keepends=True)
    except UnicodeError as error:
        raise ValueError("invalid UTF-8 equation source") from error
    result = []
    start = None
    end = None
    for index, line in enumerate(lines + ["END"], 1):
        indented = line.startswith(("    ", "\t")) and bool(line.strip())
        if indented:
            start = index if start is None else start
            end = index
        elif line.strip() and start is not None:
            text = "".join(lines[start - 1:end])
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            result.append({"id": f"{source['repository']}:{source['path']}:{start}-{end}",
                           "text": text, "repository": source["repository"], "commit": source["commit"],
                           "path": source["path"], "line_start": start, "line_end": end, "sha256": digest})
            start = end = None
    return result


def generate(*, reader=None, main_commit=None, math_commit=None, observed_at=None):
    for pin in (main_commit, math_commit):
        if pin is not None:
            identity(pin, "commit pin")
    if observed_at is None:
        observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    try:
        datetime.strptime(observed_at, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as error:
        raise ValueError("invalid UTC observation time") from error
    reader = reader if reader is not None else GitHubReader()
    repositories = []
    inventories = {}
    sources = []
    for repository, pin in zip(REPOSITORIES, (main_commit, math_commit)):
        resolved, entries = read_repository(reader, repository, pin)
        repositories.append(resolved)
        inventories[repository] = entries
        for path, row in sorted(entries.items()):
            if row["type"] == "blob" and row["mode"] in {"100644", "100755"} and public_inventory_path(path):
                sources.append({"repository": repository, "commit": resolved["commit"],
                                "path": path, "blob": row["sha"], "bytes": row["size"]})
    graph_identity, raw = read_source(reader, repositories[1], inventories[REPOSITORIES[1]], GRAPH_PATH)
    graph = parse_json(raw)
    validate_graph(graph)
    references = []
    for repository, path, label in REFERENCE_PATHS:
        resolved = repositories[REPOSITORIES.index(repository)]
        source, raw = read_source(reader, resolved, inventories[repository], path)
        try:
            raw.decode("utf-8")
        except UnicodeError as error:
            raise ValueError("invalid UTF-8 reference source") from error
        references.append({"label": label, **source})
    equations = []
    for path in EQUATION_PATHS:
        source, raw = read_source(reader, repositories[0], inventories[REPOSITORIES[0]], path)
        equations.extend(equation_blocks(raw, source))
        if len(equations) > MAX_EQUATIONS:
            raise ValueError("equation collection bound exceeded")
    return {"schema_version": 1, "observed_at": observed_at, "scientific_effect": "NONE",
            "repositories": repositories, "sources": sources,
            "graph": {"identity": graph_identity, "data": graph},
            "equations": equations, "references": references}


def write_current(output, **options):
    data = generate(**options)
    raw = (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".current-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-commit", help="immutable main checkout commit")
    parser.add_argument("--math-commit", help="immutable Math- commit; otherwise resolve its default ref once")
    args = parser.parse_args(argv)
    try:
        data = write_current(args.output, main_commit=args.main_commit, math_commit=args.math_commit)
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.exit(1, f"current source refresh refused: {error}\n")
    print(json.dumps({"output": str(args.output), "observed_at": data["observed_at"],
                      "repositories": data["repositories"], "sources": len(data["sources"]),
                      "equations": len(data["equations"]), "scientific_effect": "NONE"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
