#!/usr/bin/env python3
"""Bounded public Pages history custody; never execute downloaded content.

Version 2 retains compact immutable source material and index context while the
full current observation and static reading refresh independently. Earlier v1
archives and their original renderer remain admitted unchanged. The manifest
hashes asset-release-normalized HTML, exact JSON and exact deterministic ZIP
bytes. Restore admits all downloads and the durable manifest pin before writes.
"""
import argparse
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import zipfile

from current_site_data import parse_json
from site_asset_release import transform, without_release

ORIGIN = "https://d6g8k5htny-coder.github.io/main/site/"
MAX_SNAPSHOTS = 512
MAX_HISTORY_BYTES = 768 * 1024 * 1024
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_MEMBERS = 20006
HEX = re.compile(r"[0-9a-f]{64}\Z")
NODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\.html\Z")
LEDGER = ".current-site-restored-history.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def manifest_bytes(data):
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def semantic_digest(data):
    return digest(json.dumps({k: v for k, v in data.items() if k != "observed_at"},
                             ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def canonical_bytes(path, raw):
    if path.endswith(".html"):
        return transform(raw.decode("utf-8"), ".html", without_release).encode("utf-8")
    return raw


def closed(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("invalid closed " + label + " schema")


def sha(value):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError("invalid history digest")
    return value


def timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", value):
        raise ValueError("invalid history timestamp")
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def member_path(path):
    if path in ("current.json", "current-meta.json"):
        return path
    if not isinstance(path, str) or not path.startswith("nodes/") or not NODE.fullmatch(path[6:]):
        raise ValueError("unsafe history member path")
    return path


def descriptor(path, raw):
    raw = canonical_bytes(path, raw)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("history file capacity exceeded")
    return {"path": path, "bytes": len(raw), "sha256": digest(raw)}


def validate_descriptor(row, *, bundle=False):
    closed(row, ("path", "bytes", "sha256"), "file descriptor")
    sha(row["sha256"])
    if type(row["bytes"]) is not int or not 0 <= row["bytes"] <= MAX_FILE_BYTES:
        raise ValueError("history per-file capacity exceeded")
    if bundle:
        if not isinstance(row["path"], str) or not re.fullmatch(r"snapshots/[0-9a-f]{64}/archive\.zip", row["path"]):
            raise ValueError("unsafe history bundle path")
    else:
        member_path(row["path"])


def _v1_history_size(entries, aliases):
    # Count both uncompressed copies and ZIP bytes conservatively. Reserve 256
    # bytes for the three stylesheet release URLs on each served static page.
    total = 0
    lookup = {}
    for entry in entries:
        total += entry["bundle"]["bytes"]
        for field in ("files", "alias_files"):
            for row in entry[field]:
                total += row["bytes"] + (256 if row["path"].endswith(".html") else 0)
                if field == "alias_files":
                    lookup[(entry["id"], row["path"])] = row["bytes"] + 256
    for alias in aliases:
        key = (alias["snapshot"], alias["path"])
        if key not in lookup:
            raise ValueError("unresolved history alias binding")
        total += lookup[key]
    return total


def _v1_validate_manifest(data):
    closed(data, ("schema_version", "origin", "scientific_effect", "representation",
                  "current", "archives", "aliases", "total_bytes"), "history manifest")
    if (type(data["schema_version"]) is not int or data["schema_version"] != 1 or data["origin"] != ORIGIN or data["scientific_effect"] != "NONE"
            or data["representation"] != "asset-release-normalized-html-v1"):
        raise ValueError("invalid public history identity")
    if not isinstance(data["archives"], list) or not 1 <= len(data["archives"]) <= MAX_SNAPSHOTS:
        raise ValueError("history snapshot capacity exceeded")
    ids = set()
    for entry in data["archives"]:
        closed(entry, ("id", "semantic_sha256", "observed_at", "renderer", "files", "alias_files", "bundle"), "archive")
        if entry["renderer"] != "research-static-v1":
            raise ValueError("unadmitted historical renderer")
        sha(entry["id"]); sha(entry["semantic_sha256"]); timestamp(entry["observed_at"])
        if entry["id"] in ids:
            raise ValueError("duplicate history snapshot")
        ids.add(entry["id"])
        validate_descriptor(entry["bundle"], bundle=True)
        if entry["bundle"]["path"] != f'snapshots/{entry["id"]}/archive.zip':
            raise ValueError("history bundle does not bind its snapshot")
        for field in ("files", "alias_files"):
            rows = entry[field]
            if not isinstance(rows, list) or not 1 <= len(rows) <= 10003:
                raise ValueError("invalid history member inventory")
            seen = set()
            for row in rows:
                validate_descriptor(row)
                if row["path"].lower() in seen:
                    raise ValueError("duplicate history member")
                seen.add(row["path"].lower())
                if field == "alias_files" and not row["path"].startswith("nodes/"):
                    raise ValueError("invalid history alias member")
    if data["current"] not in ids or not isinstance(data["aliases"], list):
        raise ValueError("invalid current history pointer")
    seen = set()
    for alias in data["aliases"]:
        closed(alias, ("path", "snapshot"), "alias")
        if alias["snapshot"] not in ids or not isinstance(alias["path"], str) or not alias["path"].startswith("nodes/"):
            raise ValueError("invalid alias snapshot")
        member_path(alias["path"])
        if alias["path"].lower() in seen:
            raise ValueError("duplicate history alias")
        seen.add(alias["path"].lower())
    expected_aliases = {r["path"].lower() for e in data["archives"] for r in e["alias_files"]}
    if seen != expected_aliases:
        raise ValueError("history alias union is incomplete")
    total = _v1_history_size(data["archives"], data["aliases"])
    if type(data["total_bytes"]) is not int or data["total_bytes"] != total or total > MAX_HISTORY_BYTES:
        raise ValueError("history total capacity exceeded or inconsistent")
    return data


# Frozen v2 source-format admission. Format or selection changes require a new
# archive version; current UI/schema changes must not reinterpret older frames.
def _v2_safe_path(path):
    if (not isinstance(path, str) or not path or len(path) > 4096
            or path.startswith("/") or "\\" in path or "%" in path
            or any(ord(char) < 32 or ord(char) == 127 for char in path)
            or any(part in ("", ".", "..") for part in path.split("/"))):
        raise ValueError("unsafe source path")
    return path


def _v2_public_inventory_path(path):
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


def _v2_validate_source_payload(data):
    REPOSITORIES = {"d6g8k5htny-coder/main", "d6g8k5htny-coder/Math-"}
    MATH = "d6g8k5htny-coder/Math-"
    HEX40 = re.compile(r"[0-9a-f]{40}\Z")
    HEX64 = re.compile(r"[0-9a-f]{64}\Z")
    NODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z")
    MAX_PAYLOAD = 16 * 1024 * 1024
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or data.get("scientific_effect") != "NONE"
            or set(data) != {"schema_version", "observed_at", "scientific_effect", "repositories",
                            "sources", "graph", "equations", "references"}):
        raise ValueError("invalid current source schema")
    observed = data.get("observed_at")
    if not isinstance(observed, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", observed):
        raise ValueError("invalid observation timestamp")
    datetime.strptime(observed, "%Y-%m-%dT%H:%M:%SZ")
    repos = data.get("repositories")
    if not isinstance(repos, list) or len(repos) != 2:
        raise ValueError("incomplete pinned repository inventory")
    commits = {}
    for repo in repos:
        if (not isinstance(repo, dict) or set(repo) != {"repository", "commit", "tree"}
                or repo.get("repository") not in REPOSITORIES
                or repo["repository"] in commits
                or not isinstance(repo.get("commit"), str) or not HEX40.fullmatch(repo["commit"])
                or not isinstance(repo.get("tree"), str) or not HEX40.fullmatch(repo["tree"])):
            raise ValueError("invalid pinned repository identity")
        commits[repo["repository"]] = repo["commit"]

    def source_identity(source, digest=False, reference=False):
        fields = {"repository", "commit", "path", "blob", "bytes"}
        if digest:
            fields.add("sha256")
        if reference:
            fields.add("label")
        if (not isinstance(source, dict) or source.get("repository") not in commits
                or set(source) - fields
                or source.get("commit") != commits[source["repository"]]
                or not isinstance(source.get("blob"), str) or not HEX40.fullmatch(source["blob"])
                or type(source.get("bytes")) is not int or not 0 <= source["bytes"] <= MAX_PAYLOAD):
            raise ValueError("invalid source identity")
        _v2_safe_path(source.get("path"))
        if not _v2_public_inventory_path(source["path"]):
            raise ValueError("private or operational source ingress refused")
        if digest and (not isinstance(source.get("sha256"), str) or not HEX64.fullmatch(source["sha256"])):
            raise ValueError("invalid source digest")

    if not isinstance(data.get("sources"), list):
        raise ValueError("missing source inventory")
    identities = set()
    for source in data["sources"]:
        source_identity(source)
        key = (source["repository"], source["path"])
        if key in identities:
            raise ValueError("duplicate source inventory identity")
        identities.add(key)
    references = data.get("references")
    if not isinstance(references, list) or len(references) > 16:
        raise ValueError("missing source references")
    for reference in references:
        source_identity(reference, True, True)
        if "label" in reference and (not isinstance(reference["label"], str) or len(reference["label"]) > 256):
            raise ValueError("source reference label exceeds its public bound")
    reference_keys = {(r["repository"], r["path"]) for r in references}
    if not {("d6g8k5htny-coder/main", "STATUS.md"),
            ("d6g8k5htny-coder/main", "docs/RESEARCH_INDEX.md"),
            (MATH, "PROOF_INDEX.md")} <= reference_keys:
        raise ValueError("missing required current source reference")
    graph = data.get("graph")
    if not isinstance(graph, dict) or set(graph) != {"identity", "data"}:
        raise ValueError("invalid closed graph snapshot schema")
    source_identity(graph.get("identity"), True)
    if graph["identity"]["repository"] != MATH:
        raise ValueError("graph must be bound to the admitted Math- repository")
    content = graph.get("data")
    if not isinstance(content, dict) or not isinstance(content.get("nodes"), dict) or not isinstance(content.get("edges"), list):
        raise ValueError("invalid graph snapshot")
    nodes, edges = content["nodes"], content["edges"]
    if not nodes or len(nodes) > 10000 or len(edges) > 50000:
        raise ValueError("invalid graph population")
    filenames = set()
    for node_id, node in nodes.items():
        if (not NODE_ID.fullmatch(node_id) or node_id.lower() == "index"
                or node_id.lower() in filenames or not isinstance(node, dict)):
            raise ValueError("unsafe or colliding graph node identity")
        filenames.add(node_id.lower())
    outgoing = {node_id: [] for node_id in nodes}
    indegree = {node_id: 0 for node_id in nodes}
    for edge in edges:
        if (not isinstance(edge, dict) or edge.get("from") not in nodes or edge.get("to") not in nodes
                or type(edge.get("required")) is not bool or not isinstance(edge.get("relation"), str)):
            raise ValueError("invalid graph edge or absent node")
        if edge["required"]:
            outgoing[edge["from"]].append(edge["to"])
            indegree[edge["to"]] += 1
    ready = [node_id for node_id, count in indegree.items() if count == 0]
    visited = 0
    while ready:
        node_id = ready.pop()
        visited += 1
        for other in outgoing[node_id]:
            indegree[other] -= 1
            if indegree[other] == 0:
                ready.append(other)
    if visited != len(nodes):
        raise ValueError("required graph edges contain a cycle")
    equations = data.get("equations")
    if not isinstance(equations, list) or len(equations) > 2000:
        raise ValueError("invalid equation inventory")
    for equation in equations:
        if (not isinstance(equation, dict) or set(equation) != {"id", "repository", "commit", "path", "text", "line_start", "line_end", "sha256"}
                or equation.get("repository") not in commits
                or equation.get("commit") != commits[equation["repository"]]
                or not isinstance(equation.get("id"), str) or not isinstance(equation.get("text"), str)
                or not isinstance(equation.get("sha256"), str) or not HEX64.fullmatch(equation["sha256"])
                or type(equation.get("line_start")) is not int or type(equation.get("line_end")) is not int
                or equation["line_start"] < 1 or equation["line_end"] < equation["line_start"]):
            raise ValueError("invalid equation identity")
        _v2_safe_path(equation.get("path"))
        if not _v2_public_inventory_path(equation["path"]):
            raise ValueError("private or operational equation ingress refused")
        if hashlib.sha256(equation["text"].encode("utf-8")).hexdigest() != equation["sha256"]:
            raise ValueError("equation text digest mismatch")
    return data


def _v2_historical_render_budget(data, meta):
    """Conservative pre-render bound for both static copies, without joining them.

    Repeated source disclosures, escaped metadata, indentation, edge rows and
    the complete inventory are counted before large per-node HTML allocation.
    """
    repeated = len(_v2_material_section(data, meta, site_prefix="../../../", snapshot_prefix="../").encode()) + 6000
    graph = data["graph"]["data"]
    edge_cost = {node: 0 for node in graph["nodes"]}
    for edge in graph["edges"]:
        cost = 1500 + 16 * len(edge["relation"].encode())
        edge_cost[edge["from"]] += cost
        edge_cost[edge["to"]] += cost
    total_nodes = 0
    index_nodes = 0
    for node_id, node in graph["nodes"].items():
        stack = [(node, 0)]; values = 0
        while stack:
            value, depth = stack.pop(); values += 1
            if depth > 32:
                raise ValueError("pre-render metadata depth capacity exceeded")
            if isinstance(value, dict): stack.extend((v, depth + 1) for v in value.values())
            elif isinstance(value, list): stack.extend((v, depth + 1) for v in value)
        compact = len(json.dumps(node, ensure_ascii=False, separators=(",", ":")).encode())
        cost = repeated + 16 * compact + 64 * values + edge_cost[node_id]
        if cost > MAX_FILE_BYTES:
            raise ValueError("pre-render per-page capacity exceeded")
        total_nodes += cost
        index_nodes += 500 + 16 * (len(node_id.encode()) + len(str(node.get("kind", "")).encode())
                                  + len(str(node.get("classification", "")).encode()))
    inventory = sum(256 + len(_v1_escape(_v1_source_url(s)).encode()) + len(_v1_escape(s["path"]).encode())
                    for s in data["sources"])
    index = repeated + inventory + index_nodes + sum(3000 + 8 * len(e["text"].encode()) for e in data["equations"])
    if index > MAX_FILE_BYTES or 2 * (total_nodes + index + meta["bytes"] + 1024) > MAX_HISTORY_BYTES:
        raise ValueError("pre-render public history capacity exceeded")
    return 2 * (total_nodes + index + meta["bytes"] + 1024)


def mathematical_projection(data):
    """Finite source projection, preserving original pins without inventing joins."""
    wanted = {(data["graph"]["identity"]["repository"], data["graph"]["identity"]["path"])}
    for node in data["graph"]["data"]["nodes"].values():
        for field in ("source", "review_source"):
            value = node.get(field)
            if isinstance(value, str):
                wanted.add(("d6g8k5htny-coder/Math-", value))
    wanted.update((e["repository"], e["path"]) for e in data["equations"])
    selected = [s for s in data["sources"] if (s["repository"], s["path"]) in wanted]
    projection = {**data, "schema_version": 2, "coverage": "mathematical-material-only",
                  "sources": sorted(selected, key=lambda s: (s["repository"], s["path"]))}
    # Rendering is bound to the exact canonical projection bytes, including its
    # deterministic dictionary order, rather than the caller's insertion order.
    return parse_json(manifest_bytes(projection))


def validate_projection(data):
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data.get("schema_version") != 2 or data.get("coverage") != "mathematical-material-only"):
        raise ValueError("invalid mathematical projection coverage")
    complete_shape = {k: v for k, v in data.items() if k != "coverage"}
    complete_shape["schema_version"] = 1
    _v2_validate_source_payload(complete_shape)
    if mathematical_projection(complete_shape) != data:
        raise ValueError("irrelevant inventory in mathematical projection")
    return data


def mathematical_digest(data):
    projected = mathematical_projection(data)
    def identity(row):
        return {k: v for k, v in row.items() if k not in ("commit", "label")}
    material = {"schema_version": 2, "scientific_effect": "NONE",
                "graph": {"identity": identity(projected["graph"]["identity"]), "data": projected["graph"]["data"]},
                "sources": [identity(s) for s in projected["sources"]],
                "equations": [identity(e) for e in projected["equations"]],
                "references": sorted((identity(r) for r in projected["references"]), key=lambda r: (r["repository"], r["path"]))}
    return digest(json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def history_size(entries, aliases, observation=None):
    if observation is None and all("binding" not in a for a in aliases):
        return _v1_history_size(entries, aliases)
    # Reserve the entire admitted manifest-file allowance (avoiding a circular
    # manifest hash/size calculation), plus 64 KiB for root metadata and the
    # bounded operational ledger (the deployment pipeline excludes that ledger).
    # File/page/ZIP byte identities remain exact; aggregate accounting is safely
    # conservative and includes the served registry itself.
    total = MAX_FILE_BYTES + 64 * 1024
    for entry in entries:
        total += entry["bundle"]["bytes"]
        for field in ("files", "alias_files"):
            total += sum(r["bytes"] + (256 if r["path"].endswith(".html") else 0) for r in entry[field])
    total += sum(a["bytes"] + 256 for a in aliases)
    if observation:
        total += observation["payload"]["bytes"]
    return total


def capacity_record(entries, total):
    return {"snapshots": len(entries), "maximum_snapshots": MAX_SNAPSHOTS,
            "bytes": total, "maximum_bytes": MAX_HISTORY_BYTES,
            "warning": len(entries) * 4 >= MAX_SNAPSHOTS * 3 or total * 4 >= MAX_HISTORY_BYTES * 3}


def validate_manifest(data):
    if isinstance(data, dict) and data.get("schema_version") == 1:
        return _v1_validate_manifest(data)
    closed(data, ("schema_version", "origin", "scientific_effect", "representation", "current",
                  "current_observation", "archives", "aliases", "total_bytes", "capacity"), "history manifest v2")
    if (type(data["schema_version"]) is not int or data["schema_version"] != 2 or data["origin"] != ORIGIN
            or data["scientific_effect"] != "NONE" or data["representation"] != "asset-release-normalized-html-v1"):
        raise ValueError("invalid public history identity")
    entries = data["archives"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_SNAPSHOTS:
        raise ValueError("history snapshot capacity exceeded")
    ids = set(); union = set()
    for entry in entries:
        fields = ("id", "semantic_sha256", "observed_at", "renderer", "files", "alias_files", "bundle")
        version = entry.get("archive_version", 1) if isinstance(entry, dict) else 0
        closed(entry, fields if version == 1 else (*fields, "archive_version"), "archive")
        if ((version == 1 and entry["renderer"] != "research-static-v1")
                or (version == 2 and (type(entry["archive_version"]) is not int or entry["renderer"] != "research-static-v2"))
                or version not in (1, 2)):
            raise ValueError("unadmitted historical renderer")
        sha(entry["id"]); sha(entry["semantic_sha256"]); timestamp(entry["observed_at"])
        if entry["id"] in ids or (version == 2 and entry["semantic_sha256"] != entry["id"]):
            raise ValueError("duplicate or unbound mathematical snapshot")
        ids.add(entry["id"])
        validate_descriptor(entry["bundle"], bundle=True)
        if entry["bundle"]["path"] != f'snapshots/{entry["id"]}/archive.zip':
            raise ValueError("history bundle does not bind its snapshot")
        for field in ("files", "alias_files"):
            rows = entry[field]
            minimum = 1 if field == "files" or version == 1 else 0
            if not isinstance(rows, list) or not minimum <= len(rows) <= 10003 or (version == 2 and field == "alias_files" and rows):
                raise ValueError("invalid history member inventory")
            seen = set()
            for row in rows:
                validate_descriptor(row)
                if row["path"].lower() in seen or (field == "alias_files" and not row["path"].startswith("nodes/")):
                    raise ValueError("duplicate or invalid history member")
                seen.add(row["path"].lower())
                if row["path"].startswith("nodes/"):
                    union.add(row["path"].lower())
    if data["current"] not in ids or not isinstance(data["aliases"], list) or len(data["aliases"]) > 5120000:
        raise ValueError("invalid current history pointer")
    observation = data["current_observation"]
    closed(observation, ("payload", "files"), "current observation")
    validate_descriptor(observation["payload"])
    if observation["payload"]["path"] != "current.json" or not isinstance(observation["files"], list) or not 1 <= len(observation["files"]) <= 10001:
        raise ValueError("invalid live observation inventory")
    live = {}
    for row in observation["files"]:
        validate_descriptor(row)
        if not row["path"].startswith("nodes/") or row["path"].lower() in live:
            raise ValueError("invalid live observation route")
        live[row["path"].lower()] = row
    seen = set()
    for alias in data["aliases"]:
        closed(alias, ("path", "snapshot", "binding", "bytes", "sha256"), "alias v2")
        validate_descriptor({k: alias[k] for k in ("path", "bytes", "sha256")})
        if not alias["path"].startswith("nodes/") or alias["snapshot"] not in ids or alias["binding"] not in ("archive", "current_observation") or alias["path"].lower() in seen:
            raise ValueError("invalid alias mathematical binding")
        seen.add(alias["path"].lower())
        if alias["binding"] == "current_observation":
            if alias["snapshot"] != data["current"] or live.get(alias["path"].lower()) != {k: alias[k] for k in ("path", "bytes", "sha256")}:
                raise ValueError("live alias does not bind the exact observation")
        elif alias["path"].lower() in live:
            raise ValueError("current node rebound to historical observation")
    if seen != union or {a["path"].lower() for a in data["aliases"] if a["binding"] == "current_observation"} != set(live):
        raise ValueError("history alias union is incomplete")
    total = history_size(entries, data["aliases"], observation)
    if type(data["total_bytes"]) is not int or total != data["total_bytes"] or total > MAX_HISTORY_BYTES:
        raise ValueError("history total capacity exceeded or inconsistent")
    if data["capacity"] != capacity_record(entries, total):
        raise ValueError("history capacity disclosure mismatch")
    return data


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ValueError("public history redirect refused")


class HistoryReader:
    def __init__(self, opener=None):
        self.opener = opener or urllib.request.build_opener(NoRedirect())
        self.requests = 0
        self.total_bytes = 0

    def get_bytes(self, path):
        if path not in ("snapshots/index.json", "current-meta.json", "current.json") and not re.fullmatch(
                r"snapshots/[0-9a-f]{64}/archive\.zip", path if isinstance(path, str) else ""):
            raise ValueError("nonpublic history request path refused")
        if self.requests >= MAX_SNAPSHOTS + 3:
            raise ValueError("history request capacity exceeded")
        self.requests += 1
        url = ORIGIN + path
        request = urllib.request.Request(url, headers={"User-Agent": "current-site-public-history",
                                                      "Cache-Control": "no-cache", "Pragma": "no-cache",
                                                      "Accept-Encoding": "identity"})
        with self.opener.open(request, timeout=20) as response:
            if response.geturl() != url or response.status != 200:
                raise ValueError("public history response identity mismatch")
            encoding = response.headers.get("Content-Encoding", "identity")
            if encoding not in ("", "identity"):
                raise ValueError("encoded public history response refused")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isascii() or not length.isdecimal() or int(length) > MAX_FILE_BYTES):
                raise ValueError("history response byte bound exceeded")
            raw = response.read(MAX_FILE_BYTES + 1)
        self.total_bytes += len(raw)
        if (len(raw) > MAX_FILE_BYTES or (length is not None and len(raw) != int(length))
                or self.total_bytes > MAX_HISTORY_BYTES + 3 * MAX_FILE_BYTES):
            raise ValueError("history response byte bound exceeded")
        return raw


class StaticHTML(HTMLParser):
    TAGS = {"html", "head", "meta", "title", "link", "body", "main", "nav", "section", "h1", "h2", "h3",
            "p", "time", "code", "dl", "dt", "dd", "ul", "li", "a", "pre"}
    ATTRS = {"lang", "charset", "name", "content", "http-equiv", "rel", "href", "class", "id", "role",
             "aria-label", "datetime", "data-current-digest", "data-current-bytes", "data-axis", "data-state"}
    def __init__(self, snapshot, path, data):
        super().__init__(convert_charrefs=True)
        self.snapshot = snapshot
        self.path = path
        self.data = data
        self.bound = False
        self.csp = False
        self.capture = None
        self.h1 = ""
        self.pre = ""
        self.axes = {}
        self.local_links = []
        from current_site_build import source_url, review_urls
        self.external_links = {source_url(s) for s in data["sources"] + data["references"] + [data["graph"]["identity"]]}
        self.external_links.update(f'https://github.com/{r["repository"]}/tree/{r["commit"]}' for r in data["repositories"])
        self.external_links.update(u for n in data["graph"]["data"]["nodes"].values() for u in review_urls(n))

    def handle_starttag(self, tag, attrs):
        if tag not in self.TAGS or len(attrs) != len(dict(attrs)) or any(k not in self.ATTRS for k, v in attrs):
            raise ValueError("active or unknown history markup refused")
        attrs = dict(attrs)
        if tag in ("h1", "pre"):
            self.capture = tag
        if "data-current-digest" in attrs:
            if attrs["data-current-digest"] != self.snapshot:
                raise ValueError("static history snapshot binding mismatch")
            self.bound = True
        if tag == "meta":
            if "http-equiv" in attrs:
                if attrs["http-equiv"].lower() != "content-security-policy":
                    raise ValueError("active history metadata refused")
                parts = [p.strip().split(None, 1) for p in attrs.get("content", "").split(";") if p.strip()]
                if any(len(p) != 2 for p in parts) or len({p[0] for p in parts}) != len(parts):
                    raise ValueError("ambiguous history CSP refused")
                rules = dict(parts)
                required = {"default-src": "'none'", "script-src": "'none'", "base-uri": "'none'", "form-action": "'none'"}
                if any(rules.get(k) != v for k, v in required.items()):
                    raise ValueError("static history CSP refusal missing")
                if set(rules) - (set(required) | {"style-src", "img-src"}):
                    raise ValueError("unknown history CSP directive")
                self.csp = True
            elif not (attrs.get("charset") == "utf-8" or attrs.get("name") == "viewport"):
                raise ValueError("unknown history metadata refused")
        if "data-axis" in attrs:
            axis = attrs["data-axis"]
            if axis in self.axes or axis not in ("source", "review", "kernel", "computation", "alignment"):
                raise ValueError("invalid static evidence axis")
            self.axes[axis] = attrs.get("data-state")
        if "href" in attrs:
            if tag not in ("a", "link"):
                raise ValueError("unsupported history URL attribute")
            value = attrs["href"]
            url = urlsplit(value)
            if url.scheme or url.netloc:
                if tag != "a" or value not in self.external_links:
                    raise ValueError("unsafe external history link")
            elif (not value or value.startswith("/") or "\\" in value or "%" in value or url.query
                  or any(ord(c) < 32 for c in value)):
                raise ValueError("unsafe local history link")
            else:
                location = self.path[len("aliases/"):] if self.path.startswith("aliases/") else f"snapshots/{self.snapshot}/{self.path}"
                stack = location.split("/")[:-1]
                for part in url.path.split("/"):
                    if part == "..":
                        if not stack:
                            raise ValueError("unsafe history link escapes public site")
                        stack.pop()
                    elif part not in ("", "."):
                        stack.append(part)
                target = "/".join(stack)
                nodes = self.data["graph"]["data"]["nodes"]
                is_index = location.endswith("/index.html")
                node_id = PurePosixPath(location).stem
                neighbors = {e["to"] for e in self.data["graph"]["data"]["edges"] if e["from"] == node_id}
                neighbors.update(e["from"] for e in self.data["graph"]["data"]["edges"] if e["to"] == node_id)
                allowed = {"index.html", "explore.html", "research.html", "workspace.html", "dependencies.html", "nodes/index.html",
                           f"snapshots/{self.snapshot}/current.json", f"snapshots/{self.snapshot}/nodes/index.html"}
                allowed.update(f"snapshots/{self.snapshot}/nodes/{n}.html" for n in (nodes if is_index else neighbors | {node_id}))
                if is_index and self.path.startswith("aliases/"):
                    allowed.update(f"nodes/{n}.html" for n in nodes)
                if tag == "link":
                    if attrs.get("rel") != "stylesheet" or target not in {"brand.css", "style.css", "current.css"} or url.fragment:
                        raise ValueError("unsafe history stylesheet target")
                elif target not in allowed or url.fragment not in ("", "inventory", "current-atlas", "sources"):
                    raise ValueError("unbound local history link")
    handle_startendtag = handle_starttag

    def handle_endtag(self, tag):
        if tag not in self.TAGS:
            raise ValueError("unknown history closing tag")
        if tag == self.capture:
            self.capture = None

    def handle_data(self, value):
        if self.capture == "h1": self.h1 += value
        if self.capture == "pre": self.pre += value


def validate_static(raw, snapshot, path, data):
    parser = StaticHTML(snapshot, path, data)
    parser.feed(raw.decode("utf-8")); parser.close()
    if not parser.bound or not parser.csp:
        raise ValueError("static history binding or CSP missing")
    name = PurePosixPath(path).stem
    if name != "index":
        node = data["graph"]["data"]["nodes"].get(name)
        if node is None or parser.h1 != name or parse_json(parser.pre.encode()) != node:
            raise ValueError("static raw metadata differs from the archived node")
        from current_site_build import axes_html
        expected = {axis: "recorded" if (axis == "source" and bool(node.get("source"))) or (
            axis == "review" and any(k.startswith("review_") and bool(v) for k, v in node.items())) else "not_recorded"
                    for axis in ("source", "review", "kernel", "computation", "alignment")}
        if parser.axes != expected:
            raise ValueError("static evidence metadata differs from the archived node")


def payload_record(snapshot, files):
    from current_site_build import validate_payload
    raw = files.get("current.json")
    if raw is None:
        raise ValueError("archived payload missing")
    data = parse_json(raw)
    meta = parse_json(files.get("current-meta.json", b""))
    if not isinstance(meta, dict) or type(meta.get("bytes")) is not int:
        raise ValueError("invalid typed archived payload metadata")
    if isinstance(data, dict) and data.get("schema_version") == 2:
        validate_projection(data)
        closed(meta, ("sha256", "bytes", "observed_at", "mathematical_sha256", "archive_version"), "archived mathematical metadata")
        if type(meta.get("archive_version")) is not int or meta["archive_version"] != 2:
            raise ValueError("invalid typed archived mathematical version")
        expected = {"sha256": digest(raw), "bytes": len(raw), "observed_at": data["observed_at"],
                    "mathematical_sha256": mathematical_digest(data), "archive_version": 2}
        if meta != expected or snapshot != expected["mathematical_sha256"]:
            raise ValueError("archived mathematical payload identity mismatch")
    else:
        data = validate_payload(data)
        closed(meta, ("sha256", "bytes", "observed_at"), "archived payload metadata")
        if digest(raw) != snapshot or meta != {"sha256": snapshot, "bytes": len(raw), "observed_at": data["observed_at"]}:
            raise ValueError("archived payload metadata mismatch")
    names = {"current.json", "current-meta.json", "nodes/index.html"}
    names.update(f"nodes/{node}.html" for node in data["graph"]["data"]["nodes"])
    if set(files) != names:
        raise ValueError("archive membership differs from its admitted graph")
    return data, meta


def deterministic_zip(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, raw in sorted(files.items()):
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    raw = stream.getvalue()
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("history ZIP capacity exceeded")
    return raw


def admit_bundle(entry, raw):
    if len(raw) != entry["bundle"]["bytes"] or digest(raw) != entry["bundle"]["sha256"]:
        raise ValueError("history ZIP digest or byte count mismatch")
    descriptors = {row["path"]: row for row in entry["files"]}
    descriptors.update({"aliases/" + row["path"]: row for row in entry["alias_files"]})
    result = {}
    try:
        if not raw.startswith(b"PK\x03\x04") or raw.rfind(b"PK\x05\x06") != len(raw) - 22:
            raise ValueError("history ZIP prefix, comment or trailing bytes refused")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBERS or len(infos) != len({i.filename for i in infos}) or {i.filename for i in infos} != set(descriptors):
                raise ValueError("history ZIP membership mismatch")
            for info in infos:
                mode = info.external_attr >> 16
                if (info.is_dir() or info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))
                        or info.file_size != descriptors[info.filename]["bytes"] or info.file_size > MAX_FILE_BYTES):
                    raise ValueError("unsafe or oversized history ZIP member")
                content = archive.read(info)
                if digest(content) != descriptors[info.filename]["sha256"]:
                    raise ValueError("history member digest mismatch")
                if info.filename.endswith(".html"):
                    if canonical_bytes(info.filename, content) != content:
                        raise ValueError("history bundle representation is not canonical")
                result[info.filename] = content
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, UnicodeError) as error:
        raise ValueError("invalid history ZIP") from error
    files = {name: raw for name, raw in result.items() if not name.startswith("aliases/")}
    data, meta = payload_record(entry["id"], files)
    version = entry.get("archive_version", 1)
    if version == 2:
        if data["schema_version"] != 2 or entry["renderer"] != STATIC_RENDERER_V2 or any(n.startswith("aliases/") for n in result):
            raise ValueError("mathematical archive representation mismatch")
        if mathematical_digest(data) != entry["semantic_sha256"] or data["observed_at"] != entry["observed_at"]:
            raise ValueError("history mathematical identity mismatch")
        _v2_historical_render_budget(data, meta)
        expected = render_static_v2(data, meta, archived=True)
    else:
        if data["schema_version"] != 1 or entry["renderer"] != STATIC_RENDERER_V1:
            raise ValueError("historical renderer version mismatch")
        alias_names = {"aliases/" + path for path in files if path.startswith("nodes/")}
        if {name for name in result if name.startswith("aliases/")} != alias_names:
            raise ValueError("history alias membership differs from its graph")
        if semantic_digest(data) != entry["semantic_sha256"] or data["observed_at"] != entry["observed_at"]:
            raise ValueError("history semantic identity mismatch")
        for path, content in result.items():
            if path.endswith(".html"):
                validate_static(content, entry["id"], path, data)
        from current_site_build import ensure_render_budget
        ensure_render_budget(data, meta)
        expected = render_static_v1(data, meta, archived=True)
        expected.update({"aliases/" + path: content for path, content in render_static_v1(data, meta).items()})
    # Exact repository-owned frozen renderer admission rejects active markup,
    # forged evidence, unbound paths, omitted locators and altered visible text.
    for path, content in expected.items():
        if result.get(path) != canonical_bytes(path, content):
            raise ValueError("historical renderer source/evidence content mismatch: " + path)

    return result, data, meta


def safe_destination(site, name):
    path = site / name
    if path.is_dir() or path.is_symlink() or any(p.is_symlink() or (p.exists() and not p.is_dir())
            for p in path.parents if p != site and p.is_relative_to(site)):
        raise ValueError("unsafe history restore destination")
    return path


def atomic_outputs(site, outputs):
    for name in outputs:
        safe_destination(site, name)
    # Every network read/content admission precedes this staging and any write.
    with tempfile.TemporaryDirectory(prefix="current-site-history-") as tmp:
        stage = Path(tmp)
        for number, (name, raw) in enumerate(outputs.items()):
            (stage / str(number)).write_bytes(raw)
        for number, (name, raw) in enumerate(outputs.items()):
            destination = site / name
            if destination.exists() and destination.read_bytes() == raw:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".history-", delete=False) as stream:
                    temporary = Path(stream.name)
                    with (stage / str(number)).open("rb") as source:
                        while chunk := source.read(1024 * 1024):
                            stream.write(chunk)
                    stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, destination); temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def local_manifest(site):
    path = site / "snapshots/index.json"
    if path.is_symlink():
        raise ValueError("unsafe local history manifest")
    if not path.exists():
        return None
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("unsafe local history manifest")
    return validate_manifest(parse_json(path.read_bytes()))


def local_bundles(site, manifest):
    result = {}
    for entry in manifest["archives"]:
        path = safe_destination(site, entry["bundle"]["path"])
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("missing local history bundle")
        members, data, meta = admit_bundle(entry, path.read_bytes())
        for row in entry["files"]:
            page = safe_destination(site, f'snapshots/{entry["id"]}/{row["path"]}')
            if not page.is_file() or page.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("missing local immutable history file")
            if canonical_bytes(row["path"], page.read_bytes()) != members[row["path"]]:
                raise ValueError("local immutable history content conflict")
        expected = {row["path"] for row in entry["files"]} | {"archive.zip"}
        directory = site / "snapshots" / entry["id"]
        actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
        if expected != actual or any(p.is_symlink() for p in directory.rglob("*")):
            raise ValueError("local immutable history membership conflict")
        result[entry["id"]] = (members, data, meta)
    return result


def archive_record(snapshot, files):
    data, meta = payload_record(snapshot, files)
    version = meta.get("archive_version", 1)
    aliases = render_static_v1(data, meta) if version == 1 else {}
    members = {**files, **{"aliases/" + k: canonical_bytes(k, v) for k, v in aliases.items()}}
    raw_zip = deterministic_zip(members)
    entry = {"id": snapshot, "semantic_sha256": mathematical_digest(data) if version == 2 else semantic_digest(data),
             "observed_at": data["observed_at"], "renderer": STATIC_RENDERER_V2 if version == 2 else STATIC_RENDERER_V1,
             "files": [descriptor(k, v) for k, v in sorted(files.items())],
             "alias_files": [descriptor(k, v) for k, v in sorted(aliases.items())],
             "bundle": descriptor(f"snapshots/{snapshot}/archive.zip", raw_zip)}
    if version == 2:
        entry["archive_version"] = 2
    admitted = admit_bundle(entry, raw_zip)
    return entry, admitted, raw_zip


def archive_inventory(site):
    previous = local_manifest(site)
    known = local_bundles(site, previous) if previous else {}
    entries = {e["id"]: e for e in previous["archives"]} if previous else {}
    bundles = {}
    root = site / "snapshots"
    if not root.exists():
        return entries, known, bundles
    if root.is_symlink() or not root.is_dir():
        raise ValueError("unsafe immutable snapshot directory")
    for directory in sorted(root.iterdir()):
        if directory.name == "index.json":
            continue
        if directory.is_symlink() or not directory.is_dir() or not HEX.fullmatch(directory.name):
            raise ValueError("unknown local history entry")
        if directory.name in entries:
            continue
        if any(p.is_symlink() for p in directory.rglob("*")):
            raise ValueError("symlinked snapshot publication refused")
        files = {}
        for path in directory.rglob("*"):
            if path.is_file() and path.relative_to(directory).as_posix() != "archive.zip":
                if path.stat().st_size > MAX_FILE_BYTES:
                    raise ValueError("snapshot file capacity exceeded")
                files[path.relative_to(directory).as_posix()] = canonical_bytes(path.name, path.read_bytes())
        entry, admitted, raw_zip = archive_record(directory.name, files)
        existing_zip = directory / "archive.zip"
        if existing_zip.exists() and existing_zip.read_bytes() != raw_zip:
            raise ValueError("unregistered immutable snapshot bundle conflict")
        entries[directory.name] = entry
        known[directory.name] = admitted
        bundles[entry["bundle"]["path"]] = raw_zip
    return entries, known, bundles


def choose_archive(site, data):
    entries, admitted, _ = archive_inventory(site)
    wanted = mathematical_digest(data)
    for entry in sorted(entries.values(), key=lambda e: (e["observed_at"], e["id"])):
        members, old, meta = admitted[entry["id"]]
        if mathematical_digest(old) == wanted:
            return {"id": entry["id"], "data": old, "meta": meta, "members": members}
    return None


def historical_aliases(known):
    result = {}
    # Retain the most recent admitted representation for each disappeared route.
    for snapshot, (members, data, meta) in sorted(known.items(), key=lambda row: (row[1][1]["observed_at"], row[0])):
        pages = ({n[len("aliases/"):]: b for n, b in members.items() if n.startswith("aliases/")}
                 if meta.get("archive_version", 1) == 1 else render_static_v2(data, meta, archived=False))
        for path, raw in pages.items():
            result[path] = (snapshot, canonical_bytes(path, raw))
    return result


def manifest_plan(site, data, meta, raw_payload, live_pages, new_archive=None):
    entries, known, bundles = archive_inventory(site)
    if new_archive is not None and meta["archive"] not in entries:
        entry, admitted, raw_zip = archive_record(meta["archive"], new_archive)
        entries[entry["id"]] = entry; known[entry["id"]] = admitted
        bundles[entry["bundle"]["path"]] = raw_zip
    if meta["archive"] not in entries or mathematical_digest(known[meta["archive"]][1]) != mathematical_digest(data):
        raise ValueError("current observation has no equivalent retained mathematical archive")
    historical = historical_aliases(known)
    live = {name: canonical_bytes(name, content) for name, content in live_pages.items()}
    pages = {name: raw for name, (_, raw) in historical.items()}
    pages.update(live)
    aliases = []
    for path, raw in sorted(pages.items()):
        aliases.append({**descriptor(path, raw), "snapshot": meta["archive"] if path in live else historical[path][0],
                        "binding": "current_observation" if path in live else "archive"})
    observation = {"payload": descriptor("current.json", raw_payload),
                   "files": [descriptor(k, v) for k, v in sorted(live.items())]}
    ordered = sorted(entries.values(), key=lambda e: (e["observed_at"], e["id"]))
    total = history_size(ordered, aliases, observation)
    manifest = {"schema_version": 2, "origin": ORIGIN, "scientific_effect": "NONE",
                "representation": "asset-release-normalized-html-v1", "current": meta["archive"],
                "current_observation": observation, "archives": ordered, "aliases": aliases,
                "total_bytes": total, "capacity": capacity_record(ordered, total)}
    validate_manifest(manifest)
    if len(manifest_bytes(manifest)) > MAX_FILE_BYTES:
        raise ValueError("history manifest capacity exceeded")
    return manifest, pages, bundles


def preflight_capacity(site, new_files, aliases):
    # Compatibility helper; new builds use the exact complete manifest plan.
    manifest = local_manifest(site)
    old = manifest["total_bytes"] if manifest else 0
    if manifest and len(manifest["archives"]) >= MAX_SNAPSHOTS:
        raise ValueError("history snapshot capacity exceeded; prior publication retained")
    added = sum(len(v) + 256 for v in new_files.values()) * 2 + sum(len(v) + 256 for v in aliases.values())
    if any(len(v) > MAX_FILE_BYTES for v in (*new_files.values(), *aliases.values())) or old + added > MAX_HISTORY_BYTES:
        raise ValueError("history capacity exceeded; prior publication retained")


def publish_history(site):
    from current_site_build import validate_payload, ensure_render_budget
    site = Path(site).resolve()
    raw_payload = (site / "current.json").read_bytes()
    data = validate_payload(parse_json(raw_payload))
    meta = parse_json((site / "current-meta.json").read_bytes())
    if not isinstance(meta, dict) or set(meta) not in ({"sha256", "bytes", "observed_at"},
            {"sha256", "bytes", "observed_at", "archive"},
            {"sha256", "bytes", "observed_at", "archive", "history_manifest_sha256"}):
        raise ValueError("invalid closed local publication metadata schema")
    if "history_manifest_sha256" in meta:
        sha(meta["history_manifest_sha256"])
    current = meta.get("archive", meta.get("sha256")); sha(current)
    if {k: meta.get(k) for k in ("sha256", "bytes", "observed_at")} != {"sha256": digest(raw_payload), "bytes": len(raw_payload), "observed_at": data["observed_at"]}:
        raise ValueError("live payload metadata mismatch")
    meta["archive"] = current
    ensure_render_budget(data, meta)
    live = render_live_current_v2(data, meta)
    manifest, pages, bundles = manifest_plan(site, data, meta, raw_payload, live)
    actual = {"nodes/" + p.name: p for p in (site / "nodes").glob("*.html")}
    if set(actual) != set(pages):
        raise ValueError("convenience route membership differs from retained history")
    for name, raw in pages.items():
        path = safe_destination(site, name)
        if path.stat().st_size > MAX_FILE_BYTES or canonical_bytes(name, path.read_bytes()) != raw:
            raise ValueError("convenience route is not bound to its exact live observation or historical archive")
    raw = manifest_bytes(manifest)
    outputs = {**bundles, "snapshots/index.json": raw}
    meta["history_manifest_sha256"] = digest(raw)
    outputs["current-meta.json"] = manifest_bytes(meta)
    atomic_outputs(site, outputs)
    return manifest


def remote_metadata(reader, *, missing_allowed=False):
    try:
        raw = reader.get_bytes("current-meta.json")
    except urllib.error.HTTPError as error:
        if error.code == 404 and missing_allowed:
            return None
        raise ValueError("published history metadata unavailable") from error
    meta = parse_json(raw)
    if not isinstance(meta, dict):
        raise ValueError("malformed published history metadata")
    if "archive" in meta or "history_manifest_sha256" in meta:
        closed(meta, ("sha256", "bytes", "observed_at", "archive", "history_manifest_sha256"), "published metadata")
        sha(meta["archive"]); sha(meta["history_manifest_sha256"])
    else:
        closed(meta, ("sha256", "bytes", "observed_at"), "legacy published metadata")
    sha(meta["sha256"]); timestamp(meta["observed_at"])
    if type(meta["bytes"]) is not int or not 0 <= meta["bytes"] <= MAX_FILE_BYTES:
        raise ValueError("invalid published payload byte count")
    return meta


def restore_history(site, reader=None, expected_manifest_sha=None):
    site = Path(site).resolve()
    if expected_manifest_sha is not None:
        sha(expected_manifest_sha)
    reader = reader or HistoryReader()
    ledger_path = site / LEDGER
    ledger = None
    if ledger_path.is_symlink():
        raise ValueError("unsafe restored-history ledger")
    if ledger_path.exists():
        if ledger_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("unsafe restored-history ledger")
        ledger = parse_json(ledger_path.read_bytes())
        closed(ledger, ("schema_version", "manifest_sha256", "snapshots", "scientific_effect"), "restored history ledger")
        if (type(ledger["schema_version"]) is not int or ledger["schema_version"] != 1 or ledger["scientific_effect"] != "NONE"
                or not isinstance(ledger["snapshots"], list) or len(ledger["snapshots"]) > MAX_SNAPSHOTS
                or len(set(ledger["snapshots"])) != len(ledger["snapshots"])):
            raise ValueError("invalid restored-history ledger")
        sha(ledger["manifest_sha256"])
        for snapshot in ledger["snapshots"]: sha(snapshot)
    try:
        raw_manifest = reader.get_bytes("snapshots/index.json")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise ValueError("published history manifest unavailable") from error
        if expected_manifest_sha is not None:
            raise ValueError("published manifest pin is unavailable; bootstrap refused") from error
        remote = remote_metadata(reader, missing_allowed=True)
        if ledger is not None or (remote is not None and ("archive" in remote or "history_manifest_sha256" in remote)):
            raise ValueError("previous published history pointer exists; bootstrap refused")
        # Local reviewed candidate archives are preserved, not interpreted as a
        # previous publication. Only authoritative remote legacy absence permits bootstrap.
        return {"bootstrap": True, "snapshots": 0, "scientific_effect": "NONE"}
    if expected_manifest_sha is not None and digest(raw_manifest) != expected_manifest_sha:
        raise ValueError("published manifest does not match the exact durable receipt pin")
    manifest = validate_manifest(parse_json(raw_manifest))
    remote = remote_metadata(reader)
    if (remote.get("archive") != manifest["current"]
            or remote.get("history_manifest_sha256") != digest(raw_manifest)):
        raise ValueError("published history pointers disagree; possible stale response")
    remote_ids = {e["id"] for e in manifest["archives"]}
    if ledger and not set(ledger["snapshots"]) <= remote_ids:
        raise ValueError("published history regressed below known restored snapshots")
    outputs = {}
    members_by_id = {}
    data_by_id = {}
    for entry in manifest["archives"]:
        raw = reader.get_bytes(entry["bundle"]["path"])
        members, data, meta = admit_bundle(entry, raw)
        members_by_id[entry["id"]] = members
        data_by_id[entry["id"]] = data
        for name, content in members.items():
            if not name.startswith("aliases/"):
                outputs[f'snapshots/{entry["id"]}/{name}'] = content
        outputs[entry["bundle"]["path"]] = raw
    if manifest["schema_version"] == 1:
        required = {"nodes/index.html", *(f"nodes/{n}.html" for n in data_by_id[manifest["current"]]["graph"]["data"]["nodes"])}
        for name in required:
            if {"path": name, "snapshot": manifest["current"]} not in manifest["aliases"]:
                raise ValueError("published current alias binding missing")
        for alias in manifest["aliases"]:
            outputs[alias["path"]] = members_by_id[alias["snapshot"]]["aliases/" + alias["path"]]
    else:
        from current_site_build import validate_payload, ensure_render_budget
        raw_current = reader.get_bytes("current.json")
        row = manifest["current_observation"]["payload"]
        if descriptor("current.json", raw_current) != row or remote["sha256"] != row["sha256"] or remote["bytes"] != row["bytes"]:
            raise ValueError("published current observation bytes disagree")
        data = validate_payload(parse_json(raw_current))
        if data["observed_at"] != remote["observed_at"] or mathematical_digest(data) != mathematical_digest(data_by_id[manifest["current"]]):
            raise ValueError("published current observation has different mathematical material")
        ensure_render_budget(data, remote)
        live = render_live_current_v2(data, remote)
        if [descriptor(k, v) for k, v in sorted(live.items())] != manifest["current_observation"]["files"]:
            raise ValueError("live static reading does not match the exact observation")
        historical = {}
        for entry in manifest["archives"]:
            snapshot = entry["id"]
            members = members_by_id[snapshot]
            archived_meta = parse_json(members["current-meta.json"])
            historical[snapshot] = ({n[len("aliases/"):]: b for n, b in members.items() if n.startswith("aliases/")}
                                    if entry.get("archive_version", 1) == 1 else render_static_v2(data_by_id[snapshot], archived_meta, archived=False))
        for alias in manifest["aliases"]:
            content = live.get(alias["path"]) if alias["binding"] == "current_observation" else historical[alias["snapshot"]].get(alias["path"])
            if content is None or descriptor(alias["path"], content) != {k: alias[k] for k in ("path", "bytes", "sha256")}:
                raise ValueError("published alias differs from its admitted observation or archive")
            outputs[alias["path"]] = content
        outputs["current.json"] = raw_current
        outputs["current-meta.json"] = manifest_bytes(remote)
    # Immutable local candidates may add unpublished history, but conflicting
    # bytes for a shared identity are refused. Nothing is silently overwritten.
    for name, content in outputs.items():
        path = safe_destination(site, name)
        if name.startswith("snapshots/") and path.exists() and canonical_bytes(name, path.read_bytes()) != content:
            raise ValueError("local and published immutable history conflict")
    local_entries, local_known, local_zip = archive_inventory(site)
    extras = [e for e in local_entries.values() if e["id"] not in remote_ids]
    if extras:
        combined = sorted(manifest["archives"] + extras, key=lambda e: (e["observed_at"], e["id"]))
        paths = {a["path"] for a in manifest["aliases"]}
        selected = historical_aliases({k: v for k, v in local_known.items() if k not in remote_ids})
        if manifest["schema_version"] == 2:
            additional = []
            for path, (snapshot, raw) in sorted(selected.items()):
                if path not in paths:
                    additional.append({**descriptor(path, raw), "snapshot": snapshot, "binding": "archive"})
                    outputs[path] = raw
            aliases = sorted(manifest["aliases"] + additional, key=lambda a: a["path"])
            total = history_size(combined, aliases, manifest["current_observation"])
            manifest = {**manifest, "archives": combined, "aliases": aliases,
                        "total_bytes": total, "capacity": capacity_record(combined, total)}
        else:
            # Reviewed v2 candidates are retained as admitted directory inputs
            # during a v1 public restore. Do not pretend the older origin served
            # a v2 current observation. The immediately following offline build
            # enumerates every candidate directory into the explicit v2 registry.
            if any(e.get("archive_version", 1) == 2 for e in extras):
                prospective = []
                for alias in manifest["aliases"]:
                    content = outputs[alias["path"]]
                    prospective.append({**descriptor(alias["path"], content), "snapshot": alias["snapshot"], "binding": "archive"})
                for path, (snapshot, raw) in selected.items():
                    if path not in paths:
                        outputs[path] = raw
                        prospective.append({**descriptor(path, raw), "snapshot": snapshot, "binding": "archive"})
                if len(combined) > MAX_SNAPSHOTS or history_size(combined, prospective) + remote["bytes"] > MAX_HISTORY_BYTES:
                    raise ValueError("combined candidate history capacity exceeded")
            else:
                additional = [{"path": path, "snapshot": snapshot} for path, (snapshot, raw) in selected.items() if path not in paths]
                for alias in additional:
                    outputs[alias["path"]] = selected[alias["path"]][1]
                aliases = sorted(manifest["aliases"] + additional, key=lambda a: a["path"])
                manifest = {**manifest, "archives": combined, "aliases": aliases, "total_bytes": history_size(combined, aliases)}
        validate_manifest(manifest)
        outputs.update(local_zip)
    outputs["snapshots/index.json"] = manifest_bytes(manifest)
    outputs[LEDGER] = manifest_bytes({"schema_version": 1, "manifest_sha256": digest(raw_manifest),
                                     "snapshots": sorted(remote_ids), "scientific_effect": "NONE"})
    atomic_outputs(site, outputs)
    return {"bootstrap": False, "snapshots": len(remote_ids), "scientific_effect": "NONE"}


# Frozen admission renderer. Keep this implementation stable: a changed public
# renderer requires a new explicit version and a new admission implementation.
# No downloaded code or current_site_build rendering functions run here.
from urllib.parse import quote as _v1_quote
import html as _v1_html

STATIC_RENDERER_V1 = "research-static-v1"
_V1_MATH = "d6g8k5htny-coder/Math-"
_V1_GENERATED = "<!-- CURRENT-SITE:GENERATED-NODE -->"


def _v1_escape(value):
    return _v1_html.escape(str(value), quote=True)


def _v1_source_url(source):
    return (f"https://github.com/{source['repository']}/blob/{source['commit']}/"
            + _v1_quote(source["path"], safe="/"))


def _v1_source_list(sources):
    return "\n".join(f'<li><a href="{_v1_escape(_v1_source_url(source))}">{_v1_escape(source["repository"].split("/")[1])} · {_v1_escape(source["path"])}</a></li>'
                     for source in sources)


def _v1_current_section(data, meta, *, prefix="", search=True, snapshot_prefix=None, archived=False):
    snapshot_prefix = snapshot_prefix if snapshot_prefix is not None else f'{prefix}snapshots/{meta.get("archive", meta["sha256"])}/'
    node_index = snapshot_prefix + "nodes/index.html"
    pins = " · ".join(f'<a href="https://github.com/{_v1_escape(repo["repository"])}/tree/{repo["commit"]}">{_v1_escape(repo["repository"].split("/")[1])} <code>{repo["commit"]}</code></a>'
                      for repo in data["repositories"])
    refs = " · ".join(f'<a href="{_v1_escape(_v1_source_url(ref))}">{_v1_escape(ref.get("label", ref["path"]))}</a>'
                      for ref in data["references"])
    controls = ""
    if search:
        controls = ('<details><summary>Search the current public source inventory</summary>\n'
                    '<label>Source path or repository <input class="current-search" type="search" disabled placeholder="Search after snapshot verification"></label>\n'
                    f'<p class="current-count">{min(8, len(data["sources"]))} of {len(data["sources"])} indexed public files shown. Historical, recovery and operational paths are outside this discovery population.</p>\n'
                    '<ul class="current-results">\n' + _v1_source_list(data["sources"][:8]) + '\n</ul>\n'
                    '<button class="current-more" type="button" hidden>Show more files</button>\n'
                    '<noscript><p>Use the <a href="' + node_index + '#sources">complete static source inventory</a>.</p></noscript>\n</details>')
    return (f'<section id="current-research" class="current-research" data-current-digest="{meta["sha256"]}" data-current-bytes="{meta["bytes"]}">\n'
            + ('<h2>Snapshot research sources</h2>\n' if archived else '<h2>Current research sources</h2>\n')
            + f'<p class="current-state" role="status">Pinned source observation: <time datetime="{data["observed_at"]}">{data["observed_at"]}</time>. Compare this timestamp with the present time; observation is not scientific revalidation.</p>\n'
            f'<p>{pins}</p>\n<p>{refs}</p>\n'
            f'<p><a href="{prefix}dependencies.html#current-atlas">Current mathematical atlas</a> · <a href="{node_index}">Snapshot static graph and complete source inventory</a> · <a href="{snapshot_prefix}current.json">Exact snapshot observation payload</a>. Scientific effect: NONE.</p>\n'
            + controls + '\n</section>')


def _v1_axes_html(node):
    rows = []
    for axis in ("source", "review", "kernel", "computation", "alignment"):
        recorded = (axis == "source" and bool(node.get("source"))) or (
            axis == "review" and any(key.startswith("review_") and bool(value) for key, value in node.items()))
        state = "recorded" if recorded else "not_recorded"
        rows.append(f'<dt>{axis}</dt><dd data-axis="{axis}" data-state="{state}"><code>{state}</code></dd>')
    return '<dl class="atlas-axes">' + "\n".join(rows) + "</dl>"


def _v1_external_review_url(value):
    if not isinstance(value, str) or any(ord(char) < 32 for char in value):
        return None
    try:
        url = urlsplit(value)
    except ValueError:
        return None
    if (url.scheme != "https" or url.netloc != "github.com" or url.query
            or not re.fullmatch(r"/d6g8k5htny-coder/(?:main|Math-)/(?:issues|pull)/[1-9][0-9]*", url.path)
            or (url.fragment and not re.fullmatch(r"[A-Za-z0-9_-]+", url.fragment))):
        return None
    return value


def _v1_review_urls(node):
    found = []
    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        else:
            url = _v1_external_review_url(value)
            if url and url not in found:
                found.append(url)
    for key, value in node.items():
        if key.startswith("review_"):
            visit(value)
    return found


def _v1_node_sources(data, node):
    inventory = {(source["repository"], source["commit"], source["path"]): source for source in data["sources"]}
    commit = data["graph"]["identity"]["commit"]
    lines = []
    for field, label in (("source", "Source"), ("review_source", "Review report")):
        value = node.get(field)
        if not value:
            lines.append(f'<p>{label} metadata: <code>not_recorded</code>.</p>')
            continue
        if field == "review_source" and _v1_external_review_url(value):
            continue
        source = inventory.get((_V1_MATH, commit, value)) if isinstance(value, str) else None
        if source:
            lines.append(f'<p>{label}: <a href="{_v1_escape(_v1_source_url(source))}">{_v1_escape(value)}</a> (complete document at the pinned source revision; no document text was acquired into this page).</p>')
        else:
            lines.append(f'<p>{label} unavailable: <code>{_v1_escape(value)}</code>. No exact file is present in this snapshot’s Math- source inventory; no target inferred.</p>')
    for url in _v1_review_urls(node):
        lines.append(f'<p>Recorded external review reference (mutable): <a href="{_v1_escape(url)}">{_v1_escape(url)}</a>. This external record is separate from admitted document bytes.</p>')
    return "\n".join(lines)


def _v1_neighbors(data, node_id, prefix=""):
    groups = {(direction, required): [] for direction in ("Outgoing", "Incoming") for required in (True, False)}
    for edge in data["graph"]["data"]["edges"]:
        if edge["from"] == node_id:
            groups[("Outgoing", edge["required"])].append((edge["to"], edge))
        if edge["to"] == node_id:
            groups[("Incoming", edge["required"])].append((edge["from"], edge))
    parts = ['<h2>Typed neighbors</h2><p>Directions and relation labels are preserved from the selected graph snapshot.</p>']
    for (direction, required), rows in groups.items():
        edge_type = "required" if required else "context"
        parts.append(f'<h3>{direction} {edge_type}</h3>')
        if not rows:
            parts.append('<p>None recorded.</p>')
            continue
        parts.append('<ul>')
        for other, edge in rows:
            parts.append(f'<li><a href="{prefix}{_v1_escape(other)}.html">{_v1_escape(other)}</a> · <code>{_v1_escape(edge["relation"])}</code> · <code>{_v1_escape(edge["from"])} → {_v1_escape(edge["to"])}</code> · required: <code>{str(required).lower()}</code></li>')
        parts.append('</ul>')
    return "\n".join(parts)


def _v1_static_page(data, meta, title, body, *, archived=False, node_id=None):
    site_prefix = "../../../" if archived else "../"
    snapshot_prefix = "../" if archived else f'../snapshots/{meta["sha256"]}/'
    record = f"{node_id}.html" if node_id is not None else "index.html"
    disclosure = (f'<p>Immutable snapshot record · payload SHA-256 <code>{meta["sha256"]}</code>.</p>'
                  if archived else f'<p>This convenience route preserves its recorded observation date. Use the <a href="{snapshot_prefix}nodes/{_v1_escape(record)}">immutable snapshot record</a> to retain this exact node and its neighbors across later changes.</p>')
    return (f'<!doctype html>\n{_V1_GENERATED}\n<html lang="en"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'self\'; img-src \'self\'; script-src \'none\'; base-uri \'none\'; form-action \'none\'">\n'
            f'<title>{_v1_escape(title)} · Research dependency atlas</title>\n'
            f'<link rel="stylesheet" href="{site_prefix}brand.css"><link rel="stylesheet" href="{site_prefix}style.css"><link rel="stylesheet" href="{site_prefix}current.css">\n'
            '</head><body><main class="node-static">\n'
            f'<nav aria-label="Research site"><a href="{site_prefix}index.html">Home</a> · <a href="{site_prefix}explore.html">Explore</a> · <a href="{site_prefix}research.html">Research</a> · <a href="{site_prefix}workspace.html#inventory">Library</a> · <a href="{site_prefix}dependencies.html#current-atlas">Current dependencies</a> · <a href="index.html">Static graph index</a></nav>\n'
            + _v1_current_section(data, meta, prefix=site_prefix, search=False,
                                  snapshot_prefix=snapshot_prefix, archived=True)
            + '\n' + disclosure
            + '\n' + body + '\n</main></body></html>\n').encode("utf-8")


def render_static_v1(data, meta, *, archived=False):
    """Render exactly research-static-v1 from admitted payload and archive metadata."""
    graph = data["graph"]["data"]
    node_pages = {}
    for node_id, node in graph["nodes"].items():
        body = (f'<h1>{_v1_escape(node_id)}</h1>\n<p>Original kind: <code>{_v1_escape(node.get("kind", "not_recorded"))}</code> · recorded classification: <code>{_v1_escape(node.get("classification", "not_recorded"))}</code>.</p>\n'
                '<h2>Independent evidence metadata axes</h2>\n' + _v1_axes_html(node)
                + '\n<p>These values describe metadata presence, not a proof result, applicability or scientific acceptance. No kernel, computation or alignment record is admitted by this graph export.</p>\n'
                '<p>Graph-to-artifact binding: unresolved; artifact roles: unknown; custody: unknown. No Artifact Index mapping is inferred.</p>\n'
                '<h2>Source and recorded review references</h2>\n' + _v1_node_sources(data, node)
                + '\n<h2>Original node metadata</h2><pre>'
                + _v1_escape(json.dumps(node, ensure_ascii=False, indent=2, sort_keys=True)) + '</pre>\n'
                + _v1_neighbors(data, node_id, prefix="" if archived else f'../snapshots/{meta["sha256"]}/nodes/'))
        node_pages[f"nodes/{node_id}.html"] = _v1_static_page(data, meta, node_id, body, archived=archived, node_id=node_id)
    items = "\n".join(f'<li><a href="{_v1_escape(node_id)}.html">{_v1_escape(node_id)}</a> · {_v1_escape(node.get("kind", "not_recorded"))} · <code>{_v1_escape(node.get("classification", "not_recorded"))}</code></li>'
                      for node_id, node in graph["nodes"].items())
    required = sum(edge["required"] for edge in graph["edges"])
    body = (f'<h1>Static mathematical dependency atlas</h1>\n<p>{len(graph["nodes"])} graph nodes · {len(graph["edges"])} edges · {required} required edges · {len(graph["edges"])-required} context edges. These are graph inventory counts.</p>\n'
            f'<p>Graph source: <a href="{_v1_escape(_v1_source_url(data["graph"]["identity"]))}">{_v1_escape(data["graph"]["identity"]["path"])}</a> · SHA-256 <code>{data["graph"]["identity"]["sha256"]}</code>. The upstream graph may be unchanged while its repository advances.</p>\n'
            '<p>Every node has static source/evidence reading and required/context neighbor navigation. Graph-to-artifact binding is unresolved; artifact roles and custody are unknown. No classification is promoted here.</p>\n'
            '<ul>\n' + items + '\n</ul>\n<h2 id="sources">Complete current public source inventory</h2>\n'
            f'<p>{len(data["sources"])} indexed files at the pinned repository commits. Legacy, quarantine, recovery, scratch and operational paths are excluded from this discovery population.</p>\n<ul>\n'
            + _v1_source_list(data["sources"]) + '\n</ul>')
    node_pages["nodes/index.html"] = _v1_static_page(data, meta, "Static graph and source index", body, archived=archived)
    return node_pages


# Frozen compact mathematical-material renderer. Keep research-static-v1 above
# unchanged; future rendering changes require a separately admitted version.
STATIC_RENDERER_V2 = "research-static-v2"


def _v2_material_section(projection, meta, *, site_prefix, snapshot_prefix):
    pins = " · ".join(f'<a href="https://github.com/{_v1_escape(repo["repository"])}/tree/{repo["commit"]}">{_v1_escape(repo["repository"].split("/")[1])} <code>{repo["commit"]}</code></a>'
                      for repo in projection["repositories"])
    refs = " · ".join(f'<a href="{_v1_escape(_v1_source_url(ref))}">{_v1_escape(ref.get("label", ref["path"]))}</a> (<code>{_v1_escape(ref["path"])}</code> · Git blob <code>{_v1_escape(ref["blob"])}</code> · {_v1_escape(ref["bytes"])} bytes · SHA-256 <code>{_v1_escape(ref["sha256"])}</code>)'
                      for ref in projection["references"])
    return (f'<section id="current-research" class="current-research" data-current-digest="{_v1_escape(meta["sha256"])}" data-current-bytes="{_v1_escape(meta["bytes"])}">\n'
            '<h2>Historical mathematical material</h2>\n'
            f'<p class="current-state" role="status">Original source observation: <time datetime="{_v1_escape(projection["observed_at"])}">{_v1_escape(projection["observed_at"])}</time>. Source commits are the original observation pins. Observation is not scientific revalidation.</p>\n'
            f'<p>{pins}</p>\n<p>Original archived source and index references (original byte identities): {refs}</p>\n'
            '<p>Coverage: <code>mathematical-material-only</code>. This compact projection retains the recorded graph, equation excerpts and their selected source and review file identities.</p>\n'
            '<p>Versions record source material and archived index context. A changed version is not a new mathematical result, proof or scientific acceptance.</p>\n'
            f'<p><a href="{_v1_escape(site_prefix)}dependencies.html#current-atlas">Current mathematical atlas</a> · <a href="{_v1_escape(snapshot_prefix)}nodes/index.html">Historical graph, equation excerpts and selected sources</a> · <a href="{_v1_escape(snapshot_prefix)}current.json">Exact historical mathematical projection</a>. Scientific effect: NONE.</p>\n'
            '</section>')


def _v2_node_sources(projection, node):
    inventory = {(source["repository"], source["commit"], source["path"]): source for source in projection["sources"]}
    commit = projection["graph"]["identity"]["commit"]
    lines = []
    for field, label in (("source", "Source"), ("review_source", "Review report")):
        value = node.get(field)
        if not value:
            lines.append(f'<p>{label} metadata: <code>not_recorded</code>.</p>')
            continue
        if field == "review_source" and _v1_external_review_url(value):
            continue
        source = inventory.get((_V1_MATH, commit, value)) if isinstance(value, str) else None
        if source:
            lines.append(f'<p>{label}: <a href="{_v1_escape(_v1_source_url(source))}">{_v1_escape(value)}</a> (complete document at its original pinned source revision; no document text was acquired into this page).</p>')
        else:
            lines.append(f'<p>{label} unavailable: <code>{_v1_escape(value)}</code>. No exact file is present in this historical mathematical source projection; no target inferred.</p>')
    for url in _v1_review_urls(node):
        lines.append(f'<p>Recorded external review reference (mutable): <a href="{_v1_escape(url)}">{_v1_escape(url)}</a>. This external record is separate from admitted document bytes.</p>')
    return "\n".join(lines)


def _v2_static_page(projection, meta, title, body, *, archived, node_id=None):
    site_prefix = "../../../" if archived else "../"
    archive_id = meta["mathematical_sha256"]
    snapshot_prefix = "../" if archived else f"../snapshots/{archive_id}/"
    record = f"{node_id}.html" if node_id is not None else "index.html"
    graph_index = "index.html" if archived else f"{snapshot_prefix}nodes/index.html"
    disclosure = (f'<p>Immutable historical mathematical record · mathematical SHA-256 <code>{_v1_escape(archive_id)}</code> · exact projection SHA-256 <code>{_v1_escape(meta["sha256"])}</code>.</p>'
                  if archived else
                  '<p>Historical removed-node static route. This convenience route retains the original node metadata, source pins and observation date from its selected mathematical archive. '
                  f'Use the <a href="{_v1_escape(snapshot_prefix)}nodes/{_v1_escape(record)}">immutable historical mathematical record</a> for this exact node and its archived neighbors.</p>')
    return (f'<!doctype html>\n{_V1_GENERATED}\n<html lang="en"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'self\'; img-src \'self\'; script-src \'none\'; base-uri \'none\'; form-action \'none\'">\n'
            f'<title>{_v1_escape(title)} · Historical mathematical dependency atlas</title>\n'
            f'<link rel="stylesheet" href="{site_prefix}brand.css"><link rel="stylesheet" href="{site_prefix}style.css"><link rel="stylesheet" href="{site_prefix}current.css">\n'
            '</head><body><main class="node-static">\n'
            f'<nav aria-label="Research site"><a href="{site_prefix}index.html">Home</a> · <a href="{site_prefix}explore.html">Explore</a> · <a href="{site_prefix}research.html">Research</a> · <a href="{site_prefix}workspace.html#inventory">Library</a> · <a href="{site_prefix}dependencies.html#current-atlas">Current dependencies</a> · <a href="{_v1_escape(graph_index)}">Historical static graph index</a></nav>\n'
            + _v2_material_section(projection, meta, site_prefix=site_prefix, snapshot_prefix=snapshot_prefix)
            + '\n' + disclosure
            + '\n' + body + '\n</main></body></html>\n').encode("utf-8")


def _v2_equation_collection(projection):
    equations = projection["equations"]
    inventory = {(source["repository"], source["commit"], source["path"]): source for source in projection["sources"]}
    parts = ['<h2 id="equations">Recorded equation excerpts</h2>\n'
             f'<p>{len(equations)} exact source excerpts are retained as a separate collection. No equation-to-graph dependency mapping is inferred.</p>']
    for number, equation in enumerate(equations, 1):
        first, last = equation["line_start"], equation["line_end"]
        fragment = f"#L{first}" if first == last else f"#L{first}-L{last}"
        target = _v1_source_url(equation) + fragment
        source = inventory.get((equation["repository"], equation["commit"], equation["path"]))
        source_identity = (f'<p>Recorded source file identity: Git blob <code>{_v1_escape(source["blob"])}</code> · {_v1_escape(source["bytes"])} bytes.</p>\n'
                           if source is not None else '<p>Source file identity: <code>not_recorded</code>.</p>\n')
        parts.append(f'<section id="equation-{number}"><h3>Recorded equation excerpt {number}</h3>\n'
                     f'<p>Original excerpt identity: <code>{_v1_escape(equation["id"])}</code>.</p>\n'
                     f'<p>Source: <a href="{_v1_escape(target)}">{_v1_escape(equation["repository"])} · {_v1_escape(equation["path"])} · lines {_v1_escape(first)}–{_v1_escape(last)}</a>.</p>\n'
                     + source_identity
                     + f'<p>Exact excerpt SHA-256: <code>{_v1_escape(equation["sha256"])}</code>.</p>\n'
                     + '<pre>' + _v1_escape(equation["text"]) + '</pre>\n</section>')
    return "\n".join(parts)


def render_static_v2(projection, meta, *, archived=True):
    """Render frozen research-static-v2 compact mathematical-material pages.

    The archived form is retained in v2 bundles. The convenience form is used
    only for historical removed-node routes; live aliases have their own current
    observation renderer. All historical neighbors select this material archive.
    """
    graph = projection["graph"]["data"]
    archive_id = meta["mathematical_sha256"]
    node_prefix = "" if archived else f"../snapshots/{archive_id}/nodes/"
    pages = {}
    for node_id, node in graph["nodes"].items():
        body = (f'<h1>{_v1_escape(node_id)}</h1>\n<p>Original kind: <code>{_v1_escape(node.get("kind", "not_recorded"))}</code> · recorded classification: <code>{_v1_escape(node.get("classification", "not_recorded"))}</code>.</p>\n'
                '<h2>Independent evidence metadata axes</h2>\n' + _v1_axes_html(node)
                + '\n<p>These values describe metadata presence, not a proof result, applicability or scientific acceptance. No kernel, computation or alignment record is admitted by this graph export.</p>\n'
                '<p>Graph-to-artifact binding: unresolved; artifact roles: unknown; custody: unknown. No Artifact Index mapping is inferred.</p>\n'
                '<h2>Source and recorded review references</h2>\n' + _v2_node_sources(projection, node)
                + '\n<h2>Original node metadata</h2><pre>'
                + _v1_escape(json.dumps(node, ensure_ascii=False, indent=2, sort_keys=True)) + '</pre>\n'
                + _v1_neighbors(projection, node_id, prefix=node_prefix))
        pages[f"nodes/{node_id}.html"] = _v2_static_page(projection, meta, node_id, body, archived=archived, node_id=node_id)
    items = "\n".join(f'<li><a href="{_v1_escape(node_prefix)}{_v1_escape(node_id)}.html">{_v1_escape(node_id)}</a> · {_v1_escape(node.get("kind", "not_recorded"))} · <code>{_v1_escape(node.get("classification", "not_recorded"))}</code></li>'
                      for node_id, node in graph["nodes"].items())
    required = sum(edge["required"] for edge in graph["edges"])
    body = (f'<h1>Historical mathematical dependency atlas</h1>\n<p>{len(graph["nodes"])} graph nodes · {len(graph["edges"])} edges · {required} required edges · {len(graph["edges"])-required} context edges. These are retained graph inventory counts.</p>\n'
            f'<p>Graph source: <a href="{_v1_escape(_v1_source_url(graph_source := projection["graph"]["identity"]))}">{_v1_escape(graph_source["path"])}</a> · SHA-256 <code>{_v1_escape(graph_source["sha256"])}</code>.</p>\n'
            '<p>Every retained node has static source/evidence reading and required/context neighbor navigation at the original observation pins. Graph-to-artifact binding is unresolved; artifact roles and custody are unknown. No classification is promoted here.</p>\n'
            '<ul>\n' + items + '\n</ul>\n<h2 id="sources">Selected recorded mathematical sources</h2>\n'
            f'<p>{len(projection["sources"])} selected graph, node source/review and equation source files are retained in this mathematical-material-only projection.</p>\n<ul>\n'
            + _v1_source_list(projection["sources"]) + '\n</ul>\n'
            + _v2_equation_collection(projection))
    pages["nodes/index.html"] = _v2_static_page(projection, meta, "Historical graph, equation excerpts and selected sources", body, archived=archived)
    return pages


# Repository-owned live renderer. Current routes bind the full fresh observation;
# the separately named historical link retains its own original pins and date.
LIVE_RENDERER_V2 = "research-live-static-v2"


def _live_v2_current_section(data, meta):
    pins = " · ".join(f'<a href="https://github.com/{_v1_escape(repo["repository"])}/tree/{repo["commit"]}">{_v1_escape(repo["repository"].split("/")[1])} <code>{repo["commit"]}</code></a>'
                      for repo in data["repositories"])
    refs = " · ".join(f'<a href="{_v1_escape(_v1_source_url(ref))}">{_v1_escape(ref.get("label", ref["path"]))}</a>'
                      for ref in data["references"])
    archive = _v1_escape(meta["archive"])
    return (f'<section id="current-research" class="current-research" data-current-digest="{_v1_escape(meta["sha256"])}" data-current-bytes="{_v1_escape(meta["bytes"])}">\n'
            '<h2>Current research sources</h2>\n'
            f'<p class="current-state" role="status">Pinned source observation: <time datetime="{_v1_escape(data["observed_at"])}">{_v1_escape(data["observed_at"])}</time>. Compare this timestamp with the present time; observation is not scientific revalidation.</p>\n'
            f'<p>{pins}</p>\n<p>{refs}</p>\n'
            '<p><a href="../dependencies.html#current-atlas">Current mathematical atlas</a> · <a href="index.html">Current static graph and complete source inventory</a> · <a href="../current.json">Exact current source observation payload</a>. Scientific effect: NONE.</p>\n'
            f'<p><a href="../snapshots/{archive}/nodes/index.html">Historical mathematical material (original pins and observation)</a>.</p>\n'
            '</section>')


def _live_v2_static_page(data, meta, title, body):
    return (f'<!doctype html>\n{_V1_GENERATED}\n<html lang="en"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'self\'; img-src \'self\'; script-src \'none\'; base-uri \'none\'; form-action \'none\'">\n'
            f'<title>{_v1_escape(title)} · Current research dependency atlas</title>\n'
            '<link rel="stylesheet" href="../brand.css"><link rel="stylesheet" href="../style.css"><link rel="stylesheet" href="../current.css">\n'
            '</head><body><main class="node-static">\n'
            '<nav aria-label="Research site"><a href="../index.html">Home</a> · <a href="../explore.html">Explore</a> · <a href="../research.html">Research</a> · <a href="../workspace.html#inventory">Library</a> · <a href="../dependencies.html#current-atlas">Current dependencies</a> · <a href="index.html">Current static graph index</a></nav>\n'
            + _live_v2_current_section(data, meta)
            + '\n' + body + '\n</main></body></html>\n').encode("utf-8")


def render_live_current_v2(data, meta):
    """Render current static routes from the full fresh schema-1 observation."""
    graph = data["graph"]["data"]
    pages = {}
    for node_id, node in graph["nodes"].items():
        body = (f'<h1>{_v1_escape(node_id)}</h1>\n<p>Original kind: <code>{_v1_escape(node.get("kind", "not_recorded"))}</code> · recorded classification: <code>{_v1_escape(node.get("classification", "not_recorded"))}</code>.</p>\n'
                '<h2>Independent evidence metadata axes</h2>\n' + _v1_axes_html(node)
                + '\n<p>These values describe metadata presence, not a proof result, applicability or scientific acceptance. No kernel, computation or alignment record is admitted by this graph export.</p>\n'
                '<p>Graph-to-artifact binding: unresolved; artifact roles: unknown; custody: unknown. No Artifact Index mapping is inferred.</p>\n'
                '<h2>Source and recorded review references</h2>\n' + _v1_node_sources(data, node)
                + '\n<h2>Original node metadata</h2><pre>'
                + _v1_escape(json.dumps(node, ensure_ascii=False, indent=2, sort_keys=True)) + '</pre>\n'
                + _v1_neighbors(data, node_id))
        pages[f"nodes/{node_id}.html"] = _live_v2_static_page(data, meta, node_id, body)
    items = "\n".join(f'<li><a href="{_v1_escape(node_id)}.html">{_v1_escape(node_id)}</a> · {_v1_escape(node.get("kind", "not_recorded"))} · <code>{_v1_escape(node.get("classification", "not_recorded"))}</code></li>'
                      for node_id, node in graph["nodes"].items())
    required = sum(edge["required"] for edge in graph["edges"])
    graph_source = data["graph"]["identity"]
    body = (f'<h1>Current static mathematical dependency atlas</h1>\n<p>{len(graph["nodes"])} graph nodes · {len(graph["edges"])} edges · {required} required edges · {len(graph["edges"])-required} context edges. These are current graph inventory counts.</p>\n'
            f'<p>Graph source: <a href="{_v1_escape(_v1_source_url(graph_source))}">{_v1_escape(graph_source["path"])}</a> · SHA-256 <code>{_v1_escape(graph_source["sha256"])}</code>. The upstream graph may be unchanged while its repository advances.</p>\n'
            '<p>Every current node has static source/evidence reading and required/context neighbor navigation at the fresh observation pins. Graph-to-artifact binding is unresolved; artifact roles and custody are unknown. No classification is promoted here.</p>\n'
            '<ul>\n' + items + '\n</ul>\n<h2 id="sources">Complete current public source inventory</h2>\n'
            f'<p>{len(data["sources"])} indexed files at the pinned repository commits. Legacy, quarantine, recovery, scratch and operational paths are excluded from this discovery population.</p>\n<ul>\n'
            + _v1_source_list(data["sources"]) + '\n</ul>\n'
            + _v2_equation_collection(data))
    pages["nodes/index.html"] = _live_v2_static_page(data, meta, "Current static graph, equation excerpts and source index", body)
    return pages


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("restore", "publish"))
    parser.add_argument("--site", type=Path, default=Path(__file__).resolve().parents[1] / "docs/site")
    parser.add_argument("--expected-manifest-sha", help="exact durable publication receipt SHA-256; forbids bootstrap")
    args = parser.parse_args(argv)
    try:
        if args.action != "restore" and args.expected_manifest_sha is not None:
            raise ValueError("manifest pin applies only to restore")
        result = restore_history(args.site, expected_manifest_sha=args.expected_manifest_sha) if args.action == "restore" else publish_history(args.site)
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as error:
        parser.exit(1, f"public history custody refused: {error}\n")
    if result.get("capacity", {}).get("warning"):
        print("::warning::Public mathematical history has reached 75% of its archive or byte capacity; no history is pruned.")
    print(json.dumps({"action": args.action, "snapshots": result.get("snapshots", len(result.get("archives", []))),
                      "bootstrap": result.get("bootstrap", False), "scientific_effect": "NONE"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
