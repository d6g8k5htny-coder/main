"""Pure declared-data checks for the closed node-status v1 fixture contract.

The three byte-input APIs establish consistency with supplied inert originals.
They perform no Git, HTTP, filesystem, process, freshness, owner or proof
acquisition. A supplied revision is a declaration, never authenticated custody.
Current v0 validation, live inputs and actual records are separate and untouched.
"""
from __future__ import annotations

import hashlib as _hashlib
import json as _json
import math as _math
import re as _re
from datetime import datetime as _datetime
from urllib.parse import urlparse as _urlparse

__all__ = ["check_v1_record", "check_v1_population", "project_v1_population"]

_RECORD_LIMIT = 65536
_AGGREGATE_LIMIT = 1048576
_ORIGINAL_LIMIT = 2097152
_ARTIFACT_AGGREGATE_LIMIT = 8388608
_DEPTH_LIMIT = 64
_MATH = "d6g8k5htny-coder/Math-"
_MAIN = "d6g8k5htny-coder/main"
_GRAPH_PATH = "frontiers/downstream_gate_20260925/GRAPH.json"
_PARENT = "math.uniform-matrix-cap-lifetime"
_COMPONENTS = (
    "math.d1-component.congruence-erratum",
    "math.d1-component.section9-replacement-v1_1",
    "math.d1-component.marked-cylinder-cap",
    "math.d1-component.reconciliation-record",
)
_SEEDS = (
    "eng.d0-packet-allowlist-fix", "eng.hard-gate",
    "math.lifetime-remainder", "math.side24-coefficient",
)
_SUPPORT = {
    _PARENT: ("blob", "imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md"),
    _COMPONENTS[0]: ("blob", "imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md"),
    _COMPONENTS[1]: ("blob", "reviews/d1_section9_borel_repair_20260925/REPAIR.md"),
    _COMPONENTS[2]: ("blob", "imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md"),
    _COMPONENTS[3]: ("tree", "reviews/d1_chain_reconciliation_20260928"),
}
_ARTIFACT_PATHS = (
    "cbz/evidence/hard-gate-source-refresh-20261110/README.md",
    "cbz/evidence/hard-gate-source-refresh-20261110/current-SOURCE_FILES.json",
    "cbz/evidence/hard-gate-source-refresh-20261110/observation.json",
)
_ROLE = "engineering-source-identity-evidence"
_SCOPE = "ENGINEERING_OBJECT_IDENTITY_ONLY"
_REQUIRED = frozenset({
    "schema_version", "node_id", "owner", "next_action", "missing_input",
    "stops", "sources", "source_rev", "observed_at", "status",
    "scientific_effect", "graph_source",
})
_OPTIONAL = frozenset({
    "predecessor", "successor", "carried_forward", "refresh_error",
    "depends_on", "dependency_evidence", "artifacts", "artifact_evidence",
})
_FORBIDDEN = frozenset({
    "lemma_closed", "prizes_solved", "discharges_OBL_H5_JETMOD", "certified_C_H",
    "freeze", "inventable_attempt_accepted",
})
_GRAPH_KEYS = frozenset({"repository", "commit", "path", "git_blob", "bytes", "sha256"})
_ARTIFACT_KEYS = frozenset({
    "path", "repository", "commit", "git_mode", "git_blob", "bytes", "sha256",
})
_NODE = _re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SHA1 = _re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = _re.compile(r"[0-9a-f]{64}\Z")
_PATH = _re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]*\Z")
_UTC = _re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_EVIDENCE = _re.compile(
    r"https://github\.com/d6g8k5htny-coder/(main|Math-)/(issues|pull)/[1-9]\d*"
    r"#(issuecomment-[1-9]\d*|pullrequestreview-[1-9]\d*)\Z")
_OBJECT = _re.compile(
    r"https://github\.com/d6g8k5htny-coder/(main|Math-)/(issues|pull)/[1-9]\d*"
    r"(#(issuecomment-[1-9]\d*|pullrequestreview-[1-9]\d*))?\Z")
_PINNED = _re.compile(
    r"https://github\.com/d6g8k5htny-coder/(main|Math-)/(blob|tree)/"
    r"([0-9a-f]{40})/([A-Za-z0-9_./-]+)\Z")
_CARRIED = _re.compile(
    r"https://github\.com/[\w.-]+/[\w.-]+/(?:issues|pull)/\d+"
    r"(?:#(?:issuecomment|pullrequestreview)-\d+)?\Z")


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _text(value, minimum=1, maximum=None):
    return (type(value) is str and len(value) >= minimum
            and (maximum is None or len(value) <= maximum))


def _matches(value, pattern):
    return type(value) is str and pattern.fullmatch(value) is not None


def _closed(value, required, optional=()):
    return (type(value) is dict and all(type(key) is str for key in value)
            and set(required) <= value.keys()
            and value.keys() <= set(required) | set(optional))


def _path(value):
    return (_text(value, maximum=4096) and _matches(value, _PATH)
            and all(part not in ("", ".", "..") for part in value.split("/")))


def _utc(value):
    if not _matches(value, _UTC):
        return False
    try:
        _datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def _json_depth(text, code):
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            _require(depth <= _DEPTH_LIMIT, code)
        elif character in "]}":
            depth -= 1


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "INVALID_JSON")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("INVALID_JSON")

    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        raise ValueError("INVALID_JSON") from None
    _json_depth(text, "INVALID_JSON")
    try:
        value = _json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, RecursionError):
        raise ValueError("INVALID_JSON") from None
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is float:
            _require(_math.isfinite(item), "INVALID_JSON")
        elif type(item) is dict:
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)
    return value


def _identity_shape(value, artifact=False, evidence=False):
    required = _ARTIFACT_KEYS if artifact else _GRAPH_KEYS
    if evidence:
        required = required | {"role", "scope"}
    if not _closed(value, required):
        return False
    if not (type(value["repository"]) is str and _path(value["path"])
            and _matches(value["commit"], _SHA1)
            and _matches(value["git_blob"], _SHA1)
            and _matches(value["sha256"], _SHA256)
            and type(value["bytes"]) is int
            and (0 if artifact else 1) <= value["bytes"] <= _ORIGINAL_LIMIT):
        return False
    if artifact and (type(value["git_mode"]) is not str
                     or value["git_mode"] not in ("100644", "100755")):
        return False
    if evidence and not (_text(value["role"], minimum=0)
                         and _text(value["scope"], minimum=0)):
        return False
    return True


def _original_matches(identity, raw):
    return (identity["bytes"] == len(raw)
            and identity["sha256"] == _hashlib.sha256(raw).hexdigest()
            and identity["git_blob"] == _hashlib.sha1(
                b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest())


def _inputs(inputs):
    _require(_closed(inputs, {"schema", "graph", "artifacts"}), "INVALID_INPUTS")
    _require(type(inputs["schema"]) is str
             and inputs["schema"] == "node-status-v1.source-inputs.v1", "INVALID_INPUTS")
    graph_frame = inputs["graph"]
    _require(_closed(graph_frame, {"identity", "raw"}), "INVALID_INPUTS")
    identity, graph_raw = graph_frame["identity"], graph_frame["raw"]
    _require(_identity_shape(identity), "INVALID_INPUTS")
    _require(type(graph_raw) is bytes and 1 <= len(graph_raw) <= _ORIGINAL_LIMIT,
             "INVALID_INPUTS")
    frames = inputs["artifacts"]
    _require(type(frames) is list and len(frames) == 3, "INVALID_INPUTS")
    paths = []
    aggregate = 0
    for frame in frames:
        _require(_closed(frame, {"identity", "raw"}), "INVALID_INPUTS")
        _require(_identity_shape(frame["identity"], artifact=True), "INVALID_INPUTS")
        _require(type(frame["raw"]) is bytes and len(frame["raw"]) <= _ORIGINAL_LIMIT,
                 "INVALID_INPUTS")
        aggregate += len(frame["raw"])
        paths.append(frame["identity"]["path"])
    _require(aggregate <= _ARTIFACT_AGGREGATE_LIMIT, "INVALID_INPUTS")
    _require(len(set(paths)) == 3 and paths == sorted(paths), "INVALID_INPUTS")
    _require(tuple(paths) == _ARTIFACT_PATHS, "ARTIFACT_SET_MISMATCH")
    _require(identity["repository"] == _MATH and identity["path"] == _GRAPH_PATH
             and _original_matches(identity, graph_raw), "GRAPH_IDENTITY_MISMATCH")
    for frame in frames:
        _require(frame["identity"]["repository"] == _MAIN
                 and _original_matches(frame["identity"], frame["raw"]),
                 "ARTIFACT_IDENTITY_MISMATCH")
    graph = _decode(graph_raw)
    nodes, outgoing, cohort = _graph(graph)
    return identity, {frame["identity"]["path"]: frame["identity"] for frame in frames}, nodes, outgoing, cohort


def _graph(graph):
    _require(type(graph) is dict and type(graph.get("nodes")) is dict
             and type(graph.get("edges")) is list, "GRAPH_DECLARATION_MISMATCH")
    nodes = graph["nodes"]
    _require(all(_matches(node, _NODE) and type(value) is dict
                 for node, value in nodes.items()), "GRAPH_DECLARATION_MISMATCH")
    _require(set(_SEEDS) | set(_SUPPORT) <= nodes.keys(), "GRAPH_DECLARATION_MISMATCH")
    outgoing = {node: [] for node in nodes}
    seen_edges = set()
    required_targets = {node: set() for node in nodes}
    for edge in graph["edges"]:
        _require(_closed(edge, {"from", "to", "relation", "required"}),
                 "GRAPH_DECLARATION_MISMATCH")
        source, target = edge["from"], edge["to"]
        _require(_matches(source, _NODE) and _matches(target, _NODE)
                 and source in nodes and target in nodes
                 and type(edge["required"]) is bool
                 and _text(edge["relation"], maximum=1024),
                 "GRAPH_DECLARATION_MISMATCH")
        key = (source, target, edge["relation"], edge["required"])
        _require(key not in seen_edges, "GRAPH_DECLARATION_MISMATCH")
        seen_edges.add(key)
        if edge["required"]:
            _require(target not in required_targets[source], "GRAPH_DECLARATION_MISMATCH")
            required_targets[source].add(target)
            outgoing[source].append(edge)
    cohort = set()
    pending = list(_SEEDS)
    while pending:
        node = pending.pop()
        if node not in cohort:
            cohort.add(node)
            pending.extend(edge["to"] for edge in outgoing[node])
    # Kahn traversal handles large bounded inputs without Python recursion.
    incoming = {node: 0 for node in cohort}
    for node in cohort:
        for edge in outgoing[node]:
            incoming[edge["to"]] += 1
    ready = [node for node, count in incoming.items() if count == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for edge in outgoing[node]:
            target = edge["to"]
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    _require(visited == len(cohort), "DEPENDENCY_CYCLE")
    _require(set(_SUPPORT) <= cohort, "READING_RULE_MISMATCH")
    component_targets = [edge["to"] for edge in outgoing[_PARENT]
                         if edge["to"] in _COMPONENTS]
    _require(set(component_targets) == set(_COMPONENTS)
             and nodes[_PARENT].get("reading_rule") == component_targets,
             "READING_RULE_MISMATCH")
    for node, (_kind, path) in _SUPPORT.items():
        declared_path = path + "/" if node == _COMPONENTS[3] else path
        _require(nodes[node].get("source") == declared_path, "READING_RULE_MISMATCH")
    return nodes, outgoing, cohort


def _owner(value):
    if value is None:
        return True
    return (_closed(value, {"id", "evidence_url"}, {"released_url"})
            and _text(value["id"]) and _matches(value["evidence_url"], _EVIDENCE)
            and ("released_url" not in value or _matches(value["released_url"], _EVIDENCE)))


def _http(value):
    if not _text(value) or any(character.isspace() for character in value):
        return False
    try:
        parsed = _urlparse(value)
    except ValueError:
        return False
    return parsed.scheme.lower() in ("http", "https") and bool(parsed.netloc)


def _link(value):
    return value is None or (
        _closed(value, {"node_id", "url"}) and _text(value["node_id"])
        and not any(character.isspace() for character in value["node_id"])
        and _http(value["url"]))


def _source(value):
    if not _closed(value, {"url", "rev"}):
        return False
    url, revision = value["url"], value["rev"]
    if not _text(url):
        return False
    pinned = _PINNED.fullmatch(url)
    if pinned is not None:
        return (_path(pinned.group(4)) and _matches(revision, _SHA1)
                and revision == pinned.group(3))
    return _matches(url, _OBJECT) and _utc(revision)


def _carried(value):
    if type(value) is not list:
        return False
    seen = set()
    for item in value:
        if not (_closed(item, {"id", "text", "source_url", "status"}, {"resolution_url"})
                and _text(item["id"]) and _text(item["text"])
                and _matches(item["source_url"], _CARRIED)
                and type(item["status"]) is str and item["status"] in ("open", "resolved")):
            return False
        if item["id"] in seen:
            return False
        seen.add(item["id"])
        if "resolution_url" in item and not _matches(item["resolution_url"], _EVIDENCE):
            return False
        if item["status"] == "resolved" and "resolution_url" not in item:
            return False
    return True


def _forbidden(record):
    pending = [record]
    while pending:
        value = pending.pop()
        if type(value) is dict:
            if _FORBIDDEN & value.keys():
                return True
            pending.extend(value.values())
        elif type(value) is list:
            pending.extend(value)
    return False


def _record(raw):
    _require(type(raw) is bytes, "INVALID_INPUTS")
    _require(1 <= len(raw) <= _RECORD_LIMIT, "INVALID_JSON")
    record = _decode(raw)
    _require(type(record) is dict and "schema_version" in record, "INVALID_RECORD")
    _require(type(record["schema_version"]) is int and record["schema_version"] == 1,
             "UNSUPPORTED_VERSION")
    _require(_closed(record, _REQUIRED, _OPTIONAL) and not _forbidden(record),
             "INVALID_RECORD")
    _require(_matches(record["node_id"], _NODE) and _owner(record["owner"])
             and _text(record["next_action"]), "INVALID_RECORD")
    for key in ("missing_input", "stops"):
        _require(type(record[key]) is list and all(_text(value) for value in record[key]),
                 "INVALID_RECORD")
    sources = record["sources"]
    _require(type(sources) is list and bool(sources) and all(_source(value) for value in sources),
             "INVALID_RECORD")
    _require(type(record["source_rev"]) is str and record["source_rev"] == sources[0]["rev"],
             "INVALID_RECORD")
    _require(_utc(record["observed_at"]) and type(record["status"]) is str
             and record["status"] in ("fresh", "stale", "unknown")
             and record["scientific_effect"] == "NONE", "INVALID_RECORD")
    for key in ("predecessor", "successor"):
        if key in record:
            _require(_link(record[key]), "INVALID_RECORD")
    if "carried_forward" in record:
        _require(_carried(record["carried_forward"]), "INVALID_RECORD")
    if "refresh_error" in record:
        _require(record["refresh_error"] is None or type(record["refresh_error"]) is str,
                 "INVALID_RECORD")
    _require(_identity_shape(record["graph_source"]), "INVALID_RECORD")
    for left, right in (("depends_on", "dependency_evidence"), ("artifacts", "artifact_evidence")):
        _require((left in record) == (right in record), "INVALID_RECORD")
    if "depends_on" in record:
        dependencies, evidence = record["depends_on"], record["dependency_evidence"]
        _require(type(dependencies) is list and 1 <= len(dependencies) <= 256
                 and all(_matches(node, _NODE) for node in dependencies), "INVALID_RECORD")
        _require(len(set(dependencies)) == len(dependencies), "INVALID_RECORD")
        _require(type(evidence) is list and 1 <= len(evidence) <= 256, "INVALID_RECORD")
        for item in evidence:
            _require(_closed(item, {"node_id", "relation", "required"})
                     and _matches(item["node_id"], _NODE)
                     and _text(item["relation"], maximum=1024)
                     and item["required"] is True, "INVALID_RECORD")
    if "artifacts" in record:
        paths, evidence = record["artifacts"], record["artifact_evidence"]
        _require(type(paths) is list and 1 <= len(paths) <= 3
                 and all(_path(path) for path in paths), "INVALID_RECORD")
        _require(len(set(paths)) == len(paths), "INVALID_RECORD")
        _require(type(evidence) is list and 1 <= len(evidence) <= 3
                 and all(_identity_shape(item, artifact=True, evidence=True) for item in evidence),
                 "INVALID_RECORD")
    return record


def _semantics(filename, record, context):
    identity, artifacts, _nodes, outgoing, cohort = context
    node = record["node_id"]
    _require(type(filename) is str, "INVALID_INPUTS")
    _require(filename == node + ".json" and node in cohort, "NODE_SCOPE_MISMATCH")
    _require(record["graph_source"] == identity, "GRAPH_IDENTITY_MISMATCH")
    evidence = record.get("artifact_evidence", [])
    _require((not evidence or node == "eng.hard-gate")
             and all(item["role"] == _ROLE and item["scope"] == _SCOPE for item in evidence),
             "ARTIFACT_SCOPE_MISMATCH")
    if node == "eng.hard-gate":
        _require(tuple(record.get("artifacts", [])) == _ARTIFACT_PATHS
                 and len(evidence) == 3, "ARTIFACT_SET_MISMATCH")
        for path, item in zip(_ARTIFACT_PATHS, evidence):
            declared = {key: item[key] for key in _ARTIFACT_KEYS}
            _require(declared == artifacts[path], "ARTIFACT_IDENTITY_MISMATCH")
    if node in _SUPPORT:
        kind, path = _SUPPORT[node]
        expected_url = "https://github.com/" + _MATH + "/" + kind + "/" + identity["commit"] + "/" + path
        _require(record["sources"][0] == {"url": expected_url, "rev": identity["commit"]}
                 and record["source_rev"] == identity["commit"], "READING_RULE_MISMATCH")
    expected = [{"node_id": edge["to"], "relation": edge["relation"], "required": True}
                for edge in outgoing[node]]
    _require(record.get("depends_on", []) == [item["node_id"] for item in expected]
             and record.get("dependency_evidence", []) == expected,
             "DEPENDENCY_MISMATCH")
    return record


def check_v1_record(filename: str, raw: bytes, inputs: dict) -> dict:
    """Validate one record; this does not assert that any other record exists."""
    context = _inputs(inputs)
    return _semantics(filename, _record(raw), context)


def check_v1_population(record_files: list[tuple[str, bytes]], inputs: dict) -> dict[str, dict]:
    """Require every record in the graph-derived fixed-seed closure exactly once."""
    context = _inputs(inputs)
    _require(type(record_files) is list, "INVALID_INPUTS")
    filenames = set()
    aggregate = 0
    for item in record_files:
        _require(type(item) is tuple and len(item) == 2
                 and type(item[0]) is str and type(item[1]) is bytes, "INVALID_INPUTS")
        filename, raw = item
        _require(filename not in filenames, "DUPLICATE_FILENAME")
        filenames.add(filename)
        aggregate += len(raw)
        _require(aggregate <= _AGGREGATE_LIMIT, "INVALID_INPUTS")
    parsed = []
    by_id = {}
    for filename, raw in record_files:
        record = _record(raw)
        _require(record["node_id"] not in by_id, "DUPLICATE_NODE_ID")
        by_id[record["node_id"]] = record
        parsed.append((filename, record))
    for filename, record in parsed:
        _semantics(filename, record, context)
    _require(by_id.keys() == context[4], "POPULATION_MISMATCH")
    return by_id


def project_v1_population(record_files: list[tuple[str, bytes]], inputs: dict) -> bytes:
    """Serialize only the validated by-ID map; preserve all record values."""
    population = check_v1_population(record_files, inputs)
    raw = (_json.dumps(population, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    _require(len(raw) <= _AGGREGATE_LIMIT, "INVALID_INPUTS")
    return raw
