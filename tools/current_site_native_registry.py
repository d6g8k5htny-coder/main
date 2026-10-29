"""Pure native registry construction and declared current-bundle validation.

These four APIs join already admitted bytes and immutable metadata. They do
not acquire artifacts, rerun archive admission, establish custody, publish,
restore files, or confer scientific authority. Every refusal is ValueError;
no validation relies on assertions or mutable publication state.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import re


_MIB = 1024 * 1024
_REPOSITORY = "d6g8k5htny-coder/main"
_ORIGIN = "https://d6g8k5htny-coder.github.io/main/site/"
_MAX_SEQUENCE = 2**63 - 1
_MAX_ARCHIVES = 8192
_MAX_IMMUTABLE = 512 * _MIB
_MAX_REGISTRY = 16 * _MIB
_MAX_ASSET = 32 * _MIB
_MAX_BUNDLE = 32 * _MIB
_MAX_PAGE = 4 * _MIB
_MAX_MANIFEST = 65536
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_DECIMAL = re.compile(r"[1-9][0-9]*")
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_TEST_ID = re.compile(r"test_architecture_[A-Za-z0-9_]+\.[A-Za-z_][A-Za-z0-9_]*\.test[A-Za-z0-9_]*")
_BINDING_FIELDS = frozenset({"checked_commit", "repository", "run_id", "run_attempt",
                             "receipt_sha256", "artifact_id", "artifact_sha256", "artifact_name"})
_SELECTION_FIELDS = frozenset({"workflow_id", "workflow_path", "event",
                               "original_created_at", "producer_tree"})
_SCIENCE_FIELDS = frozenset({"schema_version", "scientific_effect", "scientific_status_authority"})
_REGISTRY_FIELDS = _SCIENCE_FIELDS | {"custody", "origin", "sequence", "previous_registry_sha256",
    "current_artifact_sha256", "entries", "immutable_bytes", "capacity"}
_ASSET_NAMES = ("architecture-run-receipt.json", "graph.json", "projection.json")
_GRAPH_SOURCE = {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "7858329974e28be79f29b22644370084ff43da4f",
    "captured_at": "2026-10-03T15:07:03Z",
    "graph_sha256": "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09",
    "gate_sha256": "a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8",
}
_MEMBER_LIMITS = {
    "ARTIFACT_SHA256SUMS": 16384, "checked-commit.txt": 128,
    "run-id.txt": 128, "run-attempt.txt": 128, "container-exit-status.txt": 128,
    "exporter.stdout": 65536, "exporter.stderr": 65536,
    "runtime-image.json": _MIB, "generated/graph.json": 4 * _MIB,
    "architecture-run-receipt.json": 4 * _MIB, "console.log": 4 * _MIB,
    "tests-normal.log": 16 * _MIB, "tests-optimized.log": 16 * _MIB,
}
_EVIDENCE_MEMBERS = frozenset({"generated/graph.json", "checked-commit.txt", "run-id.txt",
    "run-attempt.txt", "container-exit-status.txt", "tests-normal.log",
    "tests-optimized.log", "runtime-image.json"})
_IMAGE_DIGEST = "python@sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48"
_READBACK_SCOPE = "legacy3+current-native5+registry;retained-native-assets-not-observed"
_CANDIDATE_FIELDS = frozenset({"schema_version", "scientific_effect", "repository",
    "receipt_branch", "source", "run", "artifact", "manifest", "metadata", "payload", "snapshot_ids"})


def _require(condition, message="native metadata refused"):
    if not condition:
        raise ValueError(message)


def _closed(value, fields):
    _require(type(value) is dict and all(type(key) is str for key in value)
             and set(value) == set(fields), "closed native schema refused")
    return value


def _integer(value, lower, upper):
    _require(type(value) is int and lower <= value <= upper, "native integer refused")
    return value


def _pattern(value, pattern):
    _require(type(value) is str and pattern.fullmatch(value) is not None, "native identity refused")
    return value


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _same(left, right):
    # Python equality alone equates True with 1, and 1.0 with 1.
    return _canonical(left) == _canonical(right)


def _json_types(value):
    """Reject non-JSON Python values, nonfinite floats and lone surrogates."""
    try:
        kind = type(value)
        if kind is dict:
            for key, child in value.items():
                _require(type(key) is str)
                key.encode("utf-8")
                _json_types(child)
        elif kind is list:
            for child in value:
                _json_types(child)
        elif kind is str:
            value.encode("utf-8")
        elif kind is float:
            _require(math.isfinite(value))
        else:
            _require(value is None or kind in (bool, int))
    except (UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError("native JSON value refused") from error


def _parse(raw, maximum):
    """Read one strict UTF-8 JSON object; input whitespace need not be canonical."""
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, "native JSON byte capacity refused")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate native JSON key refused")
            result[key] = value
        return result

    def no_constant(value):
        raise ValueError("nonfinite native JSON refused")

    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                            parse_constant=no_constant)
        _require(type(result) is dict, "native JSON object required")
        _json_types(result)
        return result
    except (UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError("native JSON refused") from error


def _science(document):
    _integer(document["schema_version"], 1, 1)
    _require(document["scientific_effect"] == "NONE"
             and type(document["scientific_effect"]) is str
             and document["scientific_status_authority"] is False)


def _custody(document):
    _science(document)
    _require(document["custody"] == "unknown" and type(document["custody"]) is str)


def _binding(value):
    _closed(value, _BINDING_FIELDS)
    _require(all(type(item) is str for item in value.values()))
    _require(sum(len(item) for item in value.values()) <= 16 * _MIB)
    for name, pattern in (("checked_commit", _HEX40), ("run_id", _DECIMAL),
            ("run_attempt", _DECIMAL), ("receipt_sha256", _HEX64),
            ("artifact_id", _DECIMAL), ("artifact_sha256", _HEX64)):
        _pattern(value[name], pattern)
    _require(value["repository"] == _REPOSITORY)
    _require(value["artifact_name"] == "research-architecture-" + value["run_id"] + "-" + value["run_attempt"])
    return value


def _selection(value):
    _closed(value, _SELECTION_FIELDS)
    _require(all(type(item) is str for item in value.values()))
    # This identity grammar has no digit-count cap. No integer conversion is
    # needed for workflow_id, including the exact 16-MiB registry boundary.
    _pattern(value["workflow_id"], _DECIMAL)
    _pattern(value["producer_tree"], _HEX40)
    _require(value["workflow_path"] == ".github/workflows/workspace-landing.yml" and value["event"] == "push")
    _pattern(value["original_created_at"], _UTC)
    try:
        stamp = value["original_created_at"]
        datetime(int(stamp[0:4]), int(stamp[5:7]), int(stamp[8:10]),
                 int(stamp[11:13]), int(stamp[14:16]), int(stamp[17:19]))
    except (ValueError, OverflowError) as error:
        raise ValueError("native producer UTC refused") from error
    return value


def _descriptor(value, path, maximum, *, minimum=1):
    _closed(value, {"path", "bytes", "sha256"})
    _require(type(value["path"]) is str and value["path"] == path, "native descriptor path refused")
    _integer(value["bytes"], minimum, maximum)
    _pattern(value["sha256"], _HEX64)
    return value


def _describe(path, raw):
    return {"path": path, "bytes": len(raw), "sha256": _sha(raw)}


def _immutable_paths(binding):
    prefix = "native-graph/" + binding["artifact_sha256"] + "/"
    return [prefix + name for name in _ASSET_NAMES]


def _entry(value):
    _closed(value, {"binding", "selection", "files"})
    _binding(value["binding"])
    _selection(value["selection"])
    paths = _immutable_paths(value["binding"])
    _require(type(value["files"]) is list and len(value["files"]) == 3)
    total = sum(_descriptor(item, path, _MAX_ASSET)["bytes"]
                for item, path in zip(value["files"], paths))
    _require(total <= _MAX_ASSET, "native entry capacity refused")
    return total


def validate_native_registry(raw: bytes) -> dict:
    """Validate exact registry v1 identities, sorted inventory and capacity joins."""
    document = _closed(_parse(raw, _MAX_REGISTRY), _REGISTRY_FIELDS)
    _custody(document)
    _require(document["origin"] == _ORIGIN and type(document["origin"]) is str)
    _integer(document["sequence"], 1, _MAX_SEQUENCE)
    if document["previous_registry_sha256"] is not None:
        _pattern(document["previous_registry_sha256"], _HEX64)
    _pattern(document["current_artifact_sha256"], _HEX64)
    entries = document["entries"]
    _require(type(entries) is list and 1 <= len(entries) <= _MAX_ARCHIVES, "native archive count refused")
    total = 0
    previous_prefix = None
    found_current = False
    for entry in entries:
        total += _entry(entry)
        prefix = entry["binding"]["artifact_sha256"]
        _require(previous_prefix is None or previous_prefix < prefix, "native prefixes must be unique and sorted")
        previous_prefix = prefix
        found_current = found_current or prefix == document["current_artifact_sha256"]
        _require(total <= _MAX_IMMUTABLE, "native immutable capacity refused")
    _require(found_current, "native current pointer unresolved")
    _integer(document["immutable_bytes"], 1, _MAX_IMMUTABLE)
    _require(document["immutable_bytes"] == total)
    capacity = _closed(document["capacity"], {"archives", "maximum_archives", "bytes", "maximum_bytes"})
    for key, expected in (("archives", len(entries)), ("maximum_archives", _MAX_ARCHIVES),
                          ("bytes", total), ("maximum_bytes", _MAX_IMMUTABLE)):
        _integer(capacity[key], expected, expected)
    _require(len(_canonical(document)) <= _MAX_REGISTRY, "canonical native registry capacity refused")
    return document


def _context(value):
    _closed(value, {"repository", "ref", "commit", "run_id", "run_attempt", "event"})
    _require(value["repository"] == _REPOSITORY and type(value["repository"]) is str
             and value["ref"] == "refs/heads/main" and type(value["ref"]) is str)
    _pattern(value["commit"], _HEX40)
    _integer(value["run_id"], 1, _MAX_SEQUENCE)
    _integer(value["run_attempt"], 1, 100000)
    _require(type(value["event"]) is str and value["event"] in ("push", "schedule", "workflow_dispatch"))


def _graph_source(value):
    _closed(value, _GRAPH_SOURCE)
    _require(_same(value, _GRAPH_SOURCE), "fixed native graph source refused")


def _producer_receipt(value, binding, graph, members):
    """Check declared receipt closure, without interpreting unavailable log bodies."""
    _closed(value, _SCIENCE_FIELDS | {"checked_commit", "repository", "run_id", "run_attempt",
        "runtime_image", "test_modes", "graph_source", "evidence_file_sha256"})
    _science(value)
    for key in ("checked_commit", "repository", "run_id", "run_attempt"):
        _require(type(value[key]) is str and value[key] == binding[key])
    _graph_source(value["graph_source"])
    _require(_same(value["graph_source"], graph["source"]))
    image = _closed(value["runtime_image"], {"id", "repo_digest"})
    _pattern(image["id"], re.compile(r"sha256:[0-9a-f]{64}"))
    _require(type(image["repo_digest"]) is str and image["repo_digest"] == _IMAGE_DIGEST)
    evidence = _closed(value["evidence_file_sha256"], _EVIDENCE_MEMBERS)
    for name, digest in evidence.items():
        _pattern(digest, _HEX64)
        _require(digest == members[name]["sha256"], "native evidence descriptor join refused")
    modes = value["test_modes"]
    _require(type(modes) is list and len(modes) == 2)
    for row, mode in zip(modes, ("normal", "optimized")):
        _closed(row, {"mode", "test_count", "test_ids"})
        _require(type(row["mode"]) is str and row["mode"] == mode)
        ids = row["test_ids"]
        _require(type(ids) is list and 1 <= len(ids) <= 4096)
        for identity in ids:
            _pattern(identity, _TEST_ID)
        _require(ids == sorted(set(ids)))
        _integer(row["test_count"], len(ids), len(ids))
    _require(modes[0]["test_ids"] == modes[1]["test_ids"])


def _projection(document, binding, graph, receipt, graph_raw, receipt_raw):
    """Join the prior admission's declared inventory to the exact public members."""
    _closed(document, _SCIENCE_FIELDS | {"custody", "graph", "admission"})
    _custody(document)
    projected_graph = _closed(document["graph"], {"identity", "data"})
    identity = _closed(projected_graph["identity"], {"kind", "binding", "member"})
    _require(type(identity["kind"]) is str and identity["kind"] == "native-artifact-member")
    _binding(identity["binding"])
    _require(_same(identity["binding"], binding) and _same(projected_graph["data"], graph))
    graph_member = _describe("generated/graph.json", graph_raw)
    _descriptor(identity["member"], "generated/graph.json", _MAX_ASSET)
    _require(_same(identity["member"], graph_member))
    admission = _closed(document["admission"], _SCIENCE_FIELDS | {"custody", "verification_scope", "binding",
        "archive", "graph_member", "producer_receipt_sha256", "producer_receipt"})
    _custody(admission)
    _require(type(admission["verification_scope"]) is str
             and admission["verification_scope"] == "declared-native-pins-and-archive-content")
    _binding(admission["binding"])
    _require(_same(admission["binding"], binding))
    archive = _closed(admission["archive"], {"sha256", "bytes", "members"})
    _pattern(archive["sha256"], _HEX64)
    _require(archive["sha256"] == binding["artifact_sha256"])
    _integer(archive["bytes"], 22, 64 * _MIB)
    rows = archive["members"]
    names = sorted(_MEMBER_LIMITS)
    _require(type(rows) is list and len(rows) == len(names))
    members = {}
    for row, name in zip(rows, names):
        _descriptor(row, name, _MEMBER_LIMITS[name], minimum=0)
        members[name] = row
    _require(sum(row["bytes"] for row in rows) <= 64 * _MIB)
    _require(_same(members["generated/graph.json"], graph_member))
    _require(_same(members["architecture-run-receipt.json"], _describe("architecture-run-receipt.json", receipt_raw)))
    _descriptor(admission["graph_member"], "generated/graph.json", _MAX_ASSET)
    _require(_same(admission["graph_member"], graph_member))
    _pattern(admission["producer_receipt_sha256"], _HEX64)
    _require(admission["producer_receipt_sha256"] == binding["receipt_sha256"] == _sha(receipt_raw))
    _require(_same(admission["producer_receipt"], receipt))
    _producer_receipt(receipt, binding, graph, members)
    # The text members whose exact bodies are derivable from admitted pins can
    # be checked without recovering logs, raw archives, or external artifacts.
    for name, raw in (("checked-commit.txt", (binding["checked_commit"] + "\n").encode("ascii")),
            ("run-id.txt", (binding["run_id"] + "\n").encode("ascii")),
            ("run-attempt.txt", (binding["run_attempt"] + "\n").encode("ascii")),
            ("container-exit-status.txt", b"0\n"), ("exporter.stderr", b""),
            ("exporter.stdout", b"/output/generated/graph.json\n")):
        _require(_same(members[name], _describe(name, raw)))


def admit_current_native_bundle(files: dict[str, bytes], *, context: dict,
        current_payload_bytes: bytes, binding: dict, selection: dict) -> dict:
    """Return the validated manifest directly after exact five-buffer closure.

    This consumes a previously admitted bundle. Descriptor metadata can be
    checked here; original ZIP geometry, logs and source admission cannot be
    re-established from these five public files and are not claimed.
    """
    _context(context)
    _binding(binding)
    _selection(selection)
    _require(type(current_payload_bytes) is bytes and len(current_payload_bytes) <= 16 * _MIB)
    paths = _immutable_paths(binding)
    expected_paths = set(paths) | {"proof-tree.html", "current-build-manifest.json"}
    _require(type(files) is dict and all(type(path) is str for path in files)
             and set(files) == expected_paths, "exact five-output native bundle required")
    for path, raw in files.items():
        limit = _MAX_PAGE if path == "proof-tree.html" else _MAX_MANIFEST if path == "current-build-manifest.json" else _MAX_ASSET
        _require(type(raw) is bytes and 0 < len(raw) <= limit, "native bundle buffer refused")
    _require(sum(len(raw) for raw in files.values()) <= _MAX_BUNDLE, "native bundle aggregate refused")
    manifest = _closed(_parse(files["current-build-manifest.json"], _MAX_MANIFEST),
                       _SCIENCE_FIELDS | {"custody", "verification_scope", "build", "native", "files"})
    _custody(manifest)
    _require(type(manifest["verification_scope"]) is str
             and manifest["verification_scope"] == "declared-build-context-and-admitted-native-archive")
    build = _closed(manifest["build"], {"repository", "commit", "current_payload"})
    _require(type(build["repository"]) is str and build["repository"] == context["repository"])
    _pattern(build["commit"], _HEX40)
    _require(build["commit"] == context["commit"])
    _descriptor(build["current_payload"], "current.json", 16 * _MIB, minimum=0)
    _require(_same(build["current_payload"], _describe("current.json", current_payload_bytes)))
    native = _closed(manifest["native"], {"binding", "graph_source"})
    _binding(native["binding"])
    _require(_same(native["binding"], binding))
    _graph_source(native["graph_source"])
    inventory = manifest["files"]
    inventory_paths = sorted(expected_paths - {"current-build-manifest.json"})
    _require(type(inventory) is list and len(inventory) == 4)
    for item, path in zip(inventory, inventory_paths):
        _descriptor(item, path, _MAX_PAGE if path == "proof-tree.html" else _MAX_ASSET)
        _require(_same(item, _describe(path, files[path])), "native manifest descriptor join refused")
    prefix = "native-graph/" + binding["artifact_sha256"] + "/"
    graph_raw = files[prefix + "graph.json"]
    receipt_raw = files[prefix + "architecture-run-receipt.json"]
    graph = _closed(_parse(graph_raw, _MAX_ASSET), _SCIENCE_FIELDS | {"source", "nodes", "edges", "dimensions"})
    _science(graph)
    _graph_source(graph["source"])
    _require(type(graph["nodes"]) is dict and type(graph["edges"]) is list and type(graph["dimensions"]) is dict)
    _require(_same(graph["source"], native["graph_source"]))
    receipt = _parse(receipt_raw, _MAX_ASSET)
    projection = _parse(files[prefix + "projection.json"], _MAX_ASSET)
    _projection(projection, binding, graph, receipt, graph_raw, receipt_raw)
    return manifest


def _base(value):
    _closed(value, {"receipt_commit", "public_receipt_commit", "registry_sha256", "sequence"})
    for field in ("receipt_commit", "public_receipt_commit"):
        if value[field] is not None:
            _pattern(value[field], _HEX40)
    if value["registry_sha256"] is not None:
        _pattern(value["registry_sha256"], _HEX64)
    _integer(value["sequence"], 0, _MAX_SEQUENCE - 1)
    _require((value["registry_sha256"] is None) == (value["sequence"] == 0), "native base lineage refused")
    return value


def build_native_registry(previous_raw: bytes | None, current_bundle: dict[str, bytes], *,
        base: dict, context: dict, current_payload_bytes: bytes, binding: dict, selection: dict) -> bytes:
    """Conserve the complete prior registry and add at most one immutable prefix.

    Sequence and previous hash always derive from the supplied original base;
    repeated renders with that same base produce identical bytes.
    """
    _base(base)
    if previous_raw is None:
        _require(base["registry_sha256"] is None and base["sequence"] == 0)
        previous = None
    else:
        previous = validate_native_registry(previous_raw)
        _require(base["registry_sha256"] == _sha(previous_raw)
                 and base["sequence"] == previous["sequence"], "native prior registry/base mismatch")
    admit_current_native_bundle(current_bundle, context=context, current_payload_bytes=current_payload_bytes,
                                binding=binding, selection=selection)
    entry = {"binding": dict(binding), "selection": dict(selection),
             "files": [_describe(path, current_bundle[path]) for path in _immutable_paths(binding)]}
    entries = [] if previous is None else list(previous["entries"])
    prefix = binding["artifact_sha256"]
    found = next((row for row in entries if row["binding"]["artifact_sha256"] == prefix), None)
    if found is not None:
        _require(_same(found, entry), "retained native prefix identity conflict")
    else:
        entries.append(entry)
    entries.sort(key=lambda row: row["binding"]["artifact_sha256"])
    total = sum(item["bytes"] for row in entries for item in row["files"])
    _require(len(entries) <= _MAX_ARCHIVES and total <= _MAX_IMMUTABLE, "native union capacity refused")
    result = {"schema_version": 1, "scientific_effect": "NONE", "scientific_status_authority": False,
        "custody": "unknown", "origin": _ORIGIN, "sequence": base["sequence"] + 1,
        "previous_registry_sha256": base["registry_sha256"], "current_artifact_sha256": prefix,
        "entries": entries, "immutable_bytes": total, "capacity": {"archives": len(entries),
            "maximum_archives": _MAX_ARCHIVES, "bytes": total, "maximum_bytes": _MAX_IMMUTABLE}}
    raw = _canonical(result)
    validate_native_registry(raw)
    return raw


def _candidate(value):
    """Admit the full effective public candidate, excluding receipt phase fields."""
    _require(type(value) is dict)
    version = value.get("schema_version")
    _integer(version, 1, 2)
    _closed(value, _CANDIDATE_FIELDS | ({"native"} if version == 2 else set()))
    _require(type(value["scientific_effect"]) is str and value["scientific_effect"] == "NONE"
             and type(value["repository"]) is str and value["repository"] == _REPOSITORY
             and type(value["receipt_branch"]) is str and value["receipt_branch"] == "codex/site-publication-receipts")
    source = _closed(value["source"], {"commit", "ref", "workflow"})
    _pattern(source["commit"], _HEX40)
    _require(type(source["ref"]) is str and source["ref"] == "refs/heads/main"
             and type(source["workflow"]) is str and source["workflow"] == ".github/workflows/current-site.yml")
    run = _closed(value["run"], {"id", "attempt"})
    _integer(run["id"], 1, _MAX_SEQUENCE)
    _integer(run["attempt"], 1, 100000)
    artifact = _closed(value["artifact"], {"sha256", "files", "bytes"})
    _pattern(artifact["sha256"], _HEX64)
    _integer(artifact["files"], 1, 1000000)
    _integer(artifact["bytes"], 1, 1024**3)
    for name in ("manifest", "metadata", "payload"):
        row = _closed(value[name], {"bytes", "sha256"})
        _integer(row["bytes"], 1, 16 * _MIB)
        _pattern(row["sha256"], _HEX64)
    snapshots = value["snapshot_ids"]
    _require(type(snapshots) is list and 1 <= len(snapshots) <= 512)
    for item in snapshots:
        _pattern(item, _HEX64)
    _require(snapshots == sorted(set(snapshots)))
    if version == 2:
        native = _closed(value["native"], {"base", "sequence", "binding", "selection",
                                          "registry", "build_manifest", "readback_scope"})
        _base(native["base"])
        _integer(native["sequence"], 1, _MAX_SEQUENCE)
        _require(native["sequence"] == native["base"]["sequence"] + 1)
        _binding(native["binding"])
        _selection(native["selection"])
        _descriptor(native["registry"], "native-graph/index.json", _MAX_REGISTRY)
        _descriptor(native["build_manifest"], "current-build-manifest.json", _MAX_MANIFEST)
        _require(type(native["readback_scope"]) is str and native["readback_scope"] == _READBACK_SCOPE)
    return value


def _commit_identity(value):
    """Validate material native commit identity; ordinary API extra fields remain."""
    _require(type(value) is dict and {"sha", "commit", "parents"} <= set(value))
    sha = _pattern(value["sha"], _HEX40)
    commit = value["commit"]
    _require(type(commit) is dict and type(commit.get("tree")) is dict)
    tree = _pattern(commit["tree"].get("sha"), _HEX40)
    parents = value["parents"]
    _require(type(parents) is list)
    identities = []
    for parent in parents:
        _require(type(parent) is dict)
        identities.append(_pattern(parent.get("sha"), _HEX40))
    _require(len(identities) == len(set(identities)) and sha not in identities)
    return sha, tree, tuple(identities)


def _comparison_proof(value, old, new, old_tree, new_tree, known_trees, known_commits):
    """Require a complete bounded forward native response with joined endpoints."""
    _require(type(value) is dict, "PROGRESS_UNKNOWN")
    _json_types(value)
    required = {"status", "ahead_by", "behind_by", "total_commits", "base_commit", "merge_base_commit", "commits"}
    _require(required <= set(value), "PROGRESS_UNKNOWN")
    _require(type(value["status"]) is str and value["status"] == "ahead", "PROGRESS_UNKNOWN")
    count = value["ahead_by"]
    _require(type(count) is int and count > 0, "PROGRESS_UNKNOWN")
    _require(count <= 250, "PROGRESS_BOUND")
    _integer(value["behind_by"], 0, 0)
    _integer(value["total_commits"], count, count)
    rows = value["commits"]
    _require(type(rows) is list and len(rows) == count, "PROGRESS_BOUND")
    base = _commit_identity(value["base_commit"])
    merge = _commit_identity(value["merge_base_commit"])
    _require(base == merge and base[0] == old, "PROGRESS_UNKNOWN")
    if old_tree is not None:
        _require(base[1] == old_tree, "PROGRESS_UNKNOWN")
    for identity in (base, merge):
        _join_commit(identity, known_trees, known_commits)
    identities = {}
    for row in rows:
        identity = _commit_identity(row)
        _require(identity[0] != old and identity[0] not in identities, "PROGRESS_UNKNOWN")
        identities[identity[0]] = identity
        _join_commit(identity, known_trees, known_commits)
    _require(new in identities and identities[new][1] == new_tree, "PROGRESS_UNKNOWN")
    # A complete directed parent chain must connect the known endpoints. Merge
    # parents outside this comparison are permitted; a fabricated disconnected
    # list cannot establish progress merely by claiming status=ahead.
    pending = [(new, False)]
    visiting = set()
    seen = set()
    found_base = False
    while pending:
        sha, finished = pending.pop()
        if sha == old:
            found_base = True
            continue
        if sha not in identities or sha in seen:
            continue
        if finished:
            visiting.remove(sha)
            seen.add(sha)
            continue
        _require(sha not in visiting, "PROGRESS_UNKNOWN")
        visiting.add(sha)
        pending.append((sha, True))
        pending.extend((parent, False) for parent in identities[sha][2])
    _require(found_base and len(seen) == len(identities), "PROGRESS_UNKNOWN")


def _join_commit(identity, known_trees, known_commits):
    sha, tree, parents = identity
    _require(sha not in known_trees or known_trees[sha] == tree, "PROGRESS_UNKNOWN")
    _require(sha not in known_commits or known_commits[sha] == (tree, parents), "PROGRESS_UNKNOWN")
    known_trees[sha] = tree
    known_commits[sha] = tree, parents


def _comparison(value, old, new, old_tree, new_tree, known_trees, known_commits):
    try:
        _comparison_proof(value, old, new, old_tree, new_tree, known_trees, known_commits)
    except ValueError as error:
        reason = "PROGRESS_BOUND" if str(error) == "PROGRESS_BOUND" else "PROGRESS_UNKNOWN"
        raise ValueError(reason) from error


def _decimal_order(value):
    # Positive decimal grammar forbids leading zeros. Length and lexical order
    # therefore equal numeric ordering without bounded Python int conversion.
    return len(value), value


def validate_native_progress(previous: dict | None, current: dict, *, comparisons: dict) -> None:
    """Refuse build/producer rewind or unknown ancestry without mutating inputs."""
    _closed(current, {"build", "producer"})
    build = _closed(current["build"], {"repository", "commit", "tree"})
    _require(type(build["repository"]) is str and build["repository"] == _REPOSITORY)
    _pattern(build["commit"], _HEX40)
    _pattern(build["tree"], _HEX40)
    producer = _closed(current["producer"], {"binding", "selection"})
    binding = _binding(producer["binding"])
    selection = _selection(producer["selection"])
    known_trees = {build["commit"]: build["tree"]}
    _require(binding["checked_commit"] not in known_trees
             or known_trees[binding["checked_commit"]] == selection["producer_tree"], "native endpoint tree conflict")
    known_trees[binding["checked_commit"]] = selection["producer_tree"]
    known_commits = {}
    _require(type(comparisons) is dict and all(type(key) is str for key in comparisons)
             and set(comparisons) <= {"build", "producer"})
    for value in comparisons.values():
        _require(type(value) is dict, "PROGRESS_UNKNOWN")
        _json_types(value)
    if previous is None:
        return None
    _candidate(previous)
    old_build = previous["source"]["commit"]
    old_native = previous.get("native")
    old_build_tree = None
    if old_native is not None:
        old_producer_sha = old_native["binding"]["checked_commit"]
        old_producer_tree = old_native["selection"]["producer_tree"]
        _require(old_producer_sha not in known_trees or known_trees[old_producer_sha] == old_producer_tree,
                 "native endpoint tree conflict")
        known_trees[old_producer_sha] = old_producer_tree
        if old_producer_sha == old_build:
            old_build_tree = old_producer_tree
    if old_build != build["commit"]:
        _comparison(comparisons.get("build"), old_build, build["commit"], old_build_tree, build["tree"],
                    known_trees, known_commits)
    if old_native is None:
        return None
    old_binding = old_native["binding"]
    old_selection = old_native["selection"]
    if old_binding["artifact_sha256"] == binding["artifact_sha256"]:
        _require(_same(old_binding, binding) and _same(old_selection, selection), "retained native prefix identity conflict")
    old_producer = old_binding["checked_commit"]
    new_producer = binding["checked_commit"]
    if old_producer != new_producer:
        _comparison(comparisons.get("producer"), old_producer, new_producer,
                    old_selection["producer_tree"], selection["producer_tree"], known_trees, known_commits)
    else:
        _require(old_selection["producer_tree"] == selection["producer_tree"], "native producer tree conflict")
        old_order = (old_selection["original_created_at"], _decimal_order(old_binding["run_id"]))
        new_order = (selection["original_created_at"], _decimal_order(binding["run_id"]))
        _require(new_order >= old_order, "native producer run rewind refused")
        if old_binding["run_id"] == binding["run_id"]:
            _require(old_selection["original_created_at"] == selection["original_created_at"], "native original run identity conflict")
            _require(_decimal_order(binding["run_attempt"]) >= _decimal_order(old_binding["run_attempt"]),
                     "native producer attempt rewind refused")
    return None


# Publication-base admission is an in-memory consumer of already captured
# observations. Its Git proof authenticates original receipt bytes; its HTTP
# descriptors do not attest that this process independently read a service.
import base64 as _pb_base64


_PB_BRANCH = "codex/site-publication-receipts"
_PB_WORKFLOW = ".github/workflows/current-site.yml"
_PB_GUARD = "tools/current_site_native_guard.py"
_PB_BOT_NAME = "Current research site publication"
_PB_BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"
_PB_FILES = frozenset({"anchor.json", "public-anchor.json", "base.json",
    "git-proof.json", "pending-plan.json", "qualification.json"})
_PB_FILE_CAPS = {"anchor.json": 2 * _MIB, "public-anchor.json": 2 * _MIB,
    "base.json": 65536, "pending-plan.json": 65536,
    "git-proof.json": 16 * _MIB, "qualification.json": 16 * _MIB,
    "registry.json": 16 * _MIB}
_PB_RECEIPT_FIELDS = frozenset({"phase", "recorded_at", "recorded_by",
    "previous_receipt_commit"})
_PB_BASE_FIELDS = frozenset({"schema_version", "receipt_commit",
    "public_receipt_commit", "receipt_phase", "native_sequence",
    "registry_sha256", "mode"})
_PB_PLAN_FIELDS = frozenset({"schema_version", "operation", "operational_head",
    "effective_public_head", "pending_prepared_head", "consumer_commit",
    "git_proof_sha256", "qualification_sha256", "served_snapshot_ids",
    "unserved_snapshot_ids"})
_PB_QUALIFICATION_FIELDS = frozenset({"schema_version", "repository",
    "consumer_commit", "initial_receipt_head", "final_receipt_head",
    "initial_main_commit", "final_main_commit", "operation",
    "target_public_head", "no_start", "public_observation", "metadata"})
_PB_LEGACY_PATHS = {"manifest": "snapshots/index.json",
    "metadata": "current-meta.json", "payload": "current.json"}
_PB_ABSENCE_PATHS = frozenset({"current-build-manifest.json",
    "native-graph/index.json", "proof-tree.html"})


def _pb_utc(value):
    _pattern(value, _UTC)
    try:
        return datetime(int(value[:4]), int(value[5:7]), int(value[8:10]),
                        int(value[11:13]), int(value[14:16]), int(value[17:19]))
    except (ValueError, OverflowError) as error:
        raise ValueError("publication UTC refused") from error


def _pb_nullable_head(value):
    if value is not None:
        _pattern(value, _HEX40)
    return value


def _pb_anchor_json(raw):
    """Strict anchor-only object-or-null decoding; _parse stays object-only."""
    _require(type(raw) is bytes and 0 < len(raw) <= 2 * _MIB)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate publication anchor key")
            result[key] = value
        return result

    def no_constant(value):
        raise ValueError("nonfinite publication anchor")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=no_constant)
        _json_types(value)
        _require(value is None or type(value) is dict)
        return value
    except (UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError("publication anchor JSON refused") from error


def _pb_base64_bytes(value, maximum, *, minimum=1):
    # Bound encoded input before allocating a decoded buffer. Round-trip
    # equality also rejects excess padding and nonzero unused padding bits.
    _require(type(value) is str and len(value) <= 4 * ((maximum + 2) // 3),
             "publication Base64 capacity refused")
    try:
        encoded = value.encode("ascii")
        raw = _pb_base64.b64decode(encoded, validate=True)
    except (ValueError, UnicodeError) as error:
        raise ValueError("publication Base64 refused") from error
    _require(minimum <= len(raw) <= maximum
             and _pb_base64.b64encode(raw) == encoded,
             "publication Base64 spelling refused")
    return raw


def _pb_git_sha(kind, raw):
    return hashlib.sha1(kind.encode("ascii") + b" "
        + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def _pb_receipt(value):
    _require(type(value) is dict)
    version = _integer(value.get("schema_version"), 1, 2)
    candidate_fields = _CANDIDATE_FIELDS | ({"native"} if version == 2 else set())
    _closed(value, candidate_fields | _PB_RECEIPT_FIELDS
            | ({"reconciliation"} if version == 2 else set()))
    candidate = {key: value[key] for key in candidate_fields}
    _candidate(candidate)
    phase = value["phase"]
    _require(type(phase) is str and phase in
             (("prepared", "observed") if version == 1 else
              ("prepared", "observed", "superseded")))
    _pb_utc(value["recorded_at"])
    observer = _closed(value["recorded_by"],
                       {"source_commit", "run_id", "run_attempt"})
    _pattern(observer["source_commit"], _HEX40)
    _integer(observer["run_id"], 1, _MAX_SEQUENCE)
    _integer(observer["run_attempt"], 1, 100000)
    previous = _pb_nullable_head(value["previous_receipt_commit"])
    _require(phase == "prepared" or previous is not None)
    if version == 2:
        reconciliation = value["reconciliation"]
        if phase == "prepared":
            _require(reconciliation is None)
        elif reconciliation is None:
            _require(phase == "observed")
        else:
            _closed(reconciliation, {"kind", "effective_public_receipt_commit",
                "run_id", "run_attempt", "job_id", "proof_sha256"})
            _pattern(reconciliation["proof_sha256"], _HEX64)
            effective = _pb_nullable_head(reconciliation["effective_public_receipt_commit"])
            kind = reconciliation["kind"]
            _require(type(kind) is str)
            if phase == "observed":
                _require(kind == "exact-public-recovery" and effective == previous
                    and all(reconciliation[key] is None
                            for key in ("run_id", "run_attempt", "job_id")))
            else:
                _require(kind == "unstarted-deploy"
                         and effective == candidate["native"]["base"]["public_receipt_commit"])
                for key in ("run_id", "run_attempt", "job_id"):
                    _pattern(reconciliation[key], _DECIMAL)
                _require(reconciliation["run_id"] == str(candidate["run"]["id"])
                    and reconciliation["run_attempt"] == str(candidate["run"]["attempt"]))
    return candidate


def _pb_commit_raw(commit):
    _closed(commit, {"tree", "parents", "message", "author", "committer"})
    tree = _pattern(commit["tree"], _HEX40)
    parents = commit["parents"]
    _require(type(parents) is list and len(parents) <= 1)
    for parent in parents:
        _pattern(parent, _HEX40)
    message = commit["message"]
    _require(type(message) is str and 1 <= len(message) <= 256 and "\0" not in message)
    lines = ["tree " + tree] + ["parent " + parent for parent in parents]
    for role in ("author", "committer"):
        person = _closed(commit[role], {"name", "email", "date"})
        _require(type(person["name"]) is str and person["name"] == _PB_BOT_NAME
            and type(person["email"]) is str and person["email"] == _PB_BOT_EMAIL)
        delta = _pb_utc(person["date"]) - datetime(1970, 1, 1)
        epoch = delta.days * 86400 + delta.seconds
        lines.append(role + " " + _PB_BOT_NAME + " <" + _PB_BOT_EMAIL
                     + "> " + str(epoch) + " +0000")
    return ("\n".join(lines) + "\n\n" + message).encode("utf-8")


def _pb_proof(value):
    _closed(value, {"schema_version", "records"})
    _integer(value["schema_version"], 1, 1)
    rows = value["records"]
    _require(type(rows) is list and len(rows) <= 17)
    index = {}
    previous = None
    for row in rows:
        _closed(row, {"head", "commit", "receipt_b64"})
        head = _pattern(row["head"], _HEX40)
        _require(previous is None or previous < head,
                 "publication proof heads must be unique and sorted")
        previous = head
        raw = _pb_base64_bytes(row["receipt_b64"], _MIB)
        receipt = _parse(raw, _MIB)
        candidate = _pb_receipt(receipt)
        blob = _pb_git_sha("blob", raw)
        tree = _pb_git_sha("tree", b"100644 latest.json\0" + bytes.fromhex(blob))
        commit_raw = _pb_commit_raw(row["commit"])
        _require(row["commit"]["tree"] == tree,
                 "publication receipt tree integrity refused")
        expected_parents = ([] if receipt["previous_receipt_commit"] is None else
                            [receipt["previous_receipt_commit"]])
        _require(row["commit"]["parents"] == expected_parents,
                 "publication receipt parent integrity refused")
        _require(head in (_pb_git_sha("commit", commit_raw),
                          _pb_git_sha("commit", commit_raw + b"\n")),
                 "publication receipt commit integrity refused")
        index[head] = {"receipt": receipt, "candidate": candidate,
                       "blob": blob, "tree": tree}
    return index


def _pb_anchor(value, index):
    if value is None:
        return None
    _closed(value, {"head", "receipt"})
    head = _pattern(value["head"], _HEX40)
    _require(head in index and _same(value["receipt"], index[head]["receipt"]),
             "publication anchor differs from original proof")
    return head


def _pb_effective_native(record):
    if record is None or record["candidate"]["schema_version"] == 1:
        return 0, None
    native = record["candidate"]["native"]
    return native["sequence"], native["registry"]["sha256"]


def _pb_graph(index, operational_head, public_head, operation):
    """Derive material closure; final public history is a termination point."""
    used = set()
    edges = set()
    active = set()
    resolved = {}

    def take(head):
        _require(head in index, "missing publication proof witness")
        used.add(head)
        return index[head]

    def edge(source, target):
        if target is not None:
            take(target)
            edges.add((source, target))
            _require(len(edges) <= 16, "publication material edge capacity refused")

    def captured(prepared_head):
        token = ("captured", prepared_head)
        _require(token not in active, "publication proof cycle refused")
        active.add(token)
        prepared = take(prepared_head)
        receipt = prepared["receipt"]
        _require(receipt["phase"] == "prepared" and receipt["schema_version"] == 2)
        base = prepared["candidate"]["native"]["base"]
        captured_head = base["receipt_commit"]
        expected_public = base["public_receipt_commit"]
        edge(prepared_head, captured_head)
        edge(prepared_head, expected_public)
        actual_parent = receipt["previous_receipt_commit"]
        if captured_head is None:
            _require(actual_parent is None and expected_public is None)
            effective = None
        elif actual_parent == captured_head:
            effective = resolve(captured_head)
        else:
            _require(actual_parent is not None)
            bridge = take(actual_parent)
            edge(prepared_head, actual_parent)
            old = take(captured_head)
            bridge_receipt = bridge["receipt"]
            _require(old["receipt"]["phase"] == "prepared"
                and bridge_receipt["previous_receipt_commit"] == captured_head
                and _same(bridge["candidate"], old["candidate"]),
                "publication captured bridge conservation refused")
            edge(actual_parent, captured_head)
            reconciliation = bridge_receipt.get("reconciliation")
            _require(type(reconciliation) is dict)
            if bridge_receipt["phase"] == "observed":
                _require(reconciliation["kind"] == "exact-public-recovery"
                    and reconciliation["effective_public_receipt_commit"] == captured_head)
                effective = captured_head
            else:
                _require(bridge_receipt["phase"] == "superseded"
                    and reconciliation["kind"] == "unstarted-deploy")
                effective = resolve(actual_parent)
        _require(effective == expected_public, "publication captured public base refused")
        sequence, registry_hash = _pb_effective_native(
            None if effective is None else take(effective))
        _require(base["sequence"] == sequence and base["registry_sha256"] == registry_hash,
                 "publication captured native lineage refused")
        active.remove(token)
        return effective

    def resolve(head):
        if head is None:
            return None
        if head in resolved:
            return resolved[head]
        token = ("resolve", head)
        _require(token not in active, "publication proof cycle refused")
        active.add(token)
        current = take(head)
        receipt = current["receipt"]
        if receipt["phase"] in ("observed", "prepared"):
            effective = head
        else:
            prepared_head = receipt["previous_receipt_commit"]
            prepared = take(prepared_head)
            edge(head, prepared_head)
            _require(prepared["receipt"]["phase"] == "prepared"
                and prepared["receipt"]["schema_version"] == 2
                and _same(current["candidate"], prepared["candidate"]),
                "publication supersession candidate conservation refused")
            effective = captured(prepared_head)
            _require(receipt["reconciliation"]["effective_public_receipt_commit"] == effective)
            edge(head, effective)
        active.remove(token)
        resolved[head] = effective
        return effective

    if operational_head is None:
        _require(public_head is None and operation == "none" and not index)
        effective = None
    else:
        current = take(operational_head)
        phase = current["receipt"]["phase"]
        if phase == "prepared":
            if operation == "observe-prepared":
                effective = operational_head
            else:
                _require(operation == "supersede-unstarted")
                effective = captured(operational_head)
        else:
            _require(operation == "none")
            effective = resolve(operational_head)
        _require(effective == public_head, "publication effective anchor refused")
    if effective is not None:
        _require(take(effective)["receipt"]["phase"] in ("prepared", "observed"))
    _require(used == set(index), "unused or disconnected publication proof record")
    return effective


def _pb_effective_registry(raw, public):
    if public is None or public["candidate"]["schema_version"] == 1:
        _require(raw is None)
        return None
    _require(type(raw) is bytes)
    native = public["candidate"]["native"]
    _require(len(raw) == native["registry"]["bytes"]
        and _sha(raw) == native["registry"]["sha256"], "original publication registry pin refused")
    registry = validate_native_registry(raw)
    _require(registry["sequence"] == native["sequence"]
        and registry["previous_registry_sha256"] == native["base"]["registry_sha256"]
        and registry["current_artifact_sha256"] == native["binding"]["artifact_sha256"])
    current = next(row for row in registry["entries"]
                   if row["binding"]["artifact_sha256"] == registry["current_artifact_sha256"])
    _require(_same(current["binding"], native["binding"])
        and _same(current["selection"], native["selection"]),
        "publication registry current identity refused")
    return registry



def _pb_no_start(value, pending, head, consumer):
    """Validate complete recorded no-start structure, never captured programs."""
    _require(pending is not None and pending["receipt"]["schema_version"] == 2
             and pending["receipt"]["phase"] == "prepared")
    candidate = pending["candidate"]
    source = candidate["source"]["commit"]
    run_id, attempt = str(candidate["run"]["id"]), str(candidate["run"]["attempt"])
    _require(consumer != source, "same-source pending cannot be superseded automatically")
    _closed(value, {"pending_receipt_head", "pending_source_commit", "run",
        "deploy_job", "guard_step", "pages_step", "protocol_sources", "response_sha256s"})
    _require(type(value["pending_receipt_head"]) is str and value["pending_receipt_head"] == head
        and type(value["pending_source_commit"]) is str and value["pending_source_commit"] == source)
    run = _closed(value["run"], {"id", "run_attempt", "head_sha", "workflow_id", "status", "conclusion"})
    for key in ("id", "run_attempt", "workflow_id"):
        _pattern(run[key], _DECIMAL)
    _require(run["id"] == run_id and run["run_attempt"] == attempt
        and type(run["head_sha"]) is str and run["head_sha"] == source
        and type(run["status"]) is str and run["status"] == "completed"
        and type(run["conclusion"]) is str and run["conclusion"] == "failure")
    job = _closed(value["deploy_job"], {"id", "run_id", "run_attempt", "head_sha",
        "name", "status", "conclusion", "started_at", "completed_at"})
    for key in ("id", "run_id", "run_attempt"):
        _pattern(job[key], _DECIMAL)
    _require(job["run_id"] == run_id and job["run_attempt"] == attempt
        and type(job["head_sha"]) is str and job["head_sha"] == source
        and type(job["name"]) is str
        and type(job["status"]) is str and job["status"] == "completed"
        and type(job["conclusion"]) is str and job["conclusion"] == "failure")
    steps = []
    for key, name, conclusion in (
            ("guard_step", "Verify native prepared state and current Main before Pages", "failure"),
            ("pages_step", "Deploy the verified artifact", "skipped")):
        step = _closed(value[key], {"number", "name", "status", "conclusion", "started_at", "completed_at"})
        _require(type(step["number"]) is int and step["number"] > 0
            and type(step["name"]) is str and step["name"] == name
            and type(step["status"]) is str and step["status"] == "completed"
            and type(step["conclusion"]) is str and step["conclusion"] == conclusion)
        steps.append(step)
    _require(steps[0]["number"] < steps[1]["number"])
    for row in (job, *steps):
        for key in ("started_at", "completed_at"):
            if row[key] is not None:
                _pb_utc(row[key])
    protocol = _closed(value["protocol_sources"], {"consumer_commit", "pending_commit",
        "workflow_path", "workflow_blob_sha", "workflow_sha256", "workflow_bytes_b64",
        "guard_path", "guard_blob_sha", "guard_sha256", "guard_bytes_b64",
        "contents_response_sha256s"})
    _require(type(protocol["consumer_commit"]) is str and protocol["consumer_commit"] == consumer
        and type(protocol["pending_commit"]) is str and protocol["pending_commit"] == source)
    for kind, path in (("workflow", _PB_WORKFLOW), ("guard", _PB_GUARD)):
        _require(type(protocol[kind + "_path"]) is str and protocol[kind + "_path"] == path)
        _pattern(protocol[kind + "_blob_sha"], _HEX40)
        _pattern(protocol[kind + "_sha256"], _HEX64)
        raw = _pb_base64_bytes(protocol[kind + "_bytes_b64"], 262144, minimum=0)
        _require(_sha(raw) == protocol[kind + "_sha256"]
            and _pb_git_sha("blob", raw) == protocol[kind + "_blob_sha"],
            "publication protocol source body pin refused")
    contents_hashes = protocol["contents_response_sha256s"]
    _require(type(contents_hashes) is list and len(contents_hashes) == 4)
    for digest in contents_hashes:
        _pattern(digest, _HEX64)
    hashes = value["response_sha256s"]
    _require(type(hashes) is list and 1 <= len(hashes) <= 64)
    for digest in hashes:
        _pattern(digest, _HEX64)
    return value


def _pb_hash_counts(values):
    result = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _pb_metadata(value, index, consumer, pending, no_start):
    _closed(value, {"logical_calls", "physical_exchanges", "received_bytes", "elapsed_ms", "responses"})
    logical = _integer(value["logical_calls"], 0, 64)
    physical = _integer(value["physical_exchanges"], 0, 64)
    received = _integer(value["received_bytes"], 0, 16 * _MIB)
    _integer(value["elapsed_ms"], 0, 120000)
    rows = value["responses"]
    _require(type(rows) is list and len(rows) <= 64
        and len(rows) <= logical <= physical)
    expected_git = {"receipt-commit": set(index),
        "receipt-tree": {row["tree"] for row in index.values()},
        "receipt-blob": {row["blob"] for row in index.values()}}
    seen_git = {key: set() for key in expected_git}
    receipt_refs = main_refs = total_bytes = exchanges = 0
    pending_source = None if pending is None else pending["candidate"]["source"]["commit"]
    run_id = None if pending is None else str(pending["candidate"]["run"]["id"])
    attempt = None if pending is None else str(pending["candidate"]["run"]["attempt"])
    source_refs = {consumer} | (set() if pending_source is None else {pending_source})
    contents = {}
    jobs_pages = set()
    run_rows = 0
    for row in rows:
        _closed(row, {"route_kind", "locator", "bytes", "sha256", "http_exchanges"})
        kind, locator = row["route_kind"], row["locator"]
        _require(type(kind) is str and type(locator) is str)
        size = _integer(row["bytes"], 1, _MIB)
        _pattern(row["sha256"], _HEX64)
        exchanges += _integer(row["http_exchanges"], 1, 4)
        total_bytes += size
        if kind == "receipt-ref":
            _require(locator == "git/ref/heads/" + _PB_BRANCH)
            receipt_refs += 1
        elif kind == "main-ref":
            _require(locator == "git/ref/heads/main")
            main_refs += 1
        elif kind in expected_git:
            collection = {"receipt-commit": "commits", "receipt-tree": "trees",
                          "receipt-blob": "blobs"}[kind]
            prefix = "git/" + collection + "/"
            _require(locator.startswith(prefix))
            identity = _pattern(locator[len(prefix):], _HEX40)
            _require(identity in expected_git[kind], "publication metadata object join refused")
            seen_git[kind].add(identity)
        elif kind in ("workflow-contents", "guard-contents"):
            path = _PB_WORKFLOW if kind == "workflow-contents" else _PB_GUARD
            prefix = "contents/" + path + "?ref="
            _require(locator.startswith(prefix))
            identity = _pattern(locator[len(prefix):], _HEX40)
            _require(identity in source_refs, "publication metadata source ref refused")
            contents.setdefault((kind, identity), []).append(row)
        elif kind == "pending-run":
            _require(run_id is not None and locator == "actions/runs/" + run_id)
            run_rows += 1
        elif kind == "pending-jobs":
            _require(run_id is not None)
            prefix = "actions/runs/" + run_id + "/attempts/" + attempt + "/jobs?per_page=50&page="
            _require(locator.startswith(prefix) and locator[len(prefix):] in ("1", "2", "3", "4"))
            page = int(locator[len(prefix):])
            jobs_pages.add(page)
        else:
            raise ValueError("publication metadata route refused")
    _require(receipt_refs >= 2 and main_refs >= 2
        and exchanges <= physical and total_bytes <= received,
        "publication recorded metadata counters or ref guards refused")
    _require(all(seen_git[key] == identities for key, identities in expected_git.items()),
             "publication metadata proof source closure refused")
    if jobs_pages:
        _require(jobs_pages == set(range(1, max(jobs_pages) + 1)))
    if no_start is not None:
        required = [(kind, ref) for ref in (pending_source, consumer)
                    for kind in ("workflow-contents", "guard-contents")]
        _require(len(set(required)) == 4 and set(contents) == set(required)
            and all(len(contents[key]) == 1 for key in required),
            "publication four immutable source locator roles refused")
        contents_rows = [contents[key][0] for key in required]
        contents_hashes = [row["sha256"] for row in contents_rows]
        _require(_pb_hash_counts(contents_hashes) == _pb_hash_counts(
            no_start["protocol_sources"]["contents_response_sha256s"]),
            "publication source response hash vector refused")
        _require(run_rows >= 1 and jobs_pages and len(jobs_pages) <= 4,
                 "publication no-start Actions source closure refused")
        all_hashes = {row["sha256"] for row in rows}
        _require(all(digest in all_hashes for digest in no_start["response_sha256s"]),
            "publication no-start native response hashes refused")


def _pb_public_observation(value, public, registry):
    _closed(value, {"target_receipt_head", "target_candidate_sha256", "scope",
        "records", "requests", "received_bytes", "elapsed_ms"})
    requests = _integer(value["requests"], 0, 201)
    received = _integer(value["received_bytes"], 0, 128 * _MIB)
    _integer(value["elapsed_ms"], 0, 660000)
    rows = value["records"]
    _require(type(rows) is list)
    expected = {}
    native_paths = set()
    if public is None:
        _require(value["target_receipt_head"] is None and value["target_candidate_sha256"] is None
            and type(value["scope"]) is str and value["scope"] == "initial-absence5")
        for path in {"snapshots/index.json", "current-meta.json"} | _PB_ABSENCE_PATHS:
            expected[path] = (404, 0, None, 0)
    else:
        head, record = public
        candidate = record["candidate"]
        _require(type(value["target_receipt_head"]) is str and value["target_receipt_head"] == head
            and type(value["target_candidate_sha256"]) is str
            and value["target_candidate_sha256"] == _sha(_canonical(candidate)))
        for field, path in _PB_LEGACY_PATHS.items():
            descriptor = candidate[field]
            expected[path] = (200, descriptor["bytes"], descriptor["sha256"], 16 * _MIB)
        if candidate["schema_version"] == 1:
            # This API is the native-required base admission boundary; migration
            # needs actual absence for all three native root pointers.
            _require(type(value["scope"]) is str and value["scope"] == "legacy3+native-absence3")
            for path in _PB_ABSENCE_PATHS:
                expected[path] = (404, 0, None, 0)
        else:
            _require(type(value["scope"]) is str and value["scope"] == "current9"
                and registry is not None)
            native = candidate["native"]
            for field, maximum in (("registry", _MAX_REGISTRY), ("build_manifest", _MAX_MANIFEST)):
                descriptor = native[field]
                expected[descriptor["path"]] = (200, descriptor["bytes"], descriptor["sha256"], maximum)
            current = next(row for row in registry["entries"]
                           if row["binding"]["artifact_sha256"] == registry["current_artifact_sha256"])
            for descriptor in current["files"]:
                path = descriptor["path"]
                expected[path] = (200, descriptor["bytes"], descriptor["sha256"], _MAX_ASSET)
                native_paths.add(path)
            # The handoff has the page observation descriptor, not its body or
            # the public build-manifest body. F1 owns their actual admission.
            expected["proof-tree.html"] = (200, None, None, _MAX_PAGE)
            native_paths.update({"proof-tree.html", "current-build-manifest.json"})
    _require(len(rows) == len(expected) and len(rows) <= requests)
    previous = None
    total_bytes = native_bytes = 0
    for row in rows:
        _closed(row, {"path", "status", "bytes", "sha256"})
        path = row["path"]
        _require(type(path) is str and path in expected
            and (previous is None or previous < path), "publication final closure paths refused")
        previous = path
        status, length, digest, maximum = expected[path]
        _integer(row["status"], status, status)
        if status == 404:
            _integer(row["bytes"], 0, 0)
            _require(row["sha256"] is None)
        else:
            size = _integer(row["bytes"], 1, maximum)
            _pattern(row["sha256"], _HEX64)
            if length is not None:
                _require(size == length and row["sha256"] == digest,
                         "publication observed descriptor join refused")
            total_bytes += size
            if path in native_paths:
                native_bytes += size
    _require(total_bytes <= received and native_bytes <= _MAX_BUNDLE,
             "publication recorded public byte counters refused")


def _pb_validate(files, native_anchor):
    _require(type(files) is dict and all(type(key) is str for key in files),
             "publication base bytes map required")
    _require(set(files) in (_PB_FILES, _PB_FILES | {"registry.json"}),
             "publication base file membership refused")
    total = 0
    for name, raw in files.items():
        _require(type(raw) is bytes and 0 < len(raw) <= _PB_FILE_CAPS[name],
                 "publication base file capacity refused")
        total += len(raw)
    _require(total <= 64 * _MIB, "publication private base aggregate refused")
    anchor = _pb_anchor_json(files["anchor.json"])
    public_anchor = _pb_anchor_json(files["public-anchor.json"])
    base = _closed(_parse(files["base.json"], 65536), _PB_BASE_FIELDS)
    plan = _closed(_parse(files["pending-plan.json"], 65536), _PB_PLAN_FIELDS)
    proof = _parse(files["git-proof.json"], 16 * _MIB)
    qualification = _closed(_parse(files["qualification.json"], 16 * _MIB),
                             _PB_QUALIFICATION_FIELDS)
    index = _pb_proof(proof)
    head = _pb_anchor(anchor, index)
    effective_head = _pb_anchor(public_anchor, index)
    _json_types(native_anchor)
    _pb_anchor(native_anchor, index)
    _require(_same(native_anchor, anchor), "fresh native publication anchor changed")
    _integer(qualification["schema_version"], 1, 1)
    _require(type(qualification["repository"]) is str and qualification["repository"] == _REPOSITORY)
    consumer = _pattern(qualification["consumer_commit"], _HEX40)
    operation = qualification["operation"]
    _require(type(operation) is str and operation in ("none", "observe-prepared", "supersede-unstarted"))
    for field in ("initial_receipt_head", "final_receipt_head"):
        _require(_pb_nullable_head(qualification[field]) == head,
                 "publication operational ref fence changed")
    for field in ("initial_main_commit", "final_main_commit"):
        _require(_pattern(qualification[field], _HEX40) == consumer,
                 "publication Main fence changed")
    _require(_pb_nullable_head(qualification["target_public_head"]) == effective_head)
    _pb_graph(index, head, effective_head, operation)
    public = None if effective_head is None else index[effective_head]
    native = public is not None and public["candidate"]["schema_version"] == 2
    _require(("registry.json" in files) == native, "publication effective registry membership refused")
    registry_raw = files.get("registry.json")
    registry = _pb_effective_registry(registry_raw, public)
    current = None if head is None else index[head]
    pending = current if current is not None and current["receipt"]["phase"] == "prepared" else None
    if operation == "supersede-unstarted":
        no_start = _pb_no_start(qualification["no_start"], pending, head, consumer)
    else:
        _require(qualification["no_start"] is None,
                 "no-start qualification is only for current supersede-unstarted")
        no_start = None
    _pb_metadata(qualification["metadata"], index, consumer, pending, no_start)
    _pb_public_observation(qualification["public_observation"],
        None if public is None else (effective_head, public), registry)
    # The chosen-public phase occurs inside the resolver qualification scope;
    # both declared clocks must respect that intersection, not just two caps.
    _require(qualification["public_observation"]["elapsed_ms"]
        <= qualification["metadata"]["elapsed_ms"],
        "publication intersecting qualification duration refused")
    sequence, registry_hash = _pb_effective_native(public)
    expected_base = {"schema_version": 1, "receipt_commit": head,
        "public_receipt_commit": effective_head,
        "receipt_phase": None if current is None else current["receipt"]["phase"],
        "native_sequence": sequence, "registry_sha256": registry_hash,
        "mode": "initial" if public is None else "native-v1" if native else "legacy-v1"}
    _require(_same(base, expected_base), "publication derived base mismatch")
    served = [] if public is None else list(public["candidate"]["snapshot_ids"])
    unserved = (sorted(set(pending["candidate"]["snapshot_ids"]) - set(served))
                if operation == "supersede-unstarted" else [])
    expected_plan = {"schema_version": 1, "operation": operation,
        "operational_head": head, "effective_public_head": effective_head,
        "pending_prepared_head": head if operation in ("observe-prepared", "supersede-unstarted") else None,
        "consumer_commit": consumer, "git_proof_sha256": _sha(files["git-proof.json"]),
        "qualification_sha256": _sha(files["qualification.json"]),
        "served_snapshot_ids": served, "unserved_snapshot_ids": unserved}
    _require(_same(plan, expected_plan), "publication derived pending plan mismatch")
    return {"anchor": anchor, "public_anchor": public_anchor,
        "registry_raw": registry_raw, "base": base, "disposition": plan, "git_proof": proof}


def validate_publication_base(files: dict[str, bytes], *, native_anchor: dict | None) -> dict:
    """Authenticate a complete captured publication base without IO or mutation.

    The six returned fields conserve admitted original buffers and recorded
    disposition. Fresh fixed-ref provenance, actual HTTP/Actions observations,
    functional source qualification and private filesystem custody are provided
    by the Source-owned resolver, not attested by this pure parser.
    """
    try:
        return _pb_validate(files, native_anchor)
    except (UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError("publication base value refused") from error
