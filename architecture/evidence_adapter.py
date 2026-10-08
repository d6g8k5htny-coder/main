"""Validate inert evidence packets without authenticating custody or science.

All current/stale decisions concern declared records inside the packet. Only the
independently pinned repository graph gate is executable; captures are data.
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import ModuleType
from typing import Any


MAX_PACKET_BYTES = 16 * 1024 * 1024
MAX_CAPTURE_BYTES = 4 * 1024 * 1024
MAX_DEPTH = 64
TRUSTED_GATE_SHA256 = "a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8"
ROOT = Path(__file__).resolve().parents[1]
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
DECIMAL = re.compile(r"[1-9][0-9]*")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.']*")
AXIOM_LINE = re.compile(
    r"'([A-Za-z_][A-Za-z0-9_.']*)' "
    r"(?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)"
)
ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
MAIN_TOOLCHAIN = "leanprover/lean4:v4.34.1"
MAIN_VERSION = re.compile(r"\bversion 4\.34\.1(?=[,\s)])")
STATES = frozenset({"recorded", "not_recorded", "unknown", "not_applicable"})
AXES = ("source", "review", "kernel", "computation", "alignment")
PLACEHOLDERS = frozenset({
    "unknown", "unverified", "not known", "not available", "not provided", "unavailable", "unspecified",
    "pending", "tbd", "todo", "none", "null", "n/a", "na", "?", "-",
})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    require(type(value) is dict and set(value) == expected, label + ": invalid fields")
    return value


def text(value: Any, label: str) -> str:
    require(type(value) is str and bool(value.strip()), label + ": expected nonempty string")
    return value


def digest(value: Any, pattern: re.Pattern[str], label: str) -> str:
    require(type(value) is str and pattern.fullmatch(value) is not None,
            label + ": invalid identity")
    return value


def repository(value: Any, label: str) -> str:
    digest(value, REPOSITORY, label)
    require(all(part not in (".", "..") for part in value.split("/")),
            label + ": invalid repository")
    return value


def relative_path(value: Any, label: str) -> str:
    text(value, label)
    require(not value.startswith("/") and "\\" not in value
            and not any(ord(char) < 32 or ord(char) == 127 for char in value),
            label + ": unsafe path")
    require(all(part not in ("", ".", "..") for part in value.split("/"))
            and PurePosixPath(value).as_posix() == value,
            label + ": nonnormalized path")
    return value


def positive_decimal(value: Any, label: str) -> str:
    return digest(value, DECIMAL, label)


def _depth_and_types(value: Any, depth: int = 0) -> None:
    require(depth <= MAX_DEPTH, "JSON nesting exceeds limit")
    if type(value) is dict:
        require(all(type(key) is str for key in value), "nonstring JSON key")
        for child in value.values():
            _depth_and_types(child, depth + 1)
    elif type(value) is list:
        for child in value:
            _depth_and_types(child, depth + 1)
    elif type(value) is float:
        require(math.isfinite(value), "nonfinite JSON number")
    else:
        require(value is None or type(value) in (str, int, bool), "non-JSON value")


def strict_json(raw: bytes, label: str, *, limit: int = MAX_CAPTURE_BYTES) -> Any:
    require(type(raw) is bytes and len(raw) <= limit, label + ": JSON byte limit")
    decoded = raw.decode("utf-8")
    # Bound nesting before the decoder recurses, accounting for escaped strings.
    depth = 0
    quoted = False
    escaped = False
    for char in decoded:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            require(depth <= MAX_DEPTH, label + ": JSON nesting exceeds limit")
        elif char in "]}":
            depth -= 1

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, label + ": duplicate JSON key")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ValueError(label + ": nonfinite JSON constant")

    def floating(value: str) -> float:
        result = float(value)
        require(math.isfinite(result), label + ": nonfinite JSON number")
        return result

    result = json.loads(decoded, object_pairs_hook=unique,
                        parse_constant=constant, parse_float=floating)
    _depth_and_types(result)
    return result


def parse_packet(raw: bytes) -> dict[str, Any]:
    result = strict_json(raw, "packet", limit=MAX_PACKET_BYTES)
    require(type(result) is dict, "packet must be object")
    return result


def validate_expected_context(value: dict[str, str]) -> dict[str, str]:
    keys(value, {"repository", "commit", "run-id", "run-attempt"}, "expected context")
    repository(value["repository"], "expected repository")
    digest(value["commit"], HEX40, "expected commit")
    positive_decimal(value["run-id"], "expected run id")
    positive_decimal(value["run-attempt"], "expected run attempt")
    return value


@dataclass(frozen=True)
class Capture:
    ref: dict[str, Any]
    raw: bytes

    def document(self, label: str) -> dict[str, Any]:
        result = strict_json(self.raw, label)
        require(type(result) is dict, label + ": expected JSON object")
        return result


class Captures:
    """Verified identities shared across every capture role in one packet."""

    def __init__(self) -> None:
        self.git_identities: dict[tuple[str, str, str], Capture] = {}
        self.run_identities: dict[tuple[str, ...], Capture] = {}

    @staticmethod
    def raw(value: dict[str, Any], label: str) -> bytes:
        encoded = value["raw_base64"]
        require(type(encoded) is str and len(encoded) <= 4 * ((MAX_CAPTURE_BYTES + 2) // 3),
                label + ": encoded capture exceeds limit")
        try:
            raw = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeError, binascii.Error, ValueError) as exc:
            raise ValueError(label + ": invalid base64") from exc
        require(len(raw) <= MAX_CAPTURE_BYTES
                and base64.b64encode(raw).decode("ascii") == encoded,
                label + ": noncanonical or oversized base64")
        return raw

    @staticmethod
    def identity_bytes(ref: dict[str, Any], raw: bytes, label: str) -> None:
        require(type(ref["bytes"]) is int and 0 <= ref["bytes"] <= MAX_CAPTURE_BYTES,
                label + ": invalid byte count")
        digest(ref["sha256"], HEX64, label + " SHA256")
        require(ref["bytes"] == len(raw)
                and ref["sha256"] == hashlib.sha256(raw).hexdigest(),
                label + ": raw byte identity mismatch")

    def git(self, value: Any, label: str) -> Capture:
        keys(value, {"ref", "raw_base64"}, label)
        ref = keys(value["ref"], {"repository", "commit", "path", "git_blob", "sha256", "bytes"},
                   label + " GitRef")
        repository(ref["repository"], label + " repository")
        digest(ref["commit"], HEX40, label + " commit")
        relative_path(ref["path"], label + " path")
        digest(ref["git_blob"], HEX40, label + " blob")
        raw = self.raw(value, label)
        self.identity_bytes(ref, raw, label)
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw,
                            usedforsecurity=False).hexdigest()
        require(ref["git_blob"] == blob, label + ": Git blob identity mismatch")
        capture = Capture(ref, raw)
        identity = (ref["repository"], ref["commit"], ref["path"])
        previous = self.git_identities.get(identity)
        require(previous is None or (previous.raw == raw and canonical(previous.ref) == canonical(ref)),
                label + ": conflicting immutable Git capture")
        self.git_identities[identity] = capture
        return capture

    def run(self, value: Any, native: dict[str, Any], label: str) -> Capture:
        keys(value, {"ref", "raw_base64"}, label)
        ref = keys(value["ref"], {"repository", "checked_commit", "run_id", "run_attempt",
                                  "artifact_id", "member_path", "sha256", "bytes"}, label + " RunRef")
        repository(ref["repository"], label + " repository")
        digest(ref["checked_commit"], HEX40, label + " checked commit")
        for key in ("run_id", "run_attempt", "artifact_id"):
            positive_decimal(ref[key], label + " " + key)
        relative_path(ref["member_path"], label + " member path")
        require(all(ref[key] == native[key] for key in
                    ("repository", "checked_commit", "run_id", "run_attempt")),
                label + ": native run binding contradiction")
        raw = self.raw(value, label)
        self.identity_bytes(ref, raw, label)
        capture = Capture(ref, raw)
        identity = tuple(ref[key] for key in ("repository", "checked_commit", "run_id", "run_attempt",
                                              "artifact_id", "member_path"))
        previous = self.run_identities.get(identity)
        require(previous is None or (previous.raw == raw and canonical(previous.ref) == canonical(ref)),
                label + ": conflicting native artifact member")
        self.run_identities[identity] = capture
        return capture


def _gate() -> ModuleType:
    """Read one fixed regular file through anchored no-follow descriptors."""
    descriptor = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in ("docs", "site", "dependency-source"):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open("hard_gate.py", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                  dir_fd=descriptor)
        try:
            info = os.fstat(file_descriptor)
            require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_CAPTURE_BYTES,
                    "trusted gate is not a bounded regular file")
            parts = []
            remaining = MAX_CAPTURE_BYTES + 1
            while remaining:
                part = os.read(file_descriptor, min(remaining, 65536))
                if not part:
                    break
                parts.append(part)
                remaining -= len(part)
            raw = b"".join(parts)
            require(len(raw) <= MAX_CAPTURE_BYTES and hashlib.sha256(raw).hexdigest() == TRUSTED_GATE_SHA256,
                    "trusted gate SHA256 pin mismatch")
        finally:
            os.close(file_descriptor)
    finally:
        os.close(descriptor)
    path = ROOT / "docs/site/dependency-source/hard_gate.py"
    module = ModuleType("_verified_architecture_evidence_graph_gate")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def _slice(value: Any, source: Capture | None, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    keys(value, {"start_byte", "end_byte", "sha256"}, label)
    require(source is not None, label + ": slice without source")
    start, end = value["start_byte"], value["end_byte"]
    require(type(start) is int and type(end) is int and 0 <= start < end <= len(source.raw),
            label + ": invalid byte bounds")
    digest(value["sha256"], HEX64, label + " SHA256")
    require(hashlib.sha256(source.raw[start:end]).hexdigest() == value["sha256"],
            label + ": byte slice hash mismatch")
    return value


def _snapshot(value: Any, captures: Captures, gate: ModuleType, label: str) -> dict[str, Any]:
    keys(value, {"graph", "bindings"}, label)
    graph_capture = captures.git(value["graph"], label + " graph")
    graph = graph_capture.document(label + " graph")
    gate.validate_graph_fail_closed(graph)
    require(type(value["bindings"]) is list, label + ": bindings must be array")
    bindings = {}
    for binding in value["bindings"]:
        keys(binding, {"node", "source", "statement_slice", "proof_slice"}, label + " binding")
        node = text(binding["node"], label + " node")
        require(node in graph["nodes"] and node not in bindings, label + ": unknown or duplicate node binding")
        source = None if binding["source"] is None else captures.git(binding["source"], label + " source")
        statement = _slice(binding["statement_slice"], source, label + " statement slice")
        proof = _slice(binding["proof_slice"], source, label + " proof slice")
        bindings[node] = {"source": source, "statement_slice": statement, "proof_slice": proof}
    return {"graph": graph, "graph_capture": graph_capture, "bindings": bindings}


def _native(value: Any) -> dict[str, Any]:
    keys(value, {"repository", "checked_commit", "run_head_sha", "run_id", "run_attempt",
                 "purpose", "conclusion", "expected_conclusion"}, "native run")
    repository(value["repository"], "native repository")
    digest(value["checked_commit"], HEX40, "native checked commit")
    digest(value["run_head_sha"], HEX40, "native run head")
    positive_decimal(value["run_id"], "native run id")
    positive_decimal(value["run_attempt"], "native run attempt")
    require(value["purpose"] in ("check", "negative_control"), "unknown native run purpose")
    conclusions = {"success", "failure", "skipped", "cancelled", "timed_out", "action_required", "neutral", "stale"}
    require(type(value["conclusion"]) is str and value["conclusion"] in conclusions,
            "unknown native conclusion")
    require(type(value["expected_conclusion"]) is str and value["expected_conclusion"] in conclusions,
            "unknown expected native conclusion")
    return value


def _same_context(native: dict[str, Any], expected: dict[str, str]) -> bool:
    return (native["repository"] == expected["repository"]
            and native["checked_commit"] == expected["commit"]
            and native["run_id"] == expected["run-id"]
            and native["run_attempt"] == expected["run-attempt"])


def _science(document: dict[str, Any], label: str, *, main: bool) -> None:
    require(type(document.get("schema_version")) is int and document["schema_version"] == 1
            and document.get("scientific_effect") == "NONE", label + ": invalid native schema/effect")
    if main or "scientific_status_authority" in document:
        require(document.get("scientific_status_authority") is False,
                label + ": scientific authority must be false")


def _axioms(value: Any, targets: set[str], label: str) -> dict[str, list[str]]:
    require(type(value) is dict, label + ": expected axiom object")
    result = {}
    for target, axioms in value.items():
        require(target in targets and type(axioms) is list and all(type(item) is str for item in axioms),
                label + ": foreign target or malformed axiom list")
        require(len(axioms) == len(set(axioms)) and set(axioms) <= ALLOWED_AXIOMS,
                label + ": duplicate or forbidden axiom")
        result[target] = sorted(axioms)
    return result


def _axiom_dump(raw: bytes, targets: set[str]) -> dict[str, list[str]]:
    decoded = raw.decode("utf-8")
    result = {}
    for match in AXIOM_LINE.finditer(decoded):
        target, listed = match.groups()
        require(target in targets and target not in result, "foreign or duplicate original axiom target")
        axioms = [] if listed is None or not listed.strip() else [item.strip() for item in listed.split(",")]
        result.update(_axioms({target: axioms}, targets, "original axioms log"))
    require(not AXIOM_LINE.sub("", decoded).strip(), "unrecognized original axiom output")
    return result


def _target_inventory(manifest: dict[str, Any], main: bool) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    targets = manifest.get("targets")
    require(type(targets) is list and bool(targets), "empty native target inventory")
    rows = {}
    sources = {}
    if main:
        for row in targets:
            keys(row, {"name", "module", "title", "source", "informal_anchor", "does_not_claim"},
                 "main target")
            name = digest(row["name"], NAME, "main target name")
            require(name not in rows, "duplicate main target")
            for field in ("title", "source", "informal_anchor", "does_not_claim"):
                text(row[field], "main target " + field)
            relative_path(row["module"], "main target module")
            rows[name] = row
        source_rows = manifest.get("sources")
        require(type(source_rows) is list and bool(source_rows), "missing main source rows")
        for row in source_rows:
            keys(row, {"id", "repository", "commit", "path", "bytes", "sha256", "local_copy"},
                 "main source row")
            identity = text(row["id"], "main source id")
            require(identity not in sources, "duplicate main source id")
            repository(row["repository"], "main origin repository")
            digest(row["commit"], HEX40, "main origin commit")
            relative_path(row["path"], "main origin path")
            relative_path(row["local_copy"], "main local copy")
            digest(row["sha256"], HEX64, "main origin SHA256")
            require(type(row["bytes"]) is int and 0 < row["bytes"] <= MAX_CAPTURE_BYTES,
                    "main origin byte count")
            sources[identity] = row
        require(all(row["source"] in sources for row in rows.values()), "unknown main target source")
    else:
        for name in targets:
            digest(name, NAME, "Math target name")
            require(name not in rows, "duplicate Math target")
            rows[name] = None
    return sorted(rows), rows, sources


def _lineage(party: Any, *, normalized: bool) -> dict[str, str] | None:
    if type(party) is not dict:
        return None
    result = {}
    for field in ("provider", "family", "agent"):
        value = party.get(field)
        if type(value) is not str or not value.strip():
            return None
        normalized_value = " ".join(value.split()).casefold()
        if normalized_value in PLACEHOLDERS:
            return None
        compared = normalized_value if normalized else value.strip().casefold()
        result[field] = compared
    return result


def _alignment_reasons(document: dict[str, Any] | None, manifest: Capture, scope: Capture,
                       targets: list[str], *, main: bool, source_author: Any = None) -> list[str]:
    if document is None:
        return ["alignment_not_recorded"]
    reasons = []
    for field in ("manifest_sha256", "scope_sha256"):
        if field in document:
            digest(document[field], HEX64, "alignment " + field)
    if document.get("manifest_sha256") != manifest.ref["sha256"]:
        reasons.append("manifest_binding_changed")
    if document.get("scope_sha256") != scope.ref["sha256"]:
        reasons.append("scope_binding_changed")
    covered = document.get("targets")
    if not (type(covered) is list and all(type(item) is str for item in covered)
            and len(covered) == len(set(covered)) and set(covered) == set(targets)):
        reasons.append("target_coverage_changed")
    if document.get("disposition") != "ACCEPTED":
        reasons.append("alignment_not_accepted")
    reviewer = _lineage(document.get("reviewer"), normalized=not main)
    authors = [_lineage(document.get("author"), normalized=not main)]
    if main:
        retained_author = _lineage(source_author, normalized=False)
        if retained_author is None:
            reasons.append("manifest_author_unresolved")
        else:
            if authors[0] != retained_author:
                reasons.append("author_binding_changed")
            # A review cannot gain distinct lineage by relabeling its author.
            authors.append(retained_author)
    else:
        proposers = document.get("proposal_authors", [])
        if type(proposers) is not list:
            reasons.append("lineage_not_distinct")
        else:
            for proposer in proposers:
                authors.append(_lineage(proposer, normalized=True))
                if type(proposer) is dict and "targets" in proposer:
                    subset = proposer["targets"]
                    if not (type(subset) is list and bool(subset)
                            and all(type(item) is str and item in targets for item in subset)
                            and len(subset) == len(set(subset))):
                        reasons.append("target_coverage_changed")
    if reviewer is None or any(author is None or any(author[field] == reviewer[field]
                                                     for field in reviewer) for author in authors):
        reasons.append("lineage_not_distinct")
    evidence = document.get("evidence")
    if type(evidence) is not dict:
        reasons.append("alignment_evidence_unresolved")
    else:
        # These declarations are structurally checked, never fetched or credited as custody.
        try:
            repository(evidence.get("repository"), "alignment evidence repository")
            digest(evidence.get("commit"), HEX40, "alignment evidence commit")
            digest(evidence.get("sha256"), HEX64, "alignment evidence SHA256")
            relative_path(evidence.get("path"), "alignment evidence path")
        except ValueError:
            reasons.append("alignment_evidence_unresolved")
    return sorted(set(reasons))


def _formal(value: Any, captures: Captures, nodes: set[str], expected: dict[str, str]) -> dict[str, Any]:
    keys(value, {"id", "format", "manifest", "scope", "source_files", "receipt", "logs",
                 "alignment", "native_run", "node_targets"}, "formal record")
    identity = text(value["id"], "formal id")
    require(value["format"] in ("main-formal-gate/v1", "math-formal-gate/v1"), "unknown native format")
    main = value["format"] == "main-formal-gate/v1"
    native = _native(value["native_run"])
    manifest_capture = captures.git(value["manifest"], "formal manifest")
    scope = captures.git(value["scope"], "formal scope")
    manifest = manifest_capture.document("formal manifest")
    _science(manifest, "formal manifest", main=main)
    require(manifest.get("formalization_status") == "proved"
            and manifest.get("alignment_status") == "PENDING_INDEPENDENT_REVIEW",
            "native source metadata cannot award execution or review")
    if main:
        require(manifest.get("lean_toolchain") == MAIN_TOOLCHAIN,
                "unsupported main native Lean toolchain")
    targets, target_rows, source_rows = _target_inventory(manifest, main)
    target_set = set(targets)
    require(type(value["source_files"]) is list, "formal source_files must be array")
    sources = {}
    for original in value["source_files"]:
        capture = captures.git(original, "formal source file")
        require(capture.ref["path"] not in sources, "duplicate formal source file")
        sources[capture.ref["path"]] = capture
    for capture in [manifest_capture, scope, *sources.values()]:
        require(capture.ref["repository"] == native["repository"]
                and capture.ref["commit"] == native["checked_commit"],
                "native checked source binding contradiction")
    files = manifest.get("files")
    require(type(files) is dict and bool(files), "missing native file hashes")
    parent = str(PurePosixPath(manifest_capture.ref["path"]).parent)
    package_root = relative_path(manifest.get("package_root"), "main package root") if main else None
    scope_path = package_root + "/SCOPE.md" if main else "SCOPE.md" if parent == "." else parent + "/SCOPE.md"
    require(scope.ref["path"] == scope_path, "native scope must identify the original SCOPE.md")
    paths = {}
    for name, expected_hash in files.items():
        relative_path(name, "native file key")
        digest(expected_hash, HEX64, "native file SHA256")
        path = name if main or parent == "." else parent + "/" + name
        relative_path(path, "native source lookup")
        require(path not in paths, "duplicate native source lookup")
        paths[path] = expected_hash
        require(path in sources and sources[path].ref["sha256"] == expected_hash,
                "native source file missing or contradictory")
    require(scope.ref["path"] in paths and paths[scope.ref["path"]] == scope.ref["sha256"],
            "native scope is not bound by manifest files")
    if main:
        for row in source_rows.values():
            local = sources.get(row["local_copy"])
            require(local is not None and row["local_copy"] in paths
                    and local.ref["bytes"] == row["bytes"] and local.ref["sha256"] == row["sha256"],
                    "main checked local copy differs from pinned origin bytes")
        modules = manifest.get("source_modules")
        require(type(modules) is list and bool(modules) and all(type(item) is str for item in modules)
                and len(modules) == len(set(modules)), "invalid main source module inventory")
        for module in modules:
            relative_path(module, "main source module")
            require(package_root + "/" + module in paths, "unbound main source module")
        require(all(row["module"] in modules for row in target_rows.values()), "target module not registered")
        for row in target_rows.values():
            local = sources[source_rows[row["source"]]["local_copy"]]
            require(row["informal_anchor"].encode("utf-8") in local.raw,
                    "main target anchor absent from checked source")
    require(type(value["node_targets"]) is list, "node_targets must be array")
    mappings = []
    seen = set()
    for binding in value["node_targets"]:
        keys(binding, {"node", "target"}, "node target binding")
        require(type(binding["node"]) is str and binding["node"] in nodes
                and type(binding["target"]) is str and binding["target"] in target_set,
                "unknown explicit node or native target")
        pair = (binding["node"], binding["target"])
        require(pair not in seen, "duplicate explicit node target binding")
        seen.add(pair)
        mappings.append(binding)
    require(type(value["logs"]) is list, "formal logs must be array")
    logs = {}
    members = []
    for original in value["logs"]:
        capture = captures.run(original, native, "formal log")
        name = PurePosixPath(capture.ref["member_path"]).name
        require(name not in logs, "duplicate native log filename")
        logs[name] = capture
        members.append(capture)
    receipt_capture = None if value["receipt"] is None else captures.run(value["receipt"], native, "formal receipt")
    receipt = None if receipt_capture is None else receipt_capture.document("formal receipt")
    if receipt_capture is not None:
        members.append(receipt_capture)
    require(len({capture.ref["artifact_id"] for capture in members}) <= 1,
            "native receipt/log artifact identity contradiction")
    complete = False
    if receipt is not None:
        _science(receipt, "formal receipt", main=main)
        require(receipt.get("manifest_sha256") == manifest_capture.ref["sha256"]
                and receipt.get("repository") == native["repository"]
                and receipt.get("checked_commit") == native["checked_commit"]
                and receipt.get("workflow_run_id") == native["run_id"], "native receipt identity contradiction")
        require(receipt.get("formalization_status") == "kernel-checked"
                and receipt.get("alignment_status") == "PENDING_INDEPENDENT_REVIEW",
                "unsupported native receipt status")
        if main:
            require(type(manifest.get("package")) is str and receipt.get("package") == manifest["package"],
                    "main receipt package contradiction")
        require(type(manifest.get("dependency_revisions")) is dict
                and canonical(receipt.get("dependency_revisions")) == canonical(manifest["dependency_revisions"]),
                "native dependency receipt contradiction")
        for revision in manifest["dependency_revisions"].values():
            digest(revision, HEX40, "native dependency revision")
        lean_version = text(receipt.get("lean_version"), "native Lean version record")
        if main:
            require(MAIN_VERSION.search(lean_version) is not None,
                    "unsupported main native Lean version")
        recorded_logs = receipt.get("logs")
        require(type(recorded_logs) is dict, "missing native receipt log inventory")
        for name, expected_hash in recorded_logs.items():
            relative_path(name, "receipt log name")
            require("/" not in name, "receipt log key must be native basename")
            digest(expected_hash, HEX64, "receipt log SHA256")
            require(name in logs and logs[name].ref["sha256"] == expected_hash,
                    "original receipt log hash contradiction")
        require(set(logs) == set(recorded_logs), "unbound original native log")
        receipt_axioms = _axioms(receipt.get("axioms"), target_set, "native receipt axioms")
        dumped = None if "axioms.log" not in logs else _axiom_dump(logs["axioms.log"].raw, target_set)
        if dumped is not None:
            require(canonical(dumped) == canonical(receipt_axioms), "original axiom log contradicts receipt")
        complete = (dumped is not None and set(dumped) == target_set
                    and {"build.log", "leanchecker.log", "elaborated-types.log"} <= set(logs))
        controls = receipt.get("negative_controls")
        require(type(controls) is dict and all(type(name) is str and outcome in
                ("REJECTED_BY_LEAN", "REJECTED_BY_AXIOM_GATE") for name, outcome in controls.items()),
                "invalid native negative-control outcomes")
        if main:
            registered = manifest.get("negative_controls")
            require(type(registered) is dict, "missing main negative-control declarations")
            phases = dict.fromkeys(registered, "REJECTED_BY_LEAN")
            # The native producer installs these axiom controls after mutations.
            phases.update(dict.fromkeys(("sorry", "custom_imported", "native"), "REJECTED_BY_AXIOM_GATE"))
            require(all(name not in controls or controls[name] == phase for name, phase in phases.items()),
                    "main native negative-control rejection mechanism contradiction")
            complete = complete and set(phases) <= set(controls)
    alignment_capture = None if value["alignment"] is None else captures.git(value["alignment"], "formal alignment")
    alignment = None if alignment_capture is None else alignment_capture.document("formal alignment")
    alignment_reasons = _alignment_reasons(alignment, manifest_capture, scope, targets,
                                           main=main, source_author=manifest.get("author"))
    return {
        "id": identity, "main": main, "native": native, "context_current": _same_context(native, expected),
        "manifest_capture": manifest_capture, "scope": scope, "manifest": manifest,
        "targets": targets, "target_rows": target_rows, "source_rows": source_rows,
        "receipt": receipt, "complete": complete, "mappings": mappings,
        "alignment": alignment, "alignment_capture": alignment_capture, "alignment_reasons": alignment_reasons,
        "summary": {"id": identity, "format": value["format"], "manifest_targets": targets,
                    "node_targets": value["node_targets"], "retained_manifest": manifest,
                    "retained_receipt": receipt, "retained_alignment": alignment},
    }


RETROFIT_BASELINE = {"Math-": "08f86862f859ac4804b4fd6c6477a7ed0421e37f",
                     "main": "cddbb7f6cf3f57f3b495277f7da148287e027b19"}
RETROFIT_AXES = frozenset({"source_review", "provider_distinct_review", "blind_reconstruction",
                          "adversarial_attack", "formal_evidence", "numerical_reproduction",
                          "novelty", "human_reading", "custody", "observable_statement"})
RETROFIT_REPOSITORIES = frozenset({"Math-", "main", "Universal-Law-Workspace", "d6g8k5htny-coder",
                                  "google-drive", "governance-", "meta-framework", "query-", "sandbox", "trial"})
RETROFIT_EVIDENCE_KEYS = {
    "file": {"kind", "repository", "commit", "path", "blob"},
    "github_comment": {"kind", "repository", "ref"},
    "github_review": {"kind", "repository", "ref", "commit"},
    "workflow_run": {"kind", "repository", "ref", "attempt", "job", "run_head_sha",
                     "checked_commit", "purpose", "conclusion", "expected_conclusion"},
}
RETROFIT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
RETROFIT_DECIMAL = re.compile(r"[1-9][0-9]{0,19}")


def _retrofit_person(value: Any) -> None:
    keys(value, {"provider", "model_or_agent", "session"}, "retrofit performer")
    for field in value:
        text(value[field], "retrofit performer " + field)


def _retrofit_evidence(item: Any) -> None:
    require(type(item) is dict and type(item.get("kind")) is str
            and item["kind"] in RETROFIT_EVIDENCE_KEYS, "unknown retrofit evidence kind")
    kind = item["kind"]
    keys(item, RETROFIT_EVIDENCE_KEYS[kind], "retrofit evidence")
    require(type(item["repository"]) is str and item["repository"] in RETROFIT_REPOSITORIES,
            "unknown retrofit evidence repository")
    if kind == "file":
        relative_path(item["path"], "retrofit evidence file")
        digest(item["commit"], HEX40, "retrofit file commit")
        digest(item["blob"], HEX40, "retrofit file blob")
    else:
        digest(item["ref"], RETROFIT_DECIMAL, "retrofit native id")
        if kind == "github_review":
            digest(item["commit"], HEX40, "retrofit native review commit")
        if kind == "workflow_run":
            require(type(item["attempt"]) is int and item["attempt"] > 0,
                    "retrofit run attempt must be positive integer")
            if item["job"] is not None:
                digest(item["job"], RETROFIT_DECIMAL, "retrofit native job")
            digest(item["run_head_sha"], HEX40, "retrofit run head")
            if item["checked_commit"] is not None:
                digest(item["checked_commit"], HEX40, "retrofit checked commit")
            require(item["purpose"] in ("check", "negative_control"), "unknown retrofit purpose")
            require(item["conclusion"] in ("success", "failure", "cancelled", "skipped", "timed_out",
                                            "neutral", "action_required", "stale", "startup_failure"),
                    "unknown retrofit native conclusion")
            require(item["expected_conclusion"] in ("success", "failure"), "unknown retrofit expected conclusion")


def _retrofit(value: Any, captures: Captures, nodes: set[str]) -> dict[str, Any]:
    keys(value, {"id", "record", "node_records"}, "retrofit wrapper")
    identity = text(value["id"], "retrofit id")
    capture = captures.git(value["record"], "retrofit original record")
    document = capture.document("retrofit original record")
    keys(document, {"schema", "shard", "baseline", "author", "scientific_effect", "status_authority", "records"},
         "retrofit native document")
    require(document["schema"] == "retrofit-records/v0.1" and document["scientific_effect"] == "NONE"
            and document["status_authority"] is False, "invalid retrofit schema or authority")
    digest(document["shard"], re.compile(r"[A-Z][A-Za-z0-9_]{0,39}"), "retrofit shard")
    require(canonical(document["baseline"]) == canonical(RETROFIT_BASELINE), "retrofit baseline contradiction")
    _retrofit_person(document["author"])
    require(type(document["records"]) is list and bool(document["records"]), "empty retrofit native records")
    records = {}
    for record in document["records"]:
        keys(record, {"id", "subject", "delta", "axis", "state", "evidence", "performer", "exposure",
                      "independence_credit", "alias_of", "notes"}, "retrofit native record")
        record_id = digest(record["id"], RETROFIT_ID, "retrofit record id")
        require(record_id not in records, "duplicate retrofit record id")
        subject = keys(record["subject"], {"repository", "commit", "path", "blob"}, "retrofit subject")
        require(type(subject["repository"]) is str and subject["repository"] in RETROFIT_REPOSITORIES,
                "unknown retrofit subject repository")
        digest(subject["commit"], HEX40, "retrofit subject commit")
        digest(subject["blob"], HEX40, "retrofit subject blob")
        relative_path(subject["path"], "retrofit subject path")
        require(type(record["delta"]) is bool, "retrofit delta must be exact boolean")
        require(record["delta"] or RETROFIT_BASELINE.get(subject["repository"]) == subject["commit"],
                "nonbaseline retrofit subject requires delta")
        require(type(record["axis"]) is str and record["axis"] in RETROFIT_AXES, "unknown retrofit axis")
        require(type(record["state"]) is str and record["state"] in STATES, "unknown retrofit availability")
        require(type(record["evidence"]) is list, "retrofit evidence must be array")
        require(record["state"] != "recorded" or bool(record["evidence"]), "recorded retrofit lacks evidence")
        require(record["state"] not in ("not_recorded", "not_applicable") or not record["evidence"],
                "absent retrofit availability cannot carry evidence")
        for item in record["evidence"]:
            _retrofit_evidence(item)
        _retrofit_person(record["performer"])
        text(record["exposure"], "retrofit exposure")
        require(type(record["independence_credit"]) is int and record["independence_credit"] == 0,
                "retrofit independence credit must be integer zero")
        require(record["alias_of"] is None or type(record["alias_of"]) is str, "invalid retrofit alias id")
        require(type(record["notes"]) is str, "retrofit notes must be text")
        records[record_id] = record
    for record in records.values():
        if record["alias_of"] is not None:
            original = records.get(record["alias_of"])
            require(original is not None and original is not record and original["alias_of"] is None,
                    "dangling or chained retrofit alias")
            require(original["axis"] == record["axis"] and original["state"] == record["state"]
                    and sorted(canonical(item) for item in original["evidence"])
                    == sorted(canonical(item) for item in record["evidence"]), "retrofit alias differs from original")
    require(type(value["node_records"]) is list, "retrofit node_records must be array")
    bindings = []
    seen = set()
    for binding in value["node_records"]:
        keys(binding, {"node", "record_id"}, "retrofit node record")
        require(type(binding["node"]) is str and binding["node"] in nodes
                and type(binding["record_id"]) is str and binding["record_id"] in records,
                "unknown retrofit node or record")
        pair = (binding["node"], binding["record_id"])
        require(pair not in seen, "duplicate retrofit node record binding")
        seen.add(pair)
        bindings.append((binding["node"], records[binding["record_id"]]))
    return {"id": identity, "capture": capture, "document": document, "bindings": bindings}


def _fingerprint(binding: dict[str, Any] | None) -> Any:
    if binding is None:
        return None
    source = binding["source"]
    return {"source": None if source is None else {key: source.ref[key] for key in
             ("repository", "commit", "path", "bytes", "sha256")},
            "statement_slice": binding["statement_slice"], "proof_slice": binding["proof_slice"]}


def _regression(old: dict[str, Any], new: dict[str, Any], gate: ModuleType) -> dict[str, Any]:
    old_graph, new_graph = old["graph"], new["graph"]
    old_sources = {node: _fingerprint(old["bindings"].get(node)) for node in old_graph["nodes"]}
    new_sources = {node: _fingerprint(new["bindings"].get(node)) for node in new_graph["nodes"]}
    # Snapshot-relative source commits normally advance with the graph commit.
    # Ignore that paired revision change while preserving fixed-origin changes.
    for node in set(old_sources) & set(new_sources):
        before, after = old_sources[node], new_sources[node]
        if before is None or after is None or before["source"] is None or after["source"] is None:
            continue
        if all(fingerprint["source"]["repository"] == snapshot["graph_capture"].ref["repository"]
               and fingerprint["source"]["commit"] == snapshot["graph_capture"].ref["commit"]
               for fingerprint, snapshot in ((before, old), (after, new))):
            before["source"]["commit"] = after["source"]["commit"] = None
    impact = gate.reverse_impact_between(old_graph, new_graph, old_sources=old_sources, new_sources=new_sources)
    reasons: dict[str, set[str]] = {node: set() for node in impact["changed_nodes"]}
    statement_changed = []
    proof_changed = []
    unscoped = []
    for node in impact["changed_nodes"]:
        if node not in old_graph["nodes"] or node not in new_graph["nodes"]:
            reasons[node].add("node_record_changed")
        elif canonical(old_graph["nodes"][node]) != canonical(new_graph["nodes"][node]):
            reasons[node].add("node_record_changed")
        old_edges = sorted(canonical(edge) for edge in old_graph["edges"] if edge["from"] == node)
        new_edges = sorted(canonical(edge) for edge in new_graph["edges"] if edge["from"] == node)
        if old_edges != new_edges:
            reasons[node].add("dependency_edges_changed")
        if impact["context_changed"]:
            reasons[node].add("graph_context_changed")
        if node in old_sources and node in new_sources and canonical(old_sources[node]) != canonical(new_sources[node]):
            before, after = old["bindings"].get(node), new["bindings"].get(node)
            scoped = False
            if before is not None and after is not None and before["source"] is not None and after["source"] is not None:
                for field, code, changed in (("statement_slice", "statement_changed", statement_changed),
                                             ("proof_slice", "proof_body_changed", proof_changed)):
                    first, last = before[field], after[field]
                    if first is not None and last is not None and first["sha256"] != last["sha256"]:
                        changed.append(node)
                        reasons[node].add(code)
                        scoped = True
            if not scoped:
                unscoped.append(node)
                reasons[node].add("source_changed_unscoped")
    proposals = []
    for node in impact["impacted"]:
        codes = reasons.get(node, {"dependency_changed"})
        proposals.append({"node": node, "reasons": sorted(codes or {"dependency_changed"})})
    return {"changed_nodes": impact["changed_nodes"], "impacted_nodes": impact["impacted"],
            "statement_changed": sorted(statement_changed), "proof_changed": sorted(proof_changed),
            "unscoped_source_changed": sorted(unscoped), "traversal_edges": impact["traversal_edges"],
            "revalidation_required": proposals, "affected_alignment_records": []}


def _axis(state: str = "unknown", applicability: str = "unknown", *, reasons: list[str] | None = None,
          evidence: list[str] | None = None) -> dict[str, Any]:
    return {"record_state": state, "applicability": applicability, "reasons": sorted(set(reasons or [])),
            "evidence_ids": sorted(set(evidence or [])), "custody": "unknown"}


def _merge_axis(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return _axis()
    states = {record["record_state"] for record in records}
    state = "recorded" if "recorded" in states else next(iter(states)) if len(states) == 1 else "unknown"
    applications = {record["applicability"] for record in records}
    applicability = "stale" if "stale" in applications else next(iter(applications)) if len(applications) == 1 else "unknown"
    return _axis(state, applicability,
                 reasons=[code for record in records for code in record["reasons"]],
                 evidence=[identity for record in records for identity in record["evidence_ids"]])


def _main_join(formal: dict[str, Any], node: str, target: str, bindings: dict[str, Any]) -> tuple[bool, list[str]]:
    binding = bindings.get(node)
    if not formal["main"] or binding is None or binding["source"] is None:
        return False, ["missing_target_source_binding"]
    row = formal["target_rows"][target]
    origin = formal["source_rows"][row["source"]]
    capture = binding["source"]
    if any(capture.ref[key] != origin[key] for key in ("repository", "commit", "path", "bytes", "sha256")):
        return False, ["missing_target_source_binding"]
    statement = binding["statement_slice"]
    if statement is None:
        return False, ["missing_statement_slice"]
    portion = capture.raw[statement["start_byte"]:statement["end_byte"]]
    if row["informal_anchor"].encode("utf-8") not in portion:
        return False, ["missing_statement_slice"]
    return True, ["declared_byte_slice_only"]


def _kernel(formal: dict[str, Any], joined: bool, join_reasons: list[str]) -> dict[str, Any]:
    reasons = list(join_reasons)
    native = formal["native"]
    if not formal["context_current"]:
        reasons.append("native_context_changed")
    if native["purpose"] != "check":
        reasons.append("native_run_negative_control")
    if native["conclusion"] != "success":
        reasons.append("native_run_not_success")
    if not formal["complete"]:
        reasons.append("target_inventory_incomplete")
    current = (joined and formal["context_current"] and formal["complete"]
               and native["purpose"] == "check" and native["conclusion"] == "success")
    applicability = "current" if current else "stale" if not formal["context_current"] else "unknown"
    return _axis("recorded" if formal["receipt"] is not None else "unknown", applicability,
                 reasons=reasons, evidence=[formal["id"]])


def _retrofit_projection(record: dict[str, Any], binding: dict[str, Any] | None) -> dict[str, Any]:
    reasons = ["retrofit_availability_only", "native_custody_unknown"]
    source = None if binding is None else binding["source"]
    subject = record["subject"]
    # This is the native contract's explicit repository alias, not a fuzzy join.
    qualified = "d6g8k5htny-coder/" + subject["repository"]
    matching = (source is not None and source.ref["repository"] == qualified
                and source.ref["commit"] == subject["commit"] and source.ref["path"] == subject["path"]
                and source.ref["git_blob"] == subject["blob"])
    if not matching:
        reasons.append("source_binding_changed")
    for evidence in record["evidence"]:
        if evidence["kind"] == "workflow_run":
            if evidence["purpose"] != "check":
                reasons.append("native_run_negative_control")
            if evidence["conclusion"] != "success":
                reasons.append("native_run_not_success")
    applicability = "not_applicable" if record["state"] == "not_applicable" else "unknown" if matching else "stale"
    return _axis(record["state"], applicability, reasons=reasons, evidence=[record["id"]])


def _dimensions(new: dict[str, Any], formals: list[dict[str, Any]], retrofits: list[dict[str, Any]],
                regression: dict[str, Any]) -> dict[str, Any]:
    regression_reasons = {proposal["node"]: proposal["reasons"]
                          for proposal in regression["revalidation_required"]}
    alignment_holds: dict[str, set[str]] = {}
    for formal in formals:
        capture = formal["alignment_capture"]
        if capture is None:
            continue
        held_reasons = {code for mapping in formal["mappings"]
                        for code in regression_reasons.get(mapping["node"], [])}
        if held_reasons:
            alignment_holds.setdefault(canonical(capture.ref), set()).update(held_reasons)
    projected = {node: {axis: [] for axis in AXES} for node in new["graph"]["nodes"]}
    for node, original in new["graph"]["nodes"].items():
        binding = new["bindings"].get(node)
        source = None if binding is None else binding["source"]
        if source is not None:
            ref = source.ref
            source_id = ref["repository"] + "@" + ref["commit"] + ":" + ref["path"] + "#" + ref["git_blob"]
            reasons = ["declared_byte_slice_only"]
            if binding["statement_slice"] is None:
                reasons.append("missing_statement_slice")
            projected[node]["source"].append(_axis("recorded", "current", reasons=reasons, evidence=[source_id]))
        else:
            projected[node]["source"].append(_axis(reasons=["source_not_captured"]))
        review_fields = {key: value for key, value in original.items() if key.startswith("review_")}
        if review_fields:
            review_reasons = ["graph_review_record_only"]
            application = "unknown"
            if source is not None and node in regression["statement_changed"] + regression["proof_changed"]:
                review_reasons.extend(code for code, changed in
                                      (("statement_changed", regression["statement_changed"]),
                                       ("proof_body_changed", regression["proof_changed"])) if node in changed)
                application = "stale"
            # No independent native review ID is supplied by graph review text.
            projected[node]["review"].append(_axis("recorded", application, reasons=review_reasons))
    for formal in formals:
        capture = formal["alignment_capture"]
        held_reasons = set() if capture is None else alignment_holds.get(canonical(capture.ref), set())
        for mapping in formal["mappings"]:
            node, target = mapping["node"], mapping["target"]
            joined, join_reasons = _main_join(formal, node, target, new["bindings"])
            projected[node]["kernel"].append(_kernel(formal, joined, join_reasons))
            reasons = list(formal["alignment_reasons"])
            if held_reasons:
                # The hold covers every explicit mapping of this original whole record.
                reasons.extend(sorted(held_reasons))
                reasons.append("alignment_record_revalidation_required")
            if not formal["context_current"]:
                reasons.append("native_context_changed")
            unresolved_join = formal["main"] and not joined
            if unresolved_join:
                reasons.extend(join_reasons)
            if formal["alignment"] is None:
                alignment = _axis(reasons=reasons, evidence=[formal["id"]])
            else:
                contract_stale = (bool(formal["alignment_reasons"]) or not formal["context_current"]
                                  or bool(held_reasons))
                application = "stale" if contract_stale else "unknown" if unresolved_join else "current"
                alignment = _axis("recorded", application,
                                  reasons=reasons, evidence=[formal["id"]])
            projected[node]["alignment"].append(alignment)
    retrofit_mapping = {"source_review": "review", "provider_distinct_review": "review",
                       "formal_evidence": "kernel", "numerical_reproduction": "computation"}
    for retrofit in retrofits:
        for node, record in retrofit["bindings"]:
            axis = retrofit_mapping.get(record["axis"])
            if axis is not None:
                projected[node][axis].append(_retrofit_projection(record, new["bindings"].get(node)))
    return {node: {axis: _merge_axis(projected[node][axis]) for axis in AXES} for node in sorted(projected)}


def adapt_packet(packet: dict, expected_context: dict[str, str]) -> dict:
    """Produce a loss-only derived report; never modify packet or source records."""
    _depth_and_types(packet)
    require(len(canonical(packet).encode("utf-8")) <= MAX_PACKET_BYTES, "packet byte limit")
    expected = validate_expected_context(expected_context)
    keys(packet, {"schema_version", "old", "new", "formal_records", "retrofit_records"}, "packet")
    require(type(packet["schema_version"]) is int and packet["schema_version"] == 1,
            "unsupported packet schema")
    captures = Captures()
    gate = _gate()
    old = _snapshot(packet["old"], captures, gate, "old")
    new = _snapshot(packet["new"], captures, gate, "new")
    nodes = set(new["graph"]["nodes"])
    require(type(packet["formal_records"]) is list and type(packet["retrofit_records"]) is list,
            "native record collections must be arrays")
    formals = [_formal(value, captures, nodes, expected) for value in packet["formal_records"]]
    retrofits = [_retrofit(value, captures, nodes) for value in packet["retrofit_records"]]
    identities = [record["id"] for record in formals + retrofits]
    require(len(identities) == len(set(identities)), "duplicate packet record id")
    regression = _regression(old, new, gate)
    affected = {}
    impacted = set(regression["impacted_nodes"])
    for formal in formals:
        capture = formal["alignment_capture"]
        if capture is None:
            continue
        mapped_impact = any(binding["node"] in impacted for binding in formal["mappings"])
        stale_contract = bool(formal["alignment_reasons"]) or not formal["context_current"]
        if mapped_impact or stale_contract:
            affected[canonical(capture.ref)] = capture.ref
    regression["affected_alignment_records"] = [affected[identity] for identity in sorted(affected)]
    return {"schema_version": 1, "scientific_effect": "NONE", "scientific_status_authority": False,
            "custody": "unknown", "original_packet": packet,
            "dimensions": _dimensions(new, formals, retrofits, regression),
            "formal_summary": [formal["summary"] for formal in formals], "regression": regression}
