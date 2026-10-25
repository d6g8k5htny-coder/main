"""Research custody and linkage, independently of mathematical acceptance."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from model import digest

SHA = re.compile(r"^[0-9a-f]{64}$")
RELATIONS = {"context_only", "partial_support", "candidate_support"}
CLASSIFICATIONS = {"recovered_source", "new_derivation", "review", "receipt",
                   "catalog", "unreviewed"}
MAX_BYTES = 16777216


def revision_id(capture):
    return digest({k: v for k, v in capture.items() if k != "revision_id"})


def validate_capture(capture, data):
    if not isinstance(capture, dict):
        raise ValueError("research capture must be an object")
    required = {"revision_id", "source_key", "title", "sha256", "bytes", "artifact_path",
                "classification", "scope", "limitations", "links", "provenance"}
    if set(capture) != required:
        raise ValueError("research capture has missing or unsupported fields")
    for field in ("source_key", "title", "scope"):
        if not isinstance(capture[field], str) or not 1 <= len(capture[field]) <= 4000:
            raise ValueError("invalid research " + field)
    if not SHA.fullmatch(str(capture["sha256"])):
        raise ValueError("invalid research SHA-256")
    if type(capture["bytes"]) is not int or not 0 <= capture["bytes"] <= MAX_BYTES:
        raise ValueError("research artifact exceeds size bound")
    if capture["artifact_path"] != "research/objects/" + capture["sha256"]:
        raise ValueError("research artifact must use its confined content address")
    if capture["classification"] not in CLASSIFICATIONS:
        raise ValueError("invalid research classification")
    limits = capture["limitations"]
    if not isinstance(limits, list) or not limits or not all(isinstance(x, str) and x for x in limits):
        raise ValueError("research limitations must be explicit")
    if not isinstance(capture["provenance"], dict):
        raise ValueError("research provenance must be an object")
    ids = {r["id"] for r in data["records"]}
    seen = set()
    if not isinstance(capture["links"], list):
        raise ValueError("research links must be a list")
    for link in capture["links"]:
        if not isinstance(link, dict) or set(link) != {"record_id", "relation", "reason"}:
            raise ValueError("invalid research link descriptor")
        if link["record_id"] not in ids or link["record_id"] in seen:
            raise ValueError("unknown or duplicate research target")
        if link["relation"] not in RELATIONS or not isinstance(link["reason"], str) or not link["reason"]:
            raise ValueError("research link needs a scoped relation and reason")
        seen.add(link["record_id"])
    if capture["revision_id"] != revision_id(capture):
        raise ValueError("research metadata fingerprint mismatch")
    return capture


def verify_capture(workdir, capture):
    root = Path(workdir).resolve()
    path = root / capture["artifact_path"]
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("research artifact escapes runtime")
    if not path.is_file() or path.stat().st_size != capture["bytes"]:
        raise ValueError("missing or mismatched research artifact")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != capture["sha256"]:
        raise ValueError("research artifact hash mismatch")
    return raw


def validate_attachment(value, data, record_id):
    expected = {"kind", "record_id", "capture", "relation", "applicability", "formal_status"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("invalid research attachment fields")
    if (value["kind"] != "research_evidence" or value["record_id"] != record_id
            or value["applicability"] != "unvalidated" or value["formal_status"] != "not_certified"):
        raise ValueError("research registration cannot assert applicability or certification")
    capture = validate_capture(value["capture"], data)
    if not any(link["record_id"] == record_id and link["relation"] == value["relation"]
               for link in capture["links"]):
        raise ValueError("research attachment is outside its declared target or scope")
    return value


def relevant_captures(data, attachments, record_id):
    """Include only research evidence on this obligation's prerequisite ancestry."""
    obs = {o["conclusion"]: o for o in data["obligations"]}
    ancestry = set()
    def visit(rid):
        if rid in ancestry:
            return
        ancestry.add(rid)
        for parent in obs[rid]["premises"]:
            visit(parent)
    visit(record_id)
    captures = {}
    for rid in sorted(ancestry):
        for value in attachments.get(rid, []):
            if value.get("kind") == "research_evidence":
                capture = value["capture"]
                captures[capture["revision_id"]] = capture
    return [captures[k] for k in sorted(captures)]


def attach_research(attachments, record_id, value):
    bucket = attachments.setdefault(record_id, [])
    source_key = value["capture"]["source_key"]
    bucket[:] = [old for old in bucket if not (old.get("kind") == "research_evidence"
                  and old["capture"]["source_key"] == source_key)]
    bucket.append(value)
