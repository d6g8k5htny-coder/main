#!/usr/bin/env python3
"""Validate records/node_status against architecture/node-status.schema.json.

Status is freshness of the current view only. scientific_effect must stay NONE.
An owner is null or an object. released_url may be present on that object and
is never a reason to replace the owner with null. source_rev must equal
sources[0].rev, one value, never a comma-joined list. No network and no writes.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "schema_version", "node_id", "owner", "next_action", "missing_input", "stops",
    "sources", "source_rev", "observed_at", "status", "scientific_effect",
)
OPTIONAL = frozenset({"predecessor", "successor", "carried_forward", "refresh_error"})
FORBIDDEN = frozenset({
    "lemma_closed", "prizes_solved", "discharges_OBL_H5_JETMOD", "certified_C_H",
    "freeze", "inventable_attempt_accepted",
})
NODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
OBSERVED = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
# Canonical comment URL: evidence_url, released_url, and carried_forward
# resolution_url. The same characters are NODE_STATUS_EVIDENCE_URL_PATTERN
# in docs/site/current-model.mjs.
EVIDENCE_URL_PATTERN = r"^https://github\.com/d6g8k5htny-coder/(main|Math-)/(issues|pull)/[1-9]\d*#(issuecomment-[1-9]\d*|pullrequestreview-[1-9]\d*)$"
EVIDENCE_URL = re.compile(EVIDENCE_URL_PATTERN)
PINNED_URL = re.compile(
    r"^https://github\.com/d6g8k5htny-coder/(main|Math-)/(blob|tree)/"
    r"[0-9a-f]{40}/[A-Za-z0-9_./-]+$")
GITHUB_OBJECT_URL = re.compile(
    r"^https://github\.com/d6g8k5htny-coder/(main|Math-)/(issues|pull)/[1-9]\d*"
    r"(#(issuecomment-[1-9]\d*|pullrequestreview-[1-9]\d*))?$")
UTC_REV = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
CARRIED_SOURCE_URL = re.compile(
    r"^https://github\.com/[\w.-]+/[\w.-]+/(?:issues|pull)/\d+"
    r"(?:#(?:issuecomment|pullrequestreview)-\d+)?$")
STATUSES = frozenset({"fresh", "stale", "unknown"})
MAX_FILES = 256
MAX_BYTES = 65536


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def forbidden_keys(value, found):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN:
                found.append(key)
            forbidden_keys(item, found)
    elif isinstance(value, list):
        for item in value:
            forbidden_keys(item, found)
    return found


def load_json(path, max_bytes=MAX_BYTES):
    if path.is_symlink():
        raise ValueError("symlink refused")
    if path.stat().st_size > max_bytes:
        raise ValueError("record too large")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def _http_url(url):
    if not isinstance(url, str) or not url or any(ch.isspace() for ch in url):
        return False
    parsed = urlparse(url)
    return parsed.scheme.lower() in ("http", "https") and bool(parsed.netloc)


def _closed_schema_problems(record):
    extra = sorted(set(record) - set(REQUIRED) - OPTIONAL)
    missing = sorted(set(REQUIRED) - set(record))
    problems = []
    if extra:
        problems.append("unexpected fields: " + ", ".join(extra))
    if missing:
        problems.append("missing fields: " + ", ".join(missing))
    return problems


def _owner_object(schema):
    for item in schema.get("properties", {}).get("owner", {}).get("oneOf", []):
        if item.get("type") == "object":
            return item
    return None


def check_schema_file(root):
    path = root / "architecture" / "node-status.schema.json"
    problems = []
    try:
        schema = load_json(path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        return [f"schema: {error}"]
    if schema.get("required") != list(REQUIRED):
        problems.append("schema: required fields differ from interface v0")
    if schema.get("additionalProperties") is not False:
        problems.append("schema: additional properties must be refused")
    props = schema.get("properties", {})
    for name in ("predecessor", "successor", "carried_forward", "refresh_error"):
        if name not in props:
            problems.append(f"schema: optional field {name} missing")
        if name in schema.get("required", []):
            problems.append(f"schema: {name} must stay optional")
    status = props.get("status", {}).get("enum")
    if status != ["fresh", "stale", "unknown"]:
        problems.append("schema: status must be fresh, stale, or unknown")
    effect = props.get("scientific_effect", {}).get("const")
    if effect != "NONE":
        problems.append("schema: scientific_effect must be NONE")
    owner = props.get("owner", {})
    if not any(item.get("type") == "null" for item in owner.get("oneOf", [])):
        problems.append("schema: owner must allow null")
    owner_object = _owner_object(schema)
    if owner_object is None:
        problems.append("schema: owner with released_url must stay an object")
    else:
        owner_props = owner_object.get("properties", {})
        if owner_props.get("evidence_url", {}).get("pattern") != EVIDENCE_URL_PATTERN:
            problems.append("schema: evidence_url pattern drift")
        if owner_props.get("released_url", {}).get("pattern") != EVIDENCE_URL_PATTERN:
            problems.append("schema: released_url pattern drift")
        if "released_url" in owner_object.get("required", []):
            problems.append("schema: released_url must stay optional")
    resolution = (
        props.get("carried_forward", {})
        .get("items", {})
        .get("properties", {})
        .get("resolution_url", {})
    )
    if resolution.get("pattern") != EVIDENCE_URL_PATTERN:
        problems.append("schema: resolution_url pattern drift")
    return problems


def graph_ids(root):
    path = root / "docs" / "site" / "current.json"
    data = load_json(path, max_bytes=2_000_000)
    nodes = data["graph"]["data"]["nodes"]
    if not isinstance(nodes, dict):
        raise ValueError("current graph nodes are not an object")
    return set(nodes)


def check_owner(record, problems):
    """Read owner. A released_url keeps the owner object; it is not cleared."""
    owner = record["owner"]
    if owner is None:
        return
    if not isinstance(owner, dict):
        problems.append("owner must be null or an object with evidence_url")
        return
    extra = sorted(set(owner) - {"id", "evidence_url", "released_url"})
    if extra:
        problems.append("owner allows only id, evidence_url, and released_url")
    evidence = owner.get("evidence_url")
    if "evidence_url" not in owner or not isinstance(evidence, str) or evidence == "":
        problems.append("owner requires evidence_url")
        return
    if not isinstance(owner.get("id"), str) or owner.get("id") == "":
        problems.append("owner id must be a non-empty string")
    if not EVIDENCE_URL.fullmatch(evidence):
        problems.append("owner evidence_url does not match the comment pattern")
    if "released_url" in owner:
        released = owner.get("released_url")
        if not isinstance(released, str) or not EVIDENCE_URL.fullmatch(released):
            problems.append("owner released_url does not match the comment pattern")


def check_link(value, field, problems):
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {"node_id", "url"}:
        problems.append(f"{field} must be null or an object with node_id and url")
        return
    node_id = value.get("node_id")
    if not isinstance(node_id, str) or node_id == "" or any(ch.isspace() for ch in node_id):
        problems.append(f"{field}.node_id must be a non-empty string")
    if not _http_url(value.get("url")):
        problems.append(f"{field}.url must be an http(s) URL")


def check_carried(value, problems):
    if not isinstance(value, list):
        problems.append("carried_forward must be a list")
        return
    seen = set()
    allowed = {"id", "text", "source_url", "status", "resolution_url"}
    required = {"id", "text", "source_url", "status"}
    for index, item in enumerate(value):
        label = f"carried_forward[{index}]"
        if not isinstance(item, dict) or not required <= set(item) or not set(item) <= allowed:
            problems.append(f"{label} must have id, text, source_url, and status")
            continue
        item_id = item["id"]
        if not isinstance(item_id, str) or item_id == "":
            problems.append(f"{label}.id must be a non-empty string")
        elif item_id in seen:
            problems.append(f"{label} duplicates id {item_id}")
        else:
            seen.add(item_id)
        if not isinstance(item["text"], str) or item["text"] == "":
            problems.append(f"{label}.text must be a non-empty string")
        source = item["source_url"]
        if not isinstance(source, str) or not CARRIED_SOURCE_URL.fullmatch(source):
            problems.append(f"{label}.source_url must be a GitHub issue or comment URL")
        if item["status"] not in ("open", "resolved"):
            problems.append(f"{label}.status must be open or resolved")
        elif item["status"] == "resolved" and not EVIDENCE_URL.fullmatch(str(item.get("resolution_url", ""))):
            problems.append(f"{label} resolution_url does not match the comment pattern")


def check_source(source, index, problems):
    label = f"sources[{index}]"
    if not isinstance(source, dict) or set(source) != {"url", "rev"}:
        problems.append(f"{label} must have url and rev")
        return
    url, rev = source.get("url"), source.get("rev")
    if isinstance(url, str) and PINNED_URL.fullmatch(url):
        if not isinstance(rev, str) or not SHA.fullmatch(rev):
            problems.append(f"{label}.rev must be a 40-character commit")
        elif rev not in url:
            problems.append(f"{label}.rev must appear in its url")
        return
    if isinstance(url, str) and GITHUB_OBJECT_URL.fullmatch(url):
        if not isinstance(rev, str) or not UTC_REV.fullmatch(rev):
            problems.append(f"{label}.rev must be a UTC timestamp")
            return
        try:
            datetime.strptime(rev, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            problems.append(f"{label}.rev must be a UTC timestamp")
        return
    problems.append(f"{label}.url must be a pinned blob or tree, or a GitHub issue, pull, or comment")


def _source_rev_problems(record, sources):
    source_rev = record["source_rev"]
    primary = sources[0].get("rev") if isinstance(sources[0], dict) else None
    if not isinstance(source_rev, str) or source_rev == "" or "," in source_rev:
        return ["source_rev must be a single rev"]
    if source_rev != primary:
        return ["source_rev must equal sources[0].rev"]
    return []


def check_record(record, filename, known_ids):
    problems = []
    if not isinstance(record, dict):
        return ["record must be an object"]
    flags = forbidden_keys(record, [])
    if flags:
        problems.append("forbidden mathematical status flag: " + ", ".join(sorted(set(flags))))
    problems.extend(_closed_schema_problems(record))
    if problems:
        return problems
    if type(record["schema_version"]) is not int or record["schema_version"] != 0:
        problems.append("schema_version must be 0")
    node_id = record["node_id"]
    if not isinstance(node_id, str) or not NODE_ID.fullmatch(node_id):
        problems.append("invalid node_id")
    elif filename != node_id + ".json":
        problems.append("filename must match node_id")
    elif node_id not in known_ids:
        problems.append("node_id is not in the current graph")
    check_owner(record, problems)
    if not isinstance(record["next_action"], str) or not record["next_action"]:
        problems.append("next_action must be a non-empty string")
    for field in ("missing_input", "stops"):
        value = record[field]
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            problems.append(f"{field} must be an array of non-empty strings")
    sources = record["sources"]
    if not isinstance(sources, list) or not sources:
        problems.append("sources must be a non-empty array")
    else:
        for index, source in enumerate(sources):
            check_source(source, index, problems)
        problems.extend(_source_rev_problems(record, sources))
    for field in ("predecessor", "successor"):
        if field in record:
            check_link(record[field], field, problems)
    if "carried_forward" in record:
        check_carried(record["carried_forward"], problems)
    if "refresh_error" in record and record["refresh_error"] is not None and not isinstance(record["refresh_error"], str):
        problems.append("refresh_error must be a string or null")
    observed = record["observed_at"]
    if not isinstance(observed, str) or not OBSERVED.fullmatch(observed):
        problems.append("observed_at must be UTC Z")
    else:
        try:
            datetime.strptime(observed, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            problems.append("observed_at is not a real timestamp")
    if record["status"] not in STATUSES:
        problems.append("status must be fresh, stale, or unknown")
    if record["scientific_effect"] != "NONE":
        problems.append("scientific_effect must be NONE")
    return problems


def validate_directory(root, directory):
    root = Path(root).resolve()
    directory = Path(directory)
    if not directory.is_absolute():
        directory = (Path.cwd() / directory).resolve()
    problems = list(check_schema_file(root))
    if not directory.is_dir() or directory.is_symlink():
        return problems + ["records directory missing"]
    try:
        known = graph_ids(root)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, KeyError) as error:
        return problems + [f"current graph: {error}"]
    files = sorted(directory.iterdir(), key=lambda item: item.name)
    selected = [item for item in files if item.name != ".gitkeep"]
    if not selected:
        problems.append("no node-status records")
    if len(selected) > MAX_FILES:
        problems.append("too many node-status records")
    seen = set()
    for path in selected:
        label = path.name
        if not path.is_file() or path.is_symlink() or path.suffix != ".json" or PurePosixPath(path.name).name != path.name:
            problems.append(f"{label}: expected a JSON record file")
            continue
        try:
            record = load_json(path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            problems.append(f"{label}: {error}")
            continue
        node_id = record.get("node_id") if isinstance(record, dict) else None
        if isinstance(node_id, str):
            if node_id in seen:
                problems.append(f"{label}: duplicate node_id")
            seen.add(node_id)
        for problem in check_record(record, path.name, known):
            problems.append(f"{label}: {problem}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    problems = validate_directory(args.root, args.directory)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print(f"node-status records valid: {args.directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
