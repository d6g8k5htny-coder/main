#!/usr/bin/env python3
"""Validate records/node_status against architecture/node-status.schema.json.

Status is freshness of the current view only. scientific_effect must stay NONE.
An owner is refused unless evidence_url is present. No network and no writes.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "schema_version", "node_id", "owner", "next_action", "missing_input", "stops",
    "sources", "source_rev", "observed_at", "status", "scientific_effect",
)
FORBIDDEN = frozenset({
    "lemma_closed", "prizes_solved", "discharges_OBL_H5_JETMOD", "certified_C_H",
    "freeze", "inventable_attempt_accepted",
})
NODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
OBSERVED = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
COMMENT = re.compile(
    r"https://github\.com/d6g8k5htny-coder/(?:main|Math-)/(?:issues|pull)/"
    r"[1-9][0-9]*#issuecomment-[1-9][0-9]*\Z")
SOURCE_URL = re.compile(
    r"https://github\.com/d6g8k5htny-coder/(?:main|Math-)/(?:blob|tree)/"
    r"[0-9a-f]{40}/[A-Za-z0-9_./-]+\Z")
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
    status = schema.get("properties", {}).get("status", {}).get("enum")
    if status != ["fresh", "stale", "unknown"]:
        problems.append("schema: status must be fresh, stale, or unknown")
    effect = schema.get("properties", {}).get("scientific_effect", {}).get("const")
    if effect != "NONE":
        problems.append("schema: scientific_effect must be NONE")
    owner = schema.get("properties", {}).get("owner", {})
    if not any(item.get("type") == "null" for item in owner.get("oneOf", [])):
        problems.append("schema: owner must allow null")
    return problems


def graph_ids(root):
    path = root / "docs" / "site" / "current.json"
    data = load_json(path, max_bytes=2_000_000)
    nodes = data["graph"]["data"]["nodes"]
    if not isinstance(nodes, dict):
        raise ValueError("current graph nodes are not an object")
    return set(nodes)


def check_owner(record, problems):
    owner = record["owner"]
    if owner is None:
        return
    if not isinstance(owner, dict):
        problems.append("owner requires evidence_url")
        return
    if "evidence_url" not in owner or not isinstance(owner.get("evidence_url"), str) or not owner.get("evidence_url"):
        problems.append("owner requires evidence_url")
        return
    if set(owner) != {"id", "evidence_url"}:
        problems.append("owner allows only id and evidence_url")
    if not isinstance(owner.get("id"), str) or not owner["id"]:
        problems.append("owner id must be a non-empty string")
    if not COMMENT.fullmatch(owner["evidence_url"]):
        problems.append("owner evidence_url must be a GitHub comment URL")


def check_record(record, filename, known_ids):
    problems = []
    if not isinstance(record, dict):
        return ["record must be an object"]
    flags = forbidden_keys(record, [])
    if flags:
        problems.append("forbidden mathematical status flag: " + ", ".join(sorted(set(flags))))
    if set(record) != set(REQUIRED):
        extra = sorted(set(record) - set(REQUIRED))
        missing = sorted(set(REQUIRED) - set(record))
        if extra:
            problems.append("unexpected fields: " + ", ".join(extra))
        if missing:
            problems.append("missing fields: " + ", ".join(missing))
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
            if not isinstance(source, dict) or set(source) != {"url", "rev"}:
                problems.append(f"sources[{index}] must have url and rev")
                continue
            if not isinstance(source["url"], str) or not SOURCE_URL.fullmatch(source["url"]):
                problems.append(f"sources[{index}].url must be a pinned GitHub blob or tree URL")
            if not isinstance(source["rev"], str) or not SHA.fullmatch(source["rev"]):
                problems.append(f"sources[{index}].rev must be a 40-character commit")
            elif source["rev"] not in source["url"]:
                problems.append(f"sources[{index}].rev must appear in its url")
    if not isinstance(record["source_rev"], str) or not SHA.fullmatch(record["source_rev"]):
        problems.append("source_rev must be a 40-character commit")
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
    problems = [f"schema: {item}" if not item.startswith("schema:") else item for item in check_schema_file(root)]
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
