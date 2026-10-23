#!/usr/bin/env python3
"""Read a pinned public node-status view. Every result is advisory, never a lease."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPOSITORY = "d6g8k5htny-coder/main"
SOURCE_PATH = "docs/site/node-status.json"
ORIGINS = frozenset({
    "https://github.com/d6g8k5htny-coder/main.git",
    "https://github.com/d6g8k5htny-coder/main",
    "git@github.com:d6g8k5htny-coder/main.git",
    "ssh://git@github.com/d6g8k5htny-coder/main.git",
})
SHA = re.compile(r"[0-9a-f]{40}\Z")
NODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z")
UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
COMMENT = re.compile(r"https://github\.com/d6g8k5htny-coder/(?:main|Math-)/(?:issues|pull)/[1-9][0-9]*#(?:issuecomment|pullrequestreview)-[1-9][0-9]*\Z")
OBJECT = re.compile(r"https://github\.com/d6g8k5htny-coder/(?:main|Math-)/(?:issues|pull)/[1-9][0-9]*(?:#(?:issuecomment|pullrequestreview)-[1-9][0-9]*)?\Z")
PINNED = re.compile(r"https://github\.com/d6g8k5htny-coder/(?:main|Math-)/(?:blob|tree)/([0-9a-f]{40})/[A-Za-z0-9_./-]+\Z")
REQUIRED = frozenset({"schema_version", "node_id", "owner", "next_action", "missing_input", "stops", "sources", "source_rev", "observed_at", "status", "scientific_effect"})
OPTIONAL = frozenset({"predecessor", "successor", "carried_forward", "refresh_error"})
MAX_BYTES = 1_048_576
MAX_RECORDS = 256
MAX_ITEMS = 256
MAX_TEXT = 16_384
OBSERVATION_WINDOW_SECONDS = 3600


class Refused(ValueError):
    """An input cannot support a usable advisory observation."""


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise Refused("invalid command arguments")


def utc_time(value):
    if not isinstance(value, str) or UTC.fullmatch(value) is None:
        raise Refused("timestamp must be a real UTC YYYY-MM-DDTHH:MM:SSZ value")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise Refused("timestamp must be a real UTC YYYY-MM-DDTHH:MM:SSZ value") from error


def text(value):
    return isinstance(value, str) and 0 < len(value) <= MAX_TEXT


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Refused("duplicate JSON key")
        result[key] = value
    return result


def invalid_constant(value):
    raise Refused("unsupported JSON numeric value")


def bounded_integer(value):
    if len(value) > 20:
        raise Refused("JSON integer exceeds read bound")
    return int(value)


def git(root, *args):
    # No credential lookup, lazy network fetch, object replacement, or ambient
    # GIT_DIR/config injection. Only fixed read-only plumbing is called below.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0")
    command = ["git", "--no-replace-objects", "-c", "protocol.allow=never",
               "-c", "core.hooksPath=" + os.devnull, "-c", "core.fsmonitor=false",
               "-c", "gc.auto=0", "-C", str(root), *args]
    result = subprocess.run(command, env=env, capture_output=True, timeout=10, check=False)
    if result.returncode:
        # Git diagnostics can contain local paths or configuration; do not copy them.
        raise Refused("required local Git object or repository metadata unavailable")
    if len(result.stdout) > MAX_BYTES:
        raise Refused("Git output exceeds read bound")
    return result.stdout


def pinned_source(root, commit, report):
    if not isinstance(commit, str) or SHA.fullmatch(commit) is None:
        raise Refused("commit must be an exact lowercase 40-character commit ID")
    if git(root, "rev-parse", "--show-toplevel").decode().strip() != str(root):
        raise Refused("run from the repository root")
    origins = git(root, "config", "--local", "--get-all", "remote.origin.url").decode().splitlines()
    if len(origins) != 1 or origins[0] not in ORIGINS:
        raise Refused("origin is not the expected public repository")
    if git(root, "rev-parse", "--verify", commit + "^{commit}").decode().strip() != commit:
        raise Refused("commit identity mismatch")
    listing = git(root, "ls-tree", "-z", "--full-tree", commit, "--", SOURCE_PATH)
    entries = listing.split(b"\0")
    if len(entries) != 2 or entries[1] != b"":
        raise Refused("fixed node-status path is missing or ambiguous")
    metadata, separator, path = entries[0].partition(b"\t")
    parts = metadata.split(b" ")
    if (separator != b"\t" or path != SOURCE_PATH.encode() or len(parts) != 3
            or parts[0] not in (b"100644", b"100755") or parts[1] != b"blob"):
        raise Refused("node-status must be a regular Git blob")
    blob = parts[2].decode("ascii")
    if SHA.fullmatch(blob) is None:
        raise Refused("invalid Git blob identity")
    size = int(git(root, "cat-file", "-s", blob).decode().strip())
    if size < 0 or size > MAX_BYTES:
        raise Refused("node-status blob exceeds read bound")
    raw = git(root, "cat-file", "blob", blob)
    if len(raw) != size or hashlib.sha1(b"blob " + str(size).encode() + b"\0" + raw).hexdigest() != blob:
        raise Refused("Git blob identity mismatch")
    report["source"] = {"repository": REPOSITORY, "commit": commit, "path": SOURCE_PATH,
                        "blob": blob, "sha256": hashlib.sha256(raw).hexdigest(),
                        "url": "https://github.com/" + REPOSITORY + "/blob/" + commit + "/" + SOURCE_PATH}
    return raw


def string_list(value):
    return isinstance(value, list) and len(value) <= MAX_ITEMS and all(text(item) for item in value)


def check_record(record, node):
    if not isinstance(record, dict) or not REQUIRED <= set(record) or set(record) - REQUIRED - OPTIONAL:
        raise Refused("record does not match the closed v0 schema")
    if type(record["schema_version"]) is not int or record["schema_version"] != 0:
        raise Refused("unknown record schema version")
    if not isinstance(record["node_id"], str) or NODE.fullmatch(record["node_id"]) is None or record["node_id"] != node:
        raise Refused("record node identity mismatch")
    if not text(record["next_action"]):
        raise Refused("invalid next_action")
    for field in ("missing_input", "stops"):
        if not string_list(record[field]):
            raise Refused("invalid " + field)
    if record["status"] not in ("fresh", "stale", "unknown") or record["scientific_effect"] != "NONE":
        raise Refused("invalid record status or scientific effect")
    utc_time(record["observed_at"])
    owner = record["owner"]
    if owner is not None:
        if (not isinstance(owner, dict) or not {"id", "evidence_url"} <= set(owner)
                or set(owner) - {"id", "evidence_url", "released_url"} or not text(owner["id"])
                or not isinstance(owner["evidence_url"], str) or COMMENT.fullmatch(owner["evidence_url"]) is None):
            raise Refused("invalid recorded owner")
        if "released_url" in owner and (not isinstance(owner["released_url"], str) or COMMENT.fullmatch(owner["released_url"]) is None):
            raise Refused("invalid recorded owner release URL")
    sources = record["sources"]
    if not isinstance(sources, list) or not 0 < len(sources) <= MAX_ITEMS:
        raise Refused("invalid record sources")
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"url", "rev"} or not text(source["url"]) or not text(source["rev"]):
            raise Refused("invalid source identity")
        pinned = PINNED.fullmatch(source["url"])
        if pinned:
            if source["rev"] != pinned.group(1):
                raise Refused("pinned source revision identity mismatch")
        elif OBJECT.fullmatch(source["url"]):
            utc_time(source["rev"])
        else:
            raise Refused("unsupported source URL")
    if record["source_rev"] != sources[0]["rev"]:
        raise Refused("source_rev must equal the primary source revision")
    for field in ("predecessor", "successor"):
        link = record.get(field)
        if link is not None and (not isinstance(link, dict) or set(link) != {"node_id", "url"}
                                 or not text(link["node_id"]) or not text(link["url"])
                                 or re.fullmatch(r"https?://\S+", link["url"]) is None):
            raise Refused("invalid " + field)
    if "refresh_error" in record and record["refresh_error"] is not None and not isinstance(record["refresh_error"], str):
        raise Refused("invalid refresh_error")
    if "carried_forward" in record:
        carried = record["carried_forward"]
        if not isinstance(carried, list) or len(carried) > MAX_ITEMS:
            raise Refused("invalid carried_forward")
        seen = set()
        for item in carried:
            if (not isinstance(item, dict) or not {"id", "text", "source_url", "status"} <= set(item)
                    or set(item) - {"id", "text", "source_url", "status", "resolution_url"}
                    or any(not text(item[field]) for field in ("id", "text", "source_url"))
                    or item["id"] in seen or item["status"] not in ("open", "resolved")):
                raise Refused("invalid carried_forward item")
            seen.add(item["id"])
            if item["status"] == "resolved" or "resolution_url" in item:
                if not isinstance(item.get("resolution_url"), str) or COMMENT.fullmatch(item["resolution_url"]) is None:
                    raise Refused("invalid carried_forward resolution")


def base_report():
    return {"report_version": 1, "disposition": "HOLD", "authority_granted": False,
            "universal_lock": False, "advisory_only": True, "native_reconciliation_required": True,
            "scientific_effect": "NONE", "node_id": None, "claimant_id": None,
            "owner": None, "status": "unknown", "recorded_status": None, "observed_at": None,
            "observed_age_seconds": None, "observation_window_seconds": OBSERVATION_WINDOW_SECONDS,
            "stops": [], "missing_input": [], "blockers": [], "sources": [], "source_rev": None,
            "source": None, "evaluated_at": None}


def observe(args, report):
    report["node_id"] = args.node
    report["claimant_id"] = args.claimant or None
    if NODE.fullmatch(args.node) is None:
        raise Refused("invalid requested node ID")
    if args.claimant and (not text(args.claimant) or any(ord(char) < 32 or ord(char) == 127 for char in args.claimant)):
        raise Refused("invalid claimant ID")
    now = utc_time(args.now) if args.now is not None else datetime.now(timezone.utc).replace(microsecond=0)
    report["evaluated_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = pinned_source(Path.cwd().resolve(), args.commit, report)
    data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                      parse_constant=invalid_constant, parse_float=invalid_constant,
                      parse_int=bounded_integer)
    if not isinstance(data, dict) or len(data) > MAX_RECORDS:
        raise Refused("node-status must be a bounded record map")
    if args.node not in data:
        raise Refused("requested node has no recorded status")
    record = data[args.node]
    # Keep available owner/stop/source fields visible even when schema validation
    # refuses the record. Their presence does not make them validated evidence.
    if isinstance(record, dict):
        for field in ("owner", "stops", "missing_input", "sources", "source_rev", "observed_at"):
            if field in record:
                report[field] = record[field]
    check_record(record, args.node)
    report["recorded_status"] = record["status"]
    report["observed_age_seconds"] = age = int((now - utc_time(record["observed_at"])).total_seconds())
    report["status"] = ("unknown" if age < 0 else "stale" if age > OBSERVATION_WINDOW_SECONDS else record["status"])
    if age < 0:
        report["blockers"].append("observation is in the future")
    elif report["status"] != "fresh":
        report["blockers"].append("record observation is stale or unknown; owner is not released")
    if record["owner"] is None:
        report["blockers"].append("no recorded owner; fresh native pickup reconciliation is required")
    if record["stops"]:
        report["blockers"].append("recorded stops remain")
    if record["missing_input"]:
        report["blockers"].append("recorded missing inputs remain")
    if record.get("refresh_error"):
        report["blockers"].append("recorded refresh error remains")
    if any(item["status"] == "open" for item in record.get("carried_forward", [])):
        report["blockers"].append("open carried-forward findings remain")
    owner = record["owner"]
    if owner is not None and args.claimant and owner["id"] != args.claimant:
        report["disposition"] = "CONFLICT"
        report["blockers"].append("claimant differs from the recorded owner")
    elif owner is not None and "released_url" in owner:
        report["disposition"] = "EXPLICIT_HANDOFF_REQUIRED"
        report["blockers"].append("release URL preserves the owner; explicit native handoff is required")
    elif not report["blockers"]:
        report["disposition"] = "OBSERVED"


def main(argv=None):
    report = base_report()
    try:
        parser = Parser(description=__doc__, allow_abbrev=False)
        subparsers = parser.add_subparsers(dest="command", required=True, parser_class=Parser)
        command = subparsers.add_parser("report", allow_abbrev=False)
        command.add_argument("--commit", required=True)
        command.add_argument("--node", required=True)
        command.add_argument("--claimant")
        command.add_argument("--now")
        observe(parser.parse_args(argv), report)
    except (OSError, ValueError, UnicodeError, RecursionError, subprocess.TimeoutExpired) as error:
        # Do not copy arbitrary parser/Git diagnostics or input bytes into logs.
        report["blockers"].append(str(error) if isinstance(error, Refused) else "input parse or local read anomaly")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if report["disposition"] == "OBSERVED" else 2


if __name__ == "__main__":
    sys.exit(main())
