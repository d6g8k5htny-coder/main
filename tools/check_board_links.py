#!/usr/bin/env python3
"""Read-only check of board and session handoff links (LANES-20261022 Lane 4).

Input: a directory of JSON records shaped like node-status interface v0
(records/node_status/<node_id>.json) with three optional handoff fields:

  predecessor:     {"node_id": str, "url": http(s) URL} | null
  successor:       {"node_id": str, "url": http(s) URL} | null
  carried_forward: [{"id": str, "text": str, "source_url": str,
                     "status": "open" | "resolved",
                     "resolution_url": str (required when resolved)}]

Owner evidence, released_url, and resolution_url use COMMENT_URL below.
carried_forward source_url uses the carried-item pattern from interface v0:
any GitHub issue or comment URL.

Checks (each failure is reported; the exit status is 1 if any):
  1. reciprocity: A.successor -> B requires B.predecessor -> A, and the reverse;
     a link to a node with no record is a failure, not a skip; a link url that
     is not http(s) is a failure;
  2. ownership: a non-null owner needs an id and an evidence_url matching
     COMMENT_URL (the PICKUP or assignment). An owner with released_url is not
     a current owner, and still needs that id and evidence_url; released_url
     itself must match COMMENT_URL. Owner keys are closed to id, evidence_url,
     and released_url. Ownership is never inferred, and moving a board does
     not move it;
  3. carry-forward: carried_forward must be a list of objects with unique
     string ids, non-empty text, and a source_url. Item keys are only id,
     text, source_url, status, and resolution_url. Every predecessor item
     that is open must
     appear, by id, in the successor's carried_forward, either still open or
     resolved with a resolution_url. Items may not be silently dropped.
  4. refresh_error, when present, is a string or null.

The tool reads files only. It makes no network call and writes nothing.
It checks link structure, not the truth of any linked comment, and has no
scientific effect.
"""

import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

COMMENT_URL = re.compile(
    r"^https://github\.com/d6g8k5htny-coder/(main|Math-)/(issues|pull)/[1-9]\d*#(issuecomment-[1-9]\d*|pullrequestreview-[1-9]\d*)$"
)
CARRIED_SOURCE_URL = re.compile(
    r"^https://github\.com/[\w.-]+/[\w.-]+/(?:issues|pull)/\d+"
    r"(?:#(?:issuecomment|pullrequestreview)-\d+)?$")


def _http_url(url):
    if not isinstance(url, str) or not url or any(ch.isspace() for ch in url):
        return False
    parsed = urlparse(url)
    return parsed.scheme.lower() in ("http", "https") and bool(parsed.netloc)


def _comment_url(url):
    return isinstance(url, str) and COMMENT_URL.fullmatch(url) is not None


def _carried_source_url(url):
    return isinstance(url, str) and CARRIED_SOURCE_URL.fullmatch(url) is not None


def load_records(directory):
    records, errors = {}, []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: unreadable JSON ({exc})")
            continue
        node = rec.get("node_id") if isinstance(rec, dict) else None
        if not isinstance(node, str) or not node:
            errors.append(f"{path.name}: missing node_id")
            continue
        if node in records:
            errors.append(f"{path.name}: duplicate node_id {node!r}")
            continue
        records[node] = rec
    return records, errors


def _link(rec, field):
    link = rec.get(field)
    if link is None:
        return None
    if not isinstance(link, dict) or not isinstance(link.get("node_id"), str):
        return ""
    return link["node_id"]


def check_reciprocal(records):
    errors = []
    for node, rec in sorted(records.items()):
        for field, back in (("successor", "predecessor"), ("predecessor", "successor")):
            link = rec.get(field)
            target = _link(rec, field)
            if target is None:
                continue
            if target == "":
                errors.append(f"{node}: malformed {field} link")
            elif target not in records:
                errors.append(f"{node}: {field} {target!r} has no record")
            elif _link(records[target], back) != node:
                errors.append(f"{node}: {field} {target!r} does not link back as {back}")
            if isinstance(link, dict) and not _http_url(link.get("url")):
                errors.append(f"{node}: {field} url is not an http(s) URL")
    return errors


def check_ownership(records):
    errors = []
    for node, rec in sorted(records.items()):
        owner = rec.get("owner")
        if owner is None:
            continue
        if not isinstance(owner, dict):
            errors.append(f"{node}: owner without an id")
            continue
        extra = sorted(set(owner) - {"id", "evidence_url", "released_url"})
        if extra:
            errors.append(f"{node}: owner allows only id, evidence_url, and released_url")
        evidence = owner.get("evidence_url")
        if "evidence_url" not in owner or not isinstance(evidence, str) or evidence == "":
            errors.append(f"{node}: owner {owner.get('id')!r} has no PICKUP/assignment comment url")
            continue
        if not isinstance(owner.get("id"), str) or owner.get("id") == "":
            errors.append(f"{node}: owner without an id")
        if not _comment_url(evidence):
            errors.append(f"{node}: owner {owner.get('id')!r} has no PICKUP/assignment comment url")
        if "released_url" in owner and not _comment_url(owner.get("released_url")):
            errors.append(f"{node}: owner released_url is not a comment URL")
    return errors


def _carried(node, rec, errors):
    raw = rec.get("carried_forward")
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        errors.append(f"{node}: carried_forward is not a list")
        return {}
    by_id = {}
    for item in raw:
        if not isinstance(item, dict):
            errors.append(f"{node}: carried_forward item is not a dict")
            continue
        extra = sorted(set(item) - {"id", "text", "source_url", "status", "resolution_url"})
        if extra:
            errors.append(
                f"{node}: carried_forward item allows only id, text, source_url, status, and resolution_url")
        item_id = item.get("id")
        if not isinstance(item_id, str) or item_id == "":
            errors.append(f"{node}: carried_forward item without an id")
        elif item_id in by_id:
            errors.append(f"{node}: duplicate carried_forward id {item_id!r}")
        else:
            by_id[item_id] = item
        text = item.get("text")
        if not isinstance(text, str) or text == "":
            errors.append(f"{node}: item {item_id!r} has no text")
        source = item.get("source_url")
        if not isinstance(source, str) or not source:
            errors.append(f"{node}: item {item_id!r} has no source_url")
        elif not _carried_source_url(source):
            errors.append(f"{node}: item {item_id!r} source_url is not a GitHub issue or comment URL")
        if not item_id:
            continue
        if item.get("status") == "resolved" and not _comment_url(item.get("resolution_url")):
            errors.append(f"{node}: item {item_id!r} resolved without a resolution_url")
        elif item.get("status") not in ("open", "resolved"):
            errors.append(f"{node}: item {item_id!r} has status {item.get('status')!r}")
    return by_id


def check_carried_forward(records):
    errors = []
    parsed = {}
    for node, rec in sorted(records.items()):
        parsed[node] = _carried(node, rec, errors)
    for node, rec in sorted(records.items()):
        succ = _link(rec, "successor")
        if not succ or succ not in records:
            continue
        carried = parsed[succ]
        for item_id, item in parsed[node].items():
            if item_id and item.get("status") == "open" and item_id not in carried:
                errors.append(f"{node}: open item {item_id!r} not carried forward to {succ!r}")
    return errors


def check_refresh_error(records):
    errors = []
    for node, rec in sorted(records.items()):
        if "refresh_error" in rec and rec["refresh_error"] is not None and not isinstance(rec["refresh_error"], str):
            errors.append(f"{node}: refresh_error must be a string or null")
    return errors


def check(directory):
    records, errors = load_records(directory)
    return (errors + check_reciprocal(records) + check_ownership(records)
            + check_carried_forward(records) + check_refresh_error(records))


def main(argv):
    if len(argv) != 2:
        print("usage: check_board_links.py RECORDS_DIR", file=sys.stderr)
        return 2
    if not Path(argv[1]).is_dir():
        print(f"not a directory: {argv[1]}", file=sys.stderr)
        return 2
    errors = check(argv[1])
    print(json.dumps({"ok": not errors, "errors": errors, "scientific_effect": "NONE"}, indent=1))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
