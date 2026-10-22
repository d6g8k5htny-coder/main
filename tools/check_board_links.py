#!/usr/bin/env python3
"""Read-only check of board and session handoff links (LANES-20261022 Lane 4).

Input: a directory of JSON records shaped like node-status interface v0
(records/node_status/<node_id>.json) with three optional handoff fields:

  predecessor:     {"node_id": str, "url": http(s) URL} | null
  successor:       {"node_id": str, "url": http(s) URL} | null
  carried_forward: [{"id": str, "text": str, "source_url": str,
                     "status": "open" | "resolved",
                     "resolution_url": str (required when resolved)}]

source_url and resolution_url are comment URLs. Every comment URL uses
COMMENT_URL below: owner evidence, source_url, resolution_url, and released_url.

Checks (each failure is reported; the exit status is 1 if any):
  1. reciprocity: A.successor -> B requires B.predecessor -> A, and the reverse;
     a link to a node with no record is a failure, not a skip; a link url that
     is not http(s) is a failure;
  2. ownership: a non-null owner without released_url needs an id and an
     evidence_url matching COMMENT_URL (the PICKUP or assignment). An owner
     with released_url is not current, so those checks do not apply;
     released_url itself must match COMMENT_URL. Ownership is never inferred,
     and moving a board does not move it;
  3. carry-forward: carried_forward must be a list of objects with unique ids
     and a source_url. Every predecessor item that is open must appear, by id,
     in the successor's carried_forward, either still open or resolved with a
     resolution_url. Items may not be silently dropped.

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


def _http_url(url):
    if not isinstance(url, str) or not url or any(ch.isspace() for ch in url):
        return False
    parsed = urlparse(url)
    return parsed.scheme.lower() in ("http", "https") and bool(parsed.netloc)


def _comment_url(url):
    return isinstance(url, str) and COMMENT_URL.match(url) is not None


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
        if "released_url" in owner:
            if not _comment_url(owner.get("released_url")):
                errors.append(f"{node}: owner released_url is not a comment URL")
            continue
        if not owner.get("id"):
            errors.append(f"{node}: owner without an id")
            continue
        if not _comment_url(owner.get("evidence_url")):
            errors.append(f"{node}: owner {owner['id']!r} has no PICKUP/assignment comment url")
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
        item_id = item.get("id")
        if not item_id:
            errors.append(f"{node}: carried_forward item without an id")
        elif item_id in by_id:
            errors.append(f"{node}: duplicate carried_forward id {item_id!r}")
        else:
            by_id[item_id] = item
        source = item.get("source_url")
        if not isinstance(source, str) or not source:
            errors.append(f"{node}: item {item_id!r} has no source_url")
        elif not _comment_url(source):
            errors.append(f"{node}: item {item_id!r} source_url is not a comment URL")
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


def check(directory):
    records, errors = load_records(directory)
    return errors + check_reciprocal(records) + check_ownership(records) + check_carried_forward(records)


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
