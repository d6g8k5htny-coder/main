#!/usr/bin/env python3
"""Read-only check of board and session handoff links (LANES-20261022 Lane 4).

Input: a directory of JSON records shaped like node-status interface v0
(records/node_status/<node_id>.json) with three optional handoff fields:

  predecessor:     {"node_id": str, "url": str} | null
  successor:       {"node_id": str, "url": str} | null
  carried_forward: [{"id": str, "text": str, "source_url": str,
                     "status": "open" | "resolved",
                     "resolution_url": str (required when resolved)}]

Checks (each failure is reported; the exit status is 1 if any):
  1. reciprocity: A.successor -> B requires B.predecessor -> A, and the reverse;
     a link to a node with no record is a failure, not a skip;
  2. ownership: a non-null owner needs an id and an evidence_url that is a
     GitHub issue/PR comment URL (the PICKUP or assignment). Ownership is never
     inferred, and moving a board does not move it;
  3. carry-forward: every predecessor item that is open must appear, by id, in
     the successor's carried_forward, either still open or resolved with a
     resolution_url. Items may not be silently dropped.

The tool reads files only. It makes no network call and writes nothing.
It checks link structure, not the truth of any linked comment, and has no
scientific effect.
"""

import json
import re
import sys
from pathlib import Path

COMMENT_URL = re.compile(
    r"^https://github\.com/[\w.-]+/[\w.-]+/(?:issues|pull)/\d+#(?:issuecomment|pullrequestreview)-\d+$"
)


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
            target = _link(rec, field)
            if target is None:
                continue
            if target == "":
                errors.append(f"{node}: malformed {field} link")
            elif target not in records:
                errors.append(f"{node}: {field} {target!r} has no record")
            elif _link(records[target], back) != node:
                errors.append(f"{node}: {field} {target!r} does not link back as {back}")
    return errors


def check_ownership(records):
    errors = []
    for node, rec in sorted(records.items()):
        owner = rec.get("owner")
        if owner is None:
            continue
        if not isinstance(owner, dict) or not owner.get("id"):
            errors.append(f"{node}: owner without an id")
            continue
        url = owner.get("evidence_url")
        if not isinstance(url, str) or not COMMENT_URL.match(url):
            errors.append(f"{node}: owner {owner['id']!r} has no PICKUP/assignment comment url")
    return errors


def _items(rec):
    items = rec.get("carried_forward") or []
    return {i.get("id"): i for i in items if isinstance(i, dict)}


def check_carried_forward(records):
    errors = []
    for node, rec in sorted(records.items()):
        for item_id, item in _items(rec).items():
            if not item_id:
                errors.append(f"{node}: carried_forward item without an id")
            elif item.get("status") == "resolved" and not COMMENT_URL.match(str(item.get("resolution_url", ""))):
                errors.append(f"{node}: item {item_id!r} resolved without a resolution_url")
            elif item.get("status") not in ("open", "resolved"):
                errors.append(f"{node}: item {item_id!r} has status {item.get('status')!r}")
        succ = _link(rec, "successor")
        if not succ or succ not in records:
            continue
        carried = _items(records[succ])
        for item_id, item in _items(rec).items():
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
