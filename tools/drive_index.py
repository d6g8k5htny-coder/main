#!/usr/bin/env python3
"""Query the Google Drive source-map inventory.

``drive/inventory.jsonl`` holds one JSON object per Drive item (4,456 entries
from the 2026-09-17 accessibility snapshot): id, title, path, mimeType, bytes,
sha256, context, access_status, link.  This is the git-side equivalent of the
R17 **File Catalog** tab and of the Drive's `07_MODEL_ACCESSIBILITY` source map.

Usage:
    python3 tools/drive_index.py find <substring>     # title or path search
    python3 tools/drive_index.py id <drive-id>        # one entry
    python3 tools/drive_index.py sha <sha256-prefix>  # resolve a digest
    python3 tools/drive_index.py tree [<path-prefix>] [--depth N]
    python3 tools/drive_index.py stats
    python3 tools/drive_index.py archive <carrier-substring>   # members of a zip
    python3 tools/drive_index.py exceptions [<type>]

The inventory is a metadata snapshot.  Presence or absence in it never proves a
mathematical claim, and never overrides a register status
(see governance/GIT_ADAPTATION.md).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import signal
import sys

# allow piping into head/less without a BrokenPipeError traceback
try:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except (AttributeError, ValueError):  # pragma: no cover - non-POSIX
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INVENTORY = os.path.join(ROOT, "drive", "inventory.jsonl")
ARCHIVES = os.path.join(ROOT, "drive", "source_map", "Archive_Members.csv")
EXCEPTIONS = os.path.join(ROOT, "drive", "source_map", "Exceptions.csv")
FOLDER_MIME = "application/vnd.google-apps.folder"


def load(path: str = INVENTORY) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def human(n: int | None) -> str:
    if not n:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return ""


def cmd_find(entries, q):
    q = (q or "").lower()
    for e in entries:
        if q in e["title"].lower() or q in e["path"].lower():
            print(f"{e['id']}\t{human(e['bytes']):>8}\t{e['path']}")


def cmd_id(entries, fid):
    for e in entries:
        if e["id"] == fid:
            print(json.dumps(e, indent=2, ensure_ascii=False))
            return 0
    print(f"not in inventory: {fid}", file=sys.stderr)
    return 1


def cmd_sha(entries, prefix):
    prefix = (prefix or "").lower()
    hits = [e for e in entries if e.get("sha256") and e["sha256"].lower().startswith(prefix)]
    for e in hits:
        print(f"{e['sha256']}\t{e['path']}")
    if not hits:
        # fall back to archive member payload hashes
        with open(ARCHIVES, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["Payload SHA-256"].lower().startswith(prefix):
                    print(f"{r['Payload SHA-256']}\t{r['Carrier title']}!{r['Member path']}")
    return 0


def cmd_tree(entries, prefix, depth):
    prefix = prefix or ""
    seen = set()
    base = prefix.count("/") + (1 if prefix else 0)
    for e in sorted(entries, key=lambda x: x["path"]):
        if not e["path"].startswith(prefix):
            continue
        parts = e["path"].split("/")
        if len(parts) - base > depth:
            continue
        key = "/".join(parts)
        if key in seen:
            continue
        seen.add(key)
        indent = "  " * (len(parts) - 1)
        mark = "/" if e["mimeType"] == FOLDER_MIME else ""
        print(f"{indent}{parts[-1]}{mark}")


def cmd_stats(entries):
    files = [e for e in entries if e["mimeType"] != FOLDER_MIME]
    folders = len(entries) - len(files)
    total = sum(e["bytes"] or 0 for e in files)
    print(f"items={len(entries)} files={len(files)} folders={folders} bytes={total:,}")
    print("\nby lane (top-level path):")
    lanes = collections.Counter(e["path"].split("/")[0] for e in entries)
    for k, v in lanes.most_common():
        print(f"  {v:5d}  {k}")
    print("\nby context:")
    for k, v in collections.Counter(e["context"] for e in entries).most_common():
        print(f"  {v:5d}  {k}")
    print("\nby access status:")
    for k, v in collections.Counter(e["access_status"] for e in entries).most_common():
        print(f"  {v:5d}  {k}")
    print("\nby mime type:")
    for k, v in collections.Counter(e["mimeType"] for e in entries).most_common():
        print(f"  {v:5d}  {k}")


def cmd_archive(q):
    q = (q or "").lower()
    n = 0
    with open(ARCHIVES, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if q in r["Carrier title"].lower():
                n += 1
                print(f"{r['Payload SHA-256'][:16]}\t{r['Bytes']:>9}\t{r['Member path']}")
    print(f"-- {n} members", file=sys.stderr)


def cmd_exceptions(t):
    with open(EXCEPTIONS, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if t:
        rows = [r for r in rows if r["Type"] == t]
    for r in rows:
        print(f"{r['Type']}\t{r['Source or block']}\t{r['Details'][:100]}")
    print(f"-- {len(rows)} entries", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["find", "id", "sha", "tree", "stats", "archive", "exceptions"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--depth", type=int, default=2)
    a = ap.parse_args()
    if a.cmd in ("archive", "exceptions"):
        return cmd_archive(a.arg) if a.cmd == "archive" else cmd_exceptions(a.arg)
    entries = load()
    return {
        "find": lambda: cmd_find(entries, a.arg),
        "id": lambda: cmd_id(entries, a.arg),
        "sha": lambda: cmd_sha(entries, a.arg),
        "tree": lambda: cmd_tree(entries, a.arg, a.depth),
        "stats": lambda: cmd_stats(entries),
    }[a.cmd]() or 0


if __name__ == "__main__":
    raise SystemExit(main())
