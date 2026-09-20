#!/usr/bin/env python3
"""Query the Google Drive source-map inventory.

``drive/inventory.jsonl`` holds one JSON object per Drive item (4,456 entries
from the 2026-09-17 accessibility snapshot): id, title, path, mimeType, bytes,
sha256, context, access_status, link — an 8-field projection of the source
map's ``Files.csv``.  The tool queries three of the source map's seven tables
(Files, via inventory.jsonl; Archive Members; Exceptions); Payloads is read by
tools/frozen_check.py.  The remaining three -- Start Here, Reading Copies and
Reading Links -- are held as byte-exact CSV exports under
``drive/deltas/2026-09-18/07_MODEL_ACCESSIBILITY_extras/`` and this tool does
not read them.  Until 2026-09-20 this docstring said they were "not in the
repository", which was true when it was written and stopped being true when
``Start_Here.csv`` and ``Reading_Copies.csv``, and then ``Reading_Links.csv``,
were stored.  It is an index of the Drive's ``07_MODEL_ACCESSIBILITY``
snapshot, not of the R17 **File Catalog** tab (``registers/json/file_catalog.json``,
2,952 rows), which is a different snapshot with a different row set.

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

Paths after the snapshot.  The inventory is an export and is never edited. When
the Drive moves or renames things afterwards, the moves are recorded as a delta
under ``drive/deltas/<date>/PATH_CHANGES.jsonl`` (one row per affected item:
id, snapshot path, live path), derived from the Drive session's own rollback
record and cross-checked against the inventory.  By default this tool overlays
those deltas, so ``path`` is the live path and ``path_snapshot`` the export's;
``--snapshot`` shows the export exactly as published.  Identity is by Drive id
and SHA-256; a path is a navigation label.
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


DELTAS = os.path.join(ROOT, "drive", "deltas")


def path_change_files(deltas_dir: str = DELTAS) -> list[str]:
    """Every dated PATH_CHANGES.jsonl, oldest date first."""
    if not os.path.isdir(deltas_dir):
        return []
    out = []
    for date in sorted(os.listdir(deltas_dir)):
        f = os.path.join(deltas_dir, date, "PATH_CHANGES.jsonl")
        if os.path.exists(f):
            out.append(f)
    return out


def load_path_changes(files: list[str], known_ids: set[str] | None = None) -> dict[str, dict]:
    """id -> {"date", "path_snapshot", "path_live"}. Later dates win.

    Fails closed: a row naming an id the inventory does not have, or missing
    either path, is an error — a delta that cannot be tied to the export must
    not silently relabel anything.
    """
    changes: dict[str, dict] = {}
    for f in files:
        date = os.path.basename(os.path.dirname(f))
        with open(f, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                fid = row.get("id")
                # the live path is keyed by the delta's own date; the snapshot
                # path is the other path_* key (the export's date)
                live_key = "path_" + date.replace("-", "_")
                new_p = row.get(live_key) or row.get("path_live")
                olds = [v for k, v in row.items()
                        if k.startswith("path_") and k not in (live_key, "path_live")]
                old_p = olds[0] if olds else None
                if not fid or not old_p or not new_p or old_p == new_p:
                    raise ValueError(f"{f}:{n}: a path change needs id, the snapshot path and the live path")
                if known_ids is not None and fid not in known_ids:
                    raise ValueError(f"{f}:{n}: id {fid!r} is not in the inventory; refusing to overlay it")
                changes[fid] = {"date": date, "path_snapshot": old_p, "path_live": new_p}
    return changes


def load(path: str = INVENTORY, overlay: bool = True) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if not overlay:
        return entries
    changes = load_path_changes(path_change_files(), {e["id"] for e in entries})
    for e in entries:
        c = changes.get(e["id"])
        if c is None:
            continue
        if c["path_snapshot"] != e["path"]:
            raise ValueError(f"path change for {e['id']} records snapshot path {c['path_snapshot']!r} "
                             f"but the inventory has {e['path']!r}")
        e["path_snapshot"] = e["path"]
        e["path"] = c["path_live"]
        e["moved"] = c["date"]
    return entries


def moved_mark(e: dict) -> str:
    return f"  [moved {e['moved']}; snapshot: {e['path_snapshot']}]" if e.get("moved") else ""


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
        if q in e["title"].lower() or q in e["path"].lower() or q in e.get("path_snapshot", "").lower():
            print(f"{e['id']}\t{human(e['bytes']):>8}\t{e['path']}{moved_mark(e)}")


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
    moved = [e for e in entries if e.get("moved")]
    if moved:
        dates = sorted({e["moved"] for e in moved})
        print(f"paths overlaid from drive/deltas: {len(moved)} items moved or renamed after the "
              f"snapshot ({', '.join(dates)}); ids and digests unchanged; --snapshot shows the export")
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
    ap.add_argument("--snapshot", action="store_true",
                    help="show the 2026-09-17 export's paths without the drive/deltas overlay")
    a = ap.parse_args()
    if a.cmd in ("archive", "exceptions"):
        return cmd_archive(a.arg) if a.cmd == "archive" else cmd_exceptions(a.arg)
    entries = load(overlay=not a.snapshot)
    return {
        "find": lambda: cmd_find(entries, a.arg),
        "id": lambda: cmd_id(entries, a.arg),
        "sha": lambda: cmd_sha(entries, a.arg),
        "tree": lambda: cmd_tree(entries, a.arg, a.depth),
        "stats": lambda: cmd_stats(entries),
    }[a.cmd]() or 0


if __name__ == "__main__":
    raise SystemExit(main())
