#!/usr/bin/env python3
"""Search the reconciled Drive snapshot, archive occurrences and eligible text.

Default commands derive one read-only view from the original inventory,
verified migration coverage, dated reconciliation and later delivery records.
`--snapshot` retains the original 2026-09-17 metadata view. Neither is a live
account inventory or scientific verdict. See docs/DRIVE_SEARCH_GUIDE.md.

The original metadata snapshot has 4,456 items. Its Start Here, Reading Copies
and Reading Links exports are held byte-exact as Start_Here.csv,
Reading_Copies.csv and Reading_Links.csv under
``drive/deltas/2026-09-18/07_MODEL_ACCESSIBILITY_extras/``. This tool does not
read those three exports. The R17 File Catalog is a separate snapshot.

Examples:
    python3 tools/drive_index.py find rnu_ds3.py
    python3 tools/drive_index.py search Cholesky --json
    python3 tools/drive_index.py sha ac89f60b8206 --mentions --json
    python3 tools/drive_index.py id <drive-id>
    python3 tools/drive_index.py stats
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import signal
from pathlib import Path
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


def path_change_files(deltas_dir: str | None = None) -> list[str]:
    """Every dated PATH_CHANGES.jsonl, oldest date first."""
    deltas_dir = DELTAS if deltas_dir is None else deltas_dir
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


def load(path: str | None = None, overlay: bool = True) -> list[dict]:
    path = INVENTORY if path is None else path
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
    from tools.drive_search import archive_records, sha_matches
    normalized = [{**e, 'hashes': e.get('hashes', [{'sha256':e['sha256'],
                   'kind':'SOURCE_MAP_REPORTED_SHA256'}] if e.get('sha256') else [])}
                  for e in entries]
    result = sha_matches(normalized + archive_records(), prefix)
    for e in result['matches']:
        print(json.dumps(e, ensure_ascii=False))
    return 2 if result['ambiguous'] else (0 if result['matches'] else 1)


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
    ap.add_argument("cmd", choices=["find", "search", "id", "sha", "tree", "stats", "archive", "exceptions"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--snapshot", action="store_true",
                    help="show the 2026-09-17 export's paths without the drive/deltas overlay")
    ap.add_argument("--json", action="store_true", help="structured search result")
    ap.add_argument("--mentions", action="store_true", help="include separately labeled SHA text mentions")
    ap.add_argument("--build-text-cache", action="store_true", help="rebuild derived eligible text index")
    a = ap.parse_args()
    sys.path.insert(0, ROOT)
    from tools import drive_search as ds
    if a.cmd in ("archive", "exceptions"):
        return cmd_archive(a.arg) if a.cmd == "archive" else cmd_exceptions(a.arg)
    entries = load(overlay=not a.snapshot)
    if not a.snapshot:
        entries = ds.load_current(entries)
    else:
        entries = [{**e, 'record_key':'file:'+e['id'], 'record_type':'FILE',
                    'source_role':'HISTORICAL_SOURCE_MAP_METADATA',
                    'hashes':[{'sha256':e['sha256'],'kind':'SOURCE_MAP_REPORTED_SHA256',
                               'snapshot':'2026-09-17'}] if e.get('sha256') else []}
                   for e in entries]
    if a.build_text_cache:
        if a.snapshot: ap.error('text cache uses reconciled view only')
        print(json.dumps(ds.build_text_cache(entries), sort_keys=True))
        return 0
    if a.cmd in ('find', 'search', 'sha'):
        try:
            records = entries + ds.archive_records()
            if a.cmd == 'sha':
                result = ds.sha_matches(records, a.arg)
                if a.mentions:
                    if a.snapshot: ap.error('text mentions use reconciled view only')
                    result['mentions'] = ds.text_matches(entries, result['query'], hash_mention=True)
                code = 2 if result['ambiguous'] else (0 if result['matches'] else 1)
            else:
                hits = ds.keyword_matches(records, a.arg)
                if a.cmd == 'search':
                    if a.snapshot: ap.error('content search uses reconciled view only')
                    hits += ds.text_matches(entries, a.arg)
                result = {'query':a.arg, 'matches':hits, 'match_count':len(hits)}
                code = 0 if hits else 1
            result['matches'] = [ds.public_record(e) for e in result['matches']]
            if 'mentions' in result:
                result['mentions'] = [ds.public_record(e) for e in result['mentions']]
            result['view'] = 'SOURCE_MAP_20260917' if a.snapshot else 'RECONCILED_PLUS_DATED_DELIVERIES'
            result['whole_account_complete'] = False
            if a.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                for e in result['matches'] + result.get('mentions', []):
                    print(f"{e['match_kind']}\t{e['record_key']}\t{e.get('sha256','')}\t{e['source_role']}\t{e['path']}")
                print(f"{result['match_count']} matches; ambiguous={result.get('ambiguous',False)}", file=sys.stderr)
            return code
        except (ValueError, OSError) as exc:
            ap.error(str(exc))
    return {
        "find": lambda: cmd_find(entries, a.arg),
        "id": lambda: cmd_id(entries, a.arg),
        "sha": lambda: cmd_sha(entries, a.arg),
        "tree": lambda: cmd_tree(entries, a.arg, a.depth),
        "stats": lambda: cmd_stats(entries),
    }[a.cmd]() or 0


if __name__ == "__main__":
    raise SystemExit(main())
