#!/usr/bin/env python3
"""Recompute freshness for node-status records (LANES-20261022 Lane 3).

Interface: Lane 2 v0 (main#475 comment 6346276856), one JSON record per node:
  {schema_version:0, node_id, owner, next_action, missing_input, stops,
   sources:[{url, rev}], source_rev, observed_at, status, scientific_effect:"NONE"}

Only status, observed_at, source_rev (and, on failure, refresh_error) are recomputed.
status is about freshness only. It is NEVER a statement of scientific validity;
scientific_effect stays "NONE". Owner fields are never touched or inferred.

Rules:
  - every source rev unchanged             -> "fresh"
  - any source rev changed                 -> "stale"
  - any source unreachable / unparseable   -> "unknown" (error recorded; never "fresh")
  - empty or missing source rev            -> "unknown" (never "fresh")
Precedence: unknown > stale > fresh.

The default fetcher reads the public GitHub REST API (read-only, optional
GITHUB_TOKEN from the workflow). Tests inject an offline fetcher.
Report-only: without --write, records are not modified.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Callable

SCHEMA_VERSION = 0
STATUSES = ("fresh", "stale", "unknown")
API = "https://api.github.com"

Fetcher = Callable[[str], str]


class SourceError(Exception):
    """A source could not be resolved to a current revision."""


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _api_get(path: str) -> object:
    req = urllib.request.Request(API + path, headers={"Accept": "application/vnd.github+json",
                                                      "User-Agent": "node-status-refresh"})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.load(resp)
    except Exception as exc:  # network, HTTP, JSON
        raise SourceError(f"{type(exc).__name__}: {exc}") from exc


_GH = re.compile(r"^https://github\.com/([^/]+)/([^/#?]+)(/.*)?$")


def github_fetch(url: str) -> str:
    """Return the current revision string for a GitHub URL (read-only)."""
    m = _GH.match(url)
    if not m:
        raise SourceError(f"unsupported source url: {url}")
    owner, repo, rest = m.group(1), m.group(2), m.group(3) or ""
    base = f"/repos/{owner}/{repo}"
    c = re.search(r"#(?:issuecomment|discussion_r)-(\d+)$", rest)
    if c:
        kind = "issues" if "issuecomment" in rest else "pulls"
        return str(_api_get(f"{base}/{kind}/comments/{c.group(1)}")["updated_at"])
    p = re.match(r"^/pull/(\d+)", rest)
    if p:
        return str(_api_get(f"{base}/pulls/{p.group(1)}")["head"]["sha"])
    i = re.match(r"^/issues/(\d+)", rest)
    if i:
        return str(_api_get(f"{base}/issues/{i.group(1)}")["updated_at"])
    b = re.match(r"^/(?:blob|tree)/([^/]+)/(.+)$", rest)
    if b:
        commits = _api_get(f"{base}/commits?sha={b.group(1)}&path={b.group(2)}&per_page=1")
        if not commits:
            raise SourceError(f"no commits for path: {url}")
        return str(commits[0]["sha"])
    if rest in ("", "/"):
        return str(_api_get(f"{base}/commits/HEAD")["sha"])
    raise SourceError(f"unsupported source url: {url}")


def refresh_record(record: dict, fetch: Fetcher, now: str | None = None) -> dict:
    """Return a new record with freshness recomputed. Never raises for source errors."""
    out = dict(record)
    out.pop("refresh_error", None)
    out["observed_at"] = now or _utc_now()
    out["scientific_effect"] = "NONE"
    errors: list[str] = []
    changed = False
    current: list[str] = []
    sources = record.get("sources")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {record.get('schema_version')!r}")
    if not isinstance(sources, list) or not sources:
        errors.append("no sources to check")
        sources = []
    for src in sources:
        url = src.get("url") if isinstance(src, dict) else None
        if not url:
            errors.append("source without url")
            continue
        recorded = src.get("rev")
        if not isinstance(recorded, str) or not recorded:
            errors.append(f"{url}: empty or missing source revision")
            continue
        try:
            rev = fetch(url)
        except Exception as exc:  # any fetch failure is unknown, never fresh
            errors.append(f"{url}: {exc}")
            continue
        if not isinstance(rev, str) or not rev:
            errors.append(f"{url}: empty revision")
            continue
        current.append(rev)
        if rev != src.get("rev"):
            changed = True
    if errors:
        out["status"] = "unknown"
        out["refresh_error"] = "; ".join(errors)
    else:
        out["status"] = "stale" if changed else "fresh"
        out["source_rev"] = current[0] if len(current) == 1 else ",".join(current)
    return out


def refresh_dir(directory: Path, fetch: Fetcher, write: bool, now: str | None = None) -> list[dict]:
    report = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
        except Exception as exc:
            report.append({"file": path.name, "status": "unknown", "refresh_error": f"unreadable: {exc}"})
            continue
        new = refresh_record(record, fetch, now)
        report.append({"file": path.name, "node_id": new.get("node_id"), "status": new["status"],
                       "previous_status": record.get("status"),
                       "refresh_error": new.get("refresh_error")})
        if write:
            path.write_text(json.dumps(new, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("records", nargs="?", default="records/node_status")
    ap.add_argument("--write", action="store_true", help="rewrite records (default: report only)")
    args = ap.parse_args(argv)
    directory = Path(args.records)
    if not directory.is_dir():
        print(json.dumps({"records": str(directory), "report": [], "note": "no records directory; nothing to refresh",
                          "scientific_effect": "NONE"}, indent=2))
        return 0
    report = refresh_dir(directory, github_fetch, args.write)
    print(json.dumps({"records": str(directory), "report": report, "scientific_effect": "NONE"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
