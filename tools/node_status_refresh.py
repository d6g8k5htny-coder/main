#!/usr/bin/env python3
"""Recompute freshness for node-status records (LANES-20261022 Lane 3).

Interface v0 is #495's schema at 5a58e3e0 (architecture/node-status.schema.json).
Where that schema differs from the Lane 3 note, this tool follows #495:

  - source_rev is one 40-hex commit. Comma-joining is refused.
  - A source is valid only as a pinned blob or tree URL on
    d6g8k5htny-coder/main or Math-, and sources[].rev must be the 40-hex
    SHA embedded in that URL. Issue, pull, and comment updated_at strings
    are not source revs (#495's validator allows only the pinned form).
  - #495 does not say which SHA is source_rev when sources disagree.
    This tool uses the current rev of the source with the latest observed
    update. The observed update is the moving-ref committer time (the
    repository default branch, not the pinned SHA). Equal times: the later
    source in the list wins. That choice is not defined by #495.
  - #495's record shape has no refresh_error and no owner.released_url.
    The note still requires both: refresh_error is a string or null, and a
    later RELEASE comment that names the owner's PICKUP sets
    owner.released_url without changing any other owner field.

Only status, observed_at, source_rev, refresh_error, and that one owner
field are written. status is freshness only, never scientific validity.
scientific_effect stays "NONE". Owner identity is never inferred.

Rules:
  - every source rev unchanged             -> "fresh"
  - any source rev changed                 -> "stale"
  - any source unreachable / unparseable   -> "unknown" (never "fresh")
  - empty, missing, or non-40-hex rev      -> "unknown" (never "fresh")
  - anything other than a pinned blob/tree -> "unknown" (never "fresh")
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
from urllib.parse import quote

SCHEMA_VERSION = 0
STATUSES = ("fresh", "stale", "unknown")
API = "https://api.github.com"
_COMMENT_PAGE = 100
_COMMENT_PAGE_LIMIT = 10

_SHA = re.compile(r"^[0-9a-f]{40}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_PINNED = re.compile(
    r"^https://github\.com/d6g8k5htny-coder/(main|Math-)/(blob|tree)/"
    r"([0-9a-f]{40})/([A-Za-z0-9_./-]+)$")
_COMMENT = re.compile(
    r"^https://github\.com/d6g8k5htny-coder/(main|Math-)/(issues|pull)/"
    r"([1-9][0-9]*)#issuecomment-([1-9][0-9]*)$")
_RELEASE = re.compile(r"(?im)^\s*\**\s*RELEASE\b")

Fetcher = Callable[[str], object]
Comments = Callable[[str], list]


class SourceError(Exception):
    """A source could not be resolved to a current revision."""


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_z(value: object) -> str:
    if not isinstance(value, str):
        return ""
    if value.endswith("+00:00"):
        value = value[:-6] + "Z"
    return value


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


def github_fetch(url: str) -> tuple[str, str]:
    """Return (moving-ref SHA, committer time) for a pinned blob or tree URL.

    The pin in the URL is the recorded rev. Freshness compares it with the
    latest commit on the repository default branch that touches the path.
    """
    m = _PINNED.match(url)
    if not m:
        raise SourceError(f"unsupported source url: {url}")
    repo, path = m.group(1), m.group(4)
    query = f"/repos/d6g8k5htny-coder/{repo}/commits?path={quote(path, safe='/')}&per_page=1"
    commits = _api_get(query)
    if not isinstance(commits, list) or not commits or not isinstance(commits[0], dict):
        raise SourceError(f"no commits for path: {url}")
    sha = commits[0].get("sha")
    commit = commits[0].get("commit")
    committer = commit.get("committer") if isinstance(commit, dict) else None
    observed = _as_z(committer.get("date") if isinstance(committer, dict) else "")
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise SourceError(f"{url}: current revision is not a 40-hex sha")
    if not _UTC.fullmatch(observed):
        raise SourceError(f"{url}: missing observed update time")
    return sha, observed


def github_comments(evidence_url: str) -> list[dict]:
    """Read issue comments for a PICKUP URL. One short page means the read ended."""
    m = _COMMENT.fullmatch(evidence_url)
    if not m:
        raise SourceError(f"unsupported owner evidence url: {evidence_url}")
    repo, number = m.group(1), m.group(3)
    found: list[dict] = []
    page = 1
    while True:
        batch = _api_get(
            f"/repos/d6g8k5htny-coder/{repo}/issues/{number}/comments?per_page={_COMMENT_PAGE}&page={page}")
        if not isinstance(batch, list):
            raise SourceError("comment listing is not a list")
        for item in batch:
            if not isinstance(item, dict):
                raise SourceError("comment is not an object")
            found.append({"url": item.get("html_url"), "body": item.get("body") or ""})
        if len(batch) < _COMMENT_PAGE:
            return found
        page += 1
        if page > _COMMENT_PAGE_LIMIT:
            raise SourceError("comment listing incomplete")


def _release_url(evidence_url: str, thread: object) -> str | None:
    """URL of the latest RELEASE comment that names this PICKUP and follows it."""
    pickup = _COMMENT.fullmatch(evidence_url)
    if not pickup or not isinstance(thread, list):
        return None
    pickup_id = int(pickup.group(4))
    best_id = pickup_id
    best_url = None
    for item in thread:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        body = item.get("body") if isinstance(item.get("body"), str) else ""
        if not isinstance(url, str):
            continue
        comment = _COMMENT.fullmatch(url)
        if not comment:
            continue
        comment_id = int(comment.group(4))
        if comment_id <= pickup_id or comment_id <= best_id:
            continue
        if not _RELEASE.search(body):
            continue
        if not re.search(rf"(?i)\bPICKUP\b[\s:#-]*{pickup_id}\b", body):
            continue
        best_id = comment_id
        best_url = url
    return best_url


def _current(fetched: object) -> tuple[object, str]:
    if isinstance(fetched, tuple) and len(fetched) == 2:
        return fetched[0], _as_z(fetched[1])
    return fetched, ""


def _apply_release(out: dict, record: dict, comments: Comments | None, errors: list[str]) -> None:
    owner = record.get("owner")
    if not isinstance(owner, dict):
        return
    evidence = owner.get("evidence_url")
    if not isinstance(evidence, str) or not _COMMENT.fullmatch(evidence):
        return
    try:
        thread = github_comments(evidence) if comments is None else comments(evidence)
    except Exception as exc:  # incomplete or failed read stays unknown
        errors.append(f"{evidence}: {exc}")
        return
    if not isinstance(thread, list):
        errors.append(f"{evidence}: comment listing is not a list")
        return
    released = _release_url(evidence, thread)
    if not released:
        return
    updated = dict(owner)
    updated["released_url"] = released
    out["owner"] = updated


def refresh_record(record: dict, fetch: Fetcher, now: str | None = None,
                   comments: Comments | None = None) -> dict:
    """Return a new record with freshness recomputed. Never raises for source errors."""
    out = dict(record)
    out.pop("refresh_error", None)
    out["observed_at"] = now or _utc_now()
    out["scientific_effect"] = "NONE"
    errors: list[str] = []
    changed = False
    current: list[tuple[str, str]] = []
    sources = record.get("sources")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {record.get('schema_version')!r}")
    if not isinstance(sources, list) or not sources:
        errors.append("no sources to check")
        sources = []
    for src in sources:
        url = src.get("url") if isinstance(src, dict) else None
        pinned = _PINNED.match(url) if isinstance(url, str) else None
        if not pinned:
            errors.append(f"{url}: source url is not a pinned blob or tree on main or Math-")
            continue
        recorded = src.get("rev")
        if not isinstance(recorded, str) or not _SHA.fullmatch(recorded):
            errors.append(f"{url}: empty or missing source revision")
            continue
        if recorded != pinned.group(3):
            errors.append(f"{url}: source revision must appear in its url")
            continue
        try:
            fetched = fetch(url)
        except Exception as exc:  # any fetch failure is unknown, never fresh
            errors.append(f"{url}: {exc}")
            continue
        rev, observed = _current(fetched)
        if not isinstance(rev, str) or not _SHA.fullmatch(rev):
            errors.append(f"{url}: current revision is not a 40-hex sha")
            continue
        if not _UTC.fullmatch(observed):
            errors.append(f"{url}: missing observed update time")
            continue
        current.append((rev, observed))
        if rev != recorded:
            changed = True
    _apply_release(out, record, comments, errors)
    if errors:
        out["status"] = "unknown"
        out["refresh_error"] = "; ".join(errors)
    else:
        best_rev, best_at = current[0]
        for rev, observed in current[1:]:
            if observed >= best_at:
                best_rev, best_at = rev, observed
        out["status"] = "stale" if changed else "fresh"
        out["source_rev"] = best_rev
        out["refresh_error"] = None
    return out


def refresh_dir(directory: Path, fetch: Fetcher, write: bool, now: str | None = None,
                comments: Comments | None = None) -> list[dict]:
    report = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
        except Exception as exc:
            report.append({"file": path.name, "status": "unknown", "refresh_error": f"unreadable: {exc}"})
            continue
        new = refresh_record(record, fetch, now, comments)
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
