#!/usr/bin/env python3
"""Recompute freshness for node-status records (LANES-20261022 Lane 3).

Settled rule: source_rev is sources[0].rev, the primary source. For a pinned
blob or tree on d6g8k5htny-coder/main or Math-, that rev is the 40-hex sha in
the URL. The moving-ref sha is never written into the record. If that moving
ref has moved past source_rev, status is stale. If it cannot be resolved,
status is unknown and refresh_error says why.

#495 is adding the other three forms this tool already accepts: an updated_at
rev (UTC Z string) for an issue, pull, or comment URL on main or Math-; an
optional refresh_error (string or null); and an optional owner.released_url.
An issue, pull, or comment source is stale when the object's updated_at is
newer than its rev, and unknown when that updated_at cannot be fetched.

Only status, observed_at, source_rev (set to the primary rev, not a fetched
tip), refresh_error, and that one owner field are written. status is freshness
only, never scientific validity. scientific_effect stays "NONE". Owner identity
is never inferred.

Rules:
  - no source moved past its recorded rev   -> "fresh"
  - any source moved past its recorded rev  -> "stale"
  - any source unreachable / not a valid rev -> "unknown" (never "fresh")
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
# Bounded support for 2,500 comments plus a terminal short/empty page.
_COMMENT_PAGE_LIMIT = 26

_SHA = re.compile(r"^[0-9a-f]{40}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_PINNED = re.compile(
    r"^https://github\.com/d6g8k5htny-coder/(main|Math-)/(blob|tree)/"
    r"([0-9a-f]{40})/([A-Za-z0-9_./-]+)")
_DISCUSSION = re.compile(
    r"^https://github\.com/d6g8k5htny-coder/(main|Math-)/(issues|pull)/"
    r"([1-9][0-9]*)(?:#issuecomment-([1-9][0-9]*))?$")
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


def _classify(url: object) -> tuple[str, re.Match[str]] | tuple[None, None]:
    if not isinstance(url, str):
        return None, None
    pinned = _PINNED.fullmatch(url)
    if pinned:
        return "sha", pinned
    discussion = _DISCUSSION.fullmatch(url)
    if discussion:
        return "time", discussion
    return None, None


def _recorded_ok(kind: str | None, match: re.Match[str] | None, recorded: object) -> bool:
    if kind == "sha" and match is not None:
        return isinstance(recorded, str) and bool(_SHA.fullmatch(recorded)) and recorded == match.group(3)
    if kind == "time":
        return isinstance(recorded, str) and bool(_UTC.fullmatch(recorded))
    return False


def github_fetch(url: str) -> str:
    """Return the current moving-ref sha, or a discussion object's updated_at.

    Neither value is written into the record. Callers compare it with the
    recorded rev and keep sources[0].rev as source_rev.
    """
    kind, match = _classify(url)
    if kind == "sha" and match is not None:
        repo, path = match.group(1), match.group(4)
        query = f"/repos/d6g8k5htny-coder/{repo}/commits?path={quote(path, safe='/')}&per_page=1"
        commits = _api_get(query)
        if not isinstance(commits, list) or not commits or not isinstance(commits[0], dict):
            raise SourceError(f"no commits for path: {url}")
        sha = commits[0].get("sha")
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise SourceError(f"{url}: moving ref is not a 40-hex sha")
        return sha
    if kind == "time" and match is not None:
        repo, disc_kind, number, comment_id = match.group(1), match.group(2), match.group(3), match.group(4)
        if comment_id:
            path = f"/repos/d6g8k5htny-coder/{repo}/issues/comments/{comment_id}"
        elif disc_kind == "pull":
            path = f"/repos/d6g8k5htny-coder/{repo}/pulls/{number}"
        else:
            path = f"/repos/d6g8k5htny-coder/{repo}/issues/{number}"
        data = _api_get(path)
        if not isinstance(data, dict):
            raise SourceError(f"{url}: updated_at could not be read")
        updated = _as_z(data.get("updated_at"))
        if not _UTC.fullmatch(updated):
            raise SourceError(f"{url}: updated_at is not a UTC Z timestamp")
        return updated
    raise SourceError(f"unsupported source url: {url}")


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


def _current(fetched: object) -> object:
    if isinstance(fetched, tuple) and fetched:
        return fetched[0]
    return fetched


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


def _primary_rev(sources: object) -> str | None:
    if not isinstance(sources, list) or not sources or not isinstance(sources[0], dict):
        return None
    kind, match = _classify(sources[0].get("url"))
    recorded = sources[0].get("rev")
    if _recorded_ok(kind, match, recorded) and isinstance(recorded, str):
        return recorded
    return None


def refresh_record(record: dict, fetch: Fetcher, now: str | None = None,
                   comments: Comments | None = None) -> dict:
    """Return a new record with freshness recomputed. Never raises for source errors."""
    out = dict(record)
    out.pop("refresh_error", None)
    out["observed_at"] = now or _utc_now()
    out["scientific_effect"] = "NONE"
    errors: list[str] = []
    changed = False
    sources = record.get("sources")
    primary = _primary_rev(sources)
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {record.get('schema_version')!r}")
    if not isinstance(sources, list) or not sources:
        errors.append("no sources to check")
        sources = []
    for src in sources:
        url = src.get("url") if isinstance(src, dict) else None
        kind, match = _classify(url)
        recorded = src.get("rev") if isinstance(src, dict) else None
        if kind is None:
            errors.append(f"{url}: source url is not a pinned blob or tree, or a main/Math- issue, pull, or comment")
            continue
        if not isinstance(recorded, str) or not recorded:
            errors.append(f"{url}: empty or missing source revision")
            continue
        if not _recorded_ok(kind, match, recorded):
            if kind == "sha":
                errors.append(f"{url}: source revision must be the 40-hex sha in its url")
            else:
                errors.append(f"{url}: source revision must be a UTC Z timestamp")
            continue
        try:
            fetched = fetch(url)
        except Exception as exc:  # any fetch failure is unknown, never fresh
            errors.append(f"{url}: {exc}")
            continue
        current = _current(fetched)
        if kind == "sha":
            if not isinstance(current, str) or not _SHA.fullmatch(current):
                errors.append(f"{url}: moving ref is not a 40-hex sha")
                continue
            if current != recorded:
                changed = True
        else:
            current_z = _as_z(current)
            if not _UTC.fullmatch(current_z):
                errors.append(f"{url}: updated_at is not a UTC Z timestamp")
                continue
            if current_z > recorded:
                changed = True
    _apply_release(out, record, comments, errors)
    if primary is not None:
        out["source_rev"] = primary
    if errors:
        out["status"] = "unknown"
        out["refresh_error"] = "; ".join(errors)
    else:
        out["status"] = "stale" if changed else "fresh"
        out["refresh_error"] = None
    return out


def _unique_pairs(pairs):
    """Reject duplicate JSON names before observing or rewriting a record."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def refresh_dir(directory: Path, fetch: Fetcher, write: bool, now: str | None = None,
                comments: Comments | None = None) -> list[dict]:
    report = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
        except Exception as exc:
            report.append({"file": path.name, "status": "unknown", "refresh_error": f"unreadable: {exc}"})
            continue
        new = refresh_record(record, fetch, now, comments)
        row = {"file": path.name, "node_id": new.get("node_id"), "status": new["status"],
               "previous_status": record.get("status"),
               "refresh_error": new.get("refresh_error")}
        owner = new.get("owner")
        if isinstance(owner, dict) and isinstance(owner.get("released_url"), str) and owner["released_url"]:
            row["released_url"] = owner["released_url"]
        report.append(row)
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
