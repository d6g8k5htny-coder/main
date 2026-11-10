#!/usr/bin/env python3
"""Recompute freshness for node-status records (LANES-20261022 Lane 3).

CLI contract cited-object-content-v1: preserve immutable source pins; capture one
current default-branch commit per repository over GitHub HTTPS; compare cited
Git object type, mode and identity at the pin and observed head. A directory
citation compares the entire tree, not merely a selected child. Unknown or
incomplete responses cannot establish freshness. Discussion revisions compare
validated updated_at timestamps; a newer source is stale, an older observation
is unknown. Git and mutable-discussion observations are not atomic: reports
include start/end times, repository cuts and source observations.

The injectable refresh_record API and historical github_fetch helper retain
revision-token support for existing offline callers. The CLI uses ContentObserver,
not github_fetch's last-path-commit comparison. Report-only success is not a gate.
--check first validates production records, never writes, and returns nonzero
for empty, invalid, stale or unknown observations. Owner identity is never
inferred and scientific_effect stays NONE. This certifies no mathematics.

The default fetcher reads the public GitHub REST API (read-only, optional
GITHUB_TOKEN from the workflow). Tests inject an offline fetcher.
Report-only: without --write, records are not modified.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import hashlib
import tempfile
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
            return json.load(resp, object_pairs_hook=_unique_pairs)
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


class ContentObserver:
    """One HTTPS-observed default-head cut per repo; exact Git object identities.

    The injected API is a trust boundary for offline tests, not proof that a
    supplied transcript was fetched. Nonrecursive trees must be complete.
    Mutable comments are observed during the same run, not atomically with Git.
    """
    def __init__(self, api=None):
        self.api = _api_get if api is None else api
        self.heads = {}
        self.roots = {}
        self.trees = {}
        self.observations = []

    @staticmethod
    def sha(value):
        if not isinstance(value, str) or not _SHA.fullmatch(value):
            raise SourceError("invalid object identity")
        return value

    def root(self, repo, rev):
        key = (repo, rev)
        if key not in self.roots:
            value = self.api(f"/repos/d6g8k5htny-coder/{repo}/git/commits/{rev}")
            if not isinstance(value, dict) or value.get("sha") != rev:
                raise SourceError("commit response identity mismatch")
            self.roots[key] = self.sha(value.get("tree", {}).get("sha"))
        return self.roots[key]

    def head(self, repo):
        if repo not in self.heads:
            value = self.api(f"/repos/d6g8k5htny-coder/{repo}/commits/HEAD")
            if not isinstance(value, dict):
                raise SourceError("invalid head response")
            sha = self.sha(value.get("sha"))
            root = self.sha(value.get("commit", {}).get("tree", {}).get("sha"))
            self.heads[repo] = sha
            self.roots[(repo, sha)] = root
        return self.heads[repo]

    def tree(self, repo, sha):
        key = (repo, sha)
        if key not in self.trees:
            value = self.api(f"/repos/d6g8k5htny-coder/{repo}/git/trees/{sha}")
            if (not isinstance(value, dict) or value.get("sha") != sha
                    or value.get("truncated") is not False
                    or not isinstance(value.get("tree"), list)):
                raise SourceError("incomplete or mismatched tree response")
            entries = {}
            modes = {"100644": "blob", "100755": "blob", "120000": "blob",
                     "040000": "tree", "160000": "commit"}
            for entry in value["tree"]:
                if not isinstance(entry, dict):
                    raise SourceError("invalid tree entry")
                name = entry.get("path")
                if (not isinstance(name, str) or not name or "/" in name
                        or name in (".", "..") or name in entries
                        or modes.get(entry.get("mode")) != entry.get("type")):
                    raise SourceError("invalid or duplicate tree entry")
                entries[name] = (entry["type"], entry["mode"], self.sha(entry.get("sha")))
            self.trees[key] = entries
        return self.trees[key]

    def object(self, repo, rev, path, kind):
        parts = path.split("/")
        if any(part in ("", ".", "..") for part in parts):
            raise SourceError("noncanonical source path")
        sha = self.root(repo, rev)
        for index, part in enumerate(parts):
            entry = self.tree(repo, sha).get(part)
            if entry is None:
                raise SourceError("source path missing")
            typ, mode, sha = entry
            if index < len(parts) - 1 and typ != "tree":
                raise SourceError("source ancestor is not a tree")
        if typ != kind or mode not in ("100644", "100755", "040000"):
            raise SourceError("source kind mismatch or unsupported link")
        return entry

    def __call__(self, url):
        kind, match = _classify(url)
        if kind == "sha":
            repo, pin, path = match.group(1), match.group(3), match.group(4)
            current = self.head(repo)
            expected_kind = url.split("/")[5]
            before = self.object(repo, pin, path, expected_kind)
            after = self.object(repo, current, path, expected_kind)
            self.observations.append(dict(url=url, head=current, pinned_object=before,
                                          current_object=after))
            return {"object_equal": before == after}
        if kind == "time":
            repo, discussion, number, comment = match.groups()
            suffix = (f"issues/comments/{comment}" if comment else
                      f"{'pulls' if discussion == 'pull' else 'issues'}/{number}")
            value = self.api(f"/repos/d6g8k5htny-coder/{repo}/{suffix}")
            if not isinstance(value, dict):
                raise SourceError("invalid discussion response")
            expected = int(comment or number)
            if value.get("id" if comment else "number") != expected:
                raise SourceError("discussion response identity mismatch")
            if comment:
                parent = f"{API}/repos/d6g8k5htny-coder/{repo}/issues/{number}"
                if value.get("issue_url") != parent:
                    raise SourceError("discussion parent identity mismatch")
            updated = _as_z(value.get("updated_at"))
            if not _UTC.fullmatch(updated):
                raise SourceError("invalid discussion timestamp")
            _dt.datetime.strptime(updated, "%Y-%m-%dT%H:%M:%SZ")
            self.observations.append(dict(url=url, updated_at=updated))
            return updated
        raise SourceError("unsupported source url")


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
            if isinstance(current, dict) and set(current) == {"object_equal"} and type(current["object_equal"]) is bool:
                changed = changed or not current["object_equal"]
                continue
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
            try:
                _dt.datetime.strptime(recorded, "%Y-%m-%dT%H:%M:%SZ")
                _dt.datetime.strptime(current_z, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                errors.append(f"{url}: invalid calendar timestamp")
                continue
            if current_z < recorded:
                errors.append(f"{url}: observed timestamp predates recorded revision")
            elif current_z > recorded:
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
    ap.add_argument("--check", action="store_true", help="fail unless nonempty observed records are all fresh; never writes")
    args = ap.parse_args(argv)
    if args.check and args.write:
        ap.error("--check cannot be combined with --write")
    directory = Path(args.records)
    if not directory.is_dir():
        print(json.dumps({"records": str(directory), "report": [], "note": "no records directory; nothing to refresh",
                          "scientific_effect": "NONE"}, indent=2))
        return 1 if args.check else 0
    started = _utc_now()
    observer = ContentObserver()
    input_hashes = None
    drift = False
    if args.check:
        if __package__:
            from . import validate_node_status as validator
        else:
            import validate_node_status as validator

        def snapshot():
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("records directory missing or symlink")
            result = {}
            files = list(directory.iterdir())
            if len(files) > validator.MAX_FILES + 1:
                raise ValueError("too many node-status records")
            for path in files:
                if path.is_symlink() or not path.is_file():
                    raise ValueError("nonregular record input")
                if path.stat().st_size > validator.MAX_BYTES:
                    raise ValueError("record input too large")
                data = path.read_bytes()
                if len(data) > validator.MAX_BYTES:
                    raise ValueError("record input too large")
                result[path.name] = data
            return result

        try:
            before = snapshot()
            input_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(before.items())}
            # Validate and observe the SAME captured bytes, not separate mutable reads.
            with tempfile.TemporaryDirectory(prefix="node-status-check-") as tmp:
                captured = Path(tmp)
                for name, data in before.items():
                    (captured / name).write_bytes(data)
                problems = validator.validate_directory(Path(__file__).resolve().parents[1], captured)
                if problems:
                    print(json.dumps({"records": str(directory), "validation_errors": problems,
                                      "input_sha256": input_hashes, "scientific_effect": "NONE"}, indent=2))
                    return 1
                report = refresh_dir(captured, observer, False, now=started)
            drift = before != snapshot()
        except (OSError, ValueError) as exc:
            print(json.dumps({"records": str(directory), "input_error": str(exc),
                              "input_sha256": input_hashes, "scientific_effect": "NONE"}, indent=2))
            return 1
    else:
        report = refresh_dir(directory, observer, args.write, now=started)
    print(json.dumps({"records": str(directory), "report": report, "scientific_effect": "NONE",
                      "observation_started_at": started, "observation_finished_at": _utc_now(),
                      "heads": observer.heads, "sources": observer.observations,
                      "input_sha256": input_hashes, "input_drift": drift,
                      "contract": "cited-object-content-v1"}, indent=2))
    return 1 if args.check and (drift or not report or any(row["status"] != "fresh" for row in report)) else 0


if __name__ == "__main__":
    sys.exit(main())
