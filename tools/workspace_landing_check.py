#!/usr/bin/env python3
"""Offline landing-link and archive-identity checks; not scientific verification.

Supports simple inline Markdown links outside fenced code. Does not test remote
URLs, Markdown fragments, reference-style links, or undeclared documents. No writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

LINK = re.compile(r"\[[^\]\n]*\]\(([^\s)]+)\)")
ROOT = Path(__file__).resolve().parents[1]


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def local_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"expected nonempty relative path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or outside-root file: {relative}")
    return path


def inline_targets(text: str):
    fence = None
    for line in text.splitlines():
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            token = marker.group(0)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None:
            yield from LINK.findall(line)


def check(root: Path, manifest: str = "docs/LANDING_MANIFEST.json") -> dict:
    root = root.resolve()
    data = json.loads(local_file(root, manifest).read_text(encoding="utf-8"),
                      object_pairs_hook=unique_pairs)
    if not isinstance(data, dict) or data.get("scope") != "LANDING_LINKS_AND_HISTORICAL_IDENTITY_ONLY":
        raise ValueError("unknown landing manifest scope")
    pages, history = data.get("pages"), data.get("historical_files")
    if not isinstance(pages, list) or not pages or not all(isinstance(p, str) for p in pages):
        raise ValueError("pages must be a nonempty list of paths")
    if len(set(pages)) != len(pages) or not isinstance(history, list):
        raise ValueError("duplicate pages or invalid historical_files")
    failures, links, archive_paths = [], 0, set()
    for page in pages:
        path = local_file(root, page)
        for target in inline_targets(path.read_text(encoding="utf-8")):
            parsed = urlsplit(target)
            if parsed.scheme in ("https", "http", "mailto"):
                continue
            if parsed.scheme or parsed.netloc:
                failures.append(f"{page}: unsupported link {target}")
                continue
            if not parsed.path:
                continue  # Anchor validation is outside this check's scope.
            links += 1
            try:
                relative = str(path.parent.relative_to(root) / unquote(parsed.path))
                local_file(root, relative)
            except ValueError as exc:
                failures.append(f"{page}: {target}: {exc}")
    for item in history:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "git_blob_sha1"}:
            raise ValueError("invalid historical file record")
        if type(item["bytes"]) is not int or item["bytes"] < 0:
            raise ValueError("historical byte count must be a nonnegative integer")
        path = local_file(root, item["path"])
        if path in archive_paths:
            raise ValueError("duplicate historical file")
        archive_paths.add(path)
        payload = path.read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()
        if len(payload) != item["bytes"] or digest != item["git_blob_sha1"]:
            failures.append(f"{item['path']}: historical byte identity mismatch")
    return {"scope": data["scope"], "pages": len(pages), "local_links": links,
            "historical_files": len(history), "problems": failures}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", default="docs/LANDING_MANIFEST.json")
    args = parser.parse_args(argv)
    try:
        result = check(args.root, args.manifest)
    except (ValueError, OSError, TypeError) as exc:
        print(f"landing_check: INVALID_INPUT: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    print("Scope: landing links and historical bytes only; research CI and mathematical correctness not checked.")
    return 1 if result["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
