#!/usr/bin/env python3
"""Verify every content manifest in the repository.

Two manifest formats are recognised, both inherited from the Drive program:

1. ``MANIFEST.sha256`` / ``*.sha256`` files in ``sha256sum`` format
   (``<hex>  <relative path>`` per line), verified relative to the manifest's
   directory.
2. ``_MANIFEST.jsonl`` files (one JSON object per line with ``dest``, ``bytes``
   and ``sha256``) written by the Drive mirror; ``dest`` is resolved relative to
   the manifest's directory when it is not absolute.

Exit status is non-zero if any listed file is missing or has a different digest
or size.  Files listed with ``note`` starting with ``skipped`` or ``failed`` or
``tree-only`` are reported but do not fail the run.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sha256sum(manifest: str) -> tuple[int, int, list[str]]:
    base = os.path.dirname(manifest)
    ok = bad = 0
    problems = []
    for line in open(manifest, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            continue
        digest, rel = parts
        rel = rel.lstrip("*")
        path = os.path.normpath(os.path.join(base, rel))
        if not os.path.exists(path):
            bad += 1
            problems.append(f"MISSING {path}")
            continue
        if sha256(path) != digest.lower():
            bad += 1
            problems.append(f"SHA MISMATCH {path}")
        else:
            ok += 1
    return ok, bad, problems


def is_blank(path: str) -> bool:
    """True when a file holds no text: empty, whitespace only, or a lone BOM.

    bytes.strip() removes ASCII whitespace but leaves b"\xef\xbb\xbf"
    standing, so a BOM-only export is three bytes long and is not blank by that
    test alone.  Strip the BOM first.
    """
    with open(path, "rb") as handle:
        return not handle.read().lstrip(b"\xef\xbb\xbf").strip()


_EMPTY_BODY_IDS: dict = {}


def find_inventory(start: str) -> str | None:
    """The drive/inventory.jsonl governing a manifest, found by walking up.

    Deliberately NOT resolved against the module-level ROOT: that binds the
    real repository's inventory at import time, so a control running the
    checker against a synthetic root would silently corroborate its blank files
    against this repository's eight EMPTY_NATIVE_BODY ids and pass.  That is
    the defect CLAUDE.md records in tools/claims_check.py, one directory over.
    """
    here = os.path.abspath(start)
    while True:
        candidate = os.path.join(here, "drive", "inventory.jsonl")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


def empty_native_body_ids(start: str) -> set:
    """Drive ids the inventory itself records as having an empty native body.

    Read from drive/inventory.jsonl, which is exported source data this
    repository never edits, so a row cannot talk its own file into being
    acceptable -- the corroboration comes from outside the manifest.  No
    inventory in the tree means nothing corroborates anything, which is an
    empty set, not a pass.
    """
    inventory = find_inventory(start)
    if inventory is None:
        return set()
    if inventory not in _EMPTY_BODY_IDS:
        out = set()
        with open(inventory, encoding="utf-8") as handle:
            for line in handle:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("access_status") == "EMPTY_NATIVE_BODY" and rec.get("id"):
                    out.add(rec["id"])
        _EMPTY_BODY_IDS[inventory] = out
    return _EMPTY_BODY_IDS[inventory]


def check_jsonl(manifest: str) -> tuple[int, int, list[str]]:
    base = os.path.dirname(manifest)
    ok = bad = 0
    problems = []
    for line in open(manifest, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            problems.append(f"BAD JSON LINE in {manifest}: {line[:80]}")
            continue
        note = str(row.get("note", ""))
        if note.startswith(("skipped", "failed", "tree-only")):
            continue
        dest = row.get("dest")
        if not dest:
            continue
        path = dest if os.path.isabs(dest) else os.path.normpath(os.path.join(base, dest))
        if not os.path.exists(path):
            # mirrors may have been relocated into the repo: try by basename under base
            alt = os.path.join(base, os.path.basename(dest))
            if os.path.exists(alt):
                path = alt
            else:
                bad += 1
                problems.append(f"MISSING {path}")
                continue
        if "sha256" in row and sha256(path) != str(row["sha256"]).lower():
            bad += 1
            problems.append(f"SHA MISMATCH {path}")
            continue
        if "bytes" in row and os.path.getsize(path) != int(row["bytes"]):
            bad += 1
            problems.append(f"SIZE MISMATCH {path}")
            continue
        if row.get("exact") is False and row.get("stored") and is_blank(path):
            # A reading copy is a rendering of a document, so a blank one is
            # either a real property of the source or a failed fetch stored as
            # though it were a rendering.  The two are told apart from outside
            # the manifest: the inventory marks eight objects EMPTY_NATIVE_BODY,
            # and a blank export of one of those is the export the inventory
            # predicts.  A blank export of anything else is not corroborated by
            # anything and must not be carried as a stored reading copy.
            if row.get("id") not in empty_native_body_ids(base):
                bad += 1
                problems.append(f"UNCORROBORATED BLANK READING COPY {path}")
                continue
        ok += 1
    return ok, bad, problems


def main(argv: list[str]) -> int:
    roots = argv[1:] or [ROOT]
    total_ok = total_bad = 0
    all_problems = []
    manifests = 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", "__pycache__")]
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                if fn.endswith(".sha256") or fn == "MANIFEST.sha256":
                    ok, bad, pr = check_sha256sum(p)
                elif fn.endswith("_MANIFEST.jsonl") or fn == "MANIFEST.jsonl":
                    ok, bad, pr = check_jsonl(p)
                else:
                    continue
                manifests += 1
                total_ok += ok
                total_bad += bad
                all_problems += pr
    for pr in all_problems:
        print(pr)
    print(f"manifests={manifests} verified={total_ok} problems={total_bad}")
    return 1 if total_bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
