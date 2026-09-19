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
