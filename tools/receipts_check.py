#!/usr/bin/env python3
"""Validate every receipt under `engine/receipts/`.

A receipt records that a computation ran (`engine/receipt.py`). It is not
evidence, not a proof, not a certificate and not a status. This checker exists
to keep it that way. It asserts:

  1. **Schema.** Every receipt satisfies `engine.receipt.validate_receipt_object`
     — the same function the writer calls, so the two cannot drift. In
     particular every receipt has a non-empty, non-placeholder
     `does_not_establish`, and every numeric result carries one of the three
     permitted provenances (`certified_interval`, `exact_rational`,
     `float_noncertifying`) with `certifying` false on the float path.

  2. **Append-only.** Each receipt's stored `body_sha256` is recomputed from
     its canonical body; a mismatch means the file was edited after it was
     written. Additionally, every receipt that exists in `git HEAD` must still
     exist, byte for byte, in the working tree. This is the pattern
     `tests/test_registers.py` already enforces for `work_events`.

  3. **No receipt claims a status change.** The verdict fields
     (`engine.receipt.VERDICT_FIELDS`) are scanned for "discharge", "close",
     "promote", "reclassify", "proven", "certified", "solved" and their
     inflections. A receipt whose verdict says DISCHARGED, CLOSED or PROMOTED
     is rejected here. The scan is deliberately confined to those fields:
     `does_not_establish` legitimately contains "does not close Piece 2", and
     an entry point may legitimately be named `radial_gaussian_closed_form`.

  4. **Every receipt's lane exists** in `engine/lanes/`, the receipt sits in
     the directory named for its lane, its filename matches its id, and no id
     is used twice.

Exit status is non-zero on any violation, in the style of
`tools/quarantine_check.py`.

WHAT A PASS DOES NOT MEAN. It does not mean any number in any receipt is
correct, that any bound holds, or that any obligation moved. `OBL-H5-JETMOD`,
`OBL-H5-ZBAND` (hi side), `OBL-H5-REMOTE-THRESHOLD`, `OBL-D1-PROMOTE` and both
Pieces of `D3-LEMMA-RN-UNIF` are OPEN, and a green checker is not a step toward
closing any of them. Only an operator decision under `governance/` changes a
mathematical status.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.receipt import (  # noqa: E402
    STATUS_EFFECT, VERDICT_FIELDS, scan_status_words, validate_receipt_object,
)

DEFAULT_RECEIPTS = os.path.join(ROOT, "engine", "receipts")
DEFAULT_LANES = os.path.join(ROOT, "engine", "lanes")


def known_lane_keys(lanes_dir: str) -> Optional[set]:
    """Lane keys on disk, or ``None`` when there is no lane directory yet.

    ``None`` means "cannot tell": another agent owns ``engine/lanes/`` and may
    not have landed it. The lane-existence check is then skipped with a note
    rather than failing, because a missing directory is not a bad receipt.
    """
    if not os.path.isdir(lanes_dir):
        return None
    keys = set()
    for path in sorted(glob.glob(os.path.join(lanes_dir, "*.json"))):
        key = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, encoding="utf-8") as f:
                key = json.load(f).get("key") or key
        except (OSError, json.JSONDecodeError):
            pass  # mid-write; the filename still names the lane
        keys.add(str(key))
    return keys or None


def head_receipts(root: str, receipts_dir: str) -> Optional[Dict[str, bytes]]:
    """Receipt blobs as of ``git HEAD``, or ``None`` if git cannot answer."""
    rel = os.path.relpath(receipts_dir, root).replace(os.sep, "/")
    try:
        listing = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", rel],
            cwd=root, capture_output=True, text=True, timeout=60)
        if listing.returncode != 0:
            return None
        out: Dict[str, bytes] = {}
        for name in listing.stdout.splitlines():
            name = name.strip()
            if not name.endswith(".json"):
                continue
            blob = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=root,
                                  capture_output=True, timeout=60)
            if blob.returncode == 0:
                out[name] = blob.stdout
        return out
    except (OSError, subprocess.SubprocessError):
        return None


def check(receipts_dir: str = DEFAULT_RECEIPTS,
          lanes_dir: str = DEFAULT_LANES,
          root: str = ROOT,
          check_git: bool = True) -> Tuple[List[str], Dict[str, int]]:
    """Return ``(problems, counts)``. Empty ``problems`` means every check passed."""
    problems: List[str] = []
    counts = {"receipts": 0, "results": 0, "certified_interval": 0,
              "exact_rational": 0, "float_noncertifying": 0, "lanes": 0}

    paths: List[str] = []
    for dirpath, dirnames, filenames in os.walk(receipts_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        paths.extend(os.path.join(dirpath, fn)
                     for fn in filenames if fn.endswith(".json"))
    paths.sort()

    lanes = known_lane_keys(lanes_dir)
    counts["lanes"] = 0 if lanes is None else len(lanes)
    seen_ids: Dict[str, str] = {}
    lanes_seen = set()

    for path in paths:
        rel = os.path.relpath(path, root)
        counts["receipts"] += 1
        try:
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{rel}: unreadable receipt ({exc})")
            continue

        # 1 + 2(a): schema, and the body hash recomputed.
        for msg in validate_receipt_object(obj):
            problems.append(f"{rel}: {msg}")

        if not isinstance(obj, dict):
            continue

        # 3: no receipt claims a status change.
        for fieldname in VERDICT_FIELDS:
            val = obj.get(fieldname)
            if not isinstance(val, str):
                continue
            if fieldname == "status_effect" and val == STATUS_EFFECT:
                continue
            hits = sorted(set(scan_status_words(val)))
            if hits:
                problems.append(
                    f"{rel}: {fieldname} claims a status change {hits}. A "
                    "receipt records a computation; it may not discharge, "
                    "close, promote or reclassify anything.")

        rid = obj.get("receipt_id")
        lane = obj.get("lane")
        if isinstance(rid, str):
            if rid in seen_ids:
                problems.append(f"{rel}: receipt id {rid} is already used by "
                                f"{seen_ids[rid]}")
            else:
                seen_ids[rid] = rel
            if os.path.basename(path) != rid + ".json":
                problems.append(
                    f"{rel}: filename does not match receipt_id {rid!r}")
        if isinstance(lane, str):
            lanes_seen.add(lane)
            parent = os.path.basename(os.path.dirname(path))
            if parent != lane:
                problems.append(
                    f"{rel}: sits in directory {parent!r} but names lane {lane!r}")
            # 4: the lane exists.
            if lanes is not None and lane not in lanes:
                problems.append(
                    f"{rel}: lane {lane!r} has no file in "
                    f"{os.path.relpath(lanes_dir, root)}")

        for r in obj.get("results", []) or []:
            counts["results"] += 1
            prov = r.get("provenance") if isinstance(r, dict) else None
            if prov in counts:
                counts[prov] += 1

    # 2(b): append-only against git HEAD.
    if check_git:
        head = head_receipts(root, receipts_dir)
        if head is not None:
            for name, blob in sorted(head.items()):
                cur = os.path.join(root, name)
                if not os.path.exists(cur):
                    problems.append(
                        f"{name}: present in git HEAD and deleted from the "
                        "working tree (receipts are append-only)")
                    continue
                with open(cur, "rb") as f:
                    if f.read() != blob:
                        problems.append(
                            f"{name}: differs from its git HEAD content "
                            "(receipts are append-only and are never rewritten)")

    return problems, counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--receipts-dir", default=DEFAULT_RECEIPTS)
    ap.add_argument("--lanes-dir", default=DEFAULT_LANES)
    ap.add_argument("--no-git", action="store_true",
                    help="skip the git HEAD append-only comparison")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.receipts_dir):
        print(f"receipts=0 (no {os.path.relpath(args.receipts_dir, ROOT)} "
              f"directory) problems=0")
        return 0

    problems, counts = check(args.receipts_dir, args.lanes_dir, ROOT,
                             check_git=not args.no_git)
    for p in problems:
        print(p)
    note = "" if counts["lanes"] else " lanes=UNAVAILABLE(lane-existence check skipped)"
    print(f"receipts={counts['receipts']} results={counts['results']} "
          f"certified_interval={counts['certified_interval']} "
          f"exact_rational={counts['exact_rational']} "
          f"float_noncertifying={counts['float_noncertifying']}"
          f"{note} problems={len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
