#!/usr/bin/env python3
"""Enforce the logical quarantine.

`quarantine/EXCLUSIONS.json` lists objects and archive members excluded from
active consumption (OP-PROT-019 §6). Bytes stay intact; what is excluded is the
*certification claim* that rests on them.

This tool checks that:

  1. every exclusion in the register (`registers/json/quarantine_index.json`)
     appears in `EXCLUSIONS.json` with the same class, and vice versa;
  2. every archive-member exclusion resolves to a real member of a real carrier
     in `drive/source_map/Archive_Members.csv`, with a matching payload digest;
  3. no excluded payload digest appears in any repository manifest — i.e. no
     excluded artifact has been silently pulled into the verified content set;
  4. every exclusion carries a restoration test.

Exit status is non-zero on any violation.
"""
from __future__ import annotations

import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUSIONS = os.path.join(ROOT, "quarantine", "EXCLUSIONS.json")
REGISTER = os.path.join(ROOT, "registers", "json", "quarantine_index.json")
ARCHIVES = os.path.join(ROOT, "drive", "source_map", "Archive_Members.csv")


def manifest_digests() -> dict[str, str]:
    """Every SHA-256 mentioned by any manifest in the repository."""
    found: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".pytest_cache")]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if fn.endswith("_MANIFEST.jsonl") or fn == "MANIFEST.jsonl":
                for line in open(p, encoding="utf-8", errors="replace"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("sha256"):
                        found[str(row["sha256"]).lower()] = p
            elif fn.endswith(".sha256") or fn == "MANIFEST.sha256":
                for line in open(p, encoding="utf-8", errors="replace"):
                    parts = line.split(None, 1)
                    if len(parts) == 2 and len(parts[0]) == 64:
                        found[parts[0].lower()] = p
    return found


def main() -> int:
    with open(EXCLUSIONS, encoding="utf-8") as f:
        ex = json.load(f)["exclusions"]
    with open(REGISTER, encoding="utf-8") as f:
        reg = json.load(f)
    h = reg["header"]
    ki, ci = h.index("Quarantine key"), h.index("Class")
    reg_rows = {r[ki]: r[ci] for r in reg["rows"] if r and r[ki]}

    problems: list[str] = []

    by_key = {e["key"]: e for e in ex}
    for key, cls in reg_rows.items():
        if key not in by_key:
            problems.append(f"register exclusion {key} missing from EXCLUSIONS.json")
        elif by_key[key]["class"] != cls:
            problems.append(
                f"{key}: class {by_key[key]['class']!r} != register {cls!r}")
    for key in by_key:
        if key not in reg_rows:
            problems.append(f"EXCLUSIONS.json has {key}, which the register does not list")

    members: dict[tuple[str, str], dict] = {}
    with open(ARCHIVES, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            members[(row["Carrier ID"], row["Member path"])] = row

    for e in ex:
        if not e.get("restoration_test"):
            problems.append(f"{e['key']}: no restoration test recorded")
        if e.get("kind") != "archive_member":
            continue
        m = members.get((e["carrier_id"], e["member_path"]))
        if not m:
            problems.append(
                f"{e['key']}: member {e['member_path']!r} not found in carrier {e['carrier_id']}")
            continue
        if e.get("payload_sha256") and e["payload_sha256"] != m["Payload SHA-256"]:
            problems.append(f"{e['key']}: payload digest does not match the archive index")

    digests = manifest_digests()
    for e in ex:
        d = (e.get("payload_sha256") or "").lower()
        if d and d in digests:
            problems.append(
                f"{e['key']}: excluded payload {d[:16]} is consumed by manifest {digests[d]}")

    for p in problems:
        print(p)
    archive_members = sum(1 for e in ex if e.get("kind") == "archive_member")
    print(f"exclusions={len(ex)} archive_members={archive_members} "
          f"manifest_digests={len(digests)} problems={len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
