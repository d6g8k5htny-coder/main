#!/usr/bin/env python3
"""Verify the content-addressed carrier binding.

`engine/carriers/MANIFEST.json` records Drive carriers that were fetched,
hash-checked against `drive/inventory.jsonl` and (mostly) stored under
`engine/carriers/blobs/`. Binding records bytes and provenance. It is **not**
review, **not** replay and **not** endorsement, and this tool does not run any
carrier.

This tool checks that:

  1. every record carries the required fields, well-typed, with values drawn
     from the declared vocabularies;
  2. every record's title, path, byte count and SHA-256 still agree with
     `drive/inventory.jsonl` (the exported source of truth);
  3. every stored blob re-hashes to the recorded digest, has the recorded byte
     count, and lives at its content address `<sha256[:16]>__<name>`;
  4. no payload excluded by `quarantine/EXCLUSIONS.json` — by digest or by
     Drive object id — is stored, and any such record is flagged not-stored;
  5. no bound carrier comes from `02_LEGACY_Q0_ARCHIVE` (zero authority);
  6. every `lane` names a real section of `docs/OPEN_PROBLEMS.md`, or the
     explicit sentinel `NOT_IN_OPEN_PROBLEMS` for carriers no section covers;
  7. no record claims `certifying: true` on anything but exact rational
     arithmetic — high precision, intervals-in-floats, Monte Carlo and fitted
     exponents are not certified bounds;
  8. `blobs/` contains no file that no record references.

Exit status is non-zero on any violation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "engine", "carriers", "MANIFEST.json")
BLOBS = os.path.join(ROOT, "engine", "carriers", "blobs")
INVENTORY = os.path.join(ROOT, "drive", "inventory.jsonl")
EXCLUSIONS = os.path.join(ROOT, "quarantine", "EXCLUSIONS.json")
OPEN_PROBLEMS = os.path.join(ROOT, "docs", "OPEN_PROBLEMS.md")

REQUIRED = {
    "carrier_id": str,
    "title": str,
    "drive_id": str,
    "drive_path": str,
    "bytes": int,
    "sha256": str,
    "blob_stored": bool,
    "lane": str,
    "computes": str,
    "arithmetic": str,
    "certifying": bool,
    "certifying_note": str,
    "authority_tier": str,
    "source_status": str,
}
ARITHMETIC = {"exact_rational", "mpmath_float", "float", "mixed", "unknown"}
TIERS = {"active", "superseded", "legacy", "quarantined"}
NOT_STORED_REASONS = {
    "SIZE",                      # too large to carry in the repository
    "QUARANTINE_EXCLUSION",      # named by quarantine/EXCLUSIONS.json
    "QUARANTINE_PATH",           # Drive path/title declares it quarantined
    "HASH_MISMATCH",             # digest did not match the inventory
    "RECONSTRUCTION_INCOMPLETE",  # byte-exact local copy was not achieved
}
UNMAPPED = "NOT_IN_OPEN_PROBLEMS"
SECTION_RE = re.compile(r"^#{2,3}\s+([A-Z][0-9]*)\.\s")


def open_problem_sections() -> set[str]:
    """Section keys of docs/OPEN_PROBLEMS.md, e.g. {'A', 'A1', ..., 'F'}."""
    keys = set()
    with open(OPEN_PROBLEMS, encoding="utf-8") as f:
        for line in f:
            m = SECTION_RE.match(line)
            if m:
                keys.add(m.group(1))
    return keys


def inventory() -> dict[str, dict]:
    rows = {}
    with open(INVENTORY, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def excluded() -> tuple[dict[str, str], dict[str, str]]:
    """(payload digest -> exclusion key, Drive object id -> exclusion key)."""
    with open(EXCLUSIONS, encoding="utf-8") as f:
        ex = json.load(f)["exclusions"]
    digests, objects = {}, {}
    for e in ex:
        d = (e.get("payload_sha256") or "").lower()
        if d:
            digests[d] = e["key"]
        if e.get("kind") == "drive_object" and e.get("carrier_id"):
            objects[e["carrier_id"]] = e["key"]
    return digests, objects


def main() -> int:
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    carriers = manifest.get("carriers", [])
    sections = open_problem_sections()
    inv = inventory()
    ex_digests, ex_objects = excluded()

    problems: list[str] = []
    seen_ids: set[str] = set()
    referenced: set[str] = set()
    stored = 0

    for rec in carriers:
        cid = rec.get("carrier_id", "<no carrier_id>")

        for field, typ in REQUIRED.items():
            if field not in rec:
                problems.append(f"{cid}: missing required field {field!r}")
            elif not isinstance(rec[field], typ) or (
                    typ is int and isinstance(rec[field], bool)):
                problems.append(
                    f"{cid}: field {field!r} is {type(rec[field]).__name__}, "
                    f"expected {typ.__name__}")
        if set(REQUIRED) - set(rec):
            continue

        if cid in seen_ids:
            problems.append(f"{cid}: duplicate carrier_id")
        seen_ids.add(cid)

        if rec["arithmetic"] not in ARITHMETIC:
            problems.append(f"{cid}: arithmetic {rec['arithmetic']!r} not in vocabulary")
        if rec["authority_tier"] not in TIERS:
            problems.append(f"{cid}: authority_tier {rec['authority_tier']!r} not in vocabulary")
        if not rec["certifying_note"].strip():
            problems.append(f"{cid}: certifying_note is empty")
        if not rec["computes"].strip():
            problems.append(f"{cid}: computes is empty")

        # A float, mpmath, mixed or unknown carrier is never a certified bound.
        if rec["certifying"] and rec["arithmetic"] != "exact_rational":
            problems.append(
                f"{cid}: certifying true with arithmetic {rec['arithmetic']!r}; "
                "high precision is not a certified enclosure")

        if rec["lane"] != UNMAPPED and rec["lane"] not in sections:
            problems.append(
                f"{cid}: lane {rec['lane']!r} is not a section of docs/OPEN_PROBLEMS.md "
                f"(and not the sentinel {UNMAPPED})")

        row = inv.get(rec["drive_id"])
        if row is None:
            problems.append(f"{cid}: drive_id not present in drive/inventory.jsonl")
        else:
            if row.get("title") != rec["title"]:
                problems.append(f"{cid}: title differs from the inventory")
            if row.get("path") != rec["drive_path"]:
                problems.append(f"{cid}: drive_path differs from the inventory")
            if row.get("bytes") != rec["bytes"]:
                problems.append(f"{cid}: bytes {rec['bytes']} != inventory {row.get('bytes')}")
            if (row.get("sha256") or "").lower() != rec["sha256"].lower():
                problems.append(f"{cid}: sha256 differs from the inventory")
            if row.get("access_status") != rec["source_status"]:
                problems.append(f"{cid}: source_status is not the inventory access_status")
            if str(row.get("path", "")).startswith("02_LEGACY_Q0_ARCHIVE"):
                problems.append(f"{cid}: bound from 02_LEGACY_Q0_ARCHIVE, which has zero authority")

        digest = rec["sha256"].lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            problems.append(f"{cid}: sha256 is not a 64-character hex digest")

        key = ex_digests.get(digest) or ex_objects.get(rec["drive_id"])
        if key and rec["blob_stored"]:
            problems.append(
                f"{cid}: payload is excluded by {key} but its blob is stored")
        if key and rec.get("authority_tier") == "active":
            problems.append(
                f"{cid}: payload is excluded by {key} but authority_tier is active")

        if rec["blob_stored"]:
            stored += 1
            blob = rec.get("blob_path")
            if not isinstance(blob, str) or not blob:
                problems.append(f"{cid}: blob_stored true but blob_path is not a path")
                continue
            referenced.add(os.path.basename(blob))
            path = os.path.join(ROOT, blob)
            if not os.path.isfile(path):
                problems.append(f"{cid}: blob {blob} does not exist")
                continue
            with open(path, "rb") as handle:
                raw = handle.read()
            got = hashlib.sha256(raw).hexdigest()
            if got != digest:
                problems.append(f"{cid}: blob {blob} hashes to {got[:16]}, manifest says {digest[:16]}")
            if len(raw) != rec["bytes"]:
                problems.append(f"{cid}: blob {blob} is {len(raw)} bytes, manifest says {rec['bytes']}")
            if not os.path.basename(blob).startswith(digest[:16] + "__"):
                problems.append(f"{cid}: blob {blob} is not at its content address")
        else:
            if rec.get("blob_path") is not None:
                problems.append(f"{cid}: blob_stored false but blob_path is set")
            reason = rec.get("not_stored_reason")
            if reason not in NOT_STORED_REASONS:
                problems.append(
                    f"{cid}: not_stored_reason {reason!r} missing or not in vocabulary")

    if os.path.isdir(BLOBS):
        for fn in sorted(os.listdir(BLOBS)):
            # Bytecode caches appear when blobs are imported; they are not carriers.
            path = os.path.join(BLOBS, fn)
            if fn == "__pycache__" or fn.endswith(".pyc") or not os.path.isfile(path):
                continue
            if fn not in referenced:
                problems.append(f"blobs/{fn}: stored but no manifest record references it")

    if manifest.get("carriers_bound") != len(carriers):
        problems.append("header carriers_bound does not match the number of records")
    if manifest.get("blobs_stored") != stored:
        problems.append("header blobs_stored does not match the stored records")

    for p in problems:
        print(p)
    print(f"carriers={len(carriers)} blobs_stored={stored} "
          f"certifying={sum(1 for r in carriers if r.get('certifying'))} "
          f"lanes={len(sections)} problems={len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
