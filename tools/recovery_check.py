#!/usr/bin/env python3
"""Structural checker for the exception-recovery store.

``recovery/`` holds bytes recovered for the accessibility exceptions that the
2026-09-17 audit retained, plus the ledger that says, for every exception, which
routes were attempted and what came back.  This checker enforces the discipline
that makes that store trustworthy:

  1. every stored blob's SHA-256 matches the digest in its own path;
  2. no CANDIDATE is stored in the RECOVERED directory, and nothing in the
     RECOVERED directory is uncorroborated;
  3. every ledger record names at least one attempted route;
  4. no recovered payload appears in ``quarantine/EXCLUSIONS.json``;
  5. the ledger and the store agree: every blob is indexed, every indexed blob
     exists, and every record that claims a stored path points at a real file.

It checks *custody*, not mathematics.  Passing says the bytes are the bytes the
ledger says they are.  It says nothing about whether any recovered object is
correct, reviewed, or eligible for promotion — see ``recovery/README.md``.

Usage:
    python3 tools/recovery_check.py [--root DIR]
Exit status 0 if every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

BLOB = re.compile(r"^(?P<slug>.+)\.(?P<sha>[0-9a-f]{64})\.bin$")
RECOVERED_DIR = os.path.join("recovery", "recovered")
CANDIDATE_DIR = os.path.join("recovery", "candidates")
LEDGER = os.path.join("recovery", "LEDGER.json")
EXCLUSIONS = os.path.join("quarantine", "EXCLUSIONS.json")

VALID_OUTCOMES = {"RECOVERED", "CANDIDATE", "UNRECOVERABLE"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def list_blobs(root: str, rel_dir: str) -> list[str]:
    d = os.path.join(root, rel_dir)
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(rel_dir, n).replace(os.sep, "/")
        for n in os.listdir(d)
        if os.path.isfile(os.path.join(d, n))
    )


def quarantined_digests(root: str) -> set[str]:
    """Every SHA-256 named anywhere in the logical-quarantine exclusion list."""
    path = os.path.join(root, EXCLUSIONS)
    if not os.path.exists(path):
        return set()
    raw = open(path, encoding="utf-8").read()
    return {d.lower() for d in re.findall(r"\b[0-9a-f]{64}\b", raw)}


def check(root: str) -> list[str]:
    fail: list[str] = []
    ledger_path = os.path.join(root, LEDGER)
    if not os.path.exists(ledger_path):
        return [f"{LEDGER}: missing"]
    try:
        ledger = json.load(open(ledger_path, encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report, do not raise
        return [f"{LEDGER}: not valid JSON: {exc}"]

    blobs = list_blobs(root, RECOVERED_DIR) + list_blobs(root, CANDIDATE_DIR)

    # ---- 1. every stored blob's digest matches its path -------------------
    for rel in blobs:
        m = BLOB.match(os.path.basename(rel))
        if not m:
            fail.append(f"{rel}: filename does not carry a <name>.<sha256>.bin digest")
            continue
        actual = sha256_file(os.path.join(root, rel))
        if actual != m.group("sha"):
            fail.append(
                f"{rel}: digest in path is {m.group('sha')[:16]}… but the bytes hash to {actual[:16]}…"
            )

    # ---- 4. nothing recovered may be under logical quarantine ------------
    excluded = quarantined_digests(root)
    for rel in blobs:
        m = BLOB.match(os.path.basename(rel))
        if m and m.group("sha") in excluded:
            fail.append(
                f"{rel}: payload digest {m.group('sha')[:16]}… appears in {EXCLUSIONS}; "
                "a quarantined payload must not be restored into the recovery store"
            )

    # ---- index agreement --------------------------------------------------
    index = {a.get("path"): a for a in ledger.get("artifacts", [])}
    for rel in blobs:
        if rel not in index:
            fail.append(f"{rel}: stored blob is not listed in the ledger's artifacts index")
    for path, art in index.items():
        if not os.path.exists(os.path.join(root, path)):
            fail.append(f"{path}: listed in the ledger's artifacts index but not present on disk")
            continue
        m = BLOB.match(os.path.basename(path))
        if m and art.get("sha256") != m.group("sha"):
            fail.append(f"{path}: ledger records sha256 {art.get('sha256')} but the path says {m.group('sha')}")

    # ---- every blob traces back to an exception ---------------------------
    #
    # The adversarial challenge of 2026-09-18 found three blobs that the
    # artifacts index listed but that no RECORD reached. A record is the
    # ledger's per-exception audit unit: a blob indexed at the top level but
    # traceable to no exception is a file in a directory, not a recovery.
    #
    # Reachability may run through ``stored_path`` (the primary route) or
    # through a record's ``related_artifact``, which is how a payload extracted
    # alongside a different exception is legitimately carried. What is not
    # allowed is no route at all.
    reachable: dict[str, str] = {}
    for rec in ledger.get("records", []):
        rid = rec.get("record_id", "?")
        stored = rec.get("stored_path")
        if stored:
            reachable.setdefault(stored, f"{rid}.stored_path")
        for key in ("related_artifact", "related_artifacts"):
            val = rec.get(key)
            for item in (val if isinstance(val, list) else [val] if val else []):
                p = item.get("path") if isinstance(item, dict) else item
                if isinstance(p, str):
                    reachable.setdefault(p, f"{rid}.{key}")
    for rel in blobs:
        if rel not in reachable:
            fail.append(
                f"{rel}: stored blob is reachable from no ledger record — not via any "
                f"stored_path and not via any related_artifact. Give the exception it "
                f"belongs to a reference, or it is a file in a directory rather than a "
                f"recovery anyone can audit"
            )

    # ---- 2. no CANDIDATE in the RECOVERED directory -----------------------
    for path, art in index.items():
        in_recovered = path.startswith(RECOVERED_DIR.replace(os.sep, "/"))
        corroboration = art.get("corroboration")
        if in_recovered and corroboration != "CORROBORATED":
            fail.append(
                f"{path}: stored under recovered/ with corroboration {corroboration!r}; "
                "only digest-corroborated bytes may live in recovered/"
            )
        if not in_recovered and corroboration == "CORROBORATED":
            fail.append(f"{path}: marked CORROBORATED but stored outside recovered/")

    records = ledger.get("records", [])
    for rec in records:
        rid = rec.get("record_id", "<no id>")
        outcome = rec.get("outcome")
        if outcome not in VALID_OUTCOMES:
            fail.append(f"{rid}: outcome {outcome!r} is not one of {sorted(VALID_OUTCOMES)}")
        stored = rec.get("stored_path")

        # ---- 3. every record names at least one attempted route ----------
        routes = rec.get("attempted_routes") or []
        if not routes:
            fail.append(f"{rid}: no attempted route recorded; a record with no attempt is not a result")
        for i, r in enumerate(routes):
            if not r.get("route") or not r.get("attempt") or not r.get("result"):
                fail.append(f"{rid}: attempted_routes[{i}] must name a route, an attempt and a result")

        if outcome == "CANDIDATE":
            if not stored:
                fail.append(f"{rid}: CANDIDATE with no stored_path")
            elif stored.startswith(RECOVERED_DIR.replace(os.sep, "/")):
                fail.append(
                    f"{rid}: CANDIDATE stored under recovered/ ({stored}); "
                    "a candidate whose digest is not corroborated is not a recovery"
                )
        if outcome == "RECOVERED":
            if not stored:
                fail.append(f"{rid}: RECOVERED with no stored_path")
            elif not stored.startswith(RECOVERED_DIR.replace(os.sep, "/")):
                fail.append(f"{rid}: RECOVERED but stored outside recovered/ ({stored})")
            corr = rec.get("digest_corroboration") or {}
            if corr.get("status") != "CORROBORATED":
                fail.append(
                    f"{rid}: RECOVERED but digest_corroboration.status is {corr.get('status')!r}; "
                    "an uncorroborated recovery is a CANDIDATE"
                )
        if outcome == "UNRECOVERABLE":
            if stored:
                fail.append(f"{rid}: UNRECOVERABLE but names a stored_path ({stored})")
            if not rec.get("exactly_what_is_missing"):
                fail.append(
                    f"{rid}: UNRECOVERABLE without naming exactly what is missing; "
                    "that naming is the whole value of the record"
                )
        if stored and not os.path.exists(os.path.join(root, stored)):
            fail.append(f"{rid}: stored_path {stored} does not exist")

    # independence discipline is data, not prose
    disc = ledger.get("reviewer_disclosure") or {}
    if disc.get("organizational_independence_credit") != 0:
        fail.append("reviewer_disclosure.organizational_independence_credit must be 0 for this session family")
    if not ledger.get("does_not_establish"):
        fail.append("ledger must state what it does NOT establish")
    return fail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args(argv)
    failures = check(args.root)
    if failures:
        print(f"recovery_check: {len(failures)} failure(s)")
        for f in failures:
            print("  FAIL", f)
        return 1
    ledger = json.load(open(os.path.join(args.root, LEDGER), encoding="utf-8"))
    counts = ledger.get("counts", {}).get("by_outcome", {})
    print(
        "recovery_check: OK — {} records ({}), {} stored blobs, independence credit 0".format(
            len(ledger.get("records", [])),
            ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
            len(ledger.get("artifacts", [])),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
