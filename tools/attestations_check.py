#!/usr/bin/env python3
"""Check architectural admission attestations against their schema and rules.

An attestation records that a named object cleared a named, machine-checkable
ARCHITECTURAL predicate at a stated commit: the repository's own structural
gates ran and accepted it. That is all it records.

What this does NOT establish. An attestation is not a review, not a grade, not
a promotion and not an independence credit. Architectural admission says
nothing about whether any mathematics is correct: a wrong proof that is
correctly filed, correctly hashed and correctly quarantined is ADMITTED, and
that is the intended behaviour, because the predicate is about structure and
not about truth. Nothing here discharges a premise, moves a gate, or changes
any claim's grade -- `claim_grade_after` and `gate_status_after` are both
pinned to UNCHANGED and this checker refuses any other value. Every ADMITTED
record must also name, in `awaiting`, what the object is still waiting on,
because admission never finishes anything. This form is NOT DEPLOYED as an
enforcement mechanism and no scheduler consumes these records.

Fail-closed. A single non-zero exit code among the recorded checks forbids
ADMITTED. A record that claims ADMITTED while carrying a failed check is a
violation, not a judgement call.

Run:  python3 tools/attestations_check.py [--records DIR] [--schema PATH]
Every path is resolved when ``main()`` runs, never at import time.
"""
from __future__ import annotations

import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "tools"))

# The JSON Schema subset evaluator and the free-text scanners already exist for
# review records. They are reused rather than reimplemented so that a rule
# tightened in one place cannot silently stay loose in the other.
from reviews_check import (  # noqa: E402
    CONFIDENCE_PATTERNS,
    PROMOTION_PHRASES,
    WORD,
    load_json,
    normalize,
    validate_instance,
    walk_strings,
)

SCHEMA_REL = os.path.join("attestations", "attestation.schema.json")
RECORDS_REL = os.path.join("attestations", "records")

MIN_DISTINCT_TOKENS = {"does_not_establish": 25}


def check_record(record: dict, schema: dict, stem: str,
                 seen_ids: set[str], seen_prose: dict[str, str],
                 seen_successors: set[tuple[str, str]]) -> list[str]:
    problems: list[str] = []

    def fail(message: str) -> None:
        problems.append(f"{stem}.json: {message}")

    problems.extend(f"{stem}.json: {p}" for p in validate_instance(record, schema))
    if problems:
        return problems

    attestation_id = record["attestation_id"]
    if attestation_id != stem:
        fail(f"attestation_id {attestation_id!r} does not match the filename stem {stem!r}")
    if attestation_id in seen_ids:
        fail(f"duplicate attestation_id {attestation_id!r}")
    seen_ids.add(attestation_id)

    # -- fail-closed admission ------------------------------------------- #
    failed = [c for c in record["checks"] if c["exit_code"] != 0]
    if record["outcome"] == "ADMITTED" and failed:
        commands = ", ".join(repr(c["command"]) for c in failed)
        fail(f"outcome ADMITTED but {len(failed)} recorded check(s) exited non-zero "
             f"({commands}); a failed structural gate forbids admission")
    if record["outcome"] == "REFUSED" and not failed:
        fail("outcome REFUSED but every recorded check exited zero; a refusal must "
             "name the check that refused it")

    # -- the three pins that keep an attestation from becoming a promotion - #
    if record["claim_grade_after"] != "UNCHANGED":
        fail(f"claim_grade_after is {record['claim_grade_after']!r}; an attestation "
             f"never changes a grade")
    if record["gate_status_after"] != "UNCHANGED":
        fail(f"gate_status_after is {record['gate_status_after']!r}; an attestation "
             f"never moves a gate")
    if record["independence_credit"] != 0:
        fail(f"independence_credit is {record['independence_credit']!r}; an "
             f"attestation is not a review and carries no independence")
    if record["outcome"] == "ADMITTED" and not record["awaiting"]:
        fail("outcome ADMITTED with an empty awaiting list; admission finishes nothing")

    # -- anti-boilerplate -------------------------------------------------- #
    for field, floor in MIN_DISTINCT_TOKENS.items():
        text = record.get(field, "")
        distinct = {w.lower() for w in WORD.findall(text)}
        if len(distinct) < floor:
            fail(f"{field} has {len(distinct)} distinct token(s), below the floor of "
                 f"{floor}; padding by repetition is not a disclosure")
        key = normalize(text)
        if key and key in seen_prose and seen_prose[key] != stem:
            fail(f"{field} is verbatim identical to {seen_prose[key]}.json; a "
                 f"statement copied between records is not specific to this object")
        elif key:
            seen_prose.setdefault(key, stem)

    # -- the object's current bytes ---------------------------------------- #
    # An attestation speaks for the commit it names. When the object still
    # exists at that path and its bytes have since changed, the record is
    # historical and must say so: a stale digest presented as current is the
    # one failure this form cannot tolerate.
    path = record["object_id"].split(" @ ")[0].strip()
    candidate = os.path.join(ROOT, path)
    if os.path.isfile(candidate):
        with open(candidate, "rb") as handle:
            raw = handle.read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record["object_sha256"] or len(raw) != record["object_bytes"]:
            successor = record.get("superseded_by")
            if not successor:
                fail(f"object {path} now has {len(raw)} bytes / sha256 {digest[:16]}..., "
                     f"not the attested {record['object_bytes']} / "
                     f"{record['object_sha256'][:16]}...; re-run the gates and attest the "
                     f"current bytes, or set superseded_by to the record that did")
            else:
                seen_successors.add((stem, successor))

    # -- free-text scans --------------------------------------------------- #
    for path, text in walk_strings(record):
        lowered = text.lower()
        for phrase in PROMOTION_PHRASES:
            if phrase in lowered:
                fail(f"promotion language {phrase!r} in field {path}: an attestation "
                     f"records admission, never a status movement")
        for pattern in CONFIDENCE_PATTERNS:
            hit = pattern.search(text)
            if hit:
                fail(f"confidence voting {hit.group(0)!r} in field {path}")

    return problems


def check_all(records_dir: str, schema_path: str) -> list[str]:
    if not os.path.isdir(records_dir):
        return [f"records directory {records_dir} does not exist"]
    schema = load_json(schema_path)
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_prose: dict[str, str] = {}
    seen_successors: set[tuple[str, str]] = set()
    for filename in sorted(os.listdir(records_dir)):
        if not filename.endswith(".json"):
            continue
        stem = filename[: -len(".json")]
        try:
            record = load_json(os.path.join(records_dir, filename))
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            problems.append(f"{filename}: unreadable ({exc})")
            continue
        if not isinstance(record, dict):
            problems.append(f"{filename}: top level is not an object")
            continue
        problems.extend(check_record(record, schema, stem, seen_ids, seen_prose,
                                     seen_successors))
    for stem, successor in sorted(seen_successors):
        if successor not in seen_ids:
            problems.append(f"{stem}.json: superseded_by names {successor!r}, which is not a "
                            f"record in this directory; a record cannot be retired by a "
                            f"successor that does not exist")
    return problems


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    records_dir = os.path.join(ROOT, RECORDS_REL)
    schema_path = os.path.join(ROOT, SCHEMA_REL)
    while argv:
        flag = argv.pop(0)
        if flag == "--records" and argv:
            records_dir = argv.pop(0)
        elif flag == "--schema" and argv:
            schema_path = argv.pop(0)
        else:
            print(f"usage: attestations_check.py [--records DIR] [--schema PATH]  "
                  f"(unrecognized: {flag!r})", file=sys.stderr)
            return 2

    problems = check_all(records_dir, schema_path)
    count = (len([f for f in os.listdir(records_dir) if f.endswith(".json")])
             if os.path.isdir(records_dir) else 0)
    for problem in problems:
        print(f"PROBLEM  {problem}")
    shown = os.path.relpath(records_dir, ROOT) if records_dir.startswith(ROOT) else records_dir
    print(f"attestations_check: {count} attestation(s) in {shown}, {len(problems)} problem(s).")
    if not problems:
        print("attestations_check: structure only. No mathematics verified, no grade "
              "changed, no gate moved, no independence credit awarded.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
