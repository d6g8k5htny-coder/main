#!/usr/bin/env python3
"""Enforce the logical quarantine.

`quarantine/EXCLUSIONS.json` lists objects and archive members excluded from
active consumption (OP-PROT-019 §6). Bytes stay intact; what is excluded is the
*certification claim* that rests on them, at the named scope.

This tool checks that:

  1. every exclusion in the register (`registers/json/quarantine_index.json`)
     appears in `EXCLUSIONS.json` with the same class, and vice versa;
  2. every archive-member exclusion resolves to a real member of a real carrier
     in `drive/source_map/Archive_Members.csv`, with a matching payload digest;
  3. no excluded payload digest appears in any repository manifest — i.e. no
     excluded artifact has been silently pulled into the verified content set.
     Only a record carrying a `payload_sha256` can be compared; the summary
     reports `digest_comparable` and `digest_not_compared` separately, every
     record in the second group must say why it is there, and a run in which
     nothing was compared is refused;
  4. every exclusion carries a restoration test;
  5. every member this repository binds byte-exact — a record in
     `engine/rn_engine/BINDING.json` or `engine/carriers/MANIFEST.json` — whose
     payload digest or Drive object id an exclusion names carries that
     exclusion's key, class and scope under `quarantine_exclusions`. OP-PROT-019
     §6 lets the bytes stay bound ("their bytes and old manifests remain
     intact"); what may not happen is that a bound member is consumed as if
     unqualified. The binding record must say, in its own fields, that the
     member is logically quarantined at the named scope. A missing, incomplete
     or mistyped annotation fails the run; an annotation naming an exclusion
     that does not name the record fails too, so a stale annotation cannot
     stand in for a real one.

For an inventable agent this checker and `tools/vault_hygiene_check.py` are
engineering hygiene. Quarantine is not a source of truth. `Q-R17-VAULT` is
`digest_not_compared` because the exclusion names a folder, not because a
vault id was accepted as inventable authority. `vault_rows` refuses stored
bytes only when `drive_path` contains `99_DO_NOT_OPEN`. A stored vault id
under another path string is the hygiene check's refusal. A pass here does
not discharge OBL-H5-JETMOD and does not set `inventable_attempt_accepted`.
Neither checker scans `docs/math_status_probes/`. See that directory's README
and `quarantine/PATHS.md`.

Exit status is non-zero on any violation.

Run:  python3 tools/quarantine_check.py [--exclusions PATH] [--register PATH]
          [--archives PATH] [--binding PATH] [--manifest PATH] [--scan-root DIR]
The flags exist so tests can point the checker at mutated copies; the defaults
are the committed files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUSIONS = os.path.join(ROOT, "quarantine", "EXCLUSIONS.json")
REGISTER = os.path.join(ROOT, "registers", "json", "quarantine_index.json")
ARCHIVES = os.path.join(ROOT, "drive", "source_map", "Archive_Members.csv")
BINDING = os.path.join(ROOT, "engine", "rn_engine", "BINDING.json")
MANIFEST = os.path.join(ROOT, "engine", "carriers", "MANIFEST.json")

ANNOTATION_FIELD = "quarantine_exclusions"

REASON_FIELD = "payload_digest_not_compared"

# Invariant 3 can only compare a record that carries a `payload_sha256`.  Six of
# the twenty-two do not, and until this table existed the loop skipped them in
# silence while the summary printed `exclusions=22` -- a reader auditing
# quarantine coverage from that line counted twenty-two comparisons where
# sixteen had been made.
#
# The reason belongs with the record, and for a new exclusion that is where it
# goes: a `payload_digest_not_compared` field in EXCLUSIONS.json satisfies this
# check too.  These six sit here instead because two certificates pin that
# file's bytes as source identity --
# `research/rn/candidates/inner_wedge_20260920_v1.json`, at
# `/majorant/source_binding/authenticated_identities/quarantine/EXCLUSIONS.json`
# and `/source_identities/quarantine/EXCLUSIONS.json`, both 19,555 bytes /
# `8a5a89012dcd0fece1b3ea882ea2551f8952d333d22a847e06bd7935c255d1ed`.  Adding a
# field there changes those bytes and fails four replay checkers closed.  The
# label is not worth breaking a pin for, so it lives in the checker.
#
# Widening invariant 3 to these six instead would be wrong, not merely
# inconvenient: see `Q-R17-DUP-001` below.
DIGEST_NOT_COMPARED = {
    "Q-R17-DUP-001":
        "EXACT_DUPLICATE. A digest does exist, in the free-text `identity` "
        "field, and promoting it into `payload_sha256` would fail this check "
        "on a correct tree: an exact duplicate shares its bytes with a "
        "RETAINED KEEPER by definition, so invariant 3 would fire on the "
        "keeper. Verified 2026-09-24: the excluded surplus copy "
        "(1Y_3zFonLsFIAHP5KSkUfsJXqHZXIUAL2) is tree-only DO_NOT_PORT in "
        "drive/mirrors/90_QUARANTINE_AND_TRIAGE/_MANIFEST.jsonl -- stored "
        "false, sha256 null -- and the one stored row carrying those bytes is "
        "the keeper, a different Drive object "
        "(1Hc8dJvdh504xKBBHXswU5Ly-_8uYp_Sv) under the 2026-09-15 KIMI FINAL "
        "INTAKE lane. Nothing excluded has leaked; "
        "tests/test_quarantine_digest_coverage.py reads that off the "
        "manifests rather than asserting it here.",
    "Q-R17-TEMP-001":
        "UNVERIFIED. The object is a native Google document transport, whose "
        "`identity` is a document id rather than a payload digest. The corpus "
        "declares no payload digest for a native Doc, so there is nothing for "
        "invariant 3 to compare.",
    "Q-R17-RN-OLD":
        "SUPERSEDED. `identity` gives a byte count (7,201) and no digest. The "
        "verdict is about content -- 'different content, not duplicate' -- and "
        "is enforced by the record, not by a digest match.",
    "Q-R17-LOCAL-TB":
        "EXISTING_CONTAINER. The exclusion names a FOLDER (`identity` is a "
        "folder id). A folder has no payload, so invariant 3 cannot compare "
        "one. The children are explicitly not re-reviewed in this operations "
        "pass, so no per-child digest is asserted here either.",
    "Q-R17-LOCAL-P01":
        "EXISTING_CONTAINER. The exclusion names a FOLDER (`identity` is a "
        "folder id). A folder has no payload, so invariant 3 cannot compare "
        "one. The children are explicitly not re-reviewed in this operations "
        "pass, so no per-child digest is asserted here either.",
    "Q-R17-VAULT":
        "EXISTING_CONTAINER. The exclusion names a FOLDER -- the 99_DO_NOT_OPEN "
        "vault, which is metadata only and is never opened. A folder has no "
        "payload for invariant 3 to compare, and the vault is separately "
        "guarded by the vault_rows check below, which refuses any manifest row "
        "storing bytes from it.",
}


def uncompared_reason(e: dict) -> str:
    """Why invariant 3 cannot compare this record, from the record or the table.

    Empty string when neither says, which is the case the checker refuses.
    """
    return (str(e.get(REASON_FIELD) or "").strip()
            or DIGEST_NOT_COMPARED.get(e.get("key"), "").strip())


def manifest_digests(root: str) -> dict[str, str]:
    """Every SHA-256 mentioned by any manifest in the repository."""
    found: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
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


def bound_records(path: str, index_name: str) -> list[tuple[str, dict]]:
    """(index label, record) for every carrier record in a binding index.

    A missing index binds nothing. An index whose shape is not recognised is an
    error: reading it as empty would make invariant 5 pass vacuously.
    """
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    entries = doc.get("carriers") if isinstance(doc, dict) else doc
    if not isinstance(entries, list):
        raise ValueError(f"{index_name}: no 'carriers' list; cannot check bound members")
    out = []
    for rec in entries:
        if not isinstance(rec, dict) or not rec.get("carrier_id"):
            raise ValueError(f"{index_name}: carrier record without a carrier_id")
        out.append((index_name, rec))
    return out


def exclusions_naming(rec: dict, ex: list[dict]) -> list[dict]:
    """The exclusions that name this record: by payload digest, or by Drive
    object id for a drive_object exclusion."""
    digest = str(rec.get("sha256") or "").lower()
    drive_id = rec.get("drive_id")
    hits = []
    for e in ex:
        d = str(e.get("payload_sha256") or "").lower()
        if digest and d and d == digest:
            hits.append(e)
        elif drive_id and e.get("kind") == "drive_object" and e.get("carrier_id") == drive_id:
            hits.append(e)
    return hits


def check_bound_member_annotations(ex: list[dict], binding: str, manifest: str) -> list[str]:
    """Invariant 5. Returns problem strings; records nothing, unbinds nothing."""
    problems: list[str] = []
    by_key = {e["key"]: e for e in ex}
    records: list[tuple[str, dict]] = []
    for path, label in ((binding, "engine/rn_engine/BINDING.json"),
                        (manifest, "engine/carriers/MANIFEST.json")):
        try:
            records += bound_records(path, label)
        except ValueError as exc:
            problems.append(str(exc))
    for label, rec in records:
        cid = rec["carrier_id"]
        hits = exclusions_naming(rec, ex)
        ann = rec.get(ANNOTATION_FIELD)
        if hits and not isinstance(ann, list):
            for e in hits:
                problems.append(
                    f"{label} {cid}: bound member is named by exclusion {e['key']} (class {e['class']}) "
                    f"but carries no {ANNOTATION_FIELD} annotation; the bytes may stay bound, but the "
                    f"record must say the member is logically quarantined at the named scope")
            continue
        if ann is None:
            continue
        if not isinstance(ann, list):
            problems.append(f"{label} {cid}: {ANNOTATION_FIELD} must be a list")
            continue
        annotated: dict[str, dict] = {}
        for i, a in enumerate(ann):
            if not isinstance(a, dict) or not a.get("key"):
                problems.append(f"{label} {cid}: {ANNOTATION_FIELD}[{i}] has no key")
                continue
            annotated[a["key"]] = a
            e = by_key.get(a["key"])
            if e is None:
                problems.append(f"{label} {cid}: {ANNOTATION_FIELD} names {a['key']!r}, which is not an "
                                f"exclusion in EXCLUSIONS.json")
                continue
            if e not in hits:
                problems.append(f"{label} {cid}: {ANNOTATION_FIELD} names {a['key']}, which does not name "
                                f"this record's digest or Drive id; a stale annotation is not an annotation")
            if a.get("class") != e["class"]:
                problems.append(f"{label} {cid}: {ANNOTATION_FIELD} {a['key']} class {a.get('class')!r} != "
                                f"exclusion class {e['class']!r}")
            if not str(a.get("scope") or "").strip():
                problems.append(f"{label} {cid}: {ANNOTATION_FIELD} {a['key']} has no scope; the excluded "
                                f"claim must be named, not implied")
        for e in hits:
            if e["key"] not in annotated:
                problems.append(
                    f"{label} {cid}: bound member is named by exclusion {e['key']} (class {e['class']}) "
                    f"but {ANNOTATION_FIELD} does not carry it")
    return problems


VAULT_PATH = "99_DO_NOT_OPEN"


def vault_rows(scan_root: str) -> list[str]:
    """Manifest rows that store bytes of an object inside the do-not-open vault.

    CLAUDE.md rule 9: the vault is never opened for authority, proofs,
    certificates or "latest" status unless an operator names a vault id for
    forensic recovery -- metadata only.  Until this check existed the rule was
    kept by care alone: nothing refused a row, and a later port sweeping "every
    remaining native Doc" would have taken the vault with it and passed every
    checker in the tree.  An INDEX row naming the vault is metadata and is
    allowed; a row with stored bytes is not.

    Scanned from the tree passed in, never from the module's own ROOT, so a
    control can run the checker against a synthetic root and have it look there.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(scan_root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", "__pycache__")]
        for fn in filenames:
            if fn != "_MANIFEST.jsonl" and fn != "MANIFEST.jsonl":
                continue
            path = os.path.join(dirpath, fn)
            with open(path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if not row.get("stored"):
                        continue
                    if VAULT_PATH in (row.get("drive_path") or ""):
                        found.append(
                            "%s: row for Drive id %s stores bytes from %s, which is "
                            "metadata only and is never opened"
                            % (os.path.relpath(path, scan_root), row.get("id"), VAULT_PATH))
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exclusions", default=EXCLUSIONS)
    ap.add_argument("--register", default=REGISTER)
    ap.add_argument("--archives", default=ARCHIVES)
    ap.add_argument("--binding", default=BINDING)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--scan-root", default=ROOT, help="tree scanned for content manifests (invariant 3)")
    args = ap.parse_args(argv)

    with open(args.exclusions, encoding="utf-8") as f:
        ex = json.load(f)["exclusions"]
    with open(args.register, encoding="utf-8") as f:
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
    with open(args.archives, encoding="utf-8") as f:
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

    digests = manifest_digests(args.scan_root)
    digest_comparable = digest_not_compared = 0
    for e in ex:
        d = (e.get("payload_sha256") or "").lower()
        if not d:
            # Invariant 3 cannot see this record, and the summary now says so
            # rather than counting it as a comparison. See DIGEST_NOT_COMPARED.
            digest_not_compared += 1
            if not uncompared_reason(e):
                problems.append(
                    f"{e['key']}: no `payload_sha256`, so invariant 3 cannot "
                    f"compare it against any manifest, and no reason is "
                    f"recorded -- neither a `{REASON_FIELD}` field on the "
                    f"record nor an entry in DIGEST_NOT_COMPARED in this file. "
                    f"A record may sit outside the digest comparison, but not "
                    f"silently: say why, so a new exclusion cannot join the "
                    f"unchecked set by omission.")
            continue
        digest_comparable += 1
        if d in digests:
            problems.append(
                f"{e['key']}: excluded payload {d[:16]} is consumed by manifest {digests[d]}")

    # A declared reason must describe a record that is really outside the
    # comparison. Left unchecked, a stale entry would sit here excusing a record
    # that later gained a digest, or naming a key no longer in the register.
    #
    # The absent-key arm asks only of a list the table is plausibly about: a
    # `--exclusions` copy built for a control shares none of these keys, and
    # reporting all six missing there would be noise, not drift. One declared key
    # present is what makes the rest's absence meaningful.
    by_key_all = {e["key"]: e for e in ex}
    any_declared_present = any(k in by_key_all for k in DIGEST_NOT_COMPARED)
    for key in sorted(DIGEST_NOT_COMPARED):
        rec = by_key_all.get(key)
        if rec is None:
            if any_declared_present:
                problems.append(
                    f"DIGEST_NOT_COMPARED names {key!r}, which is not an "
                    f"exclusion in EXCLUSIONS.json; a reason for a record that "
                    f"is not there excuses nothing and hides the list's drift")
        elif (rec.get("payload_sha256") or "").strip():
            problems.append(
                f"DIGEST_NOT_COMPARED names {key}, which now carries a "
                f"`payload_sha256` and is compared; remove the entry rather "
                f"than leaving a reason that has stopped being true")

    if ex and digest_comparable == 0:
        # The same vacuity floor `verify_manifests.py` and
        # `noncertifying_check.py` carry. An invariant that compared nothing is
        # not an invariant that held.
        problems.append(
            f"VACUOUS RUN: {len(ex)} exclusions and not one carried a "
            f"`payload_sha256`, so invariant 3 compared nothing at all. A pass "
            f"here would mean only that the loop ran.")

    problems += check_bound_member_annotations(ex, args.binding, args.manifest)
    vault = vault_rows(args.scan_root)
    problems += vault

    for p in problems:
        print(p)
    archive_members = sum(1 for e in ex if e.get("kind") == "archive_member")
    bound = 0
    try:
        for _label, rec in bound_records(args.binding, "BINDING") + bound_records(args.manifest, "MANIFEST"):
            if exclusions_naming(rec, ex):
                bound += 1
    except ValueError:
        pass
    print(f"exclusions={len(ex)} digest_comparable={digest_comparable} "
          f"digest_not_compared={digest_not_compared} "
          f"archive_members={archive_members} "
          f"manifest_digests={len(digests)} bound_members_named={bound} "
          f"vault_rows_storing_bytes={len(vault)} problems={len(problems)}")
    print("A pass here excludes claims; it certifies none. A bound member named by an exclusion stays "
          "bound as bytes and is logically quarantined at the scope its record names.")
    print("Invariant 3 compared the digest_comparable records only. `digest_not_compared` records carry no "
          "payload digest; each states why, on the record or in DIGEST_NOT_COMPARED in this file.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
