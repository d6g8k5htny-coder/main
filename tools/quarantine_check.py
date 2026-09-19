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
     excluded artifact has been silently pulled into the verified content set;
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
    for e in ex:
        d = (e.get("payload_sha256") or "").lower()
        if d and d in digests:
            problems.append(
                f"{e['key']}: excluded payload {d[:16]} is consumed by manifest {digests[d]}")

    problems += check_bound_member_annotations(ex, args.binding, args.manifest)

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
    print(f"exclusions={len(ex)} archive_members={archive_members} "
          f"manifest_digests={len(digests)} bound_members_named={bound} problems={len(problems)}")
    print("A pass here excludes claims; it certifies none. A bound member named by an exclusion stays "
          "bound as bytes and is logically quarantined at the scope its record names.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
