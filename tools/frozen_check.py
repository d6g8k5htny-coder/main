#!/usr/bin/env python3
"""Cross-check the Drive's frozen-object register against the source map, offline.

The Drive program freezes objects by recording, in the ``frozen_objects``
register tab, a Drive ID, an expected byte count and an expected SHA-256, under
a *binding class* that says what the digest is a digest of:

* class **A** (whole-file byte freeze) and **D** (exact raw file) — the digest is
  of the whole Drive object. ``drive/inventory.jsonl`` records a SHA-256 and a
  byte count for raw files, so these rows are **comparable offline**: the two
  Drive-side records either agree or they do not.
* class **B** (marker-delimited frozen body inside a document) and **C** (native
  export of a Google Doc) — the digest is of an *extracted body*, not of the
  object the inventory measures. A native Doc has no payload digest in the
  inventory and its "bytes" are Drive's size for the Doc, so a byte or digest
  comparison against the inventory is a category error, not a drift check. These
  rows are reported ``BODY_PRESENT_AS_PAYLOAD`` when the expected body digest
  occurs anywhere in ``Payloads.csv``, ``Archive_Members.csv`` or the inventory
  (the body exists byte-exact somewhere the source map indexes), and
  ``NOT_COMPARABLE_OFFLINE`` otherwise.

What a pass means: every frozen object the register binds by whole-file digest
agrees with the accessibility publication's independent digest of the same
Drive ID, and no row is structurally unreadable. What it does not mean: nothing
here re-verifies a frozen body this repository does not hold, re-freezes
anything, or touches the register (an export; never edited). A row the register
itself marks with a non-PASS ``Drift Status`` is carried verbatim in the report
and is not re-graded here.

Exit status is nonzero on any ``MISMATCH`` or structural problem whose exact
problem string is not allowlisted in ``registers/KNOWN_FINDINGS.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FROZEN = os.path.join(ROOT, "registers", "json", "frozen_objects.json")
DEFAULT_INVENTORY = os.path.join(ROOT, "drive", "inventory.jsonl")
DEFAULT_PAYLOADS = os.path.join(ROOT, "drive", "source_map", "Payloads.csv")
DEFAULT_MEMBERS = os.path.join(ROOT, "drive", "source_map", "Archive_Members.csv")
DEFAULT_KNOWN = os.path.join(ROOT, "registers", "KNOWN_FINDINGS.json")
DEFAULT_CUSTODY = [
    os.path.join(ROOT, "drive", "custody", "2026-09-24_frozen_objects.json"),
    os.path.join(ROOT, "drive", "custody", "2026-09-23_frozen_objects_chatgpt.json"),
]

HEX64 = re.compile(r"\b([0-9a-fA-F]{64})\b")

WHOLE_FILE_CLASSES = ("A", "D")
BODY_CLASSES = ("B", "C")

# Row statuses. Only MISMATCH fails the run.
MATCH = "MATCH"
MISMATCH = "MISMATCH"
UNVERIFIABLE_NO_INVENTORY_DIGEST = "UNVERIFIABLE_NO_INVENTORY_DIGEST"
BODY_PRESENT_AS_PAYLOAD = "BODY_PRESENT_AS_PAYLOAD"
NOT_COMPARABLE_OFFLINE = "NOT_COMPARABLE_OFFLINE"
# An agreement answered from a custody supplement rather than from the accessibility
# publication. Never folded into MATCH: see load_custody.
MATCH_VIA_CUSTODY_SUPPLEMENT = "MATCH_VIA_CUSTODY_SUPPLEMENT"


def load_frozen(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["header"], d["rows"]


def load_inventory(path: str) -> Dict[str, dict]:
    inv: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                inv[r["id"]] = r
    return inv



def load_custody(paths, inventory: Dict[str, dict]) -> Tuple[Dict[str, dict], List[str]]:
    """Identities collected after the accessibility publication, kept apart from it.

    ``drive/inventory.jsonl`` derives from one dated publication and cannot
    contain an object frozen after it. A supplement supplies those rows so the
    whole-file freeze is comparable at all — but a supplement row is a weaker
    thing than an inventory row, and the difference is not cosmetic.

    frozen_check's whole-file comparison is worth something only because it puts
    two *independent* Drive-side records of the same bytes side by side. A
    supplement row copied out of the register's own expected digest would make
    that comparison circular, and the MATCH it produced would mean nothing at
    all. No check can distinguish a genuinely downloaded digest from a copied
    one, so this does not pretend to: a row answered here is reported
    MATCH_VIA_CUSTODY_SUPPLEMENT and counted separately, and a supplement may
    never answer for a Drive ID the publication already covers.

    Several supplements may be supplied, because more than one agent collects
    these objects. Two rules govern what happens when they overlap, and both are
    fail-closed:

    * **Disagreement is refused, not resolved.** If two supplements give the same
      Drive ID different bytes or a different digest, the ID is dropped and the
      conflict is reported. Picking one of two contradictory records is how a
      checker launders an unknown into an answer; there is no ordering of files
      that makes it safe.
    * **Agreement is recorded, never promoted.** Each admitted entry carries the
      set of *distinct* collectors that reported it, so the summary can say how
      many rows two parties agree on. That count says something real — two
      separate downloads of the same bytes is better evidence than one — and it
      still leaves the status at MATCH_VIA_CUSTODY_SUPPLEMENT. Two supplements
      naming the same collector are one collector twice and are counted once.
      None of this is organizational-independence credit, which is a question
      about review, not about custody of a digest.
    """
    problems: List[str] = []
    if isinstance(paths, str) or paths is None:
        paths = [paths] if paths else []
    out: Dict[str, dict] = {}
    conflicted: Set[str] = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        base = os.path.basename(path)
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError) as exc:
            problems.append(f"custody supplement {base}: unreadable ({exc})")
            continue
        if not isinstance(doc, dict):
            problems.append(f"custody supplement {base}: not a JSON object")
            continue
        for field in ("collected_by", "collection_method", "does_not_establish"):
            if not str(doc.get(field) or "").strip():
                problems.append(f"custody supplement {base}: no {field}")
        seen: Set[str] = set()
        for row in doc.get("rows", []):
            rid = str(row.get("id") or "").strip()
            if not rid:
                problems.append(f"custody supplement {base}: a row has no Drive id")
                continue
            if rid in inventory:
                problems.append(f"custody supplement {base}: {rid} is already in the inventory; a "
                                f"supplement never overrides the accessibility publication")
                continue
            if not HEX64.fullmatch(str(row.get("sha256") or "")):
                problems.append(f"custody supplement {base}: {rid} has no 64-hex sha256")
                continue
            if not isinstance(row.get("bytes"), int):
                problems.append(f"custody supplement {base}: {rid} has no integer byte count")
                continue
            collector = str(row.get("verified_by") or "").strip()
            if not collector:
                problems.append(f"custody supplement {base}: {rid} does not say who verified it")
                continue
            if rid in seen:
                problems.append(f"custody supplement {base}: {rid} appears twice")
                continue
            seen.add(rid)
            sha = str(row["sha256"]).lower()
            prior = out.get(rid)
            if prior is None:
                out[rid] = {"id": rid, "title": row.get("title"), "bytes": row["bytes"],
                            "sha256": sha, "access_status": "CUSTODY_SUPPLEMENT",
                            "collectors": {collector}, "supplements": {base}}
                continue
            if prior["sha256"] != sha or prior["bytes"] != row["bytes"]:
                problems.append(
                    f"custody supplements disagree about {rid}: "
                    f"{sorted(prior['supplements'])} say {prior['sha256'][:12]}… "
                    f"{prior['bytes']} B, {base} says {sha[:12]}… {row['bytes']} B; "
                    f"neither is used")
                conflicted.add(rid)
                continue
            prior["collectors"].add(collector)
            prior["supplements"].add(base)
    for rid in conflicted:
        out.pop(rid, None)
    return out, problems


def digest_index(inventory: Dict[str, dict], payloads: str, members: str) -> Dict[str, Set[str]]:
    """Every 64-hex digest the source map knows, mapped to where it occurs."""
    idx: Dict[str, Set[str]] = defaultdict(set)
    for r in inventory.values():
        if r.get("sha256"):
            idx[str(r["sha256"]).lower()].add("inventory")
    for fn, kind in ((payloads, "payloads"), (members, "archive_member")):
        if not os.path.exists(fn):
            continue
        with open(fn, encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            cols = [c for c in (rd.fieldnames or []) if "sha" in c.lower() or "digest" in c.lower()]
            for row in rd:
                for c in cols:
                    v = (row.get(c) or "").strip().lower()
                    if len(v) == 64:
                        idx[v].add(kind)
    return idx


def column(header: List[str], name: str) -> int:
    for i, h in enumerate(header):
        if h.strip().lower() == name.lower():
            return i
    raise KeyError(f"frozen_objects has no column {name!r}; header is {header}")


def check(frozen: str, inventory_path: str, payloads: str, members: str,
          custody_paths=None) -> Tuple[List[dict], List[str]]:
    """Return (per-row report, structural problems)."""
    header, rows = load_frozen(frozen)
    inv = load_inventory(inventory_path)
    custody, custody_problems = load_custody(custody_paths, inv)
    inv.update(custody)
    idx = digest_index(inv, payloads, members)
    c_obj = column(header, "Object ID")
    c_id = column(header, "Drive ID")
    c_cls = column(header, "Binding Class")
    c_bytes = column(header, "Expected Bytes")
    c_hex = column(header, "Expected SHA-256 / Identity")
    c_drift = column(header, "Drift Status")

    report: List[dict] = []
    problems: List[str] = list(custody_problems)

    def cell(r: List[str], i: int) -> str:
        return r[i].strip() if i < len(r) else ""

    for n, r in enumerate(rows):
        oid = cell(r, c_obj) or f"<row {n}>"
        did = cell(r, c_id)
        cls = cell(r, c_cls)
        letter = cls[:1].upper()
        expected_bytes = cell(r, c_bytes)
        hexes = [h.lower() for h in HEX64.findall(cell(r, c_hex))]
        entry = {
            "object_id": oid, "drive_id": did, "binding_class": cls,
            "register_drift_status": cell(r, c_drift),
            "status": None, "detail": "",
        }
        if letter not in WHOLE_FILE_CLASSES + BODY_CLASSES:
            problems.append(f"frozen_objects: {oid}: unknown binding class {cls!r}")
            entry["status"] = "UNKNOWN_CLASS"
            report.append(entry)
            continue
        if not hexes:
            problems.append(f"frozen_objects: {oid}: no 64-hex digest in 'Expected SHA-256 / Identity'")
            entry["status"] = "NO_DIGEST"
            report.append(entry)
            continue
        row = inv.get(did)
        if row is None:
            problems.append(f"frozen_objects: {oid}: Drive ID {did!r} is not in the inventory")
            entry["status"] = "DRIVE_ID_NOT_IN_INVENTORY"
            report.append(entry)
            continue

        if letter in WHOLE_FILE_CLASSES:
            inv_sha = str(row.get("sha256") or "").lower()
            if not inv_sha:
                entry["status"] = UNVERIFIABLE_NO_INVENTORY_DIGEST
                entry["detail"] = (f"inventory row {did} (access_status={row.get('access_status')}) carries "
                                   f"no payload digest, so the whole-file freeze cannot be compared offline")
            else:
                sha_ok = inv_sha in hexes
                bytes_ok = True
                if expected_bytes.isdigit() and row.get("bytes") is not None:
                    bytes_ok = int(expected_bytes) == int(row["bytes"])
                if sha_ok and bytes_ok:
                    via_supplement = did in custody
                    entry["status"] = MATCH_VIA_CUSTODY_SUPPLEMENT if via_supplement else MATCH
                    if via_supplement:
                        collectors = sorted(custody[did]["collectors"])
                        entry["collectors"] = collectors
                        source = ("a custody supplement collected after the accessibility "
                                  f"publication by {', '.join(collectors)}")
                    else:
                        source = "the inventory"
                    entry["detail"] = (f"sha256 {inv_sha[:12]}… and {row.get('bytes')} bytes "
                                       f"agree with {source}")
                else:
                    entry["status"] = MISMATCH
                    entry["detail"] = (f"register expects {'/'.join(h[:12] for h in hexes)}… {expected_bytes} B; "
                                       f"inventory has {inv_sha[:12]}… {row.get('bytes')} B")
                    problems.append(f"frozen_objects: {oid}: whole-file freeze (class {letter}) disagrees with "
                                    f"the inventory digest or byte count for Drive ID {did}")
        else:
            where = sorted({w for h in hexes for w in idx.get(h, ())})
            if where:
                entry["status"] = BODY_PRESENT_AS_PAYLOAD
                entry["detail"] = f"expected body digest occurs in the source map as: {', '.join(where)}"
            else:
                entry["status"] = NOT_COMPARABLE_OFFLINE
                entry["detail"] = ("the digest is of a marker-delimited or exported body; the inventory "
                                   "measures the containing object, so no offline comparison is valid")
        report.append(entry)
    return report, problems


def load_known(path: Optional[str]) -> Dict[str, str]:
    """Exact problem strings this run prints as allowlisted instead of failing on.

    Two shapes, because two of them are in use and only one of them was read.

    A top-level key whose value is a **string** is a flat entry: the problem
    string maps to its rationale. That is the shape this module has always
    accepted, and ``tests/test_frozen_check.py`` exercises it.

    A top-level key whose value is an **object** is a section. Sections whose
    name begins with ``findings`` are flattened, the way
    ``tools/registers_check.py`` reads them; any other section is skipped.

    The second half is the repair. ``registers/KNOWN_FINDINGS.json`` keeps every
    one of its 37 entries inside ``findings``, ``findings_first_visible_in_2026-09-18_export``
    and ``findings_first_keyed_2026-09-19``. Reading only flat keys made the
    loader hand back those three names plus ``observations_cross_register`` and
    ``superseded_source_export`` — section and metadata names, which no problem
    string can equal. So the escape hatch this module's docstring advertises
    worked for a file shape the repository does not use, and against the file it
    actually names it could match nothing. Not inert in general; unreachable in
    practice, which is the harder kind to notice.

    ``observations_cross_register`` stays unread. Its entries carry a
    ``proposed_repair``: they are cross-register observations, not defects
    anyone agreed to live with, and registers_check excludes it for that reason.

    This makes the file's entries reachable. It does not use them: no problem the
    committed register produces appears there, and a test pins that, so the
    repair changes no current verdict.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    known: Dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            if not key.startswith("findings"):
                continue
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
                raise ValueError(f"{path}: section {key!r} is not a mapping of "
                                 f"problem string to rationale")
            known.update(value)
        elif isinstance(value, str):
            if key.startswith("_") or key.endswith("source_export"):
                continue
            known[key] = value
        else:
            raise ValueError(f"{path}: {key!r} is neither a findings section nor a "
                             f"problem string mapped to a rationale")
    return known


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frozen", default=DEFAULT_FROZEN)
    ap.add_argument("--inventory", default=DEFAULT_INVENTORY)
    ap.add_argument("--payloads", default=DEFAULT_PAYLOADS)
    ap.add_argument("--members", default=DEFAULT_MEMBERS)
    ap.add_argument("--known", default=DEFAULT_KNOWN, help="allowlist of exact problem strings")
    ap.add_argument("--custody", action="append", default=None, metavar="PATH",
                    help="identities collected after the accessibility publication; "
                         "answers are reported apart from inventory-backed ones. Repeatable: "
                         "supplements that disagree about a Drive ID are refused, not resolved. "
                         "Pass --no-custody to read none")
    ap.add_argument("--no-custody", action="store_true",
                    help="read no custody supplement at all")
    ap.add_argument("--json", action="store_true", help="print the per-row report as JSON")
    args = ap.parse_args(argv)

    custody_paths = [] if args.no_custody else (args.custody or DEFAULT_CUSTODY)
    report, problems = check(args.frozen, args.inventory, args.payloads, args.members,
                             custody_paths)
    known = load_known(args.known)
    live = [p for p in problems if p not in known]
    allowed = [p for p in problems if p in known]

    if args.json:
        print(json.dumps({"rows": report, "problems": problems, "allowlisted": allowed}, indent=1))
    counts = Counter(e["status"] for e in report)
    corroborated = sum(1 for e in report
                       if e["status"] == MATCH_VIA_CUSTODY_SUPPLEMENT
                       and len(e.get("collectors") or ()) > 1)
    for p in live:
        print(p)
    for p in allowed:
        print(f"(allowlisted) {p}")
    print(f"frozen_check: rows={len(report)} "
          f"comparable={counts[MATCH] + counts[MISMATCH] + counts[MATCH_VIA_CUSTODY_SUPPLEMENT]} "
          f"match={counts[MATCH]} match_via_supplement={counts[MATCH_VIA_CUSTODY_SUPPLEMENT]} "
          f"supplement_collectors_agreeing={corroborated} "
          f"mismatch={counts[MISMATCH]} "
          f"unverifiable={counts[UNVERIFIABLE_NO_INVENTORY_DIGEST]} "
          f"body_present={counts[BODY_PRESENT_AS_PAYLOAD]} not_comparable={counts[NOT_COMPARABLE_OFFLINE]} "
          f"problems={len(live)}")
    print("A match here is agreement between two Drive-side records of the same bytes. It re-freezes "
          "nothing, verifies no body this repository does not hold, and moves no status. A "
          "match_via_supplement rests on a later spot-check rather than on the accessibility "
          "publication, and is counted apart from match for exactly that reason. "
          "supplement_collectors_agreeing counts the supplement-answered rows that two or more "
          "DISTINCT collectors reported identically: better evidence about bytes, and still not a "
          "match, not a verified body and not independence credit.")
    return 1 if live else 0


if __name__ == "__main__":
    raise SystemExit(main())
