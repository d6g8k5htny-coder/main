#!/usr/bin/env python3
"""Checker for registers/COLLISION_PROPOSAL.md + registers/collision_proposal.json.

The proposal is a PROPOSAL: it disambiguates the 16 structural defects recorded in
registers/KNOWN_FINDINGS.json without repairing anything.  This checker enforces that
promise mechanically:

  1. bijection  — every KNOWN_FINDINGS finding has exactly one proposal record, and
                  every proposal record names a real KNOWN_FINDINGS finding;
  2. additive   — every executable operation in the proposal is an append; deletion,
                  merge and overwrite verbs are rejected wherever they appear, and the
                  non-additive follow-ups a full remedy would need are quarantined in a
                  separate, operator-reserved field that carries no executable op;
  3. unchanged  — registers/source, registers/json and registers/csv are byte-identical
                  to git HEAD (the proposal must not have edited exported data), and the
                  export digest recorded in the proposal matches the file on disk;
  4. successors — every proposed successor identifier is unique within the proposal and
                  does not already exist anywhere in registers/json/;
  5. honesty    — independence credit is recorded at 0 with its reason, the
                  independence-requiring gates are stated to remain open, every record
                  states what it does not establish, every quoted export line matches the
                  export byte for byte, and the Markdown states that nothing was repaired.

Exit status is nonzero if any check fails.
Run:  python3 tools/collision_proposal_check.py [-v]
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, "registers", "collision_proposal.json")
MD_PATH = os.path.join(ROOT, "registers", "COLLISION_PROPOSAL.md")
KNOWN_PATH = os.path.join(ROOT, "registers", "KNOWN_FINDINGS.json")
JSON_DIR = os.path.join(ROOT, "registers", "json")
EXPORT = os.path.join(ROOT, "registers", "source",
                      "GP-REG-032_v1.2_export_2026-09-17.md")

# Operations the proposal is allowed to contain.  Anything else is a mutation.
ADDITIVE_OPS = {"APPEND_ROW"}
# Follow-ups a full remedy would eventually need.  They are recorded, never proposed
# for execution, and must be flagged operator_reserved.
RESERVED_OPS = {"REIDENTIFY_KEY_CELL", "AMEND_PROTOCOL_TABLE"}
# Verbs that would destroy or silently merge recorded state.  OP-CNS-001 §2 requires
# collisions to be preserved and disambiguated; OP-PROT-019 §6 forbids deletion
# outright ("No permanent deletion in this workflow").
FORBIDDEN_VERBS = ("DELETE", "REMOVE", "MERGE", "PURGE", "OVERWRITE", "REPLACE",
                   "DROP", "TRUNCATE", "CLEAR", "DEDUPE", "DEDUPLICATE", "COLLAPSE")
GUARDED_PATHS = ("registers/source", "registers/json", "registers/csv")
TOKEN_SPLIT = re.compile(r"[\s;,|]+")


class Result:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def check(self, ok: bool, msg: str) -> bool:
        if not ok:
            self.failures.append(msg)
        return ok

    def note(self, msg: str) -> None:
        self.notes.append(msg)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def existing_identifiers() -> set[str]:
    """Every string that already names something in registers/json/: whole cell values
    and whitespace/semicolon/comma/pipe-delimited tokens inside them, plus headers and
    tab names.  Matching is on exact values, never substrings, so a successor that
    merely *contains* an existing id (e.g. 'TR-P01-006-COLLISION-PROVENANCE' contains
    'TR-P01-006') is correctly treated as new."""
    out: set[str] = set()
    for fn in sorted(os.listdir(JSON_DIR)):
        if not fn.endswith(".json"):
            continue
        t = load_json(os.path.join(JSON_DIR, fn))
        out.add(t.get("tab", ""))
        for cell in list(t.get("header", [])) + [c for r in t.get("rows", []) for c in r]:
            cell = (cell or "").strip()
            if not cell:
                continue
            out.add(cell)
            for tok in TOKEN_SPLIT.split(cell):
                tok = tok.strip(" \t'\"()[]{}<>.:")
                if tok:
                    out.add(tok)
    out.discard("")
    return out


def walk_strings(node, path="$"):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk_strings(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk_strings(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def check_bijection(doc, res: Result) -> None:
    known = set(load_json(KNOWN_PATH).get("findings", {}))
    proposed: list[str] = [p.get("finding_key", "") for p in doc.get("proposals", [])]
    seen: set[str] = set()
    for k in proposed:
        res.check(k in known, f"proposal names a finding absent from KNOWN_FINDINGS.json: {k!r}")
        res.check(k not in seen, f"two proposal records claim the same finding key: {k!r}")
        seen.add(k)
    for k in sorted(known - seen):
        res.check(False, f"KNOWN_FINDINGS finding has no proposal record: {k!r}")
    res.check(len(proposed) == len(known),
              f"record count {len(proposed)} != finding count {len(known)}")


def check_operations(doc, res: Result) -> None:
    for p in doc.get("proposals", []):
        rid = p.get("record_id", "?")
        ops = p.get("operations")
        res.check(isinstance(ops, list) and len(ops) >= 1,
                  f"{rid}: no operations block")
        for i, op in enumerate(ops or []):
            name = op.get("operation", "")
            res.check(name in ADDITIVE_OPS,
                      f"{rid}: operation[{i}] {name!r} is not additive "
                      f"(allowed: {sorted(ADDITIVE_OPS)})")
            res.check(op.get("append_only") is True,
                      f"{rid}: operation[{i}] does not declare append_only")
            res.check(op.get("mutates_existing_rows") is False,
                      f"{rid}: operation[{i}] does not declare mutates_existing_rows=false")
            res.check(isinstance(op.get("row"), list) and len(op["row"]) == len(op.get("header", [])),
                      f"{rid}: operation[{i}] row width does not match its header")
            for verb in FORBIDDEN_VERBS:
                res.check(verb not in name.upper(),
                          f"{rid}: operation[{i}] name contains forbidden verb {verb}")
        for i, fu in enumerate(p.get("non_additive_followups", []) or []):
            name = fu.get("op", "")
            res.check(name in RESERVED_OPS,
                      f"{rid}: non_additive_followups[{i}] {name!r} is not a recognised "
                      f"operator-reserved action (allowed: {sorted(RESERVED_OPS)})")
            res.check(fu.get("operator_reserved") is True,
                      f"{rid}: non_additive_followups[{i}] is not marked operator_reserved")
            res.check(bool(fu.get("why_not_in_operations")),
                      f"{rid}: non_additive_followups[{i}] does not say why it is excluded "
                      f"from the executable operations")
            res.check(name not in ADDITIVE_OPS,
                      f"{rid}: non_additive_followups[{i}] leaks into the additive set")
            for verb in FORBIDDEN_VERBS:
                res.check(verb not in name.upper(),
                          f"{rid}: non_additive_followups[{i}] contains forbidden verb {verb}")
    # A deletion must not hide anywhere in a machine-readable action field.
    for path, val in walk_strings(doc, "$"):
        if path.endswith(".operation") or path.endswith(".op"):
            for verb in FORBIDDEN_VERBS:
                res.check(verb not in val.upper(),
                          f"forbidden verb {verb} in action field {path}: {val!r}")


def check_registers_unchanged(res: Result) -> None:
    """registers/source, registers/json and registers/csv must be byte-identical to HEAD."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--"] + list(GUARDED_PATHS),
                             cwd=ROOT, capture_output=True, text=True, timeout=60)
        git_ok = out.returncode == 0
    except (OSError, subprocess.SubprocessError):
        git_ok = False
        out = None
    if git_ok and out is not None:
        dirty = [l for l in out.stdout.splitlines() if l.strip()]
        for l in dirty:
            res.check(False, f"exported register data is not byte-unchanged against git HEAD: {l.strip()}")
        if not dirty:
            res.note("registers/source, registers/json, registers/csv byte-unchanged against git HEAD")
    else:
        res.note("git unavailable: HEAD comparison skipped; falling back to re-import equality")
        try:
            sys.path.insert(0, os.path.join(ROOT, "tools"))
            import registers_import  # noqa: E402
            rc = subprocess.run([sys.executable, registers_import.__file__, "--check"],
                                cwd=ROOT, capture_output=True, text=True, timeout=300)
            res.check(rc.returncode == 0,
                      "registers_import.py --check failed: registers/json or registers/csv "
                      "no longer match registers/source")
        except Exception as exc:  # pragma: no cover - environment dependent
            res.check(False, f"could not verify exported registers are unchanged: {exc}")


def collect_successors(doc) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in doc.get("proposals", []):
        rid = p.get("record_id", "?")
        s = p.get("successor")
        if isinstance(s, dict) and s.get("proposed_id"):
            out.append((rid, s["proposed_id"]))
        for x in p.get("successors", []) or []:
            if x.get("proposed_id"):
                out.append((rid, x["proposed_id"]))
    return out


def check_successors(doc, res: Result) -> None:
    existing = existing_identifiers()
    succ = collect_successors(doc)
    res.check(len(succ) >= 1, "proposal issues no successor identifiers at all")
    seen: dict[str, str] = {}
    for rid, sid in succ:
        res.check(sid not in seen,
                  f"{rid}: successor id {sid!r} is already issued by {seen.get(sid)}")
        seen[sid] = rid
        res.check(sid not in existing,
                  f"{rid}: successor id {sid!r} already exists in registers/json/")
    batch = doc.get("batch_level_artifacts_that_would_also_be_appended", {}) or {}
    for field in ("no_change_certificate", "correction_record"):
        pid = (batch.get(field) or {}).get("proposed_id")
        if pid:
            res.check(pid not in existing,
                      f"batch {field} id {pid!r} already exists in registers/json/")
            res.check(pid not in seen, f"batch {field} id {pid!r} collides with a successor id")
    # Every record that names a colliding key must either issue a successor or say why not.
    for p in doc.get("proposals", []):
        rid = p.get("record_id", "?")
        has = bool(p.get("successors")) or bool((p.get("successor") or {}).get("proposed_id")
                                                if isinstance(p.get("successor"), dict) else False)
        if not has:
            res.check(bool(p.get("successor_not_applicable_reason")),
                      f"{rid}: no successor issued and no successor_not_applicable_reason given")


def check_verbatim(doc, res: Result) -> None:
    if not os.path.exists(EXPORT):
        res.check(False, f"source export missing: {EXPORT}")
        return
    raw = open(EXPORT, encoding="utf-8").read()
    lines = raw.split("\n")
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    rec = (doc.get("source_of_record") or {}).get("sha256")
    res.check(rec == sha, f"recorded export sha256 {rec} != actual {sha}")
    for p in doc.get("proposals", []):
        rid = p.get("record_id", "?")
        rows = p.get("rows") or []
        res.check(len(rows) >= 1, f"{rid}: no colliding rows quoted")
        for r in rows:
            ln = r.get("export_line_number")
            quoted = r.get("verbatim_export_line", "")
            ok = isinstance(ln, int) and 1 <= ln <= len(lines) and lines[ln - 1] == quoted
            res.check(ok, f"{rid}: quoted export line {ln} does not match the export byte for byte")
            res.check(hashlib.sha256(quoted.encode("utf-8")).hexdigest() == r.get("export_line_sha256"),
                      f"{rid}: recorded digest for export line {ln} does not match the quoted bytes")


def check_honesty(doc, res: Result) -> None:
    pb = doc.get("prepared_by") or {}
    res.check(pb.get("independence_credit") == 0,
              "prepared_by.independence_credit must be 0")
    res.check(bool(pb.get("independence_credit_reason")),
              "prepared_by.independence_credit_reason is missing")
    res.check("REMAIN" in str(pb.get("independence_requiring_gates_remain_open", "")).upper(),
              "prepared_by must state that independence-requiring gates remain open")
    res.check(doc.get("nothing_repaired") is True, "doc must declare nothing_repaired=true")
    res.check(doc.get("export_remains_faithful") is True,
              "doc must declare export_remains_faithful=true")
    res.check(bool(doc.get("does_not_establish")), "doc-level does_not_establish is missing")
    for p in doc.get("proposals", []):
        rid = p.get("record_id", "?")
        res.check(bool(p.get("does_not_establish")), f"{rid}: does_not_establish is missing")
        res.check(bool(p.get("governing_clauses")), f"{rid}: governing_clauses is missing")
        res.check(bool(p.get("residual_questions")), f"{rid}: residual_questions is missing")
        if p.get("register_tab") == "quarantine_index":
            opts = p.get("options") or []
            res.check(len(opts) >= 2,
                      f"{rid}: an EXISTING_CONTAINER record must give at least two options")
            for o in opts:
                res.check(bool(o.get("consequences")),
                          f"{rid}: option {o.get('option')!r} has no consequences")
            res.check(bool(p.get("recommendation")), f"{rid}: no recommendation given")
        else:
            res.check(bool((p.get("keeper") or {}).get("reason")),
                      f"{rid}: keeper has no stated reason")


def check_markdown(res: Result) -> None:
    if not os.path.exists(MD_PATH):
        res.check(False, f"missing companion document: {MD_PATH}")
        return
    md = open(MD_PATH, encoding="utf-8").read()
    for needle, what in [
        ("requires operator action", "the proposal-requires-operator-action statement"),
        ("Nothing has been repaired", "the nothing-has-been-repaired statement"),
        ("export remains faithful", "the export-remains-faithful statement"),
        ("independence_credit = 0", "the zero-independence-credit record"),
        ("REMAINS OPEN", "the independence-gate-remains-open statement"),
    ]:
        res.check(needle.lower() in md.lower(), f"{MD_PATH} does not state {what}")
    known = load_json(KNOWN_PATH).get("findings", {})
    for k in known:
        res.check(k in md, f"{MD_PATH} does not quote the finding key {k!r}")


def main(argv: list[str]) -> int:
    verbose = "-v" in argv
    res = Result()
    if not os.path.exists(JSON_PATH):
        print(f"FAIL  missing {JSON_PATH}")
        return 1
    doc = load_json(JSON_PATH)
    check_bijection(doc, res)
    check_operations(doc, res)
    check_registers_unchanged(res)
    check_successors(doc, res)
    check_verbatim(doc, res)
    check_honesty(doc, res)
    check_markdown(res)
    for n in res.notes:
        if verbose:
            print("NOTE  " + n)
    for f in res.failures:
        print("FAIL  " + f)
    n = len(doc.get("proposals", []))
    print(f"records={n} successors={len(collect_successors(doc))} failures={len(res.failures)}")
    return 1 if res.failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
