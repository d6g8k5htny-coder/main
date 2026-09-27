#!/usr/bin/env python3
"""Structural checks on the imported registers (registers/json/*.json).

These encode the invariants the Drive protocols (OP-PROT-012, OP-PROT-019/R17,
OP-GDN-002) place on the coupled registers, expressed as machine checks:

* every tab has a header and every row has at most len(header) cells;
* primary keys in key-bearing tabs are unique (review_queue, frozen_objects,
  closure_log, work_events, quarantine_index, dispatch_queue, artifact_index,
  evidence_lineage, transition_log, operator_decisions, reusable_operations,
  operation_trials, and since 2026-09-19 relations, review_ledger and
  definitions — three id-bearing tabs the checker had never keyed, which
  carry 14 duplicate keys between them);
* every cross-register observation recorded in KNOWN_FINDINGS.json (section
  ``observations_cross_register``: contradictions between tabs, or between a
  tab and the Drive source map, that no single-row rule can see) is still
  bound to the cells, inventory rows and path changes it cites — an
  observation whose evidence has drifted is a NEW problem;
* frozen_objects rows that declare a 64-hex SHA-256 have a positive byte count;
* review_queue rows carry one of the R17 technical statuses;
* quarantine_index classes are from the R17 classification table;
* work_events is append-only: the committed file may only grow (checked by the
  CI diff job, see tests/test_registers.py).

For an inventable agent the three KNOWN lines
``quarantine_index: row 14 class 'EXISTING_CONTAINER' not in R17 table``,
``row 15``, and ``row 16`` name ``Q-R17-LOCAL-TB``, ``Q-R17-LOCAL-P01``, and
``Q-R17-VAULT``. The rationale in ``registers/KNOWN_FINDINGS.json`` says
"Accepted as-is" for that source-workbook class. ``inventable_attempt_accepted``
stays false. A green run leaves OBL-H5-JETMOD OPEN. Quarantine is not a source
of truth. Agreement of ``quarantine/EXCLUSIONS.json`` with this export stays
with ``tools/quarantine_check.py``. See ``docs/math_status_probes/README.md``.

A problem whose exact string is listed in ``registers/KNOWN_FINDINGS.json`` (in
any top-level section whose name begins with ``findings``) is printed as KNOWN
and does not fail the run: that file records defects that exist in the SOURCE
workbook, which is never edited here. Anything else is NEW and fails the run.

Run:  python3 tools/registers_check.py [--json-dir DIR] [--known PATH]
                                       [--inventory PATH] [--path-changes PATH]
The flags exist so tests can point the checker at a mutated copy; the
defaults are the committed registers, allowlist, inventory and path-change
delta. Neither the observations nor the findings repair anything: the export
is regenerated from registers/source/ and never hand-edited.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = os.path.join(ROOT, "registers", "json")
KNOWN_PATH = os.path.join(ROOT, "registers", "KNOWN_FINDINGS.json")
INVENTORY_PATH = os.path.join(ROOT, "drive", "inventory.jsonl")
PATH_CHANGES_PATH = os.path.join(ROOT, "drive", "deltas", "2026-09-18", "PATH_CHANGES.jsonl")
OBSERVATIONS_SECTION = "observations_cross_register"

R17_TECH_STATUS = {"READY", "IN_REVIEW", "PASS_TECHNICAL", "AMEND", "FAIL",
                   "CANNOT_VERIFY", "NEEDS_RECONCILIATION"}
R17_QUARANTINE_CLASSES = {"EXACT_DUPLICATE", "SUPERSEDED", "DEFECTIVE_SCOPE",
                          "UNVERIFIED", "CONFLICT", "UNVERIFIED / CONFLICT",
                          "LEGACY_INSPIRATION", "LOGICAL_QUARANTINE"}
KEYED = {
    "review_queue": 0, "frozen_objects": 0, "closure_log": 0, "work_events": 0,
    "quarantine_index": 0, "dispatch_queue": 1, "artifact_index": 0,
    "evidence_lineage": 0, "transition_log": 0, "operator_decisions": 0,
    "reusable_operations": 0, "operation_trials": 0,
    # keyed since 2026-09-19; until then their duplicate ids were invisible here
    "relations": 0, "review_ledger": 0, "definitions": 0,
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def load(name: str, json_dir: str) -> dict:
    with open(os.path.join(json_dir, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def load_known(path: str) -> dict[str, str]:
    """registers/KNOWN_FINDINGS.json maps an exact problem string to a rationale.
    Findings listed there are printed as KNOWN and do not fail the run; this keeps
    the export faithful (we never edit Drive data to make a check pass) while
    surfacing the defects for the owner to resolve at the source. Matching is on
    the exact string: a near miss is NEW.

    Every top-level section whose name begins with ``findings`` is read: the
    file keeps the findings the 2026-09-17 markdown export showed (covered by
    registers/collision_proposal.json) apart from those only the 2026-09-18 xlsx
    export delivered (awaiting a successor proposal). A section that is not a
    mapping of string to string is a malformed allowlist and is refused."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    known: dict[str, str] = {}
    for section, entries in data.items():
        if not section.startswith("findings"):
            continue
        if not isinstance(entries, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in entries.items()):
            raise ValueError(f"{path}: section {section!r} is not a mapping of problem string to rationale")
        known.update(entries)
    return known


def check(json_dir: str) -> tuple[list[str], int]:
    """Return (problem strings, number of tabs checked)."""
    problems: list[str] = []
    names = sorted(fn[:-5] for fn in os.listdir(json_dir) if fn.endswith(".json"))
    for name in names:
        t = load(name, json_dir)
        if not t.get("header"):
            problems.append(f"{name}: empty header")
            continue
        width = len(t["header"])
        for i, r in enumerate(t["rows"]):
            if len(r) > width:
                problems.append(f"{name}: row {i} has {len(r)} cells > header width {width}")
        if name in KEYED:
            col = KEYED[name]
            seen: dict[str, int] = {}
            for i, r in enumerate(t["rows"]):
                key = r[col] if col < len(r) else ""
                if not key or key.startswith("[merged]") or key == "PROTOCOL":
                    continue
                if key in seen:
                    problems.append(f"{name}: duplicate key {key!r} at rows {seen[key]} and {i}")
                seen.setdefault(key, i)
    rq = load("review_queue", json_dir)
    st = rq["header"].index("Technical status")
    for i, r in enumerate(rq["rows"]):
        if r[st] not in R17_TECH_STATUS:
            problems.append(f"review_queue: row {i} status {r[st]!r} not in R17 set")
    fo = load("frozen_objects", json_dir)
    hb, hs = fo["header"].index("Expected Bytes"), fo["header"].index("Expected SHA-256 / Identity")
    for i, r in enumerate(fo["rows"]):
        if hs < len(r) and HEX64.match(r[hs].strip()):
            b = r[hb].strip() if hb < len(r) else ""
            if not b.isdigit() or int(b) <= 0:
                problems.append(f"frozen_objects: row {i} ({r[0]}) has SHA but non-numeric byte count {b!r}")
    qi = load("quarantine_index", json_dir)
    qc = qi["header"].index("Class")
    for i, r in enumerate(qi["rows"]):
        if r[qc] not in R17_QUARANTINE_CLASSES:
            problems.append(f"quarantine_index: row {i} class {r[qc]!r} not in R17 table")
    return problems, len(names)


def load_observations(path: str) -> dict[str, dict]:
    """The ``observations_cross_register`` section of KNOWN_FINDINGS.json: a
    mapping of observation id to a record carrying ``observation`` (text),
    ``bindings`` (the cells, inventory rows and path changes it cites) and
    ``proposed_repair``. The section name does not begin with ``findings``,
    so nothing in it is ever treated as an allowlisted problem string."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    obs = data.get(OBSERVATIONS_SECTION, {})
    if not isinstance(obs, dict):
        raise ValueError(f"{path}: section {OBSERVATIONS_SECTION!r} is not a mapping")
    for oid, rec in obs.items():
        if not isinstance(rec, dict) or not isinstance(rec.get("bindings"), list) or not rec["bindings"] \
                or not isinstance(rec.get("observation"), str) or not isinstance(rec.get("proposed_repair"), str):
            raise ValueError(f"{path}: observation {oid!r} lacks observation text, a non-empty bindings list "
                             f"or a proposed_repair")
    return obs


def _jsonl_index(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    out[row["id"]] = row
    return out


def check_observations(obs: dict[str, dict], json_dir: str, inventory_path: str,
                       path_changes_path: str) -> tuple[list[str], int]:
    """Return (problem strings, number of bindings checked). A binding is one of

    {"tab", "row", "column", "equals" | "contains"}   a cell of registers/json
    {"inventory_id", ...fields}                        a drive/inventory.jsonl row whose
                                                       named fields equal (or, for
                                                       "path_prefix", begin with) the value
    {"inventory_id", "absent": true}                   the id is NOT in the inventory
    {"path_change_id", ...fields}                      a PATH_CHANGES.jsonl row, fields equal

    Every failure names the observation and the binding, so a drift in the
    exported registers or the source map cannot leave a stale observation
    standing."""
    problems: list[str] = []
    inv = _jsonl_index(inventory_path)
    pc = _jsonl_index(path_changes_path)
    tabs: dict[str, dict] = {}
    n = 0
    for oid, rec in obs.items():
        for b in rec["bindings"]:
            n += 1
            where = f"{OBSERVATIONS_SECTION}: {oid} binding {json.dumps(b, ensure_ascii=False)}"
            if "tab" in b:
                name = b["tab"]
                try:
                    tab = tabs.setdefault(name, load(name, json_dir))
                except FileNotFoundError:
                    problems.append(f"{where}: tab {name!r} not exported")
                    continue
                header, rows = tab["header"], tab["rows"]
                if b["column"] not in header:
                    problems.append(f"{where}: no column {b['column']!r}")
                    continue
                col = header.index(b["column"])
                if not (0 <= b["row"] < len(rows)) or col >= len(rows[b["row"]]):
                    problems.append(f"{where}: row {b['row']} has no such cell")
                    continue
                cell = rows[b["row"]][col]
                if "equals" in b and cell != b["equals"]:
                    problems.append(f"{where}: cell reads {cell!r}")
                if "contains" in b and b["contains"] not in str(cell):
                    problems.append(f"{where}: cell reads {cell!r}")
                if "equals" not in b and "contains" not in b:
                    problems.append(f"{where}: binding asserts nothing")
            elif "inventory_id" in b:
                row = inv.get(b["inventory_id"])
                if b.get("absent"):
                    if row is not None:
                        problems.append(f"{where}: id IS in the inventory ({row.get('title')!r})")
                    continue
                if row is None:
                    problems.append(f"{where}: id not in the inventory")
                    continue
                for k, v in b.items():
                    if k == "inventory_id":
                        continue
                    if k == "path_prefix":
                        if not str(row.get("path", "")).startswith(v):
                            problems.append(f"{where}: path reads {row.get('path')!r}")
                    elif row.get(k) != v:
                        problems.append(f"{where}: {k} reads {row.get(k)!r}")
            elif "path_change_id" in b:
                row = pc.get(b["path_change_id"])
                if row is None:
                    problems.append(f"{where}: id not in PATH_CHANGES")
                    continue
                for k, v in b.items():
                    if k != "path_change_id" and row.get(k) != v:
                        problems.append(f"{where}: {k} reads {row.get(k)!r}")
            else:
                problems.append(f"{where}: unknown binding kind")
    return problems, n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json-dir", default=JSON_DIR, help="directory of <tab>.json files to check")
    ap.add_argument("--known", default=KNOWN_PATH, help="allowlist of exact problem strings")
    ap.add_argument("--inventory", default=INVENTORY_PATH, help="drive/inventory.jsonl the observations cite")
    ap.add_argument("--path-changes", default=PATH_CHANGES_PATH, help="PATH_CHANGES.jsonl the observations cite")
    args = ap.parse_args(argv)
    problems, ntabs = check(args.json_dir)
    known = load_known(args.known)
    new = [p for p in problems if p not in known]
    for p in problems:
        print(("KNOWN  " if p in known else "NEW    ") + p)
    obs = load_observations(args.known)
    obs_problems, nbind = check_observations(obs, args.json_dir, args.inventory, args.path_changes)
    for p in obs_problems:
        print("NEW    " + p)
    print(f"tabs={ntabs} problems={len(problems)} known={len(problems) - len(new)} new={len(new)} "
          f"observations={len(obs)} bindings={nbind} unbound={len(obs_problems)}")
    return 1 if new or obs_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
