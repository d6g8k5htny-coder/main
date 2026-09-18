#!/usr/bin/env python3
"""Structural checks on the imported registers (registers/json/*.json).

These encode the invariants the Drive protocols (OP-PROT-012, OP-PROT-019/R17,
OP-GDN-002) place on the coupled registers, expressed as machine checks:

* every tab has a header and every row has at most len(header) cells;
* primary keys in key-bearing tabs are unique (review_queue, frozen_objects,
  closure_log, work_events, quarantine_index, dispatch_queue, artifact_index);
* frozen_objects rows that declare a 64-hex SHA-256 have a positive byte count;
* review_queue rows carry one of the R17 technical statuses;
* quarantine_index classes are from the R17 classification table;
* work_events is append-only: the committed file may only grow (checked by the
  CI diff job, see tests/test_registers.py).
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = os.path.join(ROOT, "registers", "json")

R17_TECH_STATUS = {"READY", "IN_REVIEW", "PASS_TECHNICAL", "AMEND", "FAIL",
                   "CANNOT_VERIFY", "NEEDS_RECONCILIATION"}
R17_QUARANTINE_CLASSES = {"EXACT_DUPLICATE", "SUPERSEDED", "DEFECTIVE_SCOPE",
                          "UNVERIFIED", "CONFLICT", "UNVERIFIED / CONFLICT",
                          "LEGACY_INSPIRATION", "LOGICAL_QUARANTINE"}
KEYED = {
    "review_queue": 0, "frozen_objects": 0, "closure_log": 0, "work_events": 0,
    "quarantine_index": 0, "dispatch_queue": 1, "artifact_index": 0,
    "evidence_lineage": 0, "transition_log": 0, "operator_decisions": 0,
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def load(name: str) -> dict:
    with open(os.path.join(JSON_DIR, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def load_known() -> dict[str, str]:
    """registers/KNOWN_FINDINGS.json maps an exact problem string to a rationale.
    Findings listed there are printed as KNOWN and do not fail the run; this keeps
    the export faithful (we never edit Drive data to make a check pass) while
    surfacing the defects for the owner to resolve at the source."""
    path = os.path.join(ROOT, "registers", "KNOWN_FINDINGS.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.get("findings", {}).items()}


def main() -> int:
    problems: list[str] = []
    names = sorted(fn[:-5] for fn in os.listdir(JSON_DIR) if fn.endswith(".json"))
    for name in names:
        t = load(name)
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
    rq = load("review_queue")
    st = rq["header"].index("Technical status")
    for i, r in enumerate(rq["rows"]):
        if r[st] not in R17_TECH_STATUS:
            problems.append(f"review_queue: row {i} status {r[st]!r} not in R17 set")
    fo = load("frozen_objects")
    hb, hs = fo["header"].index("Expected Bytes"), fo["header"].index("Expected SHA-256 / Identity")
    for i, r in enumerate(fo["rows"]):
        if hs < len(r) and HEX64.match(r[hs].strip()):
            b = r[hb].strip() if hb < len(r) else ""
            if not b.isdigit() or int(b) <= 0:
                problems.append(f"frozen_objects: row {i} ({r[0]}) has SHA but non-numeric byte count {b!r}")
    qi = load("quarantine_index")
    qc = qi["header"].index("Class")
    for i, r in enumerate(qi["rows"]):
        if r[qc] not in R17_QUARANTINE_CLASSES:
            problems.append(f"quarantine_index: row {i} class {r[qc]!r} not in R17 table")
    known = load_known()
    new = [p for p in problems if p not in known]
    for p in problems:
        print(("KNOWN  " if p in known else "NEW    ") + p)
    print(f"tabs={len(names)} problems={len(problems)} known={len(problems) - len(new)} new={len(new)}")
    return 1 if new else 0


if __name__ == "__main__":
    raise SystemExit(main())
