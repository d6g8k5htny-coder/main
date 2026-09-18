#!/usr/bin/env python3
"""Convert the GP-REG-032 (Coupled Research Registers) Google Sheets export into
JSON and CSV registers.

The source is the markdown-table export of the whole workbook produced by the
Google Drive connector on 2026-09-17 (registers/source/*.md).  Every tab becomes
one JSON file ({"tab": ..., "header": [...], "rows": [[...], ...]}) and one CSV.

Run:  python3 tools/registers_import.py [--check]
--check re-generates into a temp dir and fails if the committed outputs differ,
so CI catches hand edits that drift from the source export.
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "registers", "source", "GP-REG-032_v1.2_export_2026-09-17.md")
OUT_JSON = os.path.join(ROOT, "registers", "json")
OUT_CSV = os.path.join(ROOT, "registers", "csv")

# Tab order as exported (sheet order in the workbook) -> stable machine name.
TAB_NAMES = [
    "start_here",                 # RESEARCH HOME · R17
    "review_queue",               # Review key
    "file_catalog",               # Drive ID (metadata snapshot; 310 rows in export)
    "quarantine_index",           # Quarantine key
    "work_events",                # Event ID (append-only coordination log)
    "research_state_dashboard",   # PRIMARY FOCUS — VERIFIED RESEARCH PROGRESS
    "open_questions",             # OQ ID
    "help_board",                 # Item ID
    "activity_log",               # Modified UTC / Artifact ID
    "artifact_index",             # Artifact ID
    "context_snapshot",           # Snapshot ID
    "metadata_schema",            # Field
    "automation_config",          # Setting (GP-AUTO-034 config)
    "duplicate_flags",            # Cluster ID
    "run_log",                    # Run ID
    "consensus_ballot_retired",   # RETIRED — NO-VOTE MODEL
    "easy_closure_queue",         # Candidate ID
    "closure_log",                # Closure ID
    "transition_log",             # Transition ID
    "no_change_certificates",     # Certificate ID
    "review_ledger",              # Review ID
    "evidence_lineage",           # Evidence ID
    "global_object_audit",        # Audit ID
    "operator_decisions",         # Decision ID
    "alarms",                     # Severity
    "architecture_metrics",       # Metric
    "definitions",                # Definition ID
    "relations",                  # Relation ID
    "autonomy_control",           # Setting (CONTROL_PLANE_VERSION)
    "active_work_claims",         # Claim ID
    "dispatch_queue",             # Rank / Dispatch ID
    "task_intake",                # Intake ID
    "cold_start_tests",           # Test ID (CST)
    "prompt_intent_tests",        # Test ID (PIT)
    "p02_exact_hash_review_manifest",  # merged sheet
    "capability_records",         # Capability Record ID
    "work_orders",                # Work Order ID
    "frozen_objects",             # Object ID / Drive ID / SHA-256
    "identity_drift_watch",       # Watch ID
    "task_gates",                 # Gate ID
    "cold_start_control_view",    # Dispatch ID
    "lpw_fold_dispositions",      # Object / Exact scope
]

SEP_RE = re.compile(r"^\|(\s*:-:\s*\|)+\s*$")
UNESCAPE = [("\\_", "_"), ("\\!", "!"), ("\\#", "#"), ("\\[", "["), ("\\]", "]"),
            ("\\>", ">"), ("\\<", "<"), ("\\-", "-"), ("\\*", "*"), ("\\|", "|")]


def cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    out = []
    for c in line.split(" | "):
        c = c.strip()
        for a, b in UNESCAPE:
            c = c.replace(a, b)
        out.append(c)
    return out


def parse(source_path: str) -> list[dict]:
    lines = open(source_path, encoding="utf-8").read().split("\n")
    seps = [i for i, l in enumerate(lines) if SEP_RE.match(l)]
    seps.append(len(lines) + 1)
    tabs = []
    for n, (a, b) in enumerate(zip(seps[:-1], seps[1:])):
        header = cells(lines[a + 1])
        rows = [cells(l) for l in lines[a + 2:b - 1] if l.strip().startswith("|")]
        rows = [r for r in rows if any(r)]
        name = TAB_NAMES[n] if n < len(TAB_NAMES) else f"tab{n:02d}"
        tabs.append({"tab": name, "sheet_index": n, "header": header, "rows": rows})
    return tabs


def write(tabs: list[dict], out_json: str, out_csv: str) -> None:
    os.makedirs(out_json, exist_ok=True)
    os.makedirs(out_csv, exist_ok=True)
    for t in tabs:
        with open(os.path.join(out_json, f"{t['tab']}.json"), "w", encoding="utf-8") as f:
            json.dump(t, f, ensure_ascii=False, indent=1)
            f.write("\n")
        with open(os.path.join(out_csv, f"{t['tab']}.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(t["header"])
            for r in t["rows"]:
                w.writerow(r + [""] * (len(t["header"]) - len(r)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    tabs = parse(SOURCE)
    if len(tabs) != len(TAB_NAMES):
        print(f"expected {len(TAB_NAMES)} tabs, parsed {len(tabs)}", file=sys.stderr)
        return 2
    if not args.check:
        write(tabs, OUT_JSON, OUT_CSV)
        print(f"wrote {len(tabs)} tabs to {OUT_JSON} and {OUT_CSV}")
        return 0
    tmp = tempfile.mkdtemp()
    try:
        write(tabs, os.path.join(tmp, "json"), os.path.join(tmp, "csv"))
        bad = []
        for sub in ("json", "csv"):
            for fn in os.listdir(os.path.join(tmp, sub)):
                a = os.path.join(tmp, sub, fn)
                b = os.path.join(ROOT, "registers", sub, fn)
                if not os.path.exists(b) or not filecmp.cmp(a, b, shallow=False):
                    bad.append(f"{sub}/{fn}")
        if bad:
            print("register outputs drift from source export:", *bad, sep="\n  ", file=sys.stderr)
            return 1
        print("registers match source export")
        return 0
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    raise SystemExit(main())
