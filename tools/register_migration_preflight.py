#!/usr/bin/env python3
"""Report which checkers would refuse a candidate GP-REG-032 export, adopting nothing.

Landing a newer register export is not a file swap. The 44 tabs are coupled to
`engine/lanes/`, `reviews/records/` and the Drive source map, and several
checkers refuse an export whose cells their dependents are not ready for. Until
now, finding out which ones meant regenerating by hand into a scratch directory
and running each checker with the right redirect flag — a sequence that is easy
to get wrong in a way that reports a clean run.

This tool performs that sequence once, mechanically:

  1. regenerate the 44 tabs from ``--source`` into a temporary directory
     (``tools/registers_import.py``, whose output is the only thing read);
  2. build a sentinel-corrupted copy of those tabs, in which every non-empty
     string cell is replaced by one fixed token;
  3. for each covered checker, run it against the corrupted copy and against
     the candidate, and compare;
  4. report each checker's exit status and the lines it printed.

Step 3 is the point. ``tools/mirror_quotes_check.py`` accepts ``--register-json``
and then silently ignores any path outside the repository root: its ``_safe_path``
containment guard drops the register corpus without a diagnostic, so the run
reports twenty-four README quotations as unsourced and looks like a finding about
the candidate export. It is not one. A redirect flag that is accepted and not
honoured turns this whole tool into a machine for producing confident wrong
answers, so no checker is believed here until corrupting its input has been shown
to change what it prints. A checker whose output does not move is reported
``NOT_REDIRECTABLE`` and counted a problem, never a pass.

WHAT THIS DOES NOT ESTABLISH. Nothing here adopts an export, regenerates
``registers/json`` or ``registers/csv``, edits exported source data, or writes
anywhere inside the repository. A clean preflight says that five checkers do not
refuse the candidate's cells; it is not an import, not a migration, not a review,
and not a judgement that the candidate is a faithful export of anything — that
question is about bytes on the Drive and is settled elsewhere. It transcribes no
status, moves no gate, discharges no premise and awards zero organizational
independence credit. A checker this tool does not cover is not thereby known to
pass; the uncovered ones are named in the output.

Run: python3 tools/register_migration_preflight.py [--source PATH]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE = os.path.join(
    ROOT, "registers", "source", "GP-REG-032_v1.2_export_2026-09-23_R1.xlsx")

SENTINEL = "PREFLIGHT_SENTINEL"

# Checkers that read the register and accept a redirect to a directory of
# <tab>.json files. Each entry maps a register directory to that checker's argv
# tail. Every one is sensitivity-checked at run time before its verdict counts.
COVERED = (
    ("registers_check", lambda d: ["--json-dir", d]),
    ("reviews_check", lambda d: ["--queue", os.path.join(d, "review_queue.json")]),
    ("lanes_check", lambda d: ["--registers", d,
                               "--review-queue", os.path.join(d, "review_queue.json")]),
    ("frozen_check", lambda d: ["--frozen", os.path.join(d, "frozen_objects.json")]),
    ("quarantine_check", lambda d: ["--register", os.path.join(d, "quarantine_index.json")]),
)

# Register consumers deliberately left out, and why. Named in the output so a
# clean run cannot be read as "every checker passes".
UNCOVERED = (
    ("mirror_quotes_check", "resolves --register-json under --root and silently drops "
                            "an out-of-root path; a redirect here cannot be trusted"),
    ("consumers_check", "reads tab names and repository prose rather than register "
                        "cells, so its verdict does not gate an export swap"),
)


def run(argv: list[str]) -> tuple[int, str]:
    """Run a checker by its CLI, returning (exit status, combined output)."""
    proc = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def regenerate(source: str, out_dir: str) -> None:
    json_dir = os.path.join(out_dir, "json")
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(json_dir)
    os.makedirs(csv_dir)
    code, out = run([sys.executable, os.path.join(ROOT, "tools", "registers_import.py"),
                     "--source", source, "--out-json", json_dir, "--out-csv", csv_dir])
    if code != 0:
        raise SystemExit(f"register_migration_preflight: the importer refused "
                         f"{source!r} (exit {code}):\n{out}")


def corrupt(json_dir: str, out_dir: str) -> None:
    """A copy of the tabs with every non-empty string cell replaced by SENTINEL."""
    os.makedirs(out_dir)
    for path in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        for row in doc.get("rows", []):
            items = row.items() if isinstance(row, dict) else enumerate(row)
            for key, value in list(items):
                if isinstance(value, str) and value.strip():
                    row[key] = SENTINEL
        with open(os.path.join(out_dir, os.path.basename(path)), "w",
                  encoding="utf-8") as handle:
            json.dump(doc, handle)


def excerpt(out: str, max_lines: int, full: bool) -> list[str]:
    """The lines worth quoting from a checker's output, summary line included.

    A line the checker itself prefixes ``KNOWN`` is one its allowlist already
    carries, so it cannot be a reason the candidate is refused. Quoting those
    first is how a truncated excerpt comes to hide the problems that matter:
    ``registers_check`` prints thirty-seven allowlisted duplicate keys before
    the three review_queue statuses that are the whole point. They are ordered
    last here, and the final line — every checker in this repository ends with
    its summary — is always kept.
    """
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if full or len(lines) <= max_lines:
        return lines
    salient = [ln for ln in lines if not ln.lstrip().startswith("KNOWN")]
    known = len(lines) - len(salient)
    tail = lines[-1]
    head = [ln for ln in salient if ln != tail][:max_lines]
    kept = list(head)
    hidden = len(salient) - len(head) - (1 if tail in salient else 0)
    note = []
    if hidden > 0:
        note.append(f"{hidden} further line(s)")
    if known:
        note.append(f"{known} allowlisted KNOWN line(s)")
    if note:
        kept.append("... " + ", ".join(note) + " not shown; re-run with --full")
    kept.append(tail)
    return kept


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="the candidate .xlsx workbook export to preflight")
    ap.add_argument("--max-lines", type=int, default=6,
                    help="per-checker output lines to quote before summarising")
    ap.add_argument("--full", action="store_true",
                    help="quote every line each checker printed")
    args = ap.parse_args(argv)

    source = os.path.abspath(args.source)
    if not os.path.isfile(source):
        print(f"register_migration_preflight: no such export: {source}")
        return 2

    refusing = 0
    untrusted = 0
    with tempfile.TemporaryDirectory(prefix="reg-preflight-") as tmp:
        regenerate(source, tmp)
        json_dir = os.path.join(tmp, "json")
        bad_dir = os.path.join(tmp, "corrupt")
        corrupt(json_dir, bad_dir)
        tabs = len(glob.glob(os.path.join(json_dir, "*.json")))
        print(f"preflight of {os.path.relpath(source, ROOT)}: {tabs} tabs regenerated")

        for name, flags in COVERED:
            tool = os.path.join(ROOT, "tools", name + ".py")
            code, out = run([sys.executable, tool, *flags(json_dir)])
            _, bad_out = run([sys.executable, tool, *flags(bad_dir)])
            if out == bad_out:
                untrusted += 1
                print(f"  NOT_REDIRECTABLE  {name}: corrupting the register did not change "
                      f"what it printed, so this run says nothing about the candidate")
                continue
            verdict = "refuses" if code != 0 else "accepts"
            print(f"  {verdict:8s}          {name} (exit {code})")
            for line in excerpt(out, args.max_lines, args.full):
                print(f"      {line}")
            if code != 0:
                refusing += 1

    for name, why in UNCOVERED:
        print(f"  not covered       {name}: {why}")

    print(f"register_migration_preflight: covered={len(COVERED)} refusing={refusing} "
          f"not_redirectable={untrusted} uncovered={len(UNCOVERED)}")
    print("A preflight adopts nothing. It regenerates no committed register, edits no "
          "exported source, moves no gate and awards no independence credit.")
    return 1 if (refusing or untrusted) else 0


if __name__ == "__main__":
    raise SystemExit(main())
