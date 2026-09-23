#!/usr/bin/env python3
"""Convert the GP-REG-032 (Coupled Research Registers) workbook export into JSON
and CSV registers.

Source
------
The source is an ``.xlsx`` export of the live Google Sheet
``GP-REG-032-v1.2 — Coupled Research Registers`` (Drive id
``1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no``), taken 2026-09-18 and kept
under ``registers/source/`` next to the earlier 2026-09-17 markdown export. Both
exports are renderings of a native Sheet, not the object's own bytes;
``registers/source/SOURCES.json`` records their provenance. The markdown export
returned only a prefix of seven large tabs (file_catalog 310 of 2952 rows,
activity_log 241/518, artifact_index 206/761, transition_log 67/82,
review_ledger 101/138, evidence_lineage 137/485, relations 164/367); the xlsx
export is complete, which is why the importer now reads it.

The workbook is read with the standard library only (``zipfile`` and
``xml.etree.ElementTree``): ``xl/workbook.xml`` gives the sheet names and their
order, ``xl/_rels/workbook.xml.rels`` maps each sheet to its part,
``xl/sharedStrings.xml`` holds the shared string table, and each sheet's
``sheetData`` holds the cells. Nothing in the workbook is executed and no
formula is evaluated.

Sheet mapping
-------------
``SHEETS`` lists the 44 worksheet names in workbook order with the machine name
each becomes: the 42 tabs of the 2026-09-17 export in their original order,
plus ``reusable_operations`` and ``operation_trials`` at the positions they hold
in the workbook (sheet indexes 42 and 43). The importer fails closed (exit 2)
if the workbook's sheet count or any sheet name differs from that list, so a
renamed, inserted or removed tab cannot be imported under the wrong name.

Cell rendering (every cell is a string in the output, as before)
-------------
* shared strings (``t="s"``) and inline strings (``t="inlineStr"``): the text
  verbatim — no trimming, and newlines inside a cell are preserved (the
  markdown export could not carry a newline inside a table cell);
* booleans (``t="b"``): ``TRUE`` / ``FALSE``, as the spreadsheet displays them;
* formula string results (``t="str"``), errors (``t="e"``) and ISO dates
  (``t="d"``): the cached ``<v>`` text verbatim;
* numbers (``t="n"`` or no type): the ``<v>`` literal, rendered as a plain
  integer when it denotes an integer value (the export writes ``6874.0`` for a
  cell that displays ``6874``; the earlier markdown export rendered it ``6874``)
  and otherwise kept exactly as the cell holds it. In particular an
  ``activity_log`` "Modified UTC" cell that holds a spreadsheet date serial
  such as ``46223.95347222222`` is emitted as that string; it is not converted
  to a timestamp, and it is not rounded (the markdown export rounded it to five
  decimals);
* a formula cell contributes only its cached value; formulas are not evaluated;
* a cell with no ``<v>`` is the empty string.

Table shape (the empty-row / empty-cell rule)
-------------
* The table width ``W`` of a sheet is one more than the highest column index
  holding a non-empty value anywhere on the sheet.
* The header is the first row (in row order) that has any non-empty cell,
  padded with empty strings to ``W``; a header cell may therefore be ``""``
  when data exists in a column with no heading.
* Every later row is padded to exactly ``W`` cells (so trailing empty cells
  within the used width are kept and rows are rectangular, as before) and a
  row with no non-empty cell is dropped wherever it occurs, so row indexes in
  ``registers/KNOWN_FINDINGS.json`` count non-empty rows. Trailing all-empty
  rows and trailing all-empty columns of a sheet never appear in the output.
* A sheet with a header and no data rows (``operation_trials`` at this export)
  is written with ``rows: []``.

Output
------
``json/<tab>.json`` = ``{"tab", "sheet_index", "header", "rows"}`` and
``csv/<tab>.csv`` (header, then rows padded to the header width), unchanged in
shape from the markdown-era importer.

Cross-export diff (``--diff-exports``)
-------------
``registers/EXPORT_DIFF_2026-09-17_to_2026-09-18.json`` is the mechanical,
every-column, keyed diff between the two committed exports: the 2026-09-17
markdown rendering, read by ``parse_markdown_export`` (the markdown-era parser
retained verbatim for this one purpose; it never writes ``registers/json``),
and the xlsx export read by ``parse``. ``export_diff`` documents its own rules:
rows pair by key column and occurrence, every differing cell is listed, and
each is classified mechanically as a ``rendering_artifact`` of the markdown
export (escaping, five-decimal serials, merged cells, mojibake, date display),
an ``extension`` (the new value extends the old — an edit or a cut, which the
exports cannot tell apart) or a ``content_change``; a ``status_column`` flag
records whether the column header names a status, class, state, grade,
disposition or verdict. The diff decides nothing: a changed status word is the
register's word, transcribed, and tests/test_registers.py asserts that the
committed file equals the recomputation so the change list cannot be curated
by hand.

Run:  python3 tools/registers_import.py [--check] [--source PATH]
                                         [--out-json DIR] [--out-csv DIR]
                                         [--diff-exports [PATH]] [--old-source PATH]
``--check`` regenerates into a temp dir and fails (exit 1) if the committed
outputs differ from the regenerated ones — including a stale file in the output
directories that the source no longer produces — so CI catches hand edits that
drift from the source export. A source that does not match the sheet mapping
exits 2 in both modes. ``--diff-exports`` writes the cross-export diff (to
``registers/EXPORT_DIFF_2026-09-17_to_2026-09-18.json`` unless a path is given)
and writes nothing else.
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
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "registers", "source", "GP-REG-032_v1.2_export_2026-09-18.xlsx")
OUT_JSON = os.path.join(ROOT, "registers", "json")
OUT_CSV = os.path.join(ROOT, "registers", "csv")
MARKDOWN_SOURCE = os.path.join(ROOT, "registers", "source", "GP-REG-032_v1.2_export_2026-09-17.md")
EXPORT_DIFF = os.path.join(ROOT, "registers", "EXPORT_DIFF_2026-09-17_to_2026-09-18.json")

# Workbook sheet name -> stable machine name, in workbook order. The first 42
# are the tabs of the 2026-09-17 markdown export in their original order; the
# last two are the tabs the workbook gained on 2026-09-18, at the sheet
# positions they actually occupy.
SHEETS: list[tuple[str, str]] = [
    ("Start Here", "start_here"),                       # RESEARCH HOME · R17
    ("Review Queue", "review_queue"),                   # Review key
    ("File Catalog", "file_catalog"),                   # Drive ID (metadata snapshot)
    ("Quarantine Index", "quarantine_index"),           # Quarantine key
    ("Work Events", "work_events"),                     # Event ID (append-only coordination log)
    ("Dashboard", "research_state_dashboard"),          # PRIMARY FOCUS — VERIFIED RESEARCH PROGRESS
    ("Open Questions", "open_questions"),               # OQ ID
    ("Help Board", "help_board"),                       # Item ID
    ("Recent Activity", "activity_log"),                # Modified UTC / Artifact ID
    ("Artifact Index", "artifact_index"),               # Artifact ID
    ("Context Snapshot", "context_snapshot"),           # Snapshot ID
    ("Metadata Schema", "metadata_schema"),             # Field
    ("Config", "automation_config"),                    # Setting (GP-AUTO-034 config)
    ("Duplicate Flags", "duplicate_flags"),             # Cluster ID
    ("Run Log", "run_log"),                             # Run ID
    ("RETIRED — No-Vote History", "consensus_ballot_retired"),  # RETIRED — NO-VOTE MODEL
    ("Easy Closure Queue", "easy_closure_queue"),       # Candidate ID
    ("Closure Log", "closure_log"),                     # Closure ID
    ("Transition Register", "transition_log"),          # Transition ID
    ("No-Change Certificates", "no_change_certificates"),  # Certificate ID
    ("Review Independence", "review_ledger"),           # Review ID
    ("Evidence Lineage", "evidence_lineage"),           # Evidence ID
    ("Global Object Audit", "global_object_audit"),     # Audit ID
    ("Operator Decisions", "operator_decisions"),       # Decision ID
    ("Transition Alarms", "alarms"),                    # Severity
    ("Architecture Metrics", "architecture_metrics"),   # Metric
    ("Definition Registry", "definitions"),             # Definition ID
    ("Relation Index", "relations"),                    # Relation ID
    ("Autonomy Control", "autonomy_control"),           # Setting (CONTROL_PLANE_VERSION)
    ("Active Work Claims", "active_work_claims"),       # Claim ID
    ("Dispatch Queue", "dispatch_queue"),               # Rank / Dispatch ID
    ("Task Intake & Expansion", "task_intake"),         # Intake ID
    ("Cold Start Tests", "cold_start_tests"),           # Test ID (CST)
    ("Prompt Intent Tests", "prompt_intent_tests"),     # Test ID (PIT)
    ("P0.2 Review Manifest", "p02_exact_hash_review_manifest"),  # merged sheet
    ("Model Capabilities", "capability_records"),       # Capability Record ID
    ("Cross-Line Work Orders", "work_orders"),          # Work Order ID
    ("Frozen Objects", "frozen_objects"),               # Object ID / Drive ID / SHA-256
    ("Identity Drift Watch", "identity_drift_watch"),   # Watch ID
    ("Task Integrity Gates", "task_gates"),             # Gate ID
    ("Cold Start Control View", "cold_start_control_view"),  # Dispatch ID
    ("LPW Current State", "lpw_fold_dispositions"),     # Object / Exact scope
    ("Reusable Operations", "reusable_operations"),     # added to the workbook 2026-09-18
    ("Operation Trials", "operation_trials"),           # added to the workbook 2026-09-18
]
TAB_NAMES = [machine for _sheet, machine in SHEETS]

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_REL_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"m": NS_MAIN, "r": NS_REL_DOC, "pr": NS_REL_PKG}
T_TAG = "{%s}t" % NS_MAIN
RPH_TAG = "{%s}rPh" % NS_MAIN

CELL_REF = re.compile(r"^([A-Z]+)([0-9]+)$")


class SourceError(Exception):
    """The source does not match what this importer is built for (exit 2)."""


def column_index(ref: str) -> int | None:
    """'A' -> 0, 'Z' -> 25, 'AA' -> 26; None when the reference is unreadable."""
    m = CELL_REF.match(ref or "")
    if not m:
        return None
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def text_of(node: ET.Element) -> str:
    """Concatenate the <t> runs under a string item, skipping phonetic <rPh>."""
    parts: list[str] = []

    def walk(n: ET.Element) -> None:
        if n.tag == RPH_TAG:
            return
        if n.tag == T_TAG and n.text:
            parts.append(n.text)
        for child in n:
            walk(child)

    walk(node)
    return "".join(parts)


def render_number(literal: str) -> str:
    """Render a numeric <v> literal: integers plainly, everything else verbatim."""
    try:
        d = Decimal(literal)
    except InvalidOperation:
        return literal
    if d.is_finite() and d == d.to_integral_value():
        return str(int(d))
    return literal


def cell_value(c: ET.Element, shared: list[str]) -> str:
    t = c.get("t")
    v = c.find("m:v", NS)
    if t == "s":
        if v is None or v.text is None:
            return ""
        return shared[int(v.text)]
    if t == "inlineStr":
        is_ = c.find("m:is", NS)
        return text_of(is_) if is_ is not None else ""
    if t == "b":
        return "TRUE" if (v is not None and v.text == "1") else "FALSE"
    if t in ("str", "e", "d"):
        return v.text if (v is not None and v.text is not None) else ""
    if v is None or v.text is None:
        return ""
    return render_number(v.text)


def workbook_sheets(z: zipfile.ZipFile) -> list[tuple[str, str]]:
    """[(sheet name, part path)] in workbook order."""
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    targets: dict[str, str] = {}
    for rel in rels.findall("pr:Relationship", NS):
        target = rel.get("Target") or ""
        if target.startswith("/"):
            target = target[1:]
        elif not target.startswith("xl/"):
            target = "xl/" + target
        targets[rel.get("Id") or ""] = target
    sheets_el = wb.find("m:sheets", NS)
    if sheets_el is None:
        raise SourceError("xl/workbook.xml has no <sheets> element")
    out = []
    for s in sheets_el.findall("m:sheet", NS):
        rid = s.get("{%s}id" % NS_REL_DOC) or ""
        if rid not in targets:
            raise SourceError(f"sheet {s.get('name')!r} has no relationship target")
        out.append((s.get("name") or "", targets[rid]))
    return out


def shared_strings(z: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    return [text_of(si) for si in root.findall("m:si", NS)]


def sheet_table(z: zipfile.ZipFile, part: str, shared: list[str]) -> tuple[list[str], list[list[str]]]:
    """(header, rows) for one worksheet, under the rules in the module docstring."""
    ws = ET.fromstring(z.read(part))
    sd = ws.find("m:sheetData", NS)
    grid: dict[int, dict[int, str]] = {}
    max_col = -1
    if sd is not None:
        auto_row = 0
        for row in sd.findall("m:row", NS):
            r_attr = row.get("r")
            rn = int(r_attr) if r_attr and r_attr.isdigit() else auto_row + 1
            auto_row = rn
            next_col = 0
            for c in row.findall("m:c", NS):
                ci = column_index(c.get("r") or "")
                if ci is None:
                    ci = next_col
                next_col = ci + 1
                val = cell_value(c, shared)
                if val != "":
                    grid.setdefault(rn, {})[ci] = val
                    if ci > max_col:
                        max_col = ci
    width = max_col + 1
    ordered = sorted(grid)          # only rows with at least one non-empty cell
    if not ordered:
        return [], []
    def padded(rn: int) -> list[str]:
        cells = grid[rn]
        return [cells.get(i, "") for i in range(width)]
    header = padded(ordered[0])
    rows = [padded(rn) for rn in ordered[1:]]
    return header, rows


def parse(source_path: str) -> list[dict]:
    """Read the workbook and return the 44 tabs; raise SourceError if it is not
    the workbook this importer is built for."""
    if not os.path.isfile(source_path):
        raise SourceError(f"source not found: {source_path}")
    try:
        z = zipfile.ZipFile(source_path)
    except zipfile.BadZipFile as exc:
        raise SourceError(f"source is not a zip/xlsx container: {exc}")
    with z:
        try:
            sheets = workbook_sheets(z)
        except KeyError as exc:
            raise SourceError(f"workbook part missing: {exc}")
        expected = [name for name, _machine in SHEETS]
        found = [name for name, _part in sheets]
        if len(found) != len(expected):
            raise SourceError(
                f"expected {len(expected)} worksheets, workbook has {len(found)}: {found}")
        for i, (want, got) in enumerate(zip(expected, found)):
            if want != got:
                raise SourceError(
                    f"worksheet {i} is named {got!r}; the sheet mapping expects {want!r}. "
                    f"Update SHEETS deliberately rather than importing under the wrong name.")
        shared = shared_strings(z)
        tabs = []
        for n, ((name, part), (_sheet, machine)) in enumerate(zip(sheets, SHEETS)):
            header, rows = sheet_table(z, part, shared)
            tabs.append({"tab": machine, "sheet_index": n, "header": header, "rows": rows})
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


def check(tabs: list[dict], out_json: str, out_csv: str) -> list[str]:
    """Regenerate into a temp dir and list every committed file that differs,
    is missing, or is present in the output dirs without being produced."""
    tmp = tempfile.mkdtemp()
    try:
        write(tabs, os.path.join(tmp, "json"), os.path.join(tmp, "csv"))
        bad = []
        for sub, target in (("json", out_json), ("csv", out_csv)):
            produced = set(os.listdir(os.path.join(tmp, sub)))
            for fn in sorted(produced):
                a = os.path.join(tmp, sub, fn)
                b = os.path.join(target, fn)
                if not os.path.exists(b) or not filecmp.cmp(a, b, shallow=False):
                    bad.append(f"{sub}/{fn}")
            if os.path.isdir(target):
                for fn in sorted(os.listdir(target)):
                    if fn.endswith(f".{sub}") and fn not in produced:
                        bad.append(f"{sub}/{fn} (stale: not produced by the source)")
        return bad
    finally:
        shutil.rmtree(tmp)

# ---------------------------------------------------------------------------
# Cross-export diff: the 2026-09-17 markdown rendering vs the xlsx export
# ---------------------------------------------------------------------------

# The markdown-era parser, retained verbatim from the importer that produced
# registers/json from the 2026-09-17 export. It exists so that the diff between
# the two exports is recomputable from the two committed files alone; it is
# never used to write registers/json or registers/csv.
MARKDOWN_TAB_NAMES = TAB_NAMES[:42]
_MD_SEP_RE = re.compile(r"^\|(\s*:-:\s*\|)+\s*$")
_MD_UNESCAPE = [("\\_", "_"), ("\\!", "!"), ("\\#", "#"), ("\\[", "["), ("\\]", "]"),
                ("\\>", ">"), ("\\<", "<"), ("\\-", "-"), ("\\*", "*"), ("\\|", "|")]


def _md_cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    out = []
    for c in line.split(" | "):
        c = c.strip()
        for a, b in _MD_UNESCAPE:
            c = c.replace(a, b)
        out.append(c)
    return out


def parse_markdown_export(source_path: str) -> list[dict]:
    """Tabs of the 2026-09-17 markdown export, exactly as the markdown-era
    importer read them (same separator rule, same unescape table, same
    all-empty-row drop). Seven of its tabs are prefixes of the workbook."""
    with open(source_path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    seps = [i for i, l in enumerate(lines) if _MD_SEP_RE.match(l)]
    seps.append(len(lines) + 1)
    tabs = []
    for n, (a, b) in enumerate(zip(seps[:-1], seps[1:])):
        header = _md_cells(lines[a + 1])
        rows = [_md_cells(l) for l in lines[a + 2:b - 1] if l.strip().startswith("|")]
        rows = [r for r in rows if any(r)]
        name = MARKDOWN_TAB_NAMES[n] if n < len(MARKDOWN_TAB_NAMES) else f"tab{n:02d}"
        tabs.append({"tab": name, "sheet_index": n, "header": header, "rows": rows})
    return tabs


# Column that identifies a row for pairing: the key-bearing tabs of
# tools/registers_check.py, plus activity_log, whose first column (a modified
# time) repeats and whose second (Artifact ID) pairs every 2026-09-17 row.
DIFF_KEY_COLUMNS = {
    "review_queue": 0, "frozen_objects": 0, "closure_log": 0, "work_events": 0,
    "quarantine_index": 0, "dispatch_queue": 1, "artifact_index": 0,
    "evidence_lineage": 0, "transition_log": 0, "operator_decisions": 0,
    "reusable_operations": 0, "operation_trials": 0, "activity_log": 1,
}
_MERGED = "[merged] "
_STATUS_WORDS = {"status", "state", "class", "grade", "disposition", "verdict"}
_NOT_STATUS_WORDS = {"utc", "artifact", "why"}
_STATUS_HEADERS = {"Aging action"}
_SHEET_EPOCH = datetime(1899, 12, 30)
_MD_DATE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")

DIFF_RULES = {
    "pairing": "Rows of a tab pair by the value of the tab's key column (DIFF_KEY_COLUMNS; column 0 "
               "otherwise) and by occurrence, so the i-th row carrying a key pairs with the i-th row "
               "carrying it in the other export; a leading '[merged] ' (the markdown rendering of a "
               "merged cell) is ignored for pairing. Unpaired rows are listed as removed or added.",
    "rendering_artifact": "old and new differ only by one of: a markdown backslash escape before a "
                          "punctuation character; a numeric literal the markdown rounded to five "
                          "decimals; a merged cell the markdown repeated across the merge with a "
                          "'[merged] ' prefix; UTF-8 text the markdown rendered as Latin-1 mojibake; "
                          "or a date serial the markdown displayed as 'YYYY-MM-DD HH:MM'.",
    "extension": "old is non-empty and new begins with old: an edit that appended text, or a cell the "
                 "markdown rendering cut — the two exports cannot tell these apart.",
    "content_change": "every other differing cell, including a cell that was empty and is now filled.",
    "status_column": "the column header, split into words, contains status, state, class, grade, "
                     "disposition or verdict, contains none of utc, artifact or why, has at most four "
                     "words, or is exactly 'Aging action'. This is a rule on header names, not a "
                     "judgement about the cell.",
}


def status_column(header: str) -> bool:
    if header in _STATUS_HEADERS:
        return True
    words = re.findall(r"[a-z]+", header.lower())
    return (len(words) <= 4 and bool(set(words) & _STATUS_WORDS)
            and not (set(words) & _NOT_STATUS_WORDS))


def rendering_artifact(old: str, new: str) -> str | None:
    """The name of the markdown-rendering rule under which `old` renders `new`,
    or None when the two are not rendering-equivalent."""
    if re.sub(r"\\([^A-Za-z0-9\s])", r"\1", old) == new and old != new:
        return "markdown_escape"
    try:
        do, dn = Decimal(old), Decimal(new)
        if do.is_finite() and dn.is_finite() and dn.quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP) == do \
                and str(do) == old and old != new:
            return "five_decimal_serial"
    except InvalidOperation:
        pass
    if old.startswith(_MERGED) and new in ("", old[len(_MERGED):]):
        return "merged_cell"
    try:
        if old.encode("latin-1").decode("utf-8") == new and old != new:
            return "latin1_mojibake"
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    if _MD_DATE.match(old):
        try:
            serial = float(new)
            shown = (_SHEET_EPOCH + timedelta(days=serial)).strftime("%Y-%m-%d %H:%M")
            if shown == old:
                return "date_serial_display"
        except (ValueError, OverflowError):
            pass
    return None


def classify_cell(old: str, new: str) -> tuple[str, str | None]:
    rule = rendering_artifact(old, new)
    if rule:
        return "rendering_artifact", rule
    if old and new.startswith(old):
        return "extension", None
    return "content_change", None


def _pair_key(row: list[str], col: int) -> str:
    key = row[col] if col < len(row) else ""
    return key[len(_MERGED):] if key.startswith(_MERGED) else key


def _occurrences(rows: list[list[str]], col: int) -> list[tuple[str, int]]:
    seen: dict[str, int] = {}
    out = []
    for r in rows:
        k = _pair_key(r, col)
        i = seen.get(k, 0)
        seen[k] = i + 1
        out.append((k, i))
    return out


def export_diff(old_tabs: list[dict], new_tabs: list[dict]) -> dict:
    """Every difference between two parsed exports, under DIFF_RULES."""
    old_by = {t["tab"]: t for t in old_tabs}
    new_by = {t["tab"]: t for t in new_tabs}
    changed_cells: list[dict] = []
    header_changes: list[dict] = []
    rows_added: dict[str, list[str]] = {}
    rows_removed: dict[str, list[dict]] = {}
    identical: list[str] = []
    for tab in [t["tab"] for t in new_tabs if t["tab"] in old_by]:
        o, n = old_by[tab], new_by[tab]
        col = DIFF_KEY_COLUMNS.get(tab, 0)
        header = n["header"]
        before = len(changed_cells) + len(header_changes)
        if o["header"] != n["header"]:
            w = max(len(o["header"]), len(n["header"]))
            kinds = set()
            for c in range(w):
                x = o["header"][c] if c < len(o["header"]) else ""
                y = n["header"][c] if c < len(n["header"]) else ""
                if x != y:
                    kinds.add(classify_cell(x, y)[0])
            header_changes.append({"tab": tab, "old": o["header"], "new": n["header"],
                                   "kind": "rendering_artifact" if kinds == {"rendering_artifact"} else "content_change"})
        oo, nn = _occurrences(o["rows"], col), _occurrences(n["rows"], col)
        nmap = {t: i for i, t in enumerate(nn)}
        paired: set[int] = set()
        for i, t in enumerate(oo):
            if t not in nmap:
                rows_removed.setdefault(tab, []).append({"key": t[0], "occurrence": t[1], "row": o["rows"][i]})
                continue
            j = nmap[t]
            paired.add(j)
            a, b = o["rows"][i], n["rows"][j]
            for c in range(max(len(a), len(b))):
                x = a[c] if c < len(a) else ""
                y = b[c] if c < len(b) else ""
                if x == y:
                    continue
                kind, rule = classify_cell(x, y)
                column = header[c] if c < len(header) else f"column {c}"
                entry = {"tab": tab, "key": t[0], "occurrence": t[1], "column": column,
                         "old": x, "new": y, "kind": kind, "status_column": status_column(column)}
                if rule:
                    entry["rule"] = rule
                changed_cells.append(entry)
        added = [nn[j][0] for j in range(len(nn)) if j not in paired]
        if added:
            rows_added[tab] = added
        if len(changed_cells) + len(header_changes) == before and not added and tab not in rows_removed:
            identical.append(tab)
    new_tabs_only = {t["tab"]: len(t["rows"]) for t in new_tabs if t["tab"] not in old_by}
    status_word_changes = [
        {k: e[k] for k in ("tab", "key", "occurrence", "column", "old", "new")}
        for e in changed_cells if e["kind"] == "content_change" and e["status_column"]]
    by_kind: dict[str, int] = {}
    for e in changed_cells:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
    return {
        "rules": DIFF_RULES,
        "summary": {
            "tabs_in_both_exports": len([t for t in new_tabs if t["tab"] in old_by]),
            "tabs_only_in_new_export": new_tabs_only,
            "tabs_identical": identical,
            "changed_cells_by_kind": dict(sorted(by_kind.items())),
            "status_word_changes": len(status_word_changes),
            "header_changes": len(header_changes),
            "rows_added": {t: len(v) for t, v in rows_added.items()},
            "rows_removed": {t: len(v) for t, v in rows_removed.items()},
        },
        "status_word_changes": status_word_changes,
        "changed_cells": changed_cells,
        "header_changes": header_changes,
        "rows_added": rows_added,
        "rows_removed": rows_removed,
    }


def _file_identity(path: str) -> dict:
    import hashlib
    with open(path, "rb") as f:
        data = f.read()
    return {"file": os.path.relpath(path, ROOT), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def export_diff_document(old_source: str, new_source: str) -> dict:
    """The committed diff document: both exports' identities, then export_diff."""
    old_tabs = parse_markdown_export(old_source)
    new_tabs = parse(new_source)
    doc = {
        "_comment": "MECHANICAL DIFF between the two committed exports of GP-REG-032-v1.2, generated by "
                    "python3 tools/registers_import.py --diff-exports and checked against a recomputation "
                    "by tests/test_registers.py. Every differing cell of every tab present in both exports "
                    "is listed under changed_cells, classified only by the rules under 'rules'; "
                    "status_word_changes is the subset that is a content change in a status column. "
                    "Nothing here decides, promotes, closes or reclassifies anything: a changed word is the "
                    "register's word, transcribed. A PASS_TECHNICAL is a same-line technical pass at zero "
                    "organizational independence credit; a register row is a transcription, not a verdict.",
        "old_export": dict(_file_identity(old_source), tabs=len(old_tabs)),
        "new_export": dict(_file_identity(new_source), tabs=len(new_tabs)),
    }
    doc.update(export_diff(old_tabs, new_tabs))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="regenerate to a temp dir and fail if the committed outputs differ")
    ap.add_argument("--source", default=SOURCE, help="the .xlsx workbook export to read")
    ap.add_argument("--out-json", default=OUT_JSON)
    ap.add_argument("--out-csv", default=OUT_CSV)
    ap.add_argument("--diff-exports", nargs="?", const=EXPORT_DIFF, default=None, metavar="PATH",
                    help="write the mechanical diff of --old-source (markdown) against --source (xlsx) "
                         "to PATH (default registers/EXPORT_DIFF_2026-09-17_to_2026-09-18.json) and exit")
    ap.add_argument("--old-source", default=MARKDOWN_SOURCE,
                    help="the 2026-09-17 markdown export, the old side of --diff-exports")
    args = ap.parse_args(argv)
    try:
        tabs = parse(args.source)
    except SourceError as exc:
        print(f"registers_import: refusing the source: {exc}", file=sys.stderr)
        return 2
    if args.diff_exports:
        if not os.path.isfile(args.old_source):
            print(f"registers_import: old export not found: {args.old_source}", file=sys.stderr)
            return 2
        doc = export_diff_document(args.old_source, args.source)
        with open(args.diff_exports, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
            f.write("\n")
        sm = doc["summary"]
        print(f"wrote {args.diff_exports}: {sm['tabs_in_both_exports']} tabs compared, "
              f"changed cells {sm['changed_cells_by_kind']}, status word changes {sm['status_word_changes']}, "
              f"rows added {sum(sm['rows_added'].values())}, rows removed {sum(sm['rows_removed'].values())}, "
              f"new tabs {sm['tabs_only_in_new_export']}")
        return 0
    if len(tabs) != len(TAB_NAMES):
        print(f"expected {len(TAB_NAMES)} tabs, parsed {len(tabs)}", file=sys.stderr)
        return 2
    if not args.check:
        write(tabs, args.out_json, args.out_csv)
        print(f"wrote {len(tabs)} tabs to {args.out_json} and {args.out_csv}")
        return 0
    bad = check(tabs, args.out_json, args.out_csv)
    if bad:
        print("register outputs drift from source export:", *bad, sep="\n  ", file=sys.stderr)
        return 1
    print(f"registers match source export ({len(tabs)} tabs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
