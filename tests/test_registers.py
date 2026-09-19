"""Register invariants, run in CI, with negative controls.

Positive side:

* the JSON/CSV registers must be exactly what tools/registers_import.py produces
  from the committed source export (no hand edits that drift from the source);
* structural checks in tools/registers_check.py must pass modulo the documented
  findings allowlist (registers/KNOWN_FINDINGS.json), and every allowlisted
  finding in EVERY section the checker reads must still fire (a repaired defect
  leaves the allowlist; the check covers the findings the 2026-09-18 export
  first delivered, not only the original section);
* registers/EXPORT_DIFF_2026-09-17_to_2026-09-18.json, the every-column diff
  between the two exports, must equal what tools/registers_import.py
  --diff-exports recomputes from the two committed export files, so the list
  of what moved between the exports is mechanical and cannot be curated;
* work_events is append-only: rows present in the previous commit must still be
  present, in the same order, at the same positions — and the five OPS4 rows
  the register was first imported with stay at positions 0..4 verbatim;
* the pinned facts below are the refreshed truth of the 2026-09-18 xlsx export
  (registers/source/GP-REG-032_v1.2_export_2026-09-18.xlsx), computed by running
  tools/registers_import.py on it: 44 tabs, the per-tab row counts, the three
  review routes whose technical status the refresh moved, the five RN5
  quarantine rows and the seven RN5 frozen objects.

Negative controls (the deliverable, per CLAUDE.md): each one breaks an input in
exactly one way on a temporary copy and asserts that the importer or a checker
refuses. Nothing under registers/, engine/ or quarantine/ is modified by these
tests, and a final control asserts that the importer never writes its source.

The controls for tools/quarantine_check.py invariant 5 (a bound member named by
an exclusion must carry the exclusion on its binding record) live here because
that invariant exists for the register's own quarantine rows and no
tests/test_quarantine.py existed when it was added.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = os.path.join(ROOT, "registers", "json")
CSV_DIR = os.path.join(ROOT, "registers", "csv")
SOURCE = os.path.join(ROOT, "registers", "source", "GP-REG-032_v1.2_export_2026-09-18.xlsx")
OLD_SOURCE = os.path.join(ROOT, "registers", "source", "GP-REG-032_v1.2_export_2026-09-17.md")
SOURCES = os.path.join(ROOT, "registers", "source", "SOURCES.json")
KNOWN = os.path.join(ROOT, "registers", "KNOWN_FINDINGS.json")
EXPORT_DIFF = os.path.join(ROOT, "registers", "EXPORT_DIFF_2026-09-17_to_2026-09-18.json")
IMPORTER = os.path.join(ROOT, "tools", "registers_import.py")
CHECKER = os.path.join(ROOT, "tools", "registers_check.py")
QCHECK = os.path.join(ROOT, "tools", "quarantine_check.py")
BINDING = os.path.join(ROOT, "engine", "rn_engine", "BINDING.json")
EXCLUSIONS = os.path.join(ROOT, "quarantine", "EXCLUSIONS.json")

sys.path.insert(0, os.path.join(ROOT, "tools"))
import registers_check as RC  # noqa: E402
import registers_import as RI  # noqa: E402

# ---------------------------------------------------------------------------
# pinned truth of the 2026-09-18 export (python3 tools/registers_import.py)
# ---------------------------------------------------------------------------

SOURCE_SHA256 = "c3229ecefc642f3e32f23cb8320fb66236d11affb83f67bd135153d068a3f460"
SOURCE_BYTES = 1919741
OLD_SOURCE_SHA256 = "57d9078b9b7dbf285713e4fb50b8143d9cf4322bf486a9296c7becdd5edf8b71"
OLD_SOURCE_BYTES = 2209031

ROW_COUNTS = {
    "start_here": 29, "review_queue": 25, "file_catalog": 2952, "quarantine_index": 22,
    "work_events": 35, "research_state_dashboard": 31, "open_questions": 20, "help_board": 84,
    "activity_log": 518, "artifact_index": 761, "context_snapshot": 15, "metadata_schema": 72,
    "automation_config": 249, "duplicate_flags": 31, "run_log": 32, "consensus_ballot_retired": 2,
    "easy_closure_queue": 32, "closure_log": 34, "transition_log": 82, "no_change_certificates": 5,
    "review_ledger": 138, "evidence_lineage": 485, "global_object_audit": 29,
    "operator_decisions": 45, "alarms": 182, "architecture_metrics": 73, "definitions": 61,
    "relations": 367, "autonomy_control": 54, "active_work_claims": 71, "dispatch_queue": 101,
    "task_intake": 34, "cold_start_tests": 78, "prompt_intent_tests": 108,
    "p02_exact_hash_review_manifest": 15, "capability_records": 4, "work_orders": 18,
    "frozen_objects": 195, "identity_drift_watch": 169, "task_gates": 44,
    "cold_start_control_view": 43, "lpw_fold_dispositions": 10,
    "reusable_operations": 15, "operation_trials": 0,
}
# rows the 2026-09-17 markdown rendering delivered for the seven tabs it cut
MARKDOWN_PREFIX = {
    "file_catalog": 310, "activity_log": 241, "artifact_index": 206, "transition_log": 67,
    "review_ledger": 101, "evidence_lineage": 137, "relations": 164,
}
# rows of the 42 tabs as the markdown-era parser reads the 2026-09-17 export
# (python3 -c "import registers_import as RI; ..."): the seven cut tabs at their
# prefixes, everything else at the count the workbook then had
OLD_ROW_COUNTS = dict(ROW_COUNTS, **MARKDOWN_PREFIX, start_here=23, review_queue=24,
                      quarantine_index=17, work_events=5, frozen_objects=188,
                      identity_drift_watch=162)
for _new_tab in ("reusable_operations", "operation_trials"):
    del OLD_ROW_COUNTS[_new_tab]
# the status-word changes between the two exports, as the mechanical diff lists
# them (content changes in status columns): the register's words, transcribed
STATUS_WORD_CHANGES = [
    ("review_queue", "RV-LM004-MAIN", "Technical status", "NEEDS_RECONCILIATION", "PASS_TECHNICAL"),
    ("review_queue", "RV-LM004-MAIN", "Aging action", "ESCALATE", "EXTERNAL ONLY"),
    ("review_queue", "RV-RN-ALIGN", "Technical status", "NEEDS_RECONCILIATION", "AMEND"),
    ("review_queue", "RV-RN-ALIGN", "Independence status", "NO_CREDIT_ASSIGNED", "AUTHOR_SIDE / ZERO ORG CREDIT"),
]
# the five rows registers/json/work_events.json was first imported with
WORK_EVENTS_BASELINE = [
    "OPS4-INSTALL-20260917", "OPS4-LM009-REVIEW-20260917", "OPS4-PUBLISH-R17-20260917",
    "OPS4-RELEASE-R17-20260917", "OPS4-CUSTODY-R17-20260917",
]
RN5_QUARANTINE = ["Q-RN5-MOMENT-001", "Q-RN5-MOMENT-002", "Q-RN5-MOMENT-003",
                  "Q-RN5-MOMENT-004", "Q-RN5-LM004-A1"]
RN5_FROZEN = ["RN5-PROOF", "RN5-LM004-ERRATUM", "RN5-SCOPE-HOLDS", "RN5-MANIFEST",
              "RN5-BUNDLE", "RN5-CUSTODY", "RN5-OPS-EVIDENCE"]


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)


def load(tab):
    with open(os.path.join(JSON_DIR, f"{tab}.json"), encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# the repository as it stands
# ---------------------------------------------------------------------------

def test_registers_match_source_export():
    r = run("tools/registers_import.py", "--check")
    assert r.returncode == 0, r.stdout + r.stderr


def test_structural_checks_pass_modulo_known_findings():
    r = run("tools/registers_check.py")
    assert r.returncode == 0, r.stdout + r.stderr


def test_source_exports_are_the_recorded_bytes():
    """Both exports are on disk, unchanged, and SOURCES.json records them."""
    assert sha256(SOURCE) == SOURCE_SHA256 and os.path.getsize(SOURCE) == SOURCE_BYTES
    assert sha256(OLD_SOURCE) == OLD_SOURCE_SHA256 and os.path.getsize(OLD_SOURCE) == OLD_SOURCE_BYTES
    with open(SOURCES, encoding="utf-8") as f:
        src = json.load(f)
    by_file = {e["file"]: e for e in src["exports"]}
    assert src["current_import_source"] == os.path.basename(SOURCE)
    for path, digest, nbytes in ((SOURCE, SOURCE_SHA256, SOURCE_BYTES),
                                 (OLD_SOURCE, OLD_SOURCE_SHA256, OLD_SOURCE_BYTES)):
        e = by_file[os.path.basename(path)]
        assert e["sha256"] == digest and e["bytes"] == nbytes
        assert e["exact"] is False, "an export of a native Sheet is a rendering, never exact"
    assert by_file[os.path.basename(SOURCE)]["modified_time_at_export"] == "2026-09-18T16:45:35.319Z"
    trunc = by_file[os.path.basename(OLD_SOURCE)]["truncated_tabs"]
    assert {k: v["prefix_rows"] for k, v in trunc.items()} == MARKDOWN_PREFIX
    assert {k: v["total_rows"] for k, v in trunc.items()} == {k: ROW_COUNTS[k] for k in MARKDOWN_PREFIX}


def test_importer_reads_the_xlsx_by_default_and_fails_closed_on_the_mapping():
    assert RI.SOURCE == SOURCE
    assert len(RI.SHEETS) == 44
    assert RI.TAB_NAMES[42:] == ["reusable_operations", "operation_trials"]
    assert RI.TAB_NAMES[:42] == [
        "start_here", "review_queue", "file_catalog", "quarantine_index", "work_events",
        "research_state_dashboard", "open_questions", "help_board", "activity_log",
        "artifact_index", "context_snapshot", "metadata_schema", "automation_config",
        "duplicate_flags", "run_log", "consensus_ballot_retired", "easy_closure_queue",
        "closure_log", "transition_log", "no_change_certificates", "review_ledger",
        "evidence_lineage", "global_object_audit", "operator_decisions", "alarms",
        "architecture_metrics", "definitions", "relations", "autonomy_control",
        "active_work_claims", "dispatch_queue", "task_intake", "cold_start_tests",
        "prompt_intent_tests", "p02_exact_hash_review_manifest", "capability_records",
        "work_orders", "frozen_objects", "identity_drift_watch", "task_gates",
        "cold_start_control_view", "lpw_fold_dispositions"]


def test_forty_four_tabs_with_the_refreshed_row_counts():
    names = sorted(fn[:-5] for fn in os.listdir(JSON_DIR) if fn.endswith(".json"))
    assert names == sorted(ROW_COUNTS)
    assert sorted(fn[:-4] for fn in os.listdir(CSV_DIR) if fn.endswith(".csv")) == names
    for tab, n in ROW_COUNTS.items():
        t = load(tab)
        assert len(t["rows"]) == n, tab
        assert t["sheet_index"] == RI.TAB_NAMES.index(tab)
        assert t["header"], tab
        assert all(len(r) == len(t["header"]) for r in t["rows"]), f"{tab}: rows are rectangular"
    for tab, prefix in MARKDOWN_PREFIX.items():
        assert ROW_COUNTS[tab] > prefix, f"{tab} was a prefix under the markdown export"


def test_work_events_append_only():
    cur = load("work_events")
    assert [r[0] for r in cur["rows"][:5]] == WORK_EVENTS_BASELINE
    prev = subprocess.run(["git", "show", "HEAD:registers/json/work_events.json"], cwd=ROOT,
                          capture_output=True, text=True)
    if prev.returncode != 0:
        return  # first commit of the register: nothing to compare against
    old = json.loads(prev.stdout)
    assert old["header"] == cur["header"], "work_events header changed"
    assert len(cur["rows"]) >= len(old["rows"]), "work_events lost rows"
    for i, row in enumerate(old["rows"]):
        assert cur["rows"][i] == row, f"work_events row {i} was rewritten (append-only violation)"


def test_review_queue_transcription_of_the_refresh():
    """The three status moves the refresh carried, in the register's own words."""
    rq = load("review_queue")
    h = rq["header"]
    rows = {r[h.index("Review key")]: r for r in rq["rows"]}
    st, ind, rev = h.index("Technical status"), h.index("Independence status"), h.index("Reviewer / claim")
    assert len(rows) == 25
    assert sum(1 for r in rows.values() if r[rev] == "UNASSIGNED") == 22
    assert rows["RV-LM004-MAIN"][st] == "PASS_TECHNICAL"
    assert rows["RV-LM004-MAIN"][rev] == "Prior OpenAI technical reconciliation; ROUND5 author-line erratum; zero org credit"
    assert rows["RV-LM004-MAIN"][ind] == "EXTERNAL_REVIEW_OPEN"
    assert rows["RV-RN-ALIGN"][st] == "AMEND"
    assert rows["RV-RN-ALIGN"][ind] == "AUTHOR_SIDE / ZERO ORG CREDIT"
    assert rows["RV-RN5-MOMENT-REPAIR"][st] == "READY"
    assert rows["RV-RN5-MOMENT-REPAIR"][rev] == "UNASSIGNED"
    assert rows["RV-RN5-MOMENT-REPAIR"][h.index("Body bytes")] == "13725"
    assert rows["RV-LM009-MAIN"][st] == "PASS_TECHNICAL"
    assert all(r[st] in RC.R17_TECH_STATUS for r in rows.values())


def test_rn5_quarantine_rows_and_frozen_objects_are_transcribed():
    qi = load("quarantine_index")
    h = qi["header"]
    cls = {r[h.index("Quarantine key")]: r[h.index("Class")] for r in qi["rows"]}
    assert [k for k in cls if k.startswith("Q-RN5")] == RN5_QUARANTINE
    assert all(cls[k] == "DEFECTIVE_SCOPE" for k in RN5_QUARANTINE)
    fo = load("frozen_objects")
    h = fo["header"]
    rows = {r[h.index("Object ID")]: r for r in fo["rows"]}
    for oid in RN5_FROZEN:
        assert rows[oid][h.index("Binding Class")].startswith("D"), oid
        assert rows[oid][h.index("Expected Bytes")].isdigit(), oid


def test_number_and_boolean_rendering_rules():
    """Integer-valued numbers render plainly, booleans as TRUE/FALSE, and an
    activity_log date serial is kept as the string the cell holds."""
    rq = load("review_queue")
    h = rq["header"]
    assert rq["rows"][0][h.index("Body bytes")] == "6874"
    al = load("activity_log")
    assert al["rows"][0][0] == "46223.95347222222"
    ac = load("automation_config")
    assert any(c == "TRUE" for r in ac["rows"] for c in r)
    assert RI.render_number("6874.0") == "6874"
    assert RI.render_number("46223.95347222222") == "46223.95347222222"
    assert RI.render_number("-3.0") == "-3"
    assert RI.render_number("not a number") == "not a number"


def dead_allowlist_entries(json_dir: str, known_path: str) -> list[str]:
    """Allowlisted problem strings, from EVERY section tools/registers_check.py
    reads, that the checker no longer emits on `json_dir`. Read through
    RC.load_known so the test cannot cover fewer sections than the checker."""
    problems, _ = RC.check(json_dir)
    return sorted(set(RC.load_known(known_path)) - set(problems))


def test_every_allowlisted_finding_still_fires():
    """A finding that no longer fires has been repaired at the source and must
    leave the allowlist, so the allowlist cannot silently outgrow the defects.
    All 23 entries are covered — the 16 of 'findings' and the 7 of
    'findings_first_visible_in_2026-09-18_export' — because the set is read
    with the checker's own loader, not from one section."""
    assert dead_allowlist_entries(JSON_DIR, KNOWN) == []
    with open(KNOWN, encoding="utf-8") as f:
        known = json.load(f)
    assert known["source_export"] == os.path.relpath(SOURCE, ROOT)
    assert len(RC.load_known(KNOWN)) == 23
    assert all(v.strip() for v in RC.load_known(KNOWN).values())


def test_control_a_repaired_second_section_finding_is_reported_dead(tmp_path):
    """Repair one of the seven 2026-09-18 defects on a copy (drop the second
    EV-LS-REQ030 row) and the dead-entry check must name exactly that entry.
    The entry lives outside the 'findings' section, so a check that read only
    that section would report nothing — which is the gap this control closes."""
    j = str(tmp_path / "json")
    shutil.copytree(JSON_DIR, j)
    path = os.path.join(j, "evidence_lineage.json")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    assert d["rows"][385][0] == d["rows"][386][0] == "EV-LS-REQ030"
    del d["rows"][386]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    dead = dead_allowlist_entries(j, KNOWN)
    assert dead == ["evidence_lineage: duplicate key 'EV-LS-REQ030' at rows 385 and 386"]
    with open(KNOWN, encoding="utf-8") as f:
        known = json.load(f)
    assert dead[0] not in known["findings"], "the control must hit an entry outside the first section"
    assert dead[0] in known["findings_first_visible_in_2026-09-18_export"]
    problems, _ = RC.check(j)
    assert set(known["findings"]) <= set(problems), "the first section alone would not have noticed"


# ---------------------------------------------------------------------------
# the cross-export diff: what moved between the two exports, mechanically
# ---------------------------------------------------------------------------

def test_markdown_era_parser_still_reads_the_old_export_as_it_did():
    """The retained parser reads the 2026-09-17 export to the 42 tabs and row
    counts this repository carried before the refresh (the seven cut tabs at
    their prefixes). The diff is only as good as its old side."""
    old = RI.parse_markdown_export(OLD_SOURCE)
    assert [t["tab"] for t in old] == RI.TAB_NAMES[:42]
    assert {t["tab"]: len(t["rows"]) for t in old} == OLD_ROW_COUNTS


def test_export_diff_is_the_mechanical_recomputation():
    """The committed diff equals what the importer recomputes from the two
    committed exports, names both exports by digest, and its status-word list
    is exactly the four register moves. A hand-curated change list — one that
    drops the verdict-artifact or last-review columns, or an Age days row —
    fails here."""
    with open(EXPORT_DIFF, encoding="utf-8") as f:
        committed = json.load(f)
    recomputed = RI.export_diff_document(OLD_SOURCE, SOURCE)
    assert committed == recomputed
    assert committed["old_export"]["sha256"] == OLD_SOURCE_SHA256 and committed["old_export"]["tabs"] == 42
    assert committed["new_export"]["sha256"] == SOURCE_SHA256 and committed["new_export"]["tabs"] == 44
    assert [(e["tab"], e["key"], e["column"], e["old"], e["new"])
            for e in committed["status_word_changes"]] == STATUS_WORD_CHANGES
    assert committed["summary"]["tabs_only_in_new_export"] == {"reusable_operations": 15, "operation_trials": 0}
    assert committed["summary"]["rows_added"]["file_catalog"] == ROW_COUNTS["file_catalog"] - MARKDOWN_PREFIX["file_catalog"]
    assert committed["summary"]["rows_removed"] == {"start_here": 1}
    assert set(committed["summary"]["changed_cells_by_kind"]) == {"content_change", "extension", "rendering_artifact"}


def test_export_diff_lists_every_changed_review_queue_column():
    """The columns the first change list omitted are in the mechanical one."""
    with open(EXPORT_DIFF, encoding="utf-8") as f:
        d = json.load(f)
    cells = {(e["key"], e["column"]): (e["old"], e["new"]) for e in d["changed_cells"] if e["tab"] == "review_queue"}
    assert cells[("RV-LM004-MAIN", "Last substantive review UTC")] == ("", "2026-09-17T16:55:40.169Z")
    assert cells[("RV-LM004-MAIN", "Verdict artifact")][1].endswith("/1QKujzzkp_nzCWFLky9ZayTzP_fbStxry/view")
    assert cells[("RV-RN-ALIGN", "Last substantive review UTC")] == ("", "2026-09-17T17:19:16.450Z")
    assert cells[("RV-RN-ALIGN", "Verdict artifact")][1].endswith("/1LtnvNd0vAW-y3pzbyHLjgtF7Uw5sTph5/view")
    assert cells[("RV-LM003-MAIN", "Age days")] == ("54", "55")
    assert sum(1 for (k, c) in cells if c == "Age days") == 24, "every pre-existing route aged one day"
    starts = {(e["key"], e["column"]) for e in d["changed_cells"] if e["tab"] == "start_here"}
    assert ("Cold entry", "Open / exact range") in starts and ("Audit coverage", "Updated UTC") in starts


def test_export_diff_new_values_are_the_register_cells():
    """Every content change in the diff ends at the cell the refreshed register
    holds now, so the diff can never claim a word the register does not."""
    with open(EXPORT_DIFF, encoding="utf-8") as f:
        d = json.load(f)
    for e in d["changed_cells"]:
        t = load(e["tab"])
        col = t["header"].index(e["column"])
        kc = RI.DIFF_KEY_COLUMNS.get(e["tab"], 0)
        matches = [r for r in t["rows"] if RI._pair_key(r, kc) == e["key"]]
        assert matches[e["occurrence"]][col] == e["new"], e


def test_control_the_diff_of_an_export_against_itself_is_empty():
    tabs = RI.parse(SOURCE)
    d = RI.export_diff(tabs, tabs)
    assert d["changed_cells"] == [] and d["header_changes"] == []
    assert d["rows_added"] == {} and d["rows_removed"] == {}
    assert d["summary"]["status_word_changes"] == 0 and len(d["summary"]["tabs_identical"]) == 44


def test_control_a_status_flip_is_a_content_change_in_a_status_column():
    """Flip one Technical status on a copy of the new side: the diff must list
    exactly one more content change, in a status column, and nothing else."""
    old = RI.parse_markdown_export(OLD_SOURCE)
    new = RI.parse(SOURCE)
    rq = [t for t in new if t["tab"] == "review_queue"][0]
    row = [r for r in rq["rows"] if r[0] == "RV-LM003-MAIN"][0]
    row[rq["header"].index("Technical status")] = "PASS_TECHNICAL"
    d = RI.export_diff(old, new)
    base = RI.export_diff(RI.parse_markdown_export(OLD_SOURCE), RI.parse(SOURCE))
    extra = [e for e in d["changed_cells"] if e not in base["changed_cells"]]
    assert extra == [{"tab": "review_queue", "key": "RV-LM003-MAIN", "occurrence": 0,
                      "column": "Technical status", "old": "NEEDS_RECONCILIATION", "new": "PASS_TECHNICAL",
                      "kind": "content_change", "status_column": True}]
    assert len(d["status_word_changes"]) == len(base["status_word_changes"]) + 1


def test_control_rendering_rules_fire_only_on_rendering_differences():
    """Each artifact rule on the cell shape it exists for, and none on a real
    change: a rule that also swallowed a status move would hide it."""
    assert RI.rendering_artifact("Q\\~N(-b,2)", "Q~N(-b,2)") == "markdown_escape"
    assert RI.rendering_artifact("46223.95347", "46223.95347222222") == "five_decimal_serial"
    assert RI.rendering_artifact("[merged] TITLE", "TITLE") == "merged_cell"
    assert RI.rendering_artifact("[merged] TITLE", "") == "merged_cell"
    assert RI.rendering_artifact("\u00f0\x9f\x91\x8d continue", "\U0001f44d continue") == "latin1_mojibake"
    assert RI.rendering_artifact("2026-07-20 23:06", "46223.9625") == "date_serial_display"
    for old, new in (("NEEDS_RECONCILIATION", "PASS_TECHNICAL"), ("ESCALATE", "EXTERNAL ONLY"),
                     ("", "2026-09-17T16:55:40.169Z"), ("54", "55"), ("46223.95347", "46223.95348"),
                     ("[merged] TITLE", "OTHER"), ("2026-07-20 23:06", "46224.9625"), ("2026-07-20 23:06", "46223.97")):
        assert RI.rendering_artifact(old, new) is None, (old, new)
        assert RI.classify_cell(old, new)[0] == "content_change", (old, new)
    assert RI.classify_cell(".../vie", ".../view") == ("extension", None)
    assert RI.status_column("Technical status") and RI.status_column("Aging action")
    assert RI.status_column("Binding Class") and RI.status_column("Queue state")
    for header in ("Age days", "Reviewer / claim", "Next action", "Verdict artifact",
                   "Disposition UTC", "Mathematical Statement", "Why No Status/Governance Change"):
        assert not RI.status_column(header), header


# ---------------------------------------------------------------------------
# negative controls — the importer
# ---------------------------------------------------------------------------

def mutated_workbook(tmp_path, old: bytes, new: bytes) -> str:
    """A copy of the source with one byte string replaced inside xl/workbook.xml."""
    out = str(tmp_path / "mutated.xlsx")
    with zipfile.ZipFile(SOURCE) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "xl/workbook.xml":
                assert old in data
                data = data.replace(old, new)
            zout.writestr(item, data)
    return out


def test_control_a_renamed_sheet_is_refused(tmp_path):
    bad = mutated_workbook(tmp_path, b'name="Run Log"', b'name="Run Logs"')
    r = run(IMPORTER, "--source", bad, "--out-json", str(tmp_path / "j"), "--out-csv", str(tmp_path / "c"))
    assert r.returncode == 2
    assert "Run Logs" in r.stderr and "Run Log" in r.stderr
    assert not os.path.exists(tmp_path / "j"), "nothing is written when the source is refused"


def test_control_a_removed_sheet_is_refused(tmp_path):
    with zipfile.ZipFile(SOURCE) as z:
        wb = z.read("xl/workbook.xml")
    start = wb.index(b'<sheet state="visible" name="Operation Trials"')
    end = wb.index(b"/>", start) + 2
    bad = mutated_workbook(tmp_path, wb[start:end], b"")
    r = run(IMPORTER, "--source", bad, "--out-json", str(tmp_path / "j"), "--out-csv", str(tmp_path / "c"))
    assert r.returncode == 2
    assert "expected 44 worksheets" in r.stderr


def test_control_a_missing_source_is_refused(tmp_path):
    r = run(IMPORTER, "--check", "--source", str(tmp_path / "nope.xlsx"))
    assert r.returncode == 2


def test_control_check_detects_a_hand_edited_cell(tmp_path):
    j, c = str(tmp_path / "json"), str(tmp_path / "csv")
    shutil.copytree(JSON_DIR, j)
    shutil.copytree(CSV_DIR, c)
    assert run(IMPORTER, "--check", "--out-json", j, "--out-csv", c).returncode == 0
    path = os.path.join(j, "review_queue.json")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    h = d["header"]
    row = [r for r in d["rows"] if r[0] == "RV-LM003-MAIN"][0]
    row[h.index("Technical status")] = "PASS_TECHNICAL"      # the promotion a hand edit would make
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
        f.write("\n")
    r = run(IMPORTER, "--check", "--out-json", j, "--out-csv", c)
    assert r.returncode == 1 and "json/review_queue.json" in r.stderr


def test_control_check_detects_a_stale_output_file(tmp_path):
    j, c = str(tmp_path / "json"), str(tmp_path / "csv")
    shutil.copytree(JSON_DIR, j)
    shutil.copytree(CSV_DIR, c)
    with open(os.path.join(j, "tab44.json"), "w", encoding="utf-8") as f:
        json.dump({"tab": "tab44", "sheet_index": 44, "header": ["x"], "rows": []}, f)
    r = run(IMPORTER, "--check", "--out-json", j, "--out-csv", c)
    assert r.returncode == 1 and "stale" in r.stderr and "tab44.json" in r.stderr


def test_the_importer_never_writes_its_source(tmp_path):
    before = sha256(SOURCE), sha256(OLD_SOURCE), sha256(EXPORT_DIFF)
    assert run(IMPORTER, "--out-json", str(tmp_path / "j"), "--out-csv", str(tmp_path / "c")).returncode == 0
    r = run(IMPORTER, "--diff-exports", str(tmp_path / "diff.json"))
    assert r.returncode == 0 and os.path.exists(tmp_path / "diff.json")
    assert not os.path.exists(tmp_path / "j" / "operation_trials.json.stale")
    assert (sha256(SOURCE), sha256(OLD_SOURCE), sha256(EXPORT_DIFF)) == before
    assert run(IMPORTER, "--diff-exports", str(tmp_path / "d2.json"), "--old-source", str(tmp_path / "nope.md")).returncode == 2


# ---------------------------------------------------------------------------
# negative controls — tools/registers_check.py
# ---------------------------------------------------------------------------

class Registers:
    """A writable copy of registers/json plus a copy of the real allowlist.

    The copy starts from the committed allowlist so that the unmutated copy
    passes and every control below fails on exactly the one mutation it makes;
    `write_known` replaces the allowlist with the real findings plus `extra`.
    """

    def __init__(self, tmp_path):
        self.dir = str(tmp_path / "json")
        shutil.copytree(JSON_DIR, self.dir)
        self.known = str(tmp_path / "KNOWN_FINDINGS.json")
        shutil.copyfile(KNOWN, self.known)

    def write_known(self, extra):
        known = RC.load_known(KNOWN)
        known.update(extra)
        with open(self.known, "w", encoding="utf-8") as f:
            json.dump({"findings": known}, f)

    def tab(self, name):
        with open(os.path.join(self.dir, f"{name}.json"), encoding="utf-8") as f:
            return json.load(f)

    def write_tab(self, name, d):
        with open(os.path.join(self.dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)

    def run(self):
        return run(CHECKER, "--json-dir", self.dir, "--known", self.known)

    def new_problems(self):
        out = self.run()
        return out.returncode, [l for l in out.stdout.splitlines() if l.startswith("NEW")]


@pytest.fixture
def regs(tmp_path):
    return Registers(tmp_path)


def test_control_the_unmutated_copy_passes(regs):
    assert regs.run().returncode == 0


def test_every_findings_section_is_read_and_the_pending_section_says_why():
    """KNOWN_FINDINGS.json keeps the 16 findings the markdown export showed
    (covered by registers/collision_proposal.json) apart from the 7 the xlsx
    export first delivered, which await a successor proposal. The checker reads
    both; a finding hidden in a section whose name does not begin with
    'findings' would NOT be read, so the split cannot silently grow."""
    with open(KNOWN, encoding="utf-8") as f:
        known = json.load(f)
    sections = [k for k, v in known.items() if k.startswith("findings") and isinstance(v, dict)]
    assert sections == ["findings", "findings_first_visible_in_2026-09-18_export"]
    assert len(known["findings"]) == 16
    assert len(known["findings_first_visible_in_2026-09-18_export"]) == 7
    assert set(RC.load_known(KNOWN)) == set(known["findings"]) | set(
        known["findings_first_visible_in_2026-09-18_export"])
    note = known["_findings_first_visible_in_2026-09-18_export_note"]
    assert "NOT COVERED BY ANY COLLISION PROPOSAL YET" in note and "successor" in note


def test_control_an_allowlist_section_that_is_not_a_mapping_is_refused(tmp_path):
    path = str(tmp_path / "KNOWN_FINDINGS.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"findings": ["artifact_index: duplicate key 'X' at rows 0 and 1"]}, f)
    with pytest.raises(ValueError):
        RC.load_known(path)


def test_control_a_finding_outside_a_findings_section_is_not_allowlisted(regs):
    d = regs.tab("review_queue")
    d["rows"].append(list(d["rows"][0]))
    regs.write_tab("review_queue", d)
    rc, new = regs.new_problems()
    problem = new[0][len("NEW    "):]
    known = RC.load_known(KNOWN)
    with open(regs.known, "w", encoding="utf-8") as f:
        json.dump({"findings": known, "notes": {problem: "misfiled"}}, f)
    assert regs.run().returncode == 1


def test_control_duplicate_review_key_is_new(regs):
    d = regs.tab("review_queue")
    d["rows"].append(list(d["rows"][0]))
    regs.write_tab("review_queue", d)
    rc, new = regs.new_problems()
    assert rc == 1 and any("review_queue: duplicate key 'RV-LM003-MAIN'" in l for l in new)


def test_control_status_outside_the_r17_set_is_new(regs):
    d = regs.tab("review_queue")
    d["rows"][0][d["header"].index("Technical status")] = "APPROVED"
    regs.write_tab("review_queue", d)
    rc, new = regs.new_problems()
    assert rc == 1 and any("status 'APPROVED' not in R17 set" in l for l in new)


def test_control_quarantine_class_outside_the_table_is_new(regs):
    d = regs.tab("quarantine_index")
    d["rows"][0][d["header"].index("Class")] = "RESTORED"
    regs.write_tab("quarantine_index", d)
    rc, new = regs.new_problems()
    assert rc == 1 and any("class 'RESTORED' not in R17 table" in l for l in new)


def test_control_frozen_row_with_digest_but_no_byte_count_is_new(regs):
    d = regs.tab("frozen_objects")
    h = d["header"]
    row = [r for r in d["rows"] if r[0] == "RN5-PROOF"][0]
    row[h.index("Expected Bytes")] = "see notes"
    regs.write_tab("frozen_objects", d)
    rc, new = regs.new_problems()
    assert rc == 1 and any("RN5-PROOF" in l and "non-numeric byte count" in l for l in new)


def test_control_allowlist_matches_exact_strings_only(regs):
    d = regs.tab("review_queue")
    d["rows"].append(list(d["rows"][0]))
    regs.write_tab("review_queue", d)
    rc, new = regs.new_problems()
    problem = new[0][len("NEW    "):]
    regs.write_known({problem + " ": "near miss"})
    assert regs.run().returncode == 1
    regs.write_known({problem: "recorded for the owner"})
    out = regs.run()
    assert out.returncode == 0 and ("KNOWN  " + problem) in out.stdout


# ---------------------------------------------------------------------------
# negative controls — tools/quarantine_check.py invariant 5 (bound members)
# ---------------------------------------------------------------------------

def binding_copy(tmp_path, mutate):
    with open(BINDING, encoding="utf-8") as f:
        b = json.load(f)
    mutate({r["carrier_id"]: r for r in b["carriers"]})
    path = str(tmp_path / "BINDING.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(b, f)
    return path


def qcheck(*args):
    return run(QCHECK, *args)


def test_quarantine_check_passes_and_names_the_bound_member():
    out = qcheck()
    assert out.returncode == 0, out.stdout
    assert "bound_members_named=2" in out.stdout
    with open(BINDING, encoding="utf-8") as f:
        rec = [r for r in json.load(f)["carriers"] if r["carrier_id"] == "RNENG-03"][0]
    assert rec["blob_stored"] is True, "the bytes stay bound"
    ann = rec["quarantine_exclusions"]
    assert [a["key"] for a in ann] == ["Q-RN5-MOMENT-001"]
    assert ann[0]["class"] == "DEFECTIVE_SCOPE" and "envelope_v" in ann[0]["scope"]


def test_control_stripped_annotation_fails_closed(tmp_path):
    path = binding_copy(tmp_path, lambda recs: recs["RNENG-03"].pop("quarantine_exclusions"))
    out = qcheck("--binding", path)
    assert out.returncode == 1
    assert "RNENG-03" in out.stdout and "Q-RN5-MOMENT-001" in out.stdout and "no quarantine_exclusions" in out.stdout


def test_control_annotation_without_a_scope_fails(tmp_path):
    def blank(recs):
        recs["RNENG-03"]["quarantine_exclusions"][0]["scope"] = ""
    out = qcheck("--binding", binding_copy(tmp_path, blank))
    assert out.returncode == 1 and "has no scope" in out.stdout


def test_control_annotation_with_the_wrong_class_fails(tmp_path):
    def wrong(recs):
        recs["RNENG-03"]["quarantine_exclusions"][0]["class"] = "SUPERSEDED"
    out = qcheck("--binding", binding_copy(tmp_path, wrong))
    assert out.returncode == 1 and "class 'SUPERSEDED' != exclusion class 'DEFECTIVE_SCOPE'" in out.stdout


def test_control_stale_annotation_on_an_unnamed_member_fails(tmp_path):
    """d3_amend.py (RNENG-04) is NOT the d3_amend_v2.py that Q-RN5-MOMENT-002 names."""
    def stale(recs):
        recs["RNENG-04"]["quarantine_exclusions"] = [
            {"key": "Q-RN5-MOMENT-002", "class": "DEFECTIVE_SCOPE", "scope": "x"}]
    out = qcheck("--binding", binding_copy(tmp_path, stale))
    assert out.returncode == 1 and "RNENG-04" in out.stdout and "stale annotation" in out.stdout


def test_control_annotation_naming_a_nonexistent_exclusion_fails(tmp_path):
    def typo(recs):
        recs["RNENG-03"]["quarantine_exclusions"][0]["key"] = "Q-RN5-MOMENT-01"
    out = qcheck("--binding", binding_copy(tmp_path, typo))
    assert out.returncode == 1 and "not an exclusion" in out.stdout


def test_control_dropping_a_register_exclusion_from_the_list_fails(tmp_path):
    with open(EXCLUSIONS, encoding="utf-8") as f:
        ex = json.load(f)
    ex["exclusions"] = [e for e in ex["exclusions"] if e["key"] != "Q-RN5-MOMENT-004"]
    path = str(tmp_path / "EXCLUSIONS.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ex, f)
    out = qcheck("--exclusions", path)
    assert out.returncode == 1 and "Q-RN5-MOMENT-004 missing from EXCLUSIONS.json" in out.stdout


def test_the_checkers_never_write_their_inputs():
    before = {p: sha256(p) for p in (SOURCE, OLD_SOURCE, BINDING, EXCLUSIONS, KNOWN)}
    for cmd in ((IMPORTER, "--check"), (CHECKER,), (QCHECK,)):
        assert run(*cmd).returncode == 0
    assert {p: sha256(p) for p in before} == before
