"""Tests for registers/COLLISION_PROPOSAL.md + registers/collision_proposal.json.

Positive control: the committed proposal passes tools/collision_proposal_check.py.

Negative controls (each must make the checker exit nonzero):
  * a proposal that DELETES rather than disambiguates;
  * a successor id that collides with an identifier already in registers/json/;
  * a KNOWN_FINDINGS finding with no proposal record;
  * plus: duplicate successor ids, a proposal record for a finding that does not
    exist, a tampered verbatim export quote, a nonzero independence credit, a
    non-additive follow-up smuggled into the executable operations, and a
    Markdown companion that drops the "nothing repaired" statement.

Each negative control is built by copying the real proposal into a scratch tree and
corrupting exactly one thing, so a test that fails is telling you about the checker,
not about the fixture.

The second half of this file (from "successor" below) pins the numbered successor
registers/collision_proposal_2026-09-19.json + registers/COLLISION_PROPOSAL_2026-09-19.md:
7 records over KNOWN_FINDINGS section 'findings_first_visible_in_2026-09-18_export',
source of record the 2026-09-18 xlsx export, every colliding row quoted cell for cell from
registers/json/<tab>.json (verbatim mode 'xlsx_export_json_rows').  Its negative controls
drive the checker through --proposal in a second sandbox and cover, beyond the first
proposal's set: tampered cells, digests and indices; the xlsx digest and byte count; ids
reissued from the frozen predecessor or colliding with the register; an edited
predecessor; a record that names the right finding but quotes the wrong row, the same row
twice or a mismatched identifier; a typed (not derived) difference list or cell count; a
keeper that is not the earlier row; a same-object flag, exact-duplicate flag or successor
id suffix that contradicts the cells; a proposed Duplicate Flags cluster id that already
exists; and a successor document that drops successor_of.

The third block ("document level and companion" below) closes the gaps an adversarial
verifier found after that: a falsified row_canonical_bytes; a document-level summary count
or classification list that contradicts the records' cells; a follow-up whose 'from' is not
the colliding identifier or whose 'to' is not an id the record proposes; a number word in a
record's materiality that contradicts the recomputed cell counts; a falsified list of the
ids the predecessor issued; a wrong workbook_tab; an APPEND_ROW retargeted away from the
Duplicate Flags registry; and, in the companion, a tampered verbatim row block, a tampered
byte-count/digest line, a falsified Cell count line, a summary-table successor id that is
not the record's, a paraphrased materiality and a gate statement demoted to lowercase.
Each of these went through the CLI unnoticed before the checks existed.

The fourth block ("repository prose against the live cells") closes the next gap the same
verifier found: registers/README.md called REL-EC021-CLS141 an exact duplicate row apart
from its review date while the record three lines below recomputed six of fifteen cells
differing, and nothing read the README.  The checker now cross-reads registers/README.md
and docs/FINDINGS_2026-09-18.md; the controls here cover an exact-duplicate claim the cells
refute in either file, a cell count that contradicts the record it is attributed to, a
count attributed to the wrong pair, and — as positive controls against over-firing — a
denial in another wording, a sentence naming no colliding identifier, and a tree in which
the prose file is absent (skipped, and said so under -v).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "collision_proposal_check.py")
JSON_PATH = os.path.join(ROOT, "registers", "collision_proposal.json")
MD_PATH = os.path.join(ROOT, "registers", "COLLISION_PROPOSAL.md")
KNOWN_PATH = os.path.join(ROOT, "registers", "KNOWN_FINDINGS.json")


def run(cwd_root: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, os.path.join(cwd_root, "tools", "collision_proposal_check.py")],
                          cwd=cwd_root, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    """A scratch tree that the checker treats as its own repo root.

    registers/source, registers/json and registers/csv are symlinked to the real
    exported data (never copied, never written), so no test can drift them. Only the
    proposal, the Markdown companion and KNOWN_FINDINGS.json are real files a test
    may corrupt.
    """
    base = tmp_path_factory.mktemp("collision_proposal")
    os.makedirs(base / "tools")
    os.makedirs(base / "registers")
    for name in ("collision_proposal_check.py", "registers_import.py"):
        shutil.copy2(os.path.join(ROOT, "tools", name), base / "tools" / name)
    for sub in ("source", "json", "csv"):
        os.symlink(os.path.join(ROOT, "registers", sub), base / "registers" / sub)
    return base


def stage(sandbox, doc=None, md=None, known=None) -> str:
    """Write a (possibly corrupted) proposal into the sandbox and return its root."""
    if doc is None:
        shutil.copy2(JSON_PATH, sandbox / "registers" / "collision_proposal.json")
    else:
        with open(sandbox / "registers" / "collision_proposal.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
    if md is None:
        shutil.copy2(MD_PATH, sandbox / "registers" / "COLLISION_PROPOSAL.md")
    else:
        (sandbox / "registers" / "COLLISION_PROPOSAL.md").write_text(md, encoding="utf-8")
    if known is None:
        shutil.copy2(KNOWN_PATH, sandbox / "registers" / "KNOWN_FINDINGS.json")
    else:
        with open(sandbox / "registers" / "KNOWN_FINDINGS.json", "w", encoding="utf-8") as f:
            json.dump(known, f, ensure_ascii=False)
    return str(sandbox)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def doc():
    return load(JSON_PATH)


def known():
    return load(KNOWN_PATH)


def md():
    return open(MD_PATH, encoding="utf-8").read()


def expect_fail(root: str, needle: str):
    r = run(root)
    assert r.returncode != 0, f"checker passed but should have failed\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert needle.lower() in r.stdout.lower(), f"expected {needle!r} in output, got:\n{r.stdout}"


# --------------------------------------------------------------------------- positive

def test_committed_proposal_passes_in_place():
    r = subprocess.run([sys.executable, TOOL], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"committed proposal fails its own checker:\n{r.stdout}\n{r.stderr}"


def test_sandbox_positive_control(sandbox):
    """The unmodified fixture must pass, or every negative control below is vacuous."""
    r = run(stage(sandbox))
    assert r.returncode == 0, f"unmodified sandbox fixture fails:\n{r.stdout}\n{r.stderr}"


def test_every_finding_has_exactly_one_record():
    d, k = doc(), known()
    keys = [p["finding_key"] for p in d["proposals"]]
    assert sorted(keys) == sorted(k["findings"]), "proposal/finding sets differ"
    assert len(keys) == len(set(keys)) == 16


def test_all_operations_are_appends():
    for p in doc()["proposals"]:
        assert p["operations"], p["record_id"]
        for op in p["operations"]:
            assert op["operation"] == "APPEND_ROW"
            assert op["append_only"] is True
            assert op["mutates_existing_rows"] is False


def test_non_additive_followups_are_operator_reserved_and_never_executable():
    for p in doc()["proposals"]:
        for fu in p.get("non_additive_followups", []):
            assert fu["operator_reserved"] is True
            assert fu["op"] in {"REIDENTIFY_KEY_CELL", "AMEND_PROTOCOL_TABLE"}
            assert fu["why_not_in_operations"]


def test_independence_credit_is_zero_and_gates_stay_open():
    pb = doc()["prepared_by"]
    assert pb["independence_credit"] == 0
    assert pb["independence_credit_reason"]
    assert "REMAIN" in pb["independence_requiring_gates_remain_open"].upper()
    assert "REMAINS OPEN" in md()


def test_markdown_states_it_is_a_proposal_and_nothing_was_repaired():
    text = md()
    assert "requires operator action" in text.lower()
    assert "nothing has been repaired" in text.lower()
    assert "export remains faithful" in text.lower()


def test_quarantine_records_offer_options_and_a_recommendation_not_a_decision():
    q = [p for p in doc()["proposals"] if p["register_tab"] == "quarantine_index"]
    assert len(q) == 3
    for p in q:
        names = {o["name"] for o in p["options"]}
        assert any("CONTAINER_POINTER" in n for n in names)
        assert any("RECLASSIFY" in n for n in names)
        for o in p["options"]:
            assert o["consequences"]
        assert p["recommendation"].startswith("RECOMMEND")
        assert "not a decision" in p["recommendation"].lower()
        assert p["successor"] is None and p["successor_not_applicable_reason"]


# --------------------------------------------------------------------------- negative

def test_negative_proposal_that_deletes_rather_than_disambiguates_fails(sandbox):
    d = doc()
    d["proposals"][0]["operations"][0]["operation"] = "DELETE_ROW"
    expect_fail(stage(sandbox, doc=d), "forbidden verb DELETE")


def test_negative_merge_operation_fails(sandbox):
    d = doc()
    d["proposals"][7]["operations"][0]["operation"] = "MERGE_DUPLICATE_ROWS"
    expect_fail(stage(sandbox, doc=d), "forbidden verb MERGE")


def test_negative_deletion_hidden_in_a_followup_fails(sandbox):
    d = doc()
    d["proposals"][0]["non_additive_followups"][0]["op"] = "DELETE_DUPLICATE_KEY_CELL"
    expect_fail(stage(sandbox, doc=d), "forbidden verb DELETE")


def test_negative_operation_claiming_to_mutate_rows_fails(sandbox):
    d = doc()
    d["proposals"][2]["operations"][0]["mutates_existing_rows"] = True
    expect_fail(stage(sandbox, doc=d), "mutates_existing_rows")


def test_negative_successor_colliding_with_an_existing_id_fails(sandbox):
    d = doc()
    # TR-P01-007-COLLISION-PROVENANCE is a real Transition ID in registers/json/.
    d["proposals"][6]["successors"][0]["proposed_id"] = "TR-P01-007-COLLISION-PROVENANCE"
    expect_fail(stage(sandbox, doc=d), "already exists in registers/json")


def test_negative_successor_colliding_with_a_bare_existing_artifact_id_fails(sandbox):
    d = doc()
    d["proposals"][0]["successor"]["proposed_id"] = "GP-DER-118-v1.2"
    expect_fail(stage(sandbox, doc=d), "already exists in registers/json")


def test_negative_two_records_issuing_the_same_successor_fails(sandbox):
    d = doc()
    d["proposals"][1]["successor"]["proposed_id"] = d["proposals"][0]["successor"]["proposed_id"]
    expect_fail(stage(sandbox, doc=d), "already issued by")


def test_negative_finding_without_a_proposal_fails(sandbox):
    k = known()
    k["findings"]["artifact_index: duplicate key 'GP-XXX-999-v9.9' at rows 1 and 2"] = \
        "synthetic finding injected by a negative control; it has no proposal record"
    expect_fail(stage(sandbox, known=k), "has no proposal record")


def test_negative_dropping_a_proposal_record_fails(sandbox):
    d = doc()
    d["proposals"] = d["proposals"][1:]
    expect_fail(stage(sandbox, doc=d), "has no proposal record")


def test_negative_proposal_for_an_unknown_finding_fails(sandbox):
    d = doc()
    d["proposals"][3]["finding_key"] = "transition_log: duplicate key 'TR-XX-000' at rows 0 and 1"
    expect_fail(stage(sandbox, doc=d), "absent from KNOWN_FINDINGS")


def test_negative_tampered_verbatim_export_quote_fails(sandbox):
    d = doc()
    d["proposals"][0]["rows"][0]["verbatim_export_line"] = "| tampered |"
    expect_fail(stage(sandbox, doc=d), "does not match the export")


def test_negative_nonzero_independence_credit_fails(sandbox):
    d = doc()
    d["prepared_by"]["independence_credit"] = 1
    expect_fail(stage(sandbox, doc=d), "independence_credit must be 0")


def test_negative_claiming_something_was_repaired_fails(sandbox):
    d = doc()
    d["nothing_repaired"] = False
    expect_fail(stage(sandbox, doc=d), "nothing_repaired")


def test_negative_markdown_dropping_the_no_repair_statement_fails(sandbox):
    text = md().replace("Nothing has been repaired", "Repaired").replace(
        "nothing has been repaired", "repaired")
    expect_fail(stage(sandbox, md=text), "nothing-has-been-repaired")


def test_negative_markdown_dropping_a_finding_key_fails(sandbox):
    key = "quarantine_index: row 16 class 'EXISTING_CONTAINER' not in R17 table"
    text = md().replace(key, "(elided)")
    expect_fail(stage(sandbox, md=text), "does not quote the finding key")


def test_negative_record_without_does_not_establish_fails(sandbox):
    d = doc()
    d["proposals"][5]["does_not_establish"] = []
    expect_fail(stage(sandbox, doc=d), "does_not_establish is missing")


def test_negative_keeper_without_a_reason_fails(sandbox):
    d = doc()
    d["proposals"][4]["keeper"]["reason"] = ""
    expect_fail(stage(sandbox, doc=d), "keeper has no stated reason")


# =========================================================================== successor
# registers/collision_proposal_2026-09-19.json — the numbered successor covering the seven
# findings under KNOWN_FINDINGS section 'findings_first_visible_in_2026-09-18_export'.
# Its source of record is the 2026-09-18 xlsx export and every colliding row is quoted cell
# for cell from registers/json/<tab>.json.  The 16-pin tests above are untouched; everything
# below drives the checker through its --proposal flag so the path is resolved at call time.

import hashlib  # noqa: E402

SUCC_REL = os.path.join("registers", "collision_proposal_2026-09-19.json")
SUCC_JSON = os.path.join(ROOT, SUCC_REL)
SUCC_MD = os.path.join(ROOT, "registers", "COLLISION_PROPOSAL_2026-09-19.md")
SOURCES_PATH = os.path.join(ROOT, "registers", "source", "SOURCES.json")
SECTION = "findings_first_visible_in_2026-09-18_export"


def run_succ(cwd_root: str, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, os.path.join(cwd_root, "tools", "collision_proposal_check.py"),
                           "--proposal", SUCC_REL, *extra],
                          cwd=cwd_root, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def sandbox2(tmp_path_factory):
    """Own scratch tree for the successor, built like `sandbox`: exported data symlinked
    (never copied, never written); only the two proposals, their companions and
    KNOWN_FINDINGS.json are real files a test may corrupt."""
    base = tmp_path_factory.mktemp("collision_proposal_successor")
    os.makedirs(base / "tools")
    os.makedirs(base / "registers")
    for name in ("collision_proposal_check.py", "registers_import.py"):
        shutil.copy2(os.path.join(ROOT, "tools", name), base / "tools" / name)
    for sub in ("source", "json", "csv"):
        os.symlink(os.path.join(ROOT, "registers", sub), base / "registers" / sub)
    return base


def stage_succ(sandbox2, doc=None, md=None, known=None, first=None) -> str:
    """Write a (possibly corrupted) successor proposal into the sandbox and return its root.
    The frozen 2026-09-18 proposal is copied unchanged unless `first` overrides it."""
    if first is None:
        shutil.copy2(JSON_PATH, sandbox2 / "registers" / "collision_proposal.json")
    else:
        with open(sandbox2 / "registers" / "collision_proposal.json", "w", encoding="utf-8") as f:
            json.dump(first, f, ensure_ascii=False)
    if doc is None:
        shutil.copy2(SUCC_JSON, sandbox2 / "registers" / "collision_proposal_2026-09-19.json")
    else:
        with open(sandbox2 / "registers" / "collision_proposal_2026-09-19.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
    if md is None:
        shutil.copy2(SUCC_MD, sandbox2 / "registers" / "COLLISION_PROPOSAL_2026-09-19.md")
    else:
        (sandbox2 / "registers" / "COLLISION_PROPOSAL_2026-09-19.md").write_text(md, encoding="utf-8")
    if known is None:
        shutil.copy2(KNOWN_PATH, sandbox2 / "registers" / "KNOWN_FINDINGS.json")
    else:
        with open(sandbox2 / "registers" / "KNOWN_FINDINGS.json", "w", encoding="utf-8") as f:
            json.dump(known, f, ensure_ascii=False)
    return str(sandbox2)


def sdoc():
    return load(SUCC_JSON)


def smd():
    return open(SUCC_MD, encoding="utf-8").read()


def expect_succ_fail(root: str, needle: str, *extra: str):
    r = run_succ(root, *extra)
    assert r.returncode != 0, f"checker passed but should have failed\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert needle.lower() in r.stdout.lower(), f"expected {needle!r} in output, got:\n{r.stdout}"


def canonical_sha(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- positive

def test_successor_passes_in_place():
    r = subprocess.run([sys.executable, TOOL, "--proposal", SUCC_REL], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"successor proposal fails its own checker:\n{r.stdout}\n{r.stderr}"
    assert "proposal=" + SUCC_REL in r.stdout and "records=7" in r.stdout and "successors=7" in r.stdout
    assert "verbatim=xlsx_export_json_rows" in r.stdout


def test_first_proposal_still_passes_with_no_flags_and_names_itself():
    r = subprocess.run([sys.executable, TOOL], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0
    assert "proposal=registers/collision_proposal.json" in r.stdout
    assert "verbatim=markdown_export_lines" in r.stdout and "records=16" in r.stdout


def test_successor_sandbox_positive_control(sandbox2):
    r = run_succ(stage_succ(sandbox2))
    assert r.returncode == 0, f"unmodified successor sandbox fixture fails:\n{r.stdout}\n{r.stderr}"


def test_successor_has_exactly_seven_records_one_per_new_section_key():
    d, k = sdoc(), known()
    keys = [p["finding_key"] for p in d["proposals"]]
    assert d["findings_source_section"] == SECTION
    assert sorted(keys) == sorted(k[SECTION])
    assert len(keys) == len(set(keys)) == 7


def test_the_two_proposals_together_cover_all_23_findings_exactly_once():
    k = known()
    all_keys = list(k["findings"]) + list(k[SECTION])
    assert len(all_keys) == len(set(all_keys)) == 23
    covered = [p["finding_key"] for p in doc()["proposals"]] + [p["finding_key"] for p in sdoc()["proposals"]]
    assert len(covered) == 23
    assert sorted(covered) == sorted(all_keys), "the two proposals do not partition the findings"


def test_every_quoted_row_equals_the_live_json_row_cell_for_cell():
    d = sdoc()
    tabs = {}
    for rel in d["source_of_record"]["json_tabs"]:
        t = load(os.path.join(ROOT, rel))
        tabs[os.path.splitext(os.path.basename(rel))[0]] = t
    n = 0
    for p in d["proposals"]:
        t = tabs[p["register_tab"]]
        assert len(p["rows"]) == 2
        for r in p["rows"]:
            live = t["rows"][r["register_row_index"]]
            assert r["verbatim_row"] == live
            assert r["row_sha256"] == canonical_sha(live)
            assert r["fields"] == dict(zip(t["header"], live))
            assert r["fields"][t["header"][0]] == p["colliding_identifier"]
            n += 1
    assert n == 14


def test_successor_xlsx_digest_matches_sources_json_and_disk():
    d = sdoc()
    src = d["source_of_record"]
    assert src["kind"] == "xlsx_export_json_rows"
    with open(SOURCES_PATH, encoding="utf-8") as f:
        sources = json.load(f)
    rec = [e for e in sources["exports"] if e["file"] == os.path.basename(src["xlsx_path"])]
    assert len(rec) == 1
    assert rec[0]["sha256"] == src["xlsx_sha256"] and rec[0]["bytes"] == src["xlsx_bytes"]
    raw = open(os.path.join(ROOT, src["xlsx_path"]), "rb").read()
    assert hashlib.sha256(raw).hexdigest() == src["xlsx_sha256"] and len(raw) == src["xlsx_bytes"]


def test_successor_ids_collide_with_nothing():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import importlib
    cpc = importlib.import_module("collision_proposal_check")
    existing = cpc.existing_identifiers(os.path.join(ROOT, "registers", "json"))
    first_ids = {sid for _, sid in cpc.collect_successors(doc())} | set(cpc.batch_ids(doc()).values())
    succ = [sid for _, sid in cpc.collect_successors(sdoc())]
    batch = set(cpc.batch_ids(sdoc()).values())
    assert len(succ) == len(set(succ)) == 7
    for sid in list(succ) + sorted(batch):
        assert sid not in existing, sid
        assert sid not in first_ids, sid
    assert all(s.endswith("@AIDX-R" + str(p["successor"]["register_row_index"])) or
               s.endswith("@EVL-R" + str(p["successor"]["register_row_index"]))
               for s, p in zip(succ, sdoc()["proposals"]))


def test_successor_supersedes_nothing_and_names_its_frozen_predecessor_by_digest():
    d = sdoc()
    assert d["supersedes"] is None
    pred = d["successor_of"]
    assert pred["path"] == "registers/collision_proposal.json"
    raw = open(JSON_PATH, "rb").read()
    assert pred["sha256"] == hashlib.sha256(raw).hexdigest() and pred["bytes"] == len(raw)


def test_successor_classification_is_derived_from_the_cells():
    d = sdoc()
    by_id = {p["colliding_identifier"]: p for p in d["proposals"]}
    diff_objects = [i for i, p in by_id.items() if not p["both_rows_cite_one_drive_object"]]
    assert diff_objects == ["GP-REQ-194-v1.0"]
    p = by_id["GP-REQ-194-v1.0"]
    assert p["defect_class"] == "DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS"
    assert p["rows"][0]["fields"]["Source"] != p["rows"][1]["fields"]["Source"]
    ev = by_id["EV-LS-REQ030"]
    assert ev["register_tab"] == "evidence_lineage"
    assert ev["exact_duplicate_row"] is False
    assert ev["rows"][0]["verbatim_row"] != ev["rows"][1]["verbatim_row"]
    assert ev["rows"][0]["fields"]["Drive ID"] == ev["rows"][1]["fields"]["Drive ID"]
    for i, p in by_id.items():
        if p["both_rows_cite_one_drive_object"]:
            src = "Source URL" if p["register_tab"] == "evidence_lineage" else "Source"
            assert p["rows"][0]["fields"][src] == p["rows"][1]["fields"][src], i
        assert p["keeper"]["register_row_index"] < p["successor"]["register_row_index"], i
        assert p["keeper"]["reason"] and "append position" in p["keeper"]["reason"].lower()
        for op in p["operations"]:
            assert op["operation"] == "APPEND_ROW" and op["append_only"] is True
            assert op["mutates_existing_rows"] is False and "Duplicate Flags" in op["target_tab"]
        for fu in p["non_additive_followups"]:
            assert fu["operator_reserved"] is True and fu["op"] == "REIDENTIFY_KEY_CELL"


def test_successor_markdown_quotes_every_new_finding_key_and_the_statements():
    text = smd()
    for k in known()[SECTION]:
        assert k in text, k
    assert "requires operator action" in text.lower()
    assert "nothing has been repaired" in text.lower()
    assert "export remains faithful" in text.lower()
    assert "independence_credit = 0" in text and "REMAINS OPEN" in text
    assert "What this document does NOT establish" in text


def test_successor_independence_credit_is_zero_and_gates_stay_open():
    pb = sdoc()["prepared_by"]
    assert pb["independence_credit"] == 0 and pb["independence_credit_reason"]
    assert "REMAIN" in pb["independence_requiring_gates_remain_open"].upper()
    assert sdoc()["nothing_repaired"] is True and sdoc()["export_remains_faithful"] is True


# --------------------------------------------------------------------------- negative

def test_negative_successor_tampered_quoted_cell_fails(sandbox2):
    d = sdoc()
    d["proposals"][0]["rows"][1]["verbatim_row"][6] = "tampered status"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "does not match the live json row cell for cell")


def test_negative_successor_wrong_row_index_in_range_fails(sandbox2):
    d = sdoc()
    d["proposals"][1]["rows"][0]["register_row_index"] += 1
    expect_succ_fail(stage_succ(sandbox2, doc=d), "does not match the live json row cell for cell")


def test_negative_successor_row_index_out_of_range_fails(sandbox2):
    d = sdoc()
    d["proposals"][6]["rows"][1]["register_row_index"] = 10_000
    expect_succ_fail(stage_succ(sandbox2, doc=d), "out of range")


def test_negative_successor_wrong_row_digest_fails(sandbox2):
    d = sdoc()
    d["proposals"][2]["rows"][0]["row_sha256"] = "0" * 64
    expect_succ_fail(stage_succ(sandbox2, doc=d), "row_sha256")


def test_negative_successor_tampered_fields_dict_fails(sandbox2):
    d = sdoc()
    d["proposals"][3]["rows"][1]["fields"]["Status"] = "PROMOTED"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "dict(zip(header, row))")


def test_negative_successor_wrong_xlsx_digest_fails(sandbox2):
    d = sdoc()
    d["source_of_record"]["xlsx_sha256"] = "f" * 64
    expect_succ_fail(stage_succ(sandbox2, doc=d), "recorded xlsx sha256")


def test_negative_successor_wrong_xlsx_byte_count_fails(sandbox2):
    d = sdoc()
    d["source_of_record"]["xlsx_bytes"] += 1
    expect_succ_fail(stage_succ(sandbox2, doc=d), "recorded xlsx byte count")


def test_negative_successor_record_for_a_finding_absent_from_the_section_fails(sandbox2):
    d = sdoc()
    d["proposals"][4]["finding_key"] = "artifact_index: duplicate key 'GP-XXX-000-v0.0' at rows 1 and 2"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "absent from KNOWN_FINDINGS")


def test_negative_successor_dropped_record_fails(sandbox2):
    d = sdoc()
    d["proposals"] = d["proposals"][:-1]
    expect_succ_fail(stage_succ(sandbox2, doc=d), "has no proposal record")


def test_negative_successor_finding_added_to_section_without_a_record_fails(sandbox2):
    k = known()
    k[SECTION]["evidence_lineage: duplicate key 'EV-XX-000' at rows 1 and 2"] = "synthetic; no record"
    expect_succ_fail(stage_succ(sandbox2, known=k), "has no proposal record")


def test_negative_successor_id_equal_to_an_existing_identifier_fails(sandbox2):
    d = sdoc()
    # 'VOID-DUPLICATE-EV-LSMAN037-20260726T1943' is a real Evidence ID in registers/json/.
    d["proposals"][6]["successor"]["proposed_id"] = "VOID-DUPLICATE-EV-LSMAN037-20260726T1943"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "already exists in registers/json")


def test_negative_successor_id_equal_to_a_bare_existing_artifact_id_fails(sandbox2):
    d = sdoc()
    d["proposals"][0]["successor"]["proposed_id"] = "GP-DATA-168-v1.1"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "already exists in registers/json")


def test_negative_successor_id_reissuing_a_first_proposal_successor_fails(sandbox2):
    d = sdoc()
    d["proposals"][0]["successor"]["proposed_id"] = "GP-DER-118-v1.2@AIDX-R95"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "already issued by the predecessor")


def test_negative_successor_batch_id_reissuing_the_first_proposal_correction_record_fails(sandbox2):
    d = sdoc()
    d["batch_level_artifacts_that_would_also_be_appended"]["correction_record"]["proposed_id"] = "GP-COR-204"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "already issued by the predecessor")


def test_negative_successor_with_edited_predecessor_fails(sandbox2):
    first = doc()
    first["proposals"][0]["materiality"] += " (edited)"
    expect_succ_fail(stage_succ(sandbox2, first=first), "successor_of.sha256 does not match")


def test_negative_successor_merge_operation_fails(sandbox2):
    d = sdoc()
    d["proposals"][1]["operations"][0]["operation"] = "MERGE_DUPLICATE_ROWS"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "forbidden verb MERGE")


def test_negative_successor_delete_operation_fails(sandbox2):
    d = sdoc()
    d["proposals"][5]["operations"][0]["operation"] = "DELETE_ROW"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "forbidden verb DELETE")


def test_negative_successor_deletion_hidden_in_a_followup_fails(sandbox2):
    d = sdoc()
    d["proposals"][2]["non_additive_followups"][0]["op"] = "REMOVE_DUPLICATE_KEY_CELL"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "forbidden verb REMOVE")


def test_negative_successor_nonzero_independence_credit_fails(sandbox2):
    d = sdoc()
    d["prepared_by"]["independence_credit"] = 1
    expect_succ_fail(stage_succ(sandbox2, doc=d), "independence_credit must be 0")


def test_negative_successor_claiming_to_supersede_the_first_proposal_fails(sandbox2):
    d = sdoc()
    d["supersedes"] = "registers/collision_proposal.json"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "must not claim to supersede")


def test_negative_successor_markdown_missing_a_finding_key_fails(sandbox2):
    key = "evidence_lineage: duplicate key 'EV-LS-REQ030' at rows 385 and 386"
    text = smd().replace(key, "(elided)")
    expect_succ_fail(stage_succ(sandbox2, md=text), "does not quote the finding key")


def test_negative_successor_markdown_dropping_the_no_repair_statement_fails(sandbox2):
    text = smd().replace("Nothing has been repaired", "Repaired").replace(
        "nothing has been repaired", "repaired")
    expect_succ_fail(stage_succ(sandbox2, md=text), "nothing-has-been-repaired")


def test_negative_successor_companion_field_pointing_nowhere_fails(sandbox2):
    d = sdoc()
    d["companion"] = "registers/NO_SUCH_COMPANION.md"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "missing companion document")


def test_negative_successor_checked_against_the_wrong_section_fails(sandbox2):
    expect_succ_fail(stage_succ(sandbox2), "disagrees with the document's own", "--section", "findings")


def test_negative_successor_record_without_does_not_establish_fails(sandbox2):
    d = sdoc()
    d["proposals"][4]["does_not_establish"] = []
    expect_succ_fail(stage_succ(sandbox2, doc=d), "does_not_establish is missing")


def test_negative_successor_keeper_without_a_reason_fails(sandbox2):
    d = sdoc()
    d["proposals"][3]["keeper"]["reason"] = ""
    expect_succ_fail(stage_succ(sandbox2, doc=d), "keeper has no stated reason")


# --------------------------------------------------------------------------- binding
# The checks below bind each successor record to the finding it claims to disambiguate and
# recompute every derived field from the live cells.  Each control corrupts one thing and
# keeps everything else self-consistent (digests, fields, indices), so only the binding or
# recomputation check can catch it.


def live_row(tab: str, idx: int):
    t = load(os.path.join(ROOT, "registers", "json", tab + ".json"))
    return t["header"], t["rows"][idx]


def quoted(tab: str, idx: int, template: dict) -> dict:
    """A fully self-consistent quotation of registers/json/<tab>.json row idx."""
    header, row = live_row(tab, idx)
    r = dict(template)
    r["register_row_index"] = idx
    r["verbatim_row"] = list(row)
    r["row_sha256"] = canonical_sha(row)
    r["row_canonical_bytes"] = len(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    r["fields"] = dict(zip(header, row))
    return r


def test_successor_cell_comparison_counts_and_lists_match_a_fresh_recomputation():
    d = sdoc()
    for p in d["proposals"]:
        ia, ib = [r["register_row_index"] for r in p["rows"]]
        header, a = live_row(p["register_tab"], ia)
        _, b = live_row(p["register_tab"], ib)
        identical = [h for h, x, y in zip(header, a, b) if x == y]
        diffs = [{"field": h, f"row_{ia}": x, f"row_{ib}": y} for h, x, y in zip(header, a, b) if x != y]
        assert p["fields_identical_in_both_rows"] == identical, p["record_id"]
        assert p["field_differences"] == diffs, p["record_id"]
        cc = p["cell_comparison"]
        assert (cc["cells_total"], cc["cells_identical"], cc["cells_differing"]) == \
            (len(header), len(identical), len(diffs)), p["record_id"]
        assert cc["cells_identical"] + cc["cells_differing"] == cc["cells_total"]


NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "sixteen": 16}


def test_successor_prose_cell_counts_agree_with_the_machine_counts():
    """The EV-LS-REQ030 record states its counts in words in three places (record
    materiality, document-level classification note, companion Part 2) and once in
    docs/FINDINGS_2026-09-18.md; each must equal the recomputed cell_comparison."""
    import re
    d = sdoc()
    ev = [p for p in d["proposals"] if p["colliding_identifier"] == "EV-LS-REQ030"][0]
    cc = ev["cell_comparison"]
    assert cc["cells_total"] == 16
    m = re.search(r"(\w+) of sixteen cells agree", ev["materiality"])
    assert m and NUMBER_WORDS[m.group(1).lower()] == cc["cells_identical"]
    m = re.search(r"(\w+) cells differ:", ev["materiality"])
    assert m and NUMBER_WORDS[m.group(1).lower()] == cc["cells_differing"]
    note = d["classification_of_the_seven_pairs"]["note"]
    m = re.search(r"(\w+) of sixteen cells differ, (\w+) agree", note)
    assert m and NUMBER_WORDS[m.group(1)] == cc["cells_differing"] and NUMBER_WORDS[m.group(2)] == cc["cells_identical"]
    text = smd()
    assert ev["materiality"] in text
    m = re.search(r"they are not\.\*\* (\w+) of sixteen cells differ", text)
    assert m and NUMBER_WORDS[m.group(1).lower()] == cc["cells_differing"]
    assert f"Cell count: 16 columns compared, {cc['cells_identical']} identical, {cc['cells_differing']} differing" in text
    findings = open(os.path.join(ROOT, "docs", "FINDINGS_2026-09-18.md"), encoding="utf-8").read()
    m = re.search(r"digest but differ in (\w+) of sixteen cells", findings)
    assert m and NUMBER_WORDS[m.group(1)] == cc["cells_differing"]
    for stale in ("Nine of sixteen", "Seven cells differ", "seven cells differ", "Seven of sixteen"):
        assert stale not in text and stale not in json.dumps(d, ensure_ascii=False) and stale not in findings


def test_successor_companion_states_every_record_cell_count():
    text = smd()
    for p in sdoc()["proposals"]:
        cc, (ia, ib) = p["cell_comparison"], [r["register_row_index"] for r in p["rows"]]
        assert (f"Cell count: {cc['cells_total']} columns compared, {cc['cells_identical']} identical, "
                f"{cc['cells_differing']} differing (rows {ia} and {ib}") in text, p["record_id"]


def test_negative_successor_right_finding_wrong_row_quoted_fails(sandbox2):
    """Row 235 (GP-AUD-176-v1.0) quoted instead of 240, with digest and fields all consistent:
    only the binding to the finding key can catch it."""
    d = sdoc()
    p = d["proposals"][0]
    p["rows"][1] = quoted("artifact_index", 235, p["rows"][1])
    expect_succ_fail(stage_succ(sandbox2, doc=d), "finding key names rows 231 and 240 but the record quotes rows [231, 235]")


def test_negative_successor_same_row_quoted_twice_fails(sandbox2):
    d = sdoc()
    p = d["proposals"][0]
    p["rows"][1] = quoted("artifact_index", 231, p["rows"][1])
    expect_succ_fail(stage_succ(sandbox2, doc=d), "each of the two rows must be quoted exactly once")


def test_negative_successor_colliding_identifier_not_the_key_cell_fails(sandbox2):
    d = sdoc()
    d["proposals"][0]["colliding_identifier"] = "GP-DATA-168-v9.9"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "but colliding_identifier is 'GP-DATA-168-v9.9'")


def test_negative_successor_finding_key_naming_another_tab_fails(sandbox2):
    """The key is renamed consistently in KNOWN_FINDINGS, the record and the companion, so
    bijection and the companion check pass; the tab named by the key must still equal the
    tab whose rows are quoted."""
    old = "artifact_index: duplicate key 'GP-DATA-168-v1.1' at rows 231 and 240"
    new = "evidence_lineage: duplicate key 'GP-DATA-168-v1.1' at rows 231 and 240"
    d, k = sdoc(), known()
    d["proposals"][0]["finding_key"] = new
    k[SECTION] = {(new if key == old else key): v for key, v in k[SECTION].items()}
    expect_succ_fail(stage_succ(sandbox2, doc=d, known=k, md=smd().replace(old, new)),
                     "finding key names tab 'evidence_lineage' but the record quotes rows of 'artifact_index'")


def test_negative_successor_typed_difference_list_fails(sandbox2):
    d = sdoc()
    p = d["proposals"][0]
    p["field_differences"] = []
    p["fields_identical_in_both_rows"] = list(live_row("artifact_index", 231)[0])
    expect_succ_fail(stage_succ(sandbox2, doc=d), "field_differences does not equal the list recomputed")


def test_negative_successor_one_difference_dropped_fails(sandbox2):
    d = sdoc()
    p = d["proposals"][6]
    p["field_differences"] = [x for x in p["field_differences"] if x["field"] != "Review ID"]
    p["fields_identical_in_both_rows"].append("Review ID")
    expect_succ_fail(stage_succ(sandbox2, doc=d), "fields_identical_in_both_rows does not equal the list recomputed")


def test_negative_successor_wrong_cell_count_fails(sandbox2):
    d = sdoc()
    d["proposals"][6]["cell_comparison"]["cells_identical"] = 9
    d["proposals"][6]["cell_comparison"]["cells_differing"] = 7
    expect_succ_fail(stage_succ(sandbox2, doc=d), "cell_comparison")


def test_negative_successor_missing_cell_count_fails(sandbox2):
    d = sdoc()
    del d["proposals"][2]["cell_comparison"]
    expect_succ_fail(stage_succ(sandbox2, doc=d), "cell_comparison")


def test_negative_successor_keeper_set_to_the_later_row_fails(sandbox2):
    d = sdoc()
    p = d["proposals"][0]
    p["keeper"]["register_row_index"], p["successor"]["register_row_index"] = 240, 231
    expect_succ_fail(stage_succ(sandbox2, doc=d), "keeper.register_row_index 240 must be the earlier quoted row 231")


def test_negative_successor_same_object_flag_contradicting_the_source_cells_fails(sandbox2):
    d = sdoc()
    p = [p for p in d["proposals"] if p["colliding_identifier"] == "GP-REQ-194-v1.0"][0]
    p["both_rows_cite_one_drive_object"] = True
    expect_succ_fail(stage_succ(sandbox2, doc=d), "both_rows_cite_one_drive_object is True but the 'Source' cells")


def test_negative_successor_defect_class_contradicting_the_source_cells_fails(sandbox2):
    d = sdoc()
    p = d["proposals"][1]
    p["defect_class"] = "DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "does not say SAME_DRIVE_OBJECT although the cells agree")


def test_negative_successor_exact_duplicate_flag_flipped_fails(sandbox2):
    d = sdoc()
    d["proposals"][6]["exact_duplicate_row"] = True
    expect_succ_fail(stage_succ(sandbox2, doc=d), "exact_duplicate_row must be False")


def test_negative_successor_id_suffix_not_the_successor_row_fails(sandbox2):
    d = sdoc()
    d["proposals"][0]["successor"]["proposed_id"] = "GP-DATA-168-v1.1@AIDX-R999"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "must be 'GP-DATA-168-v1.1@AIDX-R240'")


def test_negative_successor_id_locating_the_wrong_tab_fails(sandbox2):
    d = sdoc()
    d["proposals"][6]["successor"]["proposed_id"] = "EV-LS-REQ030@AIDX-R386"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "must be 'EV-LS-REQ030@EVL-R386'")


def test_negative_successor_cluster_id_already_in_duplicate_flags_fails(sandbox2):
    d = sdoc()
    d["proposals"][0]["operations"][0]["row"][0] = "DUP-ID-GP-DER-044"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "proposed cluster id 'DUP-ID-GP-DER-044' already exists in registers/json")


def test_negative_successor_cluster_id_reissued_from_the_predecessor_fails(sandbox2):
    d = sdoc()
    d["proposals"][0]["operations"][0]["row"][0] = "DUP-REG-ARTIFACT-INDEX-GPDER118v12-20260918"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "was already proposed by the predecessor")


def test_negative_successor_two_records_proposing_one_cluster_id_fails(sandbox2):
    d = sdoc()
    d["proposals"][1]["operations"][0]["row"][0] = d["proposals"][0]["operations"][0]["row"][0]
    expect_succ_fail(stage_succ(sandbox2, doc=d), "is already proposed by CP-AIDX-GP-DATA-168-v1.1")


def test_negative_successor_without_successor_of_fails(sandbox2):
    d = sdoc()
    del d["successor_of"]
    d["proposals"][0]["successor"]["proposed_id"] = "GP-DER-118-v1.2@AIDX-R95"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "must name its frozen predecessor in successor_of")


def test_first_proposal_cluster_ids_collide_with_nothing_and_are_distinct_from_the_successors():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import importlib
    cpc = importlib.import_module("collision_proposal_check")
    existing = cpc.existing_identifiers(os.path.join(ROOT, "registers", "json"))
    first = [cid for _, cid in cpc.collect_cluster_ids(doc())]
    second = [cid for _, cid in cpc.collect_cluster_ids(sdoc())]
    assert len(first) == 16 and len(second) == 7
    assert len(set(first) | set(second)) == 23
    assert not (set(first) | set(second)) & existing


# =========================================================================== document level and companion
# Everything a reader consults first (summary_counts, the classification lists, the
# companion's verbatim row blocks and summary table) is recomputed from the records' cells
# by the checker.  Each control below corrupts exactly one of those surfaces and keeps the
# per-record fields self-consistent, so only the document-level or companion check can
# catch it.  Every control goes through the CLI in the sandbox.

import re  # noqa: E402

CELL_COUNT_PROSE = re.compile(r"\b(?:(\w+) of )?(\w+) cells (differ|agree)\b", re.IGNORECASE)


def canon(row) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def recomputed_classes():
    """Independent re-derivation (not the checker's code) of every record's class."""
    out = []
    for p in sdoc()["proposals"]:
        ia, ib = sorted(r["register_row_index"] for r in p["rows"])
        header, a = live_row(p["register_tab"], ia)
        _, b = live_row(p["register_tab"], ib)
        src = "Source URL" if "Source URL" in header else "Source"
        same = a[header.index(src)] == b[header.index(src)]
        if "Drive ID" in header:
            same = same and a[header.index("Drive ID")] == b[header.index("Drive ID")]
        cls = ("exact_duplicate_rows" if a == b else
               "same_object_different_status_text" if same else "different_objects_one_identifier")
        out.append((p["register_tab"], p["colliding_identifier"], ia, ib, cls))
    return out


def test_successor_summary_counts_equal_a_fresh_recomputation():
    d, k = sdoc(), known()
    classes = recomputed_classes()
    want = {
        "findings_in_section": len(k[SECTION]),
        "proposal_records": len(d["proposals"]),
        "artifact_index_records": sum(1 for c in classes if c[0] == "artifact_index"),
        "evidence_lineage_records": sum(1 for c in classes if c[0] == "evidence_lineage"),
        "same_drive_object_different_status_text": sum(1 for c in classes if c[4] == "same_object_different_status_text"),
        "different_objects_one_identifier": sum(1 for c in classes if c[4] == "different_objects_one_identifier"),
        "exact_duplicate_rows": sum(1 for c in classes if c[4] == "exact_duplicate_rows"),
        "successor_identifiers_proposed": 7,
        "findings_covered_by_both_proposals_together": len(k[SECTION]) + len(doc()["proposals"]),
    }
    assert d["summary_counts"] == want
    assert want["same_drive_object_different_status_text"] == 6 and want["different_objects_one_identifier"] == 1
    assert want["exact_duplicate_rows"] == 0 and want["findings_covered_by_both_proposals_together"] == 23


def test_successor_classification_lists_equal_a_fresh_recomputation():
    block = sdoc()["classification_of_the_seven_pairs"]
    classes = recomputed_classes()
    for cls in ("same_object_different_status_text", "different_objects_one_identifier", "exact_duplicate_rows"):
        assert block[cls] == [f"{i} (rows {a} and {b})" for _, i, a, b, c in classes if c == cls], cls


def test_successor_recorded_predecessor_ids_equal_what_the_predecessor_issues():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import importlib
    cpc = importlib.import_module("collision_proposal_check")
    issued = sorted({sid for _, sid in cpc.collect_successors(doc())})
    assert sorted(sdoc()["successor_of"]["successor_ids_issued_by_predecessor"]) == issued and len(issued) == 13


def test_successor_row_canonical_bytes_and_followups_are_bound_to_the_live_rows():
    for p in sdoc()["proposals"]:
        for r in p["rows"]:
            _, live = live_row(p["register_tab"], r["register_row_index"])
            assert r["row_canonical_bytes"] == len(canon(live).encode("utf-8")), p["record_id"]
        for fu in p["non_additive_followups"]:
            assert fu["from"] == p["colliding_identifier"] and fu["to"] == p["successor"]["proposed_id"], p["record_id"]


def test_successor_every_prose_cell_count_in_every_record_and_the_companion_agrees_with_the_cells():
    """Generalises the EV-LS-REQ030 pin: for every record, every number word before
    'cells differ/agree' in its materiality equals cell_comparison, the companion quotes the
    materiality verbatim, and every such phrase anywhere in the companion or the document-level
    note matches at least one record's counts (a phrase with a total no record has is a miscount)."""
    d, text = sdoc(), smd()
    counts = [(p["cell_comparison"]["cells_total"], p["cell_comparison"]["cells_identical"],
               p["cell_comparison"]["cells_differing"]) for p in d["proposals"]]
    checked = 0
    for p in d["proposals"]:
        cc = p["cell_comparison"]
        assert p["materiality"] in text, p["record_id"]
        for m in CELL_COUNT_PROSE.finditer(p["materiality"]):
            first, second, verb = m.group(1), m.group(2).lower(), m.group(3).lower()
            want = cc["cells_differing"] if verb == "differ" else cc["cells_identical"]
            if first is not None:
                if first.lower() not in NUMBER_WORDS or second not in NUMBER_WORDS:
                    continue
                assert NUMBER_WORDS[first.lower()] == want and NUMBER_WORDS[second] == cc["cells_total"], (p["record_id"], m.group(0))
            else:
                if second not in NUMBER_WORDS:
                    continue
                assert NUMBER_WORDS[second] == want, (p["record_id"], m.group(0))
            checked += 1
    assert checked >= 3
    for blob in (text, d["classification_of_the_seven_pairs"]["note"]):
        for m in CELL_COUNT_PROSE.finditer(blob):
            first, second, verb = m.group(1), m.group(2).lower(), m.group(3).lower()
            if first is None or first.lower() not in NUMBER_WORDS or second not in NUMBER_WORDS:
                continue
            n, t = NUMBER_WORDS[first.lower()], NUMBER_WORDS[second]
            assert any(tot == t and (dif if verb == "differ" else ide) == n for tot, ide, dif in counts), m.group(0)


def test_successor_companion_quotes_every_live_row_canonically_with_its_byte_count_and_digest():
    text, n = smd(), 0
    for p in sdoc()["proposals"]:
        for r in p["rows"]:
            _, live = live_row(p["register_tab"], r["register_row_index"])
            c = canon(live)
            assert f"```json\n{c}\n```" in text, (p["record_id"], r["register_row_index"])
            assert (f"`registers/json/{p['register_tab']}.json` rows[{r['register_row_index']}], canonical "
                    f"{len(c.encode('utf-8'))} bytes, SHA-256 `{canonical_sha(live)}`") in text
            n += 1
        ident, sid = p["colliding_identifier"], p["successor"]["proposed_id"]
        rows = [l for l in text.split("\n") if l.startswith(f"| `{ident}` |")]
        assert len(rows) == 1 and rows[0].rstrip().endswith(f"| `{sid}` |"), ident
        assert f"`{ident}` → `{sid}`" in text
    assert n == 14


# --------------------------------------------------------------------------- negative

def test_negative_successor_row_canonical_bytes_falsified_fails(sandbox2):
    d = sdoc()
    d["proposals"][0]["rows"][0]["row_canonical_bytes"] = 999
    expect_succ_fail(stage_succ(sandbox2, doc=d), "row_canonical_bytes 999")


def test_negative_successor_summary_count_contradicting_the_records_fails(sandbox2):
    d = sdoc()
    d["summary_counts"]["exact_duplicate_rows"] = 1
    d["summary_counts"]["same_drive_object_different_status_text"] = 5
    expect_succ_fail(stage_succ(sandbox2, doc=d), "summary_counts.exact_duplicate_rows is 1 but the records' cells give 0")


def test_negative_successor_summary_count_that_is_not_recomputed_fails(sandbox2):
    d = sdoc()
    d["summary_counts"]["pairs_reviewed"] = 7
    expect_succ_fail(stage_succ(sandbox2, doc=d), "not a count this checker recomputes")


def test_negative_successor_wrong_covered_together_total_fails(sandbox2):
    d = sdoc()
    d["summary_counts"]["findings_covered_by_both_proposals_together"] = 24
    expect_succ_fail(stage_succ(sandbox2, doc=d), "findings_covered_by_both_proposals_together is 24")


def test_negative_successor_pair_listed_under_the_wrong_class_fails(sandbox2):
    d = sdoc()
    block = d["classification_of_the_seven_pairs"]
    block["exact_duplicate_rows"] = ["GP-REQ-194-v1.0 (rows 325 and 373)"]
    block["different_objects_one_identifier"] = []
    expect_succ_fail(stage_succ(sandbox2, doc=d), "classification_of_the_seven_pairs.exact_duplicate_rows")


def test_negative_successor_followup_to_diverging_from_the_successor_id_fails(sandbox2):
    d = sdoc()
    d["proposals"][0]["non_additive_followups"][0]["to"] = "GP-DATA-168-v1.1@AIDX-R999"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "'to' 'GP-DATA-168-v1.1@AIDX-R999' is not an identifier this record proposes")


def test_negative_successor_followup_from_not_the_colliding_identifier_fails(sandbox2):
    d = sdoc()
    d["proposals"][0]["non_additive_followups"][0]["from"] = "GP-DATA-168-v1.0"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "'from' 'GP-DATA-168-v1.0' is not the colliding identifier")


def test_negative_successor_materiality_miscount_fails(sandbox2):
    """The same miscount is planted in the JSON and the companion, so the verbatim-quotation
    check stays green and only the number-word check can catch it."""
    d = sdoc()
    old, new = "Nine of twelve cells differ", "Seven of twelve cells differ"
    assert old in d["proposals"][0]["materiality"]
    d["proposals"][0]["materiality"] = d["proposals"][0]["materiality"].replace(old, new)
    expect_succ_fail(stage_succ(sandbox2, doc=d, md=smd().replace(old, new)),
                     "materiality says 'Seven of twelve cells differ' but the live rows give 9 of 12")


def test_negative_successor_bare_count_word_miscount_fails(sandbox2):
    d = sdoc()
    old, new = "Eight cells differ", "Seven cells differ"
    assert old in d["proposals"][6]["materiality"]
    d["proposals"][6]["materiality"] = d["proposals"][6]["materiality"].replace(old, new)
    expect_succ_fail(stage_succ(sandbox2, doc=d, md=smd().replace(old, new)),
                     "says 'Seven cells differ' but the live rows give 8 cells differ")


def test_negative_successor_recorded_predecessor_ids_falsified_fails(sandbox2):
    d = sdoc()
    d["successor_of"]["successor_ids_issued_by_predecessor"] = ["BOGUS"]
    expect_succ_fail(stage_succ(sandbox2, doc=d), "successor_ids_issued_by_predecessor does not equal")


def test_negative_successor_workbook_tab_wrong_fails(sandbox2):
    d = sdoc()
    d["proposals"][6]["workbook_tab"] = "Artifact Index"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "workbook_tab 'Artifact Index' is not the workbook's name for 'evidence_lineage'")


def test_negative_successor_append_retargeted_away_from_duplicate_flags_fails(sandbox2):
    d = sdoc()
    d["proposals"][0]["operations"][0]["target_tab"] = "Artifact Index (GP-REG-032-v1.2)"
    expect_succ_fail(stage_succ(sandbox2, doc=d), "must target the Duplicate Flags registry")


def test_negative_first_proposal_append_retargeted_away_from_duplicate_flags_fails(sandbox):
    d = doc()
    d["proposals"][3]["operations"][0]["target_tab"] = "Artifact Index (GP-REG-032-v1.2)"
    expect_fail(stage(sandbox, doc=d), "must target the Duplicate Flags registry")


def test_negative_first_proposal_followup_to_an_id_it_never_proposes_fails(sandbox):
    d = doc()
    d["proposals"][6]["non_additive_followups"][0]["to"] = "TR-P12-007-SOMETHING-ELSE"
    expect_fail(stage(sandbox, doc=d), "is not an identifier this record proposes")


def test_negative_successor_markdown_verbatim_row_block_tampered_fails(sandbox2):
    old = '"SAME-LINE REPAIR PASS / T2 OPEN",'
    assert smd().count(old) == 1
    expect_succ_fail(stage_succ(sandbox2, md=smd().replace(old, '"SAME-LINE REPAIR PASS / T2 TAMPERED",')),
                     "row 231 of artifact_index is not quoted as the canonical JSON of the live row")


def test_negative_successor_markdown_byte_count_line_tampered_fails(sandbox2):
    old = "rows[231], canonical 592 bytes"
    assert old in smd()
    expect_succ_fail(stage_succ(sandbox2, md=smd().replace(old, "rows[231], canonical 593 bytes")),
                     "row 231 lacks the line")


def test_negative_successor_markdown_cell_count_line_falsified_fails(sandbox2):
    old = "Cell count: 12 columns compared, 3 identical, 9 differing (rows 231 and 240"
    assert old in smd()
    expect_succ_fail(stage_succ(sandbox2, md=smd().replace(old, "Cell count: 12 columns compared, 4 identical, 8 differing (rows 231 and 240")),
                     "CP-AIDX-GP-DATA-168-v1.1 lacks the line 'Cell count: 12 columns compared, 3 identical, 9 differing")


def test_negative_successor_markdown_summary_table_successor_id_changed_fails(sandbox2):
    old = "| `GP-DATA-168-v1.1@AIDX-R240` |"
    assert smd().count(old) == 1
    expect_succ_fail(stage_succ(sandbox2, md=smd().replace(old, "| `GP-DATA-168-v1.1@AIDX-R241` |")),
                     "summary-table row for GP-DATA-168-v1.1 must read rows '231, 240'")


def test_negative_successor_markdown_summary_table_row_missing_fails(sandbox2):
    lines = smd().split("\n")
    kept = [l for l in lines if not l.startswith("| `EV-LS-REQ030` |")]
    assert len(kept) == len(lines) - 1
    expect_succ_fail(stage_succ(sandbox2, md="\n".join(kept)),
                     "must have exactly one summary-table row starting '| `EV-LS-REQ030` |', found 0")


def test_negative_successor_markdown_materiality_paraphrased_fails(sandbox2):
    old = "Nine of twelve cells differ. This is a material content difference"
    assert old in smd()
    expect_succ_fail(stage_succ(sandbox2, md=smd().replace(old, "Nine of twelve cells differ; a material content difference")),
                     "CP-AIDX-GP-DATA-168-v1.1 materiality is not quoted verbatim")


def test_negative_successor_markdown_gate_statement_demoted_to_lowercase_fails(sandbox2):
    text = smd().replace("REMAINS OPEN", "remains open")
    assert "remains open" in text.lower() and "REMAINS OPEN" not in text
    expect_succ_fail(stage_succ(sandbox2, md=text), "uppercase 'REMAINS OPEN'")


# =========================================================================== third proposal
# registers/collision_proposal_2026-09-19b.json — the third numbered successor covering the
# fourteen findings under KNOWN_FINDINGS section 'findings_first_keyed_2026-09-19' (relations,
# review_ledger, definitions).  Its successor_of names the 2026-09-19 document, which names
# the 2026-09-18 document, so the predecessor chain is two deep.  The blocks above are
# untouched; everything below drives the checker through --proposal in its own sandbox.

THIRD_REL = os.path.join("registers", "collision_proposal_2026-09-19b.json")
THIRD_JSON = os.path.join(ROOT, THIRD_REL)
THIRD_MD = os.path.join(ROOT, "registers", "COLLISION_PROPOSAL_2026-09-19b.md")
README_PATH = os.path.join(ROOT, "registers", "README.md")
FINDINGS_REL = os.path.join("docs", "FINDINGS_2026-09-18.md")
FINDINGS_PATH = os.path.join(ROOT, FINDINGS_REL)
SECTION3 = "findings_first_keyed_2026-09-19"
TABS3 = ("relations", "review_ledger", "definitions")
LOCATORS3 = {"relations": "REL", "review_ledger": "RVL", "definitions": "DEF"}
WORKBOOK3 = {"relations": "Relation Index", "review_ledger": "Review Independence", "definitions": "Definition Registry"}
RULE3 = {"relations": ["Source URL"], "review_ledger": ["Exact Object ID"], "definitions": ["Source URL"]}


def run_third(cwd_root: str, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, os.path.join(cwd_root, "tools", "collision_proposal_check.py"),
                           "--proposal", THIRD_REL, *extra],
                          cwd=cwd_root, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def sandbox3(tmp_path_factory):
    """Own scratch tree for the third proposal, built like `sandbox2`: exported data
    symlinked (never copied, never written); the three proposals, their companions,
    KNOWN_FINDINGS.json, the two prose files the checker cross-reads and the two tools are
    real files a test may corrupt."""
    base = tmp_path_factory.mktemp("collision_proposal_third")
    os.makedirs(base / "tools")
    os.makedirs(base / "registers")
    os.makedirs(base / "docs")
    for sub in ("source", "json", "csv"):
        os.symlink(os.path.join(ROOT, "registers", sub), base / "registers" / sub)
    return base


def stage_third(sandbox3, doc=None, md=None, known=None, first=None, second=None, importer=None,
                readme=None, findings=None) -> str:
    """Write a (possibly corrupted) third proposal into the sandbox and return its root.  Both
    frozen predecessors are copied unchanged unless `first` / `second` override them; the
    importer (whose SHEETS map the checker reads for workbook tab names) is copied unchanged
    unless `importer` supplies replacement source text.  registers/README.md and
    docs/FINDINGS_2026-09-18.md — the repository prose the checker cross-reads, never
    writes — are copied unchanged unless `readme` / `findings` supply replacement text."""
    for name in ("collision_proposal_check.py", "registers_import.py"):
        shutil.copy2(os.path.join(ROOT, "tools", name), sandbox3 / "tools" / name)
    if importer is not None:
        (sandbox3 / "tools" / "registers_import.py").write_text(importer, encoding="utf-8")
    for src, dst, override in ((JSON_PATH, "collision_proposal.json", first),
                               (SUCC_JSON, "collision_proposal_2026-09-19.json", second),
                               (THIRD_JSON, "collision_proposal_2026-09-19b.json", doc)):
        if override is None:
            shutil.copy2(src, sandbox3 / "registers" / dst)
        else:
            with open(sandbox3 / "registers" / dst, "w", encoding="utf-8") as f:
                json.dump(override, f, ensure_ascii=False)
    shutil.copy2(SUCC_MD, sandbox3 / "registers" / "COLLISION_PROPOSAL_2026-09-19.md")
    if md is None:
        shutil.copy2(THIRD_MD, sandbox3 / "registers" / "COLLISION_PROPOSAL_2026-09-19b.md")
    else:
        (sandbox3 / "registers" / "COLLISION_PROPOSAL_2026-09-19b.md").write_text(md, encoding="utf-8")
    if known is None:
        shutil.copy2(KNOWN_PATH, sandbox3 / "registers" / "KNOWN_FINDINGS.json")
    else:
        with open(sandbox3 / "registers" / "KNOWN_FINDINGS.json", "w", encoding="utf-8") as f:
            json.dump(known, f, ensure_ascii=False)
    for src, dst, override in ((README_PATH, sandbox3 / "registers" / "README.md", readme),
                               (FINDINGS_PATH, sandbox3 / FINDINGS_REL, findings)):
        if override is None:
            shutil.copy2(src, dst)
        else:
            open(dst, "w", encoding="utf-8").write(override)
    return str(sandbox3)


def tdoc():
    return load(THIRD_JSON)


def tmd():
    return open(THIRD_MD, encoding="utf-8").read()


def trec(ident: str) -> int:
    """Index of the record for a colliding identifier (records are looked up by id, never
    by position, so a reordering of the document cannot silently retarget a control)."""
    for i, p in enumerate(tdoc()["proposals"]):
        if p["colliding_identifier"] == ident:
            return i
    raise KeyError(ident)


def expect_third_fail(root: str, needle: str, *extra: str):
    r = run_third(root, *extra)
    assert r.returncode != 0, f"checker passed but should have failed\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert needle.lower() in r.stdout.lower(), f"expected {needle!r} in output, got:\n{r.stdout}"


def cpc_module():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import importlib
    return importlib.import_module("collision_proposal_check")


# --------------------------------------------------------------------------- positive

def test_third_passes_in_place():
    r = subprocess.run([sys.executable, TOOL, "--proposal", THIRD_REL], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"third proposal fails its own checker:\n{r.stdout}\n{r.stderr}"
    assert "proposal=" + THIRD_REL in r.stdout and "records=14" in r.stdout and "successors=14" in r.stdout
    assert "section=" + SECTION3 in r.stdout and "verbatim=xlsx_export_json_rows" in r.stdout


def test_earlier_proposals_still_print_the_same_summaries():
    """The checker gained per-tab rules and a chain walk; the two frozen documents must still
    pass with the summaries they printed before."""
    r = subprocess.run([sys.executable, TOOL], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith(
        "proposal=registers/collision_proposal.json section=findings verbatim=markdown_export_lines "
        "records=16 successors=13 failures=0")
    r = subprocess.run([sys.executable, TOOL, "--proposal", SUCC_REL], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith(
        "proposal=registers/collision_proposal_2026-09-19.json section=findings_first_visible_in_2026-09-18_export "
        "verbatim=xlsx_export_json_rows records=7 successors=7 failures=0")


def test_third_sandbox_positive_control(sandbox3):
    r = run_third(stage_third(sandbox3))
    assert r.returncode == 0, f"unmodified third sandbox fixture fails:\n{r.stdout}\n{r.stderr}"


def test_third_has_exactly_fourteen_records_one_per_section_key():
    d, k = tdoc(), known()
    keys = [p["finding_key"] for p in d["proposals"]]
    assert d["findings_source_section"] == SECTION3
    assert sorted(keys) == sorted(k[SECTION3])
    assert len(keys) == len(set(keys)) == 14
    tabs = [p["register_tab"] for p in d["proposals"]]
    assert tabs.count("relations") == 11 and tabs.count("review_ledger") == 2 and tabs.count("definitions") == 1


def test_the_three_proposals_together_cover_all_37_findings_exactly_once():
    k = known()
    all_keys = list(k["findings"]) + list(k[SECTION]) + list(k[SECTION3])
    assert len(all_keys) == len(set(all_keys)) == 37
    covered = ([p["finding_key"] for p in doc()["proposals"]] + [p["finding_key"] for p in sdoc()["proposals"]]
               + [p["finding_key"] for p in tdoc()["proposals"]])
    assert len(covered) == 37
    assert sorted(covered) == sorted(all_keys), "the three proposals do not partition the findings"


def test_third_every_quoted_row_equals_the_live_json_row_cell_for_cell():
    d = tdoc()
    tabs = {}
    for rel in d["source_of_record"]["json_tabs"]:
        t = load(os.path.join(ROOT, rel))
        tabs[os.path.splitext(os.path.basename(rel))[0]] = t
    assert set(tabs) == set(TABS3)
    n = 0
    for p in d["proposals"]:
        t = tabs[p["register_tab"]]
        assert len(p["rows"]) == 2
        for r in p["rows"]:
            live = t["rows"][r["register_row_index"]]
            assert r["verbatim_row"] == live
            assert r["row_sha256"] == canonical_sha(live)
            assert r["row_canonical_bytes"] == len(canon(live).encode("utf-8"))
            assert r["fields"] == dict(zip(t["header"], live))
            assert r["fields"][t["header"][0]] == p["colliding_identifier"]
            n += 1
    assert n == 28


def test_third_xlsx_digest_matches_sources_json_and_disk():
    src = tdoc()["source_of_record"]
    assert src["kind"] == "xlsx_export_json_rows"
    with open(SOURCES_PATH, encoding="utf-8") as f:
        sources = json.load(f)
    rec = [e for e in sources["exports"] if e["file"] == os.path.basename(src["xlsx_path"])]
    assert len(rec) == 1
    assert rec[0]["sha256"] == src["xlsx_sha256"] and rec[0]["bytes"] == src["xlsx_bytes"]
    raw = open(os.path.join(ROOT, src["xlsx_path"]), "rb").read()
    assert hashlib.sha256(raw).hexdigest() == src["xlsx_sha256"] and len(raw) == src["xlsx_bytes"]


def test_third_successor_ids_collide_with_nothing_including_both_predecessors():
    cpc = cpc_module()
    existing = cpc.existing_identifiers(os.path.join(ROOT, "registers", "json"))
    issued_before = set()
    clusters_before = set()
    for pdoc in (doc(), sdoc()):
        issued_before |= {sid for _, sid in cpc.collect_successors(pdoc)} | set(cpc.batch_ids(pdoc).values())
        clusters_before |= {cid for _, cid in cpc.collect_cluster_ids(pdoc)}
    succ = [sid for _, sid in cpc.collect_successors(tdoc())]
    clusters = [cid for _, cid in cpc.collect_cluster_ids(tdoc())]
    batch = set(cpc.batch_ids(tdoc()).values())
    assert len(succ) == len(set(succ)) == 14 and len(clusters) == len(set(clusters)) == 14 and len(batch) == 2
    for sid in list(succ) + clusters + sorted(batch):
        assert sid not in existing, sid
        assert sid not in issued_before and sid not in clusters_before, sid
    for sid, p in zip(succ, tdoc()["proposals"]):
        assert sid == f"{p['colliding_identifier']}@{LOCATORS3[p['register_tab']]}-R{p['successor']['register_row_index']}"


def test_third_names_its_chain_by_digest_and_supersedes_nothing():
    d = tdoc()
    assert d["supersedes"] is None
    pred = d["successor_of"]
    assert pred["path"] == SUCC_REL
    raw = open(SUCC_JSON, "rb").read()
    assert pred["sha256"] == hashlib.sha256(raw).hexdigest() and pred["bytes"] == len(raw)
    raw0 = open(JSON_PATH, "rb").read()
    chain = d["predecessor_chain"]
    assert [c["path"] for c in chain] == [SUCC_REL, "registers/collision_proposal.json"]
    assert chain[1]["sha256"] == hashlib.sha256(raw0).hexdigest() and chain[1]["bytes"] == len(raw0)
    cpc = cpc_module()
    assert sorted(pred["successor_ids_issued_by_predecessor"]) == sorted({s for _, s in cpc.collect_successors(sdoc())})
    assert sorted(chain[1]["successor_ids_issued"]) == sorted({s for _, s in cpc.collect_successors(doc())})
    assert d["summary_counts"]["findings_covered_by_the_chain_together"] == 14 + 7 + 16 == 37


def test_third_classification_is_derived_from_the_cells_with_the_per_tab_rule():
    d = tdoc()
    by_id = {p["colliding_identifier"]: p for p in d["proposals"]}
    for ident, p in by_id.items():
        header, a = live_row(p["register_tab"], p["rows"][0]["register_row_index"])
        _, b = live_row(p["register_tab"], p["rows"][1]["register_row_index"])
        rule = RULE3[p["register_tab"]]
        assert p["drive_object_rule_cells"] == rule, ident
        same = all(a[header.index(c)] == b[header.index(c)] for c in rule)
        assert p["both_rows_cite_one_drive_object"] is same, ident
        assert p["exact_duplicate_row"] is (a == b) and not p["exact_duplicate_row"], ident
        assert p["workbook_tab"] == WORKBOOK3[p["register_tab"]]
        if same:
            assert "SAME_DRIVE_OBJECT" in p["defect_class"] and "DIFFERENT_OBJECTS" not in p["defect_class"]
        else:
            assert p["defect_class"] == "DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS", ident
            assert p["drive_source_cited_by_both_rows"] is None
        if p["register_tab"] == "review_ledger":
            assert p["drive_source_cited_by_both_rows"] is None and p["drive_sources_cited"] is None
            assert "Source URL" not in header and "Drive ID" not in header
        if p["register_tab"] == "relations":
            triple = all(a[header.index(c)] == b[header.index(c)] for c in ("Source object", "Relation type", "Target object"))
            assert p["relation_triple_cells_agree"] is triple and ("SAME_RELATION" in p["defect_class"]) is triple
            assert p["target_url_cells_agree"] is (a[header.index("Target URL")] == b[header.index("Target URL")])
        assert p["keeper"]["register_row_index"] < p["successor"]["register_row_index"]
        assert "append position" in p["keeper"]["reason"].lower()
        for op in p["operations"]:
            assert op["operation"] == "APPEND_ROW" and op["append_only"] is True
            assert op["mutates_existing_rows"] is False and "Duplicate Flags" in op["target_tab"]
        for fu in p["non_additive_followups"]:
            assert fu["operator_reserved"] is True and fu["op"] == "REIDENTIFY_KEY_CELL"
            assert fu["from"] == ident and fu["to"] == p["successor"]["proposed_id"]
    same_ids = [i for i, p in by_id.items() if p["both_rows_cite_one_drive_object"]]
    assert same_ids == ["REL-EC021-CLS141"]
    ec = by_id["REL-EC021-CLS141"]
    assert ec["relation_triple_cells_agree"] is True and ec["target_url_cells_agree"] is False
    assert ec["cell_comparison"] == {**ec["cell_comparison"], "cells_total": 15, "cells_identical": 9, "cells_differing": 6}
    assert {x["field"] for x in ec["field_differences"]} == {"Exact scope / meaning", "Evidentiary effect", "Authority effect",
                                                              "Target URL", "Provenance", "Last reviewed"}
    assert "VOID-DUPLICATE" in ec["in_register_precedent"] and "NOT applied" in ec["in_register_precedent"]
    assert d["classification_of_the_fourteen_pairs"]["same_object_different_cells"] == ["REL-EC021-CLS141 (rows 232 and 235)"]
    assert d["classification_of_the_fourteen_pairs"]["exact_duplicate_rows"] == []
    assert len(d["classification_of_the_fourteen_pairs"]["different_objects_one_identifier"]) == 13
    assert d["summary_counts"]["same_drive_object_different_cells"] == 1
    assert d["summary_counts"]["different_objects_one_identifier"] == 13 and d["summary_counts"]["exact_duplicate_rows"] == 0


def test_third_workbook_tab_names_come_from_the_importer_not_the_json_tab_field():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import importlib
    ri = importlib.import_module("registers_import")
    sheets = {machine: sheet for sheet, machine in ri.SHEETS}
    cpc = cpc_module()
    for machine, sheet in cpc.WORKBOOK_TAB.items():
        assert sheets[machine] == sheet, machine
        t = load(os.path.join(ROOT, "registers", "json", machine + ".json"))
        assert t["tab"] == machine  # the JSON 'tab' field is the machine name, not the sheet name
    for p in tdoc()["proposals"]:
        assert p["workbook_tab"] == sheets[p["register_tab"]]


def test_third_markdown_quotes_every_finding_key_row_and_statement():
    text = tmd()
    for k in known()[SECTION3]:
        assert k in text, k
    assert "requires operator action" in text.lower()
    assert "nothing has been repaired" in text.lower()
    assert "export remains faithful" in text.lower()
    assert "independence_credit = 0" in text and "REMAINS OPEN" in text
    assert "What this document does NOT establish" in text
    assert "third numbered proposal" in text
    for pdoc_path in (JSON_PATH, SUCC_JSON):
        assert hashlib.sha256(open(pdoc_path, "rb").read()).hexdigest() in text
    n = 0
    for p in tdoc()["proposals"]:
        cc = p["cell_comparison"]
        ia, ib = [r["register_row_index"] for r in p["rows"]]
        assert (f"Cell count: {cc['cells_total']} columns compared, {cc['cells_identical']} identical, "
                f"{cc['cells_differing']} differing (rows {ia} and {ib}") in text, p["record_id"]
        assert p["materiality"] in text, p["record_id"]
        for r in p["rows"]:
            _, live = live_row(p["register_tab"], r["register_row_index"])
            c = canon(live)
            assert f"```json\n{c}\n```" in text
            assert (f"`registers/json/{p['register_tab']}.json` rows[{r['register_row_index']}], canonical "
                    f"{len(c.encode('utf-8'))} bytes, SHA-256 `{canonical_sha(live)}`") in text
            n += 1
        ident, sid = p["colliding_identifier"], p["successor"]["proposed_id"]
        rows = [l for l in text.split("\n") if l.startswith(f"| `{ident}` |")]
        assert len(rows) == 1 and rows[0].rstrip().endswith(f"| `{sid}` |"), ident
        assert f"`{ident}` → `{sid}`" in text
    assert n == 28


def test_third_prose_cell_counts_agree_with_the_machine_counts():
    """Every number word (including hyphenated 'twenty-four') before 'cells differ/agree' in
    each record's materiality equals the recomputed counts."""
    words = dict(NUMBER_WORDS, **{"thirteen": 13, "fourteen": 14, "fifteen": 15, "twenty-four": 24})
    prose_re = cpc_module().CELL_COUNT_PROSE  # the hyphen-aware form; the module-level one above is the older shape
    checked = 0
    for p in tdoc()["proposals"]:
        cc = p["cell_comparison"]
        for m in prose_re.finditer(p["materiality"]):
            first, second, verb = m.group(1), m.group(2).lower(), m.group(3).lower()
            want = cc["cells_differing"] if verb == "differ" else cc["cells_identical"]
            if first is not None:
                if first.lower() not in words or second not in words:
                    continue
                assert words[first.lower()] == want and words[second] == cc["cells_total"], (p["record_id"], m.group(0))
            else:
                if second not in words:
                    continue
                assert words[second] == want, (p["record_id"], m.group(0))
            checked += 1
    assert checked >= 14


def test_third_independence_credit_is_zero_and_gates_stay_open():
    d = tdoc()
    pb = d["prepared_by"]
    assert pb["independence_credit"] == 0 and pb["independence_credit_reason"]
    assert "REMAIN" in pb["independence_requiring_gates_remain_open"].upper()
    assert d["nothing_repaired"] is True and d["export_remains_faithful"] is True
    assert d["does_not_establish"] and all(p["does_not_establish"] for p in d["proposals"])


# --------------------------------------------------------------------------- negative

def test_negative_third_tampered_quoted_cell_fails(sandbox3):
    d = tdoc()
    d["proposals"][trec("REL-036")]["rows"][1]["verbatim_row"][13] = "PROMOTED"
    expect_third_fail(stage_third(sandbox3, doc=d), "does not match the live json row cell for cell")


def test_negative_third_wrong_row_index_fails(sandbox3):
    d = tdoc()
    d["proposals"][trec("DEF-049")]["rows"][0]["register_row_index"] = 47
    expect_third_fail(stage_third(sandbox3, doc=d), "does not match the live json row cell for cell")


def test_negative_third_right_finding_wrong_row_quoted_consistently_fails(sandbox3):
    """Row 49 (DEF-050) quoted in place of 50 with digest, bytes and fields all consistent:
    only the binding to the finding key can catch it."""
    d = tdoc()
    p = d["proposals"][trec("DEF-049")]
    p["rows"][1] = quoted("definitions", 49, p["rows"][1])
    expect_third_fail(stage_third(sandbox3, doc=d), "finding key names rows 48 and 50 but the record quotes rows [48, 49]")


def test_negative_third_wrong_xlsx_digest_fails(sandbox3):
    d = tdoc()
    d["source_of_record"]["xlsx_sha256"] = "e" * 64
    expect_third_fail(stage_third(sandbox3, doc=d), "recorded xlsx sha256")


def test_negative_third_record_for_a_finding_absent_from_the_section_fails(sandbox3):
    d = tdoc()
    d["proposals"][trec("REL-048")]["finding_key"] = "relations: duplicate key 'REL-999' at rows 1 and 2"
    expect_third_fail(stage_third(sandbox3, doc=d), "absent from KNOWN_FINDINGS")


def test_negative_third_dropped_record_fails(sandbox3):
    d = tdoc()
    d["proposals"] = d["proposals"][:-1]
    expect_third_fail(stage_third(sandbox3, doc=d), "has no proposal record")


def test_negative_third_successor_id_equal_to_an_existing_identifier_fails(sandbox3):
    d = tdoc()
    # 'REL-177-COLLISION-PROVENANCE' is a real Relation ID in registers/json/.
    d["proposals"][trec("REL-037")]["successor"]["proposed_id"] = "REL-177-COLLISION-PROVENANCE"
    expect_third_fail(stage_third(sandbox3, doc=d), "already exists in registers/json")


def test_negative_third_successor_id_equal_to_a_bare_existing_id_fails(sandbox3):
    d = tdoc()
    d["proposals"][trec("REV-P12-GP-006")]["successor"]["proposed_id"] = "REV-P12-GP-008"
    # REV-P12-GP-008 is unassigned; a bare id that IS assigned must fail.
    d["proposals"][trec("REV-P12-GP-006")]["successor"]["proposed_id"] = "REV-P12-GP-007"
    expect_third_fail(stage_third(sandbox3, doc=d), "already exists in registers/json")


def test_negative_third_successor_id_issued_by_the_first_proposal_two_levels_up_fails(sandbox3):
    d = tdoc()
    d["proposals"][trec("REL-038")]["successor"]["proposed_id"] = "GP-DER-118-v1.2@AIDX-R95"
    expect_third_fail(stage_third(sandbox3, doc=d),
                      "successor id 'GP-DER-118-v1.2@AIDX-R95' was already issued by the predecessor registers/collision_proposal.json")


def test_negative_third_successor_id_issued_by_the_second_proposal_fails(sandbox3):
    d = tdoc()
    d["proposals"][trec("REL-039")]["successor"]["proposed_id"] = "EV-LS-REQ030@EVL-R386"
    expect_third_fail(stage_third(sandbox3, doc=d),
                      "was already issued by the predecessor registers/collision_proposal_2026-09-19.json")


def test_negative_third_batch_id_issued_by_the_first_proposal_two_levels_up_fails(sandbox3):
    d = tdoc()
    d["batch_level_artifacts_that_would_also_be_appended"]["correction_record"]["proposed_id"] = "GP-COR-204"
    expect_third_fail(stage_third(sandbox3, doc=d), "was already issued by the predecessor registers/collision_proposal.json")


def test_negative_third_cluster_id_proposed_by_the_first_proposal_two_levels_up_fails(sandbox3):
    d = tdoc()
    d["proposals"][trec("REL-040")]["operations"][0]["row"][0] = "DUP-REG-TRANSITION-LOG-TRP12007-20260918"
    expect_third_fail(stage_third(sandbox3, doc=d), "was already proposed by the predecessor registers/collision_proposal.json")


def test_negative_third_merge_operation_fails(sandbox3):
    d = tdoc()
    d["proposals"][trec("REL-132")]["operations"][0]["operation"] = "MERGE_DUPLICATE_ROWS"
    expect_third_fail(stage_third(sandbox3, doc=d), "forbidden verb MERGE")


def test_negative_third_delete_operation_fails(sandbox3):
    d = tdoc()
    d["proposals"][trec("REL-133")]["operations"][0]["operation"] = "DELETE_ROW"
    expect_third_fail(stage_third(sandbox3, doc=d), "forbidden verb DELETE")


def test_negative_third_nonzero_independence_credit_fails(sandbox3):
    d = tdoc()
    d["prepared_by"]["independence_credit"] = 1
    expect_third_fail(stage_third(sandbox3, doc=d), "independence_credit must be 0")


def test_negative_third_markdown_missing_a_finding_key_fails(sandbox3):
    key = "review_ledger: duplicate key 'REV-P02-GP-INTERVAL-001' at rows 19 and 21"
    text = tmd().replace(key, "(elided)")
    expect_third_fail(stage_third(sandbox3, md=text), "does not quote the finding key")


def test_negative_third_wrong_locator_fails(sandbox3):
    d = tdoc()
    d["proposals"][trec("REV-P02-GP-INTERVAL-001")]["successor"]["proposed_id"] = "REV-P02-GP-INTERVAL-001@REV-R21"
    d["proposals"][trec("REV-P02-GP-INTERVAL-001")]["non_additive_followups"][0]["to"] = "REV-P02-GP-INTERVAL-001@REV-R21"
    expect_third_fail(stage_third(sandbox3, doc=d), "must be 'REV-P02-GP-INTERVAL-001@RVL-R21'")


def test_negative_third_locator_of_another_tab_fails(sandbox3):
    d = tdoc()
    d["proposals"][trec("DEF-049")]["successor"]["proposed_id"] = "DEF-049@REL-R50"
    d["proposals"][trec("DEF-049")]["non_additive_followups"][0]["to"] = "DEF-049@REL-R50"
    expect_third_fail(stage_third(sandbox3, doc=d), "must be 'DEF-049@DEF-R50'")


def test_negative_third_wrong_workbook_tab_fails(sandbox3):
    d = tdoc()
    d["proposals"][trec("REL-134")]["workbook_tab"] = "relations"
    expect_third_fail(stage_third(sandbox3, doc=d), "workbook_tab 'relations' is not the workbook's name for 'relations' ('Relation Index')")


def test_negative_third_checker_tab_map_disagreeing_with_the_importer_fails(sandbox3):
    """Rename a sheet in the sandbox importer's SHEETS map: the checker's WORKBOOK_TAB no
    longer matches the importer and must say so, whatever the document claims."""
    src = open(os.path.join(ROOT, "tools", "registers_import.py"), encoding="utf-8").read()
    old = '("Definition Registry", "definitions")'
    assert old in src
    expect_third_fail(stage_third(sandbox3, importer=src.replace(old, '("Definitions", "definitions")')),
                      "WORKBOOK_TAB['definitions'] = 'Definition Registry' but tools/registers_import.py maps 'definitions' to 'Definitions'")


def test_negative_third_predecessor_sha256_not_matching_fails(sandbox3):
    d = tdoc()
    d["successor_of"]["sha256"] = "0" * 64
    expect_third_fail(stage_third(sandbox3, doc=d), "successor_of.sha256 does not match registers/collision_proposal_2026-09-19.json")


def test_negative_third_edited_direct_predecessor_fails(sandbox3):
    second = sdoc()
    second["proposals"][0]["materiality"] += " (edited)"
    expect_third_fail(stage_third(sandbox3, second=second), "successor_of.sha256 does not match registers/collision_proposal_2026-09-19.json")


def test_negative_third_edited_root_predecessor_two_levels_up_fails(sandbox3):
    """The 2026-09-18 document is edited; the third document's own successor_of still matches,
    so only the chain walk (through the 2026-09-19 document's successor_of) can catch it."""
    first = doc()
    first["proposals"][0]["materiality"] += " (edited)"
    expect_third_fail(stage_third(sandbox3, first=first), "successor_of.sha256 does not match registers/collision_proposal.json")


def test_negative_third_predecessor_chain_list_falsified_fails(sandbox3):
    d = tdoc()
    d["predecessor_chain"] = d["predecessor_chain"][:1]
    expect_third_fail(stage_third(sandbox3, doc=d), "predecessor_chain lists")


def test_negative_third_predecessor_chain_loop_fails(sandbox3):
    """The direct predecessor is rewritten to name the third document as ITS predecessor
    (digests kept consistent), so the walk would loop; it must stop and fail instead."""
    d = tdoc()
    second = sdoc()
    third_raw = json.dumps(d, ensure_ascii=False).encode("utf-8")
    second["successor_of"] = {"path": THIRD_REL, "sha256": hashlib.sha256(third_raw).hexdigest(),
                              "bytes": len(third_raw), "successor_ids_issued_by_predecessor": []}
    second_raw = json.dumps(second, ensure_ascii=False).encode("utf-8")
    d["successor_of"]["sha256"] = hashlib.sha256(second_raw).hexdigest()
    d["successor_of"]["bytes"] = len(second_raw)
    # The walk reaches the second document (its digest matches) and then meets the third
    # document's own path; the revisit check fires before any digest of that level is taken.
    expect_third_fail(stage_third(sandbox3, doc=d, second=second), "predecessor chain revisits")


def test_negative_third_same_relation_class_on_a_different_relations_pair_fails(sandbox3):
    d = tdoc()
    p = d["proposals"][trec("REL-048")]
    p["defect_class"] = "DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS__SAME_RELATION"
    expect_third_fail(stage_third(sandbox3, doc=d), "says SAME_RELATION although the")


def test_negative_third_same_relation_token_dropped_where_the_triple_agrees_fails(sandbox3):
    d = tdoc()
    p = d["proposals"][trec("REL-EC021-CLS141")]
    p["defect_class"] = "DUPLICATE_REGISTER_PRIMARY_KEY__SAME_DRIVE_OBJECT__DIFFERENT_TEXT"
    expect_third_fail(stage_third(sandbox3, doc=d), "does not say SAME_RELATION although the")


def test_negative_third_same_object_flag_contradicting_the_source_url_cells_fails(sandbox3):
    d = tdoc()
    p = d["proposals"][trec("REL-132")]
    p["both_rows_cite_one_drive_object"] = True
    expect_third_fail(stage_third(sandbox3, doc=d), "both_rows_cite_one_drive_object is True but the 'Source URL' cells")


def test_negative_third_review_ledger_flag_contradicting_the_exact_object_cells_fails(sandbox3):
    d = tdoc()
    p = d["proposals"][trec("REV-P12-GP-006")]
    p["both_rows_cite_one_drive_object"] = True
    expect_third_fail(stage_third(sandbox3, doc=d), "both_rows_cite_one_drive_object is True but the 'Exact Object ID' cells")


def test_negative_third_review_ledger_drive_source_not_null_fails(sandbox3):
    d = tdoc()
    p = d["proposals"][trec("REV-P02-GP-INTERVAL-001")]
    p["drive_sources_cited"] = {"row_19": "https://example.invalid/a", "row_21": "https://example.invalid/b"}
    expect_third_fail(stage_third(sandbox3, doc=d), "drive_sources_cited must be null: review_ledger carries no URL column")


def test_negative_third_rule_cells_misdeclared_fails(sandbox3):
    d = tdoc()
    p = d["proposals"][trec("REL-049")]
    p["drive_object_rule_cells"] = ["Target URL"]
    expect_third_fail(stage_third(sandbox3, doc=d), "drive_object_rule_cells ['Target URL'] is not the rule this checker reads for 'relations'")


def test_negative_third_target_url_agreement_flag_flipped_fails(sandbox3):
    d = tdoc()
    p = d["proposals"][trec("REL-EC021-CLS141")]
    p["target_url_cells_agree"] = True
    expect_third_fail(stage_third(sandbox3, doc=d), "target_url_cells_agree is True but the 'Target URL' cells of rows 232 and 235 differ")


def test_negative_third_identity_cells_cited_falsified_fails(sandbox3):
    d = tdoc()
    p = d["proposals"][trec("DEF-049")]
    p["identity_cells_cited"]["Source URL"]["row_50"] = p["identity_cells_cited"]["Source URL"]["row_48"]
    expect_third_fail(stage_third(sandbox3, doc=d), "identity_cells_cited does not equal the rule's cells")


def test_negative_third_exact_duplicate_flag_flipped_fails(sandbox3):
    d = tdoc()
    d["proposals"][trec("REL-EC021-CLS141")]["exact_duplicate_row"] = True
    expect_third_fail(stage_third(sandbox3, doc=d), "exact_duplicate_row must be False")


def test_negative_third_materiality_miscount_fails(sandbox3):
    """Planted in the JSON and the companion alike, so only the number-word check catches it;
    the miscount uses a hyphenated total, which the checker must read as a number."""
    d = tdoc()
    old, new = "Thirteen of twenty-four cells agree", "Twelve of twenty-four cells agree"
    p = d["proposals"][trec("REV-P12-GP-006")]
    assert old in p["materiality"]
    p["materiality"] = p["materiality"].replace(old, new)
    expect_third_fail(stage_third(sandbox3, doc=d, md=tmd().replace(old, new)),
                      "says 'Twelve of twenty-four cells agree' but the live rows give 13 of 24")


def test_negative_third_summary_bucket_named_by_the_legacy_key_fails(sandbox3):
    d = tdoc()
    sc = d["summary_counts"]
    sc["same_drive_object_different_status_text"] = sc.pop("same_drive_object_different_cells")
    sc["same_drive_object_different_cells"] = 1
    expect_third_fail(stage_third(sandbox3, doc=d), "exactly one of")


def test_negative_third_chain_total_falsified_fails(sandbox3):
    d = tdoc()
    d["summary_counts"]["findings_covered_by_the_chain_together"] = 23
    expect_third_fail(stage_third(sandbox3, doc=d), "findings_covered_by_the_chain_together is 23 but the records' cells give 37")


def test_negative_third_two_document_total_key_on_a_chain_of_two_fails(sandbox3):
    d = tdoc()
    d["summary_counts"]["findings_covered_by_both_proposals_together"] = 21
    expect_third_fail(stage_third(sandbox3, doc=d), "not a count this checker recomputes")


def test_negative_third_pair_listed_under_the_wrong_class_fails(sandbox3):
    d = tdoc()
    block = d["classification_of_the_fourteen_pairs"]
    block["different_objects_one_identifier"].append(block["same_object_different_cells"].pop())
    expect_third_fail(stage_third(sandbox3, doc=d), "classification_of_the_fourteen_pairs.same_object_different_cells")


def test_negative_third_keeper_set_to_the_later_row_fails(sandbox3):
    d = tdoc()
    p = d["proposals"][trec("REV-P02-GP-INTERVAL-001")]
    p["keeper"]["register_row_index"], p["successor"]["register_row_index"] = 21, 19
    expect_third_fail(stage_third(sandbox3, doc=d), "keeper.register_row_index 21 must be the earlier quoted row 19")


def test_negative_third_markdown_summary_table_successor_id_changed_fails(sandbox3):
    old = "| `DEF-049@DEF-R50` |"
    assert tmd().count(old) == 1
    expect_third_fail(stage_third(sandbox3, md=tmd().replace(old, "| `DEF-049@DEF-R51` |")),
                      "summary-table row for DEF-049 must read rows '48, 50'")


def test_negative_third_markdown_verbatim_row_block_tampered_fails(sandbox3):
    old = '"AUTHOR-REPAIR-CANDIDATE","2026-07-21"]'
    assert tmd().count(old) == 1
    expect_third_fail(stage_third(sandbox3, md=tmd().replace(old, '"AUTHOR-REPAIR-DONE","2026-07-21"]')),
                      "row 39 of relations is not quoted as the canonical JSON of the live row")


def test_negative_third_without_successor_of_fails(sandbox3):
    d = tdoc()
    del d["successor_of"]
    del d["predecessor_chain"]
    d["summary_counts"].pop("findings_covered_by_the_chain_together")
    expect_third_fail(stage_third(sandbox3, doc=d), "must name its frozen predecessor in successor_of")


# ------------------------------------- repository prose against the live cells

def readme_text() -> str:
    return open(README_PATH, encoding="utf-8").read()


def findings_text() -> str:
    return open(FINDINGS_PATH, encoding="utf-8").read()


def test_third_prose_files_are_present_in_the_sandbox(sandbox3):
    """The prose controls below are only meaningful if the checker actually reads the two
    files; the fixture must therefore stage them."""
    root = stage_third(sandbox3)
    assert os.path.exists(os.path.join(root, "registers", "README.md"))
    assert os.path.exists(os.path.join(root, FINDINGS_REL))
    assert run_third(root).returncode == 0


def test_negative_third_readme_calling_a_non_duplicate_pair_an_exact_duplicate_fails(sandbox3):
    """The regression an adversarial verifier found: registers/README.md described
    REL-EC021-CLS141 as an exact duplicate row apart from its review date while the record
    three lines below recomputed six of fifteen cells differing.  Nothing read the README."""
    old = ("and `REL-EC021-CLS141`, one relation from one Drive document registered twice\n"
           "and not an exact duplicate row — six of fifteen cells differ: Exact scope,")
    assert readme_text().count(old) == 1
    bad = readme_text().replace(
        old, "and `REL-EC021-CLS141`, an exact duplicate row apart from its review date — Exact scope,")
    expect_third_fail(stage_third(sandbox3, readme=bad),
                      "registers/README.md calls REL-EC021-CLS141 an exact duplicate")


def test_negative_third_readme_cell_count_contradicting_the_live_rows_fails(sandbox3):
    old = "six of fifteen cells differ"
    assert readme_text().count(old) == 1
    expect_third_fail(stage_third(sandbox3, readme=readme_text().replace(old, "five of fifteen cells differ")),
                      "says 'five of fifteen cells differ' but the live rows give 6 of 15 cells differ")


def test_negative_third_findings_doc_calling_a_non_duplicate_pair_an_exact_duplicate_fails(sandbox3):
    old = "it is not the exact\nduplicate the finding text calls it"
    assert findings_text().count(old) == 1
    bad = findings_text().replace(old, "it is the exact\nduplicate the finding text calls it")
    expect_third_fail(stage_third(sandbox3, findings=bad),
                      "docs/FINDINGS_2026-09-18.md calls REL-EC021-CLS141 an exact duplicate")


def test_third_prose_accepts_a_denial_in_another_wording(sandbox3):
    """The check fires on an assertion the cells refute, not on the phrase: a differently
    worded denial must still pass, or the check would push prose into one fixed sentence."""
    old = "and not an exact duplicate row"
    assert readme_text().count(old) == 1
    r = run_third(stage_third(sandbox3, readme=readme_text().replace(old, "and it isn't an exact duplicate row")))
    assert r.returncode == 0, f"a reworded denial must pass:\n{r.stdout}"


def test_third_prose_naming_no_colliding_identifier_is_not_flagged(sandbox3):
    """The check is attribution-based, not a blanket grep: a sentence that names none of the
    fourteen identifiers is none of this checker's business."""
    bad = readme_text() + "\nSome other pair of rows is an exact duplicate row and four of nine cells differ.\n"
    r = run_third(stage_third(sandbox3, readme=bad))
    assert r.returncode == 0, f"an unattributed sentence must not be flagged:\n{r.stdout}"


def test_third_prose_count_is_attributed_to_the_nearest_identifier_named_before_it(sandbox3):
    """A count following a different identifier is checked against that identifier's record,
    so a correct number copied onto the wrong pair does not pass."""
    bad = readme_text() + "\nFor `REL-036`, six of fifteen cells differ.\n"
    expect_third_fail(stage_third(sandbox3, readme=bad),
                      "of REL-036 (record CP-REL-REL-036), says 'six of fifteen cells differ' "
                      "but the live rows give 13 of 15 cells differ")


def test_third_absent_prose_file_is_noted_not_silently_passed(sandbox3):
    """A tree without the prose file is skipped — and says so under -v, so a green run in a
    partial checkout is not mistaken for a cross-checked one."""
    root = stage_third(sandbox3)
    os.remove(os.path.join(root, "registers", "README.md"))
    r = run_third(root, "-v")
    assert r.returncode == 0, r.stdout
    assert "registers/README.md is not present" in r.stdout


# --------------------------------- the transcribed finding text, and its companion blockquote
# 'finding_text_as_recorded' is new in the third proposal: each record transcribes its
# finding's text from registers/KNOWN_FINDINGS.json and the companion blockquotes it under the
# attribution 'Finding text as recorded in `KNOWN_FINDINGS.json`:'.  An adversarial verifier
# found four mutations of that field, and one of the blockquote, that the checker did not see —
# including rewriting all fourteen texts from '...; nothing has been repaired.' to
# '...; everything has been repaired.'  The controls below drive each through the CLI.

ATTRIBUTION = "Finding text as recorded in `KNOWN_FINDINGS.json`:"


def tfinding_text(ident: str) -> str:
    return tdoc()["proposals"][trec(ident)]["finding_text_as_recorded"]


def test_third_transcribes_every_finding_text_and_the_companion_attributes_each(sandbox3):
    """The positive side of the control: all fourteen records transcribe, the companion
    attributes fourteen times, and the checker passes only when both hold."""
    d = tdoc()
    assert all("finding_text_as_recorded" in p for p in d["proposals"])
    assert tmd().count(ATTRIBUTION) == len(d["proposals"]) == 14
    assert run_third(stage_third(sandbox3)).returncode == 0


def test_negative_third_finding_text_fabricated_fails(sandbox3):
    d = tdoc()
    d["proposals"][trec("REL-036")]["finding_text_as_recorded"] = "fabricated text"
    expect_third_fail(stage_third(sandbox3, doc=d),
                      "CP-REL-REL-036: finding_text_as_recorded is not registers/KNOWN_FINDINGS.json "
                      "section 'findings_first_keyed_2026-09-19' entry")


def test_negative_third_finding_text_edited_by_one_word_fails(sandbox3):
    """The mutation the verifier ran: 'Two different relations' -> 'Two identical relations',
    a single word that reverses what the register recorded."""
    d = tdoc()
    i = trec("REL-036")
    old = d["proposals"][i]["finding_text_as_recorded"]
    assert old.count("Two different relations under one id.") == 1
    d["proposals"][i]["finding_text_as_recorded"] = old.replace(
        "Two different relations under one id.", "Two identical relations under one id.")
    expect_third_fail(stage_third(sandbox3, doc=d),
                      "CP-REL-REL-036: finding_text_as_recorded is not registers/KNOWN_FINDINGS.json")


def test_negative_third_finding_text_deleted_from_one_record_fails(sandbox3):
    d = tdoc()
    del d["proposals"][trec("DEF-049")]["finding_text_as_recorded"]
    expect_third_fail(stage_third(sandbox3, doc=d),
                      "finding_text_as_recorded is missing although 13 of 14 records of this document "
                      "transcribe the finding text")


def test_negative_third_every_finding_text_claiming_repair_fails(sandbox3):
    """The worst of the four: all fourteen transcriptions rewritten so the document tells its
    reader the register says everything has been repaired."""
    d = tdoc()
    n = 0
    for p in d["proposals"]:
        t = p["finding_text_as_recorded"]
        # The phrase need not be last: a rationale may close with a note of what
        # it said before a correction. It must be present, and inverting it must
        # be refused.
        assert "nothing has been repaired." in t
        p["finding_text_as_recorded"] = t.replace("nothing has been repaired.",
                                                  "everything has been repaired.")
        n += 1
    assert n == 14
    expect_third_fail(stage_third(sandbox3, doc=d), "finding_text_as_recorded is not registers/KNOWN_FINDINGS.json")


def test_negative_third_all_finding_texts_deleted_leaves_the_companion_attributing_nothing(sandbox3):
    """Deleting the field from every record does not buy silence: the companion still
    attributes fourteen blockquotes to the register, and the counts must agree."""
    d = tdoc()
    for p in d["proposals"]:
        del p["finding_text_as_recorded"]
    expect_third_fail(stage_third(sandbox3, doc=d),
                      "attributes text to registers/KNOWN_FINDINGS.json 14 time(s) but no record of the "
                      "document carries finding_text_as_recorded")


def test_negative_third_companion_blockquote_drifting_from_the_record_fails(sandbox3):
    """The companion presents the blockquote as the register's words; if it drifts from the
    record (which is itself held to the register), the companion is quoting nobody."""
    text = tfinding_text("REL-036")
    assert tmd().count(text) == 1
    bad = tmd().replace(text, text.replace("Two different relations under one id.",
                                           "Two identical relations under one id."))
    expect_third_fail(stage_third(sandbox3, md=bad),
                      "CP-REL-REL-036 does not quote its finding_text_as_recorded verbatim")


def test_negative_third_companion_dropping_one_attribution_fails(sandbox3):
    md = tmd()
    assert md.count(ATTRIBUTION) == 14
    bad = md.replace(ATTRIBUTION, "Finding text, paraphrased:", 1)
    assert bad.count(ATTRIBUTION) == 13
    expect_third_fail(stage_third(sandbox3, md=bad),
                      "carries 13 'Finding text as recorded in `KNOWN_FINDINGS.json`:' attribution(s) "
                      "but 14 record(s) transcribe the finding text")


def test_negative_third_companion_banner_inverted_fails(sandbox3):
    """The banner check used to be a bare substring search over the whole companion, which the
    fourteen transcriptions — each ending '...; nothing has been repaired.' — satisfied on
    their own.  Inverting the banner the reader meets first must fail."""
    md = tmd()
    lines = md.split("\n")
    assert "**Nothing has been repaired.**" in lines[3]
    lines[3] = lines[3].replace("**Nothing has been repaired.**", "Everything has been repaired.")
    bad = "\n".join(lines)
    assert "nothing has been repaired" in bad.lower(), "the transcriptions must still carry the phrase"
    expect_third_fail(stage_third(sandbox3, md=bad),
                      "does not carry the nothing-has-been-repaired banner "
                      "'**Nothing has been repaired.**' within its first 40 lines")


# ------------------------------------------ the document's own row-locator self-description

def test_negative_third_declared_row_locator_contradicting_the_issued_ids_fails(sandbox3):
    """source_of_record.row_locators told the reader which locator each tab's successor ids
    carry and was checked by nothing: the ids are held to TAB_LOCATOR, so the declaration could
    say anything."""
    d = tdoc()
    assert d["source_of_record"]["row_locators"]["relations"] == "REL"
    d["source_of_record"]["row_locators"]["relations"] = "RIX"
    expect_third_fail(stage_third(sandbox3, doc=d),
                      "source_of_record.row_locators['relations'] is 'RIX' but the successor identifiers "
                      "of that tab are checked against 'REL'")


def test_negative_third_declared_row_locator_for_an_unused_tab_still_checked(sandbox3):
    """A locator declared for a tab no record names is still the document describing itself,
    and is still held to the map the checker reads."""
    d = tdoc()
    d["source_of_record"]["row_locators"]["evidence_lineage"] = "EVIDENCE"
    expect_third_fail(stage_third(sandbox3, doc=d),
                      "source_of_record.row_locators['evidence_lineage'] is 'EVIDENCE', not the locator "
                      "this checker reads for that tab ('EVL')")


def test_existing_identifiers_refuses_to_resolve_its_own_directory():
    """CLAUDE.md records a default argument bound at import time silently re-checking the good
    graph in tools/claims_check.py.  existing_identifiers() is the one function whose job is to
    prove a successor id is new, so it must never resolve a path itself."""
    mod = cpc_module()
    with pytest.raises(TypeError):
        mod.existing_identifiers()
    with pytest.raises(ValueError):
        mod.existing_identifiers("")
    ids = mod.existing_identifiers(os.path.join(ROOT, "registers", "json"))
    assert "REL-036" in ids and "REV-P02-GP-INTERVAL-001" in ids
