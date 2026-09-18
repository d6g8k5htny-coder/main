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
