"""Rules of the architectural-admission attestation form, and a negative control
for every one of them.

What this does NOT establish: these tests check that the checker REFUSES what it
must refuse. They verify no mathematics, admit no object, move no gate and award
no independence. The attestation bridge they belong to is NOT DEPLOYED.

Each control mutates a copy of the real record and asserts the checker fails on
it. The checker is invoked through its command-line flags in a subprocess, never
imported and called with module globals already bound: a default argument that
binds a path at import time is exactly the defect that once made every mutation
test in this repository silently re-check the good input.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "attestations_check.py")
SCHEMA = os.path.join(ROOT, "attestations", "attestation.schema.json")
RECORDS = os.path.join(ROOT, "attestations", "records")
GOOD = os.path.join(RECORDS, "ATT-RN-INNER-WEDGE-20260925.json")


def run_checker(records_dir):
    return subprocess.run(
        [sys.executable, CHECKER, "--records", str(records_dir), "--schema", SCHEMA],
        capture_output=True, text=True, cwd=ROOT,
    )


def good_record():
    with open(GOOD, encoding="utf-8") as handle:
        return json.load(handle)


def write(tmp_path, record, stem="ATT-RN-INNER-WEDGE-20260925"):
    path = tmp_path / f"{stem}.json"
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return tmp_path


def refuse(tmp_path, record, needle, stem="ATT-RN-INNER-WEDGE-20260925"):
    result = run_checker(write(tmp_path, record, stem))
    assert result.returncode == 1, result.stdout
    assert needle in result.stdout, result.stdout
    return result.stdout


# --------------------------------------------------------------------------- #
# The positive side                                                            #
# --------------------------------------------------------------------------- #

def test_the_real_records_pass():
    result = run_checker(RECORDS)
    assert result.returncode == 0, result.stdout
    assert "no grade changed, no gate moved" in result.stdout


def test_checker_reports_and_summarises(tmp_path):
    result = run_checker(write(tmp_path, good_record()))
    assert result.returncode == 0
    assert "1 attestation(s)" in result.stdout


def test_unknown_flag_is_refused():
    result = subprocess.run([sys.executable, CHECKER, "--nope"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 2


# --------------------------------------------------------------------------- #
# Fail-closed admission                                                        #
# --------------------------------------------------------------------------- #

def test_admitted_with_a_failed_check_is_refused(tmp_path):
    record = good_record()
    record["checks"][2]["exit_code"] = 1
    refuse(tmp_path, record, "forbids admission")


def test_admitted_with_every_check_failing_is_refused(tmp_path):
    record = good_record()
    for check in record["checks"]:
        check["exit_code"] = 1
    refuse(tmp_path, record, "forbids admission")


def test_refused_without_a_failing_check_is_refused(tmp_path):
    record = good_record()
    record["outcome"] = "REFUSED"
    refuse(tmp_path, record, "must name the check that refused it")


def test_a_genuine_refusal_is_accepted(tmp_path):
    record = good_record()
    record["outcome"] = "REFUSED"
    record["checks"][0]["exit_code"] = 1
    result = run_checker(write(tmp_path, record))
    assert result.returncode == 0, result.stdout


def test_empty_checks_is_refused(tmp_path):
    record = good_record()
    record["checks"] = []
    refuse(tmp_path, record, "ATT-RN-INNER-WEDGE-20260925.json")


# --------------------------------------------------------------------------- #
# The pins that stop an attestation becoming a promotion                       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", ["PROMOTED", "RAISED", "CLOSED", ""])
def test_a_moved_claim_grade_is_refused(tmp_path, value):
    record = good_record()
    record["claim_grade_after"] = value
    refuse(tmp_path, record, "claim_grade_after")


@pytest.mark.parametrize("value", ["SATISFIED", "OPENED", "MOVED"])
def test_a_moved_gate_status_is_refused(tmp_path, value):
    record = good_record()
    record["gate_status_after"] = value
    refuse(tmp_path, record, "gate_status_after")


@pytest.mark.parametrize("value", [1, 2])
def test_nonzero_independence_credit_is_refused(tmp_path, value):
    record = good_record()
    record["independence_credit"] = value
    refuse(tmp_path, record, "independence_credit")


def test_admitted_with_empty_awaiting_is_refused(tmp_path):
    record = good_record()
    record["awaiting"] = []
    refuse(tmp_path, record, "ATT-RN-INNER-WEDGE-20260925.json")


# --------------------------------------------------------------------------- #
# Free-text scans                                                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("phrase", [
    "the premise is now settled",
    "this discharged the obligation",
    "it closed the remaining gap",
    "this promotes the candidate",
])
def test_promotion_language_is_refused(tmp_path, phrase):
    record = good_record()
    record["notes"] = phrase
    refuse(tmp_path, record, "promotion language")


@pytest.mark.parametrize("phrase", [
    "high confidence in the wedge",
    "95% confident the bound holds",
    "a confidence score of nine",
])
def test_confidence_voting_is_refused(tmp_path, phrase):
    record = good_record()
    record["notes"] = phrase
    refuse(tmp_path, record, "confidence voting")


# --------------------------------------------------------------------------- #
# Identity, shape and anti-boilerplate                                         #
# --------------------------------------------------------------------------- #

def test_an_unknown_field_is_refused(tmp_path):
    record = good_record()
    record["approved_by"] = "nobody"
    refuse(tmp_path, record, "ATT-RN-INNER-WEDGE-20260925.json")


def test_id_not_matching_the_filename_is_refused(tmp_path):
    record = good_record()
    refuse(tmp_path, record, "does not match the filename stem", stem="ATT-OTHER-NAME")


def test_a_short_does_not_establish_is_refused(tmp_path):
    record = good_record()
    record["does_not_establish"] = "It establishes nothing at all. " * 8
    refuse(tmp_path, record, "distinct token")


def test_a_copied_does_not_establish_is_refused(tmp_path):
    first = good_record()
    second = copy.deepcopy(first)
    second["attestation_id"] = "ATT-SECOND-RECORD"
    write(tmp_path, first)
    write(tmp_path, second, stem="ATT-SECOND-RECORD")
    result = run_checker(tmp_path)
    assert result.returncode == 1
    assert "verbatim identical" in result.stdout, result.stdout


@pytest.mark.parametrize("digest", ["", "abc", "Z" * 64])
def test_a_malformed_object_digest_is_refused(tmp_path, digest):
    record = good_record()
    record["object_sha256"] = digest
    refuse(tmp_path, record, "ATT-RN-INNER-WEDGE-20260925.json")


def test_zero_object_bytes_is_refused(tmp_path):
    record = good_record()
    record["object_bytes"] = 0
    refuse(tmp_path, record, "ATT-RN-INNER-WEDGE-20260925.json")


def test_a_missing_records_directory_is_reported(tmp_path):
    result = run_checker(tmp_path / "absent")
    assert result.returncode == 1
    assert "does not exist" in result.stdout


# --------------------------------------------------------------------------- #
# The object's current bytes                                                   #
# --------------------------------------------------------------------------- #

def test_a_stale_digest_on_a_live_path_is_refused(tmp_path):
    """The rule that caught this suite's own first real defect: PR #19 appended a
    banner to an attested document, and nothing noticed until this existed."""
    record = good_record()
    record["object_sha256"] = "0" * 64
    refuse(tmp_path, record, "re-run the gates and attest the current bytes")


def test_a_stale_byte_count_on_a_live_path_is_refused(tmp_path):
    record = good_record()
    record["object_bytes"] = record["object_bytes"] + 1
    refuse(tmp_path, record, "re-run the gates and attest the current bytes")


def test_superseded_by_exempts_a_historical_record(tmp_path):
    record = good_record()
    record["object_sha256"] = "0" * 64
    record["superseded_by"] = "ATT-SUCCESSOR-RECORD"
    successor = good_record()
    successor["attestation_id"] = "ATT-SUCCESSOR-RECORD"
    successor["does_not_establish"] = (
        "This successor record establishes nothing whatever about correctness, soundness, "
        "sharpness or acceptance of any mathematical statement anywhere in the programme; it "
        "exists purely so that its predecessor may be retired honestly rather than quietly "
        "edited, and it awards no independence credit to anybody under any circumstances."
    )
    write(tmp_path, record)
    write(tmp_path, successor, stem="ATT-SUCCESSOR-RECORD")
    result = run_checker(tmp_path)
    assert result.returncode == 0, result.stdout


def test_superseded_by_naming_a_missing_record_is_refused(tmp_path):
    record = good_record()
    record["object_sha256"] = "0" * 64
    record["superseded_by"] = "ATT-DOES-NOT-EXIST"
    refuse(tmp_path, record, "is not a record in this directory")


def test_an_object_path_absent_from_the_tree_is_not_checked(tmp_path):
    """A Drive id or a path outside this checkout cannot be byte-compared, and the
    rule must not invent a failure for one."""
    record = good_record()
    record["object_id"] = "drive:1abcDEFghiJKLmnoPQRstu @ 0123456789abcdef"
    record["object_sha256"] = "0" * 64
    result = run_checker(write(tmp_path, record))
    assert result.returncode == 0, result.stdout


# --------------------------------------------------------------------------- #
# What the form itself must keep saying                                        #
# --------------------------------------------------------------------------- #

def test_schema_pins_the_three_invariants():
    with open(SCHEMA, encoding="utf-8") as handle:
        schema = json.load(handle)
    assert schema["properties"]["claim_grade_after"]["const"] == "UNCHANGED"
    assert schema["properties"]["gate_status_after"]["const"] == "UNCHANGED"
    assert schema["properties"]["independence_credit"]["const"] == 0
    assert schema["additionalProperties"] is False
    assert schema["properties"]["awaiting"]["minItems"] == 1


def test_checker_docstring_states_what_it_does_not_establish():
    import ast
    with open(CHECKER, encoding="utf-8") as handle:
        doc = ast.get_docstring(ast.parse(handle.read())) or ""
    assert "not establish" in doc.lower()
    assert "not deployed" in doc.lower()
