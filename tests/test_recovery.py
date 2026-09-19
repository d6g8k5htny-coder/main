"""Tests for the exception-recovery store and its checker.

The positive test asserts that the real ``recovery/`` store passes.  Every other
test is a negative control: it builds a deliberately broken store in a temporary
directory and asserts that ``tools/recovery_check.py`` *rejects* it.  A checker
that cannot fail is not a check, so each control names the exact discipline it
is defending.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import recovery_check  # noqa: E402


# --------------------------------------------------------------------------
# fixtures: a minimal but valid store, which each control then breaks
# --------------------------------------------------------------------------

GOOD = b'{"ok": true}\n'
GOOD_SHA = hashlib.sha256(GOOD).hexdigest()
CAND = b"partially inflated bytes\n"
CAND_SHA = hashlib.sha256(CAND).hexdigest()


def _record(rid, outcome, stored, corroborated=True, missing=None):
    rec = {
        "record_id": rid,
        "class": "ARCHIVE_READ_FAILURE",
        "object": {"title": "fixture"},
        "attempted_routes": [{"route": "b", "attempt": "sibling carrier", "result": "found"}],
        "outcome": outcome,
        "stored_path": stored,
        "digest_corroboration": {"status": "CORROBORATED" if corroborated else "UNCORROBORATED"},
    }
    if missing:
        rec["exactly_what_is_missing"] = missing
    return rec


def make_store(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "recovery", "recovered"))
    os.makedirs(os.path.join(root, "recovery", "candidates"))
    os.makedirs(os.path.join(root, "quarantine"))

    rec_rel = f"recovery/recovered/good.json.{GOOD_SHA}.bin"
    cand_rel = f"recovery/candidates/maybe.txt.{CAND_SHA}.bin"
    open(os.path.join(root, rec_rel), "wb").write(GOOD)
    open(os.path.join(root, cand_rel), "wb").write(CAND)

    json.dump(
        {"as_of": "2026-09-17", "exclusions": [{"key": "Q-FIXTURE", "payload_sha256": "f" * 64}]},
        open(os.path.join(root, "quarantine", "EXCLUSIONS.json"), "w"),
    )

    ledger = {
        "schema": "q0-recovery-ledger/1.0",
        "reviewer_disclosure": {"session_family": "Anthropic", "organizational_independence_credit": 0},
        "does_not_establish": ["restoring bytes is not review"],
        "counts": {"by_outcome": {"RECOVERED": 1, "CANDIDATE": 1, "UNRECOVERABLE": 1}},
        "artifacts": [
            {"path": rec_rel, "sha256": GOOD_SHA, "bytes": len(GOOD), "store": "recovered",
             "corroboration": "CORROBORATED", "corroborating_source": "fixture index"},
            {"path": cand_rel, "sha256": CAND_SHA, "bytes": len(CAND), "store": "candidates",
             "corroboration": "UNCORROBORATED", "corroborating_source": None},
        ],
        "records": [
            _record("R-1", "RECOVERED", rec_rel),
            _record("R-2", "CANDIDATE", cand_rel, corroborated=False),
            _record("R-3", "UNRECOVERABLE", None, corroborated=False, missing="the whole 4,096-byte payload"),
        ],
    }
    json.dump(ledger, open(os.path.join(root, "recovery", "LEDGER.json"), "w"), indent=1)
    return root, ledger, rec_rel, cand_rel


def write_ledger(root, ledger):
    json.dump(ledger, open(os.path.join(root, "recovery", "LEDGER.json"), "w"), indent=1)


def failures_mentioning(root, needle):
    return [f for f in recovery_check.check(root) if needle in f]


# --------------------------------------------------------------------------
# positive: the real store, and the fixture store, both pass
# --------------------------------------------------------------------------

def test_real_recovery_store_passes():
    assert recovery_check.check(ROOT) == []


def test_real_store_exit_code_is_zero(capsys):
    assert recovery_check.main(["--root", ROOT]) == 0


def test_fixture_store_passes(tmp_path):
    root, _, _, _ = make_store(tmp_path)
    assert recovery_check.check(root) == []


# --------------------------------------------------------------------------
# negative control 1: a corrupted blob must be caught
# --------------------------------------------------------------------------

def test_corrupted_blob_is_caught(tmp_path):
    root, _, rec_rel, _ = make_store(tmp_path)
    with open(os.path.join(root, rec_rel), "wb") as f:
        f.write(GOOD.replace(b"true", b"fals"))  # same length, different bytes
    fails = failures_mentioning(root, "the bytes hash to")
    assert fails, "a blob whose content no longer matches the digest in its path must fail"
    assert recovery_check.main(["--root", root]) == 1


def test_truncated_blob_is_caught(tmp_path):
    root, _, rec_rel, _ = make_store(tmp_path)
    with open(os.path.join(root, rec_rel), "wb") as f:
        f.write(GOOD[:-1])
    assert failures_mentioning(root, "the bytes hash to")


def test_blob_without_digest_in_path_is_caught(tmp_path):
    root, ledger, rec_rel, _ = make_store(tmp_path)
    bad = "recovery/recovered/no_digest_here.bin"
    shutil.move(os.path.join(root, rec_rel), os.path.join(root, bad))
    ledger["artifacts"][0]["path"] = bad
    ledger["records"][0]["stored_path"] = bad
    write_ledger(root, ledger)
    assert failures_mentioning(root, "does not carry a <name>.<sha256>.bin digest")


# --------------------------------------------------------------------------
# negative control 2: a candidate promoted to recovered without corroboration
# --------------------------------------------------------------------------

def test_candidate_promoted_into_recovered_dir_is_caught(tmp_path):
    root, ledger, _, cand_rel = make_store(tmp_path)
    promoted = cand_rel.replace("recovery/candidates/", "recovery/recovered/")
    shutil.move(os.path.join(root, cand_rel), os.path.join(root, promoted))
    ledger["artifacts"][1]["path"] = promoted
    ledger["artifacts"][1]["store"] = "recovered"
    ledger["records"][1]["stored_path"] = promoted  # still outcome CANDIDATE, still uncorroborated
    write_ledger(root, ledger)
    fails = recovery_check.check(root)
    assert any("CANDIDATE stored under recovered/" in f for f in fails)
    assert any("only digest-corroborated bytes may live in recovered/" in f for f in fails)


def test_candidate_relabelled_recovered_without_corroboration_is_caught(tmp_path):
    """Changing the label without gaining a corroborating digest must not pass."""
    root, ledger, _, cand_rel = make_store(tmp_path)
    ledger["records"][1]["outcome"] = "RECOVERED"  # relabelled, still UNCORROBORATED, still in candidates/
    write_ledger(root, ledger)
    fails = recovery_check.check(root)
    assert any("digest_corroboration.status" in f for f in fails)
    assert any("stored outside recovered/" in f for f in fails)


def test_recovered_marked_uncorroborated_in_index_is_caught(tmp_path):
    root, ledger, _, _ = make_store(tmp_path)
    ledger["artifacts"][0]["corroboration"] = "UNCORROBORATED"
    write_ledger(root, ledger)
    assert failures_mentioning(root, "only digest-corroborated bytes may live in recovered/")


# --------------------------------------------------------------------------
# negative control 3: a quarantined digest must be refused
# --------------------------------------------------------------------------

def test_quarantined_digest_is_refused(tmp_path):
    root, _, _, _ = make_store(tmp_path)
    json.dump(
        {"as_of": "2026-09-17",
         "exclusions": [{"key": "Q-R17-FIXTURE", "class": "DEFECTIVE_SCOPE", "payload_sha256": GOOD_SHA}]},
        open(os.path.join(root, "quarantine", "EXCLUSIONS.json"), "w"),
    )
    fails = failures_mentioning(root, "appears in quarantine/EXCLUSIONS.json")
    assert fails, "a payload under logical quarantine must not be restored into the recovery store"


def test_quarantined_digest_found_in_free_text_identity_is_refused(tmp_path):
    """Exclusion rows name digests in an 'identity' string too; those count."""
    root, _, _, _ = make_store(tmp_path)
    json.dump(
        {"as_of": "2026-09-17",
         "exclusions": [{"key": "Q-R17-FIXTURE", "identity": f"{GOOD_SHA}; 13 bytes"}]},
        open(os.path.join(root, "quarantine", "EXCLUSIONS.json"), "w"),
    )
    assert failures_mentioning(root, "appears in quarantine/EXCLUSIONS.json")


def test_real_store_has_no_quarantined_payload():
    excluded = recovery_check.quarantined_digests(ROOT)
    assert excluded, "the real exclusion list should not be empty"
    stored = {
        os.path.basename(p).split(".")[-2]
        for p in recovery_check.list_blobs(ROOT, "recovery/recovered")
        + recovery_check.list_blobs(ROOT, "recovery/candidates")
    }
    assert stored, "the real store should not be empty"
    assert not (stored & excluded)


# --------------------------------------------------------------------------
# negative control 4: a record with no attempted route is not a result
# --------------------------------------------------------------------------

def test_record_without_attempted_route_is_caught(tmp_path):
    root, ledger, _, _ = make_store(tmp_path)
    ledger["records"][2]["attempted_routes"] = []
    write_ledger(root, ledger)
    assert failures_mentioning(root, "no attempted route recorded")


def test_route_missing_its_result_is_caught(tmp_path):
    root, ledger, _, _ = make_store(tmp_path)
    ledger["records"][0]["attempted_routes"] = [{"route": "b", "attempt": "looked"}]
    write_ledger(root, ledger)
    assert failures_mentioning(root, "must name a route, an attempt and a result")


def test_unrecoverable_must_name_what_is_missing(tmp_path):
    root, ledger, _, _ = make_store(tmp_path)
    del ledger["records"][2]["exactly_what_is_missing"]
    write_ledger(root, ledger)
    assert failures_mentioning(root, "without naming exactly what is missing")


def test_unrecoverable_may_not_carry_a_payload(tmp_path):
    root, ledger, rec_rel, _ = make_store(tmp_path)
    ledger["records"][2]["stored_path"] = rec_rel
    write_ledger(root, ledger)
    assert failures_mentioning(root, "UNRECOVERABLE but names a stored_path")


# --------------------------------------------------------------------------
# negative control 5: the independence discipline is data, not prose
# --------------------------------------------------------------------------

def test_nonzero_independence_credit_is_caught(tmp_path):
    root, ledger, _, _ = make_store(tmp_path)
    ledger["reviewer_disclosure"]["organizational_independence_credit"] = 1
    write_ledger(root, ledger)
    assert failures_mentioning(root, "independence_credit must be 0")


def test_missing_does_not_establish_is_caught(tmp_path):
    root, ledger, _, _ = make_store(tmp_path)
    ledger["does_not_establish"] = []
    write_ledger(root, ledger)
    assert failures_mentioning(root, "does NOT establish")


# --------------------------------------------------------------------------
# store/ledger agreement
# --------------------------------------------------------------------------

def test_unindexed_blob_is_caught(tmp_path):
    root, _, _, _ = make_store(tmp_path)
    stray = b"stray\n"
    p = f"recovery/recovered/stray.{hashlib.sha256(stray).hexdigest()}.bin"
    open(os.path.join(root, p), "wb").write(stray)
    assert failures_mentioning(root, "not listed in the ledger's artifacts index")


def test_indexed_but_absent_blob_is_caught(tmp_path):
    root, ledger, rec_rel, _ = make_store(tmp_path)
    os.remove(os.path.join(root, rec_rel))
    write_ledger(root, ledger)
    assert failures_mentioning(root, "not present on disk")


# --------------------------------------------------------------------------
# the real ledger's own shape
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_ledger():
    return json.load(open(os.path.join(ROOT, "recovery", "LEDGER.json"), encoding="utf-8"))


def test_every_named_exception_class_is_covered(real_ledger):
    """One current record per exception. Successors replace, they do not add."""
    counts = {}
    for rec in recovery_check.current_records(real_ledger):
        counts[rec["class"]] = counts.get(rec["class"], 0) + 1
    assert counts == {
        "EMPTY_NATIVE_BODY": 8,
        "READ_FAILED": 5,
        "ARCHIVE_READ_FAILURE": 3,
        "ENCODED_BLOCK_FAILURE": 15,
    }


def test_ledger_counts_match_the_exceptions_csv():
    """The ledger's enumeration must be the CSV's, not a remembered number."""
    import csv

    rows = list(csv.DictReader(open(os.path.join(ROOT, "drive", "source_map", "Exceptions.csv"), encoding="utf-8")))
    seen = {}
    for r in rows:
        seen[r["Type"]] = seen.get(r["Type"], 0) + 1
    ledger = json.load(open(os.path.join(ROOT, "recovery", "LEDGER.json"), encoding="utf-8"))
    declared = ledger["counts"]["enumerated_from_exceptions_csv"]
    assert declared["total_rows"] == len(rows)
    for cls in ("EMPTY_NATIVE_BODY", "READ_FAILED", "ARCHIVE_READ_FAILURE", "ENCODED_BLOCK_FAILURE"):
        assert declared[cls] == seen[cls]


def test_no_record_claims_a_review_verdict(real_ledger):
    banned = ("PASS_TECHNICAL", "DISCHARGED", "PROMOTED", "CLOSED", "APPROVED")
    for rec in real_ledger["records"]:
        blob = json.dumps(rec)
        for word in banned:
            assert f'"{word}"' not in blob, f"{rec['record_id']} must not carry a gate verdict"


def test_independence_is_recorded_as_zero_and_gates_stay_open(real_ledger):
    disc = real_ledger["reviewer_disclosure"]
    assert disc["organizational_independence_credit"] == 0
    assert "REMAINS OPEN" in disc["gate_status"]


# --------------------------------------------------------------------------
# negative control 6: successors are numbered, pinned and never in place
# --------------------------------------------------------------------------

def _succeed(root, ledger, idx, new_outcome="UNRECOVERABLE", suffix="-S1"):
    """Append a successor of ledger['records'][idx] and re-tally the counts."""
    pred = ledger["records"][idx]
    succ = dict(pred)
    succ["record_id"] = pred["record_id"] + suffix
    succ["supersedes"] = {"record_id": pred["record_id"], "record_sha256": recovery_check.record_digest(pred)}
    succ["outcome"] = new_outcome
    succ["stored_path"] = None
    succ["digest_corroboration"] = {"status": "NOT_APPLICABLE"}
    succ["exactly_what_is_missing"] = "everything, in this fixture"
    ledger["records"].append(succ)
    tally = {}
    for rec in recovery_check.current_records(ledger):
        tally[rec["outcome"]] = tally.get(rec["outcome"], 0) + 1
    ledger["counts"]["by_outcome"] = tally
    write_ledger(root, ledger)
    return succ


def test_fixture_with_a_pinned_successor_passes(tmp_path):
    root, ledger, _, _ = make_store(tmp_path)
    _succeed(root, ledger, 2)  # R-3 (UNRECOVERABLE) -> R-3-S1 (still UNRECOVERABLE)
    assert recovery_check.check(root) == []


def test_control_editing_a_superseded_record_in_place_is_caught(tmp_path):
    root, ledger, _, _ = make_store(tmp_path)
    _succeed(root, ledger, 2)
    ledger["records"][2]["notes"] = ["quietly rewritten after being superseded"]
    write_ledger(root, ledger)
    assert failures_mentioning(root, "frozen and may only be succeeded, never edited")


def test_control_a_successor_pinning_nothing_is_caught(tmp_path):
    root, ledger, _, _ = make_store(tmp_path)
    succ = _succeed(root, ledger, 2)
    succ["supersedes"] = {"record_id": "R-3"}  # no pin
    write_ledger(root, ledger)
    assert failures_mentioning(root, "must name the predecessor's record_id and record_sha256")


def test_control_a_successor_of_a_record_not_in_the_ledger_is_caught(tmp_path):
    root, ledger, _, _ = make_store(tmp_path)
    succ = _succeed(root, ledger, 2)
    succ["supersedes"]["record_id"] = "R-9"
    succ["record_id"] = "R-9-S1"
    write_ledger(root, ledger)
    assert failures_mentioning(root, "which is not in the ledger")


def test_control_a_record_superseded_twice_is_caught(tmp_path):
    root, ledger, _, _ = make_store(tmp_path)
    _succeed(root, ledger, 2, suffix="-S1")
    _succeed(root, ledger, 2, suffix="-S2")  # a second successor of the same predecessor
    assert failures_mentioning(root, "superseded twice")


def test_control_a_successor_id_must_extend_its_predecessors_id(tmp_path):
    root, ledger, _, _ = make_store(tmp_path)
    succ = _succeed(root, ledger, 2)
    succ["record_id"] = "R-4"
    write_ledger(root, ledger)
    assert failures_mentioning(root, "followed by -S<n>")


def test_control_a_remembered_count_is_caught(tmp_path):
    """counts.by_outcome is recomputed from the current records, not trusted."""
    root, ledger, _, _ = make_store(tmp_path)
    ledger["counts"]["by_outcome"]["RECOVERED"] = 2
    write_ledger(root, ledger)
    assert failures_mentioning(root, "is not the tally of current records")


def test_control_a_successor_does_not_add_to_the_exception_count(tmp_path):
    root, ledger, _, _ = make_store(tmp_path)
    ledger["counts"]["exception_records"] = 3
    _succeed(root, ledger, 2)
    assert recovery_check.check(root) == []
    ledger["counts"]["exception_records"] = 4  # counted the successor as a new exception
    write_ledger(root, ledger)
    assert failures_mentioning(root, "is not the number of current records")


# --------------------------------------------------------------------------
# the real successors of 2026-09-19, reproduced from their own inputs
# --------------------------------------------------------------------------

def _halve_blank_lines(display_bytes: bytes) -> bytes:
    """The reading volumes show every blank line twice: a run of k newlines is
    displayed as 2k-1. Invert it, then the 2026-09-18 rule (strip trailing
    newlines, append exactly one LF)."""
    import re

    text = display_bytes.decode("utf-8")
    text = re.sub(r"\n{2,}", lambda m: "\n" * ((len(m.group(0)) + 1) // 2), text)
    return (text.rstrip("\n") + "\n").encode("utf-8")


def test_real_successors_pin_their_predecessors_and_leave_them_unchanged(real_ledger):
    by_id = {r["record_id"]: r for r in real_ledger["records"]}
    succ = [r for r in real_ledger["records"] if r.get("supersedes")]
    assert {r["record_id"] for r in succ} == {"ENB-04-S1", "RDF-01-S1", "RDF-02-S1"}
    for r in succ:
        pred = by_id[r["supersedes"]["record_id"]]
        assert recovery_check.record_digest(pred) == r["supersedes"]["record_sha256"]
    # the predecessors still say what they said on 2026-09-18
    assert by_id["ENB-04"]["outcome"] == "UNRECOVERABLE"
    assert by_id["RDF-01"]["outcome"] == "CANDIDATE" and by_id["RDF-02"]["outcome"] == "CANDIDATE"
    # and the candidate blobs were not deleted when they were superseded
    for rid in ("RDF-01", "RDF-02"):
        assert os.path.exists(os.path.join(ROOT, by_id[rid]["stored_path"]))


def test_the_two_markdown_recoveries_follow_from_the_candidate_bytes(real_ledger):
    """Recompute the recovery from the 2026-09-18 candidate, not from memory."""
    by_id = {r["record_id"]: r for r in real_ledger["records"]}
    for rid in ("RDF-01", "RDF-02"):
        cand = open(os.path.join(ROOT, by_id[rid]["stored_path"]), "rb").read()
        succ = by_id[rid + "-S1"]
        recovered = open(os.path.join(ROOT, succ["stored_path"]), "rb").read()
        expected = by_id[rid]["digest_corroboration"]["expected_sha256"]
        assert _halve_blank_lines(cand) == recovered
        assert hashlib.sha256(recovered).hexdigest() == expected
        assert len(recovered) == by_id[rid]["digest_corroboration"]["expected_bytes"]
        assert succ["outcome"] == "RECOVERED" and succ["digest_corroboration"]["status"] == "CORROBORATED"
    # the same rule changes nothing that has no blank line: the three plain-text
    # payloads of the same class were already exact on 2026-09-18
    for rid in ("RDF-03", "RDF-04", "RDF-05"):
        rec = by_id[rid]
        stored = open(os.path.join(ROOT, rec["stored_path"]), "rb").read()
        assert rec["outcome"] == "RECOVERED" and b"\n\n" not in stored
        assert _halve_blank_lines(stored) == stored
    # a candidate with a different failure mode is not "fixed" by the rule
    arf = by_id["ARF-03"]
    cand = open(os.path.join(ROOT, arf["stored_path"]), "rb").read()
    assert hashlib.sha256(_halve_blank_lines(cand)).hexdigest() != arf["digest_corroboration"]["expected_sha256"]
    assert arf["outcome"] == "CANDIDATE"


def test_the_ls_data_013_members_come_from_the_mirrored_carrier_and_match_the_source_map(real_ledger):
    import csv
    import zipfile

    by_id = {r["record_id"]: r for r in real_ledger["records"]}
    rec = by_id["ENB-04-S1"]
    assert rec["outcome"] == "UNRECOVERABLE" and rec["stored_path"] is None
    members = {
        os.path.basename(a["path"]).rsplit(".", 2)[0]: a["path"] for a in rec["related_artifacts"]
    }
    assert set(members) == {
        "LS-DATA-013-v1.0_q0_law_specific_verify_v1_2_1.py",
        "LS-DATA-013-v1.0_q0_law_specific_verify_v1_2_1_report.json",
    }
    rows = {}
    with open(os.path.join(ROOT, "drive", "source_map", "Archive_Members.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Carrier ID"] == "1WZfhLuZBEzvdUUkQw7v5JubmAiI1W4gt" and "LS-DATA-013" in r["Member path"]:
                rows[r["Member path"]] = r
    assert len(rows) == 2
    zpath = os.path.join(
        ROOT, "drive", "mirrors", "2026-09-16 \u2014 HOLD_NOT_FOR_SUBMISSION",
        "CLOSE-20260917-b9c2_PROOFS_CODE_AND_VERIFICATION.zip",
    )
    with zipfile.ZipFile(zpath) as z:
        for member, row in rows.items():
            data = z.read(member)
            assert hashlib.sha256(data).hexdigest() == row["Payload SHA-256"]
            assert len(data) == int(row["Bytes"])
            stored = open(os.path.join(ROOT, members[os.path.basename(member)]), "rb").read()
            assert stored == data


def test_no_recovered_python_is_wired_into_the_engine():
    """A recovered verifier is bytes in the ledger, not code the repository runs."""
    import glob

    blobs = [b for b in glob.glob(os.path.join(ROOT, "recovery", "recovered", "*.py.*.bin"))]
    assert blobs
    hay = ""
    for path in glob.glob(os.path.join(ROOT, "engine", "**", "*.py"), recursive=True) + glob.glob(
        os.path.join(ROOT, ".github", "workflows", "*.yml")
    ):
        hay += open(path, encoding="utf-8", errors="replace").read()
    for b in blobs:
        assert os.path.basename(b) not in hay
