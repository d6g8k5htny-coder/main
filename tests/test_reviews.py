"""The R17 §4 review form, with negative controls.

`tools/reviews_check.py` passing on an empty directory proves nothing. Every
test below builds a fixture record, breaks it in exactly one way a reviewer
under time pressure might break it, and asserts that the checker rejects that
break and says which rule fired.

The fixtures are FIXTURES. They are written to a temporary directory, never to
`reviews/records/`, and they are not reviews of the objects whose route keys
they borrow: no object was read to produce them, and their prose says so. Their
route keys are real only so that the register-membership rule is exercised
against the real `registers/json/review_queue.json`.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "reviews_check.py")
SCHEMA = os.path.join(ROOT, "reviews", "review_record.schema.json")
QUEUE = os.path.join(ROOT, "registers", "json", "review_queue.json")

# Body SHA-256 as the exported Review Queue tab carries them.
RN3_SHA = "0c9446b7cb49e6e1e45e43f84bb72ce680f022882e225fdff61c8e6edede1373"
LM009_SHA = "ccc07d959428519733c7fdf6c72f1bce63a40d0b0dc956ce0fe9f2ec459eb0b1"

EXPOSURE_A = (
    "Fixture exposure statement. Before this run the session had read the "
    "repository README, docs/RESEARCH_MAP.md, docs/OPEN_PROBLEMS.md section D, "
    "the exported Review Queue tab including the row whose key this fixture "
    "borrows, the frozen v2.2 premise table, and the ERRATA note withdrawing "
    "the two-track composition. It had not opened the RN3 bundle body, the "
    "rnu_ds3 carrier, or any far-zone worksheet. Neighbouring material seen "
    "earlier: the RN5 determinant-moment defect write-up and its regression "
    "test, which shapes what a reviewer expects an envelope argument to look "
    "like and is therefore disclosed here as possible anchoring."
)
EXPOSURE_B = (
    "Second fixture exposure statement, deliberately different wording. This "
    "session had previously seen the lemma stack summary table, the register "
    "row listing LCR-DER-027 with its byte count, and the contribution plan "
    "paragraph describing which component verdicts precede the synthesis "
    "route. It had never opened the LCR-DER-027 document body, its "
    "predecessor version, or the requirement sheet that governs it. No "
    "reviewer notes from any other provider were available, and no summary of "
    "anyone else's verdict was consulted while forming this fixture text."
)
DNE_A = (
    "This fixture establishes nothing whatever about RN3, its far-zone "
    "bounds, or any premise resting on them. It is a shape test for the "
    "checker. It does not verify mathematics, does not award organizational "
    "independence, does not move OBL-H5-JETMOD or D3-LEMMA-RN-UNIF, and does "
    "not license anyone to treat the route as answered."
)
DNE_B = (
    "Nothing about LCR-DER-027 follows from this fixture: not its selected "
    "branch capture, not its uniformity claims, and not its standing in the "
    "lemma stack. The record exists to exercise duplicate detection and the "
    "schema floors. Every obligation named anywhere in the register remains "
    "exactly where the register puts it."
)


def base_record(review_id="REV-FIXTURE-A", route_key="RV-RN3"):
    return {
        "schema_version": "1.0.0",
        "review_id": review_id,
        "route_key": route_key,
        "review_utc": "2026-09-18T12:00:00Z",
        "object_id": "1FixtureDriveIdPlaceholder0000000000000",
        "object_title": "RN3 joint far-zone bundle (fixture stand-in)",
        "object_bytes": 12956,
        "object_sha256": RN3_SHA,
        "extraction_rule": ("Native Google Doc exported as text/plain at the "
                            "named revision id, decoded UTF-8, SHA-256 over the "
                            "exported bytes."),
        "obtained": True,
        "obtained_how": ("Fixture: no download was performed. In a real record "
                         "this names the download call, the revision id and the "
                         "local path of the retained bytes."),
        "reviewer_family": "anthropic",
        "reviewer_session": "fixture-session-0000",
        "author_family": "openai",
        "author_family_determination": ("Read from the Review Queue 'Author / "
                                        "provider' column and the attribution "
                                        "line inside the body."),
        "exposure_disclosure": EXPOSURE_A,
        "hypotheses": [
            "H-fixture-1: the far-zone estimate is stated at fixed separation "
            "r = 1/20 and is not extended to all small r.",
            "H-fixture-2: the cover is finite and every cell is accounted for.",
        ],
        "reconstruction": (
            "Fixture reconstruction text, long enough to clear the schema floor "
            "and structured the way a real one must be. A real reconstruction "
            "walks the object's argument in the reviewer's own words: it names "
            "the decomposition, restates each bound with its constants, follows "
            "the summation to the stated conclusion, and marks every step the "
            "reviewer could not follow from the bytes alone. It says where the "
            "object's own numbering diverges from the register's description, "
            "and it distinguishes what the body proves from what the body "
            "asserts. This fixture proves nothing and reconstructs nothing; it "
            "stands in for that prose so the checker's length floor can be "
            "exercised without inventing a verdict about a real object."
        ),
        "negative_controls_executed": [
            {
                "control": "Recompute the digest of the retained bytes before "
                           "and after the reading pass.",
                "would_have_caught": "A silently substituted or truncated body.",
                "fired": False,
            },
            {
                "control": "Substitute a deliberately wrong exponent into the "
                           "reconstructed chain and re-derive.",
                "would_have_caught": "A reconstruction that follows the object's "
                                     "wording rather than its arithmetic.",
                "fired": True,
                "detail": "Fixture: recorded as fired to exercise the field.",
            },
        ],
        "execution_performed": "None; this is a fixture.",
        "findings": [
            {
                "criterion": "scope",
                "severity": "INFO",
                "statement": "Fixture finding; carries no verdict about any "
                             "real object.",
            }
        ],
        "unresolved_dependencies": ["D3-LEMMA-RN-UNIF Piece 2 (fixture note)"],
        "technical_verdict": "CANNOT_VERIFY",
        "scope_dependency_verdict": ("Fixture: scope not assessed, dependencies "
                                     "not assessed, nothing inherited."),
        "independence_credit": 0,
        "independence_reason": ("Reviewer family is anthropic; R17 §4 records a "
                                "same-provider reviewer at zero organizational "
                                "independence, and the author lineage here is a "
                                "different family only nominally."),
        "gate_status_after": "UNCHANGED",
        "does_not_establish": DNE_A,
        "reproducible_output": ["python3 -m pytest -q tests/test_reviews.py"],
    }


def second_record():
    record = base_record("REV-FIXTURE-B", "RV-LM009-MAIN")
    record["object_title"] = "LCR-DER-027-v1.0 (fixture stand-in)"
    record["object_bytes"] = 3919
    record["object_sha256"] = LM009_SHA
    record["exposure_disclosure"] = EXPOSURE_B
    record["does_not_establish"] = DNE_B
    return record


def run(records, extra_files=None):
    """Write records to a temp dir and run the checker. Returns (rc, output)."""
    with tempfile.TemporaryDirectory() as d:
        for record in records:
            name = f"{record.get('review_id', 'unnamed')}.json"
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        for name, blob in (extra_files or {}).items():
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write(blob)
        proc = subprocess.run(
            [sys.executable, CHECKER, "--records", d, "--schema", SCHEMA,
             "--queue", QUEUE],
            capture_output=True, text=True)
        return proc.returncode, proc.stdout + proc.stderr


# --------------------------------------------------------------------------- #
# Positive controls                                                            #
# --------------------------------------------------------------------------- #

def test_committed_records_directory_passes():
    """Whatever is actually in reviews/records/ must satisfy the form."""
    proc = subprocess.run([sys.executable, CHECKER], capture_output=True,
                          text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_valid_fixture_passes():
    rc, out = run([base_record()])
    assert rc == 0, out


def test_two_distinct_fixtures_pass():
    rc, out = run([base_record(), second_record()])
    assert rc == 0, out


def test_cannot_verify_with_obtained_false_is_permitted():
    """The honest 'I could not get the bytes' record must be legal."""
    r = base_record()
    r["obtained"] = False
    r["object_sha256"] = ""
    r["object_bytes"] = 0
    r["obtained_how"] = ("Download returned a permission error on the named "
                         "Drive id; no archive carrier holds a member with this "
                         "path; nothing else was tried.")
    r["technical_verdict"] = "CANNOT_VERIFY"
    r["sha_mismatch_explanation"] = ""
    rc, out = run([r])
    assert rc == 0, out


# --------------------------------------------------------------------------- #
# Negative controls — one rule each                                            #
# --------------------------------------------------------------------------- #

def test_rejects_independence_credit_one_for_anthropic_reviewer():
    r = base_record()
    r["independence_credit"] = 1
    r["independence_evidence"] = "Fixture claim of independence."
    rc, out = run([r])
    assert rc == 1
    assert "independence_credit" in out


def test_rejects_gate_status_after_discharged():
    r = base_record()
    r["gate_status_after"] = "DISCHARGED"
    rc, out = run([r])
    assert rc == 1
    assert "gate_status_after" in out


def test_rejects_any_gate_status_other_than_unchanged():
    for value in ("OPEN", "CLOSED", "ADVANCED", "unchanged", ""):
        r = base_record()
        r["gate_status_after"] = value
        rc, out = run([r])
        assert rc == 1, f"{value!r} was accepted: {out}"
        assert "gate_status_after" in out


def test_rejects_obtained_false_with_pass_technical():
    r = base_record()
    r["obtained"] = False
    r["object_sha256"] = ""
    r["technical_verdict"] = "PASS_TECHNICAL"
    rc, out = run([r])
    assert rc == 1
    assert "CANNOT_VERIFY" in out


def test_rejects_obtained_true_without_a_digest():
    r = base_record()
    r["object_sha256"] = ""
    rc, out = run([r])
    assert rc == 1
    assert "object_sha256" in out


def test_rejects_promotion_language_in_a_findings_string():
    r = base_record()
    r["findings"][0]["statement"] = (
        "The reconstruction shows the normalizer premise is now settled.")
    rc, out = run([r])
    assert rc == 1
    assert "premise is now" in out
    assert "findings[0].statement" in out


def test_rejects_promotion_language_in_every_free_text_field():
    for field, text in (
        ("notes", "This review discharged the remaining obligation."),
        ("scope_dependency_verdict", "The scope check closed the chart-side gap."),
        ("reconstruction", "The argument promotes the rung to all small r. " * 12),
        ("independence_reason", "Same family, but independence satisfied here."),
        ("execution_performed", "Ran the replay; gate satisfied afterwards."),
    ):
        r = base_record()
        r[field] = text
        rc, out = run([r])
        assert rc == 1, f"{field} accepted promotion language: {out}"
        assert f"field $.{field}" in out


def test_rejects_unknown_route_key():
    r = base_record()
    r["route_key"] = "RV-NOT-A-REAL-ROUTE"
    rc, out = run([r])
    assert rc == 1
    assert "route_key" in out


def test_rejects_duplicated_boilerplate_does_not_establish():
    a, b = base_record(), second_record()
    b["does_not_establish"] = a["does_not_establish"]
    rc, out = run([a, b])
    assert rc == 1
    assert "does_not_establish" in out


def test_rejects_duplicated_exposure_disclosure():
    a, b = base_record(), second_record()
    b["exposure_disclosure"] = a["exposure_disclosure"]
    rc, out = run([a, b])
    assert rc == 1
    assert "exposure_disclosure" in out


def test_rejects_short_or_padded_disclosures():
    r = base_record()
    r["exposure_disclosure"] = "None."
    rc, out = run([r])
    assert rc == 1
    r = base_record()
    r["does_not_establish"] = "It does not establish anything. " * 20
    rc, out = run([r])
    assert rc == 1
    assert "distinct tokens" in out


def test_rejects_verdict_outside_the_register_vocabulary():
    for verdict in ("APPROVED", "PASS", "OK", "pass_technical", "CLOSED"):
        r = base_record()
        r["technical_verdict"] = verdict
        rc, out = run([r])
        assert rc == 1, f"{verdict!r} accepted: {out}"
        assert "technical_verdict" in out


def test_rejects_digest_disagreeing_with_the_register_row():
    r = base_record()
    r["object_sha256"] = "a" * 64
    rc, out = run([r])
    assert rc == 1
    assert "sha_mismatch_explanation" in out


def test_accepts_explained_digest_disagreement():
    r = base_record()
    r["object_sha256"] = "a" * 64
    r["sha_mismatch_explanation"] = (
        "Fixture: the register row names the bundle body while this fixture "
        "carries a placeholder digest; a real record names the successor "
        "version or the differing extraction rule here.")
    rc, out = run([r])
    assert rc == 0, out


def test_rejects_empty_negative_controls_without_a_reason():
    r = base_record()
    r["negative_controls_executed"] = []
    rc, out = run([r])
    assert rc == 1
    assert "negative_controls" in out


def test_accepts_empty_negative_controls_with_a_stated_reason():
    r = base_record()
    r["negative_controls_executed"] = []
    r["negative_controls_not_applicable_reason"] = (
        "The object is a policy document with no executable content and no "
        "numeric claim; the applicable controls are adversarial cases, recorded "
        "under execution_performed instead.")
    rc, out = run([r])
    assert rc == 0, out


def test_rejects_confidence_voting():
    for text in ("I am 95% confident in the bound.",
                 "Overall confidence level: high.",
                 "Recorded with high confidence."):
        r = base_record()
        r["notes"] = text
        rc, out = run([r])
        assert rc == 1, f"{text!r} accepted: {out}"
        assert "confidence" in out.lower()


def test_rejects_unknown_field_and_missing_required_field():
    r = base_record()
    r["independence_waiver"] = "granted"
    rc, out = run([r])
    assert rc == 1
    assert "unknown field" in out

    r = base_record()
    del r["does_not_establish"]
    rc, out = run([r])
    assert rc == 1
    assert "missing required field" in out


def test_rejects_filename_not_matching_review_id():
    r = base_record()
    rc, out = run([], extra_files={"SOMETHING-ELSE.json": json.dumps(r)})
    assert rc == 1
    assert "review_id" in out


def test_rejects_duplicate_review_id():
    a = base_record()
    b = second_record()
    b_copy = copy.deepcopy(b)
    b_copy["review_id"] = a["review_id"]
    rc, out = run([a], extra_files={"REV-FIXTURE-A-dup.json": json.dumps(b_copy)})
    assert rc == 1
    assert "review_id" in out


def test_rejects_same_family_reviewer_claiming_credit():
    r = base_record()
    r["reviewer_family"] = "openai"
    r["author_family"] = "openai"
    r["independence_credit"] = 1
    r["independence_evidence"] = "Distinct session identity only."
    rc, out = run([r])
    assert rc == 1
    assert "same-family" in out


def test_rejects_credit_against_unknown_author_lineage():
    r = base_record()
    r["reviewer_family"] = "openai"
    r["author_family"] = "unknown"
    r["independence_credit"] = 1
    r["independence_evidence"] = "Fresh session, frozen spec."
    rc, out = run([r])
    assert rc == 1
    assert "unknown" in out


def test_rejects_credit_without_documented_evidence():
    r = base_record()
    r["reviewer_family"] = "google"
    r["author_family"] = "openai"
    r["independence_credit"] = 1
    rc, out = run([r])
    assert rc == 1
    assert "independence_evidence" in out


def test_rejects_zero_credit_with_empty_reason():
    r = base_record()
    r["independence_reason"] = ""
    rc, out = run([r])
    assert rc == 1
    assert "independence_reason" in out


def test_rejects_malformed_json_record():
    rc, out = run([], extra_files={"REV-BROKEN.json": "{not json"})
    assert rc == 1
    assert "not readable as JSON" in out


def test_schema_file_is_the_one_the_checker_enforces():
    """The schema on disk must cover every field the checker's rules reference."""
    with open(SCHEMA, encoding="utf-8") as f:
        schema = json.load(f)
    required = set(schema["required"])
    for field in ("route_key", "object_sha256", "obtained", "reviewer_family",
                  "author_family", "exposure_disclosure", "technical_verdict",
                  "independence_credit", "independence_reason",
                  "gate_status_after", "does_not_establish",
                  "reproducible_output"):
        assert field in required, f"{field} is not required by the schema"
    assert schema["properties"]["gate_status_after"]["const"] == "UNCHANGED"
    assert schema["additionalProperties"] is False
