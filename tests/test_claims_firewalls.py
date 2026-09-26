"""Negative controls for the ways a claim-graph firewall failed *open*.

`tests/test_claims.py` covers the firewalls' subject matter: it promotes a
conditional theorem, composes 2D with 3D, leaks the prize track, and asserts each
is rejected. Those tests pass against a checker whose guards are fail-open,
because they mutate the data in the one shape the guard happens to recognise.

This file covers the guards themselves. Each test drives the checker into a
region where it used to answer "nothing to refuse" — an unlisted arithmetic word,
a `certifying` that is a string rather than a boolean, a premise status in neither
vocabulary, a status column nobody read, a `track` that matches no firewall's
membership test, and a firewall list nothing reconciled — and asserts the checker
now refuses. Every one is paired with a positive control: the same mutation with
the offending field put back, or the committed tree, so that a refusal is known to
come from the mutated field and not from the harness.

As in `tests/test_claims.py`, the checker is always invoked through its CLI with
an explicit `--graph`. A default-argument bug once made every mutation test in
this repository silently re-check the committed graph.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "claims_check.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import claims_check  # noqa: E402

# The unconditional claims in the committed graph all carry `depends_on: []`, so
# FW-UNCONDITIONAL's transitive closure over them is empty and no premise status
# is reached at all. Controls below therefore wire a premise underneath one on
# purpose. That wiring is a hypothetical about the checker; it records nothing
# about the mathematics, and nothing in this repository says any of these claims
# rests on any of these premises.
UNCONDITIONAL_CLAIM = "H3-BAND-FLOOR"
CARRIER_EVIDENCE = ("D3-LEMMA-RN-UNIF", 0)   # the one record naming a carrier_id
CERTIFYING_EVIDENCE = ("RN3-FAR", 0)         # certifying: true, interval arithmetic

# From engine/rn_engine/BINDING.json, RNENG-01. Genuinely float code whose own
# sentence denies exact arithmetic, and therefore mentions three exact tokens.
BOTH_VOCABULARIES = ("mpmath binary floating point at mp.dps = 100; no "
                     "fractions.Fraction, no decimal.Decimal and no interval "
                     "arithmetic occurs anywhere")


def graph():
    return claims_check.load()


def run(mutated=None, readme=None, binding=None, manifest=None):
    """Run the checker through its CLI. Returns (returncode, stdout).

    `mutated` is written to a temporary file and passed with `--graph`; passing
    None checks the committed graph. `readme`/`binding`/`manifest` take either a
    string of file contents (or a dict, JSON-dumped) to write out, or a path,
    which is passed through unchanged so a deliberately absent path is testable.
    """
    with tempfile.TemporaryDirectory() as d:
        argv = [sys.executable, CHECKER]

        def add(flag, value, name):
            if value is None:
                return
            if isinstance(value, str) and (os.path.isabs(value) or value.startswith("/")):
                argv.extend([flag, value])
                return
            path = os.path.join(d, name)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(value if isinstance(value, str) else json.dumps(value))
            argv.extend([flag, path])

        if mutated is not None:
            add("--graph", json.dumps(mutated), "graph.json")
        add("--readme", readme, "README.md")
        add("--binding", binding, "BINDING.json")
        add("--manifest", manifest, "MANIFEST.json")
        p = subprocess.run(argv, capture_output=True, text=True)
        return p.returncode, p.stdout


def evidence(g, ref):
    name, i = ref
    node = g["claims"].get(name) or g["premises"][name]
    return node["evidence"][i]


def prose():
    with open(os.path.join(ROOT, "claims", "README.md"), encoding="utf-8") as h:
        return h.read()


# ------------------------------------------------------------------ baseline --

def test_the_committed_tree_passes():
    code, out = run()
    assert code == 0, out
    assert "problems=0" in out


def test_the_summary_reports_enforced_and_unreadable_counts():
    """A count that is not printed cannot be noticed drifting."""
    code, out = run()
    assert code == 0, out
    assert "enforced=9" in out, out
    # One record's arithmetic is the BINDING sentence, which is ambiguous by
    # construction. Reporting it is the point: it is unreadable, not inert.
    assert "evidence_arithmetic_unreadable=1" in out, out


# ------------------------------- FW-FLOAT-NOT-CERTIFIED, the certifying field --

def test_rejects_a_certifying_string_that_is_not_a_boolean():
    """`certifying is True` is an identity test. "true" walked straight past it."""
    g = graph()
    evidence(g, CERTIFYING_EVIDENCE)["certifying"] = "true"
    code, out = run(g)
    assert code == 1, out
    assert "is not a boolean" in out, out


def test_the_same_record_passes_with_a_real_boolean():
    g = graph()
    evidence(g, CERTIFYING_EVIDENCE)["certifying"] = True
    assert run(g)[0] == 0


def test_rejects_certifying_with_an_unrecognised_arithmetic_word():
    """"binary64" is float. The old token list did not contain it, and an
    unlisted word answered "not float", which the caller read as nothing to
    refuse."""
    g = graph()
    ev = evidence(g, CERTIFYING_EVIDENCE)
    ev["arithmetic"], ev["certifying"] = "binary64", True
    code, out = run(g)
    assert code == 1, out
    assert "unrecognised" in out, out


def test_rejects_certifying_with_a_missing_arithmetic_field():
    g = graph()
    ev = evidence(g, CERTIFYING_EVIDENCE)
    ev.pop("arithmetic", None)
    ev["certifying"] = True
    code, out = run(g)
    assert code == 1, out
    assert "FW-FLOAT-NOT-CERTIFIED" in out, out


def test_rejects_certifying_with_not_applicable_arithmetic():
    """A record that performs no arithmetic certifies nothing. Nine records in
    the graph say `not_applicable`, and the old guard let any of them claim
    certification."""
    g = graph()
    ev = evidence(g, CERTIFYING_EVIDENCE)
    ev["arithmetic"], ev["certifying"] = "not_applicable", True
    code, out = run(g)
    assert code == 1, out
    assert "certifies nothing" in out, out


def test_rejects_certifying_with_a_sentence_from_both_vocabularies():
    """The trap: a denial of exact arithmetic mentions exact tokens, and "an
    exact token anywhere wins" classified float code as exact."""
    g = graph()
    ev = evidence(g, CERTIFYING_EVIDENCE)
    ev["arithmetic"], ev["certifying"] = BOTH_VOCABULARIES, True
    code, out = run(g)
    assert code == 1, out
    assert "ambiguous" in out, out


def test_a_non_certifying_record_may_describe_its_arithmetic_in_prose():
    """The other direction has to stay open, or the BINDING sentence itself
    becomes a failure. Nothing rests on a record that certifies nothing."""
    g = graph()
    ev = evidence(g, CERTIFYING_EVIDENCE)
    ev["arithmetic"], ev["certifying"] = BOTH_VOCABULARIES, False
    assert run(g)[0] == 0


def test_the_classifier_refuses_to_call_the_binding_sentence_exact():
    assert claims_check.classify_arithmetic(BOTH_VOCABULARIES) == claims_check.AMBIGUOUS
    assert claims_check.classify_arithmetic("exact_rational") == claims_check.EXACT
    assert claims_check.classify_arithmetic("mpmath_float") == claims_check.FLOAT
    assert claims_check.classify_arithmetic("not_applicable") == claims_check.NOT_APPLICABLE
    for unreadable in ("binary64", "", "   ", None, 3, True, ["float"]):
        assert claims_check.classify_arithmetic(unreadable) == claims_check.UNRECOGNISED


def test_is_float_arithmetic_still_answers_only_the_float_question():
    """Kept as a wrapper, so a caller that only wants the float question is not
    broken. It must not start answering True for the fail-open classes."""
    assert claims_check.is_float_arithmetic("mpmath_float") is True
    assert claims_check.is_float_arithmetic("exact_rational") is False
    assert claims_check.is_float_arithmetic("binary64") is False
    assert claims_check.is_float_arithmetic(BOTH_VOCABULARIES) is False


# ----------------------------------------------- the second carrier index --

def test_the_graph_s_only_carrier_id_resolves():
    """It did not. The graph names exactly one carrier_id and the checker read
    only the index that does not contain it, so the override that says "the
    carrier's own record wins" resolved nothing on every run."""
    idx = claims_check.carrier_indexes()
    name, i = CARRIER_EVIDENCE
    node = graph()["premises"][name]
    cid = node["evidence"][i]["carrier_id"]
    assert cid in idx, cid
    _, label = idx[cid]
    assert label == "engine/rn_engine/BINDING.json", label


def test_a_carrier_index_overrides_the_graph_and_names_its_source():
    """Point --binding at an index whose record certifies a float, and the
    refusal must quote that file rather than the graph."""
    g = graph()
    name, i = CARRIER_EVIDENCE
    ev = g["premises"][name]["evidence"][i]
    ev["arithmetic"], ev["certifying"] = "exact_rational", False
    fake = {"carriers": [{"carrier_id": ev["carrier_id"],
                          "arithmetic": "mpmath_float", "certifying": True}]}
    code, out = run(g, binding=fake)
    assert code == 1, out
    assert "FW-FLOAT-NOT-CERTIFIED" in out and "BINDING.json" in out, out


def test_both_carrier_indexes_absent_is_skipped_cleanly():
    """Skipping can only lose a refusal the graph's own record would have to
    state anyway. It must not crash and must not manufacture one."""
    missing = "/nonexistent/carrier-index.json"
    code, out = run(binding=missing, manifest=missing)
    assert code == 0, out


# ---------------------------------------- FW-UNCONDITIONAL, both columns --

def test_rejects_a_named_hypothesis_premise_under_an_unconditional_claim():
    """A named hypothesis is by definition not discharged. The old test asked
    only whether the frozen column read OPEN or NOT_CLOSED, so NAMED_HYPOTHESIS
    — carried by two premises — passed as though it were a discharge."""
    g = graph()
    g["claims"][UNCONDITIONAL_CLAIM]["depends_on"] = ["H5-RIM"]
    code, out = run(g)
    assert code == 1, out
    assert "FW-UNCONDITIONAL" in out and "NAMED_HYPOTHESIS" in out, out


def test_rejects_a_premise_open_only_in_its_register_note_column():
    """The two-column shape exists because the columns disagree. Reading one of
    them made a disagreement in the unsafe direction invisible."""
    g = graph()
    g["premises"]["H5-RIM"]["status_frozen_v2_2"] = "CLOSED"
    g["premises"]["H5-RIM"]["status_register_note"] = "OPEN"
    g["claims"][UNCONDITIONAL_CLAIM]["depends_on"] = ["H5-RIM"]
    code, out = run(g)
    assert code == 1, out
    assert "status_register_note" in out, out


def test_the_same_premise_passes_when_both_columns_are_discharged():
    """The positive control for the two above: the refusal comes from the status
    columns, not from the wiring the controls add."""
    g = graph()
    g["premises"]["H5-RIM"]["status_frozen_v2_2"] = "CLOSED"
    g["premises"]["H5-RIM"]["status_register_note"] = "CLOSED"
    g["claims"][UNCONDITIONAL_CLAIM]["depends_on"] = ["H5-RIM"]
    assert run(g)[0] == 0


def test_rejects_a_premise_status_word_in_neither_vocabulary():
    """An unrecognised status is not evidence of a discharge, and it is not
    evidence of anything else either. It has to be placed explicitly."""
    g = graph()
    g["premises"]["H5-RIM"]["status_frozen_v2_2"] = "UNDER_REVIEW"
    code, out = run(g)
    assert code == 1, out
    assert "PREMISE_DISCHARGED" in out, out


def test_restated_and_refinement_are_not_discharges():
    """Two premises carry these words today. Whether a restatement discharges a
    premise is a status decision; this checker transcribes and does not decide,
    so the safe reading is the only one available to it."""
    for word in ("RESTATED", "REFINEMENT"):
        assert word in claims_check.PREMISE_UNDISCHARGED
        assert word not in claims_check.PREMISE_DISCHARGED
    g = graph()
    g["premises"]["H5-RIM"]["status_frozen_v2_2"] = "RESTATED"
    g["premises"]["H5-RIM"]["status_register_note"] = "RESTATED"
    g["claims"][UNCONDITIONAL_CLAIM]["depends_on"] = ["H5-RIM"]
    code, out = run(g)
    assert code == 1, out
    assert "FW-UNCONDITIONAL" in out, out


def test_the_two_status_vocabularies_do_not_overlap():
    assert not (claims_check.PREMISE_DISCHARGED & claims_check.PREMISE_UNDISCHARGED)


# ------------------------------------------------- the track vocabulary --

def test_rejects_an_unrecognised_track():
    """Three firewalls test `track` by membership against a literal set, so an
    unrecognised value drops the node out of all three at once and fails
    nothing. A rename was a silent opt-out from the 2D/3D rule."""
    g = graph()
    g["premises"]["H5-RIM"]["track"] = "UPPER_2D"     # a plausible typo
    code, out = run(g)
    assert code == 1, out
    assert "is not one of" in out, out


def test_rejects_a_missing_track():
    g = graph()
    del g["premises"]["H5-RIM"]["track"]
    code, out = run(g)
    assert code == 1, out
    assert "carries no `track`" in out, out


def test_every_track_in_the_committed_graph_is_known():
    g = graph()
    for node in list(g["claims"].values()) + list(g["premises"].values()):
        assert node.get("track") in claims_check.TRACKS, node.get("track")


def test_the_q0_tracks_are_a_subset_of_the_track_vocabulary():
    assert claims_check.Q0_TRACKS <= claims_check.TRACKS


# ------------------------------------- declared / enforced / documented --

def test_rejects_a_firewall_declared_in_the_graph_and_not_enforced():
    g = graph()
    g["firewalls"].append({"id": "FW-INVENTED", "rule": "nothing applies this",
                           "source": "none"})
    code, out = run(g)
    assert code == 1, out
    assert "FW-INVENTED" in out and "ENFORCED_FIREWALLS" in out, out


def test_rejects_a_firewall_enforced_and_not_declared():
    g = graph()
    g["firewalls"] = [f for f in g["firewalls"]
                      if f["id"] != "FW-RETRACTED-NOT-UNCONDITIONAL"]
    code, out = run(g)
    assert code == 1, out
    assert "not declared in claims/graph.json" in out, out


def test_rejects_prose_that_does_not_name_an_enforced_firewall():
    """The finding this reconciliation came from: the checker enforced nine
    firewalls and claims/README.md documented eight."""
    stripped = prose().replace("FW-RETRACTED-NOT-UNCONDITIONAL", "FW-REDACTED")
    code, out = run(readme=stripped)
    assert code == 1, out
    assert "FW-RETRACTED-NOT-UNCONDITIONAL is enforced" in out, out


def test_rejects_a_drifted_claim_count_in_the_prose():
    code, out = run(readme=prose().replace("26 claims", "24 claims"))
    assert code == 1, out
    assert "claim count has drifted" in out, out


def test_rejects_a_drifted_firewall_count_in_the_prose():
    code, out = run(readme=prose().replace("9 firewalls", "8 firewalls"))
    assert code == 1, out
    assert "firewall count has drifted" in out, out


def test_rejects_an_absent_prose_document():
    """Guarding the reconciliation with "if the file exists" would have made
    deleting claims/README.md the way to switch it off."""
    code, out = run(readme="/nonexistent/claims/README.md")
    assert code == 1, out
    assert "is not a pass" in out, out


def test_the_committed_prose_states_both_counts():
    g = graph()
    text = prose()
    assert f"{len(g['claims'])} claims" in text
    assert f"{len(claims_check.ENFORCED_FIREWALLS)} firewalls" in text


def test_enforced_matches_declared_on_the_committed_graph():
    declared = {f["id"] for f in graph()["firewalls"]}
    assert declared == set(claims_check.ENFORCED_FIREWALLS), (
        declared ^ set(claims_check.ENFORCED_FIREWALLS))


# --------------------------------------------------------------- the CLI --

def test_an_unknown_flag_is_an_error_rather_than_a_silent_default():
    code, out = run()
    assert code == 0, out
    p = subprocess.run([sys.executable, CHECKER, "--graf", "x.json"],
                       capture_output=True, text=True)
    assert p.returncode == 2, p.stdout
    assert "unknown argument" in p.stdout


def test_every_path_flag_needs_a_path():
    for flag in ("--graph", "--manifest", "--binding", "--readme"):
        p = subprocess.run([sys.executable, CHECKER, flag],
                           capture_output=True, text=True)
        assert p.returncode == 2, (flag, p.stdout)
        assert "needs a path" in p.stdout, (flag, p.stdout)
