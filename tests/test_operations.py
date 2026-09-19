"""The operations registry, the trial runner and `tools/operations_check.py`,
with negative controls.

Every test below is a NEGATIVE CONTROL unless its name says otherwise: it
takes the real registry or a real trial, breaks a copy in exactly one way, and
asserts that the checker (driven through its CLI flags against temporary
directories, never through a default bound at import time) or the runner
refuses it. The positive cases are: the real registry and trials pass; every
checkable operation reproduces its displayed identity when re-run into a
temporary directory; the arithmetic is exact and float-free; the checker
writes nothing.

None of this is mathematics. A passing suite says the registry is the
register's words, that a trial is shaped like the register's trial ledger and
claims nothing, and that the four displayed identities are reproduced in exact
arithmetic from their cells. It says nothing about any operation's premises,
usefulness or novelty (UNMEASURED / NOT_ASSESSED are the register's words), it
moves no obligation, and it does not touch the five OPEN validity premises of
Theorem D1 v2.2(2) or `D3-LEMMA-RN-UNIF`, which remains open.
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.operations import trial as T  # noqa: E402

CHECKER = os.path.join(ROOT, "tools", "operations_check.py")
RUNNER = os.path.join(ROOT, "engine", "operations", "trial.py")
REGISTRY = os.path.join(ROOT, "engine", "operations", "REGISTRY.json")
REGISTER = os.path.join(ROOT, "registers", "json", "reusable_operations.json")
TRIALS_TAB = os.path.join(ROOT, "registers", "json", "operation_trials.json")
TRIALS = os.path.join(ROOT, "engine", "operations", "trials")
UTC = "2026-09-19T00:00:00Z"
SUMMARY_RE = re.compile(r"^operations_check: registry=(\d+) checkable=(\d+) trials=(\d+) "
                        r"problems=(\d+)$", re.M)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")


def snapshot(directory):
    out = {}
    for dirpath, _dirs, files in os.walk(directory):
        for fn in files:
            p = os.path.join(dirpath, fn)
            with open(p, "rb") as f:
                out[os.path.relpath(p, directory)] = hashlib.sha256(f.read()).hexdigest()
    return out


class Sandbox:
    """A copy of the registry, the two register tabs, the committed trials and
    the one research module the registry cites, laid out as in the repository,
    under a temporary root the checker is pointed at with ``--root``."""

    def __init__(self, tmp_path):
        self.root = str(tmp_path / "repo")
        self.registry = os.path.join(self.root, "engine", "operations", "REGISTRY.json")
        self.register = os.path.join(self.root, "registers", "json", "reusable_operations.json")
        self.trials_tab = os.path.join(self.root, "registers", "json", "operation_trials.json")
        self.trials = os.path.join(self.root, "engine", "operations", "trials")
        for src, dst in ((REGISTRY, self.registry), (REGISTER, self.register),
                         (TRIALS_TAB, self.trials_tab)):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
        shutil.copytree(TRIALS, self.trials)
        for e in load(REGISTRY)["entries"]:
            mod = e["git_side"]["research_module"]
            if mod:
                dst = os.path.join(self.root, mod)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(os.path.join(ROOT, mod), dst)

    def run(self, *extra, git=False):
        cmd = [sys.executable, CHECKER, "--registry", self.registry, "--register",
               self.register, "--trials", self.trials, "--trials-tab", self.trials_tab,
               "--root", self.root, *extra]
        if not git:
            cmd.append("--no-git")
        return subprocess.run(cmd, capture_output=True, text=True)

    def trial_paths(self):
        return sorted(os.path.join(self.trials, fn) for fn in os.listdir(self.trials)
                      if fn.endswith(".json"))

    def trial_for(self, op_id):
        for p in self.trial_paths():
            if load(p)["Operation ID"] == op_id:
                return p
        raise AssertionError(f"no committed trial for {op_id}")

    def mutate_registry(self, fn):
        reg = load(self.registry)
        fn(reg)
        dump(self.registry, reg)

    def mutate_trial(self, op_id, fn):
        p = self.trial_for(op_id)
        rec = load(p)
        fn(rec)
        dump(p, rec)
        return p

    def rebind_trials(self):
        """After a registry mutation, point every trial at the mutated registry
        (catalog digest, row digest, recomputed Trial ID and filename) so the
        only problems the checker can report are the mutation's own. Cells the
        mutation changed are deliberately not re-quoted."""
        reg = load(self.registry)
        entries = {e["operation_id"]: e for e in reg["entries"]}
        lib = T.sha256_file(self.registry)
        for p in self.trial_paths():
            rec = load(p)
            entry = entries.get(rec["Operation ID"])
            if entry is None:
                continue
            rec["Library / catalog SHA-256"] = lib
            rec["Problem ID / SHA-256"] = entry["row_sha256"]
            rec["Trial ID"] = T.trial_id(rec["Operation ID"], entry["row_sha256"], lib,
                                         rec["Run evidence"]["identity_spec"],
                                         rec["Run evidence"]["utc"])
            os.remove(p)
            dump(os.path.join(self.trials, rec["Trial ID"] + ".json"), rec)


@pytest.fixture
def sandbox(tmp_path):
    return Sandbox(tmp_path)


def assert_fails(result, *needles):
    assert result.returncode == 1, result.stdout + result.stderr
    m = SUMMARY_RE.search(result.stdout)
    assert m and int(m.group(4)) >= 1, result.stdout
    for needle in needles:
        assert needle in result.stdout, f"{needle!r} not in:\n{result.stdout}"


def assert_passes(result):
    assert result.returncode == 0, result.stdout + result.stderr
    m = SUMMARY_RE.search(result.stdout)
    assert m and m.group(4) == "0", result.stdout
    return m


def run_trial(*args):
    return subprocess.run([sys.executable, RUNNER, *args], capture_output=True, text=True)


def no_floats(obj):
    if isinstance(obj, float):
        return False
    if isinstance(obj, dict):
        return all(no_floats(v) for v in obj.values())
    if isinstance(obj, list):
        return all(no_floats(v) for v in obj)
    return True


# ---------------------------------------------------------------------------
# positive cases
# ---------------------------------------------------------------------------

def test_positive_real_registry_and_trials_pass():
    r = subprocess.run([sys.executable, CHECKER], capture_output=True, text=True)
    m = assert_passes(r)
    assert m.group(1) == "15" and m.group(2) == "4"
    assert int(m.group(3)) >= 4


def test_positive_registry_is_the_register_cell_for_cell():
    reg, src = load(REGISTRY), load(REGISTER)
    assert reg["header"] == src["header"]
    assert len(reg["entries"]) == len(src["rows"]) == 15
    for e in reg["entries"]:
        row = src["rows"][e["register_row_index"]]
        assert list(e["cells"].values()) == row
        assert e["row_sha256"] == hashlib.sha256(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        assert e["cells"]["Utility"] == "UNMEASURED"
        assert e["cells"]["Novelty"] == "NOT_ASSESSED"
        assert e["cells"]["Do not infer"] == row[8]
    checkable = sorted(e["operation_id"] for e in reg["entries"]
                       if e["git_side"]["exact_identity_checkable"])
    assert checkable == ["OP02", "OP03", "OP04", "OP05"] == sorted(T.IDENTITIES)


def test_positive_checker_writes_nothing(sandbox):
    before = snapshot(sandbox.root)
    assert_passes(sandbox.run())
    assert snapshot(sandbox.root) == before


@pytest.mark.parametrize("op_id", sorted(T.IDENTITIES))
def test_positive_every_checkable_op_reproduces_when_rerun(tmp_path, op_id):
    out = str(tmp_path / "trials")
    r = run_trial("--op", op_id, "--out", out, "--utc", UTC)
    assert r.returncode == 0, r.stdout + r.stderr
    files = os.listdir(out)
    assert len(files) == 1 and files[0].startswith(f"TRIAL-{op_id}-20260919T000000Z-")
    rec = load(os.path.join(out, files[0]))
    assert rec["Verified result"] == "IDENTITY_REPRODUCED"
    assert list(rec.keys()) == list(T.TRIAL_COLUMNS) + list(T.REPOSITORY_FIELDS)
    assert rec["Run evidence"]["all_equal"] is True
    assert rec["Run evidence"]["failure"] is None
    assert rec["Timeout / failure charge"] == 0
    assert rec["Verification cost"] == len(rec["Run evidence"]["steps"]) >= 2
    assert no_floats(rec), "a trial record must carry no float anywhere"
    # the same arithmetic as the committed record for this op
    committed = [load(os.path.join(TRIALS, fn)) for fn in os.listdir(TRIALS)
                 if fn.endswith(".json")]
    same = [c for c in committed if c["Operation ID"] == op_id]
    assert same, f"no committed trial for {op_id}"
    strip = lambda s: [(x["statement"], x["lhs"], x["rhs"], x["equal"]) for x in s]  # noqa: E731
    assert strip(rec["Run evidence"]["steps"]) == strip(same[-1]["Run evidence"]["steps"])
    # the register's words travel with the record
    entry = {e["operation_id"]: e for e in load(REGISTRY)["entries"]}[op_id]
    assert rec["Scope / constraints"] == entry["cells"]["Required scope"]
    assert entry["cells"]["Do not infer"] in rec["Notes"]
    assert rec["Problem ID / SHA-256"] == entry["row_sha256"]
    assert rec["authority"] == T.AUTHORITY


def test_positive_the_four_identities_are_what_the_task_names():
    """OP02 is the exponent ledger 3-5+2=0 with coefficient 1/6; OP03 the
    pushforward constant 6^(2/3)/(3 kappa^(2/3)) and exponent -1/3; OP04 the
    Hölder conjugacy 1/4+1/4+1/2=1; OP05 the witness LHS=1/4, wrong RHS=1/16."""
    s02, _ = T.evaluate(T.IDENTITIES["OP02"])
    assert s02[0]["lhs"] == "0" and s02[1]["lhs"] == "1/6"
    s03, _ = T.evaluate(T.IDENTITIES["OP03"])
    assert s03[2]["lhs"] == "-1/3"
    assert s03[1]["lhs"] == {"coefficient": "1/3",
                             "exponents": {"2": "2/3", "3": "2/3", "ell": "-1/3", "kappa": "-2/3"}}
    s04, _ = T.evaluate(T.IDENTITIES["OP04"])
    assert s04[0]["lhs"] == "1" and s04[1]["lhs"] == ["1/4", "1/4", "1/2"]
    s05, _ = T.evaluate(T.IDENTITIES["OP05"])
    assert s05[0]["lhs"] == "1/4" and s05[1]["lhs"] == "1/16" and s05[2]["lhs"] is True


def test_positive_list_runs():
    r = run_trial("--list")
    assert r.returncode == 0
    assert "registry=15 checkable=4" in r.stdout
    assert r.stdout.count("EXACT_IDENTITY") == 4
    assert r.stdout.count("NOT_MACHINE_CHECKABLE_HERE") == 11


def test_positive_not_run_record_for_a_non_checkable_op_passes(sandbox):
    r = run_trial("--op", "OP01", "--registry", sandbox.registry, "--out", sandbox.trials,
                  "--utc", UTC)
    assert r.returncode == 0, r.stdout
    rec = load(sandbox.trial_for("OP01"))
    assert rec["Verified result"] == "NOT_RUN"
    assert rec["Run evidence"]["steps"] == []
    assert rec["Run evidence"]["reason_not_run"]
    assert rec["Split"] == "NOT_APPLICABLE" and rec["Arm"] == "NOT_APPLICABLE"
    m = assert_passes(sandbox.run())
    assert int(m.group(3)) == len(sandbox.trial_paths())


# ---------------------------------------------------------------------------
# exact arithmetic
# ---------------------------------------------------------------------------

def test_positive_monomials_are_exact_and_canonical():
    six_two_thirds = T.Monomial(1, {"6": Fraction(2, 3)})
    assert six_two_thirds.exponents == {"2": Fraction(2, 3), "3": Fraction(2, 3)}
    assert T.Monomial(1, {"6": 1}) == T.Monomial(6)              # integer exponent folds
    assert T.Monomial(1, {"6": Fraction(3, 3)}).coefficient == 6
    assert (T.Monomial(6) ** Fraction(1, 3)) ** 3 == T.Monomial(6)
    assert T.Monomial(1, {"6": Fraction(1, 3)}) ** 2 == six_two_thirds
    assert T.exact_root(Fraction(1, 256), 2) == Fraction(1, 16)
    assert T.exact_root(Fraction(81), 4) == 3
    with pytest.raises(ValueError):
        T.exact_root(Fraction(2), 2)                             # not a perfect square
    with pytest.raises(ValueError):
        T.Monomial(-6) ** Fraction(1, 3)


# ---------------------------------------------------------------------------
# negative controls: the registry
# ---------------------------------------------------------------------------

def test_registry_cell_paraphrased_is_refused(sandbox):
    def para(reg):
        reg["entries"][1]["cells"]["Do not infer"] = (
            "The r^2 factor must be established by the caller. 1/6 is not a full contact "
            "coefficient or a proof of finite nonzero limiting mass.")
    sandbox.mutate_registry(para)
    assert_fails(sandbox.run(), "(OP02): cell 'Do not infer' differs from the register row")


def test_registry_utility_high_is_refused(sandbox):
    sandbox.mutate_registry(lambda reg: reg["entries"][2]["cells"].__setitem__("Utility", "HIGH"))
    assert_fails(sandbox.run(), "cell 'Utility' differs", "never stronger")


def test_registry_novelty_assessed_is_refused(sandbox):
    sandbox.mutate_registry(
        lambda reg: reg["entries"][2]["cells"].__setitem__("Novelty", "NEW"))
    assert_fails(sandbox.run(), "cell 'Novelty' differs")


def test_registry_git_side_may_not_speak_of_usefulness(sandbox):
    def why(reg):
        reg["entries"][0]["git_side"]["why"] = ("This operation is useful and novel and should "
                                                "be retrieved first in every session.")
    sandbox.mutate_registry(why)
    sandbox.rebind_trials()
    assert_fails(sandbox.run(), "git_side speaks of 'useful' without the register's "
                                "UNMEASURED / NOT_ASSESSED")


def test_registry_git_side_may_not_speak_of_utility_or_novelty_by_name(sandbox):
    """Round-1 mutation 14: the words 'utility' and 'novelty' themselves were
    stripped as header names before the scan. They are not header names."""
    def why(reg):
        reg["entries"][0]["git_side"]["why"] = (
            "The utility of this operation is high and its novelty is substantial; "
            "retrieve it first in every session.")
    sandbox.mutate_registry(why)
    sandbox.rebind_trials()
    assert_fails(sandbox.run(), "git_side speaks of 'utility' paired with the value word 'high'")

    def novelty(reg):
        reg["entries"][0]["git_side"]["why"] = (
            "The novelty of this operation is plain to anyone who reads the frame.")
    sandbox.mutate_registry(novelty)
    sandbox.rebind_trials()
    assert_fails(sandbox.run(), "git_side speaks of 'novelty' without the register's "
                                "UNMEASURED / NOT_ASSESSED")


def test_registry_git_side_header_phrase_is_exempt_but_not_a_shield(sandbox):
    """'Next useful step' is the register's header name and passes; the same
    phrase next to a claim shields nothing."""
    def ok(reg):
        reg["entries"][0]["git_side"]["why"] += (
            " The register's Next useful step names the stored matrix, which is not here.")
    sandbox.mutate_registry(ok)
    sandbox.rebind_trials()
    assert_passes(sandbox.run())

    def claim(reg):
        reg["entries"][0]["git_side"]["why"] += " Its Next useful step has HIGH utility."
    sandbox.mutate_registry(claim)
    sandbox.rebind_trials()
    assert_fails(sandbox.run(), "git_side speaks of 'utility' paired with the value word 'HIGH'")


def test_registry_git_side_may_repeat_the_register_words_only_with_them(sandbox):
    def why(reg):
        reg["entries"][0]["git_side"]["why"] += (
            " Utility stays UNMEASURED and Novelty NOT_ASSESSED, as the register says.")
    sandbox.mutate_registry(why)
    sandbox.rebind_trials()
    assert_passes(sandbox.run())


def test_registry_git_side_may_not_use_status_words(sandbox):
    def why(reg):
        reg["entries"][0]["git_side"]["why"] += " The frame determinant is thereby PROVEN."
    sandbox.mutate_registry(why)
    sandbox.rebind_trials()
    assert_fails(sandbox.run(), "git_side uses status words ['PROVEN']")


def test_registry_top_level_text_may_not_use_status_words(sandbox):
    """Round-1 mutation 12: the registry's own does_not_establish said PROVEN
    and CERTIFIED and the checker passed."""
    sandbox.mutate_registry(lambda reg: reg["does_not_establish"].append(
        "All four displayed identities are PROVEN and CERTIFIED by the trials in this "
        "directory."))
    sandbox.rebind_trials()
    assert_fails(sandbox.run(), "top-level text uses status words ['CERTIFIED', 'PROVEN']")


def test_registry_top_level_text_may_not_claim_utility_or_novelty(sandbox):
    """Round-1 mutation 13: utility_and_novelty said HIGH / NEW."""
    sandbox.mutate_registry(lambda reg: reg.__setitem__(
        "utility_and_novelty",
        "Utility is HIGH and Novelty is NEW for every operation, as measured by the trials "
        "here."))
    sandbox.rebind_trials()
    assert_fails(sandbox.run(), "utility_and_novelty must carry the register's words "
                                "UNMEASURED and NOT_ASSESSED verbatim",
                 "top-level text speaks of 'Utility' paired with the value word 'HIGH'")


def test_registry_top_level_text_must_carry_the_register_markers(sandbox):
    sandbox.mutate_registry(lambda reg: reg.__setitem__(
        "utility_and_novelty", "Utility and Novelty are whatever the trials in this directory "
                               "show them to be; see the records."))
    sandbox.rebind_trials()
    assert_fails(sandbox.run(), "utility_and_novelty must carry the register's words",
                 "top-level text speaks of 'Utility' without the register's")


def test_registry_scope_displayed_must_be_in_the_required_scope_cell(sandbox):
    """OP04's moment orders are taken from the Required scope cell; move the
    register and the registry together so only the binding fails."""
    src = load(sandbox.register)
    row = src["rows"][3]
    row[3] = row[3].replace("moment_orders=4,4,2", "moment_orders=4,4,4")
    dump(sandbox.register, src)

    def reg_fix(reg):
        reg["entries"][3]["cells"]["Required scope"] = row[3]
        reg["entries"][3]["row_sha256"] = T.row_sha256(row)
    sandbox.mutate_registry(reg_fix)
    sandbox.rebind_trials()
    assert_fails(sandbox.run(), "(OP04): the identity spec's displayed text is not verbatim in "
                                "the Action / output cell (or its scope_displayed text not in "
                                "Required scope)")
    r = run_trial("--op", "OP04", "--registry", sandbox.registry, "--out", sandbox.trials,
                  "--utc", UTC)
    assert r.returncode == 2 and "REFUSED OP04" in r.stdout


def test_registry_extra_row_is_refused(sandbox):
    def extra(reg):
        e = copy.deepcopy(reg["entries"][-1])
        e["operation_id"] = "OP16"
        e["cells"]["Operation"] = "OP16 — Invented"
        reg["entries"].append(e)
    sandbox.mutate_registry(extra)
    assert_fails(sandbox.run(), "16 entries but the register has 15 rows")


def test_registry_missing_row_is_refused(sandbox):
    sandbox.mutate_registry(lambda reg: reg["entries"].pop(7))
    assert_fails(sandbox.run(), "14 entries but the register has 15 rows")


def test_registry_row_digest_mismatch_is_refused(sandbox):
    sandbox.mutate_registry(lambda reg: reg["entries"][3].__setitem__("row_sha256", "0" * 64))
    assert_fails(sandbox.run(), "(OP04): row_sha256 does not match")


def test_registry_checkable_without_a_bound_spec_is_refused(sandbox):
    def flip(reg):
        reg["entries"][0]["git_side"]["exact_identity_checkable"] = True
        reg["entries"][0]["git_side"]["trial_kind"] = "EXACT_IDENTITY"
    sandbox.mutate_registry(flip)
    assert_fails(sandbox.run(), "(OP01): EXACT_IDENTITY but engine/operations/trial.py has no "
                                "identity spec")


def test_registry_flag_and_kind_must_agree(sandbox):
    sandbox.mutate_registry(
        lambda reg: reg["entries"][1]["git_side"].__setitem__("exact_identity_checkable", False))
    assert_fails(sandbox.run(), "(OP02): exact_identity_checkable must be the boolean")


def test_registry_git_side_extra_key_is_refused(sandbox):
    sandbox.mutate_registry(
        lambda reg: reg["entries"][1]["git_side"].__setitem__("status", "VERIFIED"))
    assert_fails(sandbox.run(), "git_side keys must be exactly")


def test_registry_research_module_must_exist(sandbox):
    sandbox.mutate_registry(lambda reg: reg["entries"][3]["git_side"].__setitem__(
        "research_module", "research/rn/does_not_exist.py"))
    assert_fails(sandbox.run(), "research_module 'research/rn/does_not_exist.py' is not an "
                                "existing file")


def test_trials_tab_header_drift_is_refused(sandbox):
    tab = load(sandbox.trials_tab)
    tab["header"][8] = "Result"
    dump(sandbox.trials_tab, tab)
    assert_fails(sandbox.run(), "header is not the eighteen operation_trials columns")


# ---------------------------------------------------------------------------
# negative controls: trials
# ---------------------------------------------------------------------------

def test_trial_verified_result_proven_is_refused(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec.__setitem__("Verified result", "PROVEN"))
    assert_fails(sandbox.run(), "Verified result 'PROVEN' is not in the closed vocabulary",
                 "status words ['PROVEN']")


def test_trial_missing_do_not_infer_is_refused(sandbox):
    sandbox.mutate_trial("OP03", lambda rec: rec.__setitem__(
        "Notes", "Trial kind: EXACT_IDENTITY | a record of a computation."))
    assert_fails(sandbox.run(), "Notes does not carry the register's 'Do not infer' cell verbatim")


def test_trial_do_not_infer_paraphrased_is_refused(sandbox):
    def para(rec):
        rec["Notes"] = rec["Notes"].replace("separate obligations", "separate obligation")
    sandbox.mutate_trial("OP03", para)
    assert_fails(sandbox.run(), "Notes does not carry the register's 'Do not infer' cell verbatim")


def test_trial_scope_not_verbatim_is_refused(sandbox):
    sandbox.mutate_trial("OP04", lambda rec: rec.__setitem__(
        "Scope / constraints", "law=any_probability_law; moment_orders=4,4,2"))
    assert_fails(sandbox.run(), "Scope / constraints is not the Required scope cell verbatim")


def test_trial_for_unknown_op_is_refused(sandbox):
    p = sandbox.trial_for("OP02")
    rec = load(p)
    rec["Operation ID"] = "OP99"
    rec["Trial ID"] = rec["Trial ID"].replace("OP02", "OP99")
    dump(os.path.join(sandbox.trials, rec["Trial ID"] + ".json"), rec)
    assert_fails(sandbox.run(), "Operation ID 'OP99' is not in the registry")


def test_trial_not_run_only_op_carrying_identity_reproduced_is_refused(sandbox):
    r = run_trial("--op", "OP13", "--registry", sandbox.registry, "--out", sandbox.trials,
                  "--utc", UTC)
    assert r.returncode == 0, r.stdout
    assert_passes(sandbox.run())
    sandbox.mutate_trial("OP13", lambda rec: rec.__setitem__("Verified result",
                                                              "IDENTITY_REPRODUCED"))
    assert_fails(sandbox.run(), "OP13 is NOT_MACHINE_CHECKABLE_HERE and may only carry NOT_RUN")


def test_trial_status_word_in_authored_field_is_refused(sandbox):
    sandbox.mutate_trial("OP05", lambda rec: rec["does_not_establish"].append(
        "This trial shows the wrong bound is refuted and the correct one is CERTIFIED here."))
    assert_fails(sandbox.run(), "status words ['CERTIFIED']")


def _move_op02_cell(sandbox):
    """Change OP02's 'Do not infer' cell in the register and the registry
    together (a new export) and append a fresh OP02 trial against the new
    registry. The four committed trials stay (append-only) and now name the
    committed registry version. Returns the old and the new cell text."""
    src = load(sandbox.register)
    row = src["rows"][1]
    old_cell = row[8]
    new_cell = old_cell + " This ledger is CLOSED in the source's own words."
    row[8] = new_cell
    dump(sandbox.register, src)

    def reg_fix(reg):
        reg["entries"][1]["cells"]["Do not infer"] = new_cell
        reg["entries"][1]["row_sha256"] = T.row_sha256(row)
    sandbox.mutate_registry(reg_fix)
    r = run_trial("--op", "OP02", "--registry", sandbox.registry, "--out", sandbox.trials,
                  "--utc", UTC)
    assert r.returncode == 0, r.stdout
    return old_cell, new_cell


def test_trial_status_word_inside_the_quoted_register_cell_is_not_a_claim(git_sandbox):
    """The scan is confined to repository-authored text: a status word that the
    register's own 'Do not infer' cell contains is not the trial's claim. The
    register row, the registry cell and the row digest move together; the
    four committed trials then name the committed registry version and are
    checked in full against it (HISTORICAL), not skipped."""
    sandbox = git_sandbox
    _move_op02_cell(sandbox)
    rec = load(sandbox.trial_for("OP02"))            # the fresh one sorts first
    assert "CLOSED" in rec["Notes"] and rec["Trial ID"].startswith("TRIAL-OP02-20260919T000000Z")
    r = sandbox.run(git=True)
    assert_passes(r)
    assert r.stdout.count("HISTORICAL") == 4
    # ... and the same word authored by the repository is refused
    sandbox.mutate_trial("OP02", lambda rec: rec.__setitem__(
        "Notes", rec["Notes"] + " The identity is CLOSED."))
    assert_fails(sandbox.run(), "status words ['CLOSED']")


def test_trial_naming_a_catalog_nobody_can_read_is_refused(sandbox):
    """Round-1 mutation 15: a made-up catalog digest opted the record out of
    every substantive check and printed a note. Now it fails."""
    def bogus(rec):
        rec["Library / catalog SHA-256"] = "1" * 64
        rec["Run evidence"]["identity_spec"]["displayed"] = "(r^3/6) r^-5 r^9 = 1/6"
        rec["Scope / constraints"] = "dimension=7; anything goes"
        rec["Notes"] = "no do-not-infer here"
    sandbox.mutate_trial("OP02", bogus)
    assert_fails(sandbox.run(), "matches neither the current REGISTRY.json nor any version of "
                                "it in git history")


def test_trial_with_a_stale_row_digest_under_the_current_catalog_is_refused(sandbox):
    """Round-1 mutation 27: only the Problem ID was stale; the cell checks were
    skipped although the catalog was current."""
    def stale_row(rec):
        rec["Problem ID / SHA-256"] = "2" * 64
        rec["Run evidence"]["identity_spec"]["displayed"] = "E[|ABC| I] <= anything at all"
        rec["Scope / constraints"] = "whatever"
    sandbox.mutate_trial("OP04", stale_row)
    assert_fails(sandbox.run(), "Problem ID / SHA-256 is not the row digest of OP04 in the "
                                "registry version the record names",
                 "the recorded identity is not displayed in the register's Action / output "
                 "cell (or Required scope) of the catalog the record names",
                 "Scope / constraints is not the Required scope cell verbatim")


def test_historical_record_is_checked_against_the_version_it_names(git_sandbox):
    """A record naming a committed earlier registry resolves through git
    history and is held to that version's cells: its Scope, Notes, spec
    binding and re-run are all checked, not skipped."""
    sandbox = git_sandbox
    _move_op02_cell(sandbox)
    r = sandbox.run(git=True)
    assert_passes(r)
    assert r.stdout.count("HISTORICAL") == 4
    # the historical OP03 record's cell-bound fields are still enforced
    # (run without the HEAD comparison so the only problem is this one)
    sandbox.mutate_trial("OP03", lambda rec: rec.__setitem__(
        "Scope / constraints", "input_measure=anything"))
    r = sandbox.run()
    assert_fails(r, "Scope / constraints is not the Required scope cell verbatim")
    assert r.stdout.count("HISTORICAL") == 4 and "problems=1" in r.stdout


def test_historical_record_with_an_undisplayed_identity_is_refused(git_sandbox):
    sandbox = git_sandbox
    _move_op02_cell(sandbox)
    assert_passes(sandbox.run(git=True))

    def undisplayed(rec):
        rec["Run evidence"]["identity_spec"]["displayed"] = "A=B=1, C=1/4 gives LHS=1/4 but wrong RHS=1/2"
        rec["Run evidence"]["identity_displayed"] = rec["Run evidence"]["identity_spec"]["displayed"]
    sandbox.mutate_trial("OP05", undisplayed)
    assert_fails(sandbox.run(), "the recorded identity is not displayed in the "
                                "register's Action / output cell (or Required scope) "
                                "of the catalog the record names")


def test_historical_record_with_fabricated_steps_is_refused(git_sandbox):
    sandbox = git_sandbox
    _move_op02_cell(sandbox)
    assert_passes(sandbox.run(git=True))

    def fabricate(rec):
        for s in rec["Run evidence"]["steps"]:
            s["lhs"], s["rhs"], s["equal"] = "42", "42", True
    sandbox.mutate_trial("OP04", fabricate)
    assert_fails(sandbox.run(), "gives steps that differ from the recorded steps")


def test_trial_with_fabricated_step_values_is_refused(sandbox):
    """Round-1 mutation 07 (BLOCKING): every step said lhs=rhs=42 with the spec
    intact and the checker compared only the verdict."""
    def fabricate(rec):
        for s in rec["Run evidence"]["steps"]:
            s["lhs"], s["rhs"], s["equal"] = "42", "42", True
    sandbox.mutate_trial("OP02", fabricate)
    assert_fails(sandbox.run(), "re-running the recorded identity here gives steps that differ "
                                "from the recorded steps (statement, lhs, rhs, equal)")


def test_trial_with_a_rewritten_step_statement_is_refused(sandbox):
    sandbox.mutate_trial("OP03", lambda rec: rec["Run evidence"]["steps"][1].__setitem__(
        "statement", "the full asymptotic follows from the Jacobian"))
    assert_fails(sandbox.run(), "gives steps that differ from the recorded steps")


def test_trial_with_a_dropped_step_is_refused(sandbox):
    def drop(rec):
        rec["Run evidence"]["steps"].pop()
        rec["Run evidence"]["checked_equalities"] -= 1
        rec["Verification cost"] -= 1
    sandbox.mutate_trial("OP05", drop)
    assert_fails(sandbox.run(), "gives steps that differ from the recorded steps")


def test_trial_run_evidence_bookkeeping_is_recomputed(sandbox):
    """Round-1 mutation 16: trial_kind, identity_displayed and
    checked_equalities were unchecked."""
    def rewrite(rec):
        ev = rec["Run evidence"]
        ev["identity_displayed"] = "E[|ABC| I] <= anything"
        ev["trial_kind"] = "NOT_MACHINE_CHECKABLE_HERE"
        ev["checked_equalities"] = 99
    sandbox.mutate_trial("OP05", rewrite)
    assert_fails(sandbox.run(),
                 "Run evidence.checked_equalities is not the number of recorded steps",
                 "Run evidence.trial_kind 'NOT_MACHINE_CHECKABLE_HERE' is not the registry's "
                 "'EXACT_IDENTITY' for OP05",
                 "Run evidence.identity_displayed is not the recorded identity_spec's displayed "
                 "text")


def test_trial_id_digest_is_recomputed(sandbox):
    """Round-1 mutation 17: the Trial ID's input digest was documented as
    deterministic and never recomputed."""
    p = sandbox.trial_for("OP03")
    rec = load(p)
    rec["Trial ID"] = rec["Trial ID"][:-16] + "0" * 16
    os.remove(p)
    dump(os.path.join(sandbox.trials, rec["Trial ID"] + ".json"), rec)
    assert_fails(sandbox.run(), "Trial ID digest does not recompute from (operation, row digest, "
                                "catalog digest, identity_spec, utc)")


def test_trial_id_recomputes_from_the_record(sandbox):
    for p in sandbox.trial_paths():
        rec = load(p)
        assert rec["Trial ID"] == T.trial_id(
            rec["Operation ID"], rec["Problem ID / SHA-256"], rec["Library / catalog SHA-256"],
            rec["Run evidence"]["identity_spec"], rec["Run evidence"]["utc"])


@pytest.mark.parametrize("word", ["RATIFIED", "RATIFICATION", "VERIFIED", "ACCEPTED",
                                  "ADMITTED", "PASSED", "PASS_TECHNICAL", "APPROVED",
                                  "SATISFIED", "VALIDATED", "CONFIRMED"])
def test_trial_review_verdict_words_are_refused(sandbox, word):
    """Round-1 mutations 08, 25, 26: text reading as a review verdict passed
    because the list stopped at INDEPENDENT."""
    sandbox.mutate_trial("OP02", lambda rec: rec["does_not_establish"].append(
        f"The pushforward adapter is hereby {word} for reuse in every lane of the program."))
    assert_fails(sandbox.run(), f"repository-authored fields use status words ['{word}']")


def test_trial_review_sentences_from_round_one_are_refused(sandbox):
    sandbox.mutate_trial("OP03", lambda rec: rec["does_not_establish"].extend([
        "The Hölder exponent arithmetic is VERIFIED and the operation is ACCEPTED for reuse "
        "here.",
        "External review PASSED for this operation and the gate is now satisfied by this "
        "trial."]))
    assert_fails(sandbox.run(), "status words ['ACCEPTED', 'PASSED', 'SATISFIED', 'VERIFIED']")


def test_trial_status_word_as_a_nested_key_is_refused(sandbox):
    """Keys are scanned too; only a string that is exactly a register column
    name ('Verified result') is the register's word."""
    sandbox.mutate_trial("OP02", lambda rec: rec["Run evidence"].__setitem__("VERIFIED", True))
    assert_fails(sandbox.run(), "status words ['VERIFIED']")


def test_trial_notes_must_quote_the_utility_and_novelty_cells(sandbox):
    """Round-1 mutation 10: Notes said HIGH / NEW and passed."""
    def high(rec):
        rec["Notes"] = rec["Notes"].replace("Utility: UNMEASURED; Novelty: NOT_ASSESSED",
                                            "Utility: HIGH; Novelty: NEW")
        assert "HIGH" in rec["Notes"]
    sandbox.mutate_trial("OP04", high)
    assert_fails(sandbox.run(), "Notes does not carry the register's Utility cell as "
                                "'Utility: UNMEASURED' verbatim",
                 "Notes does not carry the register's Novelty cell as 'Novelty: NOT_ASSESSED' "
                 "verbatim",
                 "repository-authored fields speak of 'Utility' paired with the value word "
                 "'HIGH'")


def test_trial_may_not_claim_a_measured_utility_novelty_or_gain(sandbox):
    """Round-1 mutation 11."""
    sandbox.mutate_trial("OP05", lambda rec: rec.__setitem__("does_not_establish", [
        "This trial measured the operation's utility as high and assessed its novelty as "
        "new, with a positive gain."]))
    assert_fails(sandbox.run(), "repository-authored fields speak of 'utility' paired with the "
                                "value word 'measured'")


def test_trial_may_not_speak_of_gain_without_the_register_words(sandbox):
    sandbox.mutate_trial("OP05", lambda rec: rec["does_not_establish"].append(
        "The gain from retrieving this operation first was observed in every session."))
    assert_fails(sandbox.run(), "speak of 'gain' without the register's UNMEASURED / "
                                "NOT_ASSESSED")


def test_trial_budget_field_is_scanned_too(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec.__setitem__(
        "Budget / cost unit", "unit=checked_equalities; utility=high; budget=NONE"))
    assert_fails(sandbox.run(), "speak of 'utility' paired with the value word 'high'")


def test_trial_not_run_may_not_carry_a_spec(sandbox):
    r = run_trial("--op", "OP12", "--registry", sandbox.registry, "--out", sandbox.trials,
                  "--utc", UTC)
    assert r.returncode == 0, r.stdout
    assert_passes(sandbox.run())

    def smuggle(rec):
        rec["Run evidence"]["identity_spec"] = copy.deepcopy(T.IDENTITIES["OP02"])
        rec["Run evidence"]["identity_displayed"] = T.IDENTITIES["OP02"]["displayed"]
    sandbox.mutate_trial("OP12", smuggle)
    assert_fails(sandbox.run(), "NOT_RUN must record no steps, no failure and no spec")


def test_trial_extra_top_level_key_is_refused(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec.__setitem__("Status", "OK"))
    assert_fails(sandbox.run(), "top-level keys must be exactly the 18 operation_trials columns")


def test_trial_missing_column_is_refused(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec.pop("Retrieval cost"))
    assert_fails(sandbox.run(), "missing=['Retrieval cost']")


def test_trial_cost_as_float_is_refused(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec.__setitem__("Verification cost", 3.0))
    assert_fails(sandbox.run(), "Verification cost must be a non-negative integer")


def test_trial_split_held_out_is_refused(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec.__setitem__("Split", "HELD_OUT"))
    assert_fails(sandbox.run(), "Split 'HELD_OUT' not in")


def test_trial_arm_claiming_a_benchmark_arm_is_refused(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec.__setitem__("Arm", "scope_aware_operations"))
    assert_fails(sandbox.run(), "Arm 'scope_aware_operations' not in")


def test_trial_authority_sentence_is_fixed(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec.__setitem__("authority", "operator"))
    assert_fails(sandbox.run(), "authority must be exactly")


def test_trial_empty_does_not_establish_is_refused(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec.__setitem__("does_not_establish", []))
    assert_fails(sandbox.run(), "does_not_establish must be a non-empty list of sentences")


def test_trial_reproduced_with_a_failed_step_is_refused(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec["Run evidence"]["steps"][0].__setitem__(
        "equal", False))
    assert_fails(sandbox.run(), "IDENTITY_REPRODUCED with a failed step",
                 "all_equal disagrees")


def test_trial_reproduced_for_an_identity_the_cell_does_not_display_is_refused(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec["Run evidence"]["identity_spec"].__setitem__(
        "displayed", "(r^3/6) r^-5 r^3 = 1/6"))
    assert_fails(sandbox.run(), "the recorded identity is not displayed in the register's "
                                "Action / output cell")


def test_trial_filename_must_match_trial_id(sandbox):
    p = sandbox.trial_for("OP04")
    os.rename(p, os.path.join(sandbox.trials, "TRIAL-OP04-renamed.json"))
    assert_fails(sandbox.run(), "filename does not match Trial ID")


# ---------------------------------------------------------------------------
# negative controls: the runner
# ---------------------------------------------------------------------------

def test_wrong_identity_yields_identity_not_reproduced():
    registry = T.load_registry(REGISTRY)
    sha = T.sha256_file(REGISTRY)
    bad = copy.deepcopy(T.IDENTITIES["OP02"])
    bad["factors"][2]["exponents"]["r"] = "3"          # r^3 where the cell says r^2
    rec = T.build_trial("OP02", registry, sha, utc=UTC, spec=bad)
    assert rec["Verified result"] == "IDENTITY_NOT_REPRODUCED"
    assert rec["Run evidence"]["failure"] is None
    assert any(s["equal"] is False for s in rec["Run evidence"]["steps"])
    assert rec["Run evidence"]["all_equal"] is False
    bad3 = copy.deepcopy(T.IDENTITIES["OP03"])
    bad3["expected"]["exponents"]["ell"] = "-2/3"
    assert T.build_trial("OP03", registry, sha, utc=UTC, spec=bad3)["Verified result"] == \
        "IDENTITY_NOT_REPRODUCED"
    bad4 = copy.deepcopy(T.IDENTITIES["OP04"])
    bad4["exponents"] = ["1/4", "1/4", "1/4"]
    assert T.build_trial("OP04", registry, sha, utc=UTC, spec=bad4)["Verified result"] == \
        "IDENTITY_NOT_REPRODUCED"
    bad5 = copy.deepcopy(T.IDENTITIES["OP05"])
    bad5["expected_wrong_rhs"] = "1/4"
    assert T.build_trial("OP05", registry, sha, utc=UTC, spec=bad5)["Verified result"] == \
        "IDENTITY_NOT_REPRODUCED"


def test_a_raising_computation_is_not_reproduced_and_is_charged_never_swallowed():
    registry = T.load_registry(REGISTRY)
    sha = T.sha256_file(REGISTRY)
    bad = copy.deepcopy(T.IDENTITIES["OP05"])
    bad["C"] = "not-a-number"
    rec = T.build_trial("OP05", registry, sha, utc=UTC, spec=bad)
    assert rec["Verified result"] == "IDENTITY_NOT_REPRODUCED"
    assert rec["Run evidence"]["failure"].startswith("ValueError")
    assert rec["Run evidence"]["steps"] == []
    assert rec["Timeout / failure charge"] == 1
    bad2 = copy.deepcopy(T.IDENTITIES["OP02"])
    bad2["kind"] = "no_such_evaluator"
    rec2 = T.build_trial("OP02", registry, sha, utc=UTC, spec=bad2)
    assert rec2["Verified result"] == "IDENTITY_NOT_REPRODUCED"
    assert rec2["Run evidence"]["failure"].startswith("KeyError")


def test_an_honest_not_reproduced_record_passes_but_relabelled_reproduced_fails(sandbox):
    registry = T.load_registry(sandbox.registry)
    sha = T.sha256_file(sandbox.registry)
    bad = copy.deepcopy(T.IDENTITIES["OP02"])
    bad["factors"][2]["exponents"]["r"] = "3"
    rec = T.build_trial("OP02", registry, sha, utc=UTC, spec=bad)
    path = T.write_trial(rec, sandbox.trials, root=sandbox.root)
    assert_passes(sandbox.run())                     # a record of a failed reproduction
    rec = load(path)
    rec["Verified result"] = "IDENTITY_REPRODUCED"
    dump(path, rec)
    assert_fails(sandbox.run(), "IDENTITY_REPRODUCED with a failed step",
                 "re-running the recorded identity here gives IDENTITY_NOT_REPRODUCED")


def test_runner_refuses_an_identity_the_registry_cell_no_longer_displays(sandbox):
    sandbox.mutate_registry(lambda reg: reg["entries"][1]["cells"].__setitem__(
        "Action / output", "(r^3/6) r^-5 r^3 = 1/6. Record every supplied factor separately."))
    r = run_trial("--op", "OP02", "--registry", sandbox.registry, "--out", str(sandbox.trials),
                  "--utc", UTC)
    assert r.returncode == 2
    assert "REFUSED OP02" in r.stdout and "not in the register's Action / output cell" in r.stdout
    assert not any("20260919T000000Z" in fn for fn in os.listdir(sandbox.trials))


def test_overwrite_attempt_is_refused(tmp_path):
    out = str(tmp_path / "trials")
    first = run_trial("--op", "OP02", "--out", out, "--utc", UTC)
    assert first.returncode == 0, first.stdout
    (fn,) = os.listdir(out)
    with open(os.path.join(out, fn), "rb") as f:
        before = f.read()
    second = run_trial("--op", "OP02", "--out", out, "--utc", UTC)
    assert second.returncode == 2
    assert "REFUSED" in second.stdout and "append-only" in second.stdout
    assert os.listdir(out) == [fn]
    with open(os.path.join(out, fn), "rb") as f:
        assert f.read() == before
    registry = T.load_registry(REGISTRY)
    rec = T.build_trial("OP02", registry, T.sha256_file(REGISTRY), utc=UTC)
    with pytest.raises(T.TrialRefused):
        T.write_trial(rec, out)


def test_runner_refuses_a_governed_destination():
    registry = T.load_registry(REGISTRY)
    rec = T.build_trial("OP02", registry, T.sha256_file(REGISTRY), utc=UTC)
    for governed in ("registers", "claims", "drive", "governance"):
        with pytest.raises(T.TrialRefused):
            T.write_trial(rec, os.path.join(ROOT, governed, "trials_should_not_exist"))
        assert not os.path.exists(os.path.join(ROOT, governed, "trials_should_not_exist"))


def test_runner_refuses_to_write_a_record_that_claims_a_status(monkeypatch):
    registry = T.load_registry(REGISTRY)
    sha = T.sha256_file(REGISTRY)
    monkeypatch.setattr(T, "GENERIC_DOES_NOT_ESTABLISH",
                        list(T.GENERIC_DOES_NOT_ESTABLISH) + [
                            "With this record the displayed identity is PROVEN and the gate "
                            "is CLOSED."])
    with pytest.raises(T.TrialRefused, match=r"status words \['CLOSED', 'PROVEN'\]"):
        T.build_trial("OP02", registry, sha, utc=UTC)


def test_runner_refuses_to_write_a_record_that_claims_usefulness(monkeypatch):
    registry = T.load_registry(REGISTRY)
    sha = T.sha256_file(REGISTRY)
    monkeypatch.setitem(T.OP_DOES_NOT_ESTABLISH, "OP03",
                        "The pushforward has high utility in every lane of the program.")
    with pytest.raises(T.TrialRefused, match="speaks of usefulness"):
        T.build_trial("OP03", registry, sha, utc=UTC)
    # NOT_RUN records quote git_side.why as reason_not_run: the same rule applies
    entries = {e["operation_id"]: e for e in registry["entries"]}
    entries["OP01"]["git_side"]["why"] += " Its novelty is substantial."
    with pytest.raises(T.TrialRefused, match="speaks of usefulness"):
        T.build_trial("OP01", registry, sha, utc=UTC)


def test_op04_reproduces_exponent_arithmetic_only():
    """The tautological third step (a Monomial power equal by construction)
    is gone; the record says in its own words what it reproduces."""
    steps, failure = T.evaluate(T.IDENTITIES["OP04"])
    assert failure is None and len(steps) == 2
    assert "exponent arithmetic only" in T.IDENTITIES["OP04"]["reproduces"]
    assert "is not checked" in T.IDENTITIES["OP04"]["reproduces"]
    registry = T.load_registry(REGISTRY)
    rec = T.build_trial("OP04", registry, T.sha256_file(REGISTRY), utc=UTC)
    assert rec["Verification cost"] == 2
    assert rec["Run evidence"]["reproduces"] == T.IDENTITIES["OP04"]["reproduces"]
    assert any("exponent arithmetic only" in s for s in rec["does_not_establish"])


def test_usefulness_scan_unit():
    ok = ("Utility remains UNMEASURED and Novelty remains NOT_ASSESSED, as the register says; "
          "this trial measures neither.")
    assert T.usefulness_hits(ok) == []
    assert T.usefulness_hits("The register's Next useful step names the stored matrix.") == []
    assert T.usefulness_hits("This is useful.") == [
        "'useful' without the register's UNMEASURED / NOT_ASSESSED"]
    assert T.usefulness_hits("Utility is HIGH although the register says UNMEASURED.") == [
        "'Utility' paired with the value word 'HIGH'"]
    assert T.usefulness_hits("The Next useful step has HIGH utility.") == [
        "'utility' paired with the value word 'HIGH'"]
    # the sentence, not the whole text, must carry the register's words
    assert T.usefulness_hits("Utility: UNMEASURED. The novelty is plain to see.") == [
        "'novelty' without the register's UNMEASURED / NOT_ASSESSED"]
    # column names are the register's words, as keys or as exact values
    assert T.status_word_hits({"Verified result": "x", "columns": ["Verified result"]}) == []
    assert T.status_word_hits({"Run evidence": {"VERIFIED": True}}) == ["VERIFIED"]
    assert T.status_word_hits("unverified and unverifiable") == []


def test_runner_refuses_unknown_op_and_bad_utc():
    registry = T.load_registry(REGISTRY)
    with pytest.raises(T.TrialRefused):
        T.build_trial("OP99", registry, "0" * 64, utc=UTC)
    with pytest.raises(T.TrialRefused):
        T.build_trial("OP02", registry, "0" * 64, utc="yesterday")
    r = run_trial("--op", "OP2", "--out", "/nonexistent")
    assert r.returncode == 2


# ---------------------------------------------------------------------------
# append-only against git
# ---------------------------------------------------------------------------

def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


@pytest.fixture
def git_sandbox(sandbox):
    repo = sandbox.root
    if _git(repo, "init", "-q").returncode != 0:
        pytest.skip("git unavailable")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "fixture")
    _git(repo, "add", "-A")
    out = _git(repo, "commit", "-qm", "fixture: trials committed")
    assert out.returncode == 0, out.stderr
    return sandbox


def test_positive_committed_trials_pass_against_git(git_sandbox):
    assert_passes(git_sandbox.run(git=True))


def test_committed_trial_rewritten_is_refused(git_sandbox):
    git_sandbox.mutate_trial("OP02", lambda rec: rec.__setitem__(
        "Notes", rec["Notes"] + " (edited after commit)"))
    assert_fails(git_sandbox.run(git=True), "differs from its git HEAD content")


def test_committed_trial_deleted_is_refused(git_sandbox):
    os.remove(git_sandbox.trial_for("OP03"))
    assert_fails(git_sandbox.run(git=True), "present in git HEAD and deleted from the working tree")


def test_no_git_flag_skips_only_the_git_comparison(git_sandbox):
    os.remove(git_sandbox.trial_for("OP03"))
    assert_passes(git_sandbox.run(git=False))


# ---------------------------------------------------------------------------
# what the text says about itself
# ---------------------------------------------------------------------------

def test_no_repository_authored_text_uses_status_words():
    reg = load(REGISTRY)
    authored = {"authority": reg["authority"], "utility_and_novelty": reg["utility_and_novelty"],
                "does_not_establish": reg["does_not_establish"],
                "git_side": [e["git_side"] for e in reg["entries"]]}
    assert T.status_word_hits(authored) == []
    assert T.usefulness_hits_in(authored) == []
    assert "RATIFIED" in T.STATUS_WORDS and "VERIFIED" in T.STATUS_WORDS \
        and "PASSED" in T.STATUS_WORDS
    # round 2: the verb and noun forms of the rule-1 words are listed too
    for w in ("PROVES", "PROOF", "CERTIFIES", "CERTIFICATE", "CLOSES", "CLOSURE",
              "PROMOTES", "DISCHARGES", "PASSES", "ESTABLISHED"):
        assert w in T.STATUS_WORDS, w
    for w in ("valuable", "beneficial", "improves", "important"):
        assert w in T.USEFULNESS_WORDS, w
    # round 3: promotion / release vocabulary and words of standing
    for w in ("CANONICAL", "FINAL", "AUTHORITATIVE", "OFFICIAL", "RELEASED", "SETTLED",
              "HOLDS", "RESOLVED"):
        assert w in T.STATUS_WORDS, w
    assert "beyond question" in T.VALUE_WORDS
    cells = {e["operation_id"]: e["cells"] for e in reg["entries"]}
    for fn in os.listdir(TRIALS):
        if not fn.endswith(".json"):
            continue
        rec = load(os.path.join(TRIALS, fn))
        assert rec["Verified result"] in T.VERIFIED_RESULTS
        assert "not evidence" in rec["authority"]
        assert any("UNMEASURED" in s and "NOT_ASSESSED" in s for s in rec["does_not_establish"])
        authored = T.authored_fields(rec, cells[rec["Operation ID"]])
        assert T.status_word_hits(authored) == []
        assert T.usefulness_hits_in(authored) == []


# ---------------------------------------------------------------------------
# round-2 adversarial mutations: verb / noun forms, usefulness synonyms, and
# the NOT_RUN and metadata rules SCHEMA.md states but the checker did not pin
# ---------------------------------------------------------------------------

ROUND2_STATUS_SENTENCE = (
    "This trial proves the displayed identity, certifies the constant, closes the "
    "obligation, promotes the claim and discharges the premise; it is a certificate and a "
    "proof, and external review passes it.")


def test_trial_round_two_status_sentence_is_refused(sandbox):
    """Round-2 mutation 05b: the verb forms and nouns of the rule-1 words
    passed a participle-only list."""
    sandbox.mutate_trial("OP02", lambda rec: rec["does_not_establish"].append(
        ROUND2_STATUS_SENTENCE))
    assert_fails(sandbox.run(), "status words ['CERTIFICATE', 'CERTIFIES', 'CLOSES', "
                                "'DISCHARGES', 'PASSES', 'PROMOTES', 'PROOF', 'PROVES']")


@pytest.mark.parametrize("word", ["proves", "proved", "proof", "certifies", "certify",
                                  "certificate", "certification", "closes", "closure",
                                  "promotes", "promotion", "discharges", "discharge",
                                  "passes", "pass", "established", "establishes",
                                  "approval", "verification", "acceptance", "validation",
                                  "confirmation", "ratifies", "verifies"])
def test_trial_status_verb_and_noun_forms_are_refused(sandbox, word):
    sandbox.mutate_trial("OP03", lambda rec: rec["does_not_establish"].append(
        f"After this record the Jacobian identity {word} its way through every gate of the "
        "program."))
    assert_fails(sandbox.run(), f"repository-authored fields use status words ['{word.upper()}']")


def test_trial_status_verb_form_is_refused_by_the_runner_too(monkeypatch):
    registry = T.load_registry(REGISTRY)
    sha = T.sha256_file(REGISTRY)
    monkeypatch.setitem(T.OP_DOES_NOT_ESTABLISH, "OP02",
                        "This record proves the power ledger and closes the obligation on it.")
    with pytest.raises(T.TrialRefused, match=r"status words \['CLOSES', 'PROVES'\]"):
        T.build_trial("OP02", registry, sha, utc=UTC)


def test_trial_established_beyond_doubt_is_refused(sandbox):
    """Round-2 mutation 32 passed because ESTABLISHED was not listed. The
    list is still a word list: a status claimed in words outside it is not
    caught, and the record's does_not_establish is what carries the weight."""
    sandbox.mutate_trial("OP04", lambda rec: rec["does_not_establish"].append(
        "Status of the identity after this trial: established beyond doubt, and the gate is "
        "now open no more."))
    assert_fails(sandbox.run(), "status words ['ESTABLISHED']")


ROUND2_USEFULNESS_SENTENCE = (
    "This operation is valuable, beneficial and important; every session should retrieve it "
    "first and it improves every lane it touches.")


def test_trial_round_two_usefulness_sentence_is_refused(sandbox):
    """Round-2 mutation 06: usefulness synonyms passed the word list."""
    sandbox.mutate_trial("OP02", lambda rec: rec["does_not_establish"].append(
        ROUND2_USEFULNESS_SENTENCE))
    assert_fails(sandbox.run(), "speak of 'valuable' without the register's UNMEASURED / "
                                "NOT_ASSESSED")


@pytest.mark.parametrize("word", ["valuable", "beneficial", "benefit", "improves",
                                  "improvement", "helpful", "important", "worthwhile",
                                  "effective"])
def test_trial_usefulness_synonyms_are_refused(sandbox, word):
    sandbox.mutate_trial("OP05", lambda rec: rec["does_not_establish"].append(
        f"Retrieving this operation first is {word} for every consumer in the program."))
    assert_fails(sandbox.run(), f"speak of '{word}' without the register's UNMEASURED / "
                                "NOT_ASSESSED")


def test_registry_git_side_usefulness_synonym_is_refused(sandbox):
    sandbox.mutate_registry(lambda reg: reg["entries"][2]["git_side"].__setitem__(
        "why", reg["entries"][2]["git_side"]["why"] + " It is beneficial to every lane."))
    sandbox.rebind_trials()
    assert_fails(sandbox.run(), "git_side speaks of 'beneficial' without the register's "
                                "UNMEASURED / NOT_ASSESSED")


@pytest.fixture
def not_run_sandbox(sandbox):
    """A sandbox holding, in addition to the four run records, a NOT_RUN record
    for OP01 written by the runner; it passes before any mutation."""
    r = run_trial("--op", "OP01", "--registry", sandbox.registry, "--out", sandbox.trials,
                  "--utc", UTC)
    assert r.returncode == 0, r.stdout + r.stderr
    assert_passes(sandbox.run())
    return sandbox


def test_not_run_with_a_verification_cost_is_refused(not_run_sandbox):
    """Round-2 mutation 02: a fabricated count of checked equalities on an
    operation that was never run passed."""
    not_run_sandbox.mutate_trial("OP01", lambda rec: rec.__setitem__("Verification cost", 7))
    assert_fails(not_run_sandbox.run(), "NOT_RUN must carry Verification cost 'NOT_MEASURED', "
                                        "not 7")


def test_not_run_with_a_zero_verification_cost_is_refused(not_run_sandbox):
    not_run_sandbox.mutate_trial("OP01", lambda rec: rec.__setitem__("Verification cost", 0))
    assert_fails(not_run_sandbox.run(), "NOT_RUN must carry Verification cost 'NOT_MEASURED', "
                                        "not 0")


def test_not_run_with_an_exposed_development_split_is_refused(not_run_sandbox):
    """Round-2 mutation 03 (Split half)."""
    not_run_sandbox.mutate_trial("OP01", lambda rec: rec.__setitem__(
        "Split", "EXPOSED_DEVELOPMENT"))
    assert_fails(not_run_sandbox.run(), "NOT_RUN must carry Split 'NOT_APPLICABLE', not "
                                        "'EXPOSED_DEVELOPMENT'")


def test_not_run_with_a_replay_arm_is_refused(not_run_sandbox):
    """Round-2 mutation 03 (Arm half)."""
    not_run_sandbox.mutate_trial("OP01", lambda rec: rec.__setitem__(
        "Arm", "NONE_SINGLE_EXACT_REPLAY"))
    assert_fails(not_run_sandbox.run(), "NOT_RUN must carry Arm 'NOT_APPLICABLE', not "
                                        "'NONE_SINGLE_EXACT_REPLAY'")


def test_not_run_dressed_as_a_run_is_refused_three_ways(not_run_sandbox):
    """Mutations 02 and 03 together: every fabricated field is named."""
    def dress(rec):
        rec["Verification cost"] = 7
        rec["Split"] = "EXPOSED_DEVELOPMENT"
        rec["Arm"] = "NONE_SINGLE_EXACT_REPLAY"
    not_run_sandbox.mutate_trial("OP01", dress)
    r = not_run_sandbox.run()
    assert_fails(r, "NOT_RUN must carry Verification cost", "NOT_RUN must carry Split",
                 "NOT_RUN must carry Arm")
    assert SUMMARY_RE.search(r.stdout).group(4) == "3"


def test_not_run_reason_must_be_the_catalog_reason_verbatim(not_run_sandbox):
    """Round-2 mutation 14: a bland reason_not_run passed."""
    not_run_sandbox.mutate_trial("OP01", lambda rec: rec["Run evidence"].__setitem__(
        "reason_not_run", "This repository chose not to run it."))
    assert_fails(not_run_sandbox.run(), "reason_not_run is not the catalog's git_side.why for "
                                        "OP01 verbatim")


def test_not_run_reason_paraphrased_is_refused(not_run_sandbox):
    def paraphrase(rec):
        why = rec["Run evidence"]["reason_not_run"]
        rec["Run evidence"]["reason_not_run"] = why.rstrip(".") + " (paraphrased)."
    not_run_sandbox.mutate_trial("OP01", paraphrase)
    assert_fails(not_run_sandbox.run(), "reason_not_run is not the catalog's git_side.why")


def test_run_with_a_not_applicable_split_is_refused(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec.__setitem__("Split", "NOT_APPLICABLE"))
    assert_fails(sandbox.run(), "a run must carry Split 'EXPOSED_DEVELOPMENT', not "
                                "'NOT_APPLICABLE'")


def test_run_with_a_not_applicable_arm_is_refused(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec.__setitem__("Arm", "NOT_APPLICABLE"))
    assert_fails(sandbox.run(), "a run must carry Arm 'NONE_SINGLE_EXACT_REPLAY', not "
                                "'NOT_APPLICABLE'")


def test_run_with_a_reason_not_run_is_refused(sandbox):
    sandbox.mutate_trial("OP03", lambda rec: rec["Run evidence"].__setitem__(
        "reason_not_run", "ran anyway"))
    assert_fails(sandbox.run(), "a run carries reason_not_run null, not 'ran anyway'")


@pytest.mark.parametrize("col,value", [("Search cost", 5), ("Retrieval cost", 2),
                                       ("Acquisition cost", 0), ("Maintenance cost", 1)])
def test_unmeasured_cost_given_a_number_is_refused(sandbox, col, value):
    """Round-2 mutation 04: SCHEMA.md permitted an integer, so a hand-edited
    record could claim a search or retrieval cost nothing here measures."""
    sandbox.mutate_trial("OP03", lambda rec: rec.__setitem__(col, value))
    assert_fails(sandbox.run(), f"{col} must be exactly 'NOT_MEASURED'")


def test_unmeasured_costs_on_a_not_run_record_are_pinned_too(not_run_sandbox):
    not_run_sandbox.mutate_trial("OP01", lambda rec: rec.__setitem__("Search cost", 5))
    assert_fails(not_run_sandbox.run(), "Search cost must be exactly 'NOT_MEASURED'")


def test_run_evidence_arithmetic_rewritten_is_refused(sandbox):
    """Round-2 mutation 24: the arithmetic sentence could describe floats
    while the steps were exact."""
    sandbox.mutate_trial("OP05", lambda rec: rec["Run evidence"].__setitem__(
        "arithmetic", "floating point at 200 digits; NON-CERTIFYING"))
    assert_fails(sandbox.run(), "Run evidence.arithmetic is not the runner's fixed sentence")


def test_budget_sentence_rewritten_is_refused(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec.__setitem__(
        "Budget / cost unit", "unit=checked_equalities; budget=1000; failure rule: retry"))
    assert_fails(sandbox.run(), "Budget / cost unit is not the runner's fixed sentence")


def test_op05_reproduces_the_displayed_witness_only():
    """Round-2 minor finding: the fourth step (LHS <= corrected RHS of OP04)
    was exact and true but not displayed in the OP05 cell. It is gone; the
    spec says in its own words that the corrected bound is not evaluated."""
    steps, failure = T.evaluate(T.IDENTITIES["OP05"])
    assert failure is None and len(steps) == 3
    assert all("corrected" not in s["statement"] for s in steps)
    assert "not evaluated" in T.IDENTITIES["OP05"]["reproduces"]
    registry = T.load_registry(REGISTRY)
    rec = T.build_trial("OP05", registry, T.sha256_file(REGISTRY), utc=UTC)
    assert rec["Verification cost"] == 3
    assert rec["Run evidence"]["reproduces"] == T.IDENTITIES["OP05"]["reproduces"]
    assert any("not evaluated" in s for s in rec["does_not_establish"])
    # the committed OP05 record is this record's arithmetic
    committed = load(os.path.join(TRIALS, [fn for fn in os.listdir(TRIALS)
                                            if fn.startswith("TRIAL-OP05-")][0]))
    assert committed["Verification cost"] == 3
    assert [s["statement"] for s in committed["Run evidence"]["steps"]] == \
        [s["statement"] for s in steps]


def test_status_word_scan_is_whole_word_and_case_insensitive():
    assert T.status_word_hits("It Proves and CLOSES.") == ["CLOSES", "PROVES"]
    assert T.status_word_hits("closely, provenance, compass, passage, improvement") == []
    assert T.status_word_hits("does_not_establish") == []
    assert T.status_word_hits("It does not establish that; it establishes nothing.") == \
        ["ESTABLISHES"]
    assert T.usefulness_hits("This is valuable and important.") == [
        "'valuable' without the register's UNMEASURED / NOT_ASSESSED"]
    assert T.usefulness_hits("An improvement of substantial size.") == [
        "'improvement' paired with the value word 'substantial'"]


# ---------------------------------------------------------------------------
# round-3 adversarial mutations: a fabricated run time, the unpinned
# load-bearing text of both record kinds, a free registry authority and
# provenance, a hand-written NOT_RUN for a checkable operation, the
# environment block, and foreign files under trials/
# ---------------------------------------------------------------------------

FUTURE_UTC = "2099-01-01T00:00:00Z"
CLOCK_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


def restamp(sandbox, op_id, utc):
    """Give the committed trial for ``op_id`` a new utc, recompute its Trial ID
    and rename the file, so the only problem the checker can report is the
    stamp itself."""
    p = sandbox.trial_for(op_id)
    rec = load(p)
    rec["Run evidence"]["utc"] = utc
    rec["Trial ID"] = T.trial_id(op_id, rec["Problem ID / SHA-256"],
                                 rec["Library / catalog SHA-256"],
                                 rec["Run evidence"]["identity_spec"], utc)
    os.remove(p)
    dump(os.path.join(sandbox.trials, rec["Trial ID"] + ".json"), rec)
    return rec


def test_positive_committed_records_were_stamped_by_the_clock():
    """Round-3 BLOCKING finding: the four records once carried 14:30:00Z, a
    stamp supplied through the tests-only --utc option, 78 minutes after the
    files were written. The records in the tree carry the runner's clock
    (microsecond precision, never later than now) and name a commit."""
    now = datetime.datetime.now(datetime.timezone.utc)
    seen = 0
    for fn in sorted(os.listdir(TRIALS)):
        if not fn.endswith(".json"):
            continue
        seen += 1
        rec = load(os.path.join(TRIALS, fn))
        utc = rec["Run evidence"]["utc"]
        assert CLOCK_UTC_RE.match(utc), f"{fn}: {utc!r} is not a clock stamp"
        assert T.parse_utc(utc) <= now, f"{fn}: {utc} is in the future"
        assert utc != "2026-09-19T14:30:00Z"
        commit = rec["Seed / environment"]["repository_commit_at_run"]
        assert commit is None or T.COMMIT_RE.match(commit)
        assert list(rec["Seed / environment"].keys()) == list(T.ENVIRONMENT_KEYS)
    assert seen >= 4


def test_trial_dated_in_the_future_is_refused(sandbox):
    """Round-3 mutation 02: a utc of 2099 with the Trial ID recomputed passed."""
    restamp(sandbox, "OP02", FUTURE_UTC)
    r = sandbox.run()
    assert_fails(r, "Run evidence.utc 2099-01-01T00:00:00Z is later than this check's clock",
                 "a run in the future did not happen")
    assert SUMMARY_RE.search(r.stdout).group(4) == "1"


def test_runner_refuses_a_future_utc(tmp_path):
    out = str(tmp_path / "trials")
    r = run_trial("--op", "OP02", "--out", out, "--utc", FUTURE_UTC)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "REFUSED OP02" in r.stdout and "later than the clock" in r.stdout
    assert not os.path.exists(out) or os.listdir(out) == []
    registry = T.load_registry(REGISTRY)
    with pytest.raises(T.TrialRefused, match="later than the clock"):
        T.build_trial("OP02", registry, T.sha256_file(REGISTRY), utc=FUTURE_UTC)


def test_runner_stamps_the_clock_when_no_utc_is_supplied(tmp_path):
    out = str(tmp_path / "trials")
    before = datetime.datetime.now(datetime.timezone.utc)
    r = run_trial("--op", "OP04", "--out", out)
    assert r.returncode == 0, r.stdout + r.stderr
    (fn,) = os.listdir(out)
    utc = load(os.path.join(out, fn))["Run evidence"]["utc"]
    assert CLOCK_UTC_RE.match(utc)
    assert before.replace(microsecond=0) <= T.parse_utc(utc) \
        <= datetime.datetime.now(datetime.timezone.utc)


def test_committed_trial_dated_after_its_commit_is_refused(sandbox):
    """A record cannot be committed before it was run. Commit the sandbox with
    a committer date earlier than the records' utc: every committed record is
    refused, by name, with the commit time in the message."""
    repo = sandbox.root
    if _git(repo, "init", "-q").returncode != 0:
        pytest.skip("git unavailable")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "fixture")
    _git(repo, "add", "-A")
    env = dict(os.environ, GIT_COMMITTER_DATE="2026-09-01T00:00:00Z",
               GIT_AUTHOR_DATE="2026-09-01T00:00:00Z")
    out = subprocess.run(["git", "commit", "-qm", "fixture: dated before the runs"], cwd=repo,
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    r = sandbox.run(git=True)
    assert_fails(r, "is later than the commit that added the record "
                    "(2026-09-01T00:00:00.000000Z)",
                 "cannot follow its own commit")
    assert r.stdout.count("later than the commit that added the record") == \
        len(sandbox.trial_paths())
    # the floor is a git check; the stamps themselves are in the past
    assert_passes(sandbox.run(git=False))


def test_committed_trial_run_in_the_commit_second_passes(sandbox):
    """git keeps commit times to the second; a record run and committed
    within the same second is not a record from the future."""
    repo = sandbox.root
    if _git(repo, "init", "-q").returncode != 0:
        pytest.skip("git unavailable")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "fixture")
    rec = restamp(sandbox, "OP02", "2026-09-10T12:00:00.999999Z")
    _git(repo, "add", "-A")
    env = dict(os.environ, GIT_COMMITTER_DATE="2026-09-10T12:00:00Z",
               GIT_AUTHOR_DATE="2026-09-10T12:00:00Z")
    out = subprocess.run(["git", "commit", "-qm", "fixture: same second"], cwd=repo,
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    r = sandbox.run(git=True)
    # the other records were run after 2026-09-10 and are refused; this one is not
    assert rec["Trial ID"] + ".json" not in \
        "".join(line for line in r.stdout.splitlines() if "later than the commit" in line)


def test_parse_utc_unit():
    assert T.parse_utc("2026-09-19T00:00:00Z") == datetime.datetime(
        2026, 9, 19, tzinfo=datetime.timezone.utc)
    assert T.parse_utc("2026-09-19T00:00:00.5Z") == datetime.datetime(
        2026, 9, 19, 0, 0, 0, 500000, tzinfo=datetime.timezone.utc)
    assert T.parse_utc("2026-09-19T00:00:00.000001Z").microsecond == 1
    assert T.parse_utc("2026-13-01T00:00:00Z") is None
    assert T.parse_utc("yesterday") is None and T.parse_utc(None) is None
    assert T.parse_utc(T.utc_now()) is not None


# --- does_not_establish is pinned (mutations 06 and 15) ---------------------

def test_trial_generic_sentence_deleted_is_refused(sandbox):
    """Round-3 mutation 06, one half: a generic sentence could be deleted."""
    sandbox.mutate_trial("OP04", lambda rec: rec["does_not_establish"].remove(
        T.GENERIC_DOES_NOT_ESTABLISH[0]))
    r = sandbox.run()
    assert_fails(r, "does_not_establish is missing a generic sentence the runner writes",
                 "never removed or replaced")
    assert SUMMARY_RE.search(r.stdout).group(4) == "1"


def test_trial_does_not_establish_replaced_by_a_bland_sentence_is_refused(sandbox):
    """Round-3 mutation 06."""
    sandbox.mutate_trial("OP04", lambda rec: rec.__setitem__("does_not_establish", [
        "This record is a record of a record of a computation and nothing more."]))
    r = sandbox.run()
    assert_fails(r, "missing a generic sentence", "missing the OP04 sentence")
    assert r.stdout.count("does_not_establish is missing") == \
        len(T.required_does_not_establish("OP04", "IDENTITY_REPRODUCED")) == 5


def test_trial_op_sentence_deleted_is_refused(sandbox):
    sandbox.mutate_trial("OP05", lambda rec: rec["does_not_establish"].remove(
        T.OP_DOES_NOT_ESTABLISH["OP05"]))
    assert_fails(sandbox.run(), "missing the OP05 sentence the runner writes")


def test_trial_op_sentence_paraphrased_is_refused(sandbox):
    def para(rec):
        i = rec["does_not_establish"].index(T.OP_DOES_NOT_ESTABLISH["OP02"])
        rec["does_not_establish"][i] = rec["does_not_establish"][i].replace(
            "is not justified here", "is justified elsewhere")
        assert rec["does_not_establish"][i] != T.OP_DOES_NOT_ESTABLISH["OP02"]
    sandbox.mutate_trial("OP02", para)
    assert_fails(sandbox.run(), "missing the OP02 sentence")


def test_trial_extra_does_not_establish_sentence_is_allowed_and_scanned(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec["does_not_establish"].append(
        "Nothing recorded here says anything about the far field or the fold constant."))
    assert_passes(sandbox.run())
    sandbox.mutate_trial("OP02", lambda rec: rec["does_not_establish"].append(
        "The extra sentence above is where the identity is CERTIFIED."))
    assert_fails(sandbox.run(), "status words ['CERTIFIED']")


def test_not_run_sentence_deleted_is_refused(not_run_sandbox):
    not_run_sandbox.mutate_trial("OP01", lambda rec: rec["does_not_establish"].remove(
        T.NOT_RUN_DOES_NOT_ESTABLISH))
    assert_fails(not_run_sandbox.run(), "missing the NOT_RUN sentence the runner writes")


def test_runner_writes_exactly_the_required_sentences():
    registry = T.load_registry(REGISTRY)
    sha = T.sha256_file(REGISTRY)
    for op_id, verdict in (("OP02", "IDENTITY_REPRODUCED"), ("OP07", "NOT_RUN")):
        rec = T.build_trial(op_id, registry, sha, utc=UTC)
        assert rec["Verified result"] == verdict
        assert rec["does_not_establish"] == T.required_does_not_establish(op_id, verdict)
    assert T.NOT_RUN_DOES_NOT_ESTABLISH in T.required_does_not_establish("OP07", "NOT_RUN")
    assert T.NOT_RUN_DOES_NOT_ESTABLISH not in T.required_does_not_establish(
        "OP02", "IDENTITY_REPRODUCED")


def test_registry_does_not_establish_replaced_by_a_bland_sentence_is_refused(sandbox):
    """Round-3 mutation 15."""
    sandbox.mutate_registry(lambda reg: reg.__setitem__("does_not_establish", [
        "The entries of this file are the entries of this file, nothing more and nothing "
        "less."]))
    sandbox.rebind_trials()
    r = sandbox.run()
    assert_fails(r, "does_not_establish is missing trial.py's fixed sentence "
                    "REGISTRY_DOES_NOT_ESTABLISH[0]")
    assert r.stdout.count("REGISTRY_DOES_NOT_ESTABLISH[") == \
        len(T.REGISTRY_DOES_NOT_ESTABLISH) == 4


def test_registry_does_not_establish_sentence_deleted_is_refused(sandbox):
    sandbox.mutate_registry(lambda reg: reg["does_not_establish"].pop(2))
    sandbox.rebind_trials()
    r = sandbox.run()
    assert_fails(r, "REGISTRY_DOES_NOT_ESTABLISH[2]")
    assert SUMMARY_RE.search(r.stdout).group(4) == "1"


def test_registry_extra_does_not_establish_sentence_is_allowed(sandbox):
    sandbox.mutate_registry(lambda reg: reg["does_not_establish"].append(
        "Nothing here says anything about any lane other than the register's own cells."))
    sandbox.rebind_trials()
    assert_passes(sandbox.run())


# --- registry authority and top-level sentences are pinned (mutation 03) ----

def test_registry_authority_canonical_and_final_is_refused(sandbox):
    """Round-3 mutation 03: a registry naming the operator and calling its
    entries canonical and final passed a non-empty check."""
    sandbox.mutate_registry(lambda reg: reg.__setitem__(
        "authority", "Dylan Roy — these fifteen entries are canonical and final for every "
                     "lane of the program"))
    sandbox.rebind_trials()
    assert_fails(sandbox.run(), "authority must be exactly trial.py's REGISTRY_AUTHORITY",
                 "top-level text uses status words ['CANONICAL', 'FINAL']")


def test_registry_authority_paraphrased_is_refused(sandbox):
    sandbox.mutate_registry(lambda reg: reg.__setitem__(
        "authority", reg["authority"].replace("no mathematical authority", "no authority")))
    sandbox.rebind_trials()
    r = sandbox.run()
    assert_fails(r, "authority must be exactly trial.py's REGISTRY_AUTHORITY")
    assert SUMMARY_RE.search(r.stdout).group(4) == "1"


def test_registry_utility_and_novelty_sentence_is_pinned(sandbox):
    sandbox.mutate_registry(lambda reg: reg.__setitem__(
        "utility_and_novelty", "Utility is UNMEASURED and Novelty is NOT_ASSESSED for now; "
                               "the trials in this directory will change that."))
    sandbox.rebind_trials()
    assert_fails(sandbox.run(), "utility_and_novelty must be exactly trial.py's "
                                "REGISTRY_UTILITY_AND_NOVELTY")


def test_registry_top_level_constants_are_the_file(sandbox):
    reg = load(REGISTRY)
    assert reg["authority"] == T.REGISTRY_AUTHORITY
    assert reg["utility_and_novelty"] == T.REGISTRY_UTILITY_AND_NOVELTY
    assert reg["does_not_establish"] == T.REGISTRY_DOES_NOT_ESTABLISH
    assert reg["register"]["export"] == T.REGISTER_EXPORT
    assert reg["register"]["row_sha256_rule"] == T.ROW_SHA256_RULE
    assert reg["trial_ledger"]["columns"] == list(T.TRIAL_COLUMNS)
    assert reg["trial_ledger"]["record_schema"] == T.SCHEMA


@pytest.mark.parametrize("word", ["CANONICAL", "FINAL", "AUTHORITATIVE", "OFFICIAL",
                                  "RELEASED", "SETTLED", "HOLDS", "RESOLVED"])
def test_trial_promotion_and_standing_words_are_refused(sandbox, word):
    sandbox.mutate_trial("OP03", lambda rec: rec.__setitem__(
        "Notes", rec["Notes"] + f" With this record the pushforward adapter is {word} for "
                                "every application in the program."))
    assert_fails(sandbox.run(), f"repository-authored fields use status words ['{word}']")


def test_trial_round_three_status_sentence_is_refused(sandbox):
    """Round-3 mutation 10: 'settled for good and holds in every application'
    used no listed word. SETTLED and HOLDS are listed now; the list is still
    a list, and the pinned does_not_establish is what carries the weight."""
    sandbox.mutate_trial("OP03", lambda rec: rec.__setitem__(
        "Notes", rec["Notes"] + " With this record the pushforward adapter is settled for good "
                                "and holds in every application."))
    assert_fails(sandbox.run(), "status words ['HOLDS', 'SETTLED']")


def test_trial_beyond_question_is_a_value_phrase(sandbox):
    """Round-3 mutation 16: the register's marker in the same sentence exempted
    'the usefulness of this ledger ... is beyond question'."""
    sandbox.mutate_trial("OP03", lambda rec: rec["does_not_establish"].append(
        "Utility is UNMEASURED by the register, yet the usefulness of this ledger to every "
        "fold computation is beyond question."))
    assert_fails(sandbox.run(), "speak of 'Utility' paired with the value word 'beyond question'")
    assert T.usefulness_hits("Novelty is NOT_ASSESSED, and the gain is beyond doubt.") == [
        "'Novelty' paired with the value word 'beyond doubt'"]


# --- a hand-written NOT_RUN for a checkable operation (mutation 04) ---------

def test_not_run_record_for_a_checkable_op_is_refused(sandbox):
    """Round-3 mutation 04: the runner never writes NOT_RUN for an
    EXACT_IDENTITY operation, and the checker did not refuse one."""
    entry = {e["operation_id"]: e for e in load(sandbox.registry)["entries"]}["OP02"]
    rec = load(sandbox.trial_for("OP02"))
    rec["Verified result"] = "NOT_RUN"
    rec["Split"] = rec["Arm"] = "NOT_APPLICABLE"
    rec["Verification cost"] = T.NOT_MEASURED
    ev = rec["Run evidence"]
    ev.update({"identity_displayed": None, "reproduces": None, "identity_spec": None,
               "steps": [], "checked_equalities": 0, "all_equal": False, "failure": None,
               "reason_not_run": entry["git_side"]["why"]})
    rec["does_not_establish"] = T.required_does_not_establish("OP02", "NOT_RUN")
    rec["Trial ID"] = T.trial_id("OP02", rec["Problem ID / SHA-256"],
                                 rec["Library / catalog SHA-256"], None, ev["utc"])
    dump(os.path.join(sandbox.trials, rec["Trial ID"] + ".json"), rec)
    r = sandbox.run()
    assert_fails(r, "OP02 is EXACT_IDENTITY in the catalog the record names and the runner "
                    "never records NOT_RUN for it")
    assert SUMMARY_RE.search(r.stdout).group(4) == "1"


# --- registry provenance blocks are compared, not trusted (mutation 05) -----

def test_registry_register_block_must_name_the_register_it_reads(sandbox):
    """Round-3 mutation 05, register half."""
    def prov(reg):
        reg["register"]["tab"] = "artifact_index"
        reg["register"]["sheet_index"] = 3
        reg["register"]["path"] = "registers/json/artifact_index.json"
    sandbox.mutate_registry(prov)
    sandbox.rebind_trials()
    assert_fails(sandbox.run(),
                 "register.tab is 'artifact_index', but the register this check reads says "
                 "'reusable_operations'",
                 "register.sheet_index is 3, but the register this check reads says 42",
                 "register.path is 'registers/json/artifact_index.json', but the register this "
                 "check reads says 'registers/json/reusable_operations.json'")


def test_registry_trial_ledger_block_must_describe_the_ledger(sandbox):
    """Round-3 mutation 05, ledger half."""
    def prov(reg):
        reg["trial_ledger"]["columns"] = ["Trial ID", "Outcome"]
        reg["trial_ledger"]["records_dir"] = "reviews/"
    sandbox.mutate_registry(prov)
    sandbox.rebind_trials()
    assert_fails(sandbox.run(),
                 "trial_ledger.columns is ['Trial ID', 'Outcome'], but this check reads",
                 "trial_ledger.records_dir is 'reviews/', but this check reads "
                 "'engine/operations/trials'")


@pytest.mark.parametrize("block,key,value", [
    ("register", "export", "GP-REG-032-v1.3 xlsx export of 2026-09-20 (registers/source/)"),
    ("register", "row_sha256_rule", "sha256 of the Operation cell alone"),
    ("trial_ledger", "record_schema", "q0.operation.trial/v2"),
    ("trial_ledger", "path", "registers/json/artifact_index.json"),
    ("trial_ledger", "tab", "artifact_index"),
    ("trial_ledger", "sheet_index", 3),
])
def test_registry_provenance_field_rewritten_is_refused(sandbox, block, key, value):
    sandbox.mutate_registry(lambda reg: reg[block].__setitem__(key, value))
    sandbox.rebind_trials()
    r = sandbox.run()
    assert_fails(r, f"{block}.{key} is {value!r}")
    assert SUMMARY_RE.search(r.stdout).group(4) == "1"


def test_registry_provenance_block_extra_key_is_refused(sandbox):
    sandbox.mutate_registry(lambda reg: reg["register"].__setitem__("status", "current"))
    sandbox.rebind_trials()
    assert_fails(sandbox.run(), "register block keys must be exactly")
    sandbox.mutate_registry(lambda reg: reg["trial_ledger"].__setitem__("rows", 4))
    sandbox.rebind_trials()
    assert_fails(sandbox.run(), "trial_ledger block keys must be exactly")


def test_registry_records_dir_must_be_the_trials_directory(sandbox, tmp_path):
    """Pointing the checker at another trials directory makes records_dir
    disagree: the registry may not describe a ledger the check does not read."""
    other = str(tmp_path / "repo" / "engine" / "operations" / "elsewhere")
    shutil.copytree(sandbox.trials, other)
    r = sandbox.run("--trials", other)
    assert_fails(r, "trial_ledger.records_dir is 'engine/operations/trials/', but this check "
                    "reads 'engine/operations/elsewhere'")


# --- the environment block is the runner's (mutation 07) --------------------

def test_trial_environment_block_reshaped_is_refused(sandbox):
    """Round-3 mutation 07: seed 42, a made-up commit and Python 2.7.18 passed."""
    sandbox.mutate_trial("OP05", lambda rec: rec.__setitem__("Seed / environment", {
        "seed": "42", "repository_commit_at_run": "deadbeef" * 5, "python": "2.7.18"}))
    assert_fails(sandbox.run(), "Seed / environment must be an object with exactly the keys "
                                "['seed', 'python', 'implementation', 'platform', "
                                "'repository_commit_at_run', 'runner'] in order")


def test_trial_environment_values_are_pinned(sandbox):
    def fab(rec):
        env = rec["Seed / environment"]
        env["seed"] = "42"
        env["repository_commit_at_run"] = "not-a-commit"
        env["python"] = "two point seven"
        env["runner"] = "by hand"
    sandbox.mutate_trial("OP05", fab)
    r = sandbox.run()
    assert_fails(r, "Seed / environment.seed is not the runner's fixed sentence",
                 "repository_commit_at_run must be null or a 40-hex commit, not 'not-a-commit'",
                 "Seed / environment.python 'two point seven' is not a version string",
                 "Seed / environment.runner is 'by hand'")
    assert SUMMARY_RE.search(r.stdout).group(4) == "4"


def test_trial_environment_as_a_string_is_refused(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec.__setitem__("Seed / environment",
                                                              "deterministic"))
    assert_fails(sandbox.run(), "Seed / environment must be an object with exactly the keys")


def test_trial_environment_null_commit_is_allowed(sandbox):
    sandbox.mutate_trial("OP02", lambda rec: rec["Seed / environment"].__setitem__(
        "repository_commit_at_run", None))
    assert_passes(sandbox.run())


# --- nothing but records and the README under trials/ (mutation 21) ---------

def test_foreign_file_under_trials_is_refused(sandbox):
    """Round-3 mutation 21: a SUMMARY.md claiming PROVEN and CERTIFIED was
    ignored by every scan."""
    with open(os.path.join(sandbox.trials, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("All four identities PROVEN and CERTIFIED.\n")
    r = sandbox.run()
    assert_fails(r, "SUMMARY.md: only trial records (*.json) and README.md may exist under the "
                    "trials directory")
    assert SUMMARY_RE.search(r.stdout).group(4) == "1"


def test_subdirectory_and_stray_text_under_trials_are_refused(sandbox):
    os.makedirs(os.path.join(sandbox.trials, "archive"))
    with open(os.path.join(sandbox.trials, "notes.txt"), "w", encoding="utf-8") as f:
        f.write("nothing\n")
    r = sandbox.run()
    assert_fails(r, "trials/archive: only trial records", "trials/notes.txt: only trial records")


def test_trials_readme_is_scanned(sandbox):
    with open(os.path.join(sandbox.trials, "README.md"), "a", encoding="utf-8") as f:
        f.write("\nAll four identities are PROVEN and CERTIFIED by the records above, and\n"
                "the operations' utility is high.\n")
    assert_fails(sandbox.run(), "README.md: uses status words ['CERTIFIED', 'PROVEN']",
                 "README.md: speaks of 'utility' paired with the value word 'high'")


def test_trials_readme_line_wrapping_is_not_a_sentence_break(sandbox):
    """A wrapped sentence whose marker sits on the next line is one sentence."""
    with open(os.path.join(sandbox.trials, "README.md"), "a", encoding="utf-8") as f:
        f.write("\nNo record here says whether an operation is useful, because Utility is\n"
                "`UNMEASURED` in the register and stays so.\n")
    assert_passes(sandbox.run())
    with open(os.path.join(sandbox.trials, "README.md"), "a", encoding="utf-8") as f:
        f.write("\nThe operations are useful.\n")
    assert_fails(sandbox.run(), "README.md: speaks of 'useful' without the register's")


def test_markdown_paragraphs_unit():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import operations_check as C  # noqa: E402
    assert C.markdown_paragraphs("a\nb\n\n\nc\n") == ["a b", "c"]
    assert C.markdown_paragraphs("") == []
