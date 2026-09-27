"""Receipt schema, the append-only writer, the checker, and the runner's limits.

Every test below is a NEGATIVE CONTROL unless its name says otherwise: it takes
a receipt that passes, breaks exactly one thing, and asserts that the writer or
`tools/receipts_check.py` refuses it. A control that cannot fail is decoration,
so each one is paired with the positive case it mutates.

The properties under test, in the order the task statement puts them:

* schema validation, including every provenance rule;
* a receipt missing `does_not_establish` is REJECTED by the writer;
* an append-only violation is DETECTED — by hash recomputation, and against
  `git HEAD` for a receipt already committed;
* a receipt whose verdict says DISCHARGED, CLOSED or PROMOTED is REJECTED by
  the checker;
* the runner CANNOT write to `engine/lanes/` or `claims/graph.json`, asserted
  structurally against `engine/run.py`'s parsed syntax tree and functionally
  against the one writer it imports.

None of this is mathematics. A passing suite says the recording machinery
behaves as documented; it says nothing about any number in any receipt, and it
moves no obligation. `OBL-H5-JETMOD`, `OBL-H5-ZBAND` (hi side),
`OBL-H5-REMOTE-THRESHOLD`, `OBL-D1-PROMOTE` and both Pieces of
`D3-LEMMA-RN-UNIF` are OPEN.
"""
from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine import receipt as R  # noqa: E402
from engine import run as RUN  # noqa: E402
from tools import receipts_check as RC  # noqa: E402

RUN_PY = os.path.join(ROOT, "engine", "run.py")
RECEIPT_PY = os.path.join(ROOT, "engine", "receipt.py")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def good_receipt(lane: str = "A1", **over) -> R.Receipt:
    """A receipt that passes every check. Every control below mutates this."""
    from fractions import Fraction as F

    class _Enc:  # duck-typed enclosure: exact rational endpoints
        lo = F(1, 3)
        hi = F(1, 2)

    kwargs = dict(
        lane=lane,
        module="research.bands",
        entry_point="engine.run._task_a1",
        arguments={"prec": 40, "kappa": "1/8"},
        outcome=R.OUTCOME_RAN,
        runtime_seconds=0.125,
        does_not_establish=(
            "This run does not discharge OBL-H5-JETMOD, does not close either "
            "Piece of D3-LEMMA-RN-UNIF, and certifies no jet of this program."),
        results=[
            R.NumericResult.from_interval("enclosure", _Enc(), "reference only"),
            R.NumericResult.from_fraction("exact", F(83_6263, 10_000), "exact"),
            R.NumericResult.from_float("probe", 0.5, "probe cell, not a bound"),
        ],
        notes=["a receipt is a record of a computation, not evidence"],
        commit="0" * 40,
        dirty=False,
        timestamp="2026-09-18T12:00:00.000000Z",
    )
    kwargs.update(over)
    return R.Receipt.build(**kwargs)


def write_object(tmpdir, obj, lane="A1", name=None):
    """Write a receipt object straight to disk, bypassing the writer."""
    rid = name or obj.get("receipt_id", "RCPT-A1-20260918T120000000000Z-deadbeef")
    d = os.path.join(tmpdir, lane)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, rid + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path


def rehash(obj):
    """Recompute `body_sha256` after a mutation, as a forger would."""
    body = {k: v for k, v in obj.items() if k != "body_sha256"}
    obj["body_sha256"] = R.body_sha256(body)
    return obj


def check(tmpdir, lanes_dir=None, root=None, git=False):
    return RC.check(receipts_dir=str(tmpdir),
                    lanes_dir=lanes_dir or RC.DEFAULT_LANES,
                    root=root or str(tmpdir), check_git=git)[0]


# ---------------------------------------------------------------------------
# 1. the positive case
# ---------------------------------------------------------------------------

def test_a_well_formed_receipt_passes_writer_and_checker(tmp_path):
    rec = good_receipt()
    assert R.validate_body(rec.body()) == []
    path = R.write_receipt(rec, str(tmp_path))
    assert os.path.exists(path)
    assert os.path.basename(path) == rec.receipt_id + ".json"
    assert check(tmp_path) == []


def test_shipped_receipts_and_checker_cli_agree():
    """The receipts committed in engine/receipts/ pass their own checker."""
    r = subprocess.run([sys.executable, "tools/receipts_check.py"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "problems=0" in r.stdout


def test_float_results_are_labelled_non_certifying_and_are_not_certifying():
    rec = good_receipt()
    probe = [x for x in rec.results if x.provenance == R.FLOAT_NONCERTIFYING][0]
    assert probe.certifying is False
    assert "NON-CERTIFYING" in probe.note
    for other in rec.results:
        if other is not probe:
            assert other.certifying is True


# ---------------------------------------------------------------------------
# 2. does_not_establish is mandatory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["", "   ", "n/a", "none", "TBD", "-",
                                   "nothing much"])
def test_writer_rejects_a_receipt_without_a_real_does_not_establish(tmp_path, value):
    """NEGATIVE CONTROL: the most load-bearing field cannot be empty or fake."""
    rec = good_receipt(does_not_establish=value)
    with pytest.raises(R.ReceiptRejected) as exc:
        R.write_receipt(rec, str(tmp_path))
    assert "does_not_establish" in str(exc.value)
    assert not os.path.exists(R.receipt_path(rec, str(tmp_path)))


@pytest.mark.parametrize("value", ["", "   ", "\n\t "])
def test_an_empty_does_not_establish_is_refused_as_empty_specifically(value):
    """NEGATIVE CONTROL, pinned to the emptiness rule itself.

    Without this, the length rule below would mask the emptiness rule and a
    mutation removing the emptiness check would survive the suite. Mutation
    testing found exactly that, so the message is asserted, not just the
    refusal.
    """
    problems = R.validate_body(good_receipt(does_not_establish=value).body())
    assert any("does_not_establish is empty" in p for p in problems), problems


def test_the_length_rule_is_a_separate_line_of_defence():
    """A plausible-looking but contentless caveat is still refused."""
    problems = R.validate_body(
        good_receipt(does_not_establish="Nothing much, really.").body())
    assert any("shorter than" in p for p in problems), problems


def test_checker_rejects_a_receipt_whose_does_not_establish_was_removed(tmp_path):
    """NEGATIVE CONTROL, on disk: a hand-edited receipt loses the field."""
    obj = good_receipt().to_dict()
    del obj["does_not_establish"]
    write_object(tmp_path, rehash(obj))
    problems = check(tmp_path)
    assert any("does_not_establish" in p for p in problems), problems


def test_checker_rejects_an_emptied_does_not_establish(tmp_path):
    obj = rehash({**good_receipt().to_dict(), "does_not_establish": ""})
    write_object(tmp_path, obj)
    assert any("does_not_establish" in p for p in check(tmp_path))


# ---------------------------------------------------------------------------
# 3. append-only
# ---------------------------------------------------------------------------

def test_writer_refuses_to_overwrite_an_existing_receipt(tmp_path):
    """NEGATIVE CONTROL: the same id twice is an append-only violation."""
    rec = good_receipt()
    R.write_receipt(rec, str(tmp_path))
    with pytest.raises(R.AppendOnlyViolation):
        R.write_receipt(rec, str(tmp_path))


def test_exclusive_creation_refuses_a_second_write_without_the_pre_check(
        tmp_path, monkeypatch):
    """NEGATIVE CONTROL for the SECOND line of defence, on its own.

    `write_receipt` checks `os.path.exists` first, which would mask the
    exclusive-creation mode if that mode were ever weakened to "w". Mutation
    testing showed the pre-check alone kept the suite green, so here the
    pre-check is disabled and the refusal must still happen -- which it can
    only do if the file is still created with mode "x".
    """
    rec = good_receipt()
    R.write_receipt(rec, str(tmp_path))
    monkeypatch.setattr(R.os.path, "exists", lambda p: False)
    with pytest.raises(R.AppendOnlyViolation):
        R.write_receipt(rec, str(tmp_path))
    # and the file on disk is untouched by the refused write
    with open(R.receipt_path(rec, str(tmp_path)), encoding="utf-8") as f:
        assert json.load(f)["receipt_id"] == rec.receipt_id


def _hand_built_cover_ledger(certified_and_covering: bool):
    """A one-cell bracket ledger, built by hand so no cover run is needed.

    ``True``: the cell is ACCEPTED with a value range, so ``total()`` is a
    certified enclosure of the region. ``False``: the cell is rejected as
    UNRESOLVED_BOUNDARY with no residual, so ``total()`` returns a number about
    an empty accounted part with ``covers_region=False``.
    """
    from fractions import Fraction as F
    from research.cover import Cell, Ledger, RejectKind, rn5_annulus_bracket
    from research.interval import Interval
    reg = rn5_annulus_bracket()
    dom = reg.domain()
    led = Ledger("control", dom, "cartesian", integrand="REFERENCE:radial_gaussian")
    led.add(Cell("c0", dom, 0))
    if certified_and_covering:
        area = Interval.exact(dom.param_area())
        led.accept("c0", area, Interval(F(0), F(1)), area * Interval(F(0), F(1)))
    else:
        led.reject("c0", RejectKind.UNRESOLVED_BOUNDARY,
                   "CONTROL: a straddling cell rejected without a residual",
                   dom.param_area())
    return led


def test_a5_publishes_no_certified_enclosure_when_the_ledger_refuses(monkeypatch):
    """NEGATIVE CONTROL for the A5 task's latent overclaim.

    ``Total.enclosure`` is a number about the ACCOUNTED part; whether it is an
    enclosure of the region integral is what ``certified`` and ``covers_region``
    say. The task used to publish the raw field under ``certified_interval``
    regardless -- invisible on the polar region, which rejects no cells, and a
    false certified enclosure on any region that rejects a cell without a
    residual. Drive the task on such a ledger and assert nothing certified is
    published. Reverting the task to ``total.enclosure`` fails this test.
    """
    import research.cover as C
    led = _hand_built_cover_ledger(certified_and_covering=False)
    total = led.total()
    assert total.covers_region is False          # the premise of the control
    monkeypatch.setattr(C, "run", lambda region, integrand, config: led)

    results, notes = RUN._task_a5()
    by_name = {r.name: r for r in results}
    assert "REFERENCE_cover_total_enclosure" not in by_name
    assert "REFERENCE_cover_closed_form_crosscheck" not in by_name
    assert all(not (r.provenance == R.CERTIFIED_INTERVAL and "enclosure" in r.name)
               for r in results)
    assert by_name["REFERENCE_cover_total_is_certified_enclosure"].value == "0"
    assert any("certified_enclosure() refused" in n for n in notes)
    assert any("covers_region=False" in n for n in notes)
    # the cover bookkeeping is still reported -- as bookkeeping
    assert "REFERENCE_cover_area_accounted" in by_name
    assert "REFERENCE_cover_area_rejected_bound" in by_name


def test_a5_publishes_the_enclosure_when_the_ledger_certifies_it(monkeypatch):
    """The positive half: a certified, covering Total is published as before,
    through the accessor, with the flag result reading 1."""
    import research.cover as C
    led = _hand_built_cover_ledger(certified_and_covering=True)
    assert led.total().certified_enclosure() is not None
    monkeypatch.setattr(C, "run", lambda region, integrand, config: led)

    results, notes = RUN._task_a5()
    by_name = {r.name: r for r in results}
    enc = by_name["REFERENCE_cover_total_enclosure"]
    assert enc.provenance == R.CERTIFIED_INTERVAL and enc.lo is not None
    assert by_name["REFERENCE_cover_total_is_certified_enclosure"].value == "1"
    assert "REFERENCE_cover_closed_form_crosscheck" in by_name
    assert not any("refused" in n for n in notes)


def test_writer_module_has_no_destructive_call():
    """STRUCTURAL: engine/receipt.py creates files only with mode 'x'."""
    with open(RECEIPT_PY, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    modes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "open":
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            modes.append(mode)
    assert "x" in modes, "the exclusive-creation write is gone"
    for m in modes:
        assert m in (None, "r", "rb", "x"), f"receipt.py opens a file with mode {m!r}"
    banned = {"remove", "unlink", "rmdir", "removedirs", "rename", "replace",
              "truncate", "write_text", "write_bytes", "rmtree"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in banned, f"receipt.py calls {node.attr}"
    assert "shutil" not in {n.names[0].name for n in ast.walk(tree)
                            if isinstance(n, ast.Import)}


def test_checker_detects_an_edited_receipt_by_hash_alone(tmp_path):
    """NEGATIVE CONTROL: a number was changed after the receipt was written."""
    obj = good_receipt().to_dict()
    obj["results"][1]["value"] = "1/2"          # was 836263/10000
    write_object(tmp_path, obj)                  # hash NOT recomputed
    problems = check(tmp_path)
    assert any("body_sha256 mismatch" in p for p in problems), problems
    assert any("append-only" in p for p in problems), problems


def test_checker_detects_a_widened_enclosure_by_hash_alone(tmp_path):
    """NEGATIVE CONTROL: a bound weakened after the fact is an edit."""
    obj = good_receipt().to_dict()
    enc = [r for r in obj["results"] if r["provenance"] == "certified_interval"][0]
    enc["hi"] = "99"                             # weaken the upper end
    write_object(tmp_path, obj)
    assert any("body_sha256 mismatch" in p for p in check(tmp_path))


def test_checker_detects_edits_and_deletions_against_git_head(tmp_path):
    """NEGATIVE CONTROL: rehashing after the edit does not hide it from git.

    A forger who recomputes `body_sha256` defeats the hash check. The git
    comparison is the second, independent line, exactly as
    `tests/test_registers.py` does for `work_events`.
    """
    repo = tmp_path / "repo"
    rdir = repo / "engine" / "receipts"
    (rdir / "A1").mkdir(parents=True)
    for cmd in (["git", "init", "-q"],
                ["git", "config", "user.email", "t@example.invalid"],
                ["git", "config", "user.name", "t"]):
        assert subprocess.run(cmd, cwd=repo, capture_output=True).returncode == 0

    rec = good_receipt()
    path = R.write_receipt(rec, str(rdir))
    keep = good_receipt(lane="A5", timestamp="2026-09-18T12:00:01.000000Z")
    keep_path = R.write_receipt(keep, str(rdir))
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
    assert subprocess.run(["git", "commit", "-qm", "receipts"], cwd=repo,
                          capture_output=True).returncode == 0

    assert RC.check(str(rdir), lanes_dir=str(tmp_path / "no-lanes"),
                    root=str(repo), check_git=True)[0] == []

    # (a) edit the body and recompute the hash so it is self-consistent.
    with open(path, encoding="utf-8") as handle:
        obj = json.load(handle)
    obj["results"][1]["value"] = "1/2"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rehash(obj), f, indent=2, sort_keys=True)
    problems = RC.check(str(rdir), lanes_dir=str(tmp_path / "no-lanes"),
                        root=str(repo), check_git=True)[0]
    assert any("differs from its git HEAD content" in p for p in problems), problems

    # (b) delete a committed receipt.
    os.remove(keep_path)
    problems = RC.check(str(rdir), lanes_dir=str(tmp_path / "no-lanes"),
                        root=str(repo), check_git=True)[0]
    assert any("deleted from the working tree" in p for p in problems), problems


# ---------------------------------------------------------------------------
# 4. no receipt may claim a status change
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("word", ["DISCHARGED", "CLOSED", "PROMOTED",
                                  "RECLASSIFIED", "PROVEN", "SOLVED"])
def test_checker_rejects_a_verdict_claiming_a_status_change(tmp_path, word):
    """NEGATIVE CONTROL: the headline prohibition, one word at a time."""
    obj = rehash({**good_receipt().to_dict(), "verdict": word})
    write_object(tmp_path, obj)
    problems = check(tmp_path)
    assert any("claims a status change" in p or "not permitted" in p
               for p in problems), problems


@pytest.mark.parametrize("word", ["CLOSED", "DISCHARGED", "PROMOTED"])
def test_checker_rejects_an_outcome_claiming_a_status_change(tmp_path, word):
    obj = rehash({**good_receipt().to_dict(), "outcome": word})
    write_object(tmp_path, obj)
    problems = check(tmp_path)
    assert any("outcome" in p for p in problems), problems


def test_checker_rejects_a_softened_status_effect(tmp_path):
    """NEGATIVE CONTROL: the fixed disclaimer cannot be edited away."""
    obj = rehash({**good_receipt().to_dict(),
                  "status_effect": "This receipt CLOSES the obligation."})
    write_object(tmp_path, obj)
    problems = check(tmp_path)
    assert any("status_effect" in p for p in problems), problems


def test_the_only_permitted_verdict_is_no_mathematical_verdict():
    assert R.ALLOWED_VERDICTS == (R.VERDICT_NONE,)
    assert good_receipt().verdict == "NO-MATHEMATICAL-VERDICT"


def test_the_word_scan_does_not_fire_on_honest_prose():
    """POSITIVE control for the scan's scope: caveats must survive it.

    `does_not_establish` says "does not close"; an entry point may be named
    `radial_gaussian_closed_form`. A scan that fired on those would push the
    honest wording out of the schema, which is the opposite of the point.
    """
    rec = good_receipt(
        entry_point="research.cover.driver.radial_gaussian_closed_form",
        does_not_establish=(
            "Does not close Piece 2, does not discharge OBL-H5-JETMOD, and "
            "promotes nothing. No prize problem is solved."))
    assert R.validate_body(rec.body()) == []
    assert R.scan_status_words(rec.verdict) == []


def test_extra_top_level_fields_are_rejected(tmp_path):
    """NEGATIVE CONTROL: the schema is closed, so a smuggled status is caught."""
    obj = rehash({**good_receipt().to_dict(), "status": "CLOSED"})
    write_object(tmp_path, obj)
    problems = check(tmp_path)
    assert any("unexpected top-level field 'status'" in p for p in problems), problems


# ---------------------------------------------------------------------------
# 5. provenance rules
# ---------------------------------------------------------------------------

def test_certifying_flag_cannot_be_flipped_on_a_float_result(tmp_path):
    """NEGATIVE CONTROL: a float relabelled as certifying is the whole danger."""
    obj = good_receipt().to_dict()
    probe = [r for r in obj["results"]
             if r["provenance"] == "float_noncertifying"][0]
    probe["certifying"] = True
    write_object(tmp_path, rehash(obj))
    problems = check(tmp_path)
    assert any("certifying" in p for p in problems), problems


def test_float_result_must_carry_the_non_certifying_label(tmp_path):
    obj = good_receipt().to_dict()
    probe = [r for r in obj["results"]
             if r["provenance"] == "float_noncertifying"][0]
    probe["note"] = "a high-precision value"
    write_object(tmp_path, rehash(obj))
    problems = check(tmp_path)
    assert any("NON-CERTIFYING" in p for p in problems), problems


def test_inverted_enclosure_is_rejected(tmp_path):
    """NEGATIVE CONTROL: lo > hi is a sign flip, not an enclosure."""
    obj = good_receipt().to_dict()
    enc = [r for r in obj["results"] if r["provenance"] == "certified_interval"][0]
    enc["lo"], enc["hi"] = enc["hi"], enc["lo"]
    write_object(tmp_path, rehash(obj))
    problems = check(tmp_path)
    assert any("lo > hi" in p for p in problems), problems


def test_from_interval_refuses_an_inverted_enclosure_at_construction():
    from fractions import Fraction as F

    class _Bad:
        lo, hi = F(2), F(1)

    with pytest.raises(R.ReceiptRejected):
        R.NumericResult.from_interval("bad", _Bad())


def test_unknown_provenance_is_rejected(tmp_path):
    obj = good_receipt().to_dict()
    obj["results"][0]["provenance"] = "mpmath_high_precision"
    write_object(tmp_path, rehash(obj))
    assert any("provenance" in p for p in check(tmp_path))


def test_exact_rational_value_must_parse_as_a_rational(tmp_path):
    """NEGATIVE CONTROL: an exact_rational that is not a rational is refused."""
    obj = good_receipt().to_dict()
    exact = [r for r in obj["results"] if r["provenance"] == "exact_rational"][0]
    exact["value"] = "about 83.6"
    write_object(tmp_path, rehash(obj))
    problems = check(tmp_path)
    assert any("is not an exact rational" in p for p in problems), problems


def test_outcome_ran_with_no_results_is_rejected(tmp_path):
    obj = rehash({**good_receipt().to_dict(), "results": []})
    write_object(tmp_path, obj)
    assert any("no numeric results" in p for p in check(tmp_path))


# ---------------------------------------------------------------------------
# 6. identity, lanes and argument digests
# ---------------------------------------------------------------------------

def test_argument_digest_must_match_the_arguments(tmp_path):
    """NEGATIVE CONTROL: the recorded call cannot be rewritten silently."""
    obj = good_receipt().to_dict()
    obj["run"]["arguments"]["prec"] = 8          # weaker run, same digest
    write_object(tmp_path, rehash(obj))
    problems = check(tmp_path)
    assert any("argument_digest does not match" in p for p in problems), problems


def test_receipt_whose_lane_does_not_exist_is_rejected(tmp_path):
    lanes = tmp_path / "lanes"
    lanes.mkdir()
    (lanes / "A1.json").write_text(json.dumps({"key": "A1"}), encoding="utf-8")
    rdir = tmp_path / "receipts"
    R.write_receipt(good_receipt(lane="A1"), str(rdir))
    assert RC.check(str(rdir), str(lanes), str(tmp_path), check_git=False)[0] == []
    R.write_receipt(good_receipt(lane="ZZ"), str(rdir))
    problems = RC.check(str(rdir), str(lanes), str(tmp_path), check_git=False)[0]
    assert any("has no file in" in p for p in problems), problems


def test_receipt_in_the_wrong_lane_directory_is_rejected(tmp_path):
    obj = good_receipt(lane="A1").to_dict()
    write_object(tmp_path, obj, lane="A5")
    problems = check(tmp_path)
    assert any("names lane" in p for p in problems), problems


def test_filename_must_match_the_receipt_id(tmp_path):
    obj = good_receipt().to_dict()
    write_object(tmp_path, obj, name="RCPT-A1-20260918T999999999999Z-abcdef01")
    problems = check(tmp_path)
    assert any("filename does not match" in p for p in problems), problems


def test_duplicate_receipt_ids_are_rejected(tmp_path):
    obj = good_receipt().to_dict()
    write_object(tmp_path, obj, lane="A1")
    obj2 = copy.deepcopy(obj)
    obj2["lane"] = "A1"
    d = tmp_path / "A1copy"
    d.mkdir()
    with open(d / (obj2["receipt_id"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(obj2, f, indent=2, sort_keys=True)
    problems = check(tmp_path)
    assert any("is already used by" in p for p in problems), problems


def test_checker_cli_exits_nonzero_on_a_bad_receipt(tmp_path):
    """The checker is invoked through its CLI, per CLAUDE.md."""
    obj = rehash({**good_receipt().to_dict(), "verdict": "DISCHARGED"})
    write_object(tmp_path, obj)
    r = subprocess.run(
        [sys.executable, "tools/receipts_check.py",
         "--receipts-dir", str(tmp_path), "--no-git"],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "claims a status change" in r.stdout
    assert "not permitted" in r.stdout
    assert " problems=0" not in r.stdout


# ---------------------------------------------------------------------------
# 7. the runner cannot write to engine/lanes/ or claims/graph.json
# ---------------------------------------------------------------------------

_MUTATING_ATTRS = {
    "remove", "unlink", "rmdir", "removedirs", "rename", "replace", "makedirs",
    "mkdir", "truncate", "write_text", "write_bytes", "rmtree", "copy",
    "copyfile", "move", "dump",
}


def test_run_py_contains_no_write_path_at_all():
    """STRUCTURAL: `engine/run.py` has no way to mutate the filesystem.

    This is the task's "make it structural, not a convention". If someone adds
    a writer for the lane files or the claim graph, this test fails before any
    receipt is written.
    """
    with open(RUN_PY, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "open":
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            assert mode in (None, "r", "rb"), \
                f"engine/run.py opens a file with mode {mode!r}"
        if isinstance(node, ast.Attribute):
            assert node.attr not in _MUTATING_ATTRS, \
                f"engine/run.py calls a mutating function: {node.attr}"


def test_run_py_imports_exactly_one_writer_and_it_is_the_receipt_writer():
    """STRUCTURAL: no writer for a governed path is imported."""
    with open(RUN_PY, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported[a.asname or a.name] = node.module
        elif isinstance(node, ast.Import):
            for a in node.names:
                imported[a.asname or a.name.split(".")[0]] = a.name
    writers = {n for n in imported if "write" in n.lower() or "save" in n.lower()
               or "dump" in n.lower()}
    assert writers == {"write_receipt"}, f"unexpected writers imported: {writers}"
    assert imported["write_receipt"] == "engine.receipt"
    assert "shutil" not in imported and "pathlib" not in imported
    # No module under tools/ or a lane/graph editor is imported.
    for name, mod in imported.items():
        assert not str(mod).startswith("tools."), f"run.py imports {mod}"


def test_the_receipt_writer_refuses_governed_destinations(tmp_path):
    """FUNCTIONAL: the one writer run.py has cannot reach lanes or the graph."""
    rec = good_receipt()
    for forbidden in (os.path.join(ROOT, "engine", "lanes"),
                      os.path.join(ROOT, "claims"),
                      os.path.join(ROOT, "registers"),
                      os.path.join(ROOT, "drive")):
        with pytest.raises(R.ForbiddenDestination):
            R.write_receipt(rec, forbidden)
    assert not os.path.exists(os.path.join(ROOT, "engine", "lanes", "A1",
                                           rec.receipt_id + ".json"))


def test_run_py_names_the_governed_paths_only_as_read_only_inputs():
    """`LANES_DIR` and `CLAIM_GRAPH` exist to be read, and are never written."""
    assert RUN.LANES_DIR.endswith(os.path.join("engine", "lanes"))
    assert RUN.CLAIM_GRAPH.endswith(os.path.join("claims", "graph.json"))
    with open(RUN_PY, encoding="utf-8") as handle:
        src = handle.read()
    assert "write_receipt" in src
    for bad in ("json.dump(", "os.replace(", "os.remove(", "shutil."):
        assert bad not in src, f"engine/run.py contains {bad}"


def test_dry_run_writes_nothing(tmp_path):
    before = set(os.listdir(tmp_path))
    r = subprocess.run([sys.executable, "engine/run.py", "--lane", "A1",
                        "--dry-run", "--receipts-dir", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "DRY_RUN" in r.stdout
    assert set(os.listdir(tmp_path)) == before, "a dry run wrote something"


def test_list_reports_registration_without_asserting_a_status():
    r = subprocess.run([sys.executable, "engine/run.py", "--list"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "not a" in r.stdout and "status" in r.stdout
    for key in RUN.TASKS:
        assert key in r.stdout


def test_an_unimportable_lane_module_becomes_an_unavailable_receipt(monkeypatch):
    """A sibling agent mid-write must not crash the runner, or be hidden."""
    def boom():
        raise RUN.TaskUnavailable("research.bands: not landed yet")

    task = RUN.TASKS["A1"]
    monkeypatch.setitem(RUN.TASKS, "A1",
                        RUN.LaneTask(lane=task.lane, module=task.module,
                                     entry_point=task.entry_point,
                                     arguments=task.arguments,
                                     summary=task.summary,
                                     does_not_establish=task.does_not_establish,
                                     call=boom))
    rec = RUN.run_lane("A1", RUN.load_lanes())
    assert rec.outcome == R.OUTCOME_UNAVAILABLE
    assert rec.results == ()
    assert R.validate_body(rec.body()) == []


def test_a_failing_lane_is_recorded_not_swallowed(monkeypatch):
    def boom():
        raise ZeroDivisionError("the driver blew up")

    task = RUN.TASKS["A1"]
    monkeypatch.setitem(RUN.TASKS, "A1",
                        RUN.LaneTask(lane=task.lane, module=task.module,
                                     entry_point=task.entry_point,
                                     arguments=task.arguments,
                                     summary=task.summary,
                                     does_not_establish=task.does_not_establish,
                                     call=boom))
    rec = RUN.run_lane("A1", RUN.load_lanes())
    assert rec.outcome == R.OUTCOME_FAILED
    assert "ZeroDivisionError" in (rec.error or "")
    assert R.validate_body(rec.body()) == []


def test_every_registered_lane_carries_its_own_caveat():
    """A registration without a `does_not_establish` cannot produce a receipt."""
    for key, task in RUN.TASKS.items():
        assert len(task.does_not_establish.strip()) >= R.MIN_DOES_NOT_ESTABLISH
        composed = RUN.compose_does_not_establish(task, RUN.load_lanes().get(key))
        assert "does not" in composed.lower()


# ---------------------------------------------------------------------------
# 8. canonical form
# ---------------------------------------------------------------------------

def test_canonical_json_is_stable_and_float_free():
    rec = good_receipt()
    body = rec.body()
    assert R.canonical_json(body) == R.canonical_json(json.loads(
        json.dumps(body)))
    assert R.body_sha256(body) == R.body_sha256(copy.deepcopy(body))

    def no_floats(x):
        assert not isinstance(x, float), "a float reached the canonical body"
        if isinstance(x, dict):
            for v in x.values():
                no_floats(v)
        elif isinstance(x, list):
            for v in x:
                no_floats(v)

    no_floats(body)


def test_receipt_id_carries_its_lane_and_is_validated(tmp_path):
    obj = good_receipt(lane="A1").to_dict()
    obj["lane"] = "A5"
    write_object(tmp_path, rehash(obj), lane="A5")
    problems = check(tmp_path)
    assert any("does not carry its own lane" in p for p in problems), problems
