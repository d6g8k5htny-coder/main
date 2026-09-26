"""Negative controls for tools/register_migration_preflight.py.

The preflight's whole value is that it refuses to vouch for a checker whose
redirect flag is accepted and then ignored. That failure mode is not
hypothetical: ``mirror_quotes_check`` has it, and walking into it once produced
twenty-four confident findings about an export that were entirely an artifact of
the probe. So the controls here are built around breaking the tool rather than
running it — a preflight that cannot detect a silently-dropped redirect, or that
truncates an excerpt down to the allowlisted noise, is worse than no preflight,
because both failures look like clean information.

What these tests do not establish: nothing here reads a register cell for its
meaning, adopts an export, or says anything about whether the candidate workbook
is a faithful export of the Drive. They test one tool's refusals.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "register_migration_preflight.py")
R1 = os.path.join(ROOT, "registers", "source",
                  "GP-REG-032_v1.2_export_2026-09-23_R1.xlsx")


def load():
    """Import the tool fresh, so no module global is bound at collection time."""
    spec = importlib.util.spec_from_file_location("_preflight_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(*args):
    proc = subprocess.run([sys.executable, TOOL, *args], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


requires_r1 = pytest.mark.skipif(not os.path.isfile(R1),
                                 reason="the 2026-09-23 R1 export is not in this tree")


# ---------------------------------------------------------------- the refusals

@requires_r1
def test_the_r1_export_is_refused_and_every_offending_status_is_named():
    code, out = run_cli("--source", R1)
    assert code == 1, out
    for key in ("RV-WITHDRAWAL-20260921-01", "RV-ROLLOUT-R2-20260921-01",
                "RV-H3-SOLVER-20260921-01"):
        assert key in out, f"{key} is a reason the export is refused and is not reported"
    assert "refusing=4" in out, out


@requires_r1
def test_the_summary_counts_agree_with_the_per_checker_verdicts():
    code, out = run_cli("--source", R1)
    assert code == 1
    refusing = sum(1 for line in out.splitlines() if line.strip().startswith("refuses"))
    assert f"refusing={refusing}" in out, out


def test_a_missing_export_is_reported_rather_than_traced():
    code, out = run_cli("--source", os.path.join(ROOT, "registers", "source", "nope.xlsx"))
    assert code == 2
    assert "no such export" in out
    assert "Traceback" not in out


def test_an_unreadable_export_is_refused_by_the_importer(tmp_path):
    fake = tmp_path / "not-a-workbook.xlsx"
    fake.write_bytes(b"this is not a zip container")
    code, out = run_cli("--source", str(fake))
    assert code != 0
    assert "the importer refused" in out


# ----------------------------------------------- the silently-dropped redirect

@requires_r1
def test_a_checker_that_ignores_its_redirect_is_reported_not_redirectable(capsys):
    """The control the tool exists for, run against the real instance of the bug.

    ``mirror_quotes_check`` resolves ``--register-json`` under ``--root`` and
    drops anything outside it without a diagnostic. Pointed at the candidate and
    at the sentinel-corrupted copy it therefore prints the same thing both
    times, and the preflight must refuse to read that as a pass.

    Asserting only on the exit status would be decoration: with the corpus
    dropped the checker reports twenty-four unsourced quotations and exits
    nonzero on its own, so a preflight that had lost the redirect test entirely
    would still come back 1 — for the wrong reason, and while presenting those
    twenty-four artifacts as findings about the export. The distinguishing
    claim is *why* it came back 1, so that is what is pinned here.
    """
    module = load()
    module.COVERED = (
        ("mirror_quotes_check", lambda d: ["--register-json", d]),
    )
    code = module.main(["--source", R1])
    out = capsys.readouterr().out
    assert code == 1
    assert "not_redirectable=1" in out, out
    assert "refusing=0" in out, "the run is untrustworthy, not a refusal of the export"
    assert "unsourced" not in out and "quoted text is in no stored byte" not in out, \
        "the dropped-corpus artifacts must not be quoted as findings about the export"


@requires_r1
def test_not_redirectable_is_counted_a_problem_and_not_an_acceptance(capsys):
    module = load()
    module.COVERED = (("mirror_quotes_check", lambda d: ["--register-json", d]),)
    module.main(["--source", R1])
    out = capsys.readouterr().out
    assert "NOT_REDIRECTABLE" in out
    assert "not_redirectable=1" in out
    assert "accepts" not in out


@requires_r1
def test_every_covered_checker_is_currently_sensitive_to_the_sentinel():
    """If a covered checker stops responding to its input, say so here.

    A checker added to COVERED whose output the corruption does not move would
    be reported NOT_REDIRECTABLE at run time, turning the whole preflight red.
    This is the regression control for that, naming the checker rather than
    leaving a maintainer to read a bare exit code.
    """
    module = load()
    code, out = run_cli("--source", R1)
    assert "NOT_REDIRECTABLE" not in out, out
    assert f"covered={len(module.COVERED)}" in out
    assert "not_redirectable=0" in out


# ------------------------------------------------------------- the excerpt

def test_allowlisted_known_lines_never_crowd_out_the_new_ones():
    module = load()
    out = "\n".join(["KNOWN  duplicate key %d" % i for i in range(37)]
                    + ["NEW    review_queue: row 34 status not in R17 set",
                       "tabs=44 problems=40 known=37 new=3"])
    lines = module.excerpt(out, 6, full=False)
    assert any("NEW " in ln for ln in lines), lines
    assert lines[-1].startswith("tabs=44"), lines
    assert sum(1 for ln in lines if ln.startswith("KNOWN")) == 0


def test_the_summary_line_survives_truncation_even_when_it_is_the_only_salient_one():
    module = load()
    out = "\n".join(["KNOWN  a", "KNOWN  b", "KNOWN  c", "checker: 3 problem(s)"])
    lines = module.excerpt(out, 1, full=False)
    assert lines[-1] == "checker: 3 problem(s)"


def test_full_quotes_everything_including_the_allowlisted_lines():
    module = load()
    out = "\n".join(["KNOWN  a", "KNOWN  b", "NEW    c", "summary"])
    assert module.excerpt(out, 1, full=True) == ["KNOWN  a", "KNOWN  b", "NEW    c", "summary"]


def test_a_short_output_is_never_reordered():
    module = load()
    out = "\n".join(["KNOWN  a", "NEW    b", "summary"])
    assert module.excerpt(out, 6, full=False) == ["KNOWN  a", "NEW    b", "summary"]


def test_the_truncation_note_says_how_much_was_hidden():
    module = load()
    out = "\n".join(["NEW %d" % i for i in range(20)] + ["summary"])
    lines = module.excerpt(out, 3, full=False)
    note = [ln for ln in lines if ln.startswith("... ")]
    assert note and "17 further line(s)" in note[0], lines


# ------------------------------------------------------- it writes nothing

@requires_r1
def test_the_preflight_leaves_the_exported_registers_byte_identical():
    """Rule 7: an export is never edited, and a preflight is not an import."""
    watched = [os.path.join(ROOT, "registers", d) for d in ("json", "csv", "source")]

    def snapshot():
        out = {}
        for base in watched:
            for dirpath, _dirs, names in os.walk(base):
                for name in sorted(names):
                    path = os.path.join(dirpath, name)
                    with open(path, "rb") as handle:
                        out[path] = hashlib.sha256(handle.read()).hexdigest()
        return out

    before = snapshot()
    run_cli("--source", R1)
    assert snapshot() == before


# --------------------------------------------------------- the honesty pins

def test_the_uncovered_list_names_tools_that_exist():
    """An uncovered checker that has been renamed must not sit in the list."""
    module = load()
    assert module.UNCOVERED
    for name, why in module.UNCOVERED:
        assert os.path.isfile(os.path.join(ROOT, "tools", name + ".py")), name
        assert len(why) > 40, f"{name} is excluded without saying why"


def test_the_uncovered_and_covered_lists_do_not_overlap():
    module = load()
    covered = {name for name, _flags in module.COVERED}
    uncovered = {name for name, _why in module.UNCOVERED}
    assert not (covered & uncovered)


def test_every_covered_checker_exists_in_the_tree():
    module = load()
    for name, _flags in module.COVERED:
        assert os.path.isfile(os.path.join(ROOT, "tools", name + ".py")), name


def test_the_module_states_what_it_does_not_establish():
    module = load()
    doc = module.__doc__ or ""
    assert "DOES NOT ESTABLISH" in doc
    for phrase in ("adopts", "independence credit", "moves no gate"):
        assert phrase in doc, phrase


@requires_r1
def test_the_run_prints_what_it_does_not_establish():
    _code, out = run_cli("--source", R1)
    assert "adopts nothing" in out
    assert "independence credit" in out


def test_the_preflight_is_not_wired_into_ci():
    """It refuses the export that is in the tree; gating CI on it would be a lie.

    The export is retained deliberately and is not the import source. A CI step
    running this tool would go red for a state the repository has recorded as
    correct, so the absence of that step is load-bearing rather than an
    oversight.
    """
    workflow = os.path.join(ROOT, ".github", "workflows", "ci.yml")
    with open(workflow, encoding="utf-8") as handle:
        assert "register_migration_preflight" not in handle.read()
