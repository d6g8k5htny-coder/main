"""Which definitions of the frozen RN-UNIF body a run of it can actually reach.

WHY THIS EXISTS. ``docs/ENGINE_RECOVERY.md`` §3.7 records that ``env_tau`` "is
defined but never called", and §3.8 makes the same flat statement for
``run_certification`` and ``kap_fn_p1``. §3.8 is headed "a flat statement of
what is not called", so a reader is entitled to read the list as complete. It
is not: **twenty-two** of the body's seventy-nine top-level definitions are
unreachable from its own module-level execution, and ``certify_cell`` -- whose
signature §3.8 quotes -- is one of them.

This test computes the set with ``ast`` and pins it. It never imports the
frozen body: that body is ``mpmath`` at ``mp.dps = 100`` and importing it would
put a float path inside the test suite. It parses, and it never writes. A
frozen body is never edited in place.

SCOPE, AND WHY IT MATTERS HERE. "Unreachable" means *from this module's own
top-level statements*. It does not mean "never called anywhere": the carrier
``engine/carriers/blobs/7b7cc46ba5605250__rnu_t4_push.py`` does
``import d3_rn_unif as R`` and calls ``R.env_form(k, g, d, q)`` for ``q`` in
``range(5)``. That matters for a sentence in §3.7: "``env_form`` (orders 0-2 in
``qord`` as used)" describes the orders appearing in the body's own unreachable
text, while the only code in this repository that actually calls ``env_form``
calls it at orders **0 through 4**.

WHAT THIS DOES NOT ESTABLISH. Reachability of a Python function is a fact about
source text. It discharges no premise, closes no lemma, grades nothing and
moves no status. An unreachable definition is not a defect; the frozen body is
a snapshot of work in progress and says so. This is a negative control: it
fails if a future transcription silently wires one of these in, or drops one.
"""
from __future__ import annotations

import ast
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(ROOT, "engine", "rn_engine", "frozen", "K3_SIDE24_LB",
                      "UPPER2D", "D3_percolation", "d3_rn_unif.py")
T4_PUSH = os.path.join(ROOT, "engine", "carriers", "blobs",
                       "7b7cc46ba5605250__rnu_t4_push.py")

#: Computed on 2026-09-22 and pinned. A change to this set is a change to what
#: a run of the frozen body reaches, and must be a deliberate, visible edit.
UNREACHABLE = frozenset({
    "CertStats", "Hn_diag", "_Refine", "bounds_point", "certify_cell",
    "dE_crude", "d_entry", "dm_vec_col", "env_TY6", "env_form", "env_m",
    "env_small", "env_tau", "form_thrd", "kap_fn_p1", "kappa_far_point",
    "kp_msad", "point_pieces", "qblock", "run_certification", "spair_dir",
    "wick4_grad",
})

TOTAL_DEFINITIONS = 79


# ---------------------------------------------------------------------------
# The analysis. Deliberately conservative: a name mentioned ANYWHERE inside a
# definition counts as a use, whether or not it is called. That over-counts
# reachability, so a definition this reports unreachable is unreachable under
# any finer analysis too -- the safe direction for a claim of deadness.
# ---------------------------------------------------------------------------

def _definitions(tree: ast.Module) -> dict[str, ast.AST]:
    return {n.name: n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def _names(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for x in ast.walk(node):
        if isinstance(x, ast.Name):
            out.add(x.id)
        elif isinstance(x, ast.Attribute):
            out.add(x.attr)          # catches `R.env_form` in an importer
    return out


def reachable_from_module_level(source: str) -> tuple[set[str], set[str]]:
    """(reachable, unreachable) top-level definition names."""
    tree = ast.parse(source)
    defs = _definitions(tree)
    seed: set[str] = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        seed |= _names(n)
    reach = {d for d in defs if d in seed}
    frontier = list(reach)
    while frontier:
        for nm in _names(defs[frontier.pop()]):
            if nm in defs and nm not in reach:
                reach.add(nm)
                frontier.append(nm)
    return reach, set(defs) - reach


@pytest.fixture(scope="module")
def analysis():
    with open(FROZEN, encoding="utf-8") as f:
        return reachable_from_module_level(f.read())


# ---------------------------------------------------------------------------
# The pinned facts
# ---------------------------------------------------------------------------

def test_the_definition_count_is_what_was_measured(analysis):
    reach, unreach = analysis
    assert len(reach) + len(unreach) == TOTAL_DEFINITIONS


def test_the_unreachable_set_is_exactly_the_pinned_one(analysis):
    _, unreach = analysis
    assert unreach == set(UNREACHABLE), {
        "newly reachable": sorted(set(UNREACHABLE) - unreach),
        "newly unreachable": sorted(unreach - set(UNREACHABLE)),
    }


def test_the_three_the_document_names_are_in_it(analysis):
    """§3.7 and §3.8 name these three; the document is right about them."""
    _, unreach = analysis
    for name in ("env_tau", "run_certification", "kap_fn_p1"):
        assert name in unreach


def test_the_document_s_list_is_not_the_whole_set(analysis):
    """The finding: nineteen more, including one §3.8 quotes the signature of."""
    _, unreach = analysis
    documented = {"env_tau", "run_certification", "kap_fn_p1"}
    assert len(unreach - documented) == 19
    assert "certify_cell" in unreach


def test_env_form_is_unreachable_but_its_three_call_sites_exist(analysis):
    """Unreachable is not uncalled: the callers are themselves unreachable."""
    _, unreach = analysis
    for name in ("env_form", "env_small", "bounds_point", "kappa_far_point"):
        assert name in unreach


def test_the_t4_push_carrier_calls_env_form_at_orders_zero_to_four():
    """The scope sentence, checked rather than asserted.

    §3.7 says "orders 0-2 in `qord` as used". The only code in this repository
    that actually calls `env_form` calls it at 0 through 4.
    """
    with open(T4_PUSH, encoding="utf-8") as f:
        src = f.read()
    assert "import d3_rn_unif as R" in src
    assert "R.env_form(" in src
    tree = ast.parse(src)
    ranges = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id == "range"
              and len(n.args) == 1 and isinstance(n.args[0], ast.Constant)
              and n.args[0].value == 5]
    assert ranges, "no `range(5)` over the derivative order in the carrier"


# ---------------------------------------------------------------------------
# Negative controls. The analysis must be able to say "reachable" too.
# ---------------------------------------------------------------------------

def test_control_a_wired_in_definition_is_reported_reachable():
    reach, unreach = reachable_from_module_level(
        "def used():\n    return 1\n\n\ndef spare():\n    return 2\n\n\nx = used()\n")
    assert reach == {"used"} and unreach == {"spare"}


def test_control_reachability_is_transitive():
    reach, unreach = reachable_from_module_level(
        "def a():\n    return b()\n\n\ndef b():\n    return c()\n\n\n"
        "def c():\n    return 1\n\n\ndef d():\n    return 2\n\n\ny = a()\n")
    assert reach == {"a", "b", "c"} and unreach == {"d"}


def test_control_a_mention_without_a_call_still_counts_as_reachable():
    """The conservative direction, pinned so the over-counting stays deliberate."""
    reach, _ = reachable_from_module_level(
        "def f():\n    return 1\n\n\nTABLE = {'f': f}\n")
    assert reach == {"f"}


def test_control_an_attribute_reference_counts():
    reach, _ = reachable_from_module_level(
        "import mod\n\n\ndef g():\n    return 1\n\n\nmod.g\n")
    assert reach == {"g"}


def test_control_the_analysis_is_not_vacuous(analysis):
    reach, unreach = analysis
    assert len(reach) > 50 and len(unreach) > 10
    assert "he_abs" in reach and "kplane" in reach


def test_control_a_module_with_no_top_level_code_reaches_nothing():
    reach, unreach = reachable_from_module_level("def a():\n    return 1\n")
    assert reach == set() and unreach == {"a"}


def test_this_file_imports_only_the_standard_library():
    """The frozen body is mpmath at dps 100; parsing it must not execute it.

    Checked on this file's own import statements rather than by searching its
    text, because the text necessarily contains the string it would look for.
    """
    with open(os.path.abspath(__file__), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    assert imported <= {"ast", "os", "pytest", "__future__"}, imported
