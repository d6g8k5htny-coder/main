#!/usr/bin/env python3
"""Round-7 update of main docs/LEAN_VERIFICATION_SPECIFICATION.md from the Math- REGISTER.json after the round-4
proof-packet theorems (modules Side24Analysis, Side24Interfaces, RNCluster: 13 theorems of registered Side24/RN Props,
branch claude/lean-statement-proofs-round4-20261011, PR number pending). Derived from render_spec6.py: everything
render_spec6 does is kept (every hosted-run binding of rounds 1-3 verbatim), and in addition the document

  (a) counts and tables the 13 round-4 theorems as `proved` entries WITHOUT a hosted receipt yet, marked † in the §4
      table and said so in words (branch name, PR number pending, "the first hosted lean-specification-proofs.yml run
      on that branch is their first receipt"); the 60 theorems of PR #423 head 12e072d6 are bound to their
      corroborating run 38150363063 (actual tested merge 86a6e8c8…, base 79731b96…);
  (b) names the three new modules wherever the packet's modules are listed;
  (c) records the round-4 triage in one paragraph (counts by verdict, computed from triage_table.json and asserted);
  (d) keeps every existing run binding; every mention of a hosted run binds PR head AND actual tested merge (C422-E1).

Runs against the doc as committed at 91cffbb5 (restore it first). Scientific effect: NONE. Regenerates the §4 claim
table and the §5.1 counts table from the register and applies exact-anchor prose edits; every number quoted is computed
from the register or the triage table and re-asserted. No claim-level label changes; `kernel-checked` is written for no
round-4 theorem. Same-account author: organizational-independence credit 0; alignment PENDING_INDEPENDENT_REVIEW.

usage: render_spec7.py <round-3 PR number> RUN6 M6 SHA6 --round4-pr PENDING|<number>
       e.g. render_spec7.py 423 38149888699 4363b6928c62da242952631c6b0357d096c3b252 cdeb5c73…ca8 --round4-pr PENDING
env:   RS7_DOC (default <scratch>/lvs_main/docs/LEAN_VERIFICATION_SPECIFICATION.md), RS7_REGISTER (default the Math- clone's)
"""
import argparse
import copy
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent.parent
sys.path.insert(0, str(SCRATCH / "lvs/spec_doc"))
import render_spec as R  # noqa: E402  (helpers only; its main() is not run)

DOC = Path(os.environ.get("RS7_DOC", str(SCRATCH / "lvs_main/docs/LEAN_VERIFICATION_SPECIFICATION.md")))
MANIFEST = SCRATCH / "lvs_main/formal/manifest.json"
REGISTER = Path(os.environ.get("RS7_REGISTER", str(SCRATCH / "repos/math/specifications/lean_statements_20261010/REGISTER.json")))
TRIAGE_TABLE = SCRATCH / "lvs/round4/triage_table.json"
PACKET = R.PACKET
PROOF_PACKAGE = "d6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011"
MAIN_SIDE24 = "d6g8k5htny-coder/main formal/ (UniversalLaw, side24-arithmetic-skeleton)"
MAIN_P15 = "d6g8k5htny-coder/main formal/ at 91cffbb51d1636f382e9d61a1ecb6f7cd38e1d06 (PR #358, not yet on main)"
TIERS = json.loads((SCRATCH / "lvs/register_update5/receipted_52.json").read_text())
RECEIPTED38 = set(TIERS["tier1_run_38116277597_head_00d2ac6e"])
RECEIPTED52 = set(TIERS["all_52"])
assert len(RECEIPTED38) == 38 and len(RECEIPTED52) == 52 and RECEIPTED38 <= RECEIPTED52

# ---- rounds 1-3: verbatim from render_spec6 (FACTS; never re-derived here)
RUN, C8 = "38116277597", "00d2ac6e"
RUN2, C2 = "38121713840", "910809d6"
RUN3, C3 = "38122386459", "d9b8dd33"
RUN4, C4 = "38124262242", "79731b96"
BASE = "b2d70d77876a10c365f5db5d3b6cafcf180af4e2"
M1F, M2F, M3F, M4F = ("c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c", "5b198699308e0585904c21d12bf540cf2475d2f0",
                      "866136cca66b46e6dbd3186b47c9340f7e93b4f6", "318ff194eecc42c39a6b27c1d7bf03f30d6a34d9")
M1, M2, M3, M4 = M1F[:8], M2F[:8], M3F[:8], M4F[:8]
RUNA, C5A, M5AF, SHA5A = "38149524759", "88c21058", "3306233bbdeadb15d9b4b07b94cc52a80ed8caeb", "ddefbdcc63223d05fd8ed34addbeebc199e271525d87dead01109cddc529e836"
M5A = M5AF[:8]
BASE2 = "79731b961fac555f33be3cd90e379b246a5a80e3"   # head of PR #422, the base of every round-3 hosted run
# ---- round 3 corroboration on PR #423 head 12e072d6 (FACTS), the last head with a hosted receipt
RUN7, C7, M7F, SHA7 = "38150363063", "12e072d6", "86a6e8c876f7904cadda2d006a56f8792dc5c1af", "bf0e0515e2035103b07d894e2a73eeb5b542885f6da16b98c228877a5e0c1690"
M7 = M7F[:8]
# ---- round 4 (FACTS): three modules committed on the round-4 branch; NO hosted receipt yet
BRANCH4 = "claude/lean-statement-proofs-round4-20261011"
HEAD4 = "80e80243"            # 80e80243da9de0bd81703bbc2dd19782196fbefe, one commit on 12e072d6, adds the three modules
NEW_MODULES = ("Side24Analysis", "Side24Interfaces", "RNCluster")
NEW_MODULE_COUNTS = {"Side24Analysis": 9, "Side24Interfaces": 3, "RNCluster": 1}
TRIAGE_WF = "wf_a3531fd8-891"
OTHER_RUNS = {"38068951535", "38096827443", "36356485214"}   # main `workspace-landing` / `formal-verification` push runs quoted in §5.1 (no PR merge semantics)
INTERP = ["CoordinatewiseConcavity", "HazardProductBound", "HazardPhiBound", "FullGroundPriceBound", "FullGroundBudgetIff",
          "LocalPriceBoundFull", "FullTransformedPriceBudget"]
PREDICATES = {"IsRealizedFamily", "IsGood", "IsKDecomposable", "IsGeneratorCover", "IsPaletteFeasible", "IsMinimalForbidden",
              "IsTransversal", "IsProperDownset", "IsUnitTuple", "NegDefCone"}
VERDICTS = ("PROVABLE_NOW", "PROVABLE_WITH_WORK", "BLOCKED_MISSING_MATHLIB", "BLOCKED_INTERFACE", "PREDICATE_VOCABULARY")

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("newpr", help="Math- PR number of the round-3 branch (423)")
ap.add_argument("run6", help="hosted run that receipted all 60 theorems on PR head 0dcb0016")
ap.add_argument("m6", help="actual tested merge commit of that run (40 hex)")
ap.add_argument("sha6", help="sha256 of its artifact lean-specification-proofs-<run6>-1 (64 hex)")
ap.add_argument("--round4-pr", required=True, help="PR number of the round-4 branch, or PENDING while none is open")
ARGS = ap.parse_args()
NEWPR, RUN6, M6F, SHA6 = ARGS.newpr, ARGS.run6, ARGS.m6, ARGS.sha6
assert NEWPR.isdigit() and RUN6.isdigit() and re.fullmatch(r"[0-9a-f]{40}", M6F) and re.fullmatch(r"[0-9a-f]{64}", SHA6)
assert RUN6 == "38149888699" and M6F == "4363b6928c62da242952631c6b0357d096c3b252", "round-3 binding must match the FACTS"   # the register is written with these
M6, C6 = M6F[:8], "0dcb0016"
assert ARGS.round4_pr == "PENDING" or ARGS.round4_pr.isdigit(), ARGS.round4_pr
R4PR_TXT = f"PR #{ARGS.round4_pr}" if ARGS.round4_pr.isdigit() else "PR number pending"
# (run, head, full merge, short merge, base): every hosted run this document may mention
BINDINGS = ((RUN, C8, M1F, M1, BASE), (RUN2, C2, M2F, M2, BASE), (RUN3, C3, M3F, M3, BASE), (RUN4, C4, M4F, M4, BASE),
            (RUN6, C6, M6F, M6, BASE2), (RUNA, C5A, M5AF, M5A, BASE2), (RUN7, C7, M7F, M7, BASE2))
RUNS = tuple((b[0], b[3]) for b in BINDINGS)
for run, _, mf, _, base in BINDINGS:
    assert run.isdigit() and len(run) == 11 and re.fullmatch(r"[0-9a-f]{40}", mf) and re.fullmatch(r"[0-9a-f]{40}", base)
assert len({b[0] for b in BINDINGS}) == len(BINDINGS) and not ({b[0] for b in BINDINGS} & OTHER_RUNS)


def short_decl(st, receipted):
    decl = st["declaration"]
    if st["package"] == PACKET:
        return f"`{decl.replace('UniversalLaw.Spec.', '')}`"
    if st["package"] == PROOF_PACKAGE:
        return f"proof:`{decl.replace('UniversalLaw.Proofs.', '')}` ({st['formal_progress']})" + ("" if decl in receipted else "†")
    return f"main:`{decl.replace('UniversalLaw.', '')}` ({st['formal_progress']})"


def make_render_statements(receipted):
    def render_statements(stmts):
        if not stmts:
            return "—"
        names = [short_decl(s, receipted) for s in stmts]
        if len(names) <= R.STMT_SHOW:
            return ", ".join(names)
        n_spec = sum(1 for s in stmts if s["package"] == PACKET)
        n_proof = sum(1 for s in stmts if s["package"] == PROOF_PACKAGE)
        n_pend = sum(1 for s in stmts if s["package"] == PROOF_PACKAGE and s["declaration"] not in receipted)
        n_main = len(stmts) - n_spec - n_proof
        head, tail, mid = ", ".join(names[:3]), ", ".join(names[-3:]), f"… (+{len(names) - 6} more) …"
        summary = (f"{n_spec} packet statements" + (f" + {n_main} main theorems (`proved`)" if n_main else "")
                   + (f" + {n_proof} proof-packet theorems (`proved`" + (f", {n_pend} †" if n_pend else "") + ")" if n_proof else ""))
        return f"{summary}: {head}, {mid}, {tail}"
    return render_statements


def replace_once(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:80])
    return text.replace(old, new)


def splice(text, start_marker, end_marker, body):
    i = text.index(start_marker)
    j = text.index(end_marker, i)
    return text[:i] + body + text[j:]


def occurrences(text, sub):
    i = text.find(sub)
    while i >= 0:
        yield i
        i = text.find(sub, i + 1)


def fuse_runs(text):
    """Fuse every mention of a hosted run in a register field into one whitespace-free token `run <id>/tested-merge-<m>…`
    followed by `(PR head <h>, base <b>…` so that an abridged cell keeps run, head and merge together or drops them all."""
    for run, head, mf, m, base in BINDINGS:
        text = text.replace(f"run {run} (PR head {head}; actual tested merge {mf} with base {base})",
                            f"run {run}/tested-merge-{m}… (PR head {head}, base {base[:8]}…)")
        text = text.replace(f"run {run} (actual tested merge {mf} with base {base}",          # TIER1 phrase, corroboration phrases, receipt clauses
                            f"run {run}/tested-merge-{m}… (PR head {head}, base {base[:8]}…")
        text = text.replace(f"run {run}, actual tested merge {mf} with base {base}",          # "the last head with a hosted receipt: run …, actual tested merge …"
                            f"run {run}/tested-merge-{m}… (PR head {head}, base {base[:8]}…)")
        text = text.replace(f"artifact lean-specification-proofs-{run}-1", f"artifact lean-specification-proofs-{run}/tested-merge-{m}…-1")
    text = text.replace(f"hosted run {RUN} receipted)", f"hosted run {RUN}/tested-merge-{M1}… (PR head {C8}) receipted)")   # back-reference in p15-price-boundary
    return text


def main():
    reg = json.loads(REGISTER.read_text())
    manifest = json.loads(MANIFEST.read_text())
    triage = json.loads(TRIAGE_TABLE.read_text())
    rows = reg["claims"]
    stmts = [s for r in rows for s in r["statements"]]
    c = R.counts(reg, manifest)
    pk = Counter(s["package"] for s in stmts)
    proof = [s for s in stmts if s["package"] == PROOF_PACKAGE]
    proof_names = {s["declaration"] for s in proof}
    main_names = {s["declaration"] for s in stmts if s["package"] in (MAIN_SIDE24, MAIN_P15)}
    module_of = lambda n: n.split(".")[2]
    T3 = {n for n in proof_names if module_of(n) == "P15Interpolation"}                     # receipted by RUN6 and RUNA, corroborated by RUN7
    RECEIPTED60 = RECEIPTED52 | T3
    pending = sorted(n for n in proof_names if module_of(n) in NEW_MODULES)                  # round 4: no hosted receipt yet
    NT, NPE, NREC, NPEND4 = len(proof_names), pk[PROOF_PACKAGE], len(RECEIPTED60), len(pending)
    # invariants the prose relies on (what must not have moved) and consistency of the computed numbers
    assert c["claims"] == 171, c["claims"]
    assert c["fp"] == Counter({"none": 138, "specified": 33}), c["fp"]          # no claim-level label moved
    assert c["stmt_fp"]["specified"] == 197 and c["stmt_fp"]["proved"] == 72 + NPE and c["statements"] == 197 + 72 + NPE
    assert "kernel-checked" not in c["stmt_fp"] and "proved" not in c["fp"]
    assert c["spec_distinct"] == 150 and c["spec_kind"] == Counter({"def_prop": 141, "structure": 9})
    assert c["spec_module"] == Counter({"P15": 58, "Side24": 28, "RN": 28, "Lifetime": 24, "Field": 12}), c["spec_module"]
    assert pk[PACKET] == 197 and pk[MAIN_SIDE24] == 31 and pk[MAIN_P15] == 41, pk
    assert len(main_names) == 59 == c["main_targets"]
    assert RECEIPTED52 <= proof_names and len(T3) == 8 and not (RECEIPTED52 & T3) and NREC == 60, (len(T3), NREC)
    assert set(pending) == proof_names - RECEIPTED60 and NPEND4 == 13 and NT == 73 and NPE == 83, (NPEND4, NT, NPE)
    assert Counter(module_of(n) for n in pending) == NEW_MODULE_COUNTS, Counter(module_of(n) for n in pending)
    covered = set()
    for r in rows:
        anchors = {s["anchor"] for s in r["statements"] if s["package"] == PROOF_PACKAGE}
        covered |= {s["declaration"] for s in r["statements"] if s["package"] == PACKET and s["kind"] == "def_prop" and s["anchor"] in anchors}
    NP = len(covered)
    assert NT == NP + 1 and NP == 72, (NT, NP)   # TriangleNeedsThree proved twice, nothing else
    covered_short = {n.split(".", 3)[-1] for n in covered}
    assert c["idonly"] == 82 and c["sources"] == 23
    mods = Counter(module_of(n) for n in proof_names)
    MOD_ORDER = ("P15Elementary", "P15Structure", "P15Hazard", "P15Covers", "P15Budget", "P15Interpolation", "Side24Elementary",
                 "Side24Analysis", "Side24Interfaces", "RNProbability", "RNCluster")
    assert set(mods) == set(MOD_ORDER) and mods["P15Interpolation"] == 8, mods
    mods_txt = ", ".join(f"`{m}` {mods[m]}" for m in MOD_ORDER)
    new_mods_txt = ", ".join(f"`{m}` {mods[m]}" for m in NEW_MODULES)
    proved_interp = [p for p in INTERP if p in covered_short]
    f3 = "RhoStarEnclosure" in covered_short
    f2 = "FullTransformedPriceBudget" in covered_short
    assert f2 and f3 and proved_interp == INTERP
    new_props = sorted(covered_short & (set(INTERP) | {"RhoStarEnclosure"}))
    assert len(new_props) == 8 == len(T3), new_props
    # the 13 round-4 theorems and the Props they prove (the Prop is the registered local statement with the theorem's anchor)
    prop_of = {}
    for r in rows:
        for s in r["statements"]:
            if s["package"] == PROOF_PACKAGE and s["declaration"] in pending:
                locs = [l["declaration"] for l in r["statements"] if l["package"] == PACKET and l["kind"] == "def_prop" and l["anchor"] == s["anchor"]]
                assert len(locs) == 1, (s["declaration"], locs)
                assert prop_of.setdefault(s["declaration"], locs[0]) == locs[0]
    pending_ordered = []                                    # first occurrence in the register's statement entries = the packet's inventory order
    for s in proof:
        if s["declaration"] in prop_of and s["declaration"] not in pending_ordered:
            pending_ordered.append(s["declaration"])
    assert sorted(pending_ordered) == pending
    new13_props = {m: [] for m in NEW_MODULES}
    for t in pending_ordered:
        new13_props[module_of(t)].append(prop_of[t].split(".", 3)[-1])
    NEW13_SHORT = {p for ps in new13_props.values() for p in ps}
    assert len(NEW13_SHORT) == 13 and NEW13_SHORT <= covered_short
    assert new13_props["Side24Interfaces"] == ["ConeMomentD1Identity", "TraceMoments", "RayleighSquareLaw"], new13_props
    assert new13_props["RNCluster"] == ["RareClusterPoissonObstructionO"], new13_props
    assert set(new13_props["Side24Analysis"]) == {"ThetaSummable", "DerivativePairingBound", "DerivativeBoundAtLeastOne", "DerivativeBoundAtZero",
                                                  "ReferenceOddAndPinBlocks", "ReferencePinDensityProduct", "IntegrandRatioBounds",
                                                  "LatticeImageSumBound", "DensityComparison"}, new13_props
    new13_txt = "; ".join(f"`{m}` ({len(new13_props[m])}: " + ", ".join(f"`{p}`" for p in new13_props[m]) + ")" for m in NEW_MODULES)
    # Side24 Props still without a theorem (computed, then checked against what the prose names)
    side24_props = {s["declaration"].split(".", 3)[-1] for s in stmts if s["package"] == PACKET and s["kind"] == "def_prop" and s["module"].endswith("Side24.lean")}
    side24_left = sorted(side24_props - covered_short)
    side24_left_targets = [p for p in side24_left if p not in PREDICATES]
    assert set(side24_left_targets) == {"ImageBound", "CovarianceOrdering", "EntrywiseCovarianceBound", "ReferenceCovarianceLowerBound",
                                        "ConeMomentD2Identity", "CoefficientEnclosure", "ReferenceComparison"}, side24_left
    assert set(side24_left) - set(side24_left_targets) == {"IsUnitTuple", "NegDefCone"}, side24_left
    rn_left = sorted({s["declaration"].split(".", 3)[-1] for s in stmts if s["package"] == PACKET and s["kind"] == "def_prop" and s["module"].endswith("RN.lean")} - covered_short)
    assert "RareClusterPoissonObstructionO" not in rn_left and len(rn_left) == 27 - 6, rn_left   # 27 RN def-Props (the 28th registered RN declaration is a structure)

    # ---------------------------------------------------------------- round-4 triage: counts computed from the table, asserted
    by = defaultdict(list)
    for row in triage:
        assert row["verdict"] in VERDICTS, row
        by[row["verdict"]].append(row)
    tshort = lambda row: row["declaration"].split(".", 1)[1]
    counts = {v: len(by[v]) for v in VERDICTS}
    assert counts == {"PROVABLE_NOW": 10, "PROVABLE_WITH_WORK": 5, "BLOCKED_MISSING_MATHLIB": 7, "BLOCKED_INTERFACE": 24, "PREDICATE_VOCABULARY": 28}, counts
    assert len(triage) == 74 and len({row["declaration"] for row in triage}) == 74
    triaged = {"UniversalLaw.Spec." + row["declaration"] for row in triage}
    all_props = {s["declaration"] for s in stmts if s["package"] == PACKET and s["kind"] == "def_prop"}
    covered_before = covered - {d for d in covered if d.split(".", 3)[-1] in NEW13_SHORT}
    uncovered_before = all_props - covered_before
    assert len(uncovered_before) == 82 and triaged <= uncovered_before, len(uncovered_before)
    not_triaged = sorted(d.split(".", 3)[-1] for d in uncovered_before - triaged)
    assert len(not_triaged) == 8 and set(not_triaged) <= PREDICATES and all(d.split(".")[2] == "P15" for d in uncovered_before - triaged), not_triaged
    now = [tshort(r) for r in by["PROVABLE_NOW"]]
    assert all(n in NEW13_SHORT for n in now), now
    work_done = [tshort(r) for r in by["PROVABLE_WITH_WORK"] if tshort(r) in NEW13_SHORT]
    work_left = [(tshort(r), r["effort"]) for r in by["PROVABLE_WITH_WORK"] if tshort(r) not in NEW13_SHORT]
    assert set(now) | set(work_done) == NEW13_SHORT and len(work_done) == 3 and len(work_left) == 2 and all(e == "weeks" for _, e in work_left), (work_done, work_left)
    assert sorted(w for w, _ in work_left) == ["ConeMomentD2Identity", "ReferenceCovarianceLowerBound"], work_left
    mm_side24 = sorted(tshort(r) for r in by["BLOCKED_MISSING_MATHLIB"] if r["module"] == "Side24.lean")
    mm_life = sorted(tshort(r) for r in by["BLOCKED_MISSING_MATHLIB"] if r["module"] == "Lifetime.lean")
    assert mm_side24 == ["CovarianceOrdering", "EntrywiseCovarianceBound", "ImageBound"] and len(mm_life) == 4, (mm_side24, mm_life)
    assert mm_life == ["ParentCompactWindowDensityB", "ParentUnrestrictedLeadingDensityC", "PlanarDensityGapE11", "RemainderBoundedR1"], mm_life
    assert all(r["effort"] == "weeks" for r in by["BLOCKED_MISSING_MATHLIB"])
    assert all("iteratedFDeriv_tsum" in " ".join(r["mathlib"]) for r in by["BLOCKED_MISSING_MATHLIB"] if r["module"] == "Side24.lean")
    bi = Counter(r["module"] for r in by["BLOCKED_INTERFACE"])
    assert bi == Counter({"RN.lean": 19, "Lifetime.lean": 3, "Side24.lean": 2}), bi
    n_dirac = sum(1 for r in by["BLOCKED_INTERFACE"] if "Dirac" in r["finding"])
    assert n_dirac >= 16, n_dirac
    assert all("IsEmpty PeriodicCoefficientInterface" in r["finding"] for r in by["BLOCKED_INTERFACE"] if r["module"] == "Side24.lean")
    assert all(("p ≡ 0" in r["finding"] or "p ≡ 1" in r["finding"]) for r in by["BLOCKED_INTERFACE"] if r["module"] == "Lifetime.lean")
    pv = Counter(r["module"] for r in by["PREDICATE_VOCABULARY"])
    assert pv == Counter({"Lifetime.lean": 14, "Field.lean": 10, "Side24.lean": 2, "RN.lean": 2}), pv
    assert set(side24_left_targets) == set(w for w, _ in work_left) | set(mm_side24) | {tshort(r) for r in by["BLOCKED_INTERFACE"] if r["module"] == "Side24.lean"}
    rn_left_targets = sorted(tshort(r) for r in by["BLOCKED_INTERFACE"] if r["module"] == "RN.lean")
    rn_left_preds = sorted(tshort(r) for r in by["PREDICATE_VOCABULARY"] if r["module"] == "RN.lean")
    assert set(rn_left) == set(rn_left_targets) | set(rn_left_preds) and len(rn_left_targets) == 19 and len(rn_left_preds) == 2, (rn_left, rn_left_targets, rn_left_preds)

    doc = DOC.read_text()

    # ---------------------------------------------------------------- header date and §1 bullet 4
    doc = replace_once(doc, "`session_01Cz7WZybv8znP64SpPj6sWY`, 2026-10-10), same GitHub",
                       "`session_01Cz7WZybv8znP64SpPj6sWY`, 2026-10-10, updated 2026-10-11), same GitHub")
    doc = replace_once(doc, "   existing validators (`tools/formal_gate_check.py --alignment`, Math- `formal/gate.py --alignment`).\n",
                       "   existing validators (`tools/formal_gate_check.py --alignment`, Math- `formal/gate.py --alignment`).\n"
                       f"4. **A proved Prop is not an accepted claim.** The proof packet (§3.7) proves {NP} of the 141 registered\n"
                       f"   Props exactly as stated; {NREC} of its {NT} theorems carry a hosted kernel receipt (the {len(T3)} of module\n"
                       f"   `P15Interpolation`, PR #{NEWPR}, by run {RUN6}: PR head `{C6}`, actual tested merge `{M6}…`; all {NREC} corroborated on\n"
                       f"   PR #{NEWPR} head `{C7}` by run {RUN7}, actual tested merge `{M7}…`) and the {NPEND4} of the round-4 modules\n"
                       f"   `Side24Analysis`, `Side24Interfaces` and `RNCluster` (branch `{BRANCH4}`, {R4PR_TXT}) have\n"
                       "   none yet: the first hosted run on that branch is their first receipt. Each such theorem is L5\n"
                       "   evidence for that Prop, for the formal objects it is stated about, and the claim beneath it stays\n"
                       "   `specified` or `none` (the register admits no other claim-level label) until an independent alignment\n"
                       "   review and the controlling acceptance machinery say otherwise.\n")

    # ---------------------------------------------------------------- §2.3 timing sentence (round 6, unchanged)
    doc = replace_once(doc, "9 s `P15`, each exiting 0 with zero warnings; the injected-type-error control exits 1.\n",
                       "9 s `P15`, each exiting 0 with zero warnings; the injected-type-error control exits 1. The proof packet\n"
                       f"(§3.7) rebuilt its five P15 modules in 76 s after the change at `{C2}` (`lake build`, 8,936 jobs, the rest\n"
                       "up to date); its `leanchecker` replay of the Mathlib-importing environment needs about 14 GB and was\n"
                       "memcg-killed in this 15 GB sandbox, so that receipt comes from the hosted run only.\n")

    # ---------------------------------------------------------------- §3 rows 3.1 and 3.7, gate paragraph
    doc = replace_once(doc, "28 P15 staged in this programme's uncommitted worktree, on no commit",
                       "28 P15 at commit `91cffbb5` of main PR #358, not yet on main")
    i = doc.index("| 3.6 | **Lean statement register**")
    j = doc.index("\n", i)
    row37 = ("\n| 3.7 | **Lean statement proofs** (this programme) | Math- `specifications/lean_statement_proofs_20261011/` (PR #422, "
             f"stacked on #419; round 3 in PR #{NEWPR}, stacked on #422; round 4 on branch `{BRANCH4}`, stacked on #{NEWPR}, {R4PR_TXT}) | "
             f"Lean + Mathlib, path dependency on 3.6 | {NT} theorems whose types are exactly {NP} registered Props of 3.6 "
             f"({mods_txt}); axioms exactly `propext`, `Classical.choice`, `Quot.sound` | `python3 check.py` / `--execute` | "
             f"`.lake/evidence/receipt.json` with `formalization_status: \"kernel-checked\"` (source mode says `proved`); hosted run {RUN2} "
             f"receipted the 52 theorems of PR head `{C2}` (actual tested merge `{M2}…` with base `{BASE[:8]}…`, the synthetic merge a pull_request run "
             f"checks out, same tree as the head, recorded as `checked_commit`), run {RUN} the first 38 (PR head `{C8}`; actual tested merge `{M1}…`), "
             f"run {RUN3} corroborates (PR head `{C3}`; actual tested merge `{M3}…`), run {RUN4} the same 52 on PR head `{C4}` (actual tested merge `{M4}…`); "
             f"run {RUN6} receipted all 60 on PR #{NEWPR} head `{C6}` (actual tested merge `{M6}…` with base `{BASE2[:8]}…`, the head of #422; "
             f"`leanchecker` PASS; artifact `lean-specification-proofs-{RUN6}-1`) after run {RUNA} on head `{C5A}` (actual tested merge `{M5A}…`, same base) "
             f"first receipted the {len(T3)} `P15Interpolation` theorems; run {RUN7} corroborates the same 60 on PR #{NEWPR} head `{C7}` (actual tested merge "
             f"`{M7}…`, same base; theorem_count 60; artifact `lean-specification-proofs-{RUN7}-1`); the {NPEND4} theorems of the round-4 modules ({new_mods_txt}; "
             f"branch `{BRANCH4}`, {R4PR_TXT}) have no hosted receipt yet (local `lake build` exit 0 and axiom audit on a worktree only): "
             "the first hosted `lean-specification-proofs.yml` run on that branch is their first receipt |")
    doc = doc[:j] + row37 + doc[j:]
    doc = replace_once(doc, "self-award a proof label.\n",
                       "self-award a proof label.\n\n"
                       "The proof-packet gate (3.7) requires every inventoried theorem's type to be exactly a registered Prop of 3.6\n"
                       "applied to the theorem's own binders (a theorem with an extra hypothesis, an implication `P → P`, a narrowed\n"
                       "argument or an unregistered type is refused, first by a source scan and then by a Lean-side audit of the\n"
                       "elaborated `ConstantInfo`), refuses `sorry`, `axiom`, `native_decide`, `unsafe`, `opaque`, metaprogramming\n"
                       "and `#` commands lexically, audits the transitive axioms of every constant the modules add, replays the\n"
                       "build with `leanchecker`, runs negative controls (a `sorry` caught by the axiom audit, a custom axiom, an\n"
                       "unregistered type, a smuggled hypothesis, a false arithmetic statement rejected by Lean) and writes the only\n"
                       "record that may say `kernel-checked`. Helpers live in `Aux` namespaces and are audited but not inventoried;\n"
                       "a module may import a sibling proof module of the packet.\n")

    # ---------------------------------------------------------------- §4 table
    R.short_decl = lambda st: short_decl(st, RECEIPTED60)
    R.render_statements = make_render_statements(RECEIPTED60)
    reg_tbl = copy.deepcopy(reg)
    for r in reg_tbl["claims"]:
        for k in ("not_established", "next_step"):
            r[k] = fuse_runs(r[k])
            for run, m in RUNS:      # every occurrence of a run id is a fused token (stronger than render_spec6's field-level check)
                for pos in occurrences(r[k], run):
                    assert r[k][pos + len(run):].startswith(f"/tested-merge-{m}…"), (r["claim_id"], k, r[k][max(0, pos - 40):pos + 80])
    table = R.render_table(reg_tbl)
    table = replace_once(table,
                         "`main:` prefixes a theorem of the `main` core-Lean package `formal/` listed beneath the "
                         "claim with its own `proved` manifest label (an arithmetic skeleton, never the claim).",
                         "`main:` prefixes a theorem of the `main` core-Lean package `formal/` listed beneath the "
                         "claim with its own `proved` manifest label (an arithmetic skeleton, never the claim); `proof:` prefixes a "
                         "theorem of the Math- proof packet `specifications/lean_statement_proofs_20261011` (PR #422) whose type is "
                         "exactly one of the registered Props of the row, label `proved` (52 kernel-checked by the hosted run "
                         f"{RUN2}: PR head `{C2}`, actual tested merge `{M2}…` with base `{BASE[:8]}…`, same tree as the head; run {RUN}, PR head `{C8}`, "
                         f"actual tested merge `{M1}…`, receipted the first 38; the {len(T3)} theorems of `P15Interpolation` (PR #{NEWPR}) by run {RUN6}, PR head `{C6}`, "
                         f"actual tested merge `{M6}…` with base `{BASE2[:8]}…`, the head of #422; all 60 corroborated on PR #{NEWPR} head `{C7}` by run {RUN7}, "
                         f"actual tested merge `{M7}…`; † marks one of the {NPEND4} theorems of the round-4 modules `Side24Analysis`, `Side24Interfaces`, `RNCluster` "
                         f"(branch `{BRANCH4}`, {R4PR_TXT}), added after head `{C7}` with no hosted receipt yet); a "
                         "proof-packet theorem is evidence about its Prop for the formal objects, never about the claim.")
    doc = splice(doc, "\nColumn key. **Claim**", "\n## 5. Coverage summary\n", "\n" + table + "\n")

    # ---------------------------------------------------------------- §5.1 counts table and prose
    tables = R.coverage_tables(reg, c)
    counts_tbl = replace_once(tables["{{COUNTS_TABLE}}"], "Statement entries (256)", f"Statement entries ({c['statements']})")
    counts_tbl = replace_once(counts_tbl, "statements of other packages (the `main` theorems listed beneath claims); no claim",
                              f"statements of other packages (the 59 `main` arithmetic theorems, 72 entries, and the {NT} proof-packet "
                              f"theorems, {NPE} entries, listed beneath claims; {NPEND4} of the {NT} have no hosted receipt yet); no claim")
    doc = splice(doc, "| Formal-progress label | Claims (171) | Statement entries (228) | Where the label may appear |",
                 "| `kernel-checked` | 0 | 0 | run receipts only, never in the register |", counts_tbl.rsplit("\n", 1)[0] + "\n")
    i = doc.index("The 31 `proved` entries are the SIDE24 theorems of the `main`")
    end = "every claim they sit under stays `specified` or `none`."
    j = doc.index(end, i) + len(end)
    new_props_txt = ", ".join(f"`{p}`" for p in new_props)
    what_proved = ("Proposition P1, both halves of Theorem P2 ((P7) and (P8)) for prices\n"
                   "`c_v ≤ p_v`, the restricted budget (T1) with (T4)-(T5), the six-block union bound (P11) and transformed range, (F9),\n"
                   "(F13), the sharpness clause of `ρ*`, `ρ* < 6/7` with its certificate, the (N1)-(N5) parts of Theorem N and five\n"
                   "elementary SIDE24 identities"
                   f", and, in the round-3 module `P15Interpolation` (PR #{NEWPR}, run {RUN6}), the hazard interpolation chain "
                   + ", ".join(f"`{p}`" for p in proved_interp)
                   + " ((F6)-(F8), (F12) and Theorem F (F2) itself, through the coordinatewise concavity in place of the undisplayed (F4)-(F5))"
                   " and the 20-digit enclosure (F3) `RhoStarEnclosure` (a rational certificate with explicit series remainders, not the source's Fraction procedure)"
                   f", and, in the round-4 modules `Side24Analysis`, `Side24Interfaces` and `RNCluster` (branch `{BRANCH4}`, {R4PR_TXT}; no hosted receipt yet),\n"
                   "summability of the formal theta family `n ↦ exp(-‖z + 24n‖²/2)` over `Fin d → ℤ` for every `d` (`ThetaSummable`), the Section 2 derivative\n"
                   "bounds for `q ≤ 6` (`DerivativePairingBound`, `DerivativeBoundAtLeastOne`, `DerivativeBoundAtZero`), the lattice image sum with its bound\n"
                   "`1458 e^{-288}` (`LatticeImageSumBound`), the five reference block identities and the pin density product for the formal matrices\n"
                   "(`ReferenceOddAndPinBlocks`, `ReferencePinDensityProduct`), the integrand-ratio arithmetic for `d = 2, 3` (`IntegrandRatioBounds`), Loewner\n"
                   "determinant and inverse monotonicity for positive-definite real matrices (`DensityComparison`), over `ConeMomentInterface` the `m = 1` cone\n"
                   "moment `4/3`, the trace moments `5/3`, `25/3`, `√(3/8)` and the distribution function `P(R² ≤ a) = ∫₀^a e^{-z/2}/2 dz` (`ConeMomentD1Identity`,\n"
                   "`TraceMoments`, `RayleighSquareLaw`: for every inhabitant, from the law fields alone, constructing none), and Theorem O's inequality for every\n"
                   "family of pmfs satisfying the Prop's own antecedents (H1), (H2) (`RareClusterPoissonObstructionO`)")
    side24_left_txt = ", ".join(f"`{p}`" for p in side24_left_targets)
    new_par = (
        f"The {c['stmt_fp']['proved']} `proved` entries are theorems of two other packages, each listed beneath a claim and never as the claim.\n"
        "(a) 72 entries for the 59 theorems of the `main` core-Lean package `formal/`: 31 SIDE24 targets at origin/main\n"
        "`f6deeba7` (`Ledger` 17, `ConeMoments` 10, `Endpoints` 4), whose kernel evidence is the `formal / formal-evidence`\n"
        "job of `workspace-landing` run 38068951535 (push to `main`, 2026-10-10, conclusion success; the reusable\n"
        "`formal-verification.yml` runs inside that caller, so it does not appear under its own workflow's run list; artifact\n"
        "`formal-verification-38068951535`; the last stand-alone `formal-verification.yml` run is 36356485214 at `d0c0512f`,\n"
        "2026-09-27, and the 31 Lean texts are byte-identical between the two commits), and 28 P15 targets at commit\n"
        "`91cffbb5` of main PR #358 (`PriceBoundary` 6, `RealizedCovers` 12, `FullPrice` 10; kernel evidence the\n"
        "`formal / formal-evidence` job of `workspace-landing` run 38096827443, artifact `formal-verification-38096827443`;\n"
        "listed on every P15 row whose source they anchor to, hence 41 entries); each of the 59 is a finite integer or\n"
        "cross-multiplied rational identity (`decide`, one conjunct by core `Nat.pow_pos`), an arithmetic skeleton, so none\n"
        f"states a research claim. (b) {NPE} entries for the {NT} theorems of the Math- proof packet\n"
        f"`specifications/lean_statement_proofs_20261011` (PR #422, head `{C4}`; its round-3 successor PR #{NEWPR}, head `{C7}`; and the round-4 branch\n"
        f"`{BRANCH4}`, stacked on #{NEWPR}, {R4PR_TXT}), each typed exactly as one of {NP} registered\n"
        f"Props (`TriangleNeedsThree` twice): 52 of them kernel-checked by the hosted `lean-specification-proofs` run {RUN2} on PR\n"
        f"head `{C2}` (receipt `kernel-checked`, `leanchecker` PASS, axioms within the three allowed; artifact\n"
        f"`lean-specification-proofs-{RUN2}-1`; the 38 theorems of head `{C8}` were first receipted by run {RUN}, artifact\n"
        f"`lean-specification-proofs-{RUN}-1`; run {RUN3} corroborates on head `{C3}` and run {RUN4} on head `{C4}`, theorem_count 52). A pull_request run checks out the synthetic\n"
        f"merge of the PR head into its base, so each receipt's `checked_commit` is that merge commit, not the head. Bound identities:\n"
        f"run {RUN2}, PR head `{C2}`, actual tested merge `{M2F}` with base `{BASE}`; run {RUN}, PR head `{C8}`, actual tested merge\n"
        f"`{M1F}`, same base; run {RUN3}, PR head `{C3}`, actual tested merge `{M3F}`, same base; run {RUN4}, PR head `{C4}`,\n"
        f"actual tested merge `{M4F}`, same base. Because the branch\n"
        "contains the base, each merge's tree equals its head's tree (verified by comparing the tree hashes of the merge and the head), so the\n"
        f"receipted bytes are the head's bytes. The {len(T3)} theorems of the round-3 module `P15Interpolation` ({new_props_txt}) were\n"
        f"added after head `{C4}` in PR #{NEWPR} and are kernel-checked by the hosted run {RUN6} on that PR (receipt `kernel-checked`, `leanchecker` PASS,\n"
        f"theorem_count 60, worktree CLEAN, axioms within the three allowed; artifact `lean-specification-proofs-{RUN6}-1`, sha256 `{SHA6}`): PR head `{C6}`,\n"
        f"actual tested merge `{M6F}` with base `{BASE2}` (the head of PR #422, which #{NEWPR} is stacked on); they were first receipted by run {RUNA}\n"
        f"on PR head `{C5A}`, actual tested merge `{M5AF}`, same base (same Lean bytes; artifact `lean-specification-proofs-{RUNA}-1`, sha256 `{SHA5A}`);\n"
        f"both merges have the tree of their head (the base is an ancestor of the head). All 60 theorems of PR #{NEWPR} head `{C7}` are corroborated by run {RUN7}\n"
        f"(actual tested merge `{M7F}` with base `{BASE2}`; receipt theorem_count 60; artifact `lean-specification-proofs-{RUN7}-1`, sha256 `{SHA7}`).\n"
        f"The {NPEND4} theorems of the round-4 modules {new13_txt} were added after head `{C7}` on the branch `{BRANCH4}`\n"
        f"(commit `{HEAD4}`, stacked on #{NEWPR}; {R4PR_TXT}) and have no hosted receipt yet: locally `lake build` of the packet exits 0 and `#print axioms` of\n"
        "every theorem of the three modules (the 13 inventoried and their helpers) stays within the three allowed axioms, on a worktree, which is evidence about\n"
        "no commit; the first hosted `lean-specification-proofs.yml` run on that branch is their first receipt, and this document will then bind its PR head and\n"
        f"actual tested merge as above. The packet's `check.py` in source mode inventories the {NT} theorems over {NP} distinct registered Props and labels them\n"
        "`proved`, the label of a committed proof package; `kernel-checked` is written by no local run. A proof-packet theorem is evidence about its Prop for the formal objects (the `RealizedData` record, the formal\n"
        f"downset, `coverCost` as an `sInf`, `mu`): {what_proved} are proved *as the Props state them*. Every claim they sit under stays `specified` or\n"
        "`none`: the register admits only those two labels at claim level, alignment of each Prop with its source paragraph\n"
        "is `PENDING_INDEPENDENT_REVIEW`, and the one claim-stating Prop with a theorem (`DemandOneRefutation`, row\n"
        "`p15-price-boundary`) is recorded for the operator and not relabelled. What has no theorem, and why, is tabulated\n"
        f"per group in the proof packet's README (parametrised predicates; the SIDE24 Props {side24_left_txt}; every\n"
        f"RN Prop beyond (N1)-(N5) and Theorem O ({len(rn_left_targets)} theorem-shaped Props and the predicates "
        + ", ".join(f"`{p}`" for p in rn_left_preds) + ") and every Lifetime and Field Prop).\n"
        "\n"
        f"**Round-4 triage** (workflow `{TRIAGE_WF}`, 2026-10-11; same account, organizational-independence credit 0; it moves no label, and its\n"
        "probes are scratch files, not receipts and not in any packet). Of the 82 registered Props without a theorem after round 3, the 8 P15\n"
        f"parametrised predicates were already classified by the proof packet's README and the other {len(triage)} were classified one by one:\n"
        f"{counts['PROVABLE_NOW']} PROVABLE_NOW and {counts['PROVABLE_WITH_WORK']} PROVABLE_WITH_WORK, of which the {counts['PROVABLE_NOW']} and {len(work_done)} of the "
        f"{counts['PROVABLE_WITH_WORK']} (" + ", ".join(f"`{p}`" for p in work_done) + f") are the {NPEND4} round-4 theorems above and {len(work_left)} remain\n"
        "(" + ", ".join(f"`{p}`" for p, _ in work_left) + f"; weeks); {counts['BLOCKED_MISSING_MATHLIB']} BLOCKED_MISSING_MATHLIB (" + ", ".join(f"`{p}`" for p in mm_side24)
        + ": no local, open-set `iteratedFDeriv_tsum` at Mathlib `d13f23b7`;\n" + ", ".join(f"`{p}`" for p in mm_life) + f": weeks); {counts['BLOCKED_INTERFACE']} BLOCKED_INTERFACE "
        f"({bi['RN.lean']} `RN`, {bi['Lifetime.lean']} `Lifetime`, {bi['Side24.lean']} `Side24`), interface under-determination rather than statement\n"
        f"defects: each Prop is refuted at a concrete inhabitant of its interface (a Dirac law with prescribed extra critical points in {n_dirac} cases, the free\n"
        "kernel field `Λ := 0`, a two-point law with a rare large cluster, the constant `p ≡ 0` or `p ≡ 1`; for `CoefficientEnclosure` and `ReferenceComparison`\n"
        "the universal closure would make `PeriodicCoefficientInterface` empty, a probe-level elaboration), so no theorem of its type can exist until the\n"
        f"interface carries more fields; {counts['PREDICATE_VOCABULARY']} PREDICATE_VOCABULARY ({pv['Lifetime.lean']} `Lifetime`, {pv['Field.lean']} `Field`, "
        f"{pv['Side24.lean']} `Side24`, {pv['RN.lean']} `RN`: parametrised predicates, the vocabulary of the other Props, not targets).\n"
        "The proof packet README's informal reasons for its groups 'Side24 derivative bounds', 'Side24 lattice sums', 'Side24 covariance and density' and\n"
        "'Side24 interface statements' were found not load-bearing and are superseded by the theorems; the per-Prop verdicts, efforts and Mathlib references\n"
        "are recorded in the register's `next_step` fields (30 rows). Nothing here is a review: every verdict is author-side."
    )
    doc = doc[:i] + new_par + doc[j:]
    doc = replace_once(doc, "`main formal/` (31 committed theorems listed, 28 staged P15 theorems not listed, as above)",
                       "`main formal/` (59 theorems, all listed as above)")
    # §5.2: Theorem O now has a theorem; the row stays `none` for the reason already given
    doc = replace_once(doc, "`OA-C6-RARE-CLUSTER-20260929-v1`, only Theorem O stated;",
                       "`OA-C6-RARE-CLUSTER-20260929-v1`, only Theorem O stated (and, in round 4, proved as the Prop states it, no hosted receipt yet; "
                       "the row stays `none` because Theorems C, S and (7.1) are not stated);")

    # ---------------------------------------------------------------- §6 roadmap statuses and Step 6
    doc = replace_once(doc,
                       "is the same for every step: the proof is written in Math- `formal/` under the lane's admission rules (the statement\n"
                       "packet's checker refuses `theorem`); the register then lists",
                       "is the same for every step: the proof is written in a Math- Mathlib proof package (the sibling proof packet\n"
                       "`specifications/lean_statement_proofs_20261011`, PR #422, whose gate requires each theorem's type to be exactly a\n"
                       "registered Prop applied to its own binders; the statement packet's checker refuses `theorem`); the register then lists")
    f3_status = (f"`RhoStarEnclosure` (F3) is proved as stated in the round-3 module `P15Interpolation` (PR #{NEWPR}; kernel-checked by run {RUN6}: PR head `{C6}`, actual tested merge `{M6}…`):\n"
                 "a rational certificate (series bounds for `e` and for `exp` with explicit remainders, compared by `norm_num`), not the\n"
                 "source's `full_price.py` Fraction procedure, which remains unspecified.")
    round4_tag = f"round 4 (modules `Side24Analysis`, `Side24Interfaces`; branch `{BRANCH4}`, {R4PR_TXT}; no hosted receipt yet)"
    status = {
        "claim `p15-full-price` stays `specified` because Theorem F itself ((F2)) is not touched.":
            f" *Status 2026-10-11:* `P15.RhoStarBelowSixSevenths` and `P15.SixSeventhsCertificate` are proved as stated in the proof\n"
            f"packet (§3.7; kernel-checked by run {RUN}: PR head `{C8}`, actual tested merge `{M1}…`), by exactly this route (`Real.exp_one_lt_d9`, the degree-5 Taylor\n"
            f"lower bound of `Real.sum_le_exp_of_nonneg`); {f3_status}",
        "complete research statement in the packet.":
            " *Status 2026-10-11:* done in the proof packet, and more: every Prop named here has a theorem of exactly its type (the\n"
            "816 benchmark through Proposition P1's counting lemmas, as predicted, without enumeration; `coverCost` through\n"
            "`csInf_le` / `le_csInf`), and so do (P4), (P5), (P8), (P11), (T1) with (T4)-(T5), (F9), (F13) and `SharpnessOfRhoStar`,\n"
            f"which lie beyond this step's finite scope (modules `P15Hazard`, `P15Covers`, `P15Budget`). All 52 theorems of PR head `{C2}` are\n"
            f"kernel-checked by the hosted run {RUN2} (actual tested merge `{M2}…`; 38 of them already by run {RUN}, PR head `{C8}`,\n"
            f"actual tested merge `{M1}…`). `p15-price-boundary` stays `specified`: a\n"
            "claim-level move is the operator's, after the independent alignment read.",
        "`specified`, nothing more.":
            f" *Status 2026-10-11:* done; the five theorems (`RNProbability`) are kernel-checked by run {RUN} (PR head `{C8}`; actual tested merge `{M1}…`), by the Mathlib lemmas listed;\n"
            "`rn-count-interface` stays `none` as predicted. Beyond this step, in round 4 (module `RNCluster`; branch\n"
            f"`{BRANCH4}`, {R4PR_TXT}; no hosted receipt yet) `RN.RareClusterPoissonObstructionO`, Theorem O of\n"
            "`OA-C6-RARE-CLUSTER-20260929-v1`, is proved as the Prop states it: an elementary statement about pmfs on ℕ (the upper half with the\n"
            "witness `λ = 0`, the lower half through the tail events `{n ≥ 2}` and `{n ≥ 1}`, `ProbabilityTheory.hasSum_one_poissonMeasure`,\n"
            "`Real.add_one_le_exp`, `Real.abs_exp_sub_one_sub_id_le`), for every family satisfying its own antecedents (H1), (H2) and nothing about the\n"
            "Gaussian window count; that row stays `none` because Theorems C, S and (7.1) are not stated.",
        "`ElementaryConeIntegral` is one application of the fundamental theorem of calculus.":
            " *Status 2026-10-11:* `ConeMomentD2Algebra`, `ElementaryConeIntegral` and `ReferenceCoefficientIdentity` are proved as\n"
            f"stated and kernel-checked by run {RUN} (PR head `{C8}`; actual tested merge `{M1}…`; the last as the identity between its two displayed expressions, not as the evaluation\n"
            f"of (15.2) at the reference law). In {round4_tag}\n"
            "`ConeMomentD1Identity`, `TraceMoments` and `RayleighSquareLaw` are proved over `ConeMomentInterface` from its law fields alone (a.e.-measurability\n"
            "from the law field by the Dirac-vs-`NullSingletonClass` argument, `variance_fun_id_gaussianReal`, `iteratedDeriv_mgf_zero` for the fourth moment,\n"
            "`integral_gaussian`; the Rayleigh law as a distribution function through the `polarCoord` change of variables, since Mathlib has no chi-squared or\n"
            "Rayleigh distribution), constructing no inhabitant; `ThetaSummable` for every `d` (`HurwitzKernelBounds.summable_f_int` lifted to `Fin d → ℤ` by a\n"
            "`piFinset` product bound, so the `d`-dimensional theta summability listed as lacking above is proved for the formal family), `DerivativePairingBound`,\n"
            "`DerivativeBoundAtLeastOne` and `DerivativeBoundAtZero` (the operator-norm Leibniz bound `norm_iteratedFDeriv_smul_le` and the Hermite-type\n"
            "recursion on the operator norms, no closed form of the Gaussian derivatives), `LatticeImageSumBound` (the same lift and a geometric tail through\n"
            "`successiveTermRatio`, bound `432 e^{-288} ≤ 1458 e^{-288}`), `ReferenceOddAndPinBlocks` and `ReferencePinDensityProduct` (the derivative values at\n"
            "`0` in frame directions by peeling `iteratedFDeriv_succ_apply_right` and `iteratedFDeriv_clm_apply_const_apply`), `IntegrandRatioBounds` (real\n"
            "arithmetic in `ε`, `a`, `b`) and `DensityComparison` (Loewner inverse and determinant monotonicity through `CFC.sqrt`) are proved as the Props\n"
            f"state them. Still open: {', '.join(f'`{p}`' for p, _ in work_left)} (triage PROVABLE_WITH_WORK, weeks); {', '.join(f'`{p}`' for p in mm_side24)}\n"
            "(BLOCKED_MISSING_MATHLIB: no local, open-set `iteratedFDeriv_tsum` at `d13f23b7`); `CoefficientEnclosure` and `ReferenceComparison` (predicates on\n"
            "the `PeriodicCoefficientInterface` object whose universal closure implies the interface is empty, a probe-level finding). So `side24-coefficient`\n"
            "stays `specified` over `PeriodicCoefficientInterface`, and the enclosure still needs the proof-producing interval pipeline described here.",
        "`proved`. The claims stay `specified`.":
            " *Status 2026-10-11:* unchanged for every Prop of this step: no Prop over `PinnedLawInterface`, `PairLawInterface` or\n"
            "`GaussianFieldInterface` has a theorem, and none is expected from Mathlib alone. The round-4 triage sharpened the reason: 22 of these Props\n"
            f"({bi['RN.lean']} `RN`, {bi['Lifetime.lean']} `Lifetime`) are BLOCKED_INTERFACE, each refuted at a concrete inhabitant of its interface (a Dirac law with\n"
            "prescribed extra critical points; the free kernel field `Λ := 0`; a two-point law with a rare large cluster; the constant `p ≡ 0` or `p ≡ 1` for\n"
            "`ParentUniformSelectionA`, `PlanarElderLowerE2`, `PlanarTwoSidedRateE4`), so the interfaces are under-determined and no theorem of these Props'\n"
            "types can exist until they carry the Kac–Rice data (interface under-determination, not statement defects; the inhabitation checks of (a) made\n"
            f"precise, at probe level, not as receipts); {', '.join(f'`{p}`' for p in mm_life)} are BLOCKED_MISSING_MATHLIB (weeks at best).",
    }
    for anchor, add in status.items():
        doc = replace_once(doc, anchor, anchor + add)
    step6_status = (f" *Status 2026-10-11:* done in the round-3 module `P15Interpolation` (PR #{NEWPR}): all seven Props have a theorem of exactly their\n"
                    "type, by the route predicted here: the one-coordinate split `μ_{p(t)}(A) = S_in + (S_out - S_in)·e^{-t_i}` with\n"
                    "`0 ≤ S_in ≤ S_out` for a downset (display (F4), the erase-injection), then `concaveOn_of_deriv2_nonpos` for\n"
                    "`s ↦ -log(a + b e^{-s})` with second derivative `-ab e^{-s}/(a + b e^{-s})² ≤ 0` (so the page of calculus is one\n"
                    "coordinate's, not the full sum's); (F6) is the chord inequality iterated over the coordinates, (F7) the antitonicity of\n"
                    "`μ_p(A)` in `p` for a downset, (F8) the capped-hazard comparison, (F12) the transport of (F8) to each block's\n"
                    "capacity family with `CapacityWorstCase`, and (F2) the assembly through `GlobalAssemblyHazard` and\n"
                    f"`FullBlockCoverIff`. Kernel-checked by the hosted run {RUN6} (PR head `{C6}`; actual tested merge `{M6}…` with base `{BASE2[:8]}…`, the\n"
                    f"head of #422; first receipted by run {RUNA} on head `{C5A}`, actual tested merge `{M5A}…`). `p15-full-price` stays `specified` (claim-level vocabulary, alignment\n"
                    "PENDING_INDEPENDENT_REVIEW; nothing here is acceptance of Theorem F).")
    step6 = (
        "**Step 6 — Theorem F's hazard interpolation** (`P15.CoordinatewiseConcavity`, `HazardProductBound` (F6),\n"
        "`HazardPhiBound` (F7), `FullGroundPriceBound` (F8), `FullGroundBudgetIff`, then `LocalPriceBoundFull` (F12) and\n"
        "`FullTransformedPriceBudget` (F2)). With (F13) (`GlobalAssemblyHazard`) and the cover criterion (`FullBlockCoverIff`)\n"
        "proved, (F2) reduces to (F12), and (F12) to (F8) for each block's capacity family together with `H_(n,a) ≥ h_star`\n"
        "(`CapacityWorstCase`, proved) and `ρ* = 1/h_star`. (F8) and (F7) follow from (F6), and (F6) from the separate concavity of\n"
        "`t_i ↦ F_A(t) = -log μ_{p(t)}(A)` on `[0, 1]`, `p_i(t) = 1 - e^{-t_i}`, which the source derives from the derivative\n"
        "formulas (F4)-(F5) that it does not display and the register does not specify. *Mathlib has*: `ConcaveOn` with the\n"
        "second-derivative criterion `concaveOn_of_deriv2_nonpos` (Analysis/Convex/Deriv.lean:223), derivatives of `Real.exp`,\n"
        "`Real.log` and finite sums (`HasDerivAt.sum`, Analysis/Calculus/Deriv/Add.lean:222), `Real.log_prod`. *Mathlib lacks*:\n"
        "nothing structural; the work is the explicit second derivative of `-log` of a finite sum of products of `1 - e^{-t}` and\n"
        "`e^{-t}` factors and its sign, a page of calculus per coordinate, then the chain of inequalities. Effort: moderate.\n"
        "Outcome if done: (F2) proved as the Prop states it, for the formal objects; `p15-full-price` stays `specified`\n"
        "(claim-level vocabulary, alignment PENDING_INDEPENDENT_REVIEW)." + step6_status + "\n\n"
    )
    doc = replace_once(doc, "Independent of all five steps, and available now:", step6 + "Independent of all six steps, and available now:")

    # ---------------------------------------------------------------- §7 commands (round 5, unchanged)
    doc = replace_once(doc, '(cd "$P" && lake exe cache get) && python3 -B -S "$P/check.py" --execute\n',
                       '(cd "$P" && lake exe cache get) && python3 -B -S "$P/check.py" --execute\n'
                       '\n'
                       '# Math- statement proofs (this programme; path dependency on the statement register, Mathlib cache as above)\n'
                       'Q=specifications/lean_statement_proofs_20261011\n'
                       'python3 -B -S "$Q/test_check.py" && python3 -B -O -S "$Q/test_check.py"\n'
                       'python3 -B -S "$Q/check.py"                                   # source mode: identity, lexical scan, inventory = registered Props\n'
                       '(cd "$Q" && lake exe cache get) && python3 -B -S "$Q/check.py" --execute   # fresh build, leanchecker (~14 GB), axiom audit, controls, receipt\n')

    # ---------------------------------------------------------------- output invariants
    for bad in ("<<", ">>", "{{", "}}"):
        assert bad not in doc, bad
    for para in re.split(r"\n\s*\n", doc):            # C422-E1 at paragraph level: a run is never named without its tested merge
        for run, merge in RUNS:
            if run in para:
                assert merge in para, (run, para[:100])
    head_of = {b[0]: b[1] for b in BINDINGS}
    for line in doc.split("\n"):                        # and at table-cell level, with the PR head as well
        if line.startswith("|"):
            for cell in line.split(" | "):
                for run, merge in RUNS:
                    if run in cell:
                        assert merge in cell and head_of[run] in cell, (run, cell[-240:])
    for n in set(re.findall(r"\b\d{11}\b", doc)):      # no run id outside the binding list and the main push runs
        assert n in head_of or n in OTHER_RUNS, n
    for phrase in ("uncommitted worktree", "on no commit", "staged in this programme"):
        assert phrase not in doc, phrase
    assert doc.count("†") >= 2 and "no hosted receipt yet" in doc and BRANCH4 in doc and R4PR_TXT in doc, doc.count("†")
    assert all(p in doc for p in new_props) and all(p in doc for p in NEW13_SHORT) and all(m in doc for m in NEW_MODULES)
    for m in NEW_MODULES:                               # the † rows: every statement cell naming a round-4 theorem marks it
        for mt in re.finditer(rf"proof:`{m}\.[A-Za-z0-9]+` \(proved\)", doc):
            assert doc[mt.end():mt.end() + 1] == "†", doc[mt.start() - 20:mt.end() + 5]
    for t in pending:                                   # and no round-4 theorem or module is ever said to be kernel-checked
        for sentence in re.split(r"(?<=[.;])\s", doc):
            if any(f"`{m}" in sentence for m in NEW_MODULES) or any(p in sentence for p in NEW13_SHORT):
                assert "kernel-checked by" not in sentence, sentence[:200]
    assert "Independent of all six steps" in doc
    DOC.write_text(doc)
    print(f"wrote {DOC} ({len(doc)} bytes); claims {c['claims']}, statements {c['statements']}, proved {c['stmt_fp']['proved']}, "
          f"proof theorems {NT} ({NP} Props; {NREC} with a hosted receipt, {NPEND4} without: {sorted(NEW13_SHORT)}); "
          f"round-3 PR #{NEWPR}; round 4 branch {BRANCH4} ({R4PR_TXT}); triage {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
