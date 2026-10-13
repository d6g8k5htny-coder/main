#!/usr/bin/env python3
"""Round-6 update (usage: render_spec6.py <PR number> RUN6 M6 SHA6): as round 5, with the eight P15Interpolation theorems bound to their
hosted receipts (run RUN6 on PR head 0dcb0016; first receipt run 38149524759 on PR head 88c21058; base 79731b96, the head of PR #422). Derived from render_spec5 of main docs/LEAN_VERIFICATION_SPECIFICATION.md from the Math- REGISTER.json after the round-3
proof-packet theorems (Theorem F's hazard interpolation and the F3 enclosure, module P15Interpolation). Runs against
the doc as committed at 91cffbb5 (restore it first). Scientific effect: NONE. Regenerates the §4 claim table and the
§5.1 counts table from the register and applies exact-anchor prose edits; every number quoted is computed from the
register, and every hosted-run mention binds PR head and actual tested merge (C422-E1). Theorems with no hosted
receipt yet (added after PR head 79731b96) are said to have none.

usage: render_spec5.py <new Math- PR number for the round-3 branch>
"""
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_spec as R  # noqa: E402  (helpers only; its main() is not run)

SCRATCH = HERE.parent.parent
import os
DOC = Path(os.environ.get("RS6_DOC", str(SCRATCH / "lvs_main/docs/LEAN_VERIFICATION_SPECIFICATION.md")))
MANIFEST = SCRATCH / "lvs_main/formal/manifest.json"
REGISTER = Path(os.environ.get("RS6_REGISTER", str(SCRATCH / "repos/math/specifications/lean_statements_20261010/REGISTER.json")))
PACKET = R.PACKET
PROOF_PACKAGE = "d6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011"
MAIN_SIDE24 = "d6g8k5htny-coder/main formal/ (UniversalLaw, side24-arithmetic-skeleton)"
MAIN_P15 = "d6g8k5htny-coder/main formal/ at 91cffbb51d1636f382e9d61a1ecb6f7cd38e1d06 (PR #358, not yet on main)"
TIERS = json.loads((SCRATCH / "lvs/register_update5/receipted_52.json").read_text())
RECEIPTED38 = set(TIERS["tier1_run_38116277597_head_00d2ac6e"])
RECEIPTED52 = set(TIERS["all_52"])
assert len(RECEIPTED38) == 38 and len(RECEIPTED52) == 52 and RECEIPTED38 <= RECEIPTED52
RUN, C8 = "38116277597", "00d2ac6e"
RUN2, C2 = "38121713840", "910809d6"
RUN3, C3 = "38122386459", "d9b8dd33"
RUN4, C4 = "38124262242", "79731b96"
BASE = "b2d70d77876a10c365f5db5d3b6cafcf180af4e2"
M1F, M2F, M3F, M4F = ("c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c", "5b198699308e0585904c21d12bf540cf2475d2f0",
                      "866136cca66b46e6dbd3186b47c9340f7e93b4f6", "318ff194eecc42c39a6b27c1d7bf03f30d6a34d9")
M1, M2, M3, M4 = M1F[:8], M2F[:8], M3F[:8], M4F[:8]
NEWPR, RUN6, M6F, SHA6 = sys.argv[1:5]
assert NEWPR.isdigit() and RUN6.isdigit() and re.fullmatch(r"[0-9a-f]{40}", M6F) and re.fullmatch(r"[0-9a-f]{64}", SHA6)
M6, C6 = M6F[:8], "0dcb0016"
RUNA, C5A, M5AF, SHA5A = "38149524759", "88c21058", "3306233bbdeadb15d9b4b07b94cc52a80ed8caeb", "ddefbdcc63223d05fd8ed34addbeebc199e271525d87dead01109cddc529e836"
M5A = M5AF[:8]
BASE2 = "79731b961fac555f33be3cd90e379b246a5a80e3"
RUNS = ((RUN, M1), (RUN2, M2), (RUN3, M3), (RUN4, M4), (RUN6, M6), (RUNA, M5A))
INTERP = ["CoordinatewiseConcavity", "HazardProductBound", "HazardPhiBound", "FullGroundPriceBound", "FullGroundBudgetIff",
          "LocalPriceBoundFull", "FullTransformedPriceBudget"]


def short_decl(st):
    decl = st["declaration"]
    if st["package"] == PACKET:
        return f"`{decl.replace('UniversalLaw.Spec.', '')}`"
    if st["package"] == PROOF_PACKAGE:
        return f"proof:`{decl.replace('UniversalLaw.Proofs.', '')}` ({st['formal_progress']})"
    return f"main:`{decl.replace('UniversalLaw.', '')}` ({st['formal_progress']})"


def render_statements(stmts):
    if not stmts:
        return "—"
    names = [short_decl(s) for s in stmts]
    if len(names) <= R.STMT_SHOW:
        return ", ".join(names)
    n_spec = sum(1 for s in stmts if s["package"] == PACKET)
    n_proof = sum(1 for s in stmts if s["package"] == PROOF_PACKAGE)
    n_main = len(stmts) - n_spec - n_proof
    head, tail, mid = ", ".join(names[:3]), ", ".join(names[-3:]), f"… (+{len(names) - 6} more) …"
    summary = (f"{n_spec} packet statements" + (f" + {n_main} main theorems (`proved`)" if n_main else "")
               + (f" + {n_proof} proof-packet theorems (`proved`)" if n_proof else ""))
    return f"{summary}: {head}, {mid}, {tail}"


R.short_decl = short_decl
R.render_statements = render_statements


def replace_once(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:80])
    return text.replace(old, new)


def splice(text, start_marker, end_marker, body):
    i = text.index(start_marker)
    j = text.index(end_marker, i)
    return text[:i] + body + text[j:]


def main():
    reg = json.loads(REGISTER.read_text())
    manifest = json.loads(MANIFEST.read_text())
    rows = reg["claims"]
    stmts = [s for r in rows for s in r["statements"]]
    c = R.counts(reg, manifest)
    pk = Counter(s["package"] for s in stmts)
    proof = [s for s in stmts if s["package"] == PROOF_PACKAGE]
    proof_names = {s["declaration"] for s in proof}
    main_names = {s["declaration"] for s in stmts if s["package"] in (MAIN_SIDE24, MAIN_P15)}
    pending = sorted(n for n in proof_names if n.split(".")[2] == "P15Interpolation")   # receipted by RUN6 and RUNA, not by the earlier runs
    NT, NPEND, NPE = len(proof_names), len(pending), pk[PROOF_PACKAGE]
    # invariants the prose relies on (what must not have moved) and consistency of the computed numbers
    assert c["claims"] == 171, c["claims"]
    assert c["fp"] == Counter({"none": 138, "specified": 33}), c["fp"]
    assert c["stmt_fp"]["specified"] == 197 and c["stmt_fp"]["proved"] == 72 + NPE and c["statements"] == 197 + 72 + NPE
    assert "kernel-checked" not in c["stmt_fp"]
    assert c["spec_distinct"] == 150 and c["spec_kind"] == Counter({"def_prop": 141, "structure": 9})
    assert c["spec_module"] == Counter({"P15": 58, "Side24": 28, "RN": 28, "Lifetime": 24, "Field": 12}), c["spec_module"]
    assert pk[PACKET] == 197 and pk[MAIN_SIDE24] == 31 and pk[MAIN_P15] == 41, pk
    assert len(main_names) == 59 == c["main_targets"]
    assert RECEIPTED52 <= proof_names and NT == 52 + NPEND and NPEND > 0 and not (RECEIPTED52 & set(pending)), (NT, NPEND)
    covered = set()
    for r in rows:
        anchors = {s["anchor"] for s in r["statements"] if s["package"] == PROOF_PACKAGE}
        covered |= {s["declaration"] for s in r["statements"] if s["package"] == PACKET and s["kind"] == "def_prop" and s["anchor"] in anchors}
    NP = len(covered)
    assert NT == NP + 1, (NT, NP)   # TriangleNeedsThree proved twice, nothing else
    covered_short = {n.split(".", 3)[-1] for n in covered}
    assert c["idonly"] == 82 and c["sources"] == 23
    mods = Counter(n.split(".")[2] for n in proof_names)
    assert set(mods) == {"P15Elementary", "P15Structure", "P15Hazard", "P15Covers", "P15Budget", "Side24Elementary", "RNProbability", "P15Interpolation"}, mods
    assert mods["P15Interpolation"] == NPEND and all(n.split(".")[2] == "P15Interpolation" for n in pending), pending
    mods_txt = ", ".join(f"`{m}` {mods[m]}" for m in ("P15Elementary", "P15Structure", "P15Hazard", "P15Covers", "P15Budget", "P15Interpolation",
                                                     "Side24Elementary", "RNProbability"))
    proved_interp = [p for p in INTERP if p in covered_short]
    f3 = "RhoStarEnclosure" in covered_short
    f2 = "FullTransformedPriceBudget" in covered_short
    new_props = sorted(covered_short & (set(INTERP) | {"RhoStarEnclosure"}))
    assert len(new_props) == NPEND, (new_props, pending)

    doc = DOC.read_text()

    # ---------------------------------------------------------------- header date and §1 bullet 4
    doc = replace_once(doc, "`session_01Cz7WZybv8znP64SpPj6sWY`, 2026-10-10), same GitHub",
                       "`session_01Cz7WZybv8znP64SpPj6sWY`, 2026-10-10, updated 2026-10-11), same GitHub")
    doc = replace_once(doc, "   existing validators (`tools/formal_gate_check.py --alignment`, Math- `formal/gate.py --alignment`).\n",
                       "   existing validators (`tools/formal_gate_check.py --alignment`, Math- `formal/gate.py --alignment`).\n"
                       f"4. **A proved Prop is not an accepted claim.** The proof packet (§3.7) proves {NP} of the 141 registered\n"
                       f"   Props exactly as stated, every one of its {NT} theorems with a hosted kernel receipt (the {NPEND} of module\n"
                       f"   `P15Interpolation`, PR #{NEWPR}, by run {RUN6}: PR head `{C6}`, actual tested merge `{M6}…`). Each such theorem is L5\n"
                       "   evidence for that Prop, for the formal objects it is stated about, and the claim beneath it stays\n"
                       "   `specified` or `none` (the register admits no other claim-level label) until an independent alignment\n"
                       "   review and the controlling acceptance machinery say otherwise.\n")

    # ---------------------------------------------------------------- §2.3 timing sentence
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
             f"stacked on #419; round 3 in PR #{NEWPR}, stacked on #422) | Lean + Mathlib, path dependency on 3.6 | {NT} theorems whose types are exactly {NP} registered Props of 3.6 "
             f"({mods_txt}); axioms exactly `propext`, `Classical.choice`, `Quot.sound` | `python3 check.py` / `--execute` | "
             f"`.lake/evidence/receipt.json` with `formalization_status: \"kernel-checked\"` (source mode says `proved`); hosted run {RUN2} "
             f"receipted the 52 theorems of PR head `{C2}` (actual tested merge `{M2}…` with base `{BASE[:8]}…`, the synthetic merge a pull_request run "
             f"checks out, same tree as the head, recorded as `checked_commit`), run {RUN} the first 38 (PR head `{C8}`; actual tested merge `{M1}…`), "
             f"run {RUN3} corroborates (PR head `{C3}`; actual tested merge `{M3}…`), run {RUN4} the same 52 on PR head `{C4}` (actual tested merge `{M4}…`); "
             f"run {RUN6} receipted all {NT} on PR #{NEWPR} head `{C6}` (actual tested merge `{M6}…` with base `{BASE2[:8]}…`, the head of #422; "
             f"`leanchecker` PASS; artifact `lean-specification-proofs-{RUN6}-1`) after run {RUNA} on head `{C5A}` (actual tested merge `{M5A}…`, same base) "
             f"first receipted the {NPEND} `P15Interpolation` theorems |")
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
    reg_tbl = copy.deepcopy(reg)
    for r in reg_tbl["claims"]:
        for k in ("not_established", "next_step"):
            # fuse run id and tested merge into one whitespace-free token so an abridged cell keeps both or drops both
            for run, head, mf, m, base in ((RUN, C8, M1F, M1, BASE), (RUN2, C2, M2F, M2, BASE), (RUN3, C3, M3F, M3, BASE), (RUN4, C4, M4F, M4, BASE),
                                           (RUN6, C6, M6F, M6, BASE2), (RUNA, C5A, M5AF, M5A, BASE2)):
                r[k] = r[k].replace(f"run {run} (PR head {head}; actual tested merge {mf} with base {base})",
                                    f"run {run}/tested-merge-{m}… (PR head {head}, base {base[:8]}…)")
            for run, m in ((RUN6, M6), (RUNA, M5A)):
                r[k] = r[k].replace(f"artifact lean-specification-proofs-{run}-1", f"artifact lean-specification-proofs-{run}/tested-merge-{m}…-1")
            r[k] = r[k].replace(f"hosted run {RUN} (actual tested merge {M1F} with base {BASE}; ",
                                f"hosted run {RUN}/tested-merge-{M1}… (PR head {C8}, base {BASE[:8]}…; ")
            r[k] = r[k].replace(f"run {RUN4}, actual tested merge {M4F} with base {BASE}",
                                f"run {RUN4}/tested-merge-{M4}… (base {BASE[:8]}…)")
            r[k] = r[k].replace(f"artifact lean-specification-proofs-{RUN}-1", f"artifact lean-specification-proofs-{RUN}/tested-merge-{M1}…-1")
            for run, m in RUNS:
                assert run not in r[k] or f"{run}/tested-merge-{m}" in r[k], (r["claim_id"], k, r[k][:300])
    table = R.render_table(reg_tbl)
    table = replace_once(table,
                         "`main:` prefixes a theorem of the `main` core-Lean package `formal/` listed beneath the "
                         "claim with its own `proved` manifest label (an arithmetic skeleton, never the claim).",
                         "`main:` prefixes a theorem of the `main` core-Lean package `formal/` listed beneath the "
                         "claim with its own `proved` manifest label (an arithmetic skeleton, never the claim); `proof:` prefixes a "
                         "theorem of the Math- proof packet `specifications/lean_statement_proofs_20261011` (PR #422) whose type is "
                         "exactly one of the registered Props of the row, label `proved` (52 kernel-checked by the hosted run "
                         f"{RUN2}: PR head `{C2}`, actual tested merge `{M2}…` with base `{BASE[:8]}…`, same tree as the head; run {RUN}, PR head `{C8}`, "
                         f"actual tested merge `{M1}…`, receipted the first 38; the 8 theorems of `P15Interpolation` (PR #{NEWPR}) by run {RUN6}, PR head `{C6}`, "
                         f"actual tested merge `{M6}…` with base `{BASE2[:8]}…`, the head of #422); a "
                         "proof-packet theorem is evidence about its Prop for the formal objects, never about the claim.")
    doc = splice(doc, "\nColumn key. **Claim**", "\n## 5. Coverage summary\n", "\n" + table + "\n")

    # ---------------------------------------------------------------- §5.1 counts table and prose
    tables = R.coverage_tables(reg, c)
    counts_tbl = replace_once(tables["{{COUNTS_TABLE}}"], "Statement entries (256)", f"Statement entries ({c['statements']})")
    counts_tbl = replace_once(counts_tbl, "statements of other packages (the `main` theorems listed beneath claims); no claim",
                              f"statements of other packages (the 59 `main` arithmetic theorems, 72 entries, and the {NT} proof-packet "
                              f"theorems, {NPE} entries, listed beneath claims); no claim")
    doc = splice(doc, "| Formal-progress label | Claims (171) | Statement entries (228) | Where the label may appear |",
                 "| `kernel-checked` | 0 | 0 | run receipts only, never in the register |", counts_tbl.rsplit("\n", 1)[0] + "\n")
    i = doc.index("The 31 `proved` entries are the SIDE24 theorems of the `main`")
    end = "every claim they sit under stays `specified` or `none`."
    j = doc.index(end, i) + len(end)
    new_props_txt = ", ".join(f"`{p}`" for p in new_props)
    what_proved = ("Proposition P1, both halves of Theorem P2 ((P7) and (P8)) for prices\n"
                   "`c_v ≤ p_v`, the restricted budget (T1) with (T4)-(T5), the six-block union bound (P11) and transformed range, (F9),\n"
                   "(F13), the sharpness clause of `ρ*`, `ρ* < 6/7` with its certificate, the (N1)-(N5) parts of Theorem N and five\n"
                   "elementary SIDE24 identities")
    if proved_interp or f3:
        extra = []
        if proved_interp:
            extra.append("the hazard interpolation chain " + ", ".join(f"`{p}`" for p in proved_interp)
                         + (" ((F6)-(F8), (F12) and Theorem F (F2) itself, through the coordinatewise concavity in place of the undisplayed (F4)-(F5))" if f2 else ""))
        if f3:
            extra.append("the 20-digit enclosure (F3) `RhoStarEnclosure` (a rational certificate with explicit series remainders, not the source's Fraction procedure)")
        what_proved += ", and, in the round-3 module `P15Interpolation` (PR #" + NEWPR + f", run {RUN6}), " + " and ".join(extra)
    missing = []
    if not f2 or len(proved_interp) < len(INTERP):
        left = [p for p in INTERP if p not in proved_interp]
        missing.append("the hazard interpolation Props still without a theorem: " + ", ".join(f"`{p}`" for p in left))
    if not f3:
        missing.append("the 20-digit enclosure (F3)")
    missing_txt = ("; ".join(missing) + "; " if missing else "")
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
        f"`specifications/lean_statement_proofs_20261011` (PR #422, head `{C4}`, and its round-3 successor PR #{NEWPR}), each typed exactly as one of {NP} registered\n"
        f"Props (`TriangleNeedsThree` twice): 52 of them kernel-checked by the hosted `lean-specification-proofs` run {RUN2} on PR\n"
        f"head `{C2}` (receipt `kernel-checked`, `leanchecker` PASS, axioms within the three allowed; artifact\n"
        f"`lean-specification-proofs-{RUN2}-1`; the 38 theorems of head `{C8}` were first receipted by run {RUN}, artifact\n"
        f"`lean-specification-proofs-{RUN}-1`; run {RUN3} corroborates on head `{C3}` and run {RUN4} on head `{C4}`, theorem_count 52). A pull_request run checks out the synthetic\n"
        f"merge of the PR head into its base, so each receipt's `checked_commit` is that merge commit, not the head. Bound identities:\n"
        f"run {RUN2}, PR head `{C2}`, actual tested merge `{M2F}` with base `{BASE}`; run {RUN}, PR head `{C8}`, actual tested merge\n"
        f"`{M1F}`, same base; run {RUN3}, PR head `{C3}`, actual tested merge `{M3F}`, same base; run {RUN4}, PR head `{C4}`,\n"
        f"actual tested merge `{M4F}`, same base. Because the branch\n"
        "contains the base, each merge's tree equals its head's tree (verified by comparing the tree hashes of the merge and the head), so the\n"
        f"receipted bytes are the head's bytes. The {NPEND} theorems of the round-3 module `P15Interpolation` ({new_props_txt}) were\n"
        f"added after head `{C4}` in PR #{NEWPR} and are kernel-checked by the hosted run {RUN6} on that PR (receipt `kernel-checked`, `leanchecker` PASS,\n"
        f"theorem_count {NT}, worktree CLEAN, axioms within the three allowed; artifact `lean-specification-proofs-{RUN6}-1`, sha256 `{SHA6}`): PR head `{C6}`,\n"
        f"actual tested merge `{M6F}` with base `{BASE2}` (the head of PR #422, which #{NEWPR} is stacked on); they were first receipted by run {RUNA}\n"
        f"on PR head `{C5A}`, actual tested merge `{M5AF}`, same base (same Lean bytes; artifact `lean-specification-proofs-{RUNA}-1`, sha256 `{SHA5A}`);\n"
        "both merges have the tree of their head (the base is an ancestor of the head). A proof-packet theorem is evidence about its Prop for the formal objects (the `RealizedData` record, the formal\n"
        f"downset, `coverCost` as an `sInf`, `mu`): {what_proved} are proved *as the Props state them*. Every claim they sit under stays `specified` or\n"
        "`none`: the register admits only those two labels at claim level, alignment of each Prop with its source paragraph\n"
        "is `PENDING_INDEPENDENT_REVIEW`, and the one claim-stating Prop with a theorem (`DemandOneRefutation`, row\n"
        "`p15-price-boundary`) is recorded for the operator and not relabelled. What has no theorem, and why, is tabulated\n"
        f"per group in the proof packet's README ({missing_txt}parametrised predicates; the SIDE24 Fréchet-derivative, lattice-sum and interface Props; every\n"
        "RN Prop beyond (N1)-(N5) and every Lifetime and Field Prop)."
    )
    doc = doc[:i] + new_par + doc[j:]
    doc = replace_once(doc, "`main formal/` (31 committed theorems listed, 28 staged P15 theorems not listed, as above)",
                       "`main formal/` (59 theorems, all listed as above)")

    # ---------------------------------------------------------------- §6 roadmap statuses and Step 6
    doc = replace_once(doc,
                       "is the same for every step: the proof is written in Math- `formal/` under the lane's admission rules (the statement\n"
                       "packet's checker refuses `theorem`); the register then lists",
                       "is the same for every step: the proof is written in a Math- Mathlib proof package (the sibling proof packet\n"
                       "`specifications/lean_statement_proofs_20261011`, PR #422, whose gate requires each theorem's type to be exactly a\n"
                       "registered Prop applied to its own binders; the statement packet's checker refuses `theorem`); the register then lists")
    f3_status = (f"`RhoStarEnclosure` (F3) is proved as stated in the round-3 module `P15Interpolation` (PR #{NEWPR}; kernel-checked by run {RUN6}: PR head `{C6}`, actual tested merge `{M6}…`):\n"
                 "a rational certificate (series bounds for `e` and for `exp` with explicit remainders, compared by `norm_num`), not the\n"
                 "source's `full_price.py` Fraction procedure, which remains unspecified." if f3 else "`RhoStarEnclosure` (F3) is open as described.")
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
            "`rn-count-interface` stays `none` as predicted.",
        "`ElementaryConeIntegral` is one application of the fundamental theorem of calculus.":
            " *Status 2026-10-11:* `ConeMomentD2Algebra`, `ElementaryConeIntegral` and `ReferenceCoefficientIdentity` are proved as\n"
            f"stated and kernel-checked by run {RUN} (PR head `{C8}`; actual tested merge `{M1}…`; the last as the identity between its two displayed expressions, not as the evaluation\n"
            "of (15.2) at the reference law); everything else in this step is open as described.",
        "`proved`. The claims stay `specified`.":
            " *Status 2026-10-11:* unchanged; no Prop over an interface has a theorem, and none is expected from Mathlib alone.",
    }
    for anchor, add in status.items():
        doc = replace_once(doc, anchor, anchor + add)
    if f2 and len(proved_interp) == len(INTERP):
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
    elif proved_interp:
        step6_status = (f" *Status 2026-10-11:* partly done in the round-3 module `P15Interpolation` (PR #{NEWPR}; run {RUN6}, PR head `{C6}`, actual tested merge `{M6}…`): "
                        + ", ".join(f"`{p}`" for p in proved_interp) + " have theorems of exactly their type; "
                        + ", ".join(f"`{p}`" for p in INTERP if p not in proved_interp) + " remain open as described.")
    else:
        step6_status = ""
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

    # ---------------------------------------------------------------- §7 commands
    doc = replace_once(doc, '(cd "$P" && lake exe cache get) && python3 -B -S "$P/check.py" --execute\n',
                       '(cd "$P" && lake exe cache get) && python3 -B -S "$P/check.py" --execute\n'
                       '\n'
                       '# Math- statement proofs (this programme; path dependency on the statement register, Mathlib cache as above)\n'
                       'Q=specifications/lean_statement_proofs_20261011\n'
                       'python3 -B -S "$Q/test_check.py" && python3 -B -O -S "$Q/test_check.py"\n'
                       'python3 -B -S "$Q/check.py"                                   # source mode: identity, lexical scan, inventory = registered Props\n'
                       '(cd "$Q" && lake exe cache get) && python3 -B -S "$Q/check.py" --execute   # fresh build, leanchecker (~14 GB), axiom audit, controls, receipt\n')

    for bad in ("<<", ">>", "{{", "}}"):
        assert bad not in doc, bad
    for para in re.split(r"\n\s*\n", doc):
        for run, merge in RUNS:
            if run in para:
                assert merge in para, (run, para[:100])
    for line in doc.split("\n"):
        if line.startswith("|"):
            for cell in line.split(" | "):
                for run, merge in RUNS:
                    assert run not in cell or merge in cell, (run, cell[-200:])
    for phrase in ("uncommitted worktree", "on no commit", "staged in this programme"):
        assert phrase not in doc, phrase
    assert "†" not in doc and "no hosted receipt" not in doc and all(p in doc for p in new_props)
    DOC.write_text(doc)
    print(f"wrote {DOC} ({len(doc)} bytes); claims {c['claims']}, statements {c['statements']}, proved {c['stmt_fp']['proved']}, "
          f"proof theorems {NT} ({NP} Props; {NPEND} receipted by run {RUN6}: {new_props}); PR #{NEWPR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
