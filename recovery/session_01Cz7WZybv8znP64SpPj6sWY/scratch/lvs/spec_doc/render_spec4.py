#!/usr/bin/env python3
"""Round-2 update of main docs/LEAN_VERIFICATION_SPECIFICATION.md from the Math- REGISTER.json after the
proof packet (Math- #422, head 96cd4a3a). Scientific effect: NONE. Regenerates the §4 claim table and the
§5.1 counts table from the register, and applies exact-anchor prose edits for the facts that changed
(main's 28 P15 targets are at commit 91cffbb5 with a hosted run; the proof packet exists, 52 theorems,
38 kernel-checked at 00d2ac6e; roadmap statuses). Every number quoted is asserted against the register.
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_spec as R  # noqa: E402  (helpers only; its main() is not run)

SCRATCH = HERE.parent.parent
DOC = SCRATCH / "lvs_main/docs/LEAN_VERIFICATION_SPECIFICATION.md"
MANIFEST = SCRATCH / "lvs_main/formal/manifest.json"
REGISTER = SCRATCH / "repos/math/specifications/lean_statements_20261010/REGISTER.json"
PACKET = R.PACKET
PROOF_PACKAGE = "d6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011"
MAIN_SIDE24 = "d6g8k5htny-coder/main formal/ (UniversalLaw, side24-arithmetic-skeleton)"
MAIN_P15 = "d6g8k5htny-coder/main formal/ at 91cffbb51d1636f382e9d61a1ecb6f7cd38e1d06 (PR #358, not yet on main)"
RECEIPTED = {n for mod in json.loads((SCRATCH / "lvs/register_update2/receipted_38.json").read_text())["theorems"].values() for n, _ in mod}
RUN, C8, HEAD2 = "38116277597", "00d2ac6e", "910809d6"
RUN2, C2 = "38121713840", "910809d6"
M1, M2 = "c4c66bb8", "5b198699"
BASE = "b2d70d77876a10c365f5db5d3b6cafcf180af4e2"
M1F, M2F, M3F = "c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c", "5b198699308e0585904c21d12bf540cf2475d2f0", "866136cca66b46e6dbd3186b47c9340f7e93b4f6"
RUN3, C3, M3 = "38122386459", "d9b8dd33", "866136cc"
assert len(RECEIPTED) == 38
RECEIPTED38 = set(RECEIPTED)


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
    # invariants the prose relies on
    assert c["claims"] == 171 and c["statements"] == 329, (c["claims"], c["statements"])
    assert c["fp"] == Counter({"none": 138, "specified": 33}), c["fp"]
    assert c["stmt_fp"] == Counter({"specified": 197, "proved": 132}), c["stmt_fp"]
    assert c["spec_distinct"] == 150 and c["spec_kind"] == Counter({"def_prop": 141, "structure": 9})
    assert c["spec_module"] == Counter({"P15": 58, "Side24": 28, "RN": 28, "Lifetime": 24, "Field": 12}), c["spec_module"]
    assert pk == Counter({PACKET: 197, MAIN_SIDE24: 31, MAIN_P15: 41, PROOF_PACKAGE: 60}), pk
    assert len(main_names) == 59 == c["main_targets"] and len(proof_names) == 52
    assert RECEIPTED38 <= proof_names and len(proof_names - RECEIPTED38) == 14
    proof_props = {s["anchor"] for s in proof}  # anchors copied from the Props; distinct Props counted via the statement packet
    covered = set()
    for r in rows:
        anchors = {s["anchor"] for s in r["statements"] if s["package"] == PROOF_PACKAGE}
        covered |= {s["declaration"] for s in r["statements"] if s["package"] == PACKET and s["kind"] == "def_prop" and s["anchor"] in anchors}
    assert len(covered) == 51, len(covered)
    assert c["idonly"] == 82 and c["sources"] == 23
    mods = Counter(n.split(".")[2] for n in proof_names)
    assert mods == Counter({"P15Elementary": 17, "P15Structure": 13, "P15Hazard": 3, "P15Covers": 4, "P15Budget": 5, "Side24Elementary": 5, "RNProbability": 5}), mods

    doc = DOC.read_text()

    # ---------------------------------------------------------------- header date and §1 bullet 4
    doc = replace_once(doc, "`session_01Cz7WZybv8znP64SpPj6sWY`, 2026-10-10), same GitHub",
                       "`session_01Cz7WZybv8znP64SpPj6sWY`, 2026-10-10, updated 2026-10-11), same GitHub")
    doc = replace_once(doc, "   existing validators (`tools/formal_gate_check.py --alignment`, Math- `formal/gate.py --alignment`).\n",
                       "   existing validators (`tools/formal_gate_check.py --alignment`, Math- `formal/gate.py --alignment`).\n"
                       "4. **A proved Prop is not an accepted claim.** The proof packet (§3.7) proves 51 of the 141 registered\n"
                       "   Props exactly as stated, every one of its 52 theorems with a hosted kernel receipt; each such theorem is L5\n"
                       "   evidence for that Prop, for the formal objects it is stated about, and the claim beneath it stays\n"
                       "   `specified` or `none` (the register admits no other claim-level label) until an independent alignment\n"
                       "   review and the controlling acceptance machinery say otherwise.\n")

    # ---------------------------------------------------------------- §2.3 timing sentence
    doc = replace_once(doc, "9 s `P15`, each exiting 0 with zero warnings; the injected-type-error control exits 1.\n",
                       "9 s `P15`, each exiting 0 with zero warnings; the injected-type-error control exits 1. The proof packet\n"
                       f"(§3.7) rebuilt its five P15 modules in 76 s after the change at `{HEAD2}` (`lake build`, 8,936 jobs, the rest\n"
                       "up to date); its `leanchecker` replay of the Mathlib-importing environment needs about 14 GB and was\n"
                       "memcg-killed in this 15 GB sandbox, so that receipt comes from the hosted run only.\n")

    # ---------------------------------------------------------------- §3 rows 3.1 and 3.7, gate paragraph
    doc = replace_once(doc, "28 P15 staged in this programme's uncommitted worktree, on no commit",
                       "28 P15 at commit `91cffbb5` of main PR #358, not yet on main")
    i = doc.index("| 3.6 | **Lean statement register**")
    j = doc.index("\n", i)
    row37 = ("\n| 3.7 | **Lean statement proofs** (this programme) | Math- `specifications/lean_statement_proofs_20261011/` (PR #422, "
             "stacked on #419) | Lean + Mathlib, path dependency on 3.6 | 52 theorems whose types are exactly 51 registered Props of 3.6 "
             "(`P15Elementary` 17, `P15Structure` 13, `P15Hazard` 3, `P15Covers` 4, `P15Budget` 5, `Side24Elementary` 5, "
             "`RNProbability` 5); axioms exactly `propext`, `Classical.choice`, `Quot.sound` | `python3 check.py` / `--execute` | "
             f"`.lake/evidence/receipt.json` with `formalization_status: \"kernel-checked\"` (source mode says `proved`); hosted run {RUN2} "
             f"receipted all 52 theorems (PR head `{C2}`; actual tested merge `{M2}…` with base `{BASE[:8]}…`, the synthetic merge a pull_request run "
             f"checks out, same tree as the head, recorded as `checked_commit`), run {RUN} the first 38 (PR head `{C8}`; actual tested merge `{M1}…`), "
             f"run {RUN3} corroborates (PR head `{C3}`; actual tested merge `{M3}…`) |")
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
    # the §4 table abridges Not-established / Next-step cells to 200 characters; abbreviate the 40-hex merge and base
    # ids there (the register and §5 carry them in full) so the PR-head / tested-merge binding survives the abridgement
    import copy
    reg_tbl = copy.deepcopy(reg)
    for r in reg_tbl["claims"]:
        for k in ("not_established", "next_step"):
            # fuse run id and tested merge into one whitespace-free token, so a cell cut at a word boundary either
            # keeps both or drops both; the head follows in parentheses
            for run, head, mf, m in ((RUN, C8, M1F, M1), (RUN2, C2, M2F, M2), (RUN3, C3, M3F, M3)):
                r[k] = r[k].replace(f"run {run} (PR head {head}; actual tested merge {mf} with base {BASE})",
                                    f"run {run}/tested-merge-{m}… (PR head {head}, base {BASE[:8]}…)")
            for run, m in ((RUN, M1), (RUN2, M2), (RUN3, M3)):
                assert run not in r[k] or f"{run}/tested-merge-{m}" in r[k], (r["claim_id"], k, r[k][:200])
    table = R.render_table(reg_tbl)
    table = replace_once(table,
                         "`main:` prefixes a theorem of the `main` core-Lean package `formal/` listed beneath the "
                         "claim with its own `proved` manifest label (an arithmetic skeleton, never the claim).",
                         "`main:` prefixes a theorem of the `main` core-Lean package `formal/` listed beneath the "
                         "claim with its own `proved` manifest label (an arithmetic skeleton, never the claim); `proof:` prefixes a "
                         "theorem of the Math- proof packet `specifications/lean_statement_proofs_20261011` (PR #422) whose type is "
                         "exactly one of the registered Props of the row, label `proved` (kernel-checked by the hosted run "
                         f"{RUN2}: PR head `{C2}`, actual tested merge `{M2}…` with base `{BASE[:8]}…`, same tree as the head; run {RUN}, PR head `{C8}`, "
                         f"actual tested merge `{M1}…`, receipted the first 38); a "
                         "proof-packet theorem is evidence about its Prop for the formal objects, never about the claim.")
    doc = splice(doc, "\nColumn key. **Claim**", "\n## 5. Coverage summary\n", "\n" + table + "\n")

    # ---------------------------------------------------------------- §5.1 counts table and prose
    tables = R.coverage_tables(reg, c)
    counts_tbl = replace_once(tables["{{COUNTS_TABLE}}"], "Statement entries (256)", f"Statement entries ({c['statements']})")
    counts_tbl = replace_once(counts_tbl, "statements of other packages (the `main` theorems listed beneath claims); no claim",
                              "statements of other packages (the 59 `main` arithmetic theorems, 72 entries, and the 52 proof-packet "
                              "theorems, 60 entries, listed beneath claims); no claim")
    doc = splice(doc, "| Formal-progress label | Claims (171) | Statement entries (228) | Where the label may appear |",
                 "| `kernel-checked` | 0 | 0 | run receipts only, never in the register |", counts_tbl.rsplit("\n", 1)[0] + "\n")
    i = doc.index("The 31 `proved` entries are the SIDE24 theorems of the `main`")
    end = "every claim they sit under stays `specified` or `none`."
    j = doc.index(end, i) + len(end)
    new_par = (
        "The 132 `proved` entries are theorems of two other packages, each listed beneath a claim and never as the claim.\n"
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
        "states a research claim. (b) 60 entries for the 52 theorems of the Math- proof packet\n"
        f"`specifications/lean_statement_proofs_20261011` (PR #422, head `{C2}`), each typed exactly as one of 51 registered\n"
        f"Props (`TriangleNeedsThree` twice): all 52 kernel-checked by the hosted `lean-specification-proofs` run {RUN2} on PR\n"
        f"head `{C2}` (receipt `kernel-checked`, `leanchecker` PASS, axioms within the three allowed; artifact\n"
        f"`lean-specification-proofs-{RUN2}-1`; the 38 theorems of head `{C8}` were first receipted by run {RUN}, artifact\n"
        f"`lean-specification-proofs-{RUN}-1`; run {RUN3} corroborates on head `{C3}`). A pull_request run checks out the synthetic\n"
        f"merge of the PR head into its base, so each receipt's `checked_commit` is that merge commit, not the head. Bound identities:\n"
f"run {RUN2}, PR head `{C2}`, actual tested merge `{M2F}` with base `{BASE}`; run {RUN}, PR head `{C8}`, actual tested merge\n"
f"`{M1F}`, same base; run {RUN3}, PR head `{C3}`, actual tested merge `{M3F}`, same base. Because the branch\n"
f"contains the base, each merge's tree equals its head's tree (verified by comparing the tree hashes of the merge and the head), so the\n"
f"receipted bytes are the head's bytes. A proof-packet theorem is evidence about its Prop for the formal objects (the `RealizedData` record, the formal\n"
        "downset, `coverCost` as an `sInf`, `mu`): Proposition P1, both halves of Theorem P2 ((P7) and (P8)) for prices\n"
        "`c_v ≤ p_v`, the restricted budget (T1) with (T4)-(T5), the six-block union bound (P11) and transformed range, (F9),\n"
        "(F13), the sharpness clause of `ρ*`, `ρ* < 6/7` with its certificate, the (N1)-(N5) parts of Theorem N and five\n"
        "elementary SIDE24 identities are proved *as the Props state them*. Every claim they sit under stays `specified` or\n"
        "`none`: the register admits only those two labels at claim level, alignment of each Prop with its source paragraph\n"
        "is `PENDING_INDEPENDENT_REVIEW`, and the one claim-stating Prop with a theorem (`DemandOneRefutation`, row\n"
        "`p15-price-boundary`) is recorded for the operator and not relabelled. What has no theorem, and why, is tabulated\n"
        "per group in the proof packet's README (parametrised predicates; the hazard interpolation (F4)-(F8) with (F12) and\n"
        "(F2) behind it; the 20-digit enclosure (F3); the SIDE24 Fréchet-derivative, lattice-sum and interface Props; every\n"
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
    status = {
        "claim `p15-full-price` stays `specified` because Theorem F itself ((F2)) is not touched.":
            f" *Status 2026-10-11:* `P15.RhoStarBelowSixSevenths` and `P15.SixSeventhsCertificate` are proved as stated in the proof\n"
            f"packet (§3.7; kernel-checked by run {RUN}: PR head `{C8}`, actual tested merge `{M1}…`), by exactly this route (`Real.exp_one_lt_d9`, the degree-5 Taylor\n"
            "lower bound of `Real.sum_le_exp_of_nonneg`); `RhoStarEnclosure` (F3) is open as described.",
        "complete research statement in the packet.":
            " *Status 2026-10-11:* done in the proof packet, and more: every Prop named here has a theorem of exactly its type (the\n"
            "816 benchmark through Proposition P1's counting lemmas, as predicted, without enumeration; `coverCost` through\n"
            "`csInf_le` / `le_csInf`), and so do (P4), (P5), (P8), (P11), (T1) with (T4)-(T5), (F9), (F13) and `SharpnessOfRhoStar`,\n"
            f"which lie beyond this step's finite scope (modules `P15Hazard`, `P15Covers`, `P15Budget`). All 52 theorems are\n"
            f"kernel-checked by the hosted run {RUN2} (PR head `{C2}`; actual tested merge `{M2}…`; 38 of them already by run {RUN}, PR head `{C8}`,\n"
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
        "(claim-level vocabulary, alignment PENDING_INDEPENDENT_REVIEW).\n\n"
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
    import re as _re
    for para in _re.split(r"\n\s*\n", doc):
        for run, merge in ((RUN, M1), (RUN2, M2), (RUN3, M3)):
            if run in para:
                assert merge in para, (run, para[:100])
    for line in doc.split("\n"):
        if line.startswith("|"):
            for cell in line.split(" | "):
                for run, merge in ((RUN, M1), (RUN2, M2), (RUN3, M3)):
                    assert run not in cell or merge in cell, (run, cell[-160:])
    for phrase in ("uncommitted worktree", "on no commit", "staged in this programme"):
        assert phrase not in doc, phrase
    DOC.write_text(doc)
    print(f"wrote {DOC} ({len(doc)} bytes); claims {c['claims']}, statements {c['statements']}, proved {c['stmt_fp']['proved']}, proof theorems {len(proof_names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
