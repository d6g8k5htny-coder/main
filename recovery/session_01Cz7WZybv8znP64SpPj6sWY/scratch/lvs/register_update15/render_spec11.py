#!/usr/bin/env python3
"""Round-11 update of main docs/LEAN_VERIFICATION_SPECIFICATION.md from the Math- REGISTER.json at the landed Math- cut
7f5f1c82 (the merge of Math- PR #447 into Math- main, 2026-10-13): round 6 of the proof packet (module Side24Images, PR #441,
merged aef861eb 2026-10-12T18:41:41Z, first hosted receipt run 38277458954 on PR head 9e305ea5, actual tested merge 1376175c...
with base a0323b94..., theorem_count 78) and round 7 of the statement register (PR #445, the Frontier section, merged 6b5b9a5d;
PR #447, the successor interface PeriodicCoefficientInterface2 with CoefficientEnclosure2 and ReferenceComparison2, merged
7f5f1c82), with their bounded same-account reads (credit 0) and integrations.

Unlike render_spec7-10, this renderer is INCREMENTAL: its base text is the document as committed at the branch head b4615124
(read from git and asserted equal to the working file), it regenerates the §4 claim table and the §5.1 counts table from the
7f5f1c82 register and applies exact-anchor prose edits on top. Output is a pure function of that base, the register bytes and
this script. Scientific effect: NONE. No claim-level label changes; `kernel-checked` is a receipt word; a receipt is evidence
about registered Props for the formal objects only; landing on Math- main is an engineering integration, not an acceptance
and not an alignment, and moves no label. Same-account author: organizational-independence credit 0; alignment
PENDING_INDEPENDENT_REVIEW.

usage: python3 -B -S render_spec11.py
env:   RS11_DOC (default <scratch>/lvs_main/docs/LEAN_VERIFICATION_SPECIFICATION.md), RS11_REGISTER (default: the Math- git object)
"""
import copy, hashlib, json, os, re, subprocess, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent.parent
sys.path.insert(0, str(SCRATCH / "lvs/spec_doc"))
import render_spec as R  # noqa: E402

MAIN_REPO = SCRATCH / "lvs_main"
MATH_REPO = SCRATCH / "repos/math_r5"
DOC_PATH = "docs/LEAN_VERIFICATION_SPECIFICATION.md"
DOC = Path(os.environ.get("RS11_DOC", str(MAIN_REPO / DOC_PATH)))
MANIFEST = MAIN_REPO / "formal/manifest.json"
PREV_COMMIT = "b4615124"                                      # branch head: the committed render_spec10 output (merge with main)
CUT = "7f5f1c82478dd8b983759143922224b634a27679"             # Math- main at the merge of #447
CUT5 = "4afcca6a79a8a0acef251ae73988a7914c8080e7"            # the previous cut this document rendered
REG_PATH = "specifications/lean_statements_20261010/REGISTER.json"
REG_BYTES, REG_SHA256, REG_BLOB = 626270, "64674ac463c624d1258c5b7e6766bf0482e14b4a1ad241480f0b0c4de306eaf4", "2eed9bf740c24b52d7e29a1bb8b5a01a51219fa1"
PACKET = R.PACKET
PROOF_PACKAGE = "d6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011"
MAIN_SIDE24 = "d6g8k5htny-coder/main formal/ (UniversalLaw, side24-arithmetic-skeleton)"
MAIN_P15 = "d6g8k5htny-coder/main formal/ at 91cffbb51d1636f382e9d61a1ecb6f7cd38e1d06 (PR #358, not yet on main)"

# ---- rounds 1-5 (FACTS, verbatim from render_spec10)
RUN, C8 = "38116277597", "00d2ac6e"
RUN2, C2 = "38121713840", "910809d6"
RUN3, C3 = "38122386459", "d9b8dd33"
RUN4, C4 = "38124262242", "79731b96"
BASE = "b2d70d77876a10c365f5db5d3b6cafcf180af4e2"
M1F, M2F, M3F, M4F = ("c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c", "5b198699308e0585904c21d12bf540cf2475d2f0",
                      "866136cca66b46e6dbd3186b47c9340f7e93b4f6", "318ff194eecc42c39a6b27c1d7bf03f30d6a34d9")
RUN6, C6, M6F = "38149888699", "0dcb0016", "4363b6928c62da242952631c6b0357d096c3b252"
RUNA, C5A, M5AF = "38149524759", "88c21058", "3306233bbdeadb15d9b4b07b94cc52a80ed8caeb"
BASE2 = "79731b961fac555f33be3cd90e379b246a5a80e3"
RUN7, C7, M7F = "38150363063", "12e072d6", "86a6e8c876f7904cadda2d006a56f8792dc5c1af"
RUN8, H8, M8F, BASE3F = "38180705952", "48bfcbae", "9833d6d9f9facc335e1a56e243c3144adefe6225", "12e072d614dbe6e91b66d22a2ab55cf87860d693"
RUN9, H9, M9F, BASE4F = "38195810463", "c6ce0256", "1320c5b217e491d301ea140e5cc25b07ff5a21b7", "be01479435f98f34b802258ee9ec3c3578f37169"
RUN10, H10, M10F = "38198716748", "32fc9845", "37e0f1d86ed2295e8ec0bbe72b716d934e07079c"
RUNP, RUNS_STMT, RUNS_AGG, HP, MPF = "38205232428", "38205232441", "38205232621", "de732f1e", "8fe0759edb8d93b9cfc840ebbafb5ea8cb6b7ca4"
MAIN_PR_RUN = ("38096827443", "91cffbb5", "3c02eb7d004c4b7a44f5ce7b04c2c5de94e47c0c", "6e11abfb1dc4e424e5b17d0c957157ee7389cf8a", "9357d392")
# ---- round 6 (FACTS): module Side24Images on branch claude/lean-statement-proofs-round6-20261012, Math- PR #441 based on Math- main a0323b94 (merge of #440)
BRANCH6 = "claude/lean-statement-proofs-round6-20261012"
R6_MODULE = "Side24Images"; R6_PROPS = ["ImageBound", "EntrywiseCovarianceBound", "CovarianceOrdering"]
HEAD11F, RUN11, JOB11, M11F, BASE5F = "9e305ea5141a1d1503560efd0e3deaf25441749e", "38277458954", "114906310832", "1376175ce5536a6ac74cb03706d18e65fdb695fa", "a0323b94ae54a78f3d2878dc9c358dc40cdfe1dc"
SHA11, BYTES11 = "1f32d60e08b0ff11b634c0413291686982f0b02e6809b9e96af2848350cbbf79", 47041
HEAD12F, RUN12, JOB12, M12F = "4a3fbadf56c807f7f538fb42ab304ab789d77878", "38285814238", "114933796783", "0017d0ceeabed4dddcca69a9fb44c3f29a59280c"   # receipt-binding successor
HEAD13F, RUN13, JOB13, M13F, TREE13 = "f42491e1ac057d708d0632c636ca8cc81705f335", "38291534116", "114952544961", "fc139a565e34037f6c348844686a46f8008e7fe0", "141dc082db9beba4680be08e687cb5bf972f7b8b"   # integrator's carry
RUN13_STMT, RUN13_AGG, SHA13 = "38291534102", "38291534324", "a64064499aae2aa4cf18571c82e1da9a4f2fe0872253b840c21923081a77ba99"
MERGE6, MERGED6_AT, PUSH6 = "aef861eb4eba87be910e300c5d39da60ddf7cefc", "2026-10-12T18:41:41Z", "38292645778"
REV6A, REV6B, AUDIT6, NOTRUN6, READBACK6 = "5493661142", "5494599128", "6128623199", "6130327350", "6130405801"
# ---- round 7 of the statement register (FACTS): #445 (Frontier section, documents only) and #447 (successor interface), 2026-10-13
BRANCH7A, HEAD7AF, M7AF, BASE7AF = "claude/lean-statements-frontier-20261013", "0c506fea31fd9984c8ca92db135971ca8f3fa420", "640093a349a3eb9c66d4abac1a47eb9601062cb6", "8acc48566de3ae613c1ae320903b66c1d5411797"
RUN7A_STMT, RUN7A_PRF, RUN7A_AGG, MERGE7A, MERGED7A_AT, PUSH7A = "38347541337", "38347541202", "38347541846", "6b5b9a5d7cd1c7f89c2dc5139277c36c8ebe2aaa", "2026-10-13T05:15:38Z", "38350487388"
REV7A1, REV7A2 = "6138311227", "6138611509"
BRANCH7B, HEAD7B_REVF, HEAD7BF, M7BF, TREE7B = "claude/lean-statements-round7-side24b-20261013", "06f03d06d4640a5ab895776b7233d04b9f25f3d6", "f6ba2e6ab398b2c4081d0de2a551caea2e6d46ba", "ce3b69d322466e36936d5c64988dd387266c0536", "d2f5bd70b21f033830f64b30e1811602d25b1406"
RUN7B_STMT, RUN7B_PRF, RUN7B_AGG, MERGED7B_AT, PUSH7B = "38350897827", "38350897766", "38350898066", "2026-10-13T05:27:53Z", "38351512610"
REV7B1, REV7B2, RELEASE7 = "6138083647", "6138686063", "6138984324"
NEW7 = ["PeriodicCoefficientInterface2", "CoefficientEnclosure2", "ReferenceComparison2"]
s8 = lambda x: x[:8]
H11, M11, BASE5, H12, M12, H13, M13, M7A, H7A, M7B, H7B, H7BR = map(s8, (HEAD11F, M11F, BASE5F, HEAD12F, M12F, HEAD13F, M13F, M7AF, HEAD7AF, M7BF, HEAD7BF, HEAD7B_REVF))
# (run, head, full merge, short merge, base): every Math- pull_request run this document may mention
BINDINGS = ((RUN, C8, M1F, s8(M1F), BASE), (RUN2, C2, M2F, s8(M2F), BASE), (RUN3, C3, M3F, s8(M3F), BASE), (RUN4, C4, M4F, s8(M4F), BASE),
            (RUN6, C6, M6F, s8(M6F), BASE2), (RUNA, C5A, M5AF, s8(M5AF), BASE2), (RUN7, C7, M7F, s8(M7F), BASE2), (RUN8, H8, M8F, s8(M8F), BASE3F),
            (RUN9, H9, M9F, s8(M9F), BASE4F), (RUN10, H10, M10F, s8(M10F), BASE4F),
            (RUNP, HP, MPF, s8(MPF), CUT5), (RUNS_STMT, HP, MPF, s8(MPF), CUT5), (RUNS_AGG, HP, MPF, s8(MPF), CUT5),
            (RUN11, H11, M11F, M11, BASE5F), (RUN12, H12, M12F, M12, BASE5F), (RUN13, H13, M13F, M13, BASE5F), (RUN13_STMT, H13, M13F, M13, BASE5F), (RUN13_AGG, H13, M13F, M13, BASE5F),
            (RUN7A_STMT, H7A, M7AF, M7A, BASE7AF), (RUN7A_PRF, H7A, M7AF, M7A, BASE7AF), (RUN7A_AGG, H7A, M7AF, M7A, BASE7AF),
            (RUN7B_STMT, H7B, M7BF, M7B, MERGE7A), (RUN7B_PRF, H7B, M7BF, M7B, MERGE7A), (RUN7B_AGG, H7B, M7BF, M7B, MERGE7A))
RUNS = tuple((b[0], b[3]) for b in BINDINGS)
PUSH_RUNS = {"38068951535": "f6deeba7", "36356485214": "d0c0512f", "38205855996": s8(CUT5), PUSH6: s8(MERGE6), PUSH7A: s8(MERGE7A), PUSH7B: s8(CUT)}
for run, _, mf, _, base in BINDINGS:
    assert run.isdigit() and len(run) == 11 and re.fullmatch(r"[0-9a-f]{40}", mf) and re.fullmatch(r"[0-9a-f]{40}", base), run
assert len({b[0] for b in BINDINGS}) == len(BINDINGS) and not ({b[0] for b in BINDINGS} & (set(PUSH_RUNS) | {MAIN_PR_RUN[0]}))

def git_show(repo, rev_path): return subprocess.run(["git", "-C", str(repo), "show", rev_path], check=True, capture_output=True).stdout
def replace_once(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:90]); return text.replace(old, new)
def splice(text, start_marker, end_marker, body):
    i = text.index(start_marker); j = text.index(end_marker, i); return text[:i] + body + text[j:]
def occurrences(text, sub):
    i = text.find(sub)
    while i >= 0:
        yield i; i = text.find(sub, i + 1)
def paragraphs(text): return re.split(r"\n\s*\n", text)
MENTION_W = 300
def unbound_mentions(doc):
    flat, out = doc.replace("\n", " "), []
    for run, head, mf, m, _ in BINDINGS:
        for pos in occurrences(flat, run):
            win = flat[max(0, pos - MENTION_W):pos + len(run) + MENTION_W]
            if not (head in win and m in win): out.append((run, flat[max(0, pos - 160):pos + 160]))
    return out
def short_decl(st):
    decl = st["declaration"]
    if st["package"] == PACKET: return f"`{decl.replace('UniversalLaw.Spec.', '')}`"
    if st["package"] == PROOF_PACKAGE: return f"proof:`{decl.replace('UniversalLaw.Proofs.', '')}` ({st['formal_progress']})"
    return f"main:`{decl.replace('UniversalLaw.', '')}` ({st['formal_progress']})"
def render_statements(stmts):
    if not stmts: return "—"
    names = [short_decl(s) for s in stmts]
    if len(names) <= R.STMT_SHOW: return ", ".join(names)
    n_spec = sum(1 for s in stmts if s["package"] == PACKET); n_proof = sum(1 for s in stmts if s["package"] == PROOF_PACKAGE); n_main = len(stmts) - n_spec - n_proof
    head, tail, mid = ", ".join(names[:3]), ", ".join(names[-3:]), f"… (+{len(names) - 6} more) …"
    summary = f"{n_spec} packet statements" + (f" + {n_main} main theorems (`proved`)" if n_main else "") + (f" + {n_proof} proof-packet theorems (`proved`)" if n_proof else "")
    return f"{summary}: {head}, {mid}, {tail}"
def fuse_runs(text):
    for run, head, mf, m, base in BINDINGS:
        text = text.replace(f"run {run} (PR head {head}; actual tested merge {mf} with base {base}", f"run {run}/tested-merge-{m}… (PR head {head}, base {base[:8]}…")
        text = text.replace(f"run {run} (actual tested merge {mf} with base {base}", f"run {run}/tested-merge-{m}… (PR head {head}, base {base[:8]}…")
        text = text.replace(f"run {run}, actual tested merge {mf} with base {base}", f"run {run}/tested-merge-{m}… (PR head {head}, base {base[:8]}…)")
        text = text.replace(f"artifact lean-specification-proofs-{run}-1", f"artifact lean-specification-proofs-{run}/tested-merge-{m}…-1")
    text = text.replace(f"hosted run {RUN} receipted)", f"hosted run {RUN}/tested-merge-{s8(M1F)}… (PR head {C8}) receipted)")
    return text

def main():
    reg_bytes = Path(os.environ["RS11_REGISTER"]).read_bytes() if "RS11_REGISTER" in os.environ else git_show(MATH_REPO, f"{CUT}:{REG_PATH}")
    assert len(reg_bytes) == REG_BYTES and hashlib.sha256(reg_bytes).hexdigest() == REG_SHA256
    assert hashlib.sha1(b"blob %d\0" % len(reg_bytes) + reg_bytes).hexdigest() == REG_BLOB
    reg = json.loads(reg_bytes)
    assert reg["scientific_effect"] == "NONE" and reg["scientific_status_authority"] is False and reg["alignment_status"] == "PENDING_INDEPENDENT_REVIEW"
    prev_full = subprocess.run(["git", "-C", str(MAIN_REPO), "rev-parse", PREV_COMMIT], check=True, capture_output=True, text=True).stdout.strip()
    base_doc = git_show(MAIN_REPO, f"{prev_full}:{DOC_PATH}").decode()
    assert DOC.read_text() == base_doc, f"restore {DOC} to its {PREV_COMMIT} bytes first"
    manifest = json.loads(MANIFEST.read_text())
    rows = reg["claims"]; stmts = [s for r in rows for s in r["statements"]]
    c = R.counts(reg, manifest); pk = Counter(s["package"] for s in stmts)
    proof = [s for s in stmts if s["package"] == PROOF_PACKAGE]; proof_names = {s["declaration"] for s in proof}
    module_of = lambda n: n.split(".")[2]
    NT, NPE, NMAIN = len(proof_names), pk[PROOF_PACKAGE], 59
    # invariants (what must not have moved) and the computed numbers
    assert c["claims"] == 171 and c["fp"] == Counter({"none": 138, "specified": 33}), (c["claims"], c["fp"])
    assert c["stmt_fp"]["specified"] == 200 and c["stmt_fp"]["proved"] == 72 + NPE and c["statements"] == 200 + 72 + NPE
    assert (c["statements"], c["stmt_fp"]["proved"], NPE, NT) == (360, 160, 88, 78), (c["statements"], c["stmt_fp"]["proved"], NPE, NT)
    assert "kernel-checked" not in c["stmt_fp"] and "proved" not in c["fp"]
    assert c["spec_distinct"] == 153 and c["spec_kind"] == Counter({"def_prop": 143, "structure": 10})
    assert c["spec_module"] == Counter({"P15": 58, "Side24": 31, "RN": 28, "Lifetime": 24, "Field": 12}), c["spec_module"]
    assert pk[PACKET] == 200 and pk[MAIN_SIDE24] == 31 and pk[MAIN_P15] == 41, pk
    assert c["idonly"] == 82 and c["sources"] == 23
    covered = set()
    for r in rows:
        anchors = {s["anchor"] for s in r["statements"] if s["package"] == PROOF_PACKAGE}
        covered |= {s["declaration"] for s in r["statements"] if s["package"] == PACKET and s["kind"] == "def_prop" and s["anchor"] in anchors}
    NP = len(covered); assert NT == NP + 1 and NP == 77, (NT, NP)
    covered_short = {n.split(".", 3)[-1] for n in covered}
    assert set(R6_PROPS) <= covered_short and not (set(NEW7) & covered_short)
    mods = Counter(module_of(n) for n in proof_names)
    MOD_ORDER = ("P15Elementary", "P15Structure", "P15Hazard", "P15Covers", "P15Budget", "P15Interpolation", "Side24Elementary", "Side24Analysis",
                 "Side24Interfaces", "Side24Covariance", "Side24ConeMoment", R6_MODULE, "RNProbability", "RNCluster")
    assert set(mods) == set(MOD_ORDER) and mods[R6_MODULE] == 3 and sum(mods.values()) == NT, mods
    mods_txt = ", ".join(f"`{m}` {mods[m]}" for m in MOD_ORDER)
    round6 = sorted(n for n in proof_names if module_of(n) == R6_MODULE); assert len(round6) == 3
    prop_of = {}
    for r in rows:
        for s in r["statements"]:
            if s["package"] == PROOF_PACKAGE and s["declaration"] in round6:
                locs = [l["declaration"] for l in r["statements"] if l["package"] == PACKET and l["kind"] == "def_prop" and l["anchor"] == s["anchor"]]
                assert len(locs) == 1 and prop_of.setdefault(s["declaration"], locs[0]) == locs[0], (s["declaration"], locs)
    assert {p.split(".", 3)[-1] for p in prop_of.values()} == set(R6_PROPS)
    new7_decls = {s["declaration"].split(".", 3)[-1] for s in stmts if s["package"] == PACKET and s["module"].endswith("Side24.lean")} & set(NEW7)
    assert new7_decls == set(NEW7), new7_decls
    n_without = 143 - NP; assert n_without == 66
    n_aux = 132   # the statement packet's README at 7f5f1c82: 153 registered declarations, 132 auxiliary definitions (not derivable from the register)

    doc = base_doc
    # ---------------------------------------------------------------- header, §1 bullet 4
    doc = replace_once(doc, "2026-10-10, updated 2026-10-11 and 2026-10-12), same GitHub", "2026-10-10, updated 2026-10-11, 2026-10-12 and 2026-10-13), same GitHub")
    doc = replace_once(doc, "The proof packet (§3.7) proves 74 of the 141 registered\n   Props exactly as stated, and all 75 of its theorems carry a hosted kernel receipt:",
                       f"The proof packet (§3.7) proves {NP} of the 143 registered\n   Props exactly as stated, and all {NT} of its theorems carry a hosted kernel receipt:")
    doc = replace_once(doc, "   same run receipts all 75 theorems of that head). The whole packet is on Math- main since 2026-10-12 (final cut `4afcca6a`, §3);\n",
                       "   same run receipts all 75 theorems of that head), and the 3 of the round-6 module `Side24Images` (branch\n"
                       f"   `{BRANCH6}`, PR #441, based on Math- main) by run {RUN11} on that PR: PR head `{H11}`, actual tested merge\n"
                       f"   `{M11}…` with base `{BASE5}…` (Math- main at the merge of #440; this tested merge differs from the head only outside both packets),\n"
                       f"   receipt theorem_count {NT} (their first receipt; the same run receipts all {NT} theorems of that head). The whole packet is on Math- main\n"
                       f"   (rounds 1-5 since 2026-10-12, cut `{s8(CUT5)}`; round 6 since {MERGED6_AT}, merge `{s8(MERGE6)}`; the statement register's round 7 since\n"
                       f"   2026-10-13, final cut `{s8(CUT)}`, §3);\n")
    # ---------------------------------------------------------------- §3 rows 3.6, 3.7 and the landing paragraph
    doc = replace_once(doc, "(PR #419, on Math- main since its merge `d6f2c325`, 2026-10-12) | Lean + Mathlib, statement-only | 150 registered declarations (141 `def … : Prop`, 9 interface `structure`s; the modules' other 123 declarations are non-Prop definitions), 0 theorems",
                       f"(PR #419, on Math- main since its merge `d6f2c325`, 2026-10-12; round 7, the successor interface `PeriodicCoefficientInterface2` with `CoefficientEnclosure2` and "
                       f"`ReferenceComparison2` appended to `Side24.lean` and a Frontier section classifying the Props without a theorem, by PRs #445 and #447, merged `{s8(MERGE7A)}` and `{s8(CUT)}` on 2026-10-13) | "
                       f"Lean + Mathlib, statement-only | 153 registered declarations (143 `def … : Prop`, 10 interface `structure`s; the modules' other {n_aux} declarations are non-Prop definitions), 0 theorems")
    doc = replace_once(doc, "round 5 in PR #439 on branch `claude/lean-statement-proofs-round5-20261012`; each PR was stacked on its predecessor, #422 on #419, and all five landed on Math- main by normal protected merge commits on 2026-10-12, final cut `4afcca6a`: see the landing paragraph below this table) | Lean + Mathlib, path dependency on 3.6 | 75 theorems whose types are exactly 74 registered Props of 3.6 (",
                       f"round 5 in PR #439 on branch `claude/lean-statement-proofs-round5-20261012`, round 6 in PR #441 on branch `{BRANCH6}`; each of the first five PRs was stacked on its predecessor, #422 on #419, and all five landed on Math- main by normal protected merge commits on 2026-10-12, cut `{s8(CUT5)}`; #441 was based on Math- main and landed by the same route at {MERGED6_AT}, merge `{s8(MERGE6)}`: see the landing paragraphs below this table) | "
                       f"Lean + Mathlib, path dependency on 3.6 | {NT} theorems whose types are exactly {NP} registered Props of 3.6 (")
    doc = replace_once(doc, "`Side24ConeMoment` 1, `RNProbability` 5, `RNCluster` 1); axioms exactly", f"`Side24ConeMoment` 1, `{R6_MODULE}` 3, `RNProbability` 5, `RNCluster` 1); axioms exactly")
    doc = replace_once(doc, "at the final cut the integrator's pre-merge run 38205232428 receipted 75 theorems over 74 Props on PR #439's head at merge `de732f1e`, the integration candidate (actual tested merge `8fe0759e…`, whose tree is the landing tree of `4afcca6a`) |",
                       "at the cut `4afcca6a` the integrator's pre-merge run 38205232428 receipted 75 theorems over 74 Props on PR #439's head at merge `de732f1e`, the integration candidate (actual tested merge `8fe0759e…`, whose tree is the landing tree of `4afcca6a`); "
                       f"run {RUN11} receipted all {NT} on PR #441 head `{H11}` (actual tested merge `{M11}…` with base `{BASE5}…`, Math- main at the merge of #440; `leanchecker` PASS; theorem_count {NT}; artifact `lean-specification-proofs-{RUN11}-1`, sha256 `{SHA11[:8]}…`), "
                       f"the first receipt of the 3 theorems of the round-6 module (`{R6_MODULE}` 3; branch `{BRANCH6}`); run {RUN12} corroborates on PR #441 head `{H12}` (actual tested merge `{M12}…`, same base) and run {RUN13} on the integrator's carry `{H13}` (actual tested merge `{M13}…`, same base; {NT} theorems over {NP} Props per the integrator's readback), whose tree `{TREE13[:8]}…` is the landing tree of `{s8(MERGE6)}` |")
    doc = replace_once(doc, "#439 (round 5) by `4afcca6a79a8a0acef251ae73988a7914c8080e7` at 04:57:01Z, the final cut, whose `REGISTER.json` (blob `05cc80c1`, 599,058 bytes,\nsha256 `d185a292c20084d6f12e56a4ed76f7d6b9294b0d224830e47885934b1869f87c`) is the register this document renders.",
                       "#439 (round 5) by `4afcca6a79a8a0acef251ae73988a7914c8080e7` at 04:57:01Z, the cut whose `REGISTER.json` (blob `05cc80c1`, 599,058 bytes,\nsha256 `d185a292c20084d6f12e56a4ed76f7d6b9294b0d224830e47885934b1869f87c`) this document rendered until 2026-10-13.")
    later = (f"**Later landings (2026-10-12 and 2026-10-13).** Round 6 of the proof packet, Math- PR #441 (branch `{BRANCH6}`, based on Math- main `{BASE5F}`, the merge of #440, not stacked on a PR), landed by "
             f"`{MERGE6}` at {MERGED6_AT} (GitHub `merged_at`), its head at merge the integrator's pure current-main carry `{HEAD13F}` of the reviewed receipt-binding head `{H12}`; the carry's pre-merge runs (event pull_request) are the proof run {RUN13}, "
             f"the statement run {RUN13_STMT} and the required-aggregate run {RUN13_AGG}, all on PR head `{H13}`, actual tested merge `{M13F}` (whose tree `{TREE13}` is the carry's and the landing tree); the landing push run {PUSH6} "
             f"(event push: its checked commit is the pushed merge `{s8(MERGE6)}` itself) ran only `downstream-replay`, `formal / formal-evidence` and `math-downstream-gates`. The statement register's round 7 landed on 2026-10-13: Math- PR #445 "
             f"(branch `{BRANCH7A}`, documents only: the Frontier section) by `{MERGE7A}` at {MERGED7A_AT} (head `{H7A}`; pre-merge runs {RUN7A_STMT}, {RUN7A_PRF} and {RUN7A_AGG} on PR head `{H7A}`, actual tested merge `{M7AF}` with base `{s8(BASE7AF)}…`, "
             f"same tree as the head; landing push run {PUSH7A}, checked commit the pushed `{s8(MERGE7A)}` itself), and Math- PR #447 (branch `{BRANCH7B}`, the successor interface `PeriodicCoefficientInterface2` with `CoefficientEnclosure2` and `ReferenceComparison2`) by "
             f"`{CUT}` at {MERGED7B_AT} (its head at merge the integrator's carry `{HEAD7BF}` of the reviewed head `{H7BR}`, both with tree `{TREE7B[:8]}…`; pre-merge runs {RUN7B_STMT}, {RUN7B_PRF} and {RUN7B_AGG} on PR head `{H7B}`, actual tested merge `{M7BF}` with base `{s8(MERGE7A)}…`, "
             f"same tree as the head; landing push run {PUSH7B}, checked commit the pushed `{s8(CUT)}` itself), the final cut, whose `REGISTER.json` (blob `{REG_BLOB[:8]}`, {REG_BYTES:,} bytes, sha256 `{REG_SHA256}`) is the register this document renders. "
             "Each of these merges was made by an OpenAI/Codex engineering integrator under the same project account (organizational-independence credit 0): an engineering integration, not a mathematical acceptance and not an alignment, and it moves no claim, Prop or theorem label; "
             "the push runs ran no Lean packet and are not receipts for them. The 28 P15 targets of main PR #358 (3.1) are still not on main.\n")
    doc = replace_once(doc, "run the Lean statement or proof packets and is not a receipt for them. The 28 P15 targets of main PR #358 (3.1) are still not on main.\n",
                       "run the Lean statement or proof packets and is not a receipt for them.\n\n" + later)
    # ---------------------------------------------------------------- §4 table, regenerated from the 7f5f1c82 register
    R.short_decl = short_decl; R.render_statements = render_statements
    reg_tbl = copy.deepcopy(reg)
    for r in reg_tbl["claims"]:
        for k in ("not_established", "next_step"):
            r[k] = fuse_runs(r[k])
            for run, m in RUNS:
                for pos in occurrences(r[k], run):
                    assert r[k][pos + len(run):].startswith(f"/tested-merge-{m}…"), (r["claim_id"], k, r[k][max(0, pos - 60):pos + 100])
    table = R.render_table(reg_tbl)
    old_key = doc[doc.index("Column key. **Claim**"):doc.index("\n", doc.index("Column key. **Claim**"))]
    key_start = "`main:` prefixes a theorem of the `main` core-Lean package `formal/` listed beneath the claim with its own `proved` manifest label (an arithmetic skeleton, never the claim);"
    assert key_start in old_key and table.count("`main:` prefixes a theorem of the `main` core-Lean package `formal/` listed beneath the claim with its own `proved` manifest label (an arithmetic skeleton, never the claim).") == 1
    new_key_tail = old_key[old_key.index(key_start):]
    new_key_tail = replace_once(new_key_tail, "(Math- PRs #422, #423, #434 and #439, landed on Math- main, final cut `4afcca6a`)", f"(Math- PRs #422, #423, #434, #439 and #441, landed on Math- main, cuts `{s8(CUT5)}` and `{s8(MERGE6)}`)")
    new_key_tail = replace_once(new_key_tail, "then the head of #434, which receipts all 75 theorems of that head); a proof-packet theorem",
                                f"then the head of #434, which receipts all 75 theorems of that head; the 3 theorems of the round-6 module `{R6_MODULE}` (branch `{BRANCH6}`, PR #441), added after head `{H9}`, by run {RUN11}, PR head `{H11}`, "
                                f"actual tested merge `{M11}…` with base `{BASE5}…`, Math- main at the merge of #440, which receipts all {NT} theorems of that head); a proof-packet theorem")
    table = table.replace("`main:` prefixes a theorem of the `main` core-Lean package `formal/` listed beneath the claim with its own `proved` manifest label (an arithmetic skeleton, never the claim).", new_key_tail, 1)
    assert "**Formal progress**: the claim-level label" in table and table.count("Alignment of every statement with its anchor is `PENDING_INDEPENDENT_REVIEW`.") >= 1
    doc = splice(doc, "\nColumn key. **Claim**", "\n## 5. Coverage summary\n", "\n" + table + "\n")
    # ---------------------------------------------------------------- §5.1 counts table and prose
    tables = R.coverage_tables(reg, c)
    for tok in ("{{KIND_TABLE}}", "{{SPECIFIED_LIST}}"):
        assert doc.count(tables[tok]) == 1, tok
    proved_n = c["stmt_fp"]["proved"]
    counts_tbl = replace_once(tables["{{COUNTS_TABLE}}"], "Statement entries (256)", f"Statement entries ({c['statements']})")
    counts_tbl = replace_once(counts_tbl, "statements of other packages (the `main` theorems listed beneath claims); no claim",
                              f"statements of other packages, one entry per claim a theorem is listed beneath, so {proved_n} entries are not {proved_n} distinct theorems: the {NMAIN} `main` arithmetic theorems, "
                              f"{pk[MAIN_SIDE24] + pk[MAIN_P15]} entries ({pk[MAIN_P15]} of them for the 28 P15 targets at commit `91cffbb5` of main PR #358, not yet on main), and the {NT} proof-packet theorems over {NP} Props, "
                              f"{NPE} entries, every one of the {NT} with a hosted receipt; no claim")
    doc = splice(doc, "| Formal-progress label | Claims (171) | Statement entries (354) | Where the label may appear |", "| `kernel-checked` | 0 | 0 | run receipts only, never in the register |", counts_tbl.rsplit("\n", 1)[0] + "\n")
    doc = replace_once(doc, "The 197 `specified` statement entries point at **150 distinct declarations** of the packet (141 `def … : Prop`,\n9 interface `structure`s): `Field` 12, `Lifetime` 24, `Side24` 28, `RN` 28, `P15` 58.",
                       f"The {c['stmt_fp']['specified']} `specified` statement entries point at **153 distinct declarations** of the packet (143 `def … : Prop`,\n10 interface `structure`s): `Field` 12, `Lifetime` 24, `Side24` 31, `RN` 28, `P15` 58.")
    doc = replace_once(doc, "The modules' other 123 declarations are non-Prop definitions", f"The modules' other {n_aux} declarations are non-Prop definitions")
    doc = replace_once(doc, "The 157 `proved` entries (entries, not distinct theorems: a theorem is listed beneath every claim it serves, and the 157 entries\nare 134 distinct theorems, 59 of `main` and 75 of the proof packet)",
                       f"The {proved_n} `proved` entries (entries, not distinct theorems: a theorem is listed beneath every claim it serves, and the {proved_n} entries\nare {NMAIN + NT} distinct theorems, {NMAIN} of `main` and {NT} of the proof packet)")
    doc = replace_once(doc, "(b) 85 entries for the 75 theorems of the Math- proof packet\n`specifications/lean_statement_proofs_20261011` (rounds 1-2 in PR #422, receipted at head `79731b96`; round 3 in PR #423, head `12e072d6`; round 4 in\nPR #434 on the branch `claude/lean-statement-proofs-round4-20261011`, head `48bfcbae`; round 5 in PR #439 on the branch `claude/lean-statement-proofs-round5-20261012`, heads `c6ce0256` and `32fc9845`; all\nlanded on Math- main on 2026-10-12, final cut `4afcca6a`, §3), each typed exactly as one of 74 registered\nProps",
                       f"(b) {NPE} entries for the {NT} theorems of the Math- proof packet\n`specifications/lean_statement_proofs_20261011` (rounds 1-2 in PR #422, receipted at head `79731b96`; round 3 in PR #423, head `12e072d6`; round 4 in\nPR #434 on the branch `claude/lean-statement-proofs-round4-20261011`, head `48bfcbae`; round 5 in PR #439 on the branch `claude/lean-statement-proofs-round5-20261012`, heads `c6ce0256` and `32fc9845`; round 6 in PR #441 on the branch `{BRANCH6}`, heads `{H11}`, `{H12}` and the integrator's carry `{H13}`; the first five\nlanded on Math- main on 2026-10-12, cut `{s8(CUT5)}`, and #441 later that day, merge `{s8(MERGE6)}`, §3), each typed exactly as one of {NP} registered\nProps")
    round6_par = (f"The 3 theorems of the round-6 module `{R6_MODULE}` (3: " + ", ".join(f"`{p}`" for p in R6_PROPS) + f") were added after head `{H9}` on the branch `{BRANCH6}` (PR #441, based on Math- main `{BASE5F}`,\n"
                  f"the merge of #440; not stacked on a PR) and are kernel-checked by the hosted `lean-specification-proofs.yml` run {RUN11} on that PR (PR head `{HEAD11F}`, actual tested merge\n"
                  f"`{M11F}` with base `{BASE5F}`; attempt 1; event pull_request; job proof-evidence {JOB11}, success; receipt `kernel-checked`, `leanchecker` PASS, theorem_count {NT}, worktree CLEAN,\n"
                  f"axioms within the three allowed; artifact `lean-specification-proofs-{RUN11}-1` (PR head `{H11}`, actual tested merge `{M11}…`), {BYTES11} bytes, sha256 `{SHA11}` as GitHub reports it, the artifact bytes not having been\n"
                  f"downloaded here). Unlike rounds 1-5, that tested merge does not have the head's tree: it differs from the head only outside both packets (13 files under `companions/torus_filtration_20261011/` and\n"
                  f"`.github/workflows/torus-filtration.yml`, main's changes from PRs #428 and #440), so the receipted packet bytes are the head's while the whole tree is not. That run is their first receipt and receipts all {NT}\n"
                  f"theorems of that head. The receipt-binding successor head `{HEAD12F}` (register and documents only, every Lean byte unchanged) is corroborated by run {RUN12} (PR head `{H12}`, actual tested merge\n"
                  f"`{M12F}`, same base; job proof-evidence {JOB12}, success), and the integrator's pure current-main carry `{HEAD13F}` by run {RUN13} (PR head `{H13}`, actual tested merge `{M13F}`, same base, whose tree\n"
                  f"`{TREE13}` is the carry's and the landing tree of `{s8(MERGE6)}`; job proof-evidence {JOB13}, success; the integrator's pre-merge readback {READBACK6} reports `kernel-checked`, `leanchecker` PASS, {NT} theorems over {NP} Props;\n"
                  f"artifact `lean-specification-proofs-{RUN13}-1` (PR head `{H13}`, actual tested merge `{M13}…`), sha256 `{SHA13}` as GitHub reports it). Three bounded nonauthor records on PR #441, all by OpenAI/Codex agents under the same project account\n"
                  f"(organizational-independence credit 0 each), are technical records only: review {REV6A}, a source/type comparison of the three declarations with their registered Props at head `{H11}` (PASS; no local Lean run), review {REV6B},\n"
                  f"a readback of the receipt-binding delta `{H11}..{H12}` (PASS), and the hosted-original byte audit {AUDIT6} of the run's artifacts (PASS); the readiness-triggered automatic review was NOT_RUN (usage limit, {NOTRUN6})\n"
                  "and counts as nothing. None is an alignment review, an approval or an acceptance, and the alignment gate stays open (`PENDING_INDEPENDENT_REVIEW`). ")
    doc = replace_once(doc, "alignment gate stays open (`PENDING_INDEPENDENT_REVIEW`). The packet's `check.py` in source mode\ninventories the 75 theorems over 74 distinct registered Props and labels them",
                       "alignment gate stays open (`PENDING_INDEPENDENT_REVIEW`). " + round6_par + f"The packet's `check.py` in source mode\ninventories the {NT} theorems over {NP} distinct registered Props and labels them")
    doc = replace_once(doc, "(`ConeMomentD2Identity`: for every inhabitant, from the law field `sxy_law` alone, constructing none) are proved *as the Props state them*.",
                       f"(`ConeMomentD2Identity`: for every inhabitant, from the law field `sxy_law` alone, constructing none), and, in the round-6 module `{R6_MODULE}` (branch `{BRANCH6}`, PR #441; run {RUN11}: PR head `{H11}`,\n"
                       "actual tested merge `" + M11 + "…`), display (2)'s bound on `|D^q K_24(0) - D^q K_∞(0)|` for `q ≤ 6`, through a local, open-set version of `iteratedFDeriv_tsum` proved inside the packet from Mathlib's\n"
                       "first-order `hasFDerivAt_tsum_of_isPreconnected` (`ImageBound`), the Section 3 entrywise covariance bound `2E` (`EntrywiseCovarianceBound`) and display (3), the two-sided positive-semidefinite comparison of `C_24`\n"
                       "with `C_ref` within `ε` in every orthonormal frame, from the symmetry of `C_24` (even kernel, block-swap invariance of the analytic contractions) and Cauchy-Schwarz over at most 13 coordinates\n"
                       "(`CovarianceOrdering`) are proved *as the Props state them*.")
    doc = replace_once(doc, "What has no theorem (67 of the 141 registered Props), and why, is tabulated\nper group in the proof packet's README (parametrised predicates; the SIDE24 Props `CoefficientEnclosure`, `CovarianceOrdering`, `EntrywiseCovarianceBound`, `ImageBound`, `ReferenceComparison`; every\nRN Prop beyond (N1)-(N5) and Theorem O (19 theorem-shaped Props and the predicates `HessianHasInertia`, `IsCriticalPoint`) and every Lifetime and Field Prop).",
                       f"What has no theorem ({n_without} of the 143 registered Props), and why, is tabulated\nProp by Prop in the statement packet's Frontier section (its README and SCOPE at `{s8(CUT)}`, from PR #445 with the two round-7 rows of #447; a single author's classification, moving no label): 36 vocabulary\n"
                       "predicates (not targets), 24 interface-blocked Props (19 `RN`, 3 `Lifetime` and the SIDE24 Props `CoefficientEnclosure`, `ReferenceComparison`, each refuted at an inhabitant of its interface), 4 Mathlib-blocked\n"
                       "Lifetime Props and the 2 open targets `CoefficientEnclosure2`, `ReferenceComparison2` over the round-7 successor interface `PeriodicCoefficientInterface2` (36 + 24 + 4 + 2 = 66).")
    doc = replace_once(doc, "7 BLOCKED_MISSING_MATHLIB (`CovarianceOrdering`, `EntrywiseCovarianceBound`, `ImageBound`: no local, open-set `iteratedFDeriv_tsum` at Mathlib `d13f23b7`;",
                       f"7 BLOCKED_MISSING_MATHLIB (`CovarianceOrdering`, `EntrywiseCovarianceBound`, `ImageBound`: no local, open-set `iteratedFDeriv_tsum` at Mathlib `d13f23b7`, since proved in round 6 by building that local version inside the proof packet, `{R6_MODULE}`, run {RUN11}: PR head `{H11}`, actual tested merge `{M11}…` with base `{BASE5}…`;")
    # ---------------------------------------------------------------- §5.2 analytic-coefficient row
    doc = replace_once(doc, "construct none and do not identify its `D_1`, `D_2` with `D_u`); the enclosure,",
                       f"construct none and do not identify its `D_1`, `D_2` with `D_u`; since round 7, Math- #447 merged `{s8(CUT)}`, the numbered successor interface `PeriodicCoefficientInterface2` pins the cone moment to the Gaussian cone expectation under the "
                       "conditional covariance of `K_24` given the pin `V_u = H_f u = 0`, and the successor Props `CoefficientEnclosure2`, `ReferenceComparison2` restate the headline displays over it, all `specified`, no inhabitant constructed); the enclosure,")
    # ---------------------------------------------------------------- §6 Step 4 and Step 5 statuses
    doc = replace_once(doc, "(BLOCKED_MISSING_MATHLIB) and `CoefficientEnclosure`, `ReferenceComparison` (the empty-interface finding above); `side24-coefficient` stays\n`specified`.\n",
                       "(BLOCKED_MISSING_MATHLIB) and `CoefficientEnclosure`, `ReferenceComparison` (the empty-interface finding above); `side24-coefficient` stays\n`specified`. "
                       f"*Status 2026-10-13:* `ImageBound`, `EntrywiseCovarianceBound` and `CovarianceOrdering` are proved as the Props state them in round 6 (module `{R6_MODULE}`, branch `{BRANCH6}`,\n"
                       f"PR #441; kernel-checked by run {RUN11}: PR head `{H11}`, actual tested merge `{M11}…` with base `{BASE5}…`): the missing local, open-set `iteratedFDeriv_tsum` was proved inside the packet from Mathlib's first-order\n"
                       "`hasFDerivAt_tsum_of_isPreconnected`, the lattice translates bounded on the unit ball through the round-4 pairing bounds, and the ordering obtained from the symmetry of `C_24` (even kernel, block-swap\n"
                       "invariance of the analytic contractions) with Cauchy-Schwarz over at most 13 coordinates and `78 E ≤ ε`. For `CoefficientEnclosure` and `ReferenceComparison`, round 7 of the statement register (Math- #447,\n"
                       f"merged `{s8(CUT)}`) appended the numbered successor interface `PeriodicCoefficientInterface2`, whose fields `pinCov_posDef`, `condHessianCov_posDef` and `coneMoment_def` pin the cone moment to the Gaussian cone\n"
                       "expectation under the conditional covariance of `K_24` given the pin `V_u = H_f u = 0`, and the successor Props `CoefficientEnclosure2`, `ReferenceComparison2` restating the headline displays over it; both are\n"
                       "`specified` open targets (the predecessors unchanged; `sphereMeasure` still a given beyond support and total area; no inhabitant constructed; a proof of `ReferenceComparison2` is estimated at weeks over the\n"
                       "packet's `CovarianceOrdering`, `ReferenceCovarianceLowerBound`, `DensityComparison` and `IntegrandRatioBounds`, of `CoefficientEnclosure2` at months with a proof-producing interval pipeline). `side24-coefficient`\n"
                       "stays `specified`.\n")
    doc = replace_once(doc, "`ParentCompactWindowDensityB`, `ParentUnrestrictedLeadingDensityC`, `PlanarDensityGapE11`, `RemainderBoundedR1` are BLOCKED_MISSING_MATHLIB (weeks at best).\n",
                       "`ParentCompactWindowDensityB`, `ParentUnrestrictedLeadingDensityC`, `PlanarDensityGapE11`, `RemainderBoundedR1` are BLOCKED_MISSING_MATHLIB (weeks at best). "
                       f"*Status 2026-10-13:* unchanged in the proof packet. The statement register's round 7 (Math- #445, #447, merged `{s8(MERGE7A)}` and `{s8(CUT)}`) records this classification Prop by Prop in a Frontier section\n"
                       "and opens the successor-interface route for the Side24 pair (`PeriodicCoefficientInterface2`, Step 4); the analogous Lifetime and RN successor interfaces are design packets of this programme awaiting a design read\n"
                       "and activation, in no packet, and nothing here is about them.\n")
    # ---------------------------------------------------------------- §8
    doc = replace_once(doc, "`PENDING_INDEPENDENT_REVIEW`, and the gate that requires independence remains open.\n",
                       "`PENDING_INDEPENDENT_REVIEW`, and the gate that requires independence remains open.\n\n"
                       f"*As of 2026-10-13:* Math- PR #441 (round 6) received a bounded source/type review ({REV6A}, PASS at head `{H11}`), a successor readback ({REV6B}, PASS at `{H12}`) and a hosted-original byte audit ({AUDIT6},\n"
                       f"PASS); Math- PR #445 (the Frontier section) a classification review ({REV7A1}, AMEND_SCOPED_SOURCE, its four findings resolved at head `{H7A}` and read back as PASS_SCOPED_SOURCE, {REV7A2}); Math- PR #447 (round 7)\n"
                       f"a source read ({REV7B1}, PASS_SCOPED) and a changed-base composition readback ({REV7B2}, PASS_SCOPED_COMPOSITION at `{H7BR}`); and #441, #445 and #447 were merged by OpenAI/Codex engineering integrators\n"
                       f"(`{s8(MERGE6)}`, `{s8(MERGE7A)}`, `{s8(CUT)}`; the last released at {RELEASE7}), all under the owner's same project account. Each is recorded here as a technical verdict and, separately, at organizational-independence\n"
                       "credit 0; none is an alignment record, an approval or an acceptance. The alignment dimension of the statement register and of the proof packet stays `PENDING_INDEPENDENT_REVIEW`, and the gate that requires\n"
                       "independence remains open.\n")
    # ---------------------------------------------------------------- output invariants
    for bad in ("<<", ">>", "{{", "}}"): assert bad not in doc, bad
    head_of = {b[0]: b[1] for b in BINDINGS}
    for para in paragraphs(doc):
        for run, merge in RUNS:
            if run in para: assert merge in para, (run, para[:120])
        for run, commit in PUSH_RUNS.items():
            if run in para: assert "push" in para and commit in para, (run, para[:200])
        if MAIN_PR_RUN[0] in para: assert f"PR head `{MAIN_PR_RUN[1]}`" in para and f"tested merge `{MAIN_PR_RUN[2]}`" in para
    for line in doc.split("\n"):
        if line.startswith("|"):
            for cell in line.split(" | "):
                for run, merge in RUNS:
                    if run in cell: assert merge in cell and head_of[run] in cell, (run, cell[-240:])
    for n in set(re.findall(r"\b\d{11}\b", doc)): assert n in head_of or n in PUSH_RUNS or n == MAIN_PR_RUN[0], n
    unbound = unbound_mentions(doc); assert not unbound, "\n".join(f"{r}: ...{ctx}..." for r, ctx in unbound)
    for sentence in re.split(r"(?<=[.;])\s", doc):
        for run in PUSH_RUNS:
            if run in sentence: assert "kernel-checked" not in sentence and "kernel receipt" not in sentence, sentence[:200]
    for stale in ("Statement entries (354)", "The 157 `proved`", "85 entries for", "75 theorems whose types", "74 registered Props of 3.6", "proves 74 of",
                  "67 of the 141", "final cut `4afcca6a`", "**150 distinct declarations**", "other 123 declarations", "are 134 distinct theorems", "inventories the 75 theorems"):
        assert stale not in doc, stale
    assert doc.count("final cut") >= 2 and f"final cut `{s8(CUT)}`" in doc
    for pos in occurrences(doc, "150 registered declarations"):      # historical: the receipt of the cut-4afcca6a statement run only
        assert RUNS_STMT in doc[max(0, pos - 220):pos], doc[max(0, pos - 120):pos + 60]
    assert f"Statement entries ({c['statements']})" in doc and f"proves {NP} of the 143" in doc and f"{NT} theorems whose types are exactly {NP} registered Props" in doc
    for x in (BRANCH6, RUN11, M11F, HEAD11F, BASE5F, SHA11, RUN12, M12F, RUN13, M13F, TREE13, MERGE6, MERGED6_AT, REV6A, REV6B, AUDIT6, MERGE7A, CUT, M7AF, M7BF, HEAD7BF, REV7A1, REV7A2, REV7B1, REV7B2, REG_SHA256):
        assert x in doc, x
    for p in R6_PROPS + NEW7: assert p in doc, p
    for para in paragraphs(doc):
        for rev in (REV6A, REV6B, AUDIT6, REV7A1, REV7A2, REV7B1, REV7B2):
            if rev in para: assert "credit 0" in para and "PENDING_INDEPENDENT_REVIEW" in para, (rev, para[:200])
        if RUN11 in para: assert M11 in para and H11 in para and BASE5 in para, para[:200]
    for sentence in re.split(r"(?<=[.;])\s", doc):
        if "kernel-checked by" in sentence and (f"`{R6_MODULE}" in sentence or any(p in sentence for p in R6_PROPS)): assert RUN11 in sentence, sentence[:240]
        if RUN11 in sentence or RUN12 in sentence or RUN13 in sentence: assert not re.search(r"evidence (about|for|that) the (informal|research) claim", sentence), sentence[:200]
    assert "Nothing is open" not in doc and "stacked on #440" not in doc
    for mt in re.finditer(r"head\s+of\s+(PR\s+)?#4\d\d", doc):
        assert re.search(r"then\s+the\s+$", doc[max(0, mt.start() - 12):mt.start()]), doc[max(0, mt.start() - 80):mt.end() + 20]
    bullet4 = doc[doc.index("4. **A proved Prop is not an accepted claim.**"):doc.index("review and the controlling acceptance machinery say otherwise.")]
    assert f"all {NT} of its theorems carry a hosted kernel receipt" in bullet4 and bullet4.count("their first receipt") == 3 and RUN11 in bullet4 and M11 in bullet4 and H11 in bullet4
    for sentence in re.split(r"(?<=\.)\s", bullet4):
        if "L5" in sentence: assert "receipted" in sentence and "without a hosted receipt" in sentence, sentence
    DOC.write_text(doc)
    print(f"wrote {DOC} ({len(doc.encode())} bytes); register {REG_BLOB[:8]} sha256 {REG_SHA256[:8]}… at Math- {s8(CUT)}; claims {c['claims']} ({dict(c['fp'])}), statements {c['statements']}, "
          f"proved {proved_n} (main {pk[MAIN_SIDE24] + pk[MAIN_P15]} entries / {NMAIN} theorems; proof packet {NPE} entries / {NT} theorems over {NP} Props); round 6 {R6_PROPS} first receipted by run {RUN11} "
          f"(head {H11}, tested merge {M11}…, base {BASE5}…); round 7 statements {NEW7}; {n_without} Props without a theorem")
    return 0

if __name__ == "__main__":
    sys.exit(main())
