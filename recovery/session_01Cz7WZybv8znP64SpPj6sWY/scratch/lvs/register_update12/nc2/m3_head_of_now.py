#!/usr/bin/env python3
"""Round-10 update of main docs/LEAN_VERIFICATION_SPECIFICATION.md from the Math- REGISTER.json at the landed Math- cut
4afcca6a (the merge of Math- PR #439 into Math- main, 2026-10-12): the 2 round-5 proof-packet theorems (modules
Side24Covariance, Side24ConeMoment; branch claude/lean-statement-proofs-round5-20261012, Math- PR #439) with their first
hosted receipt, lean-specification-proofs.yml run 38195810463 (attempt 1, job 114638065311) on PR head c6ce0256, actual
tested merge 1320c5b217e491d301ea140e5cc25b07ff5a21b7 with base be01479435f98f34b802258ee9ec3c3578f37169 (the head of PR
#434), receipt theorem_count 75, its corroboration run 38198716748 on the receipt-binding successor head 32fc9845 (actual
tested merge 37e0f1d8…, same base), the two bounded same-account nonauthor reviews of PR #439 (5486104267, 5486293837;
organizational-independence credit 0), and the landing of the whole Math- stack #419 -> #422 -> #423 -> #434 -> #439 on
Math- main by five normal protected merge commits. Derived from render_spec9.py the way render_spec9 was derived from
render_spec8.py: everything render_spec9 does is kept (every hosted-run binding of rounds 1-4 verbatim, the triage
paragraph, every assertion), and in addition the document

  (a) binds round 5 in every place it enumerates proof rounds: §1 bullet 4, the §3.7 row, the §4 column key and claim
      rows, the §5.1 counts table and prose, §5.2 (the analytic-coefficient row), the §6 Step 4 status and §8;
  (b) moves every count that moved (352 -> 354 statement entries, 155 -> 157 `proved` entries, 83 -> 85 proof-packet
      entries, 73 -> 75 theorems, 72 -> 74 Props), each asserted against the register, and states explicitly that entry
      counts are not distinct theorems (the coordination note main#358 6118106399) while the round-4 receipt numbers
      (theorem_count 73, covered_props 72) stay as historical constants NT4 / NP4;
  (c) replaces the present-tense 'stacked on' descriptions of the Math- PRs by their landed state and adds one dated
      landing paragraph (2026-10-12) with the five merge commits and the final cut's pre-merge runs (head and tested merge
      bound); the landing push run 38205855996 is named as a push run whose checked commit is the pushed commit itself and
      that did NOT run the Lean packets;
  (d) folds in the three post-render wording fixes that commit fe99ca31 applied by hand to render_spec9's output (the
      non-duplicated first-receipt clause, 'every control classified as intended', the OA-C6 next-step cell ending with
      its receipt), so the renderer alone now produces the document;
  (e) binds the three main-repository runs that §5.1 names: the two push runs with their pushed commits, and the
      pull_request run 38096827443 with its PR head 91cffbb5 and its actual tested merge 3c02eb7d (base 6e11abfb), read from
      the checkout log and the receipt's checked_commit in the log of its job 114344336471 (fix round: the first render said
      the merge was not recorded, the job log having been unreadable through the gh blob redirect);
  (f) corrects one sentence of §6 that contradicted the register's claim-level vocabulary ('the claim's own label moves to
      `proved` ...': a claim row may carry only `none` or `specified`), in the register's own words (fix round: no alignment of
      `DemandOneRefutation` with its source is assumed; it is 'the Prop the register names as stating the claim itself'), and
      likewise §6 Step 2's 2026-10-10 outcome ('could carry `proved`'), its status sentence and the §5.1 sentence on that Prop;
  (g) corrects two §1 sentences the full read found false: 'Three things follow' heads four items (stale since render_spec5
      added bullet 4), and bullet 3's headline 'Nothing here is reviewed' predates the bounded same-account nonauthor reviews
      (now 'Nothing here is alignment-reviewed', with those reviews and the integrations named as credit-0 technical records);
  (h) fix round, from the verifier's findings on the first commit: every mention of a Math- pull_request run carries its PR
      head and tested merge within the mention (asserted per occurrence, not only per paragraph); a run's base is 'then the
      head of #4xx'; the landing times are labelled GitHub `merged_at`, and de732f1e is named as PR #439's head at merge; the
      column key notes that two transcribed fields of row cert:math-full_price-full_price.py predate round 3; §8 says a
      cross-provider review under the shared account earns organizational-independence credit 0.

Determinism: the base text is ALWAYS the doc as committed at 91cffbb5, read from git (`git show 91cffbb5:<doc>`), never the
working file; before writing, the working file must equal the doc as committed at fe99ca31 (restore it first with
`git checkout fe99ca31 -- docs/LEAN_VERIFICATION_SPECIFICATION.md`), so that the output is a pure function of the base, the
register bytes and this script. The register is read from the Math- git object at 4afcca6a (or RS7_REGISTER) and its size,
SHA-256 and git blob id are asserted.

Scientific effect: NONE. Regenerates the §4 claim table and the §5.1 counts table from the register and applies exact-anchor
prose edits. No claim-level label changes; `kernel-checked` is a receipt word; a receipt is evidence about the registered
Props for the formal objects only, never about an informal claim; landing on Math- main is an engineering integration, not
an acceptance and not an alignment, and moves no label. Same-account author: organizational-independence credit 0;
alignment PENDING_INDEPENDENT_REVIEW. main PR #315's recomposition of #358 is not attempted here.

usage: render_spec10.py <round-3 PR number> RUN6 M6 SHA6 --round4-pr 434 --round4-run RUN8 --round4-head HEAD8 --round4-merge M8
                        --round4-sha SHA8 --round5-pr 439 --round5-run RUN9 --round5-head HEAD9 --round5-merge M9 --round5-sha SHA9
env:   RS7_DOC (default <scratch>/lvs_main/docs/LEAN_VERIFICATION_SPECIFICATION.md), RS7_REGISTER (default: the Math- git object)
"""
import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent.parent
sys.path.insert(0, str(SCRATCH / "lvs/spec_doc"))
import render_spec as R  # noqa: E402  (helpers only; its main() is not run)

MAIN_REPO = SCRATCH / "lvs_main"
MATH_REPO = SCRATCH / "repos/math"
DOC_PATH = "docs/LEAN_VERIFICATION_SPECIFICATION.md"
DOC = Path(os.environ.get("RS7_DOC", str(MAIN_REPO / DOC_PATH)))
MANIFEST = MAIN_REPO / "formal/manifest.json"
BASE_COMMIT = "91cffbb51d1636f382e9d61a1ecb6f7cd38e1d06"      # the doc every render_spec* applies its edits to
PREV_COMMIT = "fe99ca3130daba2c4a4611b29dad2b7d7d0082b4"      # the last committed render (render_spec9 + three hand fixes)
CUT = "4afcca6a79a8a0acef251ae73988a7914c8080e7"             # the final landed Math- cut (merge of #439)
REG_PATH = "specifications/lean_statements_20261010/REGISTER.json"
REG_BYTES, REG_SHA256, REG_BLOB = 599058, "d185a292c20084d6f12e56a4ed76f7d6b9294b0d224830e47885934b1869f87c", "05cc80c1ed9040aa7282fb1543679743807d55ba"
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
# ---- round 3 corroboration on PR #423 head 12e072d6 (FACTS), the last head with a hosted receipt before round 4
RUN7, C7, M7F, SHA7 = "38150363063", "12e072d6", "86a6e8c876f7904cadda2d006a56f8792dc5c1af", "bf0e0515e2035103b07d894e2a73eeb5b542885f6da16b98c228877a5e0c1690"
M7 = M7F[:8]
# ---- round 4 (FACTS): three modules committed on the round-4 branch; first hosted receipt by run 38180705952 on PR #434 head 48bfcbae (args, asserted)
BRANCH4 = "claude/lean-statement-proofs-round4-20261011"
HEAD4 = "80e80243"            # 80e80243da9de0bd81703bbc2dd19782196fbefe, one commit on 12e072d6, adds the three modules
NEW_MODULES = ("Side24Analysis", "Side24Interfaces", "RNCluster")
NEW_MODULE_COUNTS = {"Side24Analysis": 9, "Side24Interfaces": 3, "RNCluster": 1}
NT4, NP4 = 73, 72             # the round-4 receipt's theorem_count and covered_props: historical numbers, never the current ones
# ---- round 5 (FACTS): two modules on the round-5 branch; first hosted receipt by run 38195810463 on PR #439 head c6ce0256 (args, asserted)
BRANCH5 = "claude/lean-statement-proofs-round5-20261012"
R5_MODULES = ("Side24Covariance", "Side24ConeMoment")
R5_PROPS = {"Side24Covariance": "ReferenceCovarianceLowerBound", "Side24ConeMoment": "ConeMomentD2Identity"}
JOB9 = "114638065311"
BASE4F = "be01479435f98f34b802258ee9ec3c3578f37169"   # head of PR #434 after its receipt-binding commit, the base of runs RUN9 and RUN10 (FACTS)
BASE4 = BASE4F[:8]
RUN10, HEAD10F, M10F = "38198716748", "32fc98455128c9903741aece5f79744ffe928d31", "37e0f1d86ed2295e8ec0bbe72b716d934e07079c"   # corroboration (FACTS)
H10, M10 = HEAD10F[:8], M10F[:8]
REV1, REV2 = "5486104267", "5486293837"   # both OpenAI/Codex, same project account, organizational-independence credit 0 (FACTS)
# ---- landing (FACTS): five normal protected MERGE commits on Math- main, each by a distinct OpenAI/Codex engineering integrator (credit 0)
LANDING = (("419", "statement register", "d6f2c325c55b78002a113e827d23539df90cad88", "2026-10-12T03:57:27Z"),
           ("422", "proof packet, rounds 1-2", "8a61d5fb59fa8fa18bddcfc7d8f8bb88b12a46c0", "04:10:53Z"),
           ("423", "round 3", "6e366697546ad2afadb84150e0d4c609d0918945", "04:24:30Z"),
           ("434", "round 4", "7fd045a6677bc27b95c05c4149bbf51e86c364b9", "04:40:46Z"),
           ("439", "round 5", CUT, "04:57:01Z"))
RUNP, RUNS_STMT, RUNS_AGG = "38205232428", "38205232441", "38205232621"   # the final cut's pre-merge runs (FACTS)
HEADPF, MPF = "de732f1e8622ad92d80126e11df8ace85075f0ed", "8fe0759edb8d93b9cfc840ebbafb5ea8cb6b7ca4"
HP, MP = HEADPF[:8], MPF[:8]
TREE_CUT = "d323c3d63c3afd9981f679c6fb1c873f09772523"
INTEG_REVIEW, RELEASE_COMMENT = "5486666735", "6118068559"
RUN_LANDPUSH = "38205855996"   # event push at 4afcca6a; ran downstream-replay, formal/formal-evidence, math-downstream-gates only
# ---- runs that are not Math- proof-packet pull_request runs: push runs bind the pushed commit itself; the one main pull_request run binds its PR head
PUSH_RUNS = {"38068951535": "f6deeba7", "36356485214": "d0c0512f", RUN_LANDPUSH: CUT[:8]}
# workspace-landing run 38096827443, event pull_request on main PR #358, job formal / formal-evidence 114344336471: PR head 91cffbb5; actual tested
# merge 3c02eb7d (the checkout log: "HEAD is now at 3c02eb7 Merge 91cffbb5... into 6e11abfb..."; the receipt's checked_commit), whose formal/ subtree
# equals the head's (9357d392) while its whole tree does not (6e11abfb is not an ancestor of 91cffbb5). Read from the job log via the GitHub MCP server.
MAIN_PR_RUN = ("38096827443", "91cffbb5", "3c02eb7d004c4b7a44f5ce7b04c2c5de94e47c0c", "6e11abfb1dc4e424e5b17d0c957157ee7389cf8a", "9357d392")
MERGED_AT_NOTE = "GitHub `merged_at`; each merge commit's own committer date is at most one second earlier"
TRIAGE_WF = "wf_a3531fd8-891"
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
ap.add_argument("--round4-pr", required=True, help="PR number of the round-4 branch (434)")
ap.add_argument("--round4-run", required=True, help="hosted lean-specification-proofs.yml run that first receipted the 13 round-4 theorems (38180705952)")
ap.add_argument("--round4-head", required=True, help="PR head of that run (40 hex; 48bfcbae...)")
ap.add_argument("--round4-merge", required=True, help="actual tested merge commit of that run, the receipt's checked_commit (40 hex; 9833d6d9...)")
ap.add_argument("--round4-sha", required=True, help="sha256 of its artifact lean-specification-proofs-<run>-1 as GitHub reports it (64 hex)")
ap.add_argument("--round5-pr", required=True, help="PR number of the round-5 branch (439)")
ap.add_argument("--round5-run", required=True, help="hosted lean-specification-proofs.yml run that first receipted the 2 round-5 theorems (38195810463)")
ap.add_argument("--round5-head", required=True, help="PR head of that run (40 hex; c6ce0256...)")
ap.add_argument("--round5-merge", required=True, help="actual tested merge commit of that run (40 hex; 1320c5b2...)")
ap.add_argument("--round5-sha", required=True, help="sha256 of its artifact lean-specification-proofs-<run>-1 as GitHub reports it (64 hex)")
ARGS = ap.parse_args()
NEWPR, RUN6, M6F, SHA6 = ARGS.newpr, ARGS.run6, ARGS.m6, ARGS.sha6
assert NEWPR.isdigit() and RUN6.isdigit() and re.fullmatch(r"[0-9a-f]{40}", M6F) and re.fullmatch(r"[0-9a-f]{64}", SHA6)
assert RUN6 == "38149888699" and M6F == "4363b6928c62da242952631c6b0357d096c3b252", "round-3 binding must match the FACTS"   # the register is written with these
assert SHA6 == "cdeb5c73aa03124ed0db16fa14d25fb03a8fa842457fce1e0c6565dbc5ee7ca8", "round-3 artifact digest as committed at fe99ca31"
M6, C6 = M6F[:8], "0dcb0016"
assert ARGS.round4_pr.isdigit() and ARGS.round4_pr == "434", ARGS.round4_pr          # the receipt is a pull_request run of PR #434
R4PR_TXT = f"PR #{ARGS.round4_pr}"
RUN8, HEAD8F, M8F, SHA8 = ARGS.round4_run, ARGS.round4_head, ARGS.round4_merge, ARGS.round4_sha
assert RUN8.isdigit() and len(RUN8) == 11 and re.fullmatch(r"[0-9a-f]{40}", HEAD8F) and re.fullmatch(r"[0-9a-f]{40}", M8F) and re.fullmatch(r"[0-9a-f]{64}", SHA8)
assert (RUN8, HEAD8F, M8F) == ("38180705952", "48bfcbae7834c2d6f4d3123d85c69d0c5afedfa6", "9833d6d9f9facc335e1a56e243c3144adefe6225"), "round-4 binding must match the FACTS"
assert SHA8 == "87309efad9b804c53c543b60fdb466de1f797971417363cc2e8fa1e10eac26bd", "artifact digest must match the FACTS (GitHub's digest; bytes not downloaded)"
H8, M8 = HEAD8F[:8], M8F[:8]
BASE3F = "12e072d614dbe6e91b66d22a2ab55cf87860d693"   # tip of the base branch of PR #434 = PR #423 head 12e072d6 (FACTS)
BASE3 = BASE3F[:8]
TREE8 = "1956e6298d8d34ef7b1fca786903d1e0e72cb220"    # tree of the merge, equal to the head's (FACTS)
assert BASE3 == C7
assert ARGS.round5_pr == "439", ARGS.round5_pr                                     # the receipt is a pull_request run of PR #439
R5PR_TXT = f"PR #{ARGS.round5_pr}"
RUN9, HEAD9F, M9F, SHA9 = ARGS.round5_run, ARGS.round5_head, ARGS.round5_merge, ARGS.round5_sha
assert RUN9.isdigit() and len(RUN9) == 11 and re.fullmatch(r"[0-9a-f]{40}", HEAD9F) and re.fullmatch(r"[0-9a-f]{40}", M9F) and re.fullmatch(r"[0-9a-f]{64}", SHA9)
assert (RUN9, HEAD9F, M9F) == ("38195810463", "c6ce025652f76be6e580fa24947380253450cd3b", "1320c5b217e491d301ea140e5cc25b07ff5a21b7"), "round-5 binding must match the FACTS"
assert SHA9 == "d8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff", "artifact digest must match the FACTS (GitHub's digest; bytes not downloaded)"
H9, M9 = HEAD9F[:8], M9F[:8]
# (run, head, full merge, short merge, base): every Math- pull_request run this document may mention
BINDINGS = ((RUN, C8, M1F, M1, BASE), (RUN2, C2, M2F, M2, BASE), (RUN3, C3, M3F, M3, BASE), (RUN4, C4, M4F, M4, BASE),
            (RUN6, C6, M6F, M6, BASE2), (RUNA, C5A, M5AF, M5A, BASE2), (RUN7, C7, M7F, M7, BASE2), (RUN8, H8, M8F, M8, BASE3F),
            (RUN9, H9, M9F, M9, BASE4F), (RUN10, H10, M10F, M10, BASE4F),
            (RUNP, HP, MPF, MP, LANDING[3][2]), (RUNS_STMT, HP, MPF, MP, LANDING[3][2]), (RUNS_AGG, HP, MPF, MP, LANDING[3][2]))
RUNS = tuple((b[0], b[3]) for b in BINDINGS)
for run, _, mf, _, base in BINDINGS:
    assert run.isdigit() and len(run) == 11 and re.fullmatch(r"[0-9a-f]{40}", mf) and re.fullmatch(r"[0-9a-f]{40}", base)
assert len({b[0] for b in BINDINGS}) == len(BINDINGS) and not ({b[0] for b in BINDINGS} & (set(PUSH_RUNS) | {MAIN_PR_RUN[0]}))
for _, _, sha, _ in LANDING:
    assert re.fullmatch(r"[0-9a-f]{40}", sha)


def git_show(repo, rev_path):
    return subprocess.run(["git", "-C", str(repo), "show", rev_path], check=True, capture_output=True).stdout


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


def paragraphs(text):
    return re.split(r"\n\s*\n", text)


MENTION_W = 300


def unbound_mentions(doc):
    """Every mention of a BINDINGS run whose PR head or tested merge is not within MENTION_W characters of it (newlines flattened)."""
    flat, out = doc.replace("\n", " "), []
    for run, head, mf, m, _ in BINDINGS:
        for pos in occurrences(flat, run):
            win = flat[max(0, pos - MENTION_W):pos + len(run) + MENTION_W]
            if not (head in win and m in win):
                out.append((run, flat[max(0, pos - 160):pos + 160]))
    return out


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
    # ---------------------------------------------------------------- inputs, with their identities asserted
    if "RS7_REGISTER" in os.environ:
        reg_bytes = Path(os.environ["RS7_REGISTER"]).read_bytes()
    else:
        reg_bytes = git_show(MATH_REPO, f"{CUT}:{REG_PATH}")
    assert len(reg_bytes) == REG_BYTES, len(reg_bytes)
    assert hashlib.sha256(reg_bytes).hexdigest() == REG_SHA256
    assert hashlib.sha1(b"blob %d\0" % len(reg_bytes) + reg_bytes).hexdigest() == REG_BLOB
    reg = json.loads(reg_bytes)
    assert reg["scientific_effect"] == "NONE" and reg["scientific_status_authority"] is False and reg["alignment_status"] == "PENDING_INDEPENDENT_REVIEW"
    base_doc = git_show(MAIN_REPO, f"{BASE_COMMIT}:{DOC_PATH}").decode()
    prev_doc = git_show(MAIN_REPO, f"{PREV_COMMIT}:{DOC_PATH}").decode()
    assert DOC.read_text() == prev_doc, f"restore {DOC} to its {PREV_COMMIT[:8]} bytes first (git checkout {PREV_COMMIT[:8]} -- {DOC_PATH})"
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
    round4 = sorted(n for n in proof_names if module_of(n) in NEW_MODULES)                   # round 4: first receipted by RUN8 (which receipts all 73)
    RECEIPTED73 = RECEIPTED60 | set(round4)
    round5 = sorted(n for n in proof_names if module_of(n) in R5_MODULES)                    # round 5: first receipted by RUN9 (which receipts all 75)
    RECEIPTED75 = RECEIPTED73 | set(round5)
    NT, NPE, NREC, NPEND4, NR5 = len(proof_names), pk[PROOF_PACKAGE], len(RECEIPTED75), len(round4), len(round5)   # NPEND4: the round-4 count (name kept from render_spec8)
    NMAIN = 59
    # invariants the prose relies on (what must not have moved) and consistency of the computed numbers
    assert c["claims"] == 171, c["claims"]
    assert c["fp"] == Counter({"none": 138, "specified": 33}), c["fp"]          # no claim-level label moved
    assert c["stmt_fp"]["specified"] == 197 and c["stmt_fp"]["proved"] == 72 + NPE and c["statements"] == 197 + 72 + NPE
    assert (c["statements"], c["stmt_fp"]["proved"], NPE, NT) == (354, 157, 85, 75), (c["statements"], c["stmt_fp"]["proved"], NPE, NT)   # coordination note main#358 6118106399
    assert "kernel-checked" not in c["stmt_fp"] and "proved" not in c["fp"]
    assert c["spec_distinct"] == 150 and c["spec_kind"] == Counter({"def_prop": 141, "structure": 9})
    assert c["spec_module"] == Counter({"P15": 58, "Side24": 28, "RN": 28, "Lifetime": 24, "Field": 12}), c["spec_module"]
    assert pk[PACKET] == 197 and pk[MAIN_SIDE24] == 31 and pk[MAIN_P15] == 41, pk
    assert len(main_names) == NMAIN == c["main_targets"]
    assert RECEIPTED52 <= proof_names and len(T3) == 8 and not (RECEIPTED52 & T3) and len(RECEIPTED60) == 60, (len(T3), len(RECEIPTED60))
    assert len(RECEIPTED73) == NT4 and NPEND4 == 13 and NR5 == 2 and NREC == NT == NT4 + NR5 and NPE == 83 + 2, (NPEND4, NR5, NREC, NT, NPE)
    assert set(round5) == proof_names - RECEIPTED73 and RECEIPTED75 == proof_names, sorted(RECEIPTED75 ^ proof_names)   # every theorem of the packet has a hosted receipt
    assert Counter(module_of(n) for n in round4) == NEW_MODULE_COUNTS, Counter(module_of(n) for n in round4)
    assert Counter(module_of(n) for n in round5) == Counter({m: 1 for m in R5_MODULES}), round5
    assert {n.split(".")[-1] for n in round5} == {"referenceCovarianceLowerBound", "coneMomentD2Identity"}, round5
    assert sum(1 for s in proof if s["declaration"] in round5) == 2                           # each round-5 theorem listed once (row side24-coefficient)
    covered = set()
    for r in rows:
        anchors = {s["anchor"] for s in r["statements"] if s["package"] == PROOF_PACKAGE}
        covered |= {s["declaration"] for s in r["statements"] if s["package"] == PACKET and s["kind"] == "def_prop" and s["anchor"] in anchors}
    NP = len(covered)
    assert NT == NP + 1 and NP == 74 and NP == NP4 + NR5, (NT, NP)   # TriangleNeedsThree proved twice, nothing else
    assert 141 - NP == 67
    covered_short = {n.split(".", 3)[-1] for n in covered}
    assert c["idonly"] == 82 and c["sources"] == 23
    assert c["kind"] == Counter({"landing_claim": 10, "graph_node": 33, "open_obligation": 2, "proof_index_result": 95,
                                 "existing_lean_package": 7, "main_experiment_proof": 12, "exact_certificate": 12}), c["kind"]
    mods = Counter(module_of(n) for n in proof_names)
    MOD_ORDER = ("P15Elementary", "P15Structure", "P15Hazard", "P15Covers", "P15Budget", "P15Interpolation", "Side24Elementary",
                 "Side24Analysis", "Side24Interfaces", "Side24Covariance", "Side24ConeMoment", "RNProbability", "RNCluster")
    assert set(mods) == set(MOD_ORDER) and mods["P15Interpolation"] == 8 and sum(mods.values()) == NT, mods
    mods_txt = ", ".join(f"`{m}` {mods[m]}" for m in MOD_ORDER)
    new_mods_txt = ", ".join(f"`{m}` {mods[m]}" for m in NEW_MODULES)
    r5_mods_txt = ", ".join(f"`{m}` {mods[m]}" for m in R5_MODULES)
    proved_interp = [p for p in INTERP if p in covered_short]
    f3 = "RhoStarEnclosure" in covered_short
    f2 = "FullTransformedPriceBudget" in covered_short
    assert f2 and f3 and proved_interp == INTERP
    new_props = sorted(covered_short & (set(INTERP) | {"RhoStarEnclosure"}))
    assert len(new_props) == 8 == len(T3), new_props
    # the theorems of rounds 4 and 5 and the Props they prove (the Prop is the registered local statement with the theorem's anchor)
    prop_of = {}
    for r in rows:
        for s in r["statements"]:
            if s["package"] == PROOF_PACKAGE and (s["declaration"] in round4 or s["declaration"] in round5):
                locs = [l["declaration"] for l in r["statements"] if l["package"] == PACKET and l["kind"] == "def_prop" and l["anchor"] == s["anchor"]]
                assert len(locs) == 1, (s["declaration"], locs)
                assert prop_of.setdefault(s["declaration"], locs[0]) == locs[0]
    round4_ordered = []                                     # first occurrence in the register's statement entries = the packet's inventory order
    for s in proof:
        if s["declaration"] in round4 and s["declaration"] not in round4_ordered:
            round4_ordered.append(s["declaration"])
    assert sorted(round4_ordered) == round4
    new13_props = {m: [] for m in NEW_MODULES}
    for t in round4_ordered:
        new13_props[module_of(t)].append(prop_of[t].split(".", 3)[-1])
    NEW13_SHORT = {p for ps in new13_props.values() for p in ps}
    assert len(NEW13_SHORT) == 13 and NEW13_SHORT <= covered_short
    assert new13_props["Side24Interfaces"] == ["ConeMomentD1Identity", "TraceMoments", "RayleighSquareLaw"], new13_props
    assert new13_props["RNCluster"] == ["RareClusterPoissonObstructionO"], new13_props
    assert set(new13_props["Side24Analysis"]) == {"ThetaSummable", "DerivativePairingBound", "DerivativeBoundAtLeastOne", "DerivativeBoundAtZero",
                                                  "ReferenceOddAndPinBlocks", "ReferencePinDensityProduct", "IntegrandRatioBounds",
                                                  "LatticeImageSumBound", "DensityComparison"}, new13_props
    new13_txt = "; ".join(f"`{m}` ({len(new13_props[m])}: " + ", ".join(f"`{p}`" for p in new13_props[m]) + ")" for m in NEW_MODULES)
    NEW2_SHORT = {prop_of[t].split(".", 3)[-1] for t in round5}
    assert NEW2_SHORT == set(R5_PROPS.values()) and NEW2_SHORT <= covered_short and not (NEW2_SHORT & NEW13_SHORT)
    for t in round5:
        assert R5_PROPS[module_of(t)] == prop_of[t].split(".", 3)[-1] and prop_of[t].startswith("UniversalLaw.Spec.Side24."), (t, prop_of[t])
    new2_txt = " and ".join(f"`{m}` (1: `{R5_PROPS[m]}`)" for m in R5_MODULES)
    # Side24 Props still without a theorem (computed, then checked against what the prose names)
    side24_props = {s["declaration"].split(".", 3)[-1] for s in stmts if s["package"] == PACKET and s["kind"] == "def_prop" and s["module"].endswith("Side24.lean")}
    side24_left = sorted(side24_props - covered_short)
    side24_left_targets = [p for p in side24_left if p not in PREDICATES]
    assert set(side24_left_targets) == {"ImageBound", "CovarianceOrdering", "EntrywiseCovarianceBound", "CoefficientEnclosure", "ReferenceComparison"}, side24_left
    assert set(side24_left) - set(side24_left_targets) == {"IsUnitTuple", "NegDefCone"}, side24_left
    # the cert row's two pre-round-3 fields, which the column key says are stale (a register repair would make this fail: then drop the note)
    cert = next(r for r in rows if r["claim_id"] == "cert:math-full_price-full_price.py")
    assert cert["next_step"].startswith("Prove RhoStarEnclosure in the Math- Mathlib lane") and "(RhoStarEnclosure, specified only)" in cert["not_established"]
    assert "UniversalLaw.Proofs.P15Interpolation.rhoStarEnclosure" in T3 and "RhoStarEnclosure" in covered_short
    pb = next(r for r in rows if r["claim_id"] == "p15-price-boundary")     # the register wording the §5.1 and §6 sentences mirror
    assert pb["formal_progress"] == "specified" and "the independent reviewer's alignment question (PENDING_INDEPENDENT_REVIEW)" in pb["not_established"]
    assert "any claim-level label beyond none/specified is a register-schema decision for the operator after that review, which no receipt produces" in pb["next_step"]
    assert "This is the one row of the register where a theorem proves the Prop registered as stating the claim itself" in pb["not_established"]
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
    covered_before = covered - {d for d in covered if d.split(".", 3)[-1] in (NEW13_SHORT | NEW2_SHORT)}   # after round 3
    uncovered_before = all_props - covered_before
    assert len(uncovered_before) == 82 and triaged <= uncovered_before, len(uncovered_before)
    not_triaged = sorted(d.split(".", 3)[-1] for d in uncovered_before - triaged)
    assert len(not_triaged) == 8 and set(not_triaged) <= PREDICATES and all(d.split(".")[2] == "P15" for d in uncovered_before - triaged), not_triaged
    now = [tshort(r) for r in by["PROVABLE_NOW"]]
    assert all(n in NEW13_SHORT for n in now), now
    work_done = [tshort(r) for r in by["PROVABLE_WITH_WORK"] if tshort(r) in NEW13_SHORT]
    work_left = [(tshort(r), r["effort"]) for r in by["PROVABLE_WITH_WORK"] if tshort(r) not in NEW13_SHORT]   # left after round 4
    assert set(now) | set(work_done) == NEW13_SHORT and len(work_done) == 3 and len(work_left) == 2 and all(e == "weeks" for _, e in work_left), (work_done, work_left)
    assert sorted(w for w, _ in work_left) == ["ConeMomentD2Identity", "ReferenceCovarianceLowerBound"], work_left
    assert {w for w, _ in work_left} == NEW2_SHORT                                             # round 5 proved exactly the two left after round 4
    mm_side24 = sorted(tshort(r) for r in by["BLOCKED_MISSING_MATHLIB"] if r["module"] == "Side24.lean")
    mm_life = sorted(tshort(r) for r in by["BLOCKED_MISSING_MATHLIB"] if r["module"] == "Lifetime.lean")
    assert mm_side24 == ["CovarianceOrdering", "EntrywiseCovarianceBound", "ImageBound"] and len(mm_life) == 4, (mm_side24, mm_life)
    assert mm_life == ["ParentCompactWindowDensityB", "ParentUnrestrictedLeadingDensityC", "PlanarDensityGapE11", "RemainderBoundedR1"], mm_life
    assert all(r["effort"] == "weeks" for r in by["BLOCKED_MISSING_MATHLIB"])
    assert all("iteratedFDeriv_tsum" in " ".join(r["mathlib"]) for r in by["BLOCKED_MISSING_MATHLIB"] if r["module"] == "Side24.lean")
    bi = Counter(r["module"] for r in by["BLOCKED_INTERFACE"])
    assert bi == Counter({"RN.lean": 19, "Lifetime.lean": 3, "Side24.lean": 2}), bi
    OTHER_MECHANISM = {"FixedRemoteMeanMeasureA2", "FourierCutoffPlanarSecondFactorialF4", "ParentUniformSelectionA", "PlanarElderLowerE2",
                       "PlanarTwoSidedRateE4", "CoefficientEnclosure", "ReferenceComparison"}   # Lambda := 0, rare cluster, p == 0 / p == 1, IsEmpty
    dirac_rows = [r for r in by["BLOCKED_INTERFACE"] if r["declaration"].split(".")[-1] not in OTHER_MECHANISM]
    assert all("Dirac" in r["finding"] + r["route"] for r in dirac_rows), [r["declaration"] for r in dirac_rows if "Dirac" not in r["finding"] + r["route"]]
    n_dirac = len(dirac_rows)
    assert n_dirac == 17 and n_dirac + len(OTHER_MECHANISM) == len(by["BLOCKED_INTERFACE"]), n_dirac
    assert all("IsEmpty PeriodicCoefficientInterface" in r["finding"] for r in by["BLOCKED_INTERFACE"] if r["module"] == "Side24.lean")
    assert all(("p ≡ 0" in r["finding"] or "p ≡ 1" in r["finding"]) for r in by["BLOCKED_INTERFACE"] if r["module"] == "Lifetime.lean")
    pv = Counter(r["module"] for r in by["PREDICATE_VOCABULARY"])
    assert pv == Counter({"Lifetime.lean": 14, "Field.lean": 10, "Side24.lean": 2, "RN.lean": 2}), pv
    assert set(side24_left_targets) == set(mm_side24) | {tshort(r) for r in by["BLOCKED_INTERFACE"] if r["module"] == "Side24.lean"}
    rn_left_targets = sorted(tshort(r) for r in by["BLOCKED_INTERFACE"] if r["module"] == "RN.lean")
    rn_left_preds = sorted(tshort(r) for r in by["PREDICATE_VOCABULARY"] if r["module"] == "RN.lean")
    assert set(rn_left) == set(rn_left_targets) | set(rn_left_preds) and len(rn_left_targets) == 19 and len(rn_left_preds) == 2, (rn_left, rn_left_targets, rn_left_preds)
    n_triage_rows = sum(1 for r in rows if any(v in r["next_step"] for v in VERDICTS))
    assert n_triage_rows == 30, n_triage_rows

    doc = base_doc

    # ---------------------------------------------------------------- header date and §1 bullet 4
    doc = replace_once(doc, "`session_01Cz7WZybv8znP64SpPj6sWY`, 2026-10-10), same GitHub",
                       "`session_01Cz7WZybv8znP64SpPj6sWY`, 2026-10-10, updated 2026-10-11 and 2026-10-12), same GitHub")
    # §1: four items follow since render_spec5 added bullet 4 (the base says 'Three'); bullet 3's headline predates every review
    doc = replace_once(doc, "Three things follow, and the claim table below is read with them in mind:",
                       "Four things follow, and the claim table below is read with them in mind:")
    doc = replace_once(doc, "3. **Nothing here is reviewed.** Every Lean object", "3. **Nothing here is alignment-reviewed.** Every Lean object")
    doc = replace_once(doc, "   existing validators (`tools/formal_gate_check.py --alignment`, Math- `formal/gate.py --alignment`).\n",
                       "   existing validators (`tools/formal_gate_check.py --alignment`, Math- `formal/gate.py --alignment`); the bounded\n"
                       "   same-account nonauthor reviews and the engineering integrations recorded in §3, §5.1 and §8 are technical records at\n"
                       "   organizational-independence credit 0, not alignment records.\n"
                       f"4. **A proved Prop is not an accepted claim.** The proof packet (§3.7) proves {NP} of the 141 registered\n"
                       f"   Props exactly as stated, and all {NT} of its theorems carry a hosted kernel receipt: {len(RECEIPTED60)} as before (the {len(T3)} of module\n"
                       f"   `P15Interpolation`, PR #{NEWPR}, by run {RUN6}: PR head `{C6}`, actual tested merge `{M6}…`; all {len(RECEIPTED60)} corroborated on\n"
                       f"   PR #{NEWPR} head `{C7}` by run {RUN7}, actual tested merge `{M7}…`), the {NPEND4} of the round-4 modules\n"
                       f"   `Side24Analysis`, `Side24Interfaces` and `RNCluster` (branch `{BRANCH4}`, {R4PR_TXT}) by run {RUN8} on\n"
                       f"   that PR: PR head `{H8}`, actual tested merge `{M8}…` with base `{BASE3}…` (the head of #{NEWPR}), receipt theorem_count {NT4}\n"
                       f"   (their first receipt; the same run receipts all {NT4} theorems of that head), and the {NR5} of the round-5 modules `Side24Covariance`\n"
                       f"   and `Side24ConeMoment` (branch `{BRANCH5}`, {R5PR_TXT}) by run {RUN9} on that PR: PR head `{H9}`,\n"
                       f"   actual tested merge `{M9}…` with base `{BASE4}…` (then the head of #{ARGS.round4_pr}), receipt theorem_count {NT} (their first receipt; the\n"
                       f"   same run receipts all {NT} theorems of that head). The whole packet is on Math- main since 2026-10-12 (final cut `{CUT[:8]}`, §3);\n"
                       "   landing there is an engineering integration, not an acceptance and not an alignment, and it moves no label. Each receipted theorem is L5\n"
                       "   evidence for that Prop, for the formal objects it is stated about; a theorem without a hosted receipt\n"
                       "   is `proved` as a packet label and not kernel evidence until a receipt names it; and the claim beneath it stays\n"
                       "   `specified` or `none` (the register admits no other claim-level label) until an independent alignment\n"
                       "   review and the controlling acceptance machinery say otherwise.\n")

    # ---------------------------------------------------------------- §2.3 timing sentence (round 6, unchanged)
    doc = replace_once(doc, "9 s `P15`, each exiting 0 with zero warnings; the injected-type-error control exits 1.\n",
                       "9 s `P15`, each exiting 0 with zero warnings; the injected-type-error control exits 1. The proof packet\n"
                       f"(§3.7) rebuilt its five P15 modules in 76 s after the change at `{C2}` (`lake build`, 8,936 jobs, the rest\n"
                       "up to date); its `leanchecker` replay of the Mathlib-importing environment needs about 14 GB and was\n"
                       "memcg-killed in this 15 GB sandbox, so that receipt comes from the hosted run only.\n")

    # ---------------------------------------------------------------- §3 rows 3.1, 3.6 and 3.7, gate paragraph, landing paragraph
    doc = replace_once(doc, "28 P15 staged in this programme's uncommitted worktree, on no commit",
                       "28 P15 at commit `91cffbb5` of main PR #358, not yet on main")
    doc = replace_once(doc, "Math- `specifications/lean_statements_20261010/` | Lean + Mathlib, statement-only",
                       f"Math- `specifications/lean_statements_20261010/` (PR #419, on Math- main since its merge `{LANDING[0][2][:8]}`, 2026-10-12) | Lean + Mathlib, statement-only")
    i = doc.index("| 3.6 | **Lean statement register**")
    j = doc.index("\n", i)
    row37 = ("\n| 3.7 | **Lean statement proofs** (this programme) | Math- `specifications/lean_statement_proofs_20261011/` (rounds 1-2 in PR #422, "
             f"round 3 in PR #{NEWPR}, round 4 in {R4PR_TXT} on branch `{BRANCH4}`, round 5 in {R5PR_TXT} on branch `{BRANCH5}`; each PR was stacked on its "
             f"predecessor, #422 on #419, and all five landed on Math- main by normal protected merge commits on 2026-10-12, final cut `{CUT[:8]}`: see the "
             f"landing paragraph below this table) | "
             f"Lean + Mathlib, path dependency on 3.6 | {NT} theorems whose types are exactly {NP} registered Props of 3.6 "
             f"({mods_txt}); axioms exactly `propext`, `Classical.choice`, `Quot.sound` | `python3 check.py` / `--execute` | "
             f"`.lake/evidence/receipt.json` with `formalization_status: \"kernel-checked\"` (source mode says `proved`); hosted run {RUN2} "
             f"receipted the 52 theorems of PR head `{C2}` (actual tested merge `{M2}…` with base `{BASE[:8]}…`, the synthetic merge a pull_request run "
             f"checks out, same tree as the head, recorded as `checked_commit`), run {RUN} the first 38 (PR head `{C8}`; actual tested merge `{M1}…`), "
             f"run {RUN3} corroborates (PR head `{C3}`; actual tested merge `{M3}…`), run {RUN4} the same 52 on PR head `{C4}` (actual tested merge `{M4}…`); "
             f"run {RUN6} receipted all 60 on PR #{NEWPR} head `{C6}` (actual tested merge `{M6}…` with base `{BASE2[:8]}…`, then the head of #422; "
             f"`leanchecker` PASS; artifact `lean-specification-proofs-{RUN6}-1`) after run {RUNA} on head `{C5A}` (actual tested merge `{M5A}…`, same base) "
             f"first receipted the {len(T3)} `P15Interpolation` theorems; run {RUN7} corroborates the same 60 on PR #{NEWPR} head `{C7}` (actual tested merge "
             f"`{M7}…`, same base; theorem_count 60; artifact `lean-specification-proofs-{RUN7}-1`); run {RUN8} receipted all {NT4} on {R4PR_TXT} head `{H8}` "
             f"(actual tested merge `{M8}…` with base `{BASE3}…`, then the head of #{NEWPR}; `leanchecker` PASS; theorem_count {NT4}; artifact "
             f"`lean-specification-proofs-{RUN8}-1`, sha256 `{SHA8[:8]}…`), the first receipt of the {NPEND4} theorems of the round-4 modules ({new_mods_txt}; "
             f"branch `{BRANCH4}`, then stacked on #{NEWPR}); run {RUN9} receipted all {NT} on {R5PR_TXT} head `{H9}` (actual tested merge `{M9}…` with base "
             f"`{BASE4}…`, then the head of #{ARGS.round4_pr}; `leanchecker` PASS; theorem_count {NT}; artifact `lean-specification-proofs-{RUN9}-1`, sha256 `{SHA9[:8]}…`), "
             f"the first receipt of the {NR5} theorems of the round-5 modules ({r5_mods_txt}; branch `{BRANCH5}`, then stacked on #{ARGS.round4_pr}); run {RUN10} "
             f"corroborates the same {NT} on {R5PR_TXT} head `{H10}` (actual tested merge `{M10}…`, same base; theorem_count {NT}); at the final cut the "
             f"integrator's pre-merge run {RUNP} receipted {NT} theorems over {NP} Props on {R5PR_TXT}'s head at merge `{HP}`, the integration candidate (actual tested merge `{MP}…`, "
             f"whose tree is the landing tree of `{CUT[:8]}`) |")
    doc = doc[:j] + row37 + doc[j:]
    landing_par = (
        f"**Landing on Math- main (2026-10-12).** The statement register (3.6) and the proof packet (3.7) landed on Math- main through five normal\n"
        f"protected merge commits (no squash, rebase or force push), in stack order (times are {MERGED_AT_NOTE}): "
        + "; ".join(f"#{pr} ({what}) by `{sha}` at {when}" for pr, what, sha, when in LANDING)
        + f", the final cut, whose `REGISTER.json` (blob `{REG_BLOB[:8]}`, {REG_BYTES:,} bytes,\n"
        f"sha256 `{REG_SHA256}`) is the register this document renders. Each merge was made by a distinct\n"
        "OpenAI/Codex engineering integrator under the same project account (organizational-independence credit 0): an engineering integration, not a\n"
        "mathematical acceptance and not an alignment, and it moves no claim, Prop or theorem label. For the final cut the integrator's pre-merge hosted\n"
        f"runs (event pull_request) ran on PR #439's head at merge, `{HEADPF}` (the integration candidate: a merge of the then-current\n"
        f"Math- main into the PR branch), actual tested merge `{MPF}`: the proof run {RUNP} (attempt 1; receipt `kernel-checked`,\n"
        f"{NT} theorems over {NP} Props, `leanchecker` PASS), the statement run {RUNS_STMT} (PR head `{HP}`, actual tested merge `{MP}…`; receipt\n"
        f"`specified`, 150 registered declarations) and the required-aggregate run {RUNS_AGG} (PR head `{HP}`, actual tested merge `{MP}…`). The tested merge's tree\n"
        f"`{TREE_CUT}` equals the landing tree of `{CUT[:8]}` (integrator review {INTEG_REVIEW} and release comment\n"
        f"{RELEASE_COMMENT} on Math- PR #439). The landing push run {RUN_LANDPUSH} (event push: its checked commit is the pushed merge `{CUT[:8]}`\n"
        "itself, with no PR head and no synthetic merge) ran only `downstream-replay`, `formal / formal-evidence` and `math-downstream-gates`; it did not\n"
        "run the Lean statement or proof packets and is not a receipt for them. The 28 P15 targets of main PR #358 (3.1) are still not on main.\n")
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
                       "a module may import a sibling proof module of the packet.\n\n" + landing_par)

    # ---------------------------------------------------------------- §4 table
    R.short_decl = lambda st: short_decl(st, RECEIPTED75)
    R.render_statements = make_render_statements(RECEIPTED75)
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
                         "theorem of the Math- proof packet `specifications/lean_statement_proofs_20261011` (Math- PRs #422, "
                         f"#{NEWPR}, #{ARGS.round4_pr} and #{ARGS.round5_pr}, landed on Math- main, final cut `{CUT[:8]}`) whose type is "
                         "exactly one of the registered Props of the row, label `proved` (52 kernel-checked by the hosted run "
                         f"{RUN2}: PR head `{C2}`, actual tested merge `{M2}…` with base `{BASE[:8]}…`, same tree as the head; run {RUN}, PR head `{C8}`, "
                         f"actual tested merge `{M1}…`, receipted the first 38; the {len(T3)} theorems of `P15Interpolation` (PR #{NEWPR}) by run {RUN6}, PR head `{C6}`, "
                         f"actual tested merge `{M6}…` with base `{BASE2[:8]}…`, then the head of #422; all 60 corroborated on PR #{NEWPR} head `{C7}` by run {RUN7}, "
                         f"actual tested merge `{M7}…`; the {NPEND4} theorems of the round-4 modules `Side24Analysis`, `Side24Interfaces`, `RNCluster` "
                         f"(branch `{BRANCH4}`, {R4PR_TXT}), added after head `{C7}`, by run {RUN8}, PR head `{H8}`, actual tested merge `{M8}…` with base "
                         f"`{BASE3}…`, then the head of #{NEWPR}, which receipts all {NT4} theorems of that head; the {NR5} theorems of the round-5 modules "
                         f"`Side24Covariance`, `Side24ConeMoment` (branch `{BRANCH5}`, {R5PR_TXT}), added after head `{BASE4}`, by run {RUN9}, PR head `{H9}`, "
                         f"actual tested merge `{M9}…` with base `{BASE4}…`, then the head of #{ARGS.round4_pr}, which receipts all {NT} theorems of that head); a "
                         "proof-packet theorem is evidence about its Prop for the formal objects, never about the claim. Two register fields of row "
                         "`cert:math-full_price-full_price.py` are transcribed as the register has them although they predate round 3: its `not_established` "
                         "calls `RhoStarEnclosure` specified only and its `next_step` (shown abridged) names proving it, while `P15Interpolation.rhoStarEnclosure` "
                         f"has been a theorem of exactly that Prop since round 3, kernel-checked by run {RUN6} (PR head `{C6}`, actual tested merge `{M6}…`); "
                         "repairing those fields is a Math- register change, not made here, and it would move no label.")
    # fe99ca31 hand fix 3, now rendered: the abridged OA-C6 next-step cell ends with its receipt instead of an ellipsis
    table = replace_once(table, "the theorem of this row (RNCluster.rareClusterPoissonObstructionO, added after PR head 12e072d6) has a … |",
                         "the theorem of this row (RNCluster.rareClusterPoissonObstructionO, added after PR head 12e072d6) has a hosted receipt: "
                         f"run {RUN8}, PR head `{H8}`, actual tested merge `{M8}…` with base `{BASE3}…` |")
    doc = splice(doc, "\nColumn key. **Claim**", "\n## 5. Coverage summary\n", "\n" + table + "\n")

    # ---------------------------------------------------------------- §5.1 counts table and prose
    tables = R.coverage_tables(reg, c)
    for tok in ("{{KIND_TABLE}}", "{{SPECIFIED_LIST}}"):                  # rendered at 91cffbb5; still exactly what the register gives
        assert doc.count(tables[tok]) == 1, tok
    proved_n = c["stmt_fp"]["proved"]
    counts_tbl = replace_once(tables["{{COUNTS_TABLE}}"], "Statement entries (256)", f"Statement entries ({c['statements']})")
    counts_tbl = replace_once(counts_tbl, "statements of other packages (the `main` theorems listed beneath claims); no claim",
                              f"statements of other packages, one entry per claim a theorem is listed beneath, so {proved_n} entries are not {proved_n} "
                              f"distinct theorems: the {NMAIN} `main` arithmetic theorems, {pk[MAIN_SIDE24] + pk[MAIN_P15]} entries ({pk[MAIN_P15]} of them for the "
                              f"28 P15 targets at commit `91cffbb5` of main PR #358, not yet on main), and the {NT} proof-packet theorems over {NP} Props, "
                              f"{NPE} entries, every one of the {NT} with a hosted receipt; no claim")
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
                   f", and, in the round-3 module `P15Interpolation` (PR #{NEWPR}; run {RUN6}: PR head `{C6}`, actual tested merge `{M6}…`), the hazard interpolation chain "
                   + ", ".join(f"`{p}`" for p in proved_interp)
                   + " ((F6)-(F8), (F12) and Theorem F (F2) itself, through the coordinatewise concavity in place of the undisplayed (F4)-(F5))"
                   " and the 20-digit enclosure (F3) `RhoStarEnclosure` (a rational certificate with explicit series remainders, not the source's Fraction procedure)"
                   f", and, in the round-4 modules `Side24Analysis`, `Side24Interfaces` and `RNCluster` (branch `{BRANCH4}`, {R4PR_TXT}; run {RUN8}: PR head `{H8}`, actual tested merge `{M8}…`),\n"
                   "summability of the formal theta family `n ↦ exp(-‖z + 24n‖²/2)` over `Fin d → ℤ` for every `d` (`ThetaSummable`), the Section 2 derivative\n"
                   "bounds for `q ≤ 6` (`DerivativePairingBound`, `DerivativeBoundAtLeastOne`, `DerivativeBoundAtZero`), the lattice image sum with its bound\n"
                   "`1458 e^{-288}` (`LatticeImageSumBound`), the five reference block identities and the pin density product for the formal matrices\n"
                   "(`ReferenceOddAndPinBlocks`, `ReferencePinDensityProduct`), the integrand-ratio arithmetic for `d = 2, 3` (`IntegrandRatioBounds`), Loewner\n"
                   "determinant and inverse monotonicity for positive-definite real matrices (`DensityComparison`), over `ConeMomentInterface` the `m = 1` cone\n"
                   "moment `4/3`, the trace moments `5/3`, `25/3`, `√(3/8)` and the distribution function `P(R² ≤ a) = ∫₀^a e^{-z/2}/2 dz` (`ConeMomentD1Identity`,\n"
                   "`TraceMoments`, `RayleighSquareLaw`: for every inhabitant, from the law fields alone, constructing none), and Theorem O's inequality for every\n"
                   "family of pmfs satisfying the Prop's own antecedents (H1), (H2) (`RareClusterPoissonObstructionO`)"
                   f", and, in the round-5 modules `Side24Covariance` and `Side24ConeMoment` (branch `{BRANCH5}`, {R5PR_TXT}; run {RUN9}: PR head `{H9}`,\n"
                   f"actual tested merge `{M9}…`), `C_ref ≥ I/3` in the positive-semidefinite order for the formal matrix `Cref` in every orthonormal frame, `d = 2, 3`\n"
                   "(`ReferenceCovarianceLowerBound`), and over `ConeMomentInterface` the `m = 2` cone moment `(1/2)[25/3 - 20/3 + 8 - 8√(3/8)]`\n"
                   "(`ConeMomentD2Identity`: for every inhabitant, from the law field `sxy_law` alone, constructing none)")
    side24_left_txt = ", ".join(f"`{p}`" for p in side24_left_targets)
    new_par = (
        f"The {proved_n} `proved` entries (entries, not distinct theorems: a theorem is listed beneath every claim it serves, and the {proved_n} entries\n"
        f"are {NMAIN + NT} distinct theorems, {NMAIN} of `main` and {NT} of the proof packet) are theorems of two other packages, each listed beneath a claim and never as the claim.\n"
        f"(a) {pk[MAIN_SIDE24] + pk[MAIN_P15]} entries for the {NMAIN} theorems of the `main` core-Lean package `formal/`: 31 SIDE24 targets at origin/main\n"
        "`f6deeba7` (`Ledger` 17, `ConeMoments` 10, `Endpoints` 4), whose kernel evidence is the `formal / formal-evidence`\n"
        "job of `workspace-landing` run 38068951535 (event push to `main`, 2026-10-10, conclusion success; a push run checks out the pushed commit itself,\n"
        "here `f6deeba7`, so it has no PR head and no synthetic merge; the reusable\n"
        "`formal-verification.yml` runs inside that caller, so it does not appear under its own workflow's run list; artifact\n"
        "`formal-verification-38068951535`; the last stand-alone `formal-verification.yml` run is 36356485214, an event-push run whose checked commit is\n"
        "the pushed `d0c0512f` itself, 2026-09-27, and the 31 Lean texts are byte-identical between the two commits), and 28 P15 targets at commit\n"
        "`91cffbb5` of main PR #358, not yet on main (`PriceBoundary` 6, `RealizedCovers` 12, `FullPrice` 10; kernel evidence the\n"
        f"`formal / formal-evidence` job of `workspace-landing` run {MAIN_PR_RUN[0]}, event pull_request on main PR #358: PR head `{MAIN_PR_RUN[1]}`, actual\n"
        f"tested merge `{MAIN_PR_RUN[2]}` (the synthetic merge of that head into main at `{MAIN_PR_RUN[3][:8]}…`, the receipt's `checked_commit`,\n"
        f"as the job's checkout log and receipt record it; its `formal/` subtree is the head's, `{MAIN_PR_RUN[4]}…`, its whole tree is not, main having moved), artifact\n"
        f"`formal-verification-{MAIN_PR_RUN[0]}`; listed on every P15 row whose source they anchor to, hence {pk[MAIN_P15]} entries); each of the {NMAIN} is a finite integer or\n"
        "cross-multiplied rational identity (`decide`, one conjunct by core `Nat.pow_pos`), an arithmetic skeleton, so none\n"
        f"states a research claim. (b) {NPE} entries for the {NT} theorems of the Math- proof packet\n"
        f"`specifications/lean_statement_proofs_20261011` (rounds 1-2 in PR #422, receipted at head `{C4}`; round 3 in PR #{NEWPR}, head `{C7}`; round 4 in\n"
        f"{R4PR_TXT} on the branch `{BRANCH4}`, head `{H8}`; round 5 in {R5PR_TXT} on the branch `{BRANCH5}`, heads `{H9}` and `{H10}`; all\n"
        f"landed on Math- main on 2026-10-12, final cut `{CUT[:8]}`, §3), each typed exactly as one of {NP} registered\n"
        f"Props (`TriangleNeedsThree` twice): 52 of them kernel-checked by the hosted `lean-specification-proofs` run {RUN2} on PR\n"
        f"head `{C2}`, actual tested merge `{M2}…` (receipt `kernel-checked`, `leanchecker` PASS, axioms within the three allowed; artifact\n"
        f"`lean-specification-proofs-{RUN2}-1`; the 38 theorems of head `{C8}` were first receipted by run {RUN}, actual tested merge `{M1}…`, artifact\n"
        f"`lean-specification-proofs-{RUN}-1`; run {RUN3} corroborates on head `{C3}`, actual tested merge `{M3}…`, and run {RUN4} on head `{C4}`, actual\n"
        f"tested merge `{M4}…`, theorem_count 52). A pull_request run checks out the synthetic\n"
        f"merge of the PR head into its base, so each receipt's `checked_commit` is that merge commit, not the head. Bound identities:\n"
        f"run {RUN2}, PR head `{C2}`, actual tested merge `{M2F}` with base `{BASE}`; run {RUN}, PR head `{C8}`, actual tested merge\n"
        f"`{M1F}`, same base; run {RUN3}, PR head `{C3}`, actual tested merge `{M3F}`, same base; run {RUN4}, PR head `{C4}`,\n"
        f"actual tested merge `{M4F}`, same base. Because the branch\n"
        "contains the base, each merge's tree equals its head's tree (verified by comparing the tree hashes of the merge and the head), so the\n"
        f"receipted bytes are the head's bytes. The {len(T3)} theorems of the round-3 module `P15Interpolation` ({new_props_txt}) were\n"
        f"added after head `{C4}` in PR #{NEWPR} and are kernel-checked by the hosted run {RUN6} on that PR (PR head `{C6}`, actual tested merge\n"
        f"`{M6F}` with base `{BASE2}`, then the head of PR #422, on which #{NEWPR} was stacked; receipt `kernel-checked`, `leanchecker` PASS,\n"
        f"theorem_count 60, worktree CLEAN, axioms within the three allowed; artifact `lean-specification-proofs-{RUN6}-1` (PR head `{C6}`, actual tested\n"
        f"merge `{M6}…`), sha256 `{SHA6}`); they were first receipted by run {RUNA}\n"
        f"on PR head `{C5A}`, actual tested merge `{M5AF}`, same base (same Lean bytes; artifact `lean-specification-proofs-{RUNA}-1`, sha256 `{SHA5A}`);\n"
        f"both merges have the tree of their head (the base is an ancestor of the head). All 60 theorems of PR #{NEWPR} head `{C7}` are corroborated by run {RUN7}\n"
        f"(actual tested merge `{M7F}` with base `{BASE2}`; receipt theorem_count 60; artifact `lean-specification-proofs-{RUN7}-1`, sha256 `{SHA7}`).\n"
        f"The {NPEND4} theorems of the round-4 modules {new13_txt} were added after head `{C7}` on the branch `{BRANCH4}`\n"
        f"(commit `{HEAD4}`, then stacked on #{NEWPR}; {R4PR_TXT}) and are kernel-checked by the hosted `lean-specification-proofs.yml` run {RUN8} on that PR\n"
        f"(PR head `{HEAD8F}`, actual tested merge `{M8F}` with base `{BASE3F}`, then the head of PR #{NEWPR}, on which #{ARGS.round4_pr}\n"
        f"was stacked; event pull_request; job proof-evidence, success; receipt `kernel-checked`, `leanchecker` PASS, theorem_count {NT4}, covered_props {NP4}, every control\n"
        f"classified as intended (negative controls refused, positive controls accepted), worktree CLEAN, axioms within the three allowed; artifact\n"
        f"`lean-specification-proofs-{RUN8}-1` (PR head `{H8}`, actual tested merge `{M8}…`), 29 files, 43937 bytes, sha256 `{SHA8}` as GitHub reports it,\n"
        f"the artifact bytes not having been downloaded here); the merge has the tree of its head (`{TREE8[:8]}…`), so the receipted bytes are the head's. That run is their first receipt and receipts all {NT4} theorems of that head;\n"
        "the round-4 commit's local `lake build` and `#print axioms` on a worktree remain evidence about no commit.\n"
        f"The {NR5} theorems of the round-5 modules {new2_txt} were added after head `{BASE4}` of {R4PR_TXT} (its receipt-binding\n"
        f"commit, the base of the first-receipt and corroboration runs bound next) on the branch `{BRANCH5}` ({R5PR_TXT}, then stacked on #{ARGS.round4_pr}) and are\n"
        f"kernel-checked by the hosted `lean-specification-proofs.yml` run {RUN9} on that PR (PR head `{HEAD9F}`, actual tested merge\n"
        f"`{M9F}` with base `{BASE4F}`, then the head of PR #{ARGS.round4_pr}, on which #{ARGS.round5_pr} was stacked; attempt 1; event\n"
        f"pull_request; job proof-evidence {JOB9}, success; receipt `kernel-checked`, `leanchecker` PASS, theorem_count {NT}, covered_props {NP}, every control\n"
        f"classified as intended (negative controls refused, positive controls accepted), worktree CLEAN, axioms within the three allowed; artifact\n"
        f"`lean-specification-proofs-{RUN9}-1` (PR head `{H9}`, actual tested merge `{M9}…`), sha256 `{SHA9}` as GitHub reports it); the merge has the tree of its head. That run is their first receipt and receipts all {NT} theorems\n"
        f"of that head. All {NT} theorems of the receipt-binding successor head `{HEAD10F}` are corroborated by run {RUN10} (actual\n"
        f"tested merge `{M10F}` with base `{BASE4F}`; receipt theorem_count {NT}). Two bounded nonauthor\n"
        f"reviews on {R5PR_TXT}, both by OpenAI/Codex agents under the same project account (organizational-independence credit 0 each), are technical records\n"
        f"only: review {REV1}, a source/type comparison of the two declarations with their registered Props at head `{H9}` (PASS; no local Lean run), and\n"
        f"review {REV2}, a readback of the receipt-binding delta `{H9}..{H10}` (PASS); neither is an alignment review, an approval or an acceptance, and the\n"
        "alignment gate stays open (`PENDING_INDEPENDENT_REVIEW`). The packet's `check.py` in source mode\n"
        f"inventories the {NT} theorems over {NP} distinct registered Props and labels them\n"
        "`proved`, the label of a committed proof package; `kernel-checked` is written by no local run. A proof-packet theorem is evidence about its Prop for the formal objects (the `RealizedData` record, the formal\n"
        f"downset, `coverCost` as an `sInf`, `mu`): {what_proved} are proved *as the Props state them*. Every claim they sit under stays `specified` or\n"
        "`none`: the register admits only those two labels at claim level, alignment of each Prop with its source paragraph\n"
        "is `PENDING_INDEPENDENT_REVIEW`, and the one Prop with a theorem that the register names as stating its claim itself (`DemandOneRefutation`, row\n"
        f"`p15-price-boundary`; whether it says what the source paragraph says is the independent reviewer's alignment question) is recorded for the operator\n"
        f"and not relabelled. What has no theorem ({141 - NP} of the 141 registered Props), and why, is tabulated\n"
        f"per group in the proof packet's README (parametrised predicates; the SIDE24 Props {side24_left_txt}; every\n"
        f"RN Prop beyond (N1)-(N5) and Theorem O ({len(rn_left_targets)} theorem-shaped Props and the predicates "
        + ", ".join(f"`{p}`" for p in rn_left_preds) + ") and every Lifetime and Field Prop).\n"
        "\n"
        f"**Round-4 triage** (workflow `{TRIAGE_WF}`, 2026-10-11; same account, organizational-independence credit 0; it moves no label, and its\n"
        "probes are scratch files, not receipts and not in any packet). Of the 82 registered Props without a theorem after round 3, the 8 P15\n"
        f"parametrised predicates were already classified by the proof packet's README and the other {len(triage)} were classified one by one:\n"
        f"{counts['PROVABLE_NOW']} PROVABLE_NOW and {counts['PROVABLE_WITH_WORK']} PROVABLE_WITH_WORK, of which the {counts['PROVABLE_NOW']} and {len(work_done)} of the "
        f"{counts['PROVABLE_WITH_WORK']} (" + ", ".join(f"`{p}`" for p in work_done) + f") are the {NPEND4} round-4 theorems above and the other {len(work_left)}\n"
        "(" + ", ".join(f"`{p}`" for p, _ in work_left) + f"; weeks), still open after round 4, are the {NR5} round-5 theorems above; {counts['BLOCKED_MISSING_MATHLIB']} BLOCKED_MISSING_MATHLIB (" + ", ".join(f"`{p}`" for p in mm_side24)
        + ": no local, open-set `iteratedFDeriv_tsum` at Mathlib `d13f23b7`;\n" + ", ".join(f"`{p}`" for p in mm_life) + f": weeks); {counts['BLOCKED_INTERFACE']} BLOCKED_INTERFACE "
        f"({bi['RN.lean']} `RN`, {bi['Lifetime.lean']} `Lifetime`, {bi['Side24.lean']} `Side24`), interface under-determination rather than statement\n"
        f"defects: each Prop is refuted at a concrete inhabitant of its interface (a Dirac law with prescribed extra critical points in {n_dirac} cases, the free\n"
        "kernel field `Λ := 0`, a two-point law with a rare large cluster, the constant `p ≡ 0` or `p ≡ 1`; for `CoefficientEnclosure` and `ReferenceComparison`\n"
        "the universal closure would make `PeriodicCoefficientInterface` empty, a probe-level elaboration), so no theorem of its type can exist until the\n"
        f"interface carries more fields; {counts['PREDICATE_VOCABULARY']} PREDICATE_VOCABULARY ({pv['Lifetime.lean']} `Lifetime`, {pv['Field.lean']} `Field`, "
        f"{pv['Side24.lean']} `Side24`, {pv['RN.lean']} `RN`: parametrised predicates, the vocabulary of the other Props, not targets).\n"
        "The proof packet README's informal reasons for its groups 'Side24 derivative bounds', 'Side24 lattice sums', 'Side24 covariance and density' and\n"
        "'Side24 interface statements' were found not load-bearing and are superseded by the theorems; the per-Prop verdicts, efforts and Mathlib references\n"
        f"are recorded in the register's `next_step` fields ({n_triage_rows} rows). Nothing here is a review: every verdict is author-side."
    )
    doc = doc[:i] + new_par + doc[j:]
    doc = replace_once(doc, "`main formal/` (31 committed theorems listed, 28 staged P15 theorems not listed, as above)",
                       "`main formal/` (59 theorems, all listed as above)")
    # §5.2: Theorem O now has a theorem; the row stays `none` for the reason already given
    doc = replace_once(doc, "`OA-C6-RARE-CLUSTER-20260929-v1`, only Theorem O stated;",
                       "`OA-C6-RARE-CLUSTER-20260929-v1`, only Theorem O stated (and, in round 4, proved as the Prop states it and kernel-checked by run "
                       f"{RUN8}: PR head `{H8}`, actual tested merge `{M8}…` with base `{BASE3}…`; the row stays `none` because Theorems C, S and (7.1) are not stated);")
    # §5.2 analytic-coefficient row: the cone-moment theorems of rounds 4 and 5 are about a different interface and leave `D_u` a given field
    doc = replace_once(doc, "(the cone moment `D_u` is a given field);",
                       "(the cone moment `D_u` is a given field; the cone-moment theorems `ConeMomentD1Identity` (round 4) and `ConeMomentD2Identity` (round 5) "
                       "are about the separate `ConeMomentInterface`, hold for every inhabitant of it, construct none and do not identify its `D_1`, `D_2` with "
                       "`D_u`);")

    # ---------------------------------------------------------------- §6 roadmap frame, statuses and Step 6
    doc = replace_once(doc,
                       "is the same for every step: the proof is written in Math- `formal/` under the lane's admission rules (the statement\n"
                       "packet's checker refuses `theorem`); the register then lists",
                       "is the same for every step: the proof is written in a Math- Mathlib proof package (the sibling proof packet\n"
                       "`specifications/lean_statement_proofs_20261011`, on Math- main since 2026-10-12, whose gate requires each theorem's type to be exactly a\n"
                       "registered Prop applied to its own binders; the statement packet's checker refuses `theorem`); the register then lists")
    doc = replace_once(doc, "the claim's own label moves to `proved` only if the theorem proves the claim's display itself, not a sub-identity;",
                       "the claim's own label stays `specified` or `none`, the only claim-level labels the register admits, even where the theorem's type is\n"
                       "the Prop the register names as stating the claim itself (`DemandOneRefutation` for `p15-price-boundary`, the one such row; whether it\n"
                       "says what the source paragraph says is the independent reviewer's alignment question, `PENDING_INDEPENDENT_REVIEW`), and any\n"
                       "claim-level label beyond none/specified is a register-schema decision for the operator after that review, which no receipt produces;")
    # (finding 2) §6 Step 2's outcome, 2026-10-10 roadmap text: a Prop can have a theorem, the claim cannot carry `proved`
    doc = replace_once(doc, "`p15-price-boundary` (disposition `EXACT_COUNTEREXAMPLE`, transcribed) is the first claim whose own display\n"
                       "(`DemandOneRefutation`) could carry `proved`; the register's next-step column already names it as the smallest\n"
                       "complete research statement in the packet.",
                       "`p15-price-boundary` (disposition `EXACT_COUNTEREXAMPLE`, transcribed) is the first claim whose register-named Prop\n"
                       "(`DemandOneRefutation`) could have a theorem of exactly its type (the claim itself stays `specified`: any claim-level label beyond\n"
                       "none/specified is the operator's register-schema decision after the independent alignment review); the register's next-step column\n"
                       "named it, from its first commit on 2026-10-10 until the proof packet's first commit (Math- `00d2ac6e`, 2026-10-11), as the smallest\n"
                       "complete research statement in the packet.")
    f3_status = (f"`RhoStarEnclosure` (F3) is proved as stated in the round-3 module `P15Interpolation` (PR #{NEWPR}; kernel-checked by run {RUN6}: PR head `{C6}`, actual tested merge `{M6}…`):\n"
                 "a rational certificate (series bounds for `e` and for `exp` with explicit remainders, compared by `norm_num`), not the\n"
                 "source's `full_price.py` Fraction procedure, which remains unspecified.")
    round4_tag = (f"round 4 (modules `Side24Analysis`, `Side24Interfaces`; branch `{BRANCH4}`, {R4PR_TXT}; kernel-checked by run {RUN8}: PR head `{H8}`, "
                  f"actual tested merge `{M8}…` with base `{BASE3}…`)")
    step4_r5 = (f" *Status 2026-10-12:* of the Props still open on 2026-10-11, `ReferenceCovarianceLowerBound` (module `Side24Covariance`) and\n"
                f"`ConeMomentD2Identity` (module `Side24ConeMoment`) are proved as the Props state them in round 5 (branch `{BRANCH5}`, {R5PR_TXT};\n"
                f"kernel-checked by run {RUN9}: PR head `{H9}`, actual tested merge `{M9}…` with base `{BASE4}…`): `C_ref ≥ I/3` for the formal matrix in\n"
                "every orthonormal frame, `d = 2, 3` (the odd-order contractions of the even kernel vanish at `0`, the orders two, four and six reuse the round-4\n"
                "helpers, and in the frame `Cref - I/3` is a sum of three positive-semidefinite matrices; main's eigenvalue and minor theorems are not used), and\n"
                "over `ConeMomentInterface` the `m = 2` cone moment for every inhabitant, from the law field `sxy_law` alone (Fubini, the cone identification\n"
                "`NegDefCone (s, x, y) ↔ s < 0 ∧ x² + y² < s²`, polar coordinates onto `ElementaryConeIntegral` and the round-4 trace moments), constructing no\n"
                "inhabitant and not identifying `D_2` with the parent's `D_u`. Still open: `CovarianceOrdering`, `EntrywiseCovarianceBound`, `ImageBound`\n"
                "(BLOCKED_MISSING_MATHLIB) and `CoefficientEnclosure`, `ReferenceComparison` (the empty-interface finding above); `side24-coefficient` stays\n"
                "`specified`.")
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
            f"actual tested merge `{M1}…`). `p15-price-boundary` stays `specified`: any\n"
            "claim-level label beyond none/specified is a register-schema decision for the operator after the independent alignment read, which no\n"
            "receipt produces.",
        "`specified`, nothing more.":
            f" *Status 2026-10-11:* done; the five theorems (`RNProbability`) are kernel-checked by run {RUN} (PR head `{C8}`; actual tested merge `{M1}…`), by the Mathlib lemmas listed;\n"
            "`rn-count-interface` stays `none` as predicted. Beyond this step, in round 4 (module `RNCluster`; branch\n"
            f"`{BRANCH4}`, {R4PR_TXT}; kernel-checked by run {RUN8}: PR head `{H8}`, actual tested merge `{M8}…` with base `{BASE3}…`) `RN.RareClusterPoissonObstructionO`, Theorem O of\n"
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
            "stays `specified` over `PeriodicCoefficientInterface`, and the enclosure still needs the proof-producing interval pipeline described here." + step4_r5,
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
                    f"`FullBlockCoverIff`. Kernel-checked by the hosted run {RUN6} (PR head `{C6}`; actual tested merge `{M6}…` with base `{BASE2[:8]}…`, then the\n"
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

    # ---------------------------------------------------------------- §7 commands (round 5 of the renderers, unchanged)
    doc = replace_once(doc, '(cd "$P" && lake exe cache get) && python3 -B -S "$P/check.py" --execute\n',
                       '(cd "$P" && lake exe cache get) && python3 -B -S "$P/check.py" --execute\n'
                       '\n'
                       '# Math- statement proofs (this programme; path dependency on the statement register, Mathlib cache as above)\n'
                       'Q=specifications/lean_statement_proofs_20261011\n'
                       'python3 -B -S "$Q/test_check.py" && python3 -B -O -S "$Q/test_check.py"\n'
                       'python3 -B -S "$Q/check.py"                                   # source mode: identity, lexical scan, inventory = registered Props\n'
                       '(cd "$Q" && lake exe cache get) && python3 -B -S "$Q/check.py" --execute   # fresh build, leanchecker (~14 GB), axiom audit, controls, receipt\n')

    # ---------------------------------------------------------------- §8: the same-account reviews and integrations so far, at credit 0
    doc = replace_once(doc, "provider, family and agent records an alignment through the existing validators. The cheapest\n"
                       "independent review available is the cross-provider one the rollout record names: the OpenAI lane\n"
                       "reviews the Anthropic packages and vice versa.",
                       "provider, family and agent records an alignment through the existing validators. The cheapest\n"
                       "cross-provider review available is the one the rollout record names: the OpenAI lane\n"
                       "reviews the Anthropic packages and vice versa. Under the owner's shared project account such a review differs in provider, family\n"
                       "and agent but earns organizational-independence credit 0, recorded separately from its technical verdict, and the gate that requires\n"
                       "organizational independence stays open.")
    doc = replace_once(doc, "never by editing the register's label.\n",
                       "never by editing the register's label.\n\n"
                       f"*As of 2026-10-12:* Math- {R5PR_TXT} (round 5) received two bounded nonauthor reviews ({REV1}, source/type, PASS; {REV2}, receipt-binding\n"
                       "readback, PASS), and the five Math- PRs #419, #422, #423, #434 and #439 were merged by OpenAI/Codex engineering integrators, all under the\n"
                       "owner's same project account. Each is recorded here as a technical verdict and, separately, at organizational-independence credit 0; none is\n"
                       "an alignment record, an approval or an acceptance. The alignment dimension of the statement register and of the proof packet stays\n"
                       "`PENDING_INDEPENDENT_REVIEW`, and the gate that requires independence remains open.\n")

    # ---------------------------------------------------------------- output invariants
    for bad in ("<<", ">>", "{{", "}}"):
        assert bad not in doc, bad
    for para in paragraphs(doc):            # C422-E1 at paragraph level: a run is never named without its tested merge
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
                for run in list(PUSH_RUNS) + [MAIN_PR_RUN[0]]:
                    assert run not in cell, (run, cell[-240:])  # the runs outside BINDINGS are named in prose only, where their binding is asserted
    for n in set(re.findall(r"\b\d{11}\b", doc)):      # no run id outside the binding lists
        assert n in head_of or n in PUSH_RUNS or n == MAIN_PR_RUN[0], n
    for para in paragraphs(doc):                        # push runs: the paragraph says push and names the pushed (= checked) commit
        for run, commit in PUSH_RUNS.items():
            if run in para:
                assert "push" in para and commit in para, (run, para[:200])
        if MAIN_PR_RUN[0] in para:                      # the one main pull_request run: PR head and actual tested merge (C422-E1), base named
            assert f"PR head `{MAIN_PR_RUN[1]}`" in para and f"tested merge `{MAIN_PR_RUN[2]}`" in para and MAIN_PR_RUN[3][:8] in para and "pull_request" in para, para[:200]
    assert "is not recorded in this document" not in doc
    # C422-E1 at mention level: every mention of a Math- pull_request run has its PR head and its tested merge within MENTION_W characters
    # on either side; the paragraph-level check above alone passed '(PR #423, run 38149888699)' in commit 368a3417 (see unbound_mentions)
    unbound = unbound_mentions(doc)
    assert not unbound, "\n".join(f"{r}: ...{ctx}..." for r, ctx in unbound)
    flat = doc.replace("\n", " ")
    for pos in occurrences(flat, MAIN_PR_RUN[0]):
        if flat[pos - len("formal-verification-"):pos] != "formal-verification-":        # the artifact name follows the bound mention
            assert MAIN_PR_RUN[1] in flat[pos:pos + 420] and MAIN_PR_RUN[2] in flat[pos:pos + 420], flat[pos - 80:pos + 200]
    for sentence in re.split(r"(?<=[.;])\s", doc):      # the landing push did not kernel-check anything
        if RUN_LANDPUSH in sentence:
            assert "kernel-checked" not in sentence and "kernel receipt" not in sentence, sentence[:200]
    for phrase in ("uncommitted worktree", "on no commit", "staged in this programme"):
        assert phrase not in doc, phrase
    # the pending forms of render_spec8 are gone: no dagger, no 'no hosted receipt', no receipt-'none yet', no 'await'
    assert "†" not in doc and "no hosted receipt" not in doc and "await the" not in doc, "a pending form survived"
    assert "none yet" not in doc.replace("proof lane: none yet", ""), "a 'none yet' outside the register's 'proof lane: none yet' cell survived"
    # the Math- PRs are described in their landed state: every 'stacked on' is past tense
    for pos in occurrences(doc, "stacked on"):
        assert re.search(r"(then|was|each PR was) $", doc[max(0, pos - 12):pos]), doc[max(0, pos - 80):pos + 40]
    assert not re.search(r"\b(is|are) stacked\b", doc) and "Nothing is open" not in doc
    for stale in ("Statement entries (352)", "The 155 `proved`", "83 entries", "73 theorems whose types", "72 registered Props", "proves 72 of",
                  "73 distinct registered", "all 73 of its theorems"):
        assert stale not in doc, stale
    for m in re.finditer(r"\b7[23]\b", doc):            # 73 and 72 survive only as the round-4 receipt's historical numbers
        ctx = doc[max(0, m.start() - 40):m.end() + 40]
        assert re.search(r"theorem_count 73|covered_props 72|receipts all 73 theorems of that head|receipted all 73 on PR #434|"
                         r"72 entries|Basic\.lean:72|\.lean:\d*72|:72\b", ctx), ctx
    assert f"Statement entries ({c['statements']})" in doc and f"proves {NP} of the 141" in doc and f"{NT} theorems whose types are exactly {NP} registered Props" in doc
    # round 4 and round 5 bindings
    assert BRANCH4 in doc and R4PR_TXT in doc and RUN8 in doc and M8F in doc and HEAD8F in doc and BASE3F in doc and SHA8 in doc and TREE8[:8] in doc
    assert BRANCH5 in doc and R5PR_TXT in doc and RUN9 in doc and M9F in doc and HEAD9F in doc and BASE4F in doc and SHA9 in doc and JOB9 in doc
    assert RUN10 in doc and HEAD10F in doc and M10F in doc and REV1 in doc and REV2 in doc
    assert all(p in doc for p in new_props) and all(p in doc for p in NEW13_SHORT) and all(m in doc for m in NEW_MODULES)
    assert all(p in doc for p in NEW2_SHORT) and all(m in doc for m in R5_MODULES)
    for m in NEW_MODULES + R5_MODULES:                  # every statement cell naming a round-4 or round-5 theorem is unmarked
        for mt in re.finditer(rf"proof:`{m}\.[A-Za-z0-9]+` \(proved\)", doc):
            assert doc[mt.end():mt.end() + 1] != "†", doc[mt.start() - 20:mt.end() + 5]
    for sentence in re.split(r"(?<=[.;])\s", doc):      # a round-4 / round-5 theorem or module is said to be kernel-checked only with its run named
        if "kernel-checked by" in sentence and (any(f"`{m}" in sentence for m in NEW_MODULES) or any(p in sentence for p in NEW13_SHORT)):
            assert RUN8 in sentence, sentence[:240]
        if "kernel-checked by" in sentence and (any(f"`{m}" in sentence for m in R5_MODULES) or any(p in sentence for p in NEW2_SHORT)):
            assert RUN9 in sentence, sentence[:240]
    for para in paragraphs(doc):                        # C422-E1 for the new runs, with PR head and base as well
        if RUN8 in para:
            assert M8 in para and H8 in para and BASE3 in para, para[:200]
        if RUN9 in para:
            assert M9 in para and H9 in para and BASE4 in para, para[:200]
        if RUN10 in para:
            assert M10 in para and H10 in para, para[:200]
        for run in (RUNP, RUNS_STMT, RUNS_AGG):
            if run in para:
                assert MP in para and HP in para, para[:200]
        if REV1 in para or REV2 in para:                # a same-account review is never mentioned without its zero credit
            assert "credit 0" in para and "PENDING_INDEPENDENT_REVIEW" in para, para[:200]
    # the landing paragraph: five merges, exact times, the final cut's runs, and what landing is not
    lp = next(p for p in paragraphs(doc) if p.startswith("**Landing on Math- main (2026-10-12).**"))
    for pr, _, sha, when in LANDING:
        assert f"#{pr} (" in lp and sha in lp and when in lp, (pr, sha)
    for x in (HEADPF, MPF, RUNP, RUNS_STMT, RUNS_AGG, TREE_CUT, INTEG_REVIEW, RELEASE_COMMENT, RUN_LANDPUSH, REG_SHA256, "credit 0",
              "not a\nmathematical acceptance and not an alignment", "moves no claim, Prop or theorem label", "did not\nrun the Lean statement or proof packets",
              "still not on main", MERGED_AT_NOTE, f"PR #439's head at merge, `{HEADPF}`", "event pull_request"):
        assert x in lp, x
    assert "Independent of all six steps" in doc
    bullet4 = doc[doc.index("4. **A proved Prop is not an accepted claim.**"):doc.index("review and the controlling acceptance machinery say otherwise.")]
    assert "Each receipted theorem is L5" in bullet4 and "Each such theorem is L5" not in bullet4, "L5 may be claimed for receipted theorems only"
    for sentence in re.split(r"(?<=\.)\s", bullet4):
        if "L5" in sentence:
            assert "receipted" in sentence and "without a hosted receipt" in sentence, sentence   # the negative control stays: a receipt-less theorem is not L5
    assert "none yet" not in bullet4 and f"all {NT} of its theorems carry a hosted kernel receipt" in bullet4
    assert all(x in bullet4 for x in (RUN8, H8, M8, RUN9, H9, M9, BASE4, CUT[:8], "not an acceptance and not an alignment", "moves no label"))
    assert bullet4.count("their first receipt") == 2 and "which also receipts" not in bullet4      # fe99ca31 hand fix 1, for both rounds
    assert "every negative control\nrefused for its intended reason" not in doc and doc.count("every control classified as intended") + doc.count("every control\nclassified as intended") == 2
    assert "evidence about" not in bullet4 or "for that Prop" in bullet4      # a receipt is evidence about the Prop, never about the informal claim
    for sentence in re.split(r"(?<=[.;])\s", doc):
        if RUN8 in sentence or RUN9 in sentence:
            assert not re.search(r"evidence (about|for|that) the (informal|research) claim", sentence), sentence[:200]
    assert "the claim's own label moves to `proved`" not in doc                # claim rows carry only `none` / `specified`
    for wrong in ("proves the claim's display itself", "proves the\nclaim's display itself", "could carry `proved`", "claim-level move is the operator's",
                  "claim-stating Prop", "cheapest\nindependent review", "integration candidate head", "already names it as the smallest"):
        assert wrong not in doc, wrong                  # no alignment assumed, no claim-level `proved`, credit 0 for the shared account
    for mt in re.finditer(r"head\s+of\s+(PR\s+)?#4\d\d", doc):     # a run's base is described as it was then, not as the PR is now
        assert re.search(r"then\s+the\s+$", doc[max(0, mt.start() - 12):mt.start()]), doc[max(0, mt.start() - 80):mt.end() + 20]
    assert "credit 0" in doc[doc.index("cross-provider review available"):doc.index("cross-provider review available") + 600]
    n_items = len(re.findall(r"^\d\. \*\*", doc[doc.index("things follow, and the claim table"):doc.index("## 2. Environment")], re.M))
    assert n_items == 4 and "Four things follow" in doc and "Three things follow" not in doc, n_items   # the count says what follows
    assert "Nothing here is reviewed" not in doc and "Nothing here is alignment-reviewed" in doc
    assert "the base of every round-5 run" not in doc
    DOC.write_text(doc)
    print(f"wrote {DOC} ({len(doc.encode())} bytes); register {REG_BLOB[:8]} sha256 {REG_SHA256[:8]}… at Math- {CUT[:8]}; claims {c['claims']} "
          f"({dict(c['fp'])}), statements {c['statements']}, proved {proved_n} (main {pk[MAIN_SIDE24] + pk[MAIN_P15]} entries / {NMAIN} theorems; proof packet "
          f"{NPE} entries / {NT} theorems over {NP} Props, {NREC} with a hosted receipt); round 4 {NPEND4} theorems first receipted by run {RUN8} "
          f"(head {H8}, tested merge {M8}…); round 5 {NR5} theorems {sorted(NEW2_SHORT)} first receipted by run {RUN9} (head {H9}, tested merge {M9}…, "
          f"base {BASE4}…), corroborated by run {RUN10} (head {H10}, tested merge {M10}…); landed cut {CUT[:8]}; triage {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
