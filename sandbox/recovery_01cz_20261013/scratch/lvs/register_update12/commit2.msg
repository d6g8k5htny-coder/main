Lean verification specification: bind round 5 and the landed Math- cut 4afcca6a

Documentation only: docs/LEAN_VERIFICATION_SPECIFICATION.md re-rendered by the
scratch renderer render_spec10.py (derived from render_spec9.py; its base text is
always the doc as committed at 91cffbb5, read from git, and it refuses to run
unless the working file was first restored to fe99ca31) from the Math- statement
register at the final landed cut 4afcca6a:
specifications/lean_statements_20261010/REGISTER.json, blob 05cc80c1, 599058
bytes, sha256 d185a292c20084d6f12e56a4ed76f7d6b9294b0d224830e47885934b1869f87c.

What changed:
- Round 5 is bound wherever the document enumerates proof rounds (section 1
  bullet 4, the 3.7 row, the section 4 column key and claim rows, the 5.1 table
  and prose, 5.2, the section 6 Step 4 status, section 8): the theorems
  referenceCovarianceLowerBound (Side24Covariance) and coneMomentD2Identity
  (Side24ConeMoment), first receipted by lean-specification-proofs.yml run
  38195810463 on Math- PR #439 head c6ce0256, actual tested merge 1320c5b2...
  with base be014794..., theorem_count 75; corroborated by run 38198716748 on
  head 32fc9845, actual tested merge 37e0f1d8...; reviews 5486104267 and
  5486293837 (OpenAI/Codex, same project account) are recorded as technical
  records at organizational-independence credit 0.
- Counts, each asserted against the register: 354 statement entries (was 352),
  157 `proved` entries (155), 85 proof-packet entries (83), 75 theorems (73)
  over 74 Props (72). The document now says that entry counts are not distinct
  theorems (134 distinct: 59 main, 75 proof packet), and that 41 of the 72 main
  entries are the 28 P15 targets of main PR #358 at 91cffbb5, still not on main.
- Present-tense "stacked on" descriptions of the Math- PRs are replaced by
  their landed state, and a run's base reads "then the head of #4xx". One dated
  landing paragraph (2026-10-12) in section 3 gives the five protected merge
  commits (#419 d6f2c325, #422 8a61d5fb, #423 6e366697, #434 7fd045a6, #439
  4afcca6a; times labelled as GitHub merged_at) and the final cut's pre-merge
  pull_request runs 38205232428, 38205232441 and 38205232621, each bound to PR
  #439's head at merge de732f1e and actual tested merge 8fe0759e, whose tree
  d323c3d6 equals the landing tree. The landing push run 38205855996 is named
  as a push run whose checked commit is 4afcca6a itself and which did not run
  the Lean packets.
- The three wording fixes applied by hand at fe99ca31 are now rendered.
- Run bindings: every mention of a Math- pull_request run carries its PR head
  and actual tested merge within the mention (the renderer asserts this per
  occurrence; a paragraph-level check had passed "(PR #423, run 38149888699)").
  The main push runs 38068951535 and 36356485214 name their pushed commits. The
  main pull_request run 38096827443 is bound to PR head 91cffbb5 and actual
  tested merge 3c02eb7d004c4b7a44f5ce7b04c2c5de94e47c0c (base 6e11abfb), as the
  checkout log and the receipt's checked_commit in the log of its job
  114344336471 record it.
- Sentences a full read found false are corrected: "Three things follow"
  (four items follow); bullet 3's "Nothing here is reviewed" (now "Nothing here
  is alignment-reviewed", with the credit-0 reviews and integrations named);
  the section 6 frame sentence and Step 2's outcome and status, which let a
  claim's label move to `proved` (claim rows carry only `none` or
  `specified`). In the register's own words, DemandOneRefutation is the Prop
  the register names as stating p15-price-boundary; whether it says what the
  source paragraph says stays PENDING_INDEPENDENT_REVIEW; and any claim-level
  label beyond none/specified is the operator's register-schema decision after
  that review. Section 8 now says that a cross-provider review under the shared
  project account earns organizational-independence credit 0. The section 4
  column key notes that two transcribed fields of row
  cert:math-full_price-full_price.py predate round 3 (RhoStarEnclosure has had
  a theorem since round 3); the register itself is not changed here.

What this does not establish: no claim, Prop or theorem label moves (the 171
claims stay 33 specified / 138 none). Landing on Math- main is an engineering
integration, not an acceptance and not an alignment. Alignment stays
PENDING_INDEPENDENT_REVIEW. Every review and integration named carries
organizational-independence credit 0, and the gate that requires independence
remains open. A receipt is evidence about a registered Prop for the formal
objects only, never about an informal claim. The landing push did not
kernel-check the packets. The 28 P15 targets of this PR are still not on main.
The #315-first recomposition of #358 (manifest hashes, target-count test) is
not done here. Scientific effect NONE.

Checks: tools/formal_gate_check.py exit 0 (59 targets, alignment
PENDING_INDEPENDENT_REVIEW). tools/workspace_landing_check.py exit 0;
tools/navigation_check.py and --verify-public exit 0, no problems. unittest
discover over tests/ (-B -S, and again with -O): 535 tests, OK, 5 skipped.
pytest on tests/: 530 passed, 5 skipped, 10 failed; the same 10 casefold tests
fail at fe99ca31 in a detached worktree. Repository-root pytest stops at
collection on four experiments/periodic_h0 modules that need numpy, identically
at fe99ca31. The registers, claims, quarantine, manifest, mirror, directive,
noncertifying and disclosure checkers named in the session checklist do not
exist in this repository.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY
