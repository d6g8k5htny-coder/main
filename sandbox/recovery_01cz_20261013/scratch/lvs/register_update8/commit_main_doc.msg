Lean verification specification: record the round-4 proof-packet theorems (no hosted receipt yet) and the triage of the Props without a theorem

Documentation only: docs/LEAN_VERIFICATION_SPECIFICATION.md re-rendered from
the Math- statement register at commit 48bfcbae (Math- draft PR #434, stacked
on #423). What changed in the document: §1 bullet 4 now distinguishes the 60
proof-packet theorems with a hosted kernel receipt from the 13 theorems of the
round-4 modules Side24Analysis, Side24Interfaces and RNCluster, which have NONE
yet (the first hosted lean-specification-proofs.yml run on that branch is their
first receipt); only a receipted theorem is called L5 evidence, and the
renderer refuses any sentence that says otherwise. §3.7 row, §4 column key and
claim rows (dagger on every round-4 proof entry), §5.1 counts (352 statement
entries, 155 proved entries, 73 theorems over 72 Props) and prose, the round-4
triage paragraph (82 Props without a theorem classified: 10 PROVABLE_NOW all
proved, 5 PROVABLE_WITH_WORK of which 3 proved, 7 BLOCKED_MISSING_MATHLIB with
the named Mathlib gap, 24 BLOCKED_INTERFACE each refuted at a concrete
inhabitant, 28 PREDICATE_VOCABULARY), §5.2 Theorem O clause and the §6 step
statuses. Every number is asserted against the register and the triage table
by the rendering script, which refuses a paragraph naming a hosted run without
its PR head and actually tested merge commit.

Rendered twice: the first rendering's verification (same account, credit 0)
refuted one sentence that extended "L5 evidence" to the unreceipted theorems;
this rendering carries the correction and its negative control. No formal/
byte, target, manifest hash or test changes; the 28 kernel-proved P15 targets
and their receipt are untouched. Scientific effect NONE; no label moves;
alignment PENDING_INDEPENDENT_REVIEW; organizational-independence credit 0.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY
