# Formalization review lane (statement alignment)

The Lean kernel checks that a proof is valid. It cannot check that the theorem
proved is the theorem the informal text claims. That check is a human or model
review task with its own lane, distinct from Layer 0 analytic review and from
the author-side act of writing the Lean text.

## What the reviewer does

For each Lean declaration under review:

1. Read the `informal_anchor` in `registry.json` and locate it in the pinned
   local source copy (`formal/sources/...`); confirm the surrounding context.
2. Read the Lean statement (not the proof). Decide whether the Lean statement is
   the informal statement at the stated scope — the same constants, the same
   quantifiers, the same direction of inequality, the same units or scaling
   (many pilot statements are cross-multiplied integer forms of rational
   identities; the docstring states the rational form — check that the
   cross-multiplication is right).
3. Read the `does_not_claim` field. Confirm the Lean statement does not
   silently claim more than the anchor, and that what it omits is named.
4. Record the verdict per declaration: `ALIGNED`, `ALIGNED AT NARROWER SCOPE`
   (say what is narrower), or `MISALIGNED` (say exactly how). Do **not** grade
   the proof; the kernel already did. Do not grade the Layer 0 mathematics;
   that is a different lane.

The reviewer does not need to run Lean to review alignment, but should confirm
the gate is green at the reviewed commit and quote that run.

## What the reviewer records

Copy [`reviews/TEMPLATE.md`](reviews/TEMPLATE.md) to
`reviews/<date>_<scope>_<reviewer-slug>.md`. Record:

- Reviewer identity and provider. **Same-provider reviewers earn zero
  organizational-independence credit** whatever the verdict (this repository's
  existing rule; it applies to Lean text exactly as to prose).
- Source exposure: whether the reviewer read the author's Lean proof text or
  docstrings before forming a view of the statement. Reading the proof first is
  the weakest form of review and must be stated.
- The exact registry commit, Lean file SHA-256 values, and the gate output.
- One row per declaration with the verdict.

Then, in `registry.json`, set that claim's `alignment_review` to
`{"status": "reviewed", "author": ..., "reviewer": ..., "record": "formal/reviews/<file>.md"}`.
The gate refuses `reviewed` without a reviewer and an existing record file.
A `MISALIGNED` verdict leaves the status `open` and is corrected at the source:
fix the Lean statement (a new hash, a re-run kernel check) or fix the anchor.

## What this lane cannot do

- It cannot promote a Layer 0 status. `STATUS.md` and `PROOF_INDEX.md` move only
  through their own source-bound review.
- It cannot make an author-side proof independent by relabelling; independence
  is recorded from the reviewer's actual identity.
- It cannot substitute for formalising the analytic content. An aligned
  arithmetic lemma remains an arithmetic lemma.

## Current state

All 31 pilot declarations are `open`: authored by Anthropic / Claude via a
Cursor cloud agent on 2026-09-27, kernel-checked, not yet alignment-reviewed by
anyone else. A reviewer from OpenAI, Google, xAI or a human would earn
organizational-independence credit; another Anthropic model would not.
