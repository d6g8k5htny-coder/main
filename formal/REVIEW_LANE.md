# Independent statement-alignment review (this package)

The Lean kernel checks that a proof establishes its statement. It cannot check
that the statement is the one the informal text makes. That is an independent
review task with its own lane, distinct from Layer 0 analytic review and from
the author-side act of writing the Lean text. The contract is the lane's
([formal-verification guide](../docs/FORMAL_VERIFICATION.md), "Gate and trust
boundary") and is restated for this package in
[SCOPE.md § Review contract](SCOPE.md#review-contract).

## Who may review

Anyone whose provider, family **and** agent all differ from the author's
(`Anthropic` / `Claude` / the `author.agent` string of `manifest.json`: a
Cursor cloud agent on 2026-09-27 for the Side24 modules and Claude Code session
`session_01Cz7WZybv8znP64SpPj6sWY` on 2026-10-10 for the P15 modules). The validator refuses
a record where any of the three coincides, case-insensitively. The `agent` check
is an exact (trimmed, case-folded) string comparison against the single
`author.agent` string, which names both agents; a reviewer's `agent` must be
neither of the two agents named in it, not merely different from the combined
string, and the validator cannot see that distinction on its own — the
integrator checks it by hand and should re-check it when main #315 lands.
Same-provider
reviewers earn zero organizational-independence credit whatever they write;
this is the repository's existing rule and it applies to Lean text as to prose.

## What the reviewer does

Read the exact current head first; a changed source, scope, toolchain or
manifest stales any earlier record. Then, for each of the 59 targets in
`manifest.json`:

1. Locate the `informal_anchor` in the pinned local source copy under
   `sources/side24_v1/` or `sources/p15_v1/` and read its context.
2. Read the Lean **statement** (not the proof) and its docstring. Decide
   whether it is the informal statement at the scope stated in `SCOPE.md`:
   same constants, same quantifiers, same direction of inequality, same
   scaling. Most targets are cross-multiplied integer forms of rational
   identities; check the cross-multiplication.
3. Read `does_not_claim` and the "Not established" column of `SCOPE.md`.
   Confirm the Lean statement does not silently claim more than the anchor and
   that what it omits is named.
4. Record a per-target verdict: `ALIGNED`, `ALIGNED AT NARROWER SCOPE` (say
   what is narrower) or `MISALIGNED` (say exactly how). Do not grade the proof
   (the kernel did) and do not grade the Layer 0 mathematics (different lane).

The reviewer need not run Lean, but should quote the gate output at the
reviewed commit (`SOURCE_IDENTITY_PASS … <manifest digest>` and, if available,
the workflow run that produced the receipt).

## What the reviewer records

Two files under `reviews/`, named `<date>_<reviewer-slug>.md` and `.json`:

- **Markdown**: reviewer identity and provider; **source exposure** (whether the
  Lean proof text or docstrings were read before forming a view of each
  statement — reading the proof first is the weakest form and must be stated);
  the reviewed commit, manifest digest and scope digest; one row per target
  with the verdict.
- **JSON**: copy [`reviews/TEMPLATE_alignment_review.json`](reviews/TEMPLATE_alignment_review.json)
  and fill it in. `disposition` is `ACCEPTED` only if **every** target is
  `ALIGNED` or `ALIGNED AT NARROWER SCOPE` with the narrowing recorded;
  otherwise leave the record unaccepted — the validator refuses it, which is
  the intended outcome. `manifest_sha256` is the SHA-256 of
  `formal/manifest.json` at the reviewed commit; `scope_sha256` is the hash of
  `formal/SCOPE.md` recorded in the manifest's `files`. `targets` is exactly
  the 59 manifest target names. `evidence` points to the committed Markdown
  record by repository, 40-character commit, path and SHA-256.

Validate: `python3 tools/formal_gate_check.py --alignment formal/reviews/<file>.json`.
The validator checks structure, digests, coverage and lineage independence. It
does not authenticate that the review happened; a controller has to retrieve
the referenced evidence at the referenced commit.

A `MISALIGNED` verdict is corrected at the source: fix the Lean statement (new
hash, new `--run-lean` receipt) or fix the anchor, then re-review.

## What this lane cannot do

- Promote a Layer 0 status. `STATUS.md` and `PROOF_INDEX.md` move only through
  their own source-bound review.
- Make an author-side proof independent by relabelling; independence is
  recorded from the reviewer's actual identity and lineage.
- Substitute for formalising the analytic content. An aligned arithmetic lemma
  remains an arithmetic lemma.
- Flip `alignment_status` in `manifest.json`. That field stays
  `PENDING_INDEPENDENT_REVIEW`; the accepted record is the evidence, and a
  controller consumes it.

## Current state

All 59 targets: authored by Anthropic / Claude (31 Side24 targets via a Cursor
cloud agent on 2026-09-27; 28 P15 targets via Claude Code session
`session_01Cz7WZybv8znP64SpPj6sWY` on 2026-10-10, organizational-independence
credit 0); `proved` at source; `kernel-checked` only in the receipt of a
trusted run; alignment `PENDING_INDEPENDENT_REVIEW`. No record other than the
template exists. Record actual pickup in
[work item #95](https://github.com/d6g8k5htny-coder/main/issues/95) so two
reviewers do not duplicate the work.
