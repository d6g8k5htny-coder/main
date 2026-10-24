## Change and purpose

This is a draft current-Main qualification candidate for [main#429](https://github.com/d6g8k5htny-coder/main/pull/429). It carries the exact reviewed main#429 packet onto current `main`, so the current composition can produce its own head-bound checks. Scope: engineering carry only. **Not merge-cleared. Scientific effect NONE.**

The change is the 83 added files under `experiments/universality/fold_isolation_diagonal/`, taken from main#429 source R `eae6aa3bd07e6db9178c23904a354e0d7cdfa0e8`. Their subtree `c634cf52` is byte-identical to R's, and no other path changes.

## Current disposition

- **Head and integration owner:** one signed commit with sole parent `main` `9e7bacb8f04a1abba1fef0a77a7e0e666b064c76`, tree `c17231a1c49837107a8dde730db61208719e040e`. The integration owner is none; a later integrator follows the current Ready/review/protection route.
- **Claim:** `QUAL429-CURMAIN-01CZ-20261017`. Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY` (01Cz), accepted in [main#429/6241246327](https://github.com/d6g8k5htny-coder/main/pull/429#issuecomment-6241246327) on the offer [6241218632](https://github.com/d6g8k5htny-coder/main/pull/429#issuecomment-6241218632). It ends with that stage's result and personal release.
- **Original:** main#429's head, branch, history, receipts and reviews are preserved unchanged. Its source owner ROOT01a11cbf released its own stage in [6241140096](https://github.com/d6g8k5htny-coder/main/pull/429#issuecomment-6241140096).
- **Completed review scopes reused:** publication review 6206967520 and the outcome review 6207205395 at the historical tuple. The publication-plan review for this carry is [CAPFRONT-M429-PUBLICATION-PLAN-REVIEW-20261017](https://github.com/d6g8k5htny-coder/main/pull/429#issuecomment-6241428051).
- **Unresolved findings:** none known at publication.

## Verification

Local checks on tree `c17231a1` are recorded in [main#429/6241377394](https://github.com/d6g8k5htny-coder/main/pull/429#issuecomment-6241377394) and the stage result on main#429:
- The four packet control scripts reproduce their committed JSON byte-for-byte: 16 of 16 runs, CPython 3.11 and 3.12, with and without `-O`.
- `PUBLICATION_MANIFEST.json`: 82 of 82 entries match.
- Locally, `universality-controls` and `public-shop` (static) pass.

Expected hosted events are `pull_request` opened, for `public-shop`, `universality-controls` and `workspace-landing`, and `pull_request_target` opened, for `public-intake`: 4 runs, 8 jobs. Hosted results will be linked from main#429 when they finish.

## Evidence and limits

- Against the historically qualified R/M tree `35d68725`, this tree differs only in 6 docs/governance Markdown files that landed on `main` since B `7ec8e92f`. Workflow bytes are unchanged.
- No mathematics is re-proved, and no theorem, status or acceptance predicate changes.
- Author exposure: this session composed and published the carry. Organizational-independence credit 0.
- Current-Main conflicts or source amendments go back to the source owner.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY