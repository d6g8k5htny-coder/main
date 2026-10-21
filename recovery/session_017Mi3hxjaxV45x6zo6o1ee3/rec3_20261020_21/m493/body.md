Closes only the `elder-lower-review.yml` execution-wrapper slice of #322. This is a source-author candidate; it is not independent review, integration, mathematical acceptance, or scientific promotion.

## Exact identities

- Base: `5e20834d7daccb8d6bf8b89036e8eb9eb98a1452`
- Current head: `f2350f47f18105ed73a004b67c2f699ca7f2ffe4` (ahead 4 / behind 0)
- Workflow blob: `4a26f8b6b2cca40ff96942e01b134cdfb4e88c17`
- New test blob: `f2524a66e29c40031b65636ee04fe9359ec0c7a1`
- Unchanged packet blobs:
  - `SOURCE_FILES.json`: `33a83598f5f1cfeeb2bda1bdd1616ff115a5569a`
  - `REVIEW.md`: `179c1e6d3d4005b2e406b1bd4d9584caf5465793`
  - checker: `b35cf96a977f3048efafb5bbc74e397037a91ace`
  - `RESULTS.json`: `931611ed14c4d210aa8e579c9e5c8b732fcaea61`

Exactly two paths change: the wrapper and its synthetic real-shell regression test.

## Repair

The wrapper now binds baseline rc/stdout/stderr, each actual mutant's complete deterministic JSON report derived from the source-characterized false checks, and M99 rc/stdout/native-argparse stderr in normal and optimized child modes. The unknown-label expectation is independently generated from the fixed label set under the pinned interpreter; it is not learned from the checker child.

The regression test rejects unrelated traceback, silent exit 1, wrong report bytes, output noise, child stderr, wrong return codes, baseline corruption, unknown-label protocol deviations, source-guard weakening, and dirty final Git state.

## Test-first and local evidence

- RED on untouched workflow: crash and silent-exit probes produced 20 expected failures because the wrapper accepted them and emitted its terminal acceptance record.
- GREEN after the pinned-runtime correction: `python3 -B -S -m unittest discover -s tests -p 'test_elder_lower_review_workflow.py' -v` — 10 tests, OK, 57.079s.
- GREEN after the pinned-runtime correction: `python3 -B -O -S -m unittest discover -s tests -p 'test_elder_lower_review_workflow.py' -v` — 10 tests, OK, 56.739s.
- Complete repaired workflow shell against the exact authentic packet on available CPython 3.12.14: rc0, exact terminal record, empty stderr.
- Authentic checker in both modes preserved the 1,487-byte baseline SHA-256 `4a431a8a3fa1527339aa25bac3f1ff22168529aadbb45e437add2257d58e35aa`; mutant output sizes/hashes match the diagnostic characterization.

## Retained failed evidence

The first published head `3636d34b92a036a77f0bbd64ff6ce3e1da80f504` failed dedicated run [39447648015](https://github.com/d6g8k5htny-coder/Math-/actions/runs/39447648015) on pinned CPython 3.11.16: the hard-coded Python 3.12.14 argparse rendering differed. That exact ordinary technical failure is preserved; it was not a governance or access denial. The current head derives the rendering independently under the pinned interpreter. No stale result from the old head is reusable.

Exact-head hosted checks and a distinct nonauthor review are required before Ready or integration. Any head/base change invalidates prior checks and review.

Scientific effect: **NONE**. Mathematical acceptance: **false**.

[Diagnostic terminal](https://github.com/d6g8k5htny-coder/Math-/issues/322#issuecomment-6294211736) · [source pickup](https://github.com/d6g8k5htny-coder/Math-/issues/322#issuecomment-6294256881) · [canonical owner mandate](https://github.com/d6g8k5htny-coder/main/blob/b665307520948308a53d190d7341bf301df3b3a2/governance/OWNER_MANDATE_20261015.md) · [standing delegation](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OP-AUTONOMY-20260923-v2.1.md)
