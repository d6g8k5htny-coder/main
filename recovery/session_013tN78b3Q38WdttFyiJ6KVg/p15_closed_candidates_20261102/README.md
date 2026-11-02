# Candidate CLOSED-index rows for the 28 P15 targets landed by main#740 (2 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE. These are
candidate rows. Nothing here is written to `cbz/evidence/kernel/all-closed-declarations.json`, and none of these rows
is CLOSED until a kernel-index composer adopts it with a receipt.

## Why

My federation census listed main's `formal/UniversalLaw/P15/{PriceBoundary,RealizedCovers,FullPrice}.lean` as genuine
Lean that the CLOSED index does not yet contain; the index's main kernel cut is `c3377cc8`.

- main#740 landed them at merge `ec6962ff5fafe04364e094a954896dc1e62941f5` on 1 Nov 15:29Z.
- The merge's tree, `b482cbca…`, equals the tree of the tested head `52cea4c5`.
- `formal/` is unchanged from the merge to main `d2b0216e`.
- At that head the hosted check `formal / formal-evidence` (run 41320377025, job 124889085814) succeeded. It ran
  `lake build`, `leanchecker`, the transitive `#print axioms` audit and the negative controls for all 59 manifest
  targets, and uploaded artifact `formal-verification-41320377025` (ID 13088624355, SHA-256 `d04efcb1…`, 70 files,
  including `receipt.json` and `axioms.log`).

## What the rows hold (`p15_closed_candidates.json`)

There is one row per P15 target in `formal/manifest.json` at the merge:

- the declaration, path and line (each a unique `theorem` line), and the source SHA-256 at the merge;
- the `scope` and `vacuity_boundary` copied from K-MAIN-001;
- the manifest's own `title`, `informal_anchor` and `does_not_claim`;
- the hosted receipt reference.

None of the 28 is already in staging's index. The manifest's `alignment_status` is `PENDING_INDEPENDENT_REVIEW`.

## Limits

- **`axioms` is null.** The hosted gate fails closed unless every target passes the transitive axiom audit against
  `allowed_axioms`, and that job succeeded. But the exact per-declaration set is only in the artifact's `axioms.log`,
  which this session cannot download. So I record no axiom set.
- **The receipt is hosted, not committed.** Existing K-MAIN rows cite a committed receipt
  (`cbz/evidence/kernel/main-arithmetic/receipt.json`). Adopting these rows needs the index lane's own committed receipt
  for main at or after `ec6962ff`, or the artifact read into the repository.
- No Lean was run here. There is no toolchain in this session, and I ran no repository code.

## Reproduce

```sh
python3 -I -B p15_closed_candidates.py MAIN_CLONE STAGING_CLOSED_JSON OUT.json
# STAGING_CLOSED_JSON = cbz/evidence/kernel/all-closed-declarations.json at 32ab7d9013264d2874aa4b778a90eede0cda0cdf
```
