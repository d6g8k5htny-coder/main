# Website and verification contribution — Partial Baseline

This contribution repairs immutable source identity and equation-excerpt saving, preserves the existing mathematical records, and supplies independently checked, explicitly scoped verification evidence. It is not a federation CBZ-0 declaration. Canonical federation integration is coordinated on [main#684](https://github.com/d6g8k5htny-coder/main/issues/684); the source contribution is [main#700](https://github.com/d6g8k5htny-coder/main/pull/700).

## Observed results

- Main proof/gate cut `9c09dc98a639341c8bfa088edefe011131016b41`, composed with Main `c93cec70223281492709ff74ca84fed5dc6e40e8`: 31 literal arithmetic proofs, 45 constants, four exact nonvacuity contracts, seven adverse controls, and fresh kernel replay passed. These are not analytic-parent or Gaussian-model certificates.
- Current source tests, including the separately hash-bound lifecycle fix: 1,784 Python tests, zero failures, 23 explicit skips. Supplemental runs execute 18 distinct skipped tests in 29 executions; five tests still lack real-record preconditions. The original skip and failure reports remain intact. Focused tools pass 144 tests normally and 144 optimized. Configured Node regression passes 763/763 with zero skips.
- Math cut `8151f592918a845bf19eb8eafaf8444540a32734`: nine module kernel replays, 63 public target reports and the full 116-declaration compiled inventory passed the unchanged 82-command gate through a verified sequential launcher. This does not establish source-model applicability, nonvacuity, newer Math cuts, or fresh replay of all imported Mathlib dependencies.
- The contribution index preserves 207 original records: 31 CLOSED literal Main statements and 176 QUARANTINED application/source obligations. Separate formal-evidence axes do not silently promote those obligations.

The mutation graders previously could accept fixture assertions as body failures, and an outer optimized run did not optimize scanner subprocesses. The repaired synchronous grader tracks named body invocation, population and fixture phases; rejects unsupported deferred bodies; and propagates optimization. Actual controls cover 25 lifecycle cases, 10 classifier mutants, 14 original guard mutants and 12 scanner mutants in each mode. Earlier recorded mutant failures remain valid at their exact body-failure scope. Independent engineering review found and verified the deferred-body fix; no scientific-independence credit is asserted.

## Reproduce the packaged checks

From this repository root:

```sh
python -B -S cbz/evidence/website-foundation-20261029/construction/verify_proposal.py
python -B -O -S cbz/evidence/website-foundation-20261029/construction/verify_proposal.py
python -B -S -m unittest discover -s tests -p 'test_cbz_*.py' -v
python -B -O -S -m unittest discover -s tests -p 'test_cbz_*.py' -v
```

The first two commands verify record identity and byte hashes, including negative metadata controls; they do not run a kernel or certify mathematics. Full toolchain, adapter, source-cut and remaining-boundary commands are in the namespaced [STATUS](evidence/website-foundation-20261029/STATUS.proposed.md) and [Math delivery README](evidence/math-8151-run004/README.md). There are 15 documented contribution tools. Canonical admission and the federation final run remain separate tasks; this handoff does not report them green.

## CBZ-0 criteria at this contribution's verification cut

| Criterion | Observed contribution and remaining federation work |
| --- | --- |
| 1. Clean Git/worktrees | Own source is committed for delivery after the recorded tests; initial local cleanup and recovery evidence are preserved. No final all-repository Git assertion. |
| 2. No open PRs/dangling work | #700 is an integration contribution. The canonical queue still has other open work; no empty-queue assertion. |
| 3. Terminal proof index | All 207 contribution records are CLOSED or precisely QUARANTINED. This is not the complete federation census. |
| 4. Layer A audit | Finite auditors and exact adverse-input boundaries are tested. Unsafe tool-source findings, lexical coverage gaps and binary coverage remain explicit; no exhaustive Layer A certification. |
| 5. Full regression | Actual results above; five real-record cases remain unexercised. No 100% federation-suite assertion. |
| 6. Lean/kernel/axioms | Main and pinned Math scopes above passed. All-project and fresh imported-dependency closure are not inferred. |
| 7. Mutations | Declared finite contribution controls pass; no all-federation mutation result. |
| 8. Pins | Exact source/runtime/dependency cuts are bound. Full federation synchronization remains with the canonical pin verifier. |
| 9. Tools | 15 finished tools registered in the proposed namespace, with usage and self-verification. Canonical registry incorporation remains separate. |
| 10. Canonical surfaces | Seven consistent proposed surfaces supplied; canonical root surfaces deliberately not overwritten by this contributor. |
| 11. Cold start | Fresh ZIP identity verification passes without Git or network. This is not the full STATUS-only clone-and-all-gates test. |
| 12. Ledger/administration | Contribution decisions, recovery manifests and evidence are supplied. Federation integration and remaining dispositions are not declared complete. |

## Downloads and recovery

The [existing Drive folder](https://drive.google.com/drive/folders/14NFE6ng7qqL4hkbwSeNdHmhOjV0BbYAB) retains the release snapshots. The older `b7edd52` source ZIP is explicitly named `ARCHIVE_...known_grader_limits`; preserve it as history rather than using its old graders for new acceptance decisions.

The standalone [Math evidence ZIP](https://drive.google.com/file/d/1Axf7DpNL028IqvmmCB9wf_C-Kmsc_zON/view?usp=drivesdk) has SHA-256 `5da23a87bf81e13131391f35530b74fe28c83e1d4b88c5febc90eba8dcbc519b`; its independent raw Drive download matched. [Dirty-worktree recovery instructions](https://drive.google.com/file/d/1np2k1O689vvtVx62mHfvlYM893s5RGYi/view?usp=drivesdk) link all eight uploaded archive parts. Their local concatenation matched the original SHA-256; provider size/name/parent readbacks passed. Remote chunk byte hashes were not independently round-tripped. The complete original recovery archive remains local; nothing was deleted from Drive.

Layer A remains non-waivable: no placeholder closure, new unverified axioms, floating-point certification of mathematical bounds, weakened bounds, suppressed failing tests, unmarked vacuity, fabricated status, or untested auditor claims.
