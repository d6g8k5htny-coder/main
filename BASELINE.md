# CBZ baseline ledger

**State: Partial Baseline; CBZ-0 is not established.**

This ledger binds the currently observed repository snapshot to the twelve
owner criteria and final gate matrix. It records evidence and gaps; it is not a
completion certificate.

## Bound snapshot

Observed sequentially on 6 November 2026:

| Repository | Default revision |
|---|---|
| Main | Candidate containing this file; native source parent `f3b950cfe1b2428f5204448bc3ec52d4eda5d3ff`, based on default `299c7fad72f1079b87b2d044b2d30f2098678f29` |
| Math- | `f532c3e2d91a0c7679a37c733741660917a9b21b` |
| Universal-Law-Workspace | `05f3096d077a6fa4ff6cf763c074451cb1ac9972` (pins Main `299c7fad72f1079b87b2d044b2d30f2098678f29`) |

Workspace #41 refreshed the Main pin after #763 and its recorded qualification
observed all eight pins and four canonical inputs current. This later Main
candidate has since advanced, so strict-current agreement is presently
unawarded. Those observations are sequential, not atomic or perpetual.

The latest durable contribution receipt is
[Main #684 comment 6685706413](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6685706413).
It records durable checkpoint v79 and its exact hashes. The durable checkpoint
is an incomplete handoff, not proof of this baseline.

## Twelve criteria

| # | Criterion | Current disposition |
|---:|---|---|
| 1 | All Git clean; no stashes or orphan worktrees | **PARTIAL.** The available scratch root was observed with 14/14 clean checkouts, zero stashes, valid HEADs and no missing registered paths. Other hosts and future changes are not covered. |
| 2 | No open PRs or dangling deliveries/refs | **PARTIAL.** Organization-visible open PRs were zero, but 2,049 retained non-default branches still need explicit integration, archival or dormant disposition. |
| 3 | Every proof CLOSED or exactly quarantined | **BLOCKED.** [PROOF_INDEX.md](PROOF_INDEX.md) records the incomplete reconciliation boundary. |
| 4 | Full Layer A audit clear | **BLOCKED.** Scoped censuses and component audits are not a whole-federation Layer A clearance. |
| 5 | Full regression 100% | **BLOCKED.** No fresh whole-federation regression over this single snapshot is recorded. |
| 6 | Lean/build/kernel/approved axioms; no sorry | **BLOCKED.** Fresh current-composition results for the entire admitted proof scope are absent. |
| 7 | Declared mutation threshold met | **BLOCKED.** Component mutation receipts do not establish one federation threshold. |
| 8 | Pins resolve and current tips agree | **BLOCKED.** Workspace passed at its recorded merge, but its Main pin predates the current source parent. A guarded strict-current refresh is still required. |
| 9 | Tools complete, documented and self-verified | **BLOCKED.** The adverse-tested G11 attribution repair closes the 20 active gaps, while [TOOLING.md](TOOLING.md) retains 29 no-current-exercise tool candidates and the missing final orchestrator. |
| 10 | Five canonical surfaces truthful and consistent | **PARTIAL.** All five surfaces exist and record the current source relation and residue, but a final same-snapshot consistency and cold-start readback is absent. |
| 11 | Fresh clone reads STATUS alone and reruns every gate | **BLOCKED.** No STATUS-only cold-start execution over this revision is recorded. |
| 12 | Decisions complete; no pending administration | **BLOCKED.** Retained refs, proof reconciliation and final integration remain pending. |

No row marked PARTIAL or CANDIDATE is awarded.

## Final gate matrix

The required final pass must execute every row over the same bound snapshot.
Component evidence, cached output and skipped or partial jobs cannot substitute.

| Gate | Required observation | This snapshot |
|---|---|---|
| G0 | Auditor adverse mutations | Not run as one final federation gate |
| G1 | Static Layer A audit | Not run as one final federation gate |
| G2 | Approved-axiom audit | Not run as one final federation gate |
| G3 | Lean build | Not run as one final federation gate |
| G4 | Kernel verification | Not run as one final federation gate |
| G5 | Full regression | Not run as one final federation gate |
| G6 | Declared mutations | Not run as one final federation gate |
| G7 | Pins and interface drift | Component observations only; final gate absent |
| G8 | Canonical status consistency | Candidate documentation change; final gate absent |
| G9 | STATUS-only cold start | Not run |
| G10 | Git/ref administration | Local root and remote census only; retained-ref disposition incomplete |
| G11 | Tooling completeness | Workflow-scope and active-attribution repairs passed their adverse suites; 29 no-current-exercise tool candidates and final-orchestrator residue remain |

## Cold-start order

A fresh operator starts with [STATUS.md](STATUS.md), then reads
[GOVERNANCE.md](GOVERNANCE.md), [PROOF_INDEX.md](PROOF_INDEX.md),
[TOOLING.md](TOOLING.md), this ledger and [DECISIONS.md](DECISIONS.md). The
operator must bind exact native refs, run the documented component controls,
preserve every output and failure, and stop short of CBZ-0 unless all twelve
criteria and G0–G11 are observed true together.
