# CBZ baseline ledger

**State: Partial Baseline; CBZ-0 is not established.**

This ledger binds the currently observed repository snapshot to the twelve
owner criteria and final gate matrix. It records evidence and gaps; it is not a
completion certificate.

## Bound snapshot

Observed sequentially on 6 November 2026:

| Repository | Default revision |
|---|---|
| Main | `04989724b5142f8eea8d41946d0367d0372d845a` |
| Math- | `f532c3e2d91a0c7679a37c733741660917a9b21b` |
| Universal-Law-Workspace | `ed9a1e0106d616bc3a1eabb213fe4605f9e30813` |

Workspace pin verification at the recorded merge observed all eight pins and
four canonical inputs current, with 187/187 tests passing in each interpreter
mode. Those observations are sequential, not atomic or perpetual.

The latest durable contribution receipt is
[Main #684 comment 6684436967](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6684436967).
It records Library checkpoint v78 and its exact hashes. The durable checkpoint
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
| 8 | Pins resolve and current tips agree | **PARTIAL.** Workspace passed at the recorded instant; current-tip agreement is not atomic or perpetual. |
| 9 | Tools complete, documented and self-verified | **BLOCKED.** [TOOLING.md](TOOLING.md) registers components and the missing final orchestrator. |
| 10 | Five canonical surfaces truthful and consistent | **CANDIDATE.** This revision adds the three missing surfaces and reconciles STATUS, but default-branch admission and readback are still required. |
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
| G11 | Tooling completeness | Component registry present; final orchestrator absent |

## Cold-start order

A fresh operator starts with [STATUS.md](STATUS.md), then reads
[GOVERNANCE.md](GOVERNANCE.md), [PROOF_INDEX.md](PROOF_INDEX.md),
[TOOLING.md](TOOLING.md), this ledger and [DECISIONS.md](DECISIONS.md). The
operator must bind exact native refs, run the documented component controls,
preserve every output and failure, and stop short of CBZ-0 unless all twelve
criteria and G0–G11 are observed true together.
