# Three declared consumer memberships implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for this single task; obtain one fresh whole-branch engineering review.

**Goal:** Add three explicitly declared consumer memberships to existing rows 57, 62 and 311 without changing scientific status or existing members.

**Architecture:** Reuse the existing coverage ledger and deterministic hypergraph builder. One writer combines the independently proposed row deltas, then regenerates the existing two JSON outputs once.

**Tech Stack:** Existing Python metadata builder and pinned Hypergraph-DB 0.4.0; no new dependency.

**Spec:** [Root pickup](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6467415686), [Capability Frontier's exact proposal](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6467446841), [personal acceptance](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6467475210), and [accepted handoff / only-own release](https://github.com/d6g8k5htny-coder/main/issues/605#issuecomment-6467485158).

## Global constraints

- Frozen Main input: `f20a971dbbdcca8e0b653e701a2877347daf1ad4`; Math source cut: `4a0d51bbd06f0d9ac731a1d1df38f25999797135`.
- Capability Frontier supplied rows 57/62; source-inventory supplied row 311. One combined source writer; no foreign role release or integration seat.
- Retain every original row cell/member except the three additive implementation-cell entries; retain every original vertex and ordered edge identity.
- Preserve 345 rows, source 219/126, implementation-container 297/48, 30 placeholder-only and 14 bound counts. Vertex counts describe metadata, not accepted theorems.
- No mathematical/checker/workflow/replay/Lean execution; no direct Bridge/C46/C47 edge, status/GRAPH/register/Drive mutation or scientific acceptance.

## Review focus

- The existing C2/E3 workflow memberships have trigger/consumed-source roles and must remain.
- The reused E3 wrapper vertex must not be duplicated when row 57 gains it.
- Row 311 co-membership must not imply that the theta program implements every source in the aggregate row.
- Finite sums/quadrature and rounded printouts provide a declared numerical sanity check, not a certified identity or new execution.
- Historical snapshots, source hashes and earlier review scopes remain historical; the pending #482 manifest repair is not adopted.

## Task 1: Add the three memberships

**Files:** Modify `data/hypergraph/README.md`, `data/hypergraph/math_coverage_ledger_20261023.md`, `data/hypergraph/math_hypergraph.json`, `data/hypergraph/query_results.json`; retain this concise plan.

**Interfaces:** Consume only the frozen ledger, unchanged builder/GRAPH/binding-report inputs and literal declarations. Produce the same existing JSON formats and a Draft review candidate.

- [ ] Verify the clean existing 19-test index baseline.
- [ ] Append `.github/workflows/elder-third-order.yml` to row 57, `.github/workflows/far-elder-rate.yml` to row 62 and `experiments/zeta_fold_exploration/theta/project_theta_checks.py` to row 311; document exact source pins and limits.
- [ ] Run existing `test_committed_outputs_current` before regeneration and confirm its expected stale-output failure finishes.
- [ ] Run `python -m tools.hypergraph.math_hypergraph build` once.
- [ ] Run the existing 19-test suite, `build --check`, `validate` and whitespace check. Require PASS, MATCH, `[]` and exit 0.
- [ ] Compare all 345 raw rows, all old vertices, all ordered edge fields and all query leaves. Require only the declared additive memberships and their deterministic derived effects.
- [ ] Commit, obtain a separate whole-branch engineering review and publish one exact Draft with current source/testing/review identities; release only the source-author role.

No new implementation or mirror test is needed for this reversible data-only amendment; existing stale-output validation and a complete before/after audit exercise the relevant behavior. Repository-wide mathematical, browser and runtime suites are outside this metadata change.
