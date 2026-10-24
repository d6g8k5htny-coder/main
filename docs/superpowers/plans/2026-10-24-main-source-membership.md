# Main source membership implementation plan

> **For agentic workers:** Use superpowers:executing-plans for this single data task, followed by one fresh whole-branch review.

**Goal:** Restore tracked mathematical-source membership for two existing Main containers without adding theorem units or scientific acceptance.

**Architecture:** Reuse the existing coverage ledger, parser and deterministic generated outputs. Attach the 28 Markdown files under each of `experiments/periodic_h0` and `experiments/universality` at Main `32366ac0d4e2d1464eaac51866b9dce2128970b9`. These are container members, including proofs, reviews, notes and reports, not 56 independently accepted theorems.

**Tech stack:** Existing Python index tool, pinned Hypergraph-DB 0.4.0 and pytest.

**Spec:** [Finite source-author pickup](https://github.com/d6g8k5htny-coder/main/pull/579#issuecomment-6378292451), under the [owner mandate](../../../governance/OWNER_MANDATE_20261015.md) and [standing delegation](../../../governance/OP-AUTONOMY-20260923-v2.1.md). The implementation approach is the agents' choice under the user's delegated objective.

## Constraints and review focus

- Exactly 344 ledger rows. Change only source cells of rows 307 and 310; preserve every other row/cell and all implementation, node, bound and status values.
- Source-present/missing container counts become 213/131; retain the original 211/133 snapshot explicitly. Implementation counts 296/48, 30 placeholder-only, 14 bound units and existing graph-node counts stay unchanged.
- Paths must be tracked regular Markdown files at the frozen source commit, without duplicates, qualified under the correct Main container. Source presence and co-location do not establish theorem correctness or a semantic proof-to-program relationship.
- Exclude automatic classification of zeta, research-translation, formal or other discovered files. A note's declared GitHub-comment proof remains a separate custody boundary.
- No proof, source code, runtime, config, workflow, permissions, GRAPH/v2 bindings, scientific predicate, accepted status or canonical original review changes. Index tests must not execute mathematical source programs.
- Correct the dated blanket Dropbox no-connector statement using the existing CoS actor-specific disposition; represent no completed storage crawl and publish no private locators.

## Task 1: Enrich membership and regenerate the existing index

**Files:** the coverage ledger, `data/hypergraph/{math_hypergraph.json,query_results.json,README.md}`, existing `tools/hypergraph/test_math_hypergraph.py` count expectations, and this plan.

**Interface:** the existing semicolon-separated source cell and unchanged `parse_ledger`, `_resolve`, `build`, `gaps` and `validate` behavior.

- [x] Confirm the baseline hypergraph suite passes in a complete checkout; materialize tracked sparse paths when necessary without changing shared sparse configuration.
- [x] Update only existing count expectations to 213/131. Run them against old data and observe the expected count/header/CLI failures.
- [x] Replace only the two MISSING source cells with the explicit sorted 28-path lists. Update dated count documentation and regenerate with the existing builder.
- [x] Run the complete hypergraph suite, `build --check` and `validate`; check 344 rows, 56 exact member identities and all preserved counts/cells.
- [ ] Audit the complete diff and obtain fresh nonauthor review of the frozen candidate. Publish a draft source-stage PR and hand final integration to a separate eligible integrator through the current queue.

This plan records one bounded index correction. It does not claim complete repository-, history- or cloud-wide mathematical coverage. Credit 0; scientific effect NONE; eng ≠ discharge; no flags; OBL OPEN; NOT A RESULT.
