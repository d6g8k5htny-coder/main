# Formal source-membership implementation plan

> **For agentic workers:** use superpowers:executing-plans for this one coupled data correction, then a fresh whole-branch review. Steps use checkbox syntax.

**Goal:** connect existing Main coverage row 312 to the SIDE24 source carriers and all three arithmetic modules already declared in the formal manifest.

**Architecture:** retain the existing ledger and deterministic hypergraph builder. Correct one row's source and implementation membership; add no unit, schema, registry or scientific predicate.

**Tech stack:** existing Python index tool, Hypergraph-DB 0.4.0, pytest; administrative Git/JSON/SHA-256 checks only.

**Spec:** [canonical bounded pickup, Main579/6387917545](https://github.com/d6g8k5htny-coder/main/pull/579#issuecomment-6387917545), applying the owner's delegated retroactive source-to-implementation task. The implementation choice is the acting worker's under that delegation.

## Constraints and decisions

- Frozen base: Main `9c03589b1a3f0c1fd6befd09abe6b95c99cff7d7`.
- Exactly 344 existing row identities and ordered hyperedges remain. All other 343 raw rows and every node, bound, status, lane, record and Drive-hit cell remain unchanged.
- Row 312 gains `main:formal/sources/side24_v1/PROOF.md` and `main:formal/sources/side24_v1/ENCLOSURE.json` in the source cell, and `formal/UniversalLaw/Side24/Ledger.lean` beside the two existing implementation paths.
- The existing `formal/manifest.json` is the provenance sidecar, not another theorem. Its 27 PROOF-source targets (17 Ledger, 10 ConeMoments) and four ENCLOSURE-source endpoint targets declare the relationship. Verify all bound hashes and the two local-copy/source identities; do not refresh the manifest.
- ENCLOSURE.json supplies frozen published numbers. The endpoint lemmas prove arithmetic about those numbers, not enclosure of the coefficient. Formal statement alignment remains `PENDING_INDEPENDENT_REVIEW`.
- Row 313's root module contains three imports, not another proof. It keeps its MISSING source cell. Research-translation, zeta and external storage remain outside this change.
- Expected counts: 214 sourced / 130 missing containers, 1293 vertices (1290 retained plus two source vertices and one implementation vertex), 344 hyperedges. Implementation-container counts remain 296/48; adding the omitted module is a member correction, not a newly implemented container.
- #581's preservation statement applies to its earlier two-row correction. This new change explicitly corrects row 312's implementation membership.
- No formal/math source, manifest, workflow, runtime, dependency, permission, graph-binding or scientific-status mutation. No mathematical program, mathematical tests, Lean build or formal gate execution. No Ready, MAIN-LOCK or author integration.

## Review focus

Check exact path resolution across the sibling source directory; distinguish source-copy provenance from mathematical alignment; verify published-number input is not described as an enclosure proof; preserve import-only row 313; compare every old vertex and ordered edge rather than relying only on counts. These are data-audit checks, not new product tests.

## Task 1: Correct the existing membership

**Files:** modify `data/hypergraph/{README.md,math_coverage_ledger_20261023.md,math_hypergraph.json,query_results.json}` and existing count expectations in `tools/hypergraph/test_math_hypergraph.py`; create this plan.

**Interfaces:** consumes the current ledger, graph/report snapshots and manifest-declared source-to-module relationship; produces the same output schema and CLI with one corrected binding row.

- [x] Verify the isolated clean baseline with `python -B -m pytest -q tools/hypergraph`: expected 19 passed.
- [x] Update only existing sourced/missing golden expectations to 214/130. Run the suite against unchanged data: expected three count assertions fail, 16 pass. Do not add tests that merely mirror this reversible data edit.
- [x] Correct row 312, update explanatory current counts and preserve dated earlier snapshots. Regenerate with `python -B -m tools.hypergraph.math_hypergraph build`.
- [x] Audit complete source bytes, Git blobs and SHA-256; all other ledger rows/cells, all 1290 prior vertices and all ordered edge metadata; require only the declared source/module membership delta.
- [x] Run the frozen index suite (19 passed), `build --check` (match) and `validate` (empty problems), recording actual return codes. These checks establish index consistency, not mathematics.
- [ ] Commit the exact six-path candidate, obtain a fresh nonauthor whole-branch review, then publish a Draft PR via native Git objects. Verify native/local parent, tree and blob equality.
- [ ] Deliver the finite source-stage result and personal release in the existing threads. Separate eligible integration retains current-base, review, Ready-event, checks, protection and queue requirements.

No new production code or dependency is introduced. Existing source/review scopes and scientific acceptance are unchanged. credit 0; OBL OPEN; NOT A RESULT.
