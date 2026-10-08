# Research architecture implementation plan

> For agentic workers: use the Superpowers implementation and verification workflows; coordinate exact scopes in existing main307/PR discussions. The original eight-item goal persists through all phases.

**Goal:** Install and verify the eight requested enhancements while preserving source provenance and independent evidence dimensions.

**Architecture:** Reuse the current graph gate/profile/retrofit and acquired-source contracts. Separate the static publication pipeline from lease/write admission and local retrieval/routing; execute model-authored code in ephemeral environments.

**Tech stack:** stdlib Python 3.11 graph/evidence adapter, Astro static interface, transactional lease service, sqlite-vec sidecar, LiteLLM adapter, GitHub-hosted runners and Docker, Canvas with accessible HTML fallback.

**Spec:** ../specs/2026-10-08-research-architecture-design.md

## Global constraints

- Scientific effect NONE; no new scientific or global ownership register.
- Preserve original node/edge records and exact source hashes; absent evidence is not_recorded.
- Reuse source hard_gate validation; never execute it before identity verification.
- New tests and model-authored code run in disposable hosted/container environments.
- Do not edit active peer scopes or weaken existing hosted/formal or local-rule gates.
- Installed persistent local workers remain sole-local, tool-free Qwen3:8b without paid fallback.

## Review focus

- Self-consistent malicious provenance must not authorize arbitrary executable gate bytes: gate source identity requires an independent trusted pin.
- Graph context edges can be cyclic while required edges cannot; preserve gate semantics.
- A removed dependency edge must not hide a downstream alignment affected through the former edge.
- Private source retrieval and public graph export have different disclosure boundaries.
- Partial build/output and unknown provider delivery must not be accepted or retried as successful work.

## Task 1: Pinned graph export and ephemeral execution lane

Files: new tests/test_architecture_graph_export.py, tools/proof_graph_export.py, architecture/graph_export.py, architecture/__init__.py; new .github/workflows/research-architecture.yml. Existing pinned source files remain unchanged.

Interface: `proof_graph_export.py --source-dir PATH --output DIR [--expected-provenance-sha256 DIGEST]` produces deterministic graph.json with source commit/capture/graph/gate hashes, unchanged keyed nodes/edges, conservative evidence dimensions and scientific_effect NONE. The default expected provenance SHA256 is independently pinned in code; an explicit alternate digest is a trusted caller input. No provenance input can authorize a different executable gate from the independently reviewed gate digest. Nonzero on identity/shape/cycle failure; failed runs do not replace prior valid output.

- [x] Write stdlib unittest controls against CLI (real pinned snapshot, missing dependency, required/context cycles, source tamper, duplicate keys, symlink, deterministic bytes, absent dimensions, retained prior output).
- [x] Run tests before implementation in hosted nonroot/no-network/read-only container; retain expected RED logs (run37841652787; see progress record).
- [x] Implement byte verification, trusted gate-pin check, strict parse, source gate invocation and atomic output promotion (candidatef0a0c18).
- [x] Rerun normal and optimized Python controls; export and retain graph artifact from the same container run (run37842412305, artifact11577958009).
- [x] Obtain fresh source/security review, reconcile findings and preserve existing required checks at exact candidate (review5462723442; see execution record for readiness-review correction).

## Task 2: Evidence adapter and regression join

Files: new architecture/evidence_dimensions.py, tests/test_architecture_evidence_dimensions.py; extend new workflow only. Coordinate later narrow Math transition integration separately.

Interfaces: `compare_evidence(old_graph, new_graph, old_evidence, new_evidence) -> report` names changed dimensions, seeds, affected descendants and exact affected alignment-record identities using union edges. Inputs follow existing retrofit/formal contracts; no accepted kernel claim without its original source-bound receipt.

- [ ] Write changed statement, proof-only change, lost source, failed/skipped receipt, stale alignment, removed-edge and unknown mapping controls; observe RED in sandbox.
- [ ] Implement closed-schema adapters and reuse the pinned reverse-impact traversal.
- [ ] Add base/candidate input binding and report preservation on every PR.
- [ ] Join graph/evidence result into the existing required exact-candidate verify aggregate, or independently verify required-context enforcement; negative controls reject failed/skipped/missing/stale producers. The first standalone workflow is not yet required enforcement.
- [ ] Verify actual formal change controls and receive nonauthor review before transition integration.

## Task 3: Artifact Index join and Astro proof pages

Files: new architecture/artifact_index.py, tests/test_architecture_artifact_index.py; new interface/astro.config.mjs, package.json/lockfile, src/pages and public generated assets. Reconcile future docs/site and Pages workflow scopes with existing owners before touching them.

Interfaces: pinned public-allowlisted Artifact Index export + validated graph/evidence -> immutable page data/build identity; `npm ci && npm run build` consumes exact graph.json artifact. Unknown/prose joins require explicit mapping and stay unresolved.

- [ ] Test allowlisting, wrong hashes, duplicate IDs, unresolved joins, source unavailable and immutable build identity.
- [ ] Pin Astro/D3 dependencies and build proof pages with clickable source/dependency/review/Lean details and accessible tables.
- [ ] Verify deterministic build, mobile/desktop keyboard/light/dark and tampered-artifact refusal.
- [ ] Integrate verified graph artifact consumption and source-bound Pages deployment; confirm live exact build.

## Task 4: Lease broker and participating write admission

Files: separate architecture/lease_service module/client/tests plus private deployment configuration; precise persistence choice follows measured SQLite-service/Postgres compatibility probe.

Interfaces: atomic acquire/renew/release/admit-write, signed canonical capability and increasing fence; idempotent outbox to existing Work Events/PR receipt surfaces. No legacy claim ignored during migration.

- [ ] Race/expiry/tamper/stale-fence/head-conflict/restart/outbox/legacy/bypass controls in sandbox.
- [ ] Implement broker with private service-held signing key and durable transactions.
- [ ] Enroll participating agents and publication boundary; test a paused worker after successor lease.
- [ ] Verify actual enforced coverage and remaining direct writers before claiming duplicate/overwrite prevention.

## Task 5: Private semantic retrieval sidecar

Files: isolated coordinator-side sqlite-vec adapter/test/benchmark; local worker integration only after exact-final two-nonlocal review and separate tested rule quorum when required.

Interfaces: derived generation manifests with ~500 true-token chunks and exact citations; retrieve within remaining prompt budget; source availability/freshness dominates similarity; cache binds generation/query/chunks.

- [ ] Pin embedding/tokenizer/extension; held-out parity, paraphrase, deep-file, stale/access-loss/exclusion/injection/rebuild/citation/coverage/fairness controls.
- [ ] Implement generation promotion and exact-ID/literal fallback.
- [ ] Benchmark accepted-task quality/citation fidelity, tokens, cold/warm latency, construction/invalidation/RAM and worker fairness.
- [ ] Obtain required actual reviews, install reversible adapter, verify both roles on live selected sources.

## Task 6: Capability and budget router

Files: separate broker/adapter/tests; credentials private. Interfaces: deterministic task admission, explicit allowed route, budget reservation, actual usage and unknown-delivery reconciliation.

- [ ] Test local-only invariant, unauthorized provider/private disclosure, cost reservation, model substitution, failed/unknown delivery and escalation.
- [ ] Implement three admitted tiers through LiteLLM with no silent paid fallback for local roles.
- [ ] Verify actual accessible endpoints and measured cost per accepted task; preserve failures.

## Task 7: Canvas reader and execution-entrypoint migration

Files: new Canvas teaching module/accessibility controls after site-owner scope reconciliation; sandbox command adapter at every actual Lean/Python/numerical entrypoint.

- [ ] Verify elder/current-parent/neutral/essential/finite-bar conventions and cubic normalization against pinned source.
- [ ] Build interactive Canvas views with DOM/text fallback; verify layout/focus/reduced-motion/source details and live deployed source.
- [ ] Add sandbox escape/network/process/memory/time/output/cleanup controls and legitimate Lean/Python replay.
- [ ] Migrate every actual agent-code entrypoint, exporting immutable failures/receipts before disposal; record residual host execution until eliminated.

## Final completion audit

- [ ] Inspect actual deployed/runtime/PR/receipt evidence for all eight original requirements.
- [ ] Confirm all required checks and actual completed reviews against exact current candidate and protected integration.
- [ ] Keep goal active if any requirement is missing, partial, unverified or merely planned.
