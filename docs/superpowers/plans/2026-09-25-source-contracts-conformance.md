# Source Contracts and Architecture Conformance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish versioned cross-repository data contracts and a read-only conformance checker before any production source code is moved.

**Architecture:** `meta-framework` owns normative JSON contract schemas **and** the read-only architecture conformance checker/tests. `trial` owns only cross-repository rehearsal/install fixtures and must not be a runtime dependency of the checker. Production packages remain runtime-independent and this plan has no scientific promotion authority.

**Tech Stack:** Python >=3.11, standard library runtime, JSON/JSON Schema documents, unittest.

**Spec:** `docs/superpowers/specs/2026-09-25-cross-repo-source-architecture-design.md`

## Global Constraints

- Scientific effect: NONE.
- No theorem/proof/review body moves.
- No package runtime dependency across repositories.
- `sandbox` stays private and cannot appear as a public artifact source.
- Git source refs require immutable 40-hex commits and exact byte identity.
- Unknown major contract versions fail closed.
- Hashes establish identity only.
- No PyPI/npm publication.
- No changes to scientific dispositions or claim graph state.
- Existing research/review lanes remain untouched.

## Review Focus

1. Wrong repository with a valid-looking path must fail instead of binding locally.
2. A public manifest containing a sandbox path must fail.
3. Unknown schema major version must fail closed.
4. Derived navigation must not claim canonical scientific authority.
5. Workspace snapshots must reject mutable refs.

---

### Task 1: Source-reference and repository-role contracts

**Files:**
- Create: `meta-framework/schemas/source_ref.schema.json`
- Create: `meta-framework/schemas/repository_role.schema.json`
- Create: `meta-framework/examples/source_ref.git.json`
- Create: `meta-framework/examples/source_ref.drive.json`
- Create: `meta-framework/examples/repository_role.json`
- Modify: `meta-framework/README.md`
- Create: `meta-framework/tools/contract_validation.py`
- Create: `meta-framework/tests/test_source_contracts.py`

**Interfaces:**
- Produces `validate_source_ref(data, public=True)` and `validate_repository_role(data)`.

- [ ] Write failing tests for immutable commit, owner/repository, safe path, sandbox refusal, external frozen source, role fields, and major-version refusal.
- [ ] Run `python -B -S -m unittest -v tests.test_source_contracts` and observe RED.
- [ ] Add version 1.0 source-ref and role schemas/examples.
- [ ] Implement test/integration validator functions in trial.
- [ ] Re-run the tests and obtain GREEN.
- [ ] Document schema authority and non-scientific-status semantics in meta README.
- [ ] Commit meta and trial changes independently.

### Task 2: Source-manifest and workspace-snapshot contracts

**Files:**
- Create: `meta-framework/schemas/source_manifest.schema.json`
- Create: `meta-framework/schemas/workspace_snapshot.schema.json`
- Create: `meta-framework/examples/source_manifest.json`
- Create: `meta-framework/examples/workspace_snapshot.json`
- Modify: `meta-framework/tools/contract_validation.py`
- Modify: `meta-framework/tests/test_source_contracts.py`

**Interfaces:**
- Produces `validate_source_manifest(data)` and `validate_workspace_snapshot(data)`.

- [ ] Add failing tests for duplicate repos, mutable refs, private-path leakage, manifest file hashes/sizes, and `scientific_status_authority=false`.
- [ ] Observe RED.
- [ ] Add schemas/examples and validator functions.
- [ ] Obtain GREEN.
- [ ] Commit independently.

### Task 3: Read-only architecture conformance checker

**Files:**
- Create: `meta-framework/tools/architecture_conformance.py`
- Create: `meta-framework/tests/test_architecture_conformance.py`

**Interfaces:**
- Consumes a sibling-repository workspace root.
- Produces `check_workspace(root: Path) -> dict` and CLI exit 0/2.

- [ ] Write failing synthetic-workspace tests for valid roles, wrong owner, sandbox leak, mutable snapshot, and derived status-authority violation.
- [ ] Observe RED.
- [ ] Implement offline read-only checker.
- [ ] Add clean CLI refusal behavior without tracebacks.
- [ ] Obtain GREEN.
- [ ] Run against the current multi-repo fixture where available; record real violations without weakening rules.
- [ ] Commit.

### Task 4: Hosted CI, mutation controls, and trial rehearsal

**Files:**
- Create or modify: `meta-framework/.github/workflows/contracts.yml`
- Modify: `meta-framework/tests/test_source_contracts.py`
- Modify: `meta-framework/tests/test_architecture_conformance.py`
- Modify: `trial/federation/test_federation.py` only to invoke the meta checker as a subprocess/CLI against the sibling fixture; do not import meta Python modules.

- [ ] Add semantic mutations for mutable commit, wrong owner, sandbox source, major-version bump, duplicate repository.
- [ ] Run complete local federation tests.
- [ ] Ensure Actions are SHA-pinned, contents-read only, no `pull_request_target`.
- [ ] Push exact heads and inspect hosted logs.
- [ ] Commit CI update.

### Task 5: Independent engineering review and Phase-A gate

- [ ] Request a fresh reviewer on exact meta/trial heads.
- [ ] Resolve any Critical/Important findings with RED→GREEN fixes.
- [ ] Record CI/runtime/complexity metrics.
- [ ] Proceed to query migration only if hosted tests and review are clean.


### Task 6: Authority-map compatibility assertions

**Files:**
- Modify: `meta-framework/tests/test_architecture_conformance.py`
- Modify: `meta-framework/tools/architecture_conformance.py`

**Interfaces:**
- Consumes the active-base authority semantics without importing main runtime code.

- [ ] Add fixtures asserting the architecture layer never owns `classification`, `controlling`, `terminality` or `reverse_impact_revalidation`.
- [ ] Add a fixture asserting public registry artifact rows refuse `status`, `grade`, `classification`, and `disposition` fields.
- [ ] Add a fixture distinguishing landing `REVIEWED_SCOPED` from gate `PROVED_REVIEWED`.
- [ ] Run tests RED→GREEN.
- [ ] Commit and include these checks in the fresh engineering review.
