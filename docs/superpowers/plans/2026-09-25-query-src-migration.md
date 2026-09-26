# Query Source Package Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `query-` into an installable source-layout package while preserving all current root CLIs through compatibility wrappers and behavior-parity tests.

**Architecture:** Canonical implementation moves into `src/universal_law_query/`. Root scripts remain thin wrappers during migration. Runtime stays standard-library only and independent of every other repository; trial performs cross-repo install/federation tests.

**Tech Stack:** Python >=3.11, setuptools.build_meta, pyproject.toml, standard library, unittest.

**Spec:** `docs/superpowers/specs/2026-09-25-cross-repo-source-architecture-design.md`

**Prerequisite:** Source Contracts and Architecture Conformance plan reviewed cleanly.

## Global Constraints

- Scientific effect: NONE.
- Zero runtime dependencies.
- No runtime cross-repository imports.
- No PyPI publication.
- Existing root commands remain behavior-compatible.
- Catalog schema_version 1 remains accepted unchanged.
- Unknown keys/malformed identities fail closed.
- A catalog hit remains explicitly non-acceptance.
- Private sandbox artifacts remain rejected.
- No proof/review/scientific-status modifications.

## Review Focus

1. Root wrapper must work from an uninstalled source checkout.
2. Installed CLI must work in a clean environment without sibling repos.
3. Duplicate JSON keys/path traversal/symlink payloads must remain refused.
4. Wrapper and package CLI must match exit status and JSON/refusal behavior.
5. Source archive must exclude caches/private paths and be deterministic.

---

### Task 1: Package skeleton and CLI extraction

**Files:**
- Create: `query-/pyproject.toml`
- Create: `query-/src/universal_law_query/__init__.py`
- Create: `query-/src/universal_law_query/catalog.py`
- Create: `query-/src/universal_law_query/cli.py`
- Create: `query-/tests/test_catalog.py`
- Create: `query-/tests/test_cli.py`
- Modify: `query-/research_query.py`

**Interfaces:**
- `CatalogError`
- `load_catalog(path: Path) -> dict`
- `lookup(data: dict, key: str) -> dict`
- `verify(data: dict, workspace: Path) -> dict`
- `main(argv: list[str] | None = None) -> int`

- [ ] Add failing package tests by porting current behavior tests without changing expected semantics.
- [ ] Observe RED because package is absent.
- [ ] Implement package modules by moving current logic with minimal semantic change.
- [ ] Turn root `research_query.py` into a thin wrapper that prepends local `src` only when running from an uninstalled checkout and imports `universal_law_query.cli.main`.
- [ ] Run old tests plus new package tests; obtain GREEN.
- [ ] Add a parity test comparing old wrapper vs `python -m universal_law_query.cli` for list, lookup, verify and refusal paths.
- [ ] Commit.

### Task 2: Catalog-entry helper extraction

**Files:**
- Create: `query-/src/universal_law_query/catalog_entry.py`
- Create: `query-/tests/test_catalog_entry.py`
- Modify: `query-/catalog_entry_helper.py`

**Interfaces:**
- `HelperError`
- `valid_repo_path(text: str)`
- `read_public_bytes(path: Path) -> bytes`
- `resolve_commit(file_path: Path, explicit: str | None) -> str`
- `build_entry(...) -> dict`
- helper CLI `main(argv=None) -> int`

- [ ] Port current tests into package-targeted failing tests.
- [ ] Observe RED.
- [ ] Move implementation into package.
- [ ] Replace root helper with thin wrapper.
- [ ] Verify sandbox/symlink/traversal/mutable-commit refusals and wrapper parity.
- [ ] Commit.

### Task 3: Portable-stub verifier extraction

**Files:**
- Create: `query-/src/universal_law_query/stub_verify.py`
- Create: `query-/tests/test_stub_verify.py`
- Modify: `query-/verify_portable_stubs.py`

**Interfaces:**
- Preserve `load_candidates`, `validate_row`, `fetch_raw`, `check_math_tip_drift`, and `main(argv=None)`.

- [ ] Add failing package tests.
- [ ] Observe RED.
- [ ] Move code with no semantic broadening.
- [ ] Replace root script with thin wrapper.
- [ ] Run old/new tests and parity cases.
- [ ] Commit.

### Task 4: Clean-install and source-layout compatibility

**Files:**
- Modify: `query-/pyproject.toml`
- Create: `query-/tests/test_install_contract.py`
- Modify: `query-/README.md`
- Modify: `query-/AGENTS.md`

- [ ] Add failing test that installs the local package into a temporary virtual environment without dependencies and runs the console command.
- [ ] Observe RED before console entry point exists.
- [ ] Add `project.scripts.universal-law-query = "universal_law_query.cli:main"`.
- [ ] Verify clean install and CLI output.
- [ ] Document canonical source package vs compatibility wrappers.
- [ ] Update AGENTS with canonical paths and verification command.
- [ ] Commit.

### Task 5: Deterministic source manifest and archive builder

**Files:**
- Create: `query-/scripts/build_source_release.py`
- Create: `query-/tests/test_source_release.py`
- Create: `query-/SOURCE_MANIFEST.json` only if the spec's generated-view check concludes checked-in latest manifest is useful; otherwise keep manifest in release output only.

**Interfaces:**
- `build_source_archive(repo_root: Path, output: Path, source_date_epoch: int) -> dict`

- [ ] Add failing tests for sorted paths, fixed mtimes, no `.git`/cache/private path, embedded manifest, and identical archive SHA across two builds.
- [ ] Observe RED.
- [ ] Implement deterministic tar.gz or tar archive builder with normalized metadata.
- [ ] Run tests twice and assert byte-identical output.
- [ ] Do not create a GitHub Release yet; this is a dry-run artifact.
- [ ] Commit.

### Task 6: Trial federation install/parity test

**Files:**
- Modify: `trial/federation/test_federation.py`
- Create: `trial/federation/test_query_package.py` if separation keeps the file focused.

- [ ] Add failing test that installs query from sibling checkout and performs lookup/verification on synthetic catalog.
- [ ] Observe RED against pre-migration query main.
- [ ] Update fixture/import logic to prefer installed package but retain old-wrapper parity checks.
- [ ] Run full federation suite.
- [ ] Commit trial integration update.

### Task 7: Hosted CI, metrics and independent review

- [ ] Run query local legacy + package suites.
- [ ] Run clean-install test.
- [ ] Run trial federation suite.
- [ ] Inspect exact-head hosted CI logs.
- [ ] Record before/after: test runtime, files read for contribution, duplicate implementation count, wrapper count.
- [ ] Request fresh engineering review on query and trial exact heads.
- [ ] Fix Critical/Important findings with RED→GREEN.
- [ ] Mark query source package ready for first GitHub source-release plan only after review.

### Task 8: Release dry-run gate

- [ ] Produce deterministic archive twice from the same exact commit.
- [ ] Verify identical SHA256.
- [ ] Verify embedded source manifest.
- [ ] Verify no sandbox/private paths.
- [ ] Publish **no** external package registry artifact.
- [ ] Record exact commit/archive identity for later GitHub Release publication.
