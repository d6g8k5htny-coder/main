# Pytest bytecode-policy population Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Initial staging was source only. Root subsequently reported its fresh source review PASS plus a separate pytest_policy_review PASS and authorized isolated draft publication and mandatory hosted qualification; merging remains unauthorized.

**Goal:** Exercise the existing ten pytest-policy unittest bodies in normal and optimized Python modes with genuine installed pytest, retaining original evidence.

**Architecture:** A separate read-only GitHub Actions workflow prepares a fresh CPython 3.11 venv outside the repository, verifies five fixed universal wheels against acquired PyPI metadata and a dedicated hashlock, and exposes that venv's real purelib only to the parent -S interpreter. The unchanged tests launch sys.executable -I children, which use the same actual venv's site-packages and preserve intentional bytecode-adverse behavior.

**Tech Stack:** CPython 3.11, venv, pytest9.1.1, iniconfig2.3.1, packaging26.3, pluggy1.7.0, pygments2.21.0; pinned checkout/setup/upload actions already used by Main.

**Spec:** Design and constraints below, grounded in Main812a799f4cf4cee730efa0177a9101af4cf259ef/treeaf64028bde20dc69d82857bcb558aaff76a94912.

## Design and approaches

The selected existing source is tests/test_pytest_bytecode_policy.py, blob42252c0a1871189948e0d240151619b04f53e001. Its ten bodies currently skip when pytest is unavailable. No current workflow explicitly invokes this module. Ordinary navigation uses -S and must retain that flag. Existing tools/hypergraph/requirements.txt (blob2f5732e68f66aca0c0286e2f0ef2f936dcdd00a2) supplies these exact five pytest-related versions but no hashes. PyPI9.1.1 metadata permits Python>=3.10 and declares only the four selected unconditional Linux/Python3.11 dependencies; exceptiongroup/tomli apply below3.11 and colorama to Windows.

Chosen: separate five-wheel hashlocked venv and fixed module invocation. Installing all hypergraph requirements would add unrelated packages; changing ordinary-suite -S would expand scope and still obscure child interpreter identity. A stub pytest or PYTHONPATH-only base interpreter would fail the child's -I dependency requirement. These implementation decisions are the staging author's calls under the parent's autonomous scope.

## Global constraints

- Initial staging made no refs, PRs or executions. The authorized follow-up publishes an isolated Draft PR and observes its native hosted qualification, with no merge, settings changes, comments or reruns.
- Preserve the existing test, conftest.py blob2163cb9213ee8445935d4342530012357cb21b36 and Reactor validator blobfb4d8419b638b5619f8d8c6f685a036b5a6b8a61; no real Reactor evaluator, numerical fixture or proof execution.
- Parent invocation retains -B -S and -B -O -S; children retain the test's -I behavior, including intentional bytecode creation in its adverse control. No cache deletion, exclusion weakening, scipy or gudhi installation.
- Require native commit/tree and selected source/runtime identities unchanged before/after; exact ten body identities, zero skipped/failed/error/expected-failure/unexpected-success results in both modes; original logs, exits, child receipts, raw wheels and dependency metadata retained.
- contents:read, persist-credentials:false, no secret inputs; immutable upload-artifact v4 with always() and unique run/attempt artifact name.
- This is only one G5 prerequisite population. No fullG5, scientific, theorem, CBZ or terminal classification credit.

## Review Focus

- On Python3.11, -S changes venv prefix handling: use the venv executable and explicitly derived purelib, then verify actual pytest origin under that purelib; do not substitute the base executable.
- The child -I ignores PYTHONPATH: verify installed pytest and dependencies in the real venv before either mode.
- The adverse dirty-bytecode control must create actual cache evidence: preserve its sanitized child environment and original assertions.
- A count-only log can conceal another population: require the exact ten qualified body identities and literal ok outcomes in each original log, with Ran10/OK footers.
- A setup/normal failure must retain all existing evidence and still attempt the optimized cell after successful preparation; require both exits and current before/after identities before qualification.

## Plan

- [x] Stage tests/pytest-bytecode-requirements.txt with five exact versions and observed universal-wheel hashes; stage tests/pytest-bytecode-dependencies.json with exact primary JSON/wheel URLs, filenames, Requires-Python, core-metadata digests and honest acquisition provenance. Validate version correspondence to the existing committed requirements.
- [x] Stage .github/workflows/pytest-bytecode-policy.yml: bind native selected sources, prepare and verify actual dependencies, record runtime, run only the fixed module normal/optimized with separate original evidence directories, verify both strict result populations and unchanged sources/runtime, upload originals on every conclusion.
- [x] Fresh source review: Root reports its own source review PASS and a separate pytest_policy_review PASS for the staged workflow, hashlock and provenance. This is source review, not execution. No code was executed in the failed local environment; passing hosted qualification remains pending.
- [ ] Root has authorized isolated draft publication after source review. Publish on a fresh compatible Main composition and inspect actual native hosted20-body outcomes and original receipts. Hosted execution is not complete until the actual original results are observed; preserve failures and use no skips or retries as coverage.

## Exact ten-method population

- `tests.test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_changed_byte_is_not_hidden`
- `tests.test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_clean_root_run_prevents_package_bytecode`
- `tests.test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_dirty_bytecode_is_preserved_and_rejected`
- `tests.test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_discovery_keeps_neighbours_and_same_basename`
- `tests.test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_empty_cache_directory_is_accepted`
- `tests.test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_exclusion_is_exact`
- `tests.test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_explicit_B_baseline_works`
- `tests.test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_extra_file_is_not_hidden`
- `tests.test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_missing_file_is_not_hidden`
- `tests.test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_neighbour_failures_are_not_suppressed`

Dependency acquisition details are recorded in staged tests/pytest-bytecode-dependencies.json, including exact release JSON and wheel URLs, core-metadata SHA256 fields, Requires-Python and honest web-tool provenance. The candidate additionally checks actual -S parent optimize0/1 probes and all ten retained -I pytest child receipts per mode, preserving their intended adverse exit codes and bytecode observations. Pytest distribution console scripts are allowed under the actual venv root; module origin must still be inside its real purelib. Runtime-generated pyc is retained, while installed source/metadata fingerprints exclude pyc so intended cache creation does not masquerade as source drift.

## Authorized publication composition

The fresh base is Main3f05f8493afac71c6f4aac5c1db5aeb46194d98a/treeabbe9dad2d2302bd478b6d9660732c9d51c450fe, after the Q-9002 integration. The existing ten-method test, conftest, Reactor validator and hypergraph dependency-version source retain their exact original blobs. Entire current DECISIONS/GOVERNANCE/TOOLING contents are preserved before additive registration. The reviewed workflow/hashlock/provenance bytes remain unchanged. Root authorized this execution method and publication after source review; no artificial new-test baseline is required for these already existing ten dormant methods. Actual hosted results, including failed outcomes if any, must be recorded separately before any integration decision.

## Retained author-family correction 1

The initial published workflow63deafff/run42750759039 failed GitHub validation before any job: native annotation lines30/31 rejected runner.temp in job-level env. This is a zero-job configuration failure, not a test-body RED or execution result. The corrected workflow moves the two paths to prepare-step env, persists them through GITHUB_ENV at its start, and uses a fixed step-context runner.temp artifact path even when preparation cannot run. Tests, dependency hashes, method identities, commands and qualification guards are unchanged. This source correction remains subject to actual native hosted execution; the failed run is preserved in ancestry and the decision ledger.
